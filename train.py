import os
import sys
import heapq
import torch
import copy
import socket
import time
import psutil
import shutil
import pickle
import functools
import datetime
import deepspeed

import multiprocessing
import numpy as np
import torch.distributed
import torch.multiprocessing as mp

from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
from transformers import get_scheduler, LlamaModel, Qwen2Model, PreTrainedTokenizer
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from peft import get_peft_model, LoraConfig

from arguments import parse_args
from ref_server import init_socket, obtain_ref_loss
from data.build_data import build_pretrain_data, get_cache_dir, build_tokenized_datasets, get_tokenizer
from model.utils import *
try:
    from grad_filter import token_filter
except ImportError:
    print("Cannot import grad_filter, skipping the module")
from gen_node_tracing import gen_node_tracing_with_model, is_node_tracing_generated, gen_node_tracing_with_model_with_dims

from deepspeed.runtime.lr_schedules import WarmupCosineLR
from torch.utils.data.distributed import DistributedSampler

eff_bench_iters = 12  # 2 iterations for warming up


def initialize():
    args = parse_args(include_deep_speed_args=True)
    torch.random.manual_seed(args.random_seed)
    deepspeed.init_distributed(
        dist_backend="nccl",
        timeout=datetime.timedelta(minutes=60*24)
    )
    get_cache_dir(args)
    sys.argv += ["--gradient_accumulation_steps", str(args.batch_size // args.micro_batch_size // torch.distributed.get_world_size())]


def get_param_groups_by_layer(model, base_lr, decay_factor):
    increase_ratio = 1 / decay_factor
    total_layers = model.config.num_hidden_layers
    param_group = []
    param_by_index = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lm_head" in name:
            param_group.append({"params": [param], "lr": base_lr})
        elif "embed_tokens" in name:
            param_group.append({"params": [param], "lr": base_lr * increase_ratio})
        else:
            try:
                layer_index = int(name.split(".")[2]) + 1
                param_by_index[layer_index] = param_by_index.get(layer_index, []) + [param]
            except:
                print(name, 'use base lr')
                param_group.append({"params": [param]})
    for layer_index, params in param_by_index.items():
        layer_increase_ratio = base_lr + (1 - layer_index/total_layers) * (increase_ratio - 1) * base_lr
        param_group.append({"params": params, "lr": layer_increase_ratio})
    return param_group


class Trainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset):
        self.args = parse_args(include_deep_speed_args=True)
        self.tokenizer = tokenizer
        self.device = 'cuda' if not torch.distributed.is_initialized() else f'cuda:{torch.distributed.get_rank() % torch.cuda.device_count()}'
        self.model = model.to(self.device)
        self.model_config = AutoConfig.from_pretrained(self.args.model_path)

        if self.args.use_lora:
            lora_config = LoraConfig(
                r=64, 
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # "gate_proj", "down_proj", "up_proj"
                lora_dropout=0.1, 
                lora_alpha=128
            )
            self.model = get_peft_model(self.model, lora_config).to(self.device)

        if self.args.attn_filter:
            self.node_filter_dims = ""
            self._gen_node_tracing_bsz = 1 if self.args.micro_batch_size == 1 else 19
        
        if self.args.attn_filter and not self.args.use_lora: # get the filtering dims before deepspeed.initialize
            self.node_filter_dims = gen_node_tracing_with_model_with_dims(
                self.model, self._gen_node_tracing_bsz, 32*7, self.model_config.vocab_size, self.node_filter_dims)
        
        self.num_training_steps = len(train_dataset) // self.args.micro_batch_size \
            // torch.distributed.get_world_size() * self.args.epochs
        print(f"Num training steps: {self.num_training_steps}")

        ds_config = {
            "bf16": {
                "enabled": True
            },
            "train_batch_size": self.args.batch_size,
            "train_micro_batch_size_per_gpu": self.args.micro_batch_size,
            "zero_optimization": {
                "stage": 1,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "cpu_offload": False,
                "zero_quantized_nontrainable_weights": True
                # "offload_optimizer": {
                #     "device": "cpu",
                # }
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay
                }
            },
            "scheduler": {
                "type": "WarmupCosineLR",
                "params": {
                    "cos_min_ratio": 0.1,
                    "warmup_min_ratio": 0,
                    "warmup_num_steps": len(train_dataset) * self.args.epochs // self.args.batch_size // 20, # 10% of the training steps
                    "total_num_steps": len(train_dataset) * self.args.epochs // self.args.batch_size,
                    "warmup_type": "linear"
                }
            },
            "gradient_clipping": 1.0,
            "tensorboard": {
                "enabled": True,
                "output_path": self.args.model_cache_dir,
                "job_name": "tensorboard"
            },
            "csv_monitor": {
                "enabled": True,
                "output_path": self.args.model_cache_dir,
                "job_name": "csv_monitor"
            }
        }
        
        # datasets are already shuffled
        # self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, collate_fn=collate_fn)
        self.eval_dataloader = DataLoader(
            eval_dataset, batch_size=8, collate_fn=collate_fn)

        # Initialize DeepSpeed
        if 0 < self.args.layer_lr_decay < 1.0:
            param_groups = get_param_groups_by_layer(self.model, self.args.learning_rate, self.args.layer_lr_decay)
        else:
            param_groups = [param for param in self.model.parameters() if param.requires_grad if param.requires_grad]
        
        self.model, self.optimizer, self.train_dataloader, _ = deepspeed.initialize(
            model=self.model, 
            config=ds_config,
            model_parameters=param_groups, 
            training_data=train_dataset, 
            collate_fn=collate_fn
        )

        if self.args.attn_filter and self.args.use_lora:
            node_dims_cache_file = os.path.join("tmp/node_filter_dims_lora.txt")
            if os.path.exists(node_dims_cache_file):
                with open(node_dims_cache_file, "r") as f:
                    self.node_filter_dims = f.read()
            else:
                self.node_filter_dims = gen_node_tracing_with_model_with_dims(
                    self.model, self._gen_node_tracing_bsz, 32*7, self.model_config.vocab_size, self.node_filter_dims)
                with open(node_dims_cache_file, "w") as f:
                    f.write(self.node_filter_dims)

        self.step = 0
        self.completed_steps = 0
        self.accumulated_loss = [0, 0]
        
        self.train_state_dir = os.path.join(self.args.model_cache_dir, "training_states")
        if not os.path.exists(self.train_state_dir):
            os.makedirs(self.train_state_dir, exist_ok=True)
        if os.path.exists(self.args.model_cache_dir):
            self.restore_training()

        self.avg_seq_len = []

        if torch.distributed.is_initialized():
            time_log_file = f"batch_time_{torch.distributed.get_rank()}.json"
        else:
            time_log_file = "batch_time.json"
        self.batch_timer = BatchTimer(
            self.args.gradient_accumulation_steps, 
            os.path.join(self.args.model_cache_dir, time_log_file))
        
        # cache self.args to model_cache_dir
        with open(os.path.join(self.args.model_cache_dir, "args.json"), "w") as f:
            json.dump(self.args.__dict__, f)

        self.filtering_ratios = []

    def state_dict(self):
        return {"step": self.step, "completed_steps": self.completed_steps}
    
    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.completed_steps = state_dict["completed_steps"]
    
    def train_step(self, batch, ref_loss=None):
        self.step += 1
        if self.step % self.args.gradient_accumulation_steps == 1:
            self.batch_timer.start()
        
        batch['input_ids'] = batch['input_ids'].to(self.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.device)
        
        self.avg_seq_len.append(sum(batch["length"]) / len(batch["length"]))

        # Forward Pass (micro-batch)
        self.batch_timer.record_nvtx("forward")
        logits = self.model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        torch.cuda.synchronize()
        self.batch_timer.record("forward")

        # Token Filter Loss
        self.batch_timer.record_nvtx("loss_mask")
        loss, ref_mask = token_filter_loss(
            batch["input_ids"], logits, 
            attention_mask=batch["attention_mask"], 
            ref_loss=ref_loss, 
            dropping_strategy=self.args.dropping_strategy, 
            drop_rate=self.args.drop_rate,
            is_left_padding=True, return_mask=True,
            batch_timer=self.batch_timer
        )
        if ref_loss is not None:
            all_token_loss = token_filter_loss(
                batch["input_ids"], logits, 
                # attention_mask=batch["attention_mask"], (attn_mask is changed in last token_filter_loss func)
                ref_loss=None, is_left_padding=True, return_mask=False, batch_timer=None
            )
        torch.cuda.synchronize()
        self.batch_timer.record("loss_mask")

        if self.args.dropping_strategy is not None:
            self.filtering_ratios.append(1 - float(ref_mask.sum())/ref_mask.numel())

        if ref_loss is not None and self.args.attn_filter:
            try:
                self.batch_timer.record_nvtx("filter")
                token_filter.ops.backward_filter_with_dims(loss, ref_mask, self.node_filter_dims)
                torch.cuda.synchronize()
                self.batch_timer.record("filter")
            except Exception as e:
                print("Error in backward filter", e, "skipping the current step")
                return
        
        self.batch_timer.record_nvtx("backward")
        self.accumulated_loss[0] += loss.item()
        if ref_loss is not None:
            self.accumulated_loss[1] += all_token_loss.item()
        self.model.backward(loss)
        torch.cuda.synchronize()
        self.batch_timer.record("backward")

        self.batch_timer.record_nvtx("step")
        self.model.step()
        torch.cuda.synchronize()
        self.batch_timer.record("step")
        
        if self.step % self.args.gradient_accumulation_steps == 0:
            self.completed_steps += 1
            # Log loss
            self.accumulated_loss[0] /= self.args.gradient_accumulation_steps
            self.accumulated_loss[1] /= self.args.gradient_accumulation_steps
            if torch.distributed.get_rank() == 0:
                print(
                    {
                        "samples": self.step * self.args.batch_size,
                        "steps": self.completed_steps,
                        "loss/train": self.accumulated_loss[0],
                        "loss/train_all_token": self.accumulated_loss[1],
                        "learning rates": self.model.get_lr(),
                        "avg_seq_len": np.mean(self.avg_seq_len),
                    }
                )
                if self.args.dropping_strategy is not None:
                    print(f"Filtering ratio: {np.mean(self.filtering_ratios)}")
            self.batch_timer.end()
            batch_timer_last_batch = self.batch_timer.last_batch()
            events = []
            for key, value in batch_timer_last_batch.items():
                events.append((f"train/timer/{key}", value, self.model.global_samples))
            events.append(("train/loss/train", self.accumulated_loss[0], self.model.global_samples))
            events.append(("train/loss/train_all_token", self.accumulated_loss[1], self.model.global_samples))
            self.model.monitor.write_events(events)
            self.accumulated_loss = [0, 0]

        if (self.step % (self.args.eval_steps * self.args.gradient_accumulation_steps)) == 0 and self.args.task == "train":
            if torch.distributed.get_rank() == 0:
                eval_loss, perplexity = evaluate_single_gpu(self.model, self.eval_dataloader, device=self.model.device)
                print({"loss/eval": eval_loss, "perplexity": perplexity})
                events = [
                    ("eval/loss", eval_loss, self.model.global_samples), 
                    ("eval/perplexity", perplexity, self.model.global_samples)
                ]
                self.model.monitor.write_events(events)
                self.model.train()
            torch.distributed.barrier()
        if (self.step % (self.args.save_steps * self.args.gradient_accumulation_steps)) == 0:
            if not self.args.task == "eff-benchmark":
                self.save_model()
                self.save_training_states()
    
    def save_model(self):
        if torch.distributed.get_rank() == 0:
            self.model.save_pretrained(self.args.model_cache_dir)
            self.tokenizer.save_pretrained(self.args.model_cache_dir)
    
    def save_training_states(self):
        self.model.save_checkpoint(
            self.train_state_dir, tag="train", client_state=self.state_dict(),
            save_latest=True
        )

    def restore_training(self):
        try:
            _, client_state = self.model.load_checkpoint(self.train_state_dir)
            self.load_state_dict(client_state)
            print("Continuing training from step", self.completed_steps)
        except Exception as e:
            print(f"Cannot restore training in {self.args.model_cache_dir}, {e}")
    
    def end_training(self):
        if not self.args.task == "eff-benchmark":
            self.save_model()
        torch.distributed.barrier()
        # clear training states
        if torch.distributed.get_rank() == 0:
            shutil.rmtree(self.train_state_dir)
    
    def train(self):
        print(f"Training from step {self.step}")
        bar = tqdm(total=self.num_training_steps)
        for epoch in range(self.args.epochs):
            for step, batch in enumerate(self.train_dataloader, start=1+epoch*(self.num_training_steps//self.args.epochs)):
                if step <= self.step:
                    bar.update(1)
                    torch.distributed.barrier()
                    continue
                if self.args.task == "eff-benchmark" and self.completed_steps >= eff_bench_iters:
                    print("Efficiency benchmark is completed.")
                    break
                if self.args.task == "param-search" and self.completed_steps >= 1000:
                    print("Param search is completed.")
                    break
                if self.args.ref_model_backend == "none":
                    self.train_step(batch, ref_loss=None)
                elif self.args.ref_model_backend == "self":
                    self.train_step(batch, ref_loss=[[0.0]*(len(e)-1) for e in batch["input_ids"]])
                else:
                    self.train_step(batch, ref_loss=batch.get("ref_loss"))
                bar.update(1)
        self.end_training()


def main():
    initialize()
    args = parse_args(include_deep_speed_args=True)

    print("args", args)

    tokenizer = get_tokenizer(args.model_path)

    if args.task == "eff-benchmark":
        print("Running efficiency benchmark")
        if os.path.exists(args.model_cache_dir):
            batch_time_logs = [e for e in os.listdir(args.model_cache_dir) if (e.endswith(".json") and e.startswith("batch_time"))]
            if len(batch_time_logs) > 0:
                print(f"Found eff-benchmark logs {batch_time_logs}, skipping the current task")
                return 0
        config = AutoConfig.from_pretrained(args.model_path)
        vocub_size = config.vocab_size
        benchmark_steps = eff_bench_iters
        num_fake_samples = benchmark_steps * args.batch_size
        # if torch.distributed.is_initialized():
        #     num_fake_samples *= torch.distributed.get_world_size()
        fake_input_ids = torch.randint(0, vocub_size, (num_fake_samples, args.context_length), dtype=torch.long).tolist()
        fake_attn_mask = torch.ones((num_fake_samples, args.context_length), dtype=torch.long).tolist()
        fake_ref_loss = [[0.0]*(args.context_length-1) for _ in range(num_fake_samples)]
        fake_batch = {
            "input_ids": fake_input_ids, 
            "attention_mask": fake_attn_mask, 
            "ref_loss": fake_ref_loss, 
            "length": [args.context_length]*num_fake_samples
        }
        fake_train_dataset = Dataset.from_dict(fake_batch)
        fake_valid_dataset = Dataset.from_dict(fake_batch)
        tokenized_dataset = DatasetDict({"train": fake_train_dataset, "valid": fake_valid_dataset})
        print(tokenized_dataset)
        # tokenized_dataset = load_from_disk(args.dataset_path)
    else:
        tokenized_dataset = build_tokenized_datasets(args)

    print(f"Num training tokens: {sum(tokenized_dataset['train']['length'])/(1000**2):.1f}M tokens")

    if args.task == "gen-data":
        print("Data generation task is completed.")
        return 0
    
    config = AutoConfig.from_pretrained(args.model_path)
    if args.attn_impl == "eager":
        config._attn_implementation = args.attn_impl
    print("config", config)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config).to(torch.bfloat16)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"SFT Parameter size: {model_size/1000**2:.1f}M parameters")
    
    trainer = Trainer(model, tokenizer, tokenized_dataset["train"], tokenized_dataset["valid"])
    
    if args.ref_model_backend in ["none", "self"] or args.pre_compute_ref:
        trainer.train()
    elif args.ref_model_backend == "socket":
        recv_queue, send_queue = init_socket(args)  # todo: need to be updated
        bar = tqdm(total=trainer.num_training_steps // args.gradient_accumulation_steps)
        data_queue = []
        buffer_size = 4 * args.gradient_accumulation_steps
        for epoch in range(args.epochs):
            for step, batch in enumerate(
                trainer.train_dataloader, start=1+epoch*(trainer.num_training_steps//args.epochs)
            ):
                if step <= trainer.step:
                    if step % args.gradient_accumulation_steps == 0:
                        bar.update(1)
                    continue
                if args.task == "eff-benchmark" and trainer.completed_steps >= eff_bench_iters:
                    break
                # enter queue and send request
                data_queue.append(batch)
                batch_input_ids = batch["input_ids"].tolist()
                send_queue.put([batch_input_ids[e][-batch['length'][e]:] for e in range(len(batch_input_ids))])
                # train micro-batch
                if step % args.gradient_accumulation_steps == 0 and step >= buffer_size:
                    if step == buffer_size:
                        time.sleep(10)
                    for mic_i in range(args.gradient_accumulation_steps):
                        trainer.train_step(data_queue.pop(0), ref_loss=recv_queue.get()['loss'])
                    bar.update(1)
        trainer.end_training()
    else:
        raise NotImplementedError(f"Backend {args.ref_model_backend} is not implemented for training.")

if __name__ == "__main__":
    main()
