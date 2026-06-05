#!/usr/bin/env python3
"""
MegatronLM TinyLlama TP computation graph generator.
Builds a MegatronLM tensor-parallel graph from a HuggingFace Llama/TinyLlama config.
"""

import os
import sys
import time
import datetime
import argparse
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import torchviz
import math
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

import sys
sys.path.append("/data/Megatron-LM")

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core import parallel_state
from torch.utils.data.dataloader import DataLoader

from grad_filter import token_filter
from model.utils import token_filter_loss, BatchTimer
from transformers import AutoConfig
from model.megatronlm import MegatronLlamaModel, create_model_parallel_config
from model.utils import collate_fn
from gen_node_tracing import gen_node_tracing_with_model_with_dims_megatronlm
from data.build_data import create_synthetic_data


def setup_distributed_environment(tp_size=2):
    print(f"🔧 Setting up distributed environment (TP={tp_size})...")
    
    # Distributed rank / world size from torchrun env
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', tp_size))
    
    print(f"  - Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
    
    # Bind this process to its local GPU
    assert torch.cuda.is_available()
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
    print(f"  - Device: {device}")
    dist.init_process_group(
        backend='nccl', timeout=datetime.timedelta(minutes=60*24),
        rank=rank, world_size=world_size
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
    )
    print("Backend in use:", torch.distributed.get_backend())
    model_parallel_cuda_manual_seed(1234)
    return device


def generate_tinyllama_tp_graph(model, input_ids, tp_size, output_dir="tmp"):
    """Generate TinyLlama TP computation graph (debug / node tracing)."""
    rank = int(os.environ.get('RANK', 0))
    
    # Forward pass
    model.train()

    # Get filtering dims
    node_tracing_file = 'tmp/megatronlm_node_tracing.txt'
    if os.path.exists(node_tracing_file):
        with open(node_tracing_file, 'r') as f:
            filter_dims = f.read()
    else:
        filter_dims = ""
        filter_dims = gen_node_tracing_with_model_with_dims_megatronlm(
            model, 1, 224, model.llama_config.vocab_size, filter_dims)
        with open(node_tracing_file, 'w') as f:
            f.write(filter_dims)
    
    logits = model(input_ids)
    
    if rank == 0:
        print(f"  - Output shape: {logits.shape}")
    
    # Loss for a full autograd graph
    labels = torch.randint(0, model.llama_config.vocab_size, input_ids.shape, device=input_ids.device)
    loss_fct = nn.CrossEntropyLoss()
    
    # Causal LM label shift (unused when using token_filter_loss)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    loss, ref_mask = token_filter_loss(
        inputs=input_ids, logits=logits, attention_mask=torch.ones_like(input_ids),
        ref_loss=torch.randn(input_ids[:, 1:].size()).tolist(), 
        dropping_strategy='fixed', drop_rate=0.5, # the drop_rate does not impact generating the node tracing files
        is_left_padding=True, return_mask=True
    )

    # Get node names
    print(token_filter.ops.collect_node_names(loss))

    # print(filter_dims)
    
    token_filter.ops.backward_filter_with_dims(loss, ref_mask, filter_dims)
    
    if rank == 0:
        print(f"  - Loss: {loss.item():.6f}")
    
    # Barrier across ranks
    if dist.is_initialized():
        dist.barrier()
    
    try:
        loss.backward()
        # Build torchviz graph
        dot = torchviz.make_dot(
            loss, 
            params=dict(model.named_parameters()),
            show_saved=True,
            show_attrs=True
        )
        
        # Graph layout attributes
        dot.graph_attr.update(
            fontname='Arial',
            fontsize='14',
            ranksep='0.3',
            nodesep='0.2',
            pack='true',
            packmode='clust'
        )
        dot.node_attr.update(fontname='Arial', fontsize='12')
        dot.edge_attr.update(fontname='Arial', fontsize='10')
        
        # Output path
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/megatron_tinyllama_tp{tp_size}_computation_graph_{rank}"
        
        # Render SVG and PDF
        dot.render(filename, format='svg', cleanup=True)
        dot.render(filename, format='pdf', cleanup=True)
        
        print(f"  ✅ Graph saved:")
        print(f"    - SVG: {filename}.svg")  
        print(f"    - PDF: {filename}.pdf")
        
        return filename
        
    except Exception as e:
        print(f"  ❌ Graph generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="MegatronLM TinyLlama TP Computation Graph Generator")
    
    parser.add_argument("--mode", type=str, default="gen-node-tracing", choices=["gen-node-tracing", "eff-benchmark"],
                       help="Mode")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama_v1.1")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, 
                       help="Tensor parallel size")
    parser.add_argument("--num_layers", type=int, default=None,
                       help="Number of layers (default: use TinyLlama's 22 layers)")
    parser.add_argument("--batch_size", type=int, default=512,
                       help="Batch size")
    parser.add_argument("--mic_batch_size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--sequence_length", type=int, default=2048,
                       help="Sequence length")
    parser.add_argument("--output_dir", type=str, default="tmp",
                       help="Output directory")
    parser.add_argument("--attn_filter", action="store_true", default=False,
                       help="Whether to use attention filter")
    
    args = parser.parse_args()

    """
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=12356 megatron_tp_benchmark.py --tensor_parallel_size 2 --mode gen-node-tracing --model meta-llama/Llama-3.1-8B --batch_size 1 --sequence_length 224

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=12356 megatron_tp_benchmark.py --tensor_parallel_size 4 --mode eff-benchmark
    
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=12356 megatron_tp_benchmark.py --tensor_parallel_size 8 --mode eff-benchmark --model meta-llama/Llama-3.1-8B
    """
    
    # Distributed setup
    device = setup_distributed_environment(args.tensor_parallel_size)
    
    # HuggingFace model config
    llama_config = AutoConfig.from_pretrained(args.model)

    llama_config.vocab_size = int(llama_config.vocab_size / args.tensor_parallel_size)
    
    # if torch.distributed.get_rank() == 0:
    #     import pdb; pdb.set_trace()
    # torch.distributed.barrier()

    print(f"\n📋 Model Config:")
    print(f"  - Vocab size: {llama_config.vocab_size:,}")
    print(f"  - Hidden size: {llama_config.hidden_size}")
    print(f"  - Intermediate size: {llama_config.intermediate_size}")
    print(f"  - Attention heads: {llama_config.num_attention_heads}")
    print(f"  - KV heads: {getattr(llama_config, 'num_key_value_heads', llama_config.num_attention_heads)}")
    print(f"  - Layers: {llama_config.num_hidden_layers}")
    
    # Megatron ModelParallelConfig
    megatron_config = create_model_parallel_config(args.tensor_parallel_size)
    
    # Build TP model
    print(f"\n🔧 Creating TP model...")
    model = MegatronLlamaModel(llama_config, megatron_config, args.num_layers)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params:,}")

    model.train()
    filter_dims = ""
    if args.attn_filter:
        filter_dims = gen_node_tracing_with_model_with_dims_megatronlm(
                model, 1, 224, model.llama_config.vocab_size, filter_dims)
        with open(os.path.join('tmp', "filter_dims_tp.txt"), "w") as f:
            f.write(filter_dims)

    if args.mode == "gen-node-tracing":
        # Synthetic input batch
        print(f"\n📝 Creating input...")
        input_ids = torch.randint(
            0, llama_config.vocab_size, 
            (args.batch_size, args.sequence_length), 
            device=device
        )
        print(f"  - Input shape: {input_ids.shape}")
        
        # Generate computation graph
        filename = generate_tinyllama_tp_graph(
            model, input_ids, args.tensor_parallel_size, args.output_dir
        )
    
    if args.mode == "eff-benchmark":
        eff_benchmark_iterations = 2
        batch_timer = BatchTimer(args.batch_size, os.path.join(
            args.output_dir, 
            f"batch_timer_{args.model.replace('/', '_')}_bsz{args.batch_size}_seq{args.sequence_length}_tp{args.tensor_parallel_size}_attnfilter{args.attn_filter}_rank{torch.distributed.get_rank()}.json"), 
            warmup=1
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        synthetic_data = create_synthetic_data(
            llama_config.vocab_size, 
            context_length=args.sequence_length, 
            num_samples=args.batch_size * eff_benchmark_iterations
        )
        train_dataloader = DataLoader(synthetic_data, batch_size=args.mic_batch_size, collate_fn=collate_fn)
        
        bar = tqdm(total=len(train_dataloader))
        start = time.time()
        # print("Debug 1")
        for step, batch in enumerate(train_dataloader, start=1):
            # print("Debug 2")
            if step % args.batch_size == 1:
                batch_timer.start()
            
            # print("Debug 3")
            batch["input_ids"] = batch["input_ids"].to(device)
            logits = model(batch["input_ids"].to(device))

            # print("Debug 4")
            loss, ref_mask = token_filter_loss(
                inputs=batch["input_ids"], logits=logits, attention_mask=torch.ones_like(batch["input_ids"]),
                ref_loss=torch.randn(batch["input_ids"][:, 1:].size()).tolist(), 
                dropping_strategy='fixed', drop_rate=0.5,
                is_left_padding=True, return_mask=True
            )
            torch.cuda.synchronize()
            batch_timer.record("forward")

            if args.attn_filter:
                token_filter.ops.backward_filter_with_dims(loss, ref_mask, filter_dims)
                torch.cuda.synchronize()
                batch_timer.record("filter")

            loss.backward()
            torch.cuda.synchronize()
            batch_timer.record("backward")
            
            if step % args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                end = time.time()
                print(f"Time taken of one iteration: {end - start} seconds")
                start = time.time()
                batch_timer.end()
            bar.update(1)
    
    return 0

if __name__ == "__main__":
    exit(main())