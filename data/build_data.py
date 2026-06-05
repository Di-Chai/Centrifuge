import os
import sys
import pdb
import copy
import torch
import hashlib
import psutil
import argparse

from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from model.utils import obtain_ref_loss, init_socket, init_multiple_sockets


def create_synthetic_data(vocub_size, context_length, num_samples):
    fake_input_ids = torch.randint(0, vocub_size, (num_samples, context_length), dtype=torch.long).tolist()
    fake_attn_mask = torch.ones((num_samples, context_length), dtype=torch.long).tolist()
    fake_ref_loss = [[0.0]*(context_length-1) for _ in range(num_samples)]
    fake_batch = {
        "input_ids": fake_input_ids, 
        "attention_mask": fake_attn_mask, 
        "ref_loss": fake_ref_loss, 
        "length": [context_length]*num_samples
    }
    return Dataset.from_dict(fake_batch)


def get_cache_dir(args):
    copy_args = vars(copy.deepcopy(args))
    for arg in args.un_identify_args:
        if arg in copy_args:
            del copy_args[arg]
    md5_hash = hashlib.md5(str(copy_args).encode()).hexdigest()
    model_cache_dir = os.path.join(os.environ['HF_HOME'], "token_dropping", args.output_dir + "_" + md5_hash)
    data_cache_dir = os.path.join(os.environ['HF_HOME'], "token_dropping", "data", 
        args.datasets.replace(",", "-").replace("/", "-") + 
        f"_{args.data_select_strategy}_{args.data_select_ratio}_{args.random_seed}_{args.train_ratio}_{args.context_length}")
    if args.add_eos_token:
        data_cache_dir += "_weos"
    if args.packing_samples:
        data_cache_dir += "_pack"
    if args.pre_compute_ref:
        data_cache_dir += "_wref"
    os.makedirs(model_cache_dir, exist_ok=True)
    os.makedirs(data_cache_dir, exist_ok=True)
    sys.argv += ["--model_cache_dir", model_cache_dir, "--data_cache_dir", data_cache_dir]
    return model_cache_dir, data_cache_dir


def build_prompt(element, prompt, entries, entry_func):
    """
    Build prompt for the given element.
    """
    element["prompt"] = [
        prompt.format(**entry_func({entry: element[entry][i] for entry in entries})) 
        for i in range(len(element[entries[0]]))
    ]
    return element


def get_tokenizer(tokenizer_path):
    return AutoTokenizer.from_pretrained(tokenizer_path)


def build_instruct(
        ds: Dataset, cache_path, prompt="Question: {instruction}\n\nAnswer: {response}",
        entries=["instruction", "response"], entry_func=lambda x:x) -> Dataset:
    """
    Build instruction prompt for the given dataset.
    Another template: 
    prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n ### Instruction:\n{instruction}\n\n ### Response: Let's think step by step. {response}"
    """
    return ds.map(
        build_prompt, 
        fn_kwargs={"prompt": prompt, "entries": entries, "entry_func": entry_func},
        num_proc=psutil.cpu_count(), 
        batched=True, batch_size=1000,
        # load_from_cache_file=True, 
        # cache_file_name=cache_path,
        remove_columns=entries
    ).rename_column("prompt", "text")


def build_pretrain_data(
        datasets: list[str], train_val_split: list[float], seed=None,
        data_select_ratio=1.0, data_select_strategy="random"
    ) -> Dataset:
    """
    (1) High-quality math data:
        math-ai/AutoMathText
        meta-math/MetaMathQA
        math-ai/StackMathQA
        microsoft/orca-math-word-problems-200k
        
    (2) Instruction Tuning / DPO:
        TIGER-Lab/MathInstruct
        xinlai/Math-Step-DPO-10K
        
    (3) Open web math data:
        open-web-math/open-web-math
    """
    ds_mix = []
    
    for dataset in datasets:
        # Create cache directory
        dataset_cache_dir = os.path.join(os.environ['HF_HOME'], "proc_data", dataset.replace("/", "_"))
        os.makedirs(dataset_cache_dir, exist_ok=True)
        # Load and process dataset
        if dataset == "math-ai/AutoMathText":
            ds = concatenate_datasets([
                load_dataset(dataset, "web-0.50-to-1.00", split="train").remove_columns(["url", "date", "meta"]),
                load_dataset(dataset, "arxiv-0.50-to-1.00", split="train").remove_columns(["url", "title", "abstract", "meta"])
            ])
        elif dataset == "meta-math/MetaMathQA": # 
            ds = load_dataset(dataset, split="train").remove_columns(["type", "original_question"]).rename_columns({"query": "instruction"})
            ds = build_instruct(ds, os.path.join(dataset_cache_dir, "prompt.cache"))
        elif dataset == "math-ai/StackMathQA":
            ds = load_dataset(dataset, "stackmathqa1600k", split="train").remove_columns(["meta"]).rename_columns({"Q": "instruction", "A": "response"})
            ds = build_instruct(ds, os.path.join(dataset_cache_dir, "prompt.cache"))
        elif dataset == "microsoft/orca-math-word-problems-200k": # 
            ds = load_dataset(dataset, split="train").rename_columns({"question": "instruction", "answer": "response"})
            ds = build_instruct(ds, os.path.join(dataset_cache_dir, "prompt.cache"))
        elif dataset == "TIGER-Lab/MathInstruct": # 
            ds = load_dataset(dataset, split="train").remove_columns(["source"]).rename_column("output", "response")
            ds = build_instruct(ds, os.path.join(dataset_cache_dir, "prompt.cache"))
        elif dataset == "open-web-math/open-web-math":
            ds = load_dataset(dataset, split="train").remove_columns(["url", "date", "metadata"])
        elif dataset == "DKYoon/SlimPajama-6B":
            ds = load_dataset(dataset, split="train").remove_columns(["meta", "__index_level_0__"])
        elif dataset == "hkust-nlp/dart-math-uniform":
            ds = load_dataset(dataset, split="train").rename_column("query", "instruction")
            ds = build_instruct(ds, os.path.join(dataset_cache_dir, "prompt.cache"))
        elif dataset == "Vivacem/MMIQC":
            ds = load_dataset(dataset, split="train").remove_columns(["source"]).rename_column("output", "response")
            ds = build_instruct(ds, os.path.join(dataset_cache_dir, "prompt.cache"))
        elif dataset == "hkust-nlp/dart-math-hard": #
            ds = load_dataset(dataset, split="train").rename_column("query", "instruction")
            ds = build_instruct(ds, os.path.join(dataset_cache_dir, "prompt.cache"))
        elif dataset == "allenai/math_qa":
            ds = concatenate_datasets([
                load_dataset(dataset, split="train", trust_remote_code=True),
                load_dataset(dataset, split="validation", trust_remote_code=True),
                load_dataset(dataset, split="test", trust_remote_code=True)
            ]).remove_columns(["annotated_formula", "category", "correct", "linear_formula"])
            def entry_func(x):
                x['Rationale'] = x['Rationale'].strip("\"")
                return x
            ds = build_instruct(
                ds, os.path.join(dataset_cache_dir, "prompt.cache"), 
                prompt="Question: {Problem} What of the following is the right choice? Explain your answer. {options}. Answer: {Rationale}", 
                entries=["Problem", "options", "Rationale"], entry_func=entry_func
            )
        elif dataset == "lighteval/MATH":
            ds = concatenate_datasets([
                load_dataset(dataset, split="train", trust_remote_code=True),
                load_dataset(dataset, split="test", trust_remote_code=True)
            ]).remove_columns(["level", "type"])
            ds = build_instruct(
                ds, os.path.join(dataset_cache_dir, "prompt.cache"),
                prompt="Question: {problem} Answer: {solution}", 
                entries=["problem", "solution"]
            )
        elif dataset == "peiyi9979/Math-Shepherd":
            ds = load_dataset(dataset, split="train").remove_columns(["input", "task"])
            def entry_func(x):
                x['label'] = x['label'].replace("+\n", "").replace("-\n", "").strip(' -')
                return x
            ds = build_instruct(
                ds, os.path.join(dataset_cache_dir, "prompt.cache"),
                prompt="{label}", entries=["label"], entry_func=entry_func
            )
        elif dataset == "nvidia/OpenMathInstruct-2":
            ds = load_dataset(dataset, split='train_5M')
            ds = build_instruct(
                ds, os.path.join(dataset_cache_dir, "prompt.cache"),
                prompt="Question: {problem} Answer: {generated_solution}", 
                entries=["problem", "generated_solution"]
            )
        ds_mix.append(ds)

    # Concatenate all datasets and shuffle
    ds_mix: Dataset = concatenate_datasets(ds_mix)
    ds_mix = ds_mix.shuffle(seed=seed)

    # Try data selection
    assert 0 < data_select_ratio <= 1
    if data_select_ratio < 1:
        if data_select_strategy == "random":
            ds_mix = ds_mix.select(range(int(len(ds_mix) * data_select_ratio)))
            print(f"Selecting {data_select_ratio * 100}% of the data randomly")
        else:
            raise ValueError("Unknown data selection strategy")

    # Split into train and valid
    ds_mix_split = DatasetDict({
        "train": ds_mix.select(range(int(len(ds_mix) * train_val_split[0]))),
        "valid": ds_mix.select(range(int(len(ds_mix) * train_val_split[0]), len(ds_mix)))
    })
    
    print(ds_mix)
    
    # print("Dataset examples: (train and valid)")
    # print(ds_mix_split["train"][0])
    # print(ds_mix_split["valid"][0])
    
    return ds_mix_split


class VllmClient:
    def __init__(self, url):
        import requests
        from openai import OpenAI
        self.client = OpenAI(api_key="EMPTY", base_url=url)
        # response = requests.get(url + "/v1/models")
        # self.vllm_model = json.loads(response.text)['data'][0]['id']
        # import pdb; pdb.set_trace()
        self.vllm_model = self.client.models.list().data[0].id
        # # Test Request
        # completion = self.client.completions.create(
        #     model=self.vllm_model, prompt=["How is the weather today?"],
        #     echo=True, n=1, max_tokens=1, stream=False, logprobs=1,
        # )
        # print([choice.logprobs.token_logprobs[1:-1] for choice in completion.choices])
    
    def get_ref_loss(self, messages):
        completion = self.client.completions.create(
            model=self.vllm_model, prompt=messages,
            echo=True, 
            n=1, max_tokens=1, stream=False, logprobs=1,
        )
        return [[-e for e in choice.logprobs.token_logprobs[1:-1]] for choice in completion.choices]


def build_tokenized_datasets(args=None):
    
    # Try to load dataset from disk first
    try:
        print("Trying to load dataset from", args.data_cache_dir)
        tokenized_dataset = load_from_disk(args.data_cache_dir)
        print(tokenized_dataset)
        return tokenized_dataset
    except Exception as e:
        print("Cannot load dataset:", e, "\nBuilding dataset")

    # Only rank 0 should build and save the dataset
    should_build = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    
    if should_build:
        raw_datasets = build_pretrain_data(
            args.datasets.split(","), 
            train_val_split=[args.train_ratio, 1-args.train_ratio], 
            seed=args.random_seed,
            data_select_ratio=args.data_select_ratio,
            data_select_strategy=args.data_select_strategy
        )
        
        tokenizer = get_tokenizer(args.model_path)
        
        if args.pre_compute_ref:
            ref_batch_size = 8
            if args.ref_model_backend == "hf":
                ref_model = AutoModelForCausalLM.from_pretrained(
                    args.ref_model_path, torch_dtype=torch.bfloat16, device_map="auto").eval()
                ref_tokenizer = AutoTokenizer.from_pretrained(args.ref_model_path)
                # Align the tokenizer
                ref_tokenizer.add_bos_token = tokenizer.add_bos_token
                if ref_tokenizer.pad_token is None:
                    ref_tokenizer.pad_token = ref_tokenizer.eos_token
            elif args.ref_model_backend == "vllm":
                vllm_client = VllmClient(args.vllm_url)
            elif args.ref_model_backend == "socket":
                recv_send_queues = init_multiple_sockets(args.ref_socket_hosts, args.ref_socket_ports)
        
        tokenizer_args = {
            "max_length": args.context_length, "truncation": True,
            "return_overflowing_tokens": True, "return_length": True,
            # "padding_side": "left"
        }
        
        def tokenize(element, tokenizer=tokenizer, args=args):
            text_list = [
                ((e + tokenizer.eos_token) if args.add_eos_token else e)
                for e in element["text"]
            ]
            import time
            start_time = time.time()
            # if args.packing_samples:
            #     text_list = " ".join(text_list)
            outputs = tokenizer(text_list, **tokenizer_args)
            if args.packing_samples:
                # dict_keys(['input_ids', 'attention_mask', 'length', 'overflow_to_sample_mapping'])
                new_outputs = {
                    "input_ids": [],
                    # "attention_mask": [],
                    "length": [],
                    "overflow_to_sample_mapping": []
                }
                i = 0
                tmp_input_ids = []
                sample_slice_start = 0
                while i < len(outputs["input_ids"]):
                    if len(tmp_input_ids) < args.context_length:
                        len_in_need = args.context_length-len(tmp_input_ids)
                        tmp_input_ids += outputs["input_ids"][i][
                            sample_slice_start:sample_slice_start+len_in_need]
                        if len_in_need >= (len(outputs["input_ids"][i])-sample_slice_start):
                            sample_slice_start = 0
                            i += 1
                        else:
                            sample_slice_start += len_in_need
                    if len(tmp_input_ids) == args.context_length or i == len(outputs["input_ids"]): # (drop the last )
                        new_outputs["input_ids"].append(tmp_input_ids)
                        # new_outputs["attention_mask"].append([1] * len(tmp_input_ids))
                        new_outputs["length"].append(len(tmp_input_ids))
                        new_outputs["overflow_to_sample_mapping"].append(i)
                        tmp_input_ids = []
                assert sum(new_outputs["length"]) == sum(outputs["length"])
                del outputs
                outputs = new_outputs
                print("Sample Packing Done")
            # print(f"Tokenization time: {time.time() - start_time:.2f}s")

            if args.pre_compute_ref:
                if args.ref_model_backend == "hf":
                    outputs["ref_loss"] = []
                    for i in range(0, len(outputs["input_ids"]), ref_batch_size):
                        outputs["ref_loss"] += obtain_ref_loss(
                            model=ref_model, tokenizer=ref_tokenizer, 
                            target_tokenizer=tokenizer, messages=outputs['input_ids'][i:i+ref_batch_size],
                            tokenize_args={
                                "max_length": args.context_length, "truncation": True,
                                "return_overflowing_tokens": False, "return_length": False
                            }, 
                            return_pickle=False, manuel_matching=False
                        )
                elif args.ref_model_backend == "vllm":
                    outputs["ref_loss"] = vllm_client.get_ref_loss(
                        tokenizer.batch_decode(outputs["input_ids"])
                    )
                elif args.ref_model_backend == "socket":
                    start_time = time.time()
                    outputs["ref_loss"] = []
                    counter = 0
                    sq = len(recv_send_queues)
                    for i in range(0, len(outputs["input_ids"]), ref_batch_size):
                        recv_send_queues[counter % sq][1].put(outputs["input_ids"][i:i+ref_batch_size])
                        counter += 1
                    counter = 0
                    for i in range(0, len(outputs["input_ids"]), ref_batch_size):
                        outputs["ref_loss"] += recv_send_queues[counter % sq][0].get()['loss']
                        counter += 1
                    print(f"Ref loss time: {time.time() - start_time:.2f}s")
                else:
                    raise NotImplementedError(f"Backend {args.ref_model_backend} is not supported for pre-compute ref loss.")
            return outputs
        
        print("torch.distributed.is_initialized() =", torch.distributed.is_initialized())
        
        # Tokenize the dataset
        tokenized_dataset = raw_datasets.map(
            tokenize, remove_columns=raw_datasets["train"].column_names,
            num_proc=1 if args.pre_compute_ref else psutil.cpu_count(logical=False),
            batched=True, batch_size=10000,
            # load_from_cache_file=True, 
            # ToDo: this cache is not necessary!
            # cache_file_names={k: os.path.join(args.data_cache_dir, f"{k}_tokens.cache") for k in raw_datasets.keys()}, 
            fn_kwargs={"tokenizer": tokenizer, "args": args}
        )
        
        # Save to disk
        tokenized_dataset.save_to_disk(args.data_cache_dir)
        print(f"Num training tokens: {sum(tokenized_dataset['train']['length'])/(1000**2):.1f}M tokens")
        
        # Clean up ref model resources
        if args.pre_compute_ref and args.ref_model_backend == "socket":
            # Close the socket
            for recv_q, send_q in recv_send_queues:
                send_q.put([])
                recv_q.close()
                send_q.close()
        
        if args.ref_model_backend == "hf":
            del ref_model
            torch.cuda.empty_cache()
    
    # Wait for rank 0 to finish building and saving
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # All ranks load from disk
    tokenized_dataset = load_from_disk(args.data_cache_dir)
    print(tokenized_dataset)
    
    return tokenized_dataset


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--datasets", type=str, default="meta-math/MetaMathQA")
    args.add_argument("--output-json", type=str, default=None)
    args_parse = args.parse_args()
    
    if args_parse.output_json is None:
        build_pretrain_data(args_parse.datasets.split(","), train_val_split=[0.99, 0.01], seed=None)
    else:
        print("Saving Json File to", args_parse.output_json)
        ds = build_pretrain_data(args_parse.datasets.split(","), train_val_split=[0.99, 0.01], seed=None)
        concatenate_datasets([ds['train'], ds['valid']]).to_json(args_parse.output_json, lines=True)
    