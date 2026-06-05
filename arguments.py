import argparse

non_id_args = [
    # The following arguments are not used in creating the log-file name (i.e., md5)
    "ref_socket_hosts", "ref_socket_ports", "vllm_url", 
    "ref_model_path", "pre_compute_ref", 
]

def parse_args(include_deep_speed_args=False, return_parsed=True):
    args = argparse.ArgumentParser()

    args.add_argument('--local_rank', type=int, default=-1,
                      help='local rank passed from distributed launcher')
    
    args.add_argument("--task", type=str, default="train", choices=["gen-data", "train", "eff-benchmark", "param-search"])

    group = args.add_argument_group("Model and Data")
    group.add_argument("--datasets", type=str, default="open-web-math/open-web-math")
    group.add_argument("--dataset_path", type=str, help="provide the path of previous built dataset")
    group.add_argument("--model_path", type=str, default="TinyLlama/TinyLlama_v1.1")
    group.add_argument("--output_dir", type=str, default="debug")
    group.add_argument("--model_cache_dir", type=str, default=None, help="will be named automatically")
    group.add_argument("--data_cache_dir", type=str, default=None, help="will be named automatically")
    group.add_argument("--data_select_ratio", type=float, default=1.0)
    group.add_argument("--data_select_strategy", type=str, default="random")
    group.add_argument("--add_eos_token", action="store_true", default=False)
    group.add_argument("--packing_samples", action="store_true", default=False)
    group.add_argument("--use_lora", action="store_true", default=False)
    group.add_argument("--attn_impl", type=str, default="flash_attention_2", choices=["eager", "flash_attention_2", "sdpa"])
    
    group = args.add_argument_group("Training")
    group.add_argument("--batch_size", type=int, default=512)
    group.add_argument("--micro_batch_size", type=int, default=1)
    group.add_argument("--weight_decay", type=float, default=1e-2)
    group.add_argument("--context_length", type=int, default=2048)
    group.add_argument("--random_seed", type=int, default=1234)
    group.add_argument("--epochs", type=int, default=1)
    group.add_argument("--learning_rate", type=float, default=6e-5)
    group.add_argument("--use_layer_lr", action="store_true", default=False,
                       help="Enable different learning rates for different layers")
    group.add_argument("--layer_lr_decay", type=float, default=1.0,
                       help="Learning rate decay from output to input")
    group.add_argument("--eval_steps", type=int, default=1000)
    group.add_argument("--gradient_accumulation_steps", type=int, help="will be calculated automatically")
    group.add_argument("--train_ratio", type=float, default=0.999)
    group.add_argument("--save_steps", type=int, default=100)
    
    group = args.add_argument_group("Token Dropping")
    group.add_argument("--ref_model_backend", 
                       type=str, default="none", choices=["hf", "vllm", "socket", "none", "self"])
    group.add_argument("--ref_socket_hosts", 
                       type=str, default="None")
    group.add_argument("--ref_socket_ports", 
                       type=str, default="None")
    group.add_argument("--vllm_url", 
                       type=str, default="None")
    group.add_argument("--ref_model_path",
                        type=str, default="None")
    group.add_argument("--dropping_strategy",
                        type=str, default="none", choices=["none", "fixed", "positive", "dynamic", "peaking", "less_ref"])
    group.add_argument("--drop_rate",
                        type=float, default=0)
    group.add_argument("--pre_compute_ref",
                        action="store_true", default=False)
    group.add_argument("--attn_filter", 
                        action="store_true", default=False)
    group.add_argument("--forward_filter", 
                        action="store_true", default=False, help="Not implemented yet")
    group.add_argument("--filter_opt", 
                       default=0, type=int, 
                       help="0: no opt, 1: MLP, 2: + Attn MM, 3: + Attn MM + Attn Out")
    
    # The following arguments are not used in creating the log-file name (i.e., md5)
    un_identify_args = [
        "ref_socket_hosts", "ref_socket_ports",
        "vllm_url", "pre_compute_ref", "local_rank",
        "model_cache_dir", "data_cache_dir",
        "output_dir", "attn_impl"
    ]
    group = args.add_argument_group("Utils")
    group.add_argument("--un_identify_args", 
                       required=False, default=un_identify_args + ["un_identify_args"], nargs="+")
    
    if include_deep_speed_args:
        import deepspeed
        deepspeed.add_config_arguments(args)
    
    if return_parsed:
        return args.parse_args()
    else:
        return args