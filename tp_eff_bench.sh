#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=12356 megatron_tp_benchmark.py --tensor_parallel_size 4 --mode eff-benchmark --model Qwen/Qwen2.5-7B

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=12356 megatron_tp_benchmark.py --tensor_parallel_size 4 --mode eff-benchmark --model Qwen/Qwen2.5-7B --attn_filter

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=12356 megatron_tp_benchmark.py --tensor_parallel_size 8 --mode eff-benchmark --model Qwen/Qwen3-14B

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=12356 megatron_tp_benchmark.py --tensor_parallel_size 8 --mode eff-benchmark --model Qwen/Qwen3-14B --attn_filter