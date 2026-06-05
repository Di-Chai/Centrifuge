#!/bin/bash

# Test on single GPU with Megatron framework

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "Single GPU Test (TP=1, PP=1)"
echo "============================================"

# Set required environment variables for Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1

python custom_parallel_gpt.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 4 \
    --hidden-size 128 \
    --num-attention-heads 4 \
    --seq-length 64 \
    --max-position-embeddings 64 \
    --micro-batch-size 2 \
    --global-batch-size 2 \
    --train-iters 5 \
    --lr 0.0001 \
    --lr-decay-style constant \
    --vocab-size 512 \
    --padded-vocab-size 512 \
    --distributed-backend nccl \
    --transformer-impl local \
    --no-create-attention-mask-in-dataloader \
    --disable-bias-linear \
    --normalization LayerNorm \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --log-interval 1 \
    --eval-interval 1000 \
    --eval-iters 1 \
    --mock-data

echo "============================================"
echo "✓ Single GPU test passed!"
echo "============================================"
