#!/bin/bash

# Run custom parallel GPT with Megatron framework
# Uses PyTorch SDPA for attention, Megatron for TP/PP/1F1B

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "Custom Parallel GPT with Megatron Framework"
echo "============================================"
echo "Attention: PyTorch SDPA"
echo "TP/PP/1F1B: Megatron-LM"
echo "============================================"

# Configuration
NUM_GPUS=4
TP_SIZE=2
PP_SIZE=2
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=8

NUM_LAYERS=8
HIDDEN_SIZE=256
NUM_HEADS=8
SEQ_LENGTH=128
VOCAB_SIZE=1024

echo "Config: ${NUM_GPUS} GPUs (TP=${TP_SIZE}, PP=${PP_SIZE})"
echo "Layers: ${NUM_LAYERS}, Hidden: ${HIDDEN_SIZE}, Heads: ${NUM_HEADS}"
echo "============================================"

# Set required environment variables for Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Run with torchrun + Megatron args
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=6000 \
    custom_parallel_gpt.py \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters 10 \
    --lr 0.0001 \
    --lr-decay-style constant \
    --vocab-size ${VOCAB_SIZE} \
    --padded-vocab-size ${VOCAB_SIZE} \
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
echo "Training completed!"
echo "============================================"
