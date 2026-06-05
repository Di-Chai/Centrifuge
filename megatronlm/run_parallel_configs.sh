#!/bin/bash

# Test different parallel configurations
# All using Megatron's TP/PP/1F1B framework

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "Testing Multiple Parallel Configurations"
echo "============================================"

# Set required environment variables for Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Common parameters
NUM_LAYERS=8
HIDDEN_SIZE=256
NUM_HEADS=8
SEQ_LENGTH=128
VOCAB_SIZE=1024
MICRO_BATCH_SIZE=2
TRAIN_ITERS=5

# Function to run a configuration
run_config() {
    local num_gpus=$1
    local tp_size=$2
    local pp_size=$3
    local global_batch_size=$4
    
    echo ""
    echo "----------------------------------------"
    echo "Config: ${num_gpus} GPUs (TP=${tp_size}, PP=${pp_size})"
    echo "Global Batch Size: ${global_batch_size}"
    echo "----------------------------------------"
    
    torchrun \
        --nproc_per_node=${num_gpus} \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=6000 \
        custom_parallel_gpt.py \
        --tensor-model-parallel-size ${tp_size} \
        --pipeline-model-parallel-size ${pp_size} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_HEADS} \
        --seq-length ${SEQ_LENGTH} \
        --max-position-embeddings ${SEQ_LENGTH} \
        --micro-batch-size ${MICRO_BATCH_SIZE} \
        --global-batch-size ${global_batch_size} \
        --train-iters ${TRAIN_ITERS} \
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
    
    echo "✓ Config completed"
}

# Check available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Available GPUs: ${NUM_GPUS}"

if [ $NUM_GPUS -ge 4 ]; then
    echo ""
    echo "Testing with 4 GPUs..."
    
    # Test 1: TP=2, PP=2
    run_config 4 2 2 8
    
    # Test 2: TP=4, PP=1
    run_config 4 4 1 8
    
    # Test 3: TP=1, PP=4
    run_config 4 1 4 8
    
elif [ $NUM_GPUS -ge 2 ]; then
    echo ""
    echo "Testing with 2 GPUs..."
    
    # Test 1: TP=2, PP=1
    run_config 2 2 1 4
    
    # Test 2: TP=1, PP=2
    run_config 2 1 2 4
    
else
    echo "ERROR: Need at least 2 GPUs to test parallelism"
    exit 1
fi

echo ""
echo "============================================"
echo "All configurations tested successfully!"
echo "============================================"
