#!/bin/bash

# TASK=$1
# OUTPUT_DIR=$2
# NODE_RANK=$3
# HOSTS="10.0.3.1:0,1,2,3,4,5,6,7@10.0.5.1:4,5,6,7@10.0.14.1:0,1,2,3"

INPUT_ARGS=()

# ToDo: add more models and datasets in the future
MODEL=TinyLlama/TinyLlama_v1.1
DATASETS=open-web-math/open-web-math
BATCH_SIZE=1024
MIC_BATCH_SIZE=1
LR=5e-5
CONTEXT_LEN=2048
EPOCHS=1
MASTER_PORT=8010
ATTN_IMPL=flash_attention_2
RANDOM_SEED=4321
DATA_SELECTION_RATIO=1.0

# Parse input arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --task) TASK="$2"; echo "Task: $TASK"; shift ;;
    --output_dir) OUTPUT_DIR="$2"; echo "Output directory: $OUTPUT_DIR"; shift ;;
    --node_rank) NODE_RANK="$2"; echo "Node rank: $NODE_RANK"; shift ;;
    --hosts) HOSTS="$2"; echo "Hosts: $HOSTS"; shift ;;

    --master_addr) MASTER_ADDR="$2"; echo "Master address: $MASTER_ADDR"; shift ;;
    --master_port) MASTER_PORT="$2"; echo "Master port: $MASTER_PORT"; shift ;;

    --data_selection_ratio) DATA_SELECTION_RATIO="$2"; echo "Data selection ratio: $DATA_SELECTION_RATIO"; shift ;;
    --model_path) MODEL="$2"; echo "MODEL updated to $MODEL"; shift ;;
    --attn_impl) ATTN_IMPL="$2"; echo "ATTN_IMPL updated to $ATTN_IMPL"; shift ;;
    --datasets) DATASETS="$2"; echo "DATASETS updated to $DATASETS"; shift ;;
    --dataset_path) INPUT_ARGS+=(--dataset_path "$2"); shift ;;
    --ref_model_backend) INPUT_ARGS+=(--ref_model_backend "$2"); shift ;;
    --ref_socket_hosts) INPUT_ARGS+=(--ref_socket_hosts "$2"); shift ;;
    --ref_socket_ports) INPUT_ARGS+=(--ref_socket_ports "$2"); shift ;;
    --dropping_strategy) INPUT_ARGS+=(--dropping_strategy "$2"); shift ;;
    --drop_rate) INPUT_ARGS+=(--drop_rate "$2"); shift ;;
    --epochs) EPOCHS="$2"; echo "EPOCHS updated to $EPOCHS"; shift ;;
    --attn_filter) INPUT_ARGS+=(--attn_filter) ;;
    --pre_compute_ref) INPUT_ARGS+=(--pre_compute_ref) ;;
    --add_eos_token) INPUT_ARGS+=(--add_eos_token) ;;
    --filter_opt) INPUT_ARGS+=(--filter_opt "$2"); shift ;;
    --use_lora) INPUT_ARGS+=(--use_lora) ;;
    --packing_samples) INPUT_ARGS+=(--packing_samples) ;;
    --batch_size) BATCH_SIZE="$2"; echo "BATCH_SIZE updated to $BATCH_SIZE"; shift ;;
    --learning_rate) LR="$2"; echo "LR updated to $LR"; shift ;;
    --mic_batch_size) MIC_BATCH_SIZE="$2"; echo "MIC_BATCH_SIZE updated to $MIC_BATCH_SIZE"; shift ;;
    --context_length) CONTEXT_LEN="$2"; echo "CONTEXT_LEN updated to $CONTEXT_LEN"; shift ;;
    --layer_lr_decay) INPUT_ARGS+=(--layer_lr_decay "$2"); shift ;;
    --random_seed) RANDOM_SEED="$2"; echo "RANDOM_SEED updated to $RANDOM_SEED"; shift ;;
  esac
  shift
done

echo "INPUT_ARGS: ${INPUT_ARGS[@]}"

MODEL_DATA=(
  --datasets $DATASETS 
  --model_path $MODEL 
  --output_dir $OUTPUT_DIR
  --attn_impl $ATTN_IMPL
)

# source ./scripts/owm_tinyllama_train_config.sh
TRAIN_ARGS=(
  --data_select_ratio $DATA_SELECTION_RATIO
  --data_select_strategy random
  --batch_size $BATCH_SIZE 
  --micro_batch_size $MIC_BATCH_SIZE
  --weight_decay 1e-2 
  --context_length $CONTEXT_LEN 
  --random_seed $RANDOM_SEED 
  --epochs $EPOCHS
  --learning_rate $LR
  --train_ratio 0.999 
  --eval_steps 1000
  --save_steps 100
)

echo "deepspeed --include=$HOSTS \
  --hostfile=configs/hostfile \
  --no_ssh --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  train.py --task $TASK ${MODEL_DATA[@]} ${TRAIN_ARGS[@]} ${INPUT_ARGS[@]} \
  --deepspeed"

deepspeed --include=$HOSTS \
  --hostfile=configs/hostfile \
  --no_ssh --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  train.py --task $TASK ${MODEL_DATA[@]} ${TRAIN_ARGS[@]} ${INPUT_ARGS[@]} \
  --deepspeed