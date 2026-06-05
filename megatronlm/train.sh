GPUS_PER_NODE=8
# MASTER_ADDR=localhost
# MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK"
CHECKPOINT_PATH=/data/codeparrot-small
VOCAB_FILE=/data/gpt2-vocab.json
MERGE_FILE=/data/gpt2-merges.txt
DATA_PATH=/data/codeparrot/codeparrot_content_document
GPT_ARGS="--num-layers 12
--hidden-size 1024
--num-attention-heads 16
--seq-length 1024
--max-position-embeddings 1024
--micro-batch-size 4
--global-batch-size 32
--lr 0.0005
--train-iters 150000
--lr-decay-iters 150000
--lr-decay-style cosine
--lr-warmup-iters 2000
--weight-decay .1
--adam-beta2 .999
--bf16
--log-interval 10
--save-interval 100
--eval-interval 200
--eval-iters 10
"
# TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        $GPT_ARGS \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        # $TENSORBOARD_ARGS

