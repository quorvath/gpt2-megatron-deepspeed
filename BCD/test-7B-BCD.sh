#!/bin/bash

PP_SIZE=1
# TP=1
NUM_GPUS=1

GLOBAL_BATCH_SIZE=128
MICRO_BATCH_SIZE=64

SEQLEN=256
LR=0.0005
MIN_LR=0.0001
VOCAB_PATH=./dataset/vocab.json
MERGE_PATH=./dataset/merges.txt

LOG_INTERVAL=100
SAVE_INTERVAL=1000
EVAL_INTERVAL=100
EVAL_ITERS=10

DATA_PATH=../../gpt-dataset/wiki/wiki

NAME="gpt2-pp${PP_SIZE}-bsz${GLOBAL_BATCH_SIZE}-mbsz${MICRO_BATCH_SIZE}-BCD"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
host="${HOSTNAME}"
TENSORBOARD_DIR="./tensorboard/${NAME}_${host}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}
CHECKPOINT_PATH="./checkpoints/${NAME}"

NUM_LAYERS=32

GPT_ARGS=" \
    --num-layers $NUM_LAYERS \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length $SEQLEN \
    --max-position-embeddings $SEQLEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --data-impl mmap \
    --train-iters 1107026 \
    --vocab-file $VOCAB_PATH \
    --merge-file $MERGE_PATH \
    --lr $LR \
    --min-lr $MIN_LR \
    --lr-decay-style cosine \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --weight-decay 0.00001 \
    --clip-grad 1.0 \
    --hysteresis 2  \
    --seed 42 \
    --trainblock \
    --num-groups 3 \
    "
OUTPUT_ARGS=" \
    --log-interval $LOG_INTERVAL \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters $EVAL_ITERS \
    --checkpoint-activations
    "

DATA_ARGS=" \
    --split 949,50,1 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --tensorboard-dir $TENSORBOARD_DIR \
    --tensorboard-queue-size 1 \
    --init-method-std 0.02 \
    --log-timers-to-tensorboard \
    --use-checkpoint-lr-scheduler
    "

config_json="./dataset/deepspeed_config.json"


cat <<EOT > $config_json
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "steps_per_print": 1,
  "wall_clock_breakdown": false,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": $LR,
      "betas": [0.9, 0.95]
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": $MIN_LR,
      "warmup_max_lr": $LR,
      "warmup_num_steps": 1000
    },
    "min_lr": $MIN_LR 
  }
}
EOT

deepspeed_options=" \
		    --deepspeed \
		    --deepspeed_config ${config_json} \
        --deepspeed-activation-checkpointing \
        --partition-activations \
      "

ALL_ARGS="$GPT_ARGS $OUTPUT_ARGS $DATA_ARGS $deepspeed_options"

# if you can't stand pt-1.9 launcher noise
export LOGLEVEL=WARNING

LAUNCHER="deepspeed --master_port=32000 --include localhost:0"

export CMD=" \
    $LAUNCHER ../pretrain_gpt.py \
    --pipeline-model-parallel-size $PP_SIZE \
    --distributed-backend nccl \
    $ALL_ARGS \
  "
    # --tensor-model-parallel-size $TP \
echo $CMD

$CMD >> ${NAME}.log
