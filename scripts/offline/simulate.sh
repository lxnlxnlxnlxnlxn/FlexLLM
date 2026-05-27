#!/bin/bash

# Model and dataset config
MODEL_NAME="llama-3-8b"
DATASET="alpaca"

# GPU resource config
GPU_BLOCK=2048

# Generation config
OUTPUT_LEN=256

# FlexLLM scheduling config
ONLOAD_START=0.3
ONLOAD_END=0.8
BACKUP_RATE=0.5
BLOCK_SIZE=16

# Run simulator
python /workspace/flexllm/simulate.py \
    --model_name ${MODEL_NAME} \
    --dataset ${DATASET} \
    --gpu_block ${GPU_BLOCK} \
    --output_len ${OUTPUT_LEN} \
    --onload_start ${ONLOAD_START} \
    --onload_end ${ONLOAD_END} \
    --backup_rate ${BACKUP_RATE} \
    --block_size ${BLOCK_SIZE}
    