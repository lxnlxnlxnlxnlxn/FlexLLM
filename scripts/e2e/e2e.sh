#!/bin/bash

# Basic config
MODEL_NAME="llama-3-8b"
DATASET="alpaca"

# FlexLLM scheduling thresholds
ONLOAD_START=0.3
ONLOAD_END=0.8
BACKUP_RATE=0.5
BLOCK_SIZE=16

# Resource config
GPU_BLOCK=2048
CPU_BLOCK=8192
BATCH_SIZE=64
TESTCASE_NUM=100
OUTPUT_LEN=256

# Run FlexLLM inference
CUDA_VISIBLE_DEVICES=0
python /workspace/flexllm/run_inference.py \
    --model_name ${MODEL_NAME} \
    --onload_start ${ONLOAD_START} \
    --onload_end ${ONLOAD_END} \
    --backup_rate ${BACKUP_RATE} \
    --block_size ${BLOCK_SIZE} \
    --dataset ${DATASET} \
    --gpu_block ${GPU_BLOCK} \
    --cpu_block ${CPU_BLOCK} \
    --batch_size ${BATCH_SIZE} \
    --testcase_num ${TESTCASE_NUM} \
    --output_len ${OUTPUT_LEN}
    