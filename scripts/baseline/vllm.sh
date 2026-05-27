#!/bin/bash

# GPU device index
DEVICE_NUM=0

# Model and dataset settings
MODEL_NAME="llama-3-8b"
DATASET="alpaca"

# KV cache configuration
GPU_BLOCK=2048
CPU_BLOCK=8192

# Inference parameters
BATCH_SIZE=64
TESTCASE_NUM=100
OUTPUT_LEN=256

# Execute vLLM native inference
CUDA_VISIBLE_DEVICES=${DEVICE_NUM}
python /workspace/tests/offline/run_vllm_native.py \
    --model_name ${MODEL_NAME} \
    --dataset ${DATASET} \
    --gpu_block ${GPU_BLOCK} \
    --cpu_block ${CPU_BLOCK} \
    --batch_size ${BATCH_SIZE} \
    --testcase_num ${TESTCASE_NUM} \
    --output_len ${OUTPUT_LEN}
    