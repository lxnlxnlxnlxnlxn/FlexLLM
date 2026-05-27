#!/bin/bash

# GPU device
DEVICE_NUM=0

# Test parameters
TESTCASE_NUM=1000
MODEL="llama-3-8b"
DATASET="alpaca"
OUTPUT_LEN=256

# Run MII benchmark
CUDA_VISIBLE_DEVICES=${DEVICE_NUM}
python /workspace/tests/offline/mii_benchmark.py \
    --testcase_num ${TESTCASE_NUM} \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --output_len ${OUTPUT_LEN}
    