#!/bin/bash

# GPU device number
DEVICE_NUM=0

# Test parameters
TESTCASE_NUM=1000
MODEL="llama-3-8b"
DATASET="summary"
OUTPUT_LEN=256

# Run TensorRT-LLM benchmark
CUDA_VISIBLE_DEVICES=${DEVICE_NUM}
python /workspace/tests/offline/trtllm_benchmark.py \
    --testcase_num ${TESTCASE_NUM} \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --output_len ${OUTPUT_LEN}
    