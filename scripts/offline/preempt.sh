#!/bin/bash

# Model configuration
MODEL_NAME="llama-3-8b"
BLOCK_SIZE=16
PROMPT_LEN=128
OUTPUT_LEN=128
STEP=4

# Run the analysis script
python /workspace/flexllm/analyze_recomp_swap.py \
    --model_name ${MODEL_NAME} \
    --block_size ${BLOCK_SIZE} \
    --prompt_len ${PROMPT_LEN} \
    --output_len ${OUTPUT_LEN} \
    --step ${STEP}
    