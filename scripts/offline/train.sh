#!/bin/bash

# Model config
MODEL_NAME="llama-3-8b"
STAGE="prefill"

# Run training script
python /workspace/flexllm/train_predictor.py \
    --model_name ${MODEL_NAME} \
    --stage ${STAGE}
    