#!/bin/bash

# Model config
MODEL_NAME="llama-3-8b"

# Batch size settings
BS_STEP=8
BS_MAX=64

# Sequence length settings
SEQLEN_STEP=64
SEQLEN_MAX=512

# Test parameters
GROUP=5
OUTLEN=128
VOCAB_SIZE=128256
FLU=0.1

# Run benchmark
python /workspace/flexllm/benchmark_forward.py \
    --model_name ${MODEL_NAME} \
    --bs_step ${BS_STEP} \
    --bs_max ${BS_MAX} \
    --seqlen_step ${SEQLEN_STEP} \
    --seqlen_max ${SEQLEN_MAX} \
    --group ${GROUP} \
    --outlen ${OUTLEN} \
    --vocab_size ${VOCAB_SIZE} \
    --flu ${FLU}
    