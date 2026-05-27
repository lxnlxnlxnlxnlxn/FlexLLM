#!/bin/bash

# Device Config
DEVICE_NUM=1

# Testcase Config
START_SIZE=65536
END_SIZE=$((65535 * 2048))
INCREMENT_SIZE=65536

CUDA_VISIBLE_DEVICES=${DEVICE_NUM}
python /workspace/tests/offline/pcie.py \
    --start_size ${START_SIZE} \
    --end_size ${END_SIZE} \
    --increment_size ${INCREMENT_SIZE} \
    --device ${DEVICE_NUM}
