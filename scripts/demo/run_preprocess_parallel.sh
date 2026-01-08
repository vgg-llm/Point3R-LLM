#!/bin/bash
# Parallel ScanNet Preprocessing Script using torchrun
# Distributes preprocessing across multiple GPUs

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # Master node IP
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NPROC_PER_NODE=8                            # Number of GPUs to use (default: 8)

# You can override the number of GPUs by passing it as an argument
if [ ! -z "$1" ]; then
    NPROC_PER_NODE=$1
fi

echo "Starting preprocessing with $NPROC_PER_NODE GPUs"
echo "Master address: $MASTER_ADDR:$MASTER_PORT"

# ======================
# Run distributed preprocessing
# ======================
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         scripts/demo/preprocess_scannet.py

echo "Preprocessing complete!"
