#!/bin/bash
# SLURM batch script for ScanNet preprocessing task

TASK_ID=${1:-0}
TOTAL_TASKS=${2:-1}
SAVE_PATH=${3:-"./output/scannet"}
SAMPLE_CT=${4:-32}

# Set working directory
cd /cms_cvlab/home/chanyoung/Point3R-LLM

# Activate virtual environment
echo "Activating virtual environment..."
source venv2/bin/activate

echo "Starting ScanNet preprocessing on task $TASK_ID..."
echo "Python: $(which python3)"
echo "Working directory: $(pwd)"

CUDA_VISIBLE_DEVICES=0 python3 scripts/demo/preprocess_scannet_simple.py \
    --model-path "Qwen/Qwen3.5-9B" \
    --gpu-id $TASK_ID \
    --total-gpus $TOTAL_TASKS \
    --save-path $SAVE_PATH \
    --sample-ct $SAMPLE_CT


echo "Preprocessing complete!"
echo "##### END #####"
