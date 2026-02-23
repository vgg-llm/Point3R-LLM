#!/bin/bash
# SLURM batch script for ARKitScenes preprocessing task

TASK_ID=${1:-0}
TOTAL_TASKS=${2:-1}
SAVE_PATH=${3:-"./output/arkitscenes"}
SAMPLE_CT=${4:-32}

# Set working directory
cd /cms_cvlab/home/chanyoung/yoonwoo/Point3R-LLM

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo "Starting ARKitScenes preprocessing on task $TASK_ID..."
echo "Python: $(which python3)"
echo "Working directory: $(pwd)"

CUDA_VISIBLE_DEVICES=0 python3 scripts/demo/preprocess_arkit_simple.py \
    --model-path "Qwen/Qwen3-VL-8B-Instruct" \
    --curr_chunk $TASK_ID \
    --total_chunks $TOTAL_TASKS \
    --save-path $SAVE_PATH \
    --sample-ct $SAMPLE_CT \
    --metadata-path "./scripts/demo/metadata/arkit_combined.txt"


echo "Preprocessing complete!"
echo "##### END #####"
