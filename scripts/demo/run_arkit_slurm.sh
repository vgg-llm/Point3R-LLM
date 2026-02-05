#!/bin/bash

#SBATCH --job-name=arkit_preprocess
#SBATCH --output=logs/arkit_chunk_%a.log
#SBATCH --error=logs/arkit_chunk_%a.err
#SBATCH --array=0-15
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Total number of chunks (should match array size)
TOTAL_CHUNKS=16

# Get current chunk from array task ID
CURR_CHUNK=$SLURM_ARRAY_TASK_ID

echo "=================================================="
echo "Starting ARKitScenes preprocessing"
echo "Chunk: $CURR_CHUNK / $TOTAL_CHUNKS"
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

# Activate conda environment (adjust as needed)
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate your_env

# Run preprocessing script
python3 scripts/demo/preprocess_arkit_simple.py \
    --curr_chunk $CURR_CHUNK \
    --total_chunks $TOTAL_CHUNKS \
    --save-path ./output/arkit \
    --sample-ct 32 \
    --lambda-decay 1.0 \
    --model-path "Qwen/Qwen3-VL-4B-Instruct"

echo "=================================================="
echo "Chunk $CURR_CHUNK completed!"
echo "=================================================="
