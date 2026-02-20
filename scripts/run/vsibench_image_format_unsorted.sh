#!/bin/bash
#SBATCH --job-name=vsibench_image_format
#SBATCH -o sbatch_log/8b-memfeat-image-format-unsorted.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

# Experiment: 8B with memory features + original naiive image format

source venv/bin/activate

export EXP_NAME="vsibench_image_format-unsorted"
export MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
export DATASETS="vsibench_point3r"

# Memory features
export MERGE_MEMORY_FEAT="True"
export MEMORY_FUSION_METHOD="add"

# RoPE
# export ROPE_MODE="pointer_timestamp"
export POINTER_FORMAT="image"

# Evaluation
export BENCHMARKS="vsibench_point3r"
export EVAL_MODEL_TYPE="point3r_llm_v2"
export LOG_SUFFIX="vsibench_eval"

# Pointer DIR Name
export POINTER_DIR_NAME="pointer_memory_qwen3vl_8B"

# --- Train ---
bash scripts/train/train.sh

# --- Evaluate ---
export MODEL_PATH="./outputs/${EXP_NAME}"
bash scripts/evaluation/eval.sh

nvidia-smi
date
squeue --job $SLURM_JOBID
echo "##### END #####"
