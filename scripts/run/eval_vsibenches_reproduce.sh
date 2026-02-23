#!/bin/bash
#SBATCH --job-name=Eval-VSIBench-Reproduction
#SBATCH -o sbatch_log/eval_vsibench_reproduction.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

# Experiment: Evaluate VSIBench-Reproduction

source venv/bin/activate

# --- Shared config ---
export MODEL_PATH="./outputs/@archive/vsibench_Qwen3VL_8b_memfeat_32frame"
export BENCHMARKS="vsibench_point3r"
export EVAL_MODEL_TYPE="point3r_llm_v2"

# Memory features
export MERGE_MEMORY_FEAT="True"
export MEMORY_FUSION_METHOD="add"

# RoPE
# export ROPE_MODE="pointer_timestamp"
export POINTER_FORMAT="image"

# --- Eval 1: old ckpt and new data ---
export LOG_SUFFIX="vsibench_old_ckpt_new_data"
unset POINTER_DIR_NAME # 32 frame
bash scripts/evaluation/eval.sh

# --- Eval 2: old ckpt and old data ---
export LOG_SUFFIX="vsibench_old_ckpt_old_data"
export POINTER_DIR_NAME="pointer_memory_qwen3vl_8B"
bash scripts/evaluation/eval.sh

export MODEL_PATH="./outputs/vsibench_image_format-unsorted"

# --- Eval 3: new ckpt and new data ---
export LOG_SUFFIX="vsibench_new_ckpt_new_data"
unset POINTER_DIR_NAME
bash scripts/evaluation/eval.sh

# --- Eval 4: new ckpt and old data ---
export LOG_SUFFIX="vsibench_new_ckpt_old_data"
export POINTER_DIR_NAME="pointer_memory_qwen3vl_8B"
bash scripts/evaluation/eval.sh

nvidia-smi
date
# squeue --job $SLURM_JOBID
echo "##### END #####"
