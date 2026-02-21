#!/bin/bash
#SBATCH --job-name=Point3R-LLM-frame-scalability-ablation
#SBATCH -o sbatch_log/eval_frame_scalability.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

# Experiment: Evaluate scan2cap_point3r with different pointer_dir_name values

source venv/bin/activate

# --- Shared config ---
export MODEL_PATH="./outputs/vsibench_image_with_timestep_rope"
export BENCHMARKS="scan2cap_point3r"
export EVAL_MODEL_TYPE="point3r_llm_v2"

# Memory features
export MERGE_MEMORY_FEAT="True"
export MEMORY_FUSION_METHOD="add"

# RoPE
export ROPE_MODE="pointer_timestamp"
export POINTER_FORMAT="image"

# Evaluation
export BENCHMARKS="vsibench_point3r"
export EVAL_MODEL_TYPE="point3r_llm_v2"

# --- Eval 1: Default (32 frame) ---
export LOG_SUFFIX="vsibench_eval_32frame"
unset POINTER_DIR_NAME # 32 frame
bash scripts/evaluation/eval.sh

# --- Eval 2: pointer_memory_16_frame ---
export LOG_SUFFIX="vsibench_eval_16frame"
export POINTER_DIR_NAME="pointer_memory_16_frame"
bash scripts/evaluation/eval.sh

# --- Eval 3: pointer_memory_64frame ---
export LOG_SUFFIX="vsibench_eval_64frame"
export POINTER_DIR_NAME="pointer_memory_64_frame"
bash scripts/evaluation/eval.sh

# --- Eval 4: pointer_memory_128frame ---
export LOG_SUFFIX="vsibench_eval_128frame"
export POINTER_DIR_NAME="pointer_memory_128_frame"
bash scripts/evaluation/eval.sh

# --- Eval 5: pointer_memory_256frame ---
export LOG_SUFFIX="vsibench_eval_256frame"
export POINTER_DIR_NAME="pointer_memory_256_frame"
bash scripts/evaluation/eval.sh

# --- Eval 5: pointer_memory_512frame ---
export LOG_SUFFIX="vsibench_eval_512frame"
export POINTER_DIR_NAME="pointer_memory_512_frame"
bash scripts/evaluation/eval.sh

nvidia-smi
date
# squeue --job $SLURM_JOBID
echo "##### END #####"
