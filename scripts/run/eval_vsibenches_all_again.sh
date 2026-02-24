#!/bin/bash
#SBATCH --job-name="Full Evaluation of checkpoints on Revised VSIBench"
#SBATCH -o sbatch_log/eval_vsibench_reproduction.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

# Experiment: Evaluate VSIBench-Reproduction

source venv/bin/activate

# --- Shared config ---
export BENCHMARKS="vsibench_point3r"
export EVAL_MODEL_TYPE="point3r_llm_v2"

# Memory features
export MERGE_MEMORY_FEAT="True"
export MEMORY_FUSION_METHOD="add"

# Model 0
export MODEL_PATH="./outputs/@archive/vsibench_Qwen3VL_8b_memfeat_32frame"
export POINTER_FORMAT="image"
unset ROPE_MODE
export POINTER_DIR_NAME="pointer_memory_32frame_video_new"
export LOG_SUFFIX="vsibench_0_unordered_pointer"
bash scripts/evaluation/eval.sh

# Model 1
export MODEL_PATH="./outputs/vsibench_Qwen3VL_8b_memfeat_32frame_video_image_format"
export POINTER_FORMAT="image"
unset ROPE_MODE
export POINTER_DIR_NAME="pointer_memory_32frame_video_new"
export LOG_SUFFIX="vsibench_1_vsibench_Qwen3VL_8b_memfeat_32frame_video_image_format"
bash scripts/evaluation/eval.sh

# Model 2
export MODEL_PATH="./outputs/vsibench_image_with_timestep_rope"
export POINTER_FORMAT="image"
export ROPE_MODE="pointer_timestamp"
export POINTER_DIR_NAME="pointer_memory_32frame_video_new"
export LOG_SUFFIX="vsibench_2_vsibench_image_with_timestep_rope"
bash scripts/evaluation/eval.sh

# Model 3
export MODEL_PATH="./outputs/vsibench_Qwen3VL_8b_memfeat_32frame_video"
export POINTER_FORMAT="video"
unset ROPE_MODE
export POINTER_DIR_NAME="pointer_memory_32frame_video_new"
export LOG_SUFFIX="vsibench_3_vsibench_Qwen3VL_8b_memfeat_32frame_video"
bash scripts/evaluation/eval.sh

# Model 4
export MODEL_PATH="./outputs/vsibench_Qwen3VL_8b_memfeat_32frame_video_timestep_rope"
export POINTER_FORMAT="video"
export ROPE_MODE="pointer_timestamp"
export POINTER_DIR_NAME="pointer_memory_32frame_video_new"
export LOG_SUFFIX="vsibench_4_vsibench_Qwen3VL_8b_memfeat_32frame_video_timestep_rope"
bash scripts/evaluation/eval.sh

nvidia-smi
date
# squeue --job $SLURM_JOBID
echo "##### END #####"
