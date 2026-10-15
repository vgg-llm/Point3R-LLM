#!/bin/bash
#SBATCH --job-name="Balanced VSI-bench evaluation on pointer_memory_600patch (Just to reproduce the results) 4GPU"
#SBATCH -o sbatch_log/vsibench_balanced_memfeat_600patch_4gpu.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --export=ALL,MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/chanyoung/Cog3DMap
#SBATCH --wckey=project-short-name:others

# Experiment: 8B with memory features on VSIbench-balanced

source venv/bin/activate

export EXP_NAME="vsibench_balanced_memfeat_600patch"
# export MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
# export DATASETS="vsibench_balanced_point3r"

# Memory features
export USE_POINTER_MEMORY="False"
export USE_PREPROCESSED_INPUT="False"
export MERGE_MEMORY_FEAT="True"
export MEMORY_FUSION_METHOD="add"

# Evaluation
export BENCHMARKS="vsibench_point3r"
export EVAL_MODEL_TYPE="point3r_llm_v2"
export POINTER_DIR_NAME="pointer_memory_600patch"

# --- Train ---
# bash scripts/train/train.sh

# --- Evaluate ---
export MODEL_PATH="./outputs/${EXP_NAME}"
bash scripts/evaluation/eval.sh

nvidia-smi
date
squeue --job $SLURM_JOBID
echo "##### END #####"
