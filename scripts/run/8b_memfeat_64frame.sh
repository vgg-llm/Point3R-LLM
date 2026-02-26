#!/bin/bash
#SBATCH --job-name="Scan2cap Training and evaluation on pointer_memory_compact"
#SBATCH -o sbatch_log/scan2cap_point3r_Qwen3VL_8b_memfeat_64frame.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

# Experiment: 8B with memory features on Scan2Cap

source venv/bin/activate

export EXP_NAME="scan2cap_point3r_Qwen3VL_8b_memfeat_64frame"
export MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
export DATASETS="scan2cap_point3r"

# Memory features
export MERGE_MEMORY_FEAT="True"
export MEMORY_FUSION_METHOD="add"

# Evaluation
export BENCHMARKS="scan2cap_point3r"
export EVAL_MODEL_TYPE="point3r_llm_v2"
export POINTER_DIR_NAME="pointer_memory_64_frame"

# --- Train ---
bash scripts/train/train.sh

# --- Evaluate ---
export MODEL_PATH="./outputs/${EXP_NAME}"
bash scripts/evaluation/eval.sh

nvidia-smi
date
squeue --job $SLURM_JOBID
echo "##### END #####"
