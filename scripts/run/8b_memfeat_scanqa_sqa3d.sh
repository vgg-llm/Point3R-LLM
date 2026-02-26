#!/bin/bash
#SBATCH --job-name="Point3R-Qwen train on scanqa&sqa3d_revised"
#SBATCH -o sbatch_log/scanqa_sqa3d_point3r_revised_prompt.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

# Experiment: 8B with memory features on Scan2Cap

source venv/bin/activate

export EXP_NAME="scanqa_sqa3d_point3r_revised_prompt"
export MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
export DATASETS="scanqa_point3r,sqa3d_point3r"

# Memory features
export MERGE_MEMORY_FEAT="True"
export MEMORY_FUSION_METHOD="add"

# Evaluation
export BENCHMARKS="scanqa_point3r,sqa3d_point3r"
export EVAL_MODEL_TYPE="point3r_llm_v2"

# --- Train ---
# bash scripts/train/train.sh

# --- Evaluate ---
export MODEL_PATH="./outputs/${EXP_NAME}"
bash scripts/evaluation/eval.sh

nvidia-smi
date
squeue --job $SLURM_JOBID
echo "##### END #####"
