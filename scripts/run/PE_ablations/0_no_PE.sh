#!/bin/bash
#SBATCH --job-name=scan2cap_point3r_0_no_PE
#SBATCH -o sbatch_log/scan2cap_point3r_0_no_PE.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

# Experiment: 8B with memory features on Scan2Cap

source venv/bin/activate

export EXP_NAME="scan2cap_point3r_0_no_PE"
export MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
export DATASETS="scan2cap_point3r"

# Memory features
export MERGE_MEMORY_FEAT="False"
export TUNE_MEMORY_FEATURE_PROJECTOR="False"
export TUNE_MEMORY_FEATURE_FUSION="False"

# Evaluation
export BENCHMARKS="scan2cap_point3r"
export EVAL_MODEL_TYPE="point3r_llm_v2"

# --- Train ---
bash scripts/train/train.sh

# --- Evaluate ---
export MODEL_PATH="./outputs/${EXP_NAME}"
bash scripts/evaluation/eval.sh

nvidia-smi
date
squeue --job $SLURM_JOBID
echo "##### END #####"
