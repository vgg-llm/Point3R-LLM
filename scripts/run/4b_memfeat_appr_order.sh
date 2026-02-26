#!/bin/bash
#SBATCH --job-name=Point3R-LLM-4b_memfeat_appearance_order_experiment
#SBATCH -o sbatch_log/appr_order_exp-4b-memfeat.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

# Experiment: 4B with memory features on Scan2Cap

# source venv/bin/activate

export EXP_NAME="appr_order_exp"
export MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
export DATASETS="spar_subset_point3r"

# Memory features
export MERGE_MEMORY_FEAT="False"
export TUNE_MEMORY_FEATURE_PROJECTOR="False"
export TUNE_MEMORY_FEATURE_FUSION="False"
# Evaluation
export BENCHMARKS="spar_subset_point3r"
export EVAL_MODEL_TYPE="point3r_llm_v2"

# --- Train ---
# bash scripts/train/train.sh

# --- Evaluate ---
bash scripts/evaluation/eval.sh

nvidia-smi
date
# squeue --job $SLURM_JOBID
echo "##### END #####"
