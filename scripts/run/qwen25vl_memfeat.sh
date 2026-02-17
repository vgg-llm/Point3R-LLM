#!/bin/bash
#SBATCH --job-name=Point3R-LLM-qwen25vl_memfeat
#SBATCH -o sbatch_log/qwen25vl-memfeat.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

# Experiment: Qwen2.5-VL 3B with memory features on Scan2Cap

source venv/bin/activate

export EXP_NAME="scan2cap_point3r_Qwen25VL_memfeat"
export MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
export DATASETS="scan2cap_point3r"

# Memory features
export MERGE_MEMORY_FEAT="True"
export MEMORY_FUSION_METHOD="add"

# Evaluation (Qwen2.5 uses point3r_llm, not point3r_llm_v2)
export BENCHMARKS="scan2cap_point3r"
export EVAL_MODEL_TYPE="point3r_llm"

# --- Train ---
bash scripts/train/train.sh

# --- Evaluate ---
export MODEL_PATH="./outputs/${EXP_NAME}"
bash scripts/evaluation/eval.sh

nvidia-smi
date
squeue --job $SLURM_JOBID
echo "##### END #####"
