#!/bin/bash
#SBATCH --job-name=scan2cap_Qwen35VL_4b_vanilla
#SBATCH -o sbatch_log/scan2cap_Qwen35VL_4b_vanilla.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

# Experiment: 8B with memory features on Scan2Cap

# source venv/bin/activate  # Using conda env instead
export EXP_NAME="scan2cap_Qwen35VL_4b_vanilla"
export MODEL_PATH="Qwen/Qwen3.5-4B"
export DATASETS="scan2cap"

# Memory features
export MERGE_MEMORY_FEAT="False"
export TUNE_MEMORY_FEATURE_PROJECTOR="False"
export TUNE_MEMORY_FEATURE_FUSION="False"

# Smoke test overrides: use 8 GPUs (ZeRO-2 sharding to avoid OOM)
export NPROC_PER_NODE=8
export TOTAL_BATCH_SIZE=8
export NUM_TRAIN_EPOCHS=1
export SAVE_STEPS=999999
export EXTRA_TRAIN_ARGS="--max_steps 1"


# Evaluation
export BENCHMARKS="scan2cap"
export EVAL_MODEL_TYPE="point3r_llm_v3"

# --- Train ---
bash scripts/train/train.sh

export NUM_PROCESSES=8
export EVAL_LIMIT=10
# --- Evaluate ---
export MODEL_PATH="./outputs/${EXP_NAME}"
bash scripts/evaluation/eval.sh

# nvidia-smi
# date
# squeue --job $SLURM_JOBID
echo "##### END #####"
