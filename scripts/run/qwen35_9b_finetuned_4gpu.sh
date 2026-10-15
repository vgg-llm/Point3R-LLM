#!/bin/bash
#SBATCH --job-name="Scan2cap Training and evaluation with naive Qwen3.5-9B for NeurIPS rebuttal"
#SBATCH -o sbatch_log/scan2cap_Qwen35VL_9b_finetuned.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --export=ALL,MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/chanyoung/Cog3DMap
#SBATCH --wckey=project-short-name:others

# Experiment: Scan2cap Training and evaluation with naive Qwen3.5

source venv2/bin/activate  # Using conda env instead
export EXP_NAME="scan2cap_Qwen35VL_9b_finetuned"
export MODEL_PATH="Qwen/Qwen3.5-9B"
export DATASETS="scan2cap"

export NUM_PROCESSES=4

# Memory features
export MERGE_MEMORY_FEAT="False"
export TUNE_MEMORY_FEATURE_PROJECTOR="False"
export TUNE_MEMORY_FEATURE_FUSION="False"

# Smoke test overrides: use 8 GPUs (ZeRO-2 sharding to avoid OOM)
export NPROC_PER_NODE=4
export TOTAL_BATCH_SIZE=8
export NUM_TRAIN_EPOCHS=1
# export SAVE_STEPS=999999
export EXTRA_TRAIN_ARGS="--max_steps 1000"


# Evaluation
export BENCHMARKS="scan2cap"
export EVAL_MODEL_TYPE="point3r_llm_v3"

# --- Train ---
bash scripts/train/train.sh

export NUM_PROCESSES=4
# export EVAL_LIMIT=10
# --- Evaluate ---
export MODEL_PATH="./outputs/${EXP_NAME}"
bash scripts/evaluation/eval.sh

nvidia-smi
date
squeue --job $SLURM_JOBID
echo "##### END #####"
