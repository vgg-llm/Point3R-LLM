#!/bin/bash
#SBATCH --job-name=scan2cap_point3r_Qwen35VL_9b_memfeat
#SBATCH -o sbatch_log/scan2cap_point3r_Qwen35VL_9b_memfeat.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

# Experiment: 8B with memory features on Scan2Cap

source venv2/bin/activate

export EXP_NAME="scan2cap_point3r_Qwen35VL_9b_memfeat"
export MODEL_PATH="Qwen/Qwen3.5-9B"
export DATASETS="scan2cap_point3r"

# Memory features
export MERGE_MEMORY_FEAT="True"
export MEMORY_FUSION_METHOD="add"
export POINTER_DIR_NAME="pointer_memory_qwen35_9b"

# # Smoke test overrides: use 8 GPUs (ZeRO-2 sharding to avoid OOM)
# export NPROC_PER_NODE=8
# export TOTAL_BATCH_SIZE=8
# export NUM_TRAIN_EPOCHS=1
# export SAVE_STEPS=999999
# export EXTRA_TRAIN_ARGS="--max_steps 1"


# Evaluation
export BENCHMARKS="scan2cap_point3r"
export EVAL_MODEL_TYPE="point3r_llm_v3"

# --- Train ---
bash scripts/train/train.sh

export NUM_PROCESSES=8
# export EVAL_LIMIT=10
# --- Evaluate ---
export MODEL_PATH="./outputs/${EXP_NAME}"
bash scripts/evaluation/eval.sh

nvidia-smi
date
squeue --job $SLURM_JOBID
echo "##### END #####"
