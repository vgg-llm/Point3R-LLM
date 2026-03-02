#!/bin/bash
#SBATCH --job-name=Point3R-LLM-8b_memfeat_appearance_order_experiment
#SBATCH -o sbatch_log/appr_order_exp-8b-memfeat-spar.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

# Experiment: 8B with memory features on Scan2Cap

source venv/bin/activate

export EXP_NAME="appr_order_spar_subset_formatted"
export MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
export DATASETS="spar_subset_point3r"

# Memory features
# export MERGE_MEMORY_FEAT="False"
# export TUNE_MEMORY_FEATURE_PROJECTOR="False"
# export TUNE_MEMORY_FEATURE_FUSION="False"
export MERGE_MEMORY_FEAT="True"
export MEMORY_FUSION_METHOD="add"

# # Smoke test overrides: use 8 GPUs (ZeRO-2 sharding to avoid OOM)
# export NPROC_PER_NODE=8
# export TOTAL_BATCH_SIZE=8
# export NUM_TRAIN_EPOCHS=1
# export SAVE_STEPS=999999
# export EXTRA_TRAIN_ARGS="--max_steps 200"

# pointer memory specification
export POINTER_DIR_NAME="pointer_memory_qwen3vl_8B_timstamp"

# Evaluation
export BENCHMARKS="spar_subset_point3r"
export EVAL_MODEL_TYPE="point3r_llm_v2"
export POINTER_DIR_NAME="pointer_memory_600patch"

# --- Train ---
bash scripts/train/train.sh



# --- Evaluate ---
export BENCHMARKS="vsibench_point3r,spar_subset_point3r"
export EVAL_MODEL_TYPE="point3r_llm_v2"
export MODEL_PATH="./outputs/${EXP_NAME}"
export NUM_PROCESSES=8
# export EVAL_LIMIT=100
bash scripts/evaluation/eval.sh

export BENCHMARKS="spar_subset_point3r"
# --- Evaluate: video-frame-pe ---
export ROPE_MODE="pointer_timestamp"
bash scripts/evaluation/eval.sh

# --- Evaluate: image-frame-pe ---
export POINTER_FORMAT="image"
bash scripts/evaluation/eval.sh

# --- Evaluate: video-frame-pe ---
export ROPE_MODE="pointer_timestamp"
bash scripts/evaluation/eval.sh


nvidia-smi
date
# squeue --job $SLURM_JOBID

echo "##### END #####"