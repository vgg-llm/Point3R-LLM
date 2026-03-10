#!/bin/bash
# Smoke test for 4b_memfeat_appr_order experiment
# Runs 1 training step on 1 GPU with minimal data to verify the pipeline.
# Usage:
#   bash scripts/run/smoke_test_4b_memfeat_appr_order.sh          # actually run 1 step
#   DRY_RUN=1 bash scripts/run/smoke_test_8b_memfeat_appr_order.sh  # just print commands

# source venv/bin/activate

export EXP_NAME="smoke_test_scanrefer"
export MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
export DATASETS="scanrefer_point3r"

# Memory features (same as production)
export MERGE_MEMORY_FEAT="False"
export TUNE_MEMORY_FEATURE_PROJECTOR="False"
export TUNE_MEMORY_FEATURE_FUSION="False"

# Smoke test overrides: use 8 GPUs (ZeRO-2 sharding to avoid OOM)
export NPROC_PER_NODE=8
export TOTAL_BATCH_SIZE=8
export NUM_TRAIN_EPOCHS=1
export SAVE_STEPS=999999
export EXTRA_TRAIN_ARGS="--max_steps 1"

# --- Train ---
bash scripts/train/train.sh

# --- Evaluate ---
export BENCHMARKS="scanrefer_point3r"
export EVAL_MODEL_TYPE="point3r_llm_v2"
export MODEL_PATH="./outputs/${EXP_NAME}"
export NUM_PROCESSES=8
export EVAL_LIMIT=10
bash scripts/evaluation/eval.sh

echo "##### SMOKE TEST DONE #####"
