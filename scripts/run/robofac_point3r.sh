#!/bin/bash
#SBATCH --job-name="RoboFAC Point3R Training and Evaluation"
#SBATCH -o sbatch_log/robofac_point3r.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

# Experiment: RoboFAC Point3R (Robotic Failure Analysis and Correction)
# Uses pointer memory features from preprocessed video.

# source venv/bin/activate  # using conda env instead

export EXP_NAME="robofac_point3r_Qwen3VL_4b"
export MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
export DATASETS="robofac_point3r"

# Enable pointer memory for Point3R
export MERGE_MEMORY_FEAT="True"
export USE_POINTER_MEMORY="True"
export USE_PREPROCESSED_INPUT="True"

# # Smoke test overrides
# export NPROC_PER_NODE=8
# export TOTAL_BATCH_SIZE=8
# export NUM_TRAIN_EPOCHS=1
# export SAVE_STEPS=999999
# export EXTRA_TRAIN_ARGS="--max_steps 200"

export MAX_PIXELS=196608
export MIN_PIXELS=196608
export BASE_INTERVAL=1
export VIDEO_MAX_FRAMES=32
export VIDEO_MIN_FRAMES=4

export SAVE_STEPS=200

# --- Train ---
# bash scripts/train/train.sh

# --- vLLM scoring server (GPU 7) ---
# Serves LLM for open-ended question scoring during evaluation.
# VLLM_GPU=7
# VLLM_PORT=8000
# VLLM_MODEL="Qwen/Qwen3-4B-Instruct-2507"  # text-only model (no flash_attn needed)

# echo "Starting vLLM server on GPU ${VLLM_GPU}, port ${VLLM_PORT}..."
# CUDA_VISIBLE_DEVICES=${VLLM_GPU} python -m vllm.entrypoints.openai.api_server \
#     --model "${VLLM_MODEL}" \
#     --port ${VLLM_PORT} \
#     --gpu-memory-utilization 0.9 \
#     --max-model-len 4096 \
#     &
# VLLM_PID=$!

# # Wait for server to be ready
# echo "Waiting for vLLM server (PID ${VLLM_PID}) to be ready..."
# for i in $(seq 1 120); do
#     if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
#         echo "vLLM server ready after ${i}s"
#         break
#     fi
#     if ! kill -0 ${VLLM_PID} 2>/dev/null; then
#         echo "ERROR: vLLM server process died" >&2
#         exit 1
#     fi
#     sleep 1
# done
# if ! curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
#     echo "ERROR: vLLM server failed to start within 120s" >&2
#     kill ${VLLM_PID} 2>/dev/null
#     exit 1
# fi

# # Cleanup vLLM on exit (even on error/interrupt)
# trap "echo 'Stopping vLLM server...'; kill ${VLLM_PID} 2>/dev/null; wait ${VLLM_PID} 2>/dev/null" EXIT

# --- Evaluate (GPUs 0-6, GPU 7 reserved for vLLM) ---
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

export MODEL_PATH="./outputs/${EXP_NAME}"

export BENCHMARKS="robofac_point3r"
export EVAL_MODEL_TYPE="point3r_llm_v2"
# export OPENAI_API_KEY="local"
# export ROBOFAC_LLM_LOCAL="Qwen/Qwen3-4B-Instruct-2507"
# export OPENAI_API_KEY="sk-..."           # your real OpenAI API key
# export ROBOFAC_LLM_MODEL="gpt-4o"       # or any OpenAI model
# Do NOT set ROBOFAC_LLM_LOCAL
export NUM_PROCESSES=8
# export EVAL_LIMIT=500
export MAX_LENGTH=32768
bash scripts/evaluation/eval.sh

# nvidia-smi
# date

echo "##### END #####"
