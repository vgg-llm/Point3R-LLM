#!/usr/bin/env bash
# Ego3D-Bench evaluation. Usage:
#   bash scripts/run/ego3d_eval.sh baseline [limit]
#   bash scripts/run/ego3d_eval.sh pointer  [limit]
#
# Always invoked via the vgllm conda env's absolute interpreter path
# (transformers 4.57.6, matches the Qwen3-VL dev branch). Do not rely on
# whatever conda env happens to be active when this script is run.
#
# Single-GPU pin for single-process runs:
#   src/lmms_eval/models/point3r_llm_v2.py:189-197 only pins the model to
#   cuda:{local_process_index} when accelerator.num_processes > 1. With
#   num_processes == 1 it keeps device_map="auto", which shards the model
#   across every visible GPU while inputs are placed on cuda:0 -- this
#   crashes with "Expected all tensors to be on the same device" whenever
#   more than one GPU is visible. We adapt this script to that wrapper
#   behavior (rather than changing the wrapper) by pinning
#   CUDA_VISIBLE_DEVICES=0 below whenever num_processes == 1 and the caller
#   hasn't already set CUDA_VISIBLE_DEVICES themselves.
set -e

ACCELERATE_BIN=${ACCELERATE_BIN:-/home/gwakcy/miniconda3/envs/vgllm/bin/accelerate}

export LMMS_EVAL_LAUNCHER="accelerate"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
export NCCL_NVLS_ENABLE=0

mode=${1:-pointer}
limit=${2:-}
num_processes=${NUM_PROCESSES:-8}
if [ "$num_processes" -eq 1 ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi
output_path=logs/$(TZ="Asia/Seoul" date "+%Y%m%d")/ego3d_${mode}
mkdir -p "$output_path"

if [ "$mode" = "baseline" ]; then
    task=ego3d_baseline
    model_path=${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}
    model_args="pretrained=$model_path,use_flash_attention_2=true,max_length=12800,\
use_pointer_memory=False,use_preprocessed_input=False,add_frame_index=true"
else
    task=ego3d_point3r
    model_path=${MODEL_PATH:-./outputs/scan2cap_point3r_Qwen3VL_memfeat_lambda0.5}
    model_args="pretrained=$model_path,use_flash_attention_2=true,max_length=12800,\
use_pointer_memory=True,use_preprocessed_input=True,merge_memory_feat=True,\
memory_fusion_method=add,add_frame_id=true,base_dir=data/media"
fi

extra=""
if [ -n "$limit" ]; then extra="--limit $limit"; fi

"$ACCELERATE_BIN" launch --num_processes="$num_processes" --main_process_port 29511 -m lmms_eval \
    --model point3r_llm_v2 \
    --model_args "$model_args" \
    --tasks "$task" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "ego3d_${mode}" \
    --output_path "$output_path" \
    $extra \
    2>&1 | tee "${output_path}/eval.log"
