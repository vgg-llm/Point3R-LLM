#!/usr/bin/env bash
# Ego3D-Bench evaluation. Usage:
#   bash scripts/run/ego3d_eval.sh baseline          [limit]
#   bash scripts/run/ego3d_eval.sh pointer           [limit]
#   bash scripts/run/ego3d_eval.sh pointer_think     [limit]
#   bash scripts/run/ego3d_eval.sh images_finetuned  [limit]
#
# 2x2 design {stock weights, finetuned weights} x {images, pointer tokens}, all on
# the official think protocol so the pointer-vs-images comparison holds weights fixed:
#   images,    stock weights     -> `baseline` (MODEL_PATH left unset)
#   images,    finetuned weights -> `images_finetuned`
#   pointer,   stock weights     -> not supported (finetuning is what adds pointer support)
#   pointer,   finetuned weights -> `pointer_think`
# `images_finetuned` runs the same `ego3d_baseline` task (real images, add_frame_index)
# as `baseline`, but sets merge_memory_feat/memory_fusion_method to match the finetuned
# checkpoint's own training config so config_validation.py's critical-param check passes.
# It still feeds images, not pointer memory: use_pointer_memory/use_preprocessed_input
# stay False, so the memory_feat fusion path is configured into the model but never
# exercised at runtime (see src/lmms_eval/models/point3r_llm_v2.py:655-669).
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
# WEIGHTS: `pointer`/`pointer_think` default to the finetuned Point3R checkpoint,
# `baseline` defaults to stock Qwen/Qwen3-VL-4B-Instruct. Comparing those two
# defaults varies the WEIGHTS as well as the visual substrate. For a controlled
# substrate comparison, hold the weights fixed by pointing baseline at the same
# checkpoint:
#   MODEL_PATH=./outputs/scan2cap_point3r_Qwen3VL_memfeat_lambda0.5 \
#       bash scripts/run/ego3d_eval.sh baseline
# Every mode honors MODEL_PATH.
set -e
# Without pipefail, `set -e` cannot see a crashed eval through the `| tee` below and
# the script would exit 0 on a failed run.
set -o pipefail

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

pointer_model_path=${MODEL_PATH:-./outputs/scan2cap_point3r_Qwen3VL_memfeat_lambda0.5}
pointer_model_args="pretrained=$pointer_model_path,use_flash_attention_2=true,max_length=12800,\
use_pointer_memory=True,use_preprocessed_input=True,merge_memory_feat=True,\
memory_fusion_method=add,add_frame_id=true,base_dir=data/media"

main_process_port=29511
if [ "$mode" = "baseline" ]; then
    task=ego3d_baseline
    model_path=${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}
    model_args="pretrained=$model_path,use_flash_attention_2=true,max_length=12800,\
use_pointer_memory=False,use_preprocessed_input=False,add_frame_index=true"
elif [ "$mode" = "pointer_think" ]; then
    task=ego3d_point3r_think
    model_args="$pointer_model_args"
    main_process_port=29513
elif [ "$mode" = "images_finetuned" ]; then
    task=ego3d_baseline
    model_path=${MODEL_PATH:-./outputs/scan2cap_point3r_Qwen3VL_memfeat_lambda0.5}
    model_args="pretrained=$model_path,use_flash_attention_2=true,max_length=12800,\
use_pointer_memory=False,use_preprocessed_input=False,add_frame_index=true,\
merge_memory_feat=True,memory_fusion_method=add"
    main_process_port=29515
else
    task=ego3d_point3r
    model_args="$pointer_model_args"
fi

# Auto-recover from stale/orphaned processes holding a mode's default port
# (this accelerate version, 1.4.0, has no `--main_process_port 0` auto-select --
# it only raises ConnectionError on a busy port -- so we probe and bump here).
is_port_free() {
    ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}
requested_port="$main_process_port"
while ! is_port_free "$main_process_port"; do
    main_process_port=$((main_process_port + 1))
done
if [ "$main_process_port" != "$requested_port" ]; then
    echo "Port $requested_port is in use; using $main_process_port instead."
fi
echo "Using main_process_port=$main_process_port"

extra=""
if [ -n "$limit" ]; then extra="--limit $limit"; fi

"$ACCELERATE_BIN" launch --num_processes="$num_processes" --main_process_port "$main_process_port" -m lmms_eval \
    --model point3r_llm_v2 \
    --model_args "$model_args" \
    --tasks "$task" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "ego3d_${mode}" \
    --output_path "$output_path" \
    $extra \
    2>&1 | tee "${output_path}/eval.log"
