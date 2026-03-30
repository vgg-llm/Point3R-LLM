#!/bin/bash
# Visualize 3D attention on Point3R point clouds.
# Edit the variables below, then run:
#   bash scripts/demo/run_visualize_attention_3d.sh

set -euo pipefail

# ── Editable parameters ─────────────────────────────────────────────────
# Input (pick one; leave the others empty)
INPUT_IMAGES_DIR="data/media/scannet/posed_images/scene0025_00"
INPUT_VIDEO=""
POINTER_DATA_PATH=""

# Query
QUERY="Carefully watch the video and describe the object located at [2.26, -0.6, 2.45]."
QUERY="Carefully watch the video and describe the object located at [-0.59, 0.02, 1.57]."
QUERY="Carefully watch the video and describe the object located at [-2.51, -0.66, 2.12]."
QUERY="Carefully watch the video and describe the object located at [-0.98, -0.7, 2.6]."
QUERY="Carefully watch the video and describe the object located at [1.44, 0.64, 1.06]."
QUERY="Carefully watch the video and describe the object located at [0.48, -0.18, 2.69]."
# QUERY="Is there a fan?"


# Model

MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
MODEL_PATH="outputs/scan2cap_point3r_Qwen3VL_memfeat_lambda0.75"
MODEL_PATH="outputs/robofac_point3r_Qwen3VL_4b"

# Options
SCANNET=true              # true = only load *.jpg images (ScanNet format)
FAST=false                # true = downsampled points, faster startup
SAMPLE_CT=32             # number of frames to sample
MAX_NEW_TOKENS=128
LAYER_INDICES=""         # e.g. "-8 -7 -6 -5 -4 -3 -2 -1" (empty = all layers)
POINT_STRIDE=1          # spatial downsampling (1=full, 4=16x fewer)
OUTPUT_DIR="data/demo_data/visualization"
GPU=0
# ─────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Build argument list
ARGS=(
    --model_path "$MODEL_PATH"
    --query "$QUERY"
    --sample_ct "$SAMPLE_CT"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --point_stride "$POINT_STRIDE"
    --output_dir "$OUTPUT_DIR"
)

[[ -n "$INPUT_IMAGES_DIR" ]] && ARGS+=(--input_images_dir "$INPUT_IMAGES_DIR")
[[ -n "$INPUT_VIDEO" ]]      && ARGS+=(--input_video "$INPUT_VIDEO")
[[ -n "$POINTER_DATA_PATH" ]] && ARGS+=(--pointer_data_path "$POINTER_DATA_PATH")
[[ "$SCANNET" == "true" ]]   && ARGS+=(--scannet)
[[ "$FAST" == "true" ]]      && ARGS+=(--fast)
[[ -n "$LAYER_INDICES" ]]    && ARGS+=(--layer_indices $LAYER_INDICES)

CUDA_VISIBLE_DEVICES=$GPU python scripts/demo/visualize_attention_3d.py "${ARGS[@]}" "$@"
