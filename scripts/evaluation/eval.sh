#!/bin/bash
# =============================================================================
# Unified Point3R-LLM Evaluation Script
# =============================================================================
# Configure via environment variables before calling this script.
# Only MODEL_PATH is required; all others have sensible defaults.
#
# Usage:
#   export MODEL_PATH="./outputs/my_experiment"
#   export BENCHMARKS="scan2cap_point3r"
#   bash scripts/evaluation/eval.sh
#
# For a full list of configurable parameters, see the "Defaults" section below.
# =============================================================================

set -eo pipefail

# =============================================================================
# Defaults
# =============================================================================

# --- Required ---
if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH must be set." >&2
    exit 1
fi

# --- Eval Config ---
BENCHMARKS="${BENCHMARKS:-scan2cap_point3r}"
EVAL_MODEL_TYPE="${EVAL_MODEL_TYPE:-point3r_llm_v2}"
LOG_SUFFIX="${LOG_SUFFIX:-eval}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

# --- Model Args ---
MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-32}"
MAX_LENGTH="${MAX_LENGTH:-12800}"
USE_POINTER_MEMORY="${USE_POINTER_MEMORY:-True}"
USE_PREPROCESSED_INPUT="${USE_PREPROCESSED_INPUT:-True}"
MERGE_MEMORY_FEAT="${MERGE_MEMORY_FEAT:-True}"
MEMORY_FUSION_METHOD="${MEMORY_FUSION_METHOD:-add}"
ROPE_MODE="${ROPE_MODE:-}"
ROPE_POSITION_RANGE="${ROPE_POSITION_RANGE:-}"
POINTER_FORMAT="${POINTER_FORMAT:-}"
POINTER_DIR_NAME="${POINTER_DIR_NAME:-}"
ADD_FRAME_ID="${ADD_FRAME_ID:-}"
USE_POINTER_POSITION_ENCODING="${USE_POINTER_POSITION_ENCODING:-}"

# --- Extra model args passthrough ---
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-}"

# --- Sample limit (optional, for smoke tests) ---
EVAL_LIMIT="${EVAL_LIMIT:-}"

# --- Output path ---
if [[ -z "$OUTPUT_PATH" ]]; then
    MODEL_BASENAME="$(basename "$MODEL_PATH")"
    OUTPUT_PATH="logs/${MODEL_BASENAME}_$(TZ="Asia/Shanghai" date "+%Y%m%d_%H%M%S")"
fi

# =============================================================================
# Environment
# =============================================================================
export LMMS_EVAL_LAUNCHER="accelerate"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
export NCCL_NVLS_ENABLE=0

# =============================================================================
# Paths
# =============================================================================
mkdir -p "$OUTPUT_PATH"

# =============================================================================
# Build model_args_str
# =============================================================================
model_args_str="pretrained=${MODEL_PATH},use_flash_attention_2=true"
model_args_str="${model_args_str},max_num_frames=${MAX_NUM_FRAMES}"
model_args_str="${model_args_str},max_length=${MAX_LENGTH}"
model_args_str="${model_args_str},use_pointer_memory=${USE_POINTER_MEMORY}"
model_args_str="${model_args_str},use_preprocessed_input=${USE_PREPROCESSED_INPUT}"
model_args_str="${model_args_str},merge_memory_feat=${MERGE_MEMORY_FEAT}"
model_args_str="${model_args_str},memory_fusion_method=${MEMORY_FUSION_METHOD}"

# --- Conditional args ---
[[ -n "$ROPE_MODE" ]]           && model_args_str="${model_args_str},rope_mode=${ROPE_MODE}"
[[ -n "$ROPE_POSITION_RANGE" ]] && model_args_str="${model_args_str},rope_position_range=${ROPE_POSITION_RANGE}"
[[ -n "$POINTER_FORMAT" ]]      && model_args_str="${model_args_str},pointer_format=${POINTER_FORMAT}"
[[ -n "$POINTER_DIR_NAME" ]]              && model_args_str="${model_args_str},pointer_dir_name=${POINTER_DIR_NAME}"
[[ -n "$ADD_FRAME_ID" ]]                  && model_args_str="${model_args_str},add_frame_id=${ADD_FRAME_ID}"
[[ -n "$USE_POINTER_POSITION_ENCODING" ]] && model_args_str="${model_args_str},use_pointer_position_encoding=${USE_POINTER_POSITION_ENCODING}"

# --- Handle scanrefer special case ---
if [[ "$BENCHMARKS" == *"scanrefer"* ]]; then
    model_args_str="${model_args_str},add_frame_index=true"
fi

# --- Extra model args passthrough ---
if [[ -n "$EXTRA_MODEL_ARGS" ]]; then
    model_args_str="${model_args_str},${EXTRA_MODEL_ARGS}"
fi

# =============================================================================
# Preflight: verify each task's dataset resolves before allocating GPUs
# =============================================================================
# A missing local dataset dir otherwise surfaces deep inside Task.download() on
# rank 0 only, after every rank has spun up; the surviving ranks then block on a
# collective and the job hangs until the scheduler kills it.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
preflight_failed=0
for task in ${BENCHMARKS//,/ }; do
    task_yaml="${REPO_ROOT}/src/lmms_eval/tasks/${task}/${task}.yaml"
    if [[ ! -f "$task_yaml" ]]; then
        echo "WARNING: no task yaml at src/lmms_eval/tasks/${task}/${task}.yaml (group task?); skipping preflight for '${task}'." >&2
        continue
    fi
    # `|| true`: grep exits 1 when absent, which pipefail + set -e would turn into
    # a silent script abort before the clearer error below can be printed.
    dataset_path="$({ grep -m1 '^dataset_path:' "$task_yaml" || true; } | sed 's/^dataset_path:[[:space:]]*//' | tr -d '"'"'"' \r')"
    if [[ -z "$dataset_path" ]]; then
        echo "ERROR: task '${task}' has no dataset_path in ${task_yaml}." >&2
        preflight_failed=1
        continue
    fi
    # Local path (relative to repo root or absolute) must exist on disk.
    if [[ "$dataset_path" == /* || "$dataset_path" == .* || "$dataset_path" == data/* ]]; then
        abs_path="$dataset_path"
        [[ "$abs_path" != /* ]] && abs_path="${REPO_ROOT}/${dataset_path}"
        if [[ ! -e "$abs_path" ]]; then
            echo "ERROR: task '${task}' expects local dataset at '${dataset_path}' but it does not exist (${abs_path})." >&2
            preflight_failed=1
        else
            echo "[preflight] ${task}: local dataset OK (${dataset_path})"
        fi
    else
        echo "[preflight] ${task}: hub dataset '${dataset_path}' (not checked offline)"
    fi
done
if [[ "$preflight_failed" -ne 0 ]]; then
    echo "ERROR: dataset preflight failed; aborting before launch." >&2
    exit 1
fi

# =============================================================================
# Save for Reproducibility
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "${SCRIPT_DIR}/eval.sh" "${OUTPUT_PATH}/eval.sh"

cat > "${OUTPUT_PATH}/eval_resolved.sh" << RESOLVED_EOF
#!/bin/bash
# Auto-generated for reproducibility
# Model: ${MODEL_PATH}
# Benchmarks: ${BENCHMARKS}
# Generated: $(date "+%Y-%m-%d %H:%M:%S")
#
# To reproduce: bash ${OUTPUT_PATH}/eval_resolved.sh

export MODEL_PATH="${MODEL_PATH}"
export BENCHMARKS="${BENCHMARKS}"
export EVAL_MODEL_TYPE="${EVAL_MODEL_TYPE}"
export OUTPUT_PATH="${OUTPUT_PATH}"
export LOG_SUFFIX="${LOG_SUFFIX}"
export NUM_PROCESSES="${NUM_PROCESSES}"
export MAX_NUM_FRAMES="${MAX_NUM_FRAMES}"
export MAX_LENGTH="${MAX_LENGTH}"
export USE_POINTER_MEMORY="${USE_POINTER_MEMORY}"
export USE_PREPROCESSED_INPUT="${USE_PREPROCESSED_INPUT}"
export MERGE_MEMORY_FEAT="${MERGE_MEMORY_FEAT}"
export MEMORY_FUSION_METHOD="${MEMORY_FUSION_METHOD}"
export ROPE_MODE="${ROPE_MODE}"
export ROPE_POSITION_RANGE="${ROPE_POSITION_RANGE}"
export POINTER_FORMAT="${POINTER_FORMAT}"
export POINTER_DIR_NAME="${POINTER_DIR_NAME}"
export ADD_FRAME_ID="${ADD_FRAME_ID}"
export USE_POINTER_POSITION_ENCODING="${USE_POINTER_POSITION_ENCODING}"
export EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS}"

bash scripts/evaluation/eval.sh
RESOLVED_EOF
chmod +x "${OUTPUT_PATH}/eval_resolved.sh"

# =============================================================================
# Execute
# =============================================================================
echo "============================================="
echo "Evaluating: ${BENCHMARKS}"
echo "Model: ${MODEL_PATH}"
echo "Model type: ${EVAL_MODEL_TYPE}"
echo "Output: ${OUTPUT_PATH}"
echo "Model args: ${model_args_str}"
echo "============================================="

if [[ "${DRY_RUN:-}" == "1" ]]; then
    echo "[DRY RUN] Command:"
    echo "accelerate launch --num_processes=${NUM_PROCESSES} --main_process_port 29501 -m lmms_eval \\"
    echo "    --model $EVAL_MODEL_TYPE \\"
    echo "    --model_args \"$model_args_str\" \\"
    echo "    --tasks ${BENCHMARKS} \\"
    echo "    --batch_size 1 \\"
    echo "    --log_samples_suffix $LOG_SUFFIX \\"
    echo "    --log_samples \\"
    echo "    --output_path $OUTPUT_PATH"
else
    accelerate launch --num_processes=${NUM_PROCESSES} --main_process_port 29501 -m lmms_eval \
        --model "$EVAL_MODEL_TYPE" \
        --model_args "$model_args_str" \
        --tasks ${BENCHMARKS} \
        --batch_size 1 \
        --log_samples_suffix "$LOG_SUFFIX" \
        --log_samples \
        --output_path "$OUTPUT_PATH" \
        ${EVAL_LIMIT:+--limit "$EVAL_LIMIT"} \
        2>&1 | tee "${OUTPUT_PATH}/eval.log"
    echo "============================================="
    echo "Evaluation completed!"
    echo "Results saved to: ${OUTPUT_PATH}"
    echo "============================================="
fi
