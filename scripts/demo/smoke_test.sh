#!/bin/bash
# Smoke test script for preprocessing
# Tests preprocessing on just the first scene
# Usage: 
#   ./smoke_test.sh [dataset] [gpu] [save_path]
#   dataset: scannetpp, arkit, or all (default: all)
#   gpu: CUDA device ID (default: 0)
#   save_path: output directory (default: auto)

DATASET=${1:-all}
CUDA_DEVICE=${2:-0}
SAVE_PATH=${3:-""}

# Set working directory
cd /cms_cvlab/home/chanyoung/yoonwoo/Point3R-LLM

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo "Using GPU: $CUDA_DEVICE"
echo "Python: $(which python3)"
echo ""

# Function to run ScanNet++ smoke test
run_scannetpp_test() {
    local save_path=${1:-"./output/scannetpp_smoke_test"}
    echo "=========================================="
    echo "ScanNet++ Preprocessing Smoke Test"
    echo "=========================================="
    echo "Save path: $save_path"
    echo ""
    
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 scripts/demo/preprocess_scannetpp_simple.py \
        --smoke-test \
        --save-path $save_path \
        --model-path "Qwen/Qwen3-VL-8B-Instruct"
    
    echo ""
    echo "ScanNet++ smoke test complete!"
    echo ""
}

# Function to run ARKitScenes smoke test
run_arkit_test() {
    local save_path=${1:-"./output/arkit_smoke_test"}
    echo "=========================================="
    echo "ARKitScenes Preprocessing Smoke Test"
    echo "=========================================="
    echo "Save path: $save_path"
    echo ""
    
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 scripts/demo/preprocess_arkit_simple.py \
        --smoke-test \
        --save-path $save_path \
        --model-path "Qwen/Qwen3-VL-8B-Instruct"
    
    echo ""
    echo "ARKitScenes smoke test complete!"
    echo ""
}

# Run tests based on dataset parameter
case $DATASET in
    scannetpp)
        run_scannetpp_test "$SAVE_PATH"
        ;;
    arkit)
        run_arkit_test "$SAVE_PATH"
        ;;
    all)
        echo "Running smoke tests for all datasets..."
        echo ""
        run_scannetpp_test "${SAVE_PATH:-./output/scannetpp_smoke_test}"
        run_arkit_test "${SAVE_PATH:-./output/arkit_smoke_test}"
        ;;
    *)
        echo "Error: Unknown dataset '$DATASET'"
        echo "Usage: $0 [scannetpp|arkit|all] [gpu_id] [save_path]"
        exit 1
        ;;
esac

echo "=========================================="
echo "All smoke tests complete!"
echo "=========================================="
