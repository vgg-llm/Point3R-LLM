#!/bin/bash
# VSIBench + VSTIBench Point3R Evaluation Script for Qwen3-VL-8B

set -e
export LMMS_EVAL_LAUNCHER="accelerate"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"

export NCCL_NVLS_ENABLE=0

# Evaluation configuration
# choices: [vsibench_point3r, vstibench_point3r]
benchmarks="vsibench_point3r"
output_path=logs/vsibench_$(TZ="Asia/Shanghai" date "+%Y%m%d_%H%M%S")
model_path=./outputs/vsibench_Qwen3VL_8b_memfeat

model_args_str="pretrained=$model_path,use_flash_attention_2=true,max_num_frames=32,max_length=12800,use_pointer_memory=True,use_preprocessed_input=True,\
merge_memory_feat=True,memory_fusion_method=add"

mkdir -p $output_path

# Save evaluation script for reproducibility
cp "${BASH_SOURCE[0]}" "${output_path}/eval_script.sh"

echo "============================================="
echo "Evaluating VSIBench Point3R"
echo "Model: $model_path"
echo "Benchmarks: $benchmarks"
echo "Output: $output_path"
echo "============================================="

accelerate launch --num_processes=8 --main_process_port 29501 -m lmms_eval \
    --model point3r_llm_v2 \
    --model_args "$model_args_str" \
    --tasks ${benchmarks} \
    --batch_size 1 \
    --log_samples_suffix vsibench_eval \
    --log_samples \
    --output_path $output_path \
    2>&1 | tee ${output_path}/eval.log

echo "============================================="
echo "Evaluation completed!"
echo "Results saved to: $output_path"
echo "============================================="
