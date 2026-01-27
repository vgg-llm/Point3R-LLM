set -e
export LMMS_EVAL_LAUNCHER="accelerate"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"

export NCCL_NVLS_ENABLE=0
benchmark=vsibench # choices: [vsibench, cvbench, blink_spatial]
output_path=logs/$(TZ="Asia/Shanghai" date "+%Y%m%d")
model_path=Qwen/Qwen3-VL-4B-Instruct

accelerate launch --num_processes=8 -m lmms_eval \
    --model point3r_llm_v2 \
    --model_args pretrained=$model_path,use_flash_attention_2=true,max_num_frames=32,max_length=12800 \
    --tasks ${benchmark} \
    --batch_size 1 \
    --output_path $output_path