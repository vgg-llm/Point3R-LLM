set -e
export LMMS_EVAL_LAUNCHER="accelerate"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"

export NCCL_NVLS_ENABLE=0
benchmark=scan2cap_point3r # choices: [scan2cap, scanrefer, scannet_4frames, scannet_6frames]
output_path=logs/$(TZ="Asia/Shanghai" date "+%Y%m%d")
model_path=./outputs/scan2cap_point3r_point3r

model_args_str="pretrained=$model_path,use_flash_attention_2=true,max_num_frames=32,max_length=12800,use_pointer_memory=True,use_preprocessed_input=True"
if [ "$benchmark" = "scanrefer" ]; then
    model_args_str="${model_args_str},add_frame_index=true"
fi

mkdir -p $output_path

accelerate launch --num_processes=8  --main_process_port 29501 -m lmms_eval \
    --model point3r_llm \
    --model_args "$model_args_str" \
    --tasks ${benchmark} \
    --batch_size 1 \
    --log_samples_suffix original \
    --log_samples \
    --output_path $output_path \
    2>&1 | tee ${output_path}/eval.log