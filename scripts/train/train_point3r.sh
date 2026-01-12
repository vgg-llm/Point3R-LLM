#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

# ======================
# Model Configuration
# ======================
DATASETS="scan2cap_point3r"                        # [DataArguments] Dataset with sampling rate; one of "scan2cap,scanrefer,scannet_det"

# ======================
# Path Configuration
# ======================
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"  # [ModelArguments] Pretrained model path
EXP_NAME="${DATASETS}_with_pose"
OUTPUT_DIR="./outputs/${EXP_NAME}" 
CACHE_DIR="./cache"                        # [TrainingArguments] Cache directory for models
mkdir -p $OUTPUT_DIR
cp "${BASH_SOURCE[0]}" "${OUTPUT_DIR}/train_script.sh"  # Save copy of training script for reproducibility


# ======================
# Training Hyperparameters
# ======================
export NCCL_NVLS_ENABLE=0
export WANDB_PROJECT="Point3R-LLM"
RUN_NAME="run_$(date +%Y%m%d_%H%M%S)_${EXP_NAME}"
export WANDB_RUN_NAME="$RUN_NAME"
LR=1e-5
total_batch_size=16
GRADIENT_ACCUMULATION_STEPS=$(($total_batch_size / $NPROC_PER_NODE))

torchrun --nproc_per_node=$NPROC_PER_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            src/qwen_vl/train/train_qwen.py \
            --model_name_or_path $MODEL_PATH \
            --tune_mm_llm True \
            --tune_mm_vision False \
            --tune_mm_mlp False \
            --dataset_use $DATASETS \
            --output_dir $OUTPUT_DIR \
            --cache_dir $CACHE_DIR \
            --bf16 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
            --learning_rate $LR \
            --mm_projector_lr 1e-5 \
            --vision_tower_lr 1e-6 \
            --optim adamw_torch \
            --model_max_length 12800 \
            --data_flatten False \
            --max_pixels $((576*28*28)) \
            --min_pixels $((16*28*28)) \
            --base_interval 2 \
            --video_max_frames 8 \
            --video_min_frames 4 \
            --video_max_frame_pixels $((1664*28*28)) \
            --video_min_frame_pixels $((256*28*28)) \
            --num_train_epochs 1 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --weight_decay 0.01 \
            --logging_steps 10 \
            --save_steps 1000 \
            --save_total_limit 1 \
            --deepspeed "scripts/zero2_opt.json" \
            --gradient_checkpointing \
            --dataloader_num_workers 4 \
            --group_by_modality_length true \
            --seed 0 \
            --report_to "wandb" \
            --use_geometry_encoder False \
            --use_pointer_memory True \
            --use_preprocessed_input True \
            2>&1 | tee ${OUTPUT_DIR}/train.log 2>&1
