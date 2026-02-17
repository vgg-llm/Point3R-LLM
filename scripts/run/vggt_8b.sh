#!/bin/bash
#SBATCH --job-name=Point3R-LLM-vggt_8b
#SBATCH -o sbatch_log/vggt-8b.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

# Experiment: VGGT geometry encoder on Scan2Cap 8B

source venv/bin/activate

export EXP_NAME="scan2cap_Qwen3VL_8b_vggt"
export MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
export DATASETS="scan2cap"

# Disable pointer memory (VGGT uses geometry encoder instead)
export USE_POINTER_MEMORY="False"
export USE_PREPROCESSED_INPUT="False"
export MERGE_MEMORY_FEAT="False"
export TUNE_MEMORY_FEATURE_PROJECTOR="False"
export TUNE_MEMORY_FEATURE_FUSION="False"

# Geometry encoder
export USE_GEOMETRY_ENCODER="True"
export GEOMETRY_ENCODER_TYPE="vggt"
export GEOMETRY_ENCODER_PATH="facebook/VGGT-1B"
export FEATURE_FUSION_METHOD="add"

# Evaluation
export BENCHMARKS="scan2cap"
export EVAL_MODEL_TYPE="vggt_qwen3vl"

# --- Train ---
bash scripts/train/train.sh

# --- Evaluate ---
export MODEL_PATH="./outputs/${EXP_NAME}"
bash scripts/evaluation/eval.sh

nvidia-smi
date
squeue --job $SLURM_JOBID
echo "##### END #####"
