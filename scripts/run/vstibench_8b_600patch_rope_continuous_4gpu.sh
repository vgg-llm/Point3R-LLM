#!/bin/bash
#SBATCH --job-name="VSTI-bench Training and evaluation on pointer_memory_600patch_4gpu (continuous RoPE)"
#SBATCH -o sbatch_log/vstibench_8b_600patch_rope_continuous_4gpu.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --export=ALL,MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/chanyoung/Cog3DMap
#SBATCH --wckey=project-short-name:others

# Experiment: 8B with continuous (hierarchical) 3D RoPE on VSTI-bench (no memory-feature fusion)

source venv/bin/activate
export NUM_PROCESSES=4

export EXP_NAME="vstibench_8b_600patch_rope_continuous_4gpu"
export MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
export DATASETS="vstibench_point3r"

# Memory features  -> --merge_memory_feat False
export MERGE_MEMORY_FEAT="False"
export MEMORY_FUSION_METHOD="add"

# RoPE  -> --rope_mode continuous --tune_rope3d_continuous True
export ROPE_MODE="continuous"
export TUNE_ROPE3D_CONTINUOUS="True"

# Evaluation
export BENCHMARKS="vstibench_point3r"
export EVAL_MODEL_TYPE="point3r_llm_v2"
export POINTER_DIR_NAME="pointer_memory_600patch"
export LOG_SUFFIX="vstibench_eval"

# --- Train ---
bash scripts/train/train.sh

# --- Evaluate ---
export MODEL_PATH="${MODEL_OUTPUT_DIR}/${EXP_NAME}"
bash scripts/evaluation/eval.sh

nvidia-smi
date
squeue --job $SLURM_JOBID
echo "##### END #####"
