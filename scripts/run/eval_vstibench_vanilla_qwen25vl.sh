#!/bin/bash
#SBATCH --job-name=Eval-VSTIBench-Vanilla-Qwen25VL
#SBATCH -o sbatch_log/vstibench_vanilla_Qwen25VL.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

# Experiment: Evaluate VSTIBench-Reproduction

# source venv/bin/activate

# Model 0
# --- Shared config ---
export EXP_NAME="vstibench_vanilla_Qwen25VL"
export MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
export BENCHMARKS="vstibench"
export EVAL_MODEL_TYPE="point3r_llm"

# Memory features
export MERGE_MEMORY_FEAT="False"
export TUNE_MEMORY_FEATURE_PROJECTOR="False"
export TUNE_MEMORY_FEATURE_FUSION="False"

export LOG_SUFFIX="vanilla_vstibench"
bash scripts/evaluation/eval.sh

nvidia-smi
date
# squeue --job $SLURM_JOBID
echo "##### END #####"
