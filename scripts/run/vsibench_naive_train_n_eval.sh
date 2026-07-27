#!/bin/bash
#SBATCH --job-name=Cog3DMap_Rebuttal_Naive_Qwen3-VL_VSI-bench_train_n_evaluation
#SBATCH -o sbatch_log/vsibench-naive-Qwen3-VL-8b.%j.out
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --export=ALL,MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/chanyoung/Cog3DMap

# Experiment: VSTIBench with naive evaluation
source venv/bin/activate

export EXP_NAME="vsibench_naive_eval"
export MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
export DATASETS="vsibench"

# Memory features
export MERGE_MEMORY_FEAT="False"
export MEMORY_FUSION_METHOD="add"

# Evaluation
export BENCHMARKS="vsiibench"
export EVAL_MODEL_TYPE="point3r_llm_v2"
export LOG_SUFFIX="vsibench_eval"

# --- Train ---
bash scripts/train/train.sh

# --- Evaluate ---
export MODEL_PATH="${MODEL_OUTPUT_DIR}/${EXP_NAME}"
bash scripts/evaluation/eval.sh

nvidia-smi
date
squeue --job $SLURM_JOBID
echo "##### END #####"
