#!/bin/bash
#SBATCH --job-name=Point3R-LLM-vstibench_8b
#SBATCH -o sbatch_log/vstibench-8b.%j.out # Name of stdout output file
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

GPU_COUNT=$(scontrol show job "$SLURM_JOBID" | grep "TresPerNode" | grep -o "gres/gpu:[0-9]*" | cut -d: -f2 | tr -d '[:space:]')

# set the environment
source venv/bin/activate
# Run the training script
bash scripts/train/train_vstibench_qwen3vl_8b.sh
bash scripts/evaluation/eval_vstibench_point3r_qwen3vl_8b.sh
nvidia-smi
date
squeue --job $SLURM_JOBID   echo "##### END #####"
