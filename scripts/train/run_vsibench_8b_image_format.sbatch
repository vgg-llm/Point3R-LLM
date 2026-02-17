#!/bin/bash
#SBATCH --job-name=Point3R-LLM-vsibench_8b_image_format
#SBATCH -o sbatch_log/vsibench-8b-32frame-video-image-format.%j.out # Name of stdout output file
#SBATCH --partition=cms_cvlab
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

GPU_COUNT=$(scontrol show job "$SLURM_JOBID" | grep "TresPerNode" | grep -o "gres/gpu:[0-9]*" | cut -d: -f2 | tr -d '[:space:]')

# set the environment
source venv/bin/activate
# Run the training script
bash scripts/train/train_vsibench_qwen3vl_8b_image_format.sh
bash scripts/evaluation/eval_vsibench_point3r_qwen3vl_8b_image_format.sh
nvidia-smi
date
squeue --job $SLURM_JOBID
echo "##### END #####"
