#!/bin/bash
# Simple parallel preprocessing launcher for ScanNet++
# Runs N independent tasks, each with its own GPU

TOTAL_TASKS=${1:-32}
SAVE_PATH=${2:-"./output/scannetpp"}

echo "Starting ScanNet++ preprocessing on $TOTAL_TASKS tasks..."
echo "Each task will handle ~$(( (1006 + TOTAL_TASKS - 1) / TOTAL_TASKS )) scenes"

# Create logs directory if it doesn't exist
mkdir -p logs

# Launch processes in background, each with different GPU
for task_id in $(seq 0 $((TOTAL_TASKS - 1))); do
    echo "Launching task $task_id..."
    sbatch --job-name=scannetpp_preprocess_${task_id} \
    --output=logs/scannetpp_preprocess_${task_id}.log \
    --error=logs/scannetpp_preprocess_${task_id}.log \
    --partition=cms_cvlab \
    --gres=gpu:1 \
    --nodes=1 \
    scripts/demo/slurm_preprocess_task.sh ${task_id} ${TOTAL_TASKS} ${SAVE_PATH}
done

echo "All $TOTAL_TASKS jobs submitted to SLURM queue!"
echo "Check logs in logs/scannetpp_preprocess_*.log"
echo "Monitor progress with: tail -f logs/scannetpp_preprocess_*.log"
echo "Check job status with: squeue -u \$USER"
echo "##### END #####"
