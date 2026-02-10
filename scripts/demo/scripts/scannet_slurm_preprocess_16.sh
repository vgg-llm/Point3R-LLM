#!/bin/bash
# ScanNet preprocessing launcher for 16 parallel tasks
# Each task processes a different chunk of the dataset

TOTAL_TASKS=16
SAVE_NAME=${1:-"pointer_memory_qwen3vl"}
SAMPLE_CT=${2:-32}

echo "=============================================="
echo "ScanNet Preprocessing - 16 Parallel Tasks"
echo "=============================================="
echo ""
echo "Total scenes in dataset: 1513"
echo "Tasks to launch: $TOTAL_TASKS"
echo "Scenes per task: ~$(( (1513 + TOTAL_TASKS - 1) / TOTAL_TASKS ))"
echo "Save name: $SAVE_NAME"
echo "Sample count: $SAMPLE_CT"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Launch 16 independent tasks, each with different chunk ID
echo "Submitting jobs to SLURM queue..."
for task_id in $(seq 0 $((TOTAL_TASKS - 1))); do
    echo "  - Submitting task $task_id/$((TOTAL_TASKS-1))..."
    sbatch --job-name=scannet_${task_id} \
           --output=logs/scannet_chunk_${task_id}.log \
           --error=logs/scannet_chunk_${task_id}.log \
           --partition=cms_cvlab \
           --gres=gpu:1 \
           --nodes=1 \
           scripts/demo/slurm_preprocess_scannet_task.sh ${task_id} ${TOTAL_TASKS} ${SAVE_NAME} ${SAMPLE_CT}
done

echo ""
echo "✓ All $TOTAL_TASKS jobs submitted successfully!"
echo ""
echo "Monitor your jobs:"
echo "  - Check status:   squeue -u \$USER"
echo "  - View logs:      tail -f logs/scannet_chunk_*.log"
echo "  - Cancel all:     scancel -u \$USER -n scannet"
echo ""
echo "Each task processes scenes: task_id, task_id+$TOTAL_TASKS, task_id+$((TOTAL_TASKS*2)), ..."
echo "Example: Task 0 processes scenes 0, 16, 32, 48, ..."
echo "         Task 1 processes scenes 1, 17, 33, 49, ..."
echo ""
echo "##### SUBMITTED #####"
