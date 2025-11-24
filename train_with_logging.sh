#!/bin/bash

# Enhanced training script with detailed logging
# This script trains a model and saves comprehensive logs for later analysis

# Configuration
export HF_USER="${HF_USER:-your_username}"
MODEL_TYPE="${1:-baseline}"  # baseline or memory
NUM_MEM_TOKENS="${2:-0}"     # Number of memory tokens (0 for baseline)
BATCH_SIZE="${3:-4}"
STEPS="${4:-100000}"
OUTPUT_BASE="outputs/train"

# Set output directory based on model type
if [ "$MODEL_TYPE" = "memory" ]; then
    OUTPUT_DIR="${OUTPUT_BASE}/smolvla_with_memory_${NUM_MEM_TOKENS}tokens"
    JOB_NAME="smolvla_memory_${NUM_MEM_TOKENS}tokens"
else
    OUTPUT_DIR="${OUTPUT_BASE}/smolvla_baseline"
    JOB_NAME="smolvla_baseline"
    NUM_MEM_TOKENS=0
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Log file paths
LOG_FILE="${OUTPUT_DIR}/training.log"
METRICS_FILE="${OUTPUT_DIR}/metrics.csv"

echo "========================================="
echo "Training SmolVLA Model"
echo "========================================="
echo "Model type: ${MODEL_TYPE}"
echo "Memory tokens: ${NUM_MEM_TOKENS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Training steps: ${STEPS}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "========================================="

# Create CSV header for metrics
echo "step,loss,lr,grad_norm,update_time,data_time,timestamp" > "${METRICS_FILE}"

# Training command
lerobot-train \
  --policy.path=./smolvla_base \
  --dataset.repo_id=${HF_USER}/pickplace_smolvla \
  --batch_size=${BATCH_SIZE} \
  --output_dir=${OUTPUT_DIR} \
  --job_name=${JOB_NAME} \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.num_mem_tokens=${NUM_MEM_TOKENS} \
  --steps=${STEPS} \
  --log_freq=100 \
  --save_freq=10000 \
  2>&1 | tee "${LOG_FILE}"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ Training completed successfully!"
    echo "========================================="
    echo "Logs saved to: ${LOG_FILE}"
    echo "Metrics saved to: ${METRICS_FILE}"
    echo "Model saved to: ${OUTPUT_DIR}"
    echo ""
    echo "To plot training curves, run:"
    echo "  python plot_training_curves.py \\"
    echo "    --baseline_log outputs/train/smolvla_baseline/training.log \\"
    echo "    --memory_log outputs/train/smolvla_with_memory_${NUM_MEM_TOKENS}tokens/training.log \\"
    echo "    --output_dir plots/"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "❌ Training failed!"
    echo "========================================="
    echo "Check the log file for errors: ${LOG_FILE}"
    echo "========================================="
    exit 1
fi
