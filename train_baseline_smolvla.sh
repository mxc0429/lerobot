#!/bin/bash

# Training script for baseline SmolVLA (without memory)
# This is for comparison with the memory-enhanced version

# Set your Hugging Face username
export HF_USER="your_username"

# Configuration
BATCH_SIZE=4
OUTPUT_DIR="outputs/train/smolvla_baseline"
JOB_NAME="smolvla_baseline_training"

echo "========================================="
echo "Training Baseline SmolVLA (No Memory)"
echo "========================================="
echo "Batch size: ${BATCH_SIZE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "========================================="

# Train without memory tokens (num_mem_tokens=0 is default)
lerobot-train \
  --policy.path=./smolvla_base \
  --dataset.repo_id=${HF_USER}/pickplace_smolvla \
  --batch_size=${BATCH_SIZE} \
  --output_dir=${OUTPUT_DIR} \
  --job_name=${JOB_NAME} \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.num_mem_tokens=0

echo "========================================="
echo "Training completed!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "========================================="
