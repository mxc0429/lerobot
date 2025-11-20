#!/bin/bash

# Training script for SmolVLA with RMT memory module
# This script trains a SmolVLA model with learnable memory tokens

# Set your Hugging Face username
export HF_USER="your_username"

# Configuration
NUM_MEM_TOKENS=4  # Number of memory tokens (0=disabled, 4=recommended, 8=enhanced)
BATCH_SIZE=4
OUTPUT_DIR="outputs/train/smolvla_with_memory_${NUM_MEM_TOKENS}tokens"
JOB_NAME="smolvla_memory_training"

echo "========================================="
echo "Training SmolVLA with RMT Memory Module"
echo "========================================="
echo "Memory tokens: ${NUM_MEM_TOKENS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "========================================="

# Train with memory tokens
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
  --policy.mem_at_end=false \
  --policy.read_mem_from_cache=false

echo "========================================="
echo "Training completed!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "========================================="
