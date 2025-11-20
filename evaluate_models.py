#!/usr/bin/env python3
"""
Evaluation script to compare SmolVLA models with and without memory.

This script evaluates both models on the same test dataset and compares:
1. Success rate
2. Average reward
3. Inference time
4. Memory usage

Usage:
    python evaluate_models.py \
        --baseline_path outputs/train/smolvla_baseline/checkpoints/last/pretrained_model \
        --memory_path outputs/train/smolvla_with_memory_4tokens/checkpoints/last/pretrained_model \
        --dataset_repo_id ${HF_USER}/pickplace_smolvla \
        --n_episodes 50
"""

import argparse
import time
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.factory import make_dataset
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs import parser


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load a trained SmolVLA model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load config
    config_path = Path(checkpoint_path) / "config.json"
    if not config_path.exists():
        config_path = Path(checkpoint_path) / "train_config.json"
    
    # Load policy
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()
    
    return policy


def evaluate_model(policy, dataset, n_episodes: int = 50, device: str = "cuda"):
    """Evaluate a model on the dataset."""
    results = {
        "success_count": 0,
        "total_episodes": n_episodes,
        "rewards": [],
        "inference_times": [],
        "memory_usage_mb": [],
    }
    
    policy.eval()
    
    with torch.no_grad():
        for episode_idx in tqdm(range(n_episodes), desc="Evaluating"):
            # Reset policy memory for each episode
            policy.reset()
            
            # Get episode data
            episode_data = dataset.get_episode(episode_idx % len(dataset.episodes))
            
            episode_reward = 0
            episode_success = False
            
            for step_idx in range(len(episode_data)):
                # Prepare batch
                batch = dataset[episode_idx * len(episode_data) + step_idx]
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].unsqueeze(0).to(device)
                
                # Measure inference time
                start_time = time.time()
                
                # Get action
                action = policy.select_action(batch)
                
                inference_time = time.time() - start_time
                results["inference_times"].append(inference_time)
                
                # Measure memory usage
                if torch.cuda.is_available():
                    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                    results["memory_usage_mb"].append(memory_mb)
                
                # Accumulate reward (if available in dataset)
                if "reward" in batch:
                    episode_reward += batch["reward"].item()
            
            results["rewards"].append(episode_reward)
            
            # Check success (you may need to adjust this based on your task)
            if episode_reward > 0:  # Simple threshold, adjust as needed
                results["success_count"] += 1
    
    # Compute statistics
    results["success_rate"] = results["success_count"] / results["total_episodes"]
    results["avg_reward"] = np.mean(results["rewards"])
    results["std_reward"] = np.std(results["rewards"])
    results["avg_inference_time_ms"] = np.mean(results["inference_times"]) * 1000
    results["std_inference_time_ms"] = np.std(results["inference_times"]) * 1000
    
    if results["memory_usage_mb"]:
        results["avg_memory_mb"] = np.mean(results["memory_usage_mb"])
        results["max_memory_mb"] = np.max(results["memory_usage_mb"])
    
    return results


def compare_models(baseline_results, memory_results):
    """Compare two models and print results."""
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    print("\n📊 Success Rate:")
    print(f"  Baseline:      {baseline_results['success_rate']:.2%}")
    print(f"  With Memory:   {memory_results['success_rate']:.2%}")
    improvement = (memory_results['success_rate'] - baseline_results['success_rate']) * 100
    print(f"  Improvement:   {improvement:+.2f} percentage points")
    
    print("\n🎯 Average Reward:")
    print(f"  Baseline:      {baseline_results['avg_reward']:.4f} ± {baseline_results['std_reward']:.4f}")
    print(f"  With Memory:   {memory_results['avg_reward']:.4f} ± {memory_results['std_reward']:.4f}")
    reward_improvement = ((memory_results['avg_reward'] - baseline_results['avg_reward']) / 
                          baseline_results['avg_reward'] * 100)
    print(f"  Improvement:   {reward_improvement:+.2f}%")
    
    print("\n⚡ Inference Time:")
    print(f"  Baseline:      {baseline_results['avg_inference_time_ms']:.2f} ± {baseline_results['std_inference_time_ms']:.2f} ms")
    print(f"  With Memory:   {memory_results['avg_inference_time_ms']:.2f} ± {memory_results['std_inference_time_ms']:.2f} ms")
    time_overhead = ((memory_results['avg_inference_time_ms'] - baseline_results['avg_inference_time_ms']) / 
                     baseline_results['avg_inference_time_ms'] * 100)
    print(f"  Overhead:      {time_overhead:+.2f}%")
    
    if 'avg_memory_mb' in baseline_results and 'avg_memory_mb' in memory_results:
        print("\n💾 Memory Usage:")
        print(f"  Baseline:      {baseline_results['avg_memory_mb']:.2f} MB (max: {baseline_results['max_memory_mb']:.2f} MB)")
        print(f"  With Memory:   {memory_results['avg_memory_mb']:.2f} MB (max: {memory_results['max_memory_mb']:.2f} MB)")
        memory_overhead = ((memory_results['avg_memory_mb'] - baseline_results['avg_memory_mb']) / 
                          baseline_results['avg_memory_mb'] * 100)
        print(f"  Overhead:      {memory_overhead:+.2f}%")
    
    print("\n" + "="*80)
    
    # Summary
    print("\n📝 Summary:")
    if improvement > 0:
        print(f"  ✅ Memory module improves success rate by {improvement:.2f} percentage points")
    else:
        print(f"  ❌ Memory module decreases success rate by {abs(improvement):.2f} percentage points")
    
    if reward_improvement > 0:
        print(f"  ✅ Memory module improves average reward by {reward_improvement:.2f}%")
    else:
        print(f"  ❌ Memory module decreases average reward by {abs(reward_improvement):.2f}%")
    
    if time_overhead < 5:
        print(f"  ✅ Inference time overhead is minimal ({time_overhead:.2f}%)")
    else:
        print(f"  ⚠️  Inference time overhead is {time_overhead:.2f}%")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare SmolVLA models with and without memory")
    parser.add_argument("--baseline_path", type=str, required=True,
                       help="Path to baseline model checkpoint")
    parser.add_argument("--memory_path", type=str, required=True,
                       help="Path to memory-enhanced model checkpoint")
    parser.add_argument("--dataset_repo_id", type=str, required=True,
                       help="Dataset repository ID")
    parser.add_argument("--n_episodes", type=int, default=50,
                       help="Number of episodes to evaluate")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for evaluation")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Load models
    print("Loading models...")
    baseline_policy = load_model(args.baseline_path, args.device)
    memory_policy = load_model(args.memory_path, args.device)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_repo_id}")
    from lerobot.configs.train import TrainPipelineConfig
    cfg = TrainPipelineConfig()
    cfg.dataset.repo_id = args.dataset_repo_id
    dataset = make_dataset(cfg)
    
    # Evaluate baseline model
    print("\n" + "="*80)
    print("Evaluating BASELINE model...")
    print("="*80)
    baseline_results = evaluate_model(baseline_policy, dataset, args.n_episodes, args.device)
    
    # Evaluate memory model
    print("\n" + "="*80)
    print("Evaluating MEMORY-ENHANCED model...")
    print("="*80)
    memory_results = evaluate_model(memory_policy, dataset, args.n_episodes, args.device)
    
    # Compare results
    compare_models(baseline_results, memory_results)
    
    # Save results
    results = {
        "baseline": baseline_results,
        "memory": memory_results,
        "comparison": {
            "success_rate_improvement": (memory_results['success_rate'] - baseline_results['success_rate']) * 100,
            "reward_improvement_pct": ((memory_results['avg_reward'] - baseline_results['avg_reward']) / 
                                       baseline_results['avg_reward'] * 100),
            "inference_time_overhead_pct": ((memory_results['avg_inference_time_ms'] - baseline_results['avg_inference_time_ms']) / 
                                           baseline_results['avg_inference_time_ms'] * 100),
        }
    }
    
    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
