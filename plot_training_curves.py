#!/usr/bin/env python3
"""
Plot training curves to compare baseline and memory-enhanced models.

This script reads training logs and creates comparison plots for:
- Training loss
- Learning rate
- Gradient norm
- Training time per step

Usage:
    python plot_training_curves.py \
        --baseline_log outputs/train/smolvla_baseline/log.txt \
        --memory_log outputs/train/smolvla_with_memory_4tokens/log.txt \
        --output_dir plots/
"""

import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple


def parse_log_file(log_path: str) -> Dict[str, List[float]]:
    """Parse training log file and extract metrics."""
    metrics = {
        'step': [],
        'loss': [],
        'lr': [],
        'grad_norm': [],
        'update_time': [],
        'data_time': [],
    }
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Look for lines with training metrics
            # Example: "Step 100: loss=0.1234 lr=1e-4 grdn=0.5 updt_s=0.123 data_s=0.045"
            
            # Extract step number
            step_match = re.search(r'step[=\s]+(\d+)', line, re.IGNORECASE)
            if not step_match:
                continue
            
            step = int(step_match.group(1))
            
            # Extract loss
            loss_match = re.search(r'loss[=:\s]+([\d.e+-]+)', line, re.IGNORECASE)
            if loss_match:
                loss = float(loss_match.group(1))
            else:
                continue
            
            # Extract learning rate
            lr_match = re.search(r'lr[=:\s]+([\d.e+-]+)', line, re.IGNORECASE)
            lr = float(lr_match.group(1)) if lr_match else None
            
            # Extract gradient norm
            grad_match = re.search(r'(?:grdn|grad_norm)[=:\s]+([\d.e+-]+)', line, re.IGNORECASE)
            grad_norm = float(grad_match.group(1)) if grad_match else None
            
            # Extract update time
            update_match = re.search(r'(?:updt_s|update_s)[=:\s]+([\d.e+-]+)', line, re.IGNORECASE)
            update_time = float(update_match.group(1)) if update_match else None
            
            # Extract data loading time
            data_match = re.search(r'(?:data_s|dataloading_s)[=:\s]+([\d.e+-]+)', line, re.IGNORECASE)
            data_time = float(data_match.group(1)) if data_match else None
            
            # Store metrics
            metrics['step'].append(step)
            metrics['loss'].append(loss)
            if lr is not None:
                metrics['lr'].append(lr)
            if grad_norm is not None:
                metrics['grad_norm'].append(grad_norm)
            if update_time is not None:
                metrics['update_time'].append(update_time)
            if data_time is not None:
                metrics['data_time'].append(data_time)
    
    return metrics


def smooth_curve(values: List[float], window: int = 100) -> np.ndarray:
    """Apply moving average smoothing to a curve."""
    if len(values) < window:
        return np.array(values)
    
    weights = np.ones(window) / window
    smoothed = np.convolve(values, weights, mode='valid')
    
    # Pad the beginning to maintain length
    pad_length = len(values) - len(smoothed)
    smoothed = np.concatenate([np.full(pad_length, smoothed[0]), smoothed])
    
    return smoothed


def plot_comparison(baseline_metrics: Dict, memory_metrics: Dict, output_dir: Path):
    """Create comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {'baseline': '#1f77b4', 'memory': '#ff7f0e'}
    
    # 1. Loss comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if baseline_metrics['loss']:
        baseline_steps = baseline_metrics['step']
        baseline_loss = baseline_metrics['loss']
        baseline_loss_smooth = smooth_curve(baseline_loss)
        
        ax.plot(baseline_steps, baseline_loss, alpha=0.2, color=colors['baseline'])
        ax.plot(baseline_steps, baseline_loss_smooth, label='Baseline (smoothed)', 
                color=colors['baseline'], linewidth=2)
    
    if memory_metrics['loss']:
        memory_steps = memory_metrics['step']
        memory_loss = memory_metrics['loss']
        memory_loss_smooth = smooth_curve(memory_loss)
        
        ax.plot(memory_steps, memory_loss, alpha=0.2, color=colors['memory'])
        ax.plot(memory_steps, memory_loss_smooth, label='With Memory (smoothed)', 
                color=colors['memory'], linewidth=2)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_dir / 'loss_comparison.png'}")
    
    # 2. Learning rate comparison
    if baseline_metrics['lr'] or memory_metrics['lr']:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if baseline_metrics['lr']:
            ax.plot(baseline_metrics['step'][:len(baseline_metrics['lr'])], 
                   baseline_metrics['lr'], label='Baseline', 
                   color=colors['baseline'], linewidth=2)
        
        if memory_metrics['lr']:
            ax.plot(memory_metrics['step'][:len(memory_metrics['lr'])], 
                   memory_metrics['lr'], label='With Memory', 
                   color=colors['memory'], linewidth=2)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'lr_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_dir / 'lr_comparison.png'}")
    
    # 3. Gradient norm comparison
    if baseline_metrics['grad_norm'] or memory_metrics['grad_norm']:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if baseline_metrics['grad_norm']:
            baseline_grad_smooth = smooth_curve(baseline_metrics['grad_norm'])
            ax.plot(baseline_metrics['step'][:len(baseline_grad_smooth)], 
                   baseline_grad_smooth, label='Baseline', 
                   color=colors['baseline'], linewidth=2)
        
        if memory_metrics['grad_norm']:
            memory_grad_smooth = smooth_curve(memory_metrics['grad_norm'])
            ax.plot(memory_metrics['step'][:len(memory_grad_smooth)], 
                   memory_grad_smooth, label='With Memory', 
                   color=colors['memory'], linewidth=2)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Gradient Norm', fontsize=12)
        ax.set_title('Gradient Norm Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'grad_norm_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_dir / 'grad_norm_comparison.png'}")
    
    # 4. Training time comparison
    if baseline_metrics['update_time'] or memory_metrics['update_time']:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if baseline_metrics['update_time']:
            baseline_time_smooth = smooth_curve(baseline_metrics['update_time'])
            ax.plot(baseline_metrics['step'][:len(baseline_time_smooth)], 
                   baseline_time_smooth, label='Baseline', 
                   color=colors['baseline'], linewidth=2)
        
        if memory_metrics['update_time']:
            memory_time_smooth = smooth_curve(memory_metrics['update_time'])
            ax.plot(memory_metrics['step'][:len(memory_time_smooth)], 
                   memory_time_smooth, label='With Memory', 
                   color=colors['memory'], linewidth=2)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Update Time (seconds)', fontsize=12)
        ax.set_title('Training Time per Step', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_dir / 'time_comparison.png'}")
    
    # 5. Combined overview
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss
    ax = axes[0, 0]
    if baseline_metrics['loss']:
        baseline_loss_smooth = smooth_curve(baseline_metrics['loss'])
        ax.plot(baseline_metrics['step'], baseline_loss_smooth, 
               label='Baseline', color=colors['baseline'], linewidth=2)
    if memory_metrics['loss']:
        memory_loss_smooth = smooth_curve(memory_metrics['loss'])
        ax.plot(memory_metrics['step'], memory_loss_smooth, 
               label='With Memory', color=colors['memory'], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[0, 1]
    if baseline_metrics['lr']:
        ax.plot(baseline_metrics['step'][:len(baseline_metrics['lr'])], 
               baseline_metrics['lr'], label='Baseline', 
               color=colors['baseline'], linewidth=2)
    if memory_metrics['lr']:
        ax.plot(memory_metrics['step'][:len(memory_metrics['lr'])], 
               memory_metrics['lr'], label='With Memory', 
               color=colors['memory'], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Gradient norm
    ax = axes[1, 0]
    if baseline_metrics['grad_norm']:
        baseline_grad_smooth = smooth_curve(baseline_metrics['grad_norm'])
        ax.plot(baseline_metrics['step'][:len(baseline_grad_smooth)], 
               baseline_grad_smooth, label='Baseline', 
               color=colors['baseline'], linewidth=2)
    if memory_metrics['grad_norm']:
        memory_grad_smooth = smooth_curve(memory_metrics['grad_norm'])
        ax.plot(memory_metrics['step'][:len(memory_grad_smooth)], 
               memory_grad_smooth, label='With Memory', 
               color=colors['memory'], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Update time
    ax = axes[1, 1]
    if baseline_metrics['update_time']:
        baseline_time_smooth = smooth_curve(baseline_metrics['update_time'])
        ax.plot(baseline_metrics['step'][:len(baseline_time_smooth)], 
               baseline_time_smooth, label='Baseline', 
               color=colors['baseline'], linewidth=2)
    if memory_metrics['update_time']:
        memory_time_smooth = smooth_curve(memory_metrics['update_time'])
        ax.plot(memory_metrics['step'][:len(memory_time_smooth)], 
               memory_time_smooth, label='With Memory', 
               color=colors['memory'], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Time (s)')
    ax.set_title('Update Time per Step')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training Metrics Comparison: Baseline vs Memory-Enhanced', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'overview_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_dir / 'overview_comparison.png'}")


def print_statistics(baseline_metrics: Dict, memory_metrics: Dict):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("TRAINING STATISTICS SUMMARY")
    print("="*80)
    
    if baseline_metrics['loss'] and memory_metrics['loss']:
        # Final loss (average of last 100 steps)
        baseline_final_loss = np.mean(baseline_metrics['loss'][-100:])
        memory_final_loss = np.mean(memory_metrics['loss'][-100:])
        
        print(f"\n📊 Final Loss (avg of last 100 steps):")
        print(f"  Baseline:    {baseline_final_loss:.6f}")
        print(f"  With Memory: {memory_final_loss:.6f}")
        print(f"  Improvement: {(baseline_final_loss - memory_final_loss) / baseline_final_loss * 100:+.2f}%")
    
    if baseline_metrics['update_time'] and memory_metrics['update_time']:
        # Average update time
        baseline_avg_time = np.mean(baseline_metrics['update_time'])
        memory_avg_time = np.mean(memory_metrics['update_time'])
        
        print(f"\n⚡ Average Update Time:")
        print(f"  Baseline:    {baseline_avg_time:.4f} s")
        print(f"  With Memory: {memory_avg_time:.4f} s")
        print(f"  Overhead:    {(memory_avg_time - baseline_avg_time) / baseline_avg_time * 100:+.2f}%")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves comparison")
    parser.add_argument("--baseline_log", type=str, required=True,
                       help="Path to baseline model log file")
    parser.add_argument("--memory_log", type=str, required=True,
                       help="Path to memory-enhanced model log file")
    parser.add_argument("--output_dir", type=str, default="plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Parse log files
    print("📖 Parsing log files...")
    baseline_metrics = parse_log_file(args.baseline_log)
    memory_metrics = parse_log_file(args.memory_log)
    
    print(f"  Baseline: {len(baseline_metrics['step'])} data points")
    print(f"  Memory:   {len(memory_metrics['step'])} data points")
    
    # Create plots
    print("\n📈 Creating comparison plots...")
    output_dir = Path(args.output_dir)
    plot_comparison(baseline_metrics, memory_metrics, output_dir)
    
    # Print statistics
    print_statistics(baseline_metrics, memory_metrics)
    
    print(f"✅ All plots saved to: {output_dir}")
    print("\nYou can now view the plots to compare training performance!")


if __name__ == "__main__":
    main()
