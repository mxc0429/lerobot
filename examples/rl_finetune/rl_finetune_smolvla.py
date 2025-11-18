"""
RL Fine-tuning Script for SmolVLA Policy

This script demonstrates how to fine-tune a SmolVLA policy using residual RL
during the evaluation/recording process.

Usage:
    python rl_finetune_smolvla.py \
        --base_policy_path=outputs/train/smolvla_train/checkpoints/last/pretrained_model \
        --residual_checkpoint_path=outputs/rl_finetune/residual_checkpoint.pt \
        --num_episodes=5 \
        --learning_rate=3e-4 \
        --device=cuda
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
from pathlib import Path
import numpy as np
from collections import deque

from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from residual_smolvla import ResidualSmolVLAPolicy


# Import stacking reward function
try:
    from stacking_reward import StackingRewardFunction
except ImportError:
    # Fallback if not available
    class StackingRewardFunction:
        def __init__(self, *args, **kwargs):
            self.success_reward = 10.0
            self.step_penalty = -0.01
        
        def compute_reward(self, *args, **kwargs):
            return -0.01


class SimpleRewardFunction:
    """
    Simple reward function for RL fine-tuning.
    
    For block stacking task, use StackingRewardFunction instead.
    """
    
    def __init__(self):
        self.success_reward = 10.0
        self.step_penalty = -0.01
        self.distance_reward_scale = 0.1
    
    def compute_reward(
        self,
        observation: Dict[str, torch.Tensor],
        action: torch.Tensor,
        next_observation: Optional[Dict[str, torch.Tensor]] = None,
        done: bool = False,
        info: Optional[Dict] = None,
    ) -> float:
        """
        Compute reward for the current step.
        
        This is a placeholder - use StackingRewardFunction for block stacking task.
        """
        reward = self.step_penalty
        
        # Add success reward if task is completed
        if done and info and info.get("success", False):
            reward += self.success_reward
        
        return reward


class ResidualRLTrainer:
    """
    Trainer for residual RL fine-tuning of SmolVLA policy.
    
    This implements a simplified version of PPO for online fine-tuning.
    For production use, consider using a more robust RL algorithm.
    """
    
    def __init__(
        self,
        residual_policy: ResidualSmolVLAPolicy,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "cuda",
    ):
        self.residual_policy = residual_policy
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Optimizer for residual policy
        self.optimizer = optim.Adam(
            residual_policy.get_residual_parameters(),
            lr=learning_rate,
        )
        
        # Storage for rollout data
        self.reset_rollout_storage()
    
    def reset_rollout_storage(self):
        """Reset storage for a new rollout"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def compute_gae(
        self,
        rewards: list,
        values: list,
        dones: list,
        next_value: float = 0.0,
    ) -> tuple:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Returns:
            advantages: List of advantage values
            returns: List of return values
        """
        advantages = []
        returns = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value_step = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value_step = values[step + 1]
            
            delta = (
                rewards[step]
                + self.gamma * next_value_step * next_non_terminal
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        
        return advantages, returns
    
    def update_policy(self, advantages: list, returns: list):
        """
        Update the residual policy using PPO-style update.
        
        Note: This is a simplified version. For production, use a proper
        PPO implementation with multiple update epochs and minibatches.
        """
        if len(advantages) == 0:
            return
        
        # Convert to tensors
        advantages_tensor = torch.tensor(
            advantages, dtype=torch.float32, device=self.device
        )
        returns_tensor = torch.tensor(
            returns, dtype=torch.float32, device=self.device
        )
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )
        
        # Get current policy outputs with actual value computation
        # Recompute observations and actions for current policy
        obs_tensor = torch.stack([
            torch.cat([
                self.residual_policy.extract_state_features(obs),
                torch.tensor(act, dtype=torch.float32, device=self.device)
            ]) for obs, act in zip(self.observations, self.actions)
        ]).to(self.device)
        
        # Get current value estimates
        current_values = self.residual_policy.residual_policy.get_value(obs_tensor).squeeze(-1)
        
        # Compute value loss
        value_loss = 0.5 * ((current_values - returns_tensor) ** 2).mean()
        
        # Compute policy loss (simplified PPO)
        # In full PPO, you'd compute new log_probs and use importance sampling
        policy_loss = -advantages_tensor.mean()
        
        # Add entropy bonus (simplified)
        entropy_loss = -0.01  # Placeholder - would need to compute from policy distribution
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.residual_policy.get_residual_parameters(), 0.5
        )
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss,
            "total_loss": loss.item(),
        }


def fine_tune_during_evaluation(
    base_policy_path: str,
    residual_checkpoint_path: Optional[str] = None,
    num_episodes: int = 5,
    learning_rate: float = 3e-4,
    device: str = "cuda",
    save_checkpoint: bool = True,
    checkpoint_dir: str = "outputs/rl_finetune",
):
    """
    Fine-tune SmolVLA policy using residual RL during evaluation.
    
    Args:
        base_policy_path: Path to pre-trained SmolVLA model
        residual_checkpoint_path: Path to existing residual checkpoint (optional)
        num_episodes: Number of episodes to run
        learning_rate: Learning rate for residual policy
        device: Device to use
        save_checkpoint: Whether to save checkpoints
        checkpoint_dir: Directory to save checkpoints
    """
    # Load base policy
    print(f"Loading base policy from {base_policy_path}")
    base_config = PreTrainedConfig.from_pretrained(base_policy_path)
    base_policy = make_policy(base_config, ds_meta=None)
    base_policy.eval()
    
    # Create residual policy wrapper
    residual_policy = ResidualSmolVLAPolicy(
        base_policy=base_policy,
        device=device,
    )
    
    # Load existing residual weights if provided
    if residual_checkpoint_path and Path(residual_checkpoint_path).exists():
        print(f"Loading residual weights from {residual_checkpoint_path}")
        residual_policy.load_residual_weights(residual_checkpoint_path)
    
    # Create trainer
    trainer = ResidualRLTrainer(
        residual_policy=residual_policy,
        learning_rate=learning_rate,
        device=device,
    )
    
    # Create reward function
    reward_fn = SimpleRewardFunction()
    
    # Create checkpoint directory
    if save_checkpoint:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting RL fine-tuning for {num_episodes} episodes...")
    
    # Training loop
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        # Reset for new episode
        trainer.reset_rollout_storage()
        residual_policy.reset()
        
        # Simulate episode (replace with actual robot interaction)
        # This is a placeholder - integrate with your actual evaluation loop
        episode_reward = 0.0
        
        # Example: simulate some steps
        for step in range(100):  # Replace with actual episode length
            # Get observation (placeholder)
            observation = {}  # Replace with actual observation
            
            # Predict action with residual
            action = residual_policy.predict_action(
                observation,
                use_residual=True,
            )
            
            # Execute action and get reward (placeholder)
            reward = reward_fn.compute_reward(observation, action)
            done = False  # Replace with actual done signal
            
            # Store in rollout
            trainer.observations.append(observation)
            trainer.actions.append(action.cpu().numpy())
            trainer.rewards.append(reward)
            trainer.dones.append(done)
            episode_reward += reward
            
            if done:
                break
        
        # Update policy at end of episode
        if len(trainer.rewards) > 0:
            # Compute advantages and returns
            advantages, returns = trainer.compute_gae(
                trainer.rewards,
                trainer.values,
                trainer.dones,
            )
            
            # Update policy
            update_info = trainer.update_policy(advantages, returns)
            print(f"  Episode reward: {episode_reward:.2f}")
            print(f"  Policy loss: {update_info['policy_loss']:.4f}")
        
        # Save checkpoint
        if save_checkpoint and (episode + 1) % 5 == 0:
            checkpoint_file = checkpoint_path / f"residual_checkpoint_ep{episode+1}.pt"
            torch.save(
                {
                    "residual_policy_state_dict": residual_policy.residual_policy.state_dict(),
                    "episode": episode + 1,
                    "episode_reward": episode_reward,
                },
                checkpoint_file,
            )
            print(f"  Saved checkpoint to {checkpoint_file}")
    
    print("\nFine-tuning complete!")
    
    # Save final checkpoint
    if save_checkpoint:
        final_checkpoint = checkpoint_path / "residual_checkpoint_final.pt"
        torch.save(
            {
                "residual_policy_state_dict": residual_policy.residual_policy.state_dict(),
                "num_episodes": num_episodes,
            },
            final_checkpoint,
        )
        print(f"Saved final checkpoint to {final_checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Fine-tuning for SmolVLA")
    parser.add_argument(
        "--base_policy_path",
        type=str,
        required=True,
        help="Path to pre-trained SmolVLA model",
    )
    parser.add_argument(
        "--residual_checkpoint_path",
        type=str,
        default=None,
        help="Path to existing residual checkpoint (optional)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate for residual policy",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/rl_finetune",
        help="Directory to save checkpoints",
    )
    
    args = parser.parse_args()
    
    fine_tune_during_evaluation(
        base_policy_path=args.base_policy_path,
        residual_checkpoint_path=args.residual_checkpoint_path,
        num_episodes=args.num_episodes,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

