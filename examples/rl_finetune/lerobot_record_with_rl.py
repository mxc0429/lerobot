"""
Modified lerobot-record script with RL fine-tuning support

This script extends the standard lerobot-record to support residual RL fine-tuning
during the evaluation/recording process.

Usage:
    python lerobot_record_with_rl.py \
        --robot.type=so101_follower \
        --robot.port=/dev/ttyACM0 \
        --policy.path=outputs/train/smolvla_train/checkpoints/last/pretrained_model \
        --residual.enable=true \
        --residual.checkpoint_path=outputs/rl_finetune/residual_checkpoint.pt \
        --residual.learning_rate=3e-4 \
        --dataset.repo_id=Eval/eval_smolvla3 \
        --dataset.num_episodes=5
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

# Import lerobot modules
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots import make_robot_from_config, RobotConfig
from lerobot.teleoperators import make_teleoperator_from_config, TeleoperatorConfig
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device, init_logging

# Import our residual policy wrapper
from residual_smolvla import ResidualSmolVLAPolicy
from rl_finetune_smolvla import ResidualRLTrainer, SimpleRewardFunction


@dataclass
class ResidualRLConfig:
    """Configuration for residual RL fine-tuning"""
    enable: bool = False
    checkpoint_path: Optional[str] = None
    learning_rate: float = 3e-4
    save_checkpoint: bool = True
    checkpoint_interval: int = 5  # Save checkpoint every N episodes
    update_frequency: int = 1  # Update policy every N episodes


@dataclass
class RecordWithRLConfig:
    """Extended record config with RL support"""
    robot: RobotConfig
    policy: PreTrainedConfig
    residual: ResidualRLConfig = field(default_factory=ResidualRLConfig)
    dataset_repo_id: str = "Eval/eval_smolvla3"
    dataset_single_task: str = "Grab the cube"
    num_episodes: int = 5
    episode_time_s: int = 30
    reset_time_s: int = 10
    fps: int = 30
    device: str = "cuda"
    display_data: bool = True


def record_with_rl_finetuning(cfg: RecordWithRLConfig):
    """
    Record episodes with optional RL fine-tuning of residual policy.
    
    This function integrates residual RL fine-tuning into the recording loop.
    """
    init_logging()
    
    # Initialize robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    
    # Load base policy
    print(f"Loading base policy from {cfg.policy.pretrained_path}")
    base_policy = make_policy(cfg.policy, ds_meta=None)
    base_policy.eval()
    
    # Create residual policy wrapper if enabled
    residual_policy = None
    trainer = None
    reward_fn = None
    
    if cfg.residual.enable:
        print("Initializing residual RL fine-tuning...")
        residual_policy = ResidualSmolVLAPolicy(
            base_policy=base_policy,
            device=cfg.device,
        )
        
        # Load existing residual weights if provided
        if cfg.residual.checkpoint_path and Path(cfg.residual.checkpoint_path).exists():
            print(f"Loading residual weights from {cfg.residual.checkpoint_path}")
            residual_policy.load_residual_weights(cfg.residual.checkpoint_path)
        
        # Create trainer
        trainer = ResidualRLTrainer(
            residual_policy=residual_policy,
            learning_rate=cfg.residual.learning_rate,
            device=cfg.device,
        )
        
        # Create reward function
        reward_fn = SimpleRewardFunction()
        
        # Create checkpoint directory
        if cfg.residual.save_checkpoint:
            checkpoint_dir = Path("outputs/rl_finetune")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create pre/post processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=None,  # You may need to provide dataset stats
    )
    
    device = get_safe_torch_device(cfg.device)
    
    print(f"Starting recording with {'RL fine-tuning' if cfg.residual.enable else 'standard'} mode...")
    
    # Recording loop
    for episode in range(cfg.num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{cfg.num_episodes}")
        print(f"{'='*60}")
        
        # Reset policy
        if residual_policy:
            residual_policy.reset()
            trainer.reset_rollout_storage()
        else:
            base_policy.reset()
        
        if preprocessor:
            preprocessor.reset()
        if postprocessor:
            postprocessor.reset()
        
        # Episode recording loop
        episode_start_time = time.time()
        step_count = 0
        episode_reward = 0.0
        
        while (time.time() - episode_start_time) < cfg.episode_time_s:
            step_start_time = time.time()
            
            # Get observation from robot
            observation = robot.get_observation()
            
            # Predict action
            if residual_policy:
                # Use residual policy
                action = residual_policy.predict_action(
                    observation,
                    task=cfg.dataset_single_task,
                    use_residual=True,
                )
            else:
                # Use base policy
                action = predict_action(
                    observation=observation,
                    policy=base_policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    task=cfg.dataset_single_task,
                )
            
            # Send action to robot
            robot.send_action(action)
            
            # Compute reward (if RL is enabled)
            if residual_policy and reward_fn:
                # Get next observation for reward computation
                next_observation = robot.get_observation()
                
                # Compute reward (simplified - implement your actual reward function)
                reward = reward_fn.compute_reward(
                    observation=observation,
                    action=action,
                    next_observation=next_observation,
                    done=False,  # You'll need to determine this based on your task
                )
                
                # Store in rollout
                trainer.observations.append(observation)
                trainer.actions.append(action.cpu().numpy() if isinstance(action, torch.Tensor) else action)
                trainer.rewards.append(reward)
                trainer.values.append(0.0)  # Placeholder - compute actual value if needed
                trainer.log_probs.append(0.0)  # Placeholder
                trainer.dones.append(False)
                
                episode_reward += reward
            
            # Maintain FPS
            step_dt = time.time() - step_start_time
            sleep_time = (1.0 / cfg.fps) - step_dt
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            step_count += 1
        
        # Update policy at end of episode (if RL is enabled)
        if residual_policy and trainer and len(trainer.rewards) > 0:
            if (episode + 1) % cfg.residual.update_frequency == 0:
                # Compute advantages and returns
                advantages, returns = trainer.compute_gae(
                    trainer.rewards,
                    trainer.values,
                    trainer.dones,
                )
                
                # Update policy
                update_info = trainer.update_policy(advantages, returns)
                print(f"  Episode reward: {episode_reward:.2f}")
                print(f"  Policy loss: {update_info.get('policy_loss', 0):.4f}")
        
        # Save checkpoint
        if residual_policy and cfg.residual.save_checkpoint:
            if (episode + 1) % cfg.residual.checkpoint_interval == 0:
                checkpoint_file = Path("outputs/rl_finetune") / f"residual_checkpoint_ep{episode+1}.pt"
                torch.save(
                    {
                        "residual_policy_state_dict": residual_policy.residual_policy.state_dict(),
                        "episode": episode + 1,
                        "episode_reward": episode_reward,
                    },
                    checkpoint_file,
                )
                print(f"  Saved checkpoint to {checkpoint_file}")
        
        # Reset period
        print(f"Resetting environment...")
        reset_start_time = time.time()
        while (time.time() - reset_start_time) < cfg.reset_time_s:
            # You can add reset logic here
            time.sleep(0.1)
    
    # Save final checkpoint
    if residual_policy and cfg.residual.save_checkpoint:
        final_checkpoint = Path("outputs/rl_finetune") / "residual_checkpoint_final.pt"
        torch.save(
            {
                "residual_policy_state_dict": residual_policy.residual_policy.state_dict(),
                "num_episodes": cfg.num_episodes,
            },
            final_checkpoint,
        )
        print(f"\nSaved final checkpoint to {final_checkpoint}")
    
    robot.disconnect()
    print("\nRecording complete!")


@parser.wrap()
def main():
    """Main entry point - can be called from command line"""
    # This would be called with parser arguments
    # For now, create a simple example config
    pass


if __name__ == "__main__":
    # Example usage
    from lerobot.robots.so101_follower import SO101FollowerConfig
    from lerobot.teleoperators.so101_leader import SO101LeaderConfig
    
    # Create example config
    cfg = RecordWithRLConfig(
        robot=SO101FollowerConfig(
            type="so101_follower",
            port="/dev/ttyACM0",
            id="my_awesome_follower_arm",
        ),
        policy=PreTrainedConfig.from_pretrained(
            "outputs/train/smolvla_train/checkpoints/last/pretrained_model"
        ),
        residual=ResidualRLConfig(
            enable=True,
            learning_rate=3e-4,
        ),
        dataset_single_task="Grab the cube",
        num_episodes=5,
        episode_time_s=30,
        reset_time_s=10,
    )
    
    record_with_rl_finetuning(cfg)

