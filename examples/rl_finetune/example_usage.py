#!/usr/bin/env python3
"""
完整的使用示例：在lerobot-record中使用RL微调

这个示例展示了如何修改你的lerobot-record命令来集成RL微调功能。
"""

import sys
import time
from pathlib import Path

# 添加examples路径以便导入
sys.path.insert(0, str(Path(__file__).parent))

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device

from residual_smolvla import ResidualSmolVLAPolicy
from rl_finetune_smolvla import ResidualRLTrainer, SimpleRewardFunction
from stacking_reward import StackingRewardFunction


def record_with_rl_finetuning_example():
    """
    示例：使用RL微调的record函数
    
    这个函数展示了如何将你的lerobot-record命令转换为使用RL微调的版本。
    """
    
    # ========== 配置参数（对应你的lerobot-record命令） ==========
    
    # 机器人配置
    robot_config = {
        "type": "so101_follower",
        "port": "/dev/ttyACM0",
        "cameras": {
            "camera1": {
                "type": "opencv",
                "index_or_path": "/dev/video8",
                "width": 640,
                "height": 480,
                "fps": 30,
                "fourcc": "MJPG"
            },
            "camera2": {
                "type": "opencv",
                "index_or_path": "/dev/video10",
                "width": 640,
                "height": 480,
                "fps": 30,
                "fourcc": "MJPG"
            },
            "camera3": {
                "type": "opencv",
                "index_or_path": "/dev/video6",
                "width": 640,
                "height": 480,
                "fps": 30
            }
        },
        "id": "my_awesome_follower_arm"
    }
    
    # 遥操作配置（可选）
    teleop_config = {
        "type": "so101_leader",
        "port": "/dev/ttyACM1",
        "id": "my_awesome_leader_arm"
    }
    
    # 策略配置
    policy_path = "outputs/train/smolvla_train/checkpoints/last/pretrained_model"
    device = "cuda"
    
    # 数据集配置
    dataset_repo_id = "Eval/eval_smolvla3"
    single_task = "Grab the cube"
    num_episodes = 5
    episode_time_s = 30
    reset_time_s = 10
    fps = 30
    
    # RL微调配置
    enable_rl = True
    residual_checkpoint_path = "outputs/rl_finetune/residual_checkpoint.pt"
    learning_rate = 3e-4
    save_checkpoint_interval = 5  # 每N个episode保存一次
    
    # ========== 初始化 ==========
    
    print("=" * 60)
    print("Initializing RL Fine-tuning for SmolVLA")
    print("=" * 60)
    
    # 初始化机器人
    print("\n[1/5] Initializing robot...")
    robot = make_robot_from_config(robot_config)
    robot.connect()
    print("✓ Robot connected")
    
    # 初始化遥操作（如果需要）
    teleop = None
    if teleop_config:
        print("\n[2/5] Initializing teleoperator...")
        teleop = make_teleoperator_from_config(teleop_config)
        teleop.connect()
        print("✓ Teleoperator connected")
    
    # 加载基础策略
    print(f"\n[3/5] Loading base policy from {policy_path}...")
    base_config = PreTrainedConfig.from_pretrained(policy_path)
    base_policy = make_policy(base_config, ds_meta=None)
    base_policy.eval()
    print("✓ Base policy loaded")
    
    # 创建pre/post processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=base_config,
        pretrained_path=policy_path,
        dataset_stats=None,  # 你可能需要提供dataset stats
    )
    
    # 创建residual policy wrapper（如果启用RL）
    residual_policy = None
    trainer = None
    reward_fn = None
    
    if enable_rl:
        print("\n[4/5] Initializing residual RL fine-tuning...")
        residual_policy = ResidualSmolVLAPolicy(
            base_policy=base_policy,
            device=device,
        )
        
        # 加载已有的residual权重（如果有）
        if residual_checkpoint_path and Path(residual_checkpoint_path).exists():
            print(f"  Loading residual weights from {residual_checkpoint_path}")
            residual_policy.load_residual_weights(residual_checkpoint_path)
        else:
            print("  Starting with fresh residual policy")
        
        # 创建trainer
        trainer = ResidualRLTrainer(
            residual_policy=residual_policy,
            learning_rate=learning_rate,
            device=device,
        )
        
        # 创建reward函数 - 使用堆叠任务的奖励函数
        reward_fn = StackingRewardFunction(
            success_reward=50.0,
            step_penalty=-0.01,
            distance_reward_scale=0.5,
            grasp_reward=5.0,
            stacking_progress_scale=2.0,
            height_reward_scale=1.0,
            stability_reward=10.0,
        )
        print("✓ Residual RL initialized with StackingRewardFunction")
    else:
        print("\n[4/5] RL fine-tuning disabled, using base policy only")
    
    # 创建checkpoint目录
    checkpoint_dir = Path("outputs/rl_finetune")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[5/5] Ready to start recording!")
    print("=" * 60)
    
    # ========== 记录循环 ==========
    
    torch_device = get_safe_torch_device(device)
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        # Reset
        if residual_policy:
            residual_policy.reset()
            trainer.reset_rollout_storage()
            reward_fn.reset()  # Reset reward function state
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
        
        print(f"Recording episode (max {episode_time_s}s)...")
        
        while (time.time() - episode_start_time) < episode_time_s:
            step_start_time = time.time()
            
            # 获取观察
            observation = robot.get_observation()
            
            # 预测动作
            if residual_policy:
                # 使用residual policy
                action = residual_policy.predict_action(
                    observation,
                    task=single_task,
                    use_residual=True,
                )
            else:
                # 使用基础策略
                action = predict_action(
                    observation=observation,
                    policy=base_policy,
                    device=torch_device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    task=single_task,
                )
            
            # 执行动作
            robot.send_action(action)
            
            # 如果启用RL，计算奖励并存储
            if residual_policy and reward_fn:
                # 获取下一个观察（用于奖励计算）
                next_observation = robot.get_observation()
                
                # 计算奖励 - 使用堆叠任务的奖励函数
                # 注意：需要提供block位置信息，可以通过info字典或从observation提取
                info = {}  # TODO: 添加block位置和success信息
                # info = {
                #     "block1_pos": get_block1_position(),  # 需要实现
                #     "block2_pos": get_block2_position(),   # 需要实现
                #     "success": check_success(),             # 需要实现
                # }
                reward = reward_fn.compute_reward(
                    observation=observation,
                    action=action,
                    next_observation=next_observation,
                    done=False,
                    info=info,
                )
                
            # 计算value估计
            if residual_policy:
                value = residual_policy.get_value(observation, action).item()
            else:
                value = 0.0
            
            # 存储到rollout
            trainer.observations.append(observation)
            trainer.actions.append(
                action.cpu().numpy() if isinstance(action, torch.Tensor) else action
            )
            trainer.rewards.append(reward)
            trainer.values.append(value)  # 使用实际的value估计
            trainer.log_probs.append(0.0)  # TODO: 如果需要完整的PPO，需要计算log_prob
            trainer.dones.append(False)
            
            episode_reward += reward
            
            # 维持FPS
            step_dt = time.time() - step_start_time
            sleep_time = (1.0 / fps) - step_dt
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            step_count += 1
            
            # 每100步打印一次进度
            if step_count % 100 == 0:
                elapsed = time.time() - episode_start_time
                print(f"  Step {step_count}, elapsed: {elapsed:.1f}s, reward: {episode_reward:.2f}")
        
        # 更新策略（如果启用RL）
        if residual_policy and trainer and len(trainer.rewards) > 0:
            print(f"\nUpdating policy...")
            advantages, returns = trainer.compute_gae(
                trainer.rewards,
                trainer.values,
                trainer.dones,
            )
            update_info = trainer.update_policy(advantages, returns)
            print(f"  Episode reward: {episode_reward:.2f}")
            print(f"  Policy loss: {update_info.get('policy_loss', 0):.4f}")
        
        # 保存checkpoint
        if residual_policy and (episode + 1) % save_checkpoint_interval == 0:
            checkpoint_file = checkpoint_dir / f"residual_checkpoint_ep{episode+1}.pt"
            torch.save(
                {
                    "residual_policy_state_dict": residual_policy.residual_policy.state_dict(),
                    "episode": episode + 1,
                    "episode_reward": episode_reward,
                },
                checkpoint_file,
            )
            print(f"  ✓ Saved checkpoint to {checkpoint_file}")
        
        # Reset period
        if episode < num_episodes - 1:  # 最后一个episode不需要reset
            print(f"\nResetting environment ({reset_time_s}s)...")
            reset_start_time = time.time()
            while (time.time() - reset_start_time) < reset_time_s:
                time.sleep(0.1)
    
    # 保存最终checkpoint
    if residual_policy:
        final_checkpoint = checkpoint_dir / "residual_checkpoint_final.pt"
        torch.save(
            {
                "residual_policy_state_dict": residual_policy.residual_policy.state_dict(),
                "num_episodes": num_episodes,
            },
            final_checkpoint,
        )
        print(f"\n✓ Saved final checkpoint to {final_checkpoint}")
    
    # 清理
    robot.disconnect()
    if teleop:
        teleop.disconnect()
    
    print("\n" + "=" * 60)
    print("Recording complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        record_with_rl_finetuning_example()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

