# RL Fine-tuning for SmolVLA Policy

这个示例展示了如何在验证lerobot训练的smolvla模型时，使用RL文件夹中的residual policy方法进行在线微调。

## 概述

Residual RL方法的核心思想是：
1. 保持预训练的SmolVLA基础策略不变
2. 添加一个小的残差网络（Residual MLP）来学习对基础策略动作的修正
3. 最终动作 = 基础策略动作 + 残差修正
4. 使用强化学习（如PPO）在线微调残差网络

这种方法的好处：
- 保护预训练模型的知识
- 只需要训练一个小的残差网络，计算效率高
- 可以快速适应新的任务或环境

## 文件说明

- `residual_smolvla.py`: Residual SmolVLA策略包装器，将RL的residual policy方法适配到SmolVLA模型
- `rl_finetune_smolvla.py`: RL微调训练脚本，包含简化的PPO实现
- `lerobot_record_with_rl.py`: 修改后的lerobot-record脚本，集成RL微调功能
- `example_usage.py`: 使用示例

## 使用方法

### 方法1: 使用修改后的record脚本（推荐）

创建一个新的脚本，基于你的lerobot-record命令，但使用residual policy：

```python
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101LeaderConfig
from residual_smolvla import ResidualSmolVLAPolicy
from rl_finetune_smolvla import ResidualRLTrainer, SimpleRewardFunction

# 加载基础策略
base_config = PreTrainedConfig.from_pretrained(
    "outputs/train/smolvla_train/checkpoints/last/pretrained_model"
)
base_policy = make_policy(base_config, ds_meta=None)

# 创建residual policy wrapper
residual_policy = ResidualSmolVLAPolicy(
    base_policy=base_policy,
    device="cuda",
)

# 加载已有的residual权重（如果有）
# residual_policy.load_residual_weights("outputs/rl_finetune/residual_checkpoint.pt")

# 创建trainer
trainer = ResidualRLTrainer(
    residual_policy=residual_policy,
    learning_rate=3e-4,
    device="cuda",
)

# 在你的record循环中使用residual_policy.predict_action()代替原来的policy
```

### 方法2: 直接集成到现有代码

在你的验证脚本中，替换策略加载部分：

```python
# 原来的代码
# policy = make_policy(cfg.policy, ds_meta=dataset.meta)

# 改为使用residual policy
from residual_smolvla import ResidualSmolVLAPolicy

base_policy = make_policy(cfg.policy, ds_meta=dataset.meta)
residual_policy = ResidualSmolVLAPolicy(
    base_policy=base_policy,
    device=cfg.policy.device,
)

# 在predict_action时使用residual_policy
action = residual_policy.predict_action(
    observation,
    task=single_task,
    use_residual=True,  # 设置为False可以只用基础策略
)
```

### 方法3: 使用命令行脚本

```bash
python rl_finetune_smolvla.py \
    --base_policy_path=outputs/train/smolvla_train/checkpoints/last/pretrained_model \
    --residual_checkpoint_path=outputs/rl_finetune/residual_checkpoint.pt \
    --num_episodes=5 \
    --learning_rate=3e-4 \
    --device=cuda
```

## 完整示例

以下是一个完整的示例，展示如何修改你的lerobot-record命令来使用RL微调：

```python
#!/usr/bin/env python3
"""
示例：在lerobot-record中使用RL微调
"""

import sys
from pathlib import Path

# 添加examples路径
sys.path.insert(0, str(Path(__file__).parent))

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.robots import make_robot_from_config
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device

from residual_smolvla import ResidualSmolVLAPolicy
from rl_finetune_smolvla import ResidualRLTrainer, SimpleRewardFunction


def record_with_rl():
    """使用RL微调的record函数"""
    
    # 配置参数（对应你的lerobot-record命令）
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
    
    policy_path = "outputs/train/smolvla_train/checkpoints/last/pretrained_model"
    device = "cuda"
    num_episodes = 5
    episode_time_s = 30
    reset_time_s = 10
    task = "Grab the cube"
    
    # 初始化机器人
    robot = make_robot_from_config(robot_config)
    robot.connect()
    
    # 加载基础策略
    print(f"Loading base policy from {policy_path}")
    base_config = PreTrainedConfig.from_pretrained(policy_path)
    base_policy = make_policy(base_config, ds_meta=None)
    base_policy.eval()
    
    # 创建residual policy wrapper
    print("Initializing residual RL fine-tuning...")
    residual_policy = ResidualSmolVLAPolicy(
        base_policy=base_policy,
        device=device,
    )
    
    # 可选：加载已有的residual权重
    residual_checkpoint = "outputs/rl_finetune/residual_checkpoint.pt"
    if Path(residual_checkpoint).exists():
        print(f"Loading residual weights from {residual_checkpoint}")
        residual_policy.load_residual_weights(residual_checkpoint)
    
    # 创建trainer
    trainer = ResidualRLTrainer(
        residual_policy=residual_policy,
        learning_rate=3e-4,
        device=device,
    )
    
    # 创建reward函数（需要根据你的任务实现）
    reward_fn = SimpleRewardFunction()
    
    # 记录循环
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        # Reset
        residual_policy.reset()
        trainer.reset_rollout_storage()
        
        # Episode loop
        episode_start = time.time()
        while (time.time() - episode_start) < episode_time_s:
            # 获取观察
            observation = robot.get_observation()
            
            # 使用residual policy预测动作
            action = residual_policy.predict_action(
                observation,
                task=task,
                use_residual=True,
            )
            
            # 执行动作
            robot.send_action(action)
            
            # 计算奖励（简化版本，需要根据实际任务实现）
            reward = reward_fn.compute_reward(observation, action)
            
            # 存储到rollout
            trainer.observations.append(observation)
            trainer.actions.append(action.cpu().numpy() if isinstance(action, torch.Tensor) else action)
            trainer.rewards.append(reward)
            trainer.values.append(0.0)  # 需要计算实际value
            trainer.log_probs.append(0.0)  # 需要计算实际log_prob
            trainer.dones.append(False)
            
            time.sleep(1.0 / 30)  # 30 FPS
        
        # 更新策略
        if len(trainer.rewards) > 0:
            advantages, returns = trainer.compute_gae(
                trainer.rewards,
                trainer.values,
                trainer.dones,
            )
            update_info = trainer.update_policy(advantages, returns)
            print(f"  Episode reward: {sum(trainer.rewards):.2f}")
            print(f"  Policy loss: {update_info.get('policy_loss', 0):.4f}")
        
        # 保存checkpoint
        if (episode + 1) % 5 == 0:
            checkpoint_path = Path("outputs/rl_finetune") / f"residual_checkpoint_ep{episode+1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "residual_policy_state_dict": residual_policy.residual_policy.state_dict(),
                    "episode": episode + 1,
                },
                checkpoint_path,
            )
            print(f"  Saved checkpoint to {checkpoint_path}")
        
        # Reset period
        print("Resetting...")
        time.sleep(reset_time_s)
    
    robot.disconnect()
    print("\nRecording complete!")


if __name__ == "__main__":
    import time
    import torch
    record_with_rl()
```

## 关键注意事项

1. **奖励函数**: `SimpleRewardFunction`是一个占位符，你需要根据你的实际任务实现真正的奖励函数。奖励应该基于：
   - 任务成功（如物体被抓取、任务完成）
   - 到目标的距离
   - 安全约束
   - 等等

2. **状态特征提取**: `ResidualSmolVLAPolicy.extract_state_features()`目前是一个简化版本。在实际使用中，你应该：
   - 使用基础策略的编码器提取特征
   - 或者直接使用机器人状态（如果可用）
   - 或者组合多个观察模态

3. **Value和Log Prob计算**: 当前的实现中，value和log_prob是占位符。在完整的PPO实现中，你需要：
   - 使用critic网络计算value
   - 使用actor网络计算log_prob
   - 实现importance sampling来更新策略

4. **与RL文件夹的集成**: 如果你想使用RL文件夹中更完整的PPO实现，可以：
   - 参考`RL/src/train/residual_ppo.py`
   - 适配其中的ResidualDiffusionPolicy或ResidualMlpPolicy
   - 但需要注意RL文件夹中的代码是为FurnitureBench仿真环境设计的

## 下一步

1. 实现你的奖励函数
2. 改进状态特征提取
3. 完善PPO实现（如果需要更稳定的训练）
4. 调整超参数（learning_rate, action_scale等）
5. 添加更多的评估指标和日志

## 参考

- RL文件夹中的residual policy实现: `RL/src/behavior/residual_diffusion.py`
- RL文件夹中的PPO训练: `RL/src/train/residual_ppo.py`
- SmolVLA论文: https://huggingface.co/papers/2506.01844


