# 快速开始指南

## 概述

这个示例展示了如何在验证lerobot训练的smolvla模型时，使用RL文件夹中的residual policy方法进行在线微调。

## 核心概念

**Residual RL方法**：
- 基础策略（SmolVLA）保持不变
- 添加一个小的残差网络来学习动作修正
- 最终动作 = 基础动作 + 残差修正
- 使用PPO等RL算法在线微调残差网络

## 快速使用

### 步骤1: 准备你的环境

确保你已经：
1. 训练好了SmolVLA模型（在`outputs/train/smolvla_train/checkpoints/last/pretrained_model`）
2. 配置好了机器人（so101_follower等）
3. 安装了必要的依赖

### 步骤2: 运行示例

最简单的方式是直接运行示例脚本：

```bash
cd /home/cv502/Robot/lerobot/examples/rl_finetune
python example_usage.py
```

### 步骤3: 集成到你的代码

如果你想集成到现有的lerobot-record命令中，可以这样做：

```python
# 在你的record脚本中
from residual_smolvla import ResidualSmolVLAPolicy
from rl_finetune_smolvla import ResidualRLTrainer

# 加载基础策略
base_policy = make_policy(cfg.policy, ds_meta=dataset.meta)

# 创建residual policy wrapper
residual_policy = ResidualSmolVLAPolicy(
    base_policy=base_policy,
    device="cuda",
)

# 在predict_action时使用
action = residual_policy.predict_action(
    observation,
    task=single_task,
    use_residual=True,  # 设置为False可以只用基础策略
)
```

## 对应你的原始命令

你的原始lerobot-record命令：

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --policy.path=outputs/train/smolvla_train/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --dataset.repo_id=Eval/eval_smolvla3 \
  --dataset.single_task="Grab the cube" \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10
```

使用RL微调的版本（在`example_usage.py`中）：

```python
# 配置参数（对应上面的命令）
robot_config = {
    "type": "so101_follower",
    "port": "/dev/ttyACM0",
    # ... cameras配置
}

policy_path = "outputs/train/smolvla_train/checkpoints/last/pretrained_model"
device = "cuda"
dataset_repo_id = "Eval/eval_smolvla3"
single_task = "Grab the cube"
num_episodes = 5
episode_time_s = 30
reset_time_s = 10

# 然后使用residual_policy.predict_action()代替原来的predict_action
```

## 关键修改点

1. **策略加载**：
   ```python
   # 原来
   policy = make_policy(cfg.policy, ds_meta=dataset.meta)
   
   # 改为
   base_policy = make_policy(cfg.policy, ds_meta=dataset.meta)
   residual_policy = ResidualSmolVLAPolicy(base_policy=base_policy, device="cuda")
   ```

2. **动作预测**：
   ```python
   # 原来
   action = predict_action(observation, policy, ...)
   
   # 改为
   action = residual_policy.predict_action(observation, task=task, use_residual=True)
   ```

3. **添加RL训练循环**（可选）：
   ```python
   # 在episode结束后
   if len(trainer.rewards) > 0:
       advantages, returns = trainer.compute_gae(...)
       trainer.update_policy(advantages, returns)
   ```

## 注意事项

1. **奖励函数**：你需要根据你的任务实现真正的奖励函数（在`SimpleRewardFunction`中）

2. **状态特征**：`extract_state_features()`需要根据你的观察空间调整

3. **Value计算**：当前实现中value是占位符，需要实现真正的critic网络

4. **超参数调整**：
   - `residual_action_scale`: 控制残差动作的幅度（默认0.1）
   - `learning_rate`: RL学习率（默认3e-4）
   - `residual_hidden_dim`: 残差网络隐藏层维度（默认256）

## 下一步

1. 运行`example_usage.py`查看基本流程
2. 根据你的任务实现奖励函数
3. 调整超参数
4. 集成到你的实际代码中

## 文件说明

- `residual_smolvla.py`: Residual policy wrapper
- `rl_finetune_smolvla.py`: RL训练器
- `example_usage.py`: 完整使用示例
- `README.md`: 详细文档

