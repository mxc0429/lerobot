# 超参数调优指南

## 针对SO101机械臂堆叠任务的推荐超参数

### 1. Residual Policy 超参数

```python
residual_hidden_dim = 256        # 残差网络隐藏层维度
residual_num_layers = 2          # 残差网络层数
residual_action_scale = 0.1      # 残差动作缩放因子（关键参数！）
```

**`residual_action_scale` 调优建议：**
- **初始值：0.1** - 保守的开始，只允许小的修正
- **如果学习太慢**：增加到 0.15-0.2
- **如果动作不稳定**：降低到 0.05-0.08
- **精细操作任务**：0.05-0.1
- **快速移动任务**：0.1-0.2

### 2. RL训练超参数

```python
learning_rate = 3e-4             # 学习率
gamma = 0.99                     # 折扣因子
gae_lambda = 0.95                # GAE lambda
clip_epsilon = 0.2               # PPO clip epsilon
value_coef = 0.5                 # Value loss系数
entropy_coef = 0.01              # Entropy bonus系数
```

**学习率调优：**
- **初始值：3e-4** - 标准PPO学习率
- **如果训练不稳定**：降低到 1e-4
- **如果学习太慢**：增加到 5e-4（但要小心）
- **Actor和Critic可以分开**：
  ```python
  actor_lr = 3e-4
  critic_lr = 1e-3  # Critic通常可以学习更快
  ```

### 3. 奖励函数超参数

```python
success_reward = 50.0            # 成功奖励（应该远大于其他奖励）
step_penalty = -0.01             # 每步惩罚（鼓励效率）
distance_reward_scale = 0.5      # 距离奖励缩放
grasp_reward = 5.0               # 抓取奖励
stacking_progress_scale = 2.0    # 堆叠进度奖励
height_reward_scale = 1.0        # 高度奖励
stability_reward = 10.0           # 稳定性奖励
```

**奖励函数调优原则：**
1. **成功奖励应该远大于其他奖励之和**：确保智能体优先完成任务
2. **步惩罚要小**：避免智能体过早放弃
3. **距离奖励**：根据你的任务调整，如果方块距离较远，可以增加
4. **堆叠进度奖励**：鼓励对齐，可以适当增加

### 4. 训练流程超参数

```python
num_episodes = 5                 # 初始可以少一些，测试流程
episode_time_s = 30              # 每个episode的时间
update_frequency = 1             # 每N个episode更新一次策略
checkpoint_interval = 5           # 每N个episode保存checkpoint
```

### 5. 完整配置示例

```python
# 针对堆叠任务的推荐配置
config = {
    # Residual Policy
    "residual_hidden_dim": 256,
    "residual_num_layers": 2,
    "residual_action_scale": 0.1,  # 从0.1开始，根据效果调整
    
    # RL Training
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    
    # Reward Function
    "success_reward": 50.0,
    "step_penalty": -0.01,
    "distance_reward_scale": 0.5,
    "grasp_reward": 5.0,
    "stacking_progress_scale": 2.0,
    "height_reward_scale": 1.0,
    "stability_reward": 10.0,
    
    # Training
    "num_episodes": 10,  # 开始时可以少一些
    "update_frequency": 1,
    "checkpoint_interval": 5,
}
```

## 调优流程

### 第一步：基础测试
1. 使用默认超参数运行几个episode
2. 观察：
   - 动作是否稳定？
   - 奖励是否在增加？
   - 是否有明显的改进？

### 第二步：调整动作尺度
如果动作不稳定或太大：
- 降低 `residual_action_scale` 到 0.05-0.08

如果学习太慢或没有改进：
- 增加 `residual_action_scale` 到 0.15-0.2

### 第三步：调整学习率
如果训练不稳定（奖励波动大）：
- 降低 `learning_rate` 到 1e-4

如果学习太慢：
- 可以尝试增加到 5e-4，但要密切观察

### 第四步：调整奖励函数
如果智能体不完成任务：
- 增加 `success_reward`
- 增加中间奖励（`grasp_reward`, `stacking_progress_scale`）

如果智能体过早放弃：
- 减少 `step_penalty`
- 增加中间奖励

### 第五步：网络容量
如果任务复杂且学习慢：
- 增加 `residual_hidden_dim` 到 512
- 增加 `residual_num_layers` 到 3

## 常见问题

### Q: 动作不稳定，机器人抖动
**A:** 
- 降低 `residual_action_scale` 到 0.05-0.08
- 降低 `learning_rate` 到 1e-4
- 增加 `entropy_coef` 到 0.02-0.05（增加探索但可能减慢学习）

### Q: 学习太慢，没有改进
**A:**
- 增加 `residual_action_scale` 到 0.15-0.2
- 检查奖励函数是否正确
- 增加中间奖励（`grasp_reward`, `stacking_progress_scale`）
- 增加网络容量（`residual_hidden_dim`）

### Q: 智能体不完成任务
**A:**
- 大幅增加 `success_reward`（例如到 100.0）
- 增加中间奖励，让智能体知道它在正确的方向上
- 检查奖励函数是否正确计算

### Q: 训练不稳定，奖励波动大
**A:**
- 降低 `learning_rate`
- 增加 `update_frequency`（更少但更稳定的更新）
- 使用更大的batch size（如果可能）

## 监控指标

在训练过程中，关注以下指标：

1. **Episode奖励**：应该逐渐增加
2. **成功率**：完成任务的episode比例
3. **Value loss**：应该逐渐减小
4. **Policy loss**：应该逐渐减小（可能为负）
5. **动作幅度**：残差动作的幅度，应该适中

## 实验建议

1. **从保守的超参数开始**：`residual_action_scale=0.1`, `learning_rate=3e-4`
2. **一次只改变一个参数**：便于理解每个参数的影响
3. **记录实验结果**：保存不同超参数配置的结果
4. **使用checkpoint**：定期保存，可以回退到之前的版本

