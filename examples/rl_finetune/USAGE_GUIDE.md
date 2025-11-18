# 使用指南：SO101机械臂堆叠任务的RL微调

## 快速开始

### 1. 使用堆叠任务的奖励函数

在你的代码中，将`SimpleRewardFunction`替换为`StackingRewardFunction`：

```python
from stacking_reward import StackingRewardFunction

# 创建堆叠任务的奖励函数
reward_fn = StackingRewardFunction(
    success_reward=50.0,
    step_penalty=-0.01,
    distance_reward_scale=0.5,
    grasp_reward=5.0,
    stacking_progress_scale=2.0,
    height_reward_scale=1.0,
    stability_reward=10.0,
)
```

### 2. 实现block位置提取

`StackingRewardFunction`需要知道方块的位置。你有几个选择：

#### 选项A：从info字典获取（推荐，如果可用）

如果你的环境或机器人系统提供了block位置信息：

```python
# 在info字典中提供block位置
info = {
    "block1_pos": [x1, y1, z1],  # 第一个方块的位置
    "block2_pos": [x2, y2, z2],  # 第二个方块的位置
    "success": False,  # 任务是否成功
}

reward = reward_fn.compute_reward(
    observation=observation,
    action=action,
    next_observation=next_observation,
    done=done,
    info=info,
)
```

#### 选项B：从observation state获取

如果你的observation state包含block位置：

```python
# 假设state的前6个元素是机器人状态，接下来6个是block位置
# state[6:9] = block1_pos, state[9:12] = block2_pos
# 修改stacking_reward.py中的_extract_block_positions方法
```

#### 选项C：使用计算机视觉（需要实现）

使用三个相机（顶部、腕部、侧面）来检测方块位置：

```python
from stacking_reward import VisionBasedStackingReward

reward_fn = VisionBasedStackingReward(...)

# 需要实现block检测
# 可以使用：
# - YOLO等目标检测模型
# - 颜色分割（如果方块颜色固定）
# - 深度相机信息（如果有）
```

### 3. 使用改进的状态特征提取

`ResidualSmolVLAPolicy`现在会自动：
- 从observation state提取机器人状态
- 从三个相机图像提取视觉特征
- 组合成特征向量

确保你的observation包含：
- `"state"`: 机器人状态（包含位置、速度等）
- `"observation.images.camera1"`: 顶部相机图像
- `"observation.images.camera2"`: 腕部相机图像
- `"observation.images.camera3"`: 侧面相机图像

### 4. 使用Critic网络计算Value

现在`ResidualMLP`包含critic网络，会自动计算value估计：

```python
# 在存储rollout时
value = residual_policy.get_value(observation, action).item()
trainer.values.append(value)
```

### 5. 完整的训练循环示例

```python
from residual_smolvla import ResidualSmolVLAPolicy
from rl_finetune_smolvla import ResidualRLTrainer
from stacking_reward import StackingRewardFunction

# 初始化
residual_policy = ResidualSmolVLAPolicy(base_policy=base_policy, device="cuda")
trainer = ResidualRLTrainer(residual_policy=residual_policy, learning_rate=3e-4)
reward_fn = StackingRewardFunction()

# 训练循环
for episode in range(num_episodes):
    residual_policy.reset()
    trainer.reset_rollout_storage()
    reward_fn.reset()
    
    for step in range(episode_length):
        # 获取观察
        observation = robot.get_observation()
        
        # 预测动作
        action = residual_policy.predict_action(observation, task=task)
        
        # 执行动作
        robot.send_action(action)
        
        # 获取下一个观察
        next_observation = robot.get_observation()
        
        # 计算奖励（需要提供block位置信息）
        info = {
            "block1_pos": get_block1_position(),  # 你需要实现这个
            "block2_pos": get_block2_position(),   # 你需要实现这个
            "success": check_success(),             # 你需要实现这个
        }
        reward = reward_fn.compute_reward(
            observation=observation,
            action=action,
            next_observation=next_observation,
            done=False,
            info=info,
        )
        
        # 计算value
        value = residual_policy.get_value(observation, action).item()
        
        # 存储到rollout
        trainer.observations.append(observation)
        trainer.actions.append(action.cpu().numpy())
        trainer.rewards.append(reward)
        trainer.values.append(value)
        trainer.log_probs.append(0.0)
        trainer.dones.append(False)
    
    # 更新策略
    advantages, returns = trainer.compute_gae(
        trainer.rewards, trainer.values, trainer.dones
    )
    update_info = trainer.update_policy(advantages, returns)
    print(f"Episode {episode}: reward={sum(trainer.rewards):.2f}, "
          f"policy_loss={update_info['policy_loss']:.4f}, "
          f"value_loss={update_info['value_loss']:.4f}")
```

## 关键实现点

### 1. Block位置检测

你需要实现获取block位置的方法。几个选项：

**方法1：使用状态信息（如果可用）**
```python
def get_block_positions(observation):
    state = observation["state"]
    # 假设state包含block位置
    block1_pos = state[6:9]  # 调整索引
    block2_pos = state[9:12]
    return block1_pos, block2_pos
```

**方法2：使用计算机视觉**
```python
import cv2
import numpy as np

def detect_blocks_in_image(image, color_range):
    """使用颜色分割检测方块"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, color_range[0], color_range[1])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大的轮廓（假设是方块）
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None

def get_block_positions_from_cameras(observation):
    """从三个相机检测方块位置"""
    # 使用相机标定将像素坐标转换为3D坐标
    # 这需要相机标定参数
    pass
```

**方法3：使用外部跟踪系统**
```python
# 如果使用Vicon、OptiTrack等运动捕捉系统
def get_block_positions_from_tracker():
    # 从跟踪系统获取位置
    pass
```

### 2. 成功判断

实现任务成功的判断：

```python
def check_success(block1_pos, block2_pos, threshold=0.03):
    """
    判断是否成功堆叠
    
    Args:
        block1_pos: 第一个方块位置（要堆叠的方块）
        block2_pos: 第二个方块位置（底座方块）
        threshold: 成功阈值（米）
    
    Returns:
        bool: 是否成功
    """
    # 检查水平对齐（XY距离）
    xy_distance = np.linalg.norm(block1_pos[:2] - block2_pos[:2])
    
    # 检查垂直位置（block1应该在block2上方）
    height_diff = block1_pos[2] - block2_pos[2]
    
    # 成功条件：水平对齐且block1在block2上方
    return xy_distance < threshold and height_diff > 0.01
```

### 3. 调整超参数

根据实际效果调整超参数（参考`hyperparameters.md`）：

```python
# 如果动作不稳定
residual_action_scale = 0.05  # 降低

# 如果学习太慢
residual_action_scale = 0.15  # 增加
learning_rate = 5e-4          # 增加

# 如果智能体不完成任务
success_reward = 100.0        # 大幅增加
```

## 调试技巧

1. **打印奖励组成**：修改`StackingRewardFunction.compute_reward()`来打印各个奖励分量
2. **可视化block位置**：在图像上绘制检测到的block位置
3. **监控value估计**：确保value估计合理（不应该太大或太小）
4. **检查动作幅度**：打印残差动作的幅度，确保在合理范围内

## 常见问题

### Q: 奖励总是负的
**A:** 检查block位置提取是否正确，确保距离奖励在起作用

### Q: 智能体不抓取方块
**A:** 增加`grasp_reward`，确保gripper状态正确提取

### Q: 方块位置检测不准确
**A:** 考虑使用多个相机的信息融合，或使用深度相机

### Q: 训练不稳定
**A:** 降低`residual_action_scale`和`learning_rate`，增加`update_frequency`


