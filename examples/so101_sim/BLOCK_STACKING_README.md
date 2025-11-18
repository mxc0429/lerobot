# SO-101 Block Stacking Environment

## 概述

这是一个增强的MuJoCo仿真环境，用于训练机械臂执行积木堆叠任务。环境包含：

- ✅ **桌子**：机械臂工作台面
- ✅ **3个彩色积木块**：红色、蓝色、绿色（可堆叠）
- ✅ **3个相机视角**：
  - `top_cam`：顶部俯视视角
  - `wrist_cam`：腕部相机（安装在夹爪上）
  - `right_cam`：右侧视角
- ✅ **物理仿真**：真实的接触、摩擦和重力
- ✅ **随机化支持**：可随机化积木初始位置

## 文件结构

```
Sim_assets/SO-ARM100/Simulation/SO101/
└── so101_block_stacking.xml          # 新的场景文件（包含桌子、积木、相机）

src/lerobot/envs/so101_mujoco/
└── env.py                             # 更新的环境类（支持多相机和积木管理）

examples/so101_sim/
├── test_block_stacking_env.py         # 测试脚本
├── visualize_cameras.py               # 相机可视化脚本
├── teleop_record_block_stacking.py    # 遥操作数据采集脚本
└── BLOCK_STACKING_README.md           # 本文档
```

## 环境特性

### 1. 多相机支持

环境支持同时使用三个相机视角，模拟真实环境：

```python
env = So101MujocoEnv(
    So101MujocoConfig(
        xml_path="path/to/so101_block_stacking.xml",
        use_camera=True,
        camera_width=256,
        camera_height=256,
        camera_names=["top_cam", "wrist_cam", "right_cam"],  # 三个相机
    )
)
```

观察空间将包含：
- `observation.qpos`: 关节位置 (27维)
- `observation.image_top_cam`: 顶部相机图像 (256x256x3)
- `observation.image_wrist_cam`: 腕部相机图像 (256x256x3)
- `observation.image_right_cam`: 右侧相机图像 (256x256x3)

### 2. 积木块管理

环境提供了积木块位置管理和堆叠检测：

```python
# 获取积木块位置
block_positions = env.get_block_positions()
# 返回: {"block_red": [x, y, z], "block_blue": [x, y, z], "block_green": [x, y, z]}

# 检查堆叠是否成功
is_stacked = env.check_stacking_success()
# 返回: True/False
```

### 3. 随机化

支持在每次reset时随机化积木块位置：

```python
env = So101MujocoEnv(
    So101MujocoConfig(
        xml_path="path/to/so101_block_stacking.xml",
        randomize_blocks=True,  # 启用随机化
        num_blocks=3,
    )
)
```

## 使用方法

### 1. 测试环境

测试环境是否正常工作：

```bash
python3 examples/so101_sim/test_block_stacking_env.py --no_render --num_steps 100
```

带可视化窗口：

```bash
python3 examples/so101_sim/test_block_stacking_env.py --num_steps 500
```

### 2. 可视化相机视角

生成并保存三个相机的图像：

```bash
python3 examples/so101_sim/visualize_cameras.py --output_dir ./camera_views
```

这将生成：
- `top_cam.png` - 顶部视角
- `wrist_cam.png` - 腕部视角
- `right_cam.png` - 右侧视角
- `combined_view.png` - 组合视图

### 3. 遥操作数据采集

使用leader臂控制follower臂，记录演示数据：

```bash
python3 examples/so101_sim/teleop_record_block_stacking.py \
    --repo_id your_username/so101_block_stacking \
    --num_episodes 50 \
    --episode_time_s 60 \
    --fps 10 \
    --port /dev/ttyUSB0 \
    --camera_width 256 \
    --camera_height 256 \
    --randomize_blocks \
    --display
```

参数说明：
- `--repo_id`: Hugging Face数据集仓库ID
- `--num_episodes`: 要记录的演示次数
- `--episode_time_s`: 每次演示的最大时长
- `--fps`: 记录频率
- `--camera_width/height`: 相机分辨率
- `--randomize_blocks`: 每次重置时随机化积木位置
- `--display`: 显示可视化窗口
- `--push_to_hub`: 完成后上传到Hugging Face Hub

## 场景配置

### 桌子尺寸
- 桌面：80cm x 60cm x 4cm
- 位置：机械臂前方25cm处
- 材质：木质纹理，适当摩擦力

### 积木块
- 尺寸：5cm x 5cm x 5cm
- 质量：50g
- 摩擦系数：1.0（易于抓取和堆叠）
- 颜色：红色、蓝色、绿色（便于视觉识别）

### 相机位置
1. **顶部相机** (`top_cam`)
   - 位置：(0.25, 0, 0.8) - 桌子正上方80cm
   - 朝向：垂直向下
   - 视野：45度

2. **腕部相机** (`wrist_cam`)
   - 位置：相对夹爪固定
   - 朝向：跟随夹爪运动
   - 视野：60度（更宽视野）

3. **右侧相机** (`right_cam`)
   - 位置：(0.25, -0.6, 0.3) - 右侧60cm，高度30cm
   - 朝向：斜向工作区域
   - 视野：45度

## 下一步：训练SmolVLA模型

### 1. 数据采集建议

- **最少演示数量**：50-100个成功的堆叠演示
- **多样性**：
  - 不同的初始积木位置（使用`--randomize_blocks`）
  - 不同的堆叠顺序
  - 不同的抓取方式
- **质量**：确保演示动作流畅，避免碰撞和失败

### 2. 训练配置

创建训练配置文件 `configs/policy/smolvla_block_stacking.yaml`：

```yaml
policy:
  name: smolvla
  pretrained_model_name_or_path: "smolvla_base"
  
dataset:
  repo_id: "your_username/so101_block_stacking"
  image_keys:
    - "observation.image_top_cam"
    - "observation.image_wrist_cam"
    - "observation.image_right_cam"
  
training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 1e-4
  gradient_accumulation_steps: 4
```

### 3. 训练命令

```bash
lerobot-train \
    --config_path configs/policy/smolvla_block_stacking.yaml \
    --output_dir outputs/smolvla_block_stacking \
    --wandb_project so101_block_stacking
```

### 4. 评估模型

创建评估脚本来测试训练好的模型：

```python
from lerobot.common.policies.smolvla import SmolVLAPolicy
from lerobot.envs.so101_mujoco import So101MujocoEnv, So101MujocoConfig

# 加载模型
policy = SmolVLAPolicy.from_pretrained("path/to/checkpoint")

# 创建环境
env = So101MujocoEnv(So101MujocoConfig(...))

# 运行评估
for episode in range(num_eval_episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    success = env.check_stacking_success()
    print(f"Episode {episode}: {'Success' if success else 'Failed'}")
```

## 技术细节

### 物理参数

- **重力**：-9.81 m/s²
- **时间步长**：0.002s
- **控制频率**：50Hz (0.02s)
- **接触参数**：
  - solimp: [0.99, 0.99, 0.001]
  - solref: [0.01, 1]

### 关节配置

环境包含27个自由度：
- 6个机械臂关节（shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper）
- 21个积木块自由度（3个积木 × 7 DOF freejoint）

### 观察空间维度

- 关节位置：27维
- 每个相机图像：256×256×3 = 196,608维
- 总计：3个相机 = 589,824维图像数据 + 27维状态

## 故障排除

### 问题1：相机图像全黑

**原因**：相机位置或朝向配置错误

**解决**：检查XML文件中的相机定义，确保pos和quat参数正确

### 问题2：积木块掉落或穿透桌面

**原因**：物理参数不合适

**解决**：调整积木块的质量、摩擦力和接触参数

### 问题3：机械臂无法到达积木

**原因**：积木位置超出工作空间

**解决**：调整`_randomize_blocks()`中的x_min, x_max, y_min, y_max范围

### 问题4：训练时内存不足

**原因**：三个高分辨率相机图像占用大量内存

**解决**：
- 降低相机分辨率（如128×128）
- 减小batch_size
- 使用梯度累积

## 参考资料

- [LeRobot文档](https://huggingface.co/docs/lerobot)
- [SmolVLA论文](https://arxiv.org/abs/2506.01844)
- [MuJoCo文档](https://mujoco.readthedocs.io/)
- [SO-101机械臂](https://huggingface.co/docs/lerobot/so101)

## 贡献

如有问题或改进建议，欢迎提交Issue或Pull Request。
