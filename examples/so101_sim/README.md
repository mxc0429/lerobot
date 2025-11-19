# SO-101 仿真环境

SO-101 机械臂的 MuJoCo 仿真环境，支持方块堆叠任务。

## 文件说明

- `test_sim_setup.py` - 测试仿真环境是否正确安装
- `sim_random_actions.py` - 随机动作测试，验证环境功能
- `teleop_record.py` - 遥操作数据采集（需要真实硬件）

## 快速开始

### 1. 测试环境

```bash
# 基础测试（无可视化）
python examples/so101_sim/test_sim_setup.py

# 带可视化窗口
python examples/so101_sim/test_sim_setup.py --render
```

### 2. 随机动作测试

```bash
# 无可视化（更快）
python examples/so101_sim/sim_random_actions.py --num_episodes 2 --max_steps 100

# 带可视化窗口
python examples/so101_sim/sim_random_actions.py --num_episodes 2 --max_steps 100 --render

# 随机化方块位置
python examples/so101_sim/sim_random_actions.py --randomize_blocks --render
```

### 3. 遥操作数据采集（需要硬件）

```bash
python examples/so101_sim/teleop_record.py \
    --repo_id your_username/so101_dataset \
    --num_episodes 10 \
    --port /dev/ttyACM0 \
    --leader_id my_leader_arm \
    --leader_calibration_dir ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader
```

## 环境特性

- **机械臂**: SO-101 6自由度机械臂
- **任务**: 方块堆叠（3个彩色方块）
- **相机**: 3个视角（顶部、手腕、侧面）
- **物理引擎**: MuJoCo
- **控制频率**: 50Hz

## 依赖安装

```bash
pip install mujoco gymnasium numpy
```

## 参数说明

### sim_random_actions.py

- `--num_episodes` - 测试回合数
- `--max_steps` - 每回合最大步数
- `--render` - 显示可视化窗口
- `--randomize_blocks` - 随机化方块初始位置
- `--action_scale` - 动作缩放因子 (0-1)
- `--camera_width/height` - 相机分辨率

### test_sim_setup.py

- `--render` - 显示可视化窗口

## 环境配置

环境配置在 `src/lerobot/envs/so101_mujoco/env.py` 中的 `So101MujocoConfig` 类：

```python
So101MujocoConfig(
    xml_path=str(xml_file_path),
    render_mode="window",  # "window" | None
    use_camera=True,
    camera_width=256,
    camera_height=256,
    camera_names=["top_cam", "wrist_cam", "right_cam"],
    randomize_blocks=False,
    num_blocks=3,
)
```

## 故障排除

### 问题：找不到 XML 文件
确保 `Sim_assets/SO-ARM100/Simulation/SO101/` 目录存在且包含 XML 文件。

### 问题：MuJoCo 窗口不显示
添加 `--render` 参数。

### 问题：相机渲染慢
降低分辨率或减少相机数量。
