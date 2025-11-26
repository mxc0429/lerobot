# SO101 仿真示例

本目录包含SO101机械臂的多相机遥操作系统。

## 核心程序

| 文件 | 说明 | 需要硬件 |
|------|------|----------|
| `multi_camera_display_mujoco.py` | 仿真测试 - 同时显示三个相机视角 | ❌ 不需要 |
| `multi_camera_teleop_matplotlib.py` | 真实机械臂遥操作 - 高性能多相机显示 | ✅ 需要SO101 Leader |
| `calibrate_leader_follower.py` | 校准工具 - 提高控制精度 | ✅ 需要SO101 Leader |
| `teleop_record.py` | 数据录制 - 录制训练数据集 | ✅ 需要SO101 Leader |

## 快速开始

### 1. 仿真测试（无需硬件）

```bash
python examples/so101_sim/multi_camera_display_mujoco.py
```

### 2. 真实机械臂遥操作

```bash
python examples/so101_sim/multi_camera_teleop_matplotlib.py --port COM5
```

### 3. 校准（可选，提高精度）

```bash
python examples/so101_sim/calibrate_leader_follower.py --port COM5 \
    --calibration_name my_robot
```

## 详细文档

查看项目根目录的 **README_使用指南.md**

## 键盘控制

- **q** 或 **Esc**: 退出
- **空格**: 暂停/继续
- **r**: 重置环境
- **h**: 显示帮助

## 参数说明

```bash
--port COM5                    # 机械臂串口
--camera_width 320             # 相机宽度
--camera_height 240            # 相机高度
--display_fps 10               # 显示帧率
--control_fps 50               # 控制帧率
--calibration_file path.json   # 校准文件（可选）
```

## 性能配置

| 配置 | 分辨率 | FPS | 适用场景 |
|------|--------|-----|----------|
| 低性能 | 256x192 | 10-20 | 低性能电脑 |
| 推荐 | 320x240 | 8-15 | 日常使用 |
| 高质量 | 480x360 | 5-10 | 演示/录制 |
