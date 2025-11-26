# SO101多相机遥操作系统 - 使用指南

## 功能

- ✅ 真实SO101机械臂控制仿真环境
- ✅ 同时显示三个相机视角（top_cam, wrist_cam, right_cam）
- ✅ 高性能渲染（8-15 FPS @ 320x240）
- ✅ 可选的校准系统（提高控制精度）

## 快速开始

### 1. 安装依赖

```bash
# Windows
install_dependencies.bat

# 或手动安装
pip install numpy mujoco matplotlib feetech-servo-sdk pyserial
```

### 2. 运行程序

#### 仿真测试（无需机械臂）

```bash
python examples/so101_sim/multi_camera_display_mujoco.py
```

#### 真实机械臂遥操作

```bash
python examples/so101_sim/multi_camera_teleop_matplotlib.py --port COM5
```

## 键盘控制

- **q** 或 **Esc**: 退出
- **空格**: 暂停/继续
- **r**: 重置环境
- **h**: 显示帮助

## 性能配置

### 推荐配置（平衡）

```bash
python examples/so101_sim/multi_camera_teleop_matplotlib.py --port COM5 \
    --camera_width 320 \
    --camera_height 240 \
    --display_fps 10
```

### 高质量配置

```bash
python examples/so101_sim/multi_camera_teleop_matplotlib.py --port COM5 \
    --camera_width 480 \
    --camera_height 360 \
    --display_fps 8
```

### 低性能配置

```bash
python examples/so101_sim/multi_camera_teleop_matplotlib.py --port COM5 \
    --camera_width 256 \
    --camera_height 192 \
    --display_fps 5
```

## 校准系统（可选）

### 为什么需要校准？

校准可以消除真实机械臂和仿真机械臂之间的角度误差，提高控制精度到 < 0.5°。

### 校准步骤

#### 1. 运行校准程序

```bash
python examples/so101_sim/calibrate_leader_follower.py --port COM5 \
    --calibration_name my_robot
```

#### 2. 按提示操作

- 程序会要求移动每个关节到不同位置
- 每个关节采集5个数据点
- 建议位置：最小、25%、50%、75%、最大

#### 3. 使用校准文件

```bash
python examples/so101_sim/multi_camera_teleop_matplotlib.py --port COM5 \
    --calibration_file calibrations/my_robot.json
```

### 不使用校准

如果不需要校准或校准效果不好，直接运行不加 `--calibration_file` 参数：

```bash
python examples/so101_sim/multi_camera_teleop_matplotlib.py --port COM5
```

## 常见问题

### Q: 找不到串口

**Windows**: 打开设备管理器 → 端口(COM和LPT) → 查看COM号

**Linux**: `ls /dev/tty* | grep -E "(ACM|USB)"`

### Q: 缺少 scservo_sdk

```bash
pip install feetech-servo-sdk pyserial
```

### Q: FPS太低

降低分辨率或显示帧率：

```bash
python examples/so101_sim/multi_camera_teleop_matplotlib.py --port COM5 \
    --camera_width 256 \
    --camera_height 192 \
    --display_fps 5
```

### Q: 控制有误差

使用校准系统或检查机械臂连接。

## 文件说明

### 核心程序

- `examples/so101_sim/multi_camera_display_mujoco.py` - 仿真测试程序
- `examples/so101_sim/multi_camera_teleop_matplotlib.py` - 遥操作程序
- `examples/so101_sim/calibrate_leader_follower.py` - 校准工具

### 核心模块

- `src/lerobot/display/multi_camera_display.py` - 多相机显示模块

### 工具脚本

- `install_dependencies.bat` - 依赖安装脚本
- `install_scservo_sdk.bat` - Feetech SDK安装脚本

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | /dev/ttyACM0 | 机械臂串口 |
| `--camera_width` | 320 | 相机宽度 |
| `--camera_height` | 240 | 相机高度 |
| `--display_fps` | 10 | 显示帧率 |
| `--control_fps` | 50 | 控制帧率 |
| `--calibration_file` | None | 校准文件路径 |

## 性能参考

| 分辨率 | FPS | 画质 | 推荐场景 |
|--------|-----|------|----------|
| 256x192 | 10-20 | 中 | 低性能电脑 |
| 320x240 | 8-15 | 好 | 日常使用 ⭐ |
| 480x360 | 5-10 | 高 | 高质量需求 |
| 640x480 | 3-8 | 超高 | 演示/录制 |
