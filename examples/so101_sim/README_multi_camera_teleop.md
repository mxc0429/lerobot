# 多相机遥操作系统使用说明

## 功能概述

这个系统实现了使用真实SO101 Leader机械臂控制MuJoCo仿真环境中的Follower机械臂，同时实时显示三个相机视角的功能。

### 主要特性

- ✅ **真实机械臂控制**: 使用物理SO101 Leader机械臂作为输入设备
- ✅ **仿真环境**: 在MuJoCo中运行高保真机械臂仿真
- ✅ **三相机显示**: 同时显示top_cam、wrist_cam、right_cam三个视角
- ✅ **实时反馈**: 显示FPS、时间戳、连接状态等信息
- ✅ **键盘控制**: 支持暂停、重置、截图等快捷操作
- ✅ **错误恢复**: 显示线程崩溃自动重启，不影响控制

## 系统架构

```
┌─────────────────┐
│  真实Leader臂   │ (物理硬件)
└────────┬────────┘
         │ 串口通信
         ↓
┌─────────────────┐
│  控制循环线程   │ (50Hz)
│  - 读取关节位置 │
│  - 转换为动作   │
│  - 更新仿真     │
└────────┬────────┘
         │ 共享MuJoCo数据
         ↓
┌─────────────────┐
│  显示线程       │ (10Hz)
│  - 渲染相机     │
│  - 添加覆盖层   │
│  - 处理键盘     │
└─────────────────┘
```

## 安装依赖

```bash
# 基础依赖
pip install numpy opencv-python mujoco

# LeRobot依赖
pip install -e .
```

## 快速开始

### 1. 基本使用（默认配置）

```bash
python examples/so101_sim/multi_camera_teleop.py \
    --port /dev/ttyACM0
```

### 2. 自定义相机分辨率

```bash
python examples/so101_sim/multi_camera_teleop.py \
    --port /dev/ttyACM0 \
    --camera_width 640 \
    --camera_height 480
```

### 3. 调整帧率

```bash
python examples/so101_sim/multi_camera_teleop.py \
    --port /dev/ttyACM0 \
    --control_fps 100 \
    --display_fps 30
```

### 4. 使用已有校准

```bash
python examples/so101_sim/multi_camera_teleop.py \
    --port /dev/ttyACM0 \
    --leader_id my_leader_arm \
    --skip_calibration
```

## 命令行参数

### Leader机械臂配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `/dev/ttyACM0` | Leader机械臂串口 |
| `--leader_id` | `None` | Leader校准ID |
| `--leader_calibration_dir` | `None` | 校准文件目录 |
| `--skip_calibration` | `False` | 跳过自动校准 |
| `--no_degrees` | `False` | 使用归一化值而非角度 |

### 仿真环境配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--xml_path` | `so101_block_stacking.xml` | 自定义MuJoCo场景 |
| `--randomize_blocks` | `False` | 随机化方块位置 |

### 显示配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--camera_width` | `320` | 相机视图宽度 |
| `--camera_height` | `240` | 相机视图高度 |
| `--display_fps` | `10` | 显示更新频率(Hz) |
| `--no_overlays` | `False` | 隐藏覆盖层信息 |

### 控制配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--control_fps` | `50` | 控制循环频率(Hz) |
| `--max_duration` | `300` | 最大运行时间(秒) |

## 键盘控制

在显示窗口激活时，可以使用以下快捷键：

| 按键 | 功能 |
|------|------|
| `q` | 退出程序 |
| `p` | 暂停/继续 |
| `r` | 重置环境 |
| `s` | 保存截图 |
| `h` | 显示帮助 |

## 显示界面说明

### 相机布局

```
┌──────────┬──────────┬──────────┐
│ top_cam  │wrist_cam │right_cam │
│          │          │          │
│  俯视图  │  腕部视图│  侧视图  │
└──────────┴──────────┴──────────┘
```

### 覆盖层信息

- **左上角**: 相机名称（绿色）
- **右上角**: FPS计数器（黄色）
- **左下角**: 时间戳
- **右下角**: 连接状态（CONNECTED/DISCONNECTED）
- **中央**: 暂停指示器（暂停时显示）
- **顶部**: 错误信息（出错时显示）

## 故障排除

### Leader机械臂连接失败

**问题**: `Leader连接失败: [Errno 2] No such file or directory: '/dev/ttyACM0'`

**解决方案**:
1. 检查USB连接
2. 查找正确的串口: `ls /dev/tty*`
3. 使用正确的端口: `--port /dev/ttyUSB0`

### 校准问题

**问题**: `Leader机械臂未校准`

**解决方案**:
1. 首次使用时不要加 `--skip_calibration`
2. 按照提示完成校准过程
3. 后续使用可以加 `--skip_calibration` 和 `--leader_id`

### 显示窗口无响应

**问题**: 显示窗口卡住或无响应

**解决方案**:
1. 系统会自动重启显示线程
2. 如果持续出现，检查GPU驱动
3. 降低显示帧率: `--display_fps 5`

### 性能问题

**问题**: FPS过低或延迟高

**解决方案**:
1. 降低相机分辨率: `--camera_width 160 --camera_height 120`
2. 降低显示帧率: `--display_fps 5`
3. 关闭覆盖层: `--no_overlays`

### 相机未找到

**问题**: 显示 "Camera not found"

**解决方案**:
1. 检查MuJoCo XML文件中是否定义了相机
2. 确认相机名称正确: `top_cam`, `wrist_cam`, `right_cam`
3. 使用自定义XML: `--xml_path path/to/your/scene.xml`

## 高级用法

### 自定义相机配置

如果需要使用不同的相机，可以修改代码中的 `camera_names` 列表：

```python
# 在 multi_camera_teleop.py 中
camera_names = ["camera1", "camera2", "camera3"]  # 自定义相机名称
```

### 保存截图

按 `s` 键会将当前画面保存到 `snapshots/` 目录，文件名格式为：
```
snapshot_20231125_143022.png
```

### 性能监控

程序会每10秒打印一次状态信息：
```
运行中 - 步数: 500, 显示FPS: 9.8
```

## 技术细节

### 线程模型

- **主线程**: 初始化和协调
- **控制线程**: 在主线程中运行，读取Leader并更新仿真
- **显示线程**: 独立线程，渲染相机和处理键盘

### 数据流

1. Leader机械臂 → 串口读取 → 关节位置
2. 关节位置 → 转换 → 归一化动作
3. 归一化动作 → MuJoCo → 仿真更新
4. MuJoCo数据 → 相机渲染 → 显示窗口

### 错误处理

- Leader连接错误: 显示错误信息，保持显示运行
- 渲染错误: 显示占位图，继续其他相机
- 显示线程崩溃: 自动重启（最多5次）
- 窗口关闭: 安全退出整个系统

## 示例场景

### 场景1: 方块堆叠任务

```bash
python examples/so101_sim/multi_camera_teleop.py \
    --port /dev/ttyACM0 \
    --randomize_blocks
```

### 场景2: 高分辨率录制

```bash
python examples/so101_sim/multi_camera_teleop.py \
    --port /dev/ttyACM0 \
    --camera_width 640 \
    --camera_height 480 \
    --display_fps 30
```

### 场景3: 快速测试

```bash
python examples/so101_sim/multi_camera_teleop.py \
    --port /dev/ttyACM0 \
    --camera_width 160 \
    --camera_height 120 \
    --display_fps 5 \
    --max_duration 60
```

## 相关文件

- `examples/so101_sim/multi_camera_teleop.py` - 主程序
- `src/lerobot/display/multi_camera_display.py` - 多相机显示模块
- `src/lerobot/teleoperators/so101_leader/` - Leader机械臂驱动
- `src/lerobot/envs/so101_mujoco.py` - MuJoCo环境

## 参考资料

- [LeRobot文档](https://github.com/huggingface/lerobot)
- [MuJoCo文档](https://mujoco.readthedocs.io/)
- [OpenCV文档](https://docs.opencv.org/)
