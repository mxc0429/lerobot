# 快速入门指南

## 5分钟快速开始

### 步骤1: 测试显示系统（无需硬件）

首先测试显示系统是否正常工作：

```bash
# Linux/Mac
cd /path/to/lerobot
python examples/so101_sim/multi_camera_display_test.py

# Windows
cd C:\path\to\lerobot
python examples\so101_sim\multi_camera_display_test.py
```

你应该看到：
- 一个窗口显示三个相机视角
- 机械臂在仿真中随机移动
- 左上角显示相机名称
- 右上角显示FPS

**键盘测试**:
- 按 `p` 暂停/继续
- 按 `r` 重置环境
- 按 `s` 保存截图（保存到 `snapshots/` 目录）
- 按 `q` 退出

### 步骤2: 连接真实机械臂

如果你有SO101 Leader机械臂：

#### Windows

```bash
# 1. 找到串口号
# 打开设备管理器 → 端口(COM和LPT) → 查看COM号，例如COM3

# 2. 运行遥操作
python examples\so101_sim\multi_camera_teleop.py --port COM3
```

#### Linux

```bash
# 1. 找到串口
ls /dev/tty* | grep -E "(ACM|USB)"
# 通常是 /dev/ttyACM0 或 /dev/ttyUSB0

# 2. 添加串口权限（首次使用）
sudo usermod -a -G dialout $USER
# 注销并重新登录

# 3. 运行遥操作
python examples/so101_sim/multi_camera_teleop.py --port /dev/ttyACM0
```

#### Mac

```bash
# 1. 找到串口
ls /dev/tty.* | grep usb
# 通常是 /dev/tty.usbmodem* 或 /dev/tty.usbserial*

# 2. 运行遥操作
python examples/so101_sim/multi_camera_teleop.py --port /dev/tty.usbmodem14201
```

### 步骤3: 首次校准

第一次连接Leader机械臂时，系统会自动进行校准：

1. 按照屏幕提示移动机械臂到各个极限位置
2. 校准完成后，数据会自动保存
3. 下次使用时可以跳过校准：

```bash
python examples/so101_sim/multi_camera_teleop.py \
    --port /dev/ttyACM0 \
    --skip_calibration
```

## 常用命令

### 标准模式（推荐）

```bash
python examples/so101_sim/multi_camera_teleop.py --port COM3
```

### 低性能模式（电脑较慢）

```bash
python examples/so101_sim/multi_camera_teleop.py \
    --port COM3 \
    --camera_width 160 \
    --camera_height 120 \
    --display_fps 5
```

### 高性能模式（电脑较快）

```bash
python examples/so101_sim/multi_camera_teleop.py \
    --port COM3 \
    --camera_width 640 \
    --camera_height 480 \
    --display_fps 30 \
    --control_fps 100
```

### 使用快捷脚本

#### Windows

```bash
# 标准模式
run_teleop.bat

# 指定串口
run_teleop.bat --port COM5

# 测试模式
run_teleop.bat --test

# 低性能模式
run_teleop.bat --low-perf
```

#### Linux/Mac

```bash
# 添加执行权限（首次）
chmod +x examples/so101_sim/run_teleop.sh

# 标准模式
./examples/so101_sim/run_teleop.sh

# 指定串口
./examples/so101_sim/run_teleop.sh --port /dev/ttyUSB0

# 测试模式
./examples/so101_sim/run_teleop.sh --test

# 低性能模式
./examples/so101_sim/run_teleop.sh --low-perf
```

## 键盘控制速查表

| 按键 | 功能 | 说明 |
|------|------|------|
| `q` | 退出 | 安全关闭所有连接 |
| `p` | 暂停/继续 | 暂停控制循环，显示继续 |
| `r` | 重置 | 重置仿真环境 |
| `s` | 截图 | 保存到 snapshots/ 目录 |
| `h` | 帮助 | 在终端显示帮助信息 |

## 显示界面说明

```
┌─────────────────────────────────────────────────────────────┐
│ top_cam          wrist_cam         right_cam      FPS: 10.2 │
│ ┌──────────┐    ┌──────────┐     ┌──────────┐              │
│ │          │    │          │     │          │              │
│ │  俯视图  │    │  腕部图  │     │  侧视图  │              │
│ │          │    │          │     │          │              │
│ └──────────┘    └──────────┘     └──────────┘              │
│ 14:32:15.234                           CONNECTED            │
└─────────────────────────────────────────────────────────────┘
```

## 故障排除

### 问题1: 找不到串口

**症状**: `No such file or directory: '/dev/ttyACM0'`

**解决**:
```bash
# Linux
ls /dev/tty* | grep -E "(ACM|USB)"

# Windows (PowerShell)
Get-WmiObject Win32_SerialPort | Select-Object Name,DeviceID

# Mac
ls /dev/tty.* | grep usb
```

### 问题2: 权限被拒绝

**症状**: `Permission denied: '/dev/ttyACM0'`

**解决** (Linux):
```bash
sudo usermod -a -G dialout $USER
# 注销并重新登录
```

### 问题3: 显示窗口卡顿

**症状**: FPS < 5

**解决**:
```bash
# 使用低性能模式
python examples/so101_sim/multi_camera_teleop.py \
    --port COM3 \
    --camera_width 160 \
    --camera_height 120 \
    --display_fps 5 \
    --no_overlays
```

### 问题4: Leader连接失败

**症状**: `Leader连接失败`

**检查清单**:
- [ ] USB线已连接
- [ ] 机械臂已上电
- [ ] 串口号正确
- [ ] 驱动已安装（Windows）
- [ ] 权限已设置（Linux）

### 问题5: 相机显示黑屏

**症状**: 相机视图全黑或显示 "Camera not found"

**解决**:
1. 检查MuJoCo XML文件中是否定义了相机
2. 确认相机名称正确
3. 尝试使用默认场景（不指定 `--xml_path`）

## 性能优化建议

### 电脑配置较低

```bash
python examples/so101_sim/multi_camera_teleop.py \
    --port COM3 \
    --camera_width 160 \
    --camera_height 120 \
    --display_fps 5 \
    --control_fps 30 \
    --no_overlays
```

### 电脑配置中等（推荐）

```bash
python examples/so101_sim/multi_camera_teleop.py \
    --port COM3 \
    --camera_width 320 \
    --camera_height 240 \
    --display_fps 10 \
    --control_fps 50
```

### 电脑配置较高

```bash
python examples/so101_sim/multi_camera_teleop.py \
    --port COM3 \
    --camera_width 640 \
    --camera_height 480 \
    --display_fps 30 \
    --control_fps 100
```

## 下一步

- 阅读 [详细文档](README_multi_camera_teleop.md)
- 查看 [实现总结](IMPLEMENTATION_SUMMARY.md)
- 运行 [单元测试](../../tests/display/test_multi_camera_display.py)
- 自定义 [配置文件](configs/teleop_config_example.yaml)

## 获取帮助

如果遇到问题：

1. 查看终端输出的错误信息
2. 检查 [故障排除](#故障排除) 部分
3. 阅读 [详细文档](README_multi_camera_teleop.md)
4. 提交Issue到GitHub

## 示例输出

成功运行时，你应该看到类似的输出：

```
======================================================================
多相机遥操作系统
======================================================================
环境信息:
  - 关节数: 7
  - 执行器数: 7
  - 执行器名称: ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']

显示配置:
  - 相机数量: 3
  - 相机名称: ['top_cam', 'wrist_cam', 'right_cam']
  - 分辨率: 320x240
  - 显示帧率: 10 Hz

控制配置:
  - 控制帧率: 50 Hz
  - Leader端口: /dev/ttyACM0
  - 使用角度制: True

键盘控制:
  - q: 退出
  - p: 暂停/继续
  - r: 重置环境
  - s: 保存截图
  - h: 显示帮助
======================================================================

[INFO] 启动遥操作系统...
[INFO] Leader机械臂连接成功
[INFO] 运行中 - 步数: 500, 显示FPS: 9.8
[INFO] 运行中 - 步数: 1000, 显示FPS: 10.1
```

祝你使用愉快！🎉
