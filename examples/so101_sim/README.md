# SO101 仿真示例

本目录包含SO101机械臂的各种仿真示例和遥操作程序。

## 📁 文件说明

### 主要程序

| 文件 | 说明 | 需要硬件 |
|------|------|----------|
| `multi_camera_teleop.py` | **多相机遥操作系统** - 使用真实机械臂控制仿真，实时显示三个相机视角 | ✅ 需要SO101 Leader |
| `multi_camera_display_test.py` | **显示系统测试** - 纯仿真测试，无需真实机械臂 | ❌ 不需要 |
| `teleop_record.py` | 遥操作录制 - 录制数据集 | ✅ 需要SO101 Leader |
| `sim_random_actions.py` | 随机动作测试 - 测试仿真环境 | ❌ 不需要 |
| `test_sim_setup.py` | 仿真环境设置测试 | ❌ 不需要 |

### 启动脚本

| 文件 | 平台 | 说明 |
|------|------|------|
| `run_teleop.bat` | Windows | 快速启动脚本 |
| `run_teleop.sh` | Linux/Mac | 快速启动脚本 |

### 文档

| 文件 | 语言 | 说明 |
|------|------|------|
| `快速入门.md` | 中文 | 5分钟快速开始教程 |
| `QUICKSTART.md` | English | Quick start guide |
| `README_multi_camera_teleop.md` | English | 详细使用文档 |
| `IMPLEMENTATION_SUMMARY.md` | English | 实现总结和技术细节 |

### 配置

| 文件 | 说明 |
|------|------|
| `configs/teleop_config_example.yaml` | 配置文件示例 |

## 🚀 快速开始

### 1. 测试显示系统（推荐首次使用）

无需任何硬件，测试显示功能：

```bash
python examples/so101_sim/multi_camera_display_test.py
```

### 2. 使用真实机械臂

连接SO101 Leader机械臂进行遥操作：

```bash
# Windows
python examples\so101_sim\multi_camera_teleop.py --port COM3

# Linux/Mac
python examples/so101_sim/multi_camera_teleop.py --port /dev/ttyACM0
```

### 3. 使用快捷脚本

```bash
# Windows
run_teleop.bat --test

# Linux/Mac
./run_teleop.sh --test
```

## 📖 详细文档

- **中文用户**: 请阅读 [快速入门.md](快速入门.md)
- **English users**: Please read [QUICKSTART.md](QUICKSTART.md)
- **详细文档**: [README_multi_camera_teleop.md](README_multi_camera_teleop.md)
- **技术细节**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## ✨ 主要特性

### 多相机遥操作系统

- ✅ 真实SO101机械臂控制仿真环境
- ✅ 三相机实时显示（top_cam, wrist_cam, right_cam）
- ✅ 丰富的覆盖层信息（FPS、时间戳、状态等）
- ✅ 键盘控制（暂停、重置、截图等）
- ✅ 自动错误恢复
- ✅ 可配置性能模式

## 🎮 键盘控制

| 按键 | 功能 |
|------|------|
| `q` | 退出 |
| `p` | 暂停/继续 |
| `r` | 重置环境 |
| `s` | 保存截图 |
| `h` | 显示帮助 |

## ⚙️ 性能模式

### 低性能模式（电脑较慢）

```bash
./run_teleop.sh --low-perf
```

- 分辨率: 160x120
- 显示帧率: 5 Hz
- 控制帧率: 30 Hz

### 标准模式（推荐）

```bash
./run_teleop.sh
```

- 分辨率: 320x240
- 显示帧率: 10 Hz
- 控制帧率: 50 Hz

### 高性能模式（电脑较快）

```bash
./run_teleop.sh --high-perf
```

- 分辨率: 640x480
- 显示帧率: 30 Hz
- 控制帧率: 100 Hz

## 🔧 常见问题

### 找不到串口？

**Windows**:
- 打开设备管理器 → 端口(COM和LPT)
- 查看COM号（例如：COM3）

**Linux**:
```bash
ls /dev/tty* | grep -E "(ACM|USB)"
```

**Mac**:
```bash
ls /dev/tty.* | grep usb
```

### 权限被拒绝？（Linux）

```bash
sudo usermod -a -G dialout $USER
# 注销并重新登录
```

### 显示卡顿？

使用低性能模式：
```bash
./run_teleop.sh --low-perf
```

## 📊 系统要求

### 最低配置

- CPU: 双核 2.0 GHz
- 内存: 4 GB
- Python: 3.8+
- 操作系统: Windows 10 / Ubuntu 20.04 / macOS 10.15+

### 推荐配置

- CPU: 四核 3.0 GHz
- 内存: 8 GB
- GPU: 支持OpenGL 3.3+
- Python: 3.10+

## 🛠️ 依赖安装

```bash
# 基础依赖
pip install numpy opencv-python mujoco

# LeRobot依赖
pip install -e .

# 可选：Leader机械臂通信
pip install pyserial
```

## 📝 示例场景

### 场景1: 方块堆叠任务

```bash
python examples/so101_sim/multi_camera_teleop.py \
    --port COM3 \
    --randomize_blocks
```

### 场景2: 高分辨率录制

```bash
python examples/so101_sim/multi_camera_teleop.py \
    --port COM3 \
    --camera_width 640 \
    --camera_height 480 \
    --display_fps 30
```

### 场景3: 快速测试

```bash
python examples/so101_sim/multi_camera_display_test.py \
    --mode sine \
    --max_duration 60
```

## 🧪 运行测试

```bash
# 单元测试
pytest tests/display/test_multi_camera_display.py -v

# 集成测试
python examples/so101_sim/multi_camera_display_test.py --max_duration 30
```

## 📚 相关资源

- [LeRobot主页](https://github.com/huggingface/lerobot)
- [MuJoCo文档](https://mujoco.readthedocs.io/)
- [SO101使用文档](../../So101使用文档.md)

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目遵循LeRobot项目的许可证。

---

**快速链接**:
- [中文快速入门](快速入门.md) | [English Quick Start](QUICKSTART.md)
- [详细文档](README_multi_camera_teleop.md) | [实现总结](IMPLEMENTATION_SUMMARY.md)
