# 异步推理启动脚本

本目录包含用于启动 LeRobot 异步推理系统的脚本。

## 脚本说明

### 1. `test_connection_only.sh` - 测试模式 ⭐ 推荐首次使用

**用途：** 测试服务器连接、模型推理和相机采集，**不控制机械臂**

**使用场景：**
- 首次配置系统
- 验证网络连接
- 测试模型加载
- 检查相机设备

**运行方法：**
```bash
./scripts/test_connection_only.sh
```

**测试内容：**
- ✓ 检查相机设备（/dev/video2, /dev/video4, /dev/video6）
- ✓ 测试服务器连接（172.21.137.91:8080）
- ✓ 发送观察数据到服务器
- ✓ 接收服务器推理的动作
- ✗ 不连接机械臂，不执行动作

**预期结果：**
- 相机成功采集图像
- 服务器成功加载模型
- 完成 3 次观察-推理循环
- 日志保存在 `logs/connection_test_*.log`

---

### 2. `start_policy_server.sh` - 服务器端启动脚本

**用途：** 在服务器（172.21.137.91）上启动 PolicyServer

**运行方法：**
```bash
# 在服务器上执行
./scripts/start_policy_server.sh
```

**功能：**
- 监听 0.0.0.0:8080
- 自动检查端口占用
- 显示 CUDA 设备信息
- 记录日志到 `logs/policy_server_*.log`

---

### 3. `start_robot_client.sh` - 客户端启动脚本（完整模式）

**用途：** 在本地设备启动 RobotClient，**控制机械臂执行任务**

**⚠️ 警告：** 此脚本会实际控制机械臂移动，请确保：
1. 测试模式已成功运行
2. 机械臂周围无障碍物
3. 准备好紧急停止

**运行方法：**
```bash
./scripts/start_robot_client.sh
```

**功能：**
- 连接 SO101 机械臂（/dev/ttyACM0）
- 连接 3 个相机
- 与服务器通信
- 执行推理动作
- 记录日志到 `logs/robot_client_*.log`

---

## 使用流程

### 首次使用（推荐）

1. **在服务器上启动 PolicyServer**
   ```bash
   # SSH 到服务器
   ssh user@172.21.137.91
   cd /path/to/lerobot
   ./scripts/start_policy_server.sh
   ```

2. **在本地运行测试模式**
   ```bash
   cd ~/Robot/lerobot
   ./scripts/test_connection_only.sh
   ```

3. **如果测试成功，运行完整模式**
   ```bash
   # 停止测试（Ctrl+C）
   # 确保机械臂周围安全
   ./scripts/start_robot_client.sh
   ```

### 日常使用

如果系统已经验证正常，可以直接：

1. 启动服务器：`./scripts/start_policy_server.sh`
2. 启动客户端：`./scripts/start_robot_client.sh`

---

## 配置修改

所有脚本的配置参数都在文件开头的"配置参数"部分，可以根据需要修改：

**常见修改：**
- 服务器地址：`SERVER_ADDRESS="172.21.137.91:8080"`
- 机械臂端口：`ROBOT_PORT="/dev/ttyACM0"`
- 相机设备：修改 `CAMERAS` 变量
- 模型路径：`PRETRAINED_PATH="outputs/train/smolvla_so101_test2"`
- 异步参数：`ACTIONS_PER_CHUNK`, `CHUNK_SIZE_THRESHOLD`

---

## 故障排查

### 测试模式失败

**问题：** 无法连接到服务器
```bash
# 检查服务器是否启动
ssh user@172.21.137.91 "netstat -tuln | grep 8080"

# 检查网络连接
ping 172.21.137.91
telnet 172.21.137.91 8080
```

**问题：** 相机设备不存在
```bash
# 查看可用相机
ls /dev/video*

# 测试相机
python lerobot/find_cameras.py
```

**问题：** 模型加载失败
```bash
# 在服务器上检查模型文件
ssh user@172.21.137.91 "ls -la /path/to/lerobot/outputs/train/smolvla_so101_test2"
```

### 完整模式失败

**问题：** 机械臂设备不存在
```bash
# 查看可用设备
ls /dev/ttyACM*

# 检查权限
sudo chmod 666 /dev/ttyACM0
```

---

## 日志文件

所有脚本都会在 `logs/` 目录下生成日志文件：

- `logs/connection_test_*.log` - 测试模式日志
- `logs/policy_server_*.log` - 服务器日志
- `logs/robot_client_*.log` - 客户端日志

**查看实时日志：**
```bash
tail -f logs/connection_test_*.log
```

---

## 更多信息

详细文档请参考：
- [快速启动指南](../ASYNC_INFERENCE_GUIDE.md)
- [设计文档](../.kiro/specs/async-inference-setup/design.md)
- [任务列表](../.kiro/specs/async-inference-setup/tasks.md)
