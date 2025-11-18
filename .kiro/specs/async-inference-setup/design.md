# Design Document: SO101 异步推理配置

## Overview

本设计文档描述了如何配置 LeRobot 的异步推理系统，使远程服务器（172.21.137.91）负责模型推理，本地低算力设备负责机械臂控制。该架构通过网络解耦推理和执行，解决了本地设备显卡故障和算力不足的问题。

### 核心优势

1. **零空闲时间**：机械臂在执行当前动作时，服务器已在计算下一批动作
2. **算力灵活分配**：推理在服务器 GPU 上进行，本地设备只需 4GB 显存
3. **网络容错**：通过动作队列缓冲，减少网络波动影响
4. **参数可调**：支持调整动作块大小和队列阈值以优化性能

## Architecture

### 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    服务器端 (172.21.137.91)                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              PolicyServer (gRPC Server)                   │  │
│  │  - 监听端口: 8080                                          │  │
│  │  - 设备: CUDA                                             │  │
│  │  - 模型: outputs/train/smolvla_so101_test2               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            ↓↑                                    │
│                    观察数据 / 动作块                              │
└─────────────────────────────────────────────────────────────────┘
                            ↓↑ (TCP/IP)
┌─────────────────────────────────────────────────────────────────┐
│                      本地设备 (4GB 显存)                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              RobotClient (gRPC Client)                    │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │  │
│  │  │ 相机采集     │  │ 动作队列      │  │ 机械臂控制       │  │  │
│  │  │ 3x OpenCV   │→ │ Queue        │→ │ SO101 Follower  │  │  │
│  │  └─────────────┘  └──────────────┘  └─────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流程

1. **观察采集**：本地设备从 3 个相机采集图像 + 机械臂状态
2. **上行传输**：RobotClient 将观察数据序列化并发送到 PolicyServer
3. **模型推理**：PolicyServer 在 CUDA 上运行 SmolVLA 模型推理
4. **下行传输**：PolicyServer 返回动作块（多个连续动作）
5. **动作执行**：RobotClient 从队列中取出动作并控制机械臂执行


## Components and Interfaces

### 1. PolicyServer (服务器端)

**职责**：
- 接收来自 RobotClient 的观察数据
- 加载并运行策略模型进行推理
- 返回动作块给客户端

**关键配置**：
```python
PolicyServerConfig:
  - host: "0.0.0.0"  # 监听所有网络接口
  - port: 8080
  - fps: 30
  - inference_latency: 0.033  # 目标推理延迟（秒）
  - obs_queue_timeout: 1.0    # 观察队列超时（秒）
```

**启动命令**：
```bash
# 在服务器 172.21.137.91 上执行
python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080 \
    --fps=30 \
    --inference_latency=0.033 \
    --obs_queue_timeout=1.0
```

**接口方法**：
- `Ready()`: 客户端连接握手
- `SendPolicyInstructions()`: 接收策略配置（模型路径、设备等）
- `SendObservations()`: 接收观察数据流
- `GetActions()`: 返回动作块

### 2. RobotClient (本地设备)

**职责**：
- 连接并控制 SO101 机械臂
- 采集相机图像和机械臂状态
- 与 PolicyServer 通信
- 管理动作队列并执行动作

**关键配置**：
```python
RobotClientConfig:
  - server_address: "172.21.137.91:8080"
  - policy_type: "smolvla"
  - pretrained_name_or_path: "outputs/train/smolvla_so101_test2"
  - policy_device: "cuda"  # 服务器端设备
  - actions_per_chunk: 50  # 每次推理输出的动作数量
  - chunk_size_threshold: 0.5  # 队列阈值（0-1）
  - aggregate_fn_name: "weighted_average"  # 动作聚合函数
  - fps: 30
  - debug_visualize_queue_size: True
```

**启动命令**：
```bash
# 在本地设备上执行
python -m lerobot.async_inference.robot_client \
    --server_address=172.21.137.91:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{camera1: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, camera2: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, camera3: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}}" \
    --task="Pick the cube to place" \
    --policy_type=smolvla \
    --pretrained_name_or_path=outputs/train/smolvla_so101_test2 \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```

**核心线程**：
1. **主线程（控制循环）**：
   - 执行动作队列中的动作
   - 采集观察数据并发送到服务器
   
2. **动作接收线程**：
   - 从服务器接收动作块
   - 更新本地动作队列

### 3. 动作队列管理

**队列机制**：
- 使用 Python `Queue` 实现线程安全的动作缓冲
- 存储 `TimedAction` 对象（包含时间戳、步数、动作张量）

**队列更新策略**：
- 当队列大小 ≤ `chunk_size_threshold * actions_per_chunk` 时，发送新观察
- 新动作块到达时，使用聚合函数合并重叠部分

**聚合函数选项**：
```python
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,  # 偏向新动作
    "latest_only": lambda old, new: new,                          # 只用新动作
    "average": lambda old, new: 0.5 * old + 0.5 * new,           # 平均
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,      # 偏向旧动作
}
```


## Data Models

### 1. TimedObservation

```python
@dataclass
class TimedObservation:
    timestamp: float        # 时间戳（秒）
    timestep: int          # 步数
    observation: RawObservation  # 原始观察数据
    must_go: bool = False  # 是否必须处理（队列为空时）
```

**RawObservation 结构**：
```python
{
    "observation.state": torch.Tensor,  # 机械臂关节状态
    "observation.images.camera1": np.ndarray,  # 相机1图像
    "observation.images.camera2": np.ndarray,  # 相机2图像
    "observation.images.camera3": np.ndarray,  # 相机3图像
    "task": str,  # 任务描述
}
```

### 2. TimedAction

```python
@dataclass
class TimedAction:
    timestamp: float   # 时间戳（秒）
    timestep: int     # 步数
    action: torch.Tensor  # 动作张量 (action_dim,)
```

### 3. RemotePolicyConfig

```python
@dataclass
class RemotePolicyConfig:
    policy_type: str                    # 策略类型（如 "smolvla"）
    pretrained_name_or_path: str        # 模型路径
    lerobot_features: dict              # LeRobot 特征映射
    actions_per_chunk: int              # 每次输出的动作数
    device: str                         # 推理设备（如 "cuda"）
    rename_map: dict = None             # 键名映射
```

## Error Handling

### 1. 网络连接错误

**场景**：客户端无法连接到服务器

**处理策略**：
- RobotClient 启动时检测连接失败
- 记录错误日志并退出
- 不启动机械臂控制循环

**实现**：
```python
try:
    self.stub.Ready(services_pb2.Empty())
except grpc.RpcError as e:
    self.logger.error(f"Failed to connect to policy server: {e}")
    return False
```

### 2. 网络中断

**场景**：运行过程中网络连接断开

**处理策略**：
- 检测到 `grpc.RpcError` 时停止机械臂
- 设置 `shutdown_event` 标志
- 清空动作队列
- 断开机械臂连接

**实现**：
```python
except grpc.RpcError as e:
    self.logger.error(f"Error receiving actions: {e}")
    self.stop()  # 触发停止流程
```

### 3. 动作队列为空

**场景**：推理速度跟不上执行速度，队列耗尽

**处理策略**：
- 设置 `must_go` 标志，强制下一个观察立即处理
- 机械臂暂停执行，等待新动作
- 记录警告日志

**实现**：
```python
with self.action_queue_lock:
    observation.must_go = self.must_go.is_set() and self.action_queue.empty()
```

### 4. 模型加载失败

**场景**：服务器无法加载指定的模型

**处理策略**：
- PolicyServer 捕获异常并记录错误
- 返回空响应给客户端
- 客户端检测到空响应后停止

**实现**：
```python
try:
    self.policy = policy_class.from_pretrained(pretrained_name_or_path)
    self.policy.to(self.device)
except Exception as e:
    self.logger.error(f"Failed to load policy: {e}")
    raise
```

### 5. 相机采集失败

**场景**：相机设备不可用或采集超时

**处理策略**：
- 在 `robot.get_observation()` 中捕获异常
- 记录错误并跳过该帧
- 继续尝试下一帧采集

**实现**：
```python
try:
    raw_observation = self.robot.get_observation()
except Exception as e:
    self.logger.error(f"Error capturing observation: {e}")
    continue
```


## Testing Strategy

### 1. 连接测试

**目标**：验证客户端和服务器能够成功建立连接

**步骤**：
1. 启动 PolicyServer
2. 启动 RobotClient（不连接真实机械臂）
3. 检查日志确认握手成功
4. 验证策略配置正确传输

**预期结果**：
- 服务器日志显示 "Client connected and ready"
- 客户端日志显示 "Connected to policy server"
- 策略模型成功加载

### 2. 数据传输测试

**目标**：验证观察数据和动作数据能够正确传输

**步骤**：
1. 使用模拟数据发送观察
2. 检查服务器是否接收到完整数据
3. 验证服务器返回的动作块格式正确
4. 检查客户端是否正确解析动作

**预期结果**：
- 观察数据序列化/反序列化无损
- 动作块包含正确数量的动作
- 时间戳和步数正确对应

### 3. 动作队列测试

**目标**：验证动作队列管理逻辑正确

**步骤**：
1. 发送多个动作块
2. 检查队列大小变化
3. 验证动作聚合函数工作正常
4. 测试队列为空时的 must_go 机制

**预期结果**：
- 队列大小在合理范围内波动
- 重叠动作正确聚合
- 队列为空时触发 must_go 标志

### 4. 性能测试

**目标**：测量系统延迟和吞吐量

**测量指标**：
- 单向网络延迟（客户端→服务器）
- 推理延迟（服务器端）
- 端到端延迟（观察→动作执行）
- 实际 FPS vs 目标 FPS

**工具**：
- 使用 `debug_visualize_queue_size=True` 可视化队列
- 分析日志中的时间戳

**预期结果**：
- 网络延迟 < 50ms
- 推理延迟 < 100ms
- 实际 FPS ≈ 30

### 5. 参数调优测试

**目标**：找到最优的 `actions_per_chunk` 和 `chunk_size_threshold`

**测试矩阵**：
```
actions_per_chunk: [20, 30, 50, 100]
chunk_size_threshold: [0.3, 0.5, 0.7]
```

**评估标准**：
- 队列为空的频率（越低越好）
- 动作执行的平滑度
- 任务成功率

**预期结果**：
- 找到平衡延迟和稳定性的最优参数组合

### 6. 故障恢复测试

**目标**：验证系统在异常情况下的鲁棒性

**测试场景**：
1. 网络临时中断
2. 服务器重启
3. 相机设备断开
4. 模型推理超时

**预期结果**：
- 系统能够检测到异常
- 机械臂安全停止
- 错误信息清晰记录

## Configuration Parameters

### 关键参数说明

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `actions_per_chunk` | 50 | 10-100 | 每次推理输出的动作数量。值越大，队列越不容易为空，但动作可能不够精确 |
| `chunk_size_threshold` | 0.5 | 0.0-1.0 | 队列阈值。值越大，发送观察越频繁，适应性越强，但网络压力越大 |
| `fps` | 30 | 10-60 | 控制循环频率。需要根据推理延迟调整 |
| `aggregate_fn_name` | "weighted_average" | - | 动作聚合函数。推荐使用 weighted_average 偏向新动作 |
| `policy_device` | "cuda" | cpu/cuda/mps | 服务器端推理设备 |

### 调优建议

**场景 1：队列经常为空**
- 增加 `actions_per_chunk`（如 50 → 100）
- 降低 `fps`（如 30 → 20）
- 减小 `chunk_size_threshold`（如 0.5 → 0.3）

**场景 2：动作不够精确**
- 减小 `actions_per_chunk`（如 50 → 30）
- 增加 `chunk_size_threshold`（如 0.5 → 0.7）
- 使用 "weighted_average" 聚合函数

**场景 3：网络带宽受限**
- 减小图像分辨率（640x480 → 320x240）
- 减小 `chunk_size_threshold`（减少观察发送频率）
- 使用图像压缩

## Deployment Checklist

### 服务器端准备

- [ ] 确认服务器 IP 地址：172.21.137.91
- [ ] 确认端口 8080 未被占用：`netstat -tuln | grep 8080`
- [ ] 确认 CUDA 可用：`nvidia-smi`
- [ ] 确认模型文件存在：`outputs/train/smolvla_so101_test2`
- [ ] 安装异步推理依赖：`pip install -e ".[async]"`
- [ ] 配置防火墙允许 8080 端口

### 本地设备准备

- [ ] 确认机械臂连接：`ls /dev/ttyACM*`
- [ ] 确认相机设备：`ls /dev/video*`
- [ ] 测试相机采集：`python lerobot/find_cameras.py`
- [ ] 确认网络连通性：`ping 172.21.137.91`
- [ ] 测试端口连接：`telnet 172.21.137.91 8080`
- [ ] 安装异步推理依赖：`pip install -e ".[async]"`

### 启动顺序

1. **先启动服务器**：
   ```bash
   # 在服务器上
   python -m lerobot.async_inference.policy_server \
       --host=0.0.0.0 \
       --port=8080
   ```

2. **等待服务器就绪**（看到 "PolicyServer started" 日志）

3. **启动客户端**：
   ```bash
   # 在本地设备上
   python -m lerobot.async_inference.robot_client \
       --server_address=172.21.137.91:8080 \
       --robot.type=so101_follower \
       --robot.port=/dev/ttyACM0 \
       --robot.id=my_awesome_follower_arm \
       --robot.cameras="{camera1: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, camera2: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, camera3: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}}" \
       --task="Pick the cube to place" \
       --policy_type=smolvla \
       --pretrained_name_or_path=outputs/train/smolvla_so101_test2 \
       --policy_device=cuda \
       --actions_per_chunk=50 \
       --chunk_size_threshold=0.5 \
       --aggregate_fn_name=weighted_average \
       --debug_visualize_queue_size=True
   ```

4. **观察日志**：
   - 服务器：确认接收到策略配置和观察数据
   - 客户端：确认连接成功和动作执行

5. **停止顺序**：
   - 先 Ctrl+C 停止客户端
   - 再 Ctrl+C 停止服务器
