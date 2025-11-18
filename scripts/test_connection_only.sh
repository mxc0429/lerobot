#!/bin/bash
# 测试模式：只测试服务器连接、模型推理和相机采集，不控制机械臂
# 用于验证系统配置是否正确，避免机械臂意外移动

set -e

# ========================================
# 配置参数
# ========================================

# 服务器配置
SERVER_ADDRESS="172.21.137.91:8080"

# 相机配置（只测试相机，不连接机械臂）
CAMERAS="{camera1: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, camera2: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, camera3: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}}"

# 任务描述
TASK="Pick the cube to place"

# 策略配置
POLICY_TYPE="smolvla"
PRETRAINED_PATH="outputs/train/smolvla_so101_test2"
POLICY_DEVICE="cuda"

# 日志文件
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/connection_test_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "连接测试模式（不控制机械臂）"
echo "=========================================="
echo "服务器地址: $SERVER_ADDRESS"
echo "策略类型: $POLICY_TYPE"
echo "模型路径: $PRETRAINED_PATH"
echo "日志文件: $LOG_FILE"
echo "=========================================="
echo ""
echo "⚠️  注意：此模式只测试连接和推理，不会控制机械臂"
echo ""

# ========================================
# 预检查
# ========================================

# 检查相机设备
echo "检查相机设备..."
camera_found=0
for video_dev in /dev/video2 /dev/video4 /dev/video6; do
    if [ -e "$video_dev" ]; then
        echo "✓ 相机设备已找到: $video_dev"
        camera_found=$((camera_found + 1))
    else
        echo "✗ 相机设备不存在: $video_dev"
    fi
done

if [ $camera_found -eq 0 ]; then
    echo ""
    echo "错误: 没有找到任何相机设备"
    echo "请运行以下命令查看可用相机: ls /dev/video*"
    exit 1
fi

# 检查服务器连接
echo ""
echo "检查服务器连接..."
if ! ping -c 1 -W 2 172.21.137.91 &> /dev/null; then
    echo "错误: 无法 ping 通服务器 172.21.137.91"
    exit 1
fi
echo "✓ 服务器网络连接正常"

# 检查服务器端口
if nc -zv 172.21.137.91 8080 2>&1 | grep -q "succeeded\|open"; then
    echo "✓ 服务器端口 8080 可访问"
else
    echo ""
    echo "错误: 无法连接到服务器端口 8080"
    echo "请确保服务器端的 PolicyServer 已启动"
    echo ""
    echo "在服务器上运行："
    echo "  python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080"
    exit 1
fi

# ========================================
# 创建测试脚本
# ========================================

echo ""
echo "创建测试脚本..."

cat > /tmp/test_async_connection.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
测试异步推理连接（不控制机械臂）
只测试：相机采集 -> 发送观察 -> 服务器推理 -> 接收动作
"""

import sys
import time
import pickle
import threading
import numpy as np
import torch
import grpc
from queue import Queue

# 导入必要的模块
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.async_inference.helpers import (
    RawObservation,
    TimedObservation,
    RemotePolicyConfig,
    get_logger,
    map_robot_keys_to_lerobot_features,
)
from lerobot.cameras.opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

logger = get_logger("test_connection")

class ConnectionTester:
    def __init__(self, server_address, cameras_config, policy_config):
        self.server_address = server_address
        self.cameras_config = cameras_config
        self.policy_config = policy_config
        
        # 初始化相机
        self.cameras = {}
        logger.info("初始化相机...")
        for name, config in cameras_config.items():
            try:
                camera = OpenCVCamera(config)
                camera.connect()
                self.cameras[name] = camera
                logger.info(f"✓ 相机 {name} 连接成功")
            except Exception as e:
                logger.error(f"✗ 相机 {name} 连接失败: {e}")
        
        if not self.cameras:
            raise RuntimeError("没有可用的相机")
        
        # 连接服务器
        logger.info(f"连接到服务器 {server_address}...")
        self.channel = grpc.insecure_channel(
            server_address, 
            grpc_channel_options(initial_backoff="0.033s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        
        self.action_queue = Queue()
        self.shutdown_event = threading.Event()
        self.timestep = 0
        
    def start(self):
        """连接到服务器并发送策略配置"""
        try:
            # 握手
            logger.info("发送握手请求...")
            self.stub.Ready(services_pb2.Empty())
            logger.info("✓ 握手成功")
            
            # 发送策略配置
            logger.info("发送策略配置...")
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)
            self.stub.SendPolicyInstructions(policy_setup)
            logger.info("✓ 策略配置已发送，服务器正在加载模型...")
            
            # 等待一下让服务器加载模型
            time.sleep(2)
            logger.info("✓ 连接建立成功")
            return True
            
        except grpc.RpcError as e:
            logger.error(f"✗ 连接失败: {e}")
            return False
    
    def capture_observation(self, task):
        """采集观察数据"""
        observation = {}
        
        # 采集相机图像（使用硬件层面的键名，不带 observation.images 前缀）
        for name, camera in self.cameras.items():
            try:
                image = camera.async_read()
                observation[name] = image  # 直接使用 camera1, camera2, camera3
            except Exception as e:
                logger.error(f"相机 {name} 采集失败: {e}")
                return None
        
        # 模拟机械臂状态（每个关节单独一个键，这是硬件层面的格式）
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", 
                      "wrist_flex", "wrist_roll", "gripper"]
        for joint_name in joint_names:
            observation[joint_name] = 0.0  # 模拟关节位置
        
        observation["task"] = task
        
        return observation
    
    def send_observation(self, observation, task):
        """发送观察到服务器"""
        timed_obs = TimedObservation(
            timestamp=time.time(),
            timestep=self.timestep,
            observation=observation,
            must_go=True
        )
        
        try:
            obs_bytes = pickle.dumps(timed_obs)
            obs_iterator = send_bytes_in_chunks(
                obs_bytes,
                services_pb2.Observation,
                log_prefix="[TEST]",
                silent=True
            )
            self.stub.SendObservations(obs_iterator)
            logger.info(f"✓ 观察 #{self.timestep} 已发送")
            return True
        except Exception as e:
            logger.error(f"✗ 发送观察失败: {e}")
            return False
    
    def receive_actions(self):
        """接收动作"""
        try:
            actions_chunk = self.stub.GetActions(services_pb2.Empty())
            if len(actions_chunk.data) == 0:
                logger.warning("服务器返回空动作")
                return None
            
            timed_actions = pickle.loads(actions_chunk.data)
            logger.info(f"✓ 接收到 {len(timed_actions)} 个动作")
            
            # 显示第一个动作的信息
            if timed_actions:
                first_action = timed_actions[0]
                logger.info(f"  动作形状: {first_action.get_action().shape}")
                logger.info(f"  动作值范围: [{first_action.get_action().min():.3f}, {first_action.get_action().max():.3f}]")
            
            return timed_actions
        except Exception as e:
            logger.error(f"✗ 接收动作失败: {e}")
            return None
    
    def run_test(self, task, num_iterations=3):
        """运行测试"""
        logger.info(f"\n开始测试循环（{num_iterations} 次迭代）...")
        logger.info("=" * 60)
        
        for i in range(num_iterations):
            logger.info(f"\n迭代 {i+1}/{num_iterations}")
            logger.info("-" * 60)
            
            # 1. 采集观察
            logger.info("1. 采集相机图像...")
            observation = self.capture_observation(task)
            if observation is None:
                logger.error("采集失败，跳过此次迭代")
                continue
            
            # 显示图像信息
            for key, value in observation.items():
                if key.startswith("camera"):
                    logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            
            # 2. 发送观察
            logger.info("2. 发送观察到服务器...")
            if not self.send_observation(observation, task):
                continue
            
            # 3. 接收动作
            logger.info("3. 等待服务器推理并接收动作...")
            actions = self.receive_actions()
            if actions is None:
                continue
            
            logger.info(f"✓ 迭代 {i+1} 完成")
            self.timestep += 1
            
            # 等待一下再进行下一次迭代
            time.sleep(1)
        
        logger.info("\n" + "=" * 60)
        logger.info("测试完成！")
    
    def cleanup(self):
        """清理资源"""
        logger.info("\n清理资源...")
        for name, camera in self.cameras.items():
            try:
                camera.disconnect()
                logger.info(f"✓ 相机 {name} 已断开")
            except:
                pass
        self.channel.close()
        logger.info("✓ 服务器连接已关闭")

def main():
    import sys
    
    # 从命令行参数获取配置
    server_address = sys.argv[1]
    task = sys.argv[2]
    policy_type = sys.argv[3]
    pretrained_path = sys.argv[4]
    policy_device = sys.argv[5]
    
    # 相机配置
    cameras_config = {
        "camera1": OpenCVCameraConfig(
            index_or_path="/dev/video2",
            width=640, height=480, fps=30, fourcc="MJPG"
        ),
        "camera2": OpenCVCameraConfig(
            index_or_path="/dev/video4",
            width=640, height=480, fps=30, fourcc="MJPG"
        ),
        "camera3": OpenCVCameraConfig(
            index_or_path="/dev/video6",
            width=640, height=480, fps=30, fourcc="MJPG"
        ),
    }
    
    # 策略配置
    # 模拟 SO101 的特征映射（使用正确的 LeRobot 数据集特征格式）
    lerobot_features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", 
                     "wrist_flex", "wrist_roll", "gripper"],
        },
        "observation.images.camera1": {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera2": {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera3": {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex",
                     "wrist_flex", "wrist_roll", "gripper"],
        },
    }
    
    policy_config = RemotePolicyConfig(
        policy_type=policy_type,
        pretrained_name_or_path=pretrained_path,
        lerobot_features=lerobot_features,
        actions_per_chunk=50,
        device=policy_device,
    )
    
    # 创建测试器
    tester = ConnectionTester(server_address, cameras_config, policy_config)
    
    try:
        # 启动连接
        if not tester.start():
            logger.error("连接失败")
            sys.exit(1)
        
        # 运行测试
        tester.run_test(task, num_iterations=3)
        
    except KeyboardInterrupt:
        logger.info("\n用户中断")
    except Exception as e:
        logger.error(f"测试出错: {e}", exc_info=True)
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()
PYTHON_SCRIPT

chmod +x /tmp/test_async_connection.py

# ========================================
# 运行测试
# ========================================

echo ""
echo "=========================================="
echo "开始测试..."
echo "=========================================="
echo ""

python /tmp/test_async_connection.py \
    "$SERVER_ADDRESS" \
    "$TASK" \
    "$POLICY_TYPE" \
    "$PRETRAINED_PATH" \
    "$POLICY_DEVICE" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="
echo "日志已保存到: $LOG_FILE"
