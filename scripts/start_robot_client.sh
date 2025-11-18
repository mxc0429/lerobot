#!/bin/bash
# RobotClient 启动脚本
# 用于在本地设备上启动机器人客户端，连接到远程 PolicyServer

set -e  # 遇到错误立即退出

# ========================================
# 配置参数
# ========================================

# 服务器配置
SERVER_ADDRESS="172.21.137.91:8080"

# 机器人配置
ROBOT_TYPE="so101_follower"
ROBOT_PORT="/dev/ttyACM0"
ROBOT_ID="my_awesome_follower_arm"

# 相机配置（根据你的实际设备调整）
CAMERAS="{camera1: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, camera2: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, camera3: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}}"

# 任务描述
TASK="Pick the cube to place"

# 策略配置
POLICY_TYPE="smolvla"
PRETRAINED_PATH="outputs/train/smolvla_so101_test2"
POLICY_DEVICE="cuda"  # 服务器端设备

# 异步推理参数
ACTIONS_PER_CHUNK=50
CHUNK_SIZE_THRESHOLD=0.5
AGGREGATE_FN="weighted_average"
FPS=30

# 调试选项
DEBUG_VISUALIZE=true

# 日志文件
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/robot_client_$(date +%Y%m%d_%H%M%S).log"

# ========================================
# 预检查
# ========================================

echo "=========================================="
echo "启动 RobotClient"
echo "=========================================="
echo "服务器地址: $SERVER_ADDRESS"
echo "机器人类型: $ROBOT_TYPE"
echo "机器人端口: $ROBOT_PORT"
echo "策略类型: $POLICY_TYPE"
echo "模型路径: $PRETRAINED_PATH"
echo "日志文件: $LOG_FILE"
echo "=========================================="

# 检查机器人设备
if [ ! -e "$ROBOT_PORT" ]; then
    echo "错误: 机器人设备 $ROBOT_PORT 不存在"
    echo "请检查机械臂是否连接，运行: ls /dev/ttyACM*"
    exit 1
fi
echo "✓ 机器人设备已找到: $ROBOT_PORT"

# 检查相机设备
for video_dev in /dev/video2 /dev/video4 /dev/video6; do
    if [ ! -e "$video_dev" ]; then
        echo "警告: 相机设备 $video_dev 不存在"
    else
        echo "✓ 相机设备已找到: $video_dev"
    fi
done

# 检查服务器连接
echo ""
echo "检查服务器连接..."
if ! ping -c 1 -W 2 172.21.137.91 &> /dev/null; then
    echo "错误: 无法 ping 通服务器 172.21.137.91"
    echo "请检查网络连接"
    exit 1
fi
echo "✓ 服务器网络连接正常"

# 检查服务器端口（注意：如果服务器未启动，这里会失败，但不阻止继续）
if nc -zv 172.21.137.91 8080 2>&1 | grep -q "succeeded\|open"; then
    echo "✓ 服务器端口 8080 可访问"
else
    echo "警告: 无法连接到服务器端口 8080"
    echo "请确保服务器端的 PolicyServer 已启动"
    echo "按 Ctrl+C 取消，或等待 5 秒后继续..."
    sleep 5
fi

# ========================================
# 启动客户端
# ========================================

echo ""
echo "正在启动 RobotClient..."
python -m lerobot.async_inference.robot_client \
    --server_address="$SERVER_ADDRESS" \
    --robot.type="$ROBOT_TYPE" \
    --robot.port="$ROBOT_PORT" \
    --robot.id="$ROBOT_ID" \
    --robot.cameras="$CAMERAS" \
    --task="$TASK" \
    --policy_type="$POLICY_TYPE" \
    --pretrained_name_or_path="$PRETRAINED_PATH" \
    --policy_device="$POLICY_DEVICE" \
    --actions_per_chunk="$ACTIONS_PER_CHUNK" \
    --chunk_size_threshold="$CHUNK_SIZE_THRESHOLD" \
    --aggregate_fn_name="$AGGREGATE_FN" \
    --fps="$FPS" \
    --debug_visualize_queue_size="$DEBUG_VISUALIZE" \
    2>&1 | tee "$LOG_FILE"
