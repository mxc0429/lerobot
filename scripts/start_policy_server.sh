#!/bin/bash
# PolicyServer 启动脚本
# 用于在服务器端（172.21.137.91）启动异步推理服务

set -e  # 遇到错误立即退出

# 配置参数
HOST="0.0.0.0"  # 监听所有网络接口
PORT=8080
FPS=30
INFERENCE_LATENCY=0.033  # 约 30ms
OBS_QUEUE_TIMEOUT=1.0

# 日志文件
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/policy_server_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "启动 PolicyServer"
echo "=========================================="
echo "主机: $HOST"
echo "端口: $PORT"
echo "FPS: $FPS"
echo "推理延迟: ${INFERENCE_LATENCY}s"
echo "日志文件: $LOG_FILE"
echo "=========================================="

# 检查端口是否被占用
if netstat -tuln 2>/dev/null | grep -q ":$PORT "; then
    echo "错误: 端口 $PORT 已被占用"
    echo "请运行以下命令查看占用进程: sudo netstat -tulnp | grep $PORT"
    exit 1
fi

# 检查 CUDA 是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "警告: nvidia-smi 未找到，CUDA 可能不可用"
else
    echo "CUDA 设备信息:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# 启动 PolicyServer
echo "正在启动 PolicyServer..."
python -m lerobot.async_inference.policy_server \
    --host="$HOST" \
    --port="$PORT" \
    --fps="$FPS" \
    --inference_latency="$INFERENCE_LATENCY" \
    --obs_queue_timeout="$OBS_QUEUE_TIMEOUT" \
    2>&1 | tee "$LOG_FILE"
