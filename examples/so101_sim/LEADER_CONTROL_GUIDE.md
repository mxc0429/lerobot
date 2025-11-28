# SO101 Leader臂控制仿真环境指南

本指南介绍如何使用真实的SO101 Leader臂来控制PyBullet仿真环境中的Follower臂。

## 系统架构

```
真实Leader臂 (串口) → Python脚本 → PyBullet仿真环境 (Follower臂)
                                    ↓
                              OpenCV相机显示
                              (顶部/侧面/腕部视角)
```

## 快速开始

### 1. 硬件准备

- SO101 Leader臂
- USB转串口线
- 连接到计算机（通常为 `/dev/ttyACM0` 或 `/dev/ttyUSB0`）

### 2. 权限设置

```bash
# 添加当前用户到dialout组（获取串口访问权限）
sudo usermod -a -G dialout $USER

# 重新登录使权限生效
# 或者运行: newgrp dialout
```

### 3. 测试连接

```bash
# 测试Leader臂是否正确连接
python examples/so101_sim/test_leader_connection.py
```

如果测试成功，你会看到：
- Leader臂连接成功
- 实时显示关节位置
- 移动Leader臂时数值会变化

### 4. 运行仿真控制

```bash
# 方法1: 直接运行Python脚本
python examples/so101_sim/s101_leader_to_sim.py

# 方法2: 使用快速启动脚本
./examples/so101_sim/run_leader_control.sh
```

## 工作原理

### 数据流程

1. **读取Leader位置**
   - 从Leader臂读取6个关节的位置
   - 数据格式：归一化值 [-100, 100]

2. **坐标转换**
   - 将归一化值转换为弧度
   - 考虑每个关节的限制范围

3. **控制仿真Follower**
   - 将弧度值发送到PyBullet仿真环境
   - Follower臂实时跟随Leader臂运动

4. **相机渲染**
   - 每5步渲染一次相机画面
   - 三个视角同时显示

### 关节映射

| 关节名称 | 索引 | 归一化范围 | 弧度范围 |
|---------|------|-----------|---------|
| shoulder_pan | 0 | [-100, 100] | [-1.92, 1.92] |
| shoulder_lift | 1 | [-100, 100] | [-1.75, 1.75] |
| elbow_flex | 2 | [-100, 100] | [-1.69, 1.69] |
| wrist_flex | 3 | [-100, 100] | [-1.66, 1.66] |
| wrist_roll | 4 | [-100, 100] | [-2.74, 2.84] |
| gripper | 5 | [-100, 100] | [-0.17, 1.75] |

## 常见问题

### Q1: 找不到串口设备

**问题**: `FileNotFoundError: [Errno 2] No such file or directory: '/dev/ttyACM0'`

**解决方法**:
```bash
# 查看可用的串口设备
ls -l /dev/tty* | grep -E "USB|ACM"

# 使用找到的端口运行
python examples/so101_sim/s101_leader_to_sim.py --port /dev/ttyUSB0
```

### Q2: 权限被拒绝

**问题**: `PermissionError: [Errno 13] Permission denied: '/dev/ttyACM0'`

**解决方法**:
```bash
# 添加用户到dialout组
sudo usermod -a -G dialout $USER

# 重新登录或运行
newgrp dialout

# 或者临时使用sudo（不推荐）
sudo python examples/so101_sim/s101_leader_to_sim.py
```

### Q3: Leader臂需要校准

**问题**: 首次运行时提示需要校准

**解决方法**:
1. 按照提示将Leader臂移动到中间位置
2. 按ENTER键
3. 移动所有关节通过完整的运动范围
4. 按ENTER键完成校准
5. 校准数据会保存，下次不需要重新校准

### Q4: 仿真Follower不跟随Leader

**问题**: Leader臂移动但仿真中的Follower不动

**可能原因**:
1. 关节位置读取失败 - 检查终端输出
2. 坐标转换错误 - 检查关节限制设置
3. 仿真环境未正确初始化 - 重启程序

### Q5: 相机画面不显示

**问题**: OpenCV窗口不显示或黑屏

**解决方法**:
```bash
# 确保安装了OpenCV
pip install opencv-python

# 或禁用OpenCV使用PyBullet内置显示
python examples/so101_sim/s101_leader_to_sim.py --no-opencv
```

## 高级用法

### 自定义关节映射

编辑 `s101_leader_to_sim.py` 中的 `normalize_to_radians` 方法：

```python
def normalize_to_radians(self, normalized_value, joint_idx):
    min_rad, max_rad = self.joint_limits[joint_idx]
    # 添加自定义映射逻辑
    # 例如：添加死区、非线性映射等
    normalized = (normalized_value + 100) / 200.0
    return min_rad + normalized * (max_rad - min_rad)
```

### 添加延迟补偿

如果发现跟随有延迟，可以调整读取频率：

```python
# 在 run() 方法中
time.sleep(1./240.)  # 改为更高的频率，如 1./480.
```

### 记录数据

添加数据记录功能：

```python
import json

# 在 run() 方法中
data_log = []
for step in range(max_steps):
    positions = self.get_leader_positions()
    data_log.append({
        'step': step,
        'timestamp': time.time(),
        'positions': positions
    })

# 保存数据
with open('trajectory.json', 'w') as f:
    json.dump(data_log, f)
```

## 性能优化

### 1. 降低相机渲染频率

```python
# 从每5步渲染一次改为每10步
if step % 10 == 0:
    self.sim_env.render_cameras_opencv()
```

### 2. 降低相机分辨率

编辑 `s101_pybullet_sim.py` 中的相机参数：

```python
width = 160  # 从320降低到160
height = 120  # 从240降低到120
```

### 3. 使用更快的渲染器

```python
# 在 getCameraImage 中使用 ER_TINY_RENDERER
renderer=p.ER_TINY_RENDERER  # 更快但质量较低
```

## 应用场景

### 1. 遥操作测试
- 在仿真环境中测试遥操作算法
- 无需真实Follower臂即可验证控制逻辑

### 2. 数据采集
- 使用Leader臂演示任务
- 在仿真环境中记录轨迹和相机图像
- 生成训练数据集

### 3. 算法验证
- 测试运动规划算法
- 验证碰撞检测
- 评估控制策略

### 4. 教学演示
- 展示机械臂控制原理
- 可视化关节运动
- 多视角观察

## 下一步

- 尝试 `demo_pick_and_place.py` 了解预编程运动
- 修改场景添加更多物体
- 集成到你的机器人学习项目中

## 支持

如有问题，请查看：
- [README.md](README.md) - 完整文档
- [lerobot文档](https://github.com/huggingface/lerobot) - LeRobot项目
- [PyBullet文档](https://pybullet.org/) - 仿真环境
