# SO101 Leader臂校准指南

## 校准文件位置

### 项目目录（默认）
校准文件存储在项目的Calibration目录中：
```
examples/so101_sim/Calibration/{id}.json
```

其中 `{id}` 是你的Leader臂的ID（默认为 `"main"`）。

### 完整路径示例
```bash
# 默认ID为"main"的校准文件
examples/so101_sim/Calibration/main.json

# 自定义ID的校准文件
examples/so101_sim/Calibration/my_leader.json
```

### 优点
- **便于管理**: 校准文件与项目代码在一起
- **易于备份**: 随项目一起备份
- **版本控制**: 可选择性地加入Git（个人项目）
- **便于分享**: 团队成员可以共享校准文件

### 注意
- 校准文件已添加到 `.gitignore`，不会自动提交到Git
- 如需共享校准文件，需要手动添加到版本控制

## 校准文件内容

校准文件是JSON格式，包含每个关节的校准数据：

```json
{
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": 2048,
        "range_min": 1024,
        "range_max": 3072
    },
    "shoulder_lift": {
        "id": 2,
        "drive_mode": 0,
        "homing_offset": 2048,
        "range_min": 1024,
        "range_max": 3072
    },
    // ... 其他关节
}
```

### 参数说明

- **id**: 舵机ID（1-6）
- **drive_mode**: 驱动模式（通常为0）
- **homing_offset**: 归零偏移值（中间位置）
- **range_min**: 最小位置值
- **range_max**: 最大位置值

## 跳过校准的方法

### 方法1: 使用现有校准文件

如果你已经有校准文件，可以直接使用：

```python
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig

config = SO101LeaderConfig(
    port="/dev/ttyACM0",
    id="main"  # 使用已有的校准文件ID
)

leader = SO101Leader(config)
leader.connect(calibrate=False)  # 设置calibrate=False跳过校准
```

### 方法2: 复制校准文件

如果你有另一台相同型号的Leader臂的校准文件：

```bash
# 复制校准文件
cp examples/so101_sim/Calibration/old_id.json \
   examples/so101_sim/Calibration/new_id.json
```

### 方法3: 手动创建校准文件

如果你知道校准参数，可以手动创建：

```bash
# 创建校准文件
cat > examples/so101_sim/Calibration/main.json << 'EOF'
{
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": 2048,
        "range_min": 1024,
        "range_max": 3072
    },
    "shoulder_lift": {
        "id": 2,
        "drive_mode": 0,
        "homing_offset": 2048,
        "range_min": 1024,
        "range_max": 3072
    },
    "elbow_flex": {
        "id": 3,
        "drive_mode": 0,
        "homing_offset": 2048,
        "range_min": 1024,
        "range_max": 3072
    },
    "wrist_flex": {
        "id": 4,
        "drive_mode": 0,
        "homing_offset": 2048,
        "range_min": 1024,
        "range_max": 3072
    },
    "wrist_roll": {
        "id": 5,
        "drive_mode": 0,
        "homing_offset": 2048,
        "range_min": 1024,
        "range_max": 3072
    },
    "gripper": {
        "id": 6,
        "drive_mode": 0,
        "homing_offset": 2048,
        "range_min": 1024,
        "range_max": 3072
    }
}
EOF
```

**注意**: 手动创建的校准文件可能不准确，建议至少运行一次完整校准。

## 修改脚本跳过校准

### 修改 s101_leader_to_sim.py

在 `LeaderToSimController.__init__` 方法中：

```python
# 原代码
self.leader.connect(calibrate=True)

# 修改为
self.leader.connect(calibrate=False)  # 跳过校准
```

### 修改 test_leader_connection.py

在 `test_leader_connection` 函数中：

```python
# 原代码
leader.connect(calibrate=True)

# 修改为
leader.connect(calibrate=False)  # 跳过校准
```

## 校准流程详解

如果你需要进行校准，流程如下：

### 步骤1: 移动到中间位置
```
Move so101_leader to the middle of its range of motion and press ENTER....
```
- 将所有关节移动到大约中间位置
- 这个位置会被记录为归零点（homing_offset）
- 按ENTER继续

### 步骤2: 记录运动范围
```
Move all joints sequentially through their entire ranges of motion.
Recording positions. Press ENTER to stop...
```
- 依次移动每个关节到最小和最大位置
- 系统会记录每个关节的运动范围
- 完成后按ENTER

### 步骤3: 保存校准
```
Calibration saved to ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/main.json
```
- 校准数据自动保存
- 下次连接会自动加载

## 查看和管理校准文件

### 查看校准文件
```bash
# 查看校准文件内容
cat examples/so101_sim/Calibration/main.json

# 格式化显示
python -m json.tool examples/so101_sim/Calibration/main.json
```

### 备份校准文件
```bash
# 备份单个文件
cp examples/so101_sim/Calibration/main.json ~/so101_calibration_backup.json

# 备份整个目录
cp -r examples/so101_sim/Calibration ~/so101_calibration_backup

# 恢复
cp ~/so101_calibration_backup.json examples/so101_sim/Calibration/main.json
```

### 删除校准文件（重新校准）
```bash
# 删除校准文件
rm examples/so101_sim/Calibration/main.json

# 下次连接会提示重新校准
```

## 使用不同ID管理多个Leader臂

如果你有多个Leader臂：

```python
# Leader臂1
config1 = SO101LeaderConfig(
    port="/dev/ttyACM0",
    id="leader_1"
)

# Leader臂2
config2 = SO101LeaderConfig(
    port="/dev/ttyACM1",
    id="leader_2"
)
```

每个ID会有独立的校准文件：
- `examples/so101_sim/Calibration/leader_1.json`
- `examples/so101_sim/Calibration/leader_2.json`

## 修改脚本支持自定义ID

### 修改 s101_leader_to_sim.py

添加命令行参数：

```python
def main():
    parser = argparse.ArgumentParser(description="使用Leader臂控制仿真Follower臂")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--no-opencv", action="store_true")
    parser.add_argument("--id", type=str, default="main", help="Leader臂ID")
    parser.add_argument("--no-calibrate", action="store_true", help="跳过校准")
    
    args = parser.parse_args()
    
    # 在LeaderToSimController.__init__中使用
    leader_config = SO101LeaderConfig(
        port=args.port,
        id=args.id,
        use_degrees=False
    )
    
    # 在connect时使用
    self.leader.connect(calibrate=not args.no_calibrate)
```

使用方法：
```bash
# 使用自定义ID
python examples/so101_sim/s101_leader_to_sim.py --id my_leader

# 跳过校准
python examples/so101_sim/s101_leader_to_sim.py --no-calibrate

# 组合使用
python examples/so101_sim/s101_leader_to_sim.py --id my_leader --no-calibrate
```

## 常见问题

### Q: 校准文件在哪里？
A: `examples/so101_sim/Calibration/main.json`（项目目录下）

### Q: 可以跳过校准吗？
A: 可以，使用 `leader.connect(calibrate=False)`，但需要已有校准文件

### Q: 校准需要多长时间？
A: 约2-3分钟，只需要做一次

### Q: 校准文件可以共享吗？
A: 理论上可以，但不同机械臂可能有细微差异，建议每台单独校准

### Q: 如何重新校准？
A: 删除校准文件或在连接时选择重新校准

### Q: 校准失败怎么办？
A: 检查机械臂连接、串口权限，确保能正常移动所有关节

## 总结

- **校准文件位置**: `~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/{id}.json`
- **跳过校准**: 使用 `connect(calibrate=False)` 且确保有校准文件
- **首次使用**: 建议完整校准一次，数据会保存
- **多个Leader臂**: 使用不同的ID管理
- **备份重要**: 校准完成后建议备份校准文件
