# SO101 Leader臂校准文件目录

此目录用于存储SO101 Leader臂的校准文件。

## 文件格式

校准文件以JSON格式存储，文件名为 `{id}.json`，其中 `{id}` 是Leader臂的ID。

### 默认文件
- `main.json` - 默认Leader臂的校准文件

### 文件结构

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
```

## 参数说明

- **id**: 舵机ID（1-6）
- **drive_mode**: 驱动模式（通常为0）
- **homing_offset**: 归零偏移值，表示中间位置的舵机原始值
- **range_min**: 最小位置值，表示关节运动范围的最小舵机值
- **range_max**: 最大位置值，表示关节运动范围的最大舵机值

## 如何生成校准文件

### 方法1: 自动校准（推荐）

运行脚本时会自动引导校准：

```bash
python examples/so101_sim/s101_leader_to_sim.py
```

按照提示：
1. 将机械臂移动到中间位置，按ENTER
2. 移动所有关节通过完整运动范围，按ENTER
3. 校准文件自动保存到此目录

### 方法2: 测试连接时校准

```bash
python examples/so101_sim/test_leader_connection.py
```

### 方法3: 手动创建

如果你知道准确的校准参数，可以手动创建JSON文件。

## 管理多个Leader臂

如果你有多个Leader臂，可以使用不同的ID：

```bash
# Leader臂1
python examples/so101_sim/s101_leader_to_sim.py --id leader_1

# Leader臂2
python examples/so101_sim/s101_leader_to_sim.py --id leader_2
```

这会创建：
- `leader_1.json`
- `leader_2.json`

## 备份和恢复

### 备份校准文件

```bash
# 备份单个文件
cp examples/so101_sim/Calibration/main.json ~/backup_main.json

# 备份整个目录
cp -r examples/so101_sim/Calibration ~/backup_calibration
```

### 恢复校准文件

```bash
# 恢复单个文件
cp ~/backup_main.json examples/so101_sim/Calibration/main.json

# 恢复整个目录
cp -r ~/backup_calibration/* examples/so101_sim/Calibration/
```

## 重新校准

如果需要重新校准：

```bash
# 删除现有校准文件
rm examples/so101_sim/Calibration/main.json

# 重新运行校准
python examples/so101_sim/s101_leader_to_sim.py
```

## 注意事项

1. **不要手动编辑** - 除非你完全理解参数含义
2. **定期备份** - 校准数据很重要
3. **版本控制** - 可以将校准文件加入Git（如果是个人项目）
4. **共享谨慎** - 不同机械臂的校准参数可能不同

## 故障排除

### 校准文件损坏

如果校准文件损坏，删除它并重新校准：

```bash
rm examples/so101_sim/Calibration/main.json
python examples/so101_sim/s101_leader_to_sim.py
```

### 校准不准确

重新运行校准，确保：
- 机械臂能自由移动
- 移动到真正的最大和最小位置
- 中间位置准确

### 找不到校准文件

确保：
- 使用正确的ID
- 文件名正确（`{id}.json`）
- 文件在正确的目录中

## 相关文档

- [CALIBRATION_GUIDE.md](../CALIBRATION_GUIDE.md) - 详细校准指南
- [README.md](../README.md) - 项目主文档
- [QUICK_START.md](../QUICK_START.md) - 快速开始指南
