# 🚀 SO-101 积木堆叠 - 快速参考

## ✅ 已解决的问题

1. ✅ **标定文件加载** - 需要指定 `--leader_id` 和 `--leader_calibration_dir`
2. ✅ **移动延迟** - 优化了时间控制，添加了单相机模式
3. ✅ **段错误** - 禁用了Rerun，改进了清理代码

## 🎯 三步快速开始

### 1️⃣ 测试控制（30秒）

```bash
python3 examples/so101_sim/teleop_test_fast.py \
    --port /dev/ttyACM0 \
    --leader_id my_awesome_leader_arm \
    --duration 30 \
    --display
```

**检查**：Leader臂是否流畅控制follower臂

### 2️⃣ 试运行采集（1分钟）

```bash
python3 examples/so101_sim/teleop_record_block_stacking.py \
    --repo_id test/block_stacking_test \
    --num_episodes 1 \
    --port /dev/ttyACM0 \
    --leader_id my_awesome_leader_arm \
    --leader_calibration_dir ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader \
    --single_camera \
    --display
```

**检查**：数据是否正确记录

### 3️⃣ 正式采集（根据需求选择）

#### 快速方案（单相机）
```bash
python3 examples/so101_sim/teleop_record_block_stacking.py \
    --repo_id your_username/so101_block_stacking \
    --num_episodes 50 \
    --port /dev/ttyACM0 \
    --leader_id my_awesome_leader_arm \
    --leader_calibration_dir ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader \
    --single_camera \
    --camera_width 256 \
    --camera_height 256 \
    --randomize_blocks \
    --display
```

#### 完整方案（三相机）
```bash
python3 examples/so101_sim/teleop_record_block_stacking.py \
    --repo_id your_username/so101_block_stacking \
    --num_episodes 50 \
    --port /dev/ttyACM0 \
    --leader_id my_awesome_leader_arm \
    --leader_calibration_dir ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader \
    --camera_width 128 \
    --camera_height 128 \
    --randomize_blocks \
    --display
```

## 📊 性能对比

| 模式 | 相机 | 分辨率 | 速度 | 推荐用途 |
|------|------|--------|------|---------|
| 快速测试 | 0 | - | ⚡⚡⚡⚡⚡ | 测试控制 |
| 单相机 | 1 | 256 | ⚡⚡⚡⚡ | 快速采集 |
| 三相机低分辨率 | 3 | 128 | ⚡⚡⚡ | 平衡方案 |
| 三相机高分辨率 | 3 | 256 | ⚡⚡ | 最佳质量 |

## 🔧 常用参数

### 必需参数
```bash
--repo_id test/dataset_name          # 数据集名称
--port /dev/ttyACM0                  # 串口（可能是ttyUSB0）
--leader_id my_awesome_leader_arm    # 标定文件名（不含.json）
--leader_calibration_dir ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader
```

### 性能优化
```bash
--single_camera                      # 只用顶部相机（3倍速度提升）
--camera_width 128                   # 降低分辨率（2倍速度提升）
--camera_height 128
--fps 5                              # 降低采样率
```

### 数据增强
```bash
--randomize_blocks                   # 随机化积木位置
--num_episodes 50                    # 采集数量
--episode_time_s 60                  # 每次最长时间
```

### 可视化
```bash
--display                            # 显示MuJoCo窗口
```

## 📁 重要文件

### 文档
- `START_HERE.md` - 项目总览
- `CHECKLIST.md` - 进度清单
- `examples/so101_sim/TELEOPERATION_GUIDE.md` - 详细遥操作指南
- `examples/so101_sim/BLOCK_STACKING_README.md` - 环境使用文档

### 脚本
- `teleop_test_fast.py` - 快速测试（无相机）
- `teleop_record_block_stacking.py` - 数据采集
- `visualize_cameras.py` - 相机可视化
- `test_block_stacking_env.py` - 环境测试

### 配置
- `Sim_assets/SO-ARM100/Simulation/SO101/so101_block_stacking.xml` - 场景文件
- `~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/my_awesome_leader_arm.json` - 标定文件

## ⚠️ 注意事项

### 避免段错误
1. ✅ 不要使用Ctrl+C强制中断
2. ✅ 让程序自然结束
3. ✅ 已禁用Rerun可视化

### 减少延迟
1. ✅ 使用 `--single_camera`
2. ✅ 降低分辨率到128×128
3. ✅ 关闭其他GPU程序

### 标定问题
1. ✅ 必须指定 `--leader_id`
2. ✅ 必须指定 `--leader_calibration_dir`
3. ✅ ID要与文件名匹配（不含.json）

## 🎓 学习路径

```
1. 阅读 START_HERE.md
   ↓
2. 运行快速测试（teleop_test_fast.py）
   ↓
3. 试运行数据采集（1个episode）
   ↓
4. 查看 TELEOPERATION_GUIDE.md 选择方案
   ↓
5. 正式采集数据（50+ episodes）
   ↓
6. 训练SmolVLA模型
   ↓
7. 评估和优化
```

## 💡 提示

- **首次使用**：先运行快速测试，确认控制正常
- **性能不足**：使用单相机模式或降低分辨率
- **数据质量**：三相机高分辨率最好，但采集较慢
- **平衡方案**：三相机128×128是不错的选择

## 📞 获取帮助

1. 查看 `TELEOPERATION_GUIDE.md` 的常见问题部分
2. 查看 `BLOCK_STACKING_README.md` 的故障排除
3. 检查 `CHECKLIST.md` 确认完成的步骤

---

**当前状态**: ✅ 环境就绪，可以开始数据采集

**下一步**: 运行快速测试，然后开始采集数据
