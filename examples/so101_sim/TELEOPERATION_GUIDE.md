# SO-101 遥操作指南

## 问题和解决方案

### 问题1：移动延迟

**原因**：
- 三个相机同时渲染很慢（每帧约30-50ms）
- 固定的sleep时间不考虑实际执行时间

**解决方案**：

#### 方案A：使用单相机模式（推荐用于测试）

```bash
python3 examples/so101_sim/teleop_record_block_stacking.py \
    --repo_id test/block_stacking \
    --num_episodes 1 \
    --port /dev/ttyACM0 \
    --leader_id my_awesome_leader_arm \
    --leader_calibration_dir ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader \
    --single_camera \
    --display
```

使用 `--single_camera` 只启用顶部相机，速度提升3倍。

#### 方案B：降低相机分辨率

```bash
python3 examples/so101_sim/teleop_record_block_stacking.py \
    --repo_id test/block_stacking \
    --num_episodes 1 \
    --port /dev/ttyACM0 \
    --leader_id my_awesome_leader_arm \
    --leader_calibration_dir ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader \
    --camera_width 128 \
    --camera_height 128 \
    --display
```

使用128×128分辨率，速度提升约2倍。

#### 方案C：快速测试模式（无相机）

```bash
python3 examples/so101_sim/teleop_test_fast.py \
    --port /dev/ttyACM0 \
    --leader_id my_awesome_leader_arm \
    --duration 60 \
    --display
```

这个脚本完全不使用相机，可以达到50-100Hz的控制频率，用于测试leader臂控制是否正常。

### 问题2：段错误（核心已转储）

**原因**：
- Rerun可视化在程序结束时没有正确关闭
- 多线程渲染冲突

**解决方案**：

已在代码中禁用了Rerun可视化，改用MuJoCo原生窗口：

```python
# 已注释掉
# if args.display:
#     init_rerun(session_name="so101_block_stacking")
```

现在使用 `--display` 只会显示MuJoCo窗口，不会有段错误。

### 问题3：标定文件未找到

**原因**：
- 没有指定 `--leader_id`
- 标定文件路径不正确

**解决方案**：

必须同时指定leader ID和标定目录：

```bash
--leader_id my_awesome_leader_arm \
--leader_calibration_dir ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader
```

## 推荐工作流程

### 1. 快速测试（验证leader臂控制）

```bash
# 无相机，最快速度
python3 examples/so101_sim/teleop_test_fast.py \
    --port /dev/ttyACM0 \
    --leader_id my_awesome_leader_arm \
    --duration 30 \
    --display
```

**检查**：
- Leader臂移动是否流畅
- Follower臂是否跟随
- 控制频率是否足够高（>20Hz）

### 2. 单相机测试（验证数据记录）

```bash
# 只用顶部相机，速度较快
python3 examples/so101_sim/teleop_record_block_stacking.py \
    --repo_id test/block_stacking_single_cam \
    --num_episodes 1 \
    --port /dev/ttyACM0 \
    --leader_id my_awesome_leader_arm \
    --leader_calibration_dir ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader \
    --single_camera \
    --camera_width 256 \
    --camera_height 256 \
    --display
```

**检查**：
- 数据是否正确记录
- 相机图像是否清晰
- 控制延迟是否可接受

### 3. 正式数据采集

#### 选项A：三相机高分辨率（最佳质量，较慢）

```bash
python3 examples/so101_sim/teleop_record_block_stacking.py \
    --repo_id your_username/so101_block_stacking \
    --num_episodes 50 \
    --episode_time_s 60 \
    --fps 10 \
    --port /dev/ttyACM0 \
    --leader_id my_awesome_leader_arm \
    --leader_calibration_dir ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader \
    --camera_width 256 \
    --camera_height 256 \
    --randomize_blocks \
    --display
```

**特点**：
- 三个相机视角
- 256×256分辨率
- 约5-10Hz控制频率
- 最佳训练效果

#### 选项B：三相机低分辨率（平衡）

```bash
python3 examples/so101_sim/teleop_record_block_stacking.py \
    --repo_id your_username/so101_block_stacking \
    --num_episodes 50 \
    --episode_time_s 60 \
    --fps 10 \
    --port /dev/ttyACM0 \
    --leader_id my_awesome_leader_arm \
    --leader_calibration_dir ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader \
    --camera_width 128 \
    --camera_height 128 \
    --randomize_blocks \
    --display
```

**特点**：
- 三个相机视角
- 128×128分辨率
- 约10-15Hz控制频率
- 较好的训练效果，更快的采集速度

#### 选项C：单相机高分辨率（快速）

```bash
python3 examples/so101_sim/teleop_record_block_stacking.py \
    --repo_id your_username/so101_block_stacking_single \
    --num_episodes 50 \
    --episode_time_s 60 \
    --fps 10 \
    --port /dev/ttyACM0 \
    --leader_id my_awesome_leader_arm \
    --leader_calibration_dir ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader \
    --single_camera \
    --camera_width 256 \
    --camera_height 256 \
    --randomize_blocks \
    --display
```

**特点**：
- 只有顶部相机
- 256×256分辨率
- 约15-20Hz控制频率
- 快速采集，但信息较少

## 性能对比

| 配置 | 相机数量 | 分辨率 | 控制频率 | 用途 |
|------|---------|--------|---------|------|
| 快速测试 | 0 | - | 50-100Hz | 测试控制 |
| 单相机高分辨率 | 1 | 256×256 | 15-20Hz | 快速采集 |
| 三相机低分辨率 | 3 | 128×128 | 10-15Hz | 平衡方案 |
| 三相机高分辨率 | 3 | 256×256 | 5-10Hz | 最佳质量 |

## 优化建议

### 1. 硬件优化
- 使用GPU渲染（如果可用）
- 关闭其他占用GPU的程序
- 使用SSD存储数据

### 2. 软件优化
- 降低fps参数（如 `--fps 5`）
- 使用较低分辨率
- 减少相机数量

### 3. 采集策略
- 先用快速模式测试
- 再用单相机采集部分数据
- 最后用三相机采集关键数据

## 常见问题

### Q1: 控制延迟太大怎么办？
**A**: 
1. 使用 `--single_camera`
2. 降低分辨率到128×128
3. 降低fps到5
4. 使用快速测试模式验证硬件性能

### Q2: 段错误如何避免？
**A**: 
1. 不要使用Ctrl+C强制中断
2. 让程序自然结束
3. 如果必须中断，等待几秒让清理完成

### Q3: 数据采集中断了怎么办？
**A**: 
数据会自动保存已完成的episodes，可以继续采集：
```bash
# 继续采集，数据会追加
python3 examples/so101_sim/teleop_record_block_stacking.py \
    --repo_id same/repo_id \
    --num_episodes 10 \
    ...
```

### Q4: 如何查看已采集的数据？
**A**: 
```bash
lerobot-dataset-viz \
    --repo-id your_username/so101_block_stacking \
    --episode-index 0
```

## 脚本对比

| 脚本 | 用途 | 相机 | 速度 | 数据记录 |
|------|------|------|------|---------|
| `teleop_test_fast.py` | 测试控制 | 无 | 最快 | 否 |
| `teleop_record_block_stacking.py` | 数据采集 | 可选 | 中等 | 是 |
| `teleop_record_block_stacking.py --single_camera` | 快速采集 | 1个 | 较快 | 是 |

## 总结

1. **测试阶段**：使用 `teleop_test_fast.py` 验证控制
2. **试运行**：使用 `--single_camera` 采集1-2个episode
3. **正式采集**：根据需求选择相机数量和分辨率
4. **避免段错误**：让程序自然结束，不要强制中断

祝数据采集顺利！🚀
