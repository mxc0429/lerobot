# 🎯 SO-101 积木堆叠项目 - 从这里开始

## 🎉 恭喜！仿真环境已经搭建完成

你的SO-101积木堆叠仿真环境已经准备就绪，包含：
- ✅ 桌子和3个彩色积木块
- ✅ 三个相机视角（顶部、腕部、右侧）
- ✅ 完整的测试和数据采集工具
- ✅ 详细的文档和指南

## 📚 文档导航

### 1️⃣ 快速开始
**文件**: `CHECKLIST.md`
- 查看已完成的工作
- 了解下一步操作
- 跟踪项目进度

### 2️⃣ 详细使用指南
**文件**: `examples/so101_sim/BLOCK_STACKING_README.md`
- 环境特性说明
- 详细使用方法
- 训练SmolVLA指导
- 故障排除

### 3️⃣ 技术实现总结
**文件**: `BLOCK_STACKING_IMPLEMENTATION_SUMMARY.md`
- 完整的技术细节
- 性能考虑
- 已知限制和改进方向

## 🚀 立即开始

### 第一步：查看相机图像

```bash
# 查看生成的相机图像
ls -lh camera_test_output/

# 使用图像查看器打开（根据你的系统）
# Linux:
eog camera_test_output/combined_view.png
# 或
xdg-open camera_test_output/combined_view.png

# macOS:
# open camera_test_output/combined_view.png

# Windows:
# start camera_test_output/combined_view.png
```

**检查内容**：
- ✅ 顶部相机能看到整个工作区域
- ✅ 腕部相机视角合理
- ✅ 右侧相机能看到机械臂和积木
- ✅ 三个积木块清晰可见

### 第二步：测试随机化

```bash
# 生成几组随机配置的图像
python3 examples/so101_sim/visualize_cameras.py --randomize --output_dir test1
python3 examples/so101_sim/visualize_cameras.py --randomize --output_dir test2
python3 examples/so101_sim/visualize_cameras.py --randomize --output_dir test3

# 查看积木块位置是否合理变化
```

### 第三步：准备数据采集

如果相机视角满意，就可以开始数据采集了！

```bash
# 连接leader臂
# 确认端口（通常是 /dev/ttyUSB0）
ls /dev/ttyUSB*

# 试运行（采集1个测试episode）
python3 examples/so101_sim/teleop_record_block_stacking.py \
    --repo_id test/so101_block_stacking_test \
    --num_episodes 1 \
    --episode_time_s 30 \
    --port /dev/ttyUSB0 \
    --display
```

## 📋 项目文件结构

```
lerobot/
├── START_HERE.md                                    # 👈 你在这里
├── CHECKLIST.md                                     # 检查清单
├── BLOCK_STACKING_IMPLEMENTATION_SUMMARY.md         # 实现总结
│
├── Sim_assets/SO-ARM100/Simulation/SO101/
│   └── so101_block_stacking.xml                     # 新场景文件
│
├── src/lerobot/envs/so101_mujoco/
│   └── env.py                                       # 更新的环境类
│
├── examples/so101_sim/
│   ├── BLOCK_STACKING_README.md                     # 详细使用文档
│   ├── quick_start.sh                               # 快速测试脚本
│   ├── test_block_stacking_env.py                   # 环境测试
│   ├── visualize_cameras.py                         # 相机可视化
│   └── teleop_record_block_stacking.py              # 数据采集
│
└── camera_test_output/                              # 生成的测试图像
    ├── top_cam.png
    ├── wrist_cam.png
    ├── right_cam.png
    └── combined_view.png
```

## 🎯 当前状态

```
✅ 阶段一：仿真环境丰富 - 已完成
   ├── ✅ 桌子和积木块
   ├── ✅ 三个相机视角
   ├── ✅ 物理仿真
   └── ✅ 测试工具

⏳ 阶段二：数据采集 - 准备开始
   ├── ⏳ 连接leader臂
   ├── ⏳ 采集50-100个演示
   └── ⏳ 上传到Hugging Face

⏳ 阶段三：模型训练 - 等待数据
   ├── ⏳ 配置SmolVLA
   ├── ⏳ 训练模型
   └── ⏳ 监控性能

⏳ 阶段四：模型评估 - 等待训练
   ├── ⏳ 仿真测试
   ├── ⏳ 真实环境测试
   └── ⏳ 性能优化
```

## 💡 快速命令参考

### 测试环境
```bash
# 基础测试
python3 examples/so101_sim/test_block_stacking_env.py --no_render --num_steps 100

# 可视化测试
python3 examples/so101_sim/test_block_stacking_env.py --num_steps 500

# 随机化测试
python3 examples/so101_sim/test_block_stacking_env.py --randomize --num_steps 500
```

### 生成相机图像
```bash
# 标准配置
python3 examples/so101_sim/visualize_cameras.py --output_dir ./views

# 随机配置
python3 examples/so101_sim/visualize_cameras.py --randomize --output_dir ./views_random
```

### 数据采集
```bash
# 测试运行
python3 examples/so101_sim/teleop_record_block_stacking.py \
    --repo_id test/block_stacking \
    --num_episodes 2 \
    --port /dev/ttyUSB0 \
    --display

# 正式采集
python3 examples/so101_sim/teleop_record_block_stacking.py \
    --repo_id your_username/so101_block_stacking \
    --num_episodes 50 \
    --episode_time_s 60 \
    --fps 10 \
    --port /dev/ttyUSB0 \
    --camera_width 256 \
    --camera_height 256 \
    --randomize_blocks \
    --display \
    --push_to_hub
```

## ❓ 常见问题

### Q1: 相机视角不满意怎么办？
**A**: 编辑 `Sim_assets/SO-ARM100/Simulation/SO101/so101_block_stacking.xml`，找到相机定义部分，调整 `pos` 和 `quat` 参数。

### Q2: 积木块位置需要调整？
**A**: 在XML文件中修改积木块的初始 `pos`，或者调整 `env.py` 中 `_randomize_blocks()` 的范围。

### Q3: 需要更多或更少的积木块？
**A**: 
1. 在XML中添加/删除积木块body定义
2. 更新 `num_blocks` 配置参数
3. 更新 `get_block_positions()` 中的积木块名称列表

### Q4: 如何调整相机分辨率？
**A**: 使用 `--camera_width` 和 `--camera_height` 参数。推荐：
- 训练：256×256（平衡质量和性能）
- 测试：128×128（更快）
- 可视化：640×480（更清晰）

### Q5: 数据采集时leader臂不响应？
**A**: 
1. 检查USB连接：`ls /dev/ttyUSB*`
2. 检查端口权限：`sudo chmod 666 /dev/ttyUSB0`
3. 确认leader臂已校准（不使用 `--skip_calibration`）

## 📞 获取帮助

1. **查看详细文档**: `examples/so101_sim/BLOCK_STACKING_README.md`
2. **查看实现细节**: `BLOCK_STACKING_IMPLEMENTATION_SUMMARY.md`
3. **查看进度清单**: `CHECKLIST.md`
4. **LeRobot官方文档**: https://huggingface.co/docs/lerobot
5. **SmolVLA论文**: https://arxiv.org/abs/2506.01844

## 🎊 准备好了吗？

如果相机图像看起来不错，你就可以：

1. ✅ 连接leader臂
2. ✅ 开始采集数据
3. ✅ 训练SmolVLA模型
4. ✅ 在仿真和真实环境中测试

**祝你好运！🚀**

---

**提示**: 如果你是第一次使用，建议按顺序阅读：
1. 本文档（START_HERE.md）
2. 检查清单（CHECKLIST.md）
3. 详细使用指南（examples/so101_sim/BLOCK_STACKING_README.md）
