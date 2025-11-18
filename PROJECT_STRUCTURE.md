# SO-101 积木堆叠项目 - 最终文件结构

## 📁 核心文件

### 场景配置
```
Sim_assets/SO-ARM100/Simulation/SO101/
└── so101_block_stacking.xml          # 积木堆叠场景（桌子+积木+3相机）
```

### 环境代码
```
src/lerobot/envs/so101_mujoco/
└── env.py                             # MuJoCo环境类（支持多相机、积木管理）
```

### 数据采集脚本
```
examples/so101_sim/
├── teleop_record.py                   # 原始遥操作脚本（单相机）
├── teleop_record_block_stacking.py    # 积木堆叠数据采集（基础版）
├── teleop_record_ultra_fast.py        # 双环境策略（推荐，低延迟+高质量）
└── teleop_record_minimal.py           # 最小延迟版本（无相机预览）
```

### 测试和可视化
```
examples/so101_sim/
├── test_block_stacking_env.py         # 环境测试脚本
└── visualize_cameras.py               # 相机可视化工具
```

### 文档
```
根目录/
├── START_HERE.md                      # 项目入口文档
├── QUICK_REFERENCE.md                 # 快速参考
├── CHECKLIST.md                       # 项目进度清单
├── BLOCK_STACKING_IMPLEMENTATION_SUMMARY.md  # 实现总结
├── So101仿真环境.md                   # 中文环境文档
└── So101使用文档.md                   # 中文使用文档

examples/so101_sim/
├── BLOCK_STACKING_README.md           # 详细使用指南
└── TELEOPERATION_GUIDE.md             # 遥操作指南
```

## 🎯 推荐使用的脚本

### 1. 测试环境
```bash
python3 examples/so101_sim/test_block_stacking_env.py --no_render --num_steps 100
```

### 2. 查看相机
```bash
python3 examples/so101_sim/visualize_cameras.py --output_dir ./camera_views
```

### 3. 数据采集（推荐）

#### 选项A：最小延迟（无相机预览）
```bash
python3 examples/so101_sim/teleop_record_minimal.py \
    --repo_id your_username/so101_block_stacking \
    --num_episodes 50 \
    --port /dev/ttyACM0 \
    --leader_id my_awesome_leader_arm \
    --leader_calibration_dir ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader \
    --camera_width 256 \
    --camera_height 256 \
    --record_fps 10
```

#### 选项B：双环境策略（低延迟+相机预览）
```bash
python3 examples/so101_sim/teleop_record_ultra_fast.py \
    --repo_id your_username/so101_block_stacking \
    --num_episodes 50 \
    --port /dev/ttyACM0 \
    --leader_id my_awesome_leader_arm \
    --leader_calibration_dir ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader \
    --camera_width 256 \
    --camera_height 256 \
    --record_fps 10
```

## 📊 脚本对比

| 脚本 | 延迟 | 相机预览 | 推荐场景 |
|------|------|---------|---------|
| `teleop_record_minimal.py` | 最低 | 无 | **低性能GPU** |
| `teleop_record_ultra_fast.py` | 低 | 有 | 中高性能GPU |
| `teleop_record_block_stacking.py` | 中 | 有 | 测试用 |

## 🗑️ 已删除的文件

### 测试脚本（已删除）
- teleop_test_fast.py
- test_camera_views.py
- test_calibration.py
- test_leader_calibration.py
- show_home_position.py

### 旧版本脚本（已删除）
- teleop_fast_display.py
- teleop_with_camera_display.py
- teleop_high_quality.py
- teleop_optimized.py
- teleop_record_with_preview.py
- teleop_record_fast_preview.py

### 校准相关（已删除）
- calibrate_leader_to_sim.py
- calibrate_step_by_step.py
- calibration_utils.py
- camera_adjustment_guide.py

### 冗余文档（已删除）
- CAMERA_SETUP_GUIDE.md
- PERFORMANCE_AND_CALIBRATION_GUIDE.md
- DETAILED_CALIBRATION_GUIDE.md
- QUICK_START_CALIBRATION.md
- CALIBRATION_COMPLETE.md
- CALIBRATION_QUICK_REF.md
- HOME_POSITION_GUIDE.md
- CAMERA_FIX_SUMMARY.md
- ASYNC_INFERENCE_GUIDE.md

### 临时文件（已删除）
- leader_sim_calibration.json
- leader_sim_calibration_v2.json
- camera_test_output/
- camera_test_output_random/
- camera_final_fix/
- camera_fixed_test/
- camera_fixed_test2/

## 📝 文件说明

### 核心脚本

1. **teleop_record_minimal.py** ⭐
   - 最小延迟版本
   - 无相机实时预览
   - 适合低性能GPU
   - 控制频率：50Hz+

2. **teleop_record_ultra_fast.py** ⭐
   - 双环境策略
   - 有相机实时预览
   - 控制频率：50Hz
   - 录制频率：10Hz

3. **teleop_record_block_stacking.py**
   - 基础版本
   - 单环境
   - 用于测试和学习

### 工具脚本

1. **test_block_stacking_env.py**
   - 测试环境是否正常
   - 验证积木块物理
   - 检查相机配置

2. **visualize_cameras.py**
   - 生成相机图像
   - 验证相机视角
   - 保存测试图片

## 🚀 快速开始

1. **阅读文档**
   ```
   START_HERE.md → QUICK_REFERENCE.md → TELEOPERATION_GUIDE.md
   ```

2. **测试环境**
   ```bash
   python3 examples/so101_sim/test_block_stacking_env.py --no_render
   ```

3. **查看相机**
   ```bash
   python3 examples/so101_sim/visualize_cameras.py --output_dir ./test
   ```

4. **开始采集**
   ```bash
   python3 examples/so101_sim/teleop_record_minimal.py \
       --repo_id test/block_stacking \
       --num_episodes 1 \
       --port /dev/ttyACM0 \
       --leader_id my_awesome_leader_arm \
       --leader_calibration_dir ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader
   ```

## 📞 获取帮助

- **快速参考**: QUICK_REFERENCE.md
- **详细指南**: examples/so101_sim/TELEOPERATION_GUIDE.md
- **实现细节**: BLOCK_STACKING_IMPLEMENTATION_SUMMARY.md
- **项目进度**: CHECKLIST.md

---

**当前状态**: ✅ 项目清理完成，只保留必要文件

**下一步**: 开始数据采集和模型训练
