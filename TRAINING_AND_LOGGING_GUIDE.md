# 训练日志记录和可视化指南

## 🎯 概述

本指南介绍如何训练模型并记录详细日志，以便后续绘制对比图表。

## 📝 问题解决

### 1. 记忆 Token 参数日志输出

**问题**: 训练时没有显示 `num_mem_tokens` 参数

**解决**: 已在 `lerobot_train.py` 中添加记忆模块配置的日志输出

**现在的输出**:
```
INFO Creating policy
Loading HuggingFaceTB/SmolVLM2-500M-Video-Instruct weights ...
Reducing the number of VLM layers to 16 ...
INFO Output dir: outputs/train/smolvla_with_memory_4tokens
INFO cfg.steps=100000 (100K)
INFO dataset.num_frames=37919 (38K)
INFO dataset.num_episodes=50
INFO Effective batch size: 4 x 1 = 4
INFO num_learnable_params=99884832 (100M)
INFO num_total_params=450050016 (450M)
INFO Memory Module: ENABLED                    # 🆕 新增
INFO   num_mem_tokens=4                        # 🆕 新增
INFO   mem_at_end=False                        # 🆕 新增
INFO   read_mem_from_cache=False               # 🆕 新增
INFO   memory_params=3,840 (0.0009% of total) # 🆕 新增
INFO Start offline training on a fixed dataset
```

### 2. 关于 "Missing key(s)" 警告

**警告信息**:
```
WARNING Missing key(s) when loading model: {'model.mem_tokens'}
```

**原因**: 
- 你从预训练的 `smolvla_base` 加载模型
- 预训练模型没有 `mem_tokens` 参数（因为它是新添加的）
- 这是**正常的**，不影响训练

**解决**: 
- 这个警告可以忽略
- 新的 `mem_tokens` 会被随机初始化
- 训练会正常进行

---

## 🚀 使用方法

### 方法 1: 使用增强的训练脚本（推荐）

```bash
# 设置环境变量
export HF_USER="your_username"

# 训练基线模型
bash train_with_logging.sh baseline 0 4 100000

# 训练记忆增强模型（4 个 memory tokens）
bash train_with_logging.sh memory 4 4 100000
```

**参数说明**:
- 参数 1: 模型类型 (`baseline` 或 `memory`)
- 参数 2: 记忆 token 数量 (0 表示禁用)
- 参数 3: 批次大小
- 参数 4: 训练步数

### 方法 2: 使用原始训练命令

```bash
# 训练基线模型
lerobot-train \
  --policy.path=./smolvla_base \
  --dataset.repo_id=${HF_USER}/pickplace_smolvla \
  --batch_size=4 \
  --output_dir=outputs/train/smolvla_baseline \
  --job_name=smolvla_baseline \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.num_mem_tokens=0 \
  --steps=100000 \
  2>&1 | tee outputs/train/smolvla_baseline/training.log

# 训练记忆增强模型
lerobot-train \
  --policy.path=./smolvla_base \
  --dataset.repo_id=${HF_USER}/pickplace_smolvla \
  --batch_size=4 \
  --output_dir=outputs/train/smolvla_with_memory_4tokens \
  --job_name=smolvla_memory_4tokens \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.num_mem_tokens=4 \
  --steps=100000 \
  2>&1 | tee outputs/train/smolvla_with_memory_4tokens/training.log
```

**关键点**:
- `2>&1 | tee <log_file>`: 同时输出到终端和日志文件
- `--policy.num_mem_tokens=4`: 启用 4 个记忆 tokens
- `--log_freq=100`: 每 100 步记录一次日志

---

## 📊 绘制训练曲线

### 训练完成后，绘制对比图

```bash
python plot_training_curves.py \
  --baseline_log outputs/train/smolvla_baseline/training.log \
  --memory_log outputs/train/smolvla_with_memory_4tokens/training.log \
  --output_dir plots/
```

### 生成的图表

脚本会生成以下图表：

1. **loss_comparison.png** - 训练损失对比
2. **lr_comparison.png** - 学习率对比
3. **grad_norm_comparison.png** - 梯度范数对比
4. **time_comparison.png** - 训练时间对比
5. **overview_comparison.png** - 综合对比（4 合 1）

### 示例输出

```
📖 Parsing log files...
  Baseline: 1000 data points
  Memory:   1000 data points

📈 Creating comparison plots...
✅ Saved: plots/loss_comparison.png
✅ Saved: plots/lr_comparison.png
✅ Saved: plots/grad_norm_comparison.png
✅ Saved: plots/time_comparison.png
✅ Saved: plots/overview_comparison.png

================================================================================
TRAINING STATISTICS SUMMARY
================================================================================

📊 Final Loss (avg of last 100 steps):
  Baseline:    0.123456
  With Memory: 0.118234
  Improvement: +4.23%

⚡ Average Update Time:
  Baseline:    0.1234 s
  With Memory: 0.1256 s
  Overhead:    +1.78%

================================================================================

✅ All plots saved to: plots
```

---

## 📈 日志文件格式

### 训练日志示例

```
INFO 2025-11-24 10:32:59 Creating dataset
INFO 2025-11-24 10:32:59 Creating policy
INFO 2025-11-24 10:33:20 Memory Module: ENABLED
INFO 2025-11-24 10:33:20   num_mem_tokens=4
INFO 2025-11-24 10:33:20   memory_params=3,840 (0.0009% of total)
INFO 2025-11-24 10:33:20 Start offline training on a fixed dataset
INFO 2025-11-24 10:33:25 Step 100: loss=0.1234 lr=1e-4 grdn=0.5 updt_s=0.123 data_s=0.045
INFO 2025-11-24 10:33:30 Step 200: loss=0.1156 lr=9.8e-5 grdn=0.48 updt_s=0.121 data_s=0.043
...
```

### 指标说明

- **step**: 训练步数
- **loss**: 训练损失
- **lr**: 学习率
- **grdn**: 梯度范数
- **updt_s**: 更新时间（秒）
- **data_s**: 数据加载时间（秒）

---

## 🔍 监控训练进度

### 实时查看日志

```bash
# 实时查看训练日志
tail -f outputs/train/smolvla_with_memory_4tokens/training.log

# 只查看包含 "Step" 的行
tail -f outputs/train/smolvla_with_memory_4tokens/training.log | grep "Step"

# 查看最近的损失值
tail -f outputs/train/smolvla_with_memory_4tokens/training.log | grep "loss"
```

### 检查训练状态

```bash
# 查看最后 20 行日志
tail -n 20 outputs/train/smolvla_with_memory_4tokens/training.log

# 搜索错误信息
grep -i "error\|warning\|failed" outputs/train/smolvla_with_memory_4tokens/training.log

# 统计训练步数
grep -c "Step" outputs/train/smolvla_with_memory_4tokens/training.log
```

---

## 📁 输出文件结构

```
outputs/train/
├── smolvla_baseline/
│   ├── training.log              # 训练日志
│   ├── metrics.csv               # 指标 CSV（如果使用增强脚本）
│   ├── checkpoints/
│   │   ├── 010000/
│   │   ├── 020000/
│   │   └── last/
│   └── config.json
│
└── smolvla_with_memory_4tokens/
    ├── training.log
    ├── metrics.csv
    ├── checkpoints/
    │   ├── 010000/
    │   ├── 020000/
    │   └── last/
    └── config.json

plots/
├── loss_comparison.png
├── lr_comparison.png
├── grad_norm_comparison.png
├── time_comparison.png
└── overview_comparison.png
```

---

## 🎨 自定义绘图

### 修改绘图脚本

你可以编辑 `plot_training_curves.py` 来自定义图表：

```python
# 修改平滑窗口大小
def smooth_curve(values: List[float], window: int = 100):  # 改为 50 或 200
    ...

# 修改颜色
colors = {
    'baseline': '#1f77b4',  # 蓝色
    'memory': '#ff7f0e'     # 橙色
}

# 修改图表大小
fig, ax = plt.subplots(figsize=(12, 6))  # 改为 (16, 8)
```

### 添加更多指标

在 `parse_log_file` 函数中添加新的正则表达式来提取其他指标。

---

## 💡 最佳实践

### 1. 训练前

- ✅ 确保设置了 `HF_USER` 环境变量
- ✅ 检查 GPU 内存是否足够
- ✅ 确认数据集路径正确
- ✅ 创建输出目录

### 2. 训练中

- ✅ 定期检查日志文件
- ✅ 监控 GPU 使用率
- ✅ 注意损失是否收敛
- ✅ 检查是否有错误或警告

### 3. 训练后

- ✅ 保存完整的日志文件
- ✅ 绘制训练曲线
- ✅ 对比不同配置的结果
- ✅ 记录最佳超参数

---

## 🐛 常见问题

### Q1: 日志文件太大怎么办？

**A**: 使用日志轮转或压缩

```bash
# 压缩旧日志
gzip outputs/train/smolvla_baseline/training.log

# 只保留最近的日志
tail -n 10000 training.log > training_recent.log
```

### Q2: 如何对比多个模型？

**A**: 修改绘图脚本支持多个模型

```python
# 在 plot_training_curves.py 中添加第三个模型
parser.add_argument("--memory8_log", type=str, help="8 tokens model log")
```

### Q3: 训练中断后如何继续？

**A**: 使用 `--resume` 参数

```bash
lerobot-train \
  --config_path=outputs/train/smolvla_with_memory_4tokens/config.json \
  --resume=true \
  2>&1 | tee -a outputs/train/smolvla_with_memory_4tokens/training.log
```

注意使用 `tee -a` (append) 而不是 `tee` (overwrite)。

---

## 📚 相关文档

- [SMOLVLA_MEMORY_GUIDE.md](SMOLVLA_MEMORY_GUIDE.md) - 详细使用指南
- [MEMORY_IMPLEMENTATION_DETAILS.md](MEMORY_IMPLEMENTATION_DETAILS.md) - 实现细节
- [evaluate_models.py](evaluate_models.py) - 模型评估脚本

---

## ✅ 快速检查清单

训练前：
- [ ] 设置 `HF_USER` 环境变量
- [ ] 确认 GPU 可用
- [ ] 检查数据集路径
- [ ] 创建输出目录

训练中：
- [ ] 监控日志输出
- [ ] 检查损失是否下降
- [ ] 注意内存使用
- [ ] 定期保存检查点

训练后：
- [ ] 保存日志文件
- [ ] 绘制训练曲线
- [ ] 对比模型性能
- [ ] 记录实验结果

---

**现在你可以开始训练并记录详细日志了！** 🚀
