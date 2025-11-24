# 训练日志快速参考

## 🚀 快速开始

### 1. 训练并记录日志

```bash
# 设置用户名
export HF_USER="your_username"

# 训练基线模型
bash train_with_logging.sh baseline 0 4 100000

# 训练记忆增强模型
bash train_with_logging.sh memory 4 4 100000
```

### 2. 绘制对比图

```bash
python plot_training_curves.py \
  --baseline_log outputs/train/smolvla_baseline/training.log \
  --memory_log outputs/train/smolvla_with_memory_4tokens/training.log \
  --output_dir plots/
```

---

## 📊 新增的日志输出

训练时会显示记忆模块配置：

```
INFO Memory Module: ENABLED
INFO   num_mem_tokens=4
INFO   mem_at_end=False
INFO   read_mem_from_cache=False
INFO   memory_params=3,840 (0.0009% of total)
```

---

## ⚠️ 关于 "Missing key(s)" 警告

```
WARNING Missing key(s) when loading model: {'model.mem_tokens'}
```

**这是正常的！** 因为：
- 预训练模型没有 `mem_tokens`
- 新参数会被随机初始化
- 不影响训练

---

## 📈 生成的图表

1. `loss_comparison.png` - 损失对比
2. `lr_comparison.png` - 学习率
3. `grad_norm_comparison.png` - 梯度范数
4. `time_comparison.png` - 训练时间
5. `overview_comparison.png` - 综合对比

---

## 🔍 实时监控

```bash
# 查看训练进度
tail -f outputs/train/smolvla_with_memory_4tokens/training.log

# 只看损失
tail -f outputs/train/smolvla_with_memory_4tokens/training.log | grep "loss"
```

---

## 📁 文件位置

```
outputs/train/
├── smolvla_baseline/
│   └── training.log          ← 基线日志
└── smolvla_with_memory_4tokens/
    └── training.log          ← 记忆模型日志

plots/
└── *.png                     ← 对比图表
```

---

## 💡 完整命令示例

```bash
# 1. 训练基线（约 10 小时）
lerobot-train \
  --policy.path=./smolvla_base \
  --dataset.repo_id=${HF_USER}/pickplace_smolvla \
  --batch_size=4 \
  --output_dir=outputs/train/smolvla_baseline \
  --policy.num_mem_tokens=0 \
  --steps=100000 \
  2>&1 | tee outputs/train/smolvla_baseline/training.log

# 2. 训练记忆模型（约 10 小时）
lerobot-train \
  --policy.path=./smolvla_base \
  --dataset.repo_id=${HF_USER}/pickplace_smolvla \
  --batch_size=4 \
  --output_dir=outputs/train/smolvla_with_memory_4tokens \
  --policy.num_mem_tokens=4 \
  --steps=100000 \
  2>&1 | tee outputs/train/smolvla_with_memory_4tokens/training.log

# 3. 绘制对比图（约 1 分钟）
python plot_training_curves.py \
  --baseline_log outputs/train/smolvla_baseline/training.log \
  --memory_log outputs/train/smolvla_with_memory_4tokens/training.log \
  --output_dir plots/
```

---

详细文档: [TRAINING_AND_LOGGING_GUIDE.md](TRAINING_AND_LOGGING_GUIDE.md)
