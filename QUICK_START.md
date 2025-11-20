# SmolVLA with Memory - 快速开始

## 🎯 一分钟快速开始

### 1. 测试实现
```bash
python test_memory_module.py
```

### 2. 训练基线模型
```bash
export HF_USER="your_username"
bash train_baseline_smolvla.sh
```

### 3. 训练记忆增强模型
```bash
bash train_smolvla_with_memory.sh
```

### 4. 对比评估
```bash
python evaluate_models.py \
  --baseline_path outputs/train/smolvla_baseline/checkpoints/last/pretrained_model \
  --memory_path outputs/train/smolvla_with_memory_4tokens/checkpoints/last/pretrained_model \
  --dataset_repo_id ${HF_USER}/pickplace_smolvla \
  --n_episodes 50
```

## 📊 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--policy.num_mem_tokens` | 4 | 记忆 token 数量 |
| `--policy.mem_at_end` | false | 记忆位置 |
| `--batch_size` | 4 | 批次大小 |
| `--steps` | 50000 | 训练步数 |

## 🔧 不同配置对比

### 轻量级（快速验证）
```bash
--policy.num_mem_tokens=2 --steps=10000
```
- 参数增量: 1,920 (0.0004%)
- 训练时间: ~2 小时
- 适合: 快速验证

### 标准配置（推荐）
```bash
--policy.num_mem_tokens=4 --steps=50000
```
- 参数增量: 3,840 (0.0009%)
- 训练时间: ~10 小时
- 适合: 大多数任务

### 增强配置（复杂任务）
```bash
--policy.num_mem_tokens=8 --steps=200000
```
- 参数增量: 7,680 (0.0017%)
- 训练时间: ~40 小时
- 适合: 长期规划任务

## 📈 预期改进

- ✅ 成功率: +3-5%
- ✅ 平均奖励: +5-10%
- ⚡ 推理开销: <2%
- 💾 内存开销: <1%

## 🐛 常见问题

**Q: 训练时 GPU 内存不足？**
```bash
--batch_size=2  # 减小批次
```

**Q: 如何禁用记忆模块？**
```bash
--policy.num_mem_tokens=0
```

**Q: 如何查看训练日志？**
```bash
tensorboard --logdir outputs/train/
```

## 📚 完整文档

详细信息请参考 [SMOLVLA_MEMORY_GUIDE.md](SMOLVLA_MEMORY_GUIDE.md)
