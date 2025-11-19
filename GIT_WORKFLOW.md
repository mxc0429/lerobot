# Git 双分支开发工作流

## 项目结构

```
main (主分支)
├── mujoco-sim-cleanup (仿真环境修改)
└── smolvla-model-update (模型结构修改)
```

## 本地开发流程

### 1. 创建两个功能分支

```bash
# 当前在 mujoco-sim-cleanup 分支
# 已经完成仿真环境的修改

# 切换回 main 分支
git checkout main

# 创建模型修改分支
git checkout -b smolvla-model-update

# 查看所有分支
git branch -a
```

### 2. 在各自分支上开发

#### 分支 A: mujoco-sim-cleanup (仿真环境)
```bash
# 切换到仿真分支
git checkout mujoco-sim-cleanup

# 进行修改...
# 修改 src/lerobot/envs/so101_mujoco/env.py
# 修改 examples/so101_sim/*.py

# 查看修改
git status

# 添加修改
git add src/lerobot/envs/so101_mujoco/env.py
git add examples/so101_sim/

# 提交
git commit -m "Fix: 修复 MuJoCo XML 加载路径问题"

# 推送到远程
git push origin mujoco-sim-cleanup
```

#### 分支 B: smolvla-model-update (模型结构)
```bash
# 切换到模型分支
git checkout smolvla-model-update

# 进行修改...
# 修改 src/lerobot/policies/smolvla/modeling_smolvla.py
# 修改 src/lerobot/policies/smolvla/smolvlm_with_expert.py

# 查看修改
git status

# 添加修改
git add src/lerobot/policies/smolvla/

# 提交
git commit -m "Feature: 更新 SmolVLA 模型结构"

# 推送到远程
git push -u origin smolvla-model-update
```

### 3. 测试各分支

```bash
# 测试仿真分支
git checkout mujoco-sim-cleanup
python examples/so101_sim/test_sim_setup.py --render

# 测试模型分支
git checkout smolvla-model-update
python -c "from lerobot.policies.smolvla import SmolVLA; print('模型导入成功')"
```

### 4. 合并到主分支

#### 方式 A: 本地合并（推荐用于小团队）

```bash
# 切换到 main 分支
git checkout main

# 拉取最新的 main
git pull origin main

# 合并仿真分支
git merge mujoco-sim-cleanup
# 如果有冲突，解决后：
# git add .
# git commit -m "Merge: 合并仿真环境修改"

# 合并模型分支
git merge smolvla-model-update
# 如果有冲突，解决后：
# git add .
# git commit -m "Merge: 合并模型结构修改"

# 推送到远程
git push origin main
```

#### 方式 B: GitHub Pull Request（推荐用于团队协作）

1. 在 GitHub 网页上创建 PR：
   - `mujoco-sim-cleanup` → `main`
   - `smolvla-model-update` → `main`

2. 审查代码后点击 "Merge Pull Request"

3. 本地同步：
```bash
git checkout main
git pull origin main
```

## 服务器端同步

### 首次克隆（服务器上）

```bash
# 克隆仓库
git clone https://github.com/your-username/lerobot.git
cd lerobot

# 查看所有分支
git branch -a
```

### 拉取特定分支进行测试

#### 测试仿真分支

```bash
# 切换到仿真分支
git checkout mujoco-sim-cleanup

# 拉取最新代码
git pull origin mujoco-sim-cleanup

# 测试
python examples/so101_sim/test_sim_setup.py --render
```

#### 测试模型分支

```bash
# 切换到模型分支
git checkout smolvla-model-update

# 拉取最新代码
git pull origin smolvla-model-update

# 测试
python src/lerobot/scripts/lerobot_train.py --config your_config.yaml
```

#### 同时测试两个分支（合并后的效果）

```bash
# 创建临时测试分支
git checkout main
git checkout -b test-combined

# 合并两个功能分支
git merge mujoco-sim-cleanup
git merge smolvla-model-update

# 解决冲突（如果有）
# 测试完整功能
python examples/so101_sim/test_sim_setup.py --render
python src/lerobot/scripts/lerobot_train.py --config your_config.yaml

# 测试完成后删除临时分支
git checkout main
git branch -D test-combined
```

### 更新已有分支

```bash
# 查看当前分支
git branch

# 拉取所有远程分支的更新
git fetch origin

# 更新当前分支
git pull

# 或者更新特定分支
git checkout mujoco-sim-cleanup
git pull origin mujoco-sim-cleanup
```

## 常用命令速查

### 分支管理

```bash
# 查看本地分支
git branch

# 查看所有分支（包括远程）
git branch -a

# 创建新分支
git checkout -b branch-name

# 切换分支
git checkout branch-name

# 删除本地分支
git branch -d branch-name

# 删除远程分支
git push origin --delete branch-name
```

### 提交管理

```bash
# 查看状态
git status

# 添加文件
git add file1 file2
git add .  # 添加所有修改

# 提交
git commit -m "提交信息"

# 修改最后一次提交
git commit --amend

# 查看提交历史
git log --oneline -10
```

### 同步操作

```bash
# 拉取远程更新
git pull origin branch-name

# 推送到远程
git push origin branch-name

# 强制推送（谨慎使用）
git push -f origin branch-name

# 获取远程分支信息
git fetch origin
```

### 合并操作

```bash
# 合并指定分支到当前分支
git merge branch-name

# 取消合并
git merge --abort

# 查看合并冲突
git diff

# 解决冲突后
git add .
git commit -m "解决合并冲突"
```

## 完整工作流示例

### 场景：开发 → 测试 → 合并

```bash
# ========== 本地开发 ==========

# 1. 在仿真分支开发
git checkout mujoco-sim-cleanup
# 修改代码...
git add .
git commit -m "Fix: 修复 XML 加载问题"
git push origin mujoco-sim-cleanup

# 2. 在模型分支开发
git checkout smolvla-model-update
# 修改代码...
git add .
git commit -m "Feature: 添加新的注意力机制"
git push origin smolvla-model-update

# ========== 服务器测试 ==========

# 在服务器上测试仿真分支
ssh your-server
cd lerobot
git fetch origin
git checkout mujoco-sim-cleanup
git pull origin mujoco-sim-cleanup
python examples/so101_sim/test_sim_setup.py --render
# 测试通过 ✓

# 在服务器上测试模型分支
git checkout smolvla-model-update
git pull origin smolvla-model-update
python src/lerobot/scripts/lerobot_train.py --config test.yaml
# 测试通过 ✓

# ========== 合并到主分支 ==========

# 回到本地
exit

# 合并到 main
git checkout main
git pull origin main
git merge mujoco-sim-cleanup
git merge smolvla-model-update
# 解决冲突（如果有）
git push origin main

# ========== 服务器同步主分支 ==========

# 服务器拉取最新 main
ssh your-server
cd lerobot
git checkout main
git pull origin main
# 最终测试
python examples/so101_sim/test_sim_setup.py --render
python src/lerobot/scripts/lerobot_train.py --config final.yaml
```

## 冲突解决

### 当出现合并冲突时

```bash
# 1. 查看冲突文件
git status

# 2. 打开冲突文件，会看到：
# <<<<<<< HEAD
# 当前分支的代码
# =======
# 要合并分支的代码
# >>>>>>> branch-name

# 3. 手动编辑，保留需要的代码，删除标记

# 4. 标记为已解决
git add conflicted-file.py

# 5. 完成合并
git commit -m "解决合并冲突"
```

## 最佳实践

1. **频繁提交**：小步提交，便于回滚
2. **清晰的提交信息**：使用 `Fix:`、`Feature:`、`Refactor:` 等前缀
3. **推送前测试**：确保代码可运行
4. **及时同步**：经常 `git pull` 避免大冲突
5. **分支命名规范**：
   - `feature/xxx` - 新功能
   - `fix/xxx` - 修复
   - `refactor/xxx` - 重构
   - `test/xxx` - 测试

## 当前项目分支说明

| 分支名 | 用途 | 主要修改文件 |
|--------|------|-------------|
| `main` | 主分支，稳定版本 | - |
| `mujoco-sim-cleanup` | 仿真环境修改 | `src/lerobot/envs/so101_mujoco/env.py`<br>`examples/so101_sim/*.py` |
| `smolvla-model-update` | 模型结构修改 | `src/lerobot/policies/smolvla/*.py` |

## 快速参考

```bash
# 本地：切换到仿真分支开发
git checkout mujoco-sim-cleanup

# 本地：切换到模型分支开发
git checkout smolvla-model-update

# 服务器：拉取仿真分支测试
git checkout mujoco-sim-cleanup && git pull

# 服务器：拉取模型分支测试
git checkout smolvla-model-update && git pull

# 本地：合并到主分支
git checkout main && git merge mujoco-sim-cleanup && git merge smolvla-model-update

# 服务器：同步主分支
git checkout main && git pull
```
