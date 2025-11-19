# Git 快速参考卡片

## 当前分支状态

```
✓ main                    - 主分支（稳定版本）
✓ mujoco-sim-cleanup      - 仿真环境修改（已推送）
✓ smolvla-model-update    - 模型结构修改（新建）
```

## 本地开发（你的电脑）

### 在仿真分支工作
```bash
git checkout mujoco-sim-cleanup
# 修改 src/lerobot/envs/so101_mujoco/env.py
# 修改 examples/so101_sim/*.py
git add .
git commit -m "Fix: 描述修改内容"
git push origin mujoco-sim-cleanup
```

### 在模型分支工作
```bash
git checkout smolvla-model-update
# 修改 src/lerobot/policies/smolvla/*.py
git add .
git commit -m "Feature: 描述修改内容"
git push origin smolvla-model-update
```

### 查看当前在哪个分支
```bash
git branch
```

## 服务器测试

### 首次设置
```bash
# SSH 到服务器
ssh your-server

# 克隆仓库（只需一次）
git clone https://github.com/your-username/lerobot.git
cd lerobot
```

### 测试仿真分支
```bash
cd lerobot
git fetch origin
git checkout mujoco-sim-cleanup
git pull origin mujoco-sim-cleanup

# 测试
python examples/so101_sim/test_sim_setup.py --render
```

### 测试模型分支
```bash
cd lerobot
git fetch origin
git checkout smolvla-model-update
git pull origin smolvla-model-update

# 测试
python -c "from lerobot.policies.smolvla import SmolVLA; print('OK')"
```

### 测试两个分支合并后的效果
```bash
cd lerobot
git checkout main
git pull origin main

# 创建临时测试分支
git checkout -b test-both
git merge mujoco-sim-cleanup
git merge smolvla-model-update

# 测试完整功能
python examples/so101_sim/test_sim_setup.py --render

# 删除临时分支
git checkout main
git branch -D test-both
```

## 合并到主分支（两个分支都测试通过后）

### 本地合并
```bash
# 切换到 main
git checkout main
git pull origin main

# 合并仿真分支
git merge mujoco-sim-cleanup

# 合并模型分支
git merge smolvla-model-update

# 如果有冲突，解决后：
git add .
git commit -m "Merge: 合并仿真和模型修改"

# 推送
git push origin main
```

### 服务器同步最新 main
```bash
ssh your-server
cd lerobot
git checkout main
git pull origin main
```

## 常见问题

### Q: 我在哪个分支？
```bash
git branch
# * 号标记的就是当前分支
```

### Q: 如何切换分支？
```bash
git checkout branch-name
```

### Q: 修改了代码但想切换分支？
```bash
# 方式1: 提交修改
git add .
git commit -m "WIP: 临时保存"
git checkout other-branch

# 方式2: 暂存修改
git stash
git checkout other-branch
# 回来后恢复
git checkout original-branch
git stash pop
```

### Q: 如何查看修改了什么？
```bash
git status          # 查看修改的文件
git diff            # 查看具体修改内容
git log --oneline   # 查看提交历史
```

### Q: 推送失败？
```bash
# 先拉取远程更新
git pull origin branch-name
# 再推送
git push origin branch-name
```

### Q: 合并冲突怎么办？
```bash
# 1. 查看冲突文件
git status

# 2. 打开文件，找到冲突标记：
#    <<<<<<< HEAD
#    你的代码
#    =======
#    别人的代码
#    >>>>>>> branch-name

# 3. 手动编辑，保留正确的代码，删除标记

# 4. 标记为已解决
git add conflicted-file.py

# 5. 完成合并
git commit
```

## 工作流程图

```
本地开发
  ├─ mujoco-sim-cleanup 分支
  │   ├─ 修改代码
  │   ├─ git commit
  │   └─ git push
  │
  └─ smolvla-model-update 分支
      ├─ 修改代码
      ├─ git commit
      └─ git push
          ↓
服务器测试
  ├─ git checkout mujoco-sim-cleanup
  ├─ git pull
  ├─ 测试 ✓
  │
  ├─ git checkout smolvla-model-update
  ├─ git pull
  └─ 测试 ✓
          ↓
本地合并
  ├─ git checkout main
  ├─ git merge mujoco-sim-cleanup
  ├─ git merge smolvla-model-update
  └─ git push
          ↓
服务器同步
  ├─ git checkout main
  ├─ git pull
  └─ 最终测试 ✓
```

## 每日工作流

### 早上开始工作
```bash
# 1. 拉取最新代码
git checkout your-branch
git pull origin your-branch

# 2. 开始编码...
```

### 晚上结束工作
```bash
# 1. 查看修改
git status

# 2. 提交修改
git add .
git commit -m "描述今天的工作"

# 3. 推送到远程
git push origin your-branch
```

### 需要测试时
```bash
# 本地推送
git push origin your-branch

# SSH 到服务器
ssh your-server
cd lerobot
git checkout your-branch
git pull origin your-branch

# 运行测试
python your_test_script.py
```

## 提交信息规范

```bash
# 修复 bug
git commit -m "Fix: 修复 MuJoCo XML 加载路径问题"

# 新功能
git commit -m "Feature: 添加多相机支持"

# 重构
git commit -m "Refactor: 简化环境初始化代码"

# 文档
git commit -m "Docs: 更新 README"

# 测试
git commit -m "Test: 添加仿真环境单元测试"

# 临时保存
git commit -m "WIP: 正在开发新功能"
```

## 紧急情况

### 撤销最后一次提交（未推送）
```bash
git reset --soft HEAD~1
# 修改会保留，可以重新提交
```

### 放弃所有本地修改
```bash
git checkout .
git clean -fd
```

### 回到某个历史版本
```bash
git log --oneline
git checkout commit-hash
```

### 删除错误的分支
```bash
# 删除本地分支
git branch -D branch-name

# 删除远程分支
git push origin --delete branch-name
```
