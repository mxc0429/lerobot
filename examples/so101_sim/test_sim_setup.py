"""
测试纯仿真环境设置
快速验证环境是否正确配置
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# 解析命令行参数
parser = argparse.ArgumentParser(description="测试纯仿真环境设置")
parser.add_argument("--render", action="store_true", help="显示MuJoCo可视化窗口")
args = parser.parse_args()

print("=" * 70)
print("测试纯仿真环境设置")
print("=" * 70)

# 1. 检查MuJoCo
print("\n1. 检查MuJoCo...")
try:
    import mujoco
    print(f"   ✅ MuJoCo 已安装 (版本: {mujoco.__version__})")
except ImportError as e:
    print(f"   ❌ MuJoCo 未安装: {e}")
    print("   请运行: pip install mujoco")
    sys.exit(1)

# 2. 检查Gymnasium
print("\n2. 检查Gymnasium...")
try:
    import gymnasium as gym
    print(f"   ✅ Gymnasium 已安装 (版本: {gym.__version__})")
except ImportError as e:
    print(f"   ❌ Gymnasium 未安装: {e}")
    print("   请运行: pip install gymnasium")
    sys.exit(1)

# 3. 检查PyTorch
print("\n3. 检查PyTorch...")
try:
    import torch
    print(f"   ✅ PyTorch 已安装 (版本: {torch.__version__})")
    print(f"   - CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - CUDA 版本: {torch.version.cuda}")
        print(f"   - GPU 数量: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"   ❌ PyTorch 未安装: {e}")
    print("   请运行: pip install torch")
    sys.exit(1)

# 4. 检查NumPy
print("\n4. 检查NumPy...")
try:
    import numpy as np
    print(f"   ✅ NumPy 已安装 (版本: {np.__version__})")
except ImportError as e:
    print(f"   ❌ NumPy 未安装: {e}")
    sys.exit(1)

# 5. 检查XML文件
print("\n5. 检查仿真资源文件...")
BLOCK_STACKING_XML = REPO_ROOT / "Sim_assets" / "SO-ARM100" / "Simulation" / "SO101" / "so101_block_stacking.xml"
if BLOCK_STACKING_XML.exists():
    print(f"   ✅ 方块堆叠XML文件存在")
    print(f"   路径: {BLOCK_STACKING_XML}")
else:
    print(f"   ❌ 方块堆叠XML文件不存在")
    print(f"   期望路径: {BLOCK_STACKING_XML}")
    print("   请确保Sim_assets目录存在")

DEFAULT_XML = REPO_ROOT / "Sim_assets" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.xml"
if DEFAULT_XML.exists():
    print(f"   ✅ 默认XML文件存在")
    print(f"   路径: {DEFAULT_XML}")
else:
    print(f"   ⚠️  默认XML文件不存在")
    print(f"   期望路径: {DEFAULT_XML}")

# 6. 检查环境类
print("\n6. 检查环境类...")
try:
    from lerobot.envs.so101_mujoco import So101MujocoEnv, So101MujocoConfig
    print(f"   ✅ So101MujocoEnv 可以导入")
except ImportError as e:
    print(f"   ❌ 无法导入 So101MujocoEnv: {e}")
    sys.exit(1)

# 7. 测试创建环境
print("\n7. 测试创建环境...")
if args.render:
    print("   (启用可视化窗口)")
try:
    if BLOCK_STACKING_XML.exists():
        config = So101MujocoConfig(
            xml_path=str(BLOCK_STACKING_XML),
            render_mode="window" if args.render else None,
            use_camera=False,  # 不使用相机
            randomize_blocks=False,
            num_blocks=3,
        )
        env = So101MujocoEnv(config)
        print(f"   ✅ 环境创建成功")
        print(f"   - 关节数 (nq): {env.nq}")
        print(f"   - 执行器数 (nu): {env.nu}")
        print(f"   - 动作空间: {env.action_space}")
        
        # 测试reset
        obs, info = env.reset()
        print(f"   ✅ 环境重置成功")
        print(f"   - 观察空间键: {list(obs.keys())}")
        print(f"   - qpos形状: {obs['qpos'].shape}")
        
        # 测试step
        print(f"   正在执行 20 步测试...")
        import time
        for i in range(20):
            action = env.action_space.sample() * 0.1
            obs, reward, terminated, truncated, info = env.step(action)
            if args.render:
                time.sleep(0.02)  # 控制帧率
        print(f"   ✅ 环境步进成功")
        
        # 测试方块位置
        positions = env.get_block_positions()
        print(f"   ✅ 获取方块位置成功")
        print(f"   - 方块数量: {len(positions)}")
        
        # 关闭环境
        env.close()
        print(f"   ✅ 环境关闭成功")
    else:
        print(f"   ⚠️  跳过环境测试 (XML文件不存在)")
except Exception as e:
    print(f"   ❌ 环境测试失败: {e}")
    import traceback
    traceback.print_exc()

# 8. 检查脚本文件
print("\n8. 检查脚本文件...")
scripts = [
    "sim_random_actions.py",
    "sim_only_evaluation.py",
    "sim_rl_training.py",
]
script_dir = Path(__file__).parent
for script in scripts:
    script_path = script_dir / script
    if script_path.exists():
        print(f"   ✅ {script}")
    else:
        print(f"   ❌ {script} 不存在")

print("\n" + "=" * 70)
print("测试完成!")
print("=" * 70)

# 总结
print("\n如果所有检查都通过，你可以运行:")
print("\n1. 测试随机动作 (带可视化):")
print("   python examples/so101_sim/sim_random_actions.py --num_episodes 2 --max_steps 100 --render")
print("\n2. 测试随机动作 (无可视化，更快):")
print("   python examples/so101_sim/sim_random_actions.py --num_episodes 2 --max_steps 100")
print("\n3. 强化学习训练:")
print("   python examples/so101_sim/sim_rl_training.py --num_episodes 100 --max_steps 200")
print("\n4. 模型评估 (需要先训练模型):")
print("   python examples/so101_sim/sim_only_evaluation.py --policy_path /path/to/model --num_episodes 5")
print("\n提示: 添加 --render 参数可以显示 MuJoCo 可视化窗口")
print("=" * 70)
