"""
纯仿真环境 - 随机动作测试
用于测试仿真环境的基本功能，不需要训练好的模型

功能:
- 纯MuJoCo仿真，无需真实硬件
- 随机动作探索
- 多相机视角可视化
- 环境稳定性测试
"""
import argparse
import time
from pathlib import Path

import numpy as np

from lerobot.envs.so101_mujoco import So101MujocoEnv, So101MujocoConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_STACKING_XML = REPO_ROOT / "Sim_assets" / "SO-ARM100" / "Simulation" / "SO101" / "so101_block_stacking.xml"


def parse_args():
    parser = argparse.ArgumentParser("纯仿真环境随机动作测试")
    parser.add_argument("--num_episodes", type=int, default=5, help="测试回合数")
    parser.add_argument("--max_steps", type=int, default=300, help="每回合最大步数")
    parser.add_argument("--render", action="store_true", help="显示MuJoCo窗口")
    parser.add_argument("--randomize_blocks", action="store_true", help="随机化方块初始位置")
    parser.add_argument("--action_scale", type=float, default=0.1, help="动作缩放因子(0-1)")
    parser.add_argument("--camera_width", type=int, default=256)
    parser.add_argument("--camera_height", type=int, default=256)
    parser.add_argument("--fps", type=int, default=50, help="仿真帧率")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("纯仿真环境 - 随机动作测试")
    print("=" * 70)
    print(f"配置:")
    print(f"  - 回合数: {args.num_episodes}")
    print(f"  - 每回合步数: {args.max_steps}")
    print(f"  - 渲染: {args.render}")
    print(f"  - 随机化方块: {args.randomize_blocks}")
    print(f"  - 动作缩放: {args.action_scale}")
    print(f"  - 相机分辨率: {args.camera_width}x{args.camera_height}")
    print("=" * 70)
    
    # 创建仿真环境
    camera_names = ["top_cam", "wrist_cam", "right_cam"]
    env = So101MujocoEnv(
        So101MujocoConfig(
            xml_path=str(BLOCK_STACKING_XML),
            render_mode="window" if args.render else None,
            use_camera=True,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_names=camera_names,
            randomize_blocks=args.randomize_blocks,
            num_blocks=3,
        )
    )
    
    print(f"\n环境信息:")
    print(f"  - 关节数 (nq): {env.nq}")
    print(f"  - 执行器数 (nu): {env.nu}")
    print(f"  - 执行器名称: {env.actuator_names}")
    print(f"  - 动作空间: {env.action_space}")
    print(f"  - 观察空间: {env.observation_space}")
    
    step_time = 1.0 / args.fps
    success_count = 0
    
    try:
        for episode in range(args.num_episodes):
            print("\n" + "=" * 70)
            print(f"回合 {episode + 1}/{args.num_episodes}")
            print("=" * 70)
            
            # 重置环境
            obs, info = env.reset()
            
            # 获取初始方块位置
            initial_positions = env.get_block_positions()
            print("\n初始方块位置:")
            for name, pos in initial_positions.items():
                print(f"  {name}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
            
            print(f"\n开始执行 {args.max_steps} 步随机动作...")
            
            # 运行回合
            for step in range(args.max_steps):
                start_time = time.perf_counter()
                
                # 生成随机动作
                action = env.action_space.sample() * args.action_scale
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 打印状态
                if (step + 1) % 50 == 0:
                    positions = env.get_block_positions()
                    print(f"  步数 {step + 1}/{args.max_steps}")
                    for name, pos in positions.items():
                        print(f"    {name}: z={pos[2]:.3f}")
                
                if terminated or truncated:
                    print(f"  回合在第 {step + 1} 步结束")
                    break
                
                # 控制帧率
                elapsed = time.perf_counter() - start_time
                if elapsed < step_time:
                    time.sleep(step_time - elapsed)
            
            # 检查最终状态
            final_positions = env.get_block_positions()
            print("\n最终方块位置:")
            for name, pos in final_positions.items():
                print(f"  {name}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
            
            is_stacked = env.check_stacking_success()
            if is_stacked:
                success_count += 1
                print("✅ 方块成功堆叠 (随机动作偶然成功!)")
            else:
                print("❌ 方块未堆叠 (这是正常的，因为是随机动作)")
    
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
    
    finally:
        env.close()
        print("\n" + "=" * 70)
        print("测试完成")
        print("=" * 70)
        print(f"完成回合数: {episode + 1}")
        print(f"偶然成功次数: {success_count}")
        print("=" * 70)


if __name__ == "__main__":
    main()
