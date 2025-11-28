"""
S101机械臂抓取演示
演示如何控制机械臂抓取和放置立方体
"""
import pybullet as p
import time
import numpy as np
from s101_pybullet_sim import S101SimEnv


def smooth_move(env, target_positions, steps=100):
    """平滑移动到目标位置"""
    current_positions = env.get_joint_positions()
    
    for i in range(steps):
        # 线性插值
        alpha = (i + 1) / steps
        interpolated = [
            curr + alpha * (target - curr)
            for curr, target in zip(current_positions, target_positions)
        ]
        
        env.set_joint_positions(interpolated)
        env.step()
        time.sleep(1./240.)


def demo_pick_and_place():
    """演示抓取和放置"""
    # 创建环境
    env = S101SimEnv(gui=True)
    
    # 设置相机视角
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=45,
        cameraPitch=-25,
        cameraTargetPosition=[0.3, 0, 0.8]
    )
    
    print("\n=== S101机械臂抓取演示 ===\n")
    
    # 定义关键姿态
    # [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    
    # 初始姿态
    home_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # 准备姿态 - 移动到立方体上方
    ready_pose = [0.3, 0.5, -0.8, 0.5, 0.0, 0.0]
    
    # 接近姿态 - 靠近立方体
    approach_pose = [0.3, 0.8, -1.2, 0.6, 0.0, 0.0]
    
    # 抓取姿态 - 闭合夹爪
    grasp_pose = [0.3, 0.8, -1.2, 0.6, 0.0, 1.0]
    
    # 抬起姿态
    lift_pose = [0.3, 0.5, -0.8, 0.5, 0.0, 1.0]
    
    # 放置准备姿态
    place_ready_pose = [-0.3, 0.5, -0.8, 0.5, 0.0, 1.0]
    
    # 放置姿态
    place_pose = [-0.3, 0.8, -1.2, 0.6, 0.0, 1.0]
    
    # 释放姿态
    release_pose = [-0.3, 0.8, -1.2, 0.6, 0.0, 0.0]
    
    try:
        # 1. 移动到初始位置
        print("1. 移动到初始位置...")
        smooth_move(env, home_pose, steps=100)
        time.sleep(1)
        
        # 2. 移动到准备位置
        print("2. 移动到准备位置...")
        smooth_move(env, ready_pose, steps=150)
        time.sleep(1)
        
        # 3. 接近立方体
        print("3. 接近立方体...")
        smooth_move(env, approach_pose, steps=150)
        time.sleep(1)
        
        # 4. 抓取立方体
        print("4. 抓取立方体...")
        smooth_move(env, grasp_pose, steps=50)
        time.sleep(1)
        
        # 5. 抬起立方体
        print("5. 抬起立方体...")
        smooth_move(env, lift_pose, steps=150)
        time.sleep(1)
        
        # 6. 移动到放置准备位置
        print("6. 移动到放置位置...")
        smooth_move(env, place_ready_pose, steps=200)
        time.sleep(1)
        
        # 7. 下降到放置位置
        print("7. 下降...")
        smooth_move(env, place_pose, steps=150)
        time.sleep(1)
        
        # 8. 释放立方体
        print("8. 释放立方体...")
        smooth_move(env, release_pose, steps=50)
        time.sleep(1)
        
        # 9. 返回初始位置
        print("9. 返回初始位置...")
        smooth_move(env, lift_pose, steps=100)
        smooth_move(env, home_pose, steps=150)
        time.sleep(1)
        
        print("\n演示完成！")
        print("按Ctrl+C退出...")
        
        # 保持仿真运行
        while True:
            env.step()
            time.sleep(1./240.)
    
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    
    finally:
        env.close()


def demo_simple_motion():
    """简单运动演示"""
    env = S101SimEnv(gui=True)
    
    # 设置相机
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.2, 0, 0.8]
    )
    
    print("\n=== 简单运动演示 ===\n")
    print("机械臂将执行各关节的独立运动")
    
    try:
        # 测试每个关节
        joint_names = [
            "shoulder_pan",
            "shoulder_lift", 
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper"
        ]
        
        for idx, name in enumerate(joint_names):
            print(f"\n测试关节 {idx}: {name}")
            
            # 创建目标位置
            target = [0.0] * 6
            target[idx] = 0.5  # 移动到0.5弧度
            
            print(f"  移动到 {target[idx]} 弧度...")
            smooth_move(env, target, steps=100)
            time.sleep(0.5)
            
            # 返回零位
            print(f"  返回零位...")
            smooth_move(env, [0.0] * 6, steps=100)
            time.sleep(0.5)
        
        print("\n所有关节测试完成！")
        print("按Ctrl+C退出...")
        
        while True:
            env.step()
            time.sleep(1./240.)
    
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    
    finally:
        env.close()


if __name__ == "__main__":
    import sys
    
    print("选择演示模式:")
    print("1. 抓取和放置演示")
    print("2. 简单运动演示")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        demo_pick_and_place()
    elif choice == "2":
        demo_simple_motion()
    else:
        print("无效选择，运行默认演示...")
        demo_simple_motion()
