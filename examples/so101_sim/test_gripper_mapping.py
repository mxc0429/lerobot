"""
测试gripper关节的角度映射
用于验证归一化值到弧度的转换是否正确
"""

def test_gripper_mapping():
    """测试gripper关节映射"""
    print("=" * 60)
    print("Gripper关节映射测试")
    print("=" * 60)
    print()
    
    # gripper关节的弧度限制
    min_rad = -0.17
    max_rad = 1.75
    
    print(f"Gripper关节弧度范围: [{min_rad:.3f}, {max_rad:.3f}]")
    print(f"Gripper归一化范围: [0, 100]")
    print()
    
    # 测试不同的归一化值
    test_values = [0, 25, 50, 75, 100]
    
    print("归一化值 -> 弧度转换:")
    print("-" * 60)
    print(f"{'归一化值':<15} {'弧度':<15} {'状态':<20}")
    print("-" * 60)
    
    for norm_val in test_values:
        # 正确的转换（gripper使用0-100范围）
        normalized = norm_val / 100.0
        radians = min_rad + normalized * (max_rad - min_rad)
        
        # 判断状态
        if radians < 0.3:
            state = "闭合"
        elif radians < 1.0:
            state = "半开"
        else:
            state = "张开"
        
        print(f"{norm_val:<15} {radians:<15.3f} {state:<20}")
    
    print("-" * 60)
    print()
    
    # 显示预期行为
    print("预期行为:")
    print("  - 归一化值 0   -> 弧度 -0.170 -> 夹爪闭合")
    print("  - 归一化值 50  -> 弧度  0.790 -> 夹爪半开")
    print("  - 归一化值 100 -> 弧度  1.750 -> 夹爪张开")
    print()
    
    # 错误的映射（如果使用-100到100范围）
    print("错误映射（如果使用-100到100范围）:")
    print("-" * 60)
    print(f"{'归一化值':<15} {'弧度':<15} {'状态':<20}")
    print("-" * 60)
    
    for norm_val in test_values:
        # 错误的转换（假设使用-100到100范围）
        wrong_normalized = (norm_val + 100) / 200.0
        wrong_radians = min_rad + wrong_normalized * (max_rad - min_rad)
        
        if wrong_radians < 0.3:
            state = "闭合"
        elif wrong_radians < 1.0:
            state = "半开"
        else:
            state = "张开"
        
        print(f"{norm_val:<15} {wrong_radians:<15.3f} {state:<20}")
    
    print("-" * 60)
    print()
    
    print("结论:")
    print("  如果真实机械臂闭合（归一化值≈0）时，仿真显示半开，")
    print("  说明使用了错误的映射（-100到100范围）。")
    print("  正确的映射应该使用0到100范围。")
    print()
    print("=" * 60)


def test_other_joints_mapping():
    """测试其他关节的映射"""
    print()
    print("=" * 60)
    print("其他关节映射测试（使用-100到100范围）")
    print("=" * 60)
    print()
    
    joints = [
        ("shoulder_pan", -1.92, 1.92),
        ("shoulder_lift", -1.75, 1.75),
        ("elbow_flex", -1.69, 1.69),
        ("wrist_flex", -1.66, 1.66),
        ("wrist_roll", -2.74, 2.84),
    ]
    
    test_values = [-100, -50, 0, 50, 100]
    
    for joint_name, min_rad, max_rad in joints:
        print(f"\n{joint_name}: [{min_rad:.3f}, {max_rad:.3f}]")
        print("-" * 40)
        
        for norm_val in test_values:
            normalized = (norm_val + 100) / 200.0
            radians = min_rad + normalized * (max_rad - min_rad)
            print(f"  {norm_val:>4} -> {radians:>7.3f}")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    test_gripper_mapping()
    test_other_joints_mapping()
    
    print()
    print("使用方法:")
    print("  1. 运行此脚本查看映射关系")
    print("  2. 使用 --debug 参数运行主程序查看实时映射:")
    print("     python examples/so101_sim/s101_leader_to_sim.py --debug")
    print()
