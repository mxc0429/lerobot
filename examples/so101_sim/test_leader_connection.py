"""
测试SO101 Leader臂连接
用于验证Leader臂是否正确连接和工作
"""
import time
from pathlib import Path
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig


def test_leader_connection(port="/dev/ttyACM0", leader_id="main"):
    """测试Leader臂连接"""
    print("=" * 60)
    print("SO101 Leader臂连接测试")
    print("=" * 60)
    print(f"端口: {port}")
    print(f"ID: {leader_id}")
    
    # 设置校准文件目录
    calibration_dir = Path(__file__).parent / "Calibration"
    calibration_dir.mkdir(exist_ok=True)
    calibration_file = calibration_dir / f"{leader_id}.json"
    print(f"校准文件: {calibration_file}")
    print()
    
    try:
        # 创建配置
        print("1. 创建Leader臂配置...")
        config = SO101LeaderConfig(
            port=port,
            id=leader_id,
            calibration_dir=calibration_dir,
            use_degrees=False
        )
        print("   ✓ 配置创建成功")
        
        # 创建Leader实例
        print("\n2. 创建Leader臂实例...")
        leader = SO101Leader(config)
        print("   ✓ 实例创建成功")
        
        # 连接Leader臂
        print("\n3. 连接Leader臂...")
        print("   (如果是首次连接，需要进行校准)")
        leader.connect(calibrate=True)
        print("   ✓ Leader臂连接成功")
        
        # 读取关节位置
        print("\n4. 读取关节位置...")
        print("   移动Leader臂，观察位置变化")
        print("   按 Ctrl+C 停止\n")
        
        joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper"
        ]
        
        try:
            for i in range(50):  # 读取50次，约5秒
                action = leader.get_action()
                
                print(f"   读取 {i+1}/50: ", end="")
                for name in joint_names:
                    key = f"{name}.pos"
                    if key in action:
                        print(f"{name}={action[key]:6.1f} ", end="")
                print()
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n   测试被用户中断")
        
        # 断开连接
        print("\n5. 断开Leader臂连接...")
        leader.disconnect()
        print("   ✓ 断开成功")
        
        print("\n" + "=" * 60)
        print("测试完成！Leader臂工作正常")
        print("=" * 60)
        return True
    
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        print("\n可能的原因:")
        print("  1. Leader臂未连接到指定端口")
        print("  2. 没有串口访问权限")
        print("     解决方法: sudo usermod -a -G dialout $USER")
        print("     然后重新登录")
        print("  3. 端口被其他程序占用")
        print("  4. Leader臂硬件故障")
        print("\n" + "=" * 60)
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试SO101 Leader臂连接")
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Leader臂的串口 (默认: /dev/ttyACM0)"
    )
    parser.add_argument(
        "--id",
        type=str,
        default="main",
        help="Leader臂ID (默认: main)"
    )
    
    args = parser.parse_args()
    
    success = test_leader_connection(args.port, args.id)
    
    if success:
        print("\n下一步:")
        print("  运行: python examples/so101_sim/s101_leader_to_sim.py")
        print("  开始使用Leader臂控制仿真环境")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
