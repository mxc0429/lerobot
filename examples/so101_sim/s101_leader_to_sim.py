"""
使用真实SO101 Leader臂控制PyBullet仿真环境中的Follower臂
同时显示三个相机视角
"""
import time
import numpy as np
import pybullet as p
import os
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("警告: 未安装OpenCV，将使用PyBullet内置相机显示")

from s101_multi_camera_sim import S101MultiCameraEnv
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig


class LeaderToSimController:
    """使用真实Leader臂控制仿真Follower臂"""
    
    def __init__(self, leader_port="/dev/ttyACM0", leader_id="main", use_opencv=True, calibrate=True, 
                 cube_size=0.025, cube_distance=0.20):
        """
        初始化控制器
        
        Args:
            leader_port: Leader臂的串口
            leader_id: Leader臂ID，用于区分不同的机械臂
            use_opencv: 是否使用OpenCV显示相机
            calibrate: 是否进行校准（如果需要）
        """
        print("初始化Leader臂...")
        
        # 设置校准文件目录为项目下的Calibration文件夹
        current_dir = Path(__file__).parent
        calibration_dir = current_dir / "Calibration"
        calibration_dir.mkdir(exist_ok=True)
        
        print(f"校准文件目录: {calibration_dir}")
        
        # 创建Leader臂配置
        leader_config = SO101LeaderConfig(
            port=leader_port,
            id=leader_id,
            calibration_dir=calibration_dir,
            use_degrees=False  # 使用归一化范围 [-100, 100]
        )
        
        # 创建Leader臂实例
        self.leader = SO101Leader(leader_config)
        
        calibration_file = calibration_dir / f"{leader_id}.json"
        if calibrate:
            print("连接Leader臂（启用校准）...")
            if calibration_file.exists():
                print(f"找到现有校准文件: {calibration_file}")
        else:
            print("连接Leader臂（跳过校准）...")
            if not calibration_file.exists():
                print(f"警告: 未找到校准文件 {calibration_file}")
                print("建议先运行一次校准")
        
        self.leader.connect(calibrate=calibrate)
        
        print("\n初始化仿真环境...")
        # 创建仿真环境
        self.sim_env = S101MultiCameraEnv(
            gui=True, 
            use_opencv=use_opencv and HAS_CV2,
            cube_size=cube_size,
            cube_distance=cube_distance
        )
        
        # 设置主相机视角（机械臂在桌子下方）
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=150,  # 调整yaw角度以适应新位置
            cameraPitch=-35,
            cameraTargetPosition=[0, -0.2, 0.8]  # 看向机械臂和工作区域
        )
        
        # 关节名称映射
        self.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper"
        ]
        
        # 关节限制（弧度）
        self.joint_limits = [
            (-1.92, 1.92),
            (-1.75, 1.75),
            (-1.69, 1.69),
            (-1.66, 1.66),
            (-2.74, 2.84),
            (-0.17, 1.75)
        ]
        
        print("\n初始化完成！")
        print("Leader臂已连接，仿真环境已启动")
        print("移动Leader臂，仿真中的Follower臂会跟随移动")
        print("按 Ctrl+C 退出\n")
    
    def normalize_to_radians(self, normalized_value, joint_idx):
        """
        将归一化值转换为弧度
        
        Args:
            normalized_value: 归一化值
                - 前5个关节: [-100, 100]
                - gripper关节: [0, 100]
            joint_idx: 关节索引
            
        Returns:
            弧度值
        """
        min_rad, max_rad = self.joint_limits[joint_idx]
        
        # gripper关节使用不同的归一化范围 [0, 100]
        if joint_idx == 5:  # gripper是第6个关节（索引5）
            # 归一化值从 [0, 100] 映射到 [min_rad, max_rad]
            normalized = normalized_value / 100.0  # 转换到 [0, 1]
        else:
            # 其他关节使用 [-100, 100]
            normalized = (normalized_value + 100) / 200.0  # 转换到 [0, 1]
        
        return min_rad + normalized * (max_rad - min_rad)
    
    def get_leader_positions(self, debug=False):
        """从Leader臂读取关节位置"""
        action = self.leader.get_action()
        
        # 提取关节位置并转换为弧度
        positions = []
        for i, joint_name in enumerate(self.joint_names):
            key = f"{joint_name}.pos"
            if key in action:
                normalized_value = action[key]
                radians = self.normalize_to_radians(normalized_value, i)
                positions.append(radians)
                
                # 调试信息
                if debug and joint_name == "gripper":
                    print(f"  {joint_name}: 归一化值={normalized_value:.1f}, 弧度={radians:.3f}")
            else:
                positions.append(0.0)
        
        return positions
    
    def run(self, debug=False):
        """主循环：读取Leader位置并控制仿真Follower"""
        try:
            step = 0
            last_print_time = time.time()
            
            while True:
                # 从Leader臂读取位置
                leader_positions = self.get_leader_positions(debug=debug and step % 60 == 0)
                
                # 发送到仿真环境
                self.sim_env.set_joint_positions(leader_positions)
                self.sim_env.step()
                
                # 每5步渲染一次相机
                if step % 5 == 0:
                    if self.sim_env.use_opencv:
                        key = self.sim_env.render_cameras_opencv()
                        if key == ord('q'):
                            print("\n用户按下 'q' 键退出")
                            break
                    else:
                        self.sim_env.render_cameras()
                
                # 每秒打印一次位置信息
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    print(f"时间 {step/240:.1f}s, 关节位置(弧度): ", end="")
                    for name, pos in zip(self.joint_names, leader_positions):
                        print(f"{name}={pos:.3f} ", end="")
                    print()
                    last_print_time = current_time
                
                time.sleep(1./240.)
                step += 1
        
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        
        if HAS_CV2:
            cv2.destroyAllWindows()
        
        print("断开Leader臂连接...")
        self.leader.disconnect()
        
        print("关闭仿真环境...")
        self.sim_env.close()
        
        print("清理完成")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="使用Leader臂控制仿真Follower臂")
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Leader臂的串口 (默认: /dev/ttyACM0)"
    )
    parser.add_argument(
        "--no-opencv",
        action="store_true",
        help="禁用OpenCV相机显示"
    )
    parser.add_argument(
        "--id",
        type=str,
        default="main",
        help="Leader臂ID，用于区分不同的机械臂 (默认: main)"
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="跳过校准（需要已有校准文件）"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式，显示gripper关节的详细映射信息"
    )
    parser.add_argument(
        "--cube-size",
        type=float,
        default=0.025,
        help="立方体大小（米），默认0.025（2.5cm）"
    )
    parser.add_argument(
        "--cube-distance",
        type=float,
        default=0.20,
        help="立方体距离机械臂的距离（米），默认0.20（20cm）"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SO101 Leader臂控制仿真Follower臂")
    print("=" * 60)
    print(f"Leader臂端口: {args.port}")
    print(f"Leader臂ID: {args.id}")
    print(f"OpenCV显示: {'禁用' if args.no_opencv else '启用'}")
    print(f"校准模式: {'跳过' if args.no_calibrate else '启用'}")
    
    # 显示校准文件位置
    calibration_dir = Path(__file__).parent / "Calibration"
    calibration_file = calibration_dir / f"{args.id}.json"
    print(f"校准文件: {calibration_file}")
    print("=" * 60)
    
    if args.no_calibrate:
        print("\n注意: 跳过校准模式需要已有校准文件")
        if not calibration_file.exists():
            print(f"警告: 未找到校准文件 {calibration_file}")
            print("建议先运行一次校准（不使用 --no-calibrate 参数）")
        print()
    
    # 创建控制器
    controller = LeaderToSimController(
        leader_port=args.port,
        leader_id=args.id,
        use_opencv=not args.no_opencv,
        calibrate=not args.no_calibrate,
        cube_size=args.cube_size,
        cube_distance=args.cube_distance
    )
    
    # 运行主循环
    controller.run(debug=args.debug)


if __name__ == "__main__":
    main()
