"""
Leader-Follower校准工具

用于校准真实Leader机械臂和仿真Follower机械臂之间的角度映射关系。
校准后的数据会保存到文件，后续使用时可以加载。

使用方法:
1. 运行此脚本进行校准
2. 按照提示移动机械臂到各个位置
3. 校准完成后会保存到 calibrations/ 目录
4. 使用 --calibration_file 参数加载校准文件
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

from lerobot.envs.so101_mujoco import So101MujocoEnv, So101MujocoConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.utils import init_logging, log_say


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_STACKING_XML = REPO_ROOT / "Sim_assets" / "SO-ARM100" / "Simulation" / "SO101" / "so101_block_stacking.xml"
CALIBRATION_DIR = Path("calibrations")


def parse_args():
    parser = argparse.ArgumentParser("Leader-Follower校准工具")
    
    # Leader arm configuration
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Leader arm serial port")
    parser.add_argument("--leader_id", type=str, default=None, help="Leader arm ID")
    parser.add_argument("--no_degrees", dest="use_degrees", action="store_false", help="Use normalized values")
    parser.set_defaults(use_degrees=True)
    
    # Calibration configuration
    parser.add_argument("--calibration_name", type=str, default=None, help="Calibration file name")
    parser.add_argument("--num_points", type=int, default=5, help="Number of calibration points per joint")
    
    return parser.parse_args()


class LeaderFollowerCalibrator:
    """Leader-Follower校准器"""
    
    def __init__(self, args):
        self.args = args
        
        # 初始化日志
        init_logging()
        
        # 创建校准目录
        CALIBRATION_DIR.mkdir(exist_ok=True)
        
        # 创建仿真环境
        log_say("初始化仿真环境...")
        self.env = So101MujocoEnv(
            So101MujocoConfig(
                xml_path=str(BLOCK_STACKING_XML),
                render_mode="window",  # 显示仿真窗口
                use_camera=False,
            )
        )
        
        # 重置环境
        self.obs, _ = self.env.reset()
        
        # 创建Leader机械臂连接
        log_say("连接Leader机械臂...")
        self.leader = SO101Leader(
            SO101LeaderConfig(
                port=args.port,
                use_degrees=args.use_degrees,
                id=args.leader_id,
            )
        )
        
        # 连接Leader
        self.leader.connect(calibrate=True)
        if not self.leader.is_calibrated:
            raise RuntimeError("Leader机械臂未校准")
        
        log_say("Leader机械臂连接成功")
        
        # 校准数据
        self.calibration_data = {
            "timestamp": datetime.now().isoformat(),
            "leader_id": args.leader_id or "unknown",
            "use_degrees": args.use_degrees,
            "joints": {},
        }
        
        self._print_info()
    
    def _print_info(self):
        """打印系统信息"""
        print("\n" + "=" * 70)
        print("Leader-Follower校准工具")
        print("=" * 70)
        print(f"环境信息:")
        print(f"  - 执行器数: {self.env.nu}")
        print(f"  - 执行器名称: {self.env.actuator_names}")
        print(f"\nLeader配置:")
        print(f"  - 端口: {self.args.port}")
        print(f"  - 使用角度制: {self.args.use_degrees}")
        print(f"\n校准配置:")
        print(f"  - 每个关节采样点数: {self.args.num_points}")
        print(f"  - 校准文件保存目录: {CALIBRATION_DIR}")
        print("=" * 70 + "\n")
    
    def calibrate_joint(self, joint_name: str) -> Dict:
        """校准单个关节"""
        log_say(f"\n开始校准关节: {joint_name}")
        log_say(f"请移动Leader机械臂的 {joint_name} 到不同位置")
        log_say(f"需要采集 {self.args.num_points} 个数据点")
        
        leader_positions = []
        follower_positions = []
        
        for i in range(self.args.num_points):
            input(f"\n按Enter键采集第 {i+1}/{self.args.num_points} 个数据点...")
            
            # 读取Leader位置
            leader_obs = self.leader.get_action()
            leader_key = f"{joint_name}.pos"
            
            if leader_key not in leader_obs:
                log_say(f"警告: Leader中没有找到 {leader_key}")
                continue
            
            leader_pos = float(leader_obs[leader_key])
            if self.args.use_degrees:
                leader_pos = np.deg2rad(leader_pos)
            
            # 读取Follower位置（从仿真环境）
            # 找到对应的关节索引
            try:
                joint_idx = self.env.actuator_names.index(joint_name)
                follower_pos = float(self.env.data.qpos[joint_idx])
            except (ValueError, IndexError):
                log_say(f"警告: Follower中没有找到 {joint_name}")
                continue
            
            leader_positions.append(leader_pos)
            follower_positions.append(follower_pos)
            
            log_say(f"  Leader: {leader_pos:.4f} rad, Follower: {follower_pos:.4f} rad")
        
        if len(leader_positions) < 2:
            log_say(f"警告: {joint_name} 数据点不足，跳过")
            return None
        
        # 计算线性映射 (y = ax + b)
        # follower = a * leader + b
        leader_arr = np.array(leader_positions)
        follower_arr = np.array(follower_positions)
        
        # 使用最小二乘法拟合
        A = np.vstack([leader_arr, np.ones(len(leader_arr))]).T
        a, b = np.linalg.lstsq(A, follower_arr, rcond=None)[0]
        
        # 计算误差
        predicted = a * leader_arr + b
        error = np.abs(follower_arr - predicted)
        mean_error = np.mean(error)
        max_error = np.max(error)
        
        log_say(f"\n{joint_name} 校准结果:")
        log_say(f"  映射关系: follower = {a:.6f} * leader + {b:.6f}")
        log_say(f"  平均误差: {mean_error:.6f} rad ({np.rad2deg(mean_error):.2f}°)")
        log_say(f"  最大误差: {max_error:.6f} rad ({np.rad2deg(max_error):.2f}°)")
        
        return {
            "scale": float(a),
            "offset": float(b),
            "mean_error_rad": float(mean_error),
            "max_error_rad": float(max_error),
            "num_samples": len(leader_positions),
            "leader_positions": [float(x) for x in leader_positions],
            "follower_positions": [float(x) for x in follower_positions],
        }
    
    def run_calibration(self):
        """运行完整校准流程"""
        log_say("\n开始校准流程...")
        log_say("=" * 70)
        
        # 对每个执行器进行校准
        for joint_name in self.env.actuator_names:
            result = self.calibrate_joint(joint_name)
            if result:
                self.calibration_data["joints"][joint_name] = result
        
        # 保存校准数据
        self.save_calibration()
        
        log_say("\n校准完成！")
        log_say("=" * 70)
    
    def save_calibration(self):
        """保存校准数据"""
        # 生成文件名
        if self.args.calibration_name:
            filename = f"{self.args.calibration_name}.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"calibration_{timestamp}.json"
        
        filepath = CALIBRATION_DIR / filename
        
        # 保存JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.calibration_data, f, indent=2, ensure_ascii=False)
        
        log_say(f"\n校准数据已保存到: {filepath}")
        
        # 打印摘要
        print("\n" + "=" * 70)
        print("校准摘要")
        print("=" * 70)
        for joint_name, data in self.calibration_data["joints"].items():
            print(f"{joint_name}:")
            print(f"  映射: follower = {data['scale']:.6f} * leader + {data['offset']:.6f}")
            print(f"  误差: {np.rad2deg(data['mean_error_rad']):.2f}° (平均)")
        print("=" * 70)
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'leader'):
            self.leader.disconnect()
        if hasattr(self, 'env'):
            self.env.close()


def main():
    args = parse_args()
    
    calibrator = None
    try:
        calibrator = LeaderFollowerCalibrator(args)
        calibrator.run_calibration()
    except KeyboardInterrupt:
        log_say("\n用户中断校准")
    except Exception as e:
        log_say(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if calibrator:
            calibrator.cleanup()
    
    return 0


if __name__ == "__main__":
    exit(main())
