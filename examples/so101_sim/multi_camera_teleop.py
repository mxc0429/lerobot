"""
多相机遥操作系统 - 使用真实SO101机械臂控制仿真环境，实时显示三个相机画面

功能:
- 使用真实SO101 Leader机械臂作为输入设备
- 控制MuJoCo仿真环境中的Follower机械臂
- 实时显示三个相机视角 (top_cam, wrist_cam, right_cam)
- 支持键盘控制 (q:退出, p:暂停, r:重置, s:截图, h:帮助)
- 显示FPS、时间戳、连接状态等信息
"""
import argparse
import time
from pathlib import Path
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from lerobot.display.multi_camera_display import MultiCameraDisplay
from lerobot.envs.so101_mujoco import So101MujocoEnv, So101MujocoConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.utils import init_logging, log_say


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_STACKING_XML = REPO_ROOT / "Sim_assets" / "SO-ARM100" / "Simulation" / "SO101" / "so101_block_stacking.xml"


def parse_args():
    parser = argparse.ArgumentParser("多相机遥操作 - 真实机械臂控制仿真环境")
    
    # Leader arm configuration
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Leader arm serial port")
    parser.add_argument("--leader_id", type=str, default=None, help="Leader arm calibration ID")
    parser.add_argument("--leader_calibration_dir", type=str, default=None, help="Leader calibration directory")
    parser.add_argument("--skip_calibration", action="store_true", help="Skip leader auto calibration")
    parser.add_argument("--no_degrees", dest="use_degrees", action="store_false", help="Use normalized values")
    parser.set_defaults(use_degrees=True)
    
    # Simulation configuration
    parser.add_argument("--xml_path", type=str, default=None, help="Custom MuJoCo XML path")
    parser.add_argument("--randomize_blocks", action="store_true", help="Randomize block positions")
    
    # Display configuration
    parser.add_argument("--camera_width", type=int, default=320, help="Camera view width")
    parser.add_argument("--camera_height", type=int, default=240, help="Camera view height")
    parser.add_argument("--display_fps", type=int, default=10, help="Display update rate (Hz)")
    parser.add_argument("--no_overlays", dest="show_overlays", action="store_false", help="Hide overlays")
    parser.set_defaults(show_overlays=True)
    
    # Control configuration
    parser.add_argument("--control_fps", type=int, default=50, help="Control loop rate (Hz)")
    parser.add_argument("--max_duration", type=int, default=300, help="Max session duration (seconds)")
    
    return parser.parse_args()


def map_leader_to_action(
    leader_obs: Dict[str, float],
    env: So101MujocoEnv,
    use_degrees: bool,
) -> NDArray[np.float32]:
    """
    将Leader机械臂的关节读数转换为仿真环境的归一化动作
    
    Args:
        leader_obs: Leader机械臂的观测数据 (关节位置)
        env: 仿真环境
        use_degrees: Leader数据是否为角度制
        
    Returns:
        归一化的动作数组
    """
    joint_targets: Dict[str, float] = {}
    for name in env.actuator_names:
        key = f"{name}.pos"
        if key not in leader_obs:
            continue
        value = float(leader_obs[key])
        if use_degrees:
            value = np.deg2rad(value)
        joint_targets[name] = value

    return env.actuator_target_from_joint_dict(joint_targets)


class MultiCameraTeleopApp:
    """多相机遥操作应用主类"""
    
    def __init__(self, args):
        self.args = args
        self.running = False
        self.paused = False
        
        # 初始化日志
        init_logging()
        
        # 创建仿真环境
        xml_path = args.xml_path if args.xml_path else str(BLOCK_STACKING_XML)
        camera_names = ["top_cam", "wrist_cam", "right_cam"]
        
        log_say("初始化仿真环境...")
        self.env = So101MujocoEnv(
            So101MujocoConfig(
                xml_path=xml_path,
                render_mode=None,  # 不使用MuJoCo自带窗口
                use_camera=True,
                camera_width=args.camera_width,
                camera_height=args.camera_height,
                camera_names=camera_names,
                randomize_blocks=args.randomize_blocks,
                num_blocks=3,
            )
        )
        
        # 重置环境
        self.obs, _ = self.env.reset()
        
        # 创建多相机显示
        log_say("初始化多相机显示...")
        self.display = MultiCameraDisplay(
            model=self.env.model,
            data=self.env.data,
            camera_names=camera_names,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            target_fps=args.display_fps,
            window_name="SO101 Multi-Camera Teleoperation",
            show_overlays=args.show_overlays,
        )
        
        # 创建Leader机械臂连接
        log_say("连接Leader机械臂...")
        self.leader = SO101Leader(
            SO101LeaderConfig(
                port=args.port,
                use_degrees=args.use_degrees,
                id=args.leader_id,
                calibration_dir=Path(args.leader_calibration_dir).expanduser() 
                    if args.leader_calibration_dir else None,
            )
        )
        
        # 连接并校准
        try:
            self.leader.connect(calibrate=not args.skip_calibration)
            if not self.leader.is_calibrated:
                raise RuntimeError(
                    "Leader机械臂未校准。请运行时不使用 --skip_calibration 参数，"
                    "或提供 --leader_id/--leader_calibration_dir 指向现有校准文件。"
                )
            self.display.set_status("connected", True)
            log_say("Leader机械臂连接成功")
        except Exception as e:
            log_say(f"Leader机械臂连接失败: {e}")
            self.display.set_status("connected", False)
            self.display.set_status("error", f"Leader连接失败: {e}")
            raise
        
        # 打印系统信息
        self._print_system_info()
    
    def _print_system_info(self):
        """打印系统配置信息"""
        print("\n" + "=" * 70)
        print("多相机遥操作系统")
        print("=" * 70)
        print(f"环境信息:")
        print(f"  - 关节数: {self.env.nq}")
        print(f"  - 执行器数: {self.env.nu}")
        print(f"  - 执行器名称: {self.env.actuator_names}")
        print(f"\n显示配置:")
        print(f"  - 相机数量: {len(self.display.camera_names)}")
        print(f"  - 相机名称: {self.display.camera_names}")
        print(f"  - 分辨率: {self.args.camera_width}x{self.args.camera_height}")
        print(f"  - 显示帧率: {self.args.display_fps} Hz")
        print(f"\n控制配置:")
        print(f"  - 控制帧率: {self.args.control_fps} Hz")
        print(f"  - Leader端口: {self.args.port}")
        print(f"  - 使用角度制: {self.args.use_degrees}")
        print(f"\n键盘控制:")
        print(f"  - q: 退出")
        print(f"  - p: 暂停/继续")
        print(f"  - r: 重置环境")
        print(f"  - s: 保存截图")
        print(f"  - h: 显示帮助")
        print("=" * 70 + "\n")
    
    def _handle_keyboard_events(self):
        """处理键盘事件"""
        events = self.display.get_keyboard_events()
        
        if events.get('quit'):
            log_say("用户请求退出")
            self.running = False
        
        if events.get('pause'):
            self.paused = not self.paused
            self.display.set_status("paused", self.paused)
            log_say(f"{'暂停' if self.paused else '继续'}")
        
        if events.get('reset'):
            log_say("重置环境")
            self.obs, _ = self.env.reset()
            self.display.set_status("error", None)
        
        if events.get('snapshot'):
            self._save_snapshot()
        
        if events.get('help'):
            log_say("键盘控制: q=退出, p=暂停, r=重置, s=截图, h=帮助")
    
    def _save_snapshot(self):
        """保存当前相机画面截图"""
        import cv2
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = Path("snapshots")
        snapshot_dir.mkdir(exist_ok=True)
        
        # 渲染当前画面
        composed_image = self.display._render_cameras()
        if self.display.show_overlays:
            composed_image = self.display._add_overlays(composed_image)
        
        # 保存
        filename = snapshot_dir / f"snapshot_{timestamp}.png"
        cv2.imwrite(str(filename), composed_image)
        log_say(f"截图已保存: {filename}")
    
    def run(self):
        """运行遥操作主循环"""
        log_say("启动遥操作系统...")
        
        # 启动显示线程
        self.display.start()
        
        self.running = True
        start_time = time.perf_counter()
        control_dt = 1.0 / self.args.control_fps
        step_count = 0
        
        try:
            while self.running:
                loop_start = time.perf_counter()
                
                # 检查显示窗口是否关闭
                if self.display.is_window_closed():
                    log_say("显示窗口已关闭")
                    break
                
                # 处理键盘事件
                self._handle_keyboard_events()
                
                # 检查最大运行时间
                elapsed = time.perf_counter() - start_time
                if elapsed > self.args.max_duration:
                    log_say(f"达到最大运行时间 ({self.args.max_duration}秒)")
                    break
                
                # 如果暂停，跳过控制循环
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # 读取Leader机械臂状态
                try:
                    leader_obs = self.leader.get_action()
                    leader_obs = {k: float(v) for k, v in leader_obs.items()}
                    
                    # 转换为仿真动作
                    action = map_leader_to_action(leader_obs, self.env, self.args.use_degrees)
                    
                    # 执行动作
                    self.obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    # 更新连接状态
                    if not self.display._status.get("connected", False):
                        self.display.set_status("connected", True)
                        self.display.set_status("error", None)
                    
                    step_count += 1
                    
                    # 定期打印状态
                    if step_count % (self.args.control_fps * 10) == 0:  # 每10秒
                        display_fps = self.display.get_fps()
                        log_say(f"运行中 - 步数: {step_count}, 显示FPS: {display_fps:.1f}")
                    
                except Exception as e:
                    log_say(f"控制循环错误: {e}")
                    self.display.set_status("connected", False)
                    self.display.set_status("error", f"控制错误: {e}")
                    time.sleep(1.0)  # 错误后等待
                    continue
                
                # 控制循环速率
                loop_elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.0, control_dt - loop_elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            log_say("用户中断 (Ctrl+C)")
        
        finally:
            self._shutdown()
    
    def _shutdown(self):
        """安全关闭系统"""
        log_say("关闭系统...")
        
        # 停止显示
        if hasattr(self, 'display'):
            self.display.stop()
        
        # 断开Leader连接
        if hasattr(self, 'leader'):
            try:
                self.leader.disconnect()
                log_say("Leader机械臂已断开")
            except Exception as e:
                log_say(f"断开Leader时出错: {e}")
        
        # 关闭环境
        if hasattr(self, 'env'):
            self.env.close()
        
        log_say("系统已关闭")


def main():
    args = parse_args()
    
    try:
        app = MultiCameraTeleopApp(args)
        app.run()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
