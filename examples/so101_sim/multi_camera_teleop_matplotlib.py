"""
多相机遥操作系统 - MuJoCo渲染 + Matplotlib显示版本

这个版本使用MuJoCo进行相机渲染，使用Matplotlib同时显示三个相机视角，
在Windows上完美运行，无需OpenCV GUI支持。

功能:
- 使用真实SO101 Leader机械臂控制仿真环境
- 同时显示三个相机视角（top_cam, wrist_cam, right_cam）
- 使用MuJoCo渲染，Matplotlib显示
- 支持键盘控制
"""
import argparse
import time
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from numpy.typing import NDArray

from lerobot.envs.so101_mujoco import So101MujocoEnv, So101MujocoConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.utils import init_logging, log_say


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_STACKING_XML = REPO_ROOT / "Sim_assets" / "SO-ARM100" / "Simulation" / "SO101" / "so101_block_stacking.xml"


def parse_args():
    parser = argparse.ArgumentParser("多相机遥操作 - MuJoCo渲染 + Matplotlib显示")
    
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
    
    # Control configuration
    parser.add_argument("--control_fps", type=int, default=50, help="Control loop rate (Hz)")
    parser.add_argument("--max_duration", type=int, default=300, help="Max session duration (seconds)")
    
    # Calibration configuration
    parser.add_argument("--calibration_file", type=str, default=None, help="Path to calibration JSON file")
    
    return parser.parse_args()


def load_calibration(calibration_file: Path) -> Dict:
    """加载校准文件"""
    import json
    with open(calibration_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def map_leader_to_action(
    leader_obs: Dict[str, float],
    env: So101MujocoEnv,
    use_degrees: bool,
    calibration: Dict = None,
) -> NDArray[np.float32]:
    """将Leader机械臂的关节读数转换为仿真环境的归一化动作"""
    joint_targets: Dict[str, float] = {}
    for name in env.actuator_names:
        key = f"{name}.pos"
        if key not in leader_obs:
            continue
        value = float(leader_obs[key])
        if use_degrees:
            value = np.deg2rad(value)
        
        # 应用校准（如果有）
        if calibration and name in calibration.get("joints", {}):
            joint_cal = calibration["joints"][name]
            value = joint_cal["scale"] * value + joint_cal["offset"]
        
        joint_targets[name] = value
    return env.actuator_target_from_joint_dict(joint_targets)


class MultiCameraTeleopMatplotlib:
    """使用MuJoCo渲染 + Matplotlib显示的多相机遥操作应用"""
    
    def __init__(self, args):
        self.args = args
        self.running = False
        self.paused = False
        self.connected = False
        
        # 初始化日志
        init_logging()
        
        # 加载校准文件（如果提供）
        self.calibration = None
        if args.calibration_file:
            try:
                self.calibration = load_calibration(Path(args.calibration_file))
                log_say(f"已加载校准文件: {args.calibration_file}")
                log_say(f"校准时间: {self.calibration.get('timestamp', 'unknown')}")
            except Exception as e:
                log_say(f"警告: 无法加载校准文件: {e}")
        
        # 创建仿真环境
        xml_path = args.xml_path if args.xml_path else str(BLOCK_STACKING_XML)
        camera_names = ["top_cam", "wrist_cam", "right_cam"]
        
        log_say("初始化仿真环境...")
        self.env = So101MujocoEnv(
            So101MujocoConfig(
                xml_path=xml_path,
                render_mode=None,
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
        
        # 创建相机渲染器
        self.camera_names = camera_names
        self.renderers = {}
        for cam_name in camera_names:
            try:
                cam_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                if cam_id >= 0:
                    renderer = mujoco.Renderer(self.env.model, args.camera_width, args.camera_height)
                    self.renderers[cam_name] = renderer
                    log_say(f"创建相机渲染器: {cam_name}")
            except Exception as e:
                log_say(f"警告: 无法创建相机 {cam_name}: {e}")
        
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
            self.connected = True
            log_say("Leader机械臂连接成功")
        except Exception as e:
            log_say(f"Leader机械臂连接失败: {e}")
            raise
        
        # 创建Matplotlib显示
        log_say("初始化显示窗口...")
        self._setup_display()
        
        # FPS跟踪
        self.frame_times = []
        self.current_fps = 0.0
        
        # 打印系统信息
        self._print_system_info()
    
    def _setup_display(self):
        """设置Matplotlib显示"""
        plt.ion()
        
        # 计算合适的figure大小，保持宽高比
        aspect_ratio = self.args.camera_height / self.args.camera_width
        fig_width = 18  # 总宽度
        fig_height = fig_width * aspect_ratio / 3 + 1  # 3个相机并排，加1英寸给标题
        
        self.fig, self.axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
        self.fig.canvas.manager.set_window_title('Multi-Camera Teleoperation - MuJoCo + Matplotlib')
        
        # 初始化图像显示
        self.images = []
        for i, (ax, cam_name) in enumerate(zip(self.axes, self.camera_names)):
            ax.set_title(cam_name, fontsize=14, color='green', weight='bold')
            ax.axis('off')
            img = np.zeros((self.args.camera_height, self.args.camera_width, 3), dtype=np.uint8)
            im = ax.imshow(img, aspect='equal')  # 保持宽高比
            self.images.append(im)
        
        plt.tight_layout(pad=0.5)
        plt.show(block=False)
        
        # 连接事件
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('close_event', self._on_close)
    
    def _on_key_press(self, event):
        """处理键盘事件"""
        if event.key == 'q' or event.key == 'escape':
            log_say("用户请求退出")
            self.running = False
        elif event.key == ' ':
            self.paused = not self.paused
            log_say(f"{'暂停' if self.paused else '继续'}")
        elif event.key == 'r':
            log_say("重置环境")
            self.obs, _ = self.env.reset()
        elif event.key == 'h':
            log_say("键盘控制: q/Esc=退出, 空格=暂停, r=重置, h=帮助")
    
    def _on_close(self, event):
        """处理窗口关闭事件"""
        log_say("显示窗口已关闭")
        self.running = False
    
    def _print_system_info(self):
        """打印系统配置信息"""
        print("\n" + "=" * 70)
        print("多相机遥操作系统 - MuJoCo渲染 + Matplotlib显示")
        print("=" * 70)
        print(f"环境信息:")
        print(f"  - 关节数: {self.env.nq}")
        print(f"  - 执行器数: {self.env.nu}")
        print(f"  - 执行器名称: {self.env.actuator_names}")
        print(f"\n显示配置:")
        print(f"  - 相机数量: {len(self.camera_names)}")
        print(f"  - 相机名称: {self.camera_names}")
        print(f"  - 分辨率: {self.args.camera_width}x{self.args.camera_height}")
        print(f"  - 显示帧率: {self.args.display_fps} Hz")
        print(f"\n控制配置:")
        print(f"  - 控制帧率: {self.args.control_fps} Hz")
        print(f"  - Leader端口: {self.args.port}")
        print(f"  - 使用角度制: {self.args.use_degrees}")
        print(f"\n键盘控制:")
        print(f"  - q/Esc: 退出")
        print(f"  - 空格: 暂停/继续")
        print(f"  - r: 重置环境")
        print(f"  - h: 显示帮助")
        print(f"\n提示:")
        print(f"  - 三个相机视角同时显示")
        print(f"  - 窗口标题显示当前FPS和连接状态")
        print("=" * 70 + "\n")
    
    def _render_cameras(self):
        """渲染所有相机视图"""
        for i, cam_name in enumerate(self.camera_names):
            if cam_name in self.renderers:
                try:
                    renderer = self.renderers[cam_name]
                    cam_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                    renderer.update_scene(self.env.data, camera=cam_id)
                    pixels = renderer.render()
                    self.images[i].set_data(pixels)
                except Exception as e:
                    log_say(f"渲染相机 {cam_name} 时出错: {e}")
        
        # 一次性刷新所有图像
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def _update_fps(self, frame_start: float):
        """更新FPS跟踪"""
        current_time = time.perf_counter()
        self.frame_times.append(current_time)
        
        cutoff_time = current_time - 1.0
        self.frame_times = [t for t in self.frame_times if t > cutoff_time]
        
        if len(self.frame_times) > 1:
            time_span = self.frame_times[-1] - self.frame_times[0]
            if time_span > 0:
                self.current_fps = (len(self.frame_times) - 1) / time_span
                status = "CONNECTED" if self.connected else "DISCONNECTED"
                self.fig.canvas.manager.set_window_title(
                    f'Multi-Camera Teleoperation - FPS: {self.current_fps:.1f} - {status}'
                )
    
    def run(self):
        """运行遥操作主循环"""
        log_say("启动遥操作系统...")
        
        self.running = True
        start_time = time.perf_counter()
        control_dt = 1.0 / self.args.control_fps
        display_dt = 1.0 / self.args.display_fps
        step_count = 0
        last_display_time = 0
        
        try:
            while self.running:
                loop_start = time.perf_counter()
                
                # 检查窗口是否还在
                if not plt.fignum_exists(self.fig.number):
                    log_say("显示窗口已关闭")
                    break
                
                # 检查最大运行时间
                elapsed = time.perf_counter() - start_time
                if elapsed > self.args.max_duration:
                    log_say(f"达到最大运行时间 ({self.args.max_duration}秒)")
                    break
                
                # 如果暂停，跳过控制循环
                if self.paused:
                    plt.pause(0.1)
                    continue
                
                # 读取Leader机械臂状态
                try:
                    leader_obs = self.leader.get_action()
                    leader_obs = {k: float(v) for k, v in leader_obs.items()}
                    
                    # 转换为仿真动作（应用校准）
                    action = map_leader_to_action(leader_obs, self.env, self.args.use_degrees, self.calibration)
                    
                    # 执行动作
                    self.obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    if not self.connected:
                        self.connected = True
                    
                    step_count += 1
                    
                    # 定期打印状态
                    if step_count % (self.args.control_fps * 10) == 0:
                        log_say(f"运行中 - 步数: {step_count}, FPS: {self.current_fps:.1f}")
                    
                except Exception as e:
                    log_say(f"控制循环错误: {e}")
                    self.connected = False
                    time.sleep(1.0)
                    continue
                
                # 更新显示（按显示帧率）
                if (loop_start - last_display_time) >= display_dt:
                    self._render_cameras()
                    self._update_fps(loop_start)
                    last_display_time = loop_start
                
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
        
        # 关闭显示
        plt.close(self.fig)
        
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
        app = MultiCameraTeleopMatplotlib(args)
        app.run()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
