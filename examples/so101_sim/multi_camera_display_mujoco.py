"""
多相机显示测试 - MuJoCo渲染 + Matplotlib显示版本

这个版本使用MuJoCo进行相机渲染，使用Matplotlib显示多个相机视角，
在Windows上完美运行，无需OpenCV GUI支持。

功能:
- 同时显示三个相机视角（top_cam, wrist_cam, right_cam）
- 使用MuJoCo渲染，Matplotlib显示
- 纯仿真测试，无需真实机械臂
- 支持键盘控制
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from lerobot.envs.so101_mujoco import So101MujocoEnv, So101MujocoConfig
from lerobot.utils.utils import init_logging, log_say


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_STACKING_XML = REPO_ROOT / "Sim_assets" / "SO-ARM100" / "Simulation" / "SO101" / "so101_block_stacking.xml"


def parse_args():
    parser = argparse.ArgumentParser("多相机显示测试 - MuJoCo渲染 + Matplotlib显示")
    
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
    parser.add_argument("--action_scale", type=float, default=0.1, help="Random action scale (0-1)")
    parser.add_argument("--mode", type=str, default="random", choices=["random", "hold", "sine"],
                       help="Control mode: random, hold, or sine wave")
    
    return parser.parse_args()


class MultiCameraDisplayMuJoCo:
    """使用MuJoCo渲染 + Matplotlib显示的多相机应用"""
    
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
        plt.ion()  # 交互模式
        
        # 计算合适的figure大小，保持宽高比
        aspect_ratio = self.args.camera_height / self.args.camera_width
        fig_width = 18  # 总宽度
        fig_height = fig_width * aspect_ratio / 3 + 1  # 3个相机并排，加1英寸给标题
        
        self.fig, self.axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
        self.fig.canvas.manager.set_window_title('Multi-Camera Display - MuJoCo + Matplotlib')
        
        # 初始化图像显示
        self.images = []
        for i, (ax, cam_name) in enumerate(zip(self.axes, self.camera_names)):
            ax.set_title(cam_name, fontsize=14, color='green', weight='bold')
            ax.axis('off')
            # 创建空白图像
            img = np.zeros((self.args.camera_height, self.args.camera_width, 3), dtype=np.uint8)
            im = ax.imshow(img, aspect='equal')  # 保持宽高比
            self.images.append(im)
        
        plt.tight_layout(pad=0.5)
        plt.show(block=False)
        
        # 保存背景用于blit优化
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        
        # 连接键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('close_event', self._on_close)
    
    def _on_key_press(self, event):
        """处理键盘事件"""
        if event.key == 'q' or event.key == 'escape':
            log_say("用户请求退出")
            self.running = False
        elif event.key == ' ':  # 空格键
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
        print("多相机显示测试 - MuJoCo渲染 + Matplotlib显示")
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
        print(f"  - 控制模式: {self.args.mode}")
        print(f"  - 动作缩放: {self.args.action_scale}")
        print(f"\n键盘控制:")
        print(f"  - q/Esc: 退出")
        print(f"  - 空格: 暂停/继续")
        print(f"  - r: 重置环境")
        print(f"  - h: 显示帮助")
        print(f"\n提示:")
        print(f"  - 三个相机视角同时显示")
        print(f"  - 窗口标题显示当前FPS")
        print("=" * 70 + "\n")
    
    def _generate_action(self, step: int) -> np.ndarray:
        """根据模式生成动作"""
        if self.args.mode == "random":
            return self.env.action_space.sample() * self.args.action_scale
        elif self.args.mode == "hold":
            return np.zeros(self.env.nu, dtype=np.float32)
        elif self.args.mode == "sine":
            t = step / self.args.control_fps
            freq = 0.5
            action = np.sin(2 * np.pi * freq * t) * self.args.action_scale
            return np.full(self.env.nu, action, dtype=np.float32)
        else:
            return np.zeros(self.env.nu, dtype=np.float32)
    
    def _render_cameras(self):
        """渲染所有相机视图"""
        for i, cam_name in enumerate(self.camera_names):
            if cam_name in self.renderers:
                try:
                    renderer = self.renderers[cam_name]
                    cam_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                    renderer.update_scene(self.env.data, camera=cam_id)
                    pixels = renderer.render()
                    
                    # 更新图像（不触发重绘）
                    self.images[i].set_data(pixels)
                except Exception as e:
                    log_say(f"渲染相机 {cam_name} 时出错: {e}")
        
        # 使用blit优化刷新（只更新变化的部分）
        for ax in self.axes:
            self.fig.canvas.restore_region(self.background)
            ax.draw_artist(ax.images[0])
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()
    
    def _update_fps(self, frame_start: float):
        """更新FPS跟踪"""
        current_time = time.perf_counter()
        self.frame_times.append(current_time)
        
        # 保持最近1秒的帧时间
        cutoff_time = current_time - 1.0
        self.frame_times = [t for t in self.frame_times if t > cutoff_time]
        
        # 计算FPS
        if len(self.frame_times) > 1:
            time_span = self.frame_times[-1] - self.frame_times[0]
            if time_span > 0:
                self.current_fps = (len(self.frame_times) - 1) / time_span
                # 更新窗口标题
                self.fig.canvas.manager.set_window_title(
                    f'Multi-Camera Display - FPS: {self.current_fps:.1f}'
                )
    
    def run(self):
        """运行测试主循环"""
        log_say("启动显示测试...")
        
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
                
                # 生成动作
                action = self._generate_action(step_count)
                
                # 执行动作
                self.obs, reward, terminated, truncated, info = self.env.step(action)
                
                step_count += 1
                
                # 更新显示（按显示帧率）
                if (loop_start - last_display_time) >= display_dt:
                    self._render_cameras()
                    self._update_fps(loop_start)
                    last_display_time = loop_start
                
                # 定期打印状态
                if step_count % (self.args.control_fps * 10) == 0:
                    log_say(f"运行中 - 步数: {step_count}, FPS: {self.current_fps:.1f}")
                
                # 如果环境结束，重置
                if terminated or truncated:
                    log_say("环境结束，重置")
                    self.obs, _ = self.env.reset()
                
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
        
        # 关闭环境
        if hasattr(self, 'env'):
            self.env.close()
        
        log_say("系统已关闭")


def main():
    args = parse_args()
    
    try:
        app = MultiCameraDisplayMuJoCo(args)
        app.run()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
