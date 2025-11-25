"""
多相机显示测试 - 纯仿真版本（无需真实机械臂）

功能:
- 测试多相机显示系统
- 使用随机动作或键盘控制
- 不需要真实Leader机械臂
- 用于验证显示功能和性能
"""
import argparse
import time
from pathlib import Path

import numpy as np

from lerobot.display.multi_camera_display import MultiCameraDisplay
from lerobot.envs.so101_mujoco import So101MujocoEnv, So101MujocoConfig
from lerobot.utils.utils import init_logging, log_say


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_STACKING_XML = REPO_ROOT / "Sim_assets" / "SO-ARM100" / "Simulation" / "SO101" / "so101_block_stacking.xml"


def parse_args():
    parser = argparse.ArgumentParser("多相机显示测试 - 纯仿真版本")
    
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
    parser.add_argument("--action_scale", type=float, default=0.1, help="Random action scale (0-1)")
    parser.add_argument("--mode", type=str, default="random", choices=["random", "hold", "sine"],
                       help="Control mode: random, hold, or sine wave")
    
    return parser.parse_args()


class MultiCameraDisplayTest:
    """多相机显示测试应用"""
    
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
        
        # 创建多相机显示
        log_say("初始化多相机显示...")
        self.display = MultiCameraDisplay(
            model=self.env.model,
            data=self.env.data,
            camera_names=camera_names,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            target_fps=args.display_fps,
            window_name="Multi-Camera Display Test",
            show_overlays=args.show_overlays,
        )
        
        # 设置初始状态
        self.display.set_status("connected", True)
        
        # 打印系统信息
        self._print_system_info()
    
    def _print_system_info(self):
        """打印系统配置信息"""
        print("\n" + "=" * 70)
        print("多相机显示测试 - 纯仿真版本")
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
        print(f"  - 控制模式: {self.args.mode}")
        print(f"  - 动作缩放: {self.args.action_scale}")
        print(f"\n键盘控制:")
        print(f"  - q: 退出")
        print(f"  - p: 暂停/继续")
        print(f"  - r: 重置环境")
        print(f"  - s: 保存截图")
        print(f"  - h: 显示帮助")
        print("=" * 70 + "\n")
    
    def _generate_action(self, step: int) -> np.ndarray:
        """
        根据模式生成动作
        
        Args:
            step: 当前步数
            
        Returns:
            动作数组
        """
        if self.args.mode == "random":
            # 随机动作
            return self.env.action_space.sample() * self.args.action_scale
        
        elif self.args.mode == "hold":
            # 保持当前位置
            return np.zeros(self.env.nu, dtype=np.float32)
        
        elif self.args.mode == "sine":
            # 正弦波动作
            t = step / self.args.control_fps
            freq = 0.5  # Hz
            action = np.sin(2 * np.pi * freq * t) * self.args.action_scale
            return np.full(self.env.nu, action, dtype=np.float32)
        
        else:
            return np.zeros(self.env.nu, dtype=np.float32)
    
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
        filename = snapshot_dir / f"test_snapshot_{timestamp}.png"
        cv2.imwrite(str(filename), composed_image)
        log_say(f"截图已保存: {filename}")
    
    def run(self):
        """运行测试主循环"""
        log_say("启动显示测试...")
        
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
                
                # 生成动作
                action = self._generate_action(step_count)
                
                # 执行动作
                self.obs, reward, terminated, truncated, info = self.env.step(action)
                
                step_count += 1
                
                # 定期打印状态
                if step_count % (self.args.control_fps * 10) == 0:  # 每10秒
                    display_fps = self.display.get_fps()
                    log_say(f"运行中 - 步数: {step_count}, 显示FPS: {display_fps:.1f}")
                
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
        
        # 停止显示
        if hasattr(self, 'display'):
            self.display.stop()
        
        # 关闭环境
        if hasattr(self, 'env'):
            self.env.close()
        
        log_say("系统已关闭")


def main():
    args = parse_args()
    
    try:
        app = MultiCameraDisplayTest(args)
        app.run()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
