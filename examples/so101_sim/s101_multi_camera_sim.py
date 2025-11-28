"""
S101机械臂PyBullet仿真环境 - 多相机版本
包含机械臂、桌子和两个立方体，以及三个独立的相机窗口
"""
import pybullet as p
import pybullet_data
import time
import numpy as np
import os

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("警告: 未安装OpenCV，将使用PyBullet内置相机显示")

from s101_pybullet_sim import S101SimEnv


class S101MultiCameraEnv(S101SimEnv):
    """带多相机显示的S101仿真环境"""
    
    def __init__(self, gui=True, use_opencv=True, cube_size=0.025, cube_distance=0.20):
        """初始化仿真环境"""
        self.use_opencv = use_opencv and HAS_CV2
        super().__init__(gui=gui, cube_size=cube_size, cube_distance=cube_distance)
        
        # 如果使用OpenCV，禁用PyBullet内置的相机预览
        if self.use_opencv and gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            print("\n使用OpenCV显示多相机视角（已禁用PyBullet内置相机显示）")
            print("按 'q' 键关闭相机窗口")
    
    def render_cameras_opencv(self):
        """使用OpenCV渲染并显示三个相机视角"""
        if not self.use_opencv:
            return
        
        # 更新腕部相机
        self.update_wrist_camera()
        
        # 获取三个视角的RGB图像（不包含深度和分割）
        top_img = p.getCameraImage(
            self.camera_width,
            self.camera_height,
            self.top_view_matrix,
            self.projection_matrix,
            shadow=1,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        side_img = p.getCameraImage(
            self.camera_width,
            self.camera_height,
            self.side_view_matrix,
            self.projection_matrix,
            shadow=1,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        wrist_img = p.getCameraImage(
            self.camera_width,
            self.camera_height,
            self.wrist_view_matrix,
            self.projection_matrix,
            shadow=1,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # 转换为OpenCV格式 (BGR)
        top_rgb = np.array(top_img[2]).reshape(self.camera_height, self.camera_width, 4)[:, :, :3]
        side_rgb = np.array(side_img[2]).reshape(self.camera_height, self.camera_width, 4)[:, :, :3]
        wrist_rgb = np.array(wrist_img[2]).reshape(self.camera_height, self.camera_width, 4)[:, :, :3]
        
        # RGB转BGR
        top_bgr = cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR)
        side_bgr = cv2.cvtColor(side_rgb, cv2.COLOR_RGB2BGR)
        wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
        
        # 添加文字标签
        cv2.putText(top_bgr, "Top View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(side_bgr, "Side View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(wrist_bgr, "Wrist View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 水平拼接三个视图
        combined = np.hstack([top_bgr, side_bgr, wrist_bgr])
        
        # 显示
        cv2.imshow("S101 Robot - Multi Camera Views", combined)
        
        # 等待1ms，允许窗口更新
        key = cv2.waitKey(1)
        return key


def main():
    """主函数 - 多相机演示"""
    # 创建仿真环境
    env = S101MultiCameraEnv(gui=True, use_opencv=HAS_CV2)
    
    # 设置主相机视角（机械臂在桌子下方）
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=150,  # 调整yaw角度以适应新位置
        cameraPitch=-35,
        cameraTargetPosition=[0, -0.2, 0.8]  # 看向机械臂和工作区域
    )
    
    # 添加关节角度调试滑块
    joint_sliders = []
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    joint_limits = [
        (-1.92, 1.92),
        (-1.75, 1.75),
        (-1.69, 1.69),
        (-1.66, 1.66),
        (-2.74, 2.84),
        (-0.17, 1.75)
    ]
    
    for i, (name, (min_val, max_val)) in enumerate(zip(joint_names, joint_limits)):
        slider = p.addUserDebugParameter(name, min_val, max_val, 0.0)
        joint_sliders.append(slider)
    
    print("\n开始仿真演示...")
    print("使用右侧滑块控制机械臂关节")
    if HAS_CV2:
        print("OpenCV窗口显示三个相机视角:")
        print("  - 顶部视角 (俯视)")
        print("  - 侧面视角 (侧视)")
        print("  - 腕部视角 (跟随末端)")
        print("按 'q' 键或 Ctrl+C 退出\n")
    else:
        print("PyBullet左侧显示相机视角")
        print("按 Ctrl+C 退出\n")
    
    try:
        step = 0
        # 持续运行仿真
        while True:
            # 从滑块读取关节位置
            joint_positions = [p.readUserDebugParameter(slider) for slider in joint_sliders]
            
            env.set_joint_positions(joint_positions)
            env.step()
            
            # 每5步渲染一次相机
            if step % 5 == 0:
                if env.use_opencv:
                    key = env.render_cameras_opencv()
                    if key == ord('q'):
                        print("\n用户按下 'q' 键退出")
                        break
                else:
                    env.render_cameras()
            
            time.sleep(1./240.)
            
            # 每240步（约1秒）打印一次当前关节位置
            if step % 240 == 0:
                current_pos = env.get_joint_positions()
                print(f"时间 {step/240:.1f}s, 关节位置(弧度): ", end="")
                for name, pos in zip(joint_names, current_pos):
                    print(f"{name}={pos:.3f} ", end="")
                print()
            
            step += 1
    
    except KeyboardInterrupt:
        print("\n仿真被用户中断")
    
    finally:
        if HAS_CV2:
            cv2.destroyAllWindows()
        print("关闭仿真环境")
        env.close()


if __name__ == "__main__":
    main()
