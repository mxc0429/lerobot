"""
S101机械臂PyBullet仿真环境
包含机械臂、桌子和两个立方体
"""
import pybullet as p
import pybullet_data
import time
import numpy as np
import os


class S101SimEnv:
    def __init__(self, gui=True, cube_size=0.025, cube_distance=0.20):
        """
        初始化仿真环境
        
        Args:
            gui: 是否显示GUI
            cube_size: 立方体大小（米），默认0.025（2.5cm）
            cube_distance: 立方体距离机械臂的距离（米），默认0.20（20cm）
        """
        self.cube_size = cube_size
        self.cube_distance = cube_distance
        
        # 连接物理引擎
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        # 设置搜索路径
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 设置重力
        p.setGravity(0, 0, -9.81)
        
        # 加载地面
        self.plane_id = p.loadURDF("plane.urdf")
        
        # 加载桌子
        self.table_id = self._create_table()
        
        # 加载机械臂
        self.robot_id = self._load_robot()
        
        # 创建立方体（调整位置和大小，便于抓取）
        # 机械臂在(0, -0.35, 0.75)，立方体放在机械臂前方（+Y方向）
        # 计算立方体位置：机械臂基座 + 距离
        cube_y = -0.35 + self.cube_distance
        cube_z = 0.75 + self.cube_size / 2 + 0.015  # 桌面高度 + 半个立方体高度 + 小间隙
        
        self.cube1_id = self._create_cube(
            position=[0.08, cube_y, cube_z],  # 机械臂前方右侧
            color=[1, 0, 0, 1],
            size=self.cube_size
        )
        self.cube2_id = self._create_cube(
            position=[-0.08, cube_y, cube_z],  # 机械臂前方左侧
            color=[0, 0, 1, 1],
            size=self.cube_size
        )
        
        print(f"立方体大小: {self.cube_size*100:.1f}cm")
        print(f"立方体距离: {self.cube_distance*100:.1f}cm")
        
        # 获取关节信息
        self.joint_indices = self._get_joint_indices()
        
        # 设置多视角相机
        if gui:
            self._setup_cameras()
        
        print("仿真环境初始化完成")
        print(f"机械臂ID: {self.robot_id}")
        print(f"桌子ID: {self.table_id}")
        print(f"立方体1 ID: {self.cube1_id}")
        print(f"立方体2 ID: {self.cube2_id}")
        print(f"可控关节: {self.joint_indices}")
    
    def _create_table(self):
        """创建带腿的桌子"""
        # 桌面尺寸
        table_height = 0.75
        table_thickness = 0.05
        table_width = 1.0
        table_depth = 0.8
        
        # 桌腿尺寸
        leg_radius = 0.03
        leg_height = table_height - table_thickness
        
        # 创建桌面碰撞和视觉形状
        table_top_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[table_width/2, table_depth/2, table_thickness/2]
        )
        table_top_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[table_width/2, table_depth/2, table_thickness/2],
            rgbaColor=[0.6, 0.4, 0.2, 1]
        )
        
        # 创建桌腿碰撞和视觉形状
        leg_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=leg_radius,
            height=leg_height
        )
        leg_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=leg_radius,
            length=leg_height,
            rgbaColor=[0.4, 0.3, 0.2, 1]
        )
        
        # 桌腿位置（相对于桌面中心）
        leg_offset_x = table_width/2 - 0.1
        leg_offset_y = table_depth/2 - 0.1
        leg_positions = [
            [leg_offset_x, leg_offset_y, -leg_height/2 - table_thickness/2],
            [leg_offset_x, -leg_offset_y, -leg_height/2 - table_thickness/2],
            [-leg_offset_x, leg_offset_y, -leg_height/2 - table_thickness/2],
            [-leg_offset_x, -leg_offset_y, -leg_height/2 - table_thickness/2]
        ]
        
        # 组合所有碰撞和视觉形状
        collision_shapes = [table_top_collision] + [leg_collision] * 4
        visual_shapes = [table_top_visual] + [leg_visual] * 4
        
        link_positions = [[0, 0, 0]] + leg_positions
        link_orientations = [[0, 0, 0, 1]] * 5
        
        # 创建带腿的桌子
        table_id = p.createMultiBody(
            baseMass=0,  # 固定桌子，不受重力影响
            baseCollisionShapeIndex=table_top_collision,
            baseVisualShapeIndex=table_top_visual,
            basePosition=[0, 0, table_height - table_thickness/2],
            linkMasses=[0] * 4,
            linkCollisionShapeIndices=[leg_collision] * 4,
            linkVisualShapeIndices=[leg_visual] * 4,
            linkPositions=leg_positions,
            linkOrientations=[[0, 0, 0, 1]] * 4,
            linkInertialFramePositions=[[0, 0, 0]] * 4,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * 4,
            linkParentIndices=[0] * 4,
            linkJointTypes=[p.JOINT_FIXED] * 4,
            linkJointAxis=[[0, 0, 1]] * 4
        )
        
        return table_id
    
    def _load_robot(self):
        """加载S101机械臂"""
        # 获取URDF文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(
            current_dir, 
            "../../Sim_assets/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
        )
        
        # 机械臂放置在桌面下方（-Y方向）
        # 桌子深度0.8m，将机械臂放在y=-0.35的位置（桌子下方边缘）
        robot_base_position = [0, -0.35, 0.75]
        # 旋转机械臂90度（绕Z轴），使末端朝向+Y方向（前方）
        robot_base_orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])  # 90度 = π/2
        
        # 加载机械臂
        robot_id = p.loadURDF(
            urdf_path,
            basePosition=robot_base_position,
            baseOrientation=robot_base_orientation,
            useFixedBase=True,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL
        )
        
        # 统一设置所有关节的颜色为黄色
        num_joints = p.getNumJoints(robot_id)
        for i in range(-1, num_joints):  # -1 表示基座
            p.changeVisualShape(robot_id, i, rgbaColor=[1.0, 0.82, 0.12, 1.0])
        
        # 增加夹爪的摩擦力，便于抓取物体
        # 找到gripper相关的连杆并设置高摩擦力
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            
            # 为gripper和moving_jaw设置高摩擦力
            if 'gripper' in joint_name.lower() or 'jaw' in joint_name.lower():
                p.changeDynamics(
                    robot_id,
                    i,
                    lateralFriction=2.0,  # 高侧向摩擦力
                    spinningFriction=1.0,
                    rollingFriction=0.5,
                    contactStiffness=30000,  # 增加接触刚度
                    contactDamping=1000      # 增加接触阻尼
                )
        
        return robot_id
    
    def _create_cube(self, position, color, size=0.05):
        """创建立方体"""
        # 创建碰撞形状
        cube_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[size/2, size/2, size/2]
        )
        
        # 创建视觉形状
        cube_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[size/2, size/2, size/2],
            rgbaColor=color
        )
        
        # 创建立方体（较轻的质量，便于抓取）
        cube_id = p.createMultiBody(
            baseMass=0.02,  # 减轻质量到20g，更容易抓取
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=position
        )
        
        # 设置立方体的摩擦力和其他物理属性
        p.changeDynamics(
            cube_id,
            -1,  # -1表示基座
            lateralFriction=1.5,  # 增加侧向摩擦力
            spinningFriction=0.5,  # 增加旋转摩擦力
            rollingFriction=0.1,   # 增加滚动摩擦力
            restitution=0.1,       # 降低弹性（减少弹跳）
            linearDamping=0.5,     # 增加线性阻尼
            angularDamping=0.5     # 增加角阻尼
        )
        
        return cube_id
    
    def _get_joint_indices(self):
        """获取可控关节索引"""
        joint_indices = []
        num_joints = p.getNumJoints(self.robot_id)
        
        print(f"\n机械臂关节信息:")
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            print(f"关节 {i}: {joint_name}, 类型: {joint_type}")
            
            # 只添加可旋转关节 (JOINT_REVOLUTE = 0)
            if joint_type == p.JOINT_REVOLUTE:
                joint_indices.append(i)
        
        return joint_indices
    
    def _setup_cameras(self):
        """设置三个不同视角的相机"""
        # 默认禁用相机预览显示（避免闪烁，如需显示可改为1）
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        
        # 相机参数（增大分辨率以获得更清晰的画面）
        width = 640  # 从320增加到640
        height = 480  # 从240增加到480
        fov = 60
        aspect = width / height
        near = 0.02
        far = 5
        
        # 1. 顶部视角 - 正上方俯视整个工作区域
        # 机械臂在(0, -0.35, 0.75)，相机在正上方
        self.top_view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0, 0, 1.8],  # 相机在正上方
            cameraTargetPosition=[0, 0, 0.75],  # 看向桌面中心
            cameraUpVector=[0, 1, 0]  # Y轴向上（朝向机械臂后方）
        )
        
        # 2. 侧面视角 - 从机械臂左前方观察
        # 从45度角观察机械臂
        self.side_view_matrix = p.computeViewMatrix(
            cameraEyePosition=[-0.5, 0.3, 0.9],  # 相机在左前方
            cameraTargetPosition=[0, -0.35, 0.85],  # 看向机械臂中心
            cameraUpVector=[0, 0, 1]  # Z轴向上
        )
        
        # 3. 腕部视角 - 固定在gripper关节侧面，跟随机械臂移动
        # 初始位置，后续会动态更新
        self.wrist_view_matrix = p.computeViewMatrix(
            cameraEyePosition=[-0.35, 0, 1.0],
            cameraTargetPosition=[-0.35, 0, 0.85],
            cameraUpVector=[0, 0, 1]
        )
        
        # 记录gripper关节的索引（第6个关节，夹爪舵机）
        self.gripper_joint_index = 5  # 第6个关节（索引从0开始）
        
        # 投影矩阵（所有相机共用）
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near,
            farVal=far
        )
        
        self.camera_width = width
        self.camera_height = height
        
        print("\n相机视角已设置:")
        print("- 顶部视角: 正上方俯视整个工作区域")
        print("- 侧面视角: 从机械臂左前方45度角观察")
        print("- 腕部视角: 固定在gripper舵机侧面，朝向前方，模拟真实相机安装位置")
    
    def get_gripper_link_state(self):
        """获取gripper关节（夹爪舵机）的状态"""
        # 获取gripper关节对应的连杆状态
        # gripper是第6个关节（索引5）
        link_state = p.getLinkState(self.robot_id, self.gripper_joint_index, computeForwardKinematics=True)
        return link_state[0], link_state[1]  # 返回位置和方向
    
    def update_wrist_camera(self):
        """更新腕部相机视角 - 安装在gripper舵机侧面的相机"""
        try:
            # 获取gripper关节（夹爪舵机）的位置和方向
            gripper_pos, gripper_orn = self.get_gripper_link_state()
            
            # 将四元数转换为旋转矩阵，获取关节的朝向
            rotation_matrix = p.getMatrixFromQuaternion(gripper_orn)
            
            # 提取各个轴向量
            x_axis = np.array([rotation_matrix[0], rotation_matrix[3], rotation_matrix[6]])
            y_axis = np.array([rotation_matrix[1], rotation_matrix[4], rotation_matrix[7]])
            z_axis = np.array([rotation_matrix[2], rotation_matrix[5], rotation_matrix[8]])
            
            # 相机安装在gripper舵机的侧面
            # 相机位置偏移：在侧面，稍微向上
            camera_offset_local = np.array([0.01, 0.04, 0.04])
            
            # 将局部偏移转换到世界坐标系
            camera_offset_world = (
                x_axis * camera_offset_local[0] +
                y_axis * camera_offset_local[1] +
                z_axis * camera_offset_local[2]
            )
            
            # 相机位置 = gripper关节位置 + 偏移
            camera_pos = [
                gripper_pos[0] + camera_offset_world[0],
                gripper_pos[1] + camera_offset_world[1],
                gripper_pos[2] + camera_offset_world[2]
            ]
            
            # 相机朝向：水平向前看
            # 由于机械臂旋转了90度，gripper_link的Z轴指向机械臂的前方
            # 相机应该沿着gripper的Z轴方向看（前方），保持水平
            
            # 直接使用gripper的Z轴作为相机朝向（前方）
            look_direction = z_axis
            look_direction = look_direction / np.linalg.norm(look_direction)
            
            look_distance = 0.3
            target_pos = [
                camera_pos[0] + look_direction[0] * look_distance,
                camera_pos[1] + look_direction[1] * look_distance,
                camera_pos[2] + look_direction[2] * look_distance
            ]
            
            # 相机的上向量：使用世界坐标系的Z轴（真实的上方）
            # 这样相机画面的上方对应真实的上方
            self.wrist_view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_pos,
                cameraTargetPosition=target_pos,
                cameraUpVector=[0, 0, 1]  # 使用世界坐标系的上方向
            )
        except Exception as e:
            pass  # 如果获取失败，保持原有视角
    
    def render_cameras(self):
        """渲染三个相机视角"""
        # 更新腕部相机
        self.update_wrist_camera()
        
        # 获取三个视角的图像
        # 1. 顶部视角
        top_img = p.getCameraImage(
            self.camera_width,
            self.camera_height,
            self.top_view_matrix,
            self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # 2. 侧面视角
        side_img = p.getCameraImage(
            self.camera_width,
            self.camera_height,
            self.side_view_matrix,
            self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # 3. 腕部视角
        wrist_img = p.getCameraImage(
            self.camera_width,
            self.camera_height,
            self.wrist_view_matrix,
            self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        return top_img, side_img, wrist_img
    
    def set_joint_positions(self, positions):
        """设置关节位置"""
        if len(positions) != len(self.joint_indices):
            print(f"警告: 位置数量({len(positions)})与关节数量({len(self.joint_indices)})不匹配")
            return
        
        for joint_idx, pos in zip(self.joint_indices, positions):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=pos,
                force=500
            )
    
    def get_joint_positions(self):
        """获取当前关节位置"""
        positions = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            positions.append(joint_state[0])
        return positions
    
    def step(self):
        """执行一步仿真"""
        p.stepSimulation()
    
    def reset(self):
        """重置环境"""
        # 重置机械臂到初始位置
        self.set_joint_positions([0] * len(self.joint_indices))
        
        # 计算立方体位置
        cube_y = -0.35 + self.cube_distance
        cube_z = 0.75 + self.cube_size / 2 + 0.015
        
        # 重置立方体位置
        p.resetBasePositionAndOrientation(
            self.cube1_id,
            [0.08, cube_y, cube_z],
            [0, 0, 0, 1]
        )
        p.resetBasePositionAndOrientation(
            self.cube2_id,
            [-0.08, cube_y, cube_z],
            [0, 0, 0, 1]
        )
    
    def close(self):
        """关闭仿真"""
        p.disconnect()


def main():
    """主函数 - 演示仿真环境"""
    # 创建仿真环境
    env = S101SimEnv(gui=True)
    
    # 设置主相机视角（观察整个场景）
    # 机械臂在桌子下方，调整相机角度
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
    print("左侧显示三个相机视角:")
    print("  - 顶部视角 (俯视)")
    print("  - 侧面视角 (侧视)")
    print("  - 腕部视角 (跟随末端)")
    print("按 Ctrl+C 退出\n")
    
    try:
        step = 0
        # 持续运行仿真
        while True:
            # 从滑块读取关节位置
            joint_positions = [p.readUserDebugParameter(slider) for slider in joint_sliders]
            
            env.set_joint_positions(joint_positions)
            env.step()
            
            # 每10步渲染一次相机（降低渲染频率以提高性能）
            if step % 10 == 0:
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
        print("关闭仿真环境")
        env.close()


if __name__ == "__main__":
    main()
