# S101机械臂PyBullet仿真环境

这是一个使用PyBullet创建的S101机械臂仿真环境，包含机械臂、桌子和两个立方体。

## 环境组成

- **S101机械臂**: 6自由度机械臂，放置在桌面下方（0, -0.35, 0.75），旋转90度使末端朝向+Y方向（前方），统一黄色外观
- **桌子**: 1.0m x 0.8m的工作台面，高度0.75m，带有4条圆柱形桌腿
- **立方体1**: 红色立方体，默认尺寸2.5cm，位于机械臂前方20cm右侧
- **立方体2**: 蓝色立方体，默认尺寸2.5cm，位于机械臂前方20cm左侧

**立方体位置说明**:
- 默认大小: 2.5cm（便于夹爪抓取）
- 默认距离: 机械臂前方20cm（便于够到）
- 可通过命令行参数自定义大小和距离

## 特性

- **交互式关节控制**: 使用GUI滑块实时调整每个关节角度
- **多视角相机**: 左侧显示三个不同视角
  - 顶部视角：俯视整个工作区域
  - 侧面视角：从侧面观察机械臂
  - 腕部视角：跟随机械臂末端移动
- **实时反馈**: 在终端每秒显示当前关节位置
- **稳定的桌面结构**: 带腿的桌子固定在地面上
- **统一的机械臂外观**: 所有部件使用一致的黄色配色
- **持续运行**: 程序持续运行直到手动退出（Ctrl+C）
- **优化的工作空间**: 机械臂放置在桌子一侧，工作范围更大

## 依赖安装

```bash
pip install pybullet numpy
```

## 使用方法

### 1. 基本运行（交互式滑块控制）

```bash
# 基本版本（PyBullet内置相机显示）
python examples/so101_sim/s101_pybullet_sim.py

# 多相机版本（使用OpenCV显示三个独立相机窗口）
python examples/so101_sim/s101_multi_camera_sim.py
```

运行后会打开PyBullet GUI窗口，右侧会显示6个滑块，可以实时控制每个关节的角度。

- **基本版本**: 不显示相机画面，专注于机械臂控制
- **多相机版本** (推荐): 使用OpenCV在独立窗口显示三个RGB相机视角（需要安装opencv-python）
  - 顶部视角：俯视整个工作区域
  - 侧面视角：从侧面观察机械臂
  - 腕部视角：跟随机械臂末端移动
  - 三个视角水平排列在一个窗口中，实时更新

程序会持续运行直到按Ctrl+C退出（多相机版本也可以按'q'键退出）。

### 2. 使用真实Leader臂控制仿真Follower臂

#### 步骤1: 测试Leader臂连接

```bash
# 测试Leader臂是否正确连接
python examples/so101_sim/test_leader_connection.py

# 指定自定义端口
python examples/so101_sim/test_leader_connection.py --port /dev/ttyUSB0
```

这个测试脚本会：
- 验证Leader臂连接
- 读取关节位置
- 确认通信正常

#### 步骤2: 运行Leader控制仿真

```bash
# 使用默认端口 /dev/ttyACM0
python examples/so101_sim/s101_leader_to_sim.py

# 或使用快速启动脚本
./examples/so101_sim/run_leader_control.sh

# 指定自定义端口
python examples/so101_sim/s101_leader_to_sim.py --port /dev/ttyUSB0

# 禁用OpenCV相机显示
python examples/so101_sim/s101_leader_to_sim.py --no-opencv
```

这个模式下：
- 连接真实的SO101 Leader臂（通过串口）
- Leader臂的运动会实时同步到仿真环境中的Follower臂
- 同时显示三个相机视角（如果启用OpenCV）
- 适合用于测试遥操作、数据采集、算法验证等场景

**命令行参数**:
- `--port`: 指定串口（默认: /dev/ttyACM0）
- `--id`: Leader臂ID，用于区分不同机械臂（默认: main）
- `--no-calibrate`: 跳过校准，需要已有校准文件
- `--no-opencv`: 禁用OpenCV相机显示
- `--cube-size`: 立方体大小（米），默认0.025（2.5cm）
- `--cube-distance`: 立方体距离机械臂的距离（米），默认0.20（20cm）
- `--debug`: 启用调试模式，显示gripper关节映射信息

**校准说明**:
- 首次运行需要校准Leader臂（约2-3分钟）
- 校准文件保存在: `examples/so101_sim/Calibration/{id}.json`
- 校准完成后，下次可使用 `--no-calibrate` 跳过
- 详细说明请查看 [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md)
- 校准文件目录说明: [Calibration/README.md](Calibration/README.md)

**注意**: 
- 确保Leader臂已正确连接到指定的串口
- 需要有串口访问权限：`sudo usermod -a -G dialout $USER`（然后重新登录）
- Leader臂的归一化值 [-100, 100] 会自动转换为仿真环境的弧度值

### 在代码中使用

```python
from examples.so101_sim.s101_pybullet_sim import S101SimEnv

# 创建环境
env = S101SimEnv(gui=True)

# 设置关节位置 (6个关节: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
joint_positions = [0.0, 0.5, -0.5, 0.3, 0.0, 0.0]
env.set_joint_positions(joint_positions)

# 执行仿真步
for _ in range(1000):
    env.step()

# 获取当前关节位置
current_positions = env.get_joint_positions()
print(f"当前关节位置: {current_positions}")

# 重置环境
env.reset()

# 关闭环境
env.close()
```

## API说明

### S101SimEnv类

#### 初始化
- `S101SimEnv(gui=True)`: 创建仿真环境
  - `gui`: 是否显示图形界面

#### 方法
- `set_joint_positions(positions)`: 设置关节目标位置
  - `positions`: 长度为6的列表，对应6个关节的目标角度（弧度）
  
- `get_joint_positions()`: 获取当前关节位置
  - 返回: 长度为6的列表，包含当前关节角度

- `step()`: 执行一步物理仿真

- `reset()`: 重置环境到初始状态

- `close()`: 关闭仿真环境

## 关节信息

机械臂有6个可控关节：

1. **shoulder_pan**: 肩部旋转，范围 [-1.92, 1.92] 弧度
2. **shoulder_lift**: 肩部抬升，范围 [-1.75, 1.75] 弧度
3. **elbow_flex**: 肘部弯曲，范围 [-1.69, 1.69] 弧度
4. **wrist_flex**: 腕部弯曲，范围 [-1.66, 1.66] 弧度
5. **wrist_roll**: 腕部旋转，范围 [-2.74, 2.84] 弧度
6. **gripper**: 夹爪开合，范围 [-0.17, 1.75] 弧度

## 坐标系统

- 原点位于地面中心
- 桌面高度: 0.75m
- 机械臂基座位于桌面下方: (0, -0.35, 0.75)
- 机械臂朝向: 旋转90度（绕Z轴），末端朝向+Y方向（前方）
- 立方体默认位置（距离20cm）:
  - 立方体1（红色）: (0.08, -0.15, 0.765)
  - 立方体2（蓝色）: (-0.08, -0.15, 0.765)

**自定义立方体位置**:
```bash
# 更近的立方体（15cm）
python examples/so101_sim/s101_leader_to_sim.py --cube-distance 0.15

# 更小的立方体（2cm）
python examples/so101_sim/s101_leader_to_sim.py --cube-size 0.02

# 组合使用
python examples/so101_sim/s101_leader_to_sim.py --cube-size 0.02 --cube-distance 0.15
```

## 相机视角

仿真环境提供三个实时相机视角：

1. **顶部视角**: 从正上方俯视整个工作区域，便于观察机械臂的平面运动轨迹和工作空间布局
2. **侧面视角**: 从机械臂左前方45度角观察，便于观察机械臂的姿态和高度变化（机械臂在桌子下方，相机从左前方观察）
3. **腕部视角**: 模拟真实相机安装位置，固定在gripper舵机（第6关节）侧面，水平向前看并稍微向下倾斜，可以看到桌面和夹爪前方的立方体，随机械臂移动和旋转提供第一人称视角

## 演示脚本

### 抓取和放置演示

```bash
# 运行抓取演示（选项1）
python examples/so101_sim/demo_pick_and_place.py 1

# 运行简单运动演示（选项2）
python examples/so101_sim/demo_pick_and_place.py 2
```

演示脚本展示了如何：
- 使用平滑运动控制机械臂
- 执行抓取和放置任务
- 测试各个关节的独立运动

## 注意事项

1. 确保URDF文件路径正确，相对路径为 `../../Sim_assets/SO-ARM100/Simulation/SO101/so101_new_calib.urdf`
2. 关节角度使用弧度制
3. 仿真频率默认为240Hz
4. 可以通过修改`_create_cube`和`_create_table`方法来自定义物体属性
5. Leader臂控制模式需要正确的串口权限和校准文件

## 常见问题

### Gripper角度映射问题

**问题**: 真实机械臂的gripper闭合时，仿真中显示半开状态

**原因**: Gripper关节使用不同的归一化范围
- 前5个关节: 归一化范围 [-100, 100]
- Gripper关节: 归一化范围 [0, 100]

**解决方案**: 已在代码中修复，gripper使用正确的映射

**验证方法**:
```bash
# 测试映射关系
python examples/so101_sim/test_gripper_mapping.py

# 启用调试模式查看实时映射
python examples/so101_sim/s101_leader_to_sim.py --debug
```

### 抓取问题

**问题**: 立方体夹不起来或容易滑落

**已优化**:
- ✅ 增加立方体和夹爪的摩擦力
- ✅ 减轻立方体质量（20g）
- ✅ 优化接触物理参数
- ✅ 默认使用更小的立方体（2.5cm）

**抓取技巧**:
- 从正上方接近立方体
- 确保夹爪完全闭合
- 缓慢抬起机械臂
- 详细说明: [GRASPING_TIPS.md](GRASPING_TIPS.md)

### 相机画面太小

**问题**: 相机画面分辨率低，难以观察细节

**解决方案**: 已将相机分辨率从320x240增加到640x480

**如需更大画面**: 编辑 `s101_pybullet_sim.py` 中的 `width` 和 `height` 参数
