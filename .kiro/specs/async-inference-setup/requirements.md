# Requirements Document

## Introduction

本文档定义了在 LeRobot 系统中实现异步推理的需求，以支持使用远程服务器进行模型推理，同时在本地低算力设备上控制 SO101 机械臂。该方案解决了本地设备显卡故障和算力不足的问题。

## Glossary

- **PolicyServer**: 运行在服务器端的策略推理服务，负责接收观察数据并返回动作预测
- **RobotClient**: 运行在本地设备的机器人客户端，负责控制机械臂和相机，并与 PolicyServer 通信
- **AsyncInference**: 异步推理模式，将动作预测和动作执行解耦，消除等待推理的空闲时间
- **ActionChunk**: 策略一次性输出的多个连续动作序列
- **SO101**: 目标机械臂型号，包括 follower（执行臂）和 leader（示教臂）

## Requirements

### Requirement 1: 服务器端策略推理服务

**User Story:** 作为系统管理员，我希望在远程服务器上运行策略推理服务，以便利用服务器的强大算力进行模型推理。

#### Acceptance Criteria

1. WHEN 启动 PolicyServer 时，THE PolicyServer SHALL 在指定的 IP 地址和端口上监听客户端连接
2. WHEN RobotClient 发起首次握手时，THE PolicyServer SHALL 根据客户端请求加载指定的策略模型
3. WHEN PolicyServer 接收到观察数据时，THE PolicyServer SHALL 在 CUDA 设备上执行模型推理并返回动作块
4. THE PolicyServer SHALL 支持 SmolVLA 策略类型的推理
5. WHEN 推理完成时，THE PolicyServer SHALL 将动作块发送回 RobotClient

### Requirement 2: 本地机器人客户端控制

**User Story:** 作为机器人操作员，我希望在本地低算力设备上运行机器人控制程序，以便在不依赖本地 GPU 的情况下控制机械臂执行任务。

#### Acceptance Criteria

1. THE RobotClient SHALL 连接到 SO101 follower 机械臂（端口 /dev/ttyACM0）
2. THE RobotClient SHALL 连接到 SO101 leader 示教臂（端口 /dev/ttyACM1）用于遥操作
3. THE RobotClient SHALL 从三个 OpenCV 相机采集图像数据（/dev/video2, /dev/video4, /dev/video6）
4. WHEN RobotClient 启动时，THE RobotClient SHALL 与远程 PolicyServer（172.21.137.91）建立连接
5. THE RobotClient SHALL 以 30 FPS 的频率采集相机图像并发送到 PolicyServer

### Requirement 3: 异步推理流程管理

**User Story:** 作为系统架构师，我希望实现异步推理机制，以便在策略计算下一个动作块时机械臂能够继续执行当前动作，消除空闲等待时间。

#### Acceptance Criteria

1. WHEN 动作队列大小低于阈值时，THE RobotClient SHALL 发送新的观察数据到 PolicyServer
2. THE RobotClient SHALL 维护一个动作队列，存储从 PolicyServer 接收的动作块
3. WHILE 动作队列非空，THE RobotClient SHALL 持续从队列中取出动作并执行
4. THE RobotClient SHALL 使用加权平均方法聚合重叠部分的动作
5. WHEN 接收到新的动作块时，THE RobotClient SHALL 将其添加到动作队列中

### Requirement 4: 网络通信配置

**User Story:** 作为网络工程师，我希望配置客户端和服务器之间的网络通信参数，以便确保稳定的数据传输。

#### Acceptance Criteria

1. THE PolicyServer SHALL 监听在 172.21.137.91 的指定端口（建议 8080）
2. THE RobotClient SHALL 连接到服务器地址 172.21.137.91:8080
3. THE System SHALL 支持通过 TCP/IP 协议传输观察数据和动作数据
4. WHEN 网络连接中断时，THE RobotClient SHALL 停止机械臂运动并报告错误
5. THE System SHALL 在客户端和服务器之间传输压缩的图像数据以减少带宽占用

### Requirement 5: 策略模型配置

**User Story:** 作为 ML 工程师，我希望配置策略模型的加载路径和推理参数，以便使用训练好的模型进行推理。

#### Acceptance Criteria

1. THE PolicyServer SHALL 从指定路径加载策略模型（outputs/train/smolvla_so101_test2）
2. THE PolicyServer SHALL 在 CUDA 设备上运行模型推理
3. THE PolicyServer SHALL 每次推理输出指定数量的动作（actions_per_chunk）
4. THE System SHALL 支持配置任务描述（"Pick the cube to place"）
5. THE System SHALL 支持配置动作块大小阈值（chunk_size_threshold）

### Requirement 6: 相机配置和数据采集

**User Story:** 作为视觉系统工程师，我希望配置多个相机的参数，以便采集高质量的观察图像用于策略推理。

#### Acceptance Criteria

1. THE RobotClient SHALL 配置三个 OpenCV 相机，分别对应 /dev/video2、/dev/video4、/dev/video6
2. THE RobotClient SHALL 设置每个相机的分辨率为 640x480
3. THE RobotClient SHALL 设置每个相机的帧率为 30 FPS
4. THE RobotClient SHALL 使用 MJPG 编码格式采集图像
5. THE RobotClient SHALL 将相机命名为 camera1、camera2、camera3，与策略模型期望的输入键匹配

### Requirement 7: 调试和监控功能

**User Story:** 作为系统调试人员，我希望能够可视化系统运行状态，以便调优异步推理参数。

#### Acceptance Criteria

1. WHERE 启用调试模式，THE RobotClient SHALL 实时显示动作队列大小
2. THE RobotClient SHALL 记录推理延迟和执行延迟的统计信息
3. WHERE 启用数据显示，THE RobotClient SHALL 在屏幕上显示相机图像和机械臂状态
4. THE System SHALL 在控制台输出连接状态和错误信息
5. WHEN 任务完成时，THE RobotClient SHALL 生成动作队列大小的可视化图表
