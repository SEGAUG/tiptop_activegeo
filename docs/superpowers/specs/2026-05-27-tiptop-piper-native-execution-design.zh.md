# TiPToP Piper 原生执行设计

## 目标

把左侧 Piper 作为 TiPToP 的原生单臂 embodiment 接入，让 TiPToP 自己完成感知、抓取候选、IK、碰撞检查、轨迹规划和执行调度。pika 官方流程只作为调试参考，不作为 TiPToP 到真机的运动系统。

当前目标是先在新位置下安全地跑通单臂、腕部相机、慢速执行链路，最终让左侧 Piper 抓取桌面瓶子并放下。

## 当前已知状态

- 机械臂主机：`wzy` (`10.31.3.54`)
- 目标机械臂：照片左侧 Piper
- 当前可用 CAN：`can0`/`can1`，新位置下 `can_left` 不存在
- 当前只读 bridge：`http://10.31.3.54:8766`
- 当前 bridge CAN：`can0`
- 当前 wrist camera：RealSense `243722072079`
- TiPToP 当前配置：`robot.type = piper`，`cameras.hand.type = piper_bridge`
- 新位置腕部相机验证：ChArUco `detected=true`，`aruco_markers=44`，`charuco_corners=70`
- 当前运动 gate：关闭，`motion_allowed=false`

TiPToP 当前代码路径是单臂、单 wrist/hand camera 主链路。双臂不是本设计范围。

## 非目标

- 不接入 pika 作为执行系统。
- 不做双臂协同。
- 不在未完成 dry-run、轨迹检查和现场确认前执行真实抓取。
- 不让 Gemini/SAM/M2T2 输出直接变成真机动作；所有动作必须经过 TiPToP/curobo 规划和安全检查。

## 总体链路

```text
wrist RGB-D + Piper joint state
-> world_from_cam = FK(q) @ ee_from_cam
-> TiPToP perception
-> M2T2 grasp candidates
-> cuTAMP task planning
-> cuRobo IK / motion generation / collision checking
-> Piper joint trajectory
-> Piper native executor
-> low-level Piper transport bridge
```

这里的 bridge 只负责把已经规划好的 joint trajectory 或 gripper command 安全地下发到 Piper。它不负责 IK，不负责选择路径，也不负责理解任务。

## 核心组件

### 1. Piper Robot Embodiment

TiPToP 侧需要稳定支持 `robot.type = piper`：

- `PiperRobotClient.get_joint_positions()`
- `PiperRobotClient.execute_joint_path(...)`
- `PiperRobotClient.open_gripper()`
- `PiperRobotClient.close_gripper()`
- `get_robot_rerun("piper")`
- `motion_planning.py` 路由到 Piper cuRobo/cutamp asset

执行接口必须默认 dry-run 或 motion disabled。真实运动必须显式打开 gate。

### 2. Piper cuRobo/cuTAMP Asset

Piper 要成为 TiPToP 原生机器人，必须有可信的规划资产：

- URDF 与真机版本一致
- mesh 路径可被 cuRobo 正确加载
- joint names 和真机反馈顺序一致：`joint1` 到 `joint6`
- joint limits、velocity limits、acceleration limits 保守
- collision spheres 或 collision mesh 覆盖机身、前臂、腕部、夹爪
- `base_link`、`ee_link`、`tool_from_ee` 明确
- retract/home/capture joint config 位于安全工作区

第一版可以先使用保守 collision 模型，但必须通过可视化确认轨迹不会穿过桌面、另一只机械臂、玻璃挡板、笔记本和显示器。

### 3. Wrist Camera Calibration

TiPToP 真机链路依赖：

```text
world_from_cam = world_from_ee @ ee_from_cam
```

因此必须重新验证或标定 `ee_from_cam`：

- ChArUco board 放桌面，腕部相机完整看到足够多角点
- 采集多个末端姿态
- 使用 `calibrate-wrist-cam` 或等价脚本保存相机到末端外参
- 用 `viz-calibration` 检查点云是否落在桌面和物体真实位置

如果相机或夹爪安装没有变化，只移动整台机械臂理论上不改变 `ee_from_cam`；但当前现场移动较多，仍需做一次验证。

### 4. Workspace 和碰撞边界

新位置下旧 workspace 假设失效。需要重新建模：

- 桌面平面和边界
- 玻璃挡板
- 左侧 Piper 自身底座附近禁区
- 右侧 Piper 或其它机械臂禁区
- 笔记本、显示器、线缆的大致禁区
- 瓶子所在可操作区域

第一版 workspace 应该保守，宁愿少规划一些可达动作，也不能让轨迹贴近设备。

### 5. Native Executor

Native executor 接收 TiPToP/curobo 输出的 joint trajectory，然后下发到低层 Piper transport：

```text
input:  N x 6 joint positions, optional gripper events
output: real robot execution result
```

它必须做安全检查：

- 当前 joint state 新鲜
- 第一帧接近当前关节状态
- 每段 waypoint 最大关节变化不超过阈值
- 每段时间间隔不低于阈值
- 总轨迹在 joint limits 内
- gripper command 在合法开合范围内
- 默认 dry-run，仅保存轨迹和报告
- 真实执行前打印摘要并要求现场确认

推荐初始限制：

```text
max_initial_error_rad = 1 deg
max_waypoint_step_rad = 1 deg
min_waypoint_dt_s = 3-5 s
speed_percent <= 5
```

真实抓取前只允许先做 1-2 cm 等效末端位移的极慢验证。

## 数据流

1. `get_joint_positions()` 读取 Piper 当前 6 轴关节。
2. cuRobo FK 得到 `world_from_ee`。
3. 读取腕部 RGB-D。
4. 使用 `ee_from_cam` 得到 `world_from_cam`。
5. TiPToP 感知生成物体点云、mask、grasp candidates。
6. cuTAMP 选择任务 skeleton。
7. cuRobo 生成 joint trajectory。
8. executor 做 dry-run 安全报告。
9. 用户确认后，executor 才允许极慢下发。

## 验证阶段

### P0 只读健康检查

- bridge health OK
- wrist RGB-D 新鲜
- joint state 可读
- ChArUco 可检测
- motion gate 关闭

当前已通过。

### P1 Wrist Calibration 验证

- 至少采集多姿态 ChArUco 样本
- 生成或验证 `ee_from_cam`
- `viz-calibration` 中点云和桌面/瓶子位置一致

未通过 P1 前，不执行抓瓶子。

### P2 Piper Asset 验证

- cuRobo 能加载 Piper asset
- FK 与远端/官方反馈的末端姿态趋势一致
- IK 能解当前附近的小范围目标
- collision 可视化合理

### P3 TiPToP Planning Dry-run

- 用腕部相机真实图像跑 perception
- M2T2 生成瓶子 grasp
- TiPToP 生成 plan
- 保存 plan、轨迹和 rerun 可视化
- 不执行真机

### P4 极慢 Joint Trajectory Smoke Test

- 用户现场确认
- 不拿瓶子
- 从当前位姿执行极小 joint-space 轨迹
- 观察是否慢、稳、方向符合预期
- 执行后恢复 motion gate 关闭

### P5 抓瓶子实验

- 现场清空周围设备
- 急停/断电可触达
- 轨迹可视化审核通过
- 先执行 approach，不闭合夹爪
- 再执行 close/lift/place 全流程

## 风险和缓解

- 外参错误：通过 ChArUco 多姿态和 `viz-calibration` 验证。
- URDF/mesh 不一致：先 FK/IK/collision dry-run，再微动验证。
- 轨迹绕远：限制 workspace、降低速度、先只允许小范围目标。
- 夹爪和 arm 混合控制错误：夹爪走独立 SDK/bridge command，arm trajectory 固定 6 轴。
- 相机视野不足：按官方单视角要求，所有物体必须在 wrist camera 视野内。
- 现场设备碰撞：把桌面设备建成保守障碍物，真实执行前移走无关物体。

## 下一步

1. 重新运行 wrist calibration 验证流程，得到可信 `ee_from_cam`。
2. 检查并修正 Piper cuRobo asset：link、joint、limits、collision、ee。
3. 运行 Piper asset debug，确认 FK/IK/collision。
4. 运行 TiPToP perception/planning dry-run，保存 plan。
5. 写 native executor 的 dry-run 报告和真机 gate。
6. 现场确认后做极慢微动 smoke test。
