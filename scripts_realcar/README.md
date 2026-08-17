# ROS2 实机迁移实验代码

本仓库同时包含两类内容：

- Gazebo 仿真、仿真桥接与诊断代码；
- DRL exploration 从 Gazebo 迁移到 Wheeltec ROS2 实机的实验代码。

本目录及 `drl_explore_bridge` 中以 `realcar_` 开头的节点属于实机迁移实验范围。实机整理不改变 Gazebo 环境、URDF、仿真逻辑或训练算法。

## 实机基础 bringup

本机 Wheeltec 工作空间已经提供底盘、雷达和 TF 子 launch。原有的 `wheeltec_sensors.launch.py` 还会启动相机，原有的 `turn_on_wheeltec_robot.launch.py` 则依赖当前系统未安装的 `joint_state_publisher`。仓库因此提供最小组合入口，只复用官方子 launch，不复制驱动实现：

```bash
export REALCAR_BASE_WS_SETUP=/home/wheeltec/wheeltec_ros2_ws/install/setup.bash
source scripts_realcar/realcar_env.sh
setup_realcar_environment
ros2 launch drl_explore_bridge bringup_realcar.launch.py
```

该 launch 只启动：

- `/wheeltec_robot` 底盘节点（可执行文件 `wheeltec_robot_node`）；
- `ekf_filter_node` 里程计滤波及动态 TF；
- `robot_state_publisher` 和 Wheeltec 已有 static transform publishers；
- 当前配置对应的雷达驱动。

不会启动 DRL、Nav2、exploration、相机或任何 `/cmd_vel` publisher。

当前 `/home/wheeltec/wheeltec_ros2_ws/src/turn_on_wheeltec_robot/config/wheeltec_param.yaml` 的实机配置为 `car_mode: mini_4wd`、`lidar_type: ls_N10Plus_uart`。基础接口预期如下：

| 接口 | 类型 | 来源/使用者 | 坐标系或说明 |
| --- | --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | `/wheeltec_robot` 订阅 | bringup 不应存在 publisher |
| `/odom` | `nav_msgs/msg/Odometry` | `/wheeltec_robot` 发布 | `odom_combined` → `base_footprint` |
| `/scan` | `sensor_msgs/msg/LaserScan` | `/x10/lslidar_driver_node` 发布 | `frame_id: laser`，配置频率 10 Hz |

实机 TF 并不是字面上的 `odom -> base_link -> laser` 单链，而是：

```text
odom_combined -> base_footprint -> base_link
                              \-> laser
```

因此应检查 `odom_combined -> base_link` 和 `base_link -> laser`。`robot_mode_description.launch.py` 已提供 `base_footprint` 到 `base_link`、`laser` 的静态变换，不需要额外添加 `static_transform_publisher`。

bringup 启动后，在另一个终端运行只读检查：

```bash
export REALCAR_BASE_WS_SETUP=/home/wheeltec/wheeltec_ros2_ws/install/setup.bash
./scripts_realcar/check_realcar_topics.sh
```

脚本只执行 node/topic/TF 查询和短时频率采样，从不发布 `/cmd_vel`。

本机实际检查证据见 [`BRINGUP_VALIDATION_2026-08-16.md`](BRINGUP_VALIDATION_2026-08-16.md)。

## 安全边界

- `realcar_policy_dryrun_node` 只订阅 `/scan`、`/odom` 并执行策略推理，不创建 `/cmd_vel` publisher。
- `realcar_policy_safe_runner_node` 默认 `execute=false`。只有显式设置 `execute:=true` 才可能发送非零速度。
- `realcar_step_once_safe_node` **仅用于人工现场监护下的单步测试**，默认 `execute=false`；不得作为连续自主运行节点使用。
- 两个可能发送速度的节点在每次发送非零 `/cmd_vel` 前都会检查 `/scan` 与 `/odom`：消息必须持续更新，header 时间戳必须有效且未过期。任一传感器停止更新或时间戳异常时，节点停止运动并发送零速度。
- 非零速度在发布入口再次限制为 `|linear.x| <= 0.06 m/s`、`|angular.z| <= 0.40 rad/s`。

这些检查不能替代现场急停、架空轮测试、空旷场地、人工观察和底盘自身的安全保护。本仓库整理过程不包含任何真实底盘连接或运动测试。

## 路径配置

节点不再包含用户目录的固定 checkpoint 路径。运行策略节点前必须通过 ROS 参数 `checkpoint_path` 指定，或设置：

```bash
export DRL_CHECKPOINT_PATH=/absolute/path/to/A_full_method_last.pt
```

两个只读辅助脚本共用 `realcar_env.sh`。默认加载当前仓库的 `ros2_ws/install/setup.bash`；如需加载底盘工作空间或不同 ROS/桥接工作空间，可统一覆盖：

```bash
export REALCAR_BASE_WS_SETUP=/absolute/path/to/base_ws/install/setup.bash
export REALCAR_BRIDGE_WS_SETUP=/absolute/path/to/bridge_ws/install/setup.bash
export REALCAR_ROS_SETUP=/opt/ros/humble/setup.bash
```

## 只读检查

构建并设置 `DRL_CHECKPOINT_PATH` 后，可以使用：

```bash
./scripts_realcar/run_realcar_local_snap_readonly.sh
./scripts_realcar/run_realcar_policy_probe_readonly.sh
```

两个脚本都不启动运动节点。`realcar_policy_dryrun_node` 还会在启动时拒绝已有 `/cmd_vel` publisher 的环境。

时间戳阈值可通过 `scan_timeout_sec`、`odom_timeout_sec` 和 `sensor_future_tolerance_sec` 参数调整；默认分别为 `0.5`、`0.5` 和 `0.1` 秒。只有确认实机消息频率和时钟同步后才应调整。

## 真实机器人单步实验工具

`run_realcar_step_once_experiment.sh` 固定了 Round 1 单步实验入口。脚本会加载 ROS、底盘工作空间（如已配置）和本仓库工作空间，确认 `drl_explore_bridge`、`/scan`、`/odom` 存在，然后启动 `realcar_step_once_safe_node`。

默认是只观察模式，不发送非零速度：

```bash
export REALCAR_BASE_WS_SETUP=/home/wheeltec/wheeltec_ros2_ws/install/setup.bash
./scripts_realcar/run_realcar_step_once_experiment.sh
```

只有在现场安全条件、人工监护和急停均已准备好后，才可显式加入 `--execute`：

```bash
./scripts_realcar/run_realcar_step_once_experiment.sh --execute \
  -p action_idx:=2 \
  -p step_distance:=0.18
```

脚本拒绝通过附加 ROS 参数设置 `execute`，避免绕过显式的 `--execute` 开关。其余节点参数可以在脚本参数末尾使用 `-p name:=value` 传入。

节点在运动前打印 `step_experiment_start`，结束或失败后打印 `step_experiment_result`，包括动作、起止位姿、目标方向、目标距离、实际位移、耗时和失败原因。每次结果同时保存为 JSON。参数 `result_file_path` 默认是 `~/realcar_logs/`，目录会自动创建，默认文件名形如 `realcar_step_YYYYMMDD_HHMMSS.json`；也可传入一个明确的 `.json` 文件路径。已有明确文件不会被覆盖。

`execute=false` 的观察记录会以 `success=false`、`failure_reason=execution_disabled` 保存，表示没有进行运动验证。`execute=true` 仅代表允许该节点执行一次既有的低速旋转和直行控制，并不构成连续自主探索或真实运动闭环验证。

## Round 2 动作适配与运动模式

`realcar_action_adapter.py` 不改变 `ACTIONS_8`，只把既有网格增量转换为固定的 odom 目标。DRL 行、列与 odom 轴的对应关系为：

| action_idx | 方向 | odom 方向 |
| --- | --- | --- |
| 0 | N | `+y` |
| 1 | NE | `+x, +y` |
| 2 | E | `+x` |
| 3 | SE | `+x, -y` |
| 4 | S | `-y` |
| 5 | SW | `-x, -y` |
| 6 | W | `-x` |
| 7 | NW | `-x, +y` |

单步节点的 `motion_mode` 支持：

- `safe_rotate_drive`：默认模式，保留先原地旋转、再低速直行的安全验证流程；
- `adapter_drive`：实验模式，在一次闭环中根据目标点方位同时调节 `linear.x` 和 `angular.z`。偏航误差超过 `adapter_heading_stop_deg` 时线速度保持为零。

90 度及更大转角不再通过提高默认角速度来处理。`rotate_max_w` 仍默认为 `0.35 rad/s`，`rotate_wall_timeout` 默认由 6 秒延长到 40 秒；同时增加旋转进展看门狗，默认要求每 5 秒至少减少 2 度偏航误差，卡住时会提前失败。到达 `rotate_tol_deg` 后先连续确认 5 次并发布零角速度。可配置参数包括 `rotate_kp`、`rotate_min_w`、`rotate_max_w`、`rotate_tol_deg`、`rotate_wall_timeout`、`rotate_progress_timeout`、`rotate_min_progress_deg`、`adapter_angular_kp`、`adapter_heading_stop_deg` 和 `adapter_drive_wall_timeout`。

只观察新的 odom 动作解释时仍保持默认安全开关：

```bash
./scripts_realcar/run_realcar_step_once_experiment.sh \
  -p action_idx:=0 \
  -p motion_mode:=safe_rotate_drive
```

JSON 和终端结果增加 `motion_mode`、`target_pose`、`rotate_start_yaw`、`target_yaw`、`final_yaw`、`rotate_success` 和 `rotate_error`。

## Round 3 安全三步执行框架

`realcar_policy_safe_runner_node` 是最多三步的受限实验 runner，不是无限循环或完整自主探索节点。其默认参数为 `max_steps=3`、`execute=false`、`motion_mode=safe_rotate_drive` 和 `allowed_actions_mode=all8`。`max_steps` 只能设为 1～3；任何一步出现传感器超时、动作过滤、雷达门控、旋转/直行失败或其他异常，都会先发布零速度并取消剩余步骤。

每一步按以下顺序处理：

```text
获取一组新的且时间戳有效的 /scan、/odom
  -> 更新累计观测并执行一次 policy inference
  -> realcar_action_adapter 生成固定 odom 目标
  -> execute=true 时执行 safe_rotate_drive，等待动作完成并发布零速度
  -> 再等待一组新的 /scan、/odom
  -> 根据累计 odom 位移生成下一 DRL 网格状态
  -> 进入下一步，最多三步
```

`step_distance` 与训练网格 `cell_size=0.35m` 不一致时会明确警告。runner 不再按所选动作盲目把抽象状态移动一格，而是用相对实验起点的累计 odom 位移量化到原有 DRL 行列坐标；小于半个网格的位移会保留在同一抽象单元。因此该机制只用于有限步迁移实验，仍存在连续坐标到离散网格的量化误差。

只观察三次决策、绝不发送非零速度时：

```bash
source scripts_realcar/realcar_env.sh
setup_realcar_environment
ros2 run drl_explore_bridge realcar_policy_safe_runner_node --ros-args \
  -p checkpoint_path:="$DRL_CHECKPOINT_PATH"
```

节点启动时会明确打印 `execute=false`。只有人工显式传入 `-p execute:=true` 才允许非零 `/cmd_vel`；实机执行还必须满足现场监护、急停、空旷区域及每一步开始前的传感器新鲜度检查。

每次运行都会写入结构化 JSON，参数 `result_file_path` 默认是 `~/realcar_logs/`。顶层记录 `experiment_id`、请求/完成步数、总耗时、成功状态和失败原因；`steps` 中逐步记录 `step_id`、`action_idx`、`motion_mode`、起点/目标/终点位姿、实际位移、耗时、成功状态及失败原因。实验日志不得提交到本仓库。
