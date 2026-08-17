# ROS2 实机迁移实验代码

本仓库同时包含两类内容：

- Gazebo 仿真、仿真桥接与诊断代码；
- DRL exploration 从 Gazebo 迁移到 Wheeltec ROS2 实机的实验代码。

本目录及 `drl_explore_bridge` 中以 `realcar_` 开头的节点属于实机迁移实验范围。实机整理不改变 Gazebo 环境、URDF、仿真逻辑或训练算法。

## 实机基础 bringup

本机 Wheeltec 工作空间已经提供底盘、雷达和 TF 子 launch。原有的 `wheeltec_sensors.launch.py` 还会启动相机，原有的 `turn_on_wheeltec_robot.launch.py` 则依赖当前系统未安装的 `joint_state_publisher`。仓库因此提供最小组合入口，只复用官方子 launch，不复制驱动实现：

```bash
export REALCAR_BASE_WS_SETUP=/home/wheeltec/wheeltec_ros2_ws/install/setup.bash
source /home/wheeltec/git_repos/ROS2/scripts_realcar/realcar_env.sh
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
  -> 记录 post-inference sensor barrier
  -> 等待 barrier 后新收到且 receive/header 时间均有效的 /scan、/odom
  -> 使用刷新后的 /scan 重新执行 LiDAR safety gate
  -> 确定最终 executed action（可能为安全 fallback）
  -> 使用刷新后的 odom 与最终 action 生成固定 odom 目标
  -> execute=true 时执行 safe_rotate_drive，等待动作完成并发布零速度
  -> 再等待一组新的 /scan、/odom
  -> 根据累计 odom 位移生成下一 DRL 网格状态
  -> 进入下一步，最多三步
```

runner 的 `/scan`、`/odom` 订阅使用 `KEEP_LAST depth=1`，避免 policy
inference 阻塞 spin 时保留约一秒的历史队列。barrier 同时检查 callback 序列号和
monotonic receive time；刷新后的样本仍必须通过默认 `0.5s` 的 receive/header
freshness 检查，超时不会通过放宽阈值处理。逐步 JSON 和摘要日志会记录 model
load、policy state build、policy inference、pre-motion refresh 耗时，以及两类传感器
的序列号和 receive/header age。

`step_distance` 与训练网格 `cell_size=0.35m` 不一致时会明确警告。runner 不再按所选动作盲目把抽象状态移动一格，而是用相对实验起点的累计 odom 位移量化到原有 DRL 行列坐标；小于半个网格的位移会保留在同一抽象单元。因此该机制只用于有限步迁移实验，仍存在连续坐标到离散网格的量化误差。

只观察三次决策、绝不发送非零速度时：

```bash
source /home/wheeltec/git_repos/ROS2/scripts_realcar/realcar_env.sh
setup_realcar_environment
ros2 run drl_explore_bridge realcar_policy_safe_runner_node --ros-args \
  -p checkpoint_path:="$DRL_CHECKPOINT_PATH"
```

节点启动时会明确打印 `execute=false`。只有人工显式传入 `-p execute:=true` 才允许非零 `/cmd_vel`；实机执行还必须满足现场监护、急停、空旷区域及每一步开始前的传感器新鲜度检查。

每次运行都会写入结构化 JSON，参数 `result_file_path` 默认是 `~/realcar_logs/`。顶层记录 `experiment_id`、请求/完成步数、总耗时、成功状态和失败原因；`steps` 中逐步记录 `step_id`、`action_idx`、`motion_mode`、起点/目标/终点位姿、实际位移、耗时、成功状态及失败原因。实验日志不得提交到本仓库。

Round 7 的三步真实策略闭环已经在真车上完成一次验证：三步均成功，最终以
`max_steps_reached` 结束。该节点继续作为冻结回归基线，默认
`step_distance=0.10m`、`max_steps=3`、`execute=false`；Round 8 不改变这些默认值。

## Round 8A 连续探索 runner

`realcar_policy_continuous_runner_node` 是独立的、具有硬步数和运行时间上限的第一版
连续探索 runner。它默认 `execute=false`，直接运行时不会发送非零 `/cmd_vel`。
Round 8 使用一个物理尺度来源：`cell_size=0.35m`，并要求
`step_distance==cell_size`；不一致时节点拒绝启动。训练动作是标准 8 邻域网格中心转移，
所以 cardinal 动作的目标距离是 `0.35m`，diagonal 动作的 x/y 分量各为
`0.35m`，欧氏目标距离是 `sqrt(2)*0.35≈0.495m`。这不是把 diagonal 归一化为
固定路径长度。

每个决策循环为：

```text
fresh observation -> cumulative belief -> policy inference
  -> post-inference fresh scan/odom barrier
  -> refreshed-scan distance-aware gate
  -> final action + refreshed-odom grid-center target
  -> optional motion with drive-phase dynamic obstacle stop
  -> odom-derived actual state -> belief-side termination
```

真实世界完成判据只使用累计 belief：known cells、frontier 和其增长历史。传给
`CumulativeBeliefMap` 的 120x120 全零数组只是构造兼容占位；runner 不读取它的
true-map coverage。累计 belief 本身可动态扩展，Round 8 的 odom 派生 world-grid state
不再受该占位数组的 120x120 边界限制。策略动作得到的 expected grid state 只写入诊断；
实际 policy state 始终由相对 episode 起点的累计 odom 位移量化，二者不一致时记录
`grid_transition_match=false`，不会强制移动抽象状态。

continuous runner 的部署安全层使用显式圆形 safety footprint：

```text
nominal_min_corridor_width_m = 0.40m
footprint_radius_m = nominal_min_corridor_width_m / 2 = 0.20m
longitudinal_extra_margin_m = 0.05m
pre_motion_center_line_length = target_distance + 0.05m
swept_volume = center-line segment (+) radius-0.20m circle
```

该 `0.20m` 是部署安全包络，不是真实机器人半径；实车约 `0.20m x 0.30m`，半对角线
约 `0.1803m`。有效 LaserScan hit 会先按 `laser` 相对 `base_footprint` 的平移和 yaw
转换到 base frame，再投影到候选动作坐标系。点到中心线段的最短距离小于 `0.20m`
才会阻止动作；恰好位于 `0.20m` 名义边界的点允许通过（仅使用 `1e-9m` 数值容差）。
因此 `0.40m` 直通道的平行侧墙不会再仅因进入动作方向 `+/-22.5deg` 扇区而误拒，
同时 capsule 前缘仍为 cardinal `0.35+0.05+0.20=0.60m`、diagonal
`sqrt(2)*0.35+0.05+0.20≈0.745m`。

当前 Wheeltec 默认配置为 `mini_4wd`。bringup 的 `robot_model.yaml` 发布
`base_footprint -> base_link` 平面平移和 yaw 均为 0，发布
`base_footprint -> laser` 为 `x=0.03163m, y=0.00009m, yaw=0`；continuous runner
以这组仓库现场配置核查值作为可覆盖参数默认值。URDF 中 `laser_link` 的 yaw 为 pi，
但 `/scan` 使用 bringup 单独发布的 `laser` frame，二者不可混用。

兼容参数 `motion_clearance_margin=0.25m` 不再作为独立 sector 判据；启动时要求它等于
`footprint_radius_m + longitudinal_extra_margin_m`，防止两套安全语义冲突。直行阶段每
收到一组新 scan/odom 后，用中心线长度
`dynamic_stop_distance - footprint_radius_m = 0.05m` 的短 capsule 检查，保持前缘
`0.25m`。侧向进入 `0.20m` 或正前方进入
`0.25m` 都会先发布零速度并以 `dynamic_obstacle_stop` 中止；合法 `0.40m` 通道边界
不会仅因侧墙存在而停止。进入 rotate phase 前还会用长度为 0 的全周圆形 footprint
再检查一次，footprint 内有有效障碍点时 fail-stop，禁止开始旋转。

正常完成只有满足最小决策步数、最小 known-area 后的 `frontier_exhausted`。此外还会
检测 belief stagnation、重复 state 且信息不增长的 deadlock，并以 `max_steps`、
`max_runtime_sec`、sensor/motion failure、有限 no-safe-action retry 和 operator interrupt
作为有界 failsafe。所有窗口和阈值均可配置，不代表已经科学标定。

只读软件检查示例（建议显式限制为 5 个 cycle）：

```bash
cd /home/wheeltec/git_repos/ROS2
export REALCAR_BASE_WS_SETUP=/home/wheeltec/wheeltec_ros2_ws/install/setup.bash
source /home/wheeltec/git_repos/ROS2/scripts_realcar/realcar_env.sh
setup_realcar_environment
export DRL_CHECKPOINT_PATH=/home/wheeltec/drl_repos/DRL-path-finding/deploy_checkpoints/A_full_method_last.pt
ros2 run drl_explore_bridge realcar_policy_continuous_runner_node --ros-args \
  -p checkpoint_path:="$DRL_CHECKPOINT_PATH" \
  -p execute:=false \
  -p max_steps:=5
```

`execute=false` 只验证 loop、belief、inference、sensor barrier、termination plumbing 和
JSON logging；静止机器人不会提供真实运动后的状态变化，因此不能证明连续自主探索。
Round 8 的 0.35m cardinal step、约 0.495m diagonal step、动态停车阈值和净空余量均
尚未完成真车验证或标定。

`scripts_realcar/realcar_env.sh` 是 Git 跟踪文件。如果当前 checkout 使用 sparse-checkout
且未包含 `scripts_realcar`，相对路径 source 会显示 “No such file or directory”，即使
commit 中存在该文件。应先在仓库根目录确认文件已 materialize，并优先使用上面的绝对
路径；不要把 shell 中残留的 ROS 环境误当作脚本已成功执行。
