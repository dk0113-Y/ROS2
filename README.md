# ROS2 / Gazebo DRL Exploration Integration

用于移动机器人自主探索算法的 ROS2/Gazebo 仿真迁移与接口验证仓库。项目将外部 `DRL-path-finding` 策略接入 ROS2 workspace，围绕 robot URDF、Gazebo world、`/scan`、`/odom`、`/cmd_vel`、轨迹重放和诊断流程验证算法接口在连续仿真环境中的可执行性。

> 本仓库是仿真集成与诊断工程，不包含训练代码、模型权重、物理机器人日志或真实环境部署结论。`oracle_los` 和轨迹重放均有明确的诊断边界，不能写成真实传感器闭环 DRL 部署。

## 目标与范围

本仓库关注从栅格 DRL 实验到 ROS2/Gazebo 仿真接口的迁移：

- 组织标准 `ros2_ws/src/<package>` workspace/package。
- 加载 Gazebo world 和 robot URDF。
- 订阅 `/scan` 与 `/odom`，构造策略需要的局部状态。
- 从外部 checkpoint 执行策略推理。
- 将离散栅格动作转换为 `/cmd_vel`。
- 对 observation bridge 做无运动和短程诊断。
- 重放导出的 oracle/ideal trajectory，验证 waypoint execution 与 SLAM mapping 流程。

训练、checkpoint 生成和正式栅格评估由外部 `DRL-path-finding` 仓库负责。

## 系统接口

```text
Gazebo world + robot URDF
  |-- /scan ----\
  |              -> observation bridge -> external DRL policy -> discrete action
  `-- /odom ----/                                           |
                                                             v
                                                         /cmd_vel
```

轨迹重放是独立路径：

```text
exported trajectory CSV + /odom
  -> waypoint controller
  -> /cmd_vel
  -> optional SLAM consumes /scan and publishes /map
```

轨迹重放节点不加载 checkpoint、不运行 DRL policy，也不使用 `/scan` 生成策略状态。

## 技术栈

- ROS2 Humble-compatible Python tooling
- `ament_python`
- `rclpy`
- `sensor_msgs/LaserScan`
- `nav_msgs/Odometry`
- `geometry_msgs/Twist`
- Gazebo world / URDF
- NumPy
- 外部 PyTorch DRL policy runtime
- Bash runbooks
- `colcon`

## 仓库结构

```text
.
|-- assets/cell035/
|   |-- grids/                        # cell035 true grid 与 metadata
|   |-- trajectories/                 # 导出的 trajectory CSV 占位目录
|   |-- urdf/                         # mini 4WD robot URDF
|   `-- worlds/                       # Gazebo world
|-- docs/
|   |-- CHECKPOINT_USAGE.md
|   |-- cell035_bridge_diagnostics.md
|   |-- standalone_gazebo_bridge.md
|   `-- trajectory_replay_slam_mapping.md
|-- experiments/cell035_baselines/    # 已记录的仿真输出摘要
|-- ros2_ws/src/drl_explore_bridge/
|   |-- drl_explore_bridge/           # ROS2 nodes
|   |-- test/
|   |-- package.xml
|   `-- setup.py
`-- scripts/
    |-- run_cell035_standalone_bridge.sh
    |-- run_cell035_local_snap_diagnostic.sh
    |-- run_cell035_scan_template_los_diagnostic.sh
    `-- run_cell035_trajectory_replay.sh
```

## ROS2 package

workspace 位于 `ros2_ws/`，package 名为 `drl_explore_bridge`。

| Node | 作用 |
|---|---|
| `drl_standalone_gazebo_bridge_node` | 主 bridge；订阅 `/scan`、`/odom`，维护累计地图，执行策略并发布 `/cmd_vel`。 |
| `drl_trajectory_replay_node` | 从 CSV 读取 waypoint，通过 `/odom` 闭环发布 `/cmd_vel`；不运行 DRL。 |
| `drl_policy_step_once_node` | 单步策略接口与 observation conversion 实验路径。 |
| `drl_policy_multi_step_node` | 多步 wrapper，包含 coverage/no-progress 停止逻辑。 |
| `drl_policy_probe_node` | 使用 checkpoint、true grid 与 ROS 输入检查策略动作。 |
| `policy_probe_node` | 最小 policy probe；输出动作信息，不发布 `/cmd_vel`。 |
| `scan_to_local_snap_node` | 将 `/scan`、`/odom` 转换为 ASCII local-grid snapshot。 |

当前没有 ROS2 launch files；`scripts/` 中的 Bash 脚本是 `cell035` 路径的显式 runbook。

## 环境准备

需要：

- Linux + ROS2 Humble 或兼容环境
- `colcon`
- Gazebo 与机器人控制插件
- 外部 `DRL-path-finding` checkout
- 本地 checkpoint
- 可发布 `/scan`、`/odom` 并订阅 `/cmd_vel` 的仿真机器人

配置路径：

```bash
export ROS_REPO="$PWD"
export ROS_WS="$ROS_REPO/ros2_ws"
export DRL_REPO="$HOME/drl_repos/DRL-path-finding"
export CHECKPOINT="$DRL_REPO/deploy_checkpoints/A_full_method_last.pt"
export TRUE_GRID="$ROS_REPO/assets/cell035/grids/random_train_like_seed20260513_true_grid.npy"
```

模型权重未提交，`CHECKPOINT` 必须指向用户本地文件。

## 构建

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

只构建当前 package：

```bash
colcon build --packages-select drl_explore_bridge --symlink-install
```

## 运行

启动 Gazebo world、robot、controller 后，先确认 topics：

```bash
ros2 topic list | grep -E '^/scan$|^/odom$|^/cmd_vel$'
ros2 topic hz /scan
ros2 topic hz /odom
```

运行 standalone bridge：

```bash
bash scripts/run_cell035_standalone_bridge.sh
```

常用覆盖参数：

```bash
SCAN_BRIDGE_MODE=scan_template_los \
MAX_STEPS=40 \
COVERAGE_GOAL=0.95 \
bash scripts/run_cell035_standalone_bridge.sh
```

## Observation bridge 诊断

无运动 local-snap 对比：

```bash
bash scripts/run_cell035_local_snap_diagnostic.sh
```

以 `scan_template_los` 作为当前 bridge mode：

```bash
bash scripts/run_cell035_scan_template_los_diagnostic.sh
```

推荐按以下顺序扩大验证范围：

1. `diagnostic_no_motion=true`
2. `MAX_STEPS=5`
3. `MAX_STEPS=40`
4. `MAX_STEPS=300` 或 `400`

详细指标、ASCII map 符号和 mismatch 字段见
[`docs/cell035_bridge_diagnostics.md`](docs/cell035_bridge_diagnostics.md)。

当前 observation modes：

- `oracle_los`：读取 true grid，并复用训练 observation model；仅作为对齐上界。
- `ray_project`：按 LaserScan beam 投影 local grid。
- `los_compatible`：按目标 cell 查询 LaserScan 的近似 LOS。
- `scan_template_los`：以训练侧 ray templates 为外层结构的 LaserScan 近似。

后三种仍是 Gazebo LaserScan observation adapters，不是物理 LiDAR 感知验证。

## 轨迹重放

默认输入：

```text
assets/cell035/trajectories/cell035_oracle_trajectory.csv
```

该 CSV 不在当前仓库中，需从 `DRL-path-finding` 导出后再运行：

```bash
bash scripts/run_cell035_trajectory_replay.sh
```

默认输出：

```text
trajectory_replay_summary.json
trajectory_replay_log.csv
```

验收字段包括：

- `stop_reason`
- `total_waypoints`
- `reached_waypoints`
- `reached_ratio`
- `failed_waypoint_idx`
- `final_xy`

详细导出和 SLAM 流程见
[`docs/trajectory_replay_slam_mapping.md`](docs/trajectory_replay_slam_mapping.md)。

## 已记录实验

`experiments/cell035_baselines/` 保存了一次 `cell035` Gazebo 仿真路径的文本记录：

| 项目 | 值 |
|---|---|
| World | `random_train_like_seed20260513_cell035.world` |
| Grid | `random_train_like_seed20260513_true_grid.npy` |
| Robot URDF | `mini_4wd_robot_with_laser_drive_cell035.urdf` |
| Cell size | `0.35 m` |
| Grid size | `40 x 60` |
| Scan radius | `10` cells |
| Max steps | `400` |
| Coverage goal | `0.95` |
| Recorded result | `best_coverage=0.9504`, `steps=207` |

同一记录中，旧 checkpoint baseline 为 `best_coverage=0.8751`，并在 step 319 因 no-progress limit 停止。

结果边界：

- 以上数值来自已提交的历史文本记录，本次 README 修改没有重跑 Gazebo。
- 该结果属于指定 `cell035` 仿真 world、robot 和参数组合，不是跨地图 benchmark。
- 文档中的 pure-grid / Gazebo `oracle_los` 对齐结果依赖 true grid，只能作为接口诊断。
- 不应将这些结果解释为物理机器人、真实 LiDAR 或未知真实室内环境性能。

## 当前状态与限制

已实现：

- ROS2 workspace/package 与 console entry points
- Gazebo world、URDF 和 cell035 grid assets
- `/scan`、`/odom`、`/cmd_vel` bridge
- 多种 LaserScan-to-local-grid 诊断模式
- policy probe、单步/多步 bridge 和 trajectory replay nodes
- Bash runbooks 与历史仿真摘要

仍需验证或补充：

- checkpoint 和 training code 是外部依赖
- trajectory CSV 需要单独导出
- `scan_template_los` 尚未证明与训练 observation 完全一致
- 未提供物理机器人或真实传感器验证
- 未提供 ROS2 launch files
- URDF mesh path 可能依赖本地 `wheeltec_robot_urdf` assets
- trajectory replay 只证明 waypoint execution 路径，不证明 closed-loop DRL
- SLAM map 保存和碰撞检查需要在实际 ROS2/Gazebo 环境中人工确认

## 运行产物管理

默认不要提交：

- checkpoint
- trajectory runtime logs
- `trajectory_replay_summary.json`
- `trajectory_replay_log.csv`
- SLAM maps
- Gazebo/ROS bag 大文件

若公开实验产物，应同时保留 world、URDF、checkpoint provenance、参数、topic 配置和日志解释边界。
