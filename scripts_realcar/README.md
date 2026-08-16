# ROS2 实机迁移实验代码

本仓库同时包含两类内容：

- Gazebo 仿真、仿真桥接与诊断代码；
- DRL exploration 从 Gazebo 迁移到 Wheeltec ROS2 实机的实验代码。

本目录及 `drl_explore_bridge` 中以 `realcar_` 开头的节点属于实机迁移实验范围。实机整理不改变 Gazebo 环境、URDF、仿真逻辑或训练算法。

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
