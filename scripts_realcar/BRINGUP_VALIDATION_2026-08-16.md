# Wheeltec ROS2 Bringup 验证记录（2026-08-16）

## 测试范围

- 主机：Wheeltec Raspberry Pi，Ubuntu 22.04，aarch64；
- ROS：Humble，`ros-humble-ros2cli 0.18.18`；
- 车型配置：`mini_4wd`；
- 雷达配置：`ls_N10Plus_uart`；
- 设备：`/dev/wheeltec_controller`、`/dev/wheeltec_lidar` 均存在；
- 安全约束：全程没有发布 `/cmd_vel`，没有执行运动测试。

Humble 的 `ros2` 顶层命令不支持 `--version`，实际执行会返回 `unrecognized arguments: --version`。本机通过 `ROS_DISTRO=humble`、已安装的 `ros2cli` Debian 包版本及 `ros2 doctor --report` 确认 ROS2 环境正常。

## Bringup 结论

Wheeltec 原有 `turn_on_wheeltec_robot.launch.py` 在本机因缺少 `joint_state_publisher` 无法整体启动；`wheeltec_sensors.launch.py` 又包含本任务不需要的相机。因此使用本仓库的 `bringup_realcar.launch.py` 复用下列官方子 launch：

- `base_serial.launch.py`；
- `wheeltec_ekf.launch.py`；
- `robot_mode_description.launch.py`；
- `wheeltec_lidar.launch.py`。

首次运行还发现 `robot_localization` 缺失，已安装 `ros-humble-robot-localization 3.5.4`。随后发现旧版 `diagnostic_updater 4.0.6` 缺少 EKF 所需共享库，升级到 `4.0.7` 后 `ekf_filter_node` 正常运行。

EKF 启动时仍报告无法创建 Wheeltec 外部配置中的 `/home/wheeltec/debug/file.txt`，但节点持续运行、发布 `/odom_combined` 和动态 TF。本任务没有修改外部 Wheeltec 配置。

## 节点检查

运行 `check_realcar_topics.sh` 实际发现：

```text
/base_to_camera
/base_to_gyro
/base_to_laser
/base_to_link
/base_to_radar
/ekf_filter_node
/robot_state_publisher
/wheeltec_robot
/x10/lslidar_driver_node
```

此外存在 launch 自身和 TF listener 的内部节点。没有启动 DRL、Nav2 或 exploration 节点。

## Topic 检查

| Topic | 类型 | 实际端点 | 实测数据 |
| --- | --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | 0 publisher，1 subscriber：`/wheeltec_robot` | 未发布命令 |
| `/odom` | `nav_msgs/msg/Odometry` | 1 publisher：`/wheeltec_robot`；1 subscriber：`/ekf_filter_node` | 约 20.07 Hz |
| `/scan` | `sensor_msgs/msg/LaserScan` | 1 publisher：`/x10/lslidar_driver_node` | 约 10.01 Hz |

`/scan --once` 摘要：

```text
frame_id: laser
angle_min: -3.1415927410125732
angle_max: 3.1415927410125732
angle_increment: 0.01163552887737751
range_min: 0.15000000596046448
range_max: 50.0
ranges length: 540
```

`/odom --once` 摘要：

```text
frame_id: odom_combined
child_frame_id: base_footprint
pose.position: [0.0, 0.0, 0.0]
twist.linear: [0.0, 0.0, 0.0]
twist.angular.z: 0.0
```

## TF 检查

`tf2_echo odom_combined base_link` 成功收到动态变换；测试期间平移约为 `[0.000, 0.000, 0.068]`。

`tf2_echo base_link laser` 成功收到静态变换：

```text
translation: [0.032, 0.000, 0.025]
rotation quaternion: [0.000, 0.000, 0.000, 1.000]
```

实际树形关系为：

```text
odom_combined -> base_footprint -> base_link
                              \-> laser
```

Wheeltec 已有 static transform publishers 提供 `base_footprint` 到 `base_link` 和 `laser`，无需新增静态变换。

## 构建与限制

`colcon build` 已成功构建 `drl_explore_bridge`。最终提交前再次构建，以最终命令结果为准。

本次只验证静止状态下真实消息持续发布、消息类型、frame 和 TF 可达性；未验证车轮运动后的里程计尺度、方向、雷达外参精度或紧急停止行为，这些必须在后续人工监护的运动测试任务中完成。
