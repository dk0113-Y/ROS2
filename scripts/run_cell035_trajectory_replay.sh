#!/usr/bin/env bash
set -euo pipefail

ROS_REPO="${ROS_REPO:-/home/dk/ros2_repos/ROS2}"
ROS_WS="${ROS_WS:-$ROS_REPO/ros2_ws}"
ROS_DISTRO="${ROS_DISTRO:-humble}"

TRAJECTORY_CSV="${TRAJECTORY_CSV:-$ROS_REPO/assets/cell035/trajectories/cell035_oracle_trajectory.csv}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-$ROS_REPO/trajectory_replay_summary.json}"
TRAJECTORY_LOG_OUTPUT="${TRAJECTORY_LOG_OUTPUT:-$ROS_REPO/trajectory_replay_log.csv}"

CELL_SIZE="${CELL_SIZE:-0.35}"
ROWS="${ROWS:-40}"
COLS="${COLS:-60}"
WORLD_X="${WORLD_X:-21.0}"
WORLD_Y="${WORLD_Y:-14.0}"
START_INDEX="${START_INDEX:-0}"
MAX_WAYPOINTS="${MAX_WAYPOINTS:-0}"
LINEAR_SPEED="${LINEAR_SPEED:-0.10}"
TARGET_POS_TOL="${TARGET_POS_TOL:-0.07}"
ROTATE_KP="${ROTATE_KP:-2.0}"
ROTATE_MAX_W="${ROTATE_MAX_W:-2.0}"
ROTATE_MIN_W="${ROTATE_MIN_W:-0.20}"
ROTATE_TOL_DEG="${ROTATE_TOL_DEG:-4.0}"
ROTATE_SIM_TIMEOUT="${ROTATE_SIM_TIMEOUT:-12.0}"
DRIVE_SIM_TIMEOUT="${DRIVE_SIM_TIMEOUT:-12.0}"
ROTATE_WALL_TIMEOUT="${ROTATE_WALL_TIMEOUT:-60.0}"
DRIVE_WALL_TIMEOUT="${DRIVE_WALL_TIMEOUT:-60.0}"
MULTI_WALL_TIMEOUT="${MULTI_WALL_TIMEOUT:-300.0}"
WAYPOINT_PAUSE_SEC="${WAYPOINT_PAUSE_SEC:-0.3}"
STOP_REPEAT="${STOP_REPEAT:-10}"
CONTROL_DEBUG_PERIOD="${CONTROL_DEBUG_PERIOD:-1.0}"
DIAGNOSTIC_NO_MOTION="${DIAGNOSTIC_NO_MOTION:-false}"

echo "This script starts only the waypoint trajectory replay node."
echo "Start Gazebo, the robot, /odom, and a /cmd_vel subscriber first."
echo "For SLAM mapping, start slam_toolbox separately before or during replay."

ROS_SETUP="/opt/ros/$ROS_DISTRO/setup.bash"
if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ROS setup file not found: $ROS_SETUP" >&2
  exit 1
fi

set +u
source "$ROS_SETUP"
set -u

if [[ ! -d "$ROS_WS/src/drl_explore_bridge" ]]; then
  echo "ROS2 package not found: $ROS_WS/src/drl_explore_bridge" >&2
  exit 1
fi
if [[ ! -f "$TRAJECTORY_CSV" ]]; then
  echo "trajectory CSV not found: $TRAJECTORY_CSV" >&2
  exit 1
fi

cd "$ROS_WS"
colcon build --packages-select drl_explore_bridge --symlink-install

set +u
source install/setup.bash
set -u

ros2 run drl_explore_bridge drl_trajectory_replay_node --ros-args \
  -p trajectory_csv:="$TRAJECTORY_CSV" \
  -p cell_size:="$CELL_SIZE" \
  -p rows:="$ROWS" \
  -p cols:="$COLS" \
  -p world_x:="$WORLD_X" \
  -p world_y:="$WORLD_Y" \
  -p start_index:="$START_INDEX" \
  -p max_waypoints:="$MAX_WAYPOINTS" \
  -p linear_speed:="$LINEAR_SPEED" \
  -p target_pos_tol:="$TARGET_POS_TOL" \
  -p rotate_kp:="$ROTATE_KP" \
  -p rotate_max_w:="$ROTATE_MAX_W" \
  -p rotate_min_w:="$ROTATE_MIN_W" \
  -p rotate_tol_deg:="$ROTATE_TOL_DEG" \
  -p rotate_sim_timeout:="$ROTATE_SIM_TIMEOUT" \
  -p drive_sim_timeout:="$DRIVE_SIM_TIMEOUT" \
  -p rotate_wall_timeout:="$ROTATE_WALL_TIMEOUT" \
  -p drive_wall_timeout:="$DRIVE_WALL_TIMEOUT" \
  -p multi_wall_timeout:="$MULTI_WALL_TIMEOUT" \
  -p waypoint_pause_sec:="$WAYPOINT_PAUSE_SEC" \
  -p stop_repeat:="$STOP_REPEAT" \
  -p control_debug_period:="$CONTROL_DEBUG_PERIOD" \
  -p summary_output:="$SUMMARY_OUTPUT" \
  -p trajectory_log_output:="$TRAJECTORY_LOG_OUTPUT" \
  -p diagnostic_no_motion:="$DIAGNOSTIC_NO_MOTION"
