#!/usr/bin/env bash
set -euo pipefail

ROS_REPO="${ROS_REPO:-/home/dk/ros2_repos/ROS2}"
ROS_WS="${ROS_WS:-$ROS_REPO/ros2_ws}"
DRL_REPO="${DRL_REPO:-/home/dk/drl_repos/DRL-path-finding}"
ROS_DISTRO="${ROS_DISTRO:-humble}"

CHECKPOINT="${CHECKPOINT:-/home/dk/drl_repos/DRL-path-finding/deploy_checkpoints/A_full_method_last.pt}"
TRUE_GRID="${TRUE_GRID:-/home/dk/ros2_repos/ROS2/assets/cell035/grids/random_train_like_seed20260513_true_grid.npy}"
SCAN_BRIDGE_MODE="${SCAN_BRIDGE_MODE:-ray_project}"

CELL_SIZE="${CELL_SIZE:-0.35}"
ROWS="${ROWS:-40}"
COLS="${COLS:-60}"
WORLD_X="${WORLD_X:-21.0}"
WORLD_Y="${WORLD_Y:-14.0}"
SCAN_RADIUS_CELLS="${SCAN_RADIUS_CELLS:-10}"
MULTI_WALL_TIMEOUT="${MULTI_WALL_TIMEOUT:-300.0}"

echo "This script runs the local_snap alignment diagnostic only."
echo "Start the Gazebo world, robot, /scan, and /odom first."
echo "diagnostic_no_motion=true: the bridge will not execute a policy action or publish /cmd_vel."

ROS_SETUP="/opt/ros/$ROS_DISTRO/setup.bash"
if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ROS setup file not found: $ROS_SETUP" >&2
  exit 1
fi
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
source "$ROS_SETUP"

if [[ ! -d "$ROS_WS/src/drl_explore_bridge" ]]; then
  echo "ROS2 package not found: $ROS_WS/src/drl_explore_bridge" >&2
  exit 1
fi
if [[ ! -d "$DRL_REPO" ]]; then
  echo "DRL repo not found: $DRL_REPO" >&2
  exit 1
fi
if [[ ! -f "$TRUE_GRID" ]]; then
  echo "true_grid not found: $TRUE_GRID" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

cd "$ROS_WS"
colcon build --packages-select drl_explore_bridge --symlink-install
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
source install/setup.bash

ros2 run drl_explore_bridge drl_standalone_gazebo_bridge_node --ros-args \
  -p drl_repo:="$DRL_REPO" \
  -p checkpoint_path:="$CHECKPOINT" \
  -p true_grid_path:="$TRUE_GRID" \
  -p cell_size:="$CELL_SIZE" \
  -p rows:="$ROWS" \
  -p cols:="$COLS" \
  -p world_x:="$WORLD_X" \
  -p world_y:="$WORLD_Y" \
  -p scan_radius_cells:="$SCAN_RADIUS_CELLS" \
  -p scan_bridge_mode:="$SCAN_BRIDGE_MODE" \
  -p max_steps:=1 \
  -p multi_wall_timeout:="$MULTI_WALL_TIMEOUT" \
  -p diagnostic_compare_local_snaps:=true \
  -p diagnostic_print_ascii:=true \
  -p diagnostic_only_first_step:=true \
  -p diagnostic_no_motion:=true
