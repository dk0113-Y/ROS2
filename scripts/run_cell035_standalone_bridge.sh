#!/usr/bin/env bash
set -euo pipefail

ROS_REPO="${ROS_REPO:-$HOME/ROS2}"
ROS_WS="${ROS_WS:-$ROS_REPO/ros2_ws}"
DRL_REPO="${DRL_REPO:-$HOME/drl_repos/DRL-path-finding}"
ROS_DISTRO="${ROS_DISTRO:-humble}"

CHECKPOINT="${CHECKPOINT:-$DRL_REPO/deploy_checkpoints/last.pt}"
TRUE_GRID="${TRUE_GRID:-$ROS_REPO/assets/cell035/grids/random_train_like_seed20260513_true_grid.npy}"
SCAN_BRIDGE_MODE="${SCAN_BRIDGE_MODE:-los_compatible}"
MAX_STEPS="${MAX_STEPS:-400}"
COVERAGE_GOAL="${COVERAGE_GOAL:-0.95}"
NO_PROGRESS_LIMIT="${NO_PROGRESS_LIMIT:-120}"
LINEAR_SPEED="${LINEAR_SPEED:-0.10}"
CELL_SIZE="${CELL_SIZE:-0.35}"
WORLD_X="${WORLD_X:-21.0}"
WORLD_Y="${WORLD_Y:-14.0}"

ROWS="${ROWS:-40}"
COLS="${COLS:-60}"
SCAN_RADIUS_CELLS="${SCAN_RADIUS_CELLS:-10}"

echo "This script only starts the standalone DRL bridge."
echo "Start the Gazebo world, robot, sensors, odometry, and velocity controller first."

ROS_SETUP="/opt/ros/$ROS_DISTRO/setup.bash"
if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ROS setup file not found: $ROS_SETUP" >&2
  exit 1
fi
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
  -p max_steps:="$MAX_STEPS" \
  -p coverage_goal:="$COVERAGE_GOAL" \
  -p no_progress_limit:="$NO_PROGRESS_LIMIT" \
  -p linear_speed:="$LINEAR_SPEED"
