#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/realcar_env.sh"
setup_realcar_environment

echo "Safety: this script only runs scan_to_local_snap."
echo "Safety: it does NOT publish /cmd_vel."

echo "=== cmd_vel check ==="
ros2 topic info /cmd_vel || echo "/cmd_vel not available"

ros2 run drl_explore_bridge scan_to_local_snap --ros-args \
  -p cell_size:=0.35 \
  -p scan_radius_cells:=10 \
  -p laser_yaw_in_base:=0.0 \
  -p world_x:=21.0 \
  -p world_y:=14.0 \
  -p print_ascii:=true \
  -p print_period_sec:=1.0
