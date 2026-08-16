#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/realcar_env.sh"
setup_realcar_environment

: "${DRL_CHECKPOINT_PATH:?Set DRL_CHECKPOINT_PATH to the policy checkpoint file}"
if [[ ! -f "${DRL_CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${DRL_CHECKPOINT_PATH}" >&2
  exit 1
fi

echo "Safety: this script runs realcar_policy_dryrun_node only."
echo "Safety: it does NOT publish /cmd_vel."

echo "=== cmd_vel check ==="
ros2 topic info /cmd_vel || echo "/cmd_vel not available"

ros2 run drl_explore_bridge realcar_policy_dryrun_node --ros-args \
  -p checkpoint_path:="${DRL_CHECKPOINT_PATH}" \
  -p max_steps:=10
