#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=realcar_env.sh
source "${SCRIPT_DIR}/realcar_env.sh"

execute=false
if [[ "${1:-}" == "--execute" ]]; then
  execute=true
  shift
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [--execute] [additional ROS arguments]"
  echo "Default: execute=false. Use --execute explicitly to enable real motion."
  exit 0
fi

for argument in "$@"; do
  if [[ "${argument}" == *"execute:="* ]]; then
    echo "Do not pass execute as a ROS argument; use the explicit --execute flag." >&2
    exit 2
  fi
done

setup_realcar_environment

package_list="$(ros2 pkg list)"
if ! grep -Fxq "drl_explore_bridge" <<<"${package_list}"; then
  echo "ROS2 package not found: drl_explore_bridge" >&2
  exit 1
fi

topic_list="$(ros2 topic list)"
for required_topic in /scan /odom; do
  if ! grep -Fxq "${required_topic}" <<<"${topic_list}"; then
    echo "Required topic not found: ${required_topic}" >&2
    exit 1
  fi
done

if [[ "${execute}" == "true" ]]; then
  echo "WARNING: REAL ROBOT MOTION ENABLED"
  echo "Direct supervision and an emergency stop are required."
else
  echo "Observation-only run: execute=false; no non-zero motion command will be sent."
fi

ros2 run drl_explore_bridge realcar_step_once_safe_node \
  --ros-args \
  -p execute:="${execute}" \
  "$@"
