#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/realcar_env.sh"
setup_realcar_environment

CHECK_SECONDS="${REALCAR_CHECK_SECONDS:-12}"
DISCOVERY_SECONDS="${REALCAR_DISCOVERY_SECONDS:-5}"
ODOM_FRAME="${REALCAR_ODOM_FRAME:-odom_combined}"
BASE_FRAME="${REALCAR_BASE_FRAME:-base_link}"
LASER_FRAME="${REALCAR_LASER_FRAME:-laser}"
failures=0

section() {
  printf '\n=== %s ===\n' "$1"
}

observe() {
  if ! "$@"; then
    failures=1
  fi
}

observe_once() {
  local result=0
  timeout "${CHECK_SECONDS}s" "$@" || result=$?
  if [[ ${result} -ne 0 ]]; then
    failures=1
  fi
}

sample_topic_rate() {
  local topic="$1"
  local output
  output="$(
    timeout "${CHECK_SECONDS}s" \
      ros2 topic hz "${topic}" --spin-time "${DISCOVERY_SECONDS}" 2>&1 || true
  )"
  printf '%s\n' "${output}"
  if [[ "${output}" != *"average rate:"* ]]; then
    printf 'ERROR: no frequency sample received from %s\n' "${topic}" >&2
    failures=1
  fi
}

check_tf() {
  local parent_frame="$1"
  local child_frame="$2"
  local output
  output="$(
    timeout "${CHECK_SECONDS}s" \
      ros2 run tf2_ros tf2_echo "${parent_frame}" "${child_frame}" 2>&1 \
      || true
  )"
  printf '%s\n' "${output}"
  if [[ "${output}" != *"- Translation:"* ]]; then
    printf 'ERROR: no TF received for %s -> %s\n' \
      "${parent_frame}" "${child_frame}" >&2
    failures=1
  fi
}

require_topic_type() {
  local topic="$1"
  local expected_type="$2"
  local actual_type
  actual_type="$(
    ros2 topic type "${topic}" --no-daemon \
      --spin-time "${DISCOVERY_SECONDS}" 2>/dev/null || true
  )"
  if [[ "${actual_type}" != "${expected_type}" ]]; then
    printf 'ERROR: %s type is %s; expected %s\n' \
      "${topic}" "${actual_type:-missing}" "${expected_type}" >&2
    failures=1
  else
    printf 'OK: %s type=%s\n' "${topic}" "${actual_type}"
  fi
}

section "ROS nodes"
observe ros2 node list --no-daemon --spin-time "${DISCOVERY_SECONDS}"

section "ROS topics and types"
observe ros2 topic list -t --no-daemon --spin-time "${DISCOVERY_SECONDS}"
require_topic_type /cmd_vel geometry_msgs/msg/Twist
require_topic_type /odom nav_msgs/msg/Odometry
require_topic_type /scan sensor_msgs/msg/LaserScan

section "/cmd_vel endpoints (inspection only; this script never publishes)"
observe ros2 topic info /cmd_vel -v --no-daemon \
  --spin-time "${DISCOVERY_SECONDS}"

section "/odom endpoints"
observe ros2 topic info /odom -v --no-daemon \
  --spin-time "${DISCOVERY_SECONDS}"

section "/scan endpoints"
observe ros2 topic info /scan -v --no-daemon \
  --spin-time "${DISCOVERY_SECONDS}"

section "/scan header sample"
observe_once ros2 topic echo /scan sensor_msgs/msg/LaserScan \
  --once --field header --no-daemon --spin-time "${DISCOVERY_SECONDS}"

section "/odom header sample"
observe_once ros2 topic echo /odom nav_msgs/msg/Odometry \
  --once --field header --no-daemon --spin-time "${DISCOVERY_SECONDS}"

section "/odom child_frame_id sample"
observe_once ros2 topic echo /odom nav_msgs/msg/Odometry \
  --once --field child_frame_id --no-daemon \
  --spin-time "${DISCOVERY_SECONDS}"

section "/scan frequency"
sample_topic_rate /scan

section "/odom frequency"
sample_topic_rate /odom

section "TF ${ODOM_FRAME} -> ${BASE_FRAME}"
check_tf "${ODOM_FRAME}" "${BASE_FRAME}"

section "TF ${BASE_FRAME} -> ${LASER_FRAME}"
check_tf "${BASE_FRAME}" "${LASER_FRAME}"

if [[ ${failures} -ne 0 ]]; then
  printf '\nRealcar interface inspection FAILED. Review the output above.\n' >&2
  exit 1
fi

printf '\nRealcar interface inspection completed without detected errors.\n'
