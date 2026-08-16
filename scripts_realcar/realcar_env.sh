# Shared path configuration for the real-robot helper scripts.
# Override any value in the environment before running a helper script.

REALCAR_SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REALCAR_REPO_ROOT="$(cd -- "${REALCAR_SCRIPTS_DIR}/.." && pwd)"
REALCAR_ROS_DISTRO="${ROS_DISTRO:-humble}"
REALCAR_ROS_SETUP="${REALCAR_ROS_SETUP:-/opt/ros/${REALCAR_ROS_DISTRO}/setup.bash}"
REALCAR_BASE_WS_SETUP="${REALCAR_BASE_WS_SETUP:-}"
REALCAR_BRIDGE_WS_SETUP="${REALCAR_BRIDGE_WS_SETUP:-${REALCAR_REPO_ROOT}/ros2_ws/install/setup.bash}"

source_required_setup() {
  local setup_file="$1"
  if [[ ! -f "${setup_file}" ]]; then
    echo "ROS setup file not found: ${setup_file}" >&2
    return 1
  fi

  set +u
  source "${setup_file}"
  set -u
}

setup_realcar_environment() {
  source_required_setup "${REALCAR_ROS_SETUP}"
  if [[ -n "${REALCAR_BASE_WS_SETUP}" ]]; then
    source_required_setup "${REALCAR_BASE_WS_SETUP}"
  fi
  source_required_setup "${REALCAR_BRIDGE_WS_SETUP}"
}
