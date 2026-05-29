#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCAN_BRIDGE_MODE="${SCAN_BRIDGE_MODE:-scan_template_los}"

exec bash "$SCRIPT_DIR/run_cell035_local_snap_diagnostic.sh"
