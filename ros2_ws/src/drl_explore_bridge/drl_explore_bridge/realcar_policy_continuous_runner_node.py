"""
Guarded Round 8 continuous real-car exploration runner.

The real-world completion criteria in this module are belief-side only.  The
zero-valued ``true_grid`` passed to ``CumulativeBeliefMap`` is a constructor
compatibility placeholder and is never used for coverage or termination.
"""

from __future__ import annotations

import math
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import LaserScan

from drl_explore_bridge.realcar_action_adapter import ActionExecutionTarget
from drl_explore_bridge.realcar_policy_safe_runner_node import (
    ACTIONS_8,
    INVISIBLE,
    RealcarPolicySafeRunner,
    SensorRefreshBarrier,
    format_optional_float,
    load_policy_model,
    norm_angle,
    odom_delta_to_grid_offset,
    sensor_sample_is_after_barrier,
)


CONTINUOUS_CELL_SIZE_M = 0.35
CONTINUOUS_DEFAULT_MAX_STEPS = 30
CONTINUOUS_MAX_STEPS_LIMIT = 1000
DEFAULT_MOTION_CLEARANCE_MARGIN_M = 0.25
DEFAULT_DYNAMIC_STOP_DISTANCE_M = 0.25
DEFAULT_DYNAMIC_STOP_RECOVERY_LIMIT = 3
DEFAULT_LOCAL_ESCAPE_RECOVERY_LIMIT = 3
LOCAL_ESCAPE_CANDIDATE_DISTANCES_M = (0.20, 0.15, 0.10)
DEFAULT_NOMINAL_MIN_CORRIDOR_WIDTH_M = 0.40
DEFAULT_FOOTPRINT_RADIUS_M = 0.20
DEFAULT_LONGITUDINAL_EXTRA_MARGIN_M = 0.05
DEFAULT_LASER_X_IN_BASE_M = 0.03163
DEFAULT_LASER_Y_IN_BASE_M = 0.00009
FOOTPRINT_COMPARISON_TOLERANCE_M = 1.0e-9


class DynamicObstacleStop(RuntimeError):
    """Signal that a fresh drive-phase scan requires an immediate stop."""


class RotationFootprintBlocked(RuntimeError):
    """Signal that the full rotation footprint is not clear."""


class DriveSensorCycleTimeout(RuntimeError):
    """Signal that scan and odom did not both advance before the watchdog."""


@dataclass(frozen=True)
class ContinuousPreMotionPlan:
    """Describe the final action and distance-aware pre-motion gate result."""

    raw_policy_action: int
    executed_action: Optional[int]
    target: Optional[ActionExecutionTarget]
    pre_motion_pose: dict[str, float]
    target_distance: Optional[float]
    capsule_front_edge: Optional[float]
    nearest_capsule_clearance: Optional[float]
    pre_motion_footprint_passed: bool
    obstruction_type: Optional[str]
    gate_passed: bool
    safety_fallback_used: bool


@dataclass(frozen=True)
class LocalEscapePlan:
    """Describe one bounded deployment-only local recovery primitive."""

    available: bool
    action_idx: Optional[int]
    distance_m: Optional[float]
    target: Optional[ActionExecutionTarget]
    pre_motion_pose: dict[str, float]
    nearest_clearance: Optional[float]
    candidate_evaluations: list[dict[str, Any]]


@dataclass(frozen=True)
class FootprintCheck:
    """Describe scan-point clearance from a swept circular footprint."""

    passed: bool
    nearest_capsule_clearance: Optional[float]
    obstruction_type: Optional[str]
    valid_point_count: int


def scan_points_in_base(
    scan: LaserScan,
    laser_x_in_base: float,
    laser_y_in_base: float,
    laser_yaw_in_base: float,
) -> list[tuple[float, float]]:
    """Transform valid planar LaserScan hits into the base frame."""
    points: list[tuple[float, float]] = []
    for index, raw_range in enumerate(scan.ranges):
        hit_range = float(raw_range)
        if not (
            math.isfinite(hit_range)
            and float(scan.range_min) <= hit_range <= float(scan.range_max)
        ):
            continue
        scan_yaw = scan.angle_min + index * scan.angle_increment
        base_yaw = laser_yaw_in_base + scan_yaw
        points.append(
            (
                laser_x_in_base + hit_range * math.cos(base_yaw),
                laser_y_in_base + hit_range * math.sin(base_yaw),
            )
        )
    return points


def capsule_footprint_check(
    points_in_base: Sequence[tuple[float, float]],
    motion_yaw_in_base: float,
    center_line_length: float,
    footprint_radius: float,
    comparison_tolerance: float = FOOTPRINT_COMPARISON_TOLERANCE_M,
) -> FootprintCheck:
    """Check scan points against a line-segment plus circular footprint."""
    if center_line_length < 0.0:
        raise ValueError("center_line_length must be >= 0")
    if footprint_radius <= 0.0:
        raise ValueError("footprint_radius must be > 0")
    if comparison_tolerance < 0.0:
        raise ValueError("comparison_tolerance must be >= 0")

    cos_yaw = math.cos(motion_yaw_in_base)
    sin_yaw = math.sin(motion_yaw_in_base)
    nearest_clearance: Optional[float] = None
    nearest_projection = 0.0
    for point_x, point_y in points_in_base:
        forward = point_x * cos_yaw + point_y * sin_yaw
        lateral = -point_x * sin_yaw + point_y * cos_yaw
        segment_forward = min(max(forward, 0.0), center_line_length)
        distance = math.hypot(forward - segment_forward, lateral)
        clearance = distance - footprint_radius
        if nearest_clearance is None or clearance < nearest_clearance:
            nearest_clearance = clearance
            nearest_projection = forward

    if nearest_clearance is None:
        return FootprintCheck(False, None, "invalid_scan", 0)
    if nearest_clearance >= -comparison_tolerance:
        return FootprintCheck(
            True,
            nearest_clearance,
            None,
            len(points_in_base),
        )
    obstruction_type = "footprint_corridor_obstruction"
    if (
        center_line_length > 0.0
        and nearest_projection >= center_line_length
    ):
        obstruction_type = "longitudinal_path_obstruction"
    return FootprintCheck(
        False,
        nearest_clearance,
        obstruction_type,
        len(points_in_base),
    )


def scan_capsule_footprint_check(
    scan: LaserScan,
    motion_yaw_in_base: float,
    center_line_length: float,
    footprint_radius: float,
    laser_x_in_base: float,
    laser_y_in_base: float,
    laser_yaw_in_base: float,
) -> FootprintCheck:
    """Transform a scan and evaluate its swept circular footprint."""
    return capsule_footprint_check(
        scan_points_in_base(
            scan,
            laser_x_in_base,
            laser_y_in_base,
            laser_yaw_in_base,
        ),
        motion_yaw_in_base,
        center_line_length,
        footprint_radius,
    )


def drive_sensor_sequence_progress(
    scan_sequence: int,
    odom_sequence: int,
    previous_scan_sequence: int,
    previous_odom_sequence: int,
) -> tuple[bool, bool, bool]:
    """Report whether scan, odom, and the complete drive pair advanced."""
    scan_advanced = scan_sequence > previous_scan_sequence
    odom_advanced = odom_sequence > previous_odom_sequence
    return scan_advanced, odom_advanced, scan_advanced and odom_advanced


def belief_statistics(cum_map: Any) -> tuple[int, int]:
    """Return belief-side known and frontier cell counts."""
    known_cells = int(np.count_nonzero(np.asarray(cum_map.map) != INVISIBLE))
    frontier_count = int(np.count_nonzero(cum_map.get_frontier_u8() > 0))
    return known_cells, frontier_count


def frontier_exhausted(
    frontier_count: int,
    decision_steps: int,
    known_cells: int,
    minimum_decision_steps: int,
    minimum_known_cells: int,
) -> bool:
    """Guard frontier completion against empty or immature observations."""
    return (
        decision_steps >= minimum_decision_steps
        and known_cells >= minimum_known_cells
        and frontier_count == 0
    )


def known_area_stagnated(
    known_history: Sequence[int],
    window_steps: int,
    minimum_growth: int,
) -> bool:
    """Detect insufficient belief growth over a complete sliding window."""
    if window_steps <= 0:
        raise ValueError("window_steps must be > 0")
    if minimum_growth < 0:
        raise ValueError("minimum_growth must be >= 0")
    if len(known_history) < window_steps + 1:
        return False
    window = known_history[-(window_steps + 1):]
    return max(window) - window[0] < minimum_growth


def repeated_state_deadlock(
    state_history: Sequence[tuple[int, int]],
    known_history: Sequence[int],
    window_steps: int,
    maximum_unique_states: int,
    minimum_known_growth: int,
) -> bool:
    """Detect repeated-state deadlock only when belief also stops growing."""
    if window_steps <= 0:
        raise ValueError("window_steps must be > 0")
    if maximum_unique_states <= 0:
        raise ValueError("maximum_unique_states must be > 0")
    if len(state_history) < window_steps or len(known_history) < window_steps:
        return False
    states = state_history[-window_steps:]
    known = known_history[-window_steps:]
    state_repeated = len(set(states)) <= maximum_unique_states
    information_stalled = max(known) - known[0] < minimum_known_growth
    return state_repeated and information_stalled


def expected_grid_state_from_action(
    actual_state: tuple[int, int],
    action_idx: int,
) -> tuple[int, int]:
    """Return an action-derived state for diagnostics only."""
    dr, dc = ACTIONS_8[action_idx]
    return actual_state[0] + dr, actual_state[1] + dc


def grid_transition_matches(
    expected_state: tuple[int, int],
    actual_state: tuple[int, int],
) -> bool:
    """Compare diagnostic expectation with the odom-derived state."""
    return expected_state == actual_state


def motion_is_permitted(
    execute: bool,
    gate_passed: bool,
    executed_action: Optional[int],
) -> bool:
    """Require explicit execution and a safe final action before motion."""
    return execute and gate_passed and executed_action is not None


def validate_commissioning_config(
    commissioning_mode: bool,
    max_steps: int,
    commissioning_action_idx: int,
) -> None:
    """Reject unsafe or ambiguous commissioning configurations."""
    if not commissioning_mode:
        return
    if max_steps != 1:
        raise ValueError("commissioning_mode requires max_steps == 1")
    if commissioning_action_idx not in range(8):
        raise ValueError("commissioning_action_idx must be in [0, 7]")


def action_source_for_mode(commissioning_mode: bool) -> str:
    """Identify whether the motion candidate came from policy or operator."""
    if commissioning_mode:
        return "commissioning_override"
    return "policy"


def successful_step_termination_reason(
    commissioning_mode: bool,
    execute: bool,
) -> Optional[str]:
    """Return the dedicated completion reason for a successful commission."""
    if commissioning_mode:
        if execute:
            return "commissioning_complete"
        return "commissioning_dryrun_complete"
    return None


def episode_success_for_reason(
    commissioning_mode: bool,
    execute: bool,
    exit_reason: str,
) -> bool:
    """Classify normal exploration and commissioning success separately."""
    if commissioning_mode:
        return execute and exit_reason == "commissioning_complete"
    return exit_reason == "frontier_exhausted"


def hard_limit_termination(
    completed_steps: int,
    max_steps: int,
    elapsed_sec: float,
    max_runtime_sec: float,
) -> Optional[str]:
    """Return the active hard-limit reason, preferring runtime safety."""
    if elapsed_sec >= max_runtime_sec:
        return "max_runtime_reached"
    if completed_steps >= max_steps:
        return "max_steps_reached"
    return None


def repository_commit(path: Path) -> str:
    """Read the containing Git commit without mutating repository state."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


class RealcarPolicyContinuousRunner(RealcarPolicySafeRunner):
    """Run bounded, belief-terminated exploration with Round 8 safeguards."""

    NODE_NAME = "realcar_policy_continuous_runner_node"
    DEFAULT_MAX_STEPS = CONTINUOUS_DEFAULT_MAX_STEPS
    MAX_STEPS_LIMIT = CONTINUOUS_MAX_STEPS_LIMIT
    DEFAULT_STEP_DISTANCE = CONTINUOUS_CELL_SIZE_M
    SEPARATE_SENSOR_CALLBACK_GROUPS = True

    def __init__(self) -> None:
        self._sensor_condition = threading.Condition()
        super().__init__()

        self.declare_parameter("max_runtime_sec", 1800.0)
        self.declare_parameter(
            "motion_clearance_margin",
            DEFAULT_MOTION_CLEARANCE_MARGIN_M,
        )
        self.declare_parameter(
            "dynamic_stop_distance",
            DEFAULT_DYNAMIC_STOP_DISTANCE_M,
        )
        self.declare_parameter(
            "nominal_min_corridor_width_m",
            DEFAULT_NOMINAL_MIN_CORRIDOR_WIDTH_M,
        )
        self.declare_parameter(
            "footprint_radius_m",
            DEFAULT_FOOTPRINT_RADIUS_M,
        )
        self.declare_parameter(
            "longitudinal_extra_margin_m",
            DEFAULT_LONGITUDINAL_EXTRA_MARGIN_M,
        )
        self.declare_parameter(
            "laser_x_in_base_m",
            DEFAULT_LASER_X_IN_BASE_M,
        )
        self.declare_parameter(
            "laser_y_in_base_m",
            DEFAULT_LASER_Y_IN_BASE_M,
        )
        self.declare_parameter("minimum_completion_decision_steps", 3)
        self.declare_parameter("minimum_completion_known_cells", 20)
        self.declare_parameter("stagnation_window_steps", 10)
        self.declare_parameter("stagnation_min_known_growth", 1)
        self.declare_parameter("deadlock_window_steps", 8)
        self.declare_parameter("deadlock_maximum_unique_states", 2)
        self.declare_parameter("deadlock_min_known_growth", 1)
        self.declare_parameter("no_safe_action_retries", 0)
        self.declare_parameter(
            "dynamic_stop_recovery_limit",
            DEFAULT_DYNAMIC_STOP_RECOVERY_LIMIT,
        )
        self.declare_parameter(
            "local_escape_recovery_limit",
            DEFAULT_LOCAL_ESCAPE_RECOVERY_LIMIT,
        )
        self.declare_parameter("drive_sensor_cycle_timeout_sec", 0.25)
        self.declare_parameter("commissioning_mode", False)
        self.declare_parameter("commissioning_action_idx", -1)
        self.declare_parameter(
            "disable_completion_termination_in_dryrun",
            True,
        )

        self.max_runtime_sec = float(
            self.get_parameter("max_runtime_sec").value
        )
        self.motion_clearance_margin = float(
            self.get_parameter("motion_clearance_margin").value
        )
        self.dynamic_stop_distance = float(
            self.get_parameter("dynamic_stop_distance").value
        )
        self.nominal_min_corridor_width_m = float(
            self.get_parameter("nominal_min_corridor_width_m").value
        )
        self.footprint_radius_m = float(
            self.get_parameter("footprint_radius_m").value
        )
        self.longitudinal_extra_margin_m = float(
            self.get_parameter("longitudinal_extra_margin_m").value
        )
        self.laser_x_in_base_m = float(
            self.get_parameter("laser_x_in_base_m").value
        )
        self.laser_y_in_base_m = float(
            self.get_parameter("laser_y_in_base_m").value
        )
        self.dynamic_forward_center_line_extension_m = (
            self.dynamic_stop_distance - self.footprint_radius_m
        )
        self.minimum_completion_decision_steps = int(
            self.get_parameter("minimum_completion_decision_steps").value
        )
        self.minimum_completion_known_cells = int(
            self.get_parameter("minimum_completion_known_cells").value
        )
        self.stagnation_window_steps = int(
            self.get_parameter("stagnation_window_steps").value
        )
        self.stagnation_min_known_growth = int(
            self.get_parameter("stagnation_min_known_growth").value
        )
        self.deadlock_window_steps = int(
            self.get_parameter("deadlock_window_steps").value
        )
        self.deadlock_maximum_unique_states = int(
            self.get_parameter("deadlock_maximum_unique_states").value
        )
        self.deadlock_min_known_growth = int(
            self.get_parameter("deadlock_min_known_growth").value
        )
        self.no_safe_action_retries = int(
            self.get_parameter("no_safe_action_retries").value
        )
        self.dynamic_stop_recovery_limit = int(
            self.get_parameter("dynamic_stop_recovery_limit").value
        )
        self.local_escape_recovery_limit = int(
            self.get_parameter("local_escape_recovery_limit").value
        )
        self.drive_sensor_cycle_timeout_sec = float(
            self.get_parameter("drive_sensor_cycle_timeout_sec").value
        )
        self.commissioning_mode = bool(
            self.get_parameter("commissioning_mode").value
        )
        self.commissioning_action_idx = int(
            self.get_parameter("commissioning_action_idx").value
        )
        self.disable_completion_termination_in_dryrun = bool(
            self.get_parameter(
                "disable_completion_termination_in_dryrun"
            ).value
        )

        self._validate_continuous_parameters()
        self._last_dynamic_scan_sequence = self.scan_sequence
        self._dynamic_step_record: Optional[dict[str, Any]] = None
        self._episode_travel_distance = 0.0

        started_at = datetime.now().astimezone()
        repo_root = Path(__file__).resolve().parents[4]
        self.experiment_result = {
            "experiment_id": (
                f"realcar_continuous_{started_at.strftime('%Y%m%d_%H%M%S')}"
            ),
            "timestamp": started_at.isoformat(timespec="milliseconds"),
            "git_commit": repository_commit(repo_root),
            "checkpoint_path": self.checkpoint_path,
            "execute": self.execute,
            "cell_size": self.cell_size,
            "step_distance": self.step_distance,
            "diagonal_mode": self.diagonal_mode,
            "max_steps": self.max_steps,
            "max_runtime_sec": self.max_runtime_sec,
            "commissioning_mode": self.commissioning_mode,
            "commissioning_action_idx": self.commissioning_action_idx,
            "action_source": action_source_for_mode(
                self.commissioning_mode
            ),
            "termination_parameters": {
                "minimum_completion_decision_steps": (
                    self.minimum_completion_decision_steps
                ),
                "minimum_completion_known_cells": (
                    self.minimum_completion_known_cells
                ),
                "stagnation_window_steps": self.stagnation_window_steps,
                "stagnation_min_known_growth": (
                    self.stagnation_min_known_growth
                ),
                "deadlock_window_steps": self.deadlock_window_steps,
                "deadlock_maximum_unique_states": (
                    self.deadlock_maximum_unique_states
                ),
                "deadlock_min_known_growth": self.deadlock_min_known_growth,
                "no_safe_action_retries": self.no_safe_action_retries,
                "dynamic_stop_recovery_limit": (
                    self.dynamic_stop_recovery_limit
                ),
                "local_escape_recovery_limit": (
                    self.local_escape_recovery_limit
                ),
                "disable_completion_termination_in_dryrun": (
                    self.disable_completion_termination_in_dryrun
                ),
            },
            "motion_clearance_margin": self.motion_clearance_margin,
            "dynamic_stop_distance": self.dynamic_stop_distance,
            "nominal_min_corridor_width_m": (
                self.nominal_min_corridor_width_m
            ),
            "footprint_radius_m": self.footprint_radius_m,
            "longitudinal_extra_margin_m": (
                self.longitudinal_extra_margin_m
            ),
            "dynamic_forward_center_line_extension_m": (
                self.dynamic_forward_center_line_extension_m
            ),
            "laser_x_in_base_m": self.laser_x_in_base_m,
            "laser_y_in_base_m": self.laser_y_in_base_m,
            "drive_sensor_cycle_timeout_sec": (
                self.drive_sensor_cycle_timeout_sec
            ),
            "total_steps": 0,
            "successful_steps": 0,
            "travel_distance": 0.0,
            "episode_duration": 0.0,
            "termination_reason": "not_started",
            "success": False,
            "dynamic_stop_total_count": 0,
            "dynamic_stop_recovery_total_count": 0,
            "dynamic_stop_deadlock": False,
            "local_escape_total_count": 0,
            "local_escape_success_total_count": 0,
            "local_escape_deadlock": False,
            "local_escape_action_history": [],
            "steps": [],
        }
        self.get_logger().warn(
            "Round 8 continuous configuration "
            f"cell_size={self.cell_size:.3f} "
            f"step_distance={self.step_distance:.3f} "
            "cardinal_distance=0.350 "
            f"diagonal_distance={math.sqrt(2.0) * self.cell_size:.3f} "
            "nominal_min_corridor_width_m="
            f"{self.nominal_min_corridor_width_m:.3f} "
            f"footprint_radius_m={self.footprint_radius_m:.3f} "
            "footprint_status=deployment_safety_envelope_not_body_radius "
            "longitudinal_extra_margin_m="
            f"{self.longitudinal_extra_margin_m:.3f} "
            "motion_clearance_margin_legacy="
            f"{self.motion_clearance_margin:.3f} "
            f"dynamic_stop_distance={self.dynamic_stop_distance:.3f} "
            "dynamic_forward_center_line_extension_m="
            f"{self.dynamic_forward_center_line_extension_m:.3f} "
            f"laser_x_in_base_m={self.laser_x_in_base_m:.5f} "
            f"laser_y_in_base_m={self.laser_y_in_base_m:.5f} "
            f"laser_yaw_in_base={self.laser_yaw_in_base:.5f} "
            "drive_sensor_cycle_timeout_sec="
            f"{self.drive_sensor_cycle_timeout_sec:.3f} "
            "drive_sensor_watchdog_status=engineering_watchdog "
            "dynamic_stop_recovery_limit="
            f"{self.dynamic_stop_recovery_limit} "
            "local_escape_candidate_distances_m="
            f"{list(LOCAL_ESCAPE_CANDIDATE_DISTANCES_M)} "
            "local_escape_recovery_limit="
            f"{self.local_escape_recovery_limit} "
            f"commissioning_mode={self.commissioning_mode} "
            f"commissioning_action_idx={self.commissioning_action_idx} "
            f"action_source={action_source_for_mode(self.commissioning_mode)} "
            "sensor_executor=background_multithreaded "
            "sensor_callback_groups=independent "
            "completion_source=belief_only"
        )

    def scan_cb(self, msg: LaserScan) -> None:
        """Update scan state and wake control waiters from the executor."""
        with self._sensor_condition:
            super().scan_cb(msg)
            self._sensor_condition.notify_all()

    def odom_cb(self, msg: Odometry) -> None:
        """Update odom state and wake control waiters from the executor."""
        with self._sensor_condition:
            super().odom_cb(msg)
            self._sensor_condition.notify_all()

    def wait_for_control_callbacks(self, timeout_sec: float) -> None:
        """Yield to continuously running background sensor callbacks."""
        with self._sensor_condition:
            self._sensor_condition.wait(timeout=max(0.0, timeout_sec))

    def wait_for_inputs(
        self,
        after_scan_sequence: int = -1,
        after_odom_sequence: int = -1,
        after_monotonic: Optional[float] = None,
    ) -> None:
        """Wait on background callback notifications without spinning here."""
        deadline = time.monotonic() + self.decision_timeout_sec
        last_error = "no messages received"
        with self._sensor_condition:
            while rclpy.ok() and time.monotonic() < deadline:
                if self.latest_scan is not None and self.latest_odom is not None:
                    newer_scan = self.scan_sequence > after_scan_sequence
                    newer_odom = self.odom_sequence > after_odom_sequence
                    if after_monotonic is not None:
                        newer_scan = sensor_sample_is_after_barrier(
                            self.scan_sequence,
                            self.latest_scan_received_at,
                            after_scan_sequence,
                            after_monotonic,
                        )
                        newer_odom = sensor_sample_is_after_barrier(
                            self.odom_sequence,
                            self.latest_odom_received_at,
                            after_odom_sequence,
                            after_monotonic,
                        )
                    if newer_scan and newer_odom:
                        try:
                            super().require_fresh_inputs()
                        except RuntimeError as exc:
                            last_error = str(exc)
                        else:
                            return
                    else:
                        last_error = (
                            "waiting for background /scan and /odom callbacks: "
                            f"scan={self.scan_sequence}>{after_scan_sequence} "
                            f"odom={self.odom_sequence}>{after_odom_sequence}"
                        )
                remaining = deadline - time.monotonic()
                self._sensor_condition.wait(
                    timeout=min(0.1, max(0.0, remaining))
                )
        diagnostics = self.format_sensor_age_diagnostics(
            self.sensor_age_diagnostics()
        )
        raise TimeoutError(
            "timeout waiting for fresh /scan and /odom: "
            f"{last_error}; {diagnostics}"
        )

    def sensor_age_diagnostics(self) -> dict[str, Optional[float] | int]:
        """Read coherent sensor sequence and age fields under the sensor lock."""
        with self._sensor_condition:
            return super().sensor_age_diagnostics()

    def require_fresh_inputs(self) -> None:
        """Validate a coherent latest scan/odom snapshot under the lock."""
        with self._sensor_condition:
            super().require_fresh_inputs()

    def pose_record(self) -> dict[str, float]:
        """Read the latest odom pose under the sensor lock."""
        with self._sensor_condition:
            return super().pose_record()

    def pose_xy_yaw_time(self) -> tuple[float, float, float, float]:
        """Read the latest control pose under the sensor lock."""
        with self._sensor_condition:
            return super().pose_xy_yaw_time()

    def capture_sensor_refresh_barrier(self):
        """Capture both background callback sequences under the sensor lock."""
        with self._sensor_condition:
            return super().capture_sensor_refresh_barrier()

    def refresh_inputs_after_barrier(
        self,
        barrier: SensorRefreshBarrier,
    ) -> tuple[LaserScan, Odometry]:
        """Return a coherent post-barrier pair updated by the executor."""
        self.wait_for_inputs(
            after_scan_sequence=barrier.scan_sequence,
            after_odom_sequence=barrier.odom_sequence,
            after_monotonic=barrier.started_monotonic,
        )
        with self._sensor_condition:
            if self.latest_scan is None or self.latest_odom is None:
                raise RuntimeError(
                    "missing /scan or /odom after sensor refresh barrier"
                )
            return self.latest_scan, self.latest_odom

    def consume_new_step_inputs(self) -> tuple[LaserScan, Odometry]:
        """Consume a coherent pair produced by background callbacks."""
        self.wait_for_inputs(
            self.last_consumed_scan_sequence,
            self.last_consumed_odom_sequence,
        )
        with self._sensor_condition:
            if self.latest_scan is None or self.latest_odom is None:
                raise RuntimeError("missing /scan or /odom after fresh input wait")
            self.last_consumed_scan_sequence = self.scan_sequence
            self.last_consumed_odom_sequence = self.odom_sequence
            return self.latest_scan, self.latest_odom

    def refresh_after_motion(self) -> None:
        """Refresh and mark a coherent post-motion sensor pair consumed."""
        barrier = self.capture_sensor_refresh_barrier()
        self.refresh_inputs_after_barrier(barrier)
        with self._sensor_condition:
            self.last_consumed_scan_sequence = self.scan_sequence
            self.last_consumed_odom_sequence = self.odom_sequence

    def refresh_after_local_escape(
        self,
        record: dict[str, Any],
    ) -> tuple[LaserScan, Odometry]:
        """Refresh both sensors after escape and record barrier diagnostics."""
        barrier = self.capture_sensor_refresh_barrier()
        started = time.monotonic()
        try:
            scan, odom = self.refresh_inputs_after_barrier(barrier)
        finally:
            record["local_escape_refresh_duration_sec"] = (
                time.monotonic() - started
            )
            diagnostics = self.sensor_age_diagnostics()
            record.update(diagnostics)
            record["local_escape_refresh_scan_advanced"] = (
                int(diagnostics["scan_sequence"]) > barrier.scan_sequence
            )
            record["local_escape_refresh_odom_advanced"] = (
                int(diagnostics["odom_sequence"]) > barrier.odom_sequence
            )
        with self._sensor_condition:
            self.last_consumed_scan_sequence = self.scan_sequence
            self.last_consumed_odom_sequence = self.odom_sequence
        return scan, odom

    def _validate_continuous_parameters(self) -> None:
        """Validate Round 8 invariants and bounded termination settings."""
        if not math.isclose(
            self.cell_size,
            CONTINUOUS_CELL_SIZE_M,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("continuous cell_size must be 0.35m")
        if not math.isclose(
            self.step_distance,
            self.cell_size,
            rel_tol=0.01,
            abs_tol=0.005,
        ):
            raise ValueError(
                "continuous step_distance must equal cell_size (0.35m)"
            )
        if self.diagonal_mode != "grid_center":
            raise ValueError("continuous diagonal_mode must be 'grid_center'")
        if not self.all8_action_mode:
            raise ValueError("continuous runner requires allowed_actions_mode=all8")
        if self.max_runtime_sec <= 0.0:
            raise ValueError("max_runtime_sec must be > 0")
        if self.motion_clearance_margin < 0.0:
            raise ValueError("motion_clearance_margin must be >= 0")
        if self.dynamic_stop_distance <= 0.0:
            raise ValueError("dynamic_stop_distance must be > 0")
        if self.nominal_min_corridor_width_m <= 0.0:
            raise ValueError("nominal_min_corridor_width_m must be > 0")
        if self.footprint_radius_m <= 0.0:
            raise ValueError("footprint_radius_m must be > 0")
        if not math.isclose(
            self.nominal_min_corridor_width_m,
            2.0 * self.footprint_radius_m,
            rel_tol=0.0,
            abs_tol=FOOTPRINT_COMPARISON_TOLERANCE_M,
        ):
            raise ValueError(
                "nominal_min_corridor_width_m must equal "
                "2 * footprint_radius_m"
            )
        if self.longitudinal_extra_margin_m < 0.0:
            raise ValueError("longitudinal_extra_margin_m must be >= 0")
        if not math.isclose(
            self.motion_clearance_margin,
            self.footprint_radius_m + self.longitudinal_extra_margin_m,
            rel_tol=0.0,
            abs_tol=FOOTPRINT_COMPARISON_TOLERANCE_M,
        ):
            raise ValueError(
                "legacy motion_clearance_margin must equal footprint_radius_m "
                "+ longitudinal_extra_margin_m"
            )
        if self.dynamic_forward_center_line_extension_m < 0.0:
            raise ValueError(
                "dynamic_stop_distance must be >= footprint_radius_m"
            )
        if self.minimum_completion_decision_steps < 1:
            raise ValueError("minimum_completion_decision_steps must be >= 1")
        if self.minimum_completion_known_cells < 1:
            raise ValueError("minimum_completion_known_cells must be >= 1")
        if self.stagnation_window_steps < 1:
            raise ValueError("stagnation_window_steps must be >= 1")
        if self.stagnation_min_known_growth < 0:
            raise ValueError("stagnation_min_known_growth must be >= 0")
        if self.deadlock_window_steps < 1:
            raise ValueError("deadlock_window_steps must be >= 1")
        if self.deadlock_maximum_unique_states < 1:
            raise ValueError("deadlock_maximum_unique_states must be >= 1")
        if self.deadlock_min_known_growth < 0:
            raise ValueError("deadlock_min_known_growth must be >= 0")
        if self.no_safe_action_retries < 0:
            raise ValueError("no_safe_action_retries must be >= 0")
        if self.dynamic_stop_recovery_limit < 1:
            raise ValueError("dynamic_stop_recovery_limit must be >= 1")
        if self.local_escape_recovery_limit < 1:
            raise ValueError("local_escape_recovery_limit must be >= 1")
        if self.drive_sensor_cycle_timeout_sec <= 0.0:
            raise ValueError("drive_sensor_cycle_timeout_sec must be > 0")
        validate_commissioning_config(
            self.commissioning_mode,
            self.max_steps,
            self.commissioning_action_idx,
        )

    def agent_state_from_odom(
        self,
        origin_state: tuple[int, int],
        x: float,
        y: float,
    ) -> tuple[int, int]:
        """Derive an unbounded world-grid state solely from cumulative odom."""
        if self.odom_state_origin is None:
            raise RuntimeError("odom_state_origin is not initialized")
        origin_x, origin_y = self.odom_state_origin
        row_offset, col_offset = odom_delta_to_grid_offset(
            x - origin_x,
            y - origin_y,
            self.cell_size,
        )
        return origin_state[0] + row_offset, origin_state[1] + col_offset

    def target_for_action(
        self,
        action_idx: int,
        x0: float,
        y0: float,
    ) -> ActionExecutionTarget:
        """Build a grid-center target from the single cell-size source."""
        return self.action_adapter.target_for_action(
            action_idx,
            start_x=x0,
            start_y=y0,
            step_distance=self.cell_size,
        )

    def target_for_action_distance(
        self,
        action_idx: int,
        x0: float,
        y0: float,
        distance_m: float,
    ) -> ActionExecutionTarget:
        """Build an exact-Euclidean-distance target along one DRL direction."""
        if not math.isfinite(distance_m) or distance_m <= 0.0:
            raise ValueError("local escape distance must be finite and > 0")
        direction = self.action_adapter.target_for_action(
            action_idx,
            start_x=0.0,
            start_y=0.0,
            step_distance=1.0,
        )
        direction_norm = math.hypot(direction.odom_dx, direction.odom_dy)
        odom_dx = direction.odom_dx / direction_norm * distance_m
        odom_dy = direction.odom_dy / direction_norm * distance_m
        return ActionExecutionTarget(
            action_idx=direction.action_idx,
            action_name=direction.action_name,
            grid_dr=direction.grid_dr,
            grid_dc=direction.grid_dc,
            odom_direction=direction.odom_direction,
            odom_dx=odom_dx,
            odom_dy=odom_dy,
            target_x=x0 + odom_dx,
            target_y=y0 + odom_dy,
            target_yaw=direction.target_yaw,
            target_distance=distance_m,
        )

    def pre_motion_footprint_check(
        self,
        target: ActionExecutionTarget,
        pose: dict[str, float],
        scan: LaserScan,
    ) -> FootprintCheck:
        """Check an action's refreshed scan against its swept capsule."""
        target_distance = math.hypot(
            target.target_x - pose["x"],
            target.target_y - pose["y"],
        )
        motion_yaw_in_base = norm_angle(
            target.target_yaw - pose["yaw_rad"]
        )
        return scan_capsule_footprint_check(
            scan,
            motion_yaw_in_base,
            target_distance + self.longitudinal_extra_margin_m,
            self.footprint_radius_m,
            self.laser_x_in_base_m,
            self.laser_y_in_base_m,
            self.laser_yaw_in_base,
        )

    def prepare_continuous_pre_motion_plan(
        self,
        raw_policy_action: int,
        q_ranked: list[tuple[int, float]],
        scan: LaserScan,
        odom: Odometry,
        requested_action: Optional[int] = None,
        allow_fallback: bool = True,
    ) -> ContinuousPreMotionPlan:
        """Re-gate ranked actions using refreshed sensors and path length."""
        pose = self.pose_record_from_odom(odom)
        primary_action = (
            raw_policy_action if requested_action is None
            else requested_action
        )
        ranked_actions = [primary_action]
        if allow_fallback:
            ranked_actions.extend(
                action for action, _value in q_ranked
                if action != primary_action
            )
        raw_check: Optional[FootprintCheck] = None
        raw_target: Optional[ActionExecutionTarget] = None
        raw_distance: Optional[float] = None
        raw_front_edge: Optional[float] = None

        for action_idx in ranked_actions:
            if action_idx not in self.allowed_actions:
                continue
            target = self.target_for_action(action_idx, pose["x"], pose["y"])
            distance = math.hypot(
                target.target_x - pose["x"],
                target.target_y - pose["y"],
            )
            front_edge = (
                distance
                + self.longitudinal_extra_margin_m
                + self.footprint_radius_m
            )
            footprint_check = self.pre_motion_footprint_check(
                target,
                pose,
                scan,
            )
            if action_idx == primary_action:
                raw_check = footprint_check
                raw_target = target
                raw_distance = distance
                raw_front_edge = front_edge
            if footprint_check.passed:
                return ContinuousPreMotionPlan(
                    raw_policy_action=raw_policy_action,
                    executed_action=action_idx,
                    target=target,
                    pre_motion_pose=pose,
                    target_distance=distance,
                    capsule_front_edge=front_edge,
                    nearest_capsule_clearance=(
                        footprint_check.nearest_capsule_clearance
                    ),
                    pre_motion_footprint_passed=True,
                    obstruction_type=(
                        raw_check.obstruction_type
                        if action_idx != primary_action
                        and raw_check is not None
                        else None
                    ),
                    gate_passed=True,
                    safety_fallback_used=action_idx != primary_action,
                )

        return ContinuousPreMotionPlan(
            raw_policy_action=raw_policy_action,
            executed_action=None,
            target=raw_target,
            pre_motion_pose=pose,
            target_distance=raw_distance,
            capsule_front_edge=raw_front_edge,
            nearest_capsule_clearance=(
                raw_check.nearest_capsule_clearance
                if raw_check is not None
                else None
            ),
            pre_motion_footprint_passed=False,
            obstruction_type=(
                raw_check.obstruction_type
                if raw_check is not None
                else "action_not_allowed"
            ),
            gate_passed=False,
            safety_fallback_used=False,
        )

    def prepare_local_escape_plan(
        self,
        q_ranked: list[tuple[int, float]],
        scan: LaserScan,
        odom: Odometry,
    ) -> LocalEscapePlan:
        """Select the longest safe deployment-only escape using Q rank."""
        pose = self.pose_record_from_odom(odom)
        evaluations: list[dict[str, Any]] = []
        ranked_actions = sorted(
            q_ranked,
            key=lambda item: (-float(item[1]), int(item[0])),
        )
        for distance_m in LOCAL_ESCAPE_CANDIDATE_DISTANCES_M:
            for action_idx, _q_value in ranked_actions:
                if action_idx not in self.allowed_actions:
                    continue
                target = self.target_for_action_distance(
                    action_idx,
                    pose["x"],
                    pose["y"],
                    distance_m,
                )
                check = self.pre_motion_footprint_check(target, pose, scan)
                evaluations.append(
                    {
                        "distance_m": distance_m,
                        "action_idx": action_idx,
                        "clearance": check.nearest_capsule_clearance,
                        "passed": check.passed,
                        "obstruction_type": check.obstruction_type,
                    }
                )
                if check.passed:
                    return LocalEscapePlan(
                        available=True,
                        action_idx=action_idx,
                        distance_m=distance_m,
                        target=target,
                        pre_motion_pose=pose,
                        nearest_clearance=check.nearest_capsule_clearance,
                        candidate_evaluations=evaluations,
                    )
        return LocalEscapePlan(
            available=False,
            action_idx=None,
            distance_m=None,
            target=None,
            pre_motion_pose=pose,
            nearest_clearance=None,
            candidate_evaluations=evaluations,
        )

    def execute_local_escape(
        self,
        plan: LocalEscapePlan,
        record: dict[str, Any],
        origin_state: tuple[int, int],
        consecutive_count: int,
    ) -> tuple[int, int]:
        """Execute one selected escape through the complete safety chain."""
        if plan.target is None or plan.action_idx is None:
            raise RuntimeError("local escape plan is unavailable")
        if self.cmd_pub.get_subscription_count() < 1:
            raise RuntimeError("No /cmd_vel subscriber found")
        record.update(
            {
                "local_escape_attempted": True,
                "consecutive_local_escape_count": consecutive_count,
            }
        )
        self.experiment_result["local_escape_total_count"] += 1
        self.experiment_result["local_escape_action_history"].append(
            plan.action_idx
        )
        self._last_dynamic_scan_sequence = self.scan_sequence
        final_dist = self.execute_target(plan.target)
        self.stop()
        try:
            _post_scan, post_odom = self.refresh_after_local_escape(record)
        except Exception as exc:
            record["local_escape_failure_reason"] = f"sensor_failure: {exc}"
            raise
        post_pose = self.pose_record_from_odom(post_odom)
        post_state = self.agent_state_from_odom(
            origin_state,
            post_pose["x"],
            post_pose["y"],
        )
        record.update(
            {
                "local_escape_success": True,
                "local_escape_post_pose": post_pose,
                "local_escape_post_agent_state": list(post_state),
                "local_escape_final_target_error": final_dist,
                "actual_grid_state": list(post_state),
                "grid_transition_match": None,
            }
        )
        self.experiment_result["local_escape_success_total_count"] += 1
        return post_state

    def publish_velocity(self, vx: float = 0.0, wz: float = 0.0) -> None:
        """Enforce the episode wall-clock limit before any non-zero command."""
        if (
            (vx != 0.0 or wz != 0.0)
            and hasattr(self, "max_runtime_sec")
            and time.monotonic() - self.experiment_started_monotonic
            >= self.max_runtime_sec
        ):
            super().publish_velocity()
            raise RuntimeError("max_runtime_reached")
        super().publish_velocity(vx, wz)

    def rotation_footprint_check(self, scan: LaserScan) -> FootprintCheck:
        """Check the full circular deployment envelope before rotation."""
        return scan_capsule_footprint_check(
            scan,
            0.0,
            0.0,
            self.footprint_radius_m,
            self.laser_x_in_base_m,
            self.laser_y_in_base_m,
            self.laser_yaw_in_base,
        )

    def check_rotation_footprint_safety(self) -> None:
        """Fail-stop if any fresh scan hit enters the rotation footprint."""
        with self._sensor_condition:
            scan = self.latest_scan
            scan_received_at = self.latest_scan_received_at
        self.require_fresh_sensor(
            "/scan",
            scan,
            scan_received_at,
            self.scan_timeout_sec,
        )
        result = (
            self.rotation_footprint_check(scan)
            if scan is not None
            else FootprintCheck(False, None, "invalid_scan", 0)
        )
        if self._dynamic_step_record is not None:
            self._dynamic_step_record["rotation_footprint_passed"] = (
                result.passed
            )
            self._dynamic_step_record[
                "rotation_nearest_footprint_clearance"
            ] = result.nearest_capsule_clearance
        if result.passed:
            return
        self.stop(repeat=3)
        raise RotationFootprintBlocked(
            "rotation_footprint_blocked: "
            "nearest_footprint_clearance="
            f"{format_optional_float(result.nearest_capsule_clearance)}m "
            f"obstruction_type={result.obstruction_type}"
        )

    def execute_target(self, target: ActionExecutionTarget) -> float:
        """Require a fresh all-around footprint pass before rotate phase."""
        self.check_rotation_footprint_safety()
        return super().execute_target(target)

    def check_drive_dynamic_safety(self) -> None:
        """Stop before drive when a fresh scan enters the short capsule."""
        with self._sensor_condition:
            if self.scan_sequence == self._last_dynamic_scan_sequence:
                return
            self._last_dynamic_scan_sequence = self.scan_sequence
            scan = self.latest_scan
            scan_received_at = self.latest_scan_received_at
        self.require_fresh_sensor(
            "/scan",
            scan,
            scan_received_at,
            self.scan_timeout_sec,
        )
        result = FootprintCheck(False, None, "invalid_scan", 0)
        if scan is not None:
            result = scan_capsule_footprint_check(
                scan,
                0.0,
                self.dynamic_forward_center_line_extension_m,
                self.footprint_radius_m,
                self.laser_x_in_base_m,
                self.laser_y_in_base_m,
                self.laser_yaw_in_base,
            )
        if self._dynamic_step_record is not None:
            self._dynamic_step_record["dynamic_nearest_capsule_clearance"] = (
                result.nearest_capsule_clearance
            )
        if not result.passed:
            if self._dynamic_step_record is not None:
                self._dynamic_step_record["dynamic_obstacle_stop"] = True
                self._dynamic_step_record["dynamic_footprint_stop"] = True
            self.stop(repeat=3)
            raise DynamicObstacleStop(
                "dynamic_footprint_stop: "
                "nearest_capsule_clearance="
                f"{format_optional_float(result.nearest_capsule_clearance)}m "
                f"obstruction_type={result.obstruction_type} "
                f"front_edge={self.dynamic_stop_distance:.3f}m"
            )

    @staticmethod
    def _maximum_optional_age(
        current: Optional[float],
        sample: Optional[float],
    ) -> Optional[float]:
        """Accumulate a maximum sensor age while preserving missing values."""
        if sample is None:
            return current
        if current is None:
            return sample
        return max(current, sample)

    def _record_drive_sensor_cycle(
        self,
        wait_duration: float,
        scan_advanced: bool,
        odom_advanced: bool,
        timed_out: bool,
    ) -> None:
        """Update compact drive-cycle sequence, wait, and age diagnostics."""
        record = self._dynamic_step_record
        if record is None:
            return
        ages = self.sensor_age_diagnostics()
        record["drive_sensor_cycle_count"] += 1
        record["drive_sensor_cycle_max_wait_sec"] = max(
            record["drive_sensor_cycle_max_wait_sec"],
            wait_duration,
        )
        record["drive_sensor_cycle_scan_advanced"] = scan_advanced
        record["drive_sensor_cycle_odom_advanced"] = odom_advanced
        record["drive_sensor_cycle_timeout"] = timed_out
        for source_key, target_key in (
            ("scan_receive_age_sec", "drive_max_scan_receive_age_sec"),
            ("scan_header_age_sec", "drive_max_scan_header_age_sec"),
            ("odom_receive_age_sec", "drive_max_odom_receive_age_sec"),
            ("odom_header_age_sec", "drive_max_odom_header_age_sec"),
        ):
            record[target_key] = self._maximum_optional_age(
                record[target_key],
                ages[source_key],
            )

    def wait_for_drive_sensor_cycle(
        self,
        previous_scan_sequence: int,
        previous_odom_sequence: int,
    ) -> None:
        """Require a fresh scan/odom pair before the next drive command."""
        started = time.monotonic()
        deadline = started + self.drive_sensor_cycle_timeout_sec
        scan_advanced = False
        odom_advanced = False
        pair_advanced = False

        with self._sensor_condition:
            while self.sensor_callbacks_active() and time.monotonic() < deadline:
                scan_advanced, odom_advanced, pair_advanced = (
                    drive_sensor_sequence_progress(
                        self.scan_sequence,
                        self.odom_sequence,
                        previous_scan_sequence,
                        previous_odom_sequence,
                    )
                )
                if pair_advanced:
                    break
                remaining = deadline - time.monotonic()
                self._sensor_condition.wait(
                    timeout=min(0.05, max(0.0, remaining))
                )
            else:
                pair_advanced = False

        if pair_advanced:
            self._record_drive_sensor_cycle(
                time.monotonic() - started,
                scan_advanced,
                odom_advanced,
                False,
            )
            self.require_fresh_inputs()
            self.check_drive_dynamic_safety()
            return

        wait_duration = time.monotonic() - started
        self._record_drive_sensor_cycle(
            wait_duration,
            scan_advanced,
            odom_advanced,
            True,
        )
        diagnostics = self.sensor_age_diagnostics()
        self.stop(repeat=3)
        raise DriveSensorCycleTimeout(
            "drive_sensor_cycle_timeout: "
            f"wait={wait_duration:.3f}s "
            f"scan_advanced={scan_advanced} "
            f"odom_advanced={odom_advanced}; "
            f"{self.format_sensor_age_diagnostics(diagnostics)}"
        )

    def sensor_callbacks_active(self) -> bool:
        """Return whether the background executor context remains active."""
        return rclpy.ok()

    def begin_step_record(self, step_id: int) -> tuple[dict[str, Any], float]:
        """Create a compact continuous-decision JSON record."""
        record, started = super().begin_step_record(step_id)
        record.update(
            {
                "timestamp": datetime.now()
                .astimezone()
                .isoformat(timespec="milliseconds"),
                "agent_state": None,
                "known_cells": None,
                "known_cells_delta": None,
                "frontier_count": None,
                "q_values": None,
                "q_ranked": None,
                "executed_action": None,
                "target_distance": None,
                "capsule_front_edge": None,
                "motion_clearance_margin": self.motion_clearance_margin,
                "nominal_min_corridor_width_m": (
                    self.nominal_min_corridor_width_m
                ),
                "footprint_radius_m": self.footprint_radius_m,
                "longitudinal_extra_margin_m": (
                    self.longitudinal_extra_margin_m
                ),
                "nearest_capsule_clearance": None,
                "pre_motion_footprint_passed": False,
                "requested_action_obstruction_type": None,
                "rotation_footprint_passed": None,
                "rotation_nearest_footprint_clearance": None,
                "gate_passed": False,
                "expected_grid_state": None,
                "actual_grid_state": None,
                "grid_transition_match": None,
                "dynamic_obstacle_stop": False,
                "dynamic_footprint_stop": False,
                "dynamic_nearest_capsule_clearance": None,
                "dynamic_stop_recovered": False,
                "dynamic_stop_recovery_index": None,
                "consecutive_dynamic_stop_count": 0,
                "post_dynamic_stop_pose": None,
                "post_dynamic_stop_agent_state": None,
                "recovery_scan_advanced": False,
                "recovery_odom_advanced": False,
                "recovery_refresh_duration_sec": None,
                "local_escape_attempted": False,
                "local_escape_available": False,
                "local_escape_action_idx": None,
                "local_escape_distance_m": None,
                "local_escape_pre_motion_clearance": None,
                "local_escape_success": False,
                "local_escape_failure_reason": None,
                "consecutive_local_escape_count": 0,
                "local_escape_post_pose": None,
                "local_escape_post_agent_state": None,
                "local_escape_refresh_scan_advanced": False,
                "local_escape_refresh_odom_advanced": False,
                "local_escape_refresh_duration_sec": None,
                "local_escape_candidate_evaluations": [],
                "local_escape_target": None,
                "local_escape_final_target_error": None,
                "drive_sensor_cycle_timeout_sec": (
                    self.drive_sensor_cycle_timeout_sec
                ),
                "drive_sensor_cycle_count": 0,
                "drive_sensor_cycle_max_wait_sec": 0.0,
                "drive_sensor_cycle_scan_advanced": None,
                "drive_sensor_cycle_odom_advanced": None,
                "drive_sensor_cycle_timeout": False,
                "drive_max_scan_receive_age_sec": None,
                "drive_max_scan_header_age_sec": None,
                "drive_max_odom_receive_age_sec": None,
                "drive_max_odom_header_age_sec": None,
                "step_success": False,
                "commissioning_mode": self.commissioning_mode,
                "commissioning_action_idx": self.commissioning_action_idx,
                "action_source": action_source_for_mode(
                    self.commissioning_mode
                ),
            }
        )
        return record, started

    def finish_step_record(
        self,
        record: dict[str, Any],
        started_monotonic: float,
        success: bool,
        failure_reason: Optional[str],
    ) -> None:
        """Finish the base record and expose the Round 8 field name."""
        super().finish_step_record(
            record,
            started_monotonic,
            success,
            failure_reason,
        )
        record["step_success"] = bool(record["success"])

    def finish_experiment(self, exit_reason: str) -> None:
        """Persist a bounded episode summary exactly once."""
        if self.result_write_attempted:
            return
        self.result_write_attempted = True
        steps = self.experiment_result["steps"]
        success = episode_success_for_reason(
            self.commissioning_mode,
            self.execute,
            exit_reason,
        )
        duration = time.monotonic() - self.experiment_started_monotonic
        self._episode_travel_distance = sum(
            float(step["actual_distance"])
            for step in steps
            if step.get("actual_distance") is not None
        )
        self.experiment_result.update(
            {
                "end_timestamp": datetime.now()
                .astimezone()
                .isoformat(timespec="milliseconds"),
                "total_steps": len(steps),
                "successful_steps": sum(
                    1 for step in steps if bool(step.get("step_success"))
                ),
                "travel_distance": self._episode_travel_distance,
                "episode_duration": duration,
                "termination_reason": exit_reason,
                "success": success,
            }
        )
        result_path: Optional[Path] = None
        try:
            result_path = self.save_experiment_result()
        except Exception as exc:
            self.get_logger().error(
                f"Failed to save continuous episode JSON: {exc}"
            )
        self.get_logger().warn(
            "continuous_episode_result "
            f"total_steps={len(steps)} "
            f"successful_steps={self.experiment_result['successful_steps']} "
            f"travel_distance={self._episode_travel_distance:.3f} "
            "dynamic_stop_total_count="
            f"{self.experiment_result['dynamic_stop_total_count']} "
            "dynamic_stop_recovery_total_count="
            f"{self.experiment_result['dynamic_stop_recovery_total_count']} "
            "dynamic_stop_deadlock="
            f"{self.experiment_result['dynamic_stop_deadlock']} "
            "local_escape_total_count="
            f"{self.experiment_result['local_escape_total_count']} "
            "local_escape_success_total_count="
            f"{self.experiment_result['local_escape_success_total_count']} "
            "local_escape_deadlock="
            f"{self.experiment_result['local_escape_deadlock']} "
            f"termination_reason={exit_reason} "
            f"success={str(success).lower()} "
            f"result_file={result_path or 'not_saved'}"
        )

    def _completion_enabled(self) -> bool:
        """Return whether belief completion may terminate this episode."""
        if self.commissioning_mode:
            return False
        return self.execute or not self.disable_completion_termination_in_dryrun

    def _belief_termination(
        self,
        step_count: int,
        known_cells: int,
        frontier_count: int,
        state_history: Sequence[tuple[int, int]],
        known_history: Sequence[int],
    ) -> Optional[str]:
        """Evaluate belief-only completion, stagnation, and deadlock."""
        if not self._completion_enabled():
            return None
        if frontier_exhausted(
            frontier_count,
            step_count,
            known_cells,
            self.minimum_completion_decision_steps,
            self.minimum_completion_known_cells,
        ):
            return "frontier_exhausted"
        if known_area_stagnated(
            known_history,
            self.stagnation_window_steps,
            self.stagnation_min_known_growth,
        ):
            return "stagnation"
        if repeated_state_deadlock(
            state_history,
            known_history,
            self.deadlock_window_steps,
            self.deadlock_maximum_unique_states,
            self.deadlock_min_known_growth,
        ):
            return "deadlock"
        return None

    @staticmethod
    def _sensor_failure(exc: Exception) -> bool:
        """Classify freshness and refresh failures for episode reporting."""
        text = str(exc).lower()
        return isinstance(exc, TimeoutError) or any(
            marker in text
            for marker in (
                "/scan",
                "/odom",
                "sensor",
                "timestamp is stale",
                "stopped updating",
            )
        )

    def _log_continuous_step(self, record: dict[str, Any]) -> None:
        """Emit one decision summary without high-frequency scan logging."""
        self.get_logger().warn(
            "continuous_step "
            f"step_id={record['step_id']} "
            f"agent_state={record['agent_state']} "
            f"known_cells={record['known_cells']} "
            f"known_cells_delta={record['known_cells_delta']} "
            f"frontier_count={record['frontier_count']} "
            f"raw_policy_action={record['raw_policy_action']} "
            f"action_source={record['action_source']} "
            f"executed_action={record['executed_action']} "
            f"safety_fallback_used={record['safety_fallback_used']} "
            f"target_distance={format_optional_float(record['target_distance'])} "
            "capsule_front_edge="
            f"{format_optional_float(record['capsule_front_edge'])} "
            "nearest_capsule_clearance="
            f"{format_optional_float(record['nearest_capsule_clearance'])} "
            "pre_motion_footprint_passed="
            f"{record['pre_motion_footprint_passed']} "
            "requested_action_obstruction_type="
            f"{record['requested_action_obstruction_type']} "
            "rotation_footprint_passed="
            f"{record['rotation_footprint_passed']} "
            "rotation_nearest_footprint_clearance="
            f"{format_optional_float(record['rotation_nearest_footprint_clearance'])} "
            f"gate_passed={record['gate_passed']} "
            f"grid_transition_match={record['grid_transition_match']} "
            f"dynamic_obstacle_stop={record['dynamic_obstacle_stop']} "
            f"dynamic_footprint_stop={record['dynamic_footprint_stop']} "
            "dynamic_nearest_capsule_clearance="
            f"{format_optional_float(record['dynamic_nearest_capsule_clearance'])} "
            "dynamic_stop_recovered="
            f"{record['dynamic_stop_recovered']} "
            "dynamic_stop_recovery_index="
            f"{record['dynamic_stop_recovery_index']} "
            "consecutive_dynamic_stop_count="
            f"{record['consecutive_dynamic_stop_count']} "
            "post_dynamic_stop_agent_state="
            f"{record['post_dynamic_stop_agent_state']} "
            f"post_dynamic_stop_pose={record['post_dynamic_stop_pose']} "
            "recovery_scan_advanced="
            f"{record['recovery_scan_advanced']} "
            "recovery_odom_advanced="
            f"{record['recovery_odom_advanced']} "
            "recovery_refresh_duration_sec="
            f"{format_optional_float(record['recovery_refresh_duration_sec'])} "
            "local_escape_used="
            f"{record['local_escape_attempted']} "
            "local_escape_available="
            f"{record['local_escape_available']} "
            "local_escape_action_idx="
            f"{record['local_escape_action_idx']} "
            "local_escape_distance_m="
            f"{format_optional_float(record['local_escape_distance_m'])} "
            "local_escape_pre_motion_clearance="
            f"{format_optional_float(record['local_escape_pre_motion_clearance'])} "
            "local_escape_success="
            f"{record['local_escape_success']} "
            "local_escape_failure_reason="
            f"{record['local_escape_failure_reason']} "
            "consecutive_local_escape_count="
            f"{record['consecutive_local_escape_count']} "
            "local_escape_post_agent_state="
            f"{record['local_escape_post_agent_state']} "
            "local_escape_refresh_scan_advanced="
            f"{record['local_escape_refresh_scan_advanced']} "
            "local_escape_refresh_odom_advanced="
            f"{record['local_escape_refresh_odom_advanced']} "
            "local_escape_refresh_duration_sec="
            f"{format_optional_float(record['local_escape_refresh_duration_sec'])} "
            "drive_sensor_cycle_count="
            f"{record['drive_sensor_cycle_count']} "
            "drive_sensor_cycle_max_wait_sec="
            f"{record['drive_sensor_cycle_max_wait_sec']:.3f} "
            "drive_sensor_cycle_timeout="
            f"{record['drive_sensor_cycle_timeout']} "
            "drive_sensor_cycle_scan_advanced="
            f"{record['drive_sensor_cycle_scan_advanced']} "
            "drive_sensor_cycle_odom_advanced="
            f"{record['drive_sensor_cycle_odom_advanced']} "
            "drive_max_scan_receive_age_sec="
            f"{format_optional_float(record['drive_max_scan_receive_age_sec'])} "
            "drive_max_scan_header_age_sec="
            f"{format_optional_float(record['drive_max_scan_header_age_sec'])} "
            "drive_max_odom_receive_age_sec="
            f"{format_optional_float(record['drive_max_odom_receive_age_sec'])} "
            "drive_max_odom_header_age_sec="
            f"{format_optional_float(record['drive_max_odom_header_age_sec'])} "
            f"step_success={record['step_success']} "
            f"failure_reason={record['failure_reason']}"
        )

    def run(self) -> str:
        """Run bounded continuous decision cycles until a recorded stop."""
        model_load_started = time.monotonic()
        model, adapter, torch = load_policy_model(self.checkpoint_path)
        self.experiment_result["model_load_duration_sec"] = (
            time.monotonic() - model_load_started
        )

        from env.core_cummap import CumulativeBeliefMap

        compatibility_true_grid = np.zeros((120, 120), dtype=np.int8)
        origin_state = (60, 60)
        recent_positions = [origin_state]
        state_history: list[tuple[int, int]] = []
        action_history: list[int] = []
        known_history: list[int] = []
        frontier_history: list[int] = []
        cum_map = None
        no_safe_action_count = 0
        consecutive_dynamic_stop_count = 0
        consecutive_local_escape_count = 0

        for step_id in range(self.max_steps):
            limit_reason = hard_limit_termination(
                step_id,
                self.max_steps,
                time.monotonic() - self.experiment_started_monotonic,
                self.max_runtime_sec,
            )
            if limit_reason is not None:
                self.stop()
                return limit_reason

            record, step_started = self.begin_step_record(step_id)
            self._dynamic_step_record = record
            try:
                scan, odom = self.consume_new_step_inputs()
                observation_pose = self.pose_record_from_odom(odom)
                record["observation_pose"] = observation_pose
                if self.odom_state_origin is None:
                    self.odom_state_origin = (
                        observation_pose["x"],
                        observation_pose["y"],
                    )
                agent_state = self.agent_state_from_odom(
                    origin_state,
                    observation_pose["x"],
                    observation_pose["y"],
                )
                record["agent_state"] = list(agent_state)

                state_started = time.monotonic()
                snap = self.build_local_snap(scan, odom)
                if cum_map is None:
                    cum_map = CumulativeBeliefMap(
                        compatibility_true_grid,
                        agent_state,
                        snap,
                    )
                else:
                    cum_map.update(agent_state, snap)
                known_cells, frontier_count = belief_statistics(cum_map)
                known_delta = (
                    known_cells if not known_history
                    else known_cells - known_history[-1]
                )
                record.update(
                    {
                        "known_cells": known_cells,
                        "known_cells_delta": known_delta,
                        "frontier_count": frontier_count,
                    }
                )
                state_history.append(agent_state)
                known_history.append(known_cells)
                frontier_history.append(frontier_count)
                self.experiment_result["agent_state_history"] = [
                    list(state) for state in state_history
                ]
                self.experiment_result["executed_action_history"] = (
                    action_history
                )
                self.experiment_result["known_area_history"] = known_history
                self.experiment_result["frontier_count_history"] = (
                    frontier_history
                )

                termination = self._belief_termination(
                    step_id + 1,
                    known_cells,
                    frontier_count,
                    state_history,
                    known_history,
                )
                if termination is not None:
                    record["policy_state_build_duration_sec"] = (
                        time.monotonic() - state_started
                    )
                    self.stop()
                    self.finish_step_record(
                        record,
                        step_started,
                        termination == "frontier_exhausted",
                        None if termination == "frontier_exhausted" else termination,
                    )
                    self._log_continuous_step(record)
                    return termination

                state_batch, _state_meta = adapter.build_single_state_tensors(
                    cum_map,
                    agent_state,
                    recent_trajectory_positions=recent_positions,
                    return_state_meta=True,
                )
                record["policy_state_build_duration_sec"] = (
                    time.monotonic() - state_started
                )
                inference_started = time.monotonic()
                with torch.inference_mode():
                    q_values = model(**state_batch, return_aux=False)
                record["policy_inference_duration_sec"] = (
                    time.monotonic() - inference_started
                )
                barrier = self.capture_sensor_refresh_barrier()
                q_np = q_values.detach().cpu().numpy()[0]
                q_list = [round(float(value), 4) for value in q_np.tolist()]
                q_ranked = sorted(
                    list(enumerate(q_list)),
                    key=lambda item: item[1],
                    reverse=True,
                )
                raw_action = int(torch.argmax(q_values, dim=1).item())
                requested_action = (
                    self.commissioning_action_idx
                    if self.commissioning_mode
                    else raw_action
                )
                record.update(
                    {
                        "q_values": q_list,
                        "q_ranked": q_ranked,
                        "raw_policy_action": raw_action,
                        "pre_motion_requested_action": requested_action,
                    }
                )

                refresh_started = time.monotonic()
                try:
                    refreshed_scan, refreshed_odom = (
                        self.refresh_inputs_after_barrier(barrier)
                    )
                finally:
                    record["pre_motion_refresh_duration_sec"] = (
                        time.monotonic() - refresh_started
                    )
                record.update(self.sensor_age_diagnostics())
                plan = self.prepare_continuous_pre_motion_plan(
                    raw_action,
                    q_ranked,
                    refreshed_scan,
                    refreshed_odom,
                    requested_action=requested_action,
                    allow_fallback=not self.commissioning_mode,
                )
                record.update(
                    {
                        "pre_motion_pose": plan.pre_motion_pose,
                        "start_pose": plan.pre_motion_pose,
                        "executed_action": plan.executed_action,
                        "pre_motion_executed_action": plan.executed_action,
                        "safety_fallback_used": plan.safety_fallback_used,
                        "target_distance": plan.target_distance,
                        "capsule_front_edge": plan.capsule_front_edge,
                        "nearest_capsule_clearance": (
                            plan.nearest_capsule_clearance
                        ),
                        "pre_motion_footprint_passed": (
                            plan.pre_motion_footprint_passed
                        ),
                        "requested_action_obstruction_type": (
                            plan.obstruction_type
                        ),
                        "gate_passed": plan.gate_passed,
                    }
                )

                if not plan.gate_passed or plan.target is None:
                    if self.commissioning_mode:
                        self.stop()
                        self.finish_step_record(
                            record,
                            step_started,
                            False,
                            "commissioning_action_blocked",
                        )
                        self._log_continuous_step(record)
                        return "commissioning_action_blocked"

                    normal_plan_unavailable = (
                        not plan.gate_passed
                        and plan.executed_action is None
                    )
                    if (
                        normal_plan_unavailable
                        and consecutive_local_escape_count
                        >= self.local_escape_recovery_limit
                    ):
                        self.stop()
                        record.update(
                            {
                                "consecutive_local_escape_count": (
                                    consecutive_local_escape_count
                                ),
                                "local_escape_failure_reason": (
                                    "local_escape_deadlock"
                                ),
                            }
                        )
                        self.experiment_result["local_escape_deadlock"] = True
                        self.finish_step_record(
                            record,
                            step_started,
                            False,
                            "local_escape_deadlock",
                        )
                        self._log_continuous_step(record)
                        return "local_escape_deadlock"

                    escape_plan: Optional[LocalEscapePlan] = None
                    if normal_plan_unavailable:
                        escape_plan = self.prepare_local_escape_plan(
                            q_ranked,
                            refreshed_scan,
                            refreshed_odom,
                        )
                        record.update(
                            {
                                "local_escape_available": (
                                    escape_plan.available
                                ),
                                "local_escape_action_idx": (
                                    escape_plan.action_idx
                                ),
                                "local_escape_distance_m": (
                                    escape_plan.distance_m
                                ),
                                "local_escape_pre_motion_clearance": (
                                    escape_plan.nearest_clearance
                                ),
                                "local_escape_candidate_evaluations": (
                                    escape_plan.candidate_evaluations
                                ),
                                "consecutive_local_escape_count": (
                                    consecutive_local_escape_count
                                ),
                                "local_escape_target": (
                                    escape_plan.target.as_dict()
                                    if escape_plan.target is not None
                                    else None
                                ),
                            }
                        )

                    if (
                        escape_plan is not None
                        and escape_plan.available
                        and escape_plan.target is not None
                        and self.execute
                    ):
                        consecutive_local_escape_count += 1
                        post_escape_state = self.execute_local_escape(
                            escape_plan,
                            record,
                            origin_state,
                            consecutive_local_escape_count,
                        )
                        recent_positions.append(post_escape_state)
                        recent_positions = recent_positions[-8:]
                        no_safe_action_count = 0
                        consecutive_dynamic_stop_count = 0
                        record["consecutive_dynamic_stop_count"] = 0
                        self.finish_step_record(
                            record,
                            step_started,
                            False,
                            "no_safe_action_recovered",
                        )
                        self._log_continuous_step(record)
                        continue

                    if escape_plan is not None and escape_plan.available:
                        record["local_escape_failure_reason"] = (
                            "execution_disabled"
                        )
                    self.stop()
                    no_safe_action_count += 1
                    self.finish_step_record(
                        record,
                        step_started,
                        False,
                        "no_safe_action",
                    )
                    self._log_continuous_step(record)
                    if no_safe_action_count > self.no_safe_action_retries:
                        return "no_safe_action"
                    self.last_consumed_scan_sequence = self.scan_sequence
                    self.last_consumed_odom_sequence = self.odom_sequence
                    continue

                no_safe_action_count = 0
                self.set_step_target(record, plan.target)
                expected_state = expected_grid_state_from_action(
                    agent_state,
                    plan.executed_action,
                )
                record["expected_grid_state"] = list(expected_state)
                final_dist: Optional[float] = None

                if motion_is_permitted(
                    self.execute,
                    plan.gate_passed,
                    plan.executed_action,
                ):
                    if self.cmd_pub.get_subscription_count() < 1:
                        raise RuntimeError("No /cmd_vel subscriber found")
                    self._last_dynamic_scan_sequence = self.scan_sequence
                    final_dist = self.execute_target(plan.target)
                    self.stop()
                    self.refresh_after_motion()
                else:
                    self.stop()
                    self.last_consumed_scan_sequence = self.scan_sequence
                    self.last_consumed_odom_sequence = self.odom_sequence

                end_x, end_y, _end_yaw, _end_stamp = (
                    self.pose_xy_yaw_time()
                )
                actual_state = self.agent_state_from_odom(
                    origin_state,
                    end_x,
                    end_y,
                )
                record["actual_grid_state"] = list(actual_state)
                record["grid_transition_match"] = grid_transition_matches(
                    expected_state,
                    actual_state,
                )
                record["final_target_error"] = final_dist
                action_history.append(plan.executed_action)
                recent_positions.append(actual_state)
                recent_positions = recent_positions[-8:]
                consecutive_dynamic_stop_count = 0
                record["consecutive_dynamic_stop_count"] = 0
                consecutive_local_escape_count = 0
                record["consecutive_local_escape_count"] = 0
                self.finish_step_record(record, step_started, True, None)
                self._log_continuous_step(record)
                successful_termination = successful_step_termination_reason(
                    self.commissioning_mode,
                    self.execute,
                )
                if successful_termination is not None:
                    return successful_termination

            except KeyboardInterrupt:
                self.stop()
                record.update(self.sensor_age_diagnostics())
                self.finish_step_record(
                    record,
                    step_started,
                    False,
                    "operator_interrupt",
                )
                raise
            except DynamicObstacleStop as exc:
                self.stop(repeat=3)
                consecutive_dynamic_stop_count += 1
                self.experiment_result["dynamic_stop_total_count"] += 1
                record.update(
                    {
                        "dynamic_stop_recovery_index": (
                            consecutive_dynamic_stop_count
                        ),
                        "consecutive_dynamic_stop_count": (
                            consecutive_dynamic_stop_count
                        ),
                    }
                )
                recovery_barrier = self.capture_sensor_refresh_barrier()
                recovery_started = time.monotonic()
                recovery_error: Optional[Exception] = None
                recovered_odom: Optional[Odometry] = None
                try:
                    _recovered_scan, recovered_odom = (
                        self.refresh_inputs_after_barrier(recovery_barrier)
                    )
                except Exception as refresh_exc:
                    recovery_error = refresh_exc
                finally:
                    record["recovery_refresh_duration_sec"] = (
                        time.monotonic() - recovery_started
                    )
                    recovery_diagnostics = self.sensor_age_diagnostics()
                    record.update(recovery_diagnostics)
                    record["recovery_scan_advanced"] = (
                        int(recovery_diagnostics["scan_sequence"])
                        > recovery_barrier.scan_sequence
                    )
                    record["recovery_odom_advanced"] = (
                        int(recovery_diagnostics["odom_sequence"])
                        > recovery_barrier.odom_sequence
                    )

                if recovery_error is not None:
                    self.stop(repeat=3)
                    if record["local_escape_attempted"]:
                        record["local_escape_failure_reason"] = (
                            "dynamic_stop_recovery_sensor_failure: "
                            f"{recovery_error}"
                        )
                    self.finish_step_record(
                        record,
                        step_started,
                        False,
                        f"sensor_failure: {recovery_error}",
                    )
                    self._log_continuous_step(record)
                    self.get_logger().error(
                        "dynamic-stop recovery sensor failure: "
                        f"{recovery_error}"
                    )
                    return "sensor_failure"

                if recovered_odom is None:
                    self.stop(repeat=3)
                    self.finish_step_record(
                        record,
                        step_started,
                        False,
                        "sensor_failure: missing recovered /odom",
                    )
                    self._log_continuous_step(record)
                    return "sensor_failure"

                post_stop_pose = self.pose_record_from_odom(recovered_odom)
                post_stop_state = self.agent_state_from_odom(
                    origin_state,
                    post_stop_pose["x"],
                    post_stop_pose["y"],
                )
                record.update(
                    {
                        "post_dynamic_stop_pose": post_stop_pose,
                        "post_dynamic_stop_agent_state": list(post_stop_state),
                        "actual_grid_state": list(post_stop_state),
                        "grid_transition_match": (
                            grid_transition_matches(
                                tuple(record["expected_grid_state"]),
                                post_stop_state,
                            )
                            if record["expected_grid_state"] is not None
                            else None
                        ),
                    }
                )
                if record["local_escape_attempted"]:
                    record.update(
                        {
                            "local_escape_failure_reason": (
                                "dynamic_obstacle_stop"
                            ),
                            "local_escape_post_pose": post_stop_pose,
                            "local_escape_post_agent_state": list(
                                post_stop_state
                            ),
                        }
                    )
                recent_positions.append(post_stop_state)
                recent_positions = recent_positions[-8:]

                if (
                    consecutive_dynamic_stop_count
                    >= self.dynamic_stop_recovery_limit
                ):
                    self.experiment_result["dynamic_stop_deadlock"] = True
                    self.finish_step_record(
                        record,
                        step_started,
                        False,
                        "dynamic_stop_deadlock",
                    )
                    self._log_continuous_step(record)
                    self.get_logger().error(
                        f"{exc}; dynamic-stop recovery limit reached"
                    )
                    return "dynamic_stop_deadlock"

                record["dynamic_stop_recovered"] = True
                self.experiment_result[
                    "dynamic_stop_recovery_total_count"
                ] += 1
                self.finish_step_record(
                    record,
                    step_started,
                    False,
                    "dynamic_obstacle_stop_recovered",
                )
                self._log_continuous_step(record)
                self.get_logger().warn(
                    f"{exc}; abandoning target and replanning from "
                    f"post-stop agent_state={post_stop_state}"
                )
                continue
            except RotationFootprintBlocked as exc:
                self.stop()
                record.update(self.sensor_age_diagnostics())
                if record["local_escape_attempted"]:
                    record["local_escape_failure_reason"] = (
                        "rotation_footprint_blocked"
                    )
                self.finish_step_record(
                    record,
                    step_started,
                    False,
                    "rotation_footprint_blocked",
                )
                self._log_continuous_step(record)
                self.get_logger().error(str(exc))
                return "rotation_footprint_blocked"
            except DriveSensorCycleTimeout as exc:
                self.stop()
                record.update(self.sensor_age_diagnostics())
                if record["local_escape_attempted"]:
                    record["local_escape_failure_reason"] = (
                        "drive_sensor_cycle_timeout"
                    )
                self.finish_step_record(
                    record,
                    step_started,
                    False,
                    "drive_sensor_cycle_timeout",
                )
                self._log_continuous_step(record)
                self.get_logger().error(str(exc))
                return "drive_sensor_cycle_timeout"
            except Exception as exc:
                self.stop()
                record.update(self.sensor_age_diagnostics())
                reason = (
                    "sensor_failure"
                    if self._sensor_failure(exc)
                    else "motion_failure"
                )
                if str(exc) == "max_runtime_reached":
                    reason = "max_runtime_reached"
                if (
                    record["local_escape_attempted"]
                    and record["local_escape_failure_reason"] is None
                ):
                    record["local_escape_failure_reason"] = (
                        f"{reason}: {exc}"
                    )
                self.finish_step_record(
                    record,
                    step_started,
                    False,
                    f"{reason}: {exc}",
                )
                self._log_continuous_step(record)
                self.get_logger().error(
                    f"continuous step {step_id} aborted: {reason}: {exc}"
                )
                return reason
            finally:
                self._dynamic_step_record = None

        self.stop()
        return hard_limit_termination(
            self.max_steps,
            self.max_steps,
            time.monotonic() - self.experiment_started_monotonic,
            self.max_runtime_sec,
        ) or "max_steps_reached"


def main(args=None) -> None:
    """Run the guarded continuous node and always issue a final stop."""
    rclpy.init(args=args)
    node = RealcarPolicyContinuousRunner()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor_thread = threading.Thread(
        target=executor.spin,
        name="continuous_sensor_executor",
        daemon=True,
    )
    executor_thread.start()
    exit_reason = "unknown"
    try:
        exit_reason = node.run()
    except KeyboardInterrupt:
        exit_reason = "operator_interrupt"
        node.get_logger().warn(
            "KeyboardInterrupt received; sending stop command."
        )
    except Exception as exc:
        exit_reason = f"error: {exc}"
        node.get_logger().error(f"FAIL: {exc}")
    finally:
        try:
            node.stop()
            node.finish_experiment(exit_reason)
            node.get_logger().warn(f"node_exit_reason={exit_reason}")
        finally:
            executor.shutdown(timeout_sec=2.0)
            executor_thread.join(timeout=2.0)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
