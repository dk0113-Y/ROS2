"""
Guarded Round 8 continuous real-car exploration runner.

The real-world completion criteria in this module are belief-side only.  The
zero-valued ``true_grid`` passed to ``CumulativeBeliefMap`` is a constructor
compatibility placeholder and is never used for coverage or termination.
"""

from __future__ import annotations

import math
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from drl_explore_bridge.realcar_action_adapter import ActionExecutionTarget
from drl_explore_bridge.realcar_policy_safe_runner_node import (
    ACTIONS_8,
    INVISIBLE,
    RealcarPolicySafeRunner,
    format_optional_float,
    load_policy_model,
    odom_delta_to_grid_offset,
)


CONTINUOUS_CELL_SIZE_M = 0.35
CONTINUOUS_DEFAULT_MAX_STEPS = 30
CONTINUOUS_MAX_STEPS_LIMIT = 1000
DEFAULT_MOTION_CLEARANCE_MARGIN_M = 0.25
DEFAULT_DYNAMIC_STOP_DISTANCE_M = 0.25


class DynamicObstacleStop(RuntimeError):
    """Signal that a fresh drive-phase scan requires an immediate stop."""


@dataclass(frozen=True)
class ContinuousPreMotionPlan:
    """Describe the final action and distance-aware pre-motion gate result."""

    raw_policy_action: int
    executed_action: Optional[int]
    target: Optional[ActionExecutionTarget]
    pre_motion_pose: dict[str, float]
    target_distance: Optional[float]
    observed_clearance: Optional[float]
    required_clearance: Optional[float]
    gate_passed: bool
    safety_fallback_used: bool


def required_motion_clearance(
    target_distance: float,
    existing_minimum_clearance: float,
    motion_clearance_margin: float,
) -> float:
    """Return the target-length-aware clearance threshold in metres."""
    if target_distance < 0.0:
        raise ValueError("target_distance must be >= 0")
    if existing_minimum_clearance <= 0.0:
        raise ValueError("existing_minimum_clearance must be > 0")
    if motion_clearance_margin < 0.0:
        raise ValueError("motion_clearance_margin must be >= 0")
    return max(
        existing_minimum_clearance,
        target_distance + motion_clearance_margin,
    )


def clearance_gate_passed(
    observed_clearance: Optional[float],
    required_clearance: float,
) -> bool:
    """Return whether a valid observed range meets the required clearance."""
    return (
        observed_clearance is not None
        and math.isfinite(observed_clearance)
        and observed_clearance >= required_clearance
    )


def dynamic_obstacle_should_stop(
    observed_clearance: Optional[float],
    emergency_stop_distance: float,
) -> bool:
    """Fail safe when a fresh forward sector is invalid or too close."""
    if emergency_stop_distance <= 0.0:
        raise ValueError("emergency_stop_distance must be > 0")
    return (
        observed_clearance is None
        or not math.isfinite(observed_clearance)
        or observed_clearance < emergency_stop_distance
    )


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
) -> Optional[str]:
    """Return the dedicated completion reason for a successful commission."""
    if commissioning_mode:
        return "commissioning_complete"
    return None


def episode_success_for_reason(
    commissioning_mode: bool,
    exit_reason: str,
) -> bool:
    """Classify normal exploration and commissioning success separately."""
    if commissioning_mode:
        return exit_reason == "commissioning_complete"
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

    def __init__(self) -> None:
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
        self.declare_parameter("minimum_completion_decision_steps", 3)
        self.declare_parameter("minimum_completion_known_cells", 20)
        self.declare_parameter("stagnation_window_steps", 10)
        self.declare_parameter("stagnation_min_known_growth", 1)
        self.declare_parameter("deadlock_window_steps", 8)
        self.declare_parameter("deadlock_maximum_unique_states", 2)
        self.declare_parameter("deadlock_min_known_growth", 1)
        self.declare_parameter("no_safe_action_retries", 0)
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
                "disable_completion_termination_in_dryrun": (
                    self.disable_completion_termination_in_dryrun
                ),
            },
            "motion_clearance_margin": self.motion_clearance_margin,
            "dynamic_stop_distance": self.dynamic_stop_distance,
            "total_steps": 0,
            "successful_steps": 0,
            "travel_distance": 0.0,
            "episode_duration": 0.0,
            "termination_reason": "not_started",
            "success": False,
            "steps": [],
        }
        self.get_logger().warn(
            "Round 8 continuous configuration "
            f"cell_size={self.cell_size:.3f} "
            f"step_distance={self.step_distance:.3f} "
            "cardinal_distance=0.350 "
            f"diagonal_distance={math.sqrt(2.0) * self.cell_size:.3f} "
            f"motion_clearance_margin={self.motion_clearance_margin:.3f} "
            "margin_status=unvalidated_engineering_safety_margin "
            f"dynamic_stop_distance={self.dynamic_stop_distance:.3f} "
            f"commissioning_mode={self.commissioning_mode} "
            f"commissioning_action_idx={self.commissioning_action_idx} "
            f"action_source={action_source_for_mode(self.commissioning_mode)} "
            "completion_source=belief_only"
        )

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
        raw_observed: Optional[float] = None
        raw_target: Optional[ActionExecutionTarget] = None
        raw_distance: Optional[float] = None
        raw_required: Optional[float] = None

        for action_idx in ranked_actions:
            if action_idx not in self.allowed_actions:
                continue
            target = self.target_for_action(action_idx, pose["x"], pose["y"])
            distance = math.hypot(
                target.target_x - pose["x"],
                target.target_y - pose["y"],
            )
            required = required_motion_clearance(
                distance,
                self.min_sector_dist,
                self.motion_clearance_margin,
            )
            observed = self.action_sector_min(action_idx, scan, odom)
            if action_idx == primary_action:
                raw_observed = observed
                raw_target = target
                raw_distance = distance
                raw_required = required
            if clearance_gate_passed(observed, required):
                return ContinuousPreMotionPlan(
                    raw_policy_action=raw_policy_action,
                    executed_action=action_idx,
                    target=target,
                    pre_motion_pose=pose,
                    target_distance=distance,
                    observed_clearance=observed,
                    required_clearance=required,
                    gate_passed=True,
                    safety_fallback_used=action_idx != primary_action,
                )

        return ContinuousPreMotionPlan(
            raw_policy_action=raw_policy_action,
            executed_action=None,
            target=raw_target,
            pre_motion_pose=pose,
            target_distance=raw_distance,
            observed_clearance=raw_observed,
            required_clearance=raw_required,
            gate_passed=False,
            safety_fallback_used=False,
        )

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

    def check_drive_dynamic_safety(self) -> None:
        """Stop before the next drive command when a fresh scan is unsafe."""
        if self.scan_sequence == self._last_dynamic_scan_sequence:
            return
        self._last_dynamic_scan_sequence = self.scan_sequence
        self.require_fresh_sensor(
            "/scan",
            self.latest_scan,
            self.latest_scan_received_at,
            self.scan_timeout_sec,
        )
        observed = None
        if self.latest_scan is not None:
            observed = self.sector_min_dist(
                self.latest_scan,
                0.0,
                math.radians(22.5),
            )
        if self._dynamic_step_record is not None:
            self._dynamic_step_record["dynamic_observed_clearance"] = observed
        if dynamic_obstacle_should_stop(
            observed,
            self.dynamic_stop_distance,
        ):
            if self._dynamic_step_record is not None:
                self._dynamic_step_record["dynamic_obstacle_stop"] = True
            self.stop(repeat=3)
            raise DynamicObstacleStop(
                "dynamic_obstacle_stop: "
                f"observed_clearance={format_optional_float(observed)}m "
                f"< threshold={self.dynamic_stop_distance:.3f}m"
            )

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
                "observed_clearance": None,
                "required_clearance": None,
                "motion_clearance_margin": self.motion_clearance_margin,
                "gate_passed": False,
                "expected_grid_state": None,
                "actual_grid_state": None,
                "grid_transition_match": None,
                "dynamic_obstacle_stop": False,
                "dynamic_observed_clearance": None,
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
            "required_clearance="
            f"{format_optional_float(record['required_clearance'])} "
            "observed_clearance="
            f"{format_optional_float(record['observed_clearance'])} "
            f"gate_passed={record['gate_passed']} "
            f"grid_transition_match={record['grid_transition_match']} "
            f"dynamic_obstacle_stop={record['dynamic_obstacle_stop']} "
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
                        "observed_clearance": plan.observed_clearance,
                        "required_clearance": plan.required_clearance,
                        "gate_passed": plan.gate_passed,
                    }
                )

                if not plan.gate_passed or plan.target is None:
                    self.stop()
                    no_safe_action_count += 1
                    blocked_reason = (
                        "commissioning_action_blocked"
                        if self.commissioning_mode
                        else "no_safe_action"
                    )
                    self.finish_step_record(
                        record,
                        step_started,
                        False,
                        blocked_reason,
                    )
                    self._log_continuous_step(record)
                    if self.commissioning_mode:
                        return blocked_reason
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
                self.finish_step_record(record, step_started, True, None)
                self._log_continuous_step(record)
                successful_termination = successful_step_termination_reason(
                    self.commissioning_mode
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
                self.stop()
                record.update(self.sensor_age_diagnostics())
                self.finish_step_record(
                    record,
                    step_started,
                    False,
                    "dynamic_obstacle_stop",
                )
                self._log_continuous_step(record)
                self.get_logger().error(str(exc))
                return "dynamic_obstacle_stop"
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
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
