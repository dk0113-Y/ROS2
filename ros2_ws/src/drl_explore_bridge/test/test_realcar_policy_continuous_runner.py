"""Targeted unit tests for the guarded Round 8 continuous runner."""

import math
import threading

import numpy as np
import pytest
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from drl_explore_bridge.realcar_action_adapter import RealcarActionAdapter
from drl_explore_bridge.realcar_policy_continuous_runner_node import (
    CONTINUOUS_CELL_SIZE_M,
    CONTINUOUS_DEFAULT_MAX_STEPS,
    DEFAULT_FOOTPRINT_RADIUS_M,
    DEFAULT_LASER_X_IN_BASE_M,
    DEFAULT_LASER_Y_IN_BASE_M,
    DEFAULT_LONGITUDINAL_EXTRA_MARGIN_M,
    DEFAULT_NOMINAL_MIN_CORRIDOR_WIDTH_M,
    ContinuousPreMotionPlan,
    DriveSensorCycleTimeout,
    DynamicObstacleStop,
    FootprintCheck,
    RealcarPolicyContinuousRunner,
    RotationFootprintBlocked,
    action_source_for_mode,
    belief_statistics,
    capsule_footprint_check,
    drive_sensor_sequence_progress,
    episode_success_for_reason,
    expected_grid_state_from_action,
    frontier_exhausted,
    grid_transition_matches,
    hard_limit_termination,
    known_area_stagnated,
    motion_is_permitted,
    repeated_state_deadlock,
    scan_capsule_footprint_check,
    scan_points_in_base,
    successful_step_termination_reason,
    validate_commissioning_config,
)
from drl_explore_bridge.realcar_policy_safe_runner_node import (
    ACTIONS_8,
    ACTION_NAMES,
    SensorRefreshBarrier,
    create_sensor_callback_groups,
    odom_delta_to_grid_offset,
    sensor_freshness_error,
    sensor_sample_is_after_barrier,
)


def make_odom(x=0.0, y=0.0, yaw=0.0):
    """Create a minimal stamped odometry message."""
    odom = Odometry()
    odom.pose.pose.position.x = x
    odom.pose.pose.position.y = y
    odom.pose.pose.orientation.z = math.sin(yaw / 2.0)
    odom.pose.pose.orientation.w = math.cos(yaw / 2.0)
    odom.header.stamp.sec = 100
    return odom


def make_scan(distance):
    """Create a full-circle scan with one uniform valid distance."""
    scan = LaserScan()
    scan.angle_min = -math.pi
    scan.angle_increment = math.pi / 180.0
    scan.range_min = 0.05
    scan.range_max = 10.0
    scan.ranges = [float(distance)] * 360
    scan.header.stamp.sec = 100
    return scan


def make_single_hit_scan(distance, angle=0.0):
    """Create a scan containing one valid hit at an exact angle."""
    scan = LaserScan()
    scan.angle_min = angle
    scan.angle_increment = 1.0
    scan.range_min = 0.05
    scan.range_max = 10.0
    scan.ranges = [float(distance)]
    scan.header.stamp.sec = 100
    return scan


class PlanHarness:
    """Supply only the action and scan methods needed by plan selection."""

    def __init__(self, distances):
        self.allowed_actions = set(range(8))
        self.action_adapter = RealcarActionAdapter(
            ACTIONS_8,
            ACTION_NAMES,
            diagonal_mode="grid_center",
        )
        self.cell_size = CONTINUOUS_CELL_SIZE_M
        self.min_sector_dist = 0.25
        self.motion_clearance_margin = 0.25
        self.footprint_radius_m = DEFAULT_FOOTPRINT_RADIUS_M
        self.longitudinal_extra_margin_m = (
            DEFAULT_LONGITUDINAL_EXTRA_MARGIN_M
        )
        self.distances = distances
        self.action_yaws = {
            self.target_for_action(index, 0.0, 0.0).target_yaw: index
            for index in range(8)
        }

    def pose_record_from_odom(self, odom):
        """Reuse the production pose conversion."""
        return RealcarPolicyContinuousRunner.pose_record_from_odom(odom)

    def target_for_action(self, action_idx, x0, y0):
        """Build the same grid-center target as the production runner."""
        return self.action_adapter.target_for_action(
            action_idx,
            start_x=x0,
            start_y=y0,
            step_distance=self.cell_size,
        )

    def pre_motion_footprint_check(self, target, pose, scan):
        """Return deterministic per-action capsule results for plan tests."""
        del scan
        action_idx = self.action_yaws[target.target_yaw]
        target_distance = math.hypot(
            target.target_x - pose["x"],
            target.target_y - pose["y"],
        )
        front_edge = (
            target_distance
            + self.longitudinal_extra_margin_m
            + self.footprint_radius_m
        )
        clearance = self.distances[action_idx] - front_edge
        passed = clearance >= 0.0
        return FootprintCheck(
            passed,
            clearance,
            None if passed else "longitudinal_path_obstruction",
            1,
        )


class BeliefHarness:
    """Expose a small belief and frontier mask."""

    def __init__(self):
        self.map = np.array([[-1, 0], [1, 0]], dtype=np.int8)

    def get_frontier_u8(self):
        """Return two belief-derived frontier cells."""
        return np.array([[0, 1], [0, 1]], dtype=np.uint8)


class DynamicHarness:
    """Capture the immediate-stop behavior without constructing a ROS node."""

    def __init__(self, distance, angle=0.0):
        self._sensor_condition = threading.Condition()
        self.scan_sequence = 2
        self._last_dynamic_scan_sequence = 1
        self.latest_scan = make_single_hit_scan(distance, angle)
        self.latest_scan_received_at = 1.0
        self.scan_timeout_sec = 0.5
        self.dynamic_stop_distance = 0.25
        self.footprint_radius_m = DEFAULT_FOOTPRINT_RADIUS_M
        self.dynamic_forward_center_line_extension_m = 0.05
        self.laser_x_in_base_m = 0.0
        self.laser_y_in_base_m = 0.0
        self.laser_yaw_in_base = 0.0
        self._dynamic_step_record = {
            "dynamic_obstacle_stop": False,
            "dynamic_footprint_stop": False,
        }
        self.stop_calls = 0

    def require_fresh_sensor(self, *args):
        """Treat the fabricated scan as fresh."""
        del args

    def stop(self, repeat=10):
        """Record the zero-command stop request."""
        del repeat
        self.stop_calls += 1


class DriveCycleHarness:
    """Exercise the continuous drive sensor barrier without a ROS executor."""

    def __init__(self, advance_scan=False, advance_odom=False, stale=False):
        self._sensor_condition = threading.Condition()
        self.scan_sequence = 10
        self.odom_sequence = 20
        self.advance_scan = advance_scan
        self.advance_odom = advance_odom
        self.stale = stale
        self.drive_sensor_cycle_timeout_sec = 0.02
        self.stop_calls = 0
        self.freshness_checks = 0
        self.dynamic_checks = 0
        self._dynamic_step_record = {
            "drive_sensor_cycle_count": 0,
            "drive_sensor_cycle_max_wait_sec": 0.0,
            "drive_sensor_cycle_scan_advanced": None,
            "drive_sensor_cycle_odom_advanced": None,
            "drive_sensor_cycle_timeout": False,
            "drive_max_scan_receive_age_sec": None,
            "drive_max_scan_header_age_sec": None,
            "drive_max_odom_receive_age_sec": None,
            "drive_max_odom_header_age_sec": None,
        }

    def sensor_callbacks_active(self):
        """Keep the deterministic short watchdog active."""
        return True

    def start_background_updates(self):
        """Emulate independently executing subscription callbacks."""
        def update():
            with self._sensor_condition:
                if self.advance_scan:
                    self.scan_sequence += 1
                    self.advance_scan = False
                if self.advance_odom:
                    self.odom_sequence += 1
                    self.advance_odom = False
                self._sensor_condition.notify_all()

        timer = threading.Timer(0.001, update)
        timer.start()
        return timer

    def require_fresh_inputs(self):
        """Model the existing header freshness gate."""
        self.freshness_checks += 1
        if self.stale:
            raise RuntimeError("/scan timestamp is stale")

    def check_drive_dynamic_safety(self):
        """Record that dynamic safety runs after freshness."""
        self.dynamic_checks += 1

    def sensor_age_diagnostics(self):
        """Return compact deterministic age diagnostics."""
        return {
            "scan_sequence": self.scan_sequence,
            "odom_sequence": self.odom_sequence,
            "scan_receive_age_sec": 0.10,
            "scan_header_age_sec": 0.12,
            "odom_receive_age_sec": 0.08,
            "odom_header_age_sec": 0.09,
        }

    def _maximum_optional_age(self, current, sample):
        """Reuse the production maximum-age accumulator."""
        return RealcarPolicyContinuousRunner._maximum_optional_age(
            current,
            sample,
        )

    def _record_drive_sensor_cycle(self, *args):
        """Reuse production diagnostic recording."""
        return RealcarPolicyContinuousRunner._record_drive_sensor_cycle(
            self,
            *args,
        )

    def format_sensor_age_diagnostics(self, diagnostics):
        """Provide a concise timeout diagnostic for assertions."""
        return str(diagnostics)

    def stop(self, repeat=10):
        """Record the immediate zero-velocity request."""
        del repeat
        self.stop_calls += 1


class RotationHarness:
    """Exercise the all-around rotation footprint gate."""

    def __init__(self, distance):
        self._sensor_condition = threading.Condition()
        self.latest_scan = make_scan(distance)
        self.latest_scan_received_at = 1.0
        self.scan_timeout_sec = 0.5
        self.footprint_radius_m = DEFAULT_FOOTPRINT_RADIUS_M
        self.laser_x_in_base_m = 0.0
        self.laser_y_in_base_m = 0.0
        self.laser_yaw_in_base = 0.0
        self._dynamic_step_record = {
            "rotation_footprint_passed": None,
            "rotation_nearest_footprint_clearance": None,
        }
        self.stop_calls = 0

    def require_fresh_sensor(self, *args):
        """Treat the fabricated scan as fresh."""
        del args

    def rotation_footprint_check(self, scan):
        """Reuse the production circular check."""
        return RealcarPolicyContinuousRunner.rotation_footprint_check(
            self,
            scan,
        )

    def stop(self, repeat=10):
        """Record the zero-command stop request."""
        del repeat
        self.stop_calls += 1


def test_continuous_cell_size_is_035():
    assert CONTINUOUS_CELL_SIZE_M == pytest.approx(0.35)


def test_continuous_footprint_defaults_define_040_corridor():
    assert DEFAULT_NOMINAL_MIN_CORRIDOR_WIDTH_M == pytest.approx(0.40)
    assert DEFAULT_FOOTPRINT_RADIUS_M == pytest.approx(0.20)
    assert DEFAULT_LONGITUDINAL_EXTRA_MARGIN_M == pytest.approx(0.05)
    assert 2.0 * DEFAULT_FOOTPRINT_RADIUS_M == pytest.approx(
        DEFAULT_NOMINAL_MIN_CORRIDOR_WIDTH_M
    )


def test_legacy_margin_cannot_conflict_with_footprint_geometry():
    harness = type(
        "FootprintParameterHarness",
        (),
        {
            "cell_size": 0.35,
            "step_distance": 0.35,
            "diagonal_mode": "grid_center",
            "all8_action_mode": True,
            "max_runtime_sec": 1.0,
            "motion_clearance_margin": 0.25,
            "dynamic_stop_distance": 0.25,
            "nominal_min_corridor_width_m": 0.40,
            "footprint_radius_m": 0.20,
            "longitudinal_extra_margin_m": 0.04,
            "dynamic_forward_center_line_extension_m": 0.05,
        },
    )()
    with pytest.raises(ValueError, match="legacy motion_clearance_margin"):
        RealcarPolicyContinuousRunner._validate_continuous_parameters(harness)


def test_continuous_default_step_distance_is_cell_size():
    assert RealcarPolicyContinuousRunner.DEFAULT_STEP_DISTANCE == pytest.approx(
        CONTINUOUS_CELL_SIZE_M
    )


def test_continuous_loop_has_a_finite_default_step_cap():
    assert 1 <= CONTINUOUS_DEFAULT_MAX_STEPS <= 1000


def test_continuous_sensor_callback_groups_are_independent():
    scan_group, odom_group = create_sensor_callback_groups(True)
    assert scan_group is not odom_group


def test_round7_sensor_callback_group_remains_shared():
    scan_group, odom_group = create_sensor_callback_groups(False)
    assert scan_group is odom_group


def test_max_steps_hard_limit_triggers():
    assert hard_limit_termination(30, 30, 10.0, 100.0) == "max_steps_reached"


def test_max_runtime_hard_limit_triggers():
    assert hard_limit_termination(2, 30, 100.0, 100.0) == "max_runtime_reached"


@pytest.mark.parametrize(
    ("delta_x", "delta_y", "expected"),
    (
        (0.35, 0.0, (0, 1)),
        (-0.35, 0.0, (0, -1)),
        (0.0, 0.35, (-1, 0)),
        (0.0, -0.35, (1, 0)),
    ),
)
def test_cardinal_motion_quantizes_to_one_cell(delta_x, delta_y, expected):
    assert odom_delta_to_grid_offset(delta_x, delta_y, 0.35) == expected


def test_diagonal_ne_motion_quantizes_row_and_column():
    assert odom_delta_to_grid_offset(0.35, 0.35, 0.35) == (-1, 1)


def test_diagonal_grid_center_target_distance_is_sqrt_two_cells():
    adapter = RealcarActionAdapter(ACTIONS_8, ACTION_NAMES, "grid_center")
    target = adapter.target_for_action(1, 0.0, 0.0, 0.35)
    assert math.hypot(target.target_x, target.target_y) == pytest.approx(
        math.sqrt(2.0) * 0.35
    )


@pytest.mark.parametrize("delta_x", (0.325, 0.34, 0.35))
def test_cardinal_endpoint_tolerance_quantizes_to_adjacent_cell(delta_x):
    assert odom_delta_to_grid_offset(delta_x, 0.0, 0.35) == (0, 1)


def test_sub_half_cell_motion_does_not_advance_grid_state():
    assert odom_delta_to_grid_offset(0.174, 0.0, 0.35) == (0, 0)


@pytest.mark.parametrize(
    ("corridor_width", "expected_passed"),
    ((0.38, False), (0.40, True), (0.42, True)),
)
def test_nominal_corridor_width_boundary(corridor_width, expected_passed):
    half_width = corridor_width / 2.0
    points = [(0.10, half_width), (0.30, -half_width)]
    result = capsule_footprint_check(points, 0.0, 0.40, 0.20)
    assert result.passed is expected_passed
    assert result.nearest_capsule_clearance == pytest.approx(
        half_width - 0.20
    )


def test_040_corridor_is_not_rejected_by_old_sector_min_semantics():
    points = [
        (0.30, -0.20),
        (0.30, 0.20),
        (0.55, -0.20),
        (0.55, 0.20),
    ]
    old_sector_hit = math.hypot(0.55, 0.20)
    old_sector_angle = math.atan2(0.20, 0.55)
    assert old_sector_angle < math.radians(22.5)
    assert old_sector_hit < 0.60
    result = capsule_footprint_check(points, 0.0, 0.40, 0.20)
    assert result.passed
    assert result.nearest_capsule_clearance == pytest.approx(0.0)


def test_rejection_diagnostics_distinguish_path_from_corridor():
    path_result = capsule_footprint_check(
        [(0.599, 0.0)],
        0.0,
        0.40,
        0.20,
    )
    corridor_result = capsule_footprint_check(
        [(0.30, 0.19)],
        0.0,
        0.40,
        0.20,
    )
    assert path_result.obstruction_type == "longitudinal_path_obstruction"
    assert corridor_result.obstruction_type == (
        "footprint_corridor_obstruction"
    )


def test_scan_hits_are_transformed_with_verified_laser_translation():
    scan = LaserScan()
    scan.angle_min = 0.0
    scan.angle_increment = 1.0
    scan.range_min = 0.05
    scan.range_max = 10.0
    scan.ranges = [1.0]
    points = scan_points_in_base(
        scan,
        DEFAULT_LASER_X_IN_BASE_M,
        DEFAULT_LASER_Y_IN_BASE_M,
        0.0,
    )
    assert points == pytest.approx(
        [(1.0 + DEFAULT_LASER_X_IN_BASE_M, DEFAULT_LASER_Y_IN_BASE_M)]
    )


@pytest.mark.parametrize(
    ("distance", "expected_passed"),
    ((0.599, False), (0.600, True), (0.601, True)),
)
def test_cardinal_capsule_preserves_060_front_edge(
    distance,
    expected_passed,
):
    result = capsule_footprint_check([(distance, 0.0)], 0.0, 0.40, 0.20)
    assert result.passed is expected_passed


@pytest.mark.parametrize(
    ("offset", "expected_passed"),
    ((-0.001, False), (0.0, True), (0.001, True)),
)
def test_diagonal_capsule_preserves_0745_front_edge(
    offset,
    expected_passed,
):
    center_line = math.sqrt(2.0) * 0.35 + 0.05
    front_edge = center_line + 0.20
    distance = front_edge + offset
    point = (
        distance * math.cos(math.pi / 4.0),
        distance * math.sin(math.pi / 4.0),
    )
    result = capsule_footprint_check(
        [point],
        math.pi / 4.0,
        center_line,
        0.20,
    )
    assert front_edge == pytest.approx(0.7449747468)
    assert result.passed is expected_passed


def test_scan_capsule_fails_safe_without_valid_hits():
    scan = make_scan(float("inf"))
    result = scan_capsule_footprint_check(
        scan,
        0.0,
        0.40,
        0.20,
        DEFAULT_LASER_X_IN_BASE_M,
        DEFAULT_LASER_Y_IN_BASE_M,
        0.0,
    )
    assert not result.passed
    assert result.obstruction_type == "invalid_scan"


def test_normal_mode_keeps_policy_action_source_and_fallback_behavior():
    validate_commissioning_config(False, 30, -1)
    assert action_source_for_mode(False) == "policy"
    harness = PlanHarness({1: 0.70, 2: 0.80})
    plan = RealcarPolicyContinuousRunner.prepare_continuous_pre_motion_plan(
        harness,
        1,
        [(1, 1.0), (2, 0.9)],
        make_scan(1.0),
        make_odom(),
    )
    assert plan.executed_action == 2
    assert plan.safety_fallback_used


def test_commissioning_requires_exactly_one_step():
    with pytest.raises(ValueError, match="max_steps == 1"):
        validate_commissioning_config(True, 2, 2)


def test_commissioning_reaches_policy_inference_before_step_cap():
    harness = type(
        "CommissioningCompletionHarness",
        (),
        {
            "commissioning_mode": True,
            "execute": True,
            "disable_completion_termination_in_dryrun": False,
        },
    )()
    assert not RealcarPolicyContinuousRunner._completion_enabled(harness)


def test_successful_commissioning_has_successful_completion_semantics():
    reason = successful_step_termination_reason(True, True)
    assert reason == "commissioning_complete"
    assert episode_success_for_reason(True, True, reason)


def test_commissioning_dryrun_is_complete_but_not_motion_success():
    reason = successful_step_termination_reason(True, False)
    assert reason == "commissioning_dryrun_complete"
    assert not episode_success_for_reason(True, False, reason)
    assert not episode_success_for_reason(
        True,
        False,
        "commissioning_complete",
    )


def test_blocked_commissioning_is_not_successful():
    assert not episode_success_for_reason(
        True,
        True,
        "commissioning_action_blocked",
    )


def test_normal_max_steps_is_not_successful():
    assert successful_step_termination_reason(False, True) is None
    assert not episode_success_for_reason(False, True, "max_steps_reached")


def test_frontier_exhaustion_remains_normal_continuous_success():
    assert episode_success_for_reason(False, True, "frontier_exhausted")


@pytest.mark.parametrize("action_idx", (-1, 8, 99))
def test_commissioning_action_must_be_in_action_space(action_idx):
    with pytest.raises(ValueError, match=r"\[0, 7\]"):
        validate_commissioning_config(True, 1, action_idx)


def test_safe_commissioning_action_is_the_executed_action():
    harness = PlanHarness({1: 0.80, 2: 0.80, 4: 0.80})
    plan = RealcarPolicyContinuousRunner.prepare_continuous_pre_motion_plan(
        harness,
        4,
        [(4, 1.0), (1, 0.9), (2, 0.8)],
        make_scan(1.0),
        make_odom(),
        requested_action=2,
        allow_fallback=False,
    )
    assert action_source_for_mode(True) == "commissioning_override"
    assert plan.raw_policy_action == 4
    assert plan.executed_action == 2
    assert not plan.safety_fallback_used


def test_unsafe_commissioning_action_does_not_fallback():
    harness = PlanHarness({1: 0.80, 2: 0.30, 4: 0.80})
    plan = RealcarPolicyContinuousRunner.prepare_continuous_pre_motion_plan(
        harness,
        4,
        [(4, 1.0), (1, 0.9), (2, 0.8)],
        make_scan(1.0),
        make_odom(),
        requested_action=2,
        allow_fallback=False,
    )
    assert plan.raw_policy_action == 4
    assert plan.executed_action is None
    assert not plan.gate_passed
    assert not plan.safety_fallback_used


def test_commissioning_cardinal_e_target_is_one_physical_cell():
    harness = PlanHarness({2: 0.80})
    plan = RealcarPolicyContinuousRunner.prepare_continuous_pre_motion_plan(
        harness,
        7,
        [(7, 1.0), (2, 0.9)],
        make_scan(1.0),
        make_odom(),
        requested_action=2,
        allow_fallback=False,
    )
    assert plan.target is not None
    assert plan.target.target_x == pytest.approx(0.35)
    assert plan.target.target_y == pytest.approx(0.0)
    assert plan.target_distance == pytest.approx(0.35)


def test_commissioning_diagonal_ne_target_uses_grid_center_geometry():
    harness = PlanHarness({1: 0.80})
    plan = RealcarPolicyContinuousRunner.prepare_continuous_pre_motion_plan(
        harness,
        6,
        [(6, 1.0), (1, 0.9)],
        make_scan(1.0),
        make_odom(),
        requested_action=1,
        allow_fallback=False,
    )
    assert plan.target is not None
    assert plan.target.target_x == pytest.approx(0.35)
    assert plan.target.target_y == pytest.approx(0.35)
    assert plan.target_distance == pytest.approx(0.4949747468)


def test_commissioning_action_does_not_override_odom_derived_state():
    expected = expected_grid_state_from_action((60, 60), 2)
    harness = type(
        "CommissioningOdomHarness",
        (),
        {"odom_state_origin": (0.0, 0.0), "cell_size": 0.35},
    )()
    actual = RealcarPolicyContinuousRunner.agent_state_from_odom(
        harness,
        (60, 60),
        0.10,
        0.0,
    )
    assert expected == (60, 61)
    assert actual == (60, 60)
    assert not grid_transition_matches(expected, actual)


def test_footprint_aware_plan_falls_back_and_rebuilds_target():
    harness = PlanHarness({1: 0.70, 2: 0.80})
    plan = RealcarPolicyContinuousRunner.prepare_continuous_pre_motion_plan(
        harness,
        1,
        [(1, 1.0), (2, 0.9)],
        make_scan(1.0),
        make_odom(4.0, -2.0),
    )
    assert isinstance(plan, ContinuousPreMotionPlan)
    assert plan.executed_action == 2
    assert plan.safety_fallback_used
    assert plan.target is not None
    assert plan.target.target_x == pytest.approx(4.35)
    assert plan.target.target_y == pytest.approx(-2.0)
    assert plan.target_distance == pytest.approx(0.35)


def test_no_footprint_aware_safe_action_has_no_executed_action():
    harness = PlanHarness({1: 0.30, 2: 0.40})
    plan = RealcarPolicyContinuousRunner.prepare_continuous_pre_motion_plan(
        harness,
        1,
        [(1, 1.0), (2, 0.9)],
        make_scan(1.0),
        make_odom(),
    )
    assert not plan.gate_passed
    assert plan.executed_action is None
    assert plan.obstruction_type == "longitudinal_path_obstruction"
    assert not motion_is_permitted(True, plan.gate_passed, plan.executed_action)


def test_drive_dynamic_gate_stops_before_motion_can_continue():
    harness = DynamicHarness(0.199, math.pi / 2.0)
    with pytest.raises(DynamicObstacleStop, match="dynamic_footprint_stop"):
        RealcarPolicyContinuousRunner.check_drive_dynamic_safety(harness)
    assert harness.stop_calls == 1
    assert harness._dynamic_step_record["dynamic_obstacle_stop"]
    assert harness._dynamic_step_record["dynamic_footprint_stop"]


def test_drive_dynamic_gate_allows_nominal_040_corridor_wall():
    harness = DynamicHarness(0.20, math.pi / 2.0)
    RealcarPolicyContinuousRunner.check_drive_dynamic_safety(harness)
    assert harness.stop_calls == 0
    assert not harness._dynamic_step_record["dynamic_footprint_stop"]


@pytest.mark.parametrize(
    ("points", "expected_passed"),
    (
        ([(0.0, 0.199)], False),
        ([(0.0, 0.200)], True),
        ([(0.0, 0.201)], True),
        ([(0.249, 0.0)], False),
        ([(0.250, 0.0)], True),
        ([(0.251, 0.0)], True),
    ),
)
def test_drive_dynamic_capsule_keeps_side_and_front_thresholds(
    points,
    expected_passed,
):
    result = capsule_footprint_check(points, 0.0, 0.05, 0.20)
    assert result.passed is expected_passed


def test_rotation_footprint_blocks_inside_radius_before_rotate():
    harness = RotationHarness(0.199)
    with pytest.raises(
        RotationFootprintBlocked,
        match="rotation_footprint_blocked",
    ):
        RealcarPolicyContinuousRunner.check_rotation_footprint_safety(
            harness
        )
    assert harness.stop_calls == 1
    assert not harness._dynamic_step_record["rotation_footprint_passed"]


def test_rotation_footprint_allows_obstacles_outside_radius():
    harness = RotationHarness(0.201)
    RealcarPolicyContinuousRunner.check_rotation_footprint_safety(harness)
    assert harness.stop_calls == 0
    assert harness._dynamic_step_record["rotation_footprint_passed"]


def test_drive_cycle_with_only_odom_update_cannot_continue():
    harness = DriveCycleHarness(advance_odom=True)
    timer = harness.start_background_updates()
    with pytest.raises(DriveSensorCycleTimeout):
        RealcarPolicyContinuousRunner.wait_for_drive_sensor_cycle(
            harness,
            10,
            20,
        )
    timer.join()
    assert harness.stop_calls == 1
    assert not harness._dynamic_step_record["drive_sensor_cycle_scan_advanced"]
    assert harness._dynamic_step_record["drive_sensor_cycle_odom_advanced"]


def test_drive_cycle_with_only_scan_update_cannot_continue():
    harness = DriveCycleHarness(advance_scan=True)
    timer = harness.start_background_updates()
    with pytest.raises(DriveSensorCycleTimeout):
        RealcarPolicyContinuousRunner.wait_for_drive_sensor_cycle(
            harness,
            10,
            20,
        )
    timer.join()
    assert harness.stop_calls == 1
    assert harness._dynamic_step_record["drive_sensor_cycle_scan_advanced"]
    assert not harness._dynamic_step_record["drive_sensor_cycle_odom_advanced"]


def test_drive_cycle_with_fresh_scan_and_odom_can_continue():
    harness = DriveCycleHarness(advance_scan=True, advance_odom=True)
    timer = harness.start_background_updates()
    RealcarPolicyContinuousRunner.wait_for_drive_sensor_cycle(
        harness,
        10,
        20,
    )
    timer.join()
    assert harness.freshness_checks == 1
    assert harness.dynamic_checks == 1
    assert harness.stop_calls == 0
    assert harness._dynamic_step_record["drive_sensor_cycle_scan_advanced"]
    assert harness._dynamic_step_record["drive_sensor_cycle_odom_advanced"]


def test_drive_sensor_cycle_timeout_stops_and_records_abort_diagnostic():
    harness = DriveCycleHarness()
    with pytest.raises(
        DriveSensorCycleTimeout,
        match="drive_sensor_cycle_timeout",
    ):
        RealcarPolicyContinuousRunner.wait_for_drive_sensor_cycle(
            harness,
            10,
            20,
        )
    assert harness.stop_calls == 1
    assert harness._dynamic_step_record["drive_sensor_cycle_timeout"]
    assert harness._dynamic_step_record["drive_sensor_cycle_count"] == 1


def test_drive_cycle_stale_header_blocks_before_dynamic_safety():
    harness = DriveCycleHarness(
        advance_scan=True,
        advance_odom=True,
        stale=True,
    )
    timer = harness.start_background_updates()
    with pytest.raises(RuntimeError, match="timestamp is stale"):
        RealcarPolicyContinuousRunner.wait_for_drive_sensor_cycle(
            harness,
            10,
            20,
        )
    timer.join()
    assert harness.freshness_checks == 1
    assert harness.dynamic_checks == 0
    assert harness._dynamic_step_record["drive_max_scan_header_age_sec"] == 0.12


@pytest.mark.parametrize(
    ("scan_sequence", "odom_sequence", "expected"),
    (
        (11, 20, (True, False, False)),
        (10, 21, (False, True, False)),
        (11, 21, (True, True, True)),
    ),
)
def test_drive_sensor_pair_requires_both_sequences(
    scan_sequence,
    odom_sequence,
    expected,
):
    assert drive_sensor_sequence_progress(
        scan_sequence,
        odom_sequence,
        10,
        20,
    ) == expected


def test_post_inference_barrier_requires_new_scan_and_odom():
    barrier = SensorRefreshBarrier(4, 7, 20.0)
    assert not sensor_sample_is_after_barrier(4, 20.1, 4, 20.0)
    assert not sensor_sample_is_after_barrier(7, 20.1, 7, 20.0)
    assert sensor_sample_is_after_barrier(5, 20.1, 4, 20.0)
    assert sensor_sample_is_after_barrier(8, 20.1, 7, 20.0)
    assert barrier.scan_sequence == 4


@pytest.mark.parametrize("name", ("/scan", "/odom"))
def test_stale_sensor_header_remains_blocked(name):
    error = sensor_freshness_error(
        name,
        stamp_sec=99.0,
        received_at=200.0,
        now_monotonic=200.01,
        now_ros_sec=100.0,
        timeout_sec=0.5,
        future_tolerance_sec=0.1,
    )
    assert error is not None
    assert "timestamp is stale" in error


def test_belief_statistics_do_not_read_true_coverage():
    assert belief_statistics(BeliefHarness()) == (3, 2)


def test_frontier_exhaustion_requires_minimum_exploration():
    assert not frontier_exhausted(0, 1, 100, 3, 20)
    assert not frontier_exhausted(0, 3, 10, 3, 20)
    assert frontier_exhausted(0, 3, 20, 3, 20)


def test_nonzero_frontier_never_reports_exhaustion():
    assert not frontier_exhausted(1, 20, 100, 3, 20)


def test_known_area_stagnation_requires_full_window():
    assert not known_area_stagnated([10, 10, 10], 3, 1)
    assert known_area_stagnated([10, 10, 10, 10], 3, 1)


def test_known_area_growth_prevents_stagnation():
    assert not known_area_stagnated([10, 10, 11, 11], 3, 1)


def test_repeated_action_alone_cannot_trigger_deadlock():
    states = [(60, index) for index in range(8)]
    known = [100] * 8
    assert not repeated_state_deadlock(states, known, 8, 2, 1)


def test_repeated_state_plus_no_growth_triggers_deadlock():
    states = [(60, 60), (60, 61)] * 4
    known = [100] * 8
    assert repeated_state_deadlock(states, known, 8, 2, 1)


def test_expected_state_is_diagnostic_and_actual_state_is_odom_derived():
    expected = expected_grid_state_from_action((60, 60), 2)
    actual = odom_delta_to_grid_offset(0.10, 0.0, 0.35)
    actual = (60 + actual[0], 60 + actual[1])
    assert expected == (60, 61)
    assert actual == (60, 60)
    assert not grid_transition_matches(expected, actual)


def test_continuous_agent_state_comes_from_cumulative_odom():
    harness = type(
        "OdomHarness",
        (),
        {"odom_state_origin": (1.0, 2.0), "cell_size": 0.35},
    )()
    actual = RealcarPolicyContinuousRunner.agent_state_from_odom(
        harness,
        (60, 60),
        1.35,
        2.35,
    )
    assert actual == (59, 61)


def test_sensor_failure_classifier_covers_refresh_timeout():
    assert RealcarPolicyContinuousRunner._sensor_failure(
        TimeoutError("timeout waiting for fresh /scan and /odom")
    )


def test_motion_requires_explicit_execute_even_with_safe_action():
    assert not motion_is_permitted(False, True, 2)
    assert motion_is_permitted(True, True, 2)
