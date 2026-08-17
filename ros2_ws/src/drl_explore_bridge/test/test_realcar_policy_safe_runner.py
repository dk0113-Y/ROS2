import math

import pytest
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

from drl_explore_bridge.realcar_action_adapter import RealcarActionAdapter
from drl_explore_bridge.realcar_policy_safe_runner_node import (
    ACTIONS_8,
    ACTION_NAMES,
    LATEST_SENSOR_QOS,
    MAX_SAFE_STEPS,
    RealcarPolicySafeRunner,
    SensorRefreshBarrier,
    odom_delta_to_grid_offset,
    sensor_freshness_error,
    sensor_sample_is_after_barrier,
)


class PreMotionHarness:
    def __init__(self, lidar_results):
        self.allowed_actions = set(range(8))
        self.all8_action_mode = True
        self.action_adapter = RealcarActionAdapter(ACTIONS_8, ACTION_NAMES)
        self.step_distance = 0.10
        self.lidar_results = lidar_results
        self.seen_sensor_pairs = []

    def lidar_gate(self, action_idx, scan, odom):
        self.seen_sensor_pairs.append((scan, odom))
        return self.lidar_results[action_idx]

    def pose_record_from_odom(self, odom):
        return RealcarPolicySafeRunner.pose_record_from_odom(odom)

    def target_for_action(self, action_idx, x0, y0):
        return self.action_adapter.target_for_action(
            action_idx,
            start_x=x0,
            start_y=y0,
            step_distance=self.step_distance,
        )


class RefreshHarness:
    def __init__(self, old_scan, old_odom, refreshed_scan, refreshed_odom):
        self.latest_scan = old_scan
        self.latest_odom = old_odom
        self.refreshed_scan = refreshed_scan
        self.refreshed_odom = refreshed_odom
        self.wait_arguments = None

    def wait_for_inputs(self, **kwargs):
        self.wait_arguments = kwargs
        self.latest_scan = self.refreshed_scan
        self.latest_odom = self.refreshed_odom


def make_odom(x, y, yaw=0.0):
    odom = Odometry()
    odom.pose.pose.position.x = x
    odom.pose.pose.position.y = y
    odom.pose.pose.orientation.z = math.sin(yaw / 2.0)
    odom.pose.pose.orientation.w = math.cos(yaw / 2.0)
    odom.header.stamp.sec = 100
    return odom


def test_multi_step_limit_is_three():
    assert MAX_SAFE_STEPS == 3


def test_latest_sensor_qos_keeps_only_one_reliable_volatile_sample():
    assert LATEST_SENSOR_QOS.history == HistoryPolicy.KEEP_LAST
    assert LATEST_SENSOR_QOS.depth == 1
    assert LATEST_SENSOR_QOS.reliability == ReliabilityPolicy.RELIABLE
    assert LATEST_SENSOR_QOS.durability == DurabilityPolicy.VOLATILE


def test_inference_queue_old_scan_cannot_cross_refresh_barrier():
    assert not sensor_sample_is_after_barrier(7, 20.1, 7, 20.0)


def test_refresh_barrier_replaces_pre_inference_sensor_pair():
    old_scan = LaserScan()
    old_odom = Odometry()
    refreshed_scan = LaserScan()
    refreshed_odom = Odometry()
    harness = RefreshHarness(
        old_scan,
        old_odom,
        refreshed_scan,
        refreshed_odom,
    )
    barrier = SensorRefreshBarrier(7, 12, 20.0)

    scan, odom = RealcarPolicySafeRunner.refresh_inputs_after_barrier(
        harness,
        barrier,
    )

    assert scan is refreshed_scan
    assert odom is refreshed_odom
    assert scan is not old_scan
    assert odom is not old_odom
    assert harness.wait_arguments == {
        "after_scan_sequence": 7,
        "after_odom_sequence": 12,
        "after_monotonic": 20.0,
    }


def test_pre_motion_refresh_requires_post_barrier_scan_callback():
    assert sensor_sample_is_after_barrier(8, 20.1, 7, 20.0)
    assert not sensor_sample_is_after_barrier(8, 19.9, 7, 20.0)


def test_pre_motion_refresh_requires_post_barrier_odom_callback():
    barrier = SensorRefreshBarrier(11, 23, 50.0)
    assert sensor_sample_is_after_barrier(
        24,
        50.01,
        barrier.odom_sequence,
        barrier.started_monotonic,
    )
    assert not sensor_sample_is_after_barrier(
        23,
        50.01,
        barrier.odom_sequence,
        barrier.started_monotonic,
    )


def test_stale_new_scan_header_blocks_pre_motion_freshness():
    error = sensor_freshness_error(
        "/scan",
        stamp_sec=99.0,
        received_at=200.0,
        now_monotonic=200.01,
        now_ros_sec=100.0,
        timeout_sec=0.5,
        future_tolerance_sec=0.1,
    )
    assert error == "/scan timestamp is stale: age=1.000s > timeout=0.500s"


def test_stale_new_odom_header_blocks_pre_motion_freshness():
    error = sensor_freshness_error(
        "/odom",
        stamp_sec=99.0,
        received_at=200.0,
        now_monotonic=200.01,
        now_ros_sec=100.0,
        timeout_sec=0.5,
        future_tolerance_sec=0.1,
    )
    assert error == "/odom timestamp is stale: age=1.000s > timeout=0.500s"


def test_fresh_post_barrier_sensor_can_continue():
    assert sensor_sample_is_after_barrier(4, 30.01, 3, 30.0)
    assert sensor_freshness_error(
        "/scan",
        stamp_sec=100.0,
        received_at=200.0,
        now_monotonic=200.01,
        now_ros_sec=100.01,
        timeout_sec=0.5,
        future_tolerance_sec=0.1,
    ) is None


def test_pre_motion_lidar_gate_uses_refreshed_scan_and_odom():
    inference_scan = LaserScan()
    refreshed_scan = LaserScan()
    refreshed_odom = make_odom(1.0, 2.0)
    harness = PreMotionHarness({3: (True, 0.8)})

    selection, _pose, target = RealcarPolicySafeRunner.prepare_pre_motion_plan(
        harness,
        3,
        [(3, 1.0)],
        refreshed_scan,
        refreshed_odom,
    )

    assert selection.executed_action == 3
    assert target is not None
    assert harness.seen_sensor_pairs == [(refreshed_scan, refreshed_odom)]
    assert harness.seen_sensor_pairs[0][0] is not inference_scan


def test_refreshed_scan_fallback_rebuilds_target_from_latest_odom():
    refreshed_scan = LaserScan()
    refreshed_odom = make_odom(4.0, -2.0)
    harness = PreMotionHarness(
        {
            3: (False, 0.10),
            2: (True, 0.80),
        }
    )

    selection, pose, target = RealcarPolicySafeRunner.prepare_pre_motion_plan(
        harness,
        3,
        [(3, 1.0), (2, 0.9)],
        refreshed_scan,
        refreshed_odom,
    )

    assert selection.raw_policy_action == 3
    assert selection.executed_action == 2
    assert selection.safety_fallback_used
    assert pose["x"] == pytest.approx(4.0)
    assert pose["y"] == pytest.approx(-2.0)
    assert target is not None
    assert target.action_idx == 2
    assert target.target_x == pytest.approx(4.1)
    assert target.target_y == pytest.approx(-2.0)


def test_no_safe_refreshed_action_produces_no_motion_target():
    scan = LaserScan()
    odom = make_odom(4.0, -2.0)
    harness = PreMotionHarness(
        {
            3: (False, 0.10),
            2: (False, 0.12),
        }
    )

    selection, _pose, target = RealcarPolicySafeRunner.prepare_pre_motion_plan(
        harness,
        3,
        [(3, 1.0), (2, 0.9)],
        scan,
        odom,
    )

    assert not selection.lidar_gate_passed
    assert selection.executed_action is None
    assert target is None


@pytest.mark.parametrize(
    ("delta_x", "delta_y", "expected_offset"),
    (
        (0.0, 0.35, (-1, 0)),
        (0.35, 0.0, (0, 1)),
        (0.0, -0.35, (1, 0)),
        (-0.35, 0.0, (0, -1)),
        (0.35, 0.35, (-1, 1)),
    ),
)
def test_odom_displacement_uses_existing_drl_grid_axes(
    delta_x,
    delta_y,
    expected_offset,
):
    assert odom_delta_to_grid_offset(delta_x, delta_y, 0.35) == expected_offset


def test_subcell_motion_does_not_blindly_advance_state():
    assert odom_delta_to_grid_offset(0.10, 0.0, 0.35) == (0, 0)
    assert odom_delta_to_grid_offset(0.20, 0.0, 0.35) == (0, 1)


def test_invalid_cell_size_is_rejected():
    with pytest.raises(ValueError, match="cell_size must be > 0"):
        odom_delta_to_grid_offset(0.0, 0.0, 0.0)
