"""Tests for deployment-only conservative real-car belief fusion."""

import math

import numpy as np
import pytest
from sensor_msgs.msg import LaserScan

from drl_explore_bridge.realcar_conservative_belief import (
    EMPTY,
    INVISIBLE,
    OBSTACLE,
    project_scan_to_belief,
    promote_observed_obstacle_cells,
)


CELL_SIZE = 0.35
ORIGIN_STATE = (60, 60)
LASER_X = 0.0316


def make_scan(distance, angle=0.0):
    """Create a one-beam raw scan."""
    scan = LaserScan()
    scan.angle_min = angle
    scan.angle_increment = 1.0
    scan.range_min = 0.05
    scan.range_max = 10.0
    scan.ranges = [distance]
    return scan


def make_multi_scan(distances, angle_min=0.0, angle_increment=1.0):
    """Create a raw scan with explicit beam geometry."""
    scan = LaserScan()
    scan.angle_min = angle_min
    scan.angle_increment = angle_increment
    scan.range_min = 0.05
    scan.range_max = 10.0
    scan.ranges = list(distances)
    return scan


def project(
    scan,
    *,
    robot_x=0.0,
    robot_y=0.0,
    robot_yaw=0.0,
    agent_state=ORIGIN_STATE,
    laser_x=LASER_X,
):
    """Project with the production grid and verified planar extrinsic."""
    return project_scan_to_belief(
        scan,
        robot_x=robot_x,
        robot_y=robot_y,
        robot_yaw=robot_yaw,
        origin_x=0.0,
        origin_y=0.0,
        origin_state=ORIGIN_STATE,
        agent_state=agent_state,
        cell_size=CELL_SIZE,
        scan_radius_cells=10,
        laser_x_in_base=laser_x,
        laser_y_in_base=0.0,
        laser_yaw_in_base=0.0,
    )


def frontier_mask(grid):
    """Return free cells adjacent to four-connected invisible space."""
    result = np.zeros_like(grid, dtype=np.uint8)
    for row, col in zip(*np.nonzero(grid == EMPTY)):
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            other_row = row + drow
            other_col = col + dcol
            if (
                0 <= other_row < grid.shape[0]
                and 0 <= other_col < grid.shape[1]
                and grid[other_row, other_col] == INVISIBLE
            ):
                result[row, col] = 255
                break
    return result


class BeliefCacheHarness:
    """Expose the cache-maintenance surface used by the deployment helper."""

    def __init__(self):
        """Create one free frontier cell with populated derived caches."""
        self.origin_world_rc = (58, 58)
        self.map = np.full((5, 5), INVISIBLE, dtype=np.int8)
        self.map[2, 2] = EMPTY
        self.frontier_u8 = frontier_mask(self.map)
        self.frontier_revision = 1
        self._latest_frontier_stats = object()
        self._cached_obstacle_integral = np.ones((2, 2), dtype=np.int64)
        self.kpm_count = 1
        self.coverage_refreshed = False
        self.analysis_refreshed = False

    def _ensure_world_bounds(self, min_row, max_row, min_col, max_col):
        assert 58 <= min_row <= max_row <= 62
        assert 58 <= min_col <= max_col <= 62
        return None

    def _count_coverage_hits(self, rows, cols):
        assert rows.size == cols.size
        return int(rows.size)

    def _invalidate_map_state_caches(self):
        self._latest_frontier_stats = None
        self._cached_obstacle_integral = None

    @staticmethod
    def _dirty_rect_from_points(rows, cols):
        return (
            int(rows.min()),
            int(rows.max()) + 1,
            int(cols.min()),
            int(cols.max()) + 1,
        )

    @staticmethod
    def _expand_dirty_rect(rect, radius):
        return (
            rect[0] - radius,
            rect[1] + radius,
            rect[2] - radius,
            rect[3] + radius,
        )

    def _update_frontier_dirty_rects(self, dirty_rects):
        if dirty_rects:
            self.frontier_u8 = frontier_mask(self.map)
            self.frontier_revision += 1

    def _refresh_coverage(self):
        self.coverage_refreshed = True

    def _update_analysis_box(self):
        self.analysis_refreshed = True

    def get_frontier_u8(self):
        """Return the incrementally refreshed frontier cache."""
        return self.frontier_u8


def test_single_lidar_hit_marks_the_whole_drl_cell_obstacle():
    """One valid endpoint makes its categorical cell an obstacle."""
    observation = project(make_scan(0.08), laser_x=0.0)

    assert observation.obstacle_cells == frozenset({ORIGIN_STATE})
    assert observation.local_snap[10, 10] == OBSTACLE


@pytest.mark.parametrize("distance", [math.inf, math.nan, 0.04, 10.01])
def test_invalid_ray_does_not_create_a_free_corridor(distance):
    """An invalid beam contributes neither free nor obstacle evidence."""
    observation = project(make_scan(distance), laser_x=0.0)

    expected = np.full((21, 21), INVISIBLE, dtype=np.int8)
    expected[10, 10] = EMPTY
    np.testing.assert_array_equal(observation.local_snap, expected)
    assert observation.obstacle_cells == frozenset()


def test_valid_obstacle_ray_clears_only_cells_before_endpoint():
    """A valid beam marks free cells only on its pre-hit segment."""
    observation = project(make_scan(1.0), laser_x=0.0)

    np.testing.assert_array_equal(
        observation.local_snap[10, 10:13],
        np.full(3, EMPTY, dtype=np.int8),
    )
    assert observation.local_snap[10, 13] == OBSTACLE
    assert observation.obstacle_cells == frozenset({(60, 63)})


def test_cells_behind_valid_obstacle_remain_invisible():
    """No free evidence is projected beyond a valid endpoint."""
    observation = project(make_scan(1.0), laser_x=0.0)

    np.testing.assert_array_equal(
        observation.local_snap[10, 14:],
        np.full(7, INVISIBLE, dtype=np.int8),
    )


def test_mixed_valid_and_invalid_rays_preserve_valid_observation():
    """Invalid beams add no evidence beside a valid beam projection."""
    valid_only = project(make_scan(1.0), laser_x=0.0)
    mixed = project(
        make_multi_scan([1.0, math.inf], angle_increment=math.pi / 2.0),
        laser_x=0.0,
    )

    np.testing.assert_array_equal(mixed.local_snap, valid_only.local_snap)
    assert mixed.obstacle_cells == valid_only.obstacle_cells


def test_empty_cell_is_promoted_to_obstacle():
    """A later reliable hit promotes a previously free cell."""
    belief = BeliefCacheHarness()

    stats = promote_observed_obstacle_cells(belief, {ORIGIN_STATE})

    assert belief.map[2, 2] == OBSTACLE
    assert stats.obstacle_promotions_this_step == 1
    assert stats.promoted_from_empty == 1
    assert stats.promoted_from_invisible == 0


def test_invalid_ray_does_not_reset_a_known_obstacle():
    """An invalid beam cannot demote persistent cumulative evidence."""
    belief = BeliefCacheHarness()
    promote_observed_obstacle_cells(belief, {ORIGIN_STATE})
    free_observation = project(make_scan(math.inf), laser_x=0.0)

    stats = promote_observed_obstacle_cells(
        belief,
        free_observation.obstacle_cells,
    )

    assert belief.map[2, 2] == OBSTACLE
    assert stats.obstacle_promotions_this_step == 0


def test_non_center_continuous_pose_selects_the_correct_global_cell():
    """Continuous robot displacement participates before quantization."""
    observation = project(
        make_scan(0.05),
        robot_x=0.13,
        laser_x=0.0,
    )

    assert observation.obstacle_cells == frozenset({(60, 61)})
    assert observation.local_snap[10, 11] == OBSTACLE


def test_lidar_x_offset_participates_before_grid_quantization():
    """The verified LiDAR translation can move a hit across a boundary."""
    without_offset = project(make_scan(0.15), laser_x=0.0)
    with_offset = project(make_scan(0.15), laser_x=LASER_X)

    assert without_offset.obstacle_cells == frozenset({(60, 60)})
    assert with_offset.obstacle_cells == frozenset({(60, 61)})


def test_promotion_invalidates_caches_and_refreshes_frontier():
    """Promotion refreshes frontier state and invalidates map caches."""
    belief = BeliefCacheHarness()
    assert belief.get_frontier_u8()[2, 2] == 255

    promote_observed_obstacle_cells(belief, {ORIGIN_STATE})

    assert belief.get_frontier_u8()[2, 2] == 0
    assert belief.frontier_revision == 2
    assert belief._latest_frontier_stats is None
    assert belief._cached_obstacle_integral is None
    assert belief.coverage_refreshed
    assert belief.analysis_refreshed
