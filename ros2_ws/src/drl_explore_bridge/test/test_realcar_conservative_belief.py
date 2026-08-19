"""Tests for deployment-only conservative real-car belief fusion."""

import math

import numpy as np
import pytest
from sensor_msgs.msg import LaserScan

from drl_explore_bridge.realcar_conservative_belief import (
    BeliefEvidenceAccumulator,
    BeliefFusionConfig,
    EMPTY,
    INVISIBLE,
    OBSTACLE,
    ProjectedBeliefObservation,
    apply_evidence_fusion,
    apply_legacy_fusion,
    cumulative_occlusion_cells,
    frontier_semantics_snapshot,
    named_fusion_config,
    ordered_coarse_ray_cells,
    project_scan_to_belief,
    promote_observed_obstacle_cells,
    record_traversed_cells_as_free,
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
    coarse_occlusion_mode="off",
    historical_obstacle_cells=(),
    occlusion_exempt_cells=(),
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
        coarse_occlusion_mode=coarse_occlusion_mode,
        historical_obstacle_cells=historical_obstacle_cells,
        occlusion_exempt_cells=occlusion_exempt_cells,
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

    def __init__(self, size=5, origin_world_rc=(58, 58)):
        """Create one free frontier cell with populated derived caches."""
        self.origin_world_rc = tuple(origin_world_rc)
        self.map = np.full((size, size), INVISIBLE, dtype=np.int8)
        center_row = ORIGIN_STATE[0] - self.origin_world_rc[0]
        center_col = ORIGIN_STATE[1] - self.origin_world_rc[1]
        self.map[center_row, center_col] = EMPTY
        self.visit_count = np.zeros((size, size), dtype=np.int32)
        self.frontier_u8 = frontier_mask(self.map)
        self.frontier_revision = 1
        self._latest_frontier_stats = object()
        self._cached_obstacle_integral = np.ones((2, 2), dtype=np.int64)
        self.kpm_count = 1
        self.coverage_refreshed = False
        self.analysis_refreshed = False
        self.visit_cache_invalidated = False

    def _ensure_world_bounds(self, min_row, max_row, min_col, max_col):
        last_row = self.origin_world_rc[0] + self.map.shape[0] - 1
        last_col = self.origin_world_rc[1] + self.map.shape[1] - 1
        assert self.origin_world_rc[0] <= min_row <= max_row <= last_row
        assert self.origin_world_rc[1] <= min_col <= max_col <= last_col
        return None

    def _count_coverage_hits(self, rows, cols):
        assert rows.size == cols.size
        return int(rows.size)

    def _invalidate_map_state_caches(self):
        self._latest_frontier_stats = None
        self._cached_obstacle_integral = None

    def _invalidate_visit_cache(self):
        self.visit_cache_invalidated = True

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
    assert observation.free_cells == frozenset()
    assert observation.obstacle_cells == frozenset()
    assert observation.conflict_cells == frozenset()


def test_valid_obstacle_ray_clears_only_cells_before_endpoint():
    """A valid beam marks free cells only on its pre-hit segment."""
    observation = project(make_scan(1.0), laser_x=0.0)

    np.testing.assert_array_equal(
        observation.local_snap[10, 10:13],
        np.full(3, EMPTY, dtype=np.int8),
    )
    assert observation.local_snap[10, 13] == OBSTACLE
    assert observation.free_cells == frozenset(
        {(60, 60), (60, 61), (60, 62)}
    )
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


def test_visited_empty_cell_rejects_later_obstacle_endpoint():
    """Physical traversal is stronger than a later quantized endpoint."""
    belief = BeliefCacheHarness()
    belief.visit_count[2, 2] = 1

    stats = promote_observed_obstacle_cells(belief, {ORIGIN_STATE})

    assert belief.map[2, 2] == EMPTY
    assert stats.obstacle_promotions_this_step == 0


def test_unvisited_empty_cell_still_promotes_to_obstacle():
    """Conservative endpoint promotion is unchanged outside the trajectory."""
    belief = BeliefCacheHarness()

    stats = promote_observed_obstacle_cells(belief, {ORIGIN_STATE})

    assert belief.map[2, 2] == OBSTACLE
    assert stats.obstacle_promotions_this_step == 1


def test_previously_obstacle_cell_is_restored_when_traversed():
    """Later robot-center occupancy corrects an old obstacle belief."""
    belief = BeliefCacheHarness()
    belief.map[2, 2] = OBSTACLE

    corrections = record_traversed_cells_as_free(
        belief,
        {ORIGIN_STATE},
    )

    assert belief.map[2, 2] == EMPTY
    assert belief.visit_count[2, 2] == 1
    assert corrections.corrected_from_obstacle == 1


def test_current_agent_cell_remains_empty_despite_same_cell_endpoint():
    """A coarse endpoint cannot turn the occupied agent cell into obstacle."""
    belief = BeliefCacheHarness()
    belief.visit_count[2, 2] = 1
    observation = project(make_scan(0.08), laser_x=0.0)
    assert observation.obstacle_cells == frozenset({ORIGIN_STATE})

    promote_observed_obstacle_cells(
        belief,
        observation.obstacle_cells,
    )

    assert belief.map[2, 2] == EMPTY


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


def test_obstacle_to_visited_free_refreshes_frontier_and_caches():
    """Traversal correction keeps derived frontier and obstacle state coherent."""
    belief = BeliefCacheHarness()
    promote_observed_obstacle_cells(belief, {ORIGIN_STATE})
    belief._cached_obstacle_integral = np.ones((2, 2), dtype=np.int64)
    belief._latest_frontier_stats = object()
    assert belief.get_frontier_u8()[2, 2] == 0

    record_traversed_cells_as_free(belief, {ORIGIN_STATE})

    assert belief.map[2, 2] == EMPTY
    assert belief.get_frontier_u8()[2, 2] == 255
    assert belief.frontier_revision == 3
    assert belief._latest_frontier_stats is None
    assert belief._cached_obstacle_integral is None
    assert belief.visit_cache_invalidated
    assert belief.coverage_refreshed
    assert belief.analysis_refreshed


def test_frame_conflict_is_explicit_and_ray_order_independent():
    """Cross-ray FREE/endpoint overlap remains an explicit ambiguity."""
    forward = project(
        make_multi_scan([0.50, 0.08], angle_increment=0.0),
        laser_x=0.0,
    )
    reversed_order = project(
        make_multi_scan([0.08, 0.50], angle_increment=0.0),
        laser_x=0.0,
    )

    assert ORIGIN_STATE in forward.free_cells
    assert ORIGIN_STATE in forward.obstacle_cells
    assert forward.conflict_cells == frozenset({ORIGIN_STATE})
    assert reversed_order.free_cells == forward.free_cells
    assert reversed_order.obstacle_cells == forward.obstacle_cells
    assert reversed_order.conflict_cells == forward.conflict_cells
    np.testing.assert_array_equal(
        reversed_order.local_snap,
        forward.local_snap,
    )


def test_same_ray_endpoint_cell_is_not_counted_as_free_evidence():
    """Coarse quantization cannot make one ray vote both ways."""
    observation = project(make_scan(0.08), laser_x=0.0)

    assert observation.free_cells == frozenset()
    assert observation.obstacle_cells == frozenset({ORIGIN_STATE})
    assert observation.conflict_cells == frozenset()


def test_opaque_single_ray_matches_off_projection_without_prior_blocker():
    """Opt-in opacity does not change an unobstructed single-ray frame."""
    scan = make_scan(1.0)

    baseline = project(scan, laser_x=0.0)
    opaque = project(
        scan,
        laser_x=0.0,
        coarse_occlusion_mode="opaque",
    )

    np.testing.assert_array_equal(opaque.local_snap, baseline.local_snap)
    assert opaque.free_cells == baseline.free_cells
    assert opaque.obstacle_cells == baseline.obstacle_cells
    assert opaque.conflict_cells == baseline.conflict_cells
    assert opaque.occlusion_suppressed_cells == frozenset()


def test_opaque_cross_ray_endpoint_blocks_rear_free_and_obstacle_evidence():
    """A coarse endpoint blocks another ray that continues through its cell."""
    scan = make_multi_scan(
        [1.0, 1.8],
        angle_min=0.10,
        angle_increment=-0.20,
    )

    baseline = project(scan, laser_x=0.0)
    opaque = project(
        scan,
        laser_x=0.0,
        coarse_occlusion_mode="opaque",
    )

    blocker = (60, 63)
    assert (60, 64) in baseline.free_cells
    assert (61, 65) in baseline.obstacle_cells
    assert blocker in opaque.free_cells
    assert blocker in opaque.obstacle_cells
    assert opaque.conflict_cells == frozenset({blocker})
    assert (60, 64) not in opaque.free_cells
    assert (61, 65) not in opaque.obstacle_cells
    assert opaque.occlusion_suppressed_free_cells == frozenset(
        {(60, 64), (60, 65)}
    )
    assert opaque.occlusion_suppressed_obstacle_cells == frozenset(
        {(61, 65)}
    )


def test_confirmed_opaque_unconfirmed_endpoint_does_not_cross_ray_block():
    """One current endpoint cannot block until belief classifies its cell."""
    scan = make_multi_scan(
        [1.0, 1.8],
        angle_min=0.10,
        angle_increment=-0.20,
    )

    baseline = project(scan, laser_x=0.0)
    confirmed = project(
        scan,
        laser_x=0.0,
        coarse_occlusion_mode="confirmed_opaque",
    )

    assert (60, 63) in confirmed.obstacle_cells
    assert (60, 64) in confirmed.free_cells
    assert (61, 65) in confirmed.obstacle_cells
    np.testing.assert_array_equal(confirmed.local_snap, baseline.local_snap)
    assert confirmed.free_cells == baseline.free_cells
    assert confirmed.obstacle_cells == baseline.obstacle_cells
    assert confirmed.occlusion_blocker_cells == frozenset()
    assert confirmed.occlusion_suppressed_cells == frozenset()


def test_confirmed_opaque_obstacle_blocks_rear_but_receives_evidence():
    """A confirmed obstacle is visible while evidence behind it is hidden."""
    blocker = (60, 63)
    observation = project(
        make_multi_scan(
            [1.0, 1.8],
            angle_min=0.10,
            angle_increment=-0.20,
        ),
        laser_x=0.0,
        coarse_occlusion_mode="confirmed_opaque",
        historical_obstacle_cells={blocker},
    )

    assert blocker in observation.occlusion_blocker_cells
    assert blocker in observation.free_cells
    assert blocker in observation.obstacle_cells
    assert (60, 64) not in observation.free_cells
    assert (61, 65) not in observation.obstacle_cells
    assert observation.occlusion_suppressed_free_cells == frozenset(
        {(60, 64), (60, 65)}
    )
    assert observation.occlusion_suppressed_obstacle_cells == frozenset(
        {(61, 65)}
    )


def test_opaque_preserves_free_evidence_before_first_blocker():
    """Opacity suppresses only evidence strictly behind the blocker cell."""
    observation = project(
        make_scan(1.8),
        laser_x=0.0,
        coarse_occlusion_mode="opaque",
        historical_obstacle_cells={(60, 63)},
    )

    assert {(60, 60), (60, 61), (60, 62), (60, 63)} <= set(
        observation.free_cells
    )
    assert (60, 64) not in observation.free_cells
    assert observation.obstacle_cells == frozenset()


def test_historical_obstacle_can_reverse_before_rear_evidence_resumes():
    """A historical blocker receives FREE votes while its rear stays opaque."""
    belief = BeliefCacheHarness(size=15, origin_world_rc=(55, 55))
    accumulator = BeliefEvidenceAccumulator(
        named_fusion_config("candidate_a")
    )
    blocker = (60, 63)
    rear = (60, 64)
    obstacle_frame = evidence_observation(obstacle={blocker})
    apply_evidence_fusion(belief, accumulator, obstacle_frame)
    apply_evidence_fusion(belief, accumulator, obstacle_frame)
    assert belief.map[5, 8] == OBSTACLE

    for _index in range(3):
        observation = project(
            make_scan(1.8),
            laser_x=0.0,
            coarse_occlusion_mode="opaque",
            historical_obstacle_cells={blocker},
        )
        assert blocker in observation.free_cells
        assert rear not in observation.free_cells
        apply_evidence_fusion(belief, accumulator, observation)

    assert belief.map[5, 8] == EMPTY
    future = project(
        make_scan(1.8),
        laser_x=0.0,
        coarse_occlusion_mode="opaque",
    )
    assert rear in future.free_cells


def test_confirmed_obstacle_reverses_before_rear_evidence_resumes():
    """Candidate A can clear a confirmed blocker, reopening later frames."""
    belief = BeliefCacheHarness(size=15, origin_world_rc=(55, 55))
    accumulator = BeliefEvidenceAccumulator(
        named_fusion_config("candidate_a")
    )
    blocker = (60, 63)
    rear = (60, 64)
    obstacle_frame = evidence_observation(obstacle={blocker})
    apply_evidence_fusion(belief, accumulator, obstacle_frame)
    apply_evidence_fusion(belief, accumulator, obstacle_frame)
    assert belief.map[5, 8] == OBSTACLE

    for _index in range(3):
        historical_blockers, visited_cells = cumulative_occlusion_cells(
            belief
        )
        observation = project(
            make_scan(1.8),
            laser_x=0.0,
            coarse_occlusion_mode="confirmed_opaque",
            historical_obstacle_cells=historical_blockers,
            occlusion_exempt_cells=visited_cells,
        )
        assert blocker in observation.free_cells
        assert rear not in observation.free_cells
        apply_evidence_fusion(belief, accumulator, observation)

    assert belief.map[5, 8] == EMPTY
    historical_blockers, visited_cells = cumulative_occlusion_cells(belief)
    future = project(
        make_scan(1.8),
        laser_x=0.0,
        coarse_occlusion_mode="confirmed_opaque",
        historical_obstacle_cells=historical_blockers,
        occlusion_exempt_cells=visited_cells,
    )
    assert blocker not in future.occlusion_blocker_cells
    assert rear in future.free_cells


def test_occlusion_never_erases_historical_rear_free_state():
    """No current evidence leaves an already-known rear cell unchanged."""
    belief = BeliefCacheHarness(size=15, origin_world_rc=(55, 55))
    blocker = (60, 63)
    rear = (60, 64)
    belief.map[5, 8] = OBSTACLE
    belief.map[5, 9] = EMPTY
    accumulator = BeliefEvidenceAccumulator(
        named_fusion_config("candidate_a")
    )
    observation = project(
        make_scan(1.8),
        laser_x=0.0,
        coarse_occlusion_mode="opaque",
        historical_obstacle_cells={blocker},
    )

    assert rear not in observation.free_cells
    assert rear not in observation.obstacle_cells
    apply_evidence_fusion(belief, accumulator, observation)
    assert belief.map[5, 9] == EMPTY


def test_confirmed_occlusion_never_erases_historical_rear_free_state():
    """Confirmed opacity suppresses a frame without erasing rear history."""
    belief = BeliefCacheHarness(size=15, origin_world_rc=(55, 55))
    blocker = (60, 63)
    rear = (60, 64)
    belief.map[5, 8] = OBSTACLE
    belief.map[5, 9] = EMPTY
    accumulator = BeliefEvidenceAccumulator(
        named_fusion_config("candidate_a")
    )
    observation = project(
        make_scan(1.8),
        laser_x=0.0,
        coarse_occlusion_mode="confirmed_opaque",
        historical_obstacle_cells={blocker},
    )

    assert rear not in observation.free_cells
    assert rear not in observation.obstacle_cells
    apply_evidence_fusion(belief, accumulator, observation)
    assert belief.map[5, 9] == EMPTY


def test_visited_agent_cell_is_not_an_occlusion_blocker():
    """An endpoint quantized onto the robot cell cannot hide all other rays."""
    observation = project(
        make_multi_scan([0.08, 1.0], angle_increment=0.0),
        laser_x=0.0,
        coarse_occlusion_mode="opaque",
    )

    assert ORIGIN_STATE in observation.free_cells
    assert ORIGIN_STATE in observation.obstacle_cells
    assert (60, 61) in observation.free_cells
    assert (60, 62) in observation.free_cells


def test_confirmed_opaque_excludes_visited_obstacles_from_blockers():
    """Visited cells remain authoritative even if a stale map value is set."""
    belief = BeliefCacheHarness(size=15, origin_world_rc=(55, 55))
    blocker = (60, 63)
    rear = (60, 64)
    belief.map[5, 8] = OBSTACLE
    belief.visit_count[5, 8] = 1

    historical_blockers, visited_cells = cumulative_occlusion_cells(belief)
    observation = project(
        make_scan(1.8),
        laser_x=0.0,
        coarse_occlusion_mode="confirmed_opaque",
        historical_obstacle_cells=historical_blockers,
        occlusion_exempt_cells=visited_cells,
    )

    assert blocker not in historical_blockers
    assert blocker in visited_cells
    assert blocker not in observation.occlusion_blocker_cells
    assert rear in observation.free_cells


def test_ordered_supercover_is_unique_and_includes_corner_side_cells():
    """Exact diagonal crossings deterministically include both corner sides."""
    cells = ordered_coarse_ray_cells(
        0.0,
        0.0,
        0.7,
        0.7,
        origin_x=0.0,
        origin_y=0.0,
        origin_state=ORIGIN_STATE,
        cell_size=CELL_SIZE,
    )

    assert cells == (
        (60, 60),
        (59, 60),
        (60, 61),
        (59, 61),
        (58, 61),
        (59, 62),
        (58, 62),
    )
    assert len(cells) == len(set(cells))


def test_opaque_corner_supercover_blocker_stops_diagonal_evidence():
    """A blocker touched at an exact corner occludes the diagonal rear."""
    scan = make_scan(1.0, angle=math.pi / 4.0)
    baseline = project(scan, laser_x=0.0)
    opaque = project(
        scan,
        laser_x=0.0,
        coarse_occlusion_mode="opaque",
        historical_obstacle_cells={(59, 60)},
    )

    assert (59, 61) in baseline.free_cells
    assert (59, 61) not in opaque.free_cells
    assert (59, 60) in opaque.occlusion_blocker_cells


def test_confirmed_opaque_corner_supercover_is_deterministic():
    """Confirmed blockers use the same exact-corner supercover ordering."""
    observation = project(
        make_scan(1.0, angle=math.pi / 4.0),
        laser_x=0.0,
        coarse_occlusion_mode="confirmed_opaque",
        historical_obstacle_cells={(59, 60)},
    )

    assert (59, 61) not in observation.free_cells
    assert (59, 60) in observation.occlusion_blocker_cells


def test_default_occlusion_mode_is_exact_legacy_projection_regression():
    """Omitting the new parameter remains bit-for-bit equivalent to off."""
    scan = make_multi_scan([0.5, 1.0, 1.4], angle_increment=0.17)

    implicit = project(scan, laser_x=0.0)
    explicit = project(
        scan,
        laser_x=0.0,
        coarse_occlusion_mode="off",
        historical_obstacle_cells={(60, 61), (60, 62)},
    )

    np.testing.assert_array_equal(implicit.local_snap, explicit.local_snap)
    assert implicit.free_cells == explicit.free_cells
    assert implicit.obstacle_cells == explicit.obstacle_cells
    assert implicit.conflict_cells == explicit.conflict_cells


def test_many_free_rays_count_as_one_free_frame():
    """Raw beam multiplicity does not amplify free evidence."""
    observation = project(
        make_multi_scan([1.0] * 100, angle_increment=0.0),
        laser_x=0.0,
    )
    accumulator = BeliefEvidenceAccumulator(
        named_fusion_config("candidate_a")
    )

    accumulator.observe(
        observation.free_cells,
        observation.obstacle_cells,
        observation.conflict_cells,
    )

    assert accumulator.state_for(ORIGIN_STATE).free_frame_count == 1


def test_many_obstacle_endpoints_count_as_one_obstacle_frame():
    """Raw endpoint multiplicity does not amplify obstacle evidence."""
    observation = project(
        make_multi_scan([1.0] * 100, angle_increment=0.0),
        laser_x=0.0,
    )
    endpoint = (60, 63)
    accumulator = BeliefEvidenceAccumulator(
        named_fusion_config("candidate_a")
    )

    accumulator.observe(
        observation.free_cells,
        observation.obstacle_cells,
        observation.conflict_cells,
    )

    assert accumulator.state_for(endpoint).obstacle_frame_count == 1


def evidence_observation(*, free=(), obstacle=(), conflict=()):
    """Build one synthetic global evidence frame."""
    return ProjectedBeliefObservation(
        local_snap=np.full((3, 3), INVISIBLE, dtype=np.int8),
        obstacle_cells=frozenset(obstacle),
        free_cells=frozenset(free),
        conflict_cells=frozenset(conflict),
    )


def semantic_corridor_belief(left, center, right):
    """Create a bounded three-cell corridor for frontier semantics tests."""
    belief = BeliefCacheHarness()
    belief.map.fill(OBSTACLE)
    belief.map[2, 1:4] = np.asarray([left, center, right], dtype=np.int8)
    belief.visit_count.fill(0)
    belief.frontier_u8 = frontier_mask(belief.map)
    return belief


def test_never_observed_unknown_keeps_adjacent_free_frontier():
    """Case A: genuine unseen space remains an exploration frontier."""
    belief = semantic_corridor_belief(OBSTACLE, EMPTY, INVISIBLE)
    accumulator = BeliefEvidenceAccumulator(named_fusion_config("candidate_a"))

    snapshot = frontier_semantics_snapshot(
        belief,
        accumulator,
        "evidence_aware",
    )

    assert snapshot.raw_frontier_u8[2, 2] == 255
    assert snapshot.effective_frontier_u8[2, 2] == 255
    assert snapshot.never_observed_unknown_mask[2, 3]


def test_observed_unclassified_unknown_alone_removes_frontier():
    """Case B: observed ambiguity alone is not unexplored space."""
    belief = semantic_corridor_belief(OBSTACLE, EMPTY, INVISIBLE)
    accumulator = BeliefEvidenceAccumulator(named_fusion_config("candidate_a"))
    accumulator.observe((), (), {(60, 61)})
    map_before = belief.map.copy()
    raw_before = belief.frontier_u8.copy()

    snapshot = frontier_semantics_snapshot(
        belief,
        accumulator,
        "evidence_aware",
    )

    assert snapshot.raw_frontier_u8[2, 2] == 255
    assert snapshot.effective_frontier_u8[2, 2] == 0
    assert snapshot.frontier_removed_by_observed_unknown_count == 1
    assert snapshot.frontier_adjacent_observed_unclassified_count == 1
    assert np.array_equal(belief.map, map_before)
    assert np.array_equal(belief.frontier_u8, raw_before)


def test_mixed_observed_and_never_unknown_keeps_frontier():
    """Case C: one genuinely unseen neighbor is sufficient."""
    belief = semantic_corridor_belief(INVISIBLE, EMPTY, INVISIBLE)
    accumulator = BeliefEvidenceAccumulator(named_fusion_config("candidate_a"))
    accumulator.observe((), (), {(60, 59)})

    snapshot = frontier_semantics_snapshot(
        belief,
        accumulator,
        "evidence_aware",
    )

    assert snapshot.observed_unclassified_mask[2, 1]
    assert snapshot.never_observed_unknown_mask[2, 3]
    assert snapshot.effective_frontier_u8[2, 2] == 255


def test_frontier_advances_after_ambiguous_cell_becomes_free():
    """Case D: supported FREE evidence advances toward unseen space."""
    belief = semantic_corridor_belief(EMPTY, INVISIBLE, INVISIBLE)
    accumulator = BeliefEvidenceAccumulator(named_fusion_config("candidate_a"))
    accumulator.observe((), (), {ORIGIN_STATE})
    frame = evidence_observation(free={ORIGIN_STATE})

    apply_evidence_fusion(belief, accumulator, frame)
    apply_evidence_fusion(belief, accumulator, frame)
    snapshot = frontier_semantics_snapshot(
        belief,
        accumulator,
        "evidence_aware",
    )

    assert belief.map[2, 2] == EMPTY
    assert snapshot.effective_frontier_u8[2, 1] == 0
    assert snapshot.effective_frontier_u8[2, 2] == 255


def test_frontier_does_not_cross_ambiguous_cell_that_becomes_obstacle():
    """Case E: supported OBSTACLE evidence blocks frontier propagation."""
    belief = semantic_corridor_belief(EMPTY, INVISIBLE, INVISIBLE)
    accumulator = BeliefEvidenceAccumulator(named_fusion_config("candidate_a"))
    accumulator.observe((), (), {ORIGIN_STATE})
    frame = evidence_observation(obstacle={ORIGIN_STATE})

    apply_evidence_fusion(belief, accumulator, frame)
    apply_evidence_fusion(belief, accumulator, frame)
    snapshot = frontier_semantics_snapshot(
        belief,
        accumulator,
        "evidence_aware",
    )

    assert belief.map[2, 2] == OBSTACLE
    assert not np.any(snapshot.effective_frontier_u8)


def test_occlusion_suppressed_cell_is_not_observed():
    """Case F: suppression metadata never enters the accumulator."""
    belief = semantic_corridor_belief(EMPTY, INVISIBLE, INVISIBLE)
    accumulator = BeliefEvidenceAccumulator(named_fusion_config("candidate_a"))
    rear = (60, 61)
    observation = ProjectedBeliefObservation(
        local_snap=np.full((3, 3), INVISIBLE, dtype=np.int8),
        obstacle_cells=frozenset(),
        occlusion_suppressed_obstacle_cells=frozenset({rear}),
    )

    apply_evidence_fusion(belief, accumulator, observation)
    snapshot = frontier_semantics_snapshot(
        belief,
        accumulator,
        "evidence_aware",
    )

    assert not accumulator.has_accepted_evidence(rear)
    assert snapshot.never_observed_unknown_mask[2, 3]
    assert not snapshot.observed_unclassified_mask[2, 3]


def test_conflict_only_cell_is_observed_but_unclassified():
    """Case G: repeated conflict remains categorical UNKNOWN but observed."""
    belief = semantic_corridor_belief(EMPTY, INVISIBLE, OBSTACLE)
    accumulator = BeliefEvidenceAccumulator(named_fusion_config("candidate_a"))

    for _index in range(10):
        apply_evidence_fusion(
            belief,
            accumulator,
            evidence_observation(conflict={ORIGIN_STATE}),
        )
    snapshot = frontier_semantics_snapshot(
        belief,
        accumulator,
        "evidence_aware",
    )
    state = accumulator.state_for(ORIGIN_STATE)

    assert state.free_frame_count == 0
    assert state.obstacle_frame_count == 0
    assert state.conflict_frame_count == 10
    assert belief.map[2, 2] == INVISIBLE
    assert snapshot.observed_unclassified_mask[2, 2]
    assert not snapshot.never_observed_unknown_mask[2, 2]


def test_legacy_frontier_semantics_is_exact_raw_regression():
    """Case H: the default mode returns the unfiltered core frontier."""
    belief = semantic_corridor_belief(EMPTY, INVISIBLE, OBSTACLE)
    accumulator = BeliefEvidenceAccumulator(named_fusion_config("candidate_a"))
    accumulator.observe((), (), {ORIGIN_STATE})
    raw_before = belief.frontier_u8.copy()

    snapshot = frontier_semantics_snapshot(belief, accumulator)

    assert snapshot.effective_frontier_u8.dtype == raw_before.dtype
    assert snapshot.effective_frontier_u8.shape == raw_before.shape
    assert np.array_equal(snapshot.effective_frontier_u8, raw_before)
    assert snapshot.frontier_removed_by_observed_unknown_count == 0


def test_single_obstacle_frame_does_not_poison_coarse_cell():
    """One endpoint cannot change a free cell under candidate evidence mode."""
    belief = BeliefCacheHarness()
    accumulator = BeliefEvidenceAccumulator(
        named_fusion_config("candidate_a")
    )

    stats = apply_evidence_fusion(
        belief,
        accumulator,
        evidence_observation(obstacle={ORIGIN_STATE}),
    )

    assert belief.map[2, 2] == EMPTY
    assert stats.free_to_obstacle_transitions_this_step == 0


def test_repeated_obstacle_frames_can_classify_unvisited_cell():
    """Configured endpoint support can transition FREE to OBSTACLE."""
    belief = BeliefCacheHarness()
    accumulator = BeliefEvidenceAccumulator(
        named_fusion_config("candidate_a")
    )
    frame = evidence_observation(obstacle={ORIGIN_STATE})

    apply_evidence_fusion(belief, accumulator, frame)
    stats = apply_evidence_fusion(belief, accumulator, frame)

    assert belief.map[2, 2] == OBSTACLE
    assert stats.free_to_obstacle_transitions_this_step == 1


def test_repeated_free_frames_reclassify_unvisited_obstacle():
    """Sustained free evidence reverses an earlier obstacle classification."""
    belief = BeliefCacheHarness()
    accumulator = BeliefEvidenceAccumulator(
        named_fusion_config("candidate_a")
    )
    obstacle = evidence_observation(obstacle={ORIGIN_STATE})
    free = evidence_observation(free={ORIGIN_STATE})
    apply_evidence_fusion(belief, accumulator, obstacle)
    apply_evidence_fusion(belief, accumulator, obstacle)

    for _index in range(3):
        stats = apply_evidence_fusion(belief, accumulator, free)

    assert belief.map[2, 2] == EMPTY
    assert stats.obstacle_to_free_transitions_this_step == 1


def test_supported_obstacle_is_not_erased_by_one_free_frame():
    """One isolated free frame cannot erase a supported obstacle."""
    belief = BeliefCacheHarness()
    accumulator = BeliefEvidenceAccumulator(
        named_fusion_config("candidate_a")
    )
    obstacle = evidence_observation(obstacle={ORIGIN_STATE})
    apply_evidence_fusion(belief, accumulator, obstacle)
    apply_evidence_fusion(belief, accumulator, obstacle)

    stats = apply_evidence_fusion(
        belief,
        accumulator,
        evidence_observation(free={ORIGIN_STATE}),
    )

    assert belief.map[2, 2] == OBSTACLE
    assert stats.obstacle_to_free_transitions_this_step == 0


def test_evidence_endpoint_cannot_change_visited_cell():
    """Visited FREE remains authoritative after supported endpoints."""
    belief = BeliefCacheHarness()
    belief.visit_count[2, 2] = 1
    accumulator = BeliefEvidenceAccumulator(
        named_fusion_config("candidate_a")
    )
    frame = evidence_observation(obstacle={ORIGIN_STATE})

    for _index in range(5):
        apply_evidence_fusion(belief, accumulator, frame)

    assert belief.map[2, 2] == EMPTY


def test_evidence_obstacle_to_free_refreshes_frontier_and_caches():
    """Evidence reversal restores frontier and invalidates obstacle caches."""
    belief = BeliefCacheHarness()
    accumulator = BeliefEvidenceAccumulator(
        named_fusion_config("candidate_a")
    )
    obstacle = evidence_observation(obstacle={ORIGIN_STATE})
    free = evidence_observation(free={ORIGIN_STATE})
    apply_evidence_fusion(belief, accumulator, obstacle)
    apply_evidence_fusion(belief, accumulator, obstacle)
    belief._cached_obstacle_integral = np.ones((2, 2), dtype=np.int64)
    belief._latest_frontier_stats = object()

    for _index in range(3):
        apply_evidence_fusion(belief, accumulator, free)

    assert belief.map[2, 2] == EMPTY
    assert belief.get_frontier_u8()[2, 2] == 255
    assert belief._cached_obstacle_integral is None
    assert belief._latest_frontier_stats is None


def test_evidence_free_to_obstacle_refreshes_frontier_and_caches():
    """Supported endpoints remove stale frontier state."""
    belief = BeliefCacheHarness()
    accumulator = BeliefEvidenceAccumulator(
        named_fusion_config("candidate_a")
    )
    frame = evidence_observation(obstacle={ORIGIN_STATE})

    apply_evidence_fusion(belief, accumulator, frame)
    apply_evidence_fusion(belief, accumulator, frame)

    assert belief.map[2, 2] == OBSTACLE
    assert belief.get_frontier_u8()[2, 2] == 0
    assert belief._cached_obstacle_integral is None
    assert belief._latest_frontier_stats is None


def test_legacy_mode_preserves_one_frame_obstacle_promotion():
    """Explicit legacy fusion retains the original one-endpoint behavior."""
    belief = BeliefCacheHarness()
    stats = apply_legacy_fusion(
        belief,
        evidence_observation(obstacle={ORIGIN_STATE}),
    )

    assert belief.map[2, 2] == OBSTACLE
    assert stats.free_to_obstacle_transitions_this_step == 1


def test_candidate_fusion_is_independent_of_cell_iteration_order():
    """Set/list iteration order cannot change counts or classifications."""
    config = BeliefFusionConfig(
        name="order_test",
        minimum_free_frames=1,
        minimum_obstacle_frames=1,
        free_evidence_margin=1,
        obstacle_evidence_margin=1,
    )
    first = BeliefEvidenceAccumulator(config)
    second = BeliefEvidenceAccumulator(config)
    free = [(60, 60), (60, 61), (61, 60)]
    obstacle = [(61, 61), (62, 62)]

    first.observe(free, obstacle)
    second.observe(reversed(free), reversed(obstacle))

    assert first.cells == second.cells
    assert {
        cell: first.classify(cell) for cell in first.cells
    } == {
        cell: second.classify(cell) for cell in second.cells
    }
