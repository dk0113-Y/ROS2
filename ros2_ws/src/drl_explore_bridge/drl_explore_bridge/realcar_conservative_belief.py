"""Deployment-only LiDAR projection and conservative belief fusion."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np


INVISIBLE = -1
EMPTY = 0
OBSTACLE = 1


@dataclass(frozen=True)
class ProjectedBeliefObservation:
    """A legacy local snapshot plus deduplicated global frame evidence."""

    local_snap: np.ndarray
    obstacle_cells: frozenset[tuple[int, int]]
    free_cells: frozenset[tuple[int, int]] = frozenset()
    conflict_cells: frozenset[tuple[int, int]] = frozenset()


@dataclass(frozen=True)
class BeliefFusionConfig:
    """Explicit hysteresis thresholds for deployment evidence fusion."""

    name: str
    minimum_free_frames: int
    minimum_obstacle_frames: int
    free_evidence_margin: int
    obstacle_evidence_margin: int
    minimum_free_streak: int = 1
    minimum_obstacle_streak: int = 1

    def __post_init__(self) -> None:
        """Reject configurations that cannot provide hysteresis."""
        if not self.name:
            raise ValueError("fusion config name must not be empty")
        if any(
            not (character.isalnum() or character in "_-")
            for character in self.name
        ):
            raise ValueError(
                "fusion config name may contain only letters, digits, _ and -"
            )
        for field_name in (
            "minimum_free_frames",
            "minimum_obstacle_frames",
            "minimum_free_streak",
            "minimum_obstacle_streak",
        ):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} must be >= 1")
        for field_name in (
            "free_evidence_margin",
            "obstacle_evidence_margin",
        ):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} must be >= 1")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration record."""
        return asdict(self)


EVIDENCE_FUSION_CANDIDATES: dict[str, BeliefFusionConfig] = {
    "candidate_a": BeliefFusionConfig(
        name="candidate_a",
        minimum_free_frames=2,
        minimum_obstacle_frames=2,
        free_evidence_margin=1,
        obstacle_evidence_margin=1,
    ),
    "candidate_b": BeliefFusionConfig(
        name="candidate_b",
        minimum_free_frames=2,
        minimum_obstacle_frames=3,
        free_evidence_margin=1,
        obstacle_evidence_margin=2,
    ),
    "candidate_c": BeliefFusionConfig(
        name="candidate_c",
        minimum_free_frames=2,
        minimum_obstacle_frames=2,
        free_evidence_margin=1,
        obstacle_evidence_margin=1,
        minimum_free_streak=2,
        minimum_obstacle_streak=2,
    ),
}


def named_fusion_config(name: str) -> BeliefFusionConfig:
    """Return one immutable built-in evidence policy by name."""
    try:
        return EVIDENCE_FUSION_CANDIDATES[str(name)]
    except KeyError as exc:
        choices = ", ".join(sorted(EVIDENCE_FUSION_CANDIDATES))
        raise ValueError(
            f"unknown evidence fusion config {name!r}; choose {choices}"
        ) from exc


@dataclass
class CellEvidenceState:
    """Accumulate decision-frame evidence for one global policy cell."""

    free_frame_count: int = 0
    obstacle_frame_count: int = 0
    conflict_frame_count: int = 0
    consecutive_free_frames: int = 0
    consecutive_obstacle_frames: int = 0


@dataclass(frozen=True)
class MapTransitionStats:
    """Count categorical transitions applied through deployment hooks."""

    invisible_to_free: int = 0
    invisible_to_obstacle: int = 0
    free_to_obstacle: int = 0
    obstacle_to_free: int = 0


@dataclass(frozen=True)
class FusionStepStats:
    """Describe one frame of evidence and resulting map transitions."""

    evidence_free_cells_this_step: int
    evidence_obstacle_cells_this_step: int
    evidence_conflict_cells_this_step: int
    invisible_to_free_transitions_this_step: int
    invisible_to_obstacle_transitions_this_step: int
    free_to_obstacle_transitions_this_step: int
    obstacle_to_free_transitions_this_step: int


class BeliefEvidenceAccumulator:
    """Fuse frame-deduplicated global evidence with explicit hysteresis."""

    def __init__(self, config: BeliefFusionConfig):
        """Create an empty accumulator using one immutable policy."""
        self.config = config
        self.cells: dict[tuple[int, int], CellEvidenceState] = {}

    def state_for(self, cell: tuple[int, int]) -> CellEvidenceState:
        """Return the mutable evidence state for a normalized global cell."""
        normalized = (int(cell[0]), int(cell[1]))
        return self.cells.setdefault(normalized, CellEvidenceState())

    def observe(
        self,
        free_cells: Iterable[tuple[int, int]],
        obstacle_cells: Iterable[tuple[int, int]],
        conflict_cells: Iterable[tuple[int, int]] = (),
    ) -> frozenset[tuple[int, int]]:
        """Record one decision frame, counting each cell at most once."""
        free = {(int(row), int(col)) for row, col in free_cells}
        obstacle = {
            (int(row), int(col)) for row, col in obstacle_cells
        }
        conflicts = {
            (int(row), int(col)) for row, col in conflict_cells
        }
        conflicts.update(free & obstacle)
        free.difference_update(conflicts)
        obstacle.difference_update(conflicts)

        for cell in sorted(free | obstacle | conflicts):
            state = self.state_for(cell)
            if cell in conflicts:
                state.conflict_frame_count += 1
                state.consecutive_free_frames = 0
                state.consecutive_obstacle_frames = 0
            elif cell in free:
                state.free_frame_count += 1
                state.consecutive_free_frames += 1
                state.consecutive_obstacle_frames = 0
            else:
                state.obstacle_frame_count += 1
                state.consecutive_obstacle_frames += 1
                state.consecutive_free_frames = 0
        return frozenset(free | obstacle | conflicts)

    def classify(self, cell: tuple[int, int]) -> Optional[int]:
        """Classify a cell only when one side clears all configured gates."""
        state = self.state_for(cell)
        config = self.config
        free_ready = (
            state.free_frame_count >= config.minimum_free_frames
            and state.consecutive_free_frames >= config.minimum_free_streak
            and state.free_frame_count - state.obstacle_frame_count
            >= config.free_evidence_margin
        )
        obstacle_ready = (
            state.obstacle_frame_count >= config.minimum_obstacle_frames
            and state.consecutive_obstacle_frames
            >= config.minimum_obstacle_streak
            and state.obstacle_frame_count - state.free_frame_count
            >= config.obstacle_evidence_margin
        )
        if free_ready == obstacle_ready:
            return None
        return EMPTY if free_ready else OBSTACLE


@dataclass(frozen=True)
class ObstaclePromotionStats:
    """Count deployment-only obstacle-dominant belief transitions."""

    obstacle_promotions_this_step: int
    promoted_from_empty: int
    promoted_from_invisible: int
    visited_obstacle_to_free: int = 0


@dataclass(frozen=True)
class VisitedCellCorrectionStats:
    """Count physical-traversal corrections to cumulative belief."""

    corrected_from_obstacle: int
    revealed_from_invisible: int


def _nearest_cell_offset(distance: float, cell_size: float) -> int:
    """Quantize a continuous center-relative distance to one grid cell."""
    return int(math.floor(float(distance) / float(cell_size) + 0.5))


def continuous_world_to_grid(
    world_x: float,
    world_y: float,
    origin_x: float,
    origin_y: float,
    origin_state: tuple[int, int],
    cell_size: float,
) -> tuple[int, int]:
    """Quantize continuous odom-world coordinates using the DRL grid axes."""
    if not math.isfinite(cell_size) or cell_size <= 0.0:
        raise ValueError("cell_size must be finite and > 0")
    row = int(origin_state[0]) + _nearest_cell_offset(
        -(float(world_y) - float(origin_y)),
        cell_size,
    )
    col = int(origin_state[1]) + _nearest_cell_offset(
        float(world_x) - float(origin_x),
        cell_size,
    )
    return row, col


def project_scan_to_belief(
    scan: Any,
    *,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    origin_x: float,
    origin_y: float,
    origin_state: tuple[int, int],
    agent_state: tuple[int, int],
    cell_size: float,
    scan_radius_cells: int,
    laser_x_in_base: float,
    laser_y_in_base: float,
    laser_yaw_in_base: float,
) -> ProjectedBeliefObservation:
    """Project one scan into legacy categories and global frame evidence."""
    if scan_radius_cells < 1:
        raise ValueError("scan_radius_cells must be >= 1")

    local_size = 2 * int(scan_radius_cells) + 1
    center = int(scan_radius_cells)
    local_snap = np.full(
        (local_size, local_size),
        INVISIBLE,
        dtype=np.int8,
    )
    local_snap[center, center] = EMPTY
    free_cells: set[tuple[int, int]] = set()
    obstacle_cells: set[tuple[int, int]] = set()

    robot_cos = math.cos(robot_yaw)
    robot_sin = math.sin(robot_yaw)
    laser_origin_x = (
        float(robot_x)
        + robot_cos * float(laser_x_in_base)
        - robot_sin * float(laser_y_in_base)
    )
    laser_origin_y = (
        float(robot_y)
        + robot_sin * float(laser_x_in_base)
        + robot_cos * float(laser_y_in_base)
    )
    local_radius_m = int(scan_radius_cells) * float(cell_size)
    sample_step = float(cell_size) / 3.0

    def visible_world_cell(
        world_x: float,
        world_y: float,
    ) -> Optional[tuple[int, int]]:
        world_cell = continuous_world_to_grid(
            world_x,
            world_y,
            origin_x,
            origin_y,
            origin_state,
            cell_size,
        )
        dr = int(world_cell[0]) - int(agent_state[0])
        dc = int(world_cell[1]) - int(agent_state[1])
        if dr * dr + dc * dc > scan_radius_cells * scan_radius_cells:
            return None
        local_row = center + dr
        local_col = center + dc
        if not (0 <= local_row < local_size and 0 <= local_col < local_size):
            return None
        return world_cell

    for index, raw_range in enumerate(scan.ranges):
        hit_range = float(raw_range)
        if not (
            math.isfinite(hit_range)
            and float(scan.range_min) <= hit_range <= float(scan.range_max)
        ):
            continue

        ray_range = min(
            max(0.0, hit_range),
            local_radius_m + cell_size,
        )
        free_end = max(0.0, ray_range - cell_size * 0.25)
        world_yaw = (
            float(robot_yaw)
            + float(laser_yaw_in_base)
            + float(scan.angle_min)
            + index * float(scan.angle_increment)
        )
        ray_cos = math.cos(world_yaw)
        ray_sin = math.sin(world_yaw)

        hit_cell: Optional[tuple[int, int]] = None
        if hit_range <= local_radius_m + cell_size:
            hit_x = laser_origin_x + hit_range * ray_cos
            hit_y = laser_origin_y + hit_range * ray_sin
            hit_cell = visible_world_cell(hit_x, hit_y)

        distance = 0.0
        ray_free_cells: set[tuple[int, int]] = set()
        while distance <= free_end:
            free_x = laser_origin_x + distance * ray_cos
            free_y = laser_origin_y + distance * ray_sin
            free_cell = visible_world_cell(free_x, free_y)
            if free_cell is not None and free_cell != hit_cell:
                ray_free_cells.add(free_cell)
            distance += sample_step
        free_cells.update(ray_free_cells)

        if hit_cell is not None:
            obstacle_cells.add(hit_cell)

    conflict_cells = free_cells & obstacle_cells

    # Preserve the exact legacy categorical rule: any endpoint in a coarse
    # cell dominates free-ray projection in that local snapshot.  Evidence
    # mode does not consume this categorical snapshot, but legacy A/B replay
    # and the live legacy default do.
    for world_row, world_col in sorted(free_cells):
        local_row = center + int(world_row) - int(agent_state[0])
        local_col = center + int(world_col) - int(agent_state[1])
        if local_snap[local_row, local_col] == INVISIBLE:
            local_snap[local_row, local_col] = EMPTY
    for world_row, world_col in sorted(obstacle_cells):
        local_row = center + int(world_row) - int(agent_state[0])
        local_col = center + int(world_col) - int(agent_state[1])
        local_snap[local_row, local_col] = OBSTACLE

    return ProjectedBeliefObservation(
        local_snap=local_snap,
        obstacle_cells=frozenset(obstacle_cells),
        free_cells=frozenset(free_cells),
        conflict_cells=frozenset(conflict_cells),
    )


def promote_observed_obstacle_cells(
    cum_map: Any,
    obstacle_cells: Iterable[tuple[int, int]],
) -> ObstaclePromotionStats:
    """Promote only unvisited cells using the core's cache hooks."""
    visited_corrections = preserve_visited_cells_as_free(cum_map)
    unique_cells = sorted(
        {(int(row), int(col)) for row, col in obstacle_cells}
    )
    if not unique_cells:
        return ObstaclePromotionStats(
            0,
            0,
            0,
            visited_corrections.corrected_from_obstacle,
        )

    world_rows = np.asarray([cell[0] for cell in unique_cells], dtype=np.int32)
    world_cols = np.asarray([cell[1] for cell in unique_cells], dtype=np.int32)
    expansion = cum_map._ensure_world_bounds(
        int(world_rows.min()),
        int(world_rows.max()),
        int(world_cols.min()),
        int(world_cols.max()),
    )
    dirty_rects: list[Any] = []
    if expansion is not None:
        dirty_rects.extend(expansion.seam_dirty_rects)

    array_rows = world_rows - int(cum_map.origin_world_rc[0])
    array_cols = world_cols - int(cum_map.origin_world_rc[1])
    previous = np.asarray(cum_map.map[array_rows, array_cols]).copy()
    unvisited = np.asarray(
        cum_map.visit_count[array_rows, array_cols] <= 0,
        dtype=bool,
    )
    promote = unvisited & (previous != OBSTACLE)
    promoted_count = int(np.count_nonzero(promote))
    promoted_from_empty = int(np.count_nonzero(previous[promote] == EMPTY))
    promoted_from_invisible = int(
        np.count_nonzero(previous[promote] == INVISIBLE)
    )

    if promoted_count:
        promoted_rows = array_rows[promote]
        promoted_cols = array_cols[promote]
        cum_map.map[promoted_rows, promoted_cols] = OBSTACLE
        if promoted_from_invisible:
            cum_map.kpm_count += cum_map._count_coverage_hits(
                world_rows[promote][previous[promote] == INVISIBLE],
                world_cols[promote][previous[promote] == INVISIBLE],
            )
        cum_map._invalidate_map_state_caches()
        dirty = cum_map._dirty_rect_from_points(
            promoted_rows,
            promoted_cols,
        )
        dirty = cum_map._expand_dirty_rect(dirty, radius=1)
        if dirty is not None:
            dirty_rects.append(dirty)

    cum_map._update_frontier_dirty_rects(dirty_rects)
    cum_map._refresh_coverage()
    cum_map._update_analysis_box()
    return ObstaclePromotionStats(
        promoted_count,
        promoted_from_empty,
        promoted_from_invisible,
        visited_corrections.corrected_from_obstacle,
    )


def visited_only_local_snap(local_shape: tuple[int, int]) -> np.ndarray:
    """Build a core-compatible snapshot containing only robot occupancy."""
    if len(local_shape) != 2 or min(int(value) for value in local_shape) < 1:
        raise ValueError("local_shape must contain two positive dimensions")
    snap = np.full(
        tuple(int(value) for value in local_shape),
        INVISIBLE,
        dtype=np.int8,
    )
    snap[snap.shape[0] // 2, snap.shape[1] // 2] = EMPTY
    return snap


def apply_world_categorical_transitions(
    cum_map: Any,
    desired_values: Mapping[tuple[int, int], int],
) -> MapTransitionStats:
    """Apply deployment transitions while refreshing every core cache hook."""
    normalized = {
        (int(cell[0]), int(cell[1])): int(value)
        for cell, value in desired_values.items()
    }
    invalid_values = set(normalized.values()) - {EMPTY, OBSTACLE}
    if invalid_values:
        raise ValueError(
            "categorical transitions require FREE/OBSTACLE, got "
            f"{invalid_values}"
        )
    if not normalized:
        return MapTransitionStats()

    cells = sorted(normalized)
    world_rows = np.asarray([cell[0] for cell in cells], dtype=np.int32)
    world_cols = np.asarray([cell[1] for cell in cells], dtype=np.int32)
    targets = np.asarray([normalized[cell] for cell in cells], dtype=np.int8)
    expansion = cum_map._ensure_world_bounds(
        int(world_rows.min()),
        int(world_rows.max()),
        int(world_cols.min()),
        int(world_cols.max()),
    )
    dirty_rects: list[Any] = []
    if expansion is not None:
        dirty_rects.extend(expansion.seam_dirty_rects)

    array_rows = world_rows - int(cum_map.origin_world_rc[0])
    array_cols = world_cols - int(cum_map.origin_world_rc[1])
    previous = np.asarray(cum_map.map[array_rows, array_cols]).copy()
    visited = np.asarray(
        cum_map.visit_count[array_rows, array_cols] > 0,
        dtype=bool,
    )
    targets[visited] = EMPTY
    changed = previous != targets

    invisible_to_free = int(
        np.count_nonzero(
            changed & (previous == INVISIBLE) & (targets == EMPTY)
        )
    )
    invisible_to_obstacle = int(
        np.count_nonzero(
            changed & (previous == INVISIBLE) & (targets == OBSTACLE)
        )
    )
    free_to_obstacle = int(
        np.count_nonzero(changed & (previous == EMPTY) & (targets == OBSTACLE))
    )
    obstacle_to_free = int(
        np.count_nonzero(changed & (previous == OBSTACLE) & (targets == EMPTY))
    )

    if np.any(changed):
        changed_rows = array_rows[changed]
        changed_cols = array_cols[changed]
        cum_map.map[changed_rows, changed_cols] = targets[changed]
        newly_known = changed & (previous == INVISIBLE)
        if np.any(newly_known):
            cum_map.kpm_count += cum_map._count_coverage_hits(
                world_rows[newly_known],
                world_cols[newly_known],
            )
        cum_map._invalidate_map_state_caches()
        dirty = cum_map._dirty_rect_from_points(changed_rows, changed_cols)
        dirty = cum_map._expand_dirty_rect(dirty, radius=1)
        if dirty is not None:
            dirty_rects.append(dirty)

    cum_map._update_frontier_dirty_rects(dirty_rects)
    cum_map._refresh_coverage()
    cum_map._update_analysis_box()
    return MapTransitionStats(
        invisible_to_free=invisible_to_free,
        invisible_to_obstacle=invisible_to_obstacle,
        free_to_obstacle=free_to_obstacle,
        obstacle_to_free=obstacle_to_free,
    )


def apply_evidence_fusion(
    cum_map: Any,
    accumulator: BeliefEvidenceAccumulator,
    observation: ProjectedBeliefObservation,
) -> FusionStepStats:
    """Fuse one projected decision frame into the cumulative belief."""
    visited_corrections = preserve_visited_cells_as_free(cum_map)
    touched = accumulator.observe(
        observation.free_cells,
        observation.obstacle_cells,
        observation.conflict_cells,
    )
    conflicts = set(observation.conflict_cells)
    conflicts.update(
        set(observation.free_cells) & set(observation.obstacle_cells)
    )
    desired: dict[tuple[int, int], int] = {}
    for cell in sorted(touched):
        if cell in conflicts:
            continue
        classification = accumulator.classify(cell)
        if classification is not None:
            desired[cell] = classification
    transitions = apply_world_categorical_transitions(cum_map, desired)
    return FusionStepStats(
        evidence_free_cells_this_step=len(observation.free_cells),
        evidence_obstacle_cells_this_step=len(observation.obstacle_cells),
        evidence_conflict_cells_this_step=len(conflicts),
        invisible_to_free_transitions_this_step=(
            transitions.invisible_to_free
        ),
        invisible_to_obstacle_transitions_this_step=(
            transitions.invisible_to_obstacle
        ),
        free_to_obstacle_transitions_this_step=(
            transitions.free_to_obstacle
        ),
        obstacle_to_free_transitions_this_step=(
            transitions.obstacle_to_free
            + visited_corrections.corrected_from_obstacle
        ),
    )


def apply_legacy_fusion(
    cum_map: Any,
    observation: ProjectedBeliefObservation,
) -> FusionStepStats:
    """Apply the pre-evidence one-endpoint promotion behavior explicitly."""
    promotion = promote_observed_obstacle_cells(
        cum_map,
        observation.obstacle_cells,
    )
    conflicts = set(observation.conflict_cells)
    conflicts.update(
        set(observation.free_cells) & set(observation.obstacle_cells)
    )
    return FusionStepStats(
        evidence_free_cells_this_step=len(observation.free_cells),
        evidence_obstacle_cells_this_step=len(observation.obstacle_cells),
        evidence_conflict_cells_this_step=len(conflicts),
        invisible_to_free_transitions_this_step=0,
        invisible_to_obstacle_transitions_this_step=(
            promotion.promoted_from_invisible
        ),
        free_to_obstacle_transitions_this_step=promotion.promoted_from_empty,
        obstacle_to_free_transitions_this_step=(
            promotion.visited_obstacle_to_free
        ),
    )


def final_belief_cell_counts(cum_map: Any) -> dict[str, int]:
    """Count final categorical map cells without mutating belief state."""
    belief = np.asarray(cum_map.map)
    return {
        "final_free_cells": int(np.count_nonzero(belief == EMPTY)),
        "final_obstacle_cells": int(np.count_nonzero(belief == OBSTACLE)),
        "final_unknown_cells": int(np.count_nonzero(belief == INVISIBLE)),
    }


def _apply_visited_free_corrections(
    cum_map: Any,
    array_rows: np.ndarray,
    array_cols: np.ndarray,
    world_rows: np.ndarray,
    world_cols: np.ndarray,
    dirty_rects: list[Any],
) -> VisitedCellCorrectionStats:
    """Set selected visited cells FREE and refresh all derived map state."""
    previous = np.asarray(cum_map.map[array_rows, array_cols]).copy()
    correct = previous != EMPTY
    corrected_from_obstacle = int(
        np.count_nonzero(previous[correct] == OBSTACLE)
    )
    revealed_from_invisible = int(
        np.count_nonzero(previous[correct] == INVISIBLE)
    )

    if np.any(correct):
        corrected_rows = array_rows[correct]
        corrected_cols = array_cols[correct]
        cum_map.map[corrected_rows, corrected_cols] = EMPTY
        if revealed_from_invisible:
            newly_known = previous[correct] == INVISIBLE
            cum_map.kpm_count += cum_map._count_coverage_hits(
                world_rows[correct][newly_known],
                world_cols[correct][newly_known],
            )
        cum_map._invalidate_map_state_caches()
        dirty = cum_map._dirty_rect_from_points(
            corrected_rows,
            corrected_cols,
        )
        dirty = cum_map._expand_dirty_rect(dirty, radius=1)
        if dirty is not None:
            dirty_rects.append(dirty)

    cum_map._update_frontier_dirty_rects(dirty_rects)
    cum_map._refresh_coverage()
    cum_map._update_analysis_box()
    return VisitedCellCorrectionStats(
        corrected_from_obstacle,
        revealed_from_invisible,
    )


def preserve_visited_cells_as_free(
    cum_map: Any,
) -> VisitedCellCorrectionStats:
    """Enforce that every cell already in ``visit_count`` remains FREE."""
    visited_rows, visited_cols = np.nonzero(cum_map.visit_count > 0)
    if visited_rows.size == 0:
        return VisitedCellCorrectionStats(0, 0)

    needs_correction = cum_map.map[visited_rows, visited_cols] != EMPTY
    if not np.any(needs_correction):
        return VisitedCellCorrectionStats(0, 0)

    array_rows = visited_rows[needs_correction].astype(np.int32, copy=False)
    array_cols = visited_cols[needs_correction].astype(np.int32, copy=False)
    world_rows = array_rows + int(cum_map.origin_world_rc[0])
    world_cols = array_cols + int(cum_map.origin_world_rc[1])
    return _apply_visited_free_corrections(
        cum_map,
        array_rows,
        array_cols,
        world_rows,
        world_cols,
        [],
    )


def record_traversed_cells_as_free(
    cum_map: Any,
    traversed_cells: Iterable[tuple[int, int]],
) -> VisitedCellCorrectionStats:
    """Register robot-center trajectory cells and make them authoritative FREE."""
    unique_cells = sorted(
        {(int(row), int(col)) for row, col in traversed_cells}
    )
    if not unique_cells:
        return VisitedCellCorrectionStats(0, 0)

    world_rows = np.asarray([cell[0] for cell in unique_cells], dtype=np.int32)
    world_cols = np.asarray([cell[1] for cell in unique_cells], dtype=np.int32)
    expansion = cum_map._ensure_world_bounds(
        int(world_rows.min()),
        int(world_rows.max()),
        int(world_cols.min()),
        int(world_cols.max()),
    )
    dirty_rects: list[Any] = []
    if expansion is not None:
        dirty_rects.extend(expansion.seam_dirty_rects)

    array_rows = world_rows - int(cum_map.origin_world_rc[0])
    array_cols = world_cols - int(cum_map.origin_world_rc[1])
    newly_visited = cum_map.visit_count[array_rows, array_cols] <= 0
    if np.any(newly_visited):
        cum_map.visit_count[
            array_rows[newly_visited],
            array_cols[newly_visited],
        ] = 1
        cum_map._invalidate_visit_cache()

    return _apply_visited_free_corrections(
        cum_map,
        array_rows,
        array_cols,
        world_rows,
        world_cols,
        dirty_rects,
    )
