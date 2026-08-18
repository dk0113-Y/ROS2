"""Deployment-only LiDAR projection and conservative belief fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


INVISIBLE = -1
EMPTY = 0
OBSTACLE = 1


@dataclass(frozen=True)
class ProjectedBeliefObservation:
    """A local categorical observation plus exact global obstacle cells."""

    local_snap: np.ndarray
    obstacle_cells: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class ObstaclePromotionStats:
    """Count deployment-only obstacle-dominant belief transitions."""

    obstacle_promotions_this_step: int
    promoted_from_empty: int
    promoted_from_invisible: int


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
    """Project scan rays through continuous odom and the LiDAR extrinsic."""
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

    def mark_world(world_x: float, world_y: float, value: int) -> bool:
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
            return False
        local_row = center + dr
        local_col = center + dc
        if not (0 <= local_row < local_size and 0 <= local_col < local_size):
            return False
        if value == OBSTACLE:
            local_snap[local_row, local_col] = OBSTACLE
        elif local_snap[local_row, local_col] == INVISIBLE:
            local_snap[local_row, local_col] = EMPTY
        return True

    for index, raw_range in enumerate(scan.ranges):
        hit_range = float(raw_range)
        if not (
            math.isfinite(hit_range)
            and float(scan.range_min) <= hit_range <= float(scan.range_max)
        ):
            continue

        ray_range = hit_range
        ray_range = min(max(0.0, ray_range), local_radius_m + cell_size)
        free_end = max(0.0, ray_range - cell_size * 0.25)
        world_yaw = (
            float(robot_yaw)
            + float(laser_yaw_in_base)
            + float(scan.angle_min)
            + index * float(scan.angle_increment)
        )
        ray_cos = math.cos(world_yaw)
        ray_sin = math.sin(world_yaw)

        distance = 0.0
        seen_free: set[tuple[int, int]] = set()
        while distance <= free_end:
            free_x = laser_origin_x + distance * ray_cos
            free_y = laser_origin_y + distance * ray_sin
            free_cell = continuous_world_to_grid(
                free_x,
                free_y,
                origin_x,
                origin_y,
                origin_state,
                cell_size,
            )
            if free_cell not in seen_free:
                seen_free.add(free_cell)
                mark_world(free_x, free_y, EMPTY)
            distance += sample_step

        if hit_range <= local_radius_m + cell_size:
            hit_x = laser_origin_x + hit_range * ray_cos
            hit_y = laser_origin_y + hit_range * ray_sin
            hit_cell = continuous_world_to_grid(
                hit_x,
                hit_y,
                origin_x,
                origin_y,
                origin_state,
                cell_size,
            )
            if mark_world(hit_x, hit_y, OBSTACLE):
                obstacle_cells.add(hit_cell)

    return ProjectedBeliefObservation(
        local_snap=local_snap,
        obstacle_cells=frozenset(obstacle_cells),
    )


def promote_observed_obstacle_cells(
    cum_map: Any,
    obstacle_cells: Iterable[tuple[int, int]],
) -> ObstaclePromotionStats:
    """Promote observed cells to obstacles using the core's cache hooks."""
    unique_cells = sorted(
        {(int(row), int(col)) for row, col in obstacle_cells}
    )
    if not unique_cells:
        return ObstaclePromotionStats(0, 0, 0)

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
    promote = previous != OBSTACLE
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
    )
