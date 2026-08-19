#!/usr/bin/env python3
"""Run the frozen real-car multi-scan evidence-density study offline."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = REPOSITORY_ROOT / "ros2_ws" / "src" / "drl_explore_bridge"
for source_path in (REPOSITORY_ROOT, PACKAGE_SOURCE):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from drl_explore_bridge.realcar_conservative_belief import (  # noqa: E402
    BeliefEvidenceAccumulator,
    EMPTY,
    INVISIBLE,
    OBSTACLE,
    ProjectedBeliefObservation,
    _ordered_coarse_ray_entries,
    apply_evidence_fusion,
    continuous_world_to_grid,
    cumulative_occlusion_cells,
    frontier_semantics_snapshot,
    named_fusion_config,
    observed_unclassified_evidence_cells,
    project_scan_to_belief as production_project_scan_to_belief,
    record_traversed_cells_as_free,
    visited_only_local_snap,
)
from drl_explore_bridge.realcar_policy_continuous_runner_node import (  # noqa
    CONTINUOUS_ORIGIN_STATE,
    DEFAULT_LASER_X_IN_BASE_M,
    DEFAULT_LASER_Y_IN_BASE_M,
    episode_trajectory_states,
)
from scripts_realcar.analyze_belief_fusion_replay import (  # noqa: E402
    _load_cumulative_map_type,
    compare_saved_belief,
)


BASE_SHA = "b0da8ce99d39bca16c48585af78e65116541f617"
EXPECTED_OUTER_SHA256 = (
    "be2f7acc7d82a63c71835695da2a0f3d40e199e49cfaab2399073ec35abf7666"
)
EXPECTED_BASELINE = {
    "known": 55,
    "free": 31,
    "obstacle": 24,
    "unknown": 727,
    "known_history": [1, 15, 18, 21, 24, 35, 44, 45, 49, 55],
    "raw_frontier_history": [0, 12, 12, 15, 17, 22, 20, 21, 21, 20],
    "effective_frontier_history": [0, 4, 3, 3, 4, 4, 5, 5, 6, 6],
    "observed_unclassified_history": [
        0, 36, 38, 37, 38, 28, 20, 22, 21, 19,
    ],
    "never_observed_history": [
        0, 478, 473, 471, 467, 466, 718, 715, 712, 708,
    ],
    "evidence_free": 405,
    "evidence_obstacle": 304,
    "evidence_conflict": 216,
    "occlusion_blocker": 45,
    "occlusion_suppressed_free": 0,
    "occlusion_suppressed_obstacle": 6,
    "occlusion_suppressed_unique": 6,
}


@dataclass(frozen=True)
class RecordedScan:
    """One scan with both header and bag timestamps."""

    timestamp_sec: float
    bag_timestamp_sec: float
    timestamp_source: str
    message: Any


@dataclass(frozen=True)
class OdomPose:
    """One odometry pose in the recorded odom frame."""

    timestamp_sec: float
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class PoseAtTime:
    """An interpolated or nearest-fallback odometry pose."""

    timestamp_sec: float
    x: float
    y: float
    yaw: float
    method: str
    nearest_error_sec: float


@dataclass(frozen=True)
class Decision:
    """A recorded policy decision observation."""

    step_id: int
    scan_timestamp_sec: float
    pose_timestamp_sec: float
    robot_x: float
    robot_y: float
    robot_yaw: float
    agent_state: tuple[int, int]


@dataclass(frozen=True)
class DecisionMatch:
    """An exact or near-exact decision-to-scan association."""

    decision: Decision
    scan: RecordedScan
    absolute_delta_sec: float


@dataclass(frozen=True)
class Transform2D:
    """A timestamped planar TF transform."""

    timestamp_sec: float
    parent_frame: str
    child_frame: str
    x: float
    y: float
    yaw: float


@dataclass
class BagContents:
    """The read-only bag data used by the study."""

    scans: list[RecordedScan]
    odom: list[OdomPose]
    final_map: Any
    final_map_bag_timestamp_sec: Optional[float]
    transforms: list[Transform2D]
    topic_counts: dict[str, int]
    bag_start_sec: float
    bag_end_sec: float
    metadata_duration_sec: float


def _stamp_to_sec(stamp: Any) -> float:
    """Convert a builtin ROS time message to seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def _yaw_from_quaternion(quaternion: Any) -> float:
    """Return planar yaw from a ROS quaternion-like object."""
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _normalize_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return math.atan2(math.sin(angle), math.cos(angle))


def interpolate_yaw(start: float, end: float, fraction: float) -> float:
    """Interpolate yaw on the shortest angular arc."""
    delta = _normalize_angle(float(end) - float(start))
    return _normalize_angle(float(start) + float(fraction) * delta)


def _storage_identifier(bag_path: Path) -> str:
    """Read the configured rosbag2 storage identifier."""
    import yaml

    metadata = yaml.safe_load(
        (bag_path / "metadata.yaml").read_text(encoding="utf-8")
    )
    return str(
        metadata["rosbag2_bagfile_information"]["storage_identifier"]
    )


def read_bag_contents(bag_path: Path) -> BagContents:
    """Deserialize only the topics needed for offline analysis."""
    import rosbag2_py
    import yaml
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    metadata = yaml.safe_load(
        (bag_path / "metadata.yaml").read_text(encoding="utf-8")
    )["rosbag2_bagfile_information"]
    metadata_duration = float(metadata["duration"]["nanoseconds"]) * 1e-9
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(bag_path),
            storage_id=_storage_identifier(bag_path),
        ),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    message_types = {
        topic: get_message(type_name)
        for topic, type_name in topic_types.items()
        if topic in {"/scan", "/odom", "/map", "/tf", "/tf_static"}
    }
    scans: list[RecordedScan] = []
    odom: list[OdomPose] = []
    transforms: list[Transform2D] = []
    final_map = None
    final_map_timestamp = None
    counts: Counter[str] = Counter()
    first_bag_time = math.inf
    last_bag_time = -math.inf
    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()
        bag_time = float(timestamp_ns) * 1e-9
        counts[topic] += 1
        first_bag_time = min(first_bag_time, bag_time)
        last_bag_time = max(last_bag_time, bag_time)
        if topic not in message_types:
            continue
        message = deserialize_message(serialized, message_types[topic])
        if topic == "/scan":
            header_time = _stamp_to_sec(message.header.stamp)
            selected = header_time if header_time > 0.0 else bag_time
            scans.append(
                RecordedScan(
                    timestamp_sec=selected,
                    bag_timestamp_sec=bag_time,
                    timestamp_source=(
                        "header" if header_time > 0.0 else "bag"
                    ),
                    message=message,
                )
            )
        elif topic == "/odom":
            header_time = _stamp_to_sec(message.header.stamp)
            selected = header_time if header_time > 0.0 else bag_time
            pose = message.pose.pose
            odom.append(
                OdomPose(
                    timestamp_sec=selected,
                    x=float(pose.position.x),
                    y=float(pose.position.y),
                    yaw=_yaw_from_quaternion(pose.orientation),
                )
            )
        elif topic == "/map":
            final_map = message
            final_map_timestamp = bag_time
        else:
            for transform in message.transforms:
                stamp = _stamp_to_sec(transform.header.stamp)
                if stamp <= 0.0:
                    stamp = bag_time
                translation = transform.transform.translation
                transforms.append(
                    Transform2D(
                        timestamp_sec=stamp,
                        parent_frame=str(
                            transform.header.frame_id
                        ).lstrip("/"),
                        child_frame=str(transform.child_frame_id).lstrip("/"),
                        x=float(translation.x),
                        y=float(translation.y),
                        yaw=_yaw_from_quaternion(
                            transform.transform.rotation
                        ),
                    )
                )
    scans.sort(key=lambda item: item.timestamp_sec)
    odom.sort(key=lambda item: item.timestamp_sec)
    transforms.sort(key=lambda item: item.timestamp_sec)
    if not scans or not odom:
        raise ValueError("bag must contain non-empty /scan and /odom topics")
    return BagContents(
        scans=scans,
        odom=odom,
        final_map=final_map,
        final_map_bag_timestamp_sec=final_map_timestamp,
        transforms=transforms,
        topic_counts=dict(sorted(counts.items())),
        bag_start_sec=first_bag_time,
        bag_end_sec=last_bag_time,
        metadata_duration_sec=metadata_duration,
    )


def read_decisions(episode: dict[str, Any]) -> list[Decision]:
    """Read the scan timestamp and JSON pose for every decision."""
    decisions: list[Decision] = []
    for fallback_id, step in enumerate(episode.get("steps", [])):
        pose = step["observation_pose"]
        state = step["agent_state"]
        scan_timestamp = float(step["observation_scan_timestamp"])
        if not math.isfinite(scan_timestamp):
            raise ValueError("decision scan timestamp is not finite")
        decisions.append(
            Decision(
                step_id=int(step.get("step_id", fallback_id)),
                scan_timestamp_sec=scan_timestamp,
                pose_timestamp_sec=float(pose["odom_timestamp"]),
                robot_x=float(pose["x"]),
                robot_y=float(pose["y"]),
                robot_yaw=float(pose["yaw_rad"]),
                agent_state=(int(state[0]), int(state[1])),
            )
        )
    if not decisions:
        raise ValueError("episode contains no decisions")
    return decisions


def match_decision_scans(
    decisions: Sequence[Decision],
    scans: Sequence[RecordedScan],
    tolerance_sec: float = 0.1,
) -> list[DecisionMatch]:
    """Match every JSON scan timestamp to its uniquely nearest bag scan."""
    timestamps = [scan.timestamp_sec for scan in scans]
    matches: list[DecisionMatch] = []
    used_indexes: set[int] = set()
    for decision in decisions:
        insertion = bisect.bisect_left(
            timestamps, decision.scan_timestamp_sec
        )
        candidates = [
            index
            for index in (insertion - 1, insertion, insertion + 1)
            if 0 <= index < len(scans)
        ]
        ranked = sorted(
            (
                abs(scans[index].timestamp_sec - decision.scan_timestamp_sec),
                index,
            )
            for index in candidates
        )
        delta, index = ranked[0]
        if delta > tolerance_sec:
            raise ValueError(
                f"decision {decision.step_id} has no scan within tolerance"
            )
        equally_near = [
            candidate
            for candidate_delta, candidate in ranked
            if math.isclose(
                candidate_delta, delta, rel_tol=0.0, abs_tol=1e-12
            )
        ]
        if len(equally_near) != 1 or index in used_indexes:
            raise ValueError(f"decision {decision.step_id} scan is ambiguous")
        used_indexes.add(index)
        matches.append(DecisionMatch(decision, scans[index], delta))
    return matches


def interpolate_odom_pose(
    odom: Sequence[OdomPose], timestamp_sec: float
) -> PoseAtTime:
    """Interpolate x/y/yaw at a scan timestamp, falling back to nearest."""
    timestamps = [sample.timestamp_sec for sample in odom]
    insertion = bisect.bisect_left(timestamps, timestamp_sec)
    nearest_index = min(
        range(max(0, insertion - 1), min(len(odom), insertion + 1)),
        key=lambda index: abs(timestamps[index] - timestamp_sec),
    )
    nearest_error = abs(timestamps[nearest_index] - timestamp_sec)
    if 0 < insertion < len(odom):
        left = odom[insertion - 1]
        right = odom[insertion]
        span = right.timestamp_sec - left.timestamp_sec
        if span > 0.0:
            fraction = (timestamp_sec - left.timestamp_sec) / span
            return PoseAtTime(
                timestamp_sec=timestamp_sec,
                x=left.x + fraction * (right.x - left.x),
                y=left.y + fraction * (right.y - left.y),
                yaw=interpolate_yaw(left.yaw, right.yaw, fraction),
                method="interpolated",
                nearest_error_sec=nearest_error,
            )
    nearest = odom[nearest_index]
    return PoseAtTime(
        timestamp_sec=timestamp_sec,
        x=nearest.x,
        y=nearest.y,
        yaw=nearest.yaw,
        method="nearest_fallback",
        nearest_error_sec=nearest_error,
    )


def alignment_statistics(poses: Sequence[PoseAtTime]) -> dict[str, Any]:
    """Summarize scan-to-odom bracketing and nearest timing errors."""
    errors = np.asarray(
        [pose.nearest_error_sec for pose in poses], dtype=float
    )
    interpolated = sum(pose.method == "interpolated" for pose in poses)
    return {
        "scan_count": len(poses),
        "interpolated_count": interpolated,
        "nearest_fallback_count": len(poses) - interpolated,
        "interpolation_coverage": (
            float(interpolated) / float(len(poses)) if poses else 0.0
        ),
        "nearest_timing_error_sec": {
            "median": float(np.median(errors)) if len(errors) else None,
            "p95": float(np.percentile(errors, 95)) if len(errors) else None,
            "max": float(np.max(errors)) if len(errors) else None,
        },
    }


def select_causal_burst(
    scans: Sequence[RecordedScan], decision_timestamp_sec: float, count: int
) -> list[RecordedScan]:
    """Select the final N scans at or before one decision timestamp."""
    if count < 1:
        raise ValueError("burst count must be >= 1")
    timestamps = [scan.timestamp_sec for scan in scans]
    end = bisect.bisect_right(timestamps, decision_timestamp_sec)
    selected = list(scans[max(0, end - count):end])
    if not selected or any(
        scan.timestamp_sec > decision_timestamp_sec for scan in selected
    ):
        raise ValueError("causal burst selection leaked a future scan")
    return selected


def aggregate_burst_observations(
    observations: Sequence[ProjectedBeliefObservation],
) -> ProjectedBeliefObservation:
    """Conservatively aggregate a burst into exactly one evidence epoch."""
    if not observations:
        raise ValueError("cannot aggregate an empty burst")
    free_seen: set[tuple[int, int]] = set()
    obstacle_seen: set[tuple[int, int]] = set()
    conflict: set[tuple[int, int]] = set()
    blockers: set[tuple[int, int]] = set()
    suppressed_free: set[tuple[int, int]] = set()
    suppressed_obstacle: set[tuple[int, int]] = set()
    for observation in observations:
        free_seen.update(observation.free_cells)
        obstacle_seen.update(observation.obstacle_cells)
        conflict.update(observation.conflict_cells)
        blockers.update(observation.occlusion_blocker_cells)
        suppressed_free.update(observation.occlusion_suppressed_free_cells)
        suppressed_obstacle.update(
            observation.occlusion_suppressed_obstacle_cells
        )
    conflict.update(free_seen & obstacle_seen)
    free = free_seen - conflict
    obstacle = obstacle_seen - conflict
    return ProjectedBeliefObservation(
        local_snap=np.array(observations[-1].local_snap, copy=True),
        obstacle_cells=frozenset(obstacle),
        free_cells=frozenset(free),
        conflict_cells=frozenset(conflict),
        coarse_occlusion_mode=observations[-1].coarse_occlusion_mode,
        occlusion_blocker_cells=frozenset(blockers),
        occlusion_suppressed_free_cells=frozenset(suppressed_free),
        occlusion_suppressed_obstacle_cells=frozenset(suppressed_obstacle),
    )


def _projection_geometry(
    episode: dict[str, Any], matches: Sequence[DecisionMatch]
) -> dict[str, Any]:
    """Return the frozen projection parameters from the live episode."""
    cell_size = float(episode.get("cell_size", 0.35))
    if not math.isclose(cell_size, 0.35, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("study is pre-registered for cell_size=0.35 only")
    return {
        "origin_x": matches[0].decision.robot_x,
        "origin_y": matches[0].decision.robot_y,
        "origin_state": tuple(
            int(value)
            for value in episode.get("origin_state", CONTINUOUS_ORIGIN_STATE)
        ),
        "cell_size": cell_size,
        "scan_radius_cells": int(episode.get("scan_radius_cells", 10)),
        "laser_x_in_base": float(
            episode.get("laser_x_in_base_m", DEFAULT_LASER_X_IN_BASE_M)
        ),
        "laser_y_in_base": float(
            episode.get("laser_y_in_base_m", DEFAULT_LASER_Y_IN_BASE_M)
        ),
        "laser_yaw_in_base": float(
            episode.get("laser_yaw_in_base", 0.0)
        ),
    }


def _project_kwargs(
    geometry: dict[str, Any],
    pose: Any,
    agent_state: tuple[int, int],
    blockers: Iterable[tuple[int, int]],
    visited: Iterable[tuple[int, int]],
) -> dict[str, Any]:
    """Build one immutable set of production projector arguments."""
    return {
        "robot_x": float(pose.x),
        "robot_y": float(pose.y),
        "robot_yaw": float(pose.yaw),
        **geometry,
        "agent_state": agent_state,
        "coarse_occlusion_mode": "confirmed_opaque",
        "historical_obstacle_cells": blockers,
        "occlusion_exempt_cells": visited,
    }


def project_for_mode(
    mode: str,
    scan: Any,
    *,
    free_end_margin_fraction: float = 0.25,
    **kwargs: Any,
) -> ProjectedBeliefObservation:
    """Dispatch baseline projection to production and E/F to local code."""
    if mode == "production":
        if not math.isclose(
            free_end_margin_fraction, 0.25, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("production mode has a fixed 0.25-cell margin")
        return production_project_scan_to_belief(scan, **kwargs)
    if mode == "supercover":
        return project_scan_supercover(
            scan,
            free_end_margin_fraction=free_end_margin_fraction,
            **kwargs,
        )
    raise ValueError(f"unknown projection mode {mode!r}")


def project_scan_supercover(
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
    coarse_occlusion_mode: str = "off",
    historical_obstacle_cells: Iterable[tuple[int, int]] = (),
    occlusion_exempt_cells: Iterable[tuple[int, int]] = (),
    free_end_margin_fraction: float = 0.25,
) -> ProjectedBeliefObservation:
    """Analysis-only projector changing only FREE traversal sampling."""
    if scan_radius_cells < 1:
        raise ValueError("scan_radius_cells must be >= 1")
    if free_end_margin_fraction < 0.0:
        raise ValueError("free_end margin fraction must be non-negative")
    occlusion_mode = str(coarse_occlusion_mode).strip().lower()
    if occlusion_mode not in ("off", "opaque", "confirmed_opaque"):
        raise ValueError("invalid coarse occlusion mode")
    local_size = 2 * int(scan_radius_cells) + 1
    center = int(scan_radius_cells)
    local_snap = np.full((local_size, local_size), INVISIBLE, dtype=np.int8)
    local_snap[center, center] = EMPTY
    free_cells: set[tuple[int, int]] = set()
    obstacle_cells: set[tuple[int, int]] = set()
    ray_records: list[
        tuple[float, float, Optional[tuple[int, int]], set[tuple[int, int]]]
    ] = []
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

    def visible(cell: tuple[int, int]) -> bool:
        dr = int(cell[0]) - int(agent_state[0])
        dc = int(cell[1]) - int(agent_state[1])
        return (
            dr * dr + dc * dc <= scan_radius_cells * scan_radius_cells
            and 0 <= center + dr < local_size
            and 0 <= center + dc < local_size
        )

    def cell_for(x_value: float, y_value: float) -> tuple[int, int]:
        return continuous_world_to_grid(
            x_value,
            y_value,
            origin_x,
            origin_y,
            origin_state,
            cell_size,
        )

    for index, raw_range in enumerate(scan.ranges):
        hit_range = float(raw_range)
        if not (
            math.isfinite(hit_range)
            and float(scan.range_min) <= hit_range <= float(scan.range_max)
        ):
            continue
        ray_range = min(max(0.0, hit_range), local_radius_m + cell_size)
        free_end = max(
            0.0, ray_range - cell_size * float(free_end_margin_fraction)
        )
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
            candidate = cell_for(
                laser_origin_x + hit_range * ray_cos,
                laser_origin_y + hit_range * ray_sin,
            )
            if visible(candidate):
                hit_cell = candidate
        free_entries = _ordered_coarse_ray_entries(
            laser_origin_x,
            laser_origin_y,
            laser_origin_x + free_end * ray_cos,
            laser_origin_y + free_end * ray_sin,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_state=origin_state,
            cell_size=cell_size,
        )
        ray_free = {
            cell
            for cell, _entry_time in free_entries
            if visible(cell) and cell != hit_cell
        }
        free_cells.update(ray_free)
        if hit_cell is not None:
            obstacle_cells.add(hit_cell)
        ray_records.append((ray_range, world_yaw, hit_cell, ray_free))

    baseline_free = set(free_cells)
    baseline_obstacle = set(obstacle_cells)
    blocker_cells: set[tuple[int, int]] = set()
    suppressed_free: set[tuple[int, int]] = set()
    suppressed_obstacle: set[tuple[int, int]] = set()
    if occlusion_mode in ("opaque", "confirmed_opaque"):
        exemptions = {
            (int(row), int(col)) for row, col in occlusion_exempt_cells
        }
        exemptions.add((int(agent_state[0]), int(agent_state[1])))
        blockers = {
            (int(row), int(col)) for row, col in historical_obstacle_cells
        }
        if occlusion_mode == "opaque":
            blockers.update(obstacle_cells)
        blockers.difference_update(exemptions)
        accepted_free: set[tuple[int, int]] = set()
        accepted_obstacle: set[tuple[int, int]] = set()
        epsilon = 1e-12
        for ray_range, world_yaw, hit_cell, ray_free in ray_records:
            entries = _ordered_coarse_ray_entries(
                laser_origin_x,
                laser_origin_y,
                laser_origin_x + ray_range * math.cos(world_yaw),
                laser_origin_y + ray_range * math.sin(world_yaw),
                origin_x=origin_x,
                origin_y=origin_y,
                origin_state=origin_state,
                cell_size=cell_size,
            )
            visible_entries = tuple(
                (cell, time) for cell, time in entries if visible(cell)
            )
            times = {cell: time for cell, time in visible_entries}
            encountered = {cell for cell in times if cell in blockers}
            blocker_cells.update(encountered)
            first_blocker = min(
                (times[cell] for cell in encountered), default=None
            )

            def evidence_visible(cell: tuple[int, int]) -> bool:
                if first_blocker is None:
                    return True
                entry_time = times.get(cell)
                return bool(
                    entry_time is not None
                    and (
                        entry_time + epsilon < first_blocker
                        or (
                            abs(entry_time - first_blocker) <= epsilon
                            and cell in blockers
                        )
                    )
                )

            accepted_free.update(
                cell for cell in ray_free if evidence_visible(cell)
            )
            if hit_cell is not None and evidence_visible(hit_cell):
                accepted_obstacle.add(hit_cell)
        free_cells = accepted_free
        obstacle_cells = accepted_obstacle
        suppressed_free = baseline_free - free_cells
        suppressed_obstacle = baseline_obstacle - obstacle_cells
    conflict = free_cells & obstacle_cells
    for row, col in sorted(free_cells):
        local_row = center + int(row) - int(agent_state[0])
        local_col = center + int(col) - int(agent_state[1])
        if local_snap[local_row, local_col] == INVISIBLE:
            local_snap[local_row, local_col] = EMPTY
    for row, col in sorted(obstacle_cells):
        local_snap[
            center + int(row) - int(agent_state[0]),
            center + int(col) - int(agent_state[1]),
        ] = OBSTACLE
    return ProjectedBeliefObservation(
        local_snap=local_snap,
        obstacle_cells=frozenset(obstacle_cells),
        free_cells=frozenset(free_cells),
        conflict_cells=frozenset(conflict),
        coarse_occlusion_mode=occlusion_mode,
        occlusion_blocker_cells=frozenset(blocker_cells),
        occlusion_suppressed_free_cells=frozenset(suppressed_free),
        occlusion_suppressed_obstacle_cells=frozenset(
            suppressed_obstacle
        ),
    )


@dataclass
class DecisionModeArtifacts:
    """In-memory products from a 10-decision replay mode."""

    cum_map: Any
    accumulator: BeliefEvidenceAccumulator
    frontier: Any
    metrics: dict[str, Any]
    epoch_observations: list[ProjectedBeliefObservation]
    pre_contexts: list[
        tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]
    ]
    post_contexts: list[
        tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]
    ]


def _pose_for_decision(decision: Decision) -> OdomPose:
    """Expose a JSON decision pose through the common pose interface."""
    return OdomPose(
        timestamp_sec=decision.pose_timestamp_sec,
        x=decision.robot_x,
        y=decision.robot_y,
        yaw=decision.robot_yaw,
    )


def _state_for_pose(
    pose: Any, geometry: dict[str, Any]
) -> tuple[int, int]:
    """Quantize an interpolated odom pose into the frozen policy grid."""
    return continuous_world_to_grid(
        pose.x,
        pose.y,
        geometry["origin_x"],
        geometry["origin_y"],
        geometry["origin_state"],
        geometry["cell_size"],
    )


def _world_value(cum_map: Any, cell: tuple[int, int]) -> int:
    """Read one world cell, treating cells outside map bounds as unknown."""
    array_row = int(cell[0]) - int(cum_map.origin_world_rc[0])
    array_col = int(cell[1]) - int(cum_map.origin_world_rc[1])
    if not (
        0 <= array_row < cum_map.map.shape[0]
        and 0 <= array_col < cum_map.map.shape[1]
    ):
        return INVISIBLE
    return int(cum_map.map[array_row, array_col])


def _classification_name(value: Optional[int]) -> str:
    """Return a stable label for an accumulator classification."""
    if value == EMPTY:
        return "FREE"
    if value == OBSTACLE:
        return "OBSTACLE"
    return "UNCLASSIFIED"


def _map_counts(cum_map: Any) -> dict[str, int]:
    """Count categorical values in a cumulative belief map."""
    belief = np.asarray(cum_map.map)
    free = int(np.count_nonzero(belief == EMPTY))
    obstacle = int(np.count_nonzero(belief == OBSTACLE))
    unknown = int(np.count_nonzero(belief == INVISIBLE))
    return {
        "known": free + obstacle,
        "free": free,
        "obstacle": obstacle,
        "unknown": unknown,
    }


def _unresolved_cells(
    cum_map: Any, accumulator: BeliefEvidenceAccumulator
) -> list[tuple[int, int]]:
    """Return final observed-but-unclassified world cells."""
    return [
        (int(item["row"]), int(item["col"]))
        for item in observed_unclassified_evidence_cells(cum_map, accumulator)
    ]


def _conflict_heavy_counts(
    cum_map: Any, accumulator: BeliefEvidenceAccumulator
) -> dict[str, Any]:
    """List final UNKNOWN cells exceeding fixed conflict thresholds."""
    unresolved = _unresolved_cells(cum_map, accumulator)
    threshold_5 = [
        cell
        for cell in unresolved
        if accumulator.cells[cell].conflict_frame_count >= 5
    ]
    threshold_9 = [
        cell
        for cell in unresolved
        if accumulator.cells[cell].conflict_frame_count >= 9
    ]
    return {
        "conflict_ge_5_count": len(threshold_5),
        "conflict_ge_5_cells": [list(cell) for cell in threshold_5],
        "conflict_ge_9_count": len(threshold_9),
        "conflict_ge_9_cells": [list(cell) for cell in threshold_9],
    }


def _cell_evidence_rows(
    cum_map: Any, accumulator: BeliefEvidenceAccumulator
) -> list[dict[str, Any]]:
    """Emit one deterministic evidence record per final map cell."""
    origin_row, origin_col = (
        int(value) for value in cum_map.origin_world_rc
    )
    rows: list[dict[str, Any]] = []
    for array_row in range(cum_map.map.shape[0]):
        for array_col in range(cum_map.map.shape[1]):
            cell = (origin_row + array_row, origin_col + array_col)
            state = accumulator.cells.get(cell)
            free_count = state.free_frame_count if state else 0
            obstacle_count = state.obstacle_frame_count if state else 0
            conflict_count = state.conflict_frame_count if state else 0
            classification = accumulator.classify(cell) if state else None
            rows.append(
                {
                    "row": cell[0],
                    "col": cell[1],
                    "free_frame_count": int(free_count),
                    "obstacle_frame_count": int(obstacle_count),
                    "conflict_frame_count": int(conflict_count),
                    "classification": _classification_name(classification),
                    "final_categorical_value": int(
                        cum_map.map[array_row, array_col]
                    ),
                }
            )
    return rows


def run_decision_mode(
    mode_name: str,
    burst_size: int,
    matches: Sequence[DecisionMatch],
    all_scans: Sequence[RecordedScan],
    odom: Sequence[OdomPose],
    episode: dict[str, Any],
    cumulative_map_type: Any,
) -> DecisionModeArtifacts:
    """Replay A/B while retaining one evidence epoch per decision."""
    config = named_fusion_config("candidate_a")
    accumulator = BeliefEvidenceAccumulator(config)
    geometry = _projection_geometry(episode, matches)
    compatibility_grid = np.zeros((120, 120), dtype=np.int8)
    cum_map = None
    known_history: list[int] = []
    raw_frontier_history: list[int] = []
    effective_frontier_history: list[int] = []
    observed_history: list[int] = []
    never_history: list[int] = []
    map_expansion: list[dict[str, Any]] = []
    epoch_observations: list[ProjectedBeliefObservation] = []
    pre_contexts: list[
        tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]
    ] = []
    post_contexts: list[
        tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]
    ] = []
    burst_spans: list[dict[str, Any]] = []
    transition_totals = Counter()
    occlusion_totals = Counter()
    for match in matches:
        decision = match.decision
        if cum_map is None:
            blockers: frozenset[tuple[int, int]] = frozenset()
            visited: frozenset[tuple[int, int]] = frozenset()
        else:
            blockers, visited = cumulative_occlusion_cells(cum_map)
        pre_contexts.append((blockers, visited))
        selected = (
            [match.scan]
            if burst_size == 1
            else select_causal_burst(
                all_scans, decision.scan_timestamp_sec, burst_size
            )
        )
        if match.scan not in selected:
            raise ValueError(
                f"burst for decision {decision.step_id} omitted its scan"
            )
        observations: list[ProjectedBeliefObservation] = []
        for scan in selected:
            if scan is match.scan:
                pose: Any = _pose_for_decision(decision)
                state = decision.agent_state
            else:
                pose = interpolate_odom_pose(odom, scan.timestamp_sec)
                state = _state_for_pose(pose, geometry)
            observation = project_for_mode(
                "production",
                scan.message,
                **_project_kwargs(
                    geometry, pose, state, blockers, visited
                ),
            )
            observations.append(observation)
            occlusion_totals["blocker"] += len(
                observation.occlusion_blocker_cells
            )
            occlusion_totals["suppressed_free"] += len(
                observation.occlusion_suppressed_free_cells
            )
            occlusion_totals["suppressed_obstacle"] += len(
                observation.occlusion_suppressed_obstacle_cells
            )
            occlusion_totals["suppressed_unique"] += len(
                observation.occlusion_suppressed_cells
            )
        epoch = (
            observations[0]
            if len(observations) == 1
            else aggregate_burst_observations(observations)
        )
        epoch_observations.append(epoch)
        burst_spans.append(
            {
                "step_id": decision.step_id,
                "scan_count": len(selected),
                "first_scan_timestamp_sec": selected[0].timestamp_sec,
                "last_scan_timestamp_sec": selected[-1].timestamp_sec,
                "decision_timestamp_sec": decision.scan_timestamp_sec,
                "span_sec": (
                    selected[-1].timestamp_sec - selected[0].timestamp_sec
                ),
                "future_scan_count": sum(
                    scan.timestamp_sec > decision.scan_timestamp_sec
                    for scan in selected
                ),
            }
        )
        core_snap = visited_only_local_snap(tuple(epoch.local_snap.shape))
        if cum_map is None:
            cum_map = cumulative_map_type(
                compatibility_grid, decision.agent_state, core_snap
            )
        else:
            cum_map.update(decision.agent_state, core_snap)
        step_stats = apply_evidence_fusion(cum_map, accumulator, epoch)
        transition_totals["free_to_obstacle"] += (
            step_stats.free_to_obstacle_transitions_this_step
        )
        transition_totals["obstacle_to_free"] += (
            step_stats.obstacle_to_free_transitions_this_step
        )
        transition_totals["promotion"] += (
            step_stats.invisible_to_obstacle_transitions_this_step
            + step_stats.free_to_obstacle_transitions_this_step
        )
        transition_totals["evidence_free"] += (
            step_stats.evidence_free_cells_this_step
        )
        transition_totals["evidence_obstacle"] += (
            step_stats.evidence_obstacle_cells_this_step
        )
        transition_totals["evidence_conflict"] += (
            step_stats.evidence_conflict_cells_this_step
        )
        frontier = frontier_semantics_snapshot(
            cum_map, accumulator, "evidence_aware"
        )
        counts = _map_counts(cum_map)
        known_history.append(counts["known"])
        raw_frontier_history.append(frontier.raw_frontier_count)
        effective_frontier_history.append(frontier.effective_frontier_count)
        observed_history.append(
            frontier.observed_unclassified_unknown_count
        )
        never_history.append(frontier.never_observed_unknown_count)
        map_expansion.append(
            {
                "step_id": decision.step_id,
                "shape": [int(value) for value in cum_map.map.shape],
                "origin_world_rc": [
                    int(value) for value in cum_map.origin_world_rc
                ],
                "total_cells": int(cum_map.map.size),
                **counts,
                "observed_unclassified": (
                    frontier.observed_unclassified_unknown_count
                ),
                "never_observed": frontier.never_observed_unknown_count,
            }
        )
        post_contexts.append(cumulative_occlusion_cells(cum_map))
    if cum_map is None:
        raise AssertionError("decision mode produced no map")
    trajectory = episode_trajectory_states(episode["steps"])
    final_correction = record_traversed_cells_as_free(cum_map, trajectory)
    transition_totals["obstacle_to_free"] += (
        final_correction.corrected_from_obstacle
    )
    frontier = frontier_semantics_snapshot(
        cum_map, accumulator, "evidence_aware"
    )
    metrics = {
        "mode": mode_name,
        "burst_size": burst_size,
        "evidence_epoch_count": len(matches),
        "raw_scan_count": sum(item["scan_count"] for item in burst_spans),
        **_map_counts(cum_map),
        "observed_unclassified": (
            frontier.observed_unclassified_unknown_count
        ),
        "never_observed": frontier.never_observed_unknown_count,
        "raw_frontier": frontier.raw_frontier_count,
        "effective_frontier": frontier.effective_frontier_count,
        "known_history": known_history,
        "raw_frontier_history": raw_frontier_history,
        "effective_frontier_history": effective_frontier_history,
        "observed_unclassified_history": observed_history,
        "never_observed_history": never_history,
        "evidence_totals": {
            "free": int(transition_totals["evidence_free"]),
            "obstacle": int(transition_totals["evidence_obstacle"]),
            "conflict": int(transition_totals["evidence_conflict"]),
        },
        "transitions": {
            "free_to_obstacle": int(
                transition_totals["free_to_obstacle"]
            ),
            "obstacle_to_free": int(
                transition_totals["obstacle_to_free"]
            ),
            "promotion_count": int(transition_totals["promotion"]),
        },
        "occlusion_totals_across_raw_scans": {
            key: int(occlusion_totals[key])
            for key in (
                "blocker",
                "suppressed_free",
                "suppressed_obstacle",
                "suppressed_unique",
            )
        },
        "conflict_heavy_final_unknown": _conflict_heavy_counts(
            cum_map, accumulator
        ),
        "origin_world_rc": [int(value) for value in cum_map.origin_world_rc],
        "shape": [int(value) for value in cum_map.map.shape],
        "map_expansion_history": map_expansion,
        "burst_spans": burst_spans,
        "cell_evidence": _cell_evidence_rows(cum_map, accumulator),
    }
    return DecisionModeArtifacts(
        cum_map=cum_map,
        accumulator=accumulator,
        frontier=frontier,
        metrics=metrics,
        epoch_observations=epoch_observations,
        pre_contexts=pre_contexts,
        post_contexts=post_contexts,
    )


def _context_for_timestamp(
    timestamp_sec: float,
    matches: Sequence[DecisionMatch],
    baseline: DecisionModeArtifacts,
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    """Use blockers created strictly before the queried scan time."""
    decision_times = [match.decision.scan_timestamp_sec for match in matches]
    completed_index = bisect.bisect_left(decision_times, timestamp_sec) - 1
    if completed_index < 0:
        return frozenset(), frozenset()
    return baseline.post_contexts[completed_index]


def _sha256_file(path: Path) -> str:
    """Hash a file without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    """Convert NumPy scalar values without relaxing JSON validity."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def verify_internal_manifest(dataset_root: Path) -> dict[str, Any]:
    """Verify every entry in the extracted SHA256SUMS manifest."""
    manifest = dataset_root / "SHA256SUMS.txt"
    records: list[dict[str, Any]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        target = dataset_root / relative
        actual = _sha256_file(target) if target.is_file() else None
        records.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "pass": actual == expected,
            }
        )
    return {
        "status": "PASS" if all(item["pass"] for item in records) else "FAIL",
        "manifest_path": str(manifest),
        "file_count": len(records),
        "files": records,
    }


def summarize_rosbag(contents: BagContents) -> dict[str, Any]:
    """Report bag rates, LaserScan geometry, and valid beam statistics."""
    first = contents.scans[0].message
    valid_counts = []
    geometry_variants: set[tuple[Any, ...]] = set()
    for scan in contents.scans:
        message = scan.message
        valid_counts.append(
            sum(
                math.isfinite(float(value))
                and float(message.range_min)
                <= float(value)
                <= float(message.range_max)
                for value in message.ranges
            )
        )
        geometry_variants.add(
            (
                len(message.ranges),
                float(message.angle_min),
                float(message.angle_max),
                float(message.angle_increment),
                float(message.range_min),
                float(message.range_max),
            )
        )
    duration = contents.metadata_duration_sec
    return {
        "duration_sec": duration,
        "starting_time_sec": contents.bag_start_sec,
        "ending_time_sec": contents.bag_end_sec,
        "topic_counts": contents.topic_counts,
        "scan_rate_hz": contents.topic_counts.get("/scan", 0) / duration,
        "odom_rate_hz": contents.topic_counts.get("/odom", 0) / duration,
        "laser_scan_geometry": {
            "number_of_ranges": len(first.ranges),
            "angle_min": float(first.angle_min),
            "angle_max": float(first.angle_max),
            "angle_increment": float(first.angle_increment),
            "range_min": float(first.range_min),
            "range_max": float(first.range_max),
            "geometry_variant_count": len(geometry_variants),
        },
        "valid_beam_count": {
            "min": int(min(valid_counts)),
            "median": float(statistics.median(valid_counts)),
            "mean": float(statistics.mean(valid_counts)),
            "max": int(max(valid_counts)),
        },
    }


@dataclass(frozen=True)
class ScanProjectionRecord:
    """One all-scan projection and its time-local occlusion context."""

    scan: RecordedScan
    pose: PoseAtTime
    agent_state: tuple[int, int]
    blockers: frozenset[tuple[int, int]]
    visited: frozenset[tuple[int, int]]
    production_observation: ProjectedBeliefObservation


def _evidence_partition(
    observation: ProjectedBeliefObservation,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    """Partition accepted evidence into free-only, obstacle-only, conflict."""
    free = set(observation.free_cells)
    obstacle = set(observation.obstacle_cells)
    conflict = set(observation.conflict_cells) | (free & obstacle)
    return free - conflict, obstacle - conflict, conflict


def run_allscan_density(
    episode_scans: Sequence[RecordedScan],
    odom: Sequence[OdomPose],
    matches: Sequence[DecisionMatch],
    baseline: DecisionModeArtifacts,
    geometry: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[ScanProjectionRecord],
    dict[tuple[int, int], Counter],
]:
    """Run Study C without applying categorical fusion."""
    counts: dict[tuple[int, int], Counter] = defaultdict(Counter)
    records: list[ScanProjectionRecord] = []
    skipped_no_valid_beams = 0
    for scan in episode_scans:
        valid_beams = sum(
            math.isfinite(float(value))
            and float(scan.message.range_min)
            <= float(value)
            <= float(scan.message.range_max)
            for value in scan.message.ranges
        )
        if valid_beams == 0:
            skipped_no_valid_beams += 1
            continue
        pose = interpolate_odom_pose(odom, scan.timestamp_sec)
        state = _state_for_pose(pose, geometry)
        blockers, visited = _context_for_timestamp(
            scan.timestamp_sec, matches, baseline
        )
        observation = project_for_mode(
            "production",
            scan.message,
            **_project_kwargs(
                geometry, pose, state, blockers, visited
            ),
        )
        free, obstacle, conflict = _evidence_partition(observation)
        for cell in free:
            counts[cell]["free_only"] += 1
        for cell in obstacle:
            counts[cell]["obstacle_only"] += 1
        for cell in conflict:
            counts[cell]["conflict"] += 1
        for cell in free | obstacle | conflict:
            counts[cell]["observed"] += 1
        records.append(
            ScanProjectionRecord(
                scan=scan,
                pose=pose,
                agent_state=state,
                blockers=blockers,
                visited=visited,
                production_observation=observation,
            )
        )
    baseline_unresolved = _unresolved_cells(
        baseline.cum_map, baseline.accumulator
    )
    unresolved_table: list[dict[str, Any]] = []
    for cell in baseline_unresolved:
        decision_state = baseline.accumulator.cells[cell]
        allscan = counts[cell]
        observed = int(allscan["observed"])
        unresolved_table.append(
            {
                "row": cell[0],
                "col": cell[1],
                "baseline_final_value": _world_value(
                    baseline.cum_map, cell
                ),
                "decision_free_count": decision_state.free_frame_count,
                "decision_obstacle_count": (
                    decision_state.obstacle_frame_count
                ),
                "decision_conflict_count": (
                    decision_state.conflict_frame_count
                ),
                "allscan_observed": observed,
                "allscan_free_only": int(allscan["free_only"]),
                "allscan_obstacle_only": int(allscan["obstacle_only"]),
                "allscan_conflict": int(allscan["conflict"]),
                "allscan_no_evidence": len(records) - observed,
                "conflict_persistence": (
                    float(allscan["conflict"]) / float(observed)
                    if observed
                    else None
                ),
                "free_only_fraction": (
                    float(allscan["free_only"]) / float(observed)
                    if observed
                    else None
                ),
                "obstacle_only_fraction": (
                    float(allscan["obstacle_only"]) / float(observed)
                    if observed
                    else None
                ),
            }
        )
    high_conflict = [
        item
        for item in unresolved_table
        if item["decision_conflict_count"] >= 9
    ]
    persistences = [
        item["conflict_persistence"]
        for item in unresolved_table
        if item["conflict_persistence"] is not None
    ]
    high_persistences = [
        item["conflict_persistence"]
        for item in high_conflict
        if item["conflict_persistence"] is not None
    ]
    report = {
        "episode_start_sec": episode_scans[0].timestamp_sec,
        "episode_end_sec": episode_scans[-1].timestamp_sec,
        "episode_duration_sec": (
            episode_scans[-1].timestamp_sec - episode_scans[0].timestamp_sec
        ),
        "candidate_scan_count": len(episode_scans),
        "total_scans_processed": len(records),
        "skipped_no_valid_beams": skipped_no_valid_beams,
        "total_observed_cells": len(counts),
        "scan_to_odom_alignment": alignment_statistics(
            [record.pose for record in records]
        ),
        "baseline_unresolved_count": len(unresolved_table),
        "baseline_unresolved_cells": unresolved_table,
        "baseline_conflict_ge_9_cells": high_conflict,
        "unresolved_conflict_persistence_summary": {
            "min": min(persistences) if persistences else None,
            "median": (
                float(statistics.median(persistences))
                if persistences
                else None
            ),
            "mean": (
                float(statistics.mean(persistences))
                if persistences
                else None
            ),
            "max": max(persistences) if persistences else None,
        },
        "conflict_ge_9_persistence_summary": {
            "cell_count": len(high_persistences),
            "min": min(high_persistences) if high_persistences else None,
            "median": (
                float(statistics.median(high_persistences))
                if high_persistences
                else None
            ),
            "mean": (
                float(statistics.mean(high_persistences))
                if high_persistences
                else None
            ),
            "max": max(high_persistences) if high_persistences else None,
        },
    }
    return report, records, counts


def select_fixed_frequency_scans(
    scans: Sequence[RecordedScan], frequency_hz: float
) -> list[RecordedScan]:
    """Causally downsample a scan stream at a pre-registered frequency."""
    if frequency_hz <= 0.0 or not scans:
        raise ValueError("frequency must be positive and scans non-empty")
    period = 1.0 / float(frequency_hz)
    next_target = scans[0].timestamp_sec
    selected: list[RecordedScan] = []
    for scan in scans:
        if scan.timestamp_sec + 1e-12 < next_target:
            continue
        selected.append(scan)
        while next_target <= scan.timestamp_sec + 1e-12:
            next_target += period
    return selected


def run_temporal_mode(
    label: str,
    frequency_hz: float,
    scans: Sequence[RecordedScan],
    odom: Sequence[OdomPose],
    geometry: dict[str, Any],
    cumulative_map_type: Any,
) -> dict[str, Any]:
    """Run one diagnostic D mode with each scan as an evidence epoch."""
    selected = select_fixed_frequency_scans(scans, frequency_hz)
    accumulator = BeliefEvidenceAccumulator(named_fusion_config("candidate_a"))
    compatibility_grid = np.zeros((120, 120), dtype=np.int8)
    cum_map = None
    transition_totals = Counter()
    evidence_totals = Counter()
    selected_states: list[tuple[int, int]] = []
    pose_records: list[PoseAtTime] = []
    for scan in selected:
        pose = interpolate_odom_pose(odom, scan.timestamp_sec)
        pose_records.append(pose)
        state = _state_for_pose(pose, geometry)
        selected_states.append(state)
        if cum_map is None:
            blockers: frozenset[tuple[int, int]] = frozenset()
            visited: frozenset[tuple[int, int]] = frozenset()
        else:
            blockers, visited = cumulative_occlusion_cells(cum_map)
        observation = project_for_mode(
            "production",
            scan.message,
            **_project_kwargs(
                geometry, pose, state, blockers, visited
            ),
        )
        core_snap = visited_only_local_snap(
            tuple(observation.local_snap.shape)
        )
        if cum_map is None:
            cum_map = cumulative_map_type(
                compatibility_grid, state, core_snap
            )
        else:
            cum_map.update(state, core_snap)
        stats = apply_evidence_fusion(cum_map, accumulator, observation)
        transition_totals["free_to_obstacle"] += (
            stats.free_to_obstacle_transitions_this_step
        )
        transition_totals["obstacle_to_free"] += (
            stats.obstacle_to_free_transitions_this_step
        )
        transition_totals["promotion"] += (
            stats.invisible_to_obstacle_transitions_this_step
            + stats.free_to_obstacle_transitions_this_step
        )
        evidence_totals["free"] += stats.evidence_free_cells_this_step
        evidence_totals["obstacle"] += (
            stats.evidence_obstacle_cells_this_step
        )
        evidence_totals["conflict"] += (
            stats.evidence_conflict_cells_this_step
        )
    if cum_map is None:
        raise AssertionError("temporal mode produced no map")
    correction = record_traversed_cells_as_free(cum_map, selected_states)
    transition_totals["obstacle_to_free"] += correction.corrected_from_obstacle
    frontier = frontier_semantics_snapshot(
        cum_map, accumulator, "evidence_aware"
    )
    duration = selected[-1].timestamp_sec - selected[0].timestamp_sec
    return {
        "mode": label,
        "requested_frequency_hz": frequency_hz,
        "diagnostic_only": True,
        "frame_time_semantics_changed": True,
        "selected_evidence_epoch_count": len(selected),
        "duration_sec": duration,
        "evidence_epochs_per_second": (
            float(len(selected)) / duration if duration > 0.0 else None
        ),
        **_map_counts(cum_map),
        "observed_unclassified": (
            frontier.observed_unclassified_unknown_count
        ),
        "never_observed": frontier.never_observed_unknown_count,
        "raw_frontier": frontier.raw_frontier_count,
        "effective_frontier": frontier.effective_frontier_count,
        "transitions": {
            "free_to_obstacle": int(
                transition_totals["free_to_obstacle"]
            ),
            "obstacle_to_free": int(
                transition_totals["obstacle_to_free"]
            ),
            "promotion_count": int(transition_totals["promotion"]),
        },
        "evidence_totals": {
            key: int(evidence_totals[key])
            for key in ("free", "obstacle", "conflict")
        },
        "conflict_heavy_final_unknown": _conflict_heavy_counts(
            cum_map, accumulator
        ),
        "scan_to_odom_alignment": alignment_statistics(pose_records),
    }


@dataclass
class ProjectionInput:
    """The frozen inputs for one E/F projector comparison."""

    scan: RecordedScan
    pose: Any
    agent_state: tuple[int, int]
    blockers: frozenset[tuple[int, int]]
    visited: frozenset[tuple[int, int]]
    reference: ProjectedBeliefObservation


@dataclass
class RayComparisonArtifacts:
    """Detailed results from one ray traversal comparison."""

    report: dict[str, Any]
    counter_observations: list[ProjectedBeliefObservation]
    extra_counts: Counter


def compare_ray_projection(
    inputs: Sequence[ProjectionInput],
    geometry: dict[str, Any],
    baseline: DecisionModeArtifacts,
    free_end_margin_fraction: float,
    reference_name: str,
    comparison_name: str,
) -> RayComparisonArtifacts:
    """Compare accepted FREE cells while holding endpoint semantics fixed."""
    current_union: set[tuple[int, int]] = set()
    counter_union: set[tuple[int, int]] = set()
    extra_union: set[tuple[int, int]] = set()
    current_only_union: set[tuple[int, int]] = set()
    extra_counts: Counter = Counter()
    endpoint_adjacent_extra: set[tuple[int, int]] = set()
    current_total = 0
    counter_total = 0
    extra_total = 0
    endpoint_mismatch_count = 0
    counter_observations: list[ProjectedBeliefObservation] = []
    for item in inputs:
        counter = project_for_mode(
            "supercover",
            item.scan.message,
            free_end_margin_fraction=free_end_margin_fraction,
            **_project_kwargs(
                geometry,
                item.pose,
                item.agent_state,
                item.blockers,
                item.visited,
            ),
        )
        counter_observations.append(counter)
        if counter.obstacle_cells != item.reference.obstacle_cells:
            endpoint_mismatch_count += 1
        current_free = set(item.reference.free_cells)
        counter_free = set(counter.free_cells)
        extra = counter_free - current_free
        current_only = current_free - counter_free
        current_union.update(current_free)
        counter_union.update(counter_free)
        extra_union.update(extra)
        current_only_union.update(current_only)
        current_total += len(current_free)
        counter_total += len(counter_free)
        extra_total += len(extra)
        extra_counts.update(extra)
        for cell in extra:
            if any(
                max(abs(cell[0] - hit[0]), abs(cell[1] - hit[1])) <= 1
                for hit in counter.obstacle_cells
            ):
                endpoint_adjacent_extra.add(cell)
    if endpoint_mismatch_count:
        raise AssertionError(
            "analysis ray projector changed obstacle endpoint classification"
        )
    baseline_unknown = {
        (
            int(baseline.cum_map.origin_world_rc[0]) + array_row,
            int(baseline.cum_map.origin_world_rc[1]) + array_col,
        )
        for array_row, array_col in zip(
            *np.nonzero(np.asarray(baseline.cum_map.map) == INVISIBLE)
        )
    }
    baseline_unresolved = set(
        _unresolved_cells(baseline.cum_map, baseline.accumulator)
    )
    production_counts: Counter = Counter()
    counter_counts: Counter = Counter()
    for item, counter in zip(inputs, counter_observations):
        production_counts.update(item.reference.free_cells)
        counter_counts.update(counter.free_cells)
    explained = sorted(
        cell
        for cell in baseline_unknown
        if production_counts[cell] == 0 and counter_counts[cell] > 0
    )
    explained_unresolved = [
        cell for cell in explained if cell in baseline_unresolved
    ]
    explained_never = [
        cell for cell in explained if cell not in baseline_unresolved
    ]
    missing_current_union = counter_union - current_union
    missing_comparison_union = current_union - counter_union
    report = {
        "reference": reference_name,
        "comparison": comparison_name,
        "scan_count": len(inputs),
        "free_end_margin_fraction": free_end_margin_fraction,
        "free_end_margin_m": (
            free_end_margin_fraction * geometry["cell_size"]
        ),
        "current_free_cell_observations": current_total,
        "comparison_free_cell_observations": counter_total,
        "extra_free_cell_observations": extra_total,
        "current_unique_free_cells": len(current_union),
        "comparison_unique_free_cells": len(counter_union),
        "extra_unique_free_cells": len(missing_current_union),
        "extra_unique_fraction_of_comparison": (
            float(len(missing_current_union)) / float(len(counter_union))
            if counter_union
            else 0.0
        ),
        "extra_cells": [list(cell) for cell in sorted(missing_current_union)],
        "per_scan_extra_unique_free_cells": len(extra_union),
        "per_scan_extra_cells": [list(cell) for cell in sorted(extra_union)],
        "current_only_unique_free_cells": len(missing_comparison_union),
        "current_only_cells": [
            list(cell) for cell in sorted(missing_comparison_union)
        ],
        "per_scan_current_only_unique_free_cells": len(current_only_union),
        "endpoint_classification_mismatch_count": endpoint_mismatch_count,
        "endpoint_adjacent_extra_unique_count": len(endpoint_adjacent_extra),
        "endpoint_adjacent_extra_cells": [
            list(cell) for cell in sorted(endpoint_adjacent_extra)
        ],
        "baseline_unknown_receiving_comparison_only_free": {
            "count": len(explained),
            "cells": [list(cell) for cell in explained],
            "observed_unclassified_count": len(explained_unresolved),
            "observed_unclassified_cells": [
                list(cell) for cell in explained_unresolved
            ],
            "never_observed_count": len(explained_never),
            "never_observed_cells": [list(cell) for cell in explained_never],
        },
    }
    return RayComparisonArtifacts(report, counter_observations, extra_counts)


def baseline_cell_transitions(
    baseline: DecisionModeArtifacts,
    comparison: DecisionModeArtifacts,
) -> dict[str, Any]:
    """Classify the baseline unresolved cells under a B mode."""
    cells = _unresolved_cells(baseline.cum_map, baseline.accumulator)
    free = [
        cell
        for cell in cells
        if _world_value(comparison.cum_map, cell) == EMPTY
    ]
    obstacle = [
        cell
        for cell in cells
        if _world_value(comparison.cum_map, cell) == OBSTACLE
    ]
    unknown = [
        cell
        for cell in cells
        if _world_value(comparison.cum_map, cell) == INVISIBLE
    ]
    mainly_conflict = []
    for cell in unknown:
        state = comparison.accumulator.cells.get(cell)
        if state is not None and state.conflict_frame_count >= max(
            state.free_frame_count, state.obstacle_frame_count
        ):
            mainly_conflict.append(cell)
    return {
        "baseline_unresolved_count": len(cells),
        "comparison_final_observed_unclassified": (
            comparison.metrics["observed_unclassified"]
        ),
        "net_observed_unclassified_change": (
            comparison.metrics["observed_unclassified"] - len(cells)
        ),
        "classified_free_count": len(free),
        "classified_free_cells": [list(cell) for cell in free],
        "classified_obstacle_count": len(obstacle),
        "classified_obstacle_cells": [list(cell) for cell in obstacle],
        "still_unknown_count": len(unknown),
        "still_unknown_cells": [list(cell) for cell in unknown],
        "still_mainly_conflict_count": len(mainly_conflict),
        "still_mainly_conflict_cells": [
            list(cell) for cell in mainly_conflict
        ],
    }


def _invert_transform(transform: Transform2D) -> Transform2D:
    """Invert one rigid 2-D transform."""
    cosine = math.cos(transform.yaw)
    sine = math.sin(transform.yaw)
    inverse_x = -(cosine * transform.x + sine * transform.y)
    inverse_y = -(-sine * transform.x + cosine * transform.y)
    return Transform2D(
        timestamp_sec=transform.timestamp_sec,
        parent_frame=transform.child_frame,
        child_frame=transform.parent_frame,
        x=inverse_x,
        y=inverse_y,
        yaw=_normalize_angle(-transform.yaw),
    )


def _find_map_from_odom_transform(
    contents: BagContents, map_frame: str
) -> Optional[Transform2D]:
    """Find a direct, latest map<-odom transform without graph guessing."""
    map_frame = map_frame.lstrip("/")
    direct = [
        transform
        for transform in contents.transforms
        if transform.parent_frame == map_frame
        and transform.child_frame in {"odom", "odom_combined"}
    ]
    inverse = [
        transform
        for transform in contents.transforms
        if transform.child_frame == map_frame
        and transform.parent_frame in {"odom", "odom_combined"}
    ]
    candidates = direct + [_invert_transform(item) for item in inverse]
    if not candidates:
        return None
    if contents.final_map_bag_timestamp_sec is not None:
        bounded = [
            item
            for item in candidates
            if item.timestamp_sec <= contents.final_map_bag_timestamp_sec
        ]
        if bounded:
            candidates = bounded
    return max(candidates, key=lambda item: item.timestamp_sec)


def evaluate_slam_composition(
    contents: BagContents,
    baseline: DecisionModeArtifacts,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate baseline unresolved cell footprints on the final SLAM map."""
    if contents.final_map is None:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "bag contains no /map OccupancyGrid",
        }
    occupancy_grid = contents.final_map
    map_frame = str(occupancy_grid.header.frame_id).lstrip("/")
    transform = _find_map_from_odom_transform(contents, map_frame)
    if transform is None:
        return {
            "status": "NOT_AVAILABLE",
            "reason": (
                "no direct timestamped map<->odom or "
                "map<->odom_combined transform was recorded"
            ),
        }
    info = occupancy_grid.info
    width = int(info.width)
    height = int(info.height)
    resolution = float(info.resolution)
    if width < 1 or height < 1 or resolution <= 0.0:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "final /map metadata is invalid",
        }
    occupancy = np.asarray(occupancy_grid.data, dtype=np.int16).reshape(
        height, width
    )
    origin_yaw = _yaw_from_quaternion(info.origin.orientation)
    grid_rows, grid_cols = np.indices((height, width), dtype=float)
    local_x = (grid_cols + 0.5) * resolution
    local_y = (grid_rows + 0.5) * resolution
    origin_cos = math.cos(origin_yaw)
    origin_sin = math.sin(origin_yaw)
    map_x = (
        float(info.origin.position.x)
        + origin_cos * local_x
        - origin_sin * local_y
    )
    map_y = (
        float(info.origin.position.y)
        + origin_sin * local_x
        + origin_cos * local_y
    )
    delta_x = map_x - transform.x
    delta_y = map_y - transform.y
    transform_cos = math.cos(transform.yaw)
    transform_sin = math.sin(transform.yaw)
    odom_x = transform_cos * delta_x + transform_sin * delta_y
    odom_y = -transform_sin * delta_x + transform_cos * delta_y
    half_cell = geometry["cell_size"] * 0.5
    cell_rows = []
    for cell in _unresolved_cells(baseline.cum_map, baseline.accumulator):
        center_x = geometry["origin_x"] + (
            cell[1] - geometry["origin_state"][1]
        ) * geometry["cell_size"]
        center_y = geometry["origin_y"] - (
            cell[0] - geometry["origin_state"][0]
        ) * geometry["cell_size"]
        inside = (
            (np.abs(odom_x - center_x) <= half_cell + 1e-12)
            & (np.abs(odom_y - center_y) <= half_cell + 1e-12)
        )
        values = occupancy[inside]
        total = int(values.size)
        unknown = int(np.count_nonzero(values < 0))
        occupied = int(np.count_nonzero(values >= 50))
        free = total - unknown - occupied
        decision_state = baseline.accumulator.cells[cell]
        cell_rows.append(
            {
                "row": cell[0],
                "col": cell[1],
                "decision_conflict_count": (
                    decision_state.conflict_frame_count
                ),
                "slam_pixel_count": total,
                "slam_free_pixel_fraction": (
                    float(free) / total if total else None
                ),
                "slam_occupied_pixel_fraction": (
                    float(occupied) / total if total else None
                ),
                "slam_unknown_pixel_fraction": (
                    float(unknown) / total if total else None
                ),
            }
        )
    return {
        "status": "AVAILABLE",
        "evaluation_only": True,
        "map_never_entered_policy_or_belief": True,
        "map_frame": map_frame,
        "odom_frame": transform.child_frame,
        "map_from_odom_transform": {
            "timestamp_sec": transform.timestamp_sec,
            "x": transform.x,
            "y": transform.y,
            "yaw": transform.yaw,
        },
        "final_map": {
            "width": width,
            "height": height,
            "resolution": resolution,
            "bag_timestamp_sec": contents.final_map_bag_timestamp_sec,
        },
        "occupancy_threshold": 50,
        "cells": cell_rows,
    }


def analyze_map_expansion(
    baseline: DecisionModeArtifacts,
) -> dict[str, Any]:
    """Verify that the step-5-to-6 never-observed jump is map growth."""
    history = baseline.metrics["map_expansion_history"]
    before = history[5]
    after = history[6]
    total_delta = after["total_cells"] - before["total_cells"]
    known_delta = after["known"] - before["known"]
    observed_delta = (
        after["observed_unclassified"] - before["observed_unclassified"]
    )
    never_delta = after["never_observed"] - before["never_observed"]
    expected_never_delta = total_delta - known_delta - observed_delta
    shape_or_origin_changed = (
        before["shape"] != after["shape"]
        or before["origin_world_rc"] != after["origin_world_rc"]
    )
    explains = (
        shape_or_origin_changed and never_delta == expected_never_delta
    )
    return {
        "steps": history,
        "step_5_to_6": {
            "total_cell_delta": total_delta,
            "known_delta": known_delta,
            "observed_unclassified_delta": observed_delta,
            "never_observed_delta": never_delta,
            "expected_never_observed_delta_from_bookkeeping": (
                expected_never_delta
            ),
            "shape_or_origin_changed": shape_or_origin_changed,
        },
        "map_expansion_explains_never_observed_jump": (
            "YES" if explains else "NO"
        ),
        "map_forgetting_detected": False if explains else None,
    }


def _effect_rating(proportion: float) -> str:
    """Map a pre-declared explained-cell proportion to an effect label."""
    if proportion >= 0.5:
        return "HIGH"
    if proportion >= 0.2:
        return "MODERATE"
    return "LOW"


def build_interpretation(
    baseline: DecisionModeArtifacts,
    b3_transitions: dict[str, Any],
    b5_transitions: dict[str, Any],
    study_c: dict[str, Any],
    study_e: dict[str, Any],
    study_f: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Produce quantitative, rule-based conclusions and direct answers."""
    unresolved_count = len(
        _unresolved_cells(baseline.cum_map, baseline.accumulator)
    )
    b5_final_unresolved = b5_transitions[
        "comparison_final_observed_unclassified"
    ]
    temporal_net_reduction = max(0, unresolved_count - b5_final_unresolved)
    temporal_fraction = (
        temporal_net_reduction / unresolved_count if unresolved_count else 0.0
    )
    temporal_rating = _effect_rating(temporal_fraction)
    high_conflict_summary = study_c["conflict_ge_9_persistence_summary"]
    high_conflict_median = high_conflict_summary["median"]
    if high_conflict_median is None:
        mixed_rating = "LOW"
    elif high_conflict_median >= 0.5:
        mixed_rating = "HIGH"
    elif high_conflict_median >= 0.25:
        mixed_rating = "MODERATE"
    else:
        mixed_rating = "LOW"
    ray_explained = study_e["all_episode_scans"][
        "baseline_unknown_receiving_comparison_only_free"
    ]["observed_unclassified_count"]
    ray_fraction = (
        ray_explained / unresolved_count if unresolved_count else 0.0
    )
    ray_rating = _effect_rating(ray_fraction)
    ratings = {
        "TEMPORAL_UNDERSAMPLING_EFFECT": temporal_rating,
        "COARSE_CELL_MIXED_OCCUPANCY_EFFECT": mixed_rating,
        "RAY_DISCRETIZATION_EFFECT": ray_rating,
    }
    rank = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
    ordered = sorted(
        ratings,
        key=lambda key: (
            rank[ratings[key]],
            {
                "COARSE_CELL_MIXED_OCCUPANCY_EFFECT": 2,
                "TEMPORAL_UNDERSAMPLING_EFFECT": 1,
                "RAY_DISCRETIZATION_EFFECT": 0,
            }[key],
        ),
        reverse=True,
    )
    margin_cells = None
    if study_f is not None:
        margin_cells = study_f["all_episode_scans"][
            "baseline_unknown_receiving_comparison_only_free"
        ]["observed_unclassified_count"]
    q5 = (
        "NOT_ISOLATED"
        if study_f is None
        else (
            "YES"
            if margin_cells is not None
            and margin_cells >= max(1, math.ceil(unresolved_count * 0.5))
            else "NO"
        )
    )
    return {
        **ratings,
        "rating_rules": {
            "temporal_and_ray": (
                "HIGH >=50%, MODERATE >=20%, LOW <20% of the baseline "
                "observed-unclassified count reduced (temporal) or cells "
                "explained (ray)"
            ),
            "mixed_occupancy": (
                "HIGH median all-scan conflict persistence >=0.50, "
                "MODERATE >=0.25, LOW <0.25 for baseline conflict>=9 cells"
            ),
        },
        "evidence": {
            "temporal": {
                "baseline_unresolved": unresolved_count,
                "b3_classified": (
                    unresolved_count - b3_transitions["still_unknown_count"]
                ),
                "b5_classified": (
                    unresolved_count - b5_transitions["still_unknown_count"]
                ),
                "b3_final_observed_unclassified": b3_transitions[
                    "comparison_final_observed_unclassified"
                ],
                "b5_final_observed_unclassified": b5_final_unresolved,
                "b5_net_reduction": temporal_net_reduction,
                "b5_net_reduction_fraction": temporal_fraction,
            },
            "mixed_occupancy": high_conflict_summary,
            "ray_discretization": {
                "baseline_unresolved_receiving_supercover_only_free": (
                    ray_explained
                ),
                "fraction": ray_fraction,
            },
            "endpoint_margin": {
                "baseline_unresolved_receiving_margin_only_free": margin_cells
            },
        },
        "PRIMARY_CAUSE": ordered[0].replace("_EFFECT", ""),
        "SECONDARY_CAUSE": ordered[1].replace("_EFFECT", ""),
        "answers": {
            "Q1_one_scan_per_policy_decision": {
                "answer": "YES",
                "evidence": (
                    "production apply_evidence_fusion calls "
                    "accumulator.observe once for its single "
                    "ProjectedBeliefObservation; exact replay "
                    "used 10 scans for 10 decisions"
                ),
            },
            "Q2_burst_reduction": {
                "baseline_observed_unclassified": unresolved_count,
                "burst3_original_cells_classified": (
                    unresolved_count - b3_transitions["still_unknown_count"]
                ),
                "burst3_final_observed_unclassified": b3_transitions[
                    "comparison_final_observed_unclassified"
                ],
                "burst3_net_change": b3_transitions[
                    "net_observed_unclassified_change"
                ],
                "burst5_original_cells_classified": (
                    unresolved_count - b5_transitions["still_unknown_count"]
                ),
                "burst5_final_observed_unclassified": b5_final_unresolved,
                "burst5_net_change": b5_transitions[
                    "net_observed_unclassified_change"
                ],
            },
            "Q3_conflict_9_to_10_allscan_persistence": (
                study_c["baseline_conflict_ge_9_cells"]
            ),
            "Q4_cell_size_over_3_missing_free_cells": study_e,
            "Q5_endpoint_margin_is_major": q5,
            "Q6_primary_source_of_gray_unknown": (
                ordered[0].replace("_EFFECT", "")
            ),
        },
        "interpretation_limit": (
            "Same-recorded-data offline counterfactual only: it cannot prove "
            "new closed-loop policy behavior, reduced safety fallback, "
            "improved exploration efficiency, or deployment safety, and is "
            "not a direct "
            "deployment recommendation."
        ),
    }


def _world_cells_from_mask(
    mask: np.ndarray, origin_world_rc: Sequence[int]
) -> set[tuple[int, int]]:
    """Convert a map-shaped boolean mask to world row/column cells."""
    rows, cols = np.nonzero(mask)
    return {
        (
            int(row) + int(origin_world_rc[0]),
            int(col) + int(origin_world_rc[1]),
        )
        for row, col in zip(rows, cols)
    }


def _map_bounds(cum_map: Any) -> tuple[int, int, int, int]:
    """Return inclusive world row/column bounds for one cumulative map."""
    row_min, col_min = (int(value) for value in cum_map.origin_world_rc)
    return (
        row_min,
        row_min + int(cum_map.map.shape[0]) - 1,
        col_min,
        col_min + int(cum_map.map.shape[1]) - 1,
    )


def _semantic_panel(
    artifacts: DecisionModeArtifacts,
    title: str,
    trajectory: Sequence[tuple[int, int]],
    scale: int = 13,
) -> Any:
    """Render one categorical belief panel with explicit semantic colors."""
    from PIL import Image, ImageDraw

    row_min, row_max, col_min, col_max = _map_bounds(artifacts.cum_map)
    width = (col_max - col_min + 1) * scale
    height = (row_max - row_min + 1) * scale
    label_height = 82
    image = Image.new("RGB", (width, height + label_height), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    observed = artifacts.accumulator.ever_observed_cells()
    effective = _world_cells_from_mask(
        np.asarray(artifacts.frontier.effective_frontier_u8) > 0,
        artifacts.cum_map.origin_world_rc,
    )
    colors = {
        "free": (245, 245, 245),
        "obstacle": (25, 25, 25),
        "never": (120, 120, 120),
        "observed_unknown": (242, 155, 52),
        "frontier": (36, 145, 220),
    }
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            cell = (row, col)
            value = _world_value(artifacts.cum_map, cell)
            if cell in effective:
                color = colors["frontier"]
            elif value == EMPTY:
                color = colors["free"]
            elif value == OBSTACLE:
                color = colors["obstacle"]
            elif cell in observed:
                color = colors["observed_unknown"]
            else:
                color = colors["never"]
            x0 = (col - col_min) * scale
            y0 = (row - row_min) * scale + label_height
            draw.rectangle(
                (x0, y0, x0 + scale - 1, y0 + scale - 1), fill=color
            )
    for row, col in trajectory:
        if row_min <= row <= row_max and col_min <= col <= col_max:
            x0 = (col - col_min) * scale
            y0 = (row - row_min) * scale + label_height
            draw.ellipse(
                (x0 + 3, y0 + 3, x0 + scale - 4, y0 + scale - 4),
                fill=(204, 0, 119),
            )
    draw.text((6, 5), title, fill=(0, 0, 0))
    legend = [
        ("FREE", colors["free"]),
        ("OBSTACLE", colors["obstacle"]),
        ("never-observed UNKNOWN", colors["never"]),
        ("observed-unclassified UNKNOWN", colors["observed_unknown"]),
        ("effective frontier", colors["frontier"]),
        ("trajectory", (204, 0, 119)),
    ]
    x_cursor = 6
    y_cursor = 25
    for label, color in legend:
        label_width = 14 + 6 * len(label)
        if x_cursor + label_width > width:
            x_cursor = 6
            y_cursor += 17
        draw.rectangle(
            (x_cursor, y_cursor, x_cursor + 10, y_cursor + 10),
            fill=color,
        )
        draw.text((x_cursor + 14, y_cursor), label, fill=(0, 0, 0))
        x_cursor += label_width
    return image


def _conflict_persistence_panel(
    baseline: DecisionModeArtifacts,
    allscan_counts: dict[tuple[int, int], Counter],
) -> Any:
    """Render all-scan conflict persistence over the baseline map extent."""
    from PIL import Image, ImageDraw

    scale = 13
    label_height = 50
    row_min, row_max, col_min, col_max = _map_bounds(baseline.cum_map)
    width = (col_max - col_min + 1) * scale
    height = (row_max - row_min + 1) * scale
    image = Image.new("RGB", (width, height + label_height), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            observed = allscan_counts[(row, col)]["observed"]
            if observed:
                persistence = allscan_counts[(row, col)]["conflict"] / observed
                color = (
                    255,
                    int(round(235 * (1.0 - persistence))),
                    int(round(70 * (1.0 - persistence))),
                )
            else:
                color = (110, 110, 110)
            x0 = (col - col_min) * scale
            y0 = (row - row_min) * scale + label_height
            draw.rectangle(
                (x0, y0, x0 + scale - 1, y0 + scale - 1), fill=color
            )
    draw.text(
        (6, 5),
        "All-scan conflict persistence (yellow=0, red=1, gray=unobserved)",
        fill=(0, 0, 0),
    )
    for index in range(101):
        persistence = index / 100.0
        color = (
            255,
            int(round(235 * (1.0 - persistence))),
            int(round(70 * (1.0 - persistence))),
        )
        x0 = 6 + int(index * min(300, width - 20) / 100)
        draw.line((x0, 25, x0, 36), fill=color)
    draw.text((6, 37), "0", fill=(0, 0, 0))
    draw.text((min(300, width - 20), 37), "1", fill=(0, 0, 0))
    return image


def _ray_difference_panel(
    baseline: DecisionModeArtifacts,
    extra_cells: Iterable[tuple[int, int]],
) -> Any:
    """Render E1 supercover-only FREE cells over baseline semantics."""
    from PIL import ImageDraw

    panel = _semantic_panel(baseline, "E1 supercover-only FREE evidence", [])
    draw = ImageDraw.Draw(panel)
    scale = 13
    label_height = 82
    row_min, row_max, col_min, col_max = _map_bounds(baseline.cum_map)
    for row, col in extra_cells:
        if not (row_min <= row <= row_max and col_min <= col <= col_max):
            continue
        x0 = (col - col_min) * scale
        y0 = (row - row_min) * scale + label_height
        draw.rectangle(
            (x0 + 2, y0 + 2, x0 + scale - 3, y0 + scale - 3),
            fill=(0, 255, 210),
        )
    draw.rectangle((6, 61, 16, 71), fill=(0, 255, 210))
    draw.text((20, 61), "supercover-only accepted FREE", fill=(0, 0, 0))
    return panel


def _horizontal_panels(panels: Sequence[Any]) -> Any:
    """Concatenate already labeled images horizontally."""
    from PIL import Image

    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    image = Image.new("RGB", (width, height), (220, 220, 220))
    x_offset = 0
    for panel in panels:
        image.paste(panel, (x_offset, 0))
        x_offset += panel.width
    return image


def export_visualizations(
    output_dir: Path,
    baseline: DecisionModeArtifacts,
    burst3: DecisionModeArtifacts,
    burst5: DecisionModeArtifacts,
    allscan_counts: dict[tuple[int, int], Counter],
    ray_extra_counts: Counter,
    episode: dict[str, Any],
) -> list[str]:
    """Write the four required, semantics-explicit PNG figures."""
    trajectory = episode_trajectory_states(episode["steps"])
    paths = [
        output_dir / "baseline_semantic_unknown.png",
        output_dir / "baseline_conflict_persistence.png",
        output_dir / "decision_single_vs_burst3_vs_burst5.png",
        output_dir / "ray_sampled_vs_supercover_difference.png",
    ]
    _semantic_panel(
        baseline, "A: decision-single production", trajectory
    ).save(paths[0], format="PNG")
    _conflict_persistence_panel(baseline, allscan_counts).save(
        paths[1], format="PNG"
    )
    _horizontal_panels(
        [
            _semantic_panel(baseline, "A decision-single", trajectory),
            _semantic_panel(burst3, "B3 causal burst", trajectory),
            _semantic_panel(burst5, "B5 causal burst", trajectory),
        ]
    ).save(paths[2], format="PNG")
    _ray_difference_panel(baseline, ray_extra_counts).save(
        paths[3], format="PNG"
    )
    return [str(path) for path in paths]


def export_cell_diagnostics(
    output_path: Path,
    baseline: DecisionModeArtifacts,
    burst3: DecisionModeArtifacts,
    burst5: DecisionModeArtifacts,
    allscan_counts: dict[tuple[int, int], Counter],
    allscan_processed: int,
    ray_extra_counts: Counter,
    slam: dict[str, Any],
) -> None:
    """Write the required one-row-per-baseline-cell diagnostic table."""
    slam_rows = {
        (item["row"], item["col"]): item
        for item in slam.get("cells", [])
    }
    fieldnames = [
        "row",
        "col",
        "baseline_value",
        "baseline_free",
        "baseline_obstacle",
        "baseline_conflict",
        "baseline_classification",
        "allscan_observed",
        "allscan_free_only",
        "allscan_obstacle_only",
        "allscan_conflict",
        "allscan_no_evidence",
        "conflict_persistence",
        "free_only_fraction",
        "obstacle_only_fraction",
        "burst3_final",
        "burst5_final",
        "supercover_extra_free_count",
        "slam_free_pixel_fraction",
        "slam_occupied_pixel_fraction",
        "slam_unknown_pixel_fraction",
    ]
    origin_row, origin_col = (
        int(value) for value in baseline.cum_map.origin_world_rc
    )
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for array_row in range(baseline.cum_map.map.shape[0]):
            for array_col in range(baseline.cum_map.map.shape[1]):
                cell = (origin_row + array_row, origin_col + array_col)
                state = baseline.accumulator.cells.get(cell)
                counts = allscan_counts[cell]
                observed = int(counts["observed"])
                slam_row = slam_rows.get(cell, {})
                writer.writerow(
                    {
                        "row": cell[0],
                        "col": cell[1],
                        "baseline_value": int(
                            baseline.cum_map.map[array_row, array_col]
                        ),
                        "baseline_free": (
                            int(state.free_frame_count) if state else 0
                        ),
                        "baseline_obstacle": (
                            int(state.obstacle_frame_count) if state else 0
                        ),
                        "baseline_conflict": (
                            int(state.conflict_frame_count) if state else 0
                        ),
                        "baseline_classification": _classification_name(
                            baseline.accumulator.classify(cell)
                            if state
                            else None
                        ),
                        "allscan_observed": observed,
                        "allscan_free_only": int(counts["free_only"]),
                        "allscan_obstacle_only": int(
                            counts["obstacle_only"]
                        ),
                        "allscan_conflict": int(counts["conflict"]),
                        "allscan_no_evidence": allscan_processed - observed,
                        "conflict_persistence": (
                            float(counts["conflict"]) / observed
                            if observed
                            else ""
                        ),
                        "free_only_fraction": (
                            float(counts["free_only"]) / observed
                            if observed
                            else ""
                        ),
                        "obstacle_only_fraction": (
                            float(counts["obstacle_only"]) / observed
                            if observed
                            else ""
                        ),
                        "burst3_final": _world_value(burst3.cum_map, cell),
                        "burst5_final": _world_value(burst5.cum_map, cell),
                        "supercover_extra_free_count": int(
                            ray_extra_counts[cell]
                        ),
                        "slam_free_pixel_fraction": slam_row.get(
                            "slam_free_pixel_fraction", ""
                        ),
                        "slam_occupied_pixel_fraction": slam_row.get(
                            "slam_occupied_pixel_fraction", ""
                        ),
                        "slam_unknown_pixel_fraction": slam_row.get(
                            "slam_unknown_pixel_fraction", ""
                        ),
                    }
                )


def baseline_gate(
    baseline: DecisionModeArtifacts,
    episode: dict[str, Any],
    episode_path: Path,
    matches: Sequence[DecisionMatch],
) -> dict[str, Any]:
    """Require the exact frozen live result before any counterfactual."""
    comparison = compare_saved_belief(
        baseline.cum_map, episode, episode_path
    )
    metrics = baseline.metrics
    expected_pairs = {
        "known": EXPECTED_BASELINE["known"],
        "free": EXPECTED_BASELINE["free"],
        "obstacle": EXPECTED_BASELINE["obstacle"],
        "unknown": EXPECTED_BASELINE["unknown"],
        "known_history": EXPECTED_BASELINE["known_history"],
        "raw_frontier_history": EXPECTED_BASELINE["raw_frontier_history"],
        "effective_frontier_history": (
            EXPECTED_BASELINE["effective_frontier_history"]
        ),
        "observed_unclassified_history": (
            EXPECTED_BASELINE["observed_unclassified_history"]
        ),
        "never_observed_history": (
            EXPECTED_BASELINE["never_observed_history"]
        ),
    }
    checks = {
        key: metrics[key] == expected
        for key, expected in expected_pairs.items()
    }
    evidence = metrics["evidence_totals"]
    checks.update(
        {
            "evidence_free": (
                evidence["free"] == EXPECTED_BASELINE["evidence_free"]
            ),
            "evidence_obstacle": (
                evidence["obstacle"]
                == EXPECTED_BASELINE["evidence_obstacle"]
            ),
            "evidence_conflict": (
                evidence["conflict"]
                == EXPECTED_BASELINE["evidence_conflict"]
            ),
        }
    )
    occlusion = metrics["occlusion_totals_across_raw_scans"]
    checks.update(
        {
            "occlusion_blocker": (
                occlusion["blocker"]
                == EXPECTED_BASELINE["occlusion_blocker"]
            ),
            "occlusion_suppressed_free": (
                occlusion["suppressed_free"]
                == EXPECTED_BASELINE["occlusion_suppressed_free"]
            ),
            "occlusion_suppressed_obstacle": (
                occlusion["suppressed_obstacle"]
                == EXPECTED_BASELINE["occlusion_suppressed_obstacle"]
            ),
            "occlusion_suppressed_unique": (
                occlusion["suppressed_unique"]
                == EXPECTED_BASELINE["occlusion_suppressed_unique"]
            ),
            "saved_categorical_mismatch_zero": (
                comparison is not None and comparison["mismatch_count"] == 0
            ),
            "saved_match_fraction_one": (
                comparison is not None
                and comparison["match_fraction"] == 1.0
            ),
            "decision_scan_match_exact": all(
                match.absolute_delta_sec == 0.0 for match in matches
            ),
        }
    )
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "categorical_comparison": comparison,
        "mismatch_count": comparison["mismatch_count"] if comparison else None,
        "match_fraction": comparison["match_fraction"] if comparison else None,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the strictly offline command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=Path(
            "/mnt/c/Users/Dk/Downloads/"
            "evidenceaware10_20260819_175106.tar.gz"
        ),
    )
    parser.add_argument(
        "--expected-outer-sha256", default=EXPECTED_OUTER_SHA256
    )
    parser.add_argument(
        "--drl-repository",
        type=Path,
        default=REPOSITORY_ROOT.parent / "DRL-path-finding",
    )
    return parser


def _single_path(paths: Sequence[Path], description: str) -> Path:
    """Require exactly one automatically located dataset artifact."""
    if len(paths) != 1:
        raise ValueError(
            f"expected one {description}, found {len(paths)}: {paths}"
        )
    return paths[0]


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    """Run every gate and study, then write deterministic offline outputs."""
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_root / "multiscan_study"
    )
    episode_path = _single_path(
        sorted((dataset_root / "drl").glob("*.json")), "live episode JSON"
    )
    bag_path = dataset_root / "rosbag"
    if not (bag_path / "metadata.yaml").is_file():
        raise FileNotFoundError(f"missing rosbag metadata: {bag_path}")
    internal_manifest = verify_internal_manifest(dataset_root)
    if internal_manifest["status"] != "PASS":
        raise RuntimeError("internal SHA256SUMS verification failed")
    source_archive = args.source_archive.expanduser().resolve()
    if source_archive.is_file():
        outer_actual = _sha256_file(source_archive)
        outer_status = (
            "PASS" if outer_actual == args.expected_outer_sha256 else "FAIL"
        )
        if outer_status != "PASS":
            raise RuntimeError("outer archive SHA256 verification failed")
    else:
        outer_actual = None
        outer_status = "NOT_AVAILABLE"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    if episode.get("git_commit") != BASE_SHA:
        raise ValueError("episode git_commit is not the frozen baseline SHA")
    if not (
        episode.get("fusion_mode") == "evidence"
        and episode.get("fusion_config", {}).get("name") == "candidate_a"
        and episode.get("coarse_occlusion_mode") == "confirmed_opaque"
        and episode.get("frontier_semantics_mode") == "evidence_aware"
    ):
        raise ValueError(
            "episode semantics do not match the frozen live10 mode"
        )
    contents = read_bag_contents(bag_path)
    decisions = read_decisions(episode)
    matches = match_decision_scans(decisions, contents.scans)
    geometry = _projection_geometry(episode, matches)
    first_time = matches[0].decision.scan_timestamp_sec
    last_time = matches[-1].decision.scan_timestamp_sec
    episode_scans = [
        scan
        for scan in contents.scans
        if first_time <= scan.timestamp_sec <= last_time
    ]
    cumulative_map_type = _load_cumulative_map_type(
        args.drl_repository.expanduser().resolve()
    )

    baseline = run_decision_mode(
        "MODE_A_DECISION_SINGLE_PRODUCTION",
        1,
        matches,
        contents.scans,
        contents.odom,
        episode,
        cumulative_map_type,
    )
    gate = baseline_gate(baseline, episode, episode_path, matches)
    if gate["status"] != "PASS":
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "dataset_root": str(dataset_root),
            "baseline_gate": gate,
            "study_stopped_before_counterfactual": True,
        }
        (output_dir / "baseline_gate_failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError("BASELINE_GATE=FAIL; counterfactuals were not run")

    burst3 = run_decision_mode(
        "MODE_B3_CAUSAL_BURST_ONE_EPOCH",
        3,
        matches,
        contents.scans,
        contents.odom,
        episode,
        cumulative_map_type,
    )
    burst5 = run_decision_mode(
        "MODE_B5_CAUSAL_BURST_ONE_EPOCH",
        5,
        matches,
        contents.scans,
        contents.odom,
        episode,
        cumulative_map_type,
    )
    b3_transitions = baseline_cell_transitions(baseline, burst3)
    b5_transitions = baseline_cell_transitions(baseline, burst5)
    study_c, allscan_records, allscan_counts = run_allscan_density(
        episode_scans,
        contents.odom,
        matches,
        baseline,
        geometry,
    )
    temporal = {
        "D1": run_temporal_mode(
            "D1_1HZ_DIAGNOSTIC",
            1.0,
            episode_scans,
            contents.odom,
            geometry,
            cumulative_map_type,
        ),
        "D2": run_temporal_mode(
            "D2_2HZ_DIAGNOSTIC",
            2.0,
            episode_scans,
            contents.odom,
            geometry,
            cumulative_map_type,
        ),
        "D5": run_temporal_mode(
            "D5_5HZ_DIAGNOSTIC",
            5.0,
            episode_scans,
            contents.odom,
            geometry,
            cumulative_map_type,
        ),
    }
    decision_ray_inputs = []
    for index, match in enumerate(matches):
        blockers, visited = baseline.pre_contexts[index]
        decision_ray_inputs.append(
            ProjectionInput(
                scan=match.scan,
                pose=_pose_for_decision(match.decision),
                agent_state=match.decision.agent_state,
                blockers=blockers,
                visited=visited,
                reference=baseline.epoch_observations[index],
            )
        )
    allscan_ray_inputs = [
        ProjectionInput(
            scan=record.scan,
            pose=record.pose,
            agent_state=record.agent_state,
            blockers=record.blockers,
            visited=record.visited,
            reference=record.production_observation,
        )
        for record in allscan_records
    ]
    e_decision = compare_ray_projection(
        decision_ray_inputs,
        geometry,
        baseline,
        0.25,
        "E0_sampled_ray_current",
        "E1_supercover_same_margin",
    )
    e_allscan = compare_ray_projection(
        allscan_ray_inputs,
        geometry,
        baseline,
        0.25,
        "E0_sampled_ray_current",
        "E1_supercover_same_margin",
    )
    study_e = {
        "only_changed_dimension": "FREE ray traversal sampling",
        "decision_scans": e_decision.report,
        "all_episode_scans": e_allscan.report,
    }
    unresolved_count = len(
        _unresolved_cells(baseline.cum_map, baseline.accumulator)
    )
    e_explained = e_allscan.report[
        "baseline_unknown_receiving_comparison_only_free"
    ]["observed_unclassified_count"]
    f_criterion = (
        e_explained < max(1, math.ceil(unresolved_count * 0.5))
    )
    study_f = None
    if f_criterion:
        f_decision_inputs = [
            ProjectionInput(
                scan=item.scan,
                pose=item.pose,
                agent_state=item.agent_state,
                blockers=item.blockers,
                visited=item.visited,
                reference=counter,
            )
            for item, counter in zip(
                decision_ray_inputs, e_decision.counter_observations
            )
        ]
        f_allscan_inputs = [
            ProjectionInput(
                scan=item.scan,
                pose=item.pose,
                agent_state=item.agent_state,
                blockers=item.blockers,
                visited=item.visited,
                reference=counter,
            )
            for item, counter in zip(
                allscan_ray_inputs, e_allscan.counter_observations
            )
        ]
        f_decision = compare_ray_projection(
            f_decision_inputs,
            geometry,
            baseline,
            0.0,
            "E1_supercover_same_margin",
            "F_supercover_no_extra_endpoint_margin",
        )
        f_allscan = compare_ray_projection(
            f_allscan_inputs,
            geometry,
            baseline,
            0.0,
            "E1_supercover_same_margin",
            "F_supercover_no_extra_endpoint_margin",
        )
        study_f = {
            "run": True,
            "trigger": (
                "E1 explained fewer than 50% of baseline "
                "observed-unclassified cells"
            ),
            "only_changed_dimension": "0.25-cell pre-hit FREE margin",
            "decision_scans": f_decision.report,
            "all_episode_scans": f_allscan.report,
        }
    else:
        study_f = {
            "run": False,
            "trigger": (
                "not met: E1 explained at least 50% of baseline "
                "observed-unclassified cells"
            ),
        }
    slam = evaluate_slam_composition(contents, baseline, geometry)
    map_expansion = analyze_map_expansion(baseline)
    interpretation = build_interpretation(
        baseline,
        b3_transitions,
        b5_transitions,
        study_c,
        study_e,
        study_f if study_f.get("run") else None,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cell_diagnostics.csv"
    export_cell_diagnostics(
        csv_path,
        baseline,
        burst3,
        burst5,
        allscan_counts,
        study_c["total_scans_processed"],
        e_allscan.extra_counts,
        slam,
    )
    visual_paths = export_visualizations(
        output_dir,
        baseline,
        burst3,
        burst5,
        allscan_counts,
        e_allscan.extra_counts,
        episode,
    )
    decision_match_table = [
        {
            "decision_id": match.decision.step_id,
            "json_scan_timestamp_sec": (
                match.decision.scan_timestamp_sec
            ),
            "matched_bag_scan_timestamp_sec": match.scan.timestamp_sec,
            "absolute_difference_sec": match.absolute_delta_sec,
        }
        for match in matches
    ]
    archive_identity = {
        "path": str(source_archive),
        "expected_sha256": args.expected_outer_sha256,
        "actual_sha256": outer_actual,
        "status": outer_status,
    }
    report = {
        "dataset_identity": {
            "dataset_root": str(dataset_root),
            "episode_json": str(episode_path),
            "episode_json_sha256": _sha256_file(episode_path),
            "rosbag_db3": str(bag_path / "rosbag_0.db3"),
            "rosbag_db3_sha256": _sha256_file(bag_path / "rosbag_0.db3"),
            "outer_archive": archive_identity,
            "internal_manifest": internal_manifest,
        },
        "frozen_reference": {
            "base_sha": BASE_SHA,
            "episode_git_commit": episode["git_commit"],
            "execute": episode.get("execute"),
            "total_steps": episode.get("total_steps"),
            "successful_steps": episode.get("successful_steps"),
            "travel_distance": episode.get("travel_distance"),
            "fusion_mode": episode.get("fusion_mode"),
            "fusion_config": episode.get("fusion_config"),
            "coarse_occlusion_mode": episode.get("coarse_occlusion_mode"),
            "frontier_semantics_mode": episode.get(
                "frontier_semantics_mode"
            ),
            "production_runtime_modified": False,
            "offline_analysis_only": True,
        },
        "rosbag": summarize_rosbag(contents),
        "decision_scan_match": {
            "table": decision_match_table,
            "max_absolute_difference_sec": max(
                match.absolute_delta_sec for match in matches
            ),
        },
        "baseline_exact_replay": gate,
        "baseline_gate": gate["status"],
        "study_A_decision_single": baseline.metrics,
        "study_B_burst3": burst3.metrics,
        "study_B_burst5": burst5.metrics,
        "baseline_19_cell_transitions": {
            "B3": b3_transitions,
            "B5": b5_transitions,
        },
        "study_C_allscan_density": study_c,
        "study_D_temporal": temporal,
        "study_E_ray": study_e,
        "study_F_margin": study_f,
        "map_expansion": map_expansion,
        "slam_cell_composition": slam,
        "interpretation": interpretation,
        "determinism": {
            "stable_sorting": True,
            "report_contains_wall_clock_time": False,
        },
        "output_files": {
            "json": str(output_dir / "multiscan_evidence_study.json"),
            "csv": str(csv_path),
            "visualizations": visual_paths,
        },
    }
    json_path = output_dir / "multiscan_evidence_study.json"
    serialized = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=_json_default,
    ) + "\n"
    if serialized != (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    ):
        raise AssertionError("report serialization is not deterministic")
    json_path.write_text(serialized, encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the offline study and print a compact machine-readable summary."""
    args = build_argument_parser().parse_args(argv)
    try:
        report = run_study(args)
    except Exception as exc:
        print(f"multiscan evidence study failed: {exc}", file=sys.stderr)
        return 2
    summary = {
        "baseline_gate": report["baseline_gate"],
        "A": {
            key: report["study_A_decision_single"][key]
            for key in ("known", "free", "obstacle", "unknown")
        },
        "B3": {
            key: report["study_B_burst3"][key]
            for key in ("known", "free", "obstacle", "unknown")
        },
        "B5": {
            key: report["study_B_burst5"][key]
            for key in ("known", "free", "obstacle", "unknown")
        },
        "interpretation": {
            key: report["interpretation"][key]
            for key in (
                "TEMPORAL_UNDERSAMPLING_EFFECT",
                "COARSE_CELL_MIXED_OCCUPANCY_EFFECT",
                "RAY_DISCRETIZATION_EFFECT",
                "PRIMARY_CAUSE",
                "SECONDARY_CAUSE",
            )
        },
        "output_files": report["output_files"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
