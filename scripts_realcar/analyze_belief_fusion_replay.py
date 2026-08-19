#!/usr/bin/env python3
"""Replay real-car decision scans through legacy and evidence belief fusion."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = REPOSITORY_ROOT / "ros2_ws" / "src" / "drl_explore_bridge"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from drl_explore_bridge.realcar_conservative_belief import (  # noqa: E402
    BeliefEvidenceAccumulator,
    BeliefFusionConfig,
    EVIDENCE_FUSION_CANDIDATES,
    EMPTY,
    INVISIBLE,
    OBSTACLE,
    apply_evidence_fusion,
    apply_legacy_fusion,
    cumulative_occlusion_cells,
    named_fusion_config,
    project_scan_to_belief,
    record_traversed_cells_as_free,
    visited_only_local_snap,
)
from drl_explore_bridge.realcar_policy_continuous_runner_node import (  # noqa
    CONTINUOUS_ORIGIN_STATE,
    DEFAULT_LASER_X_IN_BASE_M,
    DEFAULT_LASER_Y_IN_BASE_M,
    belief_evidence_image,
    episode_trajectory_states,
)


@dataclass(frozen=True)
class RecordedScan:
    """Store one deserialized bag scan with its selected timestamp."""

    timestamp_sec: float
    bag_timestamp_sec: float
    timestamp_source: str
    message: Any


@dataclass(frozen=True)
class DecisionObservation:
    """Store one episode decision pose and policy-grid state."""

    step_id: int
    timestamp_sec: float
    robot_x: float
    robot_y: float
    robot_yaw: float
    agent_state: tuple[int, int]


@dataclass(frozen=True)
class MatchedDecision:
    """Associate one decision pose with one uniquely nearest scan."""

    decision: DecisionObservation
    scan: RecordedScan
    absolute_delta_sec: float


def _stamp_to_sec(stamp: Any) -> float:
    """Convert a ROS builtin time message to floating-point seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def read_episode_decisions(
    episode: dict[str, Any],
) -> list[DecisionObservation]:
    """Extract canonical observation-pose timestamps from an episode JSON."""
    decisions: list[DecisionObservation] = []
    for fallback_step_id, step in enumerate(episode.get("steps", [])):
        pose = step.get("observation_pose")
        state = step.get("agent_state")
        if pose is None or state is None:
            continue
        timestamp = pose.get("odom_timestamp")
        if timestamp is None or not math.isfinite(float(timestamp)):
            raise ValueError(
                "decision step has no finite observation_pose.odom_timestamp: "
                f"step_id={step.get('step_id', fallback_step_id)}"
            )
        decisions.append(
            DecisionObservation(
                step_id=int(step.get("step_id", fallback_step_id)),
                timestamp_sec=float(timestamp),
                robot_x=float(pose["x"]),
                robot_y=float(pose["y"]),
                robot_yaw=float(pose["yaw_rad"]),
                agent_state=(int(state[0]), int(state[1])),
            )
        )
    if not decisions:
        raise ValueError("episode JSON contains no replayable decision poses")
    return decisions


def _bag_storage_identifier(bag_path: Path) -> str:
    """Read the rosbag storage identifier without modifying the bag."""
    metadata_path = bag_path / "metadata.yaml"
    if metadata_path.is_file():
        try:
            import yaml

            metadata = yaml.safe_load(
                metadata_path.read_text(encoding="utf-8")
            )
            info = metadata.get("rosbag2_bagfile_information", {})
            return str(info.get("storage_identifier", ""))
        except (ImportError, OSError, AttributeError, ValueError):
            return ""
    if bag_path.suffix.lower() == ".mcap":
        return "mcap"
    if bag_path.suffix.lower() == ".db3":
        return "sqlite3"
    return ""


def read_scans_from_bag(
    bag_path: Path,
    scan_topic: str,
) -> list[RecordedScan]:
    """Deserialize raw LaserScan messages from a rosbag2 recording."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 Humble Python environment is required; source "
            "/opt/ros/humble/setup.bash"
        ) from exc

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_path),
        storage_id=_bag_storage_identifier(bag_path),
    )
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)
    topic_types = {
        topic.name: topic.type for topic in reader.get_all_topics_and_types()
    }
    if scan_topic not in topic_types:
        choices = ", ".join(sorted(topic_types))
        raise ValueError(
            f"scan topic {scan_topic!r} is absent from bag; topics: {choices}"
        )
    message_type = get_message(topic_types[scan_topic])
    scans: list[RecordedScan] = []
    while reader.has_next():
        topic, serialized, bag_timestamp_ns = reader.read_next()
        if topic != scan_topic:
            continue
        message = deserialize_message(serialized, message_type)
        bag_timestamp = float(bag_timestamp_ns) * 1.0e-9
        header_timestamp = _stamp_to_sec(message.header.stamp)
        if header_timestamp > 0.0 and math.isfinite(header_timestamp):
            selected_timestamp = header_timestamp
            source = "header"
        else:
            selected_timestamp = bag_timestamp
            source = "bag"
        scans.append(
            RecordedScan(
                timestamp_sec=selected_timestamp,
                bag_timestamp_sec=bag_timestamp,
                timestamp_source=source,
                message=message,
            )
        )
    scans.sort(key=lambda item: item.timestamp_sec)
    if not scans:
        raise ValueError(f"bag contains no messages on {scan_topic!r}")
    return scans


def match_decisions_to_scans(
    decisions: Sequence[DecisionObservation],
    scans: Sequence[RecordedScan],
    tolerance_sec: float,
    ambiguity_epsilon_sec: float = 1.0e-9,
) -> tuple[list[MatchedDecision], dict[str, Any]]:
    """Match each decision to one bounded, uniquely nearest recorded scan."""
    if tolerance_sec <= 0.0:
        raise ValueError("scan matching tolerance must be > 0")
    if ambiguity_epsilon_sec < 0.0:
        raise ValueError("ambiguity epsilon must be >= 0")
    timestamps = [scan.timestamp_sec for scan in scans]
    matches: list[MatchedDecision] = []
    ambiguous: list[dict[str, Any]] = []
    outside_tolerance: list[dict[str, Any]] = []

    for decision in decisions:
        insertion = bisect.bisect_left(timestamps, decision.timestamp_sec)
        candidate_indexes = {
            index
            for index in (insertion - 1, insertion, insertion + 1)
            if 0 <= index < len(scans)
        }
        ranked = sorted(
            (
                abs(scans[index].timestamp_sec - decision.timestamp_sec),
                index,
            )
            for index in candidate_indexes
        )
        best_delta, best_index = ranked[0]
        if best_delta > tolerance_sec:
            outside_tolerance.append(
                {
                    "step_id": decision.step_id,
                    "decision_timestamp_sec": decision.timestamp_sec,
                    "nearest_scan_timestamp_sec": (
                        scans[best_index].timestamp_sec
                    ),
                    "absolute_delta_sec": best_delta,
                }
            )
            continue
        equally_near = [
            index
            for delta, index in ranked
            if abs(delta - best_delta) <= ambiguity_epsilon_sec
        ]
        if len(equally_near) > 1:
            ambiguous.append(
                {
                    "step_id": decision.step_id,
                    "decision_timestamp_sec": decision.timestamp_sec,
                    "candidate_scan_timestamps_sec": [
                        scans[index].timestamp_sec for index in equally_near
                    ],
                    "absolute_delta_sec": best_delta,
                }
            )
            continue
        matches.append(
            MatchedDecision(
                decision=decision,
                scan=scans[best_index],
                absolute_delta_sec=best_delta,
            )
        )

    statistics = {
        "decision_count": len(decisions),
        "recorded_scan_count": len(scans),
        "matched_decision_count": len(matches),
        "outside_tolerance_count": len(outside_tolerance),
        "ambiguous_match_count": len(ambiguous),
        "scan_matching_tolerance_sec": tolerance_sec,
        "maximum_absolute_delta_sec": (
            max(match.absolute_delta_sec for match in matches)
            if matches
            else None
        ),
        "mean_absolute_delta_sec": (
            float(np.mean([match.absolute_delta_sec for match in matches]))
            if matches
            else None
        ),
        "scan_timestamp_source_counts": {
            source: sum(
                1
                for match in matches
                if match.scan.timestamp_source == source
            )
            for source in sorted(
                {match.scan.timestamp_source for match in matches}
            )
        },
        "outside_tolerance": outside_tolerance,
        "ambiguous_matches": ambiguous,
    }
    if outside_tolerance or ambiguous:
        raise ValueError(
            "scan matching failed: "
            f"outside_tolerance={len(outside_tolerance)} "
            f"ambiguous={len(ambiguous)}; details="
            f"{json.dumps(statistics, sort_keys=True)}"
        )
    if len(matches) != len(decisions):
        raise ValueError("not every decision received a scan match")
    return matches, statistics


def load_custom_fusion_config(raw: str) -> BeliefFusionConfig:
    """Load a named candidate or a JSON configuration file."""
    if raw in EVIDENCE_FUSION_CANDIDATES:
        return named_fusion_config(raw)
    path = Path(raw).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "name" not in payload:
        payload["name"] = path.stem
    return BeliefFusionConfig(**payload)


def _trajectory_obstacle_diagnostics(
    cum_map: Any,
    trajectory: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Count trajectory collisions and nearby obstacles in world cells."""
    unique_trajectory = list(dict.fromkeys(trajectory))
    origin_row, origin_col = (int(value) for value in cum_map.origin_world_rc)
    obstacle_array_rows, obstacle_array_cols = np.nonzero(
        np.asarray(cum_map.map) == OBSTACLE
    )
    obstacle_world = list(
        zip(
            (obstacle_array_rows + origin_row).tolist(),
            (obstacle_array_cols + origin_col).tolist(),
        )
    )
    trajectory_set = set(unique_trajectory)
    trajectory_obstacles = sum(
        1 for cell in obstacle_world if cell in trajectory_set
    )

    def within(radius: int) -> int:
        return sum(
            1
            for obstacle_row, obstacle_col in obstacle_world
            if any(
                max(
                    abs(obstacle_row - trajectory_row),
                    abs(obstacle_col - trajectory_col),
                )
                <= radius
                for trajectory_row, trajectory_col in unique_trajectory
            )
        )

    return {
        "trajectory_cell_count": len(unique_trajectory),
        "trajectory_cell_obstacle_count": trajectory_obstacles,
        "trajectory_cell_obstacle_ratio": (
            float(trajectory_obstacles) / float(len(unique_trajectory))
            if unique_trajectory
            else 0.0
        ),
        "obstacle_cells_within_1_policy_cell_of_trajectory": within(1),
        "obstacle_cells_within_2_policy_cells_of_trajectory": within(2),
        "trajectory_proximity_metric": "Chebyshev policy-cell distance",
    }


def replay_mode(
    mode_name: str,
    config: Optional[BeliefFusionConfig],
    matches: Sequence[MatchedDecision],
    episode: dict[str, Any],
    cumulative_map_type: Any,
    coarse_occlusion_mode: str = "off",
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Rebuild one final cumulative belief without issuing motion commands."""
    if mode_name != "legacy" and config is None:
        raise ValueError("evidence replay requires a fusion config")
    fusion_mode = "legacy" if config is None else "evidence"
    occlusion_mode = str(coarse_occlusion_mode).strip().lower()
    if occlusion_mode not in ("off", "opaque"):
        raise ValueError("coarse_occlusion_mode must be 'off' or 'opaque'")
    cell_size = float(episode.get("cell_size", 0.35))
    if not math.isclose(cell_size, 0.35, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            "replay only supports recorded 0.35m cell size, got "
            f"{cell_size}"
        )
    origin_state = tuple(
        int(value)
        for value in episode.get("origin_state", CONTINUOUS_ORIGIN_STATE)
    )
    origin_x = matches[0].decision.robot_x
    origin_y = matches[0].decision.robot_y
    laser_x = float(
        episode.get("laser_x_in_base_m", DEFAULT_LASER_X_IN_BASE_M)
    )
    laser_y = float(
        episode.get("laser_y_in_base_m", DEFAULT_LASER_Y_IN_BASE_M)
    )
    laser_yaw = float(episode.get("laser_yaw_in_base", 0.0))
    scan_radius_cells = int(episode.get("scan_radius_cells", 10))
    compatibility_true_grid = np.zeros((120, 120), dtype=np.int8)
    accumulator = (
        BeliefEvidenceAccumulator(config) if config is not None else None
    )
    cum_map = None
    known_history: list[int] = []
    frontier_history: list[int] = []
    transition_totals = {
        "free_to_obstacle_transition_count": 0,
        "obstacle_to_free_transition_count": 0,
        "evidence_conflict_frame_observation_count": 0,
        "evidence_free_frame_observation_count": 0,
        "evidence_obstacle_frame_observation_count": 0,
        "obstacle_promotion_count": 0,
        "occlusion_blocker_cell_count": 0,
        "occlusion_suppressed_free_cell_count": 0,
        "occlusion_suppressed_obstacle_cell_count": 0,
        "occlusion_suppressed_cell_count": 0,
    }
    occlusion_step_history: list[dict[str, int]] = []

    for match in matches:
        decision = match.decision
        historical_blockers: frozenset[tuple[int, int]] = frozenset()
        visited_cells: frozenset[tuple[int, int]] = frozenset()
        if occlusion_mode == "opaque" and cum_map is not None:
            historical_blockers, visited_cells = cumulative_occlusion_cells(
                cum_map
            )
        observation = project_scan_to_belief(
            match.scan.message,
            robot_x=decision.robot_x,
            robot_y=decision.robot_y,
            robot_yaw=decision.robot_yaw,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_state=origin_state,
            agent_state=decision.agent_state,
            cell_size=cell_size,
            scan_radius_cells=scan_radius_cells,
            laser_x_in_base=laser_x,
            laser_y_in_base=laser_y,
            laser_yaw_in_base=laser_yaw,
            coarse_occlusion_mode=occlusion_mode,
            historical_obstacle_cells=historical_blockers,
            occlusion_exempt_cells=visited_cells,
        )
        core_snap = (
            observation.local_snap
            if fusion_mode == "legacy"
            else visited_only_local_snap(tuple(observation.local_snap.shape))
        )
        if cum_map is None:
            cum_map = cumulative_map_type(
                compatibility_true_grid,
                decision.agent_state,
                core_snap,
            )
        else:
            cum_map.update(decision.agent_state, core_snap)
        if fusion_mode == "legacy":
            step_stats = apply_legacy_fusion(cum_map, observation)
        else:
            if accumulator is None:
                raise AssertionError("missing evidence accumulator")
            step_stats = apply_evidence_fusion(
                cum_map,
                accumulator,
                observation,
            )
        transition_totals["free_to_obstacle_transition_count"] += (
            step_stats.free_to_obstacle_transitions_this_step
        )
        transition_totals["obstacle_to_free_transition_count"] += (
            step_stats.obstacle_to_free_transitions_this_step
        )
        transition_totals[
            "evidence_conflict_frame_observation_count"
        ] += step_stats.evidence_conflict_cells_this_step
        transition_totals[
            "evidence_free_frame_observation_count"
        ] += step_stats.evidence_free_cells_this_step
        transition_totals[
            "evidence_obstacle_frame_observation_count"
        ] += step_stats.evidence_obstacle_cells_this_step
        transition_totals["obstacle_promotion_count"] += (
            step_stats.invisible_to_obstacle_transitions_this_step
            + step_stats.free_to_obstacle_transitions_this_step
        )
        step_occlusion = {
            "step_id": int(decision.step_id),
            "blocker_cells": len(observation.occlusion_blocker_cells),
            "suppressed_free_cells": len(
                observation.occlusion_suppressed_free_cells
            ),
            "suppressed_obstacle_cells": len(
                observation.occlusion_suppressed_obstacle_cells
            ),
            "suppressed_cells": len(observation.occlusion_suppressed_cells),
        }
        occlusion_step_history.append(step_occlusion)
        transition_totals["occlusion_blocker_cell_count"] += (
            step_occlusion["blocker_cells"]
        )
        transition_totals["occlusion_suppressed_free_cell_count"] += (
            step_occlusion["suppressed_free_cells"]
        )
        transition_totals[
            "occlusion_suppressed_obstacle_cell_count"
        ] += step_occlusion["suppressed_obstacle_cells"]
        transition_totals["occlusion_suppressed_cell_count"] += (
            step_occlusion["suppressed_cells"]
        )
        known_history.append(
            int(np.count_nonzero(np.asarray(cum_map.map) != INVISIBLE))
        )
        frontier_history.append(
            int(np.count_nonzero(cum_map.get_frontier_u8() > 0))
        )

    if cum_map is None:
        raise AssertionError("replay produced no cumulative map")
    trajectory = episode_trajectory_states(episode.get("steps", []))
    final_corrections = record_traversed_cells_as_free(cum_map, trajectory)
    transition_totals["obstacle_to_free_transition_count"] += (
        final_corrections.corrected_from_obstacle
    )
    belief = np.asarray(cum_map.map)
    frontier = np.array(cum_map.get_frontier_u8(), copy=True)
    unknown_count = int(np.count_nonzero(belief == INVISIBLE))
    free_count = int(np.count_nonzero(belief == EMPTY))
    obstacle_count = int(np.count_nonzero(belief == OBSTACLE))
    known_count = free_count + obstacle_count
    metrics: dict[str, Any] = {
        "mode": mode_name,
        "fusion_mode": fusion_mode,
        "fusion_config": config.as_dict() if config is not None else None,
        "coarse_occlusion_mode": occlusion_mode,
        "unknown_count": unknown_count,
        "free_count": free_count,
        "obstacle_count": obstacle_count,
        "known_count": known_count,
        "free_fraction_of_known": (
            float(free_count) / float(known_count) if known_count else 0.0
        ),
        "obstacle_fraction_of_known": (
            float(obstacle_count) / float(known_count) if known_count else 0.0
        ),
        "frontier_count": int(np.count_nonzero(frontier > 0)),
        "known_area_history": known_history,
        "frontier_count_history": frontier_history,
        "occlusion_step_history": occlusion_step_history,
        "origin_world_rc": [int(value) for value in cum_map.origin_world_rc],
        "belief_shape": [int(value) for value in belief.shape],
        "projection_config": {
            "cell_size": cell_size,
            "origin_state": list(origin_state),
            "origin_xy": [origin_x, origin_y],
            "scan_radius_cells": scan_radius_cells,
            "laser_x_in_base_m": laser_x,
            "laser_y_in_base_m": laser_y,
            "laser_yaw_in_base": laser_yaw,
            "coarse_occlusion_mode": occlusion_mode,
        },
        **transition_totals,
        **_trajectory_obstacle_diagnostics(cum_map, trajectory),
    }
    return cum_map, frontier, metrics


def _artifact_path(episode_path: Path, raw_path: Any) -> Optional[Path]:
    """Resolve a saved episode artifact path across Linux/Windows records."""
    if not raw_path:
        return None
    direct = Path(str(raw_path)).expanduser()
    if direct.is_file():
        return direct.resolve()
    basename = str(raw_path).replace("\\", "/").rsplit("/", 1)[-1]
    beside_episode = episode_path.parent / basename
    return beside_episode.resolve() if beside_episode.is_file() else None


def compare_saved_belief(
    replay_map: Any,
    episode: dict[str, Any],
    episode_path: Path,
) -> Optional[dict[str, Any]]:
    """Compare legacy replay with a saved belief in registered world cells."""
    saved_path = _artifact_path(episode_path, episode.get("belief_map_path"))
    saved_origin = episode.get("origin_world_rc")
    if saved_path is None or saved_origin is None:
        return None
    saved = np.load(saved_path, allow_pickle=False)
    replay = np.asarray(replay_map.map)
    saved_origin = (int(saved_origin[0]), int(saved_origin[1]))
    replay_origin = tuple(int(value) for value in replay_map.origin_world_rc)
    min_row = min(saved_origin[0], replay_origin[0])
    min_col = min(saved_origin[1], replay_origin[1])
    max_row = max(
        saved_origin[0] + saved.shape[0],
        replay_origin[0] + replay.shape[0],
    )
    max_col = max(
        saved_origin[1] + saved.shape[1],
        replay_origin[1] + replay.shape[1],
    )
    shape = (max_row - min_row, max_col - min_col)
    saved_world = np.full(shape, INVISIBLE, dtype=np.int8)
    replay_world = np.full(shape, INVISIBLE, dtype=np.int8)
    saved_r0 = saved_origin[0] - min_row
    saved_c0 = saved_origin[1] - min_col
    replay_r0 = replay_origin[0] - min_row
    replay_c0 = replay_origin[1] - min_col
    saved_world[
        saved_r0:saved_r0 + saved.shape[0],
        saved_c0:saved_c0 + saved.shape[1],
    ] = saved
    replay_world[
        replay_r0:replay_r0 + replay.shape[0],
        replay_c0:replay_c0 + replay.shape[1],
    ] = replay
    mismatch = saved_world != replay_world
    known_union = (saved_world != INVISIBLE) | (replay_world != INVISIBLE)
    return {
        "saved_belief_path": str(saved_path),
        "registered_world_shape": [int(value) for value in shape],
        "registered_world_cell_count": int(saved_world.size),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "match_fraction": float(np.mean(~mismatch)),
        "known_union_cell_count": int(np.count_nonzero(known_union)),
        "known_union_mismatch_count": int(
            np.count_nonzero(mismatch & known_union)
        ),
        "known_union_match_fraction": (
            float(np.mean(~mismatch[known_union]))
            if np.any(known_union)
            else 1.0
        ),
    }


def export_mode(
    output_directory: Path,
    mode_name: str,
    cum_map: Any,
    frontier: np.ndarray,
    metrics: dict[str, Any],
    trajectory: Sequence[tuple[int, int]],
) -> Path:
    """Write one mode's arrays, image, and metrics."""
    mode_directory = output_directory / mode_name
    mode_directory.mkdir(parents=True, exist_ok=True)
    np.save(
        mode_directory / "belief.npy",
        np.asarray(cum_map.map),
        allow_pickle=False,
    )
    np.save(mode_directory / "frontier.npy", frontier, allow_pickle=False)
    image = belief_evidence_image(
        np.asarray(cum_map.map),
        frontier,
        tuple(int(value) for value in cum_map.origin_world_rc),
        trajectory,
    )
    image_path = mode_directory / "belief.png"
    image.save(image_path, format="PNG")
    (mode_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return image_path


def export_comparison_figure(
    image_paths: Sequence[tuple[str, Path]],
    output_path: Path,
) -> None:
    """Place mode images in separate labeled panels without map alignment."""
    from PIL import Image, ImageDraw

    panels = [
        (label, Image.open(path).convert("RGB"))
        for label, path in image_paths
    ]
    label_height = 28
    width = sum(image.width for _label, image in panels)
    height = max(image.height for _label, image in panels) + label_height
    canvas = Image.new("RGB", (width, height), (230, 230, 230))
    draw = ImageDraw.Draw(canvas)
    x_offset = 0
    for label, panel in panels:
        draw.text((x_offset + 8, 7), label, fill=(0, 0, 0))
        canvas.paste(panel, (x_offset, label_height))
        x_offset += panel.width
    canvas.save(output_path, format="PNG")


def _load_cumulative_map_type(drl_repository: Path) -> Any:
    """Import the read-only DRL cumulative map dependency."""
    if not (drl_repository / "env" / "core_cummap.py").is_file():
        raise FileNotFoundError(
            "DRL reference repository is missing core_cummap.py: "
            f"{drl_repository}"
        )
    sys.path.insert(0, str(drl_repository))
    from env.core_cummap import CumulativeBeliefMap

    return CumulativeBeliefMap


def _csv_value(value: Any) -> Any:
    """Serialize nested values for one compact comparison CSV cell."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    """Execute scan matching, requested projection modes, and export."""
    episode_path = args.episode_json.expanduser().resolve()
    bag_path = args.bag.expanduser().resolve()
    output_directory = args.output_dir.expanduser().resolve()
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    decisions = read_episode_decisions(episode)
    scans = read_scans_from_bag(bag_path, args.scan_topic)
    matches, matching_statistics = match_decisions_to_scans(
        decisions,
        scans,
        args.scan_tolerance_sec,
    )
    configs = [
        named_fusion_config(name)
        for name in sorted(EVIDENCE_FUSION_CANDIDATES)
    ]
    for raw_config in args.fusion_config:
        config = load_custom_fusion_config(raw_config)
        configs = [item for item in configs if item.name != config.name]
        configs.append(config)
    requested_occlusion_modes = args.coarse_occlusion_mode or ["off"]
    occlusion_modes = list(dict.fromkeys(requested_occlusion_modes))
    mode_specs: list[
        tuple[str, Optional[BeliefFusionConfig], str]
    ] = [("legacy", None, "off")]
    multiple_projection_modes = len(occlusion_modes) > 1
    for config in configs:
        for occlusion_mode in occlusion_modes:
            label = (
                f"{config.name}_{occlusion_mode}"
                if multiple_projection_modes or occlusion_mode != "off"
                else config.name
            )
            mode_specs.append((label, config, occlusion_mode))
    cumulative_map_type = _load_cumulative_map_type(
        args.drl_repository.expanduser().resolve()
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    trajectory = episode_trajectory_states(episode.get("steps", []))
    comparison_rows: list[dict[str, Any]] = []
    image_paths: list[tuple[str, Path]] = []
    legacy_map = None
    replay_maps: dict[str, Any] = {}

    for mode_name, config, occlusion_mode in mode_specs:
        cum_map, frontier, metrics = replay_mode(
            mode_name,
            config,
            matches,
            episode,
            cumulative_map_type,
            coarse_occlusion_mode=occlusion_mode,
        )
        if mode_name == "legacy":
            legacy_map = cum_map
        replay_maps[mode_name] = cum_map
        image_path = export_mode(
            output_directory,
            mode_name,
            cum_map,
            frontier,
            metrics,
            trajectory,
        )
        image_paths.append((mode_name, image_path))
        comparison_rows.append(metrics)

    saved_comparison = (
        compare_saved_belief(legacy_map, episode, episode_path)
        if legacy_map is not None
        else None
    )
    saved_comparisons = {
        mode_name: compare_saved_belief(cum_map, episode, episode_path)
        for mode_name, cum_map in replay_maps.items()
    }
    warnings = [
        "SLAM occupancy was not read or used to mutate any replay belief.",
        (
            "Recorded-observation replay does not validate a new occlusion "
            "setting or any custom evidence-fusion thresholds."
        ),
        (
            "Offline occlusion replay proves only LaserScan-to-belief/frontier "
            "counterfactual behavior under recorded poses and scans; it cannot "
            "prove new policy actions, safety outcomes, motion, observations, "
            "or a reduced safety-intervention rate."
        ),
    ]
    for missing_key in (
        "origin_state",
        "scan_radius_cells",
        "laser_x_in_base_m",
        "laser_y_in_base_m",
        "laser_yaw_in_base",
    ):
        if missing_key not in episode:
            warnings.append(
                f"episode lacks {missing_key}; replay used the recorded-code "
                "default shown in each mode's projection_config"
            )
    report = {
        "bag": str(bag_path),
        "episode_json": str(episode_path),
        "scan_topic": args.scan_topic,
        "matching_statistics": matching_statistics,
        "legacy_saved_belief_comparison": saved_comparison,
        "saved_belief_comparisons": saved_comparisons,
        "modes": comparison_rows,
        "warnings": warnings,
    }
    (output_directory / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = sorted(
        {
            key
            for row in comparison_rows
            for key in row
            if key not in ("known_area_history", "frontier_count_history")
        }
    )
    with (output_directory / "comparison.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(
                {key: _csv_value(row.get(key)) for key in fieldnames}
            )
    export_comparison_figure(
        image_paths,
        output_directory / "belief_comparison.png",
    )
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for offline replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--episode-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--scan-tolerance-sec", type=float, default=0.10)
    parser.add_argument(
        "--coarse-occlusion-mode",
        action="append",
        choices=("off", "opaque"),
        default=None,
        help=(
            "projection mode to replay; repeat with off and opaque for a "
            "counterfactual comparison (default: off)"
        ),
    )
    parser.add_argument(
        "--fusion-config",
        action="append",
        default=[],
        help=(
            "additional named candidate or JSON config; built-in candidate_a, "
            "candidate_b, and candidate_c always run"
        ),
    )
    parser.add_argument(
        "--drl-repository",
        type=Path,
        default=REPOSITORY_ROOT.parent / "DRL-path-finding",
        help=(
            "read-only DRL-path-finding checkout used for "
            "CumulativeBeliefMap"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the offline replay command and print its comparison report."""
    args = build_argument_parser().parse_args(argv)
    try:
        report = run_replay(args)
    except Exception as exc:
        print(f"belief fusion replay failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
