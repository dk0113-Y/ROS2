#!/usr/bin/env python3
"""Evaluate persistent mixed UNKNOWN cells in policy input views offline."""

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
from typing import Any, Optional, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = REPOSITORY_ROOT / "ros2_ws" / "src" / "drl_explore_bridge"
for source_path in (REPOSITORY_ROOT, PACKAGE_SOURCE):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from drl_explore_bridge.realcar_action_adapter import (  # noqa: E402
    RealcarActionAdapter,
)
from drl_explore_bridge.realcar_conservative_belief import (  # noqa: E402
    BeliefEvidenceAccumulator,
    EMPTY,
    INVISIBLE,
    OBSTACLE,
    apply_evidence_fusion,
    cumulative_occlusion_cells,
    frontier_semantics_snapshot,
    named_fusion_config,
    visited_only_local_snap,
)
from drl_explore_bridge.realcar_policy_continuous_runner_node import (  # noqa
    DEFAULT_FOOTPRINT_RADIUS_M,
    build_policy_state_with_frontier_semantics,
    norm_angle,
    scan_capsule_footprint_check,
)
from drl_explore_bridge.realcar_policy_safe_runner_node import (  # noqa
    ACTION_NAMES,
    ACTIONS_8,
    load_policy_model,
)
from scripts_realcar.analyze_belief_fusion_replay import (  # noqa: E402
    _load_cumulative_map_type,
)
from scripts_realcar.analyze_multiscan_evidence_density import (  # noqa
    BASE_SHA as CATEGORICAL_BASE_SHA,
    BagContents,
    DecisionMatch,
    DecisionModeArtifacts,
    OdomPose,
    RecordedScan,
    _json_default,
    _pose_for_decision,
    _project_kwargs,
    _projection_geometry,
    _sha256_file,
    _state_for_pose,
    _unresolved_cells,
    _world_cells_from_mask,
    _world_value,
    baseline_gate,
    evaluate_slam_composition,
    match_decision_scans,
    project_for_mode,
    read_bag_contents,
    read_decisions,
    run_allscan_density,
    run_decision_mode,
    verify_internal_manifest,
)


BASE_SHA = "377c87a3db28cbfcf3f226b2caa2522ee29ced5f"
REQUESTED_CHECKPOINT = Path(
    "/home/robot/robot_repos/DRL-path-finding/"
    "deploy_checkpoints/A_full_method_last.pt"
)
DISCOVERED_READ_ONLY_CHECKPOINT = Path(
    "/mnt/c/Users/Dk/Desktop/SCI/New_A/checkpoint_store/"
    "full_method_main/A_full_method_last.pt"
)
CHECKPOINT_SHA256 = (
    "7d70d54e8e7ad95d623dea91e2f6a7ddd7499acf5e414a5617ba799280b0c3ad"
)
VARIANTS = (
    "M0_CURRENT",
    "M1_CONFLICT_ONLY_2",
    "M2_CONFLICT_DOMINANT_2",
    "M3_CONFLICT_DOMINANT_3",
)
Q_VALUE_TOLERANCE = 1.01e-4
HIGH_CONFLICT_CELLS = (
    (59, 59),
    (59, 62),
    (59, 63),
    (59, 64),
    (60, 59),
    (60, 66),
    (61, 60),
    (61, 61),
    (61, 62),
    (61, 63),
)
LIVE_INTERVENTION_STEPS = frozenset({0, 1, 6, 7, 8, 9})


@dataclass(frozen=True)
class SafetyProxy:
    """One causal scan proxy for a recorded pre-motion pose."""

    scan: RecordedScan
    pose: dict[str, float]
    timing_error_sec: float


@dataclass
class PolicyReplayArtifacts:
    """In-memory outputs from the four policy-view replays."""

    report: dict[str, Any]
    final_masks: dict[str, set[tuple[int, int]]]
    union_masks: dict[str, set[tuple[int, int]]]
    final_cum_map: Any
    final_accumulator: BeliefEvidenceAccumulator
    final_frontier: Any


class PolicyMixedBeliefView:
    """Read-only categorical policy view with an unchanged frontier mask."""

    def __init__(
        self,
        cum_map: Any,
        policy_map: np.ndarray,
        effective_frontier_u8: np.ndarray,
    ) -> None:
        """Copy policy-only arrays and delegate all geometry reads."""
        categorical = np.asarray(policy_map, dtype=np.int8)
        frontier = np.asarray(effective_frontier_u8)
        if categorical.shape != np.asarray(cum_map.map).shape:
            raise ValueError("policy map shape must match cumulative belief")
        if frontier.shape != categorical.shape:
            raise ValueError("frontier shape must match policy map")
        self._cum_map = cum_map
        self.map = np.array(categorical, copy=True)
        self._frontier_u8 = np.array(frontier, copy=True)

    @property
    def frontier_u8(self) -> np.ndarray:
        """Return the original evidence-aware effective frontier."""
        return self._frontier_u8

    def get_frontier_u8(self, refresh: bool = False) -> np.ndarray:
        """Return the frozen frontier without invoking a core refresh."""
        _ = refresh
        return self._frontier_u8

    def __getattr__(self, name: str) -> Any:
        """Delegate coordinate transforms and static map metadata."""
        return getattr(self._cum_map, name)


def _accumulator_snapshot(
    accumulator: BeliefEvidenceAccumulator,
) -> tuple[tuple[Any, ...], ...]:
    """Return an immutable, exact accumulator state snapshot."""
    return tuple(
        (
            int(cell[0]),
            int(cell[1]),
            int(state.free_frame_count),
            int(state.obstacle_frame_count),
            int(state.conflict_frame_count),
            int(state.consecutive_free_frames),
            int(state.consecutive_obstacle_frames),
        )
        for cell, state in sorted(accumulator.cells.items())
    )


def persistent_mixed_world_cells(
    cum_map: Any,
    accumulator: BeliefEvidenceAccumulator,
    variant: str,
) -> frozenset[tuple[int, int]]:
    """Select one of the three fixed persistent-mixed definitions."""
    if variant == "M0_CURRENT":
        return frozenset()
    if variant not in VARIANTS:
        raise ValueError(f"unknown mixed policy variant {variant!r}")
    selected: set[tuple[int, int]] = set()
    origin_row, origin_col = (
        int(value) for value in cum_map.origin_world_rc
    )
    for cell, state in accumulator.cells.items():
        array_row = int(cell[0]) - origin_row
        array_col = int(cell[1]) - origin_col
        if not (
            0 <= array_row < cum_map.map.shape[0]
            and 0 <= array_col < cum_map.map.shape[1]
            and int(cum_map.map[array_row, array_col]) == INVISIBLE
            and int(cum_map.visit_count[array_row, array_col]) <= 0
        ):
            continue
        conflict = int(state.conflict_frame_count)
        free = int(state.free_frame_count)
        obstacle = int(state.obstacle_frame_count)
        if variant == "M1_CONFLICT_ONLY_2":
            matches = conflict >= 2 and free == 0 and obstacle == 0
        elif variant == "M2_CONFLICT_DOMINANT_2":
            matches = conflict >= 2 and conflict > free + obstacle
        else:
            matches = conflict >= 3 and conflict > free + obstacle
        if matches:
            selected.add((int(cell[0]), int(cell[1])))
    return frozenset(selected)


def build_policy_mixed_map(
    cum_map: Any,
    accumulator: BeliefEvidenceAccumulator,
    variant: str,
) -> tuple[np.ndarray, frozenset[tuple[int, int]]]:
    """Return a temporary {-1,0,1} map and its remapped world cells."""
    policy_map = np.array(cum_map.map, dtype=np.int8, copy=True)
    cells = persistent_mixed_world_cells(cum_map, accumulator, variant)
    origin_row, origin_col = (
        int(value) for value in cum_map.origin_world_rc
    )
    for row, col in cells:
        policy_map[row - origin_row, col - origin_col] = OBSTACLE
    if not set(np.unique(policy_map)).issubset({INVISIBLE, EMPTY, OBSTACLE}):
        raise AssertionError("policy view introduced a fourth category")
    return policy_map, cells


def _state_batch_sha256(state_batch: dict[str, Any]) -> str:
    """Hash model tensors deterministically in stable key order."""
    digest = hashlib.sha256()
    for key in sorted(state_batch):
        tensor = state_batch[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def select_pre_motion_safety_proxies(
    episode: dict[str, Any], scans: Sequence[RecordedScan]
) -> list[SafetyProxy]:
    """Choose the nearest causal scan at each pre-motion odom timestamp."""
    timestamps = [scan.timestamp_sec for scan in scans]
    proxies: list[SafetyProxy] = []
    for step in episode["steps"]:
        pose = step.get("pre_motion_pose")
        if pose is None:
            raise ValueError("step lacks pre_motion_pose")
        timestamp = float(pose["odom_timestamp"])
        index = bisect.bisect_right(timestamps, timestamp) - 1
        if index < 0:
            raise ValueError("no causal scan precedes pre-motion pose")
        scan = scans[index]
        proxies.append(
            SafetyProxy(
                scan=scan,
                pose={key: float(value) for key, value in pose.items()},
                timing_error_sec=timestamp - scan.timestamp_sec,
            )
        )
    return proxies


def evaluate_action_safety(
    action_idx: int,
    proxy: SafetyProxy,
    adapter: RealcarActionAdapter,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one raw action with the production capsule helper."""
    pose = proxy.pose
    target = adapter.target_for_action(
        action_idx,
        start_x=pose["x"],
        start_y=pose["y"],
        step_distance=geometry["cell_size"],
    )
    motion_yaw = norm_angle(target.target_yaw - pose["yaw_rad"])
    check = scan_capsule_footprint_check(
        proxy.scan.message,
        motion_yaw,
        target.target_distance + 0.05,
        DEFAULT_FOOTPRINT_RADIUS_M,
        geometry["laser_x_in_base"],
        geometry["laser_y_in_base"],
        geometry["laser_yaw_in_base"],
    )
    return {
        "safety_pass": bool(check.passed),
        "obstruction_type": check.obstruction_type,
        "nearest_capsule_clearance": check.nearest_capsule_clearance,
        "valid_point_count": check.valid_point_count,
    }


def _target_relationship(
    cell: tuple[int, int],
    cum_map: Any,
    accumulator: BeliefEvidenceAccumulator,
    mixed_masks: dict[str, frozenset[tuple[int, int]]],
) -> dict[str, Any]:
    """Describe one action target relative to evidence and map semantics."""
    value = _world_value(cum_map, cell)
    observed = accumulator.has_accepted_evidence(cell)
    relationship: dict[str, Any] = {
        "row": int(cell[0]),
        "col": int(cell[1]),
        "categorical_value": value,
        "confirmed_free": value == EMPTY,
        "confirmed_obstacle": value == OBSTACLE,
        "genuinely_never_observed": value == INVISIBLE and not observed,
        "observed_unclassified": value == INVISIBLE and observed,
    }
    for variant in VARIANTS[1:]:
        mask = mixed_masks[variant]
        relationship[f"{variant}_persistent_mixed"] = cell in mask
        relationship[f"{variant}_adjacent_persistent_mixed"] = any(
            max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) == 1
            for other in mask
        )
    return relationship


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    """Return a deterministic linear percentile or None for no values."""
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _replay_policy_once(
    variants: Sequence[str],
    episode: dict[str, Any],
    matches: Sequence[DecisionMatch],
    geometry: dict[str, Any],
    cumulative_map_type: Any,
    model: Any,
    adapter: Any,
    torch: Any,
    safety_proxies: Sequence[SafetyProxy],
) -> PolicyReplayArtifacts:
    """Replay policy states while changing only temporary map copies."""
    config = named_fusion_config("candidate_a")
    accumulator = BeliefEvidenceAccumulator(config)
    compatibility_grid = np.zeros((120, 120), dtype=np.int8)
    action_adapter = RealcarActionAdapter(
        ACTIONS_8,
        ACTION_NAMES,
        diagonal_mode=str(episode.get("diagonal_mode", "grid_center")),
    )
    cum_map = None
    rows: list[dict[str, Any]] = []
    union_masks = {variant: set() for variant in variants}
    final_masks = {variant: set() for variant in variants}
    mutation_checks = Counter()
    state_hashes: dict[str, list[str]] = {
        variant: [] for variant in variants
    }
    for index, match in enumerate(matches):
        decision = match.decision
        if cum_map is None:
            blockers: frozenset[tuple[int, int]] = frozenset()
            visited: frozenset[tuple[int, int]] = frozenset()
        else:
            blockers, visited = cumulative_occlusion_cells(cum_map)
        observation = project_for_mode(
            "production",
            match.scan.message,
            **_project_kwargs(
                geometry,
                _pose_for_decision(decision),
                decision.agent_state,
                blockers,
                visited,
            ),
        )
        core_snap = visited_only_local_snap(
            tuple(observation.local_snap.shape)
        )
        if cum_map is None:
            cum_map = cumulative_map_type(
                compatibility_grid, decision.agent_state, core_snap
            )
        else:
            cum_map.update(decision.agent_state, core_snap)
        apply_evidence_fusion(cum_map, accumulator, observation)
        frontier = frontier_semantics_snapshot(
            cum_map, accumulator, "evidence_aware"
        )
        mixed_masks = {
            variant: persistent_mixed_world_cells(
                cum_map, accumulator, variant
            )
            for variant in VARIANTS
        }
        map_before = np.array(cum_map.map, copy=True)
        visits_before = np.array(cum_map.visit_count, copy=True)
        accumulator_before = _accumulator_snapshot(accumulator)
        raw_frontier_before = np.array(
            frontier.raw_frontier_u8, copy=True
        )
        effective_frontier_before = np.array(
            frontier.effective_frontier_u8, copy=True
        )
        recent_positions = [
            item.decision.agent_state for item in matches[: index + 1]
        ][-8:]
        variant_rows: dict[str, Any] = {}
        for variant in variants:
            policy_map, remapped = build_policy_mixed_map(
                cum_map, accumulator, variant
            )
            difference_world = _world_cells_from_mask(
                policy_map != map_before, cum_map.origin_world_rc
            )
            if difference_world != set(remapped):
                raise AssertionError(
                    "policy copy changed cells outside the mixed mask"
                )
            if variant == "M0_CURRENT":
                policy_belief = cum_map
            else:
                policy_belief = PolicyMixedBeliefView(
                    cum_map,
                    policy_map,
                    effective_frontier_before,
                )
            state_batch, _state_meta = (
                build_policy_state_with_frontier_semantics(
                    adapter,
                    policy_belief,
                    decision.agent_state,
                    frontier,
                    "evidence_aware",
                    recent_positions,
                )
            )
            repeated_batch, _repeated_meta = (
                build_policy_state_with_frontier_semantics(
                    adapter,
                    policy_belief,
                    decision.agent_state,
                    frontier,
                    "evidence_aware",
                    recent_positions,
                )
            )
            state_hash = _state_batch_sha256(state_batch)
            repeated_hash = _state_batch_sha256(repeated_batch)
            if state_hash != repeated_hash:
                raise AssertionError("model state generation is not stable")
            state_hashes[variant].append(state_hash)
            with torch.inference_mode():
                q_values = model(**state_batch, return_aux=False)
            q_full = [
                float(value)
                for value in q_values.detach().cpu().numpy()[0].tolist()
            ]
            raw_action = int(torch.argmax(q_values, dim=1).item())
            q_ranked = sorted(
                range(len(q_full)), key=lambda item: q_full[item], reverse=True
            )
            safety = evaluate_action_safety(
                raw_action,
                safety_proxies[index],
                action_adapter,
                geometry,
            )
            first_safe_rank = None
            first_safe_action = None
            ranked_safety: list[dict[str, Any]] = []
            for rank, action in enumerate(q_ranked, start=1):
                candidate = evaluate_action_safety(
                    action,
                    safety_proxies[index],
                    action_adapter,
                    geometry,
                )
                ranked_safety.append(
                    {
                        "rank": rank,
                        "action": action,
                        "action_name": ACTION_NAMES[action],
                        **candidate,
                    }
                )
                if first_safe_rank is None and candidate["safety_pass"]:
                    first_safe_rank = rank
                    first_safe_action = action
            action_delta = ACTIONS_8[raw_action]
            target_cell = (
                decision.agent_state[0] + int(action_delta[0]),
                decision.agent_state[1] + int(action_delta[1]),
            )
            q_ordered = sorted(q_full, reverse=True)
            variant_rows[variant] = {
                "state_batch_sha256": state_hash,
                "state_generation_repeat_match": True,
                "q_values_full_precision": q_full,
                "q_values_rounded_4dp": [
                    round(value, 4) for value in q_full
                ],
                "raw_action": raw_action,
                "raw_action_name": ACTION_NAMES[raw_action],
                "top1_top2_margin": q_ordered[0] - q_ordered[1],
                "q_ranked_actions": q_ranked,
                "raw_action_safety": safety,
                "ranked_action_safety": ranked_safety,
                "first_safe_rank": first_safe_rank,
                "first_safe_action": first_safe_action,
                "target_relationship": _target_relationship(
                    target_cell, cum_map, accumulator, mixed_masks
                ),
                "effective_frontier_mismatch_count": 0,
                "categorical_belief_mutation_count": 0,
                "remapped_cell_count": len(remapped),
                "remapped_world_cells": [
                    list(cell) for cell in sorted(remapped)
                ],
            }
            union_masks[variant].update(remapped)
            final_masks[variant] = set(remapped)
            if not np.array_equal(cum_map.map, map_before):
                mutation_checks["categorical_map_mutation"] += 1
            if not np.array_equal(cum_map.visit_count, visits_before):
                mutation_checks["visit_map_mutation"] += 1
            if _accumulator_snapshot(accumulator) != accumulator_before:
                mutation_checks["accumulator_mutation"] += 1
            if not np.array_equal(
                frontier.raw_frontier_u8, raw_frontier_before
            ):
                mutation_checks["raw_frontier_mutation"] += 1
            if not np.array_equal(
                frontier.effective_frontier_u8,
                effective_frontier_before,
            ):
                mutation_checks["effective_frontier_mutation"] += 1
        m0_action = variant_rows["M0_CURRENT"]["raw_action"]
        for variant in variants:
            variant_rows[variant]["action_changed_vs_M0"] = (
                variant_rows[variant]["raw_action"] != m0_action
            )
        rows.append(
            {
                "step_id": decision.step_id,
                "agent_state": list(decision.agent_state),
                "frontier": {
                    "raw_count": frontier.raw_frontier_count,
                    "effective_count": frontier.effective_frontier_count,
                    "effective_sha256": hashlib.sha256(
                        effective_frontier_before.tobytes()
                    ).hexdigest(),
                },
                "safety_proxy": {
                    "scan_timestamp_sec": (
                        safety_proxies[index].scan.timestamp_sec
                    ),
                    "pre_motion_odom_timestamp_sec": float(
                        safety_proxies[index].pose["odom_timestamp"]
                    ),
                    "timing_error_sec": (
                        safety_proxies[index].timing_error_sec
                    ),
                },
                "live": {
                    "raw_action": int(
                        episode["steps"][index]["raw_policy_action"]
                    ),
                    "q_values": [
                        float(value)
                        for value in episode["steps"][index]["q_values"]
                    ],
                    "safety_fallback_used": bool(
                        episode["steps"][index]["safety_fallback_used"]
                    ),
                    "requested_action_obstruction_type": (
                        episode["steps"][index].get(
                            "requested_action_obstruction_type"
                        )
                    ),
                },
                "variants": variant_rows,
            }
        )
    if cum_map is None:
        raise AssertionError("policy replay produced no cumulative map")
    if any(mutation_checks.values()):
        raise AssertionError(f"policy view mutation: {dict(mutation_checks)}")
    summary: dict[str, Any] = {}
    for variant in variants:
        variant_steps = [row["variants"][variant] for row in rows]
        obstruction_types = Counter(
            str(step["raw_action_safety"]["obstruction_type"])
            for step in variant_steps
            if not step["raw_action_safety"]["safety_pass"]
        )
        action_changes = sum(
            step["raw_action"]
            != row["variants"]["M0_CURRENT"]["raw_action"]
            for row, step in zip(rows, variant_steps)
        )
        raw_pass = sum(
            bool(step["raw_action_safety"]["safety_pass"])
            for step in variant_steps
        )
        summary[variant] = {
            "decision_count": len(variant_steps),
            "raw_safety_pass_count": raw_pass,
            "raw_safety_blocked_count": len(variant_steps) - raw_pass,
            "raw_safety_pass_rate": raw_pass / len(variant_steps),
            "raw_blockage_types": dict(sorted(obstruction_types.items())),
            "intervention_rate": (len(variant_steps) - raw_pass)
            / len(variant_steps),
            "action_change_count_vs_M0": action_changes,
            "action_change_rate_vs_M0": action_changes / len(variant_steps),
            "first_safe_rank_distribution": dict(
                sorted(
                    Counter(
                        str(step["first_safe_rank"])
                        for step in variant_steps
                    ).items()
                )
            ),
            "remapped_cell_count_per_step": [
                step["remapped_cell_count"] for step in variant_steps
            ],
            "remapped_cell_count_union": len(union_masks[variant]),
            "remapped_cell_count_final": len(final_masks[variant]),
            "state_hashes": state_hashes[variant],
        }
    return PolicyReplayArtifacts(
        report={
            "steps": rows,
            "summary": summary,
            "mutation_checks": {
                "categorical_map_mutation_count": 0,
                "visit_map_mutation_count": 0,
                "accumulator_mutation_count": 0,
                "raw_frontier_mutation_count": 0,
                "effective_frontier_mutation_count": 0,
            },
        },
        final_masks=final_masks,
        union_masks=union_masks,
        final_cum_map=cum_map,
        final_accumulator=accumulator,
        final_frontier=frontier,
    )


def replay_policy_views(
    episode: dict[str, Any],
    matches: Sequence[DecisionMatch],
    geometry: dict[str, Any],
    cumulative_map_type: Any,
    checkpoint: Path,
    scans: Sequence[RecordedScan],
) -> PolicyReplayArtifacts:
    """Pass the M0 model gate before evaluating any mixed counterfactual."""
    model, adapter, torch = load_policy_model(str(checkpoint))
    proxies = select_pre_motion_safety_proxies(episode, scans)
    m0 = _replay_policy_once(
        ("M0_CURRENT",),
        episode,
        matches,
        geometry,
        cumulative_map_type,
        model,
        adapter,
        torch,
        proxies,
    )
    actions_match = []
    q_max_errors = []
    q_full_precision_errors = []
    obstruction_matches = []
    frontier_matches = []
    for row in m0.report["steps"]:
        current = row["variants"]["M0_CURRENT"]
        actions_match.append(
            current["raw_action"] == row["live"]["raw_action"]
        )
        q_max_errors.append(
            max(
                abs(actual - expected)
                for actual, expected in zip(
                    current["q_values_rounded_4dp"],
                    row["live"]["q_values"],
                )
            )
        )
        q_full_precision_errors.append(
            max(
                abs(actual - expected)
                for actual, expected in zip(
                    current["q_values_full_precision"],
                    row["live"]["q_values"],
                )
            )
        )
        obstruction_matches.append(
            current["raw_action_safety"]["obstruction_type"]
            == row["live"]["requested_action_obstruction_type"]
        )
        live_step = episode["steps"][int(row["step_id"])]
        frontier_matches.append(
            row["frontier"]["raw_count"]
            == int(live_step["raw_frontier_count"])
            and row["frontier"]["effective_count"]
            == int(live_step["effective_frontier_count"])
        )
    gate = {
        "status": (
            "PASS"
            if all(actions_match)
            and max(q_max_errors) <= Q_VALUE_TOLERANCE
            and all(obstruction_matches)
            and all(frontier_matches)
            else "FAIL"
        ),
        "action_match_count": sum(actions_match),
        "decision_count": len(actions_match),
        "q_value_tolerance": Q_VALUE_TOLERANCE,
        "q_comparison_semantics": (
            "replayed q rounded to live JSON precision (4 decimal places)"
        ),
        "max_q_value_absolute_error": max(q_max_errors),
        "max_full_precision_vs_rounded_live_absolute_error": max(
            q_full_precision_errors
        ),
        "safety_proxy_obstruction_type_match_count": sum(
            obstruction_matches
        ),
        "frontier_count_match_count": sum(frontier_matches),
        "safety_semantics_label": "SAFETY_FEASIBILITY_PROXY",
        "proxy_timing_error_sec": {
            "min": min(proxy.timing_error_sec for proxy in proxies),
            "median": float(
                statistics.median(
                    proxy.timing_error_sec for proxy in proxies
                )
            ),
            "p95": _percentile(
                [proxy.timing_error_sec for proxy in proxies], 95.0
            ),
            "max": max(proxy.timing_error_sec for proxy in proxies),
        },
    }
    if gate["status"] != "PASS":
        raise RuntimeError("M0_MODEL_ACTION_GATE=FAIL")
    full = _replay_policy_once(
        VARIANTS,
        episode,
        matches,
        geometry,
        cumulative_map_type,
        model,
        adapter,
        torch,
        proxies,
    )
    for first, second in zip(m0.report["steps"], full.report["steps"]):
        if (
            first["variants"]["M0_CURRENT"]["state_batch_sha256"]
            != second["variants"]["M0_CURRENT"]["state_batch_sha256"]
        ):
            raise AssertionError("M0 state changed across the gate boundary")
    full.report["M0_model_action_gate"] = gate
    return full


def _cell_center(
    cell: tuple[int, int], geometry: dict[str, Any]
) -> tuple[float, float]:
    """Convert one policy-grid cell to its odom-frame center."""
    return (
        geometry["origin_x"]
        + (cell[1] - geometry["origin_state"][1])
        * geometry["cell_size"],
        geometry["origin_y"]
        - (cell[0] - geometry["origin_state"][0])
        * geometry["cell_size"],
    )


def _obstacle_endpoints(
    record: Any, geometry: dict[str, Any]
) -> np.ndarray:
    """Return finite in-radius scan endpoints in the odom frame."""
    scan = record.scan.message
    ranges = np.asarray(scan.ranges, dtype=float)
    valid = (
        np.isfinite(ranges)
        & (ranges >= float(scan.range_min))
        & (ranges <= float(scan.range_max))
        & (
            ranges
            <= geometry["scan_radius_cells"] * geometry["cell_size"]
            + geometry["cell_size"]
        )
    )
    if not np.any(valid):
        return np.empty((0, 2), dtype=float)
    pose = record.pose
    robot_cos = math.cos(pose.yaw)
    robot_sin = math.sin(pose.yaw)
    laser_x = (
        pose.x
        + robot_cos * geometry["laser_x_in_base"]
        - robot_sin * geometry["laser_y_in_base"]
    )
    laser_y = (
        pose.y
        + robot_sin * geometry["laser_x_in_base"]
        + robot_cos * geometry["laser_y_in_base"]
    )
    indexes = np.nonzero(valid)[0]
    angles = (
        pose.yaw
        + geometry["laser_yaw_in_base"]
        + float(scan.angle_min)
        + indexes * float(scan.angle_increment)
    )
    selected_ranges = ranges[indexes]
    return np.column_stack(
        (
            laser_x + selected_ranges * np.cos(angles),
            laser_y + selected_ranges * np.sin(angles),
        )
    )


def center_traversability_audit(
    cells: Sequence[tuple[int, int]],
    allscan_records: Sequence[Any],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Audit cell-center clearance using all-scan obstacle endpoints."""
    distances: dict[tuple[int, int], list[float]] = {
        tuple(cell): [] for cell in cells
    }
    centers = {cell: _cell_center(cell, geometry) for cell in distances}
    for record in allscan_records:
        observation = record.production_observation
        observed = set(observation.free_cells)
        observed.update(observation.obstacle_cells)
        observed.update(observation.conflict_cells)
        relevant = sorted(observed & set(distances))
        if not relevant:
            continue
        endpoints = _obstacle_endpoints(record, geometry)
        if endpoints.size == 0:
            continue
        for cell in relevant:
            center_x, center_y = centers[cell]
            delta = endpoints - np.asarray([center_x, center_y])
            nearest = float(np.sqrt(np.sum(delta * delta, axis=1)).min())
            distances[cell].append(nearest)
    rows = []
    for cell in sorted(distances):
        values = distances[cell]
        collisions = sum(
            value < DEFAULT_FOOTPRINT_RADIUS_M for value in values
        )
        rows.append(
            {
                "row": cell[0],
                "col": cell[1],
                "observed_scan_count": len(values),
                "median_center_clearance_m": (
                    float(statistics.median(values)) if values else None
                ),
                "p05_center_clearance_m": _percentile(values, 5.0),
                "minimum_center_clearance_m": min(values) if values else None,
                "center_collision_fraction": (
                    collisions / len(values) if values else None
                ),
                "footprint_radius_m": DEFAULT_FOOTPRINT_RADIUS_M,
            }
        )
    return {
        "semantics": (
            "analysis-only nearest valid in-radius obstacle endpoint; "
            "never entered belief or policy"
        ),
        "strict_collision_test": "center_clearance_m < 0.20",
        "cells": rows,
    }


def _slam_category(row: dict[str, Any]) -> str:
    """Return the pre-registered mutually exclusive SLAM risk label."""
    occupied = row.get("slam_occupied_pixel_fraction")
    unknown = row.get("slam_unknown_pixel_fraction")
    if occupied is None or unknown is None:
        return "not_available"
    if math.isclose(float(unknown), 1.0, abs_tol=1e-12):
        return "unknown_only"
    if float(occupied) > 0.0:
        return "occupied_fraction_gt_0"
    return "occupied_fraction_eq_0"


def _collision_bin(value: Optional[float]) -> str:
    """Return one pre-registered center-collision fraction bin."""
    if value is None:
        return "not_available"
    if value == 0.0:
        return "eq_0"
    if value <= 0.25:
        return "gt_0_le_0.25"
    if value <= 0.75:
        return "gt_0.25_le_0.75"
    return "gt_0.75"


def false_block_analysis(
    replay: PolicyReplayArtifacts,
    slam: dict[str, Any],
    traversability: dict[str, Any],
) -> dict[str, Any]:
    """Summarize SLAM and center-clearance risk for remapped cells."""
    slam_rows = {
        (row["row"], row["col"]): row for row in slam.get("cells", [])
    }
    center_rows = {
        (row["row"], row["col"]): row
        for row in traversability["cells"]
    }
    result: dict[str, Any] = {}
    for variant in VARIANTS[1:]:
        variant_result: dict[str, Any] = {}
        for population, cells in (
            ("final_decision", replay.final_masks[variant]),
            ("ever_remapped_union", replay.union_masks[variant]),
        ):
            slam_counts = Counter()
            collision_counts = Counter()
            mainly_free_cells = []
            pure_free_cells = []
            low_collision_cells = []
            for cell in sorted(cells):
                slam_row = slam_rows.get(cell, {})
                slam_counts[_slam_category(slam_row)] += 1
                free_fraction = slam_row.get("slam_free_pixel_fraction")
                unknown_fraction = slam_row.get(
                    "slam_unknown_pixel_fraction"
                )
                if free_fraction is not None and free_fraction > 0.5:
                    mainly_free_cells.append(list(cell))
                if (
                    free_fraction is not None
                    and unknown_fraction is not None
                    and math.isclose(float(free_fraction), 1.0)
                    and math.isclose(float(unknown_fraction), 0.0)
                ):
                    pure_free_cells.append(list(cell))
                center = center_rows.get(cell, {})
                collision = center.get("center_collision_fraction")
                collision_counts[_collision_bin(collision)] += 1
                if collision is not None and collision <= 0.25:
                    low_collision_cells.append(list(cell))
            variant_result[population] = {
                "cell_count": len(cells),
                "slam": {
                    **{
                        key: int(slam_counts[key])
                        for key in (
                            "occupied_fraction_gt_0",
                            "occupied_fraction_eq_0",
                            "unknown_only",
                            "not_available",
                        )
                    },
                    "mainly_free_fraction_gt_0.5_count": len(
                        mainly_free_cells
                    ),
                    "mainly_free_fraction_gt_0.5_cells": mainly_free_cells,
                    "pure_free_fraction_eq_1_cells": pure_free_cells,
                },
                "center_collision_fraction": {
                    **{
                        key: int(collision_counts[key])
                        for key in (
                            "eq_0",
                            "gt_0_le_0.25",
                            "gt_0.25_le_0.75",
                            "gt_0.75",
                            "not_available",
                        )
                    },
                    "fraction_le_0.25_cells": low_collision_cells,
                },
            }
        result[variant] = variant_result
    return result


def spatial_correlation(report: dict[str, Any]) -> dict[str, Any]:
    """Measure M0 raw targets against each mixed-region definition."""
    result: dict[str, Any] = {
        "live_intervention_steps": sorted(LIVE_INTERVENTION_STEPS),
        "step_details": [],
        "variants": {},
    }
    for row in report["steps"]:
        step_id = int(row["step_id"])
        relation = row["variants"]["M0_CURRENT"]["target_relationship"]
        result["step_details"].append(
            {
                "step_id": step_id,
                "live_intervention": step_id in LIVE_INTERVENTION_STEPS,
                "target": relation,
            }
        )
    for variant in VARIANTS[1:]:
        key = f"{variant}_persistent_mixed"
        adjacent_key = f"{variant}_adjacent_persistent_mixed"
        interventions = [
            row
            for row in result["step_details"]
            if row["live_intervention"]
        ]
        direct = sum(bool(row["target"][key]) for row in interventions)
        adjacent = sum(
            bool(row["target"][adjacent_key]) for row in interventions
        )
        either = sum(
            bool(row["target"][key] or row["target"][adjacent_key])
            for row in interventions
        )
        fraction = either / len(interventions)
        strength = (
            "STRONG"
            if fraction >= 0.5
            else "MODERATE" if fraction >= 0.25 else "WEAK"
        )
        result["variants"][variant] = {
            "direct_count": direct,
            "adjacent_count": adjacent,
            "direct_or_adjacent_count": either,
            "intervention_count": len(interventions),
            "direct_or_adjacent_fraction": fraction,
            "spatial_correlation_strength": strength,
        }
    return result


def high_conflict_focus(
    baseline: DecisionModeArtifacts,
    allscan_counts: dict[tuple[int, int], Counter],
    slam: dict[str, Any],
    traversability: dict[str, Any],
    replay: PolicyReplayArtifacts,
) -> list[dict[str, Any]]:
    """Join every requested diagnostic for the ten fixed focus cells."""
    slam_rows = {
        (row["row"], row["col"]): row for row in slam.get("cells", [])
    }
    center_rows = {
        (row["row"], row["col"]): row
        for row in traversability["cells"]
    }
    rows = []
    for cell in HIGH_CONFLICT_CELLS:
        state = baseline.accumulator.cells[cell]
        counts = allscan_counts[cell]
        observed = int(counts["observed"])
        slam_row = slam_rows.get(cell, {})
        center_row = center_rows.get(cell, {})
        rows.append(
            {
                "row": cell[0],
                "col": cell[1],
                "decision_free": int(state.free_frame_count),
                "decision_obstacle": int(state.obstacle_frame_count),
                "decision_conflict": int(state.conflict_frame_count),
                "allscan_observed": observed,
                "allscan_conflict_persistence": (
                    int(counts["conflict"]) / observed if observed else None
                ),
                "slam_free_pixel_fraction": slam_row.get(
                    "slam_free_pixel_fraction"
                ),
                "slam_occupied_pixel_fraction": slam_row.get(
                    "slam_occupied_pixel_fraction"
                ),
                "slam_unknown_pixel_fraction": slam_row.get(
                    "slam_unknown_pixel_fraction"
                ),
                "center_collision_fraction": center_row.get(
                    "center_collision_fraction"
                ),
                **{
                    f"{variant}_remapped_final": (
                        cell in replay.final_masks[variant]
                    )
                    for variant in VARIANTS[1:]
                },
                **{
                    f"{variant}_remapped_ever": (
                        cell in replay.union_masks[variant]
                    )
                    for variant in VARIANTS[1:]
                },
            }
        )
    return rows


def build_interpretation(
    replay: PolicyReplayArtifacts,
    correlation: dict[str, Any],
    false_block: dict[str, Any],
) -> dict[str, Any]:
    """Answer A-E with transparent, fixed decision rules."""
    m0_steps = [
        row["variants"]["M0_CURRENT"] for row in replay.report["steps"]
    ]
    mode_answers: dict[str, Any] = {}
    for variant in VARIANTS[1:]:
        candidate_steps = [
            row["variants"][variant] for row in replay.report["steps"]
        ]
        improved = [
            int(row["step_id"])
            for row, m0, candidate in zip(
                replay.report["steps"], m0_steps, candidate_steps
            )
            if not m0["raw_action_safety"]["safety_pass"]
            and candidate["raw_action_safety"]["safety_pass"]
        ]
        regressed = [
            int(row["step_id"])
            for row, m0, candidate in zip(
                replay.report["steps"], m0_steps, candidate_steps
            )
            if m0["raw_action_safety"]["safety_pass"]
            and not candidate["raw_action_safety"]["safety_pass"]
        ]
        changed = [
            int(row["step_id"])
            for row, candidate in zip(
                replay.report["steps"], candidate_steps
            )
            if candidate["action_changed_vs_M0"]
        ]
        m0_pass = replay.report["summary"]["M0_CURRENT"][
            "raw_safety_pass_count"
        ]
        candidate_pass = replay.report["summary"][variant][
            "raw_safety_pass_count"
        ]
        delta = candidate_pass - m0_pass
        risk = false_block[variant]["final_decision"]
        available_collision = (
            risk["cell_count"]
            - risk["center_collision_fraction"]["not_available"]
        )
        low_collision = (
            risk["center_collision_fraction"]["eq_0"]
            + risk["center_collision_fraction"]["gt_0_le_0.25"]
        )
        low_collision_fraction = (
            low_collision / available_collision
            if available_collision
            else None
        )
        available_slam = (
            risk["cell_count"] - risk["slam"]["not_available"]
        )
        slam_zero_fraction = (
            risk["slam"]["occupied_fraction_eq_0"] / available_slam
            if available_slam
            else None
        )
        localized_risk = bool(
            risk["slam"]["pure_free_fraction_eq_1_cells"]
            or risk["center_collision_fraction"][
                "fraction_le_0.25_cells"
            ]
        )
        population_level_risk = bool(
            (low_collision_fraction is not None
             and low_collision_fraction >= 0.5)
            or (slam_zero_fraction is not None
                and slam_zero_fraction >= 0.5)
        )
        if delta <= 0:
            judgement = "NOT_PROMISING"
        elif delta >= 2 and len(improved) >= 2 and not regressed:
            judgement = "PROMISING"
        else:
            judgement = "INCONCLUSIVE"
        mode_answers[variant] = {
            "raw_safety_pass_delta_vs_M0": delta,
            "improved_steps": improved,
            "regressed_steps": regressed,
            "action_changed_steps": changed,
            "improvement_distribution": (
                "NONE"
                if not improved
                else "ONE_OR_TWO_STEPS"
                if len(improved) <= 2
                else "BROAD_ACROSS_DECISIONS"
            ),
            "low_center_collision_fraction_among_available": (
                low_collision_fraction
            ),
            "slam_zero_occupied_fraction_among_available": (
                slam_zero_fraction
            ),
            "localized_false_block_evidence": localized_risk,
            "population_level_false_block_risk_flag": (
                population_level_risk
            ),
            "false_block_risk_cells": {
                "slam_pure_free": risk["slam"][
                    "pure_free_fraction_eq_1_cells"
                ],
                "slam_mainly_free": risk["slam"][
                    "mainly_free_fraction_gt_0.5_cells"
                ],
                "center_collision_fraction_le_0.25": risk[
                    "center_collision_fraction"
                ]["fraction_le_0.25_cells"],
            },
            "production_candidate_design_judgement": judgement,
        }
    return {
        "decision_rule": {
            "PROMISING": (
                "raw safety pass improves by >=2 decisions, on >=2 steps, "
                "with no safe-to-blocked regression"
            ),
            "NOT_PROMISING": "raw safety pass count does not improve",
            "INCONCLUSIVE": "all other outcomes",
            "population_level_false_block_risk_flag": (
                ">=50% low center-collision (<=0.25) or >=50% SLAM "
                "occupied-fraction-zero among available final cells"
            ),
            "localized_false_block_evidence": (
                "at least one final remapped cell is SLAM-pure-free or "
                "has center_collision_fraction <=0.25"
            ),
        },
        "A_spatial_correlation_with_live_interventions": correlation[
            "variants"
        ],
        "B_same_state_raw_action_feasibility": {
            variant: {
                "M0_pass": replay.report["summary"]["M0_CURRENT"][
                    "raw_safety_pass_count"
                ],
                "variant_pass": replay.report["summary"][variant][
                    "raw_safety_pass_count"
                ],
                "delta": mode_answers[variant][
                    "raw_safety_pass_delta_vs_M0"
                ],
            }
            for variant in VARIANTS[1:]
        },
        "C_improvement_distribution": {
            variant: {
                key: mode_answers[variant][key]
                for key in (
                    "improved_steps",
                    "regressed_steps",
                    "improvement_distribution",
                )
            }
            for variant in VARIANTS[1:]
        },
        "D_false_block_risk": {
            variant: {
                key: mode_answers[variant][key]
                for key in (
                    "low_center_collision_fraction_among_available",
                    "slam_zero_occupied_fraction_among_available",
                    "localized_false_block_evidence",
                    "population_level_false_block_risk_flag",
                    "false_block_risk_cells",
                )
            }
            for variant in VARIANTS[1:]
        },
        "E_production_candidate_design": {
            variant: mode_answers[variant][
                "production_candidate_design_judgement"
            ]
            for variant in VARIANTS[1:]
        },
        "mode_details": mode_answers,
        "deployment_recommendation": "NOT_EVALUATED_OFFLINE_ONLY",
        "interpretation_limit": (
            "Same-recorded-state counterfactual cannot establish "
            "closed-loop exploration or safety improvement."
        ),
    }


def _policy_mask_panel(
    baseline: DecisionModeArtifacts,
    remapped: set[tuple[int, int]],
    trajectory: Sequence[tuple[int, int]],
    title: str,
) -> Any:
    """Render one final policy-view mask over frozen belief semantics."""
    from PIL import Image, ImageDraw

    belief = np.asarray(baseline.cum_map.map)
    origin_row, origin_col = (
        int(value) for value in baseline.cum_map.origin_world_rc
    )
    scale = 14
    label_height = 68
    image = Image.new(
        "RGB",
        (belief.shape[1] * scale, belief.shape[0] * scale + label_height),
        (235, 235, 235),
    )
    draw = ImageDraw.Draw(image)
    observed = baseline.accumulator.ever_observed_cells()
    effective = _world_cells_from_mask(
        np.asarray(baseline.frontier.effective_frontier_u8) > 0,
        baseline.cum_map.origin_world_rc,
    )
    for array_row in range(belief.shape[0]):
        for array_col in range(belief.shape[1]):
            cell = (origin_row + array_row, origin_col + array_col)
            value = int(belief[array_row, array_col])
            if value == EMPTY:
                color = (250, 250, 250)
            elif value == OBSTACLE:
                color = (25, 25, 25)
            elif cell in observed:
                color = (235, 158, 55)
            else:
                color = (125, 125, 125)
            x0 = array_col * scale
            y0 = array_row * scale + label_height
            draw.rectangle(
                (x0, y0, x0 + scale - 1, y0 + scale - 1), fill=color
            )
            if cell in effective:
                draw.rectangle(
                    (x0 + 2, y0 + 2, x0 + scale - 3, y0 + scale - 3),
                    outline=(30, 155, 235),
                    width=2,
                )
            if cell in remapped:
                draw.rectangle(
                    (x0 + 3, y0 + 3, x0 + scale - 4, y0 + scale - 4),
                    fill=(225, 35, 125),
                )
    for row, col in trajectory:
        array_row = row - origin_row
        array_col = col - origin_col
        if (
            0 <= array_row < belief.shape[0]
            and 0 <= array_col < belief.shape[1]
        ):
            x0 = array_col * scale
            y0 = array_row * scale + label_height
            draw.ellipse(
                (x0 + 4, y0 + 4, x0 + scale - 5, y0 + scale - 5),
                fill=(0, 215, 120),
            )
    draw.text((6, 5), title, fill=(0, 0, 0))
    draw.text(
        (6, 23),
        "gray=never UNKNOWN orange=observed UNKNOWN pink=remap",
        fill=(0, 0, 0),
    )
    draw.text(
        (6, 41),
        "blue outline=effective frontier green=trajectory",
        fill=(0, 0, 0),
    )
    return image


def _combine_horizontal(panels: Sequence[Any]) -> Any:
    """Join equally meaningful labeled panels horizontally."""
    from PIL import Image

    image = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), max(p.height for p in panels)),
        (220, 220, 220),
    )
    offset = 0
    for panel in panels:
        image.paste(panel, (offset, 0))
        offset += panel.width
    return image


def _action_counterfactual_image(report: dict[str, Any]) -> Any:
    """Render the raw-action and raw-feasibility table as a PNG."""
    from PIL import Image, ImageDraw

    cell_width = 190
    row_height = 42
    left_width = 70
    header_height = 48
    image = Image.new(
        "RGB",
        (left_width + len(VARIANTS) * cell_width,
         header_height + len(report["steps"]) * row_height),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(image)
    draw.text((8, 15), "step", fill=(0, 0, 0))
    for column, variant in enumerate(VARIANTS):
        x0 = left_width + column * cell_width
        draw.rectangle(
            (x0, 0, x0 + cell_width - 1, header_height - 1),
            fill=(215, 225, 238),
            outline=(90, 90, 90),
        )
        draw.text((x0 + 5, 6), variant, fill=(0, 0, 0))
        draw.text((x0 + 5, 25), "raw action | proxy", fill=(0, 0, 0))
    for row_index, row in enumerate(report["steps"]):
        y0 = header_height + row_index * row_height
        draw.text((10, y0 + 13), str(row["step_id"]), fill=(0, 0, 0))
        for column, variant in enumerate(VARIANTS):
            item = row["variants"][variant]
            passed = item["raw_action_safety"]["safety_pass"]
            color = (190, 240, 202) if passed else (250, 188, 188)
            x0 = left_width + column * cell_width
            draw.rectangle(
                (x0, y0, x0 + cell_width - 1, y0 + row_height - 1),
                fill=color,
                outline=(100, 100, 100),
            )
            label = (
                f"{item['raw_action']} {item['raw_action_name']} | "
                f"{'PASS' if passed else 'BLOCK'}"
            )
            draw.text((x0 + 7, y0 + 5), label, fill=(0, 0, 0))
            draw.text(
                (x0 + 7, y0 + 22),
                f"margin={item['top1_top2_margin']:.4f}",
                fill=(0, 0, 0),
            )
    return image


def _traversability_panel(
    baseline: DecisionModeArtifacts,
    title: str,
    values: dict[tuple[int, int], Optional[float]],
    missing_color: tuple[int, int, int] = (105, 105, 105),
) -> Any:
    """Render a scalar candidate-cell diagnostic over the map extent."""
    from PIL import Image, ImageDraw

    belief = np.asarray(baseline.cum_map.map)
    scale = 14
    label_height = 55
    image = Image.new(
        "RGB",
        (belief.shape[1] * scale, belief.shape[0] * scale + label_height),
        (70, 70, 70),
    )
    draw = ImageDraw.Draw(image)
    origin_row, origin_col = (
        int(value) for value in baseline.cum_map.origin_world_rc
    )
    for cell, value in sorted(values.items()):
        array_row = cell[0] - origin_row
        array_col = cell[1] - origin_col
        if not (
            0 <= array_row < belief.shape[0]
            and 0 <= array_col < belief.shape[1]
        ):
            continue
        if value is None:
            color = missing_color
        else:
            bounded = min(1.0, max(0.0, float(value)))
            color = (
                int(round(255 * bounded)),
                int(round(210 * (1.0 - bounded))),
                int(round(255 * (1.0 - bounded))),
            )
        x0 = array_col * scale
        y0 = array_row * scale + label_height
        draw.rectangle(
            (x0, y0, x0 + scale - 1, y0 + scale - 1),
            fill=color,
            outline=(230, 230, 230),
        )
    draw.text((6, 5), title, fill=(255, 255, 255))
    draw.text((6, 24), "cyan=0 red=1 gray=N/A", fill=(255, 255, 255))
    return image


def export_visualizations(
    output_dir: Path,
    baseline: DecisionModeArtifacts,
    replay: PolicyReplayArtifacts,
    episode: dict[str, Any],
    traversability: dict[str, Any],
    slam: dict[str, Any],
) -> list[str]:
    """Write the three required offline PNG visualizations."""
    trajectory = [
        (int(step["agent_state"][0]), int(step["agent_state"][1]))
        for step in episode["steps"]
    ]
    mask_path = output_dir / "mixed_policy_view_masks.png"
    action_path = output_dir / "policy_action_counterfactual.png"
    traversability_path = output_dir / "mixed_cell_traversability.png"
    _combine_horizontal(
        [
            _policy_mask_panel(
                baseline,
                replay.final_masks[variant],
                trajectory,
                f"{variant}: final-decision view",
            )
            for variant in VARIANTS
        ]
    ).save(mask_path, format="PNG")
    _action_counterfactual_image(replay.report).save(
        action_path, format="PNG"
    )
    center_values = {
        (row["row"], row["col"]): row["center_collision_fraction"]
        for row in traversability["cells"]
    }
    slam_values = {
        (row["row"], row["col"]): row["slam_occupied_pixel_fraction"]
        for row in slam.get("cells", [])
    }
    persistent_values = {
        cell: 1.0
        for variant in VARIANTS[1:]
        for cell in replay.union_masks[variant]
    }
    _combine_horizontal(
        [
            _traversability_panel(
                baseline, "persistent mixed (ever)", persistent_values
            ),
            _traversability_panel(
                baseline, "center collision fraction", center_values
            ),
            _traversability_panel(
                baseline, "SLAM occupied fraction", slam_values
            ),
        ]
    ).save(traversability_path, format="PNG")
    return [str(mask_path), str(action_path), str(traversability_path)]


def export_cells_csv(
    output_path: Path,
    cells: Sequence[tuple[int, int]],
    baseline: DecisionModeArtifacts,
    allscan_counts: dict[tuple[int, int], Counter],
    replay: PolicyReplayArtifacts,
    traversability: dict[str, Any],
    slam: dict[str, Any],
) -> None:
    """Write one deterministic row per relevant mixed-policy cell."""
    center_rows = {
        (row["row"], row["col"]): row
        for row in traversability["cells"]
    }
    slam_rows = {
        (row["row"], row["col"]): row for row in slam.get("cells", [])
    }
    unresolved = set(
        _unresolved_cells(baseline.cum_map, baseline.accumulator)
    )
    remapped_steps: dict[str, dict[tuple[int, int], list[int]]] = {
        variant: defaultdict(list) for variant in VARIANTS[1:]
    }
    for row in replay.report["steps"]:
        for variant in VARIANTS[1:]:
            for cell_row, cell_col in row["variants"][variant][
                "remapped_world_cells"
            ]:
                remapped_steps[variant][(cell_row, cell_col)].append(
                    int(row["step_id"])
                )
    fieldnames = [
        "row",
        "col",
        "baseline_value",
        "baseline_final_observed_unclassified",
        "high_conflict_focus",
        "decision_free",
        "decision_obstacle",
        "decision_conflict",
        "allscan_observed",
        "allscan_free_only",
        "allscan_obstacle_only",
        "allscan_conflict",
        "allscan_conflict_persistence",
        "center_observed_scan_count",
        "median_center_clearance_m",
        "p05_center_clearance_m",
        "minimum_center_clearance_m",
        "center_collision_fraction",
        "slam_free_pixel_fraction",
        "slam_occupied_pixel_fraction",
        "slam_unknown_pixel_fraction",
    ]
    for variant in VARIANTS[1:]:
        fieldnames.extend(
            (
                f"{variant}_remapped_final",
                f"{variant}_remapped_ever",
                f"{variant}_remapped_steps",
            )
        )
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for cell in sorted(set(cells)):
            state = baseline.accumulator.cells.get(cell)
            counts = allscan_counts[cell]
            observed = int(counts["observed"])
            center = center_rows.get(cell, {})
            slam_row = slam_rows.get(cell, {})
            row: dict[str, Any] = {
                "row": cell[0],
                "col": cell[1],
                "baseline_value": _world_value(baseline.cum_map, cell),
                "baseline_final_observed_unclassified": cell in unresolved,
                "high_conflict_focus": cell in HIGH_CONFLICT_CELLS,
                "decision_free": (
                    int(state.free_frame_count) if state else 0
                ),
                "decision_obstacle": (
                    int(state.obstacle_frame_count) if state else 0
                ),
                "decision_conflict": (
                    int(state.conflict_frame_count) if state else 0
                ),
                "allscan_observed": observed,
                "allscan_free_only": int(counts["free_only"]),
                "allscan_obstacle_only": int(counts["obstacle_only"]),
                "allscan_conflict": int(counts["conflict"]),
                "allscan_conflict_persistence": (
                    int(counts["conflict"]) / observed if observed else ""
                ),
                "center_observed_scan_count": center.get(
                    "observed_scan_count", ""
                ),
                "median_center_clearance_m": center.get(
                    "median_center_clearance_m", ""
                ),
                "p05_center_clearance_m": center.get(
                    "p05_center_clearance_m", ""
                ),
                "minimum_center_clearance_m": center.get(
                    "minimum_center_clearance_m", ""
                ),
                "center_collision_fraction": center.get(
                    "center_collision_fraction", ""
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
            for variant in VARIANTS[1:]:
                row[f"{variant}_remapped_final"] = (
                    cell in replay.final_masks[variant]
                )
                row[f"{variant}_remapped_ever"] = (
                    cell in replay.union_masks[variant]
                )
                row[f"{variant}_remapped_steps"] = json.dumps(
                    remapped_steps[variant].get(cell, [])
                )
            writer.writerow(row)


def _comparison_table(replay: PolicyReplayArtifacts) -> dict[str, Any]:
    """Build the required machine-readable four-mode final table."""
    return {
        variant: {
            "mixed_cells_remapped_final": len(replay.final_masks[variant]),
            "mixed_cells_remapped_union": len(replay.union_masks[variant]),
            "raw_action_changes_vs_M0": replay.report["summary"][variant][
                "action_change_count_vs_M0"
            ],
            "raw_safety_pass": replay.report["summary"][variant][
                "raw_safety_pass_count"
            ],
            "raw_blocked": replay.report["summary"][variant][
                "raw_safety_blocked_count"
            ],
            "proxy_intervention_rate": replay.report["summary"][variant][
                "intervention_rate"
            ],
            "effective_frontier_mismatch": 0,
            "categorical_belief_mismatch": 0,
        }
        for variant in VARIANTS
    }


def _single_path(paths: Sequence[Path], description: str) -> Path:
    """Require one unambiguous automatically located input."""
    if len(paths) != 1:
        raise ValueError(
            f"expected one {description}, found {len(paths)}: {paths}"
        )
    return paths[0]


def _resolve_checkpoint(
    requested: Path, fallback: Path
) -> tuple[Path, dict[str, Any]]:
    """Resolve the authentic checkpoint without copying or modifying it."""
    requested = requested.expanduser()
    fallback = fallback.expanduser()
    if requested.is_file():
        selected = requested.resolve()
        source = "REQUESTED_PATH"
    elif fallback.is_file():
        selected = fallback.resolve()
        source = "READ_ONLY_IDENTICAL_NAMED_FALLBACK"
    else:
        raise FileNotFoundError(
            f"checkpoint absent at requested and fallback paths: "
            f"{requested}, {fallback}"
        )
    actual_sha = _sha256_file(selected)
    if actual_sha != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA256 does not match frozen identity")
    return selected, {
        "requested_path": str(requested),
        "requested_path_exists": requested.is_file(),
        "selected_read_only_path": str(selected),
        "selection_reason": source,
        "sha256": actual_sha,
        "expected_sha256": CHECKPOINT_SHA256,
        "checkpoint_modified": False,
        "checkpoint_copied_or_linked": False,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the strictly offline study interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, default=REQUESTED_CHECKPOINT
    )
    parser.add_argument(
        "--checkpoint-fallback",
        type=Path,
        default=DISCOVERED_READ_ONLY_CHECKPOINT,
    )
    parser.add_argument(
        "--drl-repository",
        type=Path,
        default=REPOSITORY_ROOT.parent / "DRL-path-finding",
    )
    return parser


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    """Run gates, fixed counterfactuals, and deterministic exports."""
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_root / "mixed_policy_view_study"
    )
    episode_path = _single_path(
        sorted((dataset_root / "drl").glob("*.json")),
        "live episode JSON",
    )
    bag_path = dataset_root / "rosbag"
    if not (bag_path / "metadata.yaml").is_file():
        raise FileNotFoundError(f"missing rosbag metadata: {bag_path}")
    manifest = verify_internal_manifest(dataset_root)
    if manifest["status"] != "PASS":
        raise RuntimeError("internal dataset SHA256 manifest failed")
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    if episode.get("git_commit") != CATEGORICAL_BASE_SHA:
        raise ValueError("episode production git_commit is not frozen")
    expected_semantics = (
        episode.get("fusion_mode") == "evidence"
        and episode.get("fusion_config", {}).get("name") == "candidate_a"
        and episode.get("coarse_occlusion_mode") == "confirmed_opaque"
        and episode.get("frontier_semantics_mode") == "evidence_aware"
    )
    if not expected_semantics:
        raise ValueError("episode does not use the frozen policy semantics")
    checkpoint, checkpoint_identity = _resolve_checkpoint(
        args.checkpoint, args.checkpoint_fallback
    )
    drl_repository = args.drl_repository.expanduser().resolve()
    cumulative_map_type = _load_cumulative_map_type(drl_repository)
    contents = read_bag_contents(bag_path)
    decisions = read_decisions(episode)
    matches = match_decision_scans(decisions, contents.scans)
    geometry = _projection_geometry(episode, matches)
    baseline = run_decision_mode(
        "M0_CATEGORICAL_BASELINE",
        1,
        matches,
        contents.scans,
        contents.odom,
        episode,
        cumulative_map_type,
    )
    categorical_gate = baseline_gate(
        baseline, episode, episode_path, matches
    )
    if categorical_gate["status"] != "PASS":
        raise RuntimeError("CATEGORICAL_BASELINE_GATE=FAIL")
    replay = replay_policy_views(
        episode,
        matches,
        geometry,
        cumulative_map_type,
        checkpoint,
        contents.scans,
    )
    first_time = matches[0].decision.scan_timestamp_sec
    last_time = matches[-1].decision.scan_timestamp_sec
    episode_scans = [
        scan
        for scan in contents.scans
        if first_time <= scan.timestamp_sec <= last_time
    ]
    allscan_report, allscan_records, allscan_counts = run_allscan_density(
        episode_scans,
        contents.odom,
        matches,
        baseline,
        geometry,
    )
    baseline_unresolved = set(
        _unresolved_cells(baseline.cum_map, baseline.accumulator)
    )
    relevant_cells = set(baseline_unresolved)
    for variant in VARIANTS[1:]:
        relevant_cells.update(replay.union_masks[variant])
    traversability = center_traversability_audit(
        sorted(relevant_cells), allscan_records, geometry
    )
    for row in traversability["cells"]:
        row["baseline_final_observed_unclassified"] = (
            (row["row"], row["col"]) in baseline_unresolved
        )
    slam = evaluate_slam_composition(
        contents,
        baseline,
        geometry,
        evaluation_cells=sorted(relevant_cells),
    )
    false_block = false_block_analysis(replay, slam, traversability)
    correlation = spatial_correlation(replay.report)
    focus = high_conflict_focus(
        baseline,
        allscan_counts,
        slam,
        traversability,
        replay,
    )
    interpretation = build_interpretation(
        replay, correlation, false_block
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mixed_policy_view_cells.csv"
    export_cells_csv(
        csv_path,
        sorted(relevant_cells),
        baseline,
        allscan_counts,
        replay,
        traversability,
        slam,
    )
    visualizations = export_visualizations(
        output_dir,
        baseline,
        replay,
        episode,
        traversability,
        slam,
    )
    report = {
        "dataset_identity": {
            "dataset_root": str(dataset_root),
            "episode_json": str(episode_path),
            "episode_json_sha256": _sha256_file(episode_path),
            "rosbag_db3_sha256": _sha256_file(bag_path / "rosbag_0.db3"),
            "internal_manifest": manifest,
        },
        "frozen_reference": {
            "analysis_base_sha": BASE_SHA,
            "production_episode_sha": CATEGORICAL_BASE_SHA,
            "branch": "realcar-mixed-policy-view-study",
            "offline_analysis_only": True,
            "production_runtime_modified": False,
            "policy_input_categories": [-1, 0, 1],
            "only_policy_categorical_copy_changed": True,
        },
        "checkpoint_identity": checkpoint_identity,
        "categorical_baseline_gate": categorical_gate,
        "M0_model_action_gate": replay.report["M0_model_action_gate"],
        "policy_counterfactual": replay.report,
        "comparison_table": _comparison_table(replay),
        "spatial_correlation": correlation,
        "allscan_density": {
            key: allscan_report[key]
            for key in (
                "episode_duration_sec",
                "total_scans_processed",
                "skipped_no_valid_beams",
                "total_observed_cells",
                "conflict_ge_9_persistence_summary",
            )
        },
        "center_traversability": traversability,
        "high_conflict_focus": focus,
        "slam_secondary_validation": slam,
        "false_block_risk": false_block,
        "interpretation": interpretation,
        "immutability": {
            "BELIEF_IMMUTABILITY": "PASS",
            "ACCUMULATOR_IMMUTABILITY": "PASS",
            "FRONTIER_IMMUTABILITY": "PASS",
            "effective_frontier_mismatch_all_modes_all_steps": 0,
            "categorical_belief_mismatch_all_modes_all_steps": 0,
        },
        "determinism": {
            "stable_sorting": True,
            "model_state_repeat_hash_match_all_steps": True,
            "report_contains_wall_clock_time": False,
        },
        "output_files": {
            "json": str(output_dir / "mixed_policy_view_study.json"),
            "csv": str(csv_path),
            "visualizations": visualizations,
        },
    }
    json_path = output_dir / "mixed_policy_view_study.json"
    serialized = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=_json_default,
    ) + "\n"
    json_path.write_text(serialized, encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the study and print a compact machine-readable result."""
    args = build_argument_parser().parse_args(argv)
    try:
        report = run_study(args)
    except Exception as exc:
        print(f"mixed policy view study failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "categorical_baseline_gate": report[
                    "categorical_baseline_gate"
                ]["status"],
                "M0_model_action_gate": report[
                    "M0_model_action_gate"
                ],
                "comparison_table": report["comparison_table"],
                "interpretation": report["interpretation"][
                    "E_production_candidate_design"
                ],
                "output_files": report["output_files"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
