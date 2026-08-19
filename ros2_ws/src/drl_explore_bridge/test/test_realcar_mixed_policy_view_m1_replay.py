"""Dataset-backed parity gate for the production M1 policy view."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from drl_explore_bridge.realcar_conservative_belief import (  # noqa: E402
    BeliefEvidenceAccumulator,
    apply_evidence_fusion,
    cumulative_occlusion_cells,
    frontier_semantics_snapshot,
    named_fusion_config,
    project_scan_to_belief,
    record_traversed_cells_as_free,
    visited_only_local_snap,
)
from drl_explore_bridge.realcar_policy_continuous_runner_node import (  # noqa: E402
    build_policy_state_with_frontier_semantics,
    persistent_mixed_unknown_cells,
)
from drl_explore_bridge.realcar_policy_safe_runner_node import (  # noqa: E402
    ACTION_NAMES,
    load_policy_model,
)
from scripts_realcar import analyze_belief_fusion_replay as replay  # noqa: E402


DATASET_ROOT = Path(
    "/home/robot/robot_data/evidenceaware10_20260819_175106"
)
DRL_REPOSITORY = Path("/home/robot/robot_repos/DRL-path-finding")
CHECKPOINT = Path(
    "/mnt/c/Users/Dk/Desktop/SCI/New_A/checkpoint_store/"
    "full_method_main/A_full_method_last.pt"
)
CHECKPOINT_SHA256 = (
    "7d70d54e8e7ad95d623dea91e2f6a7ddd7499acf5e414a5617ba799280b0c3ad"
)
EXPECTED_M0_ACTIONS = (6, 0, 6, 6, 2, 2, 3, 3, 1, 2)
EXPECTED_M1_ACTIONS = (6, 4, 6, 7, 2, 2, 3, 3, 2, 2)
EXPECTED_M1_CELLS = (
    (),
    (
        (58, 59), (58, 60), (58, 61), (58, 62),
        (59, 59), (59, 62), (59, 63), (59, 64),
        (60, 59), (60, 66), (61, 60), (61, 61),
        (61, 62), (61, 63),
    ),
    (
        (57, 61), (58, 59), (58, 61), (58, 62),
        (59, 59), (59, 62), (59, 63), (59, 64),
        (60, 59), (60, 66), (61, 60), (61, 61),
        (61, 62), (61, 63),
    ),
    (
        (57, 61), (58, 59), (58, 61), (58, 62),
        (59, 59), (59, 62), (59, 63), (59, 64),
        (60, 59), (60, 66), (61, 60), (61, 61),
        (61, 62), (61, 63),
    ),
    (
        (58, 59), (58, 61), (59, 59), (59, 62),
        (59, 63), (59, 64), (60, 59), (60, 66),
        (61, 60), (61, 61), (61, 62), (61, 63),
    ),
    (
        (58, 59), (58, 61), (59, 59), (59, 62),
        (59, 63), (59, 64), (60, 59), (60, 66),
        (61, 60), (61, 61),
    ),
    (
        (58, 59), (58, 61), (59, 59), (59, 62),
        (59, 63), (59, 64), (60, 59), (60, 66),
        (61, 60), (61, 61), (61, 71),
    ),
    (
        (58, 59), (58, 61), (59, 59), (59, 62),
        (59, 63), (59, 64), (60, 59), (60, 66),
        (61, 60), (61, 61),
    ),
    (
        (59, 59), (59, 62), (59, 63), (59, 64),
        (60, 59), (60, 66), (61, 60), (61, 61),
    ),
    (
        (59, 59), (59, 62), (59, 63), (59, 64),
        (60, 66), (61, 60), (61, 61),
    ),
)
EXPECTED_M1_Q = (
    (39.0639, 39.1635, 38.7222, 39.2224,
     39.2134, 40.1996, 40.4529, 39.8539),
    (2.1453, -7.3473, 2.9915, -2.2427,
     13.4039, 2.4184, 2.4377, -4.3460),
    (-15.0273, -3.1619, 3.4225, -0.4641,
     -9.6225, -3.5515, 9.0995, -3.7266),
    (11.6477, -4.4634, 11.7054, 0.1386,
     -7.8795, -4.5955, 7.4905, 22.2864),
    (-2.0125, 2.0829, 7.9821, -5.2322,
     -12.7990, -4.0847, -1.0765, -1.7436),
    (4.0957, -15.6228, 26.3616, -0.9259,
     -3.5420, -14.7615, -2.4271, 4.3011),
    (-13.4671, -20.1678, 24.7491, 29.1427,
     -17.1483, -2.7928, -3.1263, -2.7111),
    (-7.0638, 3.9422, 27.7224, 41.6425,
     36.8442, -0.6556, 10.6929, -0.2002),
    (-8.7632, 24.1892, 36.6476, 33.7031,
     18.3879, 12.3674, 6.9929, -8.4625),
    (-9.1426, -10.4609, 14.8346, -4.1030,
     9.4989, 8.3009, -10.4747, -21.0459),
)


def _sha256(path):
    """Return the SHA-256 digest for a frozen replay dependency."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _accumulator_snapshot(accumulator):
    """Return an immutable evidence-accumulator snapshot."""
    return tuple(
        (
            cell,
            state.free_frame_count,
            state.obstacle_frame_count,
            state.conflict_frame_count,
            state.consecutive_free_frames,
            state.consecutive_obstacle_frames,
        )
        for cell, state in sorted(accumulator.cells.items())
    )


def _episode_path():
    """Resolve the single frozen production episode manifest."""
    paths = sorted((DATASET_ROOT / "drl").glob("*.json"))
    assert len(paths) == 1
    return paths[0]


@pytest.mark.skipif(
    not DATASET_ROOT.is_dir()
    or not CHECKPOINT.is_file()
    or not DRL_REPOSITORY.is_dir(),
    reason="frozen live10 dataset, checkpoint, or DRL reference unavailable",
)
def test_live10_off_and_m1_analysis_production_parity():
    """Require exact M0 live and M1 offline actions/cells on live10."""
    assert _sha256(CHECKPOINT) == CHECKPOINT_SHA256
    episode = json.loads(_episode_path().read_text(encoding="utf-8"))
    scans = replay.read_scans_from_bag(DATASET_ROOT / "rosbag", "/scan")
    scan_times = [scan.timestamp_sec for scan in scans]
    cumulative_map_type = replay._load_cumulative_map_type(DRL_REPOSITORY)
    model, adapter, torch = load_policy_model(str(CHECKPOINT))
    config = named_fusion_config("candidate_a")
    accumulator = BeliefEvidenceAccumulator(config)
    compatibility_grid = np.zeros((120, 120), dtype=np.int8)
    origin_state = tuple(int(value) for value in episode["origin_state"])
    origin_pose = episode["steps"][0]["observation_pose"]
    origin_x = float(origin_pose["x"])
    origin_y = float(origin_pose["y"])
    cell_size = float(episode["cell_size"])
    cum_map = None
    m0_actions = []
    m1_actions = []
    m1_q = []
    mixed_history = []
    known_history = []
    raw_frontier_history = []
    effective_frontier_history = []
    observed_unclassified_history = []
    evidence_totals = Counter()
    occlusion_totals = Counter()
    transition_totals = Counter()

    for index, step in enumerate(episode["steps"]):
        scan_timestamp = float(step["observation_scan_timestamp"])
        scan_index = min(
            range(len(scans)),
            key=lambda item: abs(scan_times[item] - scan_timestamp),
        )
        assert abs(scan_times[scan_index] - scan_timestamp) == 0.0
        scan = scans[scan_index].message
        pose = step["observation_pose"]
        agent_state = tuple(int(value) for value in step["agent_state"])
        if cum_map is None:
            blockers = frozenset()
            visited = frozenset()
        else:
            blockers, visited = cumulative_occlusion_cells(cum_map)
        observation = project_scan_to_belief(
            scan,
            robot_x=float(pose["x"]),
            robot_y=float(pose["y"]),
            robot_yaw=float(pose["yaw_rad"]),
            origin_x=origin_x,
            origin_y=origin_y,
            origin_state=origin_state,
            agent_state=agent_state,
            cell_size=cell_size,
            scan_radius_cells=int(episode["scan_radius_cells"]),
            laser_x_in_base=float(episode["laser_x_in_base_m"]),
            laser_y_in_base=float(episode["laser_y_in_base_m"]),
            laser_yaw_in_base=float(episode["laser_yaw_in_base"]),
            coarse_occlusion_mode="confirmed_opaque",
            historical_obstacle_cells=blockers,
            occlusion_exempt_cells=visited,
        )
        core_snap = visited_only_local_snap(
            tuple(observation.local_snap.shape)
        )
        if cum_map is None:
            cum_map = cumulative_map_type(
                compatibility_grid, agent_state, core_snap
            )
        else:
            cum_map.update(agent_state, core_snap)
        stats = apply_evidence_fusion(cum_map, accumulator, observation)
        frontier = frontier_semantics_snapshot(
            cum_map, accumulator, "evidence_aware"
        )
        cells = persistent_mixed_unknown_cells(
            cum_map, accumulator, "conflict_only_2"
        )
        mixed_history.append(tuple(sorted(cells)))
        map_before = np.array(cum_map.map, copy=True)
        visit_before = np.array(cum_map.visit_count, copy=True)
        raw_before = np.array(frontier.raw_frontier_u8, copy=True)
        effective_before = np.array(
            frontier.effective_frontier_u8, copy=True
        )
        accumulator_before = _accumulator_snapshot(accumulator)
        recent = [
            tuple(int(value) for value in item["agent_state"])
            for item in episode["steps"][: index + 1]
        ][-8:]
        state_off, _ = build_policy_state_with_frontier_semantics(
            adapter,
            cum_map,
            agent_state,
            frontier,
            "evidence_aware",
            recent,
            "off",
            accumulator,
        )
        state_m1, _ = build_policy_state_with_frontier_semantics(
            adapter,
            cum_map,
            agent_state,
            frontier,
            "evidence_aware",
            recent,
            "conflict_only_2",
            accumulator,
        )
        with torch.inference_mode():
            q_off = model(**state_off, return_aux=False)
            q_m1 = model(**state_m1, return_aux=False)
        m0_actions.append(int(torch.argmax(q_off, dim=1).item()))
        m1_actions.append(int(torch.argmax(q_m1, dim=1).item()))
        m1_q.append(
            tuple(
                round(float(value), 4)
                for value in q_m1.detach().cpu().numpy()[0].tolist()
            )
        )
        live_q = tuple(float(value) for value in step["q_values"])
        replayed_q = tuple(
            round(float(value), 4)
            for value in q_off.detach().cpu().numpy()[0].tolist()
        )
        assert np.max(np.abs(np.subtract(replayed_q, live_q))) <= 1.01e-4
        assert np.array_equal(cum_map.map, map_before)
        assert np.array_equal(cum_map.visit_count, visit_before)
        assert np.array_equal(frontier.raw_frontier_u8, raw_before)
        assert np.array_equal(
            frontier.effective_frontier_u8, effective_before
        )
        assert _accumulator_snapshot(accumulator) == accumulator_before
        known_history.append(int(np.count_nonzero(cum_map.map != -1)))
        raw_frontier_history.append(frontier.raw_frontier_count)
        effective_frontier_history.append(frontier.effective_frontier_count)
        observed_unclassified_history.append(
            frontier.observed_unclassified_unknown_count
        )
        evidence_totals["free"] += stats.evidence_free_cells_this_step
        evidence_totals["obstacle"] += (
            stats.evidence_obstacle_cells_this_step
        )
        evidence_totals["conflict"] += (
            stats.evidence_conflict_cells_this_step
        )
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
        transition_totals["obstacle_to_free"] += (
            stats.obstacle_to_free_transitions_this_step
        )

    trajectory = [
        tuple(int(value) for value in step["agent_state"])
        for step in episode["steps"]
    ]
    correction = record_traversed_cells_as_free(cum_map, trajectory)
    transition_totals["obstacle_to_free"] += correction.corrected_from_obstacle
    final_counts = {
        "free": int(np.count_nonzero(cum_map.map == 0)),
        "obstacle": int(np.count_nonzero(cum_map.map == 1)),
        "unknown": int(np.count_nonzero(cum_map.map == -1)),
    }

    assert tuple(m0_actions) == EXPECTED_M0_ACTIONS
    assert tuple(m1_actions) == EXPECTED_M1_ACTIONS
    assert tuple(ACTION_NAMES[action] for action in m1_actions) == (
        "W", "S", "W", "NW", "E", "E", "SE", "SE", "E", "E"
    )
    assert tuple(mixed_history) == EXPECTED_M1_CELLS
    assert np.max(np.abs(np.subtract(m1_q, EXPECTED_M1_Q))) <= 1.01e-4
    assert final_counts == {"free": 31, "obstacle": 24, "unknown": 727}
    assert known_history == [1, 15, 18, 21, 24, 35, 44, 45, 49, 55]
    assert raw_frontier_history == [0, 12, 12, 15, 17, 22, 20, 21, 21, 20]
    assert effective_frontier_history == [0, 4, 3, 3, 4, 4, 5, 5, 6, 6]
    assert observed_unclassified_history == [
        0, 36, 38, 37, 38, 28, 20, 22, 21, 19
    ]
    assert evidence_totals == {"free": 405, "obstacle": 304, "conflict": 216}
    assert occlusion_totals == {
        "blocker": 45,
        "suppressed_free": 0,
        "suppressed_obstacle": 6,
        "suppressed_unique": 6,
    }
    assert transition_totals == {"obstacle_to_free": 0}
