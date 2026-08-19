"""Tests for the offline-only mixed policy categorical view study."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_SOURCE = REPOSITORY_ROOT / "ros2_ws" / "src" / "drl_explore_bridge"
for source_path in (REPOSITORY_ROOT, PACKAGE_SOURCE):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from drl_explore_bridge.realcar_conservative_belief import (  # noqa: E402
    BeliefEvidenceAccumulator,
    EMPTY,
    INVISIBLE,
    OBSTACLE,
    frontier_semantics_snapshot,
    named_fusion_config,
)
from scripts_realcar import analyze_mixed_policy_view as study  # noqa: E402


class FakeCumulativeMap:
    """Small map exposing the read contract used by policy views."""

    def __init__(self):
        self.map = np.full((5, 5), INVISIBLE, dtype=np.int8)
        self.visit_count = np.zeros((5, 5), dtype=np.int32)
        self.origin_world_rc = (0, 0)
        self.frontier_u8 = np.zeros((5, 5), dtype=np.uint8)
        self.frontier_u8[0, 1] = 255

    def get_frontier_u8(self, refresh=False):
        _ = refresh
        return self.frontier_u8


def _fixtures():
    cum_map = FakeCumulativeMap()
    accumulator = BeliefEvidenceAccumulator(
        named_fusion_config("candidate_a")
    )
    only_two = accumulator.state_for((1, 1))
    only_two.conflict_frame_count = 2
    dominant_three = accumulator.state_for((1, 2))
    dominant_three.conflict_frame_count = 3
    dominant_three.free_frame_count = 1
    dominant_two = accumulator.state_for((1, 3))
    dominant_two.conflict_frame_count = 2
    dominant_two.free_frame_count = 1
    tied = accumulator.state_for((2, 1))
    tied.conflict_frame_count = 2
    tied.free_frame_count = 1
    tied.obstacle_frame_count = 1
    return cum_map, accumulator


def test_m0_leaves_view_unchanged():
    cum_map, accumulator = _fixtures()

    policy_map, cells = study.build_policy_mixed_map(
        cum_map, accumulator, "M0_CURRENT"
    )

    assert cells == frozenset()
    assert np.array_equal(policy_map, cum_map.map)
    assert policy_map is not cum_map.map


def test_m1_exact_rule():
    cum_map, accumulator = _fixtures()

    cells = study.persistent_mixed_world_cells(
        cum_map, accumulator, "M1_CONFLICT_ONLY_2"
    )

    assert cells == frozenset({(1, 1)})


def test_m2_exact_rule():
    cum_map, accumulator = _fixtures()

    cells = study.persistent_mixed_world_cells(
        cum_map, accumulator, "M2_CONFLICT_DOMINANT_2"
    )

    assert cells == frozenset({(1, 1), (1, 2), (1, 3)})


def test_m3_exact_rule():
    cum_map, accumulator = _fixtures()

    cells = study.persistent_mixed_world_cells(
        cum_map, accumulator, "M3_CONFLICT_DOMINANT_3"
    )

    assert cells == frozenset({(1, 2)})


def test_visited_cell_is_never_remapped():
    cum_map, accumulator = _fixtures()
    cum_map.visit_count[1, 1] = 1

    cells = study.persistent_mixed_world_cells(
        cum_map, accumulator, "M1_CONFLICT_ONLY_2"
    )

    assert (1, 1) not in cells


def test_known_free_is_never_remapped():
    cum_map, accumulator = _fixtures()
    cum_map.map[1, 1] = EMPTY

    policy_map, cells = study.build_policy_mixed_map(
        cum_map, accumulator, "M1_CONFLICT_ONLY_2"
    )

    assert (1, 1) not in cells
    assert policy_map[1, 1] == EMPTY


def test_known_obstacle_stays_obstacle():
    cum_map, accumulator = _fixtures()
    cum_map.map[4, 4] = OBSTACLE

    policy_map, _cells = study.build_policy_mixed_map(
        cum_map, accumulator, "M2_CONFLICT_DOMINANT_2"
    )

    assert policy_map[4, 4] == OBSTACLE


def test_policy_map_never_introduces_fourth_category():
    cum_map, accumulator = _fixtures()

    policy_map, _cells = study.build_policy_mixed_map(
        cum_map, accumulator, "M2_CONFLICT_DOMINANT_2"
    )

    assert set(np.unique(policy_map)).issubset({-1, 0, 1})


def test_cumulative_map_is_immutable():
    cum_map, accumulator = _fixtures()
    before = np.array(cum_map.map, copy=True)

    study.build_policy_mixed_map(
        cum_map, accumulator, "M2_CONFLICT_DOMINANT_2"
    )

    assert np.array_equal(cum_map.map, before)


def test_accumulator_is_immutable():
    cum_map, accumulator = _fixtures()
    before = study._accumulator_snapshot(accumulator)

    study.build_policy_mixed_map(
        cum_map, accumulator, "M2_CONFLICT_DOMINANT_2"
    )

    assert study._accumulator_snapshot(accumulator) == before


def test_frontier_is_bit_exact_in_policy_view():
    cum_map, accumulator = _fixtures()
    snapshot = frontier_semantics_snapshot(
        cum_map, accumulator, "evidence_aware"
    )
    policy_map, _cells = study.build_policy_mixed_map(
        cum_map, accumulator, "M2_CONFLICT_DOMINANT_2"
    )

    view = study.PolicyMixedBeliefView(
        cum_map, policy_map, snapshot.effective_frontier_u8
    )

    assert np.array_equal(
        view.get_frontier_u8(), snapshot.effective_frontier_u8
    )
    assert np.array_equal(cum_map.frontier_u8, snapshot.raw_frontier_u8)


def test_model_state_hash_is_deterministic():
    torch = pytest.importorskip("torch")
    state = {
        "beta": torch.tensor([[3.0, 4.0]]),
        "alpha": torch.tensor([[1, 2]], dtype=torch.int64),
    }

    first = study._state_batch_sha256(state)
    second = study._state_batch_sha256(dict(reversed(list(state.items()))))

    assert first == second


DATASET_ROOT = Path(
    "/home/robot/robot_data/evidenceaware10_20260819_175106"
)


@pytest.mark.skipif(
    not DATASET_ROOT.is_dir()
    or not study.DISCOVERED_READ_ONLY_CHECKPOINT.is_file(),
    reason="frozen local dataset/checkpoint unavailable",
)
def test_baseline_m0_action_reproduction():
    episode_path = Path(
        glob.glob(str(DATASET_ROOT / "drl" / "*.json"))[0]
    )
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    contents = study.read_bag_contents(DATASET_ROOT / "rosbag")
    matches = study.match_decision_scans(
        study.read_decisions(episode), contents.scans
    )
    geometry = study._projection_geometry(episode, matches)
    cumulative_map_type = study._load_cumulative_map_type(
        REPOSITORY_ROOT.parent / "DRL-path-finding"
    )
    model, adapter, torch = study.load_policy_model(
        str(study.DISCOVERED_READ_ONLY_CHECKPOINT)
    )
    proxies = study.select_pre_motion_safety_proxies(
        episode, contents.scans
    )

    replay = study._replay_policy_once(
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

    assert [
        row["variants"]["M0_CURRENT"]["raw_action"]
        for row in replay.report["steps"]
    ] == [step["raw_policy_action"] for step in episode["steps"]]
