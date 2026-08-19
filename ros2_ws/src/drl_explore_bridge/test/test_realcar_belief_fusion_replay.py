"""Unit tests for offline real-car belief fusion replay matching."""

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[4]
    / "scripts_realcar"
    / "analyze_belief_fusion_replay.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_belief_fusion_replay",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


def decision(step_id, timestamp):
    """Build one minimal canonical decision pose."""
    return replay.DecisionObservation(
        step_id=step_id,
        timestamp_sec=timestamp,
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        agent_state=(60, 60),
    )


def scan(timestamp):
    """Build one timestamp-only recorded scan."""
    return replay.RecordedScan(
        timestamp_sec=timestamp,
        bag_timestamp_sec=timestamp,
        timestamp_source="header",
        message=object(),
    )


def test_episode_decisions_prefer_observation_pose_timestamp():
    """Odom pose time remains canonical even when a scan time is recorded."""
    episode = {
        "steps": [
            {
                "step_id": 3,
                "agent_state": [60, 61],
                "observation_scan_timestamp": 99.0,
                "observation_pose": {
                    "x": 1.0,
                    "y": 2.0,
                    "yaw_rad": 0.25,
                    "odom_timestamp": 100.0,
                },
            }
        ]
    }

    decisions = replay.read_episode_decisions(episode)

    assert decisions[0].timestamp_sec == 100.0
    assert decisions[0].agent_state == (60, 61)


def test_nearest_scan_match_is_bounded_and_reported():
    """A unique nearest scan inside tolerance produces delta statistics."""
    matches, statistics = replay.match_decisions_to_scans(
        [decision(0, 10.0), decision(1, 20.0)],
        [scan(9.98), scan(10.08), scan(20.03)],
        tolerance_sec=0.05,
    )

    assert [match.scan.timestamp_sec for match in matches] == [9.98, 20.03]
    assert statistics["matched_decision_count"] == 2
    assert statistics["maximum_absolute_delta_sec"] == pytest.approx(0.03)


def test_out_of_tolerance_scan_match_fails_explicitly():
    """Replay cannot silently substitute a temporally unrelated scan."""
    with pytest.raises(ValueError, match="outside_tolerance=1"):
        replay.match_decisions_to_scans(
            [decision(0, 10.0)],
            [scan(10.2)],
            tolerance_sec=0.05,
        )


def test_equidistant_scan_match_fails_as_ambiguous():
    """Equidistant scan candidates require explicit operator review."""
    with pytest.raises(ValueError, match="ambiguous=1"):
        replay.match_decisions_to_scans(
            [decision(0, 10.0)],
            [scan(9.99), scan(10.01)],
            tolerance_sec=0.05,
        )


def test_named_candidates_are_always_scheduled_by_cli_defaults():
    """The CLI exposes the required legacy plus A/B/C comparison family."""
    args = replay.build_argument_parser().parse_args(
        [
            "--bag",
            "bag",
            "--episode-json",
            "episode.json",
            "--output-dir",
            "output",
        ]
    )

    assert args.fusion_config == []
    assert set(replay.EVIDENCE_FUSION_CANDIDATES) == {
        "candidate_a",
        "candidate_b",
        "candidate_c",
    }
