"""Deterministic tests for the offline multi-scan evidence study."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_SOURCE = REPOSITORY_ROOT / "ros2_ws" / "src" / "drl_explore_bridge"
for source_path in (REPOSITORY_ROOT, PACKAGE_SOURCE):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from drl_explore_bridge.realcar_conservative_belief import (  # noqa: E402
    BeliefEvidenceAccumulator,
    INVISIBLE,
    ProjectedBeliefObservation,
    named_fusion_config,
)
from scripts_realcar import analyze_multiscan_evidence_density as study  # noqa


def _observation(
    free=(),
    obstacle=(),
    conflict=(),
    suppressed_free=(),
    suppressed_obstacle=(),
):
    return ProjectedBeliefObservation(
        local_snap=np.full((3, 3), INVISIBLE, dtype=np.int8),
        free_cells=frozenset(free),
        obstacle_cells=frozenset(obstacle),
        conflict_cells=frozenset(conflict),
        occlusion_suppressed_free_cells=frozenset(suppressed_free),
        occlusion_suppressed_obstacle_cells=frozenset(
            suppressed_obstacle
        ),
    )


@pytest.mark.parametrize("burst_size", [3, 5])
def test_burst_is_one_evidence_epoch(burst_size):
    cell = (4, 7)
    observations = [_observation(free=[cell]) for _ in range(burst_size)]
    epoch = study.aggregate_burst_observations(observations)
    accumulator = BeliefEvidenceAccumulator(named_fusion_config("candidate_a"))

    accumulator.observe(
        epoch.free_cells, epoch.obstacle_cells, epoch.conflict_cells
    )

    assert accumulator.cells[cell].free_frame_count == 1
    assert accumulator.classify(cell) is None


def test_burst_free_and_obstacle_across_scans_is_conflict():
    cell = (3, 9)
    epoch = study.aggregate_burst_observations(
        [_observation(free=[cell]), _observation(obstacle=[cell])]
    )

    assert epoch.conflict_cells == frozenset([cell])
    assert cell not in epoch.free_cells
    assert cell not in epoch.obstacle_cells


def test_accepted_conflict_remains_observed():
    cell = (5, 8)
    accumulator = BeliefEvidenceAccumulator(named_fusion_config("candidate_a"))

    accumulator.observe([], [], [cell])

    assert cell in accumulator.ever_observed_cells()
    assert accumulator.cells[cell].conflict_frame_count == 1


def test_occlusion_suppressed_evidence_is_not_observed():
    cell = (2, 6)
    epoch = study.aggregate_burst_observations(
        [_observation(suppressed_free=[cell])]
    )
    accumulator = BeliefEvidenceAccumulator(named_fusion_config("candidate_a"))

    accumulator.observe(
        epoch.free_cells, epoch.obstacle_cells, epoch.conflict_cells
    )

    assert cell not in accumulator.ever_observed_cells()


def test_causal_burst_never_uses_future_scan():
    scans = [
        study.RecordedScan(float(value), float(value), "header", None)
        for value in (0, 1, 2, 3, 4)
    ]

    selected = study.select_causal_burst(scans, 2.5, 3)

    assert [scan.timestamp_sec for scan in selected] == [0.0, 1.0, 2.0]
    assert all(scan.timestamp_sec <= 2.5 for scan in selected)


def test_yaw_interpolation_uses_shortest_wraparound_arc():
    start = math.radians(179.0)
    end = math.radians(-179.0)

    midpoint = study.interpolate_yaw(start, end, 0.5)

    assert abs(abs(midpoint) - math.pi) < 1e-12


def _scan(range_value=1.0):
    return SimpleNamespace(
        ranges=[range_value],
        range_min=0.1,
        range_max=5.0,
        angle_min=0.0,
        angle_max=0.0,
        angle_increment=1.0,
    )


def _projection_kwargs():
    return {
        "robot_x": 0.0,
        "robot_y": 0.0,
        "robot_yaw": 0.0,
        "origin_x": 0.0,
        "origin_y": 0.0,
        "origin_state": (60, 60),
        "agent_state": (60, 60),
        "cell_size": 0.35,
        "scan_radius_cells": 10,
        "laser_x_in_base": 0.0,
        "laser_y_in_base": 0.0,
        "laser_yaw_in_base": 0.0,
        "coarse_occlusion_mode": "off",
    }


def test_supercover_preserves_hit_cell_exclusion():
    observation = study.project_scan_supercover(
        _scan(), **_projection_kwargs()
    )

    assert observation.obstacle_cells
    assert observation.obstacle_cells.isdisjoint(observation.free_cells)


def test_supercover_same_margin_only_changes_free_traversal_sampling():
    production = study.project_for_mode(
        "production", _scan(), **_projection_kwargs()
    )
    supercover = study.project_for_mode(
        "supercover", _scan(), **_projection_kwargs()
    )

    assert supercover.obstacle_cells == production.obstacle_cells
    assert supercover.free_cells.issuperset(production.free_cells)
    assert supercover.local_snap.shape == production.local_snap.shape


def test_baseline_mode_dispatches_to_production_projector(monkeypatch):
    sentinel = _observation(free=[(1, 2)])
    calls = []

    def fake_projector(scan, **kwargs):
        calls.append((scan, kwargs))
        return sentinel

    monkeypatch.setattr(
        study, "production_project_scan_to_belief", fake_projector
    )
    scan = object()

    result = study.project_for_mode("production", scan, marker="baseline")

    assert result is sentinel
    assert calls == [(scan, {"marker": "baseline"})]


def test_repeated_deterministic_report_serialization_is_identical():
    report = {
        "cells": [{"row": np.int64(2), "col": 1}],
        "metrics": {"fraction": np.float64(0.25)},
    }

    first = json.dumps(
        report, sort_keys=True, allow_nan=False, default=study._json_default
    )
    second = json.dumps(
        report, sort_keys=True, allow_nan=False, default=study._json_default
    )

    assert first == second
