import pytest

from drl_explore_bridge.realcar_policy_safe_runner_node import (
    MAX_SAFE_STEPS,
    odom_delta_to_grid_offset,
)


def test_multi_step_limit_is_three():
    assert MAX_SAFE_STEPS == 3


@pytest.mark.parametrize(
    ("delta_x", "delta_y", "expected_offset"),
    (
        (0.0, 0.35, (-1, 0)),
        (0.35, 0.0, (0, 1)),
        (0.0, -0.35, (1, 0)),
        (-0.35, 0.0, (0, -1)),
        (0.35, 0.35, (-1, 1)),
    ),
)
def test_odom_displacement_uses_existing_drl_grid_axes(
    delta_x,
    delta_y,
    expected_offset,
):
    assert odom_delta_to_grid_offset(delta_x, delta_y, 0.35) == expected_offset


def test_subcell_motion_does_not_blindly_advance_state():
    assert odom_delta_to_grid_offset(0.10, 0.0, 0.35) == (0, 0)
    assert odom_delta_to_grid_offset(0.20, 0.0, 0.35) == (0, 1)


def test_invalid_cell_size_is_rejected():
    with pytest.raises(ValueError, match="cell_size must be > 0"):
        odom_delta_to_grid_offset(0.0, 0.0, 0.0)
