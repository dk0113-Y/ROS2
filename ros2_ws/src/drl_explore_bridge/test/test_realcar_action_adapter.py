import math

import pytest

from drl_explore_bridge.realcar_action_adapter import RealcarActionAdapter


ACTIONS_8 = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)
ACTION_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


@pytest.mark.parametrize(
    ("action_idx", "expected_dx", "expected_dy", "direction"),
    (
        (0, 0.0, 0.1, "+y"),
        (1, 0.1, 0.1, "+x,+y"),
        (2, 0.1, 0.0, "+x"),
        (3, 0.1, -0.1, "+x,-y"),
        (4, 0.0, -0.1, "-y"),
        (5, -0.1, -0.1, "-x,-y"),
        (6, -0.1, 0.0, "-x"),
        (7, -0.1, 0.1, "-x,+y"),
    ),
)
def test_all_actions_use_fixed_odom_axes(
    action_idx,
    expected_dx,
    expected_dy,
    direction,
):
    adapter = RealcarActionAdapter(ACTIONS_8, ACTION_NAMES)

    target = adapter.target_for_action(
        action_idx,
        start_x=1.0,
        start_y=2.0,
        step_distance=0.1,
    )

    assert target.odom_direction == direction
    assert target.target_x == pytest.approx(1.0 + expected_dx)
    assert target.target_y == pytest.approx(2.0 + expected_dy)
    assert target.target_yaw == pytest.approx(
        math.atan2(expected_dy, expected_dx)
    )


def test_diagonal_modes_preserve_grid_or_constant_distance():
    grid_adapter = RealcarActionAdapter(ACTIONS_8, ACTION_NAMES, "grid_center")
    length_adapter = RealcarActionAdapter(
        ACTIONS_8,
        ACTION_NAMES,
        "constant_length",
    )

    grid_target = grid_adapter.target_for_action(1, 0.0, 0.0, 0.1)
    length_target = length_adapter.target_for_action(1, 0.0, 0.0, 0.1)

    assert grid_target.target_distance == pytest.approx(math.sqrt(2.0) * 0.1)
    assert length_target.target_distance == pytest.approx(0.1)
    assert grid_target.odom_direction == "+x,+y"


def test_invalid_action_is_rejected():
    adapter = RealcarActionAdapter(ACTIONS_8, ACTION_NAMES)

    with pytest.raises(ValueError, match="action_idx out of range"):
        adapter.target_for_action(8, 0.0, 0.0, 0.1)
