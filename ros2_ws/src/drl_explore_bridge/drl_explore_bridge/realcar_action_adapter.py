from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class ActionExecutionTarget:
    """Odom-frame motion target derived from one DRL grid action."""

    action_idx: int
    action_name: str
    grid_dr: int
    grid_dc: int
    odom_direction: str
    odom_dx: float
    odom_dy: float
    target_x: float
    target_y: float
    target_yaw: float
    target_distance: float

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class RealcarActionAdapter:
    """Translate existing DRL action deltas into fixed odom-frame targets."""

    SUPPORTED_DIAGONAL_MODES = ("grid_center", "constant_length")

    def __init__(
        self,
        actions_8: Sequence[tuple[int, int]],
        action_names: Sequence[str],
        diagonal_mode: str = "grid_center",
    ) -> None:
        if len(actions_8) != 8 or len(action_names) != 8:
            raise ValueError(
                "realcar action adapter requires exactly 8 actions and names"
            )
        if diagonal_mode not in self.SUPPORTED_DIAGONAL_MODES:
            raise ValueError(
                "diagonal_mode must be 'grid_center' or 'constant_length'"
            )

        self._actions_8 = tuple((int(dr), int(dc)) for dr, dc in actions_8)
        self._action_names = tuple(str(name) for name in action_names)
        self.diagonal_mode = diagonal_mode

    @staticmethod
    def _direction_label(dx: float, dy: float) -> str:
        labels: list[str] = []
        if dx > 0.0:
            labels.append("+x")
        elif dx < 0.0:
            labels.append("-x")
        if dy > 0.0:
            labels.append("+y")
        elif dy < 0.0:
            labels.append("-y")
        return ",".join(labels)

    def target_for_action(
        self,
        action_idx: int,
        start_x: float,
        start_y: float,
        step_distance: float,
    ) -> ActionExecutionTarget:
        action_idx = int(action_idx)
        if not (0 <= action_idx < len(self._actions_8)):
            raise ValueError(f"action_idx out of range: {action_idx}")
        if not math.isfinite(step_distance) or step_distance <= 0.0:
            raise ValueError("step_distance must be finite and > 0")
        if not math.isfinite(start_x) or not math.isfinite(start_y):
            raise ValueError("start odom position must be finite")

        dr, dc = self._actions_8[action_idx]
        component_distance = float(step_distance)
        if self.diagonal_mode == "constant_length" and dr != 0 and dc != 0:
            component_distance /= math.sqrt(2.0)

        # DRL grid rows grow downward while odom +y points north. Grid columns
        # and odom +x both grow eastward: N (dr=-1) -> +y, E (dc=+1) -> +x.
        odom_dx = float(dc) * component_distance
        odom_dy = float(-dr) * component_distance
        target_x = float(start_x) + odom_dx
        target_y = float(start_y) + odom_dy

        return ActionExecutionTarget(
            action_idx=action_idx,
            action_name=self._action_names[action_idx],
            grid_dr=dr,
            grid_dc=dc,
            odom_direction=self._direction_label(odom_dx, odom_dy),
            odom_dx=odom_dx,
            odom_dy=odom_dy,
            target_x=target_x,
            target_y=target_y,
            target_yaw=math.atan2(odom_dy, odom_dx),
            target_distance=math.hypot(odom_dx, odom_dy),
        )
