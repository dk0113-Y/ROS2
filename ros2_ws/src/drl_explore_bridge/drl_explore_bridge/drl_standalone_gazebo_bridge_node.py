from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


ACTION_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
SUPPORTED_SCAN_BRIDGE_MODES = ("ray_project", "los_compatible", "scan_template_los", "oracle_los")


def yaw_from_quat(q: Any) -> float:
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def norm_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def stamp_to_sec(stamp: Any) -> float:
    if stamp is None:
        return 0.0
    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1.0e-9


def make_twist(linear_x: float, angular_z: float) -> Twist:
    msg = Twist()
    msg.linear.x = float(linear_x)
    msg.angular.z = float(angular_z)
    return msg


def _expand_path(raw_path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(raw_path))))


def _import_drl_core(drl_repo: Path) -> dict[str, Any]:
    repo_s = str(drl_repo)
    if repo_s not in sys.path:
        sys.path.insert(0, repo_s)

    from agents.q_value_agent import (  # pylint: disable=import-outside-toplevel
        ExplorationQNetwork,
        StateTensorAdapter,
        select_greedy_action,
    )
    from env.core_cummap import CumulativeBeliefMap  # pylint: disable=import-outside-toplevel
    from env.grid_topology import (  # pylint: disable=import-outside-toplevel
        ACTIONS_8,
        EMPTY,
        INVISIBLE,
        OBSTACLE,
        GridTopology,
    )

    return {
        "ExplorationQNetwork": ExplorationQNetwork,
        "StateTensorAdapter": StateTensorAdapter,
        "select_greedy_action": select_greedy_action,
        "CumulativeBeliefMap": CumulativeBeliefMap,
        "EMPTY": int(EMPTY),
        "INVISIBLE": int(INVISIBLE),
        "OBSTACLE": int(OBSTACLE),
        "ACTIONS_8": tuple(ACTIONS_8),
        "GridTopology": GridTopology,
    }


class DrlStandaloneGazeboBridgeNode(Node):
    """Standalone Gazebo simulation bridge for DRL exploration policy inference."""

    def __init__(self) -> None:
        super().__init__("drl_standalone_gazebo_bridge_node")

        self._declare_parameters()
        self._read_parameters()
        self._validate_parameters()

        core = _import_drl_core(self.drl_repo)
        self.ExplorationQNetwork = core["ExplorationQNetwork"]
        self.StateTensorAdapter = core["StateTensorAdapter"]
        self.select_greedy_action = core["select_greedy_action"]
        self.CumulativeBeliefMap = core["CumulativeBeliefMap"]
        self.EMPTY = int(core["EMPTY"])
        self.INVISIBLE = int(core["INVISIBLE"])
        self.OBSTACLE = int(core["OBSTACLE"])
        self.ACTIONS_8 = tuple(core["ACTIONS_8"])
        self.GridTopology = core["GridTopology"]

        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Odometry] = None
        self.cum_map = None
        self.trajectory_positions: list[tuple[int, int]] = []
        self.step_count = 0
        self.best_coverage = 0.0
        self.no_progress_count = 0
        self.stop_reason = "not_started"
        self.runner_started_wall: Optional[float] = None
        self.done = False
        self._oracle_warning_logged = False
        self._local_snap_diagnostic_done = False

        self.true_grid = self._load_true_grid(self.true_grid_path)
        self.true_free_mask = self.GridTopology.free_mask(self.true_grid)
        self.adapter = self.StateTensorAdapter(device="cpu")
        self.net = self._load_policy(self.checkpoint_path)
        self.scan_template_los_rays: Optional[tuple[tuple[tuple[int, int, int, int], ...], ...]] = None

        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self._log_startup_config()

    def _declare_parameters(self) -> None:
        self.declare_parameter("drl_repo", "~/drl_repos/DRL-path-finding")
        self.declare_parameter(
            "checkpoint_path",
            "~/drl_repos/DRL-path-finding/deploy_checkpoints/last.pt",
        )
        self.declare_parameter(
            "true_grid_path",
            "assets/cell035/grids/random_train_like_seed20260513_true_grid.npy",
        )
        self.declare_parameter("cell_size", 0.35)
        self.declare_parameter("rows", 40)
        self.declare_parameter("cols", 60)
        self.declare_parameter("world_x", 21.0)
        self.declare_parameter("world_y", 14.0)
        self.declare_parameter("scan_radius_cells", 10)
        self.declare_parameter("laser_yaw_in_base", math.pi)
        self.declare_parameter("scan_bridge_mode", "los_compatible")
        self.declare_parameter("max_steps", 400)
        self.declare_parameter("coverage_goal", 0.95)
        self.declare_parameter("no_progress_limit", 120)
        self.declare_parameter("coverage_epsilon", 1.0e-4)
        self.declare_parameter("recent_traj_limit", 64)
        self.declare_parameter("multi_wall_timeout", 1800.0)
        self.declare_parameter("step_pause_sec", 0.3)
        self.declare_parameter("linear_speed", 0.10)
        self.declare_parameter("target_pos_tol", 0.055)
        self.declare_parameter("rotate_kp", 2.0)
        self.declare_parameter("rotate_max_w", 2.0)
        self.declare_parameter("rotate_min_w", 0.20)
        self.declare_parameter("rotate_tol_deg", 4.0)
        self.declare_parameter("rotate_sim_timeout", 12.0)
        self.declare_parameter("drive_sim_timeout", 12.0)
        self.declare_parameter("rotate_wall_timeout", 60.0)
        self.declare_parameter("drive_wall_timeout", 60.0)
        self.declare_parameter("control_debug_period", 1.0)
        self.declare_parameter("los_beam_window", 1)
        self.declare_parameter("los_obstacle_tolerance", 0.55)
        self.declare_parameter("scan_template_los_beam_window", 1)
        self.declare_parameter("scan_template_los_range_margin_m", -1.0)
        self.declare_parameter("scan_template_los_enable_corner_blocking", True)
        self.declare_parameter("scan_template_los_use_training_templates", True)
        self.declare_parameter("stop_repeat", 10)
        self.declare_parameter("diagnostic_compare_local_snaps", False)
        self.declare_parameter("diagnostic_print_ascii", False)
        self.declare_parameter("diagnostic_only_first_step", True)
        self.declare_parameter("diagnostic_no_motion", False)

    def _read_parameters(self) -> None:
        self.drl_repo = self._resolve_path_parameter("drl_repo", must_exist=True)
        self.checkpoint_path = self._resolve_path_parameter("checkpoint_path", must_exist=False)
        self.true_grid_path = self._resolve_path_parameter("true_grid_path", must_exist=False)
        self.cell_size = float(self.get_parameter("cell_size").value)
        self.rows = int(self.get_parameter("rows").value)
        self.cols = int(self.get_parameter("cols").value)
        self.world_x = float(self.get_parameter("world_x").value)
        self.world_y = float(self.get_parameter("world_y").value)
        self.scan_radius_cells = int(self.get_parameter("scan_radius_cells").value)
        self.laser_yaw_in_base = float(self.get_parameter("laser_yaw_in_base").value)
        self.scan_bridge_mode = str(self.get_parameter("scan_bridge_mode").value).strip().lower()
        self.max_steps = int(self.get_parameter("max_steps").value)
        self.coverage_goal = float(self.get_parameter("coverage_goal").value)
        self.no_progress_limit = int(self.get_parameter("no_progress_limit").value)
        self.coverage_epsilon = float(self.get_parameter("coverage_epsilon").value)
        self.recent_traj_limit = int(self.get_parameter("recent_traj_limit").value)
        self.multi_wall_timeout = float(self.get_parameter("multi_wall_timeout").value)
        self.step_pause_sec = float(self.get_parameter("step_pause_sec").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.target_pos_tol = float(self.get_parameter("target_pos_tol").value)
        self.rotate_kp = float(self.get_parameter("rotate_kp").value)
        self.rotate_max_w = float(self.get_parameter("rotate_max_w").value)
        self.rotate_min_w = float(self.get_parameter("rotate_min_w").value)
        self.rotate_tol = math.radians(float(self.get_parameter("rotate_tol_deg").value))
        self.rotate_sim_timeout = float(self.get_parameter("rotate_sim_timeout").value)
        self.drive_sim_timeout = float(self.get_parameter("drive_sim_timeout").value)
        self.rotate_wall_timeout = float(self.get_parameter("rotate_wall_timeout").value)
        self.drive_wall_timeout = float(self.get_parameter("drive_wall_timeout").value)
        self.control_debug_period = float(self.get_parameter("control_debug_period").value)
        self.los_beam_window = int(self.get_parameter("los_beam_window").value)
        self.los_obstacle_tolerance = float(self.get_parameter("los_obstacle_tolerance").value)
        self.scan_template_los_beam_window = int(
            self.get_parameter("scan_template_los_beam_window").value
        )
        self.scan_template_los_range_margin_m = float(
            self.get_parameter("scan_template_los_range_margin_m").value
        )
        self.scan_template_los_enable_corner_blocking = bool(
            self.get_parameter("scan_template_los_enable_corner_blocking").value
        )
        self.scan_template_los_use_training_templates = bool(
            self.get_parameter("scan_template_los_use_training_templates").value
        )
        self.stop_repeat = int(self.get_parameter("stop_repeat").value)
        self.diagnostic_compare_local_snaps = bool(
            self.get_parameter("diagnostic_compare_local_snaps").value
        )
        self.diagnostic_print_ascii = bool(self.get_parameter("diagnostic_print_ascii").value)
        self.diagnostic_only_first_step = bool(self.get_parameter("diagnostic_only_first_step").value)
        self.diagnostic_no_motion = bool(self.get_parameter("diagnostic_no_motion").value)
        self.local_size = 2 * self.scan_radius_cells + 1
        self.center = self.scan_radius_cells

    def _resolve_path_parameter(self, name: str, must_exist: bool) -> Path:
        raw_value = str(self.get_parameter(name).value)
        raw = _expand_path(raw_value)
        if raw.is_absolute():
            if must_exist and not raw.exists():
                raise FileNotFoundError(f"{name} does not exist: {raw}")
            return raw

        candidates = [Path.cwd() / raw]
        try:
            module_path = Path(__file__).resolve()
            for parent in module_path.parents:
                candidates.append(parent / raw)
        except Exception:
            pass

        for candidate in candidates:
            if candidate.exists():
                return candidate

        fallback = candidates[0]
        if must_exist:
            checked = ", ".join(str(p) for p in candidates)
            raise FileNotFoundError(f"{name} does not exist; checked: {checked}")
        return fallback

    def _validate_parameters(self) -> None:
        if self.scan_bridge_mode not in SUPPORTED_SCAN_BRIDGE_MODES:
            raise ValueError(
                f"unsupported scan_bridge_mode={self.scan_bridge_mode!r}; "
                f"expected one of {SUPPORTED_SCAN_BRIDGE_MODES}"
            )
        if self.cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive")
        if self.scan_radius_cells < 1:
            raise ValueError("scan_radius_cells must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.no_progress_limit < 1:
            raise ValueError("no_progress_limit must be >= 1")
        if self.recent_traj_limit < 1:
            raise ValueError("recent_traj_limit must be >= 1")
        if self.stop_repeat < 1:
            raise ValueError("stop_repeat must be >= 1")
        if self.scan_template_los_beam_window < 0:
            raise ValueError("scan_template_los_beam_window must be >= 0")
        if (
            not self.scan_template_los_use_training_templates
            and (self.scan_bridge_mode == "scan_template_los" or self.diagnostic_compare_local_snaps)
        ):
            raise ValueError(
                "scan_template_los_use_training_templates=false is not supported; "
                "scan_template_los must use DRL-path-finding RadarSensor.local_ray_templates"
            )
        if self.diagnostic_no_motion and not self.diagnostic_compare_local_snaps:
            raise ValueError("diagnostic_no_motion requires diagnostic_compare_local_snaps=true")

    def _log_startup_config(self) -> None:
        self.get_logger().info(
            "startup_config "
            "Gazebo simulation only standalone bridge "
            f"drl_repo={self.drl_repo} "
            f"checkpoint_path={self.checkpoint_path} "
            f"true_grid_path={self.true_grid_path} "
            f"cell_size={self.cell_size:.3f} rows={self.rows} cols={self.cols} "
            f"world_x={self.world_x:.3f} world_y={self.world_y:.3f} "
            f"scan_radius_cells={self.scan_radius_cells} local_shape=({self.local_size},{self.local_size}) "
            f"laser_yaw_in_base={self.laser_yaw_in_base:.6f} "
            f"scan_bridge_mode={self.scan_bridge_mode} "
            f"max_steps={self.max_steps} coverage_goal={self.coverage_goal:.4f} "
            f"no_progress_limit={self.no_progress_limit} coverage_epsilon={self.coverage_epsilon:.6f} "
            f"recent_traj_limit={self.recent_traj_limit} multi_wall_timeout={self.multi_wall_timeout:.1f} "
            f"step_pause_sec={self.step_pause_sec:.3f} linear_speed={self.linear_speed:.3f} "
            f"target_pos_tol={self.target_pos_tol:.3f} rotate_tol_deg={math.degrees(self.rotate_tol):.3f} "
            f"scan_template_los_beam_window={self.scan_template_los_beam_window} "
            f"scan_template_los_range_margin_m={self.scan_template_los_range_margin_m:.3f} "
            f"scan_template_los_enable_corner_blocking={self.scan_template_los_enable_corner_blocking} "
            f"scan_template_los_use_training_templates={self.scan_template_los_use_training_templates} "
            f"diagnostic_compare_local_snaps={self.diagnostic_compare_local_snaps} "
            f"diagnostic_print_ascii={self.diagnostic_print_ascii} "
            f"diagnostic_only_first_step={self.diagnostic_only_first_step} "
            f"diagnostic_no_motion={self.diagnostic_no_motion}"
        )

    def _load_true_grid(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"true_grid_path does not exist: {path}")
        grid = np.load(path).astype(np.int8)
        expected = (int(self.rows), int(self.cols))
        if grid.shape != expected:
            raise ValueError(f"true_grid shape mismatch: got {grid.shape}, expected {expected}")
        self.get_logger().info(
            "true_grid_loaded "
            f"path={path} shape={grid.shape} empty={int(np.count_nonzero(grid == self.EMPTY))} "
            f"obstacle={int(np.count_nonzero(grid == self.OBSTACLE))}"
        )
        return grid

    def _load_policy(self, checkpoint_path: Path) -> Any:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint_path does not exist: {checkpoint_path}")
        try:
            import torch  # pylint: disable=import-outside-toplevel
        except Exception as exc:
            raise RuntimeError(f"failed to import torch for checkpoint loading: {exc}") from exc

        net = self.ExplorationQNetwork()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"checkpoint is not a dict: {checkpoint_path}")
        if "online_state_dict" not in checkpoint:
            raise RuntimeError(f"checkpoint missing required key online_state_dict: {checkpoint_path}")

        net.load_state_dict(checkpoint["online_state_dict"], strict=True)
        net.eval()
        param_count = sum(int(p.numel()) for p in net.parameters())
        self.get_logger().info(
            "checkpoint_loaded "
            f"path={checkpoint_path} key=online_state_dict strict=True params={param_count}"
        )
        return net

    def _get_scan_template_los_rays(self) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
        if self.scan_template_los_rays is None:
            self.scan_template_los_rays = self._load_scan_template_los_rays()
        return self.scan_template_los_rays

    def _load_scan_template_los_rays(self) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
        if not self.scan_template_los_use_training_templates:
            raise RuntimeError(
                "scan_template_los requires DRL-path-finding RadarSensor.local_ray_templates; "
                "no fallback template implementation is used in this bridge"
            )

        repo_s = str(self.drl_repo)
        if repo_s not in sys.path:
            sys.path.insert(0, repo_s)
        try:
            from env.core_radar import RadarSensor  # pylint: disable=import-outside-toplevel
        except Exception as exc:
            raise RuntimeError(
                f"scan_template_los failed to import RadarSensor from {repo_s}: {exc}"
            ) from exc

        sensor = RadarSensor(scan_radius=int(self.scan_radius_cells))
        rays = tuple(tuple(tuple(int(v) for v in point) for point in ray) for ray in sensor.local_ray_templates)
        expected = (self.local_size, self.local_size)
        if tuple(int(v) for v in sensor.local_shape) != expected:
            raise RuntimeError(
                "scan_template_los RadarSensor local_shape mismatch: "
                f"got {sensor.local_shape}, expected {expected}"
            )
        if not rays:
            raise RuntimeError("scan_template_los RadarSensor.local_ray_templates is empty")
        self.get_logger().info(
            "scan_template_los_templates_loaded "
            f"rays={len(rays)} scan_radius_cells={self.scan_radius_cells} local_shape={expected}"
        )
        return rays

    def scan_cb(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def odom_cb(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def world_xy_to_grid_rc(self, x: float, y: float) -> tuple[int, int]:
        col = int(math.floor((float(x) + self.world_x / 2.0) / self.cell_size))
        row = int(math.floor((self.world_y / 2.0 - float(y)) / self.cell_size))
        return row, col

    def grid_rc_to_cell_center_xy(self, row: int, col: int) -> tuple[float, float]:
        x = -self.world_x / 2.0 + (int(col) + 0.5) * self.cell_size
        y = self.world_y / 2.0 - (int(row) + 0.5) * self.cell_size
        return x, y

    def pose_xy_yaw_time(self) -> tuple[float, float, float, float]:
        if self.latest_odom is None:
            raise RuntimeError("latest /odom is not available")
        pose = self.latest_odom.pose.pose
        return (
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quat(pose.orientation),
            stamp_to_sec(self.latest_odom.header.stamp),
        )

    def stop(self, repeat: Optional[int] = None) -> None:
        if getattr(self, "diagnostic_no_motion", False):
            return
        if not hasattr(self, "cmd_pub") or self.cmd_pub is None:
            return
        count = int(self.stop_repeat if repeat is None else repeat)
        msg = make_twist(0.0, 0.0)
        for _ in range(max(1, count)):
            if not rclpy.ok():
                return
            try:
                self.cmd_pub.publish(msg)
            except Exception as exc:
                self.get_logger().warn(f"stop publish skipped: {exc}")
                return
            time.sleep(0.02)

    def current_agent_state_checked(self) -> tuple[int, int]:
        if self.latest_odom is None:
            raise RuntimeError("latest /odom is not available")
        x = float(self.latest_odom.pose.pose.position.x)
        y = float(self.latest_odom.pose.pose.position.y)
        row, col = self.world_xy_to_grid_rc(x, y)
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise RuntimeError(
                f"agent_state outside true_grid: rc=({row},{col}) xy=({x:.3f},{y:.3f})"
            )
        if int(self.true_grid[row, col]) != self.EMPTY:
            raise RuntimeError(
                f"agent is not on an EMPTY true_grid cell: rc=({row},{col}) "
                f"value={int(self.true_grid[row, col])}"
            )
        return row, col

    def _mark_cell(self, snap: np.ndarray, dr: int, dc: int, value: int) -> None:
        lr = self.center + int(dr)
        lc = self.center + int(dc)
        if not (0 <= lr < self.local_size and 0 <= lc < self.local_size):
            return
        if int(dr) * int(dr) + int(dc) * int(dc) > self.scan_radius_cells * self.scan_radius_cells:
            return
        if int(value) == self.OBSTACLE:
            snap[lr, lc] = self.OBSTACLE
        elif snap[lr, lc] == self.INVISIBLE:
            snap[lr, lc] = self.EMPTY

    def _mark_local_index(self, snap: np.ndarray, local_r: int, local_c: int, value: int) -> None:
        lr = int(local_r)
        lc = int(local_c)
        if not (0 <= lr < self.local_size and 0 <= lc < self.local_size):
            return
        if int(value) == self.OBSTACLE:
            snap[lr, lc] = self.OBSTACLE
        elif snap[lr, lc] == self.INVISIBLE:
            snap[lr, lc] = self.EMPTY

    def _scan_template_los_range_margin(self) -> float:
        if self.scan_template_los_range_margin_m > 0.0:
            return float(self.scan_template_los_range_margin_m)
        # Half a cell bridges LaserScan's continuous hit distance to the
        # training observation's cell-center LOS decision boundary.
        return 0.5 * float(self.cell_size)

    def _scan_template_los_corner_blocked(
        self,
        snap: np.ndarray,
        prev_rel_r: Optional[int],
        prev_rel_c: Optional[int],
        rel_r: int,
        rel_c: int,
    ) -> bool:
        if not self.scan_template_los_enable_corner_blocking:
            return False
        if prev_rel_r is None or prev_rel_c is None:
            return False
        if abs(int(rel_r) - int(prev_rel_r)) != 1 or abs(int(rel_c) - int(prev_rel_c)) != 1:
            return False

        side_a = (self.center + int(rel_r), self.center + int(prev_rel_c))
        side_b = (self.center + int(prev_rel_r), self.center + int(rel_c))
        for lr, lc in (side_a, side_b):
            if not (0 <= lr < self.local_size and 0 <= lc < self.local_size):
                return False

        # LaserScan-only best effort: exact training corner blocking checks
        # true side cells. Here we only block when both side cells have already
        # been inferred as visible obstacles in this local snap.
        return bool(snap[side_a[0], side_a[1]] == self.OBSTACLE) and bool(
            snap[side_b[0], side_b[1]] == self.OBSTACLE
        )

    def _ray_to_local_cells(self, angle_world: float, dist: float, hit_obstacle: bool):
        step = max(1.0e-6, self.cell_size / 3.0)
        max_d = max(0.0, float(dist))
        free_end = max_d if not hit_obstacle else max(0.0, max_d - self.cell_size * 0.25)
        seen: set[tuple[int, int]] = set()
        d = 0.0

        while d <= free_end:
            rel_x = d * math.cos(angle_world)
            rel_y = d * math.sin(angle_world)
            dc = int(round(rel_x / self.cell_size))
            dr = int(round(-rel_y / self.cell_size))
            if (dr, dc) not in seen:
                seen.add((dr, dc))
                yield dr, dc, self.EMPTY
            d += step

        if hit_obstacle:
            rel_x = max_d * math.cos(angle_world)
            rel_y = max_d * math.sin(angle_world)
            dc = int(round(rel_x / self.cell_size))
            dr = int(round(-rel_y / self.cell_size))
            yield dr, dc, self.OBSTACLE

    @staticmethod
    def _bresenham_local_line(dr: int, dc: int) -> list[tuple[int, int]]:
        r0, c0 = 0, 0
        r1, c1 = int(dr), int(dc)
        points: list[tuple[int, int]] = []
        ar = abs(r1 - r0)
        ac = abs(c1 - c0)
        sr = 1 if r1 >= r0 else -1
        sc = 1 if c1 >= c0 else -1

        r, c = r0, c0
        if ac >= ar:
            err = ac / 2
            while c != c1:
                c += sc
                err -= ar
                if err < 0:
                    r += sr
                    err += ac
                if r != 0 or c != 0:
                    points.append((int(r), int(c)))
        else:
            err = ar / 2
            while r != r1:
                r += sr
                err -= ac
                if err < 0:
                    c += sc
                    err += ar
                if r != 0 or c != 0:
                    points.append((int(r), int(c)))
        return points

    @staticmethod
    def _angle_in_scan_interval(angle: float, angle_min: float, angle_max: float) -> Optional[float]:
        candidates = [float(angle), float(angle) + 2.0 * math.pi, float(angle) - 2.0 * math.pi]
        valid = [a for a in candidates if angle_min <= a <= angle_max]
        if not valid:
            return None
        return min(valid, key=lambda a: abs(a - angle))

    def _scan_range_at_angle(
        self,
        scan: LaserScan,
        scan_angle: float,
        beam_window: Optional[int] = None,
    ) -> tuple[float, bool]:
        if len(scan.ranges) <= 0 or float(scan.angle_increment) == 0.0:
            return float(scan.range_max), False

        angle_min = float(scan.angle_min)
        angle_max = float(scan.angle_max)
        angle_inc = float(scan.angle_increment)
        normalized = self._angle_in_scan_interval(scan_angle, angle_min, angle_max)
        if normalized is None:
            return float(scan.range_max), False

        idx = int(round((normalized - angle_min) / angle_inc))
        idx = max(0, min(len(scan.ranges) - 1, idx))
        window = max(0, int(self.los_beam_window if beam_window is None else beam_window))
        i0 = max(0, idx - window)
        i1 = min(len(scan.ranges), idx + window + 1)

        finite_values: list[float] = []
        for j in range(i0, i1):
            value = scan.ranges[j]
            if math.isfinite(value):
                rr = min(max(float(value), float(scan.range_min)), float(scan.range_max))
                finite_values.append(rr)

        if finite_values:
            return min(finite_values), True
        return float(scan.range_max), False

    def build_local_snap_ray_project(self, scan: LaserScan, odom: Odometry) -> np.ndarray:
        robot_yaw = yaw_from_quat(odom.pose.pose.orientation)
        snap = np.full((self.local_size, self.local_size), self.INVISIBLE, dtype=np.int8)
        snap[self.center, self.center] = self.EMPTY

        for i, raw_range in enumerate(scan.ranges):
            scan_angle = float(scan.angle_min) + i * float(scan.angle_increment)
            if math.isfinite(raw_range):
                dist = min(max(float(raw_range), float(scan.range_min)), float(scan.range_max))
                hit_obstacle = True
            else:
                dist = float(scan.range_max)
                hit_obstacle = False

            angle_world = robot_yaw + self.laser_yaw_in_base + scan_angle
            for dr, dc, value in self._ray_to_local_cells(angle_world, dist, hit_obstacle):
                self._mark_cell(snap, dr, dc, value)

        return snap

    def build_local_snap_los_compatible(self, scan: LaserScan, odom: Odometry) -> np.ndarray:
        robot_yaw = yaw_from_quat(odom.pose.pose.orientation)
        snap = np.full((self.local_size, self.local_size), self.INVISIBLE, dtype=np.int8)
        snap[self.center, self.center] = self.EMPTY

        targets: list[tuple[int, int, int]] = []
        radius_sq = self.scan_radius_cells * self.scan_radius_cells
        for dr in range(-self.scan_radius_cells, self.scan_radius_cells + 1):
            for dc in range(-self.scan_radius_cells, self.scan_radius_cells + 1):
                if dr == 0 and dc == 0:
                    continue
                r2 = dr * dr + dc * dc
                if r2 <= radius_sq:
                    targets.append((int(dr), int(dc), int(r2)))
        targets.sort(key=lambda x: x[2])

        obstacle_tol_m = max(1.0e-6, float(self.los_obstacle_tolerance) * self.cell_size)
        max_range = float(scan.range_max)

        for target_dr, target_dc, _ in targets:
            line = self._bresenham_local_line(target_dr, target_dc)
            for dr, dc in line:
                if dr * dr + dc * dc > radius_sq:
                    break

                rel_x = float(dc) * self.cell_size
                rel_y = -float(dr) * self.cell_size
                cell_dist = math.hypot(rel_x, rel_y)
                if cell_dist > max_range + obstacle_tol_m:
                    break

                angle_world = math.atan2(rel_y, rel_x)
                scan_angle = norm_angle(angle_world - robot_yaw - self.laser_yaw_in_base)
                measured_dist, hit_obstacle = self._scan_range_at_angle(scan, scan_angle)

                if not hit_obstacle:
                    self._mark_cell(snap, dr, dc, self.EMPTY)
                    continue
                if cell_dist < measured_dist - obstacle_tol_m:
                    self._mark_cell(snap, dr, dc, self.EMPTY)
                    continue
                if abs(cell_dist - measured_dist) <= obstacle_tol_m:
                    self._mark_cell(snap, dr, dc, self.OBSTACLE)
                    break
                break

        return snap

    def build_local_snap_scan_template_los(self, scan: LaserScan, odom: Odometry) -> np.ndarray:
        robot_yaw = yaw_from_quat(odom.pose.pose.orientation)
        snap = np.full((self.local_size, self.local_size), self.INVISIBLE, dtype=np.int8)
        snap[self.center, self.center] = self.EMPTY

        range_margin = self._scan_template_los_range_margin()
        max_range = float(scan.range_max)

        for ray in self._get_scan_template_los_rays():
            prev_rel_r: Optional[int] = None
            prev_rel_c: Optional[int] = None
            for rel_r, rel_c, local_r, local_c in ray:
                rel_r_i = int(rel_r)
                rel_c_i = int(rel_c)
                if rel_r_i == 0 and rel_c_i == 0:
                    self._mark_local_index(snap, local_r, local_c, self.EMPTY)
                    prev_rel_r = rel_r_i
                    prev_rel_c = rel_c_i
                    continue

                if self._scan_template_los_corner_blocked(
                    snap,
                    prev_rel_r,
                    prev_rel_c,
                    rel_r_i,
                    rel_c_i,
                ):
                    break

                rel_x = float(rel_c_i) * self.cell_size
                rel_y = -float(rel_r_i) * self.cell_size
                cell_dist = math.hypot(rel_x, rel_y)
                angle_world_local = math.atan2(rel_y, rel_x)
                scan_angle = norm_angle(angle_world_local - robot_yaw - self.laser_yaw_in_base)
                measured_dist, hit_obstacle = self._scan_range_at_angle(
                    scan,
                    scan_angle,
                    self.scan_template_los_beam_window,
                )

                if cell_dist > max_range + range_margin:
                    break
                if not hit_obstacle:
                    self._mark_local_index(snap, local_r, local_c, self.EMPTY)
                    prev_rel_r = rel_r_i
                    prev_rel_c = rel_c_i
                    continue
                if cell_dist < measured_dist - range_margin:
                    self._mark_local_index(snap, local_r, local_c, self.EMPTY)
                    prev_rel_r = rel_r_i
                    prev_rel_c = rel_c_i
                    continue
                if abs(cell_dist - measured_dist) <= range_margin:
                    self._mark_local_index(snap, local_r, local_c, self.OBSTACLE)
                    break
                break

        return snap

    def build_local_snap_oracle_los(self, scan: LaserScan, odom: Odometry) -> np.ndarray:
        del scan
        if not self._oracle_warning_logged:
            self.get_logger().warn(
                "oracle_los uses true_grid and is diagnostic-only, not deployment-like sensing"
            )
            self._oracle_warning_logged = True

        repo_s = str(self.drl_repo)
        if repo_s not in sys.path:
            sys.path.insert(0, repo_s)
        try:
            from env.agent_version import LocalObservationModel  # pylint: disable=import-outside-toplevel
            from env.core_radar import RadarSensor  # pylint: disable=import-outside-toplevel
        except Exception as exc:
            raise RuntimeError(f"oracle_los failed to import observation modules from {repo_s}: {exc}") from exc

        x = float(odom.pose.pose.position.x)
        y = float(odom.pose.pose.position.y)
        agent_state = self.world_xy_to_grid_rc(x, y)
        sensor = RadarSensor(scan_radius=int(self.scan_radius_cells))
        obs_model = LocalObservationModel(self.true_grid, agent_state, sensor=sensor)

        if hasattr(obs_model, "observe_fast"):
            snap = obs_model.observe_fast(agent_state)
        else:
            snap, _ = obs_model.observe(agent_state)

        snap = np.asarray(snap, dtype=np.int8)
        expected = (self.local_size, self.local_size)
        if snap.shape != expected:
            raise RuntimeError(f"oracle_los snap shape mismatch: got {snap.shape}, expected {expected}")
        return snap

    def build_local_snap(self, scan: LaserScan, odom: Odometry) -> np.ndarray:
        if self.scan_bridge_mode == "ray_project":
            return self.build_local_snap_ray_project(scan, odom)
        if self.scan_bridge_mode == "los_compatible":
            return self.build_local_snap_los_compatible(scan, odom)
        if self.scan_bridge_mode == "scan_template_los":
            return self.build_local_snap_scan_template_los(scan, odom)
        if self.scan_bridge_mode == "oracle_los":
            return self.build_local_snap_oracle_los(scan, odom)
        raise RuntimeError(f"unsupported scan_bridge_mode={self.scan_bridge_mode!r}")

    def _should_emit_local_snap_diagnostic(self) -> bool:
        if not self.diagnostic_compare_local_snaps:
            return False
        if self.diagnostic_only_first_step and self._local_snap_diagnostic_done:
            return False
        return True

    def _snap_counts(self, snap: np.ndarray) -> dict[str, Any]:
        return {
            "shape": tuple(int(v) for v in snap.shape),
            "known_count": int(np.count_nonzero(snap != self.INVISIBLE)),
            "empty_count": int(np.count_nonzero(snap == self.EMPTY)),
            "obstacle_count": int(np.count_nonzero(snap == self.OBSTACLE)),
            "invisible_count": int(np.count_nonzero(snap == self.INVISIBLE)),
            "center_value": int(snap[self.center, self.center]),
        }

    def _snap_counts_log_fragment(self, name: str, snap: np.ndarray) -> str:
        counts = self._snap_counts(snap)
        return (
            f"{name}_shape={counts['shape']} "
            f"{name}_known_count={counts['known_count']} "
            f"{name}_empty_count={counts['empty_count']} "
            f"{name}_obstacle_count={counts['obstacle_count']} "
            f"{name}_invisible_count={counts['invisible_count']} "
            f"{name}_center_cell_value={counts['center_value']}"
        )

    def _snap_to_ascii_lines(self, snap: np.ndarray) -> list[str]:
        chars = {
            self.INVISIBLE: "?",
            self.EMPTY: ".",
            self.OBSTACLE: "#",
        }
        lines: list[str] = []
        for r in range(int(snap.shape[0])):
            row_chars: list[str] = []
            for c in range(int(snap.shape[1])):
                if r == self.center and c == self.center:
                    row_chars.append("A")
                else:
                    row_chars.append(chars.get(int(snap[r, c]), "!"))
            lines.append("".join(row_chars))
        return lines

    def _diff_to_ascii_lines(
        self,
        oracle: np.ndarray,
        ray_project: np.ndarray,
        los_compatible: np.ndarray,
        scan_template_los: np.ndarray,
    ) -> list[str]:
        lines: list[str] = []
        for r in range(int(oracle.shape[0])):
            row_chars: list[str] = []
            for c in range(int(oracle.shape[1])):
                ray_mismatch = bool(ray_project[r, c] != oracle[r, c])
                los_mismatch = bool(los_compatible[r, c] != oracle[r, c])
                template_mismatch = bool(scan_template_los[r, c] != oracle[r, c])
                if ray_mismatch and los_mismatch and template_mismatch:
                    row_chars.append("a")
                elif ray_mismatch and los_mismatch:
                    row_chars.append("b")
                elif ray_mismatch and template_mismatch:
                    row_chars.append("R")
                elif los_mismatch and template_mismatch:
                    row_chars.append("L")
                elif ray_mismatch:
                    row_chars.append("r")
                elif los_mismatch:
                    row_chars.append("l")
                elif template_mismatch:
                    row_chars.append("t")
                else:
                    row_chars.append(" ")
            lines.append("".join(row_chars))
        return lines

    def _log_ascii_snap(self, name: str, snap: np.ndarray) -> None:
        body = "\n".join(f"{idx:02d} {line}" for idx, line in enumerate(self._snap_to_ascii_lines(snap)))
        self.get_logger().info(f"local_snap_ascii name={name}\n{body}")

    def _log_ascii_diff(
        self,
        oracle: np.ndarray,
        ray_project: np.ndarray,
        los_compatible: np.ndarray,
        scan_template_los: np.ndarray,
    ) -> None:
        lines = self._diff_to_ascii_lines(oracle, ray_project, los_compatible, scan_template_los)
        body = "\n".join(f"{idx:02d} |{line}|" for idx, line in enumerate(lines))
        self.get_logger().info(
            "local_snap_ascii_diff name=diff_vs_oracle "
            "legend='space=match,r=ray,l=los,t=scan_template,b=ray+los,R=ray+scan_template,L=los+scan_template,a=all'\n"
            f"{body}"
        )

    def emit_local_snap_alignment_diagnostic(self, agent_state: tuple[int, int]) -> None:
        if self.latest_scan is None or self.latest_odom is None:
            raise RuntimeError("waiting for /scan and /odom")

        scan = self.latest_scan
        odom = self.latest_odom
        oracle = self.build_local_snap_oracle_los(scan, odom)
        ray_project = self.build_local_snap_ray_project(scan, odom)
        los_compatible = self.build_local_snap_los_compatible(scan, odom)
        scan_template_los = self.build_local_snap_scan_template_los(scan, odom)

        x = float(odom.pose.pose.position.x)
        y = float(odom.pose.pose.position.y)
        yaw = yaw_from_quat(odom.pose.pose.orientation)
        ranges = list(scan.ranges)
        finite_ranges = [float(v) for v in ranges if math.isfinite(v)]
        finite_count = int(len(finite_ranges))
        nan_count = int(sum(1 for v in ranges if math.isnan(float(v))))
        inf_count = int(sum(1 for v in ranges if math.isinf(float(v))))
        near_range_max_count = int(
            sum(1 for v in finite_ranges if float(scan.range_max) - v <= max(1.0e-6, self.cell_size * 0.05))
        )

        oracle_vs_ray = oracle != ray_project
        oracle_vs_los = oracle != los_compatible
        oracle_vs_scan_template_los = oracle != scan_template_los
        ray_vs_los = ray_project != los_compatible
        ray_vs_scan_template_los = ray_project != scan_template_los
        los_vs_scan_template_los = los_compatible != scan_template_los

        self.get_logger().info(
            "local_snap_alignment "
            f"step={self.step_count + 1} agent_state={agent_state} "
            f"odom_xy=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):+.1f}deg "
            f"scan_angle_min={float(scan.angle_min):+.6f} "
            f"scan_angle_max={float(scan.angle_max):+.6f} "
            f"scan_angle_increment={float(scan.angle_increment):+.6f} "
            f"scan_range_min={float(scan.range_min):.3f} "
            f"scan_range_max={float(scan.range_max):.3f} "
            f"scan_ranges_count={len(ranges)} "
            f"scan_finite_count={finite_count} scan_nan_count={nan_count} "
            f"scan_inf_count={inf_count} scan_near_range_max_count={near_range_max_count} "
            f"{self._snap_counts_log_fragment('oracle', oracle)} "
            f"{self._snap_counts_log_fragment('ray_project', ray_project)} "
            f"{self._snap_counts_log_fragment('los_compatible', los_compatible)} "
            f"{self._snap_counts_log_fragment('scan_template_los', scan_template_los)} "
            f"oracle_vs_ray_mismatch_count={int(np.count_nonzero(oracle_vs_ray))} "
            f"oracle_vs_los_mismatch_count={int(np.count_nonzero(oracle_vs_los))} "
            f"oracle_vs_scan_template_los_mismatch_count={int(np.count_nonzero(oracle_vs_scan_template_los))} "
            f"ray_vs_los_mismatch_count={int(np.count_nonzero(ray_vs_los))} "
            f"ray_vs_scan_template_los_mismatch_count={int(np.count_nonzero(ray_vs_scan_template_los))} "
            f"los_vs_scan_template_los_mismatch_count={int(np.count_nonzero(los_vs_scan_template_los))} "
            f"oracle_empty_ray_invisible_count={int(np.count_nonzero((oracle == self.EMPTY) & (ray_project == self.INVISIBLE)))} "
            f"oracle_empty_los_invisible_count={int(np.count_nonzero((oracle == self.EMPTY) & (los_compatible == self.INVISIBLE)))} "
            f"oracle_empty_scan_template_los_invisible_count={int(np.count_nonzero((oracle == self.EMPTY) & (scan_template_los == self.INVISIBLE)))} "
            f"oracle_obstacle_ray_empty_count={int(np.count_nonzero((oracle == self.OBSTACLE) & (ray_project == self.EMPTY)))} "
            f"oracle_obstacle_los_empty_count={int(np.count_nonzero((oracle == self.OBSTACLE) & (los_compatible == self.EMPTY)))} "
            f"oracle_obstacle_scan_template_los_empty_count={int(np.count_nonzero((oracle == self.OBSTACLE) & (scan_template_los == self.EMPTY)))} "
            f"oracle_obstacle_ray_invisible_count={int(np.count_nonzero((oracle == self.OBSTACLE) & (ray_project == self.INVISIBLE)))} "
            f"oracle_obstacle_los_invisible_count={int(np.count_nonzero((oracle == self.OBSTACLE) & (los_compatible == self.INVISIBLE)))} "
            f"oracle_obstacle_scan_template_los_invisible_count={int(np.count_nonzero((oracle == self.OBSTACLE) & (scan_template_los == self.INVISIBLE)))}"
        )

        if self.diagnostic_print_ascii:
            self._log_ascii_snap("oracle", oracle)
            self._log_ascii_snap("ray_project", ray_project)
            self._log_ascii_snap("los_compatible", los_compatible)
            self._log_ascii_snap("scan_template_los", scan_template_los)
            self._log_ascii_diff(oracle, ray_project, los_compatible, scan_template_los)

        self._local_snap_diagnostic_done = True

    def update_persistent_belief(self, agent_state: tuple[int, int]) -> tuple[Any, tuple[int, int, int]]:
        if self.latest_scan is None or self.latest_odom is None:
            raise RuntimeError("waiting for /scan and /odom")
        local_snap = self.build_local_snap(self.latest_scan, self.latest_odom)

        if self.cum_map is None:
            self.cum_map = self.CumulativeBeliefMap(
                true_grid=self.true_grid,
                start_state=agent_state,
                first_local_snap=local_snap,
            )
            return self.cum_map, (
                int(np.count_nonzero(local_snap != self.INVISIBLE)),
                int(np.count_nonzero(local_snap == self.EMPTY)),
                int(np.count_nonzero(local_snap == self.OBSTACLE)),
            )

        updated, delta_empty, delta_obstacle = self.cum_map.update(agent_state, local_snap)
        return self.cum_map, (int(updated), int(delta_empty), int(delta_obstacle))

    def infer_action(
        self,
    ) -> tuple[int, tuple[int, ...], Any, Any, tuple[int, int], dict[str, Any], tuple[int, int, int]]:
        agent_state = self.current_agent_state_checked()
        cum_map, update_result = self.update_persistent_belief(agent_state)

        if not self.trajectory_positions or self.trajectory_positions[-1] != agent_state:
            self.trajectory_positions.append(agent_state)
        recent_traj = self.trajectory_positions[-self.recent_traj_limit :]

        state_batch, state_meta = self.adapter.build_single_state_tensors(
            cum_map,
            agent_state,
            recent_trajectory_positions=recent_traj,
            return_state_meta=True,
        )

        import torch  # pylint: disable=import-outside-toplevel

        with torch.no_grad():
            q_values, _ = self.net(
                state_batch["advantage_canvas"],
                state_batch["value_block_features"],
                state_batch["value_entry_features"],
                state_batch["value_block_mask"],
                state_batch["value_entry_mask"],
                return_aux=True,
            )

        q = q_values.squeeze(0).cpu()
        valid = self.GridTopology.valid_action_indices_fast(self.true_free_mask, agent_state)
        if len(valid) <= 0:
            raise RuntimeError(f"no valid action at agent_state={agent_state}")
        action_idx = int(self.select_greedy_action(q, valid_action_indices=valid).item())
        return action_idx, tuple(int(v) for v in valid), q, cum_map, agent_state, state_meta, update_result

    def _refresh_progress(self, coverage: float) -> None:
        if float(coverage) > self.best_coverage + self.coverage_epsilon:
            self.best_coverage = float(coverage)
            self.no_progress_count = 0
        else:
            self.no_progress_count += 1

    def _target_for_action(self, action_idx: int, agent_state: tuple[int, int]) -> tuple[tuple[int, int], tuple[float, float]]:
        dr, dc = self.ACTIONS_8[int(action_idx)]
        target_state = (int(agent_state[0]) + int(dr), int(agent_state[1]) + int(dc))
        tr, tc = target_state
        if not (0 <= tr < self.rows and 0 <= tc < self.cols):
            raise RuntimeError(f"target_state outside true_grid: {target_state}")
        if int(self.true_grid[tr, tc]) != self.EMPTY:
            raise RuntimeError(f"target_state is not EMPTY: rc={target_state} value={int(self.true_grid[tr, tc])}")
        return target_state, self.grid_rc_to_cell_center_xy(tr, tc)

    def execute_target_cell(self, action_idx: int, start_state: tuple[int, int]) -> bool:
        x0, y0, yaw0, sim0 = self.pose_xy_yaw_time()
        target_state, (tx, ty) = self._target_for_action(action_idx, start_state)
        row0, col0 = start_state
        c0x, c0y = self.grid_rc_to_cell_center_xy(row0, col0)
        target_yaw = math.atan2(ty - y0, tx - x0)
        initial_dist = math.hypot(tx - x0, ty - y0)

        self.get_logger().info(
            "bridge_step_control "
            f"action_idx={action_idx} action_name={ACTION_NAMES[action_idx]} "
            f"start_state=({row0},{col0}) target_state={target_state} "
            f"start_xy=({x0:.3f},{y0:.3f}) start_yaw={math.degrees(yaw0):+.1f}deg "
            f"start_cell_center=({c0x:.3f},{c0y:.3f}) "
            f"target_xy=({tx:.3f},{ty:.3f}) target_yaw={math.degrees(target_yaw):+.1f}deg "
            f"initial_dist={initial_dist:.3f}"
        )

        rotate_sim_start = sim0
        rotate_wall_start = time.monotonic()
        last_rotate_debug_wall = -1.0e9
        stable_count = 0

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            x, y, yaw, sim_now = self.pose_xy_yaw_time()
            target_yaw_now = math.atan2(ty - y, tx - x)
            err = norm_angle(target_yaw_now - yaw)
            sim_elapsed = sim_now - rotate_sim_start
            wall_elapsed = time.monotonic() - rotate_wall_start

            if abs(err) < self.rotate_tol:
                stable_count += 1
                if stable_count >= 5:
                    break
            else:
                stable_count = 0

            if sim_elapsed > self.rotate_sim_timeout:
                raise RuntimeError(f"rotate_sim_timeout, last yaw error={err:.3f} rad")
            if wall_elapsed > self.rotate_wall_timeout:
                raise RuntimeError(
                    f"rotate_wall_timeout, sim_elapsed={sim_elapsed:.3f}s, last yaw error={err:.3f} rad"
                )

            wz = max(-self.rotate_max_w, min(self.rotate_max_w, self.rotate_kp * err))
            if abs(wz) < self.rotate_min_w:
                wz = math.copysign(self.rotate_min_w, wz)

            if wall_elapsed - last_rotate_debug_wall >= self.control_debug_period:
                last_rotate_debug_wall = wall_elapsed
                self.get_logger().info(
                    "rotate_debug "
                    f"xy=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):+.1f}deg "
                    f"target_yaw={math.degrees(target_yaw_now):+.1f}deg "
                    f"err={math.degrees(err):+.1f}deg wz={wz:+.3f} "
                    f"stable={stable_count} sim_elapsed={sim_elapsed:.3f}s "
                    f"wall_elapsed={wall_elapsed:.3f}s"
                )
            self.cmd_pub.publish(make_twist(0.0, wz))

        self.stop()

        xr, yr, yawr, simr = self.pose_xy_yaw_time()
        self.get_logger().info(
            "rotate_done "
            f"xy=({xr:.3f},{yr:.3f}) yaw={math.degrees(yawr):+.1f}deg "
            f"dist_to_target={math.hypot(tx - xr, ty - yr):.3f} "
            f"sim_elapsed={simr - rotate_sim_start:.3f}s"
        )

        drive_sim_start = simr
        drive_wall_start = time.monotonic()
        last_drive_debug_wall = -1.0e9

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            x, y, yaw, sim_now = self.pose_xy_yaw_time()
            dist_to_target = math.hypot(tx - x, ty - y)
            sim_elapsed = sim_now - drive_sim_start
            wall_elapsed = time.monotonic() - drive_wall_start

            if dist_to_target <= self.target_pos_tol:
                break
            if sim_elapsed > self.drive_sim_timeout:
                raise RuntimeError(f"drive_sim_timeout, distance_to_target={dist_to_target:.3f} m")
            if wall_elapsed > self.drive_wall_timeout:
                raise RuntimeError(
                    f"drive_wall_timeout, sim_elapsed={sim_elapsed:.3f}s, distance_to_target={dist_to_target:.3f} m"
                )

            target_yaw_now = math.atan2(ty - y, tx - x)
            yaw_err = norm_angle(target_yaw_now - yaw)
            wz_correction = max(-0.3, min(0.3, yaw_err))

            if wall_elapsed - last_drive_debug_wall >= self.control_debug_period:
                last_drive_debug_wall = wall_elapsed
                self.get_logger().info(
                    "drive_debug "
                    f"xy=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):+.1f}deg "
                    f"target_yaw={math.degrees(target_yaw_now):+.1f}deg "
                    f"yaw_err={math.degrees(yaw_err):+.1f}deg dist={dist_to_target:.3f} "
                    f"vx={self.linear_speed:+.3f} wz={wz_correction:+.3f} "
                    f"sim_elapsed={sim_elapsed:.3f}s wall_elapsed={wall_elapsed:.3f}s"
                )
            self.cmd_pub.publish(make_twist(self.linear_speed, wz_correction))

        self.stop()

        x1, y1, yaw1, sim1 = self.pose_xy_yaw_time()
        final_state = self.world_xy_to_grid_rc(x1, y1)
        final_dist = math.hypot(tx - x1, ty - y1)
        reached = final_state == target_state

        self.get_logger().info(
            "bridge_step_done "
            f"final_state={final_state} target_state={target_state} "
            f"final_xy=({x1:.3f},{y1:.3f}) final_yaw={math.degrees(yaw1):+.1f}deg "
            f"final_dist_to_target={final_dist:.3f} drive_sim_elapsed={sim1 - drive_sim_start:.3f}s "
            f"reached_target_cell={reached}"
        )
        if not reached:
            self.get_logger().warn(
                "bridge_step_done target cell not reached; stop command sent and runner will finish"
            )
        return bool(reached)

    def _wait_for_scan_odom(self) -> None:
        self.get_logger().info("waiting_for_scan_odom waiting for /scan and /odom")
        wait_start = time.monotonic()
        last_log = -1.0e9
        while rclpy.ok() and (self.latest_scan is None or self.latest_odom is None):
            rclpy.spin_once(self, timeout_sec=0.1)
            elapsed = time.monotonic() - wait_start
            if elapsed - last_log >= 2.0:
                last_log = elapsed
                self.get_logger().warn(
                    "waiting_for_scan_odom "
                    f"scan_ready={self.latest_scan is not None} odom_ready={self.latest_odom is not None} "
                    f"elapsed={elapsed:.1f}s"
                )
            if elapsed > self.multi_wall_timeout:
                raise RuntimeError(
                    f"multi_wall_timeout reached while waiting for /scan and /odom: elapsed={elapsed:.1f}s"
                )
        if not rclpy.ok():
            raise RuntimeError("rclpy shutdown while waiting for /scan and /odom")
        self.get_logger().info("waiting_for_scan_odom ready")

    def _check_cmd_vel_subscriber(self) -> None:
        discover_until = time.monotonic() + 1.0
        subscriber_count = 0
        while rclpy.ok() and time.monotonic() < discover_until:
            rclpy.spin_once(self, timeout_sec=0.1)
            subscriber_count = int(self.cmd_pub.get_subscription_count())
            if subscriber_count >= 1:
                break
        if subscriber_count < 1:
            self.stop()
            raise RuntimeError("No /cmd_vel subscriber found")
        self.get_logger().info(f"startup_config cmd_vel_subscribers={subscriber_count}")

    def _stop_condition(self, elapsed_wall: float) -> Optional[str]:
        if self.step_count >= self.max_steps:
            return "max_steps"
        if self.best_coverage >= self.coverage_goal:
            return "coverage_goal"
        if self.no_progress_count >= self.no_progress_limit:
            return "no_progress_limit"
        if elapsed_wall > self.multi_wall_timeout:
            return "multi_wall_timeout"
        return None

    def run(self) -> None:
        self.runner_started_wall = time.monotonic()
        try:
            if not self.diagnostic_no_motion:
                self._check_cmd_vel_subscriber()
            self._wait_for_scan_odom()
            self.runner_started_wall = time.monotonic()

            if self.diagnostic_no_motion:
                agent_state = self.current_agent_state_checked()
                self.emit_local_snap_alignment_diagnostic(agent_state)
                self.stop_reason = "diagnostic_no_motion"
                return

            self.stop_reason = "running"

            while rclpy.ok():
                elapsed_wall = time.monotonic() - self.runner_started_wall
                stop_reason = self._stop_condition(elapsed_wall)
                if stop_reason is not None:
                    self.stop_reason = stop_reason
                    break

                if self._should_emit_local_snap_diagnostic():
                    self.emit_local_snap_alignment_diagnostic(self.current_agent_state_checked())

                action_idx, valid, q, cum_map, agent_state, state_meta, update_result = self.infer_action()
                target_state, (tx, ty) = self._target_for_action(action_idx, agent_state)
                x, y, yaw, _ = self.pose_xy_yaw_time()
                coverage = float(cum_map.coverage_rate)
                frontier = int(np.count_nonzero(cum_map.get_frontier_u8() > 0))
                known = int(np.count_nonzero(cum_map.map != self.INVISIBLE))
                self._refresh_progress(coverage)

                q_values = [float(q[i]) for i in range(len(ACTION_NAMES))]
                q_str = " ".join(
                    f"{idx}:{ACTION_NAMES[idx]}={value:+.3f}{'*' if idx == action_idx else ''}"
                    for idx, value in enumerate(q_values)
                )
                self.get_logger().info(
                    "bridge_step_plan "
                    f"step={self.step_count + 1}/{self.max_steps} "
                    f"agent_state={agent_state} target_state={target_state} "
                    f"odom_xy=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):+.1f}deg "
                    f"target_xy=({tx:.3f},{ty:.3f}) "
                    f"action_idx={action_idx} action_name={ACTION_NAMES[action_idx]} "
                    f"valid_actions={list(valid)} q_values=[{q_str}] "
                    f"coverage={coverage:.4f} best_coverage={self.best_coverage:.4f} "
                    f"known={known} frontier={frontier} no_progress_count={self.no_progress_count} "
                    f"belief_update={update_result} "
                    f"state_meta_keys={sorted(str(k) for k in state_meta.keys())}"
                )

                reached = self.execute_target_cell(action_idx, agent_state)
                self.step_count += 1
                if not reached:
                    self.stop_reason = "target_cell_not_reached"
                    break

                if self.step_pause_sec > 0.0:
                    pause_until = time.monotonic() + self.step_pause_sec
                    while rclpy.ok() and time.monotonic() < pause_until:
                        rclpy.spin_once(self, timeout_sec=0.05)

            if self.stop_reason == "running":
                self.stop_reason = "rclpy_shutdown"
        except Exception as exc:
            self.stop_reason = f"fatal_exception:{exc}"
            self.stop()
            self.get_logger().error(f"bridge_failure stop_reason={self.stop_reason}")
        finally:
            self.stop()
            self.done = True
            self._log_final_summary()

    def _log_final_summary(self) -> None:
        elapsed_wall = 0.0
        if self.runner_started_wall is not None:
            elapsed_wall = time.monotonic() - self.runner_started_wall
        final_cell = None
        try:
            if self.latest_odom is not None:
                x = float(self.latest_odom.pose.pose.position.x)
                y = float(self.latest_odom.pose.pose.position.y)
                final_cell = self.world_xy_to_grid_rc(x, y)
        except Exception:
            final_cell = None

        self.get_logger().info(
            "bridge_finished "
            f"stop_reason={self.stop_reason} steps={self.step_count} "
            f"best_coverage={self.best_coverage:.4f} no_progress_count={self.no_progress_count} "
            f"elapsed_wall={elapsed_wall:.1f}s final_grid_cell={final_cell}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[DrlStandaloneGazeboBridgeNode] = None
    try:
        node = DrlStandaloneGazeboBridgeNode()
        node.run()
    finally:
        if node is not None:
            if rclpy.ok():
                node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
