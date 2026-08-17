from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


DEFAULT_CHECKPOINT = os.environ.get("DRL_CHECKPOINT_PATH", "")
MAX_LINEAR_SPEED = 0.06
MAX_ANGULAR_SPEED = 0.40

ACTIONS_8 = (
    (-1, 0),    # 0 N: left
    (-1, 1),    # 1 NE: forward-left
    (0, 1),     # 2 E: forward
    (1, 1),     # 3 SE: forward-right
    (1, 0),     # 4 S: right
    (1, -1),    # 5 SW: backward-right
    (0, -1),    # 6 W: backward
    (-1, -1),   # 7 NW: backward-left
)
ACTION_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ALLOWED_ACTIONS = {1, 2, 3}
ALL_ACTIONS = set(range(8))
ACTION_BASE_YAW = {
    0: math.radians(90.0),
    1: math.radians(45.0),
    2: 0.0,
    3: math.radians(-45.0),
    4: math.radians(-90.0),
    5: math.radians(-135.0),
    6: math.pi,
    7: math.radians(135.0),
}
DIAGNOSTIC_SECTORS = {
    "front_sector_min": 0.0,
    "front_left_sector_min": math.radians(45.0),
    "front_right_sector_min": math.radians(-45.0),
    "left_sector_min": math.radians(90.0),
    "right_sector_min": math.radians(-90.0),
    "rear_sector_min": math.pi,
}

INVISIBLE = -1
EMPTY = 0
OBSTACLE = 1


def yaw_from_quat(q) -> float:
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def norm_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def make_twist(vx: float = 0.0, wz: float = 0.0) -> Twist:
    msg = Twist()
    msg.linear.x = float(vx)
    msg.angular.z = float(wz)
    return msg


def format_optional_float(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "None"
    return f"{value:.{digits}f}"


def checkpoint_repo_dir(checkpoint_path: str) -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))


def load_policy_model(checkpoint_path: str):
    drl_repo = checkpoint_repo_dir(checkpoint_path)
    if drl_repo not in sys.path:
        sys.path.insert(0, drl_repo)

    import torch
    from agents.q_value_agent import ExplorationQNetwork, StateTensorAdapter

    model = ExplorationQNetwork()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["online_state_dict"])
    model.eval()
    adapter = StateTensorAdapter(device="cpu")
    return model, adapter, torch


class RealcarPolicySafeRunner(Node):
    def __init__(self) -> None:
        super().__init__("realcar_policy_safe_runner_node")

        self.declare_parameter("execute", False)
        self.declare_parameter("max_steps", 1)
        self.declare_parameter("checkpoint_path", DEFAULT_CHECKPOINT)
        self.declare_parameter("step_distance", 0.10)
        self.declare_parameter("linear_speed", 0.03)
        self.declare_parameter("rotate_max_w", 0.25)
        self.declare_parameter("rotate_timeout_sec", 8.0)
        self.declare_parameter("rotate_yaw_tol_deg", 10.0)
        self.declare_parameter("drive_timeout_sec", 0.0)
        self.declare_parameter("drive_timeout_margin_sec", 6.0)
        self.declare_parameter("drive_timeout_scale", 1.8)
        self.declare_parameter("target_pos_tol", 0.025)
        self.declare_parameter("front_min_dist", 0.45)
        self.declare_parameter("laser_yaw_in_base", 0.0)
        self.declare_parameter("decision_timeout_sec", 10.0)
        self.declare_parameter("scan_timeout_sec", 0.5)
        self.declare_parameter("odom_timeout_sec", 0.5)
        self.declare_parameter("sensor_future_tolerance_sec", 0.1)
        self.declare_parameter("full_action_mode", False)
        self.declare_parameter("allowed_actions_mode", "front3")
        self.declare_parameter("diagonal_mode", "grid_center")
        self.declare_parameter("min_sector_dist", 0.25)

        self.execute = bool(self.get_parameter("execute").value)
        self.max_steps = int(self.get_parameter("max_steps").value)
        self.checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        self.step_distance = float(self.get_parameter("step_distance").value)
        self.linear_speed = min(
            float(self.get_parameter("linear_speed").value),
            MAX_LINEAR_SPEED,
        )
        self.rotate_max_w = min(
            float(self.get_parameter("rotate_max_w").value),
            MAX_ANGULAR_SPEED,
        )
        self.rotate_timeout_sec = float(self.get_parameter("rotate_timeout_sec").value)
        self.rotate_yaw_tol_deg = float(self.get_parameter("rotate_yaw_tol_deg").value)
        self.drive_timeout_sec = float(self.get_parameter("drive_timeout_sec").value)
        self.drive_timeout_margin_sec = float(self.get_parameter("drive_timeout_margin_sec").value)
        self.drive_timeout_scale = float(self.get_parameter("drive_timeout_scale").value)
        self.target_pos_tol = float(self.get_parameter("target_pos_tol").value)
        self.front_min_dist = float(self.get_parameter("front_min_dist").value)
        self.laser_yaw_in_base = float(self.get_parameter("laser_yaw_in_base").value)
        self.decision_timeout_sec = float(self.get_parameter("decision_timeout_sec").value)
        self.scan_timeout_sec = float(self.get_parameter("scan_timeout_sec").value)
        self.odom_timeout_sec = float(self.get_parameter("odom_timeout_sec").value)
        self.sensor_future_tolerance_sec = float(
            self.get_parameter("sensor_future_tolerance_sec").value
        )
        self.full_action_mode = bool(self.get_parameter("full_action_mode").value)
        self.allowed_actions_mode = str(self.get_parameter("allowed_actions_mode").value)
        self.diagonal_mode = str(self.get_parameter("diagonal_mode").value)
        self.min_sector_dist = float(self.get_parameter("min_sector_dist").value)
        self.cell_size = 0.35

        if self.allowed_actions_mode not in ("front3", "all8"):
            raise ValueError("allowed_actions_mode must be 'front3' or 'all8'")

        if self.diagonal_mode not in ("grid_center", "constant_length"):
            raise ValueError("diagonal_mode must be 'grid_center' or 'constant_length'")

        self.all8_action_mode = self.full_action_mode or self.allowed_actions_mode == "all8"
        self.allowed_actions = ALL_ACTIONS if self.all8_action_mode else ALLOWED_ACTIONS

        if not self.checkpoint_path:
            raise ValueError(
                "checkpoint_path is required (or set DRL_CHECKPOINT_PATH)"
            )
        if self.max_steps < 0:
            raise ValueError("max_steps must be >= 0")
        max_step_distance = 0.35 if self.all8_action_mode else 0.25
        if self.step_distance <= 0.0 or self.step_distance > max_step_distance:
            raise ValueError(
                "step_distance must be in "
                f"(0, {max_step_distance:.2f}] for this realcar runner mode"
            )
        if self.linear_speed <= 0.0:
            raise ValueError("linear_speed must be > 0")
        if self.rotate_max_w <= 0.0:
            raise ValueError("rotate_max_w must be > 0")
        if self.rotate_timeout_sec <= 0.0:
            raise ValueError("rotate_timeout_sec must be > 0")
        if self.rotate_yaw_tol_deg <= 0.0:
            raise ValueError("rotate_yaw_tol_deg must be > 0")
        if self.drive_timeout_margin_sec < 0.0:
            raise ValueError("drive_timeout_margin_sec must be >= 0")
        if self.drive_timeout_scale <= 0.0:
            raise ValueError("drive_timeout_scale must be > 0")
        if self.target_pos_tol <= 0.0:
            raise ValueError("target_pos_tol must be > 0")
        if self.front_min_dist <= 0.0:
            raise ValueError("front_min_dist must be > 0")
        if self.min_sector_dist <= 0.0:
            raise ValueError("min_sector_dist must be > 0")
        if self.decision_timeout_sec <= 0.0:
            raise ValueError("decision_timeout_sec must be > 0")
        if self.scan_timeout_sec <= 0.0 or self.odom_timeout_sec <= 0.0:
            raise ValueError("sensor timeout parameters must be > 0")
        if self.sensor_future_tolerance_sec < 0.0:
            raise ValueError("sensor_future_tolerance_sec must be >= 0")

        if self.max_steps > 1 and not math.isclose(
            self.step_distance,
            self.cell_size,
            rel_tol=0.01,
            abs_tol=0.005,
        ):
            warning = (
                "MULTI-STEP CONFIGURATION BLOCKED: "
                f"max_steps={self.max_steps}, "
                f"step_distance={self.step_distance:.3f}m, "
                f"DRL cell_size={self.cell_size:.3f}m. "
                "The physical step and DRL state update would diverge; "
                "set max_steps:=1 or make step_distance match cell_size."
            )
            self.get_logger().warn(warning)
            raise ValueError(warning)

        self.scan_radius_cells = 10
        self.local_size = 2 * self.scan_radius_cells + 1
        self.center = self.scan_radius_cells

        self.rotate_kp = 1.2
        self.rotate_min_w = 0.08
        self.rotate_tol = math.radians(self.rotate_yaw_tol_deg)
        self.rotate_wall_timeout = self.rotate_timeout_sec
        self.control_debug_period = 0.5

        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Odometry] = None
        self.latest_scan_received_at: Optional[float] = None
        self.latest_odom_received_at: Optional[float] = None

        self.sensor_cb_group = MutuallyExclusiveCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_cb,
            10,
            callback_group=self.sensor_cb_group,
        )
        self.create_subscription(
            Odometry,
            "/odom",
            self.odom_cb,
            10,
            callback_group=self.sensor_cb_group,
        )

        self.get_logger().warn(
            "realcar_policy_safe_runner_node startup "
            f"execute={self.execute} "
            f"max_steps={self.max_steps} "
            f"checkpoint_path={self.checkpoint_path} "
            f"full_action_mode={self.full_action_mode} "
            f"allowed_actions_mode={self.allowed_actions_mode} "
            f"allowed_actions={sorted(self.allowed_actions)} "
            f"diagonal_mode={self.diagonal_mode} "
            f"step_distance={self.step_distance:.3f} "
            f"linear_speed={self.linear_speed:.3f} "
            f"rotate_max_w={self.rotate_max_w:.3f} "
            f"rotate_timeout_sec={self.rotate_timeout_sec:.3f} "
            f"rotate_yaw_tol_deg={self.rotate_yaw_tol_deg:.3f} "
            f"drive_timeout_sec={self.drive_timeout_sec:.3f} "
            f"drive_timeout_scale={self.drive_timeout_scale:.3f} "
            f"drive_timeout_margin_sec={self.drive_timeout_margin_sec:.3f} "
            f"target_pos_tol={self.target_pos_tol:.3f} "
            f"front_min_dist={self.front_min_dist:.3f} "
            f"min_sector_dist={self.min_sector_dist:.3f} "
            f"laser_yaw_in_base={self.laser_yaw_in_base:.3f} "
            f"scan_timeout_sec={self.scan_timeout_sec:.3f} "
            f"odom_timeout_sec={self.odom_timeout_sec:.3f}"
        )

    def scan_cb(self, msg: LaserScan) -> None:
        self.latest_scan = msg
        self.latest_scan_received_at = time.monotonic()

    def odom_cb(self, msg: Odometry) -> None:
        self.latest_odom = msg
        self.latest_odom_received_at = time.monotonic()

    def require_fresh_sensor(
        self,
        name: str,
        msg,
        received_at: Optional[float],
        timeout_sec: float,
    ) -> None:
        if msg is None or received_at is None:
            raise RuntimeError(f"missing {name} data")

        receive_age = time.monotonic() - received_at
        if receive_age > timeout_sec:
            raise RuntimeError(
                f"{name} stopped updating: receive_age={receive_age:.3f}s "
                f"> timeout={timeout_sec:.3f}s"
            )

        stamp_sec = stamp_to_sec(msg.header.stamp)
        if stamp_sec <= 0.0:
            raise RuntimeError(f"{name} has an invalid zero timestamp")

        stamp_age = self.get_clock().now().nanoseconds * 1e-9 - stamp_sec
        if stamp_age > timeout_sec:
            raise RuntimeError(
                f"{name} timestamp is stale: age={stamp_age:.3f}s "
                f"> timeout={timeout_sec:.3f}s"
            )
        if stamp_age < -self.sensor_future_tolerance_sec:
            raise RuntimeError(
                f"{name} timestamp is in the future: age={stamp_age:.3f}s"
            )

    def require_fresh_inputs(self) -> None:
        self.require_fresh_sensor(
            "/scan",
            self.latest_scan,
            self.latest_scan_received_at,
            self.scan_timeout_sec,
        )
        self.require_fresh_sensor(
            "/odom",
            self.latest_odom,
            self.latest_odom_received_at,
            self.odom_timeout_sec,
        )

    def publish_velocity(self, vx: float = 0.0, wz: float = 0.0) -> None:
        if vx != 0.0 or wz != 0.0:
            self.require_fresh_inputs()
        requested_vx = float(vx)
        requested_wz = float(wz)
        if not math.isfinite(requested_vx) or not math.isfinite(requested_wz):
            raise RuntimeError("refusing to publish non-finite velocity")
        limited_vx = max(-MAX_LINEAR_SPEED, min(MAX_LINEAR_SPEED, requested_vx))
        limited_wz = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, requested_wz))
        self.cmd_pub.publish(make_twist(limited_vx, limited_wz))

    def pose_xy_yaw_time(self) -> tuple[float, float, float, float]:
        if self.latest_odom is None:
            raise RuntimeError("latest_odom is None")
        p = self.latest_odom.pose.pose.position
        q = self.latest_odom.pose.pose.orientation
        return (
            float(p.x),
            float(p.y),
            yaw_from_quat(q),
            stamp_to_sec(self.latest_odom.header.stamp),
        )

    def stop(self, repeat: int = 10) -> None:
        for _ in range(repeat):
            if not rclpy.ok():
                return
            self.publish_velocity()
            time.sleep(0.02)

    def mark_cell(self, snap: np.ndarray, dr: int, dc: int, value: int) -> None:
        lr = self.center + int(dr)
        lc = self.center + int(dc)

        if not (0 <= lr < self.local_size and 0 <= lc < self.local_size):
            return

        if dr * dr + dc * dc > self.scan_radius_cells * self.scan_radius_cells:
            return

        if value == OBSTACLE:
            snap[lr, lc] = OBSTACLE
        elif snap[lr, lc] == INVISIBLE:
            snap[lr, lc] = EMPTY

    def ray_to_local_cells(self, angle_world: float, dist: float, hit_obstacle: bool):
        step = self.cell_size / 3.0
        local_radius_m = self.scan_radius_cells * self.cell_size
        max_d = min(max(0.0, float(dist)), local_radius_m + self.cell_size)
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
                yield dr, dc, EMPTY
            d += step

        if hit_obstacle and dist <= local_radius_m + self.cell_size:
            rel_x = max_d * math.cos(angle_world)
            rel_y = max_d * math.sin(angle_world)
            dc = int(round(rel_x / self.cell_size))
            dr = int(round(-rel_y / self.cell_size))
            yield dr, dc, OBSTACLE

    def build_local_snap(self, scan: LaserScan, odom: Odometry) -> np.ndarray:
        robot_yaw = yaw_from_quat(odom.pose.pose.orientation)
        snap = np.full((self.local_size, self.local_size), INVISIBLE, dtype=np.int8)
        snap[self.center, self.center] = EMPTY

        for i, raw_r in enumerate(scan.ranges):
            scan_angle = scan.angle_min + i * scan.angle_increment
            if (
                math.isfinite(raw_r)
                and float(scan.range_min) <= float(raw_r) <= float(scan.range_max)
            ):
                dist = float(raw_r)
                hit_obstacle = True
            else:
                dist = float(scan.range_max)
                hit_obstacle = False

            angle_world = robot_yaw + self.laser_yaw_in_base + scan_angle
            for dr, dc, value in self.ray_to_local_cells(angle_world, dist, hit_obstacle):
                self.mark_cell(snap, dr, dc, value)

        return snap

    def valid_scan_range(self, scan: LaserScan, raw_r: float) -> bool:
        return (
            math.isfinite(raw_r)
            and float(raw_r) > 0.0
            and float(scan.range_min) <= float(raw_r) <= float(scan.range_max)
        )

    def scan_stats(self, scan: LaserScan) -> dict[str, Any]:
        valid_ranges: list[float] = []
        finite_count = 0
        nan_count = 0
        inf_count = 0
        zero_or_negative_count = 0

        for raw_r in scan.ranges:
            r = float(raw_r)
            if math.isnan(r):
                nan_count += 1
                continue
            if math.isinf(r):
                inf_count += 1
                continue
            if math.isfinite(r):
                finite_count += 1
                if r <= 0.0:
                    zero_or_negative_count += 1
                if float(scan.range_min) <= r <= float(scan.range_max):
                    valid_ranges.append(r)

        valid_count = len(valid_ranges)
        if valid_count:
            valid_min: Optional[float] = min(valid_ranges)
            valid_max: Optional[float] = max(valid_ranges)
            valid_mean: Optional[float] = float(sum(valid_ranges) / valid_count)
        else:
            valid_min = None
            valid_max = None
            valid_mean = None

        return {
            "scan_count": len(scan.ranges),
            "range_min_msg": float(scan.range_min),
            "range_max_msg": float(scan.range_max),
            "finite_count": finite_count,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "zero_or_negative_count": zero_or_negative_count,
            "valid_count": valid_count,
            "valid_min": valid_min,
            "valid_max": valid_max,
            "valid_mean": valid_mean,
        }

    def sector_min_dist(
        self,
        scan: LaserScan,
        center_base_yaw: float,
        half_width: float,
    ) -> Optional[float]:
        observed_min: Optional[float] = None

        for i, raw_r in enumerate(scan.ranges):
            scan_angle = scan.angle_min + i * scan.angle_increment
            base_angle = norm_angle(self.laser_yaw_in_base + scan_angle)
            if abs(norm_angle(base_angle - center_base_yaw)) > half_width:
                continue

            r = float(raw_r)
            if self.valid_scan_range(scan, r):
                observed_min = r if observed_min is None else min(observed_min, r)

        return observed_min

    def sector_diagnostics(self, scan: LaserScan) -> dict[str, Optional[float]]:
        half_width = math.radians(22.5)
        return {
            name: self.sector_min_dist(scan, center_yaw, half_width)
            for name, center_yaw in DIAGNOSTIC_SECTORS.items()
        }

    def action_sector_min(self, action_idx: int, scan: LaserScan) -> Optional[float]:
        half_width = math.radians(22.5)
        return self.sector_min_dist(scan, ACTION_BASE_YAW[action_idx], half_width)

    def lidar_gate(self, action_idx: int, scan: LaserScan) -> tuple[bool, Optional[float]]:
        action_min = self.action_sector_min(action_idx, scan)
        if self.all8_action_mode:
            action_passed = action_min is not None and action_min >= self.min_sector_dist
            return action_passed, action_min

        half_width = math.radians(22.5)
        front_min = self.sector_min_dist(scan, 0.0, half_width)
        front_passed = front_min is not None and front_min >= self.front_min_dist
        action_passed = action_min is not None and action_min >= self.front_min_dist
        return front_passed and action_passed, action_min

    def choose_first_lidar_passed_action(
        self,
        q_ranked: list[tuple[int, float]],
        scan: LaserScan,
    ) -> tuple[Optional[int], Optional[float]]:
        for action_idx, _q_value in q_ranked:
            if action_idx not in self.allowed_actions:
                continue

            selected_sector_min = self.action_sector_min(action_idx, scan)
            if selected_sector_min is not None and selected_sector_min >= self.min_sector_dist:
                return action_idx, selected_sector_min

        return None, None

    def wait_for_inputs(self) -> None:
        deadline = time.monotonic() + self.decision_timeout_sec
        last_error = "no messages received"
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_scan is not None and self.latest_odom is not None:
                try:
                    self.require_fresh_inputs()
                    return
                except RuntimeError as exc:
                    last_error = str(exc)
        raise TimeoutError(f"timeout waiting for fresh /scan and /odom: {last_error}")

    def target_for_action(
        self,
        action_idx: int,
        x0: float,
        y0: float,
        yaw0: float,
    ) -> tuple[float, float, float, float]:
        dr, dc = ACTIONS_8[action_idx]
        component_distance = self.step_distance
        if self.diagonal_mode == "constant_length" and dr != 0 and dc != 0:
            component_distance = self.step_distance / math.sqrt(2.0)

        rel_forward = float(dc) * component_distance
        rel_left = float(-dr) * component_distance

        tx = x0 + rel_forward * math.cos(yaw0) - rel_left * math.sin(yaw0)
        ty = y0 + rel_forward * math.sin(yaw0) + rel_left * math.cos(yaw0)
        return tx, ty, rel_forward, rel_left

    def execute_target(self, tx: float, ty: float) -> float:
        x0, y0, _yaw0, _ = self.pose_xy_yaw_time()
        target_dist = math.hypot(tx - x0, ty - y0)
        if self.drive_timeout_sec > 0.0:
            effective_drive_timeout = self.drive_timeout_sec
        else:
            nominal_time = target_dist / max(self.linear_speed, 1.0e-6)
            effective_drive_timeout = max(
                8.0,
                nominal_time * self.drive_timeout_scale + self.drive_timeout_margin_sec,
            )

        self.get_logger().warn("EXECUTE START: publishing /cmd_vel at low speed.")

        self.get_logger().info("rotate phase start")
        rotate_wall_start = time.monotonic()
        last_debug = -1.0e9
        stable_count = 0

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            x, y, yaw, _ = self.pose_xy_yaw_time()
            target_yaw = math.atan2(ty - y, tx - x)
            err = norm_angle(target_yaw - yaw)
            wall_elapsed = time.monotonic() - rotate_wall_start

            if abs(err) < self.rotate_tol:
                stable_count += 1
                if stable_count >= 5:
                    break
            else:
                stable_count = 0

            if wall_elapsed > self.rotate_wall_timeout:
                raise RuntimeError(f"rotate timeout, yaw_err={math.degrees(err):.1f}deg")

            wz = max(-self.rotate_max_w, min(self.rotate_max_w, self.rotate_kp * err))
            if abs(wz) < self.rotate_min_w:
                wz = math.copysign(self.rotate_min_w, wz)

            if wall_elapsed - last_debug >= self.control_debug_period:
                last_debug = wall_elapsed
                self.get_logger().info(
                    f"rotate_debug xy=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):+.1f}deg "
                    f"target_yaw={math.degrees(target_yaw):+.1f}deg "
                    f"err={math.degrees(err):+.1f}deg wz={wz:+.3f}"
                )

            self.publish_velocity(0.0, wz)

        self.stop()

        self.get_logger().info(
            "drive phase start "
            f"target_dist={target_dist:.3f} "
            f"linear_speed={self.linear_speed:.3f} "
            f"effective_drive_timeout={effective_drive_timeout:.3f} "
            f"drive_timeout_sec={self.drive_timeout_sec:.3f} "
            f"drive_timeout_scale={self.drive_timeout_scale:.3f} "
            f"drive_timeout_margin_sec={self.drive_timeout_margin_sec:.3f}"
        )
        drive_wall_start = time.monotonic()
        last_debug = -1.0e9

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            x, y, yaw, _ = self.pose_xy_yaw_time()
            dist = math.hypot(tx - x, ty - y)
            wall_elapsed = time.monotonic() - drive_wall_start

            if dist <= self.target_pos_tol:
                break

            if wall_elapsed > effective_drive_timeout:
                raise RuntimeError(
                    f"drive timeout, dist={dist:.3f}m "
                    f"target_dist={target_dist:.3f}m "
                    f"effective_drive_timeout={effective_drive_timeout:.3f}s"
                )

            target_yaw = math.atan2(ty - y, tx - x)
            yaw_err = norm_angle(target_yaw - yaw)
            wz = max(-0.12, min(0.12, 0.8 * yaw_err))

            if wall_elapsed - last_debug >= self.control_debug_period:
                last_debug = wall_elapsed
                self.get_logger().info(
                    f"drive_debug xy=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):+.1f}deg "
                    f"target=({tx:.3f},{ty:.3f}) dist={dist:.3f} "
                    f"target_dist={target_dist:.3f} "
                    f"effective_drive_timeout={effective_drive_timeout:.3f} "
                    f"vx={self.linear_speed:+.3f} wz={wz:+.3f}"
                )

            self.publish_velocity(self.linear_speed, wz)

        self.stop()
        x1, y1, yaw1, _ = self.pose_xy_yaw_time()
        final_dist = math.hypot(tx - x1, ty - y1)
        self.get_logger().warn(
            f"EXECUTE DONE final_xy=({x1:.3f},{y1:.3f}) "
            f"final_yaw={math.degrees(yaw1):+.1f}deg "
            f"target_xy=({tx:.3f},{ty:.3f}) final_dist={final_dist:.3f}m"
        )
        return final_dist

    def log_step(
        self,
        step_id: int,
        raw_action_idx: int,
        q_list: list[float],
        q_ranked: list[tuple[int, float]],
        best_allowed_action_idx: int,
        best_allowed_action_q: float,
        scan_stats: dict[str, Any],
        sector_mins: dict[str, Optional[float]],
        fallback_used: bool,
        action_filter_passed: bool,
        lidar_gate_passed: bool,
        observed_min_dist: Optional[float],
        selected_sector_min: Optional[float],
        selected_action_idx: Optional[int],
        final_dist: Optional[float],
        node_exit_reason: str,
    ) -> None:
        final_dist_text = "NA" if final_dist is None else f"{final_dist:.3f}"
        scan_stats_text = (
            f"scan_count={scan_stats['scan_count']} "
            f"range_min_msg={scan_stats['range_min_msg']:.3f} "
            f"range_max_msg={scan_stats['range_max_msg']:.3f} "
            f"finite_count={scan_stats['finite_count']} "
            f"nan_count={scan_stats['nan_count']} "
            f"inf_count={scan_stats['inf_count']} "
            f"zero_or_negative_count={scan_stats['zero_or_negative_count']} "
            f"valid_count={scan_stats['valid_count']} "
            f"valid_min={format_optional_float(scan_stats['valid_min'])} "
            f"valid_max={format_optional_float(scan_stats['valid_max'])} "
            f"valid_mean={format_optional_float(scan_stats['valid_mean'])}"
        )
        sector_text = " ".join(
            f"{name}={format_optional_float(value)}"
            for name, value in sector_mins.items()
        )
        self.get_logger().warn(f"scan_diagnostics step_id={step_id} {scan_stats_text}")
        self.get_logger().warn(f"sector_diagnostics step_id={step_id} {sector_text}")
        self.get_logger().warn(
            "q_diagnostics "
            f"step_id={step_id} "
            f"q_ranked={q_ranked} "
            f"best_allowed_action_idx={best_allowed_action_idx} "
            f"best_allowed_action_q={best_allowed_action_q:.4f} "
            f"fallback_used={str(fallback_used).lower()}"
        )
        self.get_logger().warn(
            "policy_safe_step "
            f"step_id={step_id} "
            f"full_action_mode={self.full_action_mode} "
            f"allowed_actions={sorted(self.allowed_actions)} "
            f"diagonal_mode={self.diagonal_mode} "
            f"raw_action_idx={raw_action_idx} "
            f"selected_action_idx={selected_action_idx} "
            f"best_allowed_action_idx={best_allowed_action_idx} "
            f"best_allowed_action_q={best_allowed_action_q:.4f} "
            f"fallback_used={str(fallback_used).lower()} "
            f"q_values={q_list} "
            f"action_filter_passed={str(action_filter_passed).lower()} "
            f"lidar_gate_passed={str(lidar_gate_passed).lower()} "
            f"observed_min_dist={format_optional_float(observed_min_dist)} "
            f"selected_sector_min={format_optional_float(selected_sector_min)} "
            f"min_sector_dist={self.min_sector_dist:.3f} "
            f"execute={self.execute} "
            f"final_dist={final_dist_text} "
            f"node_exit_reason={node_exit_reason}"
        )

    def run(self) -> str:
        if self.max_steps == 0:
            self.stop()
            self.get_logger().warn("node_exit_reason=max_steps_zero")
            return "max_steps_zero"

        self.wait_for_inputs()
        model, adapter, torch = load_policy_model(self.checkpoint_path)

        from env.core_cummap import CumulativeBeliefMap

        true_grid = np.zeros((120, 120), dtype=np.int8)
        agent_state = (60, 60)
        recent_positions = [agent_state]
        cum_map = None

        for step_id in range(self.max_steps):
            self.wait_for_inputs()
            scan = self.latest_scan
            odom = self.latest_odom
            if scan is None or odom is None:
                raise RuntimeError("missing /scan or /odom during decision")

            scan_stats = self.scan_stats(scan)
            sector_mins = self.sector_diagnostics(scan)
            snap = self.build_local_snap(scan, odom)
            if cum_map is None:
                cum_map = CumulativeBeliefMap(true_grid, agent_state, snap)
            else:
                cum_map.update(agent_state, snap)

            state_batch, _state_meta = adapter.build_single_state_tensors(
                cum_map,
                agent_state,
                recent_trajectory_positions=recent_positions,
                return_state_meta=True,
            )
            with torch.inference_mode():
                q_values = model(**state_batch, return_aux=False)

            q_np = q_values.detach().cpu().numpy()[0]
            q_list = [round(float(v), 4) for v in q_np.tolist()]
            q_ranked = sorted(
                [(idx, round(float(value), 4)) for idx, value in enumerate(q_np.tolist())],
                key=lambda item: item[1],
                reverse=True,
            )
            best_allowed_action_idx, best_allowed_action_q = max(
                ((idx, float(q_np[idx])) for idx in sorted(self.allowed_actions)),
                key=lambda item: item[1],
            )
            raw_action_idx = int(torch.argmax(q_values, dim=1).item())

            raw_action_allowed = raw_action_idx in self.allowed_actions
            fallback_used = False
            action_filter_passed = raw_action_allowed
            lidar_gate_passed = False
            observed_min_dist: Optional[float] = None
            selected_sector_min: Optional[float] = None
            selected_action_idx: Optional[int] = None
            final_dist: Optional[float] = None

            if raw_action_allowed:
                selected_action_idx = raw_action_idx

            if not action_filter_passed:
                self.stop()
                self.log_step(
                    step_id,
                    raw_action_idx,
                    q_list,
                    q_ranked,
                    best_allowed_action_idx,
                    best_allowed_action_q,
                    scan_stats,
                    sector_mins,
                    fallback_used,
                    action_filter_passed,
                    lidar_gate_passed,
                    observed_min_dist,
                    selected_sector_min,
                    selected_action_idx,
                    final_dist,
                    "blocked_by_action_filter",
                )
                return "blocked_by_action_filter"

            if selected_action_idx not in self.allowed_actions:
                raise RuntimeError(f"selected_action_idx is not allowed: {selected_action_idx}")

            lidar_gate_passed, observed_min_dist = self.lidar_gate(selected_action_idx, scan)
            selected_sector_min = observed_min_dist

            if self.all8_action_mode and not lidar_gate_passed:
                next_action_idx, next_sector_min = (
                    self.choose_first_lidar_passed_action(q_ranked, scan)
                )
                if next_action_idx is not None:
                    selected_action_idx = next_action_idx
                    selected_sector_min = next_sector_min
                    observed_min_dist = next_sector_min
                    lidar_gate_passed = True
                    fallback_used = selected_action_idx != raw_action_idx

            if not lidar_gate_passed:
                self.stop()
                self.log_step(
                    step_id,
                    raw_action_idx,
                    q_list,
                    q_ranked,
                    best_allowed_action_idx,
                    best_allowed_action_q,
                    scan_stats,
                    sector_mins,
                    fallback_used,
                    action_filter_passed,
                    lidar_gate_passed,
                    observed_min_dist,
                    selected_sector_min,
                    selected_action_idx,
                    final_dist,
                    "blocked_by_lidar_gate",
                )
                return "blocked_by_lidar_gate"

            node_exit_reason = "dry_plan_complete"
            if self.execute:
                if self.cmd_pub.get_subscription_count() < 1:
                    raise RuntimeError("No /cmd_vel subscriber found")

                x, y, yaw, _ = self.pose_xy_yaw_time()
                tx, ty, rel_forward, rel_left = self.target_for_action(
                    selected_action_idx,
                    x,
                    y,
                    yaw,
                )
                self.get_logger().warn(
                    "execute_plan "
                    f"step_id={step_id} "
                    f"action={selected_action_idx}:{ACTION_NAMES[selected_action_idx]} "
                    f"start_xy=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):+.1f}deg "
                    f"rel_forward={rel_forward:+.3f} rel_left={rel_left:+.3f} "
                    f"target_xy=({tx:.3f},{ty:.3f})"
                )
                final_dist = self.execute_target(tx, ty)
                self.stop()
                dr, dc = ACTIONS_8[selected_action_idx]
                agent_state = (agent_state[0] + dr, agent_state[1] + dc)
                recent_positions.append(agent_state)
                recent_positions = recent_positions[-8:]
                node_exit_reason = "step_executed"
            else:
                self.stop()

            if step_id == self.max_steps - 1:
                node_exit_reason = "max_steps_reached"

            self.log_step(
                step_id,
                raw_action_idx,
                q_list,
                q_ranked,
                best_allowed_action_idx,
                best_allowed_action_q,
                scan_stats,
                sector_mins,
                fallback_used,
                action_filter_passed,
                lidar_gate_passed,
                observed_min_dist,
                selected_sector_min,
                selected_action_idx,
                final_dist,
                node_exit_reason,
            )

            if not self.execute:
                time.sleep(0.2)

        return "max_steps_reached"


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RealcarPolicySafeRunner()
    exit_reason = "unknown"
    try:
        exit_reason = node.run()
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
        node.get_logger().warn("KeyboardInterrupt received; sending stop command.")
    except Exception as exc:
        exit_reason = "error"
        node.get_logger().error(f"FAIL: {exc}")
        node.get_logger().error("A stop command has been sent.")
    finally:
        try:
            node.stop()
            node.get_logger().warn(f"node_exit_reason={exit_reason}")
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
