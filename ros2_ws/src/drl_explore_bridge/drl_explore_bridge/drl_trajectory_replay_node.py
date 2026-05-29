from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


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


def expand_path(raw_path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(raw_path))))


def is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def parse_float(value: Any) -> float:
    return float(str(value).strip())


def parse_int(value: Any) -> int:
    return int(float(str(value).strip()))


class DrlTrajectoryReplayNode(Node):
    """Replay an offline oracle/ideal trajectory as Gazebo waypoints."""

    def __init__(self) -> None:
        super().__init__("drl_trajectory_replay_node")
        self._declare_parameters()
        self._read_parameters()
        self._validate_parameters()

        self.latest_odom: Optional[Odometry] = None
        self.stop_reason = "not_started"
        self.runner_started_wall: Optional[float] = None
        self.reached_waypoints = 0
        self.skipped_waypoints = 0
        self.failed_waypoint_idx: Optional[int] = None
        self.waypoint_logs: list[dict[str, Any]] = []
        self.done = False

        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.waypoints = self._load_waypoints(self.trajectory_csv)

        self.get_logger().info(
            "trajectory_replay_startup "
            f"trajectory_csv={self.trajectory_csv} source_fields={self.waypoint_source_fields} "
            f"waypoints={len(self.waypoints)} start_index={self.start_index} "
            f"max_waypoints={self.max_waypoints} cell_size={self.cell_size:.3f} "
            f"rows={self.rows} cols={self.cols} world_x={self.world_x:.3f} world_y={self.world_y:.3f} "
            f"linear_speed={self.linear_speed:.3f} target_pos_tol={self.target_pos_tol:.3f} "
            f"diagnostic_no_motion={self.diagnostic_no_motion}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter(
            "trajectory_csv",
            "/home/dk/ros2_repos/ROS2/assets/cell035/trajectories/cell035_oracle_trajectory.csv",
        )
        self.declare_parameter("cell_size", 0.35)
        self.declare_parameter("world_x", 21.0)
        self.declare_parameter("world_y", 14.0)
        self.declare_parameter("rows", 40)
        self.declare_parameter("cols", 60)
        self.declare_parameter("start_index", 0)
        self.declare_parameter("max_waypoints", 0)
        self.declare_parameter("linear_speed", 0.10)
        self.declare_parameter("target_pos_tol", 0.07)
        self.declare_parameter("rotate_kp", 2.0)
        self.declare_parameter("rotate_max_w", 2.0)
        self.declare_parameter("rotate_min_w", 0.20)
        self.declare_parameter("rotate_tol_deg", 4.0)
        self.declare_parameter("rotate_sim_timeout", 12.0)
        self.declare_parameter("drive_sim_timeout", 12.0)
        self.declare_parameter("rotate_wall_timeout", 60.0)
        self.declare_parameter("drive_wall_timeout", 60.0)
        self.declare_parameter("multi_wall_timeout", 300.0)
        self.declare_parameter("waypoint_pause_sec", 0.3)
        self.declare_parameter("stop_repeat", 10)
        self.declare_parameter("control_debug_period", 1.0)
        self.declare_parameter("summary_output", "/home/dk/ros2_repos/ROS2/trajectory_replay_summary.json")
        self.declare_parameter("trajectory_log_output", "/home/dk/ros2_repos/ROS2/trajectory_replay_log.csv")
        self.declare_parameter("diagnostic_no_motion", False)

    def _read_parameters(self) -> None:
        self.trajectory_csv = expand_path(str(self.get_parameter("trajectory_csv").value))
        self.cell_size = float(self.get_parameter("cell_size").value)
        self.world_x = float(self.get_parameter("world_x").value)
        self.world_y = float(self.get_parameter("world_y").value)
        self.rows = int(self.get_parameter("rows").value)
        self.cols = int(self.get_parameter("cols").value)
        self.start_index = int(self.get_parameter("start_index").value)
        self.max_waypoints = int(self.get_parameter("max_waypoints").value)
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
        self.multi_wall_timeout = float(self.get_parameter("multi_wall_timeout").value)
        self.waypoint_pause_sec = float(self.get_parameter("waypoint_pause_sec").value)
        self.stop_repeat = int(self.get_parameter("stop_repeat").value)
        self.control_debug_period = float(self.get_parameter("control_debug_period").value)
        self.summary_output = expand_path(str(self.get_parameter("summary_output").value))
        self.trajectory_log_output = expand_path(str(self.get_parameter("trajectory_log_output").value))
        self.diagnostic_no_motion = bool(self.get_parameter("diagnostic_no_motion").value)

    def _validate_parameters(self) -> None:
        if self.cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive")
        if self.start_index < 0:
            raise ValueError("start_index must be >= 0")
        if self.max_waypoints < 0:
            raise ValueError("max_waypoints must be >= 0")
        if self.linear_speed <= 0.0:
            raise ValueError("linear_speed must be positive")
        if self.target_pos_tol <= 0.0:
            raise ValueError("target_pos_tol must be positive")
        if self.rotate_max_w <= 0.0 or self.rotate_min_w < 0.0:
            raise ValueError("rotate_max_w must be positive and rotate_min_w must be non-negative")
        if self.stop_repeat < 1:
            raise ValueError("stop_repeat must be >= 1")
        if not self.trajectory_csv.exists():
            raise FileNotFoundError(f"trajectory_csv does not exist: {self.trajectory_csv}")

    def odom_cb(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def grid_rc_to_cell_center_xy(self, row: int, col: int) -> tuple[float, float]:
        x = -self.world_x / 2.0 + (int(col) + 0.5) * self.cell_size
        y = self.world_y / 2.0 - (int(row) + 0.5) * self.cell_size
        return float(x), float(y)

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
        if self.diagnostic_no_motion:
            return
        count = self.stop_repeat if repeat is None else int(repeat)
        for _ in range(max(1, count)):
            self.cmd_pub.publish(make_twist(0.0, 0.0))

    def _row_waypoint(self, csv_row: dict[str, Any], csv_index: int) -> dict[str, Any] | None:
        if not is_blank(csv_row.get("target_x")) and not is_blank(csv_row.get("target_y")):
            x = parse_float(csv_row["target_x"])
            y = parse_float(csv_row["target_y"])
            source = "target_x,target_y"
            row = None if is_blank(csv_row.get("target_row")) else parse_int(csv_row.get("target_row"))
            col = None if is_blank(csv_row.get("target_col")) else parse_int(csv_row.get("target_col"))
        elif not is_blank(csv_row.get("x")) and not is_blank(csv_row.get("y")):
            x = parse_float(csv_row["x"])
            y = parse_float(csv_row["y"])
            source = "x,y"
            row = None if is_blank(csv_row.get("row")) else parse_int(csv_row.get("row"))
            col = None if is_blank(csv_row.get("col")) else parse_int(csv_row.get("col"))
        elif not is_blank(csv_row.get("target_row")) and not is_blank(csv_row.get("target_col")):
            row = parse_int(csv_row["target_row"])
            col = parse_int(csv_row["target_col"])
            x, y = self.grid_rc_to_cell_center_xy(row, col)
            source = "target_row,target_col"
        elif not is_blank(csv_row.get("row")) and not is_blank(csv_row.get("col")):
            row = parse_int(csv_row["row"])
            col = parse_int(csv_row["col"])
            x, y = self.grid_rc_to_cell_center_xy(row, col)
            source = "row,col"
        else:
            return None

        return {
            "csv_index": int(csv_index),
            "step": None if is_blank(csv_row.get("step")) else parse_int(csv_row.get("step")),
            "row": row,
            "col": col,
            "x": float(x),
            "y": float(y),
            "source_fields": source,
        }

    def _load_waypoints(self, path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            raw_waypoints = [
                waypoint
                for idx, row in enumerate(reader)
                for waypoint in [self._row_waypoint(row, idx)]
                if waypoint is not None
            ]
        if len(raw_waypoints) <= 0:
            raise RuntimeError(f"trajectory_csv has no usable waypoints: {path}")

        end = None if self.max_waypoints <= 0 else self.start_index + self.max_waypoints
        waypoints = raw_waypoints[self.start_index:end]
        if len(waypoints) <= 0:
            raise RuntimeError(
                f"trajectory selection is empty: total={len(raw_waypoints)} "
                f"start_index={self.start_index} max_waypoints={self.max_waypoints}"
            )

        source_counts: dict[str, int] = {}
        for waypoint in waypoints:
            key = str(waypoint["source_fields"])
            source_counts[key] = int(source_counts.get(key, 0)) + 1
        self.waypoint_source_fields = json.dumps(source_counts, sort_keys=True)
        return waypoints

    def _wait_for_odom(self) -> None:
        self.get_logger().info("trajectory_replay_waiting_for_odom waiting for /odom")
        wait_start = time.monotonic()
        last_log = -1.0e9
        while rclpy.ok() and self.latest_odom is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            elapsed = time.monotonic() - wait_start
            if elapsed - last_log >= 2.0:
                last_log = elapsed
                self.get_logger().warn(f"trajectory_replay_waiting_for_odom elapsed={elapsed:.1f}s")
            if elapsed > self.multi_wall_timeout:
                raise RuntimeError(f"multi_wall_timeout reached while waiting for /odom: elapsed={elapsed:.1f}s")
        if not rclpy.ok():
            raise RuntimeError("rclpy shutdown while waiting for /odom")
        self.get_logger().info("trajectory_replay_waiting_for_odom ready")

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
        self.get_logger().info(f"trajectory_replay_startup cmd_vel_subscribers={subscriber_count}")

    def _skip_initial_waypoint_if_close(self) -> None:
        if len(self.waypoints) <= 0:
            return
        x, y, _yaw, _sim = self.pose_xy_yaw_time()
        first = self.waypoints[0]
        dist = math.hypot(float(first["x"]) - x, float(first["y"]) - y)
        if dist > self.target_pos_tol:
            return
        self.skipped_waypoints += 1
        skipped = self.waypoints.pop(0)
        log_row = {
            "waypoint_idx": int(skipped["csv_index"]),
            "target_x": float(skipped["x"]),
            "target_y": float(skipped["y"]),
            "start_x": float(x),
            "start_y": float(y),
            "final_x": float(x),
            "final_y": float(y),
            "final_dist": float(dist),
            "reached": True,
            "skipped": True,
            "rotate_elapsed_sim": 0.0,
            "drive_elapsed_sim": 0.0,
            "rotate_elapsed_wall": 0.0,
            "drive_elapsed_wall": 0.0,
        }
        self.waypoint_logs.append(log_row)
        self.get_logger().info(
            "skipped_initial_waypoint "
            f"waypoint_idx={int(skipped['csv_index'])} target_xy=({float(skipped['x']):.3f},{float(skipped['y']):.3f}) "
            f"odom_xy=({x:.3f},{y:.3f}) dist={dist:.3f} tol={self.target_pos_tol:.3f}"
        )

    def execute_waypoint(self, waypoint: dict[str, Any]) -> bool:
        waypoint_idx = int(waypoint["csv_index"])
        tx = float(waypoint["x"])
        ty = float(waypoint["y"])
        x0, y0, yaw0, sim0 = self.pose_xy_yaw_time()
        target_yaw = math.atan2(ty - y0, tx - x0)
        initial_dist = math.hypot(tx - x0, ty - y0)

        self.get_logger().info(
            "trajectory_waypoint_control "
            f"waypoint_idx={waypoint_idx} step={waypoint.get('step')} source_fields={waypoint['source_fields']} "
            f"target_rc=({waypoint.get('row')},{waypoint.get('col')}) "
            f"start_xy=({x0:.3f},{y0:.3f}) start_yaw={math.degrees(yaw0):+.1f}deg "
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
                raise RuntimeError(f"rotate_sim_timeout at waypoint {waypoint_idx}, yaw_error={err:.3f} rad")
            if wall_elapsed > self.rotate_wall_timeout:
                raise RuntimeError(
                    f"rotate_wall_timeout at waypoint {waypoint_idx}, "
                    f"sim_elapsed={sim_elapsed:.3f}s yaw_error={err:.3f} rad"
                )

            wz = max(-self.rotate_max_w, min(self.rotate_max_w, self.rotate_kp * err))
            if abs(wz) < self.rotate_min_w:
                wz = math.copysign(self.rotate_min_w, wz)

            if wall_elapsed - last_rotate_debug_wall >= self.control_debug_period:
                last_rotate_debug_wall = wall_elapsed
                self.get_logger().info(
                    "trajectory_rotate_debug "
                    f"waypoint_idx={waypoint_idx} xy=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):+.1f}deg "
                    f"target_yaw={math.degrees(target_yaw_now):+.1f}deg err={math.degrees(err):+.1f}deg "
                    f"wz={wz:+.3f} stable={stable_count} sim_elapsed={sim_elapsed:.3f}s "
                    f"wall_elapsed={wall_elapsed:.3f}s"
                )
            self.cmd_pub.publish(make_twist(0.0, wz))

        self.stop()
        xr, yr, _yawr, simr = self.pose_xy_yaw_time()
        rotate_wall_elapsed = time.monotonic() - rotate_wall_start

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
                raise RuntimeError(f"drive_sim_timeout at waypoint {waypoint_idx}, distance={dist_to_target:.3f} m")
            if wall_elapsed > self.drive_wall_timeout:
                raise RuntimeError(
                    f"drive_wall_timeout at waypoint {waypoint_idx}, "
                    f"sim_elapsed={sim_elapsed:.3f}s distance={dist_to_target:.3f} m"
                )

            target_yaw_now = math.atan2(ty - y, tx - x)
            yaw_err = norm_angle(target_yaw_now - yaw)
            wz_correction = max(-0.3, min(0.3, yaw_err))

            if wall_elapsed - last_drive_debug_wall >= self.control_debug_period:
                last_drive_debug_wall = wall_elapsed
                self.get_logger().info(
                    "trajectory_drive_debug "
                    f"waypoint_idx={waypoint_idx} xy=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):+.1f}deg "
                    f"target_yaw={math.degrees(target_yaw_now):+.1f}deg "
                    f"yaw_err={math.degrees(yaw_err):+.1f}deg dist={dist_to_target:.3f} "
                    f"vx={self.linear_speed:+.3f} wz={wz_correction:+.3f} "
                    f"sim_elapsed={sim_elapsed:.3f}s wall_elapsed={wall_elapsed:.3f}s"
                )
            self.cmd_pub.publish(make_twist(self.linear_speed, wz_correction))

        self.stop()
        x1, y1, yaw1, sim1 = self.pose_xy_yaw_time()
        drive_wall_elapsed = time.monotonic() - drive_wall_start
        final_dist = math.hypot(tx - x1, ty - y1)
        reached = bool(final_dist <= self.target_pos_tol)

        log_row = {
            "waypoint_idx": waypoint_idx,
            "target_x": tx,
            "target_y": ty,
            "start_x": x0,
            "start_y": y0,
            "final_x": x1,
            "final_y": y1,
            "final_dist": final_dist,
            "reached": reached,
            "skipped": False,
            "rotate_elapsed_sim": float(simr - rotate_sim_start),
            "drive_elapsed_sim": float(sim1 - drive_sim_start),
            "rotate_elapsed_wall": float(rotate_wall_elapsed),
            "drive_elapsed_wall": float(drive_wall_elapsed),
        }
        self.waypoint_logs.append(log_row)
        self.get_logger().info(
            "trajectory_waypoint_done "
            f"waypoint_idx={waypoint_idx} final_xy=({x1:.3f},{y1:.3f}) "
            f"final_yaw={math.degrees(yaw1):+.1f}deg final_dist={final_dist:.3f} "
            f"reached={reached} rotate_sim_elapsed={simr - rotate_sim_start:.3f}s "
            f"drive_sim_elapsed={sim1 - drive_sim_start:.3f}s"
        )
        return reached

    def _pause_between_waypoints(self) -> None:
        if self.waypoint_pause_sec <= 0.0:
            return
        pause_until = time.monotonic() + self.waypoint_pause_sec
        while rclpy.ok() and time.monotonic() < pause_until:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _diagnostic_no_motion_report(self) -> None:
        x, y, yaw, _sim = self.pose_xy_yaw_time()
        first = self.waypoints[0]
        last = self.waypoints[-1]
        self.get_logger().info(
            "trajectory_replay_diagnostic_no_motion "
            f"odom_xy=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):+.1f}deg "
            f"waypoints={len(self.waypoints)} "
            f"first_idx={int(first['csv_index'])} first_xy=({float(first['x']):.3f},{float(first['y']):.3f}) "
            f"last_idx={int(last['csv_index'])} last_xy=({float(last['x']):.3f},{float(last['y']):.3f}) "
            f"source_fields={self.waypoint_source_fields}"
        )

    def run(self) -> None:
        self.runner_started_wall = time.monotonic()
        try:
            self._wait_for_odom()
            if self.diagnostic_no_motion:
                self._diagnostic_no_motion_report()
                self.stop_reason = "diagnostic_no_motion"
                return

            self._check_cmd_vel_subscriber()
            self._skip_initial_waypoint_if_close()
            self.stop_reason = "running"
            self.runner_started_wall = time.monotonic()

            for waypoint in list(self.waypoints):
                elapsed_wall = time.monotonic() - self.runner_started_wall
                if elapsed_wall > self.multi_wall_timeout:
                    self.stop_reason = "multi_wall_timeout"
                    self.failed_waypoint_idx = int(waypoint["csv_index"])
                    break

                reached = self.execute_waypoint(waypoint)
                if not reached:
                    self.stop_reason = "waypoint_not_reached"
                    self.failed_waypoint_idx = int(waypoint["csv_index"])
                    break
                self.reached_waypoints += 1
                self._pause_between_waypoints()
            else:
                self.stop_reason = "completed"

            if self.stop_reason == "running":
                self.stop_reason = "rclpy_shutdown"
        except Exception as exc:
            self.stop_reason = f"fatal_exception:{exc}"
            self.get_logger().error(f"trajectory_replay_failure stop_reason={self.stop_reason}")
        finally:
            self.stop()
            self.done = True
            self._write_outputs()
            self._log_final_summary()

    def _write_outputs(self) -> None:
        self.summary_output.parent.mkdir(parents=True, exist_ok=True)
        self.trajectory_log_output.parent.mkdir(parents=True, exist_ok=True)

        if self.waypoint_logs:
            fieldnames = [
                "waypoint_idx",
                "target_x",
                "target_y",
                "start_x",
                "start_y",
                "final_x",
                "final_y",
                "final_dist",
                "reached",
                "skipped",
                "rotate_elapsed_sim",
                "drive_elapsed_sim",
                "rotate_elapsed_wall",
                "drive_elapsed_wall",
            ]
            with self.trajectory_log_output.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.waypoint_logs)

        final_xy = None
        if self.latest_odom is not None:
            final_xy = [
                float(self.latest_odom.pose.pose.position.x),
                float(self.latest_odom.pose.pose.position.y),
            ]

        total_waypoints = int(len(self.waypoints) + self.skipped_waypoints)
        attempted_reached = int(self.reached_waypoints)
        successful = int(attempted_reached + self.skipped_waypoints)
        reached_ratio = 0.0 if total_waypoints <= 0 else float(successful) / float(total_waypoints)
        elapsed_wall = 0.0
        if self.runner_started_wall is not None:
            elapsed_wall = time.monotonic() - self.runner_started_wall

        summary = {
            "schema_version": "drl_trajectory_replay_summary/v1",
            "route_name": "oracle-planned trajectory replay with SLAM mapping",
            "total_waypoints": total_waypoints,
            "reached_waypoints": attempted_reached,
            "skipped_waypoints": int(self.skipped_waypoints),
            "failed_waypoint_idx": self.failed_waypoint_idx,
            "reached_ratio": reached_ratio,
            "final_xy": final_xy,
            "stop_reason": str(self.stop_reason),
            "elapsed_wall": float(elapsed_wall),
            "trajectory_csv": str(self.trajectory_csv),
            "trajectory_log_output": str(self.trajectory_log_output),
            "diagnostic_no_motion": bool(self.diagnostic_no_motion),
            "note": (
                "This node replays offline oracle/ideal waypoints only. It does not "
                "run a DRL policy, read true_grid, construct local_snap, or subscribe to /scan."
            ),
        }
        self.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _log_final_summary(self) -> None:
        final_xy = None
        if self.latest_odom is not None:
            final_xy = (
                float(self.latest_odom.pose.pose.position.x),
                float(self.latest_odom.pose.pose.position.y),
            )
        total_waypoints = int(len(self.waypoints) + self.skipped_waypoints)
        reached_ratio = 0.0
        if total_waypoints > 0:
            reached_ratio = float(self.reached_waypoints + self.skipped_waypoints) / float(total_waypoints)
        self.get_logger().info(
            "trajectory_replay_finished "
            f"stop_reason={self.stop_reason} total_waypoints={total_waypoints} "
            f"reached_waypoints={self.reached_waypoints} skipped_waypoints={self.skipped_waypoints} "
            f"failed_waypoint_idx={self.failed_waypoint_idx} reached_ratio={reached_ratio:.4f} "
            f"final_xy={final_xy} summary_output={self.summary_output}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[DrlTrajectoryReplayNode] = None
    try:
        node = DrlTrajectoryReplayNode()
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
