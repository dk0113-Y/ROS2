from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import LaserScan


DRL_REPO = Path("/home/dk/drl_repos/DRL-path-finding")
if str(DRL_REPO) not in sys.path:
    sys.path.insert(0, str(DRL_REPO))

from agents.q_value_agent import ExplorationQNetwork, StateTensorAdapter, select_greedy_action
from env.core_cummap import CumulativeBeliefMap
from env.grid_topology import EMPTY, INVISIBLE, OBSTACLE, ACTIONS_8, GridTopology


ACTION_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


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


class DrlPolicyStepOnceNode(Node):
    def __init__(self) -> None:
        super().__init__("drl_policy_step_once_node")

        self.declare_parameter("execute_once", False)
        self.declare_parameter("cell_size", 0.25)
        self.declare_parameter("rows", 40)
        self.declare_parameter("cols", 60)
        self.declare_parameter("world_x", 15.0)
        self.declare_parameter("world_y", 10.0)
        self.declare_parameter("scan_radius_cells", 10)
        self.declare_parameter("laser_yaw_in_base", math.pi)
        self.declare_parameter("scan_bridge_mode", "ray_project")
        self.declare_parameter("oracle_los_training_repo", "/home/dk/drl_repos/DRL-path-finding")
        self.declare_parameter("los_beam_window", 1)
        self.declare_parameter("los_obstacle_tolerance", 0.55)
        self.declare_parameter("true_grid_path", "tmp_drl_grids/aligned_10x15_rooms_true_grid.npy")
        self.declare_parameter("checkpoint_path", "/home/dk/drl_repos/DRL-path-finding/deploy_checkpoints/best.pt")

        self.declare_parameter("rotate_kp", 2.0)
        self.declare_parameter("rotate_max_w", 2.0)
        self.declare_parameter("rotate_min_w", 0.20)
        self.declare_parameter("rotate_tol_deg", 4.0)
        self.declare_parameter("rotate_sim_timeout", 12.0)
        self.declare_parameter("linear_speed", 0.10)
        self.declare_parameter("target_pos_tol", 0.055)
        self.declare_parameter("drive_sim_timeout", 12.0)
        self.declare_parameter("control_debug_period", 1.0)
        self.declare_parameter("rotate_wall_timeout", 60.0)
        self.declare_parameter("drive_wall_timeout", 60.0)

        self.execute_once = bool(self.get_parameter("execute_once").value)
        self.cell_size = float(self.get_parameter("cell_size").value)
        self.rows = int(self.get_parameter("rows").value)
        self.cols = int(self.get_parameter("cols").value)
        self.world_x = float(self.get_parameter("world_x").value)
        self.world_y = float(self.get_parameter("world_y").value)
        self.scan_radius_cells = int(self.get_parameter("scan_radius_cells").value)
        self.laser_yaw_in_base = float(self.get_parameter("laser_yaw_in_base").value)
        self.scan_bridge_mode = str(self.get_parameter("scan_bridge_mode").value)
        self.oracle_los_training_repo = str(self.get_parameter("oracle_los_training_repo").value)
        self.los_beam_window = int(self.get_parameter("los_beam_window").value)
        self.los_obstacle_tolerance = float(self.get_parameter("los_obstacle_tolerance").value)

        self.true_grid_path = Path(str(self.get_parameter("true_grid_path").value))
        self.checkpoint_path = Path(str(self.get_parameter("checkpoint_path").value))

        self.rotate_kp = float(self.get_parameter("rotate_kp").value)
        self.rotate_max_w = float(self.get_parameter("rotate_max_w").value)
        self.rotate_min_w = float(self.get_parameter("rotate_min_w").value)
        self.rotate_tol = math.radians(float(self.get_parameter("rotate_tol_deg").value))
        self.rotate_sim_timeout = float(self.get_parameter("rotate_sim_timeout").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.target_pos_tol = float(self.get_parameter("target_pos_tol").value)
        self.drive_sim_timeout = float(self.get_parameter("drive_sim_timeout").value)
        self.control_debug_period = float(self.get_parameter("control_debug_period").value)
        self.rotate_wall_timeout = float(self.get_parameter("rotate_wall_timeout").value)
        self.drive_wall_timeout = float(self.get_parameter("drive_wall_timeout").value)

        self.local_size = 2 * self.scan_radius_cells + 1
        self.center = self.scan_radius_cells

        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Odometry] = None
        self.done = False

        # Keep sensor callbacks in a different callback group from the control timer.
        # Otherwise execute_target_cell() runs inside timer_cb() and nested spin_once()
        # cannot refresh /odom while the default mutually-exclusive group is occupied.
        self.sensor_cb_group = MutuallyExclusiveCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()

        self.true_grid = self.load_true_grid(self.true_grid_path)

        self.adapter = StateTensorAdapter(device="cpu")
        self.net = ExplorationQNetwork()
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        self.net.load_state_dict(ckpt["online_state_dict"], strict=True)
        self.net.eval()

        self.cmd_pub = None
        if self.execute_once:
            self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10, callback_group=self.sensor_cb_group)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10, callback_group=self.sensor_cb_group)
        self.create_timer(0.2, self.timer_cb, callback_group=self.control_cb_group)

        self.get_logger().info(
            "drl_policy_step_once_node started: "
            f"execute_once={self.execute_once}, "
            f"true_grid={self.true_grid_path}, checkpoint={self.checkpoint_path}"
        )

    def load_true_grid(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"true_grid file not found: {path}")
        grid = np.load(path).astype(np.int8)
        if grid.shape != (self.rows, self.cols):
            raise ValueError(f"true_grid shape mismatch: {grid.shape}, expected {(self.rows, self.cols)}")
        return grid

    def scan_cb(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def odom_cb(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def world_xy_to_grid_rc(self, x: float, y: float) -> tuple[int, int]:
        col = int(math.floor((x + self.world_x / 2.0) / self.cell_size))
        row = int(math.floor((self.world_y / 2.0 - y) / self.cell_size))
        return row, col

    def grid_rc_to_cell_center_xy(self, row: int, col: int) -> tuple[float, float]:
        x = -self.world_x / 2.0 + (int(col) + 0.5) * self.cell_size
        y = self.world_y / 2.0 - (int(row) + 0.5) * self.cell_size
        return x, y

    def pose_xy_yaw_time(self) -> tuple[float, float, float, float]:
        if self.latest_odom is None:
            raise RuntimeError("latest_odom is None")
        p = self.latest_odom.pose.pose.position
        q = self.latest_odom.pose.pose.orientation
        return float(p.x), float(p.y), yaw_from_quat(q), stamp_to_sec(self.latest_odom.header.stamp)

    def stop(self, repeat: int = 10) -> None:
        if self.cmd_pub is None or not rclpy.ok():
            return
        msg = make_twist(0.0, 0.0)
        for _ in range(repeat):
            if not rclpy.ok():
                return
            try:
                self.cmd_pub.publish(msg)
            except Exception as exc:
                try:
                    self.get_logger().warn(f"stop publish skipped: {exc}")
                except Exception:
                    pass
                return
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
                yield dr, dc, EMPTY

            d += step

        if hit_obstacle:
            rel_x = max_d * math.cos(angle_world)
            rel_y = max_d * math.sin(angle_world)

            dc = int(round(rel_x / self.cell_size))
            dr = int(round(-rel_y / self.cell_size))
            yield dr, dc, OBSTACLE

    @staticmethod
    def _wrap_angle_pi(angle: float) -> float:
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _bresenham_local_line(dr: int, dc: int) -> list[tuple[int, int]]:
        """Return local grid cells from center-neighbor to target, excluding (0, 0)."""
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

    def _scan_range_at_angle(self, scan: LaserScan, scan_angle: float) -> tuple[float, bool]:
        """Query LaserScan around scan_angle.

        Returns:
          (distance, hit_obstacle)
        """
        if len(scan.ranges) <= 0 or float(scan.angle_increment) == 0.0:
            return float(scan.range_max), False

        angle_min = float(scan.angle_min)
        angle_max = float(scan.angle_max)
        angle_inc = float(scan.angle_increment)

        a = float(scan_angle)
        # Normalize into scan angular interval when possible.
        while a < angle_min:
            a += 2.0 * math.pi
        while a > angle_max:
            a -= 2.0 * math.pi

        if a < angle_min or a > angle_max:
            return float(scan.range_max), False

        idx = int(round((a - angle_min) / angle_inc))
        idx = max(0, min(len(scan.ranges) - 1, idx))

        window = max(0, int(self.los_beam_window))
        i0 = max(0, idx - window)
        i1 = min(len(scan.ranges), idx + window + 1)

        finite_values: list[float] = []
        for j in range(i0, i1):
            r = scan.ranges[j]
            if math.isfinite(r):
                rr = min(max(float(r), float(scan.range_min)), float(scan.range_max))
                finite_values.append(rr)

        if finite_values:
            return min(finite_values), True

        return float(scan.range_max), False

    def build_local_snap_ray_project(self, scan: LaserScan, odom: Odometry) -> np.ndarray:
        robot_yaw = yaw_from_quat(odom.pose.pose.orientation)
        snap = np.full((self.local_size, self.local_size), INVISIBLE, dtype=np.int8)
        snap[self.center, self.center] = EMPTY

        for i, r in enumerate(scan.ranges):
            scan_angle = scan.angle_min + i * scan.angle_increment

            if math.isfinite(r):
                dist = min(max(float(r), float(scan.range_min)), float(scan.range_max))
                hit_obstacle = True
            else:
                dist = float(scan.range_max)
                hit_obstacle = False

            angle_world = robot_yaw + self.laser_yaw_in_base + scan_angle

            for dr, dc, value in self.ray_to_local_cells(angle_world, dist, hit_obstacle):
                self.mark_cell(snap, dr, dc, value)

        return snap

    def build_local_snap_los_compatible(self, scan: LaserScan, odom: Odometry) -> np.ndarray:
        """Build a training-style LOS local snap from LaserScan.

        This mode enumerates local grid target cells first, then uses LaserScan
        ranges to answer whether each LOS cell is visible, occupied, or occluded.
        It is intended to keep the network input closer to the training
        LocalObservationModel semantics than direct per-beam rasterization.
        """
        robot_yaw = yaw_from_quat(odom.pose.pose.orientation)
        snap = np.full((self.local_size, self.local_size), INVISIBLE, dtype=np.int8)
        snap[self.center, self.center] = EMPTY

        # Enumerate candidate target cells in the same circular local support.
        targets: list[tuple[int, int, float]] = []
        for dr in range(-self.scan_radius_cells, self.scan_radius_cells + 1):
            for dc in range(-self.scan_radius_cells, self.scan_radius_cells + 1):
                if dr == 0 and dc == 0:
                    continue
                r2 = dr * dr + dc * dc
                if r2 > self.scan_radius_cells * self.scan_radius_cells:
                    continue
                targets.append((int(dr), int(dc), float(r2)))

        # Near-to-far helps nearer cells establish obstacle stops first.
        targets.sort(key=lambda x: x[2])

        obstacle_tol_m = max(1e-6, float(self.los_obstacle_tolerance) * self.cell_size)
        max_range = float(scan.range_max)

        for target_dr, target_dc, _ in targets:
            line = self._bresenham_local_line(target_dr, target_dc)
            if not line:
                continue

            for dr, dc in line:
                if dr * dr + dc * dc > self.scan_radius_cells * self.scan_radius_cells:
                    break

                rel_x = float(dc) * self.cell_size
                rel_y = -float(dr) * self.cell_size
                cell_dist = math.hypot(rel_x, rel_y)

                if cell_dist > max_range + obstacle_tol_m:
                    break

                angle_world = math.atan2(rel_y, rel_x)
                scan_angle = self._wrap_angle_pi(angle_world - robot_yaw - self.laser_yaw_in_base)

                measured_dist, hit_obstacle = self._scan_range_at_angle(scan, scan_angle)

                if not hit_obstacle:
                    self.mark_cell(snap, dr, dc, EMPTY)
                    continue

                # Cell is clearly before the measured obstacle.
                if cell_dist < measured_dist - obstacle_tol_m:
                    self.mark_cell(snap, dr, dc, EMPTY)
                    continue

                # Measured obstacle lies in this cell distance band.
                if abs(cell_dist - measured_dist) <= obstacle_tol_m:
                    self.mark_cell(snap, dr, dc, OBSTACLE)
                    break

                # Obstacle is closer than this cell, so this and following cells are occluded.
                break

        return snap

    def build_local_snap_oracle_los(self, scan: LaserScan, odom: Odometry) -> np.ndarray:
        """Diagnostic-only training-style LOS observation from true_grid.

        This mode intentionally ignores LaserScan geometry and uses the known
        true grid plus current agent_state to produce the same type of local
        observation used in training. It is only for simulation diagnosis, not
        for real deployment.
        """
        repo = Path(str(self.oracle_los_training_repo))
        repo_s = str(repo)
        if repo_s not in sys.path:
            sys.path.insert(0, repo_s)

        try:
            from env.agent_version import LocalObservationModel
            from env.core_radar import RadarSensor
        except Exception as exc:
            raise RuntimeError(
                "oracle_los failed to import training observation modules from "
                f"{repo_s}: {exc}"
            ) from exc

        # Use current odometry to recover the grid cell, so the oracle mode
        # follows the actual Gazebo-executed pose rather than an ideal planner state.
        x = float(odom.pose.pose.position.x)
        y = float(odom.pose.pose.position.y)
        agent_state = self.world_xy_to_grid_rc(x, y)

        sensor = RadarSensor(scan_radius=int(self.scan_radius_cells))
        obs_model = LocalObservationModel(self.true_grid, agent_state, sensor=sensor)

        snap = obs_model.local_snap
        if hasattr(obs_model, "observe_fast"):
            snap = obs_model.observe_fast(agent_state)

        snap = np.asarray(snap, dtype=np.int8)

        if snap.shape != (self.local_size, self.local_size):
            raise RuntimeError(
                "oracle_los snap shape mismatch: "
                f"got {snap.shape}, expected {(self.local_size, self.local_size)}"
            )

        return snap

    def build_local_snap(self, scan: LaserScan, odom: Odometry) -> np.ndarray:
        mode = str(getattr(self, "scan_bridge_mode", "ray_project")).strip().lower()

        if mode in ("ray_project", "ray", "beam"):
            return self.build_local_snap_ray_project(scan, odom)

        if mode in ("los_compatible", "training_los", "los"):
            return self.build_local_snap_los_compatible(scan, odom)

        if mode in ("oracle_los", "oracle", "true_grid_los"):
            return self.build_local_snap_oracle_los(scan, odom)

        raise RuntimeError(
            f"unsupported scan_bridge_mode={mode!r}; "
            "expected 'ray_project', 'los_compatible', or 'oracle_los'"
        )

    def infer_action(self) -> tuple[int, tuple[int, ...], torch.Tensor, CumulativeBeliefMap, tuple[int, int]]:
        if self.latest_scan is None or self.latest_odom is None:
            raise RuntimeError("waiting for /scan and /odom")

        odom = self.latest_odom
        scan = self.latest_scan

        x = float(odom.pose.pose.position.x)
        y = float(odom.pose.pose.position.y)
        agent_state = self.world_xy_to_grid_rc(x, y)
        ar, ac = agent_state

        if not (0 <= ar < self.rows and 0 <= ac < self.cols):
            raise RuntimeError(f"agent_state outside true_grid: {agent_state}")

        if int(self.true_grid[ar, ac]) != EMPTY:
            raise RuntimeError(f"agent is not on EMPTY cell: rc={agent_state}, value={int(self.true_grid[ar, ac])}")

        local_snap = self.build_local_snap(scan, odom)
        cum_map = CumulativeBeliefMap(
            true_grid=self.true_grid,
            start_state=agent_state,
            first_local_snap=local_snap,
        )

        state_batch, _ = self.adapter.build_single_state_tensors(
            cum_map,
            agent_state,
            recent_trajectory_positions=[agent_state],
            return_state_meta=True,
        )

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
        valid = GridTopology.valid_action_indices_fast(GridTopology.free_mask(self.true_grid), agent_state)
        if len(valid) <= 0:
            raise RuntimeError(f"no valid action at {agent_state}")

        action_idx = int(select_greedy_action(q, valid_action_indices=valid).item())
        return action_idx, valid, q, cum_map, agent_state

    def execute_target_cell(self, action_idx: int, start_state: tuple[int, int]) -> None:
        x0, y0, yaw0, sim0 = self.pose_xy_yaw_time()
        row0, col0 = start_state

        dr, dc = ACTIONS_8[action_idx]
        target_row = row0 + int(dr)
        target_col = col0 + int(dc)
        tx, ty = self.grid_rc_to_cell_center_xy(target_row, target_col)

        c0x, c0y = self.grid_rc_to_cell_center_xy(row0, col0)
        initial_dist = math.hypot(tx - x0, ty - y0)
        target_yaw = math.atan2(ty - y0, tx - x0)

        self.get_logger().info(
            "execute_plan "
            f"action={action_idx}:{ACTION_NAMES[action_idx]} "
            f"start_rc=({row0},{col0}) target_rc=({target_row},{target_col}) "
            f"start_xy=({x0:.3f},{y0:.3f}) start_center=({c0x:.3f},{c0y:.3f}) "
            f"target_xy=({tx:.3f},{ty:.3f}) "
            f"initial_dist={initial_dist:.3f} target_yaw={math.degrees(target_yaw):+.1f}deg"
        )

        self.get_logger().info("rotate phase start")
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
                raise RuntimeError(f"Rotate timeout, last yaw error={err:.3f} rad")
            if wall_elapsed > self.rotate_wall_timeout:
                raise RuntimeError(
                    f"Rotate wall-timeout, sim_elapsed={sim_elapsed:.3f}s, "
                    f"last yaw error={err:.3f} rad"
                )

            wz = max(-self.rotate_max_w, min(self.rotate_max_w, self.rotate_kp * err))
            if abs(wz) < self.rotate_min_w:
                wz = math.copysign(self.rotate_min_w, wz)

            if wall_elapsed - last_rotate_debug_wall >= self.control_debug_period:
                last_rotate_debug_wall = wall_elapsed
                self.get_logger().info(
                    "rotate_debug "
                    f"xy=({x:.3f},{y:.3f}) "
                    f"yaw={math.degrees(yaw):+.1f}deg "
                    f"target_yaw={math.degrees(target_yaw_now):+.1f}deg "
                    f"err={math.degrees(err):+.1f}deg "
                    f"wz={wz:+.3f} "
                    f"stable={stable_count} "
                    f"sim_elapsed={sim_elapsed:.3f}s "
                    f"wall_elapsed={wall_elapsed:.3f}s"
                )

            self.cmd_pub.publish(make_twist(0.0, wz))

        self.stop()

        xr, yr, yawr, simr = self.pose_xy_yaw_time()
        self.get_logger().info(
            "rotate phase done "
            f"xy=({xr:.3f},{yr:.3f}) yaw={math.degrees(yawr):+.1f}deg "
            f"dist_to_target={math.hypot(tx-xr, ty-yr):.3f} "
            f"sim_elapsed={simr-rotate_sim_start:.3f}s"
        )

        self.get_logger().info("drive phase start")
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
                raise RuntimeError(f"Drive timeout, distance_to_target={dist_to_target:.3f} m")
            if wall_elapsed > self.drive_wall_timeout:
                raise RuntimeError(
                    f"Drive wall-timeout, sim_elapsed={sim_elapsed:.3f}s, "
                    f"distance_to_target={dist_to_target:.3f} m"
                )

            target_yaw_now = math.atan2(ty - y, tx - x)
            yaw_err = norm_angle(target_yaw_now - yaw)
            wz_correction = max(-0.3, min(0.3, 1.0 * yaw_err))

            if wall_elapsed - last_drive_debug_wall >= self.control_debug_period:
                last_drive_debug_wall = wall_elapsed
                self.get_logger().info(
                    "drive_debug "
                    f"xy=({x:.3f},{y:.3f}) "
                    f"yaw={math.degrees(yaw):+.1f}deg "
                    f"target_yaw={math.degrees(target_yaw_now):+.1f}deg "
                    f"yaw_err={math.degrees(yaw_err):+.1f}deg "
                    f"dist={dist_to_target:.3f} "
                    f"vx={self.linear_speed:+.3f} "
                    f"wz={wz_correction:+.3f} "
                    f"sim_elapsed={sim_elapsed:.3f}s "
                    f"wall_elapsed={wall_elapsed:.3f}s"
                )

            self.cmd_pub.publish(make_twist(self.linear_speed, wz_correction))

        self.stop()

        x1, y1, yaw1, sim1 = self.pose_xy_yaw_time()
        final_row, final_col = self.world_xy_to_grid_rc(x1, y1)
        final_dist = math.hypot(tx - x1, ty - y1)

        self.get_logger().info(
            "execute_result "
            f"final_rc=({final_row},{final_col}) target_rc=({target_row},{target_col}) "
            f"final_xy=({x1:.3f},{y1:.3f}) final_yaw={math.degrees(yaw1):+.1f}deg "
            f"final_dist_to_target={final_dist:.3f} "
            f"drive_sim_elapsed={sim1-drive_sim_start:.3f}s"
        )

        if (final_row, final_col) == (target_row, target_col):
            self.get_logger().info("OK: reached target grid cell")
        else:
            self.get_logger().warn("WARN: final grid cell differs from target grid cell")

    def timer_cb(self) -> None:
        if self.done:
            return

        if self.latest_scan is None or self.latest_odom is None:
            self.get_logger().warn("waiting for /scan and /odom ...")
            return

        try:
            action_idx, valid, q, cum_map, agent_state = self.infer_action()
            x, y, yaw, _ = self.pose_xy_yaw_time()

            dr, dc = ACTIONS_8[action_idx]
            target_state = (agent_state[0] + int(dr), agent_state[1] + int(dc))
            tx, ty = self.grid_rc_to_cell_center_xy(*target_state)

            q_str = " ".join(
                f"{i}:{ACTION_NAMES[i]}={float(q[i]):+.2f}{'*' if i == action_idx else ''}"
                for i in range(len(ACTION_NAMES))
            )

            self.get_logger().info(
                "step_plan "
                f"execute_once={self.execute_once} "
                f"rc={agent_state} xy=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):+.1f}deg "
                f"valid={list(valid)} selected={action_idx}:{ACTION_NAMES[action_idx]} "
                f"target_rc={target_state} target_xy=({tx:.3f},{ty:.3f}) "
                f"known={int(np.count_nonzero(cum_map.map != INVISIBLE))} "
                f"frontier={int(np.count_nonzero(cum_map.get_frontier_u8() > 0))} "
                f"coverage={float(cum_map.coverage_rate):.4f} "
                f"q=[{q_str}]"
            )

            if self.execute_once:
                if self.cmd_pub is None:
                    raise RuntimeError("cmd_pub is None while execute_once=True")
                if self.cmd_pub.get_subscription_count() < 1:
                    raise RuntimeError("No /cmd_vel subscriber found")
                self.execute_target_cell(action_idx, agent_state)

            self.done = True
            self.get_logger().info("step_once node finished")

        except Exception as exc:
            self.stop()
            self.done = True
            self.get_logger().error(f"FAIL: {exc}")
            self.get_logger().error("A stop command has been sent.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = DrlPolicyStepOnceNode()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        if node is not None:
            if rclpy.ok():
                node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
