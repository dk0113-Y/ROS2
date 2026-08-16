from __future__ import annotations

import math
import os
import sys
import time
from typing import Optional

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

DEFAULT_CHECKPOINT = os.environ.get("DRL_CHECKPOINT_PATH", "")

INVISIBLE = -1
EMPTY = 0
OBSTACLE = 1


def yaw_from_quat(q) -> float:
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def checkpoint_repo_dir(checkpoint_path: str) -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))


def load_policy_dependencies(checkpoint_path: str):
    drl_repo = checkpoint_repo_dir(checkpoint_path)
    if drl_repo not in sys.path:
        sys.path.insert(0, drl_repo)

    import torch
    from agents.q_value_agent import ExplorationQNetwork, StateTensorAdapter
    from env.core_cummap import CumulativeBeliefMap

    return torch, ExplorationQNetwork, StateTensorAdapter, CumulativeBeliefMap


class RealcarDryRunProbe(Node):
    def __init__(self) -> None:
        super().__init__("realcar_dryrun_probe_once")

        self.declare_parameter("checkpoint_path", DEFAULT_CHECKPOINT)
        self.declare_parameter("max_steps", 10)
        self.declare_parameter("input_timeout_sec", 8.0)
        self.declare_parameter("scan_timeout_sec", 0.5)
        self.declare_parameter("odom_timeout_sec", 0.5)
        self.declare_parameter("sensor_future_tolerance_sec", 0.1)

        self.checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        self.max_steps = int(self.get_parameter("max_steps").value)
        self.input_timeout_sec = float(self.get_parameter("input_timeout_sec").value)
        self.scan_timeout_sec = float(self.get_parameter("scan_timeout_sec").value)
        self.odom_timeout_sec = float(self.get_parameter("odom_timeout_sec").value)
        self.sensor_future_tolerance_sec = float(
            self.get_parameter("sensor_future_tolerance_sec").value
        )

        if not self.checkpoint_path:
            raise ValueError(
                "checkpoint_path is required (or set DRL_CHECKPOINT_PATH)"
            )
        if self.max_steps < 0:
            raise ValueError("max_steps must be >= 0")
        if self.input_timeout_sec <= 0.0:
            raise ValueError("input_timeout_sec must be > 0")
        if self.scan_timeout_sec <= 0.0 or self.odom_timeout_sec <= 0.0:
            raise ValueError("sensor timeout parameters must be > 0")
        if self.sensor_future_tolerance_sec < 0.0:
            raise ValueError("sensor_future_tolerance_sec must be >= 0")

        self.cell_size = 0.35
        self.scan_radius_cells = 10
        self.local_size = 2 * self.scan_radius_cells + 1
        self.center = self.scan_radius_cells

        # 实车前方障碍物测试已确认：/scan 的 0 度基本对应车头前方。
        self.laser_yaw_in_base = 0.0

        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Odometry] = None
        self.latest_scan_received_at: Optional[float] = None
        self.latest_odom_received_at: Optional[float] = None

        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)

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

        # 超出局部窗口的距离不需要继续追踪，减少树莓派 CPU 负担。
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

            # LaserScan 官方定义要求丢弃小于 range_min 或大于 range_max 的距离。
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


def wait_for_inputs(node: RealcarDryRunProbe) -> None:
    deadline = time.monotonic() + node.input_timeout_sec
    last_error = "no messages received"
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.latest_scan is not None and node.latest_odom is not None:
            try:
                node.require_fresh_inputs()
                return
            except RuntimeError as exc:
                last_error = str(exc)
    raise TimeoutError(f"timeout waiting for fresh /scan and /odom: {last_error}")


def main(args=None) -> None:
    print("Safety: this script does NOT create any cmd_vel publisher.")

    rclpy.init(args=args)
    node = RealcarDryRunProbe()

    try:
        print(
            "Safety: this script reads /scan and /odom, runs policy inference "
            f"at 1Hz for {node.max_steps} steps, then exits."
        )
        cmd_vel_publishers = node.get_publishers_info_by_topic("/cmd_vel")
        print("cmd_vel_publisher_count =", len(cmd_vel_publishers))
        if len(cmd_vel_publishers) != 0:
            raise RuntimeError("Unsafe: /cmd_vel already has publisher(s). Stop before dry-run.")

        wait_for_inputs(node)

        torch, ExplorationQNetwork, StateTensorAdapter, CumulativeBeliefMap = (
            load_policy_dependencies(node.checkpoint_path)
        )
        model = ExplorationQNetwork()
        checkpoint = torch.load(
            node.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["online_state_dict"])
        model.eval()

        adapter = StateTensorAdapter(device="cpu")
        true_grid = np.zeros((120, 120), dtype=np.int8)
        agent_state = (60, 60)
        recent_positions = [agent_state]

        cum_map = None

        for step_idx in range(node.max_steps):
            # 等待两个 topic 都刷新，避免 1 Hz dry-run 循环复用过期消息。
            wait_for_inputs(node)

            scan = node.latest_scan
            odom = node.latest_odom
            assert scan is not None
            assert odom is not None

            snap = node.build_local_snap(scan, odom)

            unknown = int(np.count_nonzero(snap == INVISIBLE))
            free = int(np.count_nonzero(snap == EMPTY))
            obstacle = int(np.count_nonzero(snap == OBSTACLE))
            yaw = yaw_from_quat(odom.pose.pose.orientation)

            if cum_map is None:
                cum_map = CumulativeBeliefMap(true_grid, agent_state, snap)
                updated, delta_empty, delta_obstacle = 0, 0, 0
            else:
                updated, delta_empty, delta_obstacle = cum_map.update(agent_state, snap)

            state_batch, state_meta = adapter.build_single_state_tensors(
                cum_map,
                agent_state,
                recent_trajectory_positions=recent_positions,
                return_state_meta=True,
            )

            t0 = time.perf_counter()
            with torch.inference_mode():
                q_values = model(**state_batch, return_aux=False)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            q_np = q_values.detach().cpu().numpy()
            action_idx = int(torch.argmax(q_values, dim=1).item())
            odom_x = float(odom.pose.pose.position.x)
            odom_y = float(odom.pose.pose.position.y)

            print(
                f"step={step_idx:02d} "
                f"odom=({odom_x:.3f},{odom_y:.3f},{yaw:.3f}) "
                f"snap(u/f/o)=({unknown}/{free}/{obstacle}) "
                f"cum_update=({updated}/{delta_empty}/{delta_obstacle}) "
                f"action={action_idx} "
                f"infer_ms={infer_ms:.1f} "
                f"q={np.round(q_np[0], 3).tolist()}"
            )

            time.sleep(1.0)

        print("OK: real sensor policy dry-run loop passed; no cmd_vel was published.")

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
