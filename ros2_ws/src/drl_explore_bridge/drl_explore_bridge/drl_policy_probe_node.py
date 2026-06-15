from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
import torch
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


DRL_REPO = Path.home() / "drl_repos" / "DRL-path-finding"
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


class DrlPolicyProbeNode(Node):
    def __init__(self) -> None:
        super().__init__("drl_policy_probe_node")

        self.declare_parameter("cell_size", 0.25)
        self.declare_parameter("rows", 40)
        self.declare_parameter("cols", 60)
        self.declare_parameter("world_x", 15.0)
        self.declare_parameter("world_y", 10.0)
        self.declare_parameter("scan_radius_cells", 10)
        self.declare_parameter("laser_yaw_in_base", math.pi)
        self.declare_parameter("true_grid_path", "tmp_drl_grids/aligned_10x15_rooms_true_grid.npy")
        self.declare_parameter("checkpoint_path", str(DRL_REPO / "deploy_checkpoints" / "best.pt"))
        self.declare_parameter("print_period_sec", 1.0)

        self.cell_size = float(self.get_parameter("cell_size").value)
        self.rows = int(self.get_parameter("rows").value)
        self.cols = int(self.get_parameter("cols").value)
        self.world_x = float(self.get_parameter("world_x").value)
        self.world_y = float(self.get_parameter("world_y").value)
        self.scan_radius_cells = int(self.get_parameter("scan_radius_cells").value)
        self.laser_yaw_in_base = float(self.get_parameter("laser_yaw_in_base").value)
        self.true_grid_path = Path(str(self.get_parameter("true_grid_path").value))
        self.checkpoint_path = Path(str(self.get_parameter("checkpoint_path").value))
        self.print_period_sec = float(self.get_parameter("print_period_sec").value)

        self.local_size = 2 * self.scan_radius_cells + 1
        self.center = self.scan_radius_cells

        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Odometry] = None
        self.cum_map: Optional[CumulativeBeliefMap] = None
        self.last_agent_state: Optional[tuple[int, int]] = None
        self.recent_trajectory: list[tuple[int, int]] = []

        self.true_grid = self.load_true_grid(self.true_grid_path)

        self.adapter = StateTensorAdapter(device="cpu")
        self.net = ExplorationQNetwork()
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        self.net.load_state_dict(ckpt["online_state_dict"], strict=True)
        self.net.eval()

        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.create_timer(self.print_period_sec, self.timer_cb)

        self.get_logger().info(
            "drl_policy_probe_node started: "
            f"true_grid={self.true_grid_path}, checkpoint={self.checkpoint_path}, "
            f"local_shape=({self.local_size}, {self.local_size})"
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

    def build_local_snap(self, scan: LaserScan, odom: Odometry) -> np.ndarray:
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

    def timer_cb(self) -> None:
        if self.latest_scan is None or self.latest_odom is None:
            self.get_logger().warn("waiting for /scan and /odom ...")
            return

        odom = self.latest_odom
        scan = self.latest_scan

        x = float(odom.pose.pose.position.x)
        y = float(odom.pose.pose.position.y)
        yaw = yaw_from_quat(odom.pose.pose.orientation)
        agent_state = self.world_xy_to_grid_rc(x, y)
        ar, ac = agent_state

        if not (0 <= ar < self.rows and 0 <= ac < self.cols):
            self.get_logger().error(f"agent_state outside true_grid: {agent_state}")
            return

        if int(self.true_grid[ar, ac]) != EMPTY:
            self.get_logger().error(
                f"agent is not on EMPTY cell: rc={agent_state}, value={int(self.true_grid[ar, ac])}"
            )
            return

        local_snap = self.build_local_snap(scan, odom)

        if self.cum_map is None:
            self.cum_map = CumulativeBeliefMap(
                true_grid=self.true_grid,
                start_state=agent_state,
                first_local_snap=local_snap,
            )
            self.last_agent_state = agent_state
            self.recent_trajectory = [agent_state]
            updated = -1
            delta_empty = -1
            delta_obstacle = -1
        else:
            updated, delta_empty, delta_obstacle = self.cum_map.update(agent_state, local_snap)
            if agent_state != self.last_agent_state:
                self.recent_trajectory.append(agent_state)
                self.recent_trajectory = self.recent_trajectory[-20:]
                self.last_agent_state = agent_state

        frontier_cells = int(np.count_nonzero(self.cum_map.get_frontier_u8() > 0))

        state_batch, state_meta = self.adapter.build_single_state_tensors(
            self.cum_map,
            agent_state,
            recent_trajectory_positions=self.recent_trajectory,
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
        greedy_unmasked = int(torch.argmax(q).item())
        greedy_oracle_masked = int(select_greedy_action(q, valid_action_indices=valid).item())

        q_str = " ".join(
            f"{i}:{ACTION_NAMES[i]}={float(q[i]):+.2f}{'*' if i == greedy_oracle_masked else ''}"
            for i in range(len(ACTION_NAMES))
        )

        self.get_logger().info(
            "policy_probe "
            f"rc={agent_state} xy=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):+.1f}deg "
            f"updated={updated} d_empty={delta_empty} d_obs={delta_obstacle} "
            f"known={int(np.count_nonzero(self.cum_map.map != INVISIBLE))} "
            f"frontier={frontier_cells} coverage={float(self.cum_map.coverage_rate):.4f} "
            f"valid={list(valid)} "
            f"greedy={greedy_oracle_masked}:{ACTION_NAMES[greedy_oracle_masked]} "
            f"unmasked={greedy_unmasked}:{ACTION_NAMES[greedy_unmasked]} "
            f"q=[{q_str}] "
            f"accessible_blocks={state_meta.get('accessible_block_count')}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DrlPolicyProbeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
