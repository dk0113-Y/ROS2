from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


INVISIBLE = -1
EMPTY = 0
OBSTACLE = 1


def yaw_from_quat(q) -> float:
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class ScanToLocalSnapNode(Node):
    def __init__(self) -> None:
        super().__init__("scan_to_local_snap_node")

        self.declare_parameter("cell_size", 0.25)
        self.declare_parameter("scan_radius_cells", 10)
        self.declare_parameter("world_x", 15.0)
        self.declare_parameter("world_y", 10.0)
        self.declare_parameter("laser_yaw_in_base", math.pi)
        self.declare_parameter("print_ascii", True)
        self.declare_parameter("print_period_sec", 1.0)

        self.cell_size = float(self.get_parameter("cell_size").value)
        self.scan_radius_cells = int(self.get_parameter("scan_radius_cells").value)
        self.world_x = float(self.get_parameter("world_x").value)
        self.world_y = float(self.get_parameter("world_y").value)
        self.laser_yaw_in_base = float(self.get_parameter("laser_yaw_in_base").value)
        self.print_ascii = bool(self.get_parameter("print_ascii").value)
        self.print_period_sec = float(self.get_parameter("print_period_sec").value)

        self.local_size = 2 * self.scan_radius_cells + 1
        self.center = self.scan_radius_cells

        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Odometry] = None

        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.create_timer(self.print_period_sec, self.timer_cb)

        self.get_logger().info(
            "scan_to_local_snap_node started: "
            f"cell_size={self.cell_size}, "
            f"scan_radius_cells={self.scan_radius_cells}, "
            f"local_shape=({self.local_size}, {self.local_size}), "
            f"world=({self.world_x}m, {self.world_y}m)"
        )

    def scan_cb(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def odom_cb(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def world_xy_to_grid_rc(self, x: float, y: float) -> tuple[int, int]:
        # world: x ∈ [-7.5, 7.5], y ∈ [-5, 5]
        # grid: rows=40, cols=60, row 0 at +y side, col 0 at -x side
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

        # 命中障碍时，终点格标 OBSTACLE，之前路径标 EMPTY。
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

            # 当前 URDF 中 laser_link 相对 base_link 约 180°。
            # 后续正式版可以改成 tf2 自动获取。
            angle_world = robot_yaw + self.laser_yaw_in_base + scan_angle

            for dr, dc, value in self.ray_to_local_cells(angle_world, dist, hit_obstacle):
                self.mark_cell(snap, dr, dc, value)

        return snap

    def print_snap(self, snap: np.ndarray, odom: Odometry, scan: LaserScan) -> None:
        x = float(odom.pose.pose.position.x)
        y = float(odom.pose.pose.position.y)
        yaw = yaw_from_quat(odom.pose.pose.orientation)
        row, col = self.world_xy_to_grid_rc(x, y)

        unknown = int(np.count_nonzero(snap == INVISIBLE))
        free = int(np.count_nonzero(snap == EMPTY))
        obstacle = int(np.count_nonzero(snap == OBSTACLE))

        self.get_logger().info(
            "local_snap "
            f"grid_rc=({row},{col}) "
            f"odom_xy=({x:.3f},{y:.3f}) "
            f"yaw={yaw:.3f} "
            f"samples={len(scan.ranges)} "
            f"unknown={unknown} free={free} obstacle={obstacle}"
        )

        if not self.print_ascii:
            return

        lines = []
        for r in range(self.local_size):
            chars = []
            for c in range(self.local_size):
                if r == self.center and c == self.center:
                    chars.append("R")
                elif snap[r, c] == INVISIBLE:
                    chars.append("?")
                elif snap[r, c] == EMPTY:
                    chars.append(".")
                elif snap[r, c] == OBSTACLE:
                    chars.append("#")
                else:
                    chars.append("!")
            lines.append("".join(chars))

        print("===== local_snap ascii =====")
        print("\n".join(lines))

    def timer_cb(self) -> None:
        if self.latest_scan is None or self.latest_odom is None:
            self.get_logger().warn("waiting for /scan and /odom ...")
            return

        snap = self.build_local_snap(self.latest_scan, self.latest_odom)
        self.print_snap(snap, self.latest_odom, self.latest_scan)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScanToLocalSnapNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
