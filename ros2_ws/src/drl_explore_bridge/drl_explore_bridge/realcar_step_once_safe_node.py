from __future__ import annotations

import math
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import LaserScan


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
MAX_LINEAR_SPEED = 0.06
MAX_ANGULAR_SPEED = 0.40


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


class RealcarStepOnceSafeNode(Node):
    """Single-step motion test for use only under direct human supervision."""

    def __init__(self) -> None:
        super().__init__("realcar_step_once_safe_node")

        self.declare_parameter("execute", False)
        self.declare_parameter("action_idx", 2)
        self.declare_parameter("step_distance", 0.18)

        self.declare_parameter("rotate_kp", 1.2)
        self.declare_parameter("rotate_max_w", 0.35)
        self.declare_parameter("rotate_min_w", 0.08)
        self.declare_parameter("rotate_tol_deg", 6.0)
        self.declare_parameter("rotate_wall_timeout", 6.0)

        self.declare_parameter("linear_speed", 0.04)
        self.declare_parameter("target_pos_tol", 0.04)
        self.declare_parameter("drive_wall_timeout", 6.0)
        self.declare_parameter("control_debug_period", 0.5)
        self.declare_parameter("scan_timeout_sec", 0.5)
        self.declare_parameter("odom_timeout_sec", 0.5)
        self.declare_parameter("sensor_future_tolerance_sec", 0.1)

        self.execute = bool(self.get_parameter("execute").value)
        self.action_idx = int(self.get_parameter("action_idx").value)
        self.step_distance = float(self.get_parameter("step_distance").value)

        self.rotate_kp = float(self.get_parameter("rotate_kp").value)
        self.rotate_max_w = float(self.get_parameter("rotate_max_w").value)
        self.rotate_min_w = float(self.get_parameter("rotate_min_w").value)
        self.rotate_tol = math.radians(float(self.get_parameter("rotate_tol_deg").value))
        self.rotate_wall_timeout = float(self.get_parameter("rotate_wall_timeout").value)

        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.target_pos_tol = float(self.get_parameter("target_pos_tol").value)
        self.drive_wall_timeout = float(self.get_parameter("drive_wall_timeout").value)
        self.control_debug_period = float(self.get_parameter("control_debug_period").value)
        self.scan_timeout_sec = float(self.get_parameter("scan_timeout_sec").value)
        self.odom_timeout_sec = float(self.get_parameter("odom_timeout_sec").value)
        self.sensor_future_tolerance_sec = float(
            self.get_parameter("sensor_future_tolerance_sec").value
        )

        if not (0 <= self.action_idx < len(ACTIONS_8)):
            raise ValueError(f"action_idx out of range: {self.action_idx}")

        if self.step_distance <= 0.0 or self.step_distance > 0.25:
            raise ValueError("step_distance must be in (0, 0.25] for first realcar test")

        if self.linear_speed <= 0.0 or self.linear_speed > MAX_LINEAR_SPEED:
            raise ValueError("linear_speed must be in (0, 0.06] for first realcar test")

        if self.rotate_max_w <= 0.0 or self.rotate_max_w > MAX_ANGULAR_SPEED:
            raise ValueError("rotate_max_w must be in (0, 0.40] for first realcar test")
        if self.scan_timeout_sec <= 0.0 or self.odom_timeout_sec <= 0.0:
            raise ValueError("sensor timeout parameters must be > 0")
        if self.sensor_future_tolerance_sec < 0.0:
            raise ValueError("sensor_future_tolerance_sec must be >= 0")

        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Odometry] = None
        self.latest_scan_received_at: Optional[float] = None
        self.latest_odom_received_at: Optional[float] = None
        self.done = False

        # Keep odom subscription callbacks in a different callback group from the control timer.
        # execute_target() runs inside timer_cb() and calls spin_once() to refresh /odom.
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
        self.create_timer(0.2, self.timer_cb, callback_group=self.control_cb_group)

        self.get_logger().warn(
            "MANUAL-SUPERVISION SINGLE-STEP TEST ONLY. "
            "realcar_step_once_safe_node started. "
            f"execute={self.execute}, "
            f"action_idx={self.action_idx}:{ACTION_NAMES[self.action_idx]}, "
            f"step_distance={self.step_distance:.3f}, linear_speed={self.linear_speed:.3f}, "
            f"rotate_max_w={self.rotate_max_w:.3f}. "
            "This node can publish non-zero /cmd_vel only when execute=true (default false)."
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

    def inputs_are_ready(self) -> bool:
        if self.latest_scan is None or self.latest_odom is None:
            return False
        try:
            self.require_fresh_inputs()
            return True
        except RuntimeError as exc:
            self.get_logger().warn(f"waiting for fresh sensor data: {exc}")
            return False

    def target_for_action(
        self,
        x0: float,
        y0: float,
        yaw0: float,
    ) -> tuple[float, float, float, float]:
        dr, dc = ACTIONS_8[self.action_idx]

        rel_forward = float(dc) * self.step_distance
        rel_left = float(-dr) * self.step_distance

        tx = x0 + rel_forward * math.cos(yaw0) - rel_left * math.sin(yaw0)
        ty = y0 + rel_forward * math.sin(yaw0) + rel_left * math.cos(yaw0)

        return tx, ty, rel_forward, rel_left

    def execute_target(self, tx: float, ty: float) -> None:
        x0, y0, yaw0, _ = self.pose_xy_yaw_time()

        self.get_logger().warn(
            "EXECUTE START: publishing /cmd_vel at low speed. "
            "Keep emergency stop ready."
        )

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

        self.get_logger().info("drive phase start")
        drive_wall_start = time.monotonic()
        last_debug = -1.0e9

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            x, y, yaw, _ = self.pose_xy_yaw_time()

            dist = math.hypot(tx - x, ty - y)
            wall_elapsed = time.monotonic() - drive_wall_start

            if dist <= self.target_pos_tol:
                break

            if wall_elapsed > self.drive_wall_timeout:
                raise RuntimeError(f"drive timeout, dist={dist:.3f}m")

            target_yaw = math.atan2(ty - y, tx - x)
            yaw_err = norm_angle(target_yaw - yaw)
            wz = max(-0.12, min(0.12, 0.8 * yaw_err))

            if wall_elapsed - last_debug >= self.control_debug_period:
                last_debug = wall_elapsed
                self.get_logger().info(
                    f"drive_debug xy=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):+.1f}deg "
                    f"target=({tx:.3f},{ty:.3f}) dist={dist:.3f} "
                    f"vx={self.linear_speed:+.3f} wz={wz:+.3f}"
                )

            self.publish_velocity(self.linear_speed, wz)

        self.stop()

        x1, y1, yaw1, _ = self.pose_xy_yaw_time()
        self.get_logger().warn(
            f"EXECUTE DONE: final_xy=({x1:.3f},{y1:.3f}) "
            f"final_yaw={math.degrees(yaw1):+.1f}deg "
            f"target_xy=({tx:.3f},{ty:.3f}) "
            f"final_dist={math.hypot(tx-x1, ty-y1):.3f}m"
        )

    def timer_cb(self) -> None:
        if self.done:
            return

        if not self.inputs_are_ready():
            return

        try:
            x, y, yaw, _ = self.pose_xy_yaw_time()
            tx, ty, rel_forward, rel_left = self.target_for_action(x, y, yaw)

            self.get_logger().warn(
                "step_once_plan "
                f"execute={self.execute} "
                f"action={self.action_idx}:{ACTION_NAMES[self.action_idx]} "
                f"start_xy=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):+.1f}deg "
                f"rel_forward={rel_forward:+.3f} rel_left={rel_left:+.3f} "
                f"target_xy=({tx:.3f},{ty:.3f})"
            )

            if not self.execute:
                self.get_logger().warn("DRY PLAN ONLY: execute=false, no movement command sent.")
                self.done = True
                return

            if self.cmd_pub.get_subscription_count() < 1:
                raise RuntimeError("No /cmd_vel subscriber found")

            self.execute_target(tx, ty)
            self.done = True

        except Exception as exc:
            self.stop()
            self.done = True
            self.get_logger().error(f"FAIL: {exc}")
            self.get_logger().error("A stop command has been sent.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RealcarStepOnceSafeNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().warn("KeyboardInterrupt received; sending stop command.")
    finally:
        try:
            node.stop()
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
