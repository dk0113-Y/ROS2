from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import rclpy
import torch

from drl_explore_bridge.drl_policy_step_once_node import (
    DrlPolicyStepOnceNode,
    ACTIONS_8,
    ACTION_NAMES,
    EMPTY,
    INVISIBLE,
    CumulativeBeliefMap,
    GridTopology,
    select_greedy_action,
)


class DrlPolicyMultiStepNode(DrlPolicyStepOnceNode):
    """
    Persistent cumulative-belief multi-step Gazebo runner.

    Scope:
    - Gazebo simulation only.
    - Keeps one CumulativeBeliefMap across the whole run.
    - Updates that map with /scan + /odom before each policy inference.
    - Executes one closed-loop target-cell action per step.
    - Stops on max_steps, coverage_goal, no-progress, timeout, or failure.

    This is still not for the real chassis.
    """

    def __init__(self) -> None:
        super().__init__()

        self.declare_parameter("max_steps", 200)
        self.declare_parameter("coverage_goal", 0.90)
        self.declare_parameter("step_pause_sec", 0.3)
        self.declare_parameter("multi_wall_timeout", 1800.0)
        self.declare_parameter("no_progress_limit", 40)
        self.declare_parameter("coverage_epsilon", 1e-4)
        self.declare_parameter("recent_traj_limit", 64)

        self.max_steps = int(self.get_parameter("max_steps").value)
        self.coverage_goal = float(self.get_parameter("coverage_goal").value)
        self.step_pause_sec = float(self.get_parameter("step_pause_sec").value)
        self.multi_wall_timeout = float(self.get_parameter("multi_wall_timeout").value)
        self.no_progress_limit = int(self.get_parameter("no_progress_limit").value)
        self.coverage_epsilon = float(self.get_parameter("coverage_epsilon").value)
        self.recent_traj_limit = int(self.get_parameter("recent_traj_limit").value)

        self.cum_map: Optional[CumulativeBeliefMap] = None
        self.trajectory_positions: list[tuple[int, int]] = []

        self.step_count = 0
        self.best_coverage = -1.0
        self.no_progress_count = 0
        self.multi_started_wall: Optional[float] = None

        self.get_logger().info(
            "drl_policy_multi_step_node configured: "
            f"execute_once={self.execute_once}, "
            f"max_steps={self.max_steps}, "
            f"coverage_goal={self.coverage_goal:.4f}, "
            f"multi_wall_timeout={self.multi_wall_timeout:.1f}s, "
            f"no_progress_limit={self.no_progress_limit}, "
            f"persistent_cum_map=True"
        )

    def current_agent_state_checked(self) -> tuple[int, int]:
        x, y, _, _ = self.pose_xy_yaw_time()
        agent_state = self.world_xy_to_grid_rc(x, y)

        ar, ac = agent_state
        if not (0 <= ar < self.true_grid.shape[0] and 0 <= ac < self.true_grid.shape[1]):
            raise RuntimeError(f"agent_state outside true_grid: {agent_state}")

        if int(self.true_grid[ar, ac]) != int(EMPTY):
            raise RuntimeError(
                f"agent is not on EMPTY cell: rc={agent_state}, value={int(self.true_grid[ar, ac])}"
            )

        return agent_state

    def update_persistent_belief(self, agent_state: tuple[int, int]) -> CumulativeBeliefMap:
        if self.latest_scan is None or self.latest_odom is None:
            raise RuntimeError("waiting for /scan and /odom")

        local_snap = self.build_local_snap(self.latest_scan, self.latest_odom)

        if self.cum_map is None:
            self.cum_map = CumulativeBeliefMap(
                true_grid=self.true_grid,
                start_state=agent_state,
                first_local_snap=local_snap,
            )
            self.get_logger().info(
                "persistent_cum_map initialized "
                f"start_rc={agent_state} "
                f"coverage={float(self.cum_map.coverage_rate):.4f} "
                f"known={int(np.count_nonzero(self.cum_map.map != INVISIBLE))} "
                f"frontier={int(np.count_nonzero(self.cum_map.get_frontier_u8() > 0))}"
            )
        else:
            update_result = self.cum_map.update(agent_state, local_snap)
            self.get_logger().info(
                "persistent_cum_map updated "
                f"rc={agent_state} "
                f"update_result={update_result} "
                f"coverage={float(self.cum_map.coverage_rate):.4f} "
                f"known={int(np.count_nonzero(self.cum_map.map != INVISIBLE))} "
                f"frontier={int(np.count_nonzero(self.cum_map.get_frontier_u8() > 0))}"
            )

        return self.cum_map

    def infer_action_persistent(self):
        agent_state = self.current_agent_state_checked()
        cum_map = self.update_persistent_belief(agent_state)

        if not self.trajectory_positions or self.trajectory_positions[-1] != agent_state:
            self.trajectory_positions.append(agent_state)

        recent_traj = self.trajectory_positions[-self.recent_traj_limit :]

        state_batch, _ = self.adapter.build_single_state_tensors(
            cum_map,
            agent_state,
            recent_trajectory_positions=recent_traj,
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

        valid = GridTopology.valid_action_indices_fast(
            GridTopology.free_mask(self.true_grid),
            agent_state,
        )
        if len(valid) <= 0:
            raise RuntimeError(f"no valid action at {agent_state}")

        action_idx = int(select_greedy_action(q, valid_action_indices=valid).item())
        return action_idx, valid, q, cum_map, agent_state

    def timer_cb(self) -> None:
        if self.done:
            return

        if self.latest_scan is None or self.latest_odom is None:
            self.get_logger().warn("waiting for /scan and /odom ...")
            return

        if not self.execute_once:
            self.get_logger().error(
                "FAIL: persistent multi-step node requires execute_once:=true"
            )
            self.done = True
            return

        if self.cmd_pub is None:
            self.get_logger().error("FAIL: cmd_pub is None while execute_once=True")
            self.done = True
            return

        if self.cmd_pub.get_subscription_count() < 1:
            self.get_logger().error("FAIL: No /cmd_vel subscriber found")
            self.done = True
            return

        if self.multi_started_wall is None:
            self.multi_started_wall = time.monotonic()

        try:
            while rclpy.ok() and not self.done:
                wall_elapsed = time.monotonic() - self.multi_started_wall

                if wall_elapsed > self.multi_wall_timeout:
                    self.get_logger().warn(
                        f"STOP: multi_wall_timeout reached, wall_elapsed={wall_elapsed:.1f}s"
                    )
                    break

                if self.step_count >= self.max_steps:
                    self.get_logger().info(
                        f"STOP: max_steps reached, step_count={self.step_count}"
                    )
                    break

                if self.best_coverage >= self.coverage_goal:
                    self.get_logger().info(
                        f"STOP: coverage_goal reached, best_coverage={self.best_coverage:.4f}"
                    )
                    break

                if self.no_progress_count >= self.no_progress_limit:
                    self.get_logger().warn(
                        "STOP: no_progress_limit reached, "
                        f"no_progress_count={self.no_progress_count}, "
                        f"best_coverage={self.best_coverage:.4f}"
                    )
                    break

                action_idx, valid, q, cum_map, agent_state = self.infer_action_persistent()
                x, y, yaw, _ = self.pose_xy_yaw_time()

                dr, dc = ACTIONS_8[action_idx]
                target_state = (agent_state[0] + int(dr), agent_state[1] + int(dc))
                tx, ty = self.grid_rc_to_cell_center_xy(*target_state)

                coverage = float(cum_map.coverage_rate)
                frontier = int(np.count_nonzero(cum_map.get_frontier_u8() > 0))
                known = int(np.count_nonzero(cum_map.map != INVISIBLE))

                q_str = " ".join(
                    f"{i}:{ACTION_NAMES[i]}={float(q[i]):+.2f}{'*' if i == action_idx else ''}"
                    for i in range(len(ACTION_NAMES))
                )

                if coverage > self.best_coverage + self.coverage_epsilon:
                    self.best_coverage = coverage
                    self.no_progress_count = 0
                else:
                    self.no_progress_count += 1

                self.get_logger().info(
                    "persistent_multi_step_plan "
                    f"step={self.step_count + 1}/{self.max_steps} "
                    f"rc={agent_state} xy=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):+.1f}deg "
                    f"valid={list(valid)} selected={action_idx}:{ACTION_NAMES[action_idx]} "
                    f"target_rc={target_state} target_xy=({tx:.3f},{ty:.3f}) "
                    f"known={known} frontier={frontier} coverage={coverage:.4f} "
                    f"best_coverage={self.best_coverage:.4f} "
                    f"no_progress_count={self.no_progress_count} "
                    f"q=[{q_str}]"
                )

                self.execute_target_cell(action_idx, agent_state)
                self.step_count += 1

                x2, y2, yaw2, _ = self.pose_xy_yaw_time()
                rc2 = self.world_xy_to_grid_rc(x2, y2)

                self.get_logger().info(
                    "persistent_multi_step_done "
                    f"step={self.step_count}/{self.max_steps} "
                    f"current_rc={rc2} xy=({x2:.3f},{y2:.3f}) yaw={math.degrees(yaw2):+.1f}deg "
                    f"coverage={coverage:.4f} "
                    f"best_coverage={self.best_coverage:.4f} "
                    f"frontier={frontier} "
                    f"known={known} "
                    f"no_progress_count={self.no_progress_count}"
                )

                self.stop()
                time.sleep(self.step_pause_sec)

            self.stop()
            self.done = True
            final_coverage = self.best_coverage if self.best_coverage >= 0 else 0.0
            self.get_logger().info(
                "persistent_multi_step node finished: "
                f"steps={self.step_count}, "
                f"best_coverage={final_coverage:.4f}, "
                f"no_progress_count={self.no_progress_count}"
            )

        except Exception as exc:
            self.stop()
            self.done = True
            self.get_logger().error(f"FAIL: {exc}")
            self.get_logger().error("A stop command has been sent.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = DrlPolicyMultiStepNode()
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
