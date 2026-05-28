from __future__ import annotations

from pathlib import Path
import sys
import time

import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


DRL_REPO = Path.home() / "drl_repos" / "DRL-path-finding"
if str(DRL_REPO) not in sys.path:
    sys.path.insert(0, str(DRL_REPO))

from agents.q_value_agent import ExplorationQNetwork, StateTensorAdapter, select_greedy_action
from env.agent_version import LocalObservationModel
from env.block_random_g import RandomMapGenerator
from env.core_cummap import CumulativeBeliefMap
from env.grid_topology import ACTIONS_8, GridTopology


class DrlPolicyProbeNode(Node):
    def __init__(self) -> None:
        super().__init__("drl_policy_probe_node")

        self.declare_parameter(
            "checkpoint_path",
            str(DRL_REPO / "deploy_checkpoints" / "best.pt"),
        )
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("probe_period_sec", 2.0)
        self.declare_parameter("seed", 20261323)

        self._last_scan: LaserScan | None = None
        self._last_scan_time = 0.0

        checkpoint_path = Path(
            self.get_parameter("checkpoint_path").get_parameter_value().string_value
        )
        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        period = float(
            self.get_parameter("probe_period_sec").get_parameter_value().double_value
        )

        self._net = self._load_policy(checkpoint_path)
        self._adapter = StateTensorAdapter(device="cpu")

        self._scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self._on_scan,
            10,
        )

        self._timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(f"DRL repo: {DRL_REPO}")
        self.get_logger().info(f"checkpoint: {checkpoint_path}")
        self.get_logger().info(f"subscribed scan topic: {scan_topic}")
        self.get_logger().warn(
            "Safety: this probe node does NOT publish /cmd_vel. It only prints policy actions."
        )

    def _load_policy(self, checkpoint_path: Path) -> ExplorationQNetwork:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"checkpoint payload must be dict, got {type(payload).__name__}")
        if "online_state_dict" not in payload:
            raise KeyError("checkpoint missing online_state_dict")

        net = ExplorationQNetwork()
        missing, unexpected = net.load_state_dict(payload["online_state_dict"], strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"state_dict mismatch: missing={missing}, unexpected={unexpected}"
            )

        net.eval()

        self.get_logger().info(
            "Loaded policy: "
            f"env_steps={payload.get('env_steps')}, "
            f"learn_steps={payload.get('learn_steps')}, "
            f"train_episode_idx={payload.get('train_episode_idx')}"
        )
        return net

    def _on_scan(self, msg: LaserScan) -> None:
        self._last_scan = msg
        self._last_scan_time = time.time()

    def _build_demo_state(self):
        seed = int(self.get_parameter("seed").get_parameter_value().integer_value)
        grid, start = RandomMapGenerator(40, 60, 6, 0.20).generate_map(seed=seed)
        obs = LocalObservationModel(grid, start)
        cum_map = CumulativeBeliefMap(grid, start, obs.local_snap)
        state_batch, state_meta = self._adapter.build_single_state_tensors(
            cum_map,
            start,
            return_state_meta=True,
        )
        valid = GridTopology.valid_action_indices_fast(GridTopology.free_mask(grid), start)
        return start, state_batch, state_meta, valid

    def _on_timer(self) -> None:
        if self._last_scan is None:
            self.get_logger().warn("No /scan received yet; waiting.")
            return

        age = time.time() - self._last_scan_time
        scan_count = len(self._last_scan.ranges)

        start, state_batch, state_meta, valid = self._build_demo_state()

        with torch.inference_mode():
            q_values, aux = self._net(
                state_batch["advantage_canvas"],
                state_batch["value_block_features"],
                state_batch["value_entry_features"],
                state_batch["value_block_mask"],
                state_batch["value_entry_mask"],
                return_aux=True,
            )

        action = select_greedy_action(q_values, valid_action_indices=valid)
        action_idx = int(action.item())
        action_delta = ACTIONS_8[action_idx]

        self.get_logger().info(
            "policy_probe "
            f"scan_frame={self._last_scan.header.frame_id} "
            f"scan_ranges={scan_count} "
            f"scan_age_sec={age:.3f} "
            f"start_rc={start} "
            f"valid_actions={valid} "
            f"selected_action_idx={action_idx} "
            f"selected_action_delta_rc={action_delta} "
            f"q_values={[round(float(v), 3) for v in q_values.squeeze(0).tolist()]} "
            f"meta_accessible_blocks={state_meta.get('accessible_block_count')} "
            f"aux_keys={list(aux.keys())[:5]}"
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
