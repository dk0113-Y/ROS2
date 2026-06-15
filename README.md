# ROS2 DRL Exploration Simulation Portfolio

## Project Overview

This repository is a ROS2/Gazebo simulation portfolio for autonomous exploration
with a deep reinforcement learning policy. It packages a ROS2 Python bridge
around an external DRL-path-finding research codebase, a curated `cell035`
Gazebo world, robot URDF assets, run scripts, diagnostics, and experiment notes.

The practical purpose is to show how a paper-level autonomous exploration
algorithm can be connected to a Linux/ROS2 simulation stack:

- subscribe to Gazebo `/scan` and `/odom`;
- convert LaserScan/odometry into grid observations used by the DRL policy;
- run policy inference from an external checkpoint;
- convert discrete grid actions into `/cmd_vel` commands;
- replay exported oracle/ideal trajectories for Gazebo executability and SLAM
  mapping demonstrations.

This is a simulation and integration repository. It does not contain training
code, model weights, physical robot logs, or claims of real-world deployment.

## Relation to Research

The repository supports the research direction behind the first-author EI paper
`基于前沿引导�?DRL 端到端自主探索算法`.

The paper-side algorithm and checkpoint are managed outside this repository in a
separate DRL-path-finding project. This repository focuses on the engineering
step after algorithm development: transferring the policy interface into a ROS2
workspace and validating the behavior in Gazebo-style simulation.

The current validation boundary is:

- `oracle_los` is a diagnostic upper bound that uses the true grid and the
  training observation model. It is not deployment-like sensing.
- LaserScan bridge modes such as `ray_project`, `los_compatible`, and
  `scan_template_los` are simulation-side observation adapters.
- trajectory replay is an executability and mapping demonstration for exported
  waypoints, not closed-loop LaserScan-conditioned DRL exploration.

## Goals

- Organize the ROS2 package as a readable internship portfolio.
- Keep model weights and training data outside git.
- Provide a reproducible path for building the ROS2 package.
- Document how to start bridge, diagnostic, and trajectory replay routes.
- Preserve clear wording around what is implemented, what is diagnostic, and
  what remains a limitation.

## Tech Stack

- ROS2 Python package with `ament_python`
- `rclpy` nodes and ROS2 parameters
- ROS2 message types: `sensor_msgs/LaserScan`, `nav_msgs/Odometry`,
  `geometry_msgs/Twist`
- Gazebo-style world and URDF robot assets
- Python, NumPy, and PyTorch-based policy inference through the external
  DRL-path-finding repository
- Bash run scripts for Linux/ROS2 workflows
- `colcon` build workflow
- Lightweight lint test files generated for ROS2 Python packages

## Repository Structure

```text
.
├── assets/
�?  └── cell035/
�?      ├── grids/          # true grid and metadata for the cell035 map
�?      ├── trajectories/   # placeholder for exported trajectory CSV files
�?      ├── urdf/           # mini 4WD robot URDF variants with laser
�?      └── worlds/         # random_train_like_seed20260513 Gazebo world
├── docs/
�?  ├── CHECKPOINT_USAGE.md
�?  ├── cell035_bridge_diagnostics.md
�?  ├── standalone_gazebo_bridge.md
�?  └── trajectory_replay_slam_mapping.md
├── experiments/
�?  └── cell035_baselines/  # text summaries from previous cell035 runs
├── ros2_ws/
�?  └── src/
�?      └── drl_explore_bridge/
�?          ├── drl_explore_bridge/
�?          �?  ├── drl_standalone_gazebo_bridge_node.py
�?          �?  ├── drl_trajectory_replay_node.py
�?          �?  ├── drl_policy_step_once_node.py
�?          �?  ├── drl_policy_multi_step_node.py
�?          �?  ├── drl_policy_probe_node.py
�?          �?  ├── policy_probe_node.py
�?          �?  └── scan_to_local_snap_node.py
�?          ├── package.xml
�?          ├── setup.py
�?          └── test/
└── scripts/
    ├── run_cell035_standalone_bridge.sh
    ├── run_cell035_local_snap_diagnostic.sh
    ├── run_cell035_scan_template_los_diagnostic.sh
    └── run_cell035_trajectory_replay.sh
```

## Core Modules

| Module | Role |
| --- | --- |
| `drl_standalone_gazebo_bridge_node.py` | Main Gazebo bridge. Loads the external DRL core and checkpoint, subscribes to `/scan` and `/odom`, maintains a cumulative map, selects actions, and publishes `/cmd_vel`. |
| `drl_trajectory_replay_node.py` | Replays an exported CSV trajectory as waypoint-following `/cmd_vel` commands using `/odom`. It does not load the DRL policy or consume `/scan`. |
| `drl_policy_step_once_node.py` | Older/experimental single-step bridge path with LaserScan-to-local-snap conversion and one-step action execution. |
| `drl_policy_multi_step_node.py` | Multi-step wrapper around the step-once route with coverage/no-progress stopping logic. |
| `drl_policy_probe_node.py` | Policy probe that uses `/scan`, `/odom`, true grid, and checkpoint inputs to inspect selected policy actions. |
| `policy_probe_node.py` | Minimal probe that subscribes to `/scan` and prints policy action information from a generated demo state; it does not publish `/cmd_vel`. |
| `scan_to_local_snap_node.py` | Diagnostic node that converts `/scan` and `/odom` into an ASCII local grid snapshot. |

## ROS2 Workspace

The ROS2 workspace lives under `ros2_ws/`. The package name is
`drl_explore_bridge` and it is installed as an `ament_python` package.

Console entry points are defined in
`ros2_ws/src/drl_explore_bridge/setup.py`:

```text
policy_probe_node
scan_to_local_snap
drl_policy_probe
drl_policy_step_once_node
drl_policy_multi_step_node
drl_standalone_gazebo_bridge_node
drl_trajectory_replay_node
```

No ROS2 launch files are currently included. The repository uses Bash scripts in
`scripts/` as explicit runbooks for the current `cell035` validation route.

## Quick Start

Expected environment:

- Linux with ROS2 Humble or compatible ROS2 Python tooling;
- `colcon`;
- Python dependencies required by ROS2 and the external DRL project;
- an external DRL-path-finding checkout available through `DRL_REPO`;
- a checkpoint file supplied through `CHECKPOINT`;
- Gazebo world, robot, `/scan`, `/odom`, and `/cmd_vel` controller started
  before running closed-loop bridge scripts.

Clone and enter the repository:

```bash
git clone <this-repository-url> ROS2
cd ROS2
```

Configure paths for your machine:

```bash
export ROS_REPO="$PWD"
export ROS_WS="$ROS_REPO/ros2_ws"
export DRL_REPO="$HOME/drl_repos/DRL-path-finding"
export CHECKPOINT="$DRL_REPO/deploy_checkpoints/A_full_method_last.pt"
export TRUE_GRID="$ROS_REPO/assets/cell035/grids/random_train_like_seed20260513_true_grid.npy"
```

## Build

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

To build only this package:

```bash
colcon build --packages-select drl_explore_bridge --symlink-install
```

## Run

Start Gazebo, the robot model, `/scan`, `/odom`, and the velocity controller
first. Then run the standalone bridge:

```bash
bash scripts/run_cell035_standalone_bridge.sh
```

Useful overrides:

```bash
SCAN_BRIDGE_MODE=scan_template_los \
MAX_STEPS=40 \
COVERAGE_GOAL=0.95 \
bash scripts/run_cell035_standalone_bridge.sh
```

Run the no-motion local-snap diagnostic:

```bash
bash scripts/run_cell035_local_snap_diagnostic.sh
```

Run the template-based LaserScan diagnostic:

```bash
bash scripts/run_cell035_scan_template_los_diagnostic.sh
```

Replay an exported oracle/ideal trajectory:

```bash
bash scripts/run_cell035_trajectory_replay.sh
```

For trajectory replay, the expected CSV path is:

```text
assets/cell035/trajectories/cell035_oracle_trajectory.csv
```

That CSV is not committed in the current repository; it must be exported from
the external DRL-path-finding project before replay.

## Experiments

The `experiments/cell035_baselines/` directory contains text summaries from a
previous `cell035` simulation route. The recorded run describes:

- world: `random_train_like_seed20260513_cell035.world`;
- grid: `random_train_like_seed20260513_true_grid.npy`;
- robot URDF: `mini_4wd_robot_with_laser_drive_cell035.urdf`;
- parameters: `cell_size=0.35`, `rows=40`, `cols=60`,
  `scan_radius_cells=10`, `max_steps=400`, `coverage_goal=0.95`;
- result summary: coverage goal reached with `best_coverage=0.9504` in
  207 steps for the recorded run.

Treat these files as historical simulation evidence for this repository state,
not as a hardware result or a broad benchmark.

## Current Status and Limitations

Implemented:

- ROS2 package layout and console entry points;
- Gazebo bridge node for `/scan`, `/odom`, and `/cmd_vel`;
- multiple LaserScan-to-local-grid diagnostic modes;
- trajectory replay node for exported waypoints;
- curated `cell035` grid, world, URDF, and baseline text outputs;
- run scripts for bridge, diagnostics, and replay.

Limitations:

- model weights are intentionally not committed;
- training code and DRL core modules are external dependencies;
- trajectory CSV files are placeholders unless exported separately;
- `oracle_los` depends on the true grid and is diagnostic only;
- LaserScan bridge modes remain simulation observation adapters, not proven
  physical robot perception;
- no launch files are currently provided;
- URDF mesh paths may require a local `wheeltec_robot_urdf` asset package.

## Internship Skill Mapping

This repository can support internship discussions around:

- ROS2 workspace organization: `ros2_ws/src/drl_explore_bridge`,
  `package.xml`, `setup.py`, console scripts, and `colcon` builds.
- Linux/ROS2 workflow: Bash run scripts, environment variables, sourced ROS2
  setup files, and package-selective builds.
- Robot simulation integration: Gazebo world/URDF assets, `/scan`, `/odom`,
  and `/cmd_vel` topic wiring.
- DRL/path-planning/autonomous exploration transfer: policy checkpoint loading,
  grid observations, cumulative maps, action selection, and discrete-to-motion
  command conversion.
- ROS2 node encapsulation: separate nodes for closed-loop bridge, local-snap
  diagnostics, policy probing, and trajectory replay.
- Experiment organization: curated `cell035` baseline summaries and explicit
  diagnostic interpretation notes.
- Engineering documentation: clear reproduction commands, boundaries between
  diagnostic and deployment-like routes, and sensitive-file exclusions for
  checkpoints and runtime artifacts.
