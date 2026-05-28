# Standalone Gazebo Bridge

`drl_standalone_gazebo_bridge_node` is a single-file ROS2 node for Gazebo
simulation validation of the DRL exploration policy. It subscribes to `/scan`
and `/odom`, maintains the cumulative belief map, runs the DRL policy, converts
the selected discrete action to a target grid cell, and publishes `/cmd_vel`
until the target cell or a stop condition is reached.

## Scope

- The node is independent from the older ROS2 bridge node files and does not
  import them.
- It still imports the DRL-path-finding algorithm core modules:
  `ExplorationQNetwork`, `StateTensorAdapter`, `select_greedy_action`,
  `CumulativeBeliefMap`, `GridTopology`, `ACTIONS_8`, `EMPTY`, `INVISIBLE`,
  and `OBSTACLE`.
- It does not copy the DRL network, state adapter, or cumulative map
  implementation into this ROS2 repository.
- The intended research boundary is Gazebo-based transfer validation and a
  high-fidelity simulation prototype. It does not support claims of physical
  robot deployment or real-world indoor generalization.

## Cell035 Defaults

- `cell_size=0.35`
- `rows=40`
- `cols=60`
- `world_x=21.0`
- `world_y=14.0`
- `scan_radius_cells=10`
- `scan_bridge_mode=los_compatible`
- `coverage_goal=0.95`
- `max_steps=400`
- `no_progress_limit=120`

The coordinate convention maps `x in [-world_x/2, world_x/2]` to columns and
`y in [-world_y/2, world_y/2]` to rows:

```text
col = floor((x + world_x / 2) / cell_size)
row = floor((world_y / 2 - y) / cell_size)
```

## Scan Bridge Modes

- `ray_project`: projects each LaserScan beam into the local grid, marks beam
  traversals as free, and marks finite beam endpoints as obstacles.
- `los_compatible`: enumerates cells in the local disk and uses LaserScan
  ranges to approximate the training LOS local observation semantics.
- `oracle_los`: diagnostic-only mode that uses `true_grid` through the
  DRL-path-finding `LocalObservationModel` and `RadarSensor`.

## Run

Start the Gazebo world, robot, `/scan`, `/odom`, and the `/cmd_vel` controller
before launching the bridge. Then run:

```bash
bash scripts/run_cell035_standalone_bridge.sh
```

Common environment overrides:

```bash
CHECKPOINT=/path/to/last.pt \
TRUE_GRID=/path/to/random_train_like_seed20260513_true_grid.npy \
SCAN_BRIDGE_MODE=los_compatible \
MAX_STEPS=40 \
bash scripts/run_cell035_standalone_bridge.sh
```

The script defaults to:

- `ROS_REPO=$HOME/ROS2`
- `ROS_WS=$HOME/ROS2/ros2_ws`
- `DRL_REPO=$HOME/drl_repos/DRL-path-finding`

It builds `drl_explore_bridge`, sources the workspace, and runs:

```bash
ros2 run drl_explore_bridge drl_standalone_gazebo_bridge_node --ros-args ...
```

## Logs

The node emits structured log prefixes:

- `startup_config`
- `checkpoint_loaded`
- `true_grid_loaded`
- `waiting_for_scan_odom`
- `bridge_step_plan`
- `rotate_debug`
- `drive_debug`
- `bridge_step_done`
- `bridge_finished`
- `bridge_failure`

## Common Failures

- `/scan` is not available.
- `/odom` is not available.
- `/cmd_vel` has no subscriber.
- `checkpoint_path` does not exist.
- `true_grid_path` does not exist.
- `true_grid` shape does not match `rows x cols`.
- `cell_size`, `world_x`, or `world_y` do not match the Gazebo world.
- `laser_yaw_in_base` needs calibration for the robot sensor frame.
