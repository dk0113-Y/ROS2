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
  traversals as free, and marks finite beam endpoints as obstacles. This is the
  beam-rasterization baseline.
- `los_compatible`: enumerates cells in the local disk and uses LaserScan
  ranges to approximate the training LOS local observation semantics. It is the
  older cell-center LOS approximation and does not use training ray templates.
- `scan_template_los`: deployment-like LaserScan bridge that uses
  DRL-path-finding `RadarSensor.local_ray_templates` as the primary loop. For
  each training LOS ray it walks cells from near to far, marks visible free
  cells, marks the first visible obstacle, and leaves cells behind the first hit
  invisible. It does not use `true_grid`.
- `oracle_los`: diagnostic-only mode that uses `true_grid` through the
  DRL-path-finding `LocalObservationModel` and `RadarSensor`. Treat it as an
  upper bound for alignment, not as a real sensor or deployment result.

`scan_template_los` is intended to match the training-side
`RadarSensor.local_ray_templates` / `LocalObservationModel` footprint more
closely than `los_compatible`, including template shoulder cells. Its corner
blocking is LaserScan-only best effort: when a ray takes a diagonal step, the
bridge blocks the ray only if both side cells have already been inferred as
visible obstacles in the current local snap. Exact corner blocking requires
`true_grid` and remains available only through `oracle_los` diagnostics.

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

## Local Snap Diagnostics

For cell035 observation alignment, use:

```bash
bash scripts/run_cell035_local_snap_diagnostic.sh
```

The diagnostic compares `oracle_los`, `ray_project`, and `los_compatible`
local snaps plus `scan_template_los` from the same `/scan` and `/odom` frame.
It can run with `diagnostic_no_motion=true`, which prints the report and exits
without executing a policy action or publishing `/cmd_vel`.

To explicitly run the template-based bridge diagnostic:

```bash
bash scripts/run_cell035_scan_template_los_diagnostic.sh
```

Recommended validation order is:

1. `diagnostic_no_motion=true`
2. `MAX_STEPS=5`
3. `MAX_STEPS=40`
4. `MAX_STEPS=300` or `MAX_STEPS=400`

See `docs/cell035_bridge_diagnostics.md` for interpretation. In short,
pure-grid and `oracle_los` matching first action `SW` indicates that the policy
and Gazebo control chain are basically aligned. LaserScan modes choosing `SE`
or `S` at the same start point indicates an observation-bridge mismatch.
`oracle_los` is a diagnostic upper bound, not deployment-like sensing.

## Common Failures

- `/scan` is not available.
- `/odom` is not available.
- `/cmd_vel` has no subscriber.
- `checkpoint_path` does not exist.
- `true_grid_path` does not exist.
- `true_grid` shape does not match `rows x cols`.
- `cell_size`, `world_x`, or `world_y` do not match the Gazebo world.
- `laser_yaw_in_base` needs calibration for the robot sensor frame.
