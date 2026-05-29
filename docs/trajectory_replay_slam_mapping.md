# Oracle-Planned Trajectory Replay With SLAM Mapping

This route validates a trajectory-level claim:

```text
Oracle/ideal DRL trajectory -> Gazebo waypoint replay -> LaserScan SLAM mapping
```

It is not closed-loop LaserScan-conditioned DRL exploration. The replay node
does not load a checkpoint, run a policy, build `local_snap`, subscribe to
`/scan`, or read `true_grid`. It only reads a CSV trajectory, subscribes to
`/odom`, and publishes `/cmd_vel`.

## Purpose

Use this route to show that:

- A successful oracle/ideal DRL exploration trajectory can be executed by the
  continuous Gazebo robot.
- Real Gazebo `/scan` can be consumed by SLAM while the robot follows that
  trajectory.
- The result is a practical validation path while the LaserScan-to-local_snap
  observation gap remains unresolved.

Do not use this route as a replacement for final probe results or as evidence
of real-sensor closed-loop DRL deployment.

## Inputs And Topics

Trajectory input:

```text
assets/cell035/trajectories/cell035_oracle_trajectory.csv
```

Replay node:

```text
ros2 run drl_explore_bridge drl_trajectory_replay_node
```

Subscribed topics:

```text
/odom
```

Published topics:

```text
/cmd_vel
```

SLAM is started manually outside this node. Expected SLAM inputs and outputs:

```text
input:  /scan
input:  /tf and /tf_static as required by the SLAM configuration
output: /map
```

## Run Procedure

1. Export the oracle/ideal trajectory from DRL-path-finding:

```bash
cd /home/dk/drl_repos/DRL-path-finding
python scripts/export_oracle_cell035_trajectory.py \
  --checkpoint /home/dk/drl_repos/DRL-path-finding/deploy_checkpoints/A_full_method_last.pt \
  --true-grid /home/dk/ros2_repos/ROS2/assets/cell035/grids/random_train_like_seed20260513_true_grid.npy \
  --start-rc 20 36 \
  --cell-size 0.35 \
  --rows 40 \
  --cols 60 \
  --world-x 21.0 \
  --world-y 14.0 \
  --scan-radius-cells 10 \
  --coverage-goal 0.95 \
  --max-steps 400 \
  --output-dir /home/dk/ros2_repos/ROS2/assets/cell035/trajectories
```

2. Confirm the CSV exists:

```bash
ls -lh /home/dk/ros2_repos/ROS2/assets/cell035/trajectories/cell035_oracle_trajectory.csv
```

3. Start the Gazebo world and robot. Confirm the robot has `/odom`, `/scan`,
   and a `/cmd_vel` subscriber:

```bash
ros2 topic list | grep -E '^/odom$|^/cmd_vel$|^/scan$'
ros2 topic hz /odom
ros2 topic hz /scan
```

4. Optionally start `slam_toolbox` manually with a configuration that consumes
   `/scan` and publishes `/map`.

5. Run trajectory replay:

```bash
cd /home/dk/ros2_repos/ROS2
bash scripts/run_cell035_trajectory_replay.sh 2>&1 | tee trajectory_replay_cell035.log
```

6. Check replay completion:

```bash
grep "trajectory_replay_finished" trajectory_replay_cell035.log
cat trajectory_replay_summary.json
```

7. If SLAM was running, save the map, for example with the map saver used in
   your ROS2 environment, and retain `map.pgm`/`map.yaml` or equivalent map
   artifacts outside git unless they are explicitly curated.

## Replay Behavior

The node loads waypoints from CSV with this precedence:

1. `target_x,target_y`
2. `x,y`
3. `target_row,target_col`
4. `row,col`

If only grid coordinates are available, it uses:

```text
x = -world_x / 2.0 + (col + 0.5) * cell_size
y =  world_y / 2.0 - (row + 0.5) * cell_size
```

For each waypoint, the controller:

1. Rotates toward the waypoint center.
2. Drives forward with a small yaw correction.
3. Stops when `target_pos_tol` is reached.
4. Publishes repeated zero `/cmd_vel`.
5. Stops the full run on the first failed or timed-out waypoint.

If the first waypoint is already within `target_pos_tol` of current `/odom`, it
is skipped and logged as `skipped_initial_waypoint`.

## Outputs

Default runtime outputs:

```text
/home/dk/ros2_repos/ROS2/trajectory_replay_summary.json
/home/dk/ros2_repos/ROS2/trajectory_replay_log.csv
```

These are runtime artifacts and should not be committed by default.

Summary fields include:

- `total_waypoints`
- `reached_waypoints`
- `skipped_waypoints`
- `failed_waypoint_idx`
- `reached_ratio`
- `final_xy`
- `stop_reason`
- `elapsed_wall`
- `trajectory_csv`

Waypoint logs include:

- `waypoint_idx`
- `target_x`, `target_y`
- `start_x`, `start_y`
- `final_x`, `final_y`
- `final_dist`
- `reached`
- `rotate_elapsed_sim`, `drive_elapsed_sim`
- `rotate_elapsed_wall`, `drive_elapsed_wall`

## Acceptance Criteria

- `trajectory_replay_summary.json` has `stop_reason=completed`.
- Waypoint reached ratio is at least 95%.
- The final waypoint is reached.
- No collision occurs.
- SLAM publishes `/map`.
- A map artifact such as `map.pgm`/`map.yaml` is saved.

## Paper Wording Boundary

Acceptable wording:

```text
trajectory-level physical executability / SLAM mapping demo
oracle-planned trajectory replay with SLAM mapping
```

Do not write:

```text
real-sensor closed-loop DRL deployment
closed-loop LaserScan-conditioned DRL exploration
```

The LaserScan-to-local_snap bridge remains an observation-gap risk.
`scan_template_los` is not treated as the final deployed closed loop for this
validation route.
