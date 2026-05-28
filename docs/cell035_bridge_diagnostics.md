# Cell035 Bridge Diagnostics

## Interpretation Boundary

The current cell035 results indicate that the checkpoint, network interface,
`StateTensorAdapter`, `true_grid`, starting cell, and Gazebo control loop are
mostly aligned:

- Pure-grid reaches `coverage=0.9509` in 205 steps with first action `SW`.
- Gazebo `oracle_los` reaches `coverage=0.9509` in 206 steps with first action
  `SW`.

`oracle_los` uses `true_grid` with DRL-path-finding `LocalObservationModel` and
`RadarSensor`. Treat it as a diagnostic upper bound for observation alignment,
not as deployment-like sensing.

The first-action mismatch in LaserScan modes points to the observation bridge:

- `ray_project` first action is `SE` and stalls at `coverage=0.8556` after 400
  steps.
- `los_compatible` first action is `S`, and its initial known/coverage count is
  lower than `oracle_los`.

Do not write the `oracle_los` result as physical robot deployment or real-world
indoor generalization.

## Static Analysis

`ray_project` integrates LaserScan beams with:

```text
angle_world = robot_yaw + laser_yaw_in_base + scan_angle
dc = round(rel_x / cell_size)
dr = round(-rel_y / cell_size)
```

That row/column sign convention is consistent with the cell035 grid mapping.
The most likely issue is range semantics: every finite LaserScan range is
treated as an obstacle hit, so finite values near `range_max` can become false
obstacle endpoints. Beam projection also rasterizes sparse beam rays rather
than the training full-disk LOS cell set.

`los_compatible` is closer to training because it enumerates local cells and
queries LaserScan by target angle. It is still only an approximation:

- It does not use `RadarSensor.local_ray_templates`.
- It does not reproduce training corner blocking from `LocalObservationModel`.
- It uses a center-distance threshold and a small beam window, so cells that are
  free in oracle LOS can remain invisible.
- Its circular support omits the `RadarSensor` cardinal shoulder cells at the
  scan-radius boundary.

These differences can reduce initial known/free cells and shift Q-values enough
to change the first action.

## Diagnostic Command

Start Gazebo world, robot, `/scan`, and `/odom`, then run:

```bash
bash scripts/run_cell035_local_snap_diagnostic.sh
```

The diagnostic script uses:

- `diagnostic_compare_local_snaps=true`
- `diagnostic_print_ascii=true`
- `diagnostic_only_first_step=true`
- `diagnostic_no_motion=true`

With `diagnostic_no_motion=true`, the node waits for `/scan` and `/odom`,
prints the alignment report, and exits without executing a policy action or
publishing `/cmd_vel`.

## Reading The Output

The `local_snap_alignment` line reports:

- per-snap `known_count`, `empty_count`, `obstacle_count`, `invisible_count`
- `oracle_vs_ray_mismatch_count`
- `oracle_vs_los_mismatch_count`
- `ray_vs_los_mismatch_count`
- cases where oracle-visible free cells become invisible in LaserScan modes
- cases where oracle-visible obstacle cells become empty or invisible
- LaserScan angle/range metadata and finite/inf/nan range counts

The ASCII maps use:

- `?` invisible
- `.` empty
- `#` obstacle
- `A` agent center

The diff map compares each LaserScan mode to oracle:

- space: both match oracle
- `r`: only `ray_project` differs from oracle
- `l`: only `los_compatible` differs from oracle
- `b`: both differ from oracle
