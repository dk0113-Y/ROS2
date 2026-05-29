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

`scan_template_los` addresses the main observation mismatch by making
DRL-path-finding `RadarSensor.local_ray_templates` the outer loop. For each
training LOS ray, it walks cells from near to far with the same local
`(rel_r, rel_c, local_r, local_c)` template coordinates used by
`LocalObservationModel`: visible free cells become empty, the first visible
LaserScan hit becomes obstacle, and cells beyond the first hit remain
invisible. It writes by template local index, so shoulder cells from the
training footprint are preserved.

`scan_template_los` is still a LaserScan-only approximation. It does not read
`true_grid`; finite ranges from `/scan` are the only obstacle evidence. Corner
blocking is best effort: on diagonal ray steps, the bridge stops only when both
side cells have already been inferred as visible obstacles in the same local
snap. Exact corner blocking uses true side cells and remains available only in
the `oracle_los` diagnostic upper bound.

These differences can reduce initial known/free cells and shift Q-values enough
to change the first action.

## Diagnostic Command

Start Gazebo world, robot, `/scan`, and `/odom`, then run:

```bash
bash scripts/run_cell035_local_snap_diagnostic.sh
```

To make the deployed bridge mode itself `scan_template_los` while still
comparing all four local snaps:

```bash
bash scripts/run_cell035_scan_template_los_diagnostic.sh
```

or:

```bash
SCAN_BRIDGE_MODE=scan_template_los bash scripts/run_cell035_local_snap_diagnostic.sh
```

The diagnostic script uses:

- `diagnostic_compare_local_snaps=true`
- `diagnostic_print_ascii=true`
- `diagnostic_only_first_step=true`
- `diagnostic_no_motion=true`

With `diagnostic_no_motion=true`, the node waits for `/scan` and `/odom`,
prints the alignment report, and exits without executing a policy action or
publishing `/cmd_vel`.

After the no-motion check, validate progressively:

1. `diagnostic_no_motion=true`
2. `MAX_STEPS=5`
3. `MAX_STEPS=40`
4. `MAX_STEPS=300` or `MAX_STEPS=400`

## Reading The Output

The `local_snap_alignment` line reports:

- per-snap `known_count`, `empty_count`, `obstacle_count`, `invisible_count`
- `oracle_vs_ray_mismatch_count`
- `oracle_vs_los_mismatch_count`
- `oracle_vs_scan_template_los_mismatch_count`
- `ray_vs_los_mismatch_count`
- `ray_vs_scan_template_los_mismatch_count`
- `los_vs_scan_template_los_mismatch_count`
- cases where oracle-visible free cells become invisible in LaserScan modes
- cases where oracle-visible obstacle cells become empty or invisible
- LaserScan angle/range metadata and finite/inf/nan range counts

Current acceptance targets for the first no-motion diagnostic are:

- `scan_template_los_known_count` should be closer to `oracle_los` than
  `los_compatible` is.
- `oracle_vs_scan_template_los_mismatch_count` should be below the previous
  `los_compatible` value of 54, ideally near or below the `ray_project` value
  of 39.
- `oracle_obstacle_scan_template_los_invisible_count` should be below the
  previous `los_compatible` value of 22.
- First action returning to `SW` is a strong signal, but it is not a static code
  acceptance condition; confirm it in Linux Gazebo diagnostics.

The ASCII maps use:

- `?` invisible
- `.` empty
- `#` obstacle
- `A` agent center

The diff map compares each LaserScan mode to oracle:

- space: both match oracle
- `r`: only `ray_project` differs from oracle
- `l`: only `los_compatible` differs from oracle
- `t`: only `scan_template_los` differs from oracle
- `b`: `ray_project` and `los_compatible` differ from oracle
- `R`: `ray_project` and `scan_template_los` differ from oracle
- `L`: `los_compatible` and `scan_template_los` differ from oracle
- `a`: all three LaserScan modes differ from oracle
