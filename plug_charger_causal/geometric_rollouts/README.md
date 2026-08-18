# PlugCharger-Causal Geometric Rollouts

These files contain trajectory-shaped datasets generated from the controlled
reset states.

They are **not** ManiSkill physics rollouts. They are synthetic TCP pose
trajectories interpolated through the same geometric targets used by the official
PlugCharger motion-planning solution:

```text
approach_start_tcp -> pre_insert_target -> insert_target
```

Each trajectory has 80 frames:

- 20 frames: `approach_hold`
- 40 frames: `align_to_pre_insert`
- 20 frames: `insert`

## Files

```text
isolated_grid/trajectory.h5
isolated_grid/trajectory.json

train_mixed/trajectory.h5
train_mixed/trajectory.json
```

Each episode contains:

```text
causal_delta      (3,)
tcp_pose          (80, 7)
charger_pose      (80, 7)
receptacle_pose   (80, 7)
goal_pose         (80, 7)
qpos              (80, 9)
qvel              (80, 9)
actions           (79, 4)
phase             (80,)
```

Pose format is:

```text
[x, y, z, qw, qx, qy, qz]
```

Use this dataset to test the `P(s)` estimation pipeline and response-error
metrics before real motion-planning or policy rollouts are available.
