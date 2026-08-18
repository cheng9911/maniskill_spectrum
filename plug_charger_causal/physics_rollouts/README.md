# PlugCharger Causal Physics Rollouts

Generated from the same 63 causal reset states under:

```bash
PYTHONNOUSERSITE=1 \
LD_LIBRARY_PATH=/home/rocos/miniconda3/envs/maniskill_download/lib \
MPLCONFIGDIR=/home/rocos/sia/maniskill_spectrum/.matplotlib \
conda run -n maniskill_download \
python maniskill_spectrum/plug_charger_causal/generate_physics_rollouts.py \
  --splits isolated_grid train_mixed
```

These are true ManiSkill `env.step()` trajectories using `PlugChargerCausal-v1`,
`control_mode=pd_ee_delta_pose`, CPU PhysX, no rendering, and a scripted
controller. They are not waypoint-interpolated trajectories.

Files:

- `isolated_grid/trajectory.h5`: 13 single-generator intervention rollouts.
- `isolated_grid/trajectory.json`: metadata for the 13 isolated rollouts.
- `train_mixed/trajectory.h5`: 50 mixed-intervention rollouts.
- `train_mixed/trajectory.json`: metadata for the 50 mixed rollouts.

Each episode contains:

- `causal_delta` `(3,)`
- `tcp_pose` `(200, 7)`
- `charger_pose` `(200, 7)`
- `receptacle_pose` `(200, 7)`
- `goal_pose` `(200, 7)`
- `qpos` `(200, 9)`
- `qvel` `(200, 9)`
- `actions` `(200, 7)`
- `reward` `(200,)`
- `success` `(200,)`
- `obj_to_goal_dist` `(200,)`
- `obj_to_goal_angle` `(200,)`
- `phase` `(200,)`
- `targets/*`

Validation summary:

- `isolated_grid`: 13/13 episodes present, all `causal_delta` labels match the reset dataset.
- `train_mixed`: 50/50 episodes present, all `causal_delta` labels match the reset dataset.
- All trajectories have length 200.
- Current scripted controller does not achieve ManiSkill success on this task: `success_any=0/63`.
- Median final object-to-goal distance is about `0.059 m` for both splits.

Use this as a physics/controller response dataset, not as successful expert
demonstrations.
