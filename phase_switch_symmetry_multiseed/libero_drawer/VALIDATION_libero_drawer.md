# LIBERO Drawer Relation Probe Validation

Date: 2026-08-31

## Scope

This is a controlled LIBERO/robosuite supplement for a structurally different
task relation: prismatic drawer sliding.  The collector loads the real LIBERO
`open_the_middle_drawer_of_the_cabinet` MuJoCo model and samples the cabinet
middle-drawer joint under the same six-generator context basis used by the SE(3)
suite:

`[du, dv, dw, roll, pitch, yaw]`

The task curve `object_pose` is stored in a rail-aligned task frame so that `du`
is the drawer rail direction.  The raw MuJoCo body pose is also stored as
`libero_body_pose_world` for provenance.

## Oracle

Only the rail translation is relevant.

- `du`: ramps from 0 to 1 during `slide_open`, then stays 1.
- `dv`, `dw`, `roll`, `pitch`, `yaw`: suppressed by the drawer's prismatic
  mechanism, despite being present in the ghost target context.

Threshold metric: `M_prismatic` is correct iff the max over active phases
(`slide_open` + `hold_open`) has `alpha_du > 0.5` and all other generators
`< 0.5`.

## Data Integrity

Generated files:

- `phase_switch_symmetry_rollouts_libero_drawer/drawer_seed_20260818.h5`
- `phase_switch_symmetry_rollouts_libero_drawer/drawer_seed_20270818.h5`
- `phase_switch_symmetry_rollouts_libero_drawer/drawer_seed_20280818.h5`

Validation:

- each seed contains 75/75 usable episodes;
- phase codes are `[3, 4, 5, 6]`;
- pose frame is `rail_aligned_task_frame`;
- the context manifest hash matches `libero_drawer_experiment.json`.

## Results

`M_prismatic` accuracy:

| model | N=8 | N=15 | N=30 |
| --- | ---: | ---: | ---: |
| Frame-weighted (SE(3)) | 0.000 | 0.000 | 0.000 |
| Full operator (SE(3)) | 1.000 | 1.000 | 1.000 |
| Pdiag pointwise (SE(3)) | 1.000 | 1.000 | 1.000 |
| Pdiag finite (SE(3)) | 1.000 | 1.000 | 1.000 |

Pdiag finite active-phase max alpha:

| generator | N=8 | N=15 | N=30 |
| --- | ---: | ---: | ---: |
| du | 1.003 | 0.984 | 1.010 |
| dv | 3.98e-08 | 2.58e-09 | 7.83e-09 |
| dw | 5.18e-08 | 2.65e-08 | 7.90e-09 |
| roll | 2.40e-07 | 2.71e-08 | 1.28e-08 |
| pitch | 2.54e-08 | 1.50e-07 | 4.22e-08 |
| yaw | 8.28e-09 | 3.88e-09 | 5.83e-09 |

Pdiag finite `E_alpha`: 2.42e-04 / 1.31e-04 / 1.81e-05 for N=8/15/30.

## TP-GMM Projected Baseline

The repository TP-GMM baseline is an SE(2) trajectory model, so it is evaluated
on the rail-plane projection `[du, dv, yaw]` rather than as a full six-generator
SE(3) relevance model.  This makes it a useful classical trajectory baseline,
but it cannot certify suppression of `dw`, `roll`, or `pitch`.

Projected `M_prismatic` accuracy:

| model | N=8 | N=15 | N=30 |
| --- | ---: | ---: | ---: |
| TP-GMM additive | 1.000 | 1.000 | 1.000 |
| TP-GMM SE(2) | 1.000 | 1.000 | 1.000 |

Projected active-phase max alpha:

| model | N | du | dv | yaw |
| --- | ---: | ---: | ---: | ---: |
| TP-GMM additive | 8 | 0.996 | 0.004 | 0.003 |
| TP-GMM SE(2) | 8 | 0.963 | 0.005 | 0.006 |
| TP-GMM additive | 15 | 0.997 | 0.004 | 0.007 |
| TP-GMM SE(2) | 15 | 0.925 | 0.004 | 0.005 |
| TP-GMM additive | 30 | 0.998 | 0.004 | 0.002 |
| TP-GMM SE(2) | 30 | 0.969 | 0.005 | 0.010 |

Projected `E_alpha`: TP-GMM additive 2.46e-02 / 4.13e-03 / 3.87e-03;
TP-GMM SE(2) 2.60e-02 / 3.92e-03 / 3.84e-03 for N=8/15/30.

## Caveat

This is a controlled state-level LIBERO/robosuite rollout, not a learned or
scripted robot policy demonstration.  That is deliberate: the evidence is about
generator relevance under causal context interventions, while avoiding the cost
and instability of many full robot executions.
