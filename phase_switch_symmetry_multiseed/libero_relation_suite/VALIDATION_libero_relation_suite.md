# LIBERO/robosuite Controlled Relation Suite Validation

Date: 2026-08-31

## Scope

This supplement regenerates controlled counterfactual trajectories in real
LIBERO/robosuite MuJoCo scenes.  It does not reuse offline LIBERO demos.  Each
task carries the same six-generator context basis:

`[du, dv, dw, roll, pitch, yaw]`

The raw MuJoCo body pose is saved as `libero_body_pose_world`; the scored
`object_pose` is stored in the declared task-local frame.

## Task Set

| task | family | oracle selector |
| --- | --- | --- |
| `drawer_middle_open` | prismatic sliding | `[1,0,0,0,0,0]` |
| `stove_knob_turn` | revolute knob | `[0,0,0,0,0,1]` |
| `microwave_door_revolute` | revolute door | `[0,0,0,0,0,1]` |
| `plate_front_push` | planar one-axis push | `[1,0,0,0,0,0]` |
| `bowl_on_stove` | support placement | `[1,1,1,0,0,0]` |
| `bowl_on_plate` | stacking support | `[1,1,1,0,0,0]` |
| `cream_cheese_in_bowl` | container-in | `[1,1,1,0,0,0]` |
| `wine_bottle_on_rack` | rack / slot placement | `[1,1,1,0,0,1]` |
| `wine_bottle_on_cabinet` | upright support placement | `[1,1,1,0,0,0]` |
| `moka_pot_on_stove` | compound support subtask | `[1,1,1,0,0,0]` |

This is best described as 10 task instances covering several relation families,
not as 10 fully independent physical mechanisms.

## Data Integrity

Generated rollout root:

`phase_switch_symmetry_rollouts_libero_relation_suite/`

Validation:

- 10 task keys;
- 30 HDF5 files = 10 tasks x 3 seeds;
- every file contains 75/75 usable episodes;
- every file has solver phases `[3,4,5,6]`;
- the context manifest hash matches `libero_relation_suite_experiment.json`.

## Main Result

`M_relation` is correct iff, over active phases (`move` + `hold`), every
oracle-selected generator has max-active alpha > 0.5 and every suppressed
generator has max-active alpha < 0.5.

Breadth-sweep Pdiag config: `n_basis=8`, `basis_width=0.12`,
`nominal_iterations=1`.

| model | N=8 | N=15 | N=30 |
| --- | ---: | ---: | ---: |
| Frame-weighted (SE(3)) | 0.000 | 0.000 | 0.000 |
| Phase scalar GP (SE(3)) | 0.000 | 0.000 | 0.000 |
| TP-GMM additive (SE(3)) | 1.000 | 1.000 | 1.000 |
| TP-GMM SE(3) | 1.000 | 1.000 | 1.000 |
| Full operator (SE(3)) | 1.000 | 1.000 | 1.000 |
| Pdiag pointwise (SE(3)) | 1.000 | 1.000 | 1.000 |
| Pdiag finite (SE(3)) | 1.000 | 1.000 | 1.000 |

Pdiag finite is 1.000 for every task individually at N=8, N=15, and N=30.
Frame-weighted is 0.000 for every task individually at all N.

## Pdiag Finite N=30 Alpha Pattern

| task | du | dv | dw | roll | pitch | yaw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| drawer_middle_open | 1.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| stove_knob_turn | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.009 |
| microwave_door_revolute | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.009 |
| plate_front_push | 1.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| bowl_on_stove | 1.010 | 1.010 | 0.969 | 0.023 | 0.023 | 0.027 |
| bowl_on_plate | 1.010 | 1.010 | 0.969 | 0.022 | 0.023 | 0.027 |
| cream_cheese_in_bowl | 1.010 | 1.010 | 0.969 | 0.022 | 0.022 | 0.027 |
| wine_bottle_on_rack | 1.034 | 1.020 | 0.969 | 0.025 | 0.020 | 1.020 |
| wine_bottle_on_cabinet | 1.010 | 1.010 | 0.969 | 0.023 | 0.023 | 0.027 |
| moka_pot_on_stove | 1.010 | 1.010 | 0.969 | 0.022 | 0.023 | 0.027 |

Overall `E_alpha`:

| model | N=8 | N=15 | N=30 |
| --- | ---: | ---: | ---: |
| Phase scalar GP (SE(3)) | 1.23e-01 | 1.24e-01 | 1.23e-01 |
| TP-GMM additive (SE(3)) | 2.09e-02 | 1.55e-02 | 3.60e-03 |
| TP-GMM SE(3) | 1.82e-02 | 1.63e-02 | 6.14e-03 |
| Pdiag finite (SE(3)) | 5.77e-04 | 5.08e-04 | 4.36e-04 |

`Phase scalar GP (SE(3))` is the SE(3) lift of the repository's oracle-favorable
shared-scalar GP negative control.  It is not the TPGP algorithm.

## Historical TP-GMM Projected Baseline

The repository TP-GMM baseline is SE(2), so it is evaluated only on the
projection `[du, dv, yaw]`.  It cannot score full SE(3) suppression for `dw`,
`roll`, or `pitch`.  It is retained only for continuity with the original SE(2)
benchmark; the full SE(3) TP-GMM rows above are the main TP-GMM comparison.

Projected `M_relation`:

| model | N=8 | N=15 | N=30 |
| --- | ---: | ---: | ---: |
| TP-GMM additive | 1.000 | 1.000 | 1.000 |
| TP-GMM SE(2) | 1.000 | 1.000 | 1.000 |

Projected mean `E_alpha`:

| model | N=8 | N=15 | N=30 |
| --- | ---: | ---: | ---: |
| TP-GMM additive | 2.91e-02 | 6.60e-03 | 3.49e-03 |
| TP-GMM SE(2) | 2.95e-02 | 7.85e-03 | 6.01e-03 |

## Caveats

1. These are controlled state-level rollouts in LIBERO scenes, not full robot
   policies.  This is appropriate for alpha causality but should be stated.
2. The 10 task instances include related support-placement variants; report them
   as breadth across task instances/families, not as 10 independent mechanisms.
3. Projected SE(2) TP-GMM results should not be mixed with the full
   six-generator SE(3) alpha table.
