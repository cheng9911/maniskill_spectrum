# Geometry Transfer Validation

Date: 2026-08-31

## Question

This supplement tests whether generator relevance learned in one object/scene can
transfer to another object/scene without re-identifying alpha on the target.

The transferred method uses:

- source: Pdiag finite alpha profile learned on the source task with N=30;
- target adaptation: exactly one target nominal trajectory, condition 0;
- target interventions: used only as validation probes, not for fitting alpha.

This supports the claim that the learned relevance law is not simply bound to a
specific object identity, target location, or LIBERO scene frame.  It does not
claim pixel-level lighting invariance because the estimator operates on pose
trajectories rather than RGB.

## Transfer Pairs

| pair | source | target | relation tested |
| --- | --- | --- | --- |
| `drawer_to_plate_push` | drawer middle open | plate front push | one-axis sliding |
| `knob_to_microwave_door` | stove knob turn | microwave door revolute | yaw-only revolute relation |
| `bowl_stove_to_bowl_plate` | bowl on stove | bowl on plate | support to stacking |
| `bowl_stove_to_cream_cheese_bowl` | bowl on stove | cream cheese in bowl | support to container |
| `bowl_stove_to_wine_cabinet` | bowl on stove | wine bottle on cabinet | cross-object support |

These five pairs cover changes in object identity, approximate object size,
support/container geometry, and scene/world location.  They should be reported as
controlled pose-level geometry/relation transfer, not as visual policy transfer.

## Main Result

`M_transfer` thresholds the active-phase alpha profile at 0.5 and compares it to
the target task oracle selector.

| method | target fit interventions | N=3 | N=5 | N=8 |
| --- | ---: | ---: | ---: | ---: |
| Ours transfer: source Pdiag N=30 + target nominal N=1 | 0 | 1.000 | 1.000 | 1.000 |
| Frame-weighted target scratch | N | 0.000 | 0.000 | 0.000 |
| Phase scalar GP target scratch | N | 0.000 | 0.000 | 0.000 |
| Pdiag finite target scratch | N | 1.000 | 1.000 | 1.000 |

Mean `E_alpha`:

| method | N=3 | N=5 | N=8 |
| --- | ---: | ---: | ---: |
| Ours transfer | 2.13e-04 | 2.13e-04 | 2.13e-04 |
| Frame-weighted target scratch | 1.38e-01 | 1.30e-01 | 1.27e-01 |
| Phase scalar GP target scratch | 1.38e-01 | 1.30e-01 | 1.27e-01 |
| Pdiag finite target scratch | 3.76e-03 | 9.19e-04 | 5.87e-04 |

Mean heldout trajectory MSE, using metric-scaled SE(3) residuals:

| method | N=3 | N=5 | N=8 |
| --- | ---: | ---: | ---: |
| Ours transfer | 9.28e-09 | 9.17e-09 | 9.32e-09 |
| Frame-weighted target scratch | 8.10e-06 | 6.64e-06 | 6.01e-06 |
| Phase scalar GP target scratch | 8.10e-06 | 6.64e-06 | 6.01e-06 |
| Pdiag finite target scratch | 3.48e-07 | 2.29e-07 | 2.15e-07 |

## TP-GMM Check

A separate lightweight N=8 pass includes TP-GMM SE(3):

| method | M_transfer | E_alpha | heldout trajectory MSE |
| --- | ---: | ---: | ---: |
| Ours transfer | 1.000 | 2.13e-04 | 9.28e-09 |
| Pdiag finite target scratch | 1.000 | 7.42e-04 | 2.16e-07 |
| TP-GMM SE(3) target scratch | 1.000 | 2.62e-02 | 3.44e-05 |
| Frame-weighted target scratch | 0.000 | 1.28e-01 | 6.26e-06 |
| Phase scalar GP target scratch | 0.000 | 1.28e-01 | 6.26e-06 |

TP-GMM recovers the binary relation at N=8, but its continuous alpha profile and
heldout trajectory prediction are substantially worse than direct transfer and
Pdiag finite on this small-sample target adaptation setting.

## Files

- `geometry_transfer_fits.csv`
- `geometry_transfer_summary.csv`
- `geometry_transfer_by_pair.csv`
- `geometry_transfer_validation.json`
- TP-GMM N=8 pass: `../geometry_transfer_tpgmm_n8/`

## Caveats

1. The transfer pairs reuse controlled state-level LIBERO/robosuite rollouts.
   They are not learned visual policies.
2. The target nominal trajectory instantiates geometry and offset only; it does
   not estimate alpha from target interventions.
3. Lighting invariance should be stated only as an input-independence argument:
   pose-based alpha estimation does not consume pixels.
4. For a direct same-task size perturbation claim, add a dedicated resized-object
   planar-push or placement variant.  The present evidence already covers object
   identity/size changes across LIBERO task instances, but not a parametric
   same-task mesh-scale sweep.
