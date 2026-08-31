# 2026-08-31 Work Summary

## Overall Status

Today we completed a substantial supplement around controlled LIBERO/robosuite
relation experiments and one-shot geometry/relation transfer.

The current evidence supports the following claim:

> Generator-level relevance laws learned in one object/scene can transfer to
> matched relation families in new objects/scenes from a single target nominal
> trajectory; target interventions are used only for validation, not for
> re-identifying alpha.

This should be reported as pose-level relation/geometry transfer, not as visual
policy transfer.

## Completed Work

### 1. LIBERO/robosuite 10-Task Controlled Relation Suite

Implemented and benchmarked a controlled 10-task LIBERO relation suite:

| task | relation family | oracle selector `[du,dv,dw,roll,pitch,yaw]` |
| --- | --- | --- |
| `drawer_middle_open` | prismatic sliding | `[1,0,0,0,0,0]` |
| `stove_knob_turn` | revolute knob | `[0,0,0,0,0,1]` |
| `microwave_door_revolute` | revolute door | `[0,0,0,0,0,1]` |
| `plate_front_push` | planar one-axis push | `[1,0,0,0,0,0]` |
| `bowl_on_stove` | support placement | `[1,1,1,0,0,0]` |
| `bowl_on_plate` | stacking support | `[1,1,1,0,0,0]` |
| `cream_cheese_in_bowl` | container-in | `[1,1,1,0,0,0]` |
| `wine_bottle_on_rack` | rack/slot placement | `[1,1,1,0,0,1]` |
| `wine_bottle_on_cabinet` | upright support placement | `[1,1,1,0,0,0]` |
| `moka_pot_on_stove` | compound support subtask | `[1,1,1,0,0,0]` |

Data integrity audit:

- 10 task instances;
- 30 rollout files = 10 tasks x 3 seeds;
- 2250/2250 usable controlled episodes;
- each file contains 75 conditions;
- phase codes are `[3,4,5,6]`;
- context and subset manifest hashes match.

Main `M_relation` result:

| model | N=8 | N=15 | N=30 |
| --- | ---: | ---: | ---: |
| Frame-weighted (SE(3)) | 0.000 | 0.000 | 0.000 |
| Phase scalar GP (SE(3)) | 0.000 | 0.000 | 0.000 |
| TP-GMM additive (SE(3)) | 1.000 | 1.000 | 1.000 |
| TP-GMM SE(3) | 1.000 | 1.000 | 1.000 |
| Full operator (SE(3)) | 1.000 | 1.000 | 1.000 |
| Pdiag pointwise (SE(3)) | 1.000 | 1.000 | 1.000 |
| Pdiag finite (SE(3)) | 1.000 | 1.000 | 1.000 |

Continuous `E_alpha` at N=30:

| model | E_alpha |
| --- | ---: |
| Pdiag finite (SE(3)) | 4.36e-04 |
| TP-GMM additive (SE(3)) | 3.60e-03 |
| TP-GMM SE(3) | 6.14e-03 |
| Phase scalar GP (SE(3)) | 1.23e-01 |

### 2. SE(3) TP-GMM and Phase-Scalar GP Baselines

Added full SE(3) versions of:

- TP-GMM additive;
- TP-GMM SE(3);
- Phase scalar GP negative control.

The old projected SE(2) TP-GMM is retained for continuity, but the full SE(3)
TP-GMM rows are now the main TP-GMM comparison for six-generator alpha.

Important caveat: these additions currently modify the existing SE(3) benchmark
source files.  For strict freeze discipline, either report this as a post-freeze
baseline extension with updated hashes, or move the new baselines into a separate
extended-baseline file.

### 3. Drawer Standalone Probe

Completed a separate LIBERO drawer probe:

- 3 rollout files;
- 225/225 usable episodes;
- Pdiag finite `M_prismatic = 1.000` for N=8/15/30;
- projected TP-GMM also reaches `1.000`.

This is useful as a focused prismatic sanity check, but should not be counted as
an extra independent LIBERO task beyond the 10-task suite because the same
relation also appears as `drawer_middle_open`.

### 4. Geometry/Relation Transfer Benchmark

Implemented a one-shot transfer benchmark:

- source law: Pdiag finite alpha profile learned from source task at N=30;
- target adaptation: exactly 1 target nominal trajectory, condition 0;
- target interventions: used only for validation probes, not for alpha fitting;
- target probes: N=3/5/8.

Transfer pairs:

| pair | source | target | relation tested |
| --- | --- | --- | --- |
| `drawer_to_plate_push` | drawer middle open | plate front push | one-axis sliding |
| `knob_to_microwave_door` | stove knob turn | microwave door revolute | yaw-only revolute |
| `bowl_stove_to_bowl_plate` | bowl on stove | bowl on plate | support to stacking |
| `bowl_stove_to_cream_cheese_bowl` | bowl on stove | cream cheese in bowl | support to container |
| `bowl_stove_to_wine_cabinet` | bowl on stove | wine bottle on cabinet | cross-object support |

Main result:

| method | N=3 | N=5 | N=8 |
| --- | ---: | ---: | ---: |
| Ours transfer: source Pdiag N=30 + target nominal N=1 | 1.000 | 1.000 | 1.000 |
| Frame-weighted target scratch | 0.000 | 0.000 | 0.000 |
| Phase scalar GP target scratch | 0.000 | 0.000 | 0.000 |
| Pdiag finite target scratch | 1.000 | 1.000 | 1.000 |

Mean `E_alpha`:

| method | N=3 | N=5 | N=8 |
| --- | ---: | ---: | ---: |
| Ours transfer | 2.13e-04 | 2.13e-04 | 2.13e-04 |
| Pdiag finite target scratch | 3.76e-03 | 9.19e-04 | 5.87e-04 |
| Frame-weighted target scratch | 1.38e-01 | 1.30e-01 | 1.27e-01 |
| Phase scalar GP target scratch | 1.38e-01 | 1.30e-01 | 1.27e-01 |

TP-GMM N=8 separate pass:

| method | M_transfer | E_alpha | heldout trajectory MSE |
| --- | ---: | ---: | ---: |
| Ours transfer | 1.000 | 2.13e-04 | 9.28e-09 |
| Pdiag finite target scratch | 1.000 | 7.42e-04 | 2.16e-07 |
| TP-GMM SE(3) target scratch | 1.000 | 2.62e-02 | 3.44e-05 |
| Frame-weighted target scratch | 0.000 | 1.28e-01 | 6.26e-06 |
| Phase scalar GP target scratch | 0.000 | 1.28e-01 | 6.26e-06 |

Interpretation: TP-GMM can recover the binary selector at N=8, but its
continuous alpha profile and heldout trajectory prediction are substantially
worse than direct transfer.

### 5. Two-Scene Few-Shot Generalization Check

Extracted two structurally different target scenes for a compact main-text style
table:

| pair | relation |
| --- | --- |
| `knob_to_microwave_door` | revolute yaw-only |
| `bowl_stove_to_cream_cheese_bowl` | support/container placement |

Verification passed with `TWO_SCENE_FEWSHOT_OK`.

Representative results:

| pair | method | N | M_transfer | E_alpha |
| --- | --- | ---: | ---: | ---: |
| `knob_to_microwave_door` | Ours transfer | 3/5/8 | 1.000 | 1.4e-05 |
| `knob_to_microwave_door` | TP-GMM SE(3) target scratch | 8 | 1.000 | 3.29e-02 |
| `bowl_stove_to_cream_cheese_bowl` | Ours transfer | 3/5/8 | 1.000 | 3.45e-04 |
| `bowl_stove_to_cream_cheese_bowl` | TP-GMM SE(3) target scratch | 8 | 1.000 | 1.93e-02 |

This is the cleanest candidate for the paper main text: it is short, directly
addresses transfer, and avoids overwhelming readers with all 10 tasks.

## Recommended Paper Placement

Main text:

- include the two-scene few-shot transfer table;
- state that target adaptation uses one nominal trajectory and zero target
  interventions for alpha fitting;
- mention TP-GMM reaches binary correctness at N=8 but has much larger
  continuous alpha error;
- cite the full 5-pair transfer and 10-task LIBERO suite in Supplementary.

Supplement:

- full 10-task LIBERO breadth table;
- full 5-pair geometry transfer table;
- per-pair/per-task breakdowns;
- projected historical TP-GMM table;
- validation hashes and data-integrity checks.

## Key Claim Wording

Suggested main claim:

> The learned generator-relevance structure transfers across object and scene
> instances within matched relation families from a single target nominal
> trajectory; target interventions are reserved for validation rather than
> re-identifying alpha.

Avoid overclaiming:

- do not claim universal transfer across arbitrary unrelated tasks;
- do not claim visual/light invariance unless an RGB-based policy/encoder is
  tested;
- describe these as controlled pose-level LIBERO/robosuite experiments.

## Files Added or Updated

New/updated scripts:

- `phase_switch_symmetry/collect_libero_drawer_probe.py`
- `phase_switch_symmetry/prepare_libero_drawer_subsets.py`
- `phase_switch_symmetry/benchmark_libero_drawer_probe.py`
- `phase_switch_symmetry/libero_relation_suite_specs.py`
- `phase_switch_symmetry/collect_libero_relation_suite.py`
- `phase_switch_symmetry/prepare_libero_relation_suite_subsets.py`
- `phase_switch_symmetry/benchmark_libero_relation_suite.py`
- `phase_switch_symmetry/audit_libero_relation_suite.py`
- `phase_switch_symmetry/benchmark_geometry_transfer.py`
- `phase_switch_symmetry/make_geometry_transfer_two_scene_summary.py`
- `phase_switch_symmetry/phase_switch_se3_baselines.py`
- `phase_switch_symmetry/benchmark_se3_transfer.py`

New outputs:

- `phase_switch_symmetry_multiseed/libero_drawer/`
- `phase_switch_symmetry_multiseed/libero_relation_suite/`
- `phase_switch_symmetry_multiseed/geometry_transfer/`
- `phase_switch_symmetry_multiseed/geometry_transfer_tpgmm_n8/`
- `phase_switch_symmetry_rollouts_libero_drawer/`
- `phase_switch_symmetry_rollouts_libero_relation_suite/`

New validation docs:

- `phase_switch_symmetry_multiseed/libero_drawer/VALIDATION_libero_drawer.md`
- `phase_switch_symmetry_multiseed/libero_relation_suite/VALIDATION_libero_relation_suite.md`
- `phase_switch_symmetry_multiseed/geometry_transfer/VALIDATION_geometry_transfer.md`
- `phase_switch_symmetry_multiseed/geometry_transfer/VALIDATION_geometry_transfer_two_scene.md`

## Verification Commands Run

Successful checks included:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n maniskill_download \
  python phase_switch_symmetry/audit_libero_relation_suite.py
```

Output included:

```text
AUDIT_OK
relation_tasks: 10
relation_files: 30
relation_episodes: 2250
relation_usable: 2250
drawer_files: 3
drawer_episodes: 225
drawer_usable: 225
```

Geometry transfer checks:

```text
GEOMETRY_TRANSFER_OK
GEOMETRY_TRANSFER_TPGMM_OK
TWO_SCENE_FEWSHOT_OK
```

Syntax checks:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n maniskill_download \
  python -m py_compile phase_switch_symmetry/benchmark_geometry_transfer.py

PYTHONDONTWRITEBYTECODE=1 conda run -n maniskill_download \
  python -m py_compile phase_switch_symmetry/make_geometry_transfer_two_scene_summary.py
```

## Remaining Caveats and Next Actions

1. Freeze provenance: the SE(3) TP-GMM/GP baseline extension currently modifies
   files that were previously treated as frozen.  Before submission, either
   document this as a post-freeze baseline extension or refactor it into an
   explicit extended-baseline module.
2. The 10-task LIBERO suite should be described as 10 task instances covering
   relation families, not 10 fully independent physical mechanisms.
3. The geometry-transfer results are pose-level relation transfer results, not
   visual policy transfer.
4. No further task expansion is urgently needed.  The highest-value next step is
   to turn the two-scene transfer result into a concise main-text figure/table
   and move the full tables to Supplementary.
5. Current worktree contains uncommitted new files and outputs.
