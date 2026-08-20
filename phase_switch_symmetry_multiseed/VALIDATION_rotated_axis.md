# Frozen Validation: Rotated-Axis Geometric Consistency

This document freezes the rotated-axis experiment — the full chain from solver
repair, anchor search, strict screening, retry pilot, to the confirmatory
collection and analysis. It tests whether the phase-dependent task-local
generator law is invariant under a global task reorientation (vertical Q1 vs
horizontal Q2).

## Frozen design

```text
orientations:  Q1 = identity        [1.0, 0.0, 0.0, 0.0]   (insertion axis = world z)
               Q2 = Ry(90 deg)      [0.707107, 0.0, 0.707107, 0.0]  (insertion axis = world x)
excluded Q3:   Rx(35)Ry(45) [0.88112, 0.277816, 0.364972, -0.115075]  (oblique, reachability-limited)
common anchor: (-0.35, 0.10, 0.08)   (Q1/Q2 common feasible interior cell)
seeds:         20260818, 20270818, 20280818
contexts:      39-condition frozen manifest (30 mixed + 8 isolated + 1 zero)
retry budget:  R_max = 5 (each condition cell up to 5 execution attempts)
noise:         robot_init_qpos_noise = 0.01 rad
```

Source hashes:

```text
fixed_contexts.json                3751db801cf49ce698c9b5a74ce36cf0b39897d967ddf2959846a3965b8bd653
rotated_axis_experiment.json       3a8869de50386e825de71331ff83d3233f7aebfa522540fab20b92c8fa751189
phase_switch_symmetry_env.py       8eb52f66ea7c242ec0d5248d760c45e526a4ac62ed86351875d124de0e5c9dc7
collect_phase_switch_rotated.py    0410aef21d2eefea866d014afca30d620ec5f783e0a55c63a12dabc396dcbb0e
analyze_phase_switch_rotated.py    6818d5a115de5f0ddcd561394dc8cdccc4ef1f750e217ca38fbd54e2328d73d0
analyze_rotated_exploratory.py     8bd8ce97f9b4c9d593fc480ef9d474b03aa2ca00202ad344732c54a612298143
validate_rotated_reachability.py   ff8f135a9a3920ebbf22d276cf63f2d2e74799e381c41bb9b5d37d91a675a243
search_rotated_anchor.py           d5ca0a75da46b8ebcb117e95b572b745b3f4c57aaafedb005361734dd4c68a6a
find_common_anchor.py              77ba1dabd35c42a1d43d5e25fe8d06eb206182c7e5471d57ab16c150577c02be
run_rotated_retry_pilot.py         ca287ec5ab7cb8614c9d007b8d14b0b46a6034924154788abdcd05609e26b1c9
visualize_rotated_results.py       19579b51e9cfb3689c8884721227f992cbe2db960546e2102f75c88bc207d51c
```

---

## 1. Solver / execution repair

The naive rotation of the original task was not reachable for the horizontal
orientation. Four execution-level issues were fixed and frozen:

1. **Pickup geometry decoupled from the socket.** The peg is grasped at a FIXED
   reachable spot on a kinematic pedestal `(-0.25, -0.18, 0.28)` with its axis
   aligned to the insertion direction Q, rather than at `Q*(PEG_START-SOCKET)`.
2. **End-on grasp** (`approaching = -insertion_axis`), which keeps the gripper
   behind the peg during insertion (a top-down grasp made the pre-insert TCP
   unreachable).
3. **Incremental Cartesian insertion** (3 mm steps along the insertion axis)
   through the keyed gate, avoiding the screw-planner narrow-passage failure.
4. **Firm gripper** (stiffness 2.5e3, force limit 150 N) to hold the peg through
   the ~100 N axial contact, eliminating the ~5 cm slip.
5. **`max_episode_steps = 1500`** to accommodate the incremental insertion.

The `robot_init_qpos_noise = 0.01` (matching the multiseed replication) is the
execution perturbation that separates the three seeds; with `0.0` the horizontal
rollouts were byte-identical across seeds (a seed-variation bug that was caught
and fixed during the pilot).

## 2. Anchor search (Q1/Q2 common workspace)

Coarse feasibility map (nominal full path) over the box
`x in [-0.45,-0.25], y in [-0.15,0.15], z in [0.08,0.28]` (0.05 m grid):

```text
Q1 feasible: 62/96,  Q2 feasible: 64/96,  common: 40/96
best (margin-maximizing interior) anchor: (-0.35, 0.10, 0.08), margin 0.05 m
```

The anchor was chosen from IK/collision feasibility only, never from Pdiag
results.

**Q3 (oblique) excluded.** `Rx(35)Ry(45)` gave 0/96 nominal feasibility (90
FAIL@align_keyed): the Panda could not realize the oblique pre-insertion
configuration over the screened workspace. This is recorded as a
reachability-limited exploratory orientation, not a failure of the learned
transformation law (it never entered the learning stage).

## 3. Strict screening (7 extreme contexts)

At the frozen anchor, `2Q x 7 extreme contexts x 3 seeds = 42` runs with retry
budget R=3:

```text
Q1: first-attempt 18/21, eventual 21/21
Q2: first-attempt 21/21, eventual 21/21
```

The 3 Q1 first-attempt failures were all at seed 20280818 circular_insert
(du-, dv-, dpsi-) and recovered within 2-3 retries; no systematic failure.

## 4. Retry pilot (stochastic missingness check)

The first confirmatory collection at R=3 had ~7% mixed-context failures. A
failure-context check found **no association between failure and |du|, |dv|,
|dpsi|, or their interactions** (all Spearman |rho| < 0.06, p > 0.4), with Q1
(0.922) and Q2 (0.933) success rates nearly equal.

A retry pilot retried the 14 first-round-failed cells with 5 additional
independent attempts (same orientation/anchor/solver/noise/seed):

```text
14/14 cells recovered within 1 additional attempt.
survival: r=0 -> 14 unresolved; r=1 -> 0 unresolved (R_all: 0.922 -> 1.000)
```

The missingness is stochastic execution variability, not context-dependent. This
motivated freezing `R_max = 5`.

## 5. Confirmatory collection and analysis

Confirmatory collection (`2Q x 3 seeds x 39`, R_max=5) yielded **232/234 usable
condition cells (99.1%)**; the two residual failures (condition 22 in Q1 seed
20270818, condition 36 in Q1 seed 20280818) are isolated single-cell failures,
each usable in the other five files.

Canonical-law invariance analysis (canonicalization `G_Q^-1`, keeping the
per-episode causal intervention `c_n`; common-success set = 26 mixed + 6
isolated + 1 baseline):

**Layer 1 — structure law (5/6 pass):**

| orientation | seed | g_trans_final | g_axial_pre | g_axial_final | pass |
|---|---|---|---|---|---|
| Q1 | 20260818 | 1.029 | 0.999 | 0.016 | y |
| Q1 | 20270818 | 0.863 | 1.004 | 0.025 | n |
| Q1 | 20280818 | 0.931 | 1.002 | 0.029 | y |
| Q2 | 20260818 | 0.983 | 0.999 | 0.031 | y |
| Q2 | 20270818 | 0.973 | 0.999 | 0.020 | y |
| Q2 | 20280818 | 0.973 | 0.998 | 0.026 | y |

**Layer 2 — matched-seed 90-degree consistency:**

| seed | D_orient | rho_axial | |d_g_trans_final| | |d_g_axial_final| |
|---|---|---|---|---|---|
| 20260818 | 0.112 | 0.985 | 0.046 | 0.015 |
| 20270818 | 0.289 | 0.941 | 0.111 | 0.005 |
| 20280818 | 0.148 | 0.938 | 0.042 | 0.004 |

**Layer 3 — orientation vs seed variation:**

```text
D_orient = 0.183,  D_seed = 0.123,  D_orient - D_seed = +0.060 (about 1.5x)
```

---

## Interpretation (frozen)

The confirmatory data support the core rotated-axis hypothesis:

> **A 90-degree global task rotation changes the world realization of the skill
> but preserves the task-local axial generator law.**

Evidence: `rho_axial = 0.94-0.98`; `g_axial_pre ~ 1` and `g_axial_final ~
0.02-0.03` for both Q1 and Q2. Because Q2's local axial rotation is no longer a
world-z yaw, this directly excludes the "model only learned a world-yaw
heuristic" alternative.

Cross-orientation profile variation is comparable in scale to execution-seed
variation, with a modest excess (`D_orient ~ 1.5x D_seed`). The excess is carried
by the translation channel (`|d_g_trans_final| = 0.04-0.11`) while the axial
channel is nearly identical (`|d_g_axial_final| = 0.004-0.015`), consistent with
a residual translation slip under the horizontal contact rather than a change in
the axial law.

Caveats: one structure-law failure (Q1/20270818 g_trans=0.863, slightly below
0.9); the outlier seed moved between the R=3 and R=5 runs (20280818 -> 20270818),
indicating stochastic execution-seed variation rather than a systematic effect;
the confirmatory analysis used the common-success set (26 mixed) because two
condition cells were never realized, though each was usable elsewhere.

This is a controlled simulation study with one task geometry, one common anchor,
and a fixed three-seed replication. The oblique Q3 remains a
reachability-limited exploratory configuration, not part of the frozen
invariance test.
