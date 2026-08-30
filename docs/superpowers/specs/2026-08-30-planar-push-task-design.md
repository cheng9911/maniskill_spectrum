# Planar-Push Task — Design Spec (Generator-Selective Relational Transfer, Task 2)

**Status:** design (pre-implementation)
**Date:** 2026-08-30
**Supersedes:** single-geometry peg-in-hole only (Experiments 1 & 2 are frozen)

---

## 1. Purpose

The paper's current evidence for generator selectivity rests on a single
physical geometry: a rigid peg entering a socket, where the hole's shape
releases a DOF (rotational symmetry → `yaw` drops; translational symmetry →
`du` drops). A reviewer can fairly object: *"you built a model tailored to
peg-in-hole insertion / circular symmetry."*

This task answers that objection with a **genuinely different physical task**:
non-prehensive **planar pushing** of a block across a table to a target. The
selectivity here is the complement of peg-in-hole:

| experiment | physical mechanism | selective / tracked | suppressed / free |
|---|---|---|---|
| peg-in-hole (frozen) | cylindrical hole + rotational symmetry | 5 generators | 1 rotation (`yaw`) |
| **planar push (new)** | planar surface constraint | 3 generators (`du,dv,yaw`) | 3 generators (`dw,roll,pitch`) |

The same frozen SE(3) Pdiag model must recover **both** structures. The claim
is *generator relevance*, not *"find the released rotation"* — the model must
drop a **translation** (`dw`) and **two rotations** (`roll,pitch`) in the new
task, which a "circular-symmetry detector" would not.

## 2. The intervention paradigm (how the new task maps onto the framework)

The frozen framework defines generator relevance as the **response of the
manipulated object's trajectory to a rigid SE(3) perturbation of the task
frame**. Concretely (from `benchmark_se3_transfer.py`):

- `context` (a.k.a. `causal_delta`) = 6-vector `[du,dv,dw,roll,pitch,yaw]`.
- The task fixture is placed at `C0 ⊗ intervention(context)`.
- The solver produces a trajectory of the **manipulated object**; the model
  fits `X(s) = C0 Exp(diag(α(s)) Log(C0⁻¹ C)) C0⁻¹ X0(s)` in the recovered
  nominal frame `C0`.
- `α_j(s)` = whether the trajectory's `j`-th coordinate responds to the
  context's `j`-th DOF.

The mapping for the two tasks:

| concept | peg-in-hole (frozen) | planar push (new) |
|---|---|---|
| fixture (context target) | socket pose | ghost **target** pose |
| manipulated object | peg | **block** (pushed, never grasped) |
| trajectory curve | `peg_pose` | `block_pose` |
| nominal frame recovered from | `socket_pose` | `target_pose` |
| intervention range | ±0.012 m / ±15° / ±30° | same (ghost is unconstrained) |

The block starts at a fixed pose and ends at the target's in-plane pose; the
start is fixed, the end responds to `C` (identical start-fixed / end-perturbed
structure to peg-in-hole).

## 3. Oracle (model-independent, from the solver geometry)

Two arms, one env, distinguished by a `goal_heading` flag:

| arm | goal | oracle α (per generator; task basis `[du,dv,dw,roll,pitch,yaw]`) |
|---|---|---|
| `heading_push` (main) | block aligns in-plane position **and** heading | `[1,1,0,0,0,1]` |
| `free_yaw_push` (control) | block aligns in-plane position only, heading free | `[1,1,0,0,0,0]` |

- `du,dv,yaw` tracked (the block must respond to the target's in-plane pose).
- `dw,roll,pitch` suppressed — the block stays on the table (never lifts,
  never tilts), so perturbing the target's out-of-plane pose never changes the
  block's trajectory.

The **selector** — *which* generators are tracked vs suppressed — is static
(the same `{du,dv,yaw}` / `{dw,roll,pitch}` split throughout), a second
contrast against peg-in-hole's **phase-dependent** drop (`yaw: 1→0`). The
**phasewise** `α_j(s)` reflects *when* each tracked generator responds (the
block slides toward the target during `push`, rotates to heading during
`align`), so scoring is restricted to the block-moving ACTIVE phases
(`push` + `align`), masking `reach`/`retract` (setup/teardown, block at rest)
— the same discipline as the frozen benchmark, which scores only the four
insertion phases and excludes reach/grasp/lift. Without the mask the static
prefix dilutes `e_alpha` toward 0 for tracked generators.

Cross-task control (no new collection): the frozen peg-in-hole arm already has
`dw,roll,pitch = 1` throughout — the same model recovers them as *relevant*
when there is no table, and *suppressed* when there is. Three selectivity
structures, one model:

- peg-in-hole: 5 tracked + 1 free (`yaw`, phase-dependent)
- heading_push: 3 tracked + 3 free (`dw,roll,pitch`, static)
- free_yaw_push: 2 tracked + 4 free

## 4. Components

### 4.1 Environment (`phase_switch_symmetry_env.py`, additive)

New `@register_env("PlanarPush-v1")` class (new class; frozen SE(2)/SE(3)
classes byte-identical). Contents:

- Static **table** (large box, top surface = the constraint plane).
- A dynamic **block**: a rectangular cuboid with a clearly **non-square
  footprint** (e.g. 0.04 × 0.02 m), spawned at a fixed start pose on the
  table, so heading is a well-defined DOF. The rectangle's 2-fold (180°)
  symmetry is never reached because the yaw intervention range is ±30° (well
  inside ±90°), so heading is unique without any marker.
- A **ghost target** indicator (semi-transparent cuboid) placed at the
  context pose `C0 ⊗ intervention(context)`; the ghost renders the full 6-DOF
  target so `dw/roll/pitch` genuinely vary in the data.
- `causal_delta` = 6-vector; `goal_heading` flag (bool).
- Emits `block_pose`, `target_pose`, `causal_delta`, `solver_phase` (via the
  existing `PhaseSwitchTraceWrapper` observation path).

The ghost is the standard MetaWorld/Robomimic goal-visualization technique and
is what makes the out-of-plane context components *real* (varying) rather than
absent — necessary for the model to learn `α=0` vs `α=1`.

### 4.2 Solver (`collect_phase_switch_rotated.py`, additive `solve_planar_push`)

New function, same `PandaArmMotionPlanningSolver` skeleton as `solve_se3`,
non-prehensive push:

1. `reach` — move closed gripper tip to behind the block (along block→target
   direction), at block height.
2. `push` — incremental 3 mm steps pushing the block toward the target's
   in-plane position.
3. `align` — (heading arm only) rotate the block about vertical to match the
   target heading by nudging one corner.
4. `retract` — lift gripper, move clear.

Active (scored) phases = `push` + `align`; `reach` + `retract` are
setup/teardown (block at rest) and are masked from scoring (§3).

Phases get their own `set_phase` codes (new PHASE_CODE set for the push task);
`move_to_pose_with_screw` with RRTConnect fallback, identical to `solve_se3`.

### 4.3 Collector (`collect_planar_push_rollouts.py`, new)

Mirror of `collect_se3_rollouts.py`: 2 arms × 3 seeds × 75 conditions
(60 mixed + 14 isolated + 1 baseline), R_max=5, `PhaseSwitchTraceWrapper` +
`write_episode`, frozen manifest sha256 checked against the preregistered
experiment JSON. Output `phase_switch_symmetry_rollouts_planar_push/
{heading_push,free_yaw_push}_seed_{seed}.h5`.

### 4.4 Manifest + subsets (`generate_planar_push_contexts.py`,
`prepare_planar_push_subsets.py`, new)

Reuse the frozen SE(3) manifest/subset discipline at `dim=6`:
`CONTEXT_SCALE=[0.012,0.012,0.012, deg2rad(15),deg2rad(15),deg2rad(30)]`,
`rank==6`, `condition_number<10`, `sample_sizes=[8,15,30]`, 3 × {random,
qualified}. In-plane ranges bounded by the table half-extent so the target is
always pushable.

### 4.5 Benchmark (`benchmark_planar_push.py`, new)

Mirror of `benchmark_se3_transfer.py`, reusing the frozen 4-model SE(3) suite
(`SE3FrameWeightedModel`, `SE3FullOperatorModel`, `SE3DiagonalOperatorModel`,
`SE3SmoothFinitePDiagModel`) unchanged. Differences:

- `TASK_ORDER = ["heading_push", "free_yaw_push"]`.
- `phase_task_curve` uses `block_pose` (not `peg_pose`).
- `nominal_frame` recovers the **target** frame from `target_pose` (same
  `mean_i(target_i ⊗ intervention_i⁻¹)` arithmetic/circular mean).
- `oracle_alpha_planar_push(arm, phase_codes)` per §3.

### 4.6 Metrics

- **M1 `e_alpha`** — recovered diagonal vs oracle, both arms (lower better).
- **M-oop (out-of-plane suppression)** — headline. For the `heading_push` arm,
  correct iff `α_dw, α_roll, α_pitch < 0.5` **and** `α_du, α_dv, α_yaw > 0.5`
  (active-phase means, §3). Report accuracy across seeds/subsets.
- **M-yaw (heading-vs-free-yaw discrimination)** — correct iff
  `α_yaw(heading) > 0.5` and `α_yaw(free_yaw) < 0.5` per subset (the
  between-arm yaw contrast, analogous to frozen M2).
- **Cross-task out-of-plane contrast (fixed)** — `α_{dw,roll,pitch}` in
  `heading_push` (~0) vs the frozen peg-in-hole arm (~1), read directly from
  the frozen `se3_transfer_fits.csv` + the new fits. No new collection.

## 5. Feasibility gate (Phase 0, mandatory)

Non-prehensive pushing has real dynamics risk (block slip / off-axis rotation).
Before full collection, a probe script (`probe_planar_push.py`) verifies the
scripted pusher reaches the target across context extremes — far targets,
±30° heading, ghost tilted at ±15° — with the R_max=5 retry discipline,
recording first-attempt vs eventual success and max contact force.

- **Pass** → proceed to full collection.
- **Fail** (pusher cannot robustly reach) → fall back to **guided push**
  (grasp + slide-on-table), recorded in the frozen doc; the oracle is
  unchanged, the physical authenticity is weakened and must be stated.

## 6. Freezing discipline

Purely additive. Frozen SE(2)/SE(3) env classes, solvers, benchmarks, datasets,
and `VALIDATION_*` docs stay byte-identical; all new work lives in new files
importing from them. New frozen artifacts: `planar_push_experiment.json`
(preregistration with manifest/source sha256, before collection),
`planar_push_fixed_contexts.json`, `planar_push_subsets.json`, and a
`VALIDATION_planar_push.md` freezing design + results with source hashes.

## 7. Risks & open points

- **Push feasibility** (highest risk) — gated by Phase 0; fallback = guided
  push (§5).
- **Block-as-cuboid vs sphere** — cuboid chosen so `yaw` (heading) is a
  well-defined goal DOF; a sphere would make heading undefined and forfeit the
  yaw discrimination.
- **Active-phase mask** — `reach`/`retract` are setup/teardown where the block
  is at rest; scoring masks them and uses only `push`+`align` (§3), so the
  static prefix cannot dilute `e_alpha`.
- **Phase-code set** — the push solver needs its own `set_phase` codes and a
  matching `progress_grid` mapping; decided in the implementation plan.

## 8. Files

**New:**
- `phase_switch_symmetry_env.py` — `PlanarPush-v1` class (additive).
- `collect_phase_switch_rotated.py` — `solve_planar_push` (additive).
- `collect_planar_push_rollouts.py`, `probe_planar_push.py`,
  `generate_planar_push_contexts.py`, `prepare_planar_push_subsets.py`,
  `benchmark_planar_push.py`.
- Frozen artifacts under `phase_switch_symmetry_multiseed/planar_push/` and
  rollouts under `phase_switch_symmetry_rollouts_planar_push/`.

**Reuse (import, do not modify):** `phase_switch_se3_baselines.py` (model
suite), `benchmark_phase_switch_baselines.py` (`progress_grid`,
`switch_diagnostics`, `usable`), `analyze_phase_switch_rollouts.py`
(`resample`, `wrap_pi`), the SE(3) env/solver/collector/benchmark skeletons.
