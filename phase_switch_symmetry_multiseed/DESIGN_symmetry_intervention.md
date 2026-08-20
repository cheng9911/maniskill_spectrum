# Design (preregistration): Generator relevance under manipulated task symmetry

This document freezes the design of the causal **symmetry-intervention** experiment
before any physics data is collected. It is the fourth experiment in the sequence
and targets a claim strictly above the first three:

| experiment | what it established |
|---|---|
| A / B / C (identifiability, TP-GMM few-shot, non-oracle progress) | Pdiag expresses phase-dependent generator selectivity, has a few-shot advantage over formal baselines, and does not require privileged phase labels |
| rotated-axis (vertical Q1 vs horizontal Q2) | the generator law is **task-local**, not a world-yaw heuristic |

The claim under test here is stronger than either:

> **The model recovers generator relevance from the task's symmetry group, not
> from trajectory statistics.**

The reviewer's alternative is precise and must be addressed head-on:

> "The `alpha(s)` you observe is just a property of the demonstration
> distribution. Keyed-plug demonstrations contain a yaw correction during
> alignment, so the model learns *when* to rotate, not *why* the rotation is
> needed."

The intervention below changes **the task relation while holding the motion
statistics as similar as possible**, then asks whether the inferred generator
profile `P(s)` follows the symmetry change.

---

## 1. The two tasks and the geometry difference

The existing environment is a **keyed peg** whose circular shaft carries a
rectangular key; a keyed gate above the socket forces axial-yaw alignment until
the key clears, after which the circular shaft is free to rotate. The new task
removes the key and the keyway, leaving a **circular peg in a circular bore**.
This is the "square (keyed) peg vs circular peg" contrast.

### Frozen geometry

```text
                    KEYED (existing)                          CIRCULAR (new)
peg tip             rectangular key 0.021 x 0.014             (none)
peg shaft           radius 0.013, z in [-0.029, +0.045]       radius 0.013 (identical)
gate opening        keyway 0.023 x 0.016 (KEY + 2 mm clear.)  circular collar r = 0.013 + 0.002 = 0.015
bore below gate     radius 0.030 (identical)                  radius 0.030 (identical)
socket outer        radius 0.070 (identical)                  radius 0.070 (identical)
insertion depth     FINAL_PEG_Z = 0.060 (identical)           FINAL_PEG_Z = 0.060 (identical)
approach            end-on along the insertion axis (same)    end-on along the insertion axis (same)
robot / controller  Panda, pd_joint_pos + mplib (same)        Panda, pd_joint_pos + mplib (same)
phases              reach/grasp/lift/align/enter/unlock/insert same four semantic phases (3..6)
```

### What is held invariant vs. what changes

**Invariant** (the reviewer's checklist): robot, controller, initial pose,
solver skeleton (same four semantic phases 3–6, hence same trajectory length),
insertion depth, approach direction, socket outer geometry, pedestal pickup,
execution noise (`robot_init_qpos_noise = 0.01`), retry budget (`R_max = 5`).

**Changed, and only this**: the SO(2)-breaking feature. The keyed task reduces
the axial-rotation symmetry `SO(2)` to the discrete subgroup `C2` (the key is a
rectangle, invariant under 180° rotation, so it re-fits the keyway at yaw +π);
the circular task retains full `SO(2)`. The **translation** generators remain
relevant in both tasks (the peg must still be positioned to enter the collar and
bore); only the **axial-yaw** generator's relevance is manipulated.

Unavoidable consequence, stated explicitly rather than hidden: in the keyed task
the tight passage is **key-vs-rectangular-keyway**; in the circular task it is
**shaft-vs-circular-collar** (radius 0.013 into 0.015, same 2 mm clearance).
This is the minimal change that removes SO(2)-breaking while retaining a
two-stage tight-then-loose insertion of matched depth. It is a contact-material
difference, not a symmetry-preserving equivalence; it is recorded as such.

**Framing.** This is a *minimal geometry intervention whose intended causal
variable is symmetry breaking*, not a claim that "only symmetry changed". The
contact-mode change is acknowledged above; the placebo arm (§3) is what removes
"the model just recognized a different object" as a viable alternative, because
it keeps the yaw motion statistics while removing the causal coupling.

**Object identity is not a confound.** The models are fit from the state/action
response (the peg's task-local pose `[x, y, yaw]` regressed against the
intervention), not from RGB or object appearance. There is no visual channel, so
"the model recognized a different object" cannot explain the result, and no
shape-identity visual control is required.

---

## 2. Ground-truth generator law `alpha*` (model-independent oracle)

The oracle is derived from the symmetry group at design time, **not** from the
demonstrations. It is therefore a valid causal test target.

Let `progress s in [0,1]` cover the four semantic phases with `phase_codes`
`0=align_keyed` (`s in [0,0.25)`), `1=enter_key` (`[0.25,0.5)`),
`2=unlock_yaw` (`[0.5,0.75)`), `3=circular_insert` (`[0.75,1]`), i.e. the
existing `progress_grid(bins)`. Then:

```text
alpha*_x(s) = 1        (all s, both tasks)   -- translation stays relevant
alpha*_y(s) = 1        (all s, both tasks)
alpha*_yaw^keyed(s)    = 1  for phase in {align_keyed, enter_key}
                       = 0  for phase in {unlock_yaw, circular_insert}
alpha*_yaw^circular(s) ~ 0  (all s; ideal target, approximate under finite contact)
```

Why `alpha*_yaw^circular ~ 0`: in the circular task the intervention `d_axial`
(socket axial yaw) is a **gauge** symmetry under ideal axial symmetry — rotating
a circular socket about its axis and rotating a circular peg about its axis are
both unobservable, so the task-local trajectory is invariant under `d_axial`.
The theoretical target is therefore `alpha*_yaw^circular = 0`. Under finite
simulation/contact perturbations (friction, gripper, solver contact) a small
residual rotational coupling can remain, so the operative target is
`alpha*_yaw^circular ≈ 0`, i.e. `|d yaw / d d_axial| -> 0` within simulation
tolerance — not a mathematically exact elimination of all rotational dynamics.
M1 (§7) treats the ideal `0` as the oracle and reads any excess as identification
error rather than a failed symmetry.

Why `alpha*_yaw^keyed` is a step: during alignment/entry the keyway pins the peg
yaw to the socket yaw (response +1); once the key clears, the circular
shaft-in-bore admits any yaw and the demonstrator returns to nominal (response 0).

---

## 3. Demonstrator protocol (the crux of the memorization control)

`alpha_j(s)` is **identified from the demonstration's response to the
intervention** `c = [du, dv, d_axial]` (the conditional `d trajectory / d c`), not
from the marginal trajectory distribution. A "trajectory-statistics memorizer"
would instead read the marginal: "the peg rotates during alignment". To separate
these, the circular task is generated in **two variants**:

```text
circular-honest   : the collector ignores d_axial in yaw; the yaw channel is a flat
                    nominal. alpha* = 0 by symmetry AND by absence of rotation.
circular-placebo  : the collector performs a FIXED yaw sweep dpsi_schedule(s) that is
                    INDEPENDENT of d_axial (e.g. the mean yaw profile of the keyed
                    demonstrations). alpha* = 0 by symmetry, but rotation IS present
                    in the marginal trajectory.
```

The **placebo** variant is the decisive control. In it:

* a **relation learner** (correct) reads the *correlation* `d yaw / d d_axial`,
  which is zero, and outputs `alpha_yaw = 0` **despite** the rotation being
  present;
* a **statistics memorizer** reads "rotation happens early", outputs
  `alpha_yaw ≈ 1`, and is indistinguishable from its keyed prediction.

The contrast
`alpha_yaw^keyed = step (rotation correlated)` vs
`alpha_yaw^circular-placebo = 0 (rotation present, uncorrelated)`
is what proves the model reads the causal relation, not the motion statistics.

Both variants are collected with the same solver/environment; only a
`--yaw-mode {honest,placebo}` flag differs. `circular-honest` is the headline
arm; `circular-placebo` is the memorization control.

---

## 4. Train / test matrix

One structural fact governs the matrix: in the frozen `SmoothFinitePDiagModel`,
`alpha(s)` is a **fit-time** quantity — a function of phase only, learned from the
training demonstrations, fixed thereafter. `predict(contexts)` applies the frozen
`alpha` to new interventions. Consequently "train X, test Y" decomposes into two
distinct things:

1. **primary — representation change** — refit on task-Y demos and read the new
   `alpha`. The claim is that the fitted representation *changes with the task
   relation*. This tier is settings A and A′ and contains the placebo arm (§3),
   the decisive control.
2. **secondary — cross-task mismatch** — keep task-X `alpha` frozen and score
   trajectory prediction on task-Y held-out interventions. These are behavioral,
   subject to the "do not look at trajectory error" caveat, and are **not** called
   "transfer": `alpha` is a task-specific representation, so the meaningful
   statement is sensitivity of a frozen representation to a mismatched task, not
   cross-task generalization.

| setting | train | evaluate | operational definition | status |
|---|---|---|---|---|
| A  | keyed | keyed | refit on keyed demos, read `alpha` vs `alpha*^keyed` (step) | primary (representation) |
| A' | circular | circular | refit on circular demos (honest + placebo), read `alpha` vs `alpha*^circular` (~0) | primary (representation; placebo here) |
| B  | keyed | circular | **cross-task mismatch**: frozen keyed `alpha` (step) applied to circular held-out yaw perturbations predicts a yaw response that does not exist -> excess cross-task yaw error | secondary (behavioral) |
| C  | circular | keyed | **cross-task mismatch**: frozen circular `alpha` (0) applied to keyed held-out perturbations misses the required yaw response -> excess cross-task yaw error | secondary (behavioral) |
| D  | keyed+circular mixed | per-context | context-conditional `alpha(s; task)` recovers per-task profiles from mixed demos | scoped extension (see §8) |

The cleanest summary of the whole design, expressed as a per-arm prediction:

```text
                    alpha_yaw^pre (align+enter)     alpha_yaw^post (unlock+insert)
keyed                          ~ 1                          ~ 0
circular-honest                ~ 0                          ~ 0
circular-placebo               ~ 0                          ~ 0
```

---

## 5. Expected `alpha_yaw` response

If the theory holds (relation learner):

* **keyed**: `alpha_yaw ≈ 1` in align/enter, dropping across the unlock
  boundary to `≈ 0` in insert — the same law already frozen in the rotated-axis
  confirmatory analysis (`g_axial_pre ~ 1`, `g_axial_final ~ 0.02–0.03`).
* **circular (both variants)**: `alpha_yaw ≈ 0` over the whole `s`.
* discrimination margin `alpha_yaw^pre(keyed) − alpha_yaw^pre(circular) ≈ 1`.

If the theory fails (trajectory memorizer):

* `alpha_yaw^keyed ≈ alpha_yaw^circular-placebo ≈ 1` (cannot tell them apart
  despite the different causal relation), and `alpha_yaw^circular-honest` differs
  only because its demos omit rotation (i.e. the difference tracks the marginal
  trajectory, not the symmetry group).

---

## 6. Baseline protocol

Every model is fit on the same data and its `jacobian_diag()[:, 2]` is read as
`alpha_yaw`, so all are scored on the **identical** metrics. The model roster is
the frozen `MODEL_ORDER` from `benchmark_phase_switch_baselines.py`:

```text
Frame-weighted     scalar w(s) -> forces alpha_x = alpha_y = alpha_yaw = w(s)
Phase scalar GP    scalar w(s), GP-smoothed
TP-GMM additive / TP-GMM SE(2)   frame-product mixture; alpha via finite-difference Jacobian
Generic RBF        unstructured phase-local RBF
Full operator      dense 3x3 per-progress operator (no symmetry prior)
Pdiag pointwise    diagonal pointwise regression (ablation)
Pdiag finite       the method: alpha_max-sigmoid RBF diagonal with smoothness prior
```

Two baselines are structurally decisive and should be reported explicitly:

* **Frame-weighted / Phase scalar GP** *cannot* represent the circular ground
  truth: because translation stays relevant (`alpha_x = alpha_y ≈ 1`) in both
  tasks, the shared scalar forces `alpha_yaw ≈ 1` in both. Its discrimination
  accuracy is at chance and its `E_alpha` on the circular arm is structurally
  large. This is not a fitting failure; it is the point.
* **Full operator** can fit either task but has no symmetry prior, so at low
  `N` its `alpha_yaw` is expected to overfit / be unstable across seeds.
* **Pdiag pointwise** (already in the roster) is the "diagonal parameterization
  without the finite/smooth symmetry prior" ablation — it isolates whether the
  diagonal *structure alone* suffices, answering "is diagonal structure itself
  enough?".

The claim is not "Pdiag fits better" but "**Pdiag recovers the correct generator
relevance with fewer demonstrations**" — the same few-shot framing already
frozen in Experiment B.

---

## 7. Evaluation metrics

**M1 — generator identification error** (primary, attacks the representation).
Matches the user's definition:

```text
E_alpha(task; N, model) = (1/T) sum_t ( alpha_yaw(t) - alpha*_yaw^task(t) )^2
```

`alpha_yaw` is the fitted diagonal profile; `alpha*` is the §2 oracle.
Reported per (task, N, model, seed), summarized as mean over seeds.

**M2 — symmetry discrimination accuracy** (primary). Let `alpha_yaw^pre` be the
mean of `alpha_yaw` over `{align_keyed, enter_key}`. Classify a fitted model as
"keyed" if `alpha_yaw^pre > 0.5`, else "circular". Accuracy is the fraction of
(model, task, N, seed) fits correctly classified.

**M2b — symmetry gap** (primary, the headline number). `G_psi =
alpha_yaw^pre(keyed) − alpha_yaw^pre(circular)`, theoretical value `G_psi* = 1`
(keyed ~1, circular ~0); report `|G_psi − 1|` as the deviation from the ideal.
The decisive headline is the placebo comparison `G_psi^placebo =
alpha_yaw^pre(keyed) − alpha_yaw^pre(circular-placebo) ~ 1`.

**M0 — switch fidelity** (keyed arm only). Reuse the frozen
`switch_diagnostics` 0.5-crossing to confirm the keyed arm reproduces the
already-validated transition location; this ties the new keyed arm to the
existing rotated-axis result.

**M3 — cross-task mismatch gap** (secondary, behavioral). `Delta = E_traj^cross −
E_traj^in`, where `E_traj` is held-out trajectory prediction error
(cross = train X, evaluate Y). Reported for completeness against the "trajectory
error" caveat; a relation learner's cross-task error is confined to the yaw
channel, a memorizer's is unstructured. **Not** a primary line of evidence.

---

## 8. Data budget and settings (frozen)

```text
orientation    : Q1 = identity only (vertical). The symmetry question is orthogonal
                 to the rotated-axis question; a Q2 extension is possible later but
                 is not part of this design.
anchor         : (-0.35, 0.10, 0.08)  -- the rotated-axis common anchor, so the keyed
                 arm can REUSE the frozen rotated_Q1 seed rollouts rather than re-collect.
keyed arm      : rotated_Q1_seed_{20260818,20270818,20280818}.h5 (already frozen, 39 cond.)
circular arm   : new collection, identical protocol except geometry + yaw-mode, in BOTH
                 honest and placebo variants.
contexts       : 39-condition frozen manifest (30 mixed + 8 isolated + 1 baseline).
demonstrations : N in {5, 10, 20} mixed demos, subsampled from the 30-mixed manifest
                 via a FROZEN subset file (same discipline as fewshot_subsets.json).
perturbation tests : the 8 held-out isolated interventions (+-30deg, +-15deg yaw;
                 +-0.015 x/y) plus the zero baseline.
seeds          : 20260818, 20270818, 20280818
retries        : R_max = 5
noise          : robot_init_qpos_noise = 0.01 rad  (unchanged; never reduced)
```

**Setting D** (context-conditional `alpha(s; task)`) requires extending
`SmoothFinitePDiagModel` to condition the diagonal on a task/context label. It is
the strongest possible form of the claim ("discriminates by context") but is a
model change, so it is **scoped separately** and deferred until M1/M2 and the
placebo control are confirmed. The core claim rests on A, A', B, C.

---

## 9. Implementation plan (listed, not yet started)

1. **Environment** — add a `keyed: bool = True` flag to
   `phase_switch_symmetry_env.py` and a second `@register_env` wrapper
   (`CircularPhaseSwitch-v1`). `_build_peg` omits the key when `keyed=False`;
   `_build_socket` swaps the rectangular keyway for a circular collar
   (radius `SHAFT_RADIUS + GATE_CLEARANCE`). No other geometry or solver change.
2. **Collector** — a `collect_phase_switch_circular.py` (or a flag on
   `collect_phase_switch_rotated.py`) that emits the honest and placebo yaw
   schedules, reusing `solve_rotated`, the frozen manifest reader, the SHA256
   enforcement, and `R_max=5` / `noise=0.01`.
3. **Fit-and-score harness** — `benchmark_symmetry_transfer.py` that loads the
   keyed (reused) and circular (new) H5 files, subsamples `N in {5,10,20}` from
   a frozen subset manifest, fits all `MODEL_ORDER` models, reads
   `jacobian_diag()[:, 2]`, and scores M0/M1/M2/M3 into CSVs + a summary JSON +
   a figure.
4. **Freeze** — `symmetry_transfer_experiment.json` (preregistered, mirroring
   `rotated_axis_experiment.json`) plus this document.

### Formal execution order

```text
Step 0  circular geometry validation (validate_circular_geometry.py): nominal
        insertion success + isolated yaw intervention response, confirming
        alpha*_yaw^circular ~ 0 before any full collection.
Step 1  collect circular-honest, 3 seeds x 39 conditions, R_max=5, noise=0.01.
Step 2  collect circular-placebo, same protocol.
Step 3  fit-and-score benchmark at N in {5,10,20}.
```

---

## 10. Decision criteria

The headline result is **not** "Pdiag has the lowest trajectory error"; it is the
symmetry gap `G_psi ~ 1` between the keyed arm and the circular-placebo arm
(rotation present in both, causal coupling present in only one), recovered from
few demonstrations.

**Confirm** (relation learner):

* keyed arm: `alpha_yaw^pre ~ 1`, `alpha_yaw^post ~ 0`, switch location matches
  the frozen result (M0 passes);
* circular-honest and circular-placebo arms: `alpha_yaw ~ 0` throughout;
* discrimination accuracy 100% for Pdiag at every `N`; Frame-weighted / Phase
  scalar GP at chance;
* `E_alpha` for Pdiag stays low at `N = 5` while Full operator / TP-GMM / RBF
  degrade — the few-shot claim.

**Refute** (trajectory memorizer):

* `alpha_yaw^keyed ≈ alpha_yaw^circular-placebo` (no discrimination despite the
  different causal relation), or
* `alpha_yaw^circular-placebo ≈ 1` (rotation present but uncorrelated still
  reads as "yaw relevant").

---

## Interpretation boundary

This is a controlled simulation study with one robot, one common anchor, and a
frozen three-seed replication. The geometry contrast changes exactly one feature
(the SO(2)-breaking key/keyway); the contact material of the tight passage
necessarily differs (key-vs-keyway vs shaft-vs-collar) and is recorded, not
hidden. The ground-truth generator law is an *a priori* symmetry-group oracle,
so the experiment is a genuine causal test of "relation vs. statistics" rather
than a fit-quality benchmark. Setting D (context-conditional `alpha`) is a
scoped extension, not part of the initial confirmatory criteria.
