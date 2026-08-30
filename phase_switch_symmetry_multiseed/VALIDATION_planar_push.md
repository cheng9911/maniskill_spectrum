# Frozen Validation: Planar-Push Task (rebutting circular-symmetry tailoring)

This document freezes the planar-push supplement — a genuinely DIFFERENT
physical task added to rebut the reviewer concern "你其实针对 circular symmetry
定制了一个模型" (you tailored a model for circular symmetry). Every frozen task
so far is a peg/cylinder insertion whose socket carries circular symmetry, so a
skeptic could argue the recovered relevance vector (yaw selective, etc.) is an
artifact of that geometry rather than a general property of manipulation. This
task has no rotational symmetry at all: a **block slid across a flat table**
under a downward table constraint, to a ghost 6-DOF target.

The table constraint reverses the generator roles. In peg-in-hole the
out-of-plane generators (dw/roll/pitch) are TRACKED (the peg is keyed into the
socket at an out-of-plane tilt); in planar push they are SUPPRESSED (the block
must stay flat on the table), while the in-plane generators (du/dv, and yaw for
the heading arm) are TRACKED. The decisive question is therefore: **does the
model recover a *different* relevance vector for a different physical task —
in-plane tracked / out-of-plane suppressed here, versus the phase-dependent
yaw-selectivity of the frozen peg-in-hole task — from byte-identical
interventions, with no change to the model?**

## Frozen design

```text
env:       PlanarPush-v1  (one NEW @register_env class in its own file)
anchor:    (0.10, 0.0, 0.01) = block-centre height, Q1 identity (table normal = z)
seeds:     20260818, 20270818, 20280818
contexts:  the SAME frozen 75-condition manifest (planar_push_fixed_contexts.json,
           byte-identical 6-vectors to the frozen peg-in-hole manifest)
subsets:   the SAME frozen 18 subsets (N in {8, 15, 30} x {random, qualified} x 3)
model:     Pdiag finite (SE(3)) + the same 3 companions, byte-identical reuse
```

Two arms, one env, distinguished by a `goal_heading` flag:

```text
heading_push    goal_heading = true   block aligns position AND heading
free_yaw_push   goal_heading = false  block aligns position only, heading free
```

Both arms are solved by the same guided push (grasp the block from above,
lift + slide to the target in-plane position, then either rotate the grasped
block to the target heading — heading_push only — or leave it at grasp heading,
then lower + release). The ghost renders the full 6-DOF target at C0 o
intervention, so dw/roll/pitch genuinely vary in the demonstration data (which
is what lets the model learn `alpha = 0` for them).

Oracle (model-independent, from the solver), over the four push phases
reach / push / align / retract:

```text
generator      heading_push       free_yaw_push
du             [0,1,1,1]          [0,1,1,1]      (in-plane translation, tracked)
dv             [0,1,1,1]          [0,1,1,1]      (in-plane translation, tracked)
dw             [0,0,0,0]          [0,0,0,0]      (vertical, SUPPRESSED)
roll           [0,0,0,0]          [0,0,0,0]      (tilt, SUPPRESSED)
pitch          [0,0,0,0]          [0,0,0,0]      (tilt, SUPPRESSED)
yaw            [0,0,1,1]          [0,0,0,0]      (heading: tracked / free: suppressed)
```

Active phases = push + align (codes 1, 2); reach/retract are setup/teardown and
masked from scoring. The SELECTOR (which generators are ever tracked) is
static per arm: {du,dv,yaw} vs {dw,roll,pitch} for heading_push, {du,dv} vs
{dw,roll,pitch,yaw} for free_yaw_push — a second, cross-arm contrast against
peg-in-hole's phase-dependent yaw drop.

Source hashes:

```text
phase_switch_symmetry_planar_push_env.py   46e640f0bb7d629daeadfa052d4a11f5ea215258853645513d93750a300598c3
phase_switch_se3_baselines.py              820ac06d795f51e0332444796b37f8ac993262b5903a4c916722e04ef0e24080   (frozen, byte-identical)
collect_planar_push_rollouts.py            13c3d8c38dba346a870ffa7e52c76061a7c36df26874a70c11b896d7dcb69459
probe_planar_push.py                       f668dbc2205f18ca42f73e11d3aac83ce61ec18fd49203ca61cab7425b941d50
benchmark_planar_push.py                   c0d4d3fdc97d42701da31a151c535614d86b8d82553e89e6f06d6e549c50c637
make_planar_push_figure.py                 b9db4616829365a6f0369e4594bcad678041f0cdc0134cfbd746d5a22f6eba13
```

The frozen SE(3)/SE(2) env classes and the model suite are byte-identical; the
planar-push task is purely additive (one new env file + collector/probe/
benchmark/figure), reusing the frozen manifest and subsets. Every frozen task,
manifest, subset, and result file is untouched.

---

## 1. Phase-0 feasibility gate (probe before collection)

`probe_planar_push.py` stressed the two preregistered risks — push reach across
in-plane extremes, and align-phase yaw controllability — across 7 contexts x 2
arms. The initial non-prehensive point-push reached the target position
reliably (pos_err ~0.007) but its corner-nudge yaw control was unstable and
out of IK reach at +-30 deg. Per the gate, the solver was switched to guided
push (grasp + slide), which passes **14/14** cases: heading_push 7/7,
free_yaw_push 7/7, with free_yaw heading staying ~0 (heading_err <= tol) in
every yaw/diag context.

---

## 2. Collection

`1 env x 2 arms x 3 seeds x 75 conditions`, R_max = 5, robot-init-qpos noise
0.01, firm gripper. Result: **75/75 success per seed per arm** (60 mixed + 14
isolated + 1 baseline).

**Solver fix (documented after collection).** A post-collection data audit
found that the original align-phase wrist-yaw rotation (a single rotation by
the full heading residual) occasionally levered the grasped block OFF the table
in a rare (~2-3/50 align episodes, seed-dependent) physics instability — block
z reaching 0.69-0.77 m, pitch/roll ~0.75-0.85 rad. This spurious out-of-plane
motion is attributed to the suppressed dw generator by the Pdiag-finite model
at N=30 (`alpha_dw` 0.106 -> 0.771), degrading the headline M_oop(heading_push)
to 0.389. The rotation was changed to a closed-loop sequence of bounded 0.05
rad steps (<=60 passes); the env, manifest, subsets, model, and oracle are all
unchanged. heading_push was re-collected (all 3 seeds, 75/75 success); the
post-fix block z never exceeds 0.0255 m across every align-present episode
(0 episodes with z > 0.05, versus 5 exceeding 0.69 m in the original).
free_yaw_push has no align rotation and its original clean data is unchanged.
The Phase-0 gate re-passed 14/14 with the fixed solver.

```text
heading_push  20260818: 75/75      free_yaw_push  20260818: 75/75
heading_push  20270818: 75/75      free_yaw_push  20270818: 75/75
heading_push  20280818: 75/75      free_yaw_push  20280818: 75/75
```

Caveat (shared with the frozen benchmark): ManiSkill marks `terminated` as
soon as `success` fires, so episodes terminate during push (small |yaw|) or
align (large |yaw|); the align/retract phases are therefore often absent. A
condition is "usable" iff its final step succeeded and no step was truncated.

---

## 3. Benchmark

`benchmark_planar_push.py` ran **432 fits** (2 arms x 3 seeds x 18 subsets x 4
models), **0 failures**, each scored against the model-independent solver
oracle.

Headline metric **M_oop** (out-of-plane suppression): over the active phases
(push+align), heading_push is correct iff max-active `alpha_j < 0.5` for all of
{dw, roll, pitch} AND `> 0.5` for all of {du, dv, yaw}; free_yaw_push is
correct iff `alpha_j < 0.5` for all of {dw, roll, pitch, yaw} AND `> 0.5` for
{du, dv}.

```text
model                 heading_push (N=8 / 15 / 30)    free_yaw_push (N=8 / 15 / 30)
Pdiag finite          0.889 / 1.000 / 1.000            1.000 / 1.000 / 1.000
Pdiag pointwise       1.000 / 1.000 / 1.000            1.000 / 1.000 / 1.000
Full operator         0.889 / 1.000 / 1.000            0.944 / 1.000 / 1.000
Frame-weighted        0.000 / 0.000 / 0.000            0.000 / 0.000 / 0.000   (negative control)
```

Per-generator max-active relevance `alpha_j` (Pdiag finite, means over 18
fits):

```text
heading_push   N=8      N=15     N=30        free_yaw_push  N=8      N=15     N=30
du             0.968    1.031    1.101       du             0.991    0.997    1.011
dv             1.025    1.035    1.018       dv             0.939    1.021    0.999
dw             0.087    0.032    0.041       dw             0.020    0.011    0.007
roll           0.042    0.020    0.020       roll           0.033    0.023    0.016
pitch          0.230    0.176    0.093       pitch          0.039    0.026    0.022
yaw            0.659    0.677    0.696       yaw            0.108    0.048    0.050
```

The suppressed trio collapses toward 0 in BOTH arms (`dw` heading: 0.087 ->
0.041; `pitch` heading is the largest residual at 0.093-0.230 and falls
monotonically with N, i.e. small-sample noise, not a systematic out-of-plane
signal), while du/dv hold at ~1. The heading arm's yaw is ~0.66-0.70 (see the
residual note below) versus ~0.05-0.11 in free_yaw — the same generator is
selective in one arm and suppressed in the other.

Cross-arm contrast **M_yaw** (heading vs free): correct iff
`alpha_yaw(heading) > 0.5` AND `alpha_yaw(free) < 0.5`.

```text
model                 M_yaw accuracy (N=8 / 15 / 30)    heading yaw    free yaw
Pdiag finite          1.000 / 1.000 / 1.000              0.66-0.70      0.05-0.11
Pdiag pointwise       1.000 / 1.000 / 1.000              0.67-0.71      0.03-0.09
Full operator         1.000 / 1.000 / 1.000              0.69-0.72      0.04-0.06
Frame-weighted        0.833 / 0.944 / 1.000              0.57-0.58      0.38-0.40
```

Note the Frame-weighted row: its single shared scalar happens to straddle the
0.5 threshold (0.58 vs 0.40) so it "passes" M_yaw at N=30 — but its heading/
free separation is ~0.18, versus ~0.65 for the Pdiag models. M_yaw's
discrimination therefore lives in the *magnitude* of the separation, not the
threshold pass; M_oop (above) is the metric that cleanly fails the
Frame-weighted control (0.000), because one scalar cannot simultaneously be
`< 0.5` for dw/roll/pitch and `> 0.5` for du/dv/yaw.

Identification error **E_alpha** against the full phasewise oracle (Pdiag
finite): heading_push 0.072 / 0.062 / 0.057, free_yaw_push 0.042 / 0.038 /
0.038 — best of the four models and improving with N (Frame-weighted worst,
flat ~0.18-0.20).

Cross-task out-of-plane contrast (no new collection): the frozen peg-in-hole
KEYED arm's out-of-plane relevance (from `se3_transfer/se3_transfer_fits.csv`,
Pdiag finite, N=30) is dw 0.875 / roll 0.832 / pitch 0.818, versus the
planar-push heading arm's dw 0.041 / roll 0.020 / pitch 0.093. The same model,
on byte-identical interventions, reports out-of-plane relevance ~0.82-0.88 in
the 3D peg task and ~0.02-0.09 in the 2D table task — a ~0.8 gap in the
direction the table constraint predicts.

Fig. PP-M1 (`make_planar_push_figure.py`): 6 x 2 relevance grid, N=30, 18
fits — rows = generators [du,dv,dw,roll,pitch,yaw], columns = arms. du/dv rise
to ~1 at push in both arms; yaw rises to ~1 at align in heading_push only; the
suppressed trio stays ~0 in both.

---

## Interpretation

The planar-push task answers its decisive question **affirmatively**: the model
is not tailored to circular symmetry. In a task with no rotational symmetry —
a block on a flat table — the Pdiag-finite model recovers a *different*
relevance vector, exactly the one the table constraint dictates: the in-plane
generators du/dv are tracked (~1), the out-of-plane generators dw/roll/pitch
are suppressed (<= 0.09 at N=30), and yaw is tracked in the heading arm
(~0.70) but suppressed in the free-yaw arm (~0.05) — the same generator both
selective and non-selective across the two arms of a single task. The headline
M_oop is 1.000 at N >= 15 for the Pdiag models in both arms, and the
Frame-weighted negative control fails every fit (0.000), because a shared
scalar cannot express per-generator selectivity. The cross-task contrast is the
sharpest evidence: the same model reports out-of-plane relevance ~0.82-0.88 in
peg-in-hole (where the peg genuinely moves out of plane) and ~0.02-0.09 in
planar push (where the table forbids it), on byte-identical interventions and
with no change to the model.

Two honest caveats bound the reading. First, the heading arm's yaw relevance
sits at ~0.70, not ~1.0 like du/dv: the align loop is interrupted by success
as soon as the heading enters the success tolerance (|heading_err| < 0.10 rad),
so the block stops ~0.08-0.10 rad short of the target heading — a constant
~5 deg offset, small against the +-30 deg yaw signal, that lowers the recovered
yaw magnitude without changing its sign (still > 0.5). Second, the heading
arm's pitch residual (0.093 at N=30) is the largest of the suppressed trio and
was the driver of the 2/18 N=8 M_oop misses; it falls monotonically with N,
marking it as small-sample noise rather than a systematic out-of-plane signal —
unlike the pre-fix fling, which left a stable `alpha_dw` ~0.77 at N=30 and was
removed at the solver, not by averaging.

Combined with the frozen six-generator and multi-generator tasks, the paper's
claim now holds across three physically distinct tasks: the model recovers the
relevance *vector* a task actually specifies — yaw uniquely selective in
keyed circular insertion, du+yaw jointly selective in the slotted key, and
in-plane tracked / out-of-plane suppressed in planar push — from the data, not
from any geometry-specific prior.
