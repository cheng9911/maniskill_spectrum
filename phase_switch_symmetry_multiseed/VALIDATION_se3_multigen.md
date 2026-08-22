# Frozen Validation: Multi-Generator SE(3) Task (du + yaw)

This document freezes the multi-generator supplement — a THIRD SE(3) task that
answers the reviewer question behind the claim "generator relevance": is the
model learning a relevance **vector** (which generators matter), or just an
argmax over one winning generator? The frozen six-generator experiment showed
yaw is the UNIQUE selective generator in the keyed task; this supplement shows
the model also recovers **two simultaneously selective generators** when the
task genuinely has two: a keyed gate (constrains yaw AND lateral position
during entry) above a RECTANGULAR SLOT that releases BOTH yaw and du once the
key clears, while dv stays tracked.

The decisive question: **does the model recover BOTH selective generators
(`alpha_du > 0` AND `alpha_yaw > 0` pre-clearance, both dropping at unlock)
as the top-2 clearance-drop set, with the other four non-selective?**

## Frozen design

```text
env:       KeyedCircularPhaseSwitchSE3Multigen-v1  (keyed gate -> rectangular slot)
anchor:    (-0.15, 0.00, 0.08), Q1 identity (same as the frozen SE(3) supplement)
seeds:     20260818, 20270818, 20280818
contexts:  the SAME frozen 75-condition se3_fixed_contexts.json
           (60 mixed + 14 isolated + 1 baseline; +-0.012 m / +-15 deg / +-30 deg)
subsets:   the SAME frozen 18 subsets (N in {8, 15, 30} x {random, qualified} x 3)
model:     Pdiag finite (SE(3)), identical configuration to the frozen benchmark
```

Slot geometry (socket-local x = du, y = dv):

```text
SLOT_HALF_X = 0.044   wide in du: admits the key (0.021) plus the full +-0.012 m
                      du relaxation with envelope margin
SLOT_HALF_Y = 0.026   narrow in dv: fits the key's axis-aligned envelope at the
                      largest relative yaw in the +-30 deg range
                      (0.021 sin30 + 0.014 cos30 = 0.0226), but a full dv
                      relaxation would push the key edge to 0.014 + 0.012 =
                      0.026 = the wall, so dv stays tracked
```

Why the slot cannot be key-narrow (0.016) in y: the slot rotates WITH the
socket, so a peg relaxed to world yaw 0 sits at RELATIVE yaw = -socket_yaw.
At +-30 deg the key's axis-aligned envelope is 0.0226 > 0.016 — a key-narrow
slot would collide with the key after yaw relaxation, killing the task. The
0.026 half-height is the minimum that admits the yaw release while still
blocking a full dv relaxation (0.026 - 0.014 = 0.012 = exactly the dv
intervention magnitude, i.e. dv relaxation is infeasible, not merely
unchosen).

Oracle (model-independent, from the solver), over the four insertion phases
align_keyed / enter_key / unlock_yaw / circular_insert:

```text
generator      multigen
du             [1,1,0,0]    (selective, NEW)
dv             [1,1,1,1]    (tracked: slot's narrow y)
dw             [1,1,1,1]
roll           [1,1,1,1]
pitch          [1,1,1,1]
yaw            [1,1,0,0]    (selective, as in the keyed task)
```

Solver (`solve_se3_multigen`): the frozen `solve_se3` skeleton, with the
unlock_yaw phase performing TWO moves — first yaw -> 0 (du still matched),
then du -> 0 (`target_pose_at_relaxed_du`, socket centre recomputed with
du = 0) — so both generators share the [1,1,0,0] profile and the du
relaxation rides in the existing phase code (no new phase).

Source hashes:

```text
phase_switch_symmetry_env_multigen.py   15e6102a6912d9918694501d55ce51803bb7fb39f71c4ffdcd896e37b6878bd4
collect_se3_multigen_rollouts.py        6c55a91c689d8efb016630fc6aa0cf57a55635be65cad5064507f0cb123cb3ff
retry_multigen_cell.py                  1ac07da1ba6706ed033105324f2cf5e3cc5534dc3e0c950085b1e3ce56a941e0
benchmark_se3_multigen.py               df65a56201358e0e9b48703e97657f8daca4105e0261ba2cd0af01e385da4025
make_se3_multigen_figure.py             3b9a45956a849c62f8037c55ea854a3ffdd531aac62322d00b8cccde31a9296c
```

The frozen SE(3) supplement's files, manifest, subsets, and results are
untouched; the multigen task is purely additive in its own files, reusing the
frozen manifest and subsets.

---

## 1. Geometry feasibility (probe before collection)

Four targeted probe episodes before the full collection, all successful with
all eight phase codes:

```text
isolated du=+0.015      success, final x = -0.1451 vs relaxed goal -0.1500
                        (matched x would be -0.135: the du slide happened),
                        max contact 9.9 N (key brushing the slot during the slide)
isolated yaw=+-30 deg   success, 0 contact (key fits the slot at relative yaw)
mixed cid 49            success (the known stochastic-IK corner from the frozen
                        SE(3) collection), 0 contact
```

## 2. Collection

`1 arm x 3 seeds x 75 conditions`, R_max = 5, noise 0.01, firm gripper,
3 mm incremental steps. Result: **all 75/75 conditions usable per seed
(60/60 mixed usable)**, with the same caveat as the frozen collection:
condition 49 (`du=0.0067, dv=0.0070, dw=0.0034, roll=+13.6, pitch=-6.0,
yaw=+29.3` deg) failed 5/5 in seed 20270818 with "motion planning failed"
(the identical stochastic IK failure the frozen keyed arm showed at seed
20260818). It was re-collected with the frozen R_max=12 discipline
(`retry_multigen_cell.py`, appends to the H5): attempt 6 failed, attempt 7
succeeded; the two extra attempts are retained in the H5.

```text
multigen 20260818: 75/75 usable (60 mixed)
multigen 20270818: 75/75 usable (60 mixed, condition 49 recovered at attempt 7/12)
multigen 20280818: 75/75 usable (60 mixed)
```

## 3. Benchmark

`benchmark_se3_multigen.py` ran 216 fits (3 seeds x 18 subsets x 4 models),
**0 failures**, scoring each against the model-independent solver oracle.

Headline metric **M-multi**: rank the six generators by the post-clearance
relevance drop `Delta_j = mean(alpha_j | phase<2) - mean(alpha_j | phase>=2)`.
Correct iff the top-2 set is exactly {du, yaw}, both with `Delta > 0.2`, and
all four others `< 0.2`.

```text
model                 M-multi accuracy (N=8 / 15 / 30)
Pdiag finite          0.944 / 1.000 / 1.000      (17/18, 18/18, 18/18 fits)
Full operator         0.778 / 0.889 / 1.000
Pdiag pointwise       0.667 / 0.778 / 1.000
Frame-weighted        0.000 / 0.000 / 0.000      (negative control: a shared
                                                 scalar cannot express TWO
                                                 selective generators)
```

Per-generator clearance drop `Delta_j` (Pdiag finite, means over 18 fits):

```text
generator   N=8      N=15     N=30
du          0.372    0.392    0.404    (selective, recovered)
dv          -0.154   -0.157   -0.149   (tracked)
dw          -0.125   -0.148   -0.105
roll        -0.165   -0.155   -0.143
pitch       -0.204   -0.177   -0.191
yaw         0.532    0.562    0.540    (selective, recovered)
```

Pre-clearance relevance (Pdiag finite): du 0.755/0.780/0.797 and yaw
0.765/0.778/0.771 land at ~0.77 (the align-phase transient, as documented in
the frozen benchmark), while dv/dw/roll/pitch sit at 0.79-0.88 — the two
selective generators are separated from the tracked ones by the clearance
drop, not by the pre-magnitude.

Supporting metrics (Pdiag finite):

- **M0 switch fidelity** — both switches detected in 100% of fits at every N:
  du switch location ~= 0.69, yaw ~= 0.58 (oracle 0.5; the later du location
  reflects the two-move unlock: yaw releases first, then du).
- **E_alpha vs the multigen oracle** — Pdiag finite best and improving with N:
  0.130 -> 0.124 -> 0.119. (Frame-weighted worst, flat ~0.23-0.27.)

Control (no new fits): the FROZEN keyed single-generator arm re-scored with
the same M-multi criterion from its frozen fits CSV gives **0.000 accuracy at
every N** — in the single-generator task the top-2 set is never {du, yaw}
(`Delta_du` ~= -0.20 there: du is tracked, and the metric correctly does NOT
fire). The M-multi reading is therefore task-specific, not a model bias.

Fig. SE3-M1 (`make_se3_multigen_figure.py`): 6 x 1 relevance grid, N=30, 18
fits — du and yaw both drop across the shaded unlock phase while the four
tracked generators stay ~1.

---

## Interpretation

The multi-generator task answers its decisive question **affirmatively**: the
model recovers relevance as a **vector**, not an argmax. In a task where TWO
generators are simultaneously selective — du (the slot's wide x releases the
lateral translation) and yaw (the slot's cross-section admits the key at
every relative yaw in the intervention range, releasing the axial rotation) —
the Pdiag-finite model recovers BOTH as the top-2 clearance-drop set in
94-100% of fits (rising to 1.000 at N >= 15), with both switches detected in
100% of fits and the four tracked generators correctly held ~1. The
Frame-weighted negative control (a shared scalar) fails every fit, exactly
because per-generator selectivity cannot be expressed by one weight; the
dense Full operator and the unregularized pointwise diagonal catch up only at
N=30, consistent with their known noisier behavior in the frozen benchmark.
And the same metric applied to the frozen single-generator keyed arm fires
0% of the time — the top-2 reading is specific to the task's actual
generator structure.

Combined with the frozen six-generator supplement (yaw is the unique
selective generator in the keyed task, S_yaw ~= 0.65 vs S_others <= 0.16),
the paper's claim is now complete in both directions: the model neither
misses a relevant generator (this task) nor hallucinates one (the frozen
negative controls).
