# Frozen Validation: Basis / Coordinate Ablation (SE(3) supplement)

This document freezes the basis/coordinate ablation — the Priority-3 supplement
that answers the reviewer question behind the generator claim: **is the model
learning TASK relevance, or just benefiting from a manually chosen axis?** The
Pdiag prior is a sparsity claim about the GENERATOR BASIS: the expert response
P(s) is diagonal in the task-local socket frame `[du, dv, dw, roll, pitch,
yaw]`. This ablation re-expresses the SAME interventions in two arbitrary
rotated bases and measures what happens: if the task frame is principled, the
diagonal prior must degrade off that frame while basis-free models stay
unchanged; if the axis were just lucky, no such separation would appear.

**Answer (frozen): the diagonal sparsity holds only in the task frame.** In
the task basis the empirical response operator is diagonal (enter-phase
`diag(J) ~ [1,1,1,1,1,1]`, off-diagonal norm 0.15); in a rotated basis it is
dense (off-diagonal norm 2.77, 83% of the operator's mass). The rotated-basis
Pdiag fit degrades on every score — prediction error 2.2-13x, recovered-law
error 2.1-4.1x, and the recovered diagonal contorts — while the dense Full
operator and the provably basis-free Frame-weighted scalar are unchanged.

## Frozen design

```text
data:       the FROZEN SE(3) datasets (keyed + circular_honest), 3 seeds,
            the SAME frozen 75-condition manifest and 18 subsets
            (N in {8,15,30} x {random, qualified} x 3). No new collection.
procedure:  contexts conjugated by a fixed rotation R,
            context_rot = pose6(R^-1 T(context) R)  (exact Ad(R^-1) twist
            conjugation; rotation AND translation cross-terms)
            R1 = Rx(30) Ry(-20) Rz(40) deg,  R2 = Rx(-35) Ry(25) Rz(15) deg
            (deliberately NOT axis permutations, which would leave the
            diagonal structure intact; these mix the task-local insertion
            axis away from every basis axis)
            the nominal-frame recovery (benchmark_se3_transfer
            .nominal_frame_se3) runs per basis and adapts automatically
models:     the SAME frozen 4-model suite, SAME Pdiag configuration
scoring:    e_data = in-sample prediction error in metric units
            (basis-independent), scored at FULL precision by
            analyze_se3_basis_ablation.py — the 4-decimal summary tables of
            the ablation script round the whole effect away (local e_data
            ~1e-5 and rotated ~5e-5 both print as 0.0000), so the analysis
            script is the source of record
runs:       324 jobs (2 tasks x 3 seeds x 18 subsets x 3 bases) x 4 models
            = 1296 fits, 0 failures
```

Source hashes:

```text
benchmark_se3_basis_ablation.py   f0819a82c0ab8645f1467bca5eab572abd8337b1321ce9fd6e8ebf2b3407d064
analyze_se3_basis_ablation.py     d25304c1f5f102e9f179423dff69270617ab17f165265d82418a581c2541ca98
```

The frozen SE(3) supplement's files, manifest, subsets, datasets, and results
are untouched; the ablation is purely additive in its own files, reusing the
frozen manifest and subsets. The frozen benchmark's own "Expected" block is
CONFIRMED by this analysis at full precision.

---

## 1. Mechanism: what the rotation does and does not do

The rotated-basis Pdiag model composes the rotated context twist through its
RECOVERED nominal frame C0' (which adapts to the rotated contexts); its
response operator on the local twist is therefore `Ad(A) diag(alpha') Ad(R^-1)`
with `A = C0'^-1 C0_true`. For the rotated family to contain the true
response `diag(alpha_true)`, a diagonal solution of

```text
diag(alpha') = Ad(A) diag(alpha_true) Ad(R)
```

must exist — impossible for generic R, so **the true response is
unrepresentable off the task basis**. The best-achievable diagonal is the
diagonal projection of the right-hand side; with `A ~ I` (probe: A is a
4.44 deg rotation, first order in the context mean; A = I exactly for the
ideal model) this is the RESCALED law

```text
alpha'_k(s) ~ alpha_k(s) * R[k,k]
```

the true generator relevance times the diagonal of the basis rotation
(Ad(R) = blkdiag(R, R): each triple rescales by the same diag(R)). Two
consequences, both probe-verified on subset_id 12 (keyed, seed 20260818,
N=30 random):

1. **The selectivity STRUCTURE survives, the magnitudes do not.** The yaw
   drop rescales as 0.814 -> 0 (measured unlock-phase diag(J)_yaw = 0.34 vs
   predicted 0.41 x 0.814 = 0.334; insert-phase 0.00), but a tracked
   generator reads alpha ~ 0.72-0.81 instead of 1 — the rotated basis
   UNDER-READS relevance by `1 - R[k,k]` (R1: du 28.0%, dv 22.7%, dw 18.6%;
   R2: du 12.5%, dv 14.6%, dw 25.8%).
2. **The empirical response becomes dense.** Phase-wise regression of the
   trajectory perturbations on the intervention twists gives the response
   operator J(s): local enter-phase `diag(J) = [1.00, 0.99, 0.95, 1.00,
   1.00, 1.00]` with off-diagonal norm 0.15; rotated-1 enter-phase
   `diag(J) = [0.78, 0.74, 0.78, 0.68, 0.77, 0.80]` (vs the predicted
   rescaled `[0.72, 0.773, 0.814]` per triple) with off-diagonal norm 2.77 —
   83% of the operator's Frobenius mass is off the diagonal.

Why e_data stays small in absolute terms everywhere: the finite action
absorbs the rotation through the recovered nominal frame, so only the
off-diagonal residual of the response is unrepresentable — the ablation must
therefore be scored by RELATIVE degradation and recovered-law structure, not
by absolute fit error.

## 2. Results (aggregate over 18 fits per cell)

**e_data — relative degradation per basis (full precision):**

```text
Pdiag finite    keyed:    local 1.16/2.13/1.65 e-5  (N=8/15/30)
                          -> R1  4.51/5.48/6.10 e-5   (3.9x / 2.6x / 3.7x)
                          -> R2  3.29/4.75/4.58 e-5   (2.8x / 2.2x / 2.8x)
                circular: local 0.33/0.37/0.37 e-5
                          -> R1  3.78/3.65/4.88 e-5   (11.4x / 9.9x / 13.0x)
                          -> R2  2.32/3.10/3.38 e-5   (7.0x / 8.4x / 9.0x)
Full operator   0.95-1.10x (keyed), 1.11-1.57x (circular)     INVARIANT
Frame-weighted  1.00-1.15x (both tasks)                        INVARIANT
Pdiag pointwise 1.02-1.24x (both tasks; its own noise floor dominates)
```

The circular degradation exceeds the keyed one because circular's anisotropy
is full-trajectory (yaw off in all four phases) while keyed's lives only
post-unlock, and the circular LOCAL baseline is near-perfect (pure identity
response) — the ratio is sharper there.

**e_alpha — recovered diagonal vs the true generator law:**

```text
Pdiag finite    keyed:    local 0.105/0.102/0.095
                          -> R1  0.354/0.266/0.269    -> R2  0.247/0.251/0.202
                circular: local 0.078/0.071/0.070
                          -> R1  0.321/0.239/0.240    -> R2  0.228/0.250/0.201
```

(2.1-3.4x structure loss keyed, 2.9-4.1x circular, every N, both rotations.
The local column reproduces the frozen M1 numbers exactly — pipeline
cross-check.)

**Off-diagonal mass of the dense Full operator** (||op - diag(op)||/||op||):
0.884-0.897 local -> 0.918-0.936 rotated — the dense operator absorbs the
basis change by becoming MORE off-diagonal, exactly as a model that has no
diagonal prior must.

**Rescaling floor (closed form, analytic):** the minimum e_alpha achievable
by ANY diagonal in the rotated bases is keyed 0.0520 / 0.0289, circular
0.0491 / 0.0233. The FITTED e_alpha sits 4.9-10.7x above this floor: the
pose6-weighted objective pushes the fit to contort the diagonal and chase
the off-diagonal response. Probe evidence (subset_id 12): the fitted rotated
profile sets dv ~ 0.20 (law: 0.773), roll ~ 0.44 (0.72), and clips yaw at
the 1.25 alpha_max bound during enter (law: 0.814) before collapsing to
0.15 at insert; that contorted diagonal achieves e_data 6.9e-5 — BETTER
than the analytic projections (rescaled law 1.71e-4, exact-A projection
1.74e-4) — yet 2.6x worse than the local fit (2.6e-5). Off the task basis
the model faces a law-vs-data tension it never faces on it: fitting the
trajectories requires giving up the generator law.

---

## Interpretation

The ablation's decisive question is answered **affirmatively for the task
frame**: the diagonal sparsity prior is a property of the TASK in the
task-local basis, not of the model's parameterization.

- In the task basis the empirical response operator IS the diagonal law
  (enter-phase diag ~ 1, off-diagonal norm 0.15); in either rotated basis it
  is dense (off-diagonal norm 2.77) and its diagonal is the rescaled law
  `alpha_true * diag(R)` — the model can still see WHICH generators drop,
  but it misreads every magnitude by up to 28%, and its prediction error
  rises 2.2-13x and its recovered-law error 2.1-4.1x, at every N in both
  rotations.
- The invariance controls isolate the prior: the dense Full operator
  (1.0-1.6x, absorbing the rotation into MORE off-diagonal mass 0.89 ->
  0.93) and the Frame-weighted shared scalar (1.0-1.15x) are basis-free.
  The scalar family is provably Ad-invariant — `C0' Exp(w t') C0'^-1 X0'`
  with `C0' = C0 R` and `t' = Ad(R^-1) t` is exactly the local family — so
  its invariance is an internal consistency check on the whole pipeline, not
  just an observation.
- Beyond the unavoidable rescaling, the fitted diagonal contorts further
  (clipped yaw, collapsed dv) because the fit objective rewards chasing the
  off-diagonal response at the law's expense — a distortion that exists ONLY
  off the task basis. In the task basis, recovering the law and fitting the
  data are the same problem; that coincidence is what the manual basis buys.

Combined with the frozen six-generator supplement (yaw uniquely selective,
S_yaw ~ 0.65) and the multi-generator task (du + yaw recovered together,
M-multi 0.94-1.00), the generator framework is now closed on all three
fronts: the model finds the right generators (six-generator), finds ALL of
them (multi-generator), and does so BECAUSE of the task frame (this
ablation) — not because any diagonal parameterization would.
