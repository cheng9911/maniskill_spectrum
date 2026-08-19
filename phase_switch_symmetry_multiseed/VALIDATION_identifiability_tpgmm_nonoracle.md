# Frozen Validation: Identifiability, TP-GMM few-shot, Non-oracle progress

This document freezes the definitions and numerical results of three follow-up
experiments built on the fixed-context five-seed replication. All three reuse
existing rollouts only (`phase_switch_symmetry_multiseed/rollouts/seed_*.h5`);
no new physics data was collected.

The frozen design is the same as the multiseed/few-shot replication:

```text
seeds = 20260818, 20270818, 20280818, 20290818, 20300818
30 mixed + 9 isolated contexts, fixed; one usable rollout per condition.
experiment.json sha256 = 316e109d2e0a92fce9aaab7459ab54e99869f5bc2cc945a9827d668e3caac998
fewshot_subsets.json sha256 = 212b4362d0e3b997eba2c3ea730983eff23f32385934755987834e94a659f982
```

Source hashes:

```text
collect_model_aware_identifiability.py           250f980ddc9cb47658a569d160623aa6fba9921a6dbfc42b62fa78bbd9ddd407
analyze_model_aware_identifiability.py           916bdf1e26d2a83ef5158f70c4c95a5052ee6a7b971bcfd00919c6c0b0debe76
run_phase_switch_tpgmm_fewshot.py                0ec74d450120802f4fd95dd58a20ced9fb7385f9b6075d68c17dd9570f4b9fca
analyze_phase_switch_tpgmm_fewshot.py            61bfda09f25ece0c2220c80eac8b3f19567380b6ed2641de600d7d5426db24ab
run_phase_switch_tpgmm_fewshot_sensitivity.py    8f99fd758bf4c8aae4a7e0d28f712728b7db6e8fb1943ecece601e83182bce0c
analyze_nonoracle_progress.py                    f722116de2603b839034b1113de906e06541ae4ee2a1bd0eced026a6ad0d1e6b
```

---

## Experiment A — Model-aware identifiability (NARROW negative result)

### Frozen definition

The information matrix is a **metric-weighted Gauss--Newton** quantity, not
"Fisher information":

```text
I_D(theta) = sum_{n,k} J_nk(theta)^T W J_nk(theta),   J_nk = d yhat(c_n, s_k; theta)/d theta
W = diag(1, 1, ell^2), ell = 30 mm/rad  (W_SQRT = [1, 1, 0.03] = METRIC_SCALE)
```

* `I_post` is evaluated at each subset's own fitted parameters `theta_hat_D`.
* `I_design` is evaluated at `theta_ref = 0` (neutral prior, alpha = 0.625) with
  the fixed zero-intervention baseline curve as the nominal reference; only the
  subset's contexts vary.
* The second-difference smoothness regularizer is **not** added to the data
  information. The Jacobian is computed exactly through the chain rule
  `d yhat/d alpha` (central differences on alpha) x `alpha_max * sigmoid'(z) * phi`.
* Spectrum summaries are computed on the `lambda_max`-normalized spectrum so
  they are scale-free: `lambda_min`, `lambda_ratio = lambda_min/lambda_max`,
  `log_det(I_norm + 1e-8 I)`, `trace_inv = trace((I_norm + 1e-8 I)^{-1})`.

### Primary analysis: within-(sample_size, protocol) Spearman

Each metric is Spearman-correlated with Pdiag's recovery (task error, generator
RMSE, switch error) **within** each (sample_size, protocol) cell over 10 subsets
(seed-averaged), then the signed rho (folded by expected sign) is averaged over
the 12 cells. `+1` = the metric orders subsets correctly.

| Metric (signed rho, mean over cells) | value |
|---|---:|
| condition_number kappa(C) | +0.171 |
| trace_inv_post | +0.136 |
| lambda_min_post | +0.126 |
| lambda_ratio_design | +0.121 |
| lambda_ratio_post | +0.106 |
| log_det_design | +0.105 |
| trace_inv_design | +0.105 |
| lambda_min_design | +0.054 |
| log_det_post | +0.003 |

Supplement (pooled within-size rank, mixes random + qualified protocols):
`condition_number +0.358`, `trace_inv_post +0.207`, `lambda_min_post +0.204`,
`log_det_post -0.054`.

### Interpretation (narrowed)

> Neither simple context conditioning `kappa(C)` nor the tested local
> parameter-space Gauss--Newton spectral summaries reliably rank few-shot subset
> quality. All signed correlations are weak (`|rho| < 0.4` within-cell) and
> sign-inconsistent across sample sizes.

This does **not** rule out function-space identifiability. The information is
computed in the RBF parameter space (72 dims); overlapping RBF bases with
smoothness admit many `theta` with nearly identical `alpha_j(s)`, so `lambda_min`
can be near zero while the generator law itself is stable. A function-space
uncertainty analysis (variance of `alpha_yaw^post`, `alpha_trans^final`, or
`delta-method variance of s_0.5`) is the only remaining check worth attempting;
if it also fails, identifiability-metric development stops here.

---

## Experiment B — TP-GMM SE(2) matched few-shot (STRONG positive result)

### Frozen subset selection (declared before any TP-GMM result was inspected)

```text
N = 5, 10, 20  -> the first five random-protocol subsets (repeat 0..4)
N = 30         -> the single "all" subset
=> 16 subsets x 5 seeds = 80 fits, on the SAME (seed, N, subset) as Pdiag finite.
```

TP-GMM protocol: `frame_mode = "se2"`; component candidates = the full-data set
`(4,6,8,10,12,16,20)` capped at `N-1`; `cv_splits = min(5, N)`;
`cv_n_init = 2`; `final_n_init = 5`.

### Main result (task-trajectory error, mm, mean over subset means)

| N | Pdiag finite | TP-GMM SE(2) | Full operator | Generic RBF |
|---:|---:|---:|---:|---:|
| 5 | **2.71** | 7.58 | 4.11 | 5.91 |
| 10 | **2.67** | 4.43 | 3.94 | 5.27 |
| 20 | **2.43** | 3.78 | 3.28 | 4.02 |
| 30 | **2.39** | 3.46 | 3.07 | 3.63 |

Matched paired comparison (TP-GMM SE(2) minus Pdiag finite): Pdiag wins on
**all** matched (seed, subset) cells at every N (100% win fraction). Mean
difference: N=5 `+4.87`, N=10 `+1.76`, N=20 `+1.35`, N=30 `+1.07` mm.

### Sensitivity checks (both pass)

Expanded component candidates `(2,3,4,6,8,10,12,16,20)` capped at `N-1`
(allowing K=2,3), plus a full-EM variant for N=5 (`cv_n_init=5`,
`final_n_init=10`):

| Configuration | N=5 | N=10 |
|---|---:|---:|
| Pdiag finite | 2.83 | 2.59 |
| TP-GMM main (K>=4) | 7.58 | 4.43 |
| TP-GMM expanded-K (K>=2) | 8.44 | 5.40 |
| TP-GMM full-EM (N=5 only) | 8.47 | -- |

Allowing K=2,3 or restoring the full EM budget does **not** close the gap (it is
slightly worse). The few-shot advantage of Pdiag over the formal rotation-aware
baseline is not an artefact of the mixture-complexity search space or the EM
budget.

### Statistical note

"All 25 matched cells favor Pdiag" (5 seeds x 5 subsets at each N) is a
descriptive result. The design is crossed (`seed x subset`), not 25 independent
replicates; formal inference should use a two-way crossed bootstrap resampling
seed IDs and subset IDs.

---

## Experiment C — Non-oracle progress (PHYSICAL evaluation)

### Frozen parameterizations

```text
oracle       : four solver phases (3..6), reference / upper bound.
time         : normalized step index t / T_ref (naive baseline).
arc          : SE(3) arc-length L(t) / L_ref.
keypoint     : normalized peg-to-socket vertical descent (observable z).
dtw          : DTW on [x, y, ell*yaw] -- optimistic offline upper bound (yaw leak).
dtw-position : DTW on [x, y, z] -- no-yaw-leak variant.
```

Reference quantities (T_ref, L_ref, d_start, d_final) are computed from the 30
mixed TRAINING episodes only. The zero-intervention baseline is the fixed DTW
reference and the nominal mapping for switch offset / event alignment.

### Frozen evaluation (physical timeline)

* Task quotient is bound to the **physical key-clear event** `t_clear` of each
  trajectory (first task-span step with `key_clearance_margin > 0`), NOT to
  `s = 0.5`. Yaw residual is quotiented only for `t >= t_clear`.
* Held-out predictions are evaluated at each physical time step via the
  trajectory's own progress `s_m(t_i)`, compared to the actual peg pose `X(t_i)`.
* Switch timing is `dz_switch = z(t_switch) - z(t_clear)` in mm, mapped through
  the nominal trajectory, not a dimensionless `|s - s'|` across coordinates.
* Profile comparison is event-aligned on the nominal physical timeline
  (`rho_alpha_event`, `rmse_alpha_event`).

### Results (mean over 5 seeds)

| param | E_task^phys | dE_task^phys | dz_switch (mm) | rho_alpha_event | rmse_event | E_gen |
|---|---:|---:|---:|---:|---:|---:|
| oracle | 4.60 | 0.00 | -4.27 | 1.000 | 0.000 | 0.218 |
| time | 11.11 | +6.50 | +9.35 | 0.427 | 0.425 | 0.745 |
| arc | 5.50 | +0.90 | +2.97 | 0.845 | 0.257 | 0.328 |
| keypoint | 8.89 | +4.29 | -6.05 | 0.805 | 0.286 | 0.349 |
| dtw | 4.14 | -0.47 | +6.81 | 0.775 | 0.309 | 0.217 |
| dtw-position | 3.19 | -1.42 | -4.47 | 0.946 | 0.135 | 0.130 |

Between-seed SD of `dE_task^phys`: time `6.61`, arc `2.38`, keypoint `2.87`,
dtw `1.62`, dtw-position `1.92`.

Reference-horizon clipping (fraction of the 30 mixed trajectories with
`s_end > 1` / `< 1`): time `0.34 / 0.66`, arc `0.29 / 0.71`,
keypoint `0.41 / 0.59`.

### Interpretation (narrowed and physical)

* **time** is the clear loser: `+6.5` mm OOD cost, unstable switch placement,
  and its progress carries no pose information.
* **keypoint** (vertical descent) places the switch almost exactly at physical
  clearance (`dz_switch ~ -6 mm`, stable across seeds, vs oracle's `-4 mm`), but
  has a higher OOD task error (`+4.3 mm`) because a z-only progress signal loses
  the horizontal x/y alignment motion.
* **arc** tracks the full motion well (low `+0.9 mm` OOD cost) but its switch
  location is unstable (`+44 .. -11 mm` across seeds) because arc-length is
  dominated by translation and compresses the yaw-relevant gate crossing.
* **dtw-position** is the best (`-1.4 mm`, even below oracle) with a stable
  switch, but it is an **offline** optimal-alignment upper bound (full trajectory
  knowledge), not a deployment estimator. The `dtw` variant that uses `yaw` is an
  optimistic bound and must not be read as evidence that alignment alone closes
  the gap.

The phase-dependent generator law does not fundamentally require privileged
phase labels, but performance depends strongly on the progress coordinate:
event/geometric progress (keypoint, dtw-position) localizes the symmetry
transition correctly, while full-motion alignment (arc, dtw-position) is needed
for accurate translation tracking.

---

## Global interpretation boundary

This is a controlled simulation study with one task geometry, one random mixed
design, and a frozen five-seed replication. The three-layer result supported by
the frozen data is:

1. **Pdiag has a real few-shot advantage** over TP-GMM SE(2), Full operator, and
   Generic RBF, robust to TP-GMM mixture-complexity and EM-budget sensitivity.
2. **Simple excitation / GN-spectral criteria do not reliably predict which
   few-shot subsets will be good** (narrow negative result; the tested local
   parameter-space summaries are not the right identifiability quantity).
3. **Phase-dependent generator relevance does not require privileged phase
   labels, but the recovered law depends strongly on the progress coordinate**
   (geometric/event progress localizes the switch; full-motion alignment tracks
   translation).
