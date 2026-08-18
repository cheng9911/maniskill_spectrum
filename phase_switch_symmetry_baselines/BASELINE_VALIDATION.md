# Phase-Switch Method-Alignment Validation

## Protocol

- Dataset: `keyed_circular_phase_switch_physics_v2.h5`
- Dataset SHA-256: `32fa4f0702614a38750d93b41b7df821b45d2226a8947a4c6b9bc238c485dcb3`
- Training: 30 successful mixed PhysX rollouts only.
- Held-out test: eight successful nonzero isolated interventions only.
- Zero intervention: empirical response target only; never used for fitting.
- Progress: oracle solver phase, 25 samples in each of four phases.
- Metric: 30 mm/rad; post-clear axial yaw is quotiented out, translation is not.

Intervals in the artifacts are descriptive resampling intervals over eight
fixed interventions, not generalization confidence intervals.

## Method Alignment

`Phase scalar GP` is the renamed oracle-favorable shared-scalar baseline. It
GP-smooths the best training-only `w(s)` and is explicitly not the TPGP
algorithm.

`Pdiag pointwise` retains the old per-progress affine diagonal regression as an
ablation. `Pdiag finite` implements the Methods object:

```text
alpha_j(s) = alpha_max sigmoid(Phi(s)^T theta_j)
X_hat(s) = C0 Exp(diag(alpha(s)) Log(C0^-1 C)) C0^-1 X0(s)
```

The frozen defaults are `alpha_max=1.25`, 24 normalized RBFs, width `0.065`,
second-difference weight `0.1`, and three alternating nominal/profile updates.
`X0(s)` is estimated using mixed training trajectories only by inverse finite
actions and SE(2) pose averaging. The NPZ stores the basis, parameters, nominal
curve, profiles, and predictions; the validator independently reconstructs the
finite actions.

`TP-GMM additive` uses additive response coordinates. Both variants use one
set of latent mixture responsibilities and frame-specific conditional Gaussian
emissions. Training EM, grouped-CV scoring, and prediction all use the same
frame-product likelihood; no fitted world/local cross-covariance is discarded.
`TP-GMM SE(2)` uses the strict socket frame

```text
p_local = R(-yaw_socket) (p_world - p_socket)
yaw_local = yaw_world - yaw_socket
```

and transforms each corresponding component mean and covariance before
Gaussian-product fusion. Both variants select component count by episode-grouped
training-only five-fold product likelihood and use ten final EM initializations.

## Results

| Model | Task trajectory | Task endpoint | Generator RMSE | Final trans | Pre-yaw | Final yaw | Switch error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frame-weighted | 4.765 | 3.262 | 0.354 | 0.568 | 1.004 | 0.568 | no switch |
| Phase scalar GP | 4.764 | 3.261 | 0.354 | 0.568 | 1.004 | 0.568 | no switch |
| TP-GMM additive | 3.755 | 0.294 | 0.177 | 0.998 | 0.945 | 0.0018 | 0.00372 |
| TP-GMM SE(2) | 4.159 | 0.298 | 0.177 | 0.998 | 0.946 | 0.0018 | 0.00345 |
| Generic RBF | 3.316 | 0.251 | 0.196 | 1.001 | 1.000 | -0.0003 | 0.00093 |
| Full operator | 3.304 | 0.251 | 0.196 | 1.001 | 1.000 | -0.0003 | 0.00096 |
| Pdiag pointwise | 3.671 | 0.271 | 0.189 | 1.001 | 1.000 | -0.0002 | 0.00089 |
| Pdiag finite | 3.448 | 0.427 | 0.192 | 0.985 | 1.000 | 0.0258 | 0.00283 |

All errors are mean held-out mm-equivalent values. Switch error measures the
response half-decay relative to the isolated target at `s=0.602761`, not the
physical clearance event.

## Interpretation

The final Methods implementation passes the structural test:

```text
g_translation_final = 0.985
g_yaw_preclear = 0.9999
g_yaw_final = 0.0258
switch error = 0.00283
```

It also lowers task-trajectory error from the pointwise ablation's `3.671` to
`3.448`. Its endpoint error is higher (`0.427` versus `0.271`) but remains
sub-millimetric. The smooth constrained finite method therefore preserves the
claimed law without inheriting the pointwise estimator's exact endpoint fit.

The TP-GMM coordinate cross-check is not negligible. Rotation-aware SE(2)
coordinates change task trajectory error by `+0.404`, select `K=12` instead of
`K=10`, and leave endpoint error nearly unchanged (`+0.004`). Consequently the
formal paper baseline must be `TP-GMM SE(2)`; the additive version is a
response-coordinate ablation, not a standard task-frame implementation.

Generic RBF and Full operator remain slightly better on mean trajectory and
endpoint error than finite Pdiag. These results support the claim that shared
scalar frame relevance is structurally too coarse, but not claims that Pdiag is
the only anisotropic model or uniformly most accurate. Sample efficiency,
stability, and identifiability remain the required differentiating tests.

## Verification and Scope

The strict validator checks source/data hashes, exact split and signed isolated
grid, grouped-CV optima, finite arrays, stratified and paired summaries,
RBF-sigmoid constraints, optimizer convergence, an independently rebuilt RBF
basis, and an independent SE(2) reconstruction. Eight unit tests cover Exp/Log, finite-action recovery,
rotation-aware task-frame transforms, GMR conditioning, and structured
operators.

This is still a single-seed, oracle-phase result. The next new-data experiment
is fixed-context five-seed replication, followed by phase-switch few-shot
identification.
