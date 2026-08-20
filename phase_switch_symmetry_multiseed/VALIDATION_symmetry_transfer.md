# Frozen Results: Symmetry Intervention (Step 3)

This document freezes the fit-and-score results of the symmetry-intervention
design — the Square/keyed peg vs Circular peg contrast that tests whether the
model **discovers task relations rather than memorizing trajectory statistics**.
It follows the frozen execution order (Step 0 geometry → Step 1 honest →
Step 2 placebo → Step 3 benchmark) and the preregistered confirmatory criteria
in `symmetry_transfer_experiment.json`.

## Frozen protocol

```text
arms      : keyed (reuse rotated_Q1, KeyedCircularPhaseSwitch-v1)
            circular_honest  (CircularPhaseSwitch-v1, flat yaw, alpha*=0)
            circular_placebo (CircularPhaseSwitch-v1, fixed 15deg sweep in
                              align phase, uncorrelated with d_axial, alpha*=0)
oracle    : keyed alpha_yaw = 1 in align/enter (phase<2), 0 in unlock/insert
            circular (both) alpha_yaw = 0 everywhere
subsets   : 18 frozen (N in {5,10,20} x {random, qualified} x 3 repeats),
            same indices across arms and seeds
seeds     : 20260818, 20270818, 20280818  (3 execution seeds)
models    : Frame-weighted, Phase scalar GP, TP-GMM additive, TP-GMM SE(2),
            Generic RBF, Full operator, Pdiag pointwise, Pdiag finite
fits      : 1296 = 8 models x 3 arms x 3 seeds x 18 subsets, 0 failures
metrics   : M0 switch, M1 E_alpha, M2 discrimination, M2b symmetry gap G_psi,
            M3 cross-task mismatch (secondary)
```

All nine datasets (3 arms x 3 seeds) are 39/39 usable. The reused keyed arm had
3–4 residual missing mixed cells (stochastic planner failures in the original
rotated-axis collection), recovered with the frozen R_max=15 policy so the
subset manifest applies identically across arms.

Source hashes:

```text
symmetry_transfer_experiment.json   f82b47e867efce126d14c8c2b9e4e3520efe2ee6ed152a33729bb72a213ce757
symmetry_transfer_subsets.json      77237c65f0b8c96fdf65cfe3dd290464b2c21bcbfb242829aea7f925d6206075
benchmark_symmetry_transfer.py      20de3cd31eacd240e68fdc64dd01df318641eae5067c7bdbcbb5dad7215cbe17
phase_switch_baselines.py           13d097a1041838d74d6a4c36378cd04c91569518dc67a9224e4b4341867b2c57
collect_phase_switch_rotated.py     e25dd6a37525f902d74bb995df4b1e6e17b32ed2b8e7376550ef671d5104fd08
phase_switch_symmetry_env.py        4913fbdf91127fb5994b0e2e78ab740e70ac3a203c2e8092422f342a90914bd0
```

## Result 1 — M1: generator identification error (E_alpha, N=20)

E_alpha = mean over progress of `(alpha_yaw(t) - alpha*_yaw(t))^2`. Lower is
better. The oracle is `1` in pre-insertion phases / `0` post for keyed, and `0`
everywhere for both circular arms.

| model                | keyed | honest | placebo |
|----------------------|-------|--------|---------|
| Frame-weighted       | 0.338 | 0.262  | 0.284   |
| Phase scalar GP      | 0.338 | 0.262  | 0.284   |
| TP-GMM additive      | 0.132 | 0.001  | 0.001   |
| TP-GMM SE(2)         | 0.142 | 0.001  | 0.000   |
| Generic RBF          | 0.159 | 0.000  | 0.001   |
| Full operator        | 0.159 | 0.000  | 0.001   |
| Pdiag pointwise      | 0.158 | 0.000  | 0.000   |
| Pdiag finite         | 0.180 | 0.002  | 0.001   |

**Interpretation (frozen).** Two things are decisive, and neither is "Pdiag
finite has the lowest E_alpha" (it does not — on the keyed arm TP-GMM additive
is marginally lower, which is expected because the keyed arm's yaw switch is a
sharp phase boundary and the smooth Pdiag profile transitions ~0.1 late).

1. **On both circular arms, every structured/relational model reads
   `alpha_yaw ~ 0`** (E_alpha ~ 0.000–0.002), recovering the gauge symmetry.
2. **Only the two marginal, non-relational baselines hallucinate a yaw
   response** on the circular arms (Frame-weighted / Phase scalar GP,
   E_alpha ~ 0.26–0.28), because they regress the marginal trajectory instead of
   the context response.

The claim is therefore **not** "better fitting" but "correct generator
relevance under symmetry intervention".

## Result 2 — M2: symmetry discrimination (headline)

Classify a fit as keyed iff `alpha_pre > 0.5`; correct iff the arm is keyed.

| model          | N=5   | N=10  | N=20  |
|----------------|-------|-------|-------|
| Pdiag finite   | 100%  | 100%  | 100%  |
| Pdiag pointwise| 100%  | 100%  | 100%  |
| Full operator  | 100%  | 100%  | 100%  |
| Generic RBF    | 100%  | 100%  | 100%  |
| TP-GMM SE(2)   | 100%  | 100%  | 100%  |
| TP-GMM additive| 96%   | 100%  | 100%  |
| Phase scalar GP| 70%   | 93%   | 96%   |
| Frame-weighted | 70%   | 93%   | 96%   |

**Interpretation (frozen).** Pdiag finite discriminates the keyed from the
circular arms at **100% down to N=5** — i.e., the relation is identified without
large demonstration sets. But we do **not** claim Pdiag *uniquely* identifies
the symmetry: Full operator, Generic RBF, and Pdiag pointwise also reach 100%.
The correct statement is: *even flexible models can identify the relation under
sufficient data, while Pdiag preserves this ability in the few-shot regime with
explicit generator structure*. The two marginal baselines (Frame-weighted,
Phase scalar GP) are the only ones that fall short (70–96%, and never 100%),
consistent with them modeling the marginal trajectory rather than the response.

## Result 3 — M2b: symmetry gap G_psi = alpha_pre(keyed) - alpha_pre(circular)

Ideal `G_psi* = 1` (keyed alpha_pre ~ 1, circular alpha_pre ~ 0). The headline
is the **placebo** gap, where rotation is present but uncorrelated with
`d_axial`.

| model          | G_psi^placebo N=5 | N=10 | N=20 |
|----------------|-------------------|------|------|
| Pdiag finite   | 0.76              | 0.75 | 0.74 |
| Pdiag pointwise| 0.81              | 0.81 | 0.81 |
| Full operator  | 0.80              | 0.81 | 0.81 |
| Generic RBF    | 0.77              | 0.81 | 0.81 |
| TP-GMM SE(2)   | 0.68              | 0.78 | 0.80 |
| TP-GMM additive| 0.63              | 0.78 | 0.79 |
| Phase scalar GP| 0.16              | 0.24 | 0.27 |
| Frame-weighted | 0.16              | 0.24 | 0.27 |

**Interpretation (frozen).** The gap is large and decisive (~0.74–0.81 for the
relational models vs ~0.16–0.27 for the marginal baselines), but it falls short
of the idealized `1.0` because the keyed `alpha_pre` saturates at ~0.75 rather
than 1.0 — the smooth sigmoid gives a soft relevance (the `alpha_max = 1.25`
ceiling is not saturated across the whole pre-insertion window).

The fact that **Full operator's gap (0.81) exceeds Pdiag finite's (0.74) is not
evidence that Full operator is better**: the gap is a *location* statistic, not
a *stability* statistic. Full operator is unconstrained, so on a given subset it
can push `alpha_pre(keyed)` closer to 1 and `alpha_pre(circular)` closer to 0 by
overfitting; it lacks the smoothness and generator prior that Pdiag finite
carries. The stability contrast is quantified in Result 6.

## Result 4 — M0: keyed switch fidelity (Pdiag finite)

```text
switch detected fraction = 1.00  (all N)
switch location          = 0.606 / 0.608 / 0.608  (N=5 / 10 / 20)
```

**Interpretation (frozen).** The transition is detected in 100% of keyed fits,
consistently around progress ~0.61. We do **not** claim it matches the phase
boundary (progress 0.5) exactly: the unlock stage is a physical release, not an
instantaneous switch, and the smooth profile keeps yaw relevance through the
first ~40% of unlock before dropping it. The correct claim is *the model detects
the transition consistently around the physical release stage*, not "at 0.5".

## Result 5 — M3: cross-task mismatch gap (secondary, behavioral)

`Delta = E_traj^cross - E_traj^in`, i.e. how much worse a model fit on arm A
predicts arm B's test curves than arm A's own. For the relational models the gap
is consistently positive (cross-task prediction is always worse), with
`circular_honest <-> circular_placebo` large (~+8.7 mm-equiv). This confirms the
honest and placebo arms are **behaviorally distinct** (the placebo sweeps yaw),
even though both carry `alpha_yaw = 0` — the behavioral distinctness of M3 plus
the zero-relevance reading of M1 together show the model is not just memorizing
trajectory statistics.

## Result 6 — Few-shot reliability (variance, added post-hoc)

A reviewer's expected objection to Result 2–3 is: *"if Full operator reaches the
same identification, why do we need Pdiag?"* The answer is few-shot reliability,
not accuracy. Over the 18 (seed x subset) cells at N=5 on the circular-honest
arm (the hardest case, where there is no yaw signal to latch onto):

| metric (circular_honest, N=5) | Full operator | Pdiag finite |
|-------------------------------|---------------|--------------|
| E_alpha total std             | 0.253         | 0.011        |
| E_alpha seed std              | 0.087         | 0.006        |
| E_alpha subset std            | 0.138         | 0.009        |
| E_alpha worst case (max)      | 1.106         | 0.045        |
| hallucination rate (E_alpha > 0.10) | 1/18 (5.6%) | 0/18    |

Full operator catastrophically overfits on one unlucky (seed, subset) draw —
fitting a full spurious yaw operator (E_alpha = 1.11 where the oracle is 0) —
while Pdiag finite's generator prior caps the worst case at 0.045. This is the
structural-prior benefit in the few-shot regime: the unconstrained 3x3 operator
has a fat overfitting tail, the generator-regularized model does not. On the
placebo arm both are already stable (the fixed sweep is absorbed into the
nominal), so the reliability contrast is cleanest on the honest arm.

## Verdict (frozen)

**The core claim is confirmed.** The decisive result is the placebo arm:
rotation is present in the marginal trajectory (a fixed 15° align-phase sweep)
but carries zero correlation with `d_axial`, and the relational models read
`alpha_yaw ~ 0` (E_alpha ~ 0.001) exactly as they do on the honest arm. The
model therefore reads the **intervention-response relation**, not the marginal
trajectory. Combined with the 100% discrimination down to N=5 and the
few-shot reliability gap, this closes the symmetry-intervention link of the
chain: generator identification → frame invariance → symmetry intervention →
relation-vs-statistics.
