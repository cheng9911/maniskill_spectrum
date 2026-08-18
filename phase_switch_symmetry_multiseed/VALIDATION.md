# Fixed-Context Multi-Seed and Few-Shot Validation

## Frozen design

The experiment uses five preregistered execution seeds:

```text
20260818, 20270818, 20280818, 20290818, 20300818
```

All seeds execute the same 30 mixed training contexts and the same nine
isolated contexts, including the zero-response baseline. The context manifest
SHA-256 is:

```text
3751db801cf49ce698c9b5a74ce36cf0b39897d967ddf2959846a3965b8bd653
```

The Pdiag hyperparameters and the following criteria were frozen before the
new rollouts were collected:

```text
g_translation_final > 0.90
g_yaw_preclear > 0.90
abs(g_yaw_final) < 0.10
abs(s_0.5_model - s_0.5_empirical) < 0.03
E_task(Pdiag) < E_task(frame scalar)
```

## Physics rollout integrity

All five strict physics analyses pass. Each dataset contains one usable rollout
for every fixed condition and retains every failed planner attempt.

| Seed | Episodes | Usable | Failed attempts | Usable contact episodes |
|---:|---:|---:|---:|---:|
| 20260818 | 43 | 39 | 4 | 4 |
| 20270818 | 44 | 39 | 5 | 6 |
| 20280818 | 43 | 39 | 4 | 4 |
| 20290818 | 46 | 39 | 7 | 3 |
| 20300818 | 47 | 39 | 8 | 4 |

The generated datasets are under `phase_switch_symmetry_multiseed/rollouts/`.
Their sizes are approximately 10-11 MB each. Dataset hashes and all individual
physics checks are stored in `results/multiseed_summary.json`.

## Five independent fits

| Seed | Final translation | Pre-clear yaw | Final yaw | Switch error | Scalar task error | Pdiag task error |
|---:|---:|---:|---:|---:|---:|---:|
| 20260818 | 0.9847 | 0.9983 | 0.0234 | 0.00961 | 4.3326 | 2.1360 |
| 20270818 | 0.9972 | 0.9966 | 0.0273 | 0.00669 | 3.3965 | 2.1908 |
| 20280818 | 0.9875 | 0.9985 | 0.0250 | 0.00622 | 5.2263 | 3.4594 |
| 20290818 | 0.9821 | 0.9959 | 0.0251 | 0.00156 | 4.7578 | 2.5222 |
| 20300818 | 0.9797 | 0.9999 | 0.0283 | 0.00306 | 3.0027 | 1.6195 |

All frozen criteria pass in 5/5 seeds. The mean scalar-minus-Pdiag task-error
improvement is `1.7576 +/- 0.4653 mm-equiv` across seeds. The hierarchical
seed-then-condition bootstrap interval is `[1.2964, 2.2295] mm-equiv`.

The endpoint mechanism remains translation-specific. Aggregated over seeds,
translation-only endpoint error changes from `6.3886` for scalar weighting to
`0.5543 mm-equiv` for Pdiag. Yaw-only quotient endpoint error is already small
for both methods: `0.2053` versus `0.2168 mm-equiv`.

The formal rotation-aware TP-GMM SE(2) baseline converged with a non-boundary
component selection in every seed. Pdiag has lower mean task-trajectory error
in all five seeds, by `0.5650-0.8828 mm-equiv`. Generic RBF and the full operator
remain competitive and are not claimed to be uniformly dominated.

## Few-shot protocol

The subset manifest contains 121 subsets and was frozen before fitting:

```text
N = 3, 5, 8, 10, 15, 20, 30
10 random and 10 excitation-qualified subsets for N < 30
one common all-data subset for N = 30
```

Each subset is fitted independently on every execution seed with Pdiag finite,
the full operator, and Generic RBF. This gives 1,815 model fits. No fit raised an
exception and no subset was removed.

Mean task-trajectory errors over the five seed-level means are:

| Protocol | N | Pdiag finite | Full operator | Generic RBF |
|---|---:|---:|---:|---:|
| Random | 3 | **3.265** | 9.498 | 24.121 |
| Random | 5 | **2.673** | 3.891 | 5.297 |
| Random | 10 | **2.610** | 3.723 | 5.039 |
| Random | 30 | **2.386** | 3.075 | 3.630 |
| Qualified | 3 | **3.335** | 9.144 | 28.472 |
| Qualified | 5 | **2.985** | 5.772 | 7.770 |
| Qualified | 10 | **2.572** | 3.566 | 4.420 |
| Qualified | 30 | **2.386** | 3.075 | 3.630 |

At `N=3`, Pdiag beats both competitors on all 50 matched seed-subset pairs.
The hierarchical intervals for competitor-minus-Pdiag error are
`[5.44, 7.32]` against Full and `[19.13, 22.71]` against Generic RBF for random
subsets. The qualified intervals are `[5.21, 6.68]` and `[22.15, 28.18]`.

Pdiag detects the phase switch in 100% of subsets at every sample size. Its
generator-law pass rate is 88% for random `N=3`, 92% for qualified `N=3`, and
100% for both protocols at every `N >= 5`. At `N=3`, Full detects the switch in
70-80% of subsets but passes the complete generator law in 0%; Generic RBF
detects it in 20-30% and passes the law in 0-10%.

The Pdiag optimization audit deterministically reproduces all 605 saved
profiles. It reports convergence in 604/605 fits. The retained exception is
seed `20280818`, qualified subset 14 at `N=3`; it reaches the 500-evaluation
limit with optimality `4.52e-8` and task error `7.0400`. It is included in all
reported averages and paired comparisons.

## Interpretation boundaries

The data support a low-sample identification advantage for the structured
diagonal finite-action law, especially at `N=3`. They do not show per-subset
dominance at larger sample sizes: Pdiag paired win fractions against Full/RBF
are generally 52-86% for `N >= 5`, and only 60% at the five full-data fits.

The simple excitation qualification is not sufficient to identify good
subsets. Qualified subsets do not consistently outperform random subsets, and
within-size condition-number correlations with error are inconsistent. At
`N=3`, all context matrices have generator rank three, but the augmented
intercept-plus-context matrix cannot have rank four. This makes the affine
baselines underidentified and is part of the intended few-shot stress test.

The repeated few-shot sweep excludes TP-GMM because repeating its nested
episode-grouped component CV and multi-start EM for 605 subsets is a separate
computational experiment. TP-GMM SE(2) is included in the complete five-seed,
full-data benchmark. No claim should imply that Pdiag is the only anisotropic
model or that it is uniformly the most accurate model.

## Machine-readable artifacts

```text
results/multiseed_summary.json
results/multiseed_seed_summary.csv
results/multiseed_model_summary.csv
results/phase_switch_multiseed.{png,pdf}
fewshot_results/fewshot_summary.json
fewshot_results/fewshot_aggregate_summary.csv
fewshot_results/fewshot_paired_comparisons.csv
fewshot_results/fewshot_law_recovery_rates.csv
fewshot_results/fewshot_condition_correlations.csv
fewshot_results/fewshot_pdiag_optimization_audit.csv
fewshot_results/phase_switch_fewshot.{png,pdf}
```

`validate_phase_switch_multiseed.py` and
`validate_phase_switch_fewshot.py` independently verify the frozen manifests,
source/data hashes, exact context reuse, table cardinalities, failed-fit
retention, audit reproduction, preregistered criteria, and non-empty figures.
