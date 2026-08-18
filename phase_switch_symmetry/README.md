# Keyed-to-Circular Phase-Switch Insertion

This controlled ManiSkill task makes axial yaw task-relevant at a rectangular
entry gate and task-irrelevant after the key clears into a circular bore. The
socket intervention is `causal_delta = [dx_world, dy_world, dyaw_z]`.

The compound peg has a short rectangular key and a circular shaft. The solver
executes every path with `env.step(action)` under `physx_cpu + pd_joint_pos`:

1. `align_keyed`: align position and axial yaw with the keyed gate.
2. `enter_key`: pass the rectangular key through the gate.
3. `unlock_yaw`: keep position but return to nominal yaw after physical key clearance.
4. `circular_insert`: insert farther while preserving nominal axial yaw.

All failures are retained. A condition can be retried to mitigate stochastic
planner failures; `condition_id`, `attempt_id`, `episode_seed`, `stop_reason`,
package versions and source hashes preserve that provenance. Collection stops
on the first terminal or truncated step, after recording that PhysX state.

## Collect

```bash
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/collect_phase_switch_rollouts.py \
  --output phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5 \
  --mixed-samples 30 \
  --retries-per-condition 3
```

## Validate and analyze

```bash
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/analyze_phase_switch_rollouts.py \
  phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5 \
  --strict
```

## Validate the geometry independently of the nominal planner targets

```bash
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/collect_phase_switch_geometry_probes.py \
  --output phase_switch_symmetry_rollouts/keyed_circular_geometry_probes_v2.h5

conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/validate_phase_switch_geometry_probes.py \
  phase_switch_symmetry_rollouts/keyed_circular_geometry_probes_v2.h5 \
  --strict
```

The probes command mismatched yaw before key clearance and arbitrary yaw after
clearance. A valid geometry blocks both mismatched pre-clear attempts by
contact, while accepting both post-clear axial-yaw targets.

The strict checks require translation propagation, keyed yaw propagation,
circular yaw suppression, a held-out advantage over scalar frame weighting,
real pairwise contact in at least one episode, a fixed kinematic socket, and
full-rank mixed generator excitation, complete condition coverage, contiguous
attempt retention, disjoint train/holdout interventions and zero post-terminal
actions.

## Strong baseline benchmark

The benchmark fits all models on the same 30 successful mixed interventions and
evaluates them on the same eight nonzero isolated interventions. The zero
intervention is used only to estimate the empirical isolated response profile.

```bash
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/benchmark_phase_switch_baselines.py \
  phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5 \
  --output-root phase_switch_symmetry_baselines

conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/validate_phase_switch_baselines.py \
  phase_switch_symmetry_baselines \
  --strict
```

The compared models are frame-weighted `w(s)I`, an oracle-favorable `Phase
scalar GP`, additive and rotation-aware SE(2) TP-GMM variants, a generic
phase-local RBF regressor, a dense full operator, pointwise diagonal ablation,
and the final smooth finite-action `Pdiag`. `Phase scalar GP` is explicitly not
the TPGP algorithm; it smooths the best training-only shared scalar and tests
whether phase dependence alone fixes frame-level coarseness.

Both TP-GMM variants use shared mixture responsibilities with frame-specific
conditional Gaussian emissions. Their EM fitting, grouped-CV scoring, and
component-wise prediction use the same frame-product likelihood; the SE(2)
variant additionally rotates socket-local translation coordinates.

The final method uses 24 global RBFs by default,
`alpha(s)=1.25*sigmoid(Phi(s) theta)`, second-difference regularization, and the
finite action `C0 Exp(P(s) Log(C0^-1 C)) C0^-1 X0`. Its hyperparameters are
exposed as `--pdiag-*` command-line arguments and stored in the result JSON.
The nominal curve and nominal socket frame are estimated from mixed training
trajectories only; the zero intervention remains excluded from fitting.

Outputs include machine-readable split provenance and source/data hashes,
per-model profiles, per-episode held-out errors, generator-stratified metrics,
exact paired differences, a summary table, serialized arrays, and the
publication figure in PNG and PDF formats. Both TP-GMM component counts are
selected by episode-grouped training-only cross-validation. The task quotient metric
ignores axial yaw only after physical key clearance while retaining translation
errors in every phase.

The frozen numerical interpretation and limitations are recorded in
`phase_switch_symmetry_baselines/BASELINE_VALIDATION.md`.

## Fixed-context five-seed replication

The replication freezes the original 30 mixed and 9 isolated contexts before
collection. It changes only the simulator/planner seed and a preregistered
`0.01 rad` robot initialization perturbation. Failed attempts remain in each
HDF5 file, while one usable rollout per condition is selected by the same rule
for every seed.

```bash
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/prepare_phase_switch_multiseed.py

conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/collect_phase_switch_multiseed.py \
  --skip-complete
```

Each `phase_switch_symmetry_multiseed/rollouts/seed_<seed>.h5` is analyzed with
`analyze_phase_switch_rollouts.py --strict`, then benchmarked independently by
passing it to `benchmark_phase_switch_baselines.py`. Aggregate and validate all
five completed seed fits with:

```bash
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/analyze_phase_switch_multiseed.py

conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/validate_phase_switch_multiseed.py
```

The frozen criteria require every seed to preserve final translation, propagate
pre-clear yaw, suppress final yaw, recover switch timing within `0.03` progress,
and improve task error over scalar frame weighting. The complete numerical
record is in `phase_switch_symmetry_multiseed/VALIDATION.md`.

## Few-shot and identifiability sweep

The few-shot subset manifest is frozen before fitting and reused verbatim for
all five execution seeds. Random subsets are never dropped. Qualified subsets
require rank three and scaled context condition number below 10. Sample sizes
are `3, 5, 8, 10, 15, 20, 30`, with ten frozen subsets per protocol except for
the common full-data endpoint.

```bash
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/prepare_phase_switch_fewshot.py

conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/run_phase_switch_fewshot.py \
  --seed 20260818

conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/audit_phase_switch_fewshot_pdiag.py \
  --seed 20260818

conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/analyze_phase_switch_fewshot.py

conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/validate_phase_switch_fewshot.py
```

Repeat the two seed-specific commands for every seed listed in
`phase_switch_symmetry_multiseed/experiment.json`. The independent audit
refits Pdiag, records optimizer convergence, and checks exact reproduction of
the profiles saved by the primary sweep. TP-GMM SE(2) remains part of the
five-seed full-data benchmark; it is not included in the 605-subset repeated
few-shot sweep.
