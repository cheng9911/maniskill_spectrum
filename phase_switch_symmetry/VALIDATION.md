# Phase-Switch Pilot Validation

## Experimental object

This pilot tests one axial-yaw generator in one task whose physical symmetry
changes during execution. A rectangular key must follow socket yaw while
crossing a keyed gate. After the key clears the gate, a circular shaft and bore
admit axial rotation and the target returns to nominal yaw. Socket translation
remains task-relevant throughout.

The rollout path is executed with ManiSkill `physx_cpu + pd_joint_pos`. Every
controller action is applied through `env.step(action)`. The H5 records the
actual post-step peg, TCP, socket and target poses, robot qpos/qvel, action,
pairwise peg-socket contact force, key-clearance margin and solver phase.

## Dataset

- File: `phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5`
- Intervention conditions: 39 (5 isolated yaw, 4 isolated translation, 30 mixed)
- Stored episodes: 49
- Successful complete episodes: 39
- Successful complete mixed episodes used for fitting: 30
- Retained failed attempts: 10
- Episodes with nonzero peg-socket contact: 3 (2 successful, 1 truncated failure)
- Socket drift: below the strict 1e-6 m threshold
- Post-terminal or post-truncation actions: 0

Retries are not discarded. Each episode stores `condition_id`, `attempt_id`,
`episode_seed`, `stop_reason` and `solver_error`; root metadata stores package
versions, geometry constants and source hashes. For isolated evaluation, the
single successful complete attempt for each isolated condition is used. For
training, all 30 successful complete mixed conditions are used. Strict mode
checks that all 39 conditions have exactly one usable success, attempts are
contiguous, and no mixed intervention equals an isolated intervention.

## Falsifiable checks

All thresholds are implemented by `analyze_phase_switch_rollouts.py --strict`.

| Check | Criterion | Result |
|---|---:|---:|
| Translation remains relevant | final isolated ratio >= 0.8 | 0.9872 |
| Yaw is relevant immediately before key clearance | isolated ratio >= 0.8 | 0.9997 |
| Yaw is suppressed after unlock | unlock and final ratio <= 0.25 | 0.0028 / 0.0022 |
| Generator model beats scalar | paired improvement 95% CI > 0 | [4.711, 7.202] mm-equiv |
| True contact exists in usable data | any pairwise force > 1e-3 N | 2 successful episodes |
| Socket is kinematic | drift < 1e-6 m | pass |
| Mixed excitation is identified | rank = 3 | pass |
| Collection obeys episode boundaries | zero post-done actions | pass |
| Data split is auditable | coverage, attempts, disjointness | pass |

The learned final response is approximately
`diag(1.0024, 1.0000, -0.00031)`. A metric-matched scalar baseline fitted in
`[x, y, 0.03 yaw]` space obtains `w = 0.5681`. On held-out isolated endpoint
responses, the generator diagonal reduces mean task error from 6.554 to 0.639
mm-equivalent, a 90.2% reduction.

## Independent geometry counterfactual

The nominal solver deliberately uses the task-consistent yaw law. To verify
that the result is supported by physical geometry rather than planner targets
alone, a separate H5 commands matched and mismatched pre-clear yaw, then
commands arbitrary post-clear yaw:

- File: `phase_switch_symmetry_rollouts/keyed_circular_geometry_probes_v2.h5`
- Matched ±30 degree entry clears the gate by +8.0 and +6.4 mm.
- Mismatched entry remains blocked at approximately -28 mm clearance and
  generates 48.1 and 49.7 N peg-socket contact.
- After clearance, -30 and +30 degree peg yaw both insert successfully and end
  at -30.00 and +30.02 degrees.
- All 6 geometry checks pass with zero post-terminal actions.

## Interpretation boundary

This is a controlled simulation pilot, not yet a full generalization claim. The
analysis uses task-space peg motion, not raw TCP orientation, because robot IK
branches are realization choices. Phase profiles use within-phase normalized
progress and successful mixed trajectories; failed trajectories remain in the
dataset and attrition is reported, but failures are excluded from fitting. The
current evidence is one random mixed design and one task geometry. Cross-seed
replication, phase alignment without solver labels, rotated-axis and 6-DoF
experiments remain future validation. The earlier non-v2 pilot contains
post-terminal refinement and is retained under
`phase_switch_symmetry_rollouts/provisional_v1/` only for debugging provenance;
it must not be used for reported results.
