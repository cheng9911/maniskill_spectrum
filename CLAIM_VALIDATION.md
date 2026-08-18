# Generator-Selective Relational Transfer: Current Validation

## What Was Tested

This validation uses the files currently available in `maniskill_spectrum/`:

- `PlugChargerCausal-v1` controlled reset/state datasets
- PlugCharger causal waypoint response reconstructed from official solver geometry
- Downloaded ManiSkill `StackCube-v1` and `PushT-v1` demonstrations

It does **not** yet include controlled executed trajectories for
`PlugChargerCausal-v1`, because `mplib.Planner(...)` segfaults during planner
initialization on this machine.

## Evidence 1: Clean Causal Intervention Layer

File:

```text
plug_charger_causal/VALIDATION.md
```

Result:

- `charger_pose_max_abs_delta = 0.0`
- `tcp_pose_max_abs_delta = 0.0`
- receptacle translation error: `3.28e-09 m`
- receptacle yaw error: `1.87e-08 rad`
- isolated grid: 13 samples
- mixed training set: 50 samples

Interpretation:

The reset-level `do(Δx)`, `do(Δy)`, and `do(Δyaw)` interventions are clean. The
socket/receptacle and goal change as requested, while charger and TCP initial
poses stay fixed.

This supports the data-generation premise, not the full trajectory-level method.

## Evidence 2: PlugCharger Phase Response

File:

```text
plug_charger_causal/WAYPOINT_RESPONSE.md
```

Approximate response matrices use rows `[x, y, yaw]` and columns
`[Δx, Δy, Δyaw]`.

Approach:

```text
P ≈ 0
```

Pre-insert:

```text
diag(P) ≈ [0.998, -0.997, -0.999]
```

Insert:

```text
diag(P) ≈ [0.998, -0.997, -0.999]
```

Interpretation:

Socket perturbations do not affect the initial/approach TCP pose, but they do
affect pre-insert and insert targets. Yaw is propagated in late phases.

This supports phase-dependent relevance. It does **not** by itself defeat scalar
frame relevance `P=w(s)I`, because x/y/yaw all become similarly relevant in the
late PlugCharger waypoints.

## Evidence 3: StackCube vs PushT Yaw Relevance

File:

```text
STACK_PUSHT_YAW_RELEVANCE.md
```

StackCube:

- episodes: 1000
- final xy error mean: `0.00477 m`
- final xy error p95: `0.01036 m`
- cubeA-cubeB relative yaw median: `87.68 deg`
- cubeA-cubeB relative yaw p95: `168.75 deg`
- yaw within 10 deg: `5.8%`
- cubeA/cubeB yaw circular correlation: `0.0199`

PushT:

- episodes: 719
- final xy error mean: `0.00412 m`
- final xy error p95: `0.00709 m`
- Tee-goal yaw error median: `0.90 deg`
- Tee-goal yaw error p95: `2.80 deg`
- yaw within 10 deg: `100%`

Interpretation:

StackCube success tightly constrains position but not yaw. PushT success tightly
constrains both position and yaw.

This supports task-conditioned generator relevance:

```text
StackCube yaw: weak / suppressed
PushT yaw: relevant / propagated
```

This is observational task-constraint evidence, not a controlled intervention
test.

## Current Verdict

The broad thesis is partially supported:

```text
Manipulation skills should not treat a task frame as an indivisible 6-DoF
condition. Individual generators can be relevant or irrelevant depending on task
and phase.
```

What is already supported:

- clean single-generator intervention data generation
- phase-dependent socket relevance in PlugCharger geometry
- task-dependent yaw relevance from StackCube vs PushT demonstrations

What is not yet proven:

- learned `P_r(s)` beats scalar frame relevance `w(s)I`
- learned `P_r(s)` beats implicit conditional models such as KMP/TP-GMM/MSG
- controlled executed PlugCharger trajectories estimate a real trajectory-level
  `P_r(s)`

Most defensible current claim:

```text
The available data supports the phenomenon and benchmark design, but not yet the
full method contribution.
```

Next required experiment:

Generate successful controlled trajectories for:

```text
PlugChargerCausal-v1
StackCube-Causal yaw/xy
PushT-Causal yaw/xy
```

Then compare:

```text
P=I
P=w(s)I
P=diag(alpha_x, alpha_y, alpha_yaw)
generic conditional regressor
```

on isolated counterfactual perturbations.
