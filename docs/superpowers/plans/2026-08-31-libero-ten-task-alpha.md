# LIBERO/robosuite Ten-Task Controlled Alpha Supplement Plan

Date: 2026-08-31

## Goal

Add a low-cost controlled LIBERO/robosuite supplement with up to 10 task
instances, while keeping the alpha evidence causal: each task must regenerate
trajectories under known `[du,dv,dw,roll,pitch,yaw]` context interventions rather
than reuse uncontrolled demos.

## Reporting Discipline

Use two labels:

- `task instance`: a concrete LIBERO task/env setup.
- `relation family`: the physical relevance structure being tested.

This prevents overclaiming 10 independent mechanisms when several LIBERO tasks
are object or scene variants of the same relation.

## Already Implemented

1. `libero_goal/open_the_middle_drawer_of_the_cabinet`
   - family: prismatic sliding
   - oracle: `[du]=1`, `[dv,dw,roll,pitch,yaw]=0`
   - status: collected and benchmarked across 3 seeds, N in `{8,15,30}`
   - outputs: `phase_switch_symmetry_multiseed/libero_drawer/`

## Candidate 10-Task Set

| id | LIBERO source | task | relation family | controlled oracle sketch | priority |
| --- | --- | --- | --- | --- | --- |
| 1 | `libero_goal` | open middle drawer | prismatic sliding | rail translation tracked; all other generators suppressed | done |
| 2 | `libero_goal` | turn on stove | revolute knob | hinge/knob yaw tracked; translations and out-of-axis rotations suppressed | high |
| 3 | `libero_goal` | push plate to front of stove | planar sliding | one or two in-plane translations tracked; vertical/tilt suppressed | high |
| 4 | `libero_goal` | put bowl on stove | support placement | in-plane placement tracked; vertical/tilt partly phase dependent | medium |
| 5 | `libero_goal` | put bowl on plate | stacking / contact support | target-relative in-plane translation and height tracked; yaw often irrelevant | medium |
| 6 | `libero_goal` | put cream cheese in bowl | container insertion | container-relative translation tracked; yaw/roll/pitch mostly suppressed | medium |
| 7 | `libero_goal` | put wine bottle on rack | rack / slot placement | slot lateral and depth tracked; bottle orientation may be relevant | medium |
| 8 | `libero_goal` | put wine bottle on top of cabinet | upright placement | planar translation and support height tracked; yaw likely weak/irrelevant | medium |
| 9 | `libero_10` | put black bowl in bottom drawer and close it | compound prismatic + placement | object placement plus drawer prismatic close; score subphases separately | stretch |
| 10 | `libero_10` | turn on stove and put moka pot on it | compound revolute + placement | knob yaw plus object support placement; score subphases separately | stretch |

## Implemented Outcome

The executed supplement now includes all 10 task instances above.  The benchmark
output contains seven full SE(3) rows:

- Frame-weighted (SE(3));
- Phase scalar GP (SE(3));
- TP-GMM additive (SE(3));
- TP-GMM SE(3);
- Full operator (SE(3));
- Pdiag pointwise (SE(3));
- Pdiag finite (SE(3)).

The historical repository TP-GMM baseline is also retained on the `[du,dv,yaw]`
projection for continuity with the earlier SE(2) benchmark.

## Implementation Pattern

For each task:

1. Load the real LIBERO/robosuite env and identify relevant body/joint names.
2. Define a task-local frame and a ghost target pose carrying all six context
   interventions.
3. Generate 75 controlled conditions using the same SE(3) context design.
4. Record raw MuJoCo world pose plus task-frame `object_pose`.
5. Freeze rank-6 subsets with N in `{8,15,30}`.
6. Run the expanded SE(3) model suite: Frame-weighted, Phase scalar GP,
   TP-GMM additive, TP-GMM SE(3), Full operator, Pdiag pointwise, and Pdiag finite.
7. Add the historical projected SE(2) TP-GMM only when the relation admits a
   faithful `[du,dv,yaw]` projection.

## Recommendation

For the paper, target 6 strong relation families first:

1. peg-in-hole keyed/circular,
2. planar push,
3. prismatic drawer,
4. revolute stove knob or door,
5. support placement,
6. container/rack placement.

Then add LIBERO task instances up to 10 as supplementary breadth, clearly
grouped by family.
