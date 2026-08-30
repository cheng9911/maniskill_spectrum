from __future__ import annotations

"""Phase-0 feasibility probe for the planar-push solver (HARD GATE before full
collection).

The preregistered phase0_gate requires the scripted non-prehensive pusher to
reach the target across context extremes. This probe stresses exactly the two
things the spec flagged:

    1. Push reach across in-plane extremes (du/dv at +-0.012 m, diagonal).
    2. Align-phase yaw controllability (heading_push must rotate the block to
       +-30 deg; free_yaw_push must leave the heading at 0).

It also checks the out-of-plane suppression: a full-extreme mixed context
(dw/roll/pitch at max) must NOT change the block's in-plane outcome, because the
block stays on the table. If heading_push yaw alignment is unstable here, the
spec says to switch to guided push (grasp + slide) BEFORE collecting.

Run from the repo root (flat-module import + relative manifest paths):
    python phase_switch_symmetry/probe_planar_push.py
"""

import argparse
import math

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_planar_push_env  # noqa: F401
from collect_phase_switch_rollouts import EpisodeFinished
from collect_planar_push_rollouts import (
    PlanarPushTraceWrapper,
    solve_planar_push,
)
from phase_switch_symmetry_planar_push_env import (
    PUSH_SUCCESS_POS_TOL,
    PUSH_SUCCESS_YAW_TOL,
)

D2R = math.pi / 180.0


def _context(du=0.0, dv=0.0, dw=0.0, roll=0.0, pitch=0.0, yaw=0.0):
    return [du, dv, dw, roll, pitch, yaw]


PROBE_CONTEXTS = [
    ("baseline", _context()),
    ("du_max", _context(du=0.012)),
    ("dv_max", _context(dv=0.012)),
    ("diag_max", _context(du=0.012, dv=0.012)),
    ("yaw_p30", _context(yaw=30.0 * D2R)),
    ("yaw_m30", _context(yaw=-30.0 * D2R)),
    (
        "mixed_all_max",
        _context(du=0.012, dv=0.012, dw=0.012, roll=15.0 * D2R,
                 pitch=15.0 * D2R, yaw=30.0 * D2R),
    ),
]


def run_case(arm, name, causal_delta):
    goal_heading = arm == "heading_push"
    base = gym.make(
        "PlanarPush-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_mode=None,
        robot_init_qpos_noise=0.0,
        orientation=[1.0, 0.0, 0.0, 0.0],
        task_anchor=[0.10, 0.0, 0.01],
        goal_heading=goal_heading,
    )
    env = PlanarPushTraceWrapper(base)
    try:
        np.random.seed(0)
        torch.manual_seed(0)
        env.reset(seed=0, options={"causal_delta": causal_delta})
        env.start_trace()
        stop_reason = "solver_returned"
        try:
            solve_planar_push(env)
        except EpisodeFinished as exc:
            stop_reason = str(exc)  # "terminated" == success-triggered
        except Exception as exc:
            stop_reason = f"exception:{repr(exc)}"
        info = env.unwrapped.evaluate()
        success = bool(np.asarray(info["success"]).reshape(-1)[0])
        pos_err = float(np.asarray(info["obj_to_goal_dist"]).reshape(-1)[0])
        heading_err = float(np.asarray(info["heading_err"]).reshape(-1)[0])
        steps = len(env.trace["actions"])
        return dict(
            arm=arm, name=name, success=success, pos_err=pos_err,
            heading_err=heading_err, steps=steps, stop_reason=stop_reason,
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["heading_push", "free_yaw_push"],
                        default=None, help="default: both arms")
    args = parser.parse_args()
    arms = [args.arm] if args.arm else ["heading_push", "free_yaw_push"]

    results = []
    for arm in arms:
        for name, causal_delta in PROBE_CONTEXTS:
            row = run_case(arm, name, causal_delta)
            results.append(row)
            mark = "OK " if row["success"] else "FAIL"
            print(
                f"[{mark}] {arm:14s} {name:16s} pos_err={row['pos_err']:.4f} "
                f"heading_err={row['heading_err']:.4f} steps={row['steps']:4d} "
                f"({row['stop_reason']})",
                flush=True,
            )

    print("\n=== PHASE-0 SUMMARY ===", flush=True)
    heading_fails = [r for r in results
                     if r["arm"] == "heading_push" and not r["success"]]
    free_fails = [r for r in results
                  if r["arm"] == "free_yaw_push" and not r["success"]]
    # free_yaw: heading must stay ~0 even under yaw/diag contexts
    free_yaw_heading = [r for r in results if r["arm"] == "free_yaw_push"]
    heading_violations = [
        r for r in free_yaw_heading if r["heading_err"] > PUSH_SUCCESS_YAW_TOL
    ]
    print(f"heading_push failures: {len(heading_fails)}/{sum(1 for r in results if r['arm']=='heading_push')}")
    print(f"free_yaw_push failures: {len(free_fails)}/{len(free_yaw_heading)}")
    print(f"free_yaw heading stays ~0 (heading_err <= {PUSH_SUCCESS_YAW_TOL}): "
          f"{len(free_yaw_heading) - len(heading_violations)}/{len(free_yaw_heading)}")

    gate_pass = not heading_fails and not free_fails and not heading_violations
    if gate_pass:
        print("\nPHASE-0 GATE: PASS — safe to proceed to full collection.")
    else:
        print("\nPHASE-0 GATE: FAIL — do NOT start full collection. "
              "Fix the solver (or switch to guided push) and re-run.")
        if heading_fails:
            print("  heading_push failures:", [(r['name'], round(r['heading_err'], 3)) for r in heading_fails])
        if free_fails:
            print("  free_yaw_push failures:", [(r['name'], round(r['pos_err'], 3)) for r in free_fails])
        if heading_violations:
            print("  free_yaw heading violations:", [(r['name'], round(r['heading_err'], 3)) for r in heading_violations])
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
