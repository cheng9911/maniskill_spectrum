from __future__ import annotations

"""Step 0: circular geometry validation.

Confirms two properties before any full collection, at the frozen Q1 anchor with
the honest (and optionally placebo) yaw schedule:

  1. nominal insertion succeeds (zero-intervention episode reaches the goal
     through all four semantic phases);
  2. the peg's world yaw is (near-)invariant to the isolated yaw intervention
     d_axial, i.e. the align-phase yaw response slope ~ 0 (gauge symmetry).

This is the geometry check that alpha*_yaw^circular ~ 0 holds empirically before
freezing the full 3-seed x 39-condition collection.
"""

import argparse
import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from transforms3d.euler import quat2euler

import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_env  # noqa: F401
from collect_phase_switch_rollouts import EpisodeFinished, PhaseSwitchTraceWrapper
from collect_phase_switch_rotated import solve_circular
from mani_skill.agents.robots.panda import Panda

Panda.gripper_stiffness = 2.5e3
Panda.gripper_force_limit = 150.0

QUAT_Q1 = [1.0, 0.0, 0.0, 0.0]
ANCHOR = np.array([-0.35, 0.10, 0.08])


def _peg_yaw(states):
    yaws = [quat2euler(s["peg_pose"][3:])[2] for s in states]
    return np.unwrap(yaws)


def run_one(seed, causal_delta, yaw_mode):
    env = gym.make(
        "CircularPhaseSwitch-v1",
        num_envs=1, obs_mode="state_dict", control_mode="pd_joint_pos",
        reward_mode="sparse", sim_backend="physx_cpu", render_mode=None,
        robot_init_qpos_noise=0.01, orientation=QUAT_Q1, task_anchor=ANCHOR,
    )
    w = PhaseSwitchTraceWrapper(env)
    np.random.seed(seed)
    torch.manual_seed(seed)
    w.reset(seed=seed, options={"causal_delta": causal_delta})
    w.start_trace()
    reachable = False
    error = None
    try:
        solve_circular(w, yaw_mode=yaw_mode)
        reachable = True
    except EpisodeFinished as exc:
        reachable = bool(exc.terminated)
    except Exception as exc:
        error = repr(exc)
        reachable = False
    states = w.trace["states"]
    phases = np.asarray([int(s["solver_phase"]) for s in states])
    yaw = _peg_yaw(states)
    phase_yaw = {}
    for code in (3, 4, 5, 6):
        idx = np.flatnonzero(phases == code)
        phase_yaw[code] = float(yaw[idx].mean()) if len(idx) else float("nan")
    final_dist = float(states[-1]["obj_to_goal_dist"])
    success = bool(states[-1]["success"])
    phase_set = sorted(set(int(x) for x in phases))
    env.close()
    return dict(
        reachable=reachable, success=success, final_dist=final_dist,
        phases=phase_set, error=error,
        align_yaw=phase_yaw[3], enter_yaw=phase_yaw[4],
        unlock_yaw=phase_yaw[5], insert_yaw=phase_yaw[6],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaw-mode", choices=["honest", "placebo"], default="honest")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--out", type=Path,
        default=Path("phase_switch_symmetry_rollouts_rotated/circular_geometry.csv"),
    )
    args = parser.parse_args()

    conditions = [
        ("zero", [0.0, 0.0, 0.0]),
        ("yaw-30", [0.0, 0.0, np.deg2rad(-30.0)]),
        ("yaw-15", [0.0, 0.0, np.deg2rad(-15.0)]),
        ("yaw+15", [0.0, 0.0, np.deg2rad(15.0)]),
        ("yaw+30", [0.0, 0.0, np.deg2rad(30.0)]),
        ("trans-x", [-0.015, 0.0, 0.0]),
        ("trans+y", [0.0, 0.015, 0.0]),
    ]
    if args.smoke:
        conditions = conditions[:1]

    rows = []
    for label, delta in conditions:
        r = run_one(args.seed, delta, args.yaw_mode)
        r.update(
            label=label, d_axial=float(delta[2]),
            d_du=float(delta[0]), d_dv=float(delta[1]),
        )
        rows.append(r)
        print(
            f"{label:8s} reachable={r['reachable']} success={r['success']} "
            f"final_dist={r['final_dist']:.4f} phases={r['phases']} "
            f"align_yaw={r['align_yaw']:.4f} insert_yaw={r['insert_yaw']:.4f} "
            f"err={r['error']}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    yaw_rows = df[df.label.str.startswith("yaw") | (df.label == "zero")]
    if len(yaw_rows) >= 2:
        slope = np.polyfit(yaw_rows.d_axial, yaw_rows.align_yaw, 1)[0]
    else:
        slope = float("nan")
    print("\n=== yaw response (align phase) ===")
    print(f"  slope d(align_yaw)/d(d_axial) = {slope:.4f}  (target ~ 0 for circular)")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
