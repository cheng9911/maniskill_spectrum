from __future__ import annotations

"""Step 0 (SE(3)): geometry validation for the 6-generator supplement.

Confirms two properties before the full 2-arm x 3-seed collection, at the frozen
SE(3) anchor and orientation Q1 (identity):

  1. nominal insertion succeeds (zero-intervention episode reaches the goal
     through all four semantic insertion phases);
  2. the isolated-response diagonal is ~identity: applying a single-generator
     intervention delta_j moves the peg's align-phase pose6 (in world == local
     frame, since Q1 is identity) by ~delta_j in channel j and ~0 elsewhere, so
     the 6x6 response matrix R[k, j] = d_pose6[k] / d_intervention[j] has
     on-diagonal ~1 and off-diagonal ~0.

This is the SE(3) analog of validate_circular_geometry.py; it pins the Euler
convention (Rx Ry Rz, yaw innermost) against the isolated response before the
benchmark's nominal_frame_se3 / oracle_alpha_se3 are trusted.
"""

import argparse
import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from transforms3d.quaternions import quat2mat

import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_env  # noqa: F401
from collect_phase_switch_rollouts import EpisodeFinished, PhaseSwitchTraceWrapper
from collect_phase_switch_rotated import solve_se3
from mani_skill.agents.robots.panda import Panda
from phase_switch_se3_baselines import euler_from_matrix

Panda.gripper_stiffness = 2.5e3
Panda.gripper_force_limit = 150.0

QUAT_Q1 = [1.0, 0.0, 0.0, 0.0]
ANCHOR = np.array([-0.15, 0.00, 0.08])
SE3_GENERATOR_NAMES = ("du", "dv", "dw", "roll", "pitch", "yaw")


def _pose6_from_state(state):
    peg = np.asarray(state["peg_pose"], dtype=np.float64)
    rpy = euler_from_matrix(quat2mat(peg[3:]))
    return np.concatenate([peg[:3], rpy])


def _align_pose6(states):
    """Peg pose6 at the END of the align phase (the staged align goal), not the
    average over the whole approach (which dilutes the response toward lift)."""
    phases = np.asarray([int(s["solver_phase"]) for s in states])
    idx = np.flatnonzero(phases == 3)  # align_keyed
    if not len(idx):
        return None
    return _pose6_from_state(states[idx[-1]])


def run_one(seed, causal_delta, anchor):
    env = gym.make(
        "KeyedCircularPhaseSwitchSE3-v1",
        num_envs=1, obs_mode="state_dict", control_mode="pd_joint_pos",
        reward_mode="sparse", sim_backend="physx_cpu", render_mode=None,
        robot_init_qpos_noise=0.0, orientation=QUAT_Q1, task_anchor=anchor,
    )
    w = PhaseSwitchTraceWrapper(env)
    np.random.seed(seed)
    torch.manual_seed(seed)
    w.reset(seed=seed, options={"causal_delta": causal_delta})
    w.start_trace()
    reachable = False
    error = None
    try:
        solve_se3(w)
        reachable = True
    except EpisodeFinished as exc:
        reachable = bool(exc.terminated)
    except Exception as exc:
        error = repr(exc)
        reachable = False
    states = w.trace["states"]
    phases = np.asarray([int(s["solver_phase"]) for s in states])
    success = bool(states[-1]["success"])
    phase_set = sorted(set(int(x) for x in phases))
    align6 = _align_pose6(states)
    env.close()
    return dict(
        reachable=reachable, success=success, phases=phase_set, error=error,
        align_pose6=align6,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-anchor", nargs=3, type=float, default=ANCHOR.tolist(),
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--out", type=Path,
        default=Path("phase_switch_symmetry_rollouts_se3/se3_geometry.csv"),
    )
    args = parser.parse_args()
    anchor = np.asarray(args.task_anchor, dtype=np.float64)

    DEG = np.deg2rad
    conditions = [
        ("zero", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("du+", [0.015, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("dv+", [0.0, 0.015, 0.0, 0.0, 0.0, 0.0]),
        ("dw+", [0.0, 0.0, 0.015, 0.0, 0.0, 0.0]),
        ("roll+", [0.0, 0.0, 0.0, DEG(15), 0.0, 0.0]),
        ("pitch+", [0.0, 0.0, 0.0, 0.0, DEG(15), 0.0]),
        ("yaw+", [0.0, 0.0, 0.0, 0.0, 0.0, DEG(30)]),
    ]
    if args.smoke:
        conditions = conditions[:1]

    rows = []
    for label, delta in conditions:
        r = run_one(args.seed, delta, anchor)
        r.update(label=label, causal_delta=delta)
        rows.append(r)
        print(
            f"{label:8s} reachable={r['reachable']} success={r['success']} "
            f"phases={r['phases']} err={r['error']}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Isolated-response matrix R[k, j] = d_align_pose6[k] / d_intervention[j].
    zero_row = next(r for r in rows if r["label"] == "zero")
    zero6 = zero_row["align_pose6"]
    R = np.zeros((6, 6), dtype=np.float64)
    for j, (label, delta) in enumerate(conditions[1:]):
        row = next(r for r in rows if r["label"] == label)
        if row["align_pose6"] is None or zero6 is None or abs(delta[j]) < 1e-12:
            continue
        R[:, j] = (row["align_pose6"] - zero6) / delta[j]

    print("\n=== isolated response matrix R (target: identity) ===")
    print("          " + "".join(f"{g:>9s}" for g in SE3_GENERATOR_NAMES))
    for k, g in enumerate(SE3_GENERATOR_NAMES):
        print(f"{g:9s} " + "".join(f"{R[k, j]:9.3f}" for j in range(6)))
    diag = np.diag(R)
    off = R - np.diag(diag)
    print(f"\ndiagonal mean={diag.mean():.3f} (target ~1); "
          f"|off-diagonal| max={np.abs(off).max():.3f} (target ~0)")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
