from __future__ import annotations

"""Coarse workspace feasibility map for the rotated-axis anchor.

For a fixed orientation Q, scan a grid of task_anchor (world workspace placement
of the socket nominal center) and run the NOMINAL (zero-intervention) solver
path, recording reachability and the failing waypoint phase. This is Stage 1 of
the two-stage search: only anchors that pass the nominal check for every
orientation should advance to the strict 7-context screening.

The anchor is chosen for IK/collision feasibility only; no Pdiag result is ever
consulted.
"""

import argparse
import contextlib
import csv
import io
import re
import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien

import gymnasium as gym
import numpy as np

import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_env  # noqa: F401
from collect_phase_switch_rollouts import (
    EpisodeFinished,
    PHASES,
    PhaseSwitchTraceWrapper,
)
from collect_phase_switch_rotated import solve_rotated


PHASE_NAME = {code: name for name, code in PHASES.items()}


def check_nominal(orientation, task_anchor, seed):
    base = gym.make(
        "KeyedCircularPhaseSwitch-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_mode=None,
        robot_init_qpos_noise=0.0,
        orientation=orientation,
        task_anchor=task_anchor,
    )
    env = PhaseSwitchTraceWrapper(base)
    env.reset(seed=seed, options={"causal_delta": [0.0, 0.0, 0.0]})
    env.start_trace()
    reachable = False
    phase = -1
    error = ""
    ik_distance = float("nan")
    stderr_buf = io.StringIO()
    with contextlib.redirect_stderr(stderr_buf):
        try:
            solve_rotated(env)
            reachable = True
        except EpisodeFinished as exc:
            reachable = bool(exc.terminated)
            if not reachable:
                error = "truncated"
        except Exception as exc:
            error = repr(exc)[:120]
    phase = int(env.trace["states"][-1]["solver_phase"])
    if reachable:
        ik_distance = 0.0
    else:
        dists = [
            float(m.group(1))
            for m in re.finditer(r"Distance ([0-9.eE+-]+)", stderr_buf.getvalue())
        ]
        if dists:
            ik_distance = min(dists)  # closest the IK solver got to the target
    env.close()
    return reachable, PHASE_NAME.get(phase, "?"), error, ik_distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orientation-quat", nargs=4, type=float, required=True,
        help="Global task rotation as a unit quaternion [w, x, y, z].",
    )
    parser.add_argument("--x-range", nargs=3, type=float, default=[-0.02, 0.18, 0.05])
    parser.add_argument("--y-range", nargs=3, type=float, default=[-0.15, 0.15, 0.05])
    parser.add_argument("--z-range", nargs=3, type=float, default=[0.18, 0.38, 0.05])
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--out", type=str, default="phase_switch_symmetry_rollouts_rotated/anchor_search.csv")
    args = parser.parse_args()

    orientation = np.asarray(args.orientation_quat, dtype=np.float64)
    orientation = orientation / np.linalg.norm(orientation)
    xs = np.arange(*args.x_range)
    ys = np.arange(*args.y_range)
    zs = np.arange(*args.z_range)

    rows = []
    total = len(xs) * len(ys) * len(zs)
    idx = 0
    for x in xs:
        for y in ys:
            for z in zs:
                idx += 1
                anchor = np.array([x, y, z])
                reachable, phase, error, ik_distance = check_nominal(
                    orientation, anchor, args.seed
                )
                rows.append(
                    {
                        "x": round(float(x), 4),
                        "y": round(float(y), 4),
                        "z": round(float(z), 4),
                        "reachable": reachable,
                        "last_phase": phase,
                        "ik_distance": ik_distance,
                        "error": error,
                    }
                )
                flag = "OK " if reachable else f"FAIL@{phase}"
                print(f"[{idx}/{total}] anchor=({x:.2f},{y:.2f},{z:.2f}) {flag} "
                      f"ik={ik_distance:.3f}", flush=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    n_ok = sum(r["reachable"] for r in rows)
    print(f"\nfeasible: {n_ok}/{total}")
    print("saved:", args.out)


if __name__ == "__main__":
    main()
