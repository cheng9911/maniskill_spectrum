from __future__ import annotations

"""Rotated-axis strict reachability screening (with retry budget).

For a candidate task placement (orientation Q + task_anchor), run the 7 extreme
interventions across multiple execution seeds, with a fixed retry budget per
(seed, context). Records first-attempt and eventual success, the failed phase,
and the retry count, so systematic workspace truncation (e.g. d_axial=-30 deg
fails every seed) is distinguished from occasional stochastic RRT failure.

The anchor is chosen for IK/collision feasibility only; no Pdiag result is
consulted.
"""

import argparse
import csv
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

EXTREME_CONTEXTS = [
    ("zero", [0.0, 0.0, 0.0]),
    ("du+", [0.015, 0.0, 0.0]),
    ("du-", [-0.015, 0.0, 0.0]),
    ("dv+", [0.0, 0.015, 0.0]),
    ("dv-", [0.0, -0.015, 0.0]),
    ("dpsi+", [0.0, 0.0, np.deg2rad(30.0)]),
    ("dpsi-", [0.0, 0.0, np.deg2rad(-30.0)]),
]


def screen(orientation, task_anchor, seeds, retries):
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
    results = []
    for name, causal_delta in EXTREME_CONTEXTS:
        for seed in seeds:
            first = False
            eventual = False
            last_phase = -1
            used = 0
            for attempt in range(retries):
                used = attempt + 1
                episode_seed = seed + attempt * 7
                env.reset(seed=episode_seed, options={"causal_delta": causal_delta})
                env.start_trace()
                reachable = False
                try:
                    solve_rotated(env)
                    reachable = True
                except EpisodeFinished as exc:
                    reachable = bool(exc.terminated)
                except Exception:
                    reachable = False
                last_phase = int(env.trace["states"][-1]["solver_phase"])
                if attempt == 0:
                    first = reachable
                if reachable:
                    eventual = True
                    break
            results.append(
                {
                    "context": name,
                    "seed": seed,
                    "first_attempt": first,
                    "eventual": eventual,
                    "retries_used": used,
                    "last_phase": PHASE_NAME.get(last_phase, "?"),
                }
            )
            flag = "OK " if eventual else f"FAIL@{PHASE_NAME.get(last_phase, '?')}"
            print(
                f"  {name:7s} seed={seed} {flag} (first={int(first)}, "
                f"retries={used})",
                flush=True,
            )
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orientation-quat", nargs=4, type=float, required=True,
        help="Global task rotation as a unit quaternion [w, x, y, z].",
    )
    parser.add_argument(
        "--task-anchor", nargs=3, type=float, required=True,
        help="World workspace placement of the socket nominal center [x, y, z].",
    )
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[20260818, 20270818, 20280818])
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--out", type=str,
                        default="phase_switch_symmetry_rollouts_rotated/screening.csv")
    args = parser.parse_args()

    orientation = np.asarray(args.orientation_quat, dtype=np.float64)
    orientation = orientation / np.linalg.norm(orientation)
    task_anchor = np.asarray(args.task_anchor, dtype=np.float64)

    print(f"orientation={orientation} task_anchor={task_anchor} "
          f"seeds={args.seeds} retries={args.retries}")
    results = screen(orientation, task_anchor, args.seeds, args.retries)

    n = len(results)
    n_eventual = sum(r["eventual"] for r in results)
    n_first = sum(r["first_attempt"] for r in results)
    print(f"\nfirst-attempt: {n_first}/{n}   eventual: {n_eventual}/{n}")
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("saved:", args.out)

    # Systematic failure check: any (context) failing across ALL seeds.
    import pandas as pd

    df = pd.DataFrame(results)
    for context in [c[0] for c in EXTREME_CONTEXTS]:
        sub = df[df.context == context]
        if sub.eventual.sum() == 0:
            print(f"  SYSTEMATIC FAILURE: {context} failed every seed")
    if n_eventual != n:
        raise SystemExit(f"strict screening failed: {n_eventual}/{n} eventual success")


if __name__ == "__main__":
    main()
