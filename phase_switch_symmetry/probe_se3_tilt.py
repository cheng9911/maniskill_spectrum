from __future__ import annotations

"""Phase-0 SE(3) feasibility probe: tilt reachability.

Before the full 6-generator experiment, determine whether the Panda can tilt a
grasped peg by +-15 deg (roll/pitch) and insert it into a +-15 deg-tilted keyed
socket, at the frozen anchor. If every tilt cell is eventually reachable across
all seeds we proceed to the full experiment; otherwise we fall back to the
4-generator [du, dv, dw, yaw] design (roll/pitch geometry-locked).

The anchor is the rotated_Q1 placement (-0.35, 0.10, 0.08), chosen for IK
feasibility only; no Pdiag result is consulted.
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
from collect_phase_switch_rotated import solve_se3


PHASE_NAME = {code: name for name, code in PHASES.items()}

DEG = np.deg2rad


def _c(*angles_deg):
    """6-vector [du, dv, dw, roll, pitch, yaw] from degrees for the last three."""
    du, dv, dw, r_deg, p_deg, y_deg = angles_deg
    return [du, dv, dw, DEG(r_deg), DEG(p_deg), DEG(y_deg)]


# Extreme contexts spanning the isolated tilt/depth/yaw cells plus combined
# worst cases. All translations are zero (tilt is the only hard part).
EXTREME_CONTEXTS = [
    ("zero", _c(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ("dw+", _c(0.0, 0.0, 0.015, 0.0, 0.0, 0.0)),
    ("dw-", _c(0.0, 0.0, -0.015, 0.0, 0.0, 0.0)),
    ("roll+15", _c(0.0, 0.0, 0.0, 15.0, 0.0, 0.0)),
    ("roll-15", _c(0.0, 0.0, 0.0, -15.0, 0.0, 0.0)),
    ("pitch+15", _c(0.0, 0.0, 0.0, 0.0, 15.0, 0.0)),
    ("pitch-15", _c(0.0, 0.0, 0.0, 0.0, -15.0, 0.0)),
    ("roll+pitch+", _c(0.0, 0.0, 0.0, 15.0, 15.0, 0.0)),
    ("roll-pitch-", _c(0.0, 0.0, 0.0, -15.0, -15.0, 0.0)),
    ("tilt+yaw30", _c(0.0, 0.0, 0.0, 15.0, -15.0, 30.0)),
]


def screen(orientation, task_anchor, seeds, retries):
    base = gym.make(
        "KeyedCircularPhaseSwitchSE3-v1",
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
                    solve_se3(env)
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
                f"  {name:12s} seed={seed} {flag} (first={int(first)}, "
                f"retries={used})",
                flush=True,
            )
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orientation-quat", nargs=4, type=float, default=[1.0, 0.0, 0.0, 0.0],
        help="Global task rotation as a unit quaternion [w, x, y, z].",
    )
    parser.add_argument(
        "--task-anchor", nargs=3, type=float, default=[-0.35, 0.10, 0.08],
        help="World workspace placement of the socket nominal center [x, y, z].",
    )
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[20260818, 20270818, 20280818])
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--contexts", nargs="+", type=str, default=None,
                        help="Optional subset of context names (e.g. roll+15).")
    parser.add_argument("--out", type=str,
                        default="phase_switch_symmetry_multiseed/se3_tilt_probe.csv")
    args = parser.parse_args()

    orientation = np.asarray(args.orientation_quat, dtype=np.float64)
    orientation = orientation / np.linalg.norm(orientation)
    task_anchor = np.asarray(args.task_anchor, dtype=np.float64)

    contexts = EXTREME_CONTEXTS
    if args.contexts is not None:
        by_name = {name: cd for name, cd in EXTREME_CONTEXTS}
        contexts = [(name, by_name[name]) for name in args.contexts]

    print(f"orientation={orientation} task_anchor={task_anchor} "
          f"seeds={args.seeds} retries={args.retries} contexts={[n for n, _ in contexts]}")
    results = screen(orientation, task_anchor, args.seeds, args.retries)

    n = len(results)
    n_eventual = sum(r["eventual"] for r in results)
    n_first = sum(r["first_attempt"] for r in results)
    print(f"\nfirst-attempt: {n_first}/{n}   eventual: {n_eventual}/{n}")

    import pandas as pd

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print("saved:", args.out)

    # Systematic-failure check: any (context) failing across ALL seeds.
    by_name = {name: name for name, _ in contexts}
    for name, _ in contexts:
        sub = df[df.context == name]
        if sub.eventual.sum() == 0:
            print(f"  SYSTEMATIC FAILURE: {name} failed every seed")
    if n_eventual != n:
        raise SystemExit(f"tilt probe failed: {n_eventual}/{n} eventual success")


if __name__ == "__main__":
    main()
