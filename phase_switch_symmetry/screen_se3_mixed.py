from __future__ import annotations

"""Reachability screen for SE(3) MIXED contexts (translation + tilt + yaw).

The Phase-0 probe only covered pure tilt (translation=0). The full experiment
requires the 60 mixed contexts which combine +-0.012 m translation, +-15 deg
roll/pitch, and +-30 deg yaw. This screen isolates which factor (translation,
yaw, or their combination with tilt) drives IK failures, and samples the frozen
mixed manifest to estimate the usable fraction.
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_env  # noqa: F401
from collect_phase_switch_rollouts import EpisodeFinished, PHASES, PhaseSwitchTraceWrapper
from collect_phase_switch_rotated import solve_se3

PHASE_NAME = {code: name for name, code in PHASES.items()}
DEG = np.deg2rad


def _c(du, dv, dw, r, p, y):
    return [du, dv, dw, DEG(r), DEG(p), DEG(y)]


FACTOR_CONTEXTS = [
    ("translation_only", _c(0.012, 0.012, -0.012, 0, 0, 0)),
    ("yaw30_only", _c(0, 0, 0, 0, 0, 30)),
    ("tilt_only", _c(0, 0, 0, 15, 15, 0)),
    ("tilt+yaw15", _c(0, 0, 0, 15, 15, 15)),
    ("tilt+yaw30", _c(0, 0, 0, 15, -15, 30)),
    ("tilt+trans", _c(0.012, -0.012, 0.012, 15, -15, 0)),
    ("trans+yaw30", _c(0.012, 0.012, 0.012, 0, 0, 30)),
    ("full_mixed", _c(0.012, 0.012, -0.012, 15, -15, 30)),
]


def screen(orientation, task_anchor, contexts, seeds, retries):
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
    for name, causal_delta in contexts:
        for seed in seeds:
            eventual = False
            first = False
            used = 0
            last_phase = -1
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
                dict(context=name, seed=seed, first_attempt=first, eventual=eventual,
                     retries_used=used, last_phase=PHASE_NAME.get(last_phase, "?"))
            )
            flag = "OK " if eventual else f"FAIL@{PHASE_NAME.get(last_phase, '?')}"
            print(f"  {name:18s} seed={seed} {flag} (first={int(first)} retries={used})", flush=True)
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("phase_switch_symmetry_multiseed/se3_fixed_contexts.json"))
    parser.add_argument("--n-mixed", type=int, default=12)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--out", type=Path,
                        default=Path("phase_switch_symmetry_multiseed/se3_mixed_screen.csv"))
    args = parser.parse_args()

    orientation = np.array([1.0, 0.0, 0.0, 0.0])
    task_anchor = np.array([-0.35, 0.10, 0.08])

    with args.manifest.open(encoding="utf-8") as f:
        manifest = json.load(f)
    mixed = [r for r in manifest["conditions"] if r["generator"] == "mixed"]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(mixed), args.n_mixed, replace=False)
    sampled = [
        (f"mixed_{i}", mixed[i]["causal_delta"]) for i in sorted(idx)
    ]

    contexts = FACTOR_CONTEXTS + sampled
    print(f"screen {len(contexts)} contexts x 1 seed x {args.retries} retries")
    results = screen(orientation, task_anchor, contexts, [args.seed], args.retries)

    import pandas as pd

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    n = len(df)
    n_ev = int(df.eventual.sum())
    print(f"\neventual: {n_ev}/{n}")
    fail = df[~df.eventual]
    if len(fail):
        print("failures:")
        print(fail.to_string(index=False))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
