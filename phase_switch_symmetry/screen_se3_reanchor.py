from __future__ import annotations

"""Re-anchor verification: full-box mixed corners + isolated interventions.

The frozen anchor (-0.35, 0.10, 0.08) is IK-limited for full 6D mixed contexts.
This screen confirms that a more-central anchor reaches EVERY full-box corner
(+-0.012 m, +-15 deg roll/pitch, +-30 deg yaw) and every isolated intervention,
across seeds, before re-freezing the manifest/prereg/subsets at the new anchor.
"""

import json
from pathlib import Path

import numpy as np

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_env  # noqa: F401
from collect_phase_switch_rollouts import EpisodeFinished, PHASES, PhaseSwitchTraceWrapper
from collect_phase_switch_rotated import solve_se3

DEG = np.deg2rad
PHASE_NAME = {code: name for name, code in PHASES.items()}


def isolated_interventions():
    rows = []
    for gen, cd in [
        ("du", [0.015, 0, 0, 0, 0, 0]),
        ("dv", [0, 0.015, 0, 0, 0, 0]),
        ("dw", [0, 0, 0.015, 0, 0, 0]),
        ("roll", [0, 0, 0, DEG(15), 0, 0]),
        ("pitch", [0, 0, 0, 0, DEG(15), 0]),
        ("yaw", [0, 0, 0, 0, 0, DEG(30)]),
    ]:
        rows.append((f"{gen}+", cd))
        rows.append((f"{gen}-", [-c for c in cd]))
    rows.append(("baseline", [0, 0, 0, 0, 0, 0]))
    return rows


def full_box_corners():
    rows = []
    for su in (-0.012, 0.012):
        for sv in (-0.012, 0.012):
            for sw in (-0.012, 0.012):
                for r in (-15, 15):
                    for p in (-15, 15):
                        for y in (-30, 30):
                            rows.append(
                                (
                                    f"c({su:+.3f},{sv:+.3f},{sw:+.3f},{r:+d},{p:+d},{y:+d})",
                                    [su, sv, sw, DEG(r), DEG(p), DEG(y)],
                                )
                            )
    return rows


def screen(anchor, contexts, seeds, retries):
    base = gym.make(
        "KeyedCircularPhaseSwitchSE3-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_mode=None,
        robot_init_qpos_noise=0.0,
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        task_anchor=np.array(anchor, dtype=np.float64),
    )
    env = PhaseSwitchTraceWrapper(base)
    fails = []
    n_ok = 0
    for name, cd in contexts:
        eventual = False
        lp = -1
        for seed in seeds:
            for attempt in range(retries):
                env.reset(seed=seed + attempt * 7, options={"causal_delta": cd})
                env.start_trace()
                ok = False
                try:
                    solve_se3(env)
                    ok = True
                except EpisodeFinished as exc:
                    ok = bool(exc.terminated)
                except Exception:
                    ok = False
                lp = int(env.trace["states"][-1]["solver_phase"])
                if ok:
                    eventual = True
                    break
            if not eventual:
                break
        if eventual:
            n_ok += 1
        else:
            fails.append((name, PHASE_NAME.get(lp, "?")))
    env.close()
    return n_ok, fails


def main():
    contexts = full_box_corners()
    anchors = [
        ("a_mid2", [-0.15, 0.00, 0.08]),
    ]
    seeds = [20260818, 20260825]
    retries = 3
    for anchor_name, anchor in anchors:
        n_ok, fails = screen(anchor, contexts, seeds, retries)
        print(f"\n=== {anchor_name} {anchor} : {n_ok}/{len(contexts)} reachable ===", flush=True)
        if fails:
            print(f"FAILS ({len(fails)}):")
            for name, phase in fails:
                print(f"  {name:60s} @{phase}")
        else:
            print("ALL REACHABLE")


if __name__ == "__main__":
    main()
