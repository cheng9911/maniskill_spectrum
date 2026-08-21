from __future__ import annotations

"""Screen candidate reduced mixed-box corners for full SE(3) reachability.

Used to pick the largest fully-reachable mixed box, so the frozen manifest's 60
mixed contexts are all collectable (no triple-combination IK failures).
"""

import numpy as np

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_env  # noqa: F401
from collect_phase_switch_rollouts import EpisodeFinished, PHASES, PhaseSwitchTraceWrapper
from collect_phase_switch_rotated import solve_se3

DEG = np.deg2rad
PHASE_NAME = {code: name for name, code in PHASES.items()}


def _c(du, dv, dw, r, p, y):
    return [du, dv, dw, DEG(r), DEG(p), DEG(y)]


CORNERS = [
    ("A_10_12_15", _c(0.010, 0.010, -0.010, 12, -12, 15)),
    ("B_10_12_15", _c(-0.010, -0.010, 0.010, -12, 12, -15)),
    ("C_10_12_15", _c(0.010, 0.010, -0.010, 10, -10, 12)),
    ("D_8_10_12", _c(0.008, 0.008, -0.008, 10, -10, 12)),
    ("E_8_10_15", _c(0.008, 0.008, -0.008, 10, -10, 15)),
]


def main():
    orientation = np.array([1.0, 0.0, 0.0, 0.0])
    task_anchor = np.array([-0.35, 0.10, 0.08])
    base = gym.make(
        "KeyedCircularPhaseSwitchSE3-v1",
        num_envs=1, obs_mode="state_dict", control_mode="pd_joint_pos",
        reward_mode="sparse", sim_backend="physx_cpu", render_mode=None,
        robot_init_qpos_noise=0.0, orientation=orientation, task_anchor=task_anchor,
    )
    env = PhaseSwitchTraceWrapper(base)
    for name, cd in CORNERS:
        ev = 0
        lp = -1
        for a in range(2):
            env.reset(seed=20260818 + a * 7, options={"causal_delta": cd})
            env.start_trace()
            ok = False
            try:
                solve_se3(env)
                ok = True
            except EpisodeFinished as e:
                ok = bool(e.terminated)
            except Exception:
                ok = False
            lp = int(env.trace["states"][-1]["solver_phase"])
            if ok:
                ev = 1
                break
        print(f"{name:14s} eventual={ev} last={PHASE_NAME.get(lp, '?')}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
