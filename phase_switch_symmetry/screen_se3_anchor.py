from __future__ import annotations

"""Screen candidate anchors for full 6D mixed-box reachability.

The frozen anchor (-0.35, 0.10, 0.08) is only ~0.29 m from the Panda base; the
bent-arm configuration constrains wrist orientation and makes the combined
tilt+yaw+translation align pose IK-unsolvable. This screen tests the worst mixed
corner (all six generators at max) at candidate more-central anchors.
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


WORST_CORNERS = [
    ("allmax", _c(0.012, 0.012, -0.012, 15, -15, 30)),
    ("allmax2", _c(-0.012, -0.012, 0.012, -15, 15, -30)),
    ("negpitch", _c(0.010, 0.004, -0.006, 8, -5, 12)),
]

ANCHORS = [
    ("a_near", [-0.35, 0.10, 0.08]),   # frozen
    ("a_mid1", [-0.25, 0.00, 0.08]),
    ("a_mid2", [-0.15, 0.00, 0.08]),
    ("a_far", [-0.05, 0.00, 0.10]),
]


def main():
    orientation = np.array([1.0, 0.0, 0.0, 0.0])
    for anchor_name, anchor in ANCHORS:
        anchor = np.array(anchor, dtype=np.float64)
        base = gym.make(
            "KeyedCircularPhaseSwitchSE3-v1",
            num_envs=1, obs_mode="state_dict", control_mode="pd_joint_pos",
            reward_mode="sparse", sim_backend="physx_cpu", render_mode=None,
            robot_init_qpos_noise=0.0, orientation=orientation, task_anchor=anchor,
        )
        env = PhaseSwitchTraceWrapper(base)
        for name, cd in WORST_CORNERS:
            ev = 0
            lp = -1
            for a in range(3):
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
            print(f"{anchor_name:8s} {name:10s} eventual={ev} last={PHASE_NAME.get(lp, '?')}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
