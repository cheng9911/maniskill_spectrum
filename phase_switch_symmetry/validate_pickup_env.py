from __future__ import annotations

"""Validate the pickup_position dimension in CircularPhaseSwitchSE3-v1.

Checks (no motion planning, so it is fast):
1. default reset  -> peg spawns on the fixed pedestal (PEDESTAL_POS).
2. custom reset   -> peg + pedestal move to the pickup context.
3. default reset  -> peg + pedestal return to the default (no stale context).
4. z_top != PEDESTAL_POS[2] raises ValueError.
Also prints the horizontal clearance of each candidate pickup from the socket.

The pedestal actor's world pose must equal pickup_position - PEDESTAL_POS
because the box collision/visual is built at the LOCAL pose PEDESTAL_POS.
"""

import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
from pathlib import Path

import numpy as np

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_env as pse


def _xyz(env, attr):
    return env.unwrapped.__getattribute__(attr).pose.p[0].detach().cpu().numpy().copy()


def main():
    PED = pse.PEDESTAL_POS
    env = gym.make(
        "CircularPhaseSwitchSE3-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_mode=None,
        robot_init_qpos_noise=0.0,
        orientation=[1.0, 0.0, 0.0, 0.0],
        task_anchor=[-0.15, 0.0, 0.08],
    )
    resting = env.unwrapped._resting_height()

    # 1. default reset
    env.reset(seed=0, options={"causal_delta": np.zeros(6)})
    peg0 = _xyz(env, "peg")
    ped0 = _xyz(env, "pedestal")
    exp_peg0 = np.array([PED[0], PED[1], PED[2] + resting])
    print(f"default peg       = {peg0}  (expect {exp_peg0})")
    print(f"default pedestal  = {ped0}  (expect ~[0,0,0])")
    assert np.allclose(peg0, exp_peg0, atol=1e-3), "default peg spawn mismatch"
    assert np.allclose(ped0, np.zeros(3), atol=1e-3), "default pedestal should sit at origin"

    # 2. custom reset
    custom = np.array([-0.30, -0.02, 0.28])
    env.reset(seed=1, options={"causal_delta": np.zeros(6), "pickup_position": custom})
    peg1 = _xyz(env, "peg")
    ped1 = _xyz(env, "pedestal")
    exp_peg1 = np.array([custom[0], custom[1], custom[2] + resting])
    exp_ped1 = custom - PED
    print(f"custom peg        = {peg1}  (expect {exp_peg1})")
    print(f"custom pedestal   = {ped1}  (expect {exp_ped1})")
    assert np.allclose(peg1, exp_peg1, atol=1e-3), "custom peg spawn mismatch"
    assert np.allclose(ped1, exp_ped1, atol=1e-3), "custom pedestal pose mismatch"

    # 3. default restore
    env.reset(seed=2, options={"causal_delta": np.zeros(6)})
    peg2 = _xyz(env, "peg")
    ped2 = _xyz(env, "pedestal")
    print(f"restore peg       = {peg2}  (expect {exp_peg0})")
    assert np.allclose(peg2, exp_peg0, atol=1e-3), "default restore failed (stale context)"
    assert np.allclose(ped2, np.zeros(3), atol=1e-3), "default pedestal restore failed"

    # 4. z_top mismatch raises
    try:
        env.reset(
            seed=3,
            options={"causal_delta": np.zeros(6), "pickup_position": [-0.25, -0.18, 0.20]},
        )
        raise AssertionError("z_top mismatch did not raise ValueError")
    except ValueError as exc:
        print("z_top mismatch raised ValueError:", exc)

    # geometry clearance from the socket horizontal center (the collector's
    # default task_anchor [-0.15, 0, 0.08], NOT the env __init__ SOCKET_CENTER).
    socket_c = np.array([-0.15, 0.0], dtype=np.float64)
    print(f"\nsocket horizontal center: {socket_c}")
    for p in ([-0.25, -0.18, 0.28], [-0.30, -0.02, 0.28], [-0.20, -0.30, 0.28]):
        d = float(np.linalg.norm(np.asarray(p[:2]) - socket_c))
        flag = "" if d >= 0.10 else "  <-- BELOW 0.10 m"
        print(f"  pickup {p}: clearance {d:.3f} m{flag}")

    env.close()
    print("\nOK: reset/restore/z_top/geometry checks passed")


if __name__ == "__main__":
    main()
