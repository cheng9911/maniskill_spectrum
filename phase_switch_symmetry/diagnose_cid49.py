from __future__ import annotations

"""Diagnose keyed-seed-20260818 condition 49 (the single 74/75 failure): is it
robustly IK-unsolvable under robot_init_qpos_noise=0.01, or just unlucky on the
5 fixed attempt seeds? Sweeps a spread of episode seeds at the frozen anchor."""

import numpy as np
import torch
import gymnasium as gym

import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_env  # noqa: F401
from collect_phase_switch_rollouts import EpisodeFinished, PhaseSwitchTraceWrapper
from collect_phase_switch_rotated import solve_se3
from mani_skill.agents.robots.panda import Panda

Panda.gripper_stiffness = 2.5e3
Panda.gripper_force_limit = 150.0

CD = [0.006700197686226667, 0.007031642902322426, 0.003414856543518677,
      0.23813995930370252, -0.10468313928153686, 0.5120500496057111]
PHASE_NAME = {0: "reach", 1: "grasp", 2: "lift", 3: "align", 4: "enter",
              5: "unlock", 6: "insert", -1: "init"}


def main():
    base = gym.make(
        "KeyedCircularPhaseSwitchSE3-v1",
        num_envs=1, obs_mode="state_dict", control_mode="pd_joint_pos",
        reward_mode="sparse", sim_backend="physx_cpu", render_mode=None,
        robot_init_qpos_noise=0.01, orientation=np.array([1.0, 0, 0, 0]),
        task_anchor=np.array([-0.15, 0.0, 0.08]),
    )
    env = PhaseSwitchTraceWrapper(base)
    ok = 0
    # Replicate the collector's exact deterministic seeding: episode_seed =
    # seed + condition_id*1009 + attempt_id, with np.random + torch seeded first.
    for attempt_id in range(15):
        episode_seed = 20260818 + 49 * 1009 + attempt_id
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        env.reset(seed=episode_seed, options={"causal_delta": CD})
        env.start_trace()
        r = False
        try:
            solve_se3(env)
            r = True
        except EpisodeFinished as exc:
            r = bool(exc.terminated)
        except Exception:
            r = False
        lp = int(env.trace["states"][-1]["solver_phase"])
        ok += r
        print(f"attempt={attempt_id:2d} seed={episode_seed} reach={int(r)} "
              f"last={PHASE_NAME.get(lp,'?')}", flush=True)
    print(f"--- {ok}/15 reachable (attempts 0-14) ---")
    env.close()


if __name__ == "__main__":
    main()
