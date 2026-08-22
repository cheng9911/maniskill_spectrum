from __future__ import annotations

"""Retry a single multigen condition cell with an extended attempt budget and
APPEND the extra episodes to the existing H5 (same discipline as the frozen
SE(3) collection's R_max=12 re-collection of condition 49).

Usage:
    python retry_multigen_cell.py --h5 phase_switch_symmetry_rollouts_se3/multigen_seed_20270818.h5 \
        --condition-id 49 --attempts 12
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_env  # noqa: F401
import phase_switch_symmetry_env_multigen  # noqa: F401
from collect_phase_switch_rollouts import (
    EpisodeFinished,
    PhaseSwitchTraceWrapper,
    write_episode,
)
from collect_se3_multigen_rollouts import solve_se3_multigen
from mani_skill.agents.robots.panda import Panda

Panda.gripper_stiffness = 2.5e3
Panda.gripper_force_limit = 150.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--condition-id", type=int, required=True)
    parser.add_argument("--attempts", type=int, default=12)
    args = parser.parse_args()

    manifest_path = args.h5.with_suffix(".json")
    seed = None
    with h5py.File(args.h5, "r") as data_file:
        seed = int(data_file.attrs["seed"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    condition_rows = json.loads(
        Path("phase_switch_symmetry_multiseed/se3_fixed_contexts.json").read_text(
            encoding="utf-8"
        )
    )["conditions"]
    row = condition_rows[args.condition_id]
    row = dict(generator=str(row["generator"]),
               causal_delta=[float(x) for x in row["causal_delta"]])

    orientation = [1.0, 0.0, 0.0, 0.0]
    task_anchor = [-0.15, 0.0, 0.08]
    base = gym.make(
        "KeyedCircularPhaseSwitchSE3Multigen-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_mode=None,
        robot_init_qpos_noise=0.01,
        orientation=orientation,
        task_anchor=task_anchor,
    )
    env = PhaseSwitchTraceWrapper(base)

    done_attempts = {
        int(e["attempt_id"]) for e in manifest["episodes"]
        if e["condition_id"] == args.condition_id
    }
    new_episodes = []
    with h5py.File(args.h5, "a") as data_file:
        episode_id = int(data_file.attrs["episode_count"])
        for attempt_id in range(args.attempts):
            if attempt_id in done_attempts:
                continue
            print(
                f"[condition {args.condition_id:03d} attempt {attempt_id + 1}/{args.attempts}] "
                f"{row['generator']}",
                flush=True,
            )
            episode_seed = seed + args.condition_id * 1009 + attempt_id
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            env.reset(seed=episode_seed, options={"causal_delta": row["causal_delta"]})
            env.start_trace()
            solver_error = None
            stop_reason = "solver_returned"
            try:
                solve_se3_multigen(env)
            except EpisodeFinished as exc:
                stop_reason = str(exc)
            except Exception as exc:
                solver_error = repr(exc)
                stop_reason = "solver_exception"
                print("  solver exception:", solver_error, flush=True)
            states = env.trace["states"]
            forces = np.linalg.norm(
                np.asarray([state["contact_force"] for state in states]), axis=1
            )
            phases = np.asarray([state["solver_phase"] for state in states])
            success = bool(states[-1]["success"])
            steps = len(env.trace["actions"])
            phase_set = sorted(set(int(x) for x in phases))
            complete = all(code in phase_set for code in [3, 4, 5, 6])
            print(
                f"  steps={steps} success={success} max_contact={forces.max():.3f} "
                f"phases={phase_set}",
                flush=True,
            )
            group = data_file.create_group(f"episode_{episode_id}")
            group.attrs["condition_id"] = args.condition_id
            group.attrs["attempt_id"] = attempt_id
            group.attrs["episode_seed"] = episode_seed
            write_episode(group, row, env.trace, solver_error, stop_reason=stop_reason)
            new_episodes.append(
                dict(
                    episode_id=episode_id,
                    condition_id=args.condition_id,
                    attempt_id=attempt_id,
                    **row,
                    steps=steps,
                    success=success,
                    complete=complete,
                    stop_reason=stop_reason,
                    max_contact_force_N=float(forces.max()),
                    phases=phase_set,
                    solver_error=solver_error,
                )
            )
            episode_id += 1
            if success and complete:
                break
        data_file.attrs["episode_count"] = int(data_file.attrs["episode_count"]) + len(
            new_episodes
        )
        data_file.attrs["success_count"] = int(data_file.attrs["success_count"]) + sum(
            e["success"] for e in new_episodes
        )
    env.close()

    manifest["episodes"].extend(new_episodes)
    manifest["episodes"].sort(key=lambda e: (e["episode_id"],))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"appended {len(new_episodes)} episodes to {args.h5}")


if __name__ == "__main__":
    main()
