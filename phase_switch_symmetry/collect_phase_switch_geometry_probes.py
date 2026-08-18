from __future__ import annotations

import argparse
import json
import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_env  # noqa: F401
from collect_phase_switch_rollouts import (
    EpisodeFinished,
    PhaseSwitchTraceWrapper,
    solve,
    write_episode,
)


def probe_conditions():
    conditions = []
    for socket_yaw_deg in [-30.0, 30.0]:
        conditions.append(
            dict(
                probe_type="matched_preclear",
                socket_yaw_deg=socket_yaw_deg,
                keyed_yaw_deg=socket_yaw_deg,
                post_clear_yaw_deg=0.0,
                stop_after_phase="enter_key",
            )
        )
        conditions.append(
            dict(
                probe_type="mismatched_preclear",
                socket_yaw_deg=socket_yaw_deg,
                keyed_yaw_deg=0.0,
                post_clear_yaw_deg=0.0,
                stop_after_phase="enter_key",
            )
        )
    for post_clear_yaw_deg in [-30.0, 30.0]:
        conditions.append(
            dict(
                probe_type="arbitrary_postclear_yaw",
                socket_yaw_deg=0.0,
                keyed_yaw_deg=0.0,
                post_clear_yaw_deg=post_clear_yaw_deg,
                stop_after_phase=None,
            )
        )
    return conditions


def collect(output_path: Path, seed: int, retries_per_condition: int):
    env_id = "KeyedCircularPhaseSwitch-v1"
    base = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_mode=None,
        robot_init_qpos_noise=0.0,
    )
    env = PhaseSwitchTraceWrapper(base)
    conditions = probe_conditions()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    episode_id = 0
    with h5py.File(output_path, "w") as data_file:
        data_file.attrs["env_id"] = env_id
        data_file.attrs["task"] = "keyed_to_circular_geometry_probes"
        data_file.attrs["source_type"] = "motionplanning_physx_geometry_counterfactual"
        data_file.attrs["control_mode"] = "pd_joint_pos"
        data_file.attrs["sim_backend"] = "physx_cpu"
        data_file.attrs["seed"] = seed
        data_file.attrs["condition_count"] = len(conditions)
        data_file.attrs["retries_per_condition"] = retries_per_condition
        for condition_id, condition in enumerate(conditions):
            for attempt_id in range(retries_per_condition):
                episode_seed = seed + condition_id * 1009 + attempt_id
                np.random.seed(episode_seed)
                torch.manual_seed(episode_seed)
                causal_delta = [
                    0.0,
                    0.0,
                    float(np.deg2rad(condition["socket_yaw_deg"])),
                ]
                row = dict(generator="geometry_probe", causal_delta=causal_delta)
                print(
                    f"[probe {condition_id + 1}/{len(conditions)} "
                    f"attempt {attempt_id + 1}/{retries_per_condition}] {condition}",
                    flush=True,
                )
                env.reset(
                    seed=episode_seed, options={"causal_delta": causal_delta}
                )
                env.start_trace()
                solver_error = None
                stop_reason = "solver_returned"
                try:
                    solve(
                        env,
                        keyed_yaw_override=float(
                            np.deg2rad(condition["keyed_yaw_deg"])
                        ),
                        post_clear_yaw=float(
                            np.deg2rad(condition["post_clear_yaw_deg"])
                        ),
                        stop_after_phase=condition["stop_after_phase"],
                    )
                except EpisodeFinished as exc:
                    stop_reason = str(exc)
                except Exception as exc:
                    solver_error = repr(exc)
                    stop_reason = "solver_exception"
                    print("  solver exception:", solver_error, flush=True)

                states = env.trace["states"]
                force = np.linalg.norm(
                    np.asarray([state["contact_force"] for state in states]), axis=1
                )
                phases = sorted(
                    set(int(state["solver_phase"]) for state in states)
                )
                success = bool(states[-1]["success"])
                final_clearance = float(states[-1]["key_clearance_margin"])
                print(
                    f"  steps={len(env.trace['actions'])} success={success} "
                    f"clearance={final_clearance:.4f} max_contact={force[1:].max():.3f} "
                    f"stop={stop_reason}",
                    flush=True,
                )
                group = data_file.create_group(f"episode_{episode_id}")
                group.attrs["condition_id"] = condition_id
                group.attrs["attempt_id"] = attempt_id
                group.attrs["episode_seed"] = episode_seed
                for key, value in condition.items():
                    group.attrs[key] = "" if value is None else value
                write_episode(
                    group, row, env.trace, solver_error, stop_reason=stop_reason
                )
                manifest.append(
                    dict(
                        episode_id=episode_id,
                        condition_id=condition_id,
                        attempt_id=attempt_id,
                        episode_seed=episode_seed,
                        **condition,
                        steps=len(env.trace["actions"]),
                        success=success,
                        final_key_clearance_m=final_clearance,
                        max_contact_force_N=float(force[1:].max()),
                        stop_reason=stop_reason,
                        solver_error=solver_error,
                    )
                )
                episode_id += 1

                probe_type = condition["probe_type"]
                if probe_type == "matched_preclear" and final_clearance > 0.002:
                    break
                if (
                    probe_type == "mismatched_preclear"
                    and final_clearance <= 0.0
                    and force[1:].max() > 1e-3
                ):
                    break
                if probe_type == "arbitrary_postclear_yaw" and success:
                    break
        data_file.attrs["episode_count"] = episode_id
    env.close()
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as output_file:
        json.dump(dict(conditions=conditions, episodes=manifest), output_file, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "phase_switch_symmetry_rollouts/keyed_circular_geometry_probes.h5"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--retries-per-condition", type=int, default=3)
    args = parser.parse_args()
    collect(args.output, args.seed, args.retries_per_condition)


if __name__ == "__main__":
    main()
