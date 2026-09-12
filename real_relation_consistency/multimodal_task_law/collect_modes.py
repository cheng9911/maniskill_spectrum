from __future__ import annotations

"""Collect multimodal demonstration rollouts for the SAME circular insertion task.

Stages (all read the frozen protocol.json):
  pilot : 4 modes x {0, +-7.5} deg x 3 blocks  = 36 primary attempts
  core  : 4 modes x 13 pitch conditions x 5 blocks = 260 primary attempts
  yaw   : 2 yaw modes x {0, +-15, +-30} deg x 5 blocks = 50 primary attempts

Every attempt is retained (no deletion of failures).  The seed schedule splits the
initial-state seed (shared within a block -> paired same initial qpos) from the
planner seed (per mode x condition).  The actual initial qpos is hashed and stored
so pairing can be verified empirically rather than asserted from seeds.
"""

import argparse
import hashlib
import json
import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "phase_switch_symmetry"))
sys.path.insert(0, str(HERE))

import mani_skill.envs  # noqa: F401, E402
import phase_switch_symmetry_env  # noqa: F401, E402
from collect_phase_switch_rollouts import EpisodeFinished  # noqa: E402
from generate_contexts import load_protocol  # noqa: E402
from solver_modes import (  # noqa: E402
    FINAL,
    ModeTraceWrapper,
    solve_multimodal_se3,
    solve_yaw_challenge,
)
from mani_skill.agents.robots.panda import Panda  # noqa: E402

Panda.gripper_stiffness = 2.5e3
Panda.gripper_force_limit = 150.0

_TRACE_KEYS = [
    "tcp_pose", "peg_pose", "socket_pose", "goal_pose", "qpos", "qvel",
    "contact_force", "success", "obj_to_goal_dist", "axis_angle_err",
    "yaw_err", "key_clearance_margin", "solver_phase",
]


def _hash_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def _assembly_complete(states: list[dict], socket_axis: np.ndarray) -> bool:
    """Preliminary geometric completeness: env success + terminal depth inside the
    frozen success ball.  The solver (identical to the frozen solve_se3) stops at
    peg-centre depth ~0.070-0.072, i.e. ~11 mm short of FINAL=0.060, which is
    exactly the env's pos_err<0.012 success ball.  So 'reached the target depth'
    is calibrated to FINAL + 0.012, not FINAL + 0.003."""
    if not states or not bool(states[-1]["success"]):
        return False
    peg = np.asarray(states[-1]["peg_pose"], dtype=np.float64)[:3]
    sock = np.asarray(states[-1]["socket_pose"], dtype=np.float64)[:3]
    axial = float(np.dot(peg - sock, socket_axis))
    return axial <= FINAL + 0.012


def write_mode_episode(group, row, trace, solver_error, stop_reason):
    group.attrs["solver_error"] = "" if solver_error is None else solver_error
    group.attrs["stop_reason"] = stop_reason
    group.create_dataset(
        "causal_delta", data=np.asarray(row["causal_delta"], dtype=np.float64)
    )
    states = trace["states"]
    for key in _TRACE_KEYS:
        group.create_dataset(key, data=np.asarray([s[key] for s in states]))
    group.create_dataset("actions", data=np.asarray(trace["actions"]))
    group.create_dataset("rewards", data=np.asarray(trace["rewards"]))
    group.create_dataset("terminated", data=np.asarray(trace["terminated"], bool))
    group.create_dataset("truncated", data=np.asarray(trace["truncated"], bool))


def collect_stage(protocol, stage, output_path, seed_base, retries, smoke=False):
    p = protocol
    env_id = p["task"]["env_id"]
    orientation = np.asarray(p["task"]["orientation"], dtype=np.float64)
    task_anchor = p["task"]["task_anchor"]
    pickup = p["task"]["pickup_position"]
    noise = p["task"]["robot_init_qpos_noise"]

    if stage == "pilot":
        mode_ids = p["mode_ids"]
        pitch_deg = p["pilot_pitch_deg"]
        n_blocks = p["pilot_blocks"]
        yaw_modes = []
    elif stage == "core":
        mode_ids = p["mode_ids"]
        pitch_deg = p["pitch_deg"]
        n_blocks = p["core_blocks"]
        yaw_modes = []
    elif stage == "yaw":
        mode_ids = []
        pitch_deg = [0.0]
        n_blocks = p["yaw_blocks"]
        yaw_modes = p["yaw_mode_ids"]
    else:
        raise ValueError(stage)

    train_pitch = set(p.get("train_pitch_deg", []))
    test_pitch = set(p.get("test_pitch_deg", []))
    yaw_train = set(p.get("yaw_train_deg", []))
    yaw_test = set(p.get("yaw_test_deg", []))

    is_stride = p["seed"]["initial_state_stride"]
    pl_stride = p["seed"]["planner_stride"]

    base = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state_dict",
        control_mode=p["task"]["control_mode"],
        reward_mode="sparse",
        sim_backend=p["task"]["sim_backend"],
        render_mode=None,
        robot_init_qpos_noise=noise,
        orientation=orientation,
        task_anchor=task_anchor,
    )
    env = ModeTraceWrapper(base)

    # Build the (mode, condition, block) job list.
    jobs = []
    if mode_ids:
        for mi, mode_id in enumerate(mode_ids):
            for ci, deg in enumerate(pitch_deg):
                for bi in range(n_blocks):
                    jobs.append(dict(
                        mode_id=mode_id, yaw_mode=None,
                        mode_index=mi, condition_index=ci, block_id=bi,
                        pitch_deg=deg,
                        causal_delta=[0.0, 0.0, 0.0, 0.0, np.deg2rad(deg), 0.0],
                    ))
    if yaw_modes:
        for mi, yaw_mode in enumerate(yaw_modes):
            for ci, deg in enumerate(p["yaw_deg"]):
                for bi in range(n_blocks):
                    jobs.append(dict(
                        mode_id=None, yaw_mode=yaw_mode,
                        mode_index=mi, condition_index=ci, block_id=bi,
                        pitch_deg=0.0,
                        causal_delta=[0.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(deg)],
                    ))

    if smoke:
        jobs = jobs[:1]

    n_cond = len(pitch_deg) if mode_ids else len(p["yaw_deg"])
    manifest = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as data_file:
        data_file.attrs["env_id"] = env_id
        data_file.attrs["stage"] = stage
        data_file.attrs["source_type"] = "motionplanning_physx_multimodal"
        data_file.attrs["protocol_sha256"] = p["content_sha256"]
        data_file.attrs["orientation"] = orientation
        data_file.attrs["task_anchor"] = np.asarray(task_anchor, dtype=np.float64)
        data_file.attrs["pickup_position"] = np.asarray(pickup, dtype=np.float64)
        data_file.attrs["control_mode"] = p["task"]["control_mode"]
        data_file.attrs["sim_backend"] = p["task"]["sim_backend"]
        data_file.attrs["seed_base"] = seed_base
        data_file.attrs["mode_ids_json"] = json.dumps(mode_ids)
        data_file.attrs["yaw_mode_ids_json"] = json.dumps(yaw_modes)
        data_file.attrs["retries_per_condition"] = retries
        for label, source_path in {
            "environment": Path(phase_switch_symmetry_env.__file__),
            "collector": Path(__file__),
            "solver": HERE / "solver_modes.py",
            "protocol": HERE / "protocol.json",
        }.items():
            data_file.attrs[f"{label}_source_sha256"] = hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest()

        episode_id = 0
        for job in jobs:
            for attempt_id in range(retries):
                mi, ci, bi = job["mode_index"], job["condition_index"], job["block_id"]
                is_seed = seed_base + bi * is_stride
                planner_seed = is_seed + pl_stride * (mi * n_cond + ci) + attempt_id
                episode_seed = planner_seed

                print(
                    f"[{stage}] {job.get('mode_id') or job.get('yaw_mode')} "
                    f"block={bi} cond={ci} ({job['pitch_deg']:+g} deg) "
                    f"attempt={attempt_id}",
                    flush=True,
                )
                np.random.seed(is_seed)
                torch.manual_seed(episode_seed)
                env.reset(
                    seed=episode_seed,
                    options={
                        "causal_delta": np.asarray(job["causal_delta"], dtype=np.float64),
                        "pickup_position": np.asarray(pickup, dtype=np.float64),
                    },
                )
                initial_qpos = np.asarray(env.unwrapped.agent.robot.get_qpos())[0].copy()
                env.start_trace()
                solver_error = None
                stop_reason = "solver_returned"
                try:
                    if job["yaw_mode"] is not None:
                        solve_yaw_challenge(env, job["yaw_mode"])
                    else:
                        params = p["mode_params"][job["mode_id"]]
                        solve_multimodal_se3(
                            env, params["path_mode"], params["alignment_mode"]
                        )
                except EpisodeFinished as exc:
                    stop_reason = f"EpisodeFinished({exc})"
                except Exception as exc:  # noqa: BLE001
                    solver_error = repr(exc)
                    stop_reason = "solver_exception"
                    print("  solver exception:", solver_error, flush=True)

                states = env.trace["states"]
                forces = np.linalg.norm(
                    np.asarray([s["contact_force"] for s in states]), axis=1
                )
                success = bool(states[-1]["success"])
                # socket axis from the socket pose's own orientation
                from transforms3d.quaternions import quat2mat
                sock_q = np.asarray(states[0]["socket_pose"])[3:]
                sock_axis = quat2mat(sock_q) @ np.array([0.0, 0.0, 1.0])
                assembly_complete = _assembly_complete(states, sock_axis)
                steps = len(env.trace["actions"])

                deg = job["pitch_deg"]
                split = None
                if job["yaw_mode"] is not None:
                    split = "train" if deg in yaw_train else ("test" if deg in yaw_test else None)
                else:
                    split = "train" if deg in train_pitch else ("test" if deg in test_pitch else None)

                print(
                    f"  steps={steps} success={success} assembly_complete={assembly_complete} "
                    f"max_contact={forces.max():.3f} split={split}",
                    flush=True,
                )
                group = data_file.create_group(f"episode_{episode_id}")
                for key, value in dict(
                    mode_id=job["mode_id"] if job["mode_id"] else "",
                    yaw_mode=job["yaw_mode"] if job["yaw_mode"] else "",
                    mode_index=mi,
                    block_id=bi,
                    condition_index=ci,
                    split=split if split else "",
                    attempt_id=attempt_id,
                    initial_state_seed=is_seed,
                    planner_seed=planner_seed,
                    episode_seed=episode_seed,
                    pitch_deg=float(deg),
                    initial_qpos_hash=_hash_array(initial_qpos),
                ).items():
                    group.attrs[key] = value
                row = dict(causal_delta=job["causal_delta"])
                write_mode_episode(group, row, env.trace, solver_error, stop_reason)
                manifest.append(dict(
                    episode_id=episode_id,
                    stage=stage,
                    mode_id=job["mode_id"] if job["mode_id"] else "",
                    yaw_mode=job["yaw_mode"] if job["yaw_mode"] else "",
                    block_id=bi,
                    condition_index=ci,
                    split=split,
                    attempt_id=attempt_id,
                    pitch_deg=float(deg),
                    initial_state_seed=is_seed,
                    planner_seed=planner_seed,
                    episode_seed=episode_seed,
                    initial_qpos_hash=_hash_array(initial_qpos),
                    steps=steps,
                    success=success,
                    assembly_complete=bool(assembly_complete),
                    stop_reason=stop_reason,
                    max_contact_force_N=float(forces.max()),
                    solver_error=solver_error,
                ))
                episode_id += 1

        data_file.attrs["job_count"] = len(jobs)
        data_file.attrs["episode_count"] = len(manifest)
        data_file.attrs["success_count"] = sum(r["success"] for r in manifest)

    env.close()

    with output_path.with_suffix(".json").open("w", encoding="utf-8") as mf:
        json.dump(dict(
            stage=stage,
            protocol_sha256=p["content_sha256"],
            seed_base=seed_base,
            retries_per_condition=retries,
            job_count=len(jobs),
            episodes=manifest,
        ), mf, indent=2)

    n_succ = sum(r["success"] for r in manifest)
    print(f"[{stage}] done: {len(jobs)} jobs, {len(manifest)} episodes, "
          f"{n_succ}/{len(manifest)} success")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--stage", choices=["pilot", "core", "yaw"], default="pilot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=20260818)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    collect_stage(protocol, args.stage, args.output, args.seed_base, args.retries,
                  smoke=args.smoke)


if __name__ == "__main__":
    main()
