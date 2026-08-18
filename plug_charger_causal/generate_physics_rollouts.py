from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import sapien.core as sapien
import trimesh
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
)
from scipy.spatial.transform import Rotation
from transforms3d.euler import euler2quat

import plug_charger_causal_env  # noqa: F401 registers PlugChargerCausal-v1


def to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64)


def pose_to_matrix(raw_pose: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quat_wxyz_to_xyzw(raw_pose[3:7])).as_matrix()
    transform[:3, 3] = raw_pose[:3]
    return transform


def matrix_to_pose(transform: np.ndarray) -> np.ndarray:
    q_xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return np.concatenate([transform[:3, 3], quat_xyzw_to_wxyz(q_xyzw)])


def sapien_pose_to_raw(pose: sapien.Pose) -> np.ndarray:
    return np.concatenate([np.asarray(pose.p), np.asarray(pose.q)]).astype(np.float64)


def pose_error_action(current: np.ndarray, target: np.ndarray, pos_gain: float, rot_gain: float) -> np.ndarray:
    current_tf = pose_to_matrix(current)
    target_tf = pose_to_matrix(target)
    rel_rot = Rotation.from_matrix(target_tf[:3, :3] @ current_tf[:3, :3].T)
    pos_delta = np.clip((target[:3] - current[:3]) * pos_gain, -1.0, 1.0)
    rot_delta = np.clip(rel_rot.as_euler("xyz") * rot_gain / (2.0 * math.pi), -1.0, 1.0)
    return np.concatenate([pos_delta, rot_delta])


def obs_pose(obs: dict, name: str) -> np.ndarray:
    return to_numpy(obs["extra"][name])[0].astype(np.float64)


def obs_agent(obs: dict, name: str) -> np.ndarray:
    return to_numpy(obs["agent"][name])[0].astype(np.float64)


def rough_targets(obs: dict) -> dict[str, np.ndarray]:
    tcp = pose_to_matrix(obs_pose(obs, "tcp_pose"))
    charger = pose_to_matrix(obs_pose(obs, "charger_pose"))
    lift = np.eye(4, dtype=np.float64)
    lift[2, 3] = 0.04

    grasp_tf = tcp.copy()
    grasp_tf[:3, 3] = charger[:3, 3] + np.array([-0.02, 0.0, 0.035])
    pre_grasp_tf = grasp_tf.copy()
    pre_grasp_tf[:3, 3] += np.array([0.0, 0.0, 0.05])

    return {
        "pre_grasp": matrix_to_pose(pre_grasp_tf),
        "grasp": matrix_to_pose(grasp_tf),
        "lift": matrix_to_pose(grasp_tf @ lift),
        "charger_pose_at_plan": obs_pose(obs, "charger_pose"),
    }


def official_geometry_targets(obs: dict, env) -> dict[str, np.ndarray]:
    tcp = pose_to_matrix(obs_pose(obs, "tcp_pose"))
    base_env = env.unwrapped

    finger_length = 0.025
    charger_base_pose = base_env.charger_base_pose
    charger_base_size = np.array(base_env._base_size) * 2.0
    obb = trimesh.primitives.Box(
        extents=charger_base_size,
        transform=charger_base_pose.sp.to_transformation_matrix(),
    )

    approaching = np.array([0.0, 0.0, -1.0])
    target_closing = tcp[:3, 1]
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=finger_length,
    )
    grasp_pose = base_env.agent.build_grasp_pose(
        grasp_info["approaching"],
        grasp_info["closing"],
        grasp_info["center"],
    )
    grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0.0, np.deg2rad(15.0), 0.0))
    pre_grasp_pose = grasp_pose * sapien.Pose([0.0, 0.0, -0.05])
    lift_pose = grasp_pose * sapien.Pose([0.0, 0.0, -0.05])

    return {
        "pre_grasp": sapien_pose_to_raw(pre_grasp_pose),
        "grasp": sapien_pose_to_raw(grasp_pose),
        "lift": sapien_pose_to_raw(lift_pose),
        "charger_pose_at_plan": obs_pose(obs, "charger_pose"),
    }


def insertion_targets(obs: dict) -> dict[str, np.ndarray]:
    tcp = pose_to_matrix(obs_pose(obs, "tcp_pose"))
    charger = pose_to_matrix(obs_pose(obs, "charger_pose"))
    goal = pose_to_matrix(obs_pose(obs, "goal_pose"))
    offset = np.eye(4, dtype=np.float64)
    offset[0, 3] = -0.05
    return {
        "pre_insert": matrix_to_pose(goal @ offset @ np.linalg.inv(charger) @ tcp),
        "insert": matrix_to_pose(goal @ np.linalg.inv(charger) @ tcp),
    }


def geometric_targets(obs: dict, env, target_mode: str) -> dict[str, np.ndarray]:
    if target_mode == "rough":
        return rough_targets(obs)
    if target_mode == "official_geometry":
        return official_geometry_targets(obs, env)
    raise ValueError(f"Unknown target_mode: {target_mode}")


def append_step(buffers: dict, obs: dict, action: np.ndarray, reward, info: dict) -> None:
    buffers["tcp_pose"].append(obs_pose(obs, "tcp_pose"))
    buffers["charger_pose"].append(obs_pose(obs, "charger_pose"))
    buffers["receptacle_pose"].append(obs_pose(obs, "receptacle_pose"))
    buffers["goal_pose"].append(obs_pose(obs, "goal_pose"))
    buffers["qpos"].append(obs_agent(obs, "qpos"))
    buffers["qvel"].append(obs_agent(obs, "qvel"))
    buffers["actions"].append(action.astype(np.float32))
    buffers["reward"].append(float(to_numpy(reward)[0]))
    buffers["success"].append(bool(to_numpy(info["success"])[0]))
    buffers["obj_to_goal_dist"].append(float(to_numpy(info["obj_to_goal_dist"])[0]))
    buffers["obj_to_goal_angle"].append(float(to_numpy(info["obj_to_goal_angle"])[0]))


def rollout(env, seed: int, causal_delta: list[float], phase_steps: dict, gains: dict, target_mode: str) -> dict:
    obs, info = env.reset(seed=seed, options={"causal_delta": causal_delta})
    targets = geometric_targets(obs, env, target_mode)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    buffers = {
        "tcp_pose": [],
        "charger_pose": [],
        "receptacle_pose": [],
        "goal_pose": [],
        "qpos": [],
        "qvel": [],
        "actions": [],
        "reward": [],
        "success": [],
        "obj_to_goal_dist": [],
        "obj_to_goal_angle": [],
        "phase": [],
    }

    phases = [
        ("pre_grasp", "pre_grasp", 1.0),
        ("grasp", "grasp", 1.0),
        ("close_gripper", "grasp", -1.0),
        ("lift", "lift", -1.0),
        ("pre_insert", "pre_insert", -1.0),
        ("insert", "insert", -1.0),
        ("hold", "insert", -1.0),
    ]

    phase_names = [name for name, _, _ in phases]
    targets.update(insertion_targets(obs))
    insertion_targets_are_current = False
    done = False
    for phase_id, (phase_name, target_key, gripper) in enumerate(phases):
        if done:
            break
        if phase_name == "pre_insert" and not insertion_targets_are_current:
            targets.update(insertion_targets(obs))
            insertion_targets_are_current = True
        target = targets[target_key]
        for _ in range(phase_steps[phase_name]):
            tcp = obs_pose(obs, "tcp_pose")
            arm = pose_error_action(
                tcp,
                target,
                pos_gain=gains["pos_gain"],
                rot_gain=gains["rot_gain"],
            )
            action = np.concatenate([arm, np.asarray([gripper], dtype=np.float64)]).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            append_step(buffers, obs, action, reward, info)
            buffers["phase"].append(phase_id)
            if bool(to_numpy(terminated)[0]) or bool(to_numpy(truncated)[0]):
                done = True
                break

    out = {k: np.asarray(v) for k, v in buffers.items()}
    out["phase_names"] = phase_names
    out["targets"] = targets
    return out


def write_split(args, split: str, env) -> None:
    reset_json = args.reset_root / split / "reset_states.json"
    meta = json.loads(reset_json.read_text())
    out_dir = args.out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h5 = out_dir / "trajectory.h5"
    out_json = out_dir / "trajectory.json"

    phase_steps = {
        "pre_grasp": args.pre_grasp_steps,
        "grasp": args.grasp_steps,
        "close_gripper": args.close_steps,
        "lift": args.lift_steps,
        "pre_insert": args.pre_insert_steps,
        "insert": args.insert_steps,
        "hold": args.hold_steps,
    }
    gains = {"pos_gain": args.pos_gain, "rot_gain": args.rot_gain}
    episodes = []

    with h5py.File(out_h5, "w") as h5:
        h5.attrs["env_id"] = "PlugChargerCausal-v1"
        h5.attrs["source_type"] = "scripted_pd_ee_delta_pose_physics_rollout"
        h5.attrs["source_desc"] = (
            "True ManiSkill env.step physics rollout using a simple scripted "
            "pd_ee_delta_pose controller. Does not use mplib."
        )
        h5.attrs["control_mode"] = "pd_ee_delta_pose"
        h5.attrs["target_mode"] = args.target_mode
        h5.attrs["pose_format"] = "xyz + quaternion_wxyz"

        for sample in meta["samples"]:
            if args.limit is not None and len(episodes) >= args.limit:
                break
            ep = rollout(
                env,
                int(sample["seed"]),
                sample["causal_delta"],
                phase_steps,
                gains,
                args.target_mode,
            )
            ep_id = int(sample["sample_id"])
            group = h5.create_group(f"episode_{ep_id}")
            group.attrs["sample_id"] = ep_id
            group.attrs["generator"] = sample["generator"]
            group.attrs["seed"] = int(sample["seed"])
            group.attrs["phase_names"] = json.dumps(ep["phase_names"])
            group.create_dataset("causal_delta", data=np.asarray(sample["causal_delta"], dtype=np.float64))
            for key in [
                "tcp_pose",
                "charger_pose",
                "receptacle_pose",
                "goal_pose",
                "qpos",
                "qvel",
                "actions",
                "reward",
                "success",
                "obj_to_goal_dist",
                "obj_to_goal_angle",
                "phase",
            ]:
                group.create_dataset(key, data=ep[key])
            target_group = group.create_group("targets")
            for key, value in ep["targets"].items():
                target_group.create_dataset(key, data=value)

            episode_meta = {
                **sample,
                "episode_id": ep_id,
                "elapsed_steps": int(len(ep["tcp_pose"])),
                "success": bool(ep["success"][-1]) if len(ep["success"]) else False,
                "final_obj_to_goal_dist": float(ep["obj_to_goal_dist"][-1]) if len(ep["obj_to_goal_dist"]) else None,
                "final_obj_to_goal_angle": float(ep["obj_to_goal_angle"][-1]) if len(ep["obj_to_goal_angle"]) else None,
            }
            episodes.append(episode_meta)
            print(json.dumps({"split": split, **episode_meta}, sort_keys=True))

    out_json.write_text(
        json.dumps(
            {
                "env_id": "PlugChargerCausal-v1",
                "source_type": "scripted_pd_ee_delta_pose_physics_rollout",
                "source_desc": (
                    "True ManiSkill env.step physics rollout using a simple scripted "
                    "pd_ee_delta_pose controller. Does not use mplib."
                ),
                "control_mode": "pd_ee_delta_pose",
                "target_mode": args.target_mode,
                "pose_format": "xyz + quaternion_wxyz",
                "phase_steps": phase_steps,
                "gains": gains,
                "num_episodes": len(episodes),
                "episodes": episodes,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"split": split, "h5": str(out_h5), "json": str(out_json), "episodes": len(episodes)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-root", type=Path, default=Path("maniskill_spectrum/plug_charger_causal/reset_datasets"))
    parser.add_argument("--out-root", type=Path, default=Path("maniskill_spectrum/plug_charger_causal/physics_rollouts"))
    parser.add_argument("--splits", nargs="+", default=["isolated_grid", "train_mixed"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pre-grasp-steps", type=int, default=35)
    parser.add_argument("--grasp-steps", type=int, default=25)
    parser.add_argument("--close-steps", type=int, default=20)
    parser.add_argument("--lift-steps", type=int, default=25)
    parser.add_argument("--pre-insert-steps", type=int, default=45)
    parser.add_argument("--insert-steps", type=int, default=35)
    parser.add_argument("--hold-steps", type=int, default=15)
    parser.add_argument("--pos-gain", type=float, default=30.0)
    parser.add_argument("--rot-gain", type=float, default=1.0)
    parser.add_argument("--target-mode", choices=["rough", "official_geometry"], default="rough")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/home/rocos/sia/maniskill_spectrum/.matplotlib")
    env = gym.make(
        "PlugChargerCausal-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_backend="none",
    )
    try:
        for split in args.splits:
            write_split(args, split, env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
