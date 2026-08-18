from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


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


def pose_series_between(start: np.ndarray, end: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 0:
        return np.zeros((0, 7), dtype=np.float64)
    if steps == 1:
        return end[None].astype(np.float64)

    times = np.linspace(0.0, 1.0, steps)
    positions = (1.0 - times[:, None]) * start[:3] + times[:, None] * end[:3]
    rotations = Rotation.from_quat(
        np.stack([quat_wxyz_to_xyzw(start[3:7]), quat_wxyz_to_xyzw(end[3:7])])
    )
    slerp = Slerp([0.0, 1.0], rotations)
    quats_wxyz = np.asarray([quat_xyzw_to_wxyz(q) for q in slerp(times).as_quat()])
    return np.hstack([positions, quats_wxyz])


def constant_series(raw_pose: np.ndarray, steps: int) -> np.ndarray:
    return np.repeat(raw_pose[None].astype(np.float64), steps, axis=0)


def get_pose(sample: h5py.Group, name: str) -> np.ndarray:
    return sample["obs"]["extra"][name][0].astype(np.float64)


def get_agent(sample: h5py.Group, name: str) -> np.ndarray:
    return sample["obs"]["agent"][name][0].astype(np.float64)


def geometric_targets(sample: h5py.Group) -> dict[str, np.ndarray]:
    tcp = pose_to_matrix(get_pose(sample, "tcp_pose"))
    charger = pose_to_matrix(get_pose(sample, "charger_pose"))
    goal = pose_to_matrix(get_pose(sample, "goal_pose"))

    offset = np.eye(4, dtype=np.float64)
    offset[0, 3] = -0.05

    return {
        "approach_start_tcp": matrix_to_pose(tcp),
        "pre_insert_target": matrix_to_pose(goal @ offset @ np.linalg.inv(charger) @ tcp),
        "insert_target": matrix_to_pose(goal @ np.linalg.inv(charger) @ tcp),
    }


def generate_episode(sample: h5py.Group, hold_steps: int, align_steps: int, insert_steps: int) -> dict:
    poses = {
        "tcp": get_pose(sample, "tcp_pose"),
        "charger": get_pose(sample, "charger_pose"),
        "receptacle": get_pose(sample, "receptacle_pose"),
        "goal": get_pose(sample, "goal_pose"),
    }
    targets = geometric_targets(sample)

    # Phase 0 is intentionally invariant to socket perturbations: before alignment,
    # the official solver reaches/grasps the charger, whose pose is unchanged here.
    tcp_hold = constant_series(targets["approach_start_tcp"], hold_steps)
    tcp_align = pose_series_between(
        targets["approach_start_tcp"], targets["pre_insert_target"], align_steps + 1
    )[1:]
    tcp_insert = pose_series_between(
        targets["pre_insert_target"], targets["insert_target"], insert_steps + 1
    )[1:]
    tcp_pose = np.vstack([tcp_hold, tcp_align, tcp_insert])
    steps = len(tcp_pose)

    qpos = np.repeat(get_agent(sample, "qpos")[None], steps, axis=0)
    qvel = np.repeat(get_agent(sample, "qvel")[None], steps, axis=0)

    # Minimal geometric action: finite difference of tcp x/y/z and yaw.
    action = np.zeros((max(steps - 1, 0), 4), dtype=np.float64)
    if steps > 1:
        action[:, :3] = np.diff(tcp_pose[:, :3], axis=0)
        yaw = np.asarray(
            [
                Rotation.from_quat(quat_wxyz_to_xyzw(p[3:7])).as_euler("zyx")[0]
                for p in tcp_pose
            ]
        )
        action[:, 3] = (np.diff(yaw) + math.pi) % (2.0 * math.pi) - math.pi

    phase = np.concatenate(
        [
            np.zeros(hold_steps, dtype=np.int64),
            np.ones(align_steps, dtype=np.int64),
            np.full(insert_steps, 2, dtype=np.int64),
        ]
    )

    return {
        "tcp_pose": tcp_pose,
        "charger_pose": constant_series(poses["charger"], steps),
        "receptacle_pose": constant_series(poses["receptacle"], steps),
        "goal_pose": constant_series(poses["goal"], steps),
        "qpos": qpos,
        "qvel": qvel,
        "actions": action,
        "phase": phase,
        "phase_name": ["approach_hold", "align_to_pre_insert", "insert"],
    }


def copy_dataset(
    reset_h5: Path,
    reset_json: Path,
    out_h5: Path,
    out_json: Path,
    hold_steps: int,
    align_steps: int,
    insert_steps: int,
) -> None:
    reset_meta = json.loads(reset_json.read_text())
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    episodes = []

    with h5py.File(reset_h5, "r") as src, h5py.File(out_h5, "w") as dst:
        dst.attrs["env_id"] = "PlugChargerCausal-v1"
        dst.attrs["source_type"] = "geometric_waypoint_rollout"
        dst.attrs["source_desc"] = (
            "Synthetic TCP trajectories interpolated through official PlugCharger "
            "solver geometry targets. Not a physics rollout."
        )
        dst.attrs["pose_format"] = "xyz + quaternion_wxyz"

        for sample_info in reset_meta["samples"]:
            sample_id = int(sample_info["sample_id"])
            sample = src[f"sample_{sample_id}"]
            episode = generate_episode(sample, hold_steps, align_steps, insert_steps)

            group = dst.create_group(f"episode_{sample_id}")
            group.attrs["sample_id"] = sample_id
            group.attrs["seed"] = int(sample_info["seed"])
            group.attrs["generator"] = sample_info["generator"]
            group.create_dataset(
                "causal_delta",
                data=np.asarray(sample_info["causal_delta"], dtype=np.float64),
            )
            for key, value in episode.items():
                if key == "phase_name":
                    group.attrs[key] = json.dumps(value)
                else:
                    group.create_dataset(key, data=value)

            episodes.append(
                {
                    **sample_info,
                    "episode_id": sample_id,
                    "elapsed_steps": len(episode["tcp_pose"]),
                    "source_type": "geometric_waypoint_rollout",
                }
            )

    out_json.write_text(
        json.dumps(
            {
                "env_id": "PlugChargerCausal-v1",
                "source_type": "geometric_waypoint_rollout",
                "source_desc": (
                    "Synthetic TCP trajectories interpolated through official "
                    "PlugCharger solver geometry targets. Not a physics rollout."
                ),
                "pose_format": "xyz + quaternion_wxyz",
                "phase_names": ["approach_hold", "align_to_pre_insert", "insert"],
                "phase_steps": {
                    "hold_steps": hold_steps,
                    "align_steps": align_steps,
                    "insert_steps": insert_steps,
                },
                "num_episodes": len(episodes),
                "episodes": episodes,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset-root",
        type=Path,
        default=Path("maniskill_spectrum/plug_charger_causal/reset_datasets"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("maniskill_spectrum/plug_charger_causal/geometric_rollouts"),
    )
    parser.add_argument("--hold-steps", type=int, default=20)
    parser.add_argument("--align-steps", type=int, default=40)
    parser.add_argument("--insert-steps", type=int, default=20)
    parser.add_argument(
        "--splits", nargs="+", default=["isolated_grid", "train_mixed"]
    )
    args = parser.parse_args()

    for split in args.splits:
        reset_h5 = args.reset_root / split / "reset_states.h5"
        reset_json = args.reset_root / split / "reset_states.json"
        out_h5 = args.out_root / split / "trajectory.h5"
        out_json = args.out_root / split / "trajectory.json"
        copy_dataset(
            reset_h5,
            reset_json,
            out_h5,
            out_json,
            args.hold_steps,
            args.align_steps,
            args.insert_steps,
        )
        print(json.dumps({"split": split, "h5": str(out_h5), "json": str(out_json)}))


if __name__ == "__main__":
    main()
