from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import numpy as np
import sapien.core as sapien
import trimesh
from transforms3d.euler import euler2quat

import plug_charger_causal_env  # noqa: F401 registers PlugChargerCausal-v1
from grids import isolated_grid, mixed_training_grid, smoke_grid
from mani_skill.envs.tasks import PlugChargerEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
)
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.utils.wrappers.record import RecordEpisode


def solve(
    env: PlugChargerEnv,
    seed: int,
    causal_delta: list[float],
    debug: bool = False,
    vis: bool = False,
):
    env.reset(seed=seed, options={"causal_delta": causal_delta})
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )

    try:
        finger_length = 0.025
        base_env = env.unwrapped
        charger_base_pose = base_env.charger_base_pose
        charger_base_size = np.array(base_env._base_size) * 2

        obb = trimesh.primitives.Box(
            extents=charger_base_size,
            transform=charger_base_pose.sp.to_transformation_matrix(),
        )

        approaching = np.array([0, 0, -1])
        target_closing = (
            base_env.agent.tcp.pose.sp.to_transformation_matrix()[:3, 1]
        )
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=finger_length,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = base_env.agent.build_grasp_pose(approaching, closing, center)
        grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, np.deg2rad(15), 0))

        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        if planner.move_to_pose_with_screw(reach_pose) == -1:
            return -1
        if planner.move_to_pose_with_screw(grasp_pose) == -1:
            return -1
        planner.close_gripper()

        pre_insert_pose = (
            base_env.goal_pose.sp
            * sapien.Pose([-0.05, 0.0, 0.0])
            * base_env.charger.pose.sp.inv()
            * base_env.agent.tcp.pose.sp
        )
        insert_pose = (
            base_env.goal_pose.sp
            * base_env.charger.pose.sp.inv()
            * base_env.agent.tcp.pose.sp
        )
        if planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=0) == -1:
            return -1
        if planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=5) == -1:
            return -1
        return planner.move_to_pose_with_screw(insert_pose)
    finally:
        planner.close()


def iter_jobs(args) -> Iterable[dict]:
    if args.split == "smoke":
        base_rows = smoke_grid()
    elif args.split == "isolated_grid":
        base_rows = isolated_grid()
    elif args.split == "train_mixed":
        base_rows = mixed_training_grid(args.train_samples, args.seed)
    else:
        raise ValueError(args.split)

    for seed_offset in range(args.num_seeds):
        for intervention_index, item in enumerate(base_rows):
            job = dict(item)
            job["seed"] = args.seed + seed_offset
            job["seed_offset"] = seed_offset
            job["intervention_index"] = intervention_index
            yield job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["smoke", "isolated_grid", "train_mixed"],
        default="smoke",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=50)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("maniskill_spectrum/plug_charger_causal/datasets"),
    )
    parser.add_argument("--traj-name", default="trajectory")
    parser.add_argument("--only-count-success", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--render-mode", default=None)
    parser.add_argument("--render-backend", default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault(
        "MPLCONFIGDIR", "/home/rocos/sia/maniskill_spectrum/.matplotlib"
    )

    output_dir = args.output_root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    env = gym.make(
        "PlugChargerCausal-v1",
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        reward_mode="sparse",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
    )
    env = RecordEpisode(
        env,
        output_dir=str(output_dir),
        trajectory_name=args.traj_name,
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc=(
            "Panda motion-planning trajectories with controlled "
            "PlugCharger receptacle interventions"
        ),
        video_fps=30,
        record_reward=True,
        save_on_reset=False,
        clean_on_close=True,
    )

    attempted = 0
    saved = 0
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for job in iter_jobs(args):
            attempted += 1
            res = solve(
                env,
                seed=job["seed"],
                causal_delta=job["causal_delta"],
                vis=args.vis,
            )
            success = bool(res != -1 and res[-1]["success"].item())
            elapsed_steps = int(res[-1]["elapsed_steps"].item()) if res != -1 else 0

            save_episode = success or not args.only_count_success
            if save_episode:
                env.flush_trajectory()
                if args.save_video:
                    env.flush_video()
                episode_id = saved
                saved += 1
            else:
                env.flush_trajectory(save=False)
                if args.save_video:
                    env.flush_video(save=False)
                episode_id = None

            record = dict(
                job,
                episode_id=episode_id,
                success=success,
                elapsed_steps=elapsed_steps,
                saved=save_episode,
            )
            manifest.write(json.dumps(record, sort_keys=True) + "\n")
            manifest.flush()
            print(
                json.dumps(
                    {
                        "attempted": attempted,
                        "saved": saved,
                        "success": success,
                        "episode_id": episode_id,
                        "generator": job["generator"],
                        "delta": job["causal_delta"],
                    },
                    sort_keys=True,
                )
            )

    env.close()
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "attempted": attempted,
                "saved": saved,
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
