from __future__ import annotations

"""Collect the isolated-intervention suite under perturbed agent variants.

This collector is the controller/planner/speed-invariance companion to
``collect_phase_switch_rollouts.py``.  It reuses the frozen collector's solver,
trace wrapper, intervention design, and episode writer, and only adds the three
agent-side knobs the invariance claim must be robust to:

* ``--arm-stiffness`` / ``--arm-damping``  (Panda PD gains; defaults 1e3 / 1e2),
* ``--planner`` (``screw`` = screw-motion with RRTConnect fallback; ``rrt`` =
  force RRTConnect),
* ``--speed`` (multiplier on the solver's joint velocity/acceleration limits).

Only the isolated + zero-intervention conditions are collected (``mixed-samples
0``): the empirical isolated profile alpha_yaw(s) is identified from the 8
held-out isolated interventions + the baseline, so the 30 mixed training
trajectories used for the full Pdiag fit are unnecessary here.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import sapien
import torch

import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_env  # noqa: F401
from mani_skill.agents.robots import Panda
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)

from collect_phase_switch_rollouts import (
    PHASES,
    EpisodeFinished,
    PhaseSwitchTraceWrapper,
    intervention_rows,
    npy,
    write_episode,
)


def solve(
    env: PhaseSwitchTraceWrapper,
    planner_mode: str = "screw",
    joint_vel_limits: float = 0.5,
    joint_acc_limits: float = 0.5,
    keyed_yaw_override: float | None = None,
    post_clear_yaw: float = 0.0,
    stop_after_phase: str | None = None,
):
    base = env.unwrapped
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=joint_vel_limits,
        joint_acc_limits=joint_acc_limits,
    )
    try:
        peg_position = base.peg.pose.sp.p
        grasp_pose = base.agent.build_grasp_pose(
            approaching=np.array([0.0, 0.0, -1.0]),
            closing=np.array([0.0, 1.0, 0.0]),
            center=peg_position + np.array([0.0, 0.0, 0.020]),
        )

        def move_pose(target_pose, refine_steps=0):
            if planner_mode == "rrt":
                result = planner.move_to_pose_with_RRTConnect(
                    target_pose, refine_steps=refine_steps
                )
            else:
                result = planner.move_to_pose_with_screw(
                    target_pose, refine_steps=refine_steps
                )
                if result == -1:
                    result = planner.move_to_pose_with_RRTConnect(
                        target_pose, refine_steps=refine_steps
                    )
            if result == -1:
                raise RuntimeError("motion planning failed")
            return result

        def tcp_pose_for_peg_target(target_peg_pose):
            return target_peg_pose * base.peg.pose.sp.inv() * base.agent.tcp.pose.sp

        keyed_yaw = base.socket_yaw if keyed_yaw_override is None else keyed_yaw_override
        keyed_preinsert = base.target_pose_at(
            phase_switch_symmetry_env.PRE_ENTRY_PEG_Z, keyed_yaw
        ).sp
        keyed_entry = base.target_pose_at(
            phase_switch_symmetry_env.KEY_CLEAR_PEG_Z, keyed_yaw
        ).sp
        post_clear = base.target_pose_at(
            phase_switch_symmetry_env.KEY_CLEAR_PEG_Z, post_clear_yaw
        ).sp
        final_target = base.target_pose_at(
            phase_switch_symmetry_env.FINAL_PEG_Z, post_clear_yaw
        ).sp

        env.set_phase("reach")
        move_pose(grasp_pose * sapien.Pose([0.0, 0.0, -0.070]))
        env.set_phase("grasp")
        move_pose(grasp_pose)
        planner.close_gripper()
        env.set_phase("lift")
        move_pose(grasp_pose * sapien.Pose([0.0, 0.0, -0.095]))

        env.set_phase("align_keyed")
        move_pose(tcp_pose_for_peg_target(keyed_preinsert), refine_steps=5)
        move_pose(tcp_pose_for_peg_target(keyed_preinsert), refine_steps=5)
        env.set_phase("enter_key")
        keyed_goal = keyed_entry
        for peg_z in [0.108, 0.098, 0.088, float(keyed_goal.p[2])]:
            staged_goal = sapien.Pose(
                [keyed_goal.p[0], keyed_goal.p[1], peg_z], keyed_goal.q
            )
            move_pose(tcp_pose_for_peg_target(staged_goal), refine_steps=5)
        if stop_after_phase == "enter_key":
            return
        env.set_phase("unlock_yaw")
        move_pose(tcp_pose_for_peg_target(post_clear), refine_steps=8)
        env.set_phase("circular_insert")
        move_pose(tcp_pose_for_peg_target(final_target), refine_steps=10)
        move_pose(tcp_pose_for_peg_target(final_target), refine_steps=10)
    finally:
        planner.close()


def collect(
    rows: list[dict],
    output_path: Path,
    seed: int,
    retries_per_condition: int,
    arm_stiffness: float = 1e3,
    arm_damping: float = 1e2,
    planner_mode: str = "screw",
    speed: float = 1.0,
    robot_init_qpos_noise: float = 0.0,
):
    env_id = "KeyedCircularPhaseSwitch-v1"
    joint_vel_limits = 0.5 * speed
    joint_acc_limits = 0.5 * speed

    # Inject PD gains via the Panda class attributes (read lazily by the
    # `_controller_configs` property at robot build time).
    Panda.arm_stiffness = arm_stiffness
    Panda.arm_damping = arm_damping

    base = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_mode=None,
        robot_init_qpos_noise=robot_init_qpos_noise,
    )
    env = PhaseSwitchTraceWrapper(base)
    manifest = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as data_file:
        data_file.attrs["env_id"] = env_id
        data_file.attrs["task"] = "keyed_to_circular_phase_switch"
        data_file.attrs["source_type"] = "motionplanning_physx_phase_switch"
        data_file.attrs["control_mode"] = "pd_joint_pos"
        data_file.attrs["sim_backend"] = "physx_cpu"
        data_file.attrs["phase_names_json"] = json.dumps(PHASES)
        data_file.attrs["seed"] = seed
        data_file.attrs["retries_per_condition"] = retries_per_condition
        data_file.attrs["robot_init_qpos_noise"] = robot_init_qpos_noise
        data_file.attrs["arm_stiffness"] = arm_stiffness
        data_file.attrs["arm_damping"] = arm_damping
        data_file.attrs["planner_mode"] = planner_mode
        data_file.attrs["speed"] = speed
        data_file.attrs["joint_vel_limits"] = joint_vel_limits
        data_file.attrs["joint_acc_limits"] = joint_acc_limits
        data_file.attrs["python_version"] = platform.python_version()
        for package in ["numpy", "torch", "mani_skill", "mplib", "sapien"]:
            try:
                data_file.attrs[f"{package}_version"] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                data_file.attrs[f"{package}_version"] = "unknown"
        for label, source_path in {
            "environment": Path(phase_switch_symmetry_env.__file__),
            "collector": Path(__file__),
        }.items():
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            data_file.attrs[f"{label}_source_sha256"] = digest

        episode_id = 0
        for condition_id, row in enumerate(rows):
            for attempt_id in range(retries_per_condition):
                print(
                    f"[condition {condition_id + 1:03d}/{len(rows):03d} "
                    f"attempt {attempt_id + 1}/{retries_per_condition}] {row}",
                    flush=True,
                )
                episode_seed = seed + condition_id * 1009 + attempt_id
                np.random.seed(episode_seed)
                torch.manual_seed(episode_seed)
                env.reset(
                    seed=episode_seed,
                    options={"causal_delta": row["causal_delta"]},
                )
                env.start_trace()
                solver_error = None
                stop_reason = "solver_returned"
                try:
                    solve(
                        env,
                        planner_mode=planner_mode,
                        joint_vel_limits=joint_vel_limits,
                        joint_acc_limits=joint_acc_limits,
                    )
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
                group.attrs["condition_id"] = condition_id
                group.attrs["attempt_id"] = attempt_id
                group.attrs["episode_seed"] = episode_seed
                write_episode(
                    group, row, env.trace, solver_error, stop_reason=stop_reason
                )
                manifest.append(
                    dict(
                        episode_id=episode_id,
                        condition_id=condition_id,
                        attempt_id=attempt_id,
                        episode_seed=episode_seed,
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

        data_file.attrs["condition_count"] = len(rows)
        data_file.attrs["episode_count"] = len(manifest)
        data_file.attrs["success_count"] = sum(row["success"] for row in manifest)
    env.close()
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as manifest_file:
        json.dump(
            dict(
                env_id=env_id,
                task="keyed_to_circular_phase_switch",
                phases=PHASES,
                seed=seed,
                retries_per_condition=retries_per_condition,
                robot_init_qpos_noise=robot_init_qpos_noise,
                arm_stiffness=arm_stiffness,
                arm_damping=arm_damping,
                planner_mode=planner_mode,
                speed=speed,
                joint_vel_limits=joint_vel_limits,
                joint_acc_limits=joint_acc_limits,
                episodes=manifest,
            ),
            manifest_file,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--retries-per-condition", type=int, default=3)
    parser.add_argument("--robot-init-qpos-noise", type=float, default=0.0)
    parser.add_argument("--arm-stiffness", type=float, default=1e3)
    parser.add_argument("--arm-damping", type=float, default=1e2)
    parser.add_argument("--planner", choices=["screw", "rrt"], default="screw")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--mixed-samples", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    rows = intervention_rows(args.mixed_samples, args.seed)
    if args.smoke:
        rows = [rows[2]]  # zero-intervention baseline condition
    retries = 1 if args.smoke else args.retries_per_condition
    collect(
        rows,
        args.output,
        args.seed,
        retries,
        arm_stiffness=args.arm_stiffness,
        arm_damping=args.arm_damping,
        planner_mode=args.planner,
        speed=args.speed,
        robot_init_qpos_noise=args.robot_init_qpos_noise,
    )


if __name__ == "__main__":
    main()
