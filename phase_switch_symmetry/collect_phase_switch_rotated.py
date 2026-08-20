from __future__ import annotations

"""Collect KeyedCircularPhaseSwitch rollouts at a global task orientation.

The task-local fixture is embedded in the robot workspace by a global rigid
transform G_Q = (Q, t_Q): Q controls orientation (the insertion axis points along
Q e_z) and ``task_anchor`` controls translation (the world placement of the
socket nominal center). Interventions remain defined in the TASK-LOCAL generator
basis [du, dv, d_axial].

This is the rotated-axis geometric-consistency experiment: the canonical
generator law P^local(s) should be invariant under Q.
"""

import argparse
import hashlib
import json
import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import sapien
import torch

import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_env  # noqa: F401
from collect_phase_switch_rollouts import (
    EpisodeFinished,
    PhaseSwitchTraceWrapper,
    intervention_rows,
    write_episode,
)
from mani_skill.agents.robots.panda import Panda
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)

# Firm gripper: the default ~23 N squeeze slips under the ~22 N axial contact
# during horizontal insertion, so the peg ends ~3-5 cm short. Raising the
# stiffness/force limit gives enough friction to hold the peg through the gate.
Panda.gripper_stiffness = 2.5e3
Panda.gripper_force_limit = 150.0


def solve_rotated(env: PhaseSwitchTraceWrapper, align_yaw=None):
    base = env.unwrapped
    axis = base.insertion_axis  # [3] unit vector
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )
    try:
        peg_position = base.peg.pose.sp.p
        # End-on grasp along the insertion axis: the gripper approaches from the
        # +axis side and closes perpendicular to it. This keeps the gripper behind
        # the peg during insertion (proven reachable in the diagnosis), unlike a
        # top-down grasp whose above-peg TCP is unreachable at the socket.
        grasp_pose = base.agent.build_grasp_pose(
            approaching=-axis,
            closing=base.Q_mat @ np.array([0.0, 1.0, 0.0]),
            center=peg_position + 0.020 * axis,
        )

        def move_pose(target_pose, refine_steps=0):
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

        keyed_yaw = base.socket_yaw if align_yaw is None else float(align_yaw)
        keyed_preinsert = base.target_pose_at(
            phase_switch_symmetry_env.PRE_ENTRY_PEG_Z, keyed_yaw
        ).sp
        keyed_entry = base.target_pose_at(
            phase_switch_symmetry_env.KEY_CLEAR_PEG_Z, keyed_yaw
        ).sp
        post_clear = base.target_pose_at(
            phase_switch_symmetry_env.KEY_CLEAR_PEG_Z, 0.0
        ).sp
        final_target = base.target_pose_at(
            phase_switch_symmetry_env.FINAL_PEG_Z, 0.0
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
        # Incremental Cartesian insertion: step the key through the narrow gate
        # in small local-axis increments so the screw planner never has to cross
        # the keyed gate in one long motion. Applied uniformly to all Q.
        insertion_step = 0.003  # 3 mm along the insertion axis
        n_steps = int(np.ceil(
            (phase_switch_symmetry_env.PRE_ENTRY_PEG_Z
             - phase_switch_symmetry_env.KEY_CLEAR_PEG_Z) / insertion_step
        ))
        for i in range(1, n_steps + 1):
            peg_depth = max(
                phase_switch_symmetry_env.PRE_ENTRY_PEG_Z - i * insertion_step,
                phase_switch_symmetry_env.KEY_CLEAR_PEG_Z,
            )
            staged_goal = base.target_pose_at(peg_depth, keyed_yaw).sp
            move_pose(tcp_pose_for_peg_target(staged_goal), refine_steps=3)
        env.set_phase("unlock_yaw")
        move_pose(tcp_pose_for_peg_target(post_clear), refine_steps=8)
        env.set_phase("circular_insert")
        move_pose(tcp_pose_for_peg_target(final_target), refine_steps=10)
        move_pose(tcp_pose_for_peg_target(final_target), refine_steps=10)
    finally:
        planner.close()


def solve_circular(env: PhaseSwitchTraceWrapper, yaw_mode="honest", placebo_yaw=np.deg2rad(15.0)):
    """Circular-task solver: the same skeleton as solve_rotated, but the
    align/enter yaw is a FIXED schedule independent of the intervention d_axial.

    honest  : align_yaw = 0, so the peg stays at nominal yaw and d_axial is a
              pure gauge DOF (the task-local trajectory is yaw-invariant).
    placebo : align_yaw = placebo_yaw, a fixed sweep independent of d_axial, so
              yaw rotation IS present in the marginal trajectory but carries no
              causal response to d_axial.
    """
    align_yaw = 0.0 if yaw_mode == "honest" else float(placebo_yaw)
    solve_rotated(env, align_yaw=align_yaw)


def collect_rotated(rows, output_path, seed, orientation, retries_per_condition, task_anchor, context_manifest_path=None, robot_init_qpos_noise=0.0, yaw_mode="keyed"):
    env_id = "KeyedCircularPhaseSwitch-v1" if yaw_mode == "keyed" else "CircularPhaseSwitch-v1"
    task = {
        "keyed": "keyed_to_circular_phase_switch_rotated",
        "honest": "circular_phase_switch_honest",
        "placebo": "circular_phase_switch_placebo",
    }[yaw_mode]
    base = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_mode=None,
        robot_init_qpos_noise=robot_init_qpos_noise,
        orientation=orientation,
        task_anchor=task_anchor,
    )
    env = PhaseSwitchTraceWrapper(base)
    manifest = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as data_file:
        data_file.attrs["env_id"] = env_id
        data_file.attrs["task"] = task
        data_file.attrs["source_type"] = "motionplanning_physx_phase_switch_rotated"
        data_file.attrs["orientation"] = np.asarray(orientation, dtype=np.float64)
        data_file.attrs["orientation_json"] = json.dumps(
            np.asarray(orientation, dtype=np.float64).tolist()
        )
        data_file.attrs["task_anchor"] = np.asarray(task_anchor, dtype=np.float64)
        if context_manifest_path is not None:
            context_manifest_path = context_manifest_path.resolve()
            data_file.attrs["context_manifest"] = str(context_manifest_path)
            data_file.attrs["context_manifest_sha256"] = hashlib.sha256(
                context_manifest_path.read_bytes()
            ).hexdigest()
        data_file.attrs["control_mode"] = "pd_joint_pos"
        data_file.attrs["sim_backend"] = "physx_cpu"
        data_file.attrs["seed"] = seed
        for label, source_path in {
            "environment": Path(phase_switch_symmetry_env.__file__),
            "collector": Path(__file__),
        }.items():
            data_file.attrs[f"{label}_source_sha256"] = hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest()
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
                env.reset(seed=episode_seed, options={"causal_delta": row["causal_delta"]})
                env.start_trace()
                solver_error = None
                stop_reason = "solver_returned"
                try:
                    if yaw_mode == "keyed":
                        solve_rotated(env)
                    else:
                        solve_circular(env, yaw_mode=yaw_mode)
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
                write_episode(group, row, env.trace, solver_error, stop_reason=stop_reason)
                manifest.append(
                    dict(
                        episode_id=episode_id,
                        condition_id=condition_id,
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
        data_file.attrs["condition_count"] = len(rows)
        data_file.attrs["episode_count"] = len(manifest)
        data_file.attrs["success_count"] = sum(row["success"] for row in manifest)
    env.close()
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as manifest_file:
        json.dump(
            dict(
                env_id=env_id,
                task=task,
                seed=seed,
                orientation=np.asarray(orientation, dtype=np.float64).tolist(),
                retries_per_condition=retries_per_condition,
                episodes=manifest,
            ),
            manifest_file,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--orientation-quat",
        nargs=4,
        type=float,
        required=True,
        help="Global task rotation as a unit quaternion [w, x, y, z].",
    )
    parser.add_argument(
        "--task-anchor",
        nargs=3,
        type=float,
        required=True,
        help="World workspace placement of the socket nominal center [x, y, z].",
    )
    parser.add_argument("--mixed-samples", type=int, default=30)
    parser.add_argument(
        "--context-manifest",
        type=Path,
        default=None,
        help="Frozen JSON manifest (conditions list) replacing RNG-generated contexts.",
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=None,
        help="Preregistered rotated_axis_experiment.json whose manifest SHA256 is enforced.",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--retries-per-condition", type=int, default=3)
    parser.add_argument("--robot-init-qpos-noise", type=float, default=0.01)
    parser.add_argument(
        "--yaw-mode",
        choices=["keyed", "honest", "placebo"],
        default="keyed",
        help="keyed=existing keyed task; honest=circular flat-yaw; placebo=circular fixed-yaw-sweep.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    orientation = np.asarray(args.orientation_quat, dtype=np.float64)
    orientation = orientation / np.linalg.norm(orientation)
    task_anchor = np.asarray(args.task_anchor, dtype=np.float64)
    if args.context_manifest is None:
        rows = intervention_rows(args.mixed_samples, args.seed)
    else:
        manifest_sha256 = hashlib.sha256(args.context_manifest.read_bytes()).hexdigest()
        if args.experiment is not None:
            with args.experiment.open(encoding="utf-8") as experiment_file:
                experiment = json.load(experiment_file)
            preregistered = experiment.get("context_manifest_sha256")
            if preregistered is not None and manifest_sha256 != preregistered:
                raise RuntimeError(
                    f"context manifest SHA256 {manifest_sha256} does not match "
                    f"preregistered {preregistered}"
                )
        with args.context_manifest.open(encoding="utf-8") as manifest_file:
            context_manifest = json.load(manifest_file)
        raw_rows = context_manifest.get("conditions")
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError("contexts manifest must contain a nonempty conditions list")
        normalized_rows = []
        for expected_id, row in enumerate(raw_rows):
            if int(row.get("condition_id", expected_id)) != expected_id:
                raise ValueError("manifest condition ids must be contiguous from zero")
            generator = str(row["generator"])
            causal_delta = np.asarray(row["causal_delta"], dtype=np.float64)
            if generator not in {"yaw", "translation", "mixed"}:
                raise ValueError(f"unsupported generator: {generator}")
            if causal_delta.shape != (3,) or not np.isfinite(causal_delta).all():
                raise ValueError("every causal_delta must be a finite 3-vector")
            normalized_rows.append(
                dict(generator=generator, causal_delta=causal_delta.tolist())
            )
        rows = normalized_rows
    if args.smoke:
        zero = [
            row
            for row in rows
            if np.linalg.norm(np.asarray(row["causal_delta"])) < 1e-12
        ]
        rows = zero[:1] if zero else [rows[0]]
    retries = 1 if args.smoke else args.retries_per_condition
    collect_rotated(
        rows, args.output, args.seed, orientation, retries, task_anchor,
        context_manifest_path=args.context_manifest,
        robot_init_qpos_noise=args.robot_init_qpos_noise,
        yaw_mode=args.yaw_mode,
    )


if __name__ == "__main__":
    main()
