from __future__ import annotations

"""Collect the planar-push rollouts (supplement, own module).

Two arms, one env (PlanarPush-v1), distinguished by the goal_heading flag:
    heading_push  : block aligns in-plane position AND heading  -> [1,1,0,0,0,1]
    free_yaw_push : block aligns in-plane position only         -> [1,1,0,0,0,0]

The solver is a GUIDED PUSH (grasp + slide): the gripper grasps the block by its
short axis, slides it across the table to the target in-plane position, then (for
heading_push) rotates the wrist about the table normal to set the heading, and
releases. This supersedes the initial non-prehensive point-push, whose align-phase
yaw control was unstable (see probe_planar_push.py / experiment.json phase0_gate).
The frozen 6-vector manifest planar_push_fixed_contexts.json is the single source
of truth; its sha256 is checked against planar_push_experiment.json.

Phase codes: reach=3, push=4, align=5, retract=6 — the SAME raw codes as the
frozen task so complete()/usable()/progress_grid work unchanged.
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
from transforms3d.quaternions import qmult

import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_planar_push_env  # noqa: F401 - registers the env
from collect_phase_switch_rollouts import EpisodeFinished
from mani_skill.agents.robots.panda import Panda
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from phase_switch_symmetry_planar_push_env import (
    BLOCK_HALF_Z,
    _heading_of,
    _wrap_pi,
)

# Firm gripper (unchanged from rotated-axis / symmetry-transfer / SE(3)).
Panda.gripper_stiffness = 2.5e3
Panda.gripper_force_limit = 150.0

PUSH_PHASES = {"initial": -1, "reach": 3, "push": 4, "align": 5, "retract": 6}
ENV_ID = "PlanarPush-v1"

# Tunables for the scripted guided push (grasp + slide).
PUSH_APPROACH_Z = 0.06   # safe approach height above the grasp (TCP z offset)
PUSH_LIFT = 0.015        # m: lift the grasped block this far to clear table friction
PUSH_RETRACT_Z = 0.06    # TCP lift-away height on release
PUSH_ALIGN_TOL = 0.02    # rad: wrist yaw closed-loop tolerance in the align phase


def _rz_quat(angle):
    """Quaternion [w,x,y,z] for a rotation of ``angle`` about the world z-axis."""
    return np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])


def npy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class PlanarPushTraceWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.phase = "initial"
        self.trace = None

    def set_phase(self, phase: str):
        self.phase = phase

    def snapshot(self, zero_contact=False):
        base = self.unwrapped
        info = base.evaluate()
        # Net contact force on the block (the gripper is the only thing that
        # touches it); get_net_contact_forces works on an Actor, unlike the
        # robot Articulation which has no _bodies for get_pairwise_contact_forces.
        contact = base.block.get_net_contact_forces()
        contact_value = (
            np.zeros(3, dtype=np.float64) if zero_contact else npy(contact)[0].astype(np.float64)
        )
        return dict(
            tcp_pose=npy(base.agent.tcp.pose.raw_pose)[0].astype(np.float64),
            block_pose=npy(base.block.pose.raw_pose)[0].astype(np.float64),
            target_pose=npy(base.target.pose.raw_pose)[0].astype(np.float64),
            goal_pose=npy(base.goal_pose.raw_pose)[0].astype(np.float64),
            qpos=npy(base.agent.robot.get_qpos())[0].astype(np.float64),
            qvel=npy(base.agent.robot.get_qvel())[0].astype(np.float64),
            contact_force=contact_value,
            success=bool(npy(info["success"]).reshape(-1)[0]),
            obj_to_goal_dist=float(npy(info["obj_to_goal_dist"]).reshape(-1)[0]),
            heading_err=float(npy(info["heading_err"]).reshape(-1)[0]),
            solver_phase=PUSH_PHASES[self.phase],
        )

    def start_trace(self):
        self.phase = "initial"
        self.trace = dict(
            states=[self.snapshot(zero_contact=True)],
            actions=[],
            rewards=[],
            terminated=[],
            truncated=[],
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.trace is not None:
            self.trace["actions"].append(np.asarray(action, dtype=np.float64).copy())
            self.trace["rewards"].append(float(npy(reward).reshape(-1)[0]))
            self.trace["terminated"].append(bool(npy(terminated).reshape(-1)[0]))
            self.trace["truncated"].append(bool(npy(truncated).reshape(-1)[0]))
            self.trace["states"].append(self.snapshot())
        terminated_value = bool(npy(terminated).reshape(-1)[0])
        truncated_value = bool(npy(truncated).reshape(-1)[0])
        if terminated_value or truncated_value:
            raise EpisodeFinished(terminated_value, truncated_value)
        return obs, reward, terminated, truncated, info


def write_push_episode(group, row, trace, solver_error, stop_reason="solver_returned"):
    group.attrs["generator"] = row["generator"]
    group.attrs["solver_error"] = "" if solver_error is None else solver_error
    group.attrs["stop_reason"] = stop_reason
    group.create_dataset(
        "causal_delta", data=np.asarray(row["causal_delta"], dtype=np.float64)
    )
    states = trace["states"]
    for key in [
        "tcp_pose",
        "block_pose",
        "target_pose",
        "goal_pose",
        "qpos",
        "qvel",
        "contact_force",
        "success",
        "obj_to_goal_dist",
        "heading_err",
        "solver_phase",
    ]:
        group.create_dataset(key, data=np.asarray([state[key] for state in states]))
    group.create_dataset("actions", data=np.asarray(trace["actions"]))
    group.create_dataset("rewards", data=np.asarray(trace["rewards"]))
    group.create_dataset("terminated", data=np.asarray(trace["terminated"], bool))
    group.create_dataset("truncated", data=np.asarray(trace["truncated"], bool))


def _grasp_pose(base, xy, tcp_z=BLOCK_HALF_Z):
    """Overhead grasp: fingers point DOWN (approaching=[0,0,-1]) and close along
    the block's short axis (world y at heading 0), TCP at the block centre. The
    grasp constrains the block's in-plane pose to the wrist, so sliding the wrist
    to the target position and rotating it about the table normal set the block's
    position and heading deterministically."""
    return base.agent.build_grasp_pose(
        approaching=np.array([0.0, 0.0, -1.0], dtype=np.float64),
        closing=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        center=np.array([xy[0], xy[1], tcp_z], dtype=np.float64),
    )


def solve_planar_push(env: PlanarPushTraceWrapper):
    """Guided push (grasp + slide): reach (overhead grasp of the block by its short
    axis) -> push (lift + slide the grasped block to the target in-plane position)
    -> align (heading arm only: rotate the wrist about the table normal to the
    target heading) -> retract (lower + release + lift away). The block's heading
    is corrected only when goal_heading is True (heading_push); free_yaw_push keeps
    the block at its grasp heading (0)."""
    base = env.unwrapped
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
        def move_pose(target_pose, refine_steps=0):
            result = planner.move_to_pose_with_screw(target_pose, refine_steps=refine_steps)
            if result == -1:
                result = planner.move_to_pose_with_RRTConnect(target_pose, refine_steps=refine_steps)
            if result == -1:
                raise RuntimeError("motion planning failed")
            return result

        goal_xy = base.block_goal_position()[:2].copy()
        block_xy0 = base.block.pose.sp.p[:2].copy()
        grasp = _grasp_pose(base, block_xy0)

        env.set_phase("reach")
        move_pose(grasp * sapien.Pose([0.0, 0.0, -PUSH_APPROACH_Z]))
        move_pose(grasp, refine_steps=2)
        planner.close_gripper()

        env.set_phase("push")
        move_pose(grasp * sapien.Pose([0.0, 0.0, -PUSH_LIFT]))
        slide_pose = sapien.Pose(
            p=np.array([goal_xy[0], goal_xy[1], BLOCK_HALF_Z + PUSH_LIFT]),
            q=grasp.q,
        )
        move_pose(slide_pose, refine_steps=3)

        env.set_phase("align")
        if base.goal_heading:
            target_heading = float(base.causal_delta[5])
            # Closed-loop wrist yaw: re-read the grasped block's heading each pass
            # so grasp compliance in the rotation does not accumulate error.
            for _ in range(3):
                block_heading = _heading_of(base.block.pose.sp.q)
                delta = _wrap_pi(target_heading - block_heading)
                if abs(delta) < PUSH_ALIGN_TOL:
                    break
                current = base.agent.tcp.pose.sp
                new_q = qmult(_rz_quat(delta), current.q)
                move_pose(sapien.Pose(p=current.p, q=new_q), refine_steps=4)

        env.set_phase("retract")
        current_q = base.agent.tcp.pose.sp.q
        move_pose(
            sapien.Pose(p=np.array([goal_xy[0], goal_xy[1], BLOCK_HALF_Z]), q=current_q),
            refine_steps=3,
        )
        planner.open_gripper()
        move_pose(
            sapien.Pose(
                p=np.array([goal_xy[0], goal_xy[1], BLOCK_HALF_Z + PUSH_RETRACT_Z]),
                q=current_q,
            ),
        )
    finally:
        planner.close()


def collect_planar_push(rows, output_path, seed, orientation, retries_per_condition,
                        task_anchor, goal_heading, context_manifest_path=None,
                        robot_init_qpos_noise=0.0):
    base = gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_mode=None,
        robot_init_qpos_noise=robot_init_qpos_noise,
        orientation=orientation,
        task_anchor=task_anchor,
        goal_heading=goal_heading,
    )
    env = PlanarPushTraceWrapper(base)
    manifest = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as data_file:
        data_file.attrs["env_id"] = ENV_ID
        data_file.attrs["task"] = "planar_push_heading" if goal_heading else "planar_push_free_yaw"
        data_file.attrs["source_type"] = "motionplanning_physx_planar_push"
        data_file.attrs["orientation"] = np.asarray(orientation, dtype=np.float64)
        data_file.attrs["orientation_json"] = json.dumps(
            np.asarray(orientation, dtype=np.float64).tolist()
        )
        data_file.attrs["task_anchor"] = np.asarray(task_anchor, dtype=np.float64)
        data_file.attrs["goal_heading"] = int(bool(goal_heading))
        data_file.attrs["phase_names_json"] = json.dumps(PUSH_PHASES)
        data_file.attrs["control_mode"] = "pd_joint_pos"
        data_file.attrs["sim_backend"] = "physx_cpu"
        data_file.attrs["seed"] = seed
        data_file.attrs["retries_per_condition"] = retries_per_condition
        data_file.attrs["robot_init_qpos_noise"] = robot_init_qpos_noise
        if context_manifest_path is not None:
            context_manifest_path = context_manifest_path.resolve()
            data_file.attrs["context_manifest"] = str(context_manifest_path)
            data_file.attrs["context_manifest_sha256"] = hashlib.sha256(
                context_manifest_path.read_bytes()
            ).hexdigest()
        for label, source_path in {
            "environment": Path(phase_switch_symmetry_planar_push_env.__file__),
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
                    f"attempt {attempt_id + 1}/{retries_per_condition}] {row['generator']}",
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
                    solve_planar_push(env)
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
                # Align (5) and retract (6) are conditional: align is skipped for
                # free_yaw or when the heading already matches, and retract is
                # skipped whenever success terminates the episode mid-push. A
                # successful push therefore often ends at phase 4, so "complete"
                # means the push was actually attempted (reach -> push).
                complete = 4 in phase_set
                print(
                    f"  steps={steps} success={success} max_contact={forces.max():.3f} "
                    f"phases={phase_set} final_dist={states[-1]['obj_to_goal_dist']:.4f} "
                    f"heading_err={states[-1]['heading_err']:.4f}",
                    flush=True,
                )
                group = data_file.create_group(f"episode_{episode_id}")
                group.attrs["condition_id"] = condition_id
                group.attrs["attempt_id"] = attempt_id
                group.attrs["episode_seed"] = episode_seed
                write_push_episode(group, row, env.trace, solver_error, stop_reason=stop_reason)
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
                if success:
                    break
        data_file.attrs["condition_count"] = len(rows)
        data_file.attrs["episode_count"] = len(manifest)
        data_file.attrs["success_count"] = sum(row["success"] for row in manifest)
    env.close()
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as manifest_file:
        json.dump(
            dict(
                env_id=ENV_ID,
                task="planar_push_heading" if goal_heading else "planar_push_free_yaw",
                seed=seed,
                orientation=np.asarray(orientation, dtype=np.float64).tolist(),
                goal_heading=bool(goal_heading),
                retries_per_condition=retries_per_condition,
                phases=PUSH_PHASES,
                episodes=manifest,
            ),
            manifest_file,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--orientation-quat", nargs=4, type=float, default=[1.0, 0.0, 0.0, 0.0]
    )
    parser.add_argument("--task-anchor", nargs=3, type=float, default=[0.10, 0.0, 0.01])
    parser.add_argument(
        "--context-manifest",
        type=Path,
        default=Path(
            "phase_switch_symmetry_multiseed/planar_push/planar_push_fixed_contexts.json"
        ),
    )
    parser.add_argument("--experiment", type=Path, default=None)
    parser.add_argument("--arm", choices=["heading_push", "free_yaw_push"], default="heading_push")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--retries-per-condition", type=int, default=5)
    parser.add_argument("--robot-init-qpos-noise", type=float, default=0.01)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    orientation = np.asarray(args.orientation_quat, dtype=np.float64)
    orientation = orientation / np.linalg.norm(orientation)
    task_anchor = np.asarray(args.task_anchor, dtype=np.float64)

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
        if generator not in {"baseline", "du", "dv", "dw", "roll", "pitch", "yaw", "mixed"}:
            raise ValueError(f"unsupported generator: {generator}")
        if causal_delta.shape != (6,) or not np.isfinite(causal_delta).all():
            raise ValueError("every causal_delta must be a finite 6-vector")
        normalized_rows.append(dict(generator=generator, causal_delta=causal_delta.tolist()))
    rows = normalized_rows
    if args.smoke:
        zero = [row for row in rows if np.linalg.norm(np.asarray(row["causal_delta"])) < 1e-12]
        rows = zero[:1] if zero else [rows[0]]
    retries = 1 if args.smoke else args.retries_per_condition
    collect_planar_push(
        rows, args.output, args.seed, orientation, retries, task_anchor,
        args.arm == "heading_push",
        context_manifest_path=args.context_manifest,
        robot_init_qpos_noise=args.robot_init_qpos_noise,
    )


if __name__ == "__main__":
    main()
