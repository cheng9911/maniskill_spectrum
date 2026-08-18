from __future__ import annotations

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
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


PHASES = {
    "initial": -1,
    "reach": 0,
    "grasp": 1,
    "lift": 2,
    "align_keyed": 3,
    "enter_key": 4,
    "unlock_yaw": 5,
    "circular_insert": 6,
}


class EpisodeFinished(RuntimeError):
    def __init__(self, terminated: bool, truncated: bool):
        self.terminated = terminated
        self.truncated = truncated
        reason = "terminated" if terminated else "truncated"
        super().__init__(reason)


def npy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class PhaseSwitchTraceWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.phase = "initial"
        self.trace = None

    def set_phase(self, phase: str):
        self.phase = phase

    def snapshot(self, zero_contact=False):
        base = self.unwrapped
        info = base.evaluate()
        contact = base.scene.get_pairwise_contact_forces(base.peg, base.socket)
        contact_value = np.zeros(3, dtype=np.float64) if zero_contact else npy(contact)[0].astype(np.float64)
        return dict(
            tcp_pose=npy(base.agent.tcp.pose.raw_pose)[0].astype(np.float64),
            peg_pose=npy(base.peg.pose.raw_pose)[0].astype(np.float64),
            socket_pose=npy(base.socket.pose.raw_pose)[0].astype(np.float64),
            keyed_entry_goal_pose=npy(base.keyed_entry_pose.raw_pose)[0].astype(np.float64),
            circular_transition_goal_pose=npy(
                base.circular_transition_pose.raw_pose
            )[0].astype(np.float64),
            goal_pose=npy(base.goal_pose.raw_pose)[0].astype(np.float64),
            qpos=npy(base.agent.robot.get_qpos())[0].astype(np.float64),
            qvel=npy(base.agent.robot.get_qvel())[0].astype(np.float64),
            contact_force=contact_value,
            success=bool(npy(info["success"]).reshape(-1)[0]),
            obj_to_goal_dist=float(npy(info["obj_to_goal_dist"]).reshape(-1)[0]),
            axis_angle_err=float(npy(info["axis_angle_err"]).reshape(-1)[0]),
            yaw_err=float(npy(info["yaw_err"]).reshape(-1)[0]),
            key_clearance_margin=float(
                npy(info["key_clearance_margin"]).reshape(-1)[0]
            ),
            solver_phase=PHASES[self.phase],
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


def solve(
    env: PhaseSwitchTraceWrapper,
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
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )
    try:
        peg_position = base.peg.pose.sp.p
        grasp_pose = base.agent.build_grasp_pose(
            approaching=np.array([0.0, 0.0, -1.0]),
            closing=np.array([0.0, 1.0, 0.0]),
            center=peg_position + np.array([0.0, 0.0, 0.020]),
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
        # Recompute from the measured PhysX peg pose to remove grasp compliance
        # error before entering the narrow keyed gate.
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


def intervention_rows(n_mixed: int, seed: int):
    rows = [
        dict(generator="yaw", causal_delta=[0.0, 0.0, np.deg2rad(deg)])
        for deg in [-30, -15, 0, 15, 30]
    ]
    rows.extend(
        dict(generator="translation", causal_delta=[dx, dy, 0.0])
        for dx, dy in [(-0.015, 0.0), (0.015, 0.0), (0.0, -0.015), (0.0, 0.015)]
    )
    rng = np.random.default_rng(seed)
    for _ in range(n_mixed):
        rows.append(
            dict(
                generator="mixed",
                causal_delta=[
                    float(rng.uniform(-0.012, 0.012)),
                    float(rng.uniform(-0.012, 0.012)),
                    float(rng.uniform(np.deg2rad(-30), np.deg2rad(30))),
                ],
            )
        )
    return rows


def write_episode(group, row, trace, solver_error, stop_reason="solver_returned"):
    group.attrs["generator"] = row["generator"]
    group.attrs["solver_error"] = "" if solver_error is None else solver_error
    group.attrs["stop_reason"] = stop_reason
    group.create_dataset(
        "causal_delta", data=np.asarray(row["causal_delta"], dtype=np.float64)
    )
    states = trace["states"]
    for key in [
        "tcp_pose",
        "peg_pose",
        "socket_pose",
        "keyed_entry_goal_pose",
        "circular_transition_goal_pose",
        "goal_pose",
        "qpos",
        "qvel",
        "contact_force",
        "success",
        "obj_to_goal_dist",
        "axis_angle_err",
        "yaw_err",
        "key_clearance_margin",
        "solver_phase",
    ]:
        group.create_dataset(key, data=np.asarray([state[key] for state in states]))
    group.create_dataset("actions", data=np.asarray(trace["actions"]))
    group.create_dataset("rewards", data=np.asarray(trace["rewards"]))
    group.create_dataset("terminated", data=np.asarray(trace["terminated"], bool))
    group.create_dataset("truncated", data=np.asarray(trace["truncated"], bool))


def collect(
    rows: list[dict],
    output_path: Path,
    seed: int,
    retries_per_condition: int,
    robot_init_qpos_noise: float = 0.0,
    context_manifest_path: Path | None = None,
):
    env_id = "KeyedCircularPhaseSwitch-v1"
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
        if context_manifest_path is not None:
            context_manifest_path = context_manifest_path.resolve()
            data_file.attrs["context_manifest"] = str(context_manifest_path)
            data_file.attrs["context_manifest_sha256"] = hashlib.sha256(
                context_manifest_path.read_bytes()
            ).hexdigest()
        data_file.attrs["requested_mixed_conditions"] = sum(
            row["generator"] == "mixed" for row in rows
        )
        data_file.attrs["python_version"] = platform.python_version()
        for package in ["numpy", "torch", "mani_skill", "mplib", "sapien"]:
            try:
                data_file.attrs[f"{package}_version"] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                data_file.attrs[f"{package}_version"] = "unknown"
        geometry_names = [
            "PEG_HALF_LENGTH",
            "SHAFT_RADIUS",
            "KEY_HALF_X",
            "KEY_HALF_Y",
            "KEY_Z_MIN",
            "KEY_Z_MAX",
            "GATE_CLEARANCE",
            "GATE_Z_MIN",
            "GATE_Z_MAX",
            "BORE_INNER_RADIUS",
            "SOCKET_OUTER_RADIUS",
            "FINAL_PEG_Z",
            "KEY_CLEAR_PEG_Z",
            "PRE_ENTRY_PEG_Z",
        ]
        data_file.attrs["geometry_json"] = json.dumps(
            {
                name: float(getattr(phase_switch_symmetry_env, name))
                for name in geometry_names
            },
            sort_keys=True,
        )
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
                    solve(env)
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
                context_manifest=(
                    None
                    if context_manifest_path is None
                    else str(context_manifest_path)
                ),
                context_manifest_sha256=(
                    None
                    if context_manifest_path is None
                    else hashlib.sha256(context_manifest_path.read_bytes()).hexdigest()
                ),
                episodes=manifest,
            ),
            manifest_file,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics.h5"
        ),
    )
    parser.add_argument("--mixed-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--retries-per-condition", type=int, default=3)
    parser.add_argument("--robot-init-qpos-noise", type=float, default=0.0)
    parser.add_argument(
        "--contexts-manifest",
        type=Path,
        help="Frozen JSON manifest whose conditions replace RNG-generated contexts.",
    )
    parser.add_argument(
        "--condition-indices",
        type=int,
        nargs="+",
        help="Optional condition indices for a smoke/subset collection.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.contexts_manifest is None:
        rows = intervention_rows(args.mixed_samples, args.seed)
    else:
        with args.contexts_manifest.open(encoding="utf-8") as manifest_file:
            context_manifest = json.load(manifest_file)
        rows = context_manifest.get("conditions")
        if not isinstance(rows, list) or not rows:
            raise ValueError("contexts manifest must contain a nonempty conditions list")
        normalized_rows = []
        for expected_id, row in enumerate(rows):
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
    if args.condition_indices is not None:
        invalid = [index for index in args.condition_indices if not 0 <= index < len(rows)]
        if invalid:
            raise ValueError(f"condition indices out of range: {invalid}")
        rows = [rows[index] for index in args.condition_indices]
    elif args.smoke:
        rows = [rows[2]]
    retries = 1 if args.smoke else args.retries_per_condition
    collect(
        rows,
        args.output,
        args.seed,
        retries,
        robot_init_qpos_noise=args.robot_init_qpos_noise,
        context_manifest_path=args.contexts_manifest,
    )


if __name__ == "__main__":
    main()
