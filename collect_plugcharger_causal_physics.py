#!/usr/bin/env python3
"""
collect_plugcharger_causal_physics.py

Purpose
-------
Generate REAL ManiSkill/PhysX rollouts for the already validated 63 causal reset
states:

  isolated_grid: reset_states.h5       (13)
  train_mixed:   reset_states(1).h5    (50)

This script does NOT use the prior geometric waypoint trajectories.

It:
  * restores each exact saved scene state;
  * recomputes PlugCharger goal_pose from the restored receptacle;
  * executes the official PlugCharger motion-planning strategy through pd_joint_pos;
  * intercepts every env.step() issued by the planner;
  * saves real simulated TCP/object trajectories and contact forces;
  * keeps failed/partial episodes instead of silently pruning them.

Target stack:
  ManiSkill 3.0.1 (or compatible current 3.x)
  Python 3.10/3.11 recommended
  physx_cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import torch
import trimesh
import sapien

import mani_skill
import mani_skill.envs  # registers standard envs
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.plug_charger import PlugChargerEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
)
from mani_skill.utils.registration import register_env


# =============================================================================
# Basic utilities
# =============================================================================

def npy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def scalar(x):
    a = npy(x)
    if a.size == 1:
        v = a.reshape(-1)[0]
        if a.dtype == bool:
            return bool(v)
        return float(v)
    return a.tolist()


def quat_angle_wxyz(q1, q2):
    q1 = np.asarray(q1, np.float64)
    q2 = np.asarray(q2, np.float64)
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    d = np.clip(abs(float(np.dot(q1, q2))), 0.0, 1.0)
    return float(2 * np.arccos(d))


def read_group_recursive(g: h5py.Group):
    d = {}
    for k, obj in g.items():
        if isinstance(obj, h5py.Group):
            d[k] = read_group_recursive(obj)
        else:
            d[k] = obj[()]
    return d


def load_reset_samples(path: Path):
    samples = []
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[-1]))
        for key in keys:
            g = f[key]
            samples.append(
                dict(
                    sample_id=int(key.split("_")[-1]),
                    causal_delta=np.asarray(g["causal_delta"][()], np.float64),
                    env_state=read_group_recursive(g["env_state"]),
                    expected=dict(
                        charger_pose=np.asarray(g["obs/extra/charger_pose"][0], np.float64),
                        receptacle_pose=np.asarray(
                            g["obs/extra/receptacle_pose"][0], np.float64
                        ),
                        goal_pose=np.asarray(g["obs/extra/goal_pose"][0], np.float64),
                        tcp_pose=np.asarray(g["obs/extra/tcp_pose"][0], np.float64),
                        qpos=np.asarray(g["obs/agent/qpos"][0], np.float64),
                    ),
                )
            )
    return samples


# =============================================================================
# Environment
# =============================================================================

@register_env("PlugChargerCausalPhysics-v1", max_episode_steps=400)
class PlugChargerCausalPhysicsEnv(PlugChargerEnv):
    """
    Same physical PlugCharger task.

    Only change:
      Whenever a saved state is restored through set_state_dict(), goal_pose is
      recomputed from receptacle pose. This is required because PlugCharger
      goal_pose is task state derived from receptacle pose, not an independent
      rigid actor in the state dictionary.
    """

    def set_state_dict(self, state_dict, env_idx=None):
        ret = super().set_state_dict(state_dict, env_idx)
        self.goal_pose = self.receptacle.pose * sapien.Pose(
            q=euler2quat(0, 0, np.pi)
        )
        return ret


# =============================================================================
# Explicit recorder: catches every env.step() used by motion planner
# =============================================================================

PHASE = {
    "initial": -1,
    "reach": 0,
    "grasp_approach": 1,
    "grasp_close": 2,
    "align": 3,
    "insert": 4,
}


class PhysicsTraceRecorder(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.phase_name = "initial"
        self.trace = None

    def set_phase(self, name):
        if name not in PHASE:
            raise KeyError(name)
        self.phase_name = name

    def _snapshot(self):
        b = self.unwrapped

        tcp = npy(b.agent.tcp.pose.raw_pose)[0].astype(np.float64)
        charger = npy(b.charger.pose.raw_pose)[0].astype(np.float64)
        receptacle = npy(b.receptacle.pose.raw_pose)[0].astype(np.float64)
        goal = npy(b.goal_pose.raw_pose)[0].astype(np.float64)
        qpos = npy(b.agent.robot.get_qpos())[0].astype(np.float64)
        qvel = npy(b.agent.robot.get_qvel())[0].astype(np.float64)

        f = b.scene.get_pairwise_contact_forces(b.charger, b.receptacle)
        f = npy(f)[0].astype(np.float64)

        info = b.evaluate()
        return dict(
            tcp_pose=tcp,
            charger_pose=charger,
            receptacle_pose=receptacle,
            goal_pose=goal,
            qpos=qpos,
            qvel=qvel,
            contact_force=f,
            success=bool(npy(info["success"]).reshape(-1)[0]),
            obj_to_goal_dist=float(npy(info["obj_to_goal_dist"]).reshape(-1)[0]),
            obj_to_goal_angle=float(npy(info["obj_to_goal_angle"]).reshape(-1)[0]),
            solver_phase=int(PHASE[self.phase_name]),
        )

    def start_trace(self):
        self.trace = dict(
            states=[],
            actions=[],
            rewards=[],
            terminated=[],
            truncated=[],
        )
        self.phase_name = "initial"
        self.trace["states"].append(self._snapshot())

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.trace is not None:
            self.trace["actions"].append(np.asarray(action, np.float64).copy())
            self.trace["rewards"].append(float(npy(reward).reshape(-1)[0]))
            self.trace["terminated"].append(
                bool(npy(terminated).reshape(-1)[0])
            )
            self.trace["truncated"].append(
                bool(npy(truncated).reshape(-1)[0])
            )
            self.trace["states"].append(self._snapshot())
        return obs, reward, terminated, truncated, info


# =============================================================================
# State validation
# =============================================================================

def validate_restored_state(env, sample):
    b = env.unwrapped
    exp = sample["expected"]

    rec = npy(b.receptacle.pose.raw_pose)[0]
    chg = npy(b.charger.pose.raw_pose)[0]
    tcp = npy(b.agent.tcp.pose.raw_pose)[0]
    q = npy(b.agent.robot.get_qpos())[0]

    checks = dict(
        receptacle_pos_err_m=float(np.linalg.norm(rec[:3] - exp["receptacle_pose"][:3])),
        receptacle_rot_err_rad=quat_angle_wxyz(
            rec[3:7], exp["receptacle_pose"][3:7]
        ),
        charger_pos_err_m=float(np.linalg.norm(chg[:3] - exp["charger_pose"][:3])),
        charger_rot_err_rad=quat_angle_wxyz(chg[3:7], exp["charger_pose"][3:7]),
        tcp_pos_err_m=float(np.linalg.norm(tcp[:3] - exp["tcp_pose"][:3])),
        tcp_rot_err_rad=quat_angle_wxyz(tcp[3:7], exp["tcp_pose"][3:7]),
        qpos_max_err=float(np.max(np.abs(q - exp["qpos"]))),
    )

    # Fail fast. Causal isolation is meaningless if reset restoration is wrong.
    if checks["receptacle_pos_err_m"] > 2e-5:
        raise RuntimeError(f"Receptacle reset mismatch: {checks}")
    if checks["receptacle_rot_err_rad"] > 2e-4:
        raise RuntimeError(f"Receptacle rotation mismatch: {checks}")
    if checks["charger_pos_err_m"] > 2e-5:
        raise RuntimeError(f"Charger reset mismatch: {checks}")
    if checks["qpos_max_err"] > 2e-5:
        raise RuntimeError(f"Robot qpos reset mismatch: {checks}")

    # TCP is recomputed kinematically, allow a slightly looser float tolerance.
    if checks["tcp_pos_err_m"] > 3e-4:
        raise RuntimeError(f"TCP reset mismatch: {checks}")

    return checks


# =============================================================================
# Official PlugCharger solution with only one change:
# no internal env.reset(), so it respects our exact restored causal state.
# =============================================================================

def solve_current_state(env: PhysicsTraceRecorder, debug=False):
    b = env.unwrapped
    assert b.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"]

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=False,
        base_pose=b.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )

    results = {}
    FINGER_LENGTH = 0.025

    try:
        charger_base_pose = b.charger_base_pose
        charger_base_size = np.asarray(b._base_size) * 2
        obb = trimesh.primitives.Box(
            extents=charger_base_size,
            transform=charger_base_pose.sp.to_transformation_matrix(),
        )

        approaching = np.array([0, 0, -1])
        target_closing = b.agent.tcp.pose.sp.to_transformation_matrix()[:3, 1]
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=FINGER_LENGTH,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = b.agent.build_grasp_pose(approaching, closing, center)
        grasp_pose = grasp_pose * sapien.Pose(
            q=euler2quat(0, np.deg2rad(15), 0)
        )

        # Reach
        env.set_phase("reach")
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        results["reach"] = planner.move_to_pose_with_screw(reach_pose)

        # Grasp approach
        env.set_phase("grasp_approach")
        results["grasp_approach"] = planner.move_to_pose_with_screw(grasp_pose)

        # Close
        env.set_phase("grasp_close")
        results["grasp_close"] = planner.close_gripper()

        # Recompute targets from CURRENT causal goal and CURRENT physical charger/TCP.
        pre_insert_pose = (
            b.goal_pose.sp
            * sapien.Pose([-0.05, 0.0, 0.0])
            * b.charger.pose.sp.inv()
            * b.agent.tcp.pose.sp
        )
        insert_pose = (
            b.goal_pose.sp
            * b.charger.pose.sp.inv()
            * b.agent.tcp.pose.sp
        )

        # Align
        env.set_phase("align")
        results["align_1"] = planner.move_to_pose_with_screw(
            pre_insert_pose, refine_steps=0
        )
        results["align_2"] = planner.move_to_pose_with_screw(
            pre_insert_pose, refine_steps=5
        )

        # Insert under PhysX contacts/controller dynamics
        env.set_phase("insert")
        results["insert"] = planner.move_to_pose_with_screw(insert_pose)

    finally:
        planner.close()

    return results


# =============================================================================
# HDF5 writer
# =============================================================================

def write_episode(group, sample, trace, restore_checks, final_info, solver_error):
    group.create_dataset("causal_delta", data=sample["causal_delta"])
    group.attrs["source_sample_id"] = int(sample["sample_id"])
    group.attrs["solver_error"] = "" if solver_error is None else solver_error

    states = trace["states"]
    for key in [
        "tcp_pose",
        "charger_pose",
        "receptacle_pose",
        "goal_pose",
        "qpos",
        "qvel",
        "contact_force",
        "success",
        "obj_to_goal_dist",
        "obj_to_goal_angle",
        "solver_phase",
    ]:
        group.create_dataset(key, data=np.asarray([s[key] for s in states]))

    group.create_dataset("actions", data=np.asarray(trace["actions"]))
    group.create_dataset("rewards", data=np.asarray(trace["rewards"]))
    group.create_dataset("terminated", data=np.asarray(trace["terminated"], bool))
    group.create_dataset("truncated", data=np.asarray(trace["truncated"], bool))

    rc = group.create_group("restore_checks")
    for k, v in restore_checks.items():
        rc.attrs[k] = v

    fi = group.create_group("final_info")
    for k, v in final_info.items():
        if np.isscalar(v):
            fi.attrs[k] = v
        else:
            fi.create_dataset(k, data=np.asarray(v))


def collect_split(reset_path, output_path, split_name, debug=False, max_samples=None):
    samples = load_reset_samples(reset_path)
    if max_samples is not None:
        samples = samples[:max_samples]

    base = gym.make(
        "PlugChargerCausalPhysics-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_mode=None,
        robot_init_qpos_noise=0.0,
    )
    env = PhysicsTraceRecorder(base)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = []

    with h5py.File(output_path, "w") as fout:
        fout.attrs["env_id"] = "PlugChargerCausalPhysics-v1"
        fout.attrs["base_task"] = "PlugCharger-v1"
        fout.attrs["split"] = split_name
        fout.attrs["source_type"] = "motionplanning_causal_physics"
        fout.attrs["sim_backend"] = "physx_cpu"
        fout.attrs["control_mode"] = "pd_joint_pos"
        fout.attrs["maniskill_version"] = getattr(mani_skill, "__version__", "unknown")
        fout.attrs["note"] = (
            "Real PhysX env.step rollouts. Not geometric waypoint interpolation."
        )

        for i, sample in enumerate(samples):
            print(
                f"[{split_name}] {i+1:02d}/{len(samples)} "
                f"sample={sample['sample_id']} "
                f"delta={sample['causal_delta'].tolist()}"
            )

            # Current ManiSkill supports reset_to_env_states specifically for this.
            obs, info = env.reset(
                seed=0,
                options={
                    "reset_to_env_states": {
                        "env_states": sample["env_state"],
                        "obs": None,
                    }
                },
            )

            restore_checks = validate_restored_state(env, sample)
            env.start_trace()

            solver_error = None
            try:
                solve_current_state(env, debug=debug)
            except Exception as exc:
                solver_error = repr(exc)
                print("  solver exception:", solver_error)

            final_raw = env.unwrapped.evaluate()
            final_info = {k: scalar(v) for k, v in final_raw.items()}
            success = bool(final_info.get("success", False))
            nsteps = len(env.trace["actions"])
            max_force = (
                float(
                    np.max(
                        np.linalg.norm(
                            np.asarray(
                                [s["contact_force"] for s in env.trace["states"]]
                            ),
                            axis=1,
                        )
                    )
                )
                if env.trace["states"]
                else 0.0
            )

            g = fout.create_group(f"episode_{i}")
            write_episode(
                g,
                sample,
                env.trace,
                restore_checks,
                final_info,
                solver_error,
            )

            row = dict(
                episode_id=i,
                sample_id=sample["sample_id"],
                causal_delta=sample["causal_delta"].tolist(),
                steps=nsteps,
                success=success,
                max_contact_force_N=max_force,
                solver_error=solver_error,
                final_info=final_info,
                restore_checks=restore_checks,
            )
            manifest.append(row)
            print(
                f"  steps={nsteps} success={success} "
                f"max_contact={max_force:.3f} N"
            )

    env.close()

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            dict(
                env_id="PlugChargerCausalPhysics-v1",
                base_task="PlugCharger-v1",
                split=split_name,
                source_type="motionplanning_causal_physics",
                source_desc=(
                    "Exact causal reset states + official PlugCharger motion planner "
                    "+ real ManiSkill/PhysX pd_joint_pos execution. No waypoint interpolation."
                ),
                episodes=manifest,
            ),
            f,
            indent=2,
        )

    ns = sum(x["success"] for x in manifest)
    print(f"\n[{split_name}] SUCCESS = {ns}/{len(manifest)}")
    print("wrote:", output_path)
    print("wrote:", json_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isolated", required=True, type=Path)
    ap.add_argument("--mixed", required=True, type=Path)
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("physics_causal_rollouts"),
    )
    ap.add_argument("--debug", action="store_true")
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Use 1 first as a smoke test.",
    )
    args = ap.parse_args()

    print("ManiSkill:", getattr(mani_skill, "__version__", "unknown"))
    print("Collecting REAL PhysX rollouts with pd_joint_pos.\n")

    collect_split(
        args.isolated,
        args.output_root / "isolated_grid_physics.h5",
        "isolated_grid",
        debug=args.debug,
        max_samples=args.max_samples,
    )
    collect_split(
        args.mixed,
        args.output_root / "train_mixed_physics.h5",
        "train_mixed",
        debug=args.debug,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
