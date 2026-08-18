from __future__ import annotations

import argparse
import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import sapien
import torch

import mani_skill.envs  # noqa: F401
import vertical_peg_symmetry_env  # noqa: F401
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


PHASES = {
    "initial": -1,
    "reach": 0,
    "grasp": 1,
    "lift": 2,
    "align": 3,
    "insert": 4,
}


def npy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class TraceWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.phase = "initial"
        self.trace = None

    def set_phase(self, phase):
        self.phase = phase

    def snapshot(self):
        b = self.unwrapped
        info = b.evaluate()
        contact = b.scene.get_pairwise_contact_forces(b.peg, b.socket)
        return dict(
            tcp_pose=npy(b.agent.tcp.pose.raw_pose)[0].astype(np.float64),
            peg_pose=npy(b.peg.pose.raw_pose)[0].astype(np.float64),
            socket_pose=npy(b.socket.pose.raw_pose)[0].astype(np.float64),
            goal_pose=npy(b.goal_pose.raw_pose)[0].astype(np.float64),
            qpos=npy(b.agent.robot.get_qpos())[0].astype(np.float64),
            qvel=npy(b.agent.robot.get_qvel())[0].astype(np.float64),
            contact_force=npy(contact)[0].astype(np.float64),
            success=bool(npy(info["success"]).reshape(-1)[0]),
            obj_to_goal_dist=float(npy(info["obj_to_goal_dist"]).reshape(-1)[0]),
            axis_angle_err=float(npy(info["axis_angle_err"]).reshape(-1)[0]),
            yaw_err=float(npy(info["yaw_err"]).reshape(-1)[0]),
            solver_phase=PHASES[self.phase],
        )

    def start_trace(self):
        self.trace = dict(states=[self.snapshot()], actions=[], rewards=[], terminated=[], truncated=[])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.trace is not None:
            self.trace["actions"].append(np.asarray(action, dtype=np.float64).copy())
            self.trace["rewards"].append(float(npy(reward).reshape(-1)[0]))
            self.trace["terminated"].append(bool(npy(terminated).reshape(-1)[0]))
            self.trace["truncated"].append(bool(npy(truncated).reshape(-1)[0]))
            self.trace["states"].append(self.snapshot())
        return obs, reward, terminated, truncated, info


def solve(env: TraceWrapper):
    b = env.unwrapped
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=b.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )
    try:
        # Top-down grasp of the vertical peg. Approach is along -z, closing is y.
        peg_p = b.peg.pose.sp.p
        grasp_pose = b.agent.build_grasp_pose(
            approaching=np.array([0.0, 0.0, -1.0]),
            closing=np.array([0.0, 1.0, 0.0]),
            center=peg_p + np.array([0.0, 0.0, 0.018]),
        )
        def move_pose(pose, refine_steps=0):
            res = planner.move_to_pose_with_screw(pose, refine_steps=refine_steps)
            if res == -1:
                res = planner.move_to_pose_with_RRTConnect(
                    pose, refine_steps=refine_steps
                )
            if res == -1:
                raise RuntimeError("motion planning failed")
            return res

        env.set_phase("reach")
        move_pose(grasp_pose * sapien.Pose([0.0, 0.0, -0.07]))
        env.set_phase("grasp")
        move_pose(grasp_pose)
        planner.close_gripper()
        env.set_phase("lift")
        move_pose(grasp_pose * sapien.Pose([0.0, 0.0, -0.09]))

        pre_insert_pose = (
            b.goal_pose.sp
            * sapien.Pose([0.0, 0.0, 0.055])
            * b.peg.pose.sp.inv()
            * b.agent.tcp.pose.sp
        )
        insert_pose = b.goal_pose.sp * b.peg.pose.sp.inv() * b.agent.tcp.pose.sp

        env.set_phase("align")
        move_pose(pre_insert_pose, refine_steps=5)
        env.set_phase("insert")
        move_pose(insert_pose, refine_steps=8)
    finally:
        planner.close()


def intervention_rows(n_mixed: int, seed: int):
    isolated = []
    for deg in [-30, -15, 0, 15, 30]:
        isolated.append(dict(generator="yaw", causal_delta=[0.0, 0.0, np.deg2rad(deg)]))
    for dx, dy in [(-0.015, 0.0), (0.015, 0.0), (0.0, -0.015), (0.0, 0.015), (0.0, 0.0)]:
        isolated.append(dict(generator="translation", causal_delta=[dx, dy, 0.0]))
    rng = np.random.default_rng(seed)
    mixed = []
    for _ in range(n_mixed):
        mixed.append(
            dict(
                generator="mixed",
                causal_delta=[
                    float(rng.uniform(-0.012, 0.012)),
                    float(rng.uniform(-0.012, 0.012)),
                    float(rng.uniform(np.deg2rad(-30), np.deg2rad(30))),
                ],
            )
        )
    return isolated, mixed


def write_episode(g, task, row, trace, solver_error):
    g.attrs["task"] = task
    g.attrs["generator"] = row["generator"]
    g.attrs["solver_error"] = "" if solver_error is None else solver_error
    g.create_dataset("causal_delta", data=np.asarray(row["causal_delta"], dtype=np.float64))
    states = trace["states"]
    for k in [
        "tcp_pose",
        "peg_pose",
        "socket_pose",
        "goal_pose",
        "qpos",
        "qvel",
        "contact_force",
        "success",
        "obj_to_goal_dist",
        "axis_angle_err",
        "yaw_err",
        "solver_phase",
    ]:
        g.create_dataset(k, data=np.asarray([s[k] for s in states]))
    g.create_dataset("actions", data=np.asarray(trace["actions"]))
    g.create_dataset("rewards", data=np.asarray(trace["rewards"]))
    g.create_dataset("terminated", data=np.asarray(trace["terminated"], bool))
    g.create_dataset("truncated", data=np.asarray(trace["truncated"], bool))


def collect_task(task: str, rows: list[dict], output_path: Path):
    env_id = {
        "square": "VerticalSquarePegSymmetry-v1",
        "circular": "VerticalCircularPegSymmetry-v1",
    }[task]
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
    env = TraceWrapper(base)
    manifest = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["env_id"] = env_id
        f.attrs["task"] = task
        f.attrs["source_type"] = "vertical_peg_symmetry_motionplanning_physics"
        f.attrs["control_mode"] = "pd_joint_pos"
        f.attrs["sim_backend"] = "physx_cpu"
        for i, row in enumerate(rows):
            print(f"[{task}] {i+1:02d}/{len(rows)} {row}")
            env.reset(seed=0, options={"causal_delta": row["causal_delta"]})
            env.start_trace()
            solver_error = None
            try:
                solve(env)
            except Exception as exc:
                solver_error = repr(exc)
                print("  solver exception:", solver_error)
            states = env.trace["states"]
            forces = np.linalg.norm(np.asarray([s["contact_force"] for s in states]), axis=1)
            success = bool(states[-1]["success"])
            steps = len(env.trace["actions"])
            print(f"  steps={steps} success={success} max_contact={forces.max():.3f}")
            g = f.create_group(f"episode_{i}")
            write_episode(g, task, row, env.trace, solver_error)
            manifest.append(
                dict(
                    episode_id=i,
                    task=task,
                    **row,
                    steps=steps,
                    success=success,
                    max_contact_force_N=float(forces.max()),
                    solver_error=solver_error,
                )
            )
    env.close()
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as jf:
        json.dump(dict(task=task, episodes=manifest), jf, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=Path("vertical_peg_symmetry_rollouts"))
    ap.add_argument("--mixed-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    isolated, mixed = intervention_rows(args.mixed_samples, args.seed)
    rows = isolated[:2] + mixed[:1] if args.smoke else isolated + mixed
    for task in ["square", "circular"]:
        collect_task(task, rows, args.output_root / f"{task}_vertical_peg_physics.h5")


if __name__ == "__main__":
    main()
