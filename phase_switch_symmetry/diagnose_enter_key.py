from __future__ import annotations

"""Single-anchor enter_key root-cause diagnosis.

For a fixed (Q2, zero intervention, anchor), drive the solver through
grasp -> lift -> align (pre-insert), then for each enter_key staging depth check
separately: target-pose IK feasibility, endpoint self/environment collision, and
the screw/RRT planning status. This separates pose-infeasible vs
collision-infeasible vs planner-narrow-passage failure.
"""

import argparse
import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien

import gymnasium as gym
import numpy as np
import sapien

import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_env  # noqa: F401
from collect_phase_switch_rollouts import PhaseSwitchTraceWrapper
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)

ENTER_DEPTHS = [0.108, 0.098, 0.088, phase_switch_symmetry_env.KEY_CLEAR_PEG_Z]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orientation-quat", nargs=4, type=float, required=True)
    parser.add_argument("--task-anchor", nargs=3, type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    orientation = np.asarray(args.orientation_quat, dtype=np.float64)
    orientation = orientation / np.linalg.norm(orientation)
    anchor = np.asarray(args.task_anchor, dtype=np.float64)

    base = gym.make(
        "KeyedCircularPhaseSwitch-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_mode=None,
        robot_init_qpos_noise=0.0,
        orientation=orientation,
        task_anchor=anchor,
    )
    env = PhaseSwitchTraceWrapper(base)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )
    b = env.unwrapped
    env.reset(seed=args.seed, options={"causal_delta": [0.0, 0.0, 0.0]})
    env.start_trace()

    axis = b.insertion_axis

    def move_pose(target, refine_steps=0):
        r = planner.move_to_pose_with_screw(target, refine_steps=refine_steps)
        if r == -1:
            r = planner.move_to_pose_with_RRTConnect(target, refine_steps=refine_steps)
        if r == -1:
            raise RuntimeError("motion planning failed")

    def tcp_for_peg(peg_pose):
        return peg_pose * b.peg.pose.sp.inv() * b.agent.tcp.pose.sp

    # grasp + lift + align (to pre-insert depth 0.125)
    peg_position = b.peg.pose.sp.p
    grasp_pose = b.agent.build_grasp_pose(
        approaching=-axis,
        closing=b.Q_mat @ np.array([0.0, 1.0, 0.0]),
        center=peg_position + 0.020 * axis,
    )
    env.set_phase("reach")
    move_pose(grasp_pose * sapien.Pose([0.0, 0.0, -0.070]))
    env.set_phase("grasp")
    move_pose(grasp_pose)
    planner.close_gripper()
    env.set_phase("lift")
    move_pose(grasp_pose * sapien.Pose([0.0, 0.0, -0.095]))
    env.set_phase("align_keyed")
    preinsert = b.target_pose_at(phase_switch_symmetry_env.PRE_ENTRY_PEG_Z, b.socket_yaw).sp
    move_pose(tcp_for_peg(preinsert), refine_steps=5)
    move_pose(tcp_for_peg(preinsert), refine_steps=5)

    print(f"anchor={anchor} reached align_keyed (pre-insert depth 0.125)\n")
    print(f"{'depth':>7} | {'IK status':10} | {'self':>4} | {'env':>4} | {'screw':>5} | {'RRT':>5}")
    print("-" * 70)

    print(f"{'depth':>7} | {'IK':5} | {'screw':8} | {'RRT':8} | execution")
    print("-" * 60)
    for depth in ENTER_DEPTHS:
        peg_target = b.target_pose_at(depth, b.socket_yaw).sp
        tcp = tcp_for_peg(peg_target)
        goal_7 = np.concatenate([tcp.p, tcp.q])
        qpos = planner.robot.get_qpos().cpu().numpy()[0]
        ik_status, _ = planner.planner.IK(goal_7, qpos)
        screw = planner.planner.plan_screw(
            goal_7, qpos, time_step=planner.base_env.control_timestep,
            use_point_cloud=planner.use_point_cloud,
        )
        rrt = planner.planner.plan_qpos_to_pose(
            goal_7, qpos, time_step=planner.base_env.control_timestep,
            use_point_cloud=planner.use_point_cloud, wrt_world=True,
        )
        ik_ok = "Success" in str(ik_status)
        screw_ok = str(screw["status"]) == "Success"
        rrt_ok = str(rrt["status"]) == "Success"
        # Now EXECUTE the path (follow_path) to see if it physically succeeds.
        exec_note = "ok"
        try:
            move_pose(tcp, refine_steps=5)
        except Exception as exc:
            exec_note = f"FAIL: {str(exc)[:40]}"
        print(
            f"{depth:7.3f} | {'ok' if ik_ok else 'FAIL':5} | "
            f"{'ok' if screw_ok else 'fail':8} | {'ok' if rrt_ok else 'fail':8} | {exec_note}"
        )

    planner.close()
    env.close()


if __name__ == "__main__":
    main()
