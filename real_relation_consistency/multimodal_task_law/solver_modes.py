from __future__ import annotations

"""Multimodal solvers for the SAME circular insertion task.

Four demonstration modes = {lift, diagonal} x {early, late} alignment, plus a
two-arm yaw challenge (yaw_fixed vs yaw_follow).  Every mode shares the identical
grasp/lift (common assembly start E0) and the identical terminal insertion
control; they differ only in (a) the geometric path from E0 to the socket and
(b) WHEN the peg is re-oriented to the socket's pitch (early at the clearance
height, late just above the gate).

This is a new solver (``solve_multimodal_se3``); the frozen ``solve_se3`` in
``collect_phase_switch_rotated.py`` is left byte-for-byte unchanged as the
regression anchor.  The solver does NOT read any identified law P and does not
multiply any alpha(s) into the trajectory: mode construction is pure geometric
waypoint selection, so the resulting trajectories are honest PhysX simulations
of independent demonstration modes.
"""

import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien

import numpy as np
import sapien
from transforms3d.quaternions import qmult

import phase_switch_symmetry_env as pse
from collect_phase_switch_rollouts import PhaseSwitchTraceWrapper
from mani_skill.agents.robots.panda import Panda
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)

# Firm gripper (identical to solve_se3 / rotated-axis collection).
Panda.gripper_stiffness = 2.5e3
Panda.gripper_force_limit = 150.0

# Waypoint depths: task-local z ABOVE the socket centre; the peg descends as the
# depth drops.  PRE_ENTRY (0.125) is the frozen "just above the gate" height;
# FINAL (0.060) is the frozen insertion depth; HIGH_Z is a clearance height used
# to separate the "lift then over" path from the "diagonal" path, and to place
# the EARLY alignment (far from the hole) vs the LATE alignment (at PRE_ENTRY).
HIGH_Z = 0.22
PRE_ENTRY = pse.PRE_ENTRY_PEG_Z
FINAL = pse.FINAL_PEG_Z

PATH_MODES = ("lift", "diagonal")
ALIGNMENT_MODES = ("early", "late")
YAW_MODES = ("yaw_fixed", "yaw_follow")

# Solver phase labels for the multimodal experiment.  These are DEBUG labels
# only; the primary progress for analysis is the geometric event segmentation
# (E0..E3) recovered from the raw peg/socket poses, not these solver tags.
MODE_PHASES = {
    "initial": -1,
    "reach": 0,
    "grasp": 1,
    "lift": 2,
    "approach": 3,   # the path factor (lift-then-over vs diagonal)
    "align": 4,      # the alignment factor (early vs late pitch re-orientation)
    "insert": 5,     # terminal insertion control (identical across modes)
}


class ModeTraceWrapper(PhaseSwitchTraceWrapper):
    """Trace wrapper whose solver-phase tags use MODE_PHASES instead of the frozen
    keyed SE(2)/SE(3) phase labels.  Snapshot fields are otherwise identical so
    the raw peg/tcp/socket poses, qpos/qvel, contact force and evaluate() metrics
    are bit-compatible with the old collector."""

    def snapshot(self, zero_contact=False):
        base = self.unwrapped
        info = base.evaluate()
        contact = base.scene.get_pairwise_contact_forces(base.peg, base.socket)
        if zero_contact:
            contact_value = np.zeros(3, dtype=np.float64)
        else:
            contact_value = (
                np.asarray(contact, dtype=np.float64)[0].astype(np.float64)
            )
        return dict(
            tcp_pose=np.asarray(base.agent.tcp.pose.raw_pose)[0].astype(np.float64),
            peg_pose=np.asarray(base.peg.pose.raw_pose)[0].astype(np.float64),
            socket_pose=np.asarray(base.socket.pose.raw_pose)[0].astype(np.float64),
            goal_pose=np.asarray(base.goal_pose.raw_pose)[0].astype(np.float64),
            qpos=np.asarray(base.agent.robot.get_qpos())[0].astype(np.float64),
            qvel=np.asarray(base.agent.robot.get_qvel())[0].astype(np.float64),
            contact_force=contact_value,
            success=bool(np.asarray(info["success"]).reshape(-1)[0]),
            obj_to_goal_dist=float(
                np.asarray(info["obj_to_goal_dist"]).reshape(-1)[0]
            ),
            axis_angle_err=float(
                np.asarray(info["axis_angle_err"]).reshape(-1)[0]
            ),
            yaw_err=float(np.asarray(info["yaw_err"]).reshape(-1)[0]),
            key_clearance_margin=float(
                np.asarray(info["key_clearance_margin"]).reshape(-1)[0]
            ),
            solver_phase=MODE_PHASES[self.phase],
        )


def _make_planner(env: PhaseSwitchTraceWrapper) -> PandaArmMotionPlanningSolver:
    base = env.unwrapped
    return PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )


def solve_multimodal_se3(
    env: PhaseSwitchTraceWrapper,
    path_mode: str = "lift",
    alignment_mode: str = "early",
    align_yaw: float = 0.0,
    final_yaw: float | None = None,
):
    """Run one demonstration mode on ``CircularPhaseSwitchSE3-v1``.

    Parameters
    ----------
    path_mode : "lift" (raise to clearance, then translate over the socket) or
        "diagonal" (one straight diagonal move to the over-socket waypoint).
    alignment_mode : "early" (re-orient to the socket pitch at the clearance
        height) or "late" (re-orient only at PRE_ENTRY, just above the gate).
    align_yaw : axial yaw the peg aligns to (0 for the pitch experiment; the
        socket's own yaw for the yaw_follow challenge).
    final_yaw : axial yaw retained at the terminal depth; defaults to align_yaw.
        Set to the socket yaw for yaw_follow so there is NO unlock-to-0.
    """
    assert path_mode in PATH_MODES, path_mode
    assert alignment_mode in ALIGNMENT_MODES, alignment_mode
    base = env.unwrapped
    axis = base.insertion_axis
    planner = _make_planner(env)
    try:
        peg_position = base.peg.pose.sp.p
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

        roll = float(base.socket_rpy[0])
        pitch = float(base.socket_rpy[1])
        final_yaw = float(align_yaw) if final_yaw is None else float(final_yaw)

        socket_center = np.asarray(base.socket.pose.sp.p, dtype=np.float64)
        socket_axis = np.asarray(base.socket_axis, dtype=np.float64)

        def peg_pose(depth, r, p, y):
            return base.target_pose_at(depth, r, p, y).sp

        def peg_pose_custom(pos, r, p, y):
            q = qmult(base.orientation, pse._local_quat(r, p, y))
            return sapien.Pose(p=np.asarray(pos, dtype=np.float64), q=q)

        # --- common grasp/lift (identical to solve_se3) ---------------------
        env.set_phase("reach")
        move_pose(grasp_pose * sapien.Pose([0.0, 0.0, -0.070]))
        env.set_phase("grasp")
        move_pose(grasp_pose)
        planner.close_gripper()
        env.set_phase("lift")
        move_pose(grasp_pose * sapien.Pose([0.0, 0.0, -0.095]))

        # --- common assembly start E0 reached ---------------------------------
        env.set_phase("approach")

        over_high_nominal = peg_pose_custom(
            socket_center + HIGH_Z * socket_axis, 0.0, 0.0, align_yaw
        )
        pre_nominal = peg_pose_custom(
            socket_center + PRE_ENTRY * socket_axis, 0.0, 0.0, align_yaw
        )
        pre_tilted = peg_pose(PRE_ENTRY, roll, pitch, align_yaw)
        final_target = peg_pose(FINAL, roll, pitch, final_yaw)

        # --- path factor ------------------------------------------------------
        if path_mode == "lift":
            # L-shape: raise straight up to clearance, then translate laterally
            # to the over-socket waypoint (two distinct moves).
            peg_now = np.asarray(base.peg.pose.sp.p, dtype=np.float64)
            clearance = peg_pose_custom(
                np.array([peg_now[0], peg_now[1], socket_center[2] + HIGH_Z]),
                0.0, 0.0, align_yaw,
            )
            move_pose(tcp_pose_for_peg_target(clearance))
            move_pose(tcp_pose_for_peg_target(over_high_nominal), refine_steps=3)
        else:  # diagonal
            # One straight diagonal move from the lifted peg to over the socket.
            move_pose(tcp_pose_for_peg_target(over_high_nominal), refine_steps=3)

        # --- alignment factor -------------------------------------------------
        env.set_phase("align")
        if alignment_mode == "early":
            # tilt at the clearance height, then descend already tilted.
            move_pose(
                tcp_pose_for_peg_target(peg_pose(HIGH_Z, roll, pitch, align_yaw)),
                refine_steps=5,
            )
            move_pose(tcp_pose_for_peg_target(pre_tilted), refine_steps=5)
        else:  # late
            # descend at nominal orientation, then tilt just above the gate.
            move_pose(tcp_pose_for_peg_target(pre_nominal), refine_steps=5)
            move_pose(tcp_pose_for_peg_target(pre_tilted), refine_steps=5)

        # --- terminal insertion control (identical across modes) --------------
        env.set_phase("insert")
        move_pose(tcp_pose_for_peg_target(final_target), refine_steps=10)
        move_pose(tcp_pose_for_peg_target(final_target), refine_steps=10)
    finally:
        planner.close()


def solve_yaw_challenge(env: PhaseSwitchTraceWrapper, yaw_mode: str):
    """Fixed safe path (lift + early) with one of two axial-spin preferences.

    yaw_fixed  : align and retain nominal axial spin (yaw = 0).
    yaw_follow : align to and retain the socket's own axial spin (yaw = socket
                 yaw), with NO unlock-to-0 at the terminal depth.
    """
    assert yaw_mode in YAW_MODES, yaw_mode
    base = env.unwrapped
    socket_yaw = float(base.socket_rpy[2])
    if yaw_mode == "yaw_fixed":
        solve_multimodal_se3(env, "lift", "early", align_yaw=0.0, final_yaw=0.0)
    else:
        solve_multimodal_se3(
            env, "lift", "early", align_yaw=socket_yaw, final_yaw=socket_yaw
        )
