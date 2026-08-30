from __future__ import annotations

"""Planar-push SE(3) task (supplement, own module).

A genuinely different physical task from peg-in-hole: a rectangular cuboid
block is PUSHED (never grasped) across a table to a ghost 6-DOF target. The
selectivity here is the complement of peg-in-hole — the planar table constraint
releases THREE generators:

    tracked   : du, dv (in-plane translation), yaw (heading about the table
                normal)
    suppressed: dw (vertical translation), roll, pitch (tilts)

The block stays on the table, level, so perturbing the target's out-of-plane
pose (dw/roll/pitch) never changes the block's trajectory. The ghost target is
a kinematic semi-transparent cuboid placed at C0 o intervention, which makes the
out-of-plane context components genuinely vary in the data (so the model can
learn alpha=0 vs alpha=1) rather than being absent.

``causal_delta = [du, dv, dw, d_roll, d_pitch, d_yaw]`` in the task basis.
``goal_heading`` distinguishes the two arms:
    heading_push : block aligns in-plane position AND heading -> [1,1,0,0,0,1]
    free_yaw_push: block aligns in-plane position only          -> [1,1,0,0,0,0]

This module is PURELY additive: the frozen SE(2)/SE(3) env classes stay
byte-identical.
"""

import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
from typing import Any

import numpy as np
import sapien
import torch
from transforms3d.quaternions import qmult, quat2mat

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig

from phase_switch_symmetry_env import _local_quat, _material

# Rectangular (clearly NON-square footprint) block: 0.04 x 0.02 x 0.02 m.
# The 2:1 in-plane aspect makes heading a well-defined DOF; the rectangle's
# 2-fold (180 deg) symmetry is never reached because the yaw intervention range
# is +-30 deg (well inside +-90 deg), so heading is unique without any marker.
BLOCK_HALF_X = 0.02
BLOCK_HALF_Y = 0.01
BLOCK_HALF_Z = 0.01

# Nominal ghost-target centre. task_anchor z = BLOCK_HALF_Z so that the ghost
# (at _to_world([du, dv, dw])) sits at block-centre height when dw = 0 and
# floats +-dw otherwise. Q = identity for the preregistered experiment.
TARGET_POS = np.array([0.10, 0.0, BLOCK_HALF_Z], dtype=np.float64)

# Fixed block start on the table, pushed toward +x. Reachable from the Panda
# base at (-0.615, 0, 0).
BLOCK_START = np.array([-0.10, 0.0, BLOCK_HALF_Z], dtype=np.float64)

PUSH_SUCCESS_POS_TOL = 0.008
PUSH_SUCCESS_YAW_TOL = 0.10


def _ghost_material():
    return sapien.render.RenderMaterial(
        base_color=[0.20, 0.60, 0.90, 0.35], roughness=0.6, specular=0.2
    )


def _heading_of(q_wxyz):
    """Rotation about world z of a [w, x, y, z] quaternion (the block heading).

    For a pure Rz(yaw) this returns yaw exactly; for the (nearly flat) block the
    residual roll/pitch are ~0 so this is the heading. The push task runs at
    Q = identity, so the table normal is world z.
    """
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


@register_env("PlanarPush-v1", max_episode_steps=1500)
class PlanarPushEnv(BaseEnv):
    """Non-prehensive planar push of a cuboid block to a ghost 6-DOF target."""

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.0,
                 orientation=None, task_anchor=None, goal_heading=True, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.causal_delta = np.zeros(6, dtype=np.float64)
        self.goal_heading = bool(goal_heading)
        if orientation is None:
            orientation = np.array([1.0, 0.0, 0.0, 0.0])
        orientation = np.asarray(orientation, dtype=np.float64)
        if orientation.shape != (4,):
            raise ValueError("orientation must be a [w, x, y, z] quaternion")
        orientation = orientation / np.linalg.norm(orientation)
        self.orientation = orientation
        self.Q_mat = quat2mat(orientation)
        if task_anchor is None:
            task_anchor = TARGET_POS
        self.task_anchor = np.asarray(task_anchor, dtype=np.float64)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _to_world(self, local_rel):
        """Map a task-local offset (relative to TARGET_POS) to world via Q."""
        return self.task_anchor + self.Q_mat @ np.asarray(local_rel, dtype=np.float64)

    def block_goal_position(self):
        """Block final in-plane position: target du,dv at block height (dw ignored)."""
        target = self._to_world([self.causal_delta[0], self.causal_delta[1], 0.0])
        target[2] = BLOCK_HALF_Z
        return target

    @property
    def goal_pose(self):
        """Block final pose: in-plane position + heading (yaw) at block height."""
        p = self.block_goal_position()
        if self.goal_heading:
            q = qmult(self.orientation, _local_quat(0.0, 0.0, float(self.causal_delta[5])))
        else:
            q = self.orientation.copy()
        return Pose.create_from_pq(
            torch.tensor(p, dtype=torch.float32, device=self.device).reshape(1, 3),
            torch.tensor(q, dtype=torch.float32, device=self.device).reshape(1, 4),
        )

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.35, -0.35, 0.35], [0.0, 0.0, 0.01])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.45, -0.45, 0.42], [0.0, 0.0, 0.01])
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0.0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.block = self._build_block()
        self.target = self._build_target()

    def _build_block(self):
        builder = self.scene.create_actor_builder()
        phys = sapien.physx.PhysxMaterial(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        )
        half = [BLOCK_HALF_X, BLOCK_HALF_Y, BLOCK_HALF_Z]
        builder.add_box_collision(sapien.Pose(), half, material=phys)
        builder.add_box_visual(
            sapien.Pose(), half, material=_material([0.95, 0.42, 0.24, 1.0])
        )
        builder.initial_pose = sapien.Pose(p=BLOCK_START)
        return builder.build("planar_push_block")

    def _build_target(self):
        builder = self.scene.create_actor_builder()
        half = [BLOCK_HALF_X, BLOCK_HALF_Y, BLOCK_HALF_Z]
        builder.add_box_visual(sapien.Pose(), half, material=_ghost_material())
        builder.initial_pose = sapien.Pose(p=TARGET_POS)
        return builder.build_kinematic("planar_push_ghost_target")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            causal_delta = (
                np.zeros(6, dtype=np.float64)
                if options is None or options.get("causal_delta") is None
                else np.asarray(options["causal_delta"], dtype=np.float64)
            )
            if causal_delta.shape != (6,):
                raise ValueError(
                    "causal_delta must be [du, dv, dw, d_roll, d_pitch, d_yaw]"
                )
            self.causal_delta = causal_delta
            roll, pitch, yaw = (float(causal_delta[3]), float(causal_delta[4]),
                                float(causal_delta[5]))

            # Block: FIXED start pose on the table (heading = nominal Q). The
            # start is context-independent (identical start-fixed structure to
            # peg-in-hole); only the END responds to the intervention.
            block_p = torch.tensor(
                BLOCK_START, dtype=torch.float32, device=self.device
            ).repeat(len(env_idx), 1)
            block_q = torch.tensor(
                self.orientation, dtype=torch.float32, device=self.device
            ).repeat(len(env_idx), 1)
            self.block.set_pose(Pose.create_from_pq(block_p, block_q))

            # Ghost target: C0 o intervention = task_anchor + Q @ [du, dv, dw],
            # orientation Q * Rx(roll) Ry(pitch) Rz(yaw). Full 6-DOF so
            # dw/roll/pitch genuinely vary in the data.
            target_p = torch.tensor(
                self._to_world([causal_delta[0], causal_delta[1], causal_delta[2]]),
                dtype=torch.float32, device=self.device,
            ).repeat(len(env_idx), 1)
            target_q = torch.tensor(
                qmult(self.orientation, _local_quat(roll, pitch, yaw)),
                dtype=torch.float32, device=self.device,
            ).repeat(len(env_idx), 1)
            self.target.set_pose(Pose.create_from_pq(target_p, target_q))

            qpos = np.array(
                [0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4,
                 np.pi / 4, 0.04, 0.04], dtype=np.float64,
            )
            noise = np.random.normal(
                0.0, self.robot_init_qpos_noise, (len(env_idx), len(qpos))
            )
            qpos = qpos + noise
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0.0, 0.0]))

    def evaluate(self):
        block = self.block.pose
        goal_p = torch.tensor(
            self.block_goal_position(), dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        pos_err = torch.linalg.norm(block.p[..., :2] - goal_p[..., :2], dim=1)
        heading_err = torch.zeros_like(pos_err)
        if self.goal_heading:
            target_yaw = float(self.causal_delta[5])
            block_yaws = [_heading_of(q) for q in block.q.detach().cpu().numpy()]
            heading_err = torch.tensor(
                [_wrap_pi(y - target_yaw) for y in block_yaws],
                dtype=pos_err.dtype, device=pos_err.device,
            ).abs()
            success = (pos_err < PUSH_SUCCESS_POS_TOL) & (heading_err < PUSH_SUCCESS_YAW_TOL)
        else:
            success = pos_err < PUSH_SUCCESS_POS_TOL
        return dict(success=success, obj_to_goal_dist=pos_err, heading_err=heading_err)

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                block_pose=self.block.pose.raw_pose,
                target_pose=self.target.pose.raw_pose,
                goal_pose=self.goal_pose.raw_pose,
                causal_delta=torch.tensor(
                    self.causal_delta, dtype=torch.float32, device=self.device
                ).repeat(self.num_envs, 1),
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["success"].float()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info)
