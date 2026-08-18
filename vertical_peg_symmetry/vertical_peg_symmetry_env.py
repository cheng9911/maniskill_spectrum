from __future__ import annotations

import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
from typing import Any

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat, quat2euler

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig


PEG_HALF_LENGTH = 0.045
PEG_RADIUS = 0.022
SOCKET_CLEARANCE = 0.002
SOCKET_HEIGHT = 0.04
SOCKET_OUTER = 0.070
SOCKET_CENTER = np.array([0.10, 0.0, 0.0], dtype=np.float64)
PEG_START = np.array([-0.08, -0.18, PEG_HALF_LENGTH], dtype=np.float64)


def _make_mat(color):
    return sapien.render.RenderMaterial(
        base_color=color, roughness=0.55, specular=0.35
    )


def _yaw_quat(yaw: float):
    return euler2quat(0.0, 0.0, yaw)


def _yaw_from_wxyz(q) -> float:
    return float(quat2euler(np.asarray(q, dtype=np.float64))[2])


def _wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class VerticalPegSymmetryBaseEnv(BaseEnv):
    """Vertical peg insertion with explicit task-symmetry interventions.

    Reset option:
        causal_delta: [dx_world, dy_world, dyaw_z]

    The socket frame insertion axis is world z. For the square task, dyaw_z is
    task-relevant. For the circular task, dyaw_z is an axial gauge and the
    planner/evaluator ignore it.
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda
    peg_kind = "square"
    yaw_sensitive = True

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.0, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.socket_yaw = 0.0
        self.nominal_goal_yaw = 0.0
        self.causal_delta = np.zeros(3, dtype=np.float64)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.45, -0.45, 0.45], [0.04, -0.04, 0.08])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.55, -0.55, 0.55], [0.04, -0.04, 0.06])
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0.0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.peg = self._build_peg()
        self.socket = self._build_socket()

    def _build_peg(self):
        builder = self.scene.create_actor_builder()
        if self.peg_kind == "circular":
            z_axis_cylinder = sapien.Pose(q=euler2quat(0.0, -np.pi / 2, 0.0))
            builder.add_cylinder_collision(
                pose=z_axis_cylinder,
                radius=PEG_RADIUS,
                half_length=PEG_HALF_LENGTH,
            )
            builder.add_cylinder_visual(
                pose=z_axis_cylinder,
                radius=PEG_RADIUS,
                half_length=PEG_HALF_LENGTH,
                material=_make_mat([0.95, 0.42, 0.24, 1.0]),
            )
        else:
            half_size = [PEG_RADIUS, PEG_RADIUS, PEG_HALF_LENGTH]
            builder.add_box_collision(half_size=half_size)
            builder.add_box_visual(
                half_size=half_size,
                material=_make_mat([0.95, 0.42, 0.24, 1.0]),
            )
        builder.initial_pose = sapien.Pose(p=PEG_START)
        return builder.build("vertical_peg")

    def _build_socket(self):
        if self.peg_kind == "circular":
            return self._build_circular_socket()
        return self._build_square_socket()

    def _build_square_socket(self):
        builder = self.scene.create_actor_builder()
        inner = PEG_RADIUS + SOCKET_CLEARANCE
        outer = SOCKET_OUTER
        thickness = (outer - inner) * 0.5
        offset = inner + thickness
        z = SOCKET_HEIGHT * 0.5
        mat = _make_mat([0.35, 0.42, 0.55, 1.0])
        wall_specs = [
            ([outer, thickness, z], [0.0, offset, z]),
            ([outer, thickness, z], [0.0, -offset, z]),
            ([thickness, outer, z], [offset, 0.0, z]),
            ([thickness, outer, z], [-offset, 0.0, z]),
        ]
        for half_size, pos in wall_specs:
            pose = sapien.Pose(pos)
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)
        builder.initial_pose = sapien.Pose(p=SOCKET_CENTER)
        return builder.build_kinematic("square_socket")

    def _build_circular_socket(self):
        builder = self.scene.create_actor_builder()
        inner = PEG_RADIUS + SOCKET_CLEARANCE
        outer = SOCKET_OUTER
        radius = 0.5 * (inner + outer)
        thickness = 0.5 * (outer - inner)
        seg_len = 2 * np.pi * radius / 20.0 * 0.58
        z = SOCKET_HEIGHT * 0.5
        mat = _make_mat([0.22, 0.50, 0.48, 1.0])
        for i in range(20):
            a = 2 * np.pi * i / 20.0
            pose = sapien.Pose(
                [radius * np.cos(a), radius * np.sin(a), z],
                q=euler2quat(0.0, 0.0, a),
            )
            builder.add_box_collision(pose, [thickness, seg_len, z])
            builder.add_box_visual(pose, [thickness, seg_len, z], material=mat)
        builder.initial_pose = sapien.Pose(p=SOCKET_CENTER)
        return builder.build_kinematic("circular_socket")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            causal_delta = (
                np.zeros(3, dtype=np.float64)
                if options is None or options.get("causal_delta") is None
                else np.asarray(options["causal_delta"], dtype=np.float64)
            )
            if causal_delta.shape != (3,):
                raise ValueError("causal_delta must be [dx_world, dy_world, dyaw_z]")
            self.causal_delta = causal_delta

            peg_p = torch.tensor(PEG_START, dtype=torch.float32, device=self.device)
            peg_p = peg_p.repeat(len(env_idx), 1)
            peg_q = torch.tensor(
                _yaw_quat(self.nominal_goal_yaw), dtype=torch.float32, device=self.device
            ).repeat(len(env_idx), 1)
            self.peg.set_pose(Pose.create_from_pq(peg_p, peg_q))

            socket_center = SOCKET_CENTER.copy()
            socket_center[:2] += causal_delta[:2]
            self.socket_yaw = float(causal_delta[2])
            socket_p = torch.tensor(socket_center, dtype=torch.float32, device=self.device)
            socket_p = socket_p.repeat(len(env_idx), 1)
            socket_q = torch.tensor(
                _yaw_quat(self.socket_yaw), dtype=torch.float32, device=self.device
            ).repeat(len(env_idx), 1)
            self.socket.set_pose(Pose.create_from_pq(socket_p, socket_q))

            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0.0,
                    -np.pi * 5 / 8,
                    0.0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ],
                dtype=np.float64,
            )
            qpos = self._episode_rng.normal(0, self.robot_init_qpos_noise, (len(env_idx), len(qpos))) + qpos
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0.0, 0.0]))

    @property
    def socket_center_pose(self):
        p = self.socket.pose.p.clone()
        p[:, 2] = PEG_HALF_LENGTH
        q_yaw = self.socket_yaw if self.yaw_sensitive else self.nominal_goal_yaw
        q = torch.tensor(_yaw_quat(q_yaw), dtype=p.dtype, device=p.device).repeat(
            p.shape[0], 1
        )
        return Pose.create_from_pq(p, q)

    @property
    def goal_pose(self):
        return self.socket_center_pose

    def evaluate(self):
        peg = self.peg.pose
        goal = self.goal_pose
        pos_err = torch.linalg.norm(peg.p - goal.p, axis=1)
        axis_err = torch.zeros_like(pos_err)
        if self.yaw_sensitive:
            peg_yaw = torch.tensor(
                [_yaw_from_wxyz(q) for q in peg.q.detach().cpu().numpy()],
                dtype=pos_err.dtype,
                device=pos_err.device,
            )
            goal_yaw = torch.full_like(peg_yaw, self.socket_yaw)
            yaw_err = torch.abs(
                torch.tensor(
                    [_wrap_pi(float(x)) for x in (peg_yaw - goal_yaw).detach().cpu()],
                    dtype=pos_err.dtype,
                    device=pos_err.device,
                )
            )
        else:
            yaw_err = torch.zeros_like(pos_err)
        success = (pos_err < 0.012) & (axis_err < 0.15) & (yaw_err < 0.18)
        return dict(
            success=success,
            obj_to_goal_dist=pos_err,
            axis_angle_err=axis_err,
            yaw_err=yaw_err,
        )

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                socket_pose=self.socket.pose.raw_pose,
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


@register_env("VerticalSquarePegSymmetry-v1", max_episode_steps=260)
class VerticalSquarePegSymmetryEnv(VerticalPegSymmetryBaseEnv):
    peg_kind = "square"
    yaw_sensitive = True


@register_env("VerticalCircularPegSymmetry-v1", max_episode_steps=260)
class VerticalCircularPegSymmetryEnv(VerticalPegSymmetryBaseEnv):
    peg_kind = "circular"
    yaw_sensitive = False
