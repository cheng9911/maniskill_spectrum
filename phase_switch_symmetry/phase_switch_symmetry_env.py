from __future__ import annotations

import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
from typing import Any

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig


# The peg is circular except for a short rectangular key at its lower tip.
PEG_HALF_LENGTH = 0.045
SHAFT_RADIUS = 0.013
SHAFT_Z_MIN = -0.029
SHAFT_Z_MAX = PEG_HALF_LENGTH
KEY_HALF_X = 0.021
KEY_HALF_Y = 0.014
KEY_Z_MIN = -PEG_HALF_LENGTH
KEY_Z_MAX = SHAFT_Z_MIN

# A thin keyed gate sits above a circular bore. The key is free to rotate only
# after KEY_Z_MAX has moved below GATE_Z_MIN.
GATE_CLEARANCE = 0.002
GATE_Z_MIN = 0.053
GATE_Z_MAX = 0.065
BORE_INNER_RADIUS = 0.030
SOCKET_OUTER_RADIUS = 0.070
SOCKET_CENTER = np.array([0.10, 0.0, 0.0], dtype=np.float64)
PEG_START = np.array([-0.08, -0.18, PEG_HALF_LENGTH], dtype=np.float64)

FINAL_PEG_Z = 0.060
KEY_CLEAR_PEG_Z = GATE_Z_MIN - KEY_Z_MAX - 0.004
PRE_ENTRY_PEG_Z = GATE_Z_MAX - KEY_Z_MIN + 0.015


def _material(color):
    return sapien.render.RenderMaterial(
        base_color=color, roughness=0.55, specular=0.35
    )


def _yaw_quat(yaw: float):
    return euler2quat(0.0, 0.0, yaw)


def _axis_error_from_wxyz(q_wxyz: np.ndarray) -> float:
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64)
    axis_z = np.array(
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)]
    )
    return float(np.arccos(np.clip(axis_z[2], -1.0, 1.0)))


@register_env("KeyedCircularPhaseSwitch-v1", max_episode_steps=700)
class KeyedCircularPhaseSwitchEnv(BaseEnv):
    """Vertical insertion whose axial-yaw symmetry changes after a keyed gate.

    ``causal_delta = [dx_world, dy_world, dyaw_z]`` moves the complete socket.
    The keyed entry target follows ``dyaw_z``. Once the rectangular key clears
    the gate, the circular shaft and bore admit arbitrary axial yaw, so the
    transition and final targets retain nominal yaw.
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.0, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.causal_delta = np.zeros(3, dtype=np.float64)
        self.socket_yaw = 0.0
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.46, -0.46, 0.42], [0.04, -0.03, 0.07])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.55, -0.55, 0.50], [0.05, -0.03, 0.07])
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
        cylinder_to_z = sapien.Pose(q=euler2quat(0.0, -np.pi / 2, 0.0))
        shaft_center_z = 0.5 * (SHAFT_Z_MIN + SHAFT_Z_MAX)
        shaft_half_length = 0.5 * (SHAFT_Z_MAX - SHAFT_Z_MIN)
        shaft_pose = sapien.Pose(
            [0.0, 0.0, shaft_center_z], q=cylinder_to_z.q
        )
        builder.add_cylinder_collision(
            pose=shaft_pose, radius=SHAFT_RADIUS, half_length=shaft_half_length
        )
        builder.add_cylinder_visual(
            pose=shaft_pose,
            radius=SHAFT_RADIUS,
            half_length=shaft_half_length,
            material=_material([0.95, 0.42, 0.24, 1.0]),
        )
        key_center_z = 0.5 * (KEY_Z_MIN + KEY_Z_MAX)
        key_half_z = 0.5 * (KEY_Z_MAX - KEY_Z_MIN)
        key_pose = sapien.Pose([0.0, 0.0, key_center_z])
        key_half_size = [KEY_HALF_X, KEY_HALF_Y, key_half_z]
        builder.add_box_collision(key_pose, key_half_size)
        builder.add_box_visual(
            key_pose,
            key_half_size,
            material=_material([0.86, 0.25, 0.16, 1.0]),
        )
        builder.initial_pose = sapien.Pose(p=PEG_START)
        return builder.build("keyed_circular_peg")

    def _build_socket(self):
        builder = self.scene.create_actor_builder()
        bore_mat = _material([0.20, 0.52, 0.47, 1.0])
        gate_mat = _material([0.28, 0.37, 0.58, 1.0])

        ring_radius = 0.5 * (BORE_INNER_RADIUS + SOCKET_OUTER_RADIUS)
        ring_thickness = 0.5 * (SOCKET_OUTER_RADIUS - BORE_INNER_RADIUS)
        segment_half_length = 2.0 * np.pi * ring_radius / 24.0 * 0.58
        bore_half_height = 0.5 * GATE_Z_MIN
        for i in range(24):
            angle = 2.0 * np.pi * i / 24.0
            pose = sapien.Pose(
                [ring_radius * np.cos(angle), ring_radius * np.sin(angle), bore_half_height],
                q=euler2quat(0.0, 0.0, angle),
            )
            half_size = [ring_thickness, segment_half_length, bore_half_height]
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=bore_mat)

        inner_x = KEY_HALF_X + GATE_CLEARANCE
        inner_y = KEY_HALF_Y + GATE_CLEARANCE
        gate_half_height = 0.5 * (GATE_Z_MAX - GATE_Z_MIN)
        gate_z = 0.5 * (GATE_Z_MIN + GATE_Z_MAX)
        outer = SOCKET_OUTER_RADIUS
        side_thickness = 0.5 * (outer - inner_x)
        cap_thickness = 0.5 * (outer - inner_y)
        gate_parts = [
            ([side_thickness, outer, gate_half_height], [inner_x + side_thickness, 0.0, gate_z]),
            ([side_thickness, outer, gate_half_height], [-inner_x - side_thickness, 0.0, gate_z]),
            ([inner_x, cap_thickness, gate_half_height], [0.0, inner_y + cap_thickness, gate_z]),
            ([inner_x, cap_thickness, gate_half_height], [0.0, -inner_y - cap_thickness, gate_z]),
        ]
        for half_size, position in gate_parts:
            pose = sapien.Pose(position)
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=gate_mat)

        builder.initial_pose = sapien.Pose(p=SOCKET_CENTER)
        return builder.build_kinematic("keyed_to_circular_socket")

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
            self.socket_yaw = float(causal_delta[2])

            peg_p = torch.tensor(PEG_START, dtype=torch.float32, device=self.device).repeat(
                len(env_idx), 1
            )
            peg_q = torch.tensor(
                _yaw_quat(0.0), dtype=torch.float32, device=self.device
            ).repeat(len(env_idx), 1)
            self.peg.set_pose(Pose.create_from_pq(peg_p, peg_q))

            socket_center = SOCKET_CENTER.copy()
            socket_center[:2] += causal_delta[:2]
            socket_p = torch.tensor(
                socket_center, dtype=torch.float32, device=self.device
            ).repeat(len(env_idx), 1)
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
            qpos = self._episode_rng.normal(
                0, self.robot_init_qpos_noise, (len(env_idx), len(qpos))
            ) + qpos
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0.0, 0.0]))

    def target_pose_at(self, z: float, yaw: float):
        p = self.socket.pose.p.clone()
        p[:, 2] = z
        q = torch.tensor(_yaw_quat(yaw), dtype=p.dtype, device=p.device).repeat(
            p.shape[0], 1
        )
        return Pose.create_from_pq(p, q)

    @property
    def keyed_preinsert_pose(self):
        return self.target_pose_at(PRE_ENTRY_PEG_Z, self.socket_yaw)

    @property
    def keyed_entry_pose(self):
        return self.target_pose_at(KEY_CLEAR_PEG_Z, self.socket_yaw)

    @property
    def circular_transition_pose(self):
        return self.target_pose_at(KEY_CLEAR_PEG_Z, 0.0)

    @property
    def goal_pose(self):
        return self.target_pose_at(FINAL_PEG_Z, 0.0)

    @property
    def key_clearance_margin(self):
        return GATE_Z_MIN - (self.peg.pose.p[:, 2] + KEY_Z_MAX)

    def evaluate(self):
        peg = self.peg.pose
        goal = self.goal_pose
        pos_err = torch.linalg.norm(peg.p - goal.p, axis=1)
        axis_err = torch.tensor(
            [_axis_error_from_wxyz(q) for q in peg.q.detach().cpu().numpy()],
            dtype=pos_err.dtype,
            device=pos_err.device,
        )
        yaw_err = torch.zeros_like(pos_err)
        success = (pos_err < 0.012) & (axis_err < 0.15)
        return dict(
            success=success,
            obj_to_goal_dist=pos_err,
            axis_angle_err=axis_err,
            yaw_err=yaw_err,
            key_clearance_margin=self.key_clearance_margin,
        )

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                socket_pose=self.socket.pose.raw_pose,
                keyed_entry_goal_pose=self.keyed_entry_pose.raw_pose,
                goal_pose=self.goal_pose.raw_pose,
                causal_delta=torch.tensor(
                    self.causal_delta, dtype=torch.float32, device=self.device
                ).repeat(self.num_envs, 1),
                key_clearance_margin=self.key_clearance_margin[:, None],
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["success"].float()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info)
