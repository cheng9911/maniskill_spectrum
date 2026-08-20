from __future__ import annotations

import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
from typing import Any

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult, quat2mat

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
# Circular (symmetric) gate: an annulus that just clears the shaft so axial yaw
# is unconstrained (full SO(2)) while translation must still align to pass.
CIRCULAR_GATE_INNER_RADIUS = SHAFT_RADIUS + GATE_CLEARANCE
SOCKET_CENTER = np.array([0.10, 0.0, 0.0], dtype=np.float64)
PEG_START = np.array([-0.08, -0.18, PEG_HALF_LENGTH], dtype=np.float64)
# Grasp pedestal: a kinematic stand that holds the peg elevated so the end-on
# (along-axis) grasp is reachable for every orientation Q.
PEDESTAL_POS = np.array([-0.25, -0.18, 0.28], dtype=np.float64)  # top-surface center

FINAL_PEG_Z = 0.060
KEY_CLEAR_PEG_Z = GATE_Z_MIN - KEY_Z_MAX - 0.004
PRE_ENTRY_PEG_Z = GATE_Z_MAX - KEY_Z_MIN + 0.015


def _material(color):
    return sapien.render.RenderMaterial(
        base_color=color, roughness=0.55, specular=0.35
    )


def _yaw_quat(yaw: float):
    return euler2quat(0.0, 0.0, yaw)


def _axis_error_from_wxyz(q_wxyz: np.ndarray, axis_world: np.ndarray) -> float:
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64)
    axis_z = np.array(
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)]
    )
    return float(np.arccos(np.clip(np.dot(axis_z, np.asarray(axis_world)), -1.0, 1.0)))


@register_env("KeyedCircularPhaseSwitch-v1", max_episode_steps=1500)
class KeyedCircularPhaseSwitchEnv(BaseEnv):
    """Vertical insertion whose axial-yaw symmetry changes after a keyed gate.

    ``causal_delta = [du, dv, d_axial]`` is a task-local intervention: ``du, dv``
    are lateral translations in the socket frame and ``d_axial`` is the axial
    (insertion-axis) rotation. The whole fixture is embedded in the workspace by
    a global rigid transform ``G_Q = (Q, t_Q)``: the ``orientation`` quaternion
    ``Q`` controls direction (the insertion axis points along ``Q e_z``) and
    ``task_anchor`` controls translation (the world placement of the socket
    nominal center). The keyed entry target follows ``d_axial``; once the
    rectangular key clears the gate, the circular shaft and bore admit arbitrary
    axial yaw, so the transition and final targets retain nominal yaw.
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.0, orientation=None, task_anchor=None, keyed=True, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.causal_delta = np.zeros(3, dtype=np.float64)
        self.socket_yaw = 0.0
        self.keyed = bool(keyed)
        if orientation is None:
            orientation = np.array([1.0, 0.0, 0.0, 0.0])
        orientation = np.asarray(orientation, dtype=np.float64)
        if orientation.shape != (4,):
            raise ValueError("orientation must be a [w,x,y,z] quaternion")
        orientation = orientation / np.linalg.norm(orientation)
        self.orientation = orientation
        self.Q_mat = quat2mat(orientation)
        self.insertion_axis = self.Q_mat @ np.array([0.0, 0.0, 1.0])
        if task_anchor is None:
            task_anchor = SOCKET_CENTER
        self.task_anchor = np.asarray(task_anchor, dtype=np.float64)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _to_world(self, local_rel):
        """Map a task-local offset (relative to SOCKET_CENTER) to world via Q.

        p_world = task_anchor + Q @ local_rel, so Q is pure orientation and
        task_anchor is pure workspace placement.
        """
        return self.task_anchor + self.Q_mat @ np.asarray(local_rel, dtype=np.float64)

    def _resting_height(self):
        """Peg center height so the peg rests on the pedestal for orientation Q.

        The keyed peg bounding box (local) is x in [-KEY_HALF_X, KEY_HALF_X],
        y in [-KEY_HALF_Y, KEY_HALF_Y], z in [-PEG_HALF_LENGTH, PEG_HALF_LENGTH].
        The circular (keyed=False) peg is the shaft-only cylinder: lateral
        half-extent SHAFT_RADIUS and axial half-extent -SHAFT_Z_MIN (the shaft
        bottom), since the actor origin coincides with the keyed peg's center.
        The lowest world-z extent under Q is the Q[2,:]-weighted half-extent sum.
        """
        if self.keyed:
            half_x, half_y, half_z = KEY_HALF_X, KEY_HALF_Y, PEG_HALF_LENGTH
        else:
            half_x, half_y, half_z = SHAFT_RADIUS, SHAFT_RADIUS, -SHAFT_Z_MIN
        r2 = self.Q_mat[2, :]
        return float(
            abs(r2[0]) * half_x + abs(r2[1]) * half_y + abs(r2[2]) * half_z
        )

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
        self.pedestal = self._build_pedestal()

    def _build_pedestal(self):
        builder = self.scene.create_actor_builder()
        half = np.array([0.03, 0.03, PEDESTAL_POS[2] / 2.0])
        pose = sapien.Pose([PEDESTAL_POS[0], PEDESTAL_POS[1], PEDESTAL_POS[2] / 2.0])
        builder.add_box_collision(pose, half)
        builder.add_box_visual(pose, half, material=_material([0.45, 0.45, 0.45, 1.0]))
        return builder.build_kinematic("peg_pedestal")

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
        if self.keyed:
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
        return builder.build("circular_peg" if not self.keyed else "keyed_circular_peg")

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

        gate_half_height = 0.5 * (GATE_Z_MAX - GATE_Z_MIN)
        gate_z = 0.5 * (GATE_Z_MIN + GATE_Z_MAX)
        if self.keyed:
            inner_x = KEY_HALF_X + GATE_CLEARANCE
            inner_y = KEY_HALF_Y + GATE_CLEARANCE
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
        else:
            ring_radius = 0.5 * (CIRCULAR_GATE_INNER_RADIUS + SOCKET_OUTER_RADIUS)
            ring_thickness = 0.5 * (SOCKET_OUTER_RADIUS - CIRCULAR_GATE_INNER_RADIUS)
            segment_half_length = 2.0 * np.pi * ring_radius / 24.0 * 0.58
            for i in range(24):
                angle = 2.0 * np.pi * i / 24.0
                pose = sapien.Pose(
                    [ring_radius * np.cos(angle), ring_radius * np.sin(angle), gate_z],
                    q=euler2quat(0.0, 0.0, angle),
                )
                half_size = [ring_thickness, segment_half_length, gate_half_height]
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
                raise ValueError("causal_delta must be [du, dv, d_axial] (task-local)")
            self.causal_delta = causal_delta
            self.socket_yaw = float(causal_delta[2])

            # Peg start: FIXED grasp position on the elevated pedestal, axis aligned
            # with the insertion direction Q. This decouples the pickup geometry
            # from the socket placement and keeps the peg high enough for the
            # along-axis (end-on) grasp to be reachable for every orientation.
            peg_pos = np.array(
                [PEDESTAL_POS[0], PEDESTAL_POS[1], PEDESTAL_POS[2] + self._resting_height()],
                dtype=np.float64,
            )
            peg_p = torch.tensor(
                peg_pos, dtype=torch.float32, device=self.device
            ).repeat(len(env_idx), 1)
            peg_q = torch.tensor(
                self.orientation, dtype=torch.float32, device=self.device
            ).repeat(len(env_idx), 1)
            self.peg.set_pose(Pose.create_from_pq(peg_p, peg_q))

            # Socket: nominal center plus the task-local lateral intervention,
            # then rotated by Q; axial yaw is composed in the rotated local frame.
            socket_p = torch.tensor(
                self._to_world([causal_delta[0], causal_delta[1], 0.0]),
                dtype=torch.float32,
                device=self.device,
            ).repeat(len(env_idx), 1)
            socket_q = torch.tensor(
                qmult(self.orientation, _yaw_quat(self.socket_yaw)),
                dtype=torch.float32,
                device=self.device,
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
            # Use the global numpy RNG (seeded by the collector's np.random.seed)
            # for the robot-init noise. ManiSkill's _episode_rng is not reliably
            # reseeded through the gym Wrapper, which made seed-varied rollouts
            # byte-identical.
            noise = np.random.normal(
                0.0, self.robot_init_qpos_noise, (len(env_idx), len(qpos))
            )
            qpos = qpos + noise
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0.0, 0.0]))

    def target_pose_at(self, d: float, yaw: float):
        # d = depth along the insertion axis (task-local z); yaw = axial rotation.
        p_world = self._to_world([self.causal_delta[0], self.causal_delta[1], d])
        q_world = qmult(self.orientation, _yaw_quat(yaw))
        p = torch.tensor(p_world, dtype=torch.float32, device=self.device).reshape(1, 3)
        q = torch.tensor(q_world, dtype=torch.float32, device=self.device).reshape(1, 4)
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
        axis = torch.tensor(
            self.insertion_axis, dtype=torch.float32, device=self.device
        )
        peg_local_z = torch.sum((self.peg.pose.p - self.socket.pose.p) * axis, dim=1)
        return GATE_Z_MIN - (peg_local_z + KEY_Z_MAX)

    def evaluate(self):
        peg = self.peg.pose
        goal = self.goal_pose
        pos_err = torch.linalg.norm(peg.p - goal.p, axis=1)
        axis_err = torch.tensor(
            [_axis_error_from_wxyz(q, self.insertion_axis) for q in peg.q.detach().cpu().numpy()],
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


@register_env("CircularPhaseSwitch-v1", max_episode_steps=1500)
class CircularPhaseSwitchEnv(KeyedCircularPhaseSwitchEnv):
    """Circular peg in a circular bore: full SO(2) axial-yaw symmetry.

    Identical to KeyedCircularPhaseSwitch-v1 except the key and keyway are
    removed, so the axial-yaw generator is irrelevant throughout: the task-local
    trajectory is invariant under the intervention d_axial (a gauge symmetry).
    """

    def __init__(self, *args, **kwargs):
        kwargs["keyed"] = False
        super().__init__(*args, **kwargs)
