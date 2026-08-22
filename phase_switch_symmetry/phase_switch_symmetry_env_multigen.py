from __future__ import annotations

"""Multi-generator SE(3) env: du AND yaw are BOTH selective (additive supplement).

KeyedCircularPhaseSwitchSE3Multigen-v1 is identical to the frozen
KeyedCircularPhaseSwitchSE3-v1 except the circular bore below the keyed gate is
replaced by a RECTANGULAR SLOT that is wide in du (x) and narrow in dv (y).

Physical story: the keyed gate constrains the peg's yaw AND lateral position
during entry (identical to the keyed task). Once the key clears the gate, the
slot releases TWO generators:
  - yaw: the slot's cross-section accommodates the key's axis-aligned envelope
    at every relative yaw in the +-30 deg intervention range (the slot rotates
    with the socket, so the relaxed peg sits at relative yaw -socket_yaw; the
    envelope is 0.021|sin y| + 0.014|cos y| <= 0.0226 < SLOT_HALF_Y), so the
    expert relaxes yaw to 0 as in the keyed task;
  - du:  the slot's wide x direction (SLOT_HALF_X) admits the full +-0.012 m
    du relaxation (key 0.021 + offset 0.012 + envelope margin < 0.044), so the
    expert ALSO relaxes du to nominal;
while dv stays tracked: relaxing dv by the full intervention would push the
key's edge to 0.014 + 0.012 = 0.026 = SLOT_HALF_Y (the wall), so the expert
keeps dv matched.

Oracle over the four insertion phases (align_keyed/enter_key/unlock_yaw/
circular_insert):

    du   [1,1,0,0]    (selective, NEW)
    dv   [1,1,1,1]    (tracked)
    dw   [1,1,1,1]
    roll [1,1,1,1]
    pitch[1,1,1,1]
    yaw  [1,1,0,0]    (selective, as in the keyed task)

The multi-generator claim is that the model recovers BOTH selective generators
(alpha_du > 0 AND alpha_yaw > 0 pre-clearance, both dropping post-clearance)
rather than a single winner. This module lives in its own file so the frozen
phase_switch_symmetry_env.py keeps its recorded sha256.
"""

import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien

import numpy as np
import sapien
import torch
from transforms3d.quaternions import qmult, quat2mat

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from phase_switch_symmetry_env import (
    FINAL_PEG_Z,
    GATE_CLEARANCE,
    GATE_Z_MAX,
    GATE_Z_MIN,
    KEY_CLEAR_PEG_Z,
    KEY_HALF_X,
    KEY_HALF_Y,
    KeyedCircularPhaseSwitchSE3Env,
    SOCKET_CENTER,
    SOCKET_OUTER_RADIUS,
    _local_quat,
    _material,
)

# Slot cross-section (socket-local x = du, y = dv):
#  - x half-width admits the key (0.021) plus the full +-0.012 m du relaxation
#    with envelope margin -> 0.044.
#  - y half-height fits the key's axis-aligned envelope at the largest relative
#    yaw in the +-30 deg range (0.021 sin30 + 0.014 cos30 = 0.0226) but is too
#    narrow for a full dv relaxation (0.014 + 0.012 = 0.026 = wall), so dv stays
#    tracked.
SLOT_HALF_X = 0.044
SLOT_HALF_Y = 0.026


@register_env("KeyedCircularPhaseSwitchSE3Multigen-v1", max_episode_steps=1500)
class KeyedCircularPhaseSwitchSE3MultigenEnv(KeyedCircularPhaseSwitchSE3Env):
    """SE(3) keyed peg -> rectangular slot: du and yaw are both selective."""

    def _build_socket(self):
        builder = self.scene.create_actor_builder()
        bore_mat = _material([0.20, 0.52, 0.47, 1.0])
        gate_mat = _material([0.28, 0.37, 0.58, 1.0])
        outer = SOCKET_OUTER_RADIUS

        # Rectangular slot bore (replaces the circular ring bore): four box
        # walls, mirroring the keyed-gate construction.
        bore_half_height = 0.5 * GATE_Z_MIN
        inner_x, inner_y = SLOT_HALF_X, SLOT_HALF_Y
        side_thickness = 0.5 * (outer - inner_x)
        cap_thickness = 0.5 * (outer - inner_y)
        slot_parts = [
            ([side_thickness, outer, bore_half_height],
             [inner_x + side_thickness, 0.0, bore_half_height]),
            ([side_thickness, outer, bore_half_height],
             [-inner_x - side_thickness, 0.0, bore_half_height]),
            ([inner_x, cap_thickness, bore_half_height],
             [0.0, inner_y + cap_thickness, bore_half_height]),
            ([inner_x, cap_thickness, bore_half_height],
             [0.0, -inner_y - cap_thickness, bore_half_height]),
        ]
        for half_size, position in slot_parts:
            pose = sapien.Pose(position)
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=bore_mat)

        # Keyed gate: identical to the frozen keyed socket (4 walls).
        gate_half_height = 0.5 * (GATE_Z_MAX - GATE_Z_MIN)
        gate_z = 0.5 * (GATE_Z_MIN + GATE_Z_MAX)
        gate_inner_x = KEY_HALF_X + GATE_CLEARANCE
        gate_inner_y = KEY_HALF_Y + GATE_CLEARANCE
        gate_side_thickness = 0.5 * (outer - gate_inner_x)
        gate_cap_thickness = 0.5 * (outer - gate_inner_y)
        gate_parts = [
            ([gate_side_thickness, outer, gate_half_height],
             [gate_inner_x + gate_side_thickness, 0.0, gate_z]),
            ([gate_side_thickness, outer, gate_half_height],
             [-gate_inner_x - gate_side_thickness, 0.0, gate_z]),
            ([gate_inner_x, gate_cap_thickness, gate_half_height],
             [0.0, gate_inner_y + gate_cap_thickness, gate_z]),
            ([gate_inner_x, gate_cap_thickness, gate_half_height],
             [0.0, -gate_inner_y - gate_cap_thickness, gate_z]),
        ]
        for half_size, position in gate_parts:
            pose = sapien.Pose(position)
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=gate_mat)

        builder.initial_pose = sapien.Pose(p=SOCKET_CENTER)
        return builder.build_kinematic("keyed_to_slot_multigen_socket")

    def target_pose_at_relaxed_du(self, d: float, roll: float, pitch: float, yaw: float):
        """Peg target at depth d with du relaxed to nominal (du = 0).

        Same as target_pose_at, but the socket centre is computed with du = 0,
        so the peg returns to the task anchor's x coordinate while dv/dw stay
        tracked and the tilt/yaw are explicit.
        """
        socket_center = self._to_world(
            [0.0, self.causal_delta[1], self.causal_delta[2]]
        )
        axis_local = quat2mat(_local_quat(roll, pitch, 0.0)) @ np.array([0.0, 0.0, 1.0])
        axis_world = self.Q_mat @ axis_local
        p_world = socket_center + d * axis_world
        q_world = qmult(self.orientation, _local_quat(roll, pitch, yaw))
        p = torch.tensor(p_world, dtype=torch.float32, device=self.device).reshape(1, 3)
        q = torch.tensor(q_world, dtype=torch.float32, device=self.device).reshape(1, 4)
        return Pose.create_from_pq(p, q)

    @property
    def circular_transition_pose(self):
        r, p, _ = self.socket_rpy
        return self.target_pose_at_relaxed_du(KEY_CLEAR_PEG_Z, r, p, 0.0)

    @property
    def goal_pose(self):
        r, p, _ = self.socket_rpy
        return self.target_pose_at_relaxed_du(FINAL_PEG_Z, r, p, 0.0)
