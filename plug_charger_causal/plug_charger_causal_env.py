from __future__ import annotations

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.plug_charger import PlugChargerEnv
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("PlugChargerCausal-v1", max_episode_steps=200)
class PlugChargerCausalEnv(PlugChargerEnv):
    """PlugCharger-v1 with explicit receptacle pose interventions.

    Reset options:
        causal_delta: [delta_x_m, delta_y_m, delta_yaw_rad]

    The perturbation is applied after the official random reset. Translation is
    in world x/y. Yaw is composed with the current receptacle orientation.
    """

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        causal_delta = options.get("causal_delta") if options is not None else None
        if causal_delta is None:
            return

        if len(causal_delta) != 3:
            raise ValueError(
                "causal_delta must be [delta_x_m, delta_y_m, delta_yaw_rad]"
            )

        dx, dy, dyaw = [float(v) for v in causal_delta]
        pose = self.receptacle.pose
        pos = pose.p.clone()
        quat = pose.q.clone()

        pos[:, 0] += dx
        pos[:, 1] += dy

        delta_q = torch.tensor(
            euler2quat(0.0, 0.0, dyaw),
            dtype=quat.dtype,
            device=quat.device,
        ).repeat(len(pos), 1)
        quat = rotation_conversions.quaternion_multiply(delta_q, quat)
        quat = quat / torch.linalg.norm(quat, dim=1, keepdim=True)

        self.receptacle.set_pose(Pose.create_from_pq(pos, quat))
        self.goal_pose = self.receptacle.pose * sapien.Pose(
            q=euler2quat(0.0, 0.0, np.pi)
        )
