# Planar-Push Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second, physically distinct task — non-prehensive planar pushing of a cuboid block across a table to a ghost 6-DOF target — that the frozen SE(3) Pdiag model must solve alongside the frozen peg-in-hole task, recovering out-of-plane generators (`dw,roll,pitch`) as suppressed while keeping in-plane generators (`du,dv,yaw`) tracked.

**Architecture:** Purely additive, mirroring the frozen multigen supplement's discipline. A new env `PlanarPush-v1` (own file) exposes a table, a dynamic rectangular block, and a kinematic ghost target. A new collector/solver pushes the block (never grasped) through four phases `reach→push→align→retract` mapped onto the frozen raw phase codes `3,4,5,6` so the frozen `usable`/`complete`/`progress_grid` machinery works unchanged. A new benchmark reuses the frozen 4-model SE(3) suite (`build_models`) against `block_pose`/`target_pose`, with a push-specific oracle and selector metrics.

**Tech Stack:** Python 3, ManiSkill 3.0.1 (SAPIEN/PhysX), NumPy, SciPy, pandas, h5py, transforms3d. Conda env `maniskill_download`.

**Spec:** [docs/superpowers/specs/2026-08-30-planar-push-task-design.md](../specs/2026-08-30-planar-push-task-design.md)

## Global Constraints

- **Frozen-file discipline (hard):** `phase_switch_symmetry_env.py`, `phase_switch_se3_baselines.py`, `benchmark_se3_transfer.py`, `benchmark_phase_switch_baselines.py`, `collect_phase_switch_rollouts.py`, `collect_phase_switch_rotated.py`, `analyze_phase_switch_rollouts.py` and all `phase_switch_symmetry_multiseed/*` artifacts stay **byte-identical**. All new work lives in NEW files that import from them.
- **Generator basis (order):** `[du, dv, dw, roll, pitch, yaw]`, `SE3_DIM=6`, `YAW_INDEX=5`, `DW_INDEX=2`, `ROLL_INDEX=3`, `PITCH_INDEX=4`.
- **Oracle (frozen in the spec):** `heading_push = [1,1,0,0,0,1]`, `free_yaw_push = [1,1,0,0,0,0]` as the *selector* (which generators are ever tracked). Phasewise detail below.
- **Seeds:** `SEEDS = [20260818, 20270818, 20280818]`. **Conda:** `source ~/miniconda3/etc/profile.d/conda.sh && conda activate maniskill_download`.
- **Model suite (reused, unmodified):** `SE3FrameWeightedModel`, `SE3FullOperatorModel`, `SE3DiagonalOperatorModel`, `SE3SmoothFinitePDiagModel`, via `build_models(nominal_frame_pose, pdiag_config)`. Pdiag config: `alpha_max=1.25, n_basis=24, basis_width=0.065, smoothness_weight=0.1, nominal_iterations=3`.
- **Firm gripper:** `Panda.gripper_stiffness = 2.5e3`, `Panda.gripper_force_limit = 150.0` (same as frozen).
- **Phase codes:** the four push phases MUST use raw `solver_phase` codes `3,4,5,6` (frozen `complete()` requires all of `REQUIRED_PHASES=(3,4,5,6)`; `progress_grid` re-indexes them `0..3`).
- **Active phases for scoring:** raw `4` (push) and `5` (align) = re-indexed `1,2`. `reach` (3) and `retract` (6) are masked from threshold metrics.

---

## File Structure

**New (all under `phase_switch_symmetry/` unless noted):**
- `phase_switch_symmetry_planar_push_env.py` — `PlanarPush-v1` env (table + block + ghost target).
- `collect_planar_push_rollouts.py` — `solve_planar_push`, `PlanarPushTraceWrapper`, `write_push_episode`, `collect_planar_push`, `main`.
- `probe_planar_push.py` — Phase-0 feasibility gate.
- `generate_planar_push_contexts.py` — frozen 75-condition 6-vector manifest.
- `prepare_planar_push_subsets.py` — frozen rank-6 subsets (N ∈ {8,15,30}).
- `benchmark_planar_push.py` — oracle + metrics + 4-model fits.
- `make_planar_push_figure.py` — recovered-α figure.
- `phase_switch_symmetry_multiseed/planar_push/` — frozen artifacts (`planar_push_experiment.json`, `planar_push_fixed_contexts.json`, `planar_push_subsets.json`, fits CSVs, `VALIDATION_planar_push.md`).
- `phase_switch_symmetry_rollouts_planar_push/` — rollout H5s `{heading_push,free_yaw_push}_seed_{seed}.h5`.

**Reuse (import, do not modify):** `phase_switch_symmetry_env` (`_local_quat`, `_material`, `TableSceneBuilder` via mani_skill), `phase_switch_se3_baselines` (all SE(3) math + models), `benchmark_se3_transfer` (`build_models`, `SE3_MODEL_ORDER`, `SEEDS`, `json_ready`, `sha256`, `pose_to_pose6`), `benchmark_phase_switch_baselines` (`PHASE_CODES`, `progress_grid`, `usable`), `analyze_phase_switch_rollouts` (`resample`, `wrap_pi`), `collect_phase_switch_rollouts` (`EpisodeFinished`), `collect_phase_switch_rotated` (`intervention_rows_se3`).

---

## Task 1: `PlanarPush-v1` environment

**Files:**
- Create: `phase_switch_symmetry/phase_switch_symmetry_planar_push_env.py`
- Test: `phase_switch_symmetry/test_planar_push_env.py`

**Interfaces:**
- Consumes: `_local_quat`, `_material` from `phase_switch_symmetry_env`; `TableSceneBuilder` from `mani_skill.utils.scene_builder.table`.
- Produces: `PlanarPushEnv` with `self.block` (dynamic), `self.target` (kinematic ghost), `self.causal_delta` (6-vec), `self.goal_heading` (bool), `self.Q_mat`, `self.task_anchor`, `goal_pose` property, `block_goal_position()`, `evaluate()`, `_get_obs_extra`. Constants `BLOCK_HALF_X`, `BLOCK_HALF_Y`, `BLOCK_HALF_Z`, `TARGET_POS`, `BLOCK_START`, `PUSH_SUCCESS_POS_TOL`, `PUSH_SUCCESS_YAW_TOL`.

- [ ] **Step 1: Write the failing test** (geometry constants + frame mapping)

```python
# test_planar_push_env.py
import numpy as np
import phase_switch_symmetry_planar_push_env as pp


def test_block_footprint_is_non_square():
    # Spec §4.1: rectangular, clearly non-square footprint, 2:1 aspect.
    assert pp.BLOCK_HALF_X == 0.02
    assert pp.BLOCK_HALF_Y == 0.01
    assert pp.BLOCK_HALF_X != pp.BLOCK_HALF_Y


def test_to_world_identity_maps_local_to_anchor_plus_offset():
    env = pp.PlanarPushEnv(
        num_envs=1, obs_mode="state_dict", control_mode="pd_joint_pos",
        reward_mode="sparse", sim_backend="physx_cpu", render_mode=None,
        orientation=[1.0, 0.0, 0.0, 0.0], task_anchor=pp.TARGET_POS,
    )
    out = env._to_world([0.01, -0.02, 0.03])
    assert np.allclose(out, pp.TARGET_POS + np.array([0.01, -0.02, 0.03]))


def test_block_goal_position_is_in_plane_only():
    env = pp.PlanarPushEnv(
        num_envs=1, obs_mode="state_dict", control_mode="pd_joint_pos",
        reward_mode="sparse", sim_backend="physx_cpu", render_mode=None,
        orientation=[1.0, 0.0, 0.0, 0.0], task_anchor=pp.TARGET_POS,
    )
    env.causal_delta = np.array([0.005, -0.004, 0.012, 0.1, 0.2, 0.3])
    p = env.block_goal_position()
    # in-plane follows du,dv; z pinned to block height (dw ignored)
    assert np.allclose(p[:2], pp.TARGET_POS[:2] + np.array([0.005, -0.004]))
    assert np.isclose(p[2], pp.BLOCK_HALF_Z)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/rocos/sia/maniskill_spectrum && source ~/miniconda3/etc/profile.d/conda.sh && conda activate maniskill_download && python -m pytest phase_switch_symmetry/test_planar_push_env.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'phase_switch_symmetry_planar_push_env'`.

- [ ] **Step 3: Implement the environment**

```python
# phase_switch_symmetry_planar_push_env.py
from __future__ import annotations
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

# Rectangular (non-square footprint) block: 0.04 x 0.02 x 0.02 m.
BLOCK_HALF_X = 0.02
BLOCK_HALF_Y = 0.01
BLOCK_HALF_Z = 0.01
# Nominal ghost-target centre (block height, so the ghost marks the block's
# final in-plane pose at dw=0). task_anchor z = BLOCK_HALF_Z.
TARGET_POS = np.array([0.10, 0.0, BLOCK_HALF_Z], dtype=np.float64)
# Fixed block start (pushed toward +x). Reachable from the Panda base (-0.615,0,0).
BLOCK_START = np.array([-0.10, 0.0, BLOCK_HALF_Z], dtype=np.float64)
PUSH_SUCCESS_POS_TOL = 0.008
PUSH_SUCCESS_YAW_TOL = 0.10


def _ghost_material():
    return sapien.render.RenderMaterial(
        base_color=[0.2, 0.6, 0.9, 0.35], roughness=0.6, specular=0.2
    )


@register_env("PlanarPush-v1", max_episode_steps=1500)
class PlanarPushEnv(BaseEnv):
    """Non-prehensive planar push of a cuboid block to a ghost 6-DOF target.

    ``causal_delta = [du, dv, dw, d_roll, d_pitch, d_yaw]`` is a task-local
    SE(3) intervention on the ghost target's pose. du,dv are in-plane
    translation, yaw is rotation about the table normal (all TRACKED); dw is
    vertical translation, roll/pitch are tilts (all SUPPRESSED — the block
    stays on the table, level). The block is never grasped: the solver pushes
    it across the table to the target's in-plane pose.
    """

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
            raise ValueError("orientation must be a [w,x,y,z] quaternion")
        orientation = orientation / np.linalg.norm(orientation)
        self.orientation = orientation
        self.Q_mat = quat2mat(orientation)
        if task_anchor is None:
            task_anchor = TARGET_POS
        self.task_anchor = np.asarray(task_anchor, dtype=np.float64)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _to_world(self, local_rel):
        return self.task_anchor + self.Q_mat @ np.asarray(local_rel, dtype=np.float64)

    def block_goal_position(self):
        """Block final in-plane position: target du,dv at block height (dw ignored)."""
        target = self._to_world([self.causal_delta[0], self.causal_delta[1], 0.0])
        target[2] = BLOCK_HALF_Z
        return target

    @property
    def goal_pose(self):
        yaw = float(self.causal_delta[5]) if self.goal_heading else 0.0
        p = self.block_goal_position()
        q = qmult(self.orientation, _local_quat(0.0, 0.0, yaw))
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
        builder.add_box_visual(
            sapien.Pose(), half, material=_ghost_material()
        )
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

            # Block: FIXED start pose on the table (heading = nominal Q).
            block_p = torch.tensor(
                BLOCK_START, dtype=torch.float32, device=self.device
            ).repeat(len(env_idx), 1)
            block_q = torch.tensor(
                self.orientation, dtype=torch.float32, device=self.device
            ).repeat(len(env_idx), 1)
            self.block.set_pose(Pose.create_from_pq(block_p, block_q))

            # Ghost target: C0 o intervention = task_anchor + Q@[du,dv,dw],
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
        pos_err = torch.linalg.norm(block.p[..., :2] - goal_p[..., :2], axis=1)
        heading_err = torch.zeros_like(pos_err)
        if self.goal_heading:
            # Block heading vs target heading (yaw), wrapped to [-pi, pi].
            def _yaw_of(q_wxyz):
                w, x, y, z = q_wxyz
                return np.arctan2(
                    2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
                )
            target_yaw = float(_yaw_of(qmult(self.orientation, _local_quat(0, 0, yaw := 0.0)))) if False else float(self.causal_delta[5])
            block_yaws = torch.tensor(
                [_yaw_of(q) for q in block.q.detach().cpu().numpy()],
                dtype=pos_err.dtype, device=pos_err.device,
            )
            heading_err = torch.remainder(block_yaws - target_yaw + np.pi, 2 * np.pi) - np.pi
            heading_err = heading_err.abs()
        success = (pos_err < PUSH_SUCCESS_POS_TOL) & (heading_err < PUSH_SUCCESS_YAW_TOL)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/rocos/sia/maniskill_spectrum && source ~/miniconda3/etc/profile.d/conda.sh && conda activate maniskill_download && python -m pytest phase_switch_symmetry/test_planar_push_env.py -q`
Expected: PASS (2 unit tests on constants/frame math; `PlanarPushEnv.__init__` is instantiated directly without a scene so it exercises only the frame helpers).

- [ ] **Step 5: Smoke reset (integration)**

Run: `cd /home/rocos/sia/maniskill_spectrum && source ~/miniconda3/etc/profile.d/conda.sh && conda activate maniskill_download && python -c "import gymnasium as gym; import mani_skill.envs; import phase_switch_symmetry_planar_push_env; e = gym.make('PlanarPush-v1', num_envs=1, obs_mode='state_dict', control_mode='pd_joint_pos', reward_mode='sparse', sim_backend='physx_cpu'); o,_ = e.reset(seed=0, options={'causal_delta':[0.01,-0.01,0.01,0.1,0.1,0.2]}); print('block', o['agent'].shape, 'ok')"`
Expected: env builds and resets without exception; block and ghost target actors present.

- [ ] **Step 6: Commit**

```bash
git add phase_switch_symmetry/phase_switch_symmetry_planar_push_env.py phase_switch_symmetry/test_planar_push_env.py
git commit -m "feat: add PlanarPush-v1 env (table + block + ghost target)"
```

---

## Task 2: Frozen context manifest

**Files:**
- Create: `phase_switch_symmetry/generate_planar_push_contexts.py`
- Create (output): `phase_switch_symmetry_multiseed/planar_push/planar_push_fixed_contexts.json`

**Interfaces:**
- Consumes: `intervention_rows_se3` from `collect_phase_switch_rotated` (reused verbatim: 1 baseline + 14 isolated + 60 mixed = 75 conditions).
- Produces: a manifest JSON with `env_id="PlanarPush-v1"`, `generator_basis`, `conditions` (contiguous `condition_id`, 6-vector `causal_delta`), and a `geometry` block of push constants.

- [ ] **Step 1: Write the failing test**

```python
# test_planar_push_contexts.py
import json, hashlib
from pathlib import Path
import numpy as np


def test_manifest_shape():
    path = Path("phase_switch_symmetry_multiseed/planar_push/planar_push_fixed_contexts.json")
    m = json.loads(path.read_text())
    conds = m["conditions"]
    assert len(conds) == 75
    assert sum(c["generator"] == "baseline" for c in conds) == 1
    assert sum(c["generator"] == "mixed" for c in conds) == 60
    assert sum(c["generator"] in {"du","dv","dw","roll","pitch","yaw"} for c in conds) == 14
    ids = [c["condition_id"] for c in conds]
    assert ids == list(range(75))
    assert all(np.asarray(c["causal_delta"]).shape == (6,) for c in conds)
    assert all(np.isfinite(c["causal_delta"]).all() for c in conds)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/rocos/sia/maniskill_spectrum && source ~/miniconda3/etc/profile.d/conda.sh && conda activate maniskill_download && python -m pytest phase_switch_symmetry/test_planar_push_contexts.py -q`
Expected: FAIL — manifest file does not exist.

- [ ] **Step 3: Implement the generator** (mirror `generate_se3_contexts.py`, own file)

```python
# generate_planar_push_contexts.py
import argparse, json
from pathlib import Path
import numpy as np
import phase_switch_symmetry_planar_push_env as pp
from collect_phase_switch_rotated import intervention_rows_se3

PUSH_GEOMETRY = {
    "BLOCK_HALF_X": pp.BLOCK_HALF_X,
    "BLOCK_HALF_Y": pp.BLOCK_HALF_Y,
    "BLOCK_HALF_Z": pp.BLOCK_HALF_Z,
    "TARGET_POS": pp.TARGET_POS.tolist(),
    "BLOCK_START": pp.BLOCK_START.tolist(),
    "PUSH_SUCCESS_POS_TOL": pp.PUSH_SUCCESS_POS_TOL,
    "PUSH_SUCCESS_YAW_TOL": pp.PUSH_SUCCESS_YAW_TOL,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed-samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", type=Path,
                        default=Path("phase_switch_symmetry_multiseed/planar_push/planar_push_fixed_contexts.json"))
    args = parser.parse_args()
    rows = intervention_rows_se3(args.mixed_samples, args.seed)
    conditions = []
    for condition_id, row in enumerate(rows):
        cd = np.asarray(row["causal_delta"], dtype=np.float64)
        if cd.shape != (6,) or not np.isfinite(cd).all():
            raise ValueError(f"row {condition_id} is not a finite 6-vector")
        conditions.append({"condition_id": condition_id, "generator": row["generator"],
                           "causal_delta": cd.tolist()})
    manifest = {
        "schema_version": 2,
        "env_id": "PlanarPush-v1",
        "generator_basis": ["du", "dv", "dw", "d_roll", "d_pitch", "d_yaw"],
        "generator_scale": {"translation_m": 0.012, "roll_pitch_deg": 15.0, "yaw_deg": 30.0},
        "mixed_samples": args.mixed_samples, "seed": args.seed,
        "geometry": PUSH_GEOMETRY, "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("saved:", args.output, "conditions:", len(conditions))
```

- [ ] **Step 4: Generate the manifest**

Run: `cd /home/rocos/sia/maniskill_spectrum && source ~/miniconda3/etc/profile.d/conda.sh && conda activate maniskill_download && python phase_switch_symmetry/generate_planar_push_contexts.py`

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/rocos/sia/maniskill_spectrum && source ~/miniconda3/etc/profile.d/conda.sh && conda activate maniskill_download && python -m pytest phase_switch_symmetry/test_planar_push_contexts.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add phase_switch_symmetry/generate_planar_push_contexts.py phase_switch_symmetry/test_planar_push_contexts.py phase_switch_symmetry_multiseed/planar_push/planar_push_fixed_contexts.json
git commit -m "feat: freeze planar-push 75-condition context manifest"
```

---

## Task 3: Experiment preregistration + subsets

**Files:**
- Create: `phase_switch_symmetry_multiseed/planar_push/planar_push_experiment.json`
- Create: `phase_switch_symmetry/prepare_planar_push_subsets.py`
- Create (output): `phase_switch_symmetry_multiseed/planar_push/planar_push_subsets.json`

**Interfaces:**
- Consumes: the manifest sha256 (Task 2); `sha256` from `benchmark_se3_transfer`.
- Produces: `planar_push_experiment.json` (preregisters manifest sha256 + seeds + arms) and `planar_push_subsets.json` (18 subsets: N ∈ {8,15,30} × {random,qualified} × 3, rank-6, cond < 10).

- [ ] **Step 1: Write the experiment JSON** (mirror `se3_experiment.json`, push-specific)

```json
{
  "schema_version": 2,
  "status": "preregistered_before_collection",
  "title": "Planar-push task supplement (Task 2)",
  "context_manifest": "/home/rocos/sia/maniskill_spectrum/phase_switch_symmetry_multiseed/planar_push/planar_push_fixed_contexts.json",
  "context_manifest_sha256": "<FILL from: sha256sum .../planar_push_fixed_contexts.json>",
  "env_id": "PlanarPush-v1",
  "generator_basis": ["du", "dv", "dw", "d_roll", "d_pitch", "d_yaw"],
  "orientation": {"Q1": [1.0, 0.0, 0.0, 0.0]},
  "task_anchor": [0.10, 0.0, 0.01],
  "seeds": [20260818, 20270818, 20280818],
  "retries_per_condition": 5,
  "robot_init_qpos_noise_rad": 0.01,
  "gripper": {"stiffness": 2500.0, "force_limit": 150.0},
  "arm_definition": {
    "heading_push": {"goal_heading": true, "solver": "solve_planar_push", "oracle": [1,1,0,0,0,1]},
    "free_yaw_push": {"goal_heading": false, "solver": "solve_planar_push", "oracle": [1,1,0,0,0,0]}
  }
}
```

Fill the sha256 with: `sha256sum phase_switch_symmetry_multiseed/planar_push/planar_push_fixed_contexts.json`.

- [ ] **Step 2: Implement the subset generator** (mirror `prepare_se3_subsets.py`, own file)

```python
# prepare_planar_push_subsets.py
import argparse, hashlib, json
from pathlib import Path
import numpy as np

SAMPLE_SIZES = [8, 15, 30]
CONTEXT_SCALE = np.array([0.012, 0.012, 0.012, np.deg2rad(15.0), np.deg2rad(15.0), np.deg2rad(30.0)])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostics(contexts, indices):
    selected = contexts[np.asarray(indices, dtype=int)] / CONTEXT_SCALE
    return {"rank": int(np.linalg.matrix_rank(selected)),
            "augmented_intercept_rank": int(np.linalg.matrix_rank(np.column_stack([np.ones(len(selected)), selected]))),
            "condition_number": float(np.linalg.cond(selected))}


def draw_unique(rng, sample_size, count, predicate=lambda _: True):
    selected, seen, attempts = [], set(), 0
    while len(selected) < count:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError(f"could not draw {count} unique qualified subsets for N={sample_size}")
        indices = tuple(sorted(rng.choice(60, sample_size, replace=False).tolist()))
        if indices in seen or not predicate(indices):
            continue
        seen.add(indices)
        selected.append(indices)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path,
                        default=Path("phase_switch_symmetry_multiseed/planar_push/planar_push_experiment.json"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--selection-seed", type=int, default=20260818)
    parser.add_argument("--condition-threshold", type=float, default=10.0)
    args = parser.parse_args()

    with args.experiment.open(encoding="utf-8") as f:
        experiment = json.load(f)
    context_path = Path(experiment["context_manifest"])
    if sha256(context_path) != experiment["context_manifest_sha256"]:
        raise RuntimeError("context manifest changed after preregistration")
    with context_path.open(encoding="utf-8") as f:
        cm = json.load(f)
    mixed_rows = [r for r in cm["conditions"] if r["generator"] == "mixed"]
    contexts = np.asarray([r["causal_delta"] for r in mixed_rows])
    source_cids = [int(r["condition_id"]) for r in mixed_rows]
    if len(contexts) != 60:
        raise RuntimeError(f"requires 60 fixed mixed contexts, found {len(contexts)}")

    rng = np.random.default_rng(args.selection_seed)
    subsets, subset_id = [], 0
    for n in SAMPLE_SIZES:
        random_subsets = draw_unique(rng, n, args.repeats)
        qualified_subsets = draw_unique(rng, n, args.repeats, predicate=lambda idx: (
            diagnostics(contexts, idx)["rank"] == 6
            and diagnostics(contexts, idx)["condition_number"] < args.condition_threshold))
        for protocol, ps in [("random", random_subsets), ("qualified", qualified_subsets)]:
            for repeat, indices in enumerate(ps):
                subsets.append({"subset_id": subset_id, "protocol": protocol, "sample_size": n,
                                "repeat": repeat, "mixed_indices": list(indices),
                                "source_condition_ids": [source_cids[i] for i in indices],
                                **diagnostics(contexts, indices)})
                subset_id += 1
    out = {"schema_version": 2, "status": "frozen_before_planar_push_fitting",
           "experiment": str(args.experiment.resolve()), "experiment_sha256": sha256(args.experiment),
           "context_manifest_sha256": experiment["context_manifest_sha256"],
           "selection_seed": args.selection_seed, "sample_sizes": SAMPLE_SIZES,
           "repeats_per_protocol": args.repeats, "context_scale": CONTEXT_SCALE.tolist(),
           "qualification": {"rank": 6, "condition_number_strictly_below": args.condition_threshold,
                             "augmented_intercept_rank_is_reported_but_not_filtered": True},
           "policies": {"same_subset_indices_across_execution_seeds": True,
                        "same_subset_indices_across_tasks": True,
                        "random_protocol_drops_no_subsets": True,
                        "fit_failures_are_retained": True, "no_new_rollouts": True},
           "subsets": subsets}
    out_path = args.experiment.parent / "planar_push_subsets.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("saved:", out_path, "subsets:", len(subsets))
```

- [ ] **Step 3: Generate subsets**

Run: `cd /home/rocos/sia/maniskill_spectrum && source ~/miniconda3/etc/profile.d/conda.sh && conda activate maniskill_download && python phase_switch_symmetry/prepare_planar_push_subsets.py`
Expected: prints `subsets: 18`; every qualified subset has `rank == 6` and `condition_number < 10`.

- [ ] **Step 4: Verify subset invariants**

Run: `cd /home/rocos/sia/maniskill_spectrum && python -c "import json; s=json.load(open('phase_switch_symmetry_multiseed/planar_push/planar_push_subsets.json')); q=[x for x in s['subsets'] if x['protocol']=='qualified']; assert len(s['subsets'])==18; assert all(x['rank']==6 and x['condition_number']<10 for x in q); print('18 subsets, qualified rank-6 cond<10: OK')"`
Expected: `18 subsets, qualified rank-6 cond<10: OK`.

- [ ] **Step 5: Commit**

```bash
git add phase_switch_symmetry/prepare_planar_push_subsets.py phase_switch_symmetry_multiseed/planar_push/planar_push_experiment.json phase_switch_symmetry_multiseed/planar_push/planar_push_subsets.json
git commit -m "feat: preregister planar-push experiment and freeze rank-6 subsets"
```

---

## Task 4: Solver + collector

**Files:**
- Create: `phase_switch_symmetry/collect_planar_push_rollouts.py`

**Interfaces:**
- Consumes: `EpisodeFinished` from `collect_phase_switch_rollouts`; `PandaArmMotionPlanningSolver` from `mani_skill.examples.motionplanning.panda.motionplanner`; `PlanarPushEnv` (Task 1).
- Produces: `PUSH_PHASES = {"reach":3,"push":4,"align":5,"retract":6}`; `PlanarPushTraceWrapper` (own `snapshot` writing `block_pose`/`target_pose`); `write_push_episode`; `solve_planar_push(env, align_yaw=None)`; `collect_planar_push(...)`; `main` with `--arm {heading_push,free_yaw_push}`.

- [ ] **Step 1: Write the trace wrapper + write function**

```python
# collect_planar_push_rollouts.py (skeleton, then fill solver)
from __future__ import annotations
import argparse, hashlib, json, sqlite3  # noqa: F401
from pathlib import Path
import gymnasium as gym
import h5py
import numpy as np
import sapien
import torch

import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_planar_push_env  # noqa: F401
from collect_phase_switch_rollouts import EpisodeFinished
from mani_skill.agents.robots.panda import Panda
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

Panda.gripper_stiffness = 2.5e3
Panda.gripper_force_limit = 150.0

PUSH_PHASES = {"reach": 3, "push": 4, "align": 5, "retract": 6}
ENV_ID = "PlanarPush-v1"


def npy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class PlanarPushTraceWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.phase = "initial"
        self.trace = None

    def set_phase(self, phase: str):
        self.phase = phase

    def snapshot(self):
        base = self.unwrapped
        info = base.evaluate()
        contact = base.scene.get_pairwise_contact_forces(base.agent.robot, base.block)
        return dict(
            tcp_pose=npy(base.agent.tcp.pose.raw_pose)[0].astype(np.float64),
            block_pose=npy(base.block.pose.raw_pose)[0].astype(np.float64),
            target_pose=npy(base.target.pose.raw_pose)[0].astype(np.float64),
            goal_pose=npy(base.goal_pose.raw_pose)[0].astype(np.float64),
            qpos=npy(base.agent.robot.get_qpos())[0].astype(np.float64),
            qvel=npy(base.agent.robot.get_qvel())[0].astype(np.float64),
            contact_force=npy(contact)[0].astype(np.float64),
            success=bool(npy(info["success"]).reshape(-1)[0]),
            obj_to_goal_dist=float(npy(info["obj_to_goal_dist"]).reshape(-1)[0]),
            heading_err=float(npy(info["heading_err"]).reshape(-1)[0]),
            solver_phase=PUSH_PHASES[self.phase],
        )

    def start_trace(self):
        self.phase = "initial"
        self.trace = dict(states=[self.snapshot()], actions=[], rewards=[],
                          terminated=[], truncated=[])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.trace is not None:
            self.trace["actions"].append(np.asarray(action, dtype=np.float64).copy())
            self.trace["rewards"].append(float(npy(reward).reshape(-1)[0]))
            self.trace["terminated"].append(bool(npy(terminated).reshape(-1)[0]))
            self.trace["truncated"].append(bool(npy(truncated).reshape(-1)[0]))
            self.trace["states"].append(self.snapshot())
        if bool(npy(terminated).reshape(-1)[0]) or bool(npy(truncated).reshape(-1)[0]):
            raise EpisodeFinished(bool(npy(terminated).reshape(-1)[0]),
                                  bool(npy(truncated).reshape(-1)[0]))
        return obs, reward, terminated, truncated, info


def write_push_episode(group, row, trace, solver_error, stop_reason="solver_returned"):
    group.attrs["generator"] = row["generator"]
    group.attrs["solver_error"] = "" if solver_error is None else solver_error
    group.attrs["stop_reason"] = stop_reason
    group.create_dataset("causal_delta", data=np.asarray(row["causal_delta"], dtype=np.float64))
    states = trace["states"]
    for key in ["tcp_pose", "block_pose", "target_pose", "goal_pose", "qpos", "qvel",
                "contact_force", "success", "obj_to_goal_dist", "heading_err", "solver_phase"]:
        group.create_dataset(key, data=np.asarray([s[key] for s in states]))
    group.create_dataset("actions", data=np.asarray(trace["actions"]))
    group.create_dataset("rewards", data=np.asarray(trace["rewards"]))
    group.create_dataset("terminated", data=np.asarray(trace["terminated"], bool))
    group.create_dataset("truncated", data=np.asarray(trace["truncated"], bool))
```

- [ ] **Step 2: Write the solver `solve_planar_push`**

```python
def solve_planar_push(env: PlanarPushTraceWrapper, align_yaw=None):
    """Non-prehensive push: reach (closed gripper to behind the block) ->
    push (3 mm incremental steps toward the target in-plane position) ->
    align (heading arm only: rotate the block about vertical to the target
    heading by nudging one corner) -> retract. Never grasps.

    align_yaw: override the target heading (None = context yaw). For the
    free_yaw_push arm the collector calls with align_yaw=0.0 and the block's
    heading is never corrected, so yaw stays at its start value.
    """
    base = env.unwrapped
    planner = PandaArmMotionPlanningSolver(
        env, debug=False, vis=False, base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=False, print_env_info=False,
        joint_vel_limits=0.5, joint_acc_limits=0.5,
    )
    try:
        def move_pose(target_pose, refine_steps=0):
            result = planner.move_to_pose_with_screw(target_pose, refine_steps=refine_steps)
            if result == -1:
                result = planner.move_to_pose_with_RRTConnect(target_pose, refine_steps=refine_steps)
            if result == -1:
                raise RuntimeError("motion planning failed")
            return result

        def tcp_pose(target_position, quat_wxyz):
            return sapien.Pose(p=target_position, q=quat_wxyz)

        heading = float(base.causal_delta[5]) if align_yaw is None else float(align_yaw)
        goal = base.block_goal_position()  # [x,y,z] at block height
        block_p = base.block.pose.sp.p

        # Push direction in the table plane (normalized block->goal).
        push_dir = goal[:2] - block_p[:2]
        dist = float(np.linalg.norm(push_dir))
        if dist < 1e-6:
            push_dir = np.array([1.0, 0.0])
        else:
            push_dir = push_dir / dist
        push_theta = float(np.arctan2(push_dir[1], push_dir[0]))

        # Closed-gripper approach quaternion: fingers vertical, opening in the
        # plane, so the closed fingertip presents a face normal to push_dir.
        # (Tunable in Phase-0; the screw planner's refine steps absorb the rest.)
        approach_q = sapien.Pose(
            q=(np.cos(push_theta / 2), 0.0, 0.0, np.sin(push_theta / 2))
        ).q

        behind = np.array([block_p[0] - push_dir[0] * (0.5 * 0.05 + 0.06),
                           block_p[1] - push_dir[1] * (0.5 * 0.05 + 0.06),
                           0.05], dtype=np.float64)  # gripper tip z just above table

        env.set_phase("reach")
        move_pose(tcp_pose(behind, approach_q))
        move_pose(tcp_pose([behind[0], behind[1], 0.05], approach_q))

        env.set_phase("push")
        step = 0.003
        n_steps = int(np.ceil(dist / step))
        for i in range(1, n_steps + 1):
            frac = min(1.0, i * step / dist)
            intermediate = np.array([block_p[0] + push_dir[0] * dist * frac,
                                     block_p[1] + push_dir[1] * dist * frac,
                                     0.05], dtype=np.float64)
            move_pose(tcp_pose(intermediate, approach_q), refine_steps=2)

        env.set_phase("align")
        if align_yaw is not None or base.goal_heading:
            # Rotate the block about vertical: nudge one corner. Approximate the
            # heading correction by moving the TCP along a small arc about the
            # block centre. Phase-0 verifies this achieves the yaw tolerance.
            target_heading_q = sapien.Pose(q=(np.cos(heading / 2), 0.0, 0.0, np.sin(heading / 2))).q
            move_pose(tcp_pose([goal[0], goal[1], 0.05], target_heading_q), refine_steps=8)

        env.set_phase("retract")
        move_pose(tcp_pose([goal[0], goal[1], 0.12], approach_q))
    finally:
        planner.close()
```

- [ ] **Step 3: Write the collector `collect_planar_push` + `main`** (mirror `collect_se3_multigen.collect_se3_multigen`)

```python
def collect_planar_push(rows, output_path, seed, orientation, retries_per_condition,
                        task_anchor, goal_heading, context_manifest_path=None,
                        robot_init_qpos_noise=0.0):
    base = gym.make(ENV_ID, num_envs=1, obs_mode="state_dict", control_mode="pd_joint_pos",
                    reward_mode="sparse", sim_backend="physx_cpu", render_mode=None,
                    robot_init_qpos_noise=robot_init_qpos_noise,
                    orientation=orientation, task_anchor=task_anchor, goal_heading=goal_heading)
    env = PlanarPushTraceWrapper(base)
    manifest = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["env_id"] = ENV_ID
        f.attrs["task"] = "planar_push_heading" if goal_heading else "planar_push_free_yaw"
        f.attrs["source_type"] = "motionplanning_physx_planar_push"
        f.attrs["orientation"] = np.asarray(orientation, dtype=np.float64)
        f.attrs["task_anchor"] = np.asarray(task_anchor, dtype=np.float64)
        f.attrs["goal_heading"] = int(bool(goal_heading))
        f.attrs["control_mode"] = "pd_joint_pos"
        f.attrs["sim_backend"] = "physx_cpu"
        f.attrs["seed"] = seed
        if context_manifest_path is not None:
            context_manifest_path = context_manifest_path.resolve()
            f.attrs["context_manifest"] = str(context_manifest_path)
            f.attrs["context_manifest_sha256"] = hashlib.sha256(
                context_manifest_path.read_bytes()).hexdigest()
        f.attrs["environment_source_sha256"] = hashlib.sha256(
            Path(phase_switch_symmetry_planar_push_env.__file__).read_bytes()).hexdigest()
        f.attrs["collector_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        episode_id = 0
        for condition_id, row in enumerate(rows):
            for attempt_id in range(retries_per_condition):
                print(f"[condition {condition_id + 1:03d}/{len(rows):03d} "
                      f"attempt {attempt_id + 1}/{retries_per_condition}] {row['generator']}", flush=True)
                episode_seed = seed + condition_id * 1009 + attempt_id
                np.random.seed(episode_seed)
                torch.manual_seed(episode_seed)
                env.reset(seed=episode_seed, options={"causal_delta": row["causal_delta"]})
                env.start_trace()
                solver_error = None
                stop_reason = "solver_returned"
                try:
                    if goal_heading:
                        solve_planar_push(env)
                    else:
                        solve_planar_push(env, align_yaw=0.0)
                except EpisodeFinished as exc:
                    stop_reason = str(exc)
                except Exception as exc:
                    solver_error = repr(exc)
                    stop_reason = "solver_exception"
                    print("  solver exception:", solver_error, flush=True)
                states = env.trace["states"]
                forces = np.linalg.norm(np.asarray([s["contact_force"] for s in states]), axis=1)
                phases = np.asarray([s["solver_phase"] for s in states])
                success = bool(states[-1]["success"])
                steps = len(env.trace["actions"])
                phase_set = sorted(set(int(x) for x in phases))
                complete = all(code in phase_set for code in [3, 4, 5, 6])
                print(f"  steps={steps} success={success} max_contact={forces.max():.3f} phases={phase_set}", flush=True)
                group = f.create_group(f"episode_{episode_id}")
                group.attrs["condition_id"] = condition_id
                group.attrs["attempt_id"] = attempt_id
                group.attrs["episode_seed"] = episode_seed
                write_push_episode(group, row, env.trace, solver_error, stop_reason=stop_reason)
                manifest.append(dict(episode_id=episode_id, condition_id=condition_id,
                                     attempt_id=attempt_id, **row, steps=steps, success=success,
                                     complete=complete, stop_reason=stop_reason,
                                     max_contact_force_N=float(forces.max()),
                                     phases=phase_set, solver_error=solver_error))
                episode_id += 1
                if success and complete:
                    break
        f.attrs["condition_count"] = len(rows)
        f.attrs["episode_count"] = len(manifest)
        f.attrs["success_count"] = sum(r["success"] for r in manifest)
    env.close()
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as mf:
        json.dump(dict(env_id=ENV_ID, task="planar_push", seed=seed,
                       orientation=np.asarray(orientation, dtype=np.float64).tolist(),
                       goal_heading=bool(goal_heading), retries_per_condition=retries_per_condition,
                       episodes=manifest), mf, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orientation-quat", nargs=4, type=float, default=[1.0, 0.0, 0.0, 0.0])
    parser.add_argument("--task-anchor", nargs=3, type=float, default=[0.10, 0.0, 0.01])
    parser.add_argument("--context-manifest", type=Path,
                        default=Path("phase_switch_symmetry_multiseed/planar_push/planar_push_fixed_contexts.json"))
    parser.add_argument("--experiment", type=Path, default=None)
    parser.add_argument("--arm", choices=["heading_push", "free_yaw_push"], default="heading_push")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--retries-per-condition", type=int, default=5)
    parser.add_argument("--robot-init-qpos-noise", type=float, default=0.01)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    orientation = np.asarray(args.orientation_quat, dtype=np.float64)
    orientation = orientation / np.linalg.norm(orientation)
    task_anchor = np.asarray(args.task_anchor, dtype=np.float64)
    manifest_sha256 = hashlib.sha256(args.context_manifest.read_bytes()).hexdigest()
    if args.experiment is not None:
        with args.experiment.open(encoding="utf-8") as ef:
            experiment = json.load(ef)
        pre = experiment.get("context_manifest_sha256")
        if pre is not None and manifest_sha256 != pre:
            raise RuntimeError(f"context manifest SHA256 {manifest_sha256} != preregistered {pre}")
    with args.context_manifest.open(encoding="utf-8") as mf:
        cm = json.load(mf)
    raw_rows = cm.get("conditions")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("contexts manifest must contain a nonempty conditions list")
    normalized = []
    for expected_id, row in enumerate(raw_rows):
        if int(row.get("condition_id", expected_id)) != expected_id:
            raise ValueError("manifest condition ids must be contiguous from zero")
        generator = str(row["generator"])
        cd = np.asarray(row["causal_delta"], dtype=np.float64)
        if generator not in {"baseline", "du", "dv", "dw", "roll", "pitch", "yaw", "mixed"}:
            raise ValueError(f"unsupported generator: {generator}")
        if cd.shape != (6,) or not np.isfinite(cd).all():
            raise ValueError("every causal_delta must be a finite 6-vector")
        normalized.append(dict(generator=generator, causal_delta=cd.tolist()))
    rows = normalized
    if args.smoke:
        zero = [r for r in rows if np.linalg.norm(np.asarray(r["causal_delta"])) < 1e-12]
        rows = zero[:1] if zero else [rows[0]]
    retries = 1 if args.smoke else args.retries_per_condition
    collect_planar_push(rows, args.output, args.seed, orientation, retries, task_anchor,
                        args.arm == "heading_push",
                        context_manifest_path=args.context_manifest,
                        robot_init_qpos_noise=args.robot_init_qpos_noise)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke collection (single baseline condition)**

Run: `cd /home/rocos/sia/maniskill_spectrum && source ~/miniconda3/etc/profile.d/conda.sh && conda activate maniskill_download && python phase_switch_symmetry/collect_planar_push_rollouts.py --output phase_switch_symmetry_rollouts_planar_push/smoke_heading_seed_20260818.h5 --arm heading_push --smoke`
Expected: one episode collected; `success=True`, `phases=[3,4,5,6]`. If the solver raises, this is the Phase-0 signal — do NOT proceed to full collection; iterate on the solver first.

- [ ] **Step 5: Commit**

```bash
git add phase_switch_symmetry/collect_planar_push_rollouts.py
git commit -m "feat: add planar-push solver, trace wrapper, and collector"
```

---

## Task 5: Phase-0 feasibility probe (mandatory gate)

**Files:**
- Create: `phase_switch_symmetry/probe_planar_push.py`

**Interfaces:**
- Consumes: `solve_planar_push`, `PlanarPushTraceWrapper`, `PlanarPushEnv`.
- Produces: `phase_switch_symmetry_multiseed/planar_push/planar_push_probe.csv` — one row per corner context × arm, columns `arm, context_label, du,dv,dw,roll,pitch,yaw, first_attempt_success, eventual_success, attempts, max_contact_N, final_pos_err, final_heading_err`.

- [ ] **Step 1: Write the probe**

```python
# probe_planar_push.py
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_planar_push_env  # noqa: F401
from collect_planar_push_rollouts import solve_planar_push, PlanarPushTraceWrapper

# Corner contexts: extreme in-plane targets, +/-30 deg heading, +/-15 deg tilt.
CORNERS = [
    ("far_x",      [ 0.012,  0.000, 0.000, 0.0, 0.0, 0.0]),
    ("far_neg_x",  [-0.012,  0.000, 0.000, 0.0, 0.0, 0.0]),
    ("far_y",      [ 0.000,  0.012, 0.000, 0.0, 0.0, 0.0]),
    ("diag",       [ 0.012,  0.012, 0.000, 0.0, 0.0, 0.0]),
    ("yaw+30",     [ 0.000,  0.000, 0.000, 0.0, 0.0, np.deg2rad(30.0)]),
    ("yaw-30",     [ 0.000,  0.000, 0.000, 0.0, 0.0, -np.deg2rad(30.0)]),
    ("roll+15",    [ 0.000,  0.000, 0.012, np.deg2rad(15.0), 0.0, 0.0]),
    ("pitch-15",   [ 0.000,  0.000, -0.012, 0.0, -np.deg2rad(15.0), 0.0]),
]


def main():
    seed = 20260818
    rows = []
    for arm in ("heading_push", "free_yaw_push"):
        for label, cd in CORNERS:
            env = PlanarPushTraceWrapper(gym.make(
                "PlanarPush-v1", num_envs=1, obs_mode="state_dict", control_mode="pd_joint_pos",
                reward_mode="sparse", sim_backend="physx_cpu", render_mode=None,
                orientation=[1.0, 0.0, 0.0, 0.0], task_anchor=[0.10, 0.0, 0.01],
                goal_heading=(arm == "heading_push")))
            first = eventual = False
            attempts = 0
            max_contact = 0.0
            final_pos = final_yaw = np.nan
            for attempt in range(5):
                episode_seed = seed + attempts * 1009
                np.random.seed(episode_seed)
                torch.manual_seed(episode_seed)
                env.reset(seed=episode_seed, options={"causal_delta": cd})
                env.start_trace()
                try:
                    if arm == "heading_push":
                        solve_planar_push(env)
                    else:
                        solve_planar_push(env, align_yaw=0.0)
                except Exception:
                    pass
                states = env.trace["states"]
                forces = np.linalg.norm(np.asarray([s["contact_force"] for s in states]), axis=1)
                max_contact = max(max_contact, float(forces.max()))
                ok = bool(states[-1]["success"])
                final_pos = float(states[-1]["obj_to_goal_dist"])
                final_yaw = float(states[-1]["heading_err"])
                if ok:
                    eventual = True
                if ok and attempts == 0:
                    first = True
                attempts += 1
                if ok:
                    break
            env.close()
            rows.append(dict(arm=arm, context_label=label, **dict(zip(
                ["du", "dv", "dw", "roll", "pitch", "yaw"], cd)),
                first_attempt_success=first, eventual_success=eventual, attempts=attempts,
                max_contact_N=max_contact, final_pos_err=final_pos, final_heading_err=final_yaw))

    out = Path("phase_switch_symmetry_multiseed/planar_push/planar_push_probe.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    header = "arm,context_label,du,dv,dw,roll,pitch,yaw,first_attempt_success,eventual_success,attempts,max_contact_N,final_pos_err,final_heading_err\n"
    with out.open("w") as f:
        f.write(header)
        for r in rows:
            f.write(",".join(str(r[k]) for k in header.strip().split(",")) + "\n")
    print("saved:", out)
    n_first = sum(r["first_attempt_success"] for r in rows)
    n_eventual = sum(r["eventual_success"] for r in rows)
    print(f"first-attempt {n_first}/{len(rows)}, eventual {n_eventual}/{len(rows)}")
```

- [ ] **Step 2: Run the probe**

Run: `cd /home/rocos/sia/maniskill_spectrum && source ~/miniconda3/etc/profile.d/conda.sh && conda activate maniskill_download && python phase_switch_symmetry/probe_planar_push.py`

- [ ] **Step 3: Evaluate the gate**

- **PASS** if `eventual_success` is high (≥ 13/16) AND `align` (yaw ±30°) is controllable for the heading arm (`final_heading_err` below ~0.15 rad on `yaw±30`). → proceed to Task 6.
- **FAIL** (pusher cannot robustly reach, or yaw alignment is unstable) → switch `solve_planar_push` to the **guided push** fallback (grasp the block, slide it on the table with the gripper holding it, then release), record the switch in `planar_push_experiment.json` (`"push_mode": "guided"`), and re-run this probe. The oracle is unchanged; the physical authenticity is weakened and must be stated in the VALIDATION doc. **Do not hard-fight the scripted pusher in the collector.**

- [ ] **Step 4: Commit the probe results**

```bash
git add phase_switch_symmetry/probe_planar_push.py phase_switch_symmetry_multiseed/planar_push/planar_push_probe.csv
git commit -m "feat: Phase-0 planar-push feasibility probe"
```

---

## Task 6: Full collection (2 arms × 3 seeds)

**Files:**
- Create (output): `phase_switch_symmetry_rollouts_planar_push/{heading_push,free_yaw_push}_seed_{seed}.h5` (+ `.json` manifests)

- [ ] **Step 1: Collect heading_push (3 seeds)**

Run (each seed; note `--experiment` enforces the preregistered manifest sha256):

```bash
cd /home/rocos/sia/maniskill_spectrum && source ~/miniconda3/etc/profile.d/conda.sh && conda activate maniskill_download
for s in 20260818 20270818 20280818; do
  python phase_switch_symmetry/collect_planar_push_rollouts.py \
    --output phase_switch_symmetry_rollouts_planar_push/heading_push_seed_$s.h5 \
    --arm heading_push --seed $s \
    --experiment phase_switch_symmetry_multiseed/planar_push/planar_push_experiment.json
done
```

- [ ] **Step 2: Collect free_yaw_push (3 seeds)**

```bash
for s in 20260818 20270818 20280818; do
  python phase_switch_symmetry/collect_planar_push_rollouts.py \
    --output phase_switch_symmetry_rollouts_planar_push/free_yaw_push_seed_$s.h5 \
    --arm free_yaw_push --seed $s \
    --experiment phase_switch_symmetry_multiseed/planar_push/planar_push_experiment.json
done
```

- [ ] **Step 3: Verify collection integrity**

Run: `cd /home/rocos/sia/maniskill_spectrum && python -c "
import h5py, glob
for p in sorted(glob.glob('phase_switch_symmetry_rollouts_planar_push/*_seed_*.h5')):
    with h5py.File(p) as f:
        print(p, 'conditions', f.attrs['condition_count'], 'episodes', f.attrs['episode_count'], 'success', f.attrs['success_count'])
"`
Expected: 6 H5 files; each ~75 conditions, `success_count` ≥ 60 (mixed must all succeed for subset validity — verify no mixed condition is missing).

- [ ] **Step 4: Commit**

```bash
git add phase_switch_symmetry_rollouts_planar_push/
git commit -m "feat: collect planar-push rollouts (2 arms x 3 seeds)"
```

---

## Task 7: Benchmark (oracle + selector metrics + 4-model fits)

**Files:**
- Create: `phase_switch_symmetry/benchmark_planar_push.py`
- Test: `phase_switch_symmetry/test_planar_push_benchmark.py`

**Interfaces:**
- Consumes: `build_models`, `SE3_MODEL_ORDER`, `SEEDS`, `json_ready`, `sha256`, `pose_to_pose6` from `benchmark_se3_transfer`; `PHASE_CODES`, `progress_grid`, `usable` from `benchmark_phase_switch_baselines`; `resample`, `wrap_pi` from `analyze_phase_switch_rollouts`; model classes from `phase_switch_se3_baselines`.
- Produces: `oracle_alpha_planar_push(arm, phase_codes)`, `m_oop_correct(alpha_max_active)`, `m_yaw_correct(...)`, `phase_task_curve_planar_push`, `nominal_frame_planar_push`, `_fit_subset`, `main`.

- [ ] **Step 1: Write the failing tests** (pure functions first)

```python
# test_planar_push_benchmark.py
import numpy as np
from benchmark_planar_push import oracle_alpha_planar_push, m_oop_correct, m_yaw_correct

def test_oracle_heading_selector():
    phase_codes = np.array([0, 0, 1, 1, 2, 2, 3, 3])  # reach, push, align, retract
    o = oracle_alpha_planar_push("heading_push", phase_codes)
    # columns du,dv,dw,roll,pitch,yaw
    assert np.allclose(o[:, 0], [0, 0, 1, 1, 1, 1, 1, 1])   # du: tracked push->retract
    assert np.allclose(o[:, 1], [0, 0, 1, 1, 1, 1, 1, 1])   # dv
    assert np.allclose(o[:, 2], 0.0)                        # dw suppressed everywhere
    assert np.allclose(o[:, 3], 0.0)                        # roll
    assert np.allclose(o[:, 4], 0.0)                        # pitch
    assert np.allclose(o[:, 5], [0, 0, 0, 0, 1, 1, 1, 1])   # yaw: align->retract

def test_oracle_free_yaw_suppresses_yaw():
    phase_codes = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    o = oracle_alpha_planar_push("free_yaw_push", phase_codes)
    assert np.allclose(o[:, 5], 0.0)  # yaw suppressed everywhere

def test_m_oop_correct():
    good = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0])  # du,dv,yaw high; dw,roll,pitch low
    bad_out = np.array([1.0, 1.0, 0.9, 0.0, 0.0, 1.0])  # dw leaked
    bad_in = np.array([0.4, 1.0, 0.0, 0.0, 0.0, 1.0])   # du missing
    assert m_oop_correct(good)
    assert not m_oop_correct(bad_out)
    assert not m_oop_correct(bad_in)

def test_m_yaw_correct():
    assert m_yaw_correct(heading_yaw=0.9, free_yaw=0.1)
    assert not m_yaw_correct(heading_yaw=0.1, free_yaw=0.9)
    assert not m_yaw_correct(heading_yaw=0.9, free_yaw=0.9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/rocos/sia/maniskill_spectrum && source ~/miniconda3/etc/profile.d/conda.sh && conda activate maniskill_download && python -m pytest phase_switch_symmetry/test_planar_push_benchmark.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmark_planar_push'`.

- [ ] **Step 3: Implement the pure functions + curve/nominal helpers**

```python
# benchmark_planar_push.py (core; main mirrors benchmark_se3_multigen.main)
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from transforms3d.quaternions import quat2mat

from analyze_phase_switch_rollouts import resample, wrap_pi
from benchmark_phase_switch_baselines import PHASE_CODES, progress_grid, usable
from benchmark_se3_transfer import (
    SEEDS, SE3_MODEL_ORDER, build_models, json_ready, pose_to_pose6, sha256,
)
from phase_switch_se3_baselines import (
    SE3_DIM, euler_from_matrix, se3_from_pose6, pose6_from_se3, se3_inverse,
)

TASK_ORDER = ("heading_push", "free_yaw_push")
TASK_FILES = {
    "heading_push": "phase_switch_symmetry_rollouts_planar_push/heading_push_seed_{seed}.h5",
    "free_yaw_push": "phase_switch_symmetry_rollouts_planar_push/free_yaw_push_seed_{seed}.h5",
}
GENERATOR_NAMES = ("du", "dv", "dw", "roll", "pitch", "yaw")
DU, DV, DW, ROLL, PITCH, YAW = range(6)
TRACKED = (DU, DV, YAW)
SUPPRESSED = (DW, ROLL, PITCH)


def oracle_alpha_planar_push(arm, phase_codes):
    """6 x T generator-law oracle (re-indexed phases 0=reach,1=push,2=align,3=retract).

    heading_push : du,dv,yaw tracked (du/dv from push on, yaw from align on);
                   dw,roll,pitch suppressed everywhere.
    free_yaw_push: du,dv tracked; yaw suppressed everywhere (block never rotates).
    """
    phase_codes = np.asarray(phase_codes, dtype=int)
    oracle = np.zeros((len(phase_codes), SE3_DIM), dtype=np.float64)
    oracle[:, DU] = (phase_codes >= 1).astype(np.float64)
    oracle[:, DV] = (phase_codes >= 1).astype(np.float64)
    if arm == "heading_push":
        oracle[:, YAW] = (phase_codes >= 2).astype(np.float64)
    return oracle


def m_oop_correct(alpha_max_active):
    """Out-of-plane suppression (headline): tracked generators ever relevant
    (max over active phases > 0.5), suppressed generators never (max < 0.5)."""
    alpha_max_active = np.asarray(alpha_max_active, dtype=np.float64)
    out_of_plane_suppressed = all(alpha_max_active[j] < 0.5 for j in SUPPRESSED)
    in_plane_tracked = all(alpha_max_active[j] > 0.5 for j in TRACKED)
    return bool(out_of_plane_suppressed and in_plane_tracked)


def m_yaw_correct(heading_yaw, free_yaw):
    """Heading-vs-free-yaw yaw discrimination (active-phase max)."""
    return bool(heading_yaw > 0.5 and free_yaw < 0.5)


def phase_task_curve_planar_push(group, phase, bins):
    phases = np.asarray(group["solver_phase"])
    indices = np.flatnonzero(phases == phase)
    block = np.asarray(group["block_pose"])[indices]  # (n, 7)
    rpy = np.stack([euler_from_matrix(quat2mat(q)) for q in block[:, 3:]])
    rpy = np.unwrap(rpy, axis=0)
    pose6 = np.column_stack([block[:, 0], block[:, 1], block[:, 2], rpy])
    return resample(pose6, bins)


def task_curve_planar_push(group, bins):
    return np.concatenate(
        [phase_task_curve_planar_push(group, phase, bins) for phase in PHASE_CODES], axis=0
    )


def nominal_frame_planar_push(data_file, keys, contexts):
    """Recover the nominal target frame C0 = mean_i(target_i o intervention_i^-1)."""
    pose6s = []
    for key, context in zip(keys, contexts):
        target_pose = np.asarray(data_file[key]["target_pose"])[0]
        target_pose6 = pose_to_pose6(target_pose)
        target_T = se3_from_pose6(target_pose6)
        intervention_T = se3_from_pose6(np.asarray(context, dtype=np.float64))
        pose6s.append(pose6_from_se3(target_T @ se3_inverse(intervention_T)))
    pose6s = np.asarray(pose6s)
    nominal = np.empty(SE3_DIM, dtype=np.float64)
    nominal[:3] = pose6s[:, :3].mean(axis=0)
    for generator in range(3):
        angle = pose6s[:, 3 + generator]
        nominal[3 + generator] = float(np.arctan2(np.sin(angle).mean(), np.cos(angle).mean()))
    return nominal
```

- [ ] **Step 4: Implement `load_dataset`, `_fit_subset`, and `main`** (mirror `benchmark_se3_multigen.py` with push-specific curve/nominal/oracle; the `main` scoring emits these outputs)

```python
def load_dataset(path, bins):
    with h5py.File(path, "r") as f:
        usable_groups = {int(f[k].attrs["condition_id"]): k for k in f
                         if k.startswith("episode_") and usable(f[k])}
        mixed = {cid: k for cid, k in usable_groups.items()
                 if str(f[k].attrs["generator"]) == "mixed"}
        mixed_cids = sorted(mixed)
        mixed_keys = [mixed[c] for c in mixed_cids]
        mixed_contexts = np.asarray([np.asarray(f[k]["causal_delta"]) for k in mixed_keys])
        mixed_curves = np.asarray([task_curve_planar_push(f[k], bins) for k in mixed_keys])
    return dict(mixed_cids=mixed_cids, mixed_keys=mixed_keys,
                mixed_contexts=mixed_contexts, mixed_curves=mixed_curves)


_DATASETS = _PDIAG_CONFIG = _PROGRESS = _PHASE_CODES = None
_ACTIVE_MASK = None  # re-indexed phases 1 (push) and 2 (align)


def _fit_subset(job):
    task, seed, subset = job
    ds = _DATASETS[task][seed]
    cid_to_idx = {cid: i for i, cid in enumerate(ds["mixed_cids"])}
    source_cids = subset["source_condition_ids"]
    missing = [c for c in source_cids if c not in cid_to_idx]
    if missing:
        return [dict(task=task, seed=seed, subset_id=subset["subset_id"],
                     protocol=subset["protocol"], sample_size=subset["sample_size"],
                     repeat=subset["repeat"], model=m, fit_success=False,
                     fit_error="missing_source_condition", e_alpha=np.nan,
                     m_oop_correct=False,
                     **{f"alpha_max_active_{g}": np.nan for g in GENERATOR_NAMES})
                for m in SE3_MODEL_ORDER], []
    indices = [cid_to_idx[c] for c in source_cids]
    contexts = ds["mixed_contexts"][indices]
    curves = ds["mixed_curves"][indices]
    keys = [ds["mixed_keys"][i] for i in indices]
    try:
        with h5py.File(Path(TASK_FILES[task].format(seed=seed)), "r") as f:
            nominal_frame_pose = nominal_frame_planar_push(f, keys, contexts)
        models = build_models(nominal_frame_pose, _PDIAG_CONFIG)
    except Exception as exc:
        return [dict(task=task, seed=seed, subset_id=subset["subset_id"],
                     protocol=subset["protocol"], sample_size=subset["sample_size"],
                     repeat=subset["repeat"], model=m, fit_success=False,
                     fit_error=f"nominal_frame:{repr(exc)}", e_alpha=np.nan,
                     m_oop_correct=False,
                     **{f"alpha_max_active_{g}": np.nan for g in GENERATOR_NAMES})
                for m in SE3_MODEL_ORDER], []

    rows, profiles = [], []
    for model in models:
        started = time.perf_counter()
        try:
            model.fit(contexts, curves, _PROGRESS, _PHASE_CODES)
            profile = model.jacobian_diag()  # (T, 6)
            oracle = oracle_alpha_planar_push(task, _PHASE_CODES)
            e_alpha = float(np.mean((profile - oracle) ** 2))
            alpha_max_active = profile[_ACTIVE_MASK].max(axis=0)  # (6,)
            oop = bool(m_oop_correct(alpha_max_active))
            fit_success, fit_error = True, ""
            if model.name == "Pdiag finite (SE(3))":
                profiles.append(dict(task=task, seed=seed, subset_id=subset["subset_id"],
                                     sample_size=subset["sample_size"], profile=profile))
        except Exception as exc:
            fit_success, fit_error = False, repr(exc)
            e_alpha = float("nan")
            alpha_max_active = np.full(SE3_DIM, np.nan)
            oop = False
        rows.append(dict(task=task, seed=seed, subset_id=subset["subset_id"],
                         protocol=subset["protocol"], sample_size=subset["sample_size"],
                         repeat=subset["repeat"], model=model.name,
                         fit_success=fit_success, fit_error=fit_error,
                         fit_seconds=time.perf_counter() - started, e_alpha=e_alpha,
                         m_oop_correct=oop,
                         **{f"alpha_max_active_{GENERATOR_NAMES[j]}": float(alpha_max_active[j])
                            for j in range(SE3_DIM)}))
    return rows, profiles
```

The `main` mirrors `benchmark_se3_multigen.main`: load datasets for both arms, run `_fit_subset` over all (arm, seed, subset) jobs, then aggregate:

```python
    # Headline M-oop accuracy (heading_push arm)
    heading = fits[(fits.task == "heading_push") & fits.fit_success]
    m_oop = heading.groupby(["model", "sample_size"], sort=False)["m_oop_correct"] \
                  .agg(["mean", "sum", "count"]).rename(columns={"mean": "accuracy"}).reset_index()

    # M-yaw discrimination: yaw(max active) heading vs free_yaw, per (model,N,seed,subset)
    piv = fits[fits.fit_success].pivot_table(index=["model", "sample_size", "seed", "subset_id"],
                                             columns="task", values="alpha_max_active_yaw").reset_index()
    yaw_rows = []
    if "heading_push" in piv.columns and "free_yaw_push" in piv.columns:
        for _, row in piv.iterrows():
            if np.isfinite(row["heading_push"]) and np.isfinite(row["free_yaw_push"]):
                yaw_rows.append(dict(model=row["model"], sample_size=row["sample_size"],
                                     seed=row["seed"], subset_id=row["subset_id"],
                                     m_yaw_correct=m_yaw_correct(row["heading_push"], row["free_yaw_push"])))
    m_yaw = pd.DataFrame(yaw_rows).groupby(["model", "sample_size"], sort=False)["m_yaw_correct"] \
            .agg(["mean", "sum", "count"]).rename(columns={"mean": "accuracy"}).reset_index()

    # e_alpha per arm
    e_alpha = fits[fits.fit_success].groupby(["task", "model", "sample_size"], sort=False)["e_alpha"] \
              .agg(["mean", "std", "count"]).reset_index()

    # per-generator max-active relevance means (heading_push)
    alpha_cols = [f"alpha_max_active_{g}" for g in GENERATOR_NAMES]
    alpha_means = heading.groupby(["model", "sample_size"], sort=False)[alpha_cols].mean().reset_index()
```

Write CSVs `planar_push_fits.csv`, `planar_push_M_oop.csv`, `planar_push_M_yaw.csv`, `planar_push_e_alpha.csv`, `planar_push_alpha_max_active.csv`, `planar_push_profiles.npz`, and a `planar_push_summary.json` (via `json_ready`, with `oracle`, `source_sha256`, `fit_count`, `fit_failure_count`, `bins_per_phase`, and an `active_phase_note` stating the mask = `push`+`align`). Also read the frozen `se3_transfer/se3_transfer_fits.csv` and emit the fixed cross-task contrast: `alpha_{dw,roll,pitch}` in `heading_push` (≈0) vs frozen `keyed` (≈1).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/rocos/sia/maniskill_spectrum && source ~/miniconda3/etc/profile.d/conda.sh && conda activate maniskill_download && python -m pytest phase_switch_symmetry/test_planar_push_benchmark.py -q`
Expected: PASS (4 pure-function tests).

- [ ] **Step 6: Run the full benchmark**

Run: `cd /home/rocos/sia/maniskill_spectrum && source ~/miniconda3/etc/profile.d/conda.sh && conda activate maniskill_download && python phase_switch_symmetry/benchmark_planar_push.py`
Expected: no `fit_failure_count`; `M-oop` accuracy high for `Pdiag finite (SE(3))` (target ≈1.0) and near 0 for the Frame-weighted negative control; `M-yaw` high for Pdiag finite; suppressed generators' `alpha_max_active_{dw,roll,pitch}` ≈ 0.

- [ ] **Step 7: Commit**

```bash
git add phase_switch_symmetry/benchmark_planar_push.py phase_switch_symmetry/test_planar_push_benchmark.py phase_switch_symmetry_multiseed/planar_push/
git commit -m "feat: planar-push benchmark (oracle + selector metrics + 4-model fits)"
```

---

## Task 8: Figure + VALIDATION doc

**Files:**
- Create: `phase_switch_symmetry/make_planar_push_figure.py`
- Create: `phase_switch_symmetry_multiseed/planar_push/VALIDATION_planar_push.md`

- [ ] **Step 1: Write the figure script** (mirror `make_se3_multigen_figure.py` style: 6 panels for the recovered `Pdiag finite` diagonal α_j(s), one per generator, both arms overlaid; shade the active window push+align; annotate dw/roll/pitch ≈ 0 vs du/dv/yaw ≈ 1)

```python
# make_planar_push_figure.py
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from benchmark_planar_push import GENERATOR_NAMES

root = Path("phase_switch_symmetry_multiseed/planar_push")
profiles = np.load(root / "planar_push_profiles.npz")
progress = profiles["progress"]
phase_codes = profiles["phase_codes"]
arm = profiles["task"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 3, figsize=(9.0, 4.6), constrained_layout=True)
for j, (ax, name) in enumerate(zip(axes.flat, GENERATOR_NAMES)):
    for a, color in [("heading_push", "black"), ("free_yaw_push", "tab:red")]:
        sel = [p for p, t in zip(profiles["profile"], arm) if t == a]
        if not sel:
            continue
        mean = np.mean(sel, axis=0)[:, j]
        std = np.std(sel, axis=0)[:, j]
        ax.plot(progress, mean, color=color, lw=1.6, label=a)
        ax.fill_between(progress, mean - std, mean + std, color=color, alpha=0.18)
    for b in [0.25, 0.5, 0.75]:
        ax.axvline(b, color="0.82", lw=0.6)
    ax.axvspan(0.25, 0.75, color="0.90", alpha=0.35)  # active window (push+align)
    ax.set_title(f"$\\alpha_{{{name}}}(s)$", loc="left", fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(-0.2, 1.3)
    ax.set_xticks([0.125, 0.375, 0.625, 0.875], ["reach", "push", "align", "retract"], fontsize=6)
axes[0, 0].legend(frameon=False, fontsize=6)
fig.savefig(root / "planar_push_figure.png", dpi=300, bbox_inches="tight")
fig.savefig(root / "planar_push_figure.pdf", bbox_inches="tight")
print("saved:", root / "planar_push_figure.png")
```

- [ ] **Step 2: Run the figure script**

Run: `cd /home/rocos/sia/maniskill_spectrum && source ~/miniconda3/etc/profile.d/conda.sh && conda activate maniskill_download && python phase_switch_symmetry/make_planar_push_figure.py`

- [ ] **Step 3: Write `VALIDATION_planar_push.md`** (mirror the frozen `VALIDATION_se3_multigen.md` structure: purpose, frozen design block with source hashes, oracle table, results tables, interpretation, and the cross-task contrast with the frozen peg-in-hole `dw/roll/pitch ≈ 1`). Include the `active_phase_note`, the Phase-0 gate outcome, and — if the guided-push fallback was used — an explicit statement of the weakened physical authenticity.

- [ ] **Step 4: Commit**

```bash
git add phase_switch_symmetry/make_planar_push_figure.py phase_switch_symmetry_multiseed/planar_push/VALIDATION_planar_push.md phase_switch_symmetry_multiseed/planar_push/planar_push_figure.png phase_switch_symmetry_multiseed/planar_push/planar_push_figure.pdf
git commit -m "feat: planar-push figure and frozen validation doc"
```

---

## Self-Review

**Spec coverage:**
- §4.1 env (table + non-square rectangular block + ghost target + `goal_heading`) → Task 1. ✓
- §4.2 solver (`reach→push→align→retract`, active = push+align) → Task 4. ✓
- §4.3 collector (2 arms × 3 seeds × 75) → Tasks 4, 6. ✓
- §4.4 manifest + subsets (N∈{8,15,30}, rank-6, cond<10) → Tasks 2, 3. ✓
- §4.5 benchmark (block_pose/target_pose, target nominal frame, 4-model reuse) → Task 7. ✓
- §4.6 metrics (M1 e_alpha, M-oop, M-yaw, cross-task contrast) → Task 7. ✓
- §5 Phase-0 gate (probe, guided-push fallback) → Task 5. ✓
- §6 freezing discipline (new files only, preregistered experiment, hashes) → Global Constraints + Tasks 2, 3, 6, 8. ✓
- §3 oracle: `heading_push [1,1,0,0,0,1]`, `free_yaw [1,1,0,0,0,0]`, active-phase mask → `oracle_alpha_planar_push` + `m_oop_correct`/`m_yaw_correct`. ✓

**Placeholder scan:** none — all steps carry code or exact commands; the single `<FILL>` (experiment sha256) is resolved by an explicit `sha256sum` command in the same step.

**Type consistency:** `PUSH_PHASES` (raw 3,4,5,6) matches frozen `complete()`/`PHASE_CODES`; `nominal_frame_planar_push` returns a 6-vector consumed by `build_models`; `oracle_alpha_planar_push`/`m_oop_correct`/`m_yaw_correct` signatures match the tests; `alpha_max_active_{g}` columns match the aggregation.

**Note to the user (review point):** the spec's M-oop wording says "active-phase means > 0.5" for the tracked generators, but the honest phasewise oracle has `du/dv` onset at `push` and `yaw` onset at `align`, so a *mean* over push+align would be ~0.5 for yaw (borderline). The plan therefore scores the tracked/suppressed classification with a **max over active phases** (`m_oop_correct`), which is robust to the one-phase support offset and directly captures "which generators ever respond". The exact phase support is confirmed by the Phase-0 probe before the oracle is frozen.
