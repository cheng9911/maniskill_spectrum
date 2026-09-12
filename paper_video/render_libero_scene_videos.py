from __future__ import annotations

"""Segment 5 -- "Cross-relation breadth (LIBERO)": render state-level scene clips.

Each clip replays the nominal (condition 0 / baseline) controlled rollout of one
LIBERO relation scene, so the object performs its characteristic relation motion
(drawer slides, knob rotates, bowl is placed, ...).  The rollouts are state-level
counterfactuals -- the object pose is set directly and no robot arm is driven --
so the resulting footage is honestly labeled relation-level evidence, not demos.

Reads the frozen rollouts collected by
phase_switch_symmetry/collect_libero_relation_suite.py and re-runs the exact
alpha schedule at 512x512, then encodes one mp4 per scene.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "phase_switch_symmetry"))

from collect_libero_relation_suite import (  # noqa: E402
    make_libero_env,
    object_pose_at,
    selected_delta,
    set_articulated_joint,
    set_free_body,
)
from libero_relation_suite_specs import PHASE_CODES, specs_by_key  # noqa: E402

from video_common import FPS, write_mp4  # noqa: E402

ROLLOUT_ROOT = ROOT / "phase_switch_symmetry_rollouts_libero_relation_suite"
SEED = 20260818
CAMERA_SIZE = 512
STEPS_PER_PHASE = 12

# (task_key, on-screen relation label) -- one representative per relation family
# and the three geometry-transfer pairs (drawer->plate, knob->microwave,
# bowl->cream_cheese), so Seg 5 doubles as the "same law, new scene" montage.
SCENES = [
    ("drawer_middle_open", "sliding  (du)"),
    ("plate_front_push", "planar push  (du)"),
    ("stove_knob_turn", "revolute knob  (yaw)"),
    ("microwave_door_revolute", "revolute door  (yaw)"),
    ("bowl_on_stove", "placement  (xyz)"),
    ("cream_cheese_in_bowl", "container-in  (xyz)"),
]


def alpha_schedule() -> list[float]:
    alphas: list[float] = []
    for phase_code in PHASE_CODES:
        if phase_code == 3:
            alphas.extend([0.0] * STEPS_PER_PHASE)
        elif phase_code == 4:
            alphas.extend(np.linspace(0.0, 1.0, STEPS_PER_PHASE).tolist())
        else:
            alphas.extend([1.0] * STEPS_PER_PHASE)
    return alphas


def render_scene(task_key: str, out_dir: Path) -> Path:
    import h5py

    spec = specs_by_key()[task_key]
    h5 = ROLLOUT_ROOT / task_key / f"{task_key}_seed_{SEED}.h5"
    with h5py.File(h5, "r") as f:
        causal_delta = np.asarray(f["episode_0000"]["causal_delta"], dtype=np.float64)
        n_steps = f["episode_0000"]["object_pose"].shape[0]
    expected = len(PHASE_CODES) * STEPS_PER_PHASE
    assert n_steps == expected, (task_key, n_steps, expected)

    env, _bddl, _name = make_libero_env(spec, CAMERA_SIZE)
    env.seed(SEED)
    env.reset()
    sim = env.sim
    model = sim.model
    data = sim.data

    initial_free_qpos = None
    if spec.free_joint_name is not None:
        joint_id = model.joint_name2id(spec.free_joint_name)
        qpos_addr = int(model.jnt_qposadr[joint_id])
        initial_free_qpos = np.asarray(data.qpos[qpos_addr : qpos_addr + 7], dtype=np.float64).copy()

    nominal = np.asarray(spec.nominal_pose6, dtype=np.float64)
    frames = []
    for alpha in alpha_schedule():
        pose = object_pose_at(spec, causal_delta, float(alpha))
        pose6 = alpha * (nominal + selected_delta(spec, causal_delta))
        if spec.free_joint_name is not None:
            set_free_body(sim, spec, pose, initial_free_qpos)
        elif spec.joint_name is not None:
            set_articulated_joint(sim, spec, pose6)
        img = np.asarray(sim.render(camera_name="agentview", width=CAMERA_SIZE, height=CAMERA_SIZE))
        frames.append(img[::-1].copy())  # robosuite convention: flip vertical
    env.close()

    out = out_dir / f"seg5_libero_{task_key}.mp4"
    write_mp4(out, frames, fps=FPS)
    print(f"wrote {out}  frames={len(frames)}")
    return out


def main() -> None:
    out_dir = HERE / "segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    for task_key, _label in SCENES:
        render_scene(task_key, out_dir)


if __name__ == "__main__":
    main()
