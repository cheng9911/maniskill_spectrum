from __future__ import annotations

"""Segment 3 (grid variant) -- render 16 single-generator SE(3) interventions.

Replays the SAME solver (``solve_se3(align_yaw=0)``) as the tilt sweep on the
``CircularPhaseSwitchSE3-v1`` env, but with one task-local ``causal_delta`` per
SE(3) generator, so each clip shows the peg adapting to a different position
(du/dv/dw translation) or orientation (roll/pitch tilt) intervention.  These 16
clips are then tiled into a 4x4 grid by make_paper_video.py to argue that the
frozen relation law handles *any* SE(3) intervention in the same scene.

This is a visualization artifact only: physics/seeds/solver are identical to the
collected ``tilt_sweep`` / ``se3`` rollouts, but these frames are NOT part of the
frozen identification protocol (collection runs with render_mode=None).
"""

import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "phase_switch_symmetry"))
sys.path.insert(0, str(HERE))

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_env  # noqa: F401
from collect_phase_switch_rollouts import EpisodeFinished, PhaseSwitchTraceWrapper
from collect_phase_switch_rotated import solve_se3
from mani_skill.agents.robots.panda import Panda
from video_common import write_mp4

Panda.gripper_stiffness = 2.5e3
Panda.gripper_force_limit = 150.0

TASK_ANCHOR = [-0.15, 0.0, 0.08]
PICKUP = [-0.25, -0.18, 0.28]
ORIENTATION = [1.0, 0.0, 0.0, 0.0]
FPS = 20.0
MAX_FRAMES = 120

DEG = np.deg2rad

# (file_stem, on-screen label, causal_delta [du, dv, dw, roll, pitch, yaw])
# Row-major order for a 4x4 grid:
#   baseline, du-, du+, dv- | dv+, dw-, dw+, roll- | roll+, pitch-, pitch+, yaw-30
#   | yaw-15, yaw+15, yaw+30, pitch+10 (the held-out amplitude matched to the
#   real machine's 10.3 deg).
CASES = [
    ("baseline", "baseline", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("du_m15mm", "du −15 mm", [-0.015, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("du_p15mm", "du +15 mm", [0.015, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("dv_m15mm", "dv −15 mm", [0.0, -0.015, 0.0, 0.0, 0.0, 0.0]),
    ("dv_p15mm", "dv +15 mm", [0.0, 0.015, 0.0, 0.0, 0.0, 0.0]),
    ("dw_m15mm", "dw −15 mm", [0.0, 0.0, -0.015, 0.0, 0.0, 0.0]),
    ("dw_p15mm", "dw +15 mm", [0.0, 0.0, 0.015, 0.0, 0.0, 0.0]),
    ("roll_m15deg", "roll −15°", [0.0, 0.0, 0.0, -DEG(15.0), 0.0, 0.0]),
    ("roll_p15deg", "roll +15°", [0.0, 0.0, 0.0, DEG(15.0), 0.0, 0.0]),
    ("pitch_m15deg", "pitch −15°", [0.0, 0.0, 0.0, 0.0, -DEG(15.0), 0.0]),
    ("pitch_p15deg", "pitch +15°", [0.0, 0.0, 0.0, 0.0, DEG(15.0), 0.0]),
    ("yaw_m30deg", "yaw −30°", [0.0, 0.0, 0.0, 0.0, 0.0, -DEG(30.0)]),
    ("yaw_m15deg", "yaw −15°", [0.0, 0.0, 0.0, 0.0, 0.0, -DEG(15.0)]),
    ("yaw_p15deg", "yaw +15°", [0.0, 0.0, 0.0, 0.0, 0.0, DEG(15.0)]),
    ("yaw_p30deg", "yaw +30°", [0.0, 0.0, 0.0, 0.0, 0.0, DEG(30.0)]),
    ("pitch_p10deg_holdout", "pitch +10° (held out)", [0.0, 0.0, 0.0, 0.0, DEG(10.0), 0.0]),
]

# 16 coupled (multi-DOF) interventions: two or more generators perturbed at the
# same time, so the grid also argues that the law handles *simultaneous*
# position+orientation changes, not just one axis at a time.  Magnitudes match
# the single-generator scale (translation 15 mm, roll/pitch 15 deg).  Yaw is
# deliberately omitted: the circular hole is axisymmetric, so yaw is invisible
# (a coupled yaw cell would render identically to its yaw-free counterpart).
COUPLED_CASES = [
    ("du_dv", "du + dv", [0.015, 0.015, 0.0, 0.0, 0.0, 0.0]),
    ("du_dw", "du + dw", [0.015, 0.0, 0.015, 0.0, 0.0, 0.0]),
    ("dv_dw", "dv + dw", [0.0, 0.015, 0.015, 0.0, 0.0, 0.0]),
    ("du_roll", "du + roll", [0.015, 0.0, 0.0, DEG(15.0), 0.0, 0.0]),
    ("du_pitch", "du + pitch", [0.015, 0.0, 0.0, 0.0, DEG(15.0), 0.0]),
    ("dv_roll", "dv + roll", [0.0, 0.015, 0.0, DEG(15.0), 0.0, 0.0]),
    ("dv_pitch", "dv + pitch", [0.0, 0.015, 0.0, 0.0, DEG(15.0), 0.0]),
    ("dw_roll", "dw + roll", [0.0, 0.0, 0.015, DEG(15.0), 0.0, 0.0]),
    ("dw_pitch", "dw + pitch", [0.0, 0.0, 0.015, 0.0, DEG(15.0), 0.0]),
    ("roll_pitch", "roll + pitch", [0.0, 0.0, 0.0, DEG(15.0), DEG(15.0), 0.0]),
    ("du_dv_dw", "du + dv + dw", [0.015, 0.015, 0.015, 0.0, 0.0, 0.0]),
    ("du_dv_roll", "du + dv + roll", [0.015, 0.015, 0.0, DEG(15.0), 0.0, 0.0]),
    ("du_dv_pitch", "du + dv + pitch", [0.015, 0.015, 0.0, 0.0, DEG(15.0), 0.0]),
    ("du_roll_pitch", "du + roll + pitch", [0.015, 0.0, 0.0, DEG(15.0), DEG(15.0), 0.0]),
    ("dv_roll_pitch", "dv + roll + pitch", [0.0, 0.015, 0.0, DEG(15.0), DEG(15.0), 0.0]),
    ("all5", "du+dv+dw+roll+pitch", [0.015, 0.015, 0.015, DEG(15.0), DEG(15.0), 0.0]),
]


class RenderTraceWrapper(PhaseSwitchTraceWrapper):
    """Same trace wrapper, plus a rendered frame captured after every step."""

    def __init__(self, env, frames):
        super().__init__(env)
        self.frames = frames

    def step(self, action):
        try:
            return super().step(action)
        finally:
            self.frames.append(self._render())

    def _render(self):
        rgb = self.env.render()
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        rgb = np.asarray(rgb)
        if rgb.ndim == 4:
            rgb = rgb[0]
        return np.ascontiguousarray(rgb[..., :3])


def render_intervention(causal_delta, seed, max_frames):
    base = gym.make(
        "CircularPhaseSwitchSE3-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_mode="rgb_array",
        robot_init_qpos_noise=0.01,
        orientation=np.asarray(ORIENTATION, dtype=np.float64),
        task_anchor=TASK_ANCHOR,
    )
    frames = []
    env = RenderTraceWrapper(base, frames)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed, options={"causal_delta": np.asarray(causal_delta, dtype=np.float64),
                                  "pickup_position": PICKUP})
    env.start_trace()
    frames.append(env._render())  # initial frame (post-reset)
    try:
        solve_se3(env, align_yaw=0.0)
    except EpisodeFinished:
        pass
    except Exception as exc:  # noqa: BLE001 - report, don't crash the batch
        print(f"  solver exception: {exc!r}", flush=True)
    env.close()
    if max_frames and len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]
    return frames


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=HERE / "segments")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--only", type=str, default=None,
                        help="comma-separated stems to render (smoke test); default all")
    args = parser.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    groups = [("seg3_se3", CASES), ("seg3_coupled", COUPLED_CASES)]
    if args.only:
        want = set(args.only.split(","))
        groups = [(p, [(s, l, d) for (s, l, d) in cases if s in want])
                  for p, cases in groups]

    for prefix, cases in groups:
        for stem, label, causal_delta in cases:
            dst = out / f"{prefix}_{stem}.mp4"
            if dst.exists():
                print(f"skip {dst.name}  (already rendered)")
                continue
            print(f"rendering {label} ...", flush=True)
            frames = render_intervention(causal_delta, args.seed, args.max_frames)
            if not frames:
                print(f"  no frames captured for {label}", flush=True)
                continue
            write_mp4(dst, frames, fps=FPS)
            print(f"  {dst.name}: {len(frames)} frames", flush=True)

    print("done:", out)


if __name__ == "__main__":
    main()
