from __future__ import annotations

"""Render the tilt-amplitude SE(3) rollouts to videos (parallel to the real ones).

Replays the SAME solver used for collection (``solve_se3`` with ``align_yaw=0``)
on the ``CircularPhaseSwitchSE3-v1`` env with rendering enabled, capturing the
human render camera every step. Produces one mp4 per pitch amplitude, plus the
final frame as a PNG per amplitude for figures.

This is a visualization artifact only: physics, seeds, and solver are identical
to the collected ``tilt_sweep`` trajectories, but these frames are NOT part of
the frozen identification protocol (collection itself runs with render_mode=None).
"""

import argparse
import hashlib
import json
import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_env  # noqa: F401
from collect_phase_switch_rollouts import EpisodeFinished, PhaseSwitchTraceWrapper
from collect_phase_switch_rotated import solve_se3
from mani_skill.agents.robots.panda import Panda

Panda.gripper_stiffness = 2.5e3
Panda.gripper_force_limit = 150.0

TASK_ANCHOR = [-0.15, 0.0, 0.08]
PICKUP = [-0.25, -0.18, 0.28]
ORIENTATION = [1.0, 0.0, 0.0, 0.0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_mp4(path: Path, frames, fps: float):
    """Encode RGB frames to H.264 mp4 via the system ffmpeg (no user-site deps)."""
    height, width = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an",
        "-vcodec", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", str(path),
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


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


def render_amplitude(amp_deg, seed, max_frames):
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
    causal_delta = [0.0, 0.0, 0.0, 0.0, float(np.deg2rad(amp_deg)), 0.0]
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(
        seed=seed,
        options={"causal_delta": causal_delta, "pickup_position": PICKUP},
    )
    env.start_trace()
    frames.append(env._render())  # initial frame (post-reset)
    try:
        solve_se3(env, align_yaw=0.0)
    except EpisodeFinished:
        pass
    except Exception as exc:  # noqa: BLE001 - report, don't crash the whole batch
        print(f"  [amp {amp_deg:+}deg] solver exception: {exc!r}", flush=True)
    env.close()
    if max_frames and len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in indices]
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amplitudes", type=float, nargs="+",
                        default=[0.0, 10.0, -10.0, 15.0, -15.0],
                        help="Pitch amplitudes (deg) to render.")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("real_relation_consistency/tilt_sweep/videos"))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Cap frames per video (0 = keep all).")
    args = parser.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "env_id": "CircularPhaseSwitchSE3-v1",
        "task_anchor": TASK_ANCHOR,
        "pickup_position": PICKUP,
        "orientation": ORIENTATION,
        "seed": args.seed,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "solver": "solve_se3(align_yaw=0.0)",
        "note": "Visualization only; not part of the frozen identification protocol.",
        "source_sha256": {
            "render_tilt_videos.py": sha256(Path(__file__)),
            "phase_switch_symmetry_env.py": sha256(
                Path(__file__).with_name("phase_switch_symmetry_env.py")
            ),
            "collect_phase_switch_rotated.py": sha256(
                Path(__file__).with_name("collect_phase_switch_rotated.py")
            ),
        },
        "videos": [],
    }

    for amp in args.amplitudes:
        label = f"{amp:+g}deg".replace("+", "plus_").replace("-", "minus_")
        label = label.replace(".", "p")
        print(f"rendering pitch {amp:+g} deg ...", flush=True)
        frames = render_amplitude(amp, args.seed, args.max_frames)
        if not frames:
            print(f"  no frames captured for {amp:+g} deg", flush=True)
            continue
        mp4 = out / f"pitch_{label}.mp4"
        write_mp4(mp4, frames, args.fps)
        final_png = out / f"pitch_{label}_final.png"
        Image.fromarray(frames[-1]).save(final_png)
        manifest["videos"].append(
            dict(
                amplitude_deg=amp,
                frames=len(frames),
                mp4=str(mp4.name),
                final_frame_png=str(final_png.name),
            )
        )
        print(f"  {mp4.name}: {len(frames)} frames -> {mp4}", flush=True)

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("done:", out)


if __name__ == "__main__":
    main()
