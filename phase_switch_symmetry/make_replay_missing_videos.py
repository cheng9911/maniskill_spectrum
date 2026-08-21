from __future__ import annotations

"""Render the missing replay videos as side-by-side panels.

Three experiment groups that have no video yet:
  1. Five-seed replication   (5 seeds x condition 4 = isolated yaw +30 deg)
  2. Rotated axis Q1 vs Q2   (vertical vs horizontal, condition 4)
  3. SE(3) keyed vs circular (condition 14 = pure yaw +30 deg, 6-generator)

Each panel replays the stored qpos + peg/socket poses into a fresh ManiSkill env
(one env per panel), matching make_videos.py. Panels are hstacked into one frame
with a label + phase band per panel. Each arm is resampled to a common frame
count so the arms stay aligned.

Run from the repo root::

    python phase_switch_symmetry/make_replay_missing_videos.py [--smoke]
"""

import argparse
import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import imageio.v2 as imageio
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_env  # noqa: F401
from mani_skill.utils.structs.pose import Pose

from make_videos import load_episode, make_env, PHASE_NAMES
from make_symmetry_transfer_videos import first_success_condition

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "phase_switch_symmetry_videos"


def overlay_panel(frame, phase, label):
    from PIL import Image, ImageDraw

    name = PHASE_NAMES.get(int(phase), str(int(phase)))
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 56], fill=(0, 0, 0))
    draw.text((12, 8), label, fill=(255, 214, 92))
    draw.text((12, 32), f"phase: {name}", fill=(255, 255, 255))
    return np.asarray(img)


def _frame_indices(n, max_frames):
    if max_frames >= n:
        return np.arange(n)
    return np.linspace(0, n - 1, max_frames).astype(int)


def _render(env, data, i):
    base = env.unwrapped
    device = base.device
    base.agent.robot.set_qpos(
        torch.tensor(data["qpos"][i], dtype=torch.float32, device=device).reshape(1, -1)
    )
    base.peg.set_pose(
        Pose.create_from_pq(
            torch.tensor(data["peg_pose"][i][:3], dtype=torch.float32, device=device).reshape(1, 3),
            torch.tensor(data["peg_pose"][i][3:], dtype=torch.float32, device=device).reshape(1, 4),
        )
    )
    base.socket.set_pose(
        Pose.create_from_pq(
            torch.tensor(data["socket_pose"][i][:3], dtype=torch.float32, device=device).reshape(1, 3),
            torch.tensor(data["socket_pose"][i][3:], dtype=torch.float32, device=device).reshape(1, 4),
        )
    )
    fr = env.render()
    if hasattr(fr, "cpu"):
        fr = fr.cpu().numpy()
    return np.squeeze(np.asarray(fr))


def render_multi_arm(arms, out_path, max_frames=200, fps=30):
    """arms: list of dict(h5=Path, episode=int, label=str)."""
    loaded = []
    for a in arms:
        data, ea, ra = load_episode(Path(a["h5"]), a["episode"])
        orientation = ra.get("orientation", None)
        task_anchor = ra.get("task_anchor", None)
        env_id = ra.get("env_id", "KeyedCircularPhaseSwitch-v1")
        if isinstance(env_id, bytes):
            env_id = env_id.decode()
        if orientation is not None:
            orientation = np.asarray(orientation, dtype=np.float64)
        if task_anchor is not None:
            task_anchor = np.asarray(task_anchor, dtype=np.float64)
        env = make_env(orientation, task_anchor, env_id=env_id)
        env.reset(
            seed=int(ea.get("episode_seed", 20260818)),
            options={"causal_delta": data["causal_delta"].tolist()},
        )
        loaded.append((a["label"], env, data))

    indices = [_frame_indices(len(d["qpos"]), max_frames) for _, _, d in loaded]
    n_frames = len(indices[0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    try:
        for f in range(n_frames):
            panels = []
            for (label, env, data), idxs in zip(loaded, indices):
                frame = _render(env, data, idxs[f])
                panels.append(overlay_panel(frame, data["solver_phase"][idxs[f]], label))
            writer.append_data(np.concatenate(panels, axis=1))
    finally:
        writer.close()
        for _, env, _ in loaded:
            env.close()
    return dict(frames=n_frames, arms=[a["label"] for a in arms])


def build_jobs():
    jobs = {}

    seeds = [20260818, 20270818, 20280818, 20290818, 20300818]
    jobs["24_fiveseed_replication"] = [
        dict(
            h5=ROOT / f"phase_switch_symmetry_multiseed/rollouts/seed_{s}.h5",
            episode=first_success_condition(ROOT / f"phase_switch_symmetry_multiseed/rollouts/seed_{s}.h5", 4),
            label=f"seed {s}",
        )
        for s in seeds
    ]

    jobs["25_rotated_Q1_vs_Q2"] = [
        dict(
            h5=ROOT / "phase_switch_symmetry_rollouts_rotated/rotated_Q1_seed_20260818.h5",
            episode=first_success_condition(ROOT / "phase_switch_symmetry_rollouts_rotated/rotated_Q1_seed_20260818.h5", 4),
            label="Q1 vertical (axis z)",
        ),
        dict(
            h5=ROOT / "phase_switch_symmetry_rollouts_rotated/rotated_Q2_seed_20260818.h5",
            episode=first_success_condition(ROOT / "phase_switch_symmetry_rollouts_rotated/rotated_Q2_seed_20260818.h5", 4),
            label="Q2 horizontal (axis x)",
        ),
    ]

    jobs["26_se3_keyed_vs_circular"] = [
        dict(
            h5=ROOT / "phase_switch_symmetry_rollouts_se3/keyed_seed_20260818.h5",
            episode=first_success_condition(ROOT / "phase_switch_symmetry_rollouts_se3/keyed_seed_20260818.h5", 14),
            label="SE(3) keyed (yaw relevant)",
        ),
        dict(
            h5=ROOT / "phase_switch_symmetry_rollouts_se3/circular_honest_seed_20260818.h5",
            episode=first_success_condition(ROOT / "phase_switch_symmetry_rollouts_se3/circular_honest_seed_20260818.h5", 14),
            label="SE(3) circular honest (gauge)",
        ),
    ]

    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs()
    max_frames = 8 if args.smoke else args.max_frames

    report = []
    for label, arms in jobs.items():
        out = args.out_dir / f"{label}.mp4"
        print(f"[render] {label}: {len(arms)} arms", flush=True)
        info = render_multi_arm(arms, out, max_frames=max_frames, fps=args.fps)
        info.update(label=label, out=str(out))
        report.append(info)
        print(f"   -> {out}  frames={info['frames']}  arms={info['arms']}", flush=True)

    import json

    (args.out_dir / "missing_replay_video_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"wrote manifest: {args.out_dir / 'missing_replay_video_manifest.json'}")


if __name__ == "__main__":
    main()
