from __future__ import annotations

"""Render the three-arm symmetry-transfer rollouts as physical-replay videos.

Three arms (all at the same held-out intervention, socket yaw +30 deg):
  1. Keyed asymmetric  (KeyedCircularPhaseSwitch-v1): the key must align to the
     rotated keyed gate -> yaw is *relevant*, then irrelevant after clearance.
  2. Circular honest   (CircularPhaseSwitch-v1, align_yaw=0): circular peg in a
     circular bore -> d_axial is a pure gauge DOF, no yaw sweep in the demo.
  3. Circular placebo  (CircularPhaseSwitch-v1, align_yaw=0 + fixed 15deg sweep):
     same symmetric geometry, but the demonstrator adds a fixed yaw sweep that is
     independent of d_axial -> controls for trajectory-statistics memorization.

Reuses make_videos.render_episode_video, which replays the stored qpos + poses
into a fresh env and captures rgb_array frames (env_id is dispatched per file).
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from make_videos import render_episode_video


def first_success_condition(h5_path: Path, condition_id: int) -> int:
    with h5py.File(h5_path, "r") as f:
        for k in sorted(f.keys()):
            if not k.startswith("episode_"):
                continue
            g = f[k]
            if int(g.attrs.get("condition_id", -1)) != condition_id:
                continue
            if "success" in g and bool(np.asarray(g["success"])[-1]):
                return int(k.split("_")[1])
    raise RuntimeError(f"no successful episode with condition_id={condition_id} in {h5_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("phase_switch_symmetry_videos"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()

    root = Path(".")
    rollouts = root / "phase_switch_symmetry_rollouts_rotated"
    condition_id = 4  # isolated yaw +30 deg (d_axial = +0.5236)

    jobs = [
        (
            "21_keyed_asymmetric_yaw_p30",
            rollouts / "rotated_Q1_seed_20260818.h5",
            condition_id,
            "Keyed asymmetric: key aligns to +30deg gate (yaw relevant) then unlocks to 0",
        ),
        (
            "22_circular_honest_yaw_p30",
            rollouts / "circular_honest_seed_20260818.h5",
            condition_id,
            "Circular honest: peg inserts flat (0deg) — d_axial is a pure gauge DOF",
        ),
        (
            "23_circular_placebo_yaw_p30",
            rollouts / "circular_placebo_seed_20260818.h5",
            condition_id,
            "Circular placebo: fixed +15deg yaw sweep unrelated to d_axial, then 0deg insert",
        ),
    ]

    report = []
    for label, h5_path, cid, desc in jobs:
        ep = first_success_condition(h5_path, cid)
        out = args.out_dir / f"{label}.mp4"
        print(f"[render] {label}: {h5_path.name} cond={cid} ep={ep}  ({desc})", flush=True)
        info = render_episode_video(
            h5_path,
            ep,
            out,
            fps=args.fps,
            max_frames=args.max_frames,
            overlay=not args.no_overlay,
        )
        info.update(label=label, out=str(out), description=desc, condition_id=cid)
        report.append(info)
        print(f"   -> {out}  frames={info['frames']}/{info['steps']} success={info['success']}", flush=True)

    manifest = args.out_dir / "symmetry_transfer_video_manifest.json"
    import json

    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
