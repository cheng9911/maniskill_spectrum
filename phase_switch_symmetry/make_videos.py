from __future__ import annotations

import argparse
import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
from pathlib import Path

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_env  # noqa: F401
from mani_skill.utils.structs.pose import Pose

PHASE_NAMES = {
    -1: "initial",
    0: "reach",
    1: "grasp",
    2: "lift",
    3: "align_keyed",
    4: "enter_key",
    5: "unlock_yaw",
    6: "circular_insert",
}


def load_episode(h5_path: Path, episode_id: int):
    with h5py.File(h5_path, "r") as f:
        g = f[f"episode_{episode_id}"]
        data = dict(
            qpos=np.asarray(g["qpos"], dtype=np.float64),
            peg_pose=np.asarray(g["peg_pose"], dtype=np.float64),
            socket_pose=np.asarray(g["socket_pose"], dtype=np.float64),
            solver_phase=np.asarray(g["solver_phase"], dtype=int),
            causal_delta=np.asarray(g["causal_delta"], dtype=np.float64),
            success=bool(g["success"][-1]) if "success" in g else None,
        )
        episode_attrs = dict(g.attrs)
        root_attrs = dict(f.attrs)
    return data, episode_attrs, root_attrs


def make_env(orientation, task_anchor, env_id="KeyedCircularPhaseSwitch-v1"):
    kwargs = dict(
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_mode="rgb_array",
    )
    if orientation is not None:
        kwargs["orientation"] = orientation
    if task_anchor is not None:
        kwargs["task_anchor"] = task_anchor
    return gym.make(env_id, **kwargs)


def overlay_phase(frame, phase):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return frame
    name = PHASE_NAMES.get(int(phase), str(int(phase)))
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    text = f"phase: {name}"
    # dark band at top for legibility
    draw.rectangle([0, 0, img.width, 44], fill=(0, 0, 0))
    draw.text((12, 12), text, fill=(255, 255, 255))
    return np.asarray(img)


def render_episode_video(
    h5_path: Path,
    episode_id: int,
    out_path: Path,
    fps: int = 30,
    max_frames: int = 600,
    overlay: bool = True,
    stop_after_phase: int | None = None,
    start_step: int = 0,
):
    data, episode_attrs, root_attrs = load_episode(h5_path, episode_id)
    causal_delta = data["causal_delta"]
    orientation = root_attrs.get("orientation", None)
    task_anchor = root_attrs.get("task_anchor", None)
    env_id = root_attrs.get("env_id", "KeyedCircularPhaseSwitch-v1")
    if isinstance(env_id, bytes):
        env_id = env_id.decode()
    if orientation is not None:
        orientation = np.asarray(orientation, dtype=np.float64)
    if task_anchor is not None:
        task_anchor = np.asarray(task_anchor, dtype=np.float64)

    env = make_env(orientation, task_anchor, env_id=env_id)
    n = len(data["qpos"])
    # Subsample long episodes to keep the video concise.
    every = max(1, int(np.ceil(n / max_frames)))
    idx = list(range(0, n, every))
    if start_step > 0:
        idx = [i for i in idx if i >= start_step]
    if stop_after_phase is not None:
        idx = [i for i in idx if data["solver_phase"][i] <= stop_after_phase]
    if not idx:
        idx = [0]

    try:
        env.reset(
            seed=int(episode_attrs.get("episode_seed", 20260818)),
            options={"causal_delta": causal_delta.tolist()},
        )
        base = env.unwrapped
        device = base.device
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
        try:
            for i in idx:
                qpos = torch.tensor(
                    data["qpos"][i], dtype=torch.float32, device=device
                ).reshape(1, -1)
                peg_p = torch.tensor(
                    data["peg_pose"][i][:3], dtype=torch.float32, device=device
                ).reshape(1, 3)
                peg_q = torch.tensor(
                    data["peg_pose"][i][3:], dtype=torch.float32, device=device
                ).reshape(1, 4)
                sock_p = torch.tensor(
                    data["socket_pose"][i][:3], dtype=torch.float32, device=device
                ).reshape(1, 3)
                sock_q = torch.tensor(
                    data["socket_pose"][i][3:], dtype=torch.float32, device=device
                ).reshape(1, 4)
                base.agent.robot.set_qpos(qpos)
                base.peg.set_pose(Pose.create_from_pq(peg_p, peg_q))
                base.socket.set_pose(Pose.create_from_pq(sock_p, sock_q))
                frame = env.render()
                if hasattr(frame, "cpu"):
                    frame = frame.cpu().numpy()
                else:
                    frame = np.asarray(frame)
                frame = np.squeeze(frame)
                if overlay:
                    frame = overlay_phase(frame, data["solver_phase"][i])
                writer.append_data(frame)
        finally:
            writer.close()
    finally:
        env.close()
    return dict(frames=len(idx), steps=n, every=every, success=data["success"])


def first_success_episode(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as f:
        for k in sorted(f.keys()):
            if not k.startswith("episode_"):
                continue
            g = f[k]
            if "success" in g and bool(g["success"][-1]):
                return int(k.split("_")[1])
    raise RuntimeError(f"no successful episode in {h5_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("phase_switch_symmetry_videos"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only", type=str, default=None, help="Render a single job by label.")
    args = parser.parse_args()

    root = Path(".")
    jobs = [
        # (label, h5_path, episode_id, description)
        (
            "01_nominal_zero_intervention",
            root / "phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5",
            2,
            "Nominal insertion, yaw=0: align -> enter -> unlock -> insert",
        ),
        (
            "02_yaw_matched_plus30",
            root / "phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5",
            5,
            "Socket yaw +30deg: key follows yaw through gate, then unlocks to nominal",
        ),
        (
            "03_translation_neg15mm",
            root / "phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5",
            7,
            "Lateral translation -15mm: translation propagated, yaw suppressed",
        ),
        (
            "04_yaw_mismatched_blocked",
            root / "phase_switch_symmetry_rollouts/keyed_circular_geometry_probes_v2.h5",
            1,
            "Mismatched yaw (key=0, socket=-30deg): key blocked at gate (counterfactual)",
        ),
        (
            "05_matched_preclear_plus30",
            root / "phase_switch_symmetry_rollouts/keyed_circular_geometry_probes_v2.h5",
            2,
            "Matched yaw +30deg clears the keyed gate (geometry probe)",
        ),
        (
            "06_postclear_yaw_plus30",
            root / "phase_switch_symmetry_rollouts/keyed_circular_geometry_probes_v2.h5",
            6,
            "After key clearance, arbitrary +30deg yaw inserts into circular bore",
        ),
    ]

    # Rotated-axis (Q2 = Ry(90deg), horizontal insertion) — orientation invariance.
    rotated_h5 = root / "phase_switch_symmetry_rollouts_rotated/rotated_Q2_seed_20260818.h5"
    if rotated_h5.exists():
        jobs.append(
            (
                "07_rotated_Q2_horizontal",
                rotated_h5,
                first_success_episode(rotated_h5),
                "Horizontal (Ry90deg) insertion — generator law is orientation-invariant",
            )
        )

    if args.smoke:
        jobs = jobs[:1]

    # Per-job start-step overrides (drop free-space planner rotation before the
    # keyed gate). Step 388 is where the nominal episode's peg yaw begins its
    # final -24deg -> 0deg settle into the gate.
    start_steps = {"01_nominal_zero_intervention": 388}

    if args.only:
        jobs = [j for j in jobs if j[0] == args.only]
        if not jobs:
            raise SystemExit(f"unknown label: {args.only}")

    report = []
    for label, h5_path, ep, desc in jobs:
        out = args.out_dir / f"{label}.mp4"
        print(f"[render] {label}: {h5_path.name} ep={ep}  ({desc})", flush=True)
        info = render_episode_video(
            h5_path,
            ep,
            out,
            fps=args.fps,
            max_frames=args.max_frames,
            overlay=not args.no_overlay,
            start_step=start_steps.get(label, 0),
        )
        info.update(label=label, out=str(out), description=desc)
        report.append(info)
        print(f"   -> {out}  frames={info['frames']}/{info['steps']} success={info['success']}", flush=True)

    with (args.out_dir / "video_manifest.json").open("w", encoding="utf-8") as fh:
        import json

        json.dump(report, fh, indent=2)
    print(f"wrote manifest: {args.out_dir / 'video_manifest.json'}")


if __name__ == "__main__":
    main()
