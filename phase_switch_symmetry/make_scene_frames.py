from __future__ import annotations

"""Render individual real-simulation frames (RGB) for the paper's scene figure.

This is the primitive behind "真实仿真场景的图": it replays stored qpos + peg/socket
poses from the .h5 rollouts into a fresh env and captures env.render() frames at
selected phases, writing one PNG per (scene, phase). The composition step reads
these PNGs back and lays them out with labels + experiment-count annotations.

Run from phase_switch_symmetry/ so `phase_switch_symmetry_env` is importable.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_videos import load_episode, make_env, first_success_episode  # noqa: E402
from make_symmetry_transfer_videos import first_success_condition  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from mani_skill.utils.structs.pose import Pose  # noqa: E402

PHASE = {3: "align_keyed", 4: "enter_key", 5: "unlock_yaw", 6: "circular_insert", 0: "reach", 2: "lift"}


def render_frame(env, data, i):
    base = env.unwrapped
    device = base.device
    qpos = torch.tensor(data["qpos"][i], dtype=torch.float32, device=device).reshape(1, -1)
    peg_p = torch.tensor(data["peg_pose"][i][:3], dtype=torch.float32, device=device).reshape(1, 3)
    peg_q = torch.tensor(data["peg_pose"][i][3:], dtype=torch.float32, device=device).reshape(1, 4)
    sock_p = torch.tensor(data["socket_pose"][i][:3], dtype=torch.float32, device=device).reshape(1, 3)
    sock_q = torch.tensor(data["socket_pose"][i][3:], dtype=torch.float32, device=device).reshape(1, 4)
    base.agent.robot.set_qpos(qpos)
    base.peg.set_pose(Pose.create_from_pq(peg_p, peg_q))
    base.socket.set_pose(Pose.create_from_pq(sock_p, sock_q))
    frame = env.render()
    if hasattr(frame, "cpu"):
        frame = frame.cpu().numpy()
    else:
        frame = np.asarray(frame)
    return np.squeeze(frame)


def phase_step(data, phases):
    """first step index for each requested phase (in solver_phase order)."""
    ph = data["solver_phase"]
    out = {}
    for p in phases:
        hit = np.where(ph == p)[0]
        out[p] = int(hit[-1]) if len(hit) else None  # last frame of the phase
    return out


def render_scene(h5_path, episode_id, out_dir, label, phases=(3, 4, 5, 6)):
    data, ea, ra = load_episode(Path(h5_path), episode_id)
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
    env.reset(seed=int(ea.get("episode_seed", 20260818)),
              options={"causal_delta": data["causal_delta"].tolist()})
    steps = phase_step(data, phases)
    written = []
    for p in phases:
        i = steps[p]
        if i is None:
            print(f"  [{label}] phase {p} missing", flush=True)
            continue
        frame = render_frame(env, data, i)
        out = Path(out_dir) / f"{label}_{PHASE[p]}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out, frame)
        written.append(str(out))
        print(f"  [{label}] {PHASE[p]} step={i} -> {out.name}  {frame.shape}", flush=True)
    env.close()
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("../phase_switch_symmetry_rollouts_rotated/_scene_frames"))
    parser.add_argument("--cid", type=int, default=4)
    args = parser.parse_args()

    rr = Path("../phase_switch_symmetry_rollouts_rotated")
    jobs = [
        ("keyedQ1", rr / "rotated_Q1_seed_20260818.h5", None, (3, 4, 5, 6)),
        ("keyedQ2", rr / "rotated_Q2_seed_20260818.h5", None, (4, 6)),
        ("honest", rr / "circular_honest_seed_20260818.h5", None, (3, 6)),
        ("placebo", rr / "circular_placebo_seed_20260818.h5", None, (3, 6)),
    ]
    for label, h5, _, phases in jobs:
        if label.startswith("keyed"):
            ep = first_success_condition(h5, args.cid)
        else:
            ep = first_success_condition(h5, args.cid)
        print(f"[scene] {label} {h5.name} ep={ep}", flush=True)
        render_scene(h5, ep, args.out_dir, label, phases=phases)


if __name__ == "__main__":
    main()
