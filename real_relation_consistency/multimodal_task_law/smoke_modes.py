from __future__ import annotations

"""Smoke-test the four modes + yaw challenge: reachability and distinctness.

Runs each mode once at a fixed pitch and prints per-step summary signals so we
can confirm (a) the solver completes, (b) the 3D peg path differs between
lift/diagonal, and (c) the pitch re-orientation happens earlier/later for
early/late.  Also renders a video per mode for visual spot-checking.
"""

import sqlite3  # noqa: F401
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "phase_switch_symmetry"))
sys.path.insert(0, str(ROOT / "paper_video"))
sys.path.insert(0, str(HERE))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transforms3d.quaternions import quat2mat  # noqa: E402

import mani_skill.envs  # noqa: F401, E402
import phase_switch_symmetry_env  # noqa: F401, E402
from collect_phase_switch_rollouts import EpisodeFinished  # noqa: E402
from phase_switch_se3_baselines import euler_from_matrix  # noqa: E402
from solver_modes import (  # noqa: E402
    ModeTraceWrapper,
    solve_multimodal_se3,
    solve_yaw_challenge,
)

ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])
TASK_ANCHOR = [-0.15, 0.00, 0.08]
PICKUP = [-0.25, -0.18, 0.28]
SEED = 20260910


def run(mode_key: str, causal_delta, render=False):
    base = gym.make(
        "CircularPhaseSwitchSE3-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_mode="rgb_array" if render else None,
        robot_init_qpos_noise=0.0,
        orientation=ORIENTATION,
        task_anchor=TASK_ANCHOR,
    )
    frames = []
    env = ModeTraceWrapper(base)
    if render:
        orig_step = env.step

        def step(action):
            out = orig_step(action)
            rgb = base.render()
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.detach().cpu().numpy()
            rgb = np.asarray(rgb)
            if rgb.ndim == 4:
                rgb = rgb[0]
            frames.append(np.ascontiguousarray(rgb[..., :3]))
            return out

        env.step = step

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env.reset(seed=SEED, options={"causal_delta": np.asarray(causal_delta, dtype=np.float64),
                                  "pickup_position": PICKUP})
    env.start_trace()
    err = None
    try:
        if mode_key in ("yaw_fixed", "yaw_follow"):
            solve_yaw_challenge(env, mode_key)
        else:
            path_mode, alignment_mode = mode_key.split("_", 1)
            solve_multimodal_se3(env, path_mode, alignment_mode)
    except EpisodeFinished as exc:
        err = f"EpisodeFinished({exc})"
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
    states = env.trace["states"]
    env.close()

    peg = np.asarray([s["peg_pose"] for s in states])
    sock = np.asarray([s["socket_pose"] for s in states])
    n = len(states)
    success = bool(states[-1]["success"])
    # pitch angle of the peg (euler R=RxRyRz) over time
    rpy = np.array([euler_from_matrix(quat2mat(q)) for q in peg[:, 3:]])
    pitch_deg = np.rad2deg(np.unwrap(rpy[:, 1]))
    # axial depth of peg centre along the socket axis
    sock_axis = np.asarray(env_sock_axis(quat2mat(sock[0, 3:])), dtype=np.float64)
    depth = np.einsum("ij,j->i", peg[:, :3] - sock[0, :3], sock_axis)
    # first index where pitch reaches 50% of the final pitch
    final_pitch = pitch_deg[-1]
    half = np.argmax(np.abs(pitch_deg) >= 0.5 * abs(final_pitch)) if abs(final_pitch) > 1e-6 else -1
    # first index where depth drops below PRE_ENTRY (0.125) = entering the gate
    entry = np.argmax(depth <= 0.125) if (depth <= 0.125).any() else -1
    return dict(
        mode=mode_key, n=n, success=success, err=err,
        final_pitch_deg=float(final_pitch),
        half_pitch_step=half, entry_step=entry,
        peg_start=peg[0, :3].tolist(), peg_end=peg[-1, :3].tolist(),
        frames=frames,
    )


def env_sock_axis(R):
    # socket_axis = socket R @ [0,0,1]
    return R @ np.array([0.0, 0.0, 1.0])


def main():
    import argparse
    from video_common import write_mp4  # noqa: E402  (paper_video helper)

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=HERE / "smoke")
    args = parser.parse_args()

    pitch = np.deg2rad(7.5)
    cases = [
        ("lift_early", [0, 0, 0, 0, pitch, 0]),
        ("lift_late", [0, 0, 0, 0, pitch, 0]),
        ("diagonal_early", [0, 0, 0, 0, pitch, 0]),
        ("diagonal_late", [0, 0, 0, 0, pitch, 0]),
        ("yaw_fixed", [0, 0, 0, 0, 0, np.deg2rad(30.0)]),
        ("yaw_follow", [0, 0, 0, 0, 0, np.deg2rad(30.0)]),
    ]
    rows = []
    for mode_key, cd in cases:
        print(f"== {mode_key} ==", flush=True)
        r = run(mode_key, cd, render=args.render)
        rows.append(r)
        print(f"   n={r['n']} success={r['success']} err={r['err']} "
              f"final_pitch={r['final_pitch_deg']:.2f}deg "
              f"half_pitch_step={r['half_pitch_step']}/{r['n']} "
              f"entry_step={r['entry_step']}/{r['n']}", flush=True)
        if args.render and r["frames"]:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            write_mp4(args.out_dir / f"{mode_key}.mp4", r["frames"], fps=20.0)

    print("\n--- distinctness check (peg end positions) ---")
    for r in rows:
        print(f"{r['mode']}: end={np.round(r['peg_end'], 4)}")
    # alignment timing: early should have half_pitch_step well before entry; late near entry
    print("\n--- alignment timing (half-pitch vs entry step) ---")
    for r in rows:
        if "lift" in r["mode"] or "diagonal" in r["mode"]:
            print(f"{r['mode']}: half_pitch_step={r['half_pitch_step']} entry_step={r['entry_step']}")


if __name__ == "__main__":
    main()
