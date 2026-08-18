from __future__ import annotations

import argparse
import json
import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np

import mani_skill.envs  # noqa: F401
import vertical_peg_symmetry_env  # noqa: F401
from collect_vertical_peg_symmetry import TraceWrapper, solve, write_episode


def sample_row(
    rng: np.random.Generator,
    yaw_mode: str,
    dx_range: tuple[float, float],
    dy_range: tuple[float, float],
    yaw_deg_range: tuple[float, float],
):
    yaw_lo, yaw_hi = np.deg2rad(yaw_deg_range)
    if yaw_mode == "negative":
        yaw = rng.uniform(yaw_lo, min(yaw_hi, 0.0))
    elif yaw_mode == "positive":
        yaw = rng.uniform(max(yaw_lo, 0.0), yaw_hi)
    elif yaw_mode == "balanced":
        sign = -1.0 if rng.random() < 0.5 else 1.0
        mag_lo, mag_hi = sorted([abs(yaw_lo), abs(yaw_hi)])
        yaw = sign * rng.uniform(mag_lo, mag_hi)
    else:
        yaw = rng.uniform(yaw_lo, yaw_hi)
    return dict(
        generator="mixed",
        causal_delta=[
            float(rng.uniform(*dx_range)),
            float(rng.uniform(*dy_range)),
            float(yaw),
        ],
    )


def collect_targeted(args):
    rng = np.random.default_rng(args.seed)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env_id = "VerticalSquarePegSymmetry-v1"
    base = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_mode=None,
        robot_init_qpos_noise=0.0,
    )
    env = TraceWrapper(base)
    manifest = []
    success_count = 0
    negative_success_count = 0
    with h5py.File(output_path, "w") as f:
        f.attrs["env_id"] = env_id
        f.attrs["task"] = "square"
        f.attrs["source_type"] = "vertical_peg_symmetry_targeted_square_mixed_physics"
        f.attrs["control_mode"] = "pd_joint_pos"
        f.attrs["sim_backend"] = "physx_cpu"
        f.attrs["target_success"] = args.target_success
        f.attrs["target_negative_success"] = args.target_negative_success
        f.attrs["yaw_mode"] = args.yaw_mode
        f.attrs["dx_range"] = args.dx_range
        f.attrs["dy_range"] = args.dy_range
        f.attrs["yaw_deg_range"] = args.yaw_deg_range
        for i in range(args.max_attempts):
            row = sample_row(
                rng,
                args.yaw_mode,
                tuple(args.dx_range),
                tuple(args.dy_range),
                tuple(args.yaw_deg_range),
            )
            print(
                f"[square-targeted] {i+1:03d}/{args.max_attempts} "
                f"success={success_count} neg_success={negative_success_count} {row}"
            )
            env.reset(seed=0, options={"causal_delta": row["causal_delta"]})
            env.start_trace()
            solver_error = None
            try:
                solve(env)
            except Exception as exc:
                solver_error = repr(exc)
                print("  solver exception:", solver_error)
            states = env.trace["states"]
            forces = np.linalg.norm(
                np.asarray([s["contact_force"] for s in states]), axis=1
            )
            success = bool(states[-1]["success"])
            yaw = float(row["causal_delta"][2])
            if success:
                success_count += 1
                if yaw < 0:
                    negative_success_count += 1
            steps = len(env.trace["actions"])
            print(
                f"  steps={steps} success={success} yaw_deg={np.rad2deg(yaw):.2f} "
                f"max_contact={forces.max():.3f}"
            )
            g = f.create_group(f"episode_{i}")
            write_episode(g, "square", row, env.trace, solver_error)
            manifest.append(
                dict(
                    episode_id=i,
                    task="square",
                    **row,
                    steps=steps,
                    success=success,
                    max_contact_force_N=float(forces.max()),
                    solver_error=solver_error,
                )
            )
            if (
                success_count >= args.target_success
                and negative_success_count >= args.target_negative_success
            ):
                break
        f.attrs["success_count"] = success_count
        f.attrs["negative_success_count"] = negative_success_count
        f.attrs["attempt_count"] = len(manifest)
    env.close()
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as jf:
        json.dump(
            dict(
                task="square",
                yaw_mode=args.yaw_mode,
                target_success=args.target_success,
                target_negative_success=args.target_negative_success,
                success_count=success_count,
                negative_success_count=negative_success_count,
                episodes=manifest,
            ),
            jf,
            indent=2,
        )
    if success_count < args.target_success:
        raise RuntimeError(
            f"only collected {success_count} successes, target is {args.target_success}"
        )
    if negative_success_count < args.target_negative_success:
        raise RuntimeError(
            f"only collected {negative_success_count} negative-yaw successes, "
            f"target is {args.target_negative_success}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(
            "vertical_peg_symmetry_rollouts_square_augmented/"
            "square_mixed_targeted_physics.h5"
        ),
    )
    ap.add_argument("--target-success", type=int, default=20)
    ap.add_argument("--target-negative-success", type=int, default=15)
    ap.add_argument("--max-attempts", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--dx-range", nargs=2, type=float, default=[-0.012, 0.012])
    ap.add_argument("--dy-range", nargs=2, type=float, default=[-0.012, 0.012])
    ap.add_argument("--yaw-deg-range", nargs=2, type=float, default=[-30.0, -2.0])
    ap.add_argument(
        "--yaw-mode",
        choices=["negative", "positive", "balanced", "uniform"],
        default="negative",
    )
    args = ap.parse_args()
    collect_targeted(args)


if __name__ == "__main__":
    main()
