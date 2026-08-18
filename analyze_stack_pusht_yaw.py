from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np


def quat_yaw_wxyz(q: np.ndarray) -> float:
    w, x, y, z = q / np.linalg.norm(q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def actor_pose(group: h5py.Group, actor: str, t: int) -> tuple[np.ndarray, float]:
    raw = group["env_states"]["actors"][actor][t]
    return raw[:3].astype(np.float64), quat_yaw_wxyz(raw[3:7].astype(np.float64))


def circular_corr(a: np.ndarray, b: np.ndarray) -> float:
    # Jammalamadaka-Sengupta circular correlation.
    a_bar = math.atan2(np.sin(a).mean(), np.cos(a).mean())
    b_bar = math.atan2(np.sin(b).mean(), np.cos(b).mean())
    sa = np.sin(a - a_bar)
    sb = np.sin(b - b_bar)
    denom = math.sqrt(float(np.sum(sa * sa) * np.sum(sb * sb)))
    return float(np.sum(sa * sb) / denom) if denom > 0 else float("nan")


def analyze_stack(path: Path, meta_path: Path, max_episodes: int | None) -> dict:
    meta = json.loads(meta_path.read_text())
    rows = []
    with h5py.File(path, "r") as h5:
        episodes = meta["episodes"][:max_episodes]
        for ep in episodes:
            group = h5[f"traj_{ep['episode_id']}"]
            if not ep.get("success", True):
                continue
            cube_a_pos, cube_a_yaw = actor_pose(group, "cubeA", -1)
            cube_b_pos, cube_b_yaw = actor_pose(group, "cubeB", -1)
            rows.append(
                {
                    "xy_error": float(np.linalg.norm(cube_a_pos[:2] - cube_b_pos[:2])),
                    "height_delta": float(cube_a_pos[2] - cube_b_pos[2]),
                    "yaw_rel": float(wrap_angle(cube_a_yaw - cube_b_yaw)),
                    "cube_a_yaw": cube_a_yaw,
                    "cube_b_yaw": cube_b_yaw,
                }
            )
    yaw_rel = np.asarray([r["yaw_rel"] for r in rows])
    cube_a_yaw = np.asarray([r["cube_a_yaw"] for r in rows])
    cube_b_yaw = np.asarray([r["cube_b_yaw"] for r in rows])
    xy = np.asarray([r["xy_error"] for r in rows])
    return {
        "dataset": str(path),
        "episodes": len(rows),
        "xy_error_mean_m": float(xy.mean()),
        "xy_error_p95_m": float(np.quantile(xy, 0.95)),
        "abs_yaw_rel_mean_rad": float(np.mean(np.abs(yaw_rel))),
        "abs_yaw_rel_p50_deg": float(np.degrees(np.quantile(np.abs(yaw_rel), 0.50))),
        "abs_yaw_rel_p95_deg": float(np.degrees(np.quantile(np.abs(yaw_rel), 0.95))),
        "yaw_match_within_10deg_frac": float(np.mean(np.abs(yaw_rel) <= math.radians(10))),
        "cubeA_cubeB_yaw_circular_corr": circular_corr(cube_a_yaw, cube_b_yaw),
    }


def analyze_pusht(path: Path, meta_path: Path, max_episodes: int | None) -> dict:
    meta = json.loads(meta_path.read_text())
    rows = []
    with h5py.File(path, "r") as h5:
        episodes = meta["episodes"][:max_episodes]
        for ep in episodes:
            group = h5[f"traj_{ep['episode_id']}"]
            if not ep.get("success", True):
                continue
            tee_pos, tee_yaw = actor_pose(group, "Tee", -1)
            goal_pos, goal_yaw = actor_pose(group, "goal_Tee", -1)
            rows.append(
                {
                    "xy_error": float(np.linalg.norm(tee_pos[:2] - goal_pos[:2])),
                    "yaw_error": float(wrap_angle(tee_yaw - goal_yaw)),
                    "tee_yaw": tee_yaw,
                    "goal_yaw": goal_yaw,
                }
            )
    yaw_error = np.asarray([r["yaw_error"] for r in rows])
    tee_yaw = np.asarray([r["tee_yaw"] for r in rows])
    goal_yaw = np.asarray([r["goal_yaw"] for r in rows])
    xy = np.asarray([r["xy_error"] for r in rows])
    return {
        "dataset": str(path),
        "episodes": len(rows),
        "xy_error_mean_m": float(xy.mean()),
        "xy_error_p95_m": float(np.quantile(xy, 0.95)),
        "abs_yaw_error_mean_rad": float(np.mean(np.abs(yaw_error))),
        "abs_yaw_error_p50_deg": float(np.degrees(np.quantile(np.abs(yaw_error), 0.50))),
        "abs_yaw_error_p95_deg": float(np.degrees(np.quantile(np.abs(yaw_error), 0.95))),
        "yaw_match_within_10deg_frac": float(np.mean(np.abs(yaw_error) <= math.radians(10))),
        "tee_goal_yaw_circular_corr": circular_corr(tee_yaw, goal_yaw),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("maniskill_spectrum/STACK_PUSHT_YAW_RELEVANCE.md"),
    )
    args = parser.parse_args()

    stack_h5 = Path("maniskill_spectrum/demos/StackCube-v1/motionplanning/trajectory.h5")
    stack_json = stack_h5.with_suffix(".json")
    pusht_h5 = Path(
        "maniskill_spectrum/demos/PushT-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5"
    )
    pusht_json = pusht_h5.with_suffix(".json")

    report = {
        "stackcube": analyze_stack(stack_h5, stack_json, args.max_episodes),
        "pusht": analyze_pusht(pusht_h5, pusht_json, args.max_episodes),
        "interpretation": {
            "stackcube": "Successful stacking tightly constrains cubeA/cubeB xy, but does not require cubeA yaw to match cubeB yaw.",
            "pusht": "Successful PushT tightly constrains Tee yaw to goal_Tee yaw.",
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    args.out.write_text(
        "# StackCube vs PushT Yaw Relevance\n\n"
        "This is an observational task-constraint check on downloaded ManiSkill "
        "demonstrations. It is not a controlled intervention test.\n\n"
        f"```json\n{json.dumps(report, indent=2, sort_keys=True)}\n```\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
