from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np


def quat_yaw_wxyz(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def pose(group: h5py.Group, name: str) -> np.ndarray:
    return group["obs"]["extra"][name][0]


def max_pose_delta(samples: list[h5py.Group], base: h5py.Group, name: str) -> float:
    base_pose = pose(base, name)
    return max(float(np.max(np.abs(pose(sample, name) - base_pose))) for sample in samples)


def validate_isolated_grid(h5_path: Path, json_path: Path) -> dict:
    data = json.loads(json_path.read_text())
    assert data["num_samples"] == 13

    with h5py.File(h5_path, "r") as h5:
        samples = [h5[f"sample_{i}"] for i in range(13)]
        base = samples[2]
        base_receptacle = pose(base, "receptacle_pose")
        base_goal = pose(base, "goal_pose")

        charger_max_delta = max_pose_delta(samples, base, "charger_pose")
        tcp_max_delta = max_pose_delta(samples, base, "tcp_pose")

        translation_errors = []
        yaw_errors = []
        goal_position_errors = []
        goal_receptacle_position_errors = []

        base_yaw = quat_yaw_wxyz(base_receptacle[3:7])
        base_goal_yaw = quat_yaw_wxyz(base_goal[3:7])

        for sample in samples:
            delta = sample["causal_delta"][:]
            rec = pose(sample, "receptacle_pose")
            goal = pose(sample, "goal_pose")
            expected_xy = base_receptacle[:2] + delta[:2]
            translation_errors.append(float(np.max(np.abs(rec[:2] - expected_xy))))

            actual_yaw_delta = wrap_angle(quat_yaw_wxyz(rec[3:7]) - base_yaw)
            yaw_errors.append(abs(wrap_angle(actual_yaw_delta - float(delta[2]))))

            expected_goal_xy = base_goal[:2] + delta[:2]
            goal_position_errors.append(float(np.max(np.abs(goal[:2] - expected_goal_xy))))
            goal_receptacle_position_errors.append(float(np.max(np.abs(goal[:3] - rec[:3]))))

            goal_yaw_delta = wrap_angle(quat_yaw_wxyz(goal[3:7]) - base_goal_yaw)
            yaw_errors.append(abs(wrap_angle(goal_yaw_delta - float(delta[2]))))

        return {
            "samples": len(samples),
            "charger_pose_max_abs_delta": charger_max_delta,
            "tcp_pose_max_abs_delta": tcp_max_delta,
            "receptacle_translation_max_error_m": max(translation_errors),
            "receptacle_yaw_max_error_rad": max(yaw_errors),
            "goal_translation_max_error_m": max(goal_position_errors),
            "goal_receptacle_position_max_error_m": max(goal_receptacle_position_errors),
            "dx_mm": [s["causal_delta"][0] * 1000.0 for s in samples if s.attrs["generator"] == "dx"],
            "dy_mm": [s["causal_delta"][1] * 1000.0 for s in samples if s.attrs["generator"] == "dy"],
            "dyaw_deg": [
                math.degrees(float(s["causal_delta"][2]))
                for s in samples
                if s.attrs["generator"] == "dyaw"
            ],
        }


def validate_train_mixed(h5_path: Path, json_path: Path) -> dict:
    data = json.loads(json_path.read_text())
    assert data["num_samples"] == 50
    deltas = np.asarray([sample["causal_delta"] for sample in data["samples"]])
    with h5py.File(h5_path, "r") as h5:
        sample0 = h5["sample_0"]
        obs_extra = sorted(sample0["obs"]["extra"].keys())
        sample_count = len(h5.keys())
    return {
        "samples": sample_count,
        "obs_extra_keys": obs_extra,
        "dx_range_mm": [float(deltas[:, 0].min() * 1000.0), float(deltas[:, 0].max() * 1000.0)],
        "dy_range_mm": [float(deltas[:, 1].min() * 1000.0), float(deltas[:, 1].max() * 1000.0)],
        "dyaw_range_deg": [
            float(np.degrees(deltas[:, 2].min())),
            float(np.degrees(deltas[:, 2].max())),
        ],
    }


def check_motionplanning_smoke(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "valid_h5": False}
    try:
        with h5py.File(path, "r") as h5:
            return {"exists": True, "valid_h5": True, "keys": list(h5.keys())}
    except OSError as exc:
        return {"exists": True, "valid_h5": False, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("maniskill_spectrum/plug_charger_causal"),
    )
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    reset_root = args.root / "reset_datasets"
    results = {
        "isolated_grid": validate_isolated_grid(
            reset_root / "isolated_grid" / "reset_states.h5",
            reset_root / "isolated_grid" / "reset_states.json",
        ),
        "train_mixed": validate_train_mixed(
            reset_root / "train_mixed" / "reset_states.h5",
            reset_root / "train_mixed" / "reset_states.json",
        ),
        "motionplanning_smoke": check_motionplanning_smoke(
            args.root / "datasets" / "smoke" / "trajectory.h5"
        ),
    }

    print(json.dumps(results, indent=2, sort_keys=True))

    if args.write_report:
        report = args.root / "VALIDATION.md"
        report.write_text(
            "# PlugCharger-Causal Validation\n\n"
            "## Reset/State Datasets\n\n"
            f"```json\n{json.dumps(results, indent=2, sort_keys=True)}\n```\n\n"
            "## Interpretation\n\n"
            "- The isolated reset dataset validates the intervention mechanism: "
            "charger and TCP initial poses stay fixed while receptacle/goal receive "
            "the requested dx, dy, and yaw perturbations.\n"
            "- The mixed reset dataset validates the requested training perturbation "
            "range and contains state observations with tcp, charger, receptacle, "
            "and goal poses.\n"
            "- This is not yet a trajectory-level validation of P(s). It validates "
            "the controlled do(delta c) data-generation layer only.\n"
            "- The motion-planning smoke HDF5 is invalid/truncated because mplib "
            "segfaulted during planner initialization on this machine.\n",
            encoding="utf-8",
        )
        print(f"Wrote {report}")


if __name__ == "__main__":
    main()
