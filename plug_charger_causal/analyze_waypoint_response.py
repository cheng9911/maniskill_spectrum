from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np


def quat_to_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_yaw(rotation: np.ndarray) -> float:
    return math.atan2(rotation[1, 0], rotation[0, 0])


def pose_to_matrix(raw_pose: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_to_matrix_wxyz(raw_pose[3:7])
    transform[:3, 3] = raw_pose[:3]
    return transform


def xyz_yaw_from_transform(transform: np.ndarray) -> np.ndarray:
    return np.array(
        [transform[0, 3], transform[1, 3], matrix_to_yaw(transform[:3, :3])],
        dtype=np.float64,
    )


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def relative_xy_yaw(base: np.ndarray, target: np.ndarray) -> np.ndarray:
    rel = np.linalg.inv(base) @ target
    return np.array([rel[0, 3], rel[1, 3], wrap_angle(matrix_to_yaw(rel[:3, :3]))])


def get_pose(sample: h5py.Group, name: str) -> np.ndarray:
    return pose_to_matrix(sample["obs"]["extra"][name][0])


def phase_transforms(sample: h5py.Group) -> dict[str, np.ndarray]:
    tcp = get_pose(sample, "tcp_pose")
    charger = get_pose(sample, "charger_pose")
    goal = get_pose(sample, "goal_pose")
    offset = np.eye(4, dtype=np.float64)
    offset[0, 3] = -0.05

    return {
        "approach_start_tcp": tcp,
        "pre_insert_target": goal @ offset @ np.linalg.inv(charger) @ tcp,
        "insert_target": goal @ np.linalg.inv(charger) @ tcp,
    }


def fit_response(deltas: np.ndarray, responses: np.ndarray) -> dict:
    # Least-squares map: response ~= delta @ coeff.T
    coeff, *_ = np.linalg.lstsq(deltas, responses, rcond=None)
    pred = deltas @ coeff
    residual = responses - pred
    rmse = np.sqrt(np.mean(residual * residual, axis=0))
    return {
        "matrix_rows_response_cols_delta": coeff.T.tolist(),
        "diag": np.diag(coeff.T).tolist(),
        "rmse": rmse.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path(
            "maniskill_spectrum/plug_charger_causal/reset_datasets/isolated_grid/reset_states.h5"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("maniskill_spectrum/plug_charger_causal/WAYPOINT_RESPONSE.md"),
    )
    args = parser.parse_args()

    with h5py.File(args.h5, "r") as h5:
        samples = [h5[f"sample_{i}"] for i in range(len(h5.keys()))]
        base = next(
            sample
            for sample in samples
            if np.allclose(sample["causal_delta"][:], np.zeros(3))
        )
        base_phases = phase_transforms(base)

        rows = []
        phase_responses: dict[str, list[np.ndarray]] = {
            name: [] for name in base_phases.keys()
        }
        deltas = []

        for sample in samples:
            delta = sample["causal_delta"][:].astype(np.float64)
            phases = phase_transforms(sample)
            row = {
                "sample": sample.name,
                "generator": sample.attrs["generator"],
                "delta": delta.tolist(),
            }
            deltas.append(delta)
            for name, transform in phases.items():
                response = relative_xy_yaw(base_phases[name], transform)
                phase_responses[name].append(response)
                row[name] = response.tolist()
            rows.append(row)

    deltas_arr = np.asarray(deltas)
    summary = {}
    for name, responses in phase_responses.items():
        responses_arr = np.asarray(responses)
        summary[name] = fit_response(deltas_arr, responses_arr)
        summary[name]["max_abs_response"] = np.max(np.abs(responses_arr), axis=0).tolist()

    report = {
        "h5": str(args.h5),
        "response_coordinates": "body-frame log approximation [x_m, y_m, yaw_rad]",
        "delta_coordinates": "[delta_x_m, delta_y_m, delta_yaw_rad]",
        "summary": summary,
        "rows": rows,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    args.out.write_text(
        "# PlugCharger-Causal Waypoint Response\n\n"
        "This checks the geometry implied by the official PlugCharger motion-planning "
        "solution without invoking `mplib`. It compares phase waypoint transforms "
        "under isolated `do(dx)`, `do(dy)`, and `do(dyaw)` socket perturbations.\n\n"
        f"```json\n{json.dumps(report, indent=2, sort_keys=True)}\n```\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
