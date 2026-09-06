from __future__ import annotations

"""Collection stability: trajectory-level deviation vs law-level stability.

Given two collections of the SAME nine isolated conditions (same fixture), this
script quantifies the contrast that matters for Major Concern 3:

    trajectory:  X_A(s) != X_B(s)      ->  E_X^RMS, E_X^max, e_end  (nonzero)
    law:         alpha_A(s) ~ alpha_B(s) -> rho_alpha, E_alpha, D_alpha, s_0.5

The two collections may differ by a rerun (same env + same seed -> expected
bit-identical, a determinism check) or by a task-setup change (e.g. peg spawn
position), in which case the approach/align phase can deviate substantially
while the insertion phases and the identified profile stay stable.

The trajectory-level deviation is measured on the common progress grid (4 phases
x bins), with x/y in metres and yaw in radians scaled to mm-equivalent by the
paper's ROTATION_SCALE_M_PER_RAD = 0.03 m/rad.  It does NOT rely on matching
time indices (the two collections can differ in step count), only on the shared
phase-local progress coordinate.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from analyze_phase_switch_rollouts import episode_keys, wrap_pi
from benchmark_phase_switch_baselines import (
    PHASE_CODES,
    empirical_isolated_profile,
    progress_grid,
    switch_diagnostics,
    task_curve,
    usable,
)
from phase_switch_baselines import METRIC_SCALE


def usable_episodes(path: Path):
    with h5py.File(path, "r") as data_file:
        keys = episode_keys(data_file)
        by_condition = {}
        for key in keys:
            group = data_file[key]
            if not usable(group):
                continue
            condition_id = int(group.attrs["condition_id"])
            by_condition[condition_id] = key
    return by_condition


def alpha_yaw_profile(path: Path, bins: int):
    with h5py.File(path, "r") as data_file:
        keys = episode_keys(data_file)
        usable_keys = [key for key in keys if usable(data_file[key])]
        isolated_all = [
            key
            for key in usable_keys
            if str(data_file[key].attrs.get("generator", "")) in {"yaw", "translation"}
        ]
        baseline = [
            key
            for key in isolated_all
            if np.linalg.norm(np.asarray(data_file[key]["causal_delta"])) < 1e-12
        ][0]
        isolated = [key for key in isolated_all if key != baseline]
        diagonal = empirical_isolated_profile(data_file, baseline, isolated, bins)
    return diagonal[:, 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument(
        "--output", type=Path, default=Path("phase_switch_symmetry_invariance/rerun_robustness.json")
    )
    args = parser.parse_args()

    bins = args.bins
    progress, phase_codes = progress_grid(bins)
    n_bins = len(progress)

    rerun = usable_episodes(args.rerun)
    reference = usable_episodes(args.reference)
    common_conditions = sorted(set(rerun) & set(reference))
    if len(common_conditions) != 9:
        raise RuntimeError(
            f"expected 9 common isolated conditions, got {len(common_conditions)}"
        )

    with h5py.File(args.rerun, "r") as fr, h5py.File(args.reference, "r") as fo:
        # Trajectory-level deviation on the shared progress grid.
        per_bin = []   # mm-equiv, one scalar per (condition, bin)
        endpoint = []  # mm-equiv, one scalar per condition (final bin)
        for condition_id in common_conditions:
            curve_rerun = task_curve(fr[rerun[condition_id]], bins)
            curve_ref = task_curve(fo[reference[condition_id]], bins)
            # Wrap the yaw difference to [-pi, pi]: a full 2*pi rotation is
            # physically identical, so the raw unwrapped yaw can spuriously add
            # 2*pi*METRIC_SCALE to an otherwise-matched trajectory.
            residual = np.column_stack(
                [
                    curve_rerun[:, 0] - curve_ref[:, 0],
                    curve_rerun[:, 1] - curve_ref[:, 1],
                    np.vectorize(wrap_pi)(curve_rerun[:, 2] - curve_ref[:, 2]),
                ]
            ) * METRIC_SCALE
            err = np.linalg.norm(residual, axis=1)
            per_bin.append(err)
            endpoint.append(err[-1])
        per_bin = np.concatenate(per_bin)
        endpoint = np.asarray(endpoint)

    e_x_rms = 1000.0 * float(np.sqrt(np.mean(per_bin ** 2)))
    e_x_max = 1000.0 * float(np.max(per_bin))
    e_end_mean = 1000.0 * float(np.mean(endpoint))
    e_end_max = 1000.0 * float(np.max(endpoint))

    # Law-level stability.
    alpha_rerun = alpha_yaw_profile(args.rerun, bins)
    alpha_ref = alpha_yaw_profile(args.reference, bins)
    rho_alpha = float(np.corrcoef(alpha_rerun, alpha_ref)[0, 1])
    diff = alpha_rerun - alpha_ref
    e_alpha_rms = float(np.sqrt(np.mean(diff ** 2)))
    d_alpha_inf = float(np.max(np.abs(diff)))

    switch_rerun = switch_diagnostics(progress, alpha_rerun, unlock_start=2 * bins)
    switch_ref = switch_diagnostics(progress, alpha_ref, unlock_start=2 * bins)

    # Also report the trajectory endpoint discrepancy in native units (mm) for
    # xy and (deg) for yaw, so the reader can gauge the "0.6 mm" number.
    with h5py.File(args.rerun, "r") as fr, h5py.File(args.reference, "r") as fo:
        xy_end = []
        yaw_end = []
        for condition_id in common_conditions:
            curve_rerun = task_curve(fr[rerun[condition_id]], bins)[-1]
            curve_ref = task_curve(fo[reference[condition_id]], bins)[-1]
            xy_end.append(np.linalg.norm(curve_rerun[:2] - curve_ref[:2]) * 1e3)
            yaw_end.append(np.rad2deg(wrap_pi(curve_rerun[2] - curve_ref[2])))
        xy_end = np.asarray(xy_end)
        yaw_end = np.asarray(yaw_end)

    result = {
        "schema_version": 1,
        "rerun": str(args.rerun),
        "reference": str(args.reference),
        "bins_per_phase": bins,
        "common_conditions": common_conditions,
        "trajectory_level": {
            "E_X_rms_mm_equiv": e_x_rms,
            "E_X_max_mm_equiv": e_x_max,
            "e_end_mean_mm_equiv": e_end_mean,
            "e_end_max_mm_equiv": e_end_max,
            "endpoint_xy_mean_mm": float(np.mean(xy_end)),
            "endpoint_xy_max_mm": float(np.max(xy_end)),
            "endpoint_yaw_mean_deg": float(np.mean(np.abs(yaw_end))),
            "endpoint_yaw_max_deg": float(np.max(np.abs(yaw_end))),
            "interpretation": (
                "x/y in mm, yaw in deg at the shared endpoint.  Zero for a same-env "
                "deterministic rerun; nonzero when the two collections differ by setup "
                "(peg spawn) or execution-stack variant."
            ),
        },
        "law_level": {
            "rho_alpha": rho_alpha,
            "E_alpha_rms": e_alpha_rms,
            "D_alpha_inf": d_alpha_inf,
            "switch_s_0_5": {
                "rerun": switch_rerun["location"],
                "reference": switch_ref["location"],
            },
            "switch_s_0_5_spread": float(
                abs(switch_rerun["location"] - switch_ref["location"])
            ),
            "switch_status": {
                "rerun": switch_rerun["status"],
                "reference": switch_ref["status"],
            },
        },
        "contrast": {
            "trajectory_deviates": bool(e_x_max > 0.05),
            "law_stable": bool(rho_alpha > 0.99 and d_alpha_inf < 0.1),
            "summary": (
                "X_A(s) != X_B(s) whenever E_X_max > 0, but alpha_yaw(s) ~ alpha_ref(s) "
                "whenever rho ~ 1 and the absolute deviation is small."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=== trajectory-level (rerun vs reference) ===")
    print(f"  E_X^RMS      = {e_x_rms:.4f} mm-equiv")
    print(f"  E_X^max      = {e_x_max:.4f} mm-equiv")
    print(f"  e_end mean   = {e_end_mean:.4f} mm-equiv")
    print(f"  e_end max    = {e_end_max:.4f} mm-equiv")
    print(f"  endpoint xy  mean={np.mean(xy_end):.4f} mm  max={np.max(xy_end):.4f} mm")
    print(f"  endpoint yaw mean={np.mean(np.abs(yaw_end)):.4f} deg  max={np.max(np.abs(yaw_end)):.4f} deg")
    print()
    print("=== law-level (alpha_yaw) ===")
    print(f"  rho_alpha    = {rho_alpha:.6f}")
    print(f"  E_alpha^RMS  = {e_alpha_rms:.6f}")
    print(f"  D_alpha,inf  = {d_alpha_inf:.6f}")
    print(f"  s_0.5 rerun  = {switch_rerun['location']:.5f}")
    print(f"  s_0.5 ref     = {switch_ref['location']:.5f}")
    print(f"  s_0.5 spread  = {abs(switch_rerun['location'] - switch_ref['location']):.5f}")
    print()
    print("saved:", args.output)


if __name__ == "__main__":
    main()
