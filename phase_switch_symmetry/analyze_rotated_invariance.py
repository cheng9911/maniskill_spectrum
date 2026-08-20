from __future__ import annotations

"""Rotated-axis canonical-law invariance analysis (3 layers).

For Q1 (vertical) and Q2 (horizontal), each with three execution seeds:
  Layer 1: does each (orientation, seed) recover the correct generator law?
           g_translation_final > 0.9, g_axial_preclear > 0.9, |g_axial_final| < 0.1.
  Layer 2: does a 90-degree global rotation preserve the canonical profile?
           D_orient (RMSE between Q1/Q2 profiles), rho_axial, and per-scalar
           |Delta g| for matched seeds.
  Layer 3: is the orientation variation comparable to ordinary seed variation?
           D_orient vs D_seed (within-orientation cross-seed profile RMSE).

Profiles are canonicalized by G_Q^-1 (see analyze_phase_switch_rotated.py), so
the three generators are [u, v, axial] in the identity task frame.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_phase_switch_rotated import analyze
from benchmark_phase_switch_baselines import progress_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", nargs="+", type=Path, required=True,
        help="Collected H5 files, one per (orientation, seed).",
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/rotated_axis_experiment.json"),
    )
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument(
        "--out", type=Path,
        default=Path("phase_switch_symmetry_rollouts_rotated/rotated_invariance.csv"),
    )
    args = parser.parse_args()

    with args.experiment.open() as f:
        experiment = json.load(f)
    config = {
        "alpha_max": experiment["frozen_model"]["alpha_max"],
        "n_basis": experiment["frozen_model"]["n_basis"],
        "basis_width": experiment["frozen_model"]["basis_width"],
        "smoothness_weight": experiment["frozen_model"]["smoothness_weight"],
        "nominal_iterations": experiment["frozen_model"]["nominal_iterations"],
    }
    progress, _ = progress_grid(args.bins)
    preclear_index = 2 * args.bins - 1

    # Parse (orientation, seed) from the filename rotated_Q<1|2>_seed_<n>.h5.
    profiles = {}
    rows = []
    for path in args.files:
        name = path.stem
        q = "Q1" if "_Q1_" in name else "Q2" if "_Q2_" in name else name
        seed = int(name.split("seed_")[-1])
        profile, _ = analyze(path, config, args.bins)
        profiles[(q, seed)] = profile
        rows.append(
            {
                "orientation": q,
                "seed": seed,
                "g_translation_final": float(profile[-1, :2].mean()),
                "g_axial_preclear": float(profile[preclear_index, 2]),
                "g_axial_final": float(profile[-1, 2]),
            }
        )
    frame = pd.DataFrame(rows)

    # Layer 1: structure law per (orientation, seed).
    frame["law_pass"] = (
        (frame.g_translation_final > 0.9)
        & (frame.g_axial_preclear > 0.9)
        & (frame.g_axial_final.abs() < 0.1)
    )

    # Layer 2: matched-seed orientation difference.
    q1 = [s for (q, s) in profiles if q == "Q1"]
    q2 = [s for (q, s) in profiles if q == "Q2"]
    seeds = sorted(set(q1) & set(q2))
    layer2 = []
    for seed in seeds:
        a = profiles[("Q1", seed)]
        b = profiles[("Q2", seed)]
        layer2.append(
            {
                "seed": seed,
                "D_orient": float(np.sqrt(np.mean((a - b) ** 2))),
                "rho_axial": float(np.corrcoef(a[:, 2], b[:, 2])[0, 1]),
                "abs_delta_g_trans_final": float(abs(a[-1, :2].mean() - b[-1, :2].mean())),
                "abs_delta_g_axial_pre": float(abs(a[preclear_index, 2] - b[preclear_index, 2])),
                "abs_delta_g_axial_final": float(abs(a[-1, 2] - b[-1, 2])),
            }
        )
    layer2 = pd.DataFrame(layer2)

    # Layer 3: D_orient (mean over seeds) vs D_seed (within-orientation cross-seed).
    def cross_seed_rmse(q):
        ss = [s for (qq, s) in profiles if qq == q]
        vals = []
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                vals.append(np.sqrt(np.mean((profiles[(q, ss[i])] - profiles[(q, ss[j])]) ** 2)))
        return float(np.mean(vals)) if vals else float("nan")

    d_seed = float(np.mean([cross_seed_rmse("Q1"), cross_seed_rmse("Q2")]))
    d_orient = float(layer2.D_orient.mean())

    summary = {
        "layer1_law_pass": {
            "Q1": int(frame[(frame.orientation == "Q1")].law_pass.sum()),
            "Q2": int(frame[(frame.orientation == "Q2")].law_pass.sum()),
            "total": int(frame.law_pass.sum()),
            "n": int(len(frame)),
        },
        "layer2_matched_seed": layer2.to_dict(orient="records"),
        "layer3": {
            "D_orient_mean": d_orient,
            "D_seed_mean": d_seed,
            "D_orient_minus_D_seed": d_orient - d_seed,
        },
        "per_orientation_seed": frame.to_dict(orient="records"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out.with_name("rotated_per_seed_profiles.csv"), index=False)
    layer2.to_csv(args.out.with_name("rotated_matched_seed_diff.csv"), index=False)
    with args.out.with_name("rotated_invariance_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("=== Layer 1: structure law per (orientation, seed) ===")
    print(frame.to_string(index=False))
    print("\n=== Layer 2: matched-seed orientation difference ===")
    print(layer2.to_string(index=False))
    print(f"\n=== Layer 3 ===\nD_orient = {d_orient:.4f}, D_seed = {d_seed:.4f}, "
          f"D_orient - D_seed = {d_orient - d_seed:.4f}")
    print("saved:", args.out.with_name("rotated_invariance_summary.json"))


if __name__ == "__main__":
    main()
