from __future__ import annotations

"""Visualize the rotated-axis pilot results and the failure-context map.

Panels:
  A: six canonical axial-yaw profiles alpha_axial(s) (one per orientation/seed).
  B: six canonical translation profiles (mean of u/v).
  C: matched-seed profile difference P_Q1 - P_Q2 (axial channel).
  D: D_orient vs D_seed (mean RMSE bars).
  E: 30 mixed x 6 file success/failure heatmap.

Also runs a light failure-context check: does missingness correlate with
|du|, |dv|, |dpsi|, or their interactions?
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyze_phase_switch_rotated import transform_curve
from benchmark_phase_switch_baselines import PHASE_CODES, progress_grid, usable
from phase_switch_baselines import SmoothFinitePDiagModel
from phase_switch_symmetry_env import SOCKET_CENTER

COLORS = {"Q1": "#0072B2", "Q2": "#D55E00"}


def fit_profiles(files, manifest, config, bins):
    progress, phase_codes = progress_grid(bins)
    profiles = {}
    rows = []
    for path in files:
        q = "Q1" if "_Q1_" in path.stem else "Q2"
        seed = int(path.stem.split("seed_")[-1])
        with h5py.File(path, "r") as f:
            orientation = np.asarray(f.attrs["orientation"], dtype=np.float64)
            task_anchor = np.asarray(f.attrs["task_anchor"], dtype=np.float64)
            mixed_ids = [c["condition_id"] for c in manifest["conditions"]
                         if c["generator"] == "mixed"]
            contexts = np.asarray([manifest["conditions"][c]["causal_delta"] for c in mixed_ids])

            def curve(cid):
                for k in f:
                    if int(f[k].attrs.get("condition_id", -1)) == cid and usable(f[k]):
                        return transform_curve(f[k], bins, orientation, task_anchor, SOCKET_CENTER)
                return None

            ok_ids = [c for c in mixed_ids if curve(c) is not None]
            ok_contexts = np.asarray([manifest["conditions"][c]["causal_delta"] for c in ok_ids])
            ok_curves = np.asarray([curve(c) for c in ok_ids])
            pdiag = SmoothFinitePDiagModel(
                nominal_frame_xy=SOCKET_CENTER[:2].copy(), nominal_frame_yaw=0.0, **config
            ).fit(ok_contexts, ok_curves, progress, phase_codes)
            profile = pdiag.jacobian_diag()
            profiles[(q, seed)] = profile
            rows.append({
                "orientation": q, "seed": seed,
                "n_mixed_used": len(ok_ids),
                "g_translation_final": float(profile[-1, :2].mean()),
                "g_axial_preclear": float(profile[2 * bins - 1, 2]),
                "g_axial_final": float(profile[-1, 2]),
            })
    return profiles, pd.DataFrame(rows), progress


def failure_map(files, manifest):
    """success/failure per (orientation, seed, mixed condition)."""
    mixed = [c for c in manifest["conditions"] if c["generator"] == "mixed"]
    records = []
    for path in files:
        q = "Q1" if "_Q1_" in path.stem else "Q2"
        seed = int(path.stem.split("seed_")[-1])
        with h5py.File(path, "r") as f:
            succ = {}
            for k in f:
                g = f[k]
                cid = int(g.attrs.get("condition_id", -1))
                ok = bool(np.asarray(g["success"])[-1])
                succ[cid] = succ.get(cid, False) or ok
        for c in mixed:
            cid = c["condition_id"]
            du, dv, dp = np.abs(np.asarray(c["causal_delta"]))
            records.append({
                "orientation": q, "seed": seed, "condition_id": cid,
                "success": succ.get(cid, False),
                "abs_du": du, "abs_dv": dv, "abs_dpsi": dp,
                "abs_du_dpsi": du * dp, "abs_dv_dpsi": dv * dp,
                "radial": np.hypot(du, dv),
            })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/rotated_axis_experiment.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/fixed_contexts.json"),
    )
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument(
        "--out", type=Path,
        default=Path("phase_switch_symmetry_rollouts_rotated/rotated_pilot_figure"),
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
    with args.manifest.open() as f:
        manifest = json.load(f)

    profiles, frame, progress = fit_profiles(args.files, manifest, config, args.bins)

    # Failure-context analysis.
    fm = failure_map(args.files, manifest)
    failure = (~fm.success).astype(float)
    print("=== failure-context association (Spearman, failure vs |intervention|) ===")
    for col in ["abs_du", "abs_dv", "abs_dpsi", "abs_du_dpsi", "abs_dv_dpsi", "radial"]:
        rho, p = spearmanr(failure, fm[col])
        print(f"  {col:14s} rho={rho:+.3f} p={p:.3f}")
    print(f"\nfailure rate: {failure.mean():.3f} ({int(failure.sum())}/{len(failure)})")
    by_orient = fm.groupby("orientation").success.mean()
    print("success rate by orientation:\n", by_orient.to_string())

    # Plot.
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "legend.fontsize": 6.5, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.0), constrained_layout=True)
    for (q, seed), profile in profiles.items():
        axes[0, 0].plot(progress, profile[:, 2], color=COLORS[q], alpha=0.8,
                        label=f"{q}/{seed}")
    axes[0, 0].axhline(0.5, color="0.7", lw=0.7, ls=":")
    axes[0, 0].set_title("A  axial $\\alpha_{axial}(s)$", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("progress $s$")

    for (q, seed), profile in profiles.items():
        axes[0, 1].plot(progress, profile[:, :2].mean(axis=1), color=COLORS[q],
                        alpha=0.8, label=f"{q}/{seed}")
    axes[0, 1].set_title("B  translation $\\bar\\alpha_{u/v}(s)$", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("progress $s$")

    seeds = sorted({s for q, s in profiles if q == "Q1"} & {s for q, s in profiles if q == "Q2"})
    for seed in seeds:
        d = profiles[("Q1", seed)] - profiles[("Q2", seed)]
        axes[0, 2].plot(progress, d[:, 2], label=f"seed {seed}", alpha=0.8)
    axes[0, 2].axhline(0.0, color="0.7", lw=0.7)
    axes[0, 2].set_title("C  $\\alpha_{axial}^{Q1}-\\alpha_{axial}^{Q2}$", loc="left", fontweight="bold")
    axes[0, 2].set_xlabel("progress $s$")

    d_orient = []
    d_seed_vals = []
    for seed in seeds:
        d_orient.append(np.sqrt(np.mean((profiles[("Q1", seed)] - profiles[("Q2", seed)]) ** 2)))
    for q in ["Q1", "Q2"]:
        ss = [s for qq, s in profiles if qq == q]
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                d_seed_vals.append(np.sqrt(np.mean((profiles[(q, ss[i])] - profiles[(q, ss[j])]) ** 2)))
    axes[1, 0].bar([0, 1], [np.mean(d_orient), np.mean(d_seed_vals)],
                   color=["#D55E00", "#888888"], width=0.5)
    axes[1, 0].set_xticks([0, 1], ["$D_{orient}$", "$D_{seed}$"])
    axes[1, 0].set_ylabel("profile RMSE")
    axes[1, 0].set_title("D  orientation vs seed variation", loc="left", fontweight="bold")

    # failure heatmap: 30 mixed x 6 files
    hm = fm.pivot_table(index="condition_id", columns=["orientation", "seed"],
                        values="success", aggfunc="first")
    axes[1, 1].imshow(hm.to_numpy().astype(float), aspect="auto", cmap="RdYlGn",
                      vmin=0, vmax=1)
    axes[1, 1].set_xlabel("file (Q/seed)")
    axes[1, 1].set_ylabel("mixed condition id")
    axes[1, 1].set_title("E  success/failure map", loc="left", fontweight="bold")

    frame_f = frame.set_index(["orientation", "seed"])
    axes[1, 2].axis("off")
    axes[1, 2].text(0.05, 0.95, frame_f.to_string(), fontsize=6, va="top",
                    family="monospace")
    axes[1, 2].set_title("F  per-seed metrics", loc="left", fontweight="bold")

    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved:", args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
