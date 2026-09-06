from __future__ import annotations

"""Controller/planner/speed-invariance benchmark.

For a set of agent variants (different PD stiffness/damping, planner, or
trajectory speed) collected on the SAME keyed fixture, this script extracts the
empirical isolated yaw-relevance profile  alpha_yaw(s) = diagonal[:, 2] for each
variant and measures how strongly the profiles agree:

* Pearson  rho(alpha_yaw^A, alpha_yaw^B)  over the full progress axis;
* the same rho split per phase;
* the 0.5 downward-crossing switch location  s_0.5  (half-decay of yaw relevance)
  for each variant, plus its spread across variants.

High rho and a tight switch-location spread are the claimed evidence that the
identified phase-dependent relevance structure is a property of the task's
geometric relation, not of the particular demonstrator/controller used to roll
it out.  Translation relevance is reported as a secondary control.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_phase_switch_rollouts import episode_keys
from benchmark_phase_switch_baselines import (
    PHASE_CODES,
    PHASE_LABELS,
    empirical_isolated_profile,
    progress_grid,
    switch_diagnostics,
    usable,
)


def load_variant(path: Path, bins: int):
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
        ]
        if len(baseline) != 1:
            raise RuntimeError(f"{path}: expected one zero-intervention baseline, got {len(baseline)}")
        isolated = [key for key in isolated_all if key != baseline[0]]
        if len(isolated) != 8:
            raise RuntimeError(f"{path}: expected 8 isolated interventions, got {len(isolated)}")
        diagonal = empirical_isolated_profile(data_file, baseline[0], isolated, bins)
        metadata = {
            "arm_stiffness": float(data_file.attrs.get("arm_stiffness", float("nan"))),
            "arm_damping": float(data_file.attrs.get("arm_damping", float("nan"))),
            "planner_mode": str(data_file.attrs.get("planner_mode", "")),
            "speed": float(data_file.attrs.get("speed", float("nan"))),
            "seed": int(data_file.attrs.get("seed", -1)),
        }
    return diagonal, metadata


def pearson_matrix(profiles):
    profiles = np.asarray(profiles, dtype=np.float64)  # (V, n)
    centered = profiles - profiles.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return (centered @ centered.T) / (norms @ norms.T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", required=True, metavar="LABEL:PATH")
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument(
        "--output-root", type=Path, default=Path("phase_switch_symmetry_invariance")
    )
    args = parser.parse_args()

    variants = []
    for spec in args.variant:
        label, path = spec.split(":", 1)
        variants.append((label, Path(path)))

    progress, phase_codes = progress_grid(args.bins)
    n_phase = len(PHASE_CODES)
    per_phase_mask = [phase_codes == index for index in range(n_phase)]

    alpha_yaw = {}
    alpha_trans = {}
    metadata = {}
    switch = {}
    for label, path in variants:
        diagonal, meta = load_variant(path, args.bins)
        alpha_yaw[label] = diagonal[:, 2]
        alpha_trans[label] = 0.5 * (diagonal[:, 0] + diagonal[:, 1])
        metadata[label] = {"path": str(path), **meta}
        diag = switch_diagnostics(progress, diagonal[:, 2], unlock_start=2 * args.bins)
        switch[label] = diag

    labels = [label for label, _ in variants]
    yaw_matrix = np.asarray([alpha_yaw[label] for label in labels])
    trans_matrix = np.asarray([alpha_trans[label] for label in labels])

    rho_yaw = pearson_matrix(yaw_matrix)
    rho_trans = pearson_matrix(trans_matrix)

    rho_yaw_phase = np.empty((n_phase, len(labels), len(labels)))
    rho_trans_phase = np.empty((n_phase, len(labels), len(labels)))
    for phase_index in range(n_phase):
        mask = per_phase_mask[phase_index]
        rho_yaw_phase[phase_index] = pearson_matrix(yaw_matrix[:, mask])
        rho_trans_phase[phase_index] = pearson_matrix(trans_matrix[:, mask])

    switch_locations = {
        label: switch[label]["location"]
        for label in labels
    }
    switch_vals = np.asarray([switch[label]["location"] for label in labels])
    switch_spread = float(np.nanmax(switch_vals) - np.nanmin(switch_vals))

    # Robust per-phase agreement: Pearson rho is degenerate on near-flat
    # segments (zero variance), so also report the pooled std of alpha_yaw
    # (to flag such phases) and the pairwise max-abs deviation (L-infinity),
    # which is well-defined even when the profile is constant.
    phase_yaw_std = np.asarray(
        [yaw_matrix[:, mask].std() for mask in per_phase_mask]
    )
    phase_linf = np.empty(n_phase)
    for phase_index in range(n_phase):
        block = yaw_matrix[:, per_phase_mask[phase_index]]
        phase_linf[phase_index] = np.max(
            np.abs(block[:, None, :] - block[None, :, :])
        )

    args.output_root.mkdir(parents=True, exist_ok=True)

    # Full-profile rho matrices.
    np.savetxt(
        args.output_root / "alpha_yaw_pearson_full.csv",
        rho_yaw,
        delimiter=",",
        header=",".join(labels),
        comments="",
    )
    np.savetxt(
        args.output_root / "alpha_trans_pearson_full.csv",
        rho_trans,
        delimiter=",",
        header=",".join(labels),
        comments="",
    )

    # Per-phase rho, long-form CSV.
    rows = []
    for phase_index in range(n_phase):
        for i, label_a in enumerate(labels):
            for j, label_b in enumerate(labels):
                rows.append(
                    dict(
                        phase=PHASE_LABELS[phase_index],
                        variant_a=label_a,
                        variant_b=label_b,
                        rho_yaw=float(rho_yaw_phase[phase_index, i, j]),
                        rho_trans=float(rho_trans_phase[phase_index, i, j]),
                    )
                )
    import pandas as pd

    pd.DataFrame(rows).to_csv(args.output_root / "alpha_pearson_per_phase.csv", index=False)

    summary = {
        "schema_version": 1,
        "title": "Agent-variant invariance of the empirical isolated relevance profile",
        "bins_per_phase": args.bins,
        "variants": metadata,
        "profile_keys": {
            "yaw": "alpha_yaw(s) = diagonal[:,2] of empirical_isolated_profile",
            "translation": "0.5*(diagonal[:,0]+diagonal[:,1])",
        },
        "pearson_yaw_full": {a: {b: float(rho_yaw[i, j]) for j, b in enumerate(labels)} for i, a in enumerate(labels)},
        "pearson_translation_full": {a: {b: float(rho_trans[i, j]) for j, b in enumerate(labels)} for i, a in enumerate(labels)},
        "pearson_yaw_per_phase": {
            PHASE_LABELS[p]: {
                a: {b: float(rho_yaw_phase[p, i, j]) for j, b in enumerate(labels)}
                for i, a in enumerate(labels)
            }
            for p in range(n_phase)
        },
        "switch_locations_s_0_5": switch_locations,
        "switch_status": {label: switch[label]["status"] for label in labels},
        "switch_location_spread": switch_spread,
        "per_phase_yaw_std": {
            PHASE_LABELS[p]: float(phase_yaw_std[p]) for p in range(n_phase)
        },
        "per_phase_yaw_max_abs_deviation": {
            PHASE_LABELS[p]: float(phase_linf[p]) for p in range(n_phase)
        },
        "summary_yaw_pairwise_rho": {
            "min": float(np.min(rho_yaw)),
            "mean": float(np.mean(rho_yaw[np.triu_indices(len(labels), k=1)])),
            "diagonal": float(np.mean(np.diag(rho_yaw))),
        },
    }
    with (args.output_root / "controller_invariance_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Figure: profile overlay + full-profile rho heatmap.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), constrained_layout=True)
    for label in labels:
        axes[0].plot(progress, alpha_yaw[label], label=label, linewidth=1.4)
    for boundary in [0.25, 0.5, 0.75]:
        axes[0].axvline(boundary, color="0.82", linewidth=0.7)
    axes[0].axhline(0.5, color="0.75", linewidth=0.7, linestyle="--")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(-0.2, 1.3)
    axes[0].set_xticks([0.125, 0.375, 0.625, 0.875], ["Align", "Enter", "Unlock", "Insert"])
    axes[0].set_ylabel(r"$\alpha_{yaw}(s)$")
    axes[0].set_title("A  Yaw-relevance profiles", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=6.5)

    image = axes[1].imshow(rho_yaw, vmin=0.0, vmax=1.0, cmap="viridis")
    axes[1].set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axes[1].set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[1].text(j, i, f"{rho_yaw[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                         color="white" if rho_yaw[i, j] < 0.6 else "black")
    axes[1].set_title(r"B  $\rho(\alpha^A_{yaw}, \alpha^B_{yaw})$", loc="left", fontweight="bold")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    for ext in (".png", ".pdf"):
        fig.savefig(args.output_root / f"controller_invariance_figure{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("=== alpha_yaw switch locations (s_0.5) ===")
    for label in labels:
        print(f"  {label:8s} {switch[label]['status']:16s} s_0.5={switch_locations[label]:.5f}")
    print(f"  spread={switch_spread:.5f}")
    print()
    print("=== rho(alpha_yaw^A, alpha_yaw^B), full profile ===")
    print("        " + "  ".join(f"{label:>8s}" for label in labels))
    for i, label_a in enumerate(labels):
        print(f"{label_a:8s}" + "  ".join(f"{rho_yaw[i, j]:8.4f}" for j in range(len(labels))))
    print()
    print("=== rho per phase (yaw) ===")
    for phase_index in range(n_phase):
        phase_label = PHASE_LABELS[phase_index]
        off_diag = [
            rho_yaw_phase[phase_index, i, j]
            for i in range(len(labels))
            for j in range(i + 1, len(labels))
        ]
        print(
            f"  {phase_label:16s} rho_min={min(off_diag):.4f} rho_mean={np.mean(off_diag):.4f}"
        )
    print()
    print("=== per-phase robust agreement (yaw) ===")
    print("  phase                 std(alpha_yaw)   max_abs_deviation")
    for phase_index in range(n_phase):
        phase_label = PHASE_LABELS[phase_index]
        print(
            f"  {phase_label:16s} {phase_yaw_std[phase_index]:14.4f} {phase_linf[phase_index]:18.5f}"
        )
    print()
    print("saved:", args.output_root / "controller_invariance_summary.json")


if __name__ == "__main__":
    main()
