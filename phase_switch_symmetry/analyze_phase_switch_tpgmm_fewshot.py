from __future__ import annotations

"""Analyze TP-GMM SE(2) matched few-shot (Experiment B).

Joins the new TP-GMM SE(2) few-shot fits with the existing frozen few-shot
results (Pdiag finite / Full operator / Generic RBF) on the SAME matched
subsets and seeds, then reports paired sample-efficiency comparisons.

Primary output: does the formal rotation-aware baseline retain its full-data
anisotropic recovery under few demonstrations, and how does it compare to
Pdiag finite on matched subsets?
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ["Pdiag finite", "TP-GMM SE(2)", "Full operator", "Generic RBF"]
COLORS = {
    "Pdiag finite": "#0072B2",
    "TP-GMM SE(2)": "#D55E00",
    "Full operator": "#56B4E9",
    "Generic RBF": "#009E73",
}
MARKERS = {
    "Pdiag finite": "o",
    "TP-GMM SE(2)": "s",
    "Full operator": "^",
    "Generic RBF": "D",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/experiment.json"),
    )
    parser.add_argument(
        "--subsets",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/fewshot_subsets.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/tpgmm_fewshot"),
    )
    args = parser.parse_args()

    with args.experiment.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    seeds = experiment["seeds"]

    fewshot_frames = []
    tpgmm_frames = []
    for seed in seeds:
        fewshot_path = (
            args.experiment.parent / "fewshot" / f"seed_{seed}" / "fewshot_results.csv"
        )
        tpgmm_path = (
            args.experiment.parent / "tpgmm_fewshot" / f"seed_{seed}" / "tpgmm_fewshot_results.csv"
        )
        if not tpgmm_path.exists():
            raise FileNotFoundError(
                f"missing {tpgmm_path}; run run_phase_switch_tpgmm_fewshot.py first"
            )
        fewshot = pd.read_csv(fewshot_path)
        tpgmm = pd.read_csv(tpgmm_path)
        selected_ids = set(tpgmm.subset_id.unique())
        fewshot = fewshot[fewshot.subset_id.isin(selected_ids)]
        fewshot_frames.append(fewshot)
        tpgmm_frames.append(tpgmm)

    fewshot = pd.concat(fewshot_frames, ignore_index=True)
    tpgmm = pd.concat(tpgmm_frames, ignore_index=True)
    metric = "task_error_mean_mm_equiv"

    combined = pd.concat(
        [
            fewshot[["seed", "subset_id", "model", metric, "fit_success"]],
            tpgmm[["seed", "subset_id", "model", metric, "fit_success"]],
        ],
        ignore_index=True,
    )
    meta = tpgmm[
        ["subset_id", "sample_size", "protocol", "repeat"]
    ].drop_duplicates("subset_id")
    combined = combined.merge(meta, on="subset_id", how="left")
    combined = combined[combined.fit_success]

    # Seed-mean over subset level.
    subset_means = (
        combined.groupby(["subset_id", "sample_size", "protocol", "model"], sort=False)
        .agg(task_error_mean_mm_equiv=(metric, "mean"))
        .reset_index()
    )

    summary_rows = []
    for (sample_size, model), rows in subset_means.groupby(
        ["sample_size", "model"], sort=False
    ):
        summary_rows.append(
            {
                "sample_size": sample_size,
                "model": model,
                "task_error_mean_over_subset_means": float(
                    rows.task_error_mean_mm_equiv.mean()
                ),
                "between_subset_sd": float(rows.task_error_mean_mm_equiv.std(ddof=1)),
                "subset_count": int(len(rows)),
            }
        )
    summary = pd.DataFrame(summary_rows)

    # Paired (TP-GMM - Pdiag) per (seed, subset), on matched subsets.
    pivot = combined.pivot_table(
        index=["seed", "subset_id"], columns="model", values=metric, aggfunc="first"
    )
    paired_rows = []
    for (seed, subset_id), row in pivot.iterrows():
        if (
            "Pdiag finite" not in row
            or "TP-GMM SE(2)" not in row
            or not np.isfinite(row["Pdiag finite"])
            or not np.isfinite(row["TP-GMM SE(2)"])
        ):
            continue
        diff = row["TP-GMM SE(2)"] - row["Pdiag finite"]
        paired_rows.append(
            {
                "seed": seed,
                "subset_id": subset_id,
                "tpgmm_minus_pdiag_mm_equiv": float(diff),
            }
        )
    paired = pd.DataFrame(paired_rows).merge(
        meta, on="subset_id", how="left"
    )
    win_fraction = (
        paired.groupby("sample_size")
        .agg(
            pdiag_win_fraction=("tpgmm_minus_pdiag_mm_equiv", lambda x: float((x > 0).mean())),
            paired_count=("tpgmm_minus_pdiag_mm_equiv", "count"),
            mean_diff=("tpgmm_minus_pdiag_mm_equiv", "mean"),
        )
        .reset_index()
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_root / "tpgmm_fewshot_summary.csv", index=False)
    paired.to_csv(args.output_root / "tpgmm_fewshot_paired.csv", index=False)
    win_fraction.to_csv(args.output_root / "tpgmm_fewshot_win_fraction.csv", index=False)

    # Figure: sample-efficiency curves for all models on the matched subsets.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    sizes = sorted(subset_means.sample_size.unique())
    for model in MODELS:
        rows = subset_means[subset_means.model == model]
        means = []
        sds = []
        for size in sizes:
            sub = rows[rows.sample_size == size]
            means.append(sub.task_error_mean_mm_equiv.mean())
            sds.append(sub.task_error_mean_mm_equiv.std(ddof=1))
        means = np.asarray(means)
        sds = np.asarray(sds)
        axes[0].plot(
            sizes,
            means,
            marker=MARKERS[model],
            color=COLORS[model],
            linewidth=1.8,
            markersize=4.5,
            label=model,
        )
        axes[0].fill_between(sizes, means - sds, means + sds, color=COLORS[model], alpha=0.14)
    axes[0].set_xscale("log")
    axes[0].set_xticks(sizes, sizes)
    axes[0].set_xlabel("Mixed demonstrations")
    axes[0].set_ylabel("Task trajectory error (mm-equiv)")
    axes[0].set_title("A  Matched sample efficiency", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=7)

    axes[1].plot(
        win_fraction.sample_size,
        win_fraction.pdiag_win_fraction,
        marker="o",
        color="#0072B2",
        linewidth=1.8,
    )
    axes[1].axhline(0.5, color="0.6", linewidth=0.8, linestyle=":")
    axes[1].set_xscale("log")
    axes[1].set_xticks(sizes, sizes)
    axes[1].set_xlabel("Mixed demonstrations")
    axes[1].set_ylabel("Pdiag win fraction vs TP-GMM SE(2)")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title(
        "B  Pdiag vs TP-GMM SE(2), matched", loc="left", fontweight="bold"
    )
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.savefig(args.output_root / "tpgmm_fewshot_figure.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_root / "tpgmm_fewshot_figure.pdf", bbox_inches="tight")
    plt.close(fig)

    result = {
        "schema_version": 1,
        "experiment_sha256": sha256(args.experiment),
        "subset_manifest_sha256": sha256(args.subsets),
        "source_sha256": sha256(Path(__file__)),
        "seeds": seeds,
        "models": MODELS,
        "matched_subset_count": int(subset_means.subset_id.nunique()),
        "summary": summary.to_dict(orient="records"),
        "win_fraction": win_fraction.to_dict(orient="records"),
        "note": (
            "Paired comparison uses only subsets that both TP-GMM SE(2) and "
            "Pdiag finite fitted successfully on the same seed."
        ),
    }
    with (args.output_root / "tpgmm_fewshot_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(summary.to_string(index=False))
    print("\nPdiag win fraction vs TP-GMM SE(2):")
    print(win_fraction.to_string(index=False))


if __name__ == "__main__":
    main()
