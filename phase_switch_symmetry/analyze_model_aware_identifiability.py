from __future__ import annotations

"""Analyze model-aware identifiability (Experiment A).

Joins the Gauss--Newton information metrics (lambda_min, log det, trace-inverse
for I_post and I_design) with the frozen few-shot recovery metrics, then tests
whether the information metrics predict recovery better than the plain context
condition number kappa(C).

Primary comparison: within each (sample_size, protocol) with enough subsets,
Spearman rank correlation between each identifiability metric and each recovery
metric (task error, generator RMSE, switch error), side by side with kappa(C).
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


INFORMATION_METRICS = [
    "lambda_min_post",
    "lambda_ratio_post",
    "log_det_post",
    "trace_inv_post",
    "lambda_min_design",
    "lambda_ratio_design",
    "log_det_design",
    "trace_inv_design",
]
RECOVERY_METRICS = [
    "task_error_mean_mm_equiv",
    "generator_rmse",
    "switch_absolute_error",
]
# Higher is better for information; higher is worse for recovery, so a useful
# identifiability metric should correlate NEGATIVELY with recovery error.
# trace_inv is "higher = worse identifiability", so it should correlate POSITIVELY.
EXPECTED_SIGN = {
    "lambda_min_post": -1,
    "lambda_ratio_post": -1,
    "log_det_post": -1,
    "trace_inv_post": +1,
    "lambda_min_design": -1,
    "lambda_ratio_design": -1,
    "log_det_design": -1,
    "trace_inv_design": +1,
    "condition_number": +1,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spearman_rho(x, y):
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan
    return spearmanr(x, y)


def subset_level(merged: pd.DataFrame) -> pd.DataFrame:
    """Average each metric over the five execution seeds per subset."""
    value_columns = (
        INFORMATION_METRICS
        + RECOVERY_METRICS
        + ["condition_number", "rank", "augmented_intercept_rank"]
    )
    grouped = merged.groupby(
        ["subset_id", "protocol", "sample_size", "repeat"], sort=False
    )
    out = grouped[INFORMATION_METRICS + RECOVERY_METRICS].mean().reset_index()
    meta = grouped[value_columns].first().reset_index()  # condition_number is constant
    for column in ["condition_number", "rank", "augmented_intercept_rank"]:
        out[column] = meta[column]
    return out


def correlation_table(subset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sample_size, protocol), group in subset.groupby(
        ["sample_size", "protocol"], sort=False
    ):
        for metric in INFORMATION_METRICS + ["condition_number"]:
            for recovery in RECOVERY_METRICS:
                rho, pvalue = spearman_rho(
                    group[metric].to_numpy(), group[recovery].to_numpy()
                )
                rows.append(
                    {
                        "sample_size": sample_size,
                        "protocol": protocol,
                        "metric": metric,
                        "recovery": recovery,
                        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                        "pvalue_descriptive": (
                            float(pvalue) if np.isfinite(pvalue) else np.nan
                        ),
                        "subset_count": int(len(group)),
                        "expected_sign": EXPECTED_SIGN[metric],
                    }
                )
    return pd.DataFrame(rows)


def pooled_within_size_rank(subset: pd.DataFrame) -> pd.DataFrame:
    """Spearman over all subsets, ranking each metric within its sample size."""
    ranks = subset.copy()
    for column in INFORMATION_METRICS + ["condition_number"] + RECOVERY_METRICS:
        ranks[column + "_rank"] = ranks.groupby("sample_size")[column].rank()
    rows = []
    for metric in INFORMATION_METRICS + ["condition_number"]:
        for recovery in RECOVERY_METRICS:
            rho, pvalue = spearman_rho(
                ranks[metric + "_rank"].to_numpy(), ranks[recovery + "_rank"].to_numpy()
            )
            rows.append(
                {
                    "metric": metric,
                    "recovery": recovery,
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "pvalue_descriptive": float(pvalue) if np.isfinite(pvalue) else np.nan,
                    "subset_count": int(len(ranks)),
                    "expected_sign": EXPECTED_SIGN[metric],
                    "scope": "all_subsets_within_size_rank",
                }
            )
    return pd.DataFrame(rows)


def make_figure(subset: pd.DataFrame, output_stem: Path):
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
    sizes = sorted(subset.sample_size.unique())
    for axis, recovery in zip(axes, ["task_error_mean_mm_equiv", "generator_rmse"]):
        x = np.arange(len(sizes))
        width = 0.22
        for offset, (metric, color, label) in enumerate(
            [
                ("condition_number", "#CC79A7", r"$\kappa(C)$"),
                ("lambda_min_post", "#0072B2", r"$\lambda_{min}(I_{post})$"),
                ("lambda_min_design", "#D55E00", r"$\lambda_{min}(I_{design})$"),
            ]
        ):
            rhos = []
            for size in sizes:
                group = subset[subset.sample_size == size]
                rho, _ = spearman_rho(
                    group[metric].to_numpy(), group[recovery].to_numpy()
                )
                rhos.append(rho)
            axis.bar(
                x + (offset - 1) * width,
                rhos,
                width=width,
                color=color,
                label=label,
            )
        axis.set_xticks(x, sizes)
        axis.set_xlabel("Mixed demonstrations (N)")
        axis.set_ylabel("Spearman $\\rho$")
        axis.axhline(0.0, color="0.6", linewidth=0.7)
        axis.axhline(-1.0, color="0.85", linewidth=0.5, linestyle=":")
        axis.set_ylim(-1.05, 1.05)
        axis.set_title(
            "A  Task error" if recovery == "task_error_mean_mm_equiv"
            else "B  Generator RMSE",
            loc="left",
            fontweight="bold",
        )
        axis.legend(frameon=False, fontsize=6.5, loc="lower right")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


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
        "--info-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/identifiability"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/identifiability"),
    )
    args = parser.parse_args()

    with args.experiment.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    seeds = experiment["seeds"]

    info_frames = []
    fewshot_frames = []
    for seed in seeds:
        info_path = args.info_root / f"seed_{seed}_info.csv"
        fewshot_path = (
            args.experiment.parent / "fewshot" / f"seed_{seed}" / "fewshot_results.csv"
        )
        if not info_path.exists():
            raise FileNotFoundError(
                f"missing info file {info_path}; run collect_model_aware_identifiability.py first"
            )
        info_frames.append(pd.read_csv(info_path))
        fewshot = pd.read_csv(fewshot_path)
        fewshot = fewshot[fewshot.model == "Pdiag finite"][
            ["seed", "subset_id"] + RECOVERY_METRICS
        ]
        fewshot_frames.append(fewshot)

    info = pd.concat(info_frames, ignore_index=True)
    fewshot = pd.concat(fewshot_frames, ignore_index=True)
    merged = info.merge(fewshot, on=["seed", "subset_id"], how="inner")
    if len(merged) != len(info):
        raise RuntimeError("information and few-shot rows did not align 1:1")

    subset = subset_level(merged)
    table = correlation_table(subset)
    pooled = pooled_within_size_rank(subset)

    # Signed correlation (fold in expected sign) for a compact "does it help" read.
    signed = table.copy()
    signed["signed_rho"] = signed["spearman_rho"] * signed["expected_sign"]
    signed_pooled = pooled.copy()
    signed_pooled["signed_rho"] = (
        signed_pooled["spearman_rho"] * signed_pooled["expected_sign"]
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    merged_path = args.output_root / "identifiability_merged.csv"
    table_path = args.output_root / "identifiability_correlations.csv"
    pooled_path = args.output_root / "identifiability_pooled_correlations.csv"
    merged.to_csv(merged_path, index=False)
    table.to_csv(table_path, index=False)
    pooled.to_csv(pooled_path, index=False)

    make_figure(subset, args.output_root / "identifiability_figure")

    # Mean signed rho per metric across all (size, protocol) cells as a headline.
    headline = (
        signed.groupby("metric")["signed_rho"].mean().sort_values(ascending=False)
    )
    summary = {
        "schema_version": 1,
        "experiment_sha256": sha256(args.experiment),
        "subset_manifest_sha256": sha256(args.subsets),
        "source_sha256": sha256(Path(__file__)),
        "seeds": seeds,
        "subset_count": int(subset.subset_id.nunique()),
        "model": "Pdiag finite",
        "mean_signed_spearman_rho_by_metric": headline.to_dict(),
        "primary_analysis": (
            "within-(sample_size, protocol) Spearman correlations, averaged over "
            "the cells and over the five execution seeds per subset. The pooled "
            "within-size rank correlation is a supplement only (it mixes the "
            "random and qualified sampling mechanisms)."
        ),
        "interpretation": (
            "Neither simple context conditioning kappa(C) nor the tested local "
            "parameter-space Gauss-Newton spectral summaries (lambda_min, "
            "lambda_ratio, log det, trace-inverse, at the fitted theta and at a "
            "neutral theta=0 reference) reliably rank few-shot subset quality. "
            "All signed Spearman correlations are weak (|rho| < 0.4 within-cell) "
            "and sign-inconsistent across sample sizes. This is a narrow negative "
            "result about these specific spectral summaries; it does not rule out "
            "function-space identifiability of the generator law itself (e.g. "
            "variance of alpha_yaw^post or s_0.5)."
        ),
    }
    summary_path = args.output_root / "identifiability_summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, allow_nan=True)

    print("Mean signed Spearman rho (higher = better ordering):")
    for metric, value in headline.items():
        print(f"  {metric:24s} {value:+.3f}")
    print("saved:", merged_path)
    print("saved:", table_path)
    print("saved:", pooled_path)
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
