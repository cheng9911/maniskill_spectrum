from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


MODEL_ORDER = ["Pdiag finite", "Full operator", "Generic RBF"]
COLORS = {
    "Pdiag finite": "#0072B2",
    "Full operator": "#D55E00",
    "Generic RBF": "#009E73",
}
MARKERS = {
    "Pdiag finite": "o",
    "Full operator": "s",
    "Generic RBF": "^",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_view(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose the common N=30 fit as the endpoint of both protocols."""
    common = frame[frame.protocol == "all"]
    return pd.concat(
        [
            frame[frame.protocol != "all"],
            common.assign(protocol="random"),
            common.assign(protocol="qualified"),
        ],
        ignore_index=True,
    )


def seed_aggregates(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        frame.groupby(["seed", "protocol", "sample_size", "model"], sort=False)[
            metric
        ]
        .mean()
        .reset_index()
    )


def hierarchical_paired_interval(
    frame: pd.DataFrame,
    competitor: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    indexed = frame.set_index(["seed", "subset_id", "model"])
    pairs = []
    for seed in sorted(frame.seed.unique()):
        seed_rows = frame[frame.seed == seed]
        subset_ids = sorted(seed_rows.subset_id.unique())
        differences = []
        for subset_id in subset_ids:
            try:
                pdiag = indexed.loc[
                    (seed, subset_id, "Pdiag finite"),
                    "task_error_mean_mm_equiv",
                ]
                baseline = indexed.loc[
                    (seed, subset_id, competitor),
                    "task_error_mean_mm_equiv",
                ]
            except KeyError:
                continue
            if np.isfinite(pdiag) and np.isfinite(baseline):
                differences.append(float(baseline - pdiag))
        if differences:
            pairs.append(np.asarray(differences))
    observed = float(np.mean(np.concatenate(pairs)))
    boot = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        sampled_seed_indices = rng.integers(0, len(pairs), len(pairs))
        values = []
        for seed_index in sampled_seed_indices:
            seed_values = pairs[seed_index]
            selected = rng.integers(0, len(seed_values), len(seed_values))
            values.extend(seed_values[selected])
        boot[draw] = np.mean(values)
    low, high = np.percentile(boot, [2.5, 97.5])
    return observed, float(low), float(high), int(sum(len(row) for row in pairs))


def plot_metric(ax, aggregate, protocol, metric, ylabel):
    rows = aggregate[aggregate.protocol == protocol]
    for model in MODEL_ORDER:
        model_rows = rows[rows.model == model]
        summary = model_rows.groupby("sample_size")[metric].agg(["mean", "std"])
        x = summary.index.to_numpy()
        mean = summary["mean"].to_numpy()
        sd = summary["std"].fillna(0).to_numpy()
        ax.plot(
            x,
            mean,
            marker=MARKERS[model],
            color=COLORS[model],
            linewidth=1.8,
            markersize=4.5,
            label=model,
        )
        ax.fill_between(x, mean - sd, mean + sd, color=COLORS[model], alpha=0.14)
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 8, 10, 15, 20, 30])
    ax.set_xticklabels([3, 5, 8, 10, 15, 20, 30])
    ax.set_xlabel("Mixed demonstrations")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{protocol.capitalize()} subsets")


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
        default=Path("phase_switch_symmetry_multiseed/fewshot_results"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    with args.experiment.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    with args.subsets.open(encoding="utf-8") as subset_file:
        subset_manifest = json.load(subset_file)
    seeds = [int(seed) for seed in experiment["seeds"]]

    result_frames = []
    error_frames = []
    audit_frames = []
    run_hashes = {}
    result_hashes = {}
    for seed in seeds:
        seed_root = args.experiment.parent / "fewshot" / f"seed_{seed}"
        run_path = seed_root / "fewshot_run.json"
        result_path = seed_root / "fewshot_results.csv"
        error_path = seed_root / "fewshot_condition_errors.csv"
        audit_path = seed_root / "fewshot_pdiag_optimization_audit.csv"
        with run_path.open(encoding="utf-8") as run_file:
            run = json.load(run_file)
        if run["subset_manifest_sha256"] != sha256(args.subsets):
            raise RuntimeError(f"seed {seed} used a different subset manifest")
        result_frames.append(pd.read_csv(result_path))
        error_frames.append(pd.read_csv(error_path))
        audit_frames.append(pd.read_csv(audit_path))
        run_hashes[str(seed)] = sha256(run_path)
        result_hashes[str(seed)] = sha256(result_path)

    raw = pd.concat(result_frames, ignore_index=True)
    condition_errors = pd.concat(error_frames, ignore_index=True)
    audits = pd.concat(audit_frames, ignore_index=True)
    view = protocol_view(raw)
    successful = view[view.fit_success].copy()
    metrics = [
        "task_error_mean_mm_equiv",
        "task_endpoint_error_mean_mm_equiv",
        "generator_rmse",
        "switch_absolute_error",
    ]
    seed_tables = []
    for metric in metrics:
        current = seed_aggregates(successful, metric)
        current["metric"] = metric
        current = current.rename(columns={metric: "value"})
        seed_tables.append(current)
    seed_summary = pd.concat(seed_tables, ignore_index=True)

    aggregate_rows = []
    for (protocol, sample_size, model, metric), rows in seed_summary.groupby(
        ["protocol", "sample_size", "model", "metric"], sort=False
    ):
        aggregate_rows.append(
            {
                "protocol": protocol,
                "sample_size": sample_size,
                "model": model,
                "metric": metric,
                "mean_over_seed_means": float(rows.value.mean()),
                "between_seed_sd": float(rows.value.std(ddof=1)),
                "seed_count": int(len(rows)),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)

    paired_rows = []
    rng = np.random.default_rng(20260818)
    for protocol in ["random", "qualified"]:
        for sample_size in subset_manifest["sample_sizes"]:
            selected = successful[
                (successful.protocol == protocol)
                & (successful.sample_size == sample_size)
            ]
            for competitor in ["Full operator", "Generic RBF"]:
                observed, low, high, pair_count = hierarchical_paired_interval(
                    selected, competitor, args.bootstrap_samples, rng
                )
                piv = selected.pivot_table(
                    index=["seed", "subset_id"],
                    columns="model",
                    values="task_error_mean_mm_equiv",
                    aggfunc="first",
                ).dropna(subset=["Pdiag finite", competitor])
                differences = piv[competitor] - piv["Pdiag finite"]
                paired_rows.append(
                    {
                        "protocol": protocol,
                        "sample_size": sample_size,
                        "competitor": competitor,
                        "competitor_minus_pdiag_mean": observed,
                        "hierarchical_bootstrap_95ci_low": low,
                        "hierarchical_bootstrap_95ci_high": high,
                        "pdiag_win_fraction": float((differences > 0).mean()),
                        "paired_fit_count": pair_count,
                    }
                )
    paired = pd.DataFrame(paired_rows)

    random_subset_metrics = (
        successful[successful.protocol == "random"]
        .groupby(["subset_id", "sample_size", "model"], sort=False)
        .agg(
            condition_number=("condition_number", "first"),
            task_error_mean_mm_equiv=("task_error_mean_mm_equiv", "mean"),
            generator_rmse=("generator_rmse", "mean"),
        )
        .reset_index()
    )
    correlation_rows = []
    for (sample_size, model), rows in random_subset_metrics.groupby(
        ["sample_size", "model"], sort=False
    ):
        for metric in ["task_error_mean_mm_equiv", "generator_rmse"]:
            if len(rows) < 3 or rows.condition_number.nunique() < 2:
                correlation, pvalue = np.nan, np.nan
            else:
                correlation, pvalue = spearmanr(
                    rows.condition_number, rows[metric]
                )
            correlation_rows.append(
                {
                    "sample_size": sample_size,
                    "model": model,
                    "metric": metric,
                    "spearman_rho": float(correlation),
                    "pvalue_descriptive": float(pvalue),
                    "subset_count": int(len(rows)),
                }
            )
    correlations = pd.DataFrame(correlation_rows)

    failures = raw[~raw.fit_success]
    switch_rates = (
        view.groupby(["protocol", "sample_size", "model"], sort=False)
        .switch_detected.mean()
        .rename("switch_detection_fraction")
        .reset_index()
    )
    law_pass = view.assign(
        law_pass=lambda frame: (
            (frame.g_translation_final > 0.90)
            & (frame.g_yaw_preclear > 0.90)
            & (frame.g_yaw_final.abs() < 0.10)
        )
    )
    law_rates = (
        law_pass.groupby(["protocol", "sample_size", "model"], sort=False)
        .law_pass.mean()
        .rename("generator_law_pass_fraction")
        .reset_index()
    )

    metric_lookup = seed_summary.pivot_table(
        index=["seed", "protocol", "sample_size", "model"],
        columns="metric",
        values="value",
    ).reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), constrained_layout=True)
    plot_metric(
        axes[0, 0],
        metric_lookup,
        "random",
        "task_error_mean_mm_equiv",
        "Task trajectory error (mm-equiv)",
    )
    plot_metric(
        axes[0, 1],
        metric_lookup,
        "qualified",
        "task_error_mean_mm_equiv",
        "Task trajectory error (mm-equiv)",
    )
    plot_metric(
        axes[1, 0],
        metric_lookup,
        "qualified",
        "generator_rmse",
        "Generator profile RMSE",
    )
    for competitor, linestyle in [
        ("Full operator", "-"),
        ("Generic RBF", "--"),
    ]:
        for protocol, marker in [("random", "o"), ("qualified", "s")]:
            rows = paired[
                (paired.competitor == competitor) & (paired.protocol == protocol)
            ]
            axes[1, 1].plot(
                rows.sample_size,
                rows.pdiag_win_fraction,
                color=COLORS[competitor],
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                label=f"vs {competitor}, {protocol}",
            )
    axes[1, 1].axhline(0.5, color="#666666", linewidth=1, linestyle=":")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xticks([3, 5, 8, 10, 15, 20, 30])
    axes[1, 1].set_xticklabels([3, 5, 8, 10, 15, 20, 30])
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].set_xlabel("Mixed demonstrations")
    axes[1, 1].set_ylabel("Pdiag paired win fraction")
    axes[1, 1].set_title("Matched subset comparisons")
    axes[1, 1].legend(frameon=False, fontsize=7)
    for label, ax in zip("ABCD", axes.flat):
        ax.text(
            -0.13,
            1.07,
            label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7)
    figure_png = args.output_root / "phase_switch_fewshot.png"
    figure_pdf = args.output_root / "phase_switch_fewshot.pdf"
    fig.savefig(figure_png, dpi=300, bbox_inches="tight")
    fig.savefig(figure_pdf, bbox_inches="tight")
    plt.close(fig)

    raw_path = args.output_root / "fewshot_all_fits.csv"
    errors_path = args.output_root / "fewshot_all_condition_errors.csv"
    seed_path = args.output_root / "fewshot_seed_summary.csv"
    aggregate_path = args.output_root / "fewshot_aggregate_summary.csv"
    paired_path = args.output_root / "fewshot_paired_comparisons.csv"
    correlation_path = args.output_root / "fewshot_condition_correlations.csv"
    switch_path = args.output_root / "fewshot_switch_rates.csv"
    law_path = args.output_root / "fewshot_law_recovery_rates.csv"
    audit_path = args.output_root / "fewshot_pdiag_optimization_audit.csv"
    raw.to_csv(raw_path, index=False)
    condition_errors.to_csv(errors_path, index=False)
    seed_summary.to_csv(seed_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    paired.to_csv(paired_path, index=False)
    correlations.to_csv(correlation_path, index=False)
    switch_rates.to_csv(switch_path, index=False)
    law_rates.to_csv(law_path, index=False)
    audits.to_csv(audit_path, index=False)

    n3_random = raw[(raw.protocol == "random") & (raw.sample_size == 3)]
    summary = {
        "schema_version": 1,
        "analysis_source_sha256": sha256(Path(__file__)),
        "experiment_sha256": sha256(args.experiment),
        "subset_manifest_sha256": sha256(args.subsets),
        "seeds": seeds,
        "run_sha256": run_hashes,
        "result_sha256": result_hashes,
        "fit_count": int(len(raw)),
        "fit_failure_count": int(len(failures)),
        "pdiag_optimization_audit": {
            "fit_count": int(len(audits)),
            "success_count": int(audits.optimization_success.sum()),
            "success_fraction": float(audits.optimization_success.mean()),
            "max_profile_reproduction_error": float(
                audits.profile_max_absolute_reproduction_error.max()
            ),
        },
        "all_failed_fits_retained": True,
        "uncertainty_definition": (
            "Figure bands are sample SD across five seed-level means after "
            "averaging frozen subsets; paired intervals resample seed then subset."
        ),
        "n3_identifiability": {
            "random_subset_count": int(n3_random.subset_id.nunique()),
            "rank3_fraction": float(
                n3_random.groupby("subset_id").first()["rank"].eq(3).mean()
            ),
            "augmented_intercept_rank4_fraction": float(
                n3_random.groupby("subset_id")
                .first()["augmented_intercept_rank"]
                .eq(4)
                .mean()
            ),
            "note": (
                "N=3 can excite the three generator columns but cannot identify "
                "an unconstrained affine intercept plus all three slopes."
            ),
        },
        "scope": (
            "Few-shot comparison uses the frozen finite Pdiag, full operator, "
            "and Generic RBF. TP-GMM SE(2) remains in the full-data benchmark "
            "and is not included in this computationally repeated sweep."
        ),
        "claim_boundary": (
            "Results test sample efficiency and stability; they do not establish "
            "Pdiag dominance unless the matched comparisons support it."
        ),
    }
    summary_path = args.output_root / "fewshot_summary.json"
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
