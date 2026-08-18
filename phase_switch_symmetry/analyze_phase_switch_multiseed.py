from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "Frame-weighted",
    "Phase scalar GP",
    "TP-GMM additive",
    "TP-GMM SE(2)",
    "Generic RBF",
    "Full operator",
    "Pdiag pointwise",
    "Pdiag finite",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hierarchical_bootstrap_improvement(errors, samples=20000, seed=20260818):
    rng = np.random.default_rng(seed)
    seed_values = sorted(errors.seed.unique())
    improvements = []
    for seed_value in seed_values:
        seed_rows = errors[errors.seed == seed_value]
        scalar = seed_rows[seed_rows.model == "Frame-weighted"].set_index("episode")
        pdiag = seed_rows[seed_rows.model == "Pdiag finite"].set_index("episode")
        values = (
            scalar.loc[pdiag.index, "task_error_mm_equiv"]
            - pdiag["task_error_mm_equiv"]
        ).to_numpy()
        improvements.append(values)
    improvements = np.asarray(improvements)
    boot = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_seeds = rng.integers(0, len(seed_values), len(seed_values))
        values = []
        for sampled_seed in sampled_seeds:
            condition_indices = rng.integers(
                0, improvements.shape[1], improvements.shape[1]
            )
            values.extend(improvements[sampled_seed, condition_indices])
        boot[index] = np.mean(values)
    return np.percentile(boot, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/experiment.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/results"),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    with args.experiment.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    seeds = [int(seed) for seed in experiment["seeds"]]
    benchmark_root = args.experiment.parent / "benchmarks"
    rollout_root = args.experiment.parent / "rollouts"

    model_rows = []
    seed_rows = []
    all_errors = []
    pdiag_profiles = []
    target_profiles = []
    progress = None
    dataset_hashes = {}
    benchmark_hashes = {}
    physical_checks = {}
    for seed in seeds:
        benchmark_dir = benchmark_root / f"seed_{seed}"
        summary_path = benchmark_dir / "phase_switch_baseline_summary.json"
        arrays_path = benchmark_dir / "phase_switch_baseline_arrays.npz"
        errors_path = benchmark_dir / "phase_switch_baseline_heldout_errors.csv"
        physical_path = rollout_root / f"seed_{seed}_validation_summary.json"
        with summary_path.open(encoding="utf-8") as summary_file:
            summary = json.load(summary_file)
        with physical_path.open(encoding="utf-8") as physical_file:
            physical = json.load(physical_file)
        arrays = np.load(arrays_path)
        current_progress = arrays["progress"]
        if progress is None:
            progress = current_progress
        elif not np.array_equal(progress, current_progress):
            raise RuntimeError("seed benchmarks use different progress grids")
        pdiag_profiles.append(arrays["profile_pdiag_finite"])
        target_profiles.append(arrays["target_profile"])
        seed_errors = pd.read_csv(errors_path)
        seed_errors.insert(0, "seed", seed)
        all_errors.append(seed_errors)
        dataset_path = Path(summary["trajectory"])
        dataset_hashes[str(seed)] = sha256(dataset_path)
        benchmark_hashes[str(seed)] = sha256(summary_path)
        physical_checks[str(seed)] = physical["checks"]

        for model in MODEL_ORDER:
            model_rows.append(
                {"seed": seed, "model": model, **summary["models"][model]}
            )
        scalar = summary["models"]["Frame-weighted"]
        pdiag = summary["models"]["Pdiag finite"]
        criteria = experiment["success_criteria"]
        seed_rows.append(
            {
                "seed": seed,
                "g_translation_final": pdiag["g_translation_final"],
                "g_yaw_preclear": pdiag["g_yaw_preclear"],
                "g_yaw_final": pdiag["g_yaw_final"],
                "empirical_switch_s_0_5": summary["target_switch_s_0_5"],
                "model_switch_s_0_5": pdiag["switch_s_0_5"],
                "switch_absolute_error": pdiag["switch_absolute_error"],
                "scalar_task_error_mm_equiv": scalar[
                    "task_error_mean_mm_equiv"
                ],
                "pdiag_task_error_mm_equiv": pdiag[
                    "task_error_mean_mm_equiv"
                ],
                "task_error_improvement_mm_equiv": scalar[
                    "task_error_mean_mm_equiv"
                ]
                - pdiag["task_error_mean_mm_equiv"],
                "translation_pass": pdiag["g_translation_final"]
                > criteria["g_translation_final_min"],
                "pre_yaw_pass": pdiag["g_yaw_preclear"]
                > criteria["g_yaw_preclear_min"],
                "final_yaw_pass": abs(pdiag["g_yaw_final"])
                < criteria["abs_g_yaw_final_max"],
                "switch_pass": pdiag["switch_absolute_error"]
                < criteria["switch_absolute_error_max"],
                "improvement_pass": pdiag["task_error_mean_mm_equiv"]
                < scalar["task_error_mean_mm_equiv"],
            }
        )

    model_frame = pd.DataFrame(model_rows)
    seed_frame = pd.DataFrame(seed_rows)
    errors = pd.concat(all_errors, ignore_index=True)
    pass_columns = [
        "translation_pass",
        "pre_yaw_pass",
        "final_yaw_pass",
        "switch_pass",
        "improvement_pass",
    ]
    seed_frame["all_criteria_pass"] = seed_frame[pass_columns].all(axis=1)

    stratified = (
        errors.groupby(["seed", "model", "generator"], sort=False)[
            [
                "task_error_mm_equiv",
                "task_endpoint_error_mm_equiv",
            ]
        ]
        .agg(["mean", "median"])
        .reset_index()
    )
    stratified.columns = [
        "_".join(value for value in column if value)
        if isinstance(column, tuple)
        else column
        for column in stratified.columns
    ]

    pdiag_profiles = np.asarray(pdiag_profiles)
    target_profiles = np.asarray(target_profiles)
    hierarchy_ci = hierarchical_bootstrap_improvement(errors)
    finite_vs_competitors = {}
    indexed = model_frame.set_index(["seed", "model"])
    for competitor in ["TP-GMM SE(2)", "Generic RBF", "Full operator"]:
        differences = []
        for seed in seeds:
            differences.append(
                indexed.loc[(seed, competitor), "task_error_mean_mm_equiv"]
                - indexed.loc[(seed, "Pdiag finite"), "task_error_mean_mm_equiv"]
            )
        finite_vs_competitors[competitor] = {
            "competitor_minus_pdiag_by_seed": differences,
            "pdiag_wins": int(np.sum(np.asarray(differences) > 0)),
        }

    additive_boundary_seeds = []
    se2_boundary_seeds = []
    for seed in seeds:
        with (
            benchmark_root / f"seed_{seed}" / "phase_switch_baseline_summary.json"
        ).open(encoding="utf-8") as summary_file:
            summary = json.load(summary_file)
        for model, destination in [
            ("TP-GMM additive", additive_boundary_seeds),
            ("TP-GMM SE(2)", se2_boundary_seeds),
        ]:
            diagnostics = summary["tp_gmm"][model]
            candidates = sorted(
                int(value)
                for value in diagnostics[
                    "grouped_cv_log_likelihood_by_components"
                ]
            )
            if int(diagnostics["components_selected"]) in {
                candidates[0],
                candidates[-1],
            }:
                destination.append(seed)

    summary = {
        "schema_version": 1,
        "analysis_source_sha256": sha256(Path(__file__)),
        "experiment": str(args.experiment),
        "experiment_sha256": sha256(args.experiment),
        "context_manifest_sha256": experiment["context_manifest_sha256"],
        "seeds": seeds,
        "dataset_sha256": dataset_hashes,
        "benchmark_summary_sha256": benchmark_hashes,
        "all_physics_checks_pass": all(
            all(checks.values()) for checks in physical_checks.values()
        ),
        "all_preregistered_seed_criteria_pass": bool(
            seed_frame.all_criteria_pass.all()
        ),
        "criteria_pass_count": {
            column: int(seed_frame[column].sum()) for column in pass_columns
        },
        "task_error_improvement_mm_equiv": {
            "mean_over_seeds": float(
                seed_frame.task_error_improvement_mm_equiv.mean()
            ),
            "sample_sd_over_seeds": float(
                seed_frame.task_error_improvement_mm_equiv.std(ddof=1)
            ),
            "hierarchical_seed_condition_bootstrap_95ci": hierarchy_ci.tolist(),
            "positive_seed_count": int(
                (seed_frame.task_error_improvement_mm_equiv > 0).sum()
            ),
        },
        "pdiag_profiles": {
            "g_translation_final_mean": float(
                seed_frame.g_translation_final.mean()
            ),
            "g_translation_final_sd": float(seed_frame.g_translation_final.std(ddof=1)),
            "g_yaw_preclear_mean": float(seed_frame.g_yaw_preclear.mean()),
            "g_yaw_preclear_sd": float(seed_frame.g_yaw_preclear.std(ddof=1)),
            "g_yaw_final_mean": float(seed_frame.g_yaw_final.mean()),
            "g_yaw_final_sd": float(seed_frame.g_yaw_final.std(ddof=1)),
            "switch_error_mean": float(seed_frame.switch_absolute_error.mean()),
            "switch_error_max": float(seed_frame.switch_absolute_error.max()),
        },
        "finite_vs_competitors": finite_vs_competitors,
        "tp_gmm_diagnostics": {
            "additive_cv_boundary_seeds": additive_boundary_seeds,
            "formal_se2_cv_boundary_seeds": se2_boundary_seeds,
        },
        "interpretation": (
            "Seed-level replication supports robustness of the frozen finite "
            "Pdiag law relative to shared scalar relevance. Generic RBF and the "
            "full operator remain competitive and are not claimed to be dominated."
        ),
    }

    seed_path = args.output_root / "multiseed_seed_summary.csv"
    model_path = args.output_root / "multiseed_model_summary.csv"
    errors_path = args.output_root / "multiseed_heldout_errors.csv"
    stratified_path = args.output_root / "multiseed_stratified.csv"
    summary_path = args.output_root / "multiseed_summary.json"
    seed_frame.to_csv(seed_path, index=False)
    model_frame.to_csv(model_path, index=False)
    errors.to_csv(errors_path, index=False)
    stratified.to_csv(stratified_path, index=False)
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

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
    figure, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), constrained_layout=True)
    phases = ["Align", "Enter", "Unlock", "Insert"]
    phase_centers = [0.125, 0.375, 0.625, 0.875]
    for axis, generator, title, ylabel in [
        (axes[0, 0], 2, "A  Axial-yaw response", r"$g_{yaw}(s)$"),
        (axes[0, 1], None, "B  Translation response", r"$g_{trans}(s)$"),
    ]:
        values = (
            pdiag_profiles[:, :, generator]
            if generator is not None
            else pdiag_profiles[:, :, :2].mean(axis=2)
        )
        target_values = (
            target_profiles[:, :, generator]
            if generator is not None
            else target_profiles[:, :, :2].mean(axis=2)
        )
        for seed_index in range(len(seeds)):
            axis.plot(progress, values[seed_index], color="#0072B2", alpha=0.28)
            axis.plot(
                progress,
                target_values[seed_index],
                color="0.55",
                alpha=0.18,
                linestyle="--",
            )
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1)
        axis.fill_between(
            progress, mean - sd, mean + sd, color="#0072B2", alpha=0.18
        )
        axis.plot(progress, mean, color="#0072B2", linewidth=2.2, label="Pdiag mean")
        axis.plot(
            progress,
            target_values.mean(axis=0),
            color="0.35",
            linewidth=1.8,
            linestyle="--",
            label="Empirical mean",
        )
        for boundary in [0.25, 0.5, 0.75]:
            axis.axvline(boundary, color="0.82", linewidth=0.7)
        axis.set_xticks(phase_centers, phases)
        axis.set_xlim(0, 1)
        axis.set_ylim(-0.15, 1.2)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, loc="lower left")

    axis = axes[1, 0]
    axis.axhline(0.0, color="0.55", linewidth=0.8)
    axis.scatter(
        np.arange(len(seeds)),
        seed_frame.task_error_improvement_mm_equiv,
        color="#009E73",
        edgecolor="black",
        linewidth=0.5,
        s=38,
    )
    axis.set_xticks(np.arange(len(seeds)), [f"S{i + 1}" for i in range(len(seeds))])
    axis.set_ylabel(r"$E_{scalar}-E_{Pdiag}$ (mm-equiv.)")
    axis.set_title("C  Per-seed task-error improvement", loc="left", fontweight="bold")

    axis = axes[1, 1]
    empirical = seed_frame.empirical_switch_s_0_5.to_numpy()
    modeled = seed_frame.model_switch_s_0_5.to_numpy()
    low = min(empirical.min(), modeled.min()) - 0.005
    high = max(empirical.max(), modeled.max()) + 0.005
    axis.plot([low, high], [low, high], color="0.55", linestyle="--", linewidth=1.0)
    axis.scatter(
        empirical,
        modeled,
        color="#D55E00",
        edgecolor="black",
        linewidth=0.5,
        s=38,
    )
    for index, seed in enumerate(seeds):
        axis.annotate(
            f"S{index + 1}",
            (empirical[index], modeled[index]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
        )
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel(r"Empirical $s_{0.5}$")
    axis.set_ylabel(r"Pdiag $s_{0.5}$")
    axis.set_title("D  Switch timing", loc="left", fontweight="bold")

    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure_stem = args.output_root / "phase_switch_multiseed"
    figure.savefig(
        figure_stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.06
    )
    figure.savefig(
        figure_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06
    )
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
