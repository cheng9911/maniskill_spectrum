from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)
    print("PASS:", message)


def usable(group):
    phases = set(np.asarray(group["solver_phase"], dtype=int).tolist())
    return (
        bool(np.asarray(group["success"])[-1])
        and all(phase in phases for phase in [3, 4, 5, 6])
        and not bool(np.asarray(group["truncated"], dtype=bool).any())
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/experiment.json"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/results"),
    )
    args = parser.parse_args()

    summary_path = args.results / "multiseed_summary.json"
    seed_path = args.results / "multiseed_seed_summary.csv"
    model_path = args.results / "multiseed_model_summary.csv"
    errors_path = args.results / "multiseed_heldout_errors.csv"
    stratified_path = args.results / "multiseed_stratified.csv"
    figure_png = args.results / "phase_switch_multiseed.png"
    figure_pdf = args.results / "phase_switch_multiseed.pdf"
    for path in [
        args.experiment,
        summary_path,
        seed_path,
        model_path,
        errors_path,
        stratified_path,
        figure_png,
        figure_pdf,
    ]:
        require(path.is_file() and path.stat().st_size > 0, f"artifact exists: {path}")

    with args.experiment.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    with summary_path.open(encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    require(summary["schema_version"] == 1, "recognized multi-seed schema")
    require(
        summary["analysis_source_sha256"] == sha256(
            Path(__file__).with_name("analyze_phase_switch_multiseed.py")
        ),
        "analysis source hash matches result",
    )
    require(summary["experiment_sha256"] == sha256(args.experiment), "experiment hash matches")
    context_path = Path(experiment["context_manifest"])
    require(
        sha256(context_path) == experiment["context_manifest_sha256"],
        "frozen context manifest hash matches preregistration",
    )
    with context_path.open(encoding="utf-8") as context_file:
        context_manifest = json.load(context_file)
    expected_rows = context_manifest["conditions"]
    expected_contexts = np.asarray(
        [row["causal_delta"] for row in expected_rows], dtype=np.float64
    )
    expected_generators = [row["generator"] for row in expected_rows]
    seeds = [int(seed) for seed in experiment["seeds"]]
    require(len(seeds) == 5 and len(set(seeds)) == 5, "five unique preregistered seeds")
    require(summary["seeds"] == seeds, "result uses exactly preregistered seeds")

    benchmark_root = args.experiment.parent / "benchmarks"
    rollout_root = args.experiment.parent / "rollouts"
    for seed in seeds:
        dataset = rollout_root / f"seed_{seed}.h5"
        benchmark_dir = benchmark_root / f"seed_{seed}"
        benchmark_summary_path = benchmark_dir / "phase_switch_baseline_summary.json"
        require(sha256(dataset) == summary["dataset_sha256"][str(seed)], f"seed {seed} dataset hash matches")
        require(
            sha256(benchmark_summary_path)
            == summary["benchmark_summary_sha256"][str(seed)],
            f"seed {seed} benchmark hash matches",
        )
        with benchmark_summary_path.open(encoding="utf-8") as benchmark_file:
            benchmark = json.load(benchmark_file)
        require(
            benchmark["trajectory_sha256"] == sha256(dataset),
            f"seed {seed} benchmark is tied to its own dataset",
        )
        require(
            benchmark["pdiag_finite"]["alpha_max"]
            == experiment["frozen_model"]["alpha_max"]
            and benchmark["pdiag_finite"]["n_basis"]
            == experiment["frozen_model"]["n_basis"]
            and benchmark["pdiag_finite"]["basis_width"]
            == experiment["frozen_model"]["basis_width"]
            and benchmark["pdiag_finite"]["smoothness_weight"]
            == experiment["frozen_model"]["smoothness_weight"]
            and benchmark["pdiag_finite"]["nominal_iterations"]
            == experiment["frozen_model"]["nominal_iterations"],
            f"seed {seed} uses frozen Pdiag hyperparameters",
        )
        formal_gmm = benchmark["tp_gmm"]["TP-GMM SE(2)"]
        candidates = sorted(
            int(value)
            for value in formal_gmm["grouped_cv_log_likelihood_by_components"]
        )
        require(formal_gmm["final_converged"], f"seed {seed} formal TP-GMM converged")
        require(
            int(formal_gmm["components_selected"])
            not in {candidates[0], candidates[-1]},
            f"seed {seed} formal TP-GMM CV optimum is not a boundary",
        )

        with h5py.File(dataset, "r") as data_file:
            require(int(data_file.attrs["seed"]) == seed, f"seed {seed} H5 root seed matches")
            require(
                float(data_file.attrs["robot_init_qpos_noise"])
                == experiment["robot_init_qpos_noise_rad"],
                f"seed {seed} uses preregistered initialization noise",
            )
            require(
                str(data_file.attrs["context_manifest_sha256"])
                == experiment["context_manifest_sha256"],
                f"seed {seed} H5 references frozen contexts",
            )
            groups = [
                data_file[key]
                for key in data_file
                if key.startswith("episode_")
            ]
            for condition_id, (context, generator) in enumerate(
                zip(expected_contexts, expected_generators)
            ):
                attempts = [
                    group
                    for group in groups
                    if int(group.attrs["condition_id"]) == condition_id
                ]
                require(bool(attempts), f"seed {seed} condition {condition_id} retained")
                require(
                    all(
                        str(group.attrs["generator"]) == generator
                        and np.array_equal(
                            np.asarray(group["causal_delta"], dtype=np.float64),
                            context,
                        )
                        for group in attempts
                    ),
                    f"seed {seed} condition {condition_id} exactly matches manifest",
                )
                require(
                    sum(usable(group) for group in attempts) == 1,
                    f"seed {seed} condition {condition_id} has one usable success",
                )
        physical_path = rollout_root / f"seed_{seed}_validation_summary.json"
        with physical_path.open(encoding="utf-8") as physical_file:
            physical = json.load(physical_file)
        require(all(physical["checks"].values()), f"seed {seed} passes all physics checks")

    seed_frame = pd.read_csv(seed_path)
    model_frame = pd.read_csv(model_path)
    errors = pd.read_csv(errors_path)
    stratified = pd.read_csv(stratified_path)
    require(seed_frame.seed.tolist() == seeds, "seed summary order is fixed")
    require(len(model_frame) == 5 * 8, "model table has eight independent fits per seed")
    require(len(errors) == 5 * 8 * 8, "held-out table has eight conditions and models per seed")
    require(len(stratified) == 5 * 8 * 2, "stratified table has yaw/translation per seed and model")
    require(seed_frame.all_criteria_pass.all(), "all seed-level preregistered criteria pass")
    require((seed_frame.task_error_improvement_mm_equiv > 0).all(), "Pdiag improves over scalar in 5/5 seeds")
    criteria = experiment["success_criteria"]
    require(
        (seed_frame.g_translation_final > criteria["g_translation_final_min"]).all(),
        "final translation criterion recomputes",
    )
    require(
        (seed_frame.g_yaw_preclear > criteria["g_yaw_preclear_min"]).all(),
        "pre-clear yaw criterion recomputes",
    )
    require(
        (seed_frame.g_yaw_final.abs() < criteria["abs_g_yaw_final_max"]).all(),
        "final yaw criterion recomputes",
    )
    require(
        (seed_frame.switch_absolute_error < criteria["switch_absolute_error_max"]).all(),
        "switch timing criterion recomputes",
    )
    require(summary["all_physics_checks_pass"], "aggregate physics flag passes")
    require(summary["all_preregistered_seed_criteria_pass"], "aggregate preregistered flag passes")
    require(
        summary["task_error_improvement_mm_equiv"][
            "hierarchical_seed_condition_bootstrap_95ci"
        ][0]
        > 0,
        "hierarchical seed-condition improvement interval is positive",
    )
    require(
        summary["tp_gmm_diagnostics"]["formal_se2_cv_boundary_seeds"] == [],
        "formal TP-GMM has no CV boundary seeds",
    )
    print("All multi-seed validation checks passed.")


if __name__ == "__main__":
    main()
