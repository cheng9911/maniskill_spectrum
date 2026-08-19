from __future__ import annotations

"""TP-GMM SE(2) matched few-shot (Experiment B).

Fits the formal rotation-aware TP-GMM SE(2) baseline on a frozen, pre-declared
subset of the existing few-shot manifest, so its sample-efficiency can be
compared against Pdiag finite on the SAME (seed, sample size, protocol, subset)
without outcome-dependent subset selection.

Frozen subset selection (declared before any TP-GMM result is inspected):
  * N = 5, 10, 20  -> the first five "random"-protocol subsets (repeat 0..4).
  * N = 30         -> the single "all" subset.
This is 16 subsets x 5 seeds = 80 fits.

Few-shot TP-GMM protocol (frozen):
  * frame_mode = "se2" (the formal baseline).
  * component candidates = the full-data set (4,6,8,10,12,16,20) capped at N-1.
  * cv_splits = min(5, N); cv_n_init = 2; final_n_init = 5.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd

from benchmark_phase_switch_baselines import (
    empirical_isolated_profile,
    metric_errors,
    progress_grid,
    switch_diagnostics,
    task_curve,
    usable,
)
from phase_switch_baselines import TPGMMModel
from run_phase_switch_fewshot import nominal_frame


FULL_COMPONENT_CANDIDATES = (4, 6, 8, 10, 12, 16, 20)
TARGET_SIZES = (5, 10, 20)
REPEATS_PER_SIZE = 5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_subsets(subset_manifest):
    selected = []
    for sample_size in TARGET_SIZES:
        for repeat in range(REPEATS_PER_SIZE):
            matches = [
                s
                for s in subset_manifest["subsets"]
                if s["sample_size"] == sample_size
                and s["protocol"] == "random"
                and s["repeat"] == repeat
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one random subset for N={sample_size} repeat={repeat}"
                )
            selected.append(matches[0])
    matches = [s for s in subset_manifest["subsets"] if s["sample_size"] == 30]
    if len(matches) != 1:
        raise RuntimeError("expected one all-data subset")
    selected.append(matches[0])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
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
    parser.add_argument("--bins", type=int, default=25)
    args = parser.parse_args()

    with args.experiment.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    if args.seed not in experiment["seeds"]:
        raise ValueError("seed is not preregistered")
    with args.subsets.open(encoding="utf-8") as subset_file:
        subset_manifest = json.load(subset_file)
    if subset_manifest["experiment_sha256"] != sha256(args.experiment):
        raise RuntimeError("few-shot subsets do not match experiment")

    selected = select_subsets(subset_manifest)
    selected_ids = [int(s["subset_id"]) for s in selected]

    dataset = args.experiment.parent / "rollouts" / f"seed_{args.seed}.h5"
    output_root = args.experiment.parent / "tpgmm_fewshot" / f"seed_{args.seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    progress, phase_codes = progress_grid(args.bins)

    with h5py.File(dataset, "r") as data_file:
        usable_groups = {
            int(data_file[key].attrs["condition_id"]): key
            for key in data_file
            if key.startswith("episode_") and usable(data_file[key])
        }
        mixed_keys = [
            key
            for _, key in sorted(usable_groups.items())
            if str(data_file[key].attrs["generator"]) == "mixed"
        ]
        if len(mixed_keys) != 30:
            raise RuntimeError("expected thirty usable mixed conditions")
        all_contexts = np.asarray(
            [np.asarray(data_file[key]["causal_delta"]) for key in mixed_keys]
        )
        all_curves = np.asarray(
            [task_curve(data_file[key], args.bins) for key in mixed_keys]
        )
        isolated_keys = [
            key
            for _, key in sorted(usable_groups.items())
            if str(data_file[key].attrs["generator"]) in {"yaw", "translation"}
            and np.linalg.norm(np.asarray(data_file[key]["causal_delta"])) > 1e-12
        ]
        baseline_keys = [
            key
            for _, key in sorted(usable_groups.items())
            if str(data_file[key].attrs["generator"]) == "yaw"
            and np.linalg.norm(np.asarray(data_file[key]["causal_delta"])) <= 1e-12
        ]
        if len(isolated_keys) != 8 or len(baseline_keys) != 1:
            raise RuntimeError("expected 8 isolated and one zero condition")
        baseline_key = baseline_keys[0]
        test_contexts = np.asarray(
            [np.asarray(data_file[key]["causal_delta"]) for key in isolated_keys]
        )
        test_curves = np.asarray(
            [task_curve(data_file[key], args.bins) for key in isolated_keys]
        )
        target_profile = empirical_isolated_profile(
            data_file, baseline_key, isolated_keys, args.bins
        )

        unlock_start = 2 * args.bins
        target_switch = switch_diagnostics(progress, target_profile[:, 2], unlock_start)
        if not target_switch["detected"]:
            raise RuntimeError("empirical profile has no yaw switch")
        task_yaw_weights = (phase_codes < 2).astype(np.float64)

        rows = []
        for subset in selected:
            indices = np.asarray(subset["mixed_indices"], dtype=int)
            contexts = all_contexts[indices]
            curves = all_curves[indices]
            keys = [mixed_keys[index] for index in indices]
            frame_xy, frame_yaw = nominal_frame(data_file, keys, contexts)
            n_episodes = len(contexts)
            candidates = tuple(
                k
                for k in FULL_COMPONENT_CANDIDATES
                if k <= max(n_episodes - 1, 2)
            )
            model = TPGMMModel(
                frame_mode="se2",
                nominal_frame_xy=frame_xy,
                nominal_frame_yaw=frame_yaw,
                component_candidates=candidates,
                cv_splits=min(5, n_episodes),
                cv_n_init=2,
                final_n_init=5,
            )
            started = time.perf_counter()
            fit_error = ""
            values = {}
            try:
                model.fit(contexts, curves, progress, phase_codes)
                prediction = model.predict(test_contexts)
                profile = model.jacobian_diag()
                task_error, task_endpoint = metric_errors(
                    prediction, test_curves, yaw_weights=task_yaw_weights
                )
                model_switch = switch_diagnostics(
                    progress, profile[:, 2], unlock_start
                )
                values = {
                    "task_error_mean_mm_equiv": float(task_error.mean()),
                    "task_endpoint_error_mean_mm_equiv": float(task_endpoint.mean()),
                    "generator_rmse": float(
                        np.sqrt(np.mean((profile - target_profile) ** 2))
                    ),
                    "g_translation_final": float(profile[-1, :2].mean()),
                    "g_yaw_preclear": float(profile[2 * args.bins - 1, 2]),
                    "g_yaw_final": float(profile[-1, 2]),
                    "switch_detected": bool(model_switch["detected"]),
                    "switch_absolute_error": (
                        abs(model_switch["location"] - target_switch["location"])
                        if model_switch["detected"]
                        else np.nan
                    ),
                    "n_components": int(model.n_components),
                    "component_candidates": candidates,
                }
            except Exception as exc:
                fit_error = repr(exc)
                values = {
                    "task_error_mean_mm_equiv": np.nan,
                    "task_endpoint_error_mean_mm_equiv": np.nan,
                    "generator_rmse": np.nan,
                    "g_translation_final": np.nan,
                    "g_yaw_preclear": np.nan,
                    "g_yaw_final": np.nan,
                    "switch_detected": False,
                    "switch_absolute_error": np.nan,
                    "n_components": np.nan,
                    "component_candidates": candidates,
                }
            rows.append(
                {
                    "seed": args.seed,
                    "subset_id": int(subset["subset_id"]),
                    "protocol": subset["protocol"],
                    "sample_size": int(subset["sample_size"]),
                    "repeat": int(subset["repeat"]),
                    "model": "TP-GMM SE(2)",
                    "fit_success": fit_error == "",
                    "fit_error": fit_error,
                    "fit_seconds": time.perf_counter() - started,
                    **values,
                }
            )
            print(
                f"seed {args.seed}: subset {subset['subset_id']} "
                f"(N={subset['sample_size']}) done in {rows[-1]['fit_seconds']:.1f}s",
                flush=True,
            )

    result_path = output_root / "tpgmm_fewshot_results.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    with (output_root / "tpgmm_fewshot_run.json").open("w", encoding="utf-8") as out:
        json.dump(
            {
                "schema_version": 1,
                "seed": args.seed,
                "dataset_sha256": sha256(dataset),
                "experiment_sha256": sha256(args.experiment),
                "subset_manifest_sha256": sha256(args.subsets),
                "source_sha256": sha256(Path(__file__)),
                "model": "TP-GMM SE(2)",
                "selected_subset_ids": selected_ids,
                "selection_rule": (
                    "N in {5,10,20}: first five random-protocol subsets "
                    "(repeat 0..4); N=30: the all subset. Declared before any "
                    "TP-GMM result was inspected."
                ),
                "fit_count": len(rows),
                "fit_failure_count": int(sum(not bool(r["fit_success"]) for r in rows)),
                "component_candidates_rule": "full set (4,6,8,10,12,16,20) capped at N-1",
                "cv_splits": "min(5, N)",
                "cv_n_init": 2,
                "final_n_init": 5,
            },
            out,
            indent=2,
        )
    print("saved:", result_path)


if __name__ == "__main__":
    main()
