from __future__ import annotations

"""TP-GMM SE(2) few-shot at N=3 (fills the missing row in tab:fewshot_n3).

The frozen few-shot sweep (run_phase_switch_fewshot.py) covers Pdiag finite,
Full operator, and Generic RBF down to N=3, but explicitly excluded TP-GMM SE(2)
(see fewshot_summary.json scope note). This script fills that gap on the SAME
frozen N=3 subsets (10 random + 10 qualified per seed) so the TP-GMM SE(2) row
can be added to tab:fewshot_n3.

N=3 is below the main component grid (K starts at 4), so we use the expanded-K
sensitivity config: candidates (2,3,4,6,8,10,12,16,20) capped at N-1 -> (2,),
cv_splits = min(5, N) = 3, cv_n_init = 2, final_n_init = 5.

Metrics and aggregation match analyze_phase_switch_fewshot.py exactly:
  * E_task/E_end/E_gen and the three gains are mean-over-seed-means +- between-
    seed std (ddof=1) over the 5 seeds.
  * switch and law are fractions over the 50 seed/subset cells per protocol.
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


EXPANDED_K_CANDIDATES = (2, 3, 4, 6, 8, 10, 12, 16, 20)
N3 = 3
MODEL_NAME = "TP-GMM SE(2)"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_n3_subsets(subset_manifest):
    selected = [
        s for s in subset_manifest["subsets"] if s["sample_size"] == N3
    ]
    random_count = sum(1 for s in selected if s["protocol"] == "random")
    qualified_count = sum(1 for s in selected if s["protocol"] == "qualified")
    if random_count != 10 or qualified_count != 10:
        raise RuntimeError(
            f"expected 10 random + 10 qualified N=3 subsets, got "
            f"{random_count} + {qualified_count}"
        )
    return selected


def run_seed(seed, experiment_dir, n3_subsets, bins):
    dataset = experiment_dir / "rollouts" / f"seed_{seed}.h5"
    progress, phase_codes = progress_grid(bins)
    unlock_start = 2 * bins

    rows = []
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
            [task_curve(data_file[key], bins) for key in mixed_keys]
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
            [task_curve(data_file[key], bins) for key in isolated_keys]
        )
        target_profile = empirical_isolated_profile(
            data_file, baseline_key, isolated_keys, bins
        )
        target_switch = switch_diagnostics(progress, target_profile[:, 2], unlock_start)
        if not target_switch["detected"]:
            raise RuntimeError("empirical profile has no yaw switch")
        task_yaw_weights = (phase_codes < 2).astype(np.float64)

        for subset in n3_subsets:
            indices = np.asarray(subset["mixed_indices"], dtype=int)
            contexts = all_contexts[indices]
            curves = all_curves[indices]
            keys = [mixed_keys[index] for index in indices]
            frame_xy, frame_yaw = nominal_frame(data_file, keys, contexts)
            n_episodes = len(contexts)
            candidates = tuple(
                k for k in EXPANDED_K_CANDIDATES if k <= max(n_episodes - 1, 2)
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
                    "g_yaw_preclear": float(profile[2 * bins - 1, 2]),
                    "g_yaw_final": float(profile[-1, 2]),
                    "switch_detected": bool(model_switch["detected"]),
                    "switch_absolute_error": (
                        abs(model_switch["location"] - target_switch["location"])
                        if model_switch["detected"]
                        else np.nan
                    ),
                    "n_components": int(model.n_components),
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
                }
            rows.append(
                {
                    "seed": seed,
                    "subset_id": int(subset["subset_id"]),
                    "protocol": subset["protocol"],
                    "sample_size": int(subset["sample_size"]),
                    "repeat": int(subset["repeat"]),
                    "model": MODEL_NAME,
                    "fit_success": fit_error == "",
                    "fit_error": fit_error,
                    "fit_seconds": time.perf_counter() - started,
                    **values,
                }
            )
    return rows


def aggregate(fits):
    successful = fits[fits.fit_success].copy()
    successful["law_pass"] = (
        (successful.g_translation_final > 0.90)
        & (successful.g_yaw_preclear > 0.90)
        & (successful.g_yaw_final.abs() < 0.10)
    )
    continuous = [
        "task_error_mean_mm_equiv",
        "task_endpoint_error_mean_mm_equiv",
        "generator_rmse",
        "g_translation_final",
        "g_yaw_preclear",
        "g_yaw_final",
    ]
    out = {}
    for protocol in ["random", "qualified"]:
        cells = successful[successful.protocol == protocol]
        seed_means = cells.groupby("seed")[continuous].mean()
        out[protocol] = {
            metric: {
                "mean_over_seed_means": float(seed_means[metric].mean()),
                "between_seed_sd": float(seed_means[metric].std(ddof=1)),
            }
            for metric in continuous
        }
        out[protocol]["switch_detection_fraction"] = float(cells.switch_detected.mean())
        out[protocol]["generator_law_pass_fraction"] = float(cells.law_pass.mean())
        out[protocol]["fit_count"] = int(len(cells))
    return out


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
        default=Path("phase_switch_symmetry_multiseed/tpgmm_fewshot_n3"),
    )
    parser.add_argument("--bins", type=int, default=25)
    args = parser.parse_args()

    with args.experiment.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    with args.subsets.open(encoding="utf-8") as subset_file:
        subset_manifest = json.load(subset_file)
    if subset_manifest["experiment_sha256"] != sha256(args.experiment):
        raise RuntimeError("few-shot subsets do not match experiment")

    n3_subsets = select_n3_subsets(subset_manifest)
    seeds = [int(s) for s in experiment["seeds"]]
    experiment_dir = args.experiment.parent

    all_rows = []
    for seed in seeds:
        started = time.perf_counter()
        rows = run_seed(seed, experiment_dir, n3_subsets, args.bins)
        all_rows.extend(rows)
        print(
            f"seed {seed}: {len(rows)} fits, "
            f"{sum(not r['fit_success'] for r in rows)} failures "
            f"({time.perf_counter() - started:.1f}s)",
            flush=True,
        )

    fits = pd.DataFrame(all_rows)
    agg = aggregate(fits)

    args.output_root.mkdir(parents=True, exist_ok=True)
    fits_path = args.output_root / "tpgmm_fewshot_n3_fits.csv"
    fits.to_csv(fits_path, index=False)

    summary = {
        "schema_version": 1,
        "model": MODEL_NAME,
        "sample_size": N3,
        "experiment_sha256": sha256(args.experiment),
        "subset_manifest_sha256": sha256(args.subsets),
        "source_sha256": sha256(Path(__file__)),
        "config": {
            "component_candidates_rule": (
                "expanded-K (2,3,4,6,8,10,12,16,20) capped at N-1 -> (2,) at N=3"
            ),
            "cv_splits": "min(5, N) = 3",
            "cv_n_init": 2,
            "final_n_init": 5,
        },
        "subset_ids": [int(s["subset_id"]) for s in n3_subsets],
        "seeds": seeds,
        "fit_count": int(len(fits)),
        "fit_success_count": int(fits.fit_success.sum()),
        "fit_failure_count": int((~fits.fit_success).sum()),
        "aggregation": agg,
    }
    summary_path = args.output_root / "tpgmm_fewshot_n3_summary.json"
    with summary_path.open("w", encoding="utf-8") as out:
        json.dump(summary, out, indent=2, allow_nan=False)

    continuous = [
        "task_error_mean_mm_equiv",
        "task_endpoint_error_mean_mm_equiv",
        "generator_rmse",
        "g_translation_final",
        "g_yaw_preclear",
        "g_yaw_final",
    ]
    labels = {
        "task_error_mean_mm_equiv": "E_task",
        "task_endpoint_error_mean_mm_equiv": "E_end",
        "generator_rmse": "E_gen",
        "g_translation_final": "g_trans",
        "g_yaw_preclear": "g_psi_pre",
        "g_yaw_final": "g_psi_final",
    }
    print("\n=== TP-GMM SE(2) few-shot N=3 (mean over seed means +- between-seed sd) ===")
    for protocol in ["random", "qualified"]:
        print(f"\n[{protocol}]")
        for metric in continuous:
            m = agg[protocol][metric]
            print(
                f"  {labels[metric]:<10} {m['mean_over_seed_means']:.3f} "
                f"+- {m['between_seed_sd']:.3f}"
            )
        print(
            f"  switch     {agg[protocol]['switch_detection_fraction']*100:.0f}%"
        )
        print(
            f"  law        {agg[protocol]['generator_law_pass_fraction']*100:.0f}%"
        )
        print(f"  fits       {agg[protocol]['fit_count']}")

    if (~fits.fit_success).any():
        print("\nfit failures:")
        print(fits[~fits.fit_success][["seed", "subset_id", "protocol", "fit_error"]].to_string(index=False))
    print("\nsaved:", summary_path, "\n", fits_path)


if __name__ == "__main__":
    main()
