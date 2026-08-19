from __future__ import annotations

"""TP-GMM SE(2) few-shot sensitivity (Experiment B, small checks).

Two frozen sensitivity checks on the SAME matched subsets, so reviewers cannot
claim the few-shot TP-GMM disadvantage is an artefact of the mixture-complexity
search space or the reduced EM budget:

  1. expanded-K: include small component counts K = 2, 3 (the main run starts
     at K = 4). Candidates = (2,3,4,6,8,10,12,16,20) capped at N-1.
  2. full-EM (N=5 only): restore the full-data EM budget
     (cv_n_init=5, final_n_init=10) on the expanded-K candidate set.

Frozen subset selection is identical to the main run: for N=5 and N=10, the
first five random-protocol subsets (repeat 0..4).
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


SENSITIVITY_CANDIDATES = (2, 3, 4, 6, 8, 10, 12, 16, 20)
TARGET_SIZES = (5, 10)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_subsets(subset_manifest):
    selected = []
    for sample_size in TARGET_SIZES:
        for repeat in range(5):
            matches = [
                s
                for s in subset_manifest["subsets"]
                if s["sample_size"] == sample_size
                and s["protocol"] == "random"
                and s["repeat"] == repeat
            ]
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
                k for k in SENSITIVITY_CANDIDATES if k <= max(n_episodes - 1, 2)
            )
            variants = [("expanded-K", 2, 5)]
            if n_episodes == 5:
                variants.append(("full-EM", 5, 10))
            for variant, cv_n_init, final_n_init in variants:
                model = TPGMMModel(
                    frame_mode="se2",
                    nominal_frame_xy=frame_xy,
                    nominal_frame_yaw=frame_yaw,
                    component_candidates=candidates,
                    cv_splits=min(5, n_episodes),
                    cv_n_init=cv_n_init,
                    final_n_init=final_n_init,
                )
                started = time.perf_counter()
                fit_error = ""
                values = {}
                try:
                    model.fit(contexts, curves, progress, phase_codes)
                    prediction = model.predict(test_contexts)
                    task_error, _ = metric_errors(
                        prediction, test_curves, yaw_weights=task_yaw_weights
                    )
                    values = {
                        "task_error_mean_mm_equiv": float(task_error.mean()),
                        "n_components": int(model.n_components),
                    }
                except Exception as exc:
                    fit_error = repr(exc)
                    values = {
                        "task_error_mean_mm_equiv": np.nan,
                        "n_components": np.nan,
                    }
                rows.append(
                    {
                        "seed": args.seed,
                        "subset_id": int(subset["subset_id"]),
                        "sample_size": int(subset["sample_size"]),
                        "variant": variant,
                        "component_candidates": candidates,
                        "cv_n_init": cv_n_init,
                        "final_n_init": final_n_init,
                        "fit_success": fit_error == "",
                        "fit_error": fit_error,
                        "fit_seconds": time.perf_counter() - started,
                        **values,
                    }
                )
                print(
                    f"seed {args.seed} subset {subset['subset_id']} (N={subset['sample_size']}) "
                    f"{variant} -> {values.get('task_error_mean_mm_equiv', float('nan')):.2f} mm "
                    f"(K={values.get('n_components', float('nan'))})",
                    flush=True,
                )

    result_path = output_root / "tpgmm_fewshot_sensitivity.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    with (output_root / "tpgmm_fewshot_sensitivity_run.json").open("w") as out:
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
                "variants": ["expanded-K", "full-EM (N=5 only)"],
                "candidates_rule": "full set (2,3,4,6,8,10,12,16,20) capped at N-1",
            },
            out,
            indent=2,
        )
    print("saved:", result_path)


if __name__ == "__main__":
    main()
