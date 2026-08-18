from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from analyze_phase_switch_rollouts import pose_yaw, wrap_pi
from benchmark_phase_switch_baselines import (
    empirical_isolated_profile,
    metric_errors,
    progress_grid,
    switch_diagnostics,
    task_curve,
    usable,
)
from phase_switch_baselines import (
    FullOperatorModel,
    GenericConditionalRBF,
    SmoothFinitePDiagModel,
    to_metric,
)


MODEL_ORDER = ["Pdiag finite", "Full operator", "Generic RBF"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FewShotGenericRBF(GenericConditionalRBF):
    """The frozen RBF baseline with feasible episode-grouped CV for N < 5."""

    def fit(self, contexts, curves, progress, phase_codes):
        self.progress = np.asarray(progress, dtype=np.float64)
        self.phase_codes = np.asarray(phase_codes, dtype=int)
        self.basis = self._basis(self.progress, self.phase_codes)
        design = self._design(contexts, self.basis)
        outputs = to_metric(curves).reshape(-1, 3)
        groups = np.repeat(np.arange(len(contexts)), len(self.progress))
        cv_splits = min(5, len(contexts))
        grouped_cv = list(
            GroupKFold(n_splits=cv_splits).split(design, outputs, groups)
        )
        self.model = RidgeCV(
            alphas=np.logspace(-10, 0, 16),
            fit_intercept=False,
            cv=grouped_cv,
        ).fit(design, outputs)
        self.alpha = float(self.model.alpha_)
        self.cv_splits = cv_splits
        return self


def nominal_frame(data_file, keys, contexts):
    xy = []
    yaw = []
    for key, context in zip(keys, contexts):
        socket_pose = np.asarray(data_file[key]["socket_pose"])[0]
        xy.append(socket_pose[:2] - context[:2])
        yaw.append(wrap_pi(float(pose_yaw(socket_pose[None])[0]) - context[2]))
    xy = np.asarray(xy)
    yaw = np.asarray(yaw)
    return xy.mean(axis=0), float(
        np.arctan2(np.sin(yaw).mean(), np.cos(yaw).mean())
    )


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
    dataset = args.experiment.parent / "rollouts" / f"seed_{args.seed}.h5"
    output_root = args.experiment.parent / "fewshot" / f"seed_{args.seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    progress, phase_codes = progress_grid(args.bins)

    with h5py.File(dataset, "r") as data_file:
        usable_groups = {
            int(data_file[key].attrs["condition_id"]): key
            for key in data_file
            if key.startswith("episode_") and usable(data_file[key])
        }
        if len(usable_groups) != 39:
            raise RuntimeError("expected one usable episode for each condition")
        mixed_condition_ids = [
            condition_id
            for condition_id, key in sorted(usable_groups.items())
            if str(data_file[key].attrs["generator"]) == "mixed"
        ]
        mixed_keys = [usable_groups[condition_id] for condition_id in mixed_condition_ids]
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
            if str(data_file[key].attrs["generator"])
            in {"yaw", "translation"}
            and np.linalg.norm(np.asarray(data_file[key]["causal_delta"])) > 1e-12
        ]
        baseline_keys = [
            key
            for _, key in sorted(usable_groups.items())
            if str(data_file[key].attrs["generator"]) == "yaw"
            and np.linalg.norm(np.asarray(data_file[key]["causal_delta"])) <= 1e-12
        ]
        if len(isolated_keys) != 8 or len(baseline_keys) != 1:
            raise RuntimeError("expected 8 nonzero isolated and one zero condition")
        baseline_key = baseline_keys[0]
        test_contexts = np.asarray(
            [np.asarray(data_file[key]["causal_delta"]) for key in isolated_keys]
        )
        test_curves = np.asarray(
            [task_curve(data_file[key], args.bins) for key in isolated_keys]
        )
        test_generators = [
            str(data_file[key].attrs["generator"]) for key in isolated_keys
        ]
        target_profile = empirical_isolated_profile(
            data_file, baseline_key, isolated_keys, args.bins
        )

        unlock_start = 2 * args.bins
        target_switch = switch_diagnostics(
            progress, target_profile[:, 2], unlock_start
        )
        if not target_switch["detected"]:
            raise RuntimeError("empirical profile has no yaw switch")
        task_yaw_weights = (phase_codes < 2).astype(np.float64)
        rows = []
        error_rows = []
        profiles = []
        for subset_index, subset in enumerate(subset_manifest["subsets"]):
            indices = np.asarray(subset["mixed_indices"], dtype=int)
            contexts = all_contexts[indices]
            curves = all_curves[indices]
            keys = [mixed_keys[index] for index in indices]
            frame_xy, frame_yaw = nominal_frame(data_file, keys, contexts)
            models = [
                SmoothFinitePDiagModel(
                    nominal_frame_xy=frame_xy,
                    nominal_frame_yaw=frame_yaw,
                    alpha_max=experiment["frozen_model"]["alpha_max"],
                    n_basis=experiment["frozen_model"]["n_basis"],
                    basis_width=experiment["frozen_model"]["basis_width"],
                    smoothness_weight=experiment["frozen_model"][
                        "smoothness_weight"
                    ],
                    nominal_iterations=experiment["frozen_model"][
                        "nominal_iterations"
                    ],
                ),
                FullOperatorModel(),
                FewShotGenericRBF(),
            ]
            for model in models:
                started = time.perf_counter()
                fit_error = ""
                profile_index = -1
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
                    profile_index = len(profiles)
                    profiles.append(profile)
                    values = {
                        "task_error_mean_mm_equiv": float(task_error.mean()),
                        "task_endpoint_error_mean_mm_equiv": float(
                            task_endpoint.mean()
                        ),
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
                    }
                    for condition_index, key in enumerate(isolated_keys):
                        error_rows.append(
                            {
                                "seed": args.seed,
                                "subset_id": subset["subset_id"],
                                "model": model.name,
                                "episode": key,
                                "generator": test_generators[condition_index],
                                "task_error_mm_equiv": task_error[condition_index],
                                "task_endpoint_error_mm_equiv": task_endpoint[
                                    condition_index
                                ],
                            }
                        )
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
                    }
                rows.append(
                    {
                        "seed": args.seed,
                        "subset_id": subset["subset_id"],
                        "protocol": subset["protocol"],
                        "sample_size": subset["sample_size"],
                        "repeat": subset["repeat"],
                        "mixed_indices": json.dumps(subset["mixed_indices"]),
                        "rank": subset["rank"],
                        "augmented_intercept_rank": subset[
                            "augmented_intercept_rank"
                        ],
                        "condition_number": subset["condition_number"],
                        "model": model.name,
                        "fit_success": fit_error == "",
                        "fit_error": fit_error,
                        "fit_seconds": time.perf_counter() - started,
                        "profile_index": profile_index,
                        **values,
                    }
                )
            if (subset_index + 1) % 10 == 0 or subset_index + 1 == len(
                subset_manifest["subsets"]
            ):
                print(
                    f"seed {args.seed}: {subset_index + 1}/"
                    f"{len(subset_manifest['subsets'])} subsets",
                    flush=True,
                )

    result_path = output_root / "fewshot_results.csv"
    errors_path = output_root / "fewshot_condition_errors.csv"
    arrays_path = output_root / "fewshot_profiles.npz"
    summary_path = output_root / "fewshot_run.json"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    pd.DataFrame(error_rows).to_csv(errors_path, index=False)
    np.savez_compressed(
        arrays_path,
        progress=progress,
        target_profile=target_profile,
        profiles=np.asarray(profiles),
    )
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "schema_version": 1,
                "seed": args.seed,
                "dataset": str(dataset),
                "dataset_sha256": sha256(dataset),
                "experiment_sha256": sha256(args.experiment),
                "subset_manifest_sha256": sha256(args.subsets),
                "source_sha256": sha256(Path(__file__)),
                "models": MODEL_ORDER,
                "fit_count": len(rows),
                "fit_failure_count": int(
                    sum(not bool(row["fit_success"]) for row in rows)
                ),
                "generic_rbf_cv_rule": "min(5, N) episode-grouped folds; otherwise frozen implementation",
            },
            output_file,
            indent=2,
        )
    print("saved:", result_path)


if __name__ == "__main__":
    main()
