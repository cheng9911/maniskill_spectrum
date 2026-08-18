from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd

from benchmark_phase_switch_baselines import progress_grid, task_curve, usable
from phase_switch_baselines import SmoothFinitePDiagModel
from run_phase_switch_fewshot import nominal_frame


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    root = args.experiment.parent / "fewshot" / f"seed_{args.seed}"
    result_path = root / "fewshot_results.csv"
    profile_path = root / "fewshot_profiles.npz"
    results = pd.read_csv(result_path)
    stored_profiles = np.load(profile_path)["profiles"]
    pdiag_rows = results[results.model == "Pdiag finite"].sort_values("subset_id")
    dataset = args.experiment.parent / "rollouts" / f"seed_{args.seed}.h5"
    progress, phase_codes = progress_grid(args.bins)

    audit_rows = []
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
        for position, (_, row) in enumerate(pdiag_rows.iterrows(), start=1):
            subset = subset_manifest["subsets"][int(row.subset_id)]
            indices = np.asarray(subset["mixed_indices"], dtype=int)
            contexts = all_contexts[indices]
            curves = all_curves[indices]
            keys = [mixed_keys[index] for index in indices]
            frame_xy, frame_yaw = nominal_frame(data_file, keys, contexts)
            model = SmoothFinitePDiagModel(
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
            )
            started = time.perf_counter()
            model.fit(contexts, curves, progress, phase_codes)
            stored = stored_profiles[int(row.profile_index)]
            audit_rows.append(
                {
                    "seed": args.seed,
                    "subset_id": int(row.subset_id),
                    "protocol": row.protocol,
                    "sample_size": int(row.sample_size),
                    "optimization_success": model.optimization_success,
                    "optimization_nfev": model.optimization_nfev,
                    "optimization_cost": model.optimization_cost,
                    "optimization_optimality": model.optimization_optimality,
                    "profile_max_absolute_reproduction_error": float(
                        np.max(np.abs(model.jacobian_diag() - stored))
                    ),
                    "audit_fit_seconds": time.perf_counter() - started,
                }
            )
            if position % 10 == 0 or position == len(pdiag_rows):
                print(
                    f"seed {args.seed}: audited {position}/{len(pdiag_rows)}",
                    flush=True,
                )

    audit = pd.DataFrame(audit_rows)
    audit_path = root / "fewshot_pdiag_optimization_audit.csv"
    summary_path = root / "fewshot_pdiag_optimization_audit.json"
    audit.to_csv(audit_path, index=False)
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "schema_version": 1,
                "seed": args.seed,
                "source_sha256": sha256(Path(__file__)),
                "experiment_sha256": sha256(args.experiment),
                "subset_manifest_sha256": sha256(args.subsets),
                "fewshot_results_sha256": sha256(result_path),
                "audit_count": int(len(audit)),
                "optimization_success_count": int(
                    audit.optimization_success.sum()
                ),
                "max_profile_reproduction_error": float(
                    audit.profile_max_absolute_reproduction_error.max()
                ),
            },
            output_file,
            indent=2,
        )
    print("saved:", audit_path)


if __name__ == "__main__":
    main()
