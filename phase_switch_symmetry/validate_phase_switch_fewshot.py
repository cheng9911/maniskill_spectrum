from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = {"Pdiag finite", "Full operator", "Generic RBF"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


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
        "--results",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/fewshot_results"),
    )
    args = parser.parse_args()

    with args.experiment.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    with args.subsets.open(encoding="utf-8") as subset_file:
        subset_manifest = json.load(subset_file)
    require(
        subset_manifest["status"] == "frozen_before_fewshot_fitting",
        "subset manifest was not frozen before fitting",
    )
    require(
        subset_manifest["experiment_sha256"] == sha256(args.experiment),
        "subset manifest experiment hash mismatch",
    )
    require(
        subset_manifest["policies"]["random_protocol_drops_no_subsets"],
        "random-subset retention policy is missing",
    )
    subsets = {int(row["subset_id"]): row for row in subset_manifest["subsets"]}
    expected_fits = len(subsets) * len(MODELS)
    all_frames = []
    for seed in experiment["seeds"]:
        root = args.experiment.parent / "fewshot" / f"seed_{seed}"
        run_path = root / "fewshot_run.json"
        result_path = root / "fewshot_results.csv"
        error_path = root / "fewshot_condition_errors.csv"
        profile_path = root / "fewshot_profiles.npz"
        audit_path = root / "fewshot_pdiag_optimization_audit.csv"
        audit_summary_path = root / "fewshot_pdiag_optimization_audit.json"
        with run_path.open(encoding="utf-8") as run_file:
            run = json.load(run_file)
        require(run["seed"] == seed, f"seed metadata mismatch for {seed}")
        require(
            run["source_sha256"]
            == sha256(Path(__file__).with_name("run_phase_switch_fewshot.py")),
            f"few-shot runner source changed for {seed}",
        )
        require(
            run["experiment_sha256"] == sha256(args.experiment),
            f"experiment hash mismatch for {seed}",
        )
        require(
            run["subset_manifest_sha256"] == sha256(args.subsets),
            f"subset hash mismatch for {seed}",
        )
        require(
            run["dataset_sha256"] == sha256(Path(run["dataset"])),
            f"dataset hash mismatch for {seed}",
        )
        require(run["fit_count"] == expected_fits, f"missing fits for {seed}")
        require(set(run["models"]) == MODELS, f"model set mismatch for {seed}")
        frame = pd.read_csv(result_path)
        require(len(frame) == expected_fits, f"result row count mismatch for {seed}")
        require(set(frame.model) == MODELS, f"result models mismatch for {seed}")
        require(frame.seed.eq(seed).all(), f"result seed mismatch for {seed}")
        require(
            frame.groupby("subset_id").model.nunique().eq(len(MODELS)).all(),
            f"not every subset has every model for {seed}",
        )
        for subset_id, rows in frame.groupby("subset_id"):
            expected = subsets[int(subset_id)]
            require(
                rows.protocol.eq(expected["protocol"]).all(),
                f"protocol mismatch for seed {seed}, subset {subset_id}",
            )
            require(
                rows.sample_size.eq(expected["sample_size"]).all(),
                f"sample size mismatch for seed {seed}, subset {subset_id}",
            )
            require(
                rows["rank"].eq(expected["rank"]).all(),
                f"rank mismatch for seed {seed}, subset {subset_id}",
            )
            require(
                np.allclose(rows.condition_number, expected["condition_number"]),
                f"condition mismatch for seed {seed}, subset {subset_id}",
            )
            require(
                rows.mixed_indices.nunique() == 1
                and json.loads(rows.mixed_indices.iloc[0])
                == expected["mixed_indices"],
                f"indices mismatch for seed {seed}, subset {subset_id}",
            )
        qualified = frame[frame.protocol == "qualified"]
        threshold = subset_manifest["qualification"][
            "condition_number_strictly_below"
        ]
        require(qualified["rank"].eq(3).all(), f"qualified rank failure for {seed}")
        require(
            qualified.condition_number.lt(threshold).all(),
            f"qualified condition failure for {seed}",
        )
        failures = frame[~frame.fit_success]
        require(
            len(failures) == run["fit_failure_count"],
            f"fit failure count mismatch for {seed}",
        )
        if len(failures):
            require(
                failures.fit_error.fillna("").astype(str).str.len().gt(0).all(),
                f"failed fits lack diagnostics for {seed}",
            )
        successful = frame[frame.fit_success]
        errors = pd.read_csv(error_path)
        require(
            len(errors) == 8 * len(successful),
            f"condition errors do not match successful fits for {seed}",
        )
        profiles = np.load(profile_path)
        profile_indices = successful.profile_index.astype(int).to_numpy()
        require(
            sorted(profile_indices.tolist()) == list(range(len(profiles["profiles"]))),
            f"profile index mismatch for {seed}",
        )
        audit = pd.read_csv(audit_path)
        with audit_summary_path.open(encoding="utf-8") as audit_file:
            audit_summary = json.load(audit_file)
        pdiag_count = int((frame.model == "Pdiag finite").sum())
        require(len(audit) == pdiag_count, f"Pdiag audit count mismatch for {seed}")
        require(
            audit_summary["optimization_success_count"]
            == int(audit.optimization_success.sum()),
            f"Pdiag optimizer audit summary mismatch for seed {seed}",
        )
        require(
            audit.profile_max_absolute_reproduction_error.max() < 1e-10,
            f"Pdiag audit does not reproduce saved profiles for seed {seed}",
        )
        require(
            audit_summary["fewshot_results_sha256"] == sha256(result_path),
            f"Pdiag audit result hash mismatch for seed {seed}",
        )
        require(
            audit_summary["source_sha256"]
            == sha256(
                Path(__file__).with_name("audit_phase_switch_fewshot_pdiag.py")
            ),
            f"Pdiag audit source changed for seed {seed}",
        )
        all_frames.append(frame)

    combined = pd.concat(all_frames, ignore_index=True)
    for subset_id, rows in combined.groupby("subset_id"):
        require(
            rows.mixed_indices.nunique() == 1,
            f"subset {subset_id} changed across execution seeds",
        )

    summary_path = args.results / "fewshot_summary.json"
    with summary_path.open(encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    require(
        summary["experiment_sha256"] == sha256(args.experiment),
        "analysis experiment hash mismatch",
    )
    require(
        summary["subset_manifest_sha256"] == sha256(args.subsets),
        "analysis subset hash mismatch",
    )
    require(
        summary["analysis_source_sha256"]
        == sha256(Path(__file__).with_name("analyze_phase_switch_fewshot.py")),
        "few-shot analysis source changed after result generation",
    )
    require(summary["fit_count"] == len(combined), "analysis fit count mismatch")
    require(
        summary["fit_failure_count"] == int((~combined.fit_success).sum()),
        "analysis failure count mismatch",
    )
    paired = pd.read_csv(args.results / "fewshot_paired_comparisons.csv")
    require(
        set(paired.competitor) == {"Full operator", "Generic RBF"},
        "paired comparison model mismatch",
    )
    require(
        paired.pdiag_win_fraction.between(0, 1).all(),
        "invalid paired win fraction",
    )
    require(
        (args.results / "phase_switch_fewshot.png").stat().st_size > 10000,
        "few-shot PNG appears empty",
    )
    require(
        (args.results / "phase_switch_fewshot.pdf").stat().st_size > 10000,
        "few-shot PDF appears empty",
    )
    print("All few-shot validation checks passed.")


if __name__ == "__main__":
    main()
