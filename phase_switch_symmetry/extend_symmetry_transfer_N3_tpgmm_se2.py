from __future__ import annotations

"""Extend the symmetry-transfer benchmark to N=3 for TP-GMM SE(2) only.

The frozen symmetry-transfer subset manifest stops at N in {5, 10, 20}. This
script adds N=3 by *continuing* the frozen RNG stream (selection_seed 20260818)
exactly as prepare_symmetry_transfer_subsets.py would if 3 were appended to
SAMPLE_SIZES, so the existing 18 subsets (ids 0-17) are unchanged and the N=3
subsets get ids 18-23.

N=3 is below the main-run component grid (K starts at 4), so the TP-GMM SE(2)
fit uses the preregistered expanded-K sensitivity config
(run_phase_switch_tpgmm_fewshot_sensitivity.py): candidates (2, 3, 4, ...)
capped at N-1 -> (2,), cv_splits = min(5, N) = 3, cv_n_init = 2,
final_n_init = 5.

Metrics match benchmark_symmetry_transfer.py: M1 E_alpha, M2 discrimination,
M2b symmetry gap G_psi, M3 cross-task mismatch gap. (M0 is Pdiag-finite only,
so it is not applicable to TP-GMM SE(2).)
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_phase_switch_rollouts import pose_yaw, wrap_pi  # noqa: E402
from benchmark_phase_switch_baselines import (  # noqa: E402
    metric_errors,
    progress_grid,
    switch_diagnostics,
    task_curve,
    usable,
)
from phase_switch_baselines import TPGMMModel  # noqa: E402

import benchmark_symmetry_transfer as bst  # noqa: E402


CONTEXT_SCALE = np.array([0.012, 0.012, np.deg2rad(30.0)])
EXPANDED_K_CANDIDATES = (2, 3, 4, 6, 8, 10, 12, 16, 20)
FROZEN_SAMPLE_SIZES = [5, 10, 20]
N3 = 3
MODEL_NAME = "TP-GMM SE(2)"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostics(contexts, indices):
    selected = contexts[np.asarray(indices, dtype=int)] / CONTEXT_SCALE
    return {
        "rank": int(np.linalg.matrix_rank(selected)),
        "augmented_intercept_rank": int(
            np.linalg.matrix_rank(
                np.column_stack([np.ones(len(selected)), selected])
            )
        ),
        "condition_number": float(np.linalg.cond(selected)),
    }


def draw_unique(rng, sample_size, count, predicate=lambda _: True):
    selected = []
    seen = set()
    attempts = 0
    while len(selected) < count:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError(
                f"could not draw {count} unique subsets for N={sample_size}"
            )
        indices = tuple(sorted(rng.choice(30, sample_size, replace=False).tolist()))
        if indices in seen or not predicate(indices):
            continue
        seen.add(indices)
        selected.append(indices)
    return selected


def generate_n3_subsets(experiment, frozen_manifest):
    """Continue the frozen RNG and return the 6 N=3 subsets (ids 18-23)."""
    context_path = Path(experiment["context_manifest"])
    context_manifest = json.loads(context_path.read_text(encoding="utf-8"))
    mixed_rows = [
        row for row in context_manifest["conditions"] if row["generator"] == "mixed"
    ]
    contexts = np.asarray([row["causal_delta"] for row in mixed_rows])
    source_condition_ids = [int(row["condition_id"]) for row in mixed_rows]

    selection_seed = frozen_manifest["selection_seed"]
    repeats = frozen_manifest["repeats_per_protocol"]
    rng = np.random.default_rng(selection_seed)

    rebuilt = []
    subset_id = 0
    for sample_size in FROZEN_SAMPLE_SIZES + [N3]:
        random_subsets = draw_unique(rng, sample_size, repeats)
        qualified_subsets = draw_unique(
            rng,
            sample_size,
            repeats,
            predicate=lambda indices: (
                diagnostics(contexts, indices)["rank"] == 3
                and diagnostics(contexts, indices)["condition_number"]
                < frozen_manifest["qualification"]["condition_number_strictly_below"]
            ),
        )
        for protocol, protocol_subsets in [
            ("random", random_subsets),
            ("qualified", qualified_subsets),
        ]:
            for repeat, indices in enumerate(protocol_subsets):
                record = {
                    "subset_id": subset_id,
                    "protocol": protocol,
                    "sample_size": sample_size,
                    "repeat": repeat,
                    "mixed_indices": list(indices),
                    "source_condition_ids": [
                        source_condition_ids[index] for index in indices
                    ],
                    **diagnostics(contexts, indices),
                }
                rebuilt.append(record)
                subset_id += 1

    # Prove RNG continuation: the first 18 rebuilt subsets must equal the frozen
    # manifest byte-for-byte on the fields the manifest records.
    frozen = frozen_manifest["subsets"]
    if len(frozen) != 18:
        raise RuntimeError(f"expected 18 frozen subsets, got {len(frozen)}")
    keys = [
        "subset_id",
        "protocol",
        "sample_size",
        "repeat",
        "mixed_indices",
        "source_condition_ids",
        "rank",
        "augmented_intercept_rank",
        "condition_number",
    ]
    for i in range(18):
        frozen_rec = {k: frozen[i][k] for k in keys}
        rebuilt_rec = {k: rebuilt[i][k] for k in keys}
        if frozen_rec != rebuilt_rec:
            raise RuntimeError(
                f"RNG continuation mismatch at subset_id {i}: "
                f"frozen={frozen_rec} rebuilt={rebuilt_rec}"
            )
    n3 = rebuilt[18:]
    if [r["sample_size"] for r in n3] != [N3] * 6:
        raise RuntimeError("N=3 subsets not correctly extracted")
    print(
        "RNG continuation verified: frozen ids 0-17 unchanged; "
        f"N=3 subsets ids 18-23 = {[r['mixed_indices'] for r in n3]}",
        flush=True,
    )
    return n3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/symmetry_transfer_experiment.json"),
    )
    parser.add_argument(
        "--subsets",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/symmetry_transfer_subsets.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/symmetry_transfer"),
    )
    parser.add_argument("--bins", type=int, default=25)
    args = parser.parse_args()

    with args.experiment.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    with args.subsets.open(encoding="utf-8") as subset_file:
        subset_manifest = json.load(subset_file)
    if subset_manifest["experiment_sha256"] != sha256(args.experiment):
        raise RuntimeError("subset manifest does not match experiment")

    n3_subsets = generate_n3_subsets(experiment, subset_manifest)

    progress, phase_codes = progress_grid(args.bins)
    unlock_start = 2 * args.bins
    task_yaw_weights = (phase_codes < 2).astype(np.float64)

    datasets = {}
    for task in bst.TASK_ORDER:
        datasets[task] = {}
        for seed in bst.SEEDS:
            path = Path(bst.TASK_FILES[task].format(seed=seed))
            if not path.exists():
                raise FileNotFoundError(f"missing dataset: {path}")
            datasets[task][seed] = bst.load_dataset(path, args.bins)

    args.output_root.mkdir(parents=True, exist_ok=True)
    fit_rows = []
    cross_rows = []
    for task in bst.TASK_ORDER:
        for seed in bst.SEEDS:
            ds = datasets[task][seed]
            cid_to_idx = {cid: i for i, cid in enumerate(ds["mixed_cids"])}
            data_path = Path(bst.TASK_FILES[task].format(seed=seed))
            for subset in n3_subsets:
                source_cids = subset["source_condition_ids"]
                missing = [cid for cid in source_cids if cid not in cid_to_idx]
                if missing:
                    fit_rows.append(
                        dict(
                            task=task, seed=seed, subset_id=subset["subset_id"],
                            protocol=subset["protocol"], sample_size=N3,
                            repeat=subset["repeat"], model=MODEL_NAME,
                            fit_success=False, fit_error="missing_source_condition",
                            alpha_pre=np.nan, alpha_post=np.nan, e_alpha=np.nan,
                            switch_detected=False, switch_location=np.nan,
                        )
                    )
                    continue
                indices = [cid_to_idx[cid] for cid in source_cids]
                contexts = ds["mixed_contexts"][indices]
                curves = ds["mixed_curves"][indices]
                keys = [ds["mixed_keys"][i] for i in indices]
                with h5py.File(data_path, "r") as data_file:
                    frame_xy, frame_yaw = bst.nominal_frame(data_file, keys, contexts)

                n_episodes = len(contexts)
                candidates = tuple(
                    k
                    for k in EXPANDED_K_CANDIDATES
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
                try:
                    model.fit(contexts, curves, progress, phase_codes)
                    profile = model.jacobian_diag()
                    alpha_yaw = profile[:, 2]
                    pre = float(alpha_yaw[phase_codes < 2].mean())
                    post = float(alpha_yaw[phase_codes >= 2].mean())
                    oracle = bst.oracle_alpha_yaw(task, phase_codes)
                    e_alpha = float(np.mean((alpha_yaw - oracle) ** 2))
                    switch = switch_diagnostics(progress, alpha_yaw, unlock_start)
                    fit_success = True
                    fit_error = ""
                    if len(ds["test_contexts"]) > 0:
                        prediction = model.predict(ds["test_contexts"])
                        for other_task in bst.TASK_ORDER:
                            other = datasets[other_task][seed]
                            if other["isolated_cids"] != ds["isolated_cids"]:
                                continue
                            task_err, _ = metric_errors(
                                prediction, other["test_curves"],
                                yaw_weights=task_yaw_weights,
                            )
                            for idx, gen in enumerate(ds["test_generators"]):
                                cross_rows.append(
                                    dict(
                                        fit_task=task, eval_task=other_task,
                                        seed=seed, subset_id=subset["subset_id"],
                                        sample_size=N3, model=MODEL_NAME,
                                        generator=gen,
                                        task_error_mm_equiv=float(task_err[idx]),
                                    )
                                )
                except Exception as exc:
                    fit_success = False
                    fit_error = repr(exc)
                    pre = post = e_alpha = float("nan")
                    switch = {
                        "detected": False, "status": "fit_failed",
                        "location": float("nan"),
                    }
                fit_rows.append(
                    dict(
                        task=task, seed=seed, subset_id=subset["subset_id"],
                        protocol=subset["protocol"], sample_size=N3,
                        repeat=subset["repeat"], model=MODEL_NAME,
                        fit_success=fit_success, fit_error=fit_error,
                        fit_seconds=time.perf_counter() - started,
                        alpha_pre=pre, alpha_post=post, e_alpha=e_alpha,
                        switch_detected=bool(switch["detected"]),
                        switch_status=switch["status"],
                        switch_location=switch["location"],
                    )
                )
        print(f"{task}: fit loop done", flush=True)

    fits = pd.DataFrame(fit_rows)

    # ---- M1: E_alpha (per task, mean over seeds/repeats) ----
    m1 = (
        fits[fits.fit_success]
        .groupby(["task", "model", "sample_size"], sort=False)["e_alpha"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # ---- M2: discrimination accuracy ----
    disc = fits[fits.fit_success].copy()
    disc["pred_keyed"] = disc["alpha_pre"] > 0.5
    disc["correct"] = disc["pred_keyed"] == (disc["task"] == "keyed")
    m2 = (
        disc.groupby(["model", "sample_size"], sort=False)["correct"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "accuracy"})
        .reset_index()
    )

    # ---- M2b: symmetry gap G_psi ----
    piv = fits[fits.fit_success].pivot_table(
        index=["model", "sample_size", "seed", "subset_id"],
        columns="task", values="alpha_pre",
    ).reset_index()
    gap_rows = []
    for arm in ["circular_honest", "circular_placebo"]:
        if arm not in piv.columns:
            continue
        for _, row in piv.iterrows():
            if not np.isfinite(row["keyed"]) or not np.isfinite(row[arm]):
                continue
            gap = float(row["keyed"] - row[arm])
            gap_rows.append(
                dict(
                    model=row["model"], sample_size=row["sample_size"],
                    seed=row["seed"], subset_id=row["subset_id"],
                    arm=arm, G_psi=gap, abs_dev_from_1=abs(gap - 1.0),
                )
            )
    gap_df = pd.DataFrame(gap_rows)
    m2b = (
        gap_df.groupby(["model", "sample_size", "arm"], sort=False)
        .agg(
            G_psi_mean=("G_psi", "mean"), G_psi_std=("G_psi", "std"),
            abs_dev_mean=("abs_dev_from_1", "mean"),
        )
        .reset_index()
    )

    # ---- M3: cross-task mismatch gap ----
    cross = pd.DataFrame(cross_rows)
    m3_rows = []
    if len(cross):
        in_task = (
            cross[cross.fit_task == cross.eval_task]
            .groupby(["model", "sample_size", "seed"], sort=False)["task_error_mm_equiv"]
            .mean()
        )
        cross_task = (
            cross[cross.fit_task != cross.eval_task]
            .groupby(["model", "sample_size", "seed", "fit_task", "eval_task"], sort=False)[
                "task_error_mm_equiv"
            ]
            .mean()
        )
        for (model, n, seed, fit_task, eval_task), cross_err in cross_task.items():
            in_err = in_task.get((model, n, seed))
            if in_err is None or not np.isfinite(in_err):
                continue
            m3_rows.append(
                dict(
                    model=model, sample_size=n, seed=seed,
                    fit_task=fit_task, eval_task=eval_task,
                    cross_task_error=cross_err, in_task_error=in_err,
                    mismatch_gap=cross_err - in_err,
                )
            )
    m3 = pd.DataFrame(m3_rows)

    fits_path = args.output_root / "symmetry_transfer_N3_tpgmm_se2_fits.csv"
    m1_path = args.output_root / "symmetry_transfer_N3_tpgmm_se2_M1.csv"
    m2_path = args.output_root / "symmetry_transfer_N3_tpgmm_se2_M2.csv"
    m2b_path = args.output_root / "symmetry_transfer_N3_tpgmm_se2_M2b.csv"
    m3_path = args.output_root / "symmetry_transfer_N3_tpgmm_se2_M3.csv"
    fits.to_csv(fits_path, index=False)
    m1.to_csv(m1_path, index=False)
    m2.to_csv(m2_path, index=False)
    m2b.to_csv(m2b_path, index=False)
    m3.to_csv(m3_path, index=False)

    summary = {
        "schema_version": 1,
        "extension": "N=3 TP-GMM SE(2) only",
        "experiment_sha256": sha256(args.experiment),
        "subset_manifest_sha256": sha256(args.subsets),
        "source_sha256": sha256(Path(__file__)),
        "model": MODEL_NAME,
        "config": {
            "component_candidates_rule": (
                "expanded-K (2,3,4,6,8,10,12,16,20) capped at N-1 -> (2,) at N=3"
            ),
            "cv_splits": "min(5, N) = 3",
            "cv_n_init": 2,
            "final_n_init": 5,
        },
        "n3_subsets": n3_subsets,
        "M1_generator_identification_error": m1.to_dict(orient="records"),
        "M2_discrimination_accuracy": m2.to_dict(orient="records"),
        "M2b_symmetry_gap": m2b.to_dict(orient="records"),
        "M3_cross_task_mismatch_gap": m3.to_dict(orient="records"),
        "fit_count": int(len(fits)),
        "fit_success_count": int(fits.fit_success.sum()),
        "fit_failure_count": int((~fits.fit_success).sum()),
        "bins_per_phase": args.bins,
    }
    summary_path = args.output_root / "symmetry_transfer_N3_tpgmm_se2_summary.json"
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(bst.json_ready(summary), output_file, indent=2, allow_nan=False)

    print("\n=== M1 E_alpha (TP-GMM SE(2), N=3) ===")
    print(m1.to_string(index=False))
    print("\n=== M2 discrimination accuracy (TP-GMM SE(2), N=3) ===")
    print(m2.to_string(index=False))
    print("\n=== M2b symmetry gap G_psi (TP-GMM SE(2), N=3) ===")
    print(m2b.to_string(index=False))
    print("\n=== M3 cross-task mismatch gap (TP-GMM SE(2), N=3) ===")
    print(m3.to_string(index=False))
    print(f"\nfit failures: {int((~fits.fit_success).sum())}")
    if (~fits.fit_success).any():
        print(fits[~fits.fit_success][["task", "seed", "subset_id", "fit_error"]].to_string(index=False))
    print("\nsaved:", summary_path)


if __name__ == "__main__":
    main()
