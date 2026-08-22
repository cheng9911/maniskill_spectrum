from __future__ import annotations

"""Multi-generator SE(3) benchmark (supplement).

The decisive question: in the multigen task (keyed gate -> rectangular slot),
where the expert relaxes TWO generators at unlock (yaw -> 0 AND du -> 0, while
dv stays tracked), does the model recover BOTH selective generators rather
than a single winner? The generator relevance is a VECTOR, not an argmax.

Oracle over the four insertion phases (align_keyed/enter_key/unlock_yaw/
circular_insert):

    du   [1,1,0,0]    (selective)
    dv   [1,1,1,1]    (tracked: the slot's narrow y does not admit a full dv relaxation)
    dw   [1,1,1,1]
    roll [1,1,1,1]
    pitch[1,1,1,1]
    yaw  [1,1,0,0]    (selective)

Headline metric M-multi: rank the six generators by their post-clearance
relevance drop Delta_j = mean(alpha_j | phase<2) - mean(alpha_j | phase>=2).
Correct iff the top-2 set is exactly {du, yaw}, both with Delta > 0.2, and all
four others have Delta < 0.2. The frozen keyed SE(3) arm (single-generator)
is re-scored from its frozen fits CSV as the control: there, Delta_du ~= 0.
"""

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from benchmark_phase_switch_baselines import (
    PHASE_CODES,
    progress_grid,
    switch_diagnostics,
    usable,
)
from benchmark_se3_transfer import (
    SEEDS,
    SE3_MODEL_ORDER,
    build_models,
    json_ready,
    load_dataset,
    nominal_frame_se3,
    sha256,
)
from phase_switch_se3_baselines import SE3_DIM

TASK_ORDER = ("multigen",)
TASK_FILES = {
    "multigen": "phase_switch_symmetry_rollouts_se3/multigen_seed_{seed}.h5",
}
GENERATOR_NAMES = ("du", "dv", "dw", "roll", "pitch", "yaw")
DU_INDEX = 0
YAW_INDEX = 5
MULTI_TAU = 0.2  # clearance-drop threshold for "selective"


def oracle_alpha_multigen(phase_codes):
    """6 x T generator-law oracle: du AND yaw are both selective."""
    phase_codes = np.asarray(phase_codes, dtype=int)
    oracle = np.ones((len(phase_codes), SE3_DIM), dtype=np.float64)
    selective = (phase_codes < 2).astype(np.float64)
    oracle[:, DU_INDEX] = selective
    oracle[:, YAW_INDEX] = selective
    return oracle


def m_multi_correct(delta):
    """Top-2 = {du, yaw}, both > tau, the other four all < tau."""
    delta = np.asarray(delta, dtype=np.float64)
    top2 = set(np.argsort(-delta)[:2].tolist())
    others = [j for j in range(SE3_DIM) if j not in (DU_INDEX, YAW_INDEX)]
    return bool(
        top2 == {DU_INDEX, YAW_INDEX}
        and delta[DU_INDEX] > MULTI_TAU
        and delta[YAW_INDEX] > MULTI_TAU
        and max(delta[j] for j in others) < MULTI_TAU
    )


_DATASETS = None
_PDIAG_CONFIG = None
_PROGRESS = None
_PHASE_CODES = None
_PRE_MASK = None
_UNLOCK_START = None


def _fit_subset(job):
    task, seed, subset = job
    ds = _DATASETS[task][seed]
    cid_to_idx = {cid: i for i, cid in enumerate(ds["mixed_cids"])}
    source_cids = subset["source_condition_ids"]
    missing = [cid for cid in source_cids if cid not in cid_to_idx]
    if missing:
        return [
            dict(task=task, seed=seed, subset_id=subset["subset_id"],
                 protocol=subset["protocol"], sample_size=subset["sample_size"],
                 repeat=subset["repeat"], model=model,
                 fit_success=False, fit_error="missing_source_condition",
                 e_alpha=np.nan, m_multi_correct=False,
                 du_switch_detected=False, du_switch_location=np.nan,
                 yaw_switch_detected=False, yaw_switch_location=np.nan,
                 **{f"alpha_pre_{g}": np.nan for g in GENERATOR_NAMES},
                 **{f"delta_{g}": np.nan for g in GENERATOR_NAMES})
            for model in SE3_MODEL_ORDER
        ], []
    indices = [cid_to_idx[cid] for cid in source_cids]
    contexts = ds["mixed_contexts"][indices]
    curves = ds["mixed_curves"][indices]
    keys = [ds["mixed_keys"][i] for i in indices]
    try:
        with h5py.File(Path(TASK_FILES[task].format(seed=seed)), "r") as data_file:
            nominal_frame_pose = nominal_frame_se3(data_file, keys, contexts)
        models = build_models(nominal_frame_pose, _PDIAG_CONFIG)
    except Exception as exc:
        return [
            dict(task=task, seed=seed, subset_id=subset["subset_id"],
                 protocol=subset["protocol"], sample_size=subset["sample_size"],
                 repeat=subset["repeat"], model=model,
                 fit_success=False, fit_error=f"nominal_frame:{repr(exc)}",
                 e_alpha=np.nan, m_multi_correct=False,
                 du_switch_detected=False, du_switch_location=np.nan,
                 yaw_switch_detected=False, yaw_switch_location=np.nan,
                 **{f"alpha_pre_{g}": np.nan for g in GENERATOR_NAMES},
                 **{f"delta_{g}": np.nan for g in GENERATOR_NAMES})
            for model in SE3_MODEL_ORDER
        ], []

    rows = []
    profiles = []
    for model in models:
        started = time.perf_counter()
        try:
            model.fit(contexts, curves, _PROGRESS, _PHASE_CODES)
            profile = model.jacobian_diag()  # (n_steps, 6)
            oracle = oracle_alpha_multigen(_PHASE_CODES)
            e_alpha = float(np.mean((profile - oracle) ** 2))
            alpha_pre_vec = profile[_PRE_MASK].mean(axis=0)
            alpha_post_vec = profile[~_PRE_MASK].mean(axis=0)
            delta = alpha_pre_vec - alpha_post_vec
            du_switch = switch_diagnostics(
                _PROGRESS, profile[:, DU_INDEX], _UNLOCK_START)
            yaw_switch = switch_diagnostics(
                _PROGRESS, profile[:, YAW_INDEX], _UNLOCK_START)
            fit_success, fit_error = True, ""
            if model.name == "Pdiag finite (SE(3))":
                profiles.append(
                    dict(task=task, seed=seed, subset_id=subset["subset_id"],
                         sample_size=subset["sample_size"], profile=profile)
                )
        except Exception as exc:
            fit_success, fit_error = False, repr(exc)
            e_alpha = float("nan")
            alpha_pre_vec = np.full(SE3_DIM, np.nan)
            delta = np.full(SE3_DIM, np.nan)
            du_switch = {"detected": False, "status": "fit_failed", "location": float("nan")}
            yaw_switch = {"detected": False, "status": "fit_failed", "location": float("nan")}
        rows.append(
            dict(task=task, seed=seed, subset_id=subset["subset_id"],
                 protocol=subset["protocol"], sample_size=subset["sample_size"],
                 repeat=subset["repeat"], model=model.name,
                 fit_success=fit_success, fit_error=fit_error,
                 fit_seconds=time.perf_counter() - started,
                 e_alpha=e_alpha,
                 m_multi_correct=bool(m_multi_correct(delta)) if fit_success else False,
                 du_switch_detected=bool(du_switch["detected"]),
                 du_switch_location=du_switch["location"],
                 yaw_switch_detected=bool(yaw_switch["detected"]),
                 yaw_switch_location=yaw_switch["location"],
                 **{f"alpha_pre_{GENERATOR_NAMES[j]}": float(alpha_pre_vec[j]) for j in range(SE3_DIM)},
                 **{f"delta_{GENERATOR_NAMES[j]}": float(delta[j]) for j in range(SE3_DIM)},
            )
        )
    return rows, profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/se3_experiment.json"),
    )
    parser.add_argument(
        "--subsets",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/se3_subsets.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/se3_multigen"),
    )
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--pdiag-alpha-max", type=float, default=1.25)
    parser.add_argument("--pdiag-basis-count", type=int, default=24)
    parser.add_argument("--pdiag-basis-width", type=float, default=0.065)
    parser.add_argument("--pdiag-smoothness", type=float, default=0.1)
    parser.add_argument("--pdiag-nominal-iterations", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=0)
    args = parser.parse_args()

    pdiag_config = {
        "alpha_max": args.pdiag_alpha_max,
        "n_basis": args.pdiag_basis_count,
        "basis_width": args.pdiag_basis_width,
        "smoothness_weight": args.pdiag_smoothness,
        "nominal_iterations": args.pdiag_nominal_iterations,
    }
    with args.experiment.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    with args.subsets.open(encoding="utf-8") as subset_file:
        subset_manifest = json.load(subset_file)
    if subset_manifest["experiment_sha256"] != sha256(args.experiment):
        raise RuntimeError("subset manifest does not match experiment")

    progress, phase_codes = progress_grid(args.bins)
    unlock_start = 2 * args.bins
    pre_mask = phase_codes < 2

    datasets = {}
    for task in TASK_ORDER:
        datasets[task] = {}
        for seed in SEEDS:
            path = Path(TASK_FILES[task].format(seed=seed))
            if not path.exists():
                raise FileNotFoundError(f"missing dataset: {path}")
            datasets[task][seed] = load_dataset(path, args.bins)

    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (task, seed, subset)
        for task in TASK_ORDER
        for seed in SEEDS
        for subset in subset_manifest["subsets"]
    ]

    global _DATASETS, _PDIAG_CONFIG, _PROGRESS, _PHASE_CODES, _PRE_MASK, _UNLOCK_START
    _DATASETS = datasets
    _PDIAG_CONFIG = pdiag_config
    _PROGRESS = progress
    _PHASE_CODES = phase_codes
    _PRE_MASK = pre_mask
    _UNLOCK_START = unlock_start

    n_jobs = args.jobs or min(16, os.cpu_count() or 1)
    fit_rows = []
    profile_accum = []
    if n_jobs > 1:
        import multiprocessing
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            for rows, profiles in pool.imap_unordered(_fit_subset, jobs):
                fit_rows.extend(rows)
                profile_accum.extend(profiles)
    else:
        for job in jobs:
            rows, profiles = _fit_subset(job)
            fit_rows.extend(rows)
            profile_accum.extend(profiles)

    fits = pd.DataFrame(fit_rows)
    ok = fits[fits.fit_success]

    # ---- headline: M-multi accuracy (top-2 = {du, yaw}) ----
    m_multi = (
        ok.groupby(["model", "sample_size"], sort=False)["m_multi_correct"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "accuracy"})
        .reset_index()
    )

    # ---- per-generator clearance drop Delta_j ----
    delta_cols = [f"delta_{g}" for g in GENERATOR_NAMES]
    delta_means = (
        ok.groupby(["model", "sample_size"], sort=False)[delta_cols]
        .mean()
        .reset_index()
    )

    # ---- per-generator pre-clearance relevance alpha_pre_j ----
    alpha_pre_cols = [f"alpha_pre_{g}" for g in GENERATOR_NAMES]
    alpha_pre_means = (
        ok.groupby(["model", "sample_size"], sort=False)[alpha_pre_cols]
        .mean()
        .reset_index()
    )

    # ---- E_alpha vs the multigen oracle ----
    e_alpha = (
        ok.groupby(["model", "sample_size"], sort=False)["e_alpha"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # ---- M0 switch fidelity for BOTH selective generators (Pdiag finite) ----
    pdiag_ok = ok[ok.model == "Pdiag finite (SE(3))"].copy()
    pdiag_ok["du_switch_dev"] = (pdiag_ok["du_switch_location"] - 0.5).abs()
    pdiag_ok["yaw_switch_dev"] = (pdiag_ok["yaw_switch_location"] - 0.5).abs()
    m0 = (
        pdiag_ok.groupby(["model", "sample_size"], sort=False)
        .agg(
            du_switch_fraction=("du_switch_detected", "mean"),
            du_switch_location_mean=("du_switch_location", "mean"),
            yaw_switch_fraction=("yaw_switch_detected", "mean"),
            yaw_switch_location_mean=("yaw_switch_location", "mean"),
        )
        .reset_index()
    )

    # ---- control: re-score the FROZEN keyed single-generator arm with the
    # same Delta_j machinery (read from its frozen fits CSV; no new fits) ----
    keyed_control = pd.DataFrame()
    frozen_fits_path = Path("phase_switch_symmetry_multiseed/se3_transfer/se3_transfer_fits.csv")
    if frozen_fits_path.exists():
        frozen = pd.read_csv(frozen_fits_path)
        frozen_keyed = frozen[
            (frozen.task == "keyed") & (frozen.model == "Pdiag finite (SE(3))")
            & frozen.fit_success
        ]
        if len(frozen_keyed):
            frozen_keyed = frozen_keyed.copy()
            frozen_keyed["m_multi_correct"] = frozen_keyed[
                delta_cols
            ].apply(lambda row: m_multi_correct(row.to_numpy()), axis=1)
            keyed_control = (
                frozen_keyed.groupby(["model", "sample_size"], sort=False)
                .agg(
                    m_multi_accuracy=("m_multi_correct", "mean"),
                    **{f"{c}_mean": (c, "mean") for c in delta_cols},
                )
                .reset_index()
            )

    # ---- write outputs ----
    fits.to_csv(args.output_root / "multigen_fits.csv", index=False)
    m_multi.to_csv(args.output_root / "multigen_M_multi.csv", index=False)
    delta_means.to_csv(args.output_root / "multigen_delta_means.csv", index=False)
    alpha_pre_means.to_csv(args.output_root / "multigen_alpha_pre_means.csv", index=False)
    e_alpha.to_csv(args.output_root / "multigen_e_alpha.csv", index=False)
    m0.to_csv(args.output_root / "multigen_M0_switch.csv", index=False)
    if len(keyed_control):
        keyed_control.to_csv(args.output_root / "multigen_keyed_control.csv", index=False)
    if profile_accum:
        np.savez(
            args.output_root / "multigen_profiles.npz",
            profile=np.stack([p["profile"] for p in profile_accum]),
            task=np.asarray([p["task"] for p in profile_accum]),
            seed=np.asarray([p["seed"] for p in profile_accum]),
            subset_id=np.asarray([p["subset_id"] for p in profile_accum]),
            sample_size=np.asarray([p["sample_size"] for p in profile_accum]),
            progress=progress,
            phase_codes=phase_codes,
        )

    summary = {
        "schema_version": 1,
        "experiment_sha256": sha256(args.experiment),
        "subset_manifest_sha256": sha256(args.subsets),
        "source_sha256": {
            "benchmark_se3_multigen.py": sha256(Path(__file__)),
            "benchmark_se3_transfer.py": sha256(
                Path(__file__).with_name("benchmark_se3_transfer.py")
            ),
            "phase_switch_se3_baselines.py": sha256(
                Path(__file__).with_name("phase_switch_se3_baselines.py")
            ),
        },
        "oracle": {
            "multigen": "du/yaw = 1 in align/enter (phase<2), 0 in unlock/insert; dv/dw/roll/pitch = 1 everywhere",
        },
        "M_multi": {
            "definition": (
                "rank generators by Delta_j = mean(alpha_j|phase<2) - "
                "mean(alpha_j|phase>=2); correct iff top-2 == {du, yaw}, both "
                "Delta > 0.2, and the other four all < 0.2"
            ),
            "tau": MULTI_TAU,
            "oracle": {"delta_du": 1.0, "delta_yaw": 1.0, "delta_others": 0.0},
        },
        "M_multi_accuracy": m_multi.to_dict(orient="records"),
        "delta_means": delta_means.to_dict(orient="records"),
        "alpha_pre_means": alpha_pre_means.to_dict(orient="records"),
        "e_alpha": e_alpha.to_dict(orient="records"),
        "M0_switch_fidelity": m0.to_dict(orient="records"),
        "keyed_arm_control": (
            keyed_control.to_dict(orient="records") if len(keyed_control) else []
        ),
        "align_phase_transient_note": (
            "As in the frozen SE(3) benchmark, the align phase carries a 0->1 "
            "rise, so alpha_pre means land at ~0.7 rather than the oracle 1.0; "
            "M-multi is threshold-free except for the 0.2 clearance-drop tau."
        ),
        "fit_count": int(len(fits)),
        "fit_failure_count": int((~fits.fit_success).sum()),
        "bins_per_phase": args.bins,
    }
    with (args.output_root / "multigen_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(json_ready(summary), output_file, indent=2, allow_nan=False)

    print("\n=== M-multi accuracy (top-2 = {du, yaw}) ===")
    print(m_multi.to_string(index=False))
    print("\n=== per-generator clearance drop Delta_j (means) ===")
    print(delta_means.to_string(index=False))
    print("\n=== per-generator pre-clearance relevance alpha_pre_j (means) ===")
    print(alpha_pre_means.to_string(index=False))
    print("\n=== E_alpha vs multigen oracle ===")
    print(e_alpha.to_string(index=False))
    print("\n=== M0 switch fidelity, du and yaw (Pdiag finite) ===")
    print(m0.to_string(index=False))
    print("\n=== keyed single-generator control (frozen fits, re-scored) ===")
    print(keyed_control.to_string(index=False) if len(keyed_control) else "(no frozen fits found)")
    print("\nsaved:", args.output_root / "multigen_summary.json")


if __name__ == "__main__":
    main()
