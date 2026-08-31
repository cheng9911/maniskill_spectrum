from __future__ import annotations

"""Benchmark SE(3) relevance recovery on the LIBERO drawer relation probe."""

import argparse
import json
import os
import time
from pathlib import Path

# Multiprocessing over subsets is faster only if BLAS does not spawn a thread
# pool inside every worker.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import h5py
import numpy as np
import pandas as pd
from transforms3d.quaternions import quat2mat

from analyze_phase_switch_rollouts import resample
from benchmark_phase_switch_baselines import PHASE_CODES, progress_grid
from benchmark_se3_transfer import (
    SE3_MODEL_ORDER,
    build_models,
    json_ready,
    pose_to_pose6,
    sha256,
)
from phase_switch_baselines import TPGMMModel
from phase_switch_se3_baselines import (
    SE3_DIM,
    euler_from_matrix,
    pose6_from_se3,
    se3_from_pose6,
    se3_inverse,
)


SEEDS = [20260818, 20270818, 20280818]
TASK = "libero_drawer"
TASK_FILES = {
    TASK: "phase_switch_symmetry_rollouts_libero_drawer/drawer_seed_{seed}.h5",
}
GENERATOR_NAMES = ("du", "dv", "dw", "roll", "pitch", "yaw")
SE2_GENERATOR_NAMES = ("du", "dv", "yaw")
DU_INDEX = 0
ACTIVE_CODES = (1, 2)
THRESHOLD = 0.5
TPGMM_MODEL_ORDER = ("TP-GMM additive", "TP-GMM SE(2)")


def usable_drawer(group) -> bool:
    return (
        bool(np.asarray(group["success"])[-1])
        and not bool(np.asarray(group["truncated"]).any())
    )


def _phase_object_pose6(group, phase, bins):
    phases = np.asarray(group["solver_phase"])
    indices = np.flatnonzero(phases == phase)
    pose = np.asarray(group["object_pose"])[indices]
    rpy = np.stack([euler_from_matrix(quat2mat(q)) for q in pose[:, 3:]])
    rpy = np.unwrap(rpy, axis=0)
    pose6 = np.column_stack([pose[:, 0], pose[:, 1], pose[:, 2], rpy])
    return resample(pose6, bins)


def drawer_curve(group, bins):
    return np.concatenate(
        [_phase_object_pose6(group, phase, bins) for phase in PHASE_CODES], axis=0
    )


def drawer_curve_se2(group, bins):
    curve = drawer_curve(group, bins)
    return curve[:, [0, 1, 5]]


def nominal_frame_drawer(data_file, keys, contexts):
    """Recover C0 from mixed ghost target poses: C0 = target o intervention^-1."""
    pose6s = []
    for key, context in zip(keys, contexts):
        target_pose = np.asarray(data_file[key]["target_pose"])[0]
        target_pose6 = pose_to_pose6(target_pose)
        target_T = se3_from_pose6(target_pose6)
        intervention_T = se3_from_pose6(np.asarray(context, dtype=np.float64))
        nominal_T = target_T @ se3_inverse(intervention_T)
        pose6s.append(pose6_from_se3(nominal_T))
    pose6s = np.asarray(pose6s)
    nominal = np.empty(SE3_DIM, dtype=np.float64)
    nominal[:3] = pose6s[:, :3].mean(axis=0)
    for generator in range(3):
        angle = pose6s[:, 3 + generator]
        nominal[3 + generator] = float(
            np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())
        )
    return nominal


def oracle_alpha_drawer(phase_codes):
    """Exact phasewise oracle in the rail-aligned task basis.

    The drawer is stationary during reach, moves linearly along the rail during
    slide_open, and then holds the target displacement during hold/retract.
    """
    phase_codes = np.asarray(phase_codes, dtype=int)
    oracle = np.zeros((len(phase_codes), SE3_DIM), dtype=np.float64)
    open_indices = np.flatnonzero(phase_codes == 1)
    if len(open_indices):
        oracle[open_indices, DU_INDEX] = np.linspace(0.0, 1.0, len(open_indices))
    oracle[phase_codes >= 2, DU_INDEX] = 1.0
    return oracle


def oracle_alpha_drawer_se2(phase_codes):
    phase_codes = np.asarray(phase_codes, dtype=int)
    oracle = np.zeros((len(phase_codes), 3), dtype=np.float64)
    open_indices = np.flatnonzero(phase_codes == 1)
    if len(open_indices):
        oracle[open_indices, 0] = np.linspace(0.0, 1.0, len(open_indices))
    oracle[phase_codes >= 2, 0] = 1.0
    return oracle


def m_prismatic_correct(alpha_active_max):
    return bool(
        alpha_active_max[DU_INDEX] > THRESHOLD
        and all(
            alpha_active_max[j] < THRESHOLD
            for j in range(SE3_DIM)
            if j != DU_INDEX
        )
    )


def m_prismatic_projected_correct(alpha_active_max):
    return bool(alpha_active_max[0] > THRESHOLD and np.all(alpha_active_max[1:] < THRESHOLD))


def load_drawer_dataset(path: Path, bins: int):
    with h5py.File(path, "r") as data_file:
        usable_groups = {
            int(data_file[key].attrs["condition_id"]): key
            for key in data_file
            if key.startswith("episode_") and usable_drawer(data_file[key])
        }
        mixed_cond = {
            cid: key
            for cid, key in usable_groups.items()
            if str(data_file[key].attrs["generator"]) == "mixed"
        }
        mixed_cids = sorted(mixed_cond)
        mixed_keys = [mixed_cond[cid] for cid in mixed_cids]
        mixed_contexts = np.asarray(
            [np.asarray(data_file[key]["causal_delta"]) for key in mixed_keys]
        )
        mixed_curves = np.asarray(
            [drawer_curve(data_file[key], bins) for key in mixed_keys]
        )
        mixed_curves_se2 = np.asarray(
            [drawer_curve_se2(data_file[key], bins) for key in mixed_keys]
        )
    return {
        "mixed_cids": mixed_cids,
        "mixed_keys": mixed_keys,
        "mixed_contexts": mixed_contexts,
        "mixed_curves": mixed_curves,
        "mixed_contexts_se2": mixed_contexts[:, [0, 1, 5]],
        "mixed_curves_se2": mixed_curves_se2,
    }


_DATASETS = None
_PDIAG_CONFIG = None
_PROGRESS = None
_PHASE_CODES = None
_ACTIVE_MASK = None


def _fit_subset(job):
    seed, subset = job
    ds = _DATASETS[seed]
    cid_to_idx = {cid: i for i, cid in enumerate(ds["mixed_cids"])}
    source_cids = subset["source_condition_ids"]

    def placeholder(model, error):
        return dict(
            task=TASK,
            seed=seed,
            subset_id=subset["subset_id"],
            protocol=subset["protocol"],
            sample_size=subset["sample_size"],
            repeat=subset["repeat"],
            model=model,
            fit_success=False,
            fit_error=error,
            e_alpha=np.nan,
            m_prismatic_correct=False,
            **{f"alpha_active_{GENERATOR_NAMES[j]}": np.nan for j in range(SE3_DIM)},
        )

    missing = [cid for cid in source_cids if cid not in cid_to_idx]
    if missing:
        return [placeholder(model, "missing_source_condition") for model in SE3_MODEL_ORDER], []

    indices = [cid_to_idx[cid] for cid in source_cids]
    contexts = ds["mixed_contexts"][indices]
    curves = ds["mixed_curves"][indices]
    keys = [ds["mixed_keys"][i] for i in indices]
    try:
        with h5py.File(Path(TASK_FILES[TASK].format(seed=seed)), "r") as data_file:
            nominal_frame_pose = nominal_frame_drawer(data_file, keys, contexts)
        models = build_models(nominal_frame_pose, _PDIAG_CONFIG)
    except Exception as exc:
        return [placeholder(model, f"nominal_frame:{repr(exc)}") for model in SE3_MODEL_ORDER], []

    oracle = oracle_alpha_drawer(_PHASE_CODES)
    rows = []
    profiles = []
    for model in models:
        started = time.perf_counter()
        try:
            model.fit(contexts, curves, _PROGRESS, _PHASE_CODES)
            profile = model.jacobian_diag()
            e_alpha = float(np.mean((profile - oracle) ** 2))
            alpha_active_max = profile[_ACTIVE_MASK].max(axis=0)
            fit_success = True
            fit_error = ""
            if model.name == "Pdiag finite (SE(3))":
                profiles.append(
                    dict(
                        task=TASK,
                        seed=seed,
                        subset_id=subset["subset_id"],
                        sample_size=subset["sample_size"],
                        profile=profile,
                    )
                )
        except Exception as exc:
            fit_success, fit_error = False, repr(exc)
            e_alpha = float("nan")
            alpha_active_max = np.full(SE3_DIM, np.nan)
        rows.append(
            dict(
                task=TASK,
                seed=seed,
                subset_id=subset["subset_id"],
                protocol=subset["protocol"],
                sample_size=subset["sample_size"],
                repeat=subset["repeat"],
                model=model.name,
                fit_success=fit_success,
                fit_error=fit_error,
                fit_seconds=time.perf_counter() - started,
                e_alpha=e_alpha,
                m_prismatic_correct=(
                    bool(m_prismatic_correct(alpha_active_max)) if fit_success else False
                ),
                **{
                    f"alpha_active_{GENERATOR_NAMES[j]}": float(alpha_active_max[j])
                    for j in range(SE3_DIM)
                },
            )
        )
    return rows, profiles


def _fit_tpgmm_subset(job):
    seed, subset = job
    ds = _DATASETS[seed]
    cid_to_idx = {cid: i for i, cid in enumerate(ds["mixed_cids"])}
    source_cids = subset["source_condition_ids"]

    def placeholder(model, error):
        return dict(
            task=TASK,
            seed=seed,
            subset_id=subset["subset_id"],
            protocol=subset["protocol"],
            sample_size=subset["sample_size"],
            repeat=subset["repeat"],
            model=model,
            fit_success=False,
            fit_error=error,
            e_alpha_projected=np.nan,
            m_prismatic_projected_correct=False,
            **{f"alpha_active_{name}": np.nan for name in SE2_GENERATOR_NAMES},
        )

    missing = [cid for cid in source_cids if cid not in cid_to_idx]
    if missing:
        return [placeholder(model, "missing_source_condition") for model in TPGMM_MODEL_ORDER]

    indices = [cid_to_idx[cid] for cid in source_cids]
    contexts = ds["mixed_contexts_se2"][indices]
    curves = ds["mixed_curves_se2"][indices]
    models = [
        TPGMMModel(
            frame_mode="additive",
            component_candidates=(2, 4, 6, 8),
            cv_n_init=1,
            final_n_init=3,
        ),
        TPGMMModel(
            frame_mode="se2",
            nominal_frame_xy=np.array([0.10, 0.0], dtype=np.float64),
            nominal_frame_yaw=0.0,
            component_candidates=(2, 4, 6, 8),
            cv_n_init=1,
            final_n_init=3,
        ),
    ]
    oracle = oracle_alpha_drawer_se2(_PHASE_CODES)
    rows = []
    for model in models:
        started = time.perf_counter()
        try:
            model.fit(contexts, curves, _PROGRESS, _PHASE_CODES)
            profile = model.jacobian_diag()
            e_alpha = float(np.mean((profile - oracle) ** 2))
            alpha_active_max = profile[_ACTIVE_MASK].max(axis=0)
            fit_success = True
            fit_error = ""
        except Exception as exc:
            fit_success = False
            fit_error = repr(exc)
            e_alpha = float("nan")
            alpha_active_max = np.full(3, np.nan)
        rows.append(
            dict(
                task=TASK,
                seed=seed,
                subset_id=subset["subset_id"],
                protocol=subset["protocol"],
                sample_size=subset["sample_size"],
                repeat=subset["repeat"],
                model=model.name,
                fit_success=fit_success,
                fit_error=fit_error,
                fit_seconds=time.perf_counter() - started,
                e_alpha_projected=e_alpha,
                m_prismatic_projected_correct=(
                    bool(m_prismatic_projected_correct(alpha_active_max))
                    if fit_success
                    else False
                ),
                **{
                    f"alpha_active_{SE2_GENERATOR_NAMES[j]}": float(alpha_active_max[j])
                    for j in range(3)
                },
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path(
            "phase_switch_symmetry_multiseed/libero_drawer/libero_drawer_experiment.json"
        ),
    )
    parser.add_argument(
        "--subsets",
        type=Path,
        default=Path(
            "phase_switch_symmetry_multiseed/libero_drawer/libero_drawer_subsets.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/libero_drawer"),
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
    active_mask = np.isin(phase_codes, ACTIVE_CODES)

    datasets = {}
    for seed in SEEDS:
        path = Path(TASK_FILES[TASK].format(seed=seed))
        if not path.exists():
            raise FileNotFoundError(f"missing dataset: {path}")
        datasets[seed] = load_drawer_dataset(path, args.bins)

    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (seed, subset)
        for seed in SEEDS
        for subset in subset_manifest["subsets"]
    ]

    global _DATASETS, _PDIAG_CONFIG, _PROGRESS, _PHASE_CODES, _ACTIVE_MASK
    _DATASETS = datasets
    _PDIAG_CONFIG = pdiag_config
    _PROGRESS = progress
    _PHASE_CODES = phase_codes
    _ACTIVE_MASK = active_mask

    n_jobs = args.jobs or min(16, os.cpu_count() or 1)
    fit_rows = []
    profile_accum = []
    tpgmm_rows = []
    if n_jobs > 1:
        import multiprocessing

        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            for rows, profiles in pool.imap_unordered(_fit_subset, jobs):
                fit_rows.extend(rows)
                profile_accum.extend(profiles)
            for rows in pool.imap_unordered(_fit_tpgmm_subset, jobs):
                tpgmm_rows.extend(rows)
    else:
        for job in jobs:
            rows, profiles = _fit_subset(job)
            fit_rows.extend(rows)
            profile_accum.extend(profiles)
            tpgmm_rows.extend(_fit_tpgmm_subset(job))

    fits = pd.DataFrame(fit_rows)
    ok = fits[fits.fit_success]
    alpha_cols = [f"alpha_active_{g}" for g in GENERATOR_NAMES]

    m_prismatic = (
        ok.groupby(["model", "sample_size"], sort=False)["m_prismatic_correct"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "accuracy"})
        .reset_index()
    )
    alpha_means = (
        ok.groupby(["model", "sample_size"], sort=False)[alpha_cols]
        .mean()
        .reset_index()
    )
    e_alpha = (
        ok.groupby(["model", "sample_size"], sort=False)["e_alpha"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    tpgmm_fits = pd.DataFrame(tpgmm_rows)
    tpgmm_ok = tpgmm_fits[tpgmm_fits.fit_success] if len(tpgmm_fits) else pd.DataFrame()
    if len(tpgmm_ok):
        tpgmm_m_prismatic = (
            tpgmm_ok.groupby(["model", "sample_size"], sort=False)[
                "m_prismatic_projected_correct"
            ]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "accuracy"})
            .reset_index()
        )
        tpgmm_alpha = (
            tpgmm_ok.groupby(["model", "sample_size"], sort=False)[
                [f"alpha_active_{g}" for g in SE2_GENERATOR_NAMES]
            ]
            .mean()
            .reset_index()
        )
        tpgmm_e_alpha = (
            tpgmm_ok.groupby(["model", "sample_size"], sort=False)["e_alpha_projected"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
    else:
        tpgmm_m_prismatic = pd.DataFrame()
        tpgmm_alpha = pd.DataFrame()
        tpgmm_e_alpha = pd.DataFrame()

    fits.to_csv(args.output_root / "libero_drawer_fits.csv", index=False)
    tpgmm_fits.to_csv(args.output_root / "libero_drawer_tpgmm_fits.csv", index=False)
    m_prismatic.to_csv(args.output_root / "libero_drawer_M_prismatic.csv", index=False)
    alpha_means.to_csv(args.output_root / "libero_drawer_alpha_active_max.csv", index=False)
    e_alpha.to_csv(args.output_root / "libero_drawer_e_alpha.csv", index=False)
    if len(tpgmm_m_prismatic):
        tpgmm_m_prismatic.to_csv(
            args.output_root / "libero_drawer_tpgmm_M_prismatic_projected.csv",
            index=False,
        )
        tpgmm_alpha.to_csv(
            args.output_root / "libero_drawer_tpgmm_alpha_active_max_projected.csv",
            index=False,
        )
        tpgmm_e_alpha.to_csv(
            args.output_root / "libero_drawer_tpgmm_e_alpha_projected.csv",
            index=False,
        )
    if profile_accum:
        np.savez(
            args.output_root / "libero_drawer_profiles.npz",
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
            "benchmark_libero_drawer_probe.py": sha256(Path(__file__)),
            "benchmark_se3_transfer.py": sha256(
                Path(__file__).with_name("benchmark_se3_transfer.py")
            ),
            "phase_switch_se3_baselines.py": sha256(
                Path(__file__).with_name("phase_switch_se3_baselines.py")
            ),
        },
        "experiment": experiment,
        "oracle": (
            "du ramps from 0 to 1 during slide_open and stays 1 during "
            "hold/retract; dv/dw/roll/pitch/yaw stay 0."
        ),
        "M_prismatic": {
            "definition": (
                "max over active phases slide_open+hold_open; correct iff du > "
                "0.5 and all other SE(3) generators < 0.5."
            ),
            "threshold": THRESHOLD,
        },
        "M_prismatic_accuracy": m_prismatic.to_dict(orient="records"),
        "alpha_active_max": alpha_means.to_dict(orient="records"),
        "e_alpha": e_alpha.to_dict(orient="records"),
        "tpgmm_projected_baseline": {
            "note": (
                "The repository TP-GMM baseline is SE(2), so it is fit on the "
                "rail-plane projection [du,dv,yaw] rather than scored as a full "
                "six-generator SE(3) alpha model."
            ),
            "component_candidates": [2, 4, 6, 8],
            "cv_n_init": 1,
            "final_n_init": 3,
            "M_prismatic_projected_accuracy": (
                tpgmm_m_prismatic.to_dict(orient="records")
                if len(tpgmm_m_prismatic)
                else []
            ),
            "alpha_active_projected": (
                tpgmm_alpha.to_dict(orient="records") if len(tpgmm_alpha) else []
            ),
            "e_alpha_projected": (
                tpgmm_e_alpha.to_dict(orient="records") if len(tpgmm_e_alpha) else []
            ),
        },
        "fit_count": int(len(fits)),
        "tpgmm_fit_count": int(len(tpgmm_fits)),
        "fit_failure_count": int((~fits.fit_success).sum()),
        "tpgmm_fit_failure_count": int((~tpgmm_fits.fit_success).sum()) if len(tpgmm_fits) else 0,
        "bins_per_phase": args.bins,
    }
    with (args.output_root / "libero_drawer_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(json_ready(summary), output_file, indent=2, allow_nan=False)

    print("\n=== M-prismatic rail-selectivity accuracy ===")
    print(m_prismatic.to_string(index=False))
    print("\n=== per-generator active-phase max alpha_j (means) ===")
    print(alpha_means.to_string(index=False))
    print("\n=== E_alpha vs LIBERO drawer oracle ===")
    print(e_alpha.to_string(index=False))
    print("\n=== TP-GMM projected M-prismatic rail-selectivity accuracy ===")
    print(tpgmm_m_prismatic.to_string(index=False) if len(tpgmm_m_prismatic) else "(empty)")
    print("\n=== TP-GMM projected per-generator active-phase max alpha_j (means) ===")
    print(tpgmm_alpha.to_string(index=False) if len(tpgmm_alpha) else "(empty)")
    print("\n=== TP-GMM projected E_alpha ===")
    print(tpgmm_e_alpha.to_string(index=False) if len(tpgmm_e_alpha) else "(empty)")
    print("\nsaved:", args.output_root / "libero_drawer_summary.json")


if __name__ == "__main__":
    main()
