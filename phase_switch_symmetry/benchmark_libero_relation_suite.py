from __future__ import annotations

"""Benchmark the controlled ten-task LIBERO relation suite."""

import argparse
import json
import os
import time
from pathlib import Path

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
GENERATOR_NAMES = ("du", "dv", "dw", "roll", "pitch", "yaw")
SE2_INDICES = (0, 1, 5)
SE2_GENERATOR_NAMES = ("du", "dv", "yaw")
ACTIVE_CODES = (1, 2)
THRESHOLD = 0.5
TPGMM_MODEL_ORDER = ("TP-GMM additive", "TP-GMM SE(2)")


def usable(group) -> bool:
    return (
        bool(np.asarray(group["success"])[-1])
        and not bool(np.asarray(group["truncated"]).any())
    )


def _phase_pose6(group, phase, bins):
    phases = np.asarray(group["solver_phase"])
    indices = np.flatnonzero(phases == phase)
    pose = np.asarray(group["object_pose"])[indices]
    rpy = np.stack([euler_from_matrix(quat2mat(q)) for q in pose[:, 3:]])
    rpy = np.unwrap(rpy, axis=0)
    pose6 = np.column_stack([pose[:, 0], pose[:, 1], pose[:, 2], rpy])
    return resample(pose6, bins)


def task_curve(group, bins):
    return np.concatenate(
        [_phase_pose6(group, phase, bins) for phase in PHASE_CODES], axis=0
    )


def task_curve_se2(group, bins):
    return task_curve(group, bins)[:, SE2_INDICES]


def nominal_frame(data_file, keys, contexts):
    pose6s = []
    for key, context in zip(keys, contexts):
        target_pose = np.asarray(data_file[key]["target_pose"])[0]
        target_T = se3_from_pose6(pose_to_pose6(target_pose))
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


def oracle_alpha(selector, phase_codes):
    phase_codes = np.asarray(phase_codes, dtype=int)
    selector = np.asarray(selector, dtype=np.float64)
    oracle = np.zeros((len(phase_codes), SE3_DIM), dtype=np.float64)
    move_indices = np.flatnonzero(phase_codes == 1)
    if len(move_indices):
        ramp = np.linspace(0.0, 1.0, len(move_indices))
        oracle[move_indices] = ramp[:, None] * selector[None, :]
    oracle[phase_codes >= 2] = selector[None, :]
    return oracle


def oracle_alpha_projected(selector, phase_codes):
    phase_codes = np.asarray(phase_codes, dtype=int)
    selector = np.asarray(selector, dtype=np.float64)
    oracle = np.zeros((len(phase_codes), len(selector)), dtype=np.float64)
    move_indices = np.flatnonzero(phase_codes == 1)
    if len(move_indices):
        ramp = np.linspace(0.0, 1.0, len(move_indices))
        oracle[move_indices] = ramp[:, None] * selector[None, :]
    oracle[phase_codes >= 2] = selector[None, :]
    return oracle


def relation_correct(alpha_active_max, selector):
    selector = np.asarray(selector, dtype=bool)
    return bool(
        np.all(alpha_active_max[selector] > THRESHOLD)
        and np.all(alpha_active_max[~selector] < THRESHOLD)
    )


def load_dataset(path: Path, bins: int):
    with h5py.File(path, "r") as data_file:
        usable_groups = {
            int(data_file[key].attrs["condition_id"]): key
            for key in data_file
            if key.startswith("episode_") and usable(data_file[key])
        }
        mixed = {
            cid: key
            for cid, key in usable_groups.items()
            if str(data_file[key].attrs["generator"]) == "mixed"
        }
        mixed_cids = sorted(mixed)
        mixed_keys = [mixed[cid] for cid in mixed_cids]
        contexts = np.asarray([np.asarray(data_file[key]["causal_delta"]) for key in mixed_keys])
        curves = np.asarray([task_curve(data_file[key], bins) for key in mixed_keys])
        curves_se2 = np.asarray([task_curve_se2(data_file[key], bins) for key in mixed_keys])
        attrs = dict(
            task_key=str(data_file.attrs["task_key"]),
            family=str(data_file.attrs["family"]),
            language=str(data_file.attrs["language"]),
            oracle_selector=np.asarray(data_file.attrs["oracle_selector"], dtype=int),
        )
    return {
        "mixed_cids": mixed_cids,
        "mixed_keys": mixed_keys,
        "mixed_contexts": contexts,
        "mixed_curves": curves,
        "mixed_contexts_se2": contexts[:, SE2_INDICES],
        "mixed_curves_se2": curves_se2,
        **attrs,
    }


_DATASETS = None
_TASK_SPECS = None
_PDIAG_CONFIG = None
_PROGRESS = None
_PHASE_CODES = None
_ACTIVE_MASK = None


def _task_file(root: Path, task_key: str, seed: int) -> Path:
    return root / task_key / f"{task_key}_seed_{seed}.h5"


def _fit_subset(job):
    task_key, seed, subset = job
    ds = _DATASETS[task_key][seed]
    selector = np.asarray(ds["oracle_selector"], dtype=int)
    cid_to_idx = {cid: i for i, cid in enumerate(ds["mixed_cids"])}
    source_cids = subset["source_condition_ids"]

    def placeholder(model, error):
        return dict(
            task_key=task_key,
            family=ds["family"],
            seed=seed,
            subset_id=subset["subset_id"],
            protocol=subset["protocol"],
            sample_size=subset["sample_size"],
            repeat=subset["repeat"],
            model=model,
            fit_success=False,
            fit_error=error,
            e_alpha=np.nan,
            m_relation_correct=False,
            **{f"alpha_active_{name}": np.nan for name in GENERATOR_NAMES},
        )

    missing = [cid for cid in source_cids if cid not in cid_to_idx]
    if missing:
        return [placeholder(model, "missing_source_condition") for model in SE3_MODEL_ORDER], []

    indices = [cid_to_idx[cid] for cid in source_cids]
    contexts = ds["mixed_contexts"][indices]
    curves = ds["mixed_curves"][indices]
    keys = [ds["mixed_keys"][i] for i in indices]
    try:
        with h5py.File(_task_file(_TASK_SPECS["rollout_root"], task_key, seed), "r") as data_file:
            nominal_pose = nominal_frame(data_file, keys, contexts)
        models = build_models(nominal_pose, _PDIAG_CONFIG)
    except Exception as exc:
        return [placeholder(model, f"nominal_frame:{repr(exc)}") for model in SE3_MODEL_ORDER], []

    oracle = oracle_alpha(selector, _PHASE_CODES)
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
                        task_key=task_key,
                        family=ds["family"],
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
                task_key=task_key,
                family=ds["family"],
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
                m_relation_correct=(
                    bool(relation_correct(alpha_active_max, selector))
                    if fit_success
                    else False
                ),
                **{
                    f"alpha_active_{GENERATOR_NAMES[j]}": float(alpha_active_max[j])
                    for j in range(SE3_DIM)
                },
            )
        )
    return rows, profiles


def _fit_tpgmm_subset(job):
    task_key, seed, subset = job
    ds = _DATASETS[task_key][seed]
    selector = np.asarray(ds["oracle_selector"], dtype=int)[list(SE2_INDICES)]
    cid_to_idx = {cid: i for i, cid in enumerate(ds["mixed_cids"])}
    source_cids = subset["source_condition_ids"]

    def placeholder(model, error):
        return dict(
            task_key=task_key,
            family=ds["family"],
            seed=seed,
            subset_id=subset["subset_id"],
            protocol=subset["protocol"],
            sample_size=subset["sample_size"],
            repeat=subset["repeat"],
            model=model,
            fit_success=False,
            fit_error=error,
            e_alpha_projected=np.nan,
            m_relation_projected_correct=False,
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
            nominal_frame_xy=np.asarray(_TASK_SPECS["nominal_se2"][task_key][:2]),
            nominal_frame_yaw=float(_TASK_SPECS["nominal_se2"][task_key][2]),
            component_candidates=(2, 4, 6, 8),
            cv_n_init=1,
            final_n_init=3,
        ),
    ]
    oracle = oracle_alpha_projected(selector, _PHASE_CODES)
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
            fit_success, fit_error = False, repr(exc)
            e_alpha = float("nan")
            alpha_active_max = np.full(3, np.nan)
        rows.append(
            dict(
                task_key=task_key,
                family=ds["family"],
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
                m_relation_projected_correct=(
                    bool(relation_correct(alpha_active_max, selector))
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
            "phase_switch_symmetry_multiseed/libero_relation_suite/"
            "libero_relation_suite_experiment.json"
        ),
    )
    parser.add_argument(
        "--subsets",
        type=Path,
        default=Path(
            "phase_switch_symmetry_multiseed/libero_relation_suite/"
            "libero_relation_suite_subsets.json"
        ),
    )
    parser.add_argument(
        "--rollout-root",
        type=Path,
        default=Path("phase_switch_symmetry_rollouts_libero_relation_suite"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/libero_relation_suite"),
    )
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--pdiag-alpha-max", type=float, default=1.25)
    parser.add_argument("--pdiag-basis-count", type=int, default=24)
    parser.add_argument("--pdiag-basis-width", type=float, default=0.065)
    parser.add_argument("--pdiag-smoothness", type=float, default=0.1)
    parser.add_argument("--pdiag-nominal-iterations", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--skip-tpgmm", action="store_true")
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
    task_specs = {spec["task_key"]: spec for spec in experiment["task_specs"]}
    task_keys = args.tasks or list(task_specs)
    unknown = [task for task in task_keys if task not in task_specs]
    if unknown:
        raise ValueError(f"unknown task keys: {unknown}")

    progress, phase_codes = progress_grid(args.bins)
    active_mask = np.isin(phase_codes, ACTIVE_CODES)
    datasets = {task_key: {} for task_key in task_keys}
    for task_key in task_keys:
        for seed in SEEDS:
            path = _task_file(args.rollout_root, task_key, seed)
            if not path.exists():
                raise FileNotFoundError(f"missing dataset: {path}")
            datasets[task_key][seed] = load_dataset(path, args.bins)

    jobs = [
        (task_key, seed, subset)
        for task_key in task_keys
        for seed in SEEDS
        for subset in subset_manifest["subsets"]
    ]
    nominal_se2 = {
        task_key: (
            task_specs[task_key]["nominal_pose6"][0],
            task_specs[task_key]["nominal_pose6"][1],
            task_specs[task_key]["nominal_pose6"][5],
        )
        for task_key in task_keys
    }

    global _DATASETS, _TASK_SPECS, _PDIAG_CONFIG, _PROGRESS, _PHASE_CODES, _ACTIVE_MASK
    _DATASETS = datasets
    _TASK_SPECS = {"rollout_root": args.rollout_root, "nominal_se2": nominal_se2}
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
            if not args.skip_tpgmm:
                for rows in pool.imap_unordered(_fit_tpgmm_subset, jobs):
                    tpgmm_rows.extend(rows)
    else:
        for job in jobs:
            rows, profiles = _fit_subset(job)
            fit_rows.extend(rows)
            profile_accum.extend(profiles)
            if not args.skip_tpgmm:
                tpgmm_rows.extend(_fit_tpgmm_subset(job))

    args.output_root.mkdir(parents=True, exist_ok=True)
    fits = pd.DataFrame(fit_rows)
    ok = fits[fits.fit_success]
    alpha_cols = [f"alpha_active_{g}" for g in GENERATOR_NAMES]
    m_relation = (
        ok.groupby(["model", "task_key", "family", "sample_size"], sort=False)[
            "m_relation_correct"
        ]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "accuracy"})
        .reset_index()
    )
    m_relation_overall = (
        ok.groupby(["model", "sample_size"], sort=False)["m_relation_correct"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "accuracy"})
        .reset_index()
    )
    alpha_means = (
        ok.groupby(["model", "task_key", "sample_size"], sort=False)[alpha_cols]
        .mean()
        .reset_index()
    )
    e_alpha = (
        ok.groupby(["model", "task_key", "sample_size"], sort=False)["e_alpha"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    e_alpha_overall = (
        ok.groupby(["model", "sample_size"], sort=False)["e_alpha"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    fits.to_csv(args.output_root / "libero_relation_suite_fits.csv", index=False)
    m_relation.to_csv(args.output_root / "libero_relation_suite_M_relation_by_task.csv", index=False)
    m_relation_overall.to_csv(args.output_root / "libero_relation_suite_M_relation.csv", index=False)
    alpha_means.to_csv(args.output_root / "libero_relation_suite_alpha_active_max.csv", index=False)
    e_alpha.to_csv(args.output_root / "libero_relation_suite_e_alpha_by_task.csv", index=False)
    e_alpha_overall.to_csv(args.output_root / "libero_relation_suite_e_alpha.csv", index=False)

    tpgmm_fits = pd.DataFrame(tpgmm_rows)
    if len(tpgmm_fits):
        tpgmm_ok = tpgmm_fits[tpgmm_fits.fit_success]
        tpgmm_m = (
            tpgmm_ok.groupby(["model", "task_key", "family", "sample_size"], sort=False)[
                "m_relation_projected_correct"
            ]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "accuracy"})
            .reset_index()
        )
        tpgmm_m_overall = (
            tpgmm_ok.groupby(["model", "sample_size"], sort=False)[
                "m_relation_projected_correct"
            ]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "accuracy"})
            .reset_index()
        )
        tpgmm_alpha = (
            tpgmm_ok.groupby(["model", "task_key", "sample_size"], sort=False)[
                [f"alpha_active_{g}" for g in SE2_GENERATOR_NAMES]
            ]
            .mean()
            .reset_index()
        )
        tpgmm_e = (
            tpgmm_ok.groupby(["model", "task_key", "sample_size"], sort=False)[
                "e_alpha_projected"
            ]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        tpgmm_fits.to_csv(args.output_root / "libero_relation_suite_tpgmm_fits.csv", index=False)
        tpgmm_m.to_csv(
            args.output_root / "libero_relation_suite_tpgmm_M_relation_projected_by_task.csv",
            index=False,
        )
        tpgmm_m_overall.to_csv(
            args.output_root / "libero_relation_suite_tpgmm_M_relation_projected.csv",
            index=False,
        )
        tpgmm_alpha.to_csv(
            args.output_root / "libero_relation_suite_tpgmm_alpha_active_max_projected.csv",
            index=False,
        )
        tpgmm_e.to_csv(
            args.output_root / "libero_relation_suite_tpgmm_e_alpha_projected_by_task.csv",
            index=False,
        )
    else:
        tpgmm_m = pd.DataFrame()
        tpgmm_m_overall = pd.DataFrame()
        tpgmm_alpha = pd.DataFrame()
        tpgmm_e = pd.DataFrame()

    if profile_accum:
        np.savez(
            args.output_root / "libero_relation_suite_profiles.npz",
            profile=np.stack([p["profile"] for p in profile_accum]),
            task_key=np.asarray([p["task_key"] for p in profile_accum]),
            family=np.asarray([p["family"] for p in profile_accum]),
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
        "task_keys": task_keys,
        "task_families": {task: task_specs[task]["family"] for task in task_keys},
        "pdiag_config": pdiag_config,
        "M_relation": {
            "definition": (
                "max over active phases move+hold; each task is correct iff all "
                "oracle-selected generators are >0.5 and all suppressed "
                "generators are <0.5."
            ),
            "threshold": THRESHOLD,
        },
        "M_relation_overall": m_relation_overall.to_dict(orient="records"),
        "M_relation_by_task": m_relation.to_dict(orient="records"),
        "e_alpha_overall": e_alpha_overall.to_dict(orient="records"),
        "tpgmm_projected": {
            "note": (
                "TP-GMM is the repository SE(2) baseline, evaluated on "
                "[du,dv,yaw] projections only."
            ),
            "M_relation_projected_overall": (
                tpgmm_m_overall.to_dict(orient="records") if len(tpgmm_m_overall) else []
            ),
            "M_relation_projected_by_task": (
                tpgmm_m.to_dict(orient="records") if len(tpgmm_m) else []
            ),
        },
        "fit_count": int(len(fits)),
        "fit_failure_count": int((~fits.fit_success).sum()),
        "tpgmm_fit_count": int(len(tpgmm_fits)),
        "tpgmm_fit_failure_count": int((~tpgmm_fits.fit_success).sum()) if len(tpgmm_fits) else 0,
    }
    with (args.output_root / "libero_relation_suite_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(json_ready(summary), output_file, indent=2, allow_nan=False)

    print("\n=== M-relation overall ===")
    print(m_relation_overall.to_string(index=False))
    print("\n=== E-alpha overall ===")
    print(e_alpha_overall.to_string(index=False))
    print("\n=== TP-GMM projected M-relation overall ===")
    print(tpgmm_m_overall.to_string(index=False) if len(tpgmm_m_overall) else "(skipped)")
    print("\nsaved:", args.output_root / "libero_relation_suite_summary.json")


if __name__ == "__main__":
    main()
