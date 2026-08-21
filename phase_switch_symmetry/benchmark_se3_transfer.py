from __future__ import annotations

"""SE(3) 6-generator symmetry-transfer benchmark (supplement).

The decisive question: among six SE(3) generators ``[du, dv, dw, d_roll, d_pitch,
d_yaw]``, does the model identify the axial-yaw generator as the UNIQUE selective
one — ``alpha_yaw = 1 -> 1 -> 0 -> 0`` over the four insertion phases — while the
other five generators are always-on during insertion (``alpha = 1`` throughout)?

Oracle (model-independent, from the solver geometry):

    generator       keyed (keyed peg)      circular (round peg)
    du, dv, dw      [1, 1, 1, 1]           [1, 1, 1, 1]
    roll, pitch     [1, 1, 1, 1]           [1, 1, 1, 1]
    yaw             [1, 1, 0, 0]           [0, 0, 0, 0]

Headline metric M4 (selectivity identification), for the keyed arm: define the
post-clearance relevance drop ``Delta_j = mean(alpha_j | phase<2) - mean(alpha_j |
phase>=2)``.  Oracle ``Delta*_yaw = 1``, ``Delta*_others = 0``.  Correct iff the
model recovers ``Delta_yaw > 0.5`` AND ``Delta_j < 0.5`` for all five others.

This mirrors benchmark_symmetry_transfer.py but against the SE(3) model layer
(phase_switch_se3_baselines.py); the SE(2) frozen results are untouched.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
from transforms3d.quaternions import quat2mat

from analyze_phase_switch_rollouts import resample, wrap_pi
from benchmark_phase_switch_baselines import (
    PHASE_CODES,
    progress_grid,
    switch_diagnostics,
    usable,
)
from phase_switch_se3_baselines import (
    SE3_DIM,
    SE3DiagonalOperatorModel,
    SE3FrameWeightedModel,
    SE3FullOperatorModel,
    SE3SmoothFinitePDiagModel,
    euler_from_matrix,
    pose6_from_se3,
    se3_from_pose6,
    se3_inverse,
)


SEEDS = [20260818, 20270818, 20280818]
TASK_ORDER = ["keyed", "circular_honest"]
TASK_FILES = {
    "keyed": "phase_switch_symmetry_rollouts_se3/keyed_seed_{seed}.h5",
    "circular_honest": "phase_switch_symmetry_rollouts_se3/circular_honest_seed_{seed}.h5",
}

SE3_MODEL_ORDER = (
    "Frame-weighted (SE(3))",
    "Full operator (SE(3))",
    "Pdiag pointwise (SE(3))",
    "Pdiag finite (SE(3))",
)
SE3_MODEL_DEFINITIONS = {
    "Frame-weighted (SE(3))": "Per-progress affine regression constrained to one shared scalar w(s) for all six SE(3) generators.",
    "Full operator (SE(3))": "Per-progress affine dense 6x6 response operator.",
    "Pdiag pointwise (SE(3))": "Ablation that independently fits one affine diagonal response at each progress point (6 channels).",
    "Pdiag finite (SE(3))": "Final method: alpha_max-constrained RBF-sigmoid 6-generator profiles with second-difference smoothness, realized by C0 Exp(P(s) Log(C0^-1 C)) C0^-1 X0 in SE(3).",
}

GENERATOR_NAMES = ("du", "dv", "dw", "roll", "pitch", "yaw")
YAW_INDEX = 5


def nan_generator_columns():
    """NaN placeholders for the per-generator relevance columns."""
    cols = {}
    for generator in GENERATOR_NAMES:
        cols[f"alpha_pre_{generator}"] = np.nan
        cols[f"delta_{generator}"] = np.nan
    return cols


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def pose_to_pose6(pose):
    """SAPIEN raw pose (..., 7) = [x, y, z, w, x, y, z] -> [x, y, z, roll, pitch, yaw]
    in the local R = Rx(roll) Ry(pitch) Rz(yaw) convention used by the SE(3) model.
    Accepts a single (7,) pose or a batch (..., 7)."""
    pose = np.asarray(pose, dtype=np.float64)
    single = pose.ndim == 1
    if single:
        pose = pose[None]
    rpy = np.stack([euler_from_matrix(quat2mat(q)) for q in pose[..., 3:]])
    out = np.concatenate([pose[..., :3], rpy], axis=-1)
    return out[0] if single else out


def phase_task_curve_se3(group, phase, bins):
    phases = np.asarray(group["solver_phase"])
    indices = np.flatnonzero(phases == phase)
    peg = np.asarray(group["peg_pose"])[indices]  # (n, 7)
    rpy = np.stack([euler_from_matrix(quat2mat(q)) for q in peg[:, 3:]])
    rpy = np.unwrap(rpy, axis=0)
    pose6 = np.column_stack([peg[:, 0], peg[:, 1], peg[:, 2], rpy])
    return resample(pose6, bins)


def task_curve_se3(group, bins):
    return np.concatenate(
        [phase_task_curve_se3(group, phase, bins) for phase in PHASE_CODES], axis=0
    )


def nominal_frame_se3(data_file, keys, contexts):
    """Recover the nominal socket frame pose6 from mixed socket poses by removing
    each known SE(3) intervention: C0 = socket_world o intervention^{-1}."""
    pose6s = []
    for key, context in zip(keys, contexts):
        socket_pose = np.asarray(data_file[key]["socket_pose"])[0]
        socket_pose6 = pose_to_pose6(socket_pose)
        socket_T = se3_from_pose6(socket_pose6)
        intervention_T = se3_from_pose6(np.asarray(context, dtype=np.float64))
        nominal_T = socket_T @ se3_inverse(intervention_T)
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


def oracle_alpha_se3(task, phase_codes):
    """6 x T generator-law oracle: yaw is the only selective generator."""
    phase_codes = np.asarray(phase_codes, dtype=int)
    oracle = np.ones((len(phase_codes), SE3_DIM), dtype=np.float64)
    if task == "keyed":
        oracle[:, 5] = (phase_codes < 2).astype(np.float64)  # 1 align/enter, 0 unlock/insert
    else:
        oracle[:, 5] = 0.0  # circular: full SO(2) gauge symmetry throughout
    return oracle


def build_models(nominal_frame_pose, pdiag_config):
    return [
        SE3FrameWeightedModel(),
        SE3FullOperatorModel(),
        SE3DiagonalOperatorModel(),
        SE3SmoothFinitePDiagModel(
            nominal_frame_pose=nominal_frame_pose, **pdiag_config
        ),
    ]


def load_dataset(path: Path, bins: int):
    with h5py.File(path, "r") as data_file:
        usable_groups = {
            int(data_file[key].attrs["condition_id"]): key
            for key in data_file
            if key.startswith("episode_") and usable(data_file[key])
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
            [task_curve_se3(data_file[key], bins) for key in mixed_keys]
        )
    return {
        "mixed_cids": mixed_cids,
        "mixed_keys": mixed_keys,
        "mixed_contexts": mixed_contexts,
        "mixed_curves": mixed_curves,
    }


_DATASETS = None
_PDIAG_CONFIG = None
_PROGRESS = None
_PHASE_CODES = None
_PRE_MASK = None
_UNLOCK_START = None


def _fit_subset(job):
    """Fit all four models on one (task, seed, subset) cell; return (rows, profiles)."""
    task, seed, subset = job
    ds = _DATASETS[task][seed]
    cid_to_idx = {cid: i for i, cid in enumerate(ds["mixed_cids"])}
    sample_size = subset["sample_size"]
    source_cids = subset["source_condition_ids"]
    missing = [cid for cid in source_cids if cid not in cid_to_idx]
    if missing:
        rows = [
            dict(
                task=task, seed=seed, subset_id=subset["subset_id"],
                protocol=subset["protocol"], sample_size=sample_size,
                repeat=subset["repeat"], model=model,
                fit_success=False, fit_error="missing_source_condition",
                alpha_pre=np.nan, alpha_post=np.nan, e_alpha=np.nan,
                delta_others_max=np.nan,
                m4_correct=False,
                switch_detected=False, switch_location=np.nan,
                **nan_generator_columns(),
            )
            for model in SE3_MODEL_ORDER
        ]
        return rows, []
    indices = [cid_to_idx[cid] for cid in source_cids]
    contexts = ds["mixed_contexts"][indices]
    curves = ds["mixed_curves"][indices]
    keys = [ds["mixed_keys"][i] for i in indices]
    try:
        with h5py.File(Path(TASK_FILES[task].format(seed=seed)), "r") as data_file:
            nominal_frame_pose = nominal_frame_se3(data_file, keys, contexts)
        models = build_models(nominal_frame_pose, _PDIAG_CONFIG)
    except Exception as exc:
        rows = [
            dict(
                task=task, seed=seed, subset_id=subset["subset_id"],
                protocol=subset["protocol"], sample_size=sample_size,
                repeat=subset["repeat"], model=model,
                fit_success=False, fit_error=f"nominal_frame:{repr(exc)}",
                alpha_pre=np.nan, alpha_post=np.nan, e_alpha=np.nan,
                delta_others_max=np.nan,
                m4_correct=False,
                switch_detected=False, switch_location=np.nan,
                **nan_generator_columns(),
            )
            for model in SE3_MODEL_ORDER
        ]
        return rows, []

    rows = []
    profiles = []
    for model in models:
        started = time.perf_counter()
        try:
            model.fit(contexts, curves, _PROGRESS, _PHASE_CODES)
            profile = model.jacobian_diag()  # (n_steps, 6)
            oracle = oracle_alpha_se3(task, _PHASE_CODES)
            e_alpha = float(np.mean((profile - oracle) ** 2))
            alpha_yaw = profile[:, YAW_INDEX]
            alpha_pre = float(alpha_yaw[_PRE_MASK].mean())
            alpha_post = float(alpha_yaw[~_PRE_MASK].mean())
            switch = switch_diagnostics(_PROGRESS, alpha_yaw, _UNLOCK_START)
            alpha_pre_vec = profile[_PRE_MASK].mean(axis=0)          # (6,)
            alpha_post_vec = profile[~_PRE_MASK].mean(axis=0)        # (6,)
            delta = alpha_pre_vec - alpha_post_vec                  # (6,) clearance drop
            delta_yaw = float(delta[YAW_INDEX])
            delta_others_max = float(np.max(np.delete(delta, YAW_INDEX)))
            m4_correct = bool(delta_yaw > 0.5 and delta_others_max < 0.5)
            fit_success = True
            fit_error = ""
            if model.name == "Pdiag finite (SE(3))":
                profiles.append(
                    dict(task=task, seed=seed, subset_id=subset["subset_id"],
                         sample_size=sample_size, profile=profile)
                )
        except Exception as exc:
            fit_success = False
            fit_error = repr(exc)
            alpha_pre = alpha_post = e_alpha = float("nan")
            delta_yaw = delta_others_max = float("nan")
            m4_correct = False
            alpha_pre_vec = np.full(SE3_DIM, np.nan)
            delta = np.full(SE3_DIM, np.nan)
            switch = {"detected": False, "status": "fit_failed", "location": float("nan")}
        rows.append(
            dict(
                task=task, seed=seed, subset_id=subset["subset_id"],
                protocol=subset["protocol"], sample_size=sample_size,
                repeat=subset["repeat"], model=model.name,
                fit_success=fit_success, fit_error=fit_error,
                fit_seconds=time.perf_counter() - started,
                alpha_pre=alpha_pre, alpha_post=alpha_post, e_alpha=e_alpha,
                delta_others_max=delta_others_max,
                m4_correct=m4_correct,
                switch_detected=bool(switch["detected"]),
                switch_status=switch["status"],
                switch_location=switch["location"],
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
        default=Path("phase_switch_symmetry_multiseed/se3_transfer"),
    )
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--pdiag-alpha-max", type=float, default=1.25)
    parser.add_argument("--pdiag-basis-count", type=int, default=24)
    parser.add_argument("--pdiag-basis-width", type=float, default=0.065)
    parser.add_argument("--pdiag-smoothness", type=float, default=0.1)
    parser.add_argument("--pdiag-nominal-iterations", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=0,
                        help="parallel workers for model fits (0 = min(16, cpu_count))")
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
    profile_accum = []  # (task, seed, subset_id, sample_size, n_steps x 6) for Pdiag finite
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

    # ---- M1: E_alpha (per model x N x task, mean over seeds/repeats) ----
    m1 = (
        fits[fits.fit_success]
        .groupby(["task", "model", "sample_size"], sort=False)["e_alpha"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # ---- M2: discrimination accuracy (alpha_pre > 0.5 classifies "keyed") ----
    disc = fits[fits.fit_success].copy()
    disc["pred_keyed"] = disc["alpha_pre"] > 0.5
    disc["correct"] = disc["pred_keyed"] == (disc["task"] == "keyed")
    m2 = (
        disc.groupby(["model", "sample_size"], sort=False)["correct"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "accuracy"})
        .reset_index()
    )

    # ---- M2b: symmetry gap G_psi (keyed - circular alpha_pre) ----
    piv = fits[fits.fit_success].pivot_table(
        index=["model", "sample_size", "seed", "subset_id"],
        columns="task", values="alpha_pre",
    ).reset_index()
    gap_rows = []
    if "circular_honest" in piv.columns:
        for _, row in piv.iterrows():
            if not np.isfinite(row["keyed"]) or not np.isfinite(row["circular_honest"]):
                continue
            gap = float(row["keyed"] - row["circular_honest"])
            gap_rows.append(
                dict(
                    model=row["model"], sample_size=row["sample_size"],
                    seed=row["seed"], subset_id=row["subset_id"],
                    G_psi=gap, abs_dev_from_1=abs(gap - 1.0),
                )
            )
    gap_df = pd.DataFrame(gap_rows)
    m2b = (
        gap_df.groupby(["model", "sample_size"], sort=False)
        .agg(G_psi_mean=("G_psi", "mean"), G_psi_std=("G_psi", "std"),
             abs_dev_mean=("abs_dev_from_1", "mean"))
        .reset_index()
    ) if len(gap_df) else pd.DataFrame()

    # ---- M0: keyed switch fidelity (Pdiag finite) ----
    keyed_pdiag = fits[
        (fits.task == "keyed") & (fits.model == "Pdiag finite (SE(3))") & fits.fit_success
    ].copy()
    keyed_pdiag["switch_deviation"] = (keyed_pdiag["switch_location"] - 0.5).abs()
    m0 = (
        keyed_pdiag.groupby(["model", "sample_size"], sort=False)
        .agg(
            switch_detected_fraction=("switch_detected", "mean"),
            switch_location_mean=("switch_location", "mean"),
            switch_deviation_mean=("switch_deviation", "mean"),
        )
        .reset_index()
    )

    # ---- M4: selectivity identification (keyed arm) ----
    keyed_fits = fits[(fits.task == "keyed") & fits.fit_success]
    m4 = (
        keyed_fits.groupby(["model", "sample_size"], sort=False)
        .agg(
            m4_accuracy=("m4_correct", "mean"),
            delta_yaw_mean=("delta_yaw", "mean"),
            delta_others_max_mean=("delta_others_max", "mean"),
            count=("m4_correct", "count"),
        )
        .reset_index()
    )

    # ---- M1: generator selectivity S_j = |alpha_pre_j(keyed) - alpha_pre_j(circular)| ----
    delta_cols = [f"delta_{g}" for g in GENERATOR_NAMES]
    alpha_pre_cols = [f"alpha_pre_{g}" for g in GENERATOR_NAMES]
    sel_rows = []
    for generator in GENERATOR_NAMES:
        col = f"alpha_pre_{generator}"
        piv = fits[fits.fit_success].pivot_table(
            index=["model", "sample_size", "seed", "subset_id"],
            columns="task", values=col,
        ).reset_index()
        if "keyed" not in piv.columns or "circular_honest" not in piv.columns:
            continue
        for _, row in piv.iterrows():
            if not np.isfinite(row["keyed"]) or not np.isfinite(row["circular_honest"]):
                continue
            sel_rows.append(
                dict(model=row["model"], sample_size=row["sample_size"],
                     seed=row["seed"], subset_id=row["subset_id"],
                     generator=generator,
                     S=abs(float(row["keyed"]) - float(row["circular_honest"])))
            )
    sel_df = pd.DataFrame(sel_rows)
    m1_sel = (
        sel_df.groupby(["model", "sample_size", "generator"], sort=False)["S"]
        .agg(["mean", "std", "count"])
        .reset_index()
    ) if len(sel_df) else pd.DataFrame()

    # ---- purity: S_yaw / sum_j S_j per (model, N, seed, subset) ----
    purity_rows = []
    if len(sel_df):
        for (model, n, seed, sid), grp in sel_df.groupby(
            ["model", "sample_size", "seed", "subset_id"]
        ):
            S = grp.set_index("generator")["S"]
            total = float(S.sum())
            s_yaw = float(S.get("yaw", 0.0))
            others = S.drop("yaw", errors="ignore")
            purity_rows.append(
                dict(model=model, sample_size=n, seed=seed, subset_id=sid,
                     purity=(s_yaw / total) if total > 0 else np.nan,
                     S_yaw=s_yaw,
                     S_others_max=float(others.max()) if len(others) else np.nan,
                     S_others_mean=float(others.mean()) if len(others) else np.nan)
            )
    purity_df = pd.DataFrame(purity_rows)
    m_purity = (
        purity_df.groupby(["model", "sample_size"], sort=False)
        .agg(purity_mean=("purity", "mean"), purity_std=("purity", "std"),
             S_yaw_mean=("S_yaw", "mean"),
             S_others_max_mean=("S_others_max", "mean"))
        .reset_index()
    ) if len(purity_df) else pd.DataFrame()

    # ---- M2: rank ordering (yaw is the top clearance drop among 6, keyed arm) ----
    keyed = fits[(fits.task == "keyed") & fits.fit_success].copy()
    keyed["delta_argmax"] = keyed[delta_cols].values.argmax(axis=1)
    keyed["rank1_yaw"] = (keyed["delta_argmax"] == YAW_INDEX) & (keyed["delta_yaw"] > 0)
    m2_rank = (
        keyed.groupby(["model", "sample_size"], sort=False)["rank1_yaw"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "accuracy"})
        .reset_index()
    )

    # ---- M3: phase-transition detection (delta_yaw > tau = 0.2, keyed arm) ----
    keyed["transition_detected"] = keyed["delta_yaw"] > 0.2
    m3_trans = (
        keyed.groupby(["model", "sample_size"], sort=False)["transition_detected"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "accuracy"})
        .reset_index()
    )

    # ---- write outputs ----
    fits_path = args.output_root / "se3_transfer_fits.csv"
    m1_path = args.output_root / "se3_transfer_M1_ealpha.csv"
    m2_path = args.output_root / "se3_transfer_M2_discrimination.csv"
    m2b_path = args.output_root / "se3_transfer_M2b_symmetry_gap.csv"
    m0_path = args.output_root / "se3_transfer_M0_switch.csv"
    m4_path = args.output_root / "se3_transfer_M4_selectivity.csv"
    m1sel_path = args.output_root / "se3_transfer_M1_selectivity.csv"
    purity_path = args.output_root / "se3_transfer_purity.csv"
    m2rank_path = args.output_root / "se3_transfer_M2_rank.csv"
    m3trans_path = args.output_root / "se3_transfer_M3_transition.csv"
    profiles_path = args.output_root / "se3_transfer_profiles.npz"
    fits.to_csv(fits_path, index=False)
    m1.to_csv(m1_path, index=False)
    m2.to_csv(m2_path, index=False)
    m2b.to_csv(m2b_path, index=False)
    m0.to_csv(m0_path, index=False)
    m4.to_csv(m4_path, index=False)
    m1_sel.to_csv(m1sel_path, index=False)
    m_purity.to_csv(purity_path, index=False)
    m2_rank.to_csv(m2rank_path, index=False)
    m3_trans.to_csv(m3trans_path, index=False)
    if profile_accum:
        np.savez(
            profiles_path,
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
            "benchmark_se3_transfer.py": sha256(Path(__file__)),
            "phase_switch_se3_baselines.py": sha256(
                Path(__file__).with_name("phase_switch_se3_baselines.py")
            ),
        },
        "model_definitions": SE3_MODEL_DEFINITIONS,
        "oracle": {
            "keyed": "du/dv/dw/roll/pitch = 1 everywhere; yaw = 1 in align/enter (phase<2), 0 in unlock/insert",
            "circular_honest": "du/dv/dw/roll/pitch = 1 everywhere; yaw = 0 everywhere (SO(2) gauge symmetry)",
        },
        "M4_selectivity": {
            "definition": "Delta_j = mean(alpha_j | phase<2) - mean(alpha_j | phase>=2); correct iff Delta_yaw>0.5 AND Delta_j<0.5 for du/dv/dw/roll/pitch",
            "oracle": {"delta_yaw": 1.0, "delta_others": 0.0},
        },
        "M1_generator_identification_error": m1.to_dict(orient="records"),
        "M2_discrimination_accuracy": m2.to_dict(orient="records"),
        "M2b_symmetry_gap": m2b.to_dict(orient="records") if len(m2b) else [],
        "M0_keyed_switch_fidelity": m0.to_dict(orient="records"),
        "M4_selectivity_identification": m4.to_dict(orient="records"),
        "M1_generator_selectivity": m1_sel.to_dict(orient="records"),
        "purity": m_purity.to_dict(orient="records") if len(m_purity) else [],
        "M2_rank_ordering": m2_rank.to_dict(orient="records"),
        "M3_transition_detection": m3_trans.to_dict(orient="records"),
        "M3_transition_tau": 0.2,
        "align_phase_transient_note": (
            "The align phase (phase 0) contains a within-phase 0->1 rise for every "
            "generator as the peg moves from the lift pose to the socket's SE(3) pose; "
            "its mean is ~0.5, so the phase<2 'pre' mean lands at ~0.7 rather than the "
            "oracle 1.0. This is a real trajectory transient, not regularization "
            "attenuation (smoothness_weight has no effect on the recovered profile). "
            "Selectivity is therefore scored by the clearance drop (M2/M3) and the "
            "keyed-vs-circular gap (M1/purity), not by exact oracle magnitude."
        ),
        "fit_count": int(len(fits)),
        "fit_failure_count": int((~fits.fit_success).sum()),
        "bins_per_phase": args.bins,
    }
    summary_path = args.output_root / "se3_transfer_summary.json"
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(json_ready(summary), output_file, indent=2, allow_nan=False)

    print("\n=== M1 E_alpha (mean over seeds/repeats) ===")
    print(m1.to_string(index=False))
    print("\n=== M2 discrimination accuracy ===")
    print(m2.to_string(index=False))
    print("\n=== M2b symmetry gap G_psi ===")
    print(m2b.to_string(index=False) if len(m2b) else "(empty)")
    print("\n=== M0 keyed switch fidelity (Pdiag finite) ===")
    print(m0.to_string(index=False))
    print("\n=== M4 selectivity identification (keyed arm) ===")
    print(m4.to_string(index=False))
    print("\n=== M1 generator selectivity S_j (keyed vs circular) ===")
    print(m1_sel.to_string(index=False))
    print("\n=== purity S_yaw / sum_j S_j ===")
    print(m_purity.to_string(index=False) if len(m_purity) else "(empty)")
    print("\n=== M2 rank ordering (yaw is top clearance drop) ===")
    print(m2_rank.to_string(index=False))
    print("\n=== M3 transition detection (delta_yaw > 0.2) ===")
    print(m3_trans.to_string(index=False))
    print("\nsaved:", summary_path)


if __name__ == "__main__":
    main()
