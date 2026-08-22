from __future__ import annotations

"""Basis / coordinate ablation for the SE(3) six-generator supplement.

Reviewer question: "Are you learning TASK relevance, or just benefiting from a
manually chosen axis?" The Pdiag prior is a sparsity claim about the GENERATOR
BASIS: the expert response P(s) is diagonal in the task-local socket frame
[du, dv, dw, roll, pitch, yaw]. If that basis is principled, re-expressing the
SAME interventions in an arbitrary rotated basis must (a) degrade the diagonal
model's fit (the true response is off-diagonal there, and a diagonal cannot
represent it) while (b) leaving the dense 6x6 Full operator essentially
unchanged (it absorbs any basis change). If instead the diagonal prior were
just lucky, no such separation would appear.

This runs on the FROZEN SE(3) data (keyed + circular_honest), no new
collection: contexts are conjugated by a fixed arbitrary rotation R,
``context_rot = pose6(R^-1 T(context) R)``, and the nominal-frame recovery in
benchmark_se3_transfer.nominal_frame_se3 adapts automatically. The metric is
the basis-independent in-sample prediction error e_data (metric units), plus
the Full operator's off-diagonal mass per basis.

Expected: e_data(Pdiag, rotated) > e_data(Pdiag, local) ~= 0 while
e_data(Full, rotated) ~= e_data(Full, local); off-diag(Full, rotated) >>
off-diag(Full, local). This isolates "the diagonal sparsity prior is tied to
the task frame".
"""

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from benchmark_phase_switch_baselines import PHASE_CODES, progress_grid, usable
from benchmark_se3_transfer import (
    SE3_MODEL_ORDER,
    SEEDS,
    TASK_FILES,
    TASK_ORDER,
    build_models,
    json_ready,
    load_dataset,
    nominal_frame_se3,
    sha256,
)
from phase_switch_se3_baselines import (
    SE3_DIM,
    matrix_from_euler,
    pose6_from_se3,
    se3_from_pose6,
    se3_to_metric,
    wrap_angle,
)

BASIS_ORDER = ("local", "rotated-1", "rotated-2")
# Two arbitrary tilted rotations (Rx Ry Rz, deg). Deliberately NOT axis
# permutations, which would leave the diagonal structure intact; these mix all
# three generators, so the task-local insertion axis is not a basis axis.
ROTATION_EULERS_DEG = {
    "rotated-1": (30.0, -20.0, 40.0),
    "rotated-2": (-35.0, 25.0, 15.0),
}

_DATASETS = None
_PDIAG_CONFIG = None
_PROGRESS = None
_PHASE_CODES = None
_ROTATIONS = None


def rotate_contexts(contexts, R):
    """Conjugate each intervention by the basis rotation R: T -> R^-1 T R.

    R is a 3x3 rotation promoted to a homogeneous 4x4; conjugating the 4x4
    intervention applies the SE(3) adjoint Ad(R^-1) to its twist (rotation AND
    translation cross-terms), which is the correct re-expression of the
    intervention in the rotated basis.
    """
    R4 = np.eye(4)
    R4[:3, :3] = R
    out = np.empty_like(np.asarray(contexts, dtype=np.float64))
    for i, c in enumerate(contexts):
        out[i] = pose6_from_se3(R4.T @ se3_from_pose6(c) @ R4)
    return out


def prediction_error(model, contexts, curves):
    """In-sample prediction error in METRIC units (basis-independent)."""
    prediction = model.predict(contexts)
    residual = prediction - curves
    residual[..., 3:] = wrap_angle(residual[..., 3:])
    return float(np.mean(se3_to_metric(residual) ** 2))


def off_diagonal_mass(model):
    """||op - diag(op)|| / ||op|| for the Full operator (0 = purely diagonal)."""
    operator = model.operator_metric
    diagonal = np.diagonal(operator, axis1=1, axis2=2)
    off = operator - diagonal[..., None] * np.eye(SE3_DIM)[None, ...]
    return float(np.mean(np.linalg.norm(off, axis=(1, 2))) /
                 max(np.mean(np.linalg.norm(operator, axis=(1, 2))), 1e-12))


def _fit_subset(job):
    task, seed, subset, basis_name = job
    ds = _DATASETS[task][seed]
    cid_to_idx = {cid: i for i, cid in enumerate(ds["mixed_cids"])}
    source_cids = subset["source_condition_ids"]
    missing = [cid for cid in source_cids if cid not in cid_to_idx]
    if missing:
        return [
            dict(task=task, seed=seed, subset_id=subset["subset_id"],
                 protocol=subset["protocol"], sample_size=subset["sample_size"],
                 repeat=subset["repeat"], basis=basis_name, model=model,
                 fit_success=False, fit_error="missing_source_condition",
                 e_data=np.nan, e_alpha=np.nan, off_diag_mass=np.nan)
            for model in SE3_MODEL_ORDER
        ]
    indices = [cid_to_idx[cid] for cid in source_cids]
    contexts = ds["mixed_contexts"][indices]
    curves = ds["mixed_curves"][indices]
    keys = [ds["mixed_keys"][i] for i in indices]
    R = _ROTATIONS[basis_name]
    if R is not None:
        contexts = rotate_contexts(contexts, R)
    try:
        with h5py.File(Path(TASK_FILES[task].format(seed=seed)), "r") as data_file:
            nominal_frame_pose = nominal_frame_se3(data_file, keys, contexts)
        models = build_models(nominal_frame_pose, _PDIAG_CONFIG)
    except Exception as exc:
        return [
            dict(task=task, seed=seed, subset_id=subset["subset_id"],
                 protocol=subset["protocol"], sample_size=subset["sample_size"],
                 repeat=subset["repeat"], basis=basis_name, model=model,
                 fit_success=False, fit_error=f"nominal_frame:{repr(exc)}",
                 e_data=np.nan, e_alpha=np.nan, off_diag_mass=np.nan)
            for model in SE3_MODEL_ORDER
        ]
    rows = []
    from benchmark_se3_transfer import oracle_alpha_se3
    oracle = oracle_alpha_se3(task, _PHASE_CODES)
    for model in models:
        started = time.perf_counter()
        try:
            model.fit(contexts, curves, _PROGRESS, _PHASE_CODES)
            e_data = prediction_error(model, contexts, curves)
            # e_alpha: diagonal-vs-oracle error. Meaningful only in the local
            # basis (in a rotated basis the true response is off-diagonal); it
            # is retained to cross-check the local column against the frozen
            # se3_transfer M1 numbers.
            e_alpha = float(np.mean((model.jacobian_diag() - oracle) ** 2))
            off_diag = (
                off_diagonal_mass(model)
                if model.name == "Full operator (SE(3))" else np.nan
            )
            fit_success, fit_error = True, ""
        except Exception as exc:
            fit_success, fit_error = False, repr(exc)
            e_data = e_alpha = off_diag = np.nan
        rows.append(
            dict(task=task, seed=seed, subset_id=subset["subset_id"],
                 protocol=subset["protocol"], sample_size=subset["sample_size"],
                 repeat=subset["repeat"], basis=basis_name, model=model.name,
                 fit_success=fit_success, fit_error=fit_error,
                 fit_seconds=time.perf_counter() - started,
                 e_data=e_data, e_alpha=e_alpha, off_diag_mass=off_diag)
        )
    return rows


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
        default=Path("phase_switch_symmetry_multiseed/se3_transfer_basis_ablation"),
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

    datasets = {}
    for task in TASK_ORDER:
        datasets[task] = {}
        for seed in SEEDS:
            path = Path(TASK_FILES[task].format(seed=seed))
            if not path.exists():
                raise FileNotFoundError(f"missing dataset: {path}")
            datasets[task][seed] = load_dataset(path, args.bins)

    rotations = {
        "local": None,
        **{
            name: matrix_from_euler(
                np.deg2rad(deg[0]), np.deg2rad(deg[1]), np.deg2rad(deg[2])
            )
            for name, deg in ROTATION_EULERS_DEG.items()
        },
    }

    global _DATASETS, _PDIAG_CONFIG, _PROGRESS, _PHASE_CODES, _ROTATIONS
    _DATASETS = datasets
    _PDIAG_CONFIG = pdiag_config
    _PROGRESS = progress
    _PHASE_CODES = phase_codes
    _ROTATIONS = rotations

    jobs = [
        (task, seed, subset, basis_name)
        for task in TASK_ORDER
        for seed in SEEDS
        for subset in subset_manifest["subsets"]
        for basis_name in BASIS_ORDER
    ]

    n_jobs = args.jobs or min(16, os.cpu_count() or 1)
    fit_rows = []
    if n_jobs > 1:
        import multiprocessing
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            for rows in pool.imap_unordered(_fit_subset, jobs):
                fit_rows.extend(rows)
    else:
        for job in jobs:
            fit_rows.extend(_fit_subset(job))

    fits = pd.DataFrame(fit_rows)
    ok = fits[fits.fit_success]

    e_data = (
        ok.groupby(["model", "basis", "sample_size"], sort=False)["e_data"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    off_diag = (
        ok[ok.model == "Full operator (SE(3))"]
        .groupby(["basis", "sample_size"], sort=False)["off_diag_mass"]
        .agg(["mean", "std"])
        .reset_index()
    )
    # local-basis e_alpha cross-check against the frozen M1 (keyed arm).
    e_alpha_local = (
        ok[(ok.basis == "local")]
        .groupby(["task", "model", "sample_size"], sort=False)["e_alpha"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    fits.to_csv(args.output_root / "basis_ablation_fits.csv", index=False)
    e_data.to_csv(args.output_root / "basis_ablation_e_data.csv", index=False)
    off_diag.to_csv(args.output_root / "basis_ablation_off_diag.csv", index=False)
    e_alpha_local.to_csv(
        args.output_root / "basis_ablation_e_alpha_local.csv", index=False
    )

    summary = {
        "schema_version": 1,
        "experiment_sha256": sha256(args.experiment),
        "subset_manifest_sha256": sha256(args.subsets),
        "source_sha256": {
            "benchmark_se3_basis_ablation.py": sha256(Path(__file__)),
            "benchmark_se3_transfer.py": sha256(
                Path(__file__).with_name("benchmark_se3_transfer.py")
            ),
            "phase_switch_se3_baselines.py": sha256(
                Path(__file__).with_name("phase_switch_se3_baselines.py")
            ),
        },
        "design": {
            "question": "Is the generator basis principled (task-local socket frame) or arbitrary?",
            "procedure": (
                "Contexts conjugated by a fixed rotation: context_rot = "
                "pose6(R^-1 T(context) R); the nominal-frame recovery adapts "
                "automatically. Same frozen data, same subsets, same models."
            ),
            "metric": "e_data = mean squared in-sample prediction error in metric units (basis-independent).",
            "expected": (
                "e_data(Pdiag, rotated) >> e_data(Pdiag, local) ~= 0; "
                "e_data(Full, rotated) ~= e_data(Full, local); "
                "off-diag(Full, rotated) >> off-diag(Full, local)."
            ),
        },
        "rotations_deg": ROTATION_EULERS_DEG,
        "e_data": e_data.to_dict(orient="records"),
        "off_diag_mass_full_operator": off_diag.to_dict(orient="records"),
        "e_alpha_local_crosscheck": e_alpha_local.to_dict(orient="records"),
        "fit_count": int(len(fits)),
        "fit_failure_count": int((~fits.fit_success).sum()),
        "bins_per_phase": args.bins,
    }
    with (args.output_root / "basis_ablation_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(json_ready(summary), output_file, indent=2, allow_nan=False)

    print("\n=== e_data (in-sample prediction error, metric units) ===")
    print(e_data.to_string(index=False))
    print("\n=== off-diagonal mass, Full operator (0 = purely diagonal) ===")
    print(off_diag.to_string(index=False))
    print("\n=== e_alpha local-basis cross-check vs frozen M1 ===")
    print(e_alpha_local.to_string(index=False))
    print("\nsaved:", args.output_root / "basis_ablation_summary.json")


if __name__ == "__main__":
    main()
