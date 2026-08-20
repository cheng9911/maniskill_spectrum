from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd

from analyze_phase_switch_rollouts import pose_yaw, wrap_pi
from benchmark_phase_switch_baselines import (
    MODEL_DEFINITIONS,
    MODEL_ORDER,
    metric_errors,
    progress_grid,
    switch_diagnostics,
    task_curve,
    usable,
)
from phase_switch_baselines import (
    DiagonalOperatorModel,
    FrameWeightedModel,
    FullOperatorModel,
    GenericConditionalRBF,
    PhaseScalarGPModel,
    SmoothFinitePDiagModel,
    TPGMMModel,
)


SEEDS = [20260818, 20270818, 20280818]
TASK_ORDER = ["keyed", "circular_honest", "circular_placebo"]
TASK_FILES = {
    "keyed": "phase_switch_symmetry_rollouts_rotated/rotated_Q1_seed_{seed}.h5",
    "circular_honest": "phase_switch_symmetry_rollouts_rotated/circular_honest_seed_{seed}.h5",
    "circular_placebo": "phase_switch_symmetry_rollouts_rotated/circular_placebo_seed_{seed}.h5",
}


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


def oracle_alpha_yaw(task, phase_codes):
    """Model-independent generator-law oracle from the symmetry group (design §2)."""
    phase_codes = np.asarray(phase_codes, dtype=int)
    if task == "keyed":
        return (phase_codes < 2).astype(np.float64)  # 1 in align/enter, 0 in unlock/insert
    return np.zeros_like(phase_codes, dtype=np.float64)  # circular: ~0 everywhere


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


def build_models(nominal_frame_xy, nominal_frame_yaw, pdiag_config):
    return [
        FrameWeightedModel(),
        PhaseScalarGPModel(),
        TPGMMModel(frame_mode="additive"),
        TPGMMModel(
            frame_mode="se2",
            nominal_frame_xy=nominal_frame_xy,
            nominal_frame_yaw=nominal_frame_yaw,
        ),
        GenericConditionalRBF(),
        FullOperatorModel(),
        DiagonalOperatorModel(),
        SmoothFinitePDiagModel(
            nominal_frame_xy=nominal_frame_xy,
            nominal_frame_yaw=nominal_frame_yaw,
            **pdiag_config,
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
            [task_curve(data_file[key], bins) for key in mixed_keys]
        )
        isolated_items = sorted(
            [
                (cid, key)
                for cid, key in usable_groups.items()
                if str(data_file[key].attrs["generator"]) in {"yaw", "translation"}
                and np.linalg.norm(np.asarray(data_file[key]["causal_delta"])) > 1e-12
            ]
        )
        baseline_items = [
            (cid, key)
            for cid, key in usable_groups.items()
            if str(data_file[key].attrs["generator"]) == "yaw"
            and np.linalg.norm(np.asarray(data_file[key]["causal_delta"])) <= 1e-12
        ]
        isolated_cids = [cid for cid, _ in isolated_items]
        isolated_keys = [key for _, key in isolated_items]
        test_contexts = np.asarray(
            [np.asarray(data_file[key]["causal_delta"]) for key in isolated_keys]
        )
        test_curves = np.asarray(
            [task_curve(data_file[key], bins) for key in isolated_keys]
        )
        test_generators = [str(data_file[key].attrs["generator"]) for key in isolated_keys]
    return {
        "mixed_cids": mixed_cids,
        "mixed_keys": mixed_keys,
        "mixed_contexts": mixed_contexts,
        "mixed_curves": mixed_curves,
        "isolated_cids": isolated_cids,
        "isolated_keys": isolated_keys,
        "test_contexts": test_contexts,
        "test_curves": test_curves,
        "test_generators": test_generators,
    }


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
    parser.add_argument("--pdiag-alpha-max", type=float, default=1.25)
    parser.add_argument("--pdiag-basis-count", type=int, default=24)
    parser.add_argument("--pdiag-basis-width", type=float, default=0.065)
    parser.add_argument("--pdiag-smoothness", type=float, default=0.1)
    parser.add_argument("--pdiag-nominal-iterations", type=int, default=3)
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
    task_yaw_weights = (phase_codes < 2).astype(np.float64)

    # Pre-load all datasets so M3 cross-task error can compare matched test curves.
    datasets = {}
    for task in TASK_ORDER:
        datasets[task] = {}
        for seed in SEEDS:
            path = Path(TASK_FILES[task].format(seed=seed))
            if not path.exists():
                raise FileNotFoundError(f"missing dataset: {path}")
            datasets[task][seed] = load_dataset(path, args.bins)

    args.output_root.mkdir(parents=True, exist_ok=True)
    fit_rows = []
    cross_rows = []
    for task in TASK_ORDER:
        for seed in SEEDS:
            ds = datasets[task][seed]
            cid_to_idx = {cid: i for i, cid in enumerate(ds["mixed_cids"])}
            for subset in subset_manifest["subsets"]:
                sample_size = subset["sample_size"]
                source_cids = subset["source_condition_ids"]
                missing = [cid for cid in source_cids if cid not in cid_to_idx]
                if missing:
                    # A source mixed condition was not realized in this seed.
                    for model in MODEL_ORDER:
                        fit_rows.append(
                            dict(
                                task=task, seed=seed, subset_id=subset["subset_id"],
                                protocol=subset["protocol"], sample_size=sample_size,
                                repeat=subset["repeat"], model=model,
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
                with h5py.File(Path(TASK_FILES[task].format(seed=seed)), "r") as data_file:
                    frame_xy, frame_yaw = nominal_frame(data_file, keys, contexts)
                models = build_models(frame_xy, frame_yaw, pdiag_config)
                for model in models:
                    started = time.perf_counter()
                    try:
                        model.fit(contexts, curves, progress, phase_codes)
                        profile = model.jacobian_diag()
                        alpha_yaw = profile[:, 2]
                        pre = float(alpha_yaw[phase_codes < 2].mean())
                        post = float(alpha_yaw[phase_codes >= 2].mean())
                        oracle = oracle_alpha_yaw(task, phase_codes)
                        e_alpha = float(np.mean((alpha_yaw - oracle) ** 2))
                        switch = switch_diagnostics(progress, alpha_yaw, unlock_start)
                        fit_success = True
                        fit_error = ""
                        # M3: predict on own test contexts, compare to every task's curves.
                        if len(ds["test_contexts"]) > 0:
                            prediction = model.predict(ds["test_contexts"])
                            for other_task in TASK_ORDER:
                                other = datasets[other_task][seed]
                                if other["isolated_cids"] != ds["isolated_cids"]:
                                    continue
                                task_err, task_endpoint = metric_errors(
                                    prediction, other["test_curves"],
                                    yaw_weights=task_yaw_weights,
                                )
                                for idx, gen in enumerate(ds["test_generators"]):
                                    cross_rows.append(
                                        dict(
                                            fit_task=task, eval_task=other_task,
                                            seed=seed, subset_id=subset["subset_id"],
                                            sample_size=sample_size, model=model.name,
                                            generator=gen,
                                            task_error_mm_equiv=float(task_err[idx]),
                                        )
                                    )
                    except Exception as exc:
                        fit_success = False
                        fit_error = repr(exc)
                        pre = post = e_alpha = float("nan")
                        switch = {"detected": False, "status": "fit_failed", "location": float("nan")}
                    fit_rows.append(
                        dict(
                            task=task, seed=seed, subset_id=subset["subset_id"],
                            protocol=subset["protocol"], sample_size=sample_size,
                            repeat=subset["repeat"], model=model.name,
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

    # ---- M1: E_alpha (per model x N x task, mean over seeds/repeats) ----
    m1 = (
        fits[fits.fit_success]
        .groupby(["task", "model", "sample_size"], sort=False)["e_alpha"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # ---- M2: discrimination accuracy ----
    # alpha_pre > 0.5 classifies "keyed"; correct iff (task == "keyed")
    disc = fits[fits.fit_success].copy()
    disc["pred_keyed"] = disc["alpha_pre"] > 0.5
    disc["correct"] = disc["pred_keyed"] == (disc["task"] == "keyed")
    m2 = (
        disc.groupby(["model", "sample_size"], sort=False)["correct"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "accuracy"})
        .reset_index()
    )

    # ---- M2b: symmetry gap G_psi (per model x N x seed x subset) ----
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
    if len(gap_df):
        m2b = (
            gap_df.groupby(["model", "sample_size", "arm"], sort=False)
            .agg(G_psi_mean=("G_psi", "mean"), G_psi_std=("G_psi", "std"),
                 abs_dev_mean=("abs_dev_from_1", "mean"))
            .reset_index()
        )
    else:
        m2b = pd.DataFrame()

    # ---- M0: keyed switch fidelity (Pdiag finite) ----
    keyed_pdiag = fits[
        (fits.task == "keyed") & (fits.model == "Pdiag finite") & fits.fit_success
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

    # ---- M3: cross-task mismatch gap (per model x N x seed) ----
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

    # ---- write outputs ----
    fits_path = args.output_root / "symmetry_transfer_fits.csv"
    m1_path = args.output_root / "symmetry_transfer_M1_ealpha.csv"
    m2_path = args.output_root / "symmetry_transfer_M2_discrimination.csv"
    m2b_path = args.output_root / "symmetry_transfer_M2b_symmetry_gap.csv"
    m0_path = args.output_root / "symmetry_transfer_M0_switch.csv"
    m3_path = args.output_root / "symmetry_transfer_M3_cross_task.csv"
    fits.to_csv(fits_path, index=False)
    m1.to_csv(m1_path, index=False)
    m2.to_csv(m2_path, index=False)
    m2b.to_csv(m2b_path, index=False)
    m0.to_csv(m0_path, index=False)
    m3.to_csv(m3_path, index=False)

    summary = {
        "schema_version": 1,
        "experiment_sha256": sha256(args.experiment),
        "subset_manifest_sha256": sha256(args.subsets),
        "source_sha256": {
            "benchmark_symmetry_transfer.py": sha256(Path(__file__)),
            "phase_switch_baselines.py": sha256(
                Path(__file__).with_name("phase_switch_baselines.py")
            ),
        },
        "model_definitions": MODEL_DEFINITIONS,
        "oracle": {
            "keyed": "alpha_yaw = 1 in align/enter (phase<2), 0 in unlock/insert",
            "circular_honest": "alpha_yaw = 0 everywhere (gauge symmetry)",
            "circular_placebo": "alpha_yaw = 0 everywhere (rotation present, uncorrelated)",
            "translation": "alpha_x = alpha_y = 1 everywhere, both tasks",
        },
        "M1_generator_identification_error": m1.to_dict(orient="records"),
        "M2_discrimination_accuracy": m2.to_dict(orient="records"),
        "M2b_symmetry_gap": m2b.to_dict(orient="records") if len(m2b) else [],
        "M0_keyed_switch_fidelity": m0.to_dict(orient="records"),
        "M3_cross_task_mismatch_gap": m3.to_dict(orient="records") if len(m3) else [],
        "fit_count": int(len(fits)),
        "fit_failure_count": int((~fits.fit_success).sum()),
        "bins_per_phase": args.bins,
    }
    summary_path = args.output_root / "symmetry_transfer_summary.json"
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
    print("\nsaved:", summary_path)


if __name__ == "__main__":
    main()
