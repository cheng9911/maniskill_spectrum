from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .analyze_phase_switch_rollouts import (
        complete,
        episode_keys,
        phase_task_curve,
        pose_yaw,
        wrap_pi,
    )
    from .phase_switch_baselines import (
        DiagonalOperatorModel,
        FrameWeightedModel,
        FullOperatorModel,
        GenericConditionalRBF,
        METRIC_SCALE,
        PhaseScalarGPModel,
        SmoothFinitePDiagModel,
        TPGMMModel,
    )
except ImportError:
    from analyze_phase_switch_rollouts import (
        complete,
        episode_keys,
        phase_task_curve,
        pose_yaw,
        wrap_pi,
    )
    from phase_switch_baselines import (
        DiagonalOperatorModel,
        FrameWeightedModel,
        FullOperatorModel,
        GenericConditionalRBF,
        METRIC_SCALE,
        PhaseScalarGPModel,
        SmoothFinitePDiagModel,
        TPGMMModel,
    )


PHASE_CODES = (3, 4, 5, 6)
PHASE_LABELS = ("Align keyed", "Enter key", "Unlock yaw", "Circular insert")
MODEL_ORDER = (
    "Frame-weighted",
    "Phase scalar GP",
    "TP-GMM additive",
    "TP-GMM SE(2)",
    "Generic RBF",
    "Full operator",
    "Pdiag pointwise",
    "Pdiag finite",
)
EXPECTED_ISOLATED_CONTEXTS = np.array(
    [
        [0.0, 0.0, np.deg2rad(-30.0)],
        [0.0, 0.0, np.deg2rad(-15.0)],
        [0.0, 0.0, np.deg2rad(15.0)],
        [0.0, 0.0, np.deg2rad(30.0)],
        [-0.015, 0.0, 0.0],
        [0.015, 0.0, 0.0],
        [0.0, -0.015, 0.0],
        [0.0, 0.015, 0.0],
    ],
    dtype=np.float64,
)

MODEL_DEFINITIONS = {
    "Frame-weighted": "Per-progress affine regression constrained to one shared scalar w(s) for x, y, and axial yaw.",
    "Phase scalar GP": "Oracle-favorable phase-dependent scalar relevance: the training-only least-squares w(s) is GP-smoothed. This is not the TPGP algorithm.",
    "TP-GMM additive": "Shared-responsibility frame-product mixture over world and additive response-coordinate emissions; corresponding conditional Gaussian components are fused by the same product used during fitting and CV.",
    "TP-GMM SE(2)": "Rotation-aware shared-responsibility frame-product mixture using p_local=R(-yaw_socket)(p_world-p_socket), with component-wise task-frame transformation and the same Gaussian-product objective during fitting, CV, and prediction.",
    "Generic RBF": "Unstructured phase-local RBF conditional regressor with context interactions; ridge strength is selected by episode-grouped five-fold CV on mixed training episodes.",
    "Full operator": "Per-progress affine dense 3x3 response operator.",
    "Pdiag pointwise": "Ablation that independently fits one affine diagonal response at each progress point.",
    "Pdiag finite": "Final method: alpha_max-constrained RBF-sigmoid generator profiles with second-difference smoothness, realized by C0 Exp(P(s) Log(C0^-1 C)) C0^-1 X0 in SE(2).",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def usable(group):
    return (
        bool(np.asarray(group["success"])[-1])
        and complete(group)
        and not bool(np.asarray(group["truncated"]).any())
    )


def task_curve(group, bins):
    return np.concatenate(
        [phase_task_curve(group, phase, bins) for phase in PHASE_CODES], axis=0
    )


def progress_grid(bins):
    phase_progress = np.linspace(0.0, 1.0, bins)
    progress = np.concatenate(
        [(phase_index + phase_progress) / len(PHASE_CODES) for phase_index in range(4)]
    )
    phase_codes = np.repeat(np.arange(4), bins)
    return progress, phase_codes


def fit_scalar_slope(context_values, output_values):
    context_values = np.asarray(context_values, dtype=np.float64)
    design = np.column_stack([np.ones(len(context_values)), context_values])
    return np.linalg.lstsq(design, output_values, rcond=None)[0][1]


def empirical_isolated_profile(data_file, baseline_key, isolated_keys, bins):
    baseline_curve = task_curve(data_file[baseline_key], bins)
    diagonal = np.empty((len(baseline_curve), 3), dtype=np.float64)
    for generator in range(3):
        if generator == 0:
            keys = [
                key
                for key in isolated_keys
                if abs(float(np.asarray(data_file[key]["causal_delta"])[0])) > 0
            ]
        elif generator == 1:
            keys = [
                key
                for key in isolated_keys
                if abs(float(np.asarray(data_file[key]["causal_delta"])[1])) > 0
            ]
        else:
            keys = [
                key
                for key in isolated_keys
                if abs(float(np.asarray(data_file[key]["causal_delta"])[2])) > 0
            ]
        contexts = [0.0]
        values = [baseline_curve[:, generator]]
        for key in keys:
            contexts.append(float(np.asarray(data_file[key]["causal_delta"])[generator]))
            values.append(task_curve(data_file[key], bins)[:, generator])
        diagonal[:, generator] = fit_scalar_slope(contexts, np.asarray(values))
    return diagonal


def metric_errors(prediction, target, yaw_weights=None):
    residual = np.asarray(prediction) - np.asarray(target)
    residual = residual.copy()
    residual[..., 2] = np.vectorize(wrap_pi)(residual[..., 2])
    residual *= METRIC_SCALE
    if yaw_weights is not None:
        residual[..., 2] *= np.asarray(yaw_weights)[None, :]
    point_error = np.linalg.norm(residual, axis=-1) * 1000.0
    return point_error.mean(axis=1), point_error[:, -1]


def bootstrap_mean_ci(values, samples=10000, seed=20260820):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), (samples, len(values)))
    means = values[indices].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def switch_diagnostics(progress, yaw_profile, unlock_start):
    progress = np.asarray(progress, dtype=np.float64)
    yaw_profile = np.asarray(yaw_profile, dtype=np.float64)
    for index in range(max(unlock_start, 1), len(progress)):
        y0, y1 = yaw_profile[index - 1], yaw_profile[index]
        if y0 >= 0.5 and y1 < 0.5:
            x0, x1 = progress[index - 1], progress[index]
            if abs(y1 - y0) < 1e-12 or abs(x1 - x0) < 1e-12:
                location = float(x1)
            else:
                fraction = (0.5 - y0) / (y1 - y0)
                location = float(x0 + fraction * (x1 - x0))
            return {"detected": True, "status": "detected", "location": location}
    if yaw_profile[unlock_start] < 0.5:
        status = "left_censored"
    elif yaw_profile[-1] >= 0.5:
        status = "right_censored"
    else:
        status = "no_downward_crossing"
    return {"detected": False, "status": status, "location": float("nan")}


def fit_models(
    contexts,
    curves,
    progress,
    phase_codes,
    nominal_frame_xy,
    nominal_frame_yaw,
    pdiag_config,
):
    models = [
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
    fitted = {}
    for model in models:
        print(f"fitting {model.name}...", flush=True)
        fitted[model.name] = model.fit(contexts, curves, progress, phase_codes)
        if isinstance(model, TPGMMModel):
            print(f"  selected GMM components: {model.n_components}", flush=True)
        if isinstance(model, SmoothFinitePDiagModel):
            print(
                f"  optimization success={model.optimization_success}, "
                f"nfev={model.optimization_nfev}",
                flush=True,
            )
    return fitted


def analyze(path: Path, output_root: Path, bins: int, pdiag_config):
    output_root.mkdir(parents=True, exist_ok=True)
    progress, phase_codes = progress_grid(bins)
    with h5py.File(path, "r") as data_file:
        keys = episode_keys(data_file)
        usable_keys = [key for key in keys if usable(data_file[key])]
        mixed_keys = [
            key
            for key in usable_keys
            if str(data_file[key].attrs.get("generator", "")) == "mixed"
        ]
        isolated_all = [
            key
            for key in usable_keys
            if str(data_file[key].attrs.get("generator", ""))
            in {"yaw", "translation"}
        ]
        baseline_candidates = [
            key
            for key in isolated_all
            if np.linalg.norm(np.asarray(data_file[key]["causal_delta"])) < 1e-12
        ]
        if len(mixed_keys) != 30:
            raise RuntimeError(f"expected 30 successful mixed trajectories, got {len(mixed_keys)}")
        if len(baseline_candidates) != 1:
            raise RuntimeError("expected exactly one zero-intervention baseline")
        baseline_key = baseline_candidates[0]
        isolated_keys = [key for key in isolated_all if key != baseline_key]
        if len(isolated_keys) != 8:
            raise RuntimeError(f"expected 8 held-out isolated trajectories, got {len(isolated_keys)}")

        selected_keys = mixed_keys + isolated_keys + [baseline_key]
        condition_ids = [
            int(data_file[key].attrs.get("condition_id", -1)) for key in selected_keys
        ]
        if len(set(condition_ids)) != len(condition_ids):
            raise RuntimeError("selected train/holdout episodes must have unique condition_id")
        if sum(data_file[key].attrs.get("generator", "") == "yaw" for key in isolated_keys) != 4:
            raise RuntimeError("expected four nonzero isolated yaw interventions")
        if sum(data_file[key].attrs.get("generator", "") == "translation" for key in isolated_keys) != 4:
            raise RuntimeError("expected four nonzero isolated translation interventions")

        train_contexts = np.asarray(
            [np.asarray(data_file[key]["causal_delta"]) for key in mixed_keys]
        )
        train_curves = np.asarray(
            [task_curve(data_file[key], bins) for key in mixed_keys]
        )
        test_contexts = np.asarray(
            [np.asarray(data_file[key]["causal_delta"]) for key in isolated_keys]
        )
        test_curves = np.asarray(
            [task_curve(data_file[key], bins) for key in isolated_keys]
        )
        test_generators = [
            str(data_file[key].attrs.get("generator", "")) for key in isolated_keys
        ]
        test_condition_ids = [
            int(data_file[key].attrs.get("condition_id", -1)) for key in isolated_keys
        ]
        if len(np.unique(train_contexts, axis=0)) != len(train_contexts):
            raise RuntimeError("mixed training interventions must be unique")
        if np.linalg.matrix_rank(train_contexts - train_contexts.mean(axis=0)) != 3:
            raise RuntimeError("mixed training interventions are not full-rank")
        for context, generator in zip(test_contexts, test_generators):
            if generator == "yaw" and np.linalg.norm(context[:2]) >= 1e-12:
                raise RuntimeError("isolated yaw intervention excites translation")
            if generator == "translation" and (
                abs(context[2]) >= 1e-12
                or np.count_nonzero(np.abs(context[:2]) > 1e-12) != 1
            ):
                raise RuntimeError("isolated translation intervention has invalid support")
        unmatched = list(test_contexts)
        for expected_context in EXPECTED_ISOLATED_CONTEXTS:
            matches = [
                index
                for index, context in enumerate(unmatched)
                if np.allclose(context, expected_context, atol=1e-12, rtol=0.0)
            ]
            if not matches:
                raise RuntimeError(
                    "isolated interventions must cover signed yaw and signed x/y grid"
                )
            unmatched.pop(matches[0])
        target_profile = empirical_isolated_profile(
            data_file, baseline_key, isolated_keys, bins
        )

        # Recover C0 from mixed training conditions only. The zero-intervention
        # episode remains reserved for the empirical isolated response target.
        nominal_xy_candidates = []
        nominal_yaw_candidates = []
        for key, context in zip(mixed_keys, train_contexts):
            socket_pose = np.asarray(data_file[key]["socket_pose"])[0]
            nominal_xy_candidates.append(socket_pose[:2] - context[:2])
            socket_yaw = float(pose_yaw(socket_pose[None, :])[0])
            nominal_yaw_candidates.append(wrap_pi(socket_yaw - context[2]))
        nominal_xy_candidates = np.asarray(nominal_xy_candidates, dtype=np.float64)
        nominal_yaw_candidates = np.asarray(
            nominal_yaw_candidates, dtype=np.float64
        )
        nominal_frame_xy = nominal_xy_candidates.mean(axis=0)
        nominal_frame_yaw = float(
            np.arctan2(
                np.sin(nominal_yaw_candidates).mean(),
                np.cos(nominal_yaw_candidates).mean(),
            )
        )
        nominal_xy_spread = float(
            np.max(np.linalg.norm(nominal_xy_candidates - nominal_frame_xy, axis=1))
        )
        nominal_yaw_spread = float(
            np.max(
                np.abs(
                    [wrap_pi(yaw - nominal_frame_yaw) for yaw in nominal_yaw_candidates]
                )
            )
        )
        if nominal_xy_spread > 1e-6 or nominal_yaw_spread > 1e-6:
            raise RuntimeError(
                "mixed conditions do not imply a consistent nominal socket frame: "
                f"xy spread={nominal_xy_spread}, yaw spread={nominal_yaw_spread}"
            )

    models = fit_models(
        train_contexts,
        train_curves,
        progress,
        phase_codes,
        nominal_frame_xy,
        nominal_frame_yaw,
        pdiag_config,
    )
    predictions = {}
    profiles = {"Empirical isolated": target_profile}
    for name, model in models.items():
        predictions[name] = model.predict(test_contexts)
        profiles[name] = model.jacobian_diag()

    unlock_start = 2 * bins
    preclear_index = 2 * bins - 1
    final_index = 4 * bins - 1
    target_switch_diagnostics = switch_diagnostics(
        progress, target_profile[:, 2], unlock_start=unlock_start
    )
    if not target_switch_diagnostics["detected"]:
        raise RuntimeError("empirical isolated yaw profile has no downward 0.5 crossing")
    target_switch = target_switch_diagnostics["location"]
    task_yaw_weights = (phase_codes < 2).astype(np.float64)
    profile_rows = []
    for name, diagonal in profiles.items():
        for step in range(len(progress)):
            profile_rows.append(
                dict(
                    model=name,
                    global_progress=progress[step],
                    phase=PHASE_LABELS[phase_codes[step]],
                    phase_progress=(step % bins) / (bins - 1),
                    alpha_x=diagonal[step, 0],
                    alpha_y=diagonal[step, 1],
                    g_translation=0.5 * (diagonal[step, 0] + diagonal[step, 1]),
                    g_yaw=diagonal[step, 2],
                )
            )

    error_rows = []
    summary_rows = []
    target_trans = 0.5 * (target_profile[:, 0] + target_profile[:, 1])
    target_yaw = target_profile[:, 2]
    for name in MODEL_ORDER:
        response_error, response_endpoint_error = metric_errors(
            predictions[name], test_curves
        )
        task_error, task_endpoint_error = metric_errors(
            predictions[name], test_curves, yaw_weights=task_yaw_weights
        )
        response_ci = bootstrap_mean_ci(response_error)
        response_endpoint_ci = bootstrap_mean_ci(response_endpoint_error)
        task_ci = bootstrap_mean_ci(task_error)
        task_endpoint_ci = bootstrap_mean_ci(task_endpoint_error)
        diagonal = profiles[name]
        trans_profile = 0.5 * (diagonal[:, 0] + diagonal[:, 1])
        yaw_profile = diagonal[:, 2]
        model_switch = switch_diagnostics(progress, yaw_profile, unlock_start)
        if model_switch["detected"]:
            switch_error = abs(model_switch["location"] - target_switch)
            penalized_switch_error = switch_error
        elif model_switch["status"] == "right_censored":
            switch_error = float("nan")
            penalized_switch_error = 1.0 - target_switch
        elif model_switch["status"] == "left_censored":
            switch_error = float("nan")
            penalized_switch_error = target_switch - progress[unlock_start]
        else:
            switch_error = float("nan")
            penalized_switch_error = 1.0
        for index, key in enumerate(isolated_keys):
            error_rows.append(
                dict(
                    model=name,
                    episode=key,
                    condition_id=test_condition_ids[index],
                    generator=test_generators[index],
                    response_error_mm_equiv=response_error[index],
                    response_endpoint_error_mm_equiv=response_endpoint_error[index],
                    task_error_mm_equiv=task_error[index],
                    task_endpoint_error_mm_equiv=task_endpoint_error[index],
                )
            )
        summary_rows.append(
            dict(
                model=name,
                response_error_mean_mm_equiv=float(response_error.mean()),
                response_error_ci_low=float(response_ci[0]),
                response_error_ci_high=float(response_ci[1]),
                response_endpoint_error_mean_mm_equiv=float(
                    response_endpoint_error.mean()
                ),
                response_endpoint_error_ci_low=float(response_endpoint_ci[0]),
                response_endpoint_error_ci_high=float(response_endpoint_ci[1]),
                task_error_mean_mm_equiv=float(task_error.mean()),
                task_error_ci_low=float(task_ci[0]),
                task_error_ci_high=float(task_ci[1]),
                task_endpoint_error_mean_mm_equiv=float(task_endpoint_error.mean()),
                task_endpoint_error_ci_low=float(task_endpoint_ci[0]),
                task_endpoint_error_ci_high=float(task_endpoint_ci[1]),
                generator_rmse=float(np.sqrt(np.mean((diagonal - target_profile) ** 2))),
                translation_profile_rmse=float(
                    np.sqrt(np.mean((trans_profile - target_trans) ** 2))
                ),
                yaw_profile_rmse=float(
                    np.sqrt(np.mean((yaw_profile - target_yaw) ** 2))
                ),
                g_translation_final=float(trans_profile[final_index]),
                g_yaw_preclear=float(yaw_profile[preclear_index]),
                g_yaw_final=float(yaw_profile[final_index]),
                crossing_detected=model_switch["detected"],
                switch_status=model_switch["status"],
                switch_s_0_5=model_switch["location"],
                switch_absolute_error=switch_error,
                switch_penalized_absolute_error=float(penalized_switch_error),
            )
        )

    errors = pd.DataFrame(error_rows)
    summary = pd.DataFrame(summary_rows).set_index("model").loc[list(MODEL_ORDER)].reset_index()
    profile_frame = pd.DataFrame(profile_rows)

    pdiag_errors = errors[errors.model == "Pdiag finite"].set_index("episode")
    paired_comparisons = {}
    paired_rows = []
    for name in MODEL_ORDER:
        if name == "Pdiag finite":
            continue
        baseline_errors = errors[errors.model == name].set_index("episode")
        paired_comparisons[name] = {}
        for metric in [
            "response_error_mm_equiv",
            "response_endpoint_error_mm_equiv",
            "task_error_mm_equiv",
            "task_endpoint_error_mm_equiv",
        ]:
            improvement = (
                baseline_errors.loc[pdiag_errors.index, metric]
                - pdiag_errors[metric]
            ).to_numpy()
            ci = bootstrap_mean_ci(improvement)
            paired_comparisons[name][metric] = {
                "mean_improvement": float(improvement.mean()),
                "bootstrap_95ci": ci.tolist(),
                "all_episodes_positive": bool((improvement > 0).all()),
            }
            for episode, difference in zip(pdiag_errors.index, improvement):
                paired_rows.append(
                    {
                        "baseline": name,
                        "episode": episode,
                        "condition_id": int(
                            pdiag_errors.loc[episode, "condition_id"]
                        ),
                        "generator": pdiag_errors.loc[episode, "generator"],
                        "metric": metric,
                        "baseline_minus_pdiag_mm_equiv": float(difference),
                    }
                )

    profile_path = output_root / "phase_switch_baseline_profiles.csv"
    error_path = output_root / "phase_switch_baseline_heldout_errors.csv"
    table_path = output_root / "phase_switch_baseline_summary.csv"
    stratified_path = output_root / "phase_switch_baseline_stratified.csv"
    paired_path = output_root / "phase_switch_baseline_paired_differences.csv"
    profile_frame.to_csv(profile_path, index=False)
    errors.to_csv(error_path, index=False)
    summary.to_csv(table_path, index=False)
    stratified = (
        errors.groupby(["model", "generator"], sort=False)[
            [
                "response_error_mm_equiv",
                "response_endpoint_error_mm_equiv",
                "task_error_mm_equiv",
                "task_endpoint_error_mm_equiv",
            ]
        ]
        .agg(["mean", "median", "min", "max"])
    )
    stratified.columns = ["_".join(column) for column in stratified.columns]
    stratified.reset_index().to_csv(stratified_path, index=False)
    pd.DataFrame(paired_rows).to_csv(paired_path, index=False)

    npz_payload = {
        "progress": progress,
        "phase_codes": phase_codes,
        "train_contexts": train_contexts,
        "test_contexts": test_contexts,
        "test_curves": test_curves,
        "target_profile": target_profile,
    }
    for name in MODEL_ORDER:
        safe_name = name.lower().replace("-", "_").replace(" ", "_")
        npz_payload[f"prediction_{safe_name}"] = predictions[name]
        npz_payload[f"profile_{safe_name}"] = profiles[name]
    finite_model = models["Pdiag finite"]
    npz_payload["pdiag_finite_nominal_curve"] = finite_model.nominal_curve
    npz_payload["pdiag_finite_basis"] = finite_model.basis
    npz_payload["pdiag_finite_parameters"] = finite_model.parameters
    np.savez_compressed(output_root / "phase_switch_baseline_arrays.npz", **npz_payload)

    result = {
        "schema_version": 2,
        "trajectory": str(path),
        "trajectory_sha256": sha256(path),
        "source_sha256": {
            "benchmark_phase_switch_baselines.py": sha256(Path(__file__)),
            "phase_switch_baselines.py": sha256(
                Path(__file__).with_name("phase_switch_baselines.py")
            ),
            "validate_phase_switch_baselines.py": sha256(
                Path(__file__).with_name("validate_phase_switch_baselines.py")
            ),
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "scikit-learn": importlib.metadata.version("scikit-learn"),
            "h5py": h5py.__version__,
        },
        "training_split": {"generator": "mixed", "episodes": mixed_keys},
        "heldout_split": {
            "generators": ["yaw", "translation"],
            "episodes": isolated_keys,
            "baseline_episode": baseline_key,
        },
        "metric_scale_m_per_rad": float(METRIC_SCALE[2]),
        "task_quotient_definition": "Axial-yaw residual is ignored in unlock_yaw and circular_insert after physical key clearance; x/y residuals remain active in every phase.",
        "interval_definition": "Descriptive bootstrap intervals from resampling the eight fixed held-out interventions; they do not include training-set, planner-seed, or rollout variation.",
        "bins_per_phase": bins,
        "nominal_socket_frame": {
            "xy": nominal_frame_xy.tolist(),
            "yaw": nominal_frame_yaw,
            "source": "Recovered from mixed training socket poses by removing each known causal intervention.",
            "max_xy_spread_m": nominal_xy_spread,
            "max_yaw_spread_rad": nominal_yaw_spread,
        },
        "model_definitions": MODEL_DEFINITIONS,
        "switch_definition": "First bracketed downward crossing of g_yaw(s)=0.5 at or after unlock_yaw. This is response half-decay, not the physical clearance event.",
        "switch_failure_penalty": "Right-censored: 1-target_switch; left-censored: target_switch-unlock_start; other no-crossing: 1.",
        "target_switch_s_0_5": target_switch,
        "tp_gmm": {
            name: {
                "frame_mode": models[name].frame_mode,
                "fitting_objective": models[name].model.objective,
                "components_selected": int(models[name].n_components),
                "grouped_cv_log_likelihood_by_components": models[
                    name
                ].cv_log_likelihood_by_components,
                "cv_splits": int(models[name].cv_splits),
                "cv_n_init": int(models[name].cv_n_init),
                "final_n_init": int(models[name].final_n_init),
                "final_converged": bool(models[name].model.converged_),
                "final_lower_bound": float(models[name].model.lower_bound_),
            }
            for name in ["TP-GMM additive", "TP-GMM SE(2)"]
        },
        "phase_scalar_gp_kernel": str(models["Phase scalar GP"].gp.kernel_),
        "generic_rbf_ridge_alpha": float(models["Generic RBF"].alpha),
        "pdiag_finite": {
            "alpha_max": models["Pdiag finite"].alpha_max,
            "n_basis": models["Pdiag finite"].n_basis,
            "basis_width": models["Pdiag finite"].basis_width,
            "smoothness_weight": models["Pdiag finite"].smoothness_weight,
            "nominal_iterations": models["Pdiag finite"].nominal_iterations,
            "optimization_success": models["Pdiag finite"].optimization_success,
            "optimization_cost": models["Pdiag finite"].optimization_cost,
            "optimization_nfev": models["Pdiag finite"].optimization_nfev,
            "optimization_optimality": models[
                "Pdiag finite"
            ].optimization_optimality,
            "finite_action": "C0 Exp(diag(alpha(s)) Log(C0^-1 C)) C0^-1 X0",
            "nominal_curve_source": "Estimated from mixed training trajectories only by alternating inverse finite actions and pose averaging.",
        },
        "method_alignment_cross_checks": {
            "pdiag_finite_minus_pointwise": {
                metric: float(
                    summary.set_index("model").loc["Pdiag finite", metric]
                    - summary.set_index("model").loc["Pdiag pointwise", metric]
                )
                for metric in [
                    "task_error_mean_mm_equiv",
                    "task_endpoint_error_mean_mm_equiv",
                    "generator_rmse",
                    "switch_absolute_error",
                ]
            },
            "tp_gmm_se2_minus_additive": {
                metric: float(
                    summary.set_index("model").loc["TP-GMM SE(2)", metric]
                    - summary.set_index("model").loc["TP-GMM additive", metric]
                )
                for metric in [
                    "task_error_mean_mm_equiv",
                    "task_endpoint_error_mean_mm_equiv",
                    "generator_rmse",
                    "g_yaw_preclear",
                    "g_yaw_final",
                    "switch_absolute_error",
                ]
            },
        },
        "models": summary.set_index("model").to_dict(orient="index"),
        "pdiag_paired_comparisons": paired_comparisons,
    }
    result_path = output_root / "phase_switch_baseline_summary.json"
    with result_path.open("w", encoding="utf-8") as output_file:
        json.dump(json_ready(result), output_file, indent=2, allow_nan=False)

    create_figure(
        profile_frame,
        errors,
        summary,
        target_switch,
        output_root / "phase_switch_strong_baselines",
    )
    print(summary.to_string(index=False))
    print("saved:", result_path)
    print("saved:", profile_path)
    print("saved:", error_path)
    print("saved:", table_path)
    print("saved:", stratified_path)
    print("saved:", paired_path)


def create_figure(profile, errors, summary, target_switch, output_stem):
    colors = {
        "Frame-weighted": "#E69F00",
        "Phase scalar GP": "#D55E00",
        "TP-GMM additive": "#009E73",
        "TP-GMM SE(2)": "#0072B2",
        "Generic RBF": "#CC79A7",
        "Full operator": "#56B4E9",
        "Pdiag pointwise": "#7A7A7A",
        "Pdiag finite": "#000000",
    }
    line_styles = {
        "Frame-weighted": "--",
        "Phase scalar GP": ":",
        "TP-GMM additive": "-.",
        "TP-GMM SE(2)": (0, (6, 1, 1, 1)),
        "Generic RBF": (0, (5, 2)),
        "Full operator": (0, (3, 1, 1, 1)),
        "Pdiag pointwise": (0, (1, 1)),
        "Pdiag finite": "-",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.6, 5.6), constrained_layout=True)
    target = profile[profile.model == "Empirical isolated"]
    phase_centers = [0.125, 0.375, 0.625, 0.875]

    for axis, column, title, ylabel in [
        (axes[0, 0], "g_translation", "A  Translation response", r"$g_{trans}(s)$"),
        (axes[0, 1], "g_yaw", "B  Axial-yaw response", r"$g_{yaw}(s)$"),
    ]:
        axis.plot(
            target.global_progress,
            target[column],
            color="0.55",
            linewidth=2.8,
            alpha=0.8,
            label="Empirical isolated",
        )
        for name in MODEL_ORDER:
            rows = profile[profile.model == name]
            linewidth = 2.2 if name == "Pdiag finite" else 1.3
            axis.plot(
                rows.global_progress,
                rows[column],
                color=colors[name],
                linestyle=line_styles[name],
                linewidth=linewidth,
                label=name,
            )
        for boundary in [0.25, 0.5, 0.75]:
            axis.axvline(boundary, color="0.82", linewidth=0.7)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(-0.2, 1.3)
        axis.set_xticks(phase_centers, ["Align", "Enter", "Unlock", "Insert"])
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, ncol=3, loc="lower right")
    axes[0, 1].axhline(0.5, color="0.75", linewidth=0.7, linestyle="--")
    axes[0, 1].axvline(target_switch, color="0.45", linewidth=1.0, linestyle=":")

    for axis, error_column, low_column, high_column, title, ylabel in [
        (
            axes[1, 0],
            "task_error_mean_mm_equiv",
            "task_error_ci_low",
            "task_error_ci_high",
            "C  Held-out task-trajectory error",
            "Quotient error (mm-equiv.)",
        ),
        (
            axes[1, 1],
            "task_endpoint_error_mean_mm_equiv",
            "task_endpoint_error_ci_low",
            "task_endpoint_error_ci_high",
            "D  Held-out task-endpoint error",
            "Quotient endpoint error (mm-equiv.)",
        ),
    ]:
        x = np.arange(len(MODEL_ORDER))
        ordered = summary.set_index("model").loc[list(MODEL_ORDER)]
        means = ordered[error_column].to_numpy()
        lower = means - ordered[low_column].to_numpy()
        upper = ordered[high_column].to_numpy() - means
        axis.bar(
            x,
            means,
            color=[colors[name] for name in MODEL_ORDER],
            edgecolor="black",
            linewidth=0.5,
            yerr=np.vstack([lower, upper]),
            capsize=2,
            error_kw={"linewidth": 0.8},
        )
        point_column = (
            "task_error_mm_equiv"
            if error_column == "task_error_mean_mm_equiv"
            else "task_endpoint_error_mm_equiv"
        )
        rng = np.random.default_rng(20260820)
        for index, name in enumerate(MODEL_ORDER):
            for generator, marker in [("yaw", "o"), ("translation", "^")]:
                rows = errors[
                    (errors.model == name) & (errors.generator == generator)
                ]
                values = rows[point_column].to_numpy()
                jitter = rng.uniform(-0.12, 0.12, len(values))
                axis.scatter(
                    index + jitter,
                    values,
                    s=14,
                    marker=marker,
                    color="white",
                    edgecolor="0.25",
                    linewidth=0.5,
                    zorder=3,
                    label=(
                        f"{generator.capitalize()} intervention"
                        if index == 0
                        else None
                    ),
                )
        axis.set_xticks(
            x,
            ["Frame", "Scalar GP", "GMM add.", "GMM SE(2)", "RBF", "Full", "Diag pt.", "Diag finite"],
            rotation=25,
            ha="right",
        )
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False, loc="upper right")

    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.savefig(
        output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.06
    )
    figure.savefig(
        output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trajectory",
        type=Path,
        nargs="?",
        default=Path(
            "phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase_switch_symmetry_baselines"),
    )
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--pdiag-alpha-max", type=float, default=1.25)
    parser.add_argument("--pdiag-basis-count", type=int, default=24)
    parser.add_argument("--pdiag-basis-width", type=float, default=0.065)
    parser.add_argument("--pdiag-smoothness", type=float, default=0.1)
    parser.add_argument("--pdiag-nominal-iterations", type=int, default=3)
    args = parser.parse_args()
    analyze(
        args.trajectory,
        args.output_root,
        args.bins,
        {
            "alpha_max": args.pdiag_alpha_max,
            "n_basis": args.pdiag_basis_count,
            "basis_width": args.pdiag_basis_width,
            "smoothness_weight": args.pdiag_smoothness,
            "nominal_iterations": args.pdiag_nominal_iterations,
        },
    )


if __name__ == "__main__":
    main()
