from __future__ import annotations

"""Model-aware Gauss--Newton identifiability (Experiment A).

Frozen definition (see experiment freeze):
  * metric-weighted Gauss--Newton information, NOT "Fisher information":
        I_D(theta) = sum_{n,k} J_nk(theta)^T W J_nk(theta),
        J_nk = d yhat(c_n, s_k; theta) / d theta.
  * W = diag(1, 1, ell^2) with ell = 30 mm/rad; equivalently the residual is
    pre-scaled by S = diag(1, 1, ell) so W = S^T S and, with the existing
    METRIC_SCALE = [1, 1, 0.03], W_SQRT = [1, 1, 0.03].
  * I_post  = I(theta_hat_D): evaluated at each subset's OWN fitted parameters.
  * I_design = I(theta_ref): theta_ref = 0 (neutral prior, alpha = 0.625) with a
    fixed zero-intervention baseline curve as the nominal reference. Only the
    subset's contexts vary, so this is a pure design-level excitation metric.
  * Report the data-only J^T W J; the second-difference regularizer is NOT added
    (it would inflate zero eigenvalues). A numerical ridge is added only for the
    log-det and trace-inverse summaries and is recorded explicitly.

The information is computed analytically through the chain rule
    d yhat / d theta = (d yhat / d alpha) * (d alpha / d theta),
where d yhat / d alpha is the SE(2) geometric propagation (central differences
on alpha) and d alpha_kj / d theta_jb = alpha_max * sigmoid'(z_kj) * phi_kb.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd

from analyze_phase_switch_rollouts import pose_yaw
from benchmark_phase_switch_baselines import progress_grid, task_curve, usable
from phase_switch_baselines import METRIC_SCALE, SmoothFinitePDiagModel
from run_phase_switch_fewshot import nominal_frame


# Frozen constants.
W_SQRT = np.array([1.0, 1.0, float(METRIC_SCALE[2])], dtype=np.float64)  # [1,1,0.03]
ALPHA_FINITE_DIFF_EPS = 1e-6
RIDGE_RELATIVE = 1e-8  # numerical ridge, relative to the mean eigenvalue


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gauss_newton_info(model, contexts, nominal_curve, theta, basis, alpha_max):
    """Return J^T W J for the finite-action map at the given theta.

    model supplies the nominal frame and the SE(2) finite-action geometry;
    nominal_curve is the nominal task curve X0(s); theta is [3, n_basis].
    """
    contexts = np.asarray(contexts, dtype=np.float64)
    nominal_curve = np.asarray(nominal_curve, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    n_basis = basis.shape[1]

    logits = basis @ np.asarray(theta, dtype=np.float64).reshape(3, -1).T  # [100, 3]
    sig = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    alpha = alpha_max * sig
    d_alpha_dlogit = alpha_max * sig * (1.0 - sig)  # [100, 3]

    # Geometric propagation G[n, k, d, j] = d yhat[n, k, d] / d alpha[k, j].
    eps = ALPHA_FINITE_DIFF_EPS
    G = np.empty((len(contexts), basis.shape[0], 3, 3), dtype=np.float64)
    for j in range(3):
        ap = alpha.copy()
        am = alpha.copy()
        ap[:, j] += eps
        am[:, j] -= eps
        yp = model._predict_with(contexts, nominal_curve, ap)
        ym = model._predict_with(contexts, nominal_curve, am)
        G[:, :, :, j] = (yp - ym) / (2.0 * eps)

    # J[n, k, d, j, b] = G[n,k,d,j] * d_alpha_dlogit[k,j] * basis[k,b]
    J5 = np.einsum("nkdj,kj,kb->nkdjb", G, d_alpha_dlogit, basis)
    J = J5.reshape(len(contexts), basis.shape[0], 3, 3 * n_basis)  # [N, 100, 3, 72]
    Jw = J * W_SQRT[None, None, :, None]
    Jw = Jw.reshape(-1, 3 * n_basis)
    return Jw.T @ Jw


def info_metrics(info):
    n = info.shape[0]
    eigenvalues = np.linalg.eigvalsh(info)  # ascending
    lam_min = float(max(eigenvalues[0], 0.0))  # clip tiny numerical negatives
    lam_max = float(eigenvalues[-1])
    if lam_max > 1e-300:
        normalized = eigenvalues / lam_max  # scale-free spectrum, max = 1
        normalized = np.clip(normalized, 0.0, None)
    else:
        normalized = np.clip(eigenvalues, 0.0, None)
    ridge = RIDGE_RELATIVE
    reg = normalized + ridge
    return {
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "lambda_ratio": float(lam_min / lam_max) if lam_max > 1e-300 else 0.0,
        "log_det": float(np.sum(np.log(reg))),
        "trace_inv": float(np.sum(1.0 / reg)),
        "ridge": ridge,
    }


def collect_seed(experiment, subset_manifest, seed, args):
    config = {
        "alpha_max": experiment["frozen_model"]["alpha_max"],
        "n_basis": experiment["frozen_model"]["n_basis"],
        "basis_width": experiment["frozen_model"]["basis_width"],
        "smoothness_weight": experiment["frozen_model"]["smoothness_weight"],
        "nominal_iterations": experiment["frozen_model"]["nominal_iterations"],
    }
    alpha_max = float(config["alpha_max"])
    n_basis = int(config["n_basis"])
    dataset = args.experiment.parent / "rollouts" / f"seed_{seed}.h5"
    progress, phase_codes = progress_grid(args.bins)

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
        if len(mixed_keys) != 30:
            raise RuntimeError(f"expected 30 mixed conditions, got {len(mixed_keys)}")
        all_contexts = np.asarray(
            [np.asarray(data_file[key]["causal_delta"]) for key in mixed_keys]
        )
        all_curves = np.asarray(
            [task_curve(data_file[key], args.bins) for key in mixed_keys]
        )
        baseline_keys = [
            key
            for _, key in sorted(usable_groups.items())
            if str(data_file[key].attrs["generator"]) == "yaw"
            and np.linalg.norm(np.asarray(data_file[key]["causal_delta"])) <= 1e-12
        ]
        if len(baseline_keys) != 1:
            raise RuntimeError("expected exactly one zero-intervention baseline")
        baseline_key = baseline_keys[0]
        baseline_curve = task_curve(data_file[baseline_key], args.bins)
        ref_socket = np.asarray(data_file[baseline_key]["socket_pose"])[0]
        ref_frame_xy = ref_socket[:2].astype(np.float64)
        ref_frame_yaw = float(pose_yaw(ref_socket[None])[0])

    ref_model = SmoothFinitePDiagModel(
        nominal_frame_xy=ref_frame_xy,
        nominal_frame_yaw=ref_frame_yaw,
        **config,
    )
    basis_shared = ref_model._basis(progress)
    theta_ref = np.zeros((3, n_basis), dtype=np.float64)

    rows = []
    with h5py.File(dataset, "r") as data_file:
        for subset in subset_manifest["subsets"]:
            indices = np.asarray(subset["mixed_indices"], dtype=int)
            contexts = all_contexts[indices]
            curves = all_curves[indices]
            keys = [mixed_keys[index] for index in indices]
            frame_xy, frame_yaw = nominal_frame(data_file, keys, contexts)

            model = SmoothFinitePDiagModel(
                nominal_frame_xy=frame_xy,
                nominal_frame_yaw=frame_yaw,
                **config,
            )
            started = time.perf_counter()
            model.fit(contexts, curves, progress, phase_codes)

            info_post = gauss_newton_info(
                model, contexts, model.nominal_curve, model.parameters,
                model.basis, alpha_max,
            )
            info_design = gauss_newton_info(
                ref_model, contexts, baseline_curve, theta_ref,
                basis_shared, alpha_max,
            )
            post = info_metrics(info_post)
            design = info_metrics(info_design)

            rows.append(
                {
                    "seed": seed,
                    "subset_id": int(subset["subset_id"]),
                    "protocol": subset["protocol"],
                    "sample_size": int(subset["sample_size"]),
                    "repeat": int(subset["repeat"]),
                    "rank": int(subset["rank"]),
                    "augmented_intercept_rank": int(
                        subset["augmented_intercept_rank"]
                    ),
                    "condition_number": float(subset["condition_number"]),
                    "optimization_success": bool(model.optimization_success),
                    "optimization_nfev": int(model.optimization_nfev),
                    "lambda_min_post": post["lambda_min"],
                    "lambda_max_post": post["lambda_max"],
                    "lambda_ratio_post": post["lambda_ratio"],
                    "log_det_post": post["log_det"],
                    "trace_inv_post": post["trace_inv"],
                    "ridge_post": post["ridge"],
                    "lambda_min_design": design["lambda_min"],
                    "lambda_max_design": design["lambda_max"],
                    "lambda_ratio_design": design["lambda_ratio"],
                    "log_det_design": design["log_det"],
                    "trace_inv_design": design["trace_inv"],
                    "ridge_design": design["ridge"],
                    "fit_seconds": time.perf_counter() - started,
                }
            )
    return rows


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
        "--output-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/identifiability"),
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--bins", type=int, default=25)
    args = parser.parse_args()

    with args.experiment.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    with args.subsets.open(encoding="utf-8") as subset_file:
        subset_manifest = json.load(subset_file)
    if subset_manifest["experiment_sha256"] != sha256(args.experiment):
        raise RuntimeError("few-shot subsets do not match experiment")

    seeds = experiment["seeds"] if args.seeds is None else args.seeds
    for seed in seeds:
        if seed not in experiment["seeds"]:
            raise ValueError(f"seed {seed} is not preregistered")

    args.output_root.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        rows = collect_seed(experiment, subset_manifest, seed, args)
        frame = pd.DataFrame(rows)
        path = args.output_root / f"seed_{seed}_info.csv"
        frame.to_csv(path, index=False)
        print(f"seed {seed}: saved {path} ({len(frame)} subsets)")

    with (args.output_root / "identifiability_manifest.json").open(
        "w", encoding="utf-8"
    ) as manifest_file:
        json.dump(
            {
                "schema_version": 1,
                "experiment_sha256": sha256(args.experiment),
                "subset_manifest_sha256": sha256(args.subsets),
                "source_sha256": sha256(Path(__file__)),
                "seeds": seeds,
                "model": "Pdiag finite",
                "information_definition": {
                    "name": "metric-weighted Gauss-Newton information",
                    "I": "sum_{n,k} J_nk^T W J_nk, J_nk = d yhat / d theta",
                    "W": "diag(1, 1, ell^2), ell = 30 mm/rad; W_SQRT = [1, 1, 0.03]",
                    "post_evaluation": "theta_hat_D = each subset's own fitted parameters",
                    "design_evaluation": "theta_ref = 0 (neutral prior alpha=0.625), fixed zero-intervention baseline nominal curve",
                    "regularizer_excluded": "data-only J^T W J; second-difference regularizer not added",
                    "metrics": {
                        "lambda_min": "raw minimum eigenvalue (clipped at 0)",
                        "lambda_ratio": "lambda_min / lambda_max (scale-free inverse condition number)",
                        "log_det": "log det of the lambda_max-normalized spectrum + ridge*I",
                        "trace_inv": "trace of (lambda_max-normalized spectrum + ridge*I)^{-1}",
                    },
                    "normalization": "spectrum is divided by its largest eigenvalue before log_det/trace_inv so metrics are scale-free and comparable across sample sizes",
                    "ridge": RIDGE_RELATIVE,
                    "finite_difference_eps_alpha": ALPHA_FINITE_DIFF_EPS,
                    "jacobian_method": "chain rule: d yhat/d alpha (central diff on alpha) x alpha_max sigmoid'(z) phi",
                },
            },
            manifest_file,
            indent=2,
        )


if __name__ == "__main__":
    main()
