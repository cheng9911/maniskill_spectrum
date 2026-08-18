from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

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


# These primitives intentionally do not import the implementation under test.
def independent_wrap(value):
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def independent_pose2_matrix(pose):
    x, y, yaw = np.asarray(pose, dtype=np.float64)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array(
        [[cosine, -sine, x], [sine, cosine, y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def independent_matrix_pose2(transform):
    return np.array(
        [
            transform[0, 2],
            transform[1, 2],
            np.arctan2(transform[1, 0], transform[0, 0]),
        ],
        dtype=np.float64,
    )


def independent_se2_exp(twist):
    vx, vy, yaw = np.asarray(twist, dtype=np.float64)
    if abs(yaw) < 1e-8:
        a = 1.0 - yaw * yaw / 6.0
        b = 0.5 * yaw - yaw**3 / 24.0
    else:
        a = np.sin(yaw) / yaw
        b = (1.0 - np.cos(yaw)) / yaw
    translated = np.array([[a, -b], [b, a]]) @ np.array([vx, vy])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cosine, -sine, translated[0]],
            [sine, cosine, translated[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def independent_se2_log(transform):
    yaw = np.arctan2(transform[1, 0], transform[0, 0])
    if abs(yaw) < 1e-8:
        a = 1.0 - yaw * yaw / 6.0
        b = 0.5 * yaw - yaw**3 / 24.0
    else:
        a = np.sin(yaw) / yaw
        b = (1.0 - np.cos(yaw)) / yaw
    velocity = np.linalg.solve(
        np.array([[a, -b], [b, a]]), transform[:2, 2]
    )
    return np.array([velocity[0], velocity[1], yaw], dtype=np.float64)


def independent_socket_frame(context, nominal_xy, nominal_yaw):
    return independent_pose2_matrix(
        [
            nominal_xy[0] + context[0],
            nominal_xy[1] + context[1],
            nominal_yaw + context[2],
        ]
    )


def independent_quaternion_yaw(quaternion_wxyz):
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def safe_name(name):
    return name.lower().replace("-", "_").replace(" ", "_")


def validate(output_root: Path, dataset: Path | None, strict: bool):
    summary_path = output_root / "phase_switch_baseline_summary.json"
    table_path = output_root / "phase_switch_baseline_summary.csv"
    errors_path = output_root / "phase_switch_baseline_heldout_errors.csv"
    profiles_path = output_root / "phase_switch_baseline_profiles.csv"
    stratified_path = output_root / "phase_switch_baseline_stratified.csv"
    paired_path = output_root / "phase_switch_baseline_paired_differences.csv"
    arrays_path = output_root / "phase_switch_baseline_arrays.npz"
    for path in [
        summary_path,
        table_path,
        errors_path,
        profiles_path,
        stratified_path,
        paired_path,
        arrays_path,
    ]:
        require(path.is_file() and path.stat().st_size > 0, f"artifact exists: {path}")

    with summary_path.open(encoding="utf-8") as input_file:
        summary = json.load(input_file)
    require(summary["schema_version"] == 2, "recognized summary schema")

    if dataset is None:
        dataset = Path(summary["trajectory"])
    require(dataset.is_file(), f"dataset exists: {dataset}")
    require(sha256(dataset) == summary["trajectory_sha256"], "dataset hash matches benchmark")

    source_root = Path(__file__).parent
    for filename, expected_hash in summary["source_sha256"].items():
        require(
            sha256(source_root / filename) == expected_hash,
            f"source hash matches benchmark: {filename}",
        )

    train_keys = summary["training_split"]["episodes"]
    test_keys = summary["heldout_split"]["episodes"]
    baseline_key = summary["heldout_split"]["baseline_episode"]
    require(len(train_keys) == 30 and len(set(train_keys)) == 30, "30 unique mixed training episodes")
    require(len(test_keys) == 8 and len(set(test_keys)) == 8, "8 unique isolated held-out episodes")
    require(set(train_keys).isdisjoint(test_keys + [baseline_key]), "train and held-out keys are disjoint")

    with h5py.File(dataset, "r") as data_file:
        train_contexts = []
        nominal_xy_candidates = []
        nominal_yaw_candidates = []
        test_contexts = []
        for key in train_keys:
            group = data_file[key]
            require(group.attrs["generator"] == "mixed", f"{key} is mixed training data")
            context = np.asarray(group["causal_delta"], dtype=np.float64)
            train_contexts.append(context)
            socket_pose = np.asarray(group["socket_pose"], dtype=np.float64)[0]
            nominal_xy_candidates.append(socket_pose[:2] - context[:2])
            nominal_yaw_candidates.append(
                independent_wrap(
                    independent_quaternion_yaw(socket_pose[3:]) - context[2]
                )
            )
        for key in test_keys:
            group = data_file[key]
            generator = group.attrs["generator"]
            causal_delta = np.asarray(group["causal_delta"], dtype=np.float64)
            require(generator in {"yaw", "translation"}, f"{key} is isolated held-out data")
            require(np.linalg.norm(causal_delta) > 0, f"{key} is nonzero intervention")
            if generator == "yaw":
                require(
                    np.linalg.norm(causal_delta[:2]) < 1e-12,
                    f"{key} excites yaw only",
                )
            else:
                require(
                    abs(causal_delta[2]) < 1e-12
                    and np.count_nonzero(np.abs(causal_delta[:2]) > 1e-12) == 1,
                    f"{key} excites one translation generator only",
                )
            test_contexts.append(causal_delta)
        require(
            np.linalg.norm(np.asarray(data_file[baseline_key]["causal_delta"])) < 1e-12,
            "baseline intervention is zero",
        )
    centered = np.asarray(train_contexts) - np.mean(train_contexts, axis=0)
    require(np.linalg.matrix_rank(centered) == 3, "mixed training excitation has full rank")
    nominal_xy_candidates = np.asarray(nominal_xy_candidates)
    nominal_yaw_candidates = np.asarray(nominal_yaw_candidates)
    recovered_nominal_xy = nominal_xy_candidates.mean(axis=0)
    recovered_nominal_yaw = float(
        np.arctan2(
            np.sin(nominal_yaw_candidates).mean(),
            np.cos(nominal_yaw_candidates).mean(),
        )
    )
    require(
        np.allclose(
            recovered_nominal_xy,
            summary["nominal_socket_frame"]["xy"],
            atol=1e-12,
        )
        and abs(
            independent_wrap(
                recovered_nominal_yaw - summary["nominal_socket_frame"]["yaw"]
            )
        )
        < 1e-9,
        "nominal socket frame recomputes from mixed training episodes only",
    )

    table = pd.read_csv(table_path)
    errors = pd.read_csv(errors_path)
    profiles = pd.read_csv(profiles_path)
    stratified = pd.read_csv(stratified_path)
    paired = pd.read_csv(paired_path)
    require(table.model.tolist() == list(MODEL_ORDER), "summary model order is fixed")
    require(len(errors) == len(MODEL_ORDER) * 8, "held-out table has one row per model and episode")
    require(len(profiles) == (len(MODEL_ORDER) + 1) * 100, "profile table has 100 points per model and target")
    require(len(stratified) == len(MODEL_ORDER) * 2, "stratified table has yaw and translation rows per model")
    require(
        len(paired) == (len(MODEL_ORDER) - 1) * 8 * 4,
        "paired table has every baseline, episode, and metric difference",
    )
    require(np.isfinite(errors.select_dtypes(include=[np.number])).all().all(), "all held-out errors are finite")
    require(np.isfinite(profiles.select_dtypes(include=[np.number])).all().all(), "all response profiles are finite")

    metric_pairs = {
        "response_error_mm_equiv": "response_error_mean_mm_equiv",
        "response_endpoint_error_mm_equiv": "response_endpoint_error_mean_mm_equiv",
        "task_error_mm_equiv": "task_error_mean_mm_equiv",
        "task_endpoint_error_mm_equiv": "task_endpoint_error_mean_mm_equiv",
    }
    indexed_table = table.set_index("model")
    for model in MODEL_ORDER:
        model_errors = errors[errors.model == model]
        require(set(model_errors.episode) == set(test_keys), f"{model} uses exactly the held-out episodes")
        for error_column, mean_column in metric_pairs.items():
            observed = float(model_errors[error_column].mean())
            recorded = float(indexed_table.loc[model, mean_column])
            require(np.isclose(observed, recorded, atol=1e-12), f"{model} {mean_column} recomputes")

    arrays = np.load(arrays_path)
    require(arrays["train_contexts"].shape == (30, 3), "stored training contexts have shape 30x3")
    require(arrays["test_contexts"].shape == (8, 3), "stored held-out contexts have shape 8x3")
    require(
        np.array_equal(arrays["train_contexts"], np.asarray(train_contexts)),
        "stored training contexts exactly match H5 split",
    )
    require(
        np.array_equal(arrays["test_contexts"], np.asarray(test_contexts)),
        "stored held-out contexts exactly match H5 split",
    )
    unmatched = list(np.asarray(test_contexts))
    for expected_context in EXPECTED_ISOLATED_CONTEXTS:
        matches = [
            index
            for index, context in enumerate(unmatched)
            if np.allclose(context, expected_context, atol=1e-12, rtol=0.0)
        ]
        require(bool(matches), f"isolated grid contains {expected_context.tolist()}")
        unmatched.pop(matches[0])
    require(arrays["test_curves"].shape == (8, 100, 3), "stored held-out curves have shape 8x100x3")
    require(arrays["target_profile"].shape == (100, 3), "empirical target profile has shape 100x3")
    for model in MODEL_ORDER:
        key = safe_name(model)
        prediction = arrays[f"prediction_{key}"]
        profile = arrays[f"profile_{key}"]
        require(prediction.shape == (8, 100, 3), f"{model} prediction shape is valid")
        require(profile.shape == (100, 3), f"{model} profile shape is valid")
        require(np.isfinite(prediction).all() and np.isfinite(profile).all(), f"{model} arrays are finite")

    finite_metadata = summary["pdiag_finite"]
    finite_basis = arrays["pdiag_finite_basis"]
    finite_parameters = arrays["pdiag_finite_parameters"]
    centers = np.linspace(0.0, 1.0, int(finite_metadata["n_basis"]))
    rebuilt_basis = np.exp(
        -0.5
        * (
            (arrays["progress"][:, None] - centers[None, :])
            / float(finite_metadata["basis_width"])
        )
        ** 2
    )
    rebuilt_basis /= rebuilt_basis.sum(axis=1, keepdims=True)
    require(
        np.allclose(finite_basis, rebuilt_basis, atol=1e-15),
        "saved finite Pdiag basis independently rebuilds as normalized RBFs",
    )
    finite_alpha = float(finite_metadata["alpha_max"]) / (
        1.0
        + np.exp(
            -np.clip(finite_basis @ finite_parameters.T, -30.0, 30.0)
        )
    )
    require(
        np.allclose(finite_alpha, arrays["profile_pdiag_finite"], atol=1e-12),
        "finite Pdiag profile exactly follows RBF-sigmoid parameters",
    )
    nominal_xy = np.asarray(summary["nominal_socket_frame"]["xy"])
    nominal_yaw = float(summary["nominal_socket_frame"]["yaw"])
    nominal_frame = independent_pose2_matrix([*nominal_xy, nominal_yaw])
    nominal_inverse = np.linalg.inv(nominal_frame)
    nominal_curve = arrays["pdiag_finite_nominal_curve"]
    realized = np.empty((8, 100, 3), dtype=np.float64)
    for episode, context in enumerate(arrays["test_contexts"]):
        context_twist = independent_se2_log(
            nominal_inverse
            @ independent_socket_frame(context, nominal_xy, nominal_yaw)
        )
        for step in range(100):
            action = (
                nominal_frame
                @ independent_se2_exp(finite_alpha[step] * context_twist)
                @ nominal_inverse
            )
            realized[episode, step] = independent_matrix_pose2(
                action @ independent_pose2_matrix(nominal_curve[step])
            )
    require(
        np.allclose(
            realized,
            arrays["prediction_pdiag_finite"],
            atol=1e-10,
        ),
        "saved finite Pdiag predictions exactly match finite SE(2) realization",
    )

    for model in ["Frame-weighted", "Phase scalar GP"]:
        profile = arrays[f"profile_{safe_name(model)}"]
        require(
            np.allclose(profile[:, 0], profile[:, 1], atol=1e-12)
            and np.allclose(profile[:, 1], profile[:, 2], atol=1e-12),
            f"{model} obeys shared-scalar constraint",
        )

    for model in ["TP-GMM additive", "TP-GMM SE(2)"]:
        diagnostics = summary["tp_gmm"][model]
        cv_scores = {
            int(key): value
            for key, value in diagnostics[
                "grouped_cv_log_likelihood_by_components"
            ].items()
        }
        selected = int(diagnostics["components_selected"])
        require(
            selected == max(cv_scores, key=lambda key: np.mean(cv_scores[key])),
            f"{model} component count is the grouped-training-CV optimum",
        )
        require(
            selected not in {min(cv_scores), max(cv_scores)},
            f"{model} CV optimum is not a search boundary",
        )
        require(diagnostics["final_converged"], f"final {model} EM fit converged")

    target = arrays["target_profile"]
    pdiag = arrays["profile_pdiag_finite"]
    pointwise = arrays["profile_pdiag_pointwise"]
    preclear = 49
    final = 99
    target_translation_final = float(np.mean(target[final, :2]))
    require(target[preclear, 2] > 0.9, "empirical target propagates pre-clear yaw")
    require(abs(target[final, 2]) < 0.1, "empirical target suppresses post-clear yaw")
    require(target_translation_final > 0.9, "empirical target preserves final translation")
    require(pdiag[preclear, 2] > 0.9, "finite Pdiag recovers pre-clear yaw propagation")
    require(abs(pdiag[final, 2]) < 0.1, "finite Pdiag recovers post-clear yaw suppression")
    require(float(np.mean(pdiag[final, :2])) > 0.9, "finite Pdiag preserves final translation")
    require(pointwise[preclear, 2] > 0.9, "pointwise Pdiag recovers pre-clear yaw")
    require(abs(pointwise[final, 2]) < 0.1, "pointwise Pdiag suppresses final yaw")
    alpha_max = float(summary["pdiag_finite"]["alpha_max"])
    require(np.all((pdiag > 0.0) & (pdiag < alpha_max)), "finite Pdiag obeys alpha_max sigmoid constraint")
    require(summary["pdiag_finite"]["optimization_success"], "finite Pdiag optimizer converged")

    comparisons = summary["pdiag_paired_comparisons"]
    for baseline in ["Frame-weighted", "Phase scalar GP"]:
        require(
            comparisons[baseline]["response_error_mm_equiv"]["mean_improvement"] > 0,
            f"finite Pdiag improves full response error over {baseline}",
        )
        require(
            comparisons[baseline]["task_error_mm_equiv"]["mean_improvement"] > 0,
            f"finite Pdiag improves quotient task error over {baseline}",
        )
        require(
            not bool(indexed_table.loc[baseline, "crossing_detected"])
            and indexed_table.loc[baseline, "switch_status"] == "right_censored",
            f"{baseline} no-switch failure is recorded as right-censored",
        )
        require(
            np.isfinite(indexed_table.loc[baseline, "switch_penalized_absolute_error"]),
            f"{baseline} has a finite predeclared switch-failure penalty",
        )
    for baseline in MODEL_ORDER[:-1]:
        for metric in metric_pairs:
            rows = paired[(paired.baseline == baseline) & (paired.metric == metric)]
            require(len(rows) == 8, f"{baseline} {metric} has eight paired differences")
            expected = comparisons[baseline][metric]["mean_improvement"]
            require(
                np.isclose(rows.baseline_minus_pdiag_mm_equiv.mean(), expected),
                f"{baseline} {metric} paired mean recomputes",
            )

    if strict:
        require(
            comparisons["Frame-weighted"]["response_error_mm_equiv"]["bootstrap_95ci"][0] > 0,
            "finite Pdiag response improvement over frame weighting has positive descriptive-bootstrap interval",
        )
        require(
            comparisons["Phase scalar GP"]["response_error_mm_equiv"]["bootstrap_95ci"][0] > 0,
            "finite Pdiag response improvement over phase scalar GP has positive descriptive-bootstrap interval",
        )
        require(
            indexed_table.loc["TP-GMM additive", "g_yaw_preclear"] > 0.9,
            "additive TP-GMM recovers pre-clear yaw",
        )
        for model in ["TP-GMM additive", "TP-GMM SE(2)"]:
            require(
                indexed_table.loc[model, "g_translation_final"] > 0.9,
                f"{model} preserves final translation",
            )
            require(
                abs(indexed_table.loc[model, "g_yaw_final"]) < 0.1,
                f"{model} suppresses post-clear yaw",
            )

    print("All baseline validation checks passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_root",
        type=Path,
        nargs="?",
        default=Path("phase_switch_symmetry_baselines"),
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    validate(args.output_root, args.dataset, args.strict)


if __name__ == "__main__":
    main()
