from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transforms3d.euler import quat2euler


PHASES = {
    3: "Align keyed",
    4: "Enter key",
    5: "Unlock yaw",
    6: "Circular insert",
}
REQUIRED_PHASES = tuple(PHASES)
ROTATION_SCALE_M_PER_RAD = 0.03


def wrap_pi(value):
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def pose_yaw(pose):
    pose = np.asarray(pose)
    return np.asarray([quat2euler(q)[2] for q in pose[:, 3:]])


def episode_keys(data_file):
    return sorted(
        [key for key in data_file if key.startswith("episode_")],
        key=lambda key: int(key.split("_")[-1]),
    )


def complete(group):
    phases = set(int(value) for value in np.asarray(group["solver_phase"]))
    return all(phase in phases for phase in REQUIRED_PHASES)


def endpoint_task_pose(group, phase):
    phases = np.asarray(group["solver_phase"])
    indices = np.flatnonzero(phases == phase)
    if len(indices) == 0:
        raise ValueError(f"episode has no phase {phase}")
    pose = np.asarray(group["peg_pose"])[indices[-1]]
    return np.array([pose[0], pose[1], quat2euler(pose[3:])[2]])


def preclear_task_pose(group):
    phases = np.asarray(group["solver_phase"])
    clearance = np.asarray(group["key_clearance_margin"])
    candidates = np.flatnonzero((phases == 4) & (clearance <= 0.0))
    if len(candidates) == 0:
        raise ValueError("episode has no keyed pre-clear state")
    index = candidates[np.argmax(clearance[candidates])]
    pose = np.asarray(group["peg_pose"])[index]
    return np.array([pose[0], pose[1], quat2euler(pose[3:])[2]])


def resample(values, bins):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, bins)
    out = np.column_stack(
        [np.interp(target, source, values[:, column]) for column in range(values.shape[1])]
    )
    return out


def phase_task_curve(group, phase, bins):
    phases = np.asarray(group["solver_phase"])
    indices = np.flatnonzero(phases == phase)
    peg = np.asarray(group["peg_pose"])[indices]
    yaw = np.unwrap(pose_yaw(peg))
    return resample(np.column_stack([peg[:, 0], peg[:, 1], yaw]), bins)


def phase_scalar_curve(group, phase, key, bins):
    phases = np.asarray(group["solver_phase"])
    indices = np.flatnonzero(phases == phase)
    values = np.asarray(group[key])[indices]
    return resample(values, bins)[:, 0]


def phase_vector_norm_curve(group, phase, key, bins):
    phases = np.asarray(group["solver_phase"])
    indices = np.flatnonzero(phases == phase)
    values = np.linalg.norm(np.asarray(group[key])[indices], axis=1)
    return resample(values, bins)[:, 0]


def fit_response(causal, responses, intercept=False):
    causal = np.asarray(causal, dtype=np.float64)
    responses = np.asarray(responses, dtype=np.float64)
    design = np.column_stack([np.ones(len(causal)), causal]) if intercept else causal
    coefficients = np.linalg.lstsq(design, responses, rcond=None)[0]
    return coefficients[1:] if intercept else coefficients


def bootstrap_profile(causal, response_by_bin, samples, seed):
    rng = np.random.default_rng(seed)
    n_episodes, n_bins, _ = response_by_bin.shape
    boot = []
    for _ in range(samples):
        indices = rng.integers(0, n_episodes, n_episodes)
        design = np.column_stack(
            [np.ones(n_episodes), causal[indices]]
        )
        if np.linalg.matrix_rank(design) < causal.shape[1] + 1:
            continue
        diagonal = []
        for bin_id in range(n_bins):
            matrix = fit_response(
                causal[indices], response_by_bin[indices, bin_id], intercept=True
            )
            diagonal.append(np.diag(matrix))
        boot.append(diagonal)
    if not boot:
        shape = (n_bins, causal.shape[1])
        return np.full(shape, np.nan), np.full(shape, np.nan)
    boot = np.asarray(boot)
    return np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def weighted_endpoint_error(predicted, observed):
    residual = np.asarray(predicted) - np.asarray(observed)
    residual = residual.copy()
    residual[:, 2] = np.asarray([wrap_pi(value) for value in residual[:, 2]])
    residual[:, 2] *= ROTATION_SCALE_M_PER_RAD
    return np.linalg.norm(residual, axis=1) * 1000.0


def paired_bootstrap_mean_ci(values, samples=10000, seed=20260818):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), (samples, len(values)))
    means = values[indices].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def analyze(path: Path, bins: int, bootstrap_samples: int, strict: bool):
    output_root = path.parent
    stem = path.stem
    integrity_rows = []
    yaw_rows = []
    translation_rows = []

    with h5py.File(path, "r") as data_file:
        keys = episode_keys(data_file)
        for key in keys:
            group = data_file[key]
            force = np.linalg.norm(np.asarray(group["contact_force"]), axis=1)
            socket = np.asarray(group["socket_pose"])
            phases = sorted(set(int(value) for value in np.asarray(group["solver_phase"])))
            causal = np.asarray(group["causal_delta"])
            terminated = np.asarray(group["terminated"], dtype=bool)
            truncated = np.asarray(group["truncated"], dtype=bool)
            done = terminated | truncated
            post_done_actions = (
                0
                if not np.any(done)
                else len(done) - 1 - int(np.flatnonzero(done)[0])
            )
            integrity_rows.append(
                dict(
                    episode=key,
                    condition_id=int(group.attrs.get("condition_id", -1)),
                    attempt_id=int(group.attrs.get("attempt_id", 0)),
                    generator=str(group.attrs.get("generator", "")),
                    steps=len(group["actions"]),
                    success=bool(np.asarray(group["success"])[-1]),
                    complete=all(phase in phases for phase in REQUIRED_PHASES),
                    max_contact_force_N=float(force[1:].max()),
                    contact_frames=int(np.sum(force[1:] > 1e-3)),
                    max_socket_drift_m=float(
                        np.max(np.linalg.norm(socket[:, :3] - socket[0, :3], axis=1))
                    ),
                    final_distance_m=float(np.asarray(group["obj_to_goal_dist"])[-1]),
                    final_axis_error_rad=float(np.asarray(group["axis_angle_err"])[-1]),
                    final_key_clearance_m=float(
                        np.asarray(group["key_clearance_margin"])[-1]
                    ),
                    dx=float(causal[0]),
                    dy=float(causal[1]),
                    dyaw=float(causal[2]),
                    any_terminated=bool(terminated.any()),
                    any_truncated=bool(truncated.any()),
                    post_done_actions=post_done_actions,
                    stop_reason=str(group.attrs.get("stop_reason", "")),
                    solver_error=str(group.attrs.get("solver_error", "")),
                )
            )

        usable = [
            key
            for key in keys
            if bool(np.asarray(data_file[key]["success"])[-1])
            and complete(data_file[key])
            and not bool(np.asarray(data_file[key]["truncated"]).any())
        ]
        baseline_candidates = [
            key
            for key in usable
            if np.linalg.norm(np.asarray(data_file[key]["causal_delta"])) < 1e-10
        ]
        if not baseline_candidates:
            raise RuntimeError("no successful zero-intervention baseline")
        baseline = data_file[baseline_candidates[0]]

        yaw_keys = [
            key
            for key in usable
            if str(data_file[key].attrs.get("generator", "")) == "yaw"
            and abs(float(np.asarray(data_file[key]["causal_delta"])[2])) > 1e-10
        ]
        for key in yaw_keys:
            group = data_file[key]
            delta_yaw = float(np.asarray(group["causal_delta"])[2])
            row = dict(episode=key, intervention_yaw_deg=np.rad2deg(delta_yaw))
            observed_preclear = preclear_task_pose(group)
            reference_preclear = preclear_task_pose(baseline)
            row["Keyed pre-clear_ratio"] = abs(
                wrap_pi(observed_preclear[2] - reference_preclear[2]) / delta_yaw
            )
            for phase, label in PHASES.items():
                observed = endpoint_task_pose(group, phase)
                reference = endpoint_task_pose(baseline, phase)
                ratio = abs(wrap_pi(observed[2] - reference[2]) / delta_yaw)
                row[f"{label}_ratio"] = ratio
            yaw_rows.append(row)

        translation_keys = [
            key
            for key in usable
            if str(data_file[key].attrs.get("generator", "")) == "translation"
        ]
        baseline_final = endpoint_task_pose(baseline, 6)
        for key in translation_keys:
            group = data_file[key]
            causal = np.asarray(group["causal_delta"])
            observed = endpoint_task_pose(group, 6)
            ratio = np.linalg.norm(observed[:2] - baseline_final[:2]) / np.linalg.norm(
                causal[:2]
            )
            translation_rows.append(
                dict(
                    episode=key,
                    dx=causal[0],
                    dy=causal[1],
                    final_translation_ratio=ratio,
                )
            )

        mixed_keys = [
            key
            for key in usable
            if str(data_file[key].attrs.get("generator", "")) == "mixed"
        ]
        if len(mixed_keys) < 6:
            raise RuntimeError(f"need at least 6 successful mixed episodes, found {len(mixed_keys)}")
        causal = np.asarray(
            [np.asarray(data_file[key]["causal_delta"]) for key in mixed_keys]
        )
        if np.linalg.matrix_rank(causal) < 3:
            raise RuntimeError("successful mixed interventions do not excite all 3 generators")

        profile_rows = []
        profile_arrays = []
        lower_arrays = []
        upper_arrays = []
        force_profiles = []
        clearance_profiles = []
        global_offset = 0
        final_mixed_response = None
        for phase, label in PHASES.items():
            responses = []
            for key in mixed_keys:
                group = data_file[key]
                curve = phase_task_curve(group, phase, bins)
                responses.append(curve)
            phase_forces = [
                phase_vector_norm_curve(
                    data_file[key], phase, "contact_force", bins
                )
                for key in usable
            ]
            phase_clearances = [
                phase_scalar_curve(
                    data_file[key], phase, "key_clearance_margin", bins
                )
                for key in usable
            ]
            responses = np.asarray(responses)
            diagonals = []
            dense_matrices = []
            for bin_id in range(bins):
                matrix = fit_response(causal, responses[:, bin_id], intercept=True)
                dense_matrices.append(matrix)
                diagonals.append(np.diag(matrix))
            diagonals = np.asarray(diagonals)
            dense_matrices = np.asarray(dense_matrices)
            lower, upper = bootstrap_profile(
                causal, responses, bootstrap_samples, seed=20260818 + phase
            )
            profile_arrays.append(diagonals)
            lower_arrays.append(lower)
            upper_arrays.append(upper)
            force_profiles.append(np.max(np.asarray(phase_forces), axis=0))
            clearance_profiles.append(np.median(phase_clearances, axis=0))
            for bin_id in range(bins):
                profile_rows.append(
                    dict(
                        phase=label,
                        phase_code=phase,
                        phase_progress=bin_id / (bins - 1),
                        global_bin=global_offset + bin_id,
                        alpha_x=diagonals[bin_id, 0],
                        alpha_y=diagonals[bin_id, 1],
                        alpha_yaw=diagonals[bin_id, 2],
                        alpha_x_ci_low=lower[bin_id, 0],
                        alpha_x_ci_high=upper[bin_id, 0],
                        alpha_y_ci_low=lower[bin_id, 1],
                        alpha_y_ci_high=upper[bin_id, 1],
                        alpha_yaw_ci_low=lower[bin_id, 2],
                        alpha_yaw_ci_high=upper[bin_id, 2],
                        off_diagonal_fro=float(
                            np.linalg.norm(
                                dense_matrices[bin_id]
                                - np.diag(np.diag(dense_matrices[bin_id]))
                            )
                        ),
                    )
                )
            if phase == 6:
                final_mixed_response = responses[:, -1]
            global_offset += bins

        final_matrix = fit_response(causal, final_mixed_response, intercept=True)
        final_diagonal = np.diag(final_matrix)
        metric_scale = np.array([1.0, 1.0, ROTATION_SCALE_M_PER_RAD])
        centered_causal = causal - causal.mean(axis=0, keepdims=True)
        centered_response = final_mixed_response - final_mixed_response.mean(
            axis=0, keepdims=True
        )
        scaled_causal = centered_causal * metric_scale
        scaled_response = centered_response * metric_scale
        scalar_weight = float(
            np.sum(scaled_causal * scaled_response)
            / np.sum(scaled_causal * scaled_causal)
        )
        isolated_keys = yaw_keys + translation_keys
        isolated_causal = np.asarray(
            [np.asarray(data_file[key]["causal_delta"]) for key in isolated_keys]
        )
        isolated_observed = np.asarray(
            [endpoint_task_pose(data_file[key], 6) - baseline_final for key in isolated_keys]
        )
        isolated_observed[:, 2] = np.asarray(
            [wrap_pi(value) for value in isolated_observed[:, 2]]
        )
        scalar_prediction = scalar_weight * isolated_causal
        diagonal_prediction = isolated_causal * final_diagonal
        scalar_error = weighted_endpoint_error(scalar_prediction, isolated_observed)
        diagonal_error = weighted_endpoint_error(diagonal_prediction, isolated_observed)
        error_rows = []
        for index, key in enumerate(isolated_keys):
            error_rows.append(
                dict(
                    episode=key,
                    generator=str(data_file[key].attrs.get("generator", "")),
                    scalar_error_mm_equiv=scalar_error[index],
                    diagonal_error_mm_equiv=diagonal_error[index],
                )
            )

    integrity = pd.DataFrame(integrity_rows)
    yaw_frame = pd.DataFrame(yaw_rows)
    translation_frame = pd.DataFrame(translation_rows)
    profile = pd.DataFrame(profile_rows)
    errors = pd.DataFrame(error_rows)

    integrity_path = output_root / f"{stem}_integrity.csv"
    yaw_path = output_root / f"{stem}_isolated_yaw_phase_response.csv"
    translation_path = output_root / f"{stem}_isolated_translation_response.csv"
    profile_path = output_root / f"{stem}_learned_generator_profile.csv"
    error_path = output_root / f"{stem}_heldout_model_errors.csv"
    integrity.to_csv(integrity_path, index=False)
    yaw_frame.to_csv(yaw_path, index=False)
    translation_frame.to_csv(translation_path, index=False)
    profile.to_csv(profile_path, index=False)
    errors.to_csv(error_path, index=False)

    yaw_preclear = float(yaw_frame["Keyed pre-clear_ratio"].mean())
    yaw_enter = float(yaw_frame["Enter key_ratio"].mean())
    yaw_unlock = float(yaw_frame["Unlock yaw_ratio"].mean())
    yaw_final = float(yaw_frame["Circular insert_ratio"].mean())
    translation_final = float(translation_frame["final_translation_ratio"].mean())
    scalar_mean = float(errors["scalar_error_mm_equiv"].mean())
    diagonal_mean = float(errors["diagonal_error_mm_equiv"].mean())
    paired_improvement = errors.scalar_error_mm_equiv - errors.diagonal_error_mm_equiv
    paired_improvement_ci = paired_bootstrap_mean_ci(paired_improvement)

    condition_groups = integrity.groupby("condition_id", sort=True)
    condition_ids = sorted(integrity.condition_id.unique())
    with h5py.File(path, "r") as metadata_file:
        expected_condition_count = int(
            metadata_file.attrs.get("condition_count", len(condition_ids))
        )
    attempts_contiguous = all(
        sorted(group.attempt_id.tolist()) == list(range(int(group.attempt_id.max()) + 1))
        for _, group in condition_groups
    )
    one_success_per_condition = all(
        int((group.success & group.complete & ~group.any_truncated).sum()) == 1
        for _, group in condition_groups
    )
    all_conditions_covered = (
        condition_ids == list(range(expected_condition_count))
        and one_success_per_condition
    )
    mixed_vectors = integrity.loc[integrity.generator == "mixed", ["dx", "dy", "dyaw"]].drop_duplicates().to_numpy()
    isolated_vectors = integrity.loc[
        integrity.generator.isin(["yaw", "translation"]), ["dx", "dy", "dyaw"]
    ].drop_duplicates().to_numpy()
    train_holdout_disjoint = not any(
        np.all(np.isclose(mixed_vector, isolated_vectors, atol=1e-12), axis=1).any()
        for mixed_vector in mixed_vectors
    )
    checks = {
        "H1_translation_preserved": translation_final >= 0.8,
        "H2_keyed_yaw_relevant_before_clearance": yaw_preclear >= 0.8,
        "H3_circular_yaw_suppressed": yaw_unlock <= 0.25 and yaw_final <= 0.25,
        "H4_diagonal_beats_scalar": paired_improvement_ci[0] > 0.0,
        "physics_contact_present_in_usable_episode": bool(
            (
                integrity.success
                & integrity.complete
                & ~integrity.any_truncated
                & (integrity.contact_frames > 0)
            ).any()
        ),
        "socket_kinematic": float(integrity.max_socket_drift_m.max()) < 1e-6,
        "mixed_excitation_full_rank": int(np.linalg.matrix_rank(causal)) == 3,
        "all_conditions_have_one_usable_success": all_conditions_covered,
        "attempts_are_contiguous_and_retained": attempts_contiguous,
        "train_holdout_interventions_disjoint": train_holdout_disjoint,
        "no_post_terminal_actions": bool((integrity.post_done_actions == 0).all()),
        "no_analyzed_truncations": not bool(
            integrity.loc[integrity.success & integrity.complete, "any_truncated"].any()
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    summary = {
        "trajectory": str(path),
        "episodes": int(len(integrity)),
        "successful_episodes": int(integrity.success.sum()),
        "successful_mixed_episodes": int(len(mixed_keys)),
        "contact_episodes": int((integrity.contact_frames > 0).sum()),
        "yaw_preclear_mean_ratio": yaw_preclear,
        "yaw_enter_mean_ratio": yaw_enter,
        "yaw_unlock_mean_ratio": yaw_unlock,
        "yaw_final_mean_ratio": yaw_final,
        "translation_final_mean_ratio": translation_final,
        "phase_endpoint_diagonal": {
            label: profile.loc[profile.phase_code == phase, ["alpha_x", "alpha_y", "alpha_yaw"]]
            .iloc[-1]
            .to_dict()
            for phase, label in PHASES.items()
        },
        "final_diagonal": final_diagonal.tolist(),
        "final_scalar_weight": scalar_weight,
        "heldout_scalar_mean_mm_equiv": scalar_mean,
        "heldout_diagonal_mean_mm_equiv": diagonal_mean,
        "paired_improvement_mean_mm_equiv": float(paired_improvement.mean()),
        "paired_improvement_95ci_mm_equiv": paired_improvement_ci.tolist(),
        "heldout_relative_improvement": 1.0 - diagonal_mean / scalar_mean,
        "checks": checks,
    }
    summary_path = output_root / f"{stem}_validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
        }
    )
    colors = {"x": "#0072B2", "y": "#009E73", "yaw": "#D55E00"}
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), constrained_layout=True)
    axis = axes[0, 0]
    profile_values = np.concatenate(profile_arrays, axis=0)
    profile_low = np.concatenate(lower_arrays, axis=0)
    profile_high = np.concatenate(upper_arrays, axis=0)
    global_x = np.arange(len(profile_values))
    line_styles = {"x": "-", "y": "--", "yaw": "-."}
    for index, (name, color) in enumerate(colors.items()):
        axis.plot(
            global_x,
            profile_values[:, index],
            color=color,
            linestyle=line_styles[name],
            label=fr"$\alpha_{{{name}}}$",
        )
        axis.fill_between(
            global_x,
            profile_low[:, index],
            profile_high[:, index],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    for boundary in range(1, len(PHASES)):
        axis.axvline(boundary * bins - 0.5, color="0.72", linewidth=0.8)
    axis.set_xticks([index * bins + (bins - 1) / 2 for index in range(len(PHASES))])
    axis.set_xticklabels([label.replace(" ", "\n") for label in PHASES.values()])
    axis.set_ylabel("Recovered generator response")
    axis.set_ylim(-0.2, 1.25)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    axis.set_title("A  Phase-dependent generator profile", loc="left", fontweight="bold")

    axis = axes[0, 1]
    phase_columns = [
        "Align keyed_ratio",
        "Keyed pre-clear_ratio",
        "Unlock yaw_ratio",
        "Circular insert_ratio",
    ]
    x = np.arange(len(phase_columns))
    for _, row in yaw_frame.iterrows():
        axis.plot(
            x,
            row[phase_columns].to_numpy(dtype=float),
            color="#999999",
            marker="o",
            markersize=3,
            linewidth=0.8,
            alpha=0.7,
        )
    axis.plot(
        x,
        yaw_frame[phase_columns].mean().to_numpy(),
        color=colors["yaw"],
        marker="o",
        linewidth=2.0,
        label="Mean",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(["Align\nkeyed", "Keyed\npre-clear", "Unlock\nyaw", "Circular\ninsert"])
    axis.set_ylabel("Isolated yaw propagation ratio")
    axis.set_ylim(-0.05, 1.15)
    axis.set_title("B  Held-out axial-yaw switch", loc="left", fontweight="bold")

    axis = axes[1, 0]
    positions = np.arange(len(errors))
    axis.plot(
        [0, 1],
        [scalar_mean, diagonal_mean],
        color="0.55",
        linewidth=1.2,
        marker="o",
    )
    axis.scatter(
        np.zeros(len(errors)), errors.scalar_error_mm_equiv, color="#999999", s=15, alpha=0.65
    )
    axis.scatter(
        np.ones(len(errors)), errors.diagonal_error_mm_equiv, color="#0072B2", s=15, alpha=0.65
    )
    axis.set_xticks([0, 1], ["Scalar $wI$", "Generator diagonal"])
    axis.set_ylabel("Held-out endpoint error (mm-equiv.)")
    axis.set_title("C  Structural scalar compromise", loc="left", fontweight="bold")

    axis = axes[1, 1]
    force_curve = np.concatenate(force_profiles)
    clearance_curve = np.concatenate(clearance_profiles) * 1000.0
    axis.plot(global_x, clearance_curve, color="#009E73", label="Key clearance")
    axis.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
    axis.set_ylabel("Key clearance (mm)", color="#007A58")
    axis.tick_params(axis="y", labelcolor="#007A58")
    contact_axis = axis.twinx()
    contact_axis.plot(global_x, force_curve, color="#CC79A7", label="Contact force")
    contact_axis.set_ylabel("Max. contact force (N)", color="#A23B78")
    contact_axis.tick_params(axis="y", labelcolor="#A23B78")
    for boundary in range(1, len(PHASES)):
        axis.axvline(boundary * bins - 0.5, color="0.72", linewidth=0.8)
    axis.set_xticks([index * bins + (bins - 1) / 2 for index in range(len(PHASES))])
    axis.set_xticklabels([label.replace(" ", "\n") for label in PHASES.values()])
    axis.set_title("D  Physical gate transition", loc="left", fontweight="bold")

    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure_path_png = output_root / f"{stem}_phase_switch_validation.png"
    figure_path_pdf = output_root / f"{stem}_phase_switch_validation.pdf"
    figure.savefig(figure_path_png, dpi=300, bbox_inches="tight", pad_inches=0.06)
    figure.savefig(figure_path_pdf, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)

    print(json.dumps(summary, indent=2))
    print("saved:", summary_path)
    print("saved:", figure_path_png)
    print("saved:", figure_path_pdf)
    if strict and not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit("failed validation checks: " + ", ".join(failed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    analyze(args.trajectory, args.bins, args.bootstrap_samples, args.strict)


if __name__ == "__main__":
    main()
