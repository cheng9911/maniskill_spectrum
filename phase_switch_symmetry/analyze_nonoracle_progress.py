from __future__ import annotations

"""Non-oracle progress reparameterization (Experiment C, reworked evaluation).

Training is unchanged: for each progress parameterization, the phase-dependent
generator law P(s) is re-learned on resampled curves over a fixed [0,1] grid.

Evaluation is now done in the PHYSICAL timeline, not on the progress grid:

  * The task quotient (axial yaw ignored after physical key clearance) is bound
    to the physical key-clear event t_clear of each trajectory, NOT to s=0.5.
  * Held-out predictions are compared to the actual peg pose at each physical
    time step t_i via the trajectory's own progress value s_m(t_i).
  * Switch timing is reported as a physical offset dz_switch = z(t_switch) -
    z(t_clear) in mm, mapped back through the nominal (zero-intervention)
    trajectory, not as a dimensionless |s - s'| between different coordinates.
  * Profile comparison is event-aligned on the nominal physical timeline, so
    alpha_yaw^method and alpha_yaw^oracle are compared at the same physical
    state rather than the same progress number.

Progress parameterizations (frozen):
  * oracle   : four solver phases (3..6), reference / upper bound.
  * time     : normalized step index t / T_ref (naive baseline).
  * arc      : SE(3) arc-length L(t) / L_ref.
  * keypoint : normalized peg-to-socket vertical descent (observable z).
  * dtw      : DTW on [x, y, ell*yaw] -- OPTIMISTIC offline upper bound (leaks
               yaw, which is the quantity under evaluation).
  * dtw-position : DTW on [x, y, z] -- no-yaw-leak variant.

Reference quantities are computed from the 30 mixed TRAINING episodes only; the
zero-intervention baseline is the fixed DTW reference and the nominal mapping
for switch offset / event alignment.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transforms3d.quaternions import quat2mat

from analyze_phase_switch_rollouts import pose_yaw, wrap_pi
from benchmark_phase_switch_baselines import (
    METRIC_SCALE,
    episode_keys,
    progress_grid,
    task_curve,
    usable,
)
from phase_switch_baselines import FrameWeightedModel, SmoothFinitePDiagModel


ROTATION_SCALE = float(METRIC_SCALE[2])  # 0.03 m/rad
PARAM_ORDER = ["oracle", "time", "arc", "keypoint", "dtw", "dtw-position"]
N_FIG = 200  # fixed length for cross-seed profile aggregation


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_span(group):
    phases = np.asarray(group["solver_phase"])
    peg = np.asarray(group["peg_pose"])
    indices = np.flatnonzero((phases >= 3) & (phases <= 6))
    peg_task = peg[indices]
    yaw = np.unwrap(pose_yaw(peg_task))
    return peg_task, yaw, indices


def physical_clear_global(group):
    """Global time index where the key physically clears the gate.

    key_clearance_margin = GATE_Z_MIN - (peg_z + KEY_Z_MAX) is negative while
    the key top is still above the gate bottom (not yet cleared) and becomes
    positive once the key has passed through, so the clear event is the first
    task-span index with a strictly positive margin.
    """
    clearance = np.asarray(group["key_clearance_margin"])
    phases = np.asarray(group["solver_phase"])
    task_idx = np.flatnonzero((phases >= 3) & (phases <= 6))
    for i in task_idx:
        if clearance[i] > 0.0:
            return int(i)
    return int(task_idx[-1])


def rot_angle_rad(q1, q2):
    r1 = quat2mat(q1)
    r2 = quat2mat(q2)
    rel = r1.T @ r2
    cosine = (np.trace(rel) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def se3_step_length(peg_task):
    length = np.zeros(len(peg_task), dtype=np.float64)
    for i in range(1, len(peg_task)):
        dp = peg_task[i, :3] - peg_task[i - 1, :3]
        angle = rot_angle_rad(peg_task[i - 1, 3:7], peg_task[i, 3:7])
        length[i] = np.sqrt(np.dot(dp, dp) + (ROTATION_SCALE * angle) ** 2)
    return length


def dtw_path(a, b):
    """Monotonic alignment path: for each a-index, the aligned b-index."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = a.shape[0], b.shape[0]
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        costs = np.sum((b - a[i - 1]) ** 2, axis=1)
        row = np.full(m + 1, np.inf)
        row[0] = np.inf
        for j in range(1, m + 1):
            row[j] = costs[j - 1] + min(d[i - 1, j], row[j - 1], d[i - 1, j - 1])
        d[i] = row
    i, j = n, m
    map_to_b = np.zeros(n, dtype=np.float64)
    while i > 0 and j > 0:
        map_to_b[i - 1] = j - 1
        step = min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
        if step == d[i - 1, j - 1]:
            i -= 1
            j -= 1
        elif step == d[i - 1, j]:
            i -= 1
        else:
            j -= 1
    while i > 0:
        map_to_b[i - 1] = 0
        i -= 1
    return map_to_b


def progress_signal(group, param, refs):
    """Monotonic progress s(t) over the task span (length Tt)."""
    peg_task, yaw, _ = task_span(group)
    n = len(peg_task)
    x, y, z = peg_task[:, 0], peg_task[:, 1], peg_task[:, 2]
    if param == "oracle":
        phases = np.asarray(group["solver_phase"])
        task_idx = np.flatnonzero((phases >= 3) & (phases <= 6))
        s = np.zeros(n, dtype=np.float64)
        for phase in (3, 4, 5, 6):
            in_phase = np.flatnonzero(phases[task_idx] == phase)
            if len(in_phase) > 0:
                s[in_phase] = (phase - 3 + np.linspace(0.0, 1.0, len(in_phase))) / 4.0
        return s
    if param == "time":
        s = np.arange(n, dtype=np.float64) / refs["T_ref"]
    elif param == "arc":
        s = np.cumsum(se3_step_length(peg_task)) / refs["L_ref"]
    elif param == "keypoint":
        s = (refs["d_start"] - (z - refs["socket_z"])) / (
            refs["d_start"] - refs["d_final"]
        )
    elif param == "dtw":
        a = np.column_stack([x, y, ROTATION_SCALE * yaw])
        map_to_b = dtw_path(a, refs["dtw_reference"])
        s = map_to_b / max(refs["T_ref_base"] - 1.0, 1.0)
    elif param == "dtw-position":
        a = np.column_stack([x, y, z])
        map_to_b = dtw_path(a, refs["dtw_position_reference"])
        s = map_to_b / max(refs["T_ref_base"] - 1.0, 1.0)
    else:
        raise ValueError(param)
    return np.maximum.accumulate(np.asarray(s, dtype=np.float64))


def resample_curve(x, y, yaw, s, n_grid):
    s = np.maximum.accumulate(np.asarray(s, dtype=np.float64))
    grid = np.linspace(0.0, 1.0, n_grid)
    xs = np.interp(grid, s, x, left=x[0], right=x[-1])
    ys = np.interp(grid, s, y, left=y[0], right=y[-1])
    yaw_s = np.interp(grid, s, yaw, left=yaw[0], right=yaw[-1])
    return np.column_stack([xs, ys, yaw_s])


def curve_for(group, param, bins, refs):
    n_grid = 4 * bins
    if param == "oracle":
        return task_curve(group, bins)
    peg_task, yaw, _ = task_span(group)
    s = progress_signal(group, param, refs)
    return resample_curve(peg_task[:, 0], peg_task[:, 1], yaw, s, n_grid)


def empirical_profile(baseline_curve, isolated_curves, isolated_contexts, isolated_generators):
    diagonal = np.empty((baseline_curve.shape[0], 3), dtype=np.float64)
    for generator in range(3):
        if generator == 0:
            mask = [abs(c[0]) > 0 for c in isolated_contexts]
        elif generator == 1:
            mask = [abs(c[1]) > 0 for c in isolated_contexts]
        else:
            mask = [abs(c[2]) > 0 for c in isolated_contexts]
        contexts = [0.0]
        values = [baseline_curve[:, generator]]
        for include, context, curve in zip(mask, isolated_contexts, isolated_curves):
            if include:
                contexts.append(float(context[generator]))
                values.append(curve[:, generator])
        design = np.column_stack([np.ones(len(contexts)), np.asarray(contexts)])
        diagonal[:, generator] = np.linalg.lstsq(design, np.asarray(values), rcond=None)[0][1]
    return diagonal


def find_switch(progress, yaw_profile):
    for i in range(1, len(progress)):
        y0, y1 = yaw_profile[i - 1], yaw_profile[i]
        if y0 >= 0.5 and y1 < 0.5:
            if abs(y1 - y0) < 1e-12:
                return float(progress[i])
            frac = (0.5 - y0) / (y1 - y0)
            return float(progress[i - 1] + frac * (progress[i] - progress[i - 1]))
    return float("nan")


def predict_at(model, context, s_values, progress):
    """Model prediction at arbitrary progress values, interpolated from its grid."""
    grid_pred = model.predict(np.asarray([context], dtype=np.float64))[0]  # [100, 3]
    progress = np.asarray(progress, dtype=np.float64)
    out = np.column_stack(
        [np.interp(s_values, progress, grid_pred[:, c]) for c in range(3)]
    )
    return out


def physical_task_error(model, group, param, refs, context, progress):
    peg_task, yaw, global_idx = task_span(group)
    s = progress_signal(group, param, refs)
    t_clear = physical_clear_global(group)
    yhat = predict_at(model, context, s, progress)
    x_actual = np.column_stack([peg_task[:, 0], peg_task[:, 1], yaw])
    resid = yhat - x_actual
    resid[:, 2] = np.asarray([wrap_pi(v) for v in resid[:, 2]])
    resid *= np.array([1.0, 1.0, ROTATION_SCALE])
    # quotient axial yaw only AFTER the physical key-clear event
    resid[global_idx >= t_clear, 2] = 0.0
    return float(np.linalg.norm(resid, axis=1).mean() * 1000.0)


def nominal_mapping(group, param, refs, alpha_yaw, progress):
    """Map a learned alpha_yaw profile onto the nominal physical timeline.

    Returns (s_nom [Tt], alpha_on_time [Tt], z_nom [Tt], t_clear_global,
    task_global_indices [Tt]).
    """
    peg_task, _, global_idx = task_span(group)
    s_nom = progress_signal(group, param, refs)
    alpha_on_time = np.interp(s_nom, progress, alpha_yaw)
    return s_nom, alpha_on_time, peg_task[:, 2], physical_clear_global(group), global_idx


def analyze_seed(data_file, mixed_keys, isolated_keys, baseline_key, config, bins):
    progress, _ = progress_grid(bins)
    n_grid = len(progress)

    mixed_contexts = np.asarray(
        [np.asarray(data_file[key]["causal_delta"]) for key in mixed_keys]
    )
    isolated_contexts = np.asarray(
        [np.asarray(data_file[key]["causal_delta"]) for key in isolated_keys]
    )
    isolated_generators = [
        str(data_file[key].attrs["generator"]) for key in isolated_keys
    ]
    baseline_group = data_file[baseline_key]

    # Reference quantities from mixed TRAINING episodes only.
    task_lengths = []
    arc_totals = []
    d_starts = []
    d_finals = []
    socket_zs = []
    for key in mixed_keys:
        peg_task, _, _ = task_span(data_file[key])
        task_lengths.append(len(peg_task))
        arc_totals.append(float(np.sum(se3_step_length(peg_task))))
        socket = np.asarray(data_file[key]["socket_pose"])[0, 2]
        d_starts.append(float(peg_task[0, 2] - socket))
        d_finals.append(float(peg_task[-1, 2] - socket))
        socket_zs.append(float(socket))
    T_ref = float(np.mean(task_lengths))
    L_ref = float(np.mean(arc_totals))
    d_start = float(np.mean(d_starts))
    d_final = float(np.mean(d_finals))
    socket_z = float(np.mean(socket_zs))

    ref_peg, ref_yaw, _ = task_span(baseline_group)
    dtw_reference = np.column_stack(
        [ref_peg[:, 0], ref_peg[:, 1], ROTATION_SCALE * ref_yaw]
    )
    dtw_position_reference = np.column_stack([ref_peg[:, 0], ref_peg[:, 1], ref_peg[:, 2]])
    refs = {
        "T_ref": T_ref,
        "L_ref": L_ref,
        "d_start": d_start,
        "d_final": d_final,
        "socket_z": socket_z,
        "dtw_reference": dtw_reference,
        "dtw_position_reference": dtw_position_reference,
        "T_ref_base": len(ref_peg),
    }

    # Clipping / horizon-mismatch diagnostics (per parameterization).
    clip_rows = []
    for param in ["time", "arc", "keypoint"]:
        ends = []
        for key in mixed_keys:
            s = progress_signal(data_file[key], param, refs)
            ends.append(float(s[-1]))
        ends = np.asarray(ends)
        clip_rows.append(
            {
                "param": param,
                "s_end_mean": float(ends.mean()),
                "s_end_min": float(ends.min()),
                "s_end_max": float(ends.max()),
                "frac_s_end_gt_1": float((ends > 1.0).mean()),
                "frac_s_end_lt_1": float((ends < 1.0).mean()),
            }
        )

    results = {}
    for param in PARAM_ORDER:
        mixed_curves = np.asarray(
            [curve_for(data_file[key], param, bins, refs) for key in mixed_keys]
        )
        isolated_curves = np.asarray(
            [curve_for(data_file[key], param, bins, refs) for key in isolated_keys]
        )
        baseline_curve = curve_for(baseline_group, param, bins, refs)
        target_profile = empirical_profile(
            baseline_curve, isolated_curves, isolated_contexts, isolated_generators
        )

        nominal_xy = []
        nominal_yaw = []
        for key, context in zip(mixed_keys, mixed_contexts):
            socket_pose = np.asarray(data_file[key]["socket_pose"])[0]
            nominal_xy.append(socket_pose[:2] - context[:2])
            nominal_yaw.append(float(pose_yaw(socket_pose[None])[0] - context[2]))
        nominal_xy = np.asarray(nominal_xy)
        nominal_yaw = np.asarray(nominal_yaw)
        frame_xy = nominal_xy.mean(axis=0)
        frame_yaw = float(
            np.arctan2(np.sin(nominal_yaw).mean(), np.cos(nominal_yaw).mean())
        )

        pdiag = SmoothFinitePDiagModel(
            nominal_frame_xy=frame_xy,
            nominal_frame_yaw=frame_yaw,
            **config,
        ).fit(mixed_contexts, mixed_curves, progress, np.zeros(n_grid, dtype=int))
        scalar = FrameWeightedModel().fit(
            mixed_contexts, mixed_curves, progress, np.zeros(n_grid, dtype=int)
        )

        # Physical-timeline held-out task error.
        pdiag_errors = [
            physical_task_error(pdiag, data_file[key], param, refs, context, progress)
            for key, context in zip(isolated_keys, isolated_contexts)
        ]
        scalar_errors = [
            physical_task_error(scalar, data_file[key], param, refs, context, progress)
            for key, context in zip(isolated_keys, isolated_contexts)
        ]

        alpha_yaw = pdiag.jacobian_diag()[:, 2]
        s_switch = find_switch(progress, alpha_yaw)

        # Physical switch offset via the nominal (baseline) mapping.
        s_nom, alpha_on_time, z_nom, t_clear_nom, nom_global = nominal_mapping(
            baseline_group, param, refs, alpha_yaw, progress
        )
        t_switch_nom = int(np.argmin(np.abs(s_nom - s_switch)))
        z_switch = z_nom[t_switch_nom]
        t_clear_local = int(np.flatnonzero(nom_global == t_clear_nom)[0])
        z_clear = z_nom[t_clear_local]
        dz_switch = (z_switch - z_clear) * 1000.0  # mm

        alpha_on_time_fixed = np.interp(
            np.linspace(0.0, 1.0, N_FIG),
            np.linspace(0.0, 1.0, len(alpha_on_time)),
            alpha_on_time,
        )
        results[param] = {
            "alpha_on_time": alpha_on_time,
            "alpha_on_time_fixed": alpha_on_time_fixed,
            "E_task_physical": float(np.mean(pdiag_errors)),
            "E_task_scalar_physical": float(np.mean(scalar_errors)),
            "delta_z_switch_mm": dz_switch,
            "s_switch": s_switch,
            "E_gen": float(np.sqrt(np.mean((pdiag.jacobian_diag() - target_profile) ** 2))),
            "g_translation_final": float(pdiag.jacobian_diag()[-1, :2].mean()),
            "g_yaw_final": float(alpha_yaw[-1]),
        }

    oracle = results["oracle"]
    rows = []
    for param in PARAM_ORDER:
        r = results[param]
        rows.append(
            {
                "param": param,
                "E_task_physical": r["E_task_physical"],
                "delta_E_task_physical": r["E_task_physical"] - oracle["E_task_physical"],
                "E_task_scalar_physical": r["E_task_scalar_physical"],
                "delta_z_switch_mm": r["delta_z_switch_mm"],
                "rho_alpha_event": (
                    float(np.corrcoef(r["alpha_on_time"], oracle["alpha_on_time"])[0, 1])
                    if param != "oracle"
                    else 1.0
                ),
                "rmse_alpha_event": float(
                    np.sqrt(np.mean((r["alpha_on_time"] - oracle["alpha_on_time"]) ** 2))
                ),
                "E_gen": r["E_gen"],
                "g_translation_final": r["g_translation_final"],
                "g_yaw_final": r["g_yaw_final"],
            }
        )
    return rows, {param: results[param]["alpha_on_time_fixed"] for param in PARAM_ORDER}, clip_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/experiment.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/nonoracle_progress"),
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--bins", type=int, default=25)
    args = parser.parse_args()

    with args.experiment.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    seeds = experiment["seeds"] if args.seeds is None else args.seeds
    for seed in seeds:
        if seed not in experiment["seeds"]:
            raise ValueError(f"seed {seed} is not preregistered")
    config = {
        "alpha_max": experiment["frozen_model"]["alpha_max"],
        "n_basis": experiment["frozen_model"]["n_basis"],
        "basis_width": experiment["frozen_model"]["basis_width"],
        "smoothness_weight": experiment["frozen_model"]["smoothness_weight"],
        "nominal_iterations": experiment["frozen_model"]["nominal_iterations"],
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_clip_rows = []
    profiles_by_seed = {}
    for seed in seeds:
        dataset = args.experiment.parent / "rollouts" / f"seed_{seed}.h5"
        with h5py.File(dataset, "r") as data_file:
            keys = episode_keys(data_file)
            usable_keys = [key for key in keys if usable(data_file[key])]
            mixed_keys = [
                key
                for key in usable_keys
                if str(data_file[key].attrs["generator"]) == "mixed"
            ]
            isolated_keys = [
                key
                for key in usable_keys
                if str(data_file[key].attrs["generator"]) in {"yaw", "translation"}
                and np.linalg.norm(np.asarray(data_file[key]["causal_delta"])) > 1e-12
            ]
            baseline_keys = [
                key
                for key in usable_keys
                if np.linalg.norm(np.asarray(data_file[key]["causal_delta"])) <= 1e-12
            ]
            if len(mixed_keys) != 30 or len(isolated_keys) != 8 or len(baseline_keys) != 1:
                raise RuntimeError(
                    f"seed {seed}: expected 30 mixed, 8 isolated, 1 baseline"
                )
            rows, profiles, clip_rows = analyze_seed(
                data_file, mixed_keys, isolated_keys, baseline_keys[0], config, args.bins
            )
        for row in rows:
            row["seed"] = seed
        for row in clip_rows:
            row["seed"] = seed
        all_rows.extend(rows)
        all_clip_rows.extend(clip_rows)
        profiles_by_seed[seed] = profiles
        arc_row = next(r for r in rows if r["param"] == "arc")
        print(f"seed {seed}: arc dE_task={arc_row['delta_E_task_physical']:.2f}mm "
              f"dz_switch={arc_row['delta_z_switch_mm']:.2f}mm")

    frame = pd.DataFrame(all_rows)
    frame.to_csv(args.output_root / "nonoracle_progress.csv", index=False)
    clip_frame = pd.DataFrame(all_clip_rows)
    clip_frame.to_csv(args.output_root / "nonoracle_progress_clipping.csv", index=False)

    np.savez_compressed(
        args.output_root / "nonoracle_profiles.npz",
        progress=np.arange(len(next(iter(profiles_by_seed.values()))["oracle"])),
        **{
            f"seed_{seed}_param_{param}": profiles_by_seed[seed][param]
            for seed in seeds
            for param in PARAM_ORDER
        },
    )

    aggregate = (
        frame.groupby("param")
        .agg(
            E_task_physical_mean=("E_task_physical", "mean"),
            E_task_physical_sd=("E_task_physical", "std"),
            delta_E_task_physical_mean=("delta_E_task_physical", "mean"),
            delta_z_switch_mean=("delta_z_switch_mm", "mean"),
            rho_alpha_event_mean=("rho_alpha_event", "mean"),
            rmse_alpha_event_mean=("rmse_alpha_event", "mean"),
            E_gen_mean=("E_gen", "mean"),
            seed_count=("seed", "count"),
        )
        .reset_index()
    )
    aggregate.to_csv(args.output_root / "nonoracle_progress_aggregate.csv", index=False)

    summary = {
        "schema_version": 2,
        "experiment_sha256": sha256(args.experiment),
        "source_sha256": sha256(Path(__file__)),
        "seeds": seeds,
        "parameterizations": PARAM_ORDER,
        "evaluation": "physical timeline; yaw quotient bound to physical key-clear event",
        "dtw": {
            "dtw": "DTW on [x, y, ell*yaw] -- optimistic offline upper bound (yaw leak)",
            "dtw-position": "DTW on [x, y, z] -- no-yaw-leak variant",
        },
        "references_from": "mixed training episodes only",
        "dtw_reference": "zero-intervention baseline episode",
        "aggregate": aggregate.to_dict(orient="records"),
        "clipping": clip_frame.to_dict(orient="records"),
    }
    with (args.output_root / "nonoracle_progress_summary.json").open(
        "w", encoding="utf-8"
    ) as summary_file:
        json.dump(summary, summary_file, indent=2)

    # Figure.
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
    colors = {
        "oracle": "#000000",
        "time": "#999999",
        "arc": "#0072B2",
        "keypoint": "#009E73",
        "dtw": "#D55E00",
        "dtw-position": "#CC79A7",
    }
    styles = {
        "oracle": "-",
        "time": "--",
        "arc": "-",
        "keypoint": "-.",
        "dtw": ":",
        "dtw-position": (0, (5, 2)),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    for param in PARAM_ORDER:
        mean_profile = np.mean(
            [profiles_by_seed[seed][param] for seed in seeds], axis=0
        )
        axes[0].plot(
            np.arange(len(mean_profile)),
            mean_profile,
            color=colors[param],
            linestyle=styles[param],
            linewidth=2.0 if param in ("oracle", "arc") else 1.2,
            label=param,
        )
    axes[0].axhline(0.5, color="0.7", linewidth=0.7, linestyle=":")
    axes[0].set_xlabel("Nominal physical time (frames)")
    axes[0].set_ylabel(r"$\alpha_{yaw}$")
    axes[0].set_title(
        "A  Event-aligned yaw profile", loc="left", fontweight="bold"
    )
    axes[0].legend(frameon=False, fontsize=6.5)

    agg = aggregate.set_index("param").loc[
        ["time", "arc", "keypoint", "dtw", "dtw-position"]
    ]
    axes[1].bar(
        np.arange(len(agg)),
        agg.delta_E_task_physical_mean,
        color=[colors[p] for p in agg.index],
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].set_xticks(np.arange(len(agg)), agg.index.tolist())
    axes[1].set_xlabel("Non-oracle progress")
    axes[1].set_ylabel(r"$\Delta E_{task}^{physical}$ (mm)")
    axes[1].set_title(
        "B  Physical OOD task-error cost vs oracle", loc="left", fontweight="bold"
    )
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.savefig(args.output_root / "nonoracle_progress_figure.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_root / "nonoracle_progress_figure.pdf", bbox_inches="tight")
    plt.close(fig)

    print(aggregate.to_string(index=False))
    print("saved:", args.output_root / "nonoracle_progress.csv")
    print("saved:", args.output_root / "nonoracle_progress_clipping.csv")
    print("saved:", args.output_root / "nonoracle_progress_aggregate.csv")


if __name__ == "__main__":
    main()
