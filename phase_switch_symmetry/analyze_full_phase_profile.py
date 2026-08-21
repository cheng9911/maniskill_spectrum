from __future__ import annotations

"""Full-phase-resolution generator-selectivity re-analysis (Step 1).

The fitted model (benchmark_phase_switch_baselines.py) only sees the four
insertion phases PHASE_CODES=(3,4,5,6) = align_keyed, enter_key, unlock_yaw,
circular_insert. This script asks the question that falls outside the model's
curve: over ALL seven solver phases (reach 0, grasp 1, lift 2, align_keyed 3,
enter_key 4, unlock_yaw 5, circular_insert 6), is the axial-yaw generator really
0 -> 1 -> 0, and are the x/y generators really constant?

The answer is read off the held-out isolated interventions directly (the
empirical response diagonal), not a fitted model:

    alpha_x   (s) = (x_obs(s)  - x_ref(s))  / d_x      (x-translation episodes)
    alpha_y   (s) = (y_obs(s)  - y_ref(s))  / d_y      (y-translation episodes)
    alpha_yaw (s) = wrap_pi(yaw_obs(s) - yaw_ref(s)) / d_axial   (yaw episodes)

where obs is a held-out isolated episode, ref is the zero-intervention baseline,
and d_* is that episode's task-local intervention. The response is evaluated at
each phase ENDPOINT (the last state the solver labels with that phase code).

Expected from the solver geometry (phase_switch_symmetry_env.py): the peg is
picked up at a FIXED pedestal pose independent of the socket intervention, so
reach/grasp/lift (0,1,2) are pre-contact and nothing responds. align_keyed (3)
and enter_key (4) move the peg to the socket's shifted centre at the socket's
keyed yaw, so every generator responds. unlock_yaw (5) and circular_insert (6)
rotate the peg to a fixed yaw 0 but keep the shifted centre, so translation
responds and yaw does not:

    alpha_x, alpha_y : 0 0 0 1 1 1 1     (0 -> 1 at align_keyed)
    alpha_yaw        : 0 0 0 1 1 0 0     (0 -> 1 -> 0)

i.e. yaw is the ONLY selective generator (0 -> 1 -> 0); translation turns on at
align_keyed and stays on. Both are trivially 0 during reach/grasp/lift.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transforms3d.euler import quat2euler


PHASE_LABELS = (
    "Reach",
    "Grasp",
    "Lift",
    "Align keyed",
    "Enter key",
    "Unlock yaw",
    "Circular insert",
)
FULL_PHASES = tuple(range(7))


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


def complete_full(group):
    phases = set(int(value) for value in np.asarray(group["solver_phase"]))
    return all(phase in phases for phase in FULL_PHASES)


def usable(group):
    return (
        bool(np.asarray(group["success"])[-1])
        and complete_full(group)
        and not bool(np.asarray(group["truncated"]).any())
    )


def resample(values, bins):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, bins)
    return np.column_stack(
        [np.interp(target, source, values[:, c]) for c in range(values.shape[1])]
    )


def phase_task_curve(group, phase, bins):
    phases = np.asarray(group["solver_phase"])
    indices = np.flatnonzero(phases == phase)
    peg = np.asarray(group["peg_pose"])[indices]
    yaw = np.unwrap(pose_yaw(peg))
    return resample(np.column_stack([peg[:, 0], peg[:, 1], yaw]), bins)


def endpoint_pose(group, phase):
    phases = np.asarray(group["solver_phase"])
    indices = np.flatnonzero(phases == phase)
    if len(indices) == 0:
        return None
    pose = np.asarray(group["peg_pose"])[indices[-1]]
    return np.array([pose[0], pose[1], quat2euler(pose[3:])[2]])


def fit_scalar_slope(context_values, output_values):
    context_values = np.asarray(context_values, dtype=np.float64)
    design = np.column_stack([np.ones(len(context_values)), context_values])
    return np.linalg.lstsq(design, output_values, rcond=None)[0][1]


def analyze_seed(path: Path, bins: int):
    """Return (endpoint_rows, profile_diagonal, profile_progress)."""
    with h5py.File(path, "r") as data_file:
        keys = episode_keys(data_file)
        usable_keys = [key for key in keys if usable(data_file[key])]
        baseline_candidates = [
            key
            for key in usable_keys
            if np.linalg.norm(np.asarray(data_file[key]["causal_delta"])) < 1e-10
        ]
        if len(baseline_candidates) != 1:
            raise RuntimeError(
                f"{path}: expected 1 baseline, got {len(baseline_candidates)}"
            )
        baseline_key = baseline_candidates[0]
        baseline = data_file[baseline_key]

        yaw_keys = [
            key
            for key in usable_keys
            if str(data_file[key].attrs.get("generator", "")) == "yaw"
            and abs(float(np.asarray(data_file[key]["causal_delta"])[2])) > 1e-12
        ]
        x_keys = [
            key
            for key in usable_keys
            if str(data_file[key].attrs.get("generator", "")) == "translation"
            and abs(float(np.asarray(data_file[key]["causal_delta"])[0])) > 1e-12
        ]
        y_keys = [
            key
            for key in usable_keys
            if str(data_file[key].attrs.get("generator", "")) == "translation"
            and abs(float(np.asarray(data_file[key]["causal_delta"])[1])) > 1e-12
        ]
        if len(yaw_keys) != 4 or len(x_keys) != 2 or len(y_keys) != 2:
            raise RuntimeError(
                f"{path}: isolated split off (yaw={len(yaw_keys)}, "
                f"x={len(x_keys)}, y={len(y_keys)})"
            )

        endpoint_rows = []
        for phase in FULL_PHASES:
            ref = endpoint_pose(baseline, phase)
            if ref is None:
                continue
            for key in yaw_keys:
                group = data_file[key]
                obs = endpoint_pose(group, phase)
                d = float(np.asarray(group["causal_delta"])[2])
                endpoint_rows.append(
                    dict(
                        phase=phase,
                        phase_label=PHASE_LABELS[phase],
                        generator="yaw",
                        episode=key,
                        alpha_x=np.nan,
                        alpha_y=np.nan,
                        alpha_yaw=float(wrap_pi(obs[2] - ref[2]) / d),
                    )
                )
            for key in x_keys:
                group = data_file[key]
                obs = endpoint_pose(group, phase)
                d = float(np.asarray(group["causal_delta"])[0])
                endpoint_rows.append(
                    dict(
                        phase=phase,
                        phase_label=PHASE_LABELS[phase],
                        generator="x",
                        episode=key,
                        alpha_x=float((obs[0] - ref[0]) / d),
                        alpha_y=np.nan,
                        alpha_yaw=np.nan,
                    )
                )
            for key in y_keys:
                group = data_file[key]
                obs = endpoint_pose(group, phase)
                d = float(np.asarray(group["causal_delta"])[1])
                endpoint_rows.append(
                    dict(
                        phase=phase,
                        phase_label=PHASE_LABELS[phase],
                        generator="y",
                        episode=key,
                        alpha_x=np.nan,
                        alpha_y=float((obs[1] - ref[1]) / d),
                        alpha_yaw=np.nan,
                    )
                )

        # Continuous empirical isolated profile over the full 7-phase curve.
        baseline_curve = np.concatenate(
            [phase_task_curve(baseline, phase, bins) for phase in FULL_PHASES],
            axis=0,
        )
        diagonal = np.empty((len(baseline_curve), 3), dtype=np.float64)
        for generator in range(3):
            if generator == 0:
                keys = x_keys
                component = 0
            elif generator == 1:
                keys = y_keys
                component = 1
            else:
                keys = yaw_keys
                component = 2
            contexts = [0.0]
            values = [baseline_curve[:, component]]
            for key in keys:
                contexts.append(
                    float(np.asarray(data_file[key]["causal_delta"])[component])
                )
                curve = np.concatenate(
                    [
                        phase_task_curve(data_file[key], phase, bins)
                        for phase in FULL_PHASES
                    ],
                    axis=0,
                )
                values.append(curve[:, component])
            diagonal[:, generator] = fit_scalar_slope(
                contexts, np.asarray(values)
            )
        progress = np.concatenate(
            [
                (phase_index + np.linspace(0.0, 1.0, bins)) / len(FULL_PHASES)
                for phase_index in range(len(FULL_PHASES))
            ]
        )
        return endpoint_rows, diagonal, progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[20260818, 20270818, 20280818],
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=None,
        help="Override the keyed rotated_Q1 files (one per seed).",
    )
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/full_phase_reanalysis"),
    )
    args = parser.parse_args()

    files = args.files or [
        Path(f"phase_switch_symmetry_rollouts_rotated/rotated_Q1_seed_{seed}.h5")
        for seed in args.seeds
    ]
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    all_endpoint = []
    profiles = []
    for path in files:
        endpoint_rows, diagonal, progress = analyze_seed(path, args.bins)
        seed = path.name.split("seed_")[-1].split(".h5")[0]
        for row in endpoint_rows:
            row["seed"] = int(seed)
        all_endpoint.extend(endpoint_rows)
        profiles.append(diagonal)

    endpoint = pd.DataFrame(all_endpoint)
    endpoint.to_csv(output_root / "full_phase_endpoint_response.csv", index=False)

    # Per-phase summary: mean/std over the 12 held-out episodes (4 yaw, 2 x, 2 y
    # per seed) pooled across 3 seeds.
    summary_rows = []
    for phase in FULL_PHASES:
        sub = endpoint[endpoint.phase == phase]
        summary_rows.append(
            dict(
                phase=phase,
                phase_label=PHASE_LABELS[phase],
                alpha_x_mean=float(sub.alpha_x.dropna().mean()),
                alpha_x_std=float(sub.alpha_x.dropna().std(ddof=0)),
                alpha_y_mean=float(sub.alpha_y.dropna().mean()),
                alpha_y_std=float(sub.alpha_y.dropna().std(ddof=0)),
                alpha_yaw_mean=float(sub.alpha_yaw.dropna().mean()),
                alpha_yaw_std=float(sub.alpha_yaw.dropna().std(ddof=0)),
                n_x=int(sub.alpha_x.notna().sum()),
                n_y=int(sub.alpha_y.notna().sum()),
                n_yaw=int(sub.alpha_yaw.notna().sum()),
            )
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "full_phase_endpoint_summary.csv", index=False)

    profile_mean = np.mean(np.asarray(profiles), axis=0)
    profile_std = np.std(np.asarray(profiles), axis=0, ddof=0)
    profile_df = pd.DataFrame(
        dict(
            global_progress=progress,
            alpha_x=profile_mean[:, 0],
            alpha_y=profile_mean[:, 1],
            alpha_yaw=profile_mean[:, 2],
            alpha_x_std=profile_std[:, 0],
            alpha_y_std=profile_std[:, 1],
            alpha_yaw_std=profile_std[:, 2],
        )
    )
    profile_df.to_csv(output_root / "full_phase_empirical_profile.csv", index=False)

    # --- figure -------------------------------------------------------------
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
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), constrained_layout=True)

    ax = axes[0]
    names = ["x", "y", "yaw"]
    for name in names:
        ax.plot(
            progress,
            profile_mean[:, names.index(name)],
            color=colors[name],
            label=fr"$\alpha_{{{name}}}$",
            linewidth=1.8,
        )
        ax.fill_between(
            progress,
            profile_mean[:, names.index(name)] - profile_std[:, names.index(name)],
            profile_mean[:, names.index(name)] + profile_std[:, names.index(name)],
            color=colors[name],
            alpha=0.16,
            linewidth=0,
        )
    for boundary in range(1, len(FULL_PHASES)):
        ax.axvline(boundary / len(FULL_PHASES), color="0.75", linewidth=0.8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.25, 1.35)
    ax.set_xticks(
        [(p + 0.5) / len(FULL_PHASES) for p in range(len(FULL_PHASES))],
        [label.replace(" ", "\n") for label in PHASE_LABELS],
    )
    ax.set_ylabel("Empirical generator response")
    ax.axhline(0.0, color="0.5", linewidth=0.7)
    ax.axhline(1.0, color="0.5", linewidth=0.7, linestyle="--")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title(
        "A  Full-phase empirical response diagonal (mean over 3 seeds)",
        loc="left",
        fontweight="bold",
    )

    ax = axes[1]
    x_pos = np.arange(len(FULL_PHASES))
    width = 0.26
    for i, (name, color) in enumerate(colors.items()):
        means = summary[f"alpha_{name}_mean"].to_numpy()
        stds = summary[f"alpha_{name}_std"].to_numpy()
        ax.bar(
            x_pos + (i - 1) * width,
            means,
            width,
            yerr=stds,
            color=color,
            capsize=2,
            label=fr"$\alpha_{{{name}}}$",
            error_kw={"linewidth": 0.7},
        )
    ax.set_xticks(x_pos, [label.replace(" ", "\n") for label in PHASE_LABELS])
    ax.set_ylabel("Phase-endpoint response")
    ax.set_ylim(-0.25, 1.35)
    ax.axhline(0.0, color="0.5", linewidth=0.7)
    ax.axhline(1.0, color="0.5", linewidth=0.7, linestyle="--")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title(
        "B  Phase-endpoint response (bars: mean over 12 held-out interventions)",
        loc="left",
        fontweight="bold",
    )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(output_root / "full_phase_profile.pdf", bbox_inches="tight")
    fig.savefig(output_root / "full_phase_profile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- verdict ------------------------------------------------------------
    yaw = summary.alpha_yaw_mean.to_numpy()
    trans = 0.5 * (summary.alpha_x_mean + summary.alpha_y_mean).to_numpy()
    verdict = {
        "yaw_profile": yaw.tolist(),
        "translation_profile": trans.tolist(),
        "yaw_is_0_1_0": bool(
            np.all(yaw[:3] < 0.15)
            and np.all(yaw[3:5] > 0.85)
            and np.all(yaw[5:] < 0.15)
        ),
        "translation_is_0_then_1": bool(
            np.all(trans[:3] < 0.15) and np.all(trans[3:] > 0.85)
        ),
        "translation_is_constant_1": bool(np.all(trans > 0.85)),
        "note": (
            "yaw is 0->1->0 (0 in reach/grasp/lift, 1 in align/enter, 0 in "
            "unlock/insert). translation is 0->1 (0 in reach/grasp/lift, 1 in "
            "align/enter/unlock/insert), NOT constant 1: reach/grasp/lift are "
            "pre-contact, the peg is at a fixed pedestal pose independent of the "
            "socket intervention."
        ),
    }
    with (output_root / "full_phase_verdict.json").open("w") as handle:
        json.dump(verdict, handle, indent=2)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("=== full-phase endpoint response (mean +- std over 12 held-out) ===")
    print(
        summary[
            [
                "phase",
                "phase_label",
                "alpha_x_mean",
                "alpha_y_mean",
                "alpha_yaw_mean",
                "alpha_yaw_std",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:+.3f}")
    )
    print("\nverdict:", json.dumps(verdict, indent=2))
    print("saved:", output_root)


if __name__ == "__main__":
    main()
