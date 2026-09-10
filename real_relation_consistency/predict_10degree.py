"""Apply the frozen sim law alpha_pitch(s) at held-out +-10 deg; magnitude-only.

The frozen law is P_sim(s) = alpha(s) (Pdiag-finite SE(3)), identified from a
pitch-amplitude sweep that EXCLUDES +-10 deg. Here we apply the frozen finite
action ``C0 Exp(diag(alpha(s)) Log(C0^-1 C)) C0^-1 X0(s)`` at a +-10 deg pitch
context and read off the terminal peg-axis response.

Honest scope:
- The prediction runs in the SIMULATION's own nominal geometry (the frozen
  ``nominal_frame_pose`` / ``nominal_curve`` recovered from the sim sweep). The
  real machine contributes ONLY its terminal hand-axis magnitude (10.30 deg,
  direction-agnostic); we do NOT reconstruct a real nominal trajectory or a real
  hole frame (the real tilt axis is uncalibrated and unknown). So this is a
  frozen-sim-law prediction compared to a real terminal magnitude, not a
  real-geometry trajectory prediction.
- +-10 deg lies INSIDE the [-15, 15] training range, so the held-out check is
  INTERPOLATION, not extrapolation. We quantify it against independently
  collected sim +-10 deg pitch trajectories (sim_10degree), which are genuine
  held-out data, rather than reading the model's own self-prediction curve.
- The terminal magnitude is near-degenerate (alpha_pitch -> 1 forces response
  ~= input amplitude); the non-degenerate content is (i) the held-out +-10 deg
  interpolation error and (ii) the progress-resolved alpha_pitch(s) transient,
  which currently has simulation evidence only (no real transient comparison).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase_switch_symmetry"))
from phase_switch_se3_baselines import (  # noqa: E402
    SE3SmoothFinitePDiagModel,
    matrix_from_euler_batched,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PITCH_INDEX = 4
HELDOUT_DEG = [10.0, -10.0]
REAL_DATASETS = ("insert_jc_fix_3", "insert_10degree")


def mean_axis(axes):
    v = np.asarray(axes, dtype=np.float64).mean(axis=0)
    return v / np.linalg.norm(v)


def angle(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconstruct_model(law, pdiag_config):
    model = SE3SmoothFinitePDiagModel(
        nominal_frame_pose=law["nominal_frame_pose"], **pdiag_config
    )
    model.nominal_curve = law["nominal_curve"]
    model.diagonal = law["diagonal"]
    return model


def terminal_axis(model, context, bins):
    """Terminal peg z-axis for a context: mean z-axis over the last phase's bins."""
    pred = model.predict(np.asarray([context], dtype=np.float64))  # (1, n_steps, 6)
    pose6 = pred[0]
    rotation = matrix_from_euler_batched(pose6[..., 3:])  # (n_steps, 3, 3)
    z_axes = rotation[..., :, 2]  # (n_steps, 3)
    terminal = z_axes[-bins:]  # last phase (circular insert)
    return mean_axis(terminal)


def predicted_response(model, theta_deg, bins):
    base_axis = terminal_axis(model, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], bins)
    tilt_axis = terminal_axis(model, [0.0, 0.0, 0.0, 0.0, np.deg2rad(theta_deg), 0.0], bins)
    return float(angle(base_axis, tilt_axis))


def real_response(real, n=10000, seed=20260911):
    rng = np.random.default_rng(seed)
    means = []
    boots = []
    for name in REAL_DATASETS:
        g = real[real.dataset == name][["axis_x", "axis_y", "axis_z"]].to_numpy(dtype=np.float64)
        assert np.isfinite(g).all() and len(g) > 0, f"missing real axes for {name}"
        means.append(mean_axis(g))
        a = g[rng.integers(0, len(g), (n, len(g)))].mean(1)
        boots.append(a / np.linalg.norm(a, axis=1, keepdims=True))
    point = float(angle(means[0], means[1]))
    boot = angle(boots[0], boots[1])
    return point, boot


def load_heldout_actual(sim_dir, seeds):
    """Actual terminal peg-axis response at +-10 deg pitch, read from the
    independently collected sim_10degree trajectories (a separate manifest from
    the identification sweep). Returns {sign: {seed: response_deg}}."""
    heldout = {"plus": {}, "minus": {}}
    for file in sorted(Path(sim_dir).glob("circular_seed_*.h5")):
        seed = int(file.stem.split("_")[-1])
        if seed not in seeds:
            continue
        with h5py.File(file, "r") as handle:
            axes = {}
            for key in handle:
                group = handle[key]
                generator = str(group.attrs["generator"])
                if generator not in ("baseline", "pitch"):
                    continue
                causal = np.asarray(group["causal_delta"])
                if generator == "baseline":
                    sign = "baseline"
                else:
                    sign = "plus" if causal[PITCH_INDEX] > 0 else "minus"
                indices = np.flatnonzero(np.asarray(group["solver_phase"]) == 6)
                if not len(indices):
                    continue
                peg_pose = np.asarray(group["peg_pose"])[indices]
                z_axes = Rotation.from_quat(peg_pose[:, [4, 5, 6, 3]]).as_matrix()[:, :, 2]
                axes[sign] = mean_axis(z_axes)
            if "baseline" in axes:
                for sign in ("plus", "minus"):
                    if sign in axes:
                        heldout[sign][seed] = float(angle(axes["baseline"], axes[sign]))
    return heldout


def plot(alpha_profiles, progress, bins, response_curve, heldout_pred, heldout_actual,
         real_point, real_boot, out):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axs = plt.subplots(1, 2, figsize=(7.086614, 3.1), layout="constrained")
    phase_boundaries = [bins, 2 * bins, 3 * bins]
    phase_centers = [int(bins * 0.5), int(bins * 1.5), int(bins * 2.5), int(bins * 3.5)]
    seed_colors = ["#9ECAE1", "#6BAED6", "#3182BD"]

    # (a) frozen relation law alpha_pitch(s)
    for (seed, alpha), color in zip(alpha_profiles.items(), seed_colors):
        axs[0].plot(progress, alpha, color=color, lw=1.0, label=f"seed {seed}")
    mean_alpha = np.mean(np.asarray(list(alpha_profiles.values())), axis=0)
    axs[0].plot(progress, mean_alpha, color="#08306B", lw=1.8, label="mean")
    for b in phase_boundaries:
        axs[0].axvline(progress[b], color="0.85", lw=0.7)
    axs[0].axhline(1.0, color="0.5", ls="--", lw=0.8)
    axs[0].set(
        xlim=(0, 1),
        ylim=(-0.15, 1.25),
        xticks=[progress[c] for c in phase_centers],
        xticklabels=["Align", "Enter", "Unlock", "Insert"],
        title="a  Frozen pitch response law",
    )
    # Larger font keeps the mathtext subscript >= 6 pt (default 7 pt -> 4.9 pt).
    axs[0].set_ylabel(r"$\alpha_\mathrm{pitch}(s)$", fontsize=9)
    axs[0].legend(frameon=False, fontsize=6, loc="lower right")

    # (b) cross-amplitude terminal response; held-out +-10 deg vs actual + real.
    amplitudes = response_curve["amplitudes_deg"]
    values = response_curve["predicted_response_deg"]
    axs[1].plot(amplitudes, values, color="#3182BD", lw=1.4, marker="o", ms=3,
                label="frozen law")
    # Identity baseline is |theta|: the metric is a non-negative angle.
    axs[1].plot([-15, 0, 15], [15, 0, 15], color="0.7", ls=":", lw=1.0,
                label="identity (α≡1): |θ|")
    # Held-out predictions at +-10 deg, each computed independently.
    axs[1].scatter([HELDOUT_DEG[1], HELDOUT_DEG[0]],
                   [heldout_pred["minus"], heldout_pred["plus"]],
                   color="#C07747", s=26, zorder=3, marker="D",
                   label="held-out ±10° prediction")
    # Held-out actuals: independently collected sim_10degree pitch trajectories.
    actual_x, actual_y = [], []
    for sign, x in (("minus", HELDOUT_DEG[1]), ("plus", HELDOUT_DEG[0])):
        for value in heldout_actual[sign].values():
            actual_x.append(x)
            actual_y.append(value)
    axs[1].scatter(actual_x, actual_y, color="#1D4055", s=20, zorder=3,
                   facecolor="none", edgecolor="#1D4055", lw=1.0,
                   label="held-out actual (sim)")
    lo, hi = np.quantile(real_boot, [0.025, 0.975])
    axs[1].axhspan(lo, hi, color="#C07747", alpha=0.15, zorder=0)
    axs[1].axhline(real_point, color="#C07747", lw=1.2, label="real hand-axis 10.30°")
    axs[1].set(
        xlabel="Input pitch amplitude (°)",
        ylabel="Terminal axis response (°)",
        title="b  Held-out ±10° interpolation",
    )
    axs[1].legend(frameon=False, fontsize=6, loc="upper left")
    for ax in axs:
        ax.tick_params(axis="x", labelsize=6)
    fig.savefig(out / "prediction_10degree.png", dpi=600)
    fig.savefig(out / "prediction_10degree.pdf")
    fig.savefig(out / "prediction_10degree.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--law-dir", type=Path, default=Path("real_relation_consistency/tilt_sweep"))
    parser.add_argument("--sim-dir", type=Path, default=Path("real_relation_consistency/sim_10degree"))
    parser.add_argument("--real-csv", type=Path,
                        default=Path("real_relation_consistency/fk_results/terminal_pose_per_episode.csv"))
    parser.add_argument("--out", type=Path, default=Path("real_relation_consistency/tilt_sweep"))
    args = parser.parse_args()

    npz = np.load(args.law_dir / "frozen_law.npz")
    summary = json.loads((args.law_dir / "frozen_law.json").read_text(encoding="utf-8"))
    pdiag_config = summary["pdiag_config"]
    bins = summary["bins_per_phase"]
    seeds = summary["seeds"]
    progress = npz["progress"]

    laws = {}
    for seed in seeds:
        laws[seed] = {
            "nominal_frame_pose": npz[f"seed_{seed}_nominal_frame_pose"],
            "nominal_curve": npz[f"seed_{seed}_nominal_curve"],
            "diagonal": npz[f"seed_{seed}_diagonal"],
        }

    # Terminal response prediction at the held-out +10 deg (mean over seeds).
    predictions = {}
    for seed, law in laws.items():
        model = reconstruct_model(law, pdiag_config)
        predictions[seed] = predicted_response(model, HELDOUT_DEG[0], bins)
    predicted_values = np.asarray(list(predictions.values()))

    # Cross-amplitude response curve (mean over seeds), including both signs.
    amplitude_grid = [-15.0, -12.5, -10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5,
                      10.0, 12.5, 15.0]
    curve_values = []
    for theta in amplitude_grid:
        vals = []
        for seed, law in laws.items():
            model = reconstruct_model(law, pdiag_config)
            vals.append(predicted_response(model, theta, bins))
        curve_values.append(float(np.mean(vals)))
    holdout_plus_pred = curve_values[amplitude_grid.index(HELDOUT_DEG[0])]
    holdout_minus_pred = curve_values[amplitude_grid.index(HELDOUT_DEG[1])]
    heldout_pred = {"plus": holdout_plus_pred, "minus": holdout_minus_pred}

    # Held-out validation against independently collected sim +-10 deg trajectories.
    heldout_actual = load_heldout_actual(args.sim_dir, seeds)
    heldout_error = {
        sign: heldout_pred[sign] - float(np.mean(list(heldout_actual[sign].values())))
        for sign in ("plus", "minus")
    }

    # Alpha_pitch profiles for the plot.
    alpha_profiles = {seed: npz[f"seed_{seed}_diagonal"][:, PITCH_INDEX] for seed in seeds}

    # Real machine terminal axis response.
    real = pd.read_csv(args.real_csv)
    real_point, real_boot = real_response(real)
    lo, hi = np.quantile(real_boot, [0.025, 0.975])

    # Baselines (analytic, no model): alpha=0 -> no response; alpha=1 -> |amplitude|.
    baseline_zero = 0.0
    baseline_identity = HELDOUT_DEG[0]

    table = pd.DataFrame(
        [
            dict(
                model="frozen law α_pitch(s)",
                predicted_terminal_response_deg=float(np.mean(predicted_values)),
                sd_over_seeds_deg=float(np.std(predicted_values, ddof=1)),
                per_seed=predictions,
            ),
            dict(
                model="baseline α≡0 (no transfer)",
                predicted_terminal_response_deg=baseline_zero,
                sd_over_seeds_deg=None,
                per_seed={},
            ),
            dict(
                model="baseline α≡1 (instant identity)",
                predicted_terminal_response_deg=baseline_identity,
                sd_over_seeds_deg=None,
                per_seed={},
            ),
            dict(
                model="real machine (hand axis)",
                predicted_terminal_response_deg=real_point,
                sd_over_seeds_deg=None,
                per_seed={},
            ),
        ]
    )
    table.to_csv(args.out / "prediction_10degree.csv", index=False)

    heldout_rows = []
    for sign, theta in (("plus", HELDOUT_DEG[0]), ("minus", HELDOUT_DEG[1])):
        for seed in seeds:
            pred = predicted_response(reconstruct_model(laws[seed], pdiag_config), theta, bins)
            actual = heldout_actual[sign].get(seed)
            heldout_rows.append(
                dict(
                    sign=sign,
                    input_amplitude_deg=theta,
                    seed=seed,
                    predicted_peg_response_deg=float(pred),
                    actual_peg_response_deg=float(actual) if actual is not None else None,
                    error_pred_minus_actual_deg=(
                        float(pred - actual) if actual is not None else None
                    ),
                )
            )
    pd.DataFrame(heldout_rows).to_csv(args.out / "heldout_10degree.csv", index=False)

    result = {
        "schema_version": 2,
        "scope": (
            "Frozen sim-law prediction in the simulation's own nominal geometry; "
            "the real machine contributes only its terminal hand-axis magnitude "
            "(direction-agnostic), not a real nominal trajectory or hole frame. "
            "Held-out +-10 deg is interpolation inside [-15, 15], validated against "
            "independently collected sim_10degree pitch trajectories."
        ),
        "frozen_law_sha256": sha256(args.law_dir / "frozen_law.npz"),
        "frozen_law_summary_sha256": sha256(args.law_dir / "frozen_law.json"),
        "real_csv_sha256": sha256(args.real_csv),
        "heldout_sim_dir": str(args.sim_dir.resolve()),
        "holdout_amplitude_deg": HELDOUT_DEG,
        "predicted_terminal_response_deg": {
            "per_seed": {str(seed): float(value) for seed, value in predictions.items()},
            "mean": float(np.mean(predicted_values)),
            "sd": float(np.std(predicted_values, ddof=1)),
        },
        "real_terminal_response_deg": {
            "point": real_point,
            "bootstrap_ci95": [float(lo), float(hi)],
            "bootstrap_draws": len(real_boot),
        },
        "baselines_deg": {
            "alpha_0": baseline_zero,
            "alpha_1_identity": baseline_identity,
        },
        "law_minus_real_deg": float(np.mean(predicted_values) - real_point),
        "law_minus_identity_deg": float(np.mean(predicted_values) - baseline_identity),
        "heldout_validation": {
            "note": (
                "Independently collected sim +-10 deg pitch trajectories "
                "(sim_10degree, same env/geometry/pickup), held out of the "
                "identification sweep. Interpolation, not extrapolation."
            ),
            "predicted_peg_response_deg": {
                "plus": heldout_pred["plus"],
                "minus": heldout_pred["minus"],
            },
            "actual_peg_response_mean_deg": {
                sign: float(np.mean(list(heldout_actual[sign].values())))
                for sign in ("plus", "minus")
            },
            "actual_peg_response_per_seed_deg": {
                sign: {str(seed): value for seed, value in heldout_actual[sign].items()}
                for sign in ("plus", "minus")
            },
            "error_pred_minus_actual_deg": heldout_error,
        },
        "cross_amplitude_curve": {
            "amplitudes_deg": amplitude_grid,
            "predicted_response_deg": curve_values,
            "holdout_plus_prediction_deg": holdout_plus_pred,
            "holdout_minus_prediction_deg": holdout_minus_pred,
        },
        "interpretation": (
            "The frozen sim law (identified on a pitch sweep excluding +-10 deg, "
            "applied in its own sim nominal geometry) predicts a ~10 deg terminal "
            "peg-axis response for a +-10 deg hole tilt (alpha_pitch -> 1 at "
            "insertion), close to the real machine's 10.30 deg hand-axis terminal "
            "response. Against independently collected held-out sim +-10 deg pitch "
            "trajectories the frozen law's predicted terminal peg response differs "
            "from the actual by +{:.2f} deg (plus) / {:.2f} deg (minus). The real "
            "data is used only for the terminal magnitude, not to reconstruct real "
            "geometry. The terminal magnitude is near-degenerate (alpha_pitch -> 1); "
            "the non-degenerate evidence is the held-out +-10 deg interpolation "
            "error and the progress-resolved alpha_pitch(s) transient, which "
            "currently has simulation evidence only."
        ).format(heldout_error["plus"], heldout_error["minus"]),
        "limitations": [
            "Real tilt axis unknown; magnitude-only, direction-agnostic.",
            "Real data used only for terminal magnitude; no real nominal trajectory or hole frame reconstructed, so this is not a real-geometry trajectory prediction.",
            "Frozen law predicts peg pose in sim geometry; real metric is hand axis. The ~0.1-2 deg hand-peg gap is a SIMULATION measurement (sim_10degree), not a real-machine calibration of the hand-peg transform.",
            "Terminal magnitude is physically forced toward the input amplitude (alpha_pitch -> 1).",
            "Held-out +-10 deg is interpolation inside [-15, 15], not extrapolation; the model-output curve alone is not evidence of generalization.",
            "alpha_pitch(s) transient shape has simulation evidence only; no real transient comparison.",
            "Three simulation seeds; bootstrap is descriptive, not a population guarantee.",
            "No independently justified equivalence margin; report gaps and intervals, no pass/fail.",
        ],
    }
    with (args.out / "prediction_10degree.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)

    plot(
        alpha_profiles,
        progress,
        bins,
        {"amplitudes_deg": amplitude_grid, "predicted_response_deg": curve_values},
        heldout_pred,
        heldout_actual,
        real_point,
        real_boot,
        args.out,
    )

    print(table.to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
