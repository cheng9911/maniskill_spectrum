from __future__ import annotations

"""Identify the cross-amplitude tilt response law alpha_pitch(s) and freeze it.

P_sim(s) = the Pdiag-finite (SE(3)) generator-relevance profile ``alpha(s)``
(n_steps x 6), realized by ``C0 Exp(diag(alpha(s)) Log(C0^-1 C)) C0^-1 X0(s)``.
For a pure-pitch intervention only the pitch channel (index 4) is identifiable;
the other five are nominally ~0 (no context variation in the sweep). The pitch
sweep EXCLUDES +-10 deg (held out), so a later prediction at +10 deg is a genuine
out-of-sample interpolation between +-7.5 and +-12.5 deg.

Each seed is fit independently (matching the SE(3) benchmark discipline); the
three frozen laws are saved together with provenance so predict_10degree.py can
reconstruct the exact model object and apply the finite SE(3) action without
re-identifying.
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from benchmark_phase_switch_baselines import (
    PHASE_CODES,
    progress_grid,
    usable,
)
from benchmark_se3_transfer import (
    nominal_frame_se3,
    pose_to_pose6,
    task_curve_se3,
)
from phase_switch_se3_baselines import (
    SE3_DIM,
    SE3SmoothFinitePDiagModel,
    euler_from_matrix,
    se3_from_pose6,
    se3_inverse,
)

PITCH_INDEX = 4
GENERATOR_NAMES = ("du", "dv", "dw", "roll", "pitch", "yaw")
SEEDS = [20260910, 20270910, 20280910]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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


def load_seed(data_path: Path, bins: int):
    with h5py.File(data_path, "r") as data_file:
        usable_keys = [
            key for key in data_file if key.startswith("episode_") and usable(data_file[key])
        ]
        # Keep the manifest order (episode ids are sequential per condition).
        usable_keys.sort(key=lambda key: int(key.split("_")[-1]))
        contexts = np.asarray(
            [np.asarray(data_file[key]["causal_delta"]) for key in usable_keys],
            dtype=np.float64,
        )
        curves = np.asarray(
            [task_curve_se3(data_file[key], bins) for key in usable_keys],
            dtype=np.float64,
        )
        nominal = nominal_frame_se3(data_file, usable_keys, contexts)
    return usable_keys, contexts, curves, nominal


def fit_law(contexts, curves, nominal, pdiag_config, bins):
    model = SE3SmoothFinitePDiagModel(nominal_frame_pose=nominal, **pdiag_config)
    progress, phase_codes = progress_grid(bins)
    model.fit(contexts, curves, progress, phase_codes)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("real_relation_consistency/tilt_sweep"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("real_relation_consistency/tilt_sweep/frozen_law.npz"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("real_relation_consistency/tilt_sweep/frozen_law.json"),
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
    progress, phase_codes = progress_grid(args.bins)

    per_seed = {}
    for seed in SEEDS:
        data_path = args.sweep_dir / f"circular_seed_{seed}.h5"
        if not data_path.exists():
            raise FileNotFoundError(f"missing sweep data: {data_path}")
        keys, contexts, curves, nominal = load_seed(data_path, args.bins)
        model = fit_law(contexts, curves, nominal, pdiag_config, args.bins)
        alpha = model.diagonal  # (n_steps, 6)
        alpha_pitch = alpha[:, PITCH_INDEX]
        per_seed[seed] = {
            "nominal_frame_pose": np.asarray(nominal, dtype=np.float64),
            "nominal_curve": np.asarray(model.nominal_curve, dtype=np.float64),
            "diagonal": np.asarray(alpha, dtype=np.float64),
            "parameters": np.asarray(model.parameters, dtype=np.float64),
            "contexts": np.asarray(contexts, dtype=np.float64),
            "curves": np.asarray(curves, dtype=np.float64),
            "episode_keys": [str(key) for key in keys],
            "alpha_pitch": np.asarray(alpha_pitch, dtype=np.float64),
            "optimization_success": bool(model.optimization_success),
            "optimization_cost": float(model.optimization_cost),
            "optimization_nfev": int(model.optimization_nfev),
        }
        term = float(alpha_pitch[-1])
        mean_insert = float(alpha_pitch[2 * args.bins :].mean())
        print(
            f"seed {seed}: n_usable={len(keys)} alpha_pitch(terminal)={term:.4f} "
            f"mean_insert={mean_insert:.4f} opt_ok={model.optimization_success} "
            f"nfev={model.optimization_nfev}",
            flush=True,
        )

    np.savez_compressed(
        args.output,
        progress=np.asarray(progress, dtype=np.float64),
        phase_codes=np.asarray(phase_codes, dtype=np.int64),
        pitch_index=PITCH_INDEX,
        **{
            f"seed_{seed}_{field}": value
            for seed, law in per_seed.items()
            for field, value in law.items()
            if isinstance(value, np.ndarray)
        },
    )

    alpha_terminal = [float(law["alpha_pitch"][-1]) for law in per_seed.values()]
    summary = {
        "schema_version": 1,
        "model": "Pdiag finite (SE(3)) = SE3SmoothFinitePDiagModel",
        "finite_action": "C0 Exp(diag(alpha(s)) Log(C0^-1 C)) C0^-1 X0(s)",
        "pitch_index": PITCH_INDEX,
        "generator_names": list(GENERATOR_NAMES),
        "seeds": SEEDS,
        "holdout_amplitudes_deg": [10.0, -10.0],
        "sweep_amplitudes_deg": [0.0, 2.5, -2.5, 5.0, -5.0, 7.5, -7.5, 12.5, -12.5, 15.0, -15.0],
        "bins_per_phase": args.bins,
        "pdiag_config": {
            "alpha_max": args.pdiag_alpha_max,
            "n_basis": args.pdiag_basis_count,
            "basis_width": args.pdiag_basis_width,
            "smoothness_weight": args.pdiag_smoothness,
            "nominal_iterations": args.pdiag_nominal_iterations,
        },
        "alpha_pitch_terminal": alpha_terminal,
        "alpha_pitch_terminal_mean": float(np.mean(alpha_terminal)),
        "alpha_pitch_terminal_sd": float(np.std(alpha_terminal, ddof=1)),
        "source_sha256": {
            "identify_pitch_law.py": sha256(Path(__file__)),
            "phase_switch_se3_baselines.py": sha256(
                Path(__file__).with_name("phase_switch_se3_baselines.py")
            ),
            "benchmark_se3_transfer.py": sha256(
                Path(__file__).with_name("benchmark_se3_transfer.py")
            ),
        },
        "data_sha256": {
            str(seed): sha256(args.sweep_dir / f"circular_seed_{seed}.h5")
            for seed in SEEDS
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "h5py": h5py.__version__,
        },
        "per_seed": {
            str(seed): {
                "n_usable": len(law["episode_keys"]),
                "optimization_success": law["optimization_success"],
                "optimization_cost": law["optimization_cost"],
                "optimization_nfev": law["optimization_nfev"],
                "alpha_pitch_terminal": float(law["alpha_pitch"][-1]),
            }
            for seed, law in per_seed.items()
        },
    }
    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(summary), handle, indent=2, allow_nan=False)

    print("saved:", args.output)
    print("saved:", args.summary)
    print("alpha_pitch(terminal):", alpha_terminal)


if __name__ == "__main__":
    main()
