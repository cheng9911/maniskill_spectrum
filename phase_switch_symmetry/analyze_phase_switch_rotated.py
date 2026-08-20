from __future__ import annotations

"""Rotated-axis geometric-consistency analysis (correct canonicalization).

The whole fixture is rigidly transformed by a global transform G_Q = (Q, t_Q)
(orientation quaternion Q about rotation_center t_Q). To compare the learned
generator law across orientations, each peg trajectory is canonicalized by the
INVERSE GLOBAL transform only:

    p_canon = t_Q + Q^T (p_peg - t_Q),   R_canon = Q^-1 R_peg.

This removes the global orientation Q but KEEPS the per-episode causal
intervention c_n = [du, dv, d_axial], so the canonical trajectory
Xbar_Q(c_n) equals the identity-orientation trajectory X_I(c_n). The generator
law P^local(s) is then learned on the canonical trajectory (in the fixed
identity frame, whose nominal socket is SOCKET_CENTER) and must be invariant
across orientations.

Contrast with the (incorrect) per-episode socket-frame removal
R_socket^T (p_peg - p_socket), which cancels the causal response itself
(alpha_u -> 0 instead of 1).
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from transforms3d.euler import quat2euler
from transforms3d.quaternions import qconjugate, qmult, quat2mat

from benchmark_phase_switch_baselines import (
    PHASE_CODES,
    episode_keys,
    progress_grid,
    usable,
)
from phase_switch_baselines import FrameWeightedModel, SmoothFinitePDiagModel


def resample(values, bins):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, bins)
    return np.column_stack(
        [np.interp(target, source, values[:, c]) for c in range(values.shape[1])]
    )


def transform_curve(group, bins, orientation, task_anchor, socket_center):
    """Canonical (G_Q^-1) peg curve [4*bins, 3] in the identity task frame.

    p_world = task_anchor + Q (p_local - socket_center), so
    p_canon = socket_center + Q^T (p_world - task_anchor) = p_local.
    """
    peg = np.asarray(group["peg_pose"])
    phases = np.asarray(group["solver_phase"])
    Q_mat = quat2mat(orientation)
    p_canon = socket_center[None, :] + (peg[:, :3] - task_anchor[None, :]) @ Q_mat
    q_canon = [qmult(qconjugate(orientation), q) for q in peg[:, 3:]]
    yaw = np.unwrap(np.asarray([quat2euler(q)[2] for q in q_canon]))
    x, y = p_canon[:, 0], p_canon[:, 1]
    curves = []
    for phase in PHASE_CODES:
        idx = np.flatnonzero(phases == phase)
        curves.append(resample(np.column_stack([x[idx], y[idx], yaw[idx]]), bins))
    return np.concatenate(curves, axis=0)


def analyze(path, config, bins):
    from phase_switch_symmetry_env import SOCKET_CENTER

    progress, phase_codes = progress_grid(bins)
    with h5py.File(path, "r") as data_file:
        orientation = np.asarray(data_file.attrs["orientation"], dtype=np.float64)
        task_anchor = np.asarray(data_file.attrs["task_anchor"], dtype=np.float64)
        keys = episode_keys(data_file)
        usable_keys = [k for k in keys if usable(data_file[k])]
        mixed_keys = [
            k for k in usable_keys if str(data_file[k].attrs["generator"]) == "mixed"
        ]
        isolated_keys = [
            k
            for k in usable_keys
            if str(data_file[k].attrs["generator"]) in {"yaw", "translation"}
            and np.linalg.norm(np.asarray(data_file[k]["causal_delta"])) > 1e-12
        ]
        baseline_keys = [
            k
            for k in usable_keys
            if np.linalg.norm(np.asarray(data_file[k]["causal_delta"])) <= 1e-12
        ]
        if len(mixed_keys) != 30 or len(isolated_keys) != 8 or len(baseline_keys) != 1:
            raise RuntimeError(f"{path}: expected 30 mixed, 8 isolated, 1 baseline")

        mixed_contexts = np.asarray(
            [np.asarray(data_file[k]["causal_delta"]) for k in mixed_keys]
        )
        mixed_curves = np.asarray(
            [
                transform_curve(data_file[k], bins, orientation, task_anchor, SOCKET_CENTER)
                for k in mixed_keys
            ]
        )
        isolated_contexts = np.asarray(
            [np.asarray(data_file[k]["causal_delta"]) for k in isolated_keys]
        )
        isolated_curves = np.asarray(
            [
                transform_curve(data_file[k], bins, orientation, task_anchor, SOCKET_CENTER)
                for k in isolated_keys
            ]
        )

        # Canonical nominal socket frame is the fixed identity SOCKET_CENTER.
        nominal_frame_xy = SOCKET_CENTER[:2].copy()
        nominal_frame_yaw = 0.0

        pdiag = SmoothFinitePDiagModel(
            nominal_frame_xy=nominal_frame_xy,
            nominal_frame_yaw=nominal_frame_yaw,
            **config,
        ).fit(mixed_contexts, mixed_curves, progress, phase_codes)
        scalar = FrameWeightedModel().fit(
            mixed_contexts, mixed_curves, progress, phase_codes
        )
        return pdiag.jacobian_diag(), pdiag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        required=True,
        help="H5 rollouts, one per orientation, in order.",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None, help="Optional orientation labels."
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/experiment.json"),
    )
    parser.add_argument("--bins", type=int, default=25)
    args = parser.parse_args()

    with args.experiment.open() as f:
        experiment = json.load(f)
    config = {
        "alpha_max": experiment["frozen_model"]["alpha_max"],
        "n_basis": experiment["frozen_model"]["n_basis"],
        "basis_width": experiment["frozen_model"]["basis_width"],
        "smoothness_weight": experiment["frozen_model"]["smoothness_weight"],
        "nominal_iterations": experiment["frozen_model"]["nominal_iterations"],
    }
    labels = args.labels if args.labels else [f"orient_{i}" for i in range(len(args.files))]

    progress, _ = progress_grid(args.bins)
    profiles = {}
    rows = []
    for label, path in zip(labels, args.files):
        alpha, _ = analyze(path, config, args.bins)
        profiles[label] = alpha
        rows.append(
            {
                "label": label,
                "g_translation_final": float(alpha[-1, :2].mean()),
                "g_yaw_preclear": float(alpha[2 * args.bins - 1, 2]),
                "g_yaw_final": float(alpha[-1, 2]),
            }
        )

    # Cross-orientation invariance of the canonical generator law.
    inv_rows = []
    ref_label = labels[0]
    for label in labels[1:]:
        a, b = profiles[ref_label], profiles[label]
        inv_rows.append(
            {
                "pair": f"{ref_label} vs {label}",
                "yaw_corr": float(np.corrcoef(a[:, 2], b[:, 2])[0, 1]),
                "full_rmse": float(np.sqrt(np.mean((a - b) ** 2))),
                "yaw_rmse": float(np.sqrt(np.mean((a[:, 2] - b[:, 2]) ** 2))),
            }
        )

    out_dir = Path("phase_switch_symmetry_rollouts_rotated")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "rotated_profiles.csv", index=False)
    pd.DataFrame(inv_rows).to_csv(out_dir / "rotated_invariance.csv", index=False)
    np.savez_compressed(
        out_dir / "rotated_profiles.npz",
        progress=progress,
        **{label: profiles[label] for label in labels},
    )

    print(pd.DataFrame(rows).to_string(index=False))
    print("\nCross-orientation canonical-law invariance:")
    print(pd.DataFrame(inv_rows).to_string(index=False))


if __name__ == "__main__":
    main()
