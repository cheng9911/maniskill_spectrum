from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


ELL = 0.1


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[1], q[2], q[3], q[0]], dtype=np.float64)


def pose_to_matrix(raw_pose: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quat_wxyz_to_xyzw(raw_pose[3:7])).as_matrix()
    transform[:3, 3] = raw_pose[:3]
    return transform


def se3_response(base_pose: np.ndarray, pose: np.ndarray) -> np.ndarray:
    base_tf = pose_to_matrix(base_pose)
    pose_tf = pose_to_matrix(pose)
    rel = np.linalg.inv(base_tf) @ pose_tf
    rotvec = Rotation.from_matrix(rel[:3, :3]).as_rotvec()
    return np.concatenate([rel[:3, 3], ELL * rotvec])


def load_rollouts(path: Path) -> list[dict]:
    episodes = []
    with h5py.File(path, "r") as h5:
        for key in sorted(h5.keys(), key=lambda name: int(name.split("_")[1])):
            ep = h5[key]
            episodes.append(
                {
                    "name": key,
                    "delta": ep["causal_delta"][:].astype(np.float64),
                    "tcp_pose": ep["tcp_pose"][:].astype(np.float64),
                    "phase": ep["phase"][:],
                    "generator": ep.attrs["generator"],
                }
            )
    return episodes


def responses(episodes: list[dict], base_tcp_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = []
    y = []
    for ep in episodes:
        delta = ep["delta"].copy()
        delta[2] *= ELL
        c.append(delta)
        y.append(np.asarray([se3_response(b, p) for b, p in zip(base_tcp_pose, ep["tcp_pose"])]))
    return np.asarray(c), np.asarray(y)


def fit_dense(c: np.ndarray, y: np.ndarray) -> np.ndarray:
    # y: [N, T, 6], c: [N, 3]. Return A: [T, 6, 3], y_t ~= A_t c.
    t_count = y.shape[1]
    mats = []
    for t in range(t_count):
        coeff, *_ = np.linalg.lstsq(c, y[:, t, :], rcond=None)
        mats.append(coeff.T)
    return np.asarray(mats)


def predict_dense(a: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.einsum("tdk,nk->ntd", a, c)


def fit_constant(a_dense: np.ndarray) -> np.ndarray:
    return np.repeat(a_dense.mean(axis=0, keepdims=True), a_dense.shape[0], axis=0)


def fit_scalar(a_dense: np.ndarray) -> np.ndarray:
    # Find best rank-1 in time x flattened-operator space.
    m = a_dense.reshape(a_dense.shape[0], -1)
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    approx = s[0] * np.outer(u[:, 0], vt[0])
    return approx.reshape(a_dense.shape)


def fit_generator(a_dense: np.ndarray) -> np.ndarray:
    # Approximate A_t = A0 @ diag(alpha_t). Fit one temporal vector per generator.
    a0 = a_dense.mean(axis=0)
    out = np.zeros_like(a_dense)
    for t, mat in enumerate(a_dense):
        for j in range(a0.shape[1]):
            denom = float(np.dot(a0[:, j], a0[:, j]))
            alpha = float(np.dot(a0[:, j], mat[:, j]) / denom) if denom > 0 else 0.0
            out[t, :, j] = alpha * a0[:, j]
    return out


def energy_explained(a_dense: np.ndarray, approx: np.ndarray) -> float:
    num = np.sum((a_dense - approx) ** 2)
    den = np.sum(a_dense**2)
    return float(1.0 - num / den)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    scale = np.sqrt(np.mean(y_true**2))
    return float(rmse / scale) if scale > 0 else float("nan")


def profile_corr(isolated: list[dict], base_tcp_pose: np.ndarray) -> dict:
    profiles = {}
    for generator in ["dx", "dy", "dyaw"]:
        eps = [ep for ep in isolated if ep["generator"] == generator and not np.allclose(ep["delta"], 0)]
        per_ep = []
        for ep in eps:
            delta = ep["delta"].copy()
            delta[2] *= ELL
            mag = np.linalg.norm(delta)
            resp = np.asarray([se3_response(b, p) for b, p in zip(base_tcp_pose, ep["tcp_pose"])])
            prof = np.linalg.norm(resp, axis=1) / mag
            if prof.max() > 0:
                prof = prof / prof.max()
            per_ep.append(prof)
        profiles[generator] = np.mean(per_ep, axis=0)

    return {
        "dx_dy": float(np.corrcoef(profiles["dx"], profiles["dy"])[0, 1]),
        "dx_dyaw": float(np.corrcoef(profiles["dx"], profiles["dyaw"])[0, 1]),
        "dy_dyaw": float(np.corrcoef(profiles["dy"], profiles["dyaw"])[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("maniskill_spectrum/plug_charger_causal/geometric_rollouts"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("maniskill_spectrum/plug_charger_causal/GEOMETRIC_ROLLOUT_CLAIM_CHECK.md"),
    )
    args = parser.parse_args()

    isolated = load_rollouts(args.root / "isolated_grid" / "trajectory.h5")
    mixed = load_rollouts(args.root / "train_mixed" / "trajectory.h5")
    base = next(ep for ep in isolated if np.allclose(ep["delta"], np.zeros(3)))

    c_train, y_train = responses(mixed, base["tcp_pose"])
    c_test, y_test = responses(isolated, base["tcp_pose"])
    a_dense = fit_dense(c_train, y_train)

    models = {
        "constant": fit_constant(a_dense),
        "scalar": fit_scalar(a_dense),
        "generator": fit_generator(a_dense),
        "dense": a_dense,
    }

    report = {
        "profile_correlations": profile_corr(isolated, base["tcp_pose"]),
        "operator_energy_explained": {
            name: energy_explained(a_dense, mat) for name, mat in models.items()
        },
        "isolated_test_nrmse": {
            name: nrmse(y_test, predict_dense(mat, c_test)) for name, mat in models.items()
        },
        "isolated_test_nrmse_by_generator": {},
    }

    for gen in ["dx", "dy", "dyaw"]:
        idx = np.asarray([ep["generator"] == gen for ep in isolated])
        report["isolated_test_nrmse_by_generator"][gen] = {
            name: nrmse(y_test[idx], predict_dense(mat, c_test[idx]))
            for name, mat in models.items()
        }

    print(json.dumps(report, indent=2, sort_keys=True))
    args.out.write_text(
        "# Geometric Rollout Claim Check\n\n"
        "Independent reproduction of the key claims on `geometric_rollouts`.\n\n"
        f"```json\n{json.dumps(report, indent=2, sort_keys=True)}\n```\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
