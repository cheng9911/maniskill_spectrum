"""Estimate paired SE(3) response matrices from frozen keyed simulation traces.

Each N=30 subset is fit by affine least squares at 25 points per phase.
Input: metric-scaled Log(intervention). Output: metric-scaled spatial residual
Log(C0^-1 X X_baseline^-1 C0). Both input AND output are re-expressed in
the rotated basis; A_rot = Q.T A_local Q. No diagonal prior or random matrix.
"""
from pathlib import Path
import csv
import hashlib
import json

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
PREFIX = Path(__file__).resolve().parent / "fig3_basis_empirical"
SEEDS = (20260818, 20270818, 20280818)
PHASES = (3, 4, 5, 6)
NAMES = ("align", "enter", "unlock", "insert")
GENS = ("du", "dv", "dw", "roll", "pitch", "yaw")
SCALE = np.array([1, 1, 1, .03, .03, .03])
ROTATION_DEG = (30., -20., 40.)  # frozen rotated-1 convention Rx Ry Rz


def inverse(T):
    out = np.broadcast_to(np.eye(4), T.shape).copy()
    out[..., :3, :3] = np.swapaxes(T[..., :3, :3], -1, -2)
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", out[..., :3, :3], T[..., :3, 3])
    return out


def pose_matrix(pose):
    out = np.broadcast_to(np.eye(4), pose.shape[:-1] + (4, 4)).copy()
    out[..., :3, :3] = Rotation.from_quat(pose[..., [4, 5, 6, 3]]).as_matrix()
    out[..., :3, 3] = pose[..., :3]
    return out


def context_matrix(context):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler("XYZ", context[3:]).as_matrix()
    out[:3, 3] = context[:3]
    return out


def log_se3(T):
    """Stable SE(3) logarithm, v = J_left(w)^-1 t; supports batches."""
    shape = T.shape[:-2]
    flat = T.reshape(-1, 4, 4)
    w = Rotation.from_matrix(flat[:, :3, :3]).as_rotvec()
    theta = np.linalg.norm(w, axis=1)
    W = np.zeros((len(flat), 3, 3))
    W[:, 0, 1], W[:, 0, 2] = -w[:, 2], w[:, 1]
    W[:, 1, 0], W[:, 1, 2] = w[:, 2], -w[:, 0]
    W[:, 2, 0], W[:, 2, 1] = -w[:, 1], w[:, 0]
    coefficient = np.full(len(flat), 1/12)
    large = theta > 1e-5
    t = theta[large]
    coefficient[large] = (1 - .5*t/np.tan(.5*t)) / t**2
    Jinv = np.eye(3) - .5*W + coefficient[:, None, None]*(W @ W)
    v = np.einsum("nij,nj->ni", Jinv, flat[:, :3, 3])
    return np.concatenate([v, w], axis=1).reshape(shape + (6,))


def curve(group):
    """Match benchmark's 25-bin phase interpolation of xyz and unwrapped XYZ."""
    pose = group["peg_pose"][:]
    codes = group["solver_phase"][:]
    result = []
    for code in PHASES:
        phase = pose[codes == code]
        rpy = Rotation.from_quat(phase[:, [4, 5, 6, 3]]).as_euler("XYZ")
        values = np.column_stack([phase[:, :3], np.unwrap(rpy, axis=0)])
        xp = np.linspace(0, 1, len(values))
        if len(xp) < 2:
            raise ValueError("Phase has fewer than two samples")
        sampled = np.column_stack([np.interp(np.linspace(0, 1, 25), xp, v)
                                   for v in values.T])
        T = np.broadcast_to(np.eye(4), (25, 4, 4)).copy()
        T[:, :3, :3] = Rotation.from_euler("XYZ", sampled[:, 3:]).as_matrix()
        T[:, :3, 3] = sampled[:, :3]
        result.append(T)
    return np.concatenate(result)


def usable(g):
    return (bool(g["success"][-1]) and not bool(g["truncated"][:].any())
            and set(PHASES).issubset(set(g["solver_phase"][:])))


def off_ratio(A):
    diagonal = np.diagonal(A, axis1=-2, axis2=-1)[..., :, None]*np.eye(6)
    return np.linalg.norm(A-diagonal, axis=(-2, -1))/np.linalg.norm(A, axis=(-2, -1))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    manifest_path = ROOT / "phase_switch_symmetry_multiseed/se3_subsets.json"
    subsets = [s for s in json.loads(manifest_path.read_text())["subsets"]
               if s["sample_size"] == 30]
    R = Rotation.from_euler("XYZ", ROTATION_DEG, degrees=True).as_matrix()
    Q = np.kron(np.eye(2), R)
    operators, fit_info, provenance = [], [], []
    for seed in SEEDS:
        path = ROOT / f"phase_switch_symmetry_rollouts_se3/keyed_seed_{seed}.h5"
        with h5py.File(path, "r") as data:
            # Match benchmark selection: last usable attempt in HDF5 key order.
            selected = {int(g.attrs["condition_id"]): k for k, g in data.items() if usable(g)}
            baseline_keys = [k for k in selected.values()
                             if data[k].attrs["generator"] == "baseline"]
            if len(baseline_keys) != 1:
                raise ValueError("Expected one usable zero-intervention baseline")
            baseline = data[baseline_keys[0]]
            if np.linalg.norm(baseline["causal_delta"][:]) > 1e-12:
                raise ValueError("Baseline is not a zero intervention")
            C0 = pose_matrix(baseline["socket_pose"][0])
            X0 = curve(baseline)
            needed = sorted({cid for s in subsets for cid in s["source_condition_ids"]})
            cache = {}
            for cid in needed:
                if cid not in selected:
                    raise ValueError(f"Missing usable condition {cid} for seed {seed}")
                g = data[selected[cid]]
                U = context_matrix(g["causal_delta"][:])
                recovered = pose_matrix(g["socket_pose"][0]) @ inverse(U)
                if not np.allclose(recovered, C0, atol=2e-5):
                    raise ValueError("Nominal socket frame is inconsistent across interventions")
                x = log_se3(U)*SCALE
                y = log_se3(inverse(C0) @ curve(g) @ inverse(X0) @ C0)*SCALE
                cache[cid] = (x, y)
            for s in subsets:
                cids = s["source_condition_ids"]
                x = np.stack([cache[c][0] for c in cids])
                y = np.stack([cache[c][1] for c in cids])
                design = np.column_stack([x, np.ones(len(x))])
                coef, _, rank, _ = np.linalg.lstsq(design, y.reshape(len(x), -1), rcond=None)
                if rank != 7:
                    raise ValueError("Rank-deficient affine response estimate")
                A = coef[:6].reshape(6, 100, 6).transpose(1, 2, 0)
                rotated_design = np.column_stack([x @ Q, np.ones(len(x))])
                rotated_y = y @ Q
                rotated_coef = np.linalg.lstsq(rotated_design, rotated_y.reshape(len(x), -1), rcond=None)[0]
                Ar = rotated_coef[:6].reshape(6, 100, 6).transpose(1, 2, 0)
                np.testing.assert_allclose(Ar, Q.T @ A @ Q, atol=2e-10)
                np.testing.assert_allclose(np.linalg.norm(Ar, axis=(1, 2)),
                                           np.linalg.norm(A, axis=(1, 2)), atol=2e-10)
                operators.append(np.stack([A, Ar]).reshape(2, 4, 25, 6, 6).mean(axis=2))
                fit_info.append({"seed": seed, "subset_id": s["subset_id"],
                                 "protocol": s["protocol"], "condition_ids": cids,
                                 "episodes": [selected[c] for c in cids],
                                 "rank": int(rank),
                                 "condition_number_scaled": float(np.linalg.cond(x/x.std(axis=0))),
                                 "rmse_metric": float(np.sqrt(np.mean((design@coef-y.reshape(len(x), -1))**2)))})
            provenance.append({"file": path.relative_to(ROOT).as_posix(), "sha256": digest(path),
                               "total_attempts": len(data), "usable_conditions": len(selected),
                               "selected_unique_conditions": len(needed), "baseline": baseline_keys[0]})
    operators = np.stack(operators)  # fit x basis x phase x output x input
    mean = operators.mean(axis=0)
    np.savez_compressed(PREFIX.with_suffix(".npz"), per_fit=operators, mean=mean,
                        sd=operators.std(axis=0, ddof=1), Q=Q, metric_scale=SCALE,
                        seeds=np.array([s["seed"] for s in fit_info]),
                        subset_ids=np.array([s["subset_id"] for s in fit_info]))
    with PREFIX.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["basis", "phase", "output", "input", "mean", "sd_across_fits"])
        sd = operators.std(axis=0, ddof=1)
        for (b, p, i, j), value in np.ndenumerate(mean):
            writer.writerow([("local", "rotated-1")[b], NAMES[p], GENS[i], GENS[j], value, sd[b,p,i,j]])
    meta = {"task": "keyed", "phase_displayed": "insert", "sample_size": 30,
            "fit_count": len(fit_info), "rotation_XYZ_degrees": ROTATION_DEG,
            "operator_convention": "metric spatial response; rows=output, columns=input; both axes rotated",
            "transform": "Q=diag(R,R); A_rot=Q.T @ A_local @ Q",
            "estimator": "unregularized affine least squares per progress point; phase mean then equal fit mean",
            "uncertainty": "SD across 3 seeds x 6 overlapping subsets; not 18 independent seeds",
            "off_ratio_of_mean": off_ratio(mean).tolist(),
            "off_ratio_per_fit_mean": off_ratio(operators).mean(axis=0).tolist(),
            "off_ratio_per_fit_sd": off_ratio(operators).std(axis=0, ddof=1).tolist(),
            "provenance": provenance, "fits": fit_info,
            "manifest_sha256": digest(manifest_path), "script_sha256": digest(Path(__file__))}
    PREFIX.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
    print("off ratios of mean [local, rotated] x [align, enter, unlock, insert]:")
    print(off_ratio(mean))
    print("insert matrices:", mean[:, 3], "\nfits:", len(fit_info))


if __name__ == "__main__":
    main()
