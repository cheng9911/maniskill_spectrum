from __future__ import annotations

"""Raw response + central-difference + restricted pitch-response identification.

For every (mode, block) pair the 0-degree demonstration of that same pair is the
FROZEN nominal (never updated).  Outputs:

  r_m,b,c(s)  = Log( Y_m,b,c(s) @ inv(Y_m,b,0(s)) )   Y = inv(C0) X, C0 = socket frame at pitch=0
  k_m,b,h(s)  = [ r(+h) - r(-h) ] / (2h)               central difference

The identified law is the restricted scalar pitch response ``alpha_pitch(s)``
(xi = (0,0,0,0,theta,0); the five unexcited channels are marked NA, not fitted),
acting as  Xhat_c(s) = Rot_y( alpha_pitch(s) * theta_c , about the socket centre )
applied to the frozen nominal X_0(s).  alpha is a smooth sigmoid-bounded function
in (0, alpha_max); nominal is fixed so the fit cannot drift it toward test data.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "phase_switch_symmetry"))
sys.path.insert(0, str(HERE))

from transforms3d.quaternions import quat2mat  # noqa: E402
from generate_contexts import load_protocol  # noqa: E402
from phase_switch_se3_baselines import (  # noqa: E402
    SE3_METRIC_SCALE,
    euler_from_matrix,
    matrix_from_euler_batched,
    se3_log,
    se3_to_metric,
    wrap_angle,
)

PITCH_CHANNEL = 4  # xi index of the pitch generator


def pose7_to_T(pose7: np.ndarray) -> np.ndarray:
    R = quat2mat(pose7[3:])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pose7[:3]
    return T


def T_to_pose6(T: np.ndarray) -> np.ndarray:
    rpy = euler_from_matrix(T[:3, :3])
    return np.concatenate([T[:3, 3], rpy])


def pose7_to_pose6(pose7: np.ndarray) -> np.ndarray:
    return T_to_pose6(pose7_to_T(pose7))


def relative_response(pose_c: np.ndarray, pose_0: np.ndarray, socket_center: np.ndarray):
    """Return se3_log( Y_c @ inv(Y_0) ), a 6-vector [v; w], Y = inv(C0) X."""
    T_c = pose7_to_T(pose_c)
    T_0 = pose7_to_T(pose_0)
    p0 = socket_center
    T_c_local = T_c.copy(); T_c_local[:3, 3] -= p0
    T_0_local = T_0.copy(); T_0_local[:3, 3] -= p0
    rel = T_c_local @ np.linalg.inv(T_0_local)
    return se3_log(rel)


def axis_response(pose_c: np.ndarray, pose_0: np.ndarray):
    """Angle (rad) between the peg z-axes of c and 0."""
    a_c = quat2mat(pose_c[3:]) @ np.array([0.0, 0.0, 1.0])
    a_0 = quat2mat(pose_0[3:]) @ np.array([0.0, 0.0, 1.0])
    return float(np.arccos(np.clip(np.dot(a_c, a_0), -1.0, 1.0)))


class PitchResponseModel:
    name = "restricted pitch alpha(s), fixed nominal"

    def __init__(self, socket_center, alpha_max=1.25, n_basis=24, basis_width=0.065,
                 smoothness_weight=0.1):
        self.socket_center = np.asarray(socket_center, dtype=np.float64)
        self.alpha_max = float(alpha_max)
        self.n_basis = int(n_basis)
        self.basis_width = float(basis_width)
        self.smoothness_weight = float(smoothness_weight)

    def _basis(self, progress):
        centers = np.linspace(0.0, 1.0, self.n_basis)
        basis = np.exp(-0.5 * ((progress[:, None] - centers[None, :]) / self.basis_width) ** 2)
        return basis / basis.sum(axis=1, keepdims=True)

    def _alpha(self, params):
        logits = self.basis @ params
        return self.alpha_max / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def _apply(self, nominal_pose6, theta, alpha):
        """Rotate nominal about the socket centre by alpha[s]*theta about world y."""
        alpha = np.asarray(alpha, dtype=np.float64)
        ang = alpha * float(theta)
        cp, sp = np.cos(ang), np.sin(ang)
        R = np.zeros((len(ang), 3, 3), dtype=np.float64)
        R[:, 0, 0] = cp
        R[:, 0, 2] = sp
        R[:, 1, 1] = 1.0
        R[:, 2, 0] = -sp
        R[:, 2, 2] = cp
        p0 = self.socket_center
        pos = np.einsum("sij,sj->si", R, nominal_pose6[:, :3] - p0) + p0
        R_nom = matrix_from_euler_batched(nominal_pose6[:, 3:])
        R_out = np.einsum("sij,sjk->sik", R, R_nom)
        rot = np.array([euler_from_matrix(R_out[s]) for s in range(len(ang))])
        return np.concatenate([pos, rot], axis=1)

    def fit(self, nominal_pose7, thetas, curves_pose7, progress):
        self.progress = np.asarray(progress, dtype=np.float64)
        self.basis = self._basis(self.progress)
        n_steps = len(self.progress)
        self.nominal_pose6 = np.array([pose7_to_pose6(p) for p in nominal_pose7])
        curves = np.array([[pose7_to_pose6(p) for p in c] for c in curves_pose7])
        thetas = np.asarray(thetas, dtype=np.float64)

        # initial alpha from a pointwise pitch LSQ
        pitch_curve = curves[..., PITCH_CHANNEL] - self.nominal_pose6[:, PITCH_CHANNEL]
        design = thetas[:, None]
        pointwise = np.array([
            np.linalg.lstsq(design, pitch_curve[:, s], rcond=None)[0][0]
            for s in range(n_steps)
        ])
        pointwise = np.clip(pointwise, 1e-4, self.alpha_max - 1e-4)
        logits = np.log(pointwise / (self.alpha_max - pointwise))
        params = np.linalg.solve(
            self.basis.T @ self.basis + 1e-6 * np.eye(self.n_basis), self.basis.T @ logits
        )

        def residual(candidate):
            alpha = self._alpha(candidate)
            pred = np.array([self._apply(self.nominal_pose6, th, alpha) for th in thetas])
            data_residual = pred - curves
            data_residual[..., 3:] = wrap_angle(data_residual[..., 3:])
            data_residual = se3_to_metric(data_residual).reshape(-1)
            smooth = np.diff(alpha, n=2)
            scale = np.sqrt(self.smoothness_weight * data_residual.size / max(smooth.size, 1))
            smooth_residual = smooth * scale * np.mean(np.abs(thetas))
            return np.concatenate([data_residual, smooth_residual])

        opt = least_squares(residual, params, method="trf", max_nfev=500,
                            ftol=1e-9, xtol=1e-9, gtol=1e-9)
        self.params = opt.x
        self.alpha = self._alpha(self.params)
        self.optimization_success = bool(opt.success)
        self.optimization_cost = float(opt.cost)
        self.optimization_optimality = float(opt.optimality)
        return self

    def predict(self, nominal_pose7, theta):
        nominal_pose6 = np.array([pose7_to_pose6(p) for p in nominal_pose7])
        return self._apply(nominal_pose6, theta, self.alpha)


def identify_stage(protocol, resampled_h5, events_json, source_h5, out_prefix):
    p = protocol
    socket_center = np.asarray(p["task"]["task_anchor"], dtype=np.float64)
    train_pitch = p["train_pitch_deg"]
    with h5py.File(resampled_h5, "r") as rf, h5py.File(source_h5, "r") as sf:
        events = json.load(open(events_json))["episodes"]
        by_ep = {e["episode_id"]: e for e in events}
        # group episodes by (mode_id, block_id)
        groups = {}
        for e in events:
            if e["yaw_mode"]:
                continue  # pitch identification is pitch-only
            key = (e["mode_id"], e["block_id"])
            groups.setdefault(key, []).append(e)

        models = {}
        responses = []
        for (mode, blk), eps in sorted(groups.items()):
            ref = [e for e in eps if abs(e["pitch_deg"]) < 1e-9]
            if not ref:
                continue
            ref_ep = ref[0]
            X0 = rf[ref_ep["episode_id"]]["peg_pose"][:]  # (150,7)
            prog = rf[ref_ep["episode_id"]]["segment_progress"][:]
            seg = rf[ref_ep["episode_id"]]["segment_id"][:]
            global_prog = (seg * 50 + prog) / 150.0

            train_eps = [e for e in eps if e["pitch_deg"] in train_pitch and abs(e["pitch_deg"]) > 1e-9]
            train_eps.sort(key=lambda e: e["pitch_deg"])
            thetas = np.deg2rad([e["pitch_deg"] for e in train_eps])
            curves = np.array([rf[e["episode_id"]]["peg_pose"][:] for e in train_eps])

            model = PitchResponseModel(
                socket_center,
                alpha_max=p["identify"]["fit_config"]["alpha_max"],
                n_basis=p["identify"]["fit_config"]["n_basis"],
                basis_width=p["identify"]["fit_config"]["basis_width"],
                smoothness_weight=p["identify"]["fit_config"]["smoothness_weight"],
            )
            model.fit(X0, thetas, curves, global_prog)
            models[f"{mode}_{blk}"] = dict(
                mode_id=mode, block_id=blk,
                alpha=model.alpha.tolist(),
                optimization_success=model.optimization_success,
                optimization_cost=model.optimization_cost,
                optimization_optimality=model.optimization_optimality,
            )

            # raw responses + central diffs for every condition
            for e in eps:
                Xc = rf[e["episode_id"]]["peg_pose"][:]
                r = np.array([relative_response(Xc[s], X0[s], socket_center) for s in range(len(X0))])
                ax = np.array([axis_response(Xc[s], X0[s]) for s in range(len(X0))])
                responses.append(dict(
                    mode_id=mode, block_id=blk, condition_index=e["condition_index"],
                    pitch_deg=e["pitch_deg"], split=e["split"],
                    r_pos_mm=(r[:, :3] * 1000.0).tolist(),
                    r_pitch_deg=np.rad2deg(r[:, PITCH_CHANNEL]).tolist(),
                    r_rot_deg=np.rad2deg(np.linalg.norm(r[:, 3:], axis=1)).tolist(),
                    r_spin_deg=np.rad2deg(r[:, 5]).tolist(),
                    axis_deg=np.rad2deg(ax).tolist(),
                ))

    with open(out_prefix + ".json", "w", encoding="utf-8") as mf:
        json.dump(dict(protocol_sha256=p["content_sha256"],
                       models=models, responses=responses), mf, indent=2)
    print(f"identified {len(models)} pitch models, {len(responses)} response records")
    return models, responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--resampled", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    identify_stage(protocol, args.resampled, args.events, args.source_h5, str(args.out_prefix))


if __name__ == "__main__":
    main()
