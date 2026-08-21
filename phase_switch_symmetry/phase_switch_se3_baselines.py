from __future__ import annotations

"""SE(3) 6-generator model layer (supplement to the SE(2) baselines).

The task curve becomes the task-local peg pose [x, y, z, roll, pitch, yaw]
expressed in the nominal socket frame, and the Pdiag-finite action is realized in
SE(3): ``C0 Exp(diag(alpha(s)) Log(C0^-1 C)) C0^-1 X0``. The generator diagonal is
now 6-dimensional; the rotation channels use the local convention
``R_local = Rx(roll) Ry(pitch) Rz(yaw)`` (column-vector, yaw innermost), matching
``phase_switch_symmetry_env._local_quat`` exactly.

Nothing here touches the frozen SE(2) classes; the SE(2) model file is left
byte-for-byte unchanged.
"""

import numpy as np
from scipy.optimize import least_squares


SE3_DIM = 6
# 3 translations in metres + 3 rotations at 0.03 m/rad (same axial scale as SE(2)).
SE3_METRIC_SCALE = np.array([1.0, 1.0, 1.0, 0.03, 0.03, 0.03], dtype=np.float64)


def se3_to_metric(values):
    return np.asarray(values, dtype=np.float64) * SE3_METRIC_SCALE


def se3_from_metric(values):
    return np.asarray(values, dtype=np.float64) / SE3_METRIC_SCALE


def wrap_angle(value):
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


# --- SO(3)/SE(3) Lie-group math -------------------------------------------------


def _skew(w):
    return np.array(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]],
        dtype=np.float64,
    )


def so3_exp(w):
    w = np.asarray(w, dtype=np.float64)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + _skew(w)
    W = _skew(w)
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * W
        + ((1.0 - np.cos(theta)) / (theta * theta)) * (W @ W)
    )


def so3_log(R):
    R = np.asarray(R, dtype=np.float64)
    theta = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if theta < 1e-12:
        return 0.5 * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        )
    return (theta / (2.0 * np.sin(theta))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


def se3_exp(xi):
    xi = np.asarray(xi, dtype=np.float64)
    v = xi[:3]
    w = xi[3:]
    theta = float(np.linalg.norm(w))
    R = so3_exp(w)
    if theta < 1e-12:
        V = np.eye(3) + 0.5 * _skew(w)
    else:
        W = _skew(w)
        V = (
            np.eye(3)
            + ((1.0 - np.cos(theta)) / (theta * theta)) * W
            + ((theta - np.sin(theta)) / (theta ** 3)) * (W @ W)
        )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def se3_log(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    w = so3_log(R)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        Vinv = np.eye(3) - 0.5 * _skew(w)
    else:
        W = _skew(w)
        Vinv = (
            np.eye(3)
            - 0.5 * W
            + (1.0 / (theta * theta))
            * (1.0 - (theta * np.sin(theta)) / (2.0 * (1.0 - np.cos(theta))))
            * (W @ W)
        )
    v = Vinv @ t
    return np.concatenate([v, w])


def se3_exp_batched(twists):
    """Vectorized SE(3) exponential: (..., 6) -> (..., 4, 4)."""
    twists = np.asarray(twists, dtype=np.float64)
    v = twists[..., :3]
    w = twists[..., 3:]
    theta = np.linalg.norm(w, axis=-1)
    theta_safe = np.where(theta < 1e-12, 1.0, theta)
    small = theta < 1e-12

    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    W = np.zeros(w.shape[:-1] + (3, 3), dtype=np.float64)
    W[..., 0, 1] = -wz
    W[..., 0, 2] = wy
    W[..., 1, 0] = wz
    W[..., 1, 2] = -wx
    W[..., 2, 0] = -wy
    W[..., 2, 1] = wx
    W2 = np.einsum("...ij,...jk->...ik", W, W)

    a = np.where(small, 1.0, np.sin(theta_safe) / theta_safe)
    b = np.where(small, 0.5, (1.0 - np.cos(theta_safe)) / (theta_safe ** 2))
    c = np.where(small, 1.0 / 6.0, (theta_safe - np.sin(theta_safe)) / (theta_safe ** 3))

    eye = np.zeros(w.shape[:-1] + (3, 3), dtype=np.float64)
    eye[..., 0, 0] = eye[..., 1, 1] = eye[..., 2, 2] = 1.0
    R = eye + a[..., None, None] * W + b[..., None, None] * W2
    V = eye + b[..., None, None] * W + c[..., None, None] * W2
    t = np.einsum("...ij,...j->...i", V, v)

    T = np.zeros(w.shape[:-1] + (4, 4), dtype=np.float64)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def se3_inverse(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def se3_inverse_batched(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


# --- Local euler convention R = Rx(roll) Ry(pitch) Rz(yaw) ----------------------


def matrix_from_euler(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rx @ Ry @ Rz


def euler_from_matrix(R):
    R = np.asarray(R, dtype=np.float64)
    pitch = float(np.arcsin(np.clip(R[0, 2], -1.0, 1.0)))
    roll = float(np.arctan2(-R[1, 2], R[2, 2]))
    yaw = float(np.arctan2(-R[0, 1], R[0, 0]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def matrix_from_euler_batched(rpy):
    rpy = np.asarray(rpy, dtype=np.float64)
    roll, pitch, yaw = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.zeros(roll.shape + (3, 3), dtype=np.float64)
    Rx[..., 0, 0] = 1.0
    Rx[..., 1, 1] = cr
    Rx[..., 1, 2] = -sr
    Rx[..., 2, 1] = sr
    Rx[..., 2, 2] = cr
    Ry = np.zeros(roll.shape + (3, 3), dtype=np.float64)
    Ry[..., 0, 0] = cp
    Ry[..., 0, 2] = sp
    Ry[..., 1, 1] = 1.0
    Ry[..., 2, 0] = -sp
    Ry[..., 2, 2] = cp
    Rz = np.zeros(roll.shape + (3, 3), dtype=np.float64)
    Rz[..., 0, 0] = cy
    Rz[..., 0, 1] = -sy
    Rz[..., 1, 0] = sy
    Rz[..., 1, 1] = cy
    Rz[..., 2, 2] = 1.0
    return np.einsum("...ij,...jk,...kl->...il", Rx, Ry, Rz)


def se3_from_pose6(pose6):
    pose6 = np.asarray(pose6, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = matrix_from_euler(pose6[3], pose6[4], pose6[5])
    T[:3, 3] = pose6[:3]
    return T


def se3_from_pose6_batched(pose6):
    pose6 = np.asarray(pose6, dtype=np.float64)
    T = np.zeros(pose6.shape[:-1] + (4, 4), dtype=np.float64)
    T[..., :3, :3] = matrix_from_euler_batched(pose6[..., 3:])
    T[..., :3, 3] = pose6[..., :3]
    T[..., 3, 3] = 1.0
    return T


def pose6_from_se3_batched(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    roll = np.arctan2(-R[..., 1, 2], R[..., 2, 2])
    pitch = np.arcsin(np.clip(R[..., 0, 2], -1.0, 1.0))
    yaw = np.arctan2(-R[..., 0, 1], R[..., 0, 0])
    out = np.empty(T.shape[:-2] + (6,), dtype=np.float64)
    out[..., :3] = T[..., :3, 3]
    out[..., 3] = roll
    out[..., 4] = pitch
    out[..., 5] = yaw
    return out


def pose6_from_se3(T):
    return pose6_from_se3_batched(np.asarray(T, dtype=np.float64))


# --- Models ----------------------------------------------------------------------


class SE3DiagonalOperatorModel:
    name = "Pdiag pointwise (SE(3))"

    def fit(self, contexts, curves, progress, phase_codes):
        contexts_metric = se3_to_metric(contexts)
        curves_metric = se3_to_metric(curves)
        n_steps = curves.shape[1]
        self.intercept_metric = np.empty((n_steps, SE3_DIM), dtype=np.float64)
        self.diagonal = np.empty((n_steps, SE3_DIM), dtype=np.float64)
        for step in range(n_steps):
            for generator in range(SE3_DIM):
                design = np.column_stack(
                    [np.ones(len(contexts_metric)), contexts_metric[:, generator]]
                )
                coefficients = np.linalg.lstsq(
                    design, curves_metric[:, step, generator], rcond=None
                )[0]
                self.intercept_metric[step, generator] = coefficients[0]
                self.diagonal[step, generator] = coefficients[1]
        return self

    def predict(self, contexts):
        contexts_metric = se3_to_metric(contexts)
        prediction = (
            self.intercept_metric[None, :, :]
            + self.diagonal[None, :, :] * contexts_metric[:, None, :]
        )
        return se3_from_metric(prediction)

    def jacobian_diag(self):
        return self.diagonal.copy()


class SE3FrameWeightedModel:
    name = "Frame-weighted (SE(3))"

    def fit(self, contexts, curves, progress, phase_codes):
        contexts_metric = se3_to_metric(contexts)
        curves_metric = se3_to_metric(curves)
        n_steps = curves.shape[1]
        self.intercept_metric = np.empty((n_steps, SE3_DIM), dtype=np.float64)
        self.weight = np.empty(n_steps, dtype=np.float64)
        for step in range(n_steps):
            c_centered = contexts_metric - contexts_metric.mean(axis=0, keepdims=True)
            y = curves_metric[:, step]
            y_centered = y - y.mean(axis=0, keepdims=True)
            denominator = np.sum(c_centered * c_centered)
            self.weight[step] = np.sum(c_centered * y_centered) / denominator
            self.intercept_metric[step] = (
                y.mean(axis=0) - self.weight[step] * contexts_metric.mean(axis=0)
            )
        return self

    def predict(self, contexts):
        contexts_metric = se3_to_metric(contexts)
        prediction = (
            self.intercept_metric[None, :, :]
            + self.weight[None, :, None] * contexts_metric[:, None, :]
        )
        return se3_from_metric(prediction)

    def jacobian_diag(self):
        return np.repeat(self.weight[:, None], SE3_DIM, axis=1)


class SE3FullOperatorModel:
    name = "Full operator (SE(3))"

    def __init__(self, ridge=1e-8):
        self.ridge = ridge

    def fit(self, contexts, curves, progress, phase_codes):
        contexts_metric = se3_to_metric(contexts)
        curves_metric = se3_to_metric(curves)
        design = np.column_stack([np.ones(len(contexts_metric)), contexts_metric])
        regularizer = self.ridge * np.eye(design.shape[1])
        regularizer[0, 0] = 0.0
        inverse = np.linalg.inv(design.T @ design + regularizer) @ design.T
        n_steps = curves.shape[1]
        self.intercept_metric = np.empty((n_steps, SE3_DIM), dtype=np.float64)
        self.operator_metric = np.empty(
            (n_steps, SE3_DIM, SE3_DIM), dtype=np.float64
        )
        for step in range(n_steps):
            coefficients = inverse @ curves_metric[:, step]
            self.intercept_metric[step] = coefficients[0]
            self.operator_metric[step] = coefficients[1:]
        return self

    def predict(self, contexts):
        contexts_metric = se3_to_metric(contexts)
        prediction = np.empty(
            (len(contexts_metric), len(self.intercept_metric), SE3_DIM),
            dtype=np.float64,
        )
        for step in range(len(self.intercept_metric)):
            prediction[:, step] = (
                self.intercept_metric[step]
                + contexts_metric @ self.operator_metric[step]
            )
        return se3_from_metric(prediction)

    def jacobian_diag(self):
        return np.diagonal(self.operator_metric, axis1=1, axis2=2).copy()


class SE3SmoothFinitePDiagModel:
    name = "Pdiag finite (SE(3))"

    def __init__(
        self,
        nominal_frame_pose,
        alpha_max=1.25,
        n_basis=24,
        basis_width=0.065,
        smoothness_weight=0.1,
        nominal_iterations=3,
    ):
        self.nominal_frame_pose = np.asarray(nominal_frame_pose, dtype=np.float64)
        self.alpha_max = float(alpha_max)
        self.n_basis = int(n_basis)
        self.basis_width = float(basis_width)
        self.smoothness_weight = float(smoothness_weight)
        self.nominal_iterations = int(nominal_iterations)
        self.nominal_frame = se3_from_pose6(self.nominal_frame_pose)
        self.nominal_frame_inverse = se3_inverse(self.nominal_frame)

    def _basis(self, progress):
        centers = np.linspace(0.0, 1.0, self.n_basis)
        basis = np.exp(
            -0.5
            * (
                (np.asarray(progress)[:, None] - centers[None, :])
                / self.basis_width
            )
            ** 2
        )
        return basis / basis.sum(axis=1, keepdims=True)

    def _alpha(self, parameters):
        logits = self.basis @ parameters.reshape(SE3_DIM, self.n_basis).T
        return self.alpha_max / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def _context_twist(self, context):
        context = np.asarray(context, dtype=np.float64)
        intervention = se3_from_pose6(context)
        socket_world = self.nominal_frame @ intervention
        relative = self.nominal_frame_inverse @ socket_world
        return se3_log(relative)

    def _twists(self, contexts):
        return np.asarray([self._context_twist(context) for context in contexts])

    def _local_actions(self, contexts, alpha):
        twists = self._twists(contexts)
        scaled = twists[:, None, :] * alpha[None, :, :]
        return se3_exp_batched(scaled.reshape(-1, 6)).reshape(
            len(contexts), alpha.shape[0], 4, 4
        )

    def _predict_with(self, contexts, nominal_curve, alpha):
        contexts = np.asarray(contexts, dtype=np.float64)
        actions = self._local_actions(contexts, alpha)
        local_nominal = self.nominal_frame_inverse @ se3_from_pose6_batched(
            np.asarray(nominal_curve, dtype=np.float64)
        )
        pred_local = np.einsum("csij,sjk->csik", actions, local_nominal)
        pred_world = np.einsum("ij,csjk->csik", self.nominal_frame, pred_local)
        return pose6_from_se3_batched(pred_world)

    def _update_nominal(self, contexts, curves, alpha):
        contexts = np.asarray(contexts, dtype=np.float64)
        curves = np.asarray(curves, dtype=np.float64)
        actions = self._local_actions(contexts, alpha)
        curves_T = se3_from_pose6_batched(curves.reshape(-1, 6)).reshape(
            len(contexts), curves.shape[1], 4, 4
        )
        obs_local = np.einsum("ij,csjk->csik", self.nominal_frame_inverse, curves_T)
        actions_inv = se3_inverse_batched(actions)
        implied_local = np.einsum("csij,csjk->csik", actions_inv, obs_local)
        implied_world = np.einsum("ij,csjk->csik", self.nominal_frame, implied_local)
        implied = pose6_from_se3_batched(implied_world)
        nominal = np.empty((curves.shape[1], SE3_DIM), dtype=np.float64)
        nominal[:, :3] = implied[:, :, :3].mean(axis=0)
        for generator in range(3):
            angle = implied[:, :, 3 + generator]
            nominal[:, 3 + generator] = np.arctan2(
                np.sin(angle).mean(axis=0), np.cos(angle).mean(axis=0)
            )
        return nominal

    def _initial_parameters(self, pointwise_diagonal):
        clipped = np.clip(
            np.asarray(pointwise_diagonal), 1e-4, self.alpha_max - 1e-4
        )
        logits = np.log(clipped / (self.alpha_max - clipped))
        parameters = []
        regularizer = 1e-6 * np.eye(self.n_basis)
        inverse = np.linalg.solve(
            self.basis.T @ self.basis + regularizer, self.basis.T
        )
        for generator in range(SE3_DIM):
            parameters.append(inverse @ logits[:, generator])
        return np.asarray(parameters).reshape(-1)

    def fit(self, contexts, curves, progress, phase_codes):
        contexts = np.asarray(contexts, dtype=np.float64)
        curves = np.asarray(curves, dtype=np.float64)
        self.progress = np.asarray(progress, dtype=np.float64)
        self.basis = self._basis(self.progress)

        pointwise = SE3DiagonalOperatorModel().fit(
            contexts, curves, progress, phase_codes
        )
        self.nominal_curve = se3_from_metric(pointwise.intercept_metric)
        parameters = self._initial_parameters(pointwise.diagonal)

        for iteration in range(self.nominal_iterations):
            def residual(candidate):
                alpha = self._alpha(candidate)
                prediction = self._predict_with(
                    contexts, self.nominal_curve, alpha
                )
                data_residual = prediction - curves
                data_residual[..., 3:] = wrap_angle(data_residual[..., 3:])
                data_residual = se3_to_metric(data_residual).reshape(-1)
                smooth = np.diff(alpha, n=2, axis=0)
                scale = np.sqrt(
                    self.smoothness_weight
                    * data_residual.size
                    / max(smooth.size, 1)
                ) * np.mean(np.abs(se3_to_metric(contexts)), axis=0)
                smooth_residual = (smooth * scale[None, :]).reshape(-1)
                return np.concatenate([data_residual, smooth_residual])

            optimization = least_squares(
                residual,
                parameters,
                method="trf",
                max_nfev=500,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )
            parameters = optimization.x
            alpha = self._alpha(parameters)
            if iteration + 1 < self.nominal_iterations:
                self.nominal_curve = self._update_nominal(
                    contexts, curves, alpha
                )

        self.parameters = parameters.reshape(SE3_DIM, self.n_basis)
        self.diagonal = self._alpha(parameters)
        self.optimization_success = bool(optimization.success)
        self.optimization_cost = float(optimization.cost)
        self.optimization_nfev = int(optimization.nfev)
        self.optimization_optimality = float(optimization.optimality)
        return self

    def predict(self, contexts):
        return self._predict_with(contexts, self.nominal_curve, self.diagonal)

    def jacobian_diag(self):
        return self.diagonal.copy()
