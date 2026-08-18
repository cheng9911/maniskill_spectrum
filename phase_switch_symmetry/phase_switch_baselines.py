from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.model_selection import GroupKFold
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import RidgeCV
from threadpoolctl import threadpool_limits


ROTATION_SCALE_M_PER_RAD = 0.03
METRIC_SCALE = np.array([1.0, 1.0, ROTATION_SCALE_M_PER_RAD])


def to_metric(values):
    return np.asarray(values, dtype=np.float64) * METRIC_SCALE


def from_metric(values):
    return np.asarray(values, dtype=np.float64) / METRIC_SCALE


def wrap_angle(value):
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def pose2_matrix(pose):
    x, y, yaw = np.asarray(pose, dtype=np.float64)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    return np.array(
        [[cosine, -sine, x], [sine, cosine, y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def matrix_pose2(transform):
    transform = np.asarray(transform, dtype=np.float64)
    return np.array(
        [
            transform[0, 2],
            transform[1, 2],
            np.arctan2(transform[1, 0], transform[0, 0]),
        ]
    )


def se2_exp(twist):
    vx, vy, yaw = np.asarray(twist, dtype=np.float64)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    if abs(yaw) < 1e-8:
        a = 1.0 - yaw**2 / 6.0
        b = 0.5 * yaw - yaw**3 / 24.0
    else:
        a = sine / yaw
        b = (1.0 - cosine) / yaw
    translation = np.array([[a, -b], [b, a]]) @ np.array([vx, vy])
    return np.array(
        [
            [cosine, -sine, translation[0]],
            [sine, cosine, translation[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def se2_log(transform):
    transform = np.asarray(transform, dtype=np.float64)
    yaw = np.arctan2(transform[1, 0], transform[0, 0])
    if abs(yaw) < 1e-8:
        a = 1.0 - yaw**2 / 6.0
        b = 0.5 * yaw - yaw**3 / 24.0
    else:
        a = np.sin(yaw) / yaw
        b = (1.0 - np.cos(yaw)) / yaw
    velocity = np.linalg.solve(
        np.array([[a, -b], [b, a]]), transform[:2, 2]
    )
    return np.array([velocity[0], velocity[1], yaw], dtype=np.float64)


def socket_frame(context, nominal_xy, nominal_yaw=0.0):
    context = np.asarray(context, dtype=np.float64)
    return pose2_matrix(
        [
            nominal_xy[0] + context[0],
            nominal_xy[1] + context[1],
            nominal_yaw + context[2],
        ]
    )


class FrameWeightedModel:
    name = "Frame-weighted"

    def fit(self, contexts, curves, progress, phase_codes):
        contexts_metric = to_metric(contexts)
        curves_metric = to_metric(curves)
        n_steps = curves.shape[1]
        self.intercept_metric = np.empty((n_steps, 3), dtype=np.float64)
        self.weight = np.empty(n_steps, dtype=np.float64)
        for step in range(n_steps):
            c_centered = contexts_metric - contexts_metric.mean(axis=0, keepdims=True)
            y = curves_metric[:, step]
            y_centered = y - y.mean(axis=0, keepdims=True)
            denominator = np.sum(c_centered * c_centered)
            weight = np.sum(c_centered * y_centered) / denominator
            self.weight[step] = weight
            self.intercept_metric[step] = y.mean(axis=0) - weight * contexts_metric.mean(axis=0)
        return self

    def predict(self, contexts):
        contexts_metric = to_metric(contexts)
        prediction = (
            self.intercept_metric[None, :, :]
            + self.weight[None, :, None] * contexts_metric[:, None, :]
        )
        return from_metric(prediction)

    def jacobian_diag(self):
        return np.repeat(self.weight[:, None], 3, axis=1)


class PhaseScalarGPModel(FrameWeightedModel):
    name = "Phase scalar GP"

    def fit(self, contexts, curves, progress, phase_codes):
        super().fit(contexts, curves, progress, phase_codes)
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * RBF(length_scale=0.12, length_scale_bounds=(0.015, 1.0))
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=20260820,
        )
        self.progress = np.asarray(progress, dtype=np.float64)
        self.gp.fit(self.progress[:, None], self.weight)
        self.weight = self.gp.predict(self.progress[:, None])
        contexts_metric = to_metric(contexts)
        curves_metric = to_metric(curves)
        self.intercept_metric = (
            curves_metric.mean(axis=0)
            - self.weight[:, None] * contexts_metric.mean(axis=0)[None, :]
        )
        return self


class DiagonalOperatorModel:
    name = "Pdiag pointwise"

    def fit(self, contexts, curves, progress, phase_codes):
        contexts_metric = to_metric(contexts)
        curves_metric = to_metric(curves)
        n_steps = curves.shape[1]
        self.intercept_metric = np.empty((n_steps, 3), dtype=np.float64)
        self.diagonal = np.empty((n_steps, 3), dtype=np.float64)
        for step in range(n_steps):
            for generator in range(3):
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
        contexts_metric = to_metric(contexts)
        prediction = (
            self.intercept_metric[None, :, :]
            + self.diagonal[None, :, :] * contexts_metric[:, None, :]
        )
        return from_metric(prediction)

    def jacobian_diag(self):
        return self.diagonal.copy()


class SmoothFinitePDiagModel:
    name = "Pdiag finite"

    def __init__(
        self,
        nominal_frame_xy,
        nominal_frame_yaw=0.0,
        alpha_max=1.25,
        n_basis=24,
        basis_width=0.065,
        smoothness_weight=0.1,
        nominal_iterations=3,
    ):
        self.nominal_frame_xy = np.asarray(nominal_frame_xy, dtype=np.float64)
        self.nominal_frame_yaw = float(nominal_frame_yaw)
        self.alpha_max = float(alpha_max)
        self.n_basis = int(n_basis)
        self.basis_width = float(basis_width)
        self.smoothness_weight = float(smoothness_weight)
        self.nominal_iterations = int(nominal_iterations)
        self.nominal_frame = pose2_matrix(
            [*self.nominal_frame_xy, self.nominal_frame_yaw]
        )
        self.nominal_frame_inverse = np.linalg.inv(self.nominal_frame)

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
        logits = self.basis @ parameters.reshape(3, self.n_basis).T
        return self.alpha_max / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def _context_twist(self, context):
        relative = self.nominal_frame_inverse @ socket_frame(
            context, self.nominal_frame_xy, self.nominal_frame_yaw
        )
        return se2_log(relative)

    def _selected_action(self, context_twist, alpha):
        return (
            self.nominal_frame
            @ se2_exp(alpha * context_twist)
            @ self.nominal_frame_inverse
        )

    def _batched_local_actions(self, contexts, alpha):
        twists = np.asarray(
            [self._context_twist(context) for context in contexts]
        )
        selected = twists[:, None, :] * alpha[None, :, :]
        yaw = selected[:, :, 2]
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        small = np.abs(yaw) < 1e-8
        a = np.empty_like(yaw)
        b = np.empty_like(yaw)
        a[small] = 1.0 - yaw[small] ** 2 / 6.0
        b[small] = 0.5 * yaw[small] - yaw[small] ** 3 / 24.0
        a[~small] = sine[~small] / yaw[~small]
        b[~small] = (1.0 - cosine[~small]) / yaw[~small]
        vx = selected[:, :, 0]
        vy = selected[:, :, 1]
        translation = np.stack([a * vx - b * vy, b * vx + a * vy], axis=-1)
        return yaw, cosine, sine, translation

    def _predict_with(self, contexts, nominal_curve, alpha):
        contexts = np.asarray(contexts, dtype=np.float64)
        yaw, cosine, sine, translation = self._batched_local_actions(
            contexts, alpha
        )
        frame_rotation = self.nominal_frame[:2, :2]
        nominal_local = (
            np.asarray(nominal_curve)[:, :2] - self.nominal_frame_xy
        ) @ frame_rotation
        local_x = nominal_local[None, :, 0]
        local_y = nominal_local[None, :, 1]
        transformed_local = np.stack(
            [
                cosine * local_x - sine * local_y + translation[:, :, 0],
                sine * local_x + cosine * local_y + translation[:, :, 1],
            ],
            axis=-1,
        )
        prediction = np.empty(
            (len(contexts), len(nominal_curve), 3), dtype=np.float64
        )
        prediction[:, :, :2] = (
            transformed_local @ frame_rotation.T
            + self.nominal_frame_xy[None, None, :]
        )
        prediction[:, :, 2] = wrap_angle(
            np.asarray(nominal_curve)[None, :, 2] + yaw
        )
        return prediction

    def _initial_parameters(self, pointwise_diagonal):
        clipped = np.clip(
            np.asarray(pointwise_diagonal),
            1e-4,
            self.alpha_max - 1e-4,
        )
        logits = np.log(clipped / (self.alpha_max - clipped))
        parameters = []
        regularizer = 1e-6 * np.eye(self.n_basis)
        inverse = np.linalg.solve(
            self.basis.T @ self.basis + regularizer, self.basis.T
        )
        for generator in range(3):
            parameters.append(inverse @ logits[:, generator])
        return np.asarray(parameters).reshape(-1)

    def _update_nominal(self, contexts, curves, alpha):
        yaw, cosine, sine, translation = self._batched_local_actions(
            contexts, alpha
        )
        frame_rotation = self.nominal_frame[:2, :2]
        observed_local = (
            np.asarray(curves)[:, :, :2]
            - self.nominal_frame_xy[None, None, :]
        ) @ frame_rotation
        shifted = observed_local - translation
        implied_local = np.stack(
            [
                cosine * shifted[:, :, 0] + sine * shifted[:, :, 1],
                -sine * shifted[:, :, 0] + cosine * shifted[:, :, 1],
            ],
            axis=-1,
        )
        implied = np.empty_like(curves, dtype=np.float64)
        implied[:, :, :2] = (
            implied_local @ frame_rotation.T
            + self.nominal_frame_xy[None, None, :]
        )
        implied[:, :, 2] = wrap_angle(curves[:, :, 2] - yaw)
        nominal = np.empty((curves.shape[1], 3), dtype=np.float64)
        nominal[:, :2] = implied[:, :, :2].mean(axis=0)
        nominal[:, 2] = np.arctan2(
            np.sin(implied[:, :, 2]).mean(axis=0),
            np.cos(implied[:, :, 2]).mean(axis=0),
        )
        return nominal

    def fit(self, contexts, curves, progress, phase_codes):
        contexts = np.asarray(contexts, dtype=np.float64)
        curves = np.asarray(curves, dtype=np.float64)
        self.progress = np.asarray(progress, dtype=np.float64)
        self.basis = self._basis(self.progress)

        pointwise = DiagonalOperatorModel().fit(
            contexts, curves, progress, phase_codes
        )
        self.nominal_curve = from_metric(pointwise.intercept_metric)
        parameters = self._initial_parameters(pointwise.diagonal)

        for iteration in range(self.nominal_iterations):
            def residual(candidate):
                alpha = self._alpha(candidate)
                prediction = self._predict_with(
                    contexts, self.nominal_curve, alpha
                )
                data_residual = prediction - curves
                data_residual[:, :, 2] = wrap_angle(data_residual[:, :, 2])
                data_residual = to_metric(data_residual).reshape(-1)
                smooth = np.diff(alpha, n=2, axis=0)
                scale = np.sqrt(
                    self.smoothness_weight
                    * data_residual.size
                    / max(smooth.size, 1)
                ) * np.mean(np.abs(to_metric(contexts)), axis=0)
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

        self.parameters = parameters.reshape(3, self.n_basis)
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


class FullOperatorModel:
    name = "Full operator"

    def __init__(self, ridge=1e-8):
        self.ridge = ridge

    def fit(self, contexts, curves, progress, phase_codes):
        contexts_metric = to_metric(contexts)
        curves_metric = to_metric(curves)
        design = np.column_stack([np.ones(len(contexts_metric)), contexts_metric])
        regularizer = self.ridge * np.eye(design.shape[1])
        regularizer[0, 0] = 0.0
        inverse = np.linalg.inv(design.T @ design + regularizer) @ design.T
        n_steps = curves.shape[1]
        self.intercept_metric = np.empty((n_steps, 3), dtype=np.float64)
        self.operator_metric = np.empty((n_steps, 3, 3), dtype=np.float64)
        for step in range(n_steps):
            coefficients = inverse @ curves_metric[:, step]
            self.intercept_metric[step] = coefficients[0]
            self.operator_metric[step] = coefficients[1:]
        return self

    def predict(self, contexts):
        contexts_metric = to_metric(contexts)
        prediction = np.empty(
            (len(contexts_metric), len(self.intercept_metric), 3), dtype=np.float64
        )
        for step in range(len(self.intercept_metric)):
            prediction[:, step] = (
                self.intercept_metric[step]
                + contexts_metric @ self.operator_metric[step]
            )
        return from_metric(prediction)

    def jacobian_diag(self):
        return np.diagonal(self.operator_metric, axis1=1, axis2=2).copy()


@dataclass
class ConditionalGaussian:
    mean: np.ndarray
    covariance: np.ndarray


def _normal_logpdf_scalar(value, mean, variance):
    return -0.5 * (
        np.log(2.0 * np.pi * variance) + (value - mean) ** 2 / variance
    )


def gmr_moment(model: GaussianMixture, query_time: float):
    component_means = []
    component_covariances = []
    log_weights = []
    for component in range(model.n_components):
        mean = model.means_[component]
        covariance = model.covariances_[component]
        variance_time = max(float(covariance[0, 0]), 1e-12)
        cross = covariance[1:, 0]
        conditional_mean = mean[1:] + cross / variance_time * (
            query_time - mean[0]
        )
        conditional_covariance = (
            covariance[1:, 1:]
            - np.outer(cross, cross) / variance_time
        )
        conditional_covariance = 0.5 * (
            conditional_covariance + conditional_covariance.T
        ) + 1e-9 * np.eye(3)
        component_means.append(conditional_mean)
        component_covariances.append(conditional_covariance)
        log_weights.append(
            np.log(model.weights_[component] + 1e-300)
            + _normal_logpdf_scalar(query_time, mean[0], variance_time)
        )
    weights = np.exp(np.asarray(log_weights) - logsumexp(log_weights))
    component_means = np.asarray(component_means)
    mixture_mean = np.sum(weights[:, None] * component_means, axis=0)
    mixture_covariance = np.zeros((3, 3), dtype=np.float64)
    for weight, mean, covariance in zip(
        weights, component_means, component_covariances
    ):
        delta = mean - mixture_mean
        mixture_covariance += weight * (covariance + np.outer(delta, delta))
    mixture_covariance += 1e-8 * np.eye(3)
    return ConditionalGaussian(mixture_mean, mixture_covariance)


class SharedFrameProductMixture:
    """Mixture with shared responsibilities and frame-specific emissions.

    The fitted density is

        p(k) p(t | k) product_f p(y_f | t, k),

    which is the same frame-product model used when TP-GMM predictions fuse
    the world and task-local Gaussian experts. No world/local cross-covariance
    is fitted and then discarded.
    """

    objective = "shared-responsibility frame-product"

    def __init__(
        self,
        n_components,
        reg_covar=1e-7,
        n_init=1,
        max_iter=300,
        tol=1e-5,
        random_state=0,
    ):
        self.n_components = int(n_components)
        self.reg_covar = float(reg_covar)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)

    @staticmethod
    def _gaussian_logpdf(values, means, covariance):
        values = np.asarray(values, dtype=np.float64)
        means = np.asarray(means, dtype=np.float64)
        covariance = np.asarray(covariance, dtype=np.float64)
        cholesky = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(cholesky, (values - means).T).T
        dimension = values.shape[1]
        log_determinant = 2.0 * np.log(np.diag(cholesky)).sum()
        return -0.5 * (
            dimension * np.log(2.0 * np.pi)
            + log_determinant
            + np.sum(whitened * whitened, axis=1)
        )

    def _estimate_parameters(self, time, frame_outputs, responsibilities):
        sample_count, frame_count, output_dimension = frame_outputs.shape
        effective_count = responsibilities.sum(axis=0) + 10.0 * np.finfo(float).eps
        weights = effective_count / effective_count.sum()
        time_means = np.einsum("nk,n->k", responsibilities, time) / effective_count
        time_delta = time[:, None] - time_means[None, :]
        time_variances = (
            np.einsum("nk,nk->k", responsibilities, time_delta * time_delta)
            / effective_count
        )
        time_variances = np.maximum(time_variances, self.reg_covar)

        design = np.column_stack([np.ones(sample_count), time])
        coefficients = np.empty(
            (self.n_components, frame_count, 2, output_dimension),
            dtype=np.float64,
        )
        covariances = np.empty(
            (self.n_components, frame_count, output_dimension, output_dimension),
            dtype=np.float64,
        )
        ridge = 1e-10 * np.eye(2)
        for component in range(self.n_components):
            sample_weights = responsibilities[:, component]
            normal = design.T @ (sample_weights[:, None] * design) + ridge
            for frame in range(frame_count):
                right_hand_side = design.T @ (
                    sample_weights[:, None] * frame_outputs[:, frame]
                )
                coefficients[component, frame] = np.linalg.solve(
                    normal, right_hand_side
                )
                residual = (
                    frame_outputs[:, frame]
                    - design @ coefficients[component, frame]
                )
                covariance = np.einsum(
                    "n,ni,nj->ij", sample_weights, residual, residual
                ) / effective_count[component]
                covariances[component, frame] = (
                    0.5 * (covariance + covariance.T)
                    + self.reg_covar * np.eye(output_dimension)
                )
        return weights, time_means, time_variances, coefficients, covariances

    def _estimate_weighted_log_prob(
        self,
        time,
        frame_outputs,
        weights,
        time_means,
        time_variances,
        coefficients,
        covariances,
    ):
        sample_count = len(time)
        log_probabilities = np.empty(
            (sample_count, self.n_components), dtype=np.float64
        )
        design = np.column_stack([np.ones(sample_count), time])
        for component in range(self.n_components):
            log_probability = (
                np.log(weights[component] + 1e-300)
                - 0.5
                * (
                    np.log(2.0 * np.pi * time_variances[component])
                    + (time - time_means[component]) ** 2
                    / time_variances[component]
                )
            )
            for frame in range(frame_outputs.shape[1]):
                means = design @ coefficients[component, frame]
                log_probability += self._gaussian_logpdf(
                    frame_outputs[:, frame], means, covariances[component, frame]
                )
            log_probabilities[:, component] = log_probability
        return log_probabilities

    def _initial_responsibilities(self, time, frame_outputs, seed):
        samples = np.column_stack([time, frame_outputs.reshape(len(time), -1)])
        initializer = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            reg_covar=self.reg_covar,
            n_init=1,
            max_iter=300,
            random_state=seed,
        ).fit(samples)
        return initializer.predict_proba(samples)

    def fit(self, time, frame_outputs):
        # These are many tiny 2x2/3x3 solves. Multithreaded BLAS is slower and
        # can otherwise occupy every host core during grouped CV.
        with threadpool_limits(limits=1, user_api="blas"):
            return self._fit_single_threaded(time, frame_outputs)

    def _fit_single_threaded(self, time, frame_outputs):
        time = np.asarray(time, dtype=np.float64)
        frame_outputs = np.asarray(frame_outputs, dtype=np.float64)
        if frame_outputs.ndim != 3 or frame_outputs.shape[0] != len(time):
            raise ValueError("frame_outputs must have shape [samples, frames, dims]")
        best = None
        for initialization in range(self.n_init):
            responsibilities = self._initial_responsibilities(
                time, frame_outputs, self.random_state + initialization
            )
            previous_lower_bound = -np.inf
            converged = False
            for iteration in range(1, self.max_iter + 1):
                parameters = self._estimate_parameters(
                    time, frame_outputs, responsibilities
                )
                log_probabilities = self._estimate_weighted_log_prob(
                    time, frame_outputs, *parameters
                )
                log_normalizer = logsumexp(log_probabilities, axis=1)
                lower_bound = float(log_normalizer.mean())
                responsibilities = np.exp(
                    log_probabilities - log_normalizer[:, None]
                )
                if abs(lower_bound - previous_lower_bound) < self.tol:
                    converged = True
                    break
                previous_lower_bound = lower_bound
            candidate = (lower_bound, converged, iteration, parameters)
            if best is None or candidate[0] > best[0]:
                best = candidate

        self.lower_bound_, self.converged_, self.n_iter_, parameters = best
        (
            self.weights_,
            self.time_means_,
            self.time_variances_,
            self.emission_coefficients_,
            self.emission_covariances_,
        ) = parameters
        return self

    def score(self, time, frame_outputs):
        log_probabilities = self._estimate_weighted_log_prob(
            np.asarray(time, dtype=np.float64),
            np.asarray(frame_outputs, dtype=np.float64),
            self.weights_,
            self.time_means_,
            self.time_variances_,
            self.emission_coefficients_,
            self.emission_covariances_,
        )
        return float(logsumexp(log_probabilities, axis=1).mean())


class TPGMMModel:
    def __init__(
        self,
        frame_mode="additive",
        nominal_frame_xy=None,
        nominal_frame_yaw=0.0,
        component_candidates=(4, 6, 8, 10, 12, 16, 20),
        time_scale=0.05,
        cv_splits=5,
        cv_n_init=5,
        final_n_init=10,
    ):
        if frame_mode not in {"additive", "se2"}:
            raise ValueError("frame_mode must be 'additive' or 'se2'")
        if frame_mode == "se2" and nominal_frame_xy is None:
            raise ValueError("nominal_frame_xy is required for SE(2) task frames")
        self.frame_mode = frame_mode
        self.name = "TP-GMM additive" if frame_mode == "additive" else "TP-GMM SE(2)"
        self.nominal_frame_xy = (
            None
            if nominal_frame_xy is None
            else np.asarray(nominal_frame_xy, dtype=np.float64)
        )
        self.nominal_frame_yaw = float(nominal_frame_yaw)
        self.component_candidates = tuple(component_candidates)
        self.time_scale = time_scale
        self.cv_splits = cv_splits
        self.cv_n_init = cv_n_init
        self.final_n_init = final_n_init

    def _fit_gmm(self, time, frame_outputs, components, random_state, n_init):
        return SharedFrameProductMixture(
            n_components=components,
            reg_covar=1e-7,
            n_init=n_init,
            max_iter=300,
            tol=1e-5,
            random_state=random_state,
        ).fit(time, frame_outputs)

    def _curve_to_local_metric(self, curve, context):
        if self.frame_mode == "additive":
            return to_metric(curve) - to_metric(context)[None, :]
        frame = socket_frame(
            context, self.nominal_frame_xy, self.nominal_frame_yaw
        )
        frame_inverse = np.linalg.inv(frame)
        local = np.asarray(
            [matrix_pose2(frame_inverse @ pose2_matrix(pose)) for pose in curve]
        )
        return to_metric(local)

    def _local_to_world_gaussian(self, local, context):
        if self.frame_mode == "additive":
            return ConditionalGaussian(
                local.mean + to_metric(context), local.covariance
            )
        frame_xy = self.nominal_frame_xy + np.asarray(context[:2])
        frame_yaw = self.nominal_frame_yaw + float(context[2])
        cosine = np.cos(frame_yaw)
        sine = np.sin(frame_yaw)
        linear = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        offset = np.array(
            [frame_xy[0], frame_xy[1], ROTATION_SCALE_M_PER_RAD * frame_yaw]
        )
        return ConditionalGaussian(
            linear @ local.mean + offset,
            linear @ local.covariance @ linear.T,
        )

    def fit(self, contexts, curves, progress, phase_codes):
        curves_metric = to_metric(curves)
        n_episodes, n_steps, _ = curves_metric.shape
        time = np.tile(np.asarray(progress) * self.time_scale, n_episodes)
        world_output = curves_metric.reshape(-1, 3)
        local_output = np.asarray(
            [
                self._curve_to_local_metric(curves[episode], contexts[episode])
                for episode in range(n_episodes)
            ]
        ).reshape(-1, 3)
        frame_outputs = np.stack([world_output, local_output], axis=1)
        groups = np.repeat(np.arange(n_episodes), n_steps)

        grouped_cv = GroupKFold(n_splits=self.cv_splits)
        self.cv_log_likelihood_by_components = {}
        for components in self.component_candidates:
            fold_scores = []
            for fold, (train_indices, validation_indices) in enumerate(
                grouped_cv.split(frame_outputs, groups=groups)
            ):
                model = self._fit_gmm(
                    time[train_indices],
                    frame_outputs[train_indices],
                    components,
                    random_state=20260820 + fold,
                    n_init=self.cv_n_init,
                )
                fold_scores.append(
                    model.score(
                        time[validation_indices], frame_outputs[validation_indices]
                    )
                )
            self.cv_log_likelihood_by_components[int(components)] = fold_scores
        self.n_components = max(
            self.component_candidates,
            key=lambda components: np.mean(
                self.cv_log_likelihood_by_components[int(components)]
            ),
        )
        self.model = self._fit_gmm(
            time,
            frame_outputs,
            self.n_components,
            random_state=20260820,
            n_init=self.final_n_init,
        )
        self.progress = np.asarray(progress, dtype=np.float64)
        self.step_components = []
        for step, progress_value in enumerate(self.progress):
            query_time = progress_value * self.time_scale
            component_conditionals = []
            log_activations = []
            for component in range(self.n_components):
                time_design = np.array([1.0, query_time])
                world = ConditionalGaussian(
                    time_design
                    @ self.model.emission_coefficients_[component, 0],
                    self.model.emission_covariances_[component, 0],
                )
                local = ConditionalGaussian(
                    time_design
                    @ self.model.emission_coefficients_[component, 1],
                    self.model.emission_covariances_[component, 1],
                )
                component_conditionals.append((world, local))
                variance_time = max(
                    float(self.model.time_variances_[component]), 1e-12
                )
                log_activations.append(
                    np.log(self.model.weights_[component] + 1e-300)
                    + _normal_logpdf_scalar(
                        query_time,
                        self.model.time_means_[component],
                        variance_time,
                    )
                )
            activations = np.exp(
                np.asarray(log_activations) - logsumexp(log_activations)
            )
            self.step_components.append((activations, component_conditionals))
        return self

    def predict(self, contexts):
        contexts = np.asarray(contexts, dtype=np.float64)
        prediction = np.empty(
            (len(contexts), len(self.progress), 3), dtype=np.float64
        )
        for episode, context in enumerate(contexts):
            for step, (activations, components) in enumerate(self.step_components):
                component_means = []
                for world, local in components:
                    local_world = self._local_to_world_gaussian(local, context)
                    precision_world = np.linalg.inv(world.covariance)
                    precision_local = np.linalg.inv(local_world.covariance)
                    fused_covariance = np.linalg.inv(
                        precision_world + precision_local
                    )
                    component_means.append(
                        fused_covariance
                        @ (
                            precision_world @ world.mean
                            + precision_local @ local_world.mean
                        )
                    )
                prediction[episode, step] = np.einsum(
                    "k,kd->d", activations, np.asarray(component_means)
                )
        return from_metric(prediction)

    def jacobian_diag(self):
        zero = np.zeros((1, 3), dtype=np.float64)
        epsilon = np.array([1e-4, 1e-4, np.deg2rad(0.1)])
        diagonal = np.empty((len(self.progress), 3), dtype=np.float64)
        for generator in range(3):
            plus = zero.copy()
            minus = zero.copy()
            plus[0, generator] = epsilon[generator]
            minus[0, generator] = -epsilon[generator]
            derivative = (
                self.predict(plus)[0] - self.predict(minus)[0]
            ) / (2.0 * epsilon[generator])
            diagonal[:, generator] = derivative[:, generator]
        return diagonal


class GenericConditionalRBF:
    name = "Generic RBF"

    def __init__(self, centers_per_phase=10, width=0.16):
        self.centers_per_phase = centers_per_phase
        self.width = width
        self.context_scale_metric = np.array(
            [0.012, 0.012, ROTATION_SCALE_M_PER_RAD * np.deg2rad(30.0)]
        )

    def _basis(self, progress, phase_codes):
        progress = np.asarray(progress, dtype=np.float64)
        phase_codes = np.asarray(phase_codes, dtype=int)
        local_progress = progress * 4.0 - phase_codes
        centers = np.linspace(0.0, 1.0, self.centers_per_phase)
        local_basis = np.exp(
            -0.5
            * ((local_progress[:, None] - centers[None, :]) / self.width) ** 2
        )
        local_basis /= local_basis.sum(axis=1, keepdims=True)
        basis = np.zeros(
            (len(progress), 4 * self.centers_per_phase), dtype=np.float64
        )
        for row, phase in enumerate(phase_codes):
            start = phase * self.centers_per_phase
            basis[row, start : start + self.centers_per_phase] = local_basis[row]
        return basis

    def _design(self, contexts, basis):
        contexts_normalized = to_metric(contexts) / self.context_scale_metric
        repeated_basis = np.tile(basis, (len(contexts), 1))
        repeated_context = np.repeat(
            contexts_normalized, len(basis), axis=0
        )
        interactions = [
            repeated_basis * repeated_context[:, generator, None]
            for generator in range(3)
        ]
        return np.column_stack([repeated_basis, *interactions])

    def fit(self, contexts, curves, progress, phase_codes):
        self.progress = np.asarray(progress, dtype=np.float64)
        self.phase_codes = np.asarray(phase_codes, dtype=int)
        self.basis = self._basis(self.progress, self.phase_codes)
        design = self._design(contexts, self.basis)
        outputs = to_metric(curves).reshape(-1, 3)
        groups = np.repeat(np.arange(len(contexts)), len(self.progress))
        grouped_cv = list(GroupKFold(n_splits=5).split(design, outputs, groups))
        self.model = RidgeCV(
            alphas=np.logspace(-10, 0, 16),
            fit_intercept=False,
            cv=grouped_cv,
        ).fit(design, outputs)
        self.alpha = float(self.model.alpha_)
        return self

    def predict(self, contexts):
        design = self._design(contexts, self.basis)
        prediction = self.model.predict(design).reshape(
            len(contexts), len(self.progress), 3
        )
        return from_metric(prediction)

    def jacobian_diag(self):
        coefficients = np.asarray(self.model.coef_).T
        basis_count = self.basis.shape[1]
        diagonal = np.empty((len(self.progress), 3), dtype=np.float64)
        for generator in range(3):
            start = basis_count * (generator + 1)
            stop = start + basis_count
            derivative_metric_per_normalized_context = (
                self.basis @ coefficients[start:stop]
            )
            diagonal[:, generator] = (
                derivative_metric_per_normalized_context[:, generator]
                / self.context_scale_metric[generator]
            )
        return diagonal
