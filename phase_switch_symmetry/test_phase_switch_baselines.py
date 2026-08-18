from __future__ import annotations

import unittest

import numpy as np

from phase_switch_symmetry.phase_switch_baselines import (
    DiagonalOperatorModel,
    FrameWeightedModel,
    FullOperatorModel,
    SmoothFinitePDiagModel,
    TPGMMModel,
    gmr_moment,
    matrix_pose2,
    pose2_matrix,
    se2_exp,
    se2_log,
)
from phase_switch_symmetry.benchmark_phase_switch_baselines import switch_diagnostics


class FakeGaussianMixture:
    n_components = 1
    weights_ = np.array([1.0])
    means_ = np.array([[0.2, 1.0, -2.0, 0.5]])
    covariances_ = np.array(
        [
            [
                [0.25, 0.10, -0.05, 0.0],
                [0.10, 1.00, 0.20, 0.0],
                [-0.05, 0.20, 2.00, 0.10],
                [0.0, 0.0, 0.10, 0.50],
            ]
        ]
    )


class PhaseSwitchBaselineTests(unittest.TestCase):
    def test_se2_exp_log_and_pose_roundtrip(self):
        twist = np.array([0.013, -0.009, 0.47])
        np.testing.assert_allclose(se2_log(se2_exp(twist)), twist, atol=1e-12)
        pose = np.array([0.12, -0.04, -0.7])
        np.testing.assert_allclose(matrix_pose2(pose2_matrix(pose)), pose, atol=1e-12)

    def test_switch_diagnostics_requires_a_downward_crossing(self):
        progress = np.linspace(0.0, 1.0, 10)
        detected = switch_diagnostics(
            progress,
            np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.1, 0.0, 0.0]),
            unlock_start=4,
        )
        censored = switch_diagnostics(
            progress,
            np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.7, 0.65, 0.6, 0.58, 0.56]),
            unlock_start=4,
        )
        self.assertTrue(detected["detected"])
        self.assertEqual(detected["status"], "detected")
        self.assertAlmostEqual(detected["location"], 5.5 / 9.0)
        self.assertFalse(censored["detected"])
        self.assertEqual(censored["status"], "right_censored")

    def test_structured_models_recover_known_response(self):
        rng = np.random.default_rng(17)
        contexts = np.column_stack(
            [
                rng.uniform(-0.02, 0.02, 40),
                rng.uniform(-0.02, 0.02, 40),
                rng.uniform(-0.5, 0.5, 40),
            ]
        )
        progress = np.linspace(0.0, 1.0, 20)
        phase_codes = np.minimum((progress * 4).astype(int), 3)
        diagonal = np.column_stack(
            [
                np.ones_like(progress),
                0.8 * np.ones_like(progress),
                1.0 / (1.0 + np.exp(40.0 * (progress - 0.6))),
            ]
        )
        intercept = np.column_stack(
            [0.1 + 0.01 * progress, -0.2 * np.ones_like(progress), 0.3 * progress]
        )
        curves = intercept[None, :, :] + contexts[:, None, :] * diagonal[None, :, :]

        pdiag = DiagonalOperatorModel().fit(contexts, curves, progress, phase_codes)
        full = FullOperatorModel().fit(contexts, curves, progress, phase_codes)
        frame = FrameWeightedModel().fit(contexts, curves, progress, phase_codes)

        np.testing.assert_allclose(pdiag.jacobian_diag(), diagonal, atol=1e-11)
        np.testing.assert_allclose(full.jacobian_diag(), diagonal, atol=5e-6)
        np.testing.assert_allclose(pdiag.predict(contexts), curves, atol=1e-12)
        np.testing.assert_allclose(full.predict(contexts), curves, atol=3e-6)
        frame_profile = frame.jacobian_diag()
        np.testing.assert_allclose(frame_profile[:, 0], frame_profile[:, 1])
        np.testing.assert_allclose(frame_profile[:, 1], frame_profile[:, 2])

    def test_gmr_single_component_matches_conditional_gaussian(self):
        model = FakeGaussianMixture()
        query = 0.5
        result = gmr_moment(model, query)
        covariance = model.covariances_[0]
        expected_mean = model.means_[0, 1:] + covariance[1:, 0] / covariance[0, 0] * (
            query - model.means_[0, 0]
        )
        expected_covariance = (
            covariance[1:, 1:]
            - np.outer(covariance[1:, 0], covariance[1:, 0]) / covariance[0, 0]
            + 1.1e-8 * np.eye(3)
        )
        np.testing.assert_allclose(result.mean, expected_mean, atol=1e-12)
        np.testing.assert_allclose(result.covariance, expected_covariance, atol=1e-12)

    def test_tp_gmm_uses_shared_responsibilities_and_frame_emissions(self):
        rng = np.random.default_rng(23)
        contexts = rng.normal(scale=[0.01, 0.01, 0.2], size=(10, 3))
        progress = np.linspace(0.0, 1.0, 12)
        phase_codes = np.minimum((progress * 4).astype(int), 3)
        gain = np.column_stack(
            [
                np.ones_like(progress),
                np.ones_like(progress),
                (progress < 0.6).astype(float),
            ]
        )
        intercept = np.column_stack(
            [0.1 * progress, -0.1 * progress, 0.2 * progress]
        )
        curves = intercept[None] + contexts[:, None] * gain[None]
        model = TPGMMModel(
            component_candidates=(2,),
            cv_splits=2,
            cv_n_init=1,
            final_n_init=2,
        ).fit(contexts, curves, progress, phase_codes)

        self.assertEqual(model.model.emission_coefficients_.shape, (2, 2, 2, 3))
        self.assertEqual(model.model.emission_covariances_.shape, (2, 2, 3, 3))
        self.assertEqual(model.model.time_means_.shape, (2,))
        self.assertEqual(model.model.weights_.shape, (2,))
        self.assertEqual(model.model.objective, "shared-responsibility frame-product")
        self.assertTrue(np.isfinite(model.model.lower_bound_))
        self.assertEqual(model.predict(contexts).shape, curves.shape)
        self.assertTrue(np.isfinite(model.predict(contexts)).all())
        self.assertTrue(np.isfinite(model.jacobian_diag()).all())

    def test_rotation_aware_task_frame_roundtrip(self):
        model = TPGMMModel(
            frame_mode="se2", nominal_frame_xy=np.array([0.1, -0.02])
        )
        context = np.array([0.012, -0.007, 0.4])
        curve = np.array([[0.13, -0.01, 0.6], [0.115, -0.025, -0.2]])
        local_metric = model._curve_to_local_metric(curve, context)
        for expected, local_mean in zip(curve, local_metric):
            local = type("Gaussian", (), {
                "mean": local_mean,
                "covariance": np.eye(3) * 1e-4,
            })()
            world = model._local_to_world_gaussian(local, context)
            np.testing.assert_allclose(
                world.mean,
                expected * np.array([1.0, 1.0, 0.03]),
                atol=1e-12,
            )

    def test_smooth_finite_pdiag_recovers_finite_action_profile(self):
        rng = np.random.default_rng(29)
        contexts = rng.uniform(
            [-0.015, -0.015, -0.45], [0.015, 0.015, 0.45], size=(24, 3)
        )
        progress = np.linspace(0.0, 1.0, 24)
        phase_codes = np.minimum((progress * 4).astype(int), 3)
        generator = SmoothFinitePDiagModel(
            nominal_frame_xy=np.array([0.1, 0.0]),
            n_basis=8,
            basis_width=0.14,
            smoothness_weight=1e-3,
            nominal_iterations=2,
        )
        generator.progress = progress
        generator.basis = generator._basis(progress)
        parameters = np.vstack(
            [
                np.full(8, 1.4),
                np.full(8, 1.4),
                np.linspace(3.0, -5.0, 8),
            ]
        )
        alpha = generator._alpha(parameters.reshape(-1))
        nominal = np.column_stack(
            [
                0.1 + 0.002 * np.sin(2 * np.pi * progress),
                0.003 * np.cos(2 * np.pi * progress),
                0.1 * progress,
            ]
        )
        curves = generator._predict_with(contexts, nominal, alpha)
        fitted = SmoothFinitePDiagModel(
            nominal_frame_xy=np.array([0.1, 0.0]),
            n_basis=8,
            basis_width=0.14,
            smoothness_weight=1e-3,
            nominal_iterations=2,
        ).fit(contexts, curves, progress, phase_codes)

        self.assertTrue(fitted.optimization_success)
        self.assertLess(np.sqrt(np.mean((fitted.diagonal - alpha) ** 2)), 0.08)
        self.assertLess(
            np.sqrt(np.mean((fitted.predict(contexts) - curves) ** 2)), 5e-4
        )

    def test_batched_finite_action_matches_matrix_formula(self):
        model = SmoothFinitePDiagModel(
            nominal_frame_xy=np.array([0.1, -0.03]),
            nominal_frame_yaw=0.2,
            n_basis=6,
        )
        contexts = np.array(
            [[0.01, -0.005, 0.35], [-0.012, 0.007, -0.28]]
        )
        nominal = np.array(
            [[0.11, -0.02, 0.1], [0.09, -0.04, -0.3], [0.105, -0.025, 0.5]]
        )
        alpha = np.array(
            [[0.9, 1.0, 0.8], [1.0, 0.7, 0.4], [0.95, 0.85, 0.1]]
        )
        batched = model._predict_with(contexts, nominal, alpha)
        matrix_result = np.empty_like(batched)
        for episode, context in enumerate(contexts):
            twist = model._context_twist(context)
            for step, pose in enumerate(nominal):
                matrix_result[episode, step] = matrix_pose2(
                    model._selected_action(twist, alpha[step]) @ pose2_matrix(pose)
                )
        np.testing.assert_allclose(batched, matrix_result, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
