import unittest

import numpy as np

from qmcpy import DigitalNetB2, Gaussian, ImportanceSampling, Lebesgue, Uniform
from qmcpy.util import DimensionError, ParameterError


class TestImportanceSampling(unittest.TestCase):

    def test_sampling_delegates_to_proposal(self):
        proposal = Uniform(
            DigitalNetB2(2, seed=7),
            lower_bound=[-1, 2],
            upper_bound=[2, 6],
        )
        target = Uniform(
            proposal.discrete_distrib,
            lower_bound=[-0.5, 2.5],
            upper_bound=[1, 5],
        )
        importance_sampling = ImportanceSampling(
            target=target,
            proposal=proposal,
        )
        x = np.array([[0.1, 0.25], [0.5, 0.75], [0.9, 0.4]])

        samples = importance_sampling._jacobian_transform_r(
            x,
            return_weights=False,
        )
        samples_with_weights, jacobians = (
            importance_sampling._jacobian_transform_r(
                x,
                return_weights=True,
            )
        )
        expected_samples, expected_jacobians = proposal._jacobian_transform_r(
            x,
            return_weights=True,
        )

        np.testing.assert_allclose(samples, expected_samples)
        np.testing.assert_allclose(samples_with_weights, expected_samples)
        np.testing.assert_allclose(jacobians, expected_jacobians)

    def test_importance_weights_match_old_integrand_formula(self):
        proposal = Uniform(
            DigitalNetB2(2, seed=7),
            lower_bound=[-1, 2],
            upper_bound=[2, 6],
        )
        target = Uniform(
            proposal.discrete_distrib,
            lower_bound=[-0.5, 2.5],
            upper_bound=[1, 5],
        )
        importance_sampling = ImportanceSampling(
            target=target,
            proposal=proposal,
        )
        x = np.array([[0.1, 0.25], [0.5, 0.75], [0.9, 0.4]])

        samples, importance_weights = (
            importance_sampling._importance_sampling_transform_r(x)
        )
        expected_samples, proposal_jacobians = (
            proposal._jacobian_transform_r(x, return_weights=True)
        )
        expected_weights = (
            target._weight(expected_samples)
            * proposal_jacobians
            / proposal.discrete_distrib.pdf(x)
        )

        np.testing.assert_allclose(samples, expected_samples)
        np.testing.assert_allclose(importance_weights, expected_weights)

    def test_public_sample_weight_semantics_and_replications(self):
        for replications in (None, 2):
            with self.subTest(replications=replications):
                proposal = Uniform(
                    DigitalNetB2(1, seed=7, replications=replications)
                )
                target = Uniform(
                    proposal.discrete_distrib,
                    lower_bound=0.25,
                    upper_bound=0.75,
                )
                importance_sampling = ImportanceSampling(
                    target=target,
                    proposal=proposal,
                )
                x = proposal.discrete_distrib.gen_samples(4)
                expected_samples, expected_weights = (
                    importance_sampling._importance_sampling_transform_r(x)
                )

                samples = importance_sampling.gen_samples(4)
                weighted_samples, weights = importance_sampling.gen_samples(
                    4,
                    return_weights=True,
                )

                np.testing.assert_allclose(samples, expected_samples)
                np.testing.assert_allclose(weighted_samples, expected_samples)
                np.testing.assert_allclose(weights, expected_weights)
                expected_sample_shape = (
                    (4, 1) if replications is None else (replications, 4, 1)
                )
                expected_weight_shape = (
                    (4,) if replications is None else (replications, 4)
                )
                self.assertEqual(samples.shape, expected_sample_shape)
                self.assertEqual(weights.shape, expected_weight_shape)
                self.assertTrue(np.isfinite(samples).all())
                self.assertTrue(np.isfinite(weights).all())

        with self.assertRaises(AssertionError):
            importance_sampling.gen_samples(4, return_weights=1)

    def test_spawn_preserves_standalone_target_and_proposal(self):
        proposal = Gaussian(
            DigitalNetB2(1, seed=7),
            mean=1,
            covariance=2,
        )
        target = Gaussian(
            proposal.discrete_distrib,
            mean=0,
            covariance=1 / 2,
        )
        importance_sampling = ImportanceSampling(
            target=target,
            proposal=proposal,
        )

        spawned = importance_sampling.spawn(s=2, dimensions=[1, 2])

        self.assertEqual([measure.d for measure in spawned], [1, 2])
        for measure in spawned:
            self.assertIsInstance(measure.target, Gaussian)
            self.assertIsInstance(measure.proposal, Gaussian)
            np.testing.assert_array_equal(measure.target.mu, np.zeros(measure.d))
            np.testing.assert_array_equal(
                measure.target.covariance,
                np.eye(measure.d) / 2,
            )
            np.testing.assert_array_equal(measure.proposal.mu, np.ones(measure.d))
            np.testing.assert_array_equal(
                measure.proposal.covariance,
                np.eye(measure.d) * 2,
            )
            expected_range = np.array([[-np.inf, np.inf]])
            np.testing.assert_array_equal(
                measure.target.range,
                expected_range,
            )
            np.testing.assert_array_equal(
                measure.proposal.range,
                expected_range,
            )
            np.testing.assert_array_equal(measure.domain, measure.proposal.domain)
            np.testing.assert_array_equal(measure.range, measure.proposal.range)
            x = np.tile(np.array([[0.25], [0.5], [0.75]]), (1, measure.d))
            samples, weights = measure._importance_sampling_transform_r(x)
            self.assertEqual(samples.shape, (3, measure.d))
            self.assertEqual(weights.shape, (3,))
            self.assertTrue(np.isfinite(samples).all())
            self.assertTrue(np.isfinite(weights).all())

    def test_spawn_preserves_wrapped_target(self):
        proposal = Uniform(
            DigitalNetB2(1, seed=7),
            lower_bound=-2,
            upper_bound=2,
        )
        target = Lebesgue(proposal)
        importance_sampling = ImportanceSampling(
            target=target,
            proposal=proposal,
        )

        spawned = importance_sampling.spawn(s=1, dimensions=2)[0]

        self.assertEqual(spawned.d, 2)
        self.assertIsInstance(spawned.target, Lebesgue)
        self.assertIsInstance(spawned.proposal, Uniform)
        self.assertIsInstance(spawned.target.transform, Uniform)
        expected_range = np.array([[-2, 2], [-2, 2]])
        np.testing.assert_array_equal(spawned.target.range, expected_range)
        np.testing.assert_array_equal(
            spawned.target.transform.range,
            expected_range,
        )
        np.testing.assert_array_equal(spawned.proposal.a, [-2, -2])
        np.testing.assert_array_equal(spawned.proposal.b, [2, 2])
        np.testing.assert_array_equal(spawned.proposal.range, expected_range)
        x = np.tile(np.array([[0.25], [0.5], [0.75]]), (1, spawned.d))
        samples, weights = spawned._importance_sampling_transform_r(x)
        self.assertEqual(samples.shape, (3, 2))
        self.assertEqual(weights.shape, (3,))
        self.assertTrue(np.isfinite(samples).all())
        self.assertTrue(np.isfinite(weights).all())

    def test_target_support_must_be_contained_in_proposal_support(self):
        proposal = Uniform(
            DigitalNetB2(1, seed=7),
            lower_bound=0.25,
            upper_bound=0.75,
        )
        wider_target = Uniform(
            proposal.discrete_distrib,
            lower_bound=0,
            upper_bound=1,
        )

        with self.assertRaisesRegex(
            ParameterError,
            "target range must be contained within proposal range",
        ):
            ImportanceSampling(
                target=wider_target,
                proposal=proposal,
            )

        contained_target = Uniform(
            proposal.discrete_distrib,
            lower_bound=0.3,
            upper_bound=0.7,
        )
        importance_sampling = ImportanceSampling(
            target=contained_target,
            proposal=proposal,
        )
        self.assertIs(importance_sampling.target, contained_target)
        self.assertIs(importance_sampling.proposal, proposal)

    def test_constructor_and_input_dimension_validation(self):
        one_dimensional = Uniform(DigitalNetB2(1, seed=7))
        two_dimensional = Uniform(DigitalNetB2(2, seed=7))

        with self.assertRaises(ParameterError):
            ImportanceSampling(target=object(), proposal=one_dimensional)
        with self.assertRaises(ParameterError):
            ImportanceSampling(target=one_dimensional, proposal=object())
        with self.assertRaises(DimensionError):
            ImportanceSampling(
                target=one_dimensional,
                proposal=two_dimensional,
            )

        importance_sampling = ImportanceSampling(
            target=one_dimensional,
            proposal=one_dimensional,
        )
        with self.assertRaises(DimensionError):
            importance_sampling._importance_sampling_transform_r(
                np.ones((4, 2))
            )

    def test_importance_sampling_is_terminal(self):
        proposal = Uniform(DigitalNetB2(1, seed=7))
        target = Uniform(
            proposal.discrete_distrib,
            lower_bound=0.25,
            upper_bound=0.75,
        )
        importance_sampling = ImportanceSampling(
            target=target,
            proposal=proposal,
        )

        for outer_measure in (Uniform, Gaussian):
            with self.subTest(outer_measure=outer_measure.__name__):
                with self.assertRaisesRegex(
                    ParameterError,
                    "ImportanceSampling cannot be used as a sampler for another TrueMeasure",
                ):
                    outer_measure(importance_sampling)

        with self.assertRaisesRegex(
            ParameterError,
            "ImportanceSampling cannot be the target of another ImportanceSampling",
        ):
            ImportanceSampling(
                target=importance_sampling,
                proposal=proposal,
            )

        with self.assertRaisesRegex(
            ParameterError,
            "ImportanceSampling cannot be the proposal of another ImportanceSampling",
        ):
            ImportanceSampling(
                target=target,
                proposal=importance_sampling,
            )


if __name__ == "__main__":
    unittest.main()
