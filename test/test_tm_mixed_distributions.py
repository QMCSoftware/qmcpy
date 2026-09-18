from unittest import TestCase

import warnings

import numpy as np
import pytest
from scipy.stats import uniform

from qmcpy import (
    AbstractTrueMeasure,
    CubBayesLatticeG,
    CubBayesNetG,
    CubMCCLT,
    CubMCCLTVec,
    CubMCG,
    CubQMCLatticeG,
    CubQMCNetG,
    CubQMCRepStudentT,
    CustomFun,
    DigitalNetB2,
    DummySampler,
    Gaussian,
    IIDStdUniform,
    Kumaraswamy,
    Lattice,
    Lebesgue,
    Mixture,
    ProductMeasure,
    SciPyWrapper,
    SensitivityIndices,
    Uniform,
)
from qmcpy.util import DimensionError, MaxSamplesWarning, MethodImplementationError, ParameterError


class TransformOnlyMeasure(AbstractTrueMeasure):
    def __init__(self, sampler):
        self.parameters = []
        self.domain = np.array([[0.0, 1.0]])
        self.range = np.array([[0.0, 1.0]])
        self._parse_sampler(sampler)
        super(TransformOnlyMeasure, self).__init__()

    def _transform(self, x):
        return x


def gaussian_component(dimension, mean):
    return Gaussian(
        DigitalNetB2(dimension, seed=17),
        mean=mean,
        covariance=np.eye(dimension),
    )


class TestMixture(TestCase):
    def test_two_component_gaussian_mixture_shape(self):
        components = [gaussian_component(1, -2.0), gaussian_component(1, 3.0)]
        mixture = Mixture(DigitalNetB2(2, seed=7), components, [0.3, 0.7])

        samples = mixture(16)

        assert mixture.d == 1
        assert mixture.discrete_distrib.d == 2
        assert samples.shape == (16, 1)

    def test_selection_boundaries_preserve_order(self):
        components = [gaussian_component(1, -2.0), gaussian_component(1, 3.0)]
        mixture = Mixture(DigitalNetB2(2, seed=7), components, [0.3, 0.7])
        just_below_boundary = np.nextafter(0.3, 0.0)
        just_above_boundary = np.nextafter(0.3, 1.0)
        u = np.array(
            [
                [0.0, 0.5],
                [0.9, 0.5],
                [just_below_boundary, 0.5],
                [0.3, 0.5],
                [just_above_boundary, 0.5],
                [0.1, 0.5],
                [1.0, 0.5],
            ]
        )

        samples = mixture._transform(u)

        np.testing.assert_allclose(
            samples[:, 0], [-2.0, 3.0, -2.0, 3.0, 3.0, -2.0, 3.0]
        )

    def test_rejects_composed(self):
        inner = Kumaraswamy(DigitalNetB2(1, seed=19), a=2.0, b=1.0)
        for composed in (Uniform(inner), Gaussian(inner, mean=-2.0)):
            product = ProductMeasure(DigitalNetB2(1, seed=11), [composed])
            nested = ProductMeasure(DigitalNetB2(1, seed=13), [product])
            for component in (composed, product, nested):
                with self.subTest(component=type(component).__name__):
                    with self.assertRaisesRegex(ParameterError, "direct components"):
                        Mixture(DigitalNetB2(2, seed=7), [component], [1.0])

    def test_product_weights(self):
        product = ProductMeasure(
            DigitalNetB2(2, seed=11),
            [Uniform(DigitalNetB2(1, seed=13), 0, 2),
             Uniform(DigitalNetB2(1, seed=17), 0, 3)],
        )
        mixture = Mixture(DigitalNetB2(3, seed=7), [product], [1.0])
        u = np.array([[.25, .25, .5], [.75, .5, .25]])
        samples, weights = mixture._jacobian_transform_r(u, return_weights=True)
        np.testing.assert_allclose(samples, [[.5, 1.5], [1., .75]])
        np.testing.assert_allclose(weights, [6., 6.])

    def test_multiple_components(self):
        components = [
            gaussian_component(1, -4.0),
            gaussian_component(1, 0.0),
            gaussian_component(1, 5.0),
        ]
        mixture = Mixture(DigitalNetB2(2, seed=7), components, [0.2, 0.3, 0.5])
        u = np.array([[0.2, 0.5], [0.4, 0.5], [0.5, 0.5], [0.8, 0.5]])

        samples = mixture._transform(u)

        np.testing.assert_allclose(samples[:, 0], [0.0, 0.0, 5.0, 5.0])

    def test_heterogeneous_component_transforms(self):
        gaussian = gaussian_component(1, -2.0)
        wrapped = SciPyWrapper(DigitalNetB2(1, seed=19), uniform(loc=2.0, scale=3.0))
        mixture = Mixture(DigitalNetB2(2, seed=7), [gaussian, wrapped], [0.3, 0.7])
        u = np.array([[0.8, 0.2], [0.1, 0.7], [0.6, 0.9], [0.2, 0.4]])

        expected = np.empty((4, 1))
        expected[[1, 3]] = gaussian._transform(u[[1, 3], 1:])
        expected[[0, 2]] = wrapped._transform(u[[0, 2], 1:])

        np.testing.assert_allclose(mixture._transform(u), expected)

    def test_range_is_coordinate_wise_bounding_box(self):
        components = [
            Uniform(DigitalNetB2(2, seed=17), lower_bound=[-3, 2], upper_bound=[1, 4]),
            Uniform(DigitalNetB2(2, seed=19), lower_bound=[-1, -2], upper_bound=[5, 3]),
        ]
        mixture = Mixture(DigitalNetB2(3, seed=7), components, [0.3, 0.7])

        np.testing.assert_array_equal(mixture.range, [[-3, 5], [-2, 4]])

    def test_range_expands_shared_component_bounds(self):
        components = [
            TransformOnlyMeasure(DigitalNetB2(2, seed=17)),
            Uniform(DigitalNetB2(2, seed=19), lower_bound=[-2, 0.25], upper_bound=[-1, 2]),
        ]
        mixture = Mixture(DigitalNetB2(3, seed=7), components, [0.3, 0.7])

        np.testing.assert_array_equal(mixture.range, [[-2, 1], [0, 2]])

    def test_malformed_custom_component_range_is_rejected(self):
        class MalformedRangeMeasure(TransformOnlyMeasure):
            def __init__(self, sampler):
                super(MalformedRangeMeasure, self).__init__(sampler)
                self.range = np.zeros((3, 2))

        component = MalformedRangeMeasure(DigitalNetB2(2, seed=17))

        with pytest.raises(DimensionError, match="component range must have shape"):
            Mixture(DigitalNetB2(3, seed=7), [component], [1.0])

    def test_one_component_mixture_is_valid(self):
        component = gaussian_component(1, 1.5)
        mixture = Mixture(DigitalNetB2(2, seed=7), [component], [1.0])
        u = np.array([[0.0, 0.5], [0.4, 0.5], [1.0, 0.5]])

        samples = mixture._transform(u)

        assert mixture(4).shape == (4, 1)
        np.testing.assert_allclose(samples[:, 0], 1.5)

    def test_invalid_probabilities(self):
        for probabilities in (
            [0.0, 1.0], [-0.1, 1.1], [np.nan, np.nan], [np.inf, 0.5]
        ):
            with self.subTest(probabilities=probabilities):
                components = [gaussian_component(1, 0.0), gaussian_component(1, 1.0)]

                with pytest.raises(ParameterError, match="positive and finite"):
                    Mixture(DigitalNetB2(2, seed=7), components, probabilities)

    def test_probabilities_must_sum_to_one(self):
        components = [gaussian_component(1, 0.0), gaussian_component(1, 1.0)]

        with pytest.raises(ParameterError, match="sum to 1"):
            Mixture(DigitalNetB2(2, seed=7), components, [0.2, 0.7])

    def test_probabilities_are_owned_and_read_only(self):
        probabilities = np.array([0.3, 0.7])
        components = [gaussian_component(1, -2.0), gaussian_component(1, 3.0)]
        mixture = Mixture(DigitalNetB2(2, seed=7), components, probabilities)
        np.testing.assert_array_equal(mixture.probabilities, [0.3, 0.7])

        probabilities[:] = [0.8, 0.2]

        np.testing.assert_array_equal(mixture.probabilities, [0.3, 0.7])
        np.testing.assert_allclose(mixture._transform([[0.5, 0.5]]), [[3.0]])
        assert not mixture.probabilities.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            mixture.probabilities[0] = 0.8

    def test_number_of_probabilities_must_match_components(self):
        components = [gaussian_component(1, 0.0), gaussian_component(1, 1.0)]

        with pytest.raises(ParameterError, match="one probability per component"):
            Mixture(DigitalNetB2(2, seed=7), components, [1.0])

    def test_requires_at_least_one_component(self):
        with pytest.raises(ParameterError, match="nonempty list of components"):
            Mixture(DigitalNetB2(2, seed=7), [], [])

    def test_components_must_be_true_measures(self):
        components = [gaussian_component(1, 0.0), object()]

        with pytest.raises(ParameterError, match="AbstractTrueMeasure"):
            Mixture(DigitalNetB2(2, seed=7), components, [0.5, 0.5])

    def test_sampler_must_be_discrete_distribution(self):
        components = [gaussian_component(1, 0.0)]

        with pytest.raises(ParameterError, match="AbstractDiscreteDistribution"):
            Mixture(object(), components, [1.0])

    def test_probabilities_must_be_numeric(self):
        components = [gaussian_component(1, 0.0), gaussian_component(1, 1.0)]

        with pytest.raises(ParameterError, match="numeric"):
            Mixture(DigitalNetB2(2, seed=7), components, ["left", "right"])

    def test_probabilities_must_be_one_dimensional(self):
        components = [gaussian_component(1, 0.0), gaussian_component(1, 1.0)]

        with pytest.raises(ParameterError, match="one probability per component"):
            Mixture(DigitalNetB2(2, seed=7), components, [[0.5, 0.5]])

    def test_component_dimensions_must_match(self):
        components = [gaussian_component(1, 0.0), gaussian_component(2, [0.0, 1.0])]

        with pytest.raises(DimensionError, match="same output dimension"):
            Mixture(DigitalNetB2(2, seed=7), components, [0.5, 0.5])

    def test_sampler_requires_extra_dimension(self):
        components = [gaussian_component(1, 0.0), gaussian_component(1, 1.0)]

        with pytest.raises(DimensionError, match="component dimension plus one"):
            Mixture(DigitalNetB2(1, seed=7), components, [0.5, 0.5])

    def test_transform_input_dimension_is_validated(self):
        mixture = Mixture(
            DigitalNetB2(2, seed=7),
            [gaussian_component(1, 0.0)],
            [1.0],
        )

        with pytest.raises(DimensionError, match="expected last axis 2"):
            mixture._transform(np.zeros((3, 1)))

    def test_weight_input_dimension_is_validated(self):
        mixture = Mixture(
            DigitalNetB2(2, seed=7),
            [gaussian_component(1, 0.0)],
            [1.0],
        )

        with pytest.raises(DimensionError, match="expected last axis 1"):
            mixture._weight(np.zeros((3, 2)))

    def test_weight_is_weighted_sum_of_component_weights(self):
        components = [gaussian_component(1, -1.0), gaussian_component(1, 2.0)]
        probabilities = np.array([0.3, 0.7])
        mixture = Mixture(DigitalNetB2(2, seed=7), components, probabilities)
        x = np.array([[-2.0], [0.0], [1.5], [4.0]])

        expected = sum(
            probability * component._weight(x)
            for probability, component in zip(probabilities, components)
        )

        np.testing.assert_allclose(mixture._weight(x), expected)

    def test_weight_outside_disjoint_supports(self):
        components = [
            Uniform(DigitalNetB2(1, seed=17), lower_bound=0.0, upper_bound=1.0),
            Uniform(DigitalNetB2(1, seed=19), lower_bound=2.0, upper_bound=3.0),
        ]
        mixture = Mixture(DigitalNetB2(2, seed=7), components, [0.5, 0.5])

        np.testing.assert_allclose(
            mixture._weight(np.array([[0.5], [2.5], [1.5]])),
            [0.5, 0.5, 0.0],
        )

    def test_weight_sums_overlapping_densities(self):
        components = [
            Uniform(DigitalNetB2(1, seed=17), lower_bound=0.0, upper_bound=2.0),
            Uniform(DigitalNetB2(1, seed=19), lower_bound=1.0, upper_bound=3.0),
        ]
        mixture = Mixture(DigitalNetB2(2, seed=7), components, [0.25, 0.75])

        np.testing.assert_allclose(
            mixture._weight(np.array([[0.5], [1.5], [2.5], [3.5]])),
            [0.125, 0.5, 0.375, 0.0],
        )

    def test_univariate_mixture_moments(self):
        components = [
            Uniform(DigitalNetB2(1, seed=17), lower_bound=0.0, upper_bound=1.0),
            Uniform(DigitalNetB2(1, seed=19), lower_bound=2.0, upper_bound=3.0),
        ]
        mixture = Mixture(DigitalNetB2(2, seed=7), components, [0.3, 0.7])
        expected_variance = 68.0 / 15.0 - 1.9**2

        assert mixture.mean == pytest.approx(1.9)
        assert mixture.variance == pytest.approx(expected_variance)
        assert mixture.standard_deviation == pytest.approx(np.sqrt(expected_variance))
        np.testing.assert_allclose(mixture.covariance, [[expected_variance]])
        assert all(
            statistic in mixture.parameters
            for statistic in ("mean", "variance", "standard_deviation", "covariance")
        )

    def test_multivariate_mixture_moments_are_read_only(self):
        components = [
            Gaussian(
                DigitalNetB2(2, seed=17),
                mean=[0.0, 1.0],
                covariance=[[1.0, 0.2], [0.2, 2.0]],
            ),
            Gaussian(
                DigitalNetB2(2, seed=19),
                mean=[2.0, -1.0],
                covariance=[[0.5, -0.1], [-0.1, 1.5]],
            ),
        ]
        mixture = Mixture(DigitalNetB2(3, seed=7), components, [0.25, 0.75])

        np.testing.assert_allclose(mixture.mean, [1.5, -0.5])
        np.testing.assert_allclose(
            mixture.covariance, [[1.375, -0.775], [-0.775, 2.375]]
        )
        np.testing.assert_allclose(mixture.variance, [1.375, 2.375])
        np.testing.assert_allclose(
            mixture.standard_deviation, np.sqrt([1.375, 2.375])
        )
        for statistic in (
            mixture.mean,
            mixture.variance,
            mixture.standard_deviation,
            mixture.covariance,
        ):
            assert not statistic.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                statistic.flat[0] = 0.0

    def test_missing_moments_error(self):
        components = [
            gaussian_component(1, 0.0),
            TransformOnlyMeasure(DigitalNetB2(1, seed=23)),
        ]
        mixture = Mixture(DigitalNetB2(2, seed=7), components, [0.5, 0.5])

        assert "mean" not in mixture.parameters
        with pytest.raises(AttributeError, match="component 1.*does not provide mean"):
            _ = mixture.mean

    def test_public_sampling_with_return_weights(self):
        components = [gaussian_component(1, -1.0), gaussian_component(1, 2.0)]
        mixture = Mixture(DigitalNetB2(2, seed=7), components, [0.3, 0.7])

        samples, weights = mixture(16, return_weights=True)

        assert samples.shape == (16, 1)
        assert weights.shape == (16,)
        assert np.all(np.isfinite(samples))
        assert np.all(np.isfinite(weights))
        assert np.all(weights > 0)
        np.testing.assert_allclose(weights, 1.0 / mixture._weight(samples))

    def test_weight_preserves_leading_axes(self):
        components = [gaussian_component(2, [-1.0, 0.0]), gaussian_component(2, [2.0, 1.0])]
        probabilities = np.array([0.3, 0.7])
        mixture = Mixture(DigitalNetB2(3, seed=7), components, probabilities)
        x = np.linspace(-2.0, 3.0, 12).reshape(2, 3, 2)

        expected = sum(
            probability * component._weight(x)
            for probability, component in zip(probabilities, components)
        )
        weights = mixture._weight(x)

        assert weights.shape == (2, 3)
        np.testing.assert_allclose(weights, expected)

    def test_component_weight_failure_propagates(self):
        transform_only = TransformOnlyMeasure(DigitalNetB2(1, seed=23))
        mixture = Mixture(
            DigitalNetB2(2, seed=7),
            [gaussian_component(1, 0.0), transform_only],
            [0.5, 0.5],
        )

        with pytest.raises(MethodImplementationError, match="TransformOnlyMeasure"):
            mixture._weight(np.array([[0.5]]))

    def test_spawn_sampler_and_components(self):
        components = [gaussian_component(1, -2.0), gaussian_component(1, 3.0)]
        mixture = Mixture(DigitalNetB2(2, seed=7), components, [0.3, 0.7])

        spawned = mixture.spawn(s=2)
        explicit_same_dimension = mixture.spawn(s=1, dimensions=[1])[0]

        for child in spawned + [explicit_same_dimension]:
            assert isinstance(child, Mixture)
            assert child.d == mixture.d == 1
            np.testing.assert_array_equal(child.probabilities, mixture.probabilities)
            np.testing.assert_array_equal(child.range, mixture.range)
            assert not child.probabilities.flags.writeable
            assert child.discrete_distrib.d == 2
            assert child.discrete_distrib is not mixture.discrete_distrib
            assert all(
                child_component is parent_component
                for child_component, parent_component in zip(child.components, components)
            )
            assert child(4).shape == (4, 1)

        with pytest.raises(DimensionError, match="preserves the component dimension"):
            mixture.spawn(s=1, dimensions=2)

    def test_spawn_validates_count_and_dimensions_length(self):
        mixture = Mixture(
            DigitalNetB2(2, seed=7),
            [gaussian_component(1, 0.0)],
            [1.0],
        )

        with pytest.raises(ParameterError, match="s>0"):
            mixture.spawn(s=0)
        with pytest.raises(ParameterError, match="length s"):
            mixture.spawn(s=2, dimensions=[1])

    def test_replicated_sampler_shape_and_selection(self):
        components = [gaussian_component(1, -2.0), gaussian_component(1, 3.0)]
        mixture = Mixture(
            DigitalNetB2(2, seed=7, replications=3), components, [0.3, 0.7]
        )

        samples = mixture(8)
        manual_u = np.array(
            [
                [[0.1, 0.5], [0.9, 0.5]],
                [[0.3, 0.5], [np.nextafter(0.3, 1.0), 0.5]],
            ]
        )
        manual_samples = mixture._transform(manual_u)

        assert samples.shape == (3, 8, 1)
        assert manual_samples.shape == (2, 2, 1)
        np.testing.assert_allclose(manual_samples[..., 0], [[-2.0, 3.0], [3.0, 3.0]])


class TestMixtureIntegration(TestCase):
    """Preserve output coordinates while integrating over all driver coordinates."""

    @staticmethod
    def _mixture(sampler):
        d = sampler.d - 1
        return Mixture(
            sampler,
            [Uniform(DummySampler(d), 0, 1),
             Uniform(DummySampler(d), 1, 2)],
            [.25, .75],
        )

    @staticmethod
    def _square(t):
        return t[..., 0] ** 2

    @staticmethod
    def _moments(t):
        return np.stack([t[..., 0], t[..., 0] ** 2])

    def test_shapes(self):
        for reps in (None, 1, 3):
            for vector in (False, True):
                with self.subTest(replications=reps, vector=vector):
                    m = self._mixture(DigitalNetB2(2, seed=7, replications=reps))
                    g = CustomFun(
                        m, self._moments if vector else self._square,
                        dimension_indv=(2,) if vector else (),
                    )
                    u = m.discrete_distrib(16)
                    # Each component is an affine uniform transform. This
                    # expected value is independent of Mixture._transform.
                    t = u[..., 1] + (u[..., 0] >= .25)
                    expected = np.stack([t, t**2]) if vector else t**2
                    y = g(16)
                    self.assertEqual(y.shape, expected.shape)
                    np.testing.assert_allclose(y, expected, rtol=0, atol=0)
                    self.assertTrue(np.isfinite(y).all())
                    self.assertEqual((g.d, g.discrete_distrib.d), (1, 2))

    def test_importance(self):
        for reps in (None, 1, 3):
            with self.subTest(replications=reps):
                m = self._mixture(DigitalNetB2(2, seed=7, replications=reps))
                # The two intervals cover the complete target support [0, 2].
                g = CustomFun(Lebesgue(m), lambda t: t[..., 0])
                u = m.discrete_distrib(1024)
                t = u[..., 1] + (u[..., 0] >= .25)
                density = np.where(t < 1, .25, .75)
                y = g.f(u)
                np.testing.assert_allclose(y, t / density)
                np.testing.assert_allclose(y.mean(-1), 2, rtol=0, atol=.01)
                self.assertEqual(y.shape, u.shape[:-1])
                self.assertTrue(np.isfinite(y).all())

    def test_driver_points(self):
        cases = (
            (CubMCCLT, IIDStdUniform, None, False),
            (CubMCG, IIDStdUniform, None, False),
            (CubMCCLTVec, IIDStdUniform, None, True),
            (CubQMCNetG, DigitalNetB2, None, False),
            (CubQMCLatticeG, Lattice, None, False),
            (CubBayesNetG, DigitalNetB2, None, False),
            (CubBayesLatticeG, Lattice, None, False),
            (CubQMCRepStudentT, DigitalNetB2, 4, True),
        )
        for solver, sampler, reps, vector in cases:
            with self.subTest(solver=solver.__name__):
                m = self._mixture(sampler(2, seed=7, replications=reps))
                g = CustomFun(
                    m, self._moments if vector else self._square,
                    dimension_indv=(2,) if vector else (),
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", MaxSamplesWarning)
                    solution, data = solver(
                        g, abs_tol=.25, n_init=256, n_limit=1024
                    ).integrate()
                self.assertEqual(data.xfull.shape[-1], 2)
                self.assertEqual(data.xfull.ndim, 2 if reps is None else 3)
                self.assertEqual(data.xfull.size // 2, int(data.n_total))
                self.assertTrue(np.isfinite(data.xfull).all())
                self.assertTrue(np.isfinite(solution).all())
                expected = [1.25, 11 / 6] if vector else 11 / 6
                np.testing.assert_allclose(solution, expected, rtol=0, atol=.25)

    def test_resume_points(self):
        cases = (
            (CubMCCLTVec, IIDStdUniform, None),
            (CubQMCNetG, DigitalNetB2, None),
            (CubQMCLatticeG, Lattice, None),
            (CubBayesNetG, DigitalNetB2, None),
            (CubBayesLatticeG, Lattice, None),
            (CubQMCRepStudentT, DigitalNetB2, 4),
        )
        for solver, sampler, reps in cases:
            with self.subTest(solver=solver.__name__):
                limit = 256 * (reps or 1)

                def make_solver(n_limit):
                    m = self._mixture(sampler(2, seed=7, replications=reps))
                    return solver(
                        CustomFun(m, self._square),
                        abs_tol=1e-12, n_init=256, n_limit=n_limit,
                    )

                with warnings.catch_warnings():
                    # The tiny tolerance intentionally exhausts each budget.
                    warnings.simplefilter("ignore", MaxSamplesWarning)
                    sc = make_solver(limit)
                    _, checkpoint = sc.integrate()
                    old_x = checkpoint.xfull.copy()
                    old_n = int(checkpoint.n_total)
                    sc.n_limit = 2 * limit
                    resumed_solution, resumed = sc.integrate(resume=checkpoint)
                    fresh_solution, fresh = make_solver(2 * limit).integrate()
                self.assertGreater(int(resumed.n_total), old_n)
                self.assertEqual(resumed.xfull.shape[-1], 2)
                np.testing.assert_array_equal(checkpoint.xfull, old_x)
                np.testing.assert_array_equal(
                    resumed.xfull[..., :old_x.shape[-2], :], old_x,
                )
                np.testing.assert_array_equal(resumed.xfull, fresh.xfull)
                np.testing.assert_allclose(
                    resumed_solution, fresh_solution, rtol=1e-12, atol=1e-12,
                )
                self.assertTrue(np.isfinite(resumed_solution).all())

    def test_spawn_dims(self):
        for importance in (False, True):
            with self.subTest(importance=importance):
                m = self._mixture(DigitalNetB2(2, seed=7))
                g = CustomFun(Lebesgue(m) if importance else m, self._square)
                children = g.spawn([0, 0])
                for child in children:
                    self.assertEqual((child.d, child.discrete_distrib.d), (1, 2))
                    self.assertIsNot(child.discrete_distrib, g.discrete_distrib)
                    self.assertEqual(child(16).shape, (16,))
                    self.assertTrue(np.isfinite(child(16)).all())
                self.assertIsNot(children[0].discrete_distrib, children[1].discrete_distrib)

    def test_rejects_nesting(self):
        m = self._mixture(DigitalNetB2(2, seed=7))
        with self.assertRaises(DimensionError):
            Mixture(DigitalNetB2(2, seed=11), [m], [1.])
        with self.assertRaises(DimensionError):
            ProductMeasure(
                DigitalNetB2(2, seed=11), [m, Uniform(DummySampler(1))],
            )
        m2 = self._mixture(DigitalNetB2(3, seed=7))
        with self.assertRaises(DimensionError):
            SensitivityIndices(CustomFun(m2, self._square))

    def test_gp_dims(self):
        try:
            import gpytorch
            import torch
            from qmcpy import PFGPCI, SuggesterSimple
        except ModuleNotFoundError as error:
            self.skipTest(f"Optional GP dependencies unavailable: {error}")
        for use_init_samples in (False, True):
            with self.subTest(use_init_samples=use_init_samples):
                torch.manual_seed(17)
                components = [
                    Uniform(DigitalNetB2(1, seed=11)),
                    Uniform(DigitalNetB2(1, seed=13), lower_bound=2, upper_bound=3),
                ]
                measure = Mixture(DigitalNetB2(2, seed=7), components, [0.5, 0.5])
                integrand = CustomFun(measure, lambda t: t[..., 0])
                x_init = DigitalNetB2(2, seed=19)(8)
                init_samples = (
                    (x_init, integrand.f(x_init)) if use_init_samples else None
                )
                criterion = PFGPCI(
                    integrand,
                    failure_threshold=1.5,
                    failure_above_threshold=True,
                    abs_tol=0,
                    n_init=8,
                    n_limit=12,
                    n_batch=4,
                    n_approx=32,
                    n_ref_approx=32,
                    seed_ref_approx=23,
                    init_samples=init_samples,
                    batch_sampler=SuggesterSimple(DigitalNetB2(2, seed=29)),
                    gpytorch_prior_mean=gpytorch.means.ZeroMean(),
                    gpytorch_prior_cov=gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.MaternKernel(nu=2.5)
                    ),
                    gpytorch_likelihood=gpytorch.likelihoods.GaussianLikelihood(),
                    gpytorch_train_iter=1,
                    verbose=False,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", MaxSamplesWarning)
                    solution, data = criterion.integrate(seed=31)
                self.assertEqual(integrand.d, 1)
                self.assertEqual(criterion.d, 2)
                self.assertEqual(data.x.shape, (12, 2))
                self.assertEqual(data.qmc_pts.shape, (32, 2))
                self.assertEqual(data.n_total, 12)
                np.testing.assert_allclose(data.y, integrand.f(data.x) - 1.5)
                self.assertTrue(np.isfinite(solution))
                self.assertTrue(0 <= solution <= 1)
                self.assertTrue(np.isfinite(data.error_bound))
                self.assertEqual(criterion.ref_approx, 0.5)
                if use_init_samples:
                    np.testing.assert_array_equal(data.x[:8], x_init)
