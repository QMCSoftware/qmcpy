from qmcpy import (
    AbstractTrueMeasure,
    BernoulliCont,
    BrownianMotion,
    DigitalNetB2,
    Gaussian,
    GeometricBrownianMotion,
    IIDStdUniform,
    JohnsonsSU,
    Kumaraswamy,
    Lattice,
    Lebesgue,
    MaternGP,
    SciPyWrapper,
    Uniform,
    ZeroInflatedExpUniform,
)
from qmcpy.util import DimensionError, ParameterError
import numpy as np
import scipy.stats
from scipy.sparse import issparse
import unittest
import warnings
from qmcpy.true_measure.uniform_triangle import UniformTriangle, _UniformTriangleAdapter


def dense_covariance(covariance):
    return covariance.toarray() if issparse(covariance) else np.asarray(covariance)


def assert_sample_mean_and_covariance(measure):
    samples = measure.gen_samples(2**15)
    sample_mean = samples.mean(axis=0)
    centered_samples = samples - sample_mean
    sample_covariance = centered_samples.T @ centered_samples / len(samples)

    np.testing.assert_allclose(sample_mean, measure.mean, rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        sample_covariance, dense_covariance(measure.covariance), rtol=0, atol=1e-5
    )


class TestTrueMeasure(unittest.TestCase):
    """General tests for TrueMeasures"""

    def test_range_in_domain(self):
        cases = [
            ("exact equality", [[0, 1]], [[0, 1]], True),
            ("strict finite inclusion", [[0.25, 0.75]], [[0, 1]], True),
            ("infinite domain", [[-5, 5]], [[-np.inf, np.inf]], True),
            ("positive half-line", [[1, 3]], [[0, np.inf]], True),
            ("lower-bound failure", [[-0.1, 0.75]], [[0, 1]], False),
            ("upper-bound failure", [[0.25, 1.1]], [[0, 1]], False),
            (
                "multidimensional box",
                [[0.1, 0.8], [0.2, 0.9]],
                [[0, 1], [0, 1]],
                True,
            ),
            (
                "failure in one coordinate",
                [[0.1, 0.8], [0.2, 1.1]],
                [[0, 1], [0, 1]],
                False,
            ),
            (
                "broadcast domain",
                [[0.1, 0.8], [0.2, 0.9]],
                [[0, 1]],
                True,
            ),
            (
                "non-broadcastable rows",
                [[0.1, 0.8], [0.2, 0.9]],
                [[0, 1], [0, 1], [0, 1]],
                False,
            ),
        ]

        for name, transform_range, domain, expected in cases:
            with self.subTest(name=name):
                self.assertIs(
                    AbstractTrueMeasure._range_in_domain(transform_range, domain),
                    expected,
                )

    def test_range_in_domain_rejects_invalid_bounds(self):
        invalid_cases = [
            ("one-dimensional range", [0, 1], [[0, 1]]),
            ("three-column range", [[0, 0.5, 1]], [[0, 1]]),
            ("ragged range", [[0, 1], [0, 0.5, 1]], [[0, 1]]),
            ("one-dimensional domain", [[0, 1]], [0, 1]),
            ("three-column domain", [[0, 1]], [[0, 0.5, 1]]),
            (
                "reversed transform range",
                np.array([[0.8, 0.2]]),
                np.array([[0.0, 1.0]]),
            ),
            ("reversed domain", [[0, 1]], [[0.8, 0.2]]),
            (
                "reversed multidimensional transform range",
                [[0.1, 0.8], [0.9, 0.2]],
                [[0, 1], [0, 1]],
            ),
            (
                "reversed multidimensional domain",
                [[0.1, 0.8], [0.2, 0.9]],
                [[0, 1], [0.9, 0.2]],
            ),
            ("empty transform range", np.empty((0, 2)), [[0, 1]]),
            ("empty domain", [[0, 1]], np.empty((0, 2))),
            (
                "string bounds",
                np.array([["a", "z"]]),
                np.array([["a", "z"]]),
            ),
            (
                "object bounds",
                np.array([[0, 1]], dtype=object),
                np.array([[0, 1]], dtype=object),
            ),
            ("complex bounds", [[0 + 0j, 1 + 0j]], [[0 + 0j, 1 + 0j]]),
            ("boolean bounds", [[False, True]], [[False, True]]),
            ("NaN transform range", [[np.nan, 1]], [[0, 1]]),
            ("NaN domain", [[0, 1]], [[np.nan, 1]]),
        ]

        for name, transform_range, domain in invalid_cases:
            with self.subTest(name=name):
                self.assertIs(
                    AbstractTrueMeasure._range_in_domain(transform_range, domain),
                    False,
                )

    def test_strict_range_in_domain_chain(self):
        inner = Uniform(
            DigitalNetB2(1, seed=7), lower_bound=0.25, upper_bound=0.75
        )
        outer = Kumaraswamy(inner)

        self.assertFalse(outer.sub_compatibility_error)
        samples = outer.gen_samples(8)
        self.assertEqual(samples.shape, (8, 1))
        self.assertTrue(np.isfinite(samples).all())

    def test_multidimensional_range_in_domain_chain_broadcasts(self):
        inner = Uniform(
            DigitalNetB2(2, seed=7),
            lower_bound=[0.1, 0.2],
            upper_bound=[0.8, 0.9],
        )
        outer = Kumaraswamy(inner)

        self.assertFalse(outer.sub_compatibility_error)
        samples = outer.gen_samples(8)
        self.assertEqual(samples.shape, (8, 2))
        self.assertTrue(np.isfinite(samples).all())

    def test_out_of_domain_chain_preserves_deferred_errors(self):
        inner = Uniform(
            DigitalNetB2(1, seed=7), lower_bound=-0.1, upper_bound=0.75
        )
        incompatible = Kumaraswamy(inner)

        self.assertTrue(incompatible.sub_compatibility_error)
        with self.assertRaisesRegex(
            ParameterError,
            "The sub-transform range must be contained within the transform domain.",
        ):
            incompatible.gen_samples(8)

        with self.assertRaisesRegex(
            ParameterError,
            "The sub-sub-transform range must be contained within the sub-transform domain.",
        ):
            Kumaraswamy(incompatible)

    def test_abstract_methods(self):
        d = 2
        tms = [
            Uniform(DigitalNetB2(d, seed=7)),
            Uniform(DigitalNetB2(d, seed=7), lower_bound=[1, 2], upper_bound=[2, 3]),
            Kumaraswamy(DigitalNetB2(d, seed=7)),
            Kumaraswamy(DigitalNetB2(d, seed=7), a=[2, 4], b=[1, 3]),
            JohnsonsSU(DigitalNetB2(d, seed=7)),
            JohnsonsSU(
                DigitalNetB2(d, seed=7),
                gamma=[1, 2],
                xi=[4, 5],
                delta=[7, 8],
                lam=[10, 11],
            ),
            Gaussian(DigitalNetB2(d, seed=7)),
            Gaussian(
                DigitalNetB2(d, seed=7),
                mean=[1, 2],
                covariance=[[9, 5], [5, 9]],
                decomp_type="Cholesky",
            ),
            Gaussian(Kumaraswamy(Kumaraswamy(DigitalNetB2(d, seed=7)))),
            BrownianMotion(DigitalNetB2(d, seed=7)),
            BrownianMotion(
                DigitalNetB2(d, seed=7), t_final=2, drift=3, decomp_type="Cholesky"
            ),
            BrownianMotion(DigitalNetB2(d, seed=7), decomp_type="BrownianBridge"),
            BernoulliCont(DigitalNetB2(d, seed=7)),
            BernoulliCont(DigitalNetB2(d, seed=7), lam=[0.25, 0.75]),
            SciPyWrapper(
                DigitalNetB2(2, seed=7),
                [scipy.stats.triang(c=0.1), scipy.stats.uniform(loc=1, scale=2)],
            ),
            SciPyWrapper(
                DigitalNetB2(2, seed=7), scipy.stats.triang(c=0.1, loc=1, scale=2)
            ),
        ]
        for tm in tms:
            for _tm in [tm] + tm.spawn(1):
                t = _tm.gen_samples(4)
                self.assertEqual(t.shape, (4, 2))
                self.assertEqual(t.dtype, np.float64)
                x = _tm.discrete_distrib.gen_samples(4)
                xtf, jtf = _tm._jacobian_transform_r(x, return_weights=True)
                self.assertTrue(xtf.shape == (4, d), jtf.shape == (4,))
                w = _tm._weight(x)
                self.assertEqual(w.shape, (4,))
                s = str(_tm)

    def test_spawn(self):
        d = 3
        tms = [
            Uniform(DigitalNetB2(d, seed=7)),
            Lebesgue(Uniform(DigitalNetB2(d, seed=7))),
            Lebesgue(Gaussian(DigitalNetB2(d, seed=7))),
            Kumaraswamy(DigitalNetB2(d, seed=7)),
            JohnsonsSU(DigitalNetB2(d, seed=7)),
            Gaussian(DigitalNetB2(d, seed=7)),
            Gaussian(Kumaraswamy(Kumaraswamy(DigitalNetB2(d, seed=7)))),
            BrownianMotion(DigitalNetB2(d, seed=7)),
            BernoulliCont(DigitalNetB2(d, seed=7)),
            SciPyWrapper(
                DigitalNetB2(2, seed=7), scipy.stats.triang(c=0.1, loc=1, scale=2)
            ),
        ]
        for tm in tms:
            s = 3
            for spawn_dim in [4, [1, 4, 6]]:
                spawns = tm.spawn(s=s, dimensions=spawn_dim)
                self.assertEqual(len(spawns), s)
                self.assertTrue(all(type(spawn) == type(tm) for spawn in spawns))
                self.assertTrue(
                    (np.array([spawn.d for spawn in spawns]) == spawn_dim).all()
                )
                self.assertTrue(
                    (
                        np.array([spawn.transform.d for spawn in spawns]) == spawn_dim
                    ).all()
                )
                self.assertTrue(
                    (
                        np.array([spawn.transform.transform.d for spawn in spawns])
                        == spawn_dim
                    ).all()
                )
                self.assertTrue(
                    (
                        np.array([spawn.discrete_distrib.d for spawn in spawns])
                        == spawn_dim
                    ).all()
                )
                self.assertTrue(
                    (
                        all(
                            spawn.discrete_distrib != tm.discrete_distrib
                            for spawn in spawns
                        )
                    )
                )
                self.assertTrue(
                    all(spawn.transform != tm.transform for spawn in spawns)
                )

    def test_moment_attributes_are_public_and_consistent(self):
        measures = [
            Uniform(
                DigitalNetB2(2, seed=7),
                lower_bound=[-1, 2],
                upper_bound=[3, 8],
            ),
            Kumaraswamy(
                DigitalNetB2(2, seed=7), a=[1, 2], b=[3, 4]
            ),
            Gaussian(
                DigitalNetB2(2, seed=7),
                mean=[1, -1],
                covariance=[[4, 1], [1, 9]],
            ),
            BrownianMotion(
                DigitalNetB2(2, seed=7),
                t_final=2,
                diffusion=3,
            ),
        ]
        moment_parameters = [
            "mean",
            "variance",
            "standard_deviation",
            "covariance",
        ]

        for measure in measures:
            with self.subTest(measure=type(measure).__name__):
                for parameter in moment_parameters:
                    self.assertIn(parameter, measure.parameters)
                    self.assertIn(parameter, str(measure))
                self.assertEqual(measure.mean.shape, (measure.d,))
                self.assertEqual(measure.variance.shape, (measure.d,))
                self.assertEqual(
                    measure.standard_deviation.shape, (measure.d,)
                )
                self.assertEqual(
                    measure.covariance.shape, (measure.d, measure.d)
                )
                np.testing.assert_allclose(
                    measure.standard_deviation**2, measure.variance
                )
                np.testing.assert_allclose(
                    np.diag(dense_covariance(measure.covariance)), measure.variance
                )

    def test_moment_attributes_are_read_only(self):
        measures = [
            Uniform(DigitalNetB2(2, seed=7)),
            Kumaraswamy(DigitalNetB2(2, seed=7)),
            Gaussian(DigitalNetB2(2, seed=7), covariance=np.eye(2)),
            BrownianMotion(DigitalNetB2(2, seed=7)),
        ]

        for measure in measures:
            with self.subTest(measure=type(measure).__name__):
                for parameter in (
                    "mean",
                    "variance",
                    "standard_deviation",
                    "covariance",
                ):
                    value = getattr(measure, parameter)
                    if issparse(value):
                        # Diagonal covariances are stored sparsely; their
                        # backing data must still be read only.
                        self.assertFalse(value.data.flags.writeable)
                        with self.assertRaises(ValueError):
                            value.data[0] = 9
                    else:
                        self.assertFalse(value.flags.writeable)
                        with self.assertRaises(ValueError):
                            value.flat[0] = 9
                        with self.assertRaises(ValueError):
                            value.setflags(write=True)
                    with self.assertRaises(AttributeError):
                        setattr(measure, parameter, np.zeros_like(value))

    def test_diagonal_covariance_is_sparse(self):
        d = 500
        for measure in (
            Uniform(IIDStdUniform(d, seed=7)),
            Kumaraswamy(IIDStdUniform(d, seed=7)),
        ):
            with self.subTest(measure=type(measure).__name__):
                covariance = measure.covariance
                self.assertTrue(issparse(covariance))
                self.assertEqual(covariance.format, "dia")
                self.assertEqual(covariance.shape, (d, d))
                # Only the diagonal is stored: O(d), not O(d^2).
                self.assertEqual(covariance.data.size, d)
                np.testing.assert_allclose(
                    covariance.diagonal(), measure.variance
                )
                # Off-diagonal entries are exactly zero.
                dense = covariance.toarray()
                np.testing.assert_array_equal(
                    dense - np.diag(np.diag(dense)), np.zeros((d, d))
                )


class TestMatern(unittest.TestCase):
    def test_spawn(self):
        points = np.linspace(0, 1, 3)[:, None]
        matern = MaternGP(
            IIDStdUniform(3, seed=7),
            points=points,
            variance=0.01,
            nugget=0.002,
        )

        direct_spawn = matern._spawn(IIDStdUniform(3, seed=8))
        public_spawn = matern.spawn(1)[0]

        for spawned in (direct_spawn, public_spawn):
            self.assertIsInstance(spawned, MaternGP)
            np.testing.assert_array_equal(spawned.points, points)
            np.testing.assert_allclose(spawned.mean, matern.mean)
            np.testing.assert_allclose(spawned.covariance, matern.covariance)

        with self.assertRaises(DimensionError):
            matern.spawn(1, dimensions=4)

    def test_sklearn_equivalence(self):
        points = np.array([[5, 4], [1, 2], [0, 0]])
        mean = np.full(3, 1.1)

        m2 = MaternGP(
            Lattice(dimension=3, seed=7),
            points,
            length_scale=4,
            nu=2.5,
            variance=0.01,
            mean=mean,
            nugget=1e-6,
        )
        from sklearn import gaussian_process as gp  # checking against scikit's Matern

        kernel2 = gp.kernels.Matern(length_scale=4, nu=2.5)
        cov2 = 0.01 * kernel2.__call__(points) + 1e-6 * np.eye(m2.covariance.shape[-1])
        assert np.allclose(cov2, m2.covariance)


class TestUniform(unittest.TestCase):
    def test_sample_mean_and_covariance(self):
        uniform = Uniform(
            DigitalNetB2(2, seed=7),
            lower_bound=[-2, 1],
            upper_bound=[4, 10],
        )

        assert_sample_mean_and_covariance(uniform)

    def test_upper_bound_must_exceed_lower_bound(self):
        for lower_bound, upper_bound in [([1], [0]), ([1], [1])]:
            with self.subTest(
                lower_bound=lower_bound, upper_bound=upper_bound
            ):
                with self.assertRaisesRegex(
                    ParameterError,
                    "upper bound must be strictly greater than lower bound",
                ):
                    Uniform(
                        IIDStdUniform(1, seed=7),
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                    )

    def test_bounds_must_be_finite(self):
        for lower_bound, upper_bound in [
            (np.nan, 1),
            (0, np.nan),
            (-np.inf, 1),
            (0, np.inf),
        ]:
            with self.subTest(
                lower_bound=lower_bound, upper_bound=upper_bound
            ):
                with self.assertRaisesRegex(
                    ParameterError,
                    "upper bound and lower bound must be finite",
                ):
                    Uniform(
                        IIDStdUniform(1, seed=7),
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                    )

    def test_moment_attributes_with_scalar_bounds(self):
        uniform = Uniform(
            DigitalNetB2(3, seed=7), lower_bound=-2, upper_bound=4
        )

        np.testing.assert_allclose(uniform.mean, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(uniform.variance, [3.0, 3.0, 3.0])
        np.testing.assert_allclose(
            uniform.standard_deviation, np.sqrt([3.0, 3.0, 3.0])
        )
        np.testing.assert_allclose(
            dense_covariance(uniform.covariance),
            np.diag([3.0, 3.0, 3.0]),
        )

    def test_moment_attributes_with_vector_bounds(self):
        uniform = Uniform(
            DigitalNetB2(2, seed=7),
            lower_bound=[-2, 1],
            upper_bound=[4, 10],
        )

        np.testing.assert_allclose(uniform.mean, [1.0, 5.5])
        np.testing.assert_allclose(uniform.variance, [3.0, 6.75])
        np.testing.assert_allclose(
            uniform.standard_deviation, np.sqrt([3.0, 6.75])
        )
        np.testing.assert_allclose(
            dense_covariance(uniform.covariance),
            np.diag([3.0, 6.75]),
        )

    def test_spawn_recomputes_moment_attributes(self):
        uniform = Uniform(
            DigitalNetB2(2, seed=7), lower_bound=-2, upper_bound=4
        )
        spawn = uniform.spawn(1, dimensions=4)[0]

        np.testing.assert_allclose(spawn.mean, np.full(4, 1.0))
        np.testing.assert_allclose(spawn.variance, np.full(4, 3.0))
        np.testing.assert_allclose(
            spawn.standard_deviation, np.full(4, np.sqrt(3.0))
        )
        np.testing.assert_allclose(dense_covariance(spawn.covariance), 3.0 * np.eye(4))


class TestKumaraswamy(unittest.TestCase):
    def test_sample_mean_and_covariance(self):
        kumaraswamy = Kumaraswamy(
            DigitalNetB2(2, seed=7), a=[1, 2], b=[3, 4]
        )

        assert_sample_mean_and_covariance(kumaraswamy)

    def test_moment_attributes_with_scalar_parameters(self):
        kumaraswamy = Kumaraswamy(DigitalNetB2(3, seed=7), a=1, b=3)
        expected_mean = np.full(3, 0.25)
        expected_variance = np.full(3, 0.0375)

        np.testing.assert_allclose(kumaraswamy.mean, expected_mean)
        np.testing.assert_allclose(kumaraswamy.variance, expected_variance)
        np.testing.assert_allclose(
            kumaraswamy.standard_deviation, np.sqrt(expected_variance)
        )
        np.testing.assert_allclose(
            dense_covariance(kumaraswamy.covariance), np.diag(expected_variance)
        )

    def test_moment_attributes_with_vector_parameters(self):
        kumaraswamy = Kumaraswamy(
            DigitalNetB2(2, seed=7), a=[1, 2], b=[3, 4]
        )
        expected_mean = np.array([0.25, 128 / 315])
        expected_variance = np.array(
            [0.0375, 0.2 - (128 / 315) ** 2]
        )

        np.testing.assert_allclose(kumaraswamy.mean, expected_mean)
        np.testing.assert_allclose(kumaraswamy.variance, expected_variance)
        np.testing.assert_allclose(
            kumaraswamy.standard_deviation, np.sqrt(expected_variance)
        )
        np.testing.assert_allclose(
            dense_covariance(kumaraswamy.covariance), np.diag(expected_variance)
        )

    def test_uniform_special_case(self):
        kumaraswamy = Kumaraswamy(
            DigitalNetB2(2, seed=7), a=1, b=1
        )
        expected_variance = np.full(2, 1 / 12)

        np.testing.assert_allclose(kumaraswamy.mean, np.full(2, 0.5))
        np.testing.assert_allclose(kumaraswamy.variance, expected_variance)
        np.testing.assert_allclose(
            kumaraswamy.standard_deviation, np.sqrt(expected_variance)
        )
        np.testing.assert_allclose(
            dense_covariance(kumaraswamy.covariance), np.diag(expected_variance)
        )

    def test_spawn_recomputes_moment_attributes(self):
        kumaraswamy = Kumaraswamy(
            DigitalNetB2(2, seed=7), a=1, b=3
        )
        spawn = kumaraswamy.spawn(1, dimensions=4)[0]
        expected_variance = np.full(4, 0.0375)

        np.testing.assert_allclose(spawn.mean, np.full(4, 0.25))
        np.testing.assert_allclose(spawn.variance, expected_variance)
        np.testing.assert_allclose(
            spawn.standard_deviation, np.sqrt(expected_variance)
        )
        np.testing.assert_allclose(
            dense_covariance(spawn.covariance), np.diag(expected_variance)
        )

    def test_variance_shape(self):
        # Univariate (d==1) measures return scalar moments; multivariate
        # measures return length-d arrays.
        for d, a, b in [(1, 2, 3), (2, [1, 2], [3, 4]), (4, 3, 5)]:
            with self.subTest(d=d):
                kumaraswamy = Kumaraswamy(DigitalNetB2(d, seed=7), a=a, b=b)
                variance = kumaraswamy.variance

                if d == 1:
                    self.assertIsInstance(variance, float)
                    self.assertEqual(np.ndim(variance), 0)
                else:
                    self.assertEqual(variance.shape, (d,))
                    self.assertEqual(variance.ndim, 1)
                self.assertTrue(np.all(variance > 0))

    def test_covariance_is_d_by_d_matrix(self):
        for d, a, b in [(1, 2, 3), (2, [1, 2], [3, 4]), (4, 3, 5)]:
            with self.subTest(d=d):
                kumaraswamy = Kumaraswamy(DigitalNetB2(d, seed=7), a=a, b=b)
                covariance = kumaraswamy.covariance

                self.assertEqual(covariance.shape, (d, d))
                self.assertEqual(covariance.ndim, 2)
                # Independent marginals: covariance is diagonal with the
                # per-dimension variances on the diagonal. It is stored sparsely.
                dense = dense_covariance(covariance)
                np.testing.assert_allclose(
                    np.diag(dense), kumaraswamy.variance
                )
                np.testing.assert_allclose(
                    dense, np.diag(np.diag(dense))
                )

    def test_variance_matches_closed_form(self):
        # Kumaraswamy raw moments: M_n = b * B(1 + n/a, b), so
        # variance = M_2 - M_1**2. Compare the quadrature-based variance
        # against this closed form evaluated with scipy's beta function.
        from scipy.special import beta as beta_function

        a = np.array([0.01, 1.0, 2.0, 3.5])
        b = np.array([1, 3.0, 4.0, 1.5])
        kumaraswamy = Kumaraswamy(DigitalNetB2(4, seed=7), a=a, b=b)

        m1 = b * beta_function(1 + 1 / a, b)
        m2 = b * beta_function(1 + 2 / a, b)
        expected_variance = m2 - m1**2

        np.testing.assert_allclose(
            kumaraswamy.variance, expected_variance, rtol=1e-10
        )


class TestZeroInflatedExpUniform(unittest.TestCase):
    """Moment tests for the (1D) zero-inflated exponential true measure.

    The distribution has probability mass ``p_zero`` at 0 and, otherwise,
    an exponential with rate ``lam``. Its closed-form moments are
    ``mean = (1 - p) / lam`` and ``variance = (1 - p**2) / lam**2``.
    """

    @staticmethod
    def _closed_form(p_zero, lam):
        mean = (1.0 - p_zero) / lam
        variance = (1.0 - p_zero**2) / lam**2
        return mean, variance

    def test_moment_attributes_match_closed_form(self):
        for p_zero, lam in [(0.4, 1.5), (0.1, 0.5), (0.75, 3.0)]:
            with self.subTest(p_zero=p_zero, lam=lam):
                tm = ZeroInflatedExpUniform(
                    DigitalNetB2(1, seed=7), p_zero=p_zero, lam=lam
                )
                mean, variance = self._closed_form(p_zero, lam)

                np.testing.assert_allclose(tm.mean, [mean])
                np.testing.assert_allclose(tm.variance, [variance])
                np.testing.assert_allclose(
                    tm.standard_deviation, [np.sqrt(variance)]
                )

    def test_moment_attributes_are_scalars(self):
        tm = ZeroInflatedExpUniform(
            DigitalNetB2(1, seed=7), p_zero=0.4, lam=1.5
        )
        self.assertEqual(tm.d, 1)
        # Univariate measures return scalar (0-d) moments rather than length-1 arrays.
        for value in (tm.mean, tm.variance, tm.standard_deviation):
            self.assertIsInstance(value, float)
            self.assertEqual(np.ndim(value), 0)
        np.testing.assert_allclose(
            tm.standard_deviation**2, tm.variance
        )

    def test_covariance_is_not_exposed(self):
        # Covariance is intentionally omitted for this 1D measure: it would
        # be a 1x1 matrix equal to the variance, so it adds no information.
        tm = ZeroInflatedExpUniform(
            DigitalNetB2(1, seed=7), p_zero=0.4, lam=1.5
        )
        self.assertNotIn("covariance", tm.parameters)
        self.assertNotIn("covariance", str(tm))
        self.assertFalse(hasattr(tm, "covariance"))
        with self.assertRaises(AttributeError):
            tm.covariance

    def test_moment_parameters_are_public_and_in_repr(self):
        tm = ZeroInflatedExpUniform(
            DigitalNetB2(1, seed=7), p_zero=0.4, lam=1.5
        )
        for parameter in (
            "mean",
            "variance",
            "standard_deviation",
        ):
            self.assertIn(parameter, tm.parameters)
            self.assertIn(parameter, str(tm))

    def test_moment_attributes_are_read_only(self):
        tm = ZeroInflatedExpUniform(
            DigitalNetB2(1, seed=7), p_zero=0.4, lam=1.5
        )
        for parameter in (
            "mean",
            "variance",
            "standard_deviation",
        ):
            with self.subTest(parameter=parameter):
                value = getattr(tm, parameter)
                # Univariate moments are returned as immutable Python floats,
                # and the attribute has no setter.
                self.assertIsInstance(value, float)
                with self.assertRaises(AttributeError):
                    setattr(tm, parameter, 0.0)

    def test_sample_mean_and_variance(self):
        tm = ZeroInflatedExpUniform(
            DigitalNetB2(1, seed=7), p_zero=0.4, lam=1.5
        )
        samples = tm.gen_samples(2**18)
        sample_mean = samples.mean(axis=0)
        sample_variance = samples.var(axis=0)

        np.testing.assert_allclose(
            sample_mean, tm.mean, rtol=0, atol=1e-4
        )
        np.testing.assert_allclose(
            sample_variance, tm.variance, rtol=0, atol=1e-3
        )

    def test_spawn_preserves_type_and_moments(self):
        tm = ZeroInflatedExpUniform(
            DigitalNetB2(1, seed=7), p_zero=0.4, lam=1.5
        )
        spawn = tm.spawn(1)[0]

        self.assertIsInstance(spawn, ZeroInflatedExpUniform)
        np.testing.assert_allclose(spawn.mean, tm.mean)
        np.testing.assert_allclose(spawn.variance, tm.variance)
        np.testing.assert_allclose(
            spawn.standard_deviation, tm.standard_deviation
        )
        self.assertFalse(hasattr(spawn, "covariance"))

    def test_deprecated_2d_construction_has_no_moment_parameters(self):
        # The deprecated 2D y_split construction does not define moments.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            tm = ZeroInflatedExpUniform(
                DigitalNetB2(2, seed=7),
                p_zero=0.4,
                lam=1.5,
                y_split=0.5,
            )
        for parameter in (
            "mean",
            "variance",
            "standard_deviation",
            "covariance",
        ):
            self.assertNotIn(parameter, tm.parameters)


class TestUniformTriangle(unittest.TestCase):
    """Tests for UniformTriangle and _UniformTriangleAdapter."""

    def test_basic_usage_and_dim_error(self):
        tm = UniformTriangle(sampler=DigitalNetB2(2, seed=7))
        x = tm(8)
        self.assertEqual(x.shape, (8, 2))
        self.assertTrue(np.all(x[:, 1] <= x[:, 0]))

        with self.assertRaises(DimensionError):
            tm._joint.transform(np.ones((4, 3)))

    def test_adapter_transform(self):
        adapter = _UniformTriangleAdapter()
        u = np.array([[0.25, 0.5], [0.64, 0.75], [0.01, 0.9]])
        x = adapter.transform(u)
        self.assertEqual(x.shape, (3, 2))
        # y = u2 * sqrt(u1) <= sqrt(u1) = x  always since u2 in [0,1]
        self.assertTrue(np.all(x[:, 1] <= x[:, 0] + 1e-12))

    def test_adapter_transform_dim_error(self):
        adapter = _UniformTriangleAdapter()
        # Wrong last dimension (3 instead of 2)
        self.assertRaises(DimensionError, adapter.transform, np.ones((4, 3)))

    def test_adapter_logpdf_inside(self):
        adapter = _UniformTriangleAdapter()
        # Points inside the triangle (y <= x, x in [0,1])
        x = np.array([[0.5, 0.3], [0.8, 0.0], [1.0, 0.5]])
        lp = adapter.logpdf(x)
        self.assertEqual(lp.shape, (3,))
        self.assertTrue(np.all(np.isfinite(lp)))

    def test_adapter_logpdf_outside(self):
        adapter = _UniformTriangleAdapter()
        # Points outside the triangle (y > x)
        x = np.array([[0.3, 0.5]])
        lp = adapter.logpdf(x)
        self.assertEqual(lp[0], -np.inf)

    def test_adapter_logpdf_dim_error(self):
        adapter = _UniformTriangleAdapter()
        self.assertRaises(DimensionError, adapter.logpdf, np.ones((4, 3)))

    def test_adapter_logpdf_batched(self):
        adapter = _UniformTriangleAdapter()
        u = np.random.default_rng(42).uniform(size=(5, 2))
        x = adapter.transform(u)
        lp = adapter.logpdf(x)
        self.assertEqual(lp.shape, (5,))


class TestGaussian(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with fixed seeds for reproducibility."""
        self.seed = 42

    def test_sample_mean_and_covariance(self):
        gaussian = Gaussian(
            DigitalNetB2(2, seed=7),
            mean=[1, 2],
            covariance=[[0.09, 0.04], [0.04, 0.05]],
        )

        assert_sample_mean_and_covariance(gaussian)

    def test_gaussian_basic_output_reproducibility(self):
        """Test that basic Gaussian sample generation produces expected values with fixed seed."""
        gaussian = Gaussian(Lattice(4, seed=self.seed), mean=0, covariance=1)

        samples = gaussian.gen_samples(2)

        # Expected output based on fixed seed
        expected_samples = np.array(
            [
                [0.0751091, -0.3100827, 1.60190625, -0.19901344],
                [-1.88173437, 1.16627832, -0.13726206, 1.41268702],
            ]
        )

        np.testing.assert_array_almost_equal(
            samples,
            expected_samples,
            decimal=6,
            err_msg="Gaussian sample generation output changed unexpectedly",
        )

    def test_gaussian_transformation_consistency(self):
        """Test that Gaussian transformation is mathematically consistent."""
        n_paths = 4
        d = 8
        sampler = Lattice(d, seed=self.seed)

        # Create Gaussian measure
        gaussian = Gaussian(sampler, mean=0, covariance=1)

        # Generate uniform samples
        uniform_samples = sampler.gen_samples(n_paths)

        # Manual original transformation
        from scipy.stats import norm

        normal_samples = norm.ppf(uniform_samples)
        original_result = gaussian.mu + np.einsum(
            "...ij,kj->...ik", normal_samples, gaussian.a
        )

        # Optimized transformation
        optimized_result = gaussian._transform(uniform_samples)

        # Check consistency
        np.testing.assert_array_almost_equal(
            original_result,
            optimized_result,
            decimal=10,
            err_msg="Gaussian transformation methods are inconsistent",
        )

    def test_gaussian_custom_mean_covariance(self):
        """Test Gaussian with custom mean and covariance parameters."""
        custom_mean = np.array([1.0, 2.0])
        custom_cov = np.array([[4.0, 1.0], [1.0, 3.0]])

        gaussian = Gaussian(
            Lattice(2, seed=self.seed), mean=custom_mean, covariance=custom_cov
        )

        samples = gaussian.gen_samples(2)

        # Expected output with custom parameters
        expected_samples = np.array(
            [[0.88570154, 2.49195225], [-1.49352915, -1.65710039]]
        )

        np.testing.assert_array_almost_equal(
            samples,
            expected_samples,
            decimal=6,
            err_msg="Gaussian with custom parameters output changed unexpectedly",
        )

    def test_gaussian_weight_computation(self):
        """Test that Gaussian PDF weight computation produces expected values."""
        gaussian = Gaussian(
            Lattice(2, seed=self.seed), mean=np.array([0.0, 0.0]), covariance=np.eye(2)
        )

        # Test with specific known samples
        test_samples = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]])

        weights = gaussian._weight(test_samples)

        # Expected weights for standard bivariate normal
        expected_weights = np.array([0.15915494, 0.05854983, 0.05854983])

        np.testing.assert_array_almost_equal(
            weights,
            expected_weights,
            decimal=6,
            err_msg="Gaussian weight computation changed unexpectedly",
        )

    def test_gaussian_decomposition_types(self):
        """Test different decomposition types for Gaussian."""
        custom_cov = np.array([[4.0, 2.0], [2.0, 3.0]])

        # Test Cholesky decomposition
        gaussian_chol = Gaussian(
            Lattice(2, seed=self.seed),
            mean=0,
            covariance=custom_cov,
            decomp_type="Cholesky",
        )

        # Test PCA decomposition
        gaussian_pca = Gaussian(
            Lattice(2, seed=self.seed), mean=0, covariance=custom_cov, decomp_type="PCA"
        )

        samples_chol = gaussian_chol.gen_samples(2)
        samples_pca = gaussian_pca.gen_samples(2)

        # Both should produce valid samples (different due to decomposition method)
        self.assertEqual(samples_chol.shape, (2, 2))
        self.assertEqual(samples_pca.shape, (2, 2))
        self.assertEqual(samples_chol.dtype, np.float64)
        self.assertEqual(samples_pca.dtype, np.float64)

    def test_gaussian_mean_covariance_properties(self):
        """Test that Gaussian maintains correct mean and covariance properties."""
        custom_mean = np.array([1.0, -1.0, 2.0])
        custom_cov = np.array([[2.0, 0.5, 0.0], [0.5, 1.5, -0.3], [0.0, -0.3, 3.0]])
        expected_variance = np.array([2.0, 1.5, 3.0])

        gaussian = Gaussian(
            Lattice(3, seed=self.seed), mean=custom_mean, covariance=custom_cov
        )

        np.testing.assert_allclose(gaussian.mean, custom_mean)
        np.testing.assert_allclose(gaussian.variance, expected_variance)
        np.testing.assert_allclose(
            gaussian.standard_deviation, np.sqrt(expected_variance)
        )
        np.testing.assert_allclose(gaussian.covariance, custom_cov)

        # Verify the internal mean and decomposition remain consistent.
        np.testing.assert_array_almost_equal(
            gaussian.mu,
            custom_mean,
            decimal=10,
            err_msg="Gaussian mean property changed unexpectedly",
        )

        # Verify covariance reconstruction
        reconstructed_cov = gaussian.a @ gaussian.a.T
        np.testing.assert_array_almost_equal(
            reconstructed_cov,
            custom_cov,
            decimal=10,
            err_msg="Gaussian covariance reconstruction changed unexpectedly",
        )

    def test_gaussian_scalar_parameters(self):
        """Test Gaussian with scalar mean and covariance parameters."""
        gaussian = Gaussian(Lattice(3, seed=self.seed), mean=2.5, covariance=1.5)

        np.testing.assert_allclose(gaussian.mean, np.full(3, 2.5))
        np.testing.assert_allclose(gaussian.variance, np.full(3, 1.5))
        np.testing.assert_allclose(
            gaussian.standard_deviation, np.full(3, np.sqrt(1.5))
        )
        np.testing.assert_allclose(gaussian.covariance, 1.5 * np.eye(3))

        samples = gaussian.gen_samples(2)

        # Expected samples with scalar parameters
        expected_samples = np.array(
            [[2.59198948, 2.1202278, 4.46192646], [0.19535548, 3.92839339, 2.33188899]]
        )

        np.testing.assert_array_almost_equal(
            samples,
            expected_samples,
            decimal=6,
            err_msg="Gaussian with scalar parameters output changed unexpectedly",
        )

    def test_moment_attributes_with_diagonal_covariance_vector(self):
        gaussian = Gaussian(
            Lattice(3, seed=self.seed),
            mean=[-1, 0, 1],
            covariance=[1, 4, 9],
        )

        np.testing.assert_allclose(gaussian.mean, [-1, 0, 1])
        np.testing.assert_allclose(gaussian.variance, [1, 4, 9])
        np.testing.assert_allclose(
            gaussian.standard_deviation, [1, 2, 3]
        )
        np.testing.assert_allclose(
            gaussian.covariance, np.diag([1, 4, 9])
        )

    def test_spawn_recomputes_moment_attributes(self):
        gaussian = Gaussian(
            Lattice(2, seed=self.seed), mean=2.5, covariance=1.5
        )
        spawn = gaussian.spawn(1, dimensions=4)[0]

        np.testing.assert_allclose(spawn.mean, np.full(4, 2.5))
        np.testing.assert_allclose(spawn.variance, np.full(4, 1.5))
        np.testing.assert_allclose(
            spawn.standard_deviation, np.full(4, np.sqrt(1.5))
        )
        np.testing.assert_allclose(spawn.covariance, 1.5 * np.eye(4))


class TestBrownianMotion(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with fixed seeds for reproducibility."""
        self.seed = 7

    def test_brownian_motion_parent_values(self):
        """Test that underlying Brownian Motion values are correct."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed), t_final=1, drift=0.1, diffusion=0.2
        )

        # Test time vector from parent
        expected_time_vec = np.array([0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(
            gbm.time_vec,
            expected_time_vec,
            decimal=10,
            err_msg="Time vector from parent BrownianMotion changed unexpectedly",
        )

        # Test parent's covariance matrix
        expected_parent_cov = gbm.diffusion * np.minimum.outer(
            gbm.time_vec, gbm.time_vec
        )

        # Access parent's covariance
        if hasattr(gbm, "covariance"):
            np.testing.assert_array_almost_equal(
                gbm.covariance,
                expected_parent_cov,
                decimal=10,
                err_msg="Parent BrownianMotion covariance changed unexpectedly",
            )

    def test_brownian_bridge_output_reproducibility(self):
        """Test that Brownian Bridge construction produces expected values with fixed seed."""
        bb = BrownianMotion(DigitalNetB2(4, seed=self.seed), decomp_type="BrownianBridge")

        samples = bb.gen_samples(2)

        # Expected output based on fixed seed
        expected_samples = np.array(
            [
                [-0.02048429,  0.41054648, -0.13899299,  0.3095377 ],
                [-0.38732442, -1.19527027, -1.12175754, -1.58454187],
            ]
        )

        np.testing.assert_array_almost_equal(
            samples,
            expected_samples,
            decimal=6,
            err_msg="Brownian Bridge sample generation output changed unexpectedly",
        )

    def test_brownian_bridge_decomp_type(self):
        """Test BrownianBridge as a decomposition type for BrownianMotion."""
        bm = BrownianMotion(
            DigitalNetB2(4, seed=self.seed, replications=2),
            decomp_type="BrownianBridge")
        samples = bm.gen_samples(2)
        self.assertEqual(samples.shape, (2, 2, 4))
        self.assertEqual(samples.dtype, np.float64)

    def test_brownian_bridge_no_matrix_decomp(self):
        """BrownianBridge raises ParameterError when matrix decomposition is called."""
        bm = BrownianMotion(DigitalNetB2(4, seed=self.seed), decomp_type="BrownianBridge")
        with self.assertRaises(ParameterError):
            bm._compute_decomposition()

    def test_brownian_bridge_manual_replications_d4(self):
        """Manually construct a d=4 BrownianBridge path and compare with the automated version."""
        d, n, reps = 4, 4, 2
        t = np.linspace(1 / d, 1.0, d)

        # Automated result
        automated = BrownianMotion(
            DigitalNetB2(d, seed=self.seed, replications=reps),
            decomp_type="BrownianBridge"
        ).gen_samples(n)

        # Manual construction
        u = DigitalNetB2(d, seed=self.seed, replications=reps).gen_samples(n)
        z = scipy.stats.norm.ppf(u)

        w_0 = np.zeros((reps, n, 1))

        z_1 = z[..., 0:1]
        z_2 = z[..., 1:2]
        z_3 = z[..., 2:3]
        z_4 = z[..., 3:4]

        w_4 = np.sqrt(t[3]) * z_1

        mean = w_0 + (t[1] - 0.0) / (t[3] - 0.0) * (w_4 - w_0)
        std = np.sqrt((t[1] - 0.0) * (t[3] - t[1]) / (t[3] - 0.0))
        w_2 = mean + std * z_2

        mean = w_0 + (t[0] - 0.0) / (t[1] - 0.0) * (w_2 - w_0)
        std  = np.sqrt((t[0] - 0.0) * (t[1] - t[0]) / (t[1] - 0.0))
        w_1  = mean + std * z_4

        mean = w_2 + (t[2] - t[1]) / (t[3] - t[1]) * (w_4 - w_2)
        std  = np.sqrt((t[2] - t[1]) * (t[3] - t[2]) / (t[3] - t[1]))
        w_3  = mean + std * z_3

        expected = np.concatenate([w_1, w_2, w_3, w_4], axis=-1)

        # Check consistency
        self.assertEqual(automated.shape, (reps, n, d))
        np.testing.assert_array_almost_equal(
            expected, automated, decimal=10,
            err_msg="Manual d=4 BrownianBridge path with replications does not match automated version."
        )

    def test_brownian_bridge_manual_replications_d3(self):
        """Manually construct a d=3 BrownianBridge path and compare with the automated version."""
        d, n, reps = 3, 4, 2
        # default sampling order is van der Corput [1, 1/2, 3/4]

        # Automated result (suppress warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            automated = BrownianMotion(
                DigitalNetB2(d, seed=self.seed, replications=reps),
                decomp_type="BrownianBridge"
            ).gen_samples(n)

        # Manual construction
        u = DigitalNetB2(d, seed=self.seed, replications=reps).gen_samples(n)
        z = scipy.stats.norm.ppf(u)

        w_0 = np.zeros((reps, n, 1))

        z_1 = z[..., 0:1]
        z_2 = z[..., 1:2]
        z_3 = z[..., 2:3]

        w_3 = np.sqrt(1.0) * z_1

        mean = w_0 + (0.5 - 0.0) / (1.0 - 0.0) * (w_3 - w_0)
        std = np.sqrt((0.5 - 0.0) * (1.0 - 0.5) / (1.0 - 0.0))
        w_1 = mean + std * z_2

        mean = w_1 + (0.75 - 0.5) / (1.0 - 0.5) * (w_3 - w_1)
        std = np.sqrt((0.75 - 0.5) * (1.0 - 0.75) / (1.0 - 0.5))
        w_2 = mean + std * z_3

        expected = np.concatenate([w_1, w_2, w_3], axis=-1)

        # Check consistency
        self.assertEqual(automated.shape, (reps, n, d))
        np.testing.assert_array_almost_equal(
            expected, automated, decimal=10,
            err_msg="Manual d=3 BrownianBridge path with replications does not match automated version."
        )

    def test_brownian_bridge_custom_monitoring_times(self):
        """Manually construct a BrownianBridge path with custom monitoring times and compare with the automated version."""
        d, n, reps = 4, 4, 2
        # times in an order that hits all four anchor cases
        times = [0.6, 1.0, 0.3, 0.8]

        automated = BrownianMotion(
            DigitalNetB2(d, seed=self.seed, replications=reps),
            decomp_type="BrownianBridge", monitoring_times=times, bridge_vdc_gray_ordering=False
        )
        samples = automated.gen_samples(n)

        np.testing.assert_array_almost_equal(
            automated.time_vec, [0.3, 0.6, 0.8, 1.0],
            err_msg="time_vec should be sorted into increasing order"
        )

        u = DigitalNetB2(d, seed=self.seed, replications=reps).gen_samples(n)
        z = scipy.stats.norm.ppf(u)

        w_0 = np.zeros((reps, n, 1))

        z_1 = z[..., 0:1]
        z_2 = z[..., 1:2]
        z_3 = z[..., 2:3]
        z_4 = z[..., 3:4]

        w_2 = np.sqrt(0.6) * z_1

        w_4 = w_2 + np.sqrt(1.0 - 0.6) * z_2

        mean = w_0 + (0.3 - 0.0) / (0.6 - 0.0) * (w_2 - w_0)
        std = np.sqrt((0.3 - 0.0) * (0.6 - 0.3) / (0.6 - 0.0))
        w_1 = mean + std * z_3

        mean = w_2 + (0.8 - 0.6) / (1.0 - 0.6) * (w_4 - w_2)
        std = np.sqrt((0.8 - 0.6) * (1.0 - 0.8) / (1.0 - 0.6))
        w_3 = mean + std * z_4

        expected = np.concatenate([w_1, w_2, w_3, w_4], axis=-1)

        self.assertEqual(samples.shape, (reps, n, d))
        np.testing.assert_array_almost_equal(
            expected, samples, decimal=10,
            err_msg="Manual BrownianBridge path with custom monitoring times does not match automated version."
        )

    def test_brownian_bridge_vdc_ordering_matches_default(self):
        """Use 4 evenly spaced custom times and compare to van der Corput ordering"""
        d, n = 4, 4

        default = BrownianMotion(
            DigitalNetB2(d, seed=self.seed), decomp_type='BrownianBridge'
        ).gen_samples(n)

        reordered = BrownianMotion(
            DigitalNetB2(d, seed=self.seed),
            decomp_type='BrownianBridge', monitoring_times=np.linspace(1/d, 1.0, d)
        ).gen_samples(n)

        np.testing.assert_almost_equal(
            default, reordered, decimal=10,
            err_msg="4 evenly spaced custom times should match van der Corput ordering"
        )

    def test_brownian_bridge_output_order(self):
        """Test that custom ordered output matches given input and contains same values as increasing output"""
        times = [0.6, 1.0, 0.3, 0.8]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            increasing_output = BrownianMotion(DigitalNetB2(4, seed=self.seed),
                decomp_type="BrownianBridge", monitoring_times=times).gen_samples(8)
            custom_output = BrownianMotion(DigitalNetB2(4, seed=self.seed),
                decomp_type="BrownianBridge", monitoring_times=times,
                bridge_output_order="input").gen_samples(8)

        np.testing.assert_allclose(
            custom_output[..., np.argsort(times)], increasing_output,
            err_msg="custom ordered output should match given input and contain equivalent values to increasing output"
        )

    def test_brownian_bridge_warning_for_non_power_of_2(self):
        """BrownianBridge issues ParameterWarning for suboptimal d but still produces valid output."""
        from qmcpy.util import ParameterWarning
        with self.assertWarns(ParameterWarning):
            bm = BrownianMotion(DigitalNetB2(6, seed=self.seed), decomp_type='BrownianBridge')
        samples = bm.gen_samples(4)
        self.assertEqual(samples.shape, (4, 6))
        self.assertEqual(samples.dtype, np.float64)

    def test_brownian_bridge_lazy_decomp_false(self):
        """BrownianBridge proceeds with lazy_decomp=False"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bm = BrownianMotion(DigitalNetB2(6, seed=self.seed),
                decomp_type="BrownianBridge", lazy_decomp=False
            )
            samples = bm.gen_samples(4)
        self.assertEqual(samples.shape, (4,6))

    def test_brownian_bridge_spawn_matches_parent(self):
        """Spawn must match parent"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bm = BrownianMotion(
                DigitalNetB2(4, seed=self.seed),
                decomp_type="BrownianBridge",
                initial_value=2, drift=3, diffusion=4,
                monitoring_times=[0.6, 1.0, 0.3, 0.8],
                bridge_vdc_gray_ordering=True,
                bridge_output_order="input",
            )
            child = bm._spawn(DigitalNetB2(4, seed=self.seed), 4)
            parent_samples = bm.gen_samples(8)
            child_samples = child.gen_samples(8)
        np.testing.assert_array_equal(
            parent_samples, child_samples,
            err_msg="samples of a same dimension spawn should match parent samples"
        )

    def test_brownian_bridge_invalid_decomp_lazy_false(self):
        with self.assertRaises(ParameterError) as context:
            BrownianMotion(DigitalNetB2(4, seed=self.seed), decomp_type="invalid", lazy_decomp=False)
        self.assertIn("BrownianBridge", str(context.exception))

    def test_brownian_bridge_monitoring_times_exceed_t_final(self):
        with self.assertRaises(ParameterError):
            BrownianMotion(DigitalNetB2(4, seed=self.seed), t_final=1.0,
                           decomp_type="BrownianBridge",
                           monitoring_times=[0.1, 0.2, 0.3, 5.0])

    def test_brownian_bridge_monitoring_times_nan(self):
        with self.assertRaises(ParameterError):
            BrownianMotion(DigitalNetB2(4, seed=self.seed), t_final=1.0,
                           decomp_type="BrownianBridge",
                           monitoring_times=[0.1, 0.2, np.nan, 1.0])

    def test_brownian_motion_invalid_t_final(self):
        with self.assertRaises(ParameterError):
            BrownianMotion(DigitalNetB2(4, seed=self.seed), t_final=-8,
                           decomp_type="BrownianBridge")
        with self.assertRaises(ParameterError):
            BrownianMotion(DigitalNetB2(4, seed=self.seed), t_final=np.nan,
                           decomp_type="BrownianBridge")
    def test_moment_attributes(self):
        brownian_motion = BrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=2,
            initial_value=3,
            drift=0.5,
            diffusion=2,
        )
        expected_variance = np.array([1.0, 2.0, 3.0, 4.0])

        np.testing.assert_allclose(
            brownian_motion.variance, expected_variance
        )
        np.testing.assert_allclose(
            brownian_motion.standard_deviation,
            np.sqrt(expected_variance),
        )
        np.testing.assert_allclose(
            np.diag(brownian_motion.covariance),
            brownian_motion.variance,
        )

    def test_spawn_recomputes_moment_attributes(self):
        brownian_motion = BrownianMotion(
            DigitalNetB2(2, seed=self.seed),
            t_final=2,
            initial_value=3,
            drift=0.5,
            diffusion=2,
        )
        spawn = brownian_motion.spawn(1, dimensions=4)[0]
        expected_variance = np.array([1.0, 2.0, 3.0, 4.0])

        np.testing.assert_allclose(
            spawn.mean, np.array([3.25, 3.5, 3.75, 4.0])
        )
        np.testing.assert_allclose(spawn.variance, expected_variance)
        np.testing.assert_allclose(
            spawn.standard_deviation, np.sqrt(expected_variance)
        )
        np.testing.assert_allclose(
            spawn.covariance,
            2 * np.minimum.outer(spawn.time_vec, spawn.time_vec),
        )


class TestGeometricBrownianMotion(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with fixed seeds for reproducibility."""
        self.seed = 7

    def test_gbm_basic_output_reproducibility(self):
        """Test that basic GBM sample generation produces expected values with fixed seed."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed), t_final=2, drift=0.1, diffusion=0.2
        )

        samples = gbm.gen_samples(2)

        # Expected output
        expected_samples = np.array(
            [
                [0.92343761, 1.42069027, 1.30851806, 0.99133819],
                [0.7185916, 0.42028013, 0.42080335, 0.4696196],
            ]
        )

        np.testing.assert_array_almost_equal(
            samples,
            expected_samples,
            decimal=6,
            err_msg="GBM sample generation output changed unexpectedly",
        )

    def test_gbm_mean_computation(self):
        """Test that GBM mean computation produces expected values."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=2,
            initial_value=1,
            drift=0.1,
            diffusion=0.2,
        )

        expected_mean = np.array([1.051271096, 1.105170918, 1.161834243, 1.221402758])

        np.testing.assert_array_almost_equal(
            gbm.mean_gbm,
            expected_mean,
            decimal=6,
            err_msg="GBM mean computation changed unexpectedly",
        )

    def test_gbm_covariance_computation(self):
        """Test that GBM covariance computation produces expected values."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=2,
            initial_value=1,
            drift=0.1,
            diffusion=0.2,
        )

        # Expected covariance matrix - actual computed values
        expected_cov = np.array(
            [
                [0.116232, 0.122191, 0.128456, 0.135042],
                [0.122191, 0.270422, 0.284287, 0.298862],
                [0.128456, 0.284287, 0.47226, 0.496473],
                [0.135042, 0.298862, 0.496473, 0.733716],
            ]
        )

        np.testing.assert_array_almost_equal(
            gbm.covariance_gbm,
            expected_cov,
            decimal=6,
            err_msg="GBM covariance computation changed unexpectedly",
        )

    def test_gbm_weight_specific_values(self):
        """Test that PDF weight computation produces expected values for specific inputs."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=1,
            initial_value=1,
            drift=0.05,
            diffusion=0.1,
        )

        # Test with specific known samples
        test_samples = np.array([[1.0, 1.05, 1.1, 1.15], [0.9, 0.95, 1.0, 1.05]])

        weights = gbm._weight(test_samples)

        # Expected weights - actual computed values
        expected_weights = np.array([26.782039, 30.850616])

        np.testing.assert_array_almost_equal(
            weights,
            expected_weights,
            decimal=6,
            err_msg="GBM weight computation changed unexpectedly",
        )

    def test_transform_specific_inputs(self):
        """Test _transform method with specific uniform inputs."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=1,
            drift=0.1,
            diffusion=0.2,
            initial_value=2.0,
        )

        # Specific uniform inputs
        uniform_inputs = np.array([[0.1, 0.3, 0.7, 0.9], [0.5, 0.5, 0.5, 0.5]])

        transformed = gbm._transform(uniform_inputs)

        # Expected transformed values - actual computed values
        expected_transformed = np.array(
            [[1.73828, 1.166842, 1.297715, 1.24253], [2.0, 2.0, 2.0, 2.0]]
        )

        np.testing.assert_array_almost_equal(
            transformed,
            expected_transformed,
            decimal=6,
            err_msg="Transform method output changed unexpectedly",
        )

    def test_setup_lognormal_distribution_properties(self):
        """Test that _setup_lognormal_distribution creates correct log-space distribution."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=2,
            initial_value=1,
            drift=0.1,
            diffusion=0.2,
        )

        # Access log_mvn_scipy to trigger setup
        log_mvn = gbm.log_mvn_scipy

        # Test log-space mean: (drift - 0.5*diffusion) * time_vec
        expected_log_mean = (0.1 - 0.5 * 0.2) * gbm.time_vec
        np.testing.assert_array_almost_equal(
            log_mvn.mean,
            expected_log_mean,
            decimal=10,
            err_msg="Log-space mean computation in lognormal setup changed unexpectedly",
        )

        # Test log-space covariance structure: diffusion * min(t_i, t_j)
        expected_log_cov = 0.2 * np.minimum.outer(gbm.time_vec, gbm.time_vec)
        np.testing.assert_array_almost_equal(
            log_mvn.cov,
            expected_log_cov,
            decimal=10,
            err_msg="Log-space covariance computation in lognormal setup changed unexpectedly",
        )

    def test_lognormal_distribution_pdf_consistency(self):
        """Test that the lognormal distribution PDF is mathematically consistent."""
        gbm = GeometricBrownianMotion(
            DigitalNetB2(4, seed=self.seed),
            t_final=1,
            initial_value=2,
            drift=0.05,
            diffusion=0.1,
        )

        # Test with a known sample
        test_sample = np.array([[2.1, 2.15, 2.2, 2.25]])
        gbm_weight = gbm._weight(test_sample)[0]

        # Manually compute using log transformation
        log_sample = np.log(test_sample / gbm.initial_value).flatten()
        normal_pdf = gbm.log_mvn_scipy.pdf(log_sample)
        jacobian = 1.0 / test_sample.prod()
        manual_weight = normal_pdf * jacobian

        np.testing.assert_almost_equal(
            gbm_weight,
            manual_weight,
            decimal=10,
            err_msg="Lognormal PDF computation inconsistency detected",
        )

    def test_lognormal_setup_lazy_loading(self):
        """Test that lognormal distribution setup works with lazy loading."""
        # Test with lazy_load=True (default)
        gbm_lazy = GeometricBrownianMotion(
            DigitalNetB2(3, seed=self.seed),
            t_final=1,
            drift=0.1,
            diffusion=0.2,
            lazy_load=True,
        )
        self.assertIsNone(gbm_lazy._log_mvn_scipy_cache)
        log_mvn = gbm_lazy.log_mvn_scipy
        self.assertIsNotNone(gbm_lazy._log_mvn_scipy_cache)

        # Test with lazy_load=False
        gbm_eager = GeometricBrownianMotion(
            DigitalNetB2(3, seed=self.seed),
            t_final=1,
            drift=0.1,
            diffusion=0.2,
            lazy_load=False,
        )
        self.assertIsNotNone(gbm_eager._log_mvn_scipy_cache)


class TestAcceptanceRejection(unittest.TestCase):
    """Unit tests for AcceptanceRejection and AcceptanceRejectionReal."""

    def setUp(self):
        from qmcpy import AcceptanceRejection, AcceptanceRejectionReal
        from scipy.stats import norm
        self.AcceptanceRejection = AcceptanceRejection
        self.AcceptanceRejectionReal = AcceptanceRejectionReal
        self.norm = norm
        self.seed = 7

        def psi(x): return 2 * x[:, 0]
        self.psi = psi

        def psi_real(z): return norm.pdf(z[:, 0], loc=0, scale=1)
        def H(z): return norm.pdf(z[:, 0], loc=0, scale=2)
        self.psi_real = psi_real
        self.H = H
        self.inv_cdfs = [lambda u: norm.ppf(u, loc=0, scale=2)]

    def _make_ar(self, seed=None):
        seed = seed if seed is not None else self.seed
        return self.AcceptanceRejection(
            DigitalNetB2(dimension=2, seed=seed),
            self.psi, upper_bound=2., density_integral=1.
        )

    def _make_ar_real(self, seed=None):
        seed = seed if seed is not None else self.seed
        return self.AcceptanceRejectionReal(
            DigitalNetB2(dimension=2, seed=seed),
            self.psi_real,
            inv_cdfs=self.inv_cdfs,
            H_func=self.H,
            upper_bound=2., density_integral=1.
        )

    # --- AcceptanceRejection tests ---

    def test_basic_shape(self):
        """gen_samples(n=64) returns shape (64, 1)."""
        samples = self._make_ar().gen_samples(n=64, warn=False)
        self.assertEqual(samples.shape, (64, 1))

    def test_samples_in_unit_interval(self):
        """All accepted samples lie in [0, 1]."""
        samples = self._make_ar().gen_samples(n=256, warn=False)
        self.assertTrue((samples >= 0).all() and (samples <= 1).all())

    def test_mean_convergence(self):
        """Sample mean should be close to the true mean of 2/3."""
        samples = self._make_ar().gen_samples(n=1024, warn=False)
        self.assertAlmostEqual(samples.mean(), 2/3, delta=0.05)

    def test_return_weights(self):
        """return_weights=True gives samples and positive weights of correct shape."""
        s, w = self._make_ar().gen_samples(n=64, return_weights=True, warn=False)
        self.assertEqual(s.shape, (64, 1))
        self.assertEqual(w.shape, (64,))
        self.assertTrue(np.all(w > 0))

    def test_continued_sampling_matches_single_call(self):
        """Two batches via n_min/n_max equal one single call."""
        m1 = self._make_ar()
        b1 = m1.gen_samples(n_min=0, n_max=8)
        b2 = m1.gen_samples(n_min=8, n_max=16)

        m2 = self._make_ar()
        all_at_once = m2.gen_samples(n_min=0, n_max=16)

        np.testing.assert_array_almost_equal(
            np.concatenate([b1, b2]), all_at_once
        )

    def test_n_resets_driver(self):
        """Calling gen_samples(n=8) twice gives same result (driver resets)."""
        m = self._make_ar()
        s1 = m.gen_samples(n=8)
        s2 = m.gen_samples(n=8)
        np.testing.assert_array_equal(s1, s2)

    def test_n_max_without_n_min(self):
        """Calling with only n_max (no n_min) works and returns correct shape."""
        s = self._make_ar().gen_samples(n_max=64, warn=False)
        self.assertEqual(s.shape, (64, 1))

    def test_error_n_min_without_prior_call(self):
        """n_min > 0 without a prior call raises ParameterError."""
        m = self._make_ar()
        with self.assertRaises(ParameterError):
            m.gen_samples(n_min=4, n_max=8)

    def test_error_invalid_upper_bound(self):
        """upper_bound <= 0 raises ParameterError."""
        with self.assertRaises(ParameterError):
            self.AcceptanceRejection(
                DigitalNetB2(dimension=2, seed=self.seed),
                self.psi, upper_bound=0., density_integral=1.
            )

    def test_error_invalid_density_integral(self):
        """density_integral <= 0 raises ParameterError."""
        with self.assertRaises(ParameterError):
            self.AcceptanceRejection(
                DigitalNetB2(dimension=2, seed=self.seed),
                self.psi, upper_bound=2., density_integral=0.
            )

    def test_error_sampler_dimension_too_small(self):
        """Sampler dimension < 2 raises ParameterError."""
        with self.assertRaises(ParameterError):
            self.AcceptanceRejection(
                DigitalNetB2(dimension=1, seed=self.seed),
                self.psi, upper_bound=2., density_integral=1.
            )

    # --- AcceptanceRejectionReal tests ---

    def test_real_basic_shape(self):
        """AcceptanceRejectionReal gen_samples(n=64) returns shape (64, 1)."""
        samples = self._make_ar_real().gen_samples(n=64, warn=False)
        self.assertEqual(samples.shape, (64, 1))

    def test_real_mean_and_std(self):
        """Sample mean and std should be close to N(0,1) values."""
        samples = self._make_ar_real().gen_samples(n=512, warn=False)
        self.assertAlmostEqual(samples.mean(), 0.0, delta=0.1)
        self.assertAlmostEqual(samples.std(), 1.0, delta=0.1)

    def test_real_continued_sampling_shapes(self):
        """AcceptanceRejectionReal continued sampling returns correct shapes."""
        m = self._make_ar_real()
        b1 = m.gen_samples(n_min=0, n_max=8)
        b2 = m.gen_samples(n_min=8, n_max=16)
        self.assertEqual(b1.shape, (8, 1))
        self.assertEqual(b2.shape, (8, 1))

    def test_real_error_n_min_without_prior_call(self):
        """AcceptanceRejectionReal n_min > 0 without prior call raises ParameterError."""
        m = self._make_ar_real()
        with self.assertRaises(ParameterError):
            m.gen_samples(n_min=8, n_max=16)

    def test_real_error_inv_cdfs_length_mismatch(self):
        """inv_cdfs length != target_dim raises ParameterError."""
        with self.assertRaises(ParameterError):
            self.AcceptanceRejectionReal(
                DigitalNetB2(dimension=2, seed=self.seed),
                self.psi_real,
                inv_cdfs=[],  # wrong length
                H_func=self.H,
                upper_bound=2., density_integral=1.
            )


if __name__ == "__main__":
    unittest.main()
