import unittest

import numpy as np
import scipy.sparse as sp
import scipy.stats as stats

from qmcpy import (
    AcceptanceRejection,
    DigitalNetB2,
    DummySampler,
    Gaussian,
    GaussianCopula,
    ProductMeasure,
    SciPyWrapper,
    Uniform,
    ZeroInflatedExpUniform,
)
from qmcpy.util import DimensionError, ParameterError



class TestProductMeasure(unittest.TestCase):
    def test_zero_inflated_with_scipy_uniform_shape(self):
        n = 32
        marginals = [
            ZeroInflatedExpUniform(DummySampler(1), p_zero=0.4, lam=1.5),
            SciPyWrapper(DummySampler(1), stats.uniform(loc=2.0, scale=3.0)),
        ]
        tm = ProductMeasure(sampler=DigitalNetB2(2, seed=23), marginals=marginals)

        x = tm(n)

        self.assertEqual(x.shape, (n, 2))
        self.assertTrue(np.any(x[:, 0] == 0.0))
        self.assertTrue(np.all((2.0 <= x[:, 1]) & (x[:, 1] <= 5.0)))


    def test_replication_shape(self):
        n = 16
        r = 3
        marginals = [
            ZeroInflatedExpUniform(DummySampler(1), p_zero=0.4, lam=1.5),
            Uniform(DummySampler(1), lower_bound=2.0, upper_bound=5.0),
        ]
        tm = ProductMeasure(
            sampler=DigitalNetB2(2, seed=23, replications=r),
            marginals=marginals,
        )

        x = tm(n)

        self.assertEqual(x.shape, (r, n, 2))


    def test_statistics_for_multiple_1d_marginals(self):
        marginals = [
            Uniform(DummySampler(1), lower_bound=8.0, upper_bound=12.0),
            Uniform(DummySampler(1), lower_bound=-1.0, upper_bound=5.0),
        ]
        tm = ProductMeasure(sampler=DigitalNetB2(2, seed=23), marginals=marginals)

        np.testing.assert_allclose(tm.mean, [10.0, 2.0])
        np.testing.assert_allclose(tm.variance, [4.0 / 3.0, 3.0])
        np.testing.assert_allclose(
            tm.standard_deviation, [np.sqrt(4.0 / 3.0), np.sqrt(3.0)]
        )
        cov = tm.covariance
        cov_dense = cov.toarray() if sp.issparse(cov) else cov
        np.testing.assert_allclose(cov_dense, np.diag([4.0 / 3.0, 3.0]))

        self.assertEqual(tm.mean.shape, (2,))
        self.assertEqual(tm.variance.shape, (2,))
        self.assertEqual(tm.standard_deviation.shape, (2,))
        self.assertEqual(tm.covariance.shape, (2, 2))
        for statistic in ("mean", "variance", "standard_deviation", "covariance"):
            value = getattr(tm, statistic)
            flags = value.data.flags if sp.issparse(value) else value.flags
            self.assertFalse(flags.writeable)


    def test_normalizes_scalar_1d_statistics(self):
        marginals = [
            Uniform(DummySampler(1), lower_bound=8.0, upper_bound=12.0),
        Gaussian(DummySampler(1), mean=2.0, covariance=9.0),
        ]
        for marginal in marginals:
            self.assertIsInstance(marginal.mean, float)
            self.assertIsInstance(marginal.variance, float)
            self.assertIsInstance(marginal.standard_deviation, float)

        tm = ProductMeasure(sampler=DigitalNetB2(2, seed=29), marginals=marginals)

        np.testing.assert_allclose(tm.mean, [10.0, 2.0])
        np.testing.assert_allclose(tm.variance, [4.0 / 3.0, 9.0])
        np.testing.assert_allclose(
            tm.standard_deviation, [np.sqrt(4.0 / 3.0), 3.0]
        )
        np.testing.assert_allclose(
            tm.covariance.toarray(), np.diag([4.0 / 3.0, 9.0])
        )
        self.assertEqual(tm.mean.shape, (2,))
        self.assertEqual(tm.variance.shape, (2,))
        self.assertEqual(tm.standard_deviation.shape, (2,))
        self.assertEqual(tm.covariance.shape, (2, 2))


    def test_statistics_preserve_order_and_covariance_blocks(self):
        marginals = [
            Uniform(DummySampler(1), lower_bound=8.0, upper_bound=12.0),
            Gaussian(
                DummySampler(2),
                mean=[2.0, 5.0],
                covariance=[[2.0, 0.5], [0.5, 3.0]],
            ),
        ]
        tm = ProductMeasure(sampler=DigitalNetB2(3, seed=31), marginals=marginals)
        expected_covariance = np.array(
            [
                [4.0 / 3.0, 0.0, 0.0],
                [0.0, 2.0, 0.5],
                [0.0, 0.5, 3.0],
            ]
        )

        np.testing.assert_allclose(tm.mean, [10.0, 2.0, 5.0])
        np.testing.assert_allclose(tm.variance, [4.0 / 3.0, 2.0, 3.0])
        np.testing.assert_allclose(
            tm.standard_deviation,
            [np.sqrt(4.0 / 3.0), np.sqrt(2.0), np.sqrt(3.0)],
        )
        covariance = tm.covariance.tocsr()
        np.testing.assert_allclose(covariance.toarray(), expected_covariance)

        self.assertEqual(tm.mean.shape, (3,))
        self.assertEqual(tm.variance.shape, (3,))
        self.assertEqual(tm.standard_deviation.shape, (3,))
        self.assertEqual(covariance.shape, (3, 3))
        self.assertEqual(covariance[:1, 1:].nnz, 0)
        self.assertEqual(covariance[1:, :1].nnz, 0)


    def test_mixed_covariance_blocks_remain_sparse(self):
        d = 128
        tm = ProductMeasure(
            DummySampler(d + 2),
            [
                Uniform(DummySampler(d)),
                Gaussian(
                    DummySampler(2),
                    covariance=np.array([[1.0, 0.5], [0.5, 1.0]]),
                ),
            ],
        )

        covariance = tm.covariance
        expected = sp.block_diag(
            [marginal.covariance for marginal in tm.marginals], format="csr"
        )

        self.assertTrue(sp.issparse(covariance))
        self.assertEqual(covariance.format, "csr")
        self.assertEqual(covariance.shape, (d + 2, d + 2))
        difference = (covariance - expected).tocsr()
        difference.eliminate_zeros()
        self.assertEqual(difference.nnz, 0)
        covariance_csr = covariance.tocsr()
        np.testing.assert_allclose(
            covariance_csr[-2:, -2:].toarray(), [[1.0, 0.5], [0.5, 1.0]]
        )
        self.assertEqual(covariance_csr[:d, d:].nnz, 0)
        self.assertEqual(covariance_csr[d:, :d].nnz, 0)
        self.assertFalse(covariance.data.flags.writeable)
        with self.assertRaises(ValueError):
            covariance.data.setflags(write=True)


    def test_sparse_covariance_repr_is_compact(self):
        d = 128
        tm = ProductMeasure(
            DummySampler(d + 2),
            [
                Uniform(DummySampler(d)),
                Gaussian(
                    DummySampler(2),
                    covariance=np.array([[1.0, 0.5], [0.5, 1.0]]),
                ),
            ],
        )

        representation = repr(tm)

        self.assertIn("sparse CSR", representation)
        self.assertIn("shape=(130, 130)", representation)
        self.assertIn("nnz=132", representation)
        self.assertNotIn("Coords", representation)
        self.assertNotIn("(127, 127)", representation)


    def test_statistics_are_lazily_cached(self):
        tm = ProductMeasure(
            DummySampler(3),
            [
                Uniform(DummySampler(1), lower_bound=8.0, upper_bound=12.0),
                Gaussian(
                    DummySampler(2),
                    mean=[2.0, 5.0],
                    covariance=[[2.0, 0.5], [0.5, 3.0]],
                ),
            ],
        )

        cache = vars(tm)
        self.assertIsNone(cache["_mean_cache"])
        self.assertIsNone(cache["_variance_cache"])
        self.assertIsNone(cache["_standard_deviation_cache"])
        self.assertIsNone(cache["_covariance_cache"])

        mean = tm.mean
        np.testing.assert_allclose(mean, [10.0, 2.0, 5.0])
        self.assertIs(tm.mean, mean)
        self.assertIsNone(cache["_variance_cache"])
        self.assertIsNone(cache["_standard_deviation_cache"])
        self.assertIsNone(cache["_covariance_cache"])

        variance = tm.variance
        standard_deviation = tm.standard_deviation
        covariance = tm.covariance
        np.testing.assert_allclose(variance, [4.0 / 3.0, 2.0, 3.0])
        np.testing.assert_allclose(
            standard_deviation,
            [np.sqrt(4.0 / 3.0), np.sqrt(2.0), np.sqrt(3.0)],
        )
        np.testing.assert_allclose(
            covariance.toarray(),
            [[4.0 / 3.0, 0.0, 0.0], [0.0, 2.0, 0.5], [0.0, 0.5, 3.0]],
        )
        self.assertIs(tm.variance, variance)
        self.assertIs(tm.standard_deviation, standard_deviation)
        self.assertIs(tm.covariance, covariance)


    def test_dense_covariance_cannot_be_made_writeable(self):
        tm = ProductMeasure(
            DummySampler(3),
            [
                Gaussian(DummySampler(1), covariance=2.0),
                Gaussian(
                    DummySampler(2), covariance=[[3.0, 0.25], [0.25, 4.0]]
                ),
            ],
        )

        covariance = tm.covariance

        self.assertIsInstance(covariance, np.ndarray)
        np.testing.assert_allclose(
            covariance,
            [[2.0, 0.0, 0.0], [0.0, 3.0, 0.25], [0.0, 0.25, 4.0]],
        )
        self.assertFalse(covariance.flags.writeable)
        with self.assertRaises(ValueError):
            covariance.setflags(write=True)


    def test_missing_marginal_statistic_is_identified(self):
        marginals = [
            ZeroInflatedExpUniform(DummySampler(1), p_zero=0.4, lam=1.5),
            Uniform(DummySampler(1), lower_bound=2.0, upper_bound=5.0),
        ]
        tm = ProductMeasure(sampler=DigitalNetB2(2, seed=23), marginals=marginals)

        np.testing.assert_allclose(tm.mean, [0.4, 3.5])
        self.assertNotIn("covariance", tm.parameters)
        with self.assertRaisesRegex(
            AttributeError,
            r"marginal 0 \(ZeroInflatedExpUniform\) does not provide covariance",
        ):
            _ = tm.covariance


    def test_marginals_with_different_dimensions(self):
        n = 32
        marginals = [
            Gaussian(
                DummySampler(2),
                mean=[1.0, -1.0],
                covariance=[[2.0, 0.25], [0.25, 1.0]],
            ),
            ZeroInflatedExpUniform(DummySampler(1), p_zero=0.4, lam=1.5),
        ]
        tm = ProductMeasure(sampler=DigitalNetB2(3, seed=31), marginals=marginals)

        x = tm(n)

        self.assertEqual(tm.d, 3)
        np.testing.assert_array_equal(tm.marginal_dimensions, np.array([2, 1]))
        self.assertEqual(x.shape, (n, 3))
        self.assertTrue(np.all(np.isfinite(x[:, :2])))
        self.assertTrue(np.all(x[:, 2] >= 0.0))


    def test_block_split_range_and_weight_product(self):
        n = 16
        marginals = [
            Uniform(DummySampler(1), lower_bound=10.0, upper_bound=12.0),
            Uniform(
                DummySampler(2),
                lower_bound=[20.0, 30.0],
                upper_bound=[24.0, 36.0],
            ),
        ]
        tm = ProductMeasure(sampler=DigitalNetB2(3, seed=41), marginals=marginals)

        u = tm.discrete_distrib.gen_samples(n)
        x = tm._transform(u)
        x_call, jac = tm(n, return_weights=True)
        expected = np.concatenate(
            [
                marginals[0]._jacobian_transform_r(u[..., :1], return_weights=False),
                marginals[1]._jacobian_transform_r(u[..., 1:], return_weights=False),
            ],
            axis=-1,
        )
        expected_range = np.array([[10.0, 12.0], [20.0, 24.0], [30.0, 36.0]])

        self.assertEqual(x.shape, (n, 3))
        np.testing.assert_allclose(tm.range, expected_range)
        np.testing.assert_allclose(x, expected)
        self.assertTrue(np.all((10.0 <= x[:, 0]) & (x[:, 0] <= 12.0)))
        self.assertTrue(np.all((20.0 <= x[:, 1]) & (x[:, 1] <= 24.0)))
        self.assertTrue(np.all((30.0 <= x[:, 2]) & (x[:, 2] <= 36.0)))
        np.testing.assert_allclose(tm._weight(x), 1.0 / (2.0 * 4.0 * 6.0))
        self.assertEqual(x_call.shape, (n, 3))
        np.testing.assert_allclose(jac, 2.0 * 4.0 * 6.0)


    def test_invalid_inputs(self):
        with self.assertRaisesRegex(ParameterError, "nonempty list of marginals"):
            ProductMeasure(sampler=DigitalNetB2(1, seed=7), marginals=[])

        with self.assertRaisesRegex(ParameterError, "marginal"):
            ProductMeasure(sampler=DigitalNetB2(1, seed=7), marginals=[object()])

        with self.assertRaisesRegex(ParameterError, "AbstractDiscreteDistribution"):
            ProductMeasure(sampler=object(), marginals=[Uniform(DummySampler(1))])

        marginals = [Uniform(DummySampler(1))]
        with self.assertRaisesRegex(DimensionError, "sum of marginal dimensions"):
            ProductMeasure(sampler=DigitalNetB2(2, seed=7), marginals=marginals)


    def test_rejects_non_dimension_preserving_marginal(self):
        marginal = AcceptanceRejection(
            DigitalNetB2(2, seed=7),
            lambda x: np.ones(len(x)),
            1.0,
            1.0,
        )

        with self.assertRaisesRegex(DimensionError, "dimension-preserving"):
            ProductMeasure(DigitalNetB2(2, seed=11), [marginal])


    def test_spawn_preserves_marginal_blocks_and_replaces_outer_sampler(self):
        marginals = [
            Uniform(DummySampler(1), lower_bound=10.0, upper_bound=12.0),
            Uniform(
                DummySampler(2),
                lower_bound=[20.0, 30.0],
                upper_bound=[24.0, 36.0],
            ),
        ]
        tm = ProductMeasure(sampler=DigitalNetB2(3, seed=41), marginals=marginals)

        spawn = tm.spawn(s=1)[0]

        self.assertIsInstance(spawn, ProductMeasure)
        self.assertEqual(spawn.d, 3)
        self.assertEqual(spawn.marginals, tm.marginals)
        self.assertIsNot(spawn.discrete_distrib, tm.discrete_distrib)
        np.testing.assert_array_equal(spawn.marginal_dimensions, np.array([1, 2]))

        with self.assertRaises(DimensionError):
            tm.spawn(s=1, dimensions=4)


    def test_does_not_use_marginal_dummy_sampler_values(self):
        marginals = [
            Uniform(DummySampler(1), lower_bound=0.0, upper_bound=2.0),
            Uniform(DummySampler(1), lower_bound=10.0, upper_bound=12.0),
        ]

        with self.assertRaisesRegex(ParameterError, "construction placeholder"):
            marginals[0].discrete_distrib(4)

        tm = ProductMeasure(sampler=DigitalNetB2(2, seed=19), marginals=marginals)
        x = tm(8)

        self.assertEqual(x.shape, (8, 2))
        self.assertTrue(np.all((0.0 <= x[:, 0]) & (x[:, 0] <= 2.0)))
        self.assertTrue(np.all((10.0 <= x[:, 1]) & (x[:, 1] <= 12.0)))


    def test_same_outer_seed_matches_different_outer_seed_changes(self):
        marginals = [
            Uniform(DummySampler(1), lower_bound=0.0, upper_bound=2.0),
            Uniform(DummySampler(1), lower_bound=10.0, upper_bound=12.0),
        ]

        first = ProductMeasure(sampler=DigitalNetB2(2, seed=101), marginals=marginals)(16)
        same_outer = ProductMeasure(sampler=DigitalNetB2(2, seed=101), marginals=marginals)(16)
        different_outer = ProductMeasure(sampler=DigitalNetB2(2, seed=102), marginals=marginals)(16)

        np.testing.assert_array_equal(first, same_outer)
        self.assertFalse(np.array_equal(first, different_outer))


    def test_replication_means_close_to_uniform_targets(self):
        n = 1024
        r = 4
        marginals = [
            Uniform(DummySampler(1), lower_bound=0.0, upper_bound=2.0),
            Uniform(DummySampler(1), lower_bound=10.0, upper_bound=12.0),
        ]
        tm = ProductMeasure(
            sampler=DigitalNetB2(2, seed=101, replications=r),
            marginals=marginals,
        )

        x = tm(n)
        replication_means = x.mean(axis=1)

        self.assertEqual(x.shape, (r, n, 2))
        np.testing.assert_allclose(replication_means[:, 0], 1.0, atol=0.03)
        np.testing.assert_allclose(replication_means[:, 1], 11.0, atol=0.03)


    def test_with_scipywrapper_beta_marginal(self):
        n = 64
        marginals = [
            Uniform(DummySampler(1), lower_bound=-1.0, upper_bound=1.0),
            SciPyWrapper(DummySampler(1), stats.beta(a=2.0, b=5.0)),
        ]
        tm = ProductMeasure(sampler=DigitalNetB2(2, seed=71), marginals=marginals)

        x = tm(n)

        self.assertEqual(x.shape, (n, 2))
        self.assertTrue(np.all((-1.0 <= x[:, 0]) & (x[:, 0] <= 1.0)))
        self.assertTrue(np.all((0.0 <= x[:, 1]) & (x[:, 1] <= 1.0)))


    def test_matches_equivalent_scipywrapper(self):
        n = 128
        seed = 55
        scipy_marginals = [stats.norm(loc=0.0, scale=1.0), stats.gamma(a=2.0, scale=1.0)]
        product_marginals = [
            SciPyWrapper(DummySampler(1), scipy_marginals[0]),
            SciPyWrapper(DummySampler(1), scipy_marginals[1]),
        ]

        product_samples = ProductMeasure(
            sampler=DigitalNetB2(2, seed=seed),
            marginals=product_marginals,
        )(n)
        scipy_samples = SciPyWrapper(DigitalNetB2(2, seed=seed), scipy_marginals)(n)

        np.testing.assert_array_equal(product_samples, scipy_samples)


    def test_with_gaussian_copula_marginal(self):
        n = 64
        copula = GaussianCopula(
            DummySampler(2),
            marginals=[stats.beta(a=2.0, b=5.0), stats.gamma(a=3.0, scale=1.0)],
            correlation=[[1.0, 0.5], [0.5, 1.0]],
        )
        marginals = [copula, Uniform(DummySampler(1), lower_bound=10.0, upper_bound=12.0)]
        tm = ProductMeasure(sampler=DigitalNetB2(3, seed=81), marginals=marginals)

        x = tm(n)

        self.assertEqual(x.shape, (n, 3))
        self.assertTrue(np.all((0.0 <= x[:, 0]) & (x[:, 0] <= 1.0)))
        self.assertTrue(np.all(x[:, 1] >= 0.0))
        self.assertTrue(np.all((10.0 <= x[:, 2]) & (x[:, 2] <= 12.0)))


    def test_recursive_transform_sampling_supported_but_weights_restricted(self):
        recursive_marginal = Uniform(
            Uniform(DummySampler(1), lower_bound=0.0, upper_bound=1.0),
            lower_bound=2.0,
            upper_bound=4.0,
        )
        direct_marginal = Uniform(DummySampler(1), lower_bound=10.0, upper_bound=12.0)
        tm = ProductMeasure(
            sampler=DigitalNetB2(2, seed=91),
            marginals=[recursive_marginal, direct_marginal],
        )

        x = tm(16)

        self.assertEqual(x.shape, (16, 2))
        self.assertTrue(np.all((2.0 <= x[:, 0]) & (x[:, 0] <= 4.0)))
        self.assertTrue(np.all((10.0 <= x[:, 1]) & (x[:, 1] <= 12.0)))
        with self.assertRaisesRegex(ParameterError, "direct marginal"):
            tm(16, return_weights=True)
