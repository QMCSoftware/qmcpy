import numpy as np
import pytest
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


def test_product_measure_zero_inflated_with_scipy_uniform_shape():
    n = 32
    marginals = [
        ZeroInflatedExpUniform(DummySampler(1), p_zero=0.4, lam=1.5),
        SciPyWrapper(DummySampler(1), stats.uniform(loc=2.0, scale=3.0)),
    ]
    tm = ProductMeasure(sampler=DigitalNetB2(2, seed=23), marginals=marginals)

    x = tm(n)

    assert x.shape == (n, 2)
    assert np.any(x[:, 0] == 0.0)
    assert np.all((2.0 <= x[:, 1]) & (x[:, 1] <= 5.0))


def test_product_measure_replication_shape():
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

    assert x.shape == (r, n, 2)


def test_product_measure_statistics_for_multiple_1d_marginals():
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

    assert tm.mean.shape == (2,)
    assert tm.variance.shape == (2,)
    assert tm.standard_deviation.shape == (2,)
    assert tm.covariance.shape == (2, 2)
    for statistic in ("mean", "variance", "standard_deviation", "covariance"):
        value = getattr(tm, statistic)
        flags = value.data.flags if sp.issparse(value) else value.flags
        assert not flags.writeable


def test_product_measure_normalizes_scalar_1d_statistics():
    marginals = [
        Uniform(DummySampler(1), lower_bound=8.0, upper_bound=12.0),
        Gaussian(DummySampler(1), mean=2.0, covariance=9.0),
    ]
    for marginal in marginals:
        assert isinstance(marginal.mean, float)
        assert isinstance(marginal.variance, float)
        assert isinstance(marginal.standard_deviation, float)

    tm = ProductMeasure(sampler=DigitalNetB2(2, seed=29), marginals=marginals)

    np.testing.assert_allclose(tm.mean, [10.0, 2.0])
    np.testing.assert_allclose(tm.variance, [4.0 / 3.0, 9.0])
    np.testing.assert_allclose(
        tm.standard_deviation, [np.sqrt(4.0 / 3.0), 3.0]
    )
    np.testing.assert_allclose(
        tm.covariance.toarray(), np.diag([4.0 / 3.0, 9.0])
    )
    assert tm.mean.shape == (2,)
    assert tm.variance.shape == (2,)
    assert tm.standard_deviation.shape == (2,)
    assert tm.covariance.shape == (2, 2)


def test_product_measure_statistics_preserve_order_and_covariance_blocks():
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

    assert tm.mean.shape == (3,)
    assert tm.variance.shape == (3,)
    assert tm.standard_deviation.shape == (3,)
    assert covariance.shape == (3, 3)
    assert covariance[:1, 1:].nnz == 0
    assert covariance[1:, :1].nnz == 0


def test_product_measure_mixed_covariance_blocks_remain_sparse():
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

    assert sp.issparse(covariance)
    assert covariance.format == "csr"
    assert covariance.shape == (d + 2, d + 2)
    difference = (covariance - expected).tocsr()
    difference.eliminate_zeros()
    assert difference.nnz == 0
    covariance_csr = covariance.tocsr()
    np.testing.assert_allclose(
        covariance_csr[-2:, -2:].toarray(), [[1.0, 0.5], [0.5, 1.0]]
    )
    assert covariance_csr[:d, d:].nnz == 0
    assert covariance_csr[d:, :d].nnz == 0
    assert not covariance.data.flags.writeable
    with pytest.raises(ValueError):
        covariance.data.setflags(write=True)


def test_product_measure_sparse_covariance_repr_is_compact():
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

    assert "sparse CSR" in representation
    assert "shape=(130, 130)" in representation
    assert "nnz=132" in representation
    assert "Coords" not in representation
    assert "(127, 127)" not in representation


def test_product_measure_statistics_are_lazily_cached():
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
    assert cache["_mean_cache"] is None
    assert cache["_variance_cache"] is None
    assert cache["_standard_deviation_cache"] is None
    assert cache["_covariance_cache"] is None

    mean = tm.mean
    np.testing.assert_allclose(mean, [10.0, 2.0, 5.0])
    assert tm.mean is mean
    assert cache["_variance_cache"] is None
    assert cache["_standard_deviation_cache"] is None
    assert cache["_covariance_cache"] is None

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
    assert tm.variance is variance
    assert tm.standard_deviation is standard_deviation
    assert tm.covariance is covariance


def test_product_measure_dense_covariance_cannot_be_made_writeable():
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

    assert isinstance(covariance, np.ndarray)
    np.testing.assert_allclose(
        covariance,
        [[2.0, 0.0, 0.0], [0.0, 3.0, 0.25], [0.0, 0.25, 4.0]],
    )
    assert not covariance.flags.writeable
    with pytest.raises(ValueError):
        covariance.setflags(write=True)


def test_product_measure_missing_marginal_statistic_is_identified():
    marginals = [
        ZeroInflatedExpUniform(DummySampler(1), p_zero=0.4, lam=1.5),
        Uniform(DummySampler(1), lower_bound=2.0, upper_bound=5.0),
    ]
    tm = ProductMeasure(sampler=DigitalNetB2(2, seed=23), marginals=marginals)

    np.testing.assert_allclose(tm.mean, [0.4, 3.5])
    assert "covariance" not in tm.parameters
    with pytest.raises(
        AttributeError,
        match=r"marginal 0 \(ZeroInflatedExpUniform\) does not provide covariance",
    ):
        _ = tm.covariance


def test_product_measure_marginals_with_different_dimensions():
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

    assert tm.d == 3
    assert np.array_equal(tm.marginal_dimensions, np.array([2, 1]))
    assert x.shape == (n, 3)
    assert np.all(np.isfinite(x[:, :2]))
    assert np.all(x[:, 2] >= 0.0)


def test_product_measure_block_split_range_and_weight_product():
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

    assert x.shape == (n, 3)
    assert np.allclose(tm.range, expected_range)
    assert np.allclose(x, expected)
    assert np.all((10.0 <= x[:, 0]) & (x[:, 0] <= 12.0))
    assert np.all((20.0 <= x[:, 1]) & (x[:, 1] <= 24.0))
    assert np.all((30.0 <= x[:, 2]) & (x[:, 2] <= 36.0))
    assert np.allclose(tm._weight(x), 1.0 / (2.0 * 4.0 * 6.0))
    assert x_call.shape == (n, 3)
    assert np.allclose(jac, 2.0 * 4.0 * 6.0)


def test_product_measure_invalid_inputs():
    with pytest.raises(ParameterError, match="nonempty list of marginals"):
        ProductMeasure(sampler=DigitalNetB2(1, seed=7), marginals=[])

    with pytest.raises(ParameterError, match="marginal"):
        ProductMeasure(sampler=DigitalNetB2(1, seed=7), marginals=[object()])

    with pytest.raises(ParameterError, match="AbstractDiscreteDistribution"):
        ProductMeasure(sampler=object(), marginals=[Uniform(DummySampler(1))])

    marginals = [Uniform(DummySampler(1))]
    with pytest.raises(DimensionError, match="sum of marginal dimensions"):
        ProductMeasure(sampler=DigitalNetB2(2, seed=7), marginals=marginals)


def test_product_measure_rejects_non_dimension_preserving_marginal():
    marginal = AcceptanceRejection(
        DigitalNetB2(2, seed=7),
        lambda x: np.ones(len(x)),
        1.0,
        1.0,
    )

    with pytest.raises(DimensionError, match="dimension-preserving"):
        ProductMeasure(DigitalNetB2(2, seed=11), [marginal])


def test_product_measure_spawn_preserves_marginal_blocks_and_replaces_outer_sampler():
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

    assert isinstance(spawn, ProductMeasure)
    assert spawn.d == 3
    assert spawn.marginals == tm.marginals
    assert spawn.discrete_distrib is not tm.discrete_distrib
    assert np.array_equal(spawn.marginal_dimensions, np.array([1, 2]))

    with pytest.raises(DimensionError):
        tm.spawn(s=1, dimensions=4)


def test_product_measure_does_not_use_marginal_dummy_sampler_values():
    marginals = [
        Uniform(DummySampler(1), lower_bound=0.0, upper_bound=2.0),
        Uniform(DummySampler(1), lower_bound=10.0, upper_bound=12.0),
    ]

    with pytest.raises(ParameterError, match="construction placeholder"):
        marginals[0].discrete_distrib(4)

    tm = ProductMeasure(sampler=DigitalNetB2(2, seed=19), marginals=marginals)
    x = tm(8)

    assert x.shape == (8, 2)
    assert np.all((0.0 <= x[:, 0]) & (x[:, 0] <= 2.0))
    assert np.all((10.0 <= x[:, 1]) & (x[:, 1] <= 12.0))


def test_product_measure_same_outer_seed_matches_different_outer_seed_changes():
    marginals = [
        Uniform(DummySampler(1), lower_bound=0.0, upper_bound=2.0),
        Uniform(DummySampler(1), lower_bound=10.0, upper_bound=12.0),
    ]

    first = ProductMeasure(sampler=DigitalNetB2(2, seed=101), marginals=marginals)(16)
    same_outer = ProductMeasure(sampler=DigitalNetB2(2, seed=101), marginals=marginals)(16)
    different_outer = ProductMeasure(sampler=DigitalNetB2(2, seed=102), marginals=marginals)(16)

    assert np.array_equal(first, same_outer)
    assert not np.array_equal(first, different_outer)


def test_product_measure_replication_means_close_to_uniform_targets():
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

    assert x.shape == (r, n, 2)
    assert np.allclose(replication_means[:, 0], 1.0, atol=0.03)
    assert np.allclose(replication_means[:, 1], 11.0, atol=0.03)


def test_product_measure_with_scipywrapper_beta_marginal():
    n = 64
    marginals = [
        Uniform(DummySampler(1), lower_bound=-1.0, upper_bound=1.0),
        SciPyWrapper(DummySampler(1), stats.beta(a=2.0, b=5.0)),
    ]
    tm = ProductMeasure(sampler=DigitalNetB2(2, seed=71), marginals=marginals)

    x = tm(n)

    assert x.shape == (n, 2)
    assert np.all((-1.0 <= x[:, 0]) & (x[:, 0] <= 1.0))
    assert np.all((0.0 <= x[:, 1]) & (x[:, 1] <= 1.0))


def test_product_measure_matches_equivalent_scipywrapper():
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

    assert np.array_equal(product_samples, scipy_samples)


def test_product_measure_with_gaussian_copula_marginal():
    n = 64
    copula = GaussianCopula(
        DummySampler(2),
        marginals=[stats.beta(a=2.0, b=5.0), stats.gamma(a=3.0, scale=1.0)],
        correlation=[[1.0, 0.5], [0.5, 1.0]],
    )
    marginals = [copula, Uniform(DummySampler(1), lower_bound=10.0, upper_bound=12.0)]
    tm = ProductMeasure(sampler=DigitalNetB2(3, seed=81), marginals=marginals)

    x = tm(n)

    assert x.shape == (n, 3)
    assert np.all((0.0 <= x[:, 0]) & (x[:, 0] <= 1.0))
    assert np.all(x[:, 1] >= 0.0)
    assert np.all((10.0 <= x[:, 2]) & (x[:, 2] <= 12.0))


def test_product_measure_recursive_transform_sampling_supported_but_weights_restricted():
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

    assert x.shape == (16, 2)
    assert np.all((2.0 <= x[:, 0]) & (x[:, 0] <= 4.0))
    assert np.all((10.0 <= x[:, 1]) & (x[:, 1] <= 12.0))
    with pytest.raises(ParameterError, match="direct marginal"):
        tm(16, return_weights=True)
