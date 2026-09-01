import warnings

import numpy as np
import pytest
import scipy.stats as stats

from qmcpy import (
    AbstractCopula,
    ClaytonCopula,
    DigitalNetB2,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    StudentTCopula,
)

from qmcpy.true_measure.copula import (
    AbstractCopula as ModuleAbstractCopula,
    _apply_marginal_ppfs,
    _build_marginal_range,
    _clip_unit_interval,
    _marginal_cdfs_and_logpdf,
    _validate_correlation_matrix,
    _validate_dimension,
    _validate_marginals,
)

from qmcpy.util import DimensionError, MethodImplementationError, ParameterError


class PPFOnlyMarginal:
    def ppf(self, u):
        return np.asarray(u, dtype=float)


class NonCallablePPFMarginal:
    ppf = 1.0


class UnitPDFMarginal:
    def ppf(self, u):
        return np.asarray(u, dtype=float)

    def cdf(self, x):
        return np.asarray(x, dtype=float)

    def pdf(self, x):
        return np.ones_like(np.asarray(x, dtype=float))


class CDFOnlyMarginal(PPFOnlyMarginal):
    def cdf(self, x):
        return np.asarray(x, dtype=float)


class BadIntervalMarginal(PPFOnlyMarginal):
    def interval(self, confidence):
        raise ValueError("interval unavailable")


class BadRangeMarginal:
    def ppf(self, u):
        raise ValueError("ppf unavailable")


def _equicorrelation(d, rho):
    corr = np.full((d, d), rho, dtype=float)
    np.fill_diagonal(corr, 1.0)
    return corr


def _make_copula(copula_cls, dimension=2, marginals=None, correlation=None, seed=7):
    if marginals is None:
        marginals = [stats.norm()] * dimension
    if correlation is None:
        correlation = np.eye(dimension)

    kwargs = {}
    if copula_cls is StudentTCopula:
        kwargs["df"] = 4
    if copula_cls is ClaytonCopula:
        kwargs["theta"] = 2.0
    if copula_cls is FrankCopula:
        kwargs["theta"] = 5.0
    if copula_cls is GumbelCopula:
        kwargs["theta"] = 2.0

    common = {
        "sampler": DigitalNetB2(dimension, seed=seed),
        "marginals": marginals,
        **kwargs,
    }
    if copula_cls in [ClaytonCopula, FrankCopula, GumbelCopula]:
        return copula_cls(**common)
    return copula_cls(correlation=correlation, **common)


# Base AbstractCopula and helper tests


def test_abstract_copula_is_importable_from_public_module_path():
    assert ModuleAbstractCopula is AbstractCopula


def test_public_api_imports_and_normal_usage():
    for copula_cls in [
        GaussianCopula,
        StudentTCopula,
        ClaytonCopula,
        FrankCopula,
        GumbelCopula,
    ]:
        assert issubclass(copula_cls, AbstractCopula)

        tm = _make_copula(copula_cls)
        x = tm(8)
        x_gen = tm.gen_samples(8)
        v = tm.gen_copula_samples(8)

        assert x.shape == (8, 2)
        assert x_gen.shape == (8, 2)
        assert v.shape == (8, 2)
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(x_gen))
        assert np.all((0 <= v) & (v <= 1))


def test_abstract_copula_rejects_unimplemented_transform():
    tm = AbstractCopula(
        DigitalNetB2(2, seed=101),
        marginals=[stats.uniform(), stats.uniform()],
    )

    with pytest.raises(MethodImplementationError):
        tm.copula_transform(np.full((3, 2), 0.5))


def test_abstract_copula_rejects_invalid_sampler():
    with pytest.raises(ParameterError, match="sampler"):
        AbstractCopula(object(), marginals=[stats.uniform()])


def test_validate_marginals_error_branches():
    with pytest.raises(ParameterError, match="marginals"):
        _validate_marginals(None)

    with pytest.raises(ParameterError, match="at least one"):
        _validate_marginals([])

    with pytest.raises(ParameterError, match="ppf"):
        _validate_marginals([NonCallablePPFMarginal()])


def test_validate_dimension_error_branches():
    with pytest.raises(DimensionError, match="integer dimension"):
        _validate_dimension(object(), [stats.uniform()])

    with pytest.raises(DimensionError, match="marginals"):
        _validate_dimension(3, [stats.uniform(), stats.uniform()])


def test_apply_marginal_ppfs_clips_endpoints_and_checks_dimension():
    transformed = _apply_marginal_ppfs(
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        [stats.norm(), stats.norm()],
    )

    assert transformed.shape == (2, 2)
    assert np.all(np.isfinite(transformed))

    with pytest.raises(DimensionError, match="marginals"):
        _apply_marginal_ppfs(np.full((2, 3), 0.5), [stats.uniform(), stats.uniform()])


def test_marginal_range_falls_back_when_interval_or_ppf_fails():
    ranges = _build_marginal_range([BadIntervalMarginal(), BadRangeMarginal()])

    assert ranges.shape == (2, 2)
    assert np.all(np.isfinite(ranges[0]))
    np.testing.assert_allclose(ranges[1], [-np.inf, np.inf])


def test_marginal_cdfs_and_logpdf_pdf_branch_and_errors():
    x = np.array([[0.25, 0.75], [0.4, 0.6]])
    u, log_density = _marginal_cdfs_and_logpdf(
        x,
        [UnitPDFMarginal(), UnitPDFMarginal()],
    )

    np.testing.assert_allclose(u, x)
    np.testing.assert_allclose(log_density, np.zeros(2))

    with pytest.raises(ParameterError, match="cdf"):
        _marginal_cdfs_and_logpdf(x, [PPFOnlyMarginal(), UnitPDFMarginal()])

    with pytest.raises(ParameterError, match="pdf"):
        _marginal_cdfs_and_logpdf(x, [CDFOnlyMarginal(), UnitPDFMarginal()])


def test_validate_correlation_matrix_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        _validate_correlation_matrix([[1.0, np.nan], [np.nan, 1.0]], 2)


def test_clip_unit_interval_uses_machine_epsilon():
    clipped = _clip_unit_interval(np.array([0.0, 0.5, 1.0]))
    eps = np.finfo(float).eps

    np.testing.assert_allclose(clipped, [eps, 0.5, 1.0 - eps])


@pytest.mark.parametrize(
    "copula_cls",
    [GaussianCopula, StudentTCopula, ClaytonCopula, GumbelCopula, FrankCopula],
)
def test_copula_transform_outputs_dependent_uniforms_in_unit_cube(copula_cls):
    tm = _make_copula(copula_cls, dimension=3)
    u = np.array(
        [
            [0.1, 0.3, 0.7],
            [0.5, 0.5, 0.5],
            [0.9, 0.8, 0.2],
        ]
    )

    v = tm.copula_transform(u)

    assert v.shape == u.shape
    assert np.all(np.isfinite(v))
    assert np.all((0.0 <= v) & (v <= 1.0))


@pytest.mark.parametrize(
    "copula_cls,dimension",
    [
        (GaussianCopula, 3),
        (StudentTCopula, 3),
        (ClaytonCopula, 3),
        (FrankCopula, 3),
        (GumbelCopula, 3),
    ],
)
def test_copula_sample_shapes_are_preserved(copula_cls, dimension):
    tm = _make_copula(copula_cls, dimension=dimension, seed=9)

    one = tm(1)
    many = tm(8)
    batched_transform = tm._transform(np.full((2, 3, dimension), 0.5))

    assert one.shape == (1, dimension)
    assert many.shape == (8, dimension)
    assert batched_transform.shape == (2, 3, dimension)
    assert np.all(np.isfinite(one))
    assert np.all(np.isfinite(many))
    assert np.all(np.isfinite(batched_transform))


# Elliptical copulas


def test_output_shape_with_nonnormal_marginals():
    tm = GaussianCopula(
        sampler=DigitalNetB2(2, seed=7),
        marginals=[stats.beta(a=2, b=5), stats.gamma(a=3, scale=2)],
        correlation=[[1.0, 0.4], [0.4, 1.0]],
    )

    x = tm(16)

    assert x.shape == (16, 2)


def test_finite_output_for_normal_marginals():
    tm = GaussianCopula(
        sampler=DigitalNetB2(2, seed=11),
        marginals=[stats.norm(), stats.norm(loc=1.0, scale=2.0)],
        correlation=[[1.0, -0.3], [-0.3, 1.0]],
    )

    x = tm(128)

    assert np.all(np.isfinite(x))


def test_return_weights_shape_when_marginal_densities_available():
    tm = GaussianCopula(
        sampler=DigitalNetB2(2, seed=12),
        marginals=[stats.norm(), stats.gamma(a=2.0)],
        correlation=[[1.0, 0.25], [0.25, 1.0]],
    )

    x, weights = tm(32, return_weights=True)

    assert x.shape == (32, 2)
    assert weights.shape == (32,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)


def test_identity_correlation_matches_independent_marginal_transforms():
    marginals = [stats.norm(loc=-1.0, scale=2.0), stats.gamma(a=2.0, scale=3.0)]
    tm = GaussianCopula(
        sampler=DigitalNetB2(2, seed=13),
        marginals=marginals,
        correlation=np.eye(2),
    )
    u = np.array([[0.2, 0.7], [0.4, 0.8], [0.9, 0.1]])

    x = tm._transform(u)
    expected = np.column_stack(
        [marginals[j].ppf(u[:, j]) for j in range(len(marginals))]
    )

    np.testing.assert_allclose(x, expected, rtol=1e-12, atol=1e-12)


def test_positive_correlation_produces_positive_dependence():
    rho = 0.75
    tm = GaussianCopula(
        sampler=DigitalNetB2(2, seed=17),
        marginals=[stats.norm(), stats.norm()],
        correlation=[[1.0, rho], [rho, 1.0]],
    )

    x = tm(4096)
    empirical_corr = np.corrcoef(x.T)[0, 1]

    assert empirical_corr > 0.5
    assert abs(empirical_corr - rho) < 0.2


@pytest.mark.parametrize("copula_cls", [GaussianCopula, StudentTCopula])
@pytest.mark.parametrize("dimension", [1, 3, 5])
def test_elliptical_copulas_support_general_dimensions(copula_cls, dimension):
    correlation = _equicorrelation(dimension, 0.25)
    tm = _make_copula(
        copula_cls,
        dimension=dimension,
        marginals=[stats.norm()] * dimension,
        correlation=correlation,
        seed=19,
    )

    x = tm(16)
    one = tm(1)

    assert x.shape == (16, dimension)
    assert one.shape == (1, dimension)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(one))


@pytest.mark.parametrize("copula_cls", [GaussianCopula, StudentTCopula])
def test_elliptical_copulas_handle_valid_near_singular_correlation(copula_cls):
    dimension = 5
    tm = _make_copula(
        copula_cls,
        dimension=dimension,
        marginals=[stats.norm()] * dimension,
        correlation=_equicorrelation(dimension, 0.999),
        seed=20,
    )

    x = tm(32)

    assert x.shape == (32, dimension)
    assert np.all(np.isfinite(x))


@pytest.mark.parametrize("copula_cls", [GaussianCopula, StudentTCopula])
def test_elliptical_copulas_reject_singular_correlation(copula_cls):
    with pytest.raises(ValueError, match="positive definite"):
        _make_copula(
            copula_cls,
            dimension=3,
            marginals=[stats.norm(), stats.norm(), stats.norm()],
            correlation=np.ones((3, 3)),
            seed=22,
        )


@pytest.mark.parametrize(
    "copula_cls",
    [GaussianCopula, StudentTCopula, ClaytonCopula, FrankCopula, GumbelCopula],
)
def test_distribution_dimension_matches_number_of_marginals(copula_cls):
    tm = _make_copula(
        copula_cls,
        dimension=5,
        marginals=[
            stats.norm(),
            stats.beta(a=2, b=5),
            stats.gamma(a=3),
            stats.expon(),
            stats.lognorm(s=0.5),
        ],
        correlation=np.eye(5),
    )

    x = tm(32)

    assert x.shape == (32, 5)
    assert np.all(np.isfinite(x))


@pytest.mark.parametrize("copula_cls", [GaussianCopula, StudentTCopula])
def test_invalid_dimension_mismatches_raise(copula_cls):
    with pytest.raises(DimensionError, match="marginals"):
        _make_copula(
            copula_cls,
            dimension=2,
            marginals=[stats.norm(), stats.norm(), stats.norm()],
            correlation=np.eye(2),
        )

    with pytest.raises(ValueError, match="shape"):
        _make_copula(
            copula_cls,
            dimension=2,
            marginals=[stats.norm(), stats.norm()],
            correlation=np.eye(3),
        )

    with pytest.raises(ValueError, match="square"):
        _make_copula(
            copula_cls,
            dimension=2,
            marginals=[stats.norm(), stats.norm()],
            correlation=[[1.0, 0.2, 0.3], [0.2, 1.0, 0.4]],
        )


@pytest.mark.parametrize("copula_cls", [ClaytonCopula, FrankCopula, GumbelCopula])
def test_archimedean_dimension_mismatch_raises_dimension_error(copula_cls):
    with pytest.raises(DimensionError, match="marginals"):
        _make_copula(
            copula_cls,
            dimension=2,
            marginals=[stats.norm(), stats.norm(), stats.norm()],
        )


@pytest.mark.parametrize(
    "copula_cls",
    [GaussianCopula, StudentTCopula],
)
@pytest.mark.parametrize(
    "correlation",
    [
        [[1.0, 0.2], [0.3, 1.0]],
        [[1.0, 0.2], [0.2, 0.9]],
        [[1.0, 1.2], [1.2, 1.0]],
    ],
)
def test_invalid_correlation_matrices_raise_value_error(copula_cls, correlation):
    with pytest.raises(ValueError):
        _make_copula(
            copula_cls,
            dimension=2,
            marginals=[stats.norm(), stats.norm()],
            correlation=correlation,
        )


def test_marginal_length_mismatch_raises_dimension_error():
    with pytest.raises(DimensionError, match="marginals"):
        GaussianCopula(
            sampler=DigitalNetB2(2, seed=21),
            marginals=[stats.norm()],
            correlation=np.eye(2),
        )


def test_marginal_without_ppf_raises_clear_error():
    class NoPPF:
        pass

    with pytest.raises(ParameterError, match="ppf"):
        GaussianCopula(
            sampler=DigitalNetB2(1, seed=23),
            marginals=[NoPPF()],
            correlation=[[1.0]],
        )


@pytest.mark.parametrize(
    "copula_cls",
    [GaussianCopula, StudentTCopula, ClaytonCopula, FrankCopula, GumbelCopula],
)
def test_common_scipy_frozen_marginals_work(copula_cls):
    tm = _make_copula(
        copula_cls,
        dimension=5,
        marginals=[
            stats.norm(),
            stats.beta(a=2, b=5),
            stats.gamma(a=3),
            stats.expon(),
            stats.lognorm(s=0.5),
        ],
        correlation=np.eye(5),
        seed=47,
    )

    x = tm(128)

    assert x.shape == (128, 5)
    assert np.all(np.isfinite(x))


@pytest.mark.parametrize(
    "copula_cls",
    [GaussianCopula, StudentTCopula, ClaytonCopula, FrankCopula, GumbelCopula],
)
def test_endpoint_uniforms_are_clipped_to_finite_outputs(copula_cls):
    tm = _make_copula(
        copula_cls,
        dimension=5,
        marginals=[
            stats.norm(),
            stats.beta(a=2, b=5),
            stats.gamma(a=3),
            stats.expon(),
            stats.lognorm(s=0.5),
        ],
        correlation=np.eye(5),
        seed=53,
    )
    u = np.array(
        [
            [0.0, 1.0, 0.0, 1.0, 0.5],
            [1.0, 0.0, 1.0, 0.0, 0.5],
        ]
    )

    x = tm._transform(u)

    assert x.shape == (2, 5)
    assert np.all(np.isfinite(x))


def test_student_t_copula_output_shape_and_finite_values():
    tm = StudentTCopula(
        sampler=DigitalNetB2(2, seed=29),
        marginals=[stats.norm(), stats.gamma(a=3.0, scale=2.0)],
        correlation=[[1.0, 0.5], [0.5, 1.0]],
        df=4,
    )

    x = tm(128)

    assert x.shape == (128, 2)
    assert np.all(np.isfinite(x))


def test_student_t_copula_positive_correlation_produces_positive_dependence():
    tm = StudentTCopula(
        sampler=DigitalNetB2(2, seed=31),
        marginals=[stats.norm(), stats.norm()],
        correlation=[[1.0, 0.7], [0.7, 1.0]],
        df=5,
    )

    x = tm(4096)
    empirical_corr = np.corrcoef(x.T)[0, 1]

    assert empirical_corr > 0.45


def test_student_t_copula_has_stronger_joint_tail_than_gaussian_copula():
    rho = 0.7
    df = 4
    n = 2**12
    marginals = [stats.norm(), stats.norm()]
    correlation = [[1.0, rho], [rho, 1.0]]

    gaussian = GaussianCopula(
        sampler=DigitalNetB2(2, seed=101),
        marginals=marginals,
        correlation=correlation,
    )
    student_t = StudentTCopula(
        sampler=DigitalNetB2(2, seed=101),
        marginals=marginals,
        correlation=correlation,
        df=df,
    )

    x_gaussian = gaussian(n)
    x_student_t = student_t(n)
    threshold = stats.norm.ppf(0.99)

    def joint_tail_rate(x):
        tail_0 = x[:, 0] > threshold
        return np.mean(x[tail_0, 1] > threshold)

    gaussian_tail = joint_tail_rate(x_gaussian)
    student_t_tail = joint_tail_rate(x_student_t)

    assert student_t_tail > gaussian_tail + 0.08


def test_student_t_copula_return_weights_shape_when_density_available():
    tm = StudentTCopula(
        sampler=DigitalNetB2(2, seed=37),
        marginals=[stats.norm(), stats.gamma(a=2.0)],
        correlation=[[1.0, 0.3], [0.3, 1.0]],
        df=6,
    )

    x, weights = tm(32, return_weights=True)

    assert x.shape == (32, 2)
    assert weights.shape == (32,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)


@pytest.mark.parametrize("df", [1.0, 100.0])
def test_student_t_copula_boundary_df_values_are_finite(df):
    dimension = 3
    tm = StudentTCopula(
        sampler=DigitalNetB2(dimension, seed=39),
        marginals=[stats.norm()] * dimension,
        correlation=_equicorrelation(dimension, 0.4),
        df=df,
    )

    x = tm(128)

    assert x.shape == (128, dimension)
    assert np.all(np.isfinite(x))


def test_student_t_copula_large_df_is_close_to_gaussian_copula():
    rho = 0.6
    correlation = [[1.0, rho], [rho, 1.0]]
    marginals = [stats.norm(), stats.norm()]
    gaussian = GaussianCopula(
        sampler=DigitalNetB2(2, seed=40),
        marginals=marginals,
        correlation=correlation,
    )
    student_t = StudentTCopula(
        sampler=DigitalNetB2(2, seed=40),
        marginals=marginals,
        correlation=correlation,
        df=100,
    )

    x_gaussian = gaussian(4096)
    x_student_t = student_t(4096)
    corr_gaussian = np.corrcoef(x_gaussian.T)[0, 1]
    corr_student_t = np.corrcoef(x_student_t.T)[0, 1]

    assert abs(corr_student_t - corr_gaussian) < 0.02


@pytest.mark.parametrize("df", [0, -1, np.inf, "not-a-number"])
def test_student_t_copula_invalid_df_raises_parameter_error(df):
    with pytest.raises(ParameterError, match="df"):
        StudentTCopula(
            sampler=DigitalNetB2(2, seed=41),
            marginals=[stats.norm(), stats.norm()],
            correlation=np.eye(2),
            df=df,
        )


def test_student_t_copula_marginal_without_ppf_raises_clear_error():
    class NoPPF:
        pass

    with pytest.raises(ParameterError, match="ppf"):
        StudentTCopula(
            sampler=DigitalNetB2(1, seed=43),
            marginals=[NoPPF()],
            correlation=[[1.0]],
            df=4,
        )


# Archimedean copulas


def test_clayton_copula_output_shape_and_finite_values():
    tm = ClaytonCopula(
        sampler=DigitalNetB2(2, seed=57),
        marginals=[stats.norm(), stats.gamma(a=3.0, scale=2.0)],
        theta=2.0,
    )

    x = tm(128)

    assert x.shape == (128, 2)
    assert np.all(np.isfinite(x))


def test_clayton_copula_return_weights_shape_when_density_available():
    tm = ClaytonCopula(
        sampler=DigitalNetB2(3, seed=59),
        marginals=[stats.norm(), stats.gamma(a=2.0), stats.expon()],
        theta=1.5,
    )

    x, weights = tm(32, return_weights=True)

    assert x.shape == (32, 3)
    assert weights.shape == (32,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)


@pytest.mark.parametrize("theta", [0, -1, np.inf, "not-a-number"])
def test_clayton_copula_invalid_theta_raises_parameter_error(theta):
    with pytest.raises(ParameterError, match="theta"):
        ClaytonCopula(
            sampler=DigitalNetB2(2, seed=61),
            marginals=[stats.norm(), stats.norm()],
            theta=theta,
        )


@pytest.mark.parametrize("dimension", [2, 3, 5])
def test_clayton_copula_supports_general_dimension(dimension):
    tm = ClaytonCopula(
        sampler=DigitalNetB2(dimension, seed=63),
        marginals=[stats.norm()] * dimension,
        theta=2.0,
    )

    x = tm(128)

    assert x.shape == (128, dimension)
    assert np.all(np.isfinite(x))


def test_clayton_copula_marginal_without_ppf_raises_clear_error():
    class NoPPF:
        pass

    with pytest.raises(ParameterError, match="ppf"):
        ClaytonCopula(
            sampler=DigitalNetB2(2, seed=67),
            marginals=[stats.norm(), NoPPF()],
            theta=2.0,
        )


@pytest.mark.parametrize(
    "marginals",
    [
        [stats.norm(), stats.beta(a=2, b=5)],
        [stats.gamma(a=3), stats.expon()],
        [stats.lognorm(s=0.5), stats.norm()],
    ],
)
def test_clayton_copula_common_scipy_frozen_marginals_work(marginals):
    tm = ClaytonCopula(
        sampler=DigitalNetB2(2, seed=69),
        marginals=marginals,
        theta=2.0,
    )

    x = tm(128)

    assert x.shape == (128, 2)
    assert np.all(np.isfinite(x))


def test_clayton_copula_endpoint_uniforms_are_clipped_to_finite_outputs():
    tm = ClaytonCopula(
        sampler=DigitalNetB2(2, seed=70),
        marginals=[stats.norm(), stats.lognorm(s=0.5)],
        theta=2.0,
    )
    u = np.array([[0.0, 1.0], [1.0, 0.0]])

    x = tm._transform(u)

    assert x.shape == (2, 2)
    assert np.all(np.isfinite(x))


@pytest.mark.parametrize("dimension", [2, 3, 5])
def test_clayton_copula_tiny_theta_is_near_independent(dimension):
    marginals = [stats.uniform()] * dimension
    tm = ClaytonCopula(
        sampler=DigitalNetB2(dimension, seed=70),
        marginals=marginals,
        theta=1e-8,
    )
    u = np.array(
        [
            [0.2, 0.7, 0.4, 0.6, 0.8],
            [0.4, 0.8, 0.9, 0.3, 0.2],
            [0.9, 0.1, 0.3, 0.7, 0.5],
        ]
    )[:, :dimension]

    x = tm._transform(u)

    assert x.shape == (3, dimension)
    assert np.all(np.isfinite(x))
    np.testing.assert_allclose(x, u, atol=5e-6)


@pytest.mark.parametrize("dimension", [2, 3, 5])
@pytest.mark.parametrize("theta", [20.0, 50.0])
def test_clayton_copula_large_theta_is_finite(theta, dimension):
    tm = ClaytonCopula(
        sampler=DigitalNetB2(dimension, seed=70),
        marginals=[stats.norm()] * dimension,
        theta=theta,
    )

    x = tm(128)

    assert x.shape == (128, dimension)
    assert np.all(np.isfinite(x))


def test_clayton_copula_positive_dependence_behavior():
    tm = ClaytonCopula(
        sampler=DigitalNetB2(2, seed=71),
        marginals=[stats.uniform(), stats.uniform()],
        theta=2.0,
    )

    x = tm(4096)
    empirical_corr = np.corrcoef(x.T)[0, 1]

    assert empirical_corr > 0.45


def test_clayton_copula_has_stronger_lower_tail_than_gaussian_copula():
    theta = 2.0
    n = 2**12
    marginals = [stats.uniform(), stats.uniform()]
    # Clayton Kendall tau is theta/(theta+2); convert to Gaussian rho.
    rho = np.sin(np.pi * (theta / (theta + 2.0)) / 2.0)

    clayton = ClaytonCopula(
        sampler=DigitalNetB2(2, seed=73),
        marginals=marginals,
        theta=theta,
    )
    gaussian = GaussianCopula(
        sampler=DigitalNetB2(2, seed=73),
        marginals=marginals,
        correlation=[[1.0, rho], [rho, 1.0]],
    )

    x_clayton = clayton(n)
    x_gaussian = gaussian(n)
    threshold = 0.05

    def lower_tail_rate(x):
        tail_0 = x[:, 0] < threshold
        return np.mean(x[tail_0, 1] < threshold)

    clayton_tail = lower_tail_rate(x_clayton)
    gaussian_tail = lower_tail_rate(x_gaussian)

    assert clayton_tail > gaussian_tail + 0.2


def test_frank_copula_output_shape_for_two_dimensions():
    tm = FrankCopula(
        sampler=DigitalNetB2(2, seed=75),
        marginals=[stats.norm(), stats.gamma(a=3.0, scale=2.0)],
        theta=5.0,
    )

    x = tm(128)

    assert x.shape == (128, 2)
    assert np.all(np.isfinite(x))


@pytest.mark.parametrize("dimension", [3, 5])
def test_frank_copula_positive_theta_supports_higher_dimensions(dimension):
    tm = FrankCopula(
        sampler=DigitalNetB2(dimension, seed=76),
        marginals=[stats.norm()] * dimension,
        theta=5.0,
    )

    x = tm(128)

    assert x.shape == (128, dimension)
    assert np.all(np.isfinite(x))


def test_frank_copula_return_weights_shape_when_density_available():
    tm = FrankCopula(
        sampler=DigitalNetB2(3, seed=77),
        marginals=[stats.norm(), stats.gamma(a=2.0), stats.expon()],
        theta=4.0,
    )

    x, weights = tm(32, return_weights=True)

    assert x.shape == (32, 3)
    assert weights.shape == (32,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)


@pytest.mark.parametrize("theta", [0, np.inf, -np.inf, "not-a-number"])
def test_frank_copula_invalid_theta_raises_parameter_error(theta):
    with pytest.raises(ParameterError, match="theta"):
        FrankCopula(
            sampler=DigitalNetB2(2, seed=78),
            marginals=[stats.norm(), stats.norm()],
            theta=theta,
        )


def test_frank_copula_negative_theta_rejected_above_two_dimensions():
    with pytest.raises(ParameterError, match="d=2"):
        FrankCopula(
            sampler=DigitalNetB2(3, seed=79),
            marginals=[stats.norm(), stats.norm(), stats.norm()],
            theta=-2.0,
        )


def test_frank_copula_dimension_mismatch_raises_dimension_error():
    with pytest.raises(DimensionError, match="marginals"):
        FrankCopula(
            sampler=DigitalNetB2(2, seed=80),
            marginals=[stats.norm(), stats.norm(), stats.norm()],
            theta=5.0,
        )


def test_frank_copula_marginal_without_ppf_raises_clear_error():
    class NoPPF:
        pass

    with pytest.raises(ParameterError, match="ppf"):
        FrankCopula(
            sampler=DigitalNetB2(2, seed=82),
            marginals=[stats.norm(), NoPPF()],
            theta=5.0,
        )


def test_frank_copula_positive_dependence_behavior():
    tm = FrankCopula(
        sampler=DigitalNetB2(2, seed=84),
        marginals=[stats.uniform(), stats.uniform()],
        theta=6.0,
    )

    x = tm(4096)
    empirical_corr = np.corrcoef(x.T)[0, 1]

    assert empirical_corr > 0.45


@pytest.mark.parametrize(
    "theta,dimension",
    [
        (1e-8, 3),
        (-1e-8, 2),
    ],
)
def test_frank_copula_tiny_theta_is_close_to_independence(theta, dimension):
    marginals = [stats.uniform()] * dimension
    tm = FrankCopula(
        sampler=DigitalNetB2(dimension, seed=86),
        marginals=marginals,
        theta=theta,
    )
    u = np.array(
        [
            [0.2, 0.7, 0.4, 0.6, 0.8],
            [0.4, 0.8, 0.9, 0.3, 0.2],
            [0.9, 0.1, 0.3, 0.7, 0.5],
        ]
    )[:, :dimension]

    x = tm._transform(u)

    assert x.shape == (3, dimension)
    assert np.all(np.isfinite(x))
    np.testing.assert_allclose(x, u, atol=5e-6)


@pytest.mark.parametrize(
    "theta,dimension",
    [
        (50.0, 5),
        (-50.0, 2),
    ],
)
def test_frank_copula_large_theta_is_finite(theta, dimension):
    tm = FrankCopula(
        sampler=DigitalNetB2(dimension, seed=87),
        marginals=[stats.norm()] * dimension,
        theta=theta,
    )

    x = tm(128)

    assert x.shape == (128, dimension)
    assert np.all(np.isfinite(x))


def test_frank_copula_negative_theta_produces_negative_dependence_in_2d():
    tm = FrankCopula(
        sampler=DigitalNetB2(2, seed=88),
        marginals=[stats.uniform(), stats.uniform()],
        theta=-6.0,
    )

    x = tm(4096)
    empirical_corr = np.corrcoef(x.T)[0, 1]

    assert empirical_corr < -0.35


def test_gumbel_copula_output_shape_and_finite_values():
    tm = GumbelCopula(
        sampler=DigitalNetB2(2, seed=79),
        marginals=[stats.norm(), stats.gamma(a=3.0, scale=2.0)],
        theta=2.0,
    )

    x = tm(128)

    assert x.shape == (128, 2)
    assert np.all(np.isfinite(x))


def test_gumbel_copula_return_weights_shape_when_density_available():
    tm = GumbelCopula(
        sampler=DigitalNetB2(3, seed=81),
        marginals=[stats.norm(), stats.gamma(a=2.0), stats.expon()],
        theta=1.5,
    )

    x, weights = tm(32, return_weights=True)

    assert x.shape == (32, 3)
    assert weights.shape == (32,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)


@pytest.mark.parametrize("theta", [0, 0.5, -1, np.inf, "not-a-number"])
def test_gumbel_copula_invalid_theta_raises_parameter_error(theta):
    with pytest.raises(ParameterError, match="theta"):
        GumbelCopula(
            sampler=DigitalNetB2(2, seed=83),
            marginals=[stats.norm(), stats.norm()],
            theta=theta,
        )


def test_gumbel_copula_theta_one_is_independent_marginal_transform():
    marginals = [stats.norm(loc=-1.0, scale=2.0), stats.gamma(a=2.0, scale=3.0)]
    tm = GumbelCopula(
        sampler=DigitalNetB2(2, seed=85),
        marginals=marginals,
        theta=1.0,
    )
    u = np.array([[0.2, 0.7], [0.4, 0.8], [0.9, 0.1]])

    x = tm._transform(u)
    expected = np.column_stack(
        [marginals[j].ppf(u[:, j]) for j in range(len(marginals))]
    )

    np.testing.assert_allclose(x, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dimension", [2, 3, 5])
def test_gumbel_copula_theta_close_to_one_is_near_independent(dimension):
    marginals = [stats.uniform()] * dimension
    tm = GumbelCopula(
        sampler=DigitalNetB2(dimension, seed=85),
        marginals=marginals,
        theta=1.000001,
    )
    u = np.array(
        [
            [0.2, 0.7, 0.4, 0.6, 0.8],
            [0.4, 0.8, 0.9, 0.3, 0.2],
            [0.9, 0.1, 0.3, 0.7, 0.5],
        ]
    )[:, :dimension]

    x = tm._transform(u)

    assert x.shape == (3, dimension)
    assert np.all(np.isfinite(x))
    np.testing.assert_allclose(x, u, atol=5e-5)


@pytest.mark.parametrize("dimension", [2, 3, 5])
@pytest.mark.parametrize("theta", [20.0, 50.0])
def test_gumbel_copula_large_theta_is_finite(theta, dimension):
    tm = GumbelCopula(
        sampler=DigitalNetB2(dimension, seed=86),
        marginals=[stats.norm()] * dimension,
        theta=theta,
    )

    x = tm(128)

    assert x.shape == (128, dimension)
    assert np.all(np.isfinite(x))


@pytest.mark.parametrize("dimension", [2, 3, 5])
def test_gumbel_copula_supports_general_dimension(dimension):
    tm = GumbelCopula(
        sampler=DigitalNetB2(dimension, seed=87),
        marginals=[stats.norm()] * dimension,
        theta=2.0,
    )

    x = tm(128)

    assert x.shape == (128, dimension)
    assert np.all(np.isfinite(x))


def test_gumbel_copula_marginal_without_ppf_raises_clear_error():
    class NoPPF:
        pass

    with pytest.raises(ParameterError, match="ppf"):
        GumbelCopula(
            sampler=DigitalNetB2(2, seed=89),
            marginals=[stats.norm(), NoPPF()],
            theta=2.0,
        )


@pytest.mark.parametrize(
    "marginals",
    [
        [stats.norm(), stats.beta(a=2, b=5)],
        [stats.gamma(a=3), stats.expon()],
        [stats.lognorm(s=0.5), stats.norm()],
    ],
)
def test_gumbel_copula_common_scipy_frozen_marginals_work(marginals):
    tm = GumbelCopula(
        sampler=DigitalNetB2(2, seed=91),
        marginals=marginals,
        theta=2.0,
    )

    x = tm(128)

    assert x.shape == (128, 2)
    assert np.all(np.isfinite(x))


def test_gumbel_copula_endpoint_uniforms_are_clipped_to_finite_outputs():
    tm = GumbelCopula(
        sampler=DigitalNetB2(2, seed=93),
        marginals=[stats.norm(), stats.lognorm(s=0.5)],
        theta=2.0,
    )
    u = np.array([[0.0, 1.0], [1.0, 0.0]])

    x = tm._transform(u)

    assert x.shape == (2, 2)
    assert np.all(np.isfinite(x))


def test_gumbel_copula_positive_dependence_behavior():
    tm = GumbelCopula(
        sampler=DigitalNetB2(2, seed=95),
        marginals=[stats.uniform(), stats.uniform()],
        theta=2.0,
    )

    x = tm(4096)
    empirical_corr = np.corrcoef(x.T)[0, 1]

    assert empirical_corr > 0.45


def test_gumbel_copula_has_stronger_upper_tail_than_gaussian_copula():
    theta = 2.0
    n = 2**12
    marginals = [stats.uniform(), stats.uniform()]
    # Gumbel Kendall tau is 1 - 1/theta; convert to Gaussian rho.
    rho = np.sin(np.pi * (1.0 - 1.0 / theta) / 2.0)

    gumbel = GumbelCopula(
        sampler=DigitalNetB2(2, seed=97),
        marginals=marginals,
        theta=theta,
    )
    gaussian = GaussianCopula(
        sampler=DigitalNetB2(2, seed=97),
        marginals=marginals,
        correlation=[[1.0, rho], [rho, 1.0]],
    )

    x_gumbel = gumbel(n)
    x_gaussian = gaussian(n)
    threshold = 0.95

    def upper_tail_rate(x):
        tail_0 = x[:, 0] > threshold
        return np.mean(x[tail_0, 1] > threshold)

    gumbel_tail = upper_tail_rate(x_gumbel)
    gaussian_tail = upper_tail_rate(x_gaussian)

    assert gumbel_tail > gaussian_tail + 0.15


# Weights, fallback behavior, spawn, and edge cases


@pytest.mark.parametrize(
    "copula_cls",
    [GaussianCopula, StudentTCopula, ClaytonCopula, GumbelCopula, FrankCopula],
)
def test_copula_weight_fallback_warns_once_when_density_methods_are_missing(
    copula_cls,
):
    tm = _make_copula(
        copula_cls,
        dimension=2,
        marginals=[PPFOnlyMarginal(), PPFOnlyMarginal()],
    )
    x = np.full((4, 2), 0.5)
    expected_message = getattr(
        tm,
        "_missing_weight_warning_message",
        f"{copula_cls.__name__} marginals must implement 'cdf' and "
        "'pdf' or 'logpdf' to compute density weights. "
        "Weights will be treated as 1.",
    )

    assert "_unit_weight_with_warning" not in copula_cls.__dict__
    assert (
        tm._unit_weight_with_warning.__func__
        is AbstractCopula._unit_weight_with_warning
    )

    with pytest.warns(UserWarning) as warning_info:
        weights = tm._weight(x)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        second_weights = tm._weight(x)

    np.testing.assert_allclose(weights, np.ones(4))
    np.testing.assert_allclose(second_weights, np.ones(4))
    assert str(warning_info[0].message) == expected_message
    assert caught == []


def test_student_t_weight_falls_back_when_multivariate_t_is_unavailable():
    tm = StudentTCopula(
        DigitalNetB2(2, seed=115),
        marginals=[stats.norm(), stats.norm()],
        correlation=np.eye(2),
        df=4,
    )
    tm._mvt_scipy = None

    with pytest.warns(UserWarning, match="Weights will be treated as 1"):
        weights = tm._weight(np.full((3, 2), 0.25))

    np.testing.assert_allclose(weights, np.ones(3))


def test_gaussian_weight_uses_pdf_branch_when_logpdf_is_unavailable():
    tm = GaussianCopula(
        DigitalNetB2(2, seed=117),
        marginals=[UnitPDFMarginal(), UnitPDFMarginal()],
        correlation=[[1.0, 0.4], [0.4, 1.0]],
    )

    weights = tm._weight(np.array([[0.25, 0.5], [0.75, 0.5]]))

    assert weights.shape == (2,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)


def test_gumbel_theta_one_weight_is_independent_marginal_density():
    tm = GumbelCopula(
        DigitalNetB2(2, seed=119),
        marginals=[stats.gamma(a=2.0), stats.expon()],
        theta=1.0,
    )
    x = np.array([[1.0, 0.5], [2.0, 1.5]])
    expected = stats.gamma(a=2.0).pdf(x[:, 0]) * stats.expon().pdf(x[:, 1])

    weights = tm._weight(x)

    np.testing.assert_allclose(weights, expected)


def test_gen_copula_samples_composed_transform_branch():
    inner = GaussianCopula(
        DigitalNetB2(2, seed=121),
        marginals=[stats.uniform(), stats.uniform()],
        correlation=[[1.0, 0.3], [0.3, 1.0]],
    )
    outer = ClaytonCopula(inner, marginals=[stats.uniform(), stats.uniform()], theta=1.5)

    v = outer.gen_copula_samples(n_min=4, n_max=8)

    assert v.shape == (4, 2)
    assert np.all(np.isfinite(v))
    assert np.all((0.0 <= v) & (v <= 1.0))


@pytest.mark.parametrize(
    "copula_cls",
    [GaussianCopula, StudentTCopula, ClaytonCopula, GumbelCopula, FrankCopula],
)
def test_copula_spawn_same_dimension_and_reject_different_dimension(copula_cls):
    tm = _make_copula(copula_cls, dimension=2)

    spawned = tm.spawn(s=1, dimensions=[2])
    assert len(spawned) == 1
    assert isinstance(spawned[0], copula_cls)
    assert spawned[0](4).shape == (4, 2)

    with pytest.raises(DimensionError):
        tm._spawn(DigitalNetB2(3, seed=123), 3)


def test_frank_one_dimensional_weight_covers_zero_order_eulerian_term():
    tm = FrankCopula(
        DigitalNetB2(1, seed=125),
        marginals=[UnitPDFMarginal()],
        theta=3.0,
    )

    weights = tm._weight(np.array([[0.25], [0.75]]))

    assert weights.shape == (2,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)


def test_frank_rejects_large_negative_theta_when_exponential_overflows():
    with np.errstate(over="ignore"):
        with pytest.raises(ParameterError, match="too close to 0 or too large"):
            FrankCopula(
                DigitalNetB2(2, seed=127),
                marginals=[stats.uniform(), stats.uniform()],
                theta=-1000.0,
            )
