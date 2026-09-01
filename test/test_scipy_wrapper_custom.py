import warnings

import pytest
import numpy as np
import scipy.stats as stats

from qmcpy import DigitalNetB2, SciPyWrapper, StudentT, ZeroInflatedExpUniform

from qmcpy.true_measure.triangular import TriangularDistribution
from qmcpy.util import DimensionError, ParameterError


MISSING_PDF_WARNING = "no 'pdf' or 'logpdf'"


def _missing_pdf_warnings(caught):
    return [
        warning
        for warning in caught
        if issubclass(warning.category, UserWarning)
        and MISSING_PDF_WARNING in str(warning.message)
    ]


def test_mvn_dependence_correlation_and_moment():
    """
    Check that passing a SciPy multivariate normal through SciPyWrapper
    preserves correlation and the mixed moment E[X1 X2].
    """
    sampler = DigitalNetB2(2, seed=5)
    rho_target = 0.7
    cov = [[1.0, rho_target], [rho_target, 1.0]]
    mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=cov)
    tm_mvn = SciPyWrapper(sampler, scipy_distribs=mvn)

    n = 4096
    x = tm_mvn(n)

    rho_hat = np.corrcoef(x.T)[0, 1]
    est_moment = np.mean(x[:, 0] * x[:, 1])

    assert np.isfinite(rho_hat)
    assert np.isfinite(est_moment)

    assert abs(rho_hat - rho_target) < 0.05
    assert abs(est_moment - rho_target) < 0.05


def test_triangular_custom_marginal_range_and_shape():
    """
    Make sure our custom triangular marginal behaves sensibly:
    samples stay in the right interval and the empirical mean is close
    to the analytic mean.
    """
    tri = TriangularDistribution(c=0.3, loc=-1.0, scale=2.0)
    tm = SciPyWrapper(DigitalNetB2(1, seed=11), scipy_distribs=tri)

    n = 4096
    x = tm(n).ravel()

    assert x.min() >= -1.1
    assert x.max() <= 1.1

    a = -1.0
    b = 1.0
    m = -1.0 + 0.3 * 2.0
    true_mean = (a + b + m) / 3.0
    emp_mean = x.mean()
    assert abs(emp_mean - true_mean) < 0.05


def test_zero_inflated_zero_rate():
    """
    Check that the zero-inflated exponential distribution preserves the
    specified probability mass at X = 0.
    """
    p_zero = 0.4
    sampler = DigitalNetB2(1, seed=17)
    tm = ZeroInflatedExpUniform(sampler, p_zero=p_zero, lam=1.5)

    n = 4096
    samples = tm(n)
    x = samples.ravel()
    zero_rate = np.mean(x == 0.0)

    assert samples.shape == (n, 1)
    assert abs(zero_rate - p_zero) < 0.05


def test_zero_inflated_replications_shape():
    tm = ZeroInflatedExpUniform(
        DigitalNetB2(1, seed=17, replications=2),
        p_zero=0.4,
        lam=1.5,
    )

    x = tm(8)

    assert x.shape == (2, 8, 1)
    assert np.all(x >= 0.0)


@pytest.mark.parametrize("p_zero", [0.0, 1.0, -0.1, 1.1])
def test_zero_inflated_rejects_invalid_p_zero(p_zero):
    with pytest.raises(ParameterError, match="p_zero must be in"):
        ZeroInflatedExpUniform(
            DigitalNetB2(1, seed=17),
            p_zero=p_zero,
            lam=1.5,
        )


@pytest.mark.parametrize("lam", [0.0, -1.0])
def test_zero_inflated_rejects_nonpositive_lam(lam):
    with pytest.raises(ParameterError, match="lam must be positive"):
        ZeroInflatedExpUniform(
            DigitalNetB2(1, seed=17),
            p_zero=0.4,
            lam=lam,
        )


def test_zero_inflated_requires_one_dimensional_sampler():
    with pytest.raises(
        DimensionError,
        match="requires a one-dimensional sampler",
    ):
        ZeroInflatedExpUniform(
            DigitalNetB2(2, seed=17),
            p_zero=0.4,
            lam=1.5,
        )


def test_zero_inflated_inverse_transform_exact_values():
    tm = ZeroInflatedExpUniform(
        DigitalNetB2(1, seed=17),
        p_zero=0.4,
        lam=2.0,
    )
    u = np.array([[0.0], [0.2], [0.4], [0.7], [0.9]])

    x = tm._transform(u)

    assert x.shape == (5, 1)
    assert np.array_equal(x[:3], np.zeros((3, 1)))
    assert np.all(x[3:] > 0.0)

    u_positive = u[3:, 0]
    u_rescaled = (u_positive - 0.4) / 0.6
    expected = -np.log1p(-u_rescaled) / 2.0
    assert np.allclose(x[3:, 0], expected)


def test_zero_inflated_inverse_transform_all_zero_branch():
    tm = ZeroInflatedExpUniform(
        DigitalNetB2(1, seed=17),
        p_zero=0.4,
        lam=2.0,
    )
    u = np.array([[0.0], [0.1], [0.4]])

    x = tm._transform(u)

    assert x.shape == (3, 1)
    assert np.array_equal(x, np.zeros((3, 1)))


def test_zero_inflated_inverse_transform_clips_one():
    tm = ZeroInflatedExpUniform(
        DigitalNetB2(1, seed=17),
        p_zero=0.4,
        lam=2.0,
    )
    u = np.array([[1.0]])

    x = tm._transform(u)

    assert x.shape == (1, 1)
    assert np.isfinite(x).all()
    assert x[0, 0] > 0.0


def test_zero_inflated_construction_does_not_warn_about_missing_pdf():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tm = ZeroInflatedExpUniform(
            DigitalNetB2(1, seed=17),
            p_zero=0.4,
            lam=1.5,
        )

    assert tm.d == 1
    assert _missing_pdf_warnings(caught) == []


def test_zero_inflated_sampling_does_not_warn_about_missing_pdf():
    tm = ZeroInflatedExpUniform(DigitalNetB2(1, seed=17), p_zero=0.4, lam=1.5)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = tm(8)

    assert x.shape == (8, 1)
    assert _missing_pdf_warnings(caught) == []


def test_zero_inflated_return_weights_warns_once_for_missing_pdf():
    tm = ZeroInflatedExpUniform(DigitalNetB2(1, seed=17), p_zero=0.4, lam=1.5)

    with pytest.warns(UserWarning, match=MISSING_PDF_WARNING):
        x, jac = tm(8, return_weights=True)

    assert x.shape == (8, 1)
    assert jac.shape == (8,)
    assert np.allclose(jac, 1.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x_second, jac_second = tm(8, return_weights=True)

    assert x_second.shape == (8, 1)
    assert np.allclose(jac_second, 1.0)
    assert _missing_pdf_warnings(caught) == []


def test_zero_inflated_y_split_warns_and_uses_one_dimensional_interface():
    with pytest.warns(DeprecationWarning, match="y_split"):
        tm = ZeroInflatedExpUniform(
            DigitalNetB2(1, seed=17),
            p_zero=0.4,
            lam=1.5,
            y_split=0.5,
        )

    x = tm(4)

    assert x.shape == (4, 1)
    assert np.all(x >= 0.0)


def test_zero_inflated_y_split_preserves_deprecated_two_dimensional_usage():
    with pytest.warns(DeprecationWarning, match="2D zero-inflated"):
        tm = ZeroInflatedExpUniform(
            DigitalNetB2(2, seed=17),
            p_zero=0.4,
            lam=1.5,
            y_split=0.5,
        )

    x = tm(16)

    assert x.shape == (16, 2)
    assert np.all(x[:, 0] >= 0.0)
    assert np.all((0.0 <= x[:, 1]) & (x[:, 1] <= 1.0))
    assert np.all(x[x[:, 0] == 0.0, 1] <= 0.5)
    assert np.all(x[x[:, 0] > 0.0, 1] >= 0.5)


def test_zero_inflated_y_split_preserves_replicated_two_dimensional_usage():
    with pytest.warns(DeprecationWarning, match="2D zero-inflated"):
        tm = ZeroInflatedExpUniform(
            DigitalNetB2(2, seed=17, replications=2),
            p_zero=0.4,
            lam=1.5,
            y_split=0.5,
        )

    x = tm(16)

    assert x.shape == (2, 16, 2)
    assert np.all(x[..., 0] >= 0.0)
    assert np.all((0.0 <= x[..., 1]) & (x[..., 1] <= 1.0))
    assert np.all(x[..., 1][x[..., 0] == 0.0] <= 0.5)
    assert np.all(x[..., 1][x[..., 0] > 0.0] >= 0.5)


def test_student_t_marginals_shape():
    tm = SciPyWrapper(
        sampler=DigitalNetB2(2, seed=5),
        scipy_distribs=stats.t(df=5),
    )
    x = tm(8)
    assert x.shape == (8, 2)


def test_multivariate_student_t_joint_corr_and_cov():
    if not hasattr(stats, "multivariate_t"):
        pytest.skip("scipy.stats.multivariate_t not available in this SciPy version")

    df = 5.0
    rho = 0.8
    loc = np.array([0.0, 0.0])
    shape = np.array([[1.0, rho], [rho, 1.0]])

    tm = StudentT(DigitalNetB2(2, seed=123), loc=loc, shape=shape, df=df)

    n = 4096
    x = tm(n)
    emp_corr = np.corrcoef(x.T)[0, 1]

    assert abs(emp_corr - rho) < 0.05
