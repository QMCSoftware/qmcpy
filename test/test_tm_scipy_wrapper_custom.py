import unittest
import warnings

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


class TestSciPyWrapperCustom(unittest.TestCase):

    def test_mvn_dependence_correlation_and_moment(self):
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

        self.assertTrue(np.isfinite(rho_hat))
        self.assertTrue(np.isfinite(est_moment))

        self.assertLess(abs(rho_hat - rho_target), 0.05)
        self.assertLess(abs(est_moment - rho_target), 0.05)

    def test_triangular_custom_marginal_range_and_shape(self):
        """
        Make sure our custom triangular marginal behaves sensibly:
        samples stay in the right interval and the empirical mean is close
        to the analytic mean.
        """
        tri = TriangularDistribution(c=0.3, loc=-1.0, scale=2.0)
        tm = SciPyWrapper(DigitalNetB2(1, seed=11), scipy_distribs=tri)

        n = 4096
        x = tm(n).ravel()

        self.assertGreaterEqual(x.min(), -1.1)
        self.assertLessEqual(x.max(), 1.1)

        a = -1.0
        b = 1.0
        m = -1.0 + 0.3 * 2.0
        true_mean = (a + b + m) / 3.0
        emp_mean = x.mean()
        self.assertLess(abs(emp_mean - true_mean), 0.05)

    def test_zero_inflated_zero_rate(self):
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

        self.assertEqual(samples.shape, (n, 1))
        self.assertLess(abs(zero_rate - p_zero), 0.05)

    def test_zero_inflated_replications_shape(self):
        tm = ZeroInflatedExpUniform(
            DigitalNetB2(1, seed=17, replications=2),
            p_zero=0.4,
            lam=1.5,
        )

        x = tm(8)

        self.assertEqual(x.shape, (2, 8, 1))
        self.assertTrue(np.all(x >= 0.0))

    def test_zero_inflated_rejects_invalid_p_zero(self):
        for p_zero in [0.0, 1.0, -0.1, 1.1]:
            with self.subTest(p_zero=p_zero):
                with self.assertRaisesRegex(ParameterError, "p_zero must be in"):
                    ZeroInflatedExpUniform(
                        DigitalNetB2(1, seed=17),
                        p_zero=p_zero,
                        lam=1.5,
                    )

    def test_zero_inflated_rejects_nonpositive_lam(self):
        for lam in [0.0, -1.0]:
            with self.subTest(lam=lam):
                with self.assertRaisesRegex(ParameterError, "lam must be positive"):
                    ZeroInflatedExpUniform(
                        DigitalNetB2(1, seed=17),
                        p_zero=0.4,
                        lam=lam,
                    )

    def test_zero_inflated_requires_one_dimensional_sampler(self):
        with self.assertRaisesRegex(
            DimensionError, "requires a one-dimensional sampler"
        ):
            ZeroInflatedExpUniform(
                DigitalNetB2(2, seed=17),
                p_zero=0.4,
                lam=1.5,
            )

    def test_zero_inflated_inverse_transform_exact_values(self):
        tm = ZeroInflatedExpUniform(
            DigitalNetB2(1, seed=17),
            p_zero=0.4,
            lam=2.0,
        )
        u = np.array([[0.0], [0.2], [0.4], [0.7], [0.9]])

        x = tm._transform(u)

        self.assertEqual(x.shape, (5, 1))
        self.assertTrue(np.array_equal(x[:3], np.zeros((3, 1))))
        self.assertTrue(np.all(x[3:] > 0.0))

        u_positive = u[3:, 0]
        u_rescaled = (u_positive - 0.4) / 0.6
        expected = -np.log1p(-u_rescaled) / 2.0
        self.assertTrue(np.allclose(x[3:, 0], expected))

    def test_zero_inflated_inverse_transform_all_zero_branch(self):
        tm = ZeroInflatedExpUniform(
            DigitalNetB2(1, seed=17),
            p_zero=0.4,
            lam=2.0,
        )
        u = np.array([[0.0], [0.1], [0.4]])

        x = tm._transform(u)

        self.assertEqual(x.shape, (3, 1))
        self.assertTrue(np.array_equal(x, np.zeros((3, 1))))

    def test_zero_inflated_inverse_transform_clips_one(self):
        tm = ZeroInflatedExpUniform(
            DigitalNetB2(1, seed=17),
            p_zero=0.4,
            lam=2.0,
        )
        u = np.array([[1.0]])

        x = tm._transform(u)

        self.assertEqual(x.shape, (1, 1))
        self.assertTrue(np.isfinite(x).all())
        self.assertGreater(x[0, 0], 0.0)

    def test_zero_inflated_construction_does_not_warn_about_missing_pdf(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tm = ZeroInflatedExpUniform(
                DigitalNetB2(1, seed=17),
                p_zero=0.4,
                lam=1.5,
            )

        self.assertEqual(tm.d, 1)
        self.assertEqual(_missing_pdf_warnings(caught), [])

    def test_zero_inflated_sampling_does_not_warn_about_missing_pdf(self):
        tm = ZeroInflatedExpUniform(DigitalNetB2(1, seed=17), p_zero=0.4, lam=1.5)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            x = tm(8)

        self.assertEqual(x.shape, (8, 1))
        self.assertEqual(_missing_pdf_warnings(caught), [])

    def test_zero_inflated_return_weights_warns_once_for_missing_pdf(self):
        tm = ZeroInflatedExpUniform(DigitalNetB2(1, seed=17), p_zero=0.4, lam=1.5)

        with self.assertWarnsRegex(UserWarning, MISSING_PDF_WARNING):
            x, jac = tm(8, return_weights=True)

        self.assertEqual(x.shape, (8, 1))
        self.assertEqual(jac.shape, (8,))
        self.assertTrue(np.allclose(jac, 1.0))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            x_second, jac_second = tm(8, return_weights=True)

        self.assertEqual(x_second.shape, (8, 1))
        self.assertTrue(np.allclose(jac_second, 1.0))
        self.assertEqual(_missing_pdf_warnings(caught), [])

    def test_zero_inflated_y_split_warns_and_uses_one_dimensional_interface(self):
        with self.assertWarnsRegex(DeprecationWarning, "y_split"):
            tm = ZeroInflatedExpUniform(
                DigitalNetB2(1, seed=17),
                p_zero=0.4,
                lam=1.5,
                y_split=0.5,
            )

        x = tm(4)

        self.assertEqual(x.shape, (4, 1))
        self.assertTrue(np.all(x >= 0.0))

    def test_zero_inflated_y_split_preserves_deprecated_two_dimensional_usage(self):
        with self.assertWarnsRegex(DeprecationWarning, "2D zero-inflated"):
            tm = ZeroInflatedExpUniform(
                DigitalNetB2(2, seed=17),
                p_zero=0.4,
                lam=1.5,
                y_split=0.5,
            )

        x = tm(16)

        self.assertEqual(x.shape, (16, 2))
        self.assertTrue(np.all(x[:, 0] >= 0.0))
        self.assertTrue(np.all((0.0 <= x[:, 1]) & (x[:, 1] <= 1.0)))
        self.assertTrue(np.all(x[x[:, 0] == 0.0, 1] <= 0.5))
        self.assertTrue(np.all(x[x[:, 0] > 0.0, 1] >= 0.5))

    def test_zero_inflated_y_split_preserves_replicated_two_dimensional_usage(self):
        with self.assertWarnsRegex(DeprecationWarning, "2D zero-inflated"):
            tm = ZeroInflatedExpUniform(
                DigitalNetB2(2, seed=17, replications=2),
                p_zero=0.4,
                lam=1.5,
                y_split=0.5,
            )

        x = tm(16)

        self.assertEqual(x.shape, (2, 16, 2))
        self.assertTrue(np.all(x[..., 0] >= 0.0))
        self.assertTrue(np.all((0.0 <= x[..., 1]) & (x[..., 1] <= 1.0)))
        self.assertTrue(np.all(x[..., 1][x[..., 0] == 0.0] <= 0.5))
        self.assertTrue(np.all(x[..., 1][x[..., 0] > 0.0] >= 0.5))

    def test_student_t_marginals_shape(self):
        tm = SciPyWrapper(
            sampler=DigitalNetB2(2, seed=5),
            scipy_distribs=stats.t(df=5),
        )
        x = tm(8)
        self.assertEqual(x.shape, (8, 2))

    def test_multivariate_student_t_joint_corr_and_cov(self):
        if not hasattr(stats, "multivariate_t"):
            self.skipTest("scipy.stats.multivariate_t not available in this SciPy version")

        df = 5.0
        rho = 0.8
        loc = np.array([0.0, 0.0])
        shape = np.array([[1.0, rho], [rho, 1.0]])

        tm = StudentT(DigitalNetB2(2, seed=123), loc=loc, shape=shape, df=df)

        n = 4096
        x = tm(n)
        emp_corr = np.corrcoef(x.T)[0, 1]

        self.assertLess(abs(emp_corr - rho), 0.05)


if __name__ == "__main__":
    unittest.main()
