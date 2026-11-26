from qmcpy import (
    BayesianLRCoeffs,
    BoxIntegral,
    CustomFun,
    DigitalNetB2,
    FinancialOption,
    FourBranch2d,
    Gaussian,
    Genz,
    Hartmann6d,
    ImportanceSampling,
    Ishigami,
    Keister,
    Kumaraswamy,
    Linear0,
    Multimodal2d,
    Sin1d,
    stopping_criterion,
    Uniform,
)
from qmcpy.util import ParameterError
import numpy as np
import sys
import types
import unittest
from unittest.mock import Mock, patch


class TestIntegrand(unittest.TestCase):
    """General tests for Integrand"""

    def test_chained_true_measure_uses_full_recursive_transform(self):
        inner = Uniform(DigitalNetB2(1, seed=7), 0.25, 0.75)
        m = Kumaraswamy(inner)
        square = Mock(side_effect=lambda t: t[..., 0] ** 2)
        integrand = CustomFun(m, g=square)
        points = np.array([[0.1], [0.25], [0.5], [0.75], [0.9]])

        result = integrand.f(points)
        recursive_points = m._jacobian_transform_r(
            points,
            return_weights=False,
        )
        expected_points = m._transform(inner._transform(points))
        evaluated_points = square.call_args.args[0]

        np.testing.assert_allclose(evaluated_points, recursive_points)
        np.testing.assert_allclose(recursive_points, expected_points)
        np.testing.assert_allclose(result, expected_points[..., 0] ** 2)

    def test_abstract_methods(self):
        n = 2**3
        d = 2
        gaussian_proposal = Gaussian(DigitalNetB2(d, seed=7))
        gaussian_importance_sampler = ImportanceSampling(
            target=Gaussian(
                gaussian_proposal.discrete_distrib,
                mean=0,
                covariance=1 / 2,
            ),
            proposal=gaussian_proposal,
        )
        composed_gaussian_proposal = Gaussian(
            Kumaraswamy(DigitalNetB2(d, seed=7))
        )
        composed_gaussian_importance_sampler = ImportanceSampling(
            target=Gaussian(
                composed_gaussian_proposal.discrete_distrib,
                mean=0,
                covariance=1 / 2,
            ),
            proposal=composed_gaussian_proposal,
        )
        integrands = [
            FinancialOption(
                DigitalNetB2(d, seed=7),
                option="ASIAN",
                call_put="call",
                asian_mean="arithmetic",
            ),
            FinancialOption(
                DigitalNetB2(d, seed=7),
                option="ASIAN",
                call_put="put",
                asian_mean="arithmetic",
            ),
            FinancialOption(
                DigitalNetB2(d, seed=7),
                option="ASIAN",
                call_put="call",
                asian_mean="geometric",
            ),
            FinancialOption(
                DigitalNetB2(d, seed=7),
                option="ASIAN",
                call_put="put",
                asian_mean="geometric",
            ),
            FinancialOption(
                DigitalNetB2(d, seed=7),
                option="ASIAN",
                call_put="put",
                asian_mean="geometric",
                level=1,
                d_coarsest=1,
            ),
            BoxIntegral(DigitalNetB2(d, seed=7), s=1),
            BoxIntegral(DigitalNetB2(d, seed=7), s=[3, 5, 7]),
            CustomFun(Uniform(DigitalNetB2(d, seed=7)), lambda x: x.prod(1)),
            CustomFun(
                Uniform(
                    Kumaraswamy(
                        Kumaraswamy(DigitalNetB2(d, seed=7))
                    )
                ),
                lambda x: x.prod(1),
            ),
            CustomFun(
                Gaussian(DigitalNetB2(2, seed=7)),
                lambda x: np.moveaxis(x, -1, 0),
                dimension_indv=d,
            ),
            FinancialOption(
                DigitalNetB2(d, seed=7), option="EUROPEAN", call_put="call"
            ),
            FinancialOption(DigitalNetB2(d, seed=7), option="EUROPEAN", call_put="put"),
            Keister(DigitalNetB2(d, seed=7)),
            Keister(gaussian_importance_sampler),
            Keister(composed_gaussian_importance_sampler),
            Linear0(DigitalNetB2(d, seed=7)),
        ]
        spawned_integrands = [integrand.spawn(levels=0)[0] for integrand in integrands]
        for integrand in integrands + spawned_integrands:
            x = integrand.discrete_distrib.gen_samples(n)
            s = str(integrand)
            for ptransform in ["None", "Baker", "C0", "C1", "C1sin", "C2sin", "C3sin"]:
                y = integrand.f(x, periodization_transform=ptransform)
                self.assertEqual(y.shape, (integrand.d_indv + (n,)))
                self.assertTrue(np.isfinite(y).all())
                self.assertEqual(y.dtype, np.float64)

    def test_keister(self, dims=3):
        k = Keister(DigitalNetB2(dims, seed=7))
        exact_integ = k.exact_integ(dims)
        x = k.discrete_distrib.gen_samples(2**10)
        y = k.f(x)
        self.assertAlmostEqual(y.mean(), exact_integ, places=2)


class TestIntegrandExamples(unittest.TestCase):
    """Focused tests for individual integrand implementations."""

    def _assert_scalar_finite_output(self, integrand, n=2**10):
        x = integrand.discrete_distrib.gen_samples(n)
        y = integrand.f(x)
        self.assertEqual(y.shape, (n,))
        self.assertTrue(np.isfinite(y).all())

    def _assert_genz_output(self, kind_func):
        for kind_coeff in [1, 2, 3]:
            integrand = Genz(
                DigitalNetB2(2, seed=7),
                kind_func=kind_func,
                kind_coeff=kind_coeff,
            )
            y = integrand(32)
            self.assertEqual(y.shape, (32,))

    def test_fourbranch2d(self):
        fb = FourBranch2d(DigitalNetB2(2, seed=7))
        self._assert_scalar_finite_output(fb)

    def test_multimodal2d(self):
        mm = Multimodal2d(DigitalNetB2(2, seed=7))
        self._assert_scalar_finite_output(mm)

    def test_genz_oscillatory_all_coeffs(self):
        self._assert_genz_output("OSCILLATORY")

    def test_genz_corner_peak_all_coeffs(self):
        self._assert_genz_output("CORNER PEAK")

    def test_genz_invalid_kind_func(self):
        self.assertRaises(ParameterError, Genz, DigitalNetB2(2, seed=7), kind_func="INVALID")

    def test_genz_invalid_kind_coeff(self):
        self.assertRaises(ParameterError, Genz, DigitalNetB2(2, seed=7), kind_coeff=4)

    def test_genz_spawn(self):
        ig = Genz(DigitalNetB2(2, seed=7), kind_func="CORNER PEAK", kind_coeff=2)
        spawned = ig.spawn(levels=0)
        self.assertEqual(len(spawned), 1)

    def test_sin1d_basic_and_spawn(self):
        ig = Sin1d(DigitalNetB2(1, seed=7), k=2)
        y = ig(64)
        self.assertEqual(y.shape, (64,))
        self.assertTrue(np.isfinite(y).all())
        points = np.array([[0], [np.pi / 2], [np.pi], [3 * np.pi / 2]])
        np.testing.assert_allclose(ig.g(points), [0, 1, 0, -1], atol=1e-15)
        np.testing.assert_allclose(ig.true_measure.a, [0])
        np.testing.assert_allclose(ig.true_measure.b, [4 * np.pi])

        spawned = ig.spawn(levels=0)[0]
        self.assertEqual(spawned.k, 2)
        np.testing.assert_allclose(spawned.true_measure.b, [4 * np.pi])
        with self.assertRaises(AssertionError):
            Sin1d(DigitalNetB2(2, seed=7))

    def test_ishigami_basic_and_dimension_error(self):
        ig = Ishigami(DigitalNetB2(3, seed=7), a=7, b=0.1)
        y = ig(64)
        self.assertEqual(y.shape, (64,))
        self.assertTrue(np.isfinite(y).all())
        points = np.array(
            [[0, 0, 0], [np.pi / 2, np.pi / 2, 1]], dtype=float
        )
        np.testing.assert_allclose(ig.g(points), [0, 8.1])

        spawned = ig.spawn(levels=0)[0]
        self.assertEqual(spawned.a, 7)
        self.assertEqual(spawned.b, 0.1)
        self.assertRaises(ParameterError, Ishigami, DigitalNetB2(2, seed=7))

    def test_ishigami_exact_helpers(self):
        indices = np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [True, True, False],
                [True, False, True],
                [False, True, True],
            ],
            dtype=bool,
        )
        sens = Ishigami._exact_sensitivity_indices(indices, a=7, b=0.1)
        self.assertEqual(sens.shape, (2, 6))
        fu = Ishigami._exact_fu_functions(
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            [[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]],
            a=7,
            b=0.1,
        )
        self.assertEqual(fu.shape, (2, 8))
        np.testing.assert_allclose(fu[:, 0], 3.5)

    def test_keister_formula_exact_value_and_spawn(self):
        ig = Keister(DigitalNetB2(2, seed=7))
        points = np.array([[0, 0], [3, 4]], dtype=float)
        np.testing.assert_allclose(
            ig.g(points), np.pi * np.array([1, np.cos(5)])
        )
        expected_1d = np.sqrt(np.pi) * np.exp(-0.25)
        self.assertAlmostEqual(Keister.get_exact_value(1), expected_1d)
        self.assertAlmostEqual(ig.exact_integ(1), expected_1d)
        expected_2d = 1.808186429263620
        self.assertAlmostEqual(Keister.get_exact_value(2), expected_2d)
        self.assertAlmostEqual(ig.exact_integ(2), expected_2d)

        spawned = ig.spawn(levels=0)[0]
        self.assertIsInstance(spawned, Keister)
        self.assertEqual(spawned.d, 2)

    def test_hartmann6d_without_optional_dependencies(self):
        class FakeTensor:
            def __init__(self, array):
                self.array = np.asarray(array)

            def numpy(self):
                return self.array

        class FakeAugmentedHartmann:
            def __init__(self, negate=False):
                self.negate = negate
                self.last_input = None

            def evaluate_true(self, tensor):
                self.last_input = np.asarray(tensor)
                return FakeTensor(self.last_input.sum(axis=-1))

        botorch = types.ModuleType("botorch")
        test_functions = types.ModuleType("botorch.test_functions")
        multi_fidelity = types.ModuleType(
            "botorch.test_functions.multi_fidelity"
        )
        multi_fidelity.AugmentedHartmann = FakeAugmentedHartmann
        test_functions.multi_fidelity = multi_fidelity
        botorch.test_functions = test_functions
        torch = types.ModuleType("torch")
        torch.tensor = np.asarray
        fake_modules = {
            "botorch": botorch,
            "botorch.test_functions": test_functions,
            "botorch.test_functions.multi_fidelity": multi_fidelity,
            "torch": torch,
        }

        with patch.dict(sys.modules, fake_modules):
            ig = Hartmann6d(DigitalNetB2(6, seed=7))
            points = np.zeros((2, 6))
            y = ig.g(points)

        self.assertFalse(ig.ah.negate)
        self.assertEqual(ig.ah.last_input.shape, (2, 7))
        np.testing.assert_allclose(ig.ah.last_input[:, -1], 1)
        np.testing.assert_allclose(y, [1, 1])
        with self.assertRaises(AssertionError):
            Hartmann6d(DigitalNetB2(5, seed=7))

    def test_financial_option_invalid_inputs(self):
        self.assertRaises(
            AssertionError,
            FinancialOption,
            DigitalNetB2(2, seed=7),
            option="ASIAN",
            call_put="BAD",
        )
        self.assertRaises(
            AssertionError,
            FinancialOption,
            DigitalNetB2(2, seed=7),
            option="ASIAN",
            asian_mean="BAD",
        )
        self.assertRaises(
            AssertionError,
            FinancialOption,
            DigitalNetB2(2, seed=7),
            option="ASIAN",
            asian_mean_quadrature_rule="BAD",
        )

class TestCustomVector(unittest.TestCase):
    """Tests CustomFun vector integration."""
    def setUp(self):
        self.true_measure = Gaussian(DigitalNetB2(2, seed=7, replications=2**5))
        self.true_measure_e2e = Gaussian(DigitalNetB2(2, seed=7))
        # compute covariance matrix integrals
        self.integrands = lambda x: np.stack([x[..., 0],
                                              x[..., 1],
                                              x[...,0]**2,
                                              x[...,1]**2,
                                              x[...,0]*x[...,1],
                                              ])
        # compute correlation matrix and means
        def comb_func(ys):
            mean0 = ys[0]
            mean1 = ys[1]
            var0 = ys[2] - mean0**2
            var1 = ys[3] - mean1**2
            cov = ys[4] - mean0 * mean1
            corr = cov / np.sqrt(var0*var1)
            return np.stack([mean0, var1, corr]) # return subset so that d_indv != d_comb
        self.comb_func = comb_func

        # compute bounds on combinations
        def bound_fun(ymin,ymax):
            mean0_ub = ymax[0]
            mean0_lb = ymin[0]
            mean1_ub = ymax[1]
            mean1_lb = ymin[1]
            var0_ub = ymax[2] - mean0_lb**2
            var0_lb = ymin[2] - mean0_ub**2
            var1_ub = ymax[3] - mean1_lb**2
            var1_lb = ymin[3] - mean1_ub**2
            cov_ub = ymax[4] - mean0_lb * mean1_lb
            cov_lb = ymin[4] - mean0_ub * mean1_ub
            corr_ub = cov_ub/np.sqrt(var0_lb*var1_lb)
            corr_lb = cov_lb/np.sqrt(var0_ub*var1_ub)
            lower_bounds = np.stack([mean0_lb, var1_lb, corr_lb]) # return subset to test case of d_indv != d_comb
            upper_bounds = np.stack([mean0_ub, var1_ub, corr_ub]) # return subset to test case of d_indv != d_comb
            return lower_bounds, upper_bounds
        self.bound_fun = bound_fun

    def test_construction(self):
        """ tests for initialization of CustomFunc where comb_func is supplied
        """
        with self.assertRaises(ValueError):
            CustomFun(self.true_measure, self.integrands,
                    dimension_indv=(5,),
                    comb_fun=self.comb_func,
                    dimension_comb=None,
                    bound_fun=self.bound_fun)

    def test_spawn(self):
        integrand = CustomFun(self.true_measure, self.integrands,
                        dimension_indv=(5,),
                        comb_fun=self.comb_func,
                        dimension_comb=(3,),
                        bound_fun=self.bound_fun)
        spawned = integrand.spawn(np.arange(5))
        self.assertIsInstance(spawned, list)
        self.assertEqual(len(spawned), 5)

    def test_default_bounds(self):
        """ tests to verify expected results of bound_fun
        """
        integrand_exact_bounds = CustomFun(self.true_measure, self.integrands,
                                    dimension_indv=(5,),
                                    comb_fun=self.comb_func,
                                    dimension_comb=(3,),
                                    bound_fun=self.bound_fun)
        integrand_approx_bounds = CustomFun(self.true_measure, self.integrands,
                                    dimension_indv=(5,),
                                    comb_fun=self.comb_func,
                                    dimension_comb=(3,))
        integrands = integrand_exact_bounds(2**12)
        n_replications = integrands.shape[1]
        means = integrands.mean(-1)
        mean = means.mean(-1)
        err = means.std(-1)/np.sqrt(n_replications-1)
        integrands_lb = mean - err
        integrands_ub = mean + err
        exact_bounds_lb, exact_bounds_ub = integrand_exact_bounds.bound_fun(integrands_lb, integrands_ub)
        approx_bounds_lb, approx_bounds_ub = integrand_approx_bounds.bound_fun(integrands_lb, integrands_ub)

        np.testing.assert_allclose(exact_bounds_lb, approx_bounds_lb, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(exact_bounds_ub, approx_bounds_ub, rtol=1e-5, atol=1e-6)

    def test_results_e2e(self):
        """ tests to verify integration results against analytical integrals"""
        integrand = CustomFun(self.true_measure_e2e, self.integrands,
                        dimension_indv=(5,),
                        comb_fun=self.comb_func,
                        dimension_comb=(3,),
                        bound_fun=self.bound_fun)
        sr = stopping_criterion.CubQMCDigitalNetB2G(integrand, 1e-5, 1e-5)
        results, data = sr.integrate()
        np.testing.assert_allclose(results, np.array([0,1,0]), 1e-4, 1e-4)

class TestBayesianLRCoeffs(unittest.TestCase):
    """Tests for BayesianLRCoeffs methods not covered by unit tests."""

    def setUp(self):
        self.ig = BayesianLRCoeffs(
            DigitalNetB2(3, seed=7),
            feature_array=np.arange(8).reshape((4, 2)),
            response_vector=[0, 0, 1, 1],
        )

    def test_basic_evaluation(self):
        y = self.ig(64)
        # shape is (2, num_coeffs, n) = (2, 3, 64)
        self.assertEqual(y.shape, (2, 3, 64))

    def test_bound_fun(self):
        bound_low = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        bound_high = np.array([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]])
        low, high = self.ig.bound_fun(bound_low, bound_high)
        self.assertEqual(low.shape, (3,))
        self.assertEqual(high.shape, (3,))
        self.assertTrue((low <= high).all())

    def test_bound_fun_denominator_spans_zero(self):
        # den_bounds_low <= 0 <= den_bounds_high  -> (-inf, +inf) bounds
        bound_low = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
        bound_high = np.array([[0.2, 0.3, 0.4], [0.1, 0.2, 0.3]])
        low, high = self.ig.bound_fun(bound_low, bound_high)
        self.assertTrue(np.all(low == -np.inf) or np.any(np.isinf(low)))

    def test_dependency(self):
        comb_flags = np.array([True, False, True])
        dep = self.ig.dependency(comb_flags)
        self.assertEqual(dep.shape, (2, 3))

    def test_invalid_dimension(self):
        # Dimension 2 but feature_array has 2 features -> expects d=3
        self.assertRaises(
            ParameterError,
            BayesianLRCoeffs,
            DigitalNetB2(2, seed=7),
            np.arange(8).reshape((4, 2)),
            [0, 0, 1, 1],
        )

    def test_invalid_response_vector(self):
        # Response vector with non-binary values
        self.assertRaises(
            ParameterError,
            BayesianLRCoeffs,
            DigitalNetB2(3, seed=7),
            np.arange(8).reshape((4, 2)),
            [0, 0, 2, 1],
        )


if __name__ == "__main__":
    unittest.main()
