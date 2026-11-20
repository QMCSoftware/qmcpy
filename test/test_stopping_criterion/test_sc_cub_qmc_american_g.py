import unittest
import warnings
from qmcpy import (
    CubQMCAmericanG,
    CubQMCNetG,
    CubQMCRepStudentT,
    FinancialOption,
    Halton,
    IIDStdUniform,
    Lattice,
    Sobol,
)
from qmcpy.util import ParameterError, ParameterWarning


class TestCubQMCAmericanG(unittest.TestCase):

    def setUp(self):
        self.d = 10
        self.opt_params = {
            "volatility": 0.2,
            "start_price": 100.0,
            "strike_price": 100.0,
            "interest_rate": 0.05,
            "t_final": 1.0,
        }

    def test_non_american_rejection(self):
        euro_opt = FinancialOption(Sobol(self.d, seed=7), option="EUROPEAN", call_put="PUT", **self.opt_params)
        with self.assertRaises(ParameterError):
            CubQMCAmericanG(euro_opt)

    def test_invalid_n_train_parameter_error(self):
        opt = FinancialOption(Sobol(self.d, seed=7), option="AMERICAN", call_put="PUT", **self.opt_params)
        with self.assertRaises(ParameterError):
            CubQMCAmericanG(opt, n_train=0)
        with self.assertRaises(ParameterError):
            CubQMCAmericanG(opt, n_train=-10)

    def test_immediate_exercise_at_t0(self):
        # Deep ITM put with low volatility and positive interest rate:
        # immediate exercise payoff at t=0 is K - S0 = 100 - 50 = 50.
        # Discounted continuation value is lower than 50.
        param = dict(
            option="AMERICAN",
            call_put="PUT",
            volatility=0.01,
            start_price=50.0,
            strike_price=100.0,
            interest_rate=0.1,
        )
        ao = FinancialOption(Sobol(10, seed=7), **param)
        sol, dat = CubQMCAmericanG(ao, n_train=2**9, n_init=2**8, n_limit=2**10).integrate()
        now = ao.strike_price - ao.start_price
        self.assertEqual(float(sol), float(now))
        self.assertEqual(float(dat.solution), float(now))
        if hasattr(dat, "comb_bound_low"):
            self.assertGreaterEqual(float(dat.comb_bound_low), float(now))

    def test_conflicting_inputs_warning(self):
        ao = FinancialOption(Sobol(self.d, seed=7), option="AMERICAN", call_put="PUT", **self.opt_params)
        sc1 = CubQMCNetG(ao, abs_tol=1.0, n_init=2**8, n_limit=2**10)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            CubQMCAmericanG(ao, abs_tol=1e-4, stopping_criterion=sc1)
            warning_messages = [str(w.message) for w in caught_warnings if issubclass(w.category, ParameterWarning)]
            self.assertGreaterEqual(len(warning_messages), 1)
            self.assertTrue(any("Conflicting input values detected" in msg for msg in warning_messages))

    def test_iid_replications(self):
        # Test replications=1 and replications=2 for IID distribution
        iid1 = FinancialOption(IIDStdUniform(self.d, seed=7, replications=1), option="AMERICAN", call_put="PUT", **self.opt_params)
        sol1, _ = CubQMCAmericanG(iid1, abs_tol=1.0, n_train=64, n_init=64, n_limit=256).integrate()
        self.assertTrue(5.0 < sol1 < 7.0)

        iid2 = FinancialOption(IIDStdUniform(self.d, seed=7, replications=2), option="AMERICAN", call_put="PUT", **self.opt_params)
        sol2, _ = CubQMCAmericanG(iid2, abs_tol=1.0, n_train=64, n_init=64, n_limit=256).integrate()
        self.assertTrue(5.0 < sol2 < 7.0)

    def test_separate_training_sampler(self):
        ao = FinancialOption(Lattice(self.d, seed=7), option="AMERICAN", call_put="PUT", **self.opt_params)
        # Valid separate training sampler
        sc = CubQMCAmericanG(
            ao,
            abs_tol=0.2,
            n_train=2**9,
            n_init=2**8,
            n_limit=2**10,
            training_discrete_distrib=Sobol(self.d, seed=42),
        )
        sol, dat = sc.integrate()
        self.assertTrue(5.5 < sol < 6.5)

        # Incompatible dimension validation
        with self.assertRaises(ParameterError):
            CubQMCAmericanG(ao, training_discrete_distrib=Sobol(self.d + 1, seed=42))

        # Incompatible type validation
        with self.assertRaises(ParameterError):
            CubQMCAmericanG(ao, training_discrete_distrib="invalid_sampler")

    def test_sobol_american_put(self):
        opt = FinancialOption(Sobol(self.d, seed=7), option="AMERICAN", call_put="PUT", **self.opt_params)
        sc = CubQMCAmericanG(opt, abs_tol=0.1, n_train=2**11)
        solution, data = sc.integrate()

        # American put price under these parameters is ~6.0 - 6.1
        self.assertTrue(5.5 < solution < 6.5)
        self.assertEqual(data.n_train, 2**11)
        self.assertIsNotNone(data.betas)
        self.assertGreater(data.time_train, 0)

    def test_lattice_american_put(self):
        opt = FinancialOption(Lattice(self.d, seed=7), option="AMERICAN", call_put="PUT", **self.opt_params)
        sc = CubQMCAmericanG(opt, abs_tol=0.1, n_train=2**11)
        solution, data = sc.integrate()
        self.assertTrue(5.5 < solution < 6.5)

    def test_halton_american_put(self):
        opt = FinancialOption(Halton(self.d, seed=7), option="AMERICAN", call_put="PUT", **self.opt_params)
        sc = CubQMCAmericanG(opt, abs_tol=0.1, n_train=2**11)
        solution, data = sc.integrate()
        self.assertTrue(5.5 < solution < 6.5)

    def test_iid_american_put(self):
        opt = FinancialOption(IIDStdUniform(self.d, seed=7), option="AMERICAN", call_put="PUT", **self.opt_params)
        sc = CubQMCAmericanG(opt, abs_tol=0.1, n_train=2**11)
        solution, data = sc.integrate()
        self.assertTrue(5.5 < solution < 6.5)

    def test_training_size_stability(self):
        # Verify pricing stays stable as training sample size increases
        prices = []
        for n_tr in [2**9, 2**11, 2**12]:
            opt = FinancialOption(Sobol(self.d, seed=7), option="AMERICAN", call_put="PUT", **self.opt_params)
            sc = CubQMCAmericanG(opt, abs_tol=0.05, n_train=n_tr)
            sol, _ = sc.integrate()
            prices.append(sol)

        # Differences across training sizes should be small
        self.assertLess(max(prices) - min(prices), 0.2)

    def test_replicated_lattice_american_put(self):
        opt = FinancialOption(Lattice(self.d, seed=7, replications=16), option="AMERICAN", call_put="PUT", **self.opt_params)
        sc = CubQMCAmericanG(opt, abs_tol=0.1, n_train=2**11)
        solution, data = sc.integrate()
        self.assertTrue(5.5 < solution < 6.5)

    def test_custom_stopping_criterion(self):
        opt = FinancialOption(Sobol(self.d, seed=7, replications=16), option="AMERICAN", call_put="PUT", **self.opt_params)
        sc = CubQMCAmericanG(opt, abs_tol=0.1, n_train=2**11, stopping_criterion=CubQMCRepStudentT)
        solution, data = sc.integrate()
        self.assertTrue(5.5 < solution < 6.5)


if __name__ == "__main__":
    unittest.main()
