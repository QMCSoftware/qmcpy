import unittest
import numpy as np
from qmcpy import (
    FinancialOption,
    Sobol,
    Lattice,
    Halton,
    IIDStdUniform,
    CubQMCAmericanG,
)
from qmcpy.util import ParameterError


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
        self.assertLess(abs(prices[-1] - prices[-2]), 0.2)


if __name__ == "__main__":
    unittest.main()
