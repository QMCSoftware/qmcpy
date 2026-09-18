import unittest
import numpy as np
from qmcpy import FinancialOption, Sobol, IIDStdUniform
from qmcpy.util import ParameterError


class TestFinancialOptionAmerican(unittest.TestCase):

    def setUp(self):
        self.d = 10
        self.sampler = Sobol(self.d, seed=7)
        self.opt_params = {
            "volatility": 0.2,
            "start_price": 100.0,
            "strike_price": 100.0,
            "interest_rate": 0.05,
            "t_final": 1.0,
        }

    def test_construction_and_validation(self):
        # Valid American Put construction
        opt = FinancialOption(self.sampler, option="AMERICAN", call_put="PUT", **self.opt_params)
        self.assertEqual(opt.option, "AMERICAN")
        self.assertEqual(opt.call_put, "PUT")

        # Reject CALL for AMERICAN
        with self.assertRaises(ParameterError):
            FinancialOption(self.sampler, option="AMERICAN", call_put="CALL", **self.opt_params)

        # Reject level is not None for AMERICAN
        with self.assertRaises(ParameterError):
            FinancialOption(self.sampler, option="AMERICAN", call_put="PUT", level=0, **self.opt_params)

    def test_laguerre_basis(self):
        opt = FinancialOption(self.sampler, option="AMERICAN", call_put="PUT", **self.opt_params)
        # Test x = 0
        basis_0 = opt.laguerre_basis(0.0)
        np.testing.assert_allclose(basis_0, [1.0, 1.0, 1.0, 1.0])

        # Test x = 1
        exp_half = np.exp(-0.5)
        basis_1 = opt.laguerre_basis(1.0)
        np.testing.assert_allclose(basis_1, [1.0, exp_half, 0.0, -0.5 * exp_half])

        # Test batch input shape (N, 4)
        x_vec = np.linspace(0.5, 1.5, 100)
        basis_vec = opt.laguerre_basis(x_vec)
        self.assertEqual(basis_vec.shape, (100, 4))

    def test_lsm_policy_training_and_payoff(self):
        opt = FinancialOption(self.sampler, option="AMERICAN", call_put="PUT", **self.opt_params)
        # Verify payoff raises error if called before training
        gbm_test = opt.true_measure.gen_samples(128)
        with self.assertRaises(ParameterError):
            opt.payoff_american_put(gbm_test)

        # Train policy on training paths
        train_paths = opt.true_measure.gen_samples(1024)
        betas = opt.train_american_policy(train_paths)

        self.assertIsNotNone(opt.betas)
        self.assertEqual(len(opt.betas), self.d - 1)

        # Evaluate fixed-policy payoffs
        payoffs = opt.payoff_american_put(gbm_test)
        self.assertEqual(payoffs.shape, (128,))
        self.assertTrue((payoffs >= 0).all())

    def test_no_double_discounting(self):
        opt = FinancialOption(self.sampler, option="AMERICAN", call_put="PUT", **self.opt_params)
        train_paths = opt.true_measure.gen_samples(512)
        opt.train_american_policy(train_paths)

        gbm_test = opt.true_measure.gen_samples(128)
        payoff_vals = opt.payoff_american_put(gbm_test)
        g_vals = opt.g(gbm_test)

        # g(t) must return exact payoff without multiplying by exp(-r*T) again
        np.testing.assert_allclose(g_vals, payoff_vals)
        self.assertFalse(np.allclose(g_vals, payoff_vals * opt.discount_factor))


if __name__ == "__main__":
    unittest.main()
