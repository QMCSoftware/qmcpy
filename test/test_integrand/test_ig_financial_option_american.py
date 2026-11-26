import unittest
import numpy as np
from qmcpy import FinancialOption, Sobol
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
        opt.train_american_policy(train_paths)

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

    def test_early_exercise_deterministic_paths(self):
        # 3 monitoring steps: t1, t2, t3
        sampler = Sobol(3, seed=7)
        opt = FinancialOption(sampler, option="AMERICAN", call_put="PUT", strike_price=100.0, start_price=100.0, interest_rate=0.05, volatility=0.2, t_final=1.0)
        t_vec = opt.true_measure.time_vec
        t1, t2, t3 = t_vec[0], t_vec[1], t_vec[2]
        r = 0.05

        # Prescribe regression coefficients so that continuation values are known constants:
        # Step 0 (t1): c_hat = 20.0
        # Step 1 (t2): c_hat = 10.0
        opt.betas = [np.array([20.0, 0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0, 0.0])]

        # Construct deterministic paths:
        # Path 0: S=[60, 90, 95] -> At t1, v_imm=(100-60)*exp(-r*t1) ~ 39.34 > 20.0 -> Exercises at t1
        # Path 1: S=[85, 80, 90] -> At t1, v_imm=(100-85)*exp(-r*t1) ~ 14.75 < 20.0 -> Continues to t2
        #                           At t2, v_imm=(100-80)*exp(-r*t2) ~ 19.34 > 10.0 -> Exercises at t2
        # Path 2: S=[110, 110, 80] -> OTM at t1 & t2 -> Exercises at maturity t3 with payoff (100-80)*exp(-r*t3)
        # Path 3: S=[110, 110, 110] -> OTM at all steps -> Payoff is 0.0
        test_paths = np.array([
            [60.0, 90.0, 95.0],
            [85.0, 80.0, 90.0],
            [110.0, 110.0, 80.0],
            [110.0, 110.0, 110.0],
        ])

        payoffs = opt.payoff_american_put(test_paths)
        expected = np.array([
            (100.0 - 60.0) * np.exp(-r * t1),
            (100.0 - 80.0) * np.exp(-r * t2),
            (100.0 - 80.0) * np.exp(-r * t3),
            0.0,
        ])
        np.testing.assert_allclose(payoffs, expected)

    def test_backward_induction_cash_flow_update(self):
        # Verify backward induction updates cash flow vector upon early exercise
        sampler = Sobol(2, seed=7)
        opt = FinancialOption(sampler, option="AMERICAN", call_put="PUT", strike_price=100.0, start_price=100.0, interest_rate=0.05, volatility=0.2, t_final=1.0)
        # 10 paths where terminal payoff at t2 is 0 (S=120) but path 0 is deep ITM at t1 (S=50)
        paths = np.ones((10, 2)) * 120.0
        paths[0, 0] = 50.0  # Deep ITM at step 0
        betas = opt.train_american_policy(paths)
        self.assertEqual(len(betas), 1)
        self.assertIsNotNone(betas[0])


if __name__ == "__main__":
    unittest.main()
