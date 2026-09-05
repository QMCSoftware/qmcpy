from qmcpy import (
    CubMLMC,
    CubMLMCCont,
    CubMLQMC,
    CubMLQMCCont,
    DigitalNet,
    FinancialOption,
    IIDStdUniform,
)
import unittest


class OptionTest(unittest.TestCase):

    options_args = {
        "option": "ASIAN",
        "asian_mean": "GEOMETRIC",
        "start_price": 30,
        "strike_price": 31,
        "interest_rate": 0.05,
        "volatility": 0.75,
    }

    def test_mlmc(self):
        atol = 2.5e-2
        integrand = FinancialOption(IIDStdUniform(seed=7), **self.options_args)
        true_value = integrand.get_exact_value_inf_dim()
        solution, data = CubMLMC(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_mlmc_cont(self):
        atol = 2.5e-2
        integrand = FinancialOption(IIDStdUniform(seed=7), **self.options_args)
        true_value = integrand.get_exact_value_inf_dim()
        solution, data = CubMLMCCont(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_mlqmc(self):
        atol = 1e-3
        integrand = FinancialOption(
            DigitalNet(seed=7, replications=8), **self.options_args
        )
        true_value = integrand.get_exact_value_inf_dim()
        solution, data = CubMLQMC(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_mlqmc_cont(self):
        atol = 1e-3
        integrand = FinancialOption(
            DigitalNet(seed=7, replications=8), **self.options_args
        )
        true_value = integrand.get_exact_value_inf_dim()
        solution, data = CubMLQMCCont(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)


if __name__ == "__main__":
    unittest.main()
