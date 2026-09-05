from qmcpy import (
    CubMCCLT,
    CubMCCLTVec,
    CubMCG,
    CubQMCBayesLatticeG,
    CubQMCBayesNetG,
    CubQMCCLT,
    CubQMCLatticeG,
    CubQMCSobolG,
    DigitalNet,
    FinancialOption,
    Halton,
    IIDStdUniform,
    Lattice,
    Sobol,
)
import unittest


class OptionTest(unittest.TestCase):

    d = 52
    options_args = {
        "option": "ASIAN",
        "asian_mean": "GEOMETRIC",
        "asian_mean_quadrature_rule": "RIGHT",
    }

    def test_cub_mc_g(self):
        atol = 5e-2
        integrand = FinancialOption(IIDStdUniform(self.d, seed=7), **self.options_args)
        true_value = integrand.get_exact_value()
        solution, data = CubMCG(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_cub_mc_clt(self):
        atol = 5e-2
        integrand = FinancialOption(IIDStdUniform(self.d, seed=7), **self.options_args)
        true_value = integrand.get_exact_value()
        solution, data = CubMCCLT(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_cub_mc_clt_vec(self):
        atol = 5e-2
        integrand = FinancialOption(IIDStdUniform(self.d, seed=7), **self.options_args)
        true_value = integrand.get_exact_value()
        solution, data = CubMCCLTVec(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_cub_qmc_sobol_g(self):
        atol = 5e-3
        integrand = FinancialOption(Sobol(self.d, seed=7), **self.options_args)
        true_value = integrand.get_exact_value()
        solution, data = CubQMCSobolG(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_cub_qmc_lattice_g(self):
        atol = 5e-3
        integrand = FinancialOption(Lattice(self.d, seed=7), **self.options_args)
        true_value = integrand.get_exact_value()
        solution, data = CubQMCLatticeG(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_cub_qmc_bayes_sobol_g(self):
        atol = 5e-3
        integrand = FinancialOption(Sobol(self.d, seed=7), **self.options_args)
        true_value = integrand.get_exact_value()
        solution, data = CubQMCBayesNetG(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_cub_qmc_bayes_lattice_g(self):
        atol = 5e-3
        integrand = FinancialOption(Lattice(self.d, seed=7), **self.options_args)
        true_value = integrand.get_exact_value()
        solution, data = CubQMCBayesLatticeG(
            integrand, abs_tol=atol, ptransform="BAKER"
        ).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_cub_qmc_clt_dnb2(self):
        atol = 5e-3
        integrand = FinancialOption(
            DigitalNet(self.d, seed=7, replications=32), **self.options_args
        )
        true_value = integrand.get_exact_value()
        solution, data = CubQMCCLT(integrand, abs_tol=atol).integrate()
        self.assertTrue(abs(solution - true_value) < atol)

    def test_cub_qmc_clt_lattice(self):
        atol = 5e-3
        integrand = FinancialOption(
            Lattice(self.d, seed=7, replications=32), **self.options_args
        )
        true_value = integrand.get_exact_value()
        solution, data = CubQMCCLT(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_cub_qmc_clt_halton(self):
        atol = 5e-3
        integrand = FinancialOption(
            Halton(self.d, seed=7, replications=32), **self.options_args
        )
        true_value = integrand.get_exact_value()
        solution, data = CubQMCCLT(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)


if __name__ == "__main__":
    unittest.main()
