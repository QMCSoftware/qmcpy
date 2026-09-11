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
    Halton,
    IIDStdUniform,
    Keister,
    Lattice,
    Sobol,
)
import unittest


class KeisterProblemTest(unittest.TestCase):

    d = 2

    def test_cub_mc_g(self):
        atol = 1e-2
        integrand = Keister(IIDStdUniform(self.d, seed=7))
        true_value = integrand.get_exact_value(self.d)
        solution, data = CubMCG(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_cub_mc_clt(self):
        atol = 1e-2
        integrand = Keister(IIDStdUniform(self.d, seed=7))
        true_value = integrand.get_exact_value(self.d)
        solution, data = CubMCCLT(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_cub_mc_clt_vec(self):
        atol = 1e-2
        integrand = Keister(IIDStdUniform(self.d, seed=7))
        true_value = integrand.get_exact_value(self.d)
        solution, data = CubMCCLTVec(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_cub_qmc_sobol_g(self):
        atol = 1e-4
        integrand = Keister(Sobol(self.d, seed=7))
        true_value = integrand.get_exact_value(self.d)
        solution, data = CubQMCSobolG(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_cub_qmc_lattice_g(self):
        atol = 1e-4
        integrand = Keister(Lattice(self.d, seed=7))
        true_value = integrand.get_exact_value(self.d)
        solution, data = CubQMCLatticeG(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_cub_qmc_bayes_sobol_g(self):
        atol = 1e-4
        integrand = Keister(Sobol(self.d, seed=7))
        true_value = integrand.get_exact_value(self.d)
        solution, data = CubQMCBayesNetG(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_cub_qmc_bayes_lattice_g(self):
        atol = 1e-4
        integrand = Keister(Lattice(self.d, seed=7))
        true_value = integrand.get_exact_value(self.d)
        solution, data = CubQMCBayesLatticeG(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_cub_qmc_clt_dnb2(self):
        atol = 1e-4
        integrand = Keister(DigitalNet(self.d, seed=7, replications=32))
        true_value = integrand.get_exact_value(self.d)
        solution, data = CubQMCCLT(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_cub_qmc_clt_lattice(self):
        atol = 1e-4
        integrand = Keister(Lattice(self.d, seed=7, replications=32))
        true_value = integrand.get_exact_value(self.d)
        solution, data = CubQMCCLT(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)

    def test_cub_qmc_clt_halton(self):
        atol = 1e-4
        integrand = Keister(Halton(self.d, seed=7, replications=32))
        true_value = integrand.get_exact_value(self.d)
        solution, data = CubQMCCLT(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - true_value), atol)


if __name__ == "__main__":
    unittest.main()
