from qmcpy import (
    CubMCCLT,
    CubMCCLTVec,
    CubMCG,
    CubQMCBayesLatticeG,
    CubQMCBayesNetG,
    CubQMCCLT,
    CubQMCLatticeG,
    CubQMCSobolG,
    CustomFun,
    DigitalNet,
    Halton,
    IIDStdUniform,
    Lattice,
    Lebesgue,
    Sobol,
    Uniform,
)
import numpy as np
import unittest


class PiProblemTest(unittest.TestCase):

    def test_cub_mc_g(self):
        atol = 1e-2
        integrand = CustomFun(
            true_measure=Lebesgue(
                Uniform(IIDStdUniform(1, seed=7), lower_bound=-2, upper_bound=2)
            ),
            g=lambda x: (np.sqrt(4 - x**2) * (1.0 / 2 + x**3 * np.cos(x / 2))).sum(-1),
        )
        solution, data = CubMCG(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - np.pi), atol)

    def test_cub_mc_clt(self):
        atol = 1e-2
        integrand = CustomFun(
            true_measure=Lebesgue(
                Uniform(IIDStdUniform(1, seed=7), lower_bound=-2, upper_bound=2)
            ),
            g=lambda x: (np.sqrt(4 - x**2) * (1.0 / 2 + x**3 * np.cos(x / 2))).sum(-1),
        )
        solution, data = CubMCCLT(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - np.pi), atol)

    def test_cub_mc_clt_vec(self):
        atol = 1e-2
        integrand = CustomFun(
            true_measure=Lebesgue(
                Uniform(IIDStdUniform(1, seed=7), lower_bound=-2, upper_bound=2)
            ),
            g=lambda x: (np.sqrt(4 - x**2) * (1.0 / 2 + x**3 * np.cos(x / 2))).sum(-1),
        )
        solution, data = CubMCCLTVec(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - np.pi), atol)

    def test_cub_qmc_sobol_g(self):
        atol = 1e-3
        integrand = CustomFun(
            true_measure=Lebesgue(
                Uniform(Sobol(1, seed=7), lower_bound=-2, upper_bound=2)
            ),
            g=lambda x: (np.sqrt(4 - x**2) * (1.0 / 2 + x**3 * np.cos(x / 2))).sum(-1),
        )
        solution, data = CubQMCSobolG(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - np.pi), atol)

    def test_cub_qmc_lattice_g(self):
        atol = 1e-3
        integrand = CustomFun(
            true_measure=Lebesgue(
                Uniform(Lattice(1, seed=7), lower_bound=-2, upper_bound=2)
            ),
            g=lambda x: (np.sqrt(4 - x**2) * (1.0 / 2 + x**3 * np.cos(x / 2))).sum(-1),
        )
        solution, data = CubQMCLatticeG(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - np.pi), atol)

    def test_cub_qmc_bayes_sobol_g(self):
        atol = 1e-3
        integrand = CustomFun(
            true_measure=Lebesgue(
                Uniform(Sobol(1, seed=7), lower_bound=-2, upper_bound=2)
            ),
            g=lambda x: (np.sqrt(4 - x**2) * (1.0 / 2 + x**3 * np.cos(x / 2))).sum(-1),
        )
        solution, data = CubQMCBayesNetG(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - np.pi), atol)

    def test_cub_qmc_bayes_lattice_g(self):
        atol = 1e-3
        integrand = CustomFun(
            true_measure=Lebesgue(
                Uniform(Lattice(1, seed=7), lower_bound=-2, upper_bound=2)
            ),
            g=lambda x: (np.sqrt(4 - x**2) * (1.0 / 2 + x**3 * np.cos(x / 2))).sum(-1),
        )
        solution, data = CubQMCBayesLatticeG(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - np.pi), atol)

    def test_cub_qmc_clt_dnb2(self):
        atol = 1e-3
        integrand = CustomFun(
            true_measure=Lebesgue(
                Uniform(
                    DigitalNet(1, seed=7, replications=32),
                    lower_bound=-2,
                    upper_bound=2,
                )
            ),
            g=lambda x: (np.sqrt(4 - x**2) * (1.0 / 2 + x**3 * np.cos(x / 2))).sum(-1),
        )
        solution, data = CubQMCCLT(integrand, abs_tol=atol).integrate()
        self.assertTrue(abs(solution - np.pi) < atol)

    def test_cub_qmc_clt_lattice(self):
        atol = 1e-3
        integrand = CustomFun(
            true_measure=Lebesgue(
                Uniform(
                    Lattice(1, seed=7, replications=32), lower_bound=-2, upper_bound=2
                )
            ),
            g=lambda x: (np.sqrt(4 - x**2) * (1.0 / 2 + x**3 * np.cos(x / 2))).sum(-1),
        )
        solution, data = CubQMCCLT(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - np.pi), atol)

    def test_cub_qmc_clt_halton(self):
        atol = 1e-3
        integrand = CustomFun(
            true_measure=Lebesgue(
                Uniform(
                    Halton(1, seed=7, replications=32), lower_bound=-2, upper_bound=2
                )
            ),
            g=lambda x: (np.sqrt(4 - x**2) * (1.0 / 2 + x**3 * np.cos(x / 2))).sum(-1),
        )
        solution, data = CubQMCCLT(integrand, abs_tol=atol).integrate()
        self.assertLess(abs(solution - np.pi), atol)


if __name__ == "__main__":
    unittest.main()
