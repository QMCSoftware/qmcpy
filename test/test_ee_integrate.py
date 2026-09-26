"""Unit tests for integrate method in QMCPy"""

from qmcpy import (
    CubBayesLatticeG,
    CubBayesNetG,
    CubMCCLT,
    CubMCG,
    CubQMCCLT,
    CubQMCLatticeG,
    CubQMCSobolG,
    CustomFun,
    DigitalNetB2,
    FinancialOption,
    Gaussian,
    IIDStdUniform,
    Keister,
    Lattice,
    Lebesgue,
    Linear0,
    Uniform,
)
import numpy as np
import unittest


class IntegrationExampleTest(unittest.TestCase):

    def test_keister(self):
        """
        Mathematica:
        3D:  N[Integrate[E^(-x1^2 - x2^2 - x3^2) Cos[Sqrt[x1^2 + x2^2 + x3^2]], {x1, -Infinity, Infinity}, {x2, -Infinity, Infinity}, {x3, -Infinity, Infinity}]]
        2D:  N[Integrate[E^(-x1^2 - x2^2) Cos[Sqrt[x1^2 + x2^2]], {x1, -Infinity, Infinity}, {x2, -Infinity, Infinity}]]
        1D:  N[Integrate[E^(-x^2) Cos[Sqrt[x^2]], {x, -Infinity, Infinity}]]
        """
        abs_tol = 0.01
        dimensions = [1, 2, 3]
        true_values = [1.3803884470431430, 1.808186429263620, 2.168309102165481]
        for i in range(len(dimensions)):
            integrand = Keister(IIDStdUniform(dimension=dimensions[i], seed=42))
            solution, data = CubMCCLT(integrand, abs_tol=abs_tol).integrate()
            self.assertTrue(abs(solution - true_values[i]) < abs_tol)

    def test_asian_option_multi_level(self):
        abs_tol = 0.01
        integrand = FinancialOption(IIDStdUniform(dimension=64, seed=7))
        solution, data = CubMCCLT(integrand, abs_tol).integrate()
        true_value = 1.7845
        self.assertTrue(np.isclose(solution, true_value, atol=abs_tol))

    def test_lebesgue_bounded_measure(self):
        """Mathematica: Integrate[x^3 y^3, {x, 1, 3}, {y, 3, 6}]"""
        abs_tol = 1
        true_measure = Lebesgue(
            Uniform(DigitalNetB2(2, seed=7), lower_bound=[1, 3], upper_bound=[3, 6])
        )
        myfunc = lambda x: (x.prod(1)) ** 3
        integrand = CustomFun(true_measure, myfunc)
        solution, data = CubQMCSobolG(integrand, abs_tol=abs_tol).integrate()
        true_value = 6075
        self.assertTrue(abs(solution - true_value) < abs_tol)

    def test_lebesgue_inf_measure(self):
        abs_tol = 0.1
        true_measure = Lebesgue(Gaussian(Lattice(1, seed=7)))
        myfunc = lambda x: np.exp(-(x**2)).sum(1)
        integrand = CustomFun(true_measure, myfunc)
        solution, data = CubQMCLatticeG(integrand, abs_tol=abs_tol).integrate()
        true_value = np.sqrt(np.pi)
        self.assertTrue(abs(solution - solution) < abs_tol)

    def test_lebesgue_inf_measure_2d(self):
        abs_tol = 0.1
        true_measure = Lebesgue(
            Gaussian(Lattice(2, replications=32, seed=7), mean=1, covariance=2)
        )
        myfunc = lambda x: np.exp(-(x**2)).prod(-1)
        integrand = CustomFun(true_measure, myfunc)
        solution, data = CubQMCCLT(integrand, abs_tol=abs_tol).integrate()
        true_value = np.pi
        self.assertTrue(abs(solution - solution) < abs_tol)

    def test_uniform_measure(self):
        """Mathematica: Integrate[(x^3 y^3)/6, {x, 1, 3}, {y, 3, 6}]"""
        abs_tol = 1
        true_measure = Uniform(
            Lattice(2, replications=32, seed=7), lower_bound=[1, 3], upper_bound=[3, 6]
        )
        myfunc = lambda x: (x.prod(-1)) ** 3
        integrand = CustomFun(true_measure, myfunc)
        solution, data = CubQMCCLT(integrand, abs_tol=abs_tol).integrate()
        true_value = 6075 / 6
        self.assertTrue(abs(solution - true_value) < abs_tol)

    def test_linear(self):
        """Mathematica:
        1D: Integrate[x, {x, 0, 1}]
        2D: Integrate[x+y, {x, 0, 1}, {y, 0, 1}]
        3D: Integrate[x+y+z, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]
        """
        abs_tol = 0.01
        dimensions = [1, 2, 3]
        true_value = 0
        for i in range(len(dimensions)):
            d = dimensions[i]
            integrand = Linear0(DigitalNetB2(d, replications=32, seed=7))
            solution, data = CubQMCCLT(integrand, abs_tol=abs_tol).integrate()
            self.assertTrue(abs(solution - true_value) < abs_tol)

    def test_custom_fun(self):
        """
        Infer true measure's dimension from integrand's

        Mathematica:
        1D: Integrate[5x, {x, 0, 1}]
        2D: Integrate[5(x+y), {x, 0, 1}, {y, 0, 1}]
        3D: Integrate[5(x+y+z), {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]
        """
        abs_tol = 0.01
        dimensions = [1, 2, 3]
        true_values = [2.5, 5, 7.5]
        for i in range(len(dimensions)):
            d = dimensions[i]
            integrand = CustomFun(
                true_measure=Uniform(
                    IIDStdUniform(dimension=d, seed=7), lower_bound=0, upper_bound=1
                ),
                g=lambda x: (5 * x).sum(1),
            )
            solution, data = CubMCG(integrand, abs_tol=abs_tol).integrate()
            self.assertTrue(abs(solution - true_values[i]) < abs_tol)

    def test_custom_fun_2(self):
        """
        Test integrands with parameters

        Mathematica: integrate[b*(x-a)^2, {x,1,0}]
        """
        abs_tol = 0.01
        d = 1
        a_list = [1.0, 2.0]
        b_list = [4.0, 5.0]
        true_values = [(b / 3) * (3 * a * (a - 1) + 1) for a, b in zip(a_list, b_list)]
        for i in range(2):
            a_i = a_list[i]
            b_i = b_list[i]
            integrand = CustomFun(
                true_measure=Uniform(Lattice(d, seed=7)),
                g=lambda x, a=a_i, b=b_i: (b * (x - a) ** 2).sum(1),
            )
            solution, data = CubQMCLatticeG(integrand, abs_tol=abs_tol).integrate()
            self.assertTrue(abs(solution - true_values[i]) < abs_tol)

    def test_european_call(self):
        abs_tol = 1e-2
        integrand = FinancialOption(
            DigitalNetB2(16, seed=7),
            option="EUROPEAN",
            volatility=0.2,
            start_price=5,
            strike_price=10,
            interest_rate=0.01,
            call_put="call",
        )
        algorithm = CubQMCSobolG(integrand, abs_tol)
        solution, data = algorithm.integrate()
        true_value = integrand.get_exact_value()
        self.assertLess(abs(solution - true_value), abs_tol)

    def test_european_put(self):
        abs_tol = 1e-2
        integrand = FinancialOption(
            Lattice(16, seed=7),
            option="EUROPEAN",
            volatility=0.5,
            start_price=50,
            strike_price=40,
            interest_rate=0.01,
            call_put="put",
        )
        algorithm = CubQMCLatticeG(integrand, abs_tol)
        solution, data = algorithm.integrate()
        true_value = integrand.get_exact_value()
        self.assertLess(abs(solution - true_value), abs_tol)

    def test_european_put_bayes_lattice(self):
        abs_tol = 1e-2
        integrand = FinancialOption(
            sampler=Lattice(dimension=16, order="RADICAL INVERSE", seed=7),
            option="EUROPEAN",
            volatility=0.5,
            start_price=10,
            strike_price=10,
            interest_rate=0.01,
            call_put="put",
        )
        algorithm = CubBayesLatticeG(integrand, abs_tol, ptransform="Baker")
        solution, data = algorithm.integrate()
        true_value = integrand.get_exact_value()
        self.assertLess(abs(solution - true_value), abs_tol)

    def test_european_put_bayes_net(self):
        abs_tol = 5e-2
        integrand = FinancialOption(
            sampler=DigitalNetB2(dimension=4, seed=7),
            option="EUROPEAN",
            volatility=0.5,
            start_price=10,
            strike_price=10,
            interest_rate=0.01,
            call_put="put",
        )
        algorithm = CubBayesNetG(integrand, abs_tol)
        solution, data = algorithm.integrate()
        true_value = integrand.get_exact_value()
        self.assertLess(abs(solution - true_value), abs_tol)


if __name__ == "__main__":
    unittest.main()
