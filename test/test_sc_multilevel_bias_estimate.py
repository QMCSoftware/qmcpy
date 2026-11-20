"""Regression tests for the multilevel bias (remaining weak error) estimates.

With level corrections Y_l = E[P_l - P_{l-1}] = c 2^{-alpha l}, the bias after the
finest level L is sum_{l>L} Y_l = Y_L / (2^alpha - 1). The estimators previously
extrapolated to level L+1 before summing the tail, so they returned
Y_L 2^{-alpha} / (2^alpha - 1), which is too small by a factor of 2^alpha. That
under-estimate made CubMLQMCCont and CubMLMCCont stop at too coarse a level and
miss the requested tolerance.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from qmcpy import CubMLMCCont, CubMLQMC, DigitalNetB2, FinancialOption, IIDStdUniform


class MultilevelBiasEstimateTest(unittest.TestCase):

    def _level_means(self, levels, alpha, c=0.3):
        return c * 2.0 ** (-alpha * np.arange(levels))

    def test_mlqmc_bias_estimate(self):
        sc = CubMLQMC(FinancialOption(DigitalNetB2(seed=7, replications=4)), abs_tol=1e-2)
        for alpha in [1.0, 1.5, 2.0]:
            for levels in [3, 5]:
                with self.subTest(alpha=alpha, levels=levels):
                    means = self._level_means(levels, alpha)
                    data = SimpleNamespace(levels=levels, mean_level_reps=np.tile(means[:, None], (1, 4)))
                    sc._update_bias_estimate(data)
                    expected = means[-1] / (2**alpha - 1)
                    self.assertAlmostEqual(data.bias_estimate, expected, places=12)

    def test_mlmc_cont_bias_estimate(self):
        sc = CubMLMCCont(FinancialOption(IIDStdUniform(seed=7)), abs_tol=1e-2)
        for alpha in [1.0, 2.0]:
            levels = 5
            with self.subTest(alpha=alpha):
                means = self._level_means(levels, alpha)
                n_level = np.full(levels, 100)
                data = SimpleNamespace(sum_level=np.vstack([means * n_level, means**2 * n_level]), n_level=n_level)
                expected = means[-1] / (2**alpha - 1)
                self.assertAlmostEqual(float(sc._bias(data, levels - 1)), expected, places=12)


if __name__ == "__main__":
    unittest.main()
