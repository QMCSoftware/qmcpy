
import numpy as np
import pytest
import unittest

torch = pytest.importorskip("pymc")

from qmcpy import DigitalNetB2
from qmcpy.util import (
    DimensionError,
)
import numpy as np

class TestPyMC(unittest.TestCase):
    """Unit tests for PyMCWrapper"""

    def setUp(self):
        import pymc
        self.models = []
        # define models for tests
        with pymc.Model() as normal_model:
            pymc.Normal("x", mu=0.0, sigma=1.0)
        self.models.append({"model": normal_model, "dims":1, "name": "normal", "mean": 0.0})

        nu = 5.0
        with pymc.Model() as t_model:
            s = pymc.ChiSquared("s", nu=nu)
            pymc.Normal("x", mu=0.0, sigma=pymc.math.sqrt(nu/s))
        self.models.append({"model": t_model, "dims": 2, "name": "t_dist", "mean": [5.0, 0.0]})

        with pymc.Model() as observed_t_model:
            s = pymc.ChiSquared("s", nu=nu)
            pymc.Normal("x", mu=0.0, sigma=pymc.math.sqrt(nu/s), observed=np.ones(3))
        self.models.append({"model": observed_t_model, "dims": 4, "name": "observed_t_dist",
                            "mean": [5.0, 0.0, 0.0, 0.0]})
        self.seed = 7

    def _make_sampler(self, dims, seed=None):
        seed = seed if seed is not None else self.seed
        return DigitalNetB2(dimension=dims, seed=seed)

    def _make_pymcwrapper(self, model_dict):
        from qmcpy import PyMCWrapper
        return PyMCWrapper(self._make_sampler(model_dict["dims"]), model_dict["model"])

    # --- PyMCWrapper tests ---

    def test_basic_shape(self):
        """gen_samples(n=64) returns correct shape (64, dims)."""
        for model_dict in self.models:
            with self.subTest(name=model_dict["name"]):
                samples = self._make_pymcwrapper(model_dict).gen_samples(n=64, warn=False)
                self.assertEqual(samples.shape, (64, model_dict["dims"]))

    def test_mean_convergence(self):
        """Sample mean should be close to the true mean of 0.0."""
        for model_dict in self.models:
            with self.subTest(name=model_dict["name"]):
                if model_dict["mean"] is not None:
                    samples = self._make_pymcwrapper(model_dict).gen_samples(2**10, warn=False)
                    np.testing.assert_allclose(samples.mean(0), model_dict["mean"], atol=0.05)
                    # self.assertAlmostEqual(samples.mean(0), model_dict["mean"], delta=0.05)

    def test_return_weights(self):
        """return_weights=True gives samples and positive weights of correct shape."""
        for model_dict in self.models:
            with self.subTest(name=model_dict["name"]):
                m = self._make_pymcwrapper(model_dict)
                s, w = m.gen_samples(n=64, return_weights=True, warn=False)
                self.assertEqual(s.shape, (64, model_dict["dims"]))
                self.assertEqual(w.shape, (64,))
                self.assertTrue(np.all(w > 0))

    def test_continued_sampling_matches_single_call(self):
        """Two batches via n_min/n_max equal one single call."""
        for model_dict in self.models:
            with self.subTest(name=model_dict["name"]):
                m1 = self._make_pymcwrapper(model_dict)
                b1 = m1.gen_samples(n_min=0, n_max=8)
                b2 = m1.gen_samples(n_min=8, n_max=16)

                m2 = self._make_pymcwrapper(model_dict)
                all_at_once = m2.gen_samples(n_min=0, n_max=16)

                np.testing.assert_array_almost_equal(
                    np.concatenate([b1, b2]), all_at_once
                )

    def test_n_resets_driver(self):
        """Calling gen_samples(n=8) twice gives same result (driver resets)."""
        for model_dict in self.models:
            with self.subTest(name=model_dict["name"]):
                m = self._make_pymcwrapper(model_dict)
                s1 = m.gen_samples(n=8)
                s2 = m.gen_samples(n=8)
                np.testing.assert_array_equal(s1, s2)

    def test_error_invalid_dimension(self):
        """wrong sampler dimension raises DimensionError."""
        from qmcpy import PyMCWrapper
        for model_dict in self.models:
            with self.subTest(name=model_dict["name"]):
                with self.assertRaises(DimensionError):
                    PyMCWrapper(
                        DigitalNetB2(dimension=10, seed=self.seed),
                        model_dict["model"]
                    )

    def test_replications(self):
        """replications work as expected"""
        from qmcpy import PyMCWrapper
        for model_dict in self.models:
            with self.subTest(name=model_dict["name"]):
                m = PyMCWrapper(DigitalNetB2(dimension=model_dict["dims"], replications=2, seed=self.seed), model_dict["model"])
                s = m.gen_samples(n=8)
                self.assertEqual(s.shape, (2, 8, model_dict["dims"]))

if __name__ == "__main__":
    unittest.main()
