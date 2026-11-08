
import numpy as np
import pytest
import unittest

torch = pytest.importorskip("pymc")

from qmcpy import DigitalNetB2
from qmcpy.util import (
    DimensionError,
)
import numpy as np
from scipy import stats

class TestPyMC(unittest.TestCase):
    """Unit tests for PyMCWrapper"""

    def setUp(self):
        import pymc
        self.models = []
        # define models for tests
        with pymc.Model() as normal_model:
            pymc.Normal("x", mu=0.0, sigma=1.0)
        self.models.append({"model": normal_model, "dims":1, "name": "normal",
                            "mean": 0.0, "weight": lambda x: 1/stats.norm.pdf(x[:, 0], loc=0.0, scale=1.0),
                            "transform_unconstrained_to_constrained": lambda x: x,
                            "jac_unconstrained_to_constrained": lambda x: 1.0})

        nu = 5.0
        with pymc.Model() as t_model:
            s = pymc.ChiSquared("s", nu=nu)
            pymc.Normal("x", mu=0.0, sigma=pymc.math.sqrt(nu/s))
        self.models.append({"model": t_model, "dims": 2, "name": "t_dist",
                            "mean": [nu, 0.0], "weight": lambda x: 1/(stats.norm.pdf(x[:,1], loc=0.0, scale=np.sqrt(nu/x[:,0]))*stats.chi2.pdf(x[:,0], df=nu)),
                            "transform_unconstrained_to_constrained": lambda x: np.stack([np.exp(x[:,0]), x[:,1]], axis=1),
                            "jac_unconstrained_to_constrained": lambda x: np.exp(x[:,0])})

        # test the observed variable is removed (only prior is sampled)
        with pymc.Model() as observed_t_model:
            s = pymc.ChiSquared("s", nu=nu)
            pymc.Normal("x", mu=0.0, sigma=pymc.math.sqrt(nu/s), observed=np.ones(3))
        self.models.append({"model": observed_t_model, "dims": 1, "name": "observed_t_dist",
                            "mean": [nu], "weight": lambda x: 1/(np.prod(stats.norm.pdf(x[:,1:], loc=0.0, scale=np.sqrt(nu/x[:,[0]])), axis=1)*stats.chi2.pdf(x[:,0], df=nu)),
                            "transform_unconstrained_to_constrained": np.exp,
                            "jac_unconstrained_to_constrained": lambda x: np.squeeze(np.exp(x))})

        # designed to catch wrong reordering in internal qmc model
        with pymc.Model() as model_that_gets_reordered:
            pymc.Normal("x", mu=0.0, sigma=1.0)
            pymc.ChiSquared("s", nu=nu)
        self.models.append({"model": model_that_gets_reordered, "dims": 2, "name": "model_that_gets_reordered",
                            "mean": [0.0, nu], "weight": lambda x: 1/(stats.norm.pdf(x[:,0], loc=0.0, scale=1.0)*stats.chi2.pdf(x[:,1], df=nu)),
                            "transform_unconstrained_to_constrained": lambda x: np.stack([x[:,0], np.exp(x[:,1])], axis=1),
                            "jac_unconstrained_to_constrained": lambda x: np.exp(x[:,1])})
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
                # this is designed to catch big errors due to transformations, model reordering etc
                # it is not designed to test bit exact results. The scipy pdfs are not exactly equivalent
                # to pymc, but very close. However in edge cases they can differ
                np.testing.assert_array_almost_equal(
                    w,
                    model_dict["weight"](s), 2
                )

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

    def test_unconstrained_space_samples(self):
        from qmcpy import PyMCWrapper
        for model_dict in self.models:
            with self.subTest(name=model_dict["name"]):
                m_constrained   = PyMCWrapper(DigitalNetB2(dimension=model_dict["dims"], seed=self.seed), model_dict["model"])
                m_unconstrained = PyMCWrapper(DigitalNetB2(dimension=model_dict["dims"], seed=self.seed), model_dict["model"], unconstrained_samples=True)
                samples               = m_constrained(8)
                unconstrained_samples = m_unconstrained(8)
                transform = model_dict["transform_unconstrained_to_constrained"]
                np.testing.assert_array_almost_equal(
                    samples, transform(unconstrained_samples)
                )

    def test_unconstrained_space_weights(self):
        from qmcpy import PyMCWrapper
        for model_dict in self.models:
            with self.subTest(name=model_dict["name"]):
                m_constrained   = PyMCWrapper(DigitalNetB2(dimension=model_dict["dims"], seed=self.seed), model_dict["model"])
                m_unconstrained = PyMCWrapper(DigitalNetB2(dimension=model_dict["dims"], seed=self.seed), model_dict["model"], unconstrained_samples=True)
                samples,               weights               = m_constrained(8, return_weights=True)
                unconstrained_samples, unconstrained_weights = m_unconstrained(8, return_weights=True)
                jac       = model_dict["jac_unconstrained_to_constrained"]
                np.testing.assert_array_almost_equal(
                    1/weights * jac(unconstrained_samples), 1/unconstrained_weights
                )

    def test_spawn(self):
        for model_dict in self.models:
            with self.subTest(name=model_dict["name"]):
                tm = self._make_pymcwrapper(model_dict)
                s = 3
                spawns = tm.spawn(s=s)
                self.assertEqual(len(spawns), s)
                self.assertTrue(all(type(spawn) == type(tm) for spawn in spawns))
                self.assertTrue(
                    (
                        all(
                            spawn.discrete_distrib != tm.discrete_distrib
                            for spawn in spawns
                        )
                    )
                )
                self.assertTrue(
                    all(spawn.transform != tm.transform for spawn in spawns)
                )

    def test_range_shape(self):
        for model_dict in self.models:
            with self.subTest(name=model_dict["name"]):
                tm = self._make_pymcwrapper(model_dict)
                self.assertEqual(tm.range.shape, (model_dict["dims"], 2))

if __name__ == "__main__":
    unittest.main()
