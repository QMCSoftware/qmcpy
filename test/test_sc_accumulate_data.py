import unittest
from unittest.mock import patch

import numpy as np

from qmcpy import CubBayesNetG, DigitalNetB2, Keister
from qmcpy.stopping_criterion.pf_gp_ci import PFGPCIData


class _DummyDiscreteDistrib(object):
    def __init__(self, d):
        self.d = d


class _DummyModel(object):
    fit_calls = 0

    def __init__(self, x_t, y_t, prior_mean, prior_cov, likelihood, use_gpu):
        self.x_t = np.array(x_t)
        self.y_t = np.array(y_t)
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.likelihood = likelihood
        self.use_gpu = use_gpu

    def state_dict(self):
        return {"n": len(self.x_t)}

    def predict(self, x):
        x = np.asarray(x)
        # Deterministic outputs keep test assertions stable and fast.
        return np.zeros(len(x)), np.ones(len(x))

    def fit(self, optimizer, mll, training_iter, verbose):
        _DummyModel.fit_calls += 1

    def add_data(self, xdraw, ydrawtf):
        self.x_t = np.vstack([self.x_t, xdraw])
        self.y_t = np.hstack([self.y_t, ydrawtf])
        return self


class _DummyModelAddDataFails(_DummyModel):
    def add_data(self, xdraw, ydrawtf):
        raise RuntimeError("force refit path")


class TestPFGPCIDataFast(unittest.TestCase):
    def _make_data(self, refit=False, approx_true_solution=False):
        d = 2
        return PFGPCIData(
            stopping_crit=None,
            integrand=None,
            true_measure=None,
            discrete_distrib=_DummyDiscreteDistrib(d),
            dnb2=lambda n: np.linspace(0.0, 1.0, n * d).reshape(n, d),
            n_approx=8,
            alpha=0.1,
            refit=refit,
            gpytorch_prior_mean=None,
            gpytorch_prior_cov=None,
            gpytorch_likelihood=None,
            gpytorch_marginal_log_likelihood_func=lambda likelihood, model: None,
            torch_optimizer_func=lambda model: None,
            gpytorch_train_iter=1,
            gpytorch_use_gpu=False,
            verbose=False,
            approx_true_solution=approx_true_solution,
        )

    @patch("qmcpy.stopping_criterion.pf_gp_ci.torch.cuda.empty_cache", lambda: None)
    @patch("qmcpy.stopping_criterion.pf_gp_ci.ExactGPyTorchRegressionModel", _DummyModel)
    def test_update_data_add_data_path(self):
        _DummyModel.fit_calls = 0
        data = self._make_data(refit=False)
        xdraw = np.array([[0.1, 0.2], [0.2, 0.3], [0.7, 0.9]])
        ydrawtf = np.array([0.0, 1.0, -1.0])
        data.update_data(batch_count=1, xdraw=xdraw, ydrawtf=ydrawtf)

        self.assertEqual(data.n_batch, [3])
        self.assertEqual(data.x.shape, (3, 2))
        self.assertEqual(data.y.shape, (3,))
        self.assertEqual(len(data.saved_gps), 2)
        self.assertEqual(len(data.solutions), 1)
        self.assertEqual(len(data.error_bounds), 1)
        self.assertGreaterEqual(data.error_bounds[0], 0)

    @patch("qmcpy.stopping_criterion.pf_gp_ci.torch.cuda.empty_cache", lambda: None)
    @patch("qmcpy.stopping_criterion.pf_gp_ci.ExactGPyTorchRegressionModel", _DummyModel)
    def test_update_data_refit_path(self):
        _DummyModel.fit_calls = 0
        data = self._make_data(refit=True)
        xdraw = np.array([[0.05, 0.15], [0.25, 0.35]])
        ydrawtf = np.array([0.2, -0.1])
        data.update_data(batch_count=1, xdraw=xdraw, ydrawtf=ydrawtf)

        self.assertGreaterEqual(_DummyModel.fit_calls, 1)
        self.assertEqual(data.n_batch, [2])
        self.assertEqual(len(data.saved_gps), 2)

    @patch("qmcpy.stopping_criterion.pf_gp_ci.torch.cuda.empty_cache", lambda: None)
    @patch(
        "qmcpy.stopping_criterion.pf_gp_ci.ExactGPyTorchRegressionModel",
        _DummyModelAddDataFails,
    )
    def test_update_data_fallback_refit_on_add_data_error(self):
        _DummyModel.fit_calls = 0
        data = self._make_data(refit=False)
        xdraw = np.array([[0.1, 0.2], [0.8, 0.9]])
        ydrawtf = np.array([0.0, 1.0])
        data.update_data(batch_count=1, xdraw=xdraw, ydrawtf=ydrawtf)

        self.assertGreaterEqual(_DummyModel.fit_calls, 1)
        self.assertEqual(len(data.saved_gps), 2)

    @patch("qmcpy.stopping_criterion.pf_gp_ci.ExactGPyTorchRegressionModel", _DummyModel)
    def test_get_results_dict_with_approx_reference(self):
        data = self._make_data(refit=False, approx_true_solution=True)
        data.n_batch = [4, 4]
        data.n_sum = np.array([4, 8])
        data.error_bounds = [0.1, 0.05]
        data.ci_low = [0.2, 0.25]
        data.ci_high = [0.4, 0.35]
        data.solutions = [0.3, 0.3]
        data.solutions_ref = np.array([0.31, 0.31])
        data.error_ref = np.array([0.01, 0.01])
        data.in_ci = np.array([True, True])

        out = data.get_results_dict()
        self.assertIn("n_sum", out)
        self.assertIn("solutions_ref", out)
        self.assertIn("error_ref", out)
        self.assertIn("in_ci", out)


class TestLDBayesDataEquivalentFast(unittest.TestCase):
    def test_cub_bayes_net_g_returns_data_with_expected_fields(self):
        integrand = Keister(DigitalNetB2(2, seed=7))
        algo = CubBayesNetG(integrand, abs_tol=0.2, n_init=2**5, n_limit=2**6)
        solution, data = algo.integrate()

        self.assertTrue(np.isfinite(solution))
        self.assertTrue(hasattr(data, "comb_bound_low"))
        self.assertTrue(hasattr(data, "comb_bound_high"))
        self.assertTrue(hasattr(data, "comb_flags"))
        self.assertTrue(hasattr(data, "n_total"))
        self.assertLessEqual(data.n_total, algo.n_limit)


if __name__ == "__main__":
    unittest.main()
