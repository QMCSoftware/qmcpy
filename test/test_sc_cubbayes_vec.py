import unittest
import qmcpy as qp
import numpy as np
import scipy
from sklearn.gaussian_process import GaussianProcessRegressor, kernels


class TestCubBayesVec(unittest.TestCase):

    def test_cantilever_beam(self):
        def cantilever_beam_function(T, compute_flags):  # T is (n x 3)
            Y = np.zeros((2, len(T)), dtype=float)  # (n x 2)
            l, w, t = 100, 4, 2
            T1, T2, T3 = T[:, 0], T[:, 1], T[:, 2]  # Python is indexed from 0
            if compute_flags[0]:  # compute D. x^2 is "x**2" in Python
                Y[0, :] = 4 * l**3 / (T1 * w * t) * np.sqrt(T2**2 / t**4 + T3**2 / w**4)
            if compute_flags[1]:  # compute S
                Y[1, :] = 600 * (T2 / (w * t**2) + T3 / (w**2 * t))
            return Y

        true_measure = qp.Gaussian(
            sampler=qp.DigitalNetB2(dimension=3, seed=7),
            mean=[2.9e7, 500, 1000],
            covariance=np.diag([(1.45e6) ** 2, (100) ** 2, (100) ** 2]),
        )
        integrand = qp.CustomFun(
            true_measure, g=cantilever_beam_function, dimension_indv=2
        )
        qmc_stop_crit = qp.CubBayesNetG(
            integrand,
            abs_tol=5e-2,
        )
        solution, data = qmc_stop_crit.integrate()
        expected = [2.42575885e00, 3.75000056e04]
        self.assertTrue((abs(solution - expected).mean() < 5e-2).all())

    def test_bayesian_opt(self):
        # Bayesian Optimization using q-Expected Improvement
        f = lambda x: np.cos(10 * x) * np.exp(0.2 * x) + np.exp(-5 * (x - 0.4) ** 2)
        xplt = np.linspace(0, 1, 100)
        yplt = f(xplt)
        x = np.array([0.1, 0.2, 0.4, 0.7, 0.9])
        y = f(x)
        ymax = y.max()

        gp = GaussianProcessRegressor(
            kernel=kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
            n_restarts_optimizer=16,
        ).fit(x[:, None], y)
        yhatplt, stdhatplt = gp.predict(xplt[:, None], return_std=True)

        tpax = 32
        x0mesh, x1mesh = np.meshgrid(np.linspace(0, 1, tpax), np.linspace(0, 1, tpax))
        post_mus = np.zeros((tpax, tpax, 2), dtype=float)
        post_sqrtcovs = np.zeros((tpax, tpax, 2, 2), dtype=float)
        for j0 in range(tpax):
            for j1 in range(tpax):
                candidate = np.array([[x0mesh[j0, j1]], [x1mesh[j0, j1]]])
                post_mus[j0, j1], post_cov = gp.predict(candidate, return_cov=True)
                evals, evecs = scipy.linalg.eig(post_cov)
                post_sqrtcovs[j0, j1] = np.sqrt(np.maximum(evals.real, 0)) * evecs

        def qei_acq_vec(x, compute_flags):
            xgauss = scipy.stats.norm.ppf(x)
            n = len(x)
            qei_vals = np.zeros((tpax, tpax, n), dtype=float)
            for j0 in range(tpax):
                for j1 in range(tpax):
                    if compute_flags[j0, j1] == False:
                        continue
                    sqrt_cov = post_sqrtcovs[j0, j1]
                    mu_post = post_mus[j0, j1]
                    for i in range(len(x)):
                        yij = sqrt_cov @ xgauss[i] + mu_post
                        qei_vals[j0, j1, i] = max((yij - ymax).max(), 0)
            return qei_vals

        qei_acq_vec_qmcpy = qp.CustomFun(
            true_measure=qp.Uniform(qp.DigitalNetB2(2, seed=7)),
            g=qei_acq_vec,
            dimension_indv=(tpax, tpax),
            parallel=False,
        )
        qei_vals_true, qei_data_true = qp.CubQMCNetG(
            qei_acq_vec_qmcpy, abs_tol=0.025, rel_tol=0
        ).integrate()  # .0005
        # print(qei_data_true)

        qei_vals, qei_data = qp.CubBayesNetG(
            qei_acq_vec_qmcpy, abs_tol=0.025, rel_tol=0
        ).integrate()  # .0005
        # print(qei_vals)

        a = np.unravel_index(np.argmax(qei_vals, axis=None), qei_vals.shape)
        xnext = np.array([x0mesh[a[0], a[1]], x1mesh[a[0], a[1]]])
        fnext = f(xnext)
        self.assertTrue((abs(qei_vals_true - qei_vals) < 0.005).all())

    def test_ishigami_func(self):
        a, b = 7, 0.1
        dnb2 = qp.DigitalNetB2(3, seed=7)
        ishigami = qp.Ishigami(dnb2, a, b)
        idxs = np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [True, True, False],
                [True, False, True],
                [False, True, True],
            ],
            dtype=bool,
        )
        ishigami_si = qp.SensitivityIndices(ishigami, idxs)
        qmc_algo = qp.CubBayesNetG(ishigami_si, abs_tol=0.05)
        solution, data = qmc_algo.integrate()
        # print(data)
        si_closed = solution[0].squeeze()
        si_total = solution[1].squeeze()
        ci_comb_low_closed = data.comb_bound_low[0].squeeze()
        ci_comb_high_closed = data.comb_bound_high[0].squeeze()
        ci_comb_low_total = data.comb_bound_low[1].squeeze()
        ci_comb_high_total = data.comb_bound_high[1].squeeze()
        # print("\nApprox took %.1f sec and n = 2^(%d)" %
        #       (data.time_integrate, np.log2(data.n_total)))
        # print('\t si_closed:', si_closed)
        # print('\t si_total:', si_total)
        # print('\t ci_comb_low_closed:', ci_comb_low_closed)
        # print('\t ci_comb_high_closed:', ci_comb_high_closed)
        # print('\t ci_comb_low_total:', ci_comb_low_total)
        # print('\t ci_comb_high_total:', ci_comb_high_total)

        true_indices = qp.Ishigami._exact_sensitivity_indices(idxs, a, b)
        si_closed_true = true_indices[0]
        si_total_true = true_indices[1]

        self.assertTrue(abs(si_closed - si_closed_true).mean() < 0.05)
        self.assertTrue(abs(si_total - si_total_true).mean() < 0.05)


if __name__ == "__main__":
    unittest.main()
