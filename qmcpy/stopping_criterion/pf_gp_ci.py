from .abstract_stopping_criterion import AbstractStoppingCriterion
from ..discrete_distribution import DigitalNetB2
from ..integrand.ishigami import Ishigami
from ..true_measure.abstract_true_measure import AbstractTrueMeasure
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
from ..util.data import Data

from ..util import ExactGPyTorchRegressionModel

from ..util import MaxSamplesWarning, ParameterError
import warnings
import time
import numpy as np
from scipy.stats import norm
import torch
import gpytorch


class Suggester(object):
    pass


class PFSampleErrorDensityAR(Suggester):
    def __init__(self, verbose=False):
        self.verbose = verbose
        super(PFSampleErrorDensityAR, self).__init__()

    def suggest(self, n, d, gp, rng, efficiency, pct=0.5):
        if self.verbose:
            print(
                "\tAR sampling with efficiency %.1e, expect %d draws: "
                % (efficiency, int(n / efficiency)),
                end="",
                flush=True,
            )
        x = np.zeros((0, d), dtype=float)
        self.n_ar_tries_batch = 0
        rem = n
        draws_per_rem = int(np.ceil(np.log(pct) / np.log(1 - efficiency)))
        while rem > 0:
            newtries = rem * draws_per_rem
            self.n_ar_tries_batch += newtries
            if self.verbose:
                print("%d, " % self.n_ar_tries_batch, end="", flush=True)
            z = rng.random((newtries, d))
            u = rng.random(newtries)
            udens_z = _error_udens(gp, z)
            x = np.vstack([x, z[u <= udens_z]])
            rem = max(n - len(x), 0)
        if self.verbose:
            print()
        return x[:n]


class SuggesterSimple(Suggester):
    def __init__(self, sampler):
        self.sampler = sampler
        if isinstance(self.sampler, AbstractTrueMeasure):
            assert (self.sampler.range == [0, 1]).all()
        self.n_min = 0
        super(SuggesterSimple, self).__init__()

    def suggest(self, n, d, gp, rng, **kwargs):
        n_max = self.n_min + n
        assert d == self.sampler.d
        try:
            x = self.sampler(n_min=self.n_min, n_max=n_max)
        except TypeError:
            x = self.sampler(n)
        self.n_min = n_max
        return x


class PFGPCI(AbstractStoppingCriterion):
    """
    Probability of failure estimation using adaptive Gaussian process construction and resulting credible intervals.

    Examples:
        >>> pfgpci = PFGPCI(
        ...     integrand = Ishigami(DigitalNetB2(3,seed=17)),
        ...     failure_threshold = 0,
        ...     failure_above_threshold = False,
        ...     abs_tol = 2.5e-2,
        ...     alpha = 1e-1,
        ...     n_init = 64,
        ...     init_samples = None,
        ...     batch_sampler = PFSampleErrorDensityAR(verbose=False),
        ...     n_batch = 16,
        ...     n_limit = 128,
        ...     n_approx = 2**18,
        ...     gpytorch_prior_mean = gpytorch.means.ZeroMean(),
        ...     gpytorch_prior_cov = gpytorch.kernels.ScaleKernel(
        ...         gpytorch.kernels.MaternKernel(nu=2.5,lengthscale_constraint = gpytorch.constraints.Interval(.5,1)),
        ...         outputscale_constraint = gpytorch.constraints.Interval(1e-8,.5)),
        ...     gpytorch_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint = gpytorch.constraints.Interval(1e-12,1e-8)),
        ...     gpytorch_marginal_log_likelihood_func = lambda likelihood,gpyt_model: gpytorch.mlls.ExactMarginalLogLikelihood(likelihood,gpyt_model),
        ...     torch_optimizer_func = lambda gpyt_model: torch.optim.Adam(gpyt_model.parameters(),lr=0.1),
        ...     gpytorch_train_iter = 100,
        ...     gpytorch_use_gpu = False,
        ...     verbose = False,
        ...     n_ref_approx = 2**22,
        ...     seed_ref_approx = 11)
        >>> solution,data = pfgpci.integrate(seed=7,refit=True)
        >>> data
        PFGPCIData (Data)
            solution        0.158
            error_bound     0.022
            bound_low       0.136
            bound_high      0.180
            n_total         112
            time_integrate  ...
        PFGPCI (AbstractStoppingCriterion)
            abs_tol         0.025
            n_init          2^(6)
            n_limit         2^(7)
            n_batch         2^(4)
        Ishigami (AbstractIntegrand)
        Uniform (AbstractTrueMeasure)
            lower_bound     -3.142
            upper_bound     3.142
        DigitalNetB2 (AbstractLDDiscreteDistribution)
            d               3
            replications    1
            randomize       LMS DS
            gen_mats_source joe_kuo.6.21201.txt
            order           RADICAL INVERSE
            t               63
            alpha           1
            n_limit         2^(32)
            entropy         17
        >>> df = data.get_results_dict()
        >>> with np.printoptions(formatter={"float": lambda x: "%-10.2e"%x, "int": lambda x: "%-10d"%x, "bool": lambda x: "%-10s"%x}):
        ...     for k,v in df.items():
        ...         print("%15s: %s"%(k,str(v)))
                   iter: [0          1          2          3         ]
                  n_sum: [64         80         96         112       ]
                n_batch: [64         16         16         16        ]
           error_bounds: [5.58e-02   3.92e-02   3.05e-02   2.16e-02  ]
                 ci_low: [8.66e-02   1.16e-01   1.19e-01   1.36e-01  ]
                ci_high: [1.98e-01   1.95e-01   1.80e-01   1.80e-01  ]
              solutions: [1.42e-01   1.55e-01   1.50e-01   1.58e-01  ]
          solutions_ref: [1.62e-01   1.62e-01   1.62e-01   1.62e-01  ]
              error_ref: [2.01e-02   7.02e-03   1.28e-02   4.52e-03  ]
                  in_ci: [True       True       True       True      ]

    **References:**

    1.  Sorokin, Aleksei G., and Vishwas Rao.
        "Credible Intervals for Probability of Failure with Gaussian Processes."
        arXiv preprint arXiv:2311.07733 (2023).
    """

    def __init__(
        self,
        integrand,
        failure_threshold,
        failure_above_threshold,
        abs_tol=5e-3,
        n_init=64,
        n_limit=1000,
        alpha=1e-2,
        init_samples=None,
        batch_sampler=PFSampleErrorDensityAR(),
        n_batch=4,
        n_approx=2**20,
        gpytorch_prior_mean=gpytorch.means.ZeroMean(),
        gpytorch_prior_cov=gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        ),
        gpytorch_likelihood=gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(1e-12, 1e-8)
        ),
        gpytorch_marginal_log_likelihood_func=lambda likelihood, gpyt_model: gpytorch.mlls.ExactMarginalLogLikelihood(
            likelihood, gpyt_model
        ),
        torch_optimizer_func=lambda gpyt_model: torch.optim.Adam(
            gpyt_model.parameters(), lr=0.1
        ),
        gpytorch_train_iter=100,
        gpytorch_use_gpu=False,
        verbose=False,
        n_ref_approx=2**22,
        seed_ref_approx=None,
    ):
        """
        Args:
            integrand (AbstractIntegrand): The integrand.
            failure_threshold (float): Thresholds for failure.
            failure_above_threshold (bool): Set to `True` if failure occurs when the simulation exceeds `failure_threshold` and False otherwise.
            abs_tol (float): The desired maximum distance from the estimate to either end of the credible interval.
            n_init (float): Initial number of samples from integrand.discrete_distrib from which to build the first surrogate GP
            n_limit (int): Budget of simulations.
            n_batch (int): The number of samples per batch to draw from batch_sampler.
            alpha (float): The credible interval is constructed to hold with probability at least 1 - alpha
            init_samples (float): If the simulation has already been run, pass in (x,y) where x are past samples from the discrete distribution and y are corresponding simulation evaluations.
            batch_sampler (Suggester or AbstractDiscreteDistribution): A suggestion scheme for future samples.
            n_approx (int): Number of points from integrand.discrete_distrib used to approximate estimate and credible interval bounds
            gpytorch_prior_mean (gpytorch.means): prior mean function of the GP
            gpytorch_prior_cov (gpytorch.kernels): Prior covariance kernel of the GP
            gpytorch_likelihood (gpytorch.likelihoods): GP likelihood, require one of gpytorch.likelihoods.{GaussianLikelihood, GaussianLikelihoodWithMissingObs, FixedNoiseGaussianLikelihood}
            gpytorch_marginal_log_likelihood_func (callable): Function taking in the likelihood and gpytorch model and returning a marginal log likelihood from gpytorch.mlls
            torch_optimizer_func (callable): Function taking in the gpytorch model and returning an optimizer from torch.optim
            gpytorch_train_iter (int): Training iterations for the GP in gpytorch
            gpytorch_use_gpu (bool): If True, have gpytorch use a GPU for fitting and trining the GP
            verbose (int): If verbose > 0, print information through the call to integrate()
            n_ref_approx (int): If n_ref_approx > 0, use n_ref_approx points to get a reference QMC approximation of the true solution.
                Caution: If n_ref_approx > 0, it should be a large int e.g. 2**22, in which case it is only helpful for cheap to evaluate simulations
            seed_ref_approx (int): Seed for the reference approximation. Only applies when n_ref_approx>0
        """
        self.parameters = ["abs_tol", "n_init", "n_limit", "n_batch"]
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        self.d = self.integrand.d
        self.sampler = self.d
        self.failure_threshold = failure_threshold
        self.failure_above_threshold = failure_above_threshold
        self.abs_tol = abs_tol
        self.alpha = alpha
        assert 0 < self.alpha < 1
        self.n_init = n_init
        self.init_samples = init_samples is not None
        if self.init_samples:
            self.x_init, self.y_init = init_samples
            assert self.x_init.ndim == 2 and self.y_init.ndim == 1
            assert self.x_init.shape[1] == self.d and len(self.y_init) == len(
                self.x_init
            )
            assert self.n_init == len(self.x_init)
            self.ytf_init = self._affine_tf(self.y_init)
        self.batch_sampler = batch_sampler
        self.n_batch = n_batch
        self.n_limit = n_limit
        assert self.n_limit >= self.n_init
        self.n_approx = n_approx
        assert (self.n_approx + self.n_init) <= 2**32
        self.gpytorch_prior_mean = gpytorch_prior_mean
        self.gpytorch_prior_cov = gpytorch_prior_cov
        self.gpytorch_likelihood = gpytorch_likelihood
        self.gpytorch_marginal_log_likelihood_func = (
            gpytorch_marginal_log_likelihood_func
        )
        self.torch_optimizer_func = torch_optimizer_func
        self.gpytorch_train_iter = gpytorch_train_iter
        self.gpytorch_use_gpu = gpytorch_use_gpu
        self.verbose = verbose
        self.approx_true_solution = n_ref_approx > 0
        if self.approx_true_solution:
            x = DigitalNetB2(self.d, order="GRAY", seed=seed_ref_approx)(n_ref_approx)
            y = self.integrand.f(x).squeeze()
            ytf = self._affine_tf(y)
            self.ref_approx = (ytf >= 0).mean(0)
            if self.verbose:
                print(
                    "reference approximation with d=%d: %s" % (self.d, self.ref_approx)
                )
        super(PFGPCI, self).__init__(
            allowed_distribs=[AbstractDiscreteDistribution],
            allow_vectorized_integrals=False,
        )

    def _affine_tf(self, y):
        return (
            y - self.failure_threshold
            if self.failure_above_threshold
            else self.failure_threshold - y
        )

    def integrate(self, seed=None, refit=False, resume=None):
        t0 = time.time()
        trace = self._make_trace_logger()
        if resume is not None:
            raise ParameterError("PFGPCI does not support resume.")
        dnb2 = DigitalNetB2(self.d, randomize="DS", order="GRAY", seed=seed)
        data = PFGPCIData(
            self,
            self.integrand,
            self.true_measure,
            self.discrete_distrib,
            dnb2,
            self.n_approx,
            self.alpha,
            refit,
            self.gpytorch_prior_mean,
            self.gpytorch_prior_cov,
            self.gpytorch_likelihood,
            self.gpytorch_marginal_log_likelihood_func,
            self.torch_optimizer_func,
            self.gpytorch_train_iter,
            self.gpytorch_use_gpu,
            self.verbose,
            self.approx_true_solution,
        )
        batch_count = 0
        while True:
            if self.verbose:
                print("batch %d" % batch_count)
            if batch_count == 0:
                if self.init_samples:
                    xdraw, ydraw = self.x_init, self.y_init
                else:
                    xdraw = dnb2.spawn()[0](self.n_init)
                    ydraw = np.atleast_1d(self.integrand.f(xdraw).squeeze())
            else:
                n_new = min(self.n_batch, self.n_limit - np.sum(self.n_batch))
                xdraw = self.batch_sampler.suggest(
                    n_new,
                    self.d,
                    data.gpyt_model,
                    dnb2.rng,
                    efficiency=2 * data.emr[-1],
                )
                ydraw = np.atleast_1d(self.integrand.f(xdraw).squeeze())
            ydrawtf = self._affine_tf(ydraw)
            data.update_data(batch_count, xdraw, ydrawtf)
            data.solution = data.solutions[-1]
            data.error_bound = data.error_bounds[-1]
            data.bound_low = data.ci_low[-1]
            data.bound_high = data.ci_high[-1]
            data.bound_diff = data.bound_high - data.bound_low
            data.bound_half_width = data.error_bound
            data.n_total = int(sum(data.n_batch))
            self._set_elapsed_time(data, time.time() - t0)
            trace.iteration(data)
            batch_count += 1
            if data.error_bound <= self.abs_tol:
                break
            if sum(data.n_batch) == self.n_limit:
                warnings.warn("n_limit reached. ", MaxSamplesWarning)
                break
        elapsed = time.time() - t0
        self._set_elapsed_time(data, elapsed)
        self._finalize_integration_data(data, elapsed)
        data.n_sum = np.cumsum(data.n_batch)
        data.n_batch = np.array(data.n_batch)
        data.error_bounds = np.array(data.error_bounds)
        data.ci_low = np.array(data.ci_low)
        data.ci_high = np.array(data.ci_high)
        data.solutions = np.array(data.solutions)
        if self.approx_true_solution:
            data.solutions_ref = np.tile(self.ref_approx, len(data.n_batch))
            data.error_ref = abs(data.solutions - data.solutions_ref)
            data.in_ci = (data.ci_low <= data.solutions_ref) * (
                data.solutions_ref <= data.ci_high
            )
        trace.finalize()
        return data.solution, data


def _get_phi(gp, x):
    yhat, yhatstd = gp.predict(x)
    with np.errstate(all="ignore"):
        z = yhat / yhatstd
    return norm.cdf(z)


def _error_udens_from_phi(phi):
    return 2 * np.minimum(1 - phi, phi)


def _error_udens(gp, x):
    return _error_udens_from_phi(_get_phi(gp, x))


class PFGPCIData(Data):
    """Update and store data for the PFGPCI AbstractStoppingCriterion."""

    def __init__(
        self,
        stopping_crit,
        integrand,
        true_measure,
        discrete_distrib,
        dnb2,
        n_approx,
        alpha,
        refit,
        gpytorch_prior_mean,
        gpytorch_prior_cov,
        gpytorch_likelihood,
        gpytorch_marginal_log_likelihood_func,
        torch_optimizer_func,
        gpytorch_train_iter,
        gpytorch_use_gpu,
        verbose,
        approx_true_solution,
    ):
        self.stopping_crit = stopping_crit
        self.integrand = integrand
        self.true_measure = true_measure
        self.discrete_distrib = discrete_distrib
        self.d = self.discrete_distrib.d
        self.dnb2 = dnb2
        self.alpha = alpha
        self.refit = refit
        self.qmc_pts = self.dnb2(n_approx)
        self.gpytorch_prior_mean = gpytorch_prior_mean
        self.gpytorch_prior_cov = gpytorch_prior_cov
        self.gpytorch_likelihood = gpytorch_likelihood
        self.gpytorch_marginal_log_likelihood_func = (
            gpytorch_marginal_log_likelihood_func
        )
        self.torch_optimizer_func = torch_optimizer_func
        self.gpytorch_train_iter = gpytorch_train_iter
        self.gpytorch_use_gpu = gpytorch_use_gpu
        self.verbose = verbose
        self.approx_true_solution = approx_true_solution
        self.n_batch = []
        self.emr = []
        self.error_bounds = []
        self.ci_low = []
        self.ci_high = []
        self.solutions = []
        self.x = np.empty((0, self.d), dtype=float)
        self.y = np.array([], dtype=float)
        self.gpyt_model = ExactGPyTorchRegressionModel(
            x_t=self.x,
            y_t=self.y,
            prior_mean=self.gpytorch_prior_mean,
            prior_cov=self.gpytorch_prior_cov,
            likelihood=self.gpytorch_likelihood,
            use_gpu=self.gpytorch_use_gpu,
        )
        self.saved_gps = [self.gpyt_model.state_dict()]
        super(PFGPCIData, self).__init__(
            parameters=["solution", "error_bound", "bound_low", "bound_high", "n_total", "time_integrate"]
        )

    def update_data(self, batch_count, xdraw, ydrawtf):
        self.n_batch.append(len(xdraw))
        self.x, self.y = np.vstack([self.x, xdraw]), np.hstack([self.y, ydrawtf])
        if batch_count == 0 or self.refit:
            self.gpyt_model = ExactGPyTorchRegressionModel(
                x_t=self.x,
                y_t=self.y,
                prior_mean=self.gpytorch_prior_mean,
                prior_cov=self.gpytorch_prior_cov,
                likelihood=self.gpytorch_likelihood,
                use_gpu=self.gpytorch_use_gpu,
            )
            self.gpyt_model.fit(
                optimizer=self.torch_optimizer_func(self.gpyt_model),
                mll=self.gpytorch_marginal_log_likelihood_func(
                    self.gpyt_model.likelihood, self.gpyt_model
                ),
                training_iter=self.gpytorch_train_iter,
                verbose=self.verbose,
            )
        else:
            try:
                self.gpyt_model = self.gpyt_model.add_data(xdraw, ydrawtf)
                torch.cuda.empty_cache()
            except RuntimeError as e:
                # If adding data fails (e.g., NotPSDError), refit the entire model
                if self.verbose:
                    print(f"\tFalling back to full refit due to: {type(e).__name__}")
                self.gpyt_model = ExactGPyTorchRegressionModel(
                    x_t=self.x,
                    y_t=self.y,
                    prior_mean=self.gpytorch_prior_mean,
                    prior_cov=self.gpytorch_prior_cov,
                    likelihood=self.gpytorch_likelihood,
                    use_gpu=self.gpytorch_use_gpu,
                )
                self.gpyt_model.fit(
                    optimizer=self.torch_optimizer_func(self.gpyt_model),
                    mll=self.gpytorch_marginal_log_likelihood_func(
                        self.gpyt_model.likelihood, self.gpyt_model
                    ),
                    training_iter=self.gpytorch_train_iter,
                    verbose=self.verbose,
                )
                torch.cuda.empty_cache()
        self.saved_gps.append(self.gpyt_model.state_dict())
        phi = _get_phi(self.gpyt_model, self.qmc_pts)
        self.solutions.append(np.mean(phi >= 0.5))
        self.emr.append(np.mean(np.minimum(phi, 1 - phi)))
        gamma = self.emr[-1] / self.alpha
        self.ci_low.append(np.maximum(self.solutions[-1] - gamma, 0))
        self.ci_high.append(np.minimum(self.solutions[-1] + gamma, 1))
        self.error_bounds.append(
            np.maximum(
                self.solutions[-1] - self.ci_low[-1],
                self.ci_high[-1] - self.solutions[-1],
            )
        )

    def get_results_dict(self):
        df = {
            "iter": np.arange(len(self.n_sum)),
            "n_sum": self.n_sum,
            "n_batch": self.n_batch,
            "error_bounds": self.error_bounds,
            "ci_low": self.ci_low,
            "ci_high": np.array(self.ci_high),
            "solutions": np.array(self.solutions),
        }
        if self.approx_true_solution:
            df["solutions_ref"], df["error_ref"], df["in_ci"] = (
                self.solutions_ref,
                self.error_ref,
                self.in_ci,
            )
        return df

    def plot(self, trace_only=False, **kwargs):
        from matplotlib import pyplot

        if self.d == 1 and not trace_only and self.saved_gps != []:
            fig, gs = self.plot_1d(**kwargs)
        elif self.d == 2 and not trace_only and self.saved_gps != []:
            fig, gs = self.plot_2d(**kwargs)
        else:
            from matplotlib import gridspec

            fig = pyplot.figure(constrained_layout=False, figsize=(8, 4))
            gs = gridspec.GridSpec(1, 2, figure=fig)
        axtrace = fig.add_subplot(gs[-1, :])
        if self.approx_true_solution:
            axtrace.plot(self.n_sum, self.solutions_ref, color="c", label=r"$P(g)$")
        axtrace.plot(
            self.n_sum,
            self.solutions,
            "-o",
            color="k",
            label=r"$\hat{P}_n^\mathrm{QMC}$",
        )
        axtrace.fill_between(
            self.n_sum, self.ci_low, self.ci_high, color="k", alpha=0.15
        )
        axtrace.plot(
            self.n_sum,
            self.error_bounds,
            "-o",
            color="m",
            label=r"$\hat{\gamma}_n^{\text{QMC}}$",
        )
        if self.approx_true_solution:
            axtrace.plot(
                self.n_sum,
                self.error_ref,
                "-o",
                color="b",
                label=r"$\vert \hat{P}_n^{\text{QMC}} - P(g) \vert$",
            )
        axtrace.set_xlim(
            [self.n_sum[0] - self.n_batch[1] / 2, self.n_sum[-1] + self.n_batch[1] / 2]
        )
        # axtrace.set_xticks(self.n_sum)
        # axtrace.set_xticklabels(self.n_sum)
        axtrace.set_yscale("log", base=10)
        # axtrace.set_ylim([0,1]); axtrace.set_yticks([0,1])
        for spine in ["top", "right"]:
            axtrace.spines[spine].set_visible(False)
        axtrace.legend(
            loc="lower center",
            bbox_to_anchor=(0.05, -0.5, 0.9, 0.102),
            mode="expand",
            ncol=4,
            frameon=False,
        )
        return fig

    def plot_1d(self, meshticks=1025, ci_percentage=0.95, **kwargs):
        from matplotlib import pyplot, gridspec

        beta = norm.ppf(np.mean([ci_percentage, 1]))
        n_batches = len(self.n_batch)
        fig = pyplot.figure(constrained_layout=False, figsize=(5 * n_batches, 5 * 3))
        gs = gridspec.GridSpec(3, n_batches, figure=fig)
        xticks = np.linspace(0, 1, meshticks)[1:-1]
        if self.approx_true_solution:
            yticks = self.integrand.f(xticks[:, None]).squeeze()
            ytickstf = self.stopping_crit._affine_tf(yticks)
        for j in range(n_batches):
            i0, i1 = 0 if j == 0 else self.n_sum[j - 1], self.n_sum[j]
            gpyt_model = ExactGPyTorchRegressionModel(
                x_t=self.x[:i0],
                y_t=self.y[:i0],
                prior_mean=self.gpytorch_prior_mean,
                prior_cov=self.gpytorch_prior_cov,
                likelihood=self.gpytorch_likelihood,
                use_gpu=self.gpytorch_use_gpu,
            )
            gpyt_model.load_state_dict(self.saved_gps[j])
            ax0j = fig.add_subplot(gs[0, j])
            mr_udens = _error_udens(gpyt_model, xticks[:, None])
            ax0j.fill_between(xticks, mr_udens, color="k", alpha=0.5)
            ax0j.scatter(self.x[i0:i1].squeeze(), np.zeros(i1 - i0) + 0.015, color="r")
            ax0j.set_ylim([0, 1])
            ax0j.set_yticks([0, 1])
            gpyt_model = ExactGPyTorchRegressionModel(
                x_t=self.x[:i1],
                y_t=self.y[:i1],
                prior_mean=self.gpytorch_prior_mean,
                prior_cov=self.gpytorch_prior_cov,
                likelihood=self.gpytorch_likelihood,
                use_gpu=self.gpytorch_use_gpu,
            )
            gpyt_model.load_state_dict(self.saved_gps[j + 1])
            ax1j = fig.add_subplot(gs[1, j], sharey=None if j == 0 else ax1j)
            if self.approx_true_solution:
                ax1j.plot(xticks, ytickstf, color="c", label=r"f(x)")
            gp_mean, gp_std = gpyt_model.predict(xticks[:, None])
            ax1j.plot(xticks, gp_mean, color="k", label="gp")
            ax1j.fill_between(
                xticks,
                gp_mean - beta * gp_std,
                gp_mean + beta * gp_std,
                color="k",
                alpha=0.25,
            )
            ax1j.axhline(y=0, color="lightgreen")
            ax1j.scatter(self.x[:i0].squeeze(), self.y[:i0], color="b", label="old pts")
            ax1j.scatter(
                self.x[i0:i1].squeeze(), self.y[i0:i1], color="r", label="new pts"
            )
            # ax1j.set_xlabel(r'$u$')
            if j == 0:
                # ax1j.set_ylabel(r'$y$')
                ax0j.set_ylabel(r"$2\mathrm{ERR}_n(u)$")
                ax1j.set_ylabel("GP Regression")
        for ax in fig.axes:
            ax.set_xlim([0, 1])
            ax.set_xticks([0, 1])
            ax.xaxis.set_visible(False)
        return fig, gs

    def plot_2d(self, meshticks=257, clevels=32, **kwargs):
        from matplotlib import pyplot, gridspec, cm

        n_batches = len(self.n_batch)
        fig = pyplot.figure(constrained_layout=False, figsize=(5 * n_batches, 5 * 5))
        gs = gridspec.GridSpec(
            5 if self.approx_true_solution else 4, n_batches, figure=fig
        )
        x0mesh, x1mesh = np.meshgrid(
            np.linspace(0, 1, meshticks)[1:-1], np.linspace(0, 1, meshticks)[1:-1]
        )
        xquery = np.vstack([x0mesh.flatten(), x1mesh.flatten()]).T
        if self.approx_true_solution:
            yquery = self.integrand.f(xquery).squeeze()
            ymeshtf = self.stopping_crit._affine_tf(yquery).reshape(x0mesh.shape)
        for j in range(n_batches):
            row_idx = 0
            i0, i1 = 0 if j == 0 else self.n_sum[j - 1], self.n_sum[j]
            if self.approx_true_solution:
                ax0j = fig.add_subplot(gs[row_idx, j])
                ax0j.contourf(
                    x0mesh,
                    x1mesh,
                    ymeshtf,
                    cmap=cm.Greys,
                    vmin=ymeshtf.min(),
                    vmax=ymeshtf.max(),
                    levels=clevels,
                )
                ax0j.contour(x0mesh, x1mesh, ymeshtf, colors="lightgreen", levels=[0])
                # ax0j.set_title(r'$g(\boldsymbol{u})$')
                if j == 0:
                    ax0j.set_ylabel(r"$g(\boldsymbol{u})$")
                row_idx += 1
            gpyt_model = ExactGPyTorchRegressionModel(
                x_t=self.x[:i0],
                y_t=self.y[:i0],
                prior_mean=self.gpytorch_prior_mean,
                prior_cov=self.gpytorch_prior_cov,
                likelihood=self.gpytorch_likelihood,
                use_gpu=self.gpytorch_use_gpu,
            )
            gpyt_model.load_state_dict(self.saved_gps[j])
            ax1j = fig.add_subplot(gs[row_idx, j])
            udens_mr = _error_udens(gpyt_model, xquery).reshape(x0mesh.shape)
            ax1j.contourf(
                x0mesh, x1mesh, udens_mr, cmap=cm.Greys, levels=clevels, vmin=0, vmax=1
            )
            ax1j.scatter(self.x[i0:i1, 0], self.x[i0:i1, 1], color="r")
            # ax1j.set_title(r'$2\mathrm{ERR}_n(\boldsymbol{u})$')
            if j == 0:
                ax1j.set_ylabel(r"$2\mathrm{ERR}_n(\boldsymbol{u})$")
            row_idx += 1
            gpyt_model = ExactGPyTorchRegressionModel(
                x_t=self.x[:i0],
                y_t=self.y[:i0],
                prior_mean=self.gpytorch_prior_mean,
                prior_cov=self.gpytorch_prior_cov,
                likelihood=self.gpytorch_likelihood,
                use_gpu=self.gpytorch_use_gpu,
            )
            gpyt_model.load_state_dict(self.saved_gps[j + 1])
            gp_mean, gp_std = gpyt_model.predict(xquery)
            gp_mean_mesh, gp_std_mesh = gp_mean.reshape(x0mesh.shape), gp_std.reshape(
                x0mesh.shape
            )
            ax2j = fig.add_subplot(gs[row_idx, j])
            ax2j.contourf(
                x0mesh,
                x1mesh,
                gp_mean_mesh,
                cmap=cm.Greys,
                vmin=gp_mean.min(),
                vmax=gp_mean.max(),
                levels=clevels,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax2j.contour(
                    x0mesh, x1mesh, gp_mean_mesh, colors="lightgreen", levels=[0]
                )
            # ax2j.set_title(r'$M_n(\boldsymbol{u})$')
            if j == 0:
                ax2j.set_ylabel(r"$M_n(\boldsymbol{u})$")
            row_idx += 1
            ax3j = fig.add_subplot(gs[row_idx, j])
            ax3j.contourf(x0mesh, x1mesh, gp_std_mesh, cmap=cm.Greys, levels=clevels)
            # ax3j.set_title(r'$\sigma_n(\boldsymbol{u})$')
            if j == 0:
                ax3j.set_ylabel(r"$\sigma_n(\boldsymbol{u})$")
            for ax in [ax2j, ax3j]:
                ax.scatter(self.x[:i0, 0], self.x[:i0, 1], color="b")
                ax.scatter(self.x[i0:i1, 0], self.x[i0:i1, 1], color="r")
        for ax in fig.axes:
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            # ax.set_xlabel(r'$u_1$')
            ax.set_aspect(1)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_tick_params(
                top=False,
                bottom=False,
                left=False,
                right=False,
                labelleft=True,
                labelbottom=False,
            )
            # ax.set_xticks([0,1])
            # ax.set_yticks([0,1])
            # ax.set_ylabel(r'$u_2$')
        return fig, gs
