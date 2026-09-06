from .abstract_stopping_criterion import AbstractStoppingCriterion
from ..util.data import Data

from ..discrete_distribution import IIDStdUniform
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractIIDDiscreteDistribution,
)
from ..integrand import FinancialOption, Linear0, AbstractIntegrand
from ..util import MaxSamplesWarning, ParameterError
import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import norm
from time import time
import warnings


class CubMCG(AbstractStoppingCriterion):
    r"""IID Monte Carlo stopping criterion using Berry-Esseen inequalities in
    a two step method with guarantees for functions with bounded kurtosis.

    Examples:
        >>> ao = FinancialOption(IIDStdUniform(52,seed=7))
        >>> sc = CubMCG(ao,abs_tol=.05)
        >>> solution,data = sc.integrate()
        >>> data
        Data (Data)
            solution        1.779
            bound_low       1.729
            bound_high      1.829
            bound_diff      0.100
            n_total         112314
            time_integrate  ...
        CubMCG (AbstractStoppingCriterion)
            abs_tol         0.050
            rel_tol         0
            n_init          2^(10)
            n_limit         2^(30)
            inflate         1.200
            alpha           0.010
            kurtmax         1.478
        FinancialOption (AbstractIntegrand)
            option          ASIAN
            call_put        CALL
            volatility      2^(-1)
            start_price     30
            strike_price    35
            interest_rate   0
            t_final         1
            asian_mean      ARITHMETIC
        GeometricBrownianMotion (AbstractTrueMeasure)
            time_vec        [0.019 0.038 0.058 ... 0.962 0.981 1.   ]
            drift           0
            diffusion       2^(-2)
            mean_gbm        [30. 30. 30. ... 30. 30. 30.]
            covariance_gbm  [[  4.337   4.337   4.337 ...   4.337   4.337   4.337]
                             [  4.337   8.696   8.696 ...   8.696   8.696   8.696]
                             [  4.337   8.696  13.075 ...  13.075  13.075  13.075]
                             ...
                             [  4.337   8.696  13.075 ... 244.564 244.564 244.564]
                             [  4.337   8.696  13.075 ... 244.564 250.08  250.08 ]
                             [  4.337   8.696  13.075 ... 244.564 250.08  255.623]]
            decomp_type     PCA
        IIDStdUniform (AbstractIIDDiscreteDistribution)
            d               52
            replications    1
            entropy         7

        Control variates

        >>> iid = IIDStdUniform(52,seed=7)
        >>> ao = FinancialOption(iid,option="ASIAN")
        >>> eo = FinancialOption(iid,option="EUROPEAN")
        >>> lin0 = Linear0(iid)
        >>> sc = CubMCG(ao,abs_tol=.05,
        ...     control_variates = [eo,lin0],
        ...     control_variate_means = [eo.get_exact_value(),0])
        >>> solution,data = sc.integrate()
        >>> data
        Data (Data)
            solution        1.787
            bound_low       1.737
            bound_high      1.837
            bound_diff      0.100
            n_total         52147
            time_integrate  ...
        CubMCG (AbstractStoppingCriterion)
            abs_tol         0.050
            rel_tol         0
            n_init          2^(10)
            n_limit         2^(30)
            inflate         1.200
            alpha           0.010
            kurtmax         1.478
            cv              [FinancialOption (AbstractIntegrand)
                                 option          EUROPEAN
                                 call_put        CALL
                                 volatility      2^(-1)
                                 start_price     30
                                 strike_price    35
                                 interest_rate   0
                                 t_final         1               Linear0 (AbstractIntegrand)]
            cv_mu           [4.211 0.   ]
        FinancialOption (AbstractIntegrand)
            option          ASIAN
            call_put        CALL
            volatility      2^(-1)
            start_price     30
            strike_price    35
            interest_rate   0
            t_final         1
            asian_mean      ARITHMETIC
        GeometricBrownianMotion (AbstractTrueMeasure)
            time_vec        [0.019 0.038 0.058 ... 0.962 0.981 1.   ]
            drift           0
            diffusion       2^(-2)
            mean_gbm        [30. 30. 30. ... 30. 30. 30.]
            covariance_gbm  [[  4.337   4.337   4.337 ...   4.337   4.337   4.337]
                             [  4.337   8.696   8.696 ...   8.696   8.696   8.696]
                             [  4.337   8.696  13.075 ...  13.075  13.075  13.075]
                             ...
                             [  4.337   8.696  13.075 ... 244.564 244.564 244.564]
                             [  4.337   8.696  13.075 ... 244.564 250.08  250.08 ]
                             [  4.337   8.696  13.075 ... 244.564 250.08  255.623]]
            decomp_type     PCA
        IIDStdUniform (AbstractIIDDiscreteDistribution)
            d               52
            replications    1
            entropy         7

        Relative tolerance

        >>> ao = FinancialOption(IIDStdUniform(52,seed=7))
        >>> sc = CubMCG(ao,abs_tol=1e-3,rel_tol=5e-2)
        >>> solution,data = sc.integrate()
        >>> data
        Data (Data)
            solution        1.743
            bound_low       1.661
            bound_high      1.825
            bound_diff      0.164
            n_total         27503
            time_integrate  ...
        CubMCG (AbstractStoppingCriterion)
            abs_tol         0.001
            rel_tol         0.050
            n_init          2^(10)
            n_limit         2^(30)
            inflate         1.200
            alpha           0.010
            kurtmax         1.478
        FinancialOption (AbstractIntegrand)
            option          ASIAN
            call_put        CALL
            volatility      2^(-1)
            start_price     30
            strike_price    35
            interest_rate   0
            t_final         1
            asian_mean      ARITHMETIC
        GeometricBrownianMotion (AbstractTrueMeasure)
            time_vec        [0.019 0.038 0.058 ... 0.962 0.981 1.   ]
            drift           0
            diffusion       2^(-2)
            mean_gbm        [30. 30. 30. ... 30. 30. 30.]
            covariance_gbm  [[  4.337   4.337   4.337 ...   4.337   4.337   4.337]
                             [  4.337   8.696   8.696 ...   8.696   8.696   8.696]
                             [  4.337   8.696  13.075 ...  13.075  13.075  13.075]
                             ...
                             [  4.337   8.696  13.075 ... 244.564 244.564 244.564]
                             [  4.337   8.696  13.075 ... 244.564 250.08  250.08 ]
                             [  4.337   8.696  13.075 ... 244.564 250.08  255.623]]
            decomp_type     PCA
        IIDStdUniform (AbstractIIDDiscreteDistribution)
            d               52
            replications    1
            entropy         7

        Relative tolerance and control variates

        >>> iid = IIDStdUniform(52,seed=7)
        >>> ao = FinancialOption(iid,option="ASIAN")
        >>> eo = FinancialOption(iid,option="EUROPEAN")
        >>> lin0 = Linear0(iid)
        >>> sc = CubMCG(ao,abs_tol=1e-3,rel_tol=5e-2,
        ...     control_variates = [eo,lin0],
        ...     control_variate_means = [eo.get_exact_value(),0])
        >>> solution,data = sc.integrate()
        >>> data
        Data (Data)
            solution        1.776
            bound_low       1.692
            bound_high      1.859
            bound_diff      0.167
            n_total         12074
            time_integrate  ...
        CubMCG (AbstractStoppingCriterion)
            abs_tol         0.001
            rel_tol         0.050
            n_init          2^(10)
            n_limit         2^(30)
            inflate         1.200
            alpha           0.010
            kurtmax         1.478
            cv              [FinancialOption (AbstractIntegrand)
                                 option          EUROPEAN
                                 call_put        CALL
                                 volatility      2^(-1)
                                 start_price     30
                                 strike_price    35
                                 interest_rate   0
                                 t_final         1               Linear0 (AbstractIntegrand)]
            cv_mu           [4.211 0.   ]
        FinancialOption (AbstractIntegrand)
            option          ASIAN
            call_put        CALL
            volatility      2^(-1)
            start_price     30
            strike_price    35
            interest_rate   0
            t_final         1
            asian_mean      ARITHMETIC
        GeometricBrownianMotion (AbstractTrueMeasure)
            time_vec        [0.019 0.038 0.058 ... 0.962 0.981 1.   ]
            drift           0
            diffusion       2^(-2)
            mean_gbm        [30. 30. 30. ... 30. 30. 30.]
            covariance_gbm  [[  4.337   4.337   4.337 ...   4.337   4.337   4.337]
                             [  4.337   8.696   8.696 ...   8.696   8.696   8.696]
                             [  4.337   8.696  13.075 ...  13.075  13.075  13.075]
                             ...
                             [  4.337   8.696  13.075 ... 244.564 244.564 244.564]
                             [  4.337   8.696  13.075 ... 244.564 250.08  250.08 ]
                             [  4.337   8.696  13.075 ... 244.564 250.08  255.623]]
            decomp_type     PCA
        IIDStdUniform (AbstractIIDDiscreteDistribution)
            d               52
            replications    1
            entropy         7

    **References: **

    1.  Fred J. Hickernell, Lan Jiang, Yuewei Liu, and Art B. Owen,
        "Guaranteed conservative fixed width confidence intervals via Monte Carlo sampling,"
        Monte Carlo and Quasi-Monte Carlo Methods 2012 (J. Dick, F. Y. Kuo, G. W. Peters, and I. H. Sloan, eds.), pp. 105-128,
        Springer-Verlag, Berlin, 2014. DOI: 10.1007/978-3-642-41095-6_5

    2.  Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama,
        Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou,
        GAIL: Guaranteed Automatic Integration Library (Version 2.3) [MATLAB Software], 2019.
        [http://gailgithub.github.io/GAIL_Dev/](http://gailgithub.github.io/GAIL_Dev/).
        [https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/meanMC_g.m](https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/meanMC_g.m).
    """

    def __init__(
        self,
        integrand: AbstractIntegrand,
        abs_tol: np.ndarray = 1e-2,
        rel_tol: np.ndarray = 0.0,
        n_init: int = 1024,
        n_limit: int = 2**30,
        inflate: float = 1.2,
        alpha: np.ndarray = 0.01,
        control_variates: list = None,
        control_variate_means: np.ndarray = None,
    ) -> None:
        r"""
        Args:
            integrand (AbstractIntegrand): The integrand.
            abs_tol (np.ndarray): Absolute error tolerance.
            rel_tol (np.ndarray): Relative error tolerance.
            n_init (int): Initial number of samples.
            n_limit (int): Maximum number of samples.
            inflate (float): Inflation factor $\geq 1$ to multiply by the
                variance estimate to make it more conservative.
            alpha (np.ndarray): Uncertainty level in $(0,1)$.
            control_variates (list): Integrands to use as control variates,
                each with the same underlying discrete distribution instance.
            control_variate_means (np.ndarray): Means of each control variate.
        """
        if control_variates is None:
            control_variates = []
        if control_variate_means is None:
            control_variate_means = []
        self.parameters = ["abs_tol", "rel_tol", "n_init", "n_limit", "inflate", "alpha", "kurtmax"]
        # Set Attributes
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)
        self.n_init = float(n_init)
        self.n_limit = float(n_limit)
        self.alpha = float(alpha)
        self.inflate = float(inflate)
        self.alpha_sigma = (
            float(self.alpha) / 2.0
        )  # the uncertainty for variance estimation
        self.kurtmax = (n_init - 3) / (n_init - 1) + (self.alpha_sigma * n_init) / (
            1 - self.alpha_sigma
        ) * (1 - 1.0 / self.inflate**2) ** 2
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        super(CubMCG, self).__init__(
            allowed_distribs=[AbstractIIDDiscreteDistribution],
            allow_vectorized_integrals=False,
        )
        if not (self.integrand.d_indv == ()):
            raise AssertionError
        # control variates
        self._init_control_variates(control_variates, control_variate_means)
        if self.ncv > 0:
            if not (self.cv_mu.shape == (
                (self.ncv,) + self.integrand.d_indv
            )):
                raise AssertionError("Control variate means should have shape (len(control variates),d_indv).")
            self.parameters += ["cv", "cv_mu"]

    def _get_main_stage_samples(self, data):
        n_init = int(self.n_init)
        y = np.array(data.yfull[n_init:], copy=False)
        if self.ncv == 0:
            return y
        ycv = np.array(data.ycvfull[:, n_init:], copy=False)
        return y - ((ycv - self.cv_mu[:, None]) * self.beta[:, None]).sum(0)

    def _update_main_stage_solution(self, data):
        y_main = self._get_main_stage_samples(data)
        data.solution = y_main.mean()
        data.n_total = data.yfull.shape[-1]

    def integrate(self, resume=None):
        t_start = time()
        trace = self._make_trace_logger()
        if resume is not None:
            raise ParameterError("CubMCG does not support resume.")
        data = Data(parameters=["solution", "bound_low", "bound_high", "bound_diff", "n_total", "time_integrate"])
        data.xfull = np.empty((0, self.integrand.d))
        data.yfull = np.empty(0)
        if self.ncv > 0:
            data.ycvfull = np.empty((self.ncv, 0))
        x0 = self.discrete_distrib(n=self.n_init)
        data.xfull = np.concatenate([data.xfull, x0], 0)
        y0 = self.integrand.f(x0)
        data.yfull = np.concatenate([data.yfull, y0], -1)
        if self.ncv > 0:
            ycv0 = [None] * self.ncv
            for k in range(self.ncv):
                ycv0[k] = self.cv[k].f(x0)
            ycv0 = np.stack(ycv0, 0)
            data.ycvfull = np.concatenate([data.ycvfull, ycv0], 1)
            cvmuhats = ycv0.mean(-1)
            x4beta = ycv0 - cvmuhats[:, None]
            y4beta = y0 - y0.mean()
            self.beta = np.linalg.lstsq(x4beta.T, y4beta, rcond=None)[0]
            y0 = y0 - ((ycv0 - self.cv_mu[:, None]) * self.beta[:, None]).sum(0)
        data.sighat0 = y0.std(ddof=1)
        data.solution0 = y0.mean()
        sigma_up = self.inflate * data.sighat0
        if self.rel_tol == 0:
            self.alpha_mu = 1 - (1 - self.alpha) / (1 - self.alpha_sigma)
            toloversig = self.abs_tol / sigma_up
            n_mu, data.bound_half_width = self._nchebe(toloversig, self.alpha_mu, self.kurtmax, self.n_limit, sigma_up)
            data.n_mu = int(n_mu)
            if (self.n_init + data.n_mu) > self.n_limit:
                # cannot generate this many new samples
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples would exceed n_limit = %d.
                Will instead generate %d new samples to reach n_limit.""" % (int(self.n_init), int(data.n_mu), int(self.n_limit), int(self.n_limit - self.n_init))
                warnings.warn(warning_s, MaxSamplesWarning)
                data.n_mu = self.n_limit - self.n_init
            x = self.discrete_distrib(n=data.n_mu)
            data.xfull = np.concatenate([data.xfull, x], 0)
            y = self.integrand.f(x)
            data.yfull = np.concatenate([data.yfull, y], -1)
            if self.ncv > 0:
                ycv = [None] * self.ncv
                for k in range(self.ncv):
                    ycv[k] = self.cv[k].f(x)
                ycv = np.stack(ycv, 0)
                data.ycvfull = np.concatenate([data.ycvfull, ycv], 1)
                y = y - ((ycv - self.cv_mu[:, None]) * self.beta[:, None]).sum(0)
            data.solution = y.mean()
            data.n_total = self.n_init + data.n_mu
            self._set_elapsed_time(data, time() - t_start)
            trace.iteration(data)
        else:  # self.rel_tol > 0
            alphai = (self.alpha - self.alpha_sigma) / (2 * (1 - self.alpha_sigma))  # uncertainty to do iteration
            eps1 = self._ncbinv(self.n_init, alphai, self.kurtmax)
            data.bound_half_width = sigma_up * eps1
            data.tau = 1.0  # step of the iteration
            n = self.n_init  # default initial sample size
            data.n_total = self.n_init
            while True:
                if (data.n_total + n) > self.n_limit:
                    # cannot generate this many new samples
                    warning_s = """
                    Already generated %d samples.
                    Trying to generate %d new samples would exceeds n_limit = %d.
                    Will instead generate %d samples to meet n_limit total samples. 
                    Note that error tolerances may no longer be satisfied""" % (int(data.n_total), int(n), int(self.n_limit), int(self.n_limit - data.n_total))
                    warnings.warn(warning_s, MaxSamplesWarning)
                    n = self.n_limit - data.n_total
                x = self.discrete_distrib(n=n)
                data.xfull = np.concatenate([data.xfull, x], 0)
                y = self.integrand.f(x)
                data.yfull = np.concatenate([data.yfull, y], -1)
                if self.ncv > 0:
                    ycv = [None] * self.ncv
                    for k in range(self.ncv):
                        ycv[k] = self.cv[k].f(x)
                    ycv = np.stack(ycv, 0)
                    data.ycvfull = np.concatenate([data.ycvfull, ycv], 1)
                    y = y - ((ycv - self.cv_mu[:, None]) * self.beta[:, None]).sum(0)
                data.solution = y.mean()
                data.n_total += n
                if data.n_total >= self.n_limit:
                    self._set_elapsed_time(data, time() - t_start)
                    trace.iteration(data)
                    break
                lb_tol = _tol_fun(
                    self.abs_tol,
                    self.rel_tol,
                    0.0,
                    data.solution - data.bound_half_width,
                    "max",
                )
                ub_tol = _tol_fun(
                    self.abs_tol,
                    self.rel_tol,
                    0.0,
                    data.solution + data.bound_half_width,
                    "max",
                )
                delta_plus = (lb_tol + ub_tol) / 2.0
                if delta_plus >= data.bound_half_width:
                    # stopping criterion met
                    delta_minus = (lb_tol - ub_tol) / 2.0
                    data.solution += delta_minus  # adjust solution a bit
                    self._set_elapsed_time(data, time() - t_start)
                    trace.iteration(data)
                    break
                else:
                    self._set_elapsed_time(data, time() - t_start)
                    trace.iteration(data)
                    candidate_tol = np.maximum(
                        self.abs_tol, 0.95 * self.rel_tol * abs(data.solution)
                    )
                    data.bound_half_width = np.minimum(
                        data.bound_half_width / 2.0, candidate_tol
                    )
                    data.tau += 1
                # update next uncertainty
                toloversig = data.bound_half_width / sigma_up
                alphai = (
                    2**data.tau
                    * (self.alpha - self.alpha_sigma)
                    / (1 - self.alpha_sigma)
                )
                n = int(self._nchebe(toloversig, alphai, self.kurtmax, self.n_limit, sigma_up)[0])
        data.bound_low = data.solution - data.bound_half_width
        data.bound_high = data.solution + data.bound_half_width
        data.bound_diff = data.bound_high - data.bound_low
        self._finalize_integration_data(data, time() - t_start)
        trace.finalize()
        return data.solution, data

    def _nchebe(self, toloversig, alpha, kurtmax, n_budget, sigma_0_up):
        _b = -norm.ppf(np.finfo(float).eps)
        ncheb = np.ceil(
            1 / (alpha * toloversig**2)
        )  # sample size by Chebyshev's Inequality
        A = 18.1139
        A1 = 0.3328
        A2 = 0.429  # three constants in Berry-Esseen inequality
        M3upper = kurtmax ** (3.0 / 4)
        # the upper bound on the third moment by Jensen's inequality
        BEfun2 = lambda logsqrtn: (
            norm.cdf(-np.exp(logsqrtn) * toloversig)
            + np.exp(-logsqrtn)
            * np.minimum(
                A1 * (M3upper + A2),
                A * M3upper / (1 + (np.exp(logsqrtn) * toloversig) ** 3),
            )
            - alpha / 2.0
        )
        # Berry-Esseen function, whose solution is the sample size needed
        logsqrtnCLT = np.log(norm.ppf(1 - alpha / 2) / toloversig)
        # sample size by CLT
        rsdata = root_scalar(BEfun2, x0=logsqrtnCLT, method="toms748", bracket=(-_b, _b))
        nbe = np.ceil(np.exp(2 * rsdata.root))
        # calculate Berry-Esseen n by fsolve function (scipy)
        ncb = np.minimum(
            np.minimum(ncheb, nbe), n_budget
        )  # take the min of two sample sizes
        logsqrtn = np.log(np.sqrt(ncb))
        BEfun3 = lambda toloversig: (
            norm.cdf(-np.exp(logsqrtn) * toloversig)
            + np.exp(-logsqrtn)
            * np.minimum(
                A1 * (M3upper + A2),
                A * M3upper / (1 + (np.exp(logsqrtn) * toloversig) ** 3),
            )
            - alpha / 2.0
        )
        rsdata = root_scalar(BEfun3, x0=toloversig, method="toms748", bracket=(-_b, _b))
        err = rsdata.root * sigma_0_up
        return ncb, err

    def _ncbinv(self, n1, alpha1, kurtmax):
        _b = -norm.ppf(np.finfo(float).eps)
        NCheb_inv = 1 / np.sqrt(n1 * alpha1)
        # use Chebyshev inequality
        A = 18.1139
        A1 = 0.3328
        A2 = 0.429  # three constants in Berry-Esseen inequality
        M3upper = kurtmax ** (3.0 / 4)
        # using Jensen's inequality to bound the third moment
        BEfun = lambda logsqrtb: (
            norm.cdf(n1 * logsqrtb)
            + np.minimum(
                A1 * (M3upper + A2), A * M3upper / (1 + (np.sqrt(n1) * logsqrtb) ** 3)
            )
            / np.sqrt(n1)
            - alpha1 / 2
        )
        # Berry-Esseen inequality
        logsqrtb_clt = np.log(np.sqrt(norm.ppf(1 - alpha1 / 2) / np.sqrt(n1)))
        # use CLT to get tolerance
        rsdata = root_scalar(BEfun, x0=logsqrtb_clt, method="toms748", bracket=(-_b, _b))
        NBE_inv = np.exp(2 * rsdata.root)
        # use fsolve to get Berry-Esseen tolerance
        eps = np.minimum(NCheb_inv, NBE_inv)
        # take the min of Chebyshev and Berry Esseen tolerance
        return eps

    def set_tolerance(self, abs_tol=None, rel_tol=None, rmse_tol=None):
        if not (rmse_tol is None):
            raise AssertionError("rmse_tol not supported by this stopping criterion.")
        if abs_tol != None:
            self.abs_tol = abs_tol
        if rel_tol != None:
            self.rel_tol = rel_tol


def _tol_fun(abs_tol, rel_tol, theta, mu, toltype):
    # """
    # Generalized error tolerance function.

    # Args:
    #     abs_tol (float): absolute error tolerance
    #     rel_tol (float): relative error tolerance
    #     theta (float): parameter in 'theta' case
    #     mu (float): true mean
    #     toltype (str): different options of tolerance function

    # Returns:
    #     float: tolerance as weighted sum of absolute and relative tolerance
    # """
    if toltype == "combine":  # the linear combination of two tolerances
        # theta == 0 --> relative error tolerance
        # theta == 1 --> absolute error tolerance
        return theta * abs_tol + (1 - theta) * rel_tol * abs(mu)
    elif toltype == "max":  # the max case
        return max(abs_tol, rel_tol * abs(mu))
