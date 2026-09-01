from .abstract_integrand import AbstractIntegrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import GeometricBrownianMotion
from ..util import ParameterError
import numpy as np
from scipy.stats import norm


class FinancialOption(AbstractIntegrand):
    r"""
    Financial options.

    - Start price $S_0$
    - Strike price $K$
    - Interest rate $r$
    - Volatility $\sigma$
    - Drift $\gamma$
    - Equidistant monitoring times $\boldsymbol{\tau} = (\tau_1,\dots,\tau_d)^T$ with $\tau_d$ the final (exercise) time and $\tau_j = \tau_d j/d$.

    Define the [geometric brownian motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion) as

    $$\boldsymbol{S}(\boldsymbol{t}) = S_0 e^{(\gamma-\sigma^2/2)\boldsymbol{\tau}+\sigma\boldsymbol{t}}, \qquad \boldsymbol{T} \sim \mathcal{N}(\boldsymbol{0},\mathsf{\Sigma})$$

    where $\boldsymbol{T}$ is a standard Brownian motion so $\mathsf{\Sigma} = \left(\min\{\tau_j,\tau_{j'}\}\right)_{j,j'=1}^d$.

    The discounted payoff is

    $$g(\boldsymbol{t}) = P(\boldsymbol{S}(\boldsymbol{t}))e^{-r \tau_d}$$

    where the payoff function $P$ will be defined depending on the option.

    Below we will use $S_{-1}$ to denote the final element of $\boldsymbol{S}$, the value of the path at exercise time.

    # European Options

    *European Call and Put Options* have respective payoffs

    $$P(\boldsymbol{S}) = \max\{S_{-1}-K,0\}, \qquad P(\boldsymbol{S}) = \max\{K-S_{-1},0\}.$$

    # Asian Options

    An asian option considers the average value of an asset path across time. We use the trapezoidal rule to approximate either the *arithmetic mean* by

    $$A(\boldsymbol{S}) = \frac{1}{d}\left[\frac{1}{2} S_0 + \sum_{j=1}^{d-1} S_j + \frac{1}{2} S_{-1}\right]$$

    or the *geometric mean* by

    $$A(\boldsymbol{S}) = \left[\sqrt{S_0} \prod_{j=1}^{d-1} S_j \sqrt{S_{-1}}\right]^{1/d}.$$

    *Asian Call and Put Option* have respective payoffs

    $$P(\boldsymbol{S}) = \max\{A(\boldsymbol{S})-K,0\}, \qquad P(\boldsymbol{S}) = \max\{K-A(\boldsymbol{S}),0\}.$$

    # Barrier Options

    - Barrier $B$.

    *In* options are activate when the path crosses the barrier $B$, while *out* options are activated only if the path never crosses the barrier $B$.
    An *up* option satisfies $S_0<B$ while a *down* option satisfies $S_0>B$, both indicating the direction of the barrier from the start price.

    *Barrier Up-In Call and Put Options* have respective payoffs

    $$P(\boldsymbol{S}) = \begin{cases} \max\{S_{-1})-K,0\}, & \text{any } \boldsymbol{S} \geq B \\ 0, & \mathrm{otherwise} \end{cases}, \qquad P(\boldsymbol{S}) = \begin{cases} \max\{K-S_{-1}),0\}, & \text{any } \boldsymbol{S} \geq B \\ 0, & \mathrm{otherwise} \end{cases}.$$

    *Barrier Up-Out Call and Put Options* have respective payoffs

    $$P(\boldsymbol{S}) = \begin{cases} \max\{S_{-1})-K,0\}, & \text{all } \boldsymbol{S} < B \\ 0, & \mathrm{otherwise} \end{cases}, \qquad P(\boldsymbol{S}) = \begin{cases} \max\{K-S_{-1}),0\}, & \text{all } \boldsymbol{S} < B \\ 0, & \mathrm{otherwise} \end{cases}.$$

    *Barrier Down-In Call and Put Options* have respective payoffs

    $$P(\boldsymbol{S}) = \begin{cases} \max\{S_{-1})-K,0\}, & \text{any } \boldsymbol{S} \leq B \\ 0, & \mathrm{otherwise} \end{cases}, \qquad P(\boldsymbol{S}) = \begin{cases} \max\{K-S_{-1}),0\}, & \text{any } \boldsymbol{S} \leq B \\ 0, & \mathrm{otherwise} \end{cases}.$$

    *Barrier Down-Out Call and Put Options* have respective payoffs

    $$P(\boldsymbol{S}) = \begin{cases} \max\{S_{-1})-K,0\}, & \text{all } \boldsymbol{S} > B \\ 0, & \mathrm{otherwise} \end{cases}, \qquad P(\boldsymbol{S}) = \begin{cases} \max\{K-S_{-1}),0\}, & \text{all } \boldsymbol{S} > B \\ 0, & \mathrm{otherwise} \end{cases}.$$

    # Lookback Options

    *Lookback Call and Put Options* have respective payoffs

    $$P(\boldsymbol{S}) = S_{-1}-\min(S_0, \ldots S_{-1}), \qquad P(\boldsymbol{S}) = \max(S_0, \ldots S_{-1})-S_{-1}.$$

    # Digital Option

    - Payout $\rho$.

    *Digital Call and Put Options* have respective payoffs

    $$P(\boldsymbol{S}) = \begin{cases} \rho, & S_{-1} \geq K \\ 0, & \mathrm{otherwise} \end{cases}, \qquad P(\boldsymbol{S}) =  \begin{cases} \rho, & S_{-1} \leq K \\ 0, & \mathrm{otherwise} \end{cases}.$$

    # Multilevel Options

    - Initial level $\ell_0 \geq 0$.
    - Level $\ell \geq \ell_0$.

    Let $\boldsymbol{S}_\mathrm{fine}=\boldsymbol{S}$ be the *fine* full path. For $\ell>\ell_0$ write the *coarse* path as $\boldsymbol{S}_\mathrm{coarse} = (S_j)_{j \text{ even}}$ which only considers every other element of $\boldsymbol{S}$.
    In this multilevel setting the payoff is

    $$P_\ell(\boldsymbol{S}) = \begin{cases} P(\boldsymbol{S}_\mathrm{fine}), & \ell = \ell_0, \\ P(\boldsymbol{S}_\mathrm{fine})-P(\boldsymbol{S}_\mathrm{coarse}), & \ell > \ell_0 \end{cases}.$$

    Cancellations from the telescoping sum allow us to write

    $$\lim_{\ell \to \infty} P_\ell = P_{\ell_0} + \sum_{\ell=\ell_0+1}^\infty P_\ell.$$

    Examples:
        >>> integrand = FinancialOption(DigitalNetB2(dimension=3,seed=7),option="EUROPEAN")
        >>> y = integrand(2**10)
        >>> y.shape
        (1024,)
        >>> print("%.4f"%y.mean())
        4.2126
        >>> print("%.4f"%integrand.get_exact_value())
        4.2115
        >>> integrand
        FinancialOption (AbstractIntegrand)
            option          EUROPEAN
            call_put        CALL
            volatility      2^(-1)
            start_price     30
            strike_price    35
            interest_rate   0
            t_final         1
        >>> integrand.true_measure
        GeometricBrownianMotion (AbstractTrueMeasure)
            time_vec        [0.333 0.667 1.   ]
            drift           0
            diffusion       2^(-2)
            mean_gbm        [30. 30. 30.]
            covariance_gbm  [[ 78.214  78.214  78.214]
                             [ 78.214 163.224 163.224]
                             [ 78.214 163.224 255.623]]
            decomp_type     PCA

        >>> integrand = FinancialOption(DigitalNetB2(dimension=64,seed=7),option="ASIAN")
        >>> y = integrand(2**10)
        >>> y.shape
        (1024,)
        >>> print("%.4f"%y.mean())
        1.7782

        With independent replications

        >>> integrand = FinancialOption(DigitalNetB2(dimension=64,seed=7,replications=2**4),option="ASIAN")
        >>> y = integrand(2**6)
        >>> y.shape
        (16, 64)
        >>> muhats = y.mean(-1)
        >>> muhats.shape
        (16,)
        >>> print("%.4f"%muhats.mean())
        1.7765

        Multi-level options

        >>> seed_seq = np.random.SeedSequence(7)
        >>> d_coarsest = 8
        >>> num_levels = 4
        >>> ns = [2**11,2**10,2**9,2**8]
        >>> integrands = [FinancialOption(DigitalNetB2(dimension=2**l*d_coarsest,seed=seed_seq.spawn(1)[0]),option="ASIAN",level=l,d_coarsest=d_coarsest) for l in range(num_levels)]
        >>> ys = [integrands[l](ns[l]) for l in range(num_levels)]
        >>> for l in range(num_levels):
        ...     print("ys[%d].shape = %s"%(l,ys[l].shape))
        ys[0].shape = (2, 2048)
        ys[1].shape = (2, 1024)
        ys[2].shape = (2, 512)
        ys[3].shape = (2, 256)
        >>> ymeans = np.stack([(ys[l][1]-ys[l][0]).mean(-1) for l in range(num_levels)])
        >>> with np.printoptions(formatter={"float": lambda x: "%.2e"%x}):
        ...     ymeans
        array([1.78e+00, 2.61e-03, 5.57e-04, 1.10e-03])
        >>> print("%.4f"%ymeans.sum())
        1.7850

        Multi-level options with independent replications

        >>> seed_seq = np.random.SeedSequence(7)
        >>> d_coarsest = 8
        >>> num_levels = 4
        >>> ns = [2**7,2**6,2**5,2**4]
        >>> integrands = [FinancialOption(DigitalNetB2(dimension=2**l*d_coarsest,seed=seed_seq.spawn(1)[0],replications=2**4),option="ASIAN",level=l,d_coarsest=d_coarsest) for l in range(num_levels)]
        >>> ys = [integrands[l](ns[l]) for l in range(num_levels)]
        >>> for l in range(num_levels):
        ...     print("ys[%d].shape = %s"%(l,ys[l].shape))
        ys[0].shape = (2, 16, 128)
        ys[1].shape = (2, 16, 64)
        ys[2].shape = (2, 16, 32)
        ys[3].shape = (2, 16, 16)
        >>> muhats = np.stack([(ys[l][1]-ys[l][0]).mean(-1) for l in range(num_levels)])
        >>> muhats.shape
        (4, 16)
        >>> muhathat = muhats.mean(-1)
        >>> with np.printoptions(formatter={"float": lambda x: "%.2e"%x}):
        ...     muhathat
        array([1.80e+00, -3.08e-03, 2.64e-03, 1.14e-03])
        >>> print("%.4f"%muhathat.sum())
        1.7982

    **References:**

    1.  M.B. Giles.
        Improved multilevel Monte Carlo convergence using the Milstein scheme.
        343-358, in Monte Carlo and Quasi-Monte Carlo Methods 2006, Springer, 2008.
        [http://people.maths.ox.ac.uk/~gilesm/files/mcqmc06.pdf](http://people.maths.ox.ac.uk/~gilesm/files/mcqmc06.pdf).
    """

    def __init__(
        self,
        sampler,
        option="ASIAN",
        call_put="CALL",
        volatility=0.5,
        start_price=30,
        strike_price=35,
        interest_rate=0,
        t_final=1,
        decomp_type="PCA",
        level=None,
        d_coarsest=2,
        asian_mean="ARITHMETIC",
        asian_mean_quadrature_rule="TRAPEZOIDAL",
        barrier_in_out="IN",
        barrier_price=38,
        digital_payout=10,
    ):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]): Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            option (str): Option type in `['ASIAN', 'EUROPEAN', 'BARRIER', 'LOOKBACK', 'DIGITAL']`
            call_put (str): Either `'CALL'` or `'PUT'`.
            volatility (float): $\sigma$.
            start_price (float): $S_0$.
            strike_price (float): $K$.
            interest_rate (float): $r$.
            t_final (float): $\tau_d$.
            decomp_type (str): Method for decomposition for covariance matrix. Options include

                - `'PCA'` for principal component analysis,
                - `'Cholesky'` for cholesky decomposition, or
                - `'BrownianBridge'` or `'Bridge'` for brownian bridge construction.
            level (Union[None, int]): Level for multilevel problems
            d_coarsest (Union[None, int]): Dimension of the problem on the coarsest level.
            asian_mean (str): Either `'ARITHMETIC'` or `'GEOMETRIC'`.
            asian_mean_quadrature_rule (str): Either 'TRAPEZOIDAL' or 'RIGHT'.
            barrier_in_out (str): Either `'IN'` or `'OUT'`.
            barrier_price (float): $B$.
            digital_payout (float): $\rho$.
        """
        self.parameters = [
            "option",
            "call_put",
            "volatility",
            "start_price",
            "strike_price",
            "interest_rate",
            "t_final",
        ]
        self.t_final = t_final
        self.sampler = sampler
        self.decomp_type = decomp_type
        self.volatility = float(volatility)
        self.start_price = float(start_price)
        self.strike_price = float(strike_price)
        self.interest_rate = float(interest_rate)
        self.true_measure = GeometricBrownianMotion(
            self.sampler,
            t_final=self.t_final,
            initial_value=self.start_price,
            drift=self.interest_rate,
            diffusion=self.volatility**2,
            decomp_type=self.decomp_type,
        )
        self.discount_factor = np.exp(-self.interest_rate * self.t_final)
        self.level = level
        self.d_coarsest = d_coarsest
        if self.level is not None:
            self.multilevel = True
            self.parameters += ["level", "d_coarsest"]
            assert np.isscalar(self.level) and self.level % 1 == 0
            assert (
                np.isscalar(self.d_coarsest)
                and self.d_coarsest % 1 == 0
                and d_coarsest > 0
                and np.log2(d_coarsest) % 1 == 0
            ), "d_coarsest must be an integer power of 2"
            self.level = int(self.level)
            self.d_coarsest = int(self.d_coarsest)
            assert (
                self.sampler.d == self.d_coarsest * 2**self.level
            ), "the dimension of the sampler must equal d_coarsest*2^level = %d" % (
                d_coarsest * 2**self.level
            )
            self.cost = self.d_coarsest * 2**self.level
            dim_shape = (2,)
        else:
            self.multilevel = False
            dim_shape = ()
        self.call_put = str(call_put).upper()
        assert self.call_put in ["CALL", "PUT"], "invalid call_put = %s" % self.call_put
        self.option = str(option).upper()
        self.asian_mean = str(asian_mean).upper()
        self.asian_mean_quadrature_rule = str(asian_mean_quadrature_rule).upper()
        self.barrier_in_out = str(barrier_in_out).upper()
        assert np.isscalar(barrier_price)
        self.barrier_price = float(barrier_price)
        assert np.isscalar(digital_payout) and digital_payout > 0
        self.digital_payout = float(digital_payout)
        if self.option == "EUROPEAN":
            self.payoff = (
                self.payoff_european_call
                if self.call_put == "CALL"
                else self.payoff_european_put
            )
        elif self.option == "ASIAN":
            self.parameters += ["asian_mean"]
            assert self.asian_mean in ["ARITHMETIC", "GEOMETRIC"], (
                "invalid asian_mean = %s" % self.asian_mean
            )
            assert self.asian_mean_quadrature_rule in ["TRAPEZOIDAL", "RIGHT"], (
                "invalid asian_mean_quadrature_rule = %s"
                % self.asian_mean_quadrature_rule
            )
            if self.asian_mean == "ARITHMETIC":
                if self.asian_mean_quadrature_rule == "TRAPEZOIDAL":
                    self.payoff = (
                        self.payoff_asian_arithmetic_trap_call
                        if self.call_put == "CALL"
                        else self.payoff_asian_arithmetic_trap_put
                    )
                elif self.asian_mean_quadrature_rule == "RIGHT":
                    self.payoff = (
                        self.payoff_asian_arithmetic_right_call
                        if self.call_put == "CALL"
                        else self.payoff_asian_arithmetic_right_put
                    )
            elif self.asian_mean == "GEOMETRIC":
                if self.asian_mean_quadrature_rule == "TRAPEZOIDAL":
                    self.payoff = (
                        self.payoff_asian_geometric_trap_call
                        if self.call_put == "CALL"
                        else self.payoff_asian_geometric_trap_put
                    )
                elif self.asian_mean_quadrature_rule == "RIGHT":
                    self.payoff = (
                        self.payoff_asian_geometric_right_call
                        if self.call_put == "CALL"
                        else self.payoff_asian_geometric_right_put
                    )
        elif self.option == "BARRIER":
            self.parameters += ["barrier_in_out", "barrier_up_down", "barrier_price"]
            self.barrier_up_down = (
                "UP" if self.start_price < self.barrier_price else "DOWN"
            )
            if self.barrier_in_out == "IN":
                if self.barrier_up_down == "UP":
                    self.payoff = (
                        self.payoff_barrier_in_up_call
                        if self.call_put == "CALL"
                        else self.payoff_barrier_in_up_put
                    )
                else:
                    self.payoff = (
                        self.payoff_barrier_in_down_call
                        if self.call_put == "CALL"
                        else self.payoff_barrier_in_down_put
                    )
            elif self.barrier_in_out == "OUT":
                if self.barrier_up_down == "UP":
                    self.payoff = (
                        self.payoff_barrier_out_up_call
                        if self.call_put == "CALL"
                        else self.payoff_barrier_out_up_put
                    )
                else:
                    self.payoff = (
                        self.payoff_barrier_out_down_call
                        if self.call_put == "CALL"
                        else self.payoff_barrier_out_down_put
                    )
            else:
                raise ParameterError(
                    "invalid barrier_in_out = %s" % self.barrier_in_out
                )
        elif self.option == "LOOKBACK":
            self.payoff = (
                self.payoff_lookback_call
                if self.call_put == "CALL"
                else self.payoff_lookback_put
            )
        elif self.option == "DIGITAL":
            self.payoff = (
                self.payoff_digital_call
                if self.call_put == "CALL"
                else self.payoff_digital_put
            )
        else:
            raise ParameterError("invalid option type %s" % self.option)
        super(FinancialOption, self).__init__(
            dimension_indv=dim_shape, dimension_comb=dim_shape, parallel=False
        )

    def g(self, t, **kwargs):
        gbm = t  # GeometricBrownianMotion already provides GBM paths directly
        discounted_payoffs = self.payoff(gbm) * self.discount_factor
        if self.multilevel:
            if self.level == 0:
                discounted_payoffs_coarse = np.zeros_like(discounted_payoffs)
            else:
                discounted_payoffs_coarse = (
                    self.payoff(gbm[..., 1::2]) * self.discount_factor
                )
            discounted_payoffs = np.stack(
                [discounted_payoffs_coarse, discounted_payoffs]
            )
        return discounted_payoffs

    def payoff_european_call(self, gbm):
        return np.maximum(gbm[..., -1] - self.strike_price, 0)

    def payoff_european_put(self, gbm):
        return np.maximum(self.strike_price - gbm[..., -1], 0)

    def payoff_asian_arithmetic_trap_call(self, gbm):
        return np.maximum(
            (self.start_price / 2 + gbm[..., :-1].sum(-1) + gbm[..., -1] / 2)
            / gbm.shape[-1]
            - self.strike_price,
            0,
        )

    def payoff_asian_arithmetic_trap_put(self, gbm):
        return np.maximum(
            self.strike_price
            - (self.start_price / 2 + gbm[..., :-1].sum(-1) + gbm[..., -1] / 2)
            / gbm.shape[-1],
            0,
        )

    def payoff_asian_geometric_trap_call(self, gbm):
        return np.maximum(
            np.exp(
                (
                    np.log(self.start_price) / 2
                    + np.log(gbm[..., :-1]).sum(-1)
                    + np.log(gbm[..., -1]) / 2
                )
                / gbm.shape[-1]
            )
            - self.strike_price,
            0,
        )

    def payoff_asian_geometric_trap_put(self, gbm):
        return np.maximum(
            self.strike_price
            - np.exp(
                (
                    np.log(self.start_price) / 2
                    + np.log(gbm[..., :-1]).sum(-1)
                    + np.log(gbm[..., -1]) / 2
                )
                / gbm.shape[-1]
            ),
            0,
        )

    def payoff_asian_arithmetic_right_call(self, gbm):
        return np.maximum(gbm.sum(-1) / gbm.shape[-1] - self.strike_price, 0)

    def payoff_asian_arithmetic_right_put(self, gbm):
        return np.maximum((self.strike_price - gbm.sum(-1)) / gbm.shape[-1], 0)

    def payoff_asian_geometric_right_call(self, gbm):
        return np.maximum(
            np.exp(np.log(gbm).sum(-1) / gbm.shape[-1]) - self.strike_price, 0
        )

    def payoff_asian_geometric_right_put(self, gbm):
        return np.maximum(
            self.strike_price - np.exp(np.log(gbm).sum(-1) / gbm.shape[-1]), 0
        )

    def payoff_barrier_in_up_call(self, gbm):
        v = gbm[..., -1].copy()
        flag = (gbm >= self.barrier_price).any(-1)
        v[~flag] = 0
        v[flag] = np.maximum(v[flag] - self.strike_price, 0)
        return v

    def payoff_barrier_out_up_call(self, gbm):
        v = gbm[..., -1].copy()
        flag = (gbm < self.barrier_price).all(-1)
        v[~flag] = 0
        v[flag] = np.maximum(v[flag] - self.strike_price, 0)
        return v

    def payoff_barrier_in_down_call(self, gbm):
        v = gbm[..., -1].copy()
        flag = (gbm <= self.barrier_price).any(-1)
        v[~flag] = 0
        v[flag] = np.maximum(v[flag] - self.strike_price, 0)
        return v

    def payoff_barrier_out_down_call(self, gbm):
        v = gbm[..., -1].copy()
        flag = (gbm > self.barrier_price).all(-1)
        v[~flag] = 0
        v[flag] = np.maximum(v[flag] - self.strike_price, 0)
        return v

    def payoff_barrier_in_up_put(self, gbm):
        v = gbm[..., -1].copy()
        flag = (gbm >= self.barrier_price).any(-1)
        v[~flag] = 0
        v[flag] = np.maximum(self.strike_price - v[flag], 0)
        return v

    def payoff_barrier_out_up_put(self, gbm):
        v = gbm[..., -1].copy()
        flag = (gbm < self.barrier_price).all(-1)
        v[~flag] = 0
        v[flag] = np.maximum(self.strike_price - v[flag], 0)
        return v

    def payoff_barrier_in_down_put(self, gbm):
        v = gbm[..., -1].copy()
        flag = (gbm <= self.barrier_price).any(-1)
        v[~flag] = 0
        v[flag] = np.maximum(self.strike_price - v[flag], 0)
        return v

    def payoff_barrier_out_down_put(self, gbm):
        v = gbm[..., -1].copy()
        flag = (gbm > self.barrier_price).all(-1)
        v[~flag] = 0
        v[flag] = np.maximum(self.strike_price - v[flag], 0)
        return v

    def payoff_lookback_call(self, gbm):  # include start price in min
        min_path = np.minimum(gbm.min(-1), self.start_price)
        return gbm[..., -1] - min_path

    def payoff_lookback_put(self, gbm):  # include start price in max
        max_path = np.maximum(gbm.max(-1), self.start_price)
        return max_path - gbm[..., -1]

    def payoff_digital_call(self, gbm):
        return np.where(gbm[..., -1] >= self.strike_price, self.digital_payout, 0)

    def payoff_digital_put(self, gbm):
        return np.where(gbm[..., -1] <= self.strike_price, self.digital_payout, 0)

    def get_exact_value(self):
        """
        Compute the exact analytic fair price of the option in finite dimensions. Supports

        - `option='EUROPEAN'`
        - `option='ASIAN'` with `asian_mean='GEOMETRIC'` and `asian_mean_quadrature_rule='RIGHT'`

        Returns:
            mean (float): Exact value of the integral.
        """
        if self.option == "EUROPEAN":
            denom = self.volatility * np.sqrt(self.t_final)
            decay = self.strike_price * self.discount_factor
            if self.call_put == "CALL":
                term1 = (
                    np.log(self.start_price / self.strike_price)
                    + (self.interest_rate + self.volatility**2 / 2) * self.t_final
                )
                term2 = (
                    np.log(self.start_price / self.strike_price)
                    + (self.interest_rate - self.volatility**2 / 2) * self.t_final
                )
                fp = self.start_price * norm.cdf(term1 / denom) - decay * norm.cdf(
                    term2 / denom
                )
            elif self.call_put == "PUT":
                term1 = (
                    np.log(self.strike_price / self.start_price)
                    - (self.interest_rate - self.volatility**2 / 2) * self.t_final
                )
                term2 = (
                    np.log(self.strike_price / self.start_price)
                    - (self.interest_rate + self.volatility**2 / 2) * self.t_final
                )
                fp = decay * norm.cdf(term1 / denom) - self.start_price * norm.cdf(
                    term2 / denom
                )
        elif self.option == "ASIAN":
            assert (
                self.asian_mean == "GEOMETRIC"
                and self.asian_mean_quadrature_rule == "RIGHT"
            ), "exact value for Asian options only implemented for self.asian_mean=='GEOMETRIC' and self.asian_mean_quadrature_rule=='RIGHT'"
            Tbar = (1 + 1 / self.d) * self.t_final / 2
            sigmabar = self.volatility * np.sqrt((2 + 1 / self.d) / 3)
            rbar = self.interest_rate + (sigmabar**2 - self.volatility**2) / 2
            gmeancall, gmeanput = _eurogbmprice(
                self.start_price, rbar, Tbar, sigmabar, self.strike_price
            )
            if self.call_put == "CALL":
                fp = gmeancall * np.exp(rbar * Tbar - self.interest_rate * self.t_final)
            elif self.call_put == "PUT":
                fp = gmeanput * np.exp(rbar * Tbar - self.interest_rate * self.t_final)
            return fp
        else:
            raise ParameterError(
                "exact value not supported for option = %s" % self.option
            )
        return fp

    def get_exact_value_inf_dim(self):
        r"""
        Get the exact analytic fair price of the option in infinite dimensions. Supports

        - `option='ASIAN'` with `asian_mean='GEOMETRIC'`

        Returns:
            mean (float): Exact value of the integral.
        """
        if self.option == "ASIAN":
            assert (
                self.asian_mean == "GEOMETRIC"
            ), "get_exact_value_inf_dim for the Asian option only available for self.asian_mean=='GEOMETRIC'"
            sigma_g = self.volatility / np.sqrt(3)
            b = 1 / 2 * (self.interest_rate - 1 / 2 * sigma_g**2)
            d1 = (
                np.log(self.start_price / self.strike_price)
                + (b + 1 / 2 * sigma_g**2) * self.t_final
            ) / (sigma_g * np.sqrt(self.t_final))
            d2 = d1 - sigma_g * np.sqrt(self.t_final)
            f1 = self.start_price * np.exp((b - self.interest_rate) * self.t_final)
            f2 = self.strike_price * np.exp(-self.interest_rate * self.t_final)
            if self.call_put == "CALL":
                val = f1 * norm.cdf(d1) - f2 * norm.cdf(d2)
            elif self.call_put == "PUT":
                val = f2 * norm.cdf(-d2) - f1 * norm.cdf(-d1)
        else:
            raise NotImplementedError(
                "get_exact_value_inf_dim not implemented for option = %s" % self.option
            )
        return val

    def dimension_at_level(self, level):
        return self.d_coarsest * 2**level

    def _spawn(self, level, sampler):
        return FinancialOption(
            sampler=sampler,
            option=self.option,
            call_put=self.call_put,
            volatility=self.volatility,
            start_price=self.start_price,
            strike_price=self.strike_price,
            interest_rate=self.interest_rate,
            t_final=self.t_final,
            decomp_type=self.decomp_type,
            level=level,
            d_coarsest=self.d_coarsest,
            asian_mean=self.asian_mean,
            asian_mean_quadrature_rule=self.asian_mean_quadrature_rule,
            barrier_in_out=self.barrier_in_out,
            barrier_price=self.barrier_price,
            digital_payout=self.digital_payout,
        )


def _eurogbmprice(S0, r, T, sigma, K):
    priceratio = K * np.exp(-r * T) / S0
    xbig = np.log(priceratio) / (sigma * np.sqrt(T)) + sigma * np.sqrt(T) / 2
    xsmall = np.log(priceratio) / (sigma * np.sqrt(T)) - sigma * np.sqrt(T) / 2
    putprice = S0 * (priceratio * norm.cdf(xbig) - norm.cdf(xsmall))
    callprice = putprice + S0 * (1 - priceratio)
    return callprice, putprice


class AsianOption(FinancialOption):
    def __init__(self, *args, **kwargs):
        """Deprecated, please use FinancialOption"""
        if "option" in kwargs:
            raise ParameterError("please do not pass 'option' to AsianOption")
        super(AsianOption, self).__init__(*args, **kwargs, option="ASIAN")


class EuropeanOption(FinancialOption):
    def __init__(self, *args, **kwargs):
        """Deprecated, please use FinancialOption"""
        if "option" in kwargs:
            raise ParameterError("please do not pass 'option' to EuropeanOption")
        super(EuropeanOption, self).__init__(*args, **kwargs, option="EUROPEAN")


class BarrierOption(FinancialOption):
    def __init__(self, *args, **kwargs):
        """Deprecated, please use FinancialOption"""
        if "option" in kwargs:
            raise ParameterError("please do not pass 'option' to BarrierOption")
        super(BarrierOption, self).__init__(*args, **kwargs, option="BARRIER")


class LookbackOption(FinancialOption):
    def __init__(self, *args, **kwargs):
        """Deprecated, please use FinancialOption"""
        if "option" in kwargs:
            raise ParameterError("please do not pass 'option' to LookbackOption")
        super(LookbackOption, self).__init__(*args, **kwargs, option="LOOKBACK")


class DigitalOption(FinancialOption):
    def __init__(self, *args, **kwargs):
        """Deprecated, please use FinancialOption"""
        if "option" in kwargs:
            raise ParameterError("please do not pass 'option' to DigitalOption")
        super(DigitalOption, self).__init__(*args, **kwargs, option="DIGITAL")
