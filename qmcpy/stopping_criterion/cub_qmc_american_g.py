from .abstract_stopping_criterion import AbstractStoppingCriterion
from .cub_qmc_net_g import CubQMCNetG
from .cub_qmc_lattice_g import CubQMCLatticeG
from .cub_mc_clt import CubMCCLT
from .cub_qmc_rep_student_t import CubQMCRepStudentT
from ..discrete_distribution import DigitalNetB2, Lattice
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
    AbstractIIDDiscreteDistribution,
)
from ..integrand.abstract_integrand import AbstractIntegrand
from ..true_measure import GeometricBrownianMotion
from ..util import ParameterError, ParameterWarning
from typing import Union, Tuple, Any
from time import time
import warnings
import numpy as np


class CubQMCAmericanG(AbstractStoppingCriterion):
    r"""
    American Put Option stopping criterion using Longstaff-Schwartz policy training
    and Quasi-Monte Carlo / Monte Carlo pricing integration.

    Note:
        The error tolerance (`abs_tol`, `rel_tol`) bounds the numerical integration
        error of the option payoff given the trained exercise policy (`n_train`). Policy
        training error from finite `n_train` is separate and controlled by choosing `n_train`.

    Examples:
        >>> from qmcpy import FinancialOption, Sobol, CubQMCAmericanG
        >>> opt = FinancialOption(Sobol(10, seed=7), option="AMERICAN", call_put="PUT")
        >>> sc = CubQMCAmericanG(opt, abs_tol=0.05, n_train=2**11)
        >>> solution, data = sc.integrate()
        >>> solution
        array(9.13...)
        >>> data  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Data (Data)
            solution        9.13...
            ...
    """

    def __init__(
        self,
        integrand: AbstractIntegrand,
        abs_tol: float = 1e-2,
        rel_tol: float = 0.0,
        n_init: int = 2**10,
        n_limit: int = 2**30,
        n_train: int = 2**12,
        training_discrete_distrib: Union[None, AbstractDiscreteDistribution] = None,
        stopping_criterion: Union[None, type, AbstractStoppingCriterion] = None,
        **kwargs: dict
    ) -> None:
        r"""Initialize the CubQMCAmericanG stopping criterion.

        Args:
            integrand (AbstractIntegrand): FinancialOption instance with option="AMERICAN".
            abs_tol (float): Absolute error tolerance for integration.
            rel_tol (float): Relative error tolerance for integration.
            n_init (int): Initial number of samples for pricing integration.
            n_limit (int): Maximum number of samples for pricing integration.
            n_train (int): Number of LSM training samples.
            training_discrete_distrib (Union[None, AbstractDiscreteDistribution]): Discrete distribution
                used to generate training paths for the LSM regression. If None, defaults to an
                independent spawn of integrand.discrete_distrib.
            stopping_criterion (Union[None, type, AbstractStoppingCriterion]): Custom stopping criterion
                class or instance to use for pricing stage.
            **kwargs (dict): Extra keyword arguments passed to the pricing stopping criterion.
        """
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.n_init = n_init
        self.n_limit = n_limit
        self.n_train = int(n_train)
        if not (self.n_train > 0):
            raise ParameterError("n_train must be a positive integer")
        self.stopping_criterion_custom = stopping_criterion
        self.inner_kwargs = kwargs

        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib

        if getattr(self.integrand, "option", None) != "AMERICAN":
            raise ParameterError(
                "CubQMCAmericanG requires a FinancialOption with option='AMERICAN'"
            )

        self.training_discrete_distrib = training_discrete_distrib
        if self.training_discrete_distrib is not None:
            if not isinstance(self.training_discrete_distrib, AbstractDiscreteDistribution):
                raise ParameterError(
                    "training_discrete_distrib must be an instance of AbstractDiscreteDistribution"
                )
            if self.training_discrete_distrib.d != self.integrand.d:
                raise ParameterError(
                    f"training_discrete_distrib dimension ({self.training_discrete_distrib.d}) "
                    f"must match integrand dimension ({self.integrand.d})"
                )

        if isinstance(stopping_criterion, AbstractStoppingCriterion):
            conflicting = []
            for param in ["abs_tol", "rel_tol", "n_init", "n_limit"]:
                if hasattr(stopping_criterion, param):
                    sc_val = getattr(stopping_criterion, param)
                    self_val = getattr(self, param)
                    if sc_val != self_val:
                        conflicting.append(
                            f"{param} (CubQMCAmericanG: {self_val}, stopping_criterion: {sc_val})"
                        )
            if conflicting:
                warnings.warn(
                    "Conflicting input values detected between CubQMCAmericanG and the provided "
                    f"stopping_criterion instance: {', '.join(conflicting)}. "
                    "The stopping_criterion instance values will be used.",
                    ParameterWarning,
                )

        self.parameters = [
            "abs_tol",
            "rel_tol",
            "n_init",
            "n_limit",
            "n_train",
        ]
        if self.training_discrete_distrib is not None:
            self.parameters.append("training_discrete_distrib")

        super(CubQMCAmericanG, self).__init__(
            allowed_distribs=[AbstractDiscreteDistribution],
            allow_vectorized_integrals=False,
        )

    def _select_inner_stopping_criterion(self):
        if self.stopping_criterion_custom is not None:
            if isinstance(self.stopping_criterion_custom, type):
                return self.stopping_criterion_custom(
                    self.integrand,
                    abs_tol=self.abs_tol,
                    rel_tol=self.rel_tol,
                    n_init=self.n_init,
                    n_limit=self.n_limit,
                    **self.inner_kwargs
                )
            else:
                self.stopping_criterion_custom.integrand = self.integrand
                return self.stopping_criterion_custom

        distrib = self.discrete_distrib
        if isinstance(distrib, AbstractIIDDiscreteDistribution):
            if not distrib.no_replications:
                unrep = type(distrib)(dimension=distrib.d, seed=distrib._base_seed)
                pricing_integrand = self.integrand._spawn(None, unrep)
                pricing_integrand.betas = self.integrand.betas
                return CubMCCLT(
                    pricing_integrand,
                    abs_tol=self.abs_tol,
                    rel_tol=self.rel_tol,
                    n_init=self.n_init,
                    n_limit=self.n_limit,
                    **self.inner_kwargs
                )
            else:
                return CubMCCLT(
                    self.integrand,
                    abs_tol=self.abs_tol,
                    rel_tol=self.rel_tol,
                    n_init=self.n_init,
                    n_limit=self.n_limit,
                    **self.inner_kwargs
                )
        elif distrib.replications > 1:
            return CubQMCRepStudentT(
                self.integrand,
                abs_tol=self.abs_tol,
                rel_tol=self.rel_tol,
                n_init=self.n_init,
                n_limit=self.n_limit,
                **self.inner_kwargs
            )
        elif (
            isinstance(distrib, DigitalNetB2)
            and distrib.order == "RADICAL INVERSE"
            and distrib.no_replications
        ):
            return CubQMCNetG(
                self.integrand,
                abs_tol=self.abs_tol,
                rel_tol=self.rel_tol,
                n_init=self.n_init,
                n_limit=self.n_limit,
                **self.inner_kwargs
            )
        elif isinstance(distrib, Lattice) and distrib.no_replications:
            return CubQMCLatticeG(
                self.integrand,
                abs_tol=self.abs_tol,
                rel_tol=self.rel_tol,
                n_init=self.n_init,
                n_limit=self.n_limit,
                **self.inner_kwargs
            )
        else:
            # Replicated LD distribution (e.g. Halton with replications=1)
            try:
                replicated_sampler = type(distrib)(
                    dimension=distrib.d,
                    replications=32,
                    seed=distrib._base_seed
                )
                pricing_integrand = self.integrand._spawn(None, replicated_sampler)
                pricing_integrand.betas = self.integrand.betas
                return CubQMCRepStudentT(
                    pricing_integrand,
                    abs_tol=self.abs_tol,
                    rel_tol=self.rel_tol,
                    n_init=self.n_init,
                    n_limit=self.n_limit,
                    **self.inner_kwargs
                )
            except Exception:
                return CubMCCLT(
                    self.integrand,
                    abs_tol=self.abs_tol,
                    rel_tol=self.rel_tol,
                    n_init=self.n_init,
                    n_limit=self.n_limit,
                    **self.inner_kwargs
                )

    def integrate(self, resume: Union[None, object] = None) -> Tuple[float, Any]:
        """
        Train the Longstaff-Schwartz exercise policy and integrate the option payoff.

        Args:
            resume (Union[None, object]): Not supported by this stopping criterion.
                Defaults to None.

        Returns:
            Tuple[float, Any]: Estimated American option price and integration data container.
        """
        t_start = time()
        trace = self._make_trace_logger()
        if resume is not None:
            raise ParameterError("CubQMCAmericanG does not support resume.")

        # Step 1: LSM Policy Training Phase
        t_train_start = time()
        if self.training_discrete_distrib is not None:
            training_sampler = self.training_discrete_distrib
        elif isinstance(self.discrete_distrib, AbstractIIDDiscreteDistribution):
            training_sampler = type(self.discrete_distrib)(
                dimension=self.discrete_distrib.d, seed=self.discrete_distrib._base_seed
            )
        else:
            training_sampler = self.discrete_distrib.spawn(1)[0]

        training_gbm = GeometricBrownianMotion(
            training_sampler,
            t_final=self.integrand.t_final,
            initial_value=self.integrand.start_price,
            drift=self.integrand.interest_rate,
            diffusion=self.volatility**2 if hasattr(self, "volatility") else self.integrand.volatility**2,
            decomp_type=self.integrand.decomp_type,
        )
        training_paths = training_gbm.gen_samples(self.n_train)
        self.integrand.train_american_policy(training_paths)
        time_train = time() - t_train_start

        # Step 2: Frozen-Policy Pricing Phase via selected inner stopping criterion
        inner_sc = self._select_inner_stopping_criterion()
        solution, data = inner_sc.integrate()

        # Check immediate exercise payoff at t=0
        now = (
            max(self.integrand.strike_price - self.integrand.start_price, 0.0)
            if self.integrand.call_put == "PUT"
            else max(self.integrand.start_price - self.integrand.strike_price, 0.0)
        )
        if now > solution:
            solution = (
                np.full_like(solution, now)
                if isinstance(solution, np.ndarray)
                else float(now)
            )
            data.solution = solution
            if hasattr(data, "comb_bound_low"):
                data.comb_bound_low = np.maximum(data.comb_bound_low, now)
            if hasattr(data, "comb_bound_high"):
                data.comb_bound_high = np.maximum(data.comb_bound_high, now)
            if hasattr(data, "bound_low"):
                data.bound_low = np.maximum(data.bound_low, now)
            if hasattr(data, "bound_high"):
                data.bound_high = np.maximum(data.bound_high, now)

        # Step 3: Attach training metadata and finalize
        data.n_train = self.n_train
        data.time_train = time_train
        data.betas = self.integrand.betas
        data.stopping_crit = self

        self._set_elapsed_time(data, time() - t_start)
        trace.iteration(data)
        self._finalize_integration_data(data, time() - t_start)
        trace.finalize()

        return solution, data

        # Step 3: Attach training metadata and finalize
        data.n_train = self.n_train
        data.time_train = time_train
        data.betas = self.integrand.betas
        data.stopping_crit = self

        self._set_elapsed_time(data, time() - t_start)
        trace.iteration(data)
        self._finalize_integration_data(data, time() - t_start)
        trace.finalize()

        return solution, data

    def set_tolerance(
        self,
        abs_tol: Union[None, float] = None,
        rel_tol: Union[None, float] = None,
        rmse_tol: Union[None, float] = None,
    ) -> None:
        """
        Set error tolerances for the stopping criterion.

        Args:
            abs_tol (Union[None, float]): Absolute error tolerance.
                Defaults to None.
            rel_tol (Union[None, float]): Relative error tolerance.
                Defaults to None.
            rmse_tol (Union[None, float]): RMSE error tolerance (not supported).
                Defaults to None.
        """
        if rmse_tol is not None:
            raise ParameterError("rmse_tol not supported by this stopping criterion.")
        if abs_tol is not None:
            self.abs_tol = abs_tol
        if rel_tol is not None:
            self.rel_tol = rel_tol
