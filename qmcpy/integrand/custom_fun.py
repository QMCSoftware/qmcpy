from warnings import warn

from ..true_measure.abstract_true_measure import AbstractTrueMeasure
from typing import Union, Callable
from .abstract_integrand import AbstractIntegrand
from ..discrete_distribution import DigitalNetB2  #pylint: disable=unused-import
from ..true_measure import Gaussian, Uniform  #pylint: disable=unused-import
import numpy as np  #pylint: disable=unused-import


class CustomFun(AbstractIntegrand):
    r"""User supplied integrand $g$. In the following example we implement

    Examples:
        First we will implement

        $$g(\boldsymbol{t}) = t_1^2t_2, \qquad \boldsymbol{T}=(T_1,T_2) \sim
        \mathcal{N}((1,2)^T,\mathsf{I}).$$

        >>> integrand = CustomFun(
        ...     true_measure = Gaussian(DigitalNetB2(2,seed=7),mean=[1,2]),
        ...     g = lambda t: t[...,0]**2*t[...,1])
        >>> y = integrand(2**10)
        >>> print("%.4f"%y.mean())
        3.9991

        With independent replications

        >>> integrand = CustomFun(
        ...     true_measure = Gaussian(DigitalNetB2(2,seed=7,replications=2**4),mean=[1,2]),
        ...     g = lambda t: t[...,0]**2*t[...,1])
        >>> y = integrand(2**6)
        >>> y.shape
        (16, 64)
        >>> muhats = y.mean(1)
        >>> muhats.shape
        (16,)
        >>> print("%.4f"%muhats.mean())
        3.9330

        Next we will implement the multi-output function

        $$g(\boldsymbol{t}) = \begin{pmatrix} \sin(t_1)\cos(t_2) \\
        \cos(t_1)\sin(t_2) \\ \sin(t_1)+\cos(t_2) \\ \cos(t_1)+\sin(t_2)
        \end{pmatrix} \qquad \boldsymbol{T}=(T_1,T_2) \sim
        \mathcal{U}[0,2\pi]^2.$$

        >>> def g(t):
        ...     t1,t2 = t[...,0],t[...,1]
        ...     sint1,cost1,sint2,cost2 = np.sin(t1),np.cos(t1),np.sin(t2),np.cos(t2)
        ...     y1 = sint1*cost2
        ...     y2 = cost1*sint2
        ...     y3 = sint1+cost2
        ...     y4 = cost1+sint2
        ...     y = np.stack([y1,y2,y3,y4])
        ...     return y
        >>> integrand = CustomFun(
        ...     true_measure = Uniform(DigitalNetB2(2,seed=7),lower_bound=0,upper_bound=2*np.pi),
        ...     g = g,
        ...     dimension_indv = (4,))
        >>> x = integrand.discrete_distrib(2**10)
        >>> y = integrand.f(x)
        >>> y.shape
        (4, 1024)
        >>> with np.printoptions(formatter={"float": lambda x: "%.2e"%x}):
        ...     y.mean(-1)
        array([8.18e-04, 1.92e-06, -2.26e-10, 5.05e-07])

        Stopping criterion which supporting vectorized outputs may pass in
        Boolean `compute_flags` with `dimension_indv` shape indicating which
        output need to evaluated,
            i.e., where `compute_flags` is `False` we do not need to evaluate
            the integrand. We have not used this in inexpensive example above.

        With independent replications

        >>> integrand = CustomFun(
        ...     true_measure = Uniform(DigitalNetB2(2,seed=7,replications=2**4),lower_bound=0,upper_bound=2*np.pi),
        ...     g = g,
        ...     dimension_indv = (4,))
        >>> x = integrand.discrete_distrib(2**6)
        >>> x.shape
        (16, 64, 2)
        >>> y = integrand.f(x)
        >>> y.shape
        (4, 16, 64)
        >>> muhats = y.mean(-1)
        >>> muhats.shape
        (4, 16)
        >>> with np.printoptions(formatter={"float": lambda x: "%.2e"%x}):
        ...     muhats.mean(-1)
        array([3.83e-03, -6.78e-03, -1.56e-03, -5.65e-04])
    """

    def __init__(self, true_measure: AbstractTrueMeasure, g: Callable, dimension_indv: tuple = (),
                 parallel: Union[bool, int] = False, *,
                 dimension_comb: Union[None, tuple] = None,
                 bound_fun: Union[None, Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]] = None,
                 dependency: Union[None, Callable[[np.ndarray], np.ndarray]] = None,
                 comb_fun: Union[None, Callable[[np.ndarray], np.ndarray]] = None,
                 comb_fun_n_samples: int = 2**12) -> None:
        """Initialize a CustomFun integrand.

        Args:
            true_measure (AbstractTrueMeasure): The true measure.
            g (Callable): A function handle.
            dimension_indv (tuple): Shape of individual solution outputs from
                `g`.
            parallel (Union[bool, int]): Parallelization flag.
            dimension_comb (Union[None, tuple]): shapes of outputs from comb_fun. Default is `dimension_indv`.
                Must be specified if `comb_fun` is used.
            bound_fun (Union[None, Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]): function that
                combines the upper and lower bounds on `g` to lower and upper bounds on quantities of interest.
                Default is identity functions so that the integrands themselves are the quantities of interest.
                If `comb_fun` is specified and `bound_fun` is not, `comb_fun` is sampled to get approximate
                bounds.
                Inputs to this function has shape [dimension_indv]
                Outputs should have shape [dimension_comb]
            dependency (Union[None, Callable[[np.ndarray], np.ndarray]]): Function defining how convergence of combined integrands is
                related to convergence of the underlying integrands. See `AbstractIntegrand.dependency`
            comb_fun (Union[None, Callable[[np.ndarray], np.ndarray]]): A function combining individual solutions from `g`
                into quantities of interest. StoppingCriteria use this to check convergence if `dimension_indv != dimension_comb` and
                `bound_fun` is not given.
                Default is identity function.
                Inputs to this function has shape [num_samples,dimension_indv]
                Outputs should have shape [num_samples, dimension_comb]
            comb_fun_n_samples (int): Number of samples used when sampling `comb_fun` to derive bounds on itegrands.

                - When `parallel = 0` or `parallel = 1` then function evaluation is done in serial fashion.
                - `parallel > 1` specifies the number of processes used by `multiprocessing.Pool` or `multiprocessing.pool.ThreadPool`.

                Setting `parallel=True` is equivalent to `parallel =
                os.cpu_count()`.

        Notes:
            For `parallel > 1` do *not* set `g` to be anonymous function (i.e.
            a `lambda` function)
        """
        self.parameters = []
        self.true_measure = true_measure
        self.comb_fun_n_samples = comb_fun_n_samples
        self.sampler = self.true_measure
        self.__g = g
        self._comb_func = comb_fun
        self._bound_fun = bound_fun
        if (comb_fun is not None):
            if (bound_fun is not None):
                # `comb_fun` is not used - warn user about actual behavior
                warn("CustomFun: both `comb_fun` and `bound_fun` were specified. `comb_fun` is ignored when specifying `bound_fun`")
            else:
                # `comb_fun` is used - warn about bounds found by sampling
                warn("CustomFun: bounds derived by sampling `comb_fun` are approximate and generally tighter than those found by propogating bounds using a proper `bound_fun`.")
        if comb_fun is None and bound_fun is None:
            dimension_comb = dimension_indv
        elif dimension_comb is None:
            raise ValueError("output dimension `dimension_comb` must be specified for non identity `bound_fun` or `comb_fun`")
        self._dependency_fun = dependency
        super(CustomFun, self).__init__(
            dimension_indv=dimension_indv,
            dimension_comb=dimension_comb,
            parallel=parallel,
        )

    def g(self, t: np.ndarray, *args: tuple, **kwargs: dict) -> np.ndarray:
        """Evaluate the user-supplied function.

        Args:
            t (np.ndarray): Points distributed by the true measure.
            *args (tuple): Positional arguments forwarded to the user function.
            **kwargs (dict): Keyword arguments forwarded to the user function.

        Returns:
            np.ndarray: Function values.
        """
        return self.__g(t, *args, **kwargs)

    def bound_fun(self, bound_low: np.ndarray, bound_high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute bounds for quantities of interest obtained by `comb_fun`.

        Args:
            bound_low (np.ndarray): lower bounds for integrands
            bound_high (np.ndarray): upper bounds for integrands

        Returns:
            tuple[np.ndarray, np.ndarray]: Lower and upper bounds for quantities
        """
        if self._comb_func is None and self._bound_fun is None:
            return super().bound_fun(bound_low, bound_high)
        elif (self._bound_fun is None) and (self._comb_func is not None):
            # TODO: allow specifying tolerance, stopping criteria etc?
            # sample in the space [bound_low, bound_high]^N to find approximate bounds
            uniform_sampler = Uniform(DigitalNetB2(self.d_indv[0]), bound_low, bound_high)
            uniform_samples: np.ndarray = uniform_sampler(self.comb_fun_n_samples, return_weights=False)
            # uniform_samples.shape == [num_samples, dimension_indv]
            comb_samples = self._comb_func(uniform_samples) # [num_samples, dimension_comb]
            # take min and max
            approx_min, approx_max = np.min(comb_samples, axis=0), np.max(comb_samples, axis=0)
            return approx_min, approx_max
        elif self._bound_fun is not None:
            return self._bound_fun(bound_low, bound_high)

    def dependency(self, is_converged: np.ndarray) -> np.ndarray:
        """Takes a vector of indicators of whether of not the error bound is
        satisfied for combined integrands and returns flags for individual
        integrands.

        For example, if we are taking the ratio of 2 individual integrands,
        then getting `is_converged=True` means the ratio has been
        approximated to within the tolerance, so the dependency function should
        return `indv_flags=[True,True]` indicating that both the numerator and
        denominator integrands have converged.

        The default behavior is to keep calculating all integrals until all
        combined integrands have converged.

        Args:
            is_converged (np.ndarray): Flags of shape `integrand.d_comb`
                indicating whether the combined outputs are converged.

        Returns:
            np.ndarray: Flags of shape `integrand.d_indv` indicating True where the individual
                integrands are converged.
        """
        if self._dependency_fun is not None:
            return self._dependency_fun(is_converged)
        return super().dependency(is_converged)

    def _spawn(self, level, sampler):
        return CustomFun(
            true_measure=sampler,
            g=self.__g,
            dimension_indv=self.d_indv,
            comb_fun=self._comb_func,
            dimension_comb=self.d_comb,
            bound_fun=self._bound_fun,
            dependency=self._dependency_fun,
            comb_fun_n_samples=self.comb_fun_n_samples,
            parallel=self.parallel,
        )
