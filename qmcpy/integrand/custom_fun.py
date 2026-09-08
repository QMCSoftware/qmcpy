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
            i.e. where `compute_flags` is `False` we do not need to evaluate
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

    def __init__(self, true_measure, g, dimension_indv: tuple = (), parallel: int = False) -> None:
        """Initialize a CustomFun integrand.

        Args:
            true_measure (AbstractTrueMeasure): The true measure.
            g (callable): A function handle.
            dimension_indv (tuple): Shape of individual solution outputs from
                `g`.
            parallel (int): Parallelization flag.

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
        self.sampler = self.true_measure
        self.__g = g
        super(CustomFun, self).__init__(
            dimension_indv=dimension_indv,
            dimension_comb=dimension_indv,
            parallel=parallel,
        )

    def g(self, t, *args, **kwargs):
        return self.__g(t, *args, **kwargs)

    def _spawn(self, level, sampler):
        return CustomFun(
            true_measure=sampler,
            g=self.__g,
            dimension_indv=self.d_indv,
            parallel=self.parallel,
        )
