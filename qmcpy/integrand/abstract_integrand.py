from ..util import MethodImplementationError, _univ_repr, ParameterError
from ..true_measure.abstract_true_measure import AbstractTrueMeasure
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
import numpy as np
import os
from multiprocessing import get_context
from multiprocessing.pool import ThreadPool
from itertools import repeat


class AbstractIntegrand(object):

    def __init__(self, dimension_indv: tuple, dimension_comb: tuple, parallel: int, threadpool: bool = False) -> None:
        r"""Initialize an AbstractIntegrand integrand.

        Args:
            dimension_indv (tuple): Individual solution shape.
            dimension_comb (tuple): Combined solution shape.
            parallel (int): Parallelization flag.

                - When `parallel = 0` or `parallel = 1` then function evaluation is done in serial fashion.
                - `parallel > 1` specifies the number of processes used by `multiprocessing.Pool` or `multiprocessing.pool.ThreadPool`.

                Setting `parallel=True` is equivalent to `parallel =
                os.cpu_count()`.
            threadpool (bool): When `parallel > 1`:

                - Setting `threadpool = True` will use `multiprocessing.pool.ThreadPool`.
                - Setting `threadpool = False` will use `setting multiprocessing.Pool`.
        """
        prefix = "A concrete implementation of AbstractIntegrand must have "
        self.d = self.true_measure.d
        self.d_indv = (
            (dimension_indv,)
            if isinstance(dimension_indv, int)
            else tuple(dimension_indv)
        )
        self.d_comb = (
            (dimension_comb,)
            if isinstance(dimension_comb, int)
            else tuple(dimension_comb)
        )
        cpus = os.cpu_count()
        self.parallel = cpus if parallel is True else int(parallel)
        self.parallel = 0 if self.parallel == 1 else self.parallel
        self.threadpool = threadpool
        if not (
            hasattr(self, "sampler")
            and isinstance(
                self.sampler, (AbstractTrueMeasure, AbstractDiscreteDistribution)
            )
        ):
            raise ParameterError(
                prefix
                + "self.sampler, a AbstractTrueMeasure or AbstractDiscreteDistribution instance"
            )
        if not (
            hasattr(self, "true_measure")
            and isinstance(self.true_measure, AbstractTrueMeasure)
        ):
            raise ParameterError(
                prefix + "self.true_measure, a AbstractTrueMeasure instance"
            )
        if not hasattr(self, "parameters"):
            self.parameters = []
        if not hasattr(self, "multilevel"):
            self.multilevel = False
        if not (isinstance(self.multilevel, bool)):
            raise AssertionError
        if not hasattr(self, "max_level"):
            self.max_level = np.inf
        if not hasattr(self, "discrete_distrib"):
            self.discrete_distrib = self.true_measure.discrete_distrib
        if (
            self.true_measure.transform != self.true_measure
            and not (self.true_measure.range == self.true_measure.transform.range).all()
        ):
            raise ParameterError(
                "The range of the composed transform is not compatible with this true measure"
            )
        self.EPS = np.finfo(float).eps

    def __call__(self, n=None, n_min=None, n_max=None, warn=True):
        r"""
        - If just `n` is supplied, generate samples from the sequence at indices 0,...,`n`-1.
        - If `n_min` and `n_max` are supplied, generate samples from the sequence at indices `n_min`,...,`n_max`-1.
        - If `n` and `n_min` are supplied, then generate samples from the sequence at indices `n`,...,`n_min`-1.

        Args:
            n (Union[None, int]): Number of points to generate.
            n_min (Union[None, int]): Starting index of sequence.
            n_max (Union[None, int]): Final index of sequence.
            warn (bool): If `False`, disable warnings when generating samples.

        Returns:
            np.ndarray: Samples from the sequence.

                - If `replications` is `None` then this will be of size (`n_max`-`n_min`) $\times$ `dimension`
                - If `replications` is a positive int, then `t` will be of size `replications` $\times$ (`n_max`-`n_min`) $\times$ `dimension`
        """
        return self.gen_samples(n=n, n_min=n_min, n_max=n_max, warn=warn)

    def gen_samples(
        self, n=None, n_min=None, n_max=None, return_weights=False, warn=True
    ):
        x = self.discrete_distrib(n=n, n_min=n_min, n_max=n_max, warn=warn)
        y = self.f(x)
        return y

    def g(self, t: np.ndarray, *args: tuple, **kwargs: dict):
        r"""*Abstract method* implementing the integrand as a function of the
        true measure.

        Args:
            t (np.ndarray): Inputs with shape `(*batch_shape, d)`.
            args (tuple): positional arguments to `g`.
            kwargs (dict): keyword arguments to `g`.

                Some algorithms will additionally try to pass in a
                `compute_flags` keyword argument. This `np.ndarray` are flags
                indicating which outputs require evaluation. For example, if
                the vector function has 3 outputs and `compute_flags = [False,
                True, False]`, then the function is only required to evaluate
                the second output and may leave the remaining outputs as
                `np.nan` values, i.e., the outputs corresponding to
                `compute_flags` which are `False` will not be used in the
                computation.

        Returns:
            np.ndarray: function evaluations with shape `(*batch_shape, *dimension_indv)`
                where `dimension_indv` is the shape of the function outputs.
        """
        raise MethodImplementationError(self, "g")

    def f(self, x: np.ndarray, *args: tuple, **kwargs: dict):
        r"""Function to evaluate the transformed integrand as a function of
        the discrete distribution. Automatically applies the transformation
        determined by the true measure.

        Args:
            x (np.ndarray): Inputs with shape `(*batch_shape, d)`.
            args (tuple): positional arguments to `g`.
            kwargs (dict): keyword arguments to `g`.

                Some algorithms will additionally try to pass in a
                `compute_flags` keyword argument. This `np.ndarray` are flags
                indicating which outputs require evaluation. For example, if
                the vector function has 3 outputs and `compute_flags = [False,
                True, False]`, then the function is only required to evaluate
                the second output and may leave the remaining outputs as
                `np.nan` values, i.e., the outputs corresponding to
                `compute_flags` which are `False` will not be used in the
                computation.

                The keyword argument `periodization_transform`, a string,
                specifies a periodization transform. Options are:

                - `False`: No periodizing transform, $\psi(x) = x$.
                - `'BAKER'`: Baker tansform $\psi(x) = 1-2\lvert x-1/2 \rvert$.
                - `'C0'`: $C^0$ transform $\psi(x) = 3x^2-2x^3$.
                - `'C1'`: $C^1$ transform $\psi(x) = x^3(10-15x+6x^2)$.
                - `'C1SIN'`: Sidi $C^1$ transform $\psi(x) = x-\sin(2 \pi x)/(2 \pi)$.
                - `'C2SIN'`: Sidi $C^2$ transform $\psi(x) = (8-9 \cos(\pi x)+\cos(3 \pi x))/16$.
                - `'C3SIN'`: Sidi $C^3$ transform $\psi(x) = (12\pi x-8\sin(2 \pi x) + \sin(4 \pi x))/(12 \pi)$.

        Returns:
            np.ndarray: function evaluations with shape `(*batch_shape, *dimension_indv)`
                where `dimension_indv` is the shape of the function outputs.
        """
        if "periodization_transform" in kwargs:
            periodization_transform = kwargs["periodization_transform"]
            del kwargs["periodization_transform"]
        else:
            periodization_transform = "None"
        periodization_transform = str(periodization_transform).upper()
        batch_shape = tuple(x.shape[:-1])
        d_indv_ndim = len(self.d_indv)
        if periodization_transform == "NONE":
            periodization_transform = "FALSE"
        if (
            self.discrete_distrib.mimics != "StdUniform"
            and periodization_transform != "NONE"
        ):
            raise ParameterError(
                """
                Applying a periodization transform currently requires a discrete distribution 
                that mimics a standard uniform measure."""
            )
        if periodization_transform == "FALSE":
            xp = x
            wp = np.ones(batch_shape, dtype=float)
        elif periodization_transform == "BAKER":
            xp = 1 - 2 * abs(x - 1 / 2)
            wp = np.ones(batch_shape, dtype=float)
        elif periodization_transform == "C0":
            xp = 3 * x**2 - 2 * x**3
            wp = np.prod(6 * x * (1 - x), -1)
        elif periodization_transform == "C1":
            xp = x**3 * (10 - 15 * x + 6 * x**2)
            wp = np.prod(30 * x**2 * (1 - x) ** 2, -1)
        elif periodization_transform == "C1SIN":
            xp = x - np.sin(2 * np.pi * x) / (2 * np.pi)
            wp = np.prod(2 * np.sin(np.pi * x) ** 2, -1)
        elif periodization_transform == "C2SIN":
            xp = (8 - 9 * np.cos(np.pi * x) + np.cos(3 * np.pi * x)) / 16
            wp = np.prod(
                (9 * np.sin(np.pi * x) * np.pi - np.sin(3 * np.pi * x) * 3 * np.pi)
                / 16,
                -1,
            )
        elif periodization_transform == "C3SIN":
            xp = (
                12 * np.pi * x - 8 * np.sin(2 * np.pi * x) + np.sin(4 * np.pi * x)
            ) / (12 * np.pi)
            wp = np.prod(
                (
                    12 * np.pi
                    - 8 * np.cos(2 * np.pi * x) * 2 * np.pi
                    + np.sin(4 * np.pi * x) * 4 * np.pi
                )
                / (12 * np.pi),
                -1,
            )
        else:
            raise ParameterError(
                "The %s periodization transform is not implemented"
                % periodization_transform
            )
        if periodization_transform in ["C1", "C1SIN", "C2SIN", "C3SIN"]:
            xp[xp <= 0] = self.EPS
            xp[xp >= 1] = 1 - self.EPS
        if not (wp.shape == batch_shape):
            raise AssertionError
        if not (xp.shape == x.shape):
            raise AssertionError
        # function evaluation with chain rule
        i = (None,) * d_indv_ndim + (...,)
        if self.true_measure == self.true_measure.transform:
            # jacobian*weight/pdf will cancel so f(x) = g(\Psi(x))
            xtf = self.true_measure._jacobian_transform_r(
                xp, return_weights=False
            )  # get transformed samples, equivalent to self.true_measure._transform_r(x)
            if not (xtf.shape == xp.shape):
                raise AssertionError
            y = self._g(xtf, *args, **kwargs)
        else:  # using importance sampling --> need to compute pdf, jacobian(s), and weight explicitly
            pdf = self.discrete_distrib.pdf(xp)  # pdf of samples
            if not (pdf.shape == batch_shape):
                raise AssertionError
            xtf, jacobians = self.true_measure.transform._jacobian_transform_r(
                xp, return_weights=True
            )  # compute recursive transform+jacobian
            if not (xtf.shape == xp.shape):
                raise AssertionError
            if not (jacobians.shape == batch_shape):
                raise AssertionError
            weight = self.true_measure._weight(xtf)  # weight based on the true measure
            if not (weight.shape == batch_shape):
                raise AssertionError
            gvals = self._g(xtf, *args, **kwargs)
            if not (gvals.shape == (self.d_indv + batch_shape)):
                raise AssertionError
            y = gvals * weight[i] / pdf[i] * jacobians[i]
        if not (y.shape == (self.d_indv + batch_shape)):
            raise AssertionError
        # account for periodization weight
        y = y * wp[i]
        if not (y.shape == (self.d_indv + batch_shape)):
            raise AssertionError
        return y

    def _g(self, t, *args, **kwargs):
        if self.parallel:
            pool = (
                get_context(method="fork").Pool(processes=self.parallel)
                if self.threadpool == False
                else ThreadPool(processes=self.parallel)
            )
            t_flat = t.reshape((-1, t.shape[-1]))
            y_flat = pool.starmap(self._g2, zip(t_flat, repeat((args, kwargs))))
            pool.close()
            y = np.stack(y_flat, axis=-1).reshape((self.d_indv + t.shape[:-1]))
        else:
            y = self._g2(t, comb_args=(args, kwargs))
        expected_y_shape = self.d_indv + t.shape[:-1]
        if not (y.shape == expected_y_shape):
            raise AssertionError("expected y.shape to be %s but got %s" % (
                str(expected_y_shape),
                str(y.shape),
            ))
        return y

    def _g2(self, t, comb_args=((), {})):
        args = comb_args[0]
        kwargs = comb_args[1]
        try:
            y = self.g(t, *args, **kwargs)
        except TypeError as e:
            if "got an unexpected keyword argument 'compute_flags'" in str(e):
                del kwargs["compute_flags"]
                y = self.g(t, *args, **kwargs)
            else:
                raise e
        return y

    def bound_fun(self, bound_low: np.ndarray, bound_high: np.ndarray):
        """Compute the bounds on the combined function based on bounds for
        the individual functions.

        Defaults to the identity where we essentially do not combine
        integrands, but instead integrate each function individually.

        Args:
            bound_low (np.ndarray): Lower bounds on individual estimates with
                shape `integrand.d_indv`.
            bound_high (np.ndarray): Upper bounds on individual estimates with
                shape `integrand.d_indv`.

        Returns:
            tuple[np.ndarray, np.ndarray]: Lower and upper bounds on the
                combined estimates, respectively, each with shape `integrand.d_comb`.
        """
        if self.d_indv != self.d_comb:
            raise ParameterError(
                """
                Set bound_fun explicitly. 
                The default bound_fun is the identity map. 
                Since the individual solution dimensions d_indv = %s does not equal the combined solution dimensions d_comb = %s, 
                QMCPy cannot infer a reasonable bound function."""
                % (str(self.d_indv), str(self.d_comb))
            )
        return bound_low, bound_high

    def dependency(self, comb_flags: np.ndarray):
        """Takes a vector of indicators of weather of not the error bound is
        satisfied for combined integrands and returns flags for individual
        integrands.

        For example, if we are taking the ratio of 2 individual integrands,
        then getting `comb_flags=True` means the ratio has not been
        approximated to within the tolerance, so the dependency function should
        return `indv_flags=[True,True]` indicating that both the numerator and
        denominator integrands need to be better approximated.

        Args:
            comb_flags (np.ndarray): Flags of shape `integrand.d_comb`
                indicating whether the combined outputs are insufficiently
                approximated.

        Returns:
            np.ndarray: Flags of shape `integrand.d_indv` indicating whether the individual
                integrands require additional sampling.
        """
        return (
            comb_flags
            if self.d_indv == self.d_comb
            else np.tile((comb_flags == False).any(), self.d_indv)
        )

    def spawn(self, levels: np.ndarray):
        r"""Spawn new instances of the current integrand at different levels
        with new seeds. Used by multi-level QMC algorithms which require
        integrands at multiple levels.

        Notes:
            Use `replications` instead of using `spawn` when possible, e.g.,
            when spawning copies which all have the same level.

        Args:
            levels (np.ndarray): Levels at which to spawn new integrands.

        Returns:
            list: Integrands with new true measures and discrete distributions.
        """
        levels = np.array([levels]) if np.isscalar(levels) else np.array(levels)
        if (levels > self.max_level).any():
            raise ParameterError("requested spawn level exceeds max level")
        n_levels = len(levels)
        new_dims = np.array([self.dimension_at_level(level) for level in levels])
        tm_spawns = self.sampler.spawn(s=n_levels, dimensions=new_dims)
        spawned_integrand = [None] * n_levels
        for l, level in enumerate(levels):
            spawned_integrand[l] = self._spawn(level, tm_spawns[l])
        return spawned_integrand

    def dimension_at_level(self, level: int):
        """*Abstract method* which returns the dimension of the generator
        required for a given level.

        Notes:
            Only used for multilevel problems.

        Args:
            level (int): Level at which to return the dimension.

        Returns:
            int: Dimension at the given input level.
        """
        return self.d

    def _spawn(self, level, tm_spawn):
        raise MethodImplementationError(self, "_spawn")

    def __repr__(self):
        return _univ_repr(self, "AbstractIntegrand", self.parameters)
