from ..util import MethodImplementationError, _univ_repr, ParameterError
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
import numpy as np
from scipy import sparse


def _clip_unit_interval(u):
    """Clip unit-interval values away from endpoints for stable quantiles."""
    eps = np.finfo(float).eps
    return np.clip(u, eps, 1.0 - eps)


class AbstractTrueMeasure(object):

    def __init__(self):
        prefix = "A concrete implementation of TrueMeasure must have "
        if not hasattr(self, "domain"):
            raise ParameterError(
                prefix
                + "self.domain, 2xd ndarray of domain lower bounds (first col) and upper bounds (second col)"
            )
        if not hasattr(self, "range"):
            raise ParameterError(
                prefix
                + "self.range, 2xd ndarray of range lower bounds (first col) and upper bounds (second col)"
            )
        if not hasattr(self, "parameters"):
            self.parameters = []

    @staticmethod
    def _read_only_array(value):
        """Return an owned, read only array containing ``value``."""
        array = np.array(value, copy=True)
        array.setflags(write=False)
        return array

    @staticmethod
    def _range_in_domain(transform_range, domain):
        """Return whether a transform range is contained within a domain."""
        try:
            transform_range = np.asarray(transform_range)
            domain = np.asarray(domain)
        except (TypeError, ValueError):
            return False

        if (
            transform_range.ndim != 2
            or domain.ndim != 2
            or transform_range.shape[1] != 2
            or domain.shape[1] != 2
            or transform_range.shape[0] == 0
            or domain.shape[0] == 0
        ):
            return False

        if not (
            np.issubdtype(transform_range.dtype, np.number)
            and np.issubdtype(domain.dtype, np.number)
            and np.isrealobj(transform_range)
            and np.isrealobj(domain)
            and transform_range.dtype != np.bool_
            and domain.dtype != np.bool_
        ):
            return False

        if np.isnan(transform_range).any() or np.isnan(domain).any():
            return False

        if np.any(transform_range[:, 0] > transform_range[:, 1]) or np.any(
            domain[:, 0] > domain[:, 1]
        ):
            return False

        try:
            transform_range, domain = np.broadcast_arrays(transform_range, domain)
        except ValueError:
            return False

        lower_bounds_valid = np.all(domain[:, 0] <= transform_range[:, 0])
        upper_bounds_valid = np.all(transform_range[:, 1] <= domain[:, 1])
        return bool(lower_bounds_valid and upper_bounds_valid)

    def _set_moments(self, mean, variance, standard_deviation, covariance):
        self._mean = self._read_only_array(mean)
        self._variance = self._read_only_array(variance)
        self._standard_deviation = self._read_only_array(standard_deviation)
        if sparse.issparse(covariance):
            covariance.data.setflags(write=False)
            self._covariance = covariance
        else:
            self._covariance = self._read_only_array(covariance)

    @staticmethod
    def _read_only_view(value):
        """Return a view which cannot be made writeable while its base is read only."""
        view = value.view()
        view.setflags(write=False)
        return view

    def _scalar_if_univariate(self, value):
        """For univariate (``d == 1``) measures, return a Python ``float`` scalar
        (via :func:`numpy.squeeze`); otherwise return a read only array view."""
        if getattr(self, "d", None) == 1:
            return float(np.squeeze(value))
        return self._read_only_view(value)

    # store mean, variance, standard deviation, and covariance as read only array.
    # For univariate measures the mean, variance, and standard deviation are
    # returned as Python float scalars rather than length-1 arrays.

    @property
    def mean(self):
        return self._scalar_if_univariate(self._mean)

    @property
    def variance(self):
        return self._scalar_if_univariate(self._variance)

    @property
    def standard_deviation(self):
        return self._scalar_if_univariate(self._standard_deviation)

    @property
    def covariance(self):
        covariance = self._covariance
        if sparse.issparse(covariance):
            return covariance
        return self._read_only_view(covariance)

    def _parse_sampler(self, sampler):
        self.sub_compatibility_error = False
        if isinstance(sampler, AbstractDiscreteDistribution):
            self.transform = self  # this is the initial transformation, \Psi_0
            self.d = sampler.d  # take the dimension from the discrete distribution
            self.discrete_distrib = sampler
            if sampler.mimics == "StdUniform":
                if not (self.domain == np.tile([0, 1], (self.d, 1))).all():
                    raise ParameterError(
                        "The True measure's transform should have unit-cube domain."
                    )
            else:
                raise ParameterError(
                    "The %s true measure does not support discrete distributions that mimic a %s."
                    % (type(self).__name__, sampler.mimics)
                )
        elif isinstance(sampler, AbstractTrueMeasure):
            self.transform = sampler  # this is a composed transform, \Psi_j for j>0
            self.parameters += ["transform"]
            self.d = (
                sampler.d
            )  # take the dimension from the sub-sampler (composed transform)
            self.discrete_distrib = self.transform.discrete_distrib
            if not self._range_in_domain(self.transform.range, self.domain):
                self.sub_compatibility_error = True
            if self.transform.sub_compatibility_error:
                raise ParameterError(
                    "The sub-sub-transform range must be contained within the sub-transform domain."
                )
        else:
            raise ParameterError(
                "sampler input should either be a AbstractDiscreteDistribution or AbstractTrueMeasure"
            )

    def __call__(self, n=None, n_min=None, n_max=None, return_weights=False, warn=True):
        r"""
        - If just `n` is supplied, generate samples from the sequence at indices 0,...,`n`-1.
        - If `n_min` and `n_max` are supplied, generate samples from the sequence at indices `n_min`,...,`n_max`-1.
        - If `n` and `n_min` are supplied, then generate samples from the sequence at indices `n`,...,`n_min`-1.

        Args:
            n (Union[None, int]): Number of points to generate.
            n_min (Union[None, int]): Starting index of sequence.
            n_max (Union[None, int]): Final index of sequence.
            return_weights (bool): If `True`, return `weights` as well
            warn (bool): If `False`, disable warnings when generating samples.

        Returns:
            t (np.ndarray): Samples from the sequence.

                - If `replications` is `None` then this will be of size (`n_max`-`n_min`) $\times$ `dimension`
                - If `replications` is a positive int, then `t` will be of size `replications` $\times$ (`n_max`-`n_min`) $\times$ `dimension`
            weights (np.ndarray): Only returned when `return_weights=True`. The Jacobian weights for the transformation
        """
        return self.gen_samples(
            n=n, n_min=n_min, n_max=n_max, return_weights=return_weights, warn=warn
        )

    def gen_samples(
        self, n=None, n_min=None, n_max=None, return_weights=False, warn=True
    ):
        x = self.discrete_distrib(n=n, n_min=n_min, n_max=n_max, warn=warn)
        assert isinstance(return_weights, bool)
        return self._jacobian_transform_r(x=x, return_weights=return_weights)

    def _jacobian_transform_r(self, x, return_weights):
        r"""Recursive Jacobian transform."""
        jac = None
        if self.sub_compatibility_error:
            raise ParameterError(
                "The sub-transform range must be contained within the transform domain."
            )
        if self.transform == self:  # is \Psi_0
            if return_weights:
                t = self._transform(x)
                jac = 1 / self._weight(t)
            else:
                t = self._transform(x)
        else:  # is transform \Psi_j for j>0
            if return_weights:
                t_sub, jac_sub = self.transform._jacobian_transform_r(
                    x=x, return_weights=return_weights
                )
                t = self._transform(t_sub)
                jac = jac_sub / self._weight(t)  # |\Psi1/\lambda(\psi(x))
            else:
                t_sub = self.transform._jacobian_transform_r(
                    x=x, return_weights=return_weights
                )
                t = self._transform(t_sub)
        if return_weights:
            return t, jac
        else:
            return t

    def _transform(self, x):
        r"""Transformation from the standard uniform to the true measure distribution."""
        raise MethodImplementationError(
            self,
            "_transform. Try setting sampler to be in a PDF AbstractTrueMeasure to importance sample by.",
        )

    def _weight(self, x):
        r"""
        Non-negative weight function.
        This is often a PDF, but is not required to be
        e.g., Lebesgue weight is always 1, but is not a PDF.

        Args:
            x (np.ndarray): n x d  matrix of samples

        Returns:
            np.ndarray: length n vector of weights at locations of x
        """
        raise MethodImplementationError(
            self, "weight. Try a different true measure with a _weight method."
        )

    def spawn(self, s=1, dimensions=None):
        r"""
        Spawn new instances of the current true measure but with new seeds and dimensions.
        Used by multi-level QMC algorithms which require different seeds and dimensions on each level.

        Note:
            Use `replications` instead of using `spawn` when possible, e.g., when spawning copies which all have the same dimension.

        Args:
            s (int): Number of copies to spawn
            dimensions (np.ndarray): Length `s` array of dimensions for each copy. Defaults to the current dimension.

        Returns:
            spawned_true_measures (list): True measure with new seeds and dimensions.
        """
        sampler = self.discrete_distrib if self.transform == self else self.transform
        sampler_spawns = sampler.spawn(s=s, dimensions=dimensions)
        spawned_true_measures = [None] * len(sampler_spawns)
        for i in range(s):
            spawned_true_measures[i] = self._spawn(
                sampler_spawns[i], sampler_spawns[i].d
            )
        return spawned_true_measures

    def _spawn(self, sampler, dimension):
        raise MethodImplementationError(self, "_spawn")

    def __repr__(self):
        return _univ_repr(self, "AbstractTrueMeasure", self.parameters)
