import numpy as np
from scipy import sparse

from .abstract_true_measure import AbstractTrueMeasure
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
from ..util import DimensionError, ParameterError


class Mixture(AbstractTrueMeasure):
    r"""Mixture of true measures with fixed component probabilities.

    A sample ``u`` has one more coordinate than the mixture output. The first
    coordinate selects a component according to ``probabilities``; the
    remaining coordinates are transformed by that component. Cumulative
    intervals are left-closed and right-open, except that the final interval
    also includes ``u[0] == 1``.

    The samplers attached to the component true measures are not sampled.
    Components provide their full recursive transform and weight behavior.
    For an importance-sampling composition, the caller remains responsible for
    ensuring that the component's induced sampling distribution is appropriate
    for the intended mixture.

    When the necessary component statistics are available, mixture moments are
    exposed through ``mean``, ``variance``, ``standard_deviation``, and
    ``covariance``. Component ``range`` values are used to enforce zero mixture
    weight outside bounded component supports.

    Examples:
        >>> from qmcpy import DigitalNetB2, Gaussian, Mixture
        >>> components = [
        ...     Gaussian(DigitalNetB2(1, seed=11), mean=-2),
        ...     Gaussian(DigitalNetB2(1, seed=13), mean=2),
        ... ]
        >>> mixture = Mixture(DigitalNetB2(2, seed=7), components, [0.3, 0.7])
        >>> mixture(4).shape
        (4, 1)
    """

    def __init__(self, sampler, components, probabilities):
        """
        Args:
            sampler (AbstractDiscreteDistribution): Standard-uniform sampler
                whose dimension is one greater than the component dimension.
            components (list or tuple of AbstractTrueMeasure): True measures
                with a common output dimension.
            probabilities (array-like): Positive component probabilities that
                sum to one.
        """
        if not isinstance(components, (list, tuple)) or len(components) == 0:
            raise ParameterError("Mixture requires a nonempty list of components.")
        if not all(
            isinstance(component, AbstractTrueMeasure) for component in components
        ):
            raise ParameterError(
                "Each Mixture component must be an AbstractTrueMeasure instance."
            )
        if not isinstance(sampler, AbstractDiscreteDistribution):
            raise ParameterError(
                "Mixture sampler must be an AbstractDiscreteDistribution."
            )

        try:
            probabilities = np.asarray(probabilities, dtype=float)
        except (TypeError, ValueError) as error:
            raise ParameterError("Mixture probabilities must be numeric.") from error
        if probabilities.ndim != 1 or len(probabilities) != len(components):
            raise ParameterError(
                "Mixture requires exactly one probability per component."
            )
        if not np.all(np.isfinite(probabilities)) or not np.all(probabilities > 0):
            raise ParameterError("Mixture probabilities must be positive and finite.")
        if not np.isclose(probabilities.sum(), 1.0, rtol=1e-12, atol=1e-12):
            raise ParameterError("Mixture probabilities must sum to 1.")

        component_dimension = components[0].d
        if any(component.d != component_dimension for component in components[1:]):
            raise DimensionError(
                "All Mixture components must have the same output dimension."
            )
        if sampler.d != component_dimension + 1:
            raise DimensionError(
                "Mixture sampler dimension must equal the component dimension plus "
                f"one ({sampler.d} != {component_dimension + 1})."
            )

        self.parameters = ["components", "probabilities"]
        self.components = list(components)
        self.probabilities = self._read_only_array(probabilities)
        self._cumulative_probabilities = np.cumsum(self.probabilities)
        self._cumulative_probabilities[-1] = 1.0

        self.domain = np.array([[0.0, 1.0]])
        self._parse_sampler(sampler)
        self.d = component_dimension
        self.range = self._mixture_range()
        super(Mixture, self).__init__()

        self._mean_cache = None
        self._variance_cache = None
        self._standard_deviation_cache = None
        self._covariance_cache = None
        if all(hasattr(component, "mean") for component in self.components):
            self.parameters.append("mean")
        if all(
            hasattr(component, "mean") and hasattr(component, "covariance")
            for component in self.components
        ):
            self.parameters.extend(["variance", "standard_deviation", "covariance"])

    @staticmethod
    def _expanded_range(component):
        bounds = np.asarray(component.range)
        if bounds.shape == (1, 2):
            return np.tile(bounds, (component.d, 1))
        if bounds.shape == (component.d, 2):
            return bounds
        raise DimensionError(
            "Mixture component range must have shape (1, 2) or "
            f"({component.d}, 2)."
        )

    def _mixture_range(self):
        ranges = np.stack(
            [self._expanded_range(component) for component in self.components]
        )
        return np.column_stack(
            [ranges[..., 0].min(axis=0), ranges[..., 1].max(axis=0)]
        )

    def _component_vector(self, component, component_index, statistic):
        try:
            value = getattr(component, statistic)
        except AttributeError as error:
            raise AttributeError(
                f"Mixture component {component_index} "
                f"({type(component).__name__}) does not provide {statistic}."
            ) from error
        value = np.atleast_1d(np.asarray(value, dtype=float))
        if value.shape != (self.d,):
            raise DimensionError(
                f"Mixture component {component_index} "
                f"({type(component).__name__}) {statistic} must have shape "
                f"({self.d},), got {value.shape}."
            )
        return value

    @property
    def mean(self):
        if self._mean_cache is None:
            component_means = np.stack(
                [
                    self._component_vector(component, index, "mean")
                    for index, component in enumerate(self.components)
                ]
            )
            mean = np.sum(self.probabilities[:, None] * component_means, axis=0)
            mean = self._read_only_array(mean)
            self._mean_cache = self._scalar_if_univariate(mean)
        return self._mean_cache

    @property
    def covariance(self):
        if self._covariance_cache is None:
            mixture_mean = np.atleast_1d(np.asarray(self.mean))
            covariance = np.zeros((self.d, self.d), dtype=float)
            for index, (probability, component) in enumerate(
                zip(self.probabilities, self.components)
            ):
                component_mean = self._component_vector(component, index, "mean")
                try:
                    component_covariance = component.covariance
                except AttributeError as error:
                    raise AttributeError(
                        f"Mixture component {index} "
                        f"({type(component).__name__}) does not provide covariance."
                    ) from error
                if sparse.issparse(component_covariance):
                    component_covariance = component_covariance.toarray()
                component_covariance = np.atleast_2d(
                    np.asarray(component_covariance, dtype=float)
                )
                expected_shape = (self.d, self.d)
                if component_covariance.shape != expected_shape:
                    raise DimensionError(
                        f"Mixture component {index} "
                        f"({type(component).__name__}) covariance must have shape "
                        f"{expected_shape}, got {component_covariance.shape}."
                    )
                difference = component_mean - mixture_mean
                covariance += probability * (
                    component_covariance + np.outer(difference, difference)
                )
            self._covariance_cache = self._read_only_array(covariance)
        return self._covariance_cache

    @property
    def variance(self):
        if self._variance_cache is None:
            variance = self._read_only_array(np.diag(self.covariance))
            self._variance_cache = self._scalar_if_univariate(variance)
        return self._variance_cache

    @property
    def standard_deviation(self):
        if self._standard_deviation_cache is None:
            standard_deviation = self._read_only_array(
                np.sqrt(np.atleast_1d(self.variance))
            )
            self._standard_deviation_cache = self._scalar_if_univariate(
                standard_deviation
            )
        return self._standard_deviation_cache

    def _transform(self, x):
        x = np.asarray(x, dtype=float)
        sampler_dimension = self.d + 1
        if x.ndim == 0 or x.shape[-1] != sampler_dimension:
            received = None if x.ndim == 0 else x.shape[-1]
            raise DimensionError(
                f"Mixture expected last axis {sampler_dimension}, got {received}."
            )

        leading_shape = x.shape[:-1]
        flat_x = x.reshape(-1, sampler_dimension)
        selections = np.searchsorted(
            self._cumulative_probabilities, flat_x[:, 0], side="right"
        )
        selections = np.minimum(selections, len(self.components) - 1)
        transformed = np.empty((len(flat_x), self.d), dtype=float)

        for component_index, component in enumerate(self.components):
            selected = selections == component_index
            if np.any(selected):
                transformed[selected] = component._jacobian_transform_r(
                    flat_x[selected, 1:], return_weights=False
                )

        return transformed.reshape(*leading_shape, self.d)

    def _weight(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 0 or x.shape[-1] != self.d:
            received = None if x.ndim == 0 else x.shape[-1]
            raise DimensionError(
                f"Mixture expected last axis {self.d}, got {received}."
            )

        flat_x = x.reshape(-1, self.d)
        weight = np.zeros(len(flat_x), dtype=float)
        for probability, component in zip(self.probabilities, self.components):
            component_range = self._expanded_range(component)
            in_support = np.all(
                (flat_x >= component_range[:, 0])
                & (flat_x <= component_range[:, 1]),
                axis=-1,
            )
            if np.any(in_support):
                weight[in_support] += probability * component._weight(
                    flat_x[in_support]
                )
        return weight.reshape(x.shape[:-1])

    def spawn(self, s=1, dimensions=None):
        """Spawn mixtures with new outer samplers and the same components.

        Mixture components have fixed output dimensions, so only the current
        output dimension is supported. The spawned outer samplers retain the
        required extra selector coordinate.
        """
        s = int(s)
        if s <= 0:
            raise ParameterError("Must spawn s>0 instances")
        if dimensions is None:
            output_dimensions = np.tile(self.d, s)
        elif isinstance(dimensions, (list, tuple, np.ndarray)):
            output_dimensions = np.array(dimensions, dtype=int)
        else:
            output_dimensions = np.tile(dimensions, s)
        if not (output_dimensions.ndim == 1 and len(output_dimensions) == s):
            raise ParameterError("dimensions must be a length s np.ndarray")
        if np.any(output_dimensions != self.d):
            raise DimensionError(
                "Mixture spawning currently preserves the component dimension."
            )

        sampler_spawns = self.discrete_distrib.spawn(
            s=s, dimensions=np.tile(self.d + 1, s)
        )
        return [self._spawn(sampler, sampler.d) for sampler in sampler_spawns]

    def _spawn(self, sampler, dimension):
        if dimension != self.d + 1:
            raise DimensionError(
                "Mixture spawning currently preserves the component dimension."
            )
        return Mixture(sampler, self.components, self.probabilities)
