import numpy as np

from .abstract_true_measure import AbstractTrueMeasure
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
from ..util import DimensionError, ParameterError


class Mixture(AbstractTrueMeasure):
    r"""Mixture of true measures with fixed component probabilities.

    A sample ``u`` has one more coordinate than the mixture output. The first
    coordinate selects a component according to ``probabilities``; the
    remaining coordinates are transformed by that component.

    The samplers attached to the component true measures are not sampled.
    Components provide their full recursive transform and weight behavior.
    For an importance-sampling composition, the caller remains responsible for
    ensuring that the component's induced sampling distribution is appropriate
    for the intended mixture.

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
            self._cumulative_probabilities, flat_x[:, 0], side="left"
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

        weight = np.zeros(x.shape[:-1], dtype=float)
        for probability, component in zip(self.probabilities, self.components):
            weight += probability * component._weight(x)
        return weight

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
