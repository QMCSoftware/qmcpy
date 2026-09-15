from .abstract_true_measure import AbstractTrueMeasure
from ..util import DimensionError, ParameterError


class ImportanceSampling(AbstractTrueMeasure):
    r"""
    Explicit importance sampling with a target measure and proposal measure.

    The ``target`` defines the weight in the desired integral, while the
    ``proposal`` generates the samples. The target support must be contained
    in the proposal support according to the measures' available ``range``
    metadata.

    ``ImportanceSampling`` is terminal: it cannot be used as the sampler for
    an ordinary ``TrueMeasure``, or nested as the target or proposal of another
    ``ImportanceSampling`` object. Calling ``gen_samples(return_weights=True)``
    returns proposal samples and their importance weights, not merely proposal
    Jacobian weights.

    Examples:
        >>> import numpy as np
        >>> from qmcpy import DigitalNetB2, ImportanceSampling, Uniform
        >>> proposal = Uniform(DigitalNetB2(1,seed=7))
        >>> target = Uniform(
        ...     proposal.discrete_distrib,
        ...     lower_bound=0.25,
        ...     upper_bound=0.75,
        ... )
        >>> importance_sampling = ImportanceSampling(
        ...     target=target,
        ...     proposal=proposal,
        ... )
        >>> samples, weights = importance_sampling.gen_samples(
        ...     4,
        ...     return_weights=True,
        ... )
        >>> samples.shape, weights.shape
        ((4, 1), (4,))
        >>> bool(np.isfinite(samples).all() and np.isfinite(weights).all())
        True
    """

    _is_importance_sampling = True

    def __init__(self, target, proposal):
        r"""
        Args:
            target (AbstractTrueMeasure): Measure whose weight defines the target integral.
            proposal (AbstractTrueMeasure): Measure used to generate samples.

        Raises:
            ParameterError: If `target` or `proposal` is not an `AbstractTrueMeasure`, if either is itself an `ImportanceSampling` object, or if the target range is not contained within the proposal range.
            DimensionError: If `target` and `proposal` have different dimensions.
        """
        if not isinstance(target, AbstractTrueMeasure):
            raise ParameterError("target must be an AbstractTrueMeasure instance")
        if not isinstance(proposal, AbstractTrueMeasure):
            raise ParameterError("proposal must be an AbstractTrueMeasure instance")
        if getattr(target, "_is_importance_sampling", False):
            raise ParameterError(
                "ImportanceSampling cannot be the target of another ImportanceSampling."
            )
        if getattr(proposal, "_is_importance_sampling", False):
            raise ParameterError(
                "ImportanceSampling cannot be the proposal of another ImportanceSampling."
            )
        if target.d != proposal.d:
            raise DimensionError("target and proposal must have matching dimensions")
        if not self._range_in_domain(target.range, proposal.range):
            raise ParameterError(
                "target range must be contained within proposal range for importance sampling"
            )

        self.parameters = ["target", "proposal"]
        self.target = target
        self.proposal = proposal
        self.d = proposal.d
        self.discrete_distrib = proposal.discrete_distrib
        self.transform = self
        self.sub_compatibility_error = False
        self.domain = proposal.domain
        self.range = proposal.range
        super(ImportanceSampling, self).__init__()

    def _importance_sampling_transform_r(self, x):
        """Return proposal samples and their importance weights."""
        if x.ndim < 1 or x.shape[-1] != self.d:
            raise DimensionError(
                "importance-sampling inputs must have shape (*batch_shape, d)"
            )
        batch_shape = x.shape[:-1]
        pdf = self.discrete_distrib.pdf(x)
        assert pdf.shape == batch_shape
        proposal_samples, proposal_jacobians = (
            self.proposal._jacobian_transform_r(
                x,
                return_weights=True,
            )
        )
        assert proposal_samples.shape == x.shape
        assert proposal_jacobians.shape == batch_shape
        target_weights = self.target._weight(proposal_samples)
        assert target_weights.shape == batch_shape
        importance_weights = target_weights * proposal_jacobians / pdf
        assert importance_weights.shape == batch_shape
        return proposal_samples, importance_weights

    def _jacobian_transform_r(self, x, return_weights):
        return self.proposal._jacobian_transform_r(
            x=x,
            return_weights=return_weights,
        )

    def gen_samples(
        self, n=None, n_min=None, n_max=None, return_weights=False, warn=True
    ):
        r"""
        Generate proposal samples, optionally with importance weights.

        Args:
            n (Union[None, int]): Number of points to generate.
            n_min (Union[None, int]): Starting index of the sequence.
            n_max (Union[None, int]): Final index of the sequence.
            return_weights (bool): If `True`, return the importance weights with the proposal samples. Defaults to `False`.
            warn (bool): If `False`, disable warnings when generating samples.

        Returns:
            samples (np.ndarray): Samples generated through the proposal measure.
            importance_weights (np.ndarray): Returned only when `return_weights=True`.
        """
        x = self.discrete_distrib(n=n, n_min=n_min, n_max=n_max, warn=warn)
        assert isinstance(return_weights, bool)
        if return_weights:
            return self._importance_sampling_transform_r(x)
        return self.proposal._jacobian_transform_r(
            x=x,
            return_weights=False,
        )

    def spawn(self, s=1, dimensions=None):
        proposal_spawns = self.proposal.spawn(s=s, dimensions=dimensions)
        target_spawns = self.target.spawn(s=s, dimensions=dimensions)
        return [
            ImportanceSampling(
                target=target_spawn,
                proposal=proposal_spawn,
            )
            for target_spawn, proposal_spawn in zip(
                target_spawns,
                proposal_spawns,
            )
        ]
