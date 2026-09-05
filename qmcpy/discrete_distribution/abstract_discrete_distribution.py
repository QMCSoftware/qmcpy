from ..util import (
    ParameterError,
    MethodImplementationError,
    _univ_repr,
)
import numpy as np


class AbstractDiscreteDistribution(object):

    def __init__(self, dimension, replications, seed, d_limit, n_limit):
        self.mimics = "StdUniform"
        if not hasattr(self, "parameters"):
            self.parameters = []
        self.d_limit = d_limit
        self.n_limit = n_limit
        if not (
            np.isscalar(self.d_limit)
            and self.d_limit > 0
            and np.isscalar(self.n_limit)
            and self.n_limit > 0
        ):
            raise ParameterError("d_limit and n_limit must be greater than 0")
        self.d_limit = np.inf if self.d_limit == np.inf else int(self.d_limit)
        self.n_limit = np.inf if self.n_limit == np.inf else int(self.n_limit)
        self.no_replications = replications is None
        self.replications = 1 if self.no_replications else int(replications)
        if self.replications < 0:
            raise ParameterError("replications must be None or a postive int")
        if (
            isinstance(dimension, list)
            or isinstance(dimension, tuple)
            or isinstance(dimension, np.ndarray)
        ):
            self.dvec = np.array(dimension, dtype=int)
            self.d = len(self.dvec)
            if not (self.dvec.ndim == 1 and len(np.unique(self.dvec)) == self.d):
                raise ParameterError("dimension must be a 1d array of unique values")
        else:
            self.d = int(dimension)
            self.dvec = np.arange(self.d)
        if any(self.dvec > self.d_limit):
            raise ParameterError(
                "dimension greater than dimension limit %d" % self.d_limit
            )
        self._base_seed = (
            seed
            if isinstance(seed, np.random.SeedSequence)
            else np.random.SeedSequence(seed)
        )
        self.entropy = self._base_seed.entropy
        self.spawn_key = self._base_seed.spawn_key
        self.rng = np.random.Generator(np.random.SFC64(self._base_seed))

    def __call__(self, n=None, n_min=None, n_max=None, return_binary=False, warn=True):
        r"""
        - If just `n` is supplied, generate samples from the sequence at indices 0,...,`n`-1.
        - If `n_min` and `n_max` are supplied, generate samples from the sequence at indices `n_min`,...,`n_max`-1.
        - If `n` and `n_min` are supplied, then generate samples from the sequence at indices `n`,...,`n_min`-1.

        Args:
            n (Union[None, int]): Number of points to generate.
            n_min (Union[None, int]): Starting index of sequence.
            n_max (Union[None, int]): Final index of sequence.
            return_binary (bool): Only used for `DigitalNetB2`. If `True`,
                *only* return the integer representation `x_integer` of base 2
                digital net.
            warn (bool): If `False`, disable warnings when generating samples.

        Returns:
            Samples from the sequence.

            - If `replications` is `None` then this will be of size (`n_max`-`n_min`) $\times$ `dimension`
            - If `replications` is a positive int, then `x` will be of size `replications` $\times$ (`n_max`-`n_min`) $\times$ `dimension`

            Note that if `return_binary=True` then `x` is returned where `x`
            are integer representations of the digital net points.
        """
        return self.gen_samples(
            n=n, n_min=n_min, n_max=n_max, return_binary=return_binary, warn=warn
        )

    def gen_samples(
        self, n=None, n_min=None, n_max=None, return_binary=False, warn=True
    ):
        if n is not None and n_min is None and n_max is None:
            n_min = 0
            n_max = int(n)
        elif n is None and n_min is not None and n_max is not None:
            n_min = int(n_min)
            n_max = int(n_max)
        elif n is not None and n_min is not None and n_max is None:
            n_max = int(n_min)
            n_min = int(n)
        else:
            raise ParameterError("Please provide either n or (n_min,n_max)")
        if not (0 <= n_min and n_min <= n_max and n_max <= self.n_limit):
            raise ParameterError(
                "require 0 <= n_min (%d) <= n_max (%d) <= n_limit (%d)"
                % (n_min, n_max, self.n_limit)
            )
        x = self._gen_samples(
            n_min=n_min, n_max=n_max, return_binary=return_binary, warn=warn
        )
        if isinstance(x, np.ndarray):
            if not x.shape == (self.replications, n_max - n_min, self.d):
                raise ParameterError(
                    "x returned by _gen_samples must have shape self.replications (%d) x n (%d) x d (%d) but got shape %s"
                    % (self.replications, n_max - n_min, self.d, str(x.shape))
                )
            return x[0] if self.no_replications else x
        if not isinstance(x, tuple):
            raise ParameterError(
                "returned value from _gen_samples must be a np.ndarray or tuple of them"
            )
        for xi in x:
            if not xi.shape == (self.replications, n_max - n_min, self.d):
                raise ParameterError(
                    "xs returned by _gen_samples must each have shape self.replications (%d) x n (%d) x d (%d) but got shape %s"
                    % (self.replications, n_max - n_min, self.d, str(x.shape))
                )
        return (xi[0] if self.no_replications else xi for xi in x)

    def _gen_samples(self, *args, **kwargs):
        raise MethodImplementationError(self, "_gen_samples")

    def spawn(self, s=1, dimensions=None):
        r"""Spawn new instances of the current discrete distribution but with
        new seeds and dimensions. Used by multi-level QMC algorithms which
        require different seeds and dimensions on each level.

        Notes:
            Use `replications` instead of using `spawn` when possible, e.g.,
            when spawning copies which all have the same dimension.

        Args:
            s (int): Number of copies to spawn
            dimensions (np.ndarray): Length `s` array of dimensions for each
                copy. Defaults to the current dimension.

        Returns:
            Discrete distributions with new seeds and dimensions.
        """
        s = int(s)
        if s <= 0:
            raise ParameterError("Must spawn s>0 instances")
        if dimensions is None:
            dimensions = np.tile(self.d, s)
        elif (
            isinstance(dimensions, list)
            or isinstance(dimensions, tuple)
            or isinstance(dimensions, np.ndarray)
        ):
            dimensions = np.array(dimensions, dtype=int)
        else:
            dimensions = np.tile(dimensions, s)
        if not (dimensions.ndim == 1 and len(dimensions) == s):
            raise ParameterError("dimensions must be a length s np.ndarray")
        child_seeds = self._base_seed.spawn(s)
        spawned_discrete_distribs = [
            self._spawn(child_seeds[i], int(dimensions[i])) for i in range(s)
        ]
        return spawned_discrete_distribs

    def _spawn(self, child_seed, dimension):
        raise MethodImplementationError(self, "_spawn")

    def pdf(self, x):
        return np.ones_like(x[..., 0])

    def __repr__(self, abc_class_name):
        return _univ_repr(
            self, abc_class_name, ["d", "replications"] + self.parameters + ["entropy"]
        )


class AbstractLDDiscreteDistribution(AbstractDiscreteDistribution):
    """Low discrepancy sequence. Alias for `AbstractDiscreteDistribution`
    used for compatibility checks.
    """

    def __repr__(self):
        return super().__repr__("AbstractLDDiscreteDistribution")


class AbstractIIDDiscreteDistribution(AbstractDiscreteDistribution):
    """IID sequence. Alias for `AbstractDiscreteDistribution` used for
    compatibility checks.
    """

    def __repr__(self):
        return super().__repr__("AbstractIIDDiscreteDistribution")
