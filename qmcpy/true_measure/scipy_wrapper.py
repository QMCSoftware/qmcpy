from .abstract_true_measure import AbstractTrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
from ..discrete_distribution import DigitalNetB2

import numpy as np
import scipy.stats
import warnings


def _custom_univariate_sanity_issues(dist, n_grid=64):
    issues = []

    if not hasattr(dist, "ppf"):
        return ["missing ppf()"]

    eps = np.finfo(float).eps
    u = np.linspace(eps, 1.0 - eps, n_grid)

    try:
        x = np.asarray(dist.ppf(u), dtype=float)
    except Exception as e:
        return [f"ppf() raised {type(e).__name__}: {e}"]

    if np.any(~np.isfinite(x)):
        issues.append("ppf() returned non-finite values")

    dx = np.diff(x)
    if np.any(dx < -1e-12):
        issues.append("ppf() is not nondecreasing (looks non-monotone)")

    # Light-touch pdf/logpdf checks on quantiles (works for bounded/unbounded support)
    uq = np.linspace(0.01, 0.99, 9)
    try:
        xq = np.asarray(dist.ppf(uq), dtype=float)
    except Exception:
        xq = None

    if xq is not None and hasattr(dist, "pdf"):
        try:
            p = np.asarray(dist.pdf(xq), dtype=float)
            if np.any(~np.isfinite(p)):
                issues.append("pdf() returned non-finite values on quantiles")
            if np.any(p < -1e-12):
                issues.append("pdf() returned negative values on quantiles")
        except Exception as e:
            issues.append(f"pdf() raised {type(e).__name__}: {e}")

    if xq is not None and hasattr(dist, "logpdf"):
        try:
            lp = np.asarray(dist.logpdf(xq), dtype=float)
            if np.any(~np.isfinite(lp)):
                issues.append("logpdf() returned non-finite values on quantiles")
        except Exception as e:
            issues.append(f"logpdf() raised {type(e).__name__}: {e}")

    return issues


class _MVNAdapter:
    """
    Small adapter that turns a SciPy multivariate normal like object into
    something with a simple ``transform(u)`` interface.

    Idea:
      1. Start from u in (0,1)^d.
      2. Map to standard normals z via ``norm.ppf``.
      3. Apply Cholesky to inject the correlation structure.
    """

    def __init__(self, mvn_like):
        # Keep the original object around so that we can still call logpdf.
        self._mvn = mvn_like

        # SciPy's multivariate_normal has attributes ``mean`` and ``cov``.
        mean = np.asarray(mvn_like.mean)
        cov = np.asarray(mvn_like.cov)

        if mean.ndim != 1:
            raise ParameterError(
                "SciPyWrapper currently expects a 1D mean vector for multivariate_normal."
            )
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ParameterError(
                "SciPyWrapper expects a square covariance matrix for multivariate_normal."
            )
        if cov.shape[0] != mean.size:
            raise DimensionError(
                f"Mean dimension {mean.size} and cov shape {cov.shape} do not match."
            )

        self.dim = mean.size
        self._mean = mean
        # Cholesky factor gives L such that cov = L L^T.
        self._chol = np.linalg.cholesky(cov)

    def transform(self, u):
        """
        Take u in (0,1)^d and turn it into correlated normal samples.
        """
        u = np.asarray(u, dtype=float)
        if u.shape[-1] != self.dim:
            raise DimensionError(
                f"MVNAdapter expected last axis {self.dim}, got {u.shape[-1]}"
            )

        # Clip so we never hit exactly 0 or 1 inside norm.ppf.
        eps = np.finfo(float).eps
        u_clip = np.clip(u, eps, 1.0 - eps)

        # Map to i.i.d. standard normals.
        z = scipy.stats.norm.ppf(u_clip)

        # Flatten, apply Cholesky, then reshape back to original shape.
        z_flat = z.reshape(-1, self.dim)
        x_flat = z_flat @ self._chol.T + self._mean
        return x_flat.reshape(z.shape)

    def logpdf(self, x):
        """
        Forward to the SciPy logpdf, keeping shapes tidy.
        """
        x = np.asarray(x, dtype=float)
        if x.shape[-1] != self.dim:
            raise DimensionError(
                f"MVNAdapter expected last axis {self.dim}, got {x.shape[-1]}"
            )

        x_flat = x.reshape(-1, self.dim)
        logp = self._mvn.logpdf(x_flat)
        return np.asarray(logp).reshape(x.shape[:-1])


class SciPyWrapper(AbstractTrueMeasure):
    r"""
    True measure that wraps SciPy style distributions.

    This class keeps the original behavior of SciPyWrapper with
    independent 1D marginals and adds an optional "joint" mode for
    dependent distributions.

    Examples:
        Independent marginals from ``scipy.stats``:

        >>> import scipy.stats as stats
        >>> tm = SciPyWrapper(
        ...     sampler=DigitalNetB2(3, seed=7),
        ...     scipy_distribs=[
        ...         stats.uniform(loc=1, scale=2),
        ...         stats.norm(loc=0, scale=1),
        ...         stats.gamma(a=5, loc=0, scale=2)])
        >>> x = tm(2)
        >>> x.shape
        (2, 3)

        Joint multivariate normal passed as a single object:

        >>> mvn = stats.multivariate_normal(
        ...     mean=[0.0, 0.0],
        ...     cov=[[1.0, 0.8], [0.8, 1.0]])
        >>> tm_joint = SciPyWrapper(DigitalNetB2(2, seed=7), mvn)
        >>> tm_joint(2).shape
        (2, 2)


        2D Student t distribution (independent marginals):

        >>> df = 5
        >>> true_measure = SciPyWrapper(
        ...     sampler=DigitalNetB2(2, seed=13),
        ...     scipy_distribs=[
        ...         stats.t(df=df, loc=0.0, scale=1.0),
        ...         stats.t(df=df, loc=1.0, scale=2.0),
        ...     ],
        ... )
        >>> xs = true_measure(4)
        >>> xs.shape
        (4, 2)
    """

    def __init__(self, sampler, scipy_distribs):
        """
        Parameters
        ----------
        sampler : AbstractDiscreteDistribution
            Low discrepancy or iid sampler in dimension d, living on [0,1)^d.
        scipy_distribs :
            One of the following:

            - A single SciPy 1D continuous frozen distribution.
            - A list of such frozen distributions (independent marginals).
            - A custom 1D distribution object with ``ppf`` and ``pdf`` or
              ``logpdf`` methods.
            - A joint object with:
                * ``transform(u)`` method
                * optional ``logpdf(x)`` method
                * ``dim`` or ``dimension`` attribute (otherwise ``sampler.d``).
        """
        self.domain = np.array([[0.0, 1.0]])

        if not isinstance(sampler, AbstractDiscreteDistribution):
            if not (
                hasattr(sampler, "d")
                and hasattr(sampler, "gen_samples")
                and callable(sampler.gen_samples)
            ):
                raise ParameterError(
                    "SciPyWrapper requires sampler be an AbstractDiscreteDistribution."
                )
        self._parse_sampler(sampler)

        # Remember what the user originally passed in so that _spawn can reuse it.
        self._user_distrib_arg = scipy_distribs

        # Flags and holders for the two modes.
        self._is_joint = False
        self._joint = None
        self._joint_has_logpdf = False
        self._warned_missing_pdf = False

        if self._looks_like_joint(scipy_distribs):
            # Configure joint mode.
            self._setup_joint(scipy_distribs)
        else:
            # Configure independent marginals mode.
            self._setup_marginals(scipy_distribs)

        super(SciPyWrapper, self).__init__()

    # ------------------------------------------------------------------
    # Decide joint vs marginal mode
    # ------------------------------------------------------------------

    def _looks_like_joint(self, obj):
        """
        Heuristic check to decide if the user passed a joint distribution.

        We treat it as "joint" if:
          - it already has a ``transform(u)`` method, or
          - it looks like a SciPy multivariate_normal style object:
            has ``mean``, ``cov``, and ``logpdf`` attributes and no ``ppf``.
        """
        if hasattr(obj, "transform") and callable(obj.transform):
            return True

        looks_mvn_like = (
            hasattr(obj, "mean")
            and hasattr(obj, "cov")
            and hasattr(obj, "logpdf")
            and not hasattr(obj, "ppf")
        )
        if looks_mvn_like:
            return True

        return False

    def _setup_joint(self, joint_obj):
        """
        Configure the wrapper in "joint" mode.

        Either:
          - wrap a SciPy style multivariate normal in _MVNAdapter, or
          - use the object directly if it already has ``transform(u)``.
        """
        joint = joint_obj

        # If there is no transform but it looks like an MVN, wrap it.
        mvn_like = (
            hasattr(joint_obj, "mean")
            and hasattr(joint_obj, "cov")
            and hasattr(joint_obj, "logpdf")
            and not hasattr(joint_obj, "transform")
        )
        if mvn_like:
            joint = _MVNAdapter(joint_obj)

        if not hasattr(joint, "transform"):
            raise ParameterError(
                "Joint distribution must implement a 'transform(u)' method."
            )

        # Try to read dimension from the object, otherwise fall back to sampler.d.
        dim = getattr(joint, "dim", None)
        if dim is None:
            dim = getattr(joint, "dimension", None)
        if dim is None:
            dim = self.d

        if dim != self.d:
            raise DimensionError(
                f"Joint distribution dimension {dim} does not match sampler.d {self.d}."
            )

        self._joint = joint
        self._is_joint = True
        self._joint_has_logpdf = hasattr(joint, "logpdf")

        if not self._joint_has_logpdf:
            warnings.warn(
                "SciPyWrapper joint distribution has no 'logpdf'. "
                "Weights will be treated as 1.",
                UserWarning,
            )

        # We do not know a finite support here, so we just say R^d.
        self.range = np.tile(np.array([-np.inf, np.inf]), (self.d, 1))

    def _setup_marginals(self, scipy_distribs):
        """
        Configure the wrapper in "independent marginals" mode.

        We accept a single frozen dist or a list, and we also allow
        user defined 1D distributions that have the right methods.
        """
        rv_cont = scipy.stats._distn_infrastructure.rv_continuous_frozen

        # Normalise to a list.
        if isinstance(scipy_distribs, rv_cont) or hasattr(scipy_distribs, "ppf"):
            marginals = [scipy_distribs]
        else:
            marginals = list(scipy_distribs)

        if len(marginals) == 0:
            raise ParameterError("scipy_distribs must contain at least one marginal.")

        checked = []
        for sd in marginals:
            if isinstance(sd, rv_cont):
                # Native SciPy frozen distribution.
                checked.append(sd)
                continue

            # Custom 1D distribution.
            if not hasattr(sd, "ppf"):
                raise ParameterError(
                    "Custom univariate distributions must implement a 'ppf' method."
                )

            issues = _custom_univariate_sanity_issues(sd)
            if issues:
                warnings.warn(
                    "SciPyWrapper received a custom univariate distribution (not a scipy.stats frozen distribution) "
                    "and it failed sanity checks:\n  - " + "\n  - ".join(issues),
                    UserWarning,
                    stacklevel=2,
                )

            checked.append(sd)

        # Broadcast a single marginal across all dimensions, like the
        # original SciPyWrapper did.
        if len(checked) == 1:
            self.sds = checked * self.d
        else:
            if len(checked) != self.d:
                raise DimensionError(
                    "Length of scipy_distribs must match the dimension of the sampler."
                )
            self.sds = checked

        # Build an approximate range for each marginal.
        ranges = []
        for sd in self.sds:
            if isinstance(sd, rv_cont):
                ranges.append(sd.interval(1.0))
            else:
                # Use extreme quantiles as a loose bounding box.
                lo = float(sd.ppf(1e-10))
                hi = float(sd.ppf(1.0 - 1e-10))
                ranges.append((lo, hi))
        self.range = np.asarray(ranges)
        self._is_joint = False

        assert len(self.sds) == self.d

    def _sanity_check_univariate(self, dist):
        """
        Light sanity check for a custom 1D distribution.

        The goal is not to be perfect, just to catch obvious mistakes and
        warn the user. We never raise here, only emit warnings.

        We check on a grid 0.01..0.99 that:
          - ppf is finite and roughly increasing,
          - pdf/logpdf is finite and non negative,
          - the approximate integral of the pdf is close to 1.
        """
        try:
            # Grid of probabilities away from the hard edges.
            u_grid = np.linspace(0.01, 0.99, 21)
            x_grid = dist.ppf(u_grid)

            if not np.all(np.isfinite(x_grid)):
                warnings.warn(
                    "Custom distribution ppf returned non finite values on 0.01..0.99.",
                    UserWarning,
                )

            if np.any(np.diff(x_grid) <= 0):
                warnings.warn(
                    "Custom distribution ppf appears non increasing on 0.01..0.99.",
                    UserWarning,
                )

            # Work out density values on the same grid.
            if hasattr(dist, "pdf"):
                dens = np.asarray(dist.pdf(x_grid), dtype=float)
            elif hasattr(dist, "logpdf"):
                dens = np.exp(dist.logpdf(x_grid))
            else:
                # No density information to check.
                return

            if not np.all(np.isfinite(dens)):
                warnings.warn(
                    "Custom distribution pdf/logpdf returned non finite values.",
                    UserWarning,
                )

            if np.any(dens < 0):
                warnings.warn(
                    "Custom distribution pdf/logpdf took negative values.",
                    UserWarning,
                )

            # Very rough normalisation check based on trapezoidal rule.
            mass_est = np.trapz(dens, x_grid)
            if not np.isfinite(mass_est) or abs(mass_est - 1.0) > 0.1:
                warnings.warn(
                    f"Custom distribution pdf looks poorly normalised: "
                    f"integral ≈ {mass_est:.3f} on 0.01..0.99.",
                    UserWarning,
                )
        except Exception as err:
            warnings.warn(
                f"Could not perform sanity check on custom distribution: {err}",
                UserWarning,
            )

    # ------------------------------------------------------------------
    # Core AbstractTrueMeasure interface
    # ------------------------------------------------------------------

    def _transform(self, x):
        """
        Map unit cube samples to the physical space.

        For joint mode we delegate to the joint object.
        For marginal mode we call ``ppf`` dimension wise.
        """
        x = np.asarray(x, dtype=float)

        if self._is_joint:
            return self._joint.transform(x)

        t = np.empty_like(x, dtype=float)
        for j in range(self.d):
            t[..., j] = self.sds[j].ppf(x[..., j])
        return t

    def _weight(self, x):
        """
        Compute unnormalised density weights.

        - For joint distributions with logpdf we simply exp(logpdf).
        - For joint distributions with no density we return 1.
        - For independent marginals we multiply the marginal densities.
        """
        x = np.asarray(x, dtype=float)

        if self._is_joint:
            if self._joint_has_logpdf:
                logp = self._joint.logpdf(x)
                return np.exp(logp)
            # No density information available, just return ones.
            return np.ones(x.shape[:-1], dtype=float)

        rv_cont = scipy.stats._distn_infrastructure.rv_continuous_frozen

        rho = np.ones(x.shape[:-1], dtype=float)
        for j, sd in enumerate(self.sds):
            if isinstance(sd, rv_cont):
                rho *= sd.pdf(x[..., j])
            elif hasattr(sd, "pdf"):
                rho *= sd.pdf(x[..., j])
            elif hasattr(sd, "logpdf"):
                rho *= np.exp(sd.logpdf(x[..., j]))
            else:
                if not self._warned_missing_pdf:
                    warnings.warn(
                        "Custom univariate distribution has no 'pdf' or 'logpdf'. "
                        "Weights will be treated as 1 for this marginal.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._warned_missing_pdf = True
                # rho stays unchanged for this marginal.

        return rho

    def _spawn(self, sampler, dimension):
        """
        Create a child true measure that shares the same distribution
        configuration but uses a new sampler.

        We simply reuse the original ``scipy_distribs`` argument so the
        behavior matches the parent.
        """
        return SciPyWrapper(sampler, self._user_distrib_arg)
