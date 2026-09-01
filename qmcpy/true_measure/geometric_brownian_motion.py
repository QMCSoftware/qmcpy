from .brownian_motion import BrownianMotion
from ..discrete_distribution import DigitalNetB2
from ..util import ParameterError
from typing import Union, Tuple
from numpy import (
    exp,
    zeros,
    minimum,
    array,
    sqrt,
    log,
    pi,
    linalg,
    eye,
    cumsum,
    add,
    multiply,
    ndarray,
)
from scipy.stats import multivariate_normal, norm


class GeometricBrownianMotion(BrownianMotion):
    r"""
    A Geometric Brownian Motion (GBM) with initial value $S_0$, drift $\gamma$, and diffusion $\sigma^2$ is

    $$\mathrm{GBM}(t) = S_0 \exp[(\gamma - \sigma^2/2) t + \sigma \mathrm{BM}(t)]$$

    where BM is a Brownian Motion drift $\gamma$ and diffusion $\sigma^2$.

    Examples:
        >>> gbm = GeometricBrownianMotion(DigitalNetB2(4,seed=7), t_final=2, drift=0.1, diffusion=0.2)
        >>> gbm.gen_samples(2)
        array([[0.92343761, 1.42069027, 1.30851806, 0.99133819],
               [0.7185916 , 0.42028013, 0.42080335, 0.4696196 ]])
        >>> gbm
        GeometricBrownianMotion (AbstractTrueMeasure)
            time_vec        [0.5 1.  1.5 2. ]
            drift           0.100
            diffusion       0.200
            mean_gbm        [1.051 1.105 1.162 1.221]
            covariance_gbm  [[0.116 0.122 0.128 0.135]
                             [0.122 0.27  0.284 0.299]
                             [0.128 0.284 0.472 0.496]
                             [0.135 0.299 0.496 0.734]]
            decomp_type     PCA
    """

    def __init__(
        self,
        sampler,
        t_final=1,
        initial_value=1,
        drift=0,
        diffusion=1,
        decomp_type="PCA",
        lazy_load=True,
        lazy_decomp=True,
    ):
        r"""
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A discrete distribution or true measure.
            t_final (float): End time for the geometric Brownian motion, non-negative.
            initial_value (float): Positive initial value of the process, $S_0$.
            drift (float): Drift coefficient $\gamma$.
            diffusion (float): Positive diffusion coefficient $\sigma^2$, where $\sigma$ is volatility.
            decomp_type (str): Method of decomposition, either "PCA", "Cholesky", or "BrownianBridge".
            lazy_load (bool): If True, defer GBM-specific computations until needed.
            lazy_decomp (bool): If True, defer expensive matrix decomposition until needed.
        """
        super().__init__(
            sampler,
            t_final=t_final,
            drift=0,
            diffusion=diffusion,
            decomp_type=decomp_type,
            lazy_decomp=lazy_decomp,
        )
        self.parameters = [
            "time_vec",
            "drift",
            "diffusion",
            "mean_gbm",
            "covariance_gbm",
            "decomp_type",
        ]
        self.initial_value = initial_value
        self.drift = drift
        self.diffusion = diffusion
        self.lazy_load = lazy_load
        self.lazy_decomp = lazy_decomp

        # Cache for lazy-loaded properties
        self._mean_gbm_cache = None
        self._covariance_gbm_cache = None
        self._log_mvn_scipy_cache = None

        # Large step optimization - use fast path for large problems
        self.large_step_threshold = 1000
        self.use_large_step_optimization = (
            len(self.time_vec) > self.large_step_threshold
        )

        # Validate input early (fast operation)
        self._validate_input()

        if not lazy_load:
            # Compute everything immediately for backwards compatibility
            self.mean_gbm = self._compute_gbm_mean()
            self.covariance_gbm = self._compute_gbm_covariance()
            self._setup_lognormal_distribution()

    @property
    def mean_gbm(self):
        """Lazy-loaded GBM mean vector."""
        if self._mean_gbm_cache is None:
            self._mean_gbm_cache = self._compute_gbm_mean()
        return self._mean_gbm_cache

    @mean_gbm.setter
    def mean_gbm(self, value):
        """Allow explicit setting of mean_gbm."""
        self._mean_gbm_cache = value

    @property
    def covariance_gbm(self):
        """Lazy-loaded GBM covariance matrix."""
        if self._covariance_gbm_cache is None:
            self._covariance_gbm_cache = self._compute_gbm_covariance()
        return self._covariance_gbm_cache

    @covariance_gbm.setter
    def covariance_gbm(self, value):
        """Allow explicit setting of covariance_gbm."""
        self._covariance_gbm_cache = value

    @property
    def log_mvn_scipy(self):
        """Lazy-loaded scipy multivariate normal distribution."""
        if self._log_mvn_scipy_cache is None:
            self._setup_lognormal_distribution()
        return self._log_mvn_scipy_cache

    def _compute_gbm_mean(self):
        return self.initial_value * exp(self.drift * self.time_vec)

    def _compute_gbm_covariance(self):
        """Fast covariance computation using vectorized operations."""
        S0_sq = self.initial_value**2
        mu = self.drift
        t = array(self.time_vec)
        n = len(t)

        # Use most efficient method based on problem size
        if n <= 200:  # For small-medium matrices, broadcasting is fastest
            t_sum = t[:, None] + t[None, :]  # Shape: (n, n)
            t_min = minimum.outer(t, t)  # Shape: (n, n)
            cov_matrix = S0_sq * exp(mu * t_sum) * (exp(self.diffusion * t_min) - 1)
        else:  # For larger matrices, use memory-efficient computation
            cov_matrix = zeros((n, n))
            exp_mu_t = exp(mu * t)  # Pre-compute exp(mu * t_i)
            exp_diff_t = exp(self.diffusion * t)  # Pre-compute exp(diffusion * t_i)
            for i in range(n):  # Optimized symmetric matrix computation
                cov_matrix[i, i] = S0_sq * exp_mu_t[i] ** 2 * (exp_diff_t[i] - 1)
                for j in range(i + 1, n):
                    t_min_ij = min(t[i], t[j])
                    cov_ij = (
                        S0_sq
                        * exp_mu_t[i]
                        * exp_mu_t[j]
                        * (exp(self.diffusion * t_min_ij) - 1)
                    )
                    cov_matrix[i, j] = cov_ij
                    cov_matrix[j, i] = cov_ij  # Symmetric

        return cov_matrix

    def _transform(self, x):
        bm_samples = super()._transform(x)
        # Note: bm_samples already includes the diffusion scaling from the parent BrownianMotion class
        # So we don't multiply by self.diffusion again here
        exponent = (self.drift - 0.5 * self.diffusion) * self.time_vec + bm_samples
        samples = self.initial_value * exp(exponent)
        return samples

    def _spawn(self, sampler, dimension):
        return GeometricBrownianMotion(
            sampler,
            t_final=self.t,
            initial_value=self.initial_value,
            drift=self.drift,
            diffusion=self.diffusion,
            decomp_type=self.decomp_type,
            lazy_load=getattr(self, "lazy_load", True),  # Default to optimized mode
            lazy_decomp=getattr(self, "lazy_decomp", True),
        )

    def _validate_input(self):
        """
        Validates the input parameters of the GeometricBrownianMotion class.

        Raises:
            ValueError: If the end time `t_final' is negative.
            ValueError: If the diffusion coefficient is less than or equal to zero.
            ValueError: If the initial value is less than or equal to zero.
            ParameterError: If the decomposition type is not 'PCA', 'Cholesky', or 'BrownianBridge'.
        """
        if self.t < 0:
            raise ValueError(
                f"End time 't_final' must be non-negative. It should not be {self.t}."
            )
        if self.diffusion <= 0:
            raise ValueError(
                f"Diffusion coefficient must be positive. It should not be {self.diffusion}."
            )
        if self.initial_value <= 0:
            raise ValueError(
                f"Initial value must be positive. It should not be {self.initial_value}."
            )
        if self.decomp_type.upper() not in ["PCA", "CHOLESKY", "BROWNIANBRIDGE"]:
            raise ParameterError(
                f"Decomposition type must be 'PCA', 'Cholesky', or 'BrownianBridge'. It should not be {self.decomp_type}."
            )

    def _validate_samples(self, samples, strict=False):
        """
        Validate that generated GBM samples meet mathematical requirements.
        """
        min_val = samples.min()
        max_val = samples.max()
        num_negative = (samples <= 0).sum()
        total_samples = samples.size

        validation_results = {
            "is_valid": num_negative == 0,
            "min_value": min_val,
            "max_value": max_val,
            "num_negative": num_negative,
            "total_samples": total_samples,
            "fraction_negative": (
                num_negative / total_samples if total_samples > 0 else 0
            ),
        }

        if num_negative > 0:
            error_msg = (
                f"GeometricBrownianMotion validation FAILED: "
                f"{num_negative}/{total_samples} ({100*validation_results['fraction_negative']:.2f}%) "
                f"samples are non-positive (min: {min_val:.6f}). "
                f"This violates the mathematical definition of GBM where S(t) = S0 * exp(...) > 0."
            )

            if strict:
                raise ValueError(error_msg)
            else:
                import warnings

                warnings.warn(error_msg)

        return validation_results

    def _setup_lognormal_distribution(self):
        """Setup scipy multivariate normal for the log-transformed variables."""
        # Mean of log(S(t)/S0): (drift - 0.5*diffusion) * t
        log_mean = (self.drift - 0.5 * self.diffusion) * self.time_vec

        # Covariance of log returns: diffusion * min(t_i, t_j)
        time_matrix = minimum.outer(self.time_vec, self.time_vec)
        log_cov = self.diffusion * time_matrix

        self._log_mvn_scipy_cache = multivariate_normal(
            mean=log_mean, cov=log_cov, allow_singular=True
        )

    def _weight(self, x):
        """
        Compute PDF of multivariate log-normal distribution.
        For log-normal: f(x) = (1/∏x_i) * φ(log(x/S0)) where φ is multivariate normal PDF.

        Args:
            x (ndarray): GBM sample paths of shape (n_samples, n_timepoints)

        Returns:
            ndarray: PDF values for each sample path
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Transform to log space: log(x/S0)
        log_transformed = log(x / self.initial_value)

        # Get PDF of underlying multivariate normal
        normal_pdf = self.log_mvn_scipy.pdf(log_transformed)

        # Apply Jacobian transformation: divide by product of x values
        jacobian = 1.0 / x.prod(axis=1)

        return normal_pdf * jacobian

    def gen_samples(
        self, n=None, n_min=None, n_max=None, return_weights=False, warn=True
    ) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        """
        Generate GBM samples using the parent's transform pipeline.
        
        Args:
            n (int): number of samples to generate
            n_min (int): minimum index of sequence
            n_max (int): maximum index of sequence  
            return_weights (bool): whether to return Jacobian weights
            warn (bool): whether to warn about sample generation
            
        Returns:
            samples (Union[ndarray,tuple]): GBM samples, optionally with weights if return_weights=True
        """
        return super().gen_samples(n=n, n_min=n_min, n_max=n_max, return_weights=return_weights, warn=warn)
