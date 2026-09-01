from .abstract_true_measure import AbstractTrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2
import numpy as np
from numpy.linalg import cholesky, slogdet
from scipy.stats import norm, multivariate_normal
from scipy.linalg import eigh
from typing import Union


class Gaussian(AbstractTrueMeasure):
    """
    Gaussian (Normal) distribution as described in [https://en.wikipedia.org/wiki/Multivariate_normal_distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution).

    Note:
        - `Normal` is an alias for `Gaussian`

    Examples:
        >>> true_measure = Gaussian(DigitalNetB2(2,seed=7),mean=[1,2],covariance=[[9,4],[4,5]])
        >>> true_measure(4)
        array([[ 3.83994612,  1.19097885],
               [-1.9727727 ,  0.49405353],
               [ 5.87242307,  8.41341485],
               [ 0.61222205,  1.48402653]])
        >>> true_measure
        Gaussian (AbstractTrueMeasure)
            mean            [1. 2.]
            variance        [9. 5.]
            standard_deviation [3.    2.236]
            covariance      [[9. 4.]
                             [4. 5.]]
            decomp_type     PCA

        With independent replications

        >>> x = Gaussian(DigitalNetB2(3,seed=7,replications=2),mean=0,covariance=3)(4)
        >>> x.shape
        (2, 4, 3)
        >>> x
        array([[[-1.18721904, -1.57108272,  1.15371635],
                [ 0.81749123,  0.72242445, -0.31025434],
                [-0.0807895 ,  1.44651585, -2.41042379],
                [ 2.38133494, -0.93225637,  1.30817519]],
        <BLANKLINE>
               [[-0.22304017,  1.86337427,  0.02386568],
                [ 0.15807672, -2.96365385, -0.73502346],
                [-1.26753687, -0.94427848, -2.57683314],
                [ 1.1844196 ,  0.44964332,  1.27760936]]])
    """

    def __init__(self, sampler, mean=0.0, covariance=1.0, decomp_type="PCA"):
        """
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]): Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            mean (Union[float, np.ndarray]): Mean vector.
            covariance (Union[float, np.ndarray]): Covariance matrix. A float or vector will be expanded into a diagonal matrix.
            decomp_type (str): Method for decomposition for covariance matrix. Options include

                - `'PCA'` for principal component analysis, or
                - `'Cholesky'` for cholesky decomposition.
        """
        self.parameters = ["mean", "variance", "standard_deviation", "covariance", "decomp_type"]
        # default to transform from standard uniform
        self.domain = np.array([[0, 1]])
        self._parse_sampler(sampler)
        self._parse_gaussian_params(mean, covariance, decomp_type)
        self.range = np.array([[-np.inf, np.inf]])
        super(Gaussian, self).__init__()
        assert self.mu.shape == (self.d,) and self.a.shape == (self.d, self.d)

    def _parse_gaussian_params(self, mean, covariance, decomp_type, lazy_decomp=False):
        self.decomp_type = decomp_type.upper()
        self.lazy_decomp = lazy_decomp

        if np.isscalar(mean):
            mean = np.tile(mean, self.d)
        if np.isscalar(covariance):
            covariance = covariance * np.eye(self.d)
        self.mu = np.array(mean)
        self.sigma = np.array(covariance)
        if self.sigma.shape == (self.d,):
            self.sigma = np.diag(self.sigma)
        self.sigma = (self.sigma + self.sigma.T) / 2
        if not (len(self.mu) == self.d and self.sigma.shape == (self.d, self.d)):
            raise DimensionError(
                """
                    mean must have length d and
                    covariance must be of shape d x d"""
            )
        variance = np.diag(self.sigma)
        self._set_moments(
            mean=self.mu.astype(float, copy=False),
            variance=variance,
            standard_deviation=np.sqrt(variance),
            covariance=self.sigma,
        )

        # Cache for lazy loading
        self._a_cache = None
        self._mvn_scipy_cache = None

        if not lazy_decomp:
            # Compute immediately (backward compatibility)
            self._compute_decomposition()
            self._setup_scipy_mvn()

    def _compute_decomposition(self):
        """Compute matrix decomposition (PCA or Cholesky). Raises ParameterError for BrownianBridge."""
        if self._a_cache is not None:
            return self._a_cache

        if self.decomp_type == "PCA":
            evals, evecs = eigh(self.sigma)  # get eigenvectors and eigenvalues for
            evecs = evecs * (
                1 - 2 * (evecs[0] < 0)
            )  # force first entries of eigenvectors to be positive
            order = np.argsort(-evals)
            self._a_cache = np.dot(evecs[:, order], np.diag(np.sqrt(evals[order])))
        elif self.decomp_type == "CHOLESKY":
            self._a_cache = cholesky(self.sigma)
        elif self.decomp_type == "BROWNIANBRIDGE":
            raise ParameterError("BrownianBridge does not use matrix decomposition")
        else:
            raise ParameterError("decomp_type should be 'PCA' or 'Cholesky'")
        return self._a_cache

    def _setup_scipy_mvn(self):
        """Setup scipy multivariate normal distribution."""
        if self._mvn_scipy_cache is None:
            self._mvn_scipy_cache = multivariate_normal(
                mean=self.mu, cov=self.sigma, allow_singular=True
            )
        return self._mvn_scipy_cache

    @property
    def a(self):
        """Lazy-loaded decomposition matrix."""
        if self._a_cache is None:
            self._compute_decomposition()
        return self._a_cache

    @a.setter
    def a(self, value):
        self._a_cache = value

    @property
    def mvn_scipy(self):
        """Lazy-loaded scipy multivariate normal."""
        if self._mvn_scipy_cache is None:
            self._setup_scipy_mvn()
        return self._mvn_scipy_cache

    @mvn_scipy.setter
    def mvn_scipy(self, value):
        self._mvn_scipy_cache = value

    def _transform(self, x):
        return self.mu + np.einsum("...ij,kj->...ik", norm.ppf(x), self.a)

    def _weight(self, t):
        return self.mvn_scipy.pdf(t)

    def _spawn(self, sampler, dimension):
        if dimension == self.d:  # don't do anything if the dimension doesn't change
            spawn = Gaussian(
                sampler,
                mean=self.mu,
                covariance=self.covariance,
                decomp_type=self.decomp_type,
            )
        else:
            m = self.mu[0]
            c = self.sigma[0, 0]
            expected_cov = c * np.eye(int(self.d))
            if not ((self.mu == m).all() and (self.sigma == expected_cov).all()):
                raise DimensionError(
                    """
                        In order to spawn a Gaussian measure
                        mean (mu) must be all the same and 
                        covariance must be a scaler times I"""
                )
            spawn = Gaussian(
                sampler, mean=m, covariance=c, decomp_type=self.decomp_type
            )
        return spawn
