from .gaussian import Gaussian
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
from .abstract_true_measure import AbstractTrueMeasure
from ..discrete_distribution import DigitalNetB2
from ..util import DimensionError, ParameterError
import numpy as np
import warnings
from scipy.special import kv, gamma
from typing import Union


class MaternGP(Gaussian):
    r"""
    A Gaussian process with Matérn covariance kernel.

    Examples:
        >>> true_measure = MaternGP(DigitalNetB2(dimension=3,seed=7),points=np.linspace(0,1,3)[:,None],nu=3/2,length_scale=[3,4,5],variance=0.01,mean=np.array([.3,.4,.5]))
        >>> true_measure(4)
        array([[0.3515401 , 0.43083384, 0.51801119],
               [0.20272448, 0.31312011, 0.4241431 ],
               [0.40189226, 0.53502934, 0.63826677],
               [0.29943567, 0.38491661, 0.48296594]])
        >>> true_measure
        MaternGP (AbstractTrueMeasure)
            mean            [0.3 0.4 0.5]
            variance        [0.01 0.01 0.01]
            kernel_variance 0.010
            standard_deviation [0.1 0.1 0.1]
            covariance      [[0.01  0.01  0.009]
                             [0.01  0.01  0.01 ]
                             [0.009 0.01  0.01 ]]
            decomp_type     PCA

        The inherited `variance` attribute is the vector of marginal variances
        (the diagonal of `covariance`); use `kernel_variance` to recover the
        scalar global scaling factor supplied to the constructor.

        >>> true_measure.kernel_variance
        0.01
        >>> true_measure.variance
        array([0.010001, 0.010001, 0.010001])

        With independent replications

        >>> x = MaternGP(DigitalNetB2(dimension=3,seed=7,replications=2),points=np.linspace(0,1,3)[:,None],nu=3/2,length_scale=[3,4,5],variance=0.01,mean=np.array([.3,.4,.5]))(4)
        >>> x.shape
        (2, 4, 3)
        >>> x
        array([[[0.21490091, 0.33078241, 0.45151042],
                [0.35465127, 0.44705898, 0.53793358],
                [0.31091595, 0.39868187, 0.47660193],
                [0.42419919, 0.53572415, 0.64674883]],
        <BLANKLINE>
               [[0.31010701, 0.38522001, 0.46670381],
                [0.27221177, 0.413546  , 0.54101758],
                [0.2147053 , 0.33293508, 0.43572791],
                [0.37343973, 0.46534628, 0.56356714]]])

    **References:**

    1.  [`sklearn.gaussian_process.kernels.Matern`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html).

    2.  [https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function](https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function).
    """

    def __init__(
        self,
        sampler,
        points,
        length_scale=1.0,
        nu=1.5,
        variance=1.0,
        mean=0.0,
        nugget=1e-6,
        decomp_type="PCA",
    ):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]): Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            points (np.ndarray): The positions of points on a metric space. The array should have shape $(d,k)$ where $d$ is the dimension of the sampler and $k$ is the latent dimension.
            nu (float): The "smoothness" of the MaternGP function, e.g.,

                - $\nu = 1/2$ is equivalent to the absolute exponential kernel,
                - $\nu = 3/2$ implies a once-differentiable function,
                - $\nu = 5/2$ implies twice differentiability.
                - as $\nu \to \infty$ the kernel becomes equivalent to the RBF kernel, see [`sklearn.gaussian_process.kernels.RBF`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html#sklearn.gaussian_process.kernels.RBF).

                Note that when $\nu \notin \{1/2, 3/2, 5/2, \infty \}$ the kernel is around $10$ times slower to evaluate.
            length_scale (Union[float, np.ndarray]): Determines "peakiness", or how correlated two points are based on their distance.
            variance (float): Global scaling factor of the kernel. Retrievable
                after construction via the `kernel_variance` property. (The
                inherited `variance` attribute is the vector of marginal
                variances, i.e. the diagonal of `covariance`.)
            mean (Union[float, np.ndarray]): Mean vector for multivariate `Gaussian`.
            nugget (float): Positive nugget to add to diagonal.
            decomp_type (str): Method for decomposition for covariance matrix. Options include

                - `'PCA'` for principal component analysis, or
                - `'Cholesky'` for cholesky decomposition.
        """
        if not (
            isinstance(sampler, AbstractDiscreteDistribution)
            or isinstance(sampler, AbstractTrueMeasure)
        ):
            raise ParameterError(
                "sampler input should either be an AbstractDiscreteDistribution or AbstractTrueMeasure."
            )
        if not (
            isinstance(points, np.ndarray) and (points.ndim == 1 or points.ndim == 2)
        ):
            raise ParameterError("points must be a one or two dimensional np.ndarray.")
        if points.ndim == 1:
            points = points[:, None]
        assert (
            points.ndim == 2 and points.shape[0] == sampler.d
        ), "points should be a two dimension array with the number of points equal to the dimension of the sampler"
        mean = np.array(mean)
        if mean.size == 1:
            mean = mean.item() * np.ones(sampler.d)
        assert mean.shape == (sampler.d,), "mean should be a length d vector"
        assert np.isscalar(nu) and nu > 0, "nu should be a positive scalar"
        length_scale = np.array(length_scale)
        if length_scale.size == 1:
            length_scale = length_scale.item() * np.ones(sampler.d)
        assert (
            length_scale.shape == (sampler.d,) and (length_scale > 0).all()
        ), "length_scale should be a vector with length equal to the dimension of the sampler"
        assert (
            np.isscalar(variance) and variance > 0
        ), "variance should be a positive scalar"
        assert np.isscalar(nugget) and nugget > 0, "nugget should be a positive scalar"
        self.points = points
        self.length_scale = length_scale
        self.nu = nu
        self._kernel_variance = variance
        self._variance_deprecation_warned = False
        self.nugget = nugget
        dists = np.linalg.norm(
            points[..., :, None, :] - points[..., None, :, :], axis=-1
        )
        if nu == 1 / 2:
            covariance = np.exp(-dists / self.length_scale)
        elif nu == 3 / 2:
            covariance = (1 + np.sqrt(3) * dists / self.length_scale) * np.exp(
                -np.sqrt(3) * dists / self.length_scale
            )
        elif nu == 5 / 2:
            covariance = (
                1
                + np.sqrt(5) * dists / self.length_scale
                + 5 * dists**2 / (3 * self.length_scale**2)
            ) * np.exp(-np.sqrt(5) * dists / self.length_scale)
        elif nu == np.inf:
            covariance = np.exp(-(dists**2) / (2 * self.length_scale**2))
        else:
            k = np.sqrt(2 * nu) * dists / self.length_scale
            covariance = 2 ** (1 - nu) / gamma(nu) * k**nu * kv(nu, k)
        covariance = variance * covariance + nugget * np.eye(sampler.d)
        super().__init__(
            sampler, mean=mean, covariance=covariance, decomp_type=decomp_type
        )
        self.parameters = ["mean", "variance", "kernel_variance", "standard_deviation", "covariance", "decomp_type"]

    @property
    def variance(self):
        r"""np.ndarray: Vector of marginal variances (the diagonal of
        `covariance`), consistent with the `Gaussian` parent.
        """
        if not self._variance_deprecation_warned:
            self._variance_deprecation_warned = True
            warnings.warn(
                "MaternGP.variance now returns the vector of marginal variances "
                "(the diagonal of the covariance matrix), consistent with the "
                "Gaussian parent. In QMCPy 2.3 and earlier it returned the scalar "
                "global scaling factor. Use MaternGP.kernel_variance to obtain "
                "that scalar.",
                DeprecationWarning,
                stacklevel=2,
            )
        return super().variance

    @property
    def kernel_variance(self):
        r"""float: The scalar global scaling factor of the Matérn kernel, i.e.
        the ``variance`` value supplied to the constructor.
        """
        return self._kernel_variance

    def _spawn(self, sampler, dimension=None):
        dimension = sampler.d if dimension is None else dimension
        if dimension != self.d:
            raise DimensionError(
                "MaternGP cannot be spawned with a different dimension because "
                "its dimension is fixed by the number of points."
            )
        return MaternGP(
            sampler,
            self.points,
            length_scale=self.length_scale,
            nu=self.nu,
            variance=self.kernel_variance,
            mean=self.mean,
            nugget=self.nugget,
            decomp_type=self.decomp_type,
        )
