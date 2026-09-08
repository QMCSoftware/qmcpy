from .abstract_integrand import AbstractIntegrand
from ..discrete_distribution import DigitalNetB2  #pylint: disable=unused-import
from ..true_measure import Gaussian
from ..util import ParameterError
import numpy as np


class BayesianLRCoeffs(AbstractIntegrand):
    r"""Logistic Regression Coefficients computed as the posterior mean in a
    Bayesian framework.

    Examples:
        >>> integrand = BayesianLRCoeffs(DigitalNetB2(3,seed=7),feature_array=np.arange(8).reshape((4,2)),response_vector=[0,0,1,1])
        >>> y = integrand(2**10)
        >>> y.shape
        (2, 3, 1024)
        >>> y.mean(-1)
        array([[ 0.04517466, -0.01103669, -0.06614381],
               [ 0.02162049,  0.02162049,  0.02162049]])

        With independent replications

        >>> integrand = BayesianLRCoeffs(DigitalNetB2(3,seed=7,replications=2**4),feature_array=np.arange(8).reshape((4,2)),response_vector=[0,0,1,1])
        >>> y = integrand(2**6)
        >>> y.shape
        (2, 3, 16, 64)
        >>> muhats = y.mean(-1)
        >>> muhats.shape
        (2, 3, 16)
        >>> muhats.mean(-1)
        array([[ 0.0587368 , -0.01718134, -0.07203021],
               [ 0.02498059,  0.02498059,  0.02498059]])
    """

    def __init__(
        self, sampler, feature_array: np.ndarray, response_vector: np.ndarray, prior_mean: np.ndarray = 0, prior_covariance: np.ndarray = 10
    ) -> None:
        r"""Initialize a BayesianLRCoeffs integrand.

        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]):
                Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            feature_array (np.ndarray): Array of features with shape $(N,d-1)$
                where $N$ is the number of observations and $d$ is the
                dimension.
            response_vector (np.ndarray): Binary responses vector of length
                $N$.
            prior_mean (np.ndarray): Length $d$ vector of prior means, one for
                each coefficient.

                - The first $d-1$ inputs correspond to the $d-1$ features.
                - The last input corresponds to the intercept coefficient.
            prior_covariance (np.ndarray): Prior covariance array with shape
                $(d,d)$ d x d where indexing is consistent with the prior mean.
        """
        self.prior_mean = prior_mean
        self.prior_covariance = prior_covariance
        self.sampler = sampler
        self.true_measure = Gaussian(
            self.sampler, mean=self.prior_mean, covariance=self.prior_covariance
        )
        self.feature_array = np.array(feature_array, dtype=float)
        self.response_vector = np.array(response_vector, dtype=float)
        obs, dm1 = self.feature_array.shape
        self.num_coeffs = dm1 + 1
        if self.num_coeffs != self.true_measure.d:
            raise ParameterError(
                "sampler must have dimension one more than the number of features in the feature_array."
            )
        if (
            self.response_vector.shape != (obs,)
            or ((self.response_vector != 0) & (self.response_vector != 1)).any()
        ):
            raise ParameterError(
                "response_vector must have the same length as feature_array and contain only 0 or 1 entries."
            )
        self.feature_array = np.column_stack((self.feature_array, np.ones((obs, 1))))
        super(BayesianLRCoeffs, self).__init__(
            dimension_indv=(2, self.num_coeffs),
            dimension_comb=self.num_coeffs,
            parallel=False,
        )

    def g(self, x):
        z = np.einsum("...j,ij->...i", x, self.feature_array)
        z1 = z * self.response_vector
        with np.errstate(over="ignore"):
            den = np.exp(np.sum(z1 - np.where(z < 100, np.log(1 + np.exp(z)), z), -1))
        y = np.zeros(self.d_indv + x.shape[:-1], dtype=float)
        y[0] = x.transpose([-1] + [i for i in range(x.ndim - 1)]) * den
        y[1] = den
        return y

    def _spawn(self, level, sampler):
        return BayesianLRCoeffs(
            sampler=sampler,
            feature_array=self.feature_array,
            response_vector=self.response_vector,
            prior_mean=self.prior_mean,
            prior_covariance=self.prior_covariance,
        )

    def bound_fun(self, bound_low, bound_high):
        num_bounds_low, den_bounds_low = bound_low[0], bound_low[1]
        num_bounds_high, den_bounds_high = bound_high[0], bound_high[1]
        comb_bounds_low = np.minimum.reduce(
            [
                num_bounds_low / den_bounds_low,
                num_bounds_high / den_bounds_low,
                num_bounds_low / den_bounds_high,
                num_bounds_high / den_bounds_high,
            ]
        )
        comb_bounds_high = np.maximum.reduce(
            [
                num_bounds_low / den_bounds_low,
                num_bounds_high / den_bounds_low,
                num_bounds_low / den_bounds_high,
                num_bounds_high / den_bounds_high,
            ]
        )
        violated = (den_bounds_low <= 0) * (0 <= den_bounds_high)
        comb_bounds_low[violated], comb_bounds_high[violated] = -np.inf, np.inf
        return comb_bounds_low, comb_bounds_high

    def dependency(self, comb_flags):
        return np.vstack((comb_flags, comb_flags))
