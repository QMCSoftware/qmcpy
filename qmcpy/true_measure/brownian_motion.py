from .gaussian import Gaussian
from ..discrete_distribution import DigitalNetB2
from ..util import ParameterError, ParameterWarning
import warnings
import numpy as np
from scipy.stats import norm


class BrownianMotion(Gaussian):
    r"""
    Brownian Motion as described in [https://en.wikipedia.org/wiki/Brownian_motion](https://en.wikipedia.org/wiki/Brownian_motion).
    For a standard Brownian Motion $W$ we define the Brownian Motion $B$ with initial value $B_0$, drift $\gamma$, and diffusion $\sigma^2$ to be

    $$B(t) = B_0 + \gamma t + \sigma W(t).$$

    Examples:
        Example 1: Basic usage

        >>> true_measure = BrownianMotion(DigitalNetB2(4,seed=7),t_final=2,drift=2)
        >>> true_measure(2)
        array([[0.82189263, 2.7851793 , 3.60126805, 3.98054724],
               [0.2610643 , 0.06170064, 1.06448269, 2.30990767]])
        >>> true_measure
        BrownianMotion (AbstractTrueMeasure)
            time_vec        [0.5 1.  1.5 2. ]
            drift           2^(1)
            mean            [1. 2. 3. 4.]
            variance        [0.5 1.  1.5 2. ]
            standard_deviation [0.707 1.    1.225 1.414]
            covariance      [[0.5 0.5 0.5 0.5]
                             [0.5 1.  1.  1. ]
                             [0.5 1.  1.5 1.5]
                             [0.5 1.  1.5 2. ]]
            decomp_type     PCA

        Example 2: With independent replications

        >>> x = BrownianMotion(DigitalNetB2(3,seed=7,replications=2),t_final=2,drift=2)(4)
        >>> x.shape
        (2, 4, 3)
        >>> x
        array([[[0.66154685, 1.50620966, 3.52322901],
                [1.77064217, 3.32782204, 4.45013223],
                [1.33558688, 3.26017547, 3.40692337],
                [2.10317345, 3.78961839, 6.17948096]],
        <BLANKLINE>
               [[1.77868019, 2.75347902, 3.41161419],
                [0.44891984, 2.53987304, 4.7224811 ],
                [0.23147948, 2.25289769, 3.00039101],
                [2.06762574, 3.21756319, 4.93375923]]])

        Example 3: With Brownian Bridge construction

        >>> true_measure = BrownianMotion(DigitalNetB2(4,seed=7),decomp_type='BrownianBridge')
        >>> true_measure(2)
        array([[-0.02048429,  0.41054648, -0.13899299,  0.3095377 ],
               [-0.38732442, -1.19527027, -1.12175754, -1.58454187]])
        >>> true_measure
        BrownianMotion (AbstractTrueMeasure)
            time_vec        [0.25 0.5  0.75 1.  ]
            drift           0
            mean            [0. 0. 0. 0.]
            variance        [0.25 0.5  0.75 1.  ]
            standard_deviation [0.5   0.707 0.866 1.   ]
            covariance      [[0.25 0.25 0.25 0.25]
                             [0.25 0.5  0.5  0.5 ]
                             [0.25 0.5  0.75 0.75]
                             [0.25 0.5  0.75 1.  ]]
            decomp_type     BROWNIANBRIDGE
            bridge_construction_times [1.   0.5  0.75 0.25]
            bridge_output_times [0.25 0.5  0.75 1.  ]

        Example 4: With Brownian Bridge construction and independent replications

        >>> x = BrownianMotion(DigitalNetB2(4,seed=7,replications=3),decomp_type='BrownianBridge')(2)
        >>> x.shape
        (3, 2, 4)
        >>> x
        array([[[ 0.04920439,  0.52848898,  0.12091923, -0.17751616],
                [ 0.71498158,  0.96872916,  1.71491732,  2.21516041]],
        <BLANKLINE>
               [[ 0.12575161, -0.48324258, -0.17795825, -0.19149823],
                [ 0.28188179,  1.03215652,  0.17848014,  0.62971114]],
        <BLANKLINE>
               [[ 0.59845146,  1.10849282,  1.34022073,  1.02092441],
                [-0.20298903, -0.23324496, -0.3026512 , -0.35202342]]])

        Example 5: With custom monitoring times and passing bridge_vdc_gray_ordering=False (reaches all four cases)

        >>> true_measure = BrownianMotion(DigitalNetB2(4,seed=7),decomp_type='BrownianBridge',monitoring_times=[0.6,1.0,0.3,0.8],bridge_vdc_gray_ordering=False)        
        >>> true_measure.time_vec
        array([0.3, 0.6, 0.8, 1. ])
        >>> true_measure(2)
        array([[-0.42678211,  0.23976687,  0.19961117,  0.56330283],
               [-0.31994843, -1.22738085, -1.29415239, -1.73713917]])
        >>> true_measure.bridge_construction_times
        array([0.6, 1. , 0.3, 0.8])
        >>> true_measure.bridge_output_times
        array([0.3, 0.6, 0.8, 1. ])

        Example 6: With custom monitoring times. By default the times are sorted and inserted in van der Corput order 

        >>> true_measure = BrownianMotion(DigitalNetB2(4,seed=7),decomp_type='BrownianBridge',monitoring_times=[0.6,1.0,0.3,0.8])
        >>> true_measure.time_vec
        array([0.3, 0.6, 0.8, 1. ])
        >>> true_measure(2)
        array([[-0.02913874,  0.4363325 , -0.07341545,  0.3095377 ],
               [-0.44240726, -1.34558221, -1.22522271, -1.58454187]])
        >>> true_measure.bridge_construction_times
        array([1. , 0.6, 0.8, 0.3])
        >>> true_measure.bridge_output_times
        array([0.3, 0.6, 0.8, 1. ])

        Example 7: With custom output order

        >>> true_measure = BrownianMotion(DigitalNetB2(4,seed=7),decomp_type='BrownianBridge',monitoring_times=[0.6,1.0,0.3,0.8],bridge_output_order='input')
        >>> true_measure.time_vec
        array([0.3, 0.6, 0.8, 1. ])
        >>> true_measure(2)
        array([[ 0.4363325 ,  0.3095377 , -0.02913874, -0.07341545],
               [-1.34558221, -1.58454187, -0.44240726, -1.22522271]])
        >>> true_measure.bridge_construction_times
        array([1. , 0.6, 0.8, 0.3])
        >>> true_measure.bridge_output_times
        array([0.6, 1. , 0.3, 0.8])

        **References:**

        1.  Art B. Owen. 
            Monte Carlo theory, methods and examples.
            Section 6.4, Detailed Simulation of Brownian Motion, 2013
            [https://artowen.su.domains/mc/](https://artowen.su.domains/mc/)
    """

    def __init__(
        self,
        sampler,
        t_final=1,
        initial_value=0,
        drift=0,
        diffusion=1,
        decomp_type="PCA",
        lazy_decomp=True,
        monitoring_times=None,
        bridge_vdc_gray_ordering=True,
        bridge_output_order='increasing',
    ):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]): Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            t_final (float): End time.
            initial_value (float): Initial value $B_0$.
            drift (int): Drift $\gamma$.
            diffusion (int): Diffusion $\sigma^2$.
            decomp_type (str): Method for decomposition for covariance matrix. Options include

                - `'PCA'` for principal component analysis,
                - `'Cholesky'` for cholesky decomposition, or
                - `'BrownianBridge'` or `'Bridge'` for brownian bridge construction.
            lazy_decomp (bool): If True, defer expensive matrix decomposition until needed.
            monitoring_times (Union[np.ndarray, list]): Optional custom sampling times for `'BrownianBridge'` 
                with length d. The given order is the insertion order if `'bridge_vdc_gray_ordering'` is False.
            bridge_vdc_gray_ordering (bool): For `'BrownianBridge'` when monitoring_times is specified. If True, 
                monitoring_times is sorted to match van der Corput ordering. 
            bridge_output_order (str): If `'increasing'`, output is returned in increasing order. If `'input'`, 
                output matches the order given in `'monitoring_times'`. If a custom monitoring times is not given,
                the output is given in increasing order.
        """
        if str(decomp_type).upper() == "BRIDGE":
            decomp_type = "BrownianBridge"
        self.parameters = [
            "time_vec",
            "drift",
            "mean",
            "variance",
            "standard_deviation",
            "covariance",
            "decomp_type",
        ]
        # default to transform from standard uniform
        self.domain = np.array([[0, 1]])
        self._parse_sampler(sampler)
        if not np.isfinite(t_final) or t_final < 0:
            raise ParameterError(f"t_final must be non-negative and finite. Got {t_final}.")
        self.t = t_final  # exercise time
        self.initial_value = initial_value
        self.drift = drift
        self.diffusion = diffusion
        self.bridge_vdc_gray_ordering = bridge_vdc_gray_ordering
        if str(bridge_output_order).lower() not in ("increasing", "input"):
            raise ParameterError("bridge_output_order must be 'increasing' or 'input'.")
        self.bridge_output_order = str(bridge_output_order).lower()
        self.monitoring_times = monitoring_times
        construction_times = self._get_construction_times(monitoring_times, decomp_type, bridge_vdc_gray_ordering)
        self.time_vec = np.sort(construction_times)
        self.diffused_sigma_bm = self.diffusion * np.minimum.outer(
            self.time_vec, self.time_vec
        )
        self.drift_time_vec_plus_init = (
            self.drift * self.time_vec + self.initial_value
        )  # mean
        if str(decomp_type).upper() not in ("PCA", "CHOLESKY", "BROWNIANBRIDGE"):
            raise ParameterError(
                f"decomp_type must be 'PCA', 'Cholesky', or 'BrownianBridge'. Got '{decomp_type}'."
            )
        self._parse_gaussian_params(
            self.drift_time_vec_plus_init,
            self.diffused_sigma_bm,
            decomp_type,
            lazy_decomp if str(decomp_type).upper() != "BROWNIANBRIDGE" else True,
        )
        if self.decomp_type == "BROWNIANBRIDGE":
            self.bridge_construction_times = construction_times
            self._setup_bridge()  # precompute bridge parameters
            self._output_order = self._get_output_order()
            self.bridge_output_times = self.time_vec[self._output_order]
            self.parameters += ["bridge_construction_times", "bridge_output_times"]
            order = self._output_order
            if not np.array_equal(order, np.arange(self.d)):
                self._mean = self._mean[order]
                self.mu = self.mu[order]
                self._covariance = self._covariance[order][:, order]
                self.sigma = self.sigma[order][:, order]
        if self.decomp_type == "BROWNIANBRIDGE" and not (self.d > 0 and (self.d & (self.d - 1)) == 0):
            warnings.warn(
                f"BrownianBridge is most efficient when d is a power of 2 (e.g., 1, 2, 4, 8, 16). Got d={self.d}.",
                ParameterWarning,
                stacklevel=2
            )
        self.range = np.array([[-np.inf, np.inf]])
        super(Gaussian, self).__init__()

    def _spawn(self, sampler, dimension):
        monitoring_times = None
        if self.decomp_type == "BROWNIANBRIDGE" and dimension == self.d:
            monitoring_times = self.monitoring_times
        return BrownianMotion(
            sampler,
            t_final=self.t,
            initial_value=self.initial_value,
            drift=self.drift,
            diffusion=self.diffusion,
            decomp_type=self.decomp_type,
            lazy_decomp=self.lazy_decomp,
            monitoring_times=monitoring_times,
            bridge_vdc_gray_ordering=self.bridge_vdc_gray_ordering,
            bridge_output_order=self.bridge_output_order,
        )

    def _transform(self, x):
        if self.decomp_type == "BROWNIANBRIDGE":
            z = norm.ppf(x)
            w = self._bridge_transform(z)
            paths = self.drift_time_vec_plus_init + np.sqrt(self.diffusion) * w
            return paths[..., self._output_order]
        return super()._transform(x)

    def _get_construction_times(self, monitoring_times, decomp_type, bridge_vdc_gray_ordering):
        """Return d construction times"""
        if str(decomp_type).upper() != "BROWNIANBRIDGE":
            if monitoring_times is not None:
                raise ParameterError("monitoring_times is only valid with decomp_type='BrownianBridge'.")
            return np.linspace(self.t / self.d, self.t, self.d)  # evenly spaced
        if monitoring_times is None:
            return self._van_der_corput(self.d, self.t)  # default bridge ordering
        s = np.asarray(monitoring_times, dtype=float).flatten()
        if s.shape != (self.d,):
            raise ParameterError(f"monitoring_times must have length d={self.d}. Got length {s.shape[0]}.")
        if not np.isfinite(s).all():
            raise ParameterError("monitoring_times must be finite. Got NaN or infinite values.")
        if (s <= 0).any():
            raise ParameterError("monitoring_times must be positive.")
        if (s > self.t).any():
            raise ParameterError(f"maximum value in monitoring_times must not exceed t_final={self.t}. Got max {s.max()}.")
        if np.unique(s).size != self.d:
            raise ParameterError("monitoring_times must be distinct.")
        if bridge_vdc_gray_ordering:
            ranks = np.argsort(np.argsort(self._van_der_corput(self.d, self.t)))
            return np.sort(s)[ranks]
        return s

    def _get_output_order(self):
        """Return array for output order"""
        if self.bridge_output_order == "increasing" or self.monitoring_times is None:
            return np.arange(self.d)
        target = np.asarray(self.monitoring_times, dtype=float).flatten()
        return np.argsort(np.argsort(target))

    @staticmethod
    def _van_der_corput(d, t_final):
        """First d van der Corput points multiplied by t_final."""
        times = DigitalNetB2(1, randomize=False, order='GRAY')(d, warn=False).flatten()
        times[0] = 1.0
        return t_final * times

    def _setup_bridge(self):
        """Precompute parameters (Owen Algorithm 6.1)"""
        s = self.bridge_construction_times
        d = self.d
        left = np.full(d, -1, dtype=int)
        right = np.full(d, -1, dtype=int)
        a = np.zeros(d)
        b = np.zeros(d)
        w = np.zeros(d)
        for j in range(d):
            for k in range(j):
                if s[k] < s[j] and (left[j] == -1 or s[k] > s[left[j]]):
                    left[j] = k
                elif s[k] > s[j] and (right[j] == -1 or s[k] < s[right[j]]):
                    right[j] = k
            if left[j] >= 0 and right[j] >= 0:  # both anchors
                s_left, s_right = s[left[j]], s[right[j]]
                a[j] = (s_right - s[j]) / (s_right - s_left)
                b[j] = (s[j] - s_left) / (s_right - s_left)
                w[j] = np.sqrt((s[j] - s_left) * (s_right - s[j]) / (s_right - s_left))
            elif left[j] >= 0:  # left anchor
                a[j] = 1.0
                w[j] = np.sqrt(s[j] - s[left[j]])
            elif right[j] >= 0:  # right anchor
                s_right = s[right[j]]
                b[j] = s[j] / s_right
                w[j] = np.sqrt(s[j] * (s_right - s[j]) / s_right)
            else:  # first point
                w[j] = np.sqrt(s[j])
        self._bridge_left = left
        self._bridge_right = right
        self._bridge_a = a
        self._bridge_b = b
        self._bridge_w = w
        self._increasing_order = np.argsort(s)  # increasing time

    def _bridge_transform(self, z):
        """Build Brownian Motion paths (Owen Algorithm 6.2)"""
        left = self._bridge_left
        right = self._bridge_right
        a = self._bridge_a
        b = self._bridge_b
        w = self._bridge_w
        paths = np.empty(z.shape[:-1] + (self.d,))
        for j in range(self.d):
            paths[..., j] = w[j] * z[..., j]
            if left[j] >= 0:
                paths[..., j] += a[j] * paths[..., left[j]]
            if right[j] >= 0:
                paths[..., j] += b[j] * paths[..., right[j]]
        return paths[..., self._increasing_order]
