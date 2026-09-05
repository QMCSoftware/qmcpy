from .abstract_integrand import AbstractIntegrand
from .keister import Keister
from .box_integral import BoxIntegral
from ..discrete_distribution import DigitalNetB2
import numpy as np
from itertools import combinations
import scipy.special


class SensitivityIndices(AbstractIntegrand):
    r"""Sensitivity indices i.e. normalized Sobol' Indices.

    Examples:
        Singleton indices

        >>> function = Keister(DigitalNetB2(dimension=4,seed=7))
        >>> integrand = SensitivityIndices(function,indices='singletons')
        >>> integrand.indices
        array([[ True, False, False, False],
               [False,  True, False, False],
               [False, False,  True, False],
               [False, False, False,  True]])
        >>> y = integrand(2**10)
        >>> y.shape
        (2, 3, 4, 1024)
        >>> ymean = y.mean(-1)
        >>> ymean.shape
        (2, 3, 4)
        >>> sigma_hat = ymean[:,2,:]-ymean[:,1,:]**2
        >>> sigma_hat.shape
        (2, 4)
        >>> sigma_hat
        array([[17.81421941, 17.81421941, 17.81421941, 17.81421941],
               [17.81421941, 17.81421941, 17.81421941, 17.81421941]])
        >>> closed_total_approx = ymean[:,0]/sigma_hat
        >>> closed_total_approx.shape
        (2, 4)
        >>> closed_total_approx
        array([[0.23756849, 0.2423556 , 0.24844127, 0.24771739],
               [0.25090825, 0.25344588, 0.24531633, 0.26244379]])

        Check what all indices look like for $d=3$

        >>> integrand = SensitivityIndices(Keister(DigitalNetB2(dimension=3,seed=7)),indices='all')
        >>> integrand.indices
        array([[ True, False, False],
               [False,  True, False],
               [False, False,  True],
               [ True,  True, False],
               [ True, False,  True],
               [False,  True,  True]])

        Vectorized function for all singletons and pairs of dimensions

        >>> function = BoxIntegral(DigitalNetB2(dimension=4,seed=7,replications=2**4),s=np.arange(1,31).reshape((5,6)))
        >>> indices = np.zeros((function.d,function.d,function.d),dtype=bool)
        >>> r = np.arange(function.d)
        >>> indices[r,:,r] = True
        >>> indices[:,r,r] = True
        >>> integrand = SensitivityIndices(function,indices=indices)
        >>> integrand.indices.shape
        (4, 4, 4)
        >>> integrand.indices
        array([[[ True, False, False, False],
                [ True,  True, False, False],
                [ True, False,  True, False],
                [ True, False, False,  True]],
        <BLANKLINE>
               [[ True,  True, False, False],
                [False,  True, False, False],
                [False,  True,  True, False],
                [False,  True, False,  True]],
        <BLANKLINE>
               [[ True, False,  True, False],
                [False,  True,  True, False],
                [False, False,  True, False],
                [False, False,  True,  True]],
        <BLANKLINE>
               [[ True, False, False,  True],
                [False,  True, False,  True],
                [False, False,  True,  True],
                [False, False, False,  True]]])
        >>> y = integrand(2**10)
        >>> y.shape
        (2, 3, 4, 4, 5, 6, 16, 1024)
        >>> muhats = y.mean(-1)
        >>> muhats.shape
        (2, 3, 4, 4, 5, 6, 16)
        >>> muhathat = muhats.mean(-1)
        >>> muhathat.shape
        (2, 3, 4, 4, 5, 6)
        >>> sigma_hat = muhathat[:,2,:]-muhathat[:,1,:]**2
        >>> sigma_hat.shape
        (2, 4, 4, 5, 6)
        >>> closed_total_approx = muhathat[:,0]/sigma_hat
        >>> closed_total_approx.shape
        (2, 4, 4, 5, 6)

    **References: **

    1.  Aleksei G. Sorokin and Jagadeeswaran Rathinavel.
        On Bounding and Approximating Functions of Multiple Expectations Using Quasi-Monte Carlo.
        International Conference on Monte Carlo and Quasi-Monte Carlo Methods in Scientific Computing.
        Cham: Springer International Publishing, 2022.
        [https://link.springer.com/chapter/10.1007/978-3-031-59762-6_29](https://link.springer.com/chapter/10.1007/978-3-031-59762-6_29).

    2.  Art B. Owen.
        Monte Carlo theory, methods and examples.
        Appendix A. Equations (A.16) and (A.18). 2013.
        [https://artowen.su.domains/mc/A-anova.pdf](https://artowen.su.domains/mc/A-anova.pdf).
    """

    def __init__(self, integrand, indices="singletons"):
        r"""
        Args:
            integrand (AbstractIntegrand): Integrand to find sensitivity
                indices of.
            indices (np.ndarray): Bool array with shape $(\dots,d)$ where each
                length $d$ vector item indicates which dimensions are active in
                the subset.

                - The default `indices='singletons'` sets `indices=np.eye(d,dtype=bool)`.
                - Setting `incides='all'` sets `indices = np.array([[bool(int(b)) for b in np.binary_repr(i,width=d)] for i in range(1,2**d-1)],dtype=bool)`
        """
        self.parameters = ["indices"]
        self.integrand = integrand
        self.dtilde = self.integrand.d
        assert self.dtilde > 1, "SensitivityIndices does not make sense for d=1"
        self.indices = indices
        if isinstance(self.indices, str) and self.indices == "singletons":
            self.indices = np.eye(self.dtilde, dtype=bool)
        elif isinstance(self.indices, str) and self.indices == "all":
            self.indices = np.zeros((0, self.dtilde), dtype=bool)
            for r in range(1, self.dtilde):
                idxs_r = np.zeros(
                    (int(scipy.special.comb(self.dtilde, r)), self.dtilde), dtype=bool
                )
                for i, comb in enumerate(combinations(range(self.dtilde), r)):
                    idxs_r[i, comb] = True
                self.indices = np.vstack([self.indices, idxs_r])
        self.indices = np.atleast_1d(self.indices)
        assert (
            self.indices.dtype == bool
            and self.indices.ndim >= 1
            and self.indices.shape[-1] == self.dtilde
        )
        assert (
            not (self.indices == self.indices[..., 0, None]).all(-1).any()
        ), "indices cannot include the emptyset or the set of all dimensions"
        self.not_indices = ~self.indices
        # sensitivity_index
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib.spawn(
            s=1, dimensions=[2 * self.dtilde]
        )[0]
        self.sampler = self.integrand.sampler
        self.i_slice = (slice(None),) * len(self.integrand.d_indv)
        super(SensitivityIndices, self).__init__(
            dimension_indv=(2, 3) + self.indices.shape[:-1] + self.integrand.d_indv,
            dimension_comb=(2,) + self.indices.shape[:-1] + self.integrand.d_indv,
            parallel=False,
        )
        self.d = 2 * self.dtilde

    def f(self, x, *args, **kwargs):
        if "compute_flags" in kwargs:
            compute_flags = kwargs["compute_flags"]
            del kwargs["compute_flags"]
        else:
            compute_flags = np.ones(self.d_indv, dtype=bool)
        assert compute_flags.shape == self.d_indv
        z = x[..., self.dtilde :]
        x = x[..., : self.dtilde]
        v = np.zeros_like(x)
        y = np.zeros(self.d_indv + x.shape[:-1], dtype=float)
        f_x = self.integrand.f(x, *args, **kwargs)
        f_z = self.integrand.f(z, *args, **kwargs)
        for i in np.ndindex(self.indices.shape[:-1]):
            flags_closed = compute_flags[(0, 0) + i]
            flags_total = compute_flags[(1, 0) + i]
            flags_i = flags_closed | flags_total
            if not flags_i.any():
                continue
            u_bool = self.indices[i]
            not_u_bool = self.not_indices[i]
            v[..., u_bool] = x[..., u_bool]
            v[..., not_u_bool] = z[..., not_u_bool]
            f_v = self.integrand.f(v, compute_flags=flags_i, *args, **kwargs)
            y[(0, 0) + i + self.i_slice] = f_x * (f_v - f_z)  # A.18
            y[(1, 0) + i + self.i_slice] = (f_z - f_v) ** 2 / 2  # A.16
            y[(slice(None), 1) + i + self.i_slice] = f_x[(None,) + self.i_slice]  # mu
            y[(slice(None), 2) + i + self.i_slice] = (
                f_x[(None,) + self.i_slice] ** 2
            )  # sigma^2+mu^2
            # here we copy mu and sigma^2+mu^2 since if these these were not copied there is a chance the bounds could change
            # for mu and/or sigma and then an index which was previously approximated sufficiently woulud become insufficientlly approximated
            # and it would then be difficult ot go back and resample the numerator for that approximation
        return y

    def _spawn(self, level, sampler):
        new_integrand = self.integrand.spawn(level, sampler)
        return SensitivityIndices(integrand=new_integrand, indices=self.indices)

    def bound_fun(self, bound_low, bound_high):
        tau_low, mu_low, f2_low = bound_low[:, 0], bound_low[:, 1], bound_low[:, 2]
        tau_high, mu_high, f2_high = (
            bound_high[:, 0],
            bound_high[:, 1],
            bound_high[:, 2],
        )
        sigma2_low1, sigma2_low2 = f2_high - mu_low**2, f2_high - mu_high**2
        comb_bounds_low = np.clip(
            np.minimum.reduce([tau_low / sigma2_low1, tau_low / sigma2_low2]), 0, 1
        )
        sigma2_high1, sigma2_high2 = f2_low - mu_low**2, f2_low - mu_high**2
        comb_bounds_high = np.clip(
            np.maximum.reduce([tau_high / sigma2_high1, tau_high / sigma2_high2]), 0, 1
        )
        violated = (sigma2_high1 <= 0) | (sigma2_high2 <= 0)
        comb_bounds_low[violated], comb_bounds_high[violated] = 0, 1
        return comb_bounds_low, comb_bounds_high

    def dependency(self, comb_flags):
        return np.repeat(comb_flags[:, None], 3, axis=1)
