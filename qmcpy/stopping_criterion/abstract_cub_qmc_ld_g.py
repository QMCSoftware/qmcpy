from .abstract_stopping_criterion import AbstractStoppingCriterion
from ..util.data import Data

from ..util import MaxSamplesWarning, ParameterError, ParameterWarning, CubatureWarning
import numpy as np
from time import time
import warnings


def _default_fudge(m):
    """Default fudge factor: 5 * 2**(-m)."""
    return 5.0 * 2.0 ** (-m)


def _lstsq_pyfunc(x, y):
    """Least-squares solve used by the vectorized vlstsq attribute."""
    return np.linalg.lstsq(x.T, y, rcond=None)[0]


class AbstractCubQMCLDG(AbstractStoppingCriterion):
    """Abstract base class for guaranteed low-discrepancy QMC stopping criteria.

    Implements the fast-transform (FFT/FWT) cubature error bound shared by
    `CubQMCLatticeG`, `CubQMCNetG`, and similar guaranteed lattice/digital-net
    stopping criteria: doubling sample counts each iteration, maintaining the
    running transform coefficients (`_ytildefull`, `_kappanumap`), optional
    control-variate correction, and the cone-condition check that certifies
    the error bound.
    """

    _RESUME_REQUIRED_FIELDS = (
        "solution", "comb_bound_low", "comb_bound_high", "comb_bound_diff", "comb_flags", "n", "n_max", "xfull", "yfull"
    )
    _RESUME_STATE_FIELDS = ("_ytildefull", "_kappanumap")

    def __init__(
        self,
        integrand,
        abs_tol,
        rel_tol,
        n_init,
        n_limit,
        fudge,
        check_cone,
        control_variates,
        control_variate_means,
        update_beta,
        ptransform,
        ft,
        omega,
        allowed_distribs,
        cast_complex,
        error_fun,
    ):
        self.parameters = ["abs_tol", "rel_tol", "n_init", "n_limit"]
        # Input Checks
        if np.log2(n_init) % 1 != 0 or n_init < 2**8:
            warnings.warn(
                "n_init must be a power of two at least 2**8. Using n_init = 2**8",
                ParameterWarning,
            )
            n_init = 2**8
        if np.log2(n_limit) % 1 != 0:
            warnings.warn(
                "n_init must be a power of two. Using n_limit = 2**30", ParameterWarning
            )
            n_limit = 2**30
        # Set Attributes
        self.n_init = int(n_init)
        self.m_init = int(np.log2(n_init))
        self.n_limit = int(n_limit)
        # Ensure integrator n_limit does not exceed the discrete distribution's n_limit.
        # The discrete distribution (e.g., Lattice) may enforce a smaller maximum number
        # of samples; request to generate more samples than the distribution supports
        # will raise a ParameterError when calling the distribution. Cap the integrator
        # n_limit to avoid that situation.
        try:
            dd_n_limit = int(self.integrand.discrete_distrib.n_limit)
        except Exception:
            dd_n_limit = None
        if dd_n_limit is not None and self.n_limit > dd_n_limit:
            warnings.warn(
                f"Integrator n_limit ({self.n_limit}) exceeds discrete distribution n_limit ({dd_n_limit}). Using {dd_n_limit} instead.",
                ParameterWarning,
            )
            self.n_limit = dd_n_limit
        if not (isinstance(error_fun, str) or callable(error_fun)):
            raise AssertionError
        # _error_fun_key stores a simple, serializable string and ensures correct state saving
        # in __getstate__(), bypassing serialization of complex lambda functions, which often fails.
        self.error_fun, self._error_fun_key = self._resolve_error_fun(error_fun)
        self.fudge = fudge
        self.check_cone = check_cone
        self.ft = ft
        self.omega = omega
        self.ptransform = ptransform
        self.cast_complex = cast_complex
        self.r_lag = 4
        self.omg_circ = lambda m: 2 ** (-m)
        self.l_star = int(self.m_init - self.r_lag)
        self.omg_hat = lambda m: self.fudge(m) / (
            (1 + self.fudge(self.r_lag)) * self.omg_circ(self.r_lag)
        )
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        super(AbstractCubQMCLDG, self).__init__(
            allowed_distribs=allowed_distribs, allow_vectorized_integrals=True
        )
        if not (
            self.integrand.discrete_distrib.no_replications == True
        ):
            raise AssertionError("Require the discrete distribution has replications=None")
        if not (
            self.integrand.discrete_distrib.randomize != "FALSE"
        ):
            raise AssertionError("Require discrete distribution is randomized")
        self.set_tolerance(abs_tol, rel_tol)
        # control variates
        self._init_control_variates(control_variates, control_variate_means)
        self.update_beta = update_beta
        if self.ncv > 0:
            if not (self.cv_mu.shape == (
                (self.ncv,) + self.integrand.d_indv
            )):
                raise AssertionError("Control variate means should have shape (len(control variates),d_indv).")
            self.parameters += ["cv", "cv_mu", "update_beta"]
        else:
            self.update_beta = False
        self.vlstsq = np.vectorize(
            _lstsq_pyfunc,
            signature="(k,m),(m)->(k)",
        )

    def _update_kappanumap(self, kappanumap, ytildefull, mfrom, mto, m):
        for l in range(int(mfrom), int(mto), -1):
            nl = 2**l
            oldone = np.abs(
                np.take_along_axis(ytildefull, kappanumap[..., 1 : int(nl)], axis=-1)
            )  # earlier values of kappa, don't touch first one
            newone = np.abs(
                np.take_along_axis(
                    ytildefull, kappanumap[..., nl + 1 : 2 * nl], axis=-1
                )
            )  # later values of kappa,
            *prioridxs, flip = np.where(newone > oldone)
            flip = (
                flip + 1
            )  # add one to account for the fact that we do not consider indices which are powers of 2
            if flip.size != 0:
                additive = np.arange(0, 2**m - 1, 2 ** (l + 1))
                flipall = (flip[None, :] + additive[:, None]).flatten()
                zeroadditive = np.zeros(len(additive), dtype=int)
                pidxs = tuple(
                    (pidx[None, :] + zeroadditive[:, None]).flatten()
                    for pidx in prioridxs
                )  # alternative to tiling
                kappanumap[pidxs + (flipall,)], kappanumap[pidxs + (nl + flipall,)] = (
                    kappanumap[pidxs + (nl + flipall,)],
                    kappanumap[pidxs + (flipall,)],
                )
        return kappanumap

    def _beta_update(self, beta, kappanumap, ytildefull, ycvtildefull, mstart):
        kappa_approx = kappanumap[..., (2**mstart) :]  # kappa index used for fitting
        y4beta = np.take_along_axis(ytildefull, kappa_approx, axis=-1)
        x4beta = np.take_along_axis(ycvtildefull, kappa_approx[..., None, :], axis=-1)
        beta = self.vlstsq(x4beta, y4beta)
        return beta

    def __getstate__(self):
        state = self.__dict__.copy()
        # omg_circ and omg_hat are local lambdas that pickle cannot serialize.
        # Replace them with sentinels; __setstate__ rebuilds them.
        state['omg_circ'] = '__default__'
        state['omg_hat'] = '__default__'
        # error_fun is also a local lambda when constructed from a string keyword.
        # Replace with the canonical string form so it can be reconstructed.
        if self._error_fun_key is not None:
            state['error_fun'] = self._error_fun_key
        # fudge may be a local lambda (e.g. default arg in subclass __init__).
        # If it's the default, replace with a sentinel; otherwise leave for pickle.
        if getattr(self.fudge, '__name__', None) == '<lambda>':
            state['fudge'] = '__default_fudge__'
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Rebuild omg_circ and omg_hat from their sentinels.
        self.omg_circ = lambda m: 2 ** (-m)
        self.omg_hat = lambda m: self.fudge(m) / (
            (1 + self.fudge(self.r_lag)) * self.omg_circ(self.r_lag)
        )
        # Rebuild error_fun from its string key if present.
        if isinstance(self.error_fun, str):
            self.error_fun, _ = self._resolve_error_fun(self.error_fun)
        # Rebuild fudge from its sentinel.
        if self.fudge == '__default_fudge__':
            self.fudge = _default_fudge

    def _validate_resume(self, data):
        required_fields = self._RESUME_REQUIRED_FIELDS
        state_fields = self._RESUME_STATE_FIELDS
        if self.ncv > 0:
            required_fields = required_fields + ("ycvfull", "beta")
            state_fields = state_fields + ("_ycvtildefull",)
        self._validate_resume_with_state(
            data, required_fields=required_fields, state_fields=state_fields
        )
        n_total = int(data.n_total)
        output_shape = self.integrand.d_indv + (n_total,)
        self._validate_resume_shape("xfull", data.xfull, (n_total, self.integrand.d))
        self._validate_resume_shape("yfull", data.yfull, output_shape)
        self._validate_resume_shape("_ytildefull", data._ytildefull, output_shape)
        self._validate_resume_shape("_kappanumap", data._kappanumap, output_shape)
        self._validate_resume_shape("n", data.n, self.integrand.d_indv)
        if int(np.max(np.asarray(data.n))) != n_total:
            raise ParameterError("resume data n must be consistent with n_total.")
        if int(data.n_max) != n_total:
            raise ParameterError("resume data n_total must match n_max.")
        if not self._is_power_of_two(n_total):
            raise ParameterError("resume data n_total must be a power of 2.")
        if self.ncv > 0:
            cv_shape = self.integrand.d_indv + (self.ncv, n_total)
            self._validate_resume_shape("ycvfull", data.ycvfull, cv_shape)
            self._validate_resume_shape(
                "_ycvtildefull", data._ycvtildefull, cv_shape
            )
            self._validate_resume_shape(
                "beta", data.beta, self.integrand.d_indv + (self.ncv,)
            )

    def integrate(self, resume=None):
        """Determine the samples needed to satisfy the target tolerance.

        Doubles the sample count each iteration, updates the running fast
        transform (`_ytildefull`) and its permutation (`_kappanumap`),
        optionally corrects for control variates, and (if `self.check_cone`)
        checks the cone condition that certifies the low-discrepancy error
        bound. Stops once every combined output is within tolerance or
        `self.n_limit` would be exceeded.

        Args:
            resume (Data): Existing integration state to resume from, if
                supported. Defaults to None.

        Returns:
            tuple: Approximation to the integral with shape ``integrand.d_comb``
                and the corresponding data object.
        """
        t_start = time()
        resume_provenance = self._capture_resume_provenance(resume)
        first_resume_iter = False
        trace = self._make_trace_logger()

        data = self._prepare_resume_data(
            resume, self._validate_resume, self._restore_resume_state
        )
        if data is not None:
            # Reset flags so all components are re-evaluated against the new tolerance.
            data.flags_indv = np.tile(False, self.integrand.d_indv)
            data.compute_flags = np.tile(True, self.integrand.d_indv)
            # Set n_min to n_total so the next actual batch starts after the prior samples.
            data.n_min = int(data.n_total)
            # Restore the transform state stored by the previous integrate() call.
            ytildefull = data._ytildefull
            kappanumap = data._kappanumap
            if self.ncv > 0:
                ycvtildefull = data._ycvtildefull
            first_resume_iter = True
            self._set_elapsed_time(data, 0.0, resume_provenance=resume_provenance)
            trace.resume(data, step_value=int(np.log2(max(1, int(data.n_total)))))
        else:
            data = Data(parameters=["solution", "comb_bound_low", "comb_bound_high", "comb_bound_diff", "comb_flags", "n_total", "n", "time_integrate"])
            data.flags_indv = np.tile(False, self.integrand.d_indv)
            data.compute_flags = np.tile(True, self.integrand.d_indv)
            data.n = np.tile(self.n_init, self.integrand.d_indv)
            data.n_min = 0
            data.n_max = self.n_init
            data.solution_indv = np.tile(np.nan, self.integrand.d_indv)
            data.xfull = np.empty((0, self.integrand.d))
            data.yfull = np.empty(self.integrand.d_indv + (0,))
            if self.ncv > 0:
                data.ycvfull = np.empty(self.integrand.d_indv + (self.ncv, 0))
            data.bounds_half_width = np.tile(np.inf, self.integrand.d_indv)
            data.muhat = np.tile(np.nan, self.integrand.d_indv)
            data.beta = np.tile(np.nan, self.integrand.d_indv + (self.ncv,))
        while True:
            m = int(np.log2(data.n_max))
            mllstart = m - self.r_lag - 1
            nllstart = 2**mllstart
            if not first_resume_iter:
                xnext = self.discrete_distrib(n_min=data.n_min, n_max=data.n_max)
                data.xfull = np.concatenate([data.xfull, xnext], 0)
                ynext = self.integrand.f(
                    xnext,
                    periodization_transform=self.ptransform,
                    compute_flags=data.compute_flags,
                )
                ynext[~data.compute_flags] = np.nan
                data.yfull = np.concatenate([data.yfull, ynext], -1)
                if self.ncv > 0:
                    ycvnext = [None] * self.ncv
                    for k in range(self.ncv):
                        ycvnext_k = self.cv[k].f(
                            xnext,
                            periodization_transform=self.ptransform,
                            compute_flags=data.compute_flags,
                        )
                        ycvnext_k[~data.compute_flags] = np.nan
                        ycvnext[k] = ycvnext_k
                    ycvnext = np.stack(ycvnext, -2)
                    data.ycvfull = np.concatenate([data.ycvfull, ycvnext], -1)
            if not first_resume_iter and data.n_min == 0:  # first fresh iteration
                n = int(2**m)
                ytildefull = self.ft(ynext) / np.sqrt(n)
                kappanumap = self._update_kappanumap(
                    np.tile(np.arange(n), self.integrand.d_indv + (1,)),
                    ytildefull,
                    m - 1,
                    0,
                    m,
                )
                if self.ncv > 0:
                    ycvtildefull = self.ft(ycvnext) / np.sqrt(n)
                    data.beta = self._beta_update(
                        data.beta, kappanumap, ytildefull, ycvtildefull, mllstart
                    )
                    ytildefull = ytildefull - (ycvtildefull * data.beta[..., None]).sum(
                        -2
                    )
                    kappanumap = self._update_kappanumap(
                        np.tile(np.arange(n), self.integrand.d_indv + (1,)),
                        ytildefull,
                        m - 1,
                        0,
                        m,
                    )
            elif not first_resume_iter:  # any iteration after the first
                mnext = int(m - 1)
                n = int(2**mnext)
                if not self.update_beta:  # do not update the beta coefficients
                    if self.ncv > 0:
                        ynext[data.compute_flags] = ynext[data.compute_flags] - (
                            ycvnext[data.compute_flags]
                            * data.beta[data.compute_flags, :, None]
                        ).sum(-2)
                    ytildeomega = (
                        self.omega(mnext)
                        * self.ft(ynext[data.compute_flags])
                        / np.sqrt(n)
                    )
                    ytildefull_next = np.nan * np.ones_like(ytildefull)
                    ytildefull_next[data.compute_flags] = (
                        ytildefull[data.compute_flags] - ytildeomega
                    ) / 2
                    ytildefull[data.compute_flags] = (
                        ytildefull[data.compute_flags] + ytildeomega
                    ) / 2
                    ytildefull = np.concatenate([ytildefull, ytildefull_next], axis=-1)
                else:  # update beta
                    ytildefull = np.concatenate(
                        [ytildefull, np.tile(np.nan, ytildefull.shape)], axis=-1
                    )
                    ytildefull[data.compute_flags] = self.ft(
                        data.yfull[data.compute_flags]
                    ) / np.sqrt(2**m)
                    ycvtildefull = np.concatenate(
                        [ycvtildefull, np.tile(np.nan, ycvtildefull.shape)], axis=-1
                    )
                    ycvtildefull[data.compute_flags] = self.ft(
                        data.ycvfull[data.compute_flags]
                    ) / np.sqrt(2**m)
                    data.beta[data.compute_flags] = self._beta_update(
                        data.beta[data.compute_flags],
                        kappanumap[data.compute_flags],
                        ytildefull[data.compute_flags],
                        ycvtildefull[data.compute_flags],
                        mllstart,
                    )
                kappanumap = np.concatenate([kappanumap, n + kappanumap], axis=-1)
                kappanumap[data.compute_flags] = self._update_kappanumap(
                    kappanumap[data.compute_flags],
                    ytildefull[data.compute_flags],
                    m - 1,
                    mllstart,
                    m,
                )
            if self.ncv == 0:
                data.muhat[data.compute_flags] = data.yfull[data.compute_flags].mean(-1)
            else:
                ydiff = data.yfull[data.compute_flags] - (
                    data.ycvfull[data.compute_flags]
                    * data.beta[data.compute_flags, :, None]
                ).sum(-2)
                data.muhat[data.compute_flags] = ydiff.mean(-1) + (
                    data.beta[data.compute_flags]
                    * np.moveaxis(self.cv_mu, 0, -1)[data.compute_flags]
                ).sum(-1)
            data.bounds_half_width[data.compute_flags] = self.fudge(m) * np.abs(
                np.take_along_axis(
                    ytildefull[data.compute_flags],
                    kappanumap[data.compute_flags][..., nllstart : 2 * nllstart],
                    axis=-1,
                )
            ).sum(-1)
            data.indv_bound_low = data.muhat - data.bounds_half_width
            data.indv_bound_high = data.muhat + data.bounds_half_width
            if self.check_cone:
                data.c_stilde_low = np.tile(
                    -np.inf, (m + 1 - self.l_star,) + self.integrand.d_indv
                )
                data.c_stilde_up = np.tile(
                    np.inf, (m + 1 - self.l_star,) + self.integrand.d_indv
                )
                for l in range(
                    self.l_star, m + 1
                ):  # Storing the information for the necessary conditions
                    c_tmp = self.omg_hat(m - l) * self.omg_circ(m - l)
                    c_low = 1.0 / (1 + c_tmp)
                    c_up = 1.0 / (1 - c_tmp)
                    const1 = np.abs(
                        np.take_along_axis(
                            ytildefull[data.compute_flags],
                            kappanumap[data.compute_flags][
                                ..., int(2 ** (l - 1)) : int(2**l)
                            ],
                            axis=-1,
                        )
                    ).sum(-1)
                    idx = int(l - self.l_star)
                    data.c_stilde_low[idx, data.compute_flags] = np.maximum(
                        data.c_stilde_low[idx, data.compute_flags], c_low * const1
                    )
                    if c_tmp < 1:
                        data.c_stilde_up[idx, data.compute_flags] = np.minimum(
                            data.c_stilde_up[idx, data.compute_flags], c_up * const1
                        )
                data.cone_violation = (data.c_stilde_low > data.c_stilde_up).any(0)
                if data.cone_violation.sum() > 0:
                    warnings.warn(
                        "Cone condition violations at indices in data.cone_violation.",
                        CubatureWarning,
                    )
            else:
                data.cone_violation = None
            data.n[data.compute_flags] = data.n_max
            data.n_total = data.n_max
            data.comb_bound_low, data.comb_bound_high = self.integrand.bound_fun(
                data.indv_bound_low, data.indv_bound_high
            )
            data.comb_bound_diff = data.comb_bound_high - data.comb_bound_low
            fidxs = np.isfinite(data.comb_bound_low) & np.isfinite(data.comb_bound_high)
            slow, shigh, abs_tols, rel_tols = (
                data.comb_bound_low[fidxs],
                data.comb_bound_high[fidxs],
                self.abs_tols[fidxs],
                self.rel_tols[fidxs],
            )
            data.solution = np.tile(np.nan, data.comb_bound_low.shape)
            data.solution[fidxs] = (
                1
                / 2
                * (
                    slow
                    + shigh
                    + self.error_fun(slow, abs_tols, rel_tols)
                    - self.error_fun(shigh, abs_tols, rel_tols)
                )
            )
            data.comb_flags = np.tile(False, data.comb_bound_low.shape)
            data.comb_flags[fidxs] = (shigh - slow) <= (
                self.error_fun(slow, abs_tols, rel_tols)
                + self.error_fun(shigh, abs_tols, rel_tols)
            )
            data.flags_indv = self.integrand.dependency(data.comb_flags)
            data.compute_flags = ~data.flags_indv
            self._set_elapsed_time(data, time() - t_start, resume_provenance=resume_provenance)
            trace.iteration(data, step_value=m)
            # Save transform state so this computation can be resumed later.
            data._ytildefull = ytildefull
            data._kappanumap = kappanumap
            if self.ncv > 0:
                data._ycvtildefull = ycvtildefull
            if np.sum(data.compute_flags) == 0:
                break  # sufficiently estimated
            elif 2 * data.n_total > self.n_limit:
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples would exceeds n_limit = %d.
                No more samples will be generated.
                Note that error tolerances may not be satisfied. """ % (
                    int(data.n_total),
                    int(data.n_total),
                    int(self.n_limit),
                )
                warnings.warn(warning_s, MaxSamplesWarning)
                break
            first_resume_iter = False
            data.n_min = data.n_max
            data.n_max = 2 * data.n_min
        self._finalize_integration_data(
            data, time() - t_start, resume_provenance=resume_provenance
        )
        trace.finalize()
        return data.solution, data

    def set_tolerance(self, abs_tol=None, rel_tol=None, rmse_tol=None):
        """Update the stopping criterion's target tolerance.

        Args:
            abs_tol (float): Absolute error tolerance, broadcast to
                `self.abs_tols` with shape `integrand.d_comb`.
            rel_tol (float): Relative error tolerance, broadcast to
                `self.rel_tols` with shape `integrand.d_comb`.
            rmse_tol (float): Unsupported; must be `None`.

        Raises:
            AssertionError: If `rmse_tol` is supplied.
        """
        if not (rmse_tol is None):
            raise AssertionError("rmse_tol not supported by this stopping criterion.")
        if abs_tol is not None:
            self.abs_tol = abs_tol
            self.abs_tols = np.full(self.integrand.d_comb, self.abs_tol)
        if rel_tol is not None:
            self.rel_tol = rel_tol
            self.rel_tols = np.full(self.integrand.d_comb, self.rel_tol)
