from typing import Union
from .abstract_cub_mlmc import AbstractCubMLMC
from ..integrand.abstract_integrand import AbstractIntegrand
from ..util.data import Data
import copy
from ..discrete_distribution import IIDStdUniform
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractIIDDiscreteDistribution,
)
from ..integrand import FinancialOption
from ..util import MaxSamplesWarning, ParameterError, MaxLevelsWarning
import numpy as np
from scipy.stats import norm
from time import time
import warnings


class CubMLMC(AbstractCubMLMC):
    """
    Multilevel IID Monte Carlo stopping criterion.

    Examples:
        >>> fo = FinancialOption(IIDStdUniform(seed=7))
        >>> sc = CubMLMC(fo,abs_tol=1.5e-2)
        >>> solution,data = sc.integrate()
        >>> data
        Data (Data)
            solution        1.785
            n_total         3199033
            levels          2^(2)
            n_level         [2438191  490331  207606   62905]
            mean_level      [1.715 0.053 0.013 0.003]
            var_level       [21.829  1.761  0.452  0.11 ]
            cost_per_sample [ 2.  4.  8. 16.]
            alpha           2.008
            beta            1.997
            gamma           ...
            time_integrate  ...
        CubMLMC (AbstractStoppingCriterion)
            rmse_tol        0.006
            n_init          2^(8)
            levels_min      2^(1)
            levels_max      10
            theta           2^(-1)
        FinancialOption (AbstractIntegrand)
            option          ASIAN
            call_put        CALL
            volatility      2^(-1)
            start_price     30
            strike_price    35
            interest_rate   0
            t_final         1
            asian_mean      ARITHMETIC
        GeometricBrownianMotion (AbstractTrueMeasure)
            time_vec        1
            drift           0
            diffusion       2^(-2)
            mean_gbm        30
            covariance_gbm  255.623
            decomp_type     PCA
        IIDStdUniform (AbstractIIDDiscreteDistribution)
            d               1
            replications    1
            entropy         7

    **References:**

    1.  M.B. Giles. 'Multi-level Monte Carlo path simulation'.
        Operations Research, 56(3):607-617, 2008.
        [http://people.maths.ox.ac.uk/~gilesm/files/OPRE_2008.pdf](http://people.maths.ox.ac.uk/~gilesm/files/OPRE_2008.pdf).

    2. [http://people.maths.ox.ac.uk/~gilesm/mlmc/#MATLAB](http://people.maths.ox.ac.uk/~gilesm/mlmc/#MATLAB).
    """

    _RESUME_REQUIRED_FIELDS = (
        "levels", "n_level", "sum_level", "diff_n_level", "cost_level", "level_integrands"
    )

    def __init__(
        self,
        integrand: AbstractIntegrand,
        abs_tol: Union[float, np.ndarray] = 0.05,
        rmse_tol: Union[None, np.ndarray] = None,
        n_init: int = 256,
        n_limit: int = 10**10,
        alpha: Union[float, np.ndarray] = 0.01,
        levels_min: int = 2,
        levels_max: int = 10,
        alpha0: float = -1.0,
        beta0: float = -1.0,
        gamma0: float = -1.0,
    ) -> None:
        r"""Initialize a CubMLMC stopping criterion.

        Args:
            integrand (AbstractIntegrand): The integrand.
            abs_tol (Union[float, np.ndarray]): Absolute error tolerance.
            rmse_tol (Union[None, np.ndarray]): Root mean squared error tolerance. If
                supplied, then absolute tolerance and alpha are ignored in
                favor of the rmse tolerance.
            n_init (int): Initial number of samples.
            n_limit (int): Maximum number of samples.
            alpha (Union[float, np.ndarray]): Uncertainty level in $(0,1)$.
            levels_min (int): Minimum level of refinement $\geq 2$.
            levels_max (int): Maximum level of refinement $\geq$ `levels_min`.
            alpha0 (float): Weak error is $\mathcal{O}(2^{-\alpha_0\ell})$ in
                the level $\ell$. If `alpha0`$\leq 0$ then it will be
                estimated.
            beta0 (float): Variance is $\mathcal{O}(2^{-\beta_0\ell})$ in the
                level $\ell$. If `beta0`$\leq 0$ then it will be estimated.
            gamma0 (float): Sample cost is $\mathcal{O}(2^{\gamma_0\ell})$ in
                the level $\ell$. If `gamma0`$\leq 0$ then it will be
                estimated.
        """
        self.parameters = ["rmse_tol", "n_init", "levels_min", "levels_max", "theta"]
        if levels_min < 2:
            raise ParameterError("needs levels_min >= 2")
        if levels_max < levels_min:
            raise ParameterError("needs levels_max >= levels_min")
        if n_init <= 0:
            raise ParameterError("needs n_init > 0")
        # initialization
        if rmse_tol:
            self.rmse_tol = float(rmse_tol)
        else:  # use absolute tolerance
            self.rmse_tol = float(abs_tol) / norm.ppf(1 - alpha / 2)
        self.alpha = alpha
        if not (0 < self.alpha < 1):
            raise AssertionError
        self.n_init = n_init
        self.n_limit = n_limit
        self.levels_min = levels_min
        self.levels_max = levels_max
        self.theta = 0.5
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.gamma0 = gamma0
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        super(CubMLMC, self).__init__(
            allowed_distribs=[AbstractIIDDiscreteDistribution],
            allow_vectorized_integrals=False,
        )

    def _validate_resume(self, data):
        self._validate_resume_data(data, required_fields=self._RESUME_REQUIRED_FIELDS)
        self._validate_level_diffs(data)

    def _restore_resume_state(self, data):
        # Undo the final data.levels += 1 stored in the returned data so the
        # internal loop convention (0-indexed last level) is restored.
        data.levels -= 1
        # Recompute the optimal allocation for the new rmse_tol (may be tighter
        # or looser than the checkpoint tolerance).  With a looser tolerance
        # diff_n_level will be all-zero and the main loop exits immediately.
        n_samples = self._get_next_samples(data)
        data.diff_n_level = np.maximum(0, n_samples - data.n_level[: data.levels + 1])

    def _can_replay_resume_exactly(self, data):
        checkpoint_tol = self._checkpoint_rmse_tol(data)
        if checkpoint_tol is None or not (self.rmse_tol < checkpoint_tol):
            return False
        return hasattr(data, "level_diffs") and len(data.level_diffs) == len(data.n_level)

    @staticmethod
    def _update_trace_solution(data):
        valid = data.n_level > 0
        if np.any(valid):
            data.solution = (data.sum_level[0, valid] / data.n_level[valid]).sum()
        data.rmse_estimate = np.sqrt(
            (data.var_level / np.maximum(1, data.n_level[: len(data.var_level)])).sum()
        )

    def _run_integrate_loop(
        self,
        data,
        update_data_fn,
        trace=None,
        record_snapshots=False,
        t_start=None,
        resume_provenance=None,
    ):
        snapshots = []
        while data.diff_n_level.sum() > 0:
            update_data_fn(data)
            self._update_trace_solution(data)
            data.rmse_tol = self.rmse_tol
            if t_start is not None:
                self._set_elapsed_time(
                    data, time() - t_start, resume_provenance=resume_provenance
                )
            if record_snapshots:
                snapshots.append(copy.deepcopy(data))
            if trace is not None:
                trace.iteration(data, step_value=int(data.levels + 1))
            # set optimal number of additional samples
            n_samples = self._get_next_samples(data)
            data.diff_n_level = np.maximum(0, n_samples - data.n_level)
            # check if over sample budget
            if (data.n_total + data.diff_n_level.sum()) > self.n_limit:
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples, which would exceed n_limit = %d.
                Stopping integration process.
                Note that error tolerances may no longer be satisfied""" % (
                    int(data.n_total),
                    int(data.diff_n_level.sum()),
                    int(self.n_limit),
                )
                warnings.warn(warning_s, MaxSamplesWarning)
                break
            # if (almost) converged, estimate remaining error and decide
            # whether a new level is required
            if (data.diff_n_level > 0.01 * data.n_level).sum() == 0:
                range_ = np.arange(min(2.0, data.levels - 1) + 1)
                idx = (data.levels - range_).astype(int)
                rem = (data.mean_level[idx] / 2.0 ** (range_ * data.alpha)).max() / (
                    2.0**data.alpha - 1
                )
                if rem > np.sqrt(self.theta) * self.rmse_tol:
                    if data.levels == self.levels_max:
                        warnings.warn(
                            "Failed to achieve weak convergence. levels == levels_max.",
                            MaxLevelsWarning,
                        )
                    else:
                        self._add_level(data)
                        n_samples = self._get_next_samples(data)
                        data.diff_n_level = np.maximum(0, n_samples - data.n_level)
        data.solution = (data.sum_level[0, :] / data.n_level).sum()
        return snapshots

    def _replay_resume_exactly(self, checkpoint, t_start=None, resume_provenance=None):
        """Replay cached per-level diffs to reconstruct checkpoint state and
        trace rows.
        """
        shadow = self._construct_data()
        shadow.level_integrands = list(checkpoint.level_integrands)
        shadow.cached_level_diffs = [
            np.asarray(level_diffs, dtype=float) for level_diffs in checkpoint.level_diffs
        ]
        shadow.cached_level_positions = np.zeros(len(shadow.cached_level_diffs), dtype=int)
        snapshots = self._run_integrate_loop(
            shadow,
            self._update_replay_data,
            record_snapshots=True,
            t_start=t_start,
            resume_provenance=resume_provenance,
        )
        target_counts = np.asarray(checkpoint.n_level[: checkpoint.levels + 1], dtype=int)
        absorb_index = None
        for i, snapshot in enumerate(snapshots):
            if len(snapshot.n_level) < len(target_counts):
                continue
            if np.all(snapshot.n_level[: len(target_counts)] >= target_counts):
                absorb_index = i
                break
        if absorb_index is None:
            return None, None, None
        same_state = (
            np.array_equal(
                np.asarray(snapshots[absorb_index].n_level[: len(target_counts)]),
                target_counts,
            )
            and np.array_equal(
                np.asarray(snapshots[absorb_index].sum_level[:, : len(target_counts)]),
                np.asarray(checkpoint.sum_level[:, : len(target_counts)]),
            )
        )
        replay_iter_count = absorb_index + 1 if same_state else absorb_index
        for attr in ("cached_level_diffs", "cached_level_positions"):
            if hasattr(shadow, attr):
                delattr(shadow, attr)
        return shadow, snapshots[absorb_index:], replay_iter_count

    def integrate(self, resume: Union[None, Data] = None) -> tuple:
        """Run (or continue) the MLMC integration.

        Args:
            resume (Union[None, Data]): Checkpoint returned by a previous ``integrate()``
                call.  The new tolerance may be tighter *or* looser than the
                one used when the checkpoint was created. With a tighter
                tolerance the algorithm draws additional samples from where it
                left off.  With a looser tolerance the existing samples already
                satisfy the requirement and the method returns immediately with
                no new sampling.

        Returns:
            tuple: ``(solution, data)``.
        """
        t_start = time()
        resume_provenance = self._capture_resume_provenance(resume)
        trace = self._make_trace_logger()
        data = self._prepare_resume_data(resume, self._validate_resume, self._restore_resume_state)
        replay_snapshots = None
        replay_iter_count = None
        if self._can_replay_resume_exactly(data):
            replay_data, replay_snapshots, replay_iter_count = self._replay_resume_exactly(
                data, t_start=t_start, resume_provenance=resume_provenance
            )
            if replay_data is not None:
                data = replay_data
        if resume is not None:
            checkpoint = self._prepare_resume_data(
                resume, self._validate_resume, self._restore_resume_state
            )
            checkpoint.rmse_estimate = np.sqrt(
                (checkpoint.var_level / checkpoint.n_level[: len(checkpoint.var_level)]).sum()
            )
            if replay_iter_count is not None:
                checkpoint._iter_count = replay_iter_count
            self._set_elapsed_time(checkpoint, 0.0, resume_provenance=resume_provenance)
            trace.resume(checkpoint, step_value=int(checkpoint.levels + 1))
        if replay_snapshots is not None:
            for snapshot in replay_snapshots:
                trace.iteration(snapshot, step_value=int(snapshot.levels + 1))
            if replay_snapshots and trace.enabled:
                data._iter_count = trace.iter_count
        else:
            if data is None:
                data = self._construct_data()
            self._run_integrate_loop(
                data,
                self._update_data,
                trace=trace,
                t_start=t_start,
                resume_provenance=resume_provenance,
            )
        data.rmse_tol = self.rmse_tol
        data.levels += 1
        self._finalize_integration_data(
            data, time() - t_start, resume_provenance=resume_provenance
        )
        trace.finalize()
        data.iteration_history = getattr(self, "iteration_history", None)
        data.history_df = getattr(self, "history_df", None)
        return data.solution, data
