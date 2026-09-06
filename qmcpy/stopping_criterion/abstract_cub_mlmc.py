from .abstract_stopping_criterion import AbstractStoppingCriterion
from ..util.data import Data
from ..util import ParameterError
import numpy as np
from scipy.stats import norm


class AbstractCubMLMC(AbstractStoppingCriterion):

    @staticmethod
    def _append_level_diff_samples(data, level, dp):
        """Append raw level-difference samples when checkpoint caching is
        enabled.
        """
        if not hasattr(data, "level_diffs"):
            return
        while len(data.level_diffs) <= level:
            data.level_diffs.append(np.empty(0, dtype=float))
        data.level_diffs[level] = np.concatenate(
            [np.asarray(data.level_diffs[level], dtype=float), np.asarray(dp, dtype=float)]
        )

    def _refresh_level_statistics(self, data):
        """Recompute MLMC summary statistics after level updates."""
        # compute absolute average, variance and cost
        data.mean_level = np.absolute(
            data.sum_level[0, : data.levels + 1] / data.n_level[: data.levels + 1]
        )
        data.var_level = np.maximum(
            0,
            data.sum_level[1, : data.levels + 1] / data.n_level[: data.levels + 1]
            - data.mean_level**2,
        )
        data.cost_per_sample = (
            data.cost_level[: data.levels + 1] / data.n_level[: data.levels + 1]
        )
        # fix to cope with possible zero values for data.mean_level and data.var_level
        # (can happen in some applications when there are few samples)
        for l in range(2, data.levels + 1):
            data.mean_level[l] = np.maximum(
                data.mean_level[l], 0.5 * data.mean_level[l - 1] / 2**data.alpha
            )
            data.var_level[l] = np.maximum(
                data.var_level[l], 0.5 * data.var_level[l - 1] / 2**data.beta
            )
        # use linear regression to estimate alpha, beta, gamma if not given
        a = np.ones((data.levels, 2))
        a[:, 0] = np.arange(1, data.levels + 1)
        if self.alpha0 <= 0:
            x = np.linalg.lstsq(a, np.log2(data.mean_level[1:]), rcond=None)[0]
            data.alpha = np.maximum(0.5, -x[0])
        if self.beta0 <= 0:
            x = np.linalg.lstsq(a, np.log2(data.var_level[1:]), rcond=None)[0]
            data.beta = np.maximum(0.5, -x[0])
        if self.gamma0 <= 0:
            x = np.linalg.lstsq(a, np.log2(data.cost_per_sample[1:]), rcond=None)[0]
            data.gamma = np.maximum(0.5, x[0])
        data.n_total = data.n_level.sum()

    def _get_next_samples(self, data):
        ns = np.ceil(
            np.sqrt(data.var_level / data.cost_per_sample)
            * np.sqrt(data.var_level * data.cost_per_sample).sum()
            / ((1 - self.theta) * self.rmse_tol**2)
        )
        return ns.astype(int)

    def set_tolerance(self, abs_tol=None, rel_tol=None, rmse_tol=None):
        if not (rel_tol is None):
            raise AssertionError("rel_tol not supported by this stopping criterion.")
        if rmse_tol != None:
            self.rmse_tol = float(rmse_tol)
        elif abs_tol != None:
            self.rmse_tol = float(abs_tol) / norm.ppf(1 - self.alpha / 2.0)

    def _update_data(self, data):
        for l in range(data.levels + 1):
            if l == len(data.level_integrands):
                # haven't spawned this level's integrand yet
                data.level_integrands += self.integrand.spawn(levels=int(l))
            integrand_l = data.level_integrands[l]
            if data.diff_n_level[l] > 0:
                # evaluate integral at sampling points samples
                n = data.diff_n_level[l]
                samples = integrand_l.discrete_distrib(n=n)
                pc, pf = integrand_l.f(samples)
                dp = pf - pc
                self._append_level_diff_samples(data, l, dp)
                data.n_level[l] = data.n_level[l] + data.diff_n_level[l]
                data.sum_level[0, l] = data.sum_level[0, l] + dp.sum()
                data.sum_level[1, l] = data.sum_level[1, l] + (dp**2).sum()
                data.cost_level[l] = data.cost_level[l] + integrand_l.cost * n
        self._refresh_level_statistics(data)

    def _add_level(self, data):
        data.levels += 1
        if not len(data.n_level) > data.levels:
            data.mean_level = np.hstack(
                (data.mean_level, data.mean_level[-1] / 2**data.alpha)
            )
            data.var_level = np.hstack(
                (data.var_level, data.var_level[-1] / 2**data.beta)
            )
            data.cost_per_sample = np.hstack(
                (data.cost_per_sample, data.cost_per_sample[-1] * 2**data.gamma)
            )
            data.n_level = np.hstack((data.n_level, 0))
            data.sum_level = np.hstack((data.sum_level, np.zeros((2, 1))))
            data.cost_level = np.hstack((data.cost_level, 0.0))
            if hasattr(data, "level_diffs"):
                data.level_diffs.append(np.empty(0, dtype=float))
        else:
            data.mean_level = np.absolute(
                data.sum_level[0, : data.levels + 1] / data.n_level[: data.levels + 1]
            )
            data.var_level = np.maximum(
                0,
                data.sum_level[1, : data.levels + 1] / data.n_level[: data.levels + 1]
                - data.mean_level**2,
            )
            data.cost_per_sample = (
                data.cost_level[: data.levels + 1] / data.n_level[: data.levels + 1]
            )
            if hasattr(data, "level_diffs"):
                while len(data.level_diffs) <= data.levels:
                    data.level_diffs.append(np.empty(0, dtype=float))

    def _construct_data(self):
        """Build a fresh Data object for a new MLMC integration run.

        Returns:
            Initialised with zero sample counts and warm-up allocation.
        """
        data = Data(
            parameters=[
                "solution",
                "n_total",
                "levels",
                "n_level",
                "mean_level",
                "var_level",
                "cost_per_sample",
                "alpha",
                "beta",
                "gamma",
            ]
        )
        data.levels = int(self.levels_min)
        data.n_level = np.zeros(data.levels + 1, dtype=int)
        data.sum_level = np.zeros((2, data.levels + 1))
        data.cost_level = np.zeros(data.levels + 1)
        data.diff_n_level = self.n_init * np.ones(data.levels + 1, dtype=int)
        data.alpha = np.maximum(0, self.alpha0)
        data.beta = np.maximum(0, self.beta0)
        data.gamma = np.maximum(0, self.gamma0)
        data.level_integrands = []
        data.level_diffs = [np.empty(0, dtype=float) for _ in range(data.levels + 1)]
        return data

    @staticmethod
    def _validate_level_diffs(data):
        """Validate the ``level_diffs`` replay cache on a resume checkpoint.

        Args:
            data (Data): Resume checkpoint to validate.

        Raises:
            ParameterError: If ``level_diffs`` is present but structurally
                inconsistent with ``n_level``.
        """
        if not hasattr(data, "level_diffs"):
            return
        if len(data.level_diffs) != len(data.n_level):
            raise ParameterError(
                "resume data level_diffs length must match n_level length."
            )
        for level, (diffs, n_level) in enumerate(zip(data.level_diffs, data.n_level)):
            if len(np.asarray(diffs)) != int(n_level):
                raise ParameterError(
                    "resume data level_diffs[%d] length must match n_level[%d]."
                    % (level, level)
                )

    def _update_replay_data(self, data):
        """Replay cached level-difference samples, falling back to fresh
        draws.

        Used during exact-resume replay to reconstruct the integration state by
        consuming previously stored per-level samples before generating new
        ones.

        Args:
            data (Data): Integration state carrying ``cached_level_diffs`` and
                ``cached_level_positions`` replay bookkeeping attributes.
        """
        for l in range(data.levels + 1):
            if l == len(data.level_integrands):
                data.level_integrands += self.integrand.spawn(levels=int(l))
            if data.diff_n_level[l] <= 0:
                continue
            n_remaining = int(data.diff_n_level[l])
            if l < len(data.cached_level_diffs):
                cached = data.cached_level_diffs[l]
                pos = int(data.cached_level_positions[l])
                n_cached = min(n_remaining, len(cached) - pos)
                if n_cached > 0:
                    dp = cached[pos : pos + n_cached]
                    data.cached_level_positions[l] += n_cached
                    self._append_level_diff_samples(data, l, dp)
                    data.n_level[l] += n_cached
                    data.sum_level[0, l] += dp.sum()
                    data.sum_level[1, l] += (dp**2).sum()
                    data.cost_level[l] += data.level_integrands[l].cost * n_cached
                    n_remaining -= n_cached
            if n_remaining > 0:
                integrand_l = data.level_integrands[l]
                samples = integrand_l.discrete_distrib(n=n_remaining)
                pc, pf = integrand_l.f(samples)
                dp = pf - pc
                self._append_level_diff_samples(data, l, dp)
                data.n_level[l] += n_remaining
                data.sum_level[0, l] += dp.sum()
                data.sum_level[1, l] += (dp**2).sum()
                data.cost_level[l] += integrand_l.cost * n_remaining
        self._refresh_level_statistics(data)
