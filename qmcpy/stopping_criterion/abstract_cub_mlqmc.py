from typing import Union
from .abstract_stopping_criterion import AbstractStoppingCriterion
from ..util.data import Data
import numpy as np
from scipy.stats import norm


class AbstractCubMLQMC(AbstractStoppingCriterion):
    """Abstract base class for multilevel Quasi-Monte Carlo stopping criteria.

    Shared machinery for `CubMLQMC` and `CubMLQMCCont`: replication-based
    level statistics (`update_data`, `_update_bias_estimate`), level growth
    (`_add_level`), and resume checkpoint validation/replay used across MLQMC
    stopping criteria.
    """

    @staticmethod
    def _append_level_replication_sums(data, level, rep_sums, n_increment):
        """Append replayable per-replication sums for one MLQMC level update.
        """
        if (not hasattr(data, "level_rep_sums")) or (not hasattr(data, "level_n_increments")):
            return
        while len(data.level_rep_sums) <= level:
            data.level_rep_sums.append([])
        while len(data.level_n_increments) <= level:
            data.level_n_increments.append([])
        data.level_rep_sums[level].append(np.asarray(rep_sums, dtype=float))
        data.level_n_increments[level].append(int(n_increment))

    def _validate_level_replay_cache(self, data):
        if not hasattr(data, "level_rep_sums"):
            return
        if not hasattr(data, "level_n_increments"):
            raise ValueError("resume data level_n_increments missing while level_rep_sums is present.")
        if len(data.level_rep_sums) != len(data.n_level):
            raise ValueError("resume data level_rep_sums length must match n_level length.")
        if len(data.level_n_increments) != len(data.n_level):
            raise ValueError("resume data level_n_increments length must match n_level length.")
        for level, (rep_sums, n_increments) in enumerate(zip(data.level_rep_sums, data.level_n_increments)):
            if len(rep_sums) != len(n_increments):
                raise ValueError("resume data replay-cache lengths differ on level %d." % level)
            total = 0
            for rep_sum, n_increment in zip(rep_sums, n_increments):
                rep_sum = np.asarray(rep_sum)
                if rep_sum.shape != (int(self.replications),):
                    raise ValueError(
                        "resume data level_rep_sums[%d] entry shape %s is incompatible with replications=%d."
                        % (level, rep_sum.shape, int(self.replications))
                    )
                total += int(n_increment)
            if total != int(data.n_level[level]):
                raise ValueError(
                    "resume data replay-cache total %d does not match n_level[%d]=%d."
                    % (total, level, int(data.n_level[level]))
                )

    def _update_replay_data(self, data):
        for l in range(data.levels):
            if not data.eval_level[l]:
                continue
            if l == len(data.level_integrands):
                data.level_integrands += self.integrand.spawn(levels=l)
            n_max = self.n_init if data.n_level[l] == 0 else 2 * data.n_level[l]
            n_min = data.n_level[l]
            n = n_max - n_min
            pos = int(data.cached_level_positions[l])
            if pos >= len(data.cached_level_rep_sums[l]):
                raise ValueError("replay cache exhausted on level %d." % l)
            rep_sums = np.asarray(data.cached_level_rep_sums[l][pos], dtype=float)
            cached_n = int(data.cached_level_n_increments[l][pos])
            if cached_n != int(n):
                raise ValueError(
                    "replay cache increment %d on level %d is incompatible with requested increment %d."
                    % (cached_n, l, int(n))
                )
            data.cached_level_positions[l] += 1
            prev_sum = data.mean_level_reps[l] * data.n_level[l]
            data.mean_level_reps[l] = (rep_sums + prev_sum) / float(n_max)
            integrand_l = data.level_integrands[l]
            data.cost_level[l] += self.replications * n * integrand_l.cost
            data.n_level[l] = n_max
            data.mean_level[l] = data.mean_level_reps[l].mean()
            data.var_level[l] = data.mean_level_reps[l].var()
            cost_per_sample = data.cost_level[l] / data.n_level[l] / self.replications
            data.var_cost_ratio_level[l] = data.var_level[l] / cost_per_sample
        self._update_bias_estimate(data)
        data.n_total = self.replications * data.n_level.sum()
        data.solution = data.mean_level.sum()
        data.eval_level[:] = False

    @staticmethod
    def _resume_match_from_snapshots(snapshots, checkpoint):
        target_counts = np.asarray(checkpoint.n_level[: checkpoint.levels], dtype=int)
        for i, snapshot in enumerate(snapshots):
            if len(snapshot.n_level) < len(target_counts):
                continue
            if not np.all(snapshot.n_level[: len(target_counts)] >= target_counts):
                continue
            same_counts = np.array_equal(snapshot.n_level[: len(target_counts)], target_counts)
            same_means = (
                same_counts
                and np.array_equal(
                    np.asarray(snapshot.mean_level_reps[: checkpoint.levels]),
                    np.asarray(checkpoint.mean_level_reps[: checkpoint.levels]),
                )
            )
            resume_iter_count = i + 1 if same_means else i
            return resume_iter_count, snapshots[i:]
        return None, None

    def set_tolerance(self, abs_tol: Union[None, float] = None, rel_tol: Union[None, float] = None, rmse_tol: Union[None, float] = None) -> None:
        """Update the stopping criterion's target tolerance.

        Args:
            abs_tol (Union[None, float]): Absolute error tolerance, converted to an RMSE
                tolerance via `self.alpha`. Ignored if `rmse_tol` is supplied.
            rel_tol (Union[None, float]): Unsupported; must be `None`.
            rmse_tol (Union[None, float]): Root mean squared error tolerance. Takes
                precedence over `abs_tol` if both are supplied.

        Raises:
            AssertionError: If `rel_tol` is supplied.
        """
        if not (rel_tol is None):
            raise AssertionError("rel_tol not supported by this stopping criterion.")
        if rmse_tol != None:
            self.rmse_tol = float(rmse_tol)
        elif abs_tol != None:
            self.rmse_tol = float(abs_tol) / norm.ppf(1 - self.alpha / 2.0)

    def update_data(self, data: Data):
        """Double the sample count on every active level and refresh statistics.

        For each level with `data.eval_level[l]` set, doubles its replicated
        sample count (or draws `self.n_init` if the level is new), evaluates
        the paired coarse/fine integrand there, and folds the new
        per-replication sums into `data.mean_level_reps`, `data.mean_level`,
        `data.var_level`, and `data.var_cost_ratio_level`. Then refreshes the
        bias estimate, `data.n_total`, and `data.solution`, and clears
        `data.eval_level`.

        Args:
            data (Data): Integration state to update in place.
        """
        # update sample sums
        for l in range(data.levels):
            if not data.eval_level[l]:
                # nothing to do on this level
                continue
            if l == len(data.level_integrands):
                # haven't spawned this level's integrand yet
                data.level_integrands += self.integrand.spawn(levels=l)
            # reset dimension
            n_max = self.n_init if data.n_level[l] == 0 else 2 * data.n_level[l]
            integrand_l = data.level_integrands[l]
            n_min = data.n_level[l]
            samples = integrand_l.discrete_distrib(n_min=n_min, n_max=n_max)
            n = n_max - n_min
            pc, pf = integrand_l.f(samples).squeeze()
            dp = pf - pc
            dp_sums = np.asarray(dp.sum(-1), dtype=float)
            self._append_level_replication_sums(data, l, dp_sums, n)
            prev_sum = data.mean_level_reps[l] * data.n_level[l]
            data.mean_level_reps[l] = (dp_sums + prev_sum) / float(n_max)
            data.cost_level[l] = (
                data.cost_level[l] + self.replications * n * integrand_l.cost
            )
            data.n_level[l] = n_max
            data.mean_level[l] = data.mean_level_reps[l].mean()
            data.var_level[l] = data.mean_level_reps[l].var()
            cost_per_sample = data.cost_level[l] / data.n_level[l] / self.replications
            data.var_cost_ratio_level[l] = data.var_level[l] / cost_per_sample
        self._update_bias_estimate(data)
        data.n_total = self.replications * data.n_level.sum()
        data.solution = data.mean_level.sum()
        data.eval_level[:] = False  # Reset active levels

    def _update_bias_estimate(self, data):
        A = np.ones((2, 2))
        A[:, 0] = range(data.levels - 2, data.levels)
        y = np.ones(2)
        y[0] = np.log2(abs(data.mean_level_reps[data.levels - 2].mean()))
        y[1] = np.log2(abs(data.mean_level_reps[data.levels - 1].mean()))
        x = np.linalg.lstsq(A, y, rcond=None)[0]
        alpha = max(0.5, -x[0])
        data.bias_estimate = 2 ** (x[1] + data.levels * x[0]) / (2**alpha - 1)

    def _add_level(self, data):
        # Add another level to relevant attributes.
        data.levels += 1
        if data.levels > len(data.n_level):
            data.n_level = np.hstack((data.n_level, 0))
            data.eval_level = np.hstack((data.eval_level, True))
            data.mean_level_reps = np.vstack(
                (data.mean_level_reps, np.zeros(int(self.replications)))
            )
            data.mean_level = np.hstack((data.mean_level, 0))
            data.var_level = np.hstack((data.var_level, np.inf))
            data.cost_level = np.hstack((data.cost_level, 0))
            data.var_cost_ratio_level = np.hstack((data.var_cost_ratio_level, np.inf))
            if hasattr(data, "level_rep_sums"):
                data.level_rep_sums.append([])
            if hasattr(data, "level_n_increments"):
                data.level_n_increments.append([])

    def _add_level_MLMC(self, data):
        # Add another level to relevant attributes.
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
            data.n_level = np.hstack((data.n_level, 0.0))
            data.sum_level = np.hstack((data.sum_level, np.zeros((2, 1))))
            data.cost_level = np.hstack((data.cost_level, 0.0))
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
