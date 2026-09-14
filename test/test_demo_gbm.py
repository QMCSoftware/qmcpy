"""Unit tests for the GBM demo's sampling and statistics utilities.

Covers `demos/GBM/gbm_code/{quantlib_util,data_util}.py`. Classes and methods
are ordered by how much replication/randomization they exercise, from none
up to many.

Classes:
    TestCollectLibraryResultsStatistics: MAE/Std Dev Error math; no replication.
    TestQuantlibSeedIndependence: QuantLib's seed changes the scramble; 1 -> 2 -> 5 seeds.
    TestReplicationMeanIndependence: QuantLib replication-mean rank correlation, M=40.

QMCPy's own replication-independence tests (Sobol/Lattice, via DigitalNetB2
and Lattice) live in test/test_discrete_distribs.py, which exercises those
distribution classes directly rather than through this demo's wrapper.

Example:
    python3 -m pytest test/test_demo_gbm.py
"""
from itertools import combinations

import numpy as np
import pytest
from scipy.stats import spearmanr

qlu = pytest.importorskip("demos.GBM.gbm_code.quantlib_util")
du = pytest.importorskip("demos.GBM.gbm_code.data_util")

generate_quantlib_paths = qlu.generate_quantlib_paths

QUANTLIB_PARAMS = {
    "initial_value": 100.0,
    "mu": 0.05,
    "sigma": 0.2,
    "maturity": 1.0,
    "n_steps": 4,
    "n_paths": 8,
}


def _quantlib_paths(seed, sampler_type="Sobol", **overrides):
    """Returns generate_quantlib_paths(...) paths only, with QUANTLIB_PARAMS as defaults."""
    paths, _ = generate_quantlib_paths(
        **{**QUANTLIB_PARAMS, **overrides}, sampler_type=sampler_type, seed=seed,
    )
    return paths


def _assert_distinct(paths_by_replication):
    """Asserts no two entries in `paths_by_replication` are identical."""
    for a, b in combinations(paths_by_replication, 2):
        assert not np.array_equal(a, b), "replications produced identical paths"


class TestCollectLibraryResultsStatistics:
    """Verifies Mean/Std Dev/MAE/Std Dev Error computed by collect_library_results().

    Mocks both libraries' path generators with known arrays so the reported
    statistics can be checked exactly; no replication (each generator is
    called once per test).

    Note:
        Each test's `monkeypatch` fixture replaces `du.qlu.generate_quantlib_paths`
        and/or `du.qpu.generate_qmcpy_paths` with a lambda returning a fixed array,
        isolating the arithmetic from actual (randomized) sampling. pytest reverts
        the patch automatically after each test.
    """

    TIMING = {
        "Sobol": {"average": 0.1, "stdev": 0.01},
        "Lattice": {"average": 0.1, "stdev": 0.01},
    }
    THEORETICAL_MEAN = 2.5
    THEORETICAL_STD = 0.75

    def test_quantlib_row_statistics(self, monkeypatch):
        """Checks the QuantLib results row against hand-computed statistics."""
        # QuantLib paths are always 2D: (n_paths, n_steps + 1).
        ql_paths = np.array([[100.0, 1.0], [100.0, 2.0], [100.0, 3.0]])
        monkeypatch.setattr(
            du.qlu, "generate_quantlib_paths", lambda **kwargs: (ql_paths, None)
        )
        monkeypatch.setattr(
            du.qpu, "generate_qmcpy_paths", lambda **kwargs: (np.zeros((1, 2, 2)), None)
        )

        results = du.collect_library_results(
            "Sobol", "Paths", 2, 3, self.TIMING, self.TIMING,
            self.THEORETICAL_MEAN, self.THEORETICAL_STD,
        )

        row = next(r for r in results if r["Method"] == "QuantLib")
        assert row["Mean"] == pytest.approx(2.0)
        assert row["Std Dev"] == pytest.approx(1.0)
        assert row["Mean Absolute Error"] == pytest.approx(0.5)
        assert row["Std Dev Error"] == pytest.approx(0.25)

    def test_qmcpy_row_statistics(self, monkeypatch):
        """Checks the QMCPy results row; one replication so pooled and per-replication stats coincide."""
        qp_paths = np.array([[[100.0, 1.0], [200.0, 2.0], [300.0, 3.0]]])
        monkeypatch.setattr(
            du.qpu, "generate_qmcpy_paths", lambda **kwargs: (qp_paths, None)
        )

        results = du.collect_library_results(
            "Lattice", "Paths", 2, 3, {}, self.TIMING,
            self.THEORETICAL_MEAN, self.THEORETICAL_STD,
        )

        assert len(results) == 1
        row = results[0]
        assert row["Method"] == "QMCPy"
        assert row["Mean"] == pytest.approx(2.0)
        assert row["Std Dev"] == pytest.approx(1.0)
        assert row["Mean Absolute Error"] == pytest.approx(0.5)
        assert row["Std Dev Error"] == pytest.approx(0.25)

    def test_qmcpy_terminal_axis(self, monkeypatch):
        """Regression guard for qp_paths[:, -1] vs qp_paths[..., -1]: a marker value placed
        only at the true terminal (last) axis catches an off-by-axis regression."""
        qp_paths = np.array([[[1.0, 1.0, 999.0], [1.0, 1.0, 999.0]]])
        monkeypatch.setattr(
            du.qpu, "generate_qmcpy_paths", lambda **kwargs: (qp_paths, None)
        )

        results = du.collect_library_results(
            "Lattice", "Paths", 3, 2, {}, self.TIMING, 999.0, 0.0
        )

        row = results[0]
        assert row["Mean"] == pytest.approx(999.0)
        assert row["Std Dev"] == pytest.approx(0.0)


class TestQuantlibSeedIndependence:
    """QuantLib's `seed` must actually change the Sobol scramble.

    demos/GBM/gbm_code/data_util.py:process_sampler_data() builds replications
    by calling generate_quantlib_paths() once per replication with
    seed = base_seed + r; it previously didn't work because
    UniformLowDiscrepancySequenceGenerator ignores its seed argument for the
    fixed Jaeckel direction integers. Methods progress from a single seed to
    a 5-seed loop mirroring that replication pattern.
    """

    @pytest.mark.parametrize("sampler_type", ["IIDStdUniform", "Sobol"])
    def test_shape_and_values(self, sampler_type):
        """Checks output shape, initial value, and finiteness for a single seed."""
        paths = _quantlib_paths(7, sampler_type)
        assert paths.shape == (QUANTLIB_PARAMS["n_paths"], QUANTLIB_PARAMS["n_steps"] + 1)
        np.testing.assert_array_equal(paths[:, 0], QUANTLIB_PARAMS["initial_value"])
        assert np.isfinite(paths).all()

    def test_vectorized_matches_evolve(self):
        """Checks the vectorized Sobol evolution matches QuantLib's own evolve(), step by step.

        The Sobol branch vectorizes Euler evolution with numpy instead of
        calling QuantLib's path generator per path.
        """
        paths, gbm = generate_quantlib_paths(**QUANTLIB_PARAMS, sampler_type="Sobol", seed=7)
        ql = qlu.ql
        times = ql.TimeGrid(QUANTLIB_PARAMS["maturity"], QUANTLIB_PARAMS["n_steps"])
        uniform_rsg = ql.Burley2020SobolRsg(
            QUANTLIB_PARAMS["n_steps"], 0, ql.SobolRsg.Jaeckel, 7
        )
        gaussian_rsg = ql.InvCumulativeBurley2020SobolGaussianRsg(uniform_rsg)

        expected = np.empty_like(paths)
        expected[:, 0] = QUANTLIB_PARAMS["initial_value"]
        for i in range(QUANTLIB_PARAMS["n_paths"]):
            normals = gaussian_rsg.nextSequence().value()
            for j in range(1, QUANTLIB_PARAMS["n_steps"] + 1):
                t0 = times[j - 1]
                expected[i, j] = gbm.evolve(
                    t0, expected[i, j - 1], times[j] - t0, normals[j - 1]
                )

        np.testing.assert_allclose(paths, expected, rtol=2e-15, atol=0)

    def test_unknown_sampler_raises(self):
        """Checks that an unsupported sampler_type raises ValueError (single call)."""
        with pytest.raises(ValueError, match="Unsupported sampler type"):
            _quantlib_paths(1, sampler_type="unknown")

    @pytest.mark.parametrize("sampler_type", ["IIDStdUniform", "Sobol"])
    def test_seed_reproducible_effective(self, sampler_type):
        """Same seed reproduces paths; a different seed changes them (2 seeds)."""
        np.testing.assert_array_equal(
            _quantlib_paths(7, sampler_type), _quantlib_paths(7, sampler_type)
        )
        assert not np.array_equal(_quantlib_paths(7, sampler_type), _quantlib_paths(8, sampler_type))

    @pytest.mark.parametrize("sampler_type", ["IIDStdUniform", "Sobol"])
    def test_seed_loop_distinct(self, sampler_type):
        """Sequential seeds (mirroring the replication loop) are pairwise distinct (5 seeds)."""
        _assert_distinct([_quantlib_paths(7 + r, sampler_type) for r in range(5)])


class TestReplicationMeanIndependence:
    """QuantLib's per-replication mean statistics show no rank correlation
    across the replication/seed index -- a stronger check than "not bit-identical".

    Note:
        Checked on the *replication-level mean* terminal value, not raw
        matched-index points: low-discrepancy points are structured by
        construction, so comparing point i of one scramble to point i of
        another can show large incidental correlation even between
        genuinely independent randomizations. What process_sampler_data()
        actually relies on (per RQMC confidence interval theory) is that
        the *replication means* are independent, which is what's checked
        here. (QMCPy's analog is tested on DigitalNetB2 in
        test/test_discrete_distribs.py.)
    """

    M = 40
    N_PATHS = 64    # kept small for speed; large enough for a stable mean
    # SE of Spearman's rho under independence is ~1/sqrt(M-2) =~ 0.16 here,
    # so this threshold is a >2 sigma margin without being fragile.
    RHO_THRESHOLD = 0.5

    def test_quantlib_replication_means_uncorrelated(self):
        """Checks lag-1 rank correlation of QuantLib per-replication means is small."""
        means = np.array([
            _quantlib_paths(7 + r, n_paths=self.N_PATHS)[:, -1].mean() for r in range(self.M)
        ])
        assert means.std() > 0, "replication means are constant -- seed has no effect"
        rho, _ = spearmanr(means[:-1], means[1:])
        assert abs(rho) < self.RHO_THRESHOLD
