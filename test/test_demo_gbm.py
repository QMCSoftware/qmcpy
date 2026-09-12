"""Unit tests for the GBM demo's sampling and statistics utilities.

Covers `demos/GBM/gbm_code/{quantlib_util,data_util,plot_util}.py`. Classes and methods
are ordered by how much replication/randomization they exercise, from none
up to many.

Classes:
    TestCollectLibraryResultsStatistics: MAE/Std Dev Error math; 1--2 replications.
    TestQuantlibSeedIndependence: QuantLib's seed changes the scramble; 1 -> 2 -> 5 seeds.
    TestQuantlibSchemes: exact vs Euler evolution and high-dimensional
        reproducibility; no replication.
    TestQmcpySamplerSettings: sampler construction options; no replication.
    TestConstructionAblation: path-construction ablation; several replications.
    TestReplicationMeanIndependence: QuantLib replication-mean rank correlation, M=40.

QMCPy's own replication-independence tests (Sobol/Lattice, via DigitalNetB2
and Lattice) live in test/test_discrete_distribs.py, which exercises those
distribution classes directly rather than through this demo's wrapper.

Example:
    python3 -m pytest test/test_demo_gbm.py
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless-safe; must precede plot_util's `import matplotlib.pyplot`

cf = pytest.importorskip("demos.GBM.gbm_code.config")
qlu = pytest.importorskip("demos.GBM.gbm_code.quantlib_util")
qpu = pytest.importorskip("demos.GBM.gbm_code.qmcpy_util")
du = pytest.importorskip("demos.GBM.gbm_code.data_util")
pu = pytest.importorskip("demos.GBM.gbm_code.plot_util")
lu = pytest.importorskip("demos.GBM.gbm_code.latex_util")

generate_quantlib_paths = qlu.generate_quantlib_paths

QUANTLIB_PARAMS = {
    "initial_value": 100.0,
    "mu": 0.05,
    "sigma": 0.2,
    "maturity": 1.0,
    "n_steps": 4,
    "n_paths": 8,
}


def test_comparison_sampler_order():
    """Shared samplers precede the QMCPy-only Lattice in comparison plots."""
    samplers = cf.get_sampler_configurations()
    assert samplers["all_samplers"] == [
        "IIDStdUniform",
        "Sobol",
        "Halton",
        "Lattice",
    ]


def test_extracted_comparison_data_uses_sampler_order():
    """Bar-plot data are ordered even when input rows are not."""
    rows = [
        {"Method": "Theoretical", "Sampler": "-", "Mean": 1.0},
        *[
            {
                "Method": "QMCPy",
                "Sampler": sampler,
                "Mean Absolute Error": 1.0,
                "Std Dev Error": 1.0,
            }
            for sampler in ["Lattice", "Halton", "Sobol", "IIDStdUniform"]
        ],
    ]
    samplers, *_ = du.extract_comparison_data(pd.DataFrame(rows))
    assert samplers.tolist() == ["IIDStdUniform", "Sobol", "Halton", "Lattice"]


def test_line_plot_uses_sampler_order():
    """Cross-library line plots put Halton before Lattice."""
    plot_data = pd.DataFrame(
        {
            "Series": ["Paths"] * 4,
            "Method": ["QMCPy"] * 4,
            "Sampler": ["Lattice", "Halton", "Sobol", "IIDStdUniform"],
            "n_paths": [16] * 4,
            "MAE": [1.0] * 4,
        }
    )
    fig, ax = pu.plt.subplots()
    pu.plot_single_series(
        ax, plot_data, "Paths", "n_paths", "MAE", "MAE", "Paths", "MAE"
    )

    # QMCPy series name their construction, since QMCPy defaults to PCA while
    # QuantLib fills time steps sequentially.
    assert [line.get_label() for line in ax.lines] == [
        "QMCPy (PCA) - IIDStdUniform",
        "QMCPy (PCA) - Sobol",
        "QMCPy (PCA) - Halton",
        "QMCPy (PCA) - Lattice",
    ]
    pu.plt.close(fig)


def test_interactive_plot_replaces_output_after_slider_release():
    """The GBM widget redraws in one cleared output after slider release."""
    notebook_path = Path(__file__).parents[1] / "demos/GBM/gbm_demo.ipynb"
    notebook = json.loads(notebook_path.read_text())
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    widget_source = next(source for source in sources if "gbm_display = {'handle': None}" in source)
    plot_source = next(
        source for source in sources if "def plot_gbm_paths_with_distribution" in source
    )

    assert widget_source.count("continuous_update=False") == 5
    assert "display(fig, display_id=True)" in widget_source
    assert "gbm_display['handle'].update(fig)" in widget_source
    assert "plt.close(fig)" in widget_source
    assert "return fig" in plot_source
    assert "plt.show()" not in plot_source


def test_comparison_line_styles():
    """QMCPy curves are heavier and dashed in cross-library plots."""
    lines = pu.get_plot_styling()["lines"]
    assert lines["QuantLib"] == {"linestyle": "-", "linewidth": 2}
    assert lines["QMCPy"] == {"linestyle": "--", "linewidth": 3}


def test_error_comparison_annotations_and_replications():
    """Matched error bars show ratios; QMCPy-only samplers are not annotated."""
    fig, ax = pu.plt.subplots()
    pu.plot_error_comparison(
        ax,
        ["IIDStdUniform", "Sobol", "Lattice"],
        np.array([1.0, 1.0, 1.0]),
        [2.0, 4.0, None],
        replications=8,
    )

    assert ax.get_title() == "MAE\n(8-replication average)"
    assert [annotation.get_text() for annotation in ax.texts] == [
        "2.0x lower",
        "4.0x lower",
    ]
    assert all(annotation.arrow_patch is not None for annotation in ax.texts)
    pu.plt.close(fig)


def test_performance_annotations_distinguish_faster_and_slower():
    """Runtime ratios describe QMCPy's direction relative to QuantLib."""
    fig, ax = pu.plt.subplots()
    pu.plot_performance_comparison(
        ax,
        ["IIDStdUniform", "Sobol"],
        np.array([1.0, 4.0]),
        [2.0, 2.0],
    )

    assert [annotation.get_text() for annotation in ax.texts] == [
        "2.0x faster",
        "2.0x slower",
    ]
    pu.plt.close(fig)


def test_estimates_are_rounded_with_replication_uncertainty():
    """Table estimates carry an SE and do not imply unsupported precision."""
    df = pd.DataFrame({"Mean": [105.123456], "Mean SE": [0.037]})
    formatted = lu.format_results_dataframe(
        df, ["Mean", "Mean SE"], {"Mean": "Mean SE"}
    )
    assert formatted.loc[0, "Mean"] == "105.123 (0.037)"
    assert "Mean SE" not in formatted


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
    statistics can be checked exactly.

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
        assert row["Std Dev"] == pytest.approx(np.sqrt(2 / 3))
        assert row["Mean Absolute Error"] == pytest.approx(0.5)
        assert row["Std Dev Error"] == pytest.approx(np.sqrt(2 / 3) - 0.75)

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
        assert row["Std Dev"] == pytest.approx(np.sqrt(2 / 3))
        assert row["Mean Absolute Error"] == pytest.approx(0.5)
        assert row["Std Dev Error"] == pytest.approx(np.sqrt(2 / 3) - 0.75)

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

    def test_sweep_errors_are_averaged_over_replications(self, monkeypatch):
        """Checks both sweep errors use every replication for both libraries."""
        terminal = np.array([[1.0, 2.0, 3.0], [3.0, 5.0, 7.0]])

        def ql_paths(seed, **kwargs):
            values = terminal[seed - cf.QUANTLIB_SEED]
            return np.column_stack((np.zeros(3), values)), None

        qp_paths = np.stack(
            (np.zeros_like(terminal), terminal), axis=-1
        )
        monkeypatch.setattr(du.qlu, "generate_quantlib_paths", ql_paths)
        monkeypatch.setattr(
            du.qpu, "generate_qmcpy_paths", lambda **kwargs: (qp_paths, None)
        )

        results = du.collect_library_results(
            "Sobol", "Paths", 1, 3, self.TIMING, self.TIMING,
            theoretical_mean=3.0, theoretical_std=1.5, replications=2,
        )

        for row in results:
            assert row["Mean"] == pytest.approx(3.5)
            assert row["Std Dev"] == pytest.approx(np.sqrt(3 / 2))
            assert row["Mean Absolute Error"] == pytest.approx(1.5)
            assert row["Std Dev Error"] == pytest.approx(1 / np.sqrt(6))
            assert row["Mean SE"] == pytest.approx(1.5)
            assert row["MAE SE"] == pytest.approx(0.5)

    def test_process_sampler_preserves_quantlib_replications(self, monkeypatch):
        """Covariance callers receive every QuantLib replication, not only the last."""
        def ql_paths(seed, **kwargs):
            terminal = np.array([seed, seed + 1.0])
            return np.column_stack((np.zeros(2), terminal)), None

        qp_paths = np.zeros((2, 2, 1))
        monkeypatch.setattr(du.qlu, "generate_quantlib_paths", ql_paths)
        monkeypatch.setattr(
            du.qpu, "generate_qmcpy_paths", lambda **kwargs: (qp_paths, None)
        )
        params_ql = {
            "initial_value": 100.0,
            "mu": 0.05,
            "sigma": 0.2,
            "maturity": 1.0,
            "n_steps": 1,
            "n_paths": 2,
            "seed": 7,
        }
        params_qp = {
            "initial_value": 100.0,
            "mu": 0.05,
            "diffusion": 0.04,
            "maturity": 1.0,
            "n_steps": 1,
            "n_paths": 2,
            "replications": 2,
        }

        quantlib_paths, *_ = du.process_sampler_data(
            "Sobol", [], 0.0, 1.0, params_ql, params_qp
        )
        assert quantlib_paths.shape == (2, 2, 2)
        np.testing.assert_array_equal(quantlib_paths[:, 0, -1], [7.0, 8.0])

    def test_covariance_uses_actual_time_grid_and_all_replications(self):
        """Requested times select matching coordinates on either path convention."""
        ql_grid = np.linspace(0.0, 2.0, 5)
        qp_grid = ql_grid[1:]
        ql_paths = np.arange(2 * 3 * 5, dtype=float).reshape(2, 3, 5)
        qp_paths = ql_paths[..., 1:]

        ql_average, ql_covariances = du.extract_covariance_samples(
            ql_paths, ql_grid, (1.0, 2.0)
        )
        qp_average, qp_covariances = du.extract_covariance_samples(
            qp_paths, qp_grid, (1.0, 2.0)
        )
        np.testing.assert_allclose(ql_average, qp_average)
        np.testing.assert_allclose(ql_covariances, qp_covariances)
        assert ql_covariances.shape == (2, 2, 2)


class TestQuantlibSeedIndependence:
    """QuantLib's `seed` must actually change the Sobol scramble.

    demos/GBM/gbm_code/data_util.py:process_sampler_data() builds replications
    by calling generate_quantlib_paths() once per replication with
    seed = base_seed + r; it previously didn't work because
    UniformLowDiscrepancySequenceGenerator ignores its seed argument for the
    fixed Jaeckel direction integers. Methods progress from a single seed to
    a 5-seed loop mirroring that replication pattern.
    """

    @pytest.mark.parametrize("sampler_type", ["IIDStdUniform", "Sobol", "Halton"])
    def test_shape_and_values(self, sampler_type):
        """Checks output shape, initial value, and finiteness for a single seed."""
        paths = _quantlib_paths(7, sampler_type)
        assert paths.shape == (QUANTLIB_PARAMS["n_paths"], QUANTLIB_PARAMS["n_steps"] + 1)
        np.testing.assert_array_equal(paths[:, 0], QUANTLIB_PARAMS["initial_value"])
        assert np.isfinite(paths).all()

    @pytest.mark.parametrize("sampler_type", ["Sobol", "Halton"])
    @pytest.mark.parametrize("scheme", ["exact", "euler"])
    def test_vectorized_matches_evolve(self, sampler_type, scheme):
        """Checks the vectorized evolution matches QuantLib's own evolve(), step by step.

        The low-discrepancy branches vectorize the evolution with numpy instead
        of calling QuantLib's path generator per path, for both the exact and
        the Euler-Maruyama scheme.
        """
        paths, process = generate_quantlib_paths(
            **QUANTLIB_PARAMS, sampler_type=sampler_type, seed=7, scheme=scheme
        )
        ql = qlu.ql
        times = ql.TimeGrid(QUANTLIB_PARAMS["maturity"], QUANTLIB_PARAMS["n_steps"])
        normals = qlu._low_discrepancy_normals(
            sampler_type, QUANTLIB_PARAMS["n_steps"], QUANTLIB_PARAMS["n_paths"], 7
        )

        expected = np.empty_like(paths)
        expected[:, 0] = QUANTLIB_PARAMS["initial_value"]
        for i in range(QUANTLIB_PARAMS["n_paths"]):
            for j in range(1, QUANTLIB_PARAMS["n_steps"] + 1):
                t0 = times[j - 1]
                expected[i, j] = process.evolve(
                    t0, expected[i, j - 1], times[j] - t0, normals[i, j - 1]
                )

        np.testing.assert_allclose(paths, expected, rtol=2e-15, atol=0)

    def test_unknown_sampler_raises(self):
        """Checks that an unsupported sampler_type raises ValueError (single call)."""
        with pytest.raises(ValueError, match="Unsupported sampler type"):
            _quantlib_paths(1, sampler_type="unknown")

    @pytest.mark.parametrize("sampler_type", ["IIDStdUniform", "Sobol", "Halton"])
    def test_seed_reproducible_effective(self, sampler_type):
        """Same seed reproduces paths; a different seed changes them (2 seeds)."""
        np.testing.assert_array_equal(
            _quantlib_paths(7, sampler_type), _quantlib_paths(7, sampler_type)
        )
        assert not np.array_equal(_quantlib_paths(7, sampler_type), _quantlib_paths(8, sampler_type))

    @pytest.mark.parametrize("sampler_type", ["IIDStdUniform", "Sobol", "Halton"])
    def test_seed_loop_distinct(self, sampler_type):
        """Sequential seeds (mirroring the replication loop) are pairwise distinct (5 seeds)."""
        _assert_distinct([_quantlib_paths(7 + r, sampler_type) for r in range(5)])


class TestQmcpySamplerSettings:
    """Checks the QMCPy sampler options the demo relies on for speed.

    No replication: each method constructs samplers and inspects them.
    """

    def test_halton_trims_digit_array(self):
        """Checks Halton is built with the reduced digit count from config.

        Halton is a general-base digital net, so its cost scales with the
        (n, d, t) digit array; the demo halves t rather than falling back to a
        cheaper, less accurate randomization.
        """
        halton = qpu.create_qmcpy_sampler("Halton", 252)
        assert halton.t == cf.HALTON_DIGITS
        assert cf.HALTON_DIGITS < 63
        assert "LMS" in halton.randomize and "DP" in halton.randomize

    @pytest.mark.parametrize("sampler_type", ["IIDStdUniform", "Sobol", "Lattice"])
    def test_other_samplers_keep_defaults(self, sampler_type):
        """Checks the digit-count override is not applied to the base-2 samplers."""
        sampler = qpu.create_qmcpy_sampler(sampler_type, 8)
        assert getattr(sampler, "t", 63) == 63

    def test_unknown_sampler_raises(self):
        """Checks that an unsupported sampler_type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported sampler type"):
            qpu.create_qmcpy_sampler("Faure", 8)


class TestQuantlibSchemes:
    """Covers the 'exact' vs 'euler' evolution schemes and seeding at high dimension.

    No replication: every method compares single deterministic calls.
    """

    def test_exact_matches_closed_form(self):
        """Checks the exact scheme reproduces S_0*exp((mu-sigma^2/2)t + sigma*W_t)."""
        params = {**QUANTLIB_PARAMS, "n_steps": 8}
        paths, _ = generate_quantlib_paths(
            **params, sampler_type="Sobol", seed=7, scheme="exact"
        )
        normals = qlu._low_discrepancy_normals(
            "Sobol", params["n_steps"], params["n_paths"], 7
        )
        dt = params["maturity"] / params["n_steps"]
        drift = (params["mu"] - 0.5 * params["sigma"] ** 2) * dt
        brownian = params["sigma"] * np.sqrt(dt) * np.cumsum(normals, axis=1)
        expected = params["initial_value"] * np.exp(
            drift * np.arange(1, params["n_steps"] + 1) + brownian
        )

        np.testing.assert_allclose(paths[:, 1:], expected, rtol=1e-14, atol=0)

    def test_exact_and_euler_differ(self):
        """Checks the two schemes are actually different evolutions."""
        common = dict(**QUANTLIB_PARAMS, sampler_type="Sobol", seed=7)
        exact, _ = generate_quantlib_paths(**common, scheme="exact")
        euler, _ = generate_quantlib_paths(**common, scheme="euler")
        assert not np.allclose(exact, euler)

    def test_unknown_scheme_raises(self):
        """Checks that an unsupported scheme raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported scheme"):
            generate_quantlib_paths(
                **QUANTLIB_PARAMS, sampler_type="Sobol", seed=7, scheme="milstein"
            )

    def test_reproducible_above_direction_integer_table(self, monkeypatch):
        """Checks reproducibility past dimension 32, where the Jaeckel table ends.

        Regression test: QuantLib fills direction integers beyond the tabulated
        dimensions from a clock-seeded generator when the Sobol' seed is 0, so
        the demo's 252-step paths were not reproducible across runs. Only the
        scramble seed may vary between replications.
        """
        constructor = qlu.ql.Burley2020SobolRsg
        calls = []

        def record_constructor(*args):
            calls.append(args)
            return constructor(*args)

        monkeypatch.setattr(qlu.ql, "Burley2020SobolRsg", record_constructor)
        params = {**QUANTLIB_PARAMS, "n_steps": 64}
        first = _quantlib_paths(7, "Sobol", n_steps=params["n_steps"])
        second = _quantlib_paths(7, "Sobol", n_steps=params["n_steps"])
        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(
            first, _quantlib_paths(8, "Sobol", n_steps=params["n_steps"])
        )
        assert calls
        assert all(args[1] == cf.SOBOL_DIRECTION_SEED != 0 for args in calls)


class TestConstructionAblation:
    """Covers run_construction_ablation(), which varies only `decomp_type`.

    The ablation's whole value rests on holding the point set fixed, so these
    tests check that structure rather than the accuracy ordering, which is a
    statistical outcome and would make a flaky assertion.
    """

    PARAMS = dict(
        theoretical_mean=105.127109637,
        theoretical_std=21.237438824,
        n_paths=2**6,
        n_steps=8,
        replications=2,
    )

    def test_shape_and_columns(self):
        """Checks one row per (sampler, construction) with the expected columns."""
        df = du.run_construction_ablation(
            ["IIDStdUniform", "Sobol"], ["PCA", "Cholesky"], **self.PARAMS
        )
        assert list(df.columns) == [
            "Sampler", "Construction", "Mean Absolute Error", "Std Dev Error",
            "Runtime (s)",
        ]
        assert len(df) == 4
        assert set(df["Construction"]) == {"PCA", "Cholesky"}
        assert np.isfinite(df["Mean Absolute Error"]).all()
        assert (df["Runtime (s)"] > 0).all()

    def test_constructions_differ_for_low_discrepancy(self):
        """Checks `decomp_type` actually reaches the sampler.

        If the argument were dropped somewhere in the call chain, every
        construction would return identical errors and the ablation would be
        silently meaningless.
        """
        df = du.run_construction_ablation(
            ["Sobol"], ["PCA", "Cholesky", "BrownianBridge"], **self.PARAMS
        )
        errors = df["Mean Absolute Error"].tolist()
        assert len(set(errors)) == len(errors)

    def test_same_seed_reproduces(self):
        """Checks the ablation is deterministic, so runs are comparable.

        Excludes 'Runtime (s)', a wall-clock measurement that is never
        bit-reproducible between calls.
        """
        first = du.run_construction_ablation(["Sobol"], ["PCA"], **self.PARAMS)
        second = du.run_construction_ablation(["Sobol"], ["PCA"], **self.PARAMS)
        drop_cols = ["Runtime (s)"]
        pd.testing.assert_frame_equal(
            first.drop(columns=drop_cols), second.drop(columns=drop_cols)
        )

    def test_constructions_agree_on_the_law(self):
        """Checks all constructions describe the same process.

        They reorder which coordinate drives which part of the path; they must
        not change the distribution of S_T, so every construction's mean should
        sit near the theoretical value.
        """
        df = du.run_construction_ablation(
            ["IIDStdUniform"], ["PCA", "Cholesky", "BrownianBridge"],
            **{**self.PARAMS, "n_paths": 2**12},
        )
        monte_carlo_margin = 5 * self.PARAMS["theoretical_std"] / np.sqrt(2**12)
        assert (df["Mean Absolute Error"] < monte_carlo_margin).all()


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
