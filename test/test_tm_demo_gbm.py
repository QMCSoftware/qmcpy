"""Unit tests for the GBM demo's sampling and statistics utilities.

Covers `demos/GBM/gbm_code/{quantlib_util,data_util,plot_util}.py`. Classes and methods
are ordered by how much replication/randomization they exercise, from none
up to many.

Classes:
    TestDemoPresentation: sampler ordering, plots, and table formatting.
    TestCollectLibraryResultsStatistics: MAE/Std Dev Error math; 1--2 replications.
    TestQuantlibSeedIndependence: QuantLib's seed changes the scramble; 1 -> 2 -> 5 seeds.
    TestQuantlibSchemes: exact vs Euler evolution and high-dimensional
        reproducibility; no replication.
    TestQmcpySamplerSettings: sampler options and bounded-memory Halton paths.
    TestConstructionAblation: path-construction ablation; several replications.
    TestReplicationMeanIndependence: QuantLib replication-mean rank correlation, M=40.

QMCPy's own replication-independence tests (Sobol/Lattice, via DigitalNetB2
and Lattice) live in test/test_dd_discrete_distribs.py, which exercises those
distribution classes directly rather than through this demo's wrapper.

Example:
    python3 -m pytest test/test_tm_demo_gbm.py
"""
import ast
import gc
import json
import weakref
from itertools import combinations
from pathlib import Path
from unittest import TestCase, mock

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from qmcpy.util import ParameterError

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


class TestDemoPresentation(TestCase):
    """Checks sampler ordering, plots, and formatted tables in the demo."""

    def test_comparison_sampler_order(self):
        """Shared samplers precede the QMCPy-only Lattice in comparison plots."""
        samplers = cf.get_sampler_configurations()
        assert samplers["all_samplers"] == [
            "IIDStdUniform",
            "Sobol",
            "Halton",
            "Lattice",
        ]

    def test_extracted_comparison_data_uses_sampler_order(self):
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

    def test_line_plot_uses_sampler_order(self):
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

    def test_slider_release_replaces_output(self):
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

    def test_comparison_line_styles(self):
        """QMCPy curves are heavier and dashed in cross-library plots."""
        lines = pu.get_plot_styling()["lines"]
        assert lines["QuantLib"] == {"linestyle": "-", "linewidth": 2}
        assert lines["QMCPy"] == {"linestyle": "--", "linewidth": 3}

    def test_error_comparison_annotations_and_replications(self):
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

    def test_runtime_ratio_direction(self):
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

    def test_estimate_rounding_with_uncertainty(self):
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


class TestCollectLibraryResultsStatistics(TestCase):
    """Verifies Mean/Std Dev/MAE/Std Dev Error computed by collect_library_results().

    Mocks both libraries' path generators with known arrays so the reported
    statistics can be checked exactly.

    Note:
        Context-managed patches replace `du.qlu.generate_quantlib_paths`
        and/or `du.qpu.generate_qmcpy_paths` with a lambda returning a fixed array,
        isolating the arithmetic from actual (randomized) sampling. Each patch
        is restored automatically when its context exits.
    """

    TIMING = {
        "Sobol": {"average": 0.1, "stdev": 0.01},
        "Lattice": {"average": 0.1, "stdev": 0.01},
    }
    THEORETICAL_MEAN = 2.5
    THEORETICAL_STD = 0.75

    def test_quantlib_row_statistics(self):
        """Checks the QuantLib results row against hand-computed statistics."""
        # QuantLib paths are always 2D: (n_paths, n_steps + 1).
        ql_paths = np.array([[100.0, 1.0], [100.0, 2.0], [100.0, 3.0]])
        with mock.patch.object(
            du.qlu, "generate_quantlib_paths", lambda **kwargs: (ql_paths, None)
        ), mock.patch.object(
            du.qpu, "generate_qmcpy_paths", lambda **kwargs: (np.zeros((1, 2, 2)), None)
        ):
            results = du.collect_library_results(
                "Sobol", "Paths", 2, 3, self.TIMING, self.TIMING,
                self.THEORETICAL_MEAN, self.THEORETICAL_STD,
            )

            row = next(r for r in results if r["Method"] == "QuantLib")
            assert row["Mean"] == pytest.approx(2.0)
            assert row["Std Dev"] == pytest.approx(np.sqrt(2 / 3))
            assert row["Mean Absolute Error"] == pytest.approx(0.5)
            assert row["Std Dev Error"] == pytest.approx(np.sqrt(2 / 3) - 0.75)

    def test_qmcpy_row_statistics(self):
        """Checks the QMCPy results row; one replication so pooled and per-replication stats coincide."""
        qp_paths = np.array([[[100.0, 1.0], [200.0, 2.0], [300.0, 3.0]]])
        with mock.patch.object(
            du.qpu, "generate_qmcpy_paths", lambda **kwargs: (qp_paths, None)
        ):
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

    def test_qmcpy_terminal_axis(self):
        """Regression guard for qp_paths[:, -1] vs qp_paths[..., -1]: a marker value placed
        only at the true terminal (last) axis catches an off-by-axis regression."""
        qp_paths = np.array([[[1.0, 1.0, 999.0], [1.0, 1.0, 999.0]]])
        with mock.patch.object(
            du.qpu, "generate_qmcpy_paths", lambda **kwargs: (qp_paths, None)
        ):
            results = du.collect_library_results(
                "Lattice", "Paths", 3, 2, {}, self.TIMING, 999.0, 0.0
            )

            row = results[0]
            assert row["Mean"] == pytest.approx(999.0)
            assert row["Std Dev"] == pytest.approx(0.0)

    def test_sweep_errors_are_averaged_over_replications(self):
        """Checks both sweep errors use every replication for both libraries."""
        terminal = np.array([[1.0, 2.0, 3.0], [3.0, 5.0, 7.0]])

        def ql_paths(seed, **kwargs):
            values = terminal[seed - cf.QUANTLIB_SEED]
            return np.column_stack((np.zeros(3), values)), None

        qp_paths = np.stack(
            (np.zeros_like(terminal), terminal), axis=-1
        )
        with mock.patch.object(
            du.qlu, "generate_quantlib_paths", ql_paths
        ), mock.patch.object(
            du.qpu, "generate_qmcpy_paths", lambda **kwargs: (qp_paths, None)
        ):
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

    def test_process_keeps_quantlib_replications(self):
        """Covariance callers receive every QuantLib replication, not only the last."""
        def ql_paths(seed, **kwargs):
            terminal = np.array([seed, seed + 1.0])
            return np.column_stack((np.zeros(2), terminal)), None

        qp_paths = np.zeros((2, 2, 1))
        with mock.patch.object(
            du.qlu, "generate_quantlib_paths", ql_paths
        ), mock.patch.object(
            du.qpu, "generate_qmcpy_paths", lambda **kwargs: (qp_paths, None)
        ):
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

    def test_covariance_grid_and_replications(self):
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


class TestQuantlibSeedIndependence(TestCase):
    """QuantLib's `seed` must actually change the Sobol scramble.

    demos/GBM/gbm_code/data_util.py:process_sampler_data() builds replications
    by calling generate_quantlib_paths() once per replication with
    seed = base_seed + r; it previously didn't work because
    UniformLowDiscrepancySequenceGenerator ignores its seed argument for the
    fixed Jaeckel direction integers. Methods progress from a single seed to
    a 5-seed loop mirroring that replication pattern.
    """

    def test_shape_and_values(self):
        """Checks output shape, initial value, and finiteness for a single seed."""
        for sampler_type in ["IIDStdUniform", "Sobol", "Halton"]:
            with self.subTest(sampler_type=sampler_type):
                paths = _quantlib_paths(7, sampler_type)
                assert paths.shape == (QUANTLIB_PARAMS["n_paths"], QUANTLIB_PARAMS["n_steps"] + 1)
                np.testing.assert_array_equal(paths[:, 0], QUANTLIB_PARAMS["initial_value"])
                assert np.isfinite(paths).all()

    def test_vectorized_matches_evolve(self):
        """Checks the vectorized evolution matches QuantLib's own evolve(), step by step.

        The low-discrepancy branches vectorize the evolution with numpy instead
        of calling QuantLib's path generator per path, for both the exact and
        the Euler-Maruyama scheme.
        """
        for scheme in ["exact", "euler"]:
            for sampler_type in ["Sobol", "Halton"]:
                with self.subTest(scheme=scheme, sampler_type=sampler_type):
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
        with self.assertRaisesRegex(ValueError, "Unsupported sampler type"):
            _quantlib_paths(1, sampler_type="unknown")

    def test_seed_reproducible_effective(self):
        """Same seed reproduces paths; a different seed changes them (2 seeds)."""
        for sampler_type in ["IIDStdUniform", "Sobol", "Halton"]:
            with self.subTest(sampler_type=sampler_type):
                np.testing.assert_array_equal(
                    _quantlib_paths(7, sampler_type), _quantlib_paths(7, sampler_type)
                )
                assert not np.array_equal(_quantlib_paths(7, sampler_type), _quantlib_paths(8, sampler_type))

    def test_seed_loop_distinct(self):
        """Sequential seeds (mirroring the replication loop) are pairwise distinct (5 seeds)."""
        for sampler_type in ["IIDStdUniform", "Sobol", "Halton"]:
            with self.subTest(sampler_type=sampler_type):
                _assert_distinct([_quantlib_paths(7 + r, sampler_type) for r in range(5)])


class TestQmcpySamplerSettings(TestCase):
    """Checks sampler options and Halton batching without changing the samples."""

    path_params = {
        "initial_value": 100.0,
        "mu": 0.05,
        "diffusion": 0.04,
        "maturity": 1.0,
        "n_steps": 8,
        "n_paths": 23,
        "sampler_type": "Halton",
        "seed": 7,
    }

    def test_notebook_comparison_sizes_match(self):
        """The comparison keeps the original workload in both libraries."""
        notebook_path = Path(__file__).parents[1] / "demos/GBM/gbm_demo.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        names = {"cf.is_debug", "params_ql", "params_qp"}
        assignments = {}
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] != "code" or not any(
                f"{name} =" in source for name in names
            ):
                continue
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    name = ast.unparse(node.targets[0])
                    if name in names:
                        assignments[name] = node.value
        self.assertEqual(set(assignments), names)
        default_debug = eval(
            compile(ast.Expression(assignments["cf.is_debug"]), str(notebook_path), "eval"),
            {"__builtins__": {}, "IN_COLAB": False},
        )
        with mock.patch.object(cf, "is_debug", default_debug):
            params_ql, params_qp = [
                eval(
                    compile(ast.Expression(assignments[name]), str(notebook_path), "eval"),
                    {"__builtins__": {}, "cf": cf},
                )
                for name in ("params_ql", "params_qp")
            ]
        for key in ("n_paths", "n_steps"):
            self.assertEqual(params_ql[key], params_qp[key])
        self.assertEqual(params_qp["n_paths"], 2**14)
        self.assertEqual(params_qp["n_steps"], 252)
        self.assertEqual(params_qp["replications"], 8)

    def test_default_sweep_sizes(self):
        """The full parameter sweep keeps its original paths and time steps."""
        with mock.patch.object(cf, "is_debug", False):
            configs = cf.get_experiment_configurations()
        self.assertEqual(configs["time_steps"]["fixed_paths"], 2**12)
        self.assertEqual(configs["time_steps"]["range"], [2**i for i in range(4, 10)])
        self.assertEqual(configs["paths"]["fixed_steps"], 252)
        self.assertEqual(configs["paths"]["range"], [2**i for i in range(9, 15)])

    def test_halton_batches_bound_digit_buffers(self):
        """Actual digit allocations fit the budget, including the last batch."""
        replications, batch_size = 2, 7
        budget = 2 * replications * batch_size * 8 * cf.HALTON_DIGITS * 8
        empty, buffers = np.empty, []

        def record_empty(shape, dtype=float, **kwargs):
            result = empty(shape, dtype=dtype, **kwargs)
            if result.dtype == np.uint64 and result.ndim == 4:
                if result.shape[-2:] == (8, cf.HALTON_DIGITS):
                    buffers.append((result.shape, result.nbytes))
            return result

        with (
            mock.patch.object(cf, "HALTON_MAX_DIGIT_BYTES", budget),
            mock.patch.object(np, "empty", side_effect=record_empty),
            mock.patch.object(qpu, "create_qmcpy_sampler", wraps=qpu.create_qmcpy_sampler) as create,
        ):
            paths, _ = qpu.generate_qmcpy_paths(**self.path_params, replications=replications)
        create.assert_called_once()
        self.assertEqual(paths.shape, (replications, 23, 8))
        self.assertEqual([shape[1] for shape, _ in buffers], [7, 7, 7, 7, 7, 7, 2, 2])
        for first, second in zip(buffers[::2], buffers[1::2]):
            self.assertLessEqual(first[1] + second[1], budget)

    def test_halton_batches_preserve_paths(self):
        """Batch boundaries preserve every path and the replication axes."""
        for replications in (None, 1, 2):
            for construction in ("PCA", "Cholesky", "BrownianBridge"):
                with self.subTest(replications=replications, construction=construction):
                    count = 1 if replications is None else replications
                    budget = 2 * count * 7 * 8 * cf.HALTON_DIGITS * 8
                    with mock.patch.object(cf, "HALTON_MAX_DIGIT_BYTES", budget):
                        paths, gbm = qpu.generate_qmcpy_paths(
                            **self.path_params, replications=replications,
                            decomp_type=construction,
                        )
                    expected = gbm.gen_samples(self.path_params["n_paths"])
                    np.testing.assert_allclose(paths, expected, rtol=1e-14, atol=0)
                    self.assertEqual(paths.shape, (23, 8) if replications is None else (count, 23, 8))
                    self.assertTrue(np.isfinite(paths).all())
                    if count > 1:
                        self.assertFalse(np.array_equal(paths[0], paths[1]))

    def test_halton_releases_discarded_samplers(self):
        """Repeated timing calls free abandoned samplers while GC is disabled."""
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            expected, live = qpu.generate_qmcpy_paths(**self.path_params)
            live_ref = weakref.ref(live)
            live_perms = weakref.ref(live.discrete_distrib.perms)
            old_paths, old = qpu.generate_qmcpy_paths(**self.path_params)
            old_ref = weakref.ref(old)
            old_perms = weakref.ref(old.discrete_distrib.perms)
            del old_paths, old
            self.assertIsNotNone(old_ref())
            self.assertIsNotNone(old_perms())

            paths, _ = qpu.generate_qmcpy_paths(**self.path_params)
            self.assertIsNone(old_ref())
            self.assertIsNone(old_perms())
            self.assertIs(live_ref(), live)
            self.assertIs(live_perms(), live.discrete_distrib.perms)
            np.testing.assert_array_equal(paths, expected)
            self.assertFalse(gc.isenabled())
        finally:
            if was_enabled:
                gc.enable()
            else:
                gc.disable()

    def test_halton_batches_empty_and_invalid(self):
        """Batching keeps empty shapes and rejects counts outside the sequence."""
        with mock.patch.object(cf, "HALTON_MAX_DIGIT_BYTES", 1):
            for replications in (None, 1, 2):
                with self.subTest(replications=replications):
                    paths, _ = qpu.generate_qmcpy_paths(
                        **{**self.path_params, "n_paths": 0}, replications=replications
                    )
                    self.assertEqual(paths.shape, (0, 8) if replications is None else (replications, 0, 8))
            for n_paths in (-1, 2**32 + 1):
                with self.subTest(n_paths=n_paths):
                    with self.assertRaises(ParameterError):
                        qpu.generate_qmcpy_paths(**{**self.path_params, "n_paths": n_paths})

    def test_other_samplers_ignore_halton_budget(self):
        """A tiny Halton budget leaves other samplers' paths unchanged."""
        for sampler in ("IIDStdUniform", "Sobol", "Lattice"):
            with self.subTest(sampler=sampler):
                params = {**self.path_params, "sampler_type": sampler, "n_paths": 16}
                expected, _ = qpu.generate_qmcpy_paths(**params)
                with mock.patch.object(cf, "HALTON_MAX_DIGIT_BYTES", 1):
                    paths, _ = qpu.generate_qmcpy_paths(**params)
                np.testing.assert_array_equal(paths, expected)

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

    def test_other_samplers_keep_defaults(self):
        """Checks the digit-count override is not applied to the base-2 samplers."""
        for sampler_type in ["IIDStdUniform", "Sobol", "Lattice"]:
            with self.subTest(sampler_type=sampler_type):
                sampler = qpu.create_qmcpy_sampler(sampler_type, 8)
                assert getattr(sampler, "t", 63) == 63

    def test_unknown_sampler_raises(self):
        """Checks that an unsupported sampler_type raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Unsupported sampler type"):
            qpu.create_qmcpy_sampler("Faure", 8)


class TestQuantlibSchemes(TestCase):
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
        with self.assertRaisesRegex(ValueError, "Unsupported scheme"):
            generate_quantlib_paths(
                **QUANTLIB_PARAMS, sampler_type="Sobol", seed=7, scheme="milstein"
            )

    def test_reproducible_above_direction_integer_table(self):
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

        with mock.patch.object(
            qlu.ql, "Burley2020SobolRsg", record_constructor
        ):
            params = {**QUANTLIB_PARAMS, "n_steps": 64}
            first = _quantlib_paths(7, "Sobol", n_steps=params["n_steps"])
            second = _quantlib_paths(7, "Sobol", n_steps=params["n_steps"])
            np.testing.assert_array_equal(first, second)
            assert not np.array_equal(
                first, _quantlib_paths(8, "Sobol", n_steps=params["n_steps"])
            )
            assert calls
            assert all(args[1] == cf.SOBOL_DIRECTION_SEED != 0 for args in calls)


class TestConstructionAblation(TestCase):
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


class TestReplicationMeanIndependence(TestCase):
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
        test/test_dd_discrete_distribs.py.)
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
