"""Unit tests for subclasses of StoppingCriterion in QMCPy"""

import builtins
import copy
import importlib
import io
import pickle
import tempfile
import unittest
import warnings
import numpy as np
import pandas as pd
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from qmcpy import (
    AbstractDiscreteDistribution,
    AbstractIntegrand,
    AbstractStoppingCriterion,
    CubBayesLatticeG,
    CubBayesNetG,
    CubMCCLT,
    CubMCCLTVec,
    CubMCG,
    CubMCML,
    CubMCMLCont,
    CubMLMC,
    CubMLMCCont,
    CubMLQMC,
    CubMLQMCCont,
    CubQMCBayesLatticeG,
    CubQMCCLT,
    CubQMCLatticeG,
    CubQMCML,
    CubQMCMLCont,
    CubQMCNetG,
    CubQMCRepStudentT,
    DigitalNetB2,
    FinancialOption,
    Halton,
    IIDStdUniform,
    Ishigami,
    Keister,
    Lattice,
    PFGPCI,
    SensitivityIndices,
    SobolIndices,
)
from qmcpy.util import (
    DistributionCompatibilityError,
    MaxLevelsWarning,
    MaxSamplesWarning,
    MethodImplementationError,
    ParameterError,
    ParameterWarning
)
from qmcpy.util.data import Data
from qmcpy.stopping_criterion.diagnostics import _IterationHistoryTable, _IterationTraceLogger, _print_diagnostic, _get_iteration_log_frame

# Test functions and parameters
keister_2d_exact = 1.808186429263620
tol = 0.005
rel_tol = 0

_ISHIGAMI_A = 7
_ISHIGAMI_B = 0.1
_ISHIGAMI_INDICES = np.array(
    [
        [True, False, False],
        [False, True, False],
        [False, False, True],
        [True, True, False],
        [True, False, True],
        [False, True, True],
    ],
    dtype=bool,
)


def _sensitivity_converges(distrib_factory, sc_class, abs_tol, sc_kwargs=None, n_tries=3):
    """Return True if at least one of n_tries Ishigami sensitivity-index runs meets abs_tol."""
    if sc_kwargs is None:
        sc_kwargs = {}
    true_solution = Ishigami._exact_sensitivity_indices(
        _ISHIGAMI_INDICES, _ISHIGAMI_A, _ISHIGAMI_B
    )
    for _ in range(n_tries):
        si = SensitivityIndices(
            Ishigami(distrib_factory(), _ISHIGAMI_A, _ISHIGAMI_B), _ISHIGAMI_INDICES
        )
        solution, _ = sc_class(si, abs_tol=abs_tol, rel_tol=0, **sc_kwargs).integrate()
        if (abs(solution.squeeze() - true_solution) < abs_tol).all():
            return True
    return False


class _DummyDiscreteDistribution(AbstractDiscreteDistribution):
    def __init__(self):
        pass


class _OtherDummyDiscreteDistribution(AbstractDiscreteDistribution):
    def __init__(self):
        pass


class _DummyIntegrand(AbstractIntegrand):
    def __init__(self, true_measure, discrete_distrib, d_indv=(), d_comb=(), dependency=None):
        self.true_measure = true_measure
        self.discrete_distrib = discrete_distrib
        self.d_indv = d_indv
        self.d_comb = d_comb
        self._dependency = dependency or (lambda flags: flags)

    def dependency(self, comb_flags):
        return self._dependency(comb_flags)


class _DummyStoppingCriterion(AbstractStoppingCriterion):
    def __init__(self, integrand=None, true_measure=None, discrete_distrib=None, allowed_distribs=None, allow_vectorized_integrals=False, parameters=None):
        if integrand is not None:
            self.integrand = integrand
        if true_measure is not None:
            self.true_measure = true_measure
        if discrete_distrib is not None:
            self.discrete_distrib = discrete_distrib
        if parameters is not None:
            self.parameters = parameters
        super().__init__(allowed_distribs or [_DummyDiscreteDistribution], allow_vectorized_integrals)


########################################################################
# Abstract Base Tests
########################################################################
class TestAbstractStoppingCriterion(unittest.TestCase):
    def test_print_diagnostic_header_and_values(self):
        data = type(
            "Data",
            (),
            {
                "_iter_count": 2,
                "solution": [1.25],
                "m": np.array(4),
                "n_total": 16,
                "n_min": None,
                "xfull": np.zeros((2, 3)),
            },
        )()
        stream = io.StringIO()
        with redirect_stdout(stream):
            _print_diagnostic("resume", data, table_header=True)
        output = stream.getvalue()
        self.assertIn("stage", output)
        self.assertIn("iter", output)
        self.assertIn("resume", output)
        self.assertIn("1.2500000", output)
        self.assertIn("(2, 3)", output)
        self.assertRegex(output, r"resume\s+2\s+1\.2500000\s+None\s+16\s+4")

    def test_print_diagnostic_missing_values(self):
        data = type("Data", (), {"solution": float("nan"), "xfull": None})()
        stream = io.StringIO()
        with redirect_stdout(stream):
            _print_diagnostic("start", data)
        output = stream.getvalue()
        self.assertIn("start", output)
        self.assertIn("nan", output)
        self.assertIn("None", output)

    def test_print_diagnostic_unthrottled(self):
        data = type(
            "Data",
            (),
            {
                "solution": [1.25],
                "_iter_count": 55,
                "m": 13,
                "n_total": 16,
                "n_min": None,
                "xfull": np.zeros((2, 3)),
            },
        )()
        throttled = io.StringIO()
        with redirect_stdout(throttled):
            _print_diagnostic("ITER", data, verbose=False)
        self.assertEqual(throttled.getvalue(), "")

        unthrottled = io.StringIO()
        with redirect_stdout(unthrottled):
            _print_diagnostic("ITER", data, verbose=True)
        output = unthrottled.getvalue()
        self.assertIn("ITER", output)
        self.assertIn("1.2500000", output)
        self.assertRegex(output, r"ITER\s+55\s+1\.2500000\s+None\s+16\s+13")

    def test_print_diagnostic_hide_opt_cols(self):
        data = type(
            "Data",
            (),
            {
                "_iter_count": 1,
                "solution": [1.25],
                "n_total": 16,
            },
        )()
        stream = io.StringIO()
        with redirect_stdout(stream):
            _print_diagnostic(
                "ITER",
                data,
                table_header=True,
                visible_columns=("stage", "iter", "solution", "n_total"),
            )
        output = stream.getvalue()
        self.assertIn("stage", output)
        self.assertIn("iter", output)
        self.assertIn("n_total", output)
        self.assertNotIn("n_min", output)
        self.assertNotIn("xfull.shape", output)

    def test_resume_trace_iter_count(self):
        logger = _IterationTraceLogger(
            type(
                "SC",
                (),
                {
                    "trace_iterations": True,
                    "trace_label": "resume-test",
                    "verbose": True,
                },
            )()
        )
        data = type(
            "Data",
            (),
            {
                "_iter_count": 6,
                "solution": [1.25],
                "n_total": 16,
                "n_min": 8,
                "m": 13,
                "xfull": np.zeros((16, 2)),
            },
        )()
        stream = io.StringIO()
        with redirect_stdout(stream):
            logger.resume(data, step_value=13)
            # Simulate new samples being drawn: update n_total so state differs
            data.n_total = 20
            data.xfull = np.zeros((20, 2))
            logger.iteration(data, step_value=14)
        output = stream.getvalue()
        self.assertRegex(output, r"RESUME\s+6\s+1\.2500000\s+8\s+16\s+13")
        self.assertNotIn("RESUME       None", output)
        self.assertRegex(output, r"ITER\s+7\s+1\.2500000\s+8\s+20\s+14")

    def test_trace_history_marks_flushed_rows(self):
        logger = _IterationTraceLogger(
            type(
                "SC",
                (),
                {
                    "trace_iterations": True,
                    "trace_label": "",
                    "verbose": False,
                },
            )()
        )
        logger.iter_count = 54
        data = type(
            "Data",
            (),
            {
                "solution": [1.25],
                "n_total": 64,
                "m": 7,
                "xfull": np.zeros((64, 2)),
            },
        )()
        with redirect_stdout(io.StringIO()):
            logger.iteration(data, step_value=7)
            self.assertIsInstance(logger.history, _IterationHistoryTable)
            self.assertEqual(len(logger.history), 1)
            self.assertEqual(logger.history["stage"][0], "ITER")
            self.assertEqual(logger.history["iter"][0], 55)
            self.assertFalse(logger.history["printed"][0])
            logger.finalize()
        self.assertTrue(logger.history["printed"][0])
        self.assertEqual(
            logger.history.visible_columns,
            ("stage", "iter", "solution", "n_total", "m", "xfull.shape"),
        )

    def test_prepare_resume_data_keeps_history(self):
        distrib = _DummyDiscreteDistribution()
        integrand = _DummyIntegrand(object(), distrib)
        sc = _DummyStoppingCriterion(
            integrand=integrand,
            true_measure=integrand.true_measure,
            discrete_distrib=distrib,
        )
        history = _IterationHistoryTable()
        history._append(
            "ITER",
            {
                "iter": 1,
                "solution": 1.0,
                "abs_tol": 0.1,
                "bound_diff": None,
                "comb_bound_diff": None,
                "bound_half_width": None,
                "bias_estimate": None,
                "rmse_estimate": None,
                "rmse_tol": None,
                "n_min": None,
                "n_total": 16,
                "m": None,
                "xfull.shape": None,
            },
            visible_columns=("stage", "iter", "solution", "n_total"),
            printed=True,
        )
        resume = Data(parameters=["solution", "n_total"])
        resume.solution = 1.0
        resume.n_total = 16
        resume.iteration_history = history
        resume.history_df = pd.DataFrame({"stage": ["ITER"], "iter": ["1"]})
        resume.stopping_crit = type(
            "SavedSC", (), {"iteration_history": history, "history_df": resume.history_df}
        )()
        copied = sc._prepare_resume_data(resume, lambda data: None, lambda data: None)
        self.assertIsNotNone(copied.iteration_history)
        self.assertIsNot(copied.iteration_history, history)
        self.assertEqual(copied.iteration_history["stage"], history["stage"])
        self.assertIsNone(copied.stopping_crit.iteration_history)
        self.assertIsNone(copied.stopping_crit.history_df)
        self.assertIs(resume.iteration_history, history)
        self.assertIs(resume.stopping_crit.iteration_history, history)
        self.assertIs(resume.stopping_crit.history_df, resume.history_df)

    def test_init_requires_integrand(self):
        with self.assertRaises(ParameterError):
            _DummyStoppingCriterion()

    def test_init_requires_matching_true_measure(self):
        distrib = _DummyDiscreteDistribution()
        integrand = _DummyIntegrand(object(), distrib)
        with self.assertRaises(ParameterError):
            _DummyStoppingCriterion(
                integrand=integrand,
                true_measure=object(),
                discrete_distrib=distrib,
            )

    def test_init_requires_matching_distribution(self):
        distrib = _DummyDiscreteDistribution()
        integrand = _DummyIntegrand(object(), distrib)
        with self.assertRaises(ParameterError):
            _DummyStoppingCriterion(
                integrand=integrand,
                true_measure=integrand.true_measure,
                discrete_distrib=_DummyDiscreteDistribution(),
            )

    def test_init_rejects_bad_distribution_type(self):
        distrib = _OtherDummyDiscreteDistribution()
        integrand = _DummyIntegrand(object(), distrib)
        with self.assertRaises(DistributionCompatibilityError):
            _DummyStoppingCriterion(
                integrand=integrand,
                true_measure=integrand.true_measure,
                discrete_distrib=distrib,
            )

    def test_init_rejects_vectorized_integrals(self):
        distrib = _DummyDiscreteDistribution()
        integrand = _DummyIntegrand(object(), distrib, d_indv=(2,), d_comb=(2,))
        with self.assertRaises(ParameterError):
            _DummyStoppingCriterion(
                integrand=integrand,
                true_measure=integrand.true_measure,
                discrete_distrib=distrib,
            )

    def test_default_parameters_and_repr(self):
        distrib = _DummyDiscreteDistribution()
        integrand = _DummyIntegrand(object(), distrib)
        sc = _DummyStoppingCriterion(
            integrand=integrand,
            true_measure=integrand.true_measure,
            discrete_distrib=distrib,
        )
        self.assertEqual(sc.parameters, [])
        self.assertIn("AbstractStoppingCriterion", repr(sc))

    def test_abstract_methods_raise(self):
        distrib = _DummyDiscreteDistribution()
        integrand = _DummyIntegrand(object(), distrib)
        sc = _DummyStoppingCriterion(
            integrand=integrand,
            true_measure=integrand.true_measure,
            discrete_distrib=distrib,
        )
        with self.assertRaises(MethodImplementationError):
            sc.integrate()
        with self.assertRaises(MethodImplementationError):
            sc.set_tolerance(abs_tol=0.1)

    def test_compute_indv_alphas_identity(self):
        distrib = _DummyDiscreteDistribution()
        integrand = _DummyIntegrand(
            object(),
            distrib,
            d_indv=(2,),
            d_comb=(2,),
            dependency=lambda flags: flags,
        )
        sc = _DummyStoppingCriterion(
            integrand=integrand,
            true_measure=integrand.true_measure,
            discrete_distrib=distrib,
            allow_vectorized_integrals=True,
        )
        alphas_indv, identity_dependency = sc._compute_indv_alphas(np.array([0.2, 0.4]))
        self.assertTrue(identity_dependency)
        self.assertTrue(np.allclose(alphas_indv, [0.2, 0.4]))

    def test_compute_indv_alphas_dependency(self):
        distrib = _DummyDiscreteDistribution()

        def dependency(flags):
            return np.array([False, False, True]) if not flags[0] else np.array([True, False, False])

        integrand = _DummyIntegrand(
            object(),
            distrib,
            d_indv=(3,),
            d_comb=(2,),
            dependency=dependency,
        )
        sc = _DummyStoppingCriterion(
            integrand=integrand,
            true_measure=integrand.true_measure,
            discrete_distrib=distrib,
            allow_vectorized_integrals=True,
        )
        alphas_indv, identity_dependency = sc._compute_indv_alphas(np.array([0.3, 0.6]))
        self.assertFalse(identity_dependency)
        self.assertTrue(np.allclose(alphas_indv, [0.15, 0.15, 0.3]))

    def test_log_all_resume(self):
        # when all rows are RESUME, stage_last view returns empty DataFrame
        log_df = pd.DataFrame({"stage": ["RESUME", "RESUME"], "iter": [1, 2], "solution": [1.0, 2.0]})
        result = AbstractStoppingCriterion._apply_iteration_log_view(log_df, "stage_last")
        self.assertEqual(len(result), 0)
        self.assertListEqual(list(result.columns), list(log_df.columns))


########################################################################
# Import Fallback Tests
########################################################################
class TestFallbacks(unittest.TestCase):
    def test_import_fallback_classes(self):
        original_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name in ("torch", "gpytorch"):
                raise ImportError("forced import failure")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_fake_import):
            sc_mod = importlib.import_module("qmcpy.stopping_criterion")
            sc_mod = importlib.reload(sc_mod)
            with self.assertRaises(ModuleNotFoundError):
                sc_mod.PFGPCI()
            with self.assertRaises(ModuleNotFoundError):
                sc_mod.PFSampleErrorDensityAR()
            with self.assertRaises(ModuleNotFoundError):
                sc_mod.SuggesterSimple()
        importlib.reload(importlib.import_module("qmcpy.stopping_criterion"))


########################################################################
# CubMCCLTVec Tests
########################################################################
class TestCubMCCLTVec(unittest.TestCase):
    """Branch-focused tests for CubMCCLTVec."""

    def _make_integrand(self):
        return Keister(IIDStdUniform(dimension=2, seed=7))

    def test_constructor_normalizes_non_power_of_two_limits(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sc = CubMCCLTVec(
                self._make_integrand(), n_init=30, n_limit=1000
            )

        parameter_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, ParameterWarning)
        ]
        self.assertEqual(len(parameter_warnings), 2)
        self.assertEqual(sc.n_init, 2**5)
        self.assertEqual(sc.n_limit, 2**30)

    def test_constructor_validation(self):
        with self.assertRaises(DistributionCompatibilityError):
            CubMCCLTVec(Keister(Lattice(dimension=2, seed=7)))
        with self.assertRaises(AssertionError):
            CubMCCLTVec(self._make_integrand(), inflate=0.99)

    def test_set_tolerance(self):
        sc = CubMCCLTVec(self._make_integrand())
        sc.set_tolerance(abs_tol=0.25, rel_tol=0.1)

        self.assertEqual(sc.abs_tol, 0.25)
        self.assertEqual(sc.rel_tol, 0.1)
        np.testing.assert_allclose(sc.abs_tols, 0.25)
        np.testing.assert_allclose(sc.rel_tols, 0.1)
        with self.assertRaises(AssertionError):
            sc.set_tolerance(rmse_tol=0.1)

    def test_integrate_warns_at_sample_limit(self):
        sc = CubMCCLTVec(
            self._make_integrand(),
            abs_tol=1e-12,
            n_init=2**5,
            n_limit=2**5,
        )

        with self.assertWarns(MaxSamplesWarning):
            solution, data = sc.integrate()
        self.assertTrue(np.isfinite(solution))
        self.assertEqual(data.n_total, 2**5)


########################################################################
# CubQMCBayesLatticeG Tests
########################################################################
class TestCubQMCBayesLatticeG(unittest.TestCase):
    """Branch-focused tests for CubQMCBayesLatticeG."""

    def _make_sc(self):
        return CubQMCBayesLatticeG(
            Keister(Lattice(dimension=2, seed=7)),
            n_init=2**5,
            n_limit=2**8,
        )

    def test_constructor_validation(self):
        with self.assertRaises(DistributionCompatibilityError):
            CubQMCBayesLatticeG(
                Keister(DigitalNetB2(dimension=2, seed=7))
            )
        with self.assertRaises(ParameterError):
            CubQMCBayesLatticeG(
                Keister(
                    Lattice(dimension=2, seed=7, order="LINEAR")
                )
            )

    def test_shift_invariant_bernoulli_kernels(self):
        sc = self._make_sc()
        xun = np.array(
            [[0, 0], [0.25, 0.5], [0.5, 0.25], [0.75, 0.75]],
            dtype=float,
        )

        for order in (1, 2):
            with self.subTest(order=order):
                values = sc._shift_inv_kernel(
                    xun,
                    order=order,
                    theta=0.5,
                    avoid_cancel_error=True,
                    kern_type=1,
                    debug_enable=False,
                )
                eigenvalues, eigenvalues_ring, scale = values
                self.assertEqual(eigenvalues.shape, (4,))
                self.assertEqual(eigenvalues_ring.shape, (4,))
                self.assertTrue(np.isfinite(eigenvalues).all())
                self.assertTrue(np.isfinite(eigenvalues_ring).all())
                self.assertGreater(scale, 0)

        with self.assertRaisesRegex(
            ParameterError, "Bernoulli order not implemented"
        ):
            sc._shift_inv_kernel(
                xun,
                order=3,
                theta=0.5,
                avoid_cancel_error=True,
                kern_type=1,
                debug_enable=False,
            )

    def test_shift_invariant_geometric_kernel(self):
        sc = self._make_sc()
        xun = np.array(
            [[0, 0], [0.25, 0.5], [0.5, 0.25], [0.75, 0.75]],
            dtype=float,
        )
        eigenvalues, eigenvalues_ring, scale = sc._shift_inv_kernel(
            xun,
            order=0.5,
            theta=0.5,
            avoid_cancel_error=True,
            kern_type=2,
            debug_enable=False,
        )

        self.assertEqual(eigenvalues.shape, (4,))
        self.assertEqual(eigenvalues_ring.shape, (4,))
        self.assertTrue(np.isfinite(eigenvalues).all())
        self.assertTrue(np.isfinite(eigenvalues_ring).all())
        self.assertGreater(scale, 0)


########################################################################
# CubMCCLT Tests
########################################################################
class TestCubMCCLT(unittest.TestCase):
    """Unit tests for CubMCCLT StoppingCriterion."""

    def test_raise_dist_compat_error(self):
        integrand = Keister(Lattice(dimension=2, seed=7))
        self.assertRaises(DistributionCompatibilityError, CubMCCLT, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        algorithm = CubMCCLT(integrand, abs_tol=0.001, n_init=64, n_limit=1000)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        solution, _ = CubMCCLT(integrand, abs_tol=tol).integrate()
        self.assertLess(abs(solution - keister_2d_exact), tol)

    def test_trace_history(self):
        sc = CubMCCLT(
            Keister(IIDStdUniform(dimension=2, seed=7)),
            abs_tol=0.5,
            n_init=64,
            n_limit=4096,
        )
        sc.trace_iterations = True
        sc.trace_label = "CubMCCLT"
        sc.verbose = True
        sc.trace_print = False
        stream = io.StringIO()
        with redirect_stdout(stream):
            _, data = sc.integrate()
        self.assertEqual(stream.getvalue(), "")
        self.assertIs(data.iteration_history, sc.iteration_history)
        self.assertIsInstance(data.iteration_history, _IterationHistoryTable)
        self.assertEqual(data.iteration_history["stage"][-1], "ITER")
        self.assertTrue(all(data.iteration_history["printed"]))
        self.assertIn("n_total", data.iteration_history.visible_columns)
        self.assertGreater(len(data.iteration_history), 0)
        log_text = sc.format_iteration_log()
        self.assertIn("=== CubMCCLT iteration log ===", log_text)
        self.assertIn("ITER", log_text)
        replay_stream = io.StringIO()
        sc.print_iteration_log(file=replay_stream)
        self.assertEqual(replay_stream.getvalue().strip(), log_text)

    def test_iter_log_without_live_trace(self):
        sc = CubMCCLT(
            Keister(IIDStdUniform(dimension=2, seed=7)),
            abs_tol=0.5,
            n_init=64,
            n_limit=4096,
        )
        stream = io.StringIO()
        with redirect_stdout(stream):
            _, data = sc.integrate()
        self.assertEqual(stream.getvalue(), "")
        self.assertIs(data.iteration_history, sc.iteration_history)
        self.assertIsInstance(data.iteration_history, _IterationHistoryTable)
        log_df = sc.get_iteration_log()
        self.assertIsInstance(sc.history_df, pd.DataFrame)
        self.assertIs(data.history_df, sc.history_df)
        self.assertTrue(log_df.equals(sc.history_df))
        self.assertFalse(log_df.empty)
        self.assertNotIn("printed", log_df.columns)
        self.assertIn("stage", log_df.columns)
        self.assertIn("n_total", log_df.columns)
        self.assertIn("elapsed_time", log_df.columns)
        self.assertRegex(log_df.iloc[-1]["solution"], r"^-?\d+\.\d{7}$")
        self.assertRegex(log_df.iloc[-1]["bound_diff"], r"^-?\d\.\d{3}e[+-]\d{2}$")
        self.assertEqual(len(log_df), len(sc.get_iteration_log(printed_only=False)))
        raw_log_df = sc.get_iteration_log(formatted=False)
        self.assertIn("elapsed_time", raw_log_df.columns)
        self.assertTrue((raw_log_df["elapsed_time"].dropna().diff().fillna(0) >= 0).all())
        self.assertAlmostEqual(sc.elapsed_time, data.elapsed_time)

    def test_iter_log_rejects_unknown_view(self):
        sc = CubMCCLT(
            Keister(IIDStdUniform(dimension=2, seed=7)),
            abs_tol=0.5,
            n_init=64,
            n_limit=4096,
        )
        _, _ = sc.integrate()
        with self.assertRaises(ParameterError):
            sc.get_iteration_log(view="unknown")


########################################################################
# CubMCG Tests
########################################################################
class TestCubMCG(unittest.TestCase):
    """Unit tests for CubMCG StoppingCriterion."""

    def test_raise_dist_compat_error(self):
        integrand = Keister(Lattice(dimension=2, seed=7))
        self.assertRaises(DistributionCompatibilityError, CubMCG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        algorithm = CubMCG(integrand, abs_tol=0.001, n_init=64, n_limit=500)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        solution, _ = CubMCG(integrand, abs_tol=tol).integrate()
        self.assertLess(abs(solution - keister_2d_exact), tol)


########################################################################
# CubQMCCLT Tests
########################################################################
class TestCubQMCCLT(unittest.TestCase):
    """Unit tests for CubQMCCLT StoppingCriterion."""

    def test_raise_dist_compat_error(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        self.assertRaises(DistributionCompatibilityError, CubQMCCLT, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(Lattice(dimension=2, seed=7, replications=32))
        algorithm = CubQMCCLT(integrand, abs_tol=0.001, n_init=16, n_limit=32)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(Halton(dimension=2, seed=7, replications=32))
        solution, _ = CubQMCCLT(integrand, abs_tol=tol).integrate()
        self.assertLess(abs(solution - keister_2d_exact), tol)

    def test_sobol_indices_dnb2(self):
        self.assertTrue(_sensitivity_converges(
            lambda: DigitalNetB2(3, seed=7, replications=32), CubQMCCLT, abs_tol=5e-2
        ))

    def test_sobol_indices_lattice(self):
        self.assertTrue(_sensitivity_converges(
            lambda: Lattice(3, seed=7, replications=32), CubQMCCLT, abs_tol=5e-2
        ))

    def test_sobol_indices_halton(self):
        self.assertTrue(_sensitivity_converges(
            lambda: Halton(3, seed=7, replications=32), CubQMCCLT, abs_tol=5e-2
        ))


########################################################################
# CubQMCLatticeG Tests
########################################################################
class TestCubQMCLatticeG(unittest.TestCase):
    """Unit tests for CubQMCLatticeG StoppingCriterion."""

    def test_raise_dist_compat_error(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        self.assertRaises(DistributionCompatibilityError, CubQMCLatticeG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(Lattice(dimension=2, seed=7))
        algorithm = CubQMCLatticeG(integrand, abs_tol=0.001, n_init=2**8, n_limit=2**9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(Lattice(dimension=2, seed=7))
        solution, _ = CubQMCLatticeG(integrand, abs_tol=tol).integrate()
        self.assertLess(abs(solution - keister_2d_exact), tol)

    def test_sobol_indices(self):
        self.assertTrue(_sensitivity_converges(
            lambda: Lattice(3, seed=7), CubQMCLatticeG, abs_tol=5e-3
        ))


########################################################################
# CubQMCNetG Tests
########################################################################
class TestCubQMCNetG(unittest.TestCase):
    """Unit tests for CubQMCNetG StoppingCriterion."""

    def test_raise_dist_compat_error(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        self.assertRaises(DistributionCompatibilityError, CubQMCNetG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(DigitalNetB2(dimension=2, seed=7))
        algorithm = CubQMCNetG(integrand, abs_tol=0.001, n_init=2**8, n_limit=2**9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(DigitalNetB2(dimension=2, seed=7))
        solution, _ = CubQMCNetG(integrand, abs_tol=tol).integrate()
        self.assertLess(abs(solution - keister_2d_exact), tol)

    def test_sobol_indices(self):
        self.assertTrue(_sensitivity_converges(
            lambda: DigitalNetB2(3, seed=7), CubQMCNetG, abs_tol=1e-2
        ))


########################################################################
# CubBayesLatticeG Tests
########################################################################
class TestCubBayesLatticeG(unittest.TestCase):
    """Unit tests for CubBayesLatticeG StoppingCriterion."""

    def test_raise_dist_compat_error(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        self.assertRaises(DistributionCompatibilityError, CubBayesLatticeG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(Lattice(dimension=2, seed=7, order="RADICAL INVERSE"))
        algorithm = CubBayesLatticeG(integrand, abs_tol=0.0001, n_init=2**8, n_limit=2**9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(Lattice(dimension=2, seed=7, order="RADICAL INVERSE"))
        solution, _ = CubBayesLatticeG(integrand, abs_tol=tol, n_init=2**5).integrate()
        self.assertTrue(abs(solution - keister_2d_exact) < tol)

    def test_sobol_indices_bayes_lattice(self, dims=3, abs_tol=1e-2):
        keister_d_ = Keister(Lattice(dimension=dims, seed=7))
        keister_indices_ = SobolIndices(keister_d_, indices="singletons")
        sc_ = CubQMCLatticeG(keister_indices_, abs_tol=abs_tol, ptransform="Baker")
        solution_, _ = sc_.integrate()

        keister_d = Keister(Lattice(dimension=dims, order="RADICAL INVERSE", seed=7))
        keister_indices = SobolIndices(keister_d, indices="singletons")
        sc = CubBayesLatticeG(keister_indices, order=1, abs_tol=abs_tol, ptransform="Baker")
        solution, _ = sc.integrate()

        self.assertTrue(solution.shape, (dims, dims, 1))
        self.assertTrue(abs(solution - solution_).max() < abs_tol)

    def test_sobol_indices(self):
        self.assertTrue(_sensitivity_converges(
            lambda: Lattice(3, seed=7, order="natural"), CubBayesLatticeG,
            abs_tol=5e-2, sc_kwargs={"order": 1, "ptransform": "Baker"},
        ))


########################################################################
# CubBayesNetG Tests
########################################################################
class TestCubBayesNetG(unittest.TestCase):
    """Unit tests for CubBayesNetG StoppingCriterion."""

    def test_raise_dist_compat_error(self):
        integrand = Keister(IIDStdUniform(dimension=2, seed=7))
        self.assertRaises(DistributionCompatibilityError, CubBayesNetG, integrand)

    def test_n_max_single_level(self):
        integrand = Keister(DigitalNetB2(dimension=2, seed=7))
        algorithm = CubBayesNetG(integrand, abs_tol=0.0001, n_init=2**8, n_limit=2**9)
        self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_keister_2d(self):
        integrand = Keister(DigitalNetB2(dimension=2, seed=7))
        solution, _ = CubBayesNetG(integrand, n_init=2**5, abs_tol=tol).integrate()
        self.assertTrue(abs(solution - keister_2d_exact) < tol)

    def test_sobol_indices(self):
        self.assertTrue(_sensitivity_converges(
            lambda: DigitalNetB2(3, seed=7), CubBayesNetG, abs_tol=1e-2
        ))


########################################################################
# Multilevel Core Tests
########################################################################
class TestMLStoppingCriteria(unittest.TestCase):
    def _iid_financial_option(self):
        return FinancialOption(IIDStdUniform(seed=7), start_price=30, strike_price=30)

    def _qmc_financial_option(self):
        return FinancialOption(Lattice(replications=32, seed=7), start_price=30, strike_price=30)

    def test_raise_dist_compat_error(self):
        cases = [
            ("CubMCML", CubMCML, Lattice(seed=7)),
            ("CubQMCML", CubQMCML, IIDStdUniform(seed=7)),
        ]
        for label, cls, sampler in cases:
            with self.subTest(stopping_criterion=label):
                integrand = FinancialOption(sampler)
                with self.assertRaises(DistributionCompatibilityError):
                    cls(integrand)

    def test_n_max(self):
        cases = [
            ("CubMCML", CubMCML, FinancialOption(IIDStdUniform(seed=7), start_price=30, strike_price=30), {"rmse_tol": 0.001, "n_limit": 2**10}),
            ("CubQMCML", CubQMCML, FinancialOption(Lattice(replications=32, seed=7), start_price=30, strike_price=30), {"abs_tol": tol, "n_limit": 2**10}),
        ]
        for label, cls, integrand, kwargs in cases:
            with self.subTest(stopping_criterion=label):
                algorithm = cls(integrand, **kwargs)
                self.assertWarns(MaxSamplesWarning, algorithm.integrate)

    def test_multilevel_parameter_validation(self):
        cases = [
            ("CubMLMC", CubMLMC, self._iid_financial_option),
            ("CubMLMCCont", CubMLMCCont, self._iid_financial_option),
        ]
        invalid_kwargs = [
            ({"levels_min": 1}, "needs levels_min >= 2"),
            ({"levels_min": 2, "levels_max": 1}, "needs levels_max >= levels_min"),
            ({"n_init": 0}, "needs n_init > 0"),
        ]
        for label, cls, integrand_factory in cases:
            for kwargs, message in invalid_kwargs:
                with self.subTest(stopping_criterion=label, kwargs=kwargs):
                    with self.assertRaisesRegex(ParameterError, message):
                        cls(integrand_factory(), **kwargs)

    def test_multilevel_rmse_overrides_abs_tol(self):
        cases = [
            ("CubMLMC", CubMLMC, self._iid_financial_option, "rmse_tol"),
            ("CubMLMCCont", CubMLMCCont, self._iid_financial_option, "target_rmse_tol"),
            ("CubMLQMC", CubMLQMC, self._qmc_financial_option, "rmse_tol"),
            ("CubMLQMCCont", CubMLQMCCont, self._qmc_financial_option, "target_rmse_tol"),
        ]
        for label, cls, integrand_factory, attr in cases:
            with self.subTest(stopping_criterion=label):
                sc = cls(integrand_factory(), abs_tol=999.0, rmse_tol=0.123)
                self.assertEqual(
                    getattr(sc, attr),
                    0.123,
                    msg=f"{label} did not prioritize rmse_tol over abs_tol",
                )

    def test_continuation_warns_at_max_levels(self):
        cases = [
            ("CubMLMCCont", CubMLMCCont, self._iid_financial_option, 2),
            ("CubMLQMCCont", CubMLQMCCont, self._qmc_financial_option, 3),
        ]
        for label, cls, integrand_factory, levels_max in cases:
            with self.subTest(stopping_criterion=label):
                sc = cls(
                    integrand_factory(), rmse_tol=0.1, levels_min=2, levels_max=levels_max
                )
                sc.rmse_tol = sc.target_rmse_tol
                data = sc._construct_data()
                data.n_total = 0
                data.levels = sc.levels_max
                data.n_level[:] = 1
                with ExitStack() as stack:
                    if label == "CubMLMCCont":
                        data.sum_level[:] = 1.0
                        data.var_level = np.ones_like(data.n_level, dtype=float)
                        data.diff_n_level = np.zeros_like(data.n_level)
                        stack.enter_context(patch.object(sc, "_update_data"))
                        stack.enter_context(patch.object(sc, "_update_theta"))
                        stack.enter_context(patch.object(sc, "_get_next_samples", return_value=data.n_level.copy()))
                        stack.enter_context(patch.object(sc, "_rmse", return_value=sc.rmse_tol * 2))
                        add_level = stack.enter_context(patch.object(sc, "_add_level"))
                        with self.assertWarns(MaxLevelsWarning):
                            sc._integrate(data)
                    else:
                        data.eval_level[:] = False
                        data.mean_level_reps[:] = 1.0
                        data.var_level[:] = 0.0
                        data.var_cost_ratio_level[:] = 0.0
                        data.bias_estimate = 0.0
                        stack.enter_context(patch.object(sc, "update_data"))
                        stack.enter_context(patch.object(sc, "_update_theta"))
                        stack.enter_context(patch.object(sc, "_varest", return_value=0.0))
                        stack.enter_context(patch.object(sc, "_rmse", return_value=sc.rmse_tol * 2))
                        add_level = stack.enter_context(patch.object(sc, "_add_level"))
                        with self.assertWarns(MaxLevelsWarning):
                            sc._integrate(data, skip_level_reset=True)
                    add_level.assert_not_called()


########################################################################
# Continuation AbsTol Tests
########################################################################
class TestCubMLMCContAbsTol(unittest.TestCase):
    """Tests for the abs_tol (not rmse_tol) code path in multilevel cont. criteria."""

    def test_mlmc_cont_abs_tol_path_reachable(self):
        fo = FinancialOption(IIDStdUniform(seed=7), start_price=30, strike_price=30)
        sc = CubMLMCCont(fo, abs_tol=0.5)
        # Verify that target_rmse_tol was set (via abs_tol branch, not rmse_tol)
        self.assertGreater(sc.target_rmse_tol, 0.0)

    def test_mlqmc_cont_abs_tol_path_reachable(self):
        fo = FinancialOption(Lattice(replications=32, seed=7), start_price=30, strike_price=30)
        sc = CubMLQMCCont(fo, abs_tol=0.5)
        self.assertGreater(sc.target_rmse_tol, 0.0)


########################################################################
# Bias Convergence Path Tests
########################################################################
class TestCubMLMCBiasConvergencePaths(unittest.TestCase):
    """Tests that exercise the bias-convergence level-addition paths."""

    def setUp(self):
        warnings.filterwarnings("ignore", category=MaxSamplesWarning)
        warnings.filterwarnings("ignore", category=MaxLevelsWarning)

    def test_cub_mlmc_cont_n_limit_warning(self):
        """CubMLMCCont must warn MaxSamplesWarning when n_limit is exceeded inside _integrate."""
        fo = FinancialOption(IIDStdUniform(seed=7), start_price=30, strike_price=30)
        sc = CubMLMCCont(fo, rmse_tol=0.001, n_limit=2**10)
        with self.assertWarns(MaxSamplesWarning):
            sc.integrate()

    def test_cub_mlqmc_bad_mean_level_reps(self):
        """_validate_resume raises ParameterError if mean_level_reps has wrong shape."""
        fo = FinancialOption(Lattice(replications=16, seed=7), start_price=30, strike_price=30)
        sc = CubMLQMC(fo, abs_tol=0.3, n_limit=2**16)
        _, checkpoint = sc.integrate()

        bad = copy.deepcopy(checkpoint)
        bad.mean_level_reps = bad.mean_level_reps[:, :-1]  # wrong shape

        sc2 = CubMLQMC(FinancialOption(Lattice(replications=16, seed=7), start_price=30, strike_price=30), abs_tol=0.1, n_limit=2**16)
        with self.assertRaises(ParameterError):
            sc2.integrate(resume=bad)

    def test_cub_mlqmc_cont_bad_mean_level_reps(self):
        """Continuation resume must reject an incompatible mean_level_reps shape."""
        fo = FinancialOption(Lattice(replications=16, seed=7), start_price=30, strike_price=30)
        sc = CubMLQMCCont(fo, abs_tol=0.3, n_limit=2**16)
        _, checkpoint = sc.integrate()

        bad = copy.deepcopy(checkpoint)
        bad.mean_level_reps = bad.mean_level_reps[:, :-1]  # wrong shape

        sc2 = CubMLQMCCont(FinancialOption(Lattice(replications=16, seed=7), start_price=30, strike_price=30), abs_tol=0.1, n_limit=2**16)
        with self.assertRaises(ParameterError):
            sc2.integrate(resume=bad)

    def test_cub_mlmc_max_levels_bias_warning(self):
        """CubMLMC fires MaxLevelsWarning when bias check hits levels_max."""
        fo = FinancialOption(IIDStdUniform(seed=7), start_price=30, strike_price=30)
        # levels_min == levels_max == 2 so the algorithm cannot add levels
        sc = CubMLMC(fo, rmse_tol=0.001, levels_min=2, levels_max=2, n_limit=int(1e10))
        N = 100

        def fake_update_data(data):
            n = data.levels + 1
            data.n_level = np.full(n, N, dtype=int)
            data.sum_level = np.ones((2, n)) * N
            data.cost_level = np.ones(n, dtype=float) * N
            # mean_level large enough so rem >> sqrt(theta)*rmse_tol
            data.mean_level = np.ones(n)
            data.var_level = np.ones(n) * 1e-8
            data.cost_per_sample = np.ones(n)
            data.alpha = 1.0
            data.beta = 1.0
            data.gamma = 1.0
            data.n_total = N * n

        # Return n_level so diff_n_level = max(0, n_level - n_level) = 0
        # -> triggers the bias convergence check inside the loop
        with patch.object(sc, "_update_data", side_effect=fake_update_data):
            with patch.object(sc, "_get_next_samples",
                              side_effect=lambda data: data.n_level.copy()):
                with self.assertWarns(MaxLevelsWarning):
                    sc.integrate()

    def test_cub_mlmc_add_level_from_bias_path(self):
        """CubMLMC adds levels through the bias convergence path."""
        fo = FinancialOption(IIDStdUniform(seed=7), start_price=30, strike_price=30)
        sc = CubMLMC(fo, rmse_tol=0.5, n_limit=int(1e10))
        _, data = sc.integrate()
        # A successful run must have added at least one level beyond levels_min=2
        self.assertGreater(data.levels, 2)

    def test_cub_mlqmc_bias_add_level_path(self):
        """CubMLQMC exercises the elif bias > rmse_tol/sqrt(2) -> _add_level path."""
        fo = FinancialOption(Lattice(replications=16, seed=7), start_price=30, strike_price=30)
        sc = CubMLQMC(fo, abs_tol=0.5, n_limit=int(1e10))
        _, data = sc.integrate()
        # A successful run should have exercised at least one level addition
        self.assertGreater(data.levels, 1)

    def test_cub_mlmc_cont_add_level_and_warmup(self):
        """CubMLMCCont _integrate fires _add_level (line 302) and warm-up (lines 241-251)."""
        fo = FinancialOption(IIDStdUniform(seed=7), start_price=30, strike_price=30)
        # n_tols=1, inflate=1 means exactly one _integrate call at step_tol=rmse_tol
        sc = CubMLMCCont(fo, rmse_tol=0.1, n_tols=1, inflate=1.0, n_limit=int(1e10))

        call_count = [0]

        def patched_rmse(data):
            call_count[0] += 1
            # First call: force non-convergence -> _add_level fires (line 302)
            if call_count[0] <= 1:
                return sc.target_rmse_tol * 3.0
            # Subsequent calls: converge
            return sc.target_rmse_tol * 0.3

        with patch.object(sc, "_rmse", side_effect=patched_rmse):
            _, data = sc.integrate()

        # levels > levels_min+1 = 3 confirms _add_level fired inside _integrate
        self.assertGreater(data.levels, 3)


########################################################################
# Resume Feature Tests
########################################################################
class TestResumeFeature(unittest.TestCase):
    """Tests for the resume parameter of integrate() across all stopping criteria."""

    def setUp(self):
        warnings.filterwarnings("ignore", category=MaxSamplesWarning)
        self.seed = 7
        self.dimension = 2
        self.loose_abs_tol = 0.2
        self.tight_abs_tol = 0.1
        self.rel_tol = 0
        self.n_init = 2**8
        self.n_limit = 2**10
        self.n_limit_ml = 2**16  # multilevel algorithms need more headroom
        self.n_init_rep = 2**5
        self.n_limit_rep = 2**18

    def _iid_distribution(self):
        return IIDStdUniform(self.dimension, seed=self.seed)

    def _lattice_distribution(self):
        return Lattice(self.dimension, seed=self.seed)

    def _net_distribution(self):
        return DigitalNetB2(self.dimension, seed=self.seed)

    def _net_rep_distribution(self):
        return DigitalNetB2(self.dimension, replications=16, seed=self.seed)

    def _iid_financial_option(self):
        return FinancialOption(IIDStdUniform(self.dimension, seed=self.seed), start_price=30, strike_price=30)

    def _qmc_financial_option(self):
        return FinancialOption(Lattice(replications=32, seed=self.seed), start_price=30, strike_price=30)

    def _keister_builder(self, stopping_criterion, distribution_factory, abs_tol):
        return lambda: stopping_criterion(Keister(distribution_factory()), abs_tol=abs_tol, rel_tol=self.rel_tol, n_init=self.n_init, n_limit=self.n_limit)

    def _multilevel_builder(self, stopping_criterion, integrand_factory, tol_kwarg, tol):
        return lambda: stopping_criterion(integrand_factory(), **{tol_kwarg: tol}, n_limit=self.n_limit_ml)

    def _qmc_rep_student_t_builder(self, abs_tol):
        return lambda: CubQMCRepStudentT(
            Keister(self._net_rep_distribution()),
            abs_tol=abs_tol,
            rel_tol=self.rel_tol,
            n_init=self.n_init_rep,
            n_limit=self.n_limit_rep,
        )

    def _configure_resume_solver(self, loose_sc, tight_builder):
        tight_sc = tight_builder()
        abs_tol = getattr(tight_sc, "abs_tol", None)
        rel_tol = getattr(tight_sc, "rel_tol", None)
        rmse_tol = getattr(tight_sc, "target_rmse_tol", getattr(tight_sc, "rmse_tol", None))
        resume_attrs = set(getattr(tight_sc, "parameters", ())) | {
            "abs_tol",
            "rel_tol",
            "rmse_tol",
            "target_rmse_tol",
            "n_init",
            "n_limit",
            "n_tols",
            "inflate",
        }
        for attr in resume_attrs:
            if hasattr(tight_sc, attr):
                setattr(loose_sc, attr, copy.deepcopy(getattr(tight_sc, attr)))
        if hasattr(loose_sc, "set_tolerance"):
            loose_sc.set_tolerance(abs_tol=abs_tol, rel_tol=rel_tol, rmse_tol=rmse_tol)
        loose_sc.iteration_history = None
        loose_sc.history_df = None
        if hasattr(loose_sc, "_last_finalized_data"):
            loose_sc._last_finalized_data = None
        return loose_sc

    def _run_resume_pair(self, label, loose_builder, tight_builder):
        loose_sc = loose_builder()
        _, checkpoint = loose_sc.integrate()
        self.assertIs(
            checkpoint.stopping_crit,
            loose_sc,
            msg=f"{label} LOOSE stage should keep the original solver instance",
        )
        checkpoint_n_total = int(checkpoint.n_total)
        checkpoint_xfull = (
            np.array(checkpoint.xfull, copy=True) if hasattr(checkpoint, "xfull") else None
        )
        loose_log = loose_sc.get_iteration_log(formatted=False).copy()
        loose_df = loose_sc.get_iteration_log().copy()

        resumed_sc = self._configure_resume_solver(loose_sc, tight_builder)
        self.assertIs(
            resumed_sc,
            loose_sc,
            msg=f"{label} RESUMED stage should reuse the LOOSE solver instance",
        )
        sol_resume, resumed = resumed_sc.integrate(resume=checkpoint)
        self.assertIs(
            resumed.stopping_crit,
            loose_sc,
            msg=f"{label} RESUMED stage should finalize with the reused solver instance",
        )
        return {
            "loose_sc": loose_sc,
            "checkpoint": checkpoint,
            "checkpoint_n_total": checkpoint_n_total,
            "checkpoint_xfull": checkpoint_xfull,
            "loose_log": loose_log,
            "loose_df": loose_df,
            "resumed_sc": resumed_sc,
            "sol_resume": sol_resume,
            "resumed": resumed,
        }

    def _assert_resume_behavior(
        self,
        label,
        loose_builder,
        tight_builder,
        compare_to_fresh=False,
        rtol=1e-12,
        atol=1e-12,
        skip_exceptions=(),
    ):
        def _run_assertions():
            stages = self._run_resume_pair(label, loose_builder, tight_builder)
            sc1 = stages["loose_sc"]
            data1 = stages["checkpoint"]
            old_n_total = stages["checkpoint_n_total"]
            old_xfull = stages["checkpoint_xfull"]
            sol2 = stages["sol_resume"]
            data2 = stages["resumed"]

            self.assertIsNot(
                data2, data1, msg=f"{label} resume should not mutate the input checkpoint"
            )
            self.assertIs(
                data1.stopping_crit,
                sc1,
                msg=f"{label} resume should not rewrite the checkpoint solver instance",
            )
            self.assertTrue(hasattr(data2, "n_total"), msg=f"{label} missing n_total")
            self.assertTrue(
                data2.n_total >= old_n_total,
                msg=f"{label} resume did not preserve/increase n_total",
            )
            self.assertEqual(
                int(data1.n_total),
                old_n_total,
                msg=f"{label} mutated the input checkpoint sample count",
            )
            if old_xfull is not None:
                self.assertTrue(
                    np.array_equal(data1.xfull, old_xfull),
                    msg=f"{label} mutated the input checkpoint samples",
                )
                resumed_prefix = data2.xfull[..., : old_xfull.shape[-2], :]
                self.assertTrue(
                    np.array_equal(resumed_prefix, old_xfull),
                    msg=f"{label} did not reuse the checkpoint sample prefix exactly",
                )

            if compare_to_fresh:
                sc3 = tight_builder()
                self.assertIsNot(
                    sc3,
                    sc1,
                    msg=f"{label} FRESH stage should use a distinct solver instance",
                )
                sol3, data3 = sc3.integrate()
                self.assertIs(
                    data3.stopping_crit,
                    sc3,
                    msg=f"{label} FRESH stage should finalize with its own solver instance",
                )
                self.assertIsNot(
                    data3.stopping_crit,
                    data2.stopping_crit,
                    msg=f"{label} FRESH stage should not share the RESUMED solver instance",
                )
                self.assertTrue(
                    np.allclose(sol2, sol3, rtol=rtol, atol=atol),
                    msg=f"{label} resume solution diverged from fresh run",
                )
                self.assertTrue(
                    np.array_equal(data2.xfull, data3.xfull),
                    msg=f"{label} resume samples diverged from fresh run",
                )
                self.assertTrue(
                    np.allclose(
                        data2.yfull, data3.yfull, rtol=rtol, atol=atol, equal_nan=True
                    ),
                    msg=f"{label} resume integrand evaluations diverged from fresh run",
                )

        if skip_exceptions:
            try:
                _run_assertions()
            except skip_exceptions as exc:
                self.skipTest(f"{label} resume skipped: {exc}")
        else:
            _run_assertions()

    def _assert_resume_boundary_and_final_parity(
        self,
        label,
        loose_builder,
        tight_builder,
        skip_exceptions=(),
    ):
        def _run_assertions():
            stages = self._run_resume_pair(label, loose_builder, tight_builder)
            loose_sc = stages["loose_sc"]
            checkpoint = stages["checkpoint"]
            resumed = stages["resumed"]
            resumed_sc = stages["resumed_sc"]
            loose_log = stages["loose_log"]
            loose_last_iter = int(loose_log.loc[loose_log["stage"] == "ITER", "iter"].iloc[-1])

            resumed_log = resumed_sc.get_iteration_log(formatted=False)
            resume_row = resumed_log.iloc[len(loose_log)]
            self.assertEqual(resume_row["stage"], "RESUME")
            self.assertEqual(
                int(resume_row["iter"]),
                loose_last_iter,
                msg=f"{label} RESUME-first iter should match LOOSE-final iter",
            )

            fresh_sc = tight_builder()
            self.assertIsNot(
                fresh_sc,
                loose_sc,
                msg=f"{label} FRESH stage should use a distinct solver instance",
            )
            _, fresh = fresh_sc.integrate()
            self.assertIs(
                fresh.stopping_crit,
                fresh_sc,
                msg=f"{label} FRESH stage should finalize with its own solver instance",
            )
            self.assertIsNot(
                fresh.stopping_crit,
                resumed.stopping_crit,
                msg=f"{label} FRESH stage should not share the RESUMED solver instance",
            )
            self.assertEqual(
                int(resumed.n_total),
                int(fresh.n_total),
                msg=f"{label} resumed-final n_total should match fresh-final n_total",
            )
            self.assertEqual(
                int(resumed._iter_count),
                int(fresh._iter_count),
                msg=f"{label} resumed-final iter count should match fresh-final iter count",
            )

        if skip_exceptions:
            try:
                _run_assertions()
            except skip_exceptions as exc:
                self.skipTest(f"{label} resume parity skipped: {exc}")
        else:
            _run_assertions()

    def test_resume_none_matches_fresh_start(self):
        """resume=None must behave identically to a fresh start."""
        make_sc = self._keister_builder(CubQMCLatticeG, self._lattice_distribution, self.tight_abs_tol)
        sol1, _ = make_sc().integrate()
        sol2, _ = make_sc().integrate(resume=None)
        self.assertTrue(np.allclose(sol1, sol2, rtol=1e-5))

    def test_resume_matches_fresh_tight_solution(self):
        """Resume solutions should match a fresh run at tighter tolerance."""
        cases = [
            ("CubMCCLTVec", CubMCCLTVec, self._iid_distribution, (ImportError, NotImplementedError)),
            ("CubQMCLatticeG", CubQMCLatticeG, self._lattice_distribution, ()),
            ("CubQMCNetG", CubQMCNetG, self._net_distribution, ()),
            ("CubBayesLatticeG", CubBayesLatticeG, self._lattice_distribution, ()),
            ("CubBayesNetG", CubBayesNetG, self._net_distribution, ()),
        ]

        for label, stopping_criterion, distribution_factory, skip_exceptions in cases:
            with self.subTest(stopping_criterion=label):
                self._assert_resume_behavior(
                    label,
                    self._keister_builder(stopping_criterion, distribution_factory, self.loose_abs_tol),
                    self._keister_builder(stopping_criterion, distribution_factory, self.tight_abs_tol),
                    compare_to_fresh=True,
                    skip_exceptions=skip_exceptions,
                )

        with self.subTest(stopping_criterion="CubQMCRepStudentT"):
            self._assert_resume_behavior(
                "CubQMCRepStudentT",
                self._qmc_rep_student_t_builder(self.loose_abs_tol),
                self._qmc_rep_student_t_builder(self.tight_abs_tol),
                compare_to_fresh=True,
            )

    def test_non_ml_resume_iter_n_parity(self):
        """Non-ML resume solvers should keep iteration numbering and final totals in sync."""
        cases = [
            ("CubMCCLTVec", CubMCCLTVec, self._iid_distribution, (ImportError, NotImplementedError)),
            ("CubQMCLatticeG", CubQMCLatticeG, self._lattice_distribution, ()),
            ("CubQMCNetG", CubQMCNetG, self._net_distribution, ()),
            ("CubBayesLatticeG", CubBayesLatticeG, self._lattice_distribution, ()),
            ("CubBayesNetG", CubBayesNetG, self._net_distribution, ()),
        ]

        for label, stopping_criterion, distribution_factory, skip_exceptions in cases:
            with self.subTest(stopping_criterion=label):
                self._assert_resume_boundary_and_final_parity(
                    label,
                    self._keister_builder(stopping_criterion, distribution_factory, self.loose_abs_tol),
                    self._keister_builder(stopping_criterion, distribution_factory, self.tight_abs_tol),
                    skip_exceptions=skip_exceptions,
                )

        with self.subTest(stopping_criterion="CubQMCRepStudentT"):
            self._assert_resume_boundary_and_final_parity(
                "CubQMCRepStudentT",
                self._qmc_rep_student_t_builder(self.loose_abs_tol),
                self._qmc_rep_student_t_builder(self.tight_abs_tol),
            )

    def test_resume_appends_iteration_history(self):
        stages = self._run_resume_pair(
            "CubQMCRepStudentT",
            self._qmc_rep_student_t_builder(self.loose_abs_tol),
            self._qmc_rep_student_t_builder(self.tight_abs_tol),
        )
        loose_sc = stages["loose_sc"]
        resumed_sc = stages["resumed_sc"]
        resumed = stages["resumed"]
        loose_df = stages["loose_df"]
        resumed_sc.get_iteration_log()  # populate history_df

        self.assertIsInstance(resumed_sc.history_df, pd.DataFrame)
        self.assertIs(resumed.history_df, resumed_sc.history_df)
        self.assertGreater(len(resumed.history_df), len(loose_df))
        self.assertEqual(
            list(resumed.history_df["stage"].iloc[: len(loose_df)]),
            list(loose_df["stage"]),
        )
        self.assertEqual(resumed.history_df["stage"].iloc[len(loose_df)], "RESUME")

    def test_resume_iteration_log_views(self):
        stages = self._run_resume_pair(
            "CubQMCRepStudentT",
            self._qmc_rep_student_t_builder(self.loose_abs_tol),
            self._qmc_rep_student_t_builder(self.tight_abs_tol),
        )
        current_log = stages["resumed_sc"].get_iteration_log(formatted=False, view="all")
        without_resume = stages["resumed_sc"].get_iteration_log(
            formatted=False, view="without_resume"
        )
        stage_last = stages["resumed_sc"].get_iteration_log(
            formatted=False, view="stage_last"
        )

        self.assertEqual(int((current_log["stage"] == "RESUME").sum()), 1)
        self.assertFalse((without_resume["stage"] == "RESUME").any())
        self.assertEqual(len(without_resume), len(current_log) - 1)

        expected_rows = []
        for index, row in current_log.iterrows():
            if row["stage"] == "RESUME":
                expected_rows.append(without_resume.loc[: index - 1].iloc[-1])
        expected_rows.append(without_resume.iloc[-1])

        self.assertEqual(len(stage_last), len(expected_rows))
        for (_, actual), expected in zip(stage_last.iterrows(), expected_rows):
            self.assertTrue(
                actual.equals(expected),
                msg="stage_last should keep one non-RESUME stop row per stage",
            )

    def test_resume_increases_multilevel_samples(self):
        """Multilevel resume runs should retain or increase sample work."""
        cases = [
            ("CubMCML", CubMCML, self._iid_financial_option, "rmse_tol"),
            ("CubMCMLCont", CubMCMLCont, self._iid_financial_option, "rmse_tol"),
            ("CubQMCML", CubQMCML, self._qmc_financial_option, "abs_tol"),
            ("CubQMCMLCont", CubQMCMLCont, self._qmc_financial_option, "abs_tol"),
        ]

        for label, stopping_criterion, integrand_factory, tol_kwarg in cases:
            with self.subTest(stopping_criterion=label):
                self._assert_resume_behavior(
                    label,
                    self._multilevel_builder(stopping_criterion, integrand_factory, tol_kwarg, self.loose_abs_tol),
                    self._multilevel_builder(stopping_criterion, integrand_factory, tol_kwarg, self.tight_abs_tol),
                )

    def test_cub_mlmc_resume_matches_fresh_n(self):
        stages = self._run_resume_pair(
            "CubMLMC",
            lambda: CubMLMC(self._iid_financial_option(), rmse_tol=self.loose_abs_tol, n_limit=self.n_limit_ml),
            lambda: CubMLMC(self._iid_financial_option(), rmse_tol=self.tight_abs_tol, n_limit=self.n_limit_ml),
        )
        loose_sc = stages["loose_sc"]
        checkpoint = stages["checkpoint"]
        sol_resume = stages["sol_resume"]
        resumed = stages["resumed"]
        self.assertTrue(hasattr(checkpoint, "level_diffs"))

        fresh_sc = CubMLMC(self._iid_financial_option(), rmse_tol=self.tight_abs_tol, n_limit=self.n_limit_ml)
        sol_fresh, fresh = fresh_sc.integrate()
        self.assertIsNot(fresh_sc, loose_sc)
        self.assertIs(fresh.stopping_crit, fresh_sc)
        self.assertIsNot(
            fresh.stopping_crit, resumed.stopping_crit,
            msg="CubMLMC FRESH stage should not share the RESUMED solver instance",
        )

        self.assertEqual(int(resumed.n_total), int(fresh.n_total))
        self.assertTrue(np.array_equal(resumed.n_level, fresh.n_level))
        self.assertTrue(np.allclose(resumed.sum_level, fresh.sum_level))
        self.assertAlmostEqual(float(sol_resume), float(sol_fresh))
        self.assertEqual(int(resumed._iter_count), int(fresh._iter_count))
        self.assertIsNotNone(resumed.iteration_history)

    def test_cub_mlmc_cont_resume_fresh_n(self):
        stages = self._run_resume_pair(
            "CubMLMCCont",
            lambda: CubMLMCCont(self._iid_financial_option(), rmse_tol=self.loose_abs_tol, n_limit=self.n_limit_ml),
            lambda: CubMLMCCont(self._iid_financial_option(), rmse_tol=self.tight_abs_tol, n_limit=self.n_limit_ml),
        )
        loose_sc = stages["loose_sc"]
        checkpoint = stages["checkpoint"]
        sol_resume = stages["sol_resume"]
        resumed = stages["resumed"]
        self.assertTrue(hasattr(checkpoint, "level_diffs"))

        fresh_sc = CubMLMCCont(self._iid_financial_option(), rmse_tol=self.tight_abs_tol, n_limit=self.n_limit_ml)
        sol_fresh, fresh = fresh_sc.integrate()
        self.assertIsNot(fresh_sc, loose_sc)
        self.assertIs(fresh.stopping_crit, fresh_sc)
        self.assertIsNot(
            fresh.stopping_crit, resumed.stopping_crit,
            msg="CubMLMCCont FRESH stage should not share the RESUMED solver instance",
        )

        self.assertEqual(int(resumed.n_total), int(fresh.n_total))
        self.assertTrue(np.array_equal(resumed.n_level, fresh.n_level))
        self.assertTrue(np.allclose(resumed.sum_level, fresh.sum_level))
        self.assertAlmostEqual(float(sol_resume), float(sol_fresh))
        self.assertEqual(int(resumed._iter_count), int(fresh._iter_count))
        self.assertIsNotNone(resumed.iteration_history)

    def test_cub_mlqmc_resume_matches_fresh_n(self):
        stages = self._run_resume_pair(
            "CubMLQMC",
            lambda: CubMLQMC(self._qmc_financial_option(), abs_tol=self.loose_abs_tol, n_limit=2**18),
            lambda: CubMLQMC(self._qmc_financial_option(), abs_tol=self.tight_abs_tol, n_limit=2**18),
        )
        loose_sc = stages["loose_sc"]
        checkpoint = stages["checkpoint"]
        sol_resume = stages["sol_resume"]
        resumed = stages["resumed"]
        self.assertTrue(hasattr(checkpoint, "level_rep_sums"))

        fresh_sc = CubMLQMC(self._qmc_financial_option(), abs_tol=self.tight_abs_tol, n_limit=2**18)
        sol_fresh, fresh = fresh_sc.integrate()
        self.assertIsNot(fresh_sc, loose_sc)
        self.assertIs(fresh.stopping_crit, fresh_sc)
        self.assertIsNot(
            fresh.stopping_crit, resumed.stopping_crit,
            msg="CubMLQMC FRESH stage should not share the RESUMED solver instance",
        )

        self.assertEqual(int(resumed.n_total), int(fresh.n_total))
        self.assertTrue(np.array_equal(resumed.n_level, fresh.n_level))
        self.assertTrue(np.allclose(resumed.mean_level_reps, fresh.mean_level_reps))
        self.assertAlmostEqual(float(sol_resume), float(sol_fresh))
        self.assertIsNotNone(resumed.iteration_history)

    def test_cub_mlqmc_cont_resume_fresh_n(self):
        stages = self._run_resume_pair(
            "CubMLQMCCont",
            lambda: CubMLQMCCont(self._qmc_financial_option(), abs_tol=self.loose_abs_tol, n_limit=2**18),
            lambda: CubMLQMCCont(self._qmc_financial_option(), abs_tol=self.tight_abs_tol, n_limit=2**18),
        )
        loose_sc = stages["loose_sc"]
        checkpoint = stages["checkpoint"]
        sol_resume = stages["sol_resume"]
        resumed = stages["resumed"]
        self.assertTrue(hasattr(checkpoint, "level_rep_sums"))

        fresh_sc = CubMLQMCCont(self._qmc_financial_option(), abs_tol=self.tight_abs_tol, n_limit=2**18)
        sol_fresh, fresh = fresh_sc.integrate()
        self.assertIsNot(fresh_sc, loose_sc)
        self.assertIs(fresh.stopping_crit, fresh_sc)
        self.assertIsNot(
            fresh.stopping_crit, resumed.stopping_crit,
            msg="CubMLQMCCont FRESH stage should not share the RESUMED solver instance",
        )

        self.assertEqual(int(resumed.n_total), int(fresh.n_total))
        self.assertTrue(np.array_equal(resumed.n_level, fresh.n_level))
        self.assertTrue(np.allclose(resumed.mean_level_reps, fresh.mean_level_reps))
        self.assertAlmostEqual(float(sol_resume), float(sol_fresh))
        self.assertIsNotNone(resumed.iteration_history)

    def test_cub_mlqmc_cont_long_resume_n_parity(self):
        kwargs = {"abs_tol": self.tight_abs_tol, "n_tols": 1200, "inflate": 1.001, "n_limit": 2**24}
        stages = self._run_resume_pair(
            "CubMLQMCCont",
            lambda: CubMLQMCCont(
                self._qmc_financial_option(),
                abs_tol=self.loose_abs_tol,
                n_tols=1200,
                inflate=1.001,
                n_limit=2**24,
            ),
            lambda: CubMLQMCCont(self._qmc_financial_option(), **kwargs),
        )
        loose_sc = stages["loose_sc"]
        resumed = stages["resumed"]

        fresh_sc = CubMLQMCCont(self._qmc_financial_option(), **kwargs)
        _, fresh = fresh_sc.integrate()
        self.assertIsNot(fresh_sc, loose_sc)
        self.assertIs(fresh.stopping_crit, fresh_sc)
        self.assertIsNot(
            fresh.stopping_crit, resumed.stopping_crit,
            msg="CubMLQMCCont FRESH stage should not share the RESUMED solver instance",
        )

        self.assertEqual(int(resumed.n_total), int(fresh.n_total))

    def test_cub_mlqmc_resume_iter_boundary(self):
        stages = self._run_resume_pair(
            "CubMLQMC",
            lambda: CubMLQMC(self._qmc_financial_option(), abs_tol=self.loose_abs_tol, n_limit=2**18),
            lambda: CubMLQMC(self._qmc_financial_option(), abs_tol=self.tight_abs_tol, n_limit=2**18),
        )
        loose_sc = stages["loose_sc"]
        resumed_sc = stages["resumed_sc"]
        loose_log = stages["loose_log"]
        loose_last_iter = int(loose_log.loc[loose_log["stage"] == "ITER", "iter"].iloc[-1])
        resumed_log = resumed_sc.get_iteration_log(formatted=False)

        resume_row = resumed_log.iloc[len(loose_log)]
        self.assertEqual(resume_row["stage"], "RESUME")
        self.assertEqual(int(resume_row["iter"]), loose_last_iter)

    def test_cub_mlqmc_cont_resume_iter_boundary(self):
        stages = self._run_resume_pair(
            "CubMLQMCCont",
            lambda: CubMLQMCCont(self._qmc_financial_option(), abs_tol=self.loose_abs_tol, n_limit=2**18),
            lambda: CubMLQMCCont(self._qmc_financial_option(), abs_tol=self.tight_abs_tol, n_limit=2**18),
        )
        loose_sc = stages["loose_sc"]
        resumed_sc = stages["resumed_sc"]
        loose_log = stages["loose_log"]
        loose_last_iter = int(loose_log.loc[loose_log["stage"] == "ITER", "iter"].iloc[-1])
        resumed_log = resumed_sc.get_iteration_log(formatted=False)

        resume_row = resumed_log.iloc[len(loose_log)]
        self.assertEqual(resume_row["stage"], "RESUME")
        self.assertEqual(int(resume_row["iter"]), loose_last_iter)

    def test_cont_resume_keeps_levels(self):
        cases = [
            ("CubMLMCCont", CubMLMCCont, self._iid_financial_option, 4),
            ("CubMLQMCCont", CubMLQMCCont, self._qmc_financial_option, 5),
        ]
        for label, stopping_criterion, integrand_factory, expected_levels in cases:
            with self.subTest(stopping_criterion=label):
                sc = stopping_criterion(integrand_factory(), abs_tol=0.2, rmse_tol=0.1)
                data = type("ResumeData", (), {"solution": 0.0, "levels": 5})()
                captured = {}

                def fake_integrate(resume_data, skip_level_reset=False, step_tol=None):
                    captured["levels"] = resume_data.levels
                    captured["skip_level_reset"] = skip_level_reset
                    captured["step_tol"] = step_tol

                with patch.object(sc, "_validate_resume"), \
                     patch.object(sc, "_integrate", side_effect=fake_integrate):
                    sc.integrate(resume=data)

                self.assertEqual(captured["levels"], expected_levels)
                self.assertTrue(captured["skip_level_reset"])
                self.assertIsNotNone(captured["step_tol"])
                self.assertEqual(sc.rmse_tol, sc.target_rmse_tol)

    def test_qmc_resume_requires_transform_state(self):
        loose_sc = CubQMCLatticeG(
            Keister(Lattice(dimension=self.dimension, seed=self.seed)),
            abs_tol=self.loose_abs_tol,
            rel_tol=self.rel_tol,
            n_init=self.n_init,
            n_limit=self.n_limit,
        )
        _, checkpoint = loose_sc.integrate()
        self.assertTrue(hasattr(checkpoint, "_kappanumap"))
        del checkpoint._kappanumap

        tight_sc = CubQMCLatticeG(Keister(Lattice(dimension=self.dimension, seed=self.seed)), abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol, n_init=self.n_init, n_limit=self.n_limit)
        with self.assertRaises(ParameterError):
            tight_sc.integrate(resume=checkpoint)

    def test_bayes_resume_requires_transform(self):
        loose_sc = CubBayesLatticeG(Keister(Lattice(dimension=self.dimension, seed=self.seed, order="RADICAL INVERSE")), abs_tol=self.loose_abs_tol, rel_tol=self.rel_tol, n_init=2**5, n_limit=self.n_limit)
        _, checkpoint = loose_sc.integrate()
        self.assertTrue(hasattr(checkpoint, "_ytildefull"))
        del checkpoint._ytildefull

        tight_sc = CubBayesLatticeG(Keister(Lattice(dimension=self.dimension, seed=self.seed, order="RADICAL INVERSE")), abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol, n_init=2**5, n_limit=self.n_limit)
        with self.assertRaises(ParameterError):
            tight_sc.integrate(resume=checkpoint)

    def _qmc_rep_student_t(self):
        return CubQMCRepStudentT(Keister(self._net_rep_distribution()), abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol,
            n_init=self.n_init_rep, n_limit=self.n_limit_rep,)

    def _pfgpci(self):
        return PFGPCI(Ishigami(DigitalNetB2(3, seed=self.seed)), failure_threshold=0, failure_above_threshold=False, abs_tol=self.tight_abs_tol, n_init=8, n_limit=16, n_batch=4, n_approx=2**8, gpytorch_train_iter=1, verbose=False, n_ref_approx=0)

    def _checkpoint_integrand(self, dimension=2, seed=13):
        return Keister(IIDStdUniform(dimension=dimension, seed=seed))

    def _make_qmc_checkpoint(self):
        loose_sc = CubQMCLatticeG(
            Keister(Lattice(dimension=2, seed=13)),
            abs_tol=0.2,
            n_init=2**8,
            n_limit=2048,
        )
        _, checkpoint = loose_sc.integrate()
        return checkpoint

    def _make_bayes_checkpoint(self):
        loose_sc = CubBayesLatticeG(
            Keister(Lattice(dimension=2, seed=13, order="RADICAL INVERSE")),
            abs_tol=0.2,
            n_init=2**5,
            n_limit=2048,
        )
        _, checkpoint = loose_sc.integrate()
        return checkpoint

    def test_unsupported_resume_raises(self):
        """Stopping criteria without resume support must raise ParameterError."""
        cases = [
            (
                "CubMCCLT",
                lambda: CubMCCLT(Keister(self._iid_distribution()), abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol, n_init=self.n_init, n_limit=self.n_limit),
            ),
            (
                "CubMCG",
                lambda: CubMCG(Keister(self._iid_distribution()), abs_tol=self.tight_abs_tol, rel_tol=self.rel_tol, n_init=self.n_init, n_limit=self.n_limit),
            ),
            ("PFGPCI", self._pfgpci),
        ]

        for label, stopping_criterion_factory in cases:
            with self.subTest(stopping_criterion=label):
                try:
                    sc = stopping_criterion_factory()
                except ModuleNotFoundError as exc:
                    self.skipTest(f"{label} unavailable: {exc}")
                with self.assertRaises(ParameterError):
                    sc.integrate(resume=object())

    def test_iter_log_for_non_resume_solvers(self):
        """Solvers that do not support resume should still populate iteration logs."""
        cases = [
            (
                "CubMCCLT",
                lambda: CubMCCLT(
                    Keister(self._iid_distribution()),
                    abs_tol=self.tight_abs_tol,
                    rel_tol=self.rel_tol,
                    n_init=self.n_init,
                    n_limit=self.n_limit,
                ),
            ),
            (
                "CubMCG",
                lambda: CubMCG(
                    Keister(self._iid_distribution()),
                    abs_tol=self.tight_abs_tol,
                    rel_tol=self.rel_tol,
                    n_init=self.n_init,
                    n_limit=self.n_limit,
                ),
            ),
            ("PFGPCI", self._pfgpci),
        ]

        for label, stopping_criterion_factory in cases:
            with self.subTest(stopping_criterion=label):
                try:
                    sc = stopping_criterion_factory()
                except ModuleNotFoundError as exc:
                    self.skipTest(f"{label} unavailable: {exc}")
                stream = io.StringIO()
                with redirect_stdout(stream):
                    _, data = sc.integrate()
                self.assertEqual(stream.getvalue(), "")
                self.assertIs(data.iteration_history, sc.iteration_history)
                self.assertIsNotNone(sc.iteration_history)
                log_df = sc.get_iteration_log()
                self.assertIsNotNone(sc.history_df)
                self.assertFalse(sc.history_df.empty)
                raw_log_df = sc.get_iteration_log(formatted=False)
                self.assertFalse(raw_log_df.empty)
                self.assertIn("stage", raw_log_df.columns)
                self.assertIn("n_total", raw_log_df.columns)
                self.assertTrue((raw_log_df["stage"] == "ITER").any())
                self.assertEqual(int(raw_log_df.iloc[-1]["n_total"]), int(data.n_total))
                self.assertIn("ITER", sc.format_iteration_log())

    def test_rep_student_t_resume_requires_ysums(self):
        loose_sc = CubQMCRepStudentT(
            Keister(self._net_rep_distribution()),
            abs_tol=self.loose_abs_tol,
            rel_tol=self.rel_tol,
            n_init=self.n_init_rep,
            n_limit=self.n_limit_rep,
        )
        _, checkpoint = loose_sc.integrate()
        self.assertTrue(hasattr(checkpoint, "_ysums"))
        del checkpoint._ysums

        tight_sc = self._qmc_rep_student_t()
        with self.assertRaises(ParameterError):
            tight_sc.integrate(resume=checkpoint)

    def test_cub_mc_clt_resume_raises(self):
        loose_sc = CubMCCLT(
            self._checkpoint_integrand(),
            abs_tol=0.2,
            n_init=32,
            n_limit=2048,
        )
        _, checkpoint = loose_sc.integrate()

        tight_sc = CubMCCLT(
            self._checkpoint_integrand(),
            abs_tol=0.05,
            n_init=32,
            n_limit=2048,
        )
        with self.assertRaises(ParameterError):
            tight_sc.integrate(resume=checkpoint)

    def test_cub_mc_g_resume_raises_param_error(self):
        loose_sc = CubMCG(
            self._checkpoint_integrand(),
            abs_tol=0.2,
            n_init=32,
            n_limit=2048,
        )
        _, checkpoint = loose_sc.integrate()

        tight_sc = CubMCG(
            self._checkpoint_integrand(),
            abs_tol=0.05,
            n_init=32,
            n_limit=2048,
        )
        with self.assertRaises(ParameterError):
            tight_sc.integrate(resume=checkpoint)

    def test_resume_rejects_bad_dimension(self):
        loose_sc = CubMCCLTVec(
            self._checkpoint_integrand(),
            abs_tol=0.2,
            n_init=32,
            n_limit=2048,
        )
        _, checkpoint = loose_sc.integrate()

        incompatible_sc = CubMCCLTVec(
            self._checkpoint_integrand(dimension=3),
            abs_tol=0.05,
            n_init=32,
            n_limit=2048,
        )
        with self.assertRaises(ParameterError):
            incompatible_sc.integrate(resume=checkpoint)

    def test_data_save_load_round_trip(self):
        data = Data(parameters=["value"])
        data.value = np.array([1.0, 2.0, 3.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            plain_path = tmp_path / "checkpoint.pkl"
            gzip_path = tmp_path / "checkpoint.pkl.gz"

            data.save(plain_path)
            loaded_plain = Data.load(plain_path)
            self.assertTrue(np.array_equal(loaded_plain.value, data.value))

            data.save(gzip_path, compress=True)
            loaded_gzip = Data.load(gzip_path)
            self.assertTrue(np.array_equal(loaded_gzip.value, data.value))

    def test_data_save_returns_path_and_gz(self):
        data = Data(parameters=["value"])
        data.value = np.array([1.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "chk.pkl"
            plain_path = data.save(base)
            self.assertEqual(plain_path, str(base))
            gz_path = data.save(Path(tmpdir) / "chk2.pkl", compress=True)
            self.assertTrue(gz_path.endswith(".gz"))

    def test_data_save_file_exists(self):
        data = Data(parameters=["value"])
        data.value = np.array([1.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chk.pkl"
            data.save(path)
            with self.assertRaises(FileExistsError):
                data.save(path)

    def test_data_save_overwrite_replaces_file(self):
        data1 = Data(parameters=["value"])
        data1.value = np.array([1.0])
        data2 = Data(parameters=["value"])
        data2.value = np.array([99.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chk.pkl"
            data1.save(path)
            data2.save(path, overwrite=True)
            loaded = Data.load(path)
            self.assertTrue(np.array_equal(loaded.value, data2.value))

    def test_data_load_rejects_non_data_ckpt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "not_data.pkl"
            with open(path, "wb") as f:
                pickle.dump({"not": "data"}, f)
            with self.assertRaises(TypeError):
                Data.load(path)

    def test_resume_records_metadata(self):
        checkpoint = self._make_qmc_checkpoint()
        old_n_total = int(checkpoint.n_total)
        old_total_time = float(checkpoint.time_integrate_total)
        old_xfull = np.array(checkpoint.xfull, copy=True)

        self.assertFalse(checkpoint.resumed)
        self.assertEqual(checkpoint.n_resume_from, 0)
        self.assertAlmostEqual(checkpoint.time_integrate_previous, 0.0)
        self.assertAlmostEqual(checkpoint.time_integrate_total, checkpoint.time_integrate)
        self.assertTrue(checkpoint._stopping_criterion_class.endswith(".CubQMCLatticeG"))
        self.assertTrue(checkpoint._integrand_class.endswith(".Keister"))
        self.assertTrue(checkpoint._true_measure_class.endswith(".Gaussian"))
        self.assertTrue(
            checkpoint._discrete_distribution_class.endswith(".Lattice")
        )

        tight_sc = CubQMCLatticeG(
            Keister(Lattice(dimension=2, seed=13)),
            abs_tol=0.05,
            n_init=2**8,
            n_limit=2048,
        )
        _, resumed = tight_sc.integrate(resume=checkpoint)

        self.assertIsNot(resumed, checkpoint)
        self.assertTrue(resumed.resumed)
        self.assertEqual(resumed.n_resume_from, old_n_total)
        self.assertAlmostEqual(resumed.time_integrate_previous, old_total_time)
        self.assertAlmostEqual(resumed.time_integrate_resume, resumed.time_integrate)
        self.assertAlmostEqual(
            resumed.time_integrate_total,
            resumed.time_integrate_previous + resumed.time_integrate_resume,
        )
        self.assertAlmostEqual(resumed.elapsed_time, resumed.time_integrate_total)
        self.assertAlmostEqual(tight_sc.elapsed_time, resumed.elapsed_time)
        self.assertFalse(checkpoint.resumed)
        self.assertEqual(checkpoint.n_resume_from, 0)
        self.assertEqual(int(checkpoint.n_total), old_n_total)
        self.assertTrue(np.array_equal(checkpoint.xfull, old_xfull))

    def test_qmc_resume_rejects_bad_checkpoint(self):
        checkpoint = self._make_qmc_checkpoint()
        cases = []

        bad = copy.deepcopy(checkpoint)
        bad.xfull = bad.xfull[:-1]
        cases.append(("xfull_length", bad))

        bad = copy.deepcopy(checkpoint)
        bad.n_max = int(bad.n_total) * 2
        cases.append(("n_max", bad))

        bad = copy.deepcopy(checkpoint)
        bad.n_total = 192
        bad.n_max = 192
        bad.xfull = bad.xfull[:192]
        bad.yfull = bad.yfull[..., :192]
        bad._ytildefull = bad._ytildefull[..., :192]
        bad._kappanumap = bad._kappanumap[..., :192]
        bad.n = np.minimum(np.asarray(bad.n), 192)
        cases.append(("non_power_of_two_n_total", bad))

        tight_sc = CubQMCLatticeG(
            Keister(Lattice(dimension=2, seed=13)),
            abs_tol=0.05,
            n_init=2**8,
            n_limit=2048,
        )
        for label, bad_checkpoint in cases:
            with self.subTest(case=label):
                with self.assertRaises(ParameterError):
                    tight_sc.integrate(resume=bad_checkpoint)

    def test_bayes_resume_rejects_bad_checkpoint(self):
        checkpoint = self._make_bayes_checkpoint()
        cases = []

        bad = copy.deepcopy(checkpoint)
        bad._ytildefull = bad._ytildefull[..., :-1]
        cases.append(("transform_length", bad))

        bad = copy.deepcopy(checkpoint)
        bad.n_total = 48
        bad.n_max = 48
        bad.xfull = np.concatenate([bad.xfull, bad.xfull[:16]], axis=0)
        bad.yfull = np.concatenate([bad.yfull, bad.yfull[..., :16]], axis=-1)
        bad._ytildefull = np.concatenate(
            [bad._ytildefull, bad._ytildefull[..., :16]], axis=-1
        )
        bad.n = np.full_like(np.asarray(bad.n), 48)
        cases.append(("non_power_of_two_n_total", bad))

        tight_sc = CubBayesLatticeG(
            Keister(Lattice(dimension=2, seed=13, order="RADICAL INVERSE")),
            abs_tol=0.05,
            n_init=2**5,
            n_limit=2048,
        )
        for label, bad_checkpoint in cases:
            with self.subTest(case=label):
                with self.assertRaises(ParameterError):
                    tight_sc.integrate(resume=bad_checkpoint)

    def test_resume_rejects_bad_format_version(self):
        loose_sc = CubQMCLatticeG(
            Keister(Lattice(dimension=2, seed=13)),
            abs_tol=0.2,
            n_init=2**8,
            n_limit=2048,
        )
        _, checkpoint = loose_sc.integrate()
        checkpoint._resume_format_version = 999

        tight_sc = CubQMCLatticeG(
            Keister(Lattice(dimension=2, seed=13)),
            abs_tol=0.05,
            n_init=2**8,
            n_limit=2048,
        )
        with self.assertRaises(ParameterError):
            tight_sc.integrate(resume=checkpoint)

    def test_resume_rejects_bad_sampler_seed(self):
        loose_sc = CubQMCLatticeG(
            Keister(Lattice(dimension=2, seed=13)),
            abs_tol=0.2,
            n_init=2**8,
            n_limit=2048,
        )
        _, checkpoint = loose_sc.integrate()

        tight_sc = CubQMCLatticeG(
            Keister(Lattice(dimension=2, seed=14)),
            abs_tol=0.05,
            n_init=2**8,
            n_limit=2048,
        )
        with self.assertRaises(ParameterError):
            tight_sc.integrate(resume=checkpoint)

    def test_cub_mc_clt_vec_resume_keeps_rng(self):
        loose_sc = CubMCCLTVec(
            self._checkpoint_integrand(),
            abs_tol=0.2,
            n_init=32,
            n_limit=2048,
        )
        _, checkpoint = loose_sc.integrate()
        old_n_total = int(checkpoint.n_total)

        resumed_sc = CubMCCLTVec(
            self._checkpoint_integrand(),
            abs_tol=0.05,
            n_init=32,
            n_limit=2048,
        )
        _, resumed = resumed_sc.integrate(resume=checkpoint)

        fresh_sc = CubMCCLTVec(
            self._checkpoint_integrand(),
            abs_tol=0.05,
            n_init=32,
            n_limit=2048,
        )
        _, fresh = fresh_sc.integrate()

        self.assertGreater(int(resumed.n_total), old_n_total)
        self.assertEqual(int(resumed.n_total), int(fresh.n_total))
        self.assertTrue(np.allclose(resumed.xfull, fresh.xfull))
        self.assertTrue(np.allclose(resumed.yfull, fresh.yfull))

    def test_resume_after_load_keeps_rng_stream(self):
        qmc_n_init = 2**8
        loose_sc = CubQMCLatticeG(
            Keister(Lattice(dimension=2, seed=13)),
            abs_tol=0.2,
            n_init=qmc_n_init,
            n_limit=2048,
        )
        _, checkpoint = loose_sc.integrate()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "resume.pkl.gz"
            checkpoint.save(path, compress=True)
            loaded = Data.load(path)

        old_n_total = int(loaded.n_total)
        old_xfull = np.array(loaded.xfull, copy=True)
        tight_sc = CubQMCLatticeG(
            Keister(Lattice(dimension=2, seed=13)),
            abs_tol=0.05,
            n_init=qmc_n_init,
            n_limit=2048,
        )
        _, resumed = tight_sc.integrate(resume=loaded)
        self.assertIsNot(resumed, loaded)
        self.assertEqual(int(loaded.n_total), old_n_total)
        self.assertTrue(np.array_equal(loaded.xfull, old_xfull))
        self.assertTrue(np.array_equal(resumed.xfull[:old_n_total], old_xfull))
        n_new = int(resumed.n_total) - old_n_total
        if n_new > 0:
            new_samples = resumed.xfull[old_n_total:]
            repeated_previous_sample = np.any(
                np.all(new_samples[:, None, :] == old_xfull[None, :, :], axis=-1)
            )
            self.assertFalse(repeated_previous_sample)


########################################################################
# Resume Validation Helper Tests
########################################################################
class TestResumeValidationHelpers(unittest.TestCase):
    """Fine-grained tests for AbstractStoppingCriterion._validate_resume_* helpers."""

    def _make_sc(self, n_limit=4096):
        return CubQMCLatticeG(
            Keister(Lattice(dimension=2, seed=7)),
            abs_tol=0.1,
            n_init=2**8,
            n_limit=n_limit,
        )

    def _make_checkpoint(self, n_limit=4096):
        sc = CubQMCLatticeG(
            Keister(Lattice(dimension=2, seed=7)),
            abs_tol=0.2,
            n_init=2**8,
            n_limit=n_limit,
        )
        _, checkpoint = sc.integrate()
        return checkpoint

    def test_resume_value_equal_matching_lists(self):
        sc = self._make_sc()
        self.assertTrue(sc._resume_value_equal([1, 2, 3], [1, 2, 3]))
        self.assertFalse(sc._resume_value_equal([1, 2], [1, 3]))

    def test_resume_value_equal_matching_dicts(self):
        sc = self._make_sc()
        self.assertTrue(sc._resume_value_equal({"a": 1, "b": 2}, {"a": 1, "b": 2}))
        self.assertFalse(sc._resume_value_equal({"a": 1}, {"a": 2}))

    def test_resume_value_equal_numpy_fallback(self):
        sc = self._make_sc()
        # Simulate older numpy where equal_nan kwarg raises TypeError
        original_array_equal = np.array_equal
        call_count = {"n": 0}

        def patched_array_equal(*args, **kwargs):
            if "equal_nan" in kwargs and call_count["n"] == 0:
                call_count["n"] += 1
                raise TypeError("equal_nan not supported")
            return original_array_equal(*args)

        with patch("numpy.array_equal", side_effect=patched_array_equal):
            result = sc._resume_value_equal(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        self.assertTrue(result)

    def test_validate_resume_object_saved_none(self):
        sc = self._make_sc()
        with self.assertRaisesRegex(ParameterError, "missing test_label state"):
            sc._validate_resume_object("test_label", object(), None, ())

    def test_validate_resume_obj_type_mismatch(self):
        sc = self._make_sc()
        with self.assertRaisesRegex(ParameterError, "incompatible test_label type"):
            sc._validate_resume_object("test_label", 1, "one", ())

    def test_validate_resume_object_missing_attr(self):
        sc = self._make_sc()

        class Obj:
            pass

        current = Obj()
        current.x = 1
        saved = Obj()  # no .x attribute
        with self.assertRaisesRegex(ParameterError, "missing test_label attribute"):
            sc._validate_resume_object("test_label", current, saved, ("x",))

    def test_resume_data_rejects_bad_format_ver(self):
        checkpoint = self._make_checkpoint()
        checkpoint._resume_format_version = "not_a_number"
        sc = self._make_sc()
        with self.assertRaisesRegex(ParameterError, "invalid _resume_format_version"):
            sc._validate_resume_data(checkpoint)

    def test_validate_resume_data_wrong_sc_type(self):
        checkpoint = self._make_checkpoint()
        # Replace stopping_crit with a different type
        checkpoint.stopping_crit = CubQMCNetG(
            Keister(DigitalNetB2(dimension=2, seed=7)),
            abs_tol=0.2,
            n_init=2**8,
            n_limit=4096,
        )
        sc = self._make_sc()
        with self.assertRaisesRegex(ParameterError, "was generated by"):
            sc._validate_resume_data(checkpoint)

    def test_validate_resume_data_n_total_limit(self):
        checkpoint = self._make_checkpoint(n_limit=4096)
        # Build a normal sc; then artificially lower n_limit below n_total
        sc = self._make_sc(n_limit=4096)
        sc.n_limit = max(1, int(checkpoint.n_total) - 1)
        with self.assertRaisesRegex(ParameterError, "exceeds current n_limit"):
            sc._validate_resume_data(checkpoint)

    def test_restore_resume_rng_state_missing(self):
        sc = self._make_sc()
        data = type("Data", (), {"discrete_distrib": type("NoRNG", (), {})()})()
        with self.assertRaisesRegex(ParameterError, "missing discrete distribution RNG state"):
            sc._restore_resume_rng_state(data)


########################################################################
# Diagnostics Optional Column Tests
########################################################################
class TestDiagnosticsOptionalColumns(unittest.TestCase):
    """Tests for optional columns and edge-case paths in diagnostics helpers."""

    def _make_logger(self, verbose=False, print_live=True):
        sc = type(
            "SC",
            (),
            {
                "trace_iterations": True,
                "trace_label": "",
                "verbose": verbose,
                "trace_print": print_live,
            },
        )()
        return _IterationTraceLogger(sc)

    @staticmethod
    def _load_resume_util():
        resume_util_path = (
            Path(__file__).resolve().parents[1]
            / "demos"
            / "demo_resume_data"
            / "resume_util.py"
        )
        spec = importlib.util.spec_from_file_location(
            "demo_resume_util", resume_util_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _make_rich_data(self):
        """Data object with all optional diagnostic fields set to non-None values."""
        return type(
            "Data",
            (),
            {
                "_iter_count": 1,
                "solution": 1.5,
                "bound_diff": 0.01,
                "comb_bound_diff": 0.02,
                "bound_half_width": 0.03,
                "bias_estimate": 0.001,
                "rmse_estimate": 0.005,
                "rmse_tol": 0.01,
                "n_min": 64,
                "n_total": 128,
                "m": 8,
                "xfull": np.zeros((128, 2)),
            },
        )()

    def test_enable_diagnostics_verbose_true(self):
        resume_util = self._load_resume_util()
        sc = type("SC", (), {})()
        out = resume_util.enable_diagnostics(sc, "demo", verbose=True)
        self.assertIs(out, sc)
        self.assertTrue(sc.trace_iterations)
        self.assertEqual(sc.trace_label, "demo")
        self.assertTrue(sc.verbose)
        self.assertFalse(sc.trace_print)

    def test_enable_diagnostics_verbose_false(self):
        resume_util = self._load_resume_util()
        sc = type("SC", (), {})()
        out = resume_util.enable_diagnostics(sc, "demo", verbose=False)
        self.assertIs(out, sc)
        self.assertTrue(sc.trace_iterations)
        self.assertEqual(sc.trace_label, "demo")
        self.assertFalse(sc.verbose)
        self.assertFalse(sc.trace_print)

    def test_enable_diagnostics_print_live_true(self):
        resume_util = self._load_resume_util()
        sc = type("SC", (), {})()
        out = resume_util.enable_diagnostics(sc, "demo", verbose=False, print_live=True)
        self.assertIs(out, sc)
        self.assertTrue(sc.trace_print)

    def test_collect_resume_fresh_warnings_none(self):
        resume_util = self._load_resume_util()
        resume_stage = {"total_n": 128, "solution": 1.0, "tol": 0.1}
        fresh_stage = {"total_n": 128, "solution": 1.15}
        warning_lines = resume_util.collect_resume_fresh_warnings("DemoSolver", resume_stage, fresh_stage)
        self.assertEqual(warning_lines, [])

    def test_collect_resume_fresh_warn_mismatch(self):
        resume_util = self._load_resume_util()
        resume_stage = {"total_n": 256, "solution": 1.5, "tol": 0.1}
        fresh_stage = {"total_n": 128, "solution": 1.2}
        warning_lines = resume_util.collect_resume_fresh_warnings("DemoSolver", resume_stage, fresh_stage)
        self.assertEqual(len(warning_lines), 2)
        self.assertIn("WARNING: DemoSolver: Inconsistent total samples across stages", warning_lines[0])
        self.assertIn("WARNING: DemoSolver: Resume and fresh solutions differ by more than 2 * tol", warning_lines[1])

    def test_write_combined_report_warnings(self):
        resume_util = self._load_resume_util()
        resume_rows = [{
            "name": "DemoSolver",
            "status": "skip",
            "resume": {"total_n": 256, "solution": 1.5, "tol": 0.1},
        }]
        fresh_rows = [{
            "name": "DemoSolver",
            "status": "skip",
            "fresh": {"total_n": 128, "solution": 1.2},
        }]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.txt"
            stream = io.StringIO()
            with redirect_stdout(stream):
                resume_util.write_combined_report(path, "Demo Title", resume_rows, fresh_rows)
            report_text = path.read_text(encoding="utf-8")
        self.assertIn("WARNING: DemoSolver: Inconsistent total samples across stages", report_text)
        self.assertIn("WARNING: DemoSolver: Resume and fresh solutions differ by more than 2 * tol", report_text)
        stdout_text = stream.getvalue()
        self.assertIn("WARNING: DemoSolver: Inconsistent total samples across stages", stdout_text)
        self.assertIn("WARNING: DemoSolver: Resume and fresh solutions differ by more than 2 * tol", stdout_text)

    def test_print_stage_summary_from_histories(self):
        resume_util = self._load_resume_util()
        resume_history = _IterationHistoryTable()
        for stage, row in [
            ("ITER", {"iter": 1, "solution": 1.0, "abs_tol": 1e-2, "comb_bound_diff": 4e-2, "n_min": 0, "n_total": 16, "m": 4, "xfull.shape": (16, 2)}),
            ("ITER", {"iter": 2, "solution": 1.1, "abs_tol": 1e-2, "comb_bound_diff": 2e-2, "n_min": 16, "n_total": 32, "m": 5, "xfull.shape": (32, 2)}),
            ("RESUME", {"iter": 2, "solution": 1.1, "abs_tol": 1e-3, "comb_bound_diff": 2e-2, "n_min": 32, "n_total": 32, "m": 5, "xfull.shape": (32, 2)}),
            ("ITER", {"iter": 3, "solution": 1.125, "abs_tol": 1e-3, "comb_bound_diff": 6e-3, "n_min": 32, "n_total": 64, "m": 6, "xfull.shape": (64, 2)}),
        ]:
            resume_history._append(stage, row, visible_columns=("stage", "iter", "solution", "comb_bound_diff", "n_min", "n_total", "m", "xfull.shape"), printed=True)
        fresh_history = _IterationHistoryTable()
        for stage, row in [
            ("ITER", {"iter": 1, "solution": 1.0, "abs_tol": 1e-3, "comb_bound_diff": 4e-2, "n_min": 0, "n_total": 16, "m": 4, "xfull.shape": (16, 2)}),
            ("ITER", {"iter": 2, "solution": 1.1, "abs_tol": 1e-3, "comb_bound_diff": 2e-2, "n_min": 16, "n_total": 32, "m": 5, "xfull.shape": (32, 2)}),
            ("ITER", {"iter": 3, "solution": 1.125, "abs_tol": 1e-3, "comb_bound_diff": 6e-3, "n_min": 32, "n_total": 64, "m": 6, "xfull.shape": (64, 2)}),
        ]:
            fresh_history._append(stage, row, visible_columns=("stage", "iter", "solution", "comb_bound_diff", "n_min", "n_total", "m", "xfull.shape"), printed=True)
        resume_solver = type("ResumeSolver", (), {"iteration_history": resume_history})()
        fresh_solver = type("FreshSolver", (), {"iteration_history": fresh_history})()
        rows = resume_util.stage_summary_rows_from_histories(
            resume_solver,
            loose_data=type("LooseData", (), {"time_integrate": 0.1})(),
            resume_data=type("ResumeData", (), {"time_integrate": 0.2})(),
            fresh_solver=fresh_solver,
            fresh_data=type("FreshData", (), {"time_integrate": 0.3})(),
        )
        self.assertEqual(rows[0][:5], ("Loose", 1e-2, 32, 32, 2))
        self.assertEqual(rows[1][:5], ("Resumed", 1e-3, 64, 32, 3))
        self.assertEqual(rows[2][:5], ("Fresh", 1e-3, 64, 64, 3))
        stream = io.StringIO()
        with redirect_stdout(stream):
            resume_util.print_stage_summary(
                resume_solver=resume_solver,
                loose_data=type("LooseData", (), {"time_integrate": 0.1})(),
                resume_data=type("ResumeData", (), {"time_integrate": 0.2})(),
                fresh_solver=fresh_solver,
                fresh_data=type("FreshData", (), {"time_integrate": 0.3})(),
            )
        output = stream.getvalue()
        self.assertIn("Loose", output)
        self.assertIn("Resumed", output)
        self.assertIn("Fresh", output)
        self.assertIn("32", output)
        self.assertIn("64", output)

    def test_stage_summary_rows_from_records(self):
        resume_util = self._load_resume_util()
        loose_stage = {"tol": 1e-2, "tol_name": "abs_tol", "total_n": 32, "new_n": 32, "iters": 2, "solution": 1.1, "half_width": 1e-2, "time": 0.1}
        resume_stage = {"tol": 1e-3, "tol_name": "abs_tol", "total_n": 64, "new_n": 32, "iters": 3, "solution": 1.125, "half_width": 3e-3, "time": 0.2}
        fresh_stage = {"tol": 1e-3, "tol_name": "abs_tol", "total_n": 64, "new_n": 64, "iters": 3, "solution": 1.125, "half_width": 3e-3, "time": 0.3}
        rows = resume_util.stage_summary_rows_from_stage_records(
            loose_stage, resume_stage, fresh_stage
        )
        self.assertEqual(rows[0][:5], ("Loose", 1e-2, 32, 32, 2))
        self.assertEqual(rows[1][:5], ("Resumed", 1e-3, 64, 32, 3))
        self.assertEqual(rows[2][:5], ("Fresh", 1e-3, 64, 64, 3))
        self.assertEqual(
            resume_util.stage_summary_tol_header(loose_stage, resume_stage, fresh_stage),
            "abs_tol",
        )

    def test_print_diagnostic_auto_opt_cols(self):
        """visible_columns=None auto-detects and displays all optional columns."""
        data = self._make_rich_data()
        stream = io.StringIO()
        with redirect_stdout(stream):
            _print_diagnostic("ITER", data, table_header=True, visible_columns=None)
        output = stream.getvalue()
        for col in ("bias_estimate", "rmse_tol", "comb_bound_diff", "bound_half_width"):
            self.assertIn(col, output, msg=f"Expected column '{col}' in output header")

    def test_print_diagnostic_tolerance_decimals(self):
        """A tolerance on data drives the solution decimal-place computation."""
        data = type(
            "Data",
            (),
            {
                "_iter_count": 1,
                "solution": 1.23456789,
                "n_total": 64,
                "abs_tol": 0.001,
            },
        )()
        stream = io.StringIO()
        with redirect_stdout(stream):
            _print_diagnostic("START", data)
        output = stream.getvalue()
        self.assertIn("1.", output)

    def test_print_diagnostic_bound_float_nan(self):
        """_format_bound handles actual floats and NaN values."""
        data = type(
            "Data",
            (),
            {
                "_iter_count": 1,
                "solution": 1.0,
                "n_total": 64,
                "bound_diff": 0.05,
                "bias_estimate": float("nan"),
                "comb_bound_diff": None,
                "bound_half_width": None,
                "rmse_estimate": None,
                "rmse_tol": None,
            },
        )()
        stream = io.StringIO()
        with redirect_stdout(stream):
            _print_diagnostic("ITER", data, table_header=True, visible_columns=None)
        output = stream.getvalue()
        self.assertIn("5.000e-02", output)
        self.assertIn("nan", output)

    def test_print_diagnostic_bound_non_float(self):
        """_format_bound handles a value that cannot be cast to float."""
        data = type(
            "Data",
            (),
            {
                "_iter_count": 1,
                "solution": 1.0,
                "n_total": 64,
                "bound_diff": "non-numeric",
            },
        )()
        stream = io.StringIO()
        with redirect_stdout(stream):
            _print_diagnostic("ITER", data, table_header=True, visible_columns=None)
        output = stream.getvalue()
        self.assertIn("non-numeric", output)

    def test_emit_without_iter_sets_none(self):
        """emit() with increment=False, iter_value=None sets data._iter_count=None."""
        logger = self._make_logger()
        data = type("Data", (), {"solution": 1.0, "n_total": 10})()
        stream = io.StringIO()
        with redirect_stdout(stream):
            logger.emit("START", data, increment=False, iter_value=None)
        self.assertIsNone(data._iter_count)

    def test_trace_print_false_suppresses_stdout(self):
        logger = self._make_logger(print_live=False)
        data = type("Data", (), {"solution": 1.0, "n_total": 10})()
        stream = io.StringIO()
        with redirect_stdout(stream):
            logger.iteration(data)
        self.assertEqual(stream.getvalue(), "")
        self.assertEqual(len(logger.history), 1)
        self.assertTrue(logger.history["printed"][0])

    def test_trace_history_stores_solver_tol(self):
        sc = type(
            "SC",
            (),
            {
                "trace_iterations": True,
                "trace_label": "",
                "verbose": False,
                "abs_tol": 1e-3,
            },
        )()
        logger = _IterationTraceLogger(sc)
        data = type("Data", (), {"solution": 1.0, "n_total": 10})()
        with redirect_stdout(io.StringIO()):
            logger.iteration(data)
        self.assertEqual(logger.history["abs_tol"][0], 1e-3)

    def test_resume_non_int_iter_count(self):
        """resume() silently ignores a non-int _iter_count on the data."""
        logger = self._make_logger()
        data = type(
            "Data",
            (),
            {"_iter_count": object(),  # cannot be converted to int
                "solution": 1.0,
                "n_total": 10,
            },
        )()
        stream = io.StringIO()
        with redirect_stdout(stream):
            logger.resume(data)  # must not raise

    def test_log_all_none_columns(self):
        # drop_empty_columns=True removes columns that are all NaN
        history = _IterationHistoryTable()
        history._append("ITER",
                        {"iter": 1, "solution": 1.0, "n_total": 16, "bias_estimate": None},
                        visible_columns=("stage", "iter", "solution", "n_total", "bias_estimate"),
                        printed=True,
                        )
        df_dropped = _get_iteration_log_frame(history, drop_empty_columns=True, formatted=False)
        df_kept = _get_iteration_log_frame(history, drop_empty_columns=False, formatted=False)
        self.assertNotIn("bias_estimate", df_dropped.columns)
        self.assertIn("bias_estimate", df_kept.columns)

if __name__ == "__main__":
    unittest.main()
