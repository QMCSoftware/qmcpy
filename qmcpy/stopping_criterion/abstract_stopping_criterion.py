import copy
import sys
from typing import TYPE_CHECKING

from .diagnostics import (  # noqa: F401
    _IterationTraceLogger,
    _format_iteration_log,
    _get_iteration_log_frame,
    _print_iteration_log,
)
from ..integrand.abstract_integrand import AbstractIntegrand
from ..util import (
    DistributionCompatibilityError,
    ParameterError,
    MethodImplementationError,
    _univ_repr,
)
import numpy as np

if TYPE_CHECKING:
    import pandas


class AbstractStoppingCriterion(object):
    _RESUME_FORMAT_VERSION = 1  # Increment when checkpoint format changes in a non-backwards-compatible way
    _ITERATION_LOG_VIEWS = ("all", "current", "without_resume", "stage_last")

    def __init__(self, allowed_distribs, allow_vectorized_integrals):
        """Initialize a stopping criterion base class.

        Args:
            allowed_distribs (list): Allowed discrete distribution classes.
            allow_vectorized_integrals (bool): Whether to allow integrands with
                vectorized outputs, meaning ``integrand.d_indv != ()``.
        """
        sname = type(self).__name__
        prefix = "A concrete implementation of StoppingCriterion must have "
        # integrand check
        if (not hasattr(self, "integrand")) or (
            not isinstance(self.integrand, AbstractIntegrand)
        ):
            raise ParameterError(prefix + "self.integrand, an Integrand instance")
        # true measure check
        if (not hasattr(self, "true_measure")) or (
            self.true_measure != self.integrand.true_measure
        ):
            raise ParameterError(
                prefix + "self.true_measure=self.integrand.true_measure"
            )
        # discrete distribution check
        if (not hasattr(self, "discrete_distrib")) or (
            self.discrete_distrib != self.integrand.discrete_distrib
        ):
            raise ParameterError(
                prefix + "self.discrete_distrib=self.integrand.discrete_distrib"
            )
        if not isinstance(self.discrete_distrib, tuple(allowed_distribs)):
            raise DistributionCompatibilityError(
                "%s must have a DiscreteDistribution in %s"
                % (sname, str(allowed_distribs))
            )
        if (not allow_vectorized_integrals) and len(self.integrand.d_indv) > 0:
            raise ParameterError(
                "Vectorized integrals (with d_indv!=() outputs per sample) are not supported by this stopping criterion"
            )
        # parameter checks
        if not hasattr(self, "parameters"):
            self.parameters = []
        self.elapsed_time = float(getattr(self, "elapsed_time", 0.0))

    def integrate(self, resume=None) -> tuple:
        """Determine the samples needed to satisfy the target tolerance.

        Args:
            resume (Data, optional): Existing integration state to resume from,
                if supported. A valid resume checkpoint must continue the same
                numerical experiment without duplicating samples, losing
                accumulated statistics, or weakening the requested tolerance
                guarantee. Supported resume implementations validate and copy
                the checkpoint before mutating any state so the caller's saved
                object is preserved. Defaults to None.

        Returns:
            tuple[Union[float, np.ndarray], Data]: Approximation to the integral
                with shape ``integrand.d_comb`` and the corresponding data
                object.
        """
        raise MethodImplementationError(self, "integrate")

    def _make_trace_logger(self) -> _IterationTraceLogger:
        """Create an iteration trace logger for this stopping criterion.

        Returns:
            _IterationTraceLogger: Trace logger configured from the stopping
                criterion's optional trace attributes.
        """
        requested_trace_iterations = bool(getattr(self, "trace_iterations", False))
        trace_verbose = bool(getattr(self, "verbose", False))
        had_store = hasattr(self, "_trace_store_iterations")
        previous_store = getattr(self, "_trace_store_iterations", None)
        had_store_all = hasattr(self, "_trace_store_all_iterations")
        previous_store_all = getattr(self, "_trace_store_all_iterations", None)
        had_print = hasattr(self, "_trace_print_iterations")
        previous_print = getattr(self, "_trace_print_iterations", None)
        self._trace_store_iterations = True
        self._trace_store_all_iterations = requested_trace_iterations and trace_verbose
        self._trace_print_iterations = requested_trace_iterations
        try:
            return _IterationTraceLogger(self)
        finally:
            if had_store:
                self._trace_store_iterations = previous_store
            else:
                delattr(self, "_trace_store_iterations")
            if had_store_all:
                self._trace_store_all_iterations = previous_store_all
            else:
                delattr(self, "_trace_store_all_iterations")
            if had_print:
                self._trace_print_iterations = previous_print
            else:
                delattr(self, "_trace_print_iterations")

    def get_iteration_log(
        self,
        history=None,
        printed_only=True,
        drop_empty_columns=True,
        formatted=True,
        view="all",
    ) -> "pandas.DataFrame":
        """Return the latest iteration log as a pandas DataFrame.

        Args:
            history (list[dict] | None): Iteration history to format. If ``None``,
                uses ``self.iteration_history`` when available.
            printed_only (bool): If ``True``, include only rows that were
                selected for printed output.
            drop_empty_columns (bool): If ``True``, drop columns with no values.
            formatted (bool): If ``True``, return formatted display values when
                available.
            view (str): Which log view to return. ``"all"`` and ``"current"``
                both return the full current log, ``"without_resume"`` removes
                ``RESUME`` rows, and ``"stage_last"`` returns one stop row per
                stage by taking the last non-``RESUME`` row available for that
                stage.

        Returns:
            pandas.DataFrame: DataFrame representation of the iteration log.
        """
        self._validate_iteration_log_view(view)
        use_cache = (
            history is None
            and printed_only
            and drop_empty_columns
            and formatted
            and view in ("all", "current")
        )
        if use_cache:
            cached_history_df = getattr(self, "history_df", None)
            if cached_history_df is not None:
                return cached_history_df
        if history is None:
            history = getattr(self, "iteration_history", None)
        result = _get_iteration_log_frame(
            history,
            stopping_criterion=self,
            printed_only=printed_only,
            drop_empty_columns=False,
            formatted=formatted,
        )
        result = self._apply_iteration_log_view(result, view)
        if drop_empty_columns and not result.empty:
            result = result.dropna(axis=1, how="all")
        if use_cache:
            # Cache for future calls and populate the last data object so that
            # data.history_df is available after calling get_iteration_log().
            self.history_df = result
            last_data = getattr(self, "_last_finalized_data", None)
            if last_data is not None:
                last_data.history_df = result
        return result

    @classmethod
    def _validate_iteration_log_view(cls, view):
        if view not in cls._ITERATION_LOG_VIEWS:
            raise ParameterError(
                "get_iteration_log view must be one of %s"
                % (cls._ITERATION_LOG_VIEWS,)
            )

    @staticmethod
    def _apply_iteration_log_view(log_df, view):
        if view in ("all", "current") or log_df.empty or "stage" not in log_df.columns:
            return log_df.reset_index(drop=True)
        if view == "without_resume":
            return log_df.loc[log_df["stage"] != "RESUME"].reset_index(drop=True)
        non_resume_indices = log_df.index[log_df["stage"] != "RESUME"].to_numpy()
        if len(non_resume_indices) == 0:
            return log_df.iloc[0:0].reset_index(drop=True)
        stage_end_indices = log_df.index[log_df["stage"] == "RESUME"].tolist()
        stage_end_indices.append(log_df.index[-1])
        positions = np.searchsorted(non_resume_indices, stage_end_indices, side="right") - 1
        positions = positions[positions >= 0]
        return log_df.loc[non_resume_indices[positions]].reset_index(drop=True)

    def format_iteration_log(self, history=None, printed_only=True, include_header=True) -> str:
        """Return the iteration log as formatted text.

        Args:
            history (IterationHistoryTable | None, optional): Iteration history
                to format. If ``None``, uses ``self.iteration_history`` when
                available. Defaults to None.
            printed_only (bool, optional): If ``True``, include only rows that
                were selected for printed output. Defaults to True.
            include_header (bool, optional): If ``True``, include the trace
                label header before the table. Defaults to True.

        Returns:
            str: Formatted iteration log text.
        """
        if history is None:
            history = getattr(self, "iteration_history", None)
        return _format_iteration_log(
            history,
            stopping_criterion=self,
            printed_only=printed_only,
            include_header=include_header,
        )

    def print_iteration_log(self, history=None, printed_only=True, include_header=True, file=None) -> None:
        """Print the iteration log for the latest run or supplied history.

        Args:
            history (IterationHistoryTable | None, optional): Iteration history
                to print. If ``None``, uses ``self.iteration_history`` when
                available. Defaults to None.
            printed_only (bool, optional): If ``True``, print only rows that
                were selected for printed output. Defaults to True.
            include_header (bool, optional): If ``True``, include the trace
                label header before the table. Defaults to True.
            file (typing.TextIO | None, optional): Output stream. Defaults to
                ``sys.stdout`` when None.

        Returns:
            None: This method writes output to ``file``.
        """
        if history is None:
            history = getattr(self, "iteration_history", None)
        return _print_iteration_log(
            history,
            stopping_criterion=self,
            printed_only=printed_only,
            include_header=include_header,
            file=file,
        )

    def _prepare_resume_data(self, resume, validate_resume, restore_resume):
        """Validate and restore a resume checkpoint before integration.

        Args:
            resume (Data or None): Resume checkpoint passed to ``integrate``.
            validate_resume (callable): Validator taking ``resume`` and raising
                on incompatible state.
            restore_resume (callable): Restorer taking ``resume`` and mutating
                the current stopping criterion into a compatible resumed state.

        Returns:
            Data or None: A validated deep copy of the supplied checkpoint, or
            None when no checkpoint was supplied.
        """
        if resume is None:
            return None
        validate_resume(resume)
        data = copy.deepcopy(resume)
        self._detach_resume_stopping_criterion_history(data)
        restore_resume(data)
        return data

    @staticmethod
    def _detach_resume_stopping_criterion_history(data):
        """Detach copied solver-owned history caches while preserving checkpoint history."""
        stopping_crit = getattr(data, "stopping_crit", None)
        if stopping_crit is None:
            return
        if hasattr(stopping_crit, "iteration_history"):
            stopping_crit.iteration_history = None
        if hasattr(stopping_crit, "history_df"):
            stopping_crit.history_df = None

    def _restore_resume_state(self, data):
        """Optional hook for subclasses to align state before resuming.

        Subclasses that need to restore RNG state or rewrite checkpoint fields
        may override this method. The default implementation contains no operation.

        Args:
            data (Data): Deep-copied resume checkpoint that will be mutated by
                the resumed integration run.
        """
        return None

    def _capture_resume_provenance(self, resume):
        """Capture resume bookkeeping before the live ``Data`` object is mutated.

        Args:
            resume (Data or None): Resume checkpoint passed to ``integrate``.

        Returns:
            dict or None: Previous sample/time totals, or None for fresh runs.
        """
        if resume is None:
            return None
        previous_total_time = getattr(
            resume,
            "elapsed_time",
            getattr(resume, "time_integrate_total", getattr(resume, "time_integrate", 0.0)),
        )
        return {
            "n_total": int(getattr(resume, "n_total", 0)),
            "time_integrate_total": float(previous_total_time),
        }

    @staticmethod
    def _resume_elapsed_time(resume_provenance=None):
        if resume_provenance is None:
            return 0.0
        return float(resume_provenance.get("time_integrate_total", 0.0))

    def _set_elapsed_time(self, data, elapsed, resume_provenance=None):
        elapsed_time = self._resume_elapsed_time(resume_provenance) + float(elapsed)
        self.elapsed_time = elapsed_time
        if data is not None:
            data.elapsed_time = elapsed_time
        return elapsed_time

    @staticmethod
    def _qualified_class_name(obj):
        """Return a stable fully-qualified class name for audit metadata."""
        cls = type(obj)
        return "%s.%s" % (cls.__module__, cls.__name__)

    @staticmethod
    def _qmcpy_version():
        """Best-effort lookup of the loaded QMCPy version string."""
        qmcpy_module = sys.modules.get("qmcpy")
        return getattr(qmcpy_module, "__version__", None)

    def _annotate_checkpoint_metadata(self, data):
        """Attach lightweight checkpoint metadata used for auditing/resume."""
        data._qmcpy_version = self._qmcpy_version()
        data._resume_format_version = int(self._RESUME_FORMAT_VERSION)
        data._stopping_criterion_class = self._qualified_class_name(self)
        data._integrand_class = self._qualified_class_name(self.integrand)
        data._true_measure_class = self._qualified_class_name(self.true_measure)
        data._discrete_distribution_class = self._qualified_class_name(
            self.discrete_distrib
        )

    def _finalize_integration_data(self, data, elapsed, resume_provenance=None):
        """Attach shared integration metadata before returning ``Data``.

        Args:
            data (Data): Integration state to finalize.
            elapsed (float): Wall-clock time spent in the current ``integrate``
                call.
            resume_provenance (dict or None, optional): Output of
                :meth:`_capture_resume_provenance`. Defaults to None.
        """
        data.stopping_crit = self
        data.integrand = self.integrand
        data.true_measure = self.integrand.true_measure
        data.discrete_distrib = self.true_measure.discrete_distrib
        data.time_integrate = float(elapsed)
        previous_time = 0.0
        previous_n_total = 0
        if resume_provenance is not None:
            previous_time = float(resume_provenance.get("time_integrate_total", 0.0))
            previous_n_total = int(resume_provenance.get("n_total", 0))
        data.resumed = resume_provenance is not None
        data.n_resume_from = previous_n_total
        data.time_integrate_previous = previous_time
        data.time_integrate_resume = float(elapsed)
        data.time_integrate_total = previous_time + float(elapsed)
        data.elapsed_time = data.time_integrate_total
        self.elapsed_time = data.elapsed_time
        data.iteration_history = getattr(self, "iteration_history", None)
        data.history_df = getattr(self, "history_df", None)
        self._annotate_checkpoint_metadata(data)

    def _resume_value_equal(self, current, saved):
        """Deep equality check tolerant of arrays, lists, dicts, and QMCPy objects.

        Args:
            current: Value from the live stopping criterion.
            saved: Value from the resume checkpoint.

        Returns:
            bool: True when the two values are considered equal.
        """
        if isinstance(current, np.ndarray) or isinstance(saved, np.ndarray):
            try:
                return np.array_equal(
                    np.asarray(current), np.asarray(saved), equal_nan=True
                )
            except TypeError:
                return np.array_equal(np.asarray(current), np.asarray(saved))
        if isinstance(current, (list, tuple)) and isinstance(saved, (list, tuple)):
            return len(current) == len(saved) and all(
                self._resume_value_equal(curr_i, saved_i)
                for curr_i, saved_i in zip(current, saved)
            )
        if isinstance(current, dict) and isinstance(saved, dict):
            return current.keys() == saved.keys() and all(
                self._resume_value_equal(current[key], saved[key]) for key in current
            )
        if (
            type(current) is type(saved)
            and hasattr(current, "parameters")
            and hasattr(saved, "parameters")
        ):
            return str(current) == str(saved)
        try:
            is_equal = current == saved
        except (TypeError, ValueError, NotImplementedError):
            is_equal = str(current) == str(saved)
        if isinstance(is_equal, np.ndarray):
            return bool(np.all(is_equal))
        return bool(is_equal)

    def _require_resume_attrs(self, data, attrs):
        """Raise ParameterError if any attribute in *attrs* is absent from *data*.

        Args:
            data (Data): Resume checkpoint.
            attrs (tuple[str, ...]): Required attribute names.

        Raises:
            ParameterError: If one or more attributes are missing.
        """
        missing = [attr for attr in attrs if not hasattr(data, attr)]
        if missing:
            raise ParameterError(
                "resume data missing required attribute(s): %s"
                % ", ".join(sorted(missing))
            )

    def _validate_resume_object(self, label, current, saved, attrs):
        """Validate that a saved sub-object is compatible with the current one.

        Checks type equality and then compares each attribute listed in *attrs*
        using :meth:`_resume_value_equal`.

        Args:
            label (str): Human-readable name used in error messages.
            current: Live object (integrand, true measure, etc.).
            saved: Saved object from the resume checkpoint.
            attrs (tuple[str, ...]): Attribute names to compare.

        Raises:
            ParameterError: If the objects are incompatible.
        """
        if saved is None:
            raise ParameterError("resume data missing %s state." % label)
        if type(saved) is not type(current):
            raise ParameterError(
                "resume data has incompatible %s type %s; expected %s."
                % (label, type(saved).__name__, type(current).__name__)
            )
        missing = [
            attr for attr in attrs if not hasattr(current, attr) or not hasattr(saved, attr)
        ]
        if missing:
            raise ParameterError(
                "resume data missing %s attribute(s): %s"
                % (label, ", ".join(sorted(missing)))
            )
        for attr in attrs:
            if not self._resume_value_equal(getattr(current, attr), getattr(saved, attr)):
                raise ParameterError(
                    "resume data has incompatible %s.%s." % (label, attr)
                )

    def _validate_resume_data(self, data, required_fields=()):
        """Run standard cross-cutting resume compatibility checks.

        Validates stopping criterion type, integrand, true measure, discrete
        distribution, checkpoint format version, and ``n_total`` against
        ``n_limit``.

        Args:
            data (Data): Resume checkpoint to validate.
            required_fields (tuple[str, ...], optional): Additional attribute
                names that must be present on *data*. Defaults to ``()``.

        Raises:
            ParameterError: If any compatibility check fails.
        """
        self._require_resume_attrs(data, ("stopping_crit", "integrand", "true_measure", "discrete_distrib", "n_total") + tuple(required_fields))
        resume_format_version = getattr(
            data, "_resume_format_version", self._RESUME_FORMAT_VERSION
        )
        try:
            resume_format_version = int(resume_format_version)
        except (TypeError, ValueError) as exc:
            raise ParameterError("resume data has invalid _resume_format_version.") from exc
        if resume_format_version != self._RESUME_FORMAT_VERSION:
            raise ParameterError(
                "resume data uses checkpoint format version %d; expected %d."
                % (resume_format_version, self._RESUME_FORMAT_VERSION)
            )
        if type(data.stopping_crit) is not type(self):
            raise ParameterError("resume data was generated by %s, not %s." % (type(data.stopping_crit).__name__, type(self).__name__))
        self._validate_resume_object("integrand", self.integrand, data.integrand, ("d", "d_indv", "d_comb") + tuple(getattr(self.integrand, "parameters", [])))
        self._validate_resume_object("true_measure", self.true_measure, data.true_measure, ("d", "domain", "range") + tuple(getattr(self.true_measure, "parameters", [])))
        self._validate_resume_object(
            "discrete_distrib",
            self.discrete_distrib,
            data.discrete_distrib,
            ("d", "replications", "mimics", "entropy", "spawn_key")
            + tuple(getattr(self.discrete_distrib, "parameters", [])),
        )
        if int(data.n_total) > int(self.n_limit):
            raise ParameterError("resume data n_total=%d exceeds current n_limit=%d." % (int(data.n_total), int(self.n_limit)))

    def _validate_resume_with_state(self, data, required_fields=(), state_fields=()):
        """Validate resume data including algorithm-specific state fields.

        Calls :meth:`_validate_resume_data` and additionally checks that all
        *state_fields* are present and that ``n_total >= n_init``.

        Args:
            data (Data): Resume checkpoint to validate.
            required_fields (tuple[str, ...], optional): Extra data attributes
                required beyond the standard set. Defaults to ``()``.
            state_fields (tuple[str, ...], optional): Algorithm-state attributes
                that must also be present. Defaults to ``()``.

        Raises:
            ParameterError: If any compatibility check fails.
        """
        self._validate_resume_data(data, required_fields=required_fields)
        self._require_resume_attrs(data, state_fields)
        if hasattr(self, "n_init") and int(data.n_total) < int(self.n_init):
            raise ParameterError("resume data must include at least n_init samples.")

    def _validate_resume_shape(self, label, value, expected_shape):
        """Raise ParameterError when a resume array has the wrong shape."""
        actual_shape = np.shape(value)
        expected_shape = tuple(expected_shape)
        if actual_shape != expected_shape:
            raise ParameterError(
                "resume data %s shape must be %s; got %s."
                % (label, expected_shape, actual_shape)
            )

    @staticmethod
    def _is_power_of_two(n):
        """Return True when *n* is a positive power of two."""
        return n > 0 and (n & (n - 1)) == 0

    @staticmethod
    def _resolve_error_fun(error_fun):
        """Resolve an *error_fun* argument from a string keyword or callable.

        Args:
            error_fun (Union[str, callable]): ``'EITHER'`` or ``'BOTH'`` or a
                callable with signature ``(sv, abs_tol, rel_tol) -> tol``.

        Returns:
            tuple[callable, str or None]: The resolved callable and its canonical
                string key (``'EITHER'`` or ``'BOTH'``), or ``None`` when the
                input was already a callable.

        Raises:
            ParameterError: If a string argument is not ``'EITHER'`` or ``'BOTH'``.
        """
        _error_fun_key = None
        if isinstance(error_fun, str):
            _error_fun_key = error_fun.upper()
            if _error_fun_key == "EITHER":
                error_fun = lambda sv, abs_tol, rel_tol: np.maximum(abs_tol, abs(sv) * rel_tol)
            elif _error_fun_key == "BOTH":
                error_fun = lambda sv, abs_tol, rel_tol: np.minimum(abs_tol, abs(sv) * rel_tol)
            else:
                raise ParameterError("str error_fun must be 'EITHER' or 'BOTH'")
        return error_fun, _error_fun_key

    @staticmethod
    def _checkpoint_rmse_tol(data):
        """Extract the RMSE tolerance stored in a resume checkpoint.

        Args:
            data (Data): Resume checkpoint.

        Returns:
            float or None: The stored ``rmse_tol``, or ``None`` if absent or
                non-numeric.
        """
        for obj in (data, getattr(data, "stopping_crit", None)):
            try:
                return float(getattr(obj, "rmse_tol", None))
            except (TypeError, ValueError):
                pass
        return None

    def _init_control_variates(self, control_variates, control_variate_means):
        """Validate and store control variates and their means.

        Sets ``self.cv``, ``self.cv_mu``, and ``self.ncv`` after validating that
        every entry in *control_variates* is an ``AbstractIntegrand`` instance
        that shares the same discrete distribution and ``d_indv`` as the main
        integrand.

        Args:
            control_variates (list or AbstractIntegrand): Control variate
                integrand(s).
            control_variate_means (array-like): Known means of each control
                variate.

        Returns:
            int: Number of control variates (``self.ncv``).

        Raises:
            ParameterError: If any control variate is incompatible.
        """
        self.cv_mu = np.atleast_1d(control_variate_means)
        self.cv = control_variates
        if isinstance(self.cv, AbstractIntegrand):
            self.cv = [self.cv]
            self.cv_mu = self.cv_mu[None, ...]
        assert isinstance(self.cv, list), "cv must be a list of AbstractIntegrand objects"
        for cv in self.cv:
            if (
                (not isinstance(cv, AbstractIntegrand))
                or (cv.discrete_distrib != self.discrete_distrib)
                or (cv.d_indv != self.integrand.d_indv)
            ):
                raise ParameterError(
                    """
                        Each control variates discrete distribution must be an AbstractIntegrand instance 
                        with the same discrete distribution as the main integrand. d_indv must also match 
                        that of the main integrand instance for each control variate."""
                )
        self.ncv = len(self.cv)
        return self.ncv

    def _restore_resume_rng_state(self, data):
        """Deep-copy the saved RNG state into the live discrete distribution.

        Ensures that samples drawn after resuming are independent of those
        already stored in the checkpoint.

        Args:
            data (Data): Resume checkpoint carrying the saved distribution in
                ``data.discrete_distrib``.

        Raises:
            ParameterError: If either distribution lacks an ``rng`` attribute.
        """
        saved_distrib = data.discrete_distrib
        if not hasattr(saved_distrib, "rng") or not hasattr(self.discrete_distrib, "rng"):
            raise ParameterError("resume data is missing discrete distribution RNG state.")
        self.discrete_distrib.rng.bit_generator.state = copy.deepcopy(
            saved_distrib.rng.bit_generator.state
        )

    def _compute_indv_alphas(self, alphas_comb):
        """Distribute combined confidence levels to individual integrand dimensions.

        Uses the integrand dependency map to allocate the per-combined-output
        alpha budget down to each individual output dimension.

        Args:
            alphas_comb (np.ndarray): Per-combined-output confidence levels with
                shape ``integrand.d_comb``.

        Returns:
            tuple[np.ndarray, bool]: ``(alphas_indv, identity_dependency)``
                where *alphas_indv* has shape ``integrand.d_indv`` and
                *identity_dependency* is True when each combined output depends
                on exactly its matching individual output.
        """
        alphas_indv = np.tile(1, self.integrand.d_indv)
        identity_dependency = True
        for k in np.ndindex(self.integrand.d_comb):
            comb_flags = np.tile(True, self.integrand.d_comb)
            comb_flags[k] = False
            flags_indv = self.integrand.dependency(comb_flags)
            if (
                self.integrand.d_indv != self.integrand.d_comb
                or (flags_indv != comb_flags).any()
            ):
                identity_dependency = False
            dependents_k = ~flags_indv
            n_dep_k = dependents_k.sum()
            alpha_k = alphas_comb[k] / n_dep_k
            alpha_k_mat = alpha_k * dependents_k
            alphas_indv = np.where(alpha_k_mat == 0, alphas_indv, np.minimum(alpha_k_mat, alphas_indv))
        return alphas_indv, identity_dependency

    def set_tolerance(self, abs_tol=None, rel_tol=None, rmse_tol=None):
        """Reset the tolerances.

        Args:
            abs_tol (float): Absolute tolerance, when supported. If supplied,
                reset it; otherwise ignore it.
            rel_tol (float): Relative tolerance, when supported. If supplied,
                reset it; otherwise ignore it.
            rmse_tol (float): RMSE tolerance, when supported. If supplied,
                reset it; otherwise ignore it. If ``rmse_tol`` is not supplied
                but ``abs_tol`` is, then ``rmse_tol = abs_tol / norm.ppf(1 -
                alpha / 2)``.
        """
        raise MethodImplementationError(self, "set_tolerance")

    def __repr__(self):
        return _univ_repr(self, "AbstractStoppingCriterion", self.parameters)
