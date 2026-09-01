"""Utilities for resume demos and text-report generation."""

import math
from time import strftime

#################################################################
# Formatting and reporting helpers
#################################################################
def format_value(value, ndigits=8):
    """Format a scalar-like value for report output."""
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return str(value)


def _format_problem_inputs(stopping_criterion):
    """Build a compact problem-input summary for a stopping criterion."""
    integrand = getattr(stopping_criterion, "integrand", None)
    lines = []

    def add_scalar(label, value, *, ndigits=6):
        if value is None:
            return
        try:
            fval = float(value)
            if fval != 0 and abs(fval) < 10 ** (-ndigits):
                formatted = f"{fval:.2e}"
            else:
                formatted = format_value(value, ndigits=ndigits)
        except (TypeError, ValueError):
            formatted = str(value)
        lines.append(f"{label}: {formatted}")

    def add_int(label, value):
        if value is None:
            return
        try:
            lines.append(f"{label}: {int(value)}")
        except Exception:
            lines.append(f"{label}: {value}")

    add_scalar("abs_tol", getattr(stopping_criterion, "abs_tol", None))
    add_scalar("rel_tol", getattr(stopping_criterion, "rel_tol", None))
    _target_rmse_tol = getattr(stopping_criterion, "target_rmse_tol", None)
    if _target_rmse_tol is not None:
        add_scalar("target_rmse_tol", _target_rmse_tol)
    else:
        add_scalar("rmse_tol", getattr(stopping_criterion, "rmse_tol", None))
    add_int("n_init", getattr(stopping_criterion, "n_init", None))
    add_int("n_limit", getattr(stopping_criterion, "n_limit", None))
    add_int("levels_min", getattr(stopping_criterion, "levels_min", None))
    add_int("levels_max", getattr(stopping_criterion, "levels_max", None))
    if integrand is not None:
        add_int("dimension", getattr(integrand, "d", None))
    return "\n".join(lines)

#################################################################
# Result summaries and diagnostics
#################################################################
def _history_tol_value(row):
    """Return the most relevant tolerance stored on a history row."""
    for key in ("rel_tol", "abs_tol", "rmse_tol"):
        try:
            value = float(row.get(key, None))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            return value, key
    return None, None


def _history_half_width(row):
    """Infer a half-width style summary from a stored history row."""
    for key in ("bound_half_width", "bias_estimate", "rmse_estimate"):
        try:
            value = float(row.get(key, None))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    for key in ("comb_bound_diff", "bound_diff"):
        try:
            value = float(row.get(key, None))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value / 2
    return float("nan")


def _history_time_value(data):
    """Return a stage runtime from a QMCPy data object, or NaN if unavailable."""
    try:
        return float(getattr(data, "time_integrate", float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _stage_row_from_history(name, row, previous_n_total=0, time_value=float("nan")):
    """Convert a stored history row into a stage-summary tuple."""
    total_n = int(row.get("n_total", 0) or 0)
    tol, tol_name = _history_tol_value(row)
    return (
        name,
        tol,
        total_n,
        total_n - int(previous_n_total),
        row.get("iter", None),
        float(row.get("solution", float("nan"))),
        _history_half_width(row),
        float(time_value),
        tol_name,
    )


def stage_summary_rows_from_histories(
    resume_solver,
    *,
    loose_data=None,
    resume_data=None,
    fresh_solver=None,
    fresh_data=None,
):
    """Build stage-summary rows from solver iteration histories."""
    history = getattr(resume_solver, "iteration_history", None)
    if history is None or len(history) == 0:
        raise ValueError("resume_solver has no stored iteration history")
    rows = []
    resume_indices = [index for index, stage in enumerate(history["stage"]) if stage == "RESUME"]
    if resume_indices:
        resume_index = resume_indices[0]
        loose_index = max(
            (index for index in range(resume_index) if history["stage"][index] != "RESUME"),
            default=resume_index,
        )
        rows.append(
            _stage_row_from_history(
                "Loose",
                history._row(loose_index),
                previous_n_total=0,
                time_value=_history_time_value(loose_data),
            )
        )
        rows.append(
            _stage_row_from_history(
                "Resumed",
                history._row(len(history) - 1),
                previous_n_total=int(history["n_total"][resume_index] or 0),
                time_value=_history_time_value(resume_data),
            )
        )
    else:
        rows.append(
            _stage_row_from_history(
                "Loose" if fresh_solver is not None else "Run",
                history._row(len(history) - 1),
                previous_n_total=0,
                time_value=_history_time_value(loose_data if fresh_solver is not None else resume_data),
            )
        )
    if fresh_solver is not None:
        fresh_history = getattr(fresh_solver, "iteration_history", None)
        if fresh_history is None or len(fresh_history) == 0:
            raise ValueError("fresh_solver has no stored iteration history")
        rows.append(
            _stage_row_from_history(
                "Fresh",
                fresh_history._row(len(fresh_history) - 1),
                previous_n_total=0,
                time_value=_history_time_value(fresh_data),
            )
        )
    return rows


def stage_summary_rows_from_stage_records(loose_stage, resume_stage=None, fresh_stage=None):
    """Build stage-summary rows from stored stage-record dictionaries."""
    rows = []
    for name, stage in (("Loose", loose_stage), ("Resumed", resume_stage), ("Fresh", fresh_stage)):
        if not stage:
            continue
        rows.append(
            (
                name,
                stage.get("tol"),
                stage.get("total_n", 0),
                stage.get("new_n", 0),
                stage.get("iters"),
                stage.get("solution", float("nan")),
                stage.get("half_width", float("nan")),
                stage.get("time", 0.0),
            )
        )
    return rows


def stage_summary_tol_header(*stages):
    """Infer the preferred tolerance-column header from stage records."""
    for stage in stages:
        if not stage:
            continue
        tol_name = stage.get("tol_name")
        if tol_name == "target_rmse_tol":
            return "rmse_tol"
        if tol_name in {"abs_tol", "rel_tol", "rmse_tol"}:
            return tol_name
    return "abs_tol"


def format_stage_summary(rows, title="Stage summary", tol_header="abs_tol"):
    """Return a compact stage-by-stage sample and time summary table."""
    total_n_vals = [int(total_n) for _, _, total_n, *_ in rows]
    new_n_vals = [int(new_n) for _, _, _, new_n, *_ in rows]
    tol_values = [
        float(abs_tol)
        for _, abs_tol, *_ in rows
        if abs_tol is not None and float(abs_tol) > 0
    ]
    if tol_values:
        solution_decimals = int(math.ceil(-math.log10(min(tol_values)))) + 1
        solution_decimals = max(7, min(12, solution_decimals))
    else:
        solution_decimals = 8
    sol_w = solution_decimals + 4  # sign + leading digit + dot + decimals
    stage_w = 7
    # Keep rmse_tol values visually separated by one extra leading space.
    tol_w = max(7, len(tol_header) + (1 if tol_header == "rmse_tol" else 0))
    total_n_w = max(9, max((len(f"{value:,}") for value in total_n_vals), default=0))
    new_n_w = max(9, max((len(f"{value:,}") for value in new_n_vals), default=0))
    iters_w = 6
    hw_w = 10
    time_w = 8
    sep_len = stage_w + tol_w + total_n_w + new_n_w + iters_w + sol_w + hw_w + time_w + 7
    sep = "-" * sep_len
    lines = [
        f"\n{title}",
        sep,
        f"{'Stage':<{stage_w}} {tol_header:>{tol_w}} {'total n':>{total_n_w}} {'new n':>{new_n_w}}"
        f" {'iters':>{iters_w}} {'solution':>{sol_w}} {'half-width':>{hw_w}} {'time (s)':>{time_w}}",
        sep,
    ]
    for name, abs_tol, total_n, new_n, iters, solution, hw, tsec in rows:
        iter_str = str(int(iters)) if iters is not None else "-"
        sol_str = f"{float(solution):.{solution_decimals}f}"
        abs_tol_str = f"{float(abs_tol):.0e}" if abs_tol is not None else "---"
        lines.append(
            f"{name:<{stage_w}} {abs_tol_str:>{tol_w}} {int(total_n):>{total_n_w},} {int(new_n):>{new_n_w},}"
            f" {iter_str:>{iters_w}} {sol_str:>{sol_w}} {hw:>{hw_w}.2e} {tsec:>{time_w}.4f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def print_stage_summary(
    rows=None,
    *,
    title="Stage summary",
    tol_header=None,
    resume_solver=None,
    loose_data=None,
    resume_data=None,
    fresh_solver=None,
    fresh_data=None,
):
    """Print a compact stage-by-stage sample and time summary table."""
    if rows is None:
        rows = stage_summary_rows_from_histories(
            resume_solver,
            loose_data=loose_data,
            resume_data=resume_data,
            fresh_solver=fresh_solver,
            fresh_data=fresh_data,
        )
        if tol_header is None:
            tol_names = {row[-1] for row in rows} - {None}
            if tol_names == {"target_rmse_tol"}:
                tol_header = "rmse_tol"
            elif len(tol_names) == 1:
                tol_header = tol_names.pop()
            else:
                tol_header = "abs_tol"
        rows = [row[:-1] for row in rows]
    elif tol_header is None:
        tol_header = "abs_tol"
    print(format_stage_summary(rows, title=title, tol_header=tol_header))


def _is_multilevel_case(name, resume_stage, fresh_stage):
    """Return True for ML/MLQMC-style cases where iteration parity is not required."""
    if str(name).startswith("CubML"):
        return True
    for stage in (resume_stage, fresh_stage):
        tol_name = stage.get("tol_name") if isinstance(stage, dict) else None
        if tol_name in {"rmse_tol", "target_rmse_tol"}:
            return True
    return False


def collect_resume_fresh_warnings(name, resume_stage, fresh_stage):
    """Return warning strings for mismatched resume-vs-fresh outcomes."""
    warning_lines = []

    resume_n = int(resume_stage.get("total_n", 0))
    fresh_n = int(fresh_stage.get("total_n", 0))
    if resume_n != fresh_n:
        warning_lines.append(
            f"WARNING: {name}: Inconsistent total samples across stages "
            f"(resume_n={resume_n}, fresh_n={fresh_n})"
        )

    resume_iters = int(resume_stage.get("iters") or 0)
    fresh_iters = int(fresh_stage.get("iters") or 0)
    if resume_iters != fresh_iters and not _is_multilevel_case(name, resume_stage, fresh_stage):
        warning_lines.append(
            f"WARNING: {name}: Inconsistent iteration counts across stages "
            f"(resume_iters={resume_iters}, fresh_iters={fresh_iters})"
        )

    try:
        resume_sol = float(resume_stage.get("solution", float("nan")))
        fresh_sol = float(fresh_stage.get("solution", float("nan")))
        resume_tol = float(resume_stage.get("tol", float("nan")))
    except (TypeError, ValueError):
        return warning_lines
    if math.isfinite(resume_sol) and math.isfinite(fresh_sol) and math.isfinite(resume_tol):
        if abs(resume_sol - fresh_sol) > 2 * resume_tol:
            warning_lines.append(
                f"WARNING: {name}: Resume and fresh solutions differ by more than 2 * tol "
                f"(resume_sol={resume_sol}, fresh_sol={fresh_sol}, tol={resume_tol})"
            )
    return warning_lines


def enable_diagnostics(stopping_criterion, label, verbose=False, print_live=False):
    """Enable iteration diagnostics on a stopping criterion instance."""
    setattr(stopping_criterion, "trace_iterations", True)
    setattr(stopping_criterion, "trace_label", label)
    setattr(stopping_criterion, "verbose", verbose)
    setattr(stopping_criterion, "trace_print", print_live)
    return stopping_criterion

#################################################################
#  Case factories and runners
#################################################################
def make_case(name, loose_factory, tight_factory):
    """Create a standard loose/tight demo case record."""
    return {"name": name, "loose": loose_factory, "tight": tight_factory}


def make_tol_case(name, builder, loose_tol, tight_tol):
    """Create a case from a scalar-tolerance builder."""
    return make_case(name, lambda: builder(loose_tol), lambda: builder(tight_tol))


def make_named_tol_builder(sc_class, integrand_factory, tol_name, **fixed_kwargs):
    """Build a factory for stopping criteria with a named tolerance argument."""
    def build(tol):
        sc = sc_class(integrand_factory(), **{tol_name: tol, **fixed_kwargs})
        setattr(sc, "_resume_tol_name", tol_name)
        try:
            setattr(sc, "_resume_tol_value", float(tol))
        except (TypeError, ValueError):
            setattr(sc, "_resume_tol_value", None)
        return sc

    return build


def _safe_half_width(data):
    """Return the best available half-width or RMSE estimate; otherwise NaN."""
    try:
        return (data.comb_bound_high.item() - data.comb_bound_low.item()) / 2
    except Exception:
        pass
    try:
        return float(data.bias_estimate)
    except Exception:
        pass
    try:
        import numpy as np
        n = data.n_level
        v = data.var_level
        # Align lengths: var_level may be shorter than n_level by one (last level index)
        min_len = min(len(n), len(v))
        n, v = n[:min_len], v[:min_len]
        valid = n > 0
        return float(np.sqrt(np.sum(v[valid] / n[valid])))
    except Exception:
        return float("nan")


def _get_sc_tol(sc):
    """Return the primary positive tolerance value and its attribute name."""
    preferred_attr = getattr(sc, "_resume_tol_name", None)
    preferred_input_value = getattr(sc, "_resume_tol_value", None)

    def get_tol_value(attr):
        if attr == "rmse_tol":
            for candidate in ("target_rmse_tol", "rmse_tol"):
                try:
                    val = float(getattr(sc, candidate, None))
                except (TypeError, ValueError):
                    continue
                if val > 0:
                    return val
            return None
        try:
            val = float(getattr(sc, attr, None))
        except (TypeError, ValueError):
            return None
        return val if val > 0 else None

    if preferred_attr in {"abs_tol", "rel_tol", "rmse_tol"}:
        try:
            preferred_input_value = float(preferred_input_value)
        except (TypeError, ValueError):
            preferred_input_value = None
        if preferred_input_value is not None and preferred_input_value > 0:
            return preferred_input_value, preferred_attr
        preferred_val = get_tol_value(preferred_attr)
        if preferred_val is not None:
            return preferred_val, preferred_attr

    for attr in ("rel_tol", "abs_tol", "rmse_tol"):
        val = get_tol_value(attr)
        if val is not None:
            return val, attr
    return None, None


def _solution_value(solution):
    """Return a scalar float from a QMCPy solution object."""
    try:
        return float(solution.item())
    except Exception:
        return float(solution)


def _extract_final_iteration_number(iteration_log):
    """Extract the final iteration number from an iteration log."""
    if not iteration_log:
        return None
    lines = iteration_log.splitlines()
    # Search from the end for the last ITER or RESUME row
    for line in reversed(lines):
        stripped = line.lstrip()
        if stripped.startswith("RESUME") or stripped.startswith("ITER"):
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except (ValueError, IndexError):
                    pass
    return None


def _trim_resume_iteration_log(iteration_log, loose_stage=None):
    """Keep only rows from the first RESUME marker onward, with adjusted iteration numbers.

    The formatted log always begins with three lines:
    1) section header
    2) column header
    3) separator
    For resumed runs, we keep these lines and drop pre-resume ITER rows.
    For ML solvers, we renumber RESUME/ITER rows to continue from loose stage's final iteration.
    """
    import re
    if not iteration_log:
        return iteration_log
    lines = iteration_log.splitlines()
    if len(lines) <= 3:
        return iteration_log

    resume_idx = None
    for idx in range(3, len(lines)):
        if lines[idx].lstrip().startswith("RESUME"):
            resume_idx = idx
            break
    if resume_idx is None:
        return iteration_log

    # Determine if we should renumber (for ML solvers with loose stage reference)
    loose_final_iter = None
    if loose_stage and isinstance(loose_stage, dict):
        loose_final_iter = loose_stage.get("iters")
        try:
            loose_final_iter = int(loose_final_iter) if loose_final_iter is not None else None
        except (TypeError, ValueError):
            loose_final_iter = None

    trimmed_lines = lines[:3] + lines[resume_idx:]

    # If we have loose_final_iter, renumber the resumed iterations for continuity
    if loose_final_iter is not None:
        result_lines = trimmed_lines[:3]  # Keep header and separator
        iter_offset = None
        solution_col = None

        for line in trimmed_lines[3:]:
            stripped = line.lstrip()
            if stripped.startswith("RESUME") or stripped.startswith("ITER"):
                # Use regex to extract stage label and iteration number
                match = re.match(r'^(\s*)(RESUME|ITER)(\s+)(\d+)(\s+)(.*)$', line)
                if match:
                    indent, stage_label, spacing_before_iter, old_iter_str, spacing_after_iter, tail = match.groups()
                    old_iter = int(old_iter_str)

                    # Lock the start column of the solution field once per block.
                    if solution_col is None:
                        solution_col = match.start(6)

                    # Calculate offset from RESUME row
                    if iter_offset is None and stage_label == "RESUME":
                        iter_offset = loose_final_iter - old_iter

                    # For first ITER row after RESUME, ensure continuation
                    if iter_offset is None and stage_label == "ITER":
                        iter_offset = loose_final_iter + 1 - old_iter

                    if iter_offset is not None:
                        new_iter = old_iter + iter_offset
                        # Keep iter right-aligned while fixing the solution column position.
                        stage_prefix = indent + stage_label
                        iter_text = str(new_iter)
                        spaces_before_iter = max(1, solution_col - len(stage_prefix) - len(iter_text) - 1)
                        new_line = stage_prefix + (" " * spaces_before_iter) + iter_text + " " + tail
                        result_lines.append(new_line)
                    else:
                        result_lines.append(line)
                else:
                    result_lines.append(line)
            else:
                result_lines.append(line)
        return "\n".join(result_lines)

    return "\n".join(trimmed_lines)


def _make_stage_record(name, sc, solution, data, inputs, previous_n_total=0):
    """Build a compact stage record from a completed integration."""
    total_n = int(getattr(data, "n_total", 0))
    tol, tol_name = _get_sc_tol(sc)
    iteration_log = sc.format_iteration_log(
        history=getattr(data, "iteration_history", None)
    )
    return {
        "name": name,
        "tol": tol,
        "tol_name": tol_name,
        "total_n": total_n,
        "new_n": total_n - int(previous_n_total),
        "iters": getattr(data, "_iter_count", None),
        "solution": _solution_value(solution),
        "half_width": _safe_half_width(data),
        "time": float(getattr(data, "time_integrate", 0.0)),
        "inputs": inputs,
        "iteration_log": iteration_log,
    }


def _run_logged_case(factory, label, verbose=False, **integrate_kwargs):
    """Build, trace, integrate, and return the run outputs plus compact inputs."""
    sc = enable_diagnostics(factory(), label, verbose=verbose)
    solution, data = sc.integrate(**integrate_kwargs)
    return solution, data, _format_problem_inputs(sc), sc


def run_resume_case(case, verbose=False):
    """Run a loose solve followed by a resumed tighter solve."""
    name = case["name"]
    row = {"name": name}
    try:
        sol1, data1, loose_inputs, sc1 = _run_logged_case(
            case["loose"], f"{name}-LOOSE", verbose=verbose)
        loose_stage = _make_stage_record("Loose", sc1, sol1, data1, loose_inputs)
        old_n = loose_stage["total_n"]

        # Start RESUMED stage with the same solver, retuned to the tight tolerance.
        tight_sc = case["tight"]()
        if hasattr(tight_sc, "_resume_tol_name"):
            sc1._resume_tol_name = getattr(tight_sc, "_resume_tol_name")
        if hasattr(tight_sc, "_resume_tol_value"):
            sc1._resume_tol_value = getattr(tight_sc, "_resume_tol_value")
        if hasattr(tight_sc, "target_rmse_tol"):
            sc1.target_rmse_tol = float(tight_sc.target_rmse_tol)
        sc1.set_tolerance(
            abs_tol=getattr(tight_sc, "abs_tol", None),
            rel_tol=getattr(tight_sc, "rel_tol", None),
            rmse_tol=getattr(tight_sc, "target_rmse_tol",
                             getattr(tight_sc, "rmse_tol", None) if not hasattr(tight_sc, "abs_tol") else None,),
        )
        setattr(sc1, "trace_label", f"{name}-RESUME")
        setattr(sc1, "verbose", verbose)
        sol2, data2 = sc1.integrate(resume=data1)
        resume_inputs = _format_problem_inputs(sc1)
        resume_stage = _make_stage_record(
            "Resumed", sc1, sol2, data2, resume_inputs, previous_n_total=old_n
        )
        # Renumber resume iteration log to continue from loose stage for ML solvers
        resume_stage["iteration_log"] = _trim_resume_iteration_log(
            resume_stage["iteration_log"], loose_stage=loose_stage
        )
        # Update iteration count to match the final iteration number in the log
        final_iter = _extract_final_iteration_number(resume_stage["iteration_log"])
        if final_iter is not None:
            resume_stage["iters"] = final_iter
        row.update(
            {
                "status": "ok",
                "loose": loose_stage,
                "resume": resume_stage,
            }
        )
    except Exception as exc:
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return row


def run_fresh_case(case, verbose=False):
    """Run a fresh tight solve from scratch."""
    name = case["name"]
    row = {"name": name}
    try:
        sol, data, fresh_inputs, sc = _run_logged_case(
            case["tight"], f"{name}-FRESH", verbose=verbose
        )
        fresh_stage = _make_stage_record("Fresh", sc, sol, data, fresh_inputs)
        # Update iteration count to match the final iteration number in the log
        final_iter = _extract_final_iteration_number(fresh_stage["iteration_log"])
        if final_iter is not None:
            fresh_stage["iters"] = final_iter
        row.update(
            {
                "status": "ok",
                "fresh": fresh_stage,
            }
        )
    except Exception as exc:
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return row


#################################################################
#  Report writers
#################################################################
def write_combined_report(path, title, resume_rows, fresh_rows):
    """Write combined loose/resume/fresh reports with stage summaries."""
    separator = "~" * 60
    lines = [title, f"generated: {strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for i, (rrow, frow) in enumerate(zip(resume_rows, fresh_rows)):
        if i > 0:
            lines.append(separator)
            lines.append("")
        name = rrow.get("name", "unknown")
        rstatus = rrow.get("status", "unknown")
        fstatus = frow.get("status", "unknown")
        lines.append(f"[{name}] status={rstatus}")
        loose_stage = rrow.get("loose", {})
        resume_stage = rrow.get("resume", {})
        fresh_stage = frow.get("fresh", {})

        # Input sections
        for section_label, stage in [
            ("loose_inputs", loose_stage),
            ("resume_inputs", resume_stage),
            ("fresh_inputs", fresh_stage),
        ]:
            text = stage.get("inputs", "")
            if not text:
                continue
            lines.append("")
            lines.append(f"  {section_label}:")
            lines.extend([f"    {line}" for line in text.splitlines()])

        # Iteration log sections
        for section_label, stage in [
            ("loose_iteration_log", loose_stage),
            ("resume_iteration_log", resume_stage),
            ("fresh_iteration_log", fresh_stage),
        ]:
            text = stage.get("iteration_log", "")
            if not text:
                continue
            lines.append("")
            lines.append(f"  {section_label}:")
            lines.extend([f"    {line}" for line in text.splitlines()])

        # Stage summary table
        if rstatus == "ok" and fstatus == "ok":
            stage_rows = stage_summary_rows_from_stage_records(
                loose_stage, resume_stage, fresh_stage
            )
            tol_header = stage_summary_tol_header(
                loose_stage, resume_stage, fresh_stage
            )
            summary_text = format_stage_summary(
                stage_rows, title=f"Stage summary of {name}", tol_header=tol_header
            ).strip()
            lines.append("")
            lines.extend([f"  {line}" for line in summary_text.splitlines()])

        for warning_line in collect_resume_fresh_warnings(
            name, resume_stage, fresh_stage
        ):
            print(f"  {warning_line}")
            lines.append(f"  {warning_line}")

        # Errors
        for row_obj in (rrow, frow):
            if "error" in row_obj:
                lines.append(f"  error: {row_obj['error']}")

        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
