import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple

if __package__:  # Imported as demos.GBM.gbm_code.data_util (e.g. by pytest)
    from . import config as cf
    from . import qmcpy_util as qpu
    from . import quantlib_util as qlu
else:  # Compatibility symlink imported with demos/GBM on sys.path.
    import config as cf
    import qmcpy_util as qpu
    import quantlib_util as qlu

# Speedups are only reported between runs that share a sampler, so the column
# name states that explicitly; see create_timing_dataframe().
SPEEDUP_COLUMN = "Speedup (same sampler)"


def add_theoretical_results(
    results_data: list, theoretical_mean: float, theoretical_std: float
) -> None:
    """
    Add theoretical benchmark values to results data.

    Args:
        results_data: List to append theoretical results to
        theoretical_mean: Theoretical expected value $E[S_T]$
        theoretical_std: Theoretical standard deviation of $S_T$
    """
    results_data.append(
        {
            "Method": "Theoretical",
            "Sampler": "-",
            "Mean": theoretical_mean,
            "Std Dev": theoretical_std,
            "Mean Absolute Error": 0,
            "Std Dev Error": 0,
            "Mean SE": np.nan,
            "Std Dev SE": np.nan,
            "MAE SE": np.nan,
            "Std Dev Error SE": np.nan,
        }
    )


def _replication_standard_error(values: npt.NDArray[np.floating]) -> float:
    """Return the standard error across independent randomized replications."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return np.nan
    return values.std(ddof=1) / np.sqrt(values.size)


def _replication_summary(
    method: str,
    sampler_type: str,
    means: npt.NDArray[np.floating],
    stds: npt.NDArray[np.floating],
    theoretical_mean: float,
    theoretical_std: float,
) -> dict:
    """Summarize per-replication estimates against the theoretical moments."""
    mean_errors = np.abs(means - theoretical_mean)
    std_errors = np.abs(stds - theoretical_std)
    return {
        "Method": method,
        "Sampler": sampler_type,
        "Mean": means.mean(),
        "Std Dev": stds.mean(),
        "Mean Absolute Error": mean_errors.mean(),
        "Std Dev Error": std_errors.mean(),
        "Mean SE": _replication_standard_error(means),
        "Std Dev SE": _replication_standard_error(stds),
        "MAE SE": _replication_standard_error(mean_errors),
        "Std Dev Error SE": _replication_standard_error(std_errors),
    }


def add_quantlib_results(
    results_data: list,
    sampler_type: str,
    quantlib_final: npt.NDArray[np.floating],  # per replication mean
    theoretical_mean: float,
    theoretical_std: float,
    ql_stds: npt.NDArray[np.floating],  # per replication std dev
) -> None:
    """
    Add summary statistics for QuantLib simulations based on per-replication means.

    The `quantlib_final` array is expected to contain one value per replication,
    where each value is the mean of all simulated terminal asset prices S_T in
    that replication.

    Args:
        results_data: List to which the QuantLib summary row is appended.
        sampler_type: Identifier for the sampler used in the QuantLib experiment.
        quantlib_final: 1D array of per-replication sample means of $S_T$.
        theoretical_mean: Theoretical expected value $E[S_T]$ used as a benchmark.
        theoretical_std: Theoretical standard deviation of $S_T$ used as a benchmark.
        ql_stds: 1D array of per-replication standard deviations of $S_T$.
    """
    results_data.append(
        _replication_summary(
            "QuantLib",
            sampler_type,
            quantlib_final,
            ql_stds,
            theoretical_mean,
            theoretical_std,
        )
    )


def add_qmcpy_results(
    results_data: list,
    sampler_type: str,
    qmcpy_final: npt.NDArray[np.floating],  # per replication mean
    qp_emp_mean: float,
    theoretical_mean: float,
    theoretical_std: float,
    qp_stds: npt.NDArray[np.floating],  # per replication std dev
) -> None:
    """
    Add empirical QMCPy results, computed from per-replication means, to results data.

    The mean absolute error (MAE) is computed across replications as
    the average of the absolute differences between each per-replication
    mean in `qmcpy_final` and `theoretical_mean`.

    Args:
        results_data: List to which the QMCPy summary for this sampler is appended.
        sampler_type: Name or type of the QMCPy sampler used.
        qmcpy_final: 1D array where each entry is the mean payoff from a single
            replication of the QMCPy experiment.
        qp_emp_mean: Overall empirical mean across all replications.
        theoretical_mean: Theoretical expected value used as a benchmark.
        theoretical_std: Theoretical standard deviation used as a benchmark.
        qp_stds: 1D array of per-replication standard deviations of $S_T$.
    """
    summary = _replication_summary(
        "QMCPy",
        sampler_type,
        qmcpy_final,
        qp_stds,
        theoretical_mean,
        theoretical_std,
    )
    summary["Mean"] = qp_emp_mean
    results_data.append(summary)


def process_sampler_data(
    sampler_type: str,
    results_data: list,
    theoretical_mean: float,
    theoretical_std: float,
    params_ql: dict,
    params_qp: dict,
) -> tuple:
    """
    Process and compare data for a single sampler type across both libraries.

    Args:
        sampler_type: Type of sampler to test
        results_data: List to append comparison results to
        theoretical_mean: Theoretical expected value
        theoretical_std: Theoretical standard deviation
        params_ql: Dictionary of QuantLib parameters
        params_qp: Dictionary of QMCPy parameters

    Returns:
        tuple: (quantlib_paths, qmcpy_paths, ql_gbm, qp_gbm, params_ql, params_qp)
    """

    params_ql["sampler_type"] = sampler_type
    params_qp["sampler_type"] = sampler_type

    replications = params_qp["replications"]

    quantlib_paths, ql_gbm = None, None

    if sampler_type in cf.get_sampler_configurations()["quantlib_samplers"]:
        ql_means = np.empty(replications)
        ql_stds = np.empty(replications)
        ql_paths = []
        ql_seed = params_ql["seed"]

        for r in range(replications):
            params_ql["seed"] = ql_seed + r
            paths, ql_gbm = qlu.generate_quantlib_paths(**params_ql)
            ql_paths.append(paths)
            ql_means[r] = paths[:, -1].mean()
            ql_stds[r] = paths[:, -1].std(ddof=0)

        params_ql["seed"] = ql_seed
        quantlib_paths = np.stack(ql_paths)

    else:
        ql_means = None

    qmcpy_paths, qp_gbm = qpu.generate_qmcpy_paths(**params_qp)

    if qmcpy_paths.ndim == 3:
        qp_means = qmcpy_paths[:, :, -1].mean(axis=1)
        qp_stds = qmcpy_paths[:, :, -1].std(axis=1, ddof=0)
    else:
        qp_means = np.array([qmcpy_paths[:, -1].mean()])
        qp_stds = np.array([qmcpy_paths[:, -1].std(ddof=0)])

    if ql_means is not None:
        add_quantlib_results(
            results_data,
            sampler_type,
            ql_means,
            theoretical_mean,
            theoretical_std,
            ql_stds,
        )

    add_qmcpy_results(
        results_data,
        sampler_type,
        qp_means,
        qp_means.mean(),
        theoretical_mean,
        theoretical_std,
        qp_stds,
    )

    return quantlib_paths, qmcpy_paths, ql_gbm, qp_gbm, params_ql, params_qp


def create_timing_dataframe(
    quantlib_results: dict, qmcpy_results: dict
) -> pd.DataFrame:
    """
    Create comprehensive timing comparison table from benchmark results.

    The speedup of a QMCPy row is computed against the QuantLib run that uses
    the *same* sampler family. The comparison is not fully like-for-like: the two libraries may use different randomizations, path constructions, and implementation strategies. Samplers that QuantLib is not benchmarked with here (e.g. Lattice) have no counterpart and report "-" rather than a ratio against an unrelated QuantLib run.

    Args:
        quantlib_results: Dictionary mapping sampler names to timing results
        qmcpy_results: Dictionary mapping sampler names to timing results

    Returns:
        DataFrame with timing statistics and same-sampler speedup comparisons
    """
    timing_data = []

    # Add QuantLib data
    for sampler_type, result in quantlib_results.items():
        timing_data.append(
            {
                "Method": "QuantLib",
                "Sampler": sampler_type,
                "Mean Time (s)": result["average"],
                "Std Dev (s)": result["stdev"],
                SPEEDUP_COLUMN: "-",
            }
        )

    # Add QMCPy data with same-sampler speedup calculation
    for sampler_type, result in qmcpy_results.items():
        quantlib_result = quantlib_results.get(sampler_type)
        speedup = (
            quantlib_result["average"] / result["average"]
            if quantlib_result is not None
            else "-"
        )
        timing_data.append(
            {
                "Method": "QMCPy",
                "Sampler": sampler_type,
                "Mean Time (s)": result["average"],
                "Std Dev (s)": result["stdev"],
                SPEEDUP_COLUMN: speedup,
            }
        )

    return pd.DataFrame(timing_data)


def run_construction_ablation(
    sampler_types: list,
    decomp_types: list,
    theoretical_mean: float,
    theoretical_std: float,
    n_paths: int,
    n_steps: int,
    replications: int,
    seed: int = cf.QMCPY_SEED,
) -> pd.DataFrame:
    """
    Vary only the path construction, holding the point set fixed.

    The QMCPy-versus-QuantLib comparison changes the point set, its
    randomization, and the path construction at once, so it cannot attribute a
    difference in accuracy to any one of them. Here the sampler, seed, path
    count, and time grid are all held fixed and only `decomp_type` changes, so
    any difference in error is attributable to the construction alone.

    Include an IID sampler as a control: path construction reorders how
    variance is distributed across coordinates, which only helps when the
    coordinates are equidistributed, so IID accuracy should be roughly flat
    across constructions while the low-discrepancy samplers are not.

    Args:
        sampler_types: Samplers to test, e.g. ['IIDStdUniform', 'Sobol']
        decomp_types: Constructions to test, from 'PCA', 'Cholesky',
            'BrownianBridge'
        theoretical_mean: Theoretical expected value of S_T
        theoretical_std: Theoretical standard deviation of S_T
        n_paths: Paths per replication
        n_steps: Number of monitoring times, which is the sampling dimension
        replications: Independent randomizations averaged over
        seed: Seed shared by every run, so the point set is identical

    Returns:
        DataFrame with one row per (sampler, construction) and columns
        'Sampler', 'Construction', 'Mean Absolute Error', 'Std Dev Error'
    """
    gbm_params = cf.get_gbm_parameters()
    rows = []
    for sampler_type in sampler_types:
        for decomp_type in decomp_types:
            paths, _ = qpu.generate_qmcpy_paths(
                initial_value=gbm_params["initial_value"],
                mu=gbm_params["mu"],
                diffusion=gbm_params["sigma"] ** 2,
                maturity=gbm_params["maturity"],
                n_steps=n_steps,
                n_paths=n_paths,
                sampler_type=sampler_type,
                replications=replications,
                seed=seed,
                decomp_type=decomp_type,
            )
            terminal = paths[..., -1]
            summary = _replication_summary(
                "QMCPy",
                sampler_type,
                terminal.mean(axis=-1),
                terminal.std(axis=-1, ddof=0),
                theoretical_mean,
                theoretical_std,
            )
            rows.append(
                {
                    "Sampler": sampler_type,
                    "Construction": decomp_type,
                    "Mean Absolute Error": summary["Mean Absolute Error"],
                    "Std Dev Error": summary["Std Dev Error"],
                }
            )
    return pd.DataFrame(rows)


def extract_covariance_samples(
    paths: npt.NDArray[np.floating],
    time_grid: npt.NDArray[np.floating],
    target_times: Tuple[float, float],
) -> tuple:
    """Estimate covariance at the grid points nearest two requested times.

    Args:
        paths: Paths of shape ``(n_paths, n_times)`` or
            ``(replications, n_paths, n_times)``.
        time_grid: Actual time attached to each path coordinate.
        target_times: Two times at which to compare covariance.

    Returns:
        ``(average_covariance, replication_covariances)``. The second value is
        ``None`` for a single two-dimensional path array.

    Raises:
        ValueError: If the path and time-grid shapes are incompatible.

    Note:
        ``ddof=0`` treats each randomized point set as a quadrature rule for a
        population moment. The usual ``n - 1`` IID correction is not justified
        for dependent points from a randomized low-discrepancy sequence.
    """
    paths = np.asarray(paths)
    time_grid = np.asarray(time_grid)
    if paths.ndim not in (2, 3):
        raise ValueError("paths must have shape (n, t) or (r, n, t)")
    if time_grid.ndim != 1 or paths.shape[-1] != time_grid.size:
        raise ValueError("time_grid length must match the final paths axis")

    indices = [int(np.argmin(np.abs(time_grid - t))) for t in target_times]
    selected = paths[..., indices]
    if paths.ndim == 2:
        return np.cov(selected, rowvar=False, ddof=0), None

    covariances = np.asarray(
        [np.cov(replication, rowvar=False, ddof=0) for replication in selected]
    )
    return covariances.mean(axis=0), covariances


def extract_comparison_data(results_df: pd.DataFrame) -> tuple:
    """
    Extract data for comparison plotting from results dataframe.

    Args:
        results_df: DataFrame containing results from both libraries

    Returns:
        tuple: (samplers, qmcpy_errors, qmcpy_times, quantlib_errors,
                quantlib_times, theoretical_mean, qmcpy_sd_errors,
                quantlib_sd_errors)

    Note:
        The '*_sd_errors' entries hold the error in the estimated standard
        deviation of S_T. It is a separate accuracy metric, not an
        uncertainty attached to the Mean Absolute Error, so it belongs on its
        own axes rather than as error bars on the MAE.
    """
    qmcpy_data = results_df[results_df["Method"] == "QMCPy"].copy()
    quantlib_data = results_df[results_df["Method"] == "QuantLib"].copy()
    theoretical_data = results_df[results_df["Method"] == "Theoretical"].copy()

    sampler_order = cf.get_sampler_configurations()["all_samplers"]
    sampler_rank = {sampler: rank for rank, sampler in enumerate(sampler_order)}
    qmcpy_data["_sampler_rank"] = qmcpy_data["Sampler"].map(sampler_rank)
    qmcpy_data = qmcpy_data.sort_values("_sampler_rank")

    samplers = qmcpy_data["Sampler"].values
    qmcpy_errors = qmcpy_data["Mean Absolute Error"].values
    qmcpy_times = (
        qmcpy_data["Mean Time (s)"].values
        if "Mean Time (s)" in qmcpy_data.columns
        else None
    )

    qmcpy_sd_errors = qmcpy_data["Std Dev Error"].values

    # Get QuantLib data (only available for some samplers)
    ql_error_dict = dict(
        zip(quantlib_data["Sampler"], quantlib_data["Mean Absolute Error"])
    )
    quantlib_errors = [ql_error_dict.get(s) for s in samplers]

    ql_sd_error_dict = dict(
        zip(quantlib_data["Sampler"], quantlib_data["Std Dev Error"])
    )
    quantlib_sd_errors = [ql_sd_error_dict.get(s) for s in samplers]

    if "Mean Time (s)" in quantlib_data.columns:
        ql_time_dict = dict(
            zip(quantlib_data["Sampler"], quantlib_data["Mean Time (s)"])
        )
        quantlib_times = [ql_time_dict.get(s) for s in samplers]
    else:
        quantlib_times = [None] * len(samplers)

    # Handle case where theoretical data might be missing
    if not theoretical_data.empty:
        theoretical_mean = theoretical_data["Mean"].iloc[0]
    else:
        # Calculate theoretical mean from parameters if not in results_df
        # Using the parameters from the comparison study
        S0, mu, T = 100, 0.05, 1.0
        theoretical_mean = S0 * np.exp(mu * T)

    return (
        samplers,
        qmcpy_errors,
        qmcpy_times,
        quantlib_errors,
        quantlib_times,
        theoretical_mean,
        qmcpy_sd_errors,
        quantlib_sd_errors,
    )


def add_theoretical_row(
    results: list,
    series_name: str,
    n_steps: int,
    n_paths: int,
    theoretical_mean: float,
    theoretical_std: float,
) -> None:
    """Add theoretical benchmark row to results"""
    results.append(
        {
            "Series": series_name,
            "n_steps": n_steps,
            "n_paths": n_paths,
            "Method": "Theoretical",
            "Sampler": "-",
            "Mean": theoretical_mean,
            "Std Dev": theoretical_std,
            "Mean Absolute Error": 0,
            "Std Dev Error": 0,
            "Mean SE": np.nan,
            "Std Dev SE": np.nan,
            "MAE SE": np.nan,
            "Std Dev Error SE": np.nan,
            "Runtime (s)": 0,
            "Runtime Std (s)": 0,
        }
    )


def collect_library_results(
    sampler: str,
    series_name: str,
    n_steps: int,
    n_paths: int,
    ql_timing: dict,
    qp_timing: dict,
    theoretical_mean: float,
    theoretical_std: float,
    replications: int = 1,
) -> list:
    """Collect timing and replication-averaged accuracy for one sampler."""
    results = []
    gbm_params = cf.get_gbm_parameters()

    # QuantLib parameters
    ql_params = {**gbm_params, "n_steps": n_steps, "n_paths": n_paths}

    # QMCPy parameters (note: diffusion = sigma^2)
    qp_params = {
        "initial_value": gbm_params["initial_value"],
        "mu": gbm_params["mu"],
        "diffusion": gbm_params["sigma"] ** 2,  # Convert sigma to diffusion
        "maturity": gbm_params["maturity"],
        "n_steps": n_steps,
        "n_paths": n_paths,
    }

    # QuantLib results (if supported)
    if sampler in cf.get_sampler_configurations()["quantlib_samplers"]:
        try:
            ql_means, ql_stds = np.empty(replications), np.empty(replications)
            for r in range(replications):
                paths, _ = qlu.generate_quantlib_paths(
                    sampler_type=sampler, seed=cf.QUANTLIB_SEED + r, **ql_params
                )
                terminal = paths[:, -1]
                ql_means[r], ql_stds[r] = terminal.mean(), terminal.std(ddof=0)
            results.append({
                "Series": series_name,
                "n_steps": n_steps,
                "n_paths": n_paths,
                **_replication_summary(
                    "QuantLib", sampler, ql_means, ql_stds,
                    theoretical_mean, theoretical_std,
                ),
                "Runtime (s)": ql_timing[sampler]["average"],
                "Runtime Std (s)": ql_timing[sampler]["stdev"],
            })
        except Exception as e:
            print(f"      QuantLib {sampler} failed: {e}")

    # QMCPy results
    try:
        paths, _ = qpu.generate_qmcpy_paths(
            sampler_type=sampler,
            replications=replications,
            seed=cf.QMCPY_SEED,
            **qp_params,
        )
        terminal = np.atleast_2d(paths[..., -1])
        qp_means = terminal.mean(axis=1)
        qp_stds = terminal.std(axis=1, ddof=0)
        results.append({
            "Series": series_name,
            "n_steps": n_steps,
            "n_paths": n_paths,
            **_replication_summary(
                "QMCPy", sampler, qp_means, qp_stds,
                theoretical_mean, theoretical_std,
            ),
            "Runtime (s)": qp_timing[sampler]["average"],
            "Runtime Std (s)": qp_timing[sampler]["stdev"],
        })
    except Exception as e:
        print(f"      QMCPy {sampler} failed: {e}")

    return results
