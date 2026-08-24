import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FixedFormatter
from typing import Optional
import os
import scipy.stats as sc
import qmcpy as qp


def plot_error_comparison(
    ax: Axes,
    samplers: list,
    qmcpy_errors: npt.NDArray[np.floating],
    quantlib_errors: list,
    replications: Optional[int] = None,
) -> None:
    """
    Plot error comparison subplot.

    Args:
        ax: Matplotlib axis object
        samplers: List of sampler names
        qmcpy_errors: Array of QMCPy mean absolute errors
        quantlib_errors: List of QuantLib mean absolute errors (may contain None)
        replications: Number of replications used for averaging (optional, for title)
    """
    x = np.arange(len(samplers))
    width = 0.35
    # Plot QuantLib data first (left side)
    ql_x, ql_errors = [], []
    for i, error in enumerate(quantlib_errors):
        if error is not None:
            ql_x.append(i)
            ql_errors.append(error)
    if ql_errors:
        ax.bar(
            [x - width / 2 for x in ql_x],
            ql_errors,
            width,
            label="QuantLib",
            color="blue",
            alpha=0.8,
        )
    # Plot QMCPy data second (right side)
    ax.bar(x + width / 2, qmcpy_errors, width, label="QMCPy", color="red", alpha=0.8)

    ax.set_xlabel("Sampler Type")
    ax.set_ylabel("Mean Absolute Error (log scale)")

    # Add replications info to title if provided
    if replications is not None:
        ax.set_title(
            f"Mean Absolute Error Comparison\n(averaged over {replications} replications)",
            fontsize=16,
            fontweight="bold",
        )
    else:
        ax.set_title("Mean Absolute Error Comparison", fontsize=16, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(samplers, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_performance_comparison(
    ax: Axes,
    samplers: list,
    qmcpy_times: Optional[npt.NDArray[np.floating]],
    quantlib_times: list,
) -> None:
    """Plot performance comparison subplot"""
    x = np.arange(len(samplers))
    width = 0.35
    if qmcpy_times is not None:
        # Plot QuantLib timing data first (left side)
        ql_x, ql_times = [], []
        for i, time in enumerate(quantlib_times):
            if time is not None:
                ql_x.append(i)
                ql_times.append(time)
        if ql_times:
            ax.bar(
                [x - width / 2 for x in ql_x],
                ql_times,
                width,
                label="QuantLib",
                color="blue",
                alpha=0.8,
            )
        # Plot QMCPy data second (right side)
        ax.bar(x + width / 2, qmcpy_times, width, label="QMCPy", color="red", alpha=0.8)
        # Add speedup annotations where QuantLib data is available, at center of QMCPy bars
        if len(ql_times) > 0:
            for i, (qmc_time, ql_time) in enumerate(zip(qmcpy_times, quantlib_times)):
                if ql_time is not None:
                    speedup = ql_time / qmc_time
                    annotation_height = qmc_time + max(qmcpy_times) * 0.3
                    # Position arrow at center of QMCPy bar (i + width/2)
                    ax.annotate(
                        f"{speedup:.1f}x faster",
                        xy=(i + width / 2, qmc_time),
                        xytext=(i + width / 2, annotation_height),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="blue", lw=1),
                    )
        ax.set_xlabel("Sampler Type")
        ax.set_ylabel("Execution Time (s)")
        ax.set_xticks(x)
        ax.set_xticklabels(samplers, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "Timing data not available\nRun previous cells to generate data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

    ax.set_title("Performance Comparison", fontsize=16, fontweight="bold")


def get_plot_styling() -> dict:
    """Define colors and markers for plotting"""
    return {
        "colors": {
            "QuantLib": {"IIDStdUniform": "#1f77b4", "Sobol": "#ff7f0e"},
            "QMCPy": {
                "IIDStdUniform": "#2ca02c",
                "Sobol": "#d62728",
                "Lattice": "#9467bd",
                "Halton": "#8c564b",
            },
        },
        "markers": {
            "QuantLib": {"IIDStdUniform": "o", "Sobol": "s"},
            "QMCPy": {
                "IIDStdUniform": "^",
                "Sobol": "v",
                "Lattice": "D",
                "Halton": "p",
            },
        },
    }


def plot_single_series(
    ax: Axes,
    plot_data: pd.DataFrame,
    series_name: str,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    log_scale: bool = False,
    is_legend: bool = False,
) -> None:
    """Plot a single series (runtime or error) for one experimental series"""
    series_data = plot_data[plot_data["Series"] == series_name]
    styling = get_plot_styling()

    # Collect all unique x values from the experiments
    all_x_values = sorted(series_data[x_col].unique())

    for method in ["QuantLib", "QMCPy"]:
        method_data = series_data[series_data["Method"] == method]
        colors = styling["colors"][method]
        markers = styling["markers"][method]

        # Cache unique samplers to avoid recomputation
        unique_samplers = method_data["Sampler"].unique()

        for sampler in unique_samplers:
            sampler_data = method_data[method_data["Sampler"] == sampler].sort_values(
                x_col
            )

            if len(sampler_data) > 0:
                x_vals = sampler_data[x_col].values
                y_vals = sampler_data[y_col].values

                color = colors.get(sampler, "#000000")
                marker = markers.get(sampler, "o")

                # Plot with connecting lines for trend visualization
                if log_scale:
                    ax.loglog(
                        x_vals,
                        y_vals,
                        marker=marker,
                        color=color,
                        linewidth=2,
                        markersize=8,
                        label=f"{method} - {sampler}",
                    )
                else:
                    ax.semilogy(
                        x_vals,
                        y_vals,
                        marker=marker,
                        color=color,
                        linewidth=2,
                        markersize=8,
                        label=f"{method} - {sampler}",
                    )

    # Set x-axis ticks to show only exact experimental values
    # Pre-compute tick labels once to avoid redundant string conversions
    tick_labels = [str(int(x)) for x in all_x_values]
    ax.set_xticks(all_x_values)
    ax.set_xticklabels(tick_labels)

    # Disable minor ticks to prevent intermediate values from showing
    ax.tick_params(axis="x", which="minor", bottom=False)

    # For log plots, we need to explicitly control the x-axis formatter
    if log_scale:
        ax.xaxis.set_major_locator(FixedLocator(all_x_values))
        ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
        ax.xaxis.set_minor_locator(FixedLocator([]))  # Remove minor ticks

    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if is_legend:
        ax.legend(fontsize=10)


def create_parameter_sweep_plots(df: pd.DataFrame, replications: int) -> None:
    """Create 4-panel plots from parameter sweep data"""
    # Filter out theoretical data
    plot_data = df[df["Method"] != "Theoretical"].copy()

    # Create figure with 2x2 subplots
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    # Panel 1: Mean Absolute Error vs n_steps (upper left)
    plot_single_series(
        ax1,
        plot_data,
        "Time Steps",
        "n_steps",
        "Mean Absolute Error",
        f"Mean Absolute Error vs Number of Time Steps across {replications} Replications\n(n_paths = 4,096)",
        "Number of Time Steps",
        "Mean Absolute Error",
        log_scale=True,
    )

    # Panel 2: Runtime vs n_steps (upper right)
    plot_single_series(
        ax2,
        plot_data,
        "Time Steps",
        "n_steps",
        "Runtime (s)",
        f"Runtime vs Number of Time Steps\n(n_paths = 4,096)",
        "Number of Time Steps",
        "Runtime (seconds)",
        log_scale=True,
        is_legend=True,
    )

    # Panel 3: Mean Absolute Error vs n_paths (lower left)
    plot_single_series(
        ax3,
        plot_data,
        "Paths",
        "n_paths",
        "Mean Absolute Error",
        f"Mean Absolute Error vs Number of Paths across {replications} Replications\n(n_steps = 252)",
        "Number of Paths",
        "Mean Absolute Error",
        log_scale=True,
    )

    # Panel 4: Runtime vs n_paths (lower right)
    plot_single_series(
        ax4,
        plot_data,
        "Paths",
        "n_paths",
        "Runtime (s)",
        f"Runtime vs Number of Paths\n(n_steps = 252)",
        "Number of Paths",
        "Runtime (seconds)",
        log_scale=True,
    )


def plot_paths(
    motion_type: str,
    sampler,
    t_final: float,
    initial_value: float,
    drift: float,
    diffusion: float,
    n: int,
    png_filename: Optional[str] = None,
):
    """
    Plot realizations of Brownian Motion or Geometric Brownian Motion.

    Args:
        motion_type: 'BM' for Brownian Motion or 'GBM' for Geometric Brownian Motion
        sampler: QMCPy sampler instance
        t_final: Final time point
        initial_value: Initial value S(0)
        drift: Drift coefficient (mu)
        diffusion: Diffusion coefficient (sigma^2 for GBM, sigma for BM)
        n: Number of paths to generate
        png_filename: Optional filename to save plot (saved to images/ directory)

    Returns:
        Motion object used for generation
    """
    if motion_type.upper() == "BM":
        motion = qp.BrownianMotion(sampler, t_final, initial_value, drift, diffusion)
        title = (
            f"Realizations of Brownian Motion using " f"{type(sampler).__name__} points"
        )
        ylabel = "$W(t)$"
    elif motion_type.upper() == "GBM":
        motion = qp.GeometricBrownianMotion(
            sampler, t_final, initial_value, drift, diffusion
        )
        title = (
            f"Realizations of Geometric Brownian Motion using "
            f"{type(sampler).__name__} points"
        )
        ylabel = "$S(t)$"
    else:
        raise ValueError("motion_type must be 'BM' or 'GBM'")

    t = motion.gen_samples(n)
    initial_values = np.full((n, 1), motion.initial_value)
    t_w_init = np.hstack((initial_values, t))
    tvec_w_0 = np.hstack(([0], motion.time_vec))

    plt.figure(figsize=(7, 4))
    plt.plot(tvec_w_0, t_w_init.T)
    plt.title(title)
    plt.xlabel("$t$")
    plt.ylabel(ylabel)
    plt.xlim([tvec_w_0[0], tvec_w_0[-1]])
    if png_filename:
        os.makedirs("images", exist_ok=True)
        plt.savefig(f"images/{png_filename}.png", bbox_inches="tight")
    plt.show()

    return motion


def plot_gbm_paths_with_distribution(
    N: int,
    sampler,
    t_final: float,
    initial_value: float,
    drift: float,
    diffusion: float,
    n: int,
) -> None:
    """
    Plot GBM paths with distribution of final values.

    Combines path visualization with histogram and fitted lognormal distribution.

    Args:
        N: Number of simulations (for display purposes)
        sampler: QMCPy sampler instance
        t_final: Final time point
        initial_value: Initial value S(0)
        drift: Drift coefficient (mu)
        diffusion: Diffusion coefficient (sigma^2)
        n: Power of 2 for number of paths (generates 2^n paths)
    """
    gbm = qp.GeometricBrownianMotion(
        sampler,
        t_final=t_final,
        initial_value=initial_value,
        drift=drift,
        diffusion=diffusion,
    )
    gbm_path = gbm.gen_samples(2**n)

    _, ax = plt.subplots(figsize=(14, 7))
    T = max(gbm.time_vec)

    # Plot GBM paths
    ax.plot(gbm.time_vec, gbm_path.T, lw=0.75, alpha=0.7, color="skyblue")

    # Set up main plot
    ax.set_title(
        f"Geometric Brownian Motion Paths\n"
        f"{N} Simulations, T = {T}, $\\mu$ = {drift:.1f}, "
        f"$\\sigma$ = {diffusion:.1f}, using {type(sampler).__name__} points"
    )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$S(t)$")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, T)

    # Add histogram
    final_values = gbm_path[:, -1]
    hist_ax = ax.inset_axes([1.05, 0.0, 0.5, 1])
    hist_ax.hist(
        final_values,
        bins=20,
        density=True,
        alpha=0.5,
        color="skyblue",
        orientation="horizontal",
    )

    # Add theoretical lognormal PDF
    shape, _, scale = sc.lognorm.fit(final_values, floc=0)
    x = np.linspace(0, max(final_values), 1000)
    pdf = sc.lognorm.pdf(x, shape, loc=0, scale=scale)
    hist_ax.plot(pdf, x, "r-", lw=2, label="Lognormal PDF")

    # Finalize histogram
    hist_ax.set_title(f"E[$S_T$] = {np.mean(final_values):.4f}", pad=20)
    hist_ax.axhline(
        np.mean(final_values), color="blue", linestyle="--", lw=1.5, label=r"$E[S_T]$"
    )
    hist_ax.set_yticks([])
    hist_ax.set_xlabel("Density")
    hist_ax.legend()
    hist_ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()


def compute_theoretical_covariance(
    S0: float, mu: float, sigma: float, t1: float, t2: float
) -> npt.NDArray[np.floating]:
    """
    Compute theoretical covariance matrix for GBM at two time points.

    Args:
        S0: Initial value
        mu: Drift coefficient
        sigma: Volatility
        t1: First time point
        t2: Second time point

    Returns:
        2x2 covariance matrix
    """
    return np.array(
        [
            [
                S0**2 * np.exp(2 * mu * t1) * (np.exp(sigma**2 * t1) - 1),
                S0**2 * np.exp(mu * (t1 + t2)) * (np.exp(sigma**2 * t1) - 1),
            ],
            [
                S0**2 * np.exp(mu * (t1 + t2)) * (np.exp(sigma**2 * t1) - 1),
                S0**2 * np.exp(2 * mu * t2) * (np.exp(sigma**2 * t2) - 1),
            ],
        ]
    )


def calculate_theoretical_statistics(params: dict) -> tuple[float, float]:
    """
    Calculate theoretical mean and standard deviation for GBM.

    Args:
        params: Dictionary with keys 'initial_value', 'mu', 'sigma', 'maturity'

    Returns:
        tuple: (theoretical_mean, theoretical_std)
    """
    theoretical_mean = params["initial_value"] * np.exp(
        params["mu"] * params["maturity"]
    )
    theoretical_std = np.sqrt(
        params["initial_value"] ** 2
        * np.exp(2 * params["mu"] * params["maturity"])
        * (np.exp(params["sigma"] ** 2 * params["maturity"]) - 1)
    )
    return theoretical_mean, theoretical_std


def extract_covariance_samples(
    paths: npt.NDArray[np.floating], n_steps: int, is_quantlib: bool = True
) -> npt.NDArray[np.floating]:
    """
    Extract samples at two time points and compute covariance matrix.

    Args:
        paths: Generated paths array
        n_steps: Number of time steps
        is_quantlib: True for QuantLib paths, False for QMCPy paths

    Returns:
        Covariance matrix for samples at two time points
    """
    if is_quantlib:
        idx1, idx2 = int(0.5 * n_steps), n_steps
        samples_t1, samples_t2 = paths[:, idx1], paths[:, idx2]
    else:  # QMCPy
        idx1, idx2 = int(0.5 * (n_steps - 1)), n_steps - 1
        samples_t1, samples_t2 = paths[:, idx1], paths[:, idx2]
    return np.cov(np.vstack((samples_t1, samples_t2)))
