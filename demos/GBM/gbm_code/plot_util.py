import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FixedFormatter
from typing import Optional


def plot_error_comparison(
    ax: Axes,
    samplers: list,
    qmcpy_errors: npt.NDArray[np.floating],
    quantlib_errors: list,
    replications: Optional[int] = None,
    metric: str = "MAE",
) -> None:
    """
    Plot error comparison subplot.

    Args:
        ax: Matplotlib axis object
        samplers: List of sampler names
        qmcpy_errors: Array of QMCPy errors for the chosen metric
        quantlib_errors: List of QuantLib errors for the chosen metric
            (may contain None)
        replications: Number of replications used for averaging (optional, for title)
        metric: Accuracy metric used for the axis label and title.
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
    ax.bar(
        x + width / 2, qmcpy_errors, width, label="QMCPy (PCA)", color="red", alpha=0.8
    )

    ax.set_xlabel("Sampler")
    ax.set_ylabel(f"{metric} (log scale)")

    # Add replications info to title if provided
    if replications is not None:
        ax.set_title(
            f"{metric}\n({replications}-replication average)",
            fontsize=14,
            fontweight="bold",
        )
    else:
        ax.set_title(metric, fontsize=14, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(samplers, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Compare only sampler pairs available in both libraries.
    paired_errors = [
        (i, qmc_error, ql_error)
        for i, (qmc_error, ql_error) in enumerate(
            zip(qmcpy_errors, quantlib_errors)
        )
        if ql_error is not None
        and np.isfinite(qmc_error)
        and np.isfinite(ql_error)
        and qmc_error > 0
        and ql_error > 0
    ]
    if paired_errors:
        max_error = max(max(qmc_error, ql_error) for _, qmc_error, ql_error in paired_errors)
        for i, qmc_error, ql_error in paired_errors:
            ratio = ql_error / qmc_error
            comparison = (
                f"{ratio:.1f}x lower"
                if ratio >= 1
                else f"{1 / ratio:.1f}x higher"
            )
            ax.annotate(
                comparison,
                xy=(i + width / 2, qmc_error),
                xytext=(i + width / 2, 1.5 * max(qmc_error, ql_error)),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="blue", lw=1),
            )
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom=bottom, top=max(top, 2.2 * max_error))


def plot_performance_comparison(
    ax: Axes,
    samplers: list,
    qmcpy_times: Optional[npt.NDArray[np.floating]],
    quantlib_times: list,
    timing_repeat: Optional[int] = None,
    timing_loops: Optional[int] = None,
) -> None:
    """
    Plot runtime comparison subplot.

    Args:
        ax: Matplotlib axis object
        samplers: List of sampler names
        qmcpy_times: Array of QMCPy mean runtimes, or None if unavailable
        quantlib_times: List of QuantLib mean runtimes (may contain None)
        timing_repeat: Number of `%timeit` runs behind each mean (its `-r`)
        timing_loops: Number of loops per run (its `-n`)

    Note:
        `timing_repeat` and `timing_loops` only describe how the runtime was
        timed. They are not the replication count used by the error panels:
        each timed call generates a single replication, whereas the errors are
        averaged over independent randomizations. Both are stated in the title
        so the two cannot be confused.
    """
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
        ax.bar(
            x + width / 2, qmcpy_times, width, label="QMCPy (PCA)", color="red", alpha=0.8
        )
        # Add speedup annotations where QuantLib data is available, at center of QMCPy bars
        if len(ql_times) > 0:
            # Offset every label by the same small fraction of the tallest bar,
            # measured from the taller bar of its own pair. Anchoring to the
            # QMCPy bar alone would leave labels sitting on top of a taller
            # QuantLib bar, and adding a fraction of the tallest bar to each
            # bar's own height would push the tallest pair's label off the axes.
            max_time = max(max(qmcpy_times), max(ql_times))
            for i, (qmc_time, ql_time) in enumerate(zip(qmcpy_times, quantlib_times)):
                if ql_time is not None:
                    speedup = ql_time / qmc_time
                    comparison = (
                        f"{speedup:.1f}x faster"
                        if speedup >= 1
                        else f"{1 / speedup:.1f}x slower"
                    )
                    annotation_height = max(qmc_time, ql_time) + 0.08 * max_time
                    # Position arrow at center of QMCPy bar (i + width/2)
                    ax.annotate(
                        comparison,
                        xy=(i + width / 2, qmc_time),
                        xytext=(i + width / 2, annotation_height),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="blue", lw=1),
                    )
            # Headroom so the highest label stays inside the axes, clear of the title
            ax.set_ylim(top=max_time * 1.25)
        ax.set_xlabel("Sampler")
        ax.set_ylabel("Runtime (s)")
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

    if timing_repeat is not None and timing_loops is not None:
        title = (
            f"Runtime\n(%timeit: mean of {timing_repeat} runs "
            f"x {timing_loops} loops, 1 replication)"
        )
    else:
        title = "Runtime"
    ax.set_title(title, fontsize=14, fontweight="bold")


def plot_construction_ablation(
    ax: Axes,
    ablation_df: pd.DataFrame,
    metric: str = "Mean Absolute Error",
    legend: bool = True,
) -> None:
    """
    Plot the path-construction ablation as grouped bars.

    One group per sampler, one bar per construction. Only `decomp_type` varies
    within a group, so bar-to-bar differences inside a group are attributable
    to the construction alone. The IID group is the control and should be
    roughly flat.

    This is deliberately a separate figure rather than extra series on the
    library comparison: adding three constructions there would triple the
    QMCPy series, and the constructions are only interpretable against each
    other, not against QuantLib.

    Args:
        ax: Matplotlib axis object
        ablation_df: Output of data_util.run_construction_ablation()
        metric: Column to plot, 'Mean Absolute Error', 'Std Dev Error', or
            'Runtime (s)'. The error metrics use a log scale; runtime does
            not, since it does not span orders of magnitude here.
        legend: Whether to draw the construction legend on this axis. The
            three subplots share one legend, so callers should only set this
            for one of them.
    """
    constructions = ["PCA", "Cholesky", "BrownianBridge"]
    labels = {"PCA": "PCA", "Cholesky": "Cholesky", "BrownianBridge": "Brownian bridge"}
    colors = {"PCA": "#d62728", "Cholesky": "#1f77b4", "BrownianBridge": "#2ca02c"}

    table = ablation_df.pivot(index="Sampler", columns="Construction", values=metric)
    samplers = [s for s in ablation_df["Sampler"].unique() if s in table.index]
    table = table.reindex(samplers)

    x = np.arange(len(samplers))
    width = 0.8 / len(constructions)
    for k, construction in enumerate(constructions):
        if construction not in table.columns:
            continue
        offset = (k - (len(constructions) - 1) / 2) * width
        ax.bar(
            x + offset,
            table[construction].values,
            width,
            label=f"QMCPy ({labels[construction]})",
            color=colors[construction],
            alpha=0.8,
        )

    is_runtime = metric == "Runtime (s)"
    if not is_runtime:
        ax.set_yscale("log")
    ax.set_xlabel("Sampler")
    ax.set_ylabel(metric if is_runtime else f"{metric} (log scale)")
    ax.set_title(
        f"{metric} by Path Construction\n(same point set within each group)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(samplers, rotation=45, ha="right")
    if legend:
        ax.legend()
    ax.grid(True, alpha=0.3)


def get_plot_styling() -> dict:
    """Define colors and markers for plotting"""
    return {
        "colors": {
            "QuantLib": {
                "IIDStdUniform": "#1f77b4",
                "Sobol": "#ff7f0e",
                "Halton": "#17becf",
            },
            "QMCPy": {
                "IIDStdUniform": "#2ca02c",
                "Sobol": "#d62728",
                "Halton": "#8c564b",
                "Lattice": "#9467bd",
            },
        },
        "markers": {
            "QuantLib": {"IIDStdUniform": "o", "Sobol": "s", "Halton": "*"},
            "QMCPy": {
                "IIDStdUniform": "^",
                "Sobol": "v",
                "Halton": "p",
                "Lattice": "D",
            },
        },
        "lines": {
            "QuantLib": {"linestyle": "-", "linewidth": 2},
            "QMCPy": {"linestyle": "--", "linewidth": 3},
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
        line_style = styling["lines"][method]

        available_samplers = set(method_data["Sampler"])
        unique_samplers = [sampler for sampler in colors if sampler in available_samplers]

        for sampler in unique_samplers:
            sampler_data = method_data[method_data["Sampler"] == sampler].sort_values(
                x_col
            )

            if len(sampler_data) > 0:
                x_vals = sampler_data[x_col].values
                y_vals = sampler_data[y_col].values

                color = colors.get(sampler, "#000000")
                marker = markers.get(sampler, "o")
                # Name the construction, since QMCPy's default (PCA) differs
                # from QuantLib's sequential fill and the two are not otherwise
                # distinguishable in the legend.
                series_label = (
                    f"QMCPy (PCA) - {sampler}"
                    if method == "QMCPy"
                    else f"{method} - {sampler}"
                )

                # Plot with connecting lines for trend visualization
                if log_scale:
                    ax.loglog(
                        x_vals,
                        y_vals,
                        marker=marker,
                        color=color,
                        markersize=8,
                        label=series_label,
                        **line_style,
                    )
                else:
                    ax.semilogy(
                        x_vals,
                        y_vals,
                        marker=marker,
                        color=color,
                        markersize=8,
                        label=series_label,
                        **line_style,
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
    """Create 6-panel plots from parameter sweep data.

    Rows are the swept parameter (time steps, then paths) and columns are the
    reported metric: error in the mean of S_T, error in its standard deviation,
    and runtime. SD error is not uncertainty attached to the MAE, so it
    gets its own column rather than error bars.
    """
    # Filter out theoretical data
    plot_data = df[df["Method"] != "Theoretical"].copy()

    # Create figure with 2x3 subplots
    _, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(21, 12))
    # Panel 1: MAE vs n_steps (upper left)
    plot_single_series(
        ax1,
        plot_data,
        "Time Steps",
        "n_steps",
        "Mean Absolute Error",
        f"MAE vs Time Steps\n(n_paths = 4,096, R = {replications})",
        "Number of Time Steps",
        "MAE",
        log_scale=True,
    )

    # Panel 2: SD error vs n_steps (upper middle)
    plot_single_series(
        ax2,
        plot_data,
        "Time Steps",
        "n_steps",
        "Std Dev Error",
        f"SD Error vs Time Steps\n(n_paths = 4,096, R = {replications})",
        "Number of Time Steps",
        "SD Error",
        log_scale=True,
    )

    # Panel 3: Runtime vs n_steps (upper right)
    plot_single_series(
        ax3,
        plot_data,
        "Time Steps",
        "n_steps",
        "Runtime (s)",
        "Runtime vs Time Steps\n(n_paths = 4,096)",
        "Number of Time Steps",
        "Runtime (seconds)",
        log_scale=True,
        is_legend=True,
    )

    # Panel 4: MAE vs n_paths (lower left)
    plot_single_series(
        ax4,
        plot_data,
        "Paths",
        "n_paths",
        "Mean Absolute Error",
        f"MAE vs Paths\n(n_steps = 252, R = {replications})",
        "Number of Paths",
        "MAE",
        log_scale=True,
    )

    # Panel 5: SD error vs n_paths (lower middle)
    plot_single_series(
        ax5,
        plot_data,
        "Paths",
        "n_paths",
        "Std Dev Error",
        f"SD Error vs Paths\n(n_steps = 252, R = {replications})",
        "Number of Paths",
        "SD Error",
        log_scale=True,
    )

    # Panel 6: Runtime vs n_paths (lower right)
    plot_single_series(
        ax6,
        plot_data,
        "Paths",
        "n_paths",
        "Runtime (s)",
        "Runtime vs Paths\n(n_steps = 252)",
        "Number of Paths",
        "Runtime (seconds)",
        log_scale=True,
    )
