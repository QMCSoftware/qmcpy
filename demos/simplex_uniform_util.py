"""Numerical and plotting helpers for the simplex demo."""

import itertools
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from qmcpy import SimplexUniform


def transformed_sum_errors(
    cube_point_sets: Iterable[np.ndarray],
    transform_methods: Sequence[str],
    exact: float,
) -> dict[str, list[Any]]:
    """Compute simplex-coordinate-sum errors for several transforms.

    Args:
        cube_point_sets (Iterable[np.ndarray]): Cube point arrays, one per
            sample size.
        transform_methods (Sequence[str]): Simplex transform names to compare.
        exact (float): Exact expected sum of the stored simplex coordinates.

    Returns:
        dict[str, list[Any]]: Errors keyed by transform name.
    """
    errors = {method: [] for method in transform_methods}
    for cube_points in cube_point_sets:
        for method in transform_methods:
            weights = SimplexUniform.transform_points(
                cube_points,
                transform_method=method,
                simplex_type="corner",
            )
            estimate = weights.sum(axis=-1).mean(axis=-1)
            errors[method].append(abs(estimate - exact))
    return errors


def plot_simplex_convergence(
    sequence: str,
    point_type: str,
    dimension: int,
    sample_sizes: Sequence[int],
    transform_methods: Sequence[str],
    markers: Mapping[str, str],
    colors: Mapping[str, str],
    qmc_errors: Mapping[str, Sequence[float]],
    iid_errors: Mapping[str, Sequence[float]],
    reference: Sequence[float],
) -> tuple[Figure, str]:
    """Plot QMC and IID errors for the selectable simplex transforms.

    Args:
        sequence (str): Display name of the low-discrepancy sequence.
        point_type (str): Display name of its randomization or construction.
        dimension (int): Dimension of the stored corner-simplex coordinates.
        sample_sizes (Sequence[int]): Sample sizes corresponding to the errors.
        transform_methods (Sequence[str]): Simplex transform names to compare.
        markers (Mapping[str, str]): Marker keyed by transform name.
        colors (Mapping[str, str]): Color keyed by transform name.
        qmc_errors (Mapping[str, Sequence[float]]): QMC errors keyed by
            transform name.
        iid_errors (Mapping[str, Sequence[float]]): IID errors keyed by
            transform name.
        reference (Sequence[float]): Values of the Monte Carlo reference rate.

    Returns:
        tuple[Figure, str]: Figure and a one-line comparison
            at the largest sample size.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    for method in transform_methods:
        ax.loglog(
            sample_sizes,
            qmc_errors[method],
            marker=markers[method],
            linestyle="-",
            color=colors[method],
            label=f"QMC {method}",
        )
        ax.loglog(
            sample_sizes,
            iid_errors[method],
            marker=markers[method],
            linestyle=":",
            color=colors[method],
            label=f"IID {method}",
        )
    ax.loglog(
        sample_sizes,
        reference,
        "k--",
        linewidth=1,
        label=r"$n^{-1/2}$ reference",
    )
    ax.set_ylim(1e-9, 1e-1)
    ax.set_xlabel("n")
    ax.set_ylabel(r"error in $\sum_i w_i$")
    ax.set_title(
        f"{sequence} ({point_type}) vs. IID", fontweight="bold", fontsize=11
    )
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        fontsize=9,
        ncol=4,
    )
    fig.suptitle(
        rf"QMC vs. IID Convergence Over $K_{dimension}$, by transform_method",
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.17, 1, 0.97))

    best_iid = min(iid_errors[method][-1] for method in transform_methods)
    ratio = best_iid / qmc_errors["root"][-1]
    summary = (
        f"At the largest n, the best IID error is {ratio:.0f}× the "
        f"{sequence} ({point_type}) root error (single run, no replications)."
    )
    return fig, summary


def plot_cube_normalize_transform(
    sampler_factory: Callable[..., Any],
    sequence: str,
    point_type: str,
    method: str,
    n_plot: int = 256,
    seed: int = 42,
) -> Figure:
    """Plot shared cube, normalized, and transformed points side by side.

    Args:
        sampler_factory (Callable[..., Any]): Factory for the selected
            low-discrepancy sequence.
        sequence (str): Display name of the low-discrepancy sequence.
        point_type (str): Display name of its randomization or construction.
        method (str): Simplex transform name.
        n_plot (int): Number of points to plot.
        seed (int): Seed passed to the sampler factory.

    Returns:
        Figure: Figure containing the three point clouds.
    """
    cube_points = sampler_factory(dimension=3, seed=seed)(n_plot, warn=False)
    totals = cube_points.sum(axis=-1, keepdims=True)
    naive = np.divide(
        cube_points,
        totals,
        out=np.full_like(cube_points, np.nan),
        where=totals != 0,
    )
    weights = SimplexUniform.transform_points(
        cube_points[:, :2],
        transform_method=method,
        simplex_type="corner",
    )
    transformed = np.concatenate(
        [weights, 1 - weights.sum(axis=-1, keepdims=True)], axis=-1
    )
    boundary = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])

    stages = [
        ("Shared cube points", cube_points, "black"),
        ("Naive sum-normalized", naive, "crimson"),
        (f"{method.capitalize()}-transformed", transformed, "darkgreen"),
    ]
    fig = plt.figure(figsize=(15, 5))
    for column, (label, points, color) in enumerate(stages):
        ax = fig.add_subplot(1, 3, column + 1, projection="3d")
        ax.scatter(*points.T, s=10, alpha=0.6, color=color, depthshade=False)
        if column == 0:
            for corner, axis in itertools.product(np.ndindex(2, 2, 2), range(3)):
                if corner[axis] == 0:
                    end = list(corner)
                    end[axis] = 1
                    ax.plot(*np.array([corner, end]).T, color="0.65", linewidth=0.7)
        else:
            ax.plot(*boundary.T, color="0.25", linewidth=1.2)
        ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
        coordinate = "u" if column == 0 else "w"
        ax.set_xlabel(rf"${coordinate}_1$")
        ax.set_ylabel(rf"${coordinate}_2$")
        ax.set_zlabel(rf"${coordinate}_3$")
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=38)
    fig.suptitle(
        f"{sequence} ({point_type}), {method} transform",
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout(pad=2.0)
    return fig
