from __future__ import annotations

from typing import TYPE_CHECKING, Union
import numpy as np
import os
import qmcpy as qp

if TYPE_CHECKING:
    from ..discrete_distribution.abstract_discrete_distribution import (
        AbstractDiscreteDistribution,
    )
    from ..true_measure.abstract_true_measure import AbstractTrueMeasure
    import matplotlib.figure


def plot_proj(
    sampler: Union[AbstractDiscreteDistribution, AbstractTrueMeasure],
    n: Union[int, list] = 64,
    d_horizontal: Union[int, list] = 1,
    d_vertical: Union[int, list] = 2,
    math_ind: bool = True,
    marker_size: float = 5,
    figfac: float = 5,
    fig_title: str = "Projection of Samples",
    axis_pad: float = 0,
    want_grid: bool = True,
    font_family: str = "sans-serif",
    where_title: float = 1,
    **kwargs: dict
) -> matplotlib.figure.Figure:
    """Plot two-dimensional projections of a point set.

    Args:
        sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]): The generator of samples
            to be plotted.
        n (Union[int, list]): The number of samples or a list of samples(used
            for extensibility) to be plotted.
        d_horizontal (Union[int, list]): The dimension or list of dimensions to
            be plotted on the horizontal axes.
        d_vertical (Union[int, list]): The dimension or list of dimensions to
            be plotted on the vertical axes.
        math_ind (bool): Setting to `True` will enable user to pass in math
            indices.
        marker_size (float): The marker size (typographic points are 1/72 in.).
        figfac (float): The figure size factor.
        fig_title (str): The title of the figure.
        axis_pad (float): The padding of the axis so that points on the
            boundaries can be seen.
        want_grid (bool): Setting to `True` will enable grid on the plot.
        font_family (str): The font family of the plot.
        where_title (float): the position of the title on the plot. Default
            value is 1.
        **kwargs (dict): Additional keyword arguments passed to
            `matplotlib.pyplot.scatter`.

    Returns:
        matplotlib.figure.Figure: The created figure.
        matplotlib.axes.Axes: Array of subplot axes, one per
            (`d_horizontal`, `d_vertical`) pair.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colors

        dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.style.use(os.path.join(dir_path, "qmcpy.mplstyle"))
    except ImportError as exc:
        raise ImportError(
            "Missing matplotlib.pyplot as plt, Matplotlib must be installed to run plot_proj function"
        ) from exc
    plt.rcParams["font.family"] = font_family
    n = np.atleast_1d(n)
    d_horizontal = np.atleast_1d(d_horizontal)
    d_vertical = np.atleast_1d(d_vertical)
    samples = sampler(n[n.size - 1])
    d = samples.shape[1]
    fig, ax = plt.subplots(
        nrows=d_horizontal.size,
        ncols=d_vertical.size,
        figsize=(figfac * d_horizontal.size, figfac * d_vertical.size),
        squeeze=False,
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i in range(d_horizontal.size):
        for j in range(d_vertical.size):
            n_min = 0
            for m in range(n.size):
                n_max = n[m]
                if d_horizontal[i] == d_vertical[j]:
                    ax[i, j].remove()
                    break
                if math_ind is True:
                    x = d_horizontal[i] - 1
                    y = d_vertical[j] - 1
                    x_label_num = d_horizontal[i]
                    y_label_num = d_vertical[j]
                else:
                    x = d_horizontal[i]
                    y = d_vertical[j]
                    x_label_num = d_horizontal[i] + 1
                    y_label_num = d_vertical[j] + 1

                if isinstance(sampler, qp.AbstractDiscreteDistribution):
                    ax[i, j].set_xlim([0 - axis_pad, 1 + axis_pad])
                    ax[i, j].set_ylim([0 - axis_pad, 1 + axis_pad])
                    ax[i, j].set_xticks([0, 1 / 4, 1 / 2, 3 / 4, 1])
                    ax[i, j].set_yticks([0, 1 / 4, 1 / 2, 3 / 4, 1])
                    ax[i, j].set_aspect(1)
                    if not want_grid:
                        ax[i, j].grid(False)
                        ax[i, j].tick_params(
                            axis="both", which="both", direction="in", length=5
                        )
                    ax[i, j].grid(want_grid)
                    x_label = r"$x_{i%d}$" % (x_label_num)
                    y_label = r"$x_{i%d}$" % (y_label_num)
                else:
                    x_label = r"$t_{i%d}$" % (x_label_num)
                    y_label = r"$t_{i%d}$" % (y_label_num)

                ax[i, j].set_xlabel(x_label, fontsize=15)
                if d > 1:
                    ax[i, j].set_ylabel(y_label, fontsize=15)
                    y_axis = samples[n_min:n_max, y]
                else:
                    y_axis = []
                    for h in range(n_max - n_min):
                        y_axis.append(0.5)
                ax[i, j].scatter(
                    samples[n_min:n_max, x],
                    y_axis,
                    s=marker_size,
                    color=colors[m],
                    label="n_min = %d, n_max = %d" % (n_min, n_max),
                    **kwargs,
                )
                n_min = n[m]
    plt.suptitle(fig_title, fontsize=20, y=where_title)
    # fig.text(0.55,0.55,fig_title, ha = 'center', va = 'center', fontsize = 20) %replaced by plt.suptitle
    fig.tight_layout()  # pad=2)
    return fig, ax
