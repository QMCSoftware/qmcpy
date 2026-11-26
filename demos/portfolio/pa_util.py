"""Generic, non-portfolio-specific plotting/display utilities shared by this
demo's notebooks.
"""

import ast
import itertools
import inspect
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import ipywidgets as widgets
from IPython.display import display, HTML
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
# Needed eagerly, not just for the isinstance check below: pandas.io.formats.style is
# otherwise lazily loaded, and style_by_value's isinstance check would AttributeError
# before anything else triggered that load (e.g. when called directly on a plain
# DataFrame, not chained after style_by_frequency's own `df.style`).
from pandas.io.formats.style import Styler


def _strip_docstring(src):
    """Remove a function/class's own docstring statement from its source text.

    Args:
        src (str): Source text, e.g. from inspect.getsource; may be indented
            (a method's source keeps its class-body indentation).

    Returns:
        str: src with the docstring line(s) removed, or src unchanged if it
            has none.
    """
    tree = ast.parse(textwrap.dedent(src))
    body = tree.body[0].body
    has_docstring = (body and isinstance(body[0], ast.Expr)
                      and isinstance(body[0].value, ast.Constant)
                      and isinstance(body[0].value.value, str))
    if not has_docstring:
        return src
    doc_node = body[0]
    lines = src.splitlines(keepends=True)
    del lines[doc_node.lineno - 1:doc_node.end_lineno]
    return ''.join(lines)


def show_source(obj, style='monokai', max_lines=None, show_docstring=False):
    """Display a function/class's source with syntax highlighting.

    A plain print(inspect.getsource(obj)) renders as uncolored text.
    noclasses=True bakes each token's color into its own inline style=
    attribute instead of CSS classes resolved against a separate <style>
    block, so the highlighting survives notebook frontends that sanitize
    (strip) <style> tags from HTML outputs.

    Args:
        obj (callable or type): Function or class to display the source of.
        style (str): Pygments style name, e.g. 'monokai', 'dracula',
            'github-dark', 'one-dark', 'native'.
        max_lines (int, optional): Show only the first this many lines
            (applied after dropping the docstring, if show_docstring is
            False), with a trailing comment noting how many were omitted.
            None (default): show every line.
        show_docstring (bool): If False, omit the docstring; the
            def/class line and signature are always kept.

    Returns:
        IPython.display.HTML: Rendered source; display it or return it as
            a cell's last expression.
    """
    src = inspect.getsource(obj)
    if not show_docstring:
        src = _strip_docstring(src)
    lines = src.splitlines()
    if max_lines is not None and len(lines) > max_lines:
        omitted = len(lines) - max_lines
        lines = lines[:max_lines] + [f"# ... ({omitted} more line{'s' if omitted != 1 else ''})"]
    formatter = HtmlFormatter(style=style, noclasses=True)
    return HTML(highlight('\n'.join(lines), PythonLexer(), formatter))


def nice_log_ticks(vmin, vmax, target_n=6):
    """Evenly-spaced 'nice' tick values spanning [vmin, vmax].

    The default log locator only fills in sub-decade ticks below the nearest
    power of ten, leaving everything above it unlabeled when the data doesn't
    span a full decade. Explicit ticks sidestep that.

    Args:
        vmin (float): Lower bound of the range to tick.
        vmax (float): Upper bound of the range to tick.
        target_n (int): Approximate number of ticks to produce.

    Returns:
        ndarray: Tick values.
    """
    raw_step = (vmax - vmin) / target_n
    magnitude = 10 ** np.floor(np.log10(raw_step))
    for mult in (1, 2, 2.5, 5, 10):
        step = magnitude * mult
        if step >= raw_step:
            break
    start = np.ceil(vmin / step) * step
    return np.arange(start, vmax, step)


def set_title_with_extremes(ax, main_title, best_label, best_value, worst_label, worst_value,
                             main_fontsize=13, extreme_fontsize=10):
    """Set a subplot's title plus one color-coded line above it naming its
    two extremes side by side (e.g. best/worst sampler, or fastest/slowest).

    Args:
        ax (matplotlib.axes.Axes): Subplot to title.
        main_title (str): The subplot's own title, e.g. 'Low Risk'.
        best_label (str): Prefix for the first (green) extreme, e.g. 'Best'.
        best_value (str): The value for that extreme, e.g. a sampler name.
        worst_label (str): Prefix for the second (red) extreme, e.g. 'Worst'.
        worst_value (str): The value for that extreme.
        main_fontsize (int): Font size for the main title.
        extreme_fontsize (int): Font size for the extremes line.
    """
    ax.text(0.27, 1.14, f'{best_label}: {best_value}', transform=ax.transAxes,
            ha='center', fontsize=extreme_fontsize, fontweight='bold', color='green')
    ax.text(0.73, 1.14, f'{worst_label}: {worst_value}', transform=ax.transAxes,
            ha='center', fontsize=extreme_fontsize, fontweight='bold', color='red')
    ax.set_title(main_title, fontsize=main_fontsize, fontweight='bold')


def plot_3d_stages(stages, title_prefix='', figsize=(15, 5), elev=24, azim=38):
    """Render a row of 3D scatter panels, one per (label, points, color) stage.

    The first panel draws a unit-cube wireframe behind its points (axes
    u_1, u_2, u_3); every later panel draws a fixed simplex-boundary triangle
    behind its points instead (axes w_1, w_2, w_3), meant for a pipeline
    that moves points from the unit cube onto the simplex stage by stage.

    Args:
        stages (list[tuple[str, np.ndarray, str]]): (label, points, color)
            per panel, left to right. points must have shape (n, 3).
        title_prefix (str): Prepended to each panel's own stage label, e.g.
            a sampler name.
        figsize (tuple[float, float]): Figure size.
        elev (float): 3D view elevation angle, in degrees.
        azim (float): 3D view azimuth angle, in degrees.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    simplex_boundary = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    fig = plt.figure(figsize=figsize)
    for column, (stage_label, points, point_color) in enumerate(stages):
        ax = fig.add_subplot(1, len(stages), column + 1, projection='3d')
        ax.scatter(*points.T, s=10, alpha=0.65, color=point_color, depthshade=False)
        if column == 0:
            for corner, axis in itertools.product(np.ndindex(2, 2, 2), range(3)):
                if corner[axis] == 0:
                    end = list(corner)
                    end[axis] = 1
                    ax.plot(*np.array([corner, end]).T, color='0.65', linewidth=0.7)
            axis_symbols = ('u_1', 'u_2', 'u_3')
        else:
            ax.plot(*simplex_boundary.T, color='0.25', linewidth=1.2)
            axis_symbols = ('w_1', 'w_2', 'w_3')
        ax.set(
            xlim=(0, 1), ylim=(0, 1), zlim=(0, 1),
            xlabel=f'${axis_symbols[0]}$', ylabel=f'${axis_symbols[1]}$', zlabel=f'${axis_symbols[2]}$',
            title=f'{title_prefix}{stage_label}',
        )
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
    fig.tight_layout(pad=2.0)
    return fig


def sync_ylim(axes):
    """Give every axis in `axes` the same y-limits: the union of their own
    independently autoscaled ranges.

    Not matplotlib's own `sharey=True`: that re-hides a shared axis group's
    interior labels every time a later sibling axis is drawn, even after an
    explicit `set_visible(True)`. Computing and applying the range by hand
    sidesteps that entirely.

    Args:
        axes (array-like of matplotlib.axes.Axes): Subplots to synchronize.
            Must be non-empty and share a figure.

    Returns:
        tuple[float, float]: The shared (vmin, vmax) now applied to every axis.
    """
    axes[0].figure.canvas.draw()  # finalize autoscaling so get_ylim() below is accurate
    vmin = min(ax.get_ylim()[0] for ax in axes)
    vmax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(vmin, vmax)
    return vmin, vmax


def display_filterable_table(df, value_cols, filter_col, cmap='RdYlGn', fmt='{:.4f}', option_order=None, relabel=None, reverse_substring='std', extra_filter_cols=(), defaults=None):
    """Interactive, color-graded view of a results DataFrame, filterable by one or more categorical columns.

    Uses a plain display(styled, display_id=True) call so the table is a
    normal, statically-visible cell output even without a live kernel,
    updated in place via .update() when a filter control changes.

    Args:
        df (pd.DataFrame): Results table to display.
        value_cols (list[str]): Numeric columns to color-grade, min to max.
        filter_col (str): Primary categorical column to filter by (e.g.
            'sampler'); the only one option_order/relabel apply to.
        cmap (str): Matplotlib colormap name for the background gradient.
        fmt (str): Format string applied to value_cols.
        option_order (list[str], optional): Display order for filter_col's
            values (after relabel, if given); defaults to alphabetical.
        relabel (dict[str, str], optional): Value replacements to apply to
            filter_col before building filter options, e.g. to rename a
            caller-specific suffix for display.
        reverse_substring (str, optional): value_cols whose name contains this
            (case-insensitive) are graded with cmap reversed, since for a
            dispersion column (e.g. a standard deviation) smaller is better;
            the opposite of every other column here, where larger is better.
            Pass None to disable and grade every column the same way.
        extra_filter_cols (list[str], optional): Additional categorical
            columns to filter by, each with its own SelectMultiple control;
            every active filter combines with AND. Always plain alphabetical
            options; option_order/relabel apply only to filter_col.
        defaults (dict, optional): Initial selection for filter_col and/or
            any extra_filter_cols entry, a single value or a list/tuple/set
            of values. A column not listed here defaults to every one of its
            values selected (no filtering), this function's prior behavior.
    """
    if relabel:
        df = df.copy()
        df[filter_col] = df[filter_col].replace(relabel)

    defaults = defaults or {}
    all_filter_cols = [filter_col] + list(extra_filter_cols)

    def make_control(col, is_primary):
        values = set(df[col].unique())
        if is_primary and option_order is not None:
            options = [s for s in option_order if s in values]
        else:
            options = sorted(values)
        default = defaults.get(col)
        if default is None:
            initial = tuple(options)
        else:
            requested = tuple(default) if isinstance(default, (list, tuple, set)) else (default,)
            # Falls back to every option if none of the requested default(s) are actually
            # present, e.g. a booktest's shrunk-for-speed replacement of the real sweep values:
            # an unmatched default should degrade to "show everything," not raise/crash.
            initial = tuple(v for v in requested if v in options) or tuple(options)
        return widgets.SelectMultiple(description=col.capitalize(), options=options, value=initial,
                                       rows=min(8, len(options)))

    controls = {col: make_control(col, col == filter_col) for col in all_filter_cols}
    table_handle = {'id': None}

    if reverse_substring:
        reverse_cols = [c for c in value_cols if reverse_substring.lower() in c.lower()]
    else:
        reverse_cols = []
    forward_cols = [c for c in value_cols if c not in reverse_cols]
    reverse_cmap = cmap if cmap.endswith('_r') else f'{cmap}_r'

    def update(_=None):
        filtered = df
        for col, control in controls.items():
            filtered = filtered[filtered[col].isin(control.value)]
        styled = _set_baseline_text_color(filtered.style)
        # vmin/vmax pinned to the full (unfiltered) df, one column at a time: a plain
        # subset=forward_cols call would let background_gradient recompute each column's
        # min/max from whatever rows the filters currently leave in, so the same value
        # shades differently depending on the filter selection, not just which rows show.
        for col in forward_cols:
            styled = styled.background_gradient(cmap=cmap, subset=[col], vmin=df[col].min(), vmax=df[col].max())
        for col in reverse_cols:
            styled = styled.background_gradient(cmap=reverse_cmap, subset=[col], vmin=df[col].min(), vmax=df[col].max())
        styled = styled.format({col: fmt for col in value_cols})
        if table_handle['id'] is None:
            table_handle['id'] = display(styled, display_id=True)
        else:
            table_handle['id'].update(styled)

    for control in controls.values():
        control.observe(update, names='value')
    display(widgets.VBox(list(controls.values())))
    update()


def plot_avg_portfolio_values(all_portfolios_dict, sr_dict, colors, label_fn, n_tickers=None, sample_type=None,
                               benchmark=None, benchmark_label='S&P 500'):
    """Plot averaged portfolio values across risk levels for each sampler.

    Args:
        all_portfolios_dict (dict): Nested {sampler: {risk_level: portfolio
            value DataFrame}}.
        sr_dict (dict): Per-sampler results, used only for its keys.
        colors (dict): Maps each sampler key in sr_dict to a plot color.
        label_fn (callable): sampler key -> display label, e.g. 'sobol' ->
            'Sobol'.
        n_tickers (int, optional): Shown in the figure title if given.
        sample_type (str, optional): e.g. 'OOS'; shown in the figure
            title if given, since the same figure recurs for several ticker
            counts and both in-sample and OOS backtests.
        benchmark (pd.Series, optional): A passive $ benchmark (e.g. S&P 500,
            already scaled to the same starting principal) plotted alongside
            the samplers for reference; excluded from the best/worst
            comparison below since it is not one of the samplers.
        benchmark_label (str): Legend label for benchmark.

    Each subplot's title also names whichever sampler ends the period with
    the highest ('Best') and lowest ('Worst') average portfolio value; this
    is about the single final value only, not the whole trajectory.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    risk_levels = ['low', 'medium', 'high']
    samplers = list(sr_dict.keys())
    plot_colors = [colors[s] for s in samplers]
    line_styles = ['-' if s.endswith('_simplex') else ':' for s in samplers]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    title = 'Average Portfolio Value by Risk Level'
    sample_type_display = 'In-Sample' if sample_type == 'in-sample' else sample_type
    bits = [f'{n_tickers} Tickers' if n_tickers is not None else None, sample_type_display]
    if any(bits):
        title += f" ({', '.join(b for b in bits if b)})"
    fig.suptitle(title, fontsize=16, fontweight='bold')
    for idx, risk in enumerate(risk_levels):
        portf_data = pd.DataFrame({
            label_fn(sampler): all_portfolios_dict[sampler][risk].mean(axis=1)
            for sampler in samplers
        })
        portf_data.plot(ax=axes[idx], logy=True, linewidth=1.5, color=plot_colors, style=line_styles, legend=False)
        if benchmark is not None:
            benchmark.plot(ax=axes[idx], logy=True, linewidth=1.5, linestyle='--', color='gray', label=benchmark_label)
        final_values = portf_data.iloc[-1]
        best, worst = final_values.idxmax(), final_values.idxmin()
        set_title_with_extremes(axes[idx], f'{risk.capitalize()} Risk', 'Best', best, 'Worst', worst)
        axes[idx].set_xlabel('Date')
        axes[idx].set_ylabel('Portfolio Value ($, log scale)')

    vmin, vmax = sync_ylim(axes)
    ticks = nice_log_ticks(vmin, vmax)
    for ax in axes:
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'${t:,.0f}' for t in ticks])
        ax.yaxis.set_minor_locator(mticker.NullLocator())  # drop stray minor-tick labels (e.g. a leftover 7x10^3)

    axes[0].legend(fontsize=14, ncol=2)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22, top=0.82)  # bottom: rotated date labels; top: room for the extremes line
    return fig


def plot_diff_vs_iid(all_portfolios_dict, principal, colors, label_fn, n_tickers=None, sample_type=None):
    """Plot each sampler against an IID baseline.

    Args:
        all_portfolios_dict (dict): Nested {sampler: {risk_level: portfolio
            value DataFrame}}; must contain 'iid' or 'iid_simplex' as the
            baseline key.
        principal (float): Dollar amount invested, used to express differences
            as a fraction of principal.
        colors (dict): Maps each non-baseline sampler key to a plot color.
        label_fn (callable): sampler key -> display label, e.g. 'sobol' ->
            'Sobol'.
        n_tickers (int, optional): Shown in the figure title if given.
        sample_type (str, optional): e.g. 'OOS'; shown in the figure
            title if given, since the same figure recurs for several ticker
            counts and both in-sample and OOS backtests.

    Each subplot's title also names whichever sampler ends the period
    highest ('Best') and lowest ('Worst') relative to the baseline; this is
    about the single final value only, not the whole trajectory.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    baseline = 'iid' if 'iid' in all_portfolios_dict else 'iid_simplex'
    risk_levels = ['low', 'medium', 'high']
    comparison_samplers = [s for s in all_portfolios_dict if s != baseline]
    plot_colors = [colors[s] for s in comparison_samplers]
    # Solid line = simplex-transformed ('_simplex'); bare dots = normalized, distinct even when values are close.
    line_styles = ['-' if s.endswith('_simplex') else '.' for s in comparison_samplers]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    title = f'Portfolio Value vs. {label_fn(baseline)} Baseline, by Risk Level'
    sample_type_display = 'In-Sample' if sample_type == 'in-sample' else sample_type
    bits = [f'{n_tickers} Tickers' if n_tickers is not None else None, sample_type_display]
    if any(bits):
        title += f" ({', '.join(b for b in bits if b)})"
    fig.suptitle(title, fontsize=16, fontweight='bold')
    for idx, risk in enumerate(risk_levels):
        diff_data = pd.DataFrame({
            label_fn(sampler): (
                all_portfolios_dict[sampler][risk] - all_portfolios_dict[baseline][risk]
            ).mean(axis=1) / principal
            for sampler in comparison_samplers
        })
        diff_data.plot(ax=axes[idx], linewidth=1.5, markersize=3, color=plot_colors, style=line_styles, legend=False)
        axes[idx].axhline(0, color='black', linestyle=':', linewidth=0.5)
        final_values = diff_data.iloc[-1]
        best, worst = final_values.idxmax(), final_values.idxmin()
        set_title_with_extremes(axes[idx], f'{risk.capitalize()} Risk', 'Best', best, 'Worst', worst)
        axes[idx].set_xlabel('Date')
        axes[idx].set_ylabel(f'Difference vs. {label_fn(baseline)} (fraction of principal)')

    sync_ylim(axes)
    axes[0].legend(fontsize=14, ncol=2)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22, top=0.82)
    return fig


def display_portfolio_figures(data_by_tickers, principal, colors, label_fn, sample_type=None, default=None):
    """Interactive ticker-count selector showing both plot_avg_portfolio_values
    and plot_diff_vs_iid for the selected count, instead of a fixed run of one
    static pair per count.

    Args:
        data_by_tickers (dict): n_tickers -> {'all_portfolios_dict', 'sr_dict',
            'benchmark'}, the inputs plot_avg_portfolio_values/plot_diff_vs_iid
            need for that ticker count.
        principal (float): Dollar amount invested, passed through to plot_diff_vs_iid.
        colors (dict): Maps each sampler key to a plot color.
        label_fn (callable): sampler key -> display label.
        sample_type (str, optional): e.g. 'in-sample' or 'OOS'; shown in both
            figures' titles.
        default (int, optional): Initial n_tickers selection. Falls back to
            the largest available key if not among data_by_tickers' keys.

    Returns:
        None (renders the control and figures as a side effect).
    """
    options = sorted(data_by_tickers)
    initial = default if default in data_by_tickers else options[-1]
    dropdown = widgets.Dropdown(description='Tickers', options=options, value=initial)
    # display_id handles rather than widgets.Output: Output's capture context manager
    # hangs nbconvert's headless kernel (no attached frontend to ack its comm messages).
    handles = {}

    def update(_=None):
        entry = data_by_tickers[dropdown.value]
        fig1 = plot_avg_portfolio_values(entry['all_portfolios_dict'], entry['sr_dict'], colors, label_fn,
                                          n_tickers=dropdown.value, sample_type=sample_type,
                                          benchmark=entry['benchmark'])
        fig2 = plot_diff_vs_iid(entry['all_portfolios_dict'], principal, colors, label_fn,
                                 n_tickers=dropdown.value, sample_type=sample_type)
        if not handles:
            handles['avg'] = display(fig1, display_id=True)
            handles['diff'] = display(fig2, display_id=True)
        else:
            handles['avg'].update(fig1)
            handles['diff'].update(fig2)
        plt.close(fig1)
        plt.close(fig2)

    dropdown.observe(update, names='value')
    display(dropdown)
    update()


def plot_runtime(df, sampler_types, colors, markers, label_fn, runtime_type='Runtime_real'):
    """Compare runtime vs. dimension and vs. sample count, one line per sampler.

    Args:
        df (pd.DataFrame): Long-format runtime data with columns 'Series'
            ('Tickers' or 'Portfolios'), 'Sampler', 'Tickers', 'Portfolios',
            and the runtime_type column.
        sampler_types (list[str]): Samplers to plot, in legend order.
        colors (dict): Maps each sampler key to a plot color.
        markers (dict): Maps each sampler key to a plot marker.
        label_fn (callable): sampler key -> display label, e.g. 'sobol' ->
            'Sobol'.
        runtime_type (str): Column to plot, e.g. 'Runtime_real' or 'Runtime_CPU'.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle('CPU Time' if runtime_type == 'Runtime_CPU' else 'Real (Wall-Clock) Time', fontsize=16, fontweight='bold')

    dim_final = {}
    port_final = {}
    for sampler in sampler_types:
        color = colors[sampler]
        marker = markers[sampler]
        linestyle = '-' if sampler.endswith('_simplex') else ':'
        sampler_data = df[df['Sampler'] == sampler]
        prefix = 'MC' if sampler.split('_')[0] == 'iid' else 'QMC'

        dim_data = sampler_data[sampler_data['Series'] == 'Tickers'].sort_values('Tickers')
        axes[0].loglog(
            dim_data['Tickers'],
            dim_data[runtime_type],
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=f"{prefix} {label_fn(sampler)}"
        )
        if len(dim_data):
            val = dim_data[runtime_type].iloc[-1]
            if pd.notna(val):
                dim_final[label_fn(sampler)] = val

        sample_data = sampler_data[sampler_data['Series'] == 'Portfolios'].sort_values('Portfolios')
        axes[1].loglog(
            sample_data['Portfolios'],
            sample_data[runtime_type],
            color=color,
            marker=marker,
            linestyle=linestyle
        )
        if len(sample_data):
            val = sample_data[runtime_type].iloc[-1]
            if pd.notna(val):
                port_final[label_fn(sampler)] = val

    # "Fastest"/"slowest" are at the largest dimension/portfolio-count tested in each
    # panel's own sweep, matching plot_diff_vs_iid's final-value convention.
    dim_fast, dim_slow = min(dim_final, key=dim_final.get), max(dim_final, key=dim_final.get)
    port_fast, port_slow = min(port_final, key=port_final.get), max(port_final, key=port_final.get)
    set_title_with_extremes(axes[0], 'Runtime vs. Dimension', 'Fastest', dim_fast, 'Slowest', dim_slow)
    set_title_with_extremes(axes[1], 'Runtime vs. Portfolio Count', 'Fastest', port_fast, 'Slowest', port_slow)
    axes[0].set_xlabel("Number of tickers (dimensions)", fontweight="bold")
    axes[1].set_xlabel("Number of portfolios (samples)", fontweight="bold")
    axes[0].set_ylabel("Runtime (seconds)", fontweight="bold")
    axes[1].set_ylabel("Runtime (seconds)", fontweight="bold")
    axes[0].legend(fontsize=14, ncol=2)
    plt.tight_layout()
    fig.subplots_adjust(top=0.82)
    return fig


def display_sampler_table(df, value_cols, sampler_types, transform_method, filter_col='sampler', **kwargs):
    """display_filterable_table, with a caller-supplied sampler ordering/
    relabeling: filter options follow sampler_types' own order instead of
    alphabetical, and any '<base>_simplex' entry is relabeled
    '<base>_<transform_method>' for display. The underlying data keeps
    '_simplex' everywhere else as a generic "simplex-transformed" dict-key
    tag, independent of which transform method is active.

    Args:
        df (pd.DataFrame): Results table to display.
        value_cols (list[str]): Numeric columns to color-grade, min to max.
        sampler_types (list[str]): The caller's own canonical sampler order,
            e.g. ['iid', 'iid_simplex', 'sobol', 'sobol_simplex', ...].
        transform_method (str): The caller's active simplex-transform method
            name, used only to build the '_simplex' -> '_<method>' relabel.
        filter_col (str): Categorical column to filter by, case-insensitively
            matched against 'sampler' to decide whether the relabeling above
            applies.
        **kwargs: Passed through to display_filterable_table (e.g. fmt).
    """
    if filter_col.lower() == 'sampler' and df[filter_col].astype(str).str.endswith('_simplex').any():
        relabel = {s: s.replace('_simplex', f'_{transform_method}') for s in sampler_types if s.endswith('_simplex')}
        option_order = [relabel.get(s, s) for s in sampler_types]
    else:
        relabel = option_order = None
    display_filterable_table(df, value_cols, filter_col, option_order=option_order, relabel=relabel, **kwargs)


def tied_names(name_to_value, target, name_fixup=None):
    """Every name whose value equals target, comma-joined for display.

    Meant for a 'best'/'worst' label where several entries share the same
    extreme value: naming all of them is more informative than an opaque
    'tie' placeholder.

    Args:
        name_to_value (dict or pd.Series): Name -> comparable value, e.g.
            already rounded to display precision so exact equality means
            "tied as shown".
        target: The value to match, typically the max or min of name_to_value.
        name_fixup (callable, optional): name -> display name, e.g. to
            relabel a '_simplex' suffix. Defaults to no change.

    Returns:
        str: Comma-joined, alphabetically sorted matching names.
    """
    name_fixup = name_fixup or (lambda s: s)
    return ', '.join(sorted(name_fixup(name) for name, val in name_to_value.items() if val == target))


def _set_baseline_text_color(styled):
    """Apply a baseline black-on-white to every cell, including headers.

    Call before layering any cell-specific coloring (background_gradient, a
    custom .apply) on top, which still overrides this per cell as needed.
    Pins both color AND background-color together, not color alone: setting
    only text color leaves the background to whatever the renderer defaults
    to (e.g. VS Code's dark-theme override, or a header row's own default
    shading), which can make black-on-black/dark text disappear entirely.
    Pinning both removes that dependency on the surrounding theme.

    Args:
        styled (pandas.io.formats.style.Styler): Table to color.

    Returns:
        pandas.io.formats.style.Styler
    """
    base = {'color': 'black', 'background-color': 'white'}
    return styled.set_properties(**base).set_table_styles(
        [{'selector': 'th', 'props': list(base.items())}])


def _readable_text_color(rgb):
    """'white' or 'black', whichever reads better against an RGB(A) background.

    Args:
        rgb (tuple[float]): An (r, g, b) or (r, g, b, a) triple in [0, 1], e.g.
            from a matplotlib colormap.

    Returns:
        str: 'white' for a dark background, 'black' for a light one, by
            ITU-R BT.601 luma (a standard, cheap perceptual-brightness proxy).
    """
    r, g, b = rgb[:3]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if luma < 0.5 else 'black'


def style_by_frequency(df, green_cols=(), red_cols=(), cmap_green='Greens', cmap_red='Reds', full_df=None):
    """Shade categorical columns by how often each value recurs in that column.

    The most frequent value in a column gets the darkest shade; less-frequent
    values get lighter shades of the same color. Meant for summary-table
    columns like 'best_sampler'/'fastest_sampler' (green_cols) or
    'worst_sampler'/'slowest_sampler' (red_cols), where a sampler winning
    repeatedly across rows is the more notable signal.

    Args:
        df (pd.DataFrame): Table to style.
        green_cols (tuple[str]): Columns to shade green.
        red_cols (tuple[str]): Columns to shade red.
        cmap_green (str): Matplotlib colormap name for green_cols.
        cmap_red (str): Matplotlib colormap name for red_cols.
        full_df (pd.DataFrame, optional): Compute each column's value counts
            from this instead of df, e.g. the full, unfiltered table df is a
            filtered subset of, so a value's shade stays calibrated to the
            whole column instead of recomputing from whichever rows the
            current filter selection leaves in. Defaults to df itself.

    Returns:
        pandas.io.formats.style.Styler: The styled table.
    """
    reference = full_df if full_df is not None else df

    def shade(series, cmap_name, ref_series):
        counts = ref_series.value_counts()
        cmap = plt.get_cmap(cmap_name)
        styles = []
        for v in series:
            rgb = cmap(0.25 + 0.65 * counts.get(v, 0) / counts.max())
            styles.append(f'background-color: {mcolors.to_hex(rgb)}; color: {_readable_text_color(rgb)}; '
                           'font-weight: bold')
        return styles

    # shade() below overrides the baseline per cell for the columns it does style.
    styled = _set_baseline_text_color(df.style)
    for col in green_cols:
        styled = styled.apply(lambda s: shade(s, cmap_green, reference[col]), subset=[col])
    for col in red_cols:
        styled = styled.apply(lambda s: shade(s, cmap_red, reference[col]), subset=[col])
    return styled


def style_by_value(styled, cols, center=1.0, cmap='RdYlGn'):
    """Shade numeric columns on a diverging scale centered at a reference value.

    Meant for columns like 'best_vs_sp500'/'worst_vs_sp500', where the
    notable signal is how far a value sits from a reference point (1.0x,
    tied with the benchmark), in either direction, not how often a value
    recurs (see style_by_frequency for that case). Chainable with it: pass
    its returned Styler back in to layer both shadings onto the same table.

    Args:
        styled (pd.DataFrame or pandas.io.formats.style.Styler): Table to
            shade, or an already-styled table (e.g. from style_by_frequency)
            to layer this shading onto.
        cols (list[str]): Numeric columns to shade.
        center (float): Value treated as neutral (cmap's midpoint); values
            above are shaded toward cmap's high end, values below toward its
            low end, scaled by distance from center. The single largest
            distance seen across cols sets the scale, so every shaded
            column reads on the same intensity scale.
        cmap (str): Diverging matplotlib colormap name, low end for values
            below center, high end for values above (RdYlGn: red below,
            green above).

    Returns:
        pandas.io.formats.style.Styler
    """
    is_styler = isinstance(styled, Styler)
    df = styled.data if is_styler else styled
    spread = max((df[list(cols)] - center).abs().max().max(), 1e-9)
    cmap_obj = plt.get_cmap(cmap)

    def shade(series):
        styles = []
        for v in series:
            rgb = cmap_obj(0.5 + 0.5 * np.clip((v - center) / spread, -1, 1))
            styles.append(f'background-color: {mcolors.to_hex(rgb)}; color: {_readable_text_color(rgb)}; '
                           'font-weight: bold')
        return styles

    # See the matching comment in style_by_frequency: only needed on the plain-df branch,
    # since a Styler passed in already got this baseline from whatever built it.
    out = styled if is_styler else _set_baseline_text_color(styled.style)
    for col in cols:
        out = out.apply(shade, subset=[col])
    return out


def display_filtered(df, filter_col, style_fn, default=None, **style_kwargs):
    """Interactive, single-column categorical filter with a pluggable styling function.

    For tables whose notable columns are categorical (e.g. 'best_sampler',
    shaded by style_by_frequency) rather than the color-graded numeric
    columns display_filterable_table is built around; use that one instead
    when value_cols' min-to-max gradient is the right styling.

    Uses a plain display(styled, display_id=True) call so the table is a
    normal, statically-visible cell output even without a live kernel,
    updated in place via .update() when the filter control changes.

    Args:
        df (pd.DataFrame): Results table to display.
        filter_col (str): Categorical column to filter by (e.g. 'n_port').
        style_fn (callable): df -> Styler (e.g. style_by_frequency, or a
            lambda chaining several stylers), applied to the filtered subset
            on every update.
        default: Initial filter_col selection, a single value or a
            list/tuple/set of values. Defaults to every value in filter_col
            (no filtering).
        **style_kwargs: Passed through to style_fn.

    Returns:
        None (renders the control and table as a side effect).
    """
    options = sorted(df[filter_col].unique())
    if default is None:
        initial = tuple(options)
    else:
        requested = tuple(default) if isinstance(default, (list, tuple, set)) else (default,)
        # Falls back to every option if none of the requested default(s) are actually present,
        # e.g. a booktest's shrunk-for-speed replacement of the real sweep values: an unmatched
        # default should degrade to "show everything," not raise/crash.
        initial = tuple(v for v in requested if v in options) or tuple(options)
    control = widgets.SelectMultiple(
        description=filter_col.capitalize(), options=options, value=initial,
        rows=min(8, len(options)),
    )
    table_handle = {'id': None}
    # If style_fn accepts a full_df kwarg (e.g. style_by_frequency), pass the whole,
    # unfiltered df so its shading stays calibrated to every record, not just whichever
    # rows the current filter selection leaves in.
    style_fn_wants_full_df = 'full_df' in inspect.signature(style_fn).parameters

    def update(_=None):
        filtered = df[df[filter_col].isin(control.value)]
        extra = {'full_df': df} if style_fn_wants_full_df else {}
        styled = style_fn(filtered, **extra, **style_kwargs)
        if table_handle['id'] is None:
            table_handle['id'] = display(styled, display_id=True)
        else:
            table_handle['id'].update(styled)

    control.observe(update, names='value')
    display(widgets.VBox([control]))
    update()
