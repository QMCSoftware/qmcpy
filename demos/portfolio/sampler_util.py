"""Sampler-comparison utilities for the portfolio allocation demo: generating
simplex-transformed portfolio weights from several low-discrepancy/IID
samplers, and evaluating/timing them against each other.

Independent of the notebook's plotting code (see pa_util.py); calls into
backtest_util.py for the core backtest mechanics. portfolio_allocation_demo.ipynb
imports from this module rather than defining its own copies.
"""
import itertools
import time
import timeit

import numpy as np
import pandas as pd
import qmcpy as qp

import backtest_util as bu
import config as cf

# Excludes Hammersley (no randomization) and MPMC (trains a model per call, too slow here).
sampler_classes = {
    'lattice': qp.Lattice,
    'sobol': qp.Sobol,
    'halton': qp.Halton,
    'iid': qp.IIDStdUniform,
    'faure': qp.Faure,
    'korobov': qp.KorobovLattice,
    'kronecker': qp.Kronecker,
}


def sampler_label(sampler_type, transform='root'):
    """Display label for a sampler type, e.g. 'sobol_simplex' -> 'Sobol (Root)'.

    Args:
        sampler_type (str): One of sampler_classes' keys, e.g. 'sobol', optionally
            suffixed with '_simplex' (e.g. 'sobol_simplex').
        transform (str): Active SimplexUniform transform_method name, shown for
            a '_simplex'-suffixed sampler_type.

    Returns:
        str: Human-readable label.
    """
    base, _, suffix = sampler_type.partition('_')
    acronyms = {'iid': 'IID'}
    label = acronyms.get(base, base.capitalize())
    return f'{label} ({transform.capitalize()})' if suffix else label


def gen_weights_reps(sampler_type, n_tickers, n_ports, replications=1, seed=42,
                      cube_points=None, transform='root'):
    """Generate portfolio weights with replications using the specified sampler.

    Args:
        sampler_type (str): Any key of sampler_classes (e.g. 'sobol', 'faure',
            'korobov', 'kronecker', ...), optionally suffixed with
            '_simplex' (e.g. 'sobol_simplex'); the suffix is accepted but
            has no effect on the result, every sampler maps onto the
            weight simplex via a simplex transform.
        n_tickers (int): Number of assets (dimension).
        n_ports (int): Number of portfolios per replication.
        replications (int): Number of replications.
        seed (int): Random seed for reproducibility.
        cube_points (ndarray, optional): Shape (replications, n_ports,
            n_tickers - 1): a sampler built at n_tickers - 1 dimensions,
            since the simplex transform only consumes that many coordinates
            (the n_tickers-th weight is implicit, see Returns). Reuses these
            cube points instead of drawing new ones.
        transform (str): One of SimplexUniform's transform_method names
            ('root', 'sort', 'shift', 'origami').

    Returns:
        ndarray: Shape (replications, n_ports, n_tickers), rows summing to 1.
    """
    base_type = sampler_type[:-len('_simplex')] if sampler_type.endswith('_simplex') else sampler_type
    if base_type not in sampler_classes:
        raise ValueError(
            f"Unknown sampler type: {sampler_type}. Must be one of {list(sampler_classes.keys())}, "
            "optionally suffixed with '_simplex'"
        )

    n_dim = n_tickers - 1
    if cube_points is None:
        sampler = sampler_classes[base_type](dimension=n_dim, replications=replications, seed=seed)
        cube_points = sampler.gen_samples(n_ports)

    # SimplexUniform.transform_points maps pre-generated cube points directly, so a shared
    # cube_points array (e.g. from generate_sampler_results) can feed it without re-sampling.
    # Sliced (not just asserted) because a correctly-built sampler already has exactly n_dim
    # columns; this only guards a caller that over-provisions cube_points by mistake.
    simplex_points = cube_points[..., :n_dim]
    weights_d = qp.SimplexUniform.transform_points(simplex_points, transform_method=transform)
    return np.concatenate([weights_d, 1 - weights_d.sum(axis=-1, keepdims=True)], axis=-1)


def evaluate_sampler_sharpe_reps(sampler_type, log_ret, n_ports, replications, log_rf=None, transform='root'):
    """Evaluate one sampler (replicated) across several portfolio counts.

    Args:
        sampler_type (str): Sampler key, e.g. 'sobol' or 'sobol_simplex'.
        log_ret (pd.DataFrame): Daily log returns, one column per ticker.
        n_ports (list[int]): Portfolio counts to evaluate.
        replications (int): Number of replications.
        log_rf (pd.Series, optional): Annualized risk-free log-rate series,
            date-matched inside bu.sharpe_reps.
        transform (str): SimplexUniform transform_method name, forwarded to
            gen_weights_reps.

    Returns:
        pd.DataFrame: One row per portfolio count, with Sharpe ratios and their
            standard error (SE) across replications at each risk level.
    """
    n_tickers = log_ret.shape[1]
    rows = []

    for ports in n_ports:
        weights = gen_weights_reps(sampler_type, n_tickers, ports, replications, transform=transform)
        sr = bu.sharpe_reps(weights, log_ret, log_rf, trading_days_per_year=cf.trading_days_per_year)

        rows.append({
            'sampler': sampler_type,
            'number of tickers': n_tickers,
            'number of portfolios': ports,
            'replications': replications,
            'low risk Sharpe': sr['low risk Sharpe'],
            'medium risk Sharpe': sr['medium risk Sharpe'],
            'high risk Sharpe': sr['high risk Sharpe'],
            'low risk Sharpe SE': sr['low risk Sharpe SE'],
            'medium risk Sharpe SE': sr['medium risk Sharpe SE'],
            'high risk Sharpe SE': sr['high risk Sharpe SE'],
        })

    return pd.DataFrame(rows)


def evaluate_all_samplers(lr, n_ports, sampler_types, log_rf=None, replications=30, transform='root'):
    """Evaluate all samplers (replicated) for given log returns.

    Args:
        lr (pd.DataFrame): Daily log returns, one column per ticker.
        n_ports (list[int]): Portfolio counts to evaluate.
        sampler_types (list[str]): Samplers to evaluate, e.g. the caller's
            own canonical sampler order.
        log_rf (pd.Series, optional): Annualized risk-free log-rate series,
            date-matched inside bu.sharpe_reps.
        replications (int): Number of replications per sampler.
        transform (str): SimplexUniform transform_method name, forwarded to
            gen_weights_reps.

    Returns:
        pd.DataFrame: One row per (sampler, portfolio count).
    """
    dfs = [evaluate_sampler_sharpe_reps(st, lr, n_ports, replications, log_rf, transform) for st in sampler_types]
    # Sorted by portfolio count first: comparing different n_ports values caused a real overclaim earlier in this notebook's history.
    return pd.concat(dfs, ignore_index=True).sort_values(['number of portfolios', 'sampler'], ignore_index=True)


def _measure_one(sampler, d, n, repeats, transform='root'):
    """Time gen_weights_reps(sampler, d, n), averaged over `repeats` calls.

    Args:
        sampler (str): Sampler key, e.g. 'sobol' or 'sobol_simplex'.
        d (int): Dimension (ticker count) to pass to gen_weights_reps.
        n (int): Portfolio count to pass to gen_weights_reps.
        repeats (int): Number of calls to average over.
        transform (str): SimplexUniform transform_method name, forwarded to
            gen_weights_reps.

    Returns:
        tuple[float, float]: (real_seconds, cpu_seconds) per call, or
            (nan, nan) if gen_weights_reps raised.
    """
    try:
        start_cpu = time.process_time()
        t_real = timeit.timeit(lambda: gen_weights_reps(sampler, d, n, transform=transform), number=repeats) / repeats
        t_cpu = (time.process_time() - start_cpu) / repeats
        return t_real, t_cpu
    except Exception as e:
        print(f"Skipping sampler={sampler} d={d} n={n}: {e}")
        return float('nan'), float('nan')


def measure_runtime(sampler_types, transform='root', dimensions=None, num_samples=None,
                     fixed_dimension=50, fixed_num_samples=2**12, repeats=3):
    """Measure runtime for all sampling methods.

    Args:
        sampler_types (list[str]): Samplers to measure, e.g. the caller's own
            canonical sampler order.
        transform (str): SimplexUniform transform_method name, forwarded to
            gen_weights_reps.
        dimensions (list[int], optional): Ticker counts to sweep at
            fixed_num_samples. Defaults to [5, 10, 20, 50, 100, 200] (dropped
            500, 1000: dominated runtime for little added signal). Exposed as
            a parameter (not just a local) so a caller needing a fast,
            reduced sweep (e.g. a CI booktest) can override it without
            patching this function's source.
        num_samples (list[int], optional): Portfolio counts to sweep at
            fixed_dimension. Defaults to [2**8, ..., 2**14] (was up to
            2**17: slow, and OOM-risked at d=50).
        fixed_dimension (int): Ticker count held fixed while sweeping num_samples.
        fixed_num_samples (int): Portfolio count held fixed while sweeping dimensions.
        repeats (int): Calls to average over per (sampler, size) pair.

    Returns:
        pd.DataFrame: Real and CPU runtime, varying either ticker count or
            portfolio count with the other held fixed, for every sampler in
            sampler_types.
    """
    dimensions = dimensions if dimensions is not None else [5, 10, 20, 50, 100, 200]
    num_samples = num_samples if num_samples is not None else [2**m for m in range(8, 15)]

    results = []
    for sampler, d in itertools.product(sampler_types, dimensions):
        t_real, t_cpu = _measure_one(sampler, d, fixed_num_samples, repeats, transform)
        results.append({
            'Series': 'Tickers', 'Sampler': sampler, 'Tickers': d,
            'Portfolios': fixed_num_samples, 'Runtime_real': t_real, 'Runtime_CPU': t_cpu,
        })
    for sampler, n in itertools.product(sampler_types, num_samples):
        t_real, t_cpu = _measure_one(sampler, fixed_dimension, n, repeats, transform)
        results.append({
            'Series': 'Portfolios', 'Sampler': sampler, 'Tickers': fixed_dimension,
            'Portfolios': n, 'Runtime_real': t_real, 'Runtime_CPU': t_cpu,
        })
    return pd.DataFrame(results)


def generate_sampler_results(n_tickers, num_ports, replications, log_ret, sampler_types,
                              transform='root', return_timing=False):
    """Generate simplex-transformed results for each base sampler, from one shared
    draw of cube points.

    Args:
        n_tickers (int): Number of assets (dimension).
        num_ports (int): Number of portfolios per replication.
        replications (int): Number of replications.
        log_ret (pd.DataFrame): Daily log returns, one column per ticker.
        sampler_types (list[str]): The caller's own canonical sampler order;
            determines both which samplers are returned and the returned
            dict's key order.
        transform (str): SimplexUniform transform_method name, forwarded to
            gen_weights_reps.
        return_timing (bool): If True, also return per-base-sampler wall time.

    Returns:
        dict, or (dict, dict) if return_timing: bu.sharpe_reps() output for every
            sampler in sampler_types, and (if requested) wall-clock seconds
            per base sampler (gen_samples plus its gen_weights_reps call).
    """
    results = {}
    timing = {}
    # Only the base samplers actually requested (deduplicated, order-preserving): sampler_types
    # may be a reduced subset for a CI booktest, and computing the rest would waste runtime
    # generating/scoring samplers the caller deliberately excluded.
    requested_base_types = dict.fromkeys(
        st[:-len('_simplex')] if st.endswith('_simplex') else st for st in sampler_types
    )
    for base_type in requested_base_types:
        sampler_class = sampler_classes[base_type]
        t0 = time.perf_counter()
        # dimension=n_tickers - 1: the simplex transform only consumes that many
        # coordinates (see gen_weights_reps). Building at n_tickers and discarding the
        # last coordinate would both waste work and, for dimension-dependent
        # constructions (Faure's prime base, Korobov/Kronecker's generating vector),
        # silently benchmark/score a different sequence than a correctly-sized sampler.
        cube_points = sampler_class(
            dimension=n_tickers - 1, replications=replications, seed=42
        ).gen_samples(num_ports)
        sampler = f'{base_type}_simplex'
        weights = gen_weights_reps(
            sampler, n_tickers, num_ports, replications, cube_points=cube_points, transform=transform
        )
        results[sampler] = bu.sharpe_reps(weights, log_ret, trading_days_per_year=cf.trading_days_per_year)
        timing[base_type] = time.perf_counter() - t0
    ordered = {sampler: results[sampler] for sampler in sampler_types}
    return (ordered, timing) if return_timing else ordered


def run_backtest_case(n_tickers, sample_type, section4_data, sampler_types, num_ports=2**14,
                       principal=10000, transform='root', in_sample_replications=5, oos_replications=50):
    """Run one Section 4 backtest case (ticker count x in-sample/OOS) end to end.

    Args:
        n_tickers (int): Number of assets; one of 4, 10, 20, 40.
        sample_type (str): 'in-sample' (fit and evaluate on the full period)
            or 'OOS' (fit on [cf.start_date, cf.train_end_date], evaluate on
            [cf.test_start_date, cf.end_date]).
        section4_data (dict): n_tickers -> (tickers, log_returns), the inputs
            this ticker count's backtest needs.
        sampler_types (list[str]): The caller's own canonical sampler order,
            forwarded to generate_sampler_results.
        num_ports (int): Portfolios per replication.
        principal (float): Dollar amount invested.
        transform (str): SimplexUniform transform_method name, forwarded to
            generate_sampler_results.
        in_sample_replications (int): Replications used when sample_type is
            'in-sample'. Exposed as a parameter (not just a local) so a
            caller needing fewer replications (e.g. a CI booktest) can
            override it without patching this function's source.
        oos_replications (int): Replications used when sample_type is 'OOS'.

    Returns:
        tuple[dict, dict, dict]: (bu.compute_all_portfolios() output, sr_dict,
            per-base-sampler timing from generate_sampler_results), ready to
            plot or summarize.
    """
    tickers_n, lr_n = section4_data[n_tickers]
    lr_n = lr_n.iloc[:, :len(tickers_n)]  # keep lr_n's columns matching tickers_n's length
    df = pd.read_csv(
        f"data/df40_{cf.start_date}_to_{cf.end_date}.csv.gz",
        parse_dates=['Date'],
    )
    df = df[df['Ticker'].isin(tickers_n)]

    if sample_type == 'in-sample':
        replications = in_sample_replications
        # Begin at the first price date shared by all assets, preceding the first
        # common return. Do not value weights fitted after a late IPO back to 2014.
        common_start = df.groupby('Ticker')['Date'].min().loc[tickers_n].max()
        price_df = df[(df['Date'] >= common_start) & (df['Date'] <= lr_n.index.max())]
        log_ret = lr_n
        # Quarterly rebalancing, renormalizing over only the tickers already listed at
        # each rebalance date
        rebalance_freq = 'QS'
    elif sample_type == 'OOS':
        replications = oos_replications
        price_df = df[(df['Date'] >= cf.test_start_date) & (df['Date'] <= cf.end_date)]
        log_ret = lr_n.loc[cf.start_date:cf.train_end_date]
        rebalance_freq = None
    else:
        raise ValueError(f"Unknown sample_type: {sample_type!r}; must be 'in-sample' or 'OOS'")

    if log_ret.empty:
        raise ValueError("no returns available in the fitting window")
    print(f"{n_tickers} assets, {sample_type}: fit {log_ret.index.min().date()} to "
          f"{log_ret.index.max().date()} ({len(log_ret)} rows); "
          f"valuation {price_df['Date'].min().date()} to {price_df['Date'].max().date()}")

    stocks_dict = bu.setup_stock_dfs(price_df, tickers_n)
    stocks = tuple(stocks_dict[t] for t in log_ret.columns)
    sr_dict, timing = generate_sampler_results(n_tickers, num_ports, replications, log_ret, sampler_types,
                                                transform=transform, return_timing=True)
    all_portfolios_dict = bu.compute_all_portfolios(stocks, sr_dict, ['low', 'medium', 'high'], principal, rebalance_freq=rebalance_freq)
    return all_portfolios_dict, sr_dict, timing


def sp500_benchmark(sample_type, principal, sp500, dates=None):
    """Normalize S&P 500 to a $principal benchmark over the same date range as run_backtest_case.

    Args:
        sample_type (str): 'in-sample' or 'OOS', matching run_backtest_case's date range for each.
        principal (float): Dollar amount invested, to match the portfolios it is compared against.
        sp500 (pd.Series): Daily S&P 500 price level, indexed by Date.
        dates (pd.DatetimeIndex, optional): Portfolio valuation dates. When
            supplied, match these dates exactly, including a common-history start.

    Returns:
        pd.Series: S&P 500 value, starting at $principal on the first date in that range.
    """
    lo, hi = (cf.test_start_date, cf.end_date) if sample_type == 'OOS' else (cf.start_date, cf.end_date)
    s = sp500.loc[lo:hi]
    if dates is not None:
        s = s.reindex(dates)
    if s.empty or s.isna().any():
        raise ValueError("benchmark prices must cover every valuation date")
    return principal * s / s.iloc[0]
