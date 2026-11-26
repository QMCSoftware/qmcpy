"""Core backtest computation for the portfolio allocation demo.

Covers point-in-time universe construction and periodic/event-driven
rebalancing, independent of the notebook's plotting (see pa_util.py) and
sampler-comparison (see sampler_util.py) code. Extracted so these functions
can be unit tested directly (test/test_tm_demo_portfolio.py), rather than
only through the slow, whole-notebook booktest
(test/booktests/tb_portfolio_allocation_demo.py).

portfolio_allocation_demo.ipynb imports from this module rather than
defining its own copies. Handles several edge cases beyond plain buy-and-
hold: a late listing, a bankruptcy/permanent delisting, a temporary trading
halt, and a stop-loss.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def load_assets(path):
    """Return ticker symbols and company labels in their saved order.

    Args:
        path (str): CSV path with 'Ticker' and 'Company' columns.

    Returns:
        tuple[list[str], list[str]]: Ticker symbols and company labels.
    """
    assets = pd.read_csv(path, usecols=["Ticker", "Company"]).drop_duplicates()
    return assets["Ticker"].tolist(), assets["Company"].tolist()


def log_returns(df):
    """Return daily log returns, dropping any date with a missing price.

    Args:
        df (pd.DataFrame): Price data with 'Ticker', 'Date', and
            'Adj Close Price' columns.

    Returns:
        pd.DataFrame: Log returns indexed by Date, one column per ticker.
    """
    stocks = df.pivot(index="Date", columns="Ticker", values="Adj Close Price")
    return np.log(stocks / stocks.shift(1)).dropna()


def load_log_returns(path, price_path):
    """Read cached log returns, first rebuilding the cache from prices if missing.

    Args:
        path (str): Log-return CSV path (gzip-compressed if it ends in .gz).
        price_path (str): Price CSV path, as accepted by log_returns.

    Returns:
        pd.DataFrame: Log returns indexed by Date, one column per ticker.
    """
    if not Path(path).exists():
        prices = pd.read_csv(price_path, parse_dates=["Date"])
        log_returns(prices).to_csv(path, compression={"method": "infer", "mtime": 0})
    return pd.read_csv(path, index_col=0, parse_dates=True)


def setup_stock_dfs(df, tickers):
    """Convert ticker data to indexed stock dataframes with normalized returns.

    Args:
        df (pd.DataFrame): Raw price data with 'Ticker', 'Date', and
            'Adj Close Price' columns.
        tickers (list[str]): Ticker symbols to extract.

    Returns:
        dict[str, pd.DataFrame]: Per-ticker DataFrame indexed by Date, each with
            a 'Norm Return' column normalized to 1 at the first date.
    """
    dfs = {}
    for ticker in tickers:
        stock_df = df[df['Ticker'] == ticker].reset_index().set_index('Date')
        stock_df['Norm Return'] = stock_df['Adj Close Price'] / stock_df.iloc[0]['Adj Close Price']
        dfs[ticker] = stock_df
    return dfs


def listing_change_dates(stock_dfs):
    """Dates where the listed-ticker set changes: a first listing, a trading
    halt opening (delisting or bankruptcy), or a halt closing (relisting).

    Args:
        stock_dfs (list[pd.DataFrame]): Per-stock DataFrames indexed by Date.

    Returns:
        list[pd.Timestamp]: Dates on which some ticker enters or leaves the
            listed set, relative to the union calendar of all stock_dfs's own
            trading days, so a weekend or holiday shared by every ticker
            never counts as a gap.
    """
    all_dates = sorted(set().union(*(df.index for df in stock_dfs)))
    D = len(stock_dfs)
    idx_sets = [set(df.index) for df in stock_dfs]
    changes = []
    prev = frozenset()
    for d in all_dates:
        cur = frozenset(i for i in range(D) if d in idx_sets[i])
        if cur != prev:
            changes.append(d)
            prev = cur
    return changes


def stop_loss_dates(stock_dfs, stop_loss_drop_pct=None, stop_loss_price_floor=None):
    """First date each ticker triggers a stop-loss exit.

    Args:
        stock_dfs (list[pd.DataFrame]): Per-stock DataFrames with an
            'Adj Close Price' column.
        stop_loss_drop_pct (float, optional): Trailing-drawdown fraction
            (e.g. 0.05 for 5%) below a ticker's own running peak price since
            its listing date that triggers an exit.
        stop_loss_price_floor (float or sequence[float], optional): An
            absolute price level that triggers an exit, as a scalar applied
            to every ticker or a sequence with one value per ticker. None
            entries (in a sequence) disable the floor for that ticker.

    Returns:
        dict[int, pd.Timestamp]: Maps a stock_dfs index to the first date
            either condition fires; a ticker that never triggers is absent.
    """
    triggers = {}
    D = len(stock_dfs)
    if stop_loss_price_floor is None or np.isscalar(stop_loss_price_floor):
        floors = [stop_loss_price_floor] * D
    else:
        floors = list(stop_loss_price_floor)
        if len(floors) != D:
            raise ValueError(f"stop_loss_price_floor has {len(floors)} entries, expected one per ticker ({D})")
    for i, df in enumerate(stock_dfs):
        price = df['Adj Close Price']
        fire_dates = []
        if stop_loss_drop_pct is not None:
            fired = price < price.cummax() * (1 - stop_loss_drop_pct)
            if fired.any():
                fire_dates.append(price.index[fired.to_numpy().argmax()])
        if floors[i] is not None:
            fired = price < floors[i]
            if fired.any():
                fire_dates.append(price.index[fired.to_numpy().argmax()])
        if fire_dates:
            triggers[i] = min(fire_dates)
    return triggers


def compute_portfolio_value_reps(stock_dfs, weights_reps, principal, rebalance_freq=None,
                                  rebalance_on_universe_change=False,
                                  stop_loss_drop_pct=None, stop_loss_price_floor=None):
    """Compute portfolio value over time for each replication, given per-stock allocations.

    Args:
        stock_dfs (list[pd.DataFrame]): Per-stock DataFrames with 'Adj Close Price'
            and 'Norm Return' columns, in the same order as weights_reps' last
            axis. A stock's index may start late (a listing), end early (a
            delisting or bankruptcy), or have an internal gap (a trading halt).
        weights_reps (ndarray): Shape (R, D) portfolio weights, one row per
            replication. A row may be all-NaN (e.g. a sharpe_reps tier with
            no eligible candidate for that replication), propagated as a
            NaN value rather than a fabricated $0; a row that is only
            partially NaN raises ValueError.
        principal (float): Dollar amount invested.
        rebalance_freq (str, optional): A pandas date_range frequency alias (e.g.
            'QS') to periodically rebalance at: only listed, non-exited tickers'
            weights are renormalized and redistributed; a late-listed ticker's
            allocation remains in cash until a rebalance on or after listing,
            while an absent holding freezes at its last traded price and resumes
            compounding if it returns. If eligible target weights sum to zero,
            the available pool stays in cash. None
            (default): buy and hold, reserving each late-listed ticker's initial
            allocation as cash until listing and freezing missing quotes at the
            last traded value.
        rebalance_on_universe_change (bool): If True, also rebalance the day a
            ticker's listed status changes, instead of waiting for the next
            rebalance_freq date. Default False: no effect on this demo's
            backtests, since no current ticker has an internal gap.
        stop_loss_drop_pct (float, optional): Sell a ticker (see
            stop_loss_dates) the first day its price falls more than this
            fraction below its own running peak; proceeds are redistributed to
            the remaining tickers in proportion to their target weights, or held
            as cash if their total target weight is zero. The sold ticker never
            re-enters. None
            (default): disabled.
        stop_loss_price_floor (float or sequence[float], optional): Same as
            stop_loss_drop_pct, triggered by an absolute price instead of a
            drawdown; scalar or one value per ticker. Combinable with
            stop_loss_drop_pct; whichever fires first wins.

    Returns:
        pd.DataFrame: Portfolio value over time, one column per replication.
    """
    R, D = weights_reps.shape
    if len(stock_dfs) != D:
        raise ValueError(f"expected {D} stock series, got {len(stock_dfs)}")

    nan_counts = np.isnan(weights_reps).sum(axis=1)
    if np.any((nan_counts > 0) & (nan_counts < D)):
        raise ValueError(
            "weights_reps has a replication with some but not all weights NaN; "
            "a replication must be either fully specified or fully NaN (e.g. a "
            "sharpe_reps tier with no eligible candidate for that replication)"
        )

    all_dates = sorted(set().union(*(df.index for df in stock_dfs)))
    if (rebalance_freq is None and not rebalance_on_universe_change
            and stop_loss_drop_pct is None and stop_loss_price_floor is None):
        portfolios = []
        for r in range(R):
            positions = [
                stock_df['Norm Return'].reindex(all_dates).ffill().fillna(1.0) * alloc * principal
                for stock_df, alloc in zip(stock_dfs, weights_reps[r])
            ]
            # min_count=1 preserves a fully-NaN replication (validated above) as NaN;
            # pandas' default sum-of-all-NaN is 0, which would otherwise fabricate a
            # $0 portfolio for a replication where no tier candidate was eligible.
            portfolios.append(pd.concat(positions, axis=1).sum(axis=1, min_count=1))
        return pd.concat(portfolios, axis=1)

    exit_dates = stop_loss_dates(stock_dfs, stop_loss_drop_pct, stop_loss_price_floor)
    triggers = {all_dates[0]}
    if rebalance_on_universe_change:
        triggers.update(listing_change_dates(stock_dfs))
    triggers.update(exit_dates.values())
    if rebalance_freq is not None:
        wanted = pd.date_range(all_dates[0], all_dates[-1], freq=rebalance_freq)
        triggers.update(next((d for d in all_dates if d >= w), all_dates[-1]) for w in wanted)
    rebalance_dates = sorted(triggers)

    portfolios = []
    for r in range(R):
        w_target = weights_reps[r]
        value = pd.Series(index=all_dates, dtype=float)
        balances = np.zeros(D)  # per-ticker dollar sub-balance; 0 until first listed
        reserved_cash = np.zeros(D)  # initial allocations awaiting a late listing
        exited = np.zeros(D, dtype=bool)  # permanently sold via a stop-loss
        cash = 0.0  # available pool with no eligible positive target weight
        last_close = None
        for k, t0 in enumerate(rebalance_dates):
            t1 = rebalance_dates[k + 1] if k + 1 < len(rebalance_dates) else None
            period_dates = [d for d in all_dates if d >= t0 and (t1 is None or d < t1)]
            if not period_dates:
                continue
            listed_now = np.array([t0 in stock_dfs[i].index for i in range(D)])
            if last_close is not None:
                # Mark every held, still-listed ticker's balance to market from the
                # previous period's last close to t0: balances[eligible_idx] below was set
                # at that last close, and rel resets to 1 at t0, so without this the day's
                # (or gap's) own price move is silently dropped, understating the portfolio.
                for i in np.flatnonzero((balances != 0) & listed_now):
                    price_i = stock_dfs[i]['Adj Close Price']
                    price_prev = price_i.loc[:last_close]
                    if len(price_prev):
                        balances[i] *= price_i.loc[t0] / price_prev.iloc[-1]
            just_triggered = np.array([(i in exit_dates) and (exit_dates[i] <= t0) for i in range(D)]) & ~exited
            exited = exited | just_triggered
            eligible_idx = np.flatnonzero(listed_now & ~exited)
            frozen_idx = np.flatnonzero((~listed_now) & (~exited))
            initial_pool = principal if k == 0 else 0.0
            if k == 0:
                reserved_cash[frozen_idx] = principal * w_target[frozen_idx]
                initial_pool -= reserved_cash.sum()
            released_cash = reserved_cash[listed_now].sum()
            reserved_cash[listed_now] = 0.0
            frozen_total = balances[frozen_idx].sum() + reserved_cash.sum()
            tradeable_pool = (balances[eligible_idx].sum() + balances[just_triggered].sum()
                               + released_cash + cash + initial_pool)
            cash = 0.0
            w = w_target[eligible_idx]
            if w.sum() == 0:
                # No eligible target mass (including an empty universe): hold cash.
                cash = tradeable_pool
                value.loc[period_dates] = frozen_total + cash
                balances[eligible_idx] = 0.0
                balances[np.flatnonzero(exited)] = 0.0
                last_close = period_dates[-1]
                continue
            w = w / w.sum()
            rel = np.zeros((len(eligible_idx), len(period_dates)))
            for j, i in enumerate(eligible_idx):
                price = stock_dfs[i]['Adj Close Price'].reindex(period_dates).ffill()
                rel[j] = price.to_numpy() / price.iloc[0]
            period_value = frozen_total + (w @ rel) * tradeable_pool
            value.loc[period_dates] = period_value
            balances[eligible_idx] = w * tradeable_pool * rel[:, -1]
            balances[np.flatnonzero(exited)] = 0.0
            last_close = period_dates[-1]
            # frozen_idx (absent, not exited) is left untouched: frozen at its prior balance
        portfolios.append(value)
    return pd.concat(portfolios, axis=1)


def sharpe_reps(weights, log_ret, log_rf=None, trading_days_per_year=252):
    """Select portfolios by annualized constant-weight simple-return Sharpe.

    Log-return inputs are converted to simple returns before computing moments.
    The score describes daily constant weights; quarterly and buy-and-hold
    strategies have drifting weights and must be evaluated from realized returns.

    Args:
        weights (ndarray): Shape (R, P, D) portfolio weights.
        log_ret (pd.DataFrame): Daily log returns, one column per ticker.
        log_rf (pd.Series, optional): Annualized risk-free log-rate series
            (not daily), date-matched to log_ret (reindexed and forward-
            filled, not averaged first) and converted to a daily-equivalent
            internally, so the excess return used is contemporaneous to each
            return rather than a single rate blended across the whole
            window. If None, the raw Sharpe ratio (no risk-free rate) is
            used.
        trading_days_per_year (int): Annualization factor. Defaults to 252
            (not imported from config.py, so this module stays independently
            unit-testable); demo callers pass config.trading_days_per_year.

    Returns:
        dict: Per-risk-level selected weights ('low'/'medium'/'high', shape
            (R, D) each, NaN rows where that replication had no eligible
            candidate), their mean Sharpe ratios, the standard error (SE) of
            that mean, and the number of replications the mean/SE were
            computed from ('<level> risk valid replications'): fewer than R
            whenever that tier was empty for some replications, and NaN SE
            below 2 valid replications.
    """

    R, P, D = weights.shape

    simple_ret = np.expm1(log_ret)
    if log_rf is None:
        excess_ret = simple_ret
    else:
        # Convert the annualized log rate to a daily simple return in matching units.
        daily_rf = np.expm1(log_rf.reindex(log_ret.index).ffill() / trading_days_per_year)
        excess_ret = simple_ret.sub(daily_rf, axis=0)

    ret_arr = np.sum(weights * excess_ret.mean().values * trading_days_per_year, axis=2)

    vol_arr = np.sqrt(np.sum(weights @ (excess_ret.cov().values * trading_days_per_year) * weights, axis=2))

    sharpe_arr = ret_arr / vol_arr

    low_risk_tolerance = np.quantile(vol_arr, 1 / 3, axis=1, keepdims=True)
    medium_risk_tolerance = np.quantile(vol_arr, 2 / 3, axis=1, keepdims=True)
    low_mask = vol_arr <= low_risk_tolerance
    medium_mask = (vol_arr > low_risk_tolerance) & (vol_arr <= medium_risk_tolerance)
    high_mask = vol_arr > medium_risk_tolerance

    high_risk_idx = np.argmax(np.where(high_mask, sharpe_arr, -np.inf), axis=1)
    medium_risk_idx = np.argmax(np.where(medium_mask, sharpe_arr, -np.inf), axis=1)
    low_risk_idx = np.argmax(np.where(low_mask, sharpe_arr, -np.inf), axis=1)

    rows = np.arange(R)
    low_risk_max_sharpe = sharpe_arr[rows, low_risk_idx]
    medium_risk_max_sharpe = sharpe_arr[rows, medium_risk_idx]
    high_risk_max_sharpe = sharpe_arr[rows, high_risk_idx]

    # A tier can be empty for a given replication (e.g. tied volatilities collapse a
    # quantile boundary): argmax over an all -inf row still returns index 0, silently
    # mislabeling that candidate's weight/Sharpe as the tier's optimum. NaN those
    # replications out instead; nanmean/nanstd below then score from the remaining,
    # genuinely-populated replications rather than letting one bad draw poison the mean.
    # P == 1 is excluded: with a single candidate, medium/high are *always* structurally
    # empty (strict > against a quantile that equals the only value), by construction,
    # not from a tie among several candidates; every tier trivially holds that candidate.
    if P == 1:
        low_empty = medium_empty = high_empty = np.zeros(R, dtype=bool)
    else:
        low_empty, medium_empty, high_empty = (~low_mask.any(axis=1), ~medium_mask.any(axis=1), ~high_mask.any(axis=1))
    low_risk_max_sharpe = np.where(low_empty, np.nan, low_risk_max_sharpe)
    medium_risk_max_sharpe = np.where(medium_empty, np.nan, medium_risk_max_sharpe)
    high_risk_max_sharpe = np.where(high_empty, np.nan, high_risk_max_sharpe)
    low_weights = np.where(low_empty[:, None], np.nan, weights[rows, low_risk_idx])
    medium_weights = np.where(medium_empty[:, None], np.nan, weights[rows, medium_risk_idx])
    high_weights = np.where(high_empty[:, None], np.nan, weights[rows, high_risk_idx])

    def n_valid(arr):
        return int(np.count_nonzero(~np.isnan(arr)))

    def se(arr):
        # Divide by the number of replications actually contributing a finite score to
        # this tier (m), not R: when a tier is empty (NaN) for some replications, nanstd
        # already excludes them from the spread, so the denominator must match or the SE
        # is understated by a factor of sqrt(m/R). Undefined (NaN), not 0, below m=2.
        m = n_valid(arr)
        return np.round(np.nanstd(arr, ddof=1) / np.sqrt(m), 3) if m > 1 else np.nan

    with warnings.catch_warnings():
        # Expected, not a caller error: a tier that's empty in every replication (R == 1
        # and that tier's mask never fires) means nanmean/nanstd reduce an all-NaN slice,
        # which numpy warns on even though NaN is exactly the documented, intended result.
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
        result = {
            "number of tickers": D,
            "number of portfolios": P,
            "replications": R,

            "low": low_weights,
            "medium": medium_weights,
            "high": high_weights,

            "low risk Sharpe": np.round(np.nanmean(low_risk_max_sharpe), 3),
            "medium risk Sharpe": np.round(np.nanmean(medium_risk_max_sharpe), 3),
            "high risk Sharpe": np.round(np.nanmean(high_risk_max_sharpe), 3),

            "low risk Sharpe SE": se(low_risk_max_sharpe),
            "medium risk Sharpe SE": se(medium_risk_max_sharpe),
            "high risk Sharpe SE": se(high_risk_max_sharpe),

            "low risk valid replications": n_valid(low_risk_max_sharpe),
            "medium risk valid replications": n_valid(medium_risk_max_sharpe),
            "high risk valid replications": n_valid(high_risk_max_sharpe),
        }
    return result


def compute_all_portfolios(stocks, sr_dict, risk_levels, principal, rebalance_freq=None,
                            rebalance_on_universe_change=False,
                            stop_loss_drop_pct=None, stop_loss_price_floor=None):
    """Compute portfolio values for all samplers and risk levels.

    Args:
        stocks (list[pd.DataFrame]): Per-stock DataFrames, see
            compute_portfolio_value_reps.
        sr_dict (dict): Per-sampler sharpe_reps() output, keyed by sampler type.
        risk_levels (list[str]): Risk levels to compute, e.g. ['low', 'medium', 'high'].
        principal (float): Dollar amount invested.
        rebalance_freq (str, optional): Forwarded to compute_portfolio_value_reps.
        rebalance_on_universe_change (bool): Forwarded to compute_portfolio_value_reps.
        stop_loss_drop_pct (float, optional): Forwarded to compute_portfolio_value_reps.
        stop_loss_price_floor (float or sequence[float], optional): Forwarded
            to compute_portfolio_value_reps.

    Returns:
        dict: Nested {sampler: {risk_level: portfolio value DataFrame}}.
    """
    return {
        sampler: {
            risk: compute_portfolio_value_reps(
                stocks, sr[risk], principal, rebalance_freq=rebalance_freq,
                rebalance_on_universe_change=rebalance_on_universe_change,
                stop_loss_drop_pct=stop_loss_drop_pct, stop_loss_price_floor=stop_loss_price_floor,
            )
            for risk in risk_levels
        }
        for sampler, sr in sr_dict.items()
    }
