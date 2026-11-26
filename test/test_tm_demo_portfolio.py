"""Unit tests for the portfolio demo's backtest computation.

Covers demos/portfolio/backtest_util.py: point-in-time universe construction,
periodic/event-driven rebalancing, and Sharpe-ratio computation. Fast,
isolated regression guards for bugs found and fixed in this module: phantom
capital on a late listing, phantom liquidation of a temporarily absent
ticker, and a unit mismatch between an annualized risk-free rate and daily
returns, each previously caught only informally or not at all, as distinct
from the slow, whole-notebook check in
test/booktests/tb_portfolio_allocation_demo.py.

Classes:
    TestLoadLogReturns: log returns from prices, and the cache rebuilt from
        them when missing; no randomization.
    TestListingChangeDates: universe-membership-change detection; no randomization.
    TestStopLossDates: drawdown/floor trigger detection; no randomization.
    TestComputePortfolioValueReps: the backtest itself, from plain buy-and-
        hold up to combined rebalancing and stop-loss; all inputs are small,
        deterministic toy price series, no randomization.
    TestComputeAllPortfolios: forwarding across samplers/risk levels.
    TestSharpeReps: Sharpe-ratio computation, with and without a risk-free
        rate; no randomization.

Example:
    python3 -m pytest test/test_tm_demo_portfolio.py
"""
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

bu = pytest.importorskip("demos.portfolio.backtest_util")


class TestBacktestWindowsAndBenchmark(TestCase):
    """run_backtest_case's start-date selection and sp500_benchmark's date
    matching, across both in-sample and OOS sample types."""

    def setUp(self):
        """Import sampler_util the same way the notebook does (sys.path, not
        a package import), so patch.object on its module-level names works."""
        path = str(Path(__file__).resolve().parents[1] / "demos/portfolio")
        sys.path.insert(0, path)
        self.addCleanup(sys.path.remove, path)
        import sampler_util
        self.su = sampler_util

    def test_generates_only_requested_samplers_once(self):
        """A reduced search must never instantiate an excluded sampler."""
        su = self.su
        # dimension 2 = n_tickers - 1: the simplex transform only consumes n_tickers - 1
        # coordinates (the n_tickers-th weight is implicit).
        cube = np.random.default_rng(42).random((2, 8, 2))
        samplers = {name: Mock(return_value=Mock(gen_samples=Mock(return_value=cube)))
                    for name in ("sobol", "iid", "faure")}
        log_ret = pd.DataFrame([[.01, .02, -.01], [-.02, .01, .03], [.03, -.01, .02]])
        requested = ["sobol_simplex", "iid_simplex", "sobol_simplex"]
        with patch.object(su, "sampler_classes", samplers):
            results, timing = su.generate_sampler_results(
                3, 8, 2, log_ret, requested, return_timing=True)
        assert list(results) == ["sobol_simplex", "iid_simplex"]
        assert list(timing) == ["sobol", "iid"]
        for name in ("sobol", "iid"):
            samplers[name].assert_called_once_with(dimension=2, replications=2, seed=42)
            samplers[name].return_value.gen_samples.assert_called_once_with(8)
        samplers["faure"].assert_not_called()

    def test_gen_weights_reps_builds_sampler_at_n_tickers_minus_one(self):
        """The simplex transform consumes only n_tickers - 1 coordinates (the
        n_tickers-th weight is implicit); building the sampler at n_tickers and
        discarding a coordinate would waste work and, for a dimension-dependent
        construction (Faure's prime base, Korobov/Kronecker's generating
        vector), silently score a different sequence than a correctly-sized
        sampler would produce."""
        su = self.su
        cube = np.random.default_rng(0).random((1, 5, 2))
        mock_cls = Mock(return_value=Mock(gen_samples=Mock(return_value=cube)))
        with patch.object(su, "sampler_classes", {"sobol": mock_cls}):
            weights = su.gen_weights_reps("sobol", 3, 5)
        mock_cls.assert_called_once_with(dimension=2, replications=1, seed=42)
        mock_cls.return_value.gen_samples.assert_called_once_with(5)
        assert weights.shape == (1, 5, 3)

    def test_common_history_and_dated_benchmark(self):
        """F4: common-history valuation and dated benchmark; no future OOS fitting."""
        su = self.su
        dates = pd.bdate_range("2020-01-01", periods=8)
        tickers = list("ABCD")
        prices = pd.concat([
            pd.DataFrame({"Ticker": ticker, "Date": dates[2:] if ticker == "D" else dates,
                          "Adj Close Price": 100.}) for ticker in tickers
        ])
        returns = pd.DataFrame(.01, index=dates[3:], columns=tickers)

        for sample_type in ("in-sample", "OOS"):
            with self.subTest(sample_type=sample_type):
                fitted = []

                def fixed_selection(n_tickers, num_ports, replications, log_ret, sampler_types, **kwargs):
                    fitted.append(log_ret)
                    return {"sobol_simplex": {tier: np.full((1, 4), .25)
                                             for tier in ("low", "medium", "high")}}, {}

                with patch.object(su.cf, "start_date", str(dates[0].date())), \
                     patch.object(su.cf, "end_date", str(dates[-1].date())), \
                     patch.object(su.cf, "train_end_date", str(dates[4].date())), \
                     patch.object(su.cf, "test_start_date", str(dates[5].date())), \
                     patch.object(su.pd, "read_csv", lambda *args, **kwargs: prices.copy()), \
                     patch.object(su, "generate_sampler_results", fixed_selection):
                    portfolios, _, _ = su.run_backtest_case(
                        4, sample_type, {4: (tickers, returns)}, ["sobol_simplex"], num_ports=16)
                    values = portfolios["sobol_simplex"]["low"]
                    expected_start = dates[2] if sample_type == "in-sample" else dates[5]
                    self.assertEqual(values.index.min(), expected_start)
                    expected_fitted_index = returns.index if sample_type == "in-sample" else dates[3:5]
                    self.assertTrue(fitted[0].index.equals(expected_fitted_index))
                    benchmark = su.sp500_benchmark(sample_type, 10000, pd.Series(100., index=dates), values.index)
                    self.assertTrue(benchmark.index.equals(values.index))
                    np.testing.assert_allclose(values.iloc[:, 0], benchmark)
                    with self.assertRaisesRegex(ValueError, "cover every valuation date"):
                        su.sp500_benchmark(sample_type, 10000, pd.Series(100., index=dates[:-1]), values.index)


def _stock_df(prices, dates):
    """One ticker's DataFrame with 'Adj Close Price' and 'Norm Return', as
    setup_stock_dfs would build it."""
    df = pd.DataFrame({"Adj Close Price": list(prices)}, index=pd.DatetimeIndex(dates))
    df["Norm Return"] = df["Adj Close Price"] / df["Adj Close Price"].iloc[0]
    return df


class TestLoadLogReturns(TestCase):
    """Log-return cache rebuilt from prices; no randomization."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.prices = pd.DataFrame({
            "Ticker": ["A"] * 4 + ["B"] * 4,
            "Date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"] * 2),
            "Adj Close Price": [10.0, 11.0, 12.1, 13.31, 20.0, np.nan, 25.0, 30.0],
        })
        self.price_path = self.dir / "df.csv.gz"
        self.prices.to_csv(self.price_path, index=False)

    def test_drops_dates_with_missing_price(self):
        """A missing price drops its own date and the next, whose return needs it."""
        lr = bu.log_returns(self.prices)
        assert list(lr.index) == [pd.Timestamp("2020-01-07")]
        np.testing.assert_allclose(lr.iloc[0], np.log([13.31 / 12.1, 30.0 / 25.0]))

    def test_rebuilds_missing_cache(self):
        path = self.dir / "log_returns.csv.gz"
        lr = bu.load_log_returns(path, self.price_path)
        assert path.exists()
        assert list(lr.columns) == ["A", "B"]
        assert len(lr) == 1

    def test_reads_cache_without_prices(self):
        path = self.dir / "log_returns.csv.gz"
        bu.log_returns(self.prices).to_csv(path)
        lr = bu.load_log_returns(path, self.dir / "missing.csv.gz")
        np.testing.assert_allclose(lr.to_numpy(), bu.log_returns(self.prices).to_numpy())


class TestListingChangeDates(TestCase):
    """No replication: every case is a single deterministic call."""

    def test_same_window_no_changes(self):
        """A shared, uninterrupted window has exactly one change: day 1."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        a = _stock_df([10.0] * 5, dates)
        b = _stock_df([20.0] * 5, dates)
        assert bu.listing_change_dates([a, b]) == [dates[0]]

    def test_late_listing_detected(self):
        """A ticker listing partway through registers its own listing and delisting."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        a = _stock_df([10.0] * 5, dates)
        b = _stock_df([20.0, 21.0], dates[2:4])
        assert bu.listing_change_dates([a, b]) == [dates[0], dates[2], dates[4]]

    def test_temporary_halt_detected(self):
        """A gap in one ticker's own index (not a shared weekend) is a halt/resume pair."""
        dates = pd.bdate_range("2020-01-01", periods=7)
        a = _stock_df([10.0] * 7, dates)
        halted_dates = dates[[0, 1, 2, 5, 6]]
        b = _stock_df([20.0] * 5, halted_dates)
        assert bu.listing_change_dates([a, b]) == [dates[0], dates[3], dates[5]]


class TestStopLossDates(TestCase):
    """No replication: every case is a single deterministic call."""

    def test_no_trigger_without_conditions(self):
        """Neither condition given: never triggers, regardless of the price path."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        a = _stock_df([10.0, 9.0, 8.0, 7.0, 6.0], dates)
        assert bu.stop_loss_dates([a]) == {}

    def test_drop_pct_fires_on_first_breach(self):
        """Fires the first day price falls below (1 - drop_pct) times the running peak."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        a = _stock_df([10.0, 11.0, 9.0, 5.0, 5.0], dates)
        # running peak after day 2 (index 1) is 11.0; a 20% drawdown threshold is 8.8.
        # day index 2 (9.0) is still above 8.8; day index 3 (5.0) is the first breach.
        triggers = bu.stop_loss_dates([a], stop_loss_drop_pct=0.2)
        assert triggers == {0: dates[3]}

    def test_price_floor_fires(self):
        """Fires the first day price falls below an absolute floor."""
        dates = pd.bdate_range("2020-01-01", periods=4)
        a = _stock_df([10.0, 8.0, 6.0, 4.0], dates)
        triggers = bu.stop_loss_dates([a], stop_loss_price_floor=7.0)
        assert triggers == {0: dates[2]}

    def test_earliest_of_both_conditions_wins(self):
        """When both conditions are given, the earliest-firing one determines the date."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        a = _stock_df([10.0, 10.0, 7.0, 4.0, 4.0], dates)
        # floor=8.0 breaches first, at index 2 (7.0 < 8.0); drop_pct=0.5 from a peak of
        # 10 doesn't breach (5.0 threshold) until index 3 (4.0 < 5.0), one day later.
        triggers = bu.stop_loss_dates([a], stop_loss_drop_pct=0.5, stop_loss_price_floor=8.0)
        assert triggers == {0: dates[2]}

    def test_per_ticker_floor_exempts_none(self):
        """A sequence of floors applies independently per ticker; None exempts one."""
        dates = pd.bdate_range("2020-01-01", periods=3)
        a = _stock_df([10.0, 5.0, 1.0], dates)
        b = _stock_df([10.0, 5.0, 1.0], dates)
        triggers = bu.stop_loss_dates([a, b], stop_loss_price_floor=[6.0, None])
        assert triggers == {0: dates[1]}

    def test_floor_sequence_length_and_generator(self):
        """A sequence stop_loss_price_floor must have exactly one entry per
        ticker; any sequence type works, since it's only ever indexed once
        per ticker, never re-checked for length a second time."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        stocks = [_stock_df([10., 5.], dates)] * 2
        for floors in ([6.], [6., None, 6.]):
            with self.subTest(floors=floors), self.assertRaisesRegex(ValueError, "one per ticker"):
                bu.stop_loss_dates(stocks, stop_loss_price_floor=floors)
        assert bu.stop_loss_dates(stocks, stop_loss_price_floor=iter([6., None])) == {0: dates[1]}


class TestComputePortfolioValueReps(TestCase):
    """No replication: every case uses a single deterministic weights row."""

    def test_buy_and_hold_conserves_cash_and_missing_holdings(self):
        """F5: reserve cash before listing and retain a holding after quotes end."""
        toy = pd.DataFrame({
            "Ticker": ["A", "A", "A", "B"],
            "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-02"]),
            "Adj Close Price": [10.0, 10.0, 11.0, 10.0],
        })
        dfs = bu.setup_stock_dfs(toy, ["A", "B"])
        v = bu.compute_portfolio_value_reps(
            [dfs["A"], dfs["B"]], np.array([[0.5, 0.5]]), 100
        ).iloc[:, 0]
        np.testing.assert_allclose(v, [100.0, 100.0, 105.0])

    def test_buy_and_hold_flat_late_listing_conserves_principal(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        a = _stock_df([100.] * 4, dates)
        b = _stock_df([100.] * 2, dates[2:])
        v = bu.compute_portfolio_value_reps([a, b], np.array([[.5, .5], [0., 1.]]), 100)
        np.testing.assert_allclose(v, 100.)

    def test_buy_and_hold_halt_resumes_same_holding(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        a = _stock_df([100.] * 4, dates)
        b = _stock_df([100., 120., 150.], dates[[0, 1, 3]])
        v = bu.compute_portfolio_value_reps([a, b], np.array([[.5, .5]]), 100)
        np.testing.assert_allclose(v.iloc[:, 0], [100., 110., 110., 125.])

    def test_rebalancing_fixes_phantom_capital(self):
        """BLOCKER 13: with rebalance_freq set, a late listing is added at its fair
        value once listed, instead of leaving its allocation uninvested until then."""
        toy = pd.DataFrame({
            "Ticker": ["A", "A", "A", "B"],
            "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-02"]),
            "Adj Close Price": [10.0, 10.0, 11.0, 10.0],
        })
        dfs = bu.setup_stock_dfs(toy, ["A", "B"])
        v = bu.compute_portfolio_value_reps(
            [dfs["A"], dfs["B"]], np.array([[0.5, 0.5]]), 100, rebalance_freq="D"
        ).iloc[:, 0]
        # A rises 10 -> 10 -> 11 (its own +10% move on day 3) while its 50% balance
        # remains invested; B is absent again by day 3 and sits frozen at 50.
        assert v.tolist() == pytest.approx([100.0, 100.0, 105.0])

    def test_rebalancing_reserves_late_listing_cash(self):
        """A late ticker's target allocation earns no pre-listing return."""
        dates = pd.bdate_range("2020-01-01", periods=4)
        a = _stock_df([100.0, 200.0, 200.0, 200.0], dates)
        b = _stock_df([100.0, 100.0], dates[2:])

        v = bu.compute_portfolio_value_reps(
            [a, b], np.array([[0.5, 0.5]]), 100, rebalance_freq="QS"
        ).iloc[:, 0]

        np.testing.assert_allclose(v, [100.0, 150.0, 150.0, 150.0])

    def test_absent_ticker_freezes_balance(self):
        """Extends BLOCKER 13: a ticker that goes permanently absent (bankruptcy or a
        data feed simply stopping) has its dollar balance frozen, not redistributed
        to the other tickers, and no rebalance_freq=None-style cliff appears either."""
        dates = pd.bdate_range("2020-01-01", periods=120)
        x = _stock_df(100 * (1.0003 ** np.arange(len(dates))), dates)
        y_dates = dates[:100]
        y = _stock_df(50 * (0.998 ** np.arange(100)), y_dates)
        weights = np.array([[0.5, 0.5]])

        v_none = bu.compute_portfolio_value_reps([x, y], weights, 10_000, rebalance_freq=None).iloc[:, 0]
        v_fixed = bu.compute_portfolio_value_reps([x, y], weights, 10_000, rebalance_freq="QS").iloc[:, 0]
        li, li_next = y_dates[-1], dates[100]

        jump_none = v_none.loc[li_next] - v_none.loc[li]
        jump_fixed = v_fixed.loc[li_next] - v_fixed.loc[li]
        assert jump_none == pytest.approx(5000 * (1.0003 ** 100 - 1.0003 ** 99))
        assert abs(jump_fixed) < 10, "freezing Y's balance should leave no cliff at all"

    def test_universe_change_reacts_same_day(self):
        """A late listing is picked up the day it lists, not the next scheduled date,
        when rebalance_on_universe_change=True; the default (False) waits."""
        dates = pd.bdate_range("2020-01-01", periods=10)
        a = _stock_df([10.0] * 10, dates)
        # B jumps 20% the day after listing, so an allocation that includes B shows up
        # as a value difference the next day; flat prices throughout would leave both
        # modes at exactly the $100 principal with nothing to distinguish them.
        b = _stock_df([20.0, 24.0, 24.0, 24.0, 24.0], dates[3:8])
        weights = np.array([[0.5, 0.5]])

        v_default = bu.compute_portfolio_value_reps(
            [a, b], weights, 100, rebalance_freq="QS"
        ).iloc[:, 0]
        v_event = bu.compute_portfolio_value_reps(
            [a, b], weights, 100, rebalance_freq="QS", rebalance_on_universe_change=True
        ).iloc[:, 0]

        # B lists on dates[3]; with no quarter boundary in this short window, the
        # default never re-triggers on B's listing at all, so B stays uninvested.
        assert v_default.loc[dates[3]] == 100.0
        # The event-driven mode rebalances the instant B lists, so B's half appears.
        assert v_event.loc[dates[3]] == 100.0
        assert v_event.loc[dates[4]] != v_default.loc[dates[4]]

    def test_stop_loss_never_reenters(self):
        """A triggered stop-loss sells at that day's price, folds proceeds into the
        survivors, and the sold ticker's later price action no longer matters."""
        dates = pd.bdate_range("2020-01-01", periods=6)
        a = _stock_df([100.0] * 6, dates)
        # b falls 50% on day index 2, then recovers fully by the end; a stop-loss
        # should still have sold at the index-2 low and never bought back in.
        b = _stock_df([100.0, 100.0, 50.0, 100.0, 100.0, 100.0], dates)
        weights = np.array([[0.5, 0.5]])

        v = bu.compute_portfolio_value_reps(
            [a, b], weights, 100, rebalance_freq="D", stop_loss_drop_pct=0.1
        ).iloc[:, 0]
        # b sells at its 50-at-the-time value (half the 100 principal at 50% off):
        # 50 (a, unchanged) + 25 (b, sold at half) = 75. An absolute check, not just
        # relative to itself: a relative-only check would also pass if both sides
        # were wrong in the same way (e.g. the pre-fix bug that dropped this same
        # day's price move for every ticker, a and b alike).
        assert v.loc[dates[2]] == pytest.approx(75.0)
        # By the end, b's recovery should not be reflected: everything should be in a.
        assert v.iloc[-1] == pytest.approx(v.loc[dates[2]])

    def test_all_tickers_exit_holds_cash(self):
        """If every eligible ticker stops out on the same rebalance date, the pooled
        proceeds are held as cash (not discarded via a 0/0 = nan from an empty w)."""
        dates = pd.bdate_range("2020-01-01", periods=4)
        a = _stock_df([100.0, 50.0, 50.0, 50.0], dates)
        b = _stock_df([100.0, 50.0, 50.0, 50.0], dates)
        weights = np.array([[0.5, 0.5]])

        v = bu.compute_portfolio_value_reps(
            [a, b], weights, 100, rebalance_freq="D", stop_loss_drop_pct=0.1
        ).iloc[:, 0]
        assert v.tolist() == pytest.approx([100.0, 50.0, 50.0, 50.0])

    def test_zero_weight_survivor_holds_cash(self):
        """F1: a listed survivor with zero target weight cannot absorb proceeds."""
        dates = pd.bdate_range("2020-01-01", periods=3)
        a = _stock_df([100.0, 50.0, 100.0], dates)
        b = _stock_df([100.0, 100.0, 200.0], dates)
        weights = np.array([[1.0, 0.0], [0.5, 0.5]])
        with np.errstate(divide="raise", invalid="raise"):
            v = bu.compute_portfolio_value_reps(
                [a, b], weights, 100, rebalance_freq="D",
                stop_loss_drop_pct=0.1,
            )
        assert np.isfinite(v.to_numpy()).all()
        # The first replication holds cash; the second reinvests in B, which doubles.
        np.testing.assert_allclose(v.to_numpy(), [[100, 100], [50, 75], [50, 150]])

    def test_zero_weight_waits_for_listing(self):
        """F1: hold cash until the positive-weight stock becomes available."""
        dates = pd.bdate_range("2020-01-01", periods=3)
        a = _stock_df([100.0, 200.0, 300.0], dates)
        b = _stock_df([100.0, 110.0], dates[1:])
        weights = np.array([[0.0, 1.0]])
        for policy in ({"rebalance_freq": "D"},
                       {"rebalance_freq": "QS", "rebalance_on_universe_change": True}):
            with np.errstate(divide="raise", invalid="raise"):
                v = bu.compute_portfolio_value_reps([a, b], weights, 100, **policy)
            assert np.isfinite(v.to_numpy()).all()
            np.testing.assert_allclose(v.iloc[:, 0], [100, 100, 110])

    def test_combined_floor_and_drop_pct(self):
        """stop_loss_price_floor and stop_loss_drop_pct can both be supplied; whichever
        condition fires first for a ticker determines its sale date."""
        dates = pd.bdate_range("2020-01-01", periods=4)
        a = _stock_df([100.0] * 4, dates)
        b = _stock_df([100.0, 100.0, 40.0, 40.0], dates)
        weights = np.array([[0.5, 0.5]])
        exit_dates = bu.stop_loss_dates([a, b], stop_loss_drop_pct=0.9, stop_loss_price_floor=50.0)
        assert exit_dates == {1: dates[2]}
        v = bu.compute_portfolio_value_reps(
            [a, b], weights, 100, rebalance_freq="D", stop_loss_price_floor=[None, 50.0]
        ).iloc[:, 0]
        assert v.iloc[-1] == pytest.approx(v.loc[dates[2]])

    def test_unequal_stock_count_raises(self):
        """A mismatched weights/stock_dfs width is a configuration error, not silently ignored."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        a = _stock_df([10.0, 10.0], dates)
        with pytest.raises(ValueError):
            bu.compute_portfolio_value_reps([a], np.array([[0.5, 0.5]]), 100)

    def test_missing_selection_stays_nan_not_zero(self):
        """A replication with no eligible candidate (all-NaN weight row, as
        sharpe_reps now returns for an empty tier) must value as NaN, not a
        fabricated $0: pandas' default sum() treats an all-NaN row as 0."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        a = _stock_df([100.0, 100.0], dates)
        b = _stock_df([100.0, 100.0], dates)
        weights = np.array([[np.nan, np.nan], [0.5, 0.5]])
        v = bu.compute_portfolio_value_reps([a, b], weights, 100)
        self.assertTrue(v.iloc[:, 0].isna().all())
        np.testing.assert_allclose(v.iloc[:, 1], 100.0)

    def test_partially_nan_weight_row_raises(self):
        """A weight row with some but not all entries NaN is malformed input,
        not a valid 'missing selection' (which is all-NaN), so it must raise
        rather than silently propagating a partial sum."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        a = _stock_df([100.0, 100.0], dates)
        b = _stock_df([100.0, 100.0], dates)
        with pytest.raises(ValueError):
            bu.compute_portfolio_value_reps([a, b], np.array([[np.nan, 0.5]]), 100)

    def test_empty_tier_survives_to_valuation(self):
        """End-to-end selection-to-valuation regression (not just sharpe_reps'
        own output): a tier missing for some replications must not turn into
        an apparent total loss once those weights are valued over time."""
        simple = np.array([[.01, .01], [.02, .03], [.03, .05]])
        fit_dates = pd.bdate_range("2020-01-01", periods=3)
        log_ret = pd.DataFrame(np.log1p(simple), index=fit_dates, columns=["A", "B"])
        weights = np.array([
            [[1., 0.], [1., 0.], [0., 1.]],    # medium empty
            [[1., 0.], [1., 0.], [0., 1.]],    # medium empty
            [[1., 0.], [.5, .5], [0., 1.]],    # medium populated
            [[1., 0.], [.25, .75], [0., 1.]],  # medium populated
        ])
        sr = bu.sharpe_reps(weights, log_ret)

        val_dates = pd.bdate_range("2022-01-01", periods=2)
        a = _stock_df([100.0, 100.0], val_dates)
        b = _stock_df([100.0, 100.0], val_dates)
        values = bu.compute_portfolio_value_reps([a, b], sr["medium"], principal=100)
        self.assertTrue(values.iloc[:, :2].isna().all().all())
        np.testing.assert_allclose(values.iloc[:, 2:], 100.0)
        # The fabricated-zero bug would have pulled this mean down to 50.
        np.testing.assert_allclose(values.mean(axis=1), 100.0)


class TestComputeAllPortfolios(TestCase):
    def test_forwards_sampler_and_risk_level(self):
        """One DataFrame per (sampler, risk level), each a real backtest."""
        dates = pd.bdate_range("2020-01-01", periods=4)
        a = _stock_df([10.0] * 4, dates)
        b = _stock_df([20.0] * 4, dates)
        sr_dict = {
            "sobol": {"low": np.array([[0.5, 0.5]]), "high": np.array([[0.8, 0.2]])},
            "iid": {"low": np.array([[0.3, 0.7]]), "high": np.array([[0.9, 0.1]])},
        }
        out = bu.compute_all_portfolios([a, b], sr_dict, ["low", "high"], 100)
        assert set(out) == {"sobol", "iid"}
        for sampler in out:
            assert set(out[sampler]) == {"low", "high"}
            for risk in out[sampler]:
                assert out[sampler][risk].iloc[0, 0] == pytest.approx(100.0)


class TestSharpeReps(TestCase):
    """No replication (R=1), one portfolio (P=1), one ticker (D=1): the three
    risk tiers all select that same single portfolio, so low/medium/high
    Sharpe come out identical every time; only the Sharpe formula itself,
    not the risk-tier selection, is under test here."""

    def test_no_rf_uses_raw_return(self):
        """Without a risk-free rate, score the converted simple returns."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        log_ret = pd.DataFrame({"A": [0.02, 0.00]}, index=dates)
        weights = np.array([[[1.0]]])
        sr = bu.sharpe_reps(weights, log_ret)
        assert sr["low risk Sharpe"] == pytest.approx(11.225)
        assert sr["medium risk Sharpe"] == pytest.approx(11.225)
        assert sr["high risk Sharpe"] == pytest.approx(11.225)

    def test_rf_uses_daily_equivalent(self):
        """Convert annual log rates to daily simple returns before subtraction."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        log_ret = pd.DataFrame({"A": [0.02, 0.00]}, index=dates)
        log_rf = pd.Series([0.0504, 0.0504], index=dates)
        weights = np.array([[[1.0]]])
        sr = bu.sharpe_reps(weights, log_ret, log_rf)
        excess = np.expm1(log_ret["A"]) - np.expm1(log_rf / 252)
        expected = np.round(excess.mean() / excess.std(ddof=1) * np.sqrt(252), 3)
        for tier in ("low", "medium", "high"):
            assert sr[f"{tier} risk Sharpe"] == pytest.approx(expected)

    def test_se_nan_with_one_replication(self):
        """An SE needs at least two replications; R=1 (every other test here) gives NaN."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        log_ret = pd.DataFrame({"A": [0.02, 0.00]}, index=dates)
        weights = np.array([[[1.0]]])
        sr = bu.sharpe_reps(weights, log_ret)
        assert np.isnan(sr["medium risk Sharpe SE"])

    def test_se_across_replications(self):
        """One all-in portfolio per replication: verify the mean and SE using
        independently computed simple-return scores, across every risk tier."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        log_ret = pd.DataFrame({"A": [0.02, 0.00], "B": [0.01, 0.03]}, index=dates)
        weights = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        sr = bu.sharpe_reps(weights, log_ret)
        simple = np.expm1(log_ret)
        scores = simple.mean() / simple.std(ddof=1) * np.sqrt(252)
        assert sr["medium risk Sharpe"] == pytest.approx(np.round(scores.mean(), 3))
        assert sr["medium risk Sharpe SE"] == pytest.approx(np.round(scores.std(ddof=1) / np.sqrt(2), 3))

    def test_simple_return_score_with_varying_risk_free_rate(self):
        """F7: use exact weighted simple returns and excess-return volatility."""
        simple = np.array([[1., -.5], [-.2, .1], [.1, -.05], [.03, .02]])
        dates = pd.bdate_range("2020-01-01", periods=4)
        logs = pd.DataFrame(np.log1p(simple), index=dates)
        daily_rf = np.array([.01, .02, .005, .015])
        rf = pd.Series(252 * np.log1p(daily_rf), index=dates)
        weights = np.array([[[.5, .5]], [[.25, .75]]])
        excess = simple @ weights[:, 0].T - daily_rf[:, None]
        scores = excess.mean(axis=0) / excess.std(axis=0, ddof=1) * np.sqrt(252)
        result = bu.sharpe_reps(weights, logs, rf)
        for tier in ("low", "medium", "high"):
            np.testing.assert_allclose(result[tier], weights[:, 0])
            assert result[f"{tier} risk Sharpe"] == np.round(scores.mean(), 3)
            assert result[f"{tier} risk Sharpe SE"] == np.round(scores.std(ddof=1) / np.sqrt(2), 3)

    def test_empty_tier_is_nan(self):
        """A volatility tie can leave a tier with no candidate; argmax over an
        all -inf row used to default to index 0 and mislabel it as that tier's
        pick instead of reporting the tier as genuinely empty."""
        simple = np.array([[.01, .01], [.02, .03], [.03, .05]])
        dates = pd.bdate_range("2020-01-01", periods=3)
        log_ret = pd.DataFrame(np.log1p(simple), index=dates, columns=["A", "B"])
        weights = np.array([[[1., 0.], [1., 0.], [0., 1.]]])
        result = bu.sharpe_reps(weights, log_ret)
        self.assertTrue(np.isnan(result["medium risk Sharpe"]))
        self.assertTrue(np.all(np.isnan(result["medium"])))
        self.assertFalse(np.isnan(result["low risk Sharpe"]))
        self.assertFalse(np.isnan(result["high risk Sharpe"]))

    def test_se_uses_valid_replication_count(self):
        """A tier empty in some but not all replications must divide its SE by
        sqrt(valid count), not sqrt(R): nanstd already excludes the empty
        replications from the spread, so dividing by sqrt(R) understates the SE."""
        simple = np.array([[.01, .01], [.02, .03], [.03, .05]])
        dates = pd.bdate_range("2020-01-01", periods=3)
        log_ret = pd.DataFrame(np.log1p(simple), index=dates, columns=["A", "B"])
        weights = np.array([
            [[1., 0.], [1., 0.], [0., 1.]],    # medium empty
            [[1., 0.], [1., 0.], [0., 1.]],    # medium empty
            [[1., 0.], [.5, .5], [0., 1.]],    # medium populated
            [[1., 0.], [.25, .75], [0., 1.]],  # medium populated
        ])
        result = bu.sharpe_reps(weights, log_ret)
        self.assertEqual(result["medium risk valid replications"], 2)
        self.assertEqual(result["medium risk Sharpe SE"], 0.756)
        self.assertEqual(result["low risk valid replications"], 4)

    def test_se_nan_below_two_valid_replications(self):
        """A tier populated in only one replication cannot estimate a spread."""
        simple = np.array([[.01, .01], [.02, .03], [.03, .05]])
        dates = pd.bdate_range("2020-01-01", periods=3)
        log_ret = pd.DataFrame(np.log1p(simple), index=dates, columns=["A", "B"])
        weights = np.array([
            [[1., 0.], [1., 0.], [0., 1.]],   # medium empty
            [[1., 0.], [1., 0.], [0., 1.]],   # medium empty
            [[1., 0.], [.5, .5], [0., 1.]],   # medium populated (only one)
        ])
        result = bu.sharpe_reps(weights, log_ret)
        self.assertEqual(result["medium risk valid replications"], 1)
        self.assertTrue(np.isnan(result["medium risk Sharpe SE"]))
