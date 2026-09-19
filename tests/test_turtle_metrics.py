"""Closed-form checks of the performance arithmetic."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from seikan.turtle import metrics


def _index(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="1D")


def test_constant_growth_has_no_dispersion_and_the_compounding_cagr():
    n = 20  # ×1.5 per bar is exact in binary, so every bar return is exactly 0.5
    equity = 100.0 * 1.5 ** np.arange(1, n + 1)
    m = metrics.performance(equity, 100.0, 252, _index(n), np.ones(n, bool), np.ones(n))
    assert m["total_return"] == pytest.approx(1.5**20 - 1)
    assert m["cagr"] == pytest.approx(1.5**252 - 1)  # (1.5^20)^(252/20)
    assert m["annualized_volatility"] == 0.0
    assert m["sharpe"] is None and m["sortino"] is None and m["calmar"] is None
    assert m["max_drawdown"] == 0.0 and m["drawdown"]["peak_time"] is None
    assert m["drawdown"]["longest_drawdown_bars"] == 0
    assert m["positive_bars_fraction"] == 1.0 and m["skewness"] is None
    assert m["exposure"] == {
        "bars_in_market": n,
        "fraction_in_market": 1.0,
        "mean_gross_exposure": 1.0,
        "max_gross_exposure": 1.0,
    }
    assert (m["n_bars"], m["bars_per_year"], m["start_equity"]) == (n, 252, 100.0)


def test_drawdown_geometry_is_read_off_the_curve():
    equity = np.array([100.0, 110.0, 99.0, 95.0, 104.0, 112.0, 108.0, 109.0])
    idx = _index(len(equity))
    m = metrics.performance(equity, 100.0, 252, idx, np.ones(8, bool), np.zeros(8))
    assert m["max_drawdown"] == pytest.approx(95.0 / 110.0 - 1.0)
    dd = m["drawdown"]
    assert dd["peak_time"] == idx[1].isoformat() and dd["trough_time"] == idx[3].isoformat()
    assert dd["recovery_time"] == idx[5].isoformat()
    assert (dd["bars_to_trough"], dd["bars_to_recovery"]) == (2, 2)
    assert dd["longest_drawdown_bars"] == 3 and dd["longest_drawdown_open"] is False
    # A drawdown from the starting equity itself has no peak bar and is open at the end.
    m2 = metrics.performance(
        np.array([90.0, 80.0, 85.0]), 100.0, 252, _index(3), np.ones(3, bool), np.zeros(3)
    )
    dd2 = m2["drawdown"]
    assert dd2["peak_time"] is None and dd2["bars_to_trough"] == 2
    assert dd2["recovery_time"] is None and dd2["longest_drawdown_open"] is True
    assert m2["sharpe"] is not None and m2["sortino"] is not None and m2["calmar"] is not None


def test_relative_metrics_against_a_doubled_benchmark():
    rng = np.random.RandomState(0)
    r_b = rng.normal(0.0, 0.01, 500)
    r_s = 2.0 * r_b
    rel = metrics.relative(r_s, r_b, 252, cagr_s=0.2, cagr_b=0.1)
    assert rel["beta"] == pytest.approx(2.0)
    assert rel["alpha_bar"] == pytest.approx(0.0, abs=1e-12)
    assert rel["alpha_annualized"] == pytest.approx(0.0, abs=1e-9)
    assert rel["correlation"] == pytest.approx(1.0) and rel["r2"] == pytest.approx(1.0)
    assert rel["up_capture"] == pytest.approx(2.0) and rel["down_capture"] == pytest.approx(2.0)
    assert rel["n_up_bars"] + rel["n_down_bars"] == np.count_nonzero(r_b != 0)
    assert rel["excess_cagr"] == pytest.approx(0.1)
    assert rel["tracking_error"] == pytest.approx(np.std(r_b, ddof=1) * math.sqrt(252))
    assert rel["information_ratio"] == pytest.approx(
        np.mean(r_b) / np.std(r_b, ddof=1) * math.sqrt(252)
    )
    assert rel["excess_total_return"] == pytest.approx(np.prod(1 + r_s) - np.prod(1 + r_b))


def test_relative_metrics_null_where_the_benchmark_is_flat():
    r_b = np.zeros(10)
    r_s = np.linspace(-0.01, 0.01, 10)
    rel = metrics.relative(r_s, r_b, 252, None, None)
    assert rel["beta"] is None and rel["alpha_bar"] is None and rel["correlation"] is None
    assert rel["up_capture"] is None and rel["down_capture"] is None
    assert rel["excess_cagr"] is None and rel["n_up_bars"] == 0


def test_periodic_returns_compound_within_each_period():
    idx = pd.DatetimeIndex(["2020-01-30", "2020-01-31", "2020-02-03", "2021-01-04"])
    r = np.array([0.1, 0.1, -0.5, 0.25])
    p = metrics.periodic(r, idx)
    assert p["monthly"] == {
        "2020-01": pytest.approx(1.1 * 1.1 - 1),
        "2020-02": pytest.approx(-0.5),
        "2021-01": pytest.approx(0.25),
    }
    assert p["annual"] == {"2020": pytest.approx(1.1 * 1.1 * 0.5 - 1), "2021": pytest.approx(0.25)}


class _Trip:
    """A round trip by its numbers: ``pnl`` is net; the exit proceeds and gross pnl follow from
    the cost basis and the commission, and the other cost buckets are attribution only."""

    def __init__(
        self,
        entry_bar,
        exit_bar,
        entry_px,
        pnl,
        cost,
        units,
        n_adds,
        commission=0.0,
        slippage=0.0,
        shock=0.0,
        impact=0.0,
    ):
        self.entry_bar, self.exit_bar, self.entry_px = entry_bar, exit_bar, entry_px
        self.pnl, self.cost_basis, self.units, self.n_adds = pnl, cost, units, n_adds
        self.commission, self.slippage, self.shock, self.impact = (
            commission,
            slippage,
            shock,
            impact,
        )
        self.gross_pnl = pnl + commission
        self.proceeds = cost + self.gross_pnl


def _ledger(**overrides):
    keys = [
        "entries", "adds", "exits_stop_close", "exits_stop_gap", "exits_stop_trade",
        "exits_channel_close", "exits_channel_trade", "adds_skipped_budget",
        "adds_filled_below_level", "entries_skipped_warmup", "entries_skipped_in_position",
        "entries_skipped_zero_size", "entries_skipped_budget", "entries_skipped_end_of_data",
        "entries_cash_capped", "stop_holds",
    ]  # fmt: skip
    return {k: overrides.get(k, 0) for k in keys}


def test_trade_stats_over_closed_trips():
    trips = [
        _Trip(0, 4, 10.0, 200.0, 1000.0, 3, 2, commission=4.0, slippage=3.0, shock=2.0, impact=1.0),
        _Trip(6, 8, 10.0, -100.0, 500.0, 1, 0, commission=2.0),
        _Trip(9, 12, 10.0, 0.0, 400.0, 2, 1, slippage=0.5),
    ]
    ledger = _ledger(entries=3, adds=3, exits_stop_close=2, exits_channel_close=1, stop_holds=1)
    t = metrics.trade_stats(trips, 1, ledger, [(-0.02, 0.05), (-0.03, 0.01), (-0.01, 0.02)])
    assert (t["n_round_trips"], t["n_open"], t["n_wins"], t["n_losses"], t["n_flat"]) == (
        3,
        1,
        1,
        1,
        1,
    )
    assert t["win_rate"] == pytest.approx(1 / 3)
    assert (t["gross_profit"], t["gross_loss"], t["net_pnl"]) == (200.0, -100.0, 100.0)
    assert t["gross_pnl"] == pytest.approx(106.0)  # net plus the 6.0 of commission
    assert t["costs"] == {
        "commission": 6.0, "slippage": 3.5, "shock": 2.0, "impact": 1.0, "total": 12.5,
    }  # fmt: skip
    proceeds = (1000 + 204) + (500 - 98) + (400 + 0)
    assert t["turnover"] == pytest.approx(1900.0 + proceeds)
    assert t["cost_bps_of_turnover"] == pytest.approx(12.5 / (1900.0 + proceeds) * 1e4)
    assert t["profit_factor"] == 2.0 and t["expectancy"] == pytest.approx(100 / 3)
    assert t["expectancy_ret"] == pytest.approx((0.2 - 0.2 + 0.0) / 3)
    assert (t["avg_win"], t["avg_loss"], t["largest_win"], t["largest_loss"]) == (
        200.0,
        -100.0,
        200.0,
        -100.0,
    )
    assert t["win_loss_ratio"] == 2.0
    assert (t["avg_bars_held"], t["median_bars_held"], t["max_bars_held"]) == (3.0, 3.0, 4)
    assert t["avg_units_at_exit"] == 2.0 and t["max_units_reached"] == 3
    assert t["avg_adds_per_trip"] == 1.0
    assert t["mean_mae"] == pytest.approx(-0.02) and t["mean_mfe"] == pytest.approx(8 / 300)
    assert t["exits"] == {
        "stop_close": 2, "stop_gap": 0, "stop_trade": 0, "channel_close": 1,
        "channel_trade": 0, "end_of_data": 1,
    }  # fmt: skip
    assert t["entries"]["taken"] == 3 and t["adds"]["taken"] == 3 and t["stop_holds"] == 1


def test_trade_stats_with_no_trips_is_all_null_counts_zero():
    t = metrics.trade_stats([], 0, _ledger(entries_skipped_warmup=2), [])
    assert t["n_round_trips"] == 0 and t["win_rate"] is None and t["profit_factor"] is None
    assert t["expectancy"] is None and t["max_units_reached"] == 0 and t["net_pnl"] == 0.0
    assert t["gross_pnl"] == 0.0 and t["costs"]["total"] == 0.0
    assert t["turnover"] == 0.0 and t["cost_bps_of_turnover"] is None
    assert t["entries"]["skipped"]["warmup"] == 2


def test_excursions_read_the_held_bars_inclusive():
    trip = _Trip(2, 4, 10.0, 0.0, 1.0, 1, 0)
    low = np.array([1.0, 1.0, 9.5, 9.0, 9.8, 1.0])
    high = np.array([50.0, 50.0, 10.5, 11.0, 10.2, 50.0])
    mae, mfe = metrics.excursions(trip, high, low)
    assert (mae, mfe) == pytest.approx((-0.1, 0.1))


def test_sum_ledgers():
    assert metrics.sum_ledgers([{"a": 1, "b": 2}, {"a": 3}]) == {"a": 4, "b": 2}
