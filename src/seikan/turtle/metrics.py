"""The performance arithmetic, pure numpy/pandas/scipy over equity curves and round trips.

Conventions, stated once: simple bar returns against the previous bar's equity (the bar before
the first is the starting equity); sample standard deviations (``ddof = 1``); a risk-free rate of
zero; annualization by ``bars_per_year`` (√ for dispersion, linear for a per-bar alpha, the
compounding exponent for CAGR); scipy's population skewness and Pearson kurtosis (normal = 3),
the moments the rest of seikan reports; every ratio whose denominator is zero, and every moment
over fewer than three bars or zero dispersion, is null rather than an infinity.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from seikan import _turtle
from seikan.types.turtle import (
    AddCounts,
    DrawdownBlock,
    EntryCounts,
    ExitCounts,
    ExposureBlock,
    PerformanceMetrics,
    PeriodicReturns,
    RelativeMetrics,
    TradeStats,
)


def _finite(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def _mean_or_none(x: np.ndarray) -> float | None:
    return float(np.mean(x)) if len(x) else None


def bar_returns(equity: np.ndarray, base: float) -> np.ndarray:
    """Simple returns per bar; the first bar's return is against ``base``."""
    prev = np.concatenate([[base], equity[:-1]])
    return np.asarray(equity / prev - 1.0, dtype=float)


def compounded(returns: np.ndarray) -> float:
    return float(np.prod(1.0 + returns) - 1.0)


def cagr(start: float, end: float, n_bars: int, bars_per_year: int) -> float | None:
    """``(end/start)^(bars_per_year/n_bars) - 1``; null when the curve ends at or below zero."""
    if n_bars <= 0 or start <= 0.0 or end <= 0.0:
        return None
    return _finite((end / start) ** (bars_per_year / n_bars) - 1.0)


def _std(x: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    return _finite(float(np.std(x, ddof=1)))


def drawdown_series(equity: np.ndarray, base: float) -> np.ndarray:
    """Depth below the running peak (the starting equity included), ≤ 0 per bar."""
    peaks = np.maximum.accumulate(np.concatenate([[base], equity]))[1:]
    return np.asarray(equity / peaks - 1.0, dtype=float)


def drawdown_block(equity: np.ndarray, base: float, index: pd.DatetimeIndex) -> DrawdownBlock:
    dd = drawdown_series(equity, base)
    below = dd < 0.0
    # The longest run of bars below the peak, and whether the final run is that run (open).
    longest = current = 0
    for flag in below:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    longest_open = bool(below[-1]) and current == longest and longest > 0
    if not below.any():
        return {
            "peak_time": None,
            "trough_time": None,
            "recovery_time": None,
            "bars_to_trough": None,
            "bars_to_recovery": None,
            "longest_drawdown_bars": 0,
            "longest_drawdown_open": False,
        }
    trough = int(np.argmin(dd))
    peak_value = float(np.max(np.concatenate([[base], equity[: trough + 1]])))
    at_peak = np.flatnonzero(equity[: trough + 1] >= peak_value)
    peak_bar = int(at_peak[-1]) if len(at_peak) else None
    after = np.flatnonzero(equity[trough + 1 :] >= peak_value)
    recovery = trough + 1 + int(after[0]) if len(after) else None
    return {
        "peak_time": index[peak_bar].isoformat() if peak_bar is not None else None,
        "trough_time": index[trough].isoformat(),
        "recovery_time": index[recovery].isoformat() if recovery is not None else None,
        "bars_to_trough": trough - peak_bar if peak_bar is not None else trough + 1,
        "bars_to_recovery": recovery - trough if recovery is not None else None,
        "longest_drawdown_bars": longest,
        "longest_drawdown_open": longest_open,
    }


def performance(
    equity: np.ndarray,
    base: float,
    bars_per_year: int,
    index: pd.DatetimeIndex,
    in_market: np.ndarray,
    gross_exposure: np.ndarray,
) -> PerformanceMetrics:
    """Every metric of one equity curve (see the module conventions)."""
    n = len(equity)
    r = bar_returns(equity, base)
    end = float(equity[-1])
    growth = cagr(base, end, n, bars_per_year)
    vol = _std(r)
    mean = float(np.mean(r))
    downside = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)))
    dd = drawdown_series(equity, base)
    max_dd = float(np.min(dd))
    if n >= 3 and vol is not None and vol > 0.0:
        skewness = _finite(float(sp_stats.skew(r, bias=True)))
        kurtosis = _finite(float(sp_stats.kurtosis(r, fisher=False, bias=True)))
    else:
        skewness = kurtosis = None
    exposure: ExposureBlock = {
        "bars_in_market": int(np.count_nonzero(in_market)),
        "fraction_in_market": float(np.count_nonzero(in_market) / n),
        "mean_gross_exposure": float(np.mean(gross_exposure)),
        "max_gross_exposure": float(np.max(gross_exposure)),
    }
    return {
        "start_equity": float(base),
        "end_equity": end,
        "net_pnl": end - float(base),
        "total_return": end / float(base) - 1.0,
        "cagr": growth,
        "annualized_volatility": None if vol is None else vol * math.sqrt(bars_per_year),
        "sharpe": None if not vol else _finite(mean / vol * math.sqrt(bars_per_year)),
        "sortino": None if downside == 0.0 else _finite(mean / downside * math.sqrt(bars_per_year)),
        "calmar": None if growth is None or max_dd == 0.0 else _finite(growth / abs(max_dd)),
        "max_drawdown": max_dd,
        "drawdown": drawdown_block(equity, base, index),
        "best_bar_return": float(np.max(r)),
        "worst_bar_return": float(np.min(r)),
        "mean_bar_return": mean,
        "median_bar_return": float(np.median(r)),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "positive_bars_fraction": float(np.count_nonzero(r > 0.0) / n),
        "exposure": exposure,
        "n_bars": n,
        "bars_per_year": bars_per_year,
    }


def relative(
    r_s: np.ndarray, r_b: np.ndarray, bars_per_year: int, cagr_s: float | None, cagr_b: float | None
) -> RelativeMetrics:
    """The strategy's bar returns against the benchmark's, bar for bar."""
    n = len(r_s)
    beta = alpha_bar = alpha_year = corr = r2 = None
    if n >= 2:
        var_b = float(np.var(r_b, ddof=1))
        std_s = float(np.std(r_s, ddof=1))
        if var_b > 0.0:
            beta = _finite(float(np.cov(r_s, r_b, ddof=1)[0, 1]) / var_b)
            if beta is not None:
                alpha_bar = _finite(float(np.mean(r_s)) - beta * float(np.mean(r_b)))
                alpha_year = None if alpha_bar is None else alpha_bar * bars_per_year
            if std_s > 0.0:
                corr = _finite(float(np.corrcoef(r_s, r_b)[0, 1]))
                r2 = None if corr is None else corr * corr
    diff = r_s - r_b
    te = _std(diff)
    ir = None if not te else _finite(float(np.mean(diff)) / te * math.sqrt(bars_per_year))
    up = r_b > 0.0
    down = r_b < 0.0
    up_capture = _finite(float(np.mean(r_s[up])) / float(np.mean(r_b[up]))) if np.any(up) else None
    down_capture = (
        _finite(float(np.mean(r_s[down])) / float(np.mean(r_b[down]))) if np.any(down) else None
    )
    return {
        "beta": beta,
        "alpha_bar": alpha_bar,
        "alpha_annualized": alpha_year,
        "correlation": corr,
        "r2": r2,
        "tracking_error": None if te is None else te * math.sqrt(bars_per_year),
        "information_ratio": ir,
        "excess_total_return": compounded(r_s) - compounded(r_b),
        "excess_cagr": None if cagr_s is None or cagr_b is None else cagr_s - cagr_b,
        "up_capture": up_capture,
        "down_capture": down_capture,
        "n_up_bars": int(np.count_nonzero(up)),
        "n_down_bars": int(np.count_nonzero(down)),
    }


def periodic(returns: np.ndarray, index: pd.DatetimeIndex) -> PeriodicReturns:
    """Compounded returns per calendar month and year, in calendar order."""
    s = pd.Series(1.0 + returns, index=index)
    monthly = s.groupby(index.strftime("%Y-%m")).prod() - 1.0
    annual = s.groupby(index.strftime("%Y")).prod() - 1.0
    return {
        "monthly": {str(k): float(v) for k, v in monthly.items()},
        "annual": {str(k): float(v) for k, v in annual.items()},
    }


def excursions(trip: _turtle.RoundTrip, high: np.ndarray, low: np.ndarray) -> tuple[float, float]:
    """The raw adverse and favorable excursions against the entry price over the bars the
    position was held, entry and exit bars included: ``(min low / P0 − 1, max high / P0 − 1)``."""
    lo = float(np.min(low[trip.entry_bar : trip.exit_bar + 1]))
    hi = float(np.max(high[trip.entry_bar : trip.exit_bar + 1]))
    return lo / trip.entry_px - 1.0, hi / trip.entry_px - 1.0


def trade_stats(
    trips: list[_turtle.RoundTrip],
    n_open: int,
    ledger: dict[str, int],
    excursion_pairs: list[tuple[float, float]],
) -> TradeStats:
    """Round-trip statistics over the CLOSED trips plus the ledger's counts."""
    pnl = np.array([t.pnl for t in trips], dtype=float)
    rets = np.array([t.pnl / t.cost_basis for t in trips], dtype=float)
    held = np.array([t.exit_bar - t.entry_bar for t in trips], dtype=float)
    units = np.array([t.units for t in trips], dtype=float)
    adds = np.array([t.n_adds for t in trips], dtype=float)
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    n = len(trips)
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(losses.sum()) if len(losses) else 0.0
    avg_win = _mean_or_none(wins)
    avg_loss = _mean_or_none(losses)
    maes = np.array([m for m, _ in excursion_pairs], dtype=float)
    mfes = np.array([f for _, f in excursion_pairs], dtype=float)
    exits: ExitCounts = {
        "stop_close": ledger["exits_stop_close"],
        "stop_gap": ledger["exits_stop_gap"],
        "stop_trade": ledger["exits_stop_trade"],
        "channel_close": ledger["exits_channel_close"],
        "channel_trade": ledger["exits_channel_trade"],
        "end_of_data": n_open,
    }
    entries: EntryCounts = {
        "taken": ledger["entries"],
        "cash_capped": ledger["entries_cash_capped"],
        "skipped": {
            "warmup": ledger["entries_skipped_warmup"],
            "in_position": ledger["entries_skipped_in_position"],
            "zero_size": ledger["entries_skipped_zero_size"],
            "budget": ledger["entries_skipped_budget"],
            "end_of_data": ledger["entries_skipped_end_of_data"],
        },
    }
    add_counts: AddCounts = {
        "taken": ledger["adds"],
        "filled_below_level": ledger["adds_filled_below_level"],
        "skipped": {"budget": ledger["adds_skipped_budget"]},
    }
    return {
        "n_round_trips": n,
        "n_open": n_open,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "n_flat": int(np.count_nonzero(pnl == 0.0)),
        "win_rate": len(wins) / n if n else None,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_pnl": float(pnl.sum()) if n else 0.0,
        "profit_factor": None if gross_loss == 0.0 else gross_profit / abs(gross_loss),
        "expectancy": _mean_or_none(pnl),
        "expectancy_ret": _mean_or_none(rets),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": float(wins.max()) if len(wins) else None,
        "largest_loss": float(losses.min()) if len(losses) else None,
        "win_loss_ratio": (
            None
            if avg_win is None or avg_loss is None or avg_loss == 0.0
            else avg_win / abs(avg_loss)
        ),
        "avg_bars_held": _mean_or_none(held),
        "median_bars_held": float(np.median(held)) if n else None,
        "max_bars_held": int(held.max()) if n else None,
        "avg_units_at_exit": _mean_or_none(units),
        "max_units_reached": int(units.max()) if n else 0,
        "avg_adds_per_trip": _mean_or_none(adds),
        "mean_mae": _mean_or_none(maes),
        "mean_mfe": _mean_or_none(mfes),
        "exits": exits,
        "entries": entries,
        "adds": add_counts,
        "stop_holds": ledger["stop_holds"],
    }


def sum_ledgers(ledgers: list[dict[str, int]]) -> dict[str, int]:
    """The per-target ledgers summed key by key."""
    total: dict[str, int] = {}
    for ledger in ledgers:
        for key, value in ledger.items():
            total[key] = total.get(key, 0) + value
    return total
