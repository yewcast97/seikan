"""One nautilus_trader backtest per cell, and the run's orchestration.

Every cell (entry combo) gets a fresh ``BacktestEngine``: one venue (``SIM``, netting, a cash
account holding the whole equity), one ``Equity`` instrument and one strategy per target, the
bars and opening prints of every target added in declaration order. After the run the venue's
own books are reconciled against the kernel's ledger — fill counts, realized pnl, end cash — and
a mismatch is a seikan bug (``RuntimeError``, the exit-4 class), never a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from seikan import _turtle
from seikan.api import MarketData
from seikan.dsl.schema import Thesis
from seikan.turtle.coefficients import TurtleCoefficients
from seikan.turtle.market import VENUE, SimulationData, instrument_for, preflight, venue_data
from seikan.turtle.signals import EntryCell, entry_cells
from seikan.turtle.strategy import FillRecord, TargetSamples

#: The reconciliation tolerance, as a fraction of the starting equity.
RECONCILE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class TargetRun:
    """One target's simulation inside a cell."""

    target: str
    instrument_id: str
    fills: list[FillRecord]
    trips: list[_turtle.RoundTrip]
    open_trip: _turtle.RoundTrip | None
    ledger: dict[str, int]
    samples: TargetSamples
    end_cash: float
    end_shares: int
    end_units: int
    end_stop: float | None
    end_add_level: float | None


@dataclass(frozen=True)
class Reconciliation:
    """The kernel's ledger against the venue's books for one cell: the fill count, the cash
    change over the run (the venue's ``PnL (total)`` is its account's cash change, so an open
    position counts as its cost on both sides) and the closing cash balance."""

    n_fills_ledger: int
    n_fills_engine: int
    cash_change_ledger: float
    cash_change_engine: float
    end_cash_ledger: float
    end_cash_account: float
    matched: bool


@dataclass(frozen=True)
class CellRun:
    """One cell's complete simulation."""

    cell: EntryCell
    targets: dict[str, TargetRun]
    engine_stats: dict[str, Any]
    reconciliation: Reconciliation


@dataclass(frozen=True)
class TurtleResult:
    """The whole run: the admitted data, every cell, the resolved coefficients."""

    axes: list[str]
    cells: list[CellRun]
    data: SimulationData
    coefficients: TurtleCoefficients
    budget_per_target: float


def kernel_coefficients(c: TurtleCoefficients, budget: float) -> _turtle.Coefficients:
    """The kernel's coefficient set for one target with cash budget ``budget``."""
    return _turtle.Coefficients(
        atr_period=c.atr_period,
        add_step_n=c.add_step_n,
        max_units=c.max_units,
        stop_n=c.stop_n,
        exit_lookback=c.exit_lookback,
        risk_per_unit=c.risk_per_unit,
        stop_trigger=c.stop_trigger,
        exit_trigger=c.exit_trigger,
        stop_n_source=c.stop_n_source,
        price_precision=c.price_precision,
        budget=budget,
    )


def engine_version() -> str:
    """The installed nautilus_trader distribution's version."""
    from importlib.metadata import version

    return version("nautilus_trader")


def run_cell(cell: EntryCell, data: SimulationData, c: TurtleCoefficients) -> CellRun:
    """Simulate one cell on a fresh venue and reconcile it."""
    from nautilus_trader.backtest import BacktestEngine
    from nautilus_trader.config import BacktestEngineConfig, LoggerConfig, RiskEngineConfig
    from nautilus_trader.model import AccountType, Currency, Money, OmsType, TraderId, Venue

    from seikan.turtle.strategy import TurtleTargetConfig, TurtleTargetStrategy

    targets = list(data.targets)
    budget = c.equity / len(targets)
    kernel = kernel_coefficients(c, budget)
    currency = Currency.from_str(c.currency)
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("SEIKAN-001"),
            logging=LoggerConfig(bypass_logging=True),
            # The kernel's per-target budgets are the pre-trade check: it never sizes a buy the
            # target's cash cannot cover. The venue's risk engine is bypassed because its own
            # pre-trade checks misread a resting sell stop — it locks balance for the shares a
            # not-yet-updated position does not cover and counts the stop's notional against
            # free cash when the next buy arrives — and would deny buys the ledger affords. The
            # account's books stay live, and the reconciliation below holds them to the ledger.
            risk_engine=RiskEngineConfig(bypass=True),
        )
    )
    engine.add_venue(
        venue=Venue(VENUE),
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=currency,
        starting_balances=[Money(c.equity, currency)],
        bar_execution=True,
        bar_adaptive_high_low_ordering=False,
        reject_stop_orders=False,
        allow_cash_borrowing=False,
        support_contingent_orders=False,
        use_reduce_only=False,
        use_random_ids=False,
    )
    strategies: list[TurtleTargetStrategy] = []
    for i, target in enumerate(targets):
        instrument = instrument_for(i, c.currency, c.price_precision)
        engine.add_instrument(instrument)
        strategy = TurtleTargetStrategy(
            TurtleTargetConfig(
                order_id_tag=f"{i:03d}",
                target=target,
                instrument=instrument,
                bars=data.targets[target],
                ts_ns=data.ts_ns,
                fired=cell.fired[target],
                kernel=kernel,
            )
        )
        engine.add_strategy(strategy)
        strategies.append(strategy)
        engine.add_data(venue_data(instrument, data.targets[target], data.ts_ns))
    try:
        engine.run()
        result = engine.get_result()
        fills_report = engine.generate_order_fills_report()
        account = engine.generate_account_report(Venue(VENUE))
    finally:
        engine.dispose()
    problems = [p for s in strategies for p in s.problems]
    if problems:
        # The venue logs and swallows a handler failure; the strategies record theirs, and a
        # recorded one is a seikan bug (the exit-4 class), never a result.
        raise RuntimeError(f"the simulation broke an invariant: {problems[0]}")
    n_bars = len(data.index)
    runs: dict[str, TargetRun] = {}
    for strategy in strategies:
        target = strategy.settings.target
        last_close = float(data.targets[target].close[n_bars - 1])
        machine = strategy.machine
        if len(strategy.samples.equity) != n_bars:
            raise RuntimeError(
                f"{target}: {len(strategy.samples.equity)} bars sampled of {n_bars} fed"
            )
        runs[target] = TargetRun(
            target=target,
            instrument_id=str(strategy.settings.instrument.id),
            fills=list(strategy.fills),
            trips=list(machine.closed),
            open_trip=machine.finish(n_bars - 1, last_close),
            ledger=machine.ledger.to_dict(),
            samples=strategy.samples,
            end_cash=machine.cash,
            end_shares=machine.shares,
            end_units=machine.units,
            end_stop=machine.stop,
            end_add_level=machine.add_level,
        )
    stats = {
        "stats_pnls": dict(result.stats_pnls),
        "stats_returns": dict(result.stats_returns),
        "stats_general": dict(result.stats_general),
    }
    reconciliation = _reconcile(runs, stats, account, len(fills_report), c)
    return CellRun(cell=cell, targets=runs, engine_stats=stats, reconciliation=reconciliation)


def _reconcile(
    runs: dict[str, TargetRun],
    stats: dict[str, Any],
    account: Any,
    n_fills_engine: int,
    c: TurtleCoefficients,
) -> Reconciliation:
    n_fills_ledger = sum(len(r.fills) for r in runs.values())
    end_cash_ledger = float(sum(r.end_cash for r in runs.values()))
    cash_change_ledger = end_cash_ledger - c.equity
    pnls = stats["stats_pnls"].get(c.currency, {})
    cash_change_engine = float(pnls.get("PnL (total)", float("nan")))
    end_cash_account = float(account["total"].iloc[-1]) if len(account) else float("nan")
    tolerance = RECONCILE_TOLERANCE * c.equity
    matched = (
        n_fills_ledger == n_fills_engine
        and _close(cash_change_ledger, cash_change_engine, tolerance)
        and _close(end_cash_ledger, end_cash_account, tolerance)
    )
    rec = Reconciliation(
        n_fills_ledger=n_fills_ledger,
        n_fills_engine=n_fills_engine,
        cash_change_ledger=cash_change_ledger,
        cash_change_engine=cash_change_engine,
        end_cash_ledger=end_cash_ledger,
        end_cash_account=end_cash_account,
        matched=matched,
    )
    if not matched:
        raise RuntimeError(
            "the venue's books do not reconcile with the kernel's ledger — a seikan bug, not an "
            f"input problem: {rec}"
        )
    return rec


def _close(a: float, b: float, tolerance: float) -> bool:
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tolerance)


def run_turtle(
    thesis: Thesis, md: MarketData, coefficients: TurtleCoefficients, benchmark_path: str
) -> TurtleResult:
    """Admit the data, translate the thesis, simulate every cell."""
    axes, cells = entry_cells(thesis, md)
    data = preflight(md, benchmark_path, coefficients)
    runs = [run_cell(cell, data, coefficients) for cell in cells]
    return TurtleResult(
        axes=axes,
        cells=runs,
        data=data,
        coefficients=coefficients,
        budget_per_target=coefficients.equity / len(md.targets),
    )
