"""From a :class:`~seikan.turtle.engine.TurtleResult` to the report's sections and the three CSV
frames. Sections are plain dicts in the shapes ``seikan.types.turtle`` declares; the CLI adds the
header, identity, data report and outputs around them and validates the whole document at
emission."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from seikan import _turtle, dataio
from seikan.contract import TURTLE_FILL_CONVENTIONS
from seikan.serialize import atomic_output
from seikan.turtle.engine import (
    CellRun,
    TargetRun,
    TurtleResult,
    engine_version,
    kernel_coefficients,
)
from seikan.turtle.metrics import (
    bar_returns,
    cagr,
    costs_block,
    excursions,
    performance,
    periodic,
    relative,
    sum_ledgers,
    trade_stats,
)
from seikan.types.turtle import (
    BenchmarkBlock,
    CostsPaid,
    PortfolioPanel,
    SimulationBlock,
    TargetPanel,
    TurtleCell,
)

#: The thesis parameters the simulation reads nothing from (the turtle owns its exit).
THESIS_PARAMS_IGNORED = ["horizon", "outcome", "benchmark", "features"]

BENCHMARK_CONSTRUCTION = (
    "buy-and-hold of the index: starting_equity / open[0] units bought at the first bar's open, "
    "marked at every close, frictionless (a reference path, not a traded one); the bar before the "
    "first is the starting equity, like the strategy"
)


@dataclass(frozen=True)
class ReportSections:
    """The run-level and per-cell sections in the report's fixed order."""

    simulation: SimulationBlock
    targets: list[str]
    params: list[str]
    n_cells: int
    benchmark: BenchmarkBlock
    cells: list[TurtleCell]


@dataclass(frozen=True)
class _Benchmark:
    units: float
    equity: np.ndarray
    returns: np.ndarray
    cagr: float | None


def _benchmark(result: TurtleResult) -> _Benchmark:
    c = result.coefficients
    b = result.data.benchmark
    units = c.equity / float(b.open[0])
    equity = units * b.close
    returns = bar_returns(equity, c.equity)
    growth = cagr(c.equity, float(equity[-1]), len(equity), c.bars_per_year)
    return _Benchmark(units=units, equity=equity, returns=returns, cagr=growth)


def _paid(runs: list[TargetRun]) -> CostsPaid:
    return costs_block(
        float(sum(r.costs_paid.commission for r in runs)),
        float(sum(r.costs_paid.slippage for r in runs)),
        float(sum(r.costs_paid.shock for r in runs)),
        float(sum(r.costs_paid.impact for r in runs)),
    )


def _target_panel(
    run: TargetRun, result: TurtleResult, bench: _Benchmark, budget: float
) -> TargetPanel:
    c = result.coefficients
    bars = result.data.targets[run.target]
    equity = np.asarray(run.samples.equity, dtype=float)
    shares = np.asarray(run.samples.shares, dtype=float)
    gross = shares * bars.close / equity
    metrics = performance(equity, budget, c.bars_per_year, result.data.index, shares > 0, gross)
    pairs = [excursions(t, bars.high, bars.low) for t in run.trips]
    n_open = 0 if run.open_trip is None else 1
    return {
        "metrics": metrics,
        "relative": relative(
            bar_returns(equity, budget), bench.returns, c.bars_per_year, metrics["cagr"], bench.cagr
        ),
        "trades": trade_stats(run.trips, n_open, run.ledger, pairs),
        "end_state": {
            "shares": run.end_shares,
            "units": run.end_units,
            "cash": run.end_cash,
            "market_value": run.end_shares * float(bars.close[-1]),
            "stop": run.end_stop,
            "add_level": run.end_add_level,
            "in_position": run.end_shares > 0,
            "costs_paid": _paid([run]),
        },
    }


def _portfolio_panel(cell: CellRun, result: TurtleResult, bench: _Benchmark) -> PortfolioPanel:
    c = result.coefficients
    runs = list(cell.targets.values())
    equity = np.sum([np.asarray(r.samples.equity, dtype=float) for r in runs], axis=0)
    shares = np.array([np.asarray(r.samples.shares, dtype=float) for r in runs])
    closes = np.array([result.data.targets[r.target].close for r in runs])
    market_value = np.sum(shares * closes, axis=0)
    gross = market_value / equity
    in_market = np.any(shares > 0, axis=0)
    metrics = performance(equity, c.equity, c.bars_per_year, result.data.index, in_market, gross)
    returns = bar_returns(equity, c.equity)
    trips: list[_turtle.RoundTrip] = []
    pairs: list[tuple[float, float]] = []
    for r in runs:
        bars = result.data.targets[r.target]
        trips.extend(r.trips)
        pairs.extend(excursions(t, bars.high, bars.low) for t in r.trips)
    n_open = sum(1 for r in runs if r.open_trip is not None)
    return {
        "metrics": metrics,
        "relative": relative(returns, bench.returns, c.bars_per_year, metrics["cagr"], bench.cagr),
        "periodic": periodic(returns, result.data.index),
        "trades": trade_stats(trips, n_open, sum_ledgers([r.ledger for r in runs]), pairs),
        "costs_paid": _paid(runs),
    }


def _cell_section(cell: CellRun, result: TurtleResult, bench: _Benchmark) -> TurtleCell:
    budget = result.budget_per_target
    return {
        "cell_id": cell.cell.cell_id,
        "params": dict(cell.cell.params),
        "portfolio": _portfolio_panel(cell, result, bench),
        "by_target": {
            target: _target_panel(run, result, bench, budget)
            for target, run in cell.targets.items()
        },
    }


def report_sections(result: TurtleResult) -> ReportSections:
    """Every section after ``outputs``, in order."""
    c = result.coefficients
    data = result.data
    bench = _benchmark(result)
    n = len(data.index)
    targets = list(data.targets)
    simulation: SimulationBlock = {
        "engine": "seikan._turtle",
        "engine_version": engine_version(),
        "currency": c.currency,
        "starting_equity": c.equity,
        "n_targets": len(targets),
        "budget_per_target": result.budget_per_target,
        "budget_mode": "fixed",
        "price_precision": c.price_precision,
        "price_increment": 10.0**-c.price_precision,
        "lot_size": 1,
        "liquidity": "sqrt_impact" if c.costs.impact.coefficient > 0 else "unlimited",
        "fill_conventions": {k: str(v) for k, v in TURTLE_FILL_CONVENTIONS.items()},
        "bars_per_year": c.bars_per_year,
        "n_bars": n,
        "index_start": data.index[0].isoformat(),
        "index_end": data.index[-1].isoformat(),
        "bar_spacing": dataio.bar_spacing(data.index),
        "first_eligible_bar": kernel_coefficients(c, result.budget_per_target).first_eligible_bar(),
        "thesis_params_ignored": list(THESIS_PARAMS_IGNORED),
    }
    all_in = np.ones(n, dtype=bool)
    benchmark: BenchmarkBlock = {
        "source": data.benchmark_path,
        "construction": BENCHMARK_CONSTRUCTION,
        "units": bench.units,
        "start_equity": c.equity,
        "end_equity": float(bench.equity[-1]),
        "metrics": performance(
            bench.equity, c.equity, c.bars_per_year, data.index, all_in, np.ones(n)
        ),
        "periodic": periodic(bench.returns, data.index),
    }
    return ReportSections(
        simulation=simulation,
        targets=targets,
        params=list(result.axes),
        n_cells=len(result.cells),
        benchmark=benchmark,
        cells=[_cell_section(cell, result, bench) for cell in result.cells],
    )


# ---- the CSV frames -------------------------------------------------------------------------


def round_trips_frame(result: TurtleResult) -> pd.DataFrame:
    """One row per round trip — closed ones and the open end-of-data mark — over every cell."""
    index = result.data.index
    rows = []
    for cell in result.cells:
        axes = {k: cell.cell.params[k] for k in result.axes}
        for target, run in cell.targets.items():
            bars = result.data.targets[target]
            trips = [*run.trips, *([run.open_trip] if run.open_trip is not None else [])]
            for t in trips:
                mae, mfe = excursions(t, bars.high, bars.low)
                rows.append(
                    {
                        **axes,
                        "target": target,
                        "entry_bar": t.entry_bar,
                        "entry_time": index[t.entry_bar],
                        "exit_bar": t.exit_bar,
                        "exit_time": index[t.exit_bar],
                        "exit_reason": t.exit_reason,
                        "is_open": int(t.is_open),
                        "entry_px": t.entry_px,
                        "avg_entry_px": t.cost_basis / t.shares,
                        "exit_px": t.exit_px,
                        "unit_shares": t.unit_shares,
                        "units": t.units,
                        "shares": t.shares,
                        "cost_basis": t.cost_basis,
                        "proceeds": t.proceeds,
                        "gross_pnl": t.gross_pnl,
                        "commission": t.commission,
                        "slippage": t.slippage,
                        "shock": t.shock,
                        "impact": t.impact,
                        "pnl": t.pnl,
                        "ret": t.pnl / t.cost_basis,
                        "bars_held": t.exit_bar - t.entry_bar,
                        "n_adds": t.n_adds,
                        "adds_skipped_budget": t.adds_skipped_budget,
                        "max_stop": t.max_stop,
                        "mae": mae,
                        "mfe": mfe,
                    }
                )
    columns = [
        *result.axes,
        "target",
        "entry_bar",
        "entry_time",
        "exit_bar",
        "exit_time",
        "exit_reason",
        "is_open",
        "entry_px",
        "avg_entry_px",
        "exit_px",
        "unit_shares",
        "units",
        "shares",
        "cost_basis",
        "proceeds",
        "gross_pnl",
        "commission",
        "slippage",
        "shock",
        "impact",
        "pnl",
        "ret",
        "bars_held",
        "n_adds",
        "adds_skipped_budget",
        "max_stop",
        "mae",
        "mfe",
    ]
    return pd.DataFrame(rows, columns=columns)


def fills_frame(result: TurtleResult) -> pd.DataFrame:
    """One row per fill over every cell, with the engine's books right after it."""
    index = result.data.index
    rows = []
    for cell in result.cells:
        axes = {k: cell.cell.params[k] for k in result.axes}
        for target, run in cell.targets.items():
            for f in run.fills:
                rows.append(
                    {
                        **axes,
                        "target": target,
                        "bar": f.bar,
                        "datetime": index[f.bar],
                        "at": f.at,
                        "kind": f.kind,
                        "reason": f.reason,
                        "side": "sell" if f.kind == "exit" else "buy",
                        "shares": f.shares,
                        "reference": f.reference,
                        "price": f.price,
                        "notional": f.shares * f.price,
                        "commission": f.commission,
                        "slippage": f.slippage,
                        "shock": f.shock,
                        "impact": f.impact,
                        "cash_after": f.cash_after,
                        "shares_after": f.shares_after,
                        "units_after": f.units_after,
                        "stop_after": f.stop_after,
                    }
                )
    columns = [
        *result.axes,
        "target",
        "bar",
        "datetime",
        "at",
        "kind",
        "reason",
        "side",
        "shares",
        "reference",
        "price",
        "notional",
        "commission",
        "slippage",
        "shock",
        "impact",
        "cash_after",
        "shares_after",
        "units_after",
        "stop_after",
    ]
    return pd.DataFrame(rows, columns=columns)


def equity_frame(result: TurtleResult) -> pd.DataFrame:
    """One row per bar per cell: the portfolio curve beside the benchmark's, the cumulative cost
    buckets, and the per-target position state (``@<target>`` suffixed when several targets
    run)."""
    index = result.data.index
    bench = _benchmark(result)
    targets = list(result.data.targets)
    multi = len(targets) > 1
    frames = []
    for cell in result.cells:
        runs = [cell.targets[t] for t in targets]
        equity = np.sum([np.asarray(r.samples.equity, dtype=float) for r in runs], axis=0)
        cash = np.sum([np.asarray(r.samples.cash, dtype=float) for r in runs], axis=0)
        shares = np.array([np.asarray(r.samples.shares, dtype=float) for r in runs])
        closes = np.array([result.data.targets[t].close for t in targets])
        market_value = np.sum(shares * closes, axis=0)
        columns: dict[str, object] = {
            **{k: cell.cell.params[k] for k in result.axes},
            "datetime": index,
            "equity": equity,
            "benchmark": bench.equity,
            "cash": cash,
            "market_value": market_value,
            "gross_exposure": market_value / equity,
            "n_positions": np.count_nonzero(shares > 0, axis=0),
        }
        for bucket in ("commission_cum", "slippage_cum", "shock_cum", "impact_cum"):
            columns[bucket] = np.sum(
                [np.asarray(getattr(r.samples, bucket), dtype=float) for r in runs], axis=0
            )
        for t, r in zip(targets, runs, strict=True):
            suffix = f"@{t}" if multi else ""
            columns[f"shares{suffix}"] = r.samples.shares
            columns[f"units{suffix}"] = r.samples.units
            columns[f"stop{suffix}"] = r.samples.stop
            columns[f"add_level{suffix}"] = r.samples.add_level
        frames.append(pd.DataFrame(columns))
    return pd.concat(frames, ignore_index=True)


def write_csv(frame: pd.DataFrame, path: str) -> int:
    """Write a frame atomically without its positional index; returns the row count."""
    with atomic_output(path) as tmp:
        frame.to_csv(tmp, index=False)
    return len(frame)
