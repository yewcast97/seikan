"""One engine run per (cell × target), and the run's orchestration.

Every cell (entry combo) runs each target through the Rust engine (``seikan._turtle.simulate``)
on its own sub-account with a fixed budget of ``equity / n_targets``; the portfolio is the
sub-accounts summed bar by bar. The engine keeps the only set of books, prices every fill under
the coefficients' cost model and reports every fill, round trip and per-bar sample back here.
"""

from __future__ import annotations

from dataclasses import dataclass

from seikan import _turtle
from seikan.api import MarketData
from seikan.dsl.schema import Thesis
from seikan.turtle.coefficients import TurtleCoefficients
from seikan.turtle.market import QuantizedBars, SimulationData, preflight
from seikan.turtle.signals import EntryCell, entry_cells


@dataclass(frozen=True)
class TargetSamples:
    """The per-bar samples the engine takes after each close (plain lists, JSON-exact)."""

    equity: list[float]
    cash: list[float]
    shares: list[int]
    units: list[int]
    stop: list[float]
    add_level: list[float]
    atr: list[float]
    channel: list[float]
    commission_cum: list[float]
    slippage_cum: list[float]
    shock_cum: list[float]
    impact_cum: list[float]


@dataclass(frozen=True)
class CostsPaid:
    """The cost buckets over EVERY fill of a run (the cash fact, open positions included)."""

    commission: float
    slippage: float
    shock: float
    impact: float

    @property
    def total(self) -> float:
        return self.commission + self.slippage + self.shock + self.impact


@dataclass(frozen=True)
class TargetRun:
    """One target's simulation inside a cell."""

    target: str
    fills: list[_turtle.Fill]
    trips: list[_turtle.RoundTrip]
    open_trip: _turtle.RoundTrip | None
    ledger: dict[str, int]
    samples: TargetSamples
    costs_paid: CostsPaid
    end_cash: float
    end_shares: int
    end_units: int
    end_stop: float | None
    end_add_level: float | None


@dataclass(frozen=True)
class CellRun:
    """One cell's complete simulation."""

    cell: EntryCell
    targets: dict[str, TargetRun]


@dataclass(frozen=True)
class TurtleResult:
    """The whole run: the admitted data, every cell, the resolved coefficients."""

    axes: list[str]
    cells: list[CellRun]
    data: SimulationData
    coefficients: TurtleCoefficients
    budget_per_target: float


def kernel_coefficients(c: TurtleCoefficients, budget: float) -> _turtle.Coefficients:
    """The engine's rule coefficients for one target with cash budget ``budget``."""
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


def kernel_costs(c: TurtleCoefficients) -> _turtle.CostModel:
    """The engine's cost model from the coefficients' ``costs`` block."""
    k = c.costs
    return _turtle.CostModel(
        per_share=k.commission.per_share,
        min_per_order=k.commission.min_per_order,
        bps=k.commission.bps,
        sell_bps=k.commission.sell_bps,
        cap_bps=k.commission.cap_bps,
        slippage_bps=k.slippage.bps,
        slippage_n_fraction=k.slippage.n_fraction,
        impact_coefficient=k.impact.coefficient,
        adv_window=k.impact.adv_window,
        stop_shock=k.stop_shock,
    )


def engine_version() -> str:
    """The engine's version (the ``seikan-turtle`` crate version)."""
    return _turtle.version()


def run_target(
    target: str,
    bars: QuantizedBars,
    fired: list[bool],
    kernel: _turtle.Coefficients,
    costs: _turtle.CostModel,
) -> TargetRun:
    """Run one target through the engine and shape its result."""
    res = _turtle.simulate(
        kernel,
        costs,
        bars.open.tolist(),
        bars.high.tolist(),
        bars.low.tolist(),
        bars.close.tolist(),
        None if bars.volume is None else bars.volume.tolist(),
        fired,
    )
    n_bars = len(bars.close)
    if len(res.equity) != n_bars:
        raise RuntimeError(f"{target}: {len(res.equity)} bars sampled of {n_bars} fed")
    samples = TargetSamples(
        equity=res.equity,
        cash=res.cash,
        shares=res.shares,
        units=res.units,
        stop=res.stop,
        add_level=res.add_level,
        atr=res.atr,
        channel=res.channel,
        commission_cum=res.commission_cum,
        slippage_cum=res.slippage_cum,
        shock_cum=res.shock_cum,
        impact_cum=res.impact_cum,
    )
    last = n_bars - 1
    return TargetRun(
        target=target,
        fills=list(res.fills),
        trips=list(res.trips),
        open_trip=res.open_trip,
        ledger=res.ledger.to_dict(),
        samples=samples,
        costs_paid=CostsPaid(
            commission=res.commission_cum[last],
            slippage=res.slippage_cum[last],
            shock=res.shock_cum[last],
            impact=res.impact_cum[last],
        ),
        end_cash=res.cash[last],
        end_shares=res.shares[last],
        end_units=res.units[last],
        end_stop=None if res.open_trip is None else res.stop[last],
        end_add_level=None if res.open_trip is None else res.add_level[last],
    )


def run_cell(cell: EntryCell, data: SimulationData, c: TurtleCoefficients) -> CellRun:
    """Simulate one cell: every target on its own sub-account."""
    targets = list(data.targets)
    kernel = kernel_coefficients(c, c.equity / len(targets))
    costs = kernel_costs(c)
    runs = {
        t: run_target(t, data.targets[t], cell.fired[t].tolist(), kernel, costs) for t in targets
    }
    return CellRun(cell=cell, targets=runs)


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
