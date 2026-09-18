"""Strict frames → the simulation's inputs.

Two jobs. :func:`preflight` turns a loaded :class:`~seikan.api.MarketData` plus the benchmark
CSV into :class:`SimulationData`: every price quantized ONCE to the coefficient's grid (the
kernel's own rounding, the same the venue applies), the benchmark read through the same strict
reader and required to cover the target clock exactly, and every hole refused — a simulation
needs a price on every bar, so what the event study merely censors is exit 2 here. The
nautilus_trader builders then make the venue's objects out of those arrays: positional
instruments (target names may hold characters an instrument id may not), bars with unlimited
volume, and one opening print per bar one nanosecond before it, so a market order submitted at
that print fills at the bar's open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from seikan import _turtle, dataio
from seikan.api import MarketData
from seikan.turtle.coefficients import TurtleCoefficients
from seikan.types import DataIssue, DataReport, FileReportEntry

#: The venue every simulated instrument trades on.
VENUE = "SIM"
#: The bar volume and print sizes handed to the venue: liquidity never runs out at the quoted
#: price, so a market order fills whole at the open and a stop fills whole at its trigger.
UNLIMITED_LIQUIDITY = 10**9
#: The opening print sits this many nanoseconds before its bar's stamp.
OPEN_PRINT_OFFSET_NS = 1
#: The bar-type label under which every clock is fed (external bars are opaque to the venue;
#: the report stamps the real spacing).
BAR_TYPE_SPEC = "1-DAY-LAST-EXTERNAL"

_OHLC = ("open", "high", "low", "close")


@dataclass(frozen=True)
class QuantizedBars:
    """One instrument's OHLC on the price grid (float64 arrays over the joined index)."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray


@dataclass(frozen=True)
class SimulationData:
    """Everything a cell's backtest consumes, target by target, plus the benchmark."""

    index: pd.DatetimeIndex
    ts_ns: np.ndarray
    precision: int
    targets: dict[str, QuantizedBars]
    benchmark: QuantizedBars
    benchmark_path: str
    #: The loader's report, extended with the benchmark file's strict-read entry.
    data_report: DataReport


def quantize_array(values: np.ndarray, precision: int) -> np.ndarray:
    """Round every price with the kernel's own rounding (half away from zero) so the reference
    simulator, the venue's ``Price`` and the report all hold the same grid values."""
    return np.array([_turtle.quantize(float(v), precision) for v in values], dtype=float)


def index_ns(index: pd.DatetimeIndex) -> np.ndarray:
    """The bar stamps as nanoseconds since the epoch (naive stamps read as UTC)."""
    stamps: np.ndarray = index.values.astype("datetime64[ns]").astype("int64")
    return stamps


def _refuse(report: DataReport, issues: list[DataIssue], message: str) -> None:
    refused: dict[str, object] = dict(report)
    refused["errors"] = [*report.get("errors", []), *issues]
    refused["ok"] = False
    raise dataio.DataError(refused, message)  # type: ignore[arg-type]


def _base_report(md: MarketData) -> DataReport:
    if md.report is not None:
        return md.report
    # A hand-built MarketData carries no report; the refusal payload is honestly partial.
    return {"ok": True, "files": [], "join": None, "errors": []}


def preflight(
    md: MarketData, benchmark_path: str, coefficients: TurtleCoefficients
) -> SimulationData:
    """Admit the loaded data and the benchmark for simulation, or raise
    :class:`~seikan.dataio.DataError` (exit 2) naming what fails."""
    report = _base_report(md)
    issues: list[DataIssue] = []
    if md.target_shape != "ohlcv":
        issues.append(
            {
                "code": "spec_data_mismatch",
                "message": "stock-turtle-trade-long-only needs OHLCV targets (an ATR, a lowest-low "
                f"channel and opening prints), but the targets are {md.target_shape}-shaped",
            }
        )
        _refuse(report, issues, issues[0]["message"])
    floor = max(coefficients.atr_period, coefficients.exit_lookback) + 1
    n_bars = len(md.index)
    if n_bars < floor:
        issues.append(
            {
                "code": "insufficient_common_index",
                "message": f"{n_bars} bars, but the turtle needs at least {floor} (the ATR period "
                "or the exit lookback plus one) before a single entry can be taken",
            }
        )
    for target in md.targets:
        holes = int(sum(md.field(col)[target].isna().sum() for col in _OHLC))
        if holes:
            issues.append(
                {
                    "code": "nan_fraction",
                    "message": f"target {target!r} has {holes} missing OHLC cells over the joined "
                    "bars; the simulation needs a price on every bar",
                    "column": target,
                    "value": float(holes),
                }
            )
    frame, file_report = dataio.read_strict_csv(
        benchmark_path, role="benchmark", expected_shape="ohlcv"
    )
    entry: FileReportEntry = file_report.to_dict()
    already = any(
        f["role"] == "benchmark" and f["path"] == entry["path"] for f in report.get("files", [])
    )
    if not already:
        report = {**report, "files": [*report.get("files", []), entry]}
    if frame is None or not file_report.ok:
        issues.append(
            {
                "code": "benchmark_coverage",
                "message": f"the benchmark file {benchmark_path!r} failed the strict read",
            }
        )
        _refuse(report, issues, issues[-1]["message"])
    assert frame is not None  # narrowed by the refusal above; `python -O` still cannot reach it
    missing = md.index.difference(frame.index)
    if len(missing):
        first = missing[0].isoformat()
        issues.append(
            {
                "code": "benchmark_coverage",
                "message": f"the benchmark is missing {len(missing)} of the {n_bars} target bars "
                f"(first {first}); a buy-and-hold curve needs the index on every bar",
                "value": float(len(missing)),
            }
        )
    else:
        aligned = frame.reindex(md.index)
        holes = int(sum(aligned[col].isna().sum() for col in _OHLC))
        if holes:
            issues.append(
                {
                    "code": "nan_fraction",
                    "message": f"the benchmark has {holes} missing OHLC cells over the joined bars",
                    "column": "benchmark",
                    "value": float(holes),
                }
            )
    if issues:
        _refuse(report, issues, issues[0]["message"])
    aligned = frame.reindex(md.index)
    precision = coefficients.price_precision
    targets = {
        t: QuantizedBars(
            *(quantize_array(md.field(col)[t].to_numpy(dtype=float), precision) for col in _OHLC)
        )
        for t in md.targets
    }
    benchmark = QuantizedBars(
        *(quantize_array(aligned[col].to_numpy(dtype=float), precision) for col in _OHLC)
    )
    for name, bars in [*targets.items(), ("benchmark", benchmark)]:
        if min(bars.open.min(), bars.high.min(), bars.low.min(), bars.close.min()) <= 0.0:
            issues.append(
                {
                    "code": "integrity",
                    "message": f"{name!r} has a price that quantizes to zero or below at "
                    f"{precision} decimals; the price grid cannot represent it",
                    "column": name,
                }
            )
    if issues:
        _refuse(report, issues, issues[0]["message"])
    return SimulationData(
        index=md.index,
        ts_ns=index_ns(md.index),
        precision=precision,
        targets=targets,
        benchmark=benchmark,
        benchmark_path=benchmark_path,
        data_report=report,
    )


# ---- nautilus_trader objects ---------------------------------------------------------------


def instrument_symbol(position: int) -> str:
    """The positional symbol a target trades under (``T0``, ``T1``, …): the report always speaks
    in target names, the venue never sees one."""
    return f"T{position}"


def instrument_for(position: int, currency: str, precision: int) -> Any:
    """The nautilus ``Equity`` for the target at ``position``: whole shares, a grid of
    ``10^-precision``."""
    from nautilus_trader.model import Currency, Equity, InstrumentId, Price, Quantity, Symbol

    symbol = instrument_symbol(position)
    return Equity(
        instrument_id=InstrumentId.from_str(f"{symbol}.{VENUE}"),
        raw_symbol=Symbol(symbol),
        currency=Currency.from_str(currency),
        price_precision=precision,
        price_increment=Price.from_str(f"{10.0**-precision:.{precision}f}"),
        ts_event=0,
        ts_init=0,
        lot_size=Quantity.from_int(1),
    )


def bar_type_for(instrument: Any) -> Any:
    from nautilus_trader.model import BarType

    return BarType.from_str(f"{instrument.id}-{BAR_TYPE_SPEC}")


def venue_data(instrument: Any, bars: QuantizedBars, ts_ns: np.ndarray) -> list[Any]:
    """The venue's data for one instrument: a bar per stamp (unlimited volume) and, one
    nanosecond before each, the opening print — a quote at the open with unlimited size on both
    sides, so a market order submitted at the print fills whole at the open and a resting stop
    the print gaps through fills at the open too."""
    from nautilus_trader.model import Bar, Quantity, QuoteTick

    bar_type = bar_type_for(instrument)
    size = Quantity.from_int(UNLIMITED_LIQUIDITY)
    out: list[Any] = []
    for t, ts in enumerate(ts_ns.tolist()):
        open_px = instrument.make_price(float(bars.open[t]))
        out.append(
            QuoteTick(
                instrument.id,
                open_px,
                open_px,
                size,
                size,
                ts - OPEN_PRINT_OFFSET_NS,
                ts - OPEN_PRINT_OFFSET_NS,
            )
        )
        out.append(
            Bar(
                bar_type=bar_type,
                open=open_px,
                high=instrument.make_price(float(bars.high[t])),
                low=instrument.make_price(float(bars.low[t])),
                close=instrument.make_price(float(bars.close[t])),
                volume=size,
                ts_event=ts,
                ts_init=ts,
            )
        )
    return out
