"""Strict frames → the simulation's inputs.

:func:`preflight` turns a loaded :class:`~seikan.api.MarketData` plus the benchmark CSV into
:class:`SimulationData`: every price quantized ONCE to the coefficient's grid (the engine's own
rounding, so the fills, the levels and the report all hold the same grid values), the benchmark
read through the same strict reader and required to cover the target clock exactly, the volume
admitted only when the cost model's market impact needs it, and every hole refused — a simulation
needs a price on every bar, so what the event study merely censors is exit 2 here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from seikan import _turtle, dataio
from seikan.api import MarketData
from seikan.turtle.coefficients import TurtleCoefficients
from seikan.types import DataIssue, DataReport, FileReportEntry

_OHLC = ("open", "high", "low", "close")


@dataclass(frozen=True)
class QuantizedBars:
    """One instrument's OHLC on the price grid (float64 arrays over the joined index), plus the
    raw volume when the cost model reads it (``None`` otherwise)."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray | None = None


@dataclass(frozen=True)
class SimulationData:
    """Everything a cell's simulation consumes, target by target, plus the benchmark."""

    index: pd.DatetimeIndex
    precision: int
    targets: dict[str, QuantizedBars]
    benchmark: QuantizedBars
    benchmark_path: str
    #: The loader's report, extended with the benchmark file's strict-read entry.
    data_report: DataReport


def quantize_array(values: np.ndarray, precision: int) -> np.ndarray:
    """Round every price with the engine's own rounding (half away from zero) so the fills, the
    levels and the report all hold the same grid values."""
    return np.array([_turtle.quantize(float(v), precision) for v in values], dtype=float)


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


def _volume_issues(md: MarketData) -> list[DataIssue]:
    """Market impact divides by an average volume: every target must carry a volume column that
    is finite and strictly positive on every bar."""
    if md.volume is None:
        return [
            {
                "code": "spec_data_mismatch",
                "message": "costs.impact is enabled, so every target needs a volume column (the "
                "average volume the impact law divides by), but the data carries none",
            }
        ]
    issues: list[DataIssue] = []
    for target in md.targets:
        v = md.volume[target].to_numpy(dtype=float)
        holes = int(np.count_nonzero(~np.isfinite(v)))
        if holes:
            issues.append(
                {
                    "code": "nan_fraction",
                    "message": f"target {target!r} has {holes} missing volume cells over the "
                    "joined bars; market impact needs a volume on every bar",
                    "column": target,
                    "value": float(holes),
                }
            )
        zeros = int(np.count_nonzero(np.isfinite(v) & (v <= 0.0)))
        if zeros:
            issues.append(
                {
                    "code": "integrity",
                    "message": f"target {target!r} has {zeros} bars with non-positive volume; "
                    "market impact needs a positive average volume on every bar",
                    "column": target,
                    "value": float(zeros),
                }
            )
    return issues


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
    reads_volume = coefficients.costs.impact.coefficient > 0
    if reads_volume:
        issues.extend(_volume_issues(md))
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
    targets: dict[str, QuantizedBars] = {}
    for t in md.targets:
        o, h, lo, c = (
            quantize_array(md.field(col)[t].to_numpy(dtype=float), precision) for col in _OHLC
        )
        volume = None
        if reads_volume and md.volume is not None:
            volume = md.volume[t].to_numpy(dtype=float)
        targets[t] = QuantizedBars(o, h, lo, c, volume=volume)
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
        precision=precision,
        targets=targets,
        benchmark=benchmark,
        benchmark_path=benchmark_path,
        data_report=report,
    )
