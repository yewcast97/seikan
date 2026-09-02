"""Shared helpers for the suite: the data-binding pair, the synthetic-CSV writers, the smallest
condition builders, and the gate-result readers that several modules used to re-declare.

A thesis NAMES its series and never locates them, so materializing market data is two steps: bind
the declared keys to files (``resolve_data_files``), then load. Every test that needs data does the
same two steps, so they live here rather than being re-spelled per module — and going through the
real resolver means a test whose mapping drifts from its DSL fails the way the CLI would, instead
of loading something the thesis never asked for.

Locating a series takes TWO facts, so both helpers take two mappings: which file answers a key
(``mapping``, always required) and — for the keys whose file holds several numeric columns — which
column of it the key reads (``columns``, the ``--column KEY=COL`` half, optional because most files
name their own single value column). Passing them separately here rather than folding a column into
the path is the point: the suite types what a caller types, so a binding this suite accepts is one
the CLI accepts.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from seikan.api import DataFiles, MarketData, load_market_data, resolve_data_files
from seikan.dsl.schema import Thesis
from seikan.gate import CellReport, GateReport

__all__ = [
    "DataFiles",
    "bars",
    "files_for",
    "keys_deep",
    "load",
    "lt",
    "unmet_cell_doc",
    "unmet_cell_report",
    "unmet_checks",
    "unmet_run_doc",
    "unmet_run_report",
    "write_ohlcv",
]


def files_for(
    thesis: Thesis, mapping: dict[str, str], columns: dict[str, str] | None = None
) -> DataFiles:
    """Bind ``{key: path}`` (and any ``{key: column}``) to the thesis's declared data keys,
    refusing a mismatch."""
    return resolve_data_files(
        thesis,
        {k: str(v) for k, v in mapping.items()},
        {k: str(v) for k, v in (columns or {}).items()},
    )


def load(
    thesis: Thesis, mapping: dict[str, str], columns: dict[str, str] | None = None
) -> MarketData:
    """``resolve_data_files`` + ``load_market_data`` — exactly what ``seikan run`` does."""
    return load_market_data(thesis.data, files_for(thesis, mapping, columns))


# ---- synthetic inputs ---------------------------------------------------------------------


def write_ohlcv(path: Path, n: int = 400, seed: int = 0) -> Path:
    """A random-walk OHLCV file with real intrabar ranges (``high >= max(open, close)``,
    ``low <= min(open, close)``) and a flat volume, on a daily clock from 2018."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="1D")
    s = pd.Series(100 * np.exp(np.cumsum(rng.randn(n) * 0.01)), index=idx)
    o = s.shift(1).bfill()
    df = pd.DataFrame(
        {
            "open": o,
            "high": np.maximum(s * 1.01, o),
            "low": np.minimum(s * 0.99, o),
            "close": s,
            "volume": 1000.0,
        },
        index=idx,
    )
    df.index.name = "datetime"
    df.to_csv(path)
    return path


def bars(path: Path, closes: list[float]) -> Path:
    """Bars with ``open == high == low == close`` — the exact-fill fixture — on a daily clock
    from 2021."""
    idx = pd.date_range("2021-01-01", periods=len(closes), freq="1D")
    s = pd.Series(closes, index=idx, dtype=float)
    df = pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 1000.0}, index=idx)
    df.index.name = "datetime"
    df.to_csv(path)
    return path


def lt(v: float) -> dict:
    """The smallest entry condition: ``close < v``."""
    return {
        "type": "threshold",
        "left": {"type": "field", "column": "close"},
        "op": "<",
        "right": {"type": "constant", "value": v},
    }


# ---- readers ------------------------------------------------------------------------------


def keys_deep(obj: object) -> Iterator[str]:
    """Every dict key anywhere in a serialized report — proves a concept ABSENT, not merely
    unset at the top level."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from keys_deep(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from keys_deep(v)


def unmet_run_doc(doc: dict) -> list[str]:
    """The unmet run-level check names of an emitted report document."""
    return [c["name"] for c in doc["gate"]["run_checks"] if not c["met"]]


def unmet_checks(cell: dict) -> list[str]:
    """The unmet check names of one emitted gate cell (``gate.cells[i]``)."""
    return [c["name"] for c in cell["checks"] if not c["met"]]


def unmet_cell_doc(doc: dict, index: int = 0) -> list[str]:
    """The unmet per-cell check names of one cell of an emitted report document."""
    return unmet_checks(doc["gate"]["cells"][index])


def unmet_run_report(report: GateReport) -> list[str]:
    """The unmet run-level check names of an in-memory :class:`GateReport`."""
    return [c.name for c in report.run_checks if not c.met]


def unmet_cell_report(cell: CellReport) -> list[str]:
    """The unmet check names of one in-memory :class:`CellReport`."""
    return [c.name for c in cell.checks if not c.met]
