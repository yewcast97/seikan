"""Shared inputs for the turtle tests: explicit OHLC bar files, the rules' worked example, the
smallest theses that fire on a chosen bar, and coefficient documents — frictionless by default,
so the rules' worked-example numbers hold, with the realistic cost model one call away."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: (open, high, low, close)
Row = tuple[float, float, float, float]

__all__ = [
    "Row",
    "coefficients_doc",
    "first_true_above",
    "flat_rows",
    "frictionless",
    "thesis_doc",
    "worked_example_rows",
    "write_bars",
]


def write_bars(
    path: Path,
    rows: list[Row],
    *,
    start: str = "2020-01-01",
    freq: str = "1D",
    volume: float | None = 1000.0,
) -> Path:
    """Write explicit OHLC rows as a strict OHLCV CSV on a regular clock."""
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    if volume is not None:
        df["volume"] = volume
    df.index.name = "datetime"
    df.to_csv(path)
    return path


def flat_rows(n: int, close: float, half: float) -> list[Row]:
    """``n`` bars with open == close and a symmetric range of ``half`` (true range ``2·half``)."""
    return [(close, close + half, close - half, close)] * n


def worked_example_rows() -> list[Row]:
    """The rules' worked example (N = 2 throughout the ladder): 24 flat bars at 49.5, the firing
    bar closing at 50, opens at 50 / 51 / 52 (entry, add, add), a breakdown through the $48
    stop, then the exit open at 47 — 31 bars."""
    return [
        *flat_rows(24, 49.5, 1.0),
        (49.5, 51.0, 49.0, 50.0),
        (50.0, 52.0, 50.0, 51.0),
        (51.0, 53.0, 51.0, 52.0),
        (52.0, 53.5, 51.5, 52.5),
        (52.5, 52.5, 47.5, 47.5),
        (47.0, 47.5, 46.5, 47.0),
        (47.0, 48.0, 46.0, 47.0),
    ]


def first_true_above(level: float) -> dict:
    """``first_true(close >= level)`` — fires once, on the first close at or above ``level``."""
    return {
        "type": "first_true",
        "condition": {
            "type": "threshold",
            "left": {"type": "field", "column": "close"},
            "op": ">=",
            "right": {"type": "constant", "value": level},
        },
    }


def thesis_doc(targets: list[str], entry: dict, **params: object) -> dict:
    return {
        "name": "turtle-probe",
        "data": {"targets": targets},
        "entry": entry,
        "params": {"horizon": 5, **params},
    }


def frictionless(**overrides: object) -> dict:
    """Every cost switched off, every field spelled so a new default can never drift in;
    ``overrides`` replace whole sub-blocks (``commission=…``) or ``stop_shock``."""
    return {
        "commission": {
            "per_share": 0.0,
            "min_per_order": 0.0,
            "bps": 0.0,
            "sell_bps": 0.0,
            "cap_bps": None,
        },
        "slippage": {"bps": 0.0, "n_fraction": 0.0},
        "impact": {"coefficient": 0.0, "adv_window": 20},
        "stop_shock": 0.0,
        **overrides,
    }


def coefficients_doc(**overrides: object) -> dict:
    """A frictionless coefficients document over $100,000; ``costs=`` swaps the cost model."""
    return {"equity": 100_000.0, "costs": frictionless(), **overrides}
