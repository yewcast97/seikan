"""Pure data profiling — the engine behind ``seikan describe``.

``describe`` is a pure observer of FILES the way ``run`` is of THESES: it states what the bytes
it was handed contain — levels, changes, dispersion, range position, missingness — and MEASURES
NOTHING. It runs no entry condition, opens no observation, grades no checklist and supports no
thesis; nothing here selects, ranks, forecasts, or repairs. The moment a question pairs today's
description with what FOLLOWED, it is a thesis, and theses go through ``seikan run``.

Containment, by construction:

- plain numpy/pandas only — no numba, no RNG, and NOTHING imported from ``compiler/``,
  ``analysis/`` or ``gate/``: the one thing it shares with the run summary is the clock
  geometry stamp, :func:`seikan.dataio.bar_spacing`, which lives with the data layer so both
  emit it in one vocabulary;
- admission is :func:`seikan.dataio.read_strict_csv` — the SAME strict reader ``check-data``
  runs, called identically, so the ``data_report`` this module builds is byte-equal to
  ``check-data``'s over the same files (a refused file is refused for the same reason by both,
  and gets a stub profile here rather than a guess);
- refuse-never-repair: a NaN endpoint is NEVER repaired by skipping to the previous finite
  value, a window shorter than requested is NEVER silently shortened, and every refusal is an
  explicit ``reason`` (``insufficient_bars`` / ``endpoint_missing``) or ``ratio_reason``
  (``non_positive_endpoint``) in the block it refuses — present, never omitted;
- all three change algebras are emitted with domain-nulls (``pct``/``log`` require both
  endpoints strictly positive — the positivity rule), because choosing ONE
  algebra would be the engine deciding what a series IS;
- BOUNDED OUTPUT: no per-bar array ever rides the document — its size is independent of
  ``n_bars``. Windows are BAR counts, never days; ``bar_spacing`` states the clock.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from seikan.dataio import FileReport, bar_spacing, build_data_report, read_strict_csv
from seikan.types import (
    ChangeBlock,
    DescribeResult,
    FileProfile,
    FullSampleBlock,
    LastBarBlock,
    MissingnessBlock,
    RangeDistance,
    RangePositionBlock,
    RefusalStub,
    SeriesProfile,
    VolumeBlock,
    VolumeWindow,
)

#: Default trailing windows, in BARS (roughly: bar, week, month, quarter, half, year on a daily
#: clock — but the engine never interprets cadence; ``bar_spacing`` states the clock).
DEFAULT_WINDOWS = (1, 5, 21, 63, 126, 252)


def _finite(v: float) -> bool:
    return math.isfinite(v)


def _change_block(x: np.ndarray, w: int) -> ChangeBlock:
    """The w-bar change of the last level against the level w bars earlier.

    ``diff`` whenever both endpoints are finite; ``pct``/``log`` only when both are strictly
    positive (RATIO algebras — off a positive scale a percent change through zero returns a
    finite number with an inverted sign, so the domain refusal is the honest output). A NaN
    endpoint refuses as ``endpoint_missing`` and is never repaired by walking back to the
    previous finite value.
    """
    out = _empty_change_block()
    if len(x) < w + 1:
        out["reason"] = "insufficient_bars"
        return out
    a, b = float(x[-1 - w]), float(x[-1])
    if not (_finite(a) and _finite(b)):
        out["reason"] = "endpoint_missing"
        return out
    out["diff"] = b - a
    if a > 0 and b > 0:
        out["pct"] = b / a - 1.0
        out["log"] = math.log(b / a)
    else:
        out["ratio_reason"] = "non_positive_endpoint"
    return out


def _ratio(last: float, base: float) -> float | None:
    """``last/base − 1`` on a strictly positive pair, else ``None`` — the one positive-domain
    ratio read the range-position and full-sample blocks share (their ``ratio_reason`` says
    when any leg refused)."""
    if last > 0 and base > 0:
        return last / base - 1.0
    return None


def _empty_change_block() -> ChangeBlock:
    """The all-null change/dispersion skeleton both per-window blocks start from — one
    declaration of the key order."""
    return {"diff": None, "pct": None, "log": None, "reason": None, "ratio_reason": None}


def _dispersion_block(x: np.ndarray, w: int) -> ChangeBlock:
    """ddof=1 std of the w trailing 1-bar changes (the same w+1 trailing levels
    ``changes[w]`` reads), one value per algebra under the same domain gates.

    PER BAR, never annualized — sqrt-time scaling interprets cadence, which is the caller's
    assumption, not this engine's. ``w == 1`` holds a single change and cannot carry a ddof=1
    std, so it refuses as ``insufficient_bars``; a hole anywhere in the window is a 1-bar
    change with a missing endpoint (``endpoint_missing``), never skipped over.
    """
    out = _empty_change_block()
    if w < 2 or len(x) < w + 1:
        out["reason"] = "insufficient_bars"
        return out
    tail = np.asarray(x[-(w + 1) :], dtype=float)
    if not np.isfinite(tail).all():
        out["reason"] = "endpoint_missing"
        return out
    out["diff"] = float(np.std(np.diff(tail), ddof=1))
    if (tail > 0).all():
        ratio = tail[1:] / tail[:-1]
        out["pct"] = float(np.std(ratio - 1.0, ddof=1))
        out["log"] = float(np.std(np.log(ratio), ddof=1))
    else:
        out["ratio_reason"] = "non_positive_endpoint"
    return out


def _most_recent_extremum(tail: np.ndarray, kind: str) -> int:
    """Position of the max/min within ``tail``; ties resolve to the MOST RECENT bar — the
    ``bars_since_extremum`` rule, one tie convention engine-wide."""
    rev = tail[::-1]
    pos = int(np.argmax(rev)) if kind == "max" else int(np.argmin(rev))
    return len(tail) - 1 - pos


def _range_position_block(x: np.ndarray, index: pd.DatetimeIndex, w: int) -> RangePositionBlock:
    """Trailing high/low over EXACTLY the last w bars, distance from each, and the last
    level's right-continuous percentile rank within the window.

    Requires the full w bars: a shorter file refuses as ``insufficient_bars`` (never silently
    shortened) and a hole inside the window refuses as ``endpoint_missing`` (the extremum —
    or the rank — could be hiding in it). ``percentile_rank`` is position within the window's
    OWN range, not valuation: a trending series sits at its extreme by construction.
    """
    out: RangePositionBlock = {
        "high": None,
        "low": None,
        "from_high": None,
        "from_low": None,
        "percentile_rank": None,
        "reason": None,
        "ratio_reason": None,
    }
    if len(x) < w:
        out["reason"] = "insufficient_bars"
        return out
    tail = np.asarray(x[-w:], dtype=float)
    if not np.isfinite(tail).all():
        out["reason"] = "endpoint_missing"
        return out
    ts = index[-w:]
    hi_pos = _most_recent_extremum(tail, "max")
    lo_pos = _most_recent_extremum(tail, "min")
    hi, lo, last = float(tail[hi_pos]), float(tail[lo_pos]), float(tail[-1])
    out["high"] = {"value": hi, "timestamp": ts[hi_pos].isoformat()}
    out["low"] = {"value": lo, "timestamp": ts[lo_pos].isoformat()}
    from_high: RangeDistance = {"diff": last - hi, "pct": None}
    from_low: RangeDistance = {"diff": last - lo, "pct": None}
    from_high["pct"] = _ratio(last, hi)
    from_low["pct"] = _ratio(last, lo)
    out["from_high"], out["from_low"] = from_high, from_low
    if from_high["pct"] is None or from_low["pct"] is None:
        out["ratio_reason"] = "non_positive_endpoint"
    out["percentile_rank"] = float(np.count_nonzero(tail <= last)) / w
    return out


def _full_sample_block(x: np.ndarray, index: pd.DatetimeIndex) -> FullSampleBlock:
    """Whole-file extremes (over the OBSERVED finite levels, most-recent tie rule) plus
    drawdown/runup off them, against the ACTUAL last level.

    ``drawdown_pct = last/high - 1`` is a property of the file's EXTENT — extend or trim the
    file and it moves. A NaN last level refuses the distance reads as ``endpoint_missing``
    (the previous finite value is never substituted); the extremes themselves are observed
    facts and stay, with the missingness block right beside them stating the holes.
    """
    out: FullSampleBlock = {
        "high": None,
        "low": None,
        "drawdown_diff": None,
        "drawdown_pct": None,
        "runup_diff": None,
        "runup_pct": None,
        "reason": None,
        "ratio_reason": None,
    }
    vals = np.asarray(x, dtype=float)
    finite = np.isfinite(vals)
    if not finite.any():  # unreachable through the strict loader (all-NaN columns refuse)
        out["reason"] = "endpoint_missing"
        return out
    hi_pos = _most_recent_extremum(np.where(finite, vals, -np.inf), "max")
    lo_pos = _most_recent_extremum(np.where(finite, vals, np.inf), "min")
    hi, lo = float(vals[hi_pos]), float(vals[lo_pos])
    out["high"] = {"value": hi, "timestamp": index[hi_pos].isoformat()}
    out["low"] = {"value": lo, "timestamp": index[lo_pos].isoformat()}
    last = float(vals[-1])
    if not _finite(last):
        out["reason"] = "endpoint_missing"
        return out
    out["drawdown_diff"] = last - hi
    out["runup_diff"] = last - lo
    out["drawdown_pct"] = _ratio(last, hi)
    out["runup_pct"] = _ratio(last, lo)
    if out["drawdown_pct"] is None or out["runup_pct"] is None:
        out["ratio_reason"] = "non_positive_endpoint"
    return out


def _missingness_block(s: pd.Series) -> MissingnessBlock:
    """Pure counts — total NaN, interior NaN (between the first and last valid bar), and the
    valid extent. Threshold-flavored warnings stay in the loader's ``data_report``."""
    n_missing = int(s.isna().sum())
    first_valid = s.first_valid_index()
    last_valid = s.last_valid_index()
    if first_valid is None:  # unreachable through the strict loader
        return {
            "n_missing": n_missing,
            "n_interior_missing": 0,
            "first_valid": None,
            "last_valid": None,
        }
    return {
        "n_missing": n_missing,
        "n_interior_missing": int(s.loc[first_valid:last_valid].isna().sum()),
        "first_valid": first_valid.isoformat(),
        "last_valid": last_valid.isoformat(),
    }


def _volume_block(v: np.ndarray, windows: Sequence[int]) -> VolumeBlock:
    """Last volume, trailing mean and ``last_to_mean`` per window. The ratio carries NO
    'unusual' flag — what counts as elevated is the caller's judgment, not a property of the
    file. A non-positive trailing mean refuses the ratio (``non_positive_endpoint`` — the
    denominator is an endpoint of the ratio), never divides through zero."""
    vals = np.asarray(v, dtype=float)
    last = float(vals[-1]) if _finite(float(vals[-1])) else None
    by_window: dict[str, VolumeWindow] = {}
    for w in windows:
        entry: VolumeWindow = {
            "mean": None,
            "last_to_mean": None,
            "reason": None,
            "ratio_reason": None,
        }
        if len(vals) < w:
            entry["reason"] = "insufficient_bars"
        else:
            tail = vals[-w:]
            if not np.isfinite(tail).all():
                entry["reason"] = "endpoint_missing"
            else:
                mean = float(tail.mean())
                entry["mean"] = mean
                if last is not None and mean > 0:
                    entry["last_to_mean"] = last / mean
                else:
                    entry["ratio_reason"] = "non_positive_endpoint"
        by_window[str(w)] = entry
    return {"last": last, "windows": by_window}


def _last_bar_block(df: pd.DataFrame) -> LastBarBlock:
    """The final row verbatim — every column, NaN as null, never back-filled."""
    row = df.iloc[-1]
    return {
        "timestamp": df.index[-1].isoformat(),
        "values": {c: (float(row[c]) if _finite(float(row[c])) else None) for c in df.columns},
    }


def _series_blocks(s: pd.Series, index: pd.DatetimeIndex, windows: Sequence[int]) -> SeriesProfile:
    x = s.to_numpy(dtype=float)
    return {
        "changes": {str(w): _change_block(x, w) for w in windows},
        "dispersion": {str(w): _dispersion_block(x, w) for w in windows},
        "range_position": {str(w): _range_position_block(x, index, w) for w in windows},
        "full_sample": _full_sample_block(x, index),
        "missingness": _missingness_block(s),
    }


def _refusal_stub(report: FileReport) -> RefusalStub:
    """The profile a REFUSED file carries: identity plus the reason, nothing invented. The
    full diagnosis lives in the ``data_report`` entry the reason codes point into."""
    codes = ", ".join(dict.fromkeys(e["code"] for e in report.errors)) or "refused"
    return {"path": report.path, "sha256": report.sha256, "ok": False, "reason": codes}


def profile_file(df: pd.DataFrame, report: FileReport, windows: Sequence[int]) -> FileProfile:
    """Profile ONE admitted file: the header in run-summary vocabulary, the last bar, the
    per-series blocks, and (OHLCV with volume only) the volume block.

    An OHLCV file profiles ``close`` — the full bar rides ``last_bar``; a series-shaped file
    profiles every value column in file order. Selecting any other column is the caller's act
    (there is deliberately no ``--target-column``).
    """
    columns = ["close"] if report.shape == "ohlcv" else list(df.columns)
    profile: FileProfile = {
        "path": report.path,
        "sha256": report.sha256,
        "ok": True,
        "shape": report.shape,
        "n_bars": len(df),
        "index_start": df.index[0].isoformat(),
        "index_end": df.index[-1].isoformat(),
        "bar_spacing": bar_spacing(df.index),
        "last_bar": _last_bar_block(df),
        "series": {c: _series_blocks(df[c], df.index, windows) for c in columns},
        "volume": (
            _volume_block(df["volume"].to_numpy(dtype=float), windows)
            if report.shape == "ohlcv" and "volume" in df.columns
            else None
        ),
    }
    return profile


def describe_files(
    paths: list[str], shape: str | None = None, windows: Sequence[int] = DEFAULT_WINDOWS
) -> DescribeResult:
    """Profile ``paths`` in ARGUMENT ORDER (never sorted) → ``{data_report, profiles}``.

    Each file passes through the SAME strict read ``check-data`` performs, so ``data_report``
    is identical to check-data's over the same files and shape. An admitted file gets a full
    :func:`profile_file`; a refused file gets the ``{path, sha256, ok, reason}`` stub — the
    document is complete either way, and ``data_report["ok"]`` says which exit code the CLI
    owes (0 all admitted, 2 any refused).

    Library-boundary backstop (unreachable through the CLI, whose ``--windows`` parsing
    refuses first): windows must be a non-empty list of positive ints.
    """
    windows = [int(w) for w in windows]
    if not windows or any(w < 1 for w in windows):
        raise ValueError("windows must be a non-empty list of positive bar counts")
    reports: list[FileReport] = []
    profiles: list[FileProfile | RefusalStub] = []
    for path in paths:
        df, rep = read_strict_csv(path, expected_shape=shape)
        reports.append(rep)
        profiles.append(_refusal_stub(rep) if df is None else profile_file(df, rep, windows))
    return {"data_report": build_data_report(reports), "profiles": profiles}
