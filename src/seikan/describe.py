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
  emit it in one vocabulary, and the two rules it shares with the engine — the ``log``
  algebra's log ratio and where the normal range begins (:data:`~seikan.numeric.MIN_NORMAL`,
  its std's edge) — live in the leaf :mod:`seikan.numeric` for the same reason;
- admission is :func:`seikan.dataio.read_strict_csv` — the SAME strict reader ``check-data``
  runs, called identically, so the ``data_report`` this module builds is byte-equal to
  ``check-data``'s over the same files (a refused file is refused for the same reason by both,
  and gets a stub profile here rather than a guess);
- refuse-never-repair: a NaN endpoint is NEVER repaired by skipping to the previous finite
  value, a window shorter than requested is NEVER silently shortened, and every refusal is an
  explicit ``reason`` (``insufficient_bars`` / ``endpoint_missing`` / ``not_representable``) or
  ``ratio_reason`` (``non_positive_endpoint`` / ``not_representable``) in the block it refuses —
  present, never omitted: on a block that did not refuse, ``diff`` is never null and a null
  ratio read carries its ``ratio_reason`` (the two rules the emission seam checks); its other
  reads — an extremum, ``percentile_rank``, a volume ``mean`` — are finite by construction;
- all three change algebras are emitted with domain-nulls (``pct``/``log`` require both
  endpoints strictly positive — the positivity rule), because choosing ONE
  algebra would be the engine deciding what a series IS. ``log`` is scale-safe, so it is
  finite for every positive pair; a ``pct`` read whose ratio leaves the double range (a
  dispersion std's: any one 1-bar ratio) refuses as ``not_representable``, and a ``diff`` read
  that does (two finite levels of opposite sign farther apart than the double range, or a
  dispersion std past it) refuses its whole block by the same name, since ``diff`` rides every
  block that did not refuse. Every dispersion std is :func:`_std` — squared deviations past the
  range are rescaled, never refused;
- BOUNDED OUTPUT: no per-bar array ever rides the document — its size is independent of
  ``n_bars``. Windows are BAR counts, never days; ``bar_spacing`` states the clock.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from seikan.dataio import FileReport, bar_spacing, build_data_report, read_strict_csv
from seikan.numeric import MIN_NORMAL, log_ratio, log_ratio_array
from seikan.types import (
    ChangeBlock,
    DescribeResult,
    FileProfile,
    FullSampleBlock,
    LastBarBlock,
    MissingnessBlock,
    RangeDistance,
    RangePositionBlock,
    RatioReason,
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
    previous finite value. ``log`` is the scale-safe :func:`~seikan.numeric.log_ratio` — finite
    for every positive pair, ``1e200 → 1e-200`` included; a ``pct`` whose ratio overflows the
    double range refuses alone as ``not_representable`` while ``log`` and ``diff`` live, and a
    ``diff`` that overflows it refuses the whole block by that name.
    """
    out = _empty_change_block()
    if len(x) < w + 1:
        out["reason"] = "insufficient_bars"
        return out
    a, b = float(x[-1 - w]), float(x[-1])
    if not (_finite(a) and _finite(b)):
        out["reason"] = "endpoint_missing"
        return out
    diff = b - a
    if not _finite(diff):
        # Levels of opposite sign farther apart than the double range (−1e308 → 1e308): the
        # change has no double to be, and ``diff`` rides every block that did not refuse, so the
        # block refuses whole rather than carry a reasonless null — its ratio reads cross zero
        # and refuse on the domain anyway.
        out["reason"] = "not_representable"
        return out
    out["diff"] = diff
    if a > 0 and b > 0:
        pct = b / a - 1.0
        if _finite(pct):
            out["pct"] = pct
        else:
            out["ratio_reason"] = "not_representable"
        out["log"] = log_ratio(b, a)
    else:
        out["ratio_reason"] = "non_positive_endpoint"
    return out


def _ratio(last: float, base: float) -> tuple[float | None, RatioReason | None]:
    """``(last/base − 1, None)`` on a strictly positive pair — the one positive-domain ratio
    read the range-position and full-sample blocks share — else ``(None, tag)`` naming why it
    refused: a non-positive leg, or a ratio that left the double range (``last/base``
    overflowing; an underflow reads −1.0, exact to rounding). The caller states the tag as its
    block's ``ratio_reason``."""
    if not (last > 0 and base > 0):
        return None, "non_positive_endpoint"
    r = last / base - 1.0
    return (r, None) if _finite(r) else (None, "not_representable")


def _empty_change_block() -> ChangeBlock:
    """The all-null change/dispersion skeleton both per-window blocks start from — one
    declaration of the key order."""
    return {"diff": None, "pct": None, "log": None, "reason": None, "ratio_reason": None}


# Under ``errstate`` as a whole: a 1-bar change or ratio may leave the double range here, and
# each such read refuses BY NAME below — an overflow is an expected input, never a warning (an
# error under ``-W error``); squared deviations past it are :func:`_std`'s to rescale.
@np.errstate(over="ignore", invalid="ignore")
def _dispersion_block(x: np.ndarray, w: int) -> ChangeBlock:
    """ddof=1 std of the w trailing 1-bar changes (the same w+1 trailing levels
    ``changes[w]`` reads), one value per algebra under the same domain gates and the same
    ``not_representable`` refusals (a ``pct`` std alone, a ``diff`` std the whole block).

    PER BAR, never annualized — sqrt-time scaling interprets cadence, which is the caller's
    assumption, not this engine's. ``w == 1`` holds a single change and cannot carry a ddof=1
    std, so it refuses as ``insufficient_bars``; a hole anywhere in the window is a 1-bar
    change with a missing endpoint (``endpoint_missing``), never skipped over. Every std is
    :func:`_std`, one rule for the three algebras: finite changes whose squares leave the double
    range keep their std, so a read refuses only where a CHANGE has no double to be (or, for
    ``diff`` alone, the std itself).
    """
    out = _empty_change_block()
    if w < 2 or len(x) < w + 1:
        out["reason"] = "insufficient_bars"
        return out
    tail = np.asarray(x[-(w + 1) :], dtype=float)
    if not np.isfinite(tail).all():
        out["reason"] = "endpoint_missing"
        return out
    out["diff"] = _std(np.diff(tail))
    if (tail > 0).all():
        # Only a 1-bar ratio past the double range leaves the pct std non-finite, and it refuses
        # as ``not_representable``: squared pct deviations past it are rescaled like diff's, and
        # the std of finite pct changes — each ≥ −1 — stays under ~0.71·DBL_MAX. The log
        # changes are scale-safe (|change| < 1455), so their std is finite here.
        pct = _std(tail[1:] / tail[:-1] - 1.0)
        if _finite(pct):
            out["pct"] = pct
        else:
            out["ratio_reason"] = "not_representable"
        out["log"] = _std(log_ratio_array(tail[1:], tail[:-1]))
    else:
        out["ratio_reason"] = "non_positive_endpoint"
    # ``diff`` rides every block that did not refuse, so where ITS std has no double to be — a
    # 1-bar change of two finite levels past the double range, or a std past it (changes of
    # ±1.7e308) — the block refuses whole by name, the ratio reads with it: never a reasonless
    # null beside reads that rode.
    if out["diff"] is None or not _finite(out["diff"]):
        refused = _empty_change_block()
        refused["reason"] = "not_representable"
        return refused
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
    or the rank — could be hiding in it); a distance from an extremum past the double range
    refuses as ``not_representable``. ``percentile_rank`` is position within the window's OWN
    range, not valuation: a trending series sits at its extreme by construction.
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
    to_high, to_low = last - hi, last - lo
    if not (_finite(to_high) and _finite(to_low)):
        # A window spanning both signs farther apart than the double range: the last level's
        # distance from an extremum has no double to be, and a range distance's ``diff`` is
        # never null — so the block refuses whole, by name, like any other refusal of it.
        out["reason"] = "not_representable"
        return out
    out["high"] = {"value": hi, "timestamp": ts[hi_pos].isoformat()}
    out["low"] = {"value": lo, "timestamp": ts[lo_pos].isoformat()}
    from_high: RangeDistance = {"diff": to_high, "pct": None}
    from_low: RangeDistance = {"diff": to_low, "pct": None}
    # The two refusals cannot disagree: ``last <= hi`` keeps from_high in range, and a
    # non-positive ``hi`` or ``last`` refuses both legs on the domain.
    from_high["pct"], high_refusal = _ratio(last, hi)
    from_low["pct"], low_refusal = _ratio(last, lo)
    out["from_high"], out["from_low"] = from_high, from_low
    out["ratio_reason"] = high_refusal or low_refusal
    out["percentile_rank"] = float(np.count_nonzero(tail <= last)) / w
    return out


def _full_sample_block(x: np.ndarray, index: pd.DatetimeIndex) -> FullSampleBlock:
    """Whole-file extremes (over the OBSERVED finite levels, most-recent tie rule) plus
    drawdown/runup off them, against the ACTUAL last level.

    ``drawdown_pct = last/high - 1`` is a property of the file's EXTENT — extend or trim the
    file and it moves. A NaN last level refuses the distance reads as ``endpoint_missing``
    (the previous finite value is never substituted), a distance past the double range as
    ``not_representable``; the extremes themselves are observed facts and stay, with the
    missingness block right beside them stating the holes.
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
    drawdown, runup = last - hi, last - lo
    if not (_finite(drawdown) and _finite(runup)):
        # As in the range-position block: a distance past the double range refuses the
        # distance reads by name, never a reasonless null diff beside them.
        out["reason"] = "not_representable"
        return out
    out["drawdown_diff"] = drawdown
    out["runup_diff"] = runup
    # As in the range-position block, the two ratio refusals cannot disagree.
    out["drawdown_pct"], drawdown_refusal = _ratio(last, hi)
    out["runup_pct"], runup_refusal = _ratio(last, lo)
    out["ratio_reason"] = drawdown_refusal or runup_refusal
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


def _mean(vals: np.ndarray) -> float:
    """The arithmetic mean of finite values, overflow-safe: the direct ``vals.mean()`` wherever
    its sum stays in range, bit for bit — save an all-equal window, whose mean IS its value: the
    rounded sum of 21 copies of 1234.56, divided back, reads 1234.5600000000002, a held volume's
    ``last_to_mean`` of 0.9999999999999998 for an exact 1. Where the sum overflowed — the mean
    itself never does, lying between the values' extremes — the values are pre-scaled by the
    power of two ``2**-c`` (``2**c ≥ 2n``, which keeps every partial sum in range), the mean is
    scaled back and held in ``[min, max]`` like any convex combination. Read off the overflowed
    ``inf`` instead, the mean was null and a ratio over it a FINITE 0.0."""
    lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        return lo
    with np.errstate(over="ignore", invalid="ignore"):
        m = float(vals.mean())
    if _finite(m):
        return m
    c = (2 * vals.size).bit_length()
    with np.errstate(over="ignore"):  # a rounding past DBL_MAX is clamped back to the max below
        m = float(np.ldexp(np.ldexp(vals, -c).mean(), c))
    return min(max(m, lo), hi)


def _std(vals: np.ndarray) -> float:
    """The ddof=1 std of n ≥ 2 values — every dispersion read's, one rule for the three
    algebras — scale-safe: the direct ``np.std`` wherever its variance is a normal double, bit
    for bit, on every VARYING window. An all-equal window is exactly ``0.0``, decided before any
    variance is formed: the computed mean of bit-identical values carries its sum's rounding, so
    their direct variance is a residue, not zero (the 63 identical log changes of a doubling
    series read a std of 2.238e-16). Elsewhere the squared deviations left the double range —
    past it the std read ``inf`` (a null beside finite endpoints, with no reason, or a pct
    refused as unrepresentable while it was an ordinary double); below it a VARYING series read
    the 0.0 an all-equal one owns, or a std off a subnormal variance's lost bits — so the values
    are scaled by the power of two that brings the largest magnitude into ``[0.5, 1)`` and the
    std is scaled back: exact on the way up (scaling up by a power of two never rounds), to
    rounding on the way down, floored at the smallest subnormal as the kernels' lifted std is
    (``compiler.nb``) — a series that varies never reads zero dispersion. A REPORTED number,
    never a divisor, so it is computed rather than refused. Non-finite only where a value or the
    std itself leaves the double range — NaN off a non-finite change (a 1-bar change of two
    finite levels, or a 1-bar ratio, past it), ``inf`` off a std past DBL_MAX (four changes
    alternating ±1.7e308, whose std is ~1.96e308; a diff std only — pct changes, each ≥ −1, keep
    theirs under ~0.71·DBL_MAX) — and :func:`_dispersion_block` refuses either by name: a
    ``pct`` std under its ``ratio_reason``, a ``diff`` std its whole block."""
    lo, hi = float(vals.min()), float(vals.max())
    if not (_finite(lo) and _finite(hi)):
        return math.nan
    if lo == hi:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        var = float(np.var(vals, ddof=1))  # np.std IS the root of this variance, bit for bit
    if MIN_NORMAL <= var < math.inf:
        return math.sqrt(var)
    e = math.frexp(max(-lo, hi))[1]
    with np.errstate(over="ignore"):  # a std past DBL_MAX is unrepresentable: inf, refused
        return max(float(np.ldexp(np.std(np.ldexp(vals, -e), ddof=1), e)), 5e-324)


def _volume_block(v: np.ndarray, windows: Sequence[int]) -> VolumeBlock:
    """Last volume, trailing mean and ``last_to_mean`` per window. The ratio carries NO
    'unusual' flag — what counts as elevated is the caller's judgment, not a property of the
    file. A non-positive trailing mean refuses the ratio (``non_positive_endpoint`` — the
    denominator is an endpoint of the ratio), never divides through zero; the mean itself is
    overflow-safe (:func:`_mean`), so the ratio is never read off an overflowed sum."""
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
                mean = _mean(tail)
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
