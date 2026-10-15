"""Frozen, numba-free reference implementations for parity testing the consolidated DSL kernels
(``compiler/nb.py``).

Plain NumPy/Python-loop ports of the vectorized/numba kernels, independent of the
``sliding_window_view``/``njit`` implementations, so drift between the two is caught by the parity
tests in ``tests/test_nb_kernels.py``. A window or cross-section whose values are ALL EQUAL has
exact answers — mean the value, std / deviations 0.0, z-score and correlation undefined (NaN) —
whatever the floating-point mean of a constant decimal window rounds to. A window variance that
overflowed is NaN wherever it is divided by or reported; a VARYING window whose variance fell below
the normal range (subnormal, or 0.0) is recomputed on its values lifted by ``2**_LIFT`` — exact —
its std floored at the smallest subnormal; the correlation's variance PRODUCT of two normal
variances is taken in an unbounded exponent range (:func:`_sqrt_product`); and the EW z-score's
recurrence runs in a frame lifted by a power of two once its variance nears that range. Every
window moment is taken about the window's own offset (its first value; a cross-section's is the
midpoint of its range), never about the rounded mean of the raw values, and every MEAN a decision
reads (``rolling_agg`` / ``cross_agg`` ``mean``, a cross-section's centre) is the exactly rounded
``math.fsum`` over the count — NaN where that sum overflows.
"""

from __future__ import annotations

import itertools
import math
import sys

import numpy as np

# ---- Rolling boolean aggregations ------------------------------------------


def rolling_all(arr: np.ndarray, window: int) -> np.ndarray:
    n = arr.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    count = 0
    for i in range(n):
        if arr[i]:
            count += 1
        if i >= window and arr[i - window]:
            count -= 1
        if i >= window - 1:
            out[i] = count == window
    return out


def rolling_any(arr: np.ndarray, window: int) -> np.ndarray:
    n = arr.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    count = 0
    for i in range(n):
        if arr[i]:
            count += 1
        if i >= window and arr[i - window]:
            count -= 1
        if i >= window - 1:
            out[i] = count > 0
    return out


# ---- Transform kernels -----------------------------------------------------


#: The smallest positive NORMAL double: a window variance below it has lost significant bits.
_MIN_NORMAL = 2.0**-1022
#: The power of two such a VARYING window is lifted by: scaling up never rounds, and every lifted
#: variance is a normal double again.
_LIFT = 600


def zscore_sma(arr: np.ndarray, window: int) -> np.ndarray:
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 2 or window > n:
        return out
    for i in range(window - 1, n):
        s = 0.0
        cnt = 0
        for j in range(i - window + 1, i + 1):
            v = arr[j]
            if not np.isnan(v):
                s += v
                cnt += 1
        if cnt < window:
            continue
        if len(set(arr[i - window + 1 : i + 1].tolist())) == 1:
            continue  # a constant window has no dispersion to standardize by
        # Two-pass centered variance (population) about the window's first value, matching the
        # production kernel: the one-pass E[x²] − E[x]² form loses the mantissa to cancellation
        # at large input levels, and the raw mean's rounding loses an ulp-scale spread.
        var, dev = _shifted_moments([float(v) for v in arr[i - window + 1 : i + 1]])
        if var < _MIN_NORMAL:  # a VARYING window whose squared deviations underflowed: lift it
            var, dev = _shifted_moments([math.ldexp(v, _LIFT) for v in arr[i - window + 1 : i + 1]])
        if not (_MIN_NORMAL <= var < math.inf):
            continue  # an overflowed variance
        z = dev / math.sqrt(var)
        if np.isfinite(z):
            out[i] = z
    return out


def _exact_sum(vals: list[float]) -> float:
    """The correctly rounded sum of finite ``vals`` — NaN where an intermediate sum overflows
    (``math.fsum`` raises there)."""
    try:
        return math.fsum(vals)
    except OverflowError:
        return math.nan


def _shifted_moments(vals: list[float]) -> tuple[float, float]:
    """``(population variance, last value's deviation)`` of a window, about its first value."""
    with np.errstate(over="ignore", invalid="ignore"):
        y = np.asarray(vals, dtype=np.float64) - vals[0]
        d = y - y.mean()
        return float(np.mean(d * d)), float(d[-1])


def zscore_ema(arr: np.ndarray, window: int) -> np.ndarray:
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 2:
        return out
    alpha = 2.0 / (window + 1.0)
    # West EW recurrence, matching the production kernel: exact-arithmetic identical to the
    # EW[x²] − EW[x]² pair, but centered, so it stays translation-invariant in floating point.
    return _zscore_ema_west(arr, alpha, window)


#: The EW recurrence's frame: lifted by ``2**_LIFT`` (at most twice) once the variance is under
#: ``2**-600`` with the input, mean and deviation under ``2**-300``; lowered once an input passes
#: ``2**400`` in it.
_EW_LIFT_VAR = 2.0**-600
_EW_LIFT_AT = 2.0**-300
_EW_LOWER_AT = 2.0**400


def _zscore_ema_west(arr: np.ndarray, alpha: float, warmup: int) -> np.ndarray:
    """The West recurrence with the exact-interval rules, written out step by step: the mean's
    convex update stays between the old mean and the input; on a repeated input whose update
    cannot move the stored mean (absorbed, or landing on the input) the deviation is tracked on
    its own and shrinks by (1 − alpha) per repeat, as it does in exact arithmetic. Where the
    variance would leave the normal range the whole state is carried scaled by a power of two
    (exact), so the steps are the plain ones in an unbounded exponent range."""
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if warmup < 1 or not (0.0 < alpha <= 1.0):
        return out
    mean = np.nan
    var = 0.0
    seen = 0
    last = np.nan
    dev = None  # tracked deviation while a repeated input pins the stored mean
    scale = 0  # the state is held times 2**scale
    for i in range(n):
        raw = arr[i]
        if np.isnan(raw):
            continue
        while scale > 0 and abs(raw) > math.ldexp(_EW_LOWER_AT, -scale):
            scale -= _LIFT
            mean = math.ldexp(mean, -_LIFT)
            var = math.ldexp(var, -2 * _LIFT)
            dev = None if dev is None else math.ldexp(dev, -_LIFT)
        x = math.ldexp(raw, scale) if scale else raw
        if np.isnan(mean):
            mean, var, dev = x, 0.0, None
        elif dev is not None and raw == last:
            var = (1.0 - alpha) * (var + alpha * dev * dev)
            dev_next = (1.0 - alpha) * dev
            # exact arithmetic never reaches zero below alpha 1: the smallest subnormal, sign kept
            var = var if var != 0.0 or alpha == 1.0 else 5e-324
            dev = dev_next if dev_next != 0.0 else math.copysign(5e-324, dev)
        else:
            d = x - mean
            step = mean + alpha * d
            step = min(max(step, min(mean, x)), max(mean, x))
            moved_var = (1.0 - alpha) * (var + alpha * d * d)
            # nor where the step's deviation is known nonzero: an update that underflowed floors the
            # same way; at d == 0 (the mean on the input, nothing carried: the exact deviation
            # unknown) it stays undefined once it rounds to 0.0, at alpha >= 0.5 (below, rounding
            # holds it at a subnormal and the z 0.0 stays decided)
            if moved_var == 0.0 and alpha < 1.0 and d != 0.0:
                moved_var = 5e-324
            var = moved_var
            dev = (1.0 - alpha) * d if (raw == last and d != 0.0 and step in (mean, x)) else None
            if dev == 0.0:  # a carried deviation that underflowed: floored, sign kept
                dev = math.copysign(5e-324, d)
            mean = step
        last = raw
        seen += 1
        r = (x - mean) if dev is None else dev
        if (
            scale < 2 * _LIFT
            and var < _EW_LIFT_VAR
            and max(abs(x), abs(mean), abs(r)) < _EW_LIFT_AT
        ):
            scale += _LIFT
            mean = math.ldexp(mean, _LIFT)
            var = math.ldexp(var, 2 * _LIFT)
            dev = None if dev is None else math.ldexp(dev, _LIFT)
            r = math.ldexp(r, _LIFT)
        if seen < warmup or var <= 0.0 or not np.isfinite(var):
            continue
        z = r / math.sqrt(var)
        if np.isfinite(z):
            out[i] = z
    return out


def percentile(arr: np.ndarray, window: int) -> np.ndarray:
    # Fraction strictly below the current value: count(v < x) / window.
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 1 or window > n:
        return out
    denom = float(window)
    for i in range(window - 1, n):
        x = arr[i]
        if np.isnan(x):
            continue
        r = valid = 0
        for j in range(i - window + 1, i + 1):
            v = arr[j]
            if np.isnan(v):
                continue
            valid += 1
            if v < x:
                r += 1
        if valid < window:
            continue
        out[i] = r / denom
    return out


def rolling_agg(arr: np.ndarray, window: int, agg: str) -> np.ndarray:
    # Trailing-window max/min/mean/std (population, ddof=0). NaN unless the whole window is finite —
    # the same gate as percentile.
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 1 or window > n:
        return out
    for i in range(window - 1, n):
        vals = []
        ok = True
        for j in range(i - window + 1, i + 1):
            v = arr[j]
            if np.isnan(v):
                ok = False
                break
            vals.append(v)
        if not ok or len(vals) < window:
            continue
        constant = len(set(vals)) == 1
        vals = np.asarray(vals, dtype=np.float64)
        if agg == "max":
            out[i] = vals.max()
        elif agg == "min":
            out[i] = vals.min()
        elif agg == "mean":
            out[i] = vals[0] if constant else _exact_sum(vals.tolist()) / window
        else:  # std, population, about the window's first value
            var, _ = _shifted_moments(vals.tolist())
            if constant:
                out[i] = 0.0
            elif var < _MIN_NORMAL:  # VARYING, its squared deviations underflowed: lift it
                lvar, _ = _shifted_moments(np.ldexp(vals, _LIFT).tolist())
                out[i] = max(math.ldexp(math.sqrt(lvar), -_LIFT), 5e-324)
            elif var < math.inf:
                out[i] = math.sqrt(var)
            # else: an overflowed variance — NaN
    return out


def runup(arr: np.ndarray, window: int | None = None) -> np.ndarray:
    """Fractional height above trough: ``x / trough − 1`` (≥ 0) — positive-scale domain only."""
    n = arr.shape[0]
    if window is None:
        trough = np.full(n, np.nan, dtype=np.float64)
        cur = np.nan
        for i in range(n):
            v = arr[i]
            if np.isnan(v):
                trough[i] = cur  # hold; fmin(cur, nan) ≡ cur once seeded
                continue
            cur = v if np.isnan(cur) else min(cur, v)
            trough[i] = cur
    else:
        trough = rolling_agg(arr, window, "min")
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        v, t = arr[i], trough[i]
        if np.isnan(v) or np.isnan(t) or t <= 0.0:
            continue
        r = v / t - 1.0
        if np.isfinite(r):
            out[i] = r
    return out


def bars_since_extremum(
    arr: np.ndarray, extremum: str = "max", window: int | None = None
) -> np.ndarray:
    """Bar count since the most recent attainment of the trailing/expanding max|min."""
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    is_max = extremum == "max"
    if window is None:
        has = False
        ext = 0.0
        idx = -1
        for i in range(n):
            v = arr[i]
            if np.isnan(v):
                continue
            if not has:
                has = True
                ext = v
                idx = i
                out[i] = 0.0
                continue
            if (is_max and v >= ext) or ((not is_max) and v <= ext):
                ext = v
                idx = i
                out[i] = 0.0
            else:
                out[i] = float(i - idx)
        return out
    if window < 1 or window > n:
        return out
    for i in range(window - 1, n):
        start = i - window + 1
        ok = True
        for j in range(start, i + 1):
            if np.isnan(arr[j]):
                ok = False
                break
        if not ok:
            continue
        idx = start
        ext = arr[start]
        for j in range(start + 1, i + 1):
            v = arr[j]
            if (is_max and v >= ext) or ((not is_max) and v <= ext):
                ext = v
                idx = j
        out[i] = float(i - idx)
    return out


def change(arr: np.ndarray, periods: int, kind: str = "pct") -> np.ndarray:
    """k-period change: ``pct`` = (cur/prev − 1), ``log`` = ln(cur/prev), ``diff`` = cur − prev.

    Matches ``nb.change_apply_nb``'s per-kind NaN guards: ``pct`` and ``log`` NaN on a
    non-positive endpoint (``pct`` also past the double range, where ``log`` stays finite);
    ``diff`` only needs both bars finite (0 is a valid level base).
    """
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if kind not in ("pct", "log", "diff"):
        raise ValueError(f"unknown change kind: {kind!r}")
    for i in range(periods, n):
        prev = arr[i - periods]
        cur = arr[i]
        if np.isnan(prev) or np.isnan(cur):
            continue
        if kind == "pct":
            # Positive-domain like `log` (and the runner's outcome guard): a ratio between
            # non-positive levels returns a finite number with an inverted sign.
            if prev <= 0.0 or cur <= 0.0:
                continue
            r = cur / prev - 1.0
            if np.isfinite(r):
                out[i] = r
        elif kind == "log":
            if prev <= 0.0 or cur <= 0.0:
                continue
            # Scale-safe: ln of the quotient where it is a finite positive NORMAL double, the
            # difference of logs where it overflowed/underflowed (1e200 → 1e-200 is −921.03…,
            # never ln(0)). Python floats, so the quotient cannot warn.
            q = float(cur) / float(prev)
            if sys.float_info.min <= q < math.inf:
                out[i] = math.log(q)
            else:
                out[i] = math.log(cur) - math.log(prev)
        else:  # diff
            out[i] = cur - prev
    return out


# ---- ema ---------------------------------------------------------------------


def ema(arr: np.ndarray, window: int) -> np.ndarray:
    """EMA (alpha = 2/(window+1)), seeded at the first finite value, NaN-skipping, NaN until
    ``window`` observations (warmup).

    The first ``window``-1 bars stay NaN — the warmup the ``_latch`` init mask relies on to gate
    signals (see test_runner.py).
    """
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 1:
        return out
    alpha = 2.0 / (window + 1.0)
    e = np.nan
    seen = 0
    for i in range(n):
        x = arr[i]
        if np.isnan(x):
            continue
        e = x if np.isnan(e) else _between(alpha * x + (1.0 - alpha) * e, e, x, alpha)
        seen += 1
        if seen >= window:
            out[i] = e
    return out


def _between(v: float, a: float, b: float, alpha: float) -> float:
    """``v`` (the rounded ``alpha·b + (1 − alpha)·a``) held where the convex combination's exact
    value lies: inside ``[min(a, b), max(a, b)]``, and — as the exact value never reaches the new
    input ``b`` when ``a != b`` and ``alpha < 1`` — never ON ``b``: the next float toward ``a``."""
    held = min(max(v, min(a, b)), max(a, b))
    if held == b and a != b and alpha < 1.0:
        return float(np.nextafter(b, a))
    return held


def shift_ref(arr: np.ndarray, periods: int) -> np.ndarray:
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        j = i - periods
        if j >= 0:
            out[i] = arr[j]
    return out


def cross_agg_ref(arr: np.ndarray, agg: str, min_valid: int) -> np.ndarray:
    """Row-loop reference for the cross-sectional aggregate (rows × cols → rows × cols)."""
    rows, cols = arr.shape
    out = np.full((rows, cols), np.nan, dtype=np.float64)
    for i in range(rows):
        vals = [x for x in arr[i] if math.isfinite(x)]
        if len(vals) < max(min_valid, 2):
            continue
        out[i, :] = _cross_value(vals, agg)
    return out


def _cross_value(vals: list[float], agg: str) -> float:
    """One cross-section's aggregate (shared by the plain and grouped references)."""
    constant = len(set(vals)) == 1
    if agg == "mean":
        return vals[0] if constant else _exact_sum(vals) / len(vals)
    if agg == "median":
        return _sorted_median(vals)
    if agg == "std":
        if constant:
            return 0.0
        var = _row_variance(vals)
        if var < _MIN_NORMAL:  # a VARYING row whose squared deviations underflowed: lift it
            lvar = _row_variance([math.ldexp(x, _LIFT) for x in vals])
            return max(math.ldexp(math.sqrt(lvar), -_LIFT), 5e-324)
        return math.sqrt(var) if var < math.inf else math.nan  # NaN: an overflowed variance
    if agg == "frac_positive":
        return sum(1.0 for x in vals if x > 0.0) / len(vals)
    raise ValueError(agg)


def _row_offset(vals: list[float]) -> float:
    """A cross-section's offset: the midpoint of its range, each end halved first."""
    return max(vals) / 2.0 + min(vals) / 2.0


def _row_deviations(vals: list[float]) -> list[float]:
    """Each member's deviation from the cross-section's mean, about the row's offset."""
    c = _row_offset(vals)
    y = [x - c for x in vals]
    mu = _exact_sum(y) / len(y)
    return [v - mu for v in y]


def _row_variance(vals: list[float]) -> float:
    """A cross-section's population variance, two-pass about the row's offset."""
    return sum(d * d for d in _row_deviations(vals)) / len(vals)


def rolling_corr_ref(left: np.ndarray, right: np.ndarray, window: int) -> np.ndarray:
    """Row-loop reference for the trailing-window Pearson correlation of two 1D series.

    Population moments (ddof=0), NaN unless the whole window is finite in BOTH inputs AND both
    windows vary (neither constant) AND neither variance overflowed — a leg whose variance fell
    below the normal range is lifted by ``2**_LIFT`` first — clipped to ``[−1, 1]``; matches
    ``nb.rolling_corr_apply_nb``.
    """

    def moments(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
        with np.errstate(over="ignore", invalid="ignore"):
            da = (a - a[0]) - (a - a[0]).mean()
            db = (b - b[0]) - (b - b[0]).mean()
            va = float(np.mean(da * da))
            vb = float(np.mean(db * db))
            cov = float(np.mean(da * db))
        return va, vb, cov

    n = left.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 3 or window > n:
        return out
    for i in range(window - 1, n):
        a = left[i - window + 1 : i + 1]
        b = right[i - window + 1 : i + 1]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        if len(set(a.tolist())) == 1 or len(set(b.tolist())) == 1:
            continue
        va, vb, cov = moments(a, b)
        if va < _MIN_NORMAL or vb < _MIN_NORMAL:  # a VARYING leg's squares underflowed: lift it
            a = np.ldexp(a, _LIFT) if va < _MIN_NORMAL else a
            b = np.ldexp(b, _LIFT) if vb < _MIN_NORMAL else b
            va, vb, cov = moments(a, b)
        if not (_MIN_NORMAL <= va < math.inf and _MIN_NORMAL <= vb < math.inf):
            continue
        r = cov / _sqrt_product(va, vb)
        if math.isfinite(r):
            out[i] = min(max(r, -1.0), 1.0)  # an exactly linear window's rounding stays at ±1
    return out


def _sqrt_product(a: float, b: float) -> float:
    """``sqrt(a·b)`` for NORMAL ``a, b > 0`` in an unbounded exponent range: while the product is
    not a normal double, a factor of four moves out of (or into) BOTH factors — exact, as scaling
    a normal double by a power of two never rounds — and its root ``2**k`` is put back after the
    square root."""
    k = 0
    while not (_MIN_NORMAL <= a * b < math.inf):
        if a * b > 1.0:  # overflowed
            a, b, k = a / 4.0, b / 4.0, k + 2
        else:  # fell below the normal range
            a, b, k = a * 4.0, b * 4.0, k - 2
    return math.ldexp(math.sqrt(a * b), k)


def expanding(arr: np.ndarray, agg: str) -> np.ndarray:
    n = arr.shape[0]
    out = np.full(n, np.nan)
    cur = np.nan
    for i in range(n):
        v = arr[i]
        if np.isnan(v):
            out[i] = cur
            continue
        cur = v if np.isnan(cur) else (max(cur, v) if agg == "max" else min(cur, v))
        out[i] = cur
    return out


def drawdown(arr: np.ndarray, window: int | None) -> np.ndarray:
    n = arr.shape[0]
    peak = expanding(arr, "max") if window is None else rolling_agg(arr, window, "max")
    out = np.full(n, np.nan)
    for i in range(n):
        v, p = arr[i], peak[i]
        if np.isnan(v) or np.isnan(p) or p <= 0.0:
            continue
        r = v / p - 1.0
        if np.isfinite(r):
            out[i] = r
    return out


# ---- the event-anchor family -------------------------------------------------------------
#
# Independently authored against the spec in ``dsl.nodes``: the event is the tradable signal;
# ``s(t)`` is the latest event bar <= t (current bar included, latest wins); before the first event
# the anchor is absent while KNOWN; after a post-warmup hole it is absent and UNKNOWN until the
# next defined event.


def event_anchor(
    sig: np.ndarray, init: np.ndarray, defined: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n = sig.shape[0]
    s_idx = np.full(n, -1, dtype=np.int64)
    known = np.ones(n, dtype=np.bool_)
    s = -1
    k = True
    for t in range(n):
        if not init[t]:
            s, k = -1, True
        elif not defined[t]:
            k = False
        elif sig[t]:
            s, k = t, True
        s_idx[t] = s if k else -1
        known[t] = k
    return s_idx, known


def bars_since_event(s_idx: np.ndarray) -> np.ndarray:
    out = np.full(s_idx.shape[0], np.nan)
    for t in range(s_idx.shape[0]):
        if s_idx[t] >= 0:
            out[t] = float(t - s_idx[t])
    return out


def event_value(x: np.ndarray, s_idx: np.ndarray) -> np.ndarray:
    out = np.full(s_idx.shape[0], np.nan)
    for t in range(s_idx.shape[0]):
        s = s_idx[t]
        if s >= 0 and math.isfinite(x[s]):
            out[t] = x[s]
    return out


def event_agg(
    sig: np.ndarray, init: np.ndarray, defined: np.ndarray, x: np.ndarray, agg: str
) -> np.ndarray:
    """``agg(x[s .. t])`` per bar, recomputed from scratch over the window each time (no running
    state — the slow, obviously-correct form; the sum is the exactly rounded ``math.fsum``); a
    NaN anywhere in the window → NaN."""
    s_idx, _known = event_anchor(sig, init, defined)
    n = sig.shape[0]
    out = np.full(n, np.nan)
    for t in range(n):
        s = s_idx[t]
        if s < 0:
            continue
        win = x[s : t + 1]
        if not np.all(np.isfinite(win)):
            continue
        try:
            total = math.fsum(win.tolist())
        except OverflowError:
            total = math.nan
        if agg == "sum":
            v = total
        elif agg == "max":
            v = float(np.max(win))
        elif agg == "min":
            v = float(np.min(win))
        elif agg == "mean":
            v = float(win[0]) if len(set(win.tolist())) == 1 else total / len(win)
        else:
            raise ValueError(agg)
        if math.isfinite(v):
            out[t] = v
    return out


def mask_values(
    value: np.ndarray, init: np.ndarray, defined: np.ndarray, full: np.ndarray
) -> np.ndarray:
    """Only once ``full`` (every branch past warmup) and decided: 1.0 on a True, 0.0 on a False;
    NaN otherwise — a partially warm condition's decided Trues included."""
    out = np.full(value.shape[0], np.nan)
    for t in range(value.shape[0]):
        if full[t] and init[t] and defined[t]:
            out[t] = 1.0 if value[t] else 0.0
    return out


def lag_channels(
    value: np.ndarray, init: np.ndarray, defined: np.ndarray, periods: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All three channels shifted back ``periods`` bars; leading bars value False, init False,
    defined True."""
    n = value.shape[0]
    v = np.zeros(n, dtype=np.bool_)
    i = np.zeros(n, dtype=np.bool_)
    d = np.ones(n, dtype=np.bool_)
    for t in range(periods, n):
        v[t] = value[t - periods]
        i[t] = init[t - periods]
        d[t] = defined[t - periods]
    return v, i, d


# ---- first_true over every completion of the child's holes ---------------------------------
#
# Independently authored against the spec in ``nb.first_true_1d``. On a fully known child it is
# the deterministic false→true machine: warmup resets it; the first True after warmup does not
# fire (an initialized False must come first); the ``cooldown`` bars after a fire never fire. A
# hole (``init & ~defined``) could have been either value, so the output is DECIDED exactly where
# every completion of the holes agrees on fire/no-fire, and its value is that common answer.

_FIRST_TRUE_WARMUP = (False, False, 0)  # (prev, seen_false, cool_left)


def _first_true_step(
    state: tuple[bool, bool, int], cur: bool, cooldown: int
) -> tuple[tuple[bool, bool, int], bool]:
    """One bar of the deterministic machine: the next state and whether the bar fired."""
    prev, seen_false, cool_left = state
    if cool_left > 0:
        return (cur, seen_false or not cur, cool_left - 1), False
    fired = cur and not prev and seen_false
    return (cur, seen_false or not cur, cooldown if fired else 0), fired


def first_true_machine(value: np.ndarray, init: np.ndarray, cooldown: int) -> np.ndarray:
    """The deterministic machine over a child whose every bar is known: the fires."""
    out = np.zeros(value.shape[0], dtype=np.bool_)
    state = _FIRST_TRUE_WARMUP
    for t in range(value.shape[0]):
        if not init[t]:
            state = _FIRST_TRUE_WARMUP
            continue
        state, out[t] = _first_true_step(state, bool(value[t]), cooldown)
    return out


def first_true_completions(
    value: np.ndarray, init: np.ndarray, defined: np.ndarray, cooldown: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(out, defined_out)`` by brute force: the machine run on each of the ``2**k`` completions
    of the ``k`` holes — the definition itself, exponential, so for small ``k`` only."""
    holes = np.flatnonzero(init & ~defined)
    runs = []
    for bits in itertools.product((False, True), repeat=holes.shape[0]):
        completed = value.astype(np.bool_)
        completed[holes] = bits
        runs.append(first_true_machine(completed, init, cooldown))
    fires = np.array(runs)
    every = fires.all(axis=0)
    return every, every | ~fires.any(axis=0)


def first_true(
    value: np.ndarray, init: np.ndarray, defined: np.ndarray, cooldown: int
) -> tuple[np.ndarray, np.ndarray]:
    """``first_true_completions`` in polynomial time: completions that reach the same machine
    state share every future (the machine is deterministic), so they are carried as the SET of
    states, each advanced under every value its bar may take."""
    n = value.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    defined_out = np.ones(n, dtype=np.bool_)
    states = {_FIRST_TRUE_WARMUP}
    for t in range(n):
        if not init[t]:
            states = {_FIRST_TRUE_WARMUP}
            continue
        values = (bool(value[t]),) if defined[t] else (False, True)
        successors = set()
        fired = set()
        for state in states:
            for cur in values:
                nxt, f = _first_true_step(state, cur, cooldown)
                successors.add(nxt)
                fired.add(f)
        states = successors
        out[t] = fired == {True}
        defined_out[t] = len(fired) == 1
    return out, defined_out


# ---- grouped cross-sectional kernels --------------------------------------------------------
#
# Row loop × per-label loop, independently authored: at each bar, the members sharing one finite
# label form a cross-section of their own; a NaN label excludes the member; ``min_valid`` applies
# per group; ``cross_agg`` broadcasts the group's aggregate to that group's members only.


def _avg_ranks(vals: list[float]) -> list[float]:
    """Average ranks (1-based) with ties averaged."""
    order = sorted(range(len(vals)), key=lambda k: vals[k])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _groups(row: np.ndarray, labels: np.ndarray) -> dict[float, list[int]]:
    groups: dict[float, list[int]] = {}
    for g, lab in enumerate(labels):
        if math.isfinite(lab) and math.isfinite(row[g]):
            groups.setdefault(float(lab), []).append(g)
    return groups


def cross_rank_grouped_ref(arr: np.ndarray, labels: np.ndarray, min_valid: int = 2) -> np.ndarray:
    rows, cols = arr.shape
    out = np.full((rows, cols), np.nan)
    for i in range(rows):
        for members in _groups(arr[i], labels[i]).values():
            k = len(members)
            if k < max(min_valid, 2):
                continue
            ranks = _avg_ranks([float(arr[i, g]) for g in members])
            for g, r in zip(members, ranks, strict=True):
                out[i, g] = (r - 1.0) / (k - 1.0)
    return out


def cross_demean_grouped_ref(arr: np.ndarray, labels: np.ndarray, min_valid: int = 2) -> np.ndarray:
    rows, cols = arr.shape
    out = np.full((rows, cols), np.nan)
    for i in range(rows):
        for members in _groups(arr[i], labels[i]).values():
            if len(members) < max(min_valid, 2):
                continue
            vals = [float(arr[i, g]) for g in members]
            devs = _row_deviations(vals)
            for g, d in zip(members, devs, strict=True):
                out[i, g] = 0.0 if len(set(vals)) == 1 else d
    return out


def cross_agg_grouped_ref(
    arr: np.ndarray, labels: np.ndarray, agg: str, min_valid: int = 2
) -> np.ndarray:
    rows, cols = arr.shape
    out = np.full((rows, cols), np.nan)
    for i in range(rows):
        # a group is every member with THIS finite label — its aggregate is over the finite ones
        # and broadcasts to every labelled member of the group, finite input or not
        by_label: dict[float, list[int]] = {}
        for g, lab in enumerate(labels[i]):
            if math.isfinite(lab):
                by_label.setdefault(float(lab), []).append(g)
        for members in by_label.values():
            vals = [float(arr[i, g]) for g in members if math.isfinite(arr[i, g])]
            if len(vals) < max(min_valid, 2):
                continue
            v = _cross_value(vals, agg)
            for g in members:
                out[i, g] = v
    return out


# ---- robust window statistics and the explicit-decay EW forms -----------------------------


def _midpoint(lo: float, hi: float) -> float:
    """The midpoint of two doubles: ``(lo + hi) / 2`` wherever finite, else the halves' sum —
    exact there, an overflowing sum's operands lying far above the subnormal range."""
    mid = (lo + hi) / 2.0
    return mid if math.isfinite(mid) else lo / 2.0 + hi / 2.0


def _sorted_median(vals: list[float]) -> float:
    s = sorted(vals)
    m = len(s) // 2
    return s[m] if len(s) % 2 else _midpoint(s[m - 1], s[m])


def rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        vals = [float(v) for v in arr[i - window + 1 : i + 1]]
        if any(math.isnan(v) for v in vals):
            continue
        out[i] = _sorted_median(vals)
    return out


def rolling_mad(arr: np.ndarray, window: int) -> np.ndarray:
    """Median absolute deviation about the SAME window's median, unscaled."""
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        vals = [float(v) for v in arr[i - window + 1 : i + 1]]
        if any(math.isnan(v) for v in vals):
            continue
        med = _sorted_median(vals)
        out[i] = _sorted_median([abs(v - med) for v in vals])
    return out


def ema_alpha(arr: np.ndarray, alpha: float, warmup: int) -> np.ndarray:
    """EMA with an explicit decay: first finite value seeds, NaN-skipping, NaN until ``warmup``
    observations."""
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    e = np.nan
    seen = 0
    for i in range(n):
        x = arr[i]
        if np.isnan(x):
            continue
        e = x if np.isnan(e) else _between(alpha * x + (1.0 - alpha) * e, e, x, alpha)
        seen += 1
        if seen >= warmup:
            out[i] = e
    return out


def zscore_ema_alpha(arr: np.ndarray, alpha: float, warmup: int) -> np.ndarray:
    return _zscore_ema_west(arr, alpha, warmup)
