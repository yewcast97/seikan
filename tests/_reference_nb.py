"""Frozen, numba-free reference implementations for parity testing the consolidated DSL kernels
(``compiler/nb.py``).

Plain NumPy/Python-loop ports of the vectorized/numba kernels, independent of the
``sliding_window_view``/``njit`` implementations, so drift between the two is caught by the parity
tests in ``tests/test_nb_kernels.py``.
"""

from __future__ import annotations

import math

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
        mean = s / cnt
        # Two-pass centered variance (population), matching the production kernel: the one-pass
        # E[x²] − E[x]² form loses the mantissa to cancellation at large input levels.
        ss = 0.0
        for j in range(i - window + 1, i + 1):
            d = arr[j] - mean
            ss += d * d
        var = ss / cnt
        if var <= 0.0:
            continue
        x = arr[i]
        if np.isnan(x):
            continue
        z = (x - mean) / math.sqrt(var)
        if np.isfinite(z):
            out[i] = z
    return out


def zscore_ema(arr: np.ndarray, window: int) -> np.ndarray:
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 2:
        return out
    alpha = 2.0 / (window + 1.0)
    # West EW recurrence, matching the production kernel: exact-arithmetic identical to the
    # EW[x²] − EW[x]² pair, but centered, so it stays translation-invariant in floating point.
    mu = np.nan
    s = 0.0
    seen = 0
    for i in range(n):
        x = arr[i]
        if np.isnan(x):
            continue
        if np.isnan(mu):
            mu = x
            s = 0.0
        else:
            d = x - mu
            mu = mu + alpha * d
            s = (1.0 - alpha) * (s + alpha * d * d)
        seen += 1
        if seen < window:
            continue
        if s <= 0.0 or not np.isfinite(s):
            continue
        z = (x - mu) / math.sqrt(s)
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
        vals = np.asarray(vals, dtype=np.float64)
        if agg == "max":
            out[i] = vals.max()
        elif agg == "min":
            out[i] = vals.min()
        elif agg == "mean":
            out[i] = vals.mean()
        else:  # std, population
            m = vals.mean()
            out[i] = math.sqrt(max(float(np.mean((vals - m) ** 2)), 0.0))
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

    Matches ``nb.change_apply_nb``'s per-kind NaN guards: ``pct`` NaNs when ``prev == 0``; ``log``
    NaNs on a non-positive ratio; ``diff`` only needs both bars finite (0 is a valid level base).
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
            out[i] = math.log(cur / prev)
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
        e = x if np.isnan(e) else alpha * x + (1.0 - alpha) * e
        seen += 1
        if seen >= window:
            out[i] = e
    return out


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
        if agg == "mean":
            v = sum(vals) / len(vals)
        elif agg == "median":
            s = sorted(vals)
            m = len(s) // 2
            v = s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0
        elif agg == "std":
            mu = sum(vals) / len(vals)
            v = math.sqrt(sum((x - mu) ** 2 for x in vals) / len(vals))
        elif agg == "frac_positive":
            v = sum(1.0 for x in vals if x > 0.0) / len(vals)
        else:
            raise ValueError(agg)
        out[i, :] = v
    return out


def rolling_corr_ref(left: np.ndarray, right: np.ndarray, window: int) -> np.ndarray:
    """Row-loop reference for the trailing-window Pearson correlation of two 1D series.

    Population moments (ddof=0), NaN unless the whole window is finite in BOTH inputs AND both
    window stds > 0 — matches ``nb.rolling_corr_apply_nb``.
    """
    n = left.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 3 or window > n:
        return out
    for i in range(window - 1, n):
        a = left[i - window + 1 : i + 1]
        b = right[i - window + 1 : i + 1]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        ma, mb = a.mean(), b.mean()
        va = float(np.mean((a - ma) ** 2))
        vb = float(np.mean((b - mb) ** 2))
        if va <= 0.0 or vb <= 0.0:
            continue
        cov = float(np.mean((a - ma) * (b - mb)))
        out[i] = cov / math.sqrt(va * vb)
    return out


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
