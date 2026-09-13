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
    state — the slow, obviously-correct form); a NaN anywhere in the window → NaN."""
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
        if agg == "sum":
            v = float(np.sum(win))
        elif agg == "max":
            v = float(np.max(win))
        elif agg == "min":
            v = float(np.min(win))
        elif agg == "mean":
            v = float(np.sum(win)) / len(win)
        else:
            raise ValueError(agg)
        if math.isfinite(v):
            out[t] = v
    return out


def mask_values(value: np.ndarray, init: np.ndarray, defined: np.ndarray) -> np.ndarray:
    out = np.full(value.shape[0], np.nan)
    for t in range(value.shape[0]):
        if init[t] and defined[t]:
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
            mean = sum(float(arr[i, g]) for g in members) / len(members)
            for g in members:
                out[i, g] = arr[i, g] - mean
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
            for g in members:
                out[i, g] = v
    return out
