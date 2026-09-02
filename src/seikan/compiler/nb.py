"""Transform kernels for the DSL (``compiler/transforms.py`` dispatches DSL nodes here).

Two implementation styles, **no vectorbt**:

* **Vectorized numpy** for the windowed/elementwise kernels — z-score (SMA), percentile,
  rolling/expanding agg, drawdown/runup, change (pct/log/diff), shift, unary op, rolling
  Pearson correlation of two series, and the cross-sectional rank/demean/aggregate (axis=1,
  across the target columns at each bar — the only kernels with no 1-D form). Windowed kernels
  use ``numpy.lib.stride_tricks.sliding_window_view`` (a window requires every bar finite, else
  NaN); elementwise ones use plain lag slicing. No Python time-loop.
* **Numba** ``@njit`` for the genuinely sequential kernels — the EMA / EMA-z-score recurrences
  (``alpha = 2/(window+1)``), ``bars_since_extremum``, and the ``first_true`` episode-entry latch.
  These are scalar recurrences that don't vectorize; numba compiles the loop to machine code.

The production surface is the 2D ``*_apply_nb`` form (rows × columns → rows × columns), applied per
target. For the numpy kernels the 2D form IS the vectorized implementation; only the numba kernels
keep a 1D form (``ema_1d``, ``zscore_ema_1d``, ``first_true_1d``, ``_bars_since_extremum_1d`` —
scalar recurrences), which their 2D form loops over per column. The parity tests
(``tests/test_nb_kernels.py`` against the frozen ``tests/_reference_nb.py``) wrap the 2D forms in
their own single-column helpers.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import njit
from numpy.lib.stride_tricks import sliding_window_view

# =============================================================================
# Vectorized numpy kernels (windowed + elementwise) — no Python time-loop.
#
# The 2D ``*_apply_nb`` form is the implementation, columnwise by construction (the parity tests
# pin each column against the frozen reference). A windowed output is
# NaN unless every bar in the trailing window is finite (``sliding_window_view(...).any``), matching
# the frozen reference's ``cnt == window`` gate.
#
# FINITE-OR-NaN invariant: every kernel's output is finite or NaN. Kernels whose arithmetic can
# overflow from finite inputs (sums, ratios with denormal denominators, squared deviations)
# sanitize their results — a non-finite value becomes NaN, so it lands in the undecidable ledger
# instead of latching warmup or firing a threshold. Variances are computed CENTERED (two-pass or
# a Welford/West recurrence), never as E[x²] − E[x]²: the one-pass form cancels catastrophically
# on large-magnitude series (a price near 5000 with a std near 1 has already lost most of the
# mantissa).
# =============================================================================


def zscore_sma_apply_nb(arr: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    a = np.asarray(arr, dtype=float)
    n = a.shape[0]
    out = np.full(a.shape, np.nan)
    if window < 2 or window > n:
        return out
    win = sliding_window_view(a, window, axis=0)  # (n-window+1, cols, window)
    finite_all = ~np.isnan(win).any(axis=-1)
    with np.errstate(all="ignore"):
        mean = win.mean(axis=-1)
        # Two-pass centered population variance (the ``rolling_agg`` std pattern) — stable at any
        # input level, where the one-pass E[x²] − E[x]² form NaNs out around 1e8 already.
        var = ((win - mean[..., None]) ** 2).mean(axis=-1)
        x = a[window - 1 :]
        z = (x - mean) / np.sqrt(var)
    out[window - 1 :] = np.where(finite_all & (var > 0.0) & np.isfinite(z), z, np.nan)
    return out


def percentile_apply_nb(arr: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    # Fraction of the trailing window strictly below the current value: count(v < x) / window.
    a = np.asarray(arr, dtype=float)
    n = a.shape[0]
    out = np.full(a.shape, np.nan)
    if window < 1 or window > n:
        return out
    win = sliding_window_view(a, window, axis=0)
    finite_all = ~np.isnan(win).any(axis=-1)
    x = a[window - 1 :]
    count = (win < x[..., None]).sum(axis=-1)  # NaN < x is False, so NaNs never count
    out[window - 1 :] = np.where(finite_all, count / float(window), np.nan)
    return out


# ``agg``/``kind``/``op`` stay ``str`` rather than the DSL's Literal: each dispatch below ends in an
# unknown-value ``raise``, and a Literal parameter would make that guard statically unreachable —
# deleting the one check that catches a caller reaching a kernel around the schema.
def rolling_agg_apply_nb(
    arr: npt.NDArray[np.float64], window: int, agg: str
) -> npt.NDArray[np.float64]:
    # Trailing-window aggregate (max/min/mean/std) along the time axis; the current value sits at
    # the window's LAST bar. NaN unless the whole trailing window is finite (the percentile gate).
    # ``std`` is the population std (ddof=0), computed two-pass (deviations from the window mean) so
    # it matches the reference and stays numerically stable for large-magnitude series (e.g.
    # prices).
    a = np.asarray(arr, dtype=float)
    n = a.shape[0]
    out = np.full(a.shape, np.nan)
    if window < 1 or window > n:
        return out
    win = sliding_window_view(a, window, axis=0)
    finite_all = ~np.isnan(win).any(axis=-1)
    with np.errstate(all="ignore"):
        if agg == "max":
            v = win.max(axis=-1)
        elif agg == "min":
            v = win.min(axis=-1)
        elif agg == "mean":
            v = win.mean(axis=-1)
        elif agg == "std":
            m = win.mean(axis=-1)
            var = ((win - m[..., None]) ** 2).mean(axis=-1)
            v = np.sqrt(np.maximum(var, 0.0))
        else:
            raise ValueError(f"unknown rolling_agg agg: {agg!r}")
    # `isfinite(v)`: a finite window can still overflow its sum/squared deviations to ±inf.
    out[window - 1 :] = np.where(finite_all & np.isfinite(v), v, np.nan)
    return out


def expanding_agg_apply_nb(arr: npt.NDArray[np.float64], agg: str) -> npt.NDArray[np.float64]:
    """Expanding (all-time) max/min along the time axis; NaN-skipping via ``np.fmax``/``fmin``.

    Leading NaNs stay NaN until the first finite bar; thereafter NaN inputs hold the running peak
    (``fmax(peak, nan) == peak``). Only ``max``/``min`` — expanding mean/std are rejected upstream.
    """
    a = np.asarray(arr, dtype=float)
    if agg == "max":
        return np.fmax.accumulate(a, axis=0)
    if agg == "min":
        return np.fmin.accumulate(a, axis=0)
    raise ValueError(f"expanding_agg only supports 'max'/'min', got {agg!r}")


def drawdown_apply_nb(
    arr: npt.NDArray[np.float64], window: int | None = None
) -> npt.NDArray[np.float64]:
    """Fractional depth below peak: ``x / peak − 1`` (≤ 0) — defined on a POSITIVE scale only.

    ``window is None`` → expanding (all-time) peak; otherwise trailing ``window``-bar peak (same
    finite-window gate as ``rolling_agg`` max). Peak ≤ 0 / non-finite → NaN: the ratio is
    meaningless off a positive scale and INVERTS on a negative one ([-1, -2, -0.5] would read the
    deepest bar as a +100% run-up), so a non-positive-scale base censors instead of minting a
    sign-flipped number. With ``peak > 0`` the documented ``≤ 0`` bound holds everywhere emitted.
    """
    a = np.asarray(arr, dtype=float)
    peak = (
        expanding_agg_apply_nb(a, "max")
        if window is None
        else rolling_agg_apply_nb(a, window, "max")
    )
    with np.errstate(all="ignore"):
        dd = a / peak - 1.0
    return np.where(np.isfinite(a) & np.isfinite(peak) & (peak > 0.0) & np.isfinite(dd), dd, np.nan)


def runup_apply_nb(
    arr: npt.NDArray[np.float64], window: int | None = None
) -> npt.NDArray[np.float64]:
    """Fractional height above trough: ``x / trough − 1`` (≥ 0) — the mirror of ``drawdown``,
    defined on a POSITIVE scale only.

    ``window is None`` → expanding (all-time) trough; otherwise trailing ``window``-bar trough
    (same finite-window gate as ``rolling_agg`` min). Trough ≤ 0 / non-finite → NaN — the same
    positive-scale domain as ``drawdown``, and with ``trough > 0`` the documented ``≥ 0`` bound
    holds everywhere emitted (the running min is ≤ x, so x/trough ≥ 1).
    """
    a = np.asarray(arr, dtype=float)
    trough = (
        expanding_agg_apply_nb(a, "min")
        if window is None
        else rolling_agg_apply_nb(a, window, "min")
    )
    with np.errstate(all="ignore"):
        ru = a / trough - 1.0
    return np.where(
        np.isfinite(a) & np.isfinite(trough) & (trough > 0.0) & np.isfinite(ru), ru, np.nan
    )


@njit(cache=True)
def _bars_since_extremum_1d(
    arr: npt.NDArray[np.float64], window: int, is_max: bool
) -> npt.NDArray[np.float64]:
    """Bar count since the most recent attainment of the trailing/expanding max|min.

    ``window <= 0`` means expanding. Ties reset to the MOST RECENT attaining bar. Trailing form
    returns NaN until the window is full AND every bar in it is finite.
    """
    n = arr.shape[0]
    out = np.full(n, np.nan)
    if window == 0:
        # Expanding: track running extremum + its most-recent index.
        has = False
        extremum = 0.0
        idx = -1
        for i in range(n):
            v = arr[i]
            if np.isnan(v):
                # Hold state across holes (mirrors expanding_agg's fmax/fmin skip); output stays NaN
                # until the first finite bar, then NaN inputs don't advance the counter either —
                # the duration is only defined on finite bars.
                continue
            if not has:
                has = True
                extremum = v
                idx = i
                out[i] = 0.0
                continue
            if (is_max and v >= extremum) or ((not is_max) and v <= extremum):
                extremum = v
                idx = i
                out[i] = 0.0
            else:
                out[i] = float(i - idx)
        return out
    # Trailing window: for each i, scan [i-window+1, i] for the most recent argmax/argmin.
    if window < 1 or window > n:
        return out
    for i in range(window - 1, n):
        start = i - window + 1
        if np.isnan(arr[start]):
            continue
        ok = True
        for j in range(start + 1, i + 1):
            if np.isnan(arr[j]):
                ok = False
                break
        if not ok:
            continue
        # Re-scan for the MOST RECENT attaining index (ties → later bar).
        idx = start
        extremum = arr[start]
        for j in range(start + 1, i + 1):
            v = arr[j]
            if (is_max and v >= extremum) or ((not is_max) and v <= extremum):
                extremum = v
                idx = j
        out[i] = float(i - idx)
    return out


def bars_since_extremum_apply_nb(
    arr: npt.NDArray[np.float64],
    extremum: Literal["max", "min"] = "max",
    window: int | None = None,
) -> npt.NDArray[np.float64]:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    w = 0 if window is None else int(window)
    is_max = extremum == "max"
    out = np.empty_like(a)
    for j in range(a.shape[1]):
        out[:, j] = _bars_since_extremum_1d(a[:, j], w, is_max)
    return out


def rolling_corr_apply_nb(
    left: npt.NDArray[np.float64], right: npt.NDArray[np.float64], window: int
) -> npt.NDArray[np.float64]:
    """Trailing-window Pearson correlation of two (rows × targets) series.

    NaN unless the whole window is finite in BOTH inputs AND both window stds > 0 (zero-variance
    → NaN, never ±1) — the standard NaN-gating contract. Population moments (ddof=0) over the
    window, matching ``rolling_agg`` std.
    """
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"rolling_corr shape mismatch: {a.shape} vs {b.shape}")
    out = np.full(a.shape, np.nan)
    n = a.shape[0]
    if window < 3 or window > n:
        return out
    wa = sliding_window_view(a, window, axis=0)
    wb = sliding_window_view(b, window, axis=0)
    finite_all = ~(np.isnan(wa) | np.isnan(wb)).any(axis=-1)
    with np.errstate(all="ignore"):
        ma = wa.mean(axis=-1)
        mb = wb.mean(axis=-1)
        va = ((wa - ma[..., None]) ** 2).mean(axis=-1)
        vb = ((wb - mb[..., None]) ** 2).mean(axis=-1)
        cov = ((wa - ma[..., None]) * (wb - mb[..., None])).mean(axis=-1)
        corr = cov / np.sqrt(va * vb)
    # `isfinite(corr)`: the variance product can underflow to 0 while both factors stay > 0.
    ok = finite_all & (va > 0.0) & (vb > 0.0) & np.isfinite(corr)
    out[window - 1 :] = np.where(ok, corr, np.nan)
    return out


def change_apply_nb(
    arr: npt.NDArray[np.float64], periods: int, kind: str = "pct"
) -> npt.NDArray[np.float64]:
    """k-period change: ``pct`` = (cur/prev − 1), ``log`` = ln(cur/prev), ``diff`` = cur − prev.

    Per-kind NaN guards: ``pct`` and ``log`` require BOTH endpoints strictly positive — they are
    ratio algebras, meaningless off a positive scale, and neither fails loudly there (a percent
    change through zero or between negative levels returns a finite number with an INVERTED sign:
    −4 → −2 reads as −50% while the level rose) — the same rule the runner's forward-outcome
    measurement applies; signed levels belong to ``diff``, which only needs both bars finite
    (0 is a valid level base).
    """
    a = np.asarray(arr, dtype=float)
    n = a.shape[0]
    out = np.full(a.shape, np.nan)
    if periods < 1 or periods >= n:
        return out
    cur, prev = a[periods:], a[:-periods]
    finite = ~np.isnan(prev) & ~np.isnan(cur)
    with np.errstate(all="ignore"):
        if kind == "pct":
            valid = finite & (prev > 0.0) & (cur > 0.0)
            r = cur / prev - 1.0
        elif kind == "log":
            valid = finite & (prev > 0.0) & (cur > 0.0)
            r = np.log(cur / prev)
        elif kind == "diff":
            valid = finite
            r = cur - prev
        else:
            raise ValueError(f"unknown change kind: {kind!r}")
    # `isfinite(r)`: a denormal-positive denominator or a level-scale subtraction can overflow.
    out[periods:] = np.where(valid & np.isfinite(r), r, np.nan)
    return out


def shift_apply_nb(arr: npt.NDArray[np.float64], periods: int) -> npt.NDArray[np.float64]:
    """Backward shift: out[t] = arr[t - periods]; the leading ``periods`` rows are NaN.
    Backward-only (periods >= 1, schema-enforced) so it can never read the future."""
    a = np.asarray(arr, dtype=float)
    out = np.full(a.shape, np.nan)
    if periods < 1 or periods >= a.shape[0]:
        return out
    out[periods:] = a[:-periods]
    return out


def unary_op_apply_nb(arr: npt.NDArray[np.float64], op: str) -> npt.NDArray[np.float64]:
    """Element-wise unary arithmetic; out-of-domain (log of non-positive, sqrt of negative) → NaN,
    preserving the NaN-gating contract."""
    a = np.asarray(arr, dtype=float)
    if op == "abs":
        out = np.abs(a)
    elif op == "neg":
        out = -a
    elif op == "sign":
        out = np.sign(a)  # sign(NaN) = NaN
    elif op == "log":
        ok = a > 0.0
        out = np.where(ok, np.log(np.where(ok, a, 1.0)), np.nan)
    elif op == "sqrt":
        ok = a >= 0.0
        out = np.where(ok, np.sqrt(np.where(ok, a, 0.0)), np.nan)
    else:
        raise ValueError(f"unknown unary op: {op!r}")
    return np.where(np.isfinite(out), out, np.nan)


def cross_rank_apply_nb(arr: npt.NDArray[np.float64], min_valid: int) -> npt.NDArray[np.float64]:
    """Cross-sectional fraction rank ACROSS targets at each bar (axis=1, no time window).

    At bar t with k finite values, column g gets (avg_rank(x_g) - 1) / (k - 1) in [0, 1]
    (average ranks on ties). NaN where x_g is NaN or k < max(min_valid, 2). Inherently 2-D —
    there is no 1-D form (a single column has no cross-section and returns all NaN).
    """
    from scipy.stats import rankdata

    a = np.asarray(arr, dtype=float)
    out = np.full(a.shape, np.nan)
    if a.ndim != 2 or a.shape[1] < 2:
        return out
    finite = np.isfinite(a)
    k = finite.sum(axis=1, keepdims=True).astype(float)
    # Rank only what the denominator counts: ``nan_policy="omit"`` skips NaN alone, while ``k``
    # counts FINITE values — an unmasked ±inf would be ranked but not counted, pushing finite
    # members' fraction ranks outside [0, 1] and mis-ordering them.
    r = rankdata(np.where(finite, a, np.nan), method="average", axis=1, nan_policy="omit")
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (r - 1.0) / (k - 1.0)
    ok = finite & (k >= float(max(min_valid, 2)))
    return np.where(ok, frac, np.nan)


def cross_demean_apply_nb(arr: npt.NDArray[np.float64], min_valid: int) -> npt.NDArray[np.float64]:
    """Cross-sectional demean ACROSS targets at each bar (axis=1, no time window).

    Column g gets x_g - mean(finite values at bar t), self included. NaN where x_g is NaN or
    fewer than max(min_valid, 2) targets are finite. Inherently 2-D — no 1-D form.
    """
    a = np.asarray(arr, dtype=float)
    out = np.full(a.shape, np.nan)
    if a.ndim != 2 or a.shape[1] < 2:
        return out
    finite = np.isfinite(a)
    k = finite.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(all="ignore"):
        sums = np.where(finite, a, 0.0).sum(axis=1, keepdims=True)
        mean = sums / k
        dev = a - mean
    # `isfinite(dev)`: the row sum (or the subtraction at the float ceiling) can overflow.
    ok = finite & (k >= float(max(min_valid, 2))) & np.isfinite(dev)
    return np.where(ok, dev, np.nan)


def cross_agg_apply_nb(
    arr: npt.NDArray[np.float64], agg: str, min_valid: int
) -> npt.NDArray[np.float64]:
    """Cross-sectional AGGREGATE across targets at each bar, broadcast to every column (axis=1, no
    time window). Unlike ``cross_rank``/``cross_demean`` the value is a property of the
    cross-section itself, so every column carries it wherever at least ``max(min_valid, 2)`` targets
    are finite —
    a column whose own value is NaN still sees the group's aggregate. ``std`` is the population std
    (ddof=0, matching ``rolling_agg``); ``frac_positive`` counts finite values > 0. Inherently 2-D —
    a single column has no cross-section and returns all NaN."""
    a = np.asarray(arr, dtype=float)
    out = np.full(a.shape, np.nan)
    if a.ndim != 2 or a.shape[1] < 2:
        return out
    finite = np.isfinite(a)
    k = finite.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(all="ignore"):
        if agg == "mean":
            v = np.where(finite, a, 0.0).sum(axis=1, keepdims=True) / k
        elif agg == "median":
            masked = np.where(finite, a, np.nan)
            v = np.full((a.shape[0], 1), np.nan)
            rows = k.reshape(-1) > 0
            if rows.any():
                v[rows, 0] = np.nanmedian(masked[rows], axis=1)
        elif agg == "std":
            # Two-pass centered variance — the one-pass Σx²/k − mean² form cancels to a silent
            # 0.0 std at large member levels, which the documented z-score recipe then divides by.
            mean = np.where(finite, a, 0.0).sum(axis=1, keepdims=True) / k
            dev = np.where(finite, a - mean, 0.0)
            var = (dev * dev).sum(axis=1, keepdims=True) / k
            v = np.sqrt(np.maximum(var, 0.0))
        elif agg == "frac_positive":
            v = (finite & (a > 0.0)).sum(axis=1, keepdims=True) / k
        else:
            raise ValueError(f"unknown cross_agg agg: {agg!r}")
    # `isfinite(v)`: the row sums and squared deviations can overflow from finite members.
    ok = (k >= float(max(min_valid, 2))) & np.isfinite(v)
    # (n, 1) aggregate → all columns
    return np.broadcast_to(np.where(ok, v, np.nan), a.shape).copy()


# =============================================================================
# Numba kernels (sequential recurrences / state machine) — scalar loops, compiled.
# =============================================================================


@njit(cache=True)
def zscore_ema_1d(arr: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    """EW z-score via the West variance recurrence: ``d = x − μ; μ += α·d; s = (1−α)(s + α·d²)``.

    Algebraically identical (in exact arithmetic) to the EW[x²] − EW[x]² pair it replaces, but
    computed on centered quantities, so it is translation-invariant in floating point — the
    one-pass form lost its mantissa to cancellation at large input levels and drifted with the
    input offset. Same seeding (first finite value → μ = x, s = 0), same NaN-skipping warmup,
    same ``s > 0`` emission gate.
    """
    n = arr.shape[0]
    out = np.full(n, np.nan)
    if window < 2:
        return out
    alpha = 2.0 / (window + 1.0)
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
        z = (x - mu) / np.sqrt(s)
        if np.isfinite(z):
            out[i] = z
    return out


@njit(cache=True)
def ema_1d(arr: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    """EMA (alpha = 2/(window+1)): seeded at the first finite value, NaN-skipping, NaN until
    ``window`` observations (warmup) — the same warmup convention as the EMA inside
    ``zscore_ema``."""
    n = arr.shape[0]
    out = np.full(n, np.nan)
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


@njit(cache=True)
def first_true_1d(
    value: npt.NDArray[np.bool_],
    init: npt.NDArray[np.bool_],
    defined: npt.NDArray[np.bool_],
    cooldown: int,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """False→true transitions of a child's tradable signal, warmup-safe. Returns
    ``(out, defined_out)``.

    Operates on the child's ``value`` and ``init``: only initialized bars participate. The first
    True after warmup does NOT fire — ``seen_false`` must latch on an initialized False first.
    ``cooldown`` suppresses re-fires for that many bars after a fire (0 = every eligible
    transition). Also the crossover recipe: wrap ``threshold(fast > slow)`` in ``first_true``.

    ``defined`` marks bars where the child's truth is KNOWN. A hole breaks the
    transition state — whether the next True is a false→true edge depends on the missing bar —
    so it resets the machine like warmup does and taints the bars whose verdict it could still
    have flipped: ``max(1, cooldown)``, because a fire the hole may have suppressed would have
    armed a cooldown that long. At ``cooldown == 0`` this is exactly "the current and previous
    child bars must both be defined". The taint is bounded, not a full transitive closure (a
    fire INSIDE a tainted stretch re-arms its own cooldown); the gate refuses at the FIRST
    undefined bar in a pool, so any pool a deeper cascade could reach is already refused.
    """
    n = value.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    defined_out = np.ones(n, dtype=np.bool_)
    prev = False
    seen_false = False
    cool_left = 0
    taint = 0
    for i in range(n):
        if not init[i]:
            # Warmup: do not arm a transition from pre-init False. Vacuously defined — the
            # caller's `init` gate already excludes these bars.
            prev = False
            seen_false = False
            cool_left = 0
            taint = 0
            continue
        if not defined[i]:
            # In-data hole: the transition state cannot be carried across an unknown bar.
            prev = False
            seen_false = False
            cool_left = 0
            taint = cooldown if cooldown > 1 else 1
            defined_out[i] = False
            continue
        if taint > 0:
            taint -= 1
            defined_out[i] = False
        cur = value[i]
        if cool_left > 0:
            cool_left -= 1
            if not cur:
                seen_false = True
            prev = cur
            continue
        if cur and (not prev) and seen_false:
            out[i] = True
            cool_left = cooldown
        if not cur:
            seen_false = True
        prev = cur
    return out, defined_out


# =============================================================================
# 2D apply wrappers for the numba kernels (rows × cols → rows × cols, column-wise).
# =============================================================================


@njit(cache=True)
def zscore_ema_apply_nb(arr: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    out = np.empty_like(arr)
    for j in range(arr.shape[1]):
        out[:, j] = zscore_ema_1d(arr[:, j], window)
    return out


@njit(cache=True)
def ema_apply_nb(arr: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    out = np.empty_like(arr)
    for j in range(arr.shape[1]):
        out[:, j] = ema_1d(arr[:, j], window)
    return out


@njit(cache=True)
def first_true_apply_nb(
    value: npt.NDArray[np.bool_],
    init: npt.NDArray[np.bool_],
    defined: npt.NDArray[np.bool_],
    cooldown: int,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    rows, cols = value.shape
    out = np.empty((rows, cols), dtype=np.bool_)
    defined_out = np.empty((rows, cols), dtype=np.bool_)
    for j in range(cols):
        out[:, j], defined_out[:, j] = first_true_1d(
            value[:, j], init[:, j], defined[:, j], cooldown
        )
    return out, defined_out
