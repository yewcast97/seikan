"""Transform kernels for the DSL (``compiler/transforms.py`` dispatches DSL nodes here).

Two implementation styles, **no vectorbt**:

* **Vectorized numpy** for the windowed/elementwise kernels — z-score (SMA), percentile,
  rolling/expanding agg, drawdown/runup, change (pct/log/diff), shift, unary op, rolling
  Pearson correlation of two series, and the cross-sectional rank/demean/aggregate (axis=1,
  across the target columns at each bar — the only kernels with no 1-D form). Windowed kernels
  use ``numpy.lib.stride_tricks.sliding_window_view`` (a window requires every bar finite, else
  NaN); elementwise ones use plain lag slicing. No Python time-loop.
* **Numba** ``@njit`` for the genuinely sequential kernels — the EMA / EMA-z-score recurrences
  (``(alpha, warmup)`` from ``ema_params``: the span form ``alpha = 2/(window+1)`` or an explicit
  decay), ``bars_since_extremum``, the ``first_true`` episode-entry latch,
  and the event-anchor family (``event_anchor`` — the latest-event index ``s(t)`` a Condition's
  tradable signal defines — and ``event_agg``, the resettable running aggregate since it; the
  ``bars_since_event`` / ``event_value`` / ``mask`` reads off those are plain numpy).
  These are scalar recurrences that don't vectorize; numba compiles the loop to machine code.

The production surface is the 2D ``*_apply_nb`` form (rows × columns → rows × columns), applied per
target. For the numpy kernels the 2D form IS the vectorized implementation; only the numba kernels
keep a 1D form (``ema_1d``, ``zscore_ema_1d``, ``first_true_1d``, ``_bars_since_extremum_1d`` —
scalar recurrences), which their 2D form loops over per column. The parity tests
(``tests/test_nb_kernels.py`` against the frozen ``tests/_reference_nb.py``) wrap the 2D forms in
their own single-column helpers.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import njit
from numpy.lib.stride_tricks import sliding_window_view

from seikan.constants import MAX_BAR_COUNT_PARAM, MIN_EW_ALPHA
from seikan.numeric import log_ratio_array

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
#
# OUT-OF-RANGE intermediates: sanitizing the RESULT is not enough where an out-of-range
# intermediate yields a FINITE one — ``x / inf`` is 0.0, so a correlation whose variance product
# overflowed read 0.0 for ``corr(x, x)`` at 1e80 scale, and a z-score whose window variance
# overflowed read 0.0. A window variance that OVERFLOWED (deviations past ~1.3e154) is therefore
# NaN wherever it is divided by or reported — the undecidable ledger, never a value. One that fell
# BELOW the normal range is no overflow but a precision loss, and ordinary data reaches it: an
# ``ema`` of a change decays geometrically through it over a held stretch. The squares of
# deviations under ~1.5e-154 round to multiples of the smallest subnormal or to 0.0, reading a
# z-score past its population bound ``√(w−1)``, a correlation off in its third digit, or a
# varying window's std as the 0.0 an all-equal window owns. Such a VARYING window is recomputed
# LIFTED by ``2**_LIFT`` (``_lifted_windows``): scaling up by a power of two never rounds, and the
# lifted windows keep the direct form's layout — the same summation order — so every moment is
# the direct one times its power of two, and the z-score, correlation and std are the values the
# direct form has in an unbounded exponent range (a std under the smallest subnormal floors there:
# a varying window is never zero dispersion). The correlation's variance PRODUCT of two normal
# variances is taken with the exponents moved out wherever it leaves the normal range
# (``_sqrt_product``), to the same exactness. Every window whose variances and, for
# ``rolling_corr``, their product are normal doubles keeps every bit. The EW z-score
# (``zscore_ema_1d``) carries its variance in a recurrence, which the same rule reaches by moving
# the whole recurrence into a frame lifted by ``2**_LIFT`` once its variance nears the range
# (``_EW_LIFT_VAR``): its z is the recurrence's value in an unbounded exponent range too — not
# NaN, nor collapsed toward 0 off a floored variance, nor past its bound ``√((1−α)/α)``. Past the
# frame's reach only a HELD input's decay is left, a z exact arithmetic has sent under
# ``2**-158/α`` by then: its variance and carried deviation are floored at the smallest subnormal
# there, sign kept — decided, imprecise in magnitude; a hold whose mean sits ON the held value
# with nothing carried (``zscore_ema_1d``'s residual, its exact deviation unknown) is not floored:
# undefined once its variance rounds to 0.0 at ``α ≥ 0.5`` — below, round-to-nearest holds that
# variance at a subnormal and its z 0.0 stays decided.
#
# EXACT zero dispersion: a window (or a cross-section) whose finite values are all equal is
# detected exactly and given its exact answer instead of the arithmetic's: the mean of such a
# window is the value itself, its deviations and std are 0.0, and every read that divides by its
# dispersion (z-score, correlation) is undefined → NaN. The computed mean of a constant DECIMAL
# window is not the constant (``(0.1 + 0.1 + 0.1) / 3 != 0.1``), so without the exact test the
# residual minted a z-score of exactly ±1 and a correlation of 1 out of a constant series. The
# recurrences hold the same line (``ema_1d``, ``zscore_ema_1d``, ``event_agg_1d``). Every window
# with any variation whose variances and, for ``rolling_corr``, their product are normal doubles
# is computed exactly as before, bit for bit (the EW recurrences differ only on the steps whose
# rounded update left its exact interval, or a held input's carried deviation, and the EW z-score
# where its plain recurrence left the normal range; the out-of-range windows and the lifted EW
# frame are the note above);
# "constant" means bit-identical
# values — arithmetic on decimal levels (a spread ``y1 − y2`` moving in lockstep) leaves residue
# at the operands' scale, which is variation as far as any kernel can tell.
# =============================================================================


def _constant_windows(a: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.bool_]:
    """``(rows − window + 1, cols)``: the trailing windows (aligned like ``sliding_window_view``)
    whose values are all finite and bit-equal — a window is constant iff its ``window − 1``
    adjacent pairs are, counted with one cumulative sum (O(rows × cols), whatever the window)."""
    same = np.zeros(a.shape, dtype=np.int64)
    same[1:] = (a[1:] == a[:-1]) & np.isfinite(a[1:])
    pairs = np.cumsum(same, axis=0)
    finite_first = np.isfinite(a[: a.shape[0] - window + 1])
    constant: npt.NDArray[np.bool_] = finite_first & (
        pairs[window - 1 :] - pairs[: a.shape[0] - window + 1] == window - 1
    )
    return constant


#: The smallest positive NORMAL double: below it a value keeps fewer than 53 significant bits.
#: ``analysis._pools._TINY`` is the same constant for the pool statistics, which null a variance
#: out of the normal range on BOTH sides (``analysis._pools._normal`` says why).
_TINY = float(np.finfo(np.float64).tiny)


def _normal(v: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Elementwise: ``v`` is a positive NORMAL double — a variance that carries its full
    precision (see the out-of-range note above). False on 0.0, a subnormal, ``inf`` and NaN."""
    ok: npt.NDArray[np.bool_] = (v >= _TINY) & (v < np.inf)
    return ok


#: The power of two a varying window whose variance fell below the normal range is LIFTED by.
#: Down there every value of the window is under ~2**-440 (its deviations are under ~1.5e-154,
#: and a varying window's largest deviation is at least 2**-55 of its largest magnitude) and its
#: largest deviation at least 2**-1075 (distinct doubles differ by a subnormal step at least), so
#: the lifted values stay under 2**160 and the lifted variance is a normal double.
_LIFT = 600


def _window_moments(
    win: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The mean and the two-pass centered population variance over the LAST axis — stable at
    any input level, where the one-pass ``E[x²] − E[x]²`` form NaNs out around 1e8 already."""
    mean = win.mean(axis=-1)
    var: npt.NDArray[np.float64] = ((win - mean[..., None]) ** 2).mean(axis=-1)
    return mean, var


def _lifted_windows(a: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    """The trailing windows of ``a · 2**_LIFT`` — exact, as scaling UP by a power of two never
    rounds — in the direct form's own ``sliding_window_view`` layout, so every reduction sums in
    the same order and each lifted moment is the direct one times its power of two, bit for bit
    (a fancy-indexed copy of the few windows that need it would not be: numpy reduces a
    multi-column window view in another order than a contiguous stack). Members past ~2**424
    overflow; no window read off the lift holds one (``_LIFT``). Callers silence the overflow."""
    return sliding_window_view(np.ldexp(a, _LIFT), window, axis=0)


def _sqrt_product(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """``sqrt(a·b)`` of two positive NORMAL arrays (the caller's :func:`_normal` gate), overflow-
    and underflow-safe; any other element's result is unspecified, for that gate to discard.

    The direct ``sqrt(a·b)`` wherever the product is a normal double — bit for bit. Elsewhere the
    product overflowed (``corr(x, x)`` at 1e80 scale divided by ``inf``) or fell below the normal
    range (to 0.0 or a subnormal, a precision loss neither factor has), so each factor's exponent
    is taken out first (``frexp``, made even so the root halves it exactly) and put back after the
    root (``ldexp``). Power-of-two scaling of a normal factor never rounds, so the result is
    EXACTLY the direct form's value in an unbounded exponent range — not ``sqrt(a)·sqrt(b)``,
    which rounds twice and reads ``corr(x, x)`` one ulp off 1.0 on about half of all scales. It is
    exact only because the factors are: a SUBNORMAL factor has already lost its bits, and moving
    its exponent recovers none of them. ``analysis._pools._r2`` is the scalar twin (r²'s
    variance product) — a change to the rule changes both."""
    with np.errstate(all="ignore"):
        prod = a * b
        ma, ea = np.frexp(a)
        mb, eb = np.frexp(b)
        ma, ea = np.ldexp(ma, ea & 1), ea - (ea & 1)
        mb, eb = np.ldexp(mb, eb & 1), eb - (eb & 1)
        scaled = np.ldexp(np.sqrt(ma * mb), (ea + eb) // 2)
        out: npt.NDArray[np.float64] = np.where(
            np.isfinite(prod) & (prod >= _TINY), np.sqrt(prod), scaled
        )
    return out


def zscore_sma_apply_nb(arr: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    a = np.asarray(arr, dtype=float)
    n = a.shape[0]
    out = np.full(a.shape, np.nan)
    if window < 2 or window > n:
        return out
    win = sliding_window_view(a, window, axis=0)  # (n-window+1, cols, window)
    finite_all = ~np.isnan(win).any(axis=-1)
    # A constant window has no dispersion to standardize by — undefined, whatever residual the
    # mean's rounding leaves (see the zero-dispersion note above).
    varies = ~_constant_windows(a, window)
    with np.errstate(all="ignore"):
        # Two-pass centered population variance (the ``rolling_agg`` std pattern).
        mean, var = _window_moments(win)
        x = a[window - 1 :]
        z = (x - mean) / np.sqrt(var)
        # A VARYING window whose variance fell below the normal range (deviations under
        # ~1.5e-154: a subnormal variance read |z| past its bound √(w−1), one of 0.0 read NaN) is
        # recomputed lifted — the same z, every rounding scaled (the out-of-range note above).
        lost = finite_all & varies & (var < _TINY)
        if lost.any():
            lifted_mean, lifted_var = _window_moments(_lifted_windows(a, window))
            z = np.where(lost, (np.ldexp(x, _LIFT) - lifted_mean) / np.sqrt(lifted_var), z)
            var = np.where(lost, lifted_var, var)
    # `_normal(var)`: squared deviations past ~1.3e154 overflow the variance to inf, and the
    # z-score `(x − mean) / inf` is a FINITE 0.0 that `isfinite(z)` cannot catch.
    ok = finite_all & varies & _normal(var) & np.isfinite(z)
    out[window - 1 :] = np.where(ok, z, np.nan)
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
            # A constant window's mean IS its value (the zero-dispersion note above).
            v = np.where(_constant_windows(a, window), win[..., -1], win.mean(axis=-1))
        elif agg == "std":
            _, var = _window_moments(win)
            constant = _constant_windows(a, window)
            v = np.sqrt(var)
            # 0.0 only for a CONSTANT window. A VARYING one whose variance fell below the normal
            # range (squared deviations under ~1.5e-154 — to 0.0 itself under ~1e-162) is
            # recomputed lifted: its std exactly, rounded once to the window's scale and floored
            # at the smallest subnormal — never the 0.0 an all-equal window owns.
            lost = finite_all & ~constant & (var < _TINY)
            if lost.any():
                _, lifted_var = _window_moments(_lifted_windows(a, window))
                lifted_std = np.ldexp(np.sqrt(lifted_var), -_LIFT)
                v = np.where(lost, np.maximum(lifted_std, 5e-324), v)
            v = np.where(constant, 0.0, v)
        elif agg == "median":
            v = np.median(win, axis=-1)
        elif agg == "mad":
            # Median absolute deviation about the SAME window's median, unscaled.
            med = np.median(win, axis=-1)
            v = np.median(np.abs(win - med[..., None]), axis=-1)
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


def _pair_moments(
    wa: npt.NDArray[np.float64], wb: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The two population variances and the covariance of two aligned window stacks, two-pass
    (centered) over the LAST axis — the correlation's direct form."""
    da = wa - wa.mean(axis=-1)[..., None]
    db = wb - wb.mean(axis=-1)[..., None]
    va: npt.NDArray[np.float64] = (da**2).mean(axis=-1)
    vb: npt.NDArray[np.float64] = (db**2).mean(axis=-1)
    cov: npt.NDArray[np.float64] = (da * db).mean(axis=-1)
    return va, vb, cov


def rolling_corr_apply_nb(
    left: npt.NDArray[np.float64], right: npt.NDArray[np.float64], window: int
) -> npt.NDArray[np.float64]:
    """Trailing-window Pearson correlation of two (rows × targets) series.

    NaN unless the whole window is finite in BOTH inputs AND both window stds > 0 (zero-variance
    → NaN, never ±1; an exactly constant window is zero-variance whatever its computed variance's
    rounding residual) — the standard NaN-gating contract. Population moments (ddof=0) over the
    window, matching ``rolling_agg`` std.

    Scale-safe (the out-of-range note above): the denominator ``sqrt(va·vb)`` is
    :func:`_sqrt_product`, so a window whose variance PRODUCT leaves the double range while both
    variances are normal doubles — where ``corr(x, x)`` read 0.0 at 1e80 scale and NaN at 1e-85 —
    divides by the direct form's exact value; and a VARYING leg whose own variance fell below the
    normal range (deviations under ~1.5e-154, where its lost bits read a correlation off in its
    third digit, or NaN off a variance of 0.0) is recomputed lifted, the other leg as it is, so
    the window reads the correlation the direct form has in an unbounded exponent range. Every
    window whose variances and their product are normal doubles keeps every bit. A leg whose
    variance OVERFLOWED (deviations past ~1.3e154) is NaN: ``cov / inf`` read a finite 0.0.
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
    varies = ~_constant_windows(a, window) & ~_constant_windows(b, window)
    with np.errstate(all="ignore"):
        va, vb, cov = _pair_moments(wa, wb)
        # Each leg is lifted only on the windows where IT lost its variance: lifting a leg whose
        # variance is normal could overflow its squares (an ordinary leg beside a decayed one).
        lost_a = finite_all & varies & (va < _TINY)
        lost_b = finite_all & varies & (vb < _TINY)
        if lost_a.any() or lost_b.any():
            la = _lifted_windows(a, window)
            lb = _lifted_windows(b, window)
            for lost, xa, xb in (
                (lost_a & ~lost_b, la, wb),
                (~lost_a & lost_b, wa, lb),
                (lost_a & lost_b, la, lb),
            ):
                if lost.any():
                    lva, lvb, lcov = _pair_moments(xa, xb)
                    va = np.where(lost, lva, va)
                    vb = np.where(lost, lvb, vb)
                    cov = np.where(lost, lcov, cov)
        corr = cov / _sqrt_product(va, vb)
    # `_normal(va)` / `_normal(vb)`: an overflowed variance divides `cov` by inf — a FINITE ±0.0
    # that `isfinite(corr)` cannot catch (it still catches a covariance at the float ceiling).
    ok = finite_all & varies & _normal(va) & _normal(vb) & np.isfinite(corr)
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
    (0 is a valid level base). ``log`` is the scale-safe :func:`seikan.numeric.log_ratio_array`
    — the runner's outcome rule too — so a quotient past the double range (1e200 → 1e-200) still
    reads its finite log (−921.03…) instead of an undecidable NaN; a ``pct`` or ``diff`` past
    the range has no representable value and stays NaN.
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
            r = log_ratio_array(cur, prev)
        elif kind == "diff":
            valid = finite
            r = cur - prev
        else:
            raise ValueError(f"unknown change kind: {kind!r}")
    # `isfinite(r)`: a pct ratio over a denormal-positive denominator or a level-scale
    # subtraction can overflow (the scale-safe log ratio cannot).
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


def _constant_rows(a: npt.NDArray[np.float64], finite: npt.NDArray[np.bool_]) -> np.ndarray:
    """(rows, 1): the bars whose FINITE members all share one value (False on an empty row)."""
    hi = np.where(finite, a, -np.inf).max(axis=1, keepdims=True)
    lo = np.where(finite, a, np.inf).min(axis=1, keepdims=True)
    constant: np.ndarray = hi == lo
    return constant


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
    fewer than max(min_valid, 2) targets are finite. A cross-section whose members all share one
    value demeans to exactly 0.0 (the zero-dispersion note above). Inherently 2-D — no 1-D form.
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
        dev = np.where(_constant_rows(a, finite), 0.0, a - mean)
    # `isfinite(dev)`: the row sum (or the subtraction at the float ceiling) can overflow.
    ok = finite & (k >= float(max(min_valid, 2))) & np.isfinite(dev)
    return np.where(ok, dev, np.nan)


def _row_var(
    a: npt.NDArray[np.float64], finite: npt.NDArray[np.bool_], k: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """``(rows, 1)``: each row's population variance over its ``k`` finite members, two-pass
    (centered) — ``cross_agg`` std's direct form."""
    mean = np.where(finite, a, 0.0).sum(axis=1, keepdims=True) / k
    dev = np.where(finite, a - mean, 0.0)
    var: npt.NDArray[np.float64] = (dev * dev).sum(axis=1, keepdims=True) / k
    return var


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
    # A cross-section whose members all share one value: its mean IS that value and its std is
    # 0.0 (the zero-dispersion note above).
    constant = _constant_rows(a, finite)
    with np.errstate(all="ignore"):
        if agg == "mean":
            v = np.where(finite, a, 0.0).sum(axis=1, keepdims=True) / k
            v = np.where(constant, np.where(finite, a, -np.inf).max(axis=1, keepdims=True), v)
        elif agg == "median":
            masked = np.where(finite, a, np.nan)
            v = np.full((a.shape[0], 1), np.nan)
            rows = k.reshape(-1) > 0
            if rows.any():
                v[rows, 0] = np.nanmedian(masked[rows], axis=1)
        elif agg == "std":
            # Two-pass centered variance — the one-pass Σx²/k − mean² form cancels to a silent
            # 0.0 std at large member levels, which the documented z-score recipe then divides by.
            var = _row_var(a, finite, k)
            v = np.sqrt(var)
            # 0.0 only for a CONSTANT row; a VARYING one whose variance fell below the normal
            # range is recomputed lifted and floored (the ``rolling_agg`` std rule).
            lost = ~constant & (var < _TINY)
            if lost.any():
                lifted_std = np.ldexp(np.sqrt(_row_var(np.ldexp(a, _LIFT), finite, k)), -_LIFT)
                v = np.where(lost, np.maximum(lifted_std, 5e-324), v)
            v = np.where(constant, 0.0, v)
        elif agg == "frac_positive":
            v = (finite & (a > 0.0)).sum(axis=1, keepdims=True) / k
        else:
            raise ValueError(f"unknown cross_agg agg: {agg!r}")
    # `isfinite(v)`: the row sums and squared deviations can overflow from finite members.
    ok = (k >= float(max(min_valid, 2))) & np.isfinite(v)
    # (n, 1) aggregate → all columns
    return np.broadcast_to(np.where(ok, v, np.nan), a.shape).copy()


def cross_grouped_apply_nb(
    kernel: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    arr: npt.NDArray[np.float64],
    labels: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Apply a cross-sectional kernel WITHIN each distinct finite label at each bar.

    Grouping wraps the existing kernel per label rather than re-deriving it: for every distinct
    finite label the kernel sees the (rows × targets) input with every OTHER member masked to
    NaN, and only the labelled members' outputs are kept. Every property the ungrouped kernel
    has — the per-group ``min_valid`` floor, ``rankdata(method="average")`` ties, the overflow
    sanitization, ``cross_agg``'s broadcast (now to the group's members only) — is inherited
    unchanged, and ONE label reproduces the ungrouped result bit-exactly (a pandas
    ``groupby.transform`` was measured NOT bit-exact — summation order — and rejected). Labels are
    point-in-time: a member's label may change from bar to bar. A NaN label excludes the member.
    """
    a = np.asarray(arr, dtype=float)
    lab = np.asarray(labels, dtype=float)
    if a.shape != lab.shape:
        raise ValueError(f"cross_grouped shape mismatch: {a.shape} vs {lab.shape}")
    out = np.full(a.shape, np.nan)
    finite = np.isfinite(lab)
    for value in np.unique(lab[finite]):
        in_group = finite & (lab == value)
        res = kernel(np.where(in_group, a, np.nan))
        out[in_group] = res[in_group]
    return out


# =============================================================================
# Numba kernels (sequential recurrences / state machine) — scalar loops, compiled.
# =============================================================================


def ema_params(window: int | None, alpha: float | None) -> tuple[float, int]:
    """The ``(alpha, warmup)`` pair the EW kernels take, from EXACTLY one of the two DSL forms.

    ``window`` → ``(2/(window+1), window)`` — the classic span form. ``alpha`` →
    ``(alpha, max(1, ceil(2/alpha − 1)))`` — the equivalent span, so ``alpha = 2/(w+1)``
    reproduces warmup ``w`` exactly for every ``w`` below 5,935,198 (rounded to 9 places before
    the ceiling, so the float reciprocal cannot land one bar late); past that the reciprocal's
    error, which grows with ``w``, outgrows the 9-place rounding and the alpha form can warm ONE
    bar later than the window form (the same ``alpha``, so the same values wherever both emit).
    The ceiling's own pair is exact — ``MIN_EW_ALPHA`` is a power of two, so ``alpha = 2**-30``
    IS ``window = MAX_BAR_COUNT_PARAM``. RiskMetrics 0.06 warms 33 observations. Wilder's 1/N
    smoother is ``alpha = 1/N`` with THIS seed (the first finite value), not an SMA seed.

    The span is a bar count, held to ``MAX_BAR_COUNT_PARAM`` like every other: a span past it —
    a window over the ceiling, a positive alpha below ``MIN_EW_ALPHA`` (``1e-19`` implies a
    2e19-bar warmup), an alpha whose span is not finite (``5e-324``, ``0``, NaN) — raises
    ``ValueError`` here, naming the domain. One refusal for the whole range: a span short of 2**64
    used to come back and run all warmup (from 2**63 on numba typed it unsigned, so it crashed
    instead — "int too big to convert" — wherever the same kernel already held its signed
    specialization in the process: the outcome hung on evaluation order), a longer one reached a
    kernel as an integer no machine type holds (a numba ``TypingError`` / ``OverflowError``), and
    a non-finite one failed inside ``math.ceil`` (or divided by zero at 0). A window that is no
    bar count at all — below 1, or fractional — raises its own ``ValueError``: ``0`` and the
    negatives made ``2/(window+1)`` a decay outside ``(0, 1]`` (``-1`` divided by zero,
    ``-10**20`` reached the kernel as an integer it cannot take) and ``2.5`` was truncated to a
    2-bar warmup its own decay does not imply. The DSL refuses the same values at parse
    (``dsl.nodes``: every EW window is a ``PosInt`` / ``Ge2Int``, shared axes and sweep levels
    re-validated through them); this is defense in depth for a library caller reaching the kernels
    around the schema (the kernels' own ``0 < alpha <= 1`` guard still answers an alpha outside it
    — negative, above 1, infinite, warmup 1 here — with NaN). A span within the ceiling but longer
    than the series is legal and keeps the ordinary outcome: every bar is warmup and the kernel
    never emits.
    """
    if (window is None) == (alpha is None):
        raise ValueError("ema takes exactly one of 'window' or 'alpha'")
    if window is not None:
        if window > MAX_BAR_COUNT_PARAM:
            raise ValueError(
                f"EW window {window} exceeds the bar-count ceiling MAX_BAR_COUNT_PARAM = "
                f"{MAX_BAR_COUNT_PARAM}"
            )
        # ``>= 1`` first: NaN and -inf fail it, so ``int()`` only ever sees a finite value.
        if not (window >= 1 and window == int(window)):
            raise ValueError(
                f"EW window {window} is not a bar count: the span form takes a whole number of "
                "bars, at least 1"
            )
        return 2.0 / (window + 1.0), int(window)
    a = float(alpha)  # type: ignore[arg-type]  # narrowed by the xor above
    # Python float division never overflows to an exception (``2.0 / 5e-324 == inf``); only an
    # exact zero raises, so it is mapped onto the infinite span it denotes.
    span = round(2.0 / a - 1.0, 9) if a != 0.0 else math.inf
    if not (math.isfinite(span) and span <= MAX_BAR_COUNT_PARAM):
        raise ValueError(
            f"EW alpha {a!r} implies an equivalent span 2/alpha − 1 of {span!r} bars, "
            f"past the bar-count ceiling: the span must be finite and at most "
            f"MAX_BAR_COUNT_PARAM = {MAX_BAR_COUNT_PARAM}, i.e. alpha >= MIN_EW_ALPHA = "
            f"{MIN_EW_ALPHA!r} (2**-30)"
        )
    return a, max(1, math.ceil(span))


#: The EW z-score's recurrence (``zscore_ema_1d``) runs in a FRAME lifted by ``2**_LIFT`` — at most
#: twice, which takes every nonzero double past ``2**126`` — once its variance is under
#: ``2**-600`` while the input, the mean and the deviation are all under ``2**-300``: lifted, the
#: variance stays under ``2**600`` and the values under ``2**300``, and as one step shrinks a
#: variance by ``1 − α`` (``2**-53`` at the least) none leaves the normal range before the frame
#: moves. It lowers again once an input passes ``2**400`` in it — the gap keeps a series that
#: hovers from toggling.
_EW_LIFT_VAR = 2.0**-600
_EW_LIFT_AT = 2.0**-300
_EW_LOWER_AT = 2.0**400
_EW_MAX_LIFT = 2 * _LIFT


@njit(cache=True)
def zscore_ema_1d(
    arr: npt.NDArray[np.float64], alpha: float, warmup: int
) -> npt.NDArray[np.float64]:
    """EW z-score via the West variance recurrence: ``d = x − μ; μ += α·d; s = (1−α)(s + α·d²)``.

    Algebraically identical (in exact arithmetic) to the EW[x²] − EW[x]² pair it replaces, but
    computed on centered quantities, so it is translation-invariant in floating point — the
    one-pass form lost its mantissa to cancellation at large input levels and drifted with the
    input offset. Same seeding (first finite value → μ = x, s = 0), same NaN-skipping warmup
    (``warmup`` observations, from ``ema_params``), same ``s > 0`` emission gate.

    The mean update is a convex combination, clamped into ``[min(μ, x), max(μ, x)]`` like
    ``ema_1d``'s. On a HELD input (the same value again) exact arithmetic moves the deviation
    ``x − μ`` to ``(1−α)·(x − μ)`` every bar — it decays, keeping its sign — which μ can no
    longer represent once its update is absorbed (``α·d`` below half an ulp of μ) or lands on the
    value: from there the deviation is CARRIED exactly, ``r ← (1−α)·r``, and the z-score reads
    ``r / √s`` until the input moves. Without the carry μ sticks a few ulps off, ``d`` freezes at
    that residue, ``s`` settles at ``(1−α)·d²`` and the z-score reads exactly ``±1/√(1−α)`` for as
    long as the input holds; zeroing the deviation instead would flip the sign every ``z > 0``
    read takes on the stretch. Only those held bars, an update that left its exact interval and
    the out-of-range steps below differ from the plain recurrence.

    The carry starts on a held bar only, so a hold whose μ is ON the held value from the bar the
    input MOVED carries nothing: it reads z 0.0 off an exact deviation the kernel cannot know (μ
    carries rounding) — the documented residual, of two origins. The LANDING GAP: a step of a few
    ulps at ``α ≥ 0.5`` (an ``ema`` converging onto a held price moves in 1-ulp steps) whose
    update lands μ ON its input, where exact arithmetic keeps the step's sign (a carry started
    there would move ordinary data). And an input that moves exactly ONTO μ (5.0, 5.5, then 5.25
    held, at window 3), whose exact z is 0 where μ is exact and of either sign where it is not.
    Its variance decays with no deviation to feed it. At ``α ≥ 0.5`` it rounds to 0.0 and the z
    is undefined from there — the refusal the ledger records, never floored into a decided 0.0
    for the rest of the hold — after ~490 held bars at α 0.75 off a few-ulp step at unit levels
    and ~1070 at α 0.5 in the example (fewer off a smaller step; under ``2**-300``, where the
    frame below lifts it, earlier or later with the level). Below ``α = 0.5`` round-to-nearest
    holds it at a subnormal (``(1−α)·5e-324`` rounds back to 5e-324), so a hold moved onto μ there
    reads a decided z 0.0 for its whole length — exact where μ is, of unknown sign where it is
    not; no landing gap feeds it, the exact update lying nearer μ than its input.

    Out of range: the squares of deviations under ~1.5e-154 lose their bits, and an input that
    DECAYS geometrically — an ``ema`` of a change over a held stretch, from ordinary levels — gets
    there with a z exact arithmetic keeps constant. Read as computed, ``s`` rounded to subnormals
    or to 0.0: the z read past its bound ``√((1−α)/α)``, NaN, or (floored) collapsed toward 0 —
    deciding thresholds the exact z does not. The recurrence therefore runs in a frame lifted by
    ``2**_LIFT`` once its variance nears the range (``_EW_LIFT_VAR``), lowered once the input
    returns. Scaling by a power of two never rounds, so every step in the frame is the plain step
    times its power of two: the z is the recurrence's value in an unbounded exponent range, and a
    series whose plain recurrence stays in the normal range keeps every bit. Out of the frame's
    reach is only a HELD input's decay (a carried deviation, a mean decaying onto a held 0.0, or
    a mean already ON the held value, the residual above), whose z exact arithmetic sends to 0
    like ``(1−α)**(t/2)``: under ``2**-158/α`` (~5e-47 at RiskMetrics' 0.06; ~1e-140 on a hold at
    ordinary levels) once ``s`` leaves the normal range in the frame. Below ``α = 1`` exact
    arithmetic keeps ``s`` and the carried deviation nonzero once the input has moved; where the
    deviation is KNOWN nonzero — carried, or this step's — an update that rounded either to 0.0
    is floored at the smallest subnormal, sign kept: the z stays decided, with the carried sign
    and an imprecise magnitude. After a step whose deviation IS 0.0 the exact one is unknown (the
    residual above), and a floored ``s`` would decide the z 0.0 for the rest of the hold: its
    ``s`` keeps the 0.0 it rounds to at ``α ≥ 0.5``, undefined. An ``s`` that OVERFLOWED is
    carried forward: undefined from there on. At ``α = 1`` (the DSL refuses it) the mean is the
    last value and ``s`` IS 0: undefined throughout.
    """
    n = arr.shape[0]
    out = np.full(n, np.nan)
    if warmup < 1 or not (0.0 < alpha <= 1.0):
        return out
    mu = np.nan
    s = 0.0
    seen = 0
    prev = np.nan
    carry = np.nan  # the exact deviation x − μ while a held input's update cannot move μ
    lift = 0  # the frame's exponent: μ, s (twice over), the carry and x are held times 2**lift
    for i in range(n):
        raw = arr[i]
        if np.isnan(raw):
            continue
        x = raw
        if lift > 0:
            # the input is back in range: lower the frame (exact wherever the state is normal
            # there; what rounds is a decayed remnant the input's own deviation swamps)
            while lift > 0 and abs(raw) > math.ldexp(_EW_LOWER_AT, -lift):
                lift -= _LIFT
                mu = math.ldexp(mu, -_LIFT)
                s = math.ldexp(s, -2 * _LIFT)
                carry = math.ldexp(carry, -_LIFT)
            x = math.ldexp(raw, lift)
        if np.isnan(mu):
            mu = x
            s = 0.0
            carry = np.nan
        elif raw == prev and not np.isnan(carry):
            d = carry
            s = (1.0 - alpha) * (s + alpha * d * d)
            carry = (1.0 - alpha) * d
            # Exact arithmetic never reaches zero below alpha 1: floor the carried pair at the
            # smallest subnormal, sign kept (round-to-nearest already does so for alpha < 0.5;
            # without it a long hold at alpha >= 0.5 read undecidable, or lost its sign). At
            # alpha 1 the EW variance IS 0 — the mean is the last value — and stays undefined.
            if s == 0.0 and alpha < 1.0:
                s = 5e-324
            if carry == 0.0:
                carry = 5e-324 if d > 0.0 else -5e-324
        else:
            d = x - mu
            v = mu + alpha * d
            lo = min(mu, x)
            hi = max(mu, x)
            moved = lo if v < lo else (hi if v > hi else v)
            s_next = (1.0 - alpha) * (s + alpha * d * d)
            # The same floor where this step's deviation is KNOWN nonzero — defense in depth: the
            # frame keeps s normal across every step whose d² it loses, and a mean decaying onto
            # a held 0.0 lands, its deviation then carried, while s is still normal. Never at
            # d == 0.0, though exact arithmetic keeps s > 0 there too: a mean ON a held value
            # with nothing carried (at ANY level) reads z 0.0 off an exact deviation unknown here
            # — the landing gap's keeps the step's sign; an input moved exactly onto μ may be
            # exactly 0, or either sign off a μ that carries rounding — so its decaying s keeps
            # the 0.0 it rounds to at alpha >= 0.5: undefined. Floored, it decided that 0.0 for
            # the whole hold. Below alpha 0.5 round-to-nearest holds that s at a subnormal and
            # the 0.0 stays decided: the documented residual, never a landing gap's there.
            if s_next == 0.0 and alpha < 1.0 and d != 0.0:
                s_next = 5e-324
            s = s_next
            # On a held input whose update cannot move μ (absorbed, or landing on x) the exact
            # deviation is carried; μ takes the rounded update either way, so a landing leaves
            # no stale mean once the input moves.
            held = raw == prev and d != 0.0 and moved in (mu, x)
            carry = (1.0 - alpha) * d if held else np.nan
            if carry == 0.0:  # underflowed at alpha >= 0.5: floored, sign kept, as held below
                carry = 5e-324 if d > 0.0 else -5e-324
            mu = moved
        prev = raw
        seen += 1
        dev = (x - mu) if np.isnan(carry) else carry
        # the variance nears the subnormal range with every value it is built from small: lift
        if (
            lift < _EW_MAX_LIFT
            and s < _EW_LIFT_VAR
            and max(abs(x), abs(mu), abs(dev)) < _EW_LIFT_AT
        ):
            lift += _LIFT
            mu = math.ldexp(mu, _LIFT)
            s = math.ldexp(s, 2 * _LIFT)
            carry = math.ldexp(carry, _LIFT)
            dev = math.ldexp(dev, _LIFT)
        if seen < warmup:
            continue
        if s <= 0.0 or not np.isfinite(s):
            continue
        z = dev / np.sqrt(s)
        if np.isfinite(z):
            out[i] = z
    return out


@njit(cache=True)
def ema_1d(arr: npt.NDArray[np.float64], alpha: float, warmup: int) -> npt.NDArray[np.float64]:
    """EMA with decay ``alpha`` (``ema_params``): seeded at the first finite value, NaN-skipping,
    NaN until ``warmup`` observations — the same warmup convention as the EMA inside
    ``zscore_ema``.

    The update is a convex combination, so its exact value lies between ``e`` and ``x`` —
    strictly, for ``alpha < 1``: it never reaches ``x``. The rounded one is held there, one ulp
    short of ``x`` at the closest. Rounding could otherwise leave the interval: a constant input
    drifted a few ulps off itself (``x > ema(x)`` then read a sign on a flat feed), and a held
    stretch could settle on the wrong side of the held value or ON it (``x <= ema(x)`` then fired
    on held bars exact arithmetic keeps above the mean). The guard binds only where the rounded
    update left the open interval — every other step is bit-identical."""
    n = arr.shape[0]
    out = np.full(n, np.nan)
    if warmup < 1 or not (0.0 < alpha <= 1.0):
        return out
    e = np.nan
    seen = 0
    for i in range(n):
        x = arr[i]
        if np.isnan(x):
            continue
        if np.isnan(e):
            e = x
        else:
            v = alpha * x + (1.0 - alpha) * e
            lo = min(e, x)
            hi = max(e, x)
            v = lo if v < lo else (hi if v > hi else v)
            # The exact update lies STRICTLY between e and x (alpha < 1) — it never reaches x, so
            # a rounded landing on x keeps the nearest value on e's side instead.
            if v == x and e != x and alpha < 1.0:
                v = np.nextafter(x, e)
            e = v
        seen += 1
        if seen >= warmup:
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

    ``defined`` marks bars where the child's truth is KNOWN. A hole (``init & ~defined``) could
    have been either value, and whether a later True is a false→true edge — or falls inside a
    cooldown the hole may have armed — can depend on it for as long as the completions' machines
    stay apart. So the kernel carries EVERY completion forward: the set of machine states
    ``(prev, seen_false, cool_left)`` some completion of the holes so far is in, each advanced
    under the known value on a defined bar and under both values on a hole. ``out`` is True where
    every world fires; ``defined_out`` is True exactly where every world agrees on fire/no-fire —
    however far from the hole that is (a hole that may have shifted an alternating child's
    cooldown phase leaves the worlds disagreeing on every other bar for good: genuinely
    unknowable, and the ledger then refuses — the fail-closed direction). The set re-converges
    wherever the completions' machines meet (e.g. after a run of ``max(1, cooldown)`` defined
    Falses); a fire every world agrees on follows that, never causes it — every world fires only
    when every world is in the one ARMED state ``(F, T, 0)``. Warmup resets it to the one warmup
    state (warmup wins over a hole: it is vacuously defined). A hole-free series is a single world,
    i.e. the deterministic machine, bit for bit. A hole bar is therefore decided (False) where no
    world could fire on it — every world cooling down, already True, or not yet past its first False
    — and at ``cooldown == 0`` a bar right after a hole is undecided only if it is True. A hole
    whose worlds agree on every bar — the hole bar included — until they re-converge is ABSORBED, as
    Kleene ``F∧U = F`` absorbs an operand's: the node reads decided throughout, and a raw leaf's
    hole there reaches the gate through the source panel (``source_coverage``) alone. Deciding the
    hole bar is not enough for that: a hole no world could fire on is absorbed where every world's
    cooldown runs on past it (the later cooldown bars overwrite the ``prev`` it wrote) or a decided
    False follows it (every free world then lands on ``(F, T)``); otherwise a True right after it is
    an edge in exactly the hole's False completion — a hole on a cooldown's LAST bar, right after a
    True, or before the child's first False reads undecided on the True that follows it.

    Representation (exact; O(1) amortized per bar): a COOLING world ignores ``prev`` (every
    cooldown bar overwrites it before it is read) and has ``seen_false`` set (it fired), so its
    fire bar alone identifies it. Every world that fires on a bar lands in that one state, so a
    bar pushes at most one fire bar — in bar order: the cooling worlds form a FIFO — and pops at
    most one, the world whose cooldown ends on it (``i - fire == cooldown``, which cannot
    overflow), free from the next bar with this bar's value as ``prev``. The FIFO is a growable
    ring buffer; the FREE worlds are a 4-bit set over ``(prev, seen_false)``. The live set holds
    at most ``4 + cooldown`` worlds and the buffer at most ``min(cooldown, bars)`` fire bars —
    nothing is sized by ``cooldown`` up front.
    """
    n = value.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    defined_out = np.ones(n, dtype=np.bool_)
    # Free worlds: bit ``1 << (2*prev + seen_false)`` — 1 is the warmup state (F, F), 2 the one
    # ARMED state (F, T) a True fires, 4 is (T, F) and 8 is (T, T).
    free = 1
    fires = np.empty(8, dtype=np.int64)  # cooling worlds' fire bars, ascending from `head`
    head = 0
    size = 0
    for i in range(n):
        if not init[i]:
            # Warmup: do not arm a transition from pre-init False, and forget every world.
            # Vacuously defined — the caller's `init` gate already excludes these bars.
            free = 1
            head = 0
            size = 0
            continue
        known = defined[i]
        may_true = value[i] or not known
        may_false = not value[i] or not known
        armed = (free & 2) != 0
        some_fire = armed and may_true
        all_fire = some_fire and known and free == 2 and size == 0
        out[i] = all_fire
        defined_out[i] = all_fire or not some_fire
        nxt = 0
        if may_false and free != 0:
            nxt |= 2  # a False latches seen_false, whatever the free world
        if may_true:
            if free & 5:
                nxt |= 4  # (F, F) and (T, F) cannot fire on a True: they move to (T, F)
            if free & 8:
                nxt |= 8
        if size > 0 and i - fires[head] == cooldown:
            # The oldest cooling world takes its last cooldown step on this bar and is free from
            # the next, with this bar's value as prev.
            head = (head + 1) % fires.shape[0]
            size -= 1
            if may_false:
                nxt |= 2
            if may_true:
                nxt |= 8
        if some_fire:
            if cooldown <= 0:
                nxt |= 8
            else:
                if size == fires.shape[0]:
                    grown = np.empty(2 * size, dtype=np.int64)
                    for k in range(size):
                        grown[k] = fires[(head + k) % size]
                    fires = grown
                    head = 0
                fires[(head + size) % fires.shape[0]] = i
                size += 1
        free = nxt
    return out, defined_out


# =============================================================================
# The event-anchor family — Series reads off a Condition's tradable signal.
#
# Shared definition (``dsl.nodes``): the EVENT is the signal ``value & init & defined``; ``s(t)`` is
# the latest bar <= t at which it fired, the current bar INCLUDED; latest wins. Before the first
# event the anchor is warmup (``s = -1`` while ``known``); after a post-warmup HOLE in the
# condition (``init & ~defined``) the anchor is UNKNOWN (``s = -1``, ``~known``) until the next
# defined event — the callers turn that into "initialized but NaN", the ledgered form, never
# silent warmup. Every kernel is a causal function of the inputs through ``t``.
# =============================================================================


@njit(cache=True)
def event_anchor_1d(
    sig: npt.NDArray[np.bool_], init: npt.NDArray[np.bool_], defined: npt.NDArray[np.bool_]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """``(s_idx, known)`` per bar: the latest event index (−1 when there is none or it is
    unknowable) and whether the anchor is KNOWN (False only after a post-warmup hole, until the
    next defined event).

    ``sig`` is the condition's tradable signal, ``init``/``defined`` its own channels. A warmup
    bar resets to "no event yet" (``init`` is monotone, so this is defense in depth); a hole
    drops the anchor into the unknown state; a defined event bar re-anchors and re-establishes
    knowledge. A defined non-event bar after a hole stays unknown — whether an event fell inside
    the hole is exactly what cannot be known."""
    n = sig.shape[0]
    s_idx = np.full(n, -1, dtype=np.int64)
    known_out = np.ones(n, dtype=np.bool_)
    s = -1
    known = True
    for t in range(n):
        if not init[t]:
            s = -1
            known = True
        elif not defined[t]:
            known = False
        elif sig[t]:
            s = t
            known = True
        s_idx[t] = s if known else -1
        known_out[t] = known
    return s_idx, known_out


@njit(cache=True)
def event_anchor_apply_nb(
    sig: npt.NDArray[np.bool_], init: npt.NDArray[np.bool_], defined: npt.NDArray[np.bool_]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    rows, cols = sig.shape
    s_idx = np.empty((rows, cols), dtype=np.int64)
    known = np.empty((rows, cols), dtype=np.bool_)
    for j in range(cols):
        s_idx[:, j], known[:, j] = event_anchor_1d(sig[:, j], init[:, j], defined[:, j])
    return s_idx, known


def bars_since_event_apply_nb(s_idx: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """``t − s(t)`` where the anchor is known and exists (0 on the event bar); NaN otherwise."""
    s = np.asarray(s_idx, dtype=np.int64)
    t = np.arange(s.shape[0], dtype=np.int64).reshape(-1, *([1] * (s.ndim - 1)))
    return np.where(s >= 0, (t - s).astype(float), np.nan)


def event_value_apply_nb(
    arr: npt.NDArray[np.float64], s_idx: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    """``x[s(t)]`` — the input's value AT the latest event, held until the next one; NaN where
    there is no known anchor or the input was NaN on the event bar (a later hole in the input is
    irrelevant: the snapshot was taken)."""
    x = np.asarray(arr, dtype=float)
    s = np.asarray(s_idx, dtype=np.int64)
    if x.shape != s.shape:
        raise ValueError(f"event_value shape mismatch: {x.shape} vs {s.shape}")
    taken = np.take_along_axis(x, np.maximum(s, 0), axis=0)
    return np.where((s >= 0) & np.isfinite(taken), taken, np.nan)


@njit(cache=True)
def event_agg_1d(
    sig: npt.NDArray[np.bool_],
    init: npt.NDArray[np.bool_],
    defined: npt.NDArray[np.bool_],
    x: npt.NDArray[np.float64],
    mode: int,
) -> npt.NDArray[np.float64]:
    """``agg(x[s(t) .. t])`` — the running aggregate since the latest event, reset AT each event
    (the event bar included). ``mode`` 0 = sum, 1 = max, 2 = min, 3 = mean (sum / (t − s + 1)).

    STRICT finite gate, the ``rolling_agg`` rule: one NaN input bar inside ``[s, t]`` POISONS the
    aggregate until the next event. A warmup bar carries no state; a hole bar drops the anchor
    (nothing is emitted until the next defined event, and the hole bar itself never contributes).
    Only finite results are emitted (an overflowed sum reads NaN)."""
    n = sig.shape[0]
    out = np.full(n, np.nan)
    anchored = False
    acc = 0.0
    mx = 0.0
    mn = 0.0
    cnt = 0
    poisoned = False
    for t in range(n):
        if not init[t]:
            anchored = False
            continue
        if not defined[t]:
            anchored = False
            continue
        if sig[t]:
            anchored = True
            acc = 0.0
            mx = -np.inf
            mn = np.inf
            cnt = 0
            poisoned = False
        if not anchored or poisoned:
            continue
        v = x[t]
        if not np.isfinite(v):
            poisoned = True
            continue
        cnt += 1
        acc += v
        if v > mx:
            mx = v
        if v < mn:
            mn = v
        if mode == 0:
            r = acc
        elif mode == 1:
            r = mx
        elif mode == 2:
            r = mn
        else:
            # an all-equal stretch's mean IS its value (the zero-dispersion note above)
            r = mx if mx == mn else acc / cnt
        if np.isfinite(r):
            out[t] = r
    return out


@njit(cache=True)
def _event_agg_apply(
    sig: npt.NDArray[np.bool_],
    init: npt.NDArray[np.bool_],
    defined: npt.NDArray[np.bool_],
    x: npt.NDArray[np.float64],
    mode: int,
) -> npt.NDArray[np.float64]:
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = event_agg_1d(sig[:, j], init[:, j], defined[:, j], x[:, j], mode)
    return out


_EVENT_AGG_MODES = {"sum": 0, "max": 1, "min": 2, "mean": 3}


def event_agg_apply_nb(
    sig: npt.NDArray[np.bool_],
    init: npt.NDArray[np.bool_],
    defined: npt.NDArray[np.bool_],
    arr: npt.NDArray[np.float64],
    agg: str,
) -> npt.NDArray[np.float64]:
    """2-D ``event_agg_1d`` per column; ``agg`` is one of ``sum``/``max``/``min``/``mean``."""
    mode = _EVENT_AGG_MODES.get(agg)
    if mode is None:
        raise ValueError(f"unknown event_agg agg: {agg!r}")
    x = np.asarray(arr, dtype=float)
    if not (x.shape == sig.shape == init.shape == defined.shape):
        raise ValueError(f"event_agg shape mismatch: {x.shape} vs {sig.shape}")
    return _event_agg_apply(
        np.ascontiguousarray(sig, dtype=np.bool_),
        np.ascontiguousarray(init, dtype=np.bool_),
        np.ascontiguousarray(defined, dtype=np.bool_),
        np.ascontiguousarray(x),
        mode,
    )


def mask_apply_nb(
    value: npt.NDArray[np.bool_],
    init: npt.NDArray[np.bool_],
    defined: npt.NDArray[np.bool_],
    full: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """The condition as a number where it was decided with every condition under it past warmup
    (``full & defined``): 1.0 where its tradable signal holds (``value & init``), 0.0 where it
    does not. NaN while ANY branch warms — an ``or`` that is False then has decided nothing, and
    one that is True then would make a series of its Trues alone (a value-selected history for
    every reader that skips NaN) — or where it is undecidable."""
    decided = full & init & defined
    return np.where(decided, np.where(value, 1.0, 0.0), np.nan)


# =============================================================================
# 2D apply wrappers for the numba kernels (rows × cols → rows × cols, column-wise).
# =============================================================================


@njit(cache=True)
def zscore_ema_apply_nb(
    arr: npt.NDArray[np.float64], alpha: float, warmup: int
) -> npt.NDArray[np.float64]:
    out = np.empty_like(arr)
    for j in range(arr.shape[1]):
        out[:, j] = zscore_ema_1d(arr[:, j], alpha, warmup)
    return out


@njit(cache=True)
def ema_apply_nb(
    arr: npt.NDArray[np.float64], alpha: float, warmup: int
) -> npt.NDArray[np.float64]:
    out = np.empty_like(arr)
    for j in range(arr.shape[1]):
        out[:, j] = ema_1d(arr[:, j], alpha, warmup)
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
