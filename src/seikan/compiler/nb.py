"""Transform kernels for the DSL (``compiler/transforms.py`` dispatches DSL nodes here).

Two implementation styles, **no vectorbt**:

* **Vectorized numpy** for the windowed/elementwise kernels — z-score (SMA), percentile,
  rolling/expanding agg, drawdown/runup, change (pct/log/diff), shift, unary op, rolling
  Pearson correlation of two series, and the cross-sectional rank/demean/aggregate (axis=1,
  across the target columns at each bar — the only kernels with no 1-D form). Windowed kernels
  use ``numpy.lib.stride_tricks.sliding_window_view`` (a window requires every bar finite, else
  NaN), reduced in bounded row batches (``_window_batches``); elementwise ones use plain lag
  slicing. No Python time-loop.
* **Numba** ``@njit`` for the genuinely sequential kernels — the EMA / EMA-z-score recurrences
  (``(alpha, warmup)`` from ``ema_params``: the span form ``alpha = 2/(window+1)`` or an explicit
  decay), ``bars_since_extremum``, the ``first_true`` episode-entry latch,
  the event-anchor family (``event_anchor`` — the latest-event index ``s(t)`` a Condition's
  tradable signal defines — and ``event_agg``, the resettable running aggregate since it; the
  ``bars_since_event`` / ``event_value`` / ``mask`` reads off those are plain numpy), and the
  exact sums the decision-side means and deviations read (``_rolling_exact_sums``,
  ``_row_exact_sums``, ``cross_deviations``).
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
from collections.abc import Callable, Iterator
from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import njit
from numpy.lib.stride_tricks import sliding_window_view

from seikan.constants import MAX_BAR_COUNT_PARAM, MIN_EW_ALPHA
from seikan.numeric import MIN_NORMAL, log_ratio_array

# =============================================================================
# Vectorized numpy kernels (windowed + elementwise) — no Python time-loop.
#
# The 2D ``*_apply_nb`` form is the implementation, columnwise by construction (the parity tests
# pin each column against the frozen reference). A windowed output is
# NaN unless every bar in the trailing window is finite (``sliding_window_view(...).any``), matching
# the frozen reference's ``cnt == window`` gate.
#
# BOUNDED window stacks: the trailing windows are a ``(rows − window + 1, cols, window)`` view, and
# every reduction over it — a moment, a sort, a comparison, even the ``isnan(...).any`` gate —
# materializes temporaries of that whole size: a peak of 162 MiB for 2,000 bars × 16 targets at
# window 252 (a 0.24 MiB input; 270 MiB for a correlation), ~13–22 GiB at 5,000 × 500. The
# kernels reduce the view in row batches of at most ``_WINDOW_STACK`` values
# (``_window_batches``). A batch is a basic slice of the view, with its strides, so numpy reduces
# each window in the same order and every output keeps its bits; what reads the whole input (the
# constant windows, the exact sums, the lifted values) is computed once per call. The path side's
# first extremum (``paths._first_extremum_1d``) never stacks its windows either.
#
# FINITE-OR-NaN invariant: every kernel's output is finite or NaN. Kernels whose arithmetic can
# overflow from finite inputs (sums, ratios with denormal denominators, squared deviations)
# sanitize their results — a non-finite value becomes NaN, so it never latches its series
# initialized and never fires a threshold. Where that NaN lands is the node's warmup latch's call
# (``vectorize._build_series``), never the overflow's: once the series has initialized, in the
# undecidable ledger; before, it is the series' warmup exactly as a leading ``x / 0`` is, and no
# ledger records it. A latch initializes a series no earlier than its first finite value, save one
# arm that initializes it with no value yet: an event node whose anchor became unknowable before its
# first event, a ``mask`` undecided on its first live bar, a transform carrying state across bars
# from its input's first post-init hole (``_transform`` — these kernels skip a NaN, so the builder
# gives the series no value from there), and (``_derived_latch``) a series on a bar every input is
# initialized and one has none since its init. A ``cross_agg`` never takes it —
# its broadcast waits for its own first finite value as well as for the member's own first value. So
# a series NaN on every bar — a ``zscore`` of plain levels near 1e160, any ``cross_agg`` — never
# initializes, never fires and is counted by no coverage ledger, while one that arm initialized
# ledgers every overflowed bar from there on. Variances are computed CENTERED (two-pass or a
# Welford/West recurrence), never as E[x²] − E[x]²: the one-pass form cancels catastrophically on
# large-magnitude series (a price near 5000 with a std near 1 has already lost most of the
# mantissa). The two-pass windows are centered about the window's OWN offset (its first value; a
# cross-section's range midpoint — ``_window_moments``, ``cross_centre``), never about the rounded
# mean of the raw values: that mean's rounding is the size of the level's ulp, and a spread at that
# scale (``[2**52, 2**52 + 1]``) read a std of 0.707 for 0.5 and a z-score past its bound.
#
# EXACT means: every mean a decision reads — ``rolling_agg``'s and ``cross_agg``'s ``mean`` — and
# the runner's ``cross_mean`` bench leg (``cross_means``) is its window's or row's EXACTLY ROUNDED
# sum over its count, the read ``analysis._pools.exact_mean`` gives the graded pools (the centre
# ``cross_agg``'s std reduces about is the row's offset plus its differences' exactly rounded mean,
# ``cross_centre``, whose subtractions round); a deviation from a cross-section's mean
# that is read (``cross_demean``, the ``cross_mean`` excess) has its numerator ``k·x − S`` summed
# exactly (``cross_deviations``), so a member AT the mean reads exactly 0.0 — about the rounded
# centre ``[0, 1, 1, −1 − 2**-52, −1 + 2**-52]``'s first member read 2.2e-17, and ``x == 0 AND
# cross_demean(x) > 0`` fired. The sums are Shewchuk expansions (``_expansion_add``, the
# ``math.fsum`` algorithm ``event_agg``'s running sum keeps): sliding along a column
# (``_rolling_exact_sums``, O(rows) whatever the window) or one per row (``_row_exact_sums``).
# numpy sums pairwise, and a large term swallowed a small one before its cancelling partner
# arrived: ``[1, −2**-54, −1, 2**-54, 0]`` sums to exactly 0 and its mean read 1.1e-17, so
# ``cross_agg(x, mean) > 0`` fired on it — and read 0 after a permutation of the members, the
# decision hanging on the order the targets were declared in. The mean has the exact sum's sign and
# is 0.0 exactly where the sum is, in any member order; it is NaN where an intermediate sum leaves
# the double range (members near it), as the direct form's overflow was.
# The centre a window's variance is reduced about (``_window_moments``, ``_pair_moments``) stays the
# direct two-pass form about the window's offset: no threshold compares against it, and the
# moments, the correlation and the z-score read off it are known to the rounding of the window's
# own spread (a z-score within that rounding of 0 can read 0 or the other sign).
#
# OUT-OF-RANGE intermediates: sanitizing the RESULT is not enough where an out-of-range intermediate
# yields a FINITE one — ``x / inf`` is 0.0, so a correlation whose variance product overflowed read
# 0.0 for ``corr(x, x)`` at 1e80 scale, and a z-score whose window variance overflowed read 0.0. A
# window variance that OVERFLOWED (deviations past ~1.3e154) is therefore NaN wherever it is divided
# by or reported, never a value: in the undecidable ledger once its series has initialized, and the
# series' warmup before that (the note above) — a z-score of plain data whose every window
# overflows, its first included, never fires and no ledger counts it, like a leading ``x / 0``. One
# that fell BELOW the normal range is no overflow but a precision loss, and ordinary data reaches
# it: an ``ema`` of a change decays geometrically through it over a held stretch. The squares of
# deviations under ~1.5e-154 round to multiples of the smallest subnormal or to 0.0, reading a
# z-score past its population bound ``√(w−1)``, a correlation off in its third digit, or a varying
# window's std as the 0.0 an all-equal window owns. Such a VARYING window is recomputed LIFTED by
# ``2**_LIFT`` (``_lifted_windows``): scaling up by a power of two never rounds, and the lifted
# windows keep the direct form's layout — the same summation order — so every moment is the direct
# one times its power of two, and the z-score, correlation and std are the values the direct form
# has in an unbounded exponent range (a std under the smallest subnormal floors there: a varying
# window is never zero dispersion). The correlation's variance PRODUCT of two normal variances is
# taken with the exponents moved out wherever it leaves the normal range (``_sqrt_product``), to the
# same exactness. Every window whose variances and, for ``rolling_corr``, their product are normal
# doubles keeps every bit. The EW z-score (``zscore_ema_1d``) carries its variance in a recurrence,
# which the same rule reaches by moving the whole recurrence into a frame lifted by ``2**_LIFT``
# once its variance nears the range (``_EW_LIFT_VAR``): its z is the recurrence's value in an
# unbounded exponent range too — not NaN, nor collapsed toward 0 off a floored variance, nor past
# its bound ``√((1−α)/α)``. Past the frame's reach only a HELD input's decay is left, a z exact
# arithmetic has sent under ``2**-158/α`` by then: its variance and its deviation are floored at the
# smallest subnormal there, sign kept — decided, imprecise in magnitude.
#
# EXACT zero dispersion: a window (or a cross-section) whose finite values are all equal is
# detected exactly and given its exact answer instead of the arithmetic's: the mean of such a
# window is the value itself, its deviations and std are 0.0, and every read that divides by its
# dispersion (z-score, correlation) is undefined → NaN. The computed mean of a constant DECIMAL
# window is not the constant (``(0.1 + 0.1 + 0.1) / 3 != 0.1``), so without the exact test the
# residual minted a z-score of exactly ±1 and a correlation of 1 out of a constant series. The
# recurrences hold the same line (``ema_1d``, ``zscore_ema_1d``, ``event_agg_1d``). Every window
# with any variation whose variances and, for ``rolling_corr``, their product are normal doubles
# is the direct two-pass form about its offset, bit for bit (a correlation clipped to ``[−1, 1]``,
# where an exactly linear window's rounding carried it past; ``ema_1d`` differs from its plain
# update only on the steps whose rounded update left its exact interval, and the EW z-score carries
# its mean compensated — the plain West recurrence's value to its rounding, not bit for bit; the
# out-of-range windows and the lifted EW frame are the note above);
# "constant" means bit-identical
# values — arithmetic on decimal levels (a spread ``y1 − y2`` moving in lockstep) leaves residue
# at the operands' scale, which is variation as far as any kernel can tell.
# =============================================================================


#: The most values a windowed kernel materializes at once (512 KiB a temporary): see the bounded
#: window stacks note above.
_WINDOW_STACK = 2**16


def _window_batches(shape: tuple[int, ...]) -> Iterator[slice]:
    """Row slices of a ``(rows, cols, window)`` window stack holding at most ``_WINDOW_STACK``
    values each (one row at the least)."""
    step = max(1, _WINDOW_STACK // math.prod(shape[1:]))
    for start in range(0, shape[0], step):
        yield slice(start, min(start + step, shape[0]))


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


def _normal(v: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Elementwise: ``v`` is a positive NORMAL double — a variance that carries its full
    precision (see the out-of-range note above). False on 0.0, a subnormal, ``inf`` and NaN.
    ``analysis._pools._normal`` is the pool statistics' twin, which nulls a variance out of the
    normal range on BOTH sides (it says why); both read the one :data:`seikan.numeric.MIN_NORMAL`.
    """
    ok: npt.NDArray[np.bool_] = (v >= MIN_NORMAL) & (v < np.inf)
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
    """The two-pass centered population variance over the LAST axis, and the last bar's
    deviation from the window mean — computed around the window's OWN offset, its first value:
    ``y = x − x_first`` (exact wherever the window lies within a factor of two of that value),
    then ``y``'s mean and the deviations ``y − mean_y``.

    Centered about the rounded mean of the RAW values instead, a window whose spread sits near
    its level's ulp lost that spread to the mean's rounding: the mean of ``[2**52, 2**52 + 1]``
    rounds onto a member, so the deviations read ``[0, 1]`` for ``[−0.5, 0.5]``, the std 0.707
    for 0.5, and the z-score 1.414 — past the population bound ``√(w−1) = 1``. Around the offset
    they are the exact ``[−0.5, 0.5]``. Shifting commutes with the power-of-two lift
    (:func:`_lifted_windows`), and a difference that overflows (members farther apart than the
    double range) overflows the variance with it. Stable at any level, where the one-pass
    ``E[x²] − E[x]²`` form NaNs out around 1e8 already."""
    y = win - win[..., :1]
    dev = y - y.mean(axis=-1)[..., None]
    var: npt.NDArray[np.float64] = (dev**2).mean(axis=-1)
    last: npt.NDArray[np.float64] = dev[..., -1]
    return var, last


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
            np.isfinite(prod) & (prod >= MIN_NORMAL), np.sqrt(prod), scaled
        )
    return out


def _midpoint(lo: npt.NDArray[np.float64], hi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The midpoint of two doubles, elementwise — an even median's middle pair (a ``lo`` that IS
    ``hi`` reads the value itself, an odd median's middle).

    ``(lo + hi) / 2`` — bit for bit numpy's even median, the mean of the pair — wherever that is
    finite; where the SUM overflowed (two members past half the double range — a constant window
    of 1e308 read NaN, where its mean read 1e308), the halves' sum ``lo/2 + hi/2``. That is exact
    there: an overflowing sum has two operands of one sign, the smaller at least half the
    largest double's ulp (2**970), so halving each never rounds and their sum is the correctly
    rounded midpoint. The direct form stays wherever finite because halving a SUBNORMAL loses its
    last bit (``5e-324 / 2`` is 0.0)."""
    with np.errstate(over="ignore", invalid="ignore"):
        mid = (lo + hi) / 2.0
        out: npt.NDArray[np.float64] = np.where(np.isfinite(mid), mid, lo / 2.0 + hi / 2.0)
    return out


def _sorted_median(s: npt.NDArray[np.float64], k: int) -> npt.NDArray[np.float64]:
    """The median over the last axis of ``s``, sorted along it, whose leading ``k`` entries are
    the values (``k`` ≥ 1): the middle one, or the :func:`_midpoint` of the middle pair."""
    return _midpoint(s[..., (k - 1) // 2], s[..., k // 2])


def zscore_sma_apply_nb(arr: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    a = np.asarray(arr, dtype=float)
    n = a.shape[0]
    out = np.full(a.shape, np.nan)
    if window < 2 or window > n:
        return out
    win = sliding_window_view(a, window, axis=0)  # (n-window+1, cols, window)
    # A constant window has no dispersion to standardize by — undefined, whatever residual the
    # mean's rounding leaves (see the zero-dispersion note above).
    varies = ~_constant_windows(a, window)
    lifted: npt.NDArray[np.float64] | None = None  # built on the first batch that needs it
    for rows in _window_batches(win.shape):
        w = win[rows]
        finite_all = ~np.isnan(w).any(axis=-1)
        with np.errstate(all="ignore"):
            # Two-pass centered population moments about the window's own offset (the
            # ``rolling_agg`` std pattern): the current bar's deviation over the window std.
            var, dev = _window_moments(w)
            z = dev / np.sqrt(var)
            # A VARYING window whose variance fell below the normal range (deviations under
            # ~1.5e-154: a subnormal variance read |z| past its bound √(w−1), one of 0.0 read
            # NaN) is recomputed lifted — the same z, every rounding scaled (the out-of-range
            # note above).
            lost = finite_all & varies[rows] & (var < MIN_NORMAL)
            if lost.any():
                if lifted is None:
                    lifted = _lifted_windows(a, window)
                lifted_var, lifted_dev = _window_moments(lifted[rows])
                z = np.where(lost, lifted_dev / np.sqrt(lifted_var), z)
                var = np.where(lost, lifted_var, var)
        # `_normal(var)`: squared deviations past ~1.3e154 overflow the variance to inf, and the
        # z-score `(x − mean) / inf` is a FINITE 0.0 that `isfinite(z)` cannot catch.
        ok = finite_all & varies[rows] & _normal(var) & np.isfinite(z)
        out[window - 1 :][rows] = np.where(ok, z, np.nan)
    return out


def percentile_apply_nb(arr: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    # Fraction of the trailing window strictly below the current value: count(v < x) / window.
    a = np.asarray(arr, dtype=float)
    n = a.shape[0]
    out = np.full(a.shape, np.nan)
    if window < 1 or window > n:
        return out
    win = sliding_window_view(a, window, axis=0)
    x = a[window - 1 :]
    for rows in _window_batches(win.shape):
        w = win[rows]
        finite_all = ~np.isnan(w).any(axis=-1)
        count = (w < x[rows][..., None]).sum(axis=-1)  # NaN < x is False, so NaNs never count
        out[window - 1 :][rows] = np.where(finite_all, count / float(window), np.nan)
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
    # What `mean` and `std` read off the whole input, once per call: the windows of one value,
    # and `mean`'s exact sums — the window's exactly rounded sum over its length (the
    # exact-means note above).
    if agg in ("mean", "std"):
        constant = _constant_windows(a, window)
    if agg == "mean":
        sums = _rolling_exact_sums(np.ascontiguousarray(a), window)[window - 1 :]
    lifted: npt.NDArray[np.float64] | None = None  # built on the first batch that needs it
    for rows in _window_batches(win.shape):
        w = win[rows]
        finite_all = ~np.isnan(w).any(axis=-1)
        with np.errstate(all="ignore"):
            if agg == "max":
                v = w.max(axis=-1)
            elif agg == "min":
                v = w.min(axis=-1)
            elif agg == "mean":
                # a constant window's mean IS its value (the zero-dispersion note)
                v = np.where(constant[rows], w[..., -1], sums[rows] / window)
            elif agg == "std":
                var, _ = _window_moments(w)
                v = np.sqrt(var)
                # 0.0 only for a CONSTANT window. A VARYING one whose variance fell below the
                # normal range (squared deviations under ~1.5e-154 — to 0.0 itself under
                # ~1e-162) is recomputed lifted: its std exactly, rounded once to the window's
                # scale and floored at the smallest subnormal — never the 0.0 an all-equal
                # window owns.
                lost = finite_all & ~constant[rows] & (var < MIN_NORMAL)
                if lost.any():
                    if lifted is None:
                        lifted = _lifted_windows(a, window)
                    lifted_var, _ = _window_moments(lifted[rows])
                    lifted_std = np.ldexp(np.sqrt(lifted_var), -_LIFT)
                    v = np.where(lost, np.maximum(lifted_std, 5e-324), v)
                v = np.where(constant[rows], 0.0, v)
            elif agg in ("median", "mad"):
                # The sorted window's middle, an even window's pair met at its overflow-safe
                # midpoint (bit for bit ``np.median`` wherever that is finite): a constant window
                # reads its value and a MAD of exactly 0.0, at any level.
                v = _sorted_median(np.sort(w, axis=-1), window)
                if agg == "mad":
                    # Median absolute deviation about the SAME window's median, unscaled.
                    v = _sorted_median(np.sort(np.abs(w - v[..., None]), axis=-1), window)
            else:
                raise ValueError(f"unknown rolling_agg agg: {agg!r}")
        # `isfinite(v)`: a finite window can still overflow its sum/squared deviations to ±inf.
        out[window - 1 :][rows] = np.where(finite_all & np.isfinite(v), v, np.nan)
    return out


def expanding_agg_apply_nb(arr: npt.NDArray[np.float64], agg: str) -> npt.NDArray[np.float64]:
    """Expanding (all-time) max/min along the time axis; NaN-skipping via ``np.fmax``/``fmin``.

    Leading NaNs stay NaN until the first finite bar; thereafter NaN inputs hold the running peak
    (``fmax(peak, nan) == peak``) — past the input's init the builder reads such a NaN as a hole
    the series has no value after (``vectorize._transform``). Only ``max``/``min`` — expanding
    mean/std are rejected upstream.
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

    ``window <= 0`` means expanding: NaN bars are skipped, and past the input's init the builder
    reads one as a hole the series has no value after (``vectorize._transform``). Ties reset to
    the MOST RECENT attaining bar. Trailing form returns NaN until the window is full AND every
    bar in it is finite.
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
    (centered) over the LAST axis, each leg about its own window offset — the correlation's direct
    form (:func:`_window_moments` says why the offset: three points ``[2**52, 2**52 + 1, 2**52 +
    1]`` against ``[0, 1, 1]`` correlate exactly 1.0, and read 0.577 about the raw mean)."""
    ya = wa - wa[..., :1]
    yb = wb - wb[..., :1]
    da = ya - ya.mean(axis=-1)[..., None]
    db = yb - yb.mean(axis=-1)[..., None]
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

    The correlation is clipped to ``[−1, 1]``, numpy's ``corrcoef`` and scipy's ``pearsonr``
    rule: the rounding of an exactly linear window carries the computed ratio an ulp or two past
    its bound (``y = 3x + 7`` over ``[−83, 70, −13]`` read 1.0000000000000002, and
    ``rolling_corr(x, y) > 1`` fired where it is unreachable). The clip binds only where the
    ratio left the range.
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
    varies = ~_constant_windows(a, window) & ~_constant_windows(b, window)
    # both legs' lifted windows, built on the first batch that needs them
    lifted: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None
    for rows in _window_batches(wa.shape):
        xa, xb = wa[rows], wb[rows]
        finite_all = ~(np.isnan(xa) | np.isnan(xb)).any(axis=-1)
        with np.errstate(all="ignore"):
            va, vb, cov = _pair_moments(xa, xb)
            # Each leg is lifted only on the windows where IT lost its variance: lifting a leg
            # whose variance is normal could overflow its squares (an ordinary leg beside a
            # decayed one).
            lost_a = finite_all & varies[rows] & (va < MIN_NORMAL)
            lost_b = finite_all & varies[rows] & (vb < MIN_NORMAL)
            if lost_a.any() or lost_b.any():
                if lifted is None:
                    lifted = _lifted_windows(a, window), _lifted_windows(b, window)
                la, lb = lifted[0][rows], lifted[1][rows]
                for lost, ya, yb in (
                    (lost_a & ~lost_b, la, xb),
                    (~lost_a & lost_b, xa, lb),
                    (lost_a & lost_b, la, lb),
                ):
                    if lost.any():
                        lva, lvb, lcov = _pair_moments(ya, yb)
                        va = np.where(lost, lva, va)
                        vb = np.where(lost, lvb, vb)
                        cov = np.where(lost, lcov, cov)
            corr = np.clip(cov / _sqrt_product(va, vb), -1.0, 1.0)
        # `_normal(va)` / `_normal(vb)`: an overflowed variance divides `cov` by inf — a FINITE
        # ±0.0 that `isfinite(corr)` cannot catch (it still catches a covariance at the ceiling).
        ok = finite_all & varies[rows] & _normal(va) & _normal(vb) & np.isfinite(corr)
        out[window - 1 :][rows] = np.where(ok, corr, np.nan)
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
    (0 is a valid level base). ``log`` is the scale-safe, signed
    :func:`seikan.numeric.log_ratio_array` — the runner's outcome rule too — so a quotient past
    the double range (1e200 → 1e-200) still reads its finite log (−921.03…) instead of an
    undecidable NaN, and a move's reverse reads its exact negation (a two-bar mean of the log
    changes of 100 → 110 → 100 is exactly 0); a ``pct`` or ``diff`` past the range has no
    representable value and stays NaN.
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


def _row_offset(a: npt.NDArray[np.float64], finite: npt.NDArray[np.bool_]) -> np.ndarray:
    """(rows, 1): the offset a cross-section's centre and variance are computed around — the
    midpoint of its finite members' range, each end halved before the sum so that no member's
    difference from it overflows (0.0 on an empty row): members farther apart than the double
    range still have a representable mean and spread, which an offset AT a member would
    overflow. Differences from it are exact wherever the row lies within a factor of two of it
    (:func:`_window_moments` says why that matters)."""
    hi = np.where(finite, a, -np.inf).max(axis=1, keepdims=True)
    lo = np.where(finite, a, np.inf).min(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        mid = hi / 2.0 + lo / 2.0
    offset: np.ndarray = np.where(finite.any(axis=1, keepdims=True), mid, 0.0)
    return offset


def cross_centre(
    a: npt.NDArray[np.float64], finite: npt.NDArray[np.bool_]
) -> tuple[np.ndarray, np.ndarray]:
    """``(offset, mean)``, each (rows, 1): a cross-section's offset (:func:`_row_offset`) and the
    mean of its finite members' differences from it — their exactly rounded sum
    (:func:`_row_exact_sums`) over their count, so it has the exact sum's sign and is 0.0
    exactly where the differences cancel, in any member order — the centre ``cross_agg``'s std
    reduces about. The differences themselves round (members far from the offset), so
    ``offset + mean`` is no exact mean: a mean that is READ — the ``cross_mean`` bench leg — is
    :func:`cross_means`' instead. A member's deviation about it, ``(x − offset) − mean``, rounds
    twice, which the std's squares absorb; a deviation that is READ — ``cross_demean``, the
    ``cross_mean`` excess — is :func:`cross_deviations`' exact one instead. NaN mean on an empty
    row and where an intermediate sum leaves the double range (callers silence the 0/0 and test
    finiteness)."""
    k = finite.sum(axis=1, keepdims=True).astype(float)
    offset = _row_offset(a, finite)
    sums = _row_exact_sums(np.ascontiguousarray(a - offset), np.ascontiguousarray(finite))
    mean: np.ndarray = sums[:, None] / k
    return offset, mean


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

    Column g gets x_g - mean(finite values at bar t), self included: each member's EXACT
    deviation, rounded once over the count (:func:`cross_deviations`), so a member at the
    cross-section's mean demeans to exactly 0.0 and every other one has its deviation's sign.
    About a rounded centre it did not: ``[0, 1, 1, −1 − 2**-52, −1 + 2**-52]`` sums to exactly
    0, but its offset subtraction rounded two members and the first demeaned to 2.2e-17, so
    ``x == 0 AND cross_demean(x) > 0`` fired; about the raw rounded mean ``[2**52, 2**52 + 1,
    2**52 + 1]`` read ``[0, 1, 1]`` for ``[−2/3, 1/3, 1/3]``. A cross-section whose members all
    share one value demeans to exactly 0.0. NaN where x_g is NaN, where its deviation leaves the
    double range, or fewer than max(min_valid, 2) targets are finite. Inherently 2-D — no 1-D
    form.
    """
    a = np.asarray(arr, dtype=float)
    out = np.full(a.shape, np.nan)
    if a.ndim != 2 or a.shape[1] < 2:
        return out
    finite = np.isfinite(a)
    k = finite.sum(axis=1, keepdims=True)
    dev = cross_deviations(np.ascontiguousarray(a), np.ascontiguousarray(finite))
    ok = finite & (k >= max(min_valid, 2)) & np.isfinite(dev)
    return np.where(ok, dev, np.nan)


def _row_var(
    a: npt.NDArray[np.float64], finite: npt.NDArray[np.bool_], k: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """``(rows, 1)``: each row's population variance over its ``k`` finite members, two-pass
    (centered) about the row's offset (:func:`cross_centre`) — ``cross_agg`` std's direct form."""
    offset, mean = cross_centre(a, finite)
    dev = np.where(finite, (a - offset) - mean, 0.0)
    var: npt.NDArray[np.float64] = (dev * dev).sum(axis=1, keepdims=True) / k
    return var


def cross_agg_apply_nb(
    arr: npt.NDArray[np.float64], agg: str, min_valid: int
) -> npt.NDArray[np.float64]:
    """Cross-sectional AGGREGATE across targets at each bar, broadcast to every column (axis=1, no
    time window). Unlike ``cross_rank``/``cross_demean`` the value is a property of the
    cross-section itself, so every column carries it wherever at least ``max(min_valid, 2)`` targets
    are finite —
    a column whose own value is NaN still sees the group's aggregate. ``mean`` is the finite
    members' exactly rounded sum over their count (the exact-means note above: 0.0 exactly where
    they cancel, whatever order the targets were declared in); ``std`` is the population std
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
            sums = _row_exact_sums(np.ascontiguousarray(a), np.ascontiguousarray(finite))
            v = sums[:, None] / k
            v = np.where(constant, np.where(finite, a, -np.inf).max(axis=1, keepdims=True), v)
        elif agg == "median":
            # NaN sorts last, so each row's k finite members lead its sorted copy; the middle
            # pair meets at the overflow-safe midpoint (``np.nanmedian`` bit for bit wherever
            # finite — it summed even an odd row's middle with itself, overflowing at 1e308).
            s = np.sort(np.where(finite, a, np.nan), axis=1)
            v = np.full((a.shape[0], 1), np.nan)
            rows = np.flatnonzero(k.reshape(-1) > 0)
            kr = k.reshape(-1)[rows].astype(np.int64)
            v[rows, 0] = _midpoint(s[rows, (kr - 1) // 2], s[rows, kr // 2])
        elif agg == "std":
            # Two-pass centered variance — the one-pass Σx²/k − mean² form cancels to a silent
            # 0.0 std at large member levels, which the documented z-score recipe then divides by.
            var = _row_var(a, finite, k)
            v = np.sqrt(var)
            # 0.0 only for a CONSTANT row; a VARYING one whose variance fell below the normal
            # range is recomputed lifted and floored (the ``rolling_agg`` std rule).
            lost = ~constant & (var < MIN_NORMAL)
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


def row_group_ranks(labels: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    """``(rows × members)``: each finite label's rank among its OWN ROW's distinct finite labels
    (0 for the row's smallest), −1 for a NaN (non-finite) label — the members of row ``r`` that
    share rank ``j`` are exactly one group of that bar. Labels compare like ``==`` (``-0.0`` and
    ``0.0`` are one group). One stable sort per row, so O(rows × members · log members) however
    many distinct labels the whole history holds."""
    lab = np.asarray(labels, dtype=float)
    key = np.where(np.isfinite(lab), lab, np.nan)  # NaN sorts last
    order = np.argsort(key, axis=1, kind="stable")
    srt = np.take_along_axis(key, order, axis=1)
    step = np.ones(srt.shape, dtype=np.int64)
    step[:, 1:] = srt[:, 1:] != srt[:, :-1]
    ranked = np.where(np.isnan(srt), -1, np.cumsum(step, axis=1) - 1)
    ranks = np.empty(lab.shape, dtype=np.int64)
    np.put_along_axis(ranks, order, ranked, axis=1)
    return ranks


def cross_grouped_apply_nb(
    kernel: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    arr: npt.NDArray[np.float64],
    labels: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Apply a cross-sectional kernel WITHIN each distinct finite label at each bar.

    Grouping wraps the existing kernel rather than re-deriving it: pass ``j`` hands the kernel the
    (rows × targets) input with every member masked to NaN but those holding their row's ``j``-th
    distinct label (:func:`row_group_ranks`), and keeps only those members' outputs. Every
    property the ungrouped kernel has — the per-group ``min_valid`` floor,
    ``rankdata(method="average")`` ties, the overflow sanitization, ``cross_agg``'s broadcast (now
    to the group's members only) — is inherited unchanged, and ONE label reproduces the
    ungrouped result bit-exactly (a pandas ``groupby.transform`` was measured NOT bit-exact —
    summation order — and rejected). Labels are point-in-time: a member's label may change from
    bar to bar. A NaN label excludes the member.

    Every kernel is row-wise, so a pass per ROW-rank is bit-exact with a pass per distinct label
    (each row's masked member set is the same) and costs at most ``targets`` passes, never one
    per label the history holds: under a fresh label each bar (a date code, a running count) that
    was a pass per bar — 1.3 s for 4,000 bars of four members, where one label took under a
    millisecond."""
    a = np.asarray(arr, dtype=float)
    lab = np.asarray(labels, dtype=float)
    if a.shape != lab.shape:
        raise ValueError(f"cross_grouped shape mismatch: {a.shape} vs {lab.shape}")
    out = np.full(a.shape, np.nan)
    ranks = row_group_ranks(lab)
    for j in range(int(ranks.max(initial=-1)) + 1):
        in_group = ranks == j
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
    """EW z-score via the West variance recurrence: ``d = x − μ; μ += α·d; s = (1−α)(s + α·d²)``
    and ``z = (x − μ) / √s``.

    Algebraically identical (in exact arithmetic) to the EW[x²] − EW[x]² pair it replaces, but
    computed on centered quantities, so it is translation-invariant in floating point — the
    one-pass form lost its mantissa to cancellation at large input levels and drifted with the
    input offset. Same seeding (first finite value → μ = x, s = 0), same NaN-skipping warmup
    (``warmup`` observations, from ``ema_params``), same ``s > 0`` emission gate.

    The mean is COMPENSATED: μ is held as the unevaluated sum ``mu + lo`` of two doubles, each
    update ``μ + α·d`` taken with TwoSum (Knuth), so ``lo`` keeps exactly the part of it ``mu``
    cannot represent, and every deviation is read ``(x − mu) − lo``. The mean is then known to the
    rounding of its STEPS, not of its level. A plain update rounds at the level: over ``[1, 1,
    1 + 2**-52]`` at α 0.5 it rounded the mean back onto 1, so the deviation read the whole step
    and the z-score 2 — past its bound ``√((1−α)/α) = 1``, which the exact z sits on. On a HELD
    input (the same value again) exact arithmetic moves the deviation to ``(1−α)·(x − μ)`` every
    bar — it decays and keeps its sign — and so does the compensated one: ``mu`` settles on the
    held value and the deviation decays in ``lo``, so ``z > 0`` reads the same on every bar of a
    hold, however long (a plain mean stuck a few ulps off the value, its deviation froze at that
    residue and the z locked at ``±1/√(1−α)``). A step of a few ulps whose update rounds ``mu``
    ONTO the new value keeps its gap in ``lo``, and an input that moves exactly onto the mean reads
    a decided 0.0 for as long as it holds — exact arithmetic's z, to the mean's rounding. The z is
    clipped to ``±√((1−α)/α)``: the exact z never exceeds that bound, and a step off a flat
    stretch sits ON it, where the ratio's last rounding could carry it past.

    Out of range: the squares of deviations under ~1.5e-154 lose their bits, and an input that
    DECAYS geometrically — an ``ema`` of a change over a held stretch, from ordinary levels — gets
    there with a z exact arithmetic keeps constant. Read as computed, ``s`` rounded to subnormals
    or to 0.0: the z read past its bound, NaN, or (floored) collapsed toward 0 — deciding
    thresholds the exact z does not. The recurrence therefore runs in a frame lifted by
    ``2**_LIFT`` once its variance nears the range (``_EW_LIFT_VAR``), lowered once the input
    returns. Scaling by a power of two never rounds, so every step in the frame is the plain step
    times its power of two: the z is the recurrence's value in an unbounded exponent range, and a
    series whose recurrence stays in the normal range keeps every bit. Out of the frame's reach is
    only a HELD input's decay (the deviation decaying in ``lo`` at ordinary levels, or a mean
    decaying onto a held 0.0), whose z exact arithmetic sends to 0 like ``(1−α)**(t/2)``: under
    ``2**-158/α`` (~5e-47 at RiskMetrics' 0.06; ~1e-140 on a hold at ordinary levels) once ``s``
    leaves the normal range in the frame. Below ``α = 1`` exact arithmetic keeps ``s`` positive
    once the input has moved, and a moved step's deviation ``(1−α)·d`` nonzero: an ``s`` that
    rounds to 0.0 while it, or the step's deviation, is nonzero, and a deviation that rounds to 0.0
    (``mu`` on the input, nothing left in ``lo``) after a nonzero step, are floored at the smallest
    subnormal — the deviation with that step's sign — so the z stays decided, imprecise in
    magnitude. An ``s`` that OVERFLOWED is carried forward: undefined from there on. At ``α = 1``
    (the DSL refuses it) the mean is the last value and ``s`` IS 0: undefined throughout.
    """
    n = arr.shape[0]
    out = np.full(n, np.nan)
    if warmup < 1 or not (0.0 < alpha <= 1.0):
        return out
    bound = math.sqrt((1.0 - alpha) / alpha)
    seeded = False
    mu = 0.0
    lo = 0.0  # the part of the mean ``mu`` cannot represent: μ = mu + lo, exactly
    s = 0.0
    seen = 0
    lift = 0  # the frame's exponent: mu, lo, s (twice over) and x are held times 2**lift
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
                lo = math.ldexp(lo, -_LIFT)
                s = math.ldexp(s, -2 * _LIFT)
            x = math.ldexp(raw, lift)
        if not seeded:
            seeded = True
            mu = x
            dev = 0.0
        else:
            d = (x - mu) - lo
            s_next = (1.0 - alpha) * (s + alpha * d * d)
            # exact arithmetic keeps s > 0 below alpha 1 once the input has moved: floored where
            # it rounds to 0.0 (round-to-nearest already holds it at a subnormal below alpha 0.5)
            if s_next == 0.0 and alpha < 1.0 and (s > 0.0 or d != 0.0):
                s_next = 5e-324
            s = s_next
            # TwoSum: mu + lo after the step is mu + (lo + alpha·d) before it, exactly
            t = lo + alpha * d
            hi = mu + t
            b = hi - mu
            lo = (mu - (hi - b)) + (t - b)
            mu = hi
            # the step's exact deviation (1 − alpha)·d is nonzero: one that rounded to 0.0 is
            # floored at the smallest subnormal, the step's sign kept
            if mu == x and lo == 0.0 and d != 0.0 and alpha < 1.0:
                lo = -5e-324 if d > 0.0 else 5e-324
            dev = (x - mu) - lo
        seen += 1
        # the variance nears the subnormal range with every value it is built from small: lift
        if (
            lift < _EW_MAX_LIFT
            and s < _EW_LIFT_VAR
            and max(abs(x), abs(mu), abs(dev)) < _EW_LIFT_AT
        ):
            lift += _LIFT
            mu = math.ldexp(mu, _LIFT)
            lo = math.ldexp(lo, _LIFT)
            s = math.ldexp(s, 2 * _LIFT)
            dev = math.ldexp(dev, _LIFT)
        if seen < warmup:
            continue
        if s <= 0.0 or not np.isfinite(s):
            continue
        z = dev / np.sqrt(s)
        if np.isfinite(z):
            out[i] = min(max(z, -bound), bound)
    return out


@njit(cache=True)
def ema_1d(arr: npt.NDArray[np.float64], alpha: float, warmup: int) -> npt.NDArray[np.float64]:
    """EMA with decay ``alpha`` (``ema_params``): seeded at the first finite value, NaN-skipping
    (past the input's init the builder reads a NaN as a hole the series has no value after —
    ``vectorize._transform``), NaN until ``warmup`` observations — the same warmup convention as
    the EMA inside ``zscore_ema``.

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
def _expansion_add(
    p: npt.NDArray[np.float64], m: int, x: float
) -> tuple[npt.NDArray[np.float64], int, bool]:
    """Add ``x`` into the non-overlapping partials ``p[:m]`` of an exact running sum — Shewchuk's
    grow-expansion, the ``math.fsum`` algorithm: the sum of the partials is the sum of every
    value added, with no bit lost. Returns the partials (grown when full), their count, and
    whether an intermediate sum left the double range (the running sum then has no value)."""
    i = 0
    for k in range(m):
        y = p[k]
        if abs(x) < abs(y):
            x, y = y, x
        hi = x + y
        if not np.isfinite(hi):
            return p, m, True
        lo = y - (hi - x)
        if lo != 0.0:
            p[i] = lo
            i += 1
        x = hi
    if x != 0.0:
        if i == p.shape[0]:
            grown = np.empty(2 * p.shape[0])
            grown[:i] = p[:i]
            p = grown
        p[i] = x
        i += 1
    return p, i, False


@njit(cache=True)
def _expansion_value(p: npt.NDArray[np.float64], m: int) -> float:
    """The correctly rounded value of the partials ``p[:m]`` (:func:`_expansion_add`) —
    ``math.fsum``'s final step, its half-even correction across partials included."""
    if m == 0:
        return 0.0
    n = m - 1
    hi = float(p[n])
    lo = 0.0
    while n > 0:
        x = hi
        n -= 1
        y = p[n]
        hi = x + y
        lo = y - (hi - x)
        if lo != 0.0:
            break
    if n > 0 and ((lo < 0.0 and p[n - 1] < 0.0) or (lo > 0.0 and p[n - 1] > 0.0)):
        y = lo * 2.0
        x = hi + y
        if y == x - hi:
            hi = x
    return hi


@njit(cache=True)
def _rolling_exact_sum_1d(a: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    """The correctly rounded sum of each trailing ``window`` of ``a``, at the window's last bar;
    NaN unless the window holds ``window`` finite values.

    One SLIDING expansion (:func:`_expansion_add`): each bar takes back the value leaving the
    window and adds its own, so the partials hold the window's exact sum — O(rows) whatever the
    window, where a fresh ``math.fsum`` per window was O(rows × window). An intermediate sum that
    leaves the double range (members near it) leaves the partials without a value: they are
    rebuilt from the window on its next full bar, and the window is NaN while that rebuild
    overflows too (``math.fsum`` raises there)."""
    n = a.shape[0]
    out = np.full(n, np.nan)
    p = np.empty(8)  # the window's partials, exact while not stale
    m = 0
    n_finite = 0
    stale = False
    for i in range(n):
        if i >= window:
            old = a[i - window]
            if np.isfinite(old):
                n_finite -= 1
                if not stale:
                    p, m, stale = _expansion_add(p, m, -old)
        x = a[i]
        if np.isfinite(x):
            n_finite += 1
            if not stale:
                p, m, stale = _expansion_add(p, m, x)
        if n_finite < window:
            continue
        if stale:
            m = 0
            stale = False
            for j in range(i - window + 1, i + 1):
                p, m, stale = _expansion_add(p, m, a[j])
                if stale:
                    break
        if not stale:
            out[i] = _expansion_value(p, m)
    return out


@njit(cache=True)
def _rolling_exact_sums(a: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    """2-D :func:`_rolling_exact_sum_1d` per column."""
    out = np.empty_like(a)
    for j in range(a.shape[1]):
        out[:, j] = _rolling_exact_sum_1d(a[:, j], window)
    return out


@njit(cache=True)
def _row_exact_sums(
    a: npt.NDArray[np.float64], finite: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """``(rows,)``: the correctly rounded sum of each row's ``finite`` members — one expansion
    (:func:`_expansion_add`) per row, 0.0 on an empty row; NaN where an intermediate sum leaves
    the double range, or a member marked finite is not (a value lifted past it)."""
    rows, cols = a.shape
    out = np.full(rows, np.nan)
    p = np.empty(8)
    for r in range(rows):
        m = 0
        over = False
        for c in range(cols):
            if finite[r, c]:
                v = a[r, c]
                over = not np.isfinite(v)
                if not over:
                    p, m, over = _expansion_add(p, m, v)
                if over:
                    break
        if not over:
            out[r] = _expansion_value(p, m)
    return out


#: Veltkamp's splitter for a double's 53 bits, ``2**27 + 1``: ``x = hi + lo`` with each half at
#: most 26 significant bits, so its product with a member count below ``2**26`` is exact.
_SPLITTER = 134217729.0
#: From this magnitude a member's split, or its product with the count, can overflow — such a row
#: is computed in a frame scaled by :data:`_SPLIT_DOWN` (:func:`cross_deviations`).
_SPLIT_MAX = 2.0**996
_SPLIT_DOWN = 2.0**-28


@njit(cache=True)
def cross_deviations(
    a: npt.NDArray[np.float64], finite: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """``(rows × members)``: each ``finite`` member's deviation from its row's mean, ``x − S/k``
    for the exact sum ``S`` of the row's ``k`` finite members — the numerator ``k·x − S`` summed
    EXACTLY (:func:`_expansion_add`, with ``k·x`` the products of ``x``'s two Veltkamp halves,
    each an exact double), rounded once and divided by ``k`` once: ``cross_demean``'s values and
    the runner's ``cross_mean`` excesses. A member AT the mean deviates by exactly 0.0, and every
    deviation has the exact one's sign, whatever the members' order or magnitudes. A row whose
    largest member reaches ``2**996`` (where the split or ``k·x`` can overflow) is computed in a
    frame scaled by ``2**-28`` — exact save its members and deviations below ``2**-994``, whose
    scaled bits round — and scaled back: a deviation past the double range is ``±inf`` there, for
    the caller to test. NaN off the finite members, and on a row whose exact numerator leaves the
    double range (none can at fewer than ``2**26`` members)."""
    rows, cols = a.shape
    out = np.full((rows, cols), np.nan)
    p = np.empty(8)
    q = np.empty(8)
    for r in range(rows):
        k = 0
        big = 0.0
        for c in range(cols):
            if finite[r, c]:
                k += 1
                big = max(big, abs(a[r, c]))
        scale = _SPLIT_DOWN if big >= _SPLIT_MAX else 1.0
        m = 0
        over = False
        for c in range(cols):
            if finite[r, c]:
                p, m, over = _expansion_add(p, m, a[r, c] * scale)
                if over:
                    break
        if over:
            continue
        if q.shape[0] < p.shape[0]:
            q = np.empty(p.shape[0])
        for c in range(cols):
            if not finite[r, c]:
                continue
            for i in range(m):
                q[i] = -p[i]
            x = a[r, c] * scale
            t = x * _SPLITTER
            hi = t - (t - x)
            q, n, over = _expansion_add(q, m, k * hi)
            if not over:
                q, n, over = _expansion_add(q, n, k * (x - hi))
            if not over:
                out[r, c] = _expansion_value(q, n) / k / scale
    return out


@njit(cache=True)
def cross_means(
    a: npt.NDArray[np.float64], finite: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """``(rows,)``: each row's mean over its ``finite`` members — the runner's ``cross_mean``
    bench leg, by ``cross_agg``'s ``mean`` rule: the exactly rounded sum over the count (0.0
    exactly where the members cancel, the exact sum's sign otherwise, in any member order), and
    a row whose members all share one value reads that value. A row whose largest member reaches
    ``2**996`` is summed in :func:`cross_deviations`' frame scaled by ``2**-28`` and scaled back,
    so a row of finite members never loses its mean to a sum past the double range. NaN on an
    empty row."""
    rows, cols = a.shape
    out = np.full(rows, np.nan)
    p = np.empty(8)
    for r in range(rows):
        k = 0
        lo = np.inf
        hi = -np.inf
        for c in range(cols):
            if finite[r, c]:
                k += 1
                lo = min(lo, a[r, c])
                hi = max(hi, a[r, c])
        if k == 0:
            continue
        if lo == hi:
            out[r] = hi
            continue
        scale = _SPLIT_DOWN if max(hi, -lo) >= _SPLIT_MAX else 1.0
        m = 0
        over = False
        for c in range(cols):
            if finite[r, c]:
                p, m, over = _expansion_add(p, m, a[r, c] * scale)
                if over:
                    break
        if not over:
            out[r] = _expansion_value(p, m) / k / scale
    return out


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

    The running sum is EXACT: it is kept as non-overlapping partials (:func:`_expansion_add`)
    and read correctly rounded on every bar — ``math.fsum`` of ``x[s .. t]``. A plain ``acc +=
    v`` lost a term a larger one swallowed and never got it back when that term cancelled: after
    ``[2**54, 1, −2**54]`` the exact sum is 1 and it read 0 from then on, so
    ``event_agg(E, X, sum) == 0`` fired where it was false.

    STRICT finite gate, the ``rolling_agg`` rule: one NaN input bar inside ``[s, t]`` POISONS the
    aggregate until the next event. A warmup bar carries no state; a hole bar drops the anchor
    (nothing is emitted until the next defined event, and the hole bar itself never contributes).
    Only finite results are emitted: a sum whose running partials leave the double range reads
    NaN until the next event (as does a mean that is not of an all-equal run)."""
    n = sig.shape[0]
    out = np.full(n, np.nan)
    anchored = False
    acc = np.empty(8)  # the running sum's partials, exact
    n_acc = 0
    overflowed = False
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
            n_acc = 0
            overflowed = False
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
        if (mode == 0 or mode == 3) and not overflowed:
            acc, n_acc, overflowed = _expansion_add(acc, n_acc, v)
        if v > mx:
            mx = v
        if v < mn:
            mn = v
        if mode == 0:
            r = np.nan if overflowed else _expansion_value(acc, n_acc)
        elif mode == 1:
            r = mx
        elif mode == 2:
            r = mn
        elif mx == mn:
            # an all-equal stretch's mean IS its value (the zero-dispersion note above)
            r = mx
        else:
            r = np.nan if overflowed else _expansion_value(acc, n_acc) / cnt
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
