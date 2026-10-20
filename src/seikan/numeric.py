"""Scale-safe numeric rules shared across the layers.

A leaf module by design: it imports ``math`` and ``numpy`` and nothing from the package, so
``describe.py`` (which may import nothing from ``compiler/``, ``analysis/`` or ``gate/``) and the
engine's kernels, outcome algebra and pool statistics can share ONE rule without a dependency
between them. No numba either: no ``@njit`` kernel forms a log ratio or a percent change
(``nb.change_apply_nb``, ``drawdown_apply_nb`` and ``runup_apply_nb`` are vectorized numpy
kernels), and ``describe`` must stay numba-free.

**The normal range.** :data:`MIN_NORMAL` is where it begins, defined once: the kernels' variance
tests (``compiler.nb._normal`` and the lifted windows), the pool statistics'
(``analysis._pools._normal``, the CSCV split moments) and ``describe``'s std all read this one
constant, so no layer can drift from another on which variance still carries its bits.

**The log ratio.** ``ln(num/den)`` of two strictly positive finite levels is always a
representable double (its magnitude never exceeds 1455), and it is read SIGNED: the log of the
larger level ``hi`` over the smaller ``lo``, negated where ``num < den``, so ``ln(a/b)`` is
``−ln(b/a)`` bit for bit. Read in its own direction each quotient rounded on its own:
``ln(fl(110/100))`` is 0.09531017980432493 and ``ln(fl(100/110))`` −0.09531017980432490, so a
price cycle 100 → 110 → 100, a move that went nowhere, summed to 2e-17 — a positive mean log
return the support floor graded, and a rolling-mean sign predicate firing. A longer cycle (100 →
110 → 120 → 100) cancels only to rounding, as any sum of rounded terms does. The log is
``log1p`` of the larger level's relative EXCESS over the smaller, ``(hi − lo) / lo`` — the
percent change's own read (below), the correctly rounded quotient wherever ``hi ≤ 2·lo`` — and
``log1p`` of a non-negative argument passes at most its relative rounding through (condition
number ≤ 1), so the log carries a RELATIVE rounding however small the move: measured within 1.4
ulps of exact across every binade. ``ln(fl(hi/lo))`` rounded the quotient at the ulp of 1 first,
an ABSOLUTE rounding of up to 2**-53 under every move: 100 → 100.00000001 read
1.000000082690371e-10 for 9.999993721476353e-11, above its own percent change
9.999993721976352e-11 where ``ln(1 + r) ≤ r`` for every ``r > −1`` (``change(close, log) >
change(close, pct)`` fired on 60 of 120 bars alternating the pair), and two adjacent doubles near
100 read 2.2e-16 for 1.4e-16. The excess is 0 or at least 2**-53, so it never underflows; past
the double range it overflows to ``inf`` exactly where the quotient does (``1e200`` over
``1e-200``: there ``hi − lo`` is ``hi``), and its log, NaN after the finite-or-NaN sanitation,
would censor an outcome whose true value is the finite ``921.034…`` — there the difference of
logs ``ln(hi) − ln(lo)``, which never leaves the range, stands in. Neither the quotient's log nor
the difference of logs may stand in anywhere else: each reads other bits on almost every ordinary
pair. Each twin keeps its callers' log implementations (``math.log1p`` and ``math.log`` for the
scalar reads, ``np.log1p`` and ``np.log`` for the array reads). A test pins the twins to the same
rule.

**The percent change.** ``num/den − 1`` rounds the quotient at the ulp of 1 before the exact
subtraction, so a change under +100% carried an ABSOLUTE rounding of up to 2**-53 however small
it was: 100 → 110 read 0.10000000000000009 and 100 → 90 −0.09999999999999998, so a rise and the
mirror fall off one base averaged 5.6e-17 — a positive mean the support floor graded — and a
rolling mean of such changes read ``> 0`` where exact arithmetic reads 0. :func:`pct_change` forms
``(num − den) / den``: the difference is exact wherever ``den/2 ≤ num ≤ 2·den`` (Sterbenz — every
move from −50% to +100%), so there the change is the correctly rounded quotient (``(110 − 100) /
100`` is the double 0.1 and ``(90 − 100) / 100`` its negation, so the pair cancels exactly);
beyond it, two roundings of RELATIVE size. Three moves off one base whose exact changes cancel
(+10%, +20%, −30%) cancel only to rounding, as any sum of rounded terms does: their doubles sum
exactly to ``2**-55``. On positive levels the difference cannot overflow, and the quotient leaves
the double range exactly as the ratio did (an underflow still reads −1.0).

Positive finite levels are the CALLERS' gate (both ratio algebras are positive-domain, and each
caller already refuses or masks a non-positive endpoint); on them the log ratio is always finite,
and the percent change wherever its quotient is.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

#: The smallest positive NORMAL double, ``2**-1022`` (2.2250738585072014e-308): a value at or
#: above it keeps all 53 significant bits, one below it fewer — a variance there is no longer an
#: estimate of any dispersion. The one definition every layer reads (see the module note).
MIN_NORMAL: float = float(np.finfo(np.float64).tiny)


def midrange(lo: float, hi: float) -> float:
    """The midpoint of ``[lo, hi]``, each end halved BEFORE the sum, so it is finite for any two
    finite ends and no value between them lies farther from it than the double range holds.

    The OFFSET a dispersion is computed about — the pools' ddof=1 std and the regression legs
    (``analysis._pools``) and ``describe``'s std; the kernels' cross-sections take the same rule
    row by row (``compiler.nb._row_offset``). A value's deviation from it is exact wherever the
    values lie within a factor of two of it, so a spread at the level's ulp keeps its bits, where
    one about the ROUNDED raw mean carries that mean's rounding: ``[2**52, 2**52 + 1]``'s mean
    rounds onto ``2**52``, and its ddof=1 std read 1.0 for 0.707."""
    return hi / 2.0 + lo / 2.0


def log_ratio(num: float, den: float) -> float:
    """``ln(num/den)`` by ``math.log1p``, signed — ``ln(1 + (hi − lo)/lo)`` of the larger level
    over the smaller, negated where ``num < den``, so ``log_ratio(a, b) == −log_ratio(b, a)`` bit
    for bit — and ``ln(hi) − ln(lo)`` by ``math.log`` where that excess overflowed (the module
    note says why). ``num``/``den`` strictly positive and finite (the caller's gate); never raises
    or warns on them, numpy scalars included: both are read as Python floats (an exact
    conversion), whose division overflows to ``inf`` silently where ``np.float64``'s warns."""
    num, den = float(num), float(den)
    hi, lo = max(num, den), min(num, den)
    r = (hi - lo) / lo
    v = math.log1p(r) if r < math.inf else math.log(hi) - math.log(lo)
    return -v if num < den else v


def log_ratio_array(
    num: npt.NDArray[np.float64], den: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """The elementwise twin of :func:`log_ratio` by ``np.log1p`` and ``np.log`` — the same rule
    per element, under numpy broadcasting (0-d operands included: the result is always an array).
    Never warns: an element off the positive domain (a non-positive or NaN level) comes back as an
    unspecified value — NaN, ±inf, or a finite number where two negative levels make a positive
    quotient — for the caller's own positivity mask to discard."""
    n, d = np.broadcast_arrays(np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64))
    down = n < d
    hi, lo = np.where(down, d, n), np.where(down, n, d)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        r = (hi - lo) / lo
        # `asarray`: a no-op on the array ``np.log1p`` returns for array operands (same object,
        # same bits), a writable 0-d array for the numpy scalar it returns for 0-d ones — so the
        # fallback below can always assign into it.
        v: npt.NDArray[np.float64] = np.asarray(np.log1p(r))
        past = ~(r < np.inf)
        if past.any():
            v[past] = np.log(hi[past]) - np.log(lo[past])
    out: npt.NDArray[np.float64] = np.where(down, -v, v)
    return out


def pct_change(num: npt.ArrayLike, den: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """The percent change ``num/den − 1`` as ``(num − den) / den`` (the module note says why),
    elementwise under numpy broadcasting — scalar operands included, whose result is a 0-d
    array. Never warns: an element off the positive domain, or a quotient past the double range,
    comes back as whatever IEEE arithmetic makes of it, for the caller's own positivity and
    finiteness masks to discard."""
    n = np.asarray(num, dtype=np.float64)
    d = np.asarray(den, dtype=np.float64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out: npt.NDArray[np.float64] = np.asarray((n - d) / d)
    return out
