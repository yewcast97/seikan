"""Scale-safe numeric rules shared across the layers.

A leaf module by design: it imports ``math`` and ``numpy`` and nothing from the package, so
``describe.py`` (which may import nothing from ``compiler/``, ``analysis/`` or ``gate/``) and the
engine's kernels, outcome algebra and pool statistics can share ONE rule without a dependency
between them. No numba either: no ``@njit`` kernel forms a log ratio (``nb.change_apply_nb`` is a
vectorized numpy kernel), and ``describe`` must stay numba-free.

**The normal range.** :data:`MIN_NORMAL` is where it begins, defined once: the kernels' variance
tests (``compiler.nb._normal`` and the lifted windows), the pool statistics'
(``analysis._pools._normal``, the CSCV split moments) and ``describe``'s std all read this one
constant, so no layer can drift from another on which variance still carries its bits.

**The log ratio.** ``ln(num/den)`` of two strictly positive finite levels is always a
representable double (its magnitude never exceeds 1455), and it is read SIGNED: ``ln(hi/lo)`` of
the larger level over the smaller, negated where ``num < den``, so ``ln(a/b)`` is ``−ln(b/a)``
bit for bit. Read in its own direction each quotient rounded on its own: ``ln(fl(110/100))`` is
0.09531017980432493 and ``ln(fl(100/110))`` −0.09531017980432490, so a price cycle 100 → 110 →
100, a move that went nowhere, summed to 2e-17 — a positive mean log return the support floor
graded, and a rolling-mean sign predicate firing. A longer cycle (100 → 110 → 120 → 100) cancels
only to rounding, as any sum of rounded terms does. The quotient of the larger level by the
smaller is at least 1, so it never underflows; past the double range it overflows to ``inf``
(``1e200 / 1e-200``), where the direct form read ``ln(inf)`` — NaN after the finite-or-NaN
sanitation, a censored outcome, where the true value is the finite ``921.034…`` — and there the
difference of logs ``ln(hi) − ln(lo)``, which never leaves the range, stands in. Everywhere else
the direct form and its exact bits stand: the difference of logs rounds differently on almost
every ordinary pair. Each twin keeps its callers' log implementation (``math.log`` for the scalar
reads, ``np.log`` for the array reads). A test pins the twins to the same rule.

Positive finite levels are the CALLERS' gate (both ratio algebras are positive-domain, and each
caller already refuses or masks a non-positive endpoint); on them the fallback is always finite.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

#: The smallest positive NORMAL double, ``2**-1022`` (2.2250738585072014e-308): a value at or
#: above it keeps all 53 significant bits, one below it fewer — a quotient there is no longer the
#: direct form's exact answer under ``ln``, a variance there no longer an estimate of any
#: dispersion. The one definition every layer reads (see the module note).
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
    """``ln(num/den)`` by ``math.log``, signed — ``ln(hi/lo)`` of the larger level over the
    smaller, negated where ``num < den``, so ``log_ratio(a, b) == −log_ratio(b, a)`` bit for bit
    — and ``ln(hi) − ln(lo)`` where that quotient overflowed. ``num``/``den`` strictly positive
    and finite (the caller's gate); never raises or warns on them, numpy scalars included: both
    are read as Python floats (an exact conversion), whose division overflows to ``inf``
    silently where ``np.float64``'s warns."""
    num, den = float(num), float(den)
    hi, lo = max(num, den), min(num, den)
    q = hi / lo
    r = math.log(q) if q < math.inf else math.log(hi) - math.log(lo)
    return -r if num < den else r


def log_ratio_array(
    num: npt.NDArray[np.float64], den: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """The elementwise twin of :func:`log_ratio` by ``np.log`` — the same rule per element, under
    numpy broadcasting (0-d operands included: the result is always an array). Never warns: an
    element off the positive domain (a non-positive or NaN level) comes back as an unspecified
    value — NaN, ±inf, or a finite number where two negative levels make a positive quotient —
    for the caller's own positivity mask to discard."""
    n, d = np.broadcast_arrays(np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64))
    down = n < d
    hi, lo = np.where(down, d, n), np.where(down, n, d)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        q = hi / lo
        # `asarray`: a no-op on the array ``np.log`` returns for array operands (same object, same
        # bits), a writable 0-d array for the numpy scalar it returns for 0-d ones — so the
        # fallback below can always assign into it.
        r: npt.NDArray[np.float64] = np.asarray(np.log(q))
        past = ~(q < np.inf)
        if past.any():
            r[past] = np.log(hi[past]) - np.log(lo[past])
    out: npt.NDArray[np.float64] = np.where(down, -r, r)
    return out
