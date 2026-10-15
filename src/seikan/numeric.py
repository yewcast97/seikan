"""Scale-safe numeric rules shared across the layers.

A leaf module by design: it imports ``math`` and ``numpy`` and nothing from the package, so
``describe.py`` (which may import nothing from ``compiler/``, ``analysis/`` or ``gate/``) and the
engine's kernels, outcome algebra and pool statistics can share ONE rule without a dependency
between them. No numba either: no ``@njit`` kernel forms a log ratio (``nb.change_apply_nb`` is a
vectorized numpy kernel), and ``describe`` must stay numba-free.

**The normal range.** :data:`MIN_NORMAL` is where it begins, defined once: the log ratio's
quotient test below, the kernels' variance tests (``compiler.nb._normal`` and the lifted
windows), the pool statistics' (``analysis._pools._normal``, the CSCV split moments) and
``describe``'s std all read this one constant, so no layer can drift from another on which
variance still carries its bits.

**The log ratio.** ``ln(num/den)`` of two strictly positive finite levels is always a
representable double (its magnitude never exceeds 1455), but the QUOTIENT it is read through
need not be: it overflows to ``inf`` above the double range and underflows to ``0.0`` — or to a
subnormal whose relative precision is already gone — below the smallest normal double. ``1e200
→ 1e-200`` is a quotient of 1e-400, so the direct form read ``ln(0) = −inf`` (NaN after the
finite-or-NaN sanitation: a censored outcome, an undefined decision bar) or raised a ``math``
domain error, where the true value is the finite ``−921.034…``. The difference of logs
``ln(num) − ln(den)`` never leaves the range, but it rounds differently from the direct form on
almost every ordinary pair, so it is used ONLY where the quotient is not a finite positive NORMAL
double; everywhere else the direct form — and its exact bits — stand. Each twin keeps its
callers' log implementation (``math.log`` for the scalar reads, ``np.log`` for the array reads),
so every log whose quotient is a normal double is byte-identical. The one FINITE read that moves
is a subnormal quotient's: ``ln`` of the rounded subnormal was finite but wrong — by ~5e-4 at
3e20 → 1e-300 (−737.92536 where it is −737.92584), by up to ln 2 at the bottom of the subnormal
range — and it becomes the exact difference of logs. A test pins the twins to the same rule.

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


def log_ratio(num: float, den: float) -> float:
    """``ln(num/den)`` by ``math.log`` — the direct form wherever the quotient is a finite
    positive normal double, ``ln(num) − ln(den)`` where it overflowed, underflowed or went
    subnormal. ``num``/``den`` strictly positive and finite (the caller's gate); never raises
    or warns on them, numpy scalars included: both are read as Python floats (an exact
    conversion), whose division overflows to ``inf`` silently where ``np.float64``'s warns."""
    num, den = float(num), float(den)
    q = num / den
    if MIN_NORMAL <= q < math.inf:
        return math.log(q)
    return math.log(num) - math.log(den)


def log_ratio_array(
    num: npt.NDArray[np.float64], den: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """The elementwise twin of :func:`log_ratio` by ``np.log`` — the same rule per element, under
    numpy broadcasting (0-d operands included: the result is always an array). Never warns: an
    element off the positive domain (a non-positive or NaN level) comes back as an unspecified
    value — NaN, ±inf, or a finite number where two negative levels make a positive quotient —
    for the caller's own positivity mask to discard."""
    n, d = np.broadcast_arrays(np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64))
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        q = n / d
        # `asarray`: a no-op on the array ``np.log`` returns for array operands (same object, same
        # bits), a writable 0-d array for the numpy scalar it returns for 0-d ones — so the
        # fallback below can always assign into it.
        out: npt.NDArray[np.float64] = np.asarray(np.log(q))
        off = ~((q >= MIN_NORMAL) & (q < np.inf))
        if off.any():
            out[off] = np.log(n[off]) - np.log(d[off])
    return out
