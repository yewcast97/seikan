"""The event-time Newey-West HAC mean and the greedy non-overlap count."""

from __future__ import annotations

import math

import numpy as np

from seikan.analysis._pools import BarNumerators, _near_constant, _normal, exact_mean

_NAN = float("nan")


def newey_west_mean(rets: np.ndarray, entry_bars: np.ndarray, h: int) -> tuple[float, float]:
    """Event-time HAC (Newey-West, Bartlett-on-bar-distance) ``(t_stat, se)`` for the mean of
    forward-return observations.

    ``entry_bars`` are the observations' entry BAR indices (same basis as ``nonoverlap_count``)
    and ``h`` the measurement horizon in bars. A pair of observations whose entry bars sit ``d``
    bars apart shares ``h − d`` bars of measurement window, so its covariance enters with the
    Bartlett weight ``max(0, 1 − d/h)`` (Conley-style distance kernel — PSD in one dimension, so
    the variance stays nonnegative). Covariance terms carry the ``n − 1`` divisor (the HC1-style
    small-sample correction), so for a sparse pool whose neighboring events are ≥ ``h`` bars
    apart every cross term drops and the estimator reduces to the iid SE EXACTLY
    (``std(ddof=1)/√n``) at every n, not just asymptotically. For a contiguous every-bar pool
    (``d = k`` at event lag ``k``) it reproduces the classic overlap HAC with truncation lag
    ``h − 1`` under the same divisor — event ORDER is never mistaken for bar TIME (an ordinal-lag
    Bartlett with lag ≈ n drives the long-run variance toward 0 and fabricates significance).
    Same-bar pairs sit at weight 1 (``w(0) = 1``), so the long-run variance reads the rows only
    through their per-BAR deviation sums ``D_a``: ``Σ_a Σ_b w(|a−b|)·D_a·D_b``. It is computed
    in that form — each bar's deviation sum about the pool's EXACT mean, exact and rounded once
    (:func:`_bar_deviation_sums`), then the quadratic form over bars, each lag's terms exactly
    summed (:func:`_hac_from_bar_sums`). Accumulated row by row, the squared deviations and the
    cross terms that cancel them were summed apart, and their cancellation destroyed the
    precision of a much smaller combined variance: ten members firing together whose deviations
    cancel to ``(k − 2.5)·2**−24`` per bar read an SE of 3.73e-10 where the estimator's exact
    value is 1.05e-9 (``t`` 40.0 for 14.12). Rows with the same per-bar sums give the same
    estimate, no permutation of a bar's rows can move it (the basket pooled path), and a pool
    and its translate read the same sums; ``t`` is the reported mean — the pool's exactly rounded
    ``mean_ret`` (``_pools.exact_mean``) — over the SE.

    A (near-)constant pool (:func:`_near_constant`) is ``(NaN, NaN)``, like any other pool
    without a positive long-run variance: its dispersion is rounding residue — the computed mean
    of a constant DECIMAL pool is not the constant — and dividing by it minted ``|t| ≈ 1e16``.
    So is a pool whose BARS all sit exactly at its mean, however its rows vary — every row on one
    bar above all (:func:`_hac_from_bar_sums`); a pool whose mean's variance ``s/n`` is not a
    normal double (``_pools._normal``): past the range (returns past ~1e154) ``mean / inf`` read
    a finite ``t = 0.0`` beside an infinite SE; below it (returns under ~1e-154) the squared
    deviations had lost their bits — a t off by 0.7 — or underflowed to an SE of 0.0, and
    ``mean / 0.0`` raised ZeroDivisionError out of the run; and a pool with a deviation sum past
    the double range, which no double holds.

    No p-value is computed here: a consumer who wants one derives it from ``t_stat`` at
    ``df = n_nonoverlap − 1`` — the row count wildly overstates the information on an
    overlapping pool, and that reference is the pragmatic stand-in for Kiefer-Vogelsang fixed-b
    critical values.

    KNOWN ANTI-CONSERVATIVE on heavily overlapping pools: the Bartlett taper downweights exactly
    the lags that carry the overlap covariance, understating the long-run variance (SE ratio
    → √(2/3) ≈ 0.82 and worse as ``h`` grows; Monte Carlo under an iid-innovation null rejects
    ~10-12% at nominal 5%). No checklist reads any HAC statistic — evidence only.
    """
    rets = np.asarray(rets, dtype=float)
    bars = np.asarray(entry_bars, dtype=np.int64)
    keep = np.isfinite(rets)
    rets, bars = rets[keep], bars[keep]
    n = rets.shape[0]
    if n < 2 or h < 1:
        return (_NAN, _NAN)
    mean = exact_mean(rets)
    if not math.isfinite(mean):
        return (_NAN, _NAN)
    # A deviation from the mean may leave the double range (rows of opposite sign near ±1.7e308):
    # the zero-dispersion gate reads it as ±inf — an expected input, never a warning.
    with np.errstate(over="ignore", invalid="ignore"):
        if _near_constant(rets, mean):
            return (_NAN, _NAN)
    try:
        u, d_bar = _bar_deviation_sums(rets, bars)
    except OverflowError:  # a deviation sum no double holds
        return (_NAN, _NAN)
    return _hac_from_bar_sums(mean, u, d_bar, h, n)


def _exact_deviation_sums(
    values: list[float], groups: np.ndarray, counts: np.ndarray, scale: int
) -> np.ndarray:
    """Per group ``g``, the EXACT deviation sum ``D_g = (S_g − k_g·μ) / scale`` about the exact
    mean ``μ = Σ_g S_g / (scale·n)``, correctly rounded — ``S_g`` the sum of the group's
    ``values`` (``groups[i]`` is value ``i``'s group), ``k_g = counts[g]`` the rows it stands for
    and ``n`` their total. The values are read as integers over one power-of-two denominator
    (every double is one, ``float.as_integer_ratio``), so ``(n·S_g − k_g·Σ S) / (scale·n·den)``
    is a quotient of integers, rounded once by Python's true division: 0.0 exactly where the
    group sits at the mean, and no row's deviation rounded before its group sums it. Raises
    ``OverflowError`` where a sum leaves the double range."""
    ratios = [v.as_integer_ratio() for v in values]
    den = max(q for _, q in ratios)
    sums = [0] * counts.shape[0]
    for g, (p, q) in zip(groups.tolist(), ratios, strict=True):
        sums[g] += p * (den // q)
    total = sum(sums)
    n = int(counts.sum())
    return np.array(
        [
            (n * s - k * total) / (scale * n * den)
            for s, k in zip(sums, counts.tolist(), strict=True)
        ]
    )


def _bar_deviation_sums(rets: np.ndarray, bars: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(u, d_bar)`` of a finite pool: its distinct entry bars (ascending) and each bar's EXACT
    deviation sum ``D_a = Σ_{i∈a} r_i − k_a·μ`` about the pool's exact mean, correctly rounded
    (:func:`_exact_deviation_sums`). Rounding each row's deviation before the bar summed it —
    about the pool's range midpoint, the centre in those shifted coordinates — turned members
    whose returns cancel within a bar into variation between bars: three members alternating
    ``[0.8, −0.4, −0.4]`` and ``[0, 0, 0]`` beside a fourth at ``2**−52`` sit every bar exactly
    at the mean, and read sums of ±5.6e-17, an SE of 1.6e-18 and ``t`` 35.7 where the estimator
    has no variance at all; the fourth at ``2**−51`` on every other bar read an SE 27% high.
    Raises ``OverflowError`` where a sum leaves the double range."""
    u, group = np.unique(bars, return_inverse=True)
    return u, _exact_deviation_sums(rets.tolist(), group, np.bincount(group), 1)


def _hac_from_bar_sums(
    mean: float, u: np.ndarray, d_bar: np.ndarray, h: int, n: int
) -> tuple[float, float]:
    """``(mean / se, se)`` of a pool of ``n`` rows reported at ``mean``, read off its EXACT
    per-bar deviation sums ``d_bar`` at the distinct sorted bars ``u``: the Bartlett quadratic
    form ``Σ_a Σ_b w(|a−b|)·D_a·D_b``, each lag's terms exactly summed, over ``(n − 1)·n``
    (ddof=1).

    ``(NaN, NaN)`` — no estimate — wherever the pool has no positive long-run variance to read:
    every bar exactly at the mean (one bar above all, whose sum ``Σr − n·μ`` is 0 whatever its
    rows), a quadratic form past the double range (``math.fsum`` raises past it), or a variance
    ``s/n`` that is not a normal double (:func:`_normal`): past the range ``mean / inf`` is a
    FINITE t of 0.0 — an overflow must read as no estimate, never as "no edge" — and below
    ~1e-154 the products underflow, into lost bits or an SE of exactly 0.0.
    """
    if not np.any(d_bar):
        return (_NAN, _NAN)
    with np.errstate(over="ignore", invalid="ignore"):
        try:
            lags = [math.fsum((d_bar * d_bar).tolist())]  # gamma_0
            for k in range(1, u.shape[0]):
                w = 1.0 - (u[k:] - u[:-k]).astype(float) / h
                m = w > 0.0
                if not m.any():
                    # gaps at bar lag k are nondecreasing in k (sorted bars): no later pair
                    # overlaps
                    break
                lags.append(2.0 * math.fsum((w[m] * d_bar[k:][m] * d_bar[:-k][m]).tolist()))
            s = math.fsum(lags)
        except (OverflowError, ValueError):
            return (_NAN, _NAN)
    v = s / (n - 1.0) / n
    if not _normal(v):
        return (_NAN, _NAN)
    se = math.sqrt(v)
    return (mean / se, se)


def cross_mean_pooled_hac(
    excess: np.ndarray, closed: np.ndarray, numerators: BarNumerators, mean: float, h: int
) -> tuple[float, float]:
    """The event-time HAC ``(t_stat, se)`` of a ``cross_mean`` basket's POOLED excess mean —
    :func:`newey_west_mean`'s estimator, read off every bar's EXACT excess sum.

    ``excess`` is the (bars × members) excess leg of one horizon, ``closed`` the pool's closed
    observations (rows indexed by entry bar), ``numerators`` its bar numerators
    (:func:`~seikan.analysis._pools.cross_mean_bar_numerators`) and ``mean`` its
    :func:`~seikan.analysis._pools.cross_mean_pooled_mean` — the ``mean_ret`` the cell reports.
    The long-run variance reads the rows only through their per-bar deviation sums, and a
    ``cross_mean`` bar's closed excesses have an exact sum the rounded rows do not carry (each
    full bar's is 0, ``S − 1.0·S``): each ``D_t`` is the bar's exact numerator over the member
    count less its ``k_t`` closed rows' share of the exact mean, rounded once
    (:func:`_exact_deviation_sums`). Read off the rounded rows instead, a pool whose every bar's
    excess sum is 0 — every member firing together, or a firing subset whose excesses cancel on
    each bar — divided the residue: a ``mean_ret`` of exactly 0 beside ``t_hac`` 5.43, its
    ``hac_se`` 1.6e-19.

    ``t`` is computed here as ``mean / se``, so the panel's ``t_hac`` is its ``mean_ret`` over its
    ``hac_se`` on every pool. ``(NaN, NaN)`` under the same rules as :func:`newey_west_mean`: fewer
    than two rows, a non-finite mean, a (near-)constant pool, every bar at the mean (an all-zero
    excess pool among them), and a sum or variance out of the double range (a bar whose numerator
    did).
    """
    c = np.asarray(closed, dtype=bool)
    rows = np.asarray(excess, dtype=float)[c]
    n = int(rows.size)
    if n < 2 or h < 1 or not math.isfinite(mean):
        return (_NAN, _NAN)
    with np.errstate(over="ignore", invalid="ignore"):
        if _near_constant(rows, mean):
            return (_NAN, _NAN)
    held = np.flatnonzero(c.any(axis=1))
    parts = numerators.parts[held]
    if not np.isfinite(parts).all():
        return (_NAN, _NAN)
    try:
        d_bar = _exact_deviation_sums(
            parts.ravel().tolist(),
            np.repeat(np.arange(held.size), parts.shape[1]),
            c[held].sum(axis=1),
            c.shape[1],
        )
    except OverflowError:
        return (_NAN, _NAN)
    return _hac_from_bar_sums(mean, held, d_bar, h, n)


#: An int below any real bar index — the "no window open yet" sentinel for the greedy walk.
_BEFORE_ALL_BARS = -(10**18)


def nonoverlap_count(entry_idx: np.ndarray, horizon: int) -> int:
    """Greedy count of non-overlapping ``horizon``-bar windows among (sorted) entry bars —
    ``n_nonoverlap``: a lower bound on distinct market episodes and the overlap-honest sample
    size every df in the layer derives from. Deliberately NOT an independence certificate —
    non-overlapping windows still share regimes, factors and volatility clusters."""
    idx = np.sort(np.asarray(entry_idx, dtype=int))
    if idx.shape[0] == 0:
        return 0
    count = 0
    last_end = _BEFORE_ALL_BARS
    for t in idx:
        if t >= last_end:
            count += 1
            last_end = t + horizon
    return count
