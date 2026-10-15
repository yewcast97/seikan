"""The event-time Newey-West HAC mean and the greedy non-overlap count."""

from __future__ import annotations

import math

import numpy as np

from seikan.analysis._pools import (
    _EPS,
    _NEAR_CONSTANT_ULPS,
    _near_constant,
    _normal,
    cross_mean_bar_deviations,
    exact_mean,
)
from seikan.numeric import midrange

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
    in that form — each bar's deviation sum exactly rounded (:func:`_bar_deviation_sums`), then
    the quadratic form over bars, each lag's terms exactly summed (:func:`_hac_from_bar_sums`).
    Accumulated row by row, the squared deviations and the cross terms that cancel them were
    summed apart, and their cancellation destroyed the precision of a much smaller combined
    variance: ten members firing together whose deviations cancel to ``(k − 2.5)·2**−24`` per
    bar read an SE of 3.73e-10 where the estimator's exact value is 1.05e-9 (``t`` 40.0 for
    14.12). Rows with the same per-bar sums give the same estimate, and no permutation of a
    bar's rows can move it (the basket pooled path). The deviations are taken about the pool's
    mean read in shifted coordinates, to the rounding of the deviations themselves (see
    :func:`_bar_deviation_sums`), so a pool and its translate read the same SE; ``t`` is the
    reported mean — the pool's exactly rounded ``mean_ret`` (``_pools.exact_mean``) — over it.

    A (near-)constant pool (:func:`_near_constant`) is ``(NaN, NaN)``, like any other pool
    without a positive long-run variance: its dispersion is rounding residue — the computed mean
    of a constant DECIMAL pool is not the constant — and dividing by it minted ``|t| ≈ 1e16``.
    So is a pool whose BARS all sit at its mean, however its rows vary — every row on one bar
    above all, where the one deviation sum ``Σr − n·mean`` is exactly 0 and its computed
    residue minted the same ``|t| ≈ 1e16`` (:func:`_hac_from_bar_sums`); a pool whose mean's
    variance ``s/n`` is not a normal double (``_pools._normal``): past the range (returns past
    ~1e154) ``mean / inf`` read a finite ``t = 0.0`` beside an infinite SE; below it (returns
    under ~1e-154) the squared deviations had lost their bits — a t off by 0.7 — or underflowed
    to an SE of 0.0, and ``mean / 0.0`` raised ZeroDivisionError out of the run; and a finite
    pool whose same-bar sum leaves the double range (``math.fsum`` raises past it rather than
    returning inf), which raised OverflowError out of the run — an exit 4 on valid input.

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
    order = np.argsort(bars, kind="stable")
    rets, bars = rets[order], bars[order]
    n = rets.shape[0]
    if n < 2 or h < 1:
        return (_NAN, _NAN)
    mean = exact_mean(rets)
    if not math.isfinite(mean):
        return (_NAN, _NAN)
    # A deviation from the mean may leave the double range (rows of opposite sign near ±1.7e308):
    # the zero-dispersion gate and the one-row bars read it as ±inf, which the long-run variance
    # decides as no estimate — an expected input, never a warning; a same-bar sum of finite terms
    # past the range raises OverflowError, the same no-estimate.
    with np.errstate(over="ignore", invalid="ignore"):
        if _near_constant(rets, mean):
            return (_NAN, _NAN)
        try:
            u, count, d_bar = _bar_deviation_sums(rets, bars)
        except OverflowError:
            return (_NAN, _NAN)
    return _hac_from_bar_sums(mean, u, count, d_bar, h, n)


def _bar_deviation_sums(
    rets: np.ndarray, bars: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(u, count, d_bar)`` of a finite pool sorted by bar: its distinct entry bars, the rows
    on each, and each bar's deviation sum ``D_a = Σ_{i∈a} (r_i − μ)`` about the pool's mean.

    The sums are read in SHIFTED coordinates: ``y = r − c`` about the pool's range midpoint
    ``c`` (:func:`~seikan.numeric.midrange` — exact wherever the pool lies within a factor of two
    of it, the pools' ddof=1 std rule), the centre ``m = fsum(y) / n`` correctly rounded at THAT
    scale. The reported mean rounds at the level's ulp instead, and where the deviations are
    themselves a few ulps of the level that rounding is one of them: 10,000 returns at
    ``1 + k·2**-52`` read an SE of 1.714e-18 about it for the exact 1.312e-18, and the same pool
    shifted onto 0, whose mean is representable, the exact value. A pool and its translate read
    the same sums. A bar of one row reads ``y − m``, one rounding of its value; a bar of ``k > 1``
    rows is ONE ``math.fsum`` over its shifted returns and ``k`` copies of ``−m`` — no row's
    deviation is rounded before the bar sums it, and no ``k·m`` product is rounded either.
    Summing the rounded deviations lost the centre wherever a row's deviation rounded it away: on
    ``[0.5, −0.5, 0.5, −0.5, 2**−53]`` at bars ``[0, 0, 3, 3, 6]`` (h = 1) the two-row bars read
    0 for their exact ``−2·mean`` and the SE 1.986e-17 for the exact 2.432e-17. The order of a
    bar's rows cannot move the sum.

    Runs under the caller's ``errstate``: a one-row deviation past the double range is ±inf
    (decided by :func:`_hac_from_bar_sums`), and a sum past it raises OverflowError.
    """
    y = rets - midrange(float(np.min(rets)), float(np.max(rets)))
    m = math.fsum(y.tolist()) / y.size
    u, start, count = np.unique(bars, return_index=True, return_counts=True)
    d_bar = y[start] - m
    for i in np.flatnonzero(count > 1):
        a, k = int(start[i]), int(count[i])
        d_bar[i] = math.fsum([*y[a : a + k].tolist(), *([-m] * k)])
    return u, count, d_bar


def _hac_from_bar_sums(
    mean: float, u: np.ndarray, count: np.ndarray, d_bar: np.ndarray, h: int, n: int
) -> tuple[float, float]:
    """``(mean / se, se)`` of a pool of ``n`` rows centred on ``mean``, read off its per-bar
    deviation sums ``d_bar`` at the distinct sorted bars ``u`` (``count`` rows each): the
    Bartlett quadratic form ``Σ_a Σ_b w(|a−b|)·D_a·D_b``, each lag's terms exactly summed, over
    ``(n − 1)·n`` (ddof=1).

    ``(NaN, NaN)`` — no estimate — wherever the pool has no positive long-run variance to read:

    * one bar: its deviation sum ``Σr − n·mean`` is exactly 0 for the exact mean, and what the
      quadratic form read was the rounding of the centre — ``|t| ≈ 1e16`` for rows that vary;
    * every bar's mean deviation ``|D_a|/k_a`` under ~10 ulps of the mean — the same
      scale-relative gate :func:`_near_constant` puts on the rows, here on the bars (every bar
      at the pool's mean, however its rows vary): the centre's own rounding is ~1 ulp of the
      mean per row, so such sums are residue, not covariance;
    * a quadratic form past the double range (``math.fsum`` raises past it, and opposite
      infinities from an overflowed one-row deviation are ValueError) or a variance ``s/n``
      that is not a normal double (:func:`_normal`): past the range ``mean / inf`` is a
      FINITE t of 0.0 — an overflow must read as no estimate, never as "no edge" — and below
      ~1e-154 the products underflow, into lost bits or an SE of exactly 0.0.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        if u.shape[0] < 2:
            return (_NAN, _NAN)
        spread = float(np.max(np.abs(d_bar) / count))
        if spread == 0.0 or spread < _NEAR_CONSTANT_ULPS * _EPS * abs(mean):
            return (_NAN, _NAN)
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
    excess: np.ndarray, raw: np.ndarray, closed: np.ndarray, mean: float, h: int
) -> tuple[float, float]:
    """The event-time HAC ``(t_stat, se)`` of a ``cross_mean`` basket's POOLED excess mean —
    :func:`newey_west_mean`'s estimator, read off every bar's EXACT excess sum.

    ``excess`` and ``raw`` are the (bars × members) excess and signed raw legs of one horizon,
    ``closed`` the pool's closed observations (rows indexed by entry bar), ``mean`` the pool's
    :func:`~seikan.analysis._pools.cross_mean_pooled_mean` — the ``mean_ret`` the cell reports.
    The long-run variance reads the rows only through their per-bar deviation sums, and a
    ``cross_mean`` bar's closed excesses have an exact sum the rounded rows do not carry (each
    full bar's is 0, ``S − 1.0·S``): each ``D_t`` is formed from the raw legs, in the same
    weighted terms as the mean (:func:`~seikan.analysis._pools.cross_mean_bar_deviations`), so
    the centre and the sums it centres agree to the last bit. Read off the rounded rows instead,
    a pool whose every bar's excess sum is 0 — every member firing together, or a firing subset
    whose excesses cancel on each bar — divided the residue: a ``mean_ret`` of exactly 0 beside
    ``t_hac`` 5.43, its ``hac_se`` 1.6e-19.

    ``t`` is computed here as ``mean / se``, so the panel's ``t_hac`` is its ``mean_ret`` over its
    ``hac_se`` on every pool. ``(NaN, NaN)`` under the same rules as :func:`newey_west_mean`: fewer
    than two rows, a non-finite mean, a (near-)constant pool, every bar at the mean (an all-zero
    excess pool among them), and a sum or variance out of the double range.
    """
    c = np.asarray(closed, dtype=bool)
    rows = np.asarray(excess, dtype=float)[c]
    n = int(rows.size)
    if n < 2 or h < 1 or not math.isfinite(mean):
        return (_NAN, _NAN)
    with np.errstate(over="ignore", invalid="ignore"):
        if _near_constant(rows, mean):
            return (_NAN, _NAN)
    parts = cross_mean_bar_deviations(raw, c, mean)
    if parts is None:
        return (_NAN, _NAN)
    u, count, d_bar = parts
    return _hac_from_bar_sums(mean, u, count, d_bar, h, n)


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
