"""The event-time Newey-West HAC mean and the greedy non-overlap count."""

from __future__ import annotations

import math

import numpy as np

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
    Rows are canonically ordered by (entry bar, return) before accumulation, so the estimate is
    bit-identical under any permutation of tied-bar rows (the basket pooled path).

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
    # lexsort, not a stable bars-only sort: tied entry bars (basket pooled rows) then order by
    # return too, so the fp accumulation sequence — and with it the estimate's last bit — cannot
    # depend on the caller's row order.
    order = np.lexsort((rets, bars))
    rets, bars = rets[order], bars[order]
    n = rets.shape[0]
    if n < 2 or h < 1:
        return (_NAN, _NAN)
    mean = float(rets.mean())
    dev = rets - mean
    s = float(dev @ dev) / (n - 1.0)  # gamma_0 (long-run variance accumulator, ddof=1)
    for k in range(1, n):
        d = (bars[k:] - bars[:-k]).astype(float)
        w = 1.0 - d / h
        m = w > 0.0
        if not m.any():
            # gaps at event lag k are nondecreasing in k (sorted bars) — no later pair overlaps
            break
        s += 2.0 * float((w[m] * dev[k:][m] * dev[:-k][m]).sum()) / (n - 1.0)
    if not (s > 0.0):
        return (_NAN, _NAN)
    se = math.sqrt(s / n)
    return (mean / se, se)


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
