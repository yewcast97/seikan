"""Per-pool descriptive blocks: the pool's moment/shape reads, concentration, the
order-statistic quantile blocks, the excursion blocks, profit factor, edge ratio, the legs' OLS
attribution and the timing medians.

Every function here describes ONE observation pool from the arrays it is handed and nothing
else — per cell × target, or per basket-pooled cell — so nothing in this module can compare,
rank or aggregate across cells.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

from seikan.types import (
    BenchmarkRegressionBlock,
    BenchmarkRegressionReason,
    ConcentrationBlock,
    MaeQuantiles,
    MfeQuantiles,
    PoolMoments,
    PoolQuantiles,
    TimingBlock,
)

_NAN = float("nan")


def _near_constant(rets: np.ndarray, mean: float) -> bool:
    """Scale-relative zero-dispersion detector for the moment statistics.

    Mirrors scipy's own catastrophic-cancellation trigger (``_stats_py._demean``, gh-15905):
    a pool whose spread around its mean is below ~10 ulps of the mean's magnitude has no usable
    dispersion for skewness/kurtosis — scipy would emit a "Precision loss" RuntimeWarning and
    an unreliable number; we emit null instead. The ``spread == 0`` arm covers the exact-constant
    pool (mean ≈ 0 included, where the relative test is vacuous); genuine variation around a
    near-zero mean is correctly NOT degenerate. The gate is about the NUMERICAL reliability of
    the two moment estimators, not a policy on thin pools: ``std_ret`` and the order-statistic
    reads stay as computed (a near-constant pool's dispersion IS tiny)."""
    spread = float(np.max(np.abs(rets - mean))) if rets.size else 0.0
    return spread == 0.0 or bool(spread < 10.0 * float(np.finfo(np.float64).eps) * abs(mean))


def pool_moments(rets: np.ndarray) -> PoolMoments:
    """The moment and shape reads of ONE pool's closed returns — ``{n, mean_ret, std_ret,
    hit_rate, win_loss_ratio, skewness, kurtosis, tail_ratio, cvar_5}``.

    Non-finite entries are dropped (``n`` counts what was described); an empty pool is ``n = 0``
    with every statistic NaN → null. ``std_ret`` is ddof=1 (NaN below two observations);
    ``hit_rate`` is the share of returns strictly above zero (a zero return is not a hit);
    ``win_loss_ratio`` is the average win over the average |loss| with zero returns on neither
    side, NaN when either side is empty — the payoff-asymmetry partner of ``hit_rate``'s
    frequency; ``skewness``/``kurtosis`` are scipy's population (``bias=True``) moments,
    kurtosis PEARSON (normal = 3, not excess) — not pandas' unbiased G1/G2 — nulled on a
    near-constant pool (:func:`_near_constant`); ``tail_ratio`` is ``|p95/p05|`` of the same
    linear-interpolated percentiles :func:`pool_quantiles` reports (NaN when ``p05`` is zero,
    and a spread ratio rather than a tail read when both tails share a sign); ``cvar_5`` is the
    mean of the observations at or below ``p05``, its historical-VaR partner.

    Overlap caveat, shared with every row-level read: one market move enters ~h rows, so
    these describe the realized pool, never an estimated distribution. Evidence-only; the
    checklist reads ``n``, ``mean_ret`` and nothing else here.
    """
    v = np.asarray(rets, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    out: PoolMoments = {
        "n": n,
        "mean_ret": _NAN,
        "std_ret": _NAN,
        "hit_rate": _NAN,
        "win_loss_ratio": _NAN,
        "skewness": _NAN,
        "kurtosis": _NAN,
        "tail_ratio": _NAN,
        "cvar_5": _NAN,
    }
    if n == 0:
        return out
    mean_ret = float(np.mean(v))
    degenerate = _near_constant(v, mean_ret)
    wins, losses = v[v > 0], v[v < 0]
    out["mean_ret"] = mean_ret
    out["std_ret"] = float(np.std(v, ddof=1)) if n > 1 else _NAN
    out["hit_rate"] = float(np.mean(v > 0))
    out["win_loss_ratio"] = (
        float(np.mean(wins) / abs(np.mean(losses))) if wins.size and losses.size else _NAN
    )
    out["skewness"] = float(sps.skew(v)) if n > 2 and not degenerate else _NAN
    out["kurtosis"] = float(sps.kurtosis(v, fisher=False)) if n > 3 and not degenerate else _NAN
    if n > 1:
        p5, p95 = (float(q) for q in np.percentile(v, [5.0, 95.0]))
        out["tail_ratio"] = float(abs(p95 / p5)) if p5 != 0 and not math.isnan(p5) else _NAN
        tail = v[v <= p5]
        out["cvar_5"] = float(np.mean(tail)) if tail.size else _NAN
    return out


def concentration(rets: np.ndarray, top_frac: float = 0.05) -> ConcentrationBlock:
    """How much of the pool's return mass sits in its largest few observations — the
    "one-episode thesis" detector. ``top_share_abs`` is the |return| mass of the top ``top_frac``
    observations over the total |return| mass; near 1 means the whole edge is a handful of events
    (``n_nonoverlap`` shows overlap inflation, this shows MASS concentration). The per-cell
    concentration checklist reads ``top_share_abs`` against a ceiling; an empty pool yields NaN,
    which that checklist refuses rather than waves through."""
    rets = np.asarray(rets, dtype=float)
    rets = rets[~np.isnan(rets)]
    n = rets.shape[0]
    if n == 0:
        return {"top_share_abs": _NAN, "n_top": 0, "top_frac": float(top_frac)}
    k = max(1, math.ceil(top_frac * n))
    mag = np.sort(np.abs(rets))[::-1]
    total = float(mag.sum())
    share = float(mag[:k].sum() / total) if total > 0 else _NAN
    return {"top_share_abs": share, "n_top": int(k), "top_frac": float(top_frac)}


#: The percentiles every per-cell distribution block reports. Seven points, not a histogram:
#: a typical outcome, both shoulders, and the two 5% tails — ``p05`` doubling as the historical
#: VaR(5%) whose lower tail ``cvar_5`` averages — without inviting a reader to treat the tails
#: as estimated quantities.
_QUANTILE_PS: tuple[float, ...] = (5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0)


def pool_quantiles(vals: np.ndarray) -> PoolQuantiles:
    """Order statistics of a pool — ``{p05, p10, p25, p50, p75, p90, p95}``.

    The SHAPE read the mean cannot give: a pool whose mean sits well above its ``p50`` is carried
    by a few observations, and a reader holding only ``mean_ret`` has no way to see that. Purely
    descriptive and per pool — nothing here compares pools, and no check reads it.

    Non-finite values are dropped (so a censored path column contributes nothing rather than
    poisoning the whole block); an EMPTY pool yields NaN at every point, which serializes to null
    — "no evidence", never zero. Percentiles use ``np.percentile``'s default linear interpolation,
    the same method :func:`pool_moments` uses for its own p5/p95 and the episode bootstrap uses
    for its CI, so ``p05`` agrees with the VaR(5%) read ``cvar_5`` tails off, and
    ``tail_ratio`` is exactly ``|p95/p05|`` — all on the same pool. Each requested
    percentile is computed independently by ``np.percentile``, so widening this tuple can never
    move an existing point.
    """
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    # `_QUANTILE_PS` stays the one declaration of WHICH percentiles are computed; the literal
    # below is what lets the checker verify the emitted key set without a cast.
    if v.size == 0:
        p05 = p10 = p25 = p50 = p75 = p90 = p95 = _NAN
    else:
        p05, p10, p25, p50, p75, p90, p95 = (float(q) for q in np.percentile(v, _QUANTILE_PS))
    return {"p05": p05, "p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90, "p95": p95}


def mae_block(vals: np.ndarray) -> MaeQuantiles:
    """The MAE excursion block — ``{n, mean, p05..p95, worst}`` over the FINITE subset of the
    values handed in (a censored path column contributes nothing; ``n`` counts the subset). One
    declaration of the emitted key ORDER, shared by the per-target, pooled and episode panels."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)) if v.size else _NAN,
        **pool_quantiles(v),
        "worst": float(np.min(v)) if v.size else _NAN,
    }


def mfe_block(vals: np.ndarray) -> MfeQuantiles:
    """The MFE twin of :func:`mae_block` — ``best`` (the max) where MAE carries ``worst``."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)) if v.size else _NAN,
        **pool_quantiles(v),
        "best": float(np.max(v)) if v.size else _NAN,
    }


def profit_factor(rets: np.ndarray) -> float:
    """Gross win mass over gross loss mass — ``Σ(wins) / |Σ(losses)|`` over a pool's finite
    returns. The MASS-weighted asymmetry partnering ``win_loss_ratio`` (which compares the
    AVERAGE win to the average loss and so says nothing about how often each side occurred):
    ``profit_factor ≈ (n_wins / n_losses) × win_loss_ratio``, so it is derivable and rides as a
    convenience read, never as independent evidence. Zero returns join neither side. NaN when
    EITHER side is empty — the same no-ratio-to-form rule as ``win_loss_ratio``, and this engine
    never emits an infinity (the JSON layer refuses non-finite tokens): "no losses in sample" is
    a fact a reader takes from ``hit_rate``, not from an unbounded ratio. Evidence-only, per
    pool; no check reads it."""
    v = np.asarray(rets, dtype=float)
    v = v[np.isfinite(v)]
    wins = v[v > 0]
    losses = v[v < 0]
    if wins.size == 0 or losses.size == 0:
        return _NAN
    return float(wins.sum() / abs(losses.sum()))


def edge_ratio(mae_rows: np.ndarray, mfe_rows: np.ndarray) -> float:
    """Mean favorable excursion over mean adverse excursion magnitude —
    ``mean(mfe) / |mean(mae)|`` over the PAIRED RAW post-entry excursion rows of one cell × target.

    The one sanctioned ratio BETWEEN the excursion pools: both legs are RAW path, so the read
    stays commensurable under a benchmark, where a ``ret``-vs-excursion ratio would divide an
    EXCESS return by a raw path and mean nothing. Above 1, the typical interim gain outran the
    typical interim pain — the measured answer to "would a stop have cost more than it saved?",
    which this engine otherwise leaves to the caller's read of the two quantile blocks.
    Deliberately UNNORMALIZED, unlike vectorbt's edge ratio (which scales each excursion by a
    volatility estimate at entry before averaging): this engine carries no per-entry volatility
    snapshot in the frozen layer, and an unnormalized ratio over one cell's own pool needs none —
    it is never compared across instruments. The inputs are ROW-ALIGNED (one entry per
    observation, same order) and the ratio is formed over the PAIRED rows where BOTH legs are
    finite: a window hole censors the frame it sits in without censoring the other, and a ratio
    of two means over DIFFERENT event subsets would compare pains and gains that never shared a
    trade. The ``mae_quantiles``/``mfe_quantiles`` blocks keep their own per-leg subsets, so
    under asymmetric holes this ratio need not equal ``mfe_quantiles.mean / |mae_quantiles.mean|``
    — the pairing is the point. NaN when no pair
    survives or the paired adverse mean is exactly zero — an unbounded ratio is refused, never
    emitted. Evidence-only; no check reads it."""
    m_a = np.asarray(mae_rows, dtype=float)
    m_f = np.asarray(mfe_rows, dtype=float)
    keep = np.isfinite(m_a) & np.isfinite(m_f)
    m_a, m_f = m_a[keep], m_f[keep]
    if m_a.size == 0:
        return _NAN
    denom = abs(float(np.mean(m_a)))
    if denom == 0:
        return _NAN
    return float(np.mean(m_f)) / denom


def benchmark_regression(raw: np.ndarray, bench: np.ndarray) -> BenchmarkRegressionBlock:
    """Per-window OLS attribution of the raw leg on the benchmark leg, over PAIRED rows.

    The leg-attribution question the two means alone cannot answer: is the excess mean alpha, or
    beta ≠ 1 riding market drift? ``beta = cov(raw, bench) / var(bench)`` (both ddof=1),
    ``alpha = mean(raw) − beta·mean(bench)`` — per h-bar window, NEVER annualized, this engine
    carrying no calendar-return framing — and ``r2 = cov² / (var_raw · var_bench)``. The identity
    ``alpha + beta·mean(bench) == mean(raw)`` holds exactly and is a reader's to re-check.

    The two arrays must be ROW-ALIGNED (the same closed rows, in the same order): pairing is the
    whole point, which is why this takes the raw columns and filters PAIRS — a pre-filtered
    per-leg finite subset would silently mis-pair. Refusals return null fields with ``n`` and a
    ``reason``: ``no_paired_observations`` (which covers every unbenchmarked run — ``ret_bench``
    is all-NaN there — and every empty pool; the run-level ``benchmark`` stamp stays the
    authoritative "was there a benchmark"), ``insufficient_observations`` below n=3 (two points
    fit a line exactly — r² ≡ 1 vacuously), ``no_benchmark_variation`` on a constant bench leg.
    ``r2`` alone is NaN → null when the raw leg has zero variance: nothing to explain, while
    beta = 0 and alpha = mean(raw) are still real numbers. On overlapping pools the fit is
    inflated exactly as every row-level read is — the caveat, not a correction, covers it.
    Evidence only, read by no check; a one-regressor attribution, not a factor model."""
    r = np.asarray(raw, dtype=float)
    b = np.asarray(bench, dtype=float)
    keep = np.isfinite(r) & np.isfinite(b)
    r, b = r[keep], b[keep]
    n = int(r.size)

    def _declined(reason: BenchmarkRegressionReason) -> BenchmarkRegressionBlock:
        return {"n": n, "beta": None, "alpha": None, "r2": None, "reason": reason}

    if n == 0:
        return _declined("no_paired_observations")
    if n < 3:
        return _declined("insufficient_observations")
    var_b = float(np.var(b, ddof=1))
    if var_b == 0:
        return _declined("no_benchmark_variation")
    cv = float(np.cov(r, b, ddof=1)[0, 1])
    beta = cv / var_b
    alpha = float(np.mean(r)) - beta * float(np.mean(b))
    var_r = float(np.var(r, ddof=1))
    r2 = (cv * cv) / (var_r * var_b) if var_r > 0 else _NAN
    return {"n": n, "beta": float(beta), "alpha": float(alpha), "r2": float(r2), "reason": None}


def timing_summary(bars_to_positive: np.ndarray, bars_to_trough: np.ndarray) -> TimingBlock:
    """Path-timing medians of one pool — the aggregation that spares a reader ``--trades-out``
    for the timing pair, exactly as the excursion blocks do for ``mae``/``mfe``.

    MEDIANS only: both durations are right-censored at the horizon (a path still under water at
    h reports no ``bars_to_positive`` at all, and a trough ON the last bar is indistinguishable
    from one just past it), and a mean of censored durations misleads. Each leg filters its own
    finite subset and reports the count that scopes it — ``bars_to_positive`` is finite only on
    rows whose path ever touched positive, so its median is a SURVIVORS-ONLY conditional read,
    never a recovery probability (``n_to_positive / n`` is the caller's to form). NaN → null
    medians on an empty subset. Evidence only; no check reads it."""
    btp = np.asarray(bars_to_positive, dtype=float)
    btp = btp[np.isfinite(btp)]
    btt = np.asarray(bars_to_trough, dtype=float)
    btt = btt[np.isfinite(btt)]
    return {
        "n_to_positive": int(btp.size),
        "median_bars_to_positive": float(np.median(btp)) if btp.size else _NAN,
        "n_to_trough": int(btt.size),
        "median_bars_to_trough": float(np.median(btt)) if btt.size else _NAN,
    }
