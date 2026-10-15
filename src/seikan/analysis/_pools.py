"""Per-pool descriptive blocks: the pool's moment/shape reads, concentration, the
order-statistic quantile blocks, the excursion blocks, profit factor, edge ratio, the legs' OLS
attribution and the timing medians.

Every function here describes ONE observation pool from the arrays it is handed and nothing
else — per cell × target, or per basket-pooled cell — so nothing in this module can compare,
rank or aggregate across cells.

Finite returns can sum, square or interpolate past the double range, and every read here
decides such an operand by finiteness — :func:`_normal`, :func:`_finite_ratio`, or the NaN →
null emission of a statistic that is not finite — so the reads run under
``errstate(over, invalid)``: an overflow is an expected input, never a warning (which ``-W
error`` turns into an exception out of the run).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

from seikan.numeric import MIN_NORMAL
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


_EPS = float(np.finfo(np.float64).eps)

#: The zero-dispersion gate's scale: a pool whose largest deviation from its mean is below this
#: many ulps of the mean's magnitude has no usable dispersion (:func:`_near_constant`). CSCV's
#: split scores read the same gate off their per-block extrema.
_NEAR_CONSTANT_ULPS = 10.0


def _near_constant(rets: np.ndarray, mean: float) -> bool:
    """Scale-relative zero-dispersion detector for the moment statistics.

    Mirrors scipy's own catastrophic-cancellation trigger (``_stats_py._demean``, gh-15905):
    a pool whose spread around its mean is below ~10 ulps of the mean's magnitude has no usable
    dispersion for skewness/kurtosis — scipy would emit a "Precision loss" RuntimeWarning and
    an unreliable number; we emit null instead. The ``spread == 0`` arm covers the exact-constant
    pool (mean ≈ 0 included, where the relative test is vacuous); genuine variation around a
    near-zero mean is correctly NOT degenerate. The gate is about the NUMERICAL reliability of
    the two moment estimators, not a policy on thin pools: ``std_ret`` and the order-statistic
    reads stay as computed (a near-constant pool's dispersion IS tiny) — save the exactly
    constant pool, whose ``std_ret`` is exactly 0.0 (:func:`_ddof1_std`). The event-time HAC
    and the benchmark regression's degeneracy tests read it too."""
    spread = float(np.max(np.abs(rets - mean))) if rets.size else 0.0
    return spread == 0.0 or bool(spread < _NEAR_CONSTANT_ULPS * _EPS * abs(mean))


def _normal(x: float) -> bool:
    """``x`` is a positive NORMAL double — a variance (or a power or product of variances) that
    carries its full precision. Past the range it is ``inf``, and every quotient by it a FINITE
    0.0; below it the squared deviations it sums (deviations under ~1.5e-154) rounded to
    multiples of the smallest subnormal or to 0.0, so it has lost its significant bits — no
    estimate of a dispersion, and a varying pool's 0.0 is not zero dispersion either. A
    statistic read off such a variance is NaN → null. ``compiler.nb._normal`` is the kernels'
    twin on the overflow side, off the same :data:`seikan.numeric.MIN_NORMAL`; below the range
    the kernels LIFT the window back into it instead (``compiler.nb._lifted_windows``), because
    the series they read are derived and decay there from ordinary data (an ``ema`` of a change
    over a held stretch), where a pool reads measured returns, which reach it only off input
    data at that scale."""
    return MIN_NORMAL <= x < math.inf


def _ddof1_std(v: np.ndarray) -> float:
    """The ddof=1 std of a pool of n ≥ 2, exactly 0.0 when every observation is equal — the
    computed mean of a constant DECIMAL pool is not the constant, and ``np.std`` would read its
    rounding residue (~1e-17) as dispersion. 0.0 is reserved for that pool: a VARYING pool whose
    variance is not a normal double (:func:`_normal` — deviations below ~1.5e-154 or past
    ~1.3e154) has no representable dispersion, and neither has a pool holding a non-finite value
    (the bootstrap's draws, whose means can overflow — ``inf == inf`` is no degenerate
    distribution): NaN, never a zero or an infinite dispersion."""
    lo, hi = float(np.min(v)), float(np.max(v))
    # Every caller hands in a finite pool today (the moment reads filter, the bootstrap nulls
    # its non-finite draws first); the rule stays HERE, where `inf == inf` would pass for the
    # all-equal test below, rather than in each caller's filtering.
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return _NAN
    if hi == lo:
        return 0.0
    # Deviations past ~1.3e154 overflow the variance (a sum past the range, its mean first):
    # `_normal` decides that read, so the overflow is an expected input here, never a warning.
    with np.errstate(over="ignore", invalid="ignore"):
        var = float(np.var(v, ddof=1))  # np.std IS the root of this variance, bit for bit
    return math.sqrt(var) if _normal(var) else _NAN


def exact_mean(v: np.ndarray) -> float:
    """The mean of a pool of finite values as its EXACTLY ROUNDED sum over ``n`` — the read
    every mean the checklist grades, and its direct twins, share: ``pool_moments``' ``mean_ret``,
    the baseline's and the eras' means, the attribution legs and the HAC's centre (so ``t_hac``
    is that ``mean_ret`` over its SE).

    ``np.mean`` sums pairwise in floating point, and a large term swallows a small one before
    its cancelling partner arrives: the 49 diff returns of ``[0, 2**54, 0, −1, 0] × 10`` sum to
    exactly 0 and read a mean of 1/49 — a positive edge minted by rounding, which the support
    floor's ``mean_ret > 0`` graded. ``math.fsum`` is the correctly rounded sum, so the mean has
    the exact sum's sign and is ``0.0`` exactly where it is 0; an all-equal pool's mean is its
    value. NaN on an empty pool, on a non-finite value, and where the exact sum leaves the
    double range (``fsum`` raises rather than returning inf) — an overflowed mean is no
    evidence, as everywhere in this module."""
    n = int(v.size)
    if n == 0:
        return _NAN
    lo, hi = float(np.min(v)), float(np.max(v))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return _NAN
    if lo == hi:
        return lo
    try:
        return math.fsum(v.tolist()) / n
    except OverflowError:
        return _NAN


def _finite_ratio(num: float, den: float) -> float:
    """``num / den`` where both operands and the quotient are finite (and ``den != 0``), else NaN.

    The ratio reads of this module divide by a sum, a mean or a percentile of returns, and those
    of finite returns can still overflow: ``x / inf`` is a finite 0.0 — a profit factor, a
    win/loss ratio, a mass share of exactly zero — that a reader (or a ceiling check) would take
    at face value. An operand past the double range is no evidence: NaN → null."""
    if den == 0.0 or not (math.isfinite(num) and math.isfinite(den)):
        return _NAN
    q = num / den
    return q if math.isfinite(q) else _NAN


@np.errstate(over="ignore", invalid="ignore")
def pool_moments(rets: np.ndarray) -> PoolMoments:
    """The moment and shape reads of ONE pool's closed returns — ``{n, mean_ret, std_ret,
    hit_rate, win_loss_ratio, skewness, kurtosis, tail_ratio, cvar_5}``.

    Non-finite entries are dropped (``n`` counts what was described); an empty pool is ``n = 0``
    with every statistic NaN → null. ``std_ret`` is ddof=1 (NaN below two observations, and
    off a variance out of the normal double range — :func:`_ddof1_std`, :func:`_normal`);
    ``hit_rate`` is the share of returns strictly above zero (a zero return is not a hit);
    ``win_loss_ratio`` is the average win over the average |loss| with zero returns on neither
    side, NaN when either side is empty — the payoff-asymmetry partner of ``hit_rate``'s
    frequency; ``skewness``/``kurtosis`` are scipy's population (``bias=True``) moments,
    kurtosis PEARSON (normal = 3, not excess) — not pandas' unbiased G1/G2 — nulled on a
    near-constant pool (:func:`_near_constant`) and where the variance power they divide by
    leaves the normal range (:func:`_normal`); ``tail_ratio`` is ``|p95/p05|`` of the same
    linear-interpolated percentiles :func:`pool_quantiles` reports (NaN when ``p05`` is zero,
    and a spread ratio rather than a tail read when both tails share a sign); ``cvar_5`` is the
    mean of the observations at or below ``p05``, its historical-VaR partner (NaN where ``p05``
    itself left the double range).

    Overlap caveat, shared with every row-level read: one market move enters ~h rows, so
    these describe the realized pool, never an estimated distribution. Evidence-only; the
    checklist reads ``n``, ``mean_ret`` (a ``cross_mean`` basket's pooled ``mean_ret`` is
    re-summed by :func:`cross_mean_pooled_mean` instead) and nothing else here. Every other
    read here is ROW-level and reads each row's own rounded value, a ``cross_mean`` full bar's
    nonzero excesses included — only the bar-closed SUMS (the pooled panel's mean, the
    bootstrap's episode sums, the era and episode means, the HAC's per-bar deviation sums, the
    CSCV split means, a bucket mean over a bar it holds whole) read such a bar at its exact 0.
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
    mean_ret = exact_mean(v)
    degenerate = _near_constant(v, mean_ret)
    wins, losses = v[v > 0], v[v < 0]
    out["mean_ret"] = mean_ret
    out["std_ret"] = _ddof1_std(v) if n > 1 else _NAN
    out["hit_rate"] = float(np.mean(v > 0))
    out["win_loss_ratio"] = (
        _finite_ratio(float(np.mean(wins)), abs(float(np.mean(losses))))
        if wins.size and losses.size
        else _NAN
    )
    # scipy's moments are ``m3 / m2**1.5`` and ``m4 / m2**2``: where that power of the variance
    # leaves the normal range (a dispersion below ~3e-103 / ~1.2e-77, or past ~5e102 / ~1.2e77)
    # the moment it normalizes underflowed or overflowed with it — a kurtosis read 11.737849
    # for 11.737878 at a 1e-80 scale. Null, never a finite number read off the lost bits.
    shaped3 = shaped4 = False
    if n > 2 and not degenerate:
        m2 = np.var(v)
        shaped3, shaped4 = _normal(float(m2**1.5)), _normal(float(m2 * m2))
    # ...and the moment itself can overflow under a normal power (one deviation past ~5.6e102
    # cubes to inf beside a variance its many small rows keep in range): ±inf, never a value.
    skew = float(sps.skew(v)) if shaped3 else _NAN
    kurt = float(sps.kurtosis(v, fisher=False)) if n > 3 and shaped4 else _NAN
    out["skewness"] = skew if math.isfinite(skew) else _NAN
    out["kurtosis"] = kurt if math.isfinite(kurt) else _NAN
    if n > 1:
        p5, p95 = (float(q) for q in np.percentile(v, [5.0, 95.0]))
        out["tail_ratio"] = abs(_finite_ratio(p95, p5)) if p5 != 0 else _NAN
        tail = v[v <= p5]
        # p05's interpolation overflows to +inf across a gap past the double range, and
        # `v <= inf` is the WHOLE pool: a mean of every return (positive, even) read as the
        # lower tail — null, as p05 itself is
        out["cvar_5"] = float(np.mean(tail)) if tail.size and math.isfinite(p5) else _NAN
    return out


@np.errstate(over="ignore", invalid="ignore")
def concentration(rets: np.ndarray, top_frac: float = 0.05) -> ConcentrationBlock:
    """How much of the pool's return mass sits in its largest few observations — the
    "one-episode thesis" detector. ``top_share_abs`` is the |return| mass of the top ``top_frac``
    observations over the total |return| mass; near 1 means the whole edge is a handful of events
    (``n_nonoverlap`` shows overlap inflation, this shows MASS concentration). The per-cell
    concentration checklist reads ``top_share_abs`` against a ceiling; an empty pool yields NaN,
    which that checklist refuses rather than waves through — and so does a pool whose total
    |return| mass leaves the double range (:func:`_finite_ratio`)."""
    rets = np.asarray(rets, dtype=float)
    rets = rets[~np.isnan(rets)]
    n = rets.shape[0]
    if n == 0:
        return {"top_share_abs": _NAN, "n_top": 0, "top_frac": float(top_frac)}
    k = max(1, math.ceil(top_frac * n))
    mag = np.sort(np.abs(rets))[::-1]
    total = float(mag.sum())
    # an OVERFLOWED total mass read top/inf = 0.0 — "no concentration", which the ceiling passes
    share = _finite_ratio(float(mag[:k].sum()), total) if total > 0 else _NAN
    return {"top_share_abs": share, "n_top": int(k), "top_frac": float(top_frac)}


#: The percentiles every per-cell distribution block reports. Seven points, not a histogram:
#: a typical outcome, both shoulders, and the two 5% tails — ``p05`` doubling as the historical
#: VaR(5%) whose lower tail ``cvar_5`` averages — without inviting a reader to treat the tails
#: as estimated quantities.
_QUANTILE_PS: tuple[float, ...] = (5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0)


@np.errstate(over="ignore", invalid="ignore")
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


@np.errstate(over="ignore", invalid="ignore")
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


@np.errstate(over="ignore", invalid="ignore")
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


@np.errstate(over="ignore", invalid="ignore")
def profit_factor(rets: np.ndarray) -> float:
    """Gross win mass over gross loss mass — ``Σ(wins) / |Σ(losses)|`` over a pool's finite
    returns. The MASS-weighted asymmetry partnering ``win_loss_ratio`` (which compares the
    AVERAGE win to the average loss and so says nothing about how often each side occurred):
    ``profit_factor ≈ (n_wins / n_losses) × win_loss_ratio``, so it is derivable and rides as a
    convenience read, never as independent evidence. Zero returns join neither side. NaN when
    EITHER side is empty — the same no-ratio-to-form rule as ``win_loss_ratio``, and this engine
    never emits an infinity (the JSON layer refuses non-finite tokens): "no losses in sample" is
    a fact a reader takes from ``hit_rate``, not from an unbounded ratio. NaN too when either
    mass leaves the double range (:func:`_finite_ratio` — an overflowed loss mass read 0.0).
    Evidence-only, per pool; no check reads it."""
    v = np.asarray(rets, dtype=float)
    v = v[np.isfinite(v)]
    wins = v[v > 0]
    losses = v[v < 0]
    if wins.size == 0 or losses.size == 0:
        return _NAN
    return _finite_ratio(float(wins.sum()), abs(float(losses.sum())))


@np.errstate(over="ignore", invalid="ignore")
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
    emitted — or when either mean leaves the double range (:func:`_finite_ratio`). Evidence-only;
    no check reads it."""
    m_a = np.asarray(mae_rows, dtype=float)
    m_f = np.asarray(mfe_rows, dtype=float)
    keep = np.isfinite(m_a) & np.isfinite(m_f)
    m_a, m_f = m_a[keep], m_f[keep]
    if m_a.size == 0:
        return _NAN
    denom = abs(float(np.mean(m_a)))
    if denom == 0:
        return _NAN
    return _finite_ratio(float(np.mean(m_f)), denom)


def cross_mean_pooled_mean(raw: np.ndarray, closed: np.ndarray) -> float:
    """The pooled mean of a basket's ``cross_mean`` EXCESS returns, computed from the raw legs so
    the full-basket identity holds EXACTLY.

    ``raw`` is the (bars × members) signed raw leg of one horizon, ``closed`` the (bars ×
    members) mask of the pool's closed observations. Under ``cross_mean`` a bar's excess legs
    are ``raw[t, g] − mean(raw[t, :])``, so the bar contributes ``Σ_{closed} raw − (k_t/N)·Σ_all
    raw`` to the pooled sum — and a bar on which EVERY member closed contributes ``S − 1.0·S``,
    exactly ``0.0``. So does a bar on which every member's leg is the SAME value (a shared outcome
    feed broadcast to the basket), whatever closed: each excess there is exactly zero, which
    ``fl(k/N)`` would not cancel. Summing the rounded per-row excesses instead left ~1e-19 of
    residue whose SIGN depended on the members' declaration order, so a pool whose mean is zero by
    construction could clear ``mean_ret > 0`` in one order and not in another. Each bar's members
    are summed in SORTED order (the sum cannot depend on the order the targets were declared in)
    and the bars with ``math.fsum``; NaN on an empty pool. A bar with any closed member has every
    raw leg finite — a NaN leg voids the bar's whole cross-sectional mean, censoring every member
    there — so no NaN reaches the sums. Legs so large that a bar's sum, or the exact total, leaves
    the double range have no finite pooled mean: NaN, which the gate refuses as non-finite —
    never an exception out of the run. The same all-closed rows of ``closed`` are the full bars
    whose rows the pool's other bar-closed sums read at 0.0 (the runner's ``rets_pool``), so the
    bootstrap, the eras, the episode means and the HAC keep this identity too, as do the CSCV
    split means (a block holds whole bars — :func:`~seikan.analysis.stats.cscv_pbo`'s
    ``full_bars``) and the conditional-bucket means wherever a full bar's rows all land in one
    bucket (:func:`~seikan.analysis.stats.cell_conditional_buckets`' ``full_bars``) —
    the rotation null only through its tie band."""
    c = np.asarray(closed, dtype=bool)
    n = int(c.sum())
    if n == 0:
        return _NAN
    rows = c.any(axis=1)
    r = np.asarray(raw, dtype=float)[rows]
    c = c[rows]
    with np.errstate(over="ignore", invalid="ignore"):
        s_closed = np.sort(np.where(c, r, 0.0), axis=1).sum(axis=1)
        s_all = np.sort(r, axis=1).sum(axis=1)
        excess = s_closed - (c.sum(axis=1) / r.shape[1]) * s_all
    excess = np.where(r.max(axis=1) == r.min(axis=1), 0.0, excess)  # identical legs: no excess
    if not np.isfinite(excess).all():
        return _NAN
    try:
        total = math.fsum(excess.tolist())
    except OverflowError:  # the exact total leaves the double range
        return _NAN
    return total / n


@np.errstate(over="ignore", invalid="ignore")
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
    fit a line exactly — r² ≡ 1 vacuously), ``no_benchmark_variation`` on a (near-)constant bench
    leg (:func:`_near_constant` — a constant DECIMAL leg's computed variance is rounding residue,
    and dividing by it minted an arbitrary beta), ``not_representable`` when a VARYING bench
    leg's variance is not a normal double (:func:`_normal` — an overflowed one read ``beta = cov
    / inf = 0.0``, a fit that never happened; an underflowed one divided by its lost bits, or
    read as no variation at 0.0) or beta or alpha leaves the double range. ``r2`` alone is NaN
    → null when the raw leg has zero variance (likewise near-constant): nothing to explain, while
    beta = 0 (exactly) and alpha = mean(raw) are still real numbers — and when the raw variance
    is not a normal double. The variance PRODUCT under ``r2`` of two normal variances is taken
    in an unbounded exponent range (:func:`_r2`): past the ceiling it read ``inf / inf`` (a
    representable r2 lost), in the subnormal range it shed precision, and underflowed to 0.0 it
    raised ZeroDivisionError out of the run. On overlapping pools the fit is inflated exactly as
    every row-level read is — the caveat, not a correction, covers it. Evidence only, read by no
    check; a one-regressor attribution, not a factor model."""
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
    if _near_constant(b, float(np.mean(b))):
        return _declined("no_benchmark_variation")
    if not _normal(var_b):
        # a VARYING leg (a constant one is near-constant, above) whose variance left the range:
        # no zero variation, and no divisor either
        return _declined("not_representable")
    mean_r = float(np.mean(r))
    if _near_constant(r, mean_r):
        return {"n": n, "beta": 0.0, "alpha": mean_r, "r2": _NAN, "reason": None}
    cv = float(np.cov(r, b, ddof=1)[0, 1])
    beta = cv / var_b
    alpha = mean_r - beta * float(np.mean(b))
    if not (math.isfinite(beta) and math.isfinite(alpha)):
        return _declined("not_representable")
    var_r = float(np.var(r, ddof=1))
    r2 = _r2(cv, var_r, var_b) if _normal(var_r) else _NAN
    return {"n": n, "beta": float(beta), "alpha": float(alpha), "r2": float(r2), "reason": None}


def _r2(cv: float, var_r: float, var_b: float) -> float:
    """``cv² / (var_r · var_b)`` for two NORMAL variances (the caller's :func:`_normal` gate),
    exponent-safe: the direct form wherever the variance product is a normal double (bit for
    bit); elsewhere each variance's exponent is taken out first (made even, ``cv`` scaled by the
    half-sum) — power-of-two scaling of a normal double never rounds, so the ratio is EXACTLY the
    direct form's value in an unbounded exponent range. Only because the variances are normal: a
    subnormal one has already lost its bits (r2 read 2.0 for 0.99 off two of them). The scalar
    twin of ``compiler.nb._sqrt_product`` (the rolling correlation's denominator) — a change to
    the rule changes both."""
    prod = var_r * var_b
    if MIN_NORMAL <= prod < math.inf:
        return (cv * cv) / prod
    m_r, e_r = math.frexp(var_r)
    m_b, e_b = math.frexp(var_b)
    m_r, e_r = math.ldexp(m_r, e_r & 1), e_r - (e_r & 1)
    m_b, e_b = math.ldexp(m_b, e_b & 1), e_b - (e_b & 1)
    c = math.ldexp(cv, -((e_r + e_b) // 2))
    return (c * c) / (m_r * m_b)


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
