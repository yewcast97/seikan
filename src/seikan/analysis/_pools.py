"""The pool layer, in two halves: the exact-arithmetic core every pool read stands on (exact
sums, means and masses, the integer terms, the zero-dispersion and normal-range gates), and the
per-pool descriptive blocks — the moment/shape reads, concentration, the order-statistic quantile
blocks, the excursion blocks, profit factor, edge ratio, the legs' OLS attribution and the
timing medians.

Every function here describes ONE observation pool from the arrays it is handed and nothing
else — per cell × target, or per basket-pooled cell — so nothing in this module can compare,
rank or aggregate across cells.

Finite returns can sum, square or interpolate past the double range, and every read here
decides such an operand by finiteness — :func:`is_normal`, :func:`_finite_ratio`, or the NaN →
null emission of a statistic that is not finite — so the reads run under
``errstate(over, invalid)``: an overflow is an expected input, never a warning (which ``-W
error`` turns into an exception out of the run).
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from typing import NamedTuple

import numpy as np
from scipy import stats as sps

from seikan.numeric import MIN_NORMAL, integer_terms, midrange
from seikan.types import (
    TOP_SHARE_FRAC,
    BenchmarkRegressionBlock,
    BenchmarkRegressionReason,
    ConcentrationBlock,
    MaeQuantiles,
    MfeQuantiles,
    PoolMoments,
    PoolQuantiles,
    TimingBlock,
    top_share_count,
)

_NAN = float("nan")


EPS = float(np.finfo(np.float64).eps)

#: The zero-dispersion gate's scale: a pool whose largest deviation from its mean is below this
#: many ulps of the mean's magnitude has no usable dispersion (:func:`near_constant`). CSCV's
#: split scores read the same gate off their per-block extrema.
NEAR_CONSTANT_ULPS = 10.0


def near_constant(rets: np.ndarray, mean: float) -> bool:
    """Scale-relative zero-dispersion detector for the moment statistics.

    A pool whose spread around its mean is below ~10 ulps of the mean's magnitude has no usable
    dispersion: the spread is the rounding of the level it sits at, and a skewness/kurtosis read
    off it would describe that rounding — we emit null instead. The scale is scipy's own
    catastrophic-cancellation trigger (``_stats_py._demean``, gh-15905), read around the
    EXACTLY ROUNDED mean the cell reports, in product form. The ``spread == 0`` arm covers the
    exact-constant pool (mean ≈ 0 included, where the relative test is vacuous); genuine
    variation around a near-zero mean is correctly NOT degenerate. The gate is about the
    NUMERICAL reliability of the two moment estimators, not a policy on thin pools: ``std_ret``
    and the order-statistic reads stay as computed (a near-constant pool's dispersion IS tiny) —
    save the exactly constant pool, whose ``std_ret`` is exactly 0.0 (:func:`ddof1_std`). The
    event-time HAC and the benchmark regression's bench-leg test read it too."""
    spread = float(np.max(np.abs(rets - mean))) if rets.size else 0.0
    return spread == 0.0 or bool(spread < NEAR_CONSTANT_ULPS * EPS * abs(mean))


def is_normal(x: float) -> bool:
    """``x`` is a positive NORMAL double — a variance (or a power or product of variances) that
    carries its full precision. Past the range it is ``inf``, and every quotient by it a FINITE
    0.0; below it the squared deviations it sums (deviations under ~1.5e-154) round to
    multiples of the smallest subnormal or to 0.0, so it has lost its significant bits — no
    estimate of a dispersion, and a varying pool's 0.0 is not zero dispersion either. A
    statistic read off such a variance is NaN → null. ``compiler.nb._normal`` is the kernels'
    twin on the overflow side, off the same :data:`seikan.numeric.MIN_NORMAL`; below the range
    the kernels LIFT the window back into it instead (``compiler.nb._lifted_windows``), because
    the series they read are derived and decay there from ordinary data (an ``ema`` of a change
    over a held stretch), where a pool reads measured returns, which reach it only off input
    data at that scale."""
    return MIN_NORMAL <= x < math.inf


def ddof1_std(v: np.ndarray) -> float:
    """The ddof=1 std of a pool of n ≥ 2, exactly 0.0 when every observation is equal — the
    computed mean of a constant DECIMAL pool is not the constant, and ``np.std`` would read its
    rounding residue (~1e-17) as dispersion. 0.0 is reserved for that pool: a VARYING pool whose
    variance is not a normal double (:func:`is_normal` — deviations below ~1.5e-154 or past
    ~1.3e154) has no representable dispersion: NaN, never a zero or an infinite dispersion. The
    pool is finite (the moment reads filter it, the bootstrap nulls its non-finite draws first).

    The variance is the two-pass form about the pool's OWN offset, its range midpoint
    (:func:`~seikan.numeric.midrange` — the kernels' cross-section rule): ``np.var`` of ``v − c``,
    whose deviations are exact wherever the pool lies within a factor of two of ``c``. About the
    rounded raw mean instead, a spread at the level's ulp loses its bits."""
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi == lo:
        return 0.0
    # Deviations past ~1.3e154 overflow the variance (a sum past the range, its mean first):
    # `is_normal` decides that read, so the overflow is an expected input here, never a warning.
    with np.errstate(over="ignore", invalid="ignore"):
        var = float(np.var(v - midrange(lo, hi), ddof=1))
    return math.sqrt(var) if is_normal(var) else _NAN


def exact_sum(values: np.ndarray | list[float]) -> float:
    """The EXACTLY ROUNDED sum of ``values`` (every entry of an array; over a pool of parts rows,
    :func:`exact_mean`'s, the exact sum of the rows' values times ``scale``) — ``math.fsum``, one
    double whatever order they come in. ``fsum`` accumulates in doubles and raises where a
    running partial leaves the range even when the exact sum has a double (``[1e308, 1e308,
    −1e308, −1e308]`` in that order): there the sum is retaken as exact rationals and rounded
    once. NaN where the exact sum itself leaves the double range, or meets a non-finite term
    (``inf − inf``) — a sum with no value, which every caller's finiteness test decides."""
    if isinstance(values, np.ndarray):
        values = values.ravel().tolist()
    try:
        return math.fsum(values)
    except OverflowError:
        try:
            return float(sum(map(Fraction, values)))
        except (OverflowError, ValueError):  # the exact sum past the range; a non-finite term
            return _NAN
    except ValueError:  # inf − inf
        return _NAN


def _exact_quotient(terms: list[float], count: int) -> float:
    """``Σterms / count`` correctly rounded — the EXACT sum (its exact parts,
    :func:`_exact_parts`, as rationals) over the integer ``count``, rounded ONCE. The exactly
    rounded sum over the count rounds twice, which is no rounding of the mean and can leave the
    pool's range; rounded once, a mean never leaves it. NaN where the sum has no double
    (:func:`exact_sum`'s rule)."""
    parts = _exact_parts(terms)
    if parts is None:
        return _NAN
    return float(sum(map(Fraction, parts)) / count)


def exact_mean(v: np.ndarray, scale: int = 1) -> float:
    """The mean of a pool of finite values — its EXACT sum over ``n``, rounded once
    (:func:`_exact_quotient`) — the read every mean the checklist grades, and its direct twins,
    share: ``pool_moments``' ``mean_ret``, the baseline's and the eras' means, the attribution
    legs and the HAC's centre (so ``t_hac`` is that ``mean_ret`` over its SE).

    ``np.mean`` sums pairwise in floating point, and a large term swallows a small one before
    its cancelling partner arrives: an exactly zero sum can read a nonzero mean, a sign the
    support floor's ``mean_ret > 0`` grades. The exact sum rounded once over ``n`` has the exact
    sum's sign, is ``0.0`` exactly where it is 0 and never leaves the pool's range; an all-equal
    pool's mean is its value. NaN on an empty pool, on a non-finite value, and where the exact sum
    leaves the double range (:func:`exact_sum`) — an overflowed mean is no evidence, as everywhere
    in this module.

    ``v`` may instead be ``n`` ROWS of exact parts (n × J), each row's value its parts' exact
    sum over the integer ``scale`` — a ``cross_mean`` bar's numerator over the member count
    (:class:`BarNumerators`), which no double holds: the mean is then the parts' exact sum over
    ``scale·n``, rounded once, so it is 0.0 exactly where the rows' values cancel. Its entries
    are all equal only where each row is one part (or none): that part over ``scale``."""
    n = int(v.shape[0])
    if n == 0:
        return _NAN
    lo, hi = float(np.min(v)), float(np.max(v))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return _NAN
    if lo == hi:
        return lo / scale
    return _exact_quotient(v.ravel().tolist(), scale * n)


#: The least mass no double holds — the largest double plus half its ulp, where an exact sum rounds
#: to inf. A pool whose |return| mass reaches it has no shares (:func:`mass_share`).
PAST_DOUBLE = Fraction(sys.float_info.max) + Fraction(math.ulp(sys.float_info.max)) / 2


def exact_mass(v: np.ndarray) -> Fraction:
    """The EXACT |return| mass of ``v`` — the rational sum of its magnitudes, taken from the
    sum's exact parts (:func:`_exact_parts`) where a double holds it and from the terms
    themselves past that."""
    mag = np.abs(np.asarray(v, dtype=float)).ravel().tolist()
    parts = _exact_parts(mag)
    return sum(map(Fraction, mag if parts is None else parts), Fraction(0))


def mass_share(part: Fraction, whole: Fraction) -> float:
    """``part / whole`` — the correctly rounded value of the EXACT quotient of two masses, the
    one share arithmetic (``top_share_abs``, the episode shares, ``member_share``). Rounded once,
    monotonically: no share passes 1 (``part <= whole``), one pool's shares are ordered as its
    masses are, and the largest of ``k`` shares is at least the correctly rounded ``1/k``, which
    the quotient of two separately rounded masses does not guarantee. NaN where ``whole`` is 0 or
    no double holds it (:data:`PAST_DOUBLE`): a zero mass has no shares, and an overflowed one
    would read ``x / inf = 0.0`` — a perfectly spread pool the ceiling passes."""
    if whole <= 0 or whole >= PAST_DOUBLE:
        return _NAN
    return float(part / whole)


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
    off a variance out of the normal double range — :func:`ddof1_std`, :func:`is_normal`);
    ``hit_rate`` is the share of returns strictly above zero (a zero return is not a hit);
    ``win_loss_ratio`` is the average win over the average |loss| with zero returns on neither
    side, NaN when either side is empty — the payoff-asymmetry partner of ``hit_rate``'s
    frequency; ``skewness``/``kurtosis`` are scipy's population (``bias=True``) moments,
    kurtosis PEARSON (normal = 3, not excess) — not pandas' unbiased G1/G2 — taken about the
    pool's offset (the :func:`ddof1_std` rule: the moments are shift-invariant, and deviations
    at the level's ulp keep their bits, which about scipy's own rounded mean they lose), nulled
    on a near-constant pool (:func:`near_constant`) and where the variance power they divide by
    leaves the normal range (:func:`is_normal`); ``tail_ratio`` is ``|p95/p05|`` of the same
    linear-interpolated percentiles :func:`pool_quantiles` reports (NaN when ``p05`` is zero,
    and a spread ratio rather than a tail read when both tails share a sign); ``cvar_5`` is the
    mean of the worst ``⌈n/20⌉`` observations (:func:`exact_mean`) — whole rows, at least 5% of
    the pool — the realized expected shortfall beside its historical-VaR partner ``p05``, the
    count :func:`concentration` takes, so a tie at ``p05`` never dilutes it.

    Overlap caveat, shared with every row-level read: one market move enters ~h rows, so
    these describe the realized pool, never an estimated distribution. Evidence-only; the
    checklist reads ``n``, ``mean_ret`` (a ``cross_mean`` basket's pooled ``mean_ret`` is
    re-summed by :func:`cross_mean_pooled_mean` instead) and nothing else here. Every other
    read here is ROW-level, on each row's own rounded value (a ``cross_mean`` full bar's
    nonzero excesses included); only the bar-closed SUMS read each bar at its exact excess sum
    (:func:`cross_mean_bar_numerators`) — a full bar at its exact 0.
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
    degenerate = near_constant(v, mean_ret)
    wins, losses = v[v > 0], v[v < 0]
    out["mean_ret"] = mean_ret
    out["std_ret"] = ddof1_std(v) if n > 1 else _NAN
    out["hit_rate"] = float(np.mean(v > 0))
    out["win_loss_ratio"] = (
        _finite_ratio(float(np.mean(wins)), abs(float(np.mean(losses))))
        if wins.size and losses.size
        else _NAN
    )
    # scipy's moments are ``m3 / m2**1.5`` and ``m4 / m2**2``: where that power of the variance
    # leaves the normal range (a dispersion below ~3e-103 / ~1.2e-77, or past ~5e102 / ~1.2e77)
    # the moment it normalizes has underflowed or overflowed with it. Null, never a finite
    # number read off the lost bits.
    skew = kurt = _NAN
    if n > 2 and not degenerate:
        d = v - midrange(float(np.min(v)), float(np.max(v)))
        m2 = np.var(d)
        if is_normal(float(m2**1.5)):
            skew = float(sps.skew(d))
        if n > 3 and is_normal(float(m2 * m2)):
            kurt = float(sps.kurtosis(d, fisher=False))
    # ...and the moment itself can overflow under a normal power (one deviation past ~5.6e102
    # cubes to inf beside a variance its many small rows keep in range): ±inf, never a value.
    out["skewness"] = skew if math.isfinite(skew) else _NAN
    out["kurtosis"] = kurt if math.isfinite(kurt) else _NAN
    p5, p95 = (float(q) for q in np.percentile(v, [5.0, 95.0]))
    out["tail_ratio"] = abs(_finite_ratio(p95, p5)) if p5 != 0 else _NAN
    out["cvar_5"] = exact_mean(np.sort(v)[: math.ceil(n / 20)])
    return out


@np.errstate(over="ignore", invalid="ignore")
def concentration(rets: np.ndarray) -> ConcentrationBlock:
    """How much of the pool's return mass sits in its largest few observations — the
    "one-episode thesis" detector. ``top_share_abs`` is the |return| mass of the top
    :data:`TOP_SHARE_FRAC` of observations (``n_top`` of them, :func:`top_share_count`) over the
    total |return| mass; near 1 means the whole edge is a handful of events (``n_nonoverlap``
    shows overlap inflation, this shows MASS concentration). The per-cell concentration
    checklist reads ``top_share_abs`` against a ceiling; an empty pool yields NaN, which that
    checklist refuses rather than waves through — and so does a pool whose total |return| mass
    leaves the double range (:func:`mass_share`). The share is the correctly rounded exact
    quotient of exact masses, so it never exceeds 1 and never falls below ``n_top / n``, the
    floor ``cell_evidence`` holds it to."""
    rets = np.asarray(rets, dtype=float)
    rets = rets[np.isfinite(rets)]
    n = rets.shape[0]
    if n == 0:
        return {"top_share_abs": _NAN, "n_top": 0, "top_frac": TOP_SHARE_FRAC}
    k = top_share_count(n)
    mag = np.sort(np.abs(rets))[::-1]
    share = mass_share(exact_mass(mag[:k]), exact_mass(mag))
    return {"top_share_abs": share, "n_top": k, "top_frac": TOP_SHARE_FRAC}


#: The percentiles every per-cell distribution block reports. Seven points, not a histogram:
#: a typical outcome, both shoulders, and the two 5% tails — ``p05`` doubling as the historical
#: VaR(5%) beside ``cvar_5``'s shortfall — without inviting a reader to treat the tails as
#: estimated quantities.
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
    for its CI, so ``tail_ratio`` is exactly ``|p95/p05|`` and ``p05`` is the VaR(5%) beside
    ``cvar_5``'s shortfall — all on the same pool. Each requested percentile is computed
    independently by ``np.percentile``, so widening this tuple can never move an existing point.
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
    values handed in (a censored path column contributes nothing; ``n`` counts the subset). The
    mean is :func:`exact_mean`, so it never passes ``worst`` (a sum rounded before its division
    can). One declaration of the emitted key ORDER, shared by the per-target, pooled and episode
    panels."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    return {
        "n": int(v.size),
        "mean": exact_mean(v),
        **pool_quantiles(v),
        "worst": float(np.min(v)) if v.size else _NAN,
    }


@np.errstate(over="ignore", invalid="ignore")
def mfe_block(vals: np.ndarray) -> MfeQuantiles:
    """The MFE twin of :func:`mae_block` — ``best`` (the max) where MAE carries ``worst``, and
    the exact mean never passes it."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    return {
        "n": int(v.size),
        "mean": exact_mean(v),
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
    a fact a reader takes from ``hit_rate``, not from an unbounded ratio.

    The masses are EXACT and their quotient is rounded once (:func:`exact_mass`, the arithmetic
    of :func:`mass_share`), so the read sits on the side of 1 the returns' exact sum puts it:
    never above 1 beside a negative sum, never below it beside a positive one, and exactly 1
    where the sum is 0. A quotient of two rounded sums can cross 1 against the sum's sign. NaN
    where either mass or the quotient has no double (:data:`PAST_DOUBLE`) — an overflowed loss
    mass would read ``x / inf = 0.0``. Evidence-only, per pool; no check reads it."""
    v = np.asarray(rets, dtype=float)
    v = v[np.isfinite(v)]
    wins = v[v > 0]
    losses = v[v < 0]
    if wins.size == 0 or losses.size == 0:
        return _NAN
    win_mass, loss_mass = exact_mass(wins), exact_mass(losses)
    ratio = win_mass / loss_mass
    return float(ratio) if max(win_mass, loss_mass, ratio) < PAST_DOUBLE else _NAN


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
    — the pairing is the point; each mean is the blocks' exact one (:func:`exact_mean`), so where
    no row holds a hole it is that quotient, rounded once. NaN when no pair
    survives or the paired adverse mean is exactly zero — an unbounded ratio is refused, never
    emitted — or when either mean leaves the double range (:func:`_finite_ratio`). Evidence-only;
    no check reads it."""
    m_a = np.asarray(mae_rows, dtype=float)
    m_f = np.asarray(mfe_rows, dtype=float)
    keep = np.isfinite(m_a) & np.isfinite(m_f)
    m_a, m_f = m_a[keep], m_f[keep]
    if m_a.size == 0:
        return _NAN
    denom = abs(exact_mean(m_a))
    if denom == 0:
        return _NAN
    return _finite_ratio(exact_mean(m_f), denom)


#: The low mantissa bits a leg's HIGH half clears (:func:`cross_mean_pooled_mean`): the high half
#: keeps at most 27 significant bits and the low one at most 26, so each half's product with an
#: integer weight under 2**26 — any basket size the engine can hold — is an exact double.
_SPLIT_LOW_BITS = np.int64((1 << 26) - 1)


def cross_mean_pooled_mean(raw: np.ndarray, closed: np.ndarray) -> float:
    """The pooled mean of a basket's ``cross_mean`` EXCESS returns, computed from the raw legs so
    that it is zero exactly where the exact excess sum is, and has its sign wherever it is not.

    ``raw`` is the (bars × members) signed raw leg of one horizon, ``closed`` the (bars ×
    members) mask of the pool's closed observations. Under ``cross_mean`` a bar's excess legs
    are ``raw[t, g] − mean(raw[t, :])``, so over the pool the excess sum is
    ``Σ_t (Σ_{closed} raw − (k_t/N)·Σ_all raw) = Σ_{t,g} (N·[closed_tg] − k_t)·raw_tg / N`` —
    an INTEGER-weighted sum of the legs, which is computed exactly: each leg is split into two
    halves whose products with its weight are exact doubles, and the products' exact sum over
    ``N·n`` is rounded once (:func:`_exact_quotient`). A bar on which every member closed weighs
    every leg 0 (``S − 1.0·S``); a bar whose legs are all the SAME value (a shared outcome feed
    broadcast to the basket) has weights that sum to 0 against it, whatever closed — each excess
    there is exactly zero. Rounding ``k_t/N`` first does not cancel, and summing the rounded
    per-row excesses leaves residue whose SIGN depends on the members' declaration order; the
    exact sum depends on no order at all.

    NaN on an empty pool. A bar with any closed member has every raw leg finite — a NaN leg voids
    the bar's whole cross-sectional mean, censoring every member there — so no NaN reaches the
    sum (a non-finite leg reads NaN regardless). Legs so large that a weighted product, or the
    exact total, leaves the double range have no finite pooled mean: NaN, which the gate refuses
    as non-finite — never an exception out of the run. The pool's other bar-closed sums read
    every BAR's exact numerator off the same weighted terms (:func:`cross_mean_bar_numerators`),
    and the pooled HAC reads them about the pool's exact mean
    (:func:`~seikan.analysis.stats.cross_mean_pooled_hac`); the rotation null meets the
    identities only through its tie band."""
    c = np.asarray(closed, dtype=bool)
    n = int(c.sum())
    if n == 0:
        return _NAN
    _, _, wh, wl = _cross_mean_weighted_parts(raw, c)
    if not (np.isfinite(wh).all() and np.isfinite(wl).all()):
        return _NAN
    # NaN where the exact total leaves the double range
    return _exact_quotient([*wh.ravel().tolist(), *wl.ravel().tolist()], wh.shape[1] * n)


def _cross_mean_weighted_parts(
    raw: np.ndarray, closed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(rows, k, wh, wl)``: the bars holding a closed member (ascending), each one's closed
    count ``k_t``, and the (rows × members) products of each leg's two halves with its integer
    weight ``N·[closed_tg] − k_t`` — exact doubles whose sum over a bar is ``N`` times its closed
    members' exact excess sum (:func:`cross_mean_pooled_mean`). A non-finite leg, or a product
    past the double range, leaves a non-finite term (a NaN leg's every product, a zero weight's
    included): each caller decides its read by the terms' finiteness."""
    rows = np.flatnonzero(closed.any(axis=1))
    r = np.ascontiguousarray(np.asarray(raw, dtype=float)[rows])
    c = closed[rows]
    k = c.sum(axis=1)
    weight = (int(r.shape[1]) * c - k[:, None]).astype(float)
    high = (r.view(np.int64) & ~_SPLIT_LOW_BITS).view(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        wh, wl = weight * high, weight * (r - high)
    return rows, k, wh, wl


def _exact_parts(terms: list[float]) -> list[float] | None:
    """Nonzero doubles whose EXACT sum is the exact sum of ``terms``, largest first: its
    correctly rounded value (``math.fsum``), then the correctly rounded value of what remains,
    until nothing does — each remainder under half an ulp of the part before it, so real legs
    take one part or two. One ``math.fsum`` over them, beside any other values, is the exact
    total correctly rounded. ``None`` where a term is not finite or the sum leaves the double
    range."""
    parts: list[float] = []
    while part := exact_sum([*terms, *(-p for p in parts)]):
        if not math.isfinite(part):  # a non-finite term; the exact sum past the range
            return None
        parts.append(part)
    return parts


def parts_table(groups: list[list[float]]) -> np.ndarray:
    """(groups × J): each group's :func:`_exact_parts`, zero-padded to the longest — a NaN row
    where it has none (a non-finite term, a sum past the double range). One ``math.fsum`` over any
    rows' parts is the exactly rounded sum of their groups' terms (NaN over a NaN row)."""
    by_group = [_exact_parts(terms) for terms in groups]
    table = np.zeros((len(by_group), max([1, *(len(p) for p in by_group if p is not None)])))
    for g, parts in enumerate(by_group):
        if parts is None:
            table[g] = np.nan
        else:
            table[g, : len(parts)] = parts
    return table


class BarNumerators(NamedTuple):
    """A ``cross_mean`` basket pool's bar NUMERATORS (:func:`cross_mean_bar_numerators`):
    ``parts`` (bars × J) holds each bar's numerator ``N·Σ_{closed g} e_tg`` EXACTLY, as the
    doubles :func:`_exact_parts` peels off it (zero-padded; zeros on a bar no member closed, NaN
    on one whose sum is unknown), and ``first`` (bars,) the bar's first closed member."""

    parts: np.ndarray
    first: np.ndarray

    def at(self, bars: np.ndarray, members: np.ndarray | int) -> np.ndarray:
        """(rows × J): what each closed row — at ``bars`` and ``members`` — adds to a bar-closed
        sum: its bar's parts on the bar's first closed member, zeros on the bar's others."""
        return np.where((members == self.first[bars])[:, None], self.parts[bars], 0.0)


def cross_mean_bar_numerators(raw: np.ndarray, closed: np.ndarray) -> BarNumerators:
    """The bar NUMERATORS of a ``cross_mean`` basket's pool — what its bar-closed SUMS read.

    A bar's exact excess sum ``Σ_{closed g} e_tg`` is its numerator — ``N`` times it, the exact
    sum of the bar's weighted leg halves (:func:`_cross_mean_weighted_parts`) — over the member
    count ``N``: no double holds it where ``N`` is no power of two, and none holds the numerator
    once it needs more than 53 bits. So each bar carries its numerator EXACTLY, as the doubles
    :func:`_exact_parts` peels off that sum, on its first closed member
    (:meth:`BarNumerators.at`), and a read that sums whole bars — the bootstrap's episode sums,
    the eras, the episode means, the CSCV split means, a bucket's mean over a bar it holds whole
    (the runner's ``rets_pool``, over ``pool_scale`` ``N``) — takes their parts' exact sum over
    ``N`` times its row count, rounded once (:func:`exact_mean`): 0.0 exactly where the excesses
    it holds cancel, across bars too, and otherwise their exact sum's sign. Dividing each bar by
    ``N`` first would round every bar apart, and a numerator rounded once still misses its
    cancelling partners wherever the legs span binades. NaN parts on a bar whose sum, or one of
    its weighted products, leaves the double range: every read over it is null, never a partial
    bar. Every row-level read keeps the row's own ``ret``."""
    c = np.asarray(closed, dtype=bool)
    rows, _k, wh, wl = _cross_mean_weighted_parts(raw, c)
    by_bar = parts_table([[*hi, *lo] for hi, lo in zip(wh.tolist(), wl.tolist(), strict=True)])
    parts = np.zeros((c.shape[0], by_bar.shape[1]))
    parts[rows] = by_bar
    return BarNumerators(parts, np.argmax(c, axis=1))


def scaled_parts(v: np.ndarray, scale: int) -> np.ndarray:
    """(n × 2): each value times the integer ``scale`` (below 2**26) as two EXACT doubles — its
    high half's product and its low half's (the leg split, ``_SPLIT_LOW_BITS``) — so that rows
    of values join a pool of :class:`BarNumerators` parts over the same ``scale``
    (:func:`exact_mean`). A product past the double range is ±inf."""
    v = np.ascontiguousarray(v, dtype=float)
    high = (v.view(np.int64) & ~_SPLIT_LOW_BITS).view(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.stack((scale * high, scale * (v - high)), axis=1)


@np.errstate(over="ignore", invalid="ignore")
def benchmark_regression(raw: np.ndarray, bench: np.ndarray) -> BenchmarkRegressionBlock:
    """Per-window OLS attribution of the raw leg on the benchmark leg, over PAIRED rows.

    The leg-attribution question the two means alone cannot answer: is the excess mean alpha, or
    beta ≠ 1 riding market drift? ``beta = cov(raw, bench) / var(bench)``, ``alpha = mean(raw) −
    beta·mean(bench)`` — per h-bar window, NEVER annualized, this engine carrying no
    calendar-return framing — and ``r2 = cov² / (var_raw · var_bench)``. beta and r2 are the
    EXACT quotients of the legs' exact second moments (the values as integers over one
    power-of-two denominator, :func:`~seikan.numeric.integer_terms`), each rounded once: the same
    double on every platform at any scale, an exact line its exact slope, and r2 never past 1 —
    never ``np.cov`` (a BLAS sum, in the platform's summation order, times a rounded ``1/(n − 1)``)
    over ``np.var`` (a pairwise sum over ``n − 1``), which can read identical legs at beta ≠ 1.
    The means are the legs' exactly rounded ones (:func:`exact_mean`) — the ``mean_ret_raw`` /
    ``mean_ret_bench`` the panel reports beside the fit — so the identity ``alpha +
    beta·mean(bench) == mean(raw)`` holds to the rounding of one product and one difference, a
    reader's to re-check.

    The two arrays must be ROW-ALIGNED (the same closed rows, in the same order): pairing is the
    whole point, which is why this takes the raw columns and filters PAIRS — a pre-filtered
    per-leg finite subset would silently mis-pair. Refusals return null fields with ``n`` and a
    ``reason``: ``no_paired_observations`` (which covers every unbenchmarked run — ``ret_bench``
    is all-NaN there — and every empty pool; the run-level ``benchmark`` stamp stays the
    authoritative "was there a benchmark"), ``insufficient_observations`` below n=3 (two points
    fit a line exactly — r² ≡ 1 vacuously), ``no_benchmark_variation`` on a (near-)constant bench
    leg (:func:`near_constant` — a constant DECIMAL leg's spread is the rounding of its level,
    and dividing by it would mint an arbitrary beta), ``not_representable`` when beta or alpha
    leaves the double range. ``r2`` alone is NaN → null when the raw leg is constant — nothing
    to explain, while beta = 0 (exactly: the leg co-varies with nothing) and alpha = mean(raw)
    are still real numbers. A VARYING raw leg is fitted however small its spread: the
    near-constant gate guards the divisor, the bench leg, never the raw leg. No moment is a
    double, so none leaves the range: a varying leg whose variance as a double would overflow or
    fall below the normal range is fitted exactly. On overlapping pools the fit is inflated
    exactly as every row-level read is — the caveat, not a correction, covers it. Evidence only,
    read by no check; a one-regressor attribution, not a factor model."""
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
    mean_b = exact_mean(b)
    if near_constant(b, mean_b):
        return _declined("no_benchmark_variation")
    # Each moment is n·Σxy − Σx·Σy over the legs' integers: n·(n − 1)·den² times the ddof=1
    # (co)variance, a scale both quotients cancel. Past the gate the bench leg varies, so
    # `var_b > 0`; Python's true division rounds once and raises past the double range.
    nums, _ = integer_terms(r.tolist() + b.tolist())
    xr, xb = nums[:n], nums[n:]
    sr, sb = sum(xr), sum(xb)
    cov = n * math.sumprod(xr, xb) - sr * sb
    var_b = n * math.sumprod(xb, xb) - sb * sb
    var_r = n * math.sumprod(xr, xr) - sr * sr
    try:
        beta = cov / var_b
    except OverflowError:
        return _declined("not_representable")
    alpha = exact_mean(r) - beta * mean_b
    if not math.isfinite(alpha):
        return _declined("not_representable")
    r2 = cov * cov / (var_r * var_b) if var_r else _NAN
    return {"n": n, "beta": beta, "alpha": alpha, "r2": r2, "reason": None}


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
