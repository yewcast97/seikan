"""The per-cell episode bootstrap CI — a percentile interval over resampled overlap episodes:
honest about within-episode overlap, it assumes independence BETWEEN episodes and undercovers on
few of them (under an iid null at horizon 1, ~16% of its 95% intervals exclude the true mean at
5 episodes, ~6% at 30). Not calibrated, and no ordering against ``t_hac`` either way."""

from __future__ import annotations

import hashlib
import math
from itertools import pairwise

import numpy as np

from seikan.analysis._episodes import overlap_clusters
from seikan.analysis._pools import PAST_DOUBLE, ddof1_std, exact_mean, integer_terms
from seikan.types import (
    EpisodeBootstrapCI,
)

_NAN = float("nan")


#: Fixed base entropy word for the bootstrap seed — never the clock, never a cell label. Combined
#: with a sha256 of the POOL CONTENT (horizon, then entry bars and returns in the CANONICAL
#: lexsort order (bar, ret) — a bars-only stable sort would let tied-bar row order leak into the
#: digest), so two identical runs draw identically, row order is genuinely irrelevant, and adding
#: a cell to the grid changes no other cell's draws (positional seeding would couple sibling
#: cells' Monte Carlo jitter to grid composition — the one doctrine content-keying preserves and
#: index-keying breaks).
_BOOT_SEED = 0x5E1CA10

BOOT_N = 2000

BOOT_CI = 0.95

BOOT_MIN_EPISODES = 5

#: Max (draws × episodes) index elements per RNG call. Purely a memory bound (~32 MB of int64):
#: the chunk row count is a pure function of the episode count, so the RNG call sequence — and
#: hence every drawn value — is deterministic for a given pool.
_BOOT_CHUNK = 1 << 22

#: The limb width a draw's exact sum is carried in: every episode's integer sum splits into
#: base-2**31 limbs (the top one signed), so each limb's sum over a draw's episodes is exact int64
#: arithmetic below 2**32 episodes, and the draw's sum is its limbs' recombination.
_LIMB_BITS = 31


def episode_bootstrap_ci(
    rets: np.ndarray,
    entry_bars: np.ndarray,
    h: int,
    *,
    rets_pool: np.ndarray | None = None,
    pool_scale: int = 1,
) -> EpisodeBootstrapCI:
    """Percentile CI for the pool mean under an episode (cluster) bootstrap.

    The pool's closed observations are clustered into overlap-connected episodes over their
    ``[entry_bar, entry_bar + h)`` windows (:func:`overlap_clusters` — the frozen half-open
    greedy merge, applied per target here). Each draw resamples ``n_episodes`` episodes with
    replacement; the draw's statistic is the pooled mean over the resampled observations
    (Σ episode return-sums / Σ episode counts), EXACT and rounded once: the values read as
    integers over one power-of-two denominator (``_pools.integer_terms``), each episode's sum
    an integer, each draw's the sum of its episodes' and its mean one quotient of integers,
    correctly rounded — the rule ``mean_ret`` reads (``_pools.exact_mean``), so a draw whose
    exact mean is the pool's reads ``mean_ret`` itself. Same keys always: ``{method, ci_level,
    n_boot, n_episodes, ci_lo, ci_hi, boot_se, reason}`` — an empty pool reports
    ``reason="no_observations"``, fewer than ``BOOT_MIN_EPISODES`` episodes reports the honest
    ``n_episodes`` with ``reason="insufficient_episodes"`` (a dense every-bar pool transitively
    chains into ONE episode: there is no resampling distribution over one exchangeable unit,
    and the null fields say so rather than minting a degenerate interval). A degenerate bootstrap
    distribution is reported exactly — ``ci_lo == ci_hi``, ``boot_se`` 0.0 — instead of the
    ulp-scale spread rounding would mint (a ``mean / boot_se`` of ~1e15, the ``t_hac``
    pathology): a pool whose summed values are all equal (its interval is that value — their
    exact mean, the pool's ``mean_ret``), and any pool whose resampled means come out
    bit-identical on every draw (every episode's exact mean the same, however its rows vary).
    Out of range, the block says so: a sum past the double range — an episode's, a draw's, or an
    all-equal pool's whose rows are a ``cross_mean`` bar's parts (its ``mean_ret`` null beside
    it) — nulls the interval and ``boot_se`` together, and a draws' variance that is not a normal
    double nulls ``boot_se`` — each with ``reason="not_representable"``.

    ``rets_pool`` (row-aligned with ``rets``; default ``rets`` itself) is what the episode sums
    ADD, over ``pool_scale`` — a ``cross_mean`` basket's pooled panel hands in its bar
    numerators over the member count (``_pools.cross_mean_bar_numerators``; NaN where a bar's
    sum left the double range, which nulls the interval like an overflowed episode sum). An
    episode holds each bar's rows entirely (one bar, one window), so each episode sum is its
    bars' exact numerator sum, and a draw divides the sum of its episodes' sums by
    ``pool_scale`` times its row count once. Everything else stays on ``rets``: the canonical
    (bar, ret) order, the seed digest (so the draw stream is the one ``rets`` alone would give)
    and the episodes.
    """
    rets = np.asarray(rets, dtype=float)
    bars = np.asarray(entry_bars, dtype=np.int64)
    pool = rets if rets_pool is None else np.asarray(rets_pool, dtype=float)
    keep = np.isfinite(rets)
    rets, bars, pool = rets[keep], bars[keep], pool[keep]
    out: EpisodeBootstrapCI = {
        "method": "episode_percentile",
        "ci_level": BOOT_CI,
        "n_boot": 0,
        "n_episodes": 0,
        "ci_lo": None,
        "ci_hi": None,
        "boot_se": None,
        "reason": None,
    }
    if rets.size == 0:
        out["reason"] = "no_observations"
        return out
    # lexsort, not a stable bars-only sort: the seed digest below hashes the sorted BYTES, so
    # tied-bar rows (the basket pooled pool) must land in one canonical order or the caller's
    # row order would pick the draw stream. (rets, bars) is canonical: within a bar, rows sort
    # by return. `pool` rides the same permutation; rows tied on (bar, ret) may still carry
    # different pool values in the caller's order, which no episode sum sees (each one exact).
    order = np.lexsort((rets, bars))
    bars, rets, pool = bars[order], rets[order], pool[order]
    clusters = overlap_clusters(bars, bars + int(h))
    n_ep = len(clusters)
    out["n_episodes"] = int(n_ep)
    if n_ep < BOOT_MIN_EPISODES:
        out["reason"] = "insufficient_episodes"
        return out
    out["n_boot"] = BOOT_N
    if (pool == pool[0]).all():
        # Every row adds the same value, so every draw's mean is that value — the exact mean
        # `_pools.exact_mean` reads (`mean_ret`), never a computed mean an ulp off it. Rows of
        # parts whose exact sum has no double have no mean to read (`mean_ret` is null beside
        # them): the block says so, as for an overflowed episode sum.
        mean = exact_mean(pool, pool_scale)
        if math.isnan(mean):
            return _not_representable(out)
        out["ci_lo"] = out["ci_hi"] = mean
        out["boot_se"] = 0.0
        return out
    # A NaN pool value (a `cross_mean` bar whose sum left the double range) has no integer to be;
    # it, an episode's sum past the double range and a draw's each null the whole interval and
    # `boot_se` alike — never half an interval — and say why (`_not_representable`).
    if not np.isfinite(pool).all():
        return _not_representable(out)
    # Clusters partition the sorted rows into CONTIGUOUS runs (the greedy merge walks the rows in
    # order), so each episode is one segment of the sorted pool — `width` parts per row (one, or
    # a `cross_mean` pool's bar parts).
    width = pool.size // rets.size
    nums, den = integer_terms(pool.ravel().tolist())
    starts = np.fromiter((c[0] for c in clusters), dtype=np.int64, count=n_ep)
    ep_counts = np.diff(np.append(starts, rets.size))
    bounds = pairwise([*starts.tolist(), rets.size])
    ep_sums = [sum(nums[a * width : b * width]) for a, b in bounds]
    past = int(PAST_DOUBLE) * den  # a sum (over `den`) this large has no double
    if any(abs(s) >= past for s in ep_sums):
        return _not_representable(out)
    # The episode sums as (limbs × episodes) int64 (`_LIMB_BITS`), so a draw's exact sum is its
    # per-limb int64 sums recombined — numpy-fast where summing the Python ints is not.
    n_limbs = -(-(max(abs(s) for s in ep_sums).bit_length() + 1) // _LIMB_BITS)
    shifts = [_LIMB_BITS * k for k in range(n_limbs)]
    low = (1 << _LIMB_BITS) - 1
    limbs = np.array(
        [[(s >> k) & low for k in shifts[:-1]] + [s >> shifts[-1]] for s in ep_sums],
        dtype=np.int64,
    ).T

    digest = hashlib.sha256(np.int64(h).tobytes() + bars.tobytes() + rets.tobytes()).digest()
    rng = np.random.default_rng([_BOOT_SEED, *np.frombuffer(digest, dtype="<u4").tolist()])

    stats_arr = np.empty(BOOT_N, dtype=float)
    rows_per_chunk = max(1, _BOOT_CHUNK // n_ep)
    done = 0
    while done < BOOT_N:
        rows = min(rows_per_chunk, BOOT_N - done)
        idx = rng.integers(0, n_ep, size=(rows, n_ep))
        sums = sum(
            limb[idx].sum(axis=1).astype(object) << k for limb, k in zip(limbs, shifts, strict=True)
        )
        draws = zip(sums, ep_counts[idx].sum(axis=1).tolist(), strict=True)
        stats_arr[done : done + rows] = [
            s / (den * pool_scale * c) if abs(s) < past else _NAN for s, c in draws
        ]
        done += rows
    if not np.isfinite(stats_arr).all():
        return _not_representable(out)
    alpha = (1.0 - BOOT_CI) / 2.0
    lo, hi = np.percentile(stats_arr, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    out["ci_lo"] = float(lo)
    out["ci_hi"] = float(hi)
    # Exactly 0.0 on bit-identical draws (a degenerate distribution); NaN when the draws'
    # variance is not a normal double — never a 0.0 read off an overflow or an underflow — and
    # then the block names it, as it does the interval's.
    boot_se = ddof1_std(stats_arr)
    out["boot_se"] = boot_se
    if math.isnan(boot_se):
        out["reason"] = "not_representable"
    return out


def _not_representable(out: EpisodeBootstrapCI) -> EpisodeBootstrapCI:
    """``out`` with the interval and ``boot_se`` null together, and the reason a null evidence
    block carries — a read past the double range."""
    out["ci_lo"] = out["ci_hi"] = out["boot_se"] = _NAN
    out["reason"] = "not_representable"
    return out
