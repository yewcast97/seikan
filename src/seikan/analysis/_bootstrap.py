"""The per-cell episode bootstrap CI — the dependence-robust counterweight to the row-level
reads."""

from __future__ import annotations

import hashlib
import math
from itertools import pairwise

import numpy as np

from seikan.analysis._episodes import overlap_clusters
from seikan.analysis._pools import _ddof1_std, exact_sum
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


def episode_bootstrap_ci(
    rets: np.ndarray,
    entry_bars: np.ndarray,
    h: int,
    *,
    rets_pool: np.ndarray | None = None,
    n_boot: int = BOOT_N,
    ci: float = BOOT_CI,
    min_episodes: int = BOOT_MIN_EPISODES,
) -> EpisodeBootstrapCI:
    """Percentile CI for the pool mean under an episode (cluster) bootstrap.

    The pool's closed observations are clustered into overlap-connected episodes over their
    ``[entry_bar, entry_bar + h)`` windows (:func:`overlap_clusters` — the frozen half-open
    greedy merge, applied per target here). Each draw resamples ``n_episodes`` episodes with
    replacement; the draw's statistic is the pooled mean over the resampled observations
    (Σ episode return-sums / Σ episode counts). Each episode's return-sum is exactly rounded
    (``math.fsum``): summed in floating point, ``[1, −2**54, −1, 2**54]`` read 1 for its exact 0,
    and eight such episodes a zero-width interval at 0.25 with ``boot_se`` 0.0 beside a
    ``mean_ret`` of 0 — an interval excluding the mean it brackets. A draw's sum over its
    episodes' sums stays ordinary arithmetic. Same keys always: ``{method, ci_level, n_boot,
    n_episodes, ci_lo, ci_hi, boot_se, reason}`` — an empty pool reports
    ``reason="no_observations"``, fewer than ``min_episodes`` episodes reports the honest
    ``n_episodes`` with ``reason="insufficient_episodes"`` (a dense every-bar pool transitively
    chains into ONE episode: there is no resampling distribution over one exchangeable unit,
    and the null fields say so rather than minting a degenerate interval). A degenerate bootstrap
    distribution is reported exactly — ``ci_lo == ci_hi``, ``boot_se`` 0.0 — instead of the
    ulp-scale spread rounding would mint (a ``mean / boot_se`` of ~1e15, the ``t_hac``
    pathology): a pool whose summed values are all equal (its interval is that value — their
    exact mean, the pool's ``mean_ret``), and any pool whose
    resampled means come out bit-identical on every draw (every episode carrying the same sum
    and count). Out of range, the block says so: an episode sum or a draw past the double range
    nulls the interval and ``boot_se`` together, and a draws' variance that is not a normal
    double nulls ``boot_se`` — each with ``reason="not_representable"``.

    ``rets_pool`` (row-aligned with ``rets``; default ``rets`` itself) is what the episode sums
    ADD — a ``cross_mean`` basket's pooled panel hands in its bar values, each bar's EXACT excess
    sum on its first closed row and 0.0 on the rest (``_pools.cross_mean_bar_values``; NaN where a
    bar's sum left the double range, which nulls the interval like an overflowed episode sum). An
    episode holds each bar's rows entirely (one bar, one window), so every draw sums whole bars
    and each bar's sum survives resampling exactly. Everything else stays on ``rets``: the
    canonical (bar, ret) order (the pool values break only the ties left in it, so row order
    stays irrelevant), the seed digest (so the draw stream is the one ``rets`` alone would give)
    and the episodes.
    """
    rets = np.asarray(rets, dtype=float)
    bars = np.asarray(entry_bars, dtype=np.int64)
    pool = rets if rets_pool is None else np.asarray(rets_pool, dtype=float)
    if pool.shape != rets.shape:
        raise ValueError(
            f"rets_pool must be row-aligned with rets: shape {pool.shape} != {rets.shape}"
        )
    keep = np.isfinite(rets)
    rets, bars, pool = rets[keep], bars[keep], pool[keep]
    out: EpisodeBootstrapCI = {
        "method": "episode_percentile",
        "ci_level": float(ci),
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
    # by return. `pool` rides the same permutation and only breaks ties left in (bar, ret) —
    # rows equal in both, so the sorted bars/rets bytes and the digest stay the two-key ones,
    # while the episode sums add in one canonical order whatever pool values a caller hands in
    # (exact sums, `exact_sum`, whatever that order). Ordering on `pool` INSTEAD of `rets`
    # would hand a bar's tied zeros — and with them the hashed `rets` — back to the caller's
    # row order.
    order = np.lexsort((rets, bars) if rets_pool is None else (pool, rets, bars))
    bars, rets, pool = bars[order], rets[order], pool[order]
    clusters = overlap_clusters(bars, bars + int(h))
    n_ep = len(clusters)
    out["n_episodes"] = int(n_ep)
    if n_ep < min_episodes:
        out["reason"] = "insufficient_episodes"
        return out
    if float(pool.max()) == float(pool.min()):
        out["n_boot"] = int(n_boot)
        # Every value is equal, so `pool[0]` IS their exact mean — the `mean_ret` the pool
        # reports (`_pools.exact_mean`), where a computed mean read an ulp off it (0.3 as
        # 0.29999999999999993) and an overflowed sum a null interval beside `boot_se` 0.0.
        out["ci_lo"] = out["ci_hi"] = float(pool[0])
        out["boot_se"] = 0.0
        return out
    # Clusters partition the sorted rows into CONTIGUOUS runs (the greedy merge walks the rows in
    # order), so each episode is one segment of the sorted pool. Its sum is exactly rounded; a
    # sum past the double range is NaN, which the finiteness test below decides.
    starts = np.fromiter((c[0] for c in clusters), dtype=np.int64, count=n_ep)
    ep_counts = np.diff(np.append(starts, rets.size))
    ep_sums = np.array(
        [exact_sum(pool[a:b]) for a, b in pairwise([*starts.tolist(), int(pool.size)])]
    )

    digest = hashlib.sha256(np.int64(h).tobytes() + bars.tobytes() + rets.tobytes()).digest()
    rng = np.random.default_rng([_BOOT_SEED, *np.frombuffer(digest, dtype="<u4").tolist()])

    stats_arr = np.empty(n_boot, dtype=float)
    rows_per_chunk = max(1, _BOOT_CHUNK // n_ep)
    done = 0
    # A draw's sum of episode sums may leave the double range: the finiteness test below decides
    # every such read, so the overflow is an expected input, never a warning.
    with np.errstate(over="ignore", invalid="ignore"):
        while done < n_boot:
            rows = min(rows_per_chunk, n_boot - done)
            idx = rng.integers(0, n_ep, size=(rows, n_ep))
            stats_arr[done : done + rows] = ep_sums[idx].sum(axis=1) / ep_counts[idx].sum(axis=1)
            done += rows

    out["n_boot"] = int(n_boot)
    # An episode sum that overflowed makes every draw that resamples it NaN, and a draw's own sum
    # can leave the double range (±inf, ranked at one end: the percentile read one FINITE bound
    # over draws that left the range beside a null one). Any such sum or draw nulls the whole
    # interval and `boot_se` alike, and says why: a null evidence block carries its reason.
    if not (np.isfinite(ep_sums).all() and np.isfinite(stats_arr).all()):
        out["ci_lo"] = out["ci_hi"] = out["boot_se"] = _NAN
        out["reason"] = "not_representable"
        return out
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(stats_arr, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    out["ci_lo"] = float(lo)
    out["ci_hi"] = float(hi)
    # Exactly 0.0 on bit-identical draws (a degenerate distribution); NaN when the draws'
    # variance is not a normal double — never a 0.0 read off an overflow or an underflow — and
    # then the block names it, as it does the interval's.
    boot_se = _ddof1_std(stats_arr)
    out["boot_se"] = boot_se
    if math.isnan(boot_se):
        out["reason"] = "not_representable"
    return out
