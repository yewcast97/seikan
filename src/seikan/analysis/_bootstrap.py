"""The per-cell episode bootstrap CI — the dependence-robust counterweight to the row-level
reads."""

from __future__ import annotations

import hashlib

import numpy as np

from seikan.analysis._episodes import overlap_clusters
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
    n_boot: int = BOOT_N,
    ci: float = BOOT_CI,
    min_episodes: int = BOOT_MIN_EPISODES,
) -> EpisodeBootstrapCI:
    """Percentile CI for the pool mean under an episode (cluster) bootstrap.

    The pool's closed observations are clustered into overlap-connected episodes over their
    ``[entry_bar, entry_bar + h)`` windows (:func:`overlap_clusters` — the frozen half-open
    greedy merge, applied per target here). Each draw resamples ``n_episodes`` episodes with
    replacement; the draw's statistic is the pooled mean over the resampled observations
    (Σ episode return-sums / Σ episode counts). Same keys always: ``{method, ci_level, n_boot,
    n_episodes, ci_lo, ci_hi, boot_se, reason}`` — an empty pool reports
    ``reason="no_observations"``, fewer than ``min_episodes`` episodes reports the honest
    ``n_episodes`` with ``reason="insufficient_episodes"`` (a dense every-bar pool transitively
    chains into ONE episode: there is no resampling distribution over one exchangeable unit,
    and the null fields say so rather than minting a degenerate interval).
    """
    rets = np.asarray(rets, dtype=float)
    bars = np.asarray(entry_bars, dtype=np.int64)
    keep = np.isfinite(rets)
    rets, bars = rets[keep], bars[keep]
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
    # by return.
    order = np.lexsort((rets, bars))
    bars, rets = bars[order], rets[order]
    clusters = overlap_clusters(bars, bars + int(h))
    n_ep = len(clusters)
    out["n_episodes"] = int(n_ep)
    if n_ep < min_episodes:
        out["reason"] = "insufficient_episodes"
        return out
    # Clusters partition the sorted rows into CONTIGUOUS runs (the greedy merge walks the rows in
    # order), so each episode is a reduceat segment.
    starts = np.fromiter((c[0] for c in clusters), dtype=np.int64, count=n_ep)
    ep_sums = np.add.reduceat(rets, starts)
    ep_counts = np.diff(np.append(starts, rets.size))

    digest = hashlib.sha256(np.int64(h).tobytes() + bars.tobytes() + rets.tobytes()).digest()
    rng = np.random.default_rng([_BOOT_SEED, *np.frombuffer(digest, dtype="<u4").tolist()])

    stats_arr = np.empty(n_boot, dtype=float)
    rows_per_chunk = max(1, _BOOT_CHUNK // n_ep)
    done = 0
    while done < n_boot:
        rows = min(rows_per_chunk, n_boot - done)
        idx = rng.integers(0, n_ep, size=(rows, n_ep))
        stats_arr[done : done + rows] = ep_sums[idx].sum(axis=1) / ep_counts[idx].sum(axis=1)
        done += rows

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(stats_arr, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    out["n_boot"] = int(n_boot)
    out["ci_lo"] = float(lo)
    out["ci_hi"] = float(hi)
    out["boot_se"] = float(np.std(stats_arr, ddof=1))
    return out
