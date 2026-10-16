"""Overlap-episode clustering and its three panels: episode_stats, the time-ordered ledger, and
the episode-deduplicated profile.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pandas as pd

from seikan.analysis._pools import (
    edge_ratio,
    exact_mean,
    mae_block,
    mfe_block,
    pool_quantiles,
    profit_factor,
)
from seikan.types import (
    EpisodeLedgerBlock,
    EpisodeProfileBlock,
    EpisodeStatsBlock,
)

_NAN = float("nan")


def _iso(stamp: object) -> str:
    """One timestamp vocabulary for the episode panels — the same ``Timestamp.isoformat()`` the
    run's geometry stamps use, whatever numpy/pandas scalar the frame's ``to_numpy`` handed
    over (a raw ``datetime64`` would otherwise print nanoseconds)."""
    return str(pd.Timestamp(stamp).isoformat())


def overlap_clusters(entry: np.ndarray, exit_: np.ndarray) -> list[list[int]]:
    """Greedy merge of overlapping ``[entry, exit)`` windows (rows MUST be sorted by entry) into
    clusters of row indices — one cluster is one contiguous market episode. Half-open tie rule: a
    window starting exactly at the running cluster end starts a NEW cluster. Chain merges
    transitively (A∩B, B∩C ⇒ one cluster even when A∦C), so ``len(clusters)`` is never larger
    than a greedy non-overlapping count over the same windows."""
    clusters: list[list[int]] = []
    cur: list[int] = []
    cur_end = None
    for i in range(len(entry)):
        if cur_end is None or entry[i] >= cur_end:
            if cur:
                clusters.append(cur)
            cur = [i]
            cur_end = exit_[i]
        else:
            cur.append(i)
            if exit_[i] > cur_end:
                cur_end = exit_[i]
    if cur:
        clusters.append(cur)
    return clusters


class _EpisodeView(NamedTuple):
    df: pd.DataFrame
    entry: np.ndarray
    exit_: np.ndarray
    rets: np.ndarray
    #: What an episode MEAN sums per row — ``rets`` unless the caller handed ``rets_pool``.
    rets_pool: np.ndarray
    clusters: list[list[int]]


def _episode_view(trades: pd.DataFrame, rets_pool: np.ndarray | None = None) -> _EpisodeView | None:
    """The ONE episode pipeline — dropna → canonical stable sort → cross-target
    :func:`overlap_clusters` merge — shared by :func:`episode_stats`, :func:`episode_ledger` and
    :func:`episode_profile`, so their documented "same pipeline, step for step" claim is
    structural and ``n_episodes == n_clusters`` stays checkable. ``None`` for the empty case;
    each caller keeps its own explicit empty block.

    The sort is STABLE with a ``target`` tie-break where the frame carries one: same-``entry_time``
    rows (basket members on one bar) otherwise land in whatever order the caller's frame held —
    and the default unstable quicksort could reorder even that — moving the fp summation order,
    and with it the last bit of the mean/share reads, between equivalent inputs.
    ``(entry_time, target)`` is unique within a cell, so the order is total.

    ``rets_pool`` (row-aligned with ``trades``, finite wherever ``ret`` is) follows the rows
    through the dropna and the sort by POSITION — never as a frame column, which a caller's
    feature snapshot of the same name would collide with.
    """
    if trades is None or trades.empty or "entry_time" not in trades.columns:
        return None
    if rets_pool is not None and len(rets_pool) != len(trades):
        raise ValueError(
            f"rets_pool must be row-aligned with trades: {len(rets_pool)} != {len(trades)} rows"
        )
    # Positional labels, so the surviving rows' labels ARE their positions in `trades`.
    df = trades.reset_index(drop=True).dropna(subset=["entry_time", "exit_time", "ret"])
    if df.empty:
        return None
    keys = ["entry_time", "target"] if "target" in df.columns else ["entry_time"]
    df = df.sort_values(keys, kind="stable")
    entry = df["entry_time"].to_numpy()
    exit_ = df["exit_time"].to_numpy()
    rets = df["ret"].to_numpy(dtype=float)
    pool = rets if rets_pool is None else np.asarray(rets_pool, dtype=float)[df.index.to_numpy()]
    return _EpisodeView(df, entry, exit_, rets, pool, overlap_clusters(entry, exit_))


def _episode_means(pool: np.ndarray, clusters: list[list[int]]) -> np.ndarray:
    """Each episode's MEAN of its rows' pool values — the per-episode read the ledger lists and
    the profile aggregates — as the exactly rounded sum over the row count (:func:`exact_mean`):
    the exact sum's sign, and 0.0 exactly where it is 0, because the profile COUNTS those signs
    (hit rate, streaks). Summed pairwise, eight episodes of ``[−0.5, −2**−55, 0.5, 2**−55]``
    read a mean of 6.9e-18 each — a hit rate of 1 and a win streak of 8 beside a row
    ``mean_ret`` of exactly 0. NaN where the rows' sum leaves the double range: the true mean of
    finite returns is finite, but no computed one is, and an overflowed sum claims a sign the
    episode does not have. No value, no sign: NaN → null."""
    return np.array([exact_mean(pool[c]) for c in clusters])


# The three panels below sum |returns| and episode rows that can leave the double range; each
# such read is decided by finiteness (the shares' `0 < total < inf` rule, NaN episode means, the
# NaN → null emission of a statistic that is not finite), so the arithmetic runs under
# ``errstate(over, invalid)``: an overflow is an expected input, never a warning.
@np.errstate(over="ignore", invalid="ignore")
def episode_stats(trades: pd.DataFrame) -> EpisodeStatsBlock:
    """Cluster one cell's CLOSED observations by overlapping ``[entry, exit)`` windows.

    Diagnostic for sparse episodic pools: firings often cluster in shared regimes (crises,
    manias, event windows), so raw ``n`` / per-target ``n_nonoverlap`` can look thin while a
    handful of
    calendar clusters carry the mass. Clusters by greedy merge of overlapping intervals (sorted
    by entry — :func:`overlap_clusters`); reports ``n_clusters``, the largest-by-count cluster's
    |return|-mass share and earliest entry date, and ``max_cluster_share_abs`` — the largest
    |return|-mass share held by ANY single cluster, the one-episode detector the per-cell
    concentration checklist reads. ``mass_hhi`` (Σ share² over the clusters' |return|-mass
    shares) and ``effective_n_clusters`` (1/hhi ∈ [1, n_clusters]) are its SMOOTH companions:
    two episodes at 50% each clear a 0.60 ceiling while reading as exactly two effective
    episodes here. The remaining keys are evidence; no check reads the hhi pair.

    The rows are merged ACROSS targets, so a cluster is a market episode rather than a per-target
    one: an edge that is one crisis seen through three targets must not read as three episodes.
    An empty or all-NaN pool — a declared cell that never fired, which is reported and never
    dropped — returns the zero panel with NaN shares; a pool whose total |return| mass leaves the
    double range has NaN shares too (every share would read ``x / inf = 0.0``, which the
    one-episode ceiling passes).
    """
    empty: EpisodeStatsBlock = {
        "n": 0,
        "n_clusters": 0,
        "largest_cluster_n": 0,
        "largest_cluster_share_abs": _NAN,
        "largest_cluster_start": None,
        "max_cluster_share_abs": _NAN,
        "mass_hhi": _NAN,
        "effective_n_clusters": _NAN,
    }
    view = _episode_view(trades)
    if view is None:
        return empty
    df, entry, rets, clusters = view.df, view.entry, view.rets, view.clusters

    mag = np.abs(rets)
    total = float(mag.sum())
    shareable = 0.0 < total < math.inf  # a zero mass has no shares, an overflowed one neither
    cluster_mass = [float(mag[c].sum()) for c in clusters]
    largest = max(clusters, key=len)
    share = float(mag[largest].sum() / total) if shareable else _NAN
    # ONE shares array feeds the ceiling read AND its smooth Herfindahl companion — the ceiling
    # sees only the LARGEST share, so two 50% episodes sail under 0.60 while hhi = 0.5 says the
    # pool is exactly two effective episodes. NaN on a zero- or overflowed-mass pool, the shares'
    # own domain rule (evidence-only for the hhi pair; max_cluster_share_abs is the gated read).
    if shareable:
        shares = np.asarray(cluster_mass) / total
        max_share = float(shares.max())
        mass_hhi = float(np.sum(shares * shares))
        effective = float(1.0 / mass_hhi) if mass_hhi > 0 else _NAN
    else:
        max_share = _NAN
        mass_hhi = _NAN
        effective = _NAN
    start = _iso(entry[largest[0]]) if largest else None
    return {
        "n": len(df),
        "n_clusters": len(clusters),
        "largest_cluster_n": len(largest),
        "largest_cluster_share_abs": share,
        "largest_cluster_start": start,
        "max_cluster_share_abs": max_share,
        "mass_hhi": mass_hhi,
        "effective_n_clusters": effective,
    }


#: Ledger cap for the per-cell episode list. 32 entries is enough for a
#: narrative ("the entire edge is three episodes in 2020") while keeping the report bounded;
#: truncation is EXPLICIT and mass-conserving (``n_omitted`` + ``omitted_share_abs``), so a
#: count read off a truncated ledger is a floor, never the total.
EPISODE_LIST_MAX = 32


@np.errstate(over="ignore", invalid="ignore")
def episode_ledger(
    trades: pd.DataFrame,
    cap: int = EPISODE_LIST_MAX,
    *,
    rets_pool: np.ndarray | None = None,
) -> EpisodeLedgerBlock:
    """TIME-ORDERED per-episode ledger of one cell's CLOSED rows — the narrative companion to
    :func:`episode_stats`, which reports only the extremes.

    Same frozen pipeline as :func:`episode_stats`, step for step: rows drop NaN
    ``entry_time`` / ``exit_time`` / ``ret``, sort by ``entry_time``, and merge through
    :func:`overlap_clusters` — the half-open ``[entry, exit)`` greedy transitive merge, rows
    merged ACROSS targets so one cluster is one market episode (which is why in basket mode
    this already IS the pooled ledger). ``n_total`` therefore equals
    ``episode_stats["n_clusters"]`` on the same rows BY CONSTRUCTION — a reconciliation a
    reader re-checks from the report alone.

    Entries are EARLIEST FIRST and truncated at ``cap`` in that order — never ranked by share
    (ranking would crown episodes exactly the way the engine refuses to crown cells). Each
    entry: ``start`` / ``end`` (the cluster's earliest entry and latest exit — a chain merge
    can end on an interior row's window), ``n``, ``mean_ret``, and ``share_abs`` (the cluster's
    |return| mass over the pool total; NaN on a zero-mass pool or one whose total mass leaves the
    double range, which serializes to null).
    Truncation is mass-conserving: ``n_omitted`` counts the clusters past the cap and
    ``omitted_share_abs`` carries their combined share, so listed + omitted shares sum to ≈ 1
    and no episode silently vanishes. A dead cell returns the explicit empty ledger, never
    null. Evidence-only; no check reads it — the gated one-episode detector is
    ``episode_stats.max_cluster_share_abs``.

    ``mean_ret`` sums ``rets_pool`` (row-aligned with ``trades``; default the rows' ``ret``) —
    a ``cross_mean`` basket's bar values, each bar's exact excess sum on its first closed row and
    0.0 on the rest (``_pools.cross_mean_bar_values``); an episode holds each bar's rows
    entirely, so its mean is the exactly rounded sum of its bars' exact sums over its row count
    (:func:`_episode_means`). It is NaN → null where that sum leaves the double range (the same
    unknown mean ``episode_profile`` nulls its reads over), never an ``inf`` whose sign the
    overflow picked. The |return| shares stay on ``ret``: a row's mass is its own, not its bar's.
    """
    empty: EpisodeLedgerBlock = {
        "entries": [],
        "n_total": 0,
        "n_omitted": 0,
        "omitted_share_abs": 0.0,
        "cap": int(cap),
    }
    view = _episode_view(trades, rets_pool)
    if view is None:
        return empty
    entry, exit_, rets, clusters = view.entry, view.exit_, view.rets, view.clusters
    pool = view.rets_pool
    mag = np.abs(rets)
    total = float(mag.sum())

    def _share(rows: list[int]) -> float:
        # the episode_stats domain rule: no share of a zero or an overflowed total mass
        return float(mag[rows].sum() / total) if 0.0 < total < math.inf else _NAN

    kept = clusters[: max(int(cap), 0)]
    omitted = clusters[len(kept) :]
    means = _episode_means(pool, kept)
    return {
        "entries": [
            {
                "start": _iso(entry[c[0]]),  # rows sorted by entry: c[0] opens the episode
                "end": _iso(exit_[c].max()),
                "n": len(c),
                "mean_ret": float(m),
                "share_abs": _share(c),
            }
            for c, m in zip(kept, means, strict=True)
        ],
        "n_total": len(clusters),
        "n_omitted": len(omitted),
        # 0.0 when nothing was omitted (exact conservation over an empty remainder), NaN when
        # omitted clusters exist over a zero-mass pool — same domain rule as the entry shares.
        "omitted_share_abs": 0.0 if not omitted else float(sum(_share(c) for c in omitted)),
        "cap": int(cap),
    }


@np.errstate(over="ignore", invalid="ignore")
def episode_profile(
    trades: pd.DataFrame, *, rets_pool: np.ndarray | None = None
) -> EpisodeProfileBlock:
    """The episode-deduplicated TWIN of the row-level pool statistics — same family, EPISODE units.

    Overlapping observations from one market episode enter every row-level statistic up to ~h
    times (× members in basket): one crisis can be the whole tail of ``mae_quantiles``, most of
    ``hit_rate``, and ``profit_factor``'s loss mass. The calibration for that is deliberately NOT
    a reweighting — a correction would embed a modeling choice about the exchangeable unit this
    reporter has no business making — but a second view: cluster the rows into episodes with the
    SAME frozen pipeline as :func:`episode_stats` (dropna → sort by entry → cross-target
    :func:`overlap_clusters` merge, so one crisis seen through three targets is ONE episode),
    aggregate once per episode, and recompute the same statistics over the episodes.
    Row-vs-episode divergence is then itself the visible cluster diagnostic — the ``n`` vs
    ``n_nonoverlap`` doctrine applied to every pool statistic. In basket mode the cross-target merge
    makes this the pooled episode read for free.

    Per-episode aggregates: ``ret`` → the MEAN of the episode's rows (the ledger's established
    per-episode aggregate, summed off the same ``rets_pool`` as :func:`episode_ledger`'s
    ``mean_ret`` and exactly rounded like it — :func:`_episode_means` — and the unit that stays
    commensurable with the row-level ``mean_ret``; a SUM would scale with overlap density —
    re-importing exactly the inflation this block removes; ``mean_ret`` is their exactly
    rounded mean);
    ``mae`` → the episode's most adverse finite excursion (min), ``mfe`` → its most
    favorable (max), each excursion pool covering the episodes that carry a finite path column
    (subset ``n`` beside it, exactly like the row-level blocks) — while ``edge_ratio`` pairs at
    the EPISODE level, exactly as the row-level ratio pairs rows: only episodes carrying BOTH
    legs feed it. The streak pair is the one
    honest home for consecutive-outcome reads — over overlapping ROWS "consecutive" is a cluster
    artifact, over time-ordered episodes it is a run statistic: the longest run of episodes with
    strictly positive / strictly negative ``ret_mean``, a zero-mean episode breaking both — as
    does one whose mean overflowed, which also nulls every read over the episode means.

    Emits ALWAYS — descriptive statistics do not refuse on thinness (the bootstrap's 5-episode
    floor is a resampling impossibility, not a noise threshold; quantiles over two episodes rest
    on two episodes, which ``n_episodes`` says out loud). An empty pool is the zero block: zero
    counts and streaks, NaN statistics. ``n_episodes == episode_stats.n_clusters`` on the same
    rows by construction — a reader-checkable reconciliation, deliberately not gated. Evidence
    only; no check reads any field here."""
    view = _episode_view(trades, rets_pool)
    if view is None:
        return {
            "n_episodes": 0,
            "hit_rate": _NAN,
            "mean_ret": _NAN,
            "profit_factor": _NAN,
            "ret_quantiles": pool_quantiles(np.empty(0)),
            "worst_ret": _NAN,
            "best_ret": _NAN,
            "mae_quantiles": mae_block(np.empty(0)),
            "mfe_quantiles": mfe_block(np.empty(0)),
            "edge_ratio": _NAN,
            "max_win_streak": 0,
            "max_loss_streak": 0,
        }
    # Same frozen pipeline as `episode_stats`/`episode_ledger` — structurally, via `_episode_view`:
    # the clusters must be the very ones those blocks report, or `n_episodes == n_clusters` stops
    # being checkable.
    df, pool, clusters = view.df, view.rets_pool, view.clusters
    n_rows = len(df)
    mae = df["mae"].to_numpy(dtype=float) if "mae" in df.columns else np.full(n_rows, _NAN)
    mfe = df["mfe"].to_numpy(dtype=float) if "mfe" in df.columns else np.full(n_rows, _NAN)

    # Clusters arrive in time order (the greedy chain opens them at strictly increasing starts),
    # so the streak walk below IS the calendar walk.
    ret_means = _episode_means(pool, clusters)
    # EPISODE-ALIGNED excursion aggregates — one slot per cluster, NaN where the cluster carries
    # no finite value in that leg. The quantile blocks below read each leg's own finite subset
    # (as ever), while `edge_ratio` reads the ALIGNED pair, so an episode censored on one side
    # feeds its surviving leg's block but never the ratio.
    mae_by_episode = np.full(len(clusters), _NAN)
    mfe_by_episode = np.full(len(clusters), _NAN)
    for i, c in enumerate(clusters):
        finite_mae = mae[c][np.isfinite(mae[c])]
        if finite_mae.size:
            mae_by_episode[i] = float(finite_mae.min())
        finite_mfe = mfe[c][np.isfinite(mfe[c])]
        if finite_mfe.size:
            mfe_by_episode[i] = float(finite_mfe.max())
    mae_mins = mae_by_episode[np.isfinite(mae_by_episode)]
    mfe_maxes = mfe_by_episode[np.isfinite(mfe_by_episode)]

    # An episode whose mean is not finite — its rows' sum overflowed, though the true mean of
    # finite returns is finite — has no known value or sign: every read over the episode means
    # is NaN (the quantiles and the profit factor would read the pool without it, the extremes
    # and the hit rate off its overflow), and it breaks both streaks like a zero-mean episode.
    known = bool(np.isfinite(ret_means).all())
    win = loss = max_win = max_loss = 0
    for r in ret_means:
        if 0 < r < math.inf:
            win, loss = win + 1, 0
        elif -math.inf < r < 0:
            win, loss = 0, loss + 1
        else:  # a zero-mean episode joins neither side — the same rule as profit_factor's —
            win = loss = 0  # and neither does an overflowed one
        max_win = max(max_win, win)
        max_loss = max(max_loss, loss)

    return {
        "n_episodes": len(clusters),
        "hit_rate": float(np.mean(ret_means > 0)) if known else _NAN,
        "mean_ret": exact_mean(ret_means) if known else _NAN,
        "profit_factor": profit_factor(ret_means) if known else _NAN,
        "ret_quantiles": pool_quantiles(ret_means if known else np.empty(0)),
        "worst_ret": float(np.min(ret_means)) if known else _NAN,
        "best_ret": float(np.max(ret_means)) if known else _NAN,
        "mae_quantiles": mae_block(mae_mins),
        "mfe_quantiles": mfe_block(mfe_maxes),
        "edge_ratio": edge_ratio(mae_by_episode, mfe_by_episode),
        "max_win_streak": int(max_win),
        "max_loss_streak": int(max_loss),
    }
