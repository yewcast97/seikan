"""The unconditional baseline summary and the equal-bar subperiod panels.

Both read means of finite returns that can sum past the double range — a statistic that is not
finite emits as null — so they run under ``errstate(over, invalid)``, as the ``_pools`` reads
do: an overflow is an expected input, never a warning.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from seikan.analysis._pools import _ddof1_std, exact_mean, pool_quantiles
from seikan.types import (
    BaselineStats,
    SubperiodCounts,
)

_NAN = float("nan")


@np.errstate(over="ignore", invalid="ignore")
def baseline_summary(rets: np.ndarray) -> BaselineStats:
    """Statistical fields of ONE baseline pool, from its eligible forward returns alone.

    THE SEAM (runner integration): a baseline row splits into fields the return ARRAY can carry
    and fields only the anchor GEOMETRY can. This function computes the former —
    ``{n_eligible, mean_ret, std_ret, hit_rate, ret_quantiles, worst_ret, best_ret}`` — from
    the eligible (finite) returns it is handed; the runner supplies ``n_anchor_bars`` and the
    ``exclusions`` breakdown (the exit-reason vocabulary minus ``horizon``) and mounts the row
    as ``{"n_anchor_bars": …, "exclusions": …, **baseline_summary(rets)}``. The arithmetic pin
    ``n_eligible + Σexclusions == n_anchor_bars`` is the RUNNER'S to satisfy — the exclusion
    counts are a property of the panel it reindexed, and this function cannot re-derive what
    was excluded WHY from the survivors it is handed.

    Pool-agnostic on purpose: the same function produces the per-(horizon × target) rows AND a
    basket's pooled row over the concatenated (bar × member) eligible observations — nothing in
    it knows which pool it is describing (under ``cross_mean`` the runner replaces the pooled
    row's ``mean_ret`` with ``cross_mean_pooled_mean``). Non-finite entries are dropped defensively
    (``n_eligible`` counts what was actually described). An EMPTY pool is all-null, NEVER zeros
    — a zero base rate is a measured outcome, and a pool with no observations measured nothing.
    ``mean_ret`` is the exactly rounded sum over ``n`` and ``std_ret`` ddof=1 (NaN below two
    observations, exactly 0.0 on a constant pool) — the same rules as ``pool_moments``;
    ``ret_quantiles`` rides :func:`pool_quantiles`, so its ``p50`` agrees with any median a
    caller derives — there is deliberately no ``median_ret`` key, ``ret_quantiles.p50`` IS it.
    """
    v = np.asarray(rets, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    out: BaselineStats = {
        "n_eligible": n,
        "mean_ret": _NAN,
        "std_ret": _NAN,
        "hit_rate": _NAN,
        "ret_quantiles": pool_quantiles(v),
        "worst_ret": _NAN,
        "best_ret": _NAN,
    }
    if n == 0:
        return out
    out["mean_ret"] = exact_mean(v)
    out["std_ret"] = _ddof1_std(v) if n > 1 else _NAN
    out["hit_rate"] = float(np.mean(v > 0))
    out["worst_ret"] = float(np.min(v))
    out["best_ret"] = float(np.max(v))
    return out


def subperiod_edges(length: int, k: int = 3) -> np.ndarray:
    """Contiguous equal-bar-count segment bounds over a ``length``-bar index — the same
    ``np.linspace`` edge rule as :func:`cscv_pbo`'s blocks. Computed ONCE per run from the shared
    joined index, so every cell reads the same eras."""
    return np.linspace(0, int(length), int(k) + 1).astype(int)


@np.errstate(over="ignore", invalid="ignore")
def subperiod_means(
    rets: np.ndarray,
    entry_bars: np.ndarray,
    edges: np.ndarray,
    *,
    rets_pool: np.ndarray | None = None,
) -> list[SubperiodCounts]:
    """Per segment ``{n, mean_ret}`` over a pool's closed observations, assigned by ENTRY bar.

    Era VISIBILITY, not a train/test split: there is NO purging — an observation belongs to its
    entry bar's segment even when its forward window crosses the boundary — and nothing reads the
    result. ``mean_ret`` is None for an empty segment (no evidence, not zero).

    ``rets_pool`` (row-aligned with ``rets``; default ``rets``) is what the era mean SUMS — a
    ``cross_mean`` basket's pooled panel hands in each bar's exact excess sum
    (``_pools.cross_mean_bar_values``): an era holds each entry bar's rows entirely, so its mean
    is the exactly rounded sum of its bars' exact sums over its row count. ``n`` and the rows an
    era holds stay on ``rets``; a non-finite pool value (a bar whose sum left the double range)
    nulls its era's mean, never drops the row."""
    rets = np.asarray(rets, dtype=float)
    bars = np.asarray(entry_bars, dtype=np.int64)
    pool = rets if rets_pool is None else np.asarray(rets_pool, dtype=float)
    if pool.shape != rets.shape:
        raise ValueError(
            f"rets_pool must be row-aligned with rets: shape {pool.shape} != {rets.shape}"
        )
    keep = np.isfinite(rets)
    bars, pool = bars[keep], pool[keep]
    out: list[SubperiodCounts] = []
    for a, b in pairwise(edges):
        seg = pool[(bars >= a) & (bars < b)]
        out.append({"n": int(seg.size), "mean_ret": exact_mean(seg) if seg.size else None})
    return out
