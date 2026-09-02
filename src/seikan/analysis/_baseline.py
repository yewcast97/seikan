"""The unconditional baseline summary and the equal-bar subperiod panels."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from seikan.analysis._pools import pool_quantiles
from seikan.types import (
    BaselineStats,
    SubperiodCounts,
)

_NAN = float("nan")


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
    it knows which pool it is describing. Non-finite entries are dropped defensively
    (``n_eligible`` counts what was actually described). An EMPTY pool is all-null, NEVER zeros
    — a zero base rate is a measured outcome, and a pool with no observations measured nothing.
    ``std_ret`` is ddof=1 (NaN below two observations, the same rule as ``pool_moments``);
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
    out["mean_ret"] = float(np.mean(v))
    out["std_ret"] = float(np.std(v, ddof=1)) if n > 1 else _NAN
    out["hit_rate"] = float(np.mean(v > 0))
    out["worst_ret"] = float(np.min(v))
    out["best_ret"] = float(np.max(v))
    return out


def subperiod_edges(length: int, k: int = 3) -> np.ndarray:
    """Contiguous equal-bar-count segment bounds over a ``length``-bar index — the same
    ``np.linspace`` edge rule as :func:`cscv_pbo`'s blocks. Computed ONCE per run from the shared
    joined index, so every cell reads the same eras."""
    return np.linspace(0, int(length), int(k) + 1).astype(int)


def subperiod_means(
    rets: np.ndarray, entry_bars: np.ndarray, edges: np.ndarray
) -> list[SubperiodCounts]:
    """Per segment ``{n, mean_ret}`` over a pool's closed observations, assigned by ENTRY bar.

    Era VISIBILITY, not a train/test split: there is NO purging — an observation belongs to its
    entry bar's segment even when its forward window crosses the boundary — and nothing reads the
    result. ``mean_ret`` is None for an empty segment (no evidence, not zero)."""
    rets = np.asarray(rets, dtype=float)
    bars = np.asarray(entry_bars, dtype=np.int64)
    keep = np.isfinite(rets)
    rets, bars = rets[keep], bars[keep]
    out: list[SubperiodCounts] = []
    for a, b in pairwise(edges):
        seg = rets[(bars >= a) & (bars < b)]
        out.append({"n": int(seg.size), "mean_ret": float(np.mean(seg)) if seg.size else None})
    return out
