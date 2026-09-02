"""Transform dispatch layer (pure numpy + numba kernels in ``compiler/nb.py``; no vectorbt).

Maps a scalar-param DSL transform node to its kernel and returns a plain 2D ``np.ndarray`` shaped
like the input (rows × targets); ``vectorize.py`` wraps those back into target-columned DataFrames.
The kernel ``*_apply_nb`` functions already accept and return (rows × targets), so there is no
per-call fan-out here — and for the cross-sectional family (``cross_rank``/``cross_demean``/
``cross_agg``, axis=1 across the target columns) the 2-D shape is load-bearing rather than a
batching convenience: those are the kernels with no 1-D form. Params are already scalar at this
point — the thesis-level parameter grid is expanded by ``vectorize.iter_param_assignments`` before
we get here.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from seikan.compiler import nb
from seikan.dsl.schema import (
    EMA,
    BarsSinceExtremum,
    Change,
    CrossAgg,
    CrossDemean,
    CrossRank,
    Drawdown,
    Percentile,
    RollingAgg,
    Runup,
    Series,
    Shift,
    UnaryOp,
    ZScore,
)


def _2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _scalar(param: int | list[int]) -> int:
    """The scalar value a window/period param already holds by the time dispatch sees it.

    Transform params are DECLARED ``int | list[int]`` because the DSL lets any window or period
    SWEEP, but ``vectorize.iter_param_assignments`` expands that grid and scalarizes every node
    before a kernel is ever reached — the precondition this module's docstring opens with. The
    kernels take the scalar; this states that the expansion has already happened, which the
    declared type cannot.
    """
    return cast("int", param)


def _scalar_or_none(param: int | list[int] | None) -> int | None:
    """:func:`_scalar` for the windows that are legitimately optional, where ``None`` is a real
    value rather than an unexpanded sweep: an omitted ``window`` selects the kernel's EXPANDING
    form — the extremum taken from the first bar — not a missing one."""
    return cast("int | None", param)


def transform_values(node: Series, feed: np.ndarray) -> np.ndarray:
    """Compute a transform over feed column(s): (rows,) or (rows × targets) in → same shape out.

    Params are scalar. Every transform takes a precomputed input array (built by
    ``vectorize.build_series`` from the node's ``input`` Series).
    """
    raw = np.asarray(feed, dtype=float)
    x = _2d(raw)
    match node:
        case EMA(window=w):
            out = nb.ema_apply_nb(x, _scalar(w))
        case ZScore(window=w, mean_type=mt):
            out = (
                nb.zscore_sma_apply_nb(x, _scalar(w))
                if mt == "sma"
                else nb.zscore_ema_apply_nb(x, _scalar(w))
            )
        case Percentile(window=w):
            out = nb.percentile_apply_nb(x, _scalar(w))
        case RollingAgg(window=w, agg=agg):
            out = (
                nb.expanding_agg_apply_nb(x, agg)
                if w is None
                else nb.rolling_agg_apply_nb(x, _scalar(w), agg)
            )
        case Drawdown(window=w):
            out = nb.drawdown_apply_nb(x, _scalar_or_none(w))
        case Runup(window=w):
            out = nb.runup_apply_nb(x, _scalar_or_none(w))
        case BarsSinceExtremum(window=w, extremum=ext):
            out = nb.bars_since_extremum_apply_nb(x, extremum=ext, window=_scalar_or_none(w))
        case Change(periods=p, kind=k):
            out = nb.change_apply_nb(x, _scalar(p), k)
        case Shift(periods=p):
            out = nb.shift_apply_nb(x, _scalar(p))
        case UnaryOp(op=op):
            out = nb.unary_op_apply_nb(x, op)
        case CrossRank(min_valid=mv) | CrossDemean(min_valid=mv) | CrossAgg(min_valid=mv):
            # Cross-sectional: ranks/demeans/aggregates ACROSS the target columns at each bar, so a
            # single-column input has no cross-section (the Thesis validator already requires
            # basket mode with >= 2 targets; this guards direct callers and 1-D feed inputs).
            if x.shape[1] < 2:
                raise ValueError(
                    f"cross-sectional transform {node.type!r} requires >= 2 targets; "
                    f"got {x.shape[1]}"
                )
            if isinstance(node, CrossAgg):
                out = nb.cross_agg_apply_nb(x, node.agg, mv)
            else:
                fn = (
                    nb.cross_rank_apply_nb
                    if isinstance(node, CrossRank)
                    else nb.cross_demean_apply_nb
                )
                out = fn(x, mv)
        case _:
            raise TypeError(f"not a transform: {node!r}")
    out = _2d(np.asarray(out, dtype=float))
    return out.reshape(-1) if raw.ndim == 1 else out
