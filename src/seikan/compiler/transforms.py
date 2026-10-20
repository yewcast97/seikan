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

from collections.abc import Sequence
from typing import cast

import numpy as np

from seikan.compiler import nb
from seikan.dsl.schema import (
    EMA,
    AxisRef,
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


def scalar[T](param: T | Sequence[T] | AxisRef) -> T:
    """The scalar a sweepable param holds by the time evaluation reads it.

    Params are DECLARED ``scalar | list | AxisRef`` because the DSL lets any of them SWEEP (as a
    list, or by reading a shared axis), but ``vectorize.iter_param_assignments`` expands that
    grid and scalarizes every node before a kernel is ever reached — the precondition this
    module's docstring opens with. This states that the expansion has happened, which the
    declared type cannot. An optional param keeps its ``None``: an omitted ``window`` selects the
    kernel's EXPANDING form (or, on the EW nodes, the ``alpha`` form), not a missing one.
    """
    return cast("T", param)


def transform_values(
    node: Series, feed: np.ndarray, *, labels: np.ndarray | None = None
) -> np.ndarray:
    """Compute a transform over feed column(s): (rows,) or (rows × targets) in → same shape out.

    Params are scalar. Every transform takes a precomputed input array (built by
    ``vectorize.build_series`` from the node's ``input`` Series). ``labels`` — point-in-time
    group labels shaped like ``feed`` — are legal for the cross nodes only, and reduce each
    cross-section within its labels (``nb.cross_grouped_apply_nb``).
    """
    raw = np.asarray(feed, dtype=float)
    x = _2d(raw)
    match node:
        case EMA(window=w, alpha=a):
            out = nb.ema_apply_nb(x, *nb.ema_params(scalar(w), scalar(a)))
        case ZScore(window=w, alpha=a, mean_type=mt):
            if mt == "sma":  # the SMA form always carries a window (``ZScore``'s validator)
                out = nb.zscore_sma_apply_nb(x, cast("int", w))
            else:
                out = nb.zscore_ema_apply_nb(x, *nb.ema_params(scalar(w), scalar(a)))
        case Percentile(window=w):
            out = nb.percentile_apply_nb(x, scalar(w))
        case RollingAgg(window=w, agg=agg):
            out = (
                nb.expanding_agg_apply_nb(x, agg)
                if w is None
                else nb.rolling_agg_apply_nb(x, scalar(w), agg)
            )
        case Drawdown(window=w):
            out = nb.drawdown_apply_nb(x, scalar(w))
        case Runup(window=w):
            out = nb.runup_apply_nb(x, scalar(w))
        case BarsSinceExtremum(window=w, extremum=ext):
            out = nb.bars_since_extremum_apply_nb(x, extremum=ext, window=scalar(w))
        case Change(periods=p, kind=k):
            out = nb.change_apply_nb(x, scalar(p), k)
        case Shift(periods=p):
            out = nb.shift_apply_nb(x, scalar(p))
        case UnaryOp(op=op):
            out = nb.unary_op_apply_nb(x, op)
        case CrossRank(min_valid=mv) | CrossDemean(min_valid=mv) | CrossAgg(min_valid=mv):
            # Cross-sectional: ranks/demeans/aggregates ACROSS the target columns at each bar, so a
            # single-column input has no cross-section (the Thesis validator requires basket mode
            # with >= 2 targets; a hand-built MarketData can still carry one).
            if x.shape[1] < 2:
                raise ValueError(
                    f"cross-sectional transform {node.type!r} requires >= 2 targets; "
                    f"got {x.shape[1]}"
                )
            if isinstance(node, CrossAgg):
                cross_agg = node.agg

                def kernel(a: np.ndarray) -> np.ndarray:
                    return nb.cross_agg_apply_nb(a, cross_agg, mv)

            else:
                if isinstance(node, CrossRank):
                    fn = nb.cross_rank_apply_nb
                elif node.scale == "std":  # the cross-sectional z-score
                    fn = nb.cross_zscore_apply_nb
                else:
                    fn = nb.cross_demean_apply_nb

                def kernel(a: np.ndarray) -> np.ndarray:
                    return fn(a, mv)

            out = kernel(x) if labels is None else nb.cross_grouped_apply_nb(kernel, x, _2d(labels))
        case _:
            raise TypeError(f"not a transform: {node!r}")
    out = _2d(np.asarray(out, dtype=float))
    return out.reshape(-1) if raw.ndim == 1 else out
