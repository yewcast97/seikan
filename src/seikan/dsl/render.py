"""Deterministic presentation-only rendering of Series expressions (never part of the hash)."""

from __future__ import annotations

from seikan.dsl.nodes import (
    EMA,
    BarsSinceExtremum,
    BinaryOp,
    Calendar,
    Change,
    Constant,
    CrossAgg,
    CrossDemean,
    CrossRank,
    DaysSince,
    Drawdown,
    External,
    Field,
    Percentile,
    RollingAgg,
    RollingCorr,
    Runup,
    Series,
    Shift,
    UnaryOp,
    ZScore,
    _NumericParam,
)


def fmt_num(value: _NumericParam) -> str:
    """Compact deterministic numeric literal: integral floats render as ints (100.0 → '100')."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def render_series(node: Series) -> str:
    """Deterministic compact expression label for a scalar-param Series node — the value-column
    name the root-series output CSV uses (e.g. ``percentile(iv30,80)``, ``(close/ema(close,20))``).

    Every semantic field is rendered (windows/periods positionally, non-default modes as a trailing
    token), so distinct scalarized nodes render distinctly; the one realistic collision source is a
    user-chosen feed name shadowing another label — the CSV assembler disambiguates with ``#N``
    suffixes. Presentation only: rendering never participates in the DSL hash.
    """
    match node:
        case Field(column=column):
            return column
        case Constant(value=value):
            return fmt_num(value)
        case External(name=name):
            return name
        case Calendar(field=cal_field):
            return f"calendar({cal_field})"
        case DaysSince(name=name):
            return f"days_since({name})"
        case EMA(input=inp, window=w):
            return f"ema({render_series(inp)},{w})"
        case ZScore(input=inp, window=w, mean_type=mt):
            return f"zscore({render_series(inp)},{w}{',ema' if mt == 'ema' else ''})"
        case Percentile(input=inp, window=w):
            return f"percentile({render_series(inp)},{w})"
        case RollingAgg(input=inp, window=w, agg=agg):
            return f"rolling_agg({render_series(inp)},{'' if w is None else f'{w},'}{agg})"
        case Drawdown(input=inp, window=w):
            return f"drawdown({render_series(inp)}{'' if w is None else f',{w}'})"
        case Runup(input=inp, window=w):
            return f"runup({render_series(inp)}{'' if w is None else f',{w}'})"
        case BarsSinceExtremum(input=inp, extremum=ext, window=w):
            return f"bars_since_extremum({render_series(inp)},{ext}{'' if w is None else f',{w}'})"
        case Change(input=inp, periods=p, kind=k):
            return f"change({render_series(inp)},{p}{'' if k == 'pct' else f',{k}'})"
        case Shift(input=inp, periods=p):
            return f"shift({render_series(inp)},{p})"
        case RollingCorr(left=left, right=right, window=w):
            return f"rolling_corr({render_series(left)},{render_series(right)},{w})"
        case CrossRank(input=inp, min_valid=mv):
            return f"cross_rank({render_series(inp)}{'' if mv == 2 else f',{mv}'})"
        case CrossDemean(input=inp, min_valid=mv):
            return f"cross_demean({render_series(inp)}{'' if mv == 2 else f',{mv}'})"
        case CrossAgg(input=inp, agg=agg, min_valid=mv):
            return f"cross_agg({render_series(inp)},{agg}{'' if mv == 2 else f',{mv}'})"
        case BinaryOp(left=left, right=right, op=op):
            return f"({render_series(left)}{op}{render_series(right)})"
        case UnaryOp(input=inp, op=op):
            return f"{op}({render_series(inp)})"
        case _:
            raise TypeError(f"unknown series node: {node!r}")
