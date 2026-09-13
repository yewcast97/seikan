"""Deterministic presentation-only rendering of Series expressions and Conditions (never part
of the hash)."""

from __future__ import annotations

from seikan.dsl.conditions import (
    AndCondition,
    Condition,
    FirstTrueCondition,
    LagCondition,
    NotCondition,
    OrCondition,
    RollingCondition,
    ThresholdCondition,
)
from seikan.dsl.nodes import (
    EMA,
    BarsSinceEvent,
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
    EventAgg,
    EventValue,
    External,
    Field,
    Mask,
    Native,
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


def _decay(window: _NumericParam | None, alpha: _NumericParam | None) -> str:
    """The EW decay token: the window as before (``20``), or the alpha form as ``a=0.06``."""
    if window is not None:
        return str(window)
    return f"a={fmt_num(alpha)}" if alpha is not None else "?"


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
        case EMA(input=inp, window=w, alpha=a):
            return f"ema({render_series(inp)},{_decay(w, a)})"
        case ZScore(input=inp, window=w, alpha=a, mean_type=mt):
            return f"zscore({render_series(inp)},{_decay(w, a)}{',ema' if mt == 'ema' else ''})"
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
        case CrossRank(input=inp, min_valid=mv, where=wh, group=gr):
            return f"cross_rank({render_series(inp)}{_cross_suffix(mv, wh, gr)})"
        case CrossDemean(input=inp, min_valid=mv, where=wh, group=gr):
            return f"cross_demean({render_series(inp)}{_cross_suffix(mv, wh, gr)})"
        case CrossAgg(input=inp, agg=agg, min_valid=mv, where=wh, group=gr):
            return f"cross_agg({render_series(inp)},{agg}{_cross_suffix(mv, wh, gr)})"
        case BinaryOp(left=left, right=right, op=op):
            return f"({render_series(left)}{op}{render_series(right)})"
        case UnaryOp(input=inp, op=op):
            return f"{op}({render_series(inp)})"
        case Mask(condition=c):
            return f"mask({render_condition(c)})"
        case BarsSinceEvent(event=e):
            return f"bars_since_event({render_condition(e)})"
        case EventValue(event=e, input=inp):
            return f"event_value({render_condition(e)},{render_series(inp)})"
        case EventAgg(event=e, input=inp, agg=agg):
            return f"event_agg({render_condition(e)},{render_series(inp)},{agg})"
        case Native(name=name, expr=e):
            return f"native({name},{render_series(e)})"
        case _:
            raise TypeError(f"unknown series node: {node!r}")


def _cross_suffix(min_valid: int, where: Condition | None, group: Series | None) -> str:
    """The trailing tokens of a cross node: a non-default ``min_valid``, then the population
    selectors as ``where=<cond>`` / ``group=<series>`` — ``cross_rank(close)``,
    ``cross_rank(close,3,where=(elig>=1),group=sector)``."""
    parts = [] if min_valid == 2 else [str(min_valid)]
    if where is not None:
        parts.append(f"where={render_condition(where)}")
    if group is not None:
        parts.append(f"group={render_series(group)}")
    return "".join(f",{p}" for p in parts)


def render_condition(node: Condition) -> str:
    """Deterministic compact label for a scalar-param Condition — the spelling an embedded
    condition takes inside :func:`render_series` (``mask((close>shift(high,1)))``). A threshold
    renders parenthesized, combinators as ``and(a,b)`` / ``or(a,b)`` / ``not(x)``, ``rolling`` as
    ``rolling(agg,window[,min_count],x)``, ``first_true`` as ``first_true(x[,cooldown])`` (the
    cooldown only when non-zero) and ``lag`` as ``lag(x,periods)``. Presentation only."""
    match node:
        case ThresholdCondition(left=left, op=op, right=right):
            return f"({render_series(left)}{op}{render_series(right)})"
        case AndCondition(conditions=cs):
            return "and(" + ",".join(render_condition(c) for c in cs) + ")"
        case OrCondition(conditions=cs):
            return "or(" + ",".join(render_condition(c) for c in cs) + ")"
        case NotCondition(condition=c):
            return f"not({render_condition(c)})"
        case RollingCondition(window=w, agg=agg, min_count=mc, condition=c):
            return f"rolling({agg},{w}{'' if mc is None else f',{mc}'},{render_condition(c)})"
        case FirstTrueCondition(condition=c, cooldown=cd):
            return f"first_true({render_condition(c)}{'' if cd == 0 else f',{cd}'})"
        case LagCondition(condition=c, periods=p):
            return f"lag({render_condition(c)},{p})"
        case _:
            raise TypeError(f"unknown condition node: {node!r}")
