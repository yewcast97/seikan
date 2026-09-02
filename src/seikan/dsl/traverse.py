"""Tree walks over the node/condition vocabulary: sweep discovery and axis naming, nesting depth,
external/source/cross discovery, and the declared grid size.
"""

from __future__ import annotations

from collections.abc import Iterator

from seikan.constants import (
    MAX_DECLARED_GRID,
)
from seikan.dsl.conditions import (
    AndCondition,
    Condition,
    FirstTrueCondition,
    NotCondition,
    OrCondition,
    RollingCondition,
    ThresholdCondition,
)
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

_SERIES_INPUT_NODES = (
    EMA,
    ZScore,
    Percentile,
    RollingAgg,
    Drawdown,
    Runup,
    BarsSinceExtremum,
    Change,
    Shift,
    CrossRank,
    CrossDemean,
    CrossAgg,
    UnaryOp,
)


def _iter_child_series(node: Series) -> Iterator[Series]:
    if isinstance(node, _SERIES_INPUT_NODES):
        yield node.input
    elif isinstance(node, (BinaryOp, RollingCorr)):
        yield node.left
        yield node.right


def _series_external_names(node: Series) -> Iterator[str]:
    if isinstance(node, (External, DaysSince)):
        yield node.name
    for child in _iter_child_series(node):
        yield from _series_external_names(child)


def _series_has_sweep(node: Series) -> bool:
    """True if any window/period param anywhere under a Series node is list-valued (a sweep)."""
    if isinstance(node, Constant) and isinstance(node.value, list):
        return True
    if any(isinstance(getattr(node, attr, None), list) for attr in ("window", "periods")):
        return True
    return any(_series_has_sweep(c) for c in _iter_child_series(node))


def series_cross_nodes(node: Series) -> Iterator[CrossRank | CrossDemean | CrossAgg]:
    """Yield every cross-sectional node (CrossRank/CrossDemean/CrossAgg) anywhere under a Series
    tree."""
    if isinstance(node, (CrossRank, CrossDemean, CrossAgg)):
        yield node
    for child in _iter_child_series(node):
        yield from series_cross_nodes(child)


def _iter_series_sweep_lengths(node: Series) -> Iterator[int]:
    """Yield the length of every list-valued (swept) param under a Series node."""
    if isinstance(node, Constant) and isinstance(node.value, list):
        yield len(node.value)
    for attr in ("window", "periods"):
        value = getattr(node, attr, None)
        if isinstance(value, list):
            yield len(value)
    for child in _iter_child_series(node):
        yield from _iter_series_sweep_lengths(child)


def _iter_condition_sweep_lengths(node: Condition) -> Iterator[int]:
    """Yield the length of every swept param anywhere under a condition tree — the Series
    operands plus the conditions' own list-valued params (``RollingCondition.window``,
    ``FirstTrueCondition.cooldown``)."""
    for attr in ("window", "cooldown"):
        value = getattr(node, attr, None)
        if isinstance(value, list):
            yield len(value)
    match node:
        case ThresholdCondition(left=left, right=right):
            yield from _iter_series_sweep_lengths(left)
            yield from _iter_series_sweep_lengths(right)
        case AndCondition(conditions=conditions) | OrCondition(conditions=conditions):
            for child in conditions:
                yield from _iter_condition_sweep_lengths(child)
        case (
            NotCondition(condition=child)
            | RollingCondition(condition=child)
            | FirstTrueCondition(condition=child)
        ):
            yield from _iter_condition_sweep_lengths(child)


def _sweep_axis_name(
    kind: str,
    param: str,
    value: _NumericParam | None,
    counts: dict[str, int],
    name: str | None = None,
) -> str | None:
    """The level name ``compiler.vectorize._make_resolver`` assigns a swept (list-valued) param,
    or ``None`` when the param is scalar. Kept bit-identical to that resolver (a parity test pins
    it) so this parse-time check and the engine name every sweep axis the same."""
    if not isinstance(value, list):
        return None
    if name is not None:
        return name
    base = f"{kind}_{param}"
    counts[base] = counts.get(base, 0) + 1
    return base if counts[base] == 1 else f"{base}_{counts[base]}"


def _series_axis_names(node: Series, counts: dict[str, int], out: list[str]) -> None:
    # Mirrors compiler.vectorize._transform_series EXACTLY: recurse operands (input, or left then
    # right) BEFORE the node's own param, so the shared occurrence counter advances in engine order.
    match node:
        case Field() | External() | Calendar() | DaysSince():
            return
        case Constant(value=val, name=nm):
            lvl = _sweep_axis_name("constant", "value", val, counts, name=nm)
        case EMA(input=inp, window=w):
            _series_axis_names(inp, counts, out)
            lvl = _sweep_axis_name("ema", "window", w, counts)
        case ZScore(input=inp, window=w):
            _series_axis_names(inp, counts, out)
            lvl = _sweep_axis_name("zscore", "window", w, counts)
        case Percentile(input=inp, window=w):
            _series_axis_names(inp, counts, out)
            lvl = _sweep_axis_name("percentile", "window", w, counts)
        case RollingAgg(input=inp, window=w):
            _series_axis_names(inp, counts, out)
            lvl = _sweep_axis_name("rolling_agg", "window", w, counts)
        case Drawdown(input=inp, window=w):
            _series_axis_names(inp, counts, out)
            lvl = _sweep_axis_name("drawdown", "window", w, counts)
        case Runup(input=inp, window=w):
            _series_axis_names(inp, counts, out)
            lvl = _sweep_axis_name("runup", "window", w, counts)
        case BarsSinceExtremum(input=inp, window=w):
            _series_axis_names(inp, counts, out)
            lvl = _sweep_axis_name("bars_since_extremum", "window", w, counts)
        case Change(input=inp, periods=p):
            _series_axis_names(inp, counts, out)
            lvl = _sweep_axis_name("change", "periods", p, counts)
        case Shift(input=inp, periods=p):
            _series_axis_names(inp, counts, out)
            lvl = _sweep_axis_name("shift", "periods", p, counts)
        case UnaryOp(input=inp):
            _series_axis_names(inp, counts, out)
            lvl = None
        case RollingCorr(left=lhs, right=rhs, window=w):
            _series_axis_names(lhs, counts, out)
            _series_axis_names(rhs, counts, out)
            lvl = _sweep_axis_name("rolling_corr", "window", w, counts)
        case CrossRank(input=inp) | CrossDemean(input=inp) | CrossAgg(input=inp):
            # No swept params of their own (``min_valid`` is a plain int); the input's sweeps
            # register through the recursion — dropping these cases would fall through to the
            # wildcard and silently skip every axis inside a cross input, breaking the
            # parse-time/engine parity the pin test enforces.
            _series_axis_names(inp, counts, out)
            lvl = None
        case BinaryOp(left=lhs, right=rhs):
            _series_axis_names(lhs, counts, out)
            _series_axis_names(rhs, counts, out)
            lvl = None
        case _:
            # The arms above happen to cover the whole ``Series`` union today, so a checker reads
            # this arm as dead — but the arm is what makes the traversal TOTAL, and staying total
            # is exactly the property a new node type would take away. Silenced narrowly, never
            # deleted: without it an unhandled node raises here instead of contributing no axis.
            lvl = None  # type: ignore[unreachable]
    if lvl is not None:
        out.append(lvl)


def _condition_axis_names(node: Condition, counts: dict[str, int], out: list[str]) -> None:
    # Mirrors compiler.vectorize._transform_condition: for Rolling/FirstTrue the INNER condition
    # resolves before the node's own window/cooldown axis.
    match node:
        case ThresholdCondition(left=lhs, right=rhs):
            _series_axis_names(lhs, counts, out)
            _series_axis_names(rhs, counts, out)
        case AndCondition(conditions=cs) | OrCondition(conditions=cs):
            for c in cs:
                _condition_axis_names(c, counts, out)
        case NotCondition(condition=c):
            _condition_axis_names(c, counts, out)
        case RollingCondition(window=w, condition=inner):
            _condition_axis_names(inner, counts, out)
            lvl = _sweep_axis_name("rolling", "window", w, counts)
            if lvl is not None:
                out.append(lvl)
        case FirstTrueCondition(condition=inner, cooldown=cd):
            _condition_axis_names(inner, counts, out)
            lvl = _sweep_axis_name("first_true", "cooldown", cd, counts)
            if lvl is not None:
                out.append(lvl)


def _iter_sweep_axis_names(entry: Condition) -> list[str]:
    """Ordered sweep-axis names for every list-valued param across the entry tree — the exact names
    ``compiler.vectorize.collect_sweeps`` produces (a parity test pins the equality). Lets the
    ``Thesis`` validator refuse a reserved/duplicate/column-colliding axis at PARSE time (exit 3)
    instead of the runner discovering it after a data load (exit 4)."""
    out: list[str] = []
    _condition_axis_names(entry, {}, out)
    return out


def declared_grid_size(entry: Condition, horizon: int | list[int]) -> int:
    """The DECLARED hypothesis count: the Cartesian product of every swept entry param times
    the number of horizons — the same quantity the runner records as
    ``n_hypotheses_attempted``, computed structurally so it is knowable BEFORE any data is read.
    Non-firing combos cannot shrink it."""
    size = len(horizon) if isinstance(horizon, list) else 1
    for length in _iter_condition_sweep_lengths(entry):
        size *= length
        if size > MAX_DECLARED_GRID:  # early exit — no need to multiply out a runaway grid
            return size
    return size


def _series_depth(node: Series) -> int:
    """Operator-nesting depth. Leaves are 0; ``BinaryOp``/``UnaryOp``/``Shift`` are transparent
    (arithmetic/plumbing — they pass through the max child depth); every other operator adds one
    level."""
    child_max = max((_series_depth(c) for c in _iter_child_series(node)), default=0)
    if isinstance(node, (Field, Constant, External, Calendar, DaysSince)):
        return 0
    if isinstance(node, (BinaryOp, UnaryOp, Shift)):
        return child_max
    return 1 + child_max


def iter_condition_series(node: Condition) -> Iterator[Series]:
    """Yield every operand Series referenced anywhere under a condition tree."""
    match node:
        case ThresholdCondition(left=left, right=right):
            yield left
            yield right
        case AndCondition(conditions=conditions) | OrCondition(conditions=conditions):
            for child in conditions:
                yield from iter_condition_series(child)
        case (
            NotCondition(condition=child)
            | RollingCondition(condition=child)
            | FirstTrueCondition(condition=child)
        ):
            yield from iter_condition_series(child)


def iter_external_names(node: Condition) -> Iterator[str]:
    """Yield every external feed name referenced anywhere under a condition tree."""
    for series in iter_condition_series(node):
        yield from _series_external_names(series)


def series_source_leaves(node: Series) -> Iterator[tuple[str, str]]:
    """Yield ``(kind, name)`` for every RAW DATA leaf under a Series node."""
    if isinstance(node, Field):
        yield ("field", node.column)
    elif isinstance(node, External):
        yield ("external", node.name)
    elif isinstance(node, DaysSince):
        yield ("days_since", node.name)
    for child in _iter_child_series(node):
        yield from series_source_leaves(child)


def iter_source_leaves(node: Condition) -> Iterator[tuple[str, str]]:
    """Deduplicated ``(kind, name)`` raw decision inputs a condition tree reads.

    These are the leaves whose AVAILABILITY the engine must account for: the
    three-valued ``defined`` channel reports whether the ROOT condition was decidable, which a
    decisive sibling can mask (Kleene ``F∧U = F``) and which a NaN-skipping recursive kernel
    (EMA, expanding aggregates) can launder into a finite value on a later bar. Availability is
    read at the SOURCE instead, where no operator can absorb it.

    ``Constant`` and ``Calendar`` are excluded — both are total by construction (a constant is
    finite by validation, a calendar attribute is a property of the index itself)."""
    seen: set[tuple[str, str]] = set()
    for series in iter_condition_series(node):
        for leaf in series_source_leaves(series):
            if leaf not in seen:
                seen.add(leaf)
                yield leaf


def iter_cross_series(node: Condition) -> Iterator[CrossRank | CrossDemean | CrossAgg]:
    """Deduplicated cross-sectional nodes (``CrossRank``/``CrossDemean``/``CrossAgg``) a
    condition tree reads, first-seen pre-order.

    The breadth companion of :func:`iter_source_leaves`: each cross kernel reduces over the
    per-bar FINITE member count ``k`` and then discards it, while member warmup legally thins
    the cross-section (a late start is warmup, not a hole) — so which members a cross read
    actually stood on is visible nowhere unless it is recorded. The runner walks these nodes
    per scalarized combo to emit ``summary["cross_breadth"]``, recomputing ``k`` bit-exactly
    off each node's memoized input frame. Deduplicated by canonical JSON: two identical nodes
    share one memoized frame and one breadth profile."""
    seen: set[str] = set()
    for series in iter_condition_series(node):
        for cross in series_cross_nodes(series):
            key = cross.model_dump_json()
            if key not in seen:
                seen.add(key)
                yield cross
