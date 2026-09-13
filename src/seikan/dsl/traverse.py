"""Tree walks over the node/condition vocabulary: sweep discovery and axis naming, nesting depth,
external/source/cross discovery, and the declared grid size.

Two vocabularies, one tree. A Condition reads Series operands at its thresholds, and a Series can
embed a Condition (the event-anchor family: ``mask``, ``bars_since_event``, ``event_value``,
``event_agg``), so the walks come in two layers:

* the DIRECT layer — :func:`_iter_child_series` (a Series node's own Series children) and
  :func:`_iter_child_conditions` (its own embedded Conditions), :func:`_iter_threshold_operands`
  (a Condition tree's own threshold operands, never crossing into an embedded Condition);
* the CROSSING layer — :func:`_iter_series_tree` (every Series node under a Series, crossing into
  embedded Conditions through their operands) and the public :func:`iter_condition_series` (every
  threshold operand under a Condition, embedded ones included, each yielded as a ROOT).

Every node is reached by exactly one path in each crossing walk (its parent's direct child, or
its embedding threshold), so nothing is yielded twice.
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
    EventValue,
    EventAgg,
)

#: The Series nodes that embed a Condition (the event-anchor family).
_EVENT_NODES = (BarsSinceEvent, EventValue, EventAgg)


# ---- the direct layer ---------------------------------------------------------------------


def _iter_child_series(node: Series) -> Iterator[Series]:
    """A Series node's DIRECT Series children (never its embedded Conditions' operands)."""
    if isinstance(node, _SERIES_INPUT_NODES):
        yield node.input
    elif isinstance(node, (BinaryOp, RollingCorr)):
        yield node.left
        yield node.right


def _iter_child_conditions(node: Series) -> Iterator[Condition]:
    """A Series node's DIRECT embedded Conditions (``mask`` and the event nodes' ``event``)."""
    if isinstance(node, Mask):
        yield node.condition
    elif isinstance(node, _EVENT_NODES):
        yield node.event


def _iter_threshold_operands(node: Condition) -> Iterator[Series]:
    """The threshold operands of a Condition tree — the condition-tree walk ONLY, never crossing
    into a Condition embedded in one of those operands."""
    match node:
        case ThresholdCondition(left=left, right=right):
            yield left
            yield right
        case AndCondition(conditions=conditions) | OrCondition(conditions=conditions):
            for child in conditions:
                yield from _iter_threshold_operands(child)
        case (
            NotCondition(condition=child)
            | RollingCondition(condition=child)
            | FirstTrueCondition(condition=child)
            | LagCondition(condition=child)
        ):
            yield from _iter_threshold_operands(child)


# ---- the crossing layer -------------------------------------------------------------------


def iter_embedded_conditions(node: Series) -> Iterator[Condition]:
    """Every Condition embedded anywhere in a Series subtree, pre-order — reached through the
    Series' own children only (never crossing into a Condition already yielded: what those
    embed is reached through :func:`iter_condition_series` on the yielded Condition)."""
    yield from _iter_child_conditions(node)
    for child in _iter_child_series(node):
        yield from iter_embedded_conditions(child)


def _iter_series_tree(node: Series) -> Iterator[Series]:
    """Every Series node under ``node`` (itself included), pre-order, CROSSING into embedded
    Conditions through their threshold operands — the walk the external/source/cross discovery
    runs over."""
    yield node
    for cond in _iter_child_conditions(node):
        for operand in _iter_threshold_operands(cond):
            yield from _iter_series_tree(operand)
    for child in _iter_child_series(node):
        yield from _iter_series_tree(child)


def iter_condition_series(node: Condition) -> Iterator[Series]:
    """Yield every threshold operand Series anywhere under a condition tree — the outer
    thresholds' operands AND the operands of every Condition embedded in them (pre-order: an
    outer operand, then its embedded conditions' operands). Each is a ROOT in its own right:
    depth-checked on its own, listed by ``--root-series-out``."""
    for operand in _iter_threshold_operands(node):
        yield operand
        for inner in iter_embedded_conditions(operand):
            yield from iter_condition_series(inner)


def iter_series_depth_roots(node: Series) -> Iterator[Series]:
    """``node`` itself plus every threshold operand of every Condition embedded under it — the
    roots the nesting-depth check measures for a ``params.features`` series (an embedded
    Condition is a decision, not a transform level, so its operands are checked as roots rather
    than added to the embedding node's depth)."""
    yield node
    for cond in iter_embedded_conditions(node):
        yield from iter_condition_series(cond)


# ---- discovery ----------------------------------------------------------------------------


def _series_external_names(node: Series) -> Iterator[str]:
    for s in _iter_series_tree(node):
        if isinstance(s, (External, DaysSince)):
            yield s.name


def series_cross_nodes(node: Series) -> Iterator[CrossRank | CrossDemean | CrossAgg]:
    """Yield every cross-sectional node (CrossRank/CrossDemean/CrossAgg) anywhere under a Series
    tree, embedded conditions included."""
    for s in _iter_series_tree(node):
        if isinstance(s, (CrossRank, CrossDemean, CrossAgg)):
            yield s


def series_source_leaves(node: Series) -> Iterator[tuple[str, str]]:
    """Yield ``(kind, name)`` for every RAW DATA leaf under a Series node, embedded conditions
    included. The three source kinds are the only raw leaves: an event node reads its input and
    its event condition's operands, never a source of its own."""
    for s in _iter_series_tree(node):
        if isinstance(s, Field):
            yield ("field", s.column)
        elif isinstance(s, External):
            yield ("external", s.name)
        elif isinstance(s, DaysSince):
            yield ("days_since", s.name)


# ---- sweeps -------------------------------------------------------------------------------

#: The Series-node params that may sweep (list-valued), beside ``Constant.value``.
_SWEPT_SERIES_ATTRS = ("window", "periods")
#: The Condition-node params that may sweep: ``rolling.window``, ``first_true.cooldown``,
#: ``lag.periods``.
_SWEPT_CONDITION_ATTRS = ("window", "cooldown", "periods")


def _iter_series_sweep_lengths(node: Series) -> Iterator[int]:
    """Yield the length of every list-valued (swept) param under a Series node — its own, its
    children's, and (delegated to the condition walk, so each counts exactly once) every swept
    param inside a Condition it embeds."""
    if isinstance(node, Constant) and isinstance(node.value, list):
        yield len(node.value)
    for attr in _SWEPT_SERIES_ATTRS:
        value = getattr(node, attr, None)
        if isinstance(value, list):
            yield len(value)
    for cond in _iter_child_conditions(node):
        yield from _iter_condition_sweep_lengths(cond)
    for child in _iter_child_series(node):
        yield from _iter_series_sweep_lengths(child)


def _iter_condition_sweep_lengths(node: Condition) -> Iterator[int]:
    """Yield the length of every swept param anywhere under a condition tree — the Series
    operands' (embedded conditions included) plus the conditions' own list-valued params
    (``RollingCondition.window``, ``FirstTrueCondition.cooldown``, ``LagCondition.periods``)."""
    for attr in _SWEPT_CONDITION_ATTRS:
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
            | LagCondition(condition=child)
        ):
            yield from _iter_condition_sweep_lengths(child)


def _series_has_sweep(node: Series) -> bool:
    """True if any param anywhere under a Series node is list-valued (a sweep) — inside an
    embedded Condition included, so a feature carrying a swept ``rolling.window`` in its event
    node is refused like one carrying a swept transform window."""
    return next(_iter_series_sweep_lengths(node), None) is not None


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
    # Mirrors compiler.vectorize._transform_series EXACTLY: recurse operands (an embedded
    # condition first, then input, or left then right) BEFORE the node's own param, so the shared
    # occurrence counter advances in engine order.
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
        case Mask(condition=c):
            _condition_axis_names(c, counts, out)
            lvl = None
        case BarsSinceEvent(event=e):
            _condition_axis_names(e, counts, out)
            lvl = None
        case EventValue(event=e, input=inp) | EventAgg(event=e, input=inp):
            # Embedded condition BEFORE the input — the engine constructs them in that order.
            _condition_axis_names(e, counts, out)
            _series_axis_names(inp, counts, out)
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
    # Mirrors compiler.vectorize._transform_condition: for Rolling/FirstTrue/Lag the INNER
    # condition resolves before the node's own window/cooldown/periods axis.
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
        case LagCondition(condition=inner, periods=p):
            _condition_axis_names(inner, counts, out)
            lvl = _sweep_axis_name("lag", "periods", p, counts)
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


# ---- depth --------------------------------------------------------------------------------


def _series_depth(node: Series) -> int:
    """Operator-nesting depth. Leaves are 0; ``BinaryOp``/``UnaryOp``/``Shift``/``Mask`` are
    transparent (arithmetic/plumbing — they pass through the max child depth); every other
    operator adds one level over its Series children (``bars_since_event`` has none, so it is
    exactly one; ``event_value``/``event_agg`` count one over their ``input``). An embedded
    Condition is invisible here by construction — it is a decision, not a transform level — and
    its operands are depth-checked as roots of their own (:func:`iter_condition_series`,
    :func:`iter_series_depth_roots`)."""
    child_max = max((_series_depth(c) for c in _iter_child_series(node)), default=0)
    if isinstance(node, (Field, Constant, External, Calendar, DaysSince)):
        return 0
    if isinstance(node, (BinaryOp, UnaryOp, Shift, Mask)):
        return child_max
    return 1 + child_max


# ---- condition-level discovery ------------------------------------------------------------


def iter_external_names(node: Condition) -> Iterator[str]:
    """Yield every external feed name referenced anywhere under a condition tree (embedded
    conditions included; a name may repeat)."""
    for series in _iter_threshold_operands(node):
        yield from _series_external_names(series)


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
    for series in _iter_threshold_operands(node):
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
    for series in _iter_threshold_operands(node):
        for cross in series_cross_nodes(series):
            key = cross.model_dump_json()
            if key not in seen:
                seen.add(key)
                yield cross
