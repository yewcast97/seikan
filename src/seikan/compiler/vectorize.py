"""Vectorized signal builder.

Two responsibilities:

1. **Parameter grid** — ``iter_param_assignments`` walks a thesis's entry tree, finds every
   list-valued transform param, and yields one scalarized ``(combo, entry)`` per
   point of the Cartesian product (a transform-param grid). Each swept param gets a stable
   level name (``"{kind}_{param}"``, disambiguated by occurrence) used to label result columns.

2. **Signal arrays** — ``build_series``/``build_condition`` turn a (scalar-param) node into a
   (rows × targets) DataFrame. A Series carries ``(value, init)``; a Condition carries THREE
   channels ``(value, init, defined)``:

   - ``value`` — the mechanical boolean (NaN operand → comparison False).
   - ``init`` — the observer's warmup latch (once a column first produces a finite value it
     stays initialized), so e.g. ``not <warming threshold>`` does not fire during warmup.
   - ``defined`` — three-valued (Kleene) definedness: was the condition
     DECIDED on this bar, or is its truth unknown because an input was missing? A comparison
     is decided only where both operands are finite; ``and``/``or`` recover a verdict from a
     decisive child (F∧U = F, T∨U = T); ``not`` passes definedness through.

   Warmup is **vacuously defined** (see ``_vacuous``) — it is the observer's doctrinal False,
   not a data hole — so ``init & ~defined`` isolates exactly the post-warmup in-data holes.
   That mask is what the runner's ``signal_coverage`` ledger counts and the gate refuses:
   without it a missing decision input would silently become "no signal" (and ``not`` over a
   hole silently a FIRING), so deleting adverse inputs could improve a verdict unseen.

   A condition's tradable signal is ``value & init & defined`` (see ``signal``).
"""

from __future__ import annotations

import operator
from collections.abc import Iterator, Sequence
from itertools import product
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from seikan.compiler import nb
from seikan.compiler import transforms as xf
from seikan.compiler.data import MarketData
from seikan.constants import RESERVED_SWEEP_LEVELS
from seikan.dsl.schema import (
    EMA,
    AndCondition,
    BarsSinceExtremum,
    BinaryOp,
    Calendar,
    Change,
    Condition,
    Constant,
    CrossAgg,
    CrossDemean,
    CrossRank,
    DaysSince,
    Drawdown,
    External,
    Field,
    FirstTrueCondition,
    NotCondition,
    OrCondition,
    Percentile,
    RollingAgg,
    RollingCondition,
    RollingCorr,
    Runup,
    Series,
    Shift,
    ThresholdCondition,
    UnaryOp,
    ZScore,
)
from seikan.types import ParamValue

_OPS = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
    "==": np.equal,
    "!=": np.not_equal,
}
_ARITH = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


# ---- parameter grid --------------------------------------------------------

#: A transform param as the DSL writes it: the scalar every kernel takes, or the list that declares
#: it a sweep axis (``PosIntParam`` / ``FloatParam`` in ``dsl.schema`` — an int axis and a float
#: axis, never a mixed list). Both tree walks below hand exactly this to their resolver.
type _SweptParam = ParamValue | list[int] | list[float]


class _SweepCallback(Protocol):
    """What a resolver does when it meets a list: name the axis, hand back this pass's scalar.

    The two implementations are the two passes — ``collect_sweeps`` records the axis and takes the
    first level, ``iter_param_assignments`` looks the level up in the combo it is building.
    """

    def __call__(self, level: str, values: Sequence[ParamValue], /) -> ParamValue: ...


class _ParamResolver(Protocol):
    """The per-param callback the tree walks apply: scalar in → same scalar out, list in → the
    level chosen by this pass's :class:`_SweepCallback`. ``name`` labels a swept ``Constant``'s axis
    verbatim (see :func:`_make_resolver`).

    Kind-preserving by construction — an int param resolves to one of ITS levels and a float
    param to one of its own — which is what lets a resolved value go straight back into the node
    it came from (``EMA.window`` takes ints, ``Constant.value`` floats).
    """

    def __call__[T: (int, float)](
        self, kind: str, param: str, value: T | list[T], name: str | None = None
    ) -> T: ...


def _transform_series(node: Series, resolve: _ParamResolver) -> Series:
    match node:
        case Field() | External() | Calendar() | DaysSince():
            return node
        case Constant(value=val, name=nm):
            # The scalarized node DROPS the sweep-axis name: the axis label already did its one
            # job inside `resolve` (naming which combo level applies), execution never reads a
            # scalar constant's name, and the DSL refuses one — keeping it would also split the
            # evaluation memo into one slot per axis spelling of the same value.
            return Constant(value=resolve("constant", "value", val, name=nm))
        case EMA(input=inp, window=w):
            return EMA(input=_transform_series(inp, resolve), window=resolve("ema", "window", w))
        case ZScore(input=inp, window=w, mean_type=mt):
            return ZScore(
                input=_transform_series(inp, resolve),
                window=resolve("zscore", "window", w),
                mean_type=mt,
            )
        case Percentile(input=inp, window=w):
            return Percentile(
                input=_transform_series(inp, resolve),
                window=resolve("percentile", "window", w),
            )
        case RollingAgg(input=inp, window=w, agg=agg):
            if w is None:
                return RollingAgg(input=_transform_series(inp, resolve), window=None, agg=agg)
            return RollingAgg(
                input=_transform_series(inp, resolve),
                window=resolve("rolling_agg", "window", w),
                agg=agg,
            )
        case Drawdown(input=inp, window=w):
            inp_t = _transform_series(inp, resolve)
            if w is None:
                return Drawdown(input=inp_t, window=None)
            return Drawdown(input=inp_t, window=resolve("drawdown", "window", w))
        case Runup(input=inp, window=w):
            inp_t = _transform_series(inp, resolve)
            if w is None:
                return Runup(input=inp_t, window=None)
            return Runup(input=inp_t, window=resolve("runup", "window", w))
        case BarsSinceExtremum(input=inp, window=w, extremum=ext):
            inp_t = _transform_series(inp, resolve)
            if w is None:
                return BarsSinceExtremum(input=inp_t, window=None, extremum=ext)
            return BarsSinceExtremum(
                input=inp_t,
                window=resolve("bars_since_extremum", "window", w),
                extremum=ext,
            )
        case Change(input=inp, periods=p, kind=k):
            return Change(
                input=_transform_series(inp, resolve),
                periods=resolve("change", "periods", p),
                kind=k,
            )
        case Shift(input=inp, periods=p):
            return Shift(
                input=_transform_series(inp, resolve), periods=resolve("shift", "periods", p)
            )
        case UnaryOp(input=inp, op=op):
            return UnaryOp(input=_transform_series(inp, resolve), op=op)
        case RollingCorr(left=lhs, right=rhs, window=w):
            return RollingCorr(
                left=_transform_series(lhs, resolve),
                right=_transform_series(rhs, resolve),
                window=resolve("rolling_corr", "window", w),
            )
        case CrossRank(input=inp, min_valid=mv):
            # No swept params of its own (min_valid is a plain int); the inner input's sweeps
            # register through the recursive call.
            return CrossRank(input=_transform_series(inp, resolve), min_valid=mv)
        case CrossDemean(input=inp, min_valid=mv):
            return CrossDemean(input=_transform_series(inp, resolve), min_valid=mv)
        case CrossAgg(input=inp, agg=agg, min_valid=mv):
            return CrossAgg(input=_transform_series(inp, resolve), agg=agg, min_valid=mv)
        case BinaryOp(left=lhs, right=rhs, op=op):
            return BinaryOp(
                left=_transform_series(lhs, resolve), right=_transform_series(rhs, resolve), op=op
            )
        case _:
            raise TypeError(f"unknown series node: {node!r}")


def _transform_condition(node: Condition, resolve: _ParamResolver) -> Condition:
    match node:
        case ThresholdCondition(left=lhs, op=op, right=rhs):
            return ThresholdCondition(
                left=_transform_series(lhs, resolve), op=op, right=_transform_series(rhs, resolve)
            )
        case AndCondition(conditions=cs):
            return AndCondition(conditions=[_transform_condition(c, resolve) for c in cs])
        case OrCondition(conditions=cs):
            return OrCondition(conditions=[_transform_condition(c, resolve) for c in cs])
        case NotCondition(condition=c):
            return NotCondition(condition=_transform_condition(c, resolve))
        case RollingCondition(window=w, agg=a, min_count=mc, condition=inner):
            # Resolve the inner first so its nested sweeps register before this node's own
            # ``rolling_window`` axis (mirrors the transform convention — see EMA above).
            inner_t = _transform_condition(inner, resolve)
            return RollingCondition(
                window=resolve("rolling", "window", w), agg=a, min_count=mc, condition=inner_t
            )
        case FirstTrueCondition(condition=inner, cooldown=cd):
            inner_t = _transform_condition(inner, resolve)
            return FirstTrueCondition(
                condition=inner_t, cooldown=resolve("first_true", "cooldown", cd)
            )
        case _:
            raise TypeError(f"unknown condition node: {node!r}")


def _make_resolver(on_sweep: _SweepCallback) -> _ParamResolver:
    """Build a ``resolve(kind, param, value, name=None)`` that names each swept (list) param
    deterministically.

    Both the collection and the assignment passes walk the trees in the same order and detect lists
    identically (the original tree still holds the lists), so the generated level names line up. An
    explicit ``name`` (used by a swept ``Constant``) labels the axis verbatim, bypassing the
    ``{kind}_{param}`` auto-naming and its occurrence counter.
    """
    counts: dict[str, int] = {}

    def resolve(kind: str, param: str, value: _SweptParam, name: str | None = None) -> ParamValue:
        if isinstance(value, list):
            if name is not None:
                level = name
            else:
                base = f"{kind}_{param}"
                counts[base] = counts.get(base, 0) + 1
                level = base if counts[base] == 1 else f"{base}_{counts[base]}"
            return on_sweep(level, value)
        return value

    # ``resolve`` reads one param at a time, so nothing in its body ties the scalar it returns to
    # the scalar KIND it was handed — the levels arrive through a callback that spans every axis of
    # the grid at once. The cast states the property the two passes do hold (a level of an int
    # param is an int, a level of a float param a float), which is what keeps the resolved value
    # assignable to the node field it came from.
    return cast("_ParamResolver", resolve)


# Level names a swept constant must not take: structural names the runner adds/uses downstream
# (``"target"`` is the trades-frame target column; ``"horizon"`` is appended when the measurement
# horizon is swept). Collisions among sweep axes (constant↔constant or constant↔transform) corrupt
# the per-combo counting silently, so they are rejected here — the single point where every entry
# level name is known. (Trade-column / feature-name collisions are caught in the runner, which owns
# those names.)
_RESERVED_LEVELS = RESERVED_SWEEP_LEVELS


def collect_sweeps(entry: Condition) -> list[tuple[str, list[ParamValue]]]:
    """Ordered ``[(level_name, values)]`` for every list-valued param across the entry tree.

    Raises ``ValueError`` on a duplicate or reserved level name (a swept ``Constant`` whose ``name``
    collides with another sweep axis or a structural column would otherwise miscount silently).
    """
    sweeps: list[tuple[str, list[ParamValue]]] = []

    def record(level: str, values: Sequence[ParamValue]) -> ParamValue:
        # This pass only names the axes; the scalar it hands back (the first level) keeps the walk
        # type-correct and is discarded with the tree it builds.
        sweeps.append((level, list(values)))
        return values[0]

    resolve = _make_resolver(record)
    _transform_condition(entry, resolve)
    seen: set[str] = set()
    for lvl, _ in sweeps:
        if lvl in _RESERVED_LEVELS:
            raise ValueError(
                f"sweep axis name {lvl!r} is reserved; rename the swept constant's 'name'"
            )
        if lvl in seen:
            raise ValueError(
                f"duplicate sweep axis name {lvl!r}; each swept constant 'name' must be unique and "
                f"must not collide with a transform axis (e.g. 'ema_window')"
            )
        seen.add(lvl)
    return sweeps


def iter_param_assignments(entry: Condition) -> Iterator[tuple[dict[str, ParamValue], Condition]]:
    """Yield ``(combo, entry)`` for each point of the swept-param Cartesian product.

    ``combo`` maps level name → chosen scalar (empty when nothing is swept). ``entry`` is a fresh
    scalarized condition tree.
    """
    sweeps = collect_sweeps(entry)
    levels = [lvl for lvl, _ in sweeps]
    grids = [vals for _, vals in sweeps]

    def resolver_for(combo: dict[str, ParamValue]) -> _ParamResolver:
        # ``combo`` is a PARAMETER here, so each point of the product binds its own — the callback
        # can never read a later iteration's assignment.
        return _make_resolver(lambda level, _value: combo[level])

    for point in product(*grids) if grids else [()]:
        # One level per axis by construction: ``levels`` and ``grids`` are the two halves of the
        # same ``sweeps`` list, and ``product`` yields one value per grid.
        combo = dict(zip(levels, point, strict=True))
        yield combo, _transform_condition(entry, resolver_for(combo))


# ---- vectorized evaluation -------------------------------------------------


def _df(arr: np.ndarray, md: MarketData) -> pd.DataFrame:
    return pd.DataFrame(arr, index=md.index, columns=md.targets)


def _channels(
    value: np.ndarray, init: np.ndarray, defined: np.ndarray, md: MarketData
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Package a condition's three channels as frames, warmup made vacuously defined — the one
    return shape every composite condition ends in."""
    return _df(value, md), _df(init, md), _df(_vacuous(defined, init), md)


def _kleene_reduce(
    parts: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    *,
    conjunction: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Kleene three-valued reduction ``and``/``or`` share — ``conjunction`` names which dual.

    ``value``/``init`` reduce under the connective itself; ``defined`` is decided where every
    child is, OR where any child is decided at the connective's ABSORBING element (F for ∧, T
    for ∨) — one decided-False child settles a conjunction (F∧U = F), one decided-True child a
    disjunction (T∨U = T), whatever the unknown children would have said."""
    vals = [p[0].to_numpy() for p in parts]
    inits = [p[1].to_numpy() for p in parts]
    defs = [p[2].to_numpy() for p in parts]
    connective = np.logical_and if conjunction else np.logical_or
    value = connective.reduce(vals)
    init = connective.reduce(inits)
    absorbed = [d & ~v if conjunction else d & v for d, v in zip(defs, vals, strict=True)]
    defined = np.logical_and.reduce(defs) | np.logical_or.reduce(absorbed)
    return value, init, defined


def _rolling_full_window(frame: pd.DataFrame, window: int) -> np.ndarray:
    """True where EVERY bar of the trailing ``window`` is True — the rolling-all read the
    Rolling case applies to its inner condition's value, init and defined channels alike."""
    counts = frame.astype(int).rolling(window, min_periods=window).sum()
    result: np.ndarray = (counts == window).fillna(False).to_numpy().astype(bool)
    return result


def _scalarized(param: object) -> int:
    """States (and types) the precondition ``iter_param_assignments`` already established: every
    sweepable parameter reaching evaluation is a RESOLVED scalar — the same contract the Constant
    note in ``_build_series`` documents for values."""
    return int(cast("int", param))


def _latch(values: np.ndarray) -> npt.NDArray[np.bool_]:
    """Once a column first produces a finite value it stays 'initialized' (matches the observer).

    ``isfinite``, matching the docstring's word: the kernels sanitize their outputs to finite or
    NaN, so this is defense-in-depth — a ±inf that slipped through must read as warmup/hole, never
    latch the column initialized."""
    return np.maximum.accumulate(np.isfinite(values).astype(np.int8), axis=0).astype(bool)


def _all_true(md: MarketData) -> npt.NDArray[np.bool_]:
    return np.ones((len(md.index), len(md.targets)), dtype=bool)


def _vacuous(defined: npt.NDArray[np.bool_], init: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Warmup is vacuously DEFINED — it is the observer's doctrinal False, not a data hole.

    Applied at EVERY condition node so the invariant ``~init ⇒ defined`` holds throughout the
    tree. Two things depend on it: ``init & ~defined`` then isolates exactly the post-warmup
    in-data holes (the ledger's definition), and a still-warming ``or`` branch can never mark a
    live bar undefined — ``or``'s init is any-child, so without this an ordinary multi-branch
    thesis with staggered warmups would refuse on its own warmup.
    """
    return defined | ~init


def _calendar_values(md: MarketData, field: str) -> npt.NDArray[np.float64]:
    """Calendar attribute of each bar's timestamp as a float (rows × targets) array. Calendar-day
    arithmetic only — every field is knowable at the bar itself (no session look-ahead)."""
    idx = md.index
    if field == "month":
        col = idx.month
    elif field == "day_of_week":
        col = idx.dayofweek  # 0 = Monday .. 6 = Sunday
    elif field == "day_of_month":
        col = idx.day
    elif field == "days_to_month_end":
        col = idx.days_in_month - idx.day  # calendar days remaining; 0 = last calendar day
    else:
        raise ValueError(f"unknown calendar field: {field!r}")
    return np.repeat(np.asarray(col, dtype=float).reshape(-1, 1), len(md.targets), axis=1)


def build_series(node: Series, md: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(value, init)`` DataFrames (rows × targets) for a scalar-param Series node.

    Every build memoizes through ``md.series_memo`` (keyed by the node's canonical JSON), so a
    sweep grid that revisits the same sub-series across combos (e.g. a constant-threshold sweep
    that leaves the transforms unchanged) computes each only once — and, the memo riding the one
    materialized ``MarketData``, so do the backtest and listing paths of a single run. Returned
    frames — and the arrays derived from them via ``to_numpy`` — are treated as immutable by
    callers, which is what makes sharing them safe.
    """
    key = node.model_dump_json()
    hit = md.series_memo.get(key)
    if hit is not None:
        return hit
    out = _build_series(node, md)
    md.series_memo[key] = out
    return out


def _build_series(node: Series, md: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
    match node:
        # Data leaves latch like every transform: a LEADING NaN is warmup, a later one is an
        # in-data hole the condition layer must report as undefined. Initializing them
        # unconditionally would make a feed hole indistinguishable from a real value — and
        # `not <threshold over the hole>` would fire on missing data.
        case Field(column=col):
            arr = md.field(col).astype(float)
            return arr, _df(_latch(arr.to_numpy(dtype=float)), md)
        case Constant(value=val):
            # Scalar-param node (see the docstring): ``iter_param_assignments`` resolved every
            # swept list into one level before this tree was built, so the list form the DSL also
            # admits never reaches a kernel. The three casts in this file's evaluation half say
            # exactly that and nothing else.
            arr = np.full((len(md.index), len(md.targets)), float(cast("float", val)))
            return _df(arr, md), _df(_all_true(md), md)  # always finite
        case External(name=name):
            arr = md.external_values(name)
            return _df(arr, md), _df(_latch(arr), md)
        case Calendar(field=f):
            arr = _calendar_values(md, f)
            return _df(arr, md), _df(_all_true(md), md)
        case DaysSince(name=name):
            arr = md.days_since_values(name)
            return _df(arr, md), _df(_latch(arr), md)
        case CrossAgg():
            # The aggregate broadcasts a finite value into EVERY column wherever the
            # cross-section is thick enough — including a column whose own series has not yet
            # produced a value. Letting that column latch ``init`` off the broadcast would FIRE
            # a member before its own data begins (and classify every such firing
            # ``no_outcome``, hard-refusing the cell), so the warmup latch is gated by the
            # member's OWN input as well: a member sees the group's breadth (the sanctioned
            # broadcast VALUE) but cannot fire before its own series exists.
            cv = build_series(node.input, md)[0]
            x = cv.to_numpy(dtype=float)
            arr = xf.transform_values(node, x)
            return _df(arr, md), _df(_latch(arr) & _latch(x), md)
        case (
            EMA()
            | ZScore()
            | Percentile()
            | RollingAgg()
            | Drawdown()
            | Runup()
            | BarsSinceExtremum()
            | Change()
            | Shift()
            | CrossRank()
            | CrossDemean()
            | UnaryOp()
        ):
            cv = build_series(node.input, md)[0]
            arr = xf.transform_values(node, cv.to_numpy(dtype=float))
            return _df(arr, md), _df(_latch(arr), md)
        case RollingCorr(left=lhs, right=rhs, window=w):
            a = build_series(lhs, md)[0].to_numpy(dtype=float)
            b = build_series(rhs, md)[0].to_numpy(dtype=float)
            arr = nb.rolling_corr_apply_nb(a, b, cast("int", w))  # scalarized — see Constant above
            return _df(arr, md), _df(_latch(arr), md)
        case BinaryOp(left=lhs, right=rhs, op=op):
            a = build_series(lhs, md)[0].to_numpy(dtype=float)
            b = build_series(rhs, md)[0].to_numpy(dtype=float)
            with np.errstate(all="ignore"):
                arr = _ARITH[op](a, b)
            # EVERY op sanitizes, not just `/` (x/0 → inf, 0/0 → nan): finite operands can
            # overflow `+`/`-`/`*` to ±inf too, and an inf that survived here would latch warmup
            # and pass a threshold as a decided True — an overflow must never fire a signal.
            arr = np.where(np.isfinite(arr), arr, np.nan)
            return _df(arr, md), _df(_latch(arr), md)
        case _:
            raise TypeError(f"unknown series node: {node!r}")


def build_condition(
    node: Condition, md: MarketData
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(value, init, defined)`` boolean DataFrames (rows × targets) for a Condition.

    See the module docstring for what each channel means; ``init & ~defined`` is the
    post-warmup undefined-decision mask the ``signal_coverage`` ledger counts.

    Memoized through ``md.condition_memo``, the Condition twin of :func:`build_series`'s memo.
    """
    key = node.model_dump_json()
    hit = md.condition_memo.get(key)
    if hit is not None:
        return hit
    out = _build_condition(node, md)
    md.condition_memo[key] = out
    return out


def _build_condition(
    node: Condition, md: MarketData
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    match node:
        case ThresholdCondition(left=lhs, op=op, right=rhs):
            lv, li = build_series(lhs, md)
            rv, ri = build_series(rhs, md)
            a, b = lv.to_numpy(), rv.to_numpy()
            mask = np.isfinite(a) & np.isfinite(b)
            value = np.where(mask, _OPS[op](a, b), False)
            init = li.to_numpy() & ri.to_numpy()
            # The comparison is DECIDED only where both operands are finite — the one place
            # definedness is minted; every other node composes it.
            return _channels(value, init, mask, md)
        case AndCondition(conditions=cs):
            value, init, defined = _kleene_reduce(
                [build_condition(c, md) for c in cs], conjunction=True
            )
            return _channels(value, init, defined, md)
        case OrCondition(conditions=cs):
            value, init, defined = _kleene_reduce(
                [build_condition(c, md) for c in cs], conjunction=False
            )
            return _channels(value, init, defined, md)
        case NotCondition(condition=c):
            cv, ci, cd_ = build_condition(c, md)
            # Negation preserves definedness (¬U = U). The phantom fire it would otherwise
            # produce — `~False` over a data hole reading as True — dies at the tradable mask
            # in `signal`.
            return _df(~cv.to_numpy(), md), ci, cd_
        case RollingCondition(window=w, agg=a, min_count=mc, condition=inner):
            wi = _scalarized(w)  # resolved per combo — see the Constant note in `_build_series`
            cv, ci, cd_ = build_condition(inner, md)
            counts = cv.astype(int).rolling(wi, min_periods=wi).sum()
            if a == "all":
                raw = counts == wi
            elif a == "any":
                raw = counts > 0
            else:  # count — "at least min_count of the last window bars" (mc non-None: validator)
                raw = counts >= mc
            value = raw.fillna(False).to_numpy().astype(bool)
            # Initialized only where the inner condition was initialized across the WHOLE window
            # (rolling-all of its init) — so a rolling over a still-warming inner does not fire, and
            # `not(rolling(...))` stays gated during warmup. This subsumes a w-1 warmup floor.
            init = _rolling_full_window(ci, wi)
            # Conservative: the window's verdict counts as decided only when EVERY bar in it was.
            # A per-agg refinement is possible (an `any` is settled by one decided True), but the
            # over-approximation errs only toward undefined — fail-closed and monotone, so hiding
            # a bar can never manufacture definedness.
            defined = _rolling_full_window(cd_, wi)
            return _channels(value, init, defined, md)
        case FirstTrueCondition(condition=inner, cooldown=cd):
            cv, ci, cd_ = build_condition(inner, md)
            # Tradable-signal transitions: kernel sees child's value + init (warmup-safe latch)
            # + defined, and reports which output bars a hole could still have flipped.
            value, defined = nb.first_true_apply_nb(
                cv.to_numpy(dtype=bool),
                ci.to_numpy(dtype=bool),
                cd_.to_numpy(dtype=bool),
                _scalarized(cd),
            )
            init = ci.to_numpy()
            return _df(value, md), ci, _df(_vacuous(defined, init), md)
        case _:
            raise TypeError(f"unknown condition node: {node!r}")


def condition_arrays(
    node: Condition, md: MarketData
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Memoized numpy views of a condition's ``(value, init, defined)`` channels —
    ``md.array_memo``, beside the two frame-level memos.

    :func:`signal` and :func:`undefined_mask` are called back-to-back per combo on BOTH compute
    paths (the runner's measurement loop and ``api.list_entries``); without this layer each call
    re-ran ``.to_numpy()`` three times on the very frames the memo had already built. Arrays are
    immutable by the same contract as the cached frames (see :func:`build_series`).
    """
    key = node.model_dump_json()
    hit = md.array_memo.get(key)
    if hit is not None:
        return hit
    value, init, defined = build_condition(node, md)
    out = (value.to_numpy(), init.to_numpy(), defined.to_numpy())
    md.array_memo[key] = out
    return out


def signal(node: Condition, md: MarketData) -> pd.DataFrame:
    """Tradable boolean signal for a condition: ``value & init & defined`` (rows × targets).

    The ONE definition of a firing — the backtest runner and ``api.list_entries`` both read it,
    so entry listings are bit-identical to the measured mask by construction. ``defined`` is
    part of the conjunction: an undecided bar is not a firing, so ``not`` over a post-warmup
    data hole does not fire on missing data.
    """
    value, init, defined = condition_arrays(node, md)
    return _df(value & init & defined, md)


def undefined_mask(node: Condition, md: MarketData) -> npt.NDArray[np.bool_]:
    """Post-warmup undefined decision bars: ``init & ~defined`` (rows × targets).

    The complement of the tradable mask's honesty — where the thesis could not be evaluated
    because an input was missing, as opposed to evaluating to False. The runner counts these
    per gate pool into ``summary["signal_coverage"]``.
    """
    _value, init, defined = condition_arrays(node, md)
    return init & ~defined
