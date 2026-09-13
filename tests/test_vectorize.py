"""Vectorized builder parity + parameter-sweep tests.

``build_series``/``build_condition`` (value component) must reproduce an independent reference
oracle (``_ref_series``/``_ref_cond``) built on the frozen, vectorbt-free kernels.
``collect_sweeps``/``iter_param_assignments`` expand the param grid. There is no
technical-indicator family (no RSI/ROC/ATR/ADX/OBV/VWAP/Rank) and no dedicated ``CrossCondition``
state machine: ``Change``
spells the percent, log and diff readings of one series in a single node, and the crossover recipe
is ``first_true(threshold(fast, ">", slow))``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seikan.compiler import vectorize as vz
from seikan.compiler.data import MarketData
from seikan.dsl.schema import (
    EMA,
    AndCondition,
    BarsSinceEvent,
    BarsSinceExtremum,
    BinaryOp,
    Change,
    Constant,
    CrossAgg,
    CrossDemean,
    CrossRank,
    Drawdown,
    EventAgg,
    EventValue,
    External,
    Field,
    FirstTrueCondition,
    LagCondition,
    Mask,
    Native,
    NotCondition,
    OrCondition,
    Percentile,
    RollingAgg,
    RollingCondition,
    Runup,
    Shift,
    ThresholdCondition,
    UnaryOp,
    ZScore,
)

from . import _reference_nb as ref

_OPS = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
    "==": np.equal,
    "!=": np.not_equal,
}


@pytest.fixture()
def ohlcv() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    idx = pd.date_range("2020-01-01", periods=300, freq="1D")
    close = pd.Series(100 + rng.randn(len(idx)).cumsum(), index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.randint(1_000, 10_000, len(idx)).astype(float),
        },
        index=idx,
    )


def _md(df: pd.DataFrame, externals=(), natives=None) -> MarketData:
    """Single-target MarketData from a fixture DataFrame (``_ext_<name>`` columns become feeds;
    ``natives`` — ``{feed: Series on its own stamps}`` — become the retained native prints)."""

    def frame(col):
        return pd.DataFrame({"t": df[col]})

    ext = {name: df[f"_ext_{name}"] for name in externals}
    return MarketData(
        close=frame("close"),
        open=frame("open"),
        high=frame("high"),
        low=frame("low"),
        volume=frame("volume") if "volume" in df else None,
        externals=ext,
        targets=["t"],
        externals_native=dict(natives) if natives else None,
    )


def _c(df, name):
    return df[name].to_numpy(dtype=float)


# ---- independent reference oracle -------------------------------------------
#
# ``_ref_series``/``_ref_cond`` are frozen, independently-authored implementations (not delegating
# to ``compiler/nb.py``) that the vectorized builder's output must reproduce exactly. ``_ref_init``/
# ``_ref_cond_init`` mirror the builder's warmup-latch semantics (``vectorize._latch``:
# once a value first turns finite, "initialized" stays true) so ``first_true`` — which fires on a
# false→true transition of the child's TRADABLE (value & init) signal — can be checked the same way
# ``build_condition`` computes it.


def _ref_unary(arr: np.ndarray, op: str) -> np.ndarray:
    if op == "abs":
        out = np.abs(arr)
    elif op == "neg":
        out = -arr
    elif op == "sign":
        out = np.sign(arr)
    elif op == "log":
        ok = arr > 0.0
        out = np.where(ok, np.log(np.where(ok, arr, 1.0)), np.nan)
    elif op == "sqrt":
        ok = arr >= 0.0
        out = np.where(ok, np.sqrt(np.where(ok, arr, 0.0)), np.nan)
    else:
        raise ValueError(f"unknown unary op: {op!r}")
    return np.where(np.isfinite(out), out, np.nan)


def _ref_series(node, df) -> np.ndarray:
    t = node.type
    if t == "field":
        return _c(df, node.column)
    if t == "constant":
        return np.full(len(df), float(node.value))
    if t == "external":
        return _c(df, f"_ext_{node.name}")
    if t == "ema":
        src = _ref_series(node.input, df)
        if node.window is not None:
            return ref.ema(src, node.window)
        return ref.ema_alpha(src, node.alpha, _warmup_of(node.alpha))
    if t == "zscore":
        src = _ref_series(node.input, df)
        if node.mean_type == "sma":
            return ref.zscore_sma(src, node.window)
        if node.window is not None:
            return ref.zscore_ema(src, node.window)
        return ref.zscore_ema_alpha(src, node.alpha, _warmup_of(node.alpha))
    if t == "percentile":
        return ref.percentile(_ref_series(node.input, df), node.window)
    if t == "rolling_agg":
        src = _ref_series(node.input, df)
        if node.window is None:
            return ref.expanding(src, node.agg)
        if node.agg == "median":
            return ref.rolling_median(src, node.window)
        if node.agg == "mad":
            return ref.rolling_mad(src, node.window)
        return ref.rolling_agg(src, node.window, node.agg)
    if t == "drawdown":
        return ref.drawdown(_ref_series(node.input, df), node.window)
    if t == "runup":
        return ref.runup(_ref_series(node.input, df), node.window)
    if t == "bars_since_extremum":
        return ref.bars_since_extremum(_ref_series(node.input, df), node.extremum, node.window)
    if t == "change":
        return ref.change(_ref_series(node.input, df), node.periods, node.kind)
    if t == "shift":
        return ref.shift_ref(_ref_series(node.input, df), node.periods)
    if t == "unary_op":
        return _ref_unary(_ref_series(node.input, df), node.op)
    if t == "rolling_corr":
        return ref.rolling_corr_ref(
            _ref_series(node.left, df), _ref_series(node.right, df), node.window
        )
    if t == "binary_op":
        a, b = _ref_series(node.left, df), _ref_series(node.right, df)
        fn = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}[node.op]
        with np.errstate(all="ignore"):
            out = fn(a, b)
        return np.where(np.isfinite(out), out, np.nan)  # every op sanitizes, like production
    if t == "mask":
        return ref.mask_values(*_ref_cond_channels(node.condition, df))
    if t in _EVENT_NODES:
        v, i, d = _ref_cond_channels(node.event, df)
        s_idx, _known = ref.event_anchor(v & i & d, i, d)
        if t == "bars_since_event":
            return ref.bars_since_event(s_idx)
        if t == "event_value":
            return ref.event_value(_ref_series(node.input, df), s_idx)
        return ref.event_agg(v & i & d, i, d, _ref_series(node.input, df), node.agg)
    raise NotImplementedError(t)


_EVENT_NODES = ("bars_since_event", "event_value", "event_agg")


def _warmup_of(alpha: float) -> int:
    """The alpha form's warmup, spelled independently of ``nb.ema_params``."""
    import math

    return max(1, math.ceil(round(2.0 / alpha - 1.0, 9)))


def _ref_cond_channels(node, df) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _ref_cond(node, df), _ref_cond_init(node, df), _ref_cond_defined(node, df)


#: Only always-finite leaves are unconditionally initialized: data leaves —
#: ``field``/``external`` — latch on their first finite value like every transform, so a LEADING
#: NaN is warmup and a later one is an in-data hole.
_LEAF_ALWAYS_INIT = ("constant", "calendar")


def _ref_init(node, df) -> np.ndarray:
    """Latch-based init mask matching ``vectorize._latch``: always-finite leaves are always
    initialized; every other Series node is initialized once (and forever after) its value first
    turns finite. ``mask`` carries its condition's OWN latch; an event node additionally latches
    where its anchor became unknowable after warmup (a post-warmup hole in the event condition
    before the first event is a hole, not warmup)."""
    if node.type in _LEAF_ALWAYS_INIT:
        return np.ones(len(df), dtype=bool)
    if node.type == "mask":
        return _ref_cond_init(node.condition, df)
    values = _ref_series(node, df)
    latch = np.maximum.accumulate((~np.isnan(values)).astype(np.int8)).astype(bool)
    if node.type in _EVENT_NODES:
        v, i, d = _ref_cond_channels(node.event, df)
        _s, known = ref.event_anchor(v & i & d, i, d)
        latch |= np.maximum.accumulate((i & ~known).astype(np.int8)).astype(bool)
    return latch


def _ref_first_true(
    value: np.ndarray, init: np.ndarray, defined: np.ndarray, cooldown: int
) -> tuple[np.ndarray, np.ndarray]:
    """False→true transition of the child's tradable signal, warmup-safe (mirrors
    ``nb.first_true_1d``'s spec, independently authored): the first True bar after warmup does not
    fire (must have seen an initialized False first); ``cooldown`` suppresses re-fires. An
    undefined child bar resets the machine like warmup and taints ``max(1, cooldown)`` following
    bars. Returns ``(out, defined_out)``."""
    n = value.shape[0]
    out = np.zeros(n, dtype=bool)
    defined_out = np.ones(n, dtype=bool)
    prev = False
    seen_false = False
    cool_left = 0
    taint = 0
    for i in range(n):
        if not init[i]:
            prev = False
            seen_false = False
            cool_left = 0
            taint = 0
            continue
        if not defined[i]:
            prev = False
            seen_false = False
            cool_left = 0
            taint = max(1, cooldown)
            defined_out[i] = False
            continue
        if taint > 0:
            taint -= 1
            defined_out[i] = False
        cur = bool(value[i])
        if cool_left > 0:
            cool_left -= 1
            if not cur:
                seen_false = True
            prev = cur
            continue
        if cur and (not prev) and seen_false:
            out[i] = True
            cool_left = cooldown
        if not cur:
            seen_false = True
        prev = cur
    return out, defined_out


def _ref_cond(node, df) -> np.ndarray:
    t = node.type
    n = len(df)
    if t == "threshold":
        a, b = _ref_series(node.left, df), _ref_series(node.right, df)
        out = np.zeros(n, dtype=bool)
        mask = ~(np.isnan(a) | np.isnan(b))
        out[mask] = _OPS[node.op](a[mask], b[mask])
        return out
    if t == "and":
        return np.logical_and.reduce([_ref_cond(c, df) for c in node.conditions])
    if t == "or":
        return np.logical_or.reduce([_ref_cond(c, df) for c in node.conditions])
    if t == "not":
        return ~_ref_cond(node.condition, df)
    if t == "rolling":
        inner = _ref_cond(node.condition, df)
        return (
            ref.rolling_all(inner, node.window)
            if node.agg == "all"
            else ref.rolling_any(inner, node.window)
        )
    if t == "first_true":
        val = _ref_cond(node.condition, df)
        init = _ref_cond_init(node.condition, df)
        defined = _ref_cond_defined(node.condition, df)
        return _ref_first_true(val, init, defined, int(node.cooldown))[0]
    if t == "lag":
        return ref.lag_channels(*_ref_cond_channels(node.condition, df), int(node.periods))[0]
    raise NotImplementedError(t)


def _ref_cond_init(node, df) -> np.ndarray:
    t = node.type
    if t == "threshold":
        return _ref_init(node.left, df) & _ref_init(node.right, df)
    if t in ("and", "or"):
        return np.logical_and.reduce([_ref_cond_init(c, df) for c in node.conditions])
    if t == "not":
        return _ref_cond_init(node.condition, df)
    if t == "rolling":
        inner_init = _ref_cond_init(node.condition, df)
        return ref.rolling_all(inner_init, node.window)
    if t == "first_true":
        return _ref_cond_init(node.condition, df)
    if t == "lag":
        return ref.lag_channels(*_ref_cond_channels(node.condition, df), int(node.periods))[1]
    raise NotImplementedError(t)


def _ref_cond_defined(node, df) -> np.ndarray:
    """Kleene definedness, independently authored against ``vectorize``'s spec:
    a comparison is decided where both operands are finite, ``and``/``or`` recover a verdict
    from a decisive child (F∧U=F, T∨U=T), ``not`` passes through, ``rolling`` needs the whole
    window decided — and warmup is vacuously defined at every node."""
    t = node.type
    init = _ref_cond_init(node, df)
    if t == "threshold":
        a, b = _ref_series(node.left, df), _ref_series(node.right, df)
        out = ~(np.isnan(a) | np.isnan(b))
    elif t in ("and", "or"):
        vals = [_ref_cond(c, df) for c in node.conditions]
        defs = [_ref_cond_defined(c, df) for c in node.conditions]
        decisive = [d & (v if t == "or" else ~v) for d, v in zip(defs, vals, strict=True)]
        out = np.logical_and.reduce(defs) | np.logical_or.reduce(decisive)
    elif t == "not":
        out = _ref_cond_defined(node.condition, df)
    elif t == "rolling":
        out = ref.rolling_all(_ref_cond_defined(node.condition, df), node.window)
    elif t == "first_true":
        val = _ref_cond(node.condition, df)
        inner_init = _ref_cond_init(node.condition, df)
        inner_def = _ref_cond_defined(node.condition, df)
        out = _ref_first_true(val, inner_init, inner_def, int(node.cooldown))[1]
    elif t == "lag":
        out = ref.lag_channels(*_ref_cond_channels(node.condition, df), int(node.periods))[2]
    else:
        raise NotImplementedError(t)
    return out | ~init


def _series_val(node, df, externals=()):
    return vz.build_series(node, _md(df, externals))[0]["t"].to_numpy(dtype=float)


def _cond_val(node, df, externals=()):
    return vz.build_condition(node, _md(df, externals))[0]["t"].to_numpy()


def _assert_series(node, df, **kw):
    np.testing.assert_allclose(
        _series_val(node, df, **kw), _ref_series(node, df), rtol=1e-9, atol=1e-9, equal_nan=True
    )


def _assert_cond(node, df, **kw):
    np.testing.assert_array_equal(_cond_val(node, df, **kw), _ref_cond(node, df))


# ---- series parity ---------------------------------------------------------


def test_field_and_constant(ohlcv):
    _assert_series(Field(column="close"), ohlcv)
    _assert_series(Constant(value=42.0), ohlcv)


def test_ema_field_parity(ohlcv):
    _assert_series(EMA(input=Field(column="close"), window=10), ohlcv)
    _assert_series(EMA(input=Field(column="volume"), window=10), ohlcv)


def test_rolling_agg_trailing_and_expanding_parity(ohlcv):
    _assert_series(RollingAgg(input=Field(column="close"), window=15, agg="mean"), ohlcv)
    _assert_series(RollingAgg(input=Field(column="close"), window=15, agg="std"), ohlcv)
    _assert_series(RollingAgg(input=Field(column="close"), window=None, agg="max"), ohlcv)
    _assert_series(RollingAgg(input=Field(column="close"), window=None, agg="min"), ohlcv)


def test_drawdown_runup_bars_since_extremum_parity(ohlcv):
    _assert_series(Drawdown(window=20), ohlcv)
    _assert_series(Drawdown(window=None), ohlcv)
    _assert_series(Runup(window=20), ohlcv)
    _assert_series(Runup(window=None), ohlcv)
    _assert_series(BarsSinceExtremum(extremum="max", window=20), ohlcv)
    _assert_series(BarsSinceExtremum(extremum="min", window=None), ohlcv)


def test_change_pct_log_diff_parity(ohlcv):
    for kind in ("pct", "log", "diff"):
        _assert_series(Change(input=Field(column="close"), periods=5, kind=kind), ohlcv)


def test_drawdown_input_matches_close_reading_default(ohlcv):
    # Drawdown(input unset) must be bit-identical to the explicit close-reading form (input fills
    # to close) — the "default-input-close" short form, which Drawdown is the node that offers.
    default = _series_val(Drawdown(window=20), ohlcv)
    general = _series_val(Drawdown(window=20, input=Field(column="close")), ohlcv)
    np.testing.assert_array_equal(general, default)


# ---- transforms over an external feed --------------------------------------


@pytest.fixture()
def ohlcv_ext(ohlcv) -> pd.DataFrame:
    df = ohlcv.copy()
    df["_ext_rate"] = np.random.RandomState(11).randn(len(df)).cumsum()
    return df


@pytest.mark.parametrize("mt", ["sma", "ema"])
def test_zscore_external(ohlcv_ext, mt):
    _assert_series(
        ZScore(input=External(name="rate"), window=20, mean_type=mt), ohlcv_ext, externals={"rate"}
    )


def test_percentile_external(ohlcv_ext):
    _assert_series(
        Percentile(input=External(name="rate"), window=30), ohlcv_ext, externals={"rate"}
    )


def test_ema_external_parity(ohlcv_ext):
    _assert_series(EMA(input=External(name="rate"), window=15), ohlcv_ext, externals={"rate"})


def test_change_external_parity(ohlcv):
    df = ohlcv.copy()
    df["_ext_px"] = np.cumsum(np.abs(np.random.RandomState(7).randn(len(df)))) + 100.0
    _assert_series(Change(input=External(name="px"), periods=5, kind="pct"), df, externals={"px"})
    _assert_series(Change(input=External(name="px"), periods=1, kind="log"), df, externals={"px"})
    _assert_series(Change(input=External(name="px"), periods=3, kind="diff"), df, externals={"px"})


# ---- nested inputs / binary_op (bounded pipelining) ------------------------


def test_ema_over_change_parity(ohlcv):
    # EMA of change (depth 2): a transform applied to another transform's output.
    _assert_series(
        EMA(input=Change(input=Field(column="close"), periods=1, kind="pct"), window=10), ohlcv
    )


def test_transform_over_transform_parity(ohlcv):
    # ZScore of change: transform applied to another transform's output (depth 2) matches the
    # oracle.
    _assert_series(
        ZScore(
            input=Change(input=Field(column="close"), periods=10, kind="pct"),
            window=20,
            mean_type="sma",
        ),
        ohlcv,
    )


def test_binary_op_parity(ohlcv):
    _assert_series(BinaryOp(left=Field(column="close"), right=Constant(value=2.0), op="*"), ohlcv)
    _assert_series(BinaryOp(left=Field(column="high"), right=Field(column="low"), op="-"), ohlcv)
    _assert_series(
        BinaryOp(
            left=Change(input=Field(column="close"), periods=10, kind="pct"),
            right=Constant(value=100.0),
            op="/",
        ),
        ohlcv,
    )


def test_binary_op_division_by_zero_is_nan(ohlcv):
    df = ohlcv.copy()
    den = np.ones(len(df))
    den[:10] = 0.0
    df["_ext_den"] = den
    df["_ext_num"] = np.arange(1.0, len(df) + 1.0)
    node = BinaryOp(left=External(name="num"), right=External(name="den"), op="/")
    _assert_series(node, df, externals={"num", "den"})
    out = _series_val(node, df, externals={"num", "den"})
    assert np.isnan(out[:10]).all() and np.isfinite(out[10:]).all()


def test_unary_abs_of_change_parity(ohlcv):
    _assert_series(
        UnaryOp(input=Change(input=Field(column="close"), periods=1, kind="pct"), op="abs"), ohlcv
    )


# ---- condition parity ------------------------------------------------------


def test_threshold_external(ohlcv_ext):
    _assert_cond(
        ThresholdCondition(
            left=ZScore(input=External(name="rate"), window=20, mean_type="sma"),
            op="<",
            right=Constant(value=-1.0),
        ),
        ohlcv_ext,
        externals={"rate"},
    )


def test_first_true_cross_recipe_matches_reference(ohlcv_ext):
    # There is no dedicated CrossCondition state machine; the crossover recipe is
    # first_true(threshold(fast, ">", slow)) — fires only on the false→true transition, not every
    # bar the threshold holds.
    fast = ZScore(input=External(name="rate"), window=20, mean_type="sma")
    inner = ThresholdCondition(left=fast, op=">", right=Constant(value=0.0))
    _assert_cond(FirstTrueCondition(condition=inner), ohlcv_ext, externals={"rate"})


@pytest.mark.parametrize("cooldown", [0, 3])
def test_first_true_cooldown_matches_reference(ohlcv_ext, cooldown):
    fast = ZScore(input=External(name="rate"), window=20, mean_type="sma")
    inner = ThresholdCondition(left=fast, op="<", right=Constant(value=-0.5))
    _assert_cond(
        FirstTrueCondition(condition=inner, cooldown=cooldown), ohlcv_ext, externals={"rate"}
    )


def test_first_true_warmup_and_cooldown_semantics():
    # Direct unit-level check of the false→true latch's documented edge cases (independent of any
    # particular transform's warmup shape): a regime already true when the child warms up must not
    # phantom-fire, and a cooldown must suppress re-fires for exactly that many bars.
    from seikan.compiler import nb

    def _run(value, init, cooldown):
        defined = np.ones((value.shape[0], 1), dtype=bool)
        return nb.first_true_apply_nb(value.reshape(-1, 1), init.reshape(-1, 1), defined, cooldown)

    value = np.array([False, False, True, True, True])
    init = np.array([False, False, True, True, True])
    out, _ = _run(value, init, 0)
    assert not out.any()

    value2 = np.array([False, False, False, True, True, False, True])
    init2 = np.array([False, True, True, True, True, True, True])
    out2, _ = _run(value2, init2, 0)
    np.testing.assert_array_equal(out2.reshape(-1), [False, False, False, True, False, False, True])

    out3, _ = _run(value2, init2, 5)
    np.testing.assert_array_equal(
        out3.reshape(-1), [False, False, False, True, False, False, False]
    )


def test_and_or_not_rolling_mixed(ohlcv_ext):
    t1 = ThresholdCondition(
        left=ZScore(input=External(name="rate"), window=20, mean_type="sma"),
        op="<",
        right=Constant(value=0.0),
    )
    t2 = ThresholdCondition(left=Field(column="close"), op=">", right=Field(column="open"))
    kw = {"externals": {"rate"}}
    _assert_cond(AndCondition(conditions=[t1, t2]), ohlcv_ext, **kw)
    _assert_cond(OrCondition(conditions=[t1, t2]), ohlcv_ext, **kw)
    _assert_cond(NotCondition(condition=t1), ohlcv_ext, **kw)
    _assert_cond(RollingCondition(window=5, agg="all", condition=t1), ohlcv_ext, **kw)
    _assert_cond(RollingCondition(window=5, agg="any", condition=t2), ohlcv_ext, **kw)


# ---- per-md build memo (grid memoization) ----------------------------------


def test_build_memo_is_correctness_preserving(ohlcv):
    # A condition that references the same transform twice; the memo riding md must build the
    # sub-series once and return a result bit-identical to a full rebuild over a SECOND, freshly
    # constructed MarketData with a fresh memo (the memo only memoizes, never alters).
    ema = EMA(input=Field(column="close"), window=20)
    cond = AndCondition(
        conditions=[
            ThresholdCondition(left=ema, op=">", right=Field(column="open")),
            ThresholdCondition(left=ema, op=">", right=Field(column="low")),
        ]
    )
    md = _md(ohlcv)
    first = vz.signal(cond, md).to_numpy()
    fresh = vz.signal(cond, _md(ohlcv)).to_numpy()  # new md ⇒ new memo ⇒ full rebuild
    np.testing.assert_array_equal(first, fresh)
    # the shared EMA sub-series was memoized under its canonical node key (so the two thresholds
    # reuse it), and the root condition under its own
    assert ema.model_dump_json() in md.series_memo
    assert cond.model_dump_json() in md.condition_memo
    # a repeat build over the same md is a pure memo read
    np.testing.assert_array_equal(vz.signal(cond, md).to_numpy(), first)


# ---- multi-target: each column matches its own 1D reference ----------------


def test_multi_target_columns_independent():
    rng = np.random.RandomState(0)
    idx = pd.date_range("2020-01-01", periods=120, freq="1D")
    a = pd.Series(100 + rng.randn(120).cumsum(), index=idx)
    b = pd.Series(50 + rng.randn(120).cumsum(), index=idx)

    def frame(sa, sb):
        return pd.DataFrame({"A": sa, "B": sb})

    md = MarketData(
        close=frame(a, b),
        open=frame(a, b),
        high=frame(a * 1.01, b * 1.01),
        low=frame(a * 0.99, b * 0.99),
        volume=None,
        externals={},
        targets=["A", "B"],
    )
    node = Change(input=Field(column="close"), periods=10, kind="pct")
    val = vz.build_series(node, md)[0]
    np.testing.assert_allclose(
        val["A"].to_numpy(), ref.change(a.to_numpy(), 10, "pct"), rtol=1e-9, equal_nan=True
    )
    np.testing.assert_allclose(
        val["B"].to_numpy(), ref.change(b.to_numpy(), 10, "pct"), rtol=1e-9, equal_nan=True
    )


# ---- per-target external feeds (DataFrame externals) ------------------------


def _md2(externals) -> tuple[MarketData, pd.Series, pd.Series]:
    rng = np.random.RandomState(0)
    idx = pd.date_range("2020-01-01", periods=120, freq="1D")
    a = pd.Series(100 + rng.randn(120).cumsum(), index=idx)
    b = pd.Series(50 + rng.randn(120).cumsum(), index=idx)

    def frame(sa, sb):
        return pd.DataFrame({"A": sa, "B": sb})

    md = MarketData(
        close=frame(a, b),
        open=frame(a, b),
        high=frame(a * 1.01, b * 1.01),
        low=frame(a * 0.99, b * 0.99),
        volume=None,
        externals=externals,
        targets=["A", "B"],
    )
    return md, a, b


def test_per_target_external_keeps_columns():
    rng = np.random.RandomState(5)
    idx = pd.date_range("2020-01-01", periods=120, freq="1D")
    fa = pd.Series(rng.randn(120).cumsum(), index=idx)
    fb = pd.Series(rng.randn(120).cumsum() + 10, index=idx)
    md, _, _ = _md2({"iv": pd.DataFrame({"A": fa, "B": fb})})

    val = vz.build_series(External(name="iv"), md)[0]
    np.testing.assert_allclose(val["A"].to_numpy(), fa.to_numpy())
    np.testing.assert_allclose(val["B"].to_numpy(), fb.to_numpy())


def test_per_target_transform_matches_per_column_reference():
    rng = np.random.RandomState(6)
    idx = pd.date_range("2020-01-01", periods=120, freq="1D")
    fa = pd.Series(rng.randn(120).cumsum(), index=idx)
    fb = pd.Series(rng.randn(120).cumsum() + 10, index=idx)
    md, _, _ = _md2({"iv": pd.DataFrame({"A": fa, "B": fb})})

    val = vz.build_series(ZScore(input=External(name="iv"), window=20, mean_type="sma"), md)[0]
    np.testing.assert_allclose(
        val["A"].to_numpy(), ref.zscore_sma(fa.to_numpy(), 20), rtol=1e-9, equal_nan=True
    )
    np.testing.assert_allclose(
        val["B"].to_numpy(), ref.zscore_sma(fb.to_numpy(), 20), rtol=1e-9, equal_nan=True
    )


def test_shared_series_equivalent_to_replicated_dataframe():
    rng = np.random.RandomState(7)
    idx = pd.date_range("2020-01-01", periods=120, freq="1D")
    feed = pd.Series(rng.randn(120).cumsum(), index=idx)
    md_series, _, _ = _md2({"f": feed})
    md_frame, _, _ = _md2({"f": pd.DataFrame({"A": feed, "B": feed})})

    node = Percentile(input=External(name="f"), window=15)
    v1 = vz.build_series(node, md_series)[0]
    v2 = vz.build_series(node, md_frame)[0]
    pd.testing.assert_frame_equal(v1, v2)


# ---- cross-sectional transforms (CrossRank / CrossDemean / CrossAgg) --------
#
# These operate ACROSS targets at each bar (axis=1) — the pandas row-wise rank/mean is the oracle.


def _md3() -> MarketData:
    rng = np.random.RandomState(11)
    idx = pd.date_range("2020-01-01", periods=120, freq="1D")
    cols = {name: pd.Series(100 + rng.randn(120).cumsum(), index=idx) for name in ("A", "B", "C")}

    def frame(scale=1.0):
        return pd.DataFrame({n: s * scale for n, s in cols.items()})

    return MarketData(
        close=frame(),
        open=frame(),
        high=frame(1.01),
        low=frame(0.99),
        volume=None,
        externals={},
        targets=["A", "B", "C"],
    )


def _row_rank_ref(frame: pd.DataFrame, min_valid: int = 2) -> pd.DataFrame:
    k = frame.notna().sum(axis=1).to_numpy()
    frac = (frame.rank(axis=1, method="average") - 1).div(np.maximum(k - 1, 1), axis=0)
    ok = frame.notna().to_numpy() & (k >= max(min_valid, 2))[:, None]
    return frac.where(ok)


def test_cross_rank_matches_row_rank_reference():
    md = _md3()
    val = vz.build_series(CrossRank(input=Field(column="close")), md)[0]
    pd.testing.assert_frame_equal(val, _row_rank_ref(md.close), check_names=False)


def test_cross_demean_matches_row_mean_reference():
    md = _md3()
    val = vz.build_series(CrossDemean(input=Field(column="close")), md)[0]
    want = md.close.sub(md.close.mean(axis=1), axis=0)
    pd.testing.assert_frame_equal(val, want, check_names=False)


def test_cross_rank_of_warming_input_stays_gated():
    # zscore needs `window` bars per column; before that the cross-section has k < min_valid=3
    # finite columns, so the cross value is NaN and init stays unlatched — the signal cannot fire.
    md = _md3()
    node = CrossRank(
        input=ZScore(input=Field(column="close"), window=20, mean_type="sma"), min_valid=3
    )
    value, init = vz.build_series(node, md)
    assert value.iloc[:19].isna().all().all()
    assert not init.iloc[:19].any().any()
    assert init.iloc[19:].all().all()  # all three columns z-score-finite from bar 19 on

    sig = vz.signal(ThresholdCondition(left=node, op=">=", right=Constant(value=0.0)), md)
    assert not sig.iloc[:19].any().any()
    assert sig.iloc[19:].any().any()


def test_cross_rank_inner_sweep_registers_no_new_axis():
    entry = ThresholdCondition(
        left=CrossRank(input=Change(input=Field(column="close"), periods=[5, 10])),
        op="<=",
        right=Constant(value=0.34),
    )
    sweeps = vz.collect_sweeps(entry)
    assert [lvl for lvl, _ in sweeps] == [
        "change_periods"
    ]  # the cross node adds no axis of its own
    combos = list(vz.iter_param_assignments(entry))
    assert [c for c, _ in combos] == [{"change_periods": 5}, {"change_periods": 10}]
    for combo, scalar_entry in combos:
        assert scalar_entry.left.type == "cross_rank"
        assert scalar_entry.left.input.periods == combo["change_periods"]
        assert scalar_entry.left.min_valid == 2  # preserved through scalarization


def test_cross_agg_multi_target_breadth():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    close = pd.DataFrame(
        {"a": [1, 2, 3, 4], "b": [2, 1, 3, 5], "c": [3, 3, 3, 3]}, index=idx, dtype=float
    )
    md = MarketData(
        close=close,
        open=close,
        high=close,
        low=close,
        volume=None,
        externals={},
        targets=["a", "b", "c"],
    )
    node = CrossAgg(input=Field(column="close"), agg="mean", min_valid=2)
    got = vz.build_series(node, md)[0].to_numpy()
    want = close.mean(axis=1).to_numpy().reshape(-1, 1).repeat(3, axis=1)
    np.testing.assert_allclose(got, want)


def test_cross_agg_init_gates_on_the_members_own_input():
    # The staggered-listing seam, at the unit level. CrossAgg broadcasts a finite aggregate into
    # EVERY column wherever k >= min_valid — including a column whose own series has not yet
    # produced a value — and that broadcast is the SANCTIONED value semantics (breadth is a
    # property of the cross-section, deliberately visible to a still-listing member). But the
    # warmup latch must NOT ride it: `_latch(arr)` alone would initialize the late column at bar
    # 0 and let it FIRE before its own data begins, so init is `_latch(arr) & _latch(x)` — gated
    # by the member's OWN input. Value and init deliberately disagree over the leading stretch.
    K = 15
    rng = np.random.RandomState(13)
    idx = pd.date_range("2020-01-01", periods=60, freq="1D")
    close = pd.DataFrame(
        {name: 100 + rng.randn(60).cumsum() for name in ("A", "B", "C")}, index=idx
    )
    close.iloc[:K, close.columns.get_loc("C")] = np.nan  # C lists K bars late
    md = MarketData(
        close=close,
        open=close,
        high=close,
        low=close,
        volume=None,
        externals={},
        targets=["A", "B", "C"],
    )
    value, init = vz.build_series(
        CrossAgg(input=Field(column="close"), agg="mean", min_valid=2), md
    )
    # VALUE: C sees the group's aggregate from bar 0 — A and B keep k = 2 >= min_valid, and the
    # broadcast is over the FINITE members (C absent until it lists, present after).
    k_early = close.iloc[:K][["A", "B"]].mean(axis=1).to_numpy()
    k_late = close.iloc[K:].mean(axis=1).to_numpy()
    np.testing.assert_allclose(value.iloc[:K]["C"].to_numpy(), k_early)
    np.testing.assert_allclose(value.iloc[K:]["C"].to_numpy(), k_late)
    # INIT: C stays un-initialized until its OWN first finite row, then latches — while its
    # siblings, whose own series exist from bar 0, are initialized throughout.
    assert not init.iloc[:K]["C"].any()
    assert init.iloc[K:]["C"].all()
    assert init[["A", "B"]].all().all()


# ---- parameter sweeps ------------------------------------------------------


def test_collect_sweeps_orders_and_names():
    # All sweeps live in the entry tree (there is no exit); an AndCondition walks left-to-right.
    entry = AndCondition(
        conditions=[
            ThresholdCondition(
                left=EMA(input=Field(column="close"), window=[10, 20]),
                op=">",
                right=Constant(value=0.0),
            ),
            ThresholdCondition(
                left=Percentile(input=Field(column="close"), window=[14, 21, 28]),
                op=">",
                right=Constant(value=0.7),
            ),
        ]
    )
    sweeps = vz.collect_sweeps(entry)
    assert [lvl for lvl, _ in sweeps] == ["ema_window", "percentile_window"]
    assert [vals for _, vals in sweeps] == [[10, 20], [14, 21, 28]]


def test_iter_param_assignments_cartesian():
    entry = AndCondition(
        conditions=[
            ThresholdCondition(
                left=EMA(input=Field(column="close"), window=[10, 20]),
                op=">",
                right=Constant(value=0.0),
            ),
            ThresholdCondition(
                left=Percentile(input=Field(column="close"), window=[14, 21]),
                op=">",
                right=Constant(value=0.7),
            ),
        ]
    )
    combos = [combo for combo, _ in vz.iter_param_assignments(entry)]
    assert len(combos) == 4
    assert {(c["ema_window"], c["percentile_window"]) for c in combos} == {
        (10, 14),
        (10, 21),
        (20, 14),
        (20, 21),
    }
    # scalarized trees carry the chosen scalar
    _, e2 = next(vz.iter_param_assignments(entry))
    assert isinstance(e2.conditions[0].left.window, int)


def test_collect_sweeps_through_nested_input():
    # A swept window inside a nested input registers before the outer node's own param.
    entry = ThresholdCondition(
        left=ZScore(input=EMA(input=Field(column="close"), window=[10, 20]), window=[30, 60]),
        op=">",
        right=Constant(value=0.0),
    )
    assert [lvl for lvl, _ in vz.collect_sweeps(entry)] == ["ema_window", "zscore_window"]


def test_binary_op_child_sweeps_collected():
    entry = ThresholdCondition(
        left=BinaryOp(
            left=EMA(input=Field(column="close"), window=[5, 10]),
            right=Percentile(input=Field(column="close"), window=[14, 21]),
            op="-",
        ),
        op=">",
        right=Constant(value=0.0),
    )
    sweeps = vz.collect_sweeps(entry)
    assert [lvl for lvl, _ in sweeps] == ["ema_window", "percentile_window"]
    assert [vals for _, vals in sweeps] == [[5, 10], [14, 21]]


def test_no_sweep_yields_single_assignment():
    entry = ThresholdCondition(
        left=Percentile(input=Field(column="close"), window=14), op="<", right=Constant(value=0.3)
    )
    combos = list(vz.iter_param_assignments(entry))
    assert len(combos) == 1 and combos[0][0] == {}


def test_collect_sweeps_named_constant_axis():
    # A swept constant becomes its own axis labelled by `name`, ordered after entry-tree transforms.
    entry = ThresholdCondition(
        left=Percentile(input=Field(column="close"), window=[14, 21]),
        op=">",
        right=Constant(value=[0.55, 0.6, 0.65], name="percentile_thresh"),
    )
    sweeps = vz.collect_sweeps(entry)
    assert [lvl for lvl, _ in sweeps] == ["percentile_window", "percentile_thresh"]
    assert [vals for _, vals in sweeps] == [[14, 21], [0.55, 0.6, 0.65]]


def test_scalar_constant_is_not_a_sweep_axis():
    entry = ThresholdCondition(
        left=Percentile(input=Field(column="close"), window=[14, 21]),
        op=">",
        right=Constant(value=0.6),
    )
    assert [lvl for lvl, _ in vz.collect_sweeps(entry)] == ["percentile_window"]


def test_iter_param_assignments_includes_constant_axis():
    entry = ThresholdCondition(
        left=Percentile(input=Field(column="close"), window=[14, 21]),
        op=">",
        right=Constant(value=[0.55, 0.6, 0.65], name="percentile_thresh"),
    )
    combos = [combo for combo, _ in vz.iter_param_assignments(entry)]
    assert len(combos) == 6  # 2 windows × 3 thresholds
    assert {(c["percentile_window"], c["percentile_thresh"]) for c in combos} == {
        (w, t) for w in (14, 21) for t in (0.55, 0.6, 0.65)
    }
    # the scalarized tree carries the chosen float constant, never the list
    combo, e2 = next(vz.iter_param_assignments(entry))
    assert e2.right.value == combo["percentile_thresh"] and isinstance(e2.right.value, float)


def test_collect_sweeps_rejects_duplicate_axis_name():
    # A constant `name` colliding with a transform axis (or another constant) would miscount
    # silently.
    entry = ThresholdCondition(
        left=Percentile(input=Field(column="close"), window=[14, 21]),
        op=">",
        right=Constant(value=[1, 2], name="percentile_window"),
    )
    with pytest.raises(ValueError, match="duplicate sweep axis"):
        vz.collect_sweeps(entry)


@pytest.mark.parametrize("reserved", ["target", "horizon"])
def test_collect_sweeps_rejects_reserved_axis_name(reserved):
    entry = ThresholdCondition(
        left=Field(column="close"), op=">", right=Constant(value=[1, 2], name=reserved)
    )
    with pytest.raises(ValueError, match="reserved"):
        vz.collect_sweeps(entry)


def test_sweep_axis_names_match_collect_sweeps_order_exact():
    # The schema's parse-time _iter_sweep_axis_names must name every axis EXACTLY as the engine's
    # collect_sweeps does — same order, same occurrence-counter spelling — or the Thesis validator
    # would refuse a different set of names than the runner assigns. Pins the two walkers together.
    from seikan.dsl import schema as sc
    from seikan.dsl.traverse import _iter_sweep_axes, _iter_sweep_axis_names

    def c(col):
        return sc.Field(column=col)

    trees = [
        sc.ThresholdCondition(
            left=sc.Percentile(window=14, input=c("close")),
            op="<",
            right=sc.Constant(value=[0.3, 0.35], name="cut"),
        ),
        sc.ThresholdCondition(
            left=sc.EMA(input=sc.EMA(input=c("close"), window=[5, 10]), window=[20, 30]),
            op=">",
            right=sc.Constant(value=[1.0, 2.0], name="k"),
        ),
        sc.RollingCondition(
            window=[3, 5],
            agg="count",
            min_count=2,
            condition=sc.ThresholdCondition(
                left=sc.ZScore(window=[10, 20], input=c("close")),
                op=">",
                right=sc.Constant(value=1.0),
            ),
        ),
        sc.FirstTrueCondition(
            cooldown=[1, 2],
            condition=sc.ThresholdCondition(
                left=sc.Drawdown(input=c("close"), window=[10, 20]),
                op="<",
                right=sc.Constant(value=[-0.1, -0.2], name="depth"),
            ),
        ),
        sc.AndCondition(
            conditions=[
                sc.ThresholdCondition(
                    left=sc.RollingCorr(left=c("close"), right=c("open"), window=[5, 10]),
                    op=">",
                    right=sc.Constant(value=0.5),
                ),
                sc.ThresholdCondition(
                    left=sc.Change(input=c("close"), periods=[3, 5]),
                    op="<",
                    right=sc.Constant(value=[0.0, 0.1], name="chg"),
                ),
            ]
        ),
        sc.ThresholdCondition(
            left=sc.BinaryOp(
                left=sc.Shift(input=c("close"), periods=[1, 2]),
                op="-",
                right=sc.UnaryOp(input=sc.Percentile(window=[7, 14], input=c("high")), op="abs"),
            ),
            op=">",
            right=sc.Constant(value=0.0),
        ),
        sc.ThresholdCondition(
            left=sc.RollingAgg(
                input=sc.Change(input=c("close"), periods=[2, 4]), window=None, agg="max"
            ),
            op=">",
            right=sc.Constant(value=[1.0, 2.0], name="lvl"),
        ),
        sc.OrCondition(
            conditions=[
                sc.ThresholdCondition(
                    left=sc.Runup(input=c("close"), window=[3, 6]),
                    op=">",
                    right=sc.Constant(value=0.05),
                ),
                sc.NotCondition(
                    condition=sc.ThresholdCondition(
                        left=sc.BarsSinceExtremum(
                            input=c("close"), window=[10, 20], extremum="max"
                        ),
                        op=">=",
                        right=sc.Constant(value=[5, 10], name="n"),
                    )
                ),
            ]
        ),
        # Cross nodes recurse their input without an axis of their own — the exact case the
        # schema walker would silently skip if it fell through to the wildcard.
        sc.ThresholdCondition(
            left=sc.CrossRank(input=sc.Change(input=c("close"), periods=[5, 10]), min_valid=3),
            op=">=",
            right=sc.Constant(value=[0.6, 0.8], name="q"),
        ),
        sc.ThresholdCondition(
            left=sc.BinaryOp(
                left=sc.CrossDemean(input=sc.Change(input=c("close"), periods=[3, 5])),
                op="/",
                right=sc.CrossAgg(input=sc.Change(input=c("close"), periods=[3, 5]), agg="std"),
            ),
            op=">",
            right=sc.Constant(value=1.0),
        ),
        # The event-anchor family: the embedded condition's sweeps register BEFORE the input's,
        # and a lag's periods after its inner condition — in engine order.
        sc.ThresholdCondition(
            left=sc.EventValue(
                event=sc.FirstTrueCondition(
                    cooldown=[2, 4],
                    condition=sc.ThresholdCondition(
                        left=sc.Percentile(window=[10, 20], input=c("close")),
                        op=">",
                        right=sc.Constant(value=[0.8, 0.9], name="hi"),
                    ),
                ),
                input=sc.Shift(input=c("high"), periods=[1, 2]),
            ),
            op=">",
            right=c("close"),
        ),
        sc.AndCondition(
            conditions=[
                sc.ThresholdCondition(
                    left=sc.BarsSinceEvent(
                        event=sc.LagCondition(
                            periods=[1, 3],
                            condition=sc.ThresholdCondition(
                                left=sc.EMA(input=c("close"), window=[5, 8]),
                                op=">",
                                right=c("close"),
                            ),
                        )
                    ),
                    op=">=",
                    right=sc.Constant(value=[1.0, 2.0], name="age"),
                ),
                sc.ThresholdCondition(
                    left=sc.EventAgg(
                        event=sc.RollingCondition(
                            window=[3, 4],
                            agg="any",
                            condition=sc.ThresholdCondition(
                                left=sc.Mask(
                                    condition=sc.ThresholdCondition(
                                        left=sc.Change(input=c("close"), periods=[2, 3]),
                                        op="<",
                                        right=sc.Constant(value=0.0),
                                    )
                                ),
                                op=">",
                                right=sc.Constant(value=0.5),
                            ),
                        ),
                        input=sc.ZScore(input=c("close"), window=[10, 15]),
                        agg="max",
                    ),
                    op=">",
                    right=sc.Constant(value=[1.0, 1.5], name="zmax"),
                ),
            ]
        ),
        # ema_alpha / zscore_alpha are their own axes; a window-swept EMA beside an alpha-swept
        # one keeps two independent occurrence counters.
        sc.ThresholdCondition(
            left=sc.BinaryOp(
                left=sc.EMA(input=c("close"), alpha=[0.05, 0.1]),
                op="-",
                right=sc.EMA(input=sc.EMA(input=c("close"), window=[5, 10]), alpha=[0.2, 0.3]),
            ),
            op=">",
            right=sc.ZScore(input=c("close"), alpha=[0.06, 0.1], mean_type="ema"),
        ),
        # A cross node's sweeps register input → where → group, in that order.
        sc.ThresholdCondition(
            left=sc.CrossRank(
                input=sc.Change(input=c("close"), periods=[5, 10]),
                where=sc.ThresholdCondition(
                    left=sc.RollingAgg(input=c("volume"), window=[20, 40], agg="mean"),
                    op=">",
                    right=sc.Constant(value=[1.0, 2.0], name="liq"),
                ),
                group=sc.Shift(input=sc.External(name="sector"), periods=[1, 2]),
            ),
            op=">=",
            right=sc.Constant(value=[0.6, 0.8], name="q"),
        ),
        # A native-clock expr: its sweeps register through the recursion like any input's.
        sc.ThresholdCondition(
            left=sc.Native(
                name="eps",
                expr=sc.BinaryOp(
                    left=sc.Change(input=sc.External(name="eps"), periods=[4, 8], kind="diff"),
                    op="/",
                    right=sc.RollingAgg(
                        input=sc.Change(input=sc.External(name="eps"), periods=[4, 8], kind="diff"),
                        window=[8, 12],
                        agg="std",
                    ),
                ),
            ),
            op=">",
            right=sc.Constant(value=[1.0, 2.0], name="sue"),
        ),
    ]
    for entry in trees:
        assert _iter_sweep_axis_names(entry) == [lvl for lvl, _ in vz.collect_sweeps(entry)]
        assert _iter_sweep_axes(entry) == vz.collect_sweeps(entry)
    # ... and with SHARED axes: the full (level, values) lists, dedupe rule included.
    N = sc.AxisRef(axis="N")
    K = sc.AxisRef(axis="K")
    axes = {"N": [20, 60], "K": [0, 3]}
    dc = sc.Change(input=c("close"), kind="diff")
    dv = sc.Change(input=c("volume"), kind="diff")
    shared_trees = [
        # the rolling beta: three sites, one axis
        sc.ThresholdCondition(
            left=sc.BinaryOp(
                left=sc.BinaryOp(
                    left=sc.RollingCorr(left=dc, right=dv, window=N),
                    op="*",
                    right=sc.RollingAgg(input=dc, window=N, agg="std"),
                ),
                op="/",
                right=sc.RollingAgg(input=dv, window=N, agg="std"),
            ),
            op=">",
            right=sc.Constant(value=0.5),
        ),
        # a ref beside two list-swept emas: the ref keeps its name, the counter is untouched
        sc.ThresholdCondition(
            left=sc.EMA(input=sc.EMA(input=c("close"), window=[5, 10]), window=N),
            op=">",
            right=sc.EMA(input=c("close"), window=[3, 4]),
        ),
        # refs across rolling.window / first_true.cooldown / a constant in two `and` branches
        sc.AndCondition(
            conditions=[
                sc.RollingCondition(
                    window=N,
                    agg="any",
                    condition=sc.FirstTrueCondition(
                        cooldown=K,
                        condition=sc.ThresholdCondition(
                            left=c("close"), op=">", right=sc.Constant(value=N)
                        ),
                    ),
                ),
                sc.ThresholdCondition(
                    left=sc.Percentile(input=c("close"), window=N),
                    op="<",
                    right=sc.Constant(value=N),
                ),
            ]
        ),
    ]
    for entry in shared_trees:
        assert _iter_sweep_axes(entry, axes) == vz.collect_sweeps(entry, axes)


# ---- rolling-condition window sweep ----------------------------------------


def test_collect_sweeps_rolling_window_axis():
    # A list `window` on a rolling condition becomes its own auto-named `rolling_window` axis.
    inner = ThresholdCondition(
        left=Percentile(input=Field(column="close"), window=14), op=">", right=Constant(value=0.5)
    )
    sweeps = vz.collect_sweeps(RollingCondition(window=[3, 5], agg="all", condition=inner))
    assert [lvl for lvl, _ in sweeps] == ["rolling_window"]
    assert [vals for _, vals in sweeps] == [[3, 5]]


def test_collect_sweeps_rolling_window_orders_after_inner_sweep():
    # The inner condition's swept transform registers before the rolling's own window (inner-first).
    inner = ThresholdCondition(
        left=Percentile(input=Field(column="close"), window=[14, 21]),
        op=">",
        right=Constant(value=0.5),
    )
    entry = RollingCondition(window=[3, 5], agg="all", condition=inner)
    assert [lvl for lvl, _ in vz.collect_sweeps(entry)] == ["percentile_window", "rolling_window"]


def test_collect_sweeps_two_rolling_windows_use_occurrence_counter():
    # Two swept rolling windows disambiguate via the occurrence counter, like repeated transforms.
    c1 = ThresholdCondition(left=Field(column="close"), op=">", right=Field(column="open"))
    c2 = ThresholdCondition(left=Field(column="close"), op="<", right=Field(column="high"))
    entry = AndCondition(
        conditions=[
            RollingCondition(window=[3, 5], agg="all", condition=c1),
            RollingCondition(window=[2, 4], agg="any", condition=c2),
        ]
    )
    assert [lvl for lvl, _ in vz.collect_sweeps(entry)] == ["rolling_window", "rolling_window_2"]


def test_iter_param_assignments_rolling_window_scalarized():
    inner = ThresholdCondition(
        left=Percentile(input=Field(column="close"), window=[14, 21]),
        op=">",
        right=Constant(value=0.5),
    )
    entry = RollingCondition(window=[3, 5], agg="all", condition=inner)
    combos = [combo for combo, _ in vz.iter_param_assignments(entry)]
    assert len(combos) == 4  # 2 percentile windows × 2 rolling windows
    assert {(c["percentile_window"], c["rolling_window"]) for c in combos} == {
        (w, rw) for w in (14, 21) for rw in (3, 5)
    }
    # the scalarized tree carries the chosen scalar window, never the list
    combo, e2 = next(vz.iter_param_assignments(entry))
    assert e2.window == combo["rolling_window"] and isinstance(e2.window, int)


def test_rolling_window_sweep_parity(ohlcv_ext):
    # Each scalarized rolling-window value reproduces the reference oracle for that scalar window.
    t1 = ThresholdCondition(
        left=ZScore(input=External(name="rate"), window=20, mean_type="sma"),
        op="<",
        right=Constant(value=0.0),
    )
    entry = RollingCondition(window=[3, 5, 8], agg="all", condition=t1)
    for combo, scalarized in vz.iter_param_assignments(entry):
        assert scalarized.window == combo["rolling_window"]
        _assert_cond(scalarized, ohlcv_ext, externals={"rate"})


def test_collect_sweeps_change_periods_axis(ohlcv):
    entry = ThresholdCondition(
        left=Change(input=Field(column="close"), periods=[1, 5], kind="pct"),
        op=">",
        right=Constant(value=0.0),
    )
    sweeps = vz.collect_sweeps(entry)
    assert sweeps == [("change_periods", [1, 5])]
    combos = [c for c, _ in vz.iter_param_assignments(entry)]
    assert combos == [{"change_periods": 1}, {"change_periods": 5}]


def test_collect_sweeps_percentile_input_child(ohlcv_ext):
    # sweeps inside a Percentile's input register through the recursive walk
    entry = ThresholdCondition(
        left=Percentile(window=[10, 14], input=Change(input=External(name="rate"), periods=[1, 2])),
        op="<",
        right=Constant(value=0.3),
    )
    sweeps = dict(vz.collect_sweeps(entry))
    assert sweeps == {"change_periods": [1, 2], "percentile_window": [10, 14]}


def test_collect_sweeps_shift_periods_axis(ohlcv):
    from seikan.dsl.schema import Shift

    entry = ThresholdCondition(
        left=Shift(input=Field(column="close"), periods=[1, 5]),
        op=">",
        right=Constant(value=0.0),
    )
    sweeps = vz.collect_sweeps(entry)
    assert sweeps == [("shift_periods", [1, 5])]
    combos = [c for c, _ in vz.iter_param_assignments(entry)]
    assert combos == [{"shift_periods": 1}, {"shift_periods": 5}]


# ---- external feed validation ----------------------------------------------


def test_undeclared_external_raises(ohlcv):
    with pytest.raises(ValueError, match="external feed"):
        vz.build_series(External(name="missing"), _md(ohlcv))
    with pytest.raises(ValueError, match="external feed"):
        vz.build_series(ZScore(input=External(name="missing"), window=5), _md(ohlcv))


# ---- rolling_corr (trailing-window Pearson correlation of two series) ------


def test_rolling_corr_matches_reference_of_two_fields(ohlcv):
    from seikan.dsl.schema import RollingCorr

    node = RollingCorr(left=Field(column="close"), right=Field(column="volume"), window=20)
    got = _series_val(node, ohlcv)
    want = ref.rolling_corr_ref(_c(ohlcv, "close"), _c(ohlcv, "volume"), 20)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_rolling_corr_nan_gates_on_warmup(ohlcv_ext):
    from seikan.dsl.schema import RollingCorr

    node = RollingCorr(left=Field(column="close"), right=External(name="rate"), window=15)
    value, init = vz.build_series(node, _md(ohlcv_ext, {"rate"}))
    assert value["t"].iloc[:14].isna().all()
    assert not init["t"].iloc[:14].any()
    assert init["t"].iloc[14:].all()
    got = value["t"].to_numpy(dtype=float)
    want = ref.rolling_corr_ref(_c(ohlcv_ext, "close"), _c(ohlcv_ext, "_ext_rate"), 15)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_rolling_corr_sweep_registers_rolling_corr_window():
    from seikan.dsl.schema import RollingCorr

    entry = ThresholdCondition(
        left=RollingCorr(left=Field(column="close"), right=Field(column="volume"), window=[10, 20]),
        op=">",
        right=Constant(value=0.0),
    )
    sweeps = vz.collect_sweeps(entry)
    assert sweeps == [("rolling_corr_window", [10, 20])]
    combos = [c for c, _ in vz.iter_param_assignments(entry)]
    assert combos == [{"rolling_corr_window": 10}, {"rolling_corr_window": 20}]


# ---- Phase-1 algebra nodes through the builder ------------------------------


def test_shift_parity(ohlcv):
    from seikan.dsl.schema import Shift

    _assert_series(Shift(input=Field(column="close"), periods=5), ohlcv)


def test_ema_percentile_change_warmup_gates(ohlcv):
    # NaN-gated warmup boundary for the surviving/new transform family, checked via the `init` mask.
    _ema_val, ema_init = vz.build_series(EMA(input=Field(column="close"), window=10), _md(ohlcv))
    assert not ema_init["t"].iloc[:9].any() and ema_init["t"].iloc[9:].all()

    _pct_val, pct_init = vz.build_series(
        Percentile(input=Field(column="close"), window=20), _md(ohlcv)
    )
    assert not pct_init["t"].iloc[:19].any() and pct_init["t"].iloc[19:].all()

    _chg_val, chg_init = vz.build_series(
        Change(input=Field(column="close"), periods=5, kind="pct"), _md(ohlcv)
    )
    assert not chg_init["t"].iloc[:5].any() and chg_init["t"].iloc[5:].all()


def test_calendar_values(ohlcv):
    from seikan.dsl.schema import Calendar

    idx = ohlcv.index
    cases = {
        "month": idx.month,
        "day_of_week": idx.dayofweek,
        "day_of_month": idx.day,
        "days_to_month_end": idx.days_in_month - idx.day,
    }
    for field, want in cases.items():
        got = _series_val(Calendar(field=field), ohlcv)
        np.testing.assert_allclose(got, np.asarray(want, dtype=float))
    # calendar is always-initialized: a threshold on it fires from bar 0
    cond = ThresholdCondition(
        left=Calendar(field="day_of_month"), op=">=", right=Constant(value=1.0)
    )
    assert vz.signal(cond, _md(ohlcv)).to_numpy().all()


def test_days_since_through_load_market_data(tmp_path):
    from seikan.compiler.data import DataFiles, load_market_data
    from seikan.dsl.schema import DataSpec, DaysSince, ExternalFeed

    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    px = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}, index=idx
    )
    px_path = tmp_path / "px.csv"
    px.to_csv(px_path)
    ev = pd.DataFrame({"v": [1.0, 2.0]}, index=pd.DatetimeIndex(["2024-01-03", "2024-01-08"]))
    ev_path = tmp_path / "ev.csv"
    ev.to_csv(ev_path)

    # the spec NAMES its series; the files are the invocation's own answer
    files = DataFiles(targets={"target": str(px_path)}, feeds={"ev": str(ev_path)})
    md = load_market_data(DataSpec(targets=["target"], external={"ev": ExternalFeed()}), files)
    got = vz.build_series(DaysSince(name="ev"), md)[0]["target"].to_numpy()
    want = np.array([np.nan, np.nan, 0, 1, 2, 3, 4, 0, 1, 2], dtype=float)
    np.testing.assert_allclose(got, want, equal_nan=True)

    # a publication lag shifts availability: with lag=1d the first stamp lands on 01-04
    md_lag = load_market_data(
        DataSpec(targets=["target"], external={"ev": ExternalFeed(lag=1)}), files
    )
    got_lag = vz.build_series(DaysSince(name="ev"), md_lag)[0]["target"].to_numpy()
    want_lag = np.array([np.nan, np.nan, np.nan, 0, 1, 2, 3, 4, 0, 1], dtype=float)
    np.testing.assert_allclose(got_lag, want_lag, equal_nan=True)


# ---- three-valued (Kleene) decision evaluation -------------------------------


def _hole_md(values, close=100.0):
    """Single-target MarketData whose external feed ``x`` is ``values`` (NaN = a data hole)."""
    idx = pd.date_range("2021-01-01", periods=len(values), freq="1D")
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
            "_ext_x": pd.Series(values, index=idx, dtype=float),
        },
        index=idx,
    )
    return _md(df, externals={"x"})


def _chan(node, md):
    v, i, d = vz.build_condition(node, md)
    return (v.to_numpy().ravel(), i.to_numpy().ravel(), d.to_numpy().ravel())


def _x_gt(v):
    return ThresholdCondition(left=External(name="x"), op=">", right=Constant(value=v))


def test_threshold_defined_only_where_operands_are_finite():
    md = _hole_md([1.0, np.nan, 3.0])
    value, init, defined = _chan(_x_gt(2.0), md)
    np.testing.assert_array_equal(value, [False, False, True])
    np.testing.assert_array_equal(init, [True, True, True])
    # the middle bar's comparison is UNDECIDED, not an ordinary False
    np.testing.assert_array_equal(defined, [True, False, True])
    np.testing.assert_array_equal(vz.undefined_mask(_x_gt(2.0), md).ravel(), [False, True, False])


def test_leading_nan_leaf_is_warmup_not_a_hole():
    # A feed that simply starts late is warmup (init False), never an undefined decision: the
    # ledger must not refuse a thesis for its own warmup.
    md = _hole_md([np.nan, np.nan, 3.0, 4.0])
    _value, init, defined = _chan(_x_gt(2.0), md)
    np.testing.assert_array_equal(init, [False, False, True, True])
    assert defined.all()  # warmup is vacuously defined
    assert not vz.undefined_mask(_x_gt(2.0), md).any()


def test_not_over_a_hole_does_not_fire():
    # The phantom fire two-valued logic would manufacture: NaN → comparison False → `not` negates
    # it to True → a FIRING out of missing data. The tradable mask requires `defined`.
    md = _hole_md([1.0, np.nan, 3.0])
    node = NotCondition(condition=_x_gt(2.0))
    value, _init, defined = _chan(node, md)
    np.testing.assert_array_equal(value, [True, True, False])  # value channel unchanged
    np.testing.assert_array_equal(defined, [True, False, True])
    np.testing.assert_array_equal(vz.signal(node, md).to_numpy().ravel(), [True, False, False])


def test_and_kleene_false_dominates():
    # F∧U = F: a decided-False child settles the conjunction whatever the unknown one was.
    md = _hole_md([1.0, np.nan, np.nan])
    #                       x>0 :  T   U   U        (undefined where x is NaN)
    #        close < 1 (always False) :  F   F   F
    decisive_false = ThresholdCondition(
        left=Field(column="close"), op="<", right=Constant(value=1.0)
    )
    node = AndCondition(conditions=[_x_gt(0.0), decisive_false])
    _value, _init, defined = _chan(node, md)
    assert defined.all(), "a decided-False conjunct settles the AND — nothing is unknown"
    # …but with a True partner the unknown propagates.
    decisive_true = ThresholdCondition(
        left=Field(column="close"), op=">", right=Constant(value=1.0)
    )
    _v2, _i2, d2 = _chan(AndCondition(conditions=[_x_gt(0.0), decisive_true]), md)
    np.testing.assert_array_equal(d2, [True, False, False])


def test_or_kleene_true_dominates():
    # T∨U = T; F∨U = U.
    md = _hole_md([1.0, np.nan, np.nan])
    always_true = ThresholdCondition(left=Field(column="close"), op=">", right=Constant(value=1.0))
    _v, _i, d = _chan(OrCondition(conditions=[_x_gt(0.0), always_true]), md)
    assert d.all()
    always_false = ThresholdCondition(left=Field(column="close"), op="<", right=Constant(value=1.0))
    _v2, _i2, d2 = _chan(OrCondition(conditions=[_x_gt(0.0), always_false]), md)
    np.testing.assert_array_equal(d2, [True, False, False])


def test_or_with_a_warming_branch_is_warmup_not_undefined():
    # `or`'s init is ANY child, so without the warmup-vacuous convention a still-warming branch
    # would mark every early bar of an ordinary multi-branch thesis undefined — and refuse it.
    md = _hole_md([1.0, 2.0, 3.0, 4.0, 5.0])
    warming = ThresholdCondition(  # a 3-bar SMA z-score: bars 0..1 are warmup
        left=ZScore(input=Field(column="close"), window=3, mean_type="sma"),
        op=">",
        right=Constant(value=0.0),
    )
    node = OrCondition(conditions=[_x_gt(0.0), warming])
    assert not vz.undefined_mask(node, md).any()


def test_rolling_defined_requires_the_whole_window_decided():
    md = _hole_md([1.0, 1.0, np.nan, 1.0, 1.0, 1.0])
    node = RollingCondition(window=3, agg="all", condition=_x_gt(0.0))
    _v, _i, defined = _chan(node, md)
    # every window containing the hole (bars 2,3,4) is undecided; bar 5's window is clean again
    np.testing.assert_array_equal(defined[2:], [False, False, False, True])


def test_first_true_hole_taints_the_following_bar():
    # A hole breaks the transition state: whether bar 3's True is a false→true EDGE depends on
    # the missing bar, so it cannot be reported as decided.
    md = _hole_md([-1.0, np.nan, 1.0, 1.0])
    node = FirstTrueCondition(condition=_x_gt(0.0))
    _v, _i, defined = _chan(node, md)
    np.testing.assert_array_equal(defined, [True, False, False, True])


def test_defined_channel_parity_with_reference_on_holed_feed(ohlcv_ext):
    # The composed Kleene tables must match the independently-authored oracle, holes included.
    df = ohlcv_ext.copy()
    df.loc[df.index[50], "_ext_rate"] = np.nan
    df.loc[df.index[120], "_ext_rate"] = np.nan
    inner = ThresholdCondition(
        left=ZScore(input=External(name="rate"), window=20, mean_type="sma"),
        op="<",
        right=Constant(value=-0.5),
    )
    other = ThresholdCondition(left=Field(column="close"), op=">", right=Field(column="open"))
    for node in (
        inner,
        NotCondition(condition=inner),
        AndCondition(conditions=[inner, other]),
        OrCondition(conditions=[inner, other]),
        RollingCondition(window=5, agg="all", condition=inner),
        FirstTrueCondition(condition=inner, cooldown=3),
    ):
        got = vz.build_condition(node, _md(df, externals={"rate"}))[2]["t"].to_numpy()
        np.testing.assert_array_equal(got, _ref_cond_defined(node, df), err_msg=repr(node.type))


def test_signal_is_unchanged_on_fully_finite_inputs(ohlcv_ext):
    # `defined` narrows the mask ONLY where a decision is undefined; with no holes anywhere the
    # tradable signal must be bit-identical to plain value & init.
    md = _md(ohlcv_ext, externals={"rate"})
    inner = ThresholdCondition(
        left=ZScore(input=External(name="rate"), window=20, mean_type="sma"),
        op="<",
        right=Constant(value=-0.5),
    )
    for node in (
        inner,
        NotCondition(condition=inner),
        RollingCondition(window=5, agg="any", condition=inner),
        FirstTrueCondition(condition=inner, cooldown=2),
    ):
        value, init, _defined = vz.build_condition(node, md)
        np.testing.assert_array_equal(
            vz.signal(node, md).to_numpy(), value.to_numpy() & init.to_numpy()
        )


@pytest.mark.parametrize(
    ("op", "feed_value", "left", "right"),
    [
        ("+", 1.5e308, External(name="x"), External(name="x")),
        ("-", 1.5e308, External(name="x"), Constant(value=-1.5e308)),
        ("*", 1e200, External(name="x"), External(name="x")),
        ("/", 1e-320, Constant(value=1.0), External(name="x")),
    ],
)
def test_binary_op_overflow_is_undefined_never_a_firing(op, feed_value, left, right):
    # A finite feed value whose arithmetic overflows (or a denormal denominator) must land in
    # the undecidable ledger — the audit's 1e308·2 probe fired [True, True, True] with full
    # definedness under the old divide-only sanitization.
    md = _hole_md([1.0, feed_value, 1.0])
    node = ThresholdCondition(
        left=BinaryOp(left=left, op=op, right=right),
        op=">",
        right=Constant(value=0.0),
    )
    value, init, defined = _chan(node, md)
    assert not value[1]  # the overflow bar never reads True
    np.testing.assert_array_equal(init, [True, True, True])
    np.testing.assert_array_equal(defined, [True, False, True])
    np.testing.assert_array_equal(vz.undefined_mask(node, md).ravel(), [False, True, False])
    assert not vz.signal(node, md).to_numpy().ravel()[1]


def test_threshold_with_inf_operand_is_undefined():
    # A hand-built md can carry the inf a strict CSV never could: definedness is minted from
    # FINITENESS, so an inf operand is an undecided bar, never a decided True — and a LEADING
    # inf is warmup, not initialization.
    md = _hole_md([1.0, np.inf, 3.0])
    value, _init, defined = _chan(_x_gt(2.0), md)
    np.testing.assert_array_equal(value, [False, False, True])
    np.testing.assert_array_equal(defined, [True, False, True])
    md2 = _hole_md([np.inf, 1.0, 3.0])
    _v, init2, _d = _chan(_x_gt(2.0), md2)
    np.testing.assert_array_equal(init2, [False, True, True])


# ---- the event algebra: mask / lag / bars_since_event / event_value / event_agg ----------


def _ev_fixture(seed: int = 5, n: int = 200, hole_at=()) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="1D")
    close = pd.Series(100 + rng.randn(n).cumsum(), index=idx)
    z = pd.Series(rng.randn(n), index=idx)
    for h in hole_at:
        z.iloc[h] = np.nan
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000.0,
            "_ext_z": z,
        },
        index=idx,
    )


def _z_gt(v):
    return ThresholdCondition(left=External(name="z"), op=">", right=Constant(value=v))


def _c_gt(v):
    return ThresholdCondition(left=Field(column="close"), op=">", right=Constant(value=v))


_EVENT = FirstTrueCondition(condition=_z_gt(1.0))


def test_mask_parity():
    df = _ev_fixture()
    _assert_series(Mask(condition=_z_gt(0.0)), df, externals=("z",))
    _assert_series(Mask(condition=_EVENT), df, externals=("z",))


def test_lag_parity():
    df = _ev_fixture(hole_at=(50, 51, 120))
    for k in (1, 3):
        _assert_cond(LagCondition(condition=_z_gt(0.0), periods=k), df, externals=("z",))
        _assert_cond(
            AndCondition(
                conditions=[
                    RollingCondition(
                        window=4, agg="any", condition=LagCondition(condition=_z_gt(0.5))
                    ),
                    _c_gt(90.0),
                ]
            ),
            df,
            externals=("z",),
        )


def test_bars_since_event_parity():
    df = _ev_fixture(hole_at=(70, 130))
    _assert_series(BarsSinceEvent(event=_EVENT), df, externals=("z",))
    _assert_series(BarsSinceEvent(event=_z_gt(1.5)), df, externals=("z",))


def test_event_value_parity():
    df = _ev_fixture(hole_at=(30,))
    node = EventValue(
        event=_EVENT, input=Shift(input=RollingAgg(input=Field(column="high"), window=5, agg="max"))
    )
    _assert_series(node, df, externals=("z",))


@pytest.mark.parametrize("agg", ["sum", "max", "min", "mean"])
def test_event_agg_parity(agg):
    df = _ev_fixture(hole_at=(30, 90))
    node = EventAgg(event=_EVENT, input=Change(input=Field(column="close"), kind="diff"), agg=agg)
    _assert_series(node, df, externals=("z",))
    _assert_series(
        EventAgg(event=_z_gt(1.0), input=Mask(condition=_z_gt(0.0)), agg=agg), df, externals=("z",)
    )


def test_event_node_init_channel_parity():
    # The reference init (the standard latch OR "anchor became unknowable post-warmup", and the
    # condition's own latch for mask) must match the builder's init frame, hole cases included.
    df = _ev_fixture(hole_at=(3, 4, 60))
    md = _md(df, externals=("z",))
    for node in (
        Mask(condition=_z_gt(0.0)),
        BarsSinceEvent(event=_z_gt(1.5)),
        EventValue(event=_z_gt(1.5), input=Field(column="close")),
        EventAgg(event=_z_gt(1.5), input=Field(column="close"), agg="sum"),
    ):
        got = vz.build_series(node, md)[1]["t"].to_numpy()
        np.testing.assert_array_equal(got, _ref_init(node, df))


def _ev_md(z, close=None):
    n = len(z)
    idx = pd.date_range("2021-01-01", periods=n, freq="1D")
    c = pd.Series(np.asarray(close if close is not None else [100.0] * n, float), index=idx)
    df = pd.DataFrame(
        {"open": c, "high": c, "low": c, "close": c, "_ext_z": np.asarray(z, float)}, index=idx
    )
    return _md(df, externals=("z",))


def _series3(node, md):
    v, i = vz.build_series(node, md)
    return v["t"].to_numpy(), i["t"].to_numpy()


def test_mask_channels_warmup_hole_false_true():
    md = _ev_md([np.nan, np.nan, 0.0, 2.0, np.nan, 0.0, 3.0])
    v, i = _series3(Mask(condition=_z_gt(1.0)), md)
    # warmup → NaN/not init; decided False → 0.0; decided True → 1.0; a post-warmup hole → NaN
    # while INITIALIZED (so a threshold over it is undefined → the ledger)
    np.testing.assert_array_equal(v, [np.nan, np.nan, 0.0, 1.0, np.nan, 0.0, 1.0])
    np.testing.assert_array_equal(i, [False, False, True, True, True, True, True])
    cond = ThresholdCondition(left=Mask(condition=_z_gt(1.0)), op=">", right=Constant(value=0.5))
    np.testing.assert_array_equal(
        vz.undefined_mask(cond, md)[:, 0], [False, False, False, False, True, False, False]
    )


def test_mask_init_is_the_conditions_own_latch():
    # An `or` initializes when ANY branch does, so mask(or(...)) is initialized before either
    # branch alone would latch a finite value — the condition's latch, not `_latch(values)`.
    md = _ev_md([np.nan, 1.0, 1.0], close=[100.0, 100.0, 100.0])
    cond = OrCondition(conditions=[_z_gt(0.5), _c_gt(50.0)])
    _v, i = _series3(Mask(condition=cond), md)
    ci = vz.build_condition(cond, md)[1]["t"].to_numpy()
    np.testing.assert_array_equal(i, ci)
    assert i.all()


def test_lag_shifts_all_three_channels_and_leading_bars_are_warmup():
    md = _ev_md([np.nan, 2.0, 0.0, np.nan, 2.0, 0.0])
    v, i, d = (
        f["t"].to_numpy()
        for f in vz.build_condition(LagCondition(condition=_z_gt(1.0), periods=2), md)
    )
    np.testing.assert_array_equal(v, [False, False, False, True, False, False])
    np.testing.assert_array_equal(i, [False, False, False, True, True, True])
    # the hole at bar 3 moves with the decision it affects → undefined at bar 5
    np.testing.assert_array_equal(d, [True, True, True, True, True, False])


def test_lag_periods_beyond_length_never_initializes():
    md = _ev_md([2.0, 2.0, 2.0])
    _v, i, d = (
        f["t"].to_numpy()
        for f in vz.build_condition(LagCondition(condition=_z_gt(1.0), periods=3), md)
    )
    assert not i.any() and d.all()
    assert not vz.signal(LagCondition(condition=_z_gt(1.0), periods=3), md).to_numpy().any()


def test_bars_since_event_is_warmup_before_first_event_and_undefined_after_a_hole():
    md = _ev_md([0.0, 0.0, 5.0, 0.0, np.nan, 0.0, 5.0, 0.0])
    v, i = _series3(BarsSinceEvent(event=_z_gt(1.0)), md)
    np.testing.assert_array_equal(v, [np.nan, np.nan, 0, 1, np.nan, np.nan, 0, 1])
    np.testing.assert_array_equal(i, [False, False, True, True, True, True, True, True])
    cond = ThresholdCondition(
        left=BarsSinceEvent(event=_z_gt(1.0)), op=">=", right=Constant(value=0.0)
    )
    np.testing.assert_array_equal(vz.undefined_mask(cond, md)[:, 0], [0, 0, 0, 0, 1, 1, 0, 0])


def test_event_hole_before_first_event_is_ledgered_not_warmup():
    # A post-warmup hole in E before E ever fired: the plain value latch would read every bar as
    # warmup and absorb the hole; the init channel latches on the unknowable anchor instead.
    md = _ev_md([0.0, np.nan, 0.0, 0.0])
    v, i = _series3(BarsSinceEvent(event=_z_gt(1.0)), md)
    assert np.isnan(v).all()
    np.testing.assert_array_equal(i, [False, True, True, True])


def test_event_value_nan_input_at_event_stays_nan_until_next_event():
    md = _ev_md([5.0, 0.0, 5.0, 0.0], close=[np.nan, 1.0, 2.0, 3.0])
    v, _i = _series3(EventValue(event=_z_gt(1.0), input=Field(column="close")), md)
    np.testing.assert_array_equal(v, [np.nan, np.nan, 2.0, 2.0])


def test_event_agg_poisons_until_next_event():
    md = _ev_md([5.0, 0.0, 0.0, 0.0, 5.0, 0.0], close=[1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    v, _i = _series3(EventAgg(event=_z_gt(1.0), input=Field(column="close"), agg="sum"), md)
    np.testing.assert_array_equal(v, [1.0, 3.0, np.nan, np.nan, 5.0, 11.0])


def test_event_agg_mask_recipes():
    # any / all / count of C since S — the three counting recipes over mask(C)
    md = _ev_md([5.0, 0.0, 0.0, 5.0, 0.0, 0.0], close=[1.0, 9.0, 1.0, 9.0, 9.0, 1.0])
    S = _z_gt(1.0)
    C = _c_gt(5.0)
    any_since = EventAgg(event=S, input=Mask(condition=C), agg="max")
    all_since = EventAgg(event=S, input=Mask(condition=C), agg="min")
    count_since = EventAgg(event=S, input=Mask(condition=C), agg="sum")
    np.testing.assert_array_equal(_series3(any_since, md)[0], [0, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(_series3(all_since, md)[0], [0, 0, 0, 1, 1, 0])
    np.testing.assert_array_equal(_series3(count_since, md)[0], [0, 1, 1, 1, 2, 2])


def test_event_and_input_at_the_same_bar():
    # event and input both at t: the snapshot IS x[t] and bars_since_event is 0 — the current
    # bar is included by definition (the exclusive forms are shift/lag).
    md = _ev_md([0.0, 5.0, 0.0], close=[1.0, 7.0, 3.0])
    assert _series3(EventValue(event=_z_gt(1.0), input=Field(column="close")), md)[0][1] == 7.0
    assert _series3(BarsSinceEvent(event=_z_gt(1.0)), md)[0][1] == 0.0
    assert (
        _series3(EventAgg(event=_z_gt(1.0), input=Field(column="close"), agg="sum"), md)[0][1]
        == 7.0
    )


def test_strict_ordering_recipe():
    # The review's synthetic case: A and B first turn true TOGETHER at bar 2. `rolling(any, A, N)`
    # includes the current bar and fires there; `rolling(any, lag(A, 1), N)` needs A on a PRIOR
    # bar and does not.
    md = _ev_md([0, 0, 1, 0, 0, 0, 1], close=[0, 0, 1, 1, 0, 1, 0])
    A = _c_gt(0.5)
    B = _z_gt(0.5)
    loose = AndCondition(conditions=[RollingCondition(window=3, agg="any", condition=A), B])
    strict = AndCondition(
        conditions=[
            RollingCondition(window=3, agg="any", condition=LagCondition(condition=A, periods=1)),
            B,
        ]
    )
    np.testing.assert_array_equal(np.flatnonzero(vz.signal(loose, md).to_numpy()[:, 0]), [2, 6])
    np.testing.assert_array_equal(np.flatnonzero(vz.signal(strict, md).to_numpy()[:, 0]), [6])


def test_event_condition_shared_by_several_nodes_is_built_once():
    # Memo keys are the canonical JSON (nested conditions included), so an event condition shared
    # by several nodes is built once and both nodes read the same channels.
    md = _md(_ev_fixture(), externals=("z",))
    ev = FirstTrueCondition(condition=_z_gt(0.5))
    entry = AndCondition(
        conditions=[
            ThresholdCondition(left=BarsSinceEvent(event=ev), op=">=", right=Constant(value=1.0)),
            ThresholdCondition(
                left=EventValue(event=ev, input=Field(column="close")),
                op="<",
                right=Field(column="close"),
            ),
        ]
    )
    calls: list[str] = []
    real = vz._build_condition

    def spy(node, md_):
        calls.append(node.model_dump_json())
        return real(node, md_)

    vz._build_condition = spy
    try:
        vz.signal(entry, md)
    finally:
        vz._build_condition = real
    assert calls.count(ev.model_dump_json()) == 1


def test_collect_sweeps_lag_periods_axis():
    entry = AndCondition(
        conditions=[
            RollingCondition(
                window=3, agg="any", condition=LagCondition(condition=_c_gt(1.0), periods=[1, 2])
            ),
            _c_gt(2.0),
        ]
    )
    assert vz.collect_sweeps(entry) == [("lag_periods", [1, 2])]
    combos = [combo for combo, _ in vz.iter_param_assignments(entry)]
    assert combos == [{"lag_periods": 1}, {"lag_periods": 2}]


def test_embedded_condition_sweeps_count_once_in_declared_grid():
    from seikan.dsl.schema import declared_grid_size

    # A swept cooldown inside the event condition and a swept input window are two axes, named
    # in engine order (embedded condition first) and each counted ONCE by the grid-size walk —
    # the embedded condition's sweeps are delegated to the condition walk, never double-counted.
    ev = FirstTrueCondition(condition=_c_gt(1.0), cooldown=[0, 3])
    single = ThresholdCondition(
        left=EventValue(event=ev, input=EMA(input=Field(column="close"), window=[5, 10])),
        op="<",
        right=Field(column="close"),
    )
    assert vz.collect_sweeps(single) == [("first_true_cooldown", [0, 3]), ("ema_window", [5, 10])]
    assert declared_grid_size(single, 1) == 4
    # The SAME event condition written into two nodes is two OCCURRENCES of a swept param — two
    # independent axes (`first_true_cooldown`, `first_true_cooldown_2`), exactly as two swept
    # transform windows are today; a shared axis is what ties them (see the `axes` field).
    entry = AndCondition(
        conditions=[
            ThresholdCondition(left=BarsSinceEvent(event=ev), op=">=", right=Constant(value=1.0)),
            single,
        ]
    )
    assert [lvl for lvl, _ in vz.collect_sweeps(entry)] == [
        "first_true_cooldown",
        "first_true_cooldown_2",
        "ema_window",
    ]
    assert declared_grid_size(entry, 1) == 8


# ---- native-clock transforms ---------------------------------------------------------------


def _native_prints(stamps, values) -> pd.Series:
    return pd.Series(np.asarray(values, float), index=pd.DatetimeIndex(stamps))


def _ref_native(expr, prints: np.ndarray) -> np.ndarray:
    """The reference evaluation of a native expr over the print sequence: the frozen oracles of
    ``_ref_series`` applied to a one-column frame whose ``_ext_<feed>`` column is the prints."""
    frame = pd.DataFrame({"_ext_eps": prints})
    return _ref_series(expr, frame)


def _ref_anchor(out: np.ndarray, stamps: pd.DatetimeIndex, index: pd.DatetimeIndex) -> np.ndarray:
    """Backward asof by hand: each bar takes the latest print stamped at-or-before it."""
    res = np.full(len(index), np.nan)
    for i, ts in enumerate(index):
        j = np.searchsorted(stamps.to_numpy(), np.datetime64(ts), side="right") - 1
        if j >= 0:
            res[i] = out[j]
    return res


def test_native_parity_with_native_clock_reference(ohlcv):
    rng = np.random.RandomState(9)
    # ~3 prints a week, some inside one bar's day, over the bar index's span
    stamps = pd.DatetimeIndex(
        sorted(set(ohlcv.index[0] + pd.to_timedelta(rng.randint(0, 299 * 24, 140), unit="h")))
    )
    prints = _native_prints(stamps, rng.randn(len(stamps)).cumsum() + 50)
    md = _md(ohlcv, natives={"eps": prints})
    eps = External(name="eps")
    exprs = [
        RollingAgg(input=eps, window=5, agg="mean"),
        EMA(input=eps, window=10),
        BinaryOp(
            left=Change(input=eps, periods=4, kind="diff"),
            op="/",
            right=RollingAgg(input=Change(input=eps, periods=4, kind="diff"), window=8, agg="std"),
        ),
        ZScore(input=eps, window=12, mean_type="ema"),
        Percentile(input=Shift(input=eps, periods=2), window=6),
        UnaryOp(input=Drawdown(input=eps, window=7), op="abs"),
    ]
    for expr in exprs:
        got = vz.build_series(Native(name="eps", expr=expr), md)[0]["t"].to_numpy()
        want = _ref_anchor(_ref_native(expr, prints.to_numpy()), stamps, ohlcv.index)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_native_windows_count_native_prints_not_bars(ohlcv):
    # A weekly print on daily bars: rolling_agg(external, 3) needs three BARS of the ffilled feed
    # (defined from the third bar after the first print), native(rolling_agg(eps, 3)) needs three
    # PRINTS (defined from the third print's bar).
    stamps = ohlcv.index[::7][:10]
    prints = _native_prints(stamps, np.arange(10, dtype=float))
    df = ohlcv.copy()
    df["_ext_eps"] = prints.reindex(ohlcv.index, method="ffill")
    md = _md(df, externals=("eps",), natives={"eps": prints})
    eps = External(name="eps")
    bar = vz.build_series(RollingAgg(input=eps, window=3, agg="mean"), md)[0]["t"].to_numpy()
    nat = vz.build_series(Native(name="eps", expr=RollingAgg(input=eps, window=3, agg="mean")), md)[
        0
    ]["t"].to_numpy()
    assert int(np.argmax(np.isfinite(bar))) == 2
    assert int(np.argmax(np.isfinite(nat))) == 14
    assert nat[14] == pytest.approx(1.0) and bar[2] == pytest.approx(0.0)


def test_native_per_target_feed_evaluates_each_member_on_its_own_clock():
    idx = pd.date_range("2022-01-01", periods=12, freq="1D")
    close = pd.DataFrame({"a": 100.0, "b": 100.0}, index=idx)
    pa = _native_prints(["2022-01-02", "2022-01-05", "2022-01-09"], [1.0, 3.0, 5.0])
    pb = _native_prints(
        ["2022-01-01", "2022-01-03", "2022-01-04", "2022-01-11"], [10.0, 30.0, 50.0, 70.0]
    )
    md = MarketData(
        close=close,
        open=close,
        high=close,
        low=close,
        volume=None,
        externals={},
        targets=["a", "b"],
        externals_native={"eps": {"a": pa, "b": pb}},
    )
    got = vz.build_series(
        Native(name="eps", expr=RollingAgg(input=External(name="eps"), window=2, agg="mean")), md
    )[0]
    want_a = _ref_anchor(ref.rolling_agg(pa.to_numpy(), 2, "mean"), pa.index, idx)
    want_b = _ref_anchor(ref.rolling_agg(pb.to_numpy(), 2, "mean"), pb.index, idx)
    np.testing.assert_allclose(got["a"].to_numpy(), want_a, equal_nan=True)
    np.testing.assert_allclose(got["b"].to_numpy(), want_b, equal_nan=True)
    assert got["a"].to_numpy()[4] == pytest.approx(2.0) and got["b"].to_numpy()[3] == pytest.approx(
        40.0
    )


def test_native_hand_built_md_without_native_prints_raises_naming_the_feed(ohlcv_ext):
    md = _md(ohlcv_ext, externals=("rate",))
    with pytest.raises(ValueError, match="external feed 'rate' has no retained native prints"):
        vz.build_series(Native(name="rate", expr=EMA(input=External(name="rate"), window=3)), md)


def test_native_explicit_nan_print_is_a_hole_on_the_native_clock(ohlcv):
    stamps = ohlcv.index[[0, 3, 6, 9, 12, 15]]
    prints = _native_prints(stamps, [1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    md = _md(ohlcv, natives={"eps": prints})
    got = vz.build_series(
        Native(name="eps", expr=RollingAgg(input=External(name="eps"), window=2, agg="mean")), md
    )[0]["t"].to_numpy()
    # the NaN print poisons the two windows it sits in (prints 2 and 3), and stays NaN on the
    # bars those prints cover — a hole on the native clock is a hole on every bar it anchors to
    assert np.isnan(got[6:12]).all()
    assert got[3] == pytest.approx(1.5) and got[12] == pytest.approx(4.5)
    assert vz.build_series(
        Native(name="eps", expr=RollingAgg(input=External(name="eps"), window=2, agg="mean")), md
    )[1]["t"].to_numpy()[6]


# ---- the cross-sectional population model: where / group ---------------------------------


def _md4(x, elig=None, sector=None, n=8) -> MarketData:
    """Four flat-priced targets with per-target feeds ``x`` (constant per member), optional
    ``elig`` and ``sector`` (constant per member, or a per-member array over the bars)."""
    idx = pd.date_range("2022-01-01", periods=n, freq="1D")
    targets = ["T0", "T1", "T2", "T3"]
    close = pd.DataFrame(dict.fromkeys(targets, 100.0), index=idx)

    def feed(values):
        return pd.DataFrame(
            {
                t: np.broadcast_to(np.asarray(v, float), (n,))
                for t, v in zip(targets, values, strict=True)
            },
            index=idx,
        )

    ext = {"x": feed(x)}
    if elig is not None:
        ext["elig"] = feed(elig)
    if sector is not None:
        ext["sector"] = feed(sector)
    return MarketData(
        close=close, open=close, high=close, low=close, volume=None, externals=ext, targets=targets
    )


def _e(name, op=">=", v=1.0):
    return ThresholdCondition(left=External(name=name), op=op, right=Constant(value=v))


_X = External(name="x")
_ELIG = _e("elig")


def test_cross_membership_without_where_or_group_is_input_finiteness():
    md = _md3()
    node = CrossRank(input=ZScore(input=Field(column="close"), window=20))
    x = vz.build_series(node.input, md)[0].to_numpy(dtype=float)
    np.testing.assert_array_equal(vz.cross_membership(node, md), np.isfinite(x))
    assert vz.cross_population(node, md).all()


def test_cross_rank_where_ranks_within_the_eligible_population():
    # The review's case: values [100, 90, 80, 70], the last two eligible → ranks within {80, 70}
    # are [nan, nan, 1.0, 0.0], and the kernel saw k = 2.
    md = _md4(x=[100, 90, 80, 70], elig=[0, 0, 1, 1])
    v, i = vz.build_series(CrossRank(input=_X, where=_ELIG), md)
    np.testing.assert_array_equal(v.iloc[-1].to_numpy(), [np.nan, np.nan, 1.0, 0.0])
    np.testing.assert_array_equal(i.iloc[-1].to_numpy(), [False, False, True, True])
    assert (vz.cross_membership(CrossRank(input=_X, where=_ELIG), md).sum(axis=1) == 2).all()
    # without `where` the global ranks put 80 in the bottom half
    np.testing.assert_allclose(
        vz.build_series(CrossRank(input=_X), md)[0].iloc[-1].to_numpy(), [1.0, 2 / 3, 1 / 3, 0.0]
    )


def test_cross_rank_where_late_eligibility_feed_is_warmup_not_a_hole():
    # T0's eligibility feed starts late (leading NaN): warmup, so T0 is simply out of the
    # population there — no bar is voided and nothing is undefined.
    elig0 = np.array([np.nan, np.nan, np.nan, 1, 1, 1, 1, 1])
    md = _md4(x=[100, 90, 80, 70], elig=[elig0, 1, 1, 1])
    node = CrossRank(input=_X, where=_ELIG)
    cond = ThresholdCondition(left=node, op=">=", right=Constant(value=0.0))
    assert not vz.undefined_mask(cond, md).any()
    k = vz.cross_membership(node, md).sum(axis=1)
    np.testing.assert_array_equal(k, [3, 3, 3, 4, 4, 4, 4, 4])


def test_cross_rank_where_undefined_member_voids_the_whole_bar():
    # A post-warmup hole in ONE member's eligibility voids the bar for EVERY member (fail-closed
    # population): eligible, initialized members read undefined there. And the outer guard does
    # NOT absorb an undefined eligibility of its own (U ∧ U = U) — that hole is ledgered too.
    elig0 = np.array([1, 1, 1, np.nan, 1, 1, 1, 1])
    md = _md4(x=[100, 90, 80, 70], elig=[elig0, 0, 1, 1])
    node = CrossRank(input=_X, where=_ELIG)
    guarded = AndCondition(
        conditions=[_ELIG, ThresholdCondition(left=node, op=">=", right=Constant(value=0.8))]
    )
    undef = vz.undefined_mask(guarded, md)
    np.testing.assert_array_equal(undef[3], [True, False, True, True])
    assert undef.sum() == 3
    assert vz.cross_membership(node, md).sum(axis=1)[3] == 0
    np.testing.assert_array_equal(vz.signal(guarded, md).to_numpy()[3], [False] * 4)


def test_cross_rank_where_outer_guard_makes_ineligible_decidedly_false():
    # The canonical idiom: a once-eligible member that turns ineligible reads post-init NaN from
    # the node (a bare threshold would be undefined); `and(E, …)` absorbs it as False (F ∧ U = F).
    elig3 = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    md = _md4(x=[100, 90, 80, 70], elig=[1, 1, 1, elig3])
    node = CrossRank(input=_X, where=_ELIG)
    bare = ThresholdCondition(left=node, op=">=", right=Constant(value=0.0))
    guarded = AndCondition(conditions=[_ELIG, bare])
    assert vz.undefined_mask(bare, md)[4:, 3].all()
    assert not vz.undefined_mask(guarded, md).any()
    assert not vz.signal(guarded, md).to_numpy()[4:, 3].any()
    assert vz.signal(guarded, md).to_numpy()[:4, 3].all()


def test_cross_rank_where_bare_threshold_leaves_a_previously_eligible_member_undefined():
    elig3 = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    md = _md4(x=[100, 90, 80, 70], elig=[1, 1, 1, elig3])
    bare = ThresholdCondition(
        left=CrossRank(input=_X, where=_ELIG), op=">=", right=Constant(value=0.0)
    )
    u = vz.undefined_mask(bare, md)
    assert u[4:, 3].all() and not u[:4].any() and not u[:, :3].any()


def test_cross_agg_where_excluded_member_reads_no_aggregate_and_init_stays_own_input_gated():
    md = _md4(x=[100, 90, 80, 70], elig=[0, 1, 1, 1])
    v, i = vz.build_series(CrossAgg(input=_X, agg="mean", where=_ELIG), md)
    np.testing.assert_array_equal(v.iloc[-1].to_numpy(), [np.nan, 80.0, 80.0, 80.0])
    np.testing.assert_array_equal(i.iloc[-1].to_numpy(), [False, True, True, True])
    # a warming member INSIDE the population still sees the group's value, but cannot fire off it
    x1 = np.array([np.nan, np.nan, 90, 90, 90, 90, 90, 90])
    md = _md4(x=[100, x1, 80, 70], elig=[1, 1, 1, 1])
    v, i = vz.build_series(CrossAgg(input=_X, agg="mean", where=_ELIG), md)
    assert v.iloc[0, 1] == pytest.approx((100 + 80 + 70) / 3) and not i.iloc[0, 1]
    assert i.iloc[2, 1]


def test_cross_group_constant_label_is_bit_exact_with_ungrouped():
    md = _md3()
    for plain, grouped in (
        (
            CrossRank(input=Field(column="close")),
            CrossRank(input=Field(column="close"), group=Constant(value=7.0)),
        ),
        (
            CrossDemean(input=Field(column="close"), min_valid=3),
            CrossDemean(input=Field(column="close"), min_valid=3, group=Constant(value=1.0)),
        ),
        (
            CrossAgg(input=ZScore(input=Field(column="close"), window=10), agg="std"),
            CrossAgg(
                input=ZScore(input=Field(column="close"), window=10),
                agg="std",
                group=Constant(value=0.0),
            ),
        ),
    ):
        pv, pi = vz.build_series(plain, md)
        gv, gi = vz.build_series(grouped, md)
        assert np.array_equal(pv.to_numpy(), gv.to_numpy(), equal_nan=True)
        assert np.array_equal(pi.to_numpy(), gi.to_numpy())


def test_cross_rank_group_ranks_within_sector():
    md = _md4(x=[100, 90, 80, 70], sector=[1, 1, 2, 2])
    v = vz.build_series(CrossRank(input=_X, group=External(name="sector")), md)[0]
    np.testing.assert_array_equal(v.iloc[-1].to_numpy(), [1.0, 0.0, 1.0, 0.0])
    a = vz.build_series(CrossAgg(input=_X, agg="mean", group=External(name="sector")), md)[0]
    np.testing.assert_array_equal(a.iloc[-1].to_numpy(), [95.0, 95.0, 75.0, 75.0])


def test_cross_rank_group_nan_label_is_excluded():
    md = _md4(x=[100, 90, 80, 70], sector=[1, 1, 1, np.nan])
    node = CrossRank(input=_X, group=External(name="sector"))
    v, i = vz.build_series(node, md)
    np.testing.assert_array_equal(v.iloc[-1].to_numpy(), [1.0, 0.5, 0.0, np.nan])
    assert not i.iloc[-1, 3]
    np.testing.assert_array_equal(vz.cross_membership(node, md)[-1], [True, True, True, False])


def test_cross_rank_group_label_changes_over_time():
    sector0 = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    md = _md4(x=[100, 90, 80, 70], sector=[sector0, 1, 2, 2])
    v = vz.build_series(CrossRank(input=_X, group=External(name="sector")), md)[0].to_numpy()
    np.testing.assert_array_equal(v[0], [1.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(v[-1], [1.0, np.nan, 0.5, 0.0])  # T1 alone in sector 1


def test_cross_rank_group_of_size_one_is_nan():
    md = _md4(x=[100, 90, 80, 70], sector=[1, 2, 2, 2])
    v = vz.build_series(CrossRank(input=_X, group=External(name="sector")), md)[0]
    assert np.isnan(v.iloc[-1, 0])


def test_collect_sweeps_cross_where_and_group_order_is_input_where_group():
    entry = ThresholdCondition(
        left=CrossRank(
            input=Change(input=Field(column="close"), periods=[5, 10]),
            where=RollingCondition(window=[3, 5], agg="any", condition=_e("elig")),
            group=Shift(input=External(name="sector"), periods=[1, 2]),
        ),
        op=">=",
        right=Constant(value=0.8),
    )
    assert [lvl for lvl, _ in vz.collect_sweeps(entry)] == [
        "change_periods",
        "rolling_window",
        "shift_periods",
    ]


def test_iter_param_assignments_scalarizes_where_and_group():
    entry = ThresholdCondition(
        left=CrossAgg(
            input=_X,
            agg="mean",
            where=RollingCondition(window=[3, 5], agg="any", condition=_e("elig")),
            group=Shift(input=External(name="sector"), periods=[1, 2]),
        ),
        op=">",
        right=Constant(value=0.0),
    )
    combos = list(vz.iter_param_assignments(entry))
    assert len(combos) == 4
    for combo, tree in combos:
        assert tree.left.where.window == combo["rolling_window"]
        assert tree.left.group.periods == combo["shift_periods"]


# ---- explicit-decay EW forms, robust window statistics, intraday calendar fields -----------


def test_ema_and_zscore_alpha_parity(ohlcv):
    _assert_series(EMA(input=Field(column="close"), alpha=0.06), ohlcv)
    _assert_series(EMA(input=Field(column="close"), alpha=1.0), ohlcv)
    _assert_series(ZScore(input=Field(column="close"), alpha=0.1, mean_type="ema"), ohlcv)
    # alpha = 2/(w+1) is the window form, bit for bit
    a = vz.build_series(EMA(input=Field(column="close"), alpha=2.0 / 21.0), _md(ohlcv))[0]
    w = vz.build_series(EMA(input=Field(column="close"), window=20), _md(ohlcv))[0]
    assert np.array_equal(a.to_numpy(), w.to_numpy(), equal_nan=True)


@pytest.mark.parametrize("agg", ["median", "mad"])
def test_rolling_median_and_mad_parity(ohlcv, agg):
    _assert_series(RollingAgg(input=Field(column="close"), window=9, agg=agg), ohlcv)
    _assert_series(RollingAgg(input=Change(input=Field(column="close")), window=5, agg=agg), ohlcv)


def test_robust_zscore_recipe_matches_hand_computation():
    # (x − median_N) / (1.4826 · mad_N): a flat window (mad = 0) divides by zero → NaN, no firing.
    idx = pd.date_range("2020-01-01", periods=8, freq="1D")
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, 5.0, 5.0, 5.0], index=idx)
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=idx)
    x = Field(column="close")
    med = RollingAgg(input=x, window=3, agg="median")
    mad = RollingAgg(input=x, window=3, agg="mad")
    rz = BinaryOp(
        left=BinaryOp(left=x, op="-", right=med),
        op="/",
        right=BinaryOp(left=Constant(value=1.4826), op="*", right=mad),
    )
    got = vz.build_series(rz, _md(df))[0]["t"].to_numpy()
    # bar 4: window [3, 4, 100] → median 4, |dev| [1, 0, 96] → mad 1 → (100 − 4) / 1.4826
    assert got[4] == pytest.approx(96.0 / 1.4826)
    # bar 7: window [5, 5, 5] → mad 0 → NaN (never fires)
    assert np.isnan(got[7])
    fired = vz.signal(ThresholdCondition(left=rz, op=">", right=Constant(value=3.0)), _md(df))
    np.testing.assert_array_equal(fired["t"].to_numpy(), [0, 0, 0, 0, 1, 0, 0, 0])


def test_calendar_hour_minute_on_intraday_bars():
    idx = pd.date_range("2020-01-01 09:30", periods=6, freq="30min")
    close = pd.Series(100.0, index=idx)
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=idx)
    md = _md(df)
    from seikan.dsl.schema import Calendar

    np.testing.assert_array_equal(
        vz.build_series(Calendar(field="hour"), md)[0]["t"].to_numpy(), [9, 10, 10, 11, 11, 12]
    )
    np.testing.assert_array_equal(
        vz.build_series(Calendar(field="minute"), md)[0]["t"].to_numpy(), [30, 0, 30, 0, 30, 0]
    )


# ---- shared sweep axes ---------------------------------------------------------------------


def _beta_tree(n):
    from seikan.dsl.schema import AxisRef, RollingCorr

    dc = Change(input=Field(column="close"), kind="diff")
    dv = Change(input=Field(column="volume"), kind="diff")
    if isinstance(n, str):
        n = AxisRef(axis=n)
    return ThresholdCondition(
        left=BinaryOp(
            left=BinaryOp(
                left=RollingCorr(left=dc, right=dv, window=n),
                op="*",
                right=RollingAgg(input=dc, window=n, agg="std"),
            ),
            op="/",
            right=RollingAgg(input=dv, window=n, agg="std"),
        ),
        op=">",
        right=Constant(value=0.5),
    )


def test_collect_sweeps_shared_axis_recorded_once_rolling_beta():
    # The review's case: three [20, 60] lists were three axes and eight cells; one shared N is
    # one axis, two combos, and every window in each scalarized tree reads combo["N"].
    entry = _beta_tree("N")
    assert vz.collect_sweeps(entry, {"N": [20, 60]}) == [("N", [20, 60])]
    combos = list(vz.iter_param_assignments(entry, {"N": [20, 60]}))
    assert [c for c, _ in combos] == [{"N": 20}, {"N": 60}]
    for combo, tree in combos:
        assert tree.left.left.left.window == combo["N"]
        assert tree.left.left.right.window == combo["N"]
        assert tree.left.right.window == combo["N"]
    # the listed form, for contrast: 8 cells
    listed = _beta_tree([20, 60])
    assert len(list(vz.iter_param_assignments(listed))) == 8


def test_collect_sweeps_shared_axis_does_not_advance_occurrence_counter():
    from seikan.dsl.schema import AxisRef

    entry = ThresholdCondition(
        left=EMA(input=EMA(input=Field(column="close"), window=[5, 10]), window=AxisRef(axis="N")),
        op=">",
        right=EMA(input=Field(column="close"), window=[3, 4]),
    )
    assert vz.collect_sweeps(entry, {"N": [20, 60]}) == [
        ("ema_window", [5, 10]),
        ("N", [20, 60]),
        ("ema_window_2", [3, 4]),
    ]


def test_collect_sweeps_shared_axis_colliding_with_auto_axis_refused():
    from seikan.dsl.schema import AxisRef

    entry = AndCondition(
        conditions=[
            ThresholdCondition(
                left=EMA(input=Field(column="close"), window=[3, 4]),
                op=">",
                right=Field(column="close"),
            ),
            ThresholdCondition(
                left=EMA(input=Field(column="close"), window=AxisRef(axis="ema_window")),
                op=">",
                right=Field(column="close"),
            ),
        ]
    )
    with pytest.raises(ValueError, match="duplicate sweep axis name 'ema_window'"):
        vz.collect_sweeps(entry, {"ema_window": [5, 6]})


@pytest.mark.parametrize("reserved", ["target", "horizon"])
def test_collect_sweeps_shared_axis_reserved_name_refused(reserved):
    with pytest.raises(ValueError, match=f"sweep axis name '{reserved}' is reserved"):
        vz.collect_sweeps(_beta_tree(reserved), {reserved: [20, 60]})


def test_make_resolver_refuses_axis_ref_without_axes():
    with pytest.raises(
        ValueError,
        match=r"rolling_corr.window references shared axis 'N', .*\(declared: \[\]\)",
    ):
        vz.collect_sweeps(_beta_tree("N"))
    with pytest.raises(ValueError, match=r"declared: \['M'\]"):
        vz.collect_sweeps(_beta_tree("N"), {"M": [20, 60]})


def test_iter_param_assignments_shared_axis_serves_constant_and_window():
    from seikan.dsl.schema import AxisRef

    entry = AndCondition(
        conditions=[
            ThresholdCondition(
                left=EMA(input=Field(column="close"), window=AxisRef(axis="N")),
                op=">",
                right=Field(column="close"),
            ),
            ThresholdCondition(
                left=Field(column="close"), op=">", right=Constant(value=AxisRef(axis="N"))
            ),
        ]
    )
    combos = list(vz.iter_param_assignments(entry, {"N": [20, 60]}))
    assert [c for c, _ in combos] == [{"N": 20}, {"N": 60}]
    for combo, tree in combos:
        assert tree.conditions[0].left.window == combo["N"]
        assert tree.conditions[1].right.value == float(combo["N"])
        assert tree.conditions[1].right.name is None
