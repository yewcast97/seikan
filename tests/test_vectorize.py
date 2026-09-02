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
    BarsSinceExtremum,
    BinaryOp,
    Change,
    Constant,
    CrossAgg,
    CrossDemean,
    CrossRank,
    Drawdown,
    External,
    Field,
    FirstTrueCondition,
    NotCondition,
    OrCondition,
    Percentile,
    RollingAgg,
    RollingCondition,
    Runup,
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


def _md(df: pd.DataFrame, externals=()) -> MarketData:
    """Single-target MarketData from a fixture DataFrame (``_ext_<name>`` columns become feeds)."""

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
        return ref.ema(_ref_series(node.input, df), node.window)
    if t == "zscore":
        src = _ref_series(node.input, df)
        return (
            ref.zscore_sma(src, node.window)
            if node.mean_type == "sma"
            else ref.zscore_ema(src, node.window)
        )
    if t == "percentile":
        return ref.percentile(_ref_series(node.input, df), node.window)
    if t == "rolling_agg":
        src = _ref_series(node.input, df)
        if node.window is None:
            return ref.expanding(src, node.agg)
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
    raise NotImplementedError(t)


#: Only always-finite leaves are unconditionally initialized: data leaves —
#: ``field``/``external`` — latch on their first finite value like every transform, so a LEADING
#: NaN is warmup and a later one is an in-data hole.
_LEAF_ALWAYS_INIT = ("constant", "calendar")


def _ref_init(node, df) -> np.ndarray:
    """Latch-based init mask matching ``vectorize._latch``: always-finite leaves are always
    initialized; every other Series node is initialized once (and forever after) its value first
    turns finite."""
    if node.type in _LEAF_ALWAYS_INIT:
        return np.ones(len(df), dtype=bool)
    values = _ref_series(node, df)
    return np.maximum.accumulate((~np.isnan(values)).astype(np.int8)).astype(bool)


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
    from seikan.dsl.traverse import _iter_sweep_axis_names

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
    ]
    for entry in trees:
        assert _iter_sweep_axis_names(entry) == [lvl for lvl, _ in vz.collect_sweeps(entry)]


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
