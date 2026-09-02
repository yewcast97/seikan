"""Parity tests: the consolidated DSL kernels (``compiler/nb.py``) must reproduce the frozen
reference series (``tests/_reference_nb.py``) exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from seikan.compiler import nb

from . import _reference_nb as ref


def _rand(n: int, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randn(n).astype(np.float64)


def _assert(got, want, rtol=1e-9):
    np.testing.assert_allclose(got, want, rtol=rtol, atol=1e-9, equal_nan=True)


# ---- single-column helpers over the production 2D ``*_apply_nb`` surface -------------------
#
# nb.py ships only the 2D forms for the numpy kernels, so the 1D wrappers are test-only
# scaffolding and live here; each helper routes one series through the same production code path
# the engine uses, so a parity assertion below still pins the shipped implementation.


def _c1(arr) -> np.ndarray:
    return np.asarray(arr, dtype=float).reshape(-1, 1)


def zscore_sma_1d(arr, window):
    return nb.zscore_sma_apply_nb(_c1(arr), window).reshape(-1)


def percentile_1d(arr, window):
    return nb.percentile_apply_nb(_c1(arr), window).reshape(-1)


def rolling_agg_1d(arr, window, agg):
    return nb.rolling_agg_apply_nb(_c1(arr), window, agg).reshape(-1)


def expanding_agg_1d(arr, agg):
    return nb.expanding_agg_apply_nb(_c1(arr), agg).reshape(-1)


def drawdown_1d(arr, window=None):
    return nb.drawdown_apply_nb(_c1(arr), window).reshape(-1)


def runup_1d(arr, window=None):
    return nb.runup_apply_nb(_c1(arr), window).reshape(-1)


def bars_since_extremum_1d(arr, extremum="max", window=None):
    return nb.bars_since_extremum_apply_nb(_c1(arr), extremum=extremum, window=window).reshape(-1)


def rolling_corr_1d(left, right, window):
    return nb.rolling_corr_apply_nb(_c1(left), _c1(right), window).reshape(-1)


def change_1d(arr, periods, kind="pct"):
    return nb.change_apply_nb(_c1(arr), periods, kind).reshape(-1)


def shift_1d(arr, periods):
    return nb.shift_apply_nb(_c1(arr), periods).reshape(-1)


def unary_op_1d(arr, op):
    return nb.unary_op_apply_nb(_c1(arr), op).reshape(-1)


# ---- transforms vs reference ----------------------------------------------


def test_zscore_sma_matches_reference():
    arr = _rand(200)
    _assert(zscore_sma_1d(arr, 20), ref.zscore_sma(arr, 20))


def test_zscore_ema_matches_reference():
    arr = _rand(200)
    _assert(nb.zscore_ema_1d(arr, 20), ref.zscore_ema(arr, 20))


def test_percentile_matches_reference():
    arr = _rand(150, seed=4)
    _assert(percentile_1d(arr, 30), ref.percentile(arr, 30))


@pytest.mark.parametrize("agg", ["max", "min", "mean", "std"])
def test_rolling_agg_matches_reference(agg):
    arr = (
        _rand(180, seed=5) * 10.0 + 100.0
    )  # price-scale magnitudes exercise std's cancellation risk
    _assert(rolling_agg_1d(arr, 20, agg), ref.rolling_agg(arr, 20, agg))


def test_rolling_agg_flat_window_std_is_zero():
    # A flat window has zero population variance — std must be 0 (not NaN from sqrt of a tiny
    # negative).
    arr = np.full(40, 7.0)
    _assert(rolling_agg_1d(arr, 10, "std"), ref.rolling_agg(arr, 10, "std"))
    assert np.nanmax(np.abs(rolling_agg_1d(arr, 10, "std"))) == 0.0


def test_rolling_agg_nan_window_is_censored():
    # Any NaN in the trailing window censors the output (matches percentile's finite-window gate).
    arr = _rand(60, seed=6)
    arr[25] = np.nan
    got = rolling_agg_1d(arr, 5, "max")
    _assert(got, ref.rolling_agg(arr, 5, "max"))
    assert np.all(np.isnan(got[25:30]))  # every window covering bar 25 is NaN


# ---- change (pct / log / diff) vs reference --------------------------------


def test_change_pct_matches_reference():
    arr = np.cumsum(np.abs(_rand(120))) + 100.0
    _assert(change_1d(arr, 5, "pct"), ref.change(arr, 5, "pct"))


def test_change_log_matches_reference():
    arr = np.array([1.0, 2.0, -1.0, 4.0, 0.0, 8.0, 16.0, 2.0])
    _assert(change_1d(arr, 1, "log"), ref.change(arr, 1, "log"))


def test_change_diff_matches_reference():
    arr = np.array([1.0, 2.0, -1.0, 4.0, 0.0, 8.0, 16.0, 2.0])
    _assert(change_1d(arr, 1, "diff"), ref.change(arr, 1, "diff"))


def test_change_diff_allows_zero_base():
    # pct/log NaN on a zero/non-positive base; diff only needs both bars finite (0 is a valid
    # level).
    arr = np.array([0.0, 5.0, np.nan, 3.0])
    got = change_1d(arr, 1, "diff")
    want = np.array([np.nan, 5.0, np.nan, np.nan])
    _assert(got, want)
    assert np.isnan(change_1d(arr, 1, "pct")[1])


def test_change_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown change kind"):
        nb.change_apply_nb(np.zeros((5, 1)), 1, "bogus")


# ---- ema vs reference -------------------------------------------------------


def test_ema_matches_reference():
    arr = _rand(200)
    _assert(nb.ema_1d(arr, 20), ref.ema(arr, 20))


def test_ema_skips_nan_and_seeds_first_finite():
    arr = np.array([np.nan, np.nan, 5.0, 6.0, np.nan, 7.0, 8.0])
    _assert(nb.ema_1d(arr, 3), ref.ema(arr, 3))


# ---- 2D apply forms are columnwise (no cross-column leakage), pinned against the reference --


def test_apply_nb_is_columnwise_1d():
    rng = np.random.RandomState(2)
    a = rng.randn(120, 3)
    got = nb.zscore_sma_apply_nb(a, 20)
    for j in range(3):
        _assert(got[:, j], ref.zscore_sma(a[:, j], 20))


def test_numpy_apply_nb_is_columnwise():
    # Each column of a multi-column apply must equal the frozen 1D reference on that column alone —
    # columnwise purity AND correctness in one assertion (a wrapper comparison would be circular,
    # the single-column helpers themselves routing through the 2D forms).
    rng = np.random.RandomState(12)
    a = np.abs(rng.randn(120, 3)) + 1.0  # strictly positive for change("log")
    for j in range(3):
        _assert(nb.percentile_apply_nb(a, 15)[:, j], ref.percentile(a[:, j], 15))
        _assert(nb.rolling_agg_apply_nb(a, 12, "std")[:, j], ref.rolling_agg(a[:, j], 12, "std"))
        for kind in ("pct", "log", "diff"):
            _assert(nb.change_apply_nb(a, 5, kind)[:, j], ref.change(a[:, j], 5, kind))
        _assert(nb.ema_apply_nb(a, 10)[:, j], ref.ema(a[:, j], 10))
        _assert(nb.shift_apply_nb(a, 3)[:, j], ref.shift_ref(a[:, j], 3))


# ---- rolling_corr vs scipy/pandas reference --------------------------------


def _rolling_corr_pandas_ref(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    import pandas as pd

    sa, sb = pd.Series(a), pd.Series(b)
    corr = sa.rolling(window, min_periods=window).corr(sb)
    # pandas' rolling .corr degenerates (0/0 or ±1 pathologically) on a zero-variance window;
    # the schema's contract is NaN there — mask those windows explicitly for the comparison.
    win_a = sliding_window_view(a, window)
    win_b = sliding_window_view(b, window)
    finite = ~(np.isnan(win_a) | np.isnan(win_b)).any(axis=-1)
    zero_var = (np.nanstd(win_a, axis=-1) == 0.0) | (np.nanstd(win_b, axis=-1) == 0.0)
    ok = finite & ~zero_var
    out = corr.to_numpy().copy()
    out[window - 1 :] = np.where(ok, out[window - 1 :], np.nan)
    return out


def test_rolling_corr_matches_pandas_reference():
    rng = np.random.RandomState(9)
    a = rng.randn(150)
    b = 0.6 * a + rng.randn(150) * 0.4  # correlated but not degenerate
    got = rolling_corr_1d(a, b, 20)
    want = _rolling_corr_pandas_ref(a, b, 20)
    _assert(got, want)


def test_rolling_corr_matches_reference_module():
    rng = np.random.RandomState(10)
    a = rng.randn(120)
    b = rng.randn(120)
    got = rolling_corr_1d(a, b, 15)
    want = ref.rolling_corr_ref(a, b, 15)
    _assert(got, want)


def test_rolling_corr_zero_variance_is_nan_never_pm_one():
    # A flat window has zero variance in one leg — correlation is undefined (NaN), never ±1.
    a = np.concatenate([np.array([1.0, 2.0, 3.0]), np.full(10, 7.0)])
    b = np.random.RandomState(0).randn(13)
    got = rolling_corr_1d(a, b, 5)
    # windows fully inside the flat run (index >= 3 + window - 1) are degenerate
    assert np.all(np.isnan(got[7:]))


def test_rolling_corr_window_below_three_is_all_nan():
    a = np.random.RandomState(1).randn(20)
    b = np.random.RandomState(2).randn(20)
    assert np.isnan(rolling_corr_1d(a, b, 2)).all()
    assert np.isnan(rolling_corr_1d(a, b, 1)).all()


def test_rolling_corr_apply_nb_2d_matches_1d_per_column():
    rng = np.random.RandomState(11)
    a = rng.randn(80, 3)
    b = rng.randn(80, 3)
    got = nb.rolling_corr_apply_nb(a, b, 10)
    for j in range(3):
        _assert(got[:, j], rolling_corr_1d(a[:, j], b[:, j], 10))


def test_rolling_corr_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        nb.rolling_corr_apply_nb(np.zeros((10, 2)), np.zeros((10, 3)), 5)


# ---- Phase-1 kernels: shift / unary_op --------------------------------------


def test_shift_matches_reference():
    arr = _rand(120, seed=7)
    arr[10] = np.nan
    for p in (1, 3, 17):
        _assert(shift_1d(arr, p), ref.shift_ref(arr, p))


def test_shift_beyond_length_is_all_nan():
    arr = _rand(10)
    assert np.isnan(shift_1d(arr, 10)).all()


def test_unary_op_known_answers():
    arr = np.array([-4.0, -1.0, 0.0, 1.0, 9.0, np.nan])
    _assert(unary_op_1d(arr, "abs"), np.array([4.0, 1.0, 0.0, 1.0, 9.0, np.nan]))
    _assert(unary_op_1d(arr, "neg"), np.array([4.0, 1.0, -0.0, -1.0, -9.0, np.nan]))
    _assert(unary_op_1d(arr, "sign"), np.array([-1.0, -1.0, 0.0, 1.0, 1.0, np.nan]))
    # domain: log needs > 0, sqrt needs >= 0 — out-of-domain maps to NaN (never fires)
    _assert(unary_op_1d(arr, "log"), np.array([np.nan, np.nan, np.nan, 0.0, np.log(9.0), np.nan]))
    _assert(unary_op_1d(arr, "sqrt"), np.array([np.nan, np.nan, 0.0, 1.0, 3.0, np.nan]))


def test_unary_op_unknown_op_raises():
    with pytest.raises(ValueError, match="unknown unary op"):
        nb.unary_op_apply_nb(np.zeros((2, 1)), "exp")


# ---- cross-sectional kernels vs pandas reference ---------------------------
#
# CrossRank/CrossDemean/CrossAgg operate ACROSS columns at each row (axis=1) — the only kernels
# with no 1-D form. The oracle is pandas' row-wise rank/mean with an explicit row-broadcast
# min_valid mask (and the frozen row-loop ``cross_agg_ref`` for the aggregate).


def _cross_rank_ref(a: np.ndarray, min_valid: int) -> np.ndarray:
    import pandas as pd

    df = pd.DataFrame(a)
    k = df.notna().sum(axis=1).to_numpy()
    frac = (df.rank(axis=1, method="average") - 1).div(np.maximum(k - 1, 1), axis=0).to_numpy()
    ok = df.notna().to_numpy() & (k >= max(min_valid, 2))[:, None]
    return np.where(ok, frac, np.nan)


def _cross_demean_ref(a: np.ndarray, min_valid: int) -> np.ndarray:
    import pandas as pd

    df = pd.DataFrame(a)
    k = df.notna().sum(axis=1).to_numpy()
    dm = df.sub(df.mean(axis=1), axis=0).to_numpy()
    ok = df.notna().to_numpy() & (k >= max(min_valid, 2))[:, None]
    return np.where(ok, dm, np.nan)


def _nan_riddled(rows: int = 60, cols: int = 5, seed: int = 7) -> np.ndarray:
    rng = np.random.RandomState(seed)
    a = rng.randn(rows, cols)
    a[rng.rand(rows, cols) < 0.2] = np.nan  # scattered NaNs
    a[10, :] = np.nan  # an all-NaN row
    a[20, 1:] = np.nan  # a single-survivor row (k=1 < any min_valid)
    return a


def test_cross_rank_matches_pandas_reference():
    a = _nan_riddled()
    for mv in (2, 3, 5):
        _assert(nb.cross_rank_apply_nb(a, mv), _cross_rank_ref(a, mv))


def test_cross_rank_handles_ties():
    a = np.array([[1.0, 1.0, 2.0], [3.0, 3.0, 3.0], [1.0, 2.0, 2.0]])
    _assert(nb.cross_rank_apply_nb(a, 2), _cross_rank_ref(a, 2))
    # a fully tied row ranks everyone at the midpoint 0.5
    _assert(nb.cross_rank_apply_nb(a, 2)[1], np.full(3, 0.5))


def test_cross_rank_bounds_and_orientation():
    a = np.array([[10.0, 30.0, 20.0]])
    got = nb.cross_rank_apply_nb(a, 2)[0]
    _assert(got, np.array([0.0, 1.0, 0.5]))  # lowest → 0, highest → 1


def test_cross_rank_excludes_non_finite_from_the_cross_section():
    # HAND-COMPUTED on purpose — the pandas rank oracle above walks into the same trap: `notna()`
    # counts ±inf as a member while `k` counts FINITE values, so an unmasked -inf is RANKED but
    # not counted, taking rank 1 and pushing the finite members' fraction ranks to [1/(k-1),
    # (k)/(k-1)] — outside [0, 1] at the top. The kernel ranks `np.where(finite, a, np.nan)`:
    # ±inf leaves the cross-section exactly as it leaves the denominator, so the k=2 finite
    # members rank [0.0, 1.0] and the non-finite columns are NaN like any hole.
    row = np.array([[1.0, -np.inf, 3.0, np.nan]])
    _assert(nb.cross_rank_apply_nb(row, 2), np.array([[0.0, np.nan, 1.0, np.nan]]))
    # an all-±inf row has NO finite cross-section: k = 0 < min_valid → all NaN, never a ranking
    all_inf = np.array([[np.inf, -np.inf, np.inf]])
    assert np.isnan(nb.cross_rank_apply_nb(all_inf, 2)).all()


def test_cross_demean_matches_pandas_reference():
    a = _nan_riddled(seed=8)
    for mv in (2, 4):
        _assert(nb.cross_demean_apply_nb(a, mv), _cross_demean_ref(a, mv))


def test_cross_kernels_single_column_is_all_nan():
    a = np.random.RandomState(0).randn(30, 1)
    assert np.isnan(nb.cross_rank_apply_nb(a, 2)).all()
    assert np.isnan(nb.cross_demean_apply_nb(a, 2)).all()


@pytest.mark.parametrize("agg", ["mean", "median", "std", "frac_positive"])
def test_cross_agg_matches_reference(agg):
    rng = np.random.RandomState(11)
    arr = rng.randn(80, 5)
    arr[rng.rand(80, 5) < 0.15] = np.nan  # sparse cross-sections exercise the min_valid gate
    for mv in (2, 4):
        _assert(nb.cross_agg_apply_nb(arr, agg, mv), ref.cross_agg_ref(arr, agg, mv))


def test_cross_agg_broadcasts_to_nan_columns():
    # The aggregate is a property of the cross-section: a column whose own value is NaN still
    # carries the group's value (unlike cross_rank/cross_demean).
    arr = np.array([[1.0, np.nan, 3.0]])
    out = nb.cross_agg_apply_nb(arr, "mean", 2)
    np.testing.assert_allclose(out[0], [2.0, 2.0, 2.0])


def test_cross_agg_single_column_is_all_nan():
    assert np.isnan(nb.cross_agg_apply_nb(np.ones((5, 1)), "mean", 2)).all()


# ---- expanding agg / drawdown / first_true (rebound primitives) -------------


def test_expanding_max_skips_nan_prefix():
    arr = np.array([np.nan, np.nan, 5.0, 3.0, 7.0, np.nan, 6.0])
    got = expanding_agg_1d(arr, "max")
    want = np.array([np.nan, np.nan, 5.0, 5.0, 7.0, 7.0, 7.0])
    _assert(got, want)


def test_expanding_min_skips_nan_prefix():
    arr = np.array([np.nan, 4.0, 2.0, np.nan, 3.0])
    got = expanding_agg_1d(arr, "min")
    want = np.array([np.nan, 4.0, 2.0, 2.0, 2.0])
    _assert(got, want)


def test_expanding_agg_rejects_mean():
    with pytest.raises(ValueError, match="only supports"):
        expanding_agg_1d(np.ones(5), "mean")


def test_drawdown_expanding_known_answers():
    # peak path: 100 → 100 → 80 → 120 → 90  ⇒ dd: 0, 0, -0.2, 0, -0.25
    arr = np.array([100.0, 100.0, 80.0, 120.0, 90.0])
    got = drawdown_1d(arr, None)
    want = np.array([0.0, 0.0, -0.2, 0.0, -0.25])
    _assert(got, want)


def test_drawdown_trailing_matches_composed_ratio():
    rng = np.random.RandomState(3)
    arr = 100 + rng.randn(80).cumsum()
    window = 10
    peak = rolling_agg_1d(arr, window, "max")
    with np.errstate(invalid="ignore", divide="ignore"):
        want = np.where(np.isfinite(arr) & np.isfinite(peak) & (peak > 0), arr / peak - 1.0, np.nan)
    _assert(drawdown_1d(arr, window), want)


def test_drawdown_nan_prefix_stays_nan():
    arr = np.array([np.nan, np.nan, 10.0, 8.0])
    got = drawdown_1d(arr, None)
    assert np.isnan(got[0]) and np.isnan(got[1])
    _assert(got[2:], np.array([0.0, -0.2]))


def test_first_true_fires_on_false_to_true_only():
    value = np.array([False, False, True, True, False, True])
    init = np.ones(6, dtype=bool)
    got, _defined = nb.first_true_1d(value, init, np.ones(6, dtype=bool), 0)
    want = np.array([False, False, True, False, False, True])
    np.testing.assert_array_equal(got, want)


def test_first_true_warmup_does_not_phantom_fire():
    # Regime already true when init latches — must NOT fire until a later False→True.
    value = np.array([True, True, True, False, True])
    init = np.array([False, False, True, True, True])
    got, _defined = nb.first_true_1d(value, init, np.ones(5, dtype=bool), 0)
    want = np.array([False, False, False, False, True])
    np.testing.assert_array_equal(got, want)


def test_first_true_cooldown_suppresses_refires():
    value = np.array([False, True, False, True, False, True])
    init = np.ones(6, dtype=bool)
    got, _defined = nb.first_true_1d(
        value, init, np.ones(6, dtype=bool), 2
    )  # suppress 2 bars after each fire
    # fire at 1; bars 2,3 cooldown; bar 3 is True but suppressed; next eligible False→True at 5
    want = np.array([False, True, False, False, False, True])
    np.testing.assert_array_equal(got, want)


def test_first_true_apply_nb_is_columnwise():
    value = np.array([[False, False], [True, False], [True, True], [False, True]])
    init = np.ones((4, 2), dtype=bool)
    defined = np.ones((4, 2), dtype=bool)
    got, _def_out = nb.first_true_apply_nb(value, init, defined, 0)
    for j in range(2):
        np.testing.assert_array_equal(
            got[:, j], nb.first_true_1d(value[:, j], init[:, j], defined[:, j], 0)[0]
        )


# ---- runup / bars_since_extremum (left-side mean-reversion primitives) ------


def test_runup_expanding_known_answers():
    # trough path: 100 → 80 → 80 → 120 → 90  ⇒ ru: 0, 0, 0, 0.5, 0.125
    arr = np.array([100.0, 80.0, 80.0, 120.0, 90.0])
    got = runup_1d(arr, None)
    want = np.array([0.0, 0.0, 0.0, 0.5, 0.125])
    _assert(got, want)


def test_runup_trailing_matches_composed_ratio():
    rng = np.random.RandomState(5)
    arr = 100 + rng.randn(80).cumsum()
    window = 10
    trough = rolling_agg_1d(arr, window, "min")
    with np.errstate(invalid="ignore", divide="ignore"):
        want = np.where(
            np.isfinite(arr) & np.isfinite(trough) & (trough > 0), arr / trough - 1.0, np.nan
        )
    _assert(runup_1d(arr, window), want)


def test_runup_matches_reference():
    rng = np.random.RandomState(6)
    arr = 50 + rng.randn(100).cumsum()
    arr[10] = np.nan
    _assert(runup_1d(arr, None), ref.runup(arr, None))
    _assert(runup_1d(arr, 8), ref.runup(arr, 8))


def test_bars_since_extremum_expanding_max_resets_on_retest():
    # peaks: 10, 10 (tie→reset), 8, 12, 11  ⇒ since-max: 0, 0, 1, 0, 1
    arr = np.array([10.0, 10.0, 8.0, 12.0, 11.0])
    got = bars_since_extremum_1d(arr, "max", None)
    want = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
    _assert(got, want)


def test_bars_since_extremum_expanding_min_stabilization():
    # troughs: 10, 8, 8 (tie), 9, 7  ⇒ since-min: 0, 0, 0, 1, 0
    arr = np.array([10.0, 8.0, 8.0, 9.0, 7.0])
    got = bars_since_extremum_1d(arr, "min", None)
    want = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    _assert(got, want)


def test_bars_since_extremum_trailing_nan_gate():
    arr = np.array([1.0, 3.0, 2.0, np.nan, 4.0, 1.0])
    got = bars_since_extremum_1d(arr, "max", 3)
    # window full at i=2: max=3 at i=1 → since=1; i=3 NaN window; i=4 NaN in win;
    # i=5 win=[nan,4,1] NaN
    assert np.isnan(got[0]) and np.isnan(got[1])
    _assert(got[2:3], np.array([1.0]))
    assert np.isnan(got[3]) and np.isnan(got[4]) and np.isnan(got[5])


def test_bars_since_extremum_matches_reference():
    rng = np.random.RandomState(7)
    arr = 100 + rng.randn(60).cumsum()
    arr[15] = np.nan
    for ext in ("max", "min"):
        _assert(
            bars_since_extremum_1d(arr, ext, None),
            ref.bars_since_extremum(arr, ext, None),
        )
        _assert(
            bars_since_extremum_1d(arr, ext, 5),
            ref.bars_since_extremum(arr, ext, 5),
        )


def test_bars_since_extremum_apply_nb_is_columnwise():
    arr = np.column_stack([np.array([5.0, 4.0, 6.0, 3.0]), np.array([1.0, 2.0, 1.5, 0.5])])
    got = nb.bars_since_extremum_apply_nb(arr, "max", None)
    for j in range(2):
        _assert(got[:, j], bars_since_extremum_1d(arr[:, j], "max", None))


# ---- numerical stability + the finite-or-NaN invariant -----------------------------------------


@pytest.mark.parametrize("base", [0.0, 1e8, 1e12])
def test_zscore_sma_translation_invariant_across_levels(base):
    # The one-pass E[x²]−E[x]² variance lost its mantissa at large levels (all-NaN at 1e12);
    # the centered two-pass form is level-free up to input quantization (~eps·base/σ).
    r = _rand(200, seed=11)
    z0 = zscore_sma_1d(r, 20)
    z = zscore_sma_1d(r + base, 20)
    tol = 1e-9 if base == 0.0 else (1e-6 if base <= 1e8 else 1e-3)
    np.testing.assert_allclose(z, z0, rtol=0, atol=tol, equal_nan=True)
    assert np.isfinite(z[19:]).all()


@pytest.mark.parametrize("base", [0.0, 1e8, 1e12])
def test_zscore_ema_translation_invariant_across_levels(base):
    r = _rand(200, seed=12)
    z0 = nb.zscore_ema_1d(r, 20)
    z = nb.zscore_ema_1d(r + base, 20)
    tol = 1e-9 if base == 0.0 else (1e-6 if base <= 1e8 else 1e-3)
    np.testing.assert_allclose(z, z0, rtol=0, atol=tol, equal_nan=True)
    assert np.isfinite(z[19:]).all()


@pytest.mark.parametrize("base", [0.0, 1e8, 1e12])
def test_cross_agg_std_translation_invariant_across_levels(base):
    rng = np.random.RandomState(13)
    a = rng.randn(50, 4)
    s0 = nb.cross_agg_apply_nb(a, "std", 2)
    s = nb.cross_agg_apply_nb(a + base, "std", 2)
    tol = 1e-9 if base == 0.0 else (1e-6 if base <= 1e8 else 1e-3)
    np.testing.assert_allclose(s, s0, rtol=0, atol=tol, equal_nan=True)
    # The one-pass form clamped the cancelled variance to a silent 0.0 std at large levels,
    # which the documented z-score recipe then divided by; the centered form stays positive.
    assert np.isfinite(s).all() and (s > 0).all()


def test_kernel_overflow_yields_nan_never_inf():
    # Finite inputs whose arithmetic overflows (or divides by a denormal) must emit NaN — the
    # undecidable ledger's food — never ±inf, which would latch warmup and decide thresholds.
    huge = np.full(30, 1e308)
    probes = [
        zscore_sma_1d(huge, 4),
        rolling_agg_1d(huge, 4, "mean"),
        rolling_agg_1d(np.tile([1e308, -1e308], 15), 4, "std"),
        change_1d(np.array([1e-320, 1.0, 1.0]), 1, "pct"),
        change_1d(np.array([1e308, -1e308, 1e308]), 1, "diff"),
        drawdown_1d(np.array([1e-320, -1e308])),
        runup_1d(np.array([1e-320, 1e308])),
        rolling_corr_1d(np.tile([0.0, 1e-160], 6), np.tile([0.0, 1e-160], 6), 4),
        nb.zscore_ema_1d(np.array([0.0, 2e154, 1.0, 1.0, 1.0]), 2),
        nb.cross_demean_apply_nb(np.array([[1.7e308, -1.7e308, -1.7e308]]), 2).ravel(),
        nb.cross_agg_apply_nb(np.array([[1.6e308, 1.6e308, 1.6e308]]), "mean", 2).ravel(),
        nb.cross_agg_apply_nb(np.array([[1.6e308, -1.6e308, 1.6e308]]), "std", 2).ravel(),
    ]
    for got in probes:
        assert not np.isinf(got).any()


def test_change_pct_negative_endpoints_are_nan():
    # −4 → −2 read as −50% under the old `prev != 0` rule while the level ROSE; now any pair
    # with a non-positive endpoint censors, and only 2 → 4 (both strictly positive) measures.
    arr = np.array([-4.0, -2.0, 2.0, 4.0, -1.0])
    got = change_1d(arr, 1, "pct")
    _assert(got, np.array([np.nan, np.nan, np.nan, 1.0, np.nan]))
    _assert(got, ref.change(arr, 1, "pct"))


def test_drawdown_runup_nonpositive_extremum_is_nan():
    # The audit's probe: on [-1, -2, -0.5] the unguarded ratio emitted drawdown [0, 1, 0] — the
    # deepest bar reading as a +100% run-up — and runup [0, 0, -0.75]. Non-positive extremum
    # now censors, and the documented bounds hold everywhere a value is emitted.
    signed = np.array([-1.0, -2.0, -0.5])
    assert np.isnan(drawdown_1d(signed)).all()
    assert np.isnan(runup_1d(signed)).all()
    crossing = np.array([-1.0, 2.0, 4.0])  # expanding peak [-1, 2, 4]; trough [-1, -1, -1]
    _assert(drawdown_1d(crossing), np.array([np.nan, 0.0, 0.0]))
    assert np.isnan(runup_1d(crossing)).all()
    px = np.array([3.0, 2.0, 4.0, 1.0, 5.0])
    dd, ru = drawdown_1d(px), runup_1d(px)
    assert (dd[np.isfinite(dd)] <= 0).all() and (ru[np.isfinite(ru)] >= 0).all()
