"""Parity tests: the consolidated DSL kernels (``compiler/nb.py``) must reproduce the frozen
reference series (``tests/_reference_nb.py``) exactly.
"""

from __future__ import annotations

import math
import time
from fractions import Fraction

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
    _assert(nb.zscore_ema_1d(arr, *nb.ema_params(20, None)), ref.zscore_ema(arr, 20))


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


def _rtol(got, want):
    """Parity at ANY scale — ``_assert``'s atol would pass every value under 1e-9."""
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=0.0, equal_nan=True)


@pytest.mark.parametrize("k", [536, 565, 1000])
def test_varying_window_whose_variance_underflows_reads_its_exact_std_never_zero(k):
    # Below the normal range a window's squared deviations underflow: at 2**-536 they land in the
    # SUBNORMAL range — the std of (1, 3, 2)·2**-536 read 3.85e-162 where it is 3.63e-162 — and
    # from ~2**-540 down they vanish, the VARYING window's std reading the exact 0.0 an all-equal
    # window owns (`std > 0` decided False on moving data). Lifted by a power of two the moments
    # are computed in range, so the std is the unscaled one times 2**-k, every bit.
    x = np.array([1.0, 3.0, 2.0, 5.0])
    s = 2.0**-k
    want = rolling_agg_1d(x, 3, "std") * s
    np.testing.assert_array_equal(rolling_agg_1d(x * s, 3, "std"), want)
    _rtol(ref.rolling_agg(x * s, 3, "std"), want)
    row = x[None, :3]
    want_row = nb.cross_agg_apply_nb(row, "std", 2) * s
    np.testing.assert_array_equal(nb.cross_agg_apply_nb(row * s, "std", 2), want_row)
    _rtol(ref.cross_agg_ref(row * s, "std", 2), want_row)
    # the all-equal window keeps its exact 0.0 at the same scale
    flat = np.full(4, s)
    np.testing.assert_array_equal(rolling_agg_1d(flat, 3, "std"), [np.nan, np.nan, 0.0, 0.0])
    np.testing.assert_array_equal(nb.cross_agg_apply_nb(flat[None, :3], "std", 2), np.zeros((1, 3)))


def test_subnormal_windows_read_the_unscaled_z_and_a_std_floored_above_zero():
    # Windows of SUBNORMAL values — an ema settling onto its floor walks 3, 2, 1 smallest steps:
    # every square underflowed to 0.0, so each varying window read std 0.0 and z NaN, the
    # all-equal window's answers. Lifted, the z-score is the unscaled windows' z, bit for bit;
    # the std is rounded once to the subnormal grid, and where it would round to 0.0 (0.47 of
    # a step) it floors at the smallest subnormal — a varying window is never zero dispersion.
    steps = np.array([3.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    t = steps * 5e-324
    want_std = [np.nan, np.nan, 5e-324, 5e-324, 0.0, 0.0]
    np.testing.assert_array_equal(rolling_agg_1d(t, 3, "std"), want_std)
    np.testing.assert_array_equal(ref.rolling_agg(t, 3, "std"), want_std)
    want_z = zscore_sma_1d(steps, 3)  # [nan, nan, -√1.5, -√0.5, nan, nan]
    np.testing.assert_allclose(want_z[2:4], [-math.sqrt(1.5), -math.sqrt(0.5)], rtol=1e-15)
    np.testing.assert_array_equal(zscore_sma_1d(t, 3), want_z)
    _rtol(ref.zscore_sma(t, 3), want_z)
    # a two-member cross-section one step apart: std half a step, a tie that rounds to 0.0
    row = np.array([[1.0, 2.0]]) * 5e-324
    np.testing.assert_array_equal(nb.cross_agg_apply_nb(row, "std", 2), [[5e-324, 5e-324]])
    np.testing.assert_array_equal(ref.cross_agg_ref(row, "std", 2), [[5e-324, 5e-324]])


def test_std_keeps_every_bit_while_the_variance_is_a_normal_double():
    # The out-of-range rule binds only below the normal range: at 2**-500 (deviations ~1e-151,
    # variances ~1e-301) every moment scales exactly by its power of two, so both stds are the
    # unscaled bits times 2**-500 — no window the direct form represented moves.
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    s = 2.0**-500
    np.testing.assert_array_equal(rolling_agg_1d(x * s, 3, "std"), rolling_agg_1d(x, 3, "std") * s)
    np.testing.assert_array_equal(
        nb.cross_agg_apply_nb(x[None, :] * s, "std", 2),
        nb.cross_agg_apply_nb(x[None, :], "std", 2) * s,
    )


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
    _assert(nb.ema_1d(arr, *nb.ema_params(20, None)), ref.ema(arr, 20))


def test_ema_skips_nan_and_seeds_first_finite():
    arr = np.array([np.nan, np.nan, 5.0, 6.0, np.nan, 7.0, 8.0])
    _assert(nb.ema_1d(arr, *nb.ema_params(3, None)), ref.ema(arr, 3))


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
        _assert(nb.ema_apply_nb(a, *nb.ema_params(10, None))[:, j], ref.ema(a[:, j], 10))
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


@pytest.mark.parametrize(
    "scale", [1e80, 1e150, 2.0**500, 1e-85, 1e-150, 2.0**-500, 1e-160, 2.0**-1000, 5e-324]
)
def test_rolling_corr_of_a_series_with_itself_is_exactly_one_at_any_scale(scale):
    # The review's probe: at 1e80 the variance PRODUCT overflowed and `cov / inf` read a finite
    # 0.0 — a perfect correlation reported as none; at 1e-85 it underflowed to 0 and read NaN.
    # Taken with the exponents moved out, the denominator is the direct form's exact value
    # (sqrt(va²) == va), so corr(x, x) is exactly 1.0 at every scale whose variances are normal
    # doubles (deviations up to ~1.3e154), as it is at 1 — and below that range (1e-160 down to
    # subnormal values, where the variances themselves underflowed and read NaN) the lifted
    # window's variances are normal again.
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0])
    want = [np.nan, np.nan] + [1.0] * 5
    np.testing.assert_array_equal(rolling_corr_1d(scale * x, scale * x, 3), want)
    np.testing.assert_array_equal(ref.rolling_corr_ref(scale * x, scale * x, 3), want)


@pytest.mark.parametrize("k", [270, 500, -280, -500])
def test_rolling_corr_is_exactly_scale_invariant_past_the_variance_product_range(k):
    # Scaling both legs by 2**k scales every moment exactly, so the exponent-safe denominator
    # returns the SAME bits as the unscaled windows — with the variance product near 2**(4k),
    # far past the double range both ways (the direct form read 0.0 above it, NaN below).
    rng = np.random.RandomState(14)
    a = rng.randn(60)
    b = 0.5 * a + rng.randn(60)
    base = rolling_corr_1d(a, b, 10)
    assert np.isfinite(base[9:]).all()
    s = 2.0**k
    np.testing.assert_array_equal(rolling_corr_1d(a * s, b * s, 10), base)
    _assert(ref.rolling_corr_ref(a * s, b * s, 10), base)


def test_rolling_corr_overflowed_variance_is_nan_never_zero():
    # corr(1e160·x, x): the LEFT variance itself overflows (deviations past ~1.3e154), and
    # `cov / sqrt(inf · vb)` read a finite 0.0 — "uncorrelated" for a perfectly correlated pair.
    # A variance past the double range is no dispersion estimate: NaN, the undecidable ledger.
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 7.0])
    for left, right in ((1e160 * x, x), (x, 1e160 * x)):
        np.testing.assert_array_equal(rolling_corr_1d(left, right, 3), np.full(6, np.nan))
        np.testing.assert_array_equal(ref.rolling_corr_ref(left, right, 3), np.full(6, np.nan))


def test_rolling_corr_with_an_underflowed_variance_reads_the_unscaled_correlation():
    # Deviations of ~1e-161 square into the SUBNORMAL range: the variances (and the covariance)
    # rounded to a few multiples of 5e-324. Divided by the direct product where that stayed
    # normal (a wide right leg), they read 0.99745 for 0.99968; at 2**-1000 every square vanished
    # and the window read NaN. The leg whose variance underflowed is lifted by a power of two —
    # the other leg as it is — so each pair reads the unscaled windows' correlation, every bit.
    a = np.array([1.0, -1.0, 2.0, 0.5, -0.5, 1.5])
    b = np.array([1.1, -0.9, 2.0, 0.4, -0.6, 1.2])
    exact = rolling_corr_1d(a, b, 3)
    assert np.isfinite(exact[2:]).all()  # the unscaled windows are ordinary
    pairs = (
        (a * 2.0**-535, b * 2.0**60),
        (a * 2.0**-535, b * 2.0**-535),
        (a, b * 2.0**-1000),
        (a * 2.0**-1000, b * 2.0**-600),
    )
    for left, right in pairs:
        np.testing.assert_array_equal(rolling_corr_1d(left, right, 3), exact)
        _rtol(ref.rolling_corr_ref(left, right, 3), exact)
    # the same windows one exponent step up — variances normal — keep the unscaled bits exactly
    np.testing.assert_array_equal(rolling_corr_1d(a * 2.0**-500, b * 2.0**-500, 3), exact)


def test_ordinary_windows_are_bit_identical_to_the_direct_forms():
    # The out-of-range rules bind only where a variance left the normal range (or the
    # correlation's variance product did): on ordinary windows — scales from 1e-60 to 1e60 — the
    # correlation, the SMA z-score and both stds keep the direct forms' exact bits.
    rng = np.random.RandomState(15)
    for _ in range(40):
        scale = 10.0 ** rng.uniform(-60.0, 60.0)
        level = scale * rng.choice([0.0, 1.0, 1e4])
        a = _c1(level + scale * rng.randn(240).cumsum())
        b = _c1(level + scale * (rng.randn(240) + 0.3 * rng.randn(240).cumsum()))
        w = int(rng.randint(3, 40))
        wa = sliding_window_view(a, w, axis=0)
        wb = sliding_window_view(b, w, axis=0)
        ma, mb = wa.mean(axis=-1), wb.mean(axis=-1)
        va = ((wa - ma[..., None]) ** 2).mean(axis=-1)
        vb = ((wb - mb[..., None]) ** 2).mean(axis=-1)
        cov = ((wa - ma[..., None]) * (wb - mb[..., None])).mean(axis=-1)
        direct = {
            "corr": cov / np.sqrt(va * vb),
            "z": (a[w - 1 :] - ma) / np.sqrt(va),
            "std": np.sqrt(np.maximum(va, 0.0)),
        }
        assert np.isfinite(direct["corr"]).all() and (va > 0.0).all()
        np.testing.assert_array_equal(nb.rolling_corr_apply_nb(a, b, w)[w - 1 :], direct["corr"])
        np.testing.assert_array_equal(nb.zscore_sma_apply_nb(a, w)[w - 1 :], direct["z"])
        np.testing.assert_array_equal(nb.rolling_agg_apply_nb(a, w, "std")[w - 1 :], direct["std"])
        rows = (level + scale * rng.randn(50, 6)).astype(float)
        dev = rows - rows.sum(axis=1, keepdims=True) / 6.0
        cross = np.sqrt(np.maximum((dev * dev).sum(axis=1, keepdims=True) / 6.0, 0.0))
        np.testing.assert_array_equal(nb.cross_agg_apply_nb(rows, "std", 2), np.repeat(cross, 6, 1))


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
    defined[1, 1] = False  # one column holed: its worlds never leak into the other
    got, def_out = nb.first_true_apply_nb(value, init, defined, 0)
    for j in range(2):
        want, want_defined = nb.first_true_1d(value[:, j], init[:, j], defined[:, j], 0)
        np.testing.assert_array_equal(got[:, j], want)
        np.testing.assert_array_equal(def_out[:, j], want_defined)
    np.testing.assert_array_equal(def_out[:, 0], [True, True, True, True])
    np.testing.assert_array_equal(def_out[:, 1], [True, False, False, True])


def test_first_true_hole_free_is_the_deterministic_machine():
    # Without a hole there is exactly one world, so the kernel IS the deterministic machine, bit
    # for bit, and every bar is decided — across cooldowns and warmup patterns (a leading warmup,
    # warmup re-entered mid-series).
    rng = np.random.RandomState(21)
    for _ in range(400):
        n = int(rng.randint(1, 120))
        cooldown = int(rng.choice([0, 1, 2, 3, 5, 8, 40]))
        value = rng.rand(n) < rng.choice([0.2, 0.5, 0.8])
        init = rng.rand(n) < 0.9
        init[: rng.randint(0, 6)] = False
        got, defined = nb.first_true_1d(value, init, np.ones(n, dtype=bool), cooldown)
        np.testing.assert_array_equal(got, ref.first_true_machine(value, init, cooldown))
        assert defined.all()


def _holed_child(rng: np.random.RandomState, max_holes: int):
    """A random child with a warmup prefix (sometimes re-entered mid-series) and at most
    ``max_holes`` post-warmup holes — placed at random, and often on the first live bar, the last
    bar and an adjacent pair. A hole stamped on a warmup bar is no hole: warmup wins."""
    n = int(rng.randint(1, 28))
    value = rng.rand(n) < rng.choice([0.3, 0.5, 0.7])
    init = np.ones(n, dtype=bool)
    init[: rng.randint(0, 4)] = False
    if n > 6 and rng.rand() < 0.25:
        start = int(rng.randint(3, n - 2))
        init[start : start + rng.randint(1, 3)] = False
    defined = rng.rand(n) >= 0.15
    live = np.flatnonzero(init)
    if live.size:
        if rng.rand() < 0.3:
            defined[live[0]] = False
        if rng.rand() < 0.3:
            defined[-1] = False
        if rng.rand() < 0.3 and n > 1:
            at = int(rng.randint(0, n - 1))
            defined[at : at + 2] = False
    holes = np.flatnonzero(init & ~defined)
    defined[holes[max_holes:]] = True
    return value, init, defined


@pytest.mark.parametrize("cooldown", [0, 1, 2, 3, 4])
def test_first_true_is_every_completion_of_the_holes(cooldown):
    # The definition, by brute force: run the deterministic machine on all 2**k completions of the
    # k holes; a bar is decided where they all agree on fire/no-fire, and fires where they all
    # fire. The polynomial reference (completions merged by machine state) must agree with both.
    rng = np.random.RandomState(100 + cooldown)
    for _ in range(300):
        value, init, defined = _holed_child(rng, max_holes=6)
        want, want_defined = ref.first_true_completions(value, init, defined, cooldown)
        got, got_defined = nb.first_true_1d(value, init, defined, cooldown)
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(got_defined, want_defined)
        merged, merged_defined = ref.first_true(value, init, defined, cooldown)
        np.testing.assert_array_equal(merged, want)
        np.testing.assert_array_equal(merged_defined, want_defined)


@pytest.mark.parametrize("cooldown", [0, 1, 3, 9, 30])
def test_first_true_long_holed_series_matches_the_merged_reference(cooldown):
    # Holes far beyond brute force — scattered, a stretch longer than every cooldown here (up to
    # 30 cooling worlds at once, past the kernel's initial buffer and around its ring), warmup
    # re-entered: the kernel's compact world set must equal the reference's full state set.
    rng = np.random.RandomState(7 + cooldown)
    n = 1500
    value = rng.rand(n) < 0.5
    init = np.ones(n, dtype=bool)
    init[:10] = False
    init[700:705] = False
    defined = rng.rand(n) >= 0.15
    defined[300:340] = False
    want, want_defined = ref.first_true(value, init, defined, cooldown)
    got, got_defined = nb.first_true_1d(value, init, defined, cooldown)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(got_defined, want_defined)
    assert want.any() and not want_defined.all()


def test_first_true_hole_in_an_alternating_child_is_undefined_while_the_worlds_disagree():
    # The review's probe: a child alternating F, T with a hole at bar 5, cooldown 3. Had the hole
    # been True it fired there (then 9, 13, …); had it been False the next edge was bar 7 (then
    # 11, 15, …). The two phases never meet again, so every True bar from the hole on is
    # undecidable, while every False bar — no world fires on it — is decided. A taint of
    # `cooldown` bars declared the decision known again from bar 9, on one world's phase.
    n = 24
    value = np.arange(n) % 2 == 1
    defined = np.ones(n, dtype=bool)
    defined[5] = False
    got, got_defined = nb.first_true_1d(value, np.ones(n, dtype=bool), defined, 3)
    np.testing.assert_array_equal(got_defined, ~((np.arange(n) >= 5) & value))
    np.testing.assert_array_equal(np.flatnonzero(got), [1])


def test_first_true_hole_bar_is_decided_where_no_world_could_fire():
    ones = np.ones(7, dtype=bool)
    # Every world cooling down (a cooldown of 3 armed at bar 1): the hole at bar 3 fires in no
    # completion and changes nothing a later bar reads — every bar is decided, and the first edge
    # after the cooldown fires.
    value = np.array([False, True, False, False, False, False, True])
    defined = ones.copy()
    defined[3] = False
    got, got_defined = nb.first_true_1d(value, ones, defined, 3)
    np.testing.assert_array_equal(np.flatnonzero(got), [1, 6])
    assert got_defined.all()
    # The only world already True (it just fired, cooldown 0): the hole cannot be an edge, but
    # bar 3's True is one exactly when the hole was False.
    value = np.array([False, True, False, True, True, False, True])
    defined = ones.copy()
    defined[2] = False
    got, got_defined = nb.first_true_1d(value, ones, defined, 0)
    np.testing.assert_array_equal(got_defined, [True, True, True, False, True, True, True])
    np.testing.assert_array_equal(np.flatnonzero(got), [1, 6])
    # No world past its first False: the hole cannot fire either, but its False completion arms
    # the True after it.
    value = np.array([True, False, True, False, True, True, True])
    defined = ones.copy()
    defined[1] = False
    got, got_defined = nb.first_true_1d(value, ones, defined, 0)
    np.testing.assert_array_equal(got_defined, [True, True, False, True, True, True, True])
    np.testing.assert_array_equal(np.flatnonzero(got), [4])


@pytest.mark.parametrize(
    ("child", "cooldown", "fires", "undecided"),
    [
        # Every world cooling and the cooldown running on past the hole: the later cooldown bars
        # overwrite the prev it wrote, so even a True right after it is no edge in any completion.
        ("FT?TFTF", 2, [1, 5], []),
        # The hole on the cooldown's LAST bar writes the prev the first free bar reads: a True
        # there is an edge in exactly the hole's False completion, whose cooldown then shifts the
        # next fire too; a False there re-arms every completion alike.
        ("FTF?TFT", 2, [1], [4, 6]),
        ("FTF?FTF", 2, [1, 5], []),
        # Before the child's first False (seen_false) and right after a True (prev): the same split.
        ("T?TFTFT", 3, [], [2, 4, 6]),
        ("T?FT", 3, [3], []),
        ("FT?TFT", 0, [1, 5], [3]),
        ("FT?FT", 0, [1, 4], []),
    ],
)
def test_first_true_hole_is_absorbed_only_where_its_completions_reconverge_first(
    child, cooldown, fires, undecided
):
    # Deciding the hole bar (no completion could fire on it in any case here) does not absorb the
    # hole: that takes the completions' machines meeting again before any bar they disagree on.
    # Only an absorbed hole leaves `signal_coverage` clean and reaches the gate through
    # `source_coverage` alone.
    value = np.array([c == "T" for c in child])
    defined = np.array([c != "?" for c in child])
    got, got_defined = nb.first_true_1d(value, np.ones(value.size, dtype=bool), defined, cooldown)
    np.testing.assert_array_equal(np.flatnonzero(got), fires)
    np.testing.assert_array_equal(np.flatnonzero(~got_defined), undecided)


def test_first_true_world_set_stays_cheap_on_a_long_holed_series():
    # A smoke test against per-bar cost growing with the cooldown or the history, not a benchmark:
    # the kernel is O(1) amortized per bar (~10 ms for each call here); the bound sits two orders
    # of magnitude above that so a loaded machine cannot flake it.
    n = 10**6
    value = np.random.RandomState(5).rand(n) < 0.5
    init = np.ones(n, dtype=bool)
    defined = np.ones(n, dtype=bool)
    defined[::7] = False  # periodic holes: the worlds keep splitting and re-converging
    every_bar = np.ones(n, dtype=bool)
    every_bar[1:] = False  # one False, then nothing but holes: a fire may land on every bar
    nb.first_true_1d(value[:64], init[:64], defined[:64], 5)  # compile/load outside the clock
    start = time.perf_counter()
    _out, got_defined = nb.first_true_1d(value, init, defined, 5)
    # a cooldown longer than the series keeps a cooling world per bar alive: ~n of them, held in
    # a buffer sized by the worlds that exist, never by the cooldown
    _out, all_holes = nb.first_true_1d(np.zeros(n, dtype=bool), init, every_bar, 2**31 - 1)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, elapsed
    assert not got_defined.all()
    assert all_holes[0] and not all_holes[1:].any()


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


def test_zscore_sma_overflowed_variance_is_nan_and_an_underflowed_one_is_lifted():
    # At 1e160 the squared deviations overflow the window variance to inf, and `(x − mean) / inf`
    # read a FINITE 0.0 — "exactly at the mean" on every bar, which `z > 0` / `z < 0` decided:
    # NaN now, the undecidable ledger.
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 7.0])
    np.testing.assert_array_equal(zscore_sma_1d(1e160 * x, 3), np.full(6, np.nan))
    np.testing.assert_array_equal(ref.zscore_sma(1e160 * x, 3), np.full(6, np.nan))
    # Below the normal range the squares lost their bits — at 2**-536 z read 1.3608 where it is
    # 1.3363 (on random walks, past its population bound √(w−1)) — or vanished (2**-600 on: z
    # NaN off a variance of 0.0). The lifted window reads the unscaled z, every bit, as does the
    # direct form anywhere in between (2**±500).
    want = zscore_sma_1d(x, 3)
    for k in (500, -500, -536, -600, -1000):
        np.testing.assert_array_equal(zscore_sma_1d(x * 2.0**k, 3), want)
        _rtol(ref.zscore_sma(x * 2.0**k, 3), want)


def _geometric_ew_z(alpha: float, r: float) -> float:
    """The EW z-score a geometric input ``c·r**t`` converges to in exact arithmetic: the mean
    settles at ``k = α·r / (r − 1 + α)`` times the input and the variance at ``σ`` times its
    square, ``σ·(r² − 1 + α) = (1 − α)·α·(r − k)²``, so z is ``(1 − k)/√σ`` (signed as the input)
    for ANY scale ``c`` — while ``r² > 1 − α``; below that the variance outlives the input and z
    decays to 0."""
    k = alpha * r / (r - 1.0 + alpha)
    sigma = (1.0 - alpha) * alpha * (r - k) ** 2 / (r * r - 1.0 + alpha)
    return (1.0 - k) / math.sqrt(sigma)


def test_an_ema_decaying_through_a_hold_stays_decided_until_it_settles():
    # A price that moves, holds for 2,000 bars at an ordinary level (~100), then moves again: its
    # 1-bar change is 0.0 on the hold, and an `ema` of that change decays geometrically — through
    # the range where a window's squares underflow — until it settles on the smallest subnormal.
    # Every window of the decay VARIES: its std is positive and its z-score defined, so the guard
    # recipe `std > 0` and the z it guards agree on every bar; only the settled ema's windows
    # read zero dispersion (std 0.0, z NaN). Read as computed, the decay windows' variance of 0.0
    # made their std the all-equal window's 0.0 (the guard False on moving data) and their z NaN.
    # The EW z-score's variance rounded to 0.0 on the decay at alpha >= 0.5 (window 2 or 3,
    # alpha 0.6) — NaN on 760-920 bars beside the guard's True; its recurrence runs lifted there
    # (`nb._EW_LIFT_VAR`), so it is defined on every bar past warmup.
    rng = np.random.RandomState(7)
    px = 100.0 * np.exp(np.cumsum(0.01 * rng.randn(2500)))
    px[300:2300] = px[299]
    e = nb.ema_apply_nb(nb.change_apply_nb(_c1(px), 1, "pct"), *nb.ema_params(5, None)).ravel()
    w = 20
    for series in (e, -e):  # a decay from above zero and from below it
        windows = sliding_window_view(series, w)[40:]  # past every warmup
        constant = windows.min(axis=-1) == windows.max(axis=-1)
        assert constant.sum() > 100 and (~constant).sum() > 2000
        std = rolling_agg_1d(series, w, "std")[w - 1 :][40:]
        z = zscore_sma_1d(series, w)[w - 1 :][40:]
        np.testing.assert_array_equal(std == 0.0, constant)
        assert (std[~constant] > 0.0).all()
        np.testing.assert_array_equal(np.isnan(z), constant)
        assert np.abs(z[~constant]).max() <= math.sqrt(w - 1)
        for alpha, warm in (
            nb.ema_params(2, None),
            nb.ema_params(3, None),
            nb.ema_params(None, 0.6),
        ):
            z_ew = nb.zscore_ema_1d(series, alpha, warm)
            assert np.isfinite(z_ew[w - 1 :][40:]).all()
            np.testing.assert_array_equal(z_ew, ref.zscore_ema_alpha(series, alpha, warm))
    # ...and at its VALUE. Above alpha 5/9 the EW z of the (ratio 2/3) decay converges to a
    # constant however deep it goes — −1/(2√2) at window 2. Where the ema is still a normal double
    # but its squares are not, the variance read as computed rounded to 0.0 (z NaN) and, floored
    # at the smallest subnormal, collapsed z toward 0: `z > -0.25` held on ~600 bars exact
    # arithmetic keeps at −0.354. Lifted, every one of them reads the constant.
    deep = (np.abs(e) < 1.5e-154) & (np.abs(e) >= 2.0**-1022)
    assert deep.sum() > 800
    r = 1.0 - nb.ema_params(5, None)[0]  # the ema's own decay factor on a 0.0 change
    for alpha, warm in (nb.ema_params(2, None), nb.ema_params(None, 0.6)):
        for series in (e, -e):
            want = _geometric_ew_z(alpha, r) * np.sign(series[deep])
            got = nb.zscore_ema_1d(series, alpha, warm)[deep]
            np.testing.assert_allclose(got, want, rtol=1e-9, atol=0.0)
    assert _geometric_ew_z(2.0 / 3.0, 2.0 / 3.0) == pytest.approx(-1.0 / (2.0 * math.sqrt(2.0)))
    # two members decaying side by side never read an all-equal cross-section
    cross = nb.cross_agg_apply_nb(np.c_[e, 2.0 * e][40:], "std", 2)[:, 0]
    assert (cross > 0.0).all()


@pytest.mark.parametrize("base", [0.0, 1e8, 1e12])
def test_zscore_ema_translation_invariant_across_levels(base):
    r = _rand(200, seed=12)
    z0 = nb.zscore_ema_1d(r, *nb.ema_params(20, None))
    z = nb.zscore_ema_1d(r + base, *nb.ema_params(20, None))
    tol = 1e-9 if base == 0.0 else (1e-6 if base <= 1e8 else 1e-3)
    np.testing.assert_allclose(z, z0, rtol=0, atol=tol, equal_nan=True)
    assert np.isfinite(z[19:]).all()


@pytest.mark.parametrize("k", [500, -536, -560, -700, -1000, -1050])
def test_zscore_ema_is_exactly_scale_invariant_past_the_variance_range(k):
    # The EW z-score of x·2**k is the z-score of x — every bit, at every scale: scaling by a power
    # of two never rounds, so the recurrence on the scaled input is the same steps. Below ~2**-510
    # the variance's squares left the normal range: at 2**-536 the z read 0.5103 for 0.5275 (and
    # past its bound √((1−α)/α) on random walks), from ~2**-540 down NaN — and floored at the
    # smallest subnormal, 0.0 on every bar. The recurrence runs lifted there instead.
    rng = np.random.RandomState(0)
    walk = np.round(rng.randn(300).cumsum() * 4.0)  # integers: every scaled value is exact
    walk[[50, 51]] = np.nan
    steps = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 11.0])
    held = np.r_[steps, np.full(40, 11.0), steps]  # a carried deviation, then moving again
    for x in (walk, steps, held):
        for spec in ((2, None), (3, None), (5, None), (20, None), (None, 0.06), (None, 0.9)):
            alpha, _ = nb.ema_params(*spec)
            want = nb.zscore_ema_1d(x, alpha, 2)
            assert np.isfinite(want[2:][~np.isnan(x[2:])]).all()
            np.testing.assert_array_equal(nb.zscore_ema_1d(x * 2.0**k, alpha, 2), want)
            np.testing.assert_array_equal(ref.zscore_ema_alpha(x * 2.0**k, alpha, 2), want)


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
    # undecidable ledger's food — never ±inf, which would latch warmup and decide thresholds, and
    # never a FINITE number read off an overflowed intermediate. Exact values, not "not inf": a
    # bare not-inf check passed the 0.0 that `x / inf` makes of an overflowed denominator.
    nan = np.nan
    huge = np.full(30, 1e308)
    cases = [
        # a constant window has no dispersion, and its mean IS its value however large
        (zscore_sma_1d(huge, 4), np.full(30, nan)),
        (rolling_agg_1d(huge, 4, "mean"), [nan] * 3 + [1e308] * 27),
        # ±1e308 deviations overflow the variance itself
        (rolling_agg_1d(np.tile([1e308, -1e308], 15), 4, "std"), np.full(30, nan)),
        (change_1d(np.array([1e-320, 1.0, 1.0]), 1, "pct"), [nan, nan, 0.0]),
        (change_1d(np.array([1e308, -1e308, 1e308]), 1, "diff"), [nan, nan, nan]),
        (drawdown_1d(np.array([1e-320, -1e308])), [0.0, nan]),
        (runup_1d(np.array([1e-320, 1e308])), [0.0, nan]),
        # a series against itself at a SUBNORMAL-variance scale (deviations 5e-161): the squares
        # underflowed and the product divided by 0.0 → inf; lifted, the windows read exactly 1
        (
            rolling_corr_1d(np.tile([0.0, 1e-160], 6), np.tile([0.0, 1e-160], 6), 4),
            [nan] * 3 + [1.0] * 9,
        ),
        # an overflowed EW variance stays inf in the recurrence: never a z read off it
        (
            nb.zscore_ema_1d(np.array([0.0, 2e154, 1.0, 1.0, 1.0]), *nb.ema_params(2, None)),
            np.full(5, nan),
        ),
        # only the member whose deviation overflows is NaN
        (
            nb.cross_demean_apply_nb(np.array([[1.7e308, -1.7e308, -1.7e308]]), 2).ravel(),
            [nan, -1.7e308 - (-1.7e308 / 3.0), -1.7e308 - (-1.7e308 / 3.0)],
        ),
        (
            nb.cross_agg_apply_nb(np.array([[1.6e308, 1.6e308, 1.6e308]]), "mean", 2).ravel(),
            [1.6e308] * 3,
        ),
        (
            nb.cross_agg_apply_nb(np.array([[1.6e308, -1.6e308, 1.6e308]]), "std", 2).ravel(),
            [nan] * 3,
        ),
    ]
    for got, want in cases:
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("c", [0.1, 100.1, 0.7, -2.7])
def test_constant_decimal_window_has_exact_zero_dispersion(c):
    # The review's probe: the computed mean of a constant DECIMAL window is not the constant
    # ((0.1 + 0.1 + 0.1) / 3 != 0.1), and the residual minted zscore_sma([0.1]*3) == -1,
    # [100.1]*3 == +1 and rolling_corr(x, x) == 1. A constant window has exact answers: the mean
    # is the value, the std and the demeaned values are 0.0, z-score and correlation undefined.
    flat = np.full(6, c)
    assert np.isnan(zscore_sma_1d(flat, 3)).all()
    assert np.isnan(rolling_corr_1d(flat, flat, 3)).all()
    assert np.isnan(rolling_corr_1d(np.arange(6.0), flat, 3)).all()
    np.testing.assert_array_equal(rolling_agg_1d(flat, 3, "mean"), [np.nan, np.nan, c, c, c, c])
    np.testing.assert_array_equal(rolling_agg_1d(flat, 3, "std"), [np.nan, np.nan, 0, 0, 0, 0])
    row = np.array([[c, c, c, np.nan]])
    np.testing.assert_array_equal(nb.cross_demean_apply_nb(row, 2), [[0.0, 0.0, 0.0, np.nan]])
    np.testing.assert_array_equal(nb.cross_agg_apply_nb(row, "mean", 2), np.full((1, 4), c))
    np.testing.assert_array_equal(nb.cross_agg_apply_nb(row, "std", 2), np.zeros((1, 4)))
    # the frozen references state the same exact answers
    np.testing.assert_array_equal(ref.zscore_sma(flat, 3), zscore_sma_1d(flat, 3))
    np.testing.assert_array_equal(
        ref.rolling_corr_ref(flat, flat, 3), rolling_corr_1d(flat, flat, 3)
    )
    for agg in ("mean", "std"):
        np.testing.assert_array_equal(ref.rolling_agg(flat, 3, agg), rolling_agg_1d(flat, 3, agg))
        np.testing.assert_array_equal(
            ref.cross_agg_ref(row, agg, 2), nb.cross_agg_apply_nb(row, agg, 2)
        )


def test_constant_window_inside_a_varying_series_is_exact_only_there():
    # Only the windows whose values are all equal change; every other window is computed
    # exactly as before (bit for bit — a window one ulp off constant is NOT constant).
    arr = np.array([1.0, 2.0, 0.1, 0.1, 0.1, 0.4, np.nextafter(0.4, 1.0), 0.4])
    z = zscore_sma_1d(arr, 3)
    assert np.isnan(z[:2]).all() and np.isnan(z[4])
    assert np.isfinite(z[[2, 3, 5, 6, 7]]).all()
    np.testing.assert_array_equal(z, ref.zscore_sma(arr, 3))
    std = rolling_agg_1d(arr, 3, "std")
    assert std[4] == 0.0 and (std[[2, 3, 5, 6, 7]] > 0.0).all()


def test_grouped_constant_cross_section_is_exact():
    # Grouping wraps the same kernels, so a group whose members share one value demeans to 0.0
    # and aggregates exactly — beside a varying group computed as before.
    a = np.array([[0.1, 0.1, 0.1, 0.2, 0.5]])
    lab = np.array([[1.0, 1.0, 1.0, 2.0, 2.0]])
    dm = nb.cross_grouped_apply_nb(lambda x: nb.cross_demean_apply_nb(x, 2), a, lab)
    np.testing.assert_array_equal(dm[0, :3], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(dm, ref.cross_demean_grouped_ref(a, lab))
    sd = nb.cross_grouped_apply_nb(lambda x: nb.cross_agg_apply_nb(x, "std", 2), a, lab)
    np.testing.assert_array_equal(sd[0, :3], [0.0, 0.0, 0.0])
    assert sd[0, 3] == sd[0, 4] > 0.0
    mean = nb.cross_grouped_apply_nb(lambda x: nb.cross_agg_apply_nb(x, "mean", 2), a, lab)
    np.testing.assert_array_equal(mean[0, :3], [0.1, 0.1, 0.1])


@pytest.mark.parametrize("c", [0.1, 100.1, 0.7, -2.7])
def test_rolling_corr_constant_left_window_is_nan(c):
    # Either leg's constancy voids the window: a constant LEFT leg has its own pin.
    flat = np.full(6, c)
    varying = np.array([0.3, 0.1, 0.7, 0.2, 0.9, 0.4])
    assert np.isnan(rolling_corr_1d(flat, np.arange(6.0), 3)).all()
    assert np.isnan(rolling_corr_1d(flat, varying, 3)).all()
    np.testing.assert_array_equal(
        ref.rolling_corr_ref(flat, varying, 3), rolling_corr_1d(flat, varying, 3)
    )


def test_signed_zero_windows_are_constant():
    # -0.0 == 0.0: a window of signed zeros has no dispersion either.
    x = np.array([1.0, 0.0, -0.0, 0.0, -0.0, 0.0])
    assert np.isnan(zscore_sma_1d(x, 3)[3:]).all()
    np.testing.assert_array_equal(rolling_agg_1d(x, 3, "std")[3:], 0.0)
    assert np.isnan(rolling_corr_1d(x, np.arange(6.0), 3)[3:]).all()
    row = np.array([[0.0, -0.0, 0.0]])
    np.testing.assert_array_equal(nb.cross_demean_apply_nb(row, 2), np.zeros((1, 3)))
    np.testing.assert_array_equal(nb.cross_agg_apply_nb(row, "std", 2), np.zeros((1, 3)))


@pytest.mark.parametrize("c", [0.1, 100.1, 0.7, -2.7])
def test_cross_agg_median_and_frac_positive_are_exact_on_constant_rows(c):
    for row in (np.array([[c, c, c, np.nan]]), np.array([[c, c, c, c]])):
        np.testing.assert_array_equal(
            nb.cross_agg_apply_nb(row, "median", 2), np.full(row.shape, c)
        )
        np.testing.assert_array_equal(
            nb.cross_agg_apply_nb(row, "frac_positive", 2),
            np.full(row.shape, 1.0 if c > 0 else 0.0),
        )


@pytest.mark.parametrize("c", [0.1, 0.007, 100.1, 0.7])
def test_ema_of_a_constant_decimal_feed_is_the_constant(c):
    # The convex update alpha*x + (1-alpha)*e does not return e when x == e: a flat feed's EMA
    # drifted to a fixed point a few ulps off the feed, and `x > ema(x)` read a sign on every bar.
    flat = np.full(400, c)
    for w in (3, 10, 50):
        got = nb.ema_1d(flat, *nb.ema_params(w, None))
        np.testing.assert_array_equal(got[w - 1 :], c)
        np.testing.assert_array_equal(got, ref.ema(flat, w))
    got = nb.ema_1d(flat, *nb.ema_params(None, 0.06))
    np.testing.assert_array_equal(got[32:], c)
    np.testing.assert_array_equal(got, ref.ema_alpha(flat, 0.06, 33))


def test_ema_never_crosses_a_held_value():
    # Move, then hold: exact arithmetic approaches the held value from the side it came from and
    # never reaches it. The rounded convex update could land past it (a held stretch read the
    # opposite sign) or ON it (`x <= ema(x)` fired); held inside the open interval it stays
    # strictly on its own side, one ulp short of the value at the closest.
    rng = np.random.default_rng(5)
    for _ in range(300):
        c = round(float(rng.uniform(0.01, 200.0)), 2)
        start = c * (1.0 + float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.001, 0.2)))
        x = np.concatenate([np.full(5, start), np.full(600, c)])
        w = int(rng.integers(2, 60))
        tail = nb.ema_1d(x, *nb.ema_params(w, None))[300:]
        assert (tail > c).all() if start > c else (tail < c).all(), (c, start, w)
        np.testing.assert_array_equal(nb.ema_1d(x, *nb.ema_params(w, None)), ref.ema(x, w))


def test_zscore_ema_of_a_held_stretch_decays_instead_of_locking():
    # A series that moves and then holds: the EW mean converges until its update falls below
    # half an ulp, then stuck a few ulps off the held value; the deviation froze at that residue,
    # the variance settled at (1-alpha)*d^2, and the z-score locked at exactly ±1/sqrt(1-alpha)
    # (1.2247 at window 5) for as long as the input held. Exact arithmetic decays it to 0.
    rng = np.random.default_rng(6)
    for _ in range(200):
        c = round(float(rng.uniform(0.01, 200.0)), 2)
        x = np.concatenate([c + rng.normal(0.0, 0.05 * c, 30), np.full(700, c)])
        w = int(rng.integers(2, 40))
        z = nb.zscore_ema_1d(x, *nb.ema_params(w, None))
        tail = z[-200:]
        assert np.all(np.isnan(tail) | (np.abs(tail) < 1e-3)), (c, w, np.nanmax(np.abs(tail)))
        np.testing.assert_array_equal(z, ref.zscore_ema(x, w))
    alpha_z = nb.zscore_ema_1d(x, *nb.ema_params(None, 0.06))
    np.testing.assert_array_equal(alpha_z, ref.zscore_ema_alpha(x, 0.06, 33))


def test_zscore_ema_of_a_held_stretch_keeps_its_exact_sign():
    # Exact arithmetic moves a held input's deviation x − μ to (1 − α)·(x − μ) every bar: it
    # decays and KEEPS ITS SIGN, so `z > 0` ("above its EW mean") holds on every bar of a hold
    # reached from below. A mean that rounds onto or past the held value, or a deviation zeroed
    # once the update is absorbed, read the opposite sign on most held bars.
    rng = np.random.default_rng(8)
    for _ in range(40):
        c = round(float(rng.uniform(0.01, 200.0)), 2)
        w = int(rng.choice([3, 5, 10, 20]))
        x = np.concatenate([c + rng.normal(0.0, 0.05 * c, 20), np.full(250, c)])
        z = nb.zscore_ema_1d(x, *nb.ema_params(w, None))
        a = Fraction(2.0 / (w + 1.0))
        mu = Fraction(float(x[0]))
        for i in range(1, x.size):
            xi = Fraction(float(x[i]))
            mu += a * (xi - mu)
            if i >= 20 and np.isfinite(z[i]):
                assert (z[i] > 0) == (xi > mu) and (z[i] < 0) == (xi < mu), (c, w, i)
        np.testing.assert_array_equal(z, ref.zscore_ema(x, w))


@pytest.mark.parametrize("c", [0.007, 0.1, 100.1])
def test_zscore_ema_carry_needs_a_held_input(c):
    # An update absorbed on an input that MOVED (c and its next float, alternating) is
    # variation: the plain recurrence runs, bit for bit — only a repeated input carries its
    # deviation.
    x = np.tile([c, np.nextafter(c, np.inf)], 200)
    for w in (3, 10):
        got = nb.zscore_ema_1d(x, *nb.ema_params(w, None))
        np.testing.assert_array_equal(got, ref.zscore_ema(x, w))
        assert np.nanmax(np.abs(got)) > 0.5
    np.testing.assert_array_equal(
        nb.zscore_ema_1d(x, *nb.ema_params(None, 0.06)), ref.zscore_ema_alpha(x, 0.06, 33)
    )


def test_event_agg_mean_of_a_run_one_ulp_off_constant_is_sum_over_count():
    # Only a bit-identical run takes its value as the mean; one ulp off is variation, read as
    # sum / count like any other run.
    n = 40
    x = np.full(n, 0.1)
    x[10] = np.nextafter(0.1, 1.0)
    sig = np.zeros(n, dtype=bool)
    sig[0] = True
    ones = np.ones(n, dtype=bool)
    got = nb.event_agg_apply_nb(sig[:, None], ones[:, None], ones[:, None], x[:, None], "mean")
    np.testing.assert_array_equal(got.ravel()[10:], (np.cumsum(x) / np.arange(1, n + 1))[10:])


@pytest.mark.parametrize(("lo", "hi"), [(99.75, 100.0), (1.0, 1.25), (100.25, 100.0), (0.1, 0.2)])
@pytest.mark.parametrize("w", [3, 5, 10])
def test_ema_of_a_held_stretch_never_reaches_the_held_value(lo, hi, w):
    # Exact arithmetic: after a step the EW mean approaches the held value and NEVER reaches it,
    # so `x > ema(x)` (after a step up) holds on every held bar — the same predicate as the
    # engine's own `zscore(x, w, ema) > 0`, which carries its deviation to keep exactly that sign.
    # An ema that lands ON the held value reads `x <= ema(x)` there: firings exact arithmetic
    # never makes.
    x = np.r_[np.full(5, lo), np.full(400, hi)]
    alpha, warm = nb.ema_params(w, None)
    e = nb.ema_1d(x, alpha, warm)[10:]
    assert ((e < hi) if hi > lo else (e > hi)).all()
    z = nb.zscore_ema_1d(x, alpha, warm)[10:]
    np.testing.assert_array_equal(np.sign(hi - e), np.sign(z))


def _plain_west(x, alpha, warmup):
    """The 4.0.0 West recurrence, no clamp, no carry."""
    out = np.full(x.size, np.nan)
    mu, s, seen = np.nan, 0.0, 0
    for i, xi in enumerate(x):
        if np.isnan(xi):
            continue
        if np.isnan(mu):
            mu, s = xi, 0.0
        else:
            d = xi - mu
            mu = mu + alpha * d
            s = (1.0 - alpha) * (s + alpha * d * d)
        seen += 1
        if seen >= warmup and s > 0.0:
            out[i] = (xi - mu) / math.sqrt(s)
    return out


@pytest.mark.parametrize("level", [4500.0, 100.0])
def test_zscore_ema_carry_leaves_no_trace_once_the_input_moves(level):
    # Window 3 (alpha = 0.5): the held stretch's last mean update ties onto the held value, so the
    # deviation is carried from there. Once the input MOVES the recurrence is the plain one again
    # (no clamp binds on these bars): the post-hold bars read exactly what the plain West
    # recurrence — and exact arithmetic — reads, not a mean left one ulp short of the held value.
    x = np.r_[
        np.full(5, level - 0.25), np.full(200, level), [level + 0.25, level + 0.5, level + 0.25]
    ]
    alpha, warm = nb.ema_params(3, None)
    z = nb.zscore_ema_1d(x, alpha, warm)
    np.testing.assert_array_equal(z[205:], _plain_west(x, alpha, warm)[205:])


@pytest.mark.parametrize("w", [2, 3])
def test_zscore_ema_long_hold_stays_defined_and_signed(w):
    # alpha >= 0.5: the carried deviation and the EW variance shrink by (1 - alpha) per held bar
    # and would round to exactly 0.0 once subnormal (alpha < 0.5 rounds back up to the smallest
    # subnormal and keeps a signed z for ever). Exact arithmetic keeps z > 0 on every held bar of
    # a hold reached from below — never 0.0, never undefined (no refusal on its own hold).
    x = np.r_[np.full(30, 0.05), np.full(3000, 0.0525)]
    z = nb.zscore_ema_1d(x, *nb.ema_params(w, None))[30:]
    assert np.isfinite(z).all()
    assert (z > 0.0).all()


@pytest.mark.parametrize("side", [1.0, -1.0])
def test_zscore_ema_deviation_that_underflows_keeps_its_exact_sign(side):
    # Window 2 (alpha 2/3) at the smallest subnormals — where an `ema` of a change settles after a
    # hold: the input steps (∓3 to ∓1 of them, or ∓1 of them to 0.0) and holds for 1,500 bars.
    # Exact arithmetic moves the deviation to (1 − α)·d every held bar: it decays and keeps the
    # step's sign. Read as computed, the EW variance rounded to 0.0 (z NaN on the first bars, or
    # on every bar of a hold onto 0.0), and a carried deviation (1 − α)·d that rounded to ±0.0
    # was floored at the NEGATIVE subnormal whatever d's sign: z below the mean on a hold exact
    # arithmetic keeps above it. Floored where it rounded to 0.0, the variance collapsed z to
    # 2.2e-162 from the first held bar (the bound √((1−α)/α) = 0.707 there). Lifted, the decay
    # reads the z-scores of the same steps at ordinary scale, bit for bit, as long as those stay
    # in range (~650 bars); past the frame's reach the variance and the carry are floored where
    # they underflow, the sign kept.
    u = 5e-324
    alpha, warm = nb.ema_params(2, None)
    for first, held in ((-3.0 * u, -u), (-u, 0.0)):
        x = side * np.r_[first, np.full(1500, held)]
        z = nb.zscore_ema_1d(x, alpha, warm)
        np.testing.assert_array_equal(z[:600], nb.zscore_ema_1d(x / u, alpha, warm)[:600])
        assert z[1] == pytest.approx(side * math.sqrt((1.0 - alpha) / alpha))
        np.testing.assert_array_equal(np.sign(z[1:]), np.sign(x[1] - x[0]))
        np.testing.assert_array_equal(z, ref.zscore_ema_alpha(x, alpha, warm))


def test_zscore_ema_at_alpha_one_is_undefined_on_every_bar():
    # At alpha 1 (the DSL refuses it) the EW mean IS the last value and the EW variance exactly 0:
    # undefined throughout. A full-weight update that rounds off the value (1.1 + (0.1 − 1.1) is
    # 0.10000000000000009) carried a deviation into the hold, and the held branch's floor read
    # z = −2.2e-162 off a variance exact arithmetic keeps at 0 — every bar from the fourth on.
    x = np.r_[1.1, np.full(5, 0.1)]
    np.testing.assert_array_equal(nb.zscore_ema_1d(x, 1.0, 1), np.full(6, np.nan))
    np.testing.assert_array_equal(ref.zscore_ema_alpha(x, 1.0, 1), np.full(6, np.nan))


def test_event_agg_mean_of_a_constant_run_is_the_value():
    # The since-event mean is a window mean like rolling_agg's: an all-equal run's mean is its
    # value, not sum/count (0.1 * k / k drifts an ulp for most k).
    n = 60
    x = np.full(n, 0.1)
    sig = np.zeros(n, dtype=bool)
    sig[[0, 25]] = True
    ones = np.ones(n, dtype=bool)
    got = nb.event_agg_apply_nb(sig[:, None], ones[:, None], ones[:, None], x[:, None], "mean")
    np.testing.assert_array_equal(got.ravel(), 0.1)
    np.testing.assert_array_equal(got.ravel(), ref.event_agg(sig, ones, ones, x, "mean"))


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


# ---- the event-anchor family -----------------------------------------------------------


def _b(*flags) -> np.ndarray:
    return np.asarray(flags, dtype=bool)


def _event_channels(seed: int = 3, n: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A random tradable signal with a warmup prefix and scattered post-warmup holes."""
    rng = np.random.RandomState(seed)
    init = np.ones(n, dtype=bool)
    init[: rng.randint(3, 12)] = False
    defined = rng.rand(n) > 0.08
    defined[~init] = True  # warmup is vacuously defined
    sig = (rng.rand(n) > 0.8) & init & defined
    return sig, init, defined


def _anchor_1d(sig, init, defined):
    s, k = nb.event_anchor_apply_nb(
        _c1(sig).astype(bool), _c1(init).astype(bool), _c1(defined).astype(bool)
    )
    return s.reshape(-1), k.reshape(-1)


def test_event_anchor_matches_reference():
    sig, init, defined = _event_channels()
    s, k = _anchor_1d(sig, init, defined)
    rs, rk = ref.event_anchor(sig, init, defined)
    np.testing.assert_array_equal(s, rs)
    np.testing.assert_array_equal(k, rk)


def test_event_anchor_known_answers():
    #        0  1  2  3  4  5  6  7
    sig = _b(0, 0, 1, 0, 1, 0, 0, 0)
    init = _b(0, 1, 1, 1, 1, 1, 1, 1)
    defd = _b(1, 1, 1, 1, 1, 0, 1, 1)
    s, k = _anchor_1d(sig, init, defd)
    # warmup prefix: no anchor, known; an event replaces the previous one; a hole makes the
    # anchor UNKNOWN until the next defined event — a defined False after the hole stays unknown.
    np.testing.assert_array_equal(s, [-1, -1, 2, 2, 4, -1, -1, -1])
    np.testing.assert_array_equal(k, [1, 1, 1, 1, 1, 0, 0, 0])
    # the next defined EVENT re-establishes knowledge
    sig2 = _b(0, 0, 1, 0, 1, 0, 1, 0)
    s2, k2 = _anchor_1d(sig2, init, defd)
    np.testing.assert_array_equal(s2, [-1, -1, 2, 2, 4, -1, 6, 6])
    np.testing.assert_array_equal(k2, [1, 1, 1, 1, 1, 0, 1, 1])


def test_event_anchor_apply_nb_is_columnwise():
    cols = [_event_channels(seed) for seed in (1, 2, 3)]
    sig = np.column_stack([c[0] for c in cols])
    init = np.column_stack([c[1] for c in cols])
    defd = np.column_stack([c[2] for c in cols])
    s, k = nb.event_anchor_apply_nb(sig, init, defd)
    for j, (sj, ij, dj) in enumerate(cols):
        rs, rk = ref.event_anchor(sj, ij, dj)
        np.testing.assert_array_equal(s[:, j], rs)
        np.testing.assert_array_equal(k[:, j], rk)


@pytest.mark.parametrize("agg", ["sum", "max", "min", "mean"])
def test_event_agg_matches_reference(agg):
    sig, init, defined = _event_channels(seed=11)
    x = _rand(sig.shape[0], seed=12)
    x[np.random.RandomState(13).rand(x.shape[0]) < 0.05] = np.nan
    got = nb.event_agg_apply_nb(
        _c1(sig).astype(bool), _c1(init).astype(bool), _c1(defined).astype(bool), _c1(x), agg
    )
    _assert(got.reshape(-1), ref.event_agg(sig, init, defined, x, agg))


def test_event_agg_known_answers():
    #                 0    1    2    3    4    5    6    7
    x = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    sig = _b(0, 1, 0, 0, 1, 0, 0, 0)
    init = _b(0, 1, 1, 1, 1, 1, 1, 1)
    defd = _b(1, 1, 1, 1, 1, 1, 0, 1)

    def agg(kind):
        return nb.event_agg_apply_nb(
            _c1(sig).astype(bool), _c1(init).astype(bool), _c1(defd).astype(bool), _c1(x), kind
        ).reshape(-1)

    # reset at each event (the event bar included); mean = sum / (t − s + 1); the hole bar drops
    # the anchor, so nothing is emitted from it until the next event
    _assert(agg("sum"), [np.nan, 2, 5, 9, 5, 11, np.nan, np.nan])
    _assert(agg("mean"), [np.nan, 2, 2.5, 3, 5, 5.5, np.nan, np.nan])
    _assert(agg("max"), [np.nan, 2, 3, 4, 5, 6, np.nan, np.nan])
    _assert(agg("min"), [np.nan, 2, 2, 2, 5, 5, np.nan, np.nan])
    # a NaN input INSIDE the window poisons until the next event
    xn = x.copy()
    xn[2] = np.nan
    got = nb.event_agg_apply_nb(
        _c1(sig).astype(bool), _c1(init).astype(bool), _c1(defd).astype(bool), _c1(xn), "sum"
    ).reshape(-1)
    _assert(got, [np.nan, 2, np.nan, np.nan, 5, 11, np.nan, np.nan])


def test_event_agg_overflow_yields_nan_never_inf():
    big = np.full(4, 1e308)
    sig = _b(1, 0, 0, 0)
    ones = np.ones(4, dtype=bool)
    got = nb.event_agg_apply_nb(
        _c1(sig).astype(bool), _c1(ones), _c1(ones), _c1(big), "sum"
    ).reshape(-1)
    assert got[0] == 1e308 and np.isnan(got[1:]).all()
    assert not np.isinf(got).any()


def test_event_agg_apply_nb_is_columnwise():
    cols = [_event_channels(seed) for seed in (4, 5)]
    n = cols[0][0].shape[0]
    x = np.column_stack([_rand(n, seed=20), _rand(n, seed=21)])
    sig = np.column_stack([c[0] for c in cols])
    init = np.column_stack([c[1] for c in cols])
    defd = np.column_stack([c[2] for c in cols])
    got = nb.event_agg_apply_nb(sig, init, defd, x, "mean")
    for j, (sj, ij, dj) in enumerate(cols):
        _assert(got[:, j], ref.event_agg(sj, ij, dj, x[:, j], "mean"))


def test_event_agg_unknown_agg_raises():
    ones = np.ones((3, 1), dtype=bool)
    with pytest.raises(ValueError, match="unknown event_agg agg"):
        nb.event_agg_apply_nb(ones, ones, ones, np.ones((3, 1)), "median")


def test_bars_since_event_and_event_value_known_answers():
    s_idx = np.asarray([-1, -1, 2, 2, 4, -1, 6, 6], dtype=np.int64).reshape(-1, 1)
    x = np.asarray([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, np.nan]).reshape(-1, 1)
    _assert(
        nb.bars_since_event_apply_nb(s_idx).reshape(-1), [np.nan, np.nan, 0, 1, 0, np.nan, 0, 1]
    )
    # x[s]: a NaN input ON the event bar reads NaN until the next event; a later NaN in x is
    # irrelevant once the snapshot was taken
    _assert(
        nb.event_value_apply_nb(x, s_idx).reshape(-1),
        [np.nan, np.nan, np.nan, np.nan, 5, np.nan, 7, 7],
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        nb.event_value_apply_nb(x[:3], s_idx)


def test_mask_apply_nb_known_answers():
    value = _b(0, 1, 1, 0, 1)
    init = _b(0, 1, 1, 1, 1)
    defd = _b(1, 1, 0, 1, 1)
    _assert(nb.mask_apply_nb(value, init, defd, init), [np.nan, 1.0, np.nan, 0.0, 1.0])
    # Only a FULLY-initialized condition is a number: before every branch is live a False is no
    # claim, and the Trues alone would be a value-selected series — both read NaN.
    full = _b(0, 0, 0, 0, 1)
    _assert(nb.mask_apply_nb(value, init, defd, full), [np.nan, np.nan, np.nan, np.nan, 1.0])


# ---- grouped cross-sectional kernels --------------------------------------------------------


def _labels(rows: int = 60, cols: int = 5, seed: int = 8) -> np.ndarray:
    rng = np.random.RandomState(seed)
    lab = rng.randint(1, 4, size=(rows, cols)).astype(float)
    lab[rng.rand(rows, cols) < 0.1] = np.nan
    return lab


def _rank_kernel(mv=2):
    return lambda a: nb.cross_rank_apply_nb(a, mv)


def _demean_kernel(mv=2):
    return lambda a: nb.cross_demean_apply_nb(a, mv)


def _agg_kernel(agg, mv=2):
    return lambda a: nb.cross_agg_apply_nb(a, agg, mv)


def test_cross_grouped_matches_reference():
    a = _nan_riddled()
    lab = _labels()
    _assert(nb.cross_grouped_apply_nb(_rank_kernel(), a, lab), ref.cross_rank_grouped_ref(a, lab))
    _assert(
        nb.cross_grouped_apply_nb(_demean_kernel(), a, lab), ref.cross_demean_grouped_ref(a, lab)
    )
    for agg in ("mean", "median", "std", "frac_positive"):
        _assert(
            nb.cross_grouped_apply_nb(_agg_kernel(agg), a, lab),
            ref.cross_agg_grouped_ref(a, lab, agg),
        )


def test_cross_grouped_single_label_is_bit_exact_with_ungrouped():
    a = _nan_riddled()
    one = np.ones_like(a)
    for kernel in (_rank_kernel(3), _demean_kernel(), _agg_kernel("std"), _agg_kernel("mean")):
        assert np.array_equal(nb.cross_grouped_apply_nb(kernel, a, one), kernel(a), equal_nan=True)


def test_cross_grouped_nan_label_and_singleton_group_are_nan():
    a = np.array([[1.0, 2.0, 3.0, 4.0]])
    lab = np.array([[1.0, 1.0, 2.0, np.nan]])
    got = nb.cross_grouped_apply_nb(_rank_kernel(), a, lab)
    _assert(got, [[0.0, 1.0, np.nan, np.nan]])  # member 2 is alone (k=1 < 2); member 3 unlabelled
    got = nb.cross_grouped_apply_nb(_agg_kernel("mean"), a, lab)
    _assert(got, [[1.5, 1.5, np.nan, np.nan]])


def test_cross_grouped_labels_are_point_in_time():
    a = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    lab = np.array([[1.0, 1.0, 2.0, 2.0], [1.0, 2.0, 1.0, 2.0]])
    got = nb.cross_grouped_apply_nb(_rank_kernel(), a, lab)
    _assert(got, [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])


def test_cross_grouped_agg_broadcasts_within_the_group_only():
    a = np.array([[1.0, 3.0, np.nan, 10.0, 30.0]])
    lab = np.array([[1.0, 1.0, 1.0, 2.0, 2.0]])
    got = nb.cross_grouped_apply_nb(_agg_kernel("mean"), a, lab)
    # the NaN-input member of group 1 still sees ITS group's mean (the sanctioned broadcast),
    # never group 2's
    _assert(got, [[2.0, 2.0, 2.0, 20.0, 20.0]])


def test_cross_grouped_min_valid_is_per_group():
    a = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    lab = np.array([[1.0, 1.0, 1.0, 2.0, 2.0]])
    got = nb.cross_grouped_apply_nb(_rank_kernel(3), a, lab)
    _assert(got, [[0.0, 0.5, 1.0, np.nan, np.nan]])  # group 2 has k=2 < min_valid=3


def test_cross_grouped_shape_mismatch_raises():
    with pytest.raises(ValueError, match="cross_grouped shape mismatch"):
        nb.cross_grouped_apply_nb(_rank_kernel(), np.ones((3, 2)), np.ones((3, 3)))


# ---- rolling median / mad, and the explicit-decay EW forms --------------------------------


@pytest.mark.parametrize("agg", ["median", "mad"])
def test_rolling_median_and_mad_match_reference(agg):
    arr = _rand(200, seed=21)
    arr[[7, 40, 41, 150]] = np.nan
    want = ref.rolling_median(arr, 9) if agg == "median" else ref.rolling_mad(arr, 9)
    _assert(rolling_agg_1d(arr, 9, agg), want)


def test_rolling_median_window_two_is_the_midpoint():
    _assert(rolling_agg_1d(np.array([1.0, 3.0, 8.0]), 2, "median"), [np.nan, 2.0, 5.5])


def test_rolling_mad_centres_on_the_same_windows_median():
    # Window [1, 2, 10]: median 2, deviations [1, 0, 8] → MAD 1. The naive nesting
    # median(|x − rolling_median(x)|) reads a DIFFERENT number here, because each bar's deviation
    # is taken about its own window's median: rolling_median over [1,2,10] (window 3) is
    # [nan, nan, 2], and |x − that| over the same three bars is [nan, nan, 8] — not a MAD at all.
    x = np.array([1.0, 2.0, 10.0])
    _assert(rolling_agg_1d(x, 3, "mad"), [np.nan, np.nan, 1.0])
    naive = rolling_agg_1d(np.abs(x - rolling_agg_1d(x, 3, "median")), 3, "median")
    assert np.isnan(naive).all()
    # a flat window has zero MAD; a NaN window is censored
    _assert(rolling_agg_1d(np.array([5.0, 5.0, 5.0, 5.0]), 3, "mad"), [np.nan, np.nan, 0.0, 0.0])
    _assert(
        rolling_agg_1d(np.array([1.0, np.nan, 3.0, 4.0]), 2, "mad"), [np.nan, np.nan, np.nan, 0.5]
    )


def test_rolling_median_mad_apply_nb_are_columnwise():
    a = np.column_stack([_rand(50, seed=1), _rand(50, seed=2)])
    for agg, fn in (("median", ref.rolling_median), ("mad", ref.rolling_mad)):
        got = nb.rolling_agg_apply_nb(a, 5, agg)
        for j in range(2):
            _assert(got[:, j], fn(a[:, j], 5))


def test_rolling_mad_overflow_yields_nan_never_inf():
    x = np.array([-1e308, 1e308, 1e308])
    got = rolling_agg_1d(x, 3, "mad")
    assert not np.isinf(got).any()


def test_ema_params_window_form_and_alpha_form():
    assert nb.ema_params(20, None) == (2.0 / 21.0, 20)
    assert nb.ema_params(None, 0.06) == (0.06, 33)
    assert nb.ema_params(None, 1.0) == (1.0, 1)
    for w in range(1, 501):
        assert nb.ema_params(None, 2.0 / (w + 1.0))[1] == w, w
    with pytest.raises(ValueError, match="exactly one of 'window' or 'alpha'"):
        nb.ema_params(20, 0.06)
    with pytest.raises(ValueError, match="exactly one of 'window' or 'alpha'"):
        nb.ema_params(None, None)


def test_ema_alpha_form_is_bit_exact_with_window_form():
    arr = _rand(300, seed=5)
    arr[[3, 77]] = np.nan
    for w in (1, 2, 5, 20, 63):
        a, warm = nb.ema_params(None, 2.0 / (w + 1.0))
        assert np.array_equal(
            nb.ema_1d(arr, a, warm), nb.ema_1d(arr, *nb.ema_params(w, None)), equal_nan=True
        )
        assert np.array_equal(
            nb.zscore_ema_1d(arr, a, warm),
            nb.zscore_ema_1d(arr, *nb.ema_params(w, None)),
            equal_nan=True,
        )


def test_ema_alpha_one_is_the_input():
    arr = _rand(20, seed=6)
    arr[4] = np.nan
    _assert(nb.ema_1d(arr, *nb.ema_params(None, 1.0)), arr)


def test_ema_riskmetrics_known_answer_vs_hand_recurrence():
    arr = _rand(120, seed=7)
    got = nb.ema_1d(arr, *nb.ema_params(None, 0.06))
    e = arr[0]
    hand = np.full(120, np.nan)
    for i in range(1, 120):
        e = 0.06 * arr[i] + 0.94 * e
        if i + 1 >= 33:
            hand[i] = e
    _assert(got, hand)
    _assert(got, ref.ema_alpha(arr, 0.06, 33))
    assert int(np.argmax(np.isfinite(got))) == 32


def test_zscore_ema_alpha_matches_reference():
    arr = _rand(150, seed=8)
    arr[[10, 11]] = np.nan
    a, warm = nb.ema_params(None, 0.1)
    _assert(nb.zscore_ema_1d(arr, a, warm), ref.zscore_ema_alpha(arr, a, warm))
    got2d = nb.zscore_ema_apply_nb(np.column_stack([arr, arr * 2]), a, warm)
    _assert(got2d[:, 1], ref.zscore_ema_alpha(arr * 2, a, warm))
