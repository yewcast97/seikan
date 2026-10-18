"""The runner's path kernels (``compiler/paths.py``) against their direct definitions."""

from __future__ import annotations

import tracemalloc

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from seikan.compiler.paths import bars_to_trough_full, excursion_extremum


def _trough_off_the_stack(adverse: np.ndarray, base: np.ndarray, h: int, sign: float) -> np.ndarray:
    """The trough index by its definition, read off the full (anchors × targets × h) window
    stack: ``argmin`` / ``argmax`` (the first attainment on ties) over every H/L window, the exit
    print where strictly beyond it, NaN unless the window and the exit print are finite."""
    n = adverse.shape[0]
    out = np.full(adverse.shape, np.nan)
    if h + 1 > n:
        return out
    sw = sliding_window_view(adverse, h, axis=0)[: n - h]
    idx = sw.argmin(axis=-1) if sign > 0 else sw.argmax(axis=-1)
    ext = np.take_along_axis(sw, idx[..., None], axis=-1)[..., 0]
    exit_print = base[h:]
    valid = np.isfinite(sw).all(axis=-1) & np.isfinite(exit_print)
    exit_wins = (exit_print < ext) if sign > 0 else (exit_print > ext)
    out[: n - h] = np.where(valid, np.where(exit_wins, float(h), idx.astype(float)), np.nan)
    return out


def test_bars_to_trough_is_the_first_attainment_of_the_window_extremum():
    # Paths of four levels — ties everywhere — with holes and ±inf in the H/L path and holes in
    # the exit print, in both directions: the first bar attaining the window's extremum, the exit
    # print only where it is strictly beyond it, NaN wherever a window member or the exit print
    # is not finite.
    rng = np.random.default_rng(23)
    for _ in range(300):
        n = int(rng.integers(3, 80))
        h = int(rng.integers(1, min(n, 12)))
        g = int(rng.integers(1, 4))
        adverse = rng.choice([0.0, 1.0, 2.0, 3.0], (n, g))
        base = rng.choice([0.0, 1.0, 2.0, 3.0], (n, g))
        adverse[rng.random((n, g)) < 0.08] = np.nan
        adverse[rng.random((n, g)) < 0.03] = np.inf
        adverse[rng.random((n, g)) < 0.03] = -np.inf
        base[rng.random((n, g)) < 0.05] = np.nan
        for sign in (1.0, -1.0):
            np.testing.assert_array_equal(
                bars_to_trough_full(adverse, base, h, sign),
                _trough_off_the_stack(adverse, base, h, sign),
            )
    # a horizon past the data censors every anchor
    assert np.isnan(bars_to_trough_full(np.ones((3, 2)), np.ones((3, 2)), 3, 1.0)).all()


def _excursion_by_definition(frame: np.ndarray, base: np.ndarray, h: int, which: str) -> np.ndarray:
    """The excursion extremum by its definition: the min or max over the window's H/L path
    ``frame[t .. t+h-1]`` and its exit print ``base[t+h]``, NaN unless every member is finite."""
    n, g = frame.shape
    out = np.full(frame.shape, np.nan)
    pick = np.min if which == "min" else np.max
    for t in range(n - h):
        for c in range(g):
            members = np.append(frame[t : t + h, c], base[t + h, c])
            if np.isfinite(members).all():
                out[t, c] = pick(members)
    return out


def test_excursion_extremum_is_its_window_extremum_beside_any_hole():
    # Holes and ±inf anywhere in the H/L path and the exit print, in both directions: NaN wherever
    # a window member is not finite, and every window a hole is NOT in reads its own extremum.
    # scipy's comparison-based filters carried a NaN's state past its window, so a fully observed
    # window behind a hole read the wrong extremum (the reviewer's fixture is
    # `test_mae_and_mfe_behind_a_hole_read_their_own_window`). The extremum and the trough index
    # come off one scan: the bar the trough indexes (the exit print at j = h) holds the extremum.
    rng = np.random.default_rng(29)
    for _ in range(300):
        n = int(rng.integers(3, 80))
        h = int(rng.integers(1, min(n, 13)))
        g = int(rng.integers(1, 4))
        frame = rng.choice([0.0, 1.0, 2.0, 3.0], (n, g)) + rng.normal(0.0, 0.1, (n, g))
        base = rng.choice([0.0, 1.0, 2.0, 3.0], (n, g))
        frame[rng.random((n, g)) < 0.08] = np.nan
        frame[rng.random((n, g)) < 0.03] = np.inf
        frame[rng.random((n, g)) < 0.03] = -np.inf
        base[rng.random((n, g)) < 0.05] = np.nan
        for which, sign in (("min", 1.0), ("max", -1.0)):
            ext = excursion_extremum(frame, base, h, which)
            np.testing.assert_array_equal(ext, _excursion_by_definition(frame, base, h, which))
            trough = bars_to_trough_full(frame, base, h, sign)
            np.testing.assert_array_equal(np.isnan(ext), np.isnan(trough))
            rows, cols = np.nonzero(~np.isnan(trough))
            j = trough[rows, cols].astype(np.int64)
            at = np.where(j < h, frame[rows + j, cols], base[rows + h, cols])
            np.testing.assert_array_equal(ext[rows, cols], at)


def test_bars_to_trough_runs_in_memory_linear_in_the_frame():
    # `argmin` over the non-contiguous `sliding_window_view` of the H/L windows copied the whole
    # (anchors × targets × h) stack: 170 MiB for this 0.73 MiB frame of 3,000 bars × 32 targets
    # at h = 252 (~4.5 GiB at 5,000 × 500), where the excursion extrema beside it are O(n) per
    # column. One scan per column keeps the frame's own scale. tracemalloc counts numpy's
    # allocations, which is where a stacked window lands.
    rng = np.random.default_rng(0)
    adverse = rng.standard_normal((3000, 32))
    base = rng.standard_normal((3000, 32))
    bars_to_trough_full(adverse[:300], base[:300], 252, 1.0)  # compile outside the measurement
    tracemalloc.start()
    try:
        bars_to_trough_full(adverse, base, 252, 1.0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * adverse.nbytes
