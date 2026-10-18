"""Path kernels for the runner: window shifts, excursion-window extrema, and the
bars-to-positive / bars-to-trough timing frames — numpy, save the one sequential scan (numba)
the excursion extrema and the trough index are read off.

Every function here is a leaf — plain (rows × targets) array in, array out, no seikan imports —
computed once per horizon over the FULL anchor set (combo-independent, so per-cell computation
would repeat identical work up to grid-size times over), in memory linear in the frame: no kernel
stacks its (anchors × targets × h) windows. The doctrine for each kernel rides its own
docstring; the runner composes them in ``_HorizonFrames``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from numba import njit

#: The runner's measurement-algebra closure: (numerator frame, denominator frame) → returns.
type MeasureFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def shift_rows(arr: np.ndarray, k: int) -> np.ndarray:
    """Pull each row up by ``k`` (``out[t] = arr[t+k]``), filling the trailing ``k`` rows with
    NaN."""
    out = np.full_like(arr, np.nan)
    if 0 <= k < arr.shape[0]:
        out[: arr.shape[0] - k] = arr[k:]
    return out


def shift_down(arr: np.ndarray, k: int) -> np.ndarray:
    """Push each row down by ``k`` (``out[t] = arr[t-k]``), filling the leading ``k`` rows with
    NaN — the mirror of :func:`shift_rows` for a BACKWARD look (the pre-entry drift window)."""
    out = np.full_like(arr, np.nan)
    if 0 <= k < arr.shape[0]:
        out[k:] = arr[: arr.shape[0] - k]
    return out


@njit(cache=True)
def _first_extremum_1d(
    a: npt.NDArray[np.float64], h: int, is_min: bool
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Per anchor ``t``: the offset in ``[0, h)`` of the FIRST bar of ``a[t .. t+h-1]`` attaining
    the window's minimum (``is_min``) or maximum, and that extremum — ``-1`` / NaN where the
    window holds a non-finite value or runs past the end.

    One monotonic-deque scan, O(n) whatever ``h``: the deque holds the window's candidate bars in
    index order, each strictly better than every later one — a new value evicts only the
    candidates it strictly beats, so an equal earlier bar stays in front and the first attainment
    wins a tie (``argmin``'s rule). A non-finite value empties the deque and marks every window
    holding it."""
    n = a.shape[0]
    offset = np.full(n, -1, dtype=np.int64)
    ext = np.full(n, np.nan)
    dq = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0
    last_bad = -1
    for e in range(n):
        v = a[e]
        if not np.isfinite(v):
            last_bad = e
            head = 0
            tail = 0
            continue
        while tail > head and (a[dq[tail - 1]] > v if is_min else a[dq[tail - 1]] < v):
            tail -= 1
        dq[tail] = e
        tail += 1
        t = e - h + 1
        if t < 0:
            continue
        while dq[head] < t:
            head += 1
        if last_bad < t:
            offset[t] = dq[head] - t
            ext[t] = a[dq[head]]
    return offset, ext


@njit(cache=True)
def _first_extrema(
    a: npt.NDArray[np.float64], h: int, is_min: bool
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """2-D :func:`_first_extremum_1d` per column."""
    offset = np.empty(a.shape, dtype=np.int64)
    ext = np.empty(a.shape)
    for j in range(a.shape[1]):
        offset[:, j], ext[:, j] = _first_extremum_1d(a[:, j], h, is_min)
    return offset, ext


def excursion_extremum(frame: np.ndarray, base: np.ndarray, h: int, which: str) -> np.ndarray:
    """The excursion-window extremum anchored at ``t``: full high/low over ``[t, t+h-1]`` plus
    ONLY the exit print ``base[t+h]`` — the mark the ``h``-bar return closes at.

    The exit bar's own high/low print AFTER the exit and belong to the NEXT holding period:
    including them (the pre-v3 inclusive window ``[t, t+h]``) let post-exit prices alter the
    reported path. ``h=1`` degenerates to the fill bar's own extremum plus the exit print, and
    the brackets ``mae <= 0 <= mfe`` (the fill bar's H/L straddle the fill open) and
    ``mae <= ret <= mfe`` (the exit print is a member of the extremum set) hold by construction.
    The H/L part is the extremum of the scan :func:`bars_to_trough_full` takes its trough index
    from (:func:`_first_extremum_1d`), so the two share ONE member set: a non-finite value
    anywhere in the H/L part invalidates, and a window the hole is not in keeps its own
    extremum. A non-finite exit print invalidates too, but such a row's ``ret`` is
    already NaN (unclosed), so no emitted number rides that leg. Right-censoring falls out of
    the joint mask: the H/L part censors ``t+h-1 >= n`` and the shifted exit print censors
    ``t > n-1-h``. On a custom-outcome frame (``frame is base`` — feed outcomes, series-shaped
    targets) the split is IDENTICAL to the old inclusive window on every closed row: the exit
    "print" is the measured series itself.
    """
    _, hl = _first_extrema(np.ascontiguousarray(frame, dtype=float), h, which == "min")
    exit_print = shift_rows(base, h)
    ext = np.minimum(hl, exit_print) if which == "min" else np.maximum(hl, exit_print)
    return np.where(np.isfinite(hl) & np.isfinite(exit_print), ext, np.nan)


def bars_to_positive_full(
    base_np: np.ndarray, h: int, sign: float, measure_fn: MeasureFn
) -> np.ndarray:
    """Full-anchor recovery timing, (rows × targets): ``out[f, g]`` = first j ∈ {1..h} where
    ``sign·measure(base[f+j, g], base[f, g]) >= 0``; NaN if never.

    j=0 is the fill itself (measure ≡ 0) and is skipped — recovery means returning to/above entry
    after at least one forward bar. A non-finite step before recovery, a non-finite entry, and a
    right-censored anchor (window past end) all yield NaN. A pure function of the anchor index
    given (horizon, target) — the same shape as ``mae`` — so it is computed ONCE per horizon over
    every anchor and indexed per cell, instead of per (combo × fill × j) with 1-element arrays.
    The equivalent per-anchor scan is a state machine over j = 1..h (non-finite → stop as NaN;
    first ``sign·v >= 0`` → j; exhaustion → NaN); the ``undecided`` latch below reproduces each
    transition on the same per-element floats, so the recorded counts are bit-identical.
    """
    n = base_np.shape[0]
    out = np.full(base_np.shape, np.nan)
    undecided = np.ones(base_np.shape, dtype=bool)
    for j in range(1, h + 1):
        v = sign * measure_fn(shift_rows(base_np, j), base_np)
        finite = np.isfinite(v)
        recover = finite & (v >= 0.0)
        out[undecided & recover] = float(j)
        undecided &= finite & ~recover
    # The per-fill scan skipped (→ NaN) anchors with a non-finite entry and anchors whose window
    # runs past the data end; ``measure_fn`` alone does not imply either (a ±inf entry can still
    # measure finite under pct), so both skips are restored explicitly.
    out[~np.isfinite(base_np)] = np.nan
    out[max(n - h, 0) :] = np.nan
    return out


def bars_to_trough_full(
    adverse_np: np.ndarray, base_np: np.ndarray, h: int, sign: float
) -> np.ndarray:
    """Full-anchor trough timing, (rows × targets): ``out[f, g]`` = first j ∈ {0..h} at which
    the excursion path attains its extremum (the MAE trough) — ``j < h`` indexes the H/L path
    over ``[f, f+h-1]``, ``j = h`` means the trough was attained at the exit print
    ``base[f+h]``.

    Long → the first minimum of the low path; short → the first maximum of the high path
    (first attainment on ties, argmin/argmax's rule), and a path-vs-exit-print tie resolves to
    the EARLIER (path) index — the strict inequality below. Validity = finite H/L part + finite
    exit print, the SAME member set as the mae extremum (:func:`excursion_extremum`), so the
    trough index always points into the window the mae was taken over; right-censored anchors
    are NaN. Computed once per horizon over every anchor (see ``bars_to_positive_full``); NOT
    derived from the per-horizon ``mae_ext`` frame + equality matching, which would mis-rank a
    window containing ±inf. The H/L windows are scanned once per column
    (:func:`_first_extremum_1d`, O(n) whatever ``h``): ``argmin`` over a ``sliding_window_view``
    of them copied the whole (anchors × targets × h) stack — 170 MiB for a 0.73 MiB frame of
    3,000 bars × 32 targets at h = 252, ~4.5 GiB at 5,000 × 500.
    Evidence for adverse-path trough duration: how many bars until the worst interim mark.
    """
    n = adverse_np.shape[0]
    out = np.full(adverse_np.shape, np.nan)
    if h + 1 > n:
        return out
    idx, hl_ext = _first_extrema(np.ascontiguousarray(adverse_np, dtype=float), h, sign > 0)
    idx, hl_ext = idx[: n - h], hl_ext[: n - h]  # H/L windows [t, t+h-1], t in 0..n-h-1
    exit_print = base_np[h:]  # base[t+h] for anchors t in 0..n-h-1
    valid = (idx >= 0) & np.isfinite(exit_print)
    exit_wins = (exit_print < hl_ext) if sign > 0 else (exit_print > hl_ext)
    out[: n - h] = np.where(valid, np.where(exit_wins, float(h), idx.astype(float)), np.nan)
    return out
