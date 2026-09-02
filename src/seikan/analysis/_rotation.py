"""The circular-shift rotation null and the two reliability passes composed on it."""

from __future__ import annotations

import math

import numpy as np

from seikan.analysis._hac import newey_west_mean, nonoverlap_count
from seikan.types import (
    CellKey,
    ReliabilityCell,
    ReliabilityRead,
    ReliabilitySummary,
)

_NAN = float("nan")


def _fwd_ffts(fwd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(rfft(zero-filled fwd), rfft(finite indicator))`` — the two forward-side spectra of the
    circular cross-correlations in :func:`_rotation_num_den`.

    A function of the forward column ALONE, so cells sharing a (horizon, target) share it —
    :func:`reliability_summary` memoizes it per distinct ``fwd`` content instead of recomputing
    both FFTs for every combo sibling.
    """
    fwd = np.asarray(fwd, dtype=float)
    finite = np.isfinite(fwd)
    f = np.where(finite, fwd, 0.0)
    valid = finite.astype(float)
    return np.fft.rfft(f), np.fft.rfft(valid)


def _rotation_num_den(
    mask: np.ndarray, f_fft: np.ndarray, valid_fft: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """The two circular cross-correlation channels of one firing mask against precomputed
    forward-side spectra — ``num[tau] = Σ_t roll(mask, tau)[t] · zerofilled(fwd)[t]`` and
    ``den[tau] = Σ_t roll(mask, tau)[t] · finite(fwd)[t]``, every shift at once via
    ``irfft(x_fft · conj(rfft(mask)))``; the circular cross-correlation
    ``r[tau] = sum_t s[(t-tau) mod T] * x[t]`` (== ``sum(roll(s, tau) * x)``) is
    ``irfft(rfft(x) * conj(rfft(s)))``. ``means[tau] = num[tau] / den[tau]`` is the conditional
    mean under shifting the mask by ``tau`` (``tau = 0`` the observed, no-shift value), the
    denominator recomputed per shift so NaN forward bars are handled correctly. Kept as the two
    raw channels so the POOLED null (:func:`pooled_reliability_summary`) can sum ``num``/``den``
    ACROSS members before the one division — the same FFT code either way, never duplicated."""
    s = (np.asarray(mask) > 0).astype(float)
    s_conj = np.conj(np.fft.rfft(s))
    num = np.fft.irfft(f_fft * s_conj, n=n)
    den = np.fft.irfft(valid_fft * s_conj, n=n)
    return num, den


def _ffts_for(
    fwd: np.ndarray, memo: dict[bytes, tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """The content-keyed FFT lookup both reliability passes share: one transform per DISTINCT
    forward series, keyed by its bytes (immune to synthetic inputs whose nominal key fields do
    not determine the column; the O(T) hash is cheap beside the FFT it saves)."""
    fkey = fwd.tobytes()
    ffts = memo.get(fkey)
    if ffts is None:
        ffts = memo[fkey] = _fwd_ffts(fwd)
    return ffts


def _tail_p(observed: float, null: np.ndarray) -> tuple[float, int]:
    """One-sided right-tail p with the (1+·)/(1+·) small-sample correction, and the number of
    DEFINED null values it was formed over. NaN-safe: an undefined shift (the rotated mask lands
    on no finite forward bar) drops out of the null, so the p's own floor is ``1/(1 + n_null)``
    — the per-cell resolution the runner reports beside the p as ``rot_n_null``, which on a
    sparse mask sits above the run-level ``1/(1 + n_shifts)``."""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    n_null = int(null.shape[0])
    if n_null == 0 or math.isnan(observed):
        return _NAN, n_null
    return float((1.0 + np.sum(null >= observed)) / (1.0 + n_null)), n_null


def _reliability_read(
    members: list[ReliabilityCell],
    length: int,
    shifts: np.ndarray,
    fft_memo: dict[bytes, tuple[np.ndarray, np.ndarray]],
) -> ReliabilityRead:
    """ONE overlap-aware read over a member list — the shared kernel of both reliability passes.

    The per-target pass hands in a single cell (G = 1, where the accumulate-then-divide rotation
    arithmetic reduces exactly to the direct division: ``0.0 + x`` is ``x`` in IEEE-754 for every
    finite ``x``, and the ``> 0.5`` denominator guard covers the rest); the pooled pass hands in
    a combo's members, whose rotation numerators/denominators SUM across the group (the
    common-shift null) and whose closed rows concatenate into one pooled pool.
    """
    h = int(members[0]["h"])  # shared across the group: a swept horizon rides the key prefix
    num_sum = np.zeros(length)
    den_sum = np.zeros(length)
    bars_parts: list[np.ndarray] = []
    rets_parts: list[np.ndarray] = []
    for cell in members:
        mask = np.asarray(cell["mask_col"])
        fwd = np.asarray(cell["fwd_col"], dtype=float)
        num, den = _rotation_num_den(mask, *_ffts_for(fwd, fft_memo), length)
        num_sum += num
        den_sum += den
        t_closed = np.flatnonzero((mask > 0) & np.isfinite(fwd))
        bars_parts.append(t_closed)
        rets_parts.append(fwd[t_closed])
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.where(den_sum > 0.5, num_sum / den_sum, _NAN)
    rot_p, rot_n_null = _tail_p(float(means[0]), means[shifts])
    bars = np.concatenate(bars_parts)
    rets = np.concatenate(rets_parts)
    n_nonoverlap = nonoverlap_count(bars, h)
    t_hac, hac_se = newey_west_mean(rets, bars, h)
    return {
        "rot_p": rot_p,
        "rot_n_null": rot_n_null,
        "t_hac": t_hac,
        "hac_se": hac_se,
        "n_nonoverlap": int(n_nonoverlap),
    }


def reliability_summary(
    cells: list[ReliabilityCell], length: int, targets: list[str]
) -> ReliabilitySummary:
    """Per-cell overlap-aware inference over the declared grid — one independent descriptive read
    per (param combo × horizon × target). Not a selector: nothing here compares cells.

    Each ``cell`` is ``{"key": (param values…, target), "mask_col": (T,), "fwd_col": (T,) signed,
    "h": horizon}`` — the LAST key element is the target. Every cell is measured on its OWN firing
    mask against its OWN circular-shift null, so two cells' numbers are directly comparable
    precisely because neither has been conditioned on the other, and adding a cell to the grid
    changes no other cell's numbers. ``targets`` names the regime the cells span; no cross-target
    statistic is formed from it here — the conjunction is reported target by target and the reader
    (or the per-cell checklist) applies the weakest-target rule.

    Per cell: ``rot_p`` — the one-sided right-tail p of the observed conditional mean against the
    circular-shift null, where the forward-return series is FIXED and only the firing mask rotates,
    preserving the firing count, the temporal clustering AND the overlap structure exactly;
    ``rot_n_null`` — how many of those shifts were DEFINED for this mask (a shift whose rotated
    mask lands on no finite forward bar contributes nothing), so the cell's own resolution is
    ``1/(1 + rot_n_null)``; ``t_hac``/``hac_se`` — the event-time Newey-West mean (Bartlett
    weights on actual entry-bar distances; a consumer's p reads it at df = n_nonoverlap − 1);
    and ``n_nonoverlap`` — the greedy non-overlapping window count that makes the overlap
    inflation visible (n = 400, n_nonoverlap = 18).

    The null uses ALL ``length − 1`` non-identity circular shifts: the FFT in
    :func:`_rotation_num_den` computes every shift anyway, and a deterministic evenly-spaced
    subsample suffers residue aliasing (on periodic signals whole shift phases go missing from the
    null). ``n_shifts`` is returned so a consumer can read the run-level resolution — the
    smallest attainable ``rot_p`` over a fully defined null is ``1/(1 + n_shifts)``, and a p at
    a cell's own floor means "no shift beat the observation", not "p ≈ 0".

    KNOWN CAVEATS, both anti-conservative, and the reason these numbers certify nothing on their
    own: the rotation null assumes the forward returns are exchangeable under time rotation, which
    fails when volatility clusters where the signal fires (rotated masks land in calm stretches and
    the null is too narrow); and the Bartlett taper downweights exactly the lags carrying the
    overlap covariance, so ``hac_se`` understates the long-run variance.

    Returns ``{"per_cell": {full_key: {rot_p, rot_n_null, t_hac, hac_se, n_nonoverlap}},
    "n_shifts": int}``. A cell that never fired has no entry — the runner tracks the DECLARED
    grid separately, so a missing key here reads as "no firings", never as "dropped". Memory:
    one ``(length,)`` float64 null array is live at a time (each cell's shifts collapse to a
    tail probability immediately), so
    the pass is O(length) in space regardless of grid size.
    """
    empty: ReliabilitySummary = {"per_cell": {}, "n_shifts": 0}
    if not cells or length < 3:
        return empty
    shifts = np.arange(1, length, dtype=int)
    n_shifts = shifts.shape[0]

    per_cell: dict[CellKey, ReliabilityRead] = {}
    # The forward-side FFTs are a function of `fwd_col` alone, which combo siblings sharing a
    # (horizon, target) pass in with identical content — memoized by content (bytes) so the O(T
    # log T) transforms run once per distinct forward series, not once per cell. Content-keyed
    # rather than (h, target)-keyed on purpose: it is immune to synthetic inputs whose key fields
    # do not determine the column, and the O(T) hash is cheap beside the FFT it saves.
    fft_memo: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    for cell in cells:
        per_cell[tuple(cell["key"])] = _reliability_read([cell], length, shifts, fft_memo)

    return {
        "per_cell": per_cell,
        "n_shifts": int(n_shifts),  # shifts actually used — min achievable p = 1/(1+n_shifts)
    }


def pooled_reliability_summary(cells: list[ReliabilityCell], length: int) -> ReliabilitySummary:
    """Basket-mode sibling of :func:`reliability_summary`: ONE pooled overlap-aware read per
    (param combo × horizon), over the union of that combo's member (target) pools.

    Consumes the SAME cell entries (``{"key": (param values…, target), "mask_col", "fwd_col",
    "h"}``), grouped by ``key[:-1]`` — the combo prefix — in first-seen order; a group's members
    share one ``h`` by construction (a swept horizon rides the prefix). ``G = 1`` reduces
    EXACTLY to the per-target read: same spectra, same division, same kernels on the same rows.

    ``rot_p`` — the COMMON-SHIFT rotation null: one shift ``tau`` rotates EVERY member's mask
    as a block. That is the null a cross-sectional signal warrants: it preserves each member's
    firing count, clustering and overlap structure (as the per-target null does) AND the
    per-bar cross-sectional firing pattern — a rank entry fires exactly k members per firing
    bar, and independent per-member shifts would scramble that pattern into masks the signal
    could never emit. Computed as ``pooled_means[tau] = Σ_g num_g[tau] / Σ_g den_g[tau]`` from
    the same per-member FFT spectra as the per-target null (:func:`_rotation_num_den`; the
    ``fwd``-content memo is reused, so members sharing a forward column share its transforms),
    defined where the POOLED denominator clears the same ``> 0.5`` guard; same shift set
    ``1..length−1``, same :func:`_tail_p`.

    ``t_hac`` / ``hac_se`` — :func:`newey_west_mean` over the CONCATENATED member rows (closed
    bars/returns appended in cells-list order, which the runner fixes as target-declaration
    order). Two same-bar observations sit ``d = 0`` apart, so their cross term enters at
    Bartlett weight ``max(0, 1 − 0/h) = 1``: one market move seen through several members is
    ONE cluster, priced at full covariance, never counted as independent evidence — the
    cluster-robust treatment for free, the kernel stays PSD under ties, and its canonical
    (bar, ret) lexsort makes the estimate bit-identical under any tied-row input order — the
    same-bar covariance is order-symmetric, so only the fp accumulation sequence was ever at
    stake. Performance note: pooled sorted bars repeat, so the
    kernel's event-lag loop runs ~(members × h) lags before its no-overlap break — still
    trivial beside the FFTs.

    ``n_nonoverlap`` — :func:`nonoverlap_count` over the concatenated entry bars: the SAME greedy
    non-overlapping kernel as everywhere else in the engine, deliberately NOT the episode
    count, so ``n_nonoverlap`` keeps exactly one meaning engine-wide; same-bar cross-member firings
    collapse to one independent window automatically (a duplicate bar adds nothing to a greedy
    non-overlap count).

    Returns ``{"per_cell": {combo_key: {rot_p, rot_n_null, t_hac, hac_se, n_nonoverlap}},
    "n_shifts": length − 1}``
    — the empty/too-short guard mirrors :func:`reliability_summary`. Grouping is structural,
    so adding a combo to the grid changes no other combo's pooled numbers: the same
    independence invariant as the per-target pass. Evidence-only, like its sibling — no check
    reads any of it directly; the runner mounts it under each basket cell's ``pooled`` block.
    """
    empty: ReliabilitySummary = {"per_cell": {}, "n_shifts": 0}
    if not cells or length < 3:
        return empty
    shifts = np.arange(1, length, dtype=int)

    groups: dict[CellKey, list[ReliabilityCell]] = {}
    for cell in cells:
        groups.setdefault(tuple(cell["key"])[:-1], []).append(cell)

    per_cell: dict[CellKey, ReliabilityRead] = {}
    fft_memo: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    for combo_key, members in groups.items():
        per_cell[combo_key] = _reliability_read(members, length, shifts, fft_memo)

    return {
        "per_cell": per_cell,
        "n_shifts": int(shifts.shape[0]),  # same resolution read as the per-target pass
    }
