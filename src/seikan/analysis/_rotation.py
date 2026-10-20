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

#: The constant of the rotation null's tie band. Each FFT-computed cross-correlation value
#: ``num[tau]`` carries a rounding error of order ``ε · log2(T) · ‖f‖₂ · √k`` (``f`` the
#: zero-filled forward column, ``k`` the mask's firing count — Higham's forward-error envelope for
#: an FFT convolution: two forward transforms, a product, an inverse; the inverse term is an
#: estimate that assumes the rounding errors spread across the outputs, not a rigorous
#: per-element bound). Measured per-value errors stay below ~3x that unscaled envelope in every
#: configuration tried (constant, dense-mask columns — the ones the band exists for — are the
#: worst case), so ``C = 16`` keeps a margin of at least ~5x. The band is ABSOLUTE and scales with
#: the column's 2-norm — the transform's own error does — which on return columns is ~1e-13 to
#: 1e-11 of the return scale, far below the O(σ/√k) spacing of any null with real dispersion. It
#: is formed, like the transform, in the column's unit frame (:func:`_ffts_for`).
_FFT_TIE_C = 16.0

#: One forward column's rotation inputs, in its UNIT FRAME (:func:`_ffts_for`): the two spectra
#: of :func:`_fwd_ffts`, the framed column's 2-norm, and the frame's exponent ``e`` (the column
#: is ``fwd · 2**-e``).
type _Spectra = tuple[np.ndarray, np.ndarray, float, int]


def _fft_tie_band(n: int, f_norm: float, n_fire: int) -> float:
    """The absolute error envelope of one member's FFT-computed ``num[tau]`` (any ``tau``), for a
    column of 2-norm ``f_norm`` — a unit-frame column's, at most ``√n``."""
    eps = float(np.finfo(np.float64).eps)
    return _FFT_TIE_C * eps * math.log2(n) * f_norm * math.sqrt(n_fire)


def _fwd_ffts(fwd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(rfft(zero-filled fwd), rfft(finite indicator))`` — the two forward-side spectra of the
    circular cross-correlations in :func:`_rotation_num_den`.

    A function of the forward column ALONE, so cells sharing a (horizon, target) share it
    (:func:`_ffts_for`'s content-keyed memo).
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
    ACROSS members before the one division — the same FFT code either way, never duplicated.
    ``den`` counts coincidences, an integer the transform only approximates (residue ~1e-13), so
    it is returned ROUNDED — exact, like the count it is."""
    s = (np.asarray(mask) > 0).astype(float)
    s_conj = np.conj(np.fft.rfft(s))
    num = np.fft.irfft(f_fft * s_conj, n=n)
    den = np.rint(np.fft.irfft(valid_fft * s_conj, n=n))
    return num, den


def _ffts_for(fwd: np.ndarray, memo: dict[bytes, _Spectra]) -> _Spectra:
    """The content-keyed FFT lookup both reliability passes share: one transform per DISTINCT
    forward series, keyed by its bytes (immune to synthetic inputs whose nominal key fields do
    not determine the column; the O(T) hash is cheap beside the FFT it saves).

    The transform runs in the column's UNIT FRAME: ``fwd · 2**-e``, ``e`` the ``frexp`` exponent
    of its largest finite magnitude (0 for a column of zeros), so that magnitude lies in
    ``[0.5, 1)``. The FFT is bit-exactly equivariant under a power-of-two scaling that leaves its
    arithmetic in range, so every column the raw transform could carry reads the same comparisons
    in the frame, and the rest stay in range: raw, a column near the double ceiling overflows its
    partial sums, and a SUBNORMAL one runs the transform on values of a few bits, its ties read
    as rounding noise. Scaling up never rounds; scaling down drops only bits far inside the
    transform's own error. Returns the two spectra of the framed column, its 2-norm (the scale of
    their rounding error) and ``e``."""
    fkey = fwd.tobytes()
    ffts = memo.get(fkey)
    if ffts is None:
        finite = np.isfinite(fwd)
        e = math.frexp(float(np.max(np.abs(fwd[finite]), initial=0.0)))[1]
        framed = np.ldexp(fwd, -e)
        f_norm = float(np.linalg.norm(np.where(finite, framed, 0.0)))
        ffts = memo[fkey] = (*_fwd_ffts(framed), f_norm, e)
    return ffts


def _tail_p(
    observed: float,
    null: np.ndarray,
    tol_obs: float = 0.0,
    tol_null: np.ndarray | None = None,
) -> tuple[float, int]:
    """One-sided right-tail p with the (1+·)/(1+·) small-sample correction, and the number of
    DEFINED null values it was formed over. NaN-safe: an undefined shift (the rotated mask lands
    on no finite forward bar) drops out of the null, so the p's own floor is ``1/(1 + n_null)``
    — the per-cell resolution the runner reports beside the p as ``rot_n_null``, which on a
    sparse mask sits above the run-level ``1/(1 + n_shifts)``. An observed mean past the double
    range (its sum overflowed) is no observation either: NaN, never the p floor an ``inf`` would
    claim above every finite null value.

    ``tol_obs`` / ``tol_null`` are the absolute error tolerances of the observed and of each null
    value (estimated envelopes — see ``_FFT_TIE_C`` — not rigorous bounds): a null value within
    their sum below the observed one is a TIE the arithmetic cannot
    resolve, and ties count as ``>=`` — conservative by construction (a band can only raise p).
    Zero by default, the exact comparison."""
    null = np.asarray(null, dtype=float)
    keep = np.isfinite(null)
    band = tol_obs + (0.0 if tol_null is None else np.asarray(tol_null, dtype=float)[keep])
    null = null[keep]
    n_null = int(null.shape[0])
    if n_null == 0 or not math.isfinite(observed):
        return _NAN, n_null
    return float((1.0 + np.sum(null >= observed - band)) / (1.0 + n_null)), n_null


def _reliability_read(
    members: list[ReliabilityCell],
    length: int,
    shifts: np.ndarray,
    fft_memo: dict[bytes, _Spectra],
) -> ReliabilityRead:
    """ONE overlap-aware read over a member list — the shared kernel of both reliability passes.

    The per-target pass hands in a single cell (G = 1, where the accumulate-then-divide rotation
    arithmetic reduces exactly to the direct division: ``0.0 + x`` is ``x`` in IEEE-754 for every
    finite ``x``, and the ``> 0.5`` denominator guard covers the rest); the pooled pass hands in
    a combo's members, whose rotation numerators/denominators SUM across the group (the
    common-shift null) and whose closed rows concatenate into one pooled pool.

    The rotation means live in ONE frame, the largest member's unit frame (:func:`_ffts_for`;
    a member of zeros sets none): each member's numerator and tie band come into it by an exact
    power of two, the counts ``den`` are scale-free, and so is the p — nothing is scaled back.
    No mean in the frame can leave the double range (each member's column is bounded by 1).

    Ties are judged within the transform's rounding: each member's ``num`` carries an error well
    inside :func:`_fft_tie_band` at every shift (an estimated envelope with a safety factor, see
    ``_FFT_TIE_C``), the bands add across members, and the exact (rounded) ``den`` divides the
    sum — so a forward series of identical returns, whose every shift ties the observed mean
    exactly, reads ``rot_p == 1`` at any scale, not the coin flips of its rounding residue.
    """
    h = int(members[0]["h"])  # shared across the group: a swept horizon rides the key prefix
    fwds = [np.asarray(cell["fwd_col"], dtype=float) for cell in members]
    spectra = [_ffts_for(fwd, fft_memo) for fwd in fwds]
    frame = max((e for _, _, f_norm, e in spectra if f_norm > 0.0), default=0)
    num_sum = np.zeros(length)
    den_sum = np.zeros(length)
    tol_num = 0.0
    bars_parts: list[np.ndarray] = []
    rets_parts: list[np.ndarray] = []
    for cell, fwd, (f_fft, valid_fft, f_norm, e) in zip(members, fwds, spectra, strict=True):
        mask = np.asarray(cell["mask_col"])
        num, den = _rotation_num_den(mask, f_fft, valid_fft, length)
        num_sum += np.ldexp(num, e - frame)
        den_sum += den
        band = _fft_tie_band(length, f_norm, int(np.count_nonzero(mask > 0)))
        tol_num += math.ldexp(band, e - frame)
        t_closed = np.flatnonzero((mask > 0) & np.isfinite(fwd))
        bars_parts.append(t_closed)
        rets_parts.append(fwd[t_closed])
    defined = den_sum > 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.where(defined, num_sum / den_sum, _NAN)
        tol = np.where(defined, tol_num / den_sum, _NAN)
    # Every DEFINED shift's mean is finite in the frame (each member's column is bounded by 1
    # there, and a defined `den` is at least 1), so the null's size is the count of defined
    # shifts — the mask's geometry, off the exact, rounded `den`.
    rot_p, rot_n_null = _tail_p(float(means[0]), means[shifts], float(tol[0]), tol[shifts])
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


def reliability_summary(cells: list[ReliabilityCell], length: int) -> ReliabilitySummary:
    """Per-cell overlap-aware inference over the declared grid — one independent descriptive read
    per (param combo × horizon × target). Not a selector: nothing here compares cells.

    Each ``cell`` is ``{"key": (param values…, target), "mask_col": (T,), "fwd_col": (T,) signed,
    "h": horizon}`` — the LAST key element is the target. Every cell is measured on its OWN firing
    mask against its OWN circular-shift null, so two cells' numbers are directly comparable
    precisely because neither has been conditioned on the other, and adding a cell to the grid
    changes no other cell's numbers. No cross-target statistic is formed here — the conjunction is
    reported target by target and the reader (or the per-cell checklist) applies the
    weakest-target rule.

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
    the FFT memo holds one pair of spectra per distinct forward column (per horizon × target),
    and one ``(length,)`` null array is live at a time (each cell's shifts collapse to a tail
    probability immediately).
    """
    empty: ReliabilitySummary = {"per_cell": {}, "n_shifts": 0}
    if not cells or length < 3:
        return empty
    shifts = np.arange(1, length, dtype=int)
    n_shifts = shifts.shape[0]

    per_cell: dict[CellKey, ReliabilityRead] = {}
    # One transform per distinct forward series (`_ffts_for`'s content key): combo siblings
    # sharing a (horizon, target) pass in identical `fwd_col` content.
    fft_memo: dict[bytes, _Spectra] = {}
    for cell in cells:
        per_cell[tuple(cell["key"])] = _reliability_read([cell], length, shifts, fft_memo)

    return {
        "per_cell": per_cell,
        "n_shifts": int(n_shifts),  # shifts actually used — min achievable p = 1/(1+n_shifts)
    }


def pooled_reliability_summary(cells: list[ReliabilityCell], length: int) -> ReliabilitySummary:
    """Basket-mode sibling of :func:`reliability_summary`: ONE pooled overlap-aware read per
    (param combo × horizon), over the union of that combo's member (target) pools.

    Consumes the SAME cell-entry shape (``{"key": (param values…, target), "mask_col",
    "fwd_col", "h"}``), grouped by ``key[:-1]`` — the combo prefix — in first-seen order; a
    group's members share one ``h`` by construction (a swept horizon rides the prefix). The runner
    hands it the basket's own roster — every FIRED member's complete mask, closed rows or not, a
    superset of the per-target pass's closed-only cells — so a member with no closed row (every
    firing censored — past the data end or by a hole — or its only firing on the final bar)
    still rotates with the block while adding nothing to the rows. ``G = 1`` reduces EXACTLY to
    the per-target read: same spectra, same division, same kernels on the same rows.

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
    order); under ``cross_mean`` the runner replaces the pair with
    :func:`~seikan.analysis._hac.cross_mean_pooled_hac`, the same estimator over every bar's exact
    excess sum. Two same-bar observations sit ``d = 0`` apart, so their cross term enters at
    Bartlett weight ``max(0, 1 − 0/h) = 1``: one market move seen through several members is
    ONE cluster, priced at full covariance, never counted as independent evidence — the
    cluster-robust treatment for free, the kernel stays PSD under ties, and its exactly rounded
    per-bar deviation sums make the estimate bit-identical under any tied-row input order (the
    same-bar covariance is order-symmetric; only an fp accumulation order could move it).
    Performance note: the kernel's lag loop runs over the distinct entry bars, ~h lags before its
    no-overlap break — trivial beside the FFTs.

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
    fft_memo: dict[bytes, _Spectra] = {}
    for combo_key, members in groups.items():
        per_cell[combo_key] = _reliability_read(members, length, shifts, fft_memo)

    return {
        "per_cell": per_cell,
        "n_shifts": int(shifts.shape[0]),  # same resolution read as the per-target pass
    }
