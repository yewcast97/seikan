"""CSCV -> PBO: the combinatorially-symmetric-cross-validation probability of backtest overfitting
over the declared grid."""

from __future__ import annotations

import math
from collections.abc import Container, Mapping
from itertools import combinations
from typing import NamedTuple, cast

import numpy as np

from seikan.analysis._pools import (
    _NEAR_CONSTANT_ULPS,
    BarNumerators,
    _near_constant,
    _normal,
    exact_mean,
    exact_sum,
    parts_table,
)
from seikan.numeric import MIN_NORMAL
from seikan.types import (
    CellKey,
    DegradationSlopeReason,
    ParamValue,
    PboBlock,
    ReliabilityCell,
)

_NAN = float("nan")


#: Minimum scored symmetric splits for a PBO to be reported at all — S=4's full complement
#: C(4,2)=6. A "PBO" over 2-3 dependent splits has support too coarse to describe a search space
#: (it can flip on one split); below the floor the result is ``insufficient_data``.
_PBO_MIN_SPLITS = 6

#: The coarser partitions ``cscv_pbo`` falls back through when too few splits score at the
#: caller's S (sparse block-local pools typical of rare episodic firings).
_PBO_BLOCK_FALLBACKS = (6, 4)


class _Column(NamedTuple):
    """One scored column's per-block accumulators over its kept rows: ``moments`` (3, S) — the
    row counts, the exact block sums of the rows' shifts from the block's own offset
    (:func:`_offsets`, off its extrema), and the exact block sums of those shifts' squared
    deviations from their block's mean (each block's centered second moment) — ``extrema`` (2, S)
    the per-block ``(min, max)`` (``+inf`` / ``-inf`` on an empty block: a split's pool is EXACTLY
    constant iff its chosen blocks' min equals their max), and ``sums`` (S, J) each block's exact
    row sum times ``scale``, as exact parts (``_pools.parts_table``). Reducing them over a split's
    blocks makes its score a pure arithmetic reduction (:func:`_split_scores`)."""

    moments: np.ndarray
    extrema: np.ndarray
    sums: np.ndarray
    scale: int


def _kept_rows(cell: ReliabilityCell, edges: np.ndarray, off: int) -> tuple[np.ndarray, np.ndarray]:
    """``(t, b)``: a cell's fired rows with a finite forward return whose window closes inside
    their own block (block-local windows are pre-purged), and each one's block."""
    mask = np.asarray(cell["mask_col"]) > 0
    fwd = np.asarray(cell["fwd_col"], dtype=float)
    t = np.flatnonzero(mask & np.isfinite(fwd))
    b = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, edges.shape[0] - 2)
    keep = (t + off + int(cell["h"])) < edges[b + 1]
    return t[keep], b[keep]


def _offsets(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Each pool's offset — the midpoint of its range ``(lo, hi)``, the rule of
    :func:`~seikan.numeric.midrange` over arrays (the centre the pools' ddof=1 std and the kernels'
    cross-sections take): a value's deviation from it is exact wherever the pool lies within a
    factor of two of it. NaN on an empty block's ``(+inf, −inf)``."""
    out: np.ndarray = hi / 2.0 + lo / 2.0
    return out


def _column(b: np.ndarray, r: np.ndarray, parts: np.ndarray, scale: int, s: int) -> _Column:
    """The :class:`_Column` over rows in blocks ``b`` with returns ``r``, whose rows of exact
    ``parts`` sum to ``scale`` times the rows' sum: each row's own return over 1, or on a
    ``cross_mean`` grid the rows' bar numerators over the member count
    (``_pools.BarNumerators.at``). The second moments are two-pass about each BLOCK's own offset
    (:func:`_offsets`) — the block's shifts are exact wherever it lies within a factor of two of
    it, then each block's shifts about their own mean — and :func:`_split_scores` merges the
    blocks in the split's own frame, so a block keeps its spread whatever level the column's
    other blocks sit at. Shifted by the whole COLUMN's mean instead, a block far below the
    column's level lost its spread to that mean's ulp: one alternating 0 and 2**-56 beside blocks
    at 1 read every shift as −0.55, a variance of 0 and a split Sharpe of NaN where it is √(19/20),
    and two of a grid's 70 splits went unscored. A shift, a deviation or a square past the double
    range is no input to a score: :func:`_split_scores` nulls every split it reaches
    (``_representable``), so it overflows silently here."""
    n = np.bincount(b, minlength=s).astype(float)
    lo = np.full(s, np.inf)
    hi = np.full(s, -np.inf)
    np.minimum.at(lo, b, r)
    np.maximum.at(hi, b, r)
    with np.errstate(over="ignore", invalid="ignore"):
        shift = r - _offsets(lo, hi)[b]
        s1 = _block_sums(b, shift, s)
        dev = shift - (s1 / np.maximum(n, 1.0))[b]
        m2 = _block_sums(b, dev * dev, s)
    return _Column(
        np.stack((n, s1, m2)),
        np.stack((lo, hi)),
        parts_table([parts[b == g].ravel().tolist() for g in range(s)]),
        scale,
    )


def _block_sums(b: np.ndarray, w: np.ndarray, s: int) -> np.ndarray:
    """``(s,)``: each block's EXACTLY ROUNDED sum of the values ``w`` in it (``b`` is each value's
    block; 0.0 on an empty block) — one double whatever order the block's rows come in. A running
    sum (``np.bincount``'s weights) rounded in row order: two candidates holding the same returns
    in every block, in other orders, scored apart by their last bits, and every split between them
    — a tie in exact arithmetic — read a winner and a loser (``pbo`` 0.571 for 0)."""
    return np.array([exact_sum(w[b == g]) for g in range(s)])


def _disperses(lo: np.ndarray, hi: np.ndarray, mean: np.ndarray, eps: float) -> np.ndarray:
    """A pool with extrema ``(lo, hi)`` and mean ``mean`` has usable dispersion: it is not exactly
    constant (``hi > lo``; an empty pool's ``(+inf, -inf)`` is not either) and its largest
    deviation from the mean exceeds ``_NEAR_CONSTANT_ULPS`` ulps of the mean's magnitude."""
    with np.errstate(invalid="ignore"):
        spread = np.maximum(hi - mean, mean - lo)
        ok: np.ndarray = (hi > lo) & ~(spread < _NEAR_CONSTANT_ULPS * eps * np.abs(mean))
    return ok


def _representable(mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """The split's ``mean`` is finite and its ``var`` a positive NORMAL double (the
    ``_pools._normal`` rule) — the Sharpe ``mean / sd`` is read off representable moments only
    (see :func:`_split_scores`)."""
    ok: np.ndarray = np.isfinite(mean) & (var >= MIN_NORMAL) & (var < np.inf)
    return ok


def _split_scores(
    moments: np.ndarray,
    extrema: np.ndarray,
    sums: np.ndarray,
    scales: np.ndarray,
    block_idx: np.ndarray,
) -> np.ndarray:
    """Per-candidate split score over the chosen blocks — the per-observation Sharpe of each of
    the candidate's scored columns (:class:`_Column`), the weakest of them: under conjunction a
    column per target (the weakest-target score its caller selects on), under basket ONE column,
    the members' kept rows pooled (:func:`cscv_pbo`). ``moments`` (Q, C, 3, S), ``extrema``
    (Q, C, 2, S), ``sums`` (Q, C, S, J) and ``scales`` (Q, C) stack the columns' fields.

    A split's MEAN is its rows' EXACTLY ROUNDED sum — one ``math.fsum`` over its blocks' exact
    parts, over ``scale`` times its row count — so it has the exact sum's sign and is 0.0
    exactly where the rows cancel, however they fall across blocks. Rebuilt as the split's mean
    shift plus the column's mean, an exact 0 read the shift's residue (−2.6e-18), and a split
    whose block sums cancelled read the float sum over them (−0.25 for 0 off ``[2**54, 1, −2**54,
    −1]``): two candidates cycling through ±0.3 read ``pbo`` 2/70 and ``prob_oos_loss`` 1/70
    where no exact split mean was negative. On a ``cross_mean`` grid the parts are the rows' bar
    numerators over the member count (:class:`~seikan.analysis._pools.BarNumerators`): a CSCV
    block holds whole bars — its purge reads the bar and the cell's horizon alone, so a bar's rows
    land in one block or are all purged — and a bar's rounded per-row excesses sum to residue
    where its exact sum is 0 (every full bar; any bar whose selected excesses cancel), which the
    rows ranked in the order the targets were declared: on a grid of all-full candidates every
    split mean is exactly 0.0 and every scoreable candidate ties.

    A split's VARIANCE merges its blocks' centered moments, ``Σ m2_b + Σ n_b·(m_b − m)²`` — each
    block's squared deviations about its own mean, then each block mean's about the split's —
    every term a square, so nothing cancels at any level: the raw ``Σx² − n·mean²`` form read a
    diff outcome at level 1e9 as pure rounding noise, and its column-shifted twin refused as
    residue every split inside one regime of a column at ±0.01 with a 1e-9 spread. The block
    means are placed in the split's OWN frame — each block's offset (its range midpoint,
    :func:`_offsets`) less the split's, exact wherever the split lies within a factor of two of
    its midpoint, plus the block's mean shift — never in the column's: shifted by the column's
    mean, a block at 2**-56 beside blocks at 1 read its every shift as that mean's ulp, a variance
    of 0 and NaN for √(19/20), and a grid scored 68 of its 70 splits. The second moments read the
    rows' own returns — a row's excess is its own, only its bar's SUM is exact — so the Sharpe is
    the exact mean over the rows' sd.

    A split pool with no usable dispersion scores NaN: one whose chosen blocks' min equals their
    max is EXACTLY constant, and one whose largest deviation from its mean is within
    ``_NEAR_CONSTANT_ULPS`` ulps of the mean is constant to rounding — the gate the HAC applies to
    a cell's pool (``_pools._near_constant``), read here off the extrema, so it does not depend on
    what the column's OTHER blocks hold. A split whose mean left the double range (or met a bar
    whose sum did), or whose variance is not a normal double, scores NaN as well: ``mean / inf``
    is a FINITE 0.0 Sharpe (block moments can each stay finite while their merge overflows), an
    infinite mean would win every ranking, and a subnormal variance (returns under ~1e-154) has
    lost the bits the Sharpe divides by.
    """
    n_b, s1_b, m2_b = (moments[:, :, i][:, :, block_idx] for i in range(3))  # (Q, C, B) each
    lo_b, hi_b = (extrema[:, :, i][:, :, block_idx] for i in range(2))  # (Q, C, B) each
    lo_g, hi_g = lo_b.min(axis=2), hi_b.max(axis=2)  # (Q, C)
    chosen = sums[:, :, block_idx].reshape(scales.size, -1)  # each column's chosen parts
    total = np.array([exact_sum(v) for v in chosen]).reshape(scales.shape)
    eps = float(np.finfo(np.float64).eps)
    # Moments past the double range (returns past ~1e154 square out of it), an empty block's
    # offset (inf − inf, masked below) and splits with fewer than two rows: `_representable` and
    # the count gate decide every such split — an expected input, never a warning.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        n_g = n_b.sum(axis=2)
        mean = total / (scales * n_g)
        # each block's mean in the split's own frame (see the docstring); 0 on an empty block
        frame = _offsets(lo_b, hi_b) - _offsets(lo_g, hi_g)[:, :, None]
        m_b = np.where(n_b > 0, frame + s1_b / np.maximum(n_b, 1.0), 0.0)
        m = (n_b * m_b).sum(axis=2) / n_g
        ss_g = m2_b.sum(axis=2) + (n_b * (m_b - m[:, :, None]) ** 2).sum(axis=2)
        var_g = ss_g / (n_g - 1.0)
        scoreable_g = (n_g >= 2) & _disperses(lo_g, hi_g, mean, eps) & _representable(mean, var_g)
        sr = np.where(scoreable_g, mean / np.sqrt(var_g), np.nan)
    # The weakest column's Sharpe, one value per combo — an axis-wise reduction, which
    # numpy's own signature can only type as `Any` (its element-wise overload returns a
    # scalar); the cast states the (Q,) array it is here.
    return cast("np.ndarray", np.min(sr, axis=1))


def _degradation_slope(
    tr_a: np.ndarray, te_a: np.ndarray
) -> tuple[float | None, DegradationSlopeReason | None]:
    """Centered OLS slope of the splits' (train-winner, its test score) pairs, or
    ``(None, "degenerate_train_scores")`` when the train scores carry no usable spread.

    The degeneracy test is SCALE-RELATIVE — the pools' zero-dispersion gate
    (``_pools._near_constant``: a spread within 10 ulps of the scores' exactly rounded mean) —
    not exact-zero: a grid of near-clones passes an exact-zero test into a rank-deficient fit
    whose min-norm slope is an arbitrary number (numpy's polyfit additionally warned "poorly
    conditioned" — ~0.5 on one probe). Nor absolute: floored at a magnitude of 1, it read train
    scores spread from 0 to 3.5e-16 as degenerate and nulled an exact slope of −1. Their centered
    sum of squares must be a normal double too (``_pools._normal``): below it the fit divides by
    bits it has lost. The slope of an undefined regression is null with its reason said out
    loud, never a number. The well-conditioned path is the plain centered OLS identity
    cov(tr, te)/var(tr) — no Vandermonde, no RankWarning channel at all."""
    mean = exact_mean(tr_a)
    tr_c = tr_a - mean
    ss = float(tr_c @ tr_c)
    if _near_constant(tr_a, mean) or not _normal(ss):
        return None, "degenerate_train_scores"
    te_c = te_a - te_a.mean()
    return float(tr_c @ te_c) / ss, None


def cscv_pbo(
    cells: list[ReliabilityCell],
    length: int,
    targets: list[str],
    *,
    n_blocks: int = 8,
    off: int = 1,
    mode: str = "conjunction",
    n_combos_declared: int,
    bar_numerators: Mapping[CellKey, BarNumerators] | None = None,
) -> PboBlock:
    """CSCV over the runner's per-cell (mask, fwd) arrays → ``pbo`` + degradation diagnostics.

    Tries ``n_blocks`` then falls back through 6 → 4 when too few splits score (sparse
    block-local pools typical of rare episodic firings); a partition scoring fewer than
    ``_PBO_MIN_SPLITS`` valid splits is never reported as a PBO. Reports the S that worked in
    ``blocks``. Returns ONE dict — the runner mounts it nested at ``summary["pbo"]`` — carrying
    ``pbo`` (float, or None with ``reason`` set: "single_combo" — fewer than two DISTINCT
    candidates, a one-combo grid or a grid of byte-identical clones, has no selection to
    overfit; "insufficient_data" — too little block-local data even at S=4),
    ``n_splits``/``n_splits_attempted``/``blocks`` (the scored symmetric splits, the C(S, S/2)
    the partition offered, and the S that worked), ``n_candidates_min`` (the smallest finite
    candidate population any scored split ranked over — the per-split thinning the global
    ledger below cannot show), ``lambda_mean`` (mean rank logit),
    ``oos_degradation_slope`` (centered OLS of test-score on train-score of each split's winner;
    < 0 flags inverse transfer; null with ``oos_degradation_slope_reason`` =
    "degenerate_train_scores" when the train scores carry no usable spread — see
    :func:`_degradation_slope`), and ``prob_oos_loss`` (fraction of splits whose winner scored
    below zero on test). A split's mean is its rows' exactly rounded sum and its variance the
    merge of its blocks' centered moments (:func:`_split_scores`), so a mean has its exact sign
    and level-heavy return columns score instead of cancelling to zero variance. Grid-level
    evidence — no cell's grade reads it.

    The population is ledgered, never assumed: ``n_combos <= n_combos_scoreable <=
    n_combos_declared`` — the DISTINCT candidates entering CSCV (the population a scored block
    ranks over), the combos admissible with at least one fired-and-closed pool (pre-collapse),
    and the caller's declared combo × horizon grid (the runner passes
    ``n_hypotheses_attempted``). CSCV never sees a declared combo that produced no closed pool,
    so the ledger is what says how far short of the declared grid the scored population fell.
    It is a property of the cells alone, established before any length or partition check, so
    a DECLINED block, whatever its reason, carries the real counts — never zeros that would
    claim no candidate existed. Byte-identical candidate combos (same per-target masks, forward
    returns and horizons — a sweep axis that never moved the mask) collapse to ONE
    declaration-order representative BEFORE scoring: under selection-rank arithmetic M clones
    would hand the train winner the worst test rank purely for tying with itself. Test ranks
    are MIDRANKS (ties share their average rank; a full tie is λ = 0 — a split that cannot
    distinguish candidates is not evidence of overfitting), and the training winner's tie-break
    is first-index / declaration order (``np.nanargmax``), deterministic by construction. A
    column's block moments and sums are exact, the same doubles in any row order, so two
    candidates holding the same returns in every block tie on every split — under basket however
    the returns fall across members.

    ``mode`` is the thesis's declared ``target_mode`` and sets the SPLIT SCORE — the score
    mirrors the statistic the caller of that mode selects on, because PBO describes the
    CALLER'S selection (the engine has none). ``"conjunction"`` (default): a combo is
    admissible only when EVERY regime target has a pool, and its score is its WEAKEST target's
    per-observation Sharpe — the weakest-target rule the conjunction implies.
    ``"basket"``: the members form ONE pool, so a combo is admissible with ≥ 1 member present —
    an absent member adds no rows, exactly a member that never fired — and its score is the
    per-observation Sharpe of ONE column, its members' kept rows pooled (:func:`_column`). No
    min: a basket caller selects on the pooled read. Scored off per-member moments, each member's
    rows about its own mean and pooled through the cross-term expansion, two candidates holding
    the same six returns in every block, split between the members differently, parted in their
    last bits and a full tie read ``pbo`` 0.343.

    ``bar_numerators`` (basket only; the runner's per-cell ``cross_mean`` bar numerators over
    ``targets``, keyed by the combo prefix — ``_pools.cross_mean_bar_numerators``; a candidate it
    does not key sums its rows' own returns) carries each bar's exact numerator on its first
    closed member, zeros on the others: every split's mean is their exactly rounded sum over the
    member count times its row count — the read the cell's pooled panel's bar-closed sums take —
    so a split's mean is the exact sum of the excesses it holds, never rounding residue, and on
    a grid of all-full candidates every scored split is a full tie (λ = 0, no test loss: a
    scored block reads ``pbo``, ``lambda_mean`` and ``prob_oos_loss`` 0.0). Its tied train
    winner is the FIRST-declared scoreable candidate (the tie rule above),
    and a split whose winner holds no test pool is skipped, so a sparse such grid can score
    fewer splits, fall back to a coarser S or decline ``insufficient_data`` — identically in
    any target declaration order. Every other grid — conjunction, a market or unbenchmarked
    basket — scores without it.
    """
    if mode not in ("conjunction", "basket"):
        raise ValueError(f"unknown target_mode {mode!r}")

    targets = list(targets)  # one local copy; iterated many times below

    # Combo admissibility mirrors the mode's evidence rule: conjunction demands every regime
    # target's pool (a combo missing one has no weakest member to score), basket demands ≥ 1
    # member (the pooled pool exists; an absent member adds no rows, not missing evidence).
    quantifier = any if mode == "basket" else all

    def _admits(present: Container[ParamValue | str]) -> bool:
        return quantifier(tg in present for tg in targets)

    # Candidate roster in declaration order (dict insertion), grouped by combo prefix; the
    # admissibility filter and the duplicate collapse both run HERE, before any block edges
    # exist — presence is a property of the cells, not of a partition, so no per-S re-check can
    # ever disagree with this one.
    by_combo: dict[CellKey, list[ReliabilityCell]] = {}
    for cell in cells:
        key = tuple(cell["key"])
        by_combo.setdefault(key[:-1], []).append(cell)
    admissible = {
        combo: members
        for combo, members in by_combo.items()
        if _admits({tuple(m["key"])[-1] for m in members})
    }
    n_combos_scoreable = len(admissible)
    # Byte-identical candidates collapse to ONE (declaration-order representative): identity is
    # the full-resolution pool — per-member (target, h, mask, fwd) bytes — so a sweep axis that
    # never moved the mask cannot mint candidates, and M clones cannot hand the train winner
    # the worst test rank purely for tying with itself.
    seen_fingerprints: set[tuple[tuple[str, int, bytes, bytes], ...]] = set()
    kept_cells: list[ReliabilityCell] = []
    n_distinct = 0
    for members in admissible.values():
        fingerprint = tuple(
            sorted(
                (
                    str(tuple(m["key"])[-1]),
                    int(m["h"]),
                    (np.asarray(m["mask_col"]) > 0).tobytes(),
                    np.asarray(m["fwd_col"], dtype=float).tobytes(),
                )
                for m in members
            )
        )
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        n_distinct += 1
        kept_cells.extend(members)

    # The declined block carries the population ledger established ABOVE — admissibility and the
    # collapse read the cells alone, never a partition or the sample length — so EVERY declined
    # path (a sample under 4 bars, fewer than two distinct candidates, a sample no block
    # partition fits, too few scored splits at every S) reports the candidate population a
    # scored block over these cells would. Zeros there claimed no candidate ever existed when
    # the sample was merely too thin to split them.
    empty: PboBlock = {
        "pbo": None,
        "reason": "insufficient_data",
        "n_splits": 0,
        "n_splits_attempted": 0,
        "n_candidates_min": 0,
        "n_combos": n_distinct,
        "n_combos_scoreable": n_combos_scoreable,
        "n_combos_declared": int(n_combos_declared),
        "blocks": int(n_blocks),
        "lambda_mean": _NAN,
        "oos_degradation_slope": _NAN,
        "oos_degradation_slope_reason": None,
        "prob_oos_loss": _NAN,
    }
    if not cells or length < 4:
        return empty
    if n_distinct < 2:
        return {**empty, "reason": "single_combo"}

    # Adaptive block counts: prefer the caller's S, then the coarser fallback partitions for
    # sparse pools; dict.fromkeys dedupes while keeping the preference order.
    block_candidates = [
        s
        for s in dict.fromkeys((n_blocks, *_PBO_BLOCK_FALLBACKS))
        if s >= 2 and s % 2 == 0 and length >= 2 * s
    ]
    # The bar-closed sums a `cross_mean` grid's split means read (basket only); every other
    # grid builds no pool tensor.
    numerators_by_combo = bar_numerators if mode == "basket" and bar_numerators is not None else {}
    member: dict[ParamValue | str, int] = {tg: g for g, tg in enumerate(targets)}

    last_empty = empty
    for s in block_candidates:
        edges = np.linspace(0, length, s + 1).astype(int)
        # Each candidate's kept rows per member — blocks, returns and the rows' exact parts:
        # each row's own return or, on a `cross_mean` grid, its bar numerator (each bar's exact
        # parts on its first closed member, zeros on the others; the purge keeps or drops a bar's
        # rows together — it reads the bar and the cell's horizon alone — so each block holds
        # whole bars). Per-S by necessity: the purge depends on this partition's edges. Every
        # surviving cell belongs to an admissible, distinct candidate (filtered and collapsed
        # above, S-independently).
        rows_by_combo: dict[CellKey, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        for cell in kept_cells:
            key = tuple(cell["key"])
            t, b = _kept_rows(cell, edges, off)
            r = np.asarray(cell["fwd_col"], dtype=float)[t]
            numerators = numerators_by_combo.get(key[:-1])
            parts = r[:, None] if numerators is None else numerators.at(t, member[key[-1]])
            rows_by_combo.setdefault(key[:-1], []).append((b, r, parts))
        # The scored columns: a conjunction candidate's per target, a basket candidate's ONE —
        # its members' rows pooled, so its block sums are those of the pool its caller reads,
        # the same doubles however the rows fall across members.
        columns = [
            (
                [
                    _column(
                        np.concatenate([b for b, _, _ in rows]),
                        np.concatenate([r for _, r, _ in rows]),
                        np.concatenate([p for _, _, p in rows]),
                        len(targets) if combo in numerators_by_combo else 1,
                        s,
                    )
                ]
                if mode == "basket"
                else [_column(b, r, parts, 1, s) for b, r, parts in rows]
            )
            for combo, rows in rows_by_combo.items()
        ]
        n_combos = len(columns)
        moments = np.array([[c.moments for c in cols] for cols in columns])  # (Q, C, 3, S)
        extrema = np.array([[c.extrema for c in cols] for cols in columns])  # (Q, C, 2, S)
        j = max(c.sums.shape[1] for cols in columns for c in cols)
        sums = np.array(
            [[np.pad(c.sums, ((0, 0), (0, j - c.sums.shape[1]))) for c in cols] for cols in columns]
        )  # (Q, C, S, J)
        scales = np.array([[c.scale for c in cols] for cols in columns])  # (Q, C)

        lambdas: list[float] = []
        pairs: list[tuple[float, float]] = []
        # The per-split candidate population: a candidate with too little block-local data
        # carries no finite test score and drops out of THAT split's ranking (its rank
        # denominator is the finite count), and a split whose train winner has none is skipped
        # outright. Canonical CSCV assumes a fixed candidate count, so both thinnings are
        # ledgered — ``n_splits_attempted`` beside ``n_splits``, and the smallest finite
        # candidate count any scored split ranked over — rather than averaged away silently.
        n_splits_attempted = 0
        n_candidates_min = n_combos
        for train in combinations(range(s), s // 2):
            n_splits_attempted += 1
            train_idx = np.asarray(train)
            test_idx = np.setdiff1d(np.arange(s), train_idx)
            tr = _split_scores(moments, extrema, sums, scales, train_idx)
            if np.all(np.isnan(tr)):
                continue
            bi = int(np.nanargmax(tr))
            te = _split_scores(moments, extrema, sums, scales, test_idx)
            te_best = te[bi]
            finite_te = te[~np.isnan(te)]
            if math.isnan(te_best) or finite_te.shape[0] < 2:
                continue
            # MIDRANK among the finite test scores, rank 1 = worst: ties share their average
            # rank, so a full tie — every candidate identical on test — lands at (N+1)/2,
            # ω = 0.5, λ = 0: a split that cannot DISTINGUISH candidates is not evidence of
            # overfitting. (Strict `<` + 1 handed the winner the WORST rank for tying, reading
            # a grid of near-clones as maximal overfitting.) Untied scores reproduce the
            # classic count-strictly-worse + 1 exactly.
            worse = float(np.sum(finite_te < te_best))
            tied = float(np.sum(finite_te == te_best))  # >= 1: te_best itself is finite here
            rank = worse + (tied + 1.0) / 2.0
            n_candidates_min = min(n_candidates_min, int(finite_te.shape[0]))
            omega = rank / (finite_te.shape[0] + 1.0)
            lambdas.append(math.log(omega / (1.0 - omega)))
            pairs.append((float(tr[bi]), float(te_best)))

        if len(lambdas) < _PBO_MIN_SPLITS:
            # `empty` already carries the population ledger (`n_combos` IS `n_distinct`: every
            # kept cell has its triple, so the roster here is exactly the distinct candidates).
            last_empty = {
                **empty,
                "n_splits": len(lambdas),
                "n_splits_attempted": n_splits_attempted,
                "n_candidates_min": n_candidates_min if lambdas else 0,
                "blocks": s,
            }
            continue
        lam = np.asarray(lambdas)
        tr_a = np.array([p[0] for p in pairs])
        te_a = np.array([p[1] for p in pairs])
        slope, slope_reason = _degradation_slope(tr_a, te_a)
        return {
            "pbo": float(np.mean(lam < 0.0)),
            "reason": None,
            "n_splits": int(lam.shape[0]),
            "n_splits_attempted": n_splits_attempted,
            "n_candidates_min": n_candidates_min,
            "n_combos": n_combos,
            "n_combos_scoreable": n_combos_scoreable,
            "n_combos_declared": int(n_combos_declared),
            "blocks": int(s),
            "lambda_mean": float(np.mean(lam)),
            "oos_degradation_slope": slope,
            "oos_degradation_slope_reason": slope_reason,
            "prob_oos_loss": float(np.mean(te_a < 0.0)),
        }

    return last_empty
