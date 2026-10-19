"""CSCV -> PBO: the combinatorially-symmetric-cross-validation probability of backtest overfitting
over the declared grid."""

from __future__ import annotations

import math
from collections.abc import Container, Mapping, Sequence
from itertools import combinations
from typing import NamedTuple, cast

import numpy as np

from seikan.analysis._pools import (
    _EPS,
    _NEAR_CONSTANT_ULPS,
    BarNumerators,
    _fsum,
    _near_constant,
    _normal,
    exact_mean,
    integer_terms,
    parts_table,
)
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
    """One scored column's per-block reads over its kept rows, one entry per block: ``counts``
    the row counts; ``lo`` / ``hi`` the extrema (``+inf`` / ``-inf`` on an empty block: a split's
    pool is EXACTLY constant iff its chosen blocks' min equals their max); ``sums`` each block's
    exact row sum times ``scale``, as exact parts (``_pools.parts_table``); and ``first`` /
    ``second`` the rows' moments as EXACT integers — every return ``P / den`` over the column's
    one power-of-two denominator (``_pools.integer_terms``), ``first[g] = ΣP`` and ``second[g]
    = ΣP²`` over block ``g``'s rows. A split's score is a pure arithmetic reduction of its blocks'
    entries (:func:`_split_score`)."""

    counts: list[int]
    lo: list[float]
    hi: list[float]
    sums: list[list[float]]
    scale: int
    first: list[int]
    second: list[int]
    den: int


def _kept_rows(cell: ReliabilityCell, edges: np.ndarray, off: int) -> tuple[np.ndarray, np.ndarray]:
    """``(t, b)``: a cell's fired rows with a finite forward return whose window closes inside
    their own block (block-local windows are pre-purged), and each one's block."""
    mask = np.asarray(cell["mask_col"]) > 0
    fwd = np.asarray(cell["fwd_col"], dtype=float)
    t = np.flatnonzero(mask & np.isfinite(fwd))
    b = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, edges.shape[0] - 2)
    keep = (t + off + int(cell["h"])) < edges[b + 1]
    return t[keep], b[keep]


def _column(b: np.ndarray, r: np.ndarray, parts: np.ndarray, scale: int, s: int) -> _Column:
    """The :class:`_Column` over rows in blocks ``b`` with returns ``r``, whose rows of exact
    ``parts`` sum to ``scale`` times the rows' sum: each row's own return over 1, or on a
    ``cross_mean`` grid the rows' bar numerators over the member count
    (``_pools.BarNumerators.at``). Every read is exact — integers, exact parts, extrema — so a
    column's blocks carry the same entries whatever order their rows come in."""
    lo = np.full(s, np.inf)
    hi = np.full(s, -np.inf)
    np.minimum.at(lo, b, r)
    np.maximum.at(hi, b, r)
    nums, den = integer_terms(r.tolist())
    first, second = [0] * s, [0] * s
    for g, p in zip(b.tolist(), nums, strict=True):
        first[g] += p
        second[g] += p * p
    return _Column(
        np.bincount(b, minlength=s).tolist(),
        lo.tolist(),
        hi.tolist(),
        parts_table([parts[b == g].ravel().tolist() for g in range(s)]).tolist(),
        scale,
        first,
        second,
        den,
    )


def _disperses(lo: float, hi: float, mean: float) -> bool:
    """A pool with extrema ``(lo, hi)`` and mean ``mean`` has usable dispersion: it is not exactly
    constant (``hi > lo``) and its largest deviation from the mean exceeds
    ``_NEAR_CONSTANT_ULPS`` ulps of the mean's magnitude."""
    return hi > lo and not max(hi - mean, mean - lo) < _NEAR_CONSTANT_ULPS * _EPS * abs(mean)


def _split_variance(col: _Column, blocks: Sequence[int], n: int) -> float:
    """The ddof=1 variance of the ``n`` rows in ``col``'s ``blocks``, EXACT and rounded once:
    ``(n·ΣP² − (ΣP)²) / (den²·n·(n − 1))`` over the blocks' integer moments (:class:`_Column`),
    one quotient of integers, correctly rounded by Python's true division — ``inf`` where it
    leaves the double range (its ``OverflowError``)."""
    s1 = sum(col.first[g] for g in blocks)
    s2 = sum(col.second[g] for g in blocks)
    try:
        return (n * s2 - s1 * s1) / (col.den * col.den * n * (n - 1))
    except OverflowError:
        return math.inf


def _split_score(col: _Column, blocks: Sequence[int]) -> float:
    """The per-observation Sharpe of ``col``'s rows in the chosen ``blocks``, or NaN.

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

    A split's VARIANCE is exact arithmetic's, rounded once (:func:`_split_variance`): its blocks'
    integer moments merge exactly, so it is a function of the split's exact count, sum and sum of
    squares alone — at any level, in any row order, whatever the held-out blocks hold. Merged
    from the blocks' centered moments, ``Σ m2_b + Σ n_b·(m_b − m)²``, every row's squared
    deviation was rounded on its own: two candidates holding ``α·[10, 13, 13, 16]`` and
    ``α·[11, 11, 14, 16]`` in every block — the same count, sum and sum of squares, so the same
    Sharpe on every split — scored apart in their last bits (16 of one grid's 70 splits), and 99
    of 100 such grids read a nonzero ``pbo`` (up to 0.214) where every split is a full tie. The
    second moments read the rows' own returns — a row's excess is its own, only its bar's SUM is
    exact — so the Sharpe is the exact mean over the rows' sd.

    A split pool with no usable dispersion scores NaN: one whose chosen blocks' min equals their
    max is EXACTLY constant, and one whose largest deviation from its mean is within
    ``_NEAR_CONSTANT_ULPS`` ulps of the mean is constant to rounding — the gate the HAC applies to
    a cell's pool (``_pools._near_constant``), read here off the extrema, so it does not depend on
    what the column's OTHER blocks hold. A split of fewer than two rows scores NaN, and so does
    one whose mean left the double range (or met a bar whose sum did) or whose variance is not a
    normal double (``_pools._normal``): ``mean / inf`` is a FINITE 0.0 Sharpe, an infinite mean
    would win every ranking, and a subnormal variance (returns under ~1e-154) holds fewer bits
    than the Sharpe divides by.
    """
    n = sum(col.counts[g] for g in blocks)
    if n < 2:
        return _NAN
    lo, hi = min(col.lo[g] for g in blocks), max(col.hi[g] for g in blocks)
    mean = _fsum([p for g in blocks for p in col.sums[g]]) / (col.scale * n)
    var = _split_variance(col, blocks, n)
    if not (_disperses(lo, hi, mean) and math.isfinite(mean) and _normal(var)):
        return _NAN
    return mean / math.sqrt(var)


def _split_scores(columns: list[list[_Column]], blocks: Sequence[int]) -> np.ndarray:
    """(Q,): each candidate's score over the chosen ``blocks`` — the weakest of its scored
    columns' Sharpes (:func:`_split_score`; NaN where any is): under conjunction a column per
    target (the weakest-target score its caller selects on), under basket ONE column, the
    members' kept rows pooled (:func:`cscv_pbo`)."""
    scores = np.array([[_split_score(c, blocks) for c in cols] for cols in columns])
    # The weakest column's Sharpe, one value per combo — an axis-wise reduction, which
    # numpy's own signature can only type as `Any` (its element-wise overload returns a
    # scalar); the cast states the (Q,) array it is here.
    return cast("np.ndarray", np.min(scores, axis=1))


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
    below zero on test). A split's mean is its rows' exactly rounded sum and its variance exact
    arithmetic's, rounded once (:func:`_split_score`), so a mean has its exact sign and
    level-heavy return columns score instead of cancelling to zero variance. Grid-level
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
    column's block moments and sums are exact, the same in any row order, so two candidates whose
    rows have the same count, sum and sum of squares in every block tie on every split — under
    basket however the returns fall across members.

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
            tr = _split_scores(columns, train)
            if np.all(np.isnan(tr)):
                continue
            bi = int(np.nanargmax(tr))
            te = _split_scores(columns, [g for g in range(s) if g not in train])
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
