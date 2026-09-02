"""CSCV -> PBO: the combinatorially-symmetric-cross-validation probability of backtest overfitting
over the declared grid."""

from __future__ import annotations

import math
from collections.abc import Container
from itertools import combinations
from typing import cast

import numpy as np

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

#: One (combo × member) pool's per-block moment accumulators: the block counts, the block sums of
#: returns and the block sums of squares, each a ``(S,)`` array over the S blocks. Summing them
#: over a split's blocks is what makes the split score a pure arithmetic reduction.
type _MomentTriple = tuple[np.ndarray, np.ndarray, np.ndarray]


def _split_scores(
    moments: np.ndarray, refs: np.ndarray, mode: str, block_idx: np.ndarray
) -> np.ndarray:
    """Per-candidate split score over the chosen blocks — the statistic the declared mode's
    caller selects on: the pooled per-observation Sharpe under basket, the weakest-target
    per-observation Sharpe under conjunction.

    ``moments`` is the (Q, T, 3, S) SHIFTED moment tensor: each (combo, target) column's kept
    returns are shifted by that column's own overall mean (``refs``, (Q, T); 0.0 for absent
    members, whose zero triple then contributes exactly nothing), so the one-pass variance runs
    on centered data — the raw Σx² − n·mean² form cancelled to a silent zero on level-heavy
    series (a diff outcome at level 1e9 scored as pure rounding noise). Means are reconstructed
    exactly: per member ``mean = shifted-mean + ref``; pooled through the exact cross-term
    expansion ``Σ(x − m)² = Σ_g [s2_g + 2(ref_g − m)·s1_g + n_g(ref_g − m)²]``.
    """
    mg = moments[:, :, :, block_idx].sum(axis=3)  # (Q, T, 3) per-member block sums
    n_g, s1_g, s2_g = mg[:, :, 0], mg[:, :, 1], mg[:, :, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        if mode == "basket":
            n = n_g.sum(axis=1)
            mean = (s1_g + n_g * refs).sum(axis=1) / n
            d = refs - mean[:, None]
            ss = (s2_g + 2.0 * d * s1_g + n_g * d * d).sum(axis=1)
            sd = np.sqrt(np.maximum(ss / (n - 1.0), 0.0))
            return np.where((n >= 2) & (sd > 0.0), mean / sd, np.nan)
        mean_shift = s1_g / n_g
        mean = mean_shift + refs
        var = (s2_g - n_g * mean_shift * mean_shift) / (n_g - 1.0)
        sd = np.sqrt(np.maximum(var, 0.0))
        sr = np.where((n_g >= 2) & (sd > 0.0), mean / sd, np.nan)
    # The weakest target's Sharpe, one value per combo — an axis-wise reduction, which
    # numpy's own signature can only type as `Any` (its element-wise overload returns a
    # scalar); the cast states the (Q,) array it is here.
    return cast("np.ndarray", np.min(sr, axis=1))


def _degradation_slope(
    tr_a: np.ndarray, te_a: np.ndarray
) -> tuple[float | None, DegradationSlopeReason | None]:
    """Centered OLS slope of the splits' (train-winner, its test score) pairs, or
    ``(None, "degenerate_train_scores")`` when the train scores carry no usable spread.

    The degeneracy test is SCALE-RELATIVE (10 ulps of the scores' magnitude), not exact-zero:
    a grid of near-clones passes an exact-zero test into a rank-deficient fit whose min-norm
    slope is an arbitrary number (numpy's polyfit additionally warned "poorly conditioned" —
    ~0.5 on one probe). The slope of an undefined regression is null with its reason said out
    loud, never a number. The well-conditioned path is the plain centered OLS identity
    cov(tr, te)/var(tr) — no Vandermonde, no RankWarning channel at all."""
    scale = max(1.0, float(np.max(np.abs(tr_a))))
    if float(np.std(tr_a)) <= 10.0 * float(np.finfo(np.float64).eps) * scale:
        return None, "degenerate_train_scores"
    tr_c = tr_a - tr_a.mean()
    te_c = te_a - te_a.mean()
    return float((tr_c @ te_c) / (tr_c @ tr_c)), None


def cscv_pbo(
    cells: list[ReliabilityCell],
    length: int,
    targets: list[str],
    *,
    n_blocks: int = 8,
    off: int = 1,
    mode: str = "conjunction",
    n_combos_declared: int,
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
    below zero on test). Split scores run on SHIFTED moments (:func:`_split_scores`), so
    level-heavy return columns score instead of cancelling to zero variance. Grid-level
    evidence — no cell's grade reads it.

    The population is ledgered, never assumed: ``n_combos <= n_combos_scoreable <=
    n_combos_declared`` — the DISTINCT candidates scored, the combos admissible with at least
    one fired-and-closed pool (pre-collapse), and the caller's declared combo × horizon grid
    (the runner passes ``n_hypotheses_attempted``). CSCV never sees a declared combo that
    produced no closed pool, so the ledger is what says how far short of the declared grid the
    scored population fell. Byte-identical candidate combos (same per-target masks, forward
    returns and horizons — a sweep axis that never moved the mask) collapse to ONE
    declaration-order representative BEFORE scoring: under selection-rank arithmetic M clones
    would hand the train winner the worst test rank purely for tying with itself. Test ranks
    are MIDRANKS (ties share their average rank; a full tie is λ = 0 — a split that cannot
    distinguish candidates is not evidence of overfitting), and the training winner's tie-break
    is first-index / declaration order (``np.nanargmax``), deterministic by construction.

    ``mode`` is the thesis's declared ``target_mode`` and sets the SPLIT SCORE — the score
    mirrors the statistic the caller of that mode selects on, because PBO describes the
    CALLER'S selection (the engine has none). ``"conjunction"`` (default): a combo is
    admissible only when EVERY regime target has a pool, and its score is its WEAKEST target's
    per-observation Sharpe — the weakest-target rule the conjunction implies.
    ``"basket"``: the members form ONE pool, so a combo is
    admissible with ≥ 1 member present — an absent member contributes zero moments, exactly a
    member that never fired — and its score is the POOLED per-observation Sharpe over the
    concatenated member observations. No min: a basket caller selects on the pooled read.
    """
    if mode not in ("conjunction", "basket"):
        raise ValueError(f"unknown target_mode {mode!r}")

    empty: PboBlock = {
        "pbo": None,
        "reason": "insufficient_data",
        "n_splits": 0,
        "n_splits_attempted": 0,
        "n_candidates_min": 0,
        "n_combos": 0,
        "n_combos_scoreable": 0,
        "n_combos_declared": int(n_combos_declared),
        "blocks": int(n_blocks),
        "lambda_mean": _NAN,
        "oos_degradation_slope": _NAN,
        "oos_degradation_slope_reason": None,
        "prob_oos_loss": _NAN,
    }
    if not cells or length < 4:
        return empty
    targets = list(targets)  # one local copy; iterated many times below

    # Combo admissibility mirrors the mode's evidence rule: conjunction demands every regime
    # target's pool (a combo missing one has no weakest member to score), basket demands ≥ 1
    # member (the pooled pool exists; absent members are zero contribution, not missing evidence).
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
    if n_distinct < 2:
        return {
            **empty,
            "reason": "single_combo",
            "n_combos": n_distinct,
            "n_combos_scoreable": n_combos_scoreable,
        }

    # Adaptive block counts: prefer the caller's S, then the coarser fallback partitions for
    # sparse pools; dict.fromkeys dedupes while keeping the preference order.
    block_candidates = [
        s
        for s in dict.fromkeys((n_blocks, *_PBO_BLOCK_FALLBACKS))
        if s >= 2 and s % 2 == 0 and length >= 2 * s
    ]

    last_empty = empty
    for s in block_candidates:
        edges = np.linspace(0, length, s + 1).astype(int)
        stats_by_key: dict[CellKey, _MomentTriple] = {}
        refs_by_key: dict[CellKey, float] = {}
        for cell in kept_cells:
            key = tuple(cell["key"])
            mask = np.asarray(cell["mask_col"]) > 0
            fwd = np.asarray(cell["fwd_col"], dtype=float)
            h = int(cell["h"])
            t = np.flatnonzero(mask & np.isfinite(fwd))
            b = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, s - 1)
            keep = (t + off + h) < edges[b + 1]
            t, b = t[keep], b[keep]
            r = fwd[t]
            # SHIFTED moments: bincount the column's returns about its own overall kept mean, and
            # carry that reference beside the triple — `_split_scores` reconstructs exact means
            # while the variance arithmetic runs centered (level-safe). Per-S by necessity: the
            # purge above depends on this partition's edges.
            ref = float(r.mean()) if r.size else 0.0
            r_shift = r - ref
            stats_by_key[key] = (
                np.bincount(b, minlength=s).astype(float),
                np.bincount(b, weights=r_shift, minlength=s),
                np.bincount(b, weights=r_shift * r_shift, minlength=s),
            )
            refs_by_key[key] = ref

        # Every surviving cell belongs to an admissible, distinct candidate (filtered and
        # collapsed above, S-independently), so the roster here is exactly that candidate set.
        combos: dict[CellKey, dict[ParamValue | str, _MomentTriple]] = {}
        for key in stats_by_key:
            combos.setdefault(key[:-1], {})[key[-1]] = stats_by_key[key]
        combo_keys = list(combos)
        n_combos = len(combo_keys)

        # Absent members (basket admissibility only — conjunction admits full combos alone)
        # zero-fill: zero count / zero sum / zero sum-of-squares is EXACTLY the moment triple of
        # a member that never fired, so the pooled arithmetic never special-cases absence.
        zero_fill = np.zeros((3, s))
        moments = np.array(
            [
                [np.stack(combos[c][tg]) if tg in combos[c] else zero_fill for tg in targets]
                for c in combo_keys
            ]
        )  # (Q, T, 3, S)
        refs = np.array(
            [[refs_by_key.get((*c, tg), 0.0) for tg in targets] for c in combo_keys]
        )  # (Q, T) — 0.0 for absent members, whose zero triple contributes exactly nothing

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
            tr = _split_scores(moments, refs, mode, train_idx)
            if np.all(np.isnan(tr)):
                continue
            bi = int(np.nanargmax(tr))
            te = _split_scores(moments, refs, mode, test_idx)
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
            last_empty = {
                **empty,
                "n_splits": len(lambdas),
                "n_splits_attempted": n_splits_attempted,
                "n_candidates_min": n_candidates_min if lambdas else 0,
                "n_combos": n_combos,
                "n_combos_scoreable": n_combos_scoreable,
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
