"""Per-cell conditional buckets, bucket monotonicity, and the per-target feature association."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats as sps

from seikan.analysis._pools import exact_mean, scaled_parts
from seikan.types import (
    BucketMemberRecord,
    BucketMonotonicity,
    BucketRecord,
    CellBucketPanels,
    FeatureAssociation,
    FeatureBucketPanel,
)

_NAN = float("nan")


def _record(
    rets: np.ndarray, sums: np.ndarray, scale: int, commensurable: bool
) -> BucketMemberRecord:
    """The ONE record arithmetic of every bucket panel, over one record's rows: ``n`` and
    ``hit_rate`` over the rows' own returns, ``mean_ret`` the exactly rounded sum of what the rows
    add (``sums``, :func:`_bucket_sums`) over ``scale`` times ``n`` — NaN under
    ``commensurable=False`` (several targets' level units add to nothing)."""
    return {
        "n": int(rets.size),
        "mean_ret": exact_mean(sums, scale) if commensurable else _NAN,
        "hit_rate": float(np.mean(rets > 0)),
    }


def _records(
    categories: Sequence[object],
    codes: np.ndarray,
    rets: np.ndarray,
    sums: np.ndarray,
    scale: int,
    commensurable: bool,
) -> list[BucketRecord]:
    """One labelled :func:`_record` per populated category, in order. ``codes`` is each row's
    category index (−1 where no bin holds the row)."""
    records: list[BucketRecord] = []
    for i, label in enumerate(categories):
        in_bucket = codes == i
        if in_bucket.any():
            member = _record(rets[in_bucket], sums[in_bucket], scale, commensurable)
            records.append(
                {
                    "bucket": str(label),
                    "n": member["n"],
                    "mean_ret": member["mean_ret"],
                    "hit_rate": member["hit_rate"],
                }
            )
    return records


def _bucket_monotonicity(records: list[BucketRecord]) -> BucketMonotonicity | None:
    """Spearman rank correlation between bucket ORDER and bucket ``mean_ret`` — is the edge
    monotone in the feature? (alphalens' mean-return-by-quantile, reduced to one number.) The
    records arrive in ascending feature order, so ``rho`` > 0 means "higher feature → higher return"
    and ``sign`` is its sign (for a feature keyed on depth or heat, a strong monotone response is
    the signature).
    ``None`` when fewer than 3 populated buckets or the means carry no rank signal (constant),
    and when a populated bucket's computed mean is not finite — its return sum overflowed
    (``inf``, or ``inf − inf``), though the true mean of finite returns is finite: its place in
    the ranking is unknown, so the rank signal is too (``inf`` ranked it above every finite mean;
    dropping it would rank the rest as if it were empty). Evidence-only — like the buckets
    themselves, no checklist reads it.

    Over several targets it ranks the POOLED means, which weight the targets by the rows each
    puts in an ordinal — equal per-target bin counts do not equalize those proportions — so the
    composition alone can reverse a trend every target shows: tied feature values fill two
    targets' ordinals 50/25/25 and 25/25/50, and two members rising bucket by bucket read
    ``rho`` −1. Each multi-target record's ``by_target`` is that composition read."""
    # NaN in memory, null once emitted: an overflowed mean either way
    means = [_NAN if (m := r.get("mean_ret")) is None else m for r in records if r.get("n", 0) > 0]
    # an overflowed mean → no rank signal; constant means → none either (undefined rho)
    if not all(math.isfinite(m) for m in means):
        return None
    if len(means) < 3 or min(means) == max(means):
        return None
    order = np.arange(len(means), dtype=float)
    rho, _p = sps.spearmanr(order, means)
    if math.isnan(rho):
        return None
    return {"rho": float(rho), "sign": int(np.sign(rho))}


def _qcut_index(values: pd.Series, q: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Each row's qcut bin INDEX (float; NaN where no bin holds the row) over one pool's feature
    values, beside pandas' own bin edges — ``None`` where they overflowed and the bins were read
    off the halved pool.

    pandas interpolates a quantile as ``a + (b − a)·t``, and ``b − a`` overflows between two
    adjacent order statistics ``a < 0 < b`` whose spread passes the double range (a feature
    spanning ±1.7e308): the edge reads ``±inf`` or NaN, and the bins cut on it are no bins
    (1/17/6 where 6/6/6/6 is true). So the qcut runs under errstate and its edges decide:
    finite — every pool whose quantiles stay in range — and the bins are pandas' own, byte-
    identical; otherwise the bins are those of the pool halved, which ARE the pool's: such
    ``a`` and ``b`` lie past 2**970 (~1e292) in magnitude, and so does every value of the pool,
    none lying between them — where the halving, the halves' interpolation (no term nears the
    subnormal range) and every comparison are exact, the bins unbounded-exponent arithmetic
    reads. Those edges print as no interval label (the pool's extreme edge is past ~9e307). A
    non-finite feature is pandas' read, as ever — no proof above holds for it, and none reaches
    here but a hand-built frame's (the loader refuses inf; every kernel is finite-or-NaN)."""
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        codes, edges = pd.qcut(values, q=q, labels=False, retbins=True, duplicates="drop")
        return np.asarray(codes, dtype=float), edges
    with np.errstate(over="ignore", invalid="ignore"):  # an overflowed quantile: judged below
        codes, edges = pd.qcut(values, q=q, labels=False, retbins=True, duplicates="drop")
    if np.isfinite(edges).all():
        return np.asarray(codes, dtype=float), edges
    halved = pd.qcut(values * 0.5, q=q, labels=False, duplicates="drop")  # exact: see above
    return np.asarray(halved, dtype=float), None


def _qcut_buckets(values: pd.Series, q: int) -> pd.Categorical:
    """The kernel's bucket key over one pool's feature values: pandas' interval categorical
    wherever its labels print the bins faithfully, else each populated bin's ordinal ``q1..qk``.

    Membership is the bin INDEX (:func:`_qcut_index`). pandas renders each label by rounding its
    edges through ``x·10**digits``, which leaves the double range for an edge past ~1.8e305 (the
    edge reads ``inf``) or under ~1e-306 in magnitude (NaN — no interval can even be built: a
    qcut ``ValueError``, or no category at all), and the categorical merges the bins whose
    labels collided or drops the rows of an interval it could not build: three true bins
    of 30/15/15 read "(-1.001, inf]" 30 / "(inf, inf]" 30. So the bins are read first, as
    indices, and the labels rendered beside pandas' own edges are kept where they are distinct
    and finite, every ordinary feature (the categorical itself, byte-identical); otherwise —
    and always beside a halved pool's bins — each populated bin takes its ordinal, the multi-
    target arm's label for an interval no string can print. A non-finite feature is pandas'
    read, as ever."""
    arr = values.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        return pd.qcut(arr, q=q, duplicates="drop")
    index, edges = _qcut_index(values, q)
    if edges is not None:
        try:
            with np.errstate(over="ignore", invalid="ignore"):  # the rendering, judged below
                buckets = pd.qcut(arr, q=q, duplicates="drop")
            cats = buckets.categories
            breaks = np.append(cats.left.to_numpy()[:1], cats.right.to_numpy())
            if len(cats) == len(edges) - 1 and np.isfinite(breaks).all():
                return buckets
        except ValueError:  # a NaN break: the rendering itself left the range
            pass
    held = np.unique(index[~np.isnan(index)])  # the populated bins, ascending
    ordinal = np.where(np.isnan(index), -1, np.searchsorted(held, index))
    names = [f"q{i + 1}" for i in range(held.size)]
    return pd.Categorical.from_codes(ordinal, categories=names, ordered=True)


def _bucket_keys(values: pd.Series, q: int) -> tuple[list[object], np.ndarray]:
    """One pool's bucket KEYS (:func:`_qcut_buckets`): the categories, ascending, and each row's
    category index (−1 where no bin holds the row)."""
    buckets = _qcut_buckets(values, q)
    return list(buckets.categories), np.asarray(buckets.codes, dtype=np.int64)


def conditional_buckets(trades: pd.DataFrame, feature: str, q: int = 4) -> pd.DataFrame:
    """Group trade returns into quantile buckets of an entry-time feature — ONE pool's qcut.

    Tests whether the signal's edge is conditional on the feature (monotonicity across buckets).
    Per-target edges are the caller's job (:func:`cell_conditional_buckets`); this kernel cuts
    exactly the frame it is handed. The index is the qcut interval, or the populated bucket's
    ordinal ``q1..qk`` where pandas cannot print an edge (:func:`_qcut_buckets`). ``mean_ret`` is
    the bucket's exactly rounded mean (``_pools.exact_mean``): buckets holding the same returns
    read the same mean in any row order, NaN where the sum leaves the double range.
    """
    if feature not in trades.columns:
        raise KeyError(f"feature {feature!r} not in trades columns")
    valid = trades[trades[feature].notna() & trades["ret"].notna()]
    if valid.empty:
        return pd.DataFrame(columns=["n", "mean_ret", "hit_rate"])
    categories, codes = _bucket_keys(valid[feature], q)
    rets = valid["ret"].to_numpy(dtype=float)
    records = _records(categories, codes, rets, rets, 1, True)
    populated = [c for i, c in enumerate(categories) if (codes == i).any()]  # the records' keys
    index = pd.CategoricalIndex(populated, categories=categories, ordered=True, name=feature)
    columns = ["bucket", "n", "mean_ret", "hit_rate"]
    return pd.DataFrame(records, index=index, columns=columns).drop(columns="bucket")


#: Minimum valid (feature, ret) rows for a cell's bucket panel to attempt a qcut at all. Below it
#: a q=4 split averages under five rows per bucket — bucket means that are noise wearing a
#: pattern — so the panel refuses with a reason instead.
BUCKET_MIN_N = 20


class _BarPool(NamedTuple):
    """A ``cross_mean`` pool's bar numerators over one feature's valid rows, in those rows' order:
    each row's entry ``bar``, its bar's exact numerator ``parts`` (on the bar's first closed
    member, zeros on the others — ``_pools.BarNumerators.at``), ``bar_k`` (the rows every bar
    holds in the cell's WHOLE pool) and the member count ``scale`` the numerators are over."""

    bars: np.ndarray
    parts: np.ndarray
    bar_k: np.ndarray
    scale: int

    def rows(self, order: np.ndarray) -> _BarPool:
        """The same pool over the rows at ``order``, in that order."""
        return self._replace(bars=self.bars[order], parts=self.parts[order])


def _whole_bars(bars: np.ndarray, codes: np.ndarray, bar_k: np.ndarray) -> np.ndarray:
    """The rows (aligned with ``bars`` / ``codes``) of every bar whose rows are ALL here and
    share ONE bucket — ``bar_k[b]`` is the number of rows bar ``b`` holds in the cell's pool.
    Only such a bar's rows enter one record together, where their sum is the bar's exact excess
    sum; no record holds the whole sum of a bar split across buckets, or of one with a row the
    feature left unbucketed, so each of its rows keeps its return."""
    held, inv, n_here = np.unique(bars, return_inverse=True, return_counts=True)
    lo = np.full(held.size, np.iinfo(np.int64).max)
    hi = np.full(held.size, np.iinfo(np.int64).min)
    np.minimum.at(lo, inv, codes)
    np.maximum.at(hi, inv, codes)
    whole: np.ndarray = ((n_here == bar_k[held]) & (lo == hi) & (lo >= 0))[inv]
    return whole


def _bucket_sums(
    rets: np.ndarray, codes: np.ndarray, pool: _BarPool | None
) -> tuple[np.ndarray, int]:
    """What each record's MEAN sums per row, and the scale it is over: the rows' own returns —
    or, where a bar lies whole in a record (:func:`_whole_bars`), every row in numerator units
    over the pool's ``scale``: a whole bar's exact parts (on its first row, zeros on the rest),
    any other row its return times ``scale``, exactly (``_pools.scaled_parts``) — so each mean is
    ONE exactly rounded sum over ``scale`` times its ``n``."""
    if pool is None:
        return rets, 1
    whole = _whole_bars(pool.bars, codes, pool.bar_k)
    if not whole.any():
        return rets, 1
    sums = np.zeros((rets.size, max(pool.parts.shape[1], 2)))
    sums[whole, : pool.parts.shape[1]] = pool.parts[whole]
    sums[~whole, :2] = scaled_parts(rets[~whole], pool.scale)
    return sums, pool.scale


def _aggregate_bucket_ordinals(
    groups: list[pd.DataFrame],
    name: str,
    q: int,
    pool: _BarPool | None,
    *,
    commensurable: bool = True,
) -> list[BucketRecord]:
    """Per-target qcut codes aggregated by bucket ORDINAL — the multi-target arm of
    :func:`cell_conditional_buckets`.

    Each participating target's rows are cut on their OWN quantile edges (the kernel's
    ``qcut(q, duplicates="drop")`` rule), codes compressed to populated-bucket ordinals, and
    ordinal ``i``'s record pools the union rows (:func:`_records`: ``n`` summed, ``hit_rate``
    over the pooled returns, ``mean_ret`` their exactly rounded mean). Returns ``[]`` — the
    caller's ``insufficient_distinct_values`` refusal — when any participating target forms
    fewer than 2 populated buckets, fails the qcut outright, or forms a DIFFERENT
    populated-bucket count than a sibling: no silent partial pools, ever (a panel that quietly
    dropped a member's rows would move with data quality, not with the hypothesis). Records are
    labelled ``q1..qk`` — per-target edges make interval strings target-relative and
    unprintable as one label.

    ``pool`` (a ``cross_mean`` pool's bar numerators over the valid rows, which the groups index
    — see :func:`cell_conditional_buckets`) puts the rows of every bar wholly inside one record
    into that record's ``mean_ret`` through their bar's exact numerator (:func:`_bucket_sums`),
    the exact excess sum their rounded excesses miss by residue. ``commensurable=False`` (a
    ``diff`` outcome: each target in its own level units) leaves every record's ``mean_ret``
    NaN — a sum across the targets' units has none — while ``n`` and ``hit_rate``, a count and a
    share of signs, stand.

    Every record carries its COMPOSITION as ``by_target``: each participating target's own
    :func:`_record` over its rows of the ordinal, off their own returns — one target's rows hold
    no bar whole, so never the pool values — with its mean standing under
    ``commensurable=False`` (a target is its own unit). Every (target, ordinal) holds rows: the
    ordinals are each target's populated bins, the same count for every target."""
    ordinals: list[np.ndarray] = []
    bucket_counts: set[int] = set()
    for group in groups:
        try:
            # The bin indices alone: nothing reads a label on this path, and pandas renders its
            # interval labels by rounding every edge through x·10**precision — which leaves the
            # range on a feature past ~1.8e305, where every edge read "(inf, inf]" and the
            # categorical's codes merged the buckets those colliding labels named, and under
            # ~1e-306 in magnitude, where a NaN break made the qcut raise (the panel refused).
            # A feature spanning more than the double range reads the bins of its halved pool.
            cut, _edges = _qcut_index(group[name], q)
        except ValueError:
            return []
        # A row no bin holds (a constant feature's single edge) is NaN here, -1 as a code.
        codes = np.where(np.isnan(cut), -1.0, cut).astype(int)
        uniq, ordinal = np.unique(codes, return_inverse=True)
        if uniq.size < 2:
            return []
        bucket_counts.add(int(uniq.size))
        ordinals.append(ordinal)
    if len(bucket_counts) != 1:
        return []
    k = bucket_counts.pop()
    # The union rows in target-then-row order: a record's rows are the very sequence the
    # per-target concatenation gave, so a mean over them is the same sum, bit for bit.
    order = np.concatenate([group.index.to_numpy() for group in groups])
    codes = np.concatenate(ordinals)
    rets = np.concatenate([group["ret"].to_numpy(dtype=float) for group in groups])
    sums, scale = _bucket_sums(rets, codes, None if pool is None else pool.rows(order))
    records = _records([f"q{i + 1}" for i in range(k)], codes, rets, sums, scale, commensurable)
    members = [
        (str(group["target"].iloc[0]), group["ret"].to_numpy(dtype=float), ordinal)
        for group, ordinal in zip(groups, ordinals, strict=True)
    ]
    for i, record in enumerate(records):
        record["by_target"] = {
            name: _record(r[o == i], r[o == i], 1, True) for name, r, o in members
        }
    return records


def cell_conditional_buckets(
    trades: pd.DataFrame,
    feature_names: list[str],
    q: int = 4,
    min_n: int = BUCKET_MIN_N,
    *,
    rets_pool: np.ndarray | None = None,
    pool_scale: int = 1,
    commensurable: bool = True,
) -> CellBucketPanels:
    """ONE cell's conditional-bucket view over its CLOSED rows — bucket EDGES per TARGET,
    records aggregated across the cell's targets by bucket ordinal.

    Returns ``{"conditional_buckets": {feature: {"buckets": [...], "reason": ...}},
    "bucket_monotonicity": {feature: {"rho", "sign"}}}`` — keyed by the two panel names it
    mounts, so the runner writes ``cell.update(cell_conditional_buckets(rows_closed, names))``.
    Every requested feature gets an entry under ``conditional_buckets`` (refusals are explicit,
    never absent); ``bucket_monotonicity`` carries only features that bucketed AND showed a rank
    signal (:func:`_bucket_monotonicity`'s ≥ 3-populated-buckets rule).

    Edges are PER TARGET (v3) for the same reason :func:`feature_outcome_association` ranks per
    target: one qcut over raw feature levels pooled across members conflates level differences
    BETWEEN members with variation over time — the top pooled bucket can simply BE the
    high-level member, a Simpson's inversion wearing a conditioning read's clothes. Each
    target's valid (feature, ret) rows are cut on their own ascending quantile edges
    (``duplicates="drop"``, mirroring the frozen :func:`conditional_buckets` kernel) and the
    records aggregate by bucket ORDINAL (:func:`_aggregate_bucket_ordinals`, labels ``q1..qk``).
    With ONE participating target — every single-target cell, and any hand-built pool without a
    ``target`` column — the kernel's records pass through verbatim, interval labels included
    (``q1..qk`` there too where pandas cannot print an edge — :func:`_qcut_buckets`).

    Refusal reasons, mutually exclusive and checked in order: ``no_closed_observations`` — no
    row carries both the feature snapshot and a closed ``ret`` (a dead cell, or a feature column
    entirely NaN/absent over these rows); ``insufficient_observations`` — fewer than ``min_n``
    valid rows POOLED (the ONE support floor — no per-target floor is added: a member too thin
    to match the common bucket count refuses below, and a second sealed constant would be a
    second exam in a frozen layer); ``insufficient_distinct_values`` — a target forms fewer
    than 2 populated buckets, fails the qcut, or forms a different populated-bucket count than
    a sibling (the no-silent-partial-pools contract). Aggregated across the cell's TARGETS,
    never across cells. Evidence-only; overlap-inflated like every trades-pool statistic, and
    "associated in this sample" is the entire claim a bucket pattern supports.

    ``rets_pool`` / ``pool_scale`` (a ``cross_mean`` basket's bar numerators, row-aligned with
    ``trades`` — each bar's exact parts on its first closed member, zeros on the others, over the
    member count: ``_pools.cross_mean_bar_numerators``, the runner's pool values) carry the exact
    bar sums into the bucket MEANS: a bar's rounded per-row excesses miss its exact sum by
    residue, so per feature a bar whose rows ALL land in ONE bucket — bucketed by a basket-wide
    feature (a calendar field, a shared external feed), whose per-target edges agree where the
    members fire on the same bars, or a one-row bar — adds its exact sum to that record's mean
    (:func:`_bucket_sums`, in BOTH arms: with one participating target the only bar the cut can
    hold whole is a one-row bar, the basket's other members' rows sitting unbucketed there, and
    it enters at its exact numerator like any other — read off the rounded row, a one-member
    pool's every bucket mean carried residue the pool's episode means and CSCV split means did
    not). A bar split across buckets, or partly unbucketed, keeps each row's own return (no
    record holds its whole sum); ``n`` and ``hit_rate`` stay on each row's return. On a pool
    whose every bar is full, bucketed by a basket-wide feature, every mean is exactly 0.0 — no
    rank signal, so no ``bucket_monotonicity`` entry, in any declaration order — where the
    rounded rows ranked their residue.

    ``commensurable=False`` (a ``diff`` outcome, each target in its own level units) nulls the
    multi-target arm's ``mean_ret`` (:func:`_aggregate_bucket_ordinals`) — its records pool
    several targets' returns — so no ``bucket_monotonicity`` ranks them; one participating
    target's records keep theirs, as does every ``by_target`` entry.

    The multi-target arm's records carry their composition, ``by_target`` (each participating
    target's own n / mean_ret / hit_rate over its rows of the ordinal): the pooled means weight
    the targets by those rows, so the pooled trend can be the composition's rather than any
    target's (:func:`_bucket_monotonicity`). One participating target's records carry none.
    """
    buckets_out: dict[str, FeatureBucketPanel] = {}
    mono_out: dict[str, BucketMonotonicity] = {}
    # Per bar, the rows it holds in this pool — a record holds a bar whole when all of them land
    # in it — and the pool values row-aligned with `trades`.
    pool_all: np.ndarray | None = None
    bar_k: np.ndarray | None = None
    if rets_pool is not None and not trades.empty:
        pool_all = np.asarray(rets_pool, dtype=float)
        if pool_all.shape[0] != len(trades):
            raise ValueError(
                f"rets_pool must be row-aligned with trades: {pool_all.shape} != ({len(trades)},)"
            )
        bar_k = np.bincount(trades["entry_bar"].to_numpy(dtype=np.int64))
    for name in feature_names:
        if trades.empty or name not in trades.columns or "ret" not in trades.columns:
            n_valid = 0
        else:
            is_valid = (trades[name].notna() & trades["ret"].notna()).to_numpy()
            # Positional labels, so each group's labels ARE its rows' positions in `valid`, the
            # positions its pool values ride by (never a frame column, which a feature of the
            # same name would collide with).
            valid = trades[is_valid].reset_index(drop=True)
            n_valid = len(valid)
        if n_valid == 0:
            buckets_out[name] = {"buckets": [], "reason": "no_closed_observations"}
            continue
        if n_valid < min_n:
            buckets_out[name] = {"buckets": [], "reason": "insufficient_observations"}
            continue
        pool = (
            None
            if pool_all is None or bar_k is None
            else _BarPool(
                valid["entry_bar"].to_numpy(dtype=np.int64), pool_all[is_valid], bar_k, pool_scale
            )
        )
        groups = (
            [group for _, group in valid.groupby("target", sort=False)]
            if "target" in valid.columns
            else [valid]
        )
        if len(groups) == 1:
            # One participating target: the frozen kernel's cut, interval labels included —
            # bit-identical to the pre-v3 pooled path on every single-target cell whose edges
            # pandas can print — over the same record arithmetic as the multi-target arm.
            try:
                categories, codes = _bucket_keys(valid[name], q)
            except ValueError:
                records = []  # qcut could not form real bins — same refusal as < 2 buckets
            else:
                rets = valid["ret"].to_numpy(dtype=float)
                sums, scale = _bucket_sums(rets, codes, pool)
                records = _records(categories, codes, rets, sums, scale, True)
        else:
            records = _aggregate_bucket_ordinals(groups, name, q, pool, commensurable=commensurable)
        if len(records) < 2:
            buckets_out[name] = {"buckets": [], "reason": "insufficient_distinct_values"}
            continue
        buckets_out[name] = {"buckets": records, "reason": None}
        mono = _bucket_monotonicity(records)
        if mono is not None:
            mono_out[name] = mono
    return {"conditional_buckets": buckets_out, "bucket_monotonicity": mono_out}


#: Minimum paired (snapshot, return) observations for a per-(cell × feature × target) Spearman
#: to be reported — rank correlations on single-digit pools are sign noise.
ASSOC_MIN_N = 10


def feature_outcome_association(
    vals: np.ndarray, rets: np.ndarray, min_n: int = ASSOC_MIN_N
) -> FeatureAssociation:
    """Spearman rank correlation between ONE (cell × feature × target) pool's entry-time feature
    snapshots and its realized CLOSED returns — ``{"rho", "n", "reason"}``.

    Per target on purpose, in BOTH target modes: the time axis within one target is the only
    axis on which "higher snapshot → higher outcome" is a rank statement about the SIGNAL — a
    pooled cross-member Spearman would conflate level differences BETWEEN members with variation
    over time, an attribution artifact wearing an association's clothes.

    Pairs drop where either side is non-finite (``n`` counts the pairs actually ranked).
    Refusals are explicit: ``insufficient_observations`` below ``min_n``; ``no_rank_variation``
    when either side is constant — Spearman is undefined without ranks to correlate, and the
    reason is reported rather than ``rho = 0``, which would claim a MEASURED absence of
    association.

    Deliberately NO p-value: on overlapping forward-return pools a Spearman p is
    overlap-inflated — exactly the over-trustable number the doctrine forbids — and unlike
    ``t_hac`` there is no honest event-time correction to offer in its place. ``rho`` + ``n``
    are the whole claim: "associated in this sample", never "predicts". Evidence-only; no check
    reads it.
    """
    v = np.asarray(vals, dtype=float)
    r = np.asarray(rets, dtype=float)
    keep = np.isfinite(v) & np.isfinite(r)
    v, r = v[keep], r[keep]
    n = int(v.size)
    if n < min_n or n == 0:
        return {"rho": None, "n": n, "reason": "insufficient_observations"}
    if float(np.min(v)) == float(np.max(v)) or float(np.min(r)) == float(np.max(r)):
        return {"rho": None, "n": n, "reason": "no_rank_variation"}
    rho, _p = sps.spearmanr(v, r)
    if math.isnan(rho):  # degenerate rank structure scipy flags after the fact — same refusal
        return {"rho": None, "n": n, "reason": "no_rank_variation"}
    return {"rho": float(rho), "n": n, "reason": None}
