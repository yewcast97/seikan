"""Per-cell conditional buckets, bucket monotonicity, and the per-target feature association."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sps

from seikan.types import (
    BucketMonotonicity,
    BucketRecord,
    CellBucketPanels,
    FeatureAssociation,
    FeatureBucketPanel,
)

_NAN = float("nan")


def _finite_mean(mean: float) -> float:
    """A bucket's mean, NaN where its returns' sum left the double range: the true mean of
    finite returns is finite, the computed ``±inf`` (or ``inf − inf``) is no value — nor even a
    sign, the overflow having picked it — so it is NaN → null, never a number that ranks."""
    return mean if math.isfinite(mean) else _NAN


def _bucket_records(table: pd.DataFrame) -> list[BucketRecord]:
    """JSON-friendly records from a ``conditional_buckets`` table (index = the qcut interval,
    or the bucket ordinal where pandas cannot print it)."""
    return [
        {
            "bucket": str(idx),
            "n": int(row["n"]),
            "mean_ret": _finite_mean(float(row["mean_ret"])),
            "hit_rate": float(row["hit_rate"]),
        }
        for idx, row in table.iterrows()
    ]


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
    themselves, no checklist reads it."""
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


def _qcut_buckets(values: pd.Series, q: int) -> pd.Series | pd.Categorical:
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
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        return pd.qcut(values, q=q, duplicates="drop")
    index, edges = _qcut_index(values, q)
    if edges is not None:
        try:
            with np.errstate(over="ignore", invalid="ignore"):  # the rendering, judged below
                buckets = pd.qcut(values, q=q, duplicates="drop")
            cats = buckets.cat.categories
            breaks = np.append(cats.left.to_numpy()[:1], cats.right.to_numpy())
            if len(cats) == len(edges) - 1 and np.isfinite(breaks).all():
                return buckets
        except ValueError:  # a NaN break: the rendering itself left the range
            pass
    held = np.unique(index[~np.isnan(index)])  # the populated bins, ascending
    ordinal = np.where(np.isnan(index), -1, np.searchsorted(held, index))
    names = [f"q{i + 1}" for i in range(held.size)]
    return pd.Categorical.from_codes(ordinal, categories=names, ordered=True)


def conditional_buckets(trades: pd.DataFrame, feature: str, q: int = 4) -> pd.DataFrame:
    """Group trade returns into quantile buckets of an entry-time feature — ONE pool's qcut.

    Tests whether the signal's edge is conditional on the feature (monotonicity across buckets).
    Per-target edges are the caller's job (:func:`cell_conditional_buckets`); this kernel cuts
    exactly the frame it is handed. The index is the qcut interval, or the populated bucket's
    ordinal ``q1..qk`` where pandas cannot print an edge (:func:`_qcut_buckets`).
    """
    if feature not in trades.columns:
        raise KeyError(f"feature {feature!r} not in trades columns")
    valid = trades[trades[feature].notna() & trades["ret"].notna()]
    if valid.empty:
        return pd.DataFrame(columns=["n", "mean_ret", "hit_rate"])
    buckets = _qcut_buckets(valid[feature], q)
    grouped = valid.groupby(buckets, observed=True)["ret"]
    return pd.DataFrame(
        {
            "n": grouped.size(),
            "mean_ret": grouped.mean(),
            "hit_rate": grouped.apply(lambda s: float((s > 0).mean())),
        }
    )


#: Minimum valid (feature, ret) rows for a cell's bucket panel to attempt a qcut at all. Below it
#: a q=4 split averages under five rows per bucket — bucket means that are noise wearing a
#: pattern — so the panel refuses with a reason instead.
BUCKET_MIN_N = 20


def _aggregate_bucket_ordinals(groups: list[pd.DataFrame], name: str, q: int) -> list[BucketRecord]:
    """Per-target qcut codes aggregated by bucket ORDINAL — the multi-target arm of
    :func:`cell_conditional_buckets`.

    Each participating target's rows are cut on their OWN quantile edges (the kernel's
    ``qcut(q, duplicates="drop")`` rule), codes compressed to populated-bucket ordinals, and
    ordinal ``i``'s record pools the union rows with exact row-level statistics (``n`` summed,
    ``mean_ret``/``hit_rate`` over the pooled returns). Returns ``[]`` — the caller's
    ``insufficient_distinct_values`` refusal — when any participating target forms fewer than 2
    populated buckets, fails the qcut outright, or forms a DIFFERENT populated-bucket count than
    a sibling: no silent partial pools, ever (a panel that quietly dropped a member's rows would
    move with data quality, not with the hypothesis). Records are labelled ``q1..qk`` — per-
    target edges make interval strings target-relative and unprintable as one label."""
    parts: list[tuple[np.ndarray, np.ndarray]] = []
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
        k = int(uniq.size)
        if k < 2:
            return []
        bucket_counts.add(k)
        parts.append((ordinal, group["ret"].to_numpy(dtype=float)))
    if len(bucket_counts) != 1:
        return []
    k = bucket_counts.pop()
    records: list[BucketRecord] = []
    for i in range(k):
        pool = np.concatenate([rets[ordinal == i] for ordinal, rets in parts])
        with np.errstate(over="ignore", invalid="ignore"):  # a sum past the range: NaN below
            mean = float(np.mean(pool)) if pool.size else _NAN
        records.append(
            {
                "bucket": f"q{i + 1}",
                "n": int(pool.size),
                "mean_ret": _finite_mean(mean),
                "hit_rate": float(np.mean(pool > 0)) if pool.size else _NAN,
            }
        )
    return records


def cell_conditional_buckets(
    trades: pd.DataFrame,
    feature_names: list[str],
    q: int = 4,
    min_n: int = BUCKET_MIN_N,
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
    """
    buckets_out: dict[str, FeatureBucketPanel] = {}
    mono_out: dict[str, BucketMonotonicity] = {}
    for name in feature_names:
        if trades.empty or name not in trades.columns or "ret" not in trades.columns:
            n_valid = 0
        else:
            valid = trades[trades[name].notna() & trades["ret"].notna()]
            n_valid = len(valid)
        if n_valid == 0:
            buckets_out[name] = {"buckets": [], "reason": "no_closed_observations"}
            continue
        if n_valid < min_n:
            buckets_out[name] = {"buckets": [], "reason": "insufficient_observations"}
            continue
        groups = (
            [group for _, group in valid.groupby("target", sort=False)]
            if "target" in valid.columns
            else [valid]
        )
        if len(groups) == 1:
            # One participating target: the frozen kernel's records verbatim, interval labels
            # included — bit-identical to the pre-v3 pooled path on every single-target cell
            # whose edges pandas can print.
            try:
                records = _bucket_records(conditional_buckets(groups[0], name, q=q))
            except ValueError:
                records = []  # qcut could not form real bins — same refusal as < 2 buckets
        else:
            records = _aggregate_bucket_ordinals(groups, name, q)
        if len(records) < 2:
            buckets_out[name] = {"buckets": [], "reason": "insufficient_distinct_values"}
            continue
        buckets_out[name] = {"buckets": records, "reason": None}
        # A pool whose returns are all equal carries no rank signal, whatever ulp-scale spread
        # its bucket means' rounding shows (bucket sizes differ, so their means round apart).
        mono = (
            None
            if float(valid["ret"].min()) == float(valid["ret"].max())
            else _bucket_monotonicity(records)
        )
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
