"""Emitted-shape declarations: pool-level statistical blocks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypedDict

# ---- pool-level statistical blocks --------------------------------------------------------
#
# The blocks ``analysis.stats`` computes over ONE observation pool. The runner mounts them under
# each cell's per-target panel and (in basket mode) under its pooled panel, so every one of them
# appears twice in a report and has exactly one shape.


class PoolQuantiles(TypedDict):
    """Order statistics of one pool — ``analysis.stats.pool_quantiles``.

    Mounted as ``ret_quantiles`` (and inside the excursion blocks and the baseline rows) by
    ``compiler.runner``; read by no check — evidence only. NaN → null at every point on an empty
    pool ("no evidence", never zero). ``p05`` doubles as the historical VaR(5%) read — the
    quantile whose lower tail ``cvar_5`` averages — and ``p95`` as its favorable mirror; on a
    thin pool the outer points rest on one or two observations, so they ride beside the ``n``
    that scopes them, exactly like every other order statistic here.
    """

    p05: float | None
    p10: float | None
    p25: float | None
    p50: float | None
    p75: float | None
    p90: float | None
    p95: float | None


class MaeQuantiles(PoolQuantiles):
    """The adverse-excursion block: ``pool_quantiles`` over the pool's ``mae`` column plus its own
    ``n``, the subset's ``mean`` and the single worst excursion. Produced by ``compiler.runner``;
    evidence only.

    ``n`` may sit BELOW the panel's ``n``: a hole in the excursion window censors ``mae`` on a row
    whose ``ret`` closed cleanly, and those rows are dropped rather than zero-filled. ``mean``
    covers exactly that same subset — the statistic rides beside the ``n`` that scopes it; both
    are null on an empty subset.
    """

    n: int
    mean: float | None
    worst: float | None


class MfeQuantiles(PoolQuantiles):
    """The favorable mirror of :class:`MaeQuantiles` over the pool's ``mfe`` column — same
    subset-``n`` rule and subset ``mean``, ``best`` instead of ``worst``. Produced by
    ``compiler.runner``; evidence only."""

    n: int
    mean: float | None
    best: float | None


#: Why :func:`~seikan.analysis.stats.benchmark_regression` declined to fit.
type BenchmarkRegressionReason = Literal[
    "no_paired_observations", "insufficient_observations", "no_benchmark_variation"
]


class BenchmarkRegressionBlock(TypedDict):
    """Per-window OLS attribution of the raw leg on the benchmark leg —
    ``analysis.stats.benchmark_regression`` over one pool's paired ``ret_raw``/``ret_bench``
    rows.

    The leg-attribution question the two means alone cannot answer: is the excess mean alpha, or
    beta ≠ 1 riding market drift? ``beta`` is unitless, ``alpha`` is in outcome units per h-bar
    window — NEVER annualized (this engine carries no calendar-return framing) — and
    ``alpha + beta·mean(bench) == mean(raw)`` is an identity a reader re-checks. All three
    statistical fields are genuinely None — not NaN — when the block declined to fit, with
    ``reason`` naming why (``no_paired_observations`` — which covers every unbenchmarked run —
    / ``insufficient_observations`` / ``no_benchmark_variation``); ``r2`` alone is NaN → null
    when the raw leg has zero variance (nothing to explain; beta and alpha are still real).
    Evidence only, read by no check; not a factor model.
    """

    n: int
    beta: float | None
    alpha: float | None
    r2: float | None
    reason: BenchmarkRegressionReason | None


class TimingBlock(TypedDict):
    """Path-timing medians of one pool — ``analysis.stats.timing_summary`` over the RAW
    ``bars_to_positive``/``bars_to_trough`` columns.

    The aggregation that spares a reader ``--trades-out`` for the timing pair, exactly as the
    excursion blocks do for ``mae``/``mfe``. Medians only: both durations are censored at the
    horizon, and a mean of censored durations misleads. Each leg covers its own finite subset
    with the count that scopes it — ``bars_to_positive`` is finite only on rows whose path ever
    touched positive, a survivors-only conditional read, never a recovery probability. NaN →
    null medians on an empty subset. Evidence only.
    """

    n_to_positive: int
    median_bars_to_positive: float | None
    n_to_trough: int
    median_bars_to_trough: float | None


class ConcentrationBlock(TypedDict):
    """|return|-mass share of a pool's largest observations — ``analysis.stats.concentration``.

    Mounted per (cell × target) and, in basket mode, on the pooled panel; ``top_share_abs`` is the
    one number the per-cell ``concentration`` check reads (an empty or zero-mass pool yields
    NaN → null, which that check refuses rather than waves through).
    """

    top_share_abs: float | None
    n_top: int
    top_frac: float


class EpisodeBootstrapCI(TypedDict):
    """Episode-bootstrap percentile CI for a pool's mean — ``analysis.stats.episode_bootstrap_ci``.

    Mounted as ``boot`` on every per-target panel (and the pooled panel) by ``compiler.runner``;
    read by no check. The three interval fields are genuinely ``None`` — not NaN — when the pool
    carries no resampling distribution, and ``reason`` says which refusal applied
    (``no_observations`` / ``insufficient_episodes``).
    """

    method: Literal["episode_percentile"]
    ci_level: float
    n_boot: int
    n_episodes: int
    ci_lo: float | None
    ci_hi: float | None
    boot_se: float | None
    reason: Literal["no_observations", "insufficient_episodes"] | None


class SubperiodCounts(TypedDict):
    """One era's counts alone — ``analysis.stats.subperiod_means``, which is handed a pool's
    returns and entry bars and knows nothing about the calendar those bars sit on.

    THE SEAM: the statistical layer counts, and ``compiler.runner`` stamps the window it counted
    over, mounting these two fields under the timestamps as a :class:`SubperiodEntry`.
    """

    n: int
    mean_ret: float | None


class SubperiodEntry(SubperiodCounts):
    """One era of the run's fixed three-segment split — ``analysis.stats.subperiod_means`` for the
    counts, ``compiler.runner`` for the window timestamps it mounts them under.

    Era visibility, not a train/test split: no purging, and nothing reads it. ``start``/``end`` are
    None only for a degenerate (empty) segment; ``mean_ret`` is None for a segment with no
    observations — no evidence, never zero.
    """

    start: str | None
    end: str | None


class EpisodeStatsBlock(TypedDict):
    """One cell's cross-target episode clustering — ``analysis.stats.episode_stats``.

    Produced over the cell's CLOSED rows and mounted as ``episode_stats``. The checklist reads
    exactly two fields: ``n`` (``cell_evidence`` reconciles it against the per-target total) and
    ``max_cluster_share_abs`` (the one-episode ceiling in ``concentration``); the rest is
    evidence. ``mass_hhi`` (Σ share² over the clusters' |return|-mass shares) and
    ``effective_n_clusters`` (1/hhi ∈ [1, n_clusters]) are the ceiling's SMOOTH companions —
    two episodes at 50% each clear a 0.60 ceiling while reading as exactly two effective
    episodes — and the gate reads neither. An empty or zero-mass pool reports zero counts with
    NaN → null shares, hhi pair included, and a null start.
    """

    n: int
    n_clusters: int
    largest_cluster_n: int
    largest_cluster_share_abs: float | None
    largest_cluster_start: str | None
    max_cluster_share_abs: float | None
    mass_hhi: float | None
    effective_n_clusters: float | None


class EpisodeLedgerEntry(TypedDict):
    """One listed episode of :class:`EpisodeLedgerBlock` — earliest first, never ranked.

    ``start``/``end`` are the cluster's earliest entry and latest exit (a chain merge can end on
    an interior row's window); ``share_abs`` is its |return| mass over the pool total, NaN → null
    on a zero-mass pool. ``mean_ret`` stays plain ``float``: the ledger's rows are built over
    dropna'd returns, so it is finite by construction.
    """

    start: str
    end: str
    n: int
    mean_ret: float
    share_abs: float | None


class EpisodeLedgerBlock(TypedDict):
    """One cell's time-ordered episode ledger — ``analysis.stats.episode_ledger``, mounted as
    ``episodes``. Evidence only; no check reads it.

    Truncation at ``cap`` is explicit and mass-conserving: ``n_omitted`` counts the clusters past
    the cap and ``omitted_share_abs`` carries their combined share (0.0 when nothing was omitted,
    NaN → null when omitted clusters sit over a zero-mass pool). ``n_total`` equals
    ``episode_stats.n_clusters`` on the same rows by construction.
    """

    entries: list[EpisodeLedgerEntry]
    n_total: int
    n_omitted: int
    omitted_share_abs: float | None
    cap: int


class EpisodeProfileBlock(TypedDict):
    """The episode-deduplicated twin of the row-level pool statistics —
    ``analysis.stats.episode_profile``, mounted per cell as ``episode_profile``.

    Overlapping observations from ONE market episode enter the row-level statistics up to ~h
    times (× members in basket): one crisis can be the whole tail of ``mae_quantiles``, most of
    ``hit_rate``'s streak of wins, and ``profit_factor``'s loss mass. This block recomputes the
    same statistic family in EPISODE units — the same frozen cross-target overlap merge as
    ``episode_stats``, one aggregate per episode (the MEAN of member rows' ``ret``, matching the
    ledger's per-episode ``mean_ret``; the extreme ``mae``/``mfe`` within the episode) — so
    row-vs-episode divergence is itself the visible cluster diagnostic, the ``n`` vs
    ``n_nonoverlap``
    doctrine applied to every pool statistic. Reported, never a correction: no row-level number
    is reweighted, and no check reads any field here.

    ``n_episodes == episode_stats.n_clusters`` on the same rows by construction (a reader-
    checkable reconciliation, deliberately not gated). Emits ALWAYS — descriptive statistics do
    not refuse on thinness (quantiles over two episodes rest on two episodes, which
    ``n_episodes`` says out loud); an empty pool is zero counts, NaN → null statistics, and the
    excursion blocks' own subset ``n`` may sit below ``n_episodes`` where a whole episode's path
    columns were censored — while ``edge_ratio`` pairs at the episode level (only episodes
    carrying BOTH legs feed it). The streak pair is the one honest home for consecutive-outcome
    reads
    (row-level streaks over overlapping observations are cluster artifacts and are deliberately
    not emitted): the longest run of consecutive episodes, in time order, whose ``ret_mean`` is
    strictly positive / strictly negative — a zero-mean episode breaks both runs. In basket the
    cross-target merge makes this the pooled episode read for free.
    """

    n_episodes: int
    hit_rate: float | None
    mean_ret: float | None
    profit_factor: float | None
    ret_quantiles: PoolQuantiles
    worst_ret: float | None
    best_ret: float | None
    mae_quantiles: MaeQuantiles
    mfe_quantiles: MfeQuantiles
    edge_ratio: float | None
    max_win_streak: int
    max_loss_streak: int


class BucketRecord(TypedDict):
    """One quantile bucket of a per-cell conditional panel, over the ascending-feature qcut.

    ``bucket`` is the rendered interval label when ONE target's rows formed the panel
    (``analysis.stats._bucket_records``), or the ordinal ``q1..qk`` when per-target edges
    aggregated across targets (``_aggregate_bucket_ordinals`` — interval strings are
    target-relative there and unprintable as one label)."""

    bucket: str
    n: int
    mean_ret: float
    hit_rate: float


class FeatureBucketPanel(TypedDict):
    """One feature's conditional-bucket view within a cell —
    ``analysis.stats.cell_conditional_buckets``, mounted under ``conditional_buckets``.

    Every requested feature gets an entry: a refusal is an empty ``buckets`` list plus an explicit
    ``reason`` (``no_closed_observations`` / ``insufficient_observations`` /
    ``insufficient_distinct_values``), never an absent key. ``reason`` is None when the buckets
    are real. Evidence only.
    """

    buckets: list[BucketRecord]
    reason: (
        Literal[
            "no_closed_observations", "insufficient_observations", "insufficient_distinct_values"
        ]
        | None
    )


class BucketMonotonicity(TypedDict):
    """Spearman rank correlation between bucket ORDER and bucket ``mean_ret`` —
    ``analysis.stats._bucket_monotonicity``, mounted under ``bucket_monotonicity``.

    Carried only for features that bucketed AND showed a rank signal (fewer than 3 populated
    buckets, or constant means, yields no entry at all). Evidence only.
    """

    rho: float
    sign: int


class CellBucketPanels(TypedDict):
    """The TWO panels ``analysis.stats.cell_conditional_buckets`` returns together — the runner
    mounts them onto the cell in one ``cell.update(...)``, which is why they travel as one dict
    keyed by the two names they mount under rather than as two returns.

    ``conditional_buckets`` carries EVERY requested feature (a refusal is an empty bucket list
    plus its reason, never an absent key); ``bucket_monotonicity`` carries only the features that
    bucketed AND showed a rank signal. Both are per-cell and evidence-only.
    """

    conditional_buckets: dict[str, FeatureBucketPanel]
    bucket_monotonicity: dict[str, BucketMonotonicity]


class FeatureAssociation(TypedDict):
    """One (cell × feature × target) Spearman between the entry-time snapshot and the realized
    closed return — ``analysis.stats.feature_outcome_association``, mounted under
    ``feature_association``.

    ``rho`` is genuinely None on a refusal, with ``reason`` naming it
    (``insufficient_observations`` / ``no_rank_variation``) — reporting ``rho = 0`` would claim a
    MEASURED absence of association. No p-value, deliberately. Evidence only.
    """

    rho: float | None
    n: int
    reason: Literal["insufficient_observations", "no_rank_variation"] | None


class MemberShareBlock(TypedDict):
    """The basket's |return|-mass decomposition over its members — built by ``compiler.runner``
    and mounted on the pooled panel.

    ``by_target`` covers EVERY declared member (NaN → null shares on a zero-mass pool) and is
    attribution, never a ranking; the checklist reads only ``max_member_share_abs``, the
    one-name-basket detector in ``concentration``. ``Mapping`` rather than ``dict`` because the
    value type must admit the emitted null while the runner builds a plain ``dict[str, float]``
    (NaN, not None, in memory) — covariance is exactly the seam being described.
    """

    by_target: Mapping[str, float | None]
    max_member_share_abs: float | None


class PoolMoments(TypedDict):
    """The moment and shape reads of ONE observation pool — ``analysis.stats.pool_moments``.

    INTERNAL to the pipeline: ``compiler.runner`` mounts these nine fields into every
    :class:`CellTargetPanel` (and pooled panel), which is where they are emitted, so the fields
    stay plain ``float`` and state the in-memory arithmetic (NaN on an empty or degenerate pool;
    the panel's ``T | None`` is the emitted form). The checklist reads ``n`` and ``mean_ret``;
    everything else is evidence.
    """

    n: int
    mean_ret: float
    std_ret: float
    hit_rate: float
    win_loss_ratio: float
    skewness: float
    kurtosis: float
    tail_ratio: float
    cvar_5: float
