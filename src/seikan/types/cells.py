"""Emitted-shape declarations: the per-cell panel."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from seikan.types.pools import (
    BenchmarkRegressionBlock,
    BucketMonotonicity,
    ConcentrationBlock,
    EpisodeBootstrapCI,
    EpisodeLedgerBlock,
    EpisodeProfileBlock,
    EpisodeStatsBlock,
    FeatureAssociation,
    FeatureBucketPanel,
    MaeQuantiles,
    MemberShareBlock,
    MfeQuantiles,
    PoolQuantiles,
    SubperiodEntry,
    TimingBlock,
)
from seikan.types.scalars import (
    ComboKey,
    ExitReason,
    ParamValue,
)

# ---- the per-cell panel -------------------------------------------------------------------
#
# ``summary["cells"]`` — one entry per DECLARED (param combo × horizon), non-firing ones
# included. Assembled by ``compiler.runner``; graded entry by entry by ``gate``'s five per-cell
# checks; emitted verbatim by ``serialize``/``cli``.


class CellTargetPanel(TypedDict):
    """ONE (cell × target) evidence panel — ``cells[i].by_target[target]``, built by
    ``compiler.runner``.

    A target with no closed rows still gets a full entry (``n = 0``, NaN → null statistics, null
    evidence blocks carrying their ``reason``): an OMITTED target would read as "not applicable
    here" when what happened is "this cell produced no evidence for a target it claims to hold
    across" — which is why every statistical field here admits ``None``.
    ``gate._check_cell_evidence`` reconciles ``n``/``n_nonoverlap`` against the coverage ledger, and
    ``support``/``concentration`` read ``n``/``n_nonoverlap``/``mean_ret`` and ``concentration``
    under
    the CONJUNCTION rubric; everything else here is evidence. ``mean_ret_bench`` is null on every
    unbenchmarked run ("no benchmark leg", never a measured zero). The shape/dispersion reads
    (``win_loss_ratio``/``std_ret``/``skewness``/``kurtosis``/``tail_ratio``/``cvar_5``) are
    ``pool_moments`` over the same closed pool; ``tail_ratio ≡ |p95/p05|`` of ``ret_quantiles``
    and rides as the same derivable-convenience class as ``profit_factor``. ``rot_n_null`` is
    the number of defined shifts this pool's rotation null was formed over — its own p floor is
    ``1/(1 + rot_n_null)``.
    """

    n: int
    n_nonoverlap: int
    mean_ret: float | None
    mean_ret_raw: float | None
    mean_ret_bench: float | None
    benchmark_regression: BenchmarkRegressionBlock
    hit_rate: float | None
    win_loss_ratio: float | None
    profit_factor: float | None
    std_ret: float | None
    skewness: float | None
    kurtosis: float | None
    tail_ratio: float | None
    cvar_5: float | None
    t_hac: float | None
    hac_se: float | None
    rot_p: float | None
    rot_n_null: int
    concentration: ConcentrationBlock
    boot: EpisodeBootstrapCI
    subperiods: list[SubperiodEntry]
    ret_quantiles: PoolQuantiles
    worst_ret: float | None
    best_ret: float | None
    mae_quantiles: MaeQuantiles
    mfe_quantiles: MfeQuantiles
    edge_ratio: float | None
    timing: TimingBlock


class CellPooledPanel(CellTargetPanel):
    """A BASKET cell's one cross-target evidence pool — ``cells[i].pooled``, built by
    ``compiler.runner`` over the concatenated (bar × member) closed rows.

    Mirrors :class:`CellTargetPanel` field for field so there is one mental model, and adds the
    member-mass decomposition. This is the panel the basket rubric GRADES: ``support`` reads
    ``n``/``n_nonoverlap``/``mean_ret`` here instead of per member, ``concentration`` reads
    ``concentration.top_share_abs`` and ``member_share.max_member_share_abs``, and
    ``cell_evidence`` reconciles ``n``/``n_nonoverlap`` against the member panels and the bar clock.
    Absent — not null — on conjunction cells, where the gate REFUSES it as a restamped basket.
    """

    member_share: MemberShareBlock


class CellOutcomeCoverage(TypedDict):
    """One target's censoring ledger over a cell's FULL rows — ``cells[i].outcome_coverage[t]``,
    built by ``compiler.runner``.

    All four exit reasons are reported, zeros included, so ``sum(exit_reasons) == n_attempted``
    and ``n_closed == exit_reasons["horizon"]`` are re-checkable from the report alone — which is
    exactly what ``gate._check_cell_evidence`` re-checks. ``gate._check_cell_outcome_coverage``
    then refuses any ``no_outcome``/``no_benchmark`` firing while allowing ``open`` at any count.
    """

    n_attempted: int
    n_closed: int
    exit_reasons: dict[ExitReason, int]


class CellSignalCoverage(TypedDict):
    """One target's decision ledger — ``cells[i].signal_coverage[t]``, built by
    ``compiler.runner`` from the three-valued ``init & ~defined`` mask.

    ``n_bars`` is the joined index length (pure geometry), so ``n_undefined <= n_bars`` is
    arithmetic a reader re-checks; ``gate._check_cell_signal_coverage`` refuses any undecidable
    decision bar, and ``cell_evidence`` verifies ``n_bars`` against the summary's.
    """

    n_bars: int
    n_undefined: int


class SummaryCell(TypedDict):
    """ONE declared (param combo × horizon) cell — an entry of ``summary["cells"]``.

    Built by ``compiler.runner`` (in declaration order, non-firing cells included, so
    ``len(cells) == n_hypotheses_attempted`` holds by construction), graded by ``gate``'s five
    per-cell checks, and emitted verbatim. ``cell_id`` is a rendered LABEL — identity is
    ``params`` plus the position in the panel — and ``params`` always names the horizon, even
    when it was never swept.

    ``pooled`` is the one mode-gated block: present on basket cells, ABSENT (not null) on
    conjunction cells, where the gate refuses its presence as the signature of a restamped basket.
    """

    cell_id: str
    params: dict[str, ParamValue]
    by_target: dict[str, CellTargetPanel]
    pooled: NotRequired[CellPooledPanel]
    episode_stats: EpisodeStatsBlock
    episodes: EpisodeLedgerBlock
    episode_profile: EpisodeProfileBlock
    outcome_coverage: dict[str, CellOutcomeCoverage]
    signal_coverage: dict[str, CellSignalCoverage]
    conditional_buckets: dict[str, FeatureBucketPanel]
    bucket_monotonicity: dict[str, BucketMonotonicity]
    feature_association: dict[str, dict[str, FeatureAssociation]]


class DeclaredCell(TypedDict):
    """The runner's INTERNAL record of one declared cell, appended in the measurement loop before
    anything about firing is known — ``compiler.runner``'s ``declared_cells``, read only by the
    per-cell panel loop in the same module.

    ``combo_key`` carries the swept axes in ``param_levels`` order (the horizon included when it
    was swept) and indexes the trades partition; ``combo_tuple`` carries the ENTRY-TREE axes only
    and indexes the undefined-decision masks, which are keyed by combo because the horizon has no
    say in decidability.
    """

    params: dict[str, ParamValue]
    combo_key: ComboKey
    combo_tuple: ComboKey
    h: int
