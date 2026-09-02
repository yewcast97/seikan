"""Emitted-shape declarations: run-level stamps."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from seikan.types.cells import (
    SummaryCell,
)
from seikan.types.pools import (
    PoolQuantiles,
)
from seikan.types.reliability import (
    PboBlock,
)
from seikan.types.scalars import (
    BenchmarkMode,
    Direction,
    EvidenceBasis,
    ExclusionReason,
    OutcomeKind,
    OutcomeUnits,
    ParamValue,
    TargetMode,
    TargetShape,
)

# ---- run-level stamps ---------------------------------------------------------------------
#
# Combo-independent properties of the run: geometry, provenance, self-description, and the
# panels the three run-level checks read.


class BarSpacing(TypedDict):
    """The joined index's clock geometry — ``dataio.bar_spacing``, stamped as
    ``summary["bar_spacing"]`` and into every ``describe`` profile.

    Ints where the spacing is a whole second, floats otherwise, None below two bars. Pure
    self-description: the engine never interprets cadence.
    """

    min_seconds: int | float | None
    median_seconds: int | float | None
    max_seconds: int | float | None


class OutcomeStamp(TypedDict):
    """What was measured and in which algebra — ``summary["outcome"]``, stamped by
    ``compiler.runner``.

    ALWAYS the explicit dict (the default run stamps ``{target, pct, fraction}``, never a null):
    ``gate._check_evidence_complete`` and ``gate._incommensurable_pool`` both refuse a missing,
    null or partial stamp, because reading an absent stamp as "commensurable" would let stripping
    it bypass the guard outright.
    """

    series: str
    kind: OutcomeKind
    units: OutcomeUnits


class RotationStamp(TypedDict):
    """The rotation null's resolution — ``summary["rotation"]``, stamped by ``compiler.runner``.

    ``p_resolution`` is ``1/(1 + n_shifts)`` (NaN → null when no shift ran — a grid that never
    fired): a ``rot_p`` sitting at that floor means "no shift beat the observation", not
    "p ≈ 0". Evidence only.
    """

    n_shifts: int
    p_resolution: float | None


class SourceAvailability(TypedDict):
    """ONE raw decision leaf's availability within one target —
    ``summary["sources"][target]["by_source"][label]``, built by
    ``compiler.runner.source_coverage`` and read by ``gate._check_source_coverage``.

    ``n_missing`` counts holes STRICTLY AFTER the leaf's first available bar; a leaf that merely
    starts late is warmup, and ``first_available`` reports the late start as evidence. A NULL
    ``first_available`` is the never-available marker: the ledger stays factual (zero post-warmup
    holes, vacuously) and ``gate._check_source_coverage`` REFUSES it (policy v3) — warmup
    requires a start to exist.
    """

    n_missing: int
    first_available: str | None


class SourceCoverage(TypedDict):
    """ONE target's per-source availability ledger — ``summary["sources"][target]``, built
    UNCONDITIONALLY by ``compiler.runner`` (the leaf set is combo-independent) and read by
    ``gate._check_source_coverage``, which refuses any missing bar and re-checks the arithmetic:
    ``n_bars`` equals the summary's, every per-source count sits in ``0..n_bars``, and the union
    ``n_missing`` cannot outnumber the sum of its parts."""

    n_bars: int
    n_missing: int
    by_source: dict[str, SourceAvailability]


class CrossBreadthEntry(TypedDict):
    """One (cross node × combo) effective-universe profile — ``summary["cross_breadth"]``,
    built by ``compiler.runner`` off the node's memoized input frame. Evidence only; read by
    no check.

    ``k`` is the per-bar count of FINITE member inputs — recomputed bit-exactly as the cross
    kernels (``compiler.nb``) compute it before reducing, then summarized instead of
    discarded: member warmup legally thins the cross-section (a late start is warmup, not a
    hole), and this panel is what makes the thinning visible. ``params`` carries the swept
    signal axes only (``{}`` when nothing swept; the horizon has no say in the entry
    condition), so combos that do not move the node's input repeat the same entry — honest
    repetition, the ``signal_coverage`` precedent, never a sum. ``min_valid`` is the effective
    floor the kernel applied (``max(declared, 2)``); a bar below it emitted no cross-section
    at all and is not "evaluated". ``k_min``/``k_median``/``k_max`` cover the evaluated bars
    (null when none); ``n_bars_below_full`` counts evaluated bars with ``k < len(targets)``;
    ``first_full_bar`` is the first bar the whole declared basket was present (null if never)
    — the warmup-tail mirror of ``first_available``. The list is ALWAYS present and ``[]`` on
    a conjunction run (cross nodes are basket-gated by validation).
    """

    node: str
    params: dict[str, ParamValue]
    min_valid: int
    n_bars: int
    n_bars_evaluated: int
    n_bars_below_full: int
    k_min: int | None
    k_median: float | None
    k_max: int | None
    first_full_bar: str | None


class BaselineStats(TypedDict):
    """The statistical fields of ONE unconditional baseline pool —
    ``analysis.stats.baseline_summary``, computed from the eligible forward returns alone.

    An empty pool is all-NaN → null, never zeros — a zero base rate is a measured outcome, and a
    pool with no observations measured nothing. There is deliberately no ``median_ret`` key:
    ``ret_quantiles.p50`` is it.
    """

    n_eligible: int
    mean_ret: float | None
    std_ret: float | None
    hit_rate: float | None
    ret_quantiles: PoolQuantiles
    worst_ret: float | None
    best_ret: float | None


class BaselinePool(BaselineStats):
    """One baseline ROW — the statistics :class:`BaselineStats` computes plus the anchor geometry
    only ``compiler.runner`` can supply, mounted per (horizon × target) and, in basket mode, once
    pooled per horizon.

    The arithmetic pin ``n_eligible + Σexclusions == n_anchor_bars`` is the runner's to satisfy
    and a reader's to re-check: it is the honesty channel that keeps a data hole from silently
    shrinking the base rate's denominator. Evidence only — no check reads it, and there is
    deliberately NO uplift field anywhere: the conditional-vs-base-rate comparison is the
    caller's.
    """

    n_anchor_bars: int
    exclusions: dict[ExclusionReason, int]


class BaselineEntry(TypedDict):
    """One horizon's unconditional baseline — an entry of ``summary["baseline"]``, in horizon
    declaration order, built by ``compiler.runner``.

    ``pooled`` rides ONLY in basket mode (the honest base rate for a pooled conditional claim,
    whose counts are the per-target sums), exactly as ``pooled`` rides only basket cells.
    """

    horizon: int
    by_target: dict[str, BaselinePool]
    pooled: NotRequired[BaselinePool]


class RunSummary(TypedDict):
    """THE summary — the engine's whole report, produced by ``compiler.runner.run_backtest``,
    carried on ``EventStudyResult.summary``, graded by ``gate.evaluate_gate`` and emitted verbatim
    by ``serialize.serialize_result``.

    A GRID, not a headline: it deliberately carries no pooled scalar, no rollup and no
    breakdown table — the grid has no single result, only cells, and every cross-cell aggregate
    (a mean of per-cell means over different horizons) would be a number in no unit that moved
    with grid composition. ``params`` names the swept entry axes (with ``horizon`` only when it
    was swept) and ``targets`` the regime, which ``gate`` reads as the panel-coverage reference
    for every per-target check. Then the run-level stamps: which mechanics produced it
    (``statistics_version``) and over what sample (``gate_evidence_basis``), the geometry and
    extent of the evaluated index, the DECLARED search burden (``n_hypotheses_attempted`` — the
    ONLY multiplicity input in the report, and nothing here is corrected for it), the
    self-description every consumer needs to know which scale it is reading
    (``direction``/``benchmark``/``benchmark_source``/``outcome``/``target_shape``/``target_mode``),
    the rotation resolution, the grid-level CSCV block, the per-source availability panel, the
    cross-sectional breadth panel (``cross_breadth`` — always present, ``[]`` outside basket),
    the unconditional baseline, and finally ``cells`` — one entry per DECLARED combo × horizon,
    in declaration order, non-firing combos included.

    ``index_start``/``index_end`` are None only on an empty index. ``benchmark`` is None on an
    unbenchmarked run and ``benchmark_source`` is None unless a benchmark FILE was loaded.
    """

    params: list[str]
    targets: list[str]
    statistics_version: int
    gate_evidence_basis: EvidenceBasis
    n_bars: int
    index_start: str | None
    index_end: str | None
    bar_spacing: BarSpacing
    n_hypotheses_attempted: int
    direction: Direction
    benchmark: BenchmarkMode | None
    benchmark_source: str | None
    outcome: OutcomeStamp
    target_shape: TargetShape
    target_mode: TargetMode
    rotation: RotationStamp
    pbo: PboBlock
    sources: dict[str, SourceCoverage]
    cross_breadth: list[CrossBreadthEntry]
    baseline: list[BaselineEntry]
    cells: list[SummaryCell]
