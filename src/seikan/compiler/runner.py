"""Run a thesis through the observer-native forward-return event study → ``EventStudyResult``.

For each point of the entry parameter grid (``vectorize.iter_param_assignments``) we build the
firing mask, then for each measurement ``horizon`` h record an OVERLAPPING forward-return
observation at
every firing bar: there is no exit condition and no one-position-at-a-time state machine. An
observation's return is the forward return from the fill bar to h bars later (sign set by
``direction``) — raw by default, or an EXCESS return when ``params.benchmark`` subtracts a
benchmark series' ("market") or the basket's own mean ("cross_mean", basket mode) same-window
return; observations whose horizon runs past the data end are right-censored (flagged
``is_open``/``exit_reason="open"`` for charting, excluded from the statistics; a
benchmark-missing window censors as ``"no_benchmark"``).

The runner is a PER-HYPOTHESIS REPORTER. Every DECLARED (param combo × horizon) cell is measured
once over the whole index and emitted in ``summary["cells"]`` — one entry per declared cell, in
declaration order, INCLUDING combos and horizons that never fired (an explicit zero/NaN record).
Nothing here selects, ranks or crowns a cell: no best combo, no binding target, no headline
scalar, and no statistic corrected for the size of the grid. ``thesis.target_mode`` declares what
the targets ARE: under "conjunction" they are the thesis's REGIME (a conjunction a cell must hold
across, never a search axis), reported target by target so the weakest one speaks for itself;
under "basket" they form ONE cross-section, each cell additionally carries a POOLED cross-target
block, and the per-target panel becomes attribution evidence.

Each cell carries its own nominal evidence — per-target descriptives plus the overlap-aware
``rot_p``/``t_hac``/``hac_se``/``n_nonoverlap``, its cross-target episode-cluster panel, its
|return|-mass concentration, its censoring ledger (``outcome_coverage``, over the cell's FULL rows
so an unmeasured firing stays visible) and its decision ledger (``signal_coverage``, the
undecidable-bar count that makes a firing suppressed by a missing input visible). Run-level and
combo-independent: the geometry stamps (``n_bars``, ``index_start``/``index_end``), the DECLARED
grid size ``n_hypotheses_attempted`` — the multiplicity the CALLING agent prices its own selection
against — the per-source availability panel, and the grid-level CSCV ``pbo``. Every statistic is
computed over the CLOSED subset only; the full frame (censored rows included) rides on the result
for charting and for the coverage ledgers.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from typing import Literal, NamedTuple, NotRequired, TypedDict

import numpy as np
import pandas as pd

from seikan.analysis.result import EventStudyResult
from seikan.analysis.stats import (
    STATISTICS_VERSION,
    baseline_summary,
    benchmark_regression,
    cell_conditional_buckets,
    concentration,
    cscv_pbo,
    edge_ratio,
    episode_bootstrap_ci,
    episode_ledger,
    episode_profile,
    episode_stats,
    feature_outcome_association,
    mae_block,
    mfe_block,
    pool_moments,
    pool_quantiles,
    pooled_reliability_summary,
    profit_factor,
    reliability_summary,
    subperiod_edges,
    subperiod_means,
    timing_summary,
)
from seikan.compiler import vectorize
from seikan.compiler.data import MarketData
from seikan.compiler.paths import (
    bars_to_positive_full,
    bars_to_trough_full,
    excursion_extremum,
    shift_down,
    shift_rows,
)
from seikan.compiler.sources import source_availability, source_coverage
from seikan.constants import DEFAULT_FEATURE_NAMES, TRADE_COLUMNS
from seikan.dataio import bar_spacing
from seikan.dsl.schema import (
    Change,
    Field,
    RollingAgg,
    Series,
    Thesis,
    fmt_num,
    iter_condition_series,
    iter_cross_series,
    render_series,
    series_cross_nodes,
)
from seikan.types import (
    EXCLUSION_REASONS,
    EXIT_REASONS,
    BaselineEntry,
    BaselinePool,
    CellKey,
    CellOutcomeCoverage,
    CellPooledPanel,
    CellSignalCoverage,
    CellTargetPanel,
    ComboKey,
    CrossBreadthEntry,
    DeclaredCell,
    ExclusionReason,
    FeatureAssociation,
    OutcomeKind,
    OutcomeUnits,
    ParamValue,
    ReliabilityCell,
    ReliabilityRead,
    ReliabilitySummary,
    RunSummary,
    SummaryCell,
)

_NAN = float("nan")

#: The human-readable denomination each outcome algebra measures in, stamped beside the algebra
#: itself so no consumer re-derives what a ``mean_ret`` of 0.03 IS (3% vs 0.03 log units vs 0.03
#: index points) from out-of-band knowledge.
_OUTCOME_UNITS: dict[OutcomeKind, OutcomeUnits] = {
    "pct": "fraction",
    "log": "log",
    "diff": "level_diff",
}


# Default entry-time feature snapshots (grouping variables for conditional analysis), price-only:
# short/medium momentum plus realized log-return vol — the most common "edge is actually a regime
# beta" confound. Overridden by `thesis.params.features`.
_FEATURE_NODES: dict[str, Series] = {
    "ret_5": Change(input=Field(column="close"), periods=5),
    "ret_20": Change(input=Field(column="close"), periods=20),
    "vol_14": RollingAgg(
        input=Change(input=Field(column="close"), kind="log"),
        window=14,
        agg="std",
    ),
}
# dsl.schema validates that no sweep axis shadows a default feature name against this same tuple —
# checked with a real raise, not `assert`, so the invariant survives `python -O`.
if tuple(_FEATURE_NODES) != DEFAULT_FEATURE_NAMES:  # pragma: no cover - import-time invariant
    raise RuntimeError(
        f"_FEATURE_NODES keys {tuple(_FEATURE_NODES)} drifted from "
        f"constants.DEFAULT_FEATURE_NAMES {DEFAULT_FEATURE_NAMES}"
    )
#: The columns this module writes into the trades frame. Declared in :mod:`seikan.constants` so
#: ``dsl.schema`` can reject a colliding ``params.features`` key at VALIDATION time — long
#: before a runner exists to be corrupted by it.
_TRADE_COLUMNS = list(TRADE_COLUMNS)


def _finite_column(df: pd.DataFrame, name: str) -> np.ndarray:
    """The FINITE values of one column of a trades sub-frame, as a float array.

    The path columns (``mae`` / ``mfe``) are NaN on rows whose excursion window held a hole even
    though the row's ``ret`` closed, so their pools are a subset of the cell's ``n`` and the
    non-finite entries must be dropped before any order statistic is taken — not counted, not
    zero-filled.
    """
    if df.empty:
        return np.empty(0)
    col: np.ndarray = df[name].to_numpy(dtype=float)
    finite: np.ndarray = col[np.isfinite(col)]
    return finite


class _HorizonFrames(NamedTuple):
    """The combo-independent per-horizon frames — everything the measurement loop and the cells
    panel index at ``[bar, target]``: the (excess) forward return and its outcome-leg
    finiteness, the attribution legs, the pre-entry drift, the two excursion extrema, and the
    two path-timing frames."""

    fwd: np.ndarray
    tgt_finite: np.ndarray
    raw: np.ndarray
    bench: np.ndarray | None
    bwd: np.ndarray
    mae_ext: np.ndarray
    mfe_ext: np.ndarray
    bars_to_positive: np.ndarray
    bars_to_trough: np.ndarray


#: The reliability read of a cell with no scored pool — every field at its "no evidence" value.
#: A module-level constant (never mutated) so the panel assembly reads `.get(key,
#: _MISSING_RELIABILITY)` with a TOTAL type instead of a union-with-empty-dict and four
#: per-field defaults.
_MISSING_RELIABILITY: ReliabilityRead = {
    "rot_p": _NAN,
    "rot_n_null": 0,
    "t_hac": _NAN,
    "hac_se": _NAN,
    "n_nonoverlap": 0,
}


def _aligned_column(df: pd.DataFrame, name: str) -> np.ndarray:
    """One column ROW-ALIGNED and unfiltered (empty when the frame is empty or lacks it) — the
    defensive read for the PAIRED consumers (``edge_ratio``, ``benchmark_regression``), which
    filter pairs themselves: feeding them :func:`_finite_column`'s per-leg finite subsets would
    silently mis-pair rows under asymmetric window holes. Both legs of a pair convert through
    this together, so the empty-frame arm can never mis-align one leg against the other."""
    if name not in df.columns or df.empty:
        return np.empty(0)
    col: np.ndarray = df[name].to_numpy(dtype=float)
    return col


def _norm_key(key: object) -> CellKey:
    """A pandas groupby key → a plain-python tuple, so it compares and hashes identically to the
    parameter values the declared grid carries.

    ``groupby`` hands back numpy scalars (and a bare scalar rather than a 1-tuple for a single
    grouping column); the declared cells hold the DSL's own python values. Normalizing both sides
    through one function is what lets a cell find its rows by exact dictionary lookup instead of a
    per-cell rescan of the frame."""
    values = key if isinstance(key, tuple) else (key,)
    return tuple(v.item() if hasattr(v, "item") else v for v in values)


def _baseline_panel(
    frames_by_h: dict[int, _HorizonFrames],
    horizon_values: list[int],
    n_bars: int,
    off: int,
    targets: list[str],
    bench_mode: str | None,
    custom_outcome: bool,
    basket: bool,
) -> list[BaselineEntry]:
    """The run-level unconditional base rate per horizon × target (see the summary comment at
    the call site): the same measurement with the entry condition removed, exclusions pinned by
    ``n_eligible + Σexclusions == n_anchor_bars``, and a pooled row per horizon in basket."""
    anchors = np.arange(max(n_bars - 1, 0))
    baseline_panel: list[BaselineEntry] = []
    for h in horizon_values:
        fwd_h = frames_by_h[h].fwd
        tf_h = frames_by_h[h].tgt_finite
        past_end = anchors + off + h > n_bars - 1
        by_target_base: dict[str, BaselinePool] = {}
        pooled_rets: list[np.ndarray] = []
        pooled_excl: dict[ExclusionReason, int] = dict.fromkeys(EXCLUSION_REASONS, 0)
        for g, tgt in enumerate(targets):
            col = fwd_h[anchors, g]
            eligible = np.isfinite(col)
            miss = ~eligible
            # The SAME discrimination the trades classifier applies: an adjusted-leg hole is
            # "no_benchmark" only where the outcome leg alone was measurable; past-end geometry
            # is "open"; an in-bounds hole in the measured leg is "no_outcome".
            bench_missing = (
                (miss & tf_h[anchors, g]) if (bench_mode or custom_outcome) else np.zeros_like(miss)
            )
            excl: dict[ExclusionReason, int] = {
                "open": int((miss & ~bench_missing & past_end).sum()),
                "no_outcome": int((miss & ~bench_missing & ~past_end).sum()),
                "no_benchmark": int(bench_missing.sum()),
            }
            rets_b = col[eligible]
            bs = baseline_summary(rets_b)
            # The anchor geometry first, then the statistical fields the return array carries —
            # `n_eligible` named ahead of the spread that re-sets it to the same value, so the row
            # reads counts-then-statistics in the order the arithmetic pin is checked in.
            by_target_base[tgt] = {
                "n_anchor_bars": int(anchors.size),
                "n_eligible": bs["n_eligible"],
                "exclusions": excl,
                **bs,
            }
            if basket:
                pooled_rets.append(rets_b)
                pooled_excl = {k: v + excl[k] for k, v in pooled_excl.items()}
        entry_h: BaselineEntry = {"horizon": h, "by_target": by_target_base}
        if basket:
            bs_p = baseline_summary(np.concatenate(pooled_rets) if pooled_rets else np.empty(0))
            entry_h["pooled"] = {
                "n_anchor_bars": int(anchors.size) * len(targets),
                "n_eligible": bs_p["n_eligible"],
                "exclusions": pooled_excl,
                **bs_p,
            }
        baseline_panel.append(entry_h)
    return baseline_panel


class _RunConfig(NamedTuple):
    """Everything ``run_backtest`` resolves from (thesis, md) before any measurement: the
    directional sign, the declared mode, the anchor offset, the outcome read, the measured base
    frame and its geometry, the feature snapshots, the horizon axis, the sweep levels, and the
    benchmark column. Resolution also holds the four library-boundary guards (basket+diff,
    cross-nodes-outside-basket, market+diff, cross_mean-outside-basket) — each unreachable
    through ``Thesis.model_validate`` and re-refused here so a ``model_construct`` forgery
    cannot reach the arithmetic."""

    sign: float
    basket: bool
    off: int
    outcome_series: str
    outcome_kind: Literal["pct", "log", "diff"]
    custom_outcome: bool
    base_np: np.ndarray
    n_bars: int
    targets: list[str]
    index: pd.DatetimeIndex
    features: dict[str, np.ndarray]
    feature_names: list[str]
    horizon_values: list[int]
    param_levels: list[str]
    sweep_levels: list[str]
    bench_mode: Literal["market", "cross_mean"] | None
    bench_col: np.ndarray | None


def _resolve_run_config(thesis: Thesis, md: MarketData) -> _RunConfig:
    sign = -1.0 if thesis.params.direction == "shortonly" else 1.0
    # The declared target semantics (dsl.schema.Thesis.target_mode). Basket ADDS the pooled
    # cross-target reads without changing any per-target number; conjunction emits none of them.
    basket = thesis.target_mode == "basket"
    # A firing bar t fills at the NEXT bar's open — the only tradable anchor (a same-bar
    # close[t]→close[t+h] measurement would read from a price the decision itself consumed).
    off = 1
    # The measured frame (params.outcome). Default = the target's open frame (for a series target
    # the synthesized open IS the value column), measured as a pct return. An outcome feed measures
    # that feed's asof values at the same next-bar anchor; `kind` picks the measurement algebra
    # (pct / log / diff — diff for rates/spreads where a percent of a near-zero level is
    # meaningless).
    outcome_series = thesis.params.outcome.series
    outcome_kind = thesis.params.outcome.kind
    custom_outcome = outcome_series != "target"
    if basket and outcome_kind == "diff":
        # Unreachable through `Thesis.model_validate` (the DSL refuses this pairing); this closes
        # the library boundary, where a hand-built `model_construct` thesis would otherwise pool
        # level-unit returns across members the engine cannot certify commensurable.
        raise ValueError(
            "target_mode='basket' is incommensurable with outcome kind 'diff' (pooled "
            "cross-member returns need a common unit); use outcome kind 'pct'/'log'"
        )
    cross_kinds = sorted(
        {
            n.type
            for series in (
                *iter_condition_series(thesis.entry),
                *(thesis.params.features or {}).values(),
            )
            for n in series_cross_nodes(series)
        }
    )
    if cross_kinds and not basket:
        # Unreachable through `Thesis.model_validate`; closes the library boundary like the
        # guards above — the transforms-layer >= 2-columns check cannot see the MODE, so a
        # model_construct thesis could otherwise rank across a conjunction's targets.
        raise ValueError(f"cross-sectional node(s) {cross_kinds} require target_mode='basket'")
    if custom_outcome:
        base_np = md.external_values(outcome_series)  # (rows × targets) asof feed values
    else:
        base_np = md.open.to_numpy(dtype=float)  # fill-price frame
    n_bars = base_np.shape[0]
    targets = md.targets
    index = md.index

    # The evaluation memo rides the ``MarketData`` memos (see ``vectorize.build_series``): identical
    # sub-series
    # across sweep combos (e.g. transforms a constant-threshold sweep leaves untouched) are built
    # once, and ``api.list_entries`` over the same ``md`` rebuilds nothing this backtest built.

    # Entry-time feature snapshots — any scalar-param Series, taken at the firing (decision) bar.
    # `is not None`, never truthiness: an explicitly-empty features dict is refused at DSL
    # validation, so the ONLY spelling of "use the defaults" is the omitted field.
    feature_nodes = thesis.params.features if thesis.params.features is not None else _FEATURE_NODES
    features: dict[str, np.ndarray] = {
        name: vectorize.build_series(node, md)[0].to_numpy(dtype=float)
        for name, node in feature_nodes.items()
    }
    feature_names = list(feature_nodes)

    # Horizon is the forward measurement window. A scalar runs one window; a list sweeps it as its
    # own "horizon" result axis (like the transform-window sweeps) — a return response curve.
    hz = thesis.params.horizon
    swept_horizon = isinstance(hz, list)
    horizon_values: list[int] = list(hz) if isinstance(hz, list) else [hz]

    param_levels = [lvl for lvl, _ in vectorize.collect_sweeps(thesis.entry)]
    # The entry-tree sweep axes alone — a combo dict's exact key set, BEFORE the horizon axis is
    # appended below. The canonical per-combo lookup key everywhere a combo indexes a dict.
    sweep_levels = list(param_levels)
    if swept_horizon:
        param_levels.append("horizon")
    # A swept-constant 'name' colliding with the columns this module writes (target / trade fields /
    # feature snapshots) would silently overwrite one of them. collect_sweeps already rejects
    # constant↔constant and reserved-level (target/horizon) collisions; this covers the column
    # namespace it can't see (features live in params).
    reserved_cols = {"target", *_TRADE_COLUMNS, *feature_names}
    collisions = reserved_cols.intersection(param_levels)
    if collisions:
        raise ValueError(
            f"sweep axis name(s) {sorted(collisions)} collide with reserved trade/feature columns; "
            f"rename the swept constant's 'name'"
        )

    # Benchmark adjustment (params.benchmark). "market" subtracts the `benchmark` key's series same-
    # window return; "cross_mean" (basket mode) subtracts the basket's own same-window mean.
    # Applied INSIDE _forward, before anything is recorded, so every downstream statistic
    # (rotation null, HAC, buckets, PBO) automatically describes EXCESS returns. None keeps the
    # raw path.
    bench_mode = thesis.params.benchmark
    bench_col: np.ndarray | None = None
    if bench_mode == "market":
        if outcome_kind == "diff":
            # Unreachable through `Thesis.model_validate` (the DSL refuses this pairing); this
            # closes the library boundary, where a hand-built `model_construct` thesis would
            # otherwise reach the excess arithmetic and subtract a return from a level.
            raise ValueError(
                "params.benchmark='market' is incommensurable with outcome kind 'diff' "
                "(level units minus a benchmark return); use outcome kind 'pct'/'log'"
            )
        if md.benchmark_open is None:
            raise ValueError(
                "params.benchmark='market' but MarketData has no benchmark_open "
                "(load via DataSpec.benchmark)"
            )
        bench_col = md.benchmark_open.to_numpy(dtype=float).reshape(-1, 1)
    elif bench_mode == "cross_mean":
        if not basket or len(targets) < 2:
            # Unreachable through `Thesis.model_validate` (cross_mean requires
            # target_mode='basket', which requires >= 2 targets); closes the library boundary
            # exactly like the market+diff guard above.
            raise ValueError(
                "params.benchmark='cross_mean' requires target_mode='basket' with >= 2 targets"
            )

    return _RunConfig(
        sign=sign,
        basket=basket,
        off=off,
        outcome_series=outcome_series,
        outcome_kind=outcome_kind,
        custom_outcome=custom_outcome,
        base_np=base_np,
        n_bars=n_bars,
        targets=targets,
        index=index,
        features=features,
        feature_names=feature_names,
        horizon_values=horizon_values,
        param_levels=param_levels,
        sweep_levels=sweep_levels,
        bench_mode=bench_mode,
        bench_col=bench_col,
    )


def _run_summary(
    thesis: Thesis,
    md: MarketData,
    cfg: _RunConfig,
    frames_by_h: dict[int, _HorizonFrames],
    rel: ReliabilitySummary,
    rel_cells: list[ReliabilityCell],
    cross_breadth: list[CrossBreadthEntry],
    n_combos_attempted: int,
    cells: list[SummaryCell],
) -> RunSummary:
    """THE summary, built as ONE literal IN EMISSION ORDER (dict insertion is the emitted key
    order): the two grid labels, versions and basis, index geometry, the declared search burden,
    the self-description stamps, rotation resolution, the grid-level CSCV block, the per-source
    availability panel, the cross-sectional breadth ledger, the unconditional baseline, and the
    per-cell panel. The section comments carry the doctrine, stamp by stamp."""
    # Rotation-resolution transparency (evidence-only): the shift count the rotation null used —
    # every non-identity shift of the series — and the implied smallest achievable
    # p = 1/(1+n_shifts) over a fully defined null. A cell's own floor is 1/(1+rot_n_null),
    # which a sparse mask lifts above this; either way a `rot_p` at its floor means "no shift
    # beat the observation", not "p ≈ 0".
    n_shifts = int(rel.get("n_shifts", 0))
    # Per-SOURCE availability, computed UNCONDITIONALLY. Every raw decision leaf
    # the entry tree reads (`Field`/`External`/`DaysSince`), counted over the WHOLE evaluated
    # interval after its own first available bar. This is the layer beneath each cell's
    # `signal_coverage`: the root condition's `defined` channel answers "was the condition
    # DECIDED?", which a decisive sibling can settle (Kleene F∧U = F) and a NaN-skipping recursive
    # kernel can launder (state carried across a hole, finite output afterwards) — reading the raw
    # inputs directly puts no operator between the hole and the count. It is combo-independent
    # (the leaf SET is a property of the entry tree, not of any parameter assignment), so it is a
    # run-level panel rather than a per-cell one, and it is built whether or not anything fired —
    # a run whose sources are holed must say so even when the holes suppressed every firing.
    src_avail = source_availability(thesis, md)
    return {
        # The two grid labels: the swept ENTRY axes (with "horizon" only when it was swept) and
        # the regime — the reference every per-target panel is verified against.
        "params": list(cfg.param_levels),
        "targets": list(cfg.targets),
        # Which statistical mechanics produced this summary (two summaries compare only under
        # the same version) — see analysis.stats.STATISTICS_VERSION.
        "statistics_version": STATISTICS_VERSION,
        # Which sample every panel in this summary describes. There is no holdout and no
        # embargo: each declared cell is measured once over the whole index. Stamped anyway so
        # a consumer verifies the basis it is reading instead of assuming it — a summary that
        # does not say what it describes is not evidence.
        "gate_evidence_basis": "full_sample",
        # Geometry + extent of the evaluated interval. `n_bars` is the joined index length,
        # pure geometry no property of the data can shrink, which is what makes the per-cell
        # decision ledger's `n_undefined <= n_bars` re-checkable arithmetic. The endpoints let
        # a caller tell two runs over different data windows apart — the bookkeeping any
        # cross-run search discipline needs and a stateless verifier cannot itself perform.
        "n_bars": int(cfg.n_bars),
        "index_start": cfg.index[0].isoformat() if cfg.n_bars else None,
        "index_end": cfg.index[-1].isoformat() if cfg.n_bars else None,
        "bar_spacing": bar_spacing(cfg.index),
        # The DECLARED search burden: every param combo × horizon the sweep declared, whether
        # or not it fired. Non-firing combos cannot shrink it, and it is the ONLY multiplicity
        # input this engine emits — nothing in the report is corrected for it. Selection
        # across cells (and across runs, DSLs and data windows, which a single run cannot see
        # at all) belongs to the calling agent, and this is the number it prices that
        # selection against.
        "n_hypotheses_attempted": n_combos_attempted * len(cfg.horizon_values),
        "direction": thesis.params.direction,  # evidence stays self-describing
        # Benchmark self-description: when set, EVERY return in this summary is an excess
        # return — consumers must know which scale they are reading.
        "benchmark": cfg.bench_mode,
        "benchmark_source": md.benchmark_path,
        # Outcome self-description: what was measured (series + algebra + units) and the
        # target shape — a ``diff`` outcome's mean_ret is in the series' own units (bp, index
        # points), not a percent. ALWAYS the explicit dict: the default run stamps {target,
        # pct, fraction} rather than a null, so no consumer — and no checklist — ever has to
        # decode a missing stamp into a meaning.
        "outcome": {
            "series": cfg.outcome_series,
            "kind": cfg.outcome_kind,
            "units": _OUTCOME_UNITS[cfg.outcome_kind],
        },
        "target_shape": md.target_shape,
        # Target-mode self-description: which target semantics produced every cross-target
        # read in this summary. ALWAYS stamped — the checklist dispatches its rubric on it, so
        # a summary that does not say which exam applies is drifted input, not gradable.
        "target_mode": thesis.target_mode,
        "rotation": {
            "n_shifts": n_shifts,
            "p_resolution": (1.0 / (1.0 + n_shifts)) if n_shifts else float("nan"),
        },
        # CSCV → PBO over the grid's fired-and-closed cells: the symmetric block splits are
        # their own train/test discipline (block-local windows are pre-purged). A GRID-LEVEL
        # property of the search space the caller is about to select from — "if you pick the
        # best cell off this grid, how often would that pick fail to travel?" — attached to no
        # hypothesis and read by no cell's grade. Mounted as ONE nested block. The declared
        # count is the DECLARED combo × horizon grid (== n_hypotheses_attempted — one declared
        # cell per candidate combo, sans the target dimension), so the block's population
        # ledger can say how far short of the declared grid the scored population fell:
        # rel_cells holds only fired-and-closed cells, and CSCV never sees the rest.
        "pbo": cscv_pbo(
            rel_cells,
            cfg.n_bars,
            cfg.targets,
            off=cfg.off,
            mode=thesis.target_mode,
            n_combos_declared=len(cells),
        ),
        "sources": {
            tgt: source_coverage(src_avail, cfg.index, g) for g, tgt in enumerate(cfg.targets)
        },
        # The cross-sectional breadth panel collected in the combo loop — the effective-
        # universe ledger beside the availability one: `sources` says whether an input EXISTED
        # on a bar, `cross_breadth` says how many members a cross read actually stood on.
        # Always present; `[]` when the entry tree holds no cross node (every conjunction run,
        # by validation).
        "cross_breadth": cross_breadth,
        # Unconditional baseline: the same measurement with the entry condition removed —
        # every fillable anchor bar opens an observation under the SAME algebra, benchmark and
        # direction as the cells, so "conditional mean 3.1% on firing bars vs 0.4% on all
        # bars" is readable from the report alone. Pure reindexing of the per-horizon frames —
        # no new measurement. Exclusions reuse the exit-reason vocabulary minus "horizon" (a
        # baseline row has nothing to close), and `n_eligible + Σexclusions == n_anchor_bars`
        # is re-checkable arithmetic — the honesty channel that keeps a data hole from
        # silently shrinking the base rate's denominator. NO cell-vs-baseline difference or
        # uplift field is ever derived: the comparison is the caller's. In basket mode each
        # horizon entry ALSO carries the pooled (bar × member) row — the honest base rate for
        # a pooled conditional claim, whose counts are the per-target sums (hand-averaging
        # per-target baselines mis-weights under holes).
        "baseline": _baseline_panel(
            frames_by_h,
            cfg.horizon_values,
            cfg.n_bars,
            cfg.off,
            cfg.targets,
            cfg.bench_mode,
            cfg.custom_outcome,
            cfg.basket,
        ),
        # The report's spine — one entry per DECLARED combo × horizon, in declaration order,
        # non-firing combos included (built by `_cells_panel`, which walks the declared grid).
        "cells": cells,
    }


class _CellIdentity(TypedDict):
    """The head of a :class:`SummaryCell` as the panel loop opens it — identity, the per-target
    panel, and the mode-gated ``pooled`` block — so the cell literal below can be completed in
    one expression while keeping ``pooled`` right after ``by_target`` in the emitted order."""

    cell_id: str
    params: dict[str, ParamValue]
    by_target: dict[str, CellTargetPanel]
    pooled: NotRequired[CellPooledPanel]


def _pool_panel(
    closed: pd.DataFrame,
    rc: ReliabilityRead,
    h: int,
    sub_edges: np.ndarray,
    sub_windows: list[tuple[str | None, str | None]],
) -> CellTargetPanel:
    """ONE pool's complete evidence panel over its CLOSED rows — the per-target panel
    ``cells[i].by_target[t]`` and, in basket mode, the pooled cross-target panel are the SAME
    builder over different row sets, so the two mirror each other by construction rather than
    by convention. ``rc`` is the pool's reliability read (or the no-evidence default).

    A pool with no closed rows still gets a full panel — n=0 and NaN statistics. An OMITTED
    target would read as "not applicable here" when what actually happened is "this cell
    produced no evidence for a target it claims to hold across", and the regime conjunction is
    exactly what must not be quietly narrowed.
    """
    rets = closed["ret"].to_numpy(dtype=float) if not closed.empty else np.empty(0)
    bars = (
        closed["entry_bar"].to_numpy(dtype=np.int64)
        if not closed.empty
        else np.empty(0, dtype=np.int64)
    )
    moments = pool_moments(rets)
    # The attribution legs — on closed rows both are finite exactly where `ret` is (a NaN in
    # either leg propagated into `ret` upstream), except `ret_bench`, which is all-NaN on an
    # unbenchmarked run and so aggregates to NaN → null. Their means read the finite subset;
    # `benchmark_regression` filters PAIRS itself off the row-aligned columns — feeding it the
    # per-leg finite subsets would silently mis-pair.
    raw = _finite_column(closed, "ret_raw")
    bench = _finite_column(closed, "ret_bench")
    return {
        "n": moments["n"],
        "n_nonoverlap": int(rc["n_nonoverlap"]),
        "mean_ret": moments["mean_ret"],
        # The excess mean's own legs (evidence-only): mean_ret ≈ mean_ret_raw − mean_ret_bench
        # over the SAME closed pool, so a caller can tell "+0.8% because the target made +3.2%
        # against a +2.4% market" from "+0.8% because the target lost less than a falling
        # market". Under `cross_mean` the bench leg is each bar's basket mean; the POOLED
        # mean_ret is then the FIRING subset's cross-sectional selection tilt — each closed
        # bar's FULL cross-section demeans to zero, but the pool holds only the members that
        # fired, so ≈ 0 appears only when firings are basket-wide — and mean_ret_bench ≈
        # mean_ret_raw is the basket's own realized drift.
        "mean_ret_raw": float(np.mean(raw)) if raw.size else _NAN,
        "mean_ret_bench": float(np.mean(bench)) if bench.size else _NAN,
        # The legs' regression attribution (evidence-only): is the excess mean alpha, or
        # beta ≠ 1 riding market drift — the question the two means alone cannot answer.
        # Null fields + reason on an unbenchmarked or degenerate pool.
        "benchmark_regression": benchmark_regression(
            _aligned_column(closed, "ret_raw"), _aligned_column(closed, "ret_bench")
        ),
        "hit_rate": moments["hit_rate"],
        # Average win over average loss (evidence-only): the payoff asymmetry partnering
        # hit_rate's frequency; then gross win mass over gross loss mass — the hit-weighted
        # asymmetry partnering both; NaN, never infinity, when either side is empty.
        "win_loss_ratio": moments["win_loss_ratio"],
        "profit_factor": profit_factor(rets),
        # Distribution shape and tail reads (evidence-only): dispersion, the two moments that
        # say when the mean is a poor description, tail_ratio ≡ |p95/p05|, and cvar_5 — the
        # mean of the observations at or below ret_quantiles.p05, its historical-VaR partner.
        "std_ret": moments["std_ret"],
        "skewness": moments["skewness"],
        "kurtosis": moments["kurtosis"],
        "tail_ratio": moments["tail_ratio"],
        "cvar_5": moments["cvar_5"],
        "t_hac": rc["t_hac"],
        "hac_se": rc["hac_se"],
        "rot_p": rc["rot_p"],
        "rot_n_null": int(rc["rot_n_null"]),
        # |return|-mass concentration of THIS pool: a target riding one whale event cannot pass
        # through the regime claim on another target's breadth.
        "concentration": concentration(rets),
        # Episode-bootstrap CI over THIS pool (evidence-only): the dependence-robust
        # counterweight to the anti-conservative t_hac / rot_p.
        "boot": episode_bootstrap_ci(rets, bars, h),
        # Era visibility (evidence-only): the same three eras for every cell, this pool's
        # n / mean per era.
        "subperiods": [
            {"start": w[0], "end": w[1], **seg}
            for w, seg in zip(sub_windows, subperiod_means(rets, bars, sub_edges), strict=True)
        ],
        # Distribution shape (evidence-only): what a TYPICAL observation in this pool looked
        # like, which `mean_ret` cannot say. A positive mean sitting above a negative `p50` is
        # a pool carried by a few observations — a failure mode the concentration check does
        # not cover, since mild right skew concentrates no |return| mass in one episode. No
        # `n` here: the pool IS `n`.
        "ret_quantiles": pool_quantiles(rets),
        "worst_ret": float(np.min(rets)) if rets.size else _NAN,
        "best_ret": float(np.max(rets)) if rets.size else _NAN,
        # Holding-period path (evidence-only): how deep the interim drawdown ran and how far
        # the interim gain reached, before the horizon closed. RAW path on both sides — under a
        # benchmark `ret` is EXCESS while these are not, so the two are not commensurable and
        # no ret-vs-excursion ratio is meaningful here. `edge_ratio` is the ONE sanctioned
        # excursion ratio: both of its legs are RAW, so it survives a benchmark unchanged — and
        # it PAIRS its rows (both legs finite), so under asymmetric window holes it need not
        # equal the blocks' means' ratio. Each block covers its own finite subset (a window
        # hole censors `mae`/`mfe` on a row whose `ret` closed cleanly), so its `n` can sit
        # below the pool's.
        "mae_quantiles": mae_block(_finite_column(closed, "mae")),
        "mfe_quantiles": mfe_block(_finite_column(closed, "mfe")),
        "edge_ratio": edge_ratio(_aligned_column(closed, "mae"), _aligned_column(closed, "mfe")),
        # Path-timing medians (evidence-only): the timing pair aggregated the way the excursion
        # pair is, so the WHEN of the path needs no --trades-out either.
        "timing": timing_summary(
            _finite_column(closed, "bars_to_positive"),
            _finite_column(closed, "bars_to_trough"),
        ),
    }


def _cells_panel(
    cfg: _RunConfig,
    declared_cells: list[DeclaredCell],
    trades: pd.DataFrame,
    per_cell: dict[CellKey, ReliabilityRead],
    pooled_per_cell: dict[CellKey, ReliabilityRead],
    undef_by_combo: dict[ComboKey, np.ndarray],
) -> list[SummaryCell]:
    """The report's spine: ONE entry per DECLARED combo × horizon, in declaration order,
    non-firing combos included — every per-cell panel, evidence block and coverage ledger (the
    section comments below carry the doctrine, block by block)."""
    basket = cfg.basket
    feature_names = cfg.feature_names
    index = cfg.index
    n_bars = cfg.n_bars
    targets = cfg.targets
    # The cell axes: the swept entry axes with the horizon LAST and ALWAYS present — the
    # trades frame carries every one of them as a column, so this is both the partition key and
    # the rendered label's order.
    cell_axes = [*cfg.sweep_levels, "horizon"]
    # ---- the per-cell panel: one entry per DECLARED hypothesis ---------------------------------
    #
    # Partition the trades frame ONCE by the cell axes. A per-cell rescan of the frame would be
    # O(cells × rows); the grid is capped at 64 cells but the row count is not, and the
    # partition is exact — every row carries the assignment that produced it. The finer
    # (cell × target) partition is what the per-target loop reads, in the frame's own ascending
    # row order (`.indices` preserves it), so the sub-frames are exact.
    row_groups: dict[CellKey, np.ndarray] = {}
    target_groups: dict[CellKey, np.ndarray] = {}
    if not trades.empty:
        grouped = trades.groupby(cell_axes, sort=False, dropna=False).indices
        row_groups = {_norm_key(k): idx for k, idx in grouped.items()}
        grouped_t = trades.groupby([*cell_axes, "target"], sort=False, dropna=False).indices
        target_groups = {_norm_key(k): idx for k, idx in grouped_t.items()}
    empty_rows = trades.iloc[0:0]

    # Subperiod geometry, computed ONCE from the shared index so every cell reads
    # the SAME three eras. The window timestamps are run geometry — a cell that never fired still
    # reports the real eras with n=0, exactly as it reports the real targets with NaN statistics.
    sub_edges = subperiod_edges(n_bars)
    sub_windows = [
        (index[a].isoformat(), index[b - 1].isoformat()) if b > a else (None, None)
        for a, b in pairwise(sub_edges)
    ]

    undef_union: np.ndarray | None = None
    cells_panel: list[SummaryCell] = []
    for dc in declared_cells:
        cell_key = tuple(dc["params"][lvl] for lvl in cell_axes)
        combo_key = dc["combo_key"]
        idx = row_groups.get(_norm_key(cell_key))
        rows = trades.iloc[idx] if idx is not None else empty_rows
        # The cell needs BOTH pools: the FULL rows (censored firings included) are the only place
        # an unmeasured firing is visible, and the CLOSED rows are the only ones any statistic may
        # describe.
        rows_closed = rows[~rows["is_open"]] if not rows.empty else rows

        # The combo's undefined-decision mask. The lookup is by construction a hit (the declared
        # cell recorded the very combo tuple the mask was stored under); if it ever missed,
        # reporting zero would fail OPEN — a silent all-clear on a decision-side hole. The
        # fallback is the union over every attempted combo, which is strictly conservative: it
        # can only ever RAISE the count, never hide a hole.
        undef = undef_by_combo.get(dc["combo_tuple"])
        if undef is None:
            if undef_union is None:
                undef_union = (
                    np.logical_or.reduce(list(undef_by_combo.values()))
                    if undef_by_combo
                    else np.zeros((n_bars, len(targets)), dtype=bool)
                )
            undef = undef_union

        by_tgt: dict[str, CellTargetPanel] = {}
        cov: dict[str, CellOutcomeCoverage] = {}
        sig: dict[str, CellSignalCoverage] = {}
        # Feature ↔ outcome association (evidence-only): Spearman between the
        # entry-time snapshot and the realized closed return, per (feature × target) — time-axis
        # within one target only in BOTH modes (a pooled cross-member rank would conflate
        # cross-member level differences with time variation, which is exactly what the entry
        # signal itself already embodies in basket mode).
        fa: dict[str, dict[str, FeatureAssociation]] = {}
        # Per-member |return|-mass, the raw material of the basket's `member_share` decomposition.
        mass_by_target: dict[str, float] = {}
        for g, tgt in enumerate(targets):
            idx_t = target_groups.get(_norm_key((*cell_key, tgt)))
            rows_t = trades.iloc[idx_t] if idx_t is not None else empty_rows
            closed_t = rows_t[~rows_t["is_open"]] if not rows_t.empty else rows_t
            # The reliability pass holds only cells that fired with closed rows; a cell it never
            # measured reads its fields off the no-evidence default.
            rc = per_cell.get((*combo_key, tgt), _MISSING_RELIABILITY)
            by_tgt[tgt] = _pool_panel(closed_t, rc, dc["h"], sub_edges, sub_windows)
            rets_t = closed_t["ret"].to_numpy(dtype=float) if not closed_t.empty else np.empty(0)
            # Censoring ledger over the cell's FULL rows — ONE pool, because there is no holdout
            # and no embargo to split it into. Every firing lands under exactly one of the four
            # exit reasons, so the counts sum to `n_attempted` and no firing can vanish. This is
            # the panel that makes a DELETED outcome visible: the statistics above silently skip a
            # NaN-outcome row, which is how a vendor outage or an adversarial file could remove
            # adverse results and leave a clean-looking cell.
            counts = rows_t["exit_reason"].value_counts().to_dict() if not rows_t.empty else {}
            cov[tgt] = {
                "n_attempted": len(rows_t),
                "n_closed": int(counts.get("horizon", 0)),
                # The per-cell censoring ledger reports ALL FOUR reasons for every target, zeros
                # included, so a reader never has to guess whether an absent key means "none"
                # or "not counted" — and the arithmetic (the reasons sum to the pool's attempted
                # count) is re-checkable from the report alone.
                "exit_reasons": {k: int(counts.get(k, 0)) for k in EXIT_REASONS},
            }
            # Decision ledger — the twin of the censoring ledger on the other side of the firing.
            # The censoring ledger can only account for bars that FIRED; a missing decision input
            # suppresses the firing itself (three-valued evaluation: `init & ~defined`), leaving
            # no trace there at all. `n_bars` is the index length — pure geometry — so
            # `n_undefined <= n_bars` is arithmetic a reader re-checks, not a claim.
            sig[tgt] = {
                "n_bars": int(n_bars),
                "n_undefined": int(undef[:, g].sum()),
            }
            mass_by_target[tgt] = float(np.abs(rets_t).sum())
            for fname in feature_names:
                vals = closed_t[fname].to_numpy(dtype=float) if not closed_t.empty else np.empty(0)
                fa.setdefault(fname, {})[tgt] = feature_outcome_association(vals, rets_t)

        # Rendered cell label: the swept axes with the horizon LAST. The label is a CONVENIENCE
        # — a cell's identity is its `params` dict plus its position in this list, which is why
        # nothing downstream may assume labels are unique.
        head: _CellIdentity = {
            "cell_id": ",".join(f"{lvl}={fmt_num(dc['params'][lvl])}" for lvl in cell_axes),
            "params": dict(dc["params"]),
            "by_target": by_tgt,
        }
        if basket:
            # The basket's POOLED cross-target block — the panel the checklist grades in basket
            # mode, the SAME builder as `by_target[t]` over the cell's closed rows in frame
            # order (target declaration order within a bar — the deterministic tie order the
            # bootstrap's content-keyed seed sees), plus the member-mass decomposition.
            # `by_target` rides along as attribution evidence read by no check: which members
            # carried the pool is exactly what an industry report needs, and is never a
            # per-member verdict.
            prc = pooled_per_cell.get(combo_key, _MISSING_RELIABILITY)
            total_mass = sum(mass_by_target.values())
            # A full decomposition over every declared member, NEVER ranked — the raw material
            # of the "not one name" ceiling. Zero total mass → NaN shares, which the checklist
            # refuses rather than waves through (the empty-pool concentration precedent).
            member_share = {
                tgt: (mass_by_target[tgt] / total_mass if total_mass > 0 else _NAN)
                for tgt in targets
            }
            head["pooled"] = {
                **_pool_panel(rows_closed, prc, dc["h"], sub_edges, sub_windows),
                "member_share": {
                    "by_target": member_share,
                    "max_member_share_abs": (
                        max(member_share.values()) if total_mass > 0 else _NAN
                    ),
                },
            }
        # The two panels `cell_conditional_buckets` computes together, mounted below under the
        # names it keys them by.
        buckets = cell_conditional_buckets(rows_closed, feature_names)
        cell: SummaryCell = {
            **head,
            # Episode clustering over the cell's CLOSED rows, merged ACROSS targets: one
            # cluster is one market episode, so an edge that is a single crisis seen through
            # three targets cannot read as three episodes.
            "episode_stats": episode_stats(rows_closed),
            # The time-ordered episode LEDGER (evidence-only): the narrative companion to the
            # shares above — earliest first, never ranked, truncation explicit and
            # mass-conserving. In basket mode the same cross-target merge makes it the pooled
            # ledger for free.
            "episodes": episode_ledger(rows_closed),
            # The episode-deduplicated TWIN of the row-level statistics (evidence-only): the
            # same frozen cross-target merge, one aggregate per episode, the same statistic
            # family over the episodes — so one crisis smeared across ~h overlapping rows reads
            # as ONE episode here, and row-vs-episode divergence is the visible cluster
            # diagnostic. Reported, never a correction.
            "episode_profile": episode_profile(rows_closed),
            "outcome_coverage": cov,
            "signal_coverage": sig,
            # Per-cell conditional buckets: there is no run-level pooled version — a pooled
            # qcut would re-cut every time a cell joined the grid, so the same bar's
            # "conditioning" would move with grid composition. Per cell the pool is the
            # cell's own closed rows and nothing else can move it.
            "conditional_buckets": buckets["conditional_buckets"],
            "bucket_monotonicity": buckets["bucket_monotonicity"],
            "feature_association": fa,
        }
        cells_panel.append(cell)

    # One entry per declared (combo × horizon), emitted in declaration order — the loop above walks
    # `declared_cells`, which the measurement loop appended to exactly once per iteration of the
    # same product, so this identity holds BY CONSTRUCTION and no cell can be dropped for having
    # produced nothing.
    return cells_panel


class _GridResult(NamedTuple):
    """Everything the measurement loop produces: the per-observation frames, the
    reliability-pass inputs (per-target and pooled), the DECLARED grid (recorded before anything
    about firing is known — the honesty invariant's raw material), the per-combo
    undefined-decision masks, the cross-sectional breadth ledger, and the attempted combo
    count."""

    frames: list[pd.DataFrame]
    rel_cells: list[ReliabilityCell]
    pooled_cells: list[ReliabilityCell]
    declared_cells: list[DeclaredCell]
    n_combos_attempted: int
    undef_by_combo: dict[ComboKey, np.ndarray]
    cross_breadth: list[CrossBreadthEntry]


def _measure_grid(
    thesis: Thesis,
    md: MarketData,
    cfg: _RunConfig,
    frames_by_h: dict[int, _HorizonFrames],
    measure: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> _GridResult:
    """ONE pass over the declared grid: per combo, build the firing mask and its undefined
    ledger; per (combo × horizon × target), record every observation (the section comments
    below carry the doctrine)."""
    _measure = measure
    base_np = cfg.base_np
    basket = cfg.basket
    bench_mode = cfg.bench_mode
    custom_outcome = cfg.custom_outcome
    features = cfg.features
    horizon_values = cfg.horizon_values
    index = cfg.index
    n_bars = cfg.n_bars
    off = cfg.off
    param_levels = cfg.param_levels
    sign = cfg.sign
    sweep_levels = cfg.sweep_levels
    targets = cfg.targets
    frames: list[pd.DataFrame] = []
    rel_cells: list[ReliabilityCell] = []
    # The basket's OWN reliability roster, kept SEPARATE from `rel_cells` on purpose: the pooled
    # rotation null must rotate every FIRED member's mask — including a member whose every
    # observation is right-censored, which `rel_cells` (closed-rows-only by construction) never
    # holds — while `rel_cells` feeds the per-target reliability pass AND `cscv_pbo`, whose
    # combo-admissibility reads would move if fired-but-uncloseable members appeared in it.
    # Appending here instead keeps every conjunction number and both modes' CSCV byte-stable.
    pooled_cells: list[ReliabilityCell] = []
    # THE DECLARED GRID — one entry per (param combo × horizon), recorded as the loop declares it
    # and BEFORE anything about firing is known. `rel_cells` above only ever holds cells that fired
    # with closed rows, and a groupby over the trades frame cannot invent a row for a combo with no
    # observations, so driving the per-cell panel off either of those would silently DELETE every
    # non-firing hypothesis from the report — precisely the hypothesis a reader most needs to see,
    # because its absence is what makes a surviving cell look inevitable. The panel is built off
    # this list instead, which is why `len(summary["cells"]) == n_hypotheses_attempted` holds by
    # construction rather than by luck.
    declared_cells: list[DeclaredCell] = []

    n_combos_attempted = 0
    # Post-warmup undefined decision bars per attempted combo — the raw material of
    # the per-cell `signal_coverage` ledger below. Keyed by COMBO, not by (combo, horizon): the
    # entry condition is what a missing input renders undecidable, and the measurement horizon has
    # no say in it. Horizon siblings therefore legitimately report the same counts — each cell is
    # graded alone, so nothing is ever summed across them.
    undef_by_combo: dict[ComboKey, np.ndarray] = {}
    # Cross-sectional breadth (evidence-only): the per-bar finite-member count k each cross
    # kernel reduced over — recomputed bit-exactly off the node's memoized input frame
    # (`build_series` is a memo hit below: the signal build already materialized it) and
    # SUMMARIZED instead of discarded, so member warmup thinning the effective universe is
    # visible. Keyed per (node × combo): a swept input moves the warmup, so per-combo entries
    # are the only non-understating key — combos that do not move a node's input repeat the
    # same entry, the honest repetition `signal_coverage` already accepts, never a sum. ENTRY
    # tree only (features are evidence-side snapshots); cross nodes are basket-gated, so a
    # conjunction run emits the empty panel by construction. Read by no check.
    cross_breadth: list[CrossBreadthEntry] = []
    for combo, entry in vectorize.iter_param_assignments(thesis.entry):
        n_combos_attempted += 1
        mask = vectorize.signal(entry, md).to_numpy()  # (bars × targets) bool, warmup-gated
        combo_tuple = tuple(combo[lvl] for lvl in sweep_levels)
        undef_by_combo[combo_tuple] = vectorize.undefined_mask(entry, md)
        combo_params = {lvl: combo[lvl] for lvl in sweep_levels}
        for cross_node in iter_cross_series(entry):
            vals = vectorize.build_series(cross_node.input, md)[0].to_numpy(dtype=float)
            k = np.isfinite(vals).sum(axis=1)
            eff_min_valid = max(int(cross_node.min_valid), 2)  # the kernels' hard floor
            evaluated = k >= eff_min_valid
            k_eval = k[evaluated]
            full = k == len(targets)
            cross_breadth.append(
                {
                    "node": render_series(cross_node),
                    "params": combo_params,
                    "min_valid": eff_min_valid,
                    "n_bars": int(n_bars),
                    "n_bars_evaluated": int(evaluated.sum()),
                    "n_bars_below_full": int((evaluated & ~full).sum()),
                    "k_min": int(k_eval.min()) if k_eval.size else None,
                    "k_median": float(np.median(k_eval)) if k_eval.size else None,
                    "k_max": int(k_eval.max()) if k_eval.size else None,
                    "first_full_bar": (
                        index[int(np.argmax(full))].isoformat() if full.any() else None
                    ),
                }
            )
        for h in horizon_values:
            hf = frames_by_h[h]
            fwd = hf.fwd
            bwd = hf.bwd
            # The cell's identity carries the horizon ALWAYS, even when it is a fixed scalar the
            # sweep never varied: a cell that does not name its measurement window is not
            # reproducible from the report. `param_levels` holds "horizon" only when it was SWEPT,
            # so it stays the frame/lookup axis list while `params` stays the full identity.
            params = {**combo, "horizon": h}
            combo_key = tuple(params[lvl] for lvl in param_levels)
            declared_cells.append(
                {"params": params, "combo_key": combo_key, "combo_tuple": combo_tuple, "h": h}
            )
            for g, target in enumerate(targets):
                fire = np.flatnonzero(mask[:, g])
                fire = fire[fire + off <= n_bars - 1]  # must be able to fill the entry
                if fire.size == 0:
                    continue
                if basket:
                    # Every FIRED member's mask reaches the pooled rotation null, closed rows
                    # or not — a member whose firings are all right-censored still shaped the
                    # basket's cross-sectional firing pattern, and the common shift must
                    # preserve what it rotates.
                    pooled_cells.append(
                        {
                            "key": (*combo_key, target),
                            "mask_col": mask[:, g],
                            "fwd_col": fwd[:, g],
                            "h": h,
                        }
                    )
                ret = fwd[fire, g]
                closed = np.isfinite(ret)
                fe = fire + off
                exit_full = fe + h
                exit_capped = np.minimum(exit_full, n_bars - 1)
                exit_idx = np.where(closed, exit_capped, n_bars - 1)
                entry_px = base_np[fe, g]
                exit_px = np.where(closed, base_np[exit_capped, g], np.nan)

                # The row's cell identity: every swept entry axis and, ALWAYS, the horizon —
                # a trades row names its measurement window whether or not it was swept.
                df = pd.DataFrame(
                    {lvl: params[lvl] for lvl in (*sweep_levels, "horizon")},
                    index=range(fire.size),
                )
                df["target"] = target
                df["entry_time"] = index[fe]
                df["exit_time"] = index[exit_idx]
                # The firing BAR index (the basis of `nonoverlap_count` and the rotation cells) —
                # carried so every trades sub-frame can run the event-time HAC on actual bar
                # distances instead of event ordinals.
                df["entry_bar"] = fire
                df["entry_px"] = entry_px
                df["exit_px"] = exit_px
                df["ret"] = ret
                # The excess return's own legs (evidence-only attribution:
                # ret = ret_raw − ret_bench, exactly — the subtraction in `_forward` is
                # per-observation). Masked to CLOSED rows like `mae`/`mfe`, never like `pre_ret`:
                # the legs attribute `ret`, so they must cover exactly the pool `ret`'s statistics
                # describe — an unmasked raw leg on a `no_benchmark` row would hand a consumer a
                # raw mean over a LARGER pool than the excess mean it explains. Unbenchmarked
                # runs: ret_raw == ret and ret_bench is all-NaN — "no benchmark leg", never a
                # zero one.
                df["ret_raw"] = np.where(closed, hf.raw[fire, g], np.nan)
                bench_h = hf.bench
                df["ret_bench"] = (
                    np.where(closed, bench_h[fire, 0], np.nan) if bench_h is not None else np.nan
                )
                df["pre_ret"] = bwd[fire, g]  # raw price drift into the entry (evidence-only)
                # Post-entry MAE (signed ≤ 0): worst interim adverse mark over the full H/L of
                # [fill, fill+h−1] plus the exit print open[fill+h] — never the exit bar's own
                # high/low, which print after the exit. RAW path (never benchmark-adjusted).
                # Right-censored → NaN.
                ext_at = hf.mae_ext[fe, g]
                mae_raw = sign * _measure(ext_at, entry_px)
                df["mae"] = np.where(closed & np.isfinite(mae_raw), mae_raw, np.nan)
                # Post-entry MFE (signed ≥ 0): best interim favorable mark over the same window,
                # under the same RAW-path rule. The give-back companion to `mae`.
                fext_at = hf.mfe_ext[fe, g]
                mfe_raw = sign * _measure(fext_at, entry_px)
                df["mfe"] = np.where(closed & np.isfinite(mfe_raw), mfe_raw, np.nan)
                # Time-to-recovery companion: first forward bar the measured path is back ≥ entry.
                df["bars_to_positive"] = np.where(closed, hf.bars_to_positive[fe, g], np.nan)
                # Adverse-path trough duration: bars from fill until the MAE extremum.
                df["bars_to_trough"] = np.where(closed, hf.bars_to_trough[fe, g], np.nan)
                # Terminal right-censoring ("open") vs an in-data hole in the MEASURED leg
                # ("no_outcome") — the SAME discrimination on every path. The two are not
                # interchangeable: `open` is structural geometry every cell near the index end
                # exhibits and the per-cell coverage checklist allows it, while an in-bounds NaN
                # is a data hole that DELETED an outcome and refuses. Under a market benchmark an
                # in-bounds NaN target leg is the latter, so labelling it "open" would hide the
                # adverse results the hole removed.
                past_end = (fire + off + h) > (n_bars - 1)
                leg = np.where(past_end, "open", "no_outcome")
                if bench_mode or custom_outcome:
                    # Outcome leg measurable but the adjusted return NaN ⇒ the benchmark (or the
                    # cross-section) was missing over the window — censored, but distinguishably
                    # so. `past_end` implies the leg itself is NaN, so a terminal row still
                    # reads "open".
                    tf = hf.tgt_finite[fire, g]
                    df["exit_reason"] = np.where(
                        closed, "horizon", np.where(tf, "no_benchmark", leg)
                    )
                else:
                    df["exit_reason"] = np.where(closed, "horizon", leg)
                df["is_open"] = ~closed
                for fname, farr in features.items():
                    df[fname] = farr[fire, g]
                frames.append(df)

                if closed.any():
                    rel_cells.append(
                        {
                            "key": (*combo_key, target),
                            "mask_col": mask[:, g],
                            "fwd_col": fwd[:, g],
                            "h": h,
                        }
                    )

    return _GridResult(
        frames=frames,
        rel_cells=rel_cells,
        pooled_cells=pooled_cells,
        declared_cells=declared_cells,
        n_combos_attempted=n_combos_attempted,
        undef_by_combo=undef_by_combo,
        cross_breadth=cross_breadth,
    )


def run_backtest(thesis: Thesis, md: MarketData) -> EventStudyResult:
    cfg = _resolve_run_config(thesis, md)
    # Unpacked once: the names the remaining spine (and the _measure/_forward closures) reads.
    sign, basket, off = cfg.sign, cfg.basket, cfg.off
    outcome_kind = cfg.outcome_kind
    custom_outcome, base_np, n_bars = cfg.custom_outcome, cfg.base_np, cfg.n_bars
    targets = cfg.targets
    feature_names = cfg.feature_names
    horizon_values = cfg.horizon_values
    bench_mode, bench_col = cfg.bench_mode, cfg.bench_col

    # The measurement algebra (params.outcome.kind): pct = (b/a − 1), log = ln(b/a),
    # diff = (b − a). Non-finite (0-division, log of a non-positive ratio) → NaN, censoring the
    # observation like any other unmeasurable window.
    #
    # pct and log additionally require both endpoints STRICTLY POSITIVE. Both
    # are ratio algebras and are meaningless off a positive scale, but neither fails loudly:
    # a percent change through zero or between negative levels returns a FINITE number with an
    # inverted sign (−4 → −2 reads as +50% "gain" while the level fell), and ln of a
    # negative/negative ratio is finite too. The engine would have recorded those as real
    # returns. Censoring makes them `no_outcome`, which the gate's missingness contract then
    # refuses inside a gate pool — the fail-closed direction. `diff` is untouched: signed
    # levels are exactly what it exists to measure (a 10y yield crossing zero is a −0.5 move,
    # not a domain error). OHLCV targets cannot reach this (the strict loader already refuses
    # non-positive prices); series-shaped targets and feed outcomes can.
    def _measure(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            if outcome_kind == "pct":
                r = np.where((num > 0.0) & (den > 0.0), num / den - 1.0, np.nan)
            elif outcome_kind == "log":
                r = np.where((num > 0.0) & (den > 0.0), np.log(num / den), np.nan)
            else:  # diff
                r = num - den
        return r

    # Forward-return frame per horizon, anchored at the FIRING bar t (so it lines up with the firing
    # mask for the rotation null): fwd[t] = sign·measure(base[t+off+h], base[t+off]), NaN past the
    # edge. Returns (fwd, tgt_finite, raw, bench): tgt_finite marks where the OUTCOME leg alone is
    # measurable, so a benchmark-caused NaN can be labelled "no_benchmark" instead of "open" in the
    # trades frame; raw is the pre-subtraction signed leg and bench the direction-signed benchmark
    # leg (None when no benchmark is declared) — retained so the trades frame and the panels can
    # ATTRIBUTE each excess observation (`ret = ret_raw − ret_bench`) instead of discarding the
    # legs the subtraction consumed. Memory: one extra (rows × targets) frame per horizon only when
    # a benchmark is declared (unbenchmarked, `raw` aliases `fwd` — no copy; the branches below
    # rebind, they never mutate in place) plus a (rows × 1) bench column.
    def _forward(h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        den = shift_rows(base_np, off)
        num = shift_rows(base_np, off + h)
        fwd = sign * _measure(num, den)
        fwd = np.where(np.isfinite(fwd), fwd, np.nan)
        tgt_finite = np.isfinite(fwd)
        raw = fwd
        bench: np.ndarray | None = None
        # A benchmark column is loaded for `params.benchmark == "market"` and for nothing else,
        # so its presence IS that mode.
        if bench_col is not None:
            bden = shift_rows(bench_col, off)
            bnum = shift_rows(bench_col, off + h)
            # The benchmark leg is measured in the SAME algebra as the outcome: subtracting a
            # PERCENT benchmark return from a LOG outcome would mix units silently.
            # With matching algebra a log outcome yields the true log-excess
            # ln(tgt ratio) − ln(bench ratio). `diff` + benchmark is refused at DSL validation
            # (level units minus a return is incommensurable by construction), so only the two
            # ratio algebras arrive here.
            bret = _measure(bnum, bden)  # (rows × 1), broadcast across targets
            bench = sign * bret  # the direction-signed leg the subtraction consumes
            fwd = fwd - bench  # sign·(tgt_ret − bench_ret)
            fwd = np.where(np.isfinite(fwd), fwd, np.nan)
        elif bench_mode == "cross_mean":
            # Excess over the BASKET's own same-window mean (self included), in the outcome's own
            # algebra (`diff` never reaches here — basket refuses it at validation). The plain row
            # mean IS the fail-closed contract: any member's leg NaN at a bar propagates NaN into
            # the whole bar's benchmark leg, censoring every member's firing there as
            # `no_benchmark` — never a partial-basket mean, which would quietly demean the
            # surviving members by a DIFFERENT basket while their rows escape the censoring
            # ledger. `fwd` is already signed, and sign·(tgt − mean(tgt)) ≡ signed − mean(signed),
            # so demeaning after the sign is exact for shortonly too — which is also why the bar's
            # mean IS the signed benchmark leg, needing no further sign.
            mu = fwd.mean(axis=1, keepdims=True)
            bench = mu
            fwd = fwd - mu
            fwd = np.where(np.isfinite(fwd), fwd, np.nan)
        return fwd, tgt_finite, raw, bench

    # Pre-entry drift frame per horizon, anchored at the FIRING bar t like ``fwd`` so it aligns with
    # the firing mask: bwd[t] = sign·measure(base[t+off], base[t+off-h]) — the RAW move INTO the
    # entry over the same h-bar window as the forward measurement (never benchmark-adjusted: it is a
    # descriptive read of the measured path, not an outcome). Sign-aligned with ``fwd`` so NEGATIVE
    # means the series moved AGAINST the eventual position before entry (moved-against-position)
    # and POSITIVE means momentum continuation. Firing bars whose pre-window precedes the
    # data start are NaN (excluded), mirroring the forward-window censoring.
    def _backward(h: int) -> np.ndarray:
        num = shift_rows(base_np, off)  # base[t+off]
        den = shift_down(base_np, h - off)  # base[t+off-h]
        bwd = sign * _measure(num, den)
        return np.where(np.isfinite(bwd), bwd, np.nan)

    # Adverse frame for post-entry MAE (RAW path, never benchmark-adjusted — same doctrine as
    # ``pre_ret``). Long → the low frame (worst mark-to-market); short → the high frame. Feed
    # outcomes / series-shaped targets use the measured series itself (synthesized low=high=value).
    if custom_outcome:
        adverse_np = base_np
    elif sign > 0:
        adverse_np = md.low.to_numpy(dtype=float)
    else:
        adverse_np = md.high.to_numpy(dtype=float)
    # Excursion-window extremum of the adverse frame per horizon: full H/L over [t, t+h−1] plus
    # ONLY the exit print base[t+h] — long → min low, short → max high. MAE at fill bar fe is
    # then sign·measure(ext[fe], base[fe]) — ≤ 0 when defined, and ≤ ret because the exit print
    # is a member of the extremum set. The exit bar's own high/low never enter: they print after
    # the exit and belong to the next holding period.
    # The FAVORABLE mirror — same frame selection reflected: long → the high frame
    # (best interim mark), short → the low frame, custom outcomes → the measured series itself.
    # RAW path like the adverse side, never benchmark-adjusted. MFE is what a "gave back the paper
    # gain" statement is read off; it is an interim MARK, never an attainable exit, because the
    # observer has no exit rule to attain it with.
    if custom_outcome:
        favorable_np = base_np
    elif sign > 0:
        favorable_np = md.high.to_numpy(dtype=float)
    else:
        favorable_np = md.low.to_numpy(dtype=float)
    # EVERY combo-independent per-horizon frame, built once per horizon in ONE pass and carried
    # as a single record instead of nine parallel `*_by_h` dicts (each entry's arithmetic is
    # independent, so interleaving the builds changes no value). The path-evidence companions
    # (bars-to-positive/-trough) are hoisted here for the same reason the extrema are:
    # combo-independent, so computing them per cell repeated identical work up to grid-size
    # times over.
    frames_by_h: dict[int, _HorizonFrames] = {}
    for h in horizon_values:
        fwd_h, tgt_finite_h, raw_h, bench_h = _forward(h)
        frames_by_h[h] = _HorizonFrames(
            fwd=fwd_h,
            tgt_finite=tgt_finite_h,
            raw=raw_h,
            bench=bench_h,
            bwd=_backward(h),
            mae_ext=excursion_extremum(adverse_np, base_np, h, "min" if sign > 0 else "max"),
            mfe_ext=excursion_extremum(favorable_np, base_np, h, "max" if sign > 0 else "min"),
            bars_to_positive=bars_to_positive_full(base_np, h, sign, _measure),
            bars_to_trough=bars_to_trough_full(adverse_np, base_np, h, sign),
        )

    grid = _measure_grid(thesis, md, cfg, frames_by_h, _measure)
    frames = grid.frames
    rel_cells = grid.rel_cells
    pooled_cells = grid.pooled_cells
    declared_cells = grid.declared_cells
    n_combos_attempted = grid.n_combos_attempted
    undef_by_combo = grid.undef_by_combo
    cross_breadth = grid.cross_breadth

    # The trades frame's leading columns are a row's CELL identity — the swept entry axes, then
    # the horizon (always), then the target — ahead of the engine's own columns and the
    # feature snapshots.
    columns = [*cfg.sweep_levels, "horizon", "target", *_TRADE_COLUMNS, *feature_names]
    trades = (
        pd.concat(frames, ignore_index=True)[columns] if frames else pd.DataFrame(columns=columns)
    )

    # Overlap-aware inference, ONE pass over the whole index. Every declared cell that fired is
    # measured on its OWN firing mask against its OWN circular-shift null, so no cell's numbers
    # depend on any other cell's — adding a hypothesis to the grid changes nothing already
    # measured. The per-cell reads (`rot_p`, `t_hac`/`hac_se`, `n_nonoverlap`) are mounted into
    # each cell's per-target panel below; they are EVIDENCE (both estimators are known
    # anti-conservative — see analysis.stats), never a certificate.
    rel = reliability_summary(rel_cells, n_bars, targets)
    # The basket's pooled reliability reads, derived from the SAME rel_cells
    # entries — one pooled entry per combo × horizon: the common-shift rotation null (one shift
    # rotates every member's mask as a block, preserving the per-bar cross-sectional firing
    # pattern a rank signal fixes) and the pooled event-time HAC / greedy n_nonoverlap over the
    # concatenated member rows. Conjunction runs form no pooled read at all.
    pooled_per_cell = pooled_reliability_summary(pooled_cells, n_bars)["per_cell"] if basket else {}

    cells = _cells_panel(
        cfg,
        declared_cells,
        trades,
        rel["per_cell"],
        pooled_per_cell,
        undef_by_combo,
    )
    # ---- the summary: the run-level stamps (what was measured, over what geometry, in which
    # algebra — everything that is a property of the run, not of any hypothesis) and the per-cell
    # panel that carries everything that IS a property of a hypothesis, as ONE literal.
    summary = _run_summary(
        thesis, md, cfg, frames_by_h, rel, rel_cells, cross_breadth, n_combos_attempted, cells
    )

    return EventStudyResult(
        trades=trades,
        summary=summary,
        thesis=thesis,
    )
