"""The public library API — the one home a host imports from.

Load once with :func:`load_market_data` (after :func:`resolve_data_files` has bound the
thesis's declared data keys to files), then hand the materialized :class:`MarketData` to
:func:`compile_thesis` (the full-grid event study) and/or :func:`list_entries` (the firing
listing); neither loads. :func:`canonical_dsl_hash` is the thesis identity the CLI's ``hash``
subcommand emits. Everything here is stateless: no database, no network, no home directory.
"""

from __future__ import annotations

from typing import NoReturn, cast

import pandas as pd

from seikan import dataio
from seikan.analysis.result import EntryListReport, EventStudyResult
from seikan.compiler import vectorize
from seikan.compiler.data import DataFiles, MarketData, load_market_data, resolve_data_files
from seikan.compiler.runner import run_backtest
from seikan.dsl.schema import (
    Constant,
    Thesis,
    fmt_num,
    iter_condition_series,
    iter_source_leaves,
    render_series,
    series_source_leaves,
)
from seikan.emitted import validate_summary
from seikan.gate import canonical_dsl_hash
from seikan.types import DataIssue, DataReport, EntryRow, ParamValue

#: ``MarketData`` / ``load_market_data`` — and the ``DataFiles`` / ``resolve_data_files`` pair that
#: locates what the DSL only names — are re-exported deliberately: this module is the public API's
#: one home, and the loader is its front door. A caller resolves the thesis's data keys against the
#: files it has, loads ONCE with ``load_market_data(thesis.data, files)``, and hands the
#: materialized ``MarketData`` to ``compile_thesis`` and/or ``list_entries``, which never load.
#: ``canonical_dsl_hash`` (from ``gate.py``) is re-exported for the same reason: a caller computing
#: a thesis's identity imports it from the public home, never from an internal module — and the
#: ``seikan hash`` subcommand is the same identity without writing Python at all.
__all__ = [
    "DataFiles",
    "MarketData",
    "canonical_dsl_hash",
    "compile_thesis",
    "list_entries",
    "load_market_data",
    "resolve_data_files",
]


def _refuse(md: MarketData, issues: list[DataIssue], message: str) -> NoReturn:
    """Refuse a pre-flight guard by AMENDING the loaded report — never rebuilding one, so the
    refusal can never invent a file it never read. A hand-built ``MarketData`` may carry no
    report at all (the loader always attaches one), in which case the payload carries the
    top-level errors alone — a partial report, honestly partial."""
    report = md.report
    prior = report.get("errors") if report is not None else None
    refused: dict[str, object] = dict(report) if report is not None else {}
    refused["errors"] = [*(prior or []), *issues]
    refused["ok"] = False
    # ONE documented cast (previously two, one per guard): the no-report branch is a partial
    # DataReport by design, and widening DataError for one defensive branch would cost more
    # than it says.
    raise dataio.DataError(cast("DataReport", refused), message)


def _ensure_sufficient(md: MarketData, thesis: Thesis) -> None:
    """Refuse up front when the joined index cannot close a single observation at the longest
    horizon (raises :class:`~seikan.dataio.DataError` carrying the full data_report); a
    barely-sufficient index lands as a join warning in the report instead."""
    errors, warnings = dataio.sufficiency_check(len(md.index), thesis.params.horizon)
    # A hand-built ``MarketData`` may carry no report at all (the loader always attaches one), so
    # every read below is guarded — the refusal still has to name what failed.
    report = md.report
    join = report.get("join") if report is not None else None
    if join is not None and warnings:
        # Membership-deduped, so repeat compiles over one shared ``md`` record the warning ONCE:
        # `sufficiency_check`'s message is a pure function of (index length, max horizon), so an
        # identical entry is the same fact restated, while a different thesis (longer horizon)
        # still records its own distinct warning.
        existing = join.setdefault("warnings", [])
        for w in warnings:
            if w not in existing:
                existing.append(w)
    if errors:
        _refuse(md, errors, errors[0]["message"])


def _ensure_volume_available(md: MarketData, thesis: Thesis, *, include_features: bool) -> None:
    """Refuse up front when the thesis reads OHLCV field ``volume`` but the loaded data carries no
    volume column (series-shaped targets synthesize none) — raising
    :class:`~seikan.dataio.DataError` (CLI exit 2, the missing-input class) rather than letting
    ``MarketData.field`` raise its late ``ValueError`` deep in the runner (exit 4). With
    ``include_features`` the ``params.features`` series are scanned too (``compile_thesis`` builds
    them; ``list_entries`` does not)."""
    if md.volume is not None:
        return
    leaves = set(iter_source_leaves(thesis.entry))
    if include_features:
        for node in (thesis.params.features or {}).values():
            leaves.update(series_source_leaves(node))
    if ("field", "volume") not in leaves:
        return
    where = "the entry condition" + (" (or a params.features series)" if include_features else "")
    message = (
        f"{where} reads OHLCV field 'volume' but the loaded data carries no volume column "
        "(series-shaped targets synthesize none)"
    )
    _refuse(md, [{"code": "missing_volume", "message": message}], message)


def compile_thesis(thesis: Thesis, md: MarketData) -> EventStudyResult:
    """Compile a thesis to a forward-return event study and return its measured result.

    The caller loads: ``md`` is the materialized market data from
    ``load_market_data(thesis.data, files)`` (the strict-CSV front door, which raises
    :class:`~seikan.dataio.DataError` with a complete per-file report on any violation) — one load
    can serve both this function and
    :func:`list_entries`. The measurement-side guards live HERE: an index too short to close one
    observation at the longest horizon refuses, as does a ``volume`` read against volume-less data
    (``params.features`` included). A barely-sufficient index is recorded as a join warning into
    ``md.report`` IN PLACE — visible to anything else sharing that ``md`` (harmless, the warning
    describes the same load) and recorded ONCE: a repeat call over the same ``md`` dedupes it.

    The measurement: builds the entry firing mask for every target and parameter combination,
    records an overlapping raw forward-return observation at each firing bar over every measurement
    horizon (no exit condition, no position state machine), and returns the recorded observations
    plus the summary: one independently measured entry per DECLARED parameter × horizon cell —
    non-firing combos included — with overlap-honest per-cell statistics. Nothing is selected,
    ranked, or crowned; the whole grid is measured on the whole sample. The result carries the
    ``data_report`` describing everything that was read and checked.

    The thesis is revalidated as a private copy at this boundary: the models are frozen but
    freezing is shallow, so a caller that mutated a nested list or dict after validation gets a
    ``ValidationError`` here (the CLI maps it to exit 3) instead of an engine run over a document
    no validator ever saw. The cost is microseconds against a backtest. The summary handed back
    is validated against its declared contract the same way (:func:`seikan.emitted.
    validate_summary`) — a few milliseconds against an O(grid × length) run.
    """
    thesis = Thesis.model_validate(thesis.model_dump(mode="json"))
    _ensure_sufficient(md, thesis)
    _ensure_volume_available(md, thesis, include_features=True)
    result = run_backtest(thesis, md)
    # The engine verifies its own output at the public seam: the summary is validated against
    # the RunSummary contract here, on the library path, exactly as the CLI validates the whole
    # document at emission — a shape the engine does not declare is a seikan bug, never a
    # result to hand a caller.
    validate_summary(result.summary)
    result.data_report = md.report
    return result


def list_entries(thesis: Thesis, md: MarketData) -> EntryListReport:
    """List every timestamp at which the thesis's entry condition fires over the provided series —
    the signal-listing counterpart to :func:`compile_thesis` (no backtest, no statistics, no
    horizon-runway requirement).

    Shares the event study's DSL and compiler: it consumes the same materialized ``MarketData`` (the
    caller loads once with ``load_market_data(thesis.data, files)`` — the CLI shares one load
    across both compute paths, and the evaluation memo rides it as the ``MarketData`` memos, so a
    backtest that
    already ran over this ``md`` leaves nothing here to rebuild) and builds the same entry firing
    mask, then
    reports every True bar's timestamp
    in ascending order — for each (swept signal param combo × target). Because it reuses
    ``vectorize.signal``, the listed firings are bit-identical to the bars the backtest measures at.
    These are raw signal firings, not observations: a last-bar firing is listed even though it has
    no next bar to fill at. The measurement ``horizon`` does not affect the firing mask, so it is
    not part of the iteration (``iter_param_assignments`` expands only the entry-tree sweeps).
    Whether the provided series is up to date is the caller's business — ``series_end`` only
    reports where the given joined index stops.

    The report also carries the per-bar evidence in two frames, and the CLI writes each one on its
    own flag. ``root_series`` holds every deduplicated root series node (each threshold operand
    except bare constants, scalarized per combo — the value frames are already in the ``MarketData``
    memos
    from this function's own ``signal`` calls, or from a backtest that ran first, so no series is
    ever built twice); ``seikan run --root-series-out`` writes it, because it
    answers why a bar did or did not fire. ``entry_flags`` holds the 0/1 column per (combo ×
    target), and ``seikan run --entry-flags-out`` writes it. That the mask gets an output of its
    own is exactly the last-bar seam noted above: the runner drops a firing it cannot fill
    (``fire + off <= n_bars - 1``), so a final-bar firing produces no trades row and no
    ``outcome_coverage`` count, and nothing in observation shape can report it. Flags and
    ``timestamps`` are read off the SAME ``vectorize.signal`` mask, so they agree bit for bit —
    with each other and with the backtest. What stays library-only is the raw ``entries`` ROWS: the
    per-(combo × target) timestamp lists have no CLI output, so a caller wanting firings in list
    shape rather than as a per-bar matrix comes through here.

    Like :func:`compile_thesis`, the thesis is revalidated as a private copy at this boundary —
    frozen models are shallow, and a mutated nested list/dict must refuse here, not run.
    """
    thesis = Thesis.model_validate(thesis.model_dump(mode="json"))
    _ensure_volume_available(md, thesis, include_features=False)
    rows: list[EntryRow] = []
    value_cols: dict[str, tuple[str, pd.DataFrame]] = {}  # node canonical JSON -> (label, values)
    # (combo, boolean signal frame)
    entry_cols: list[tuple[dict[str, ParamValue], pd.DataFrame]] = []
    for combo, entry in vectorize.iter_param_assignments(thesis.entry):
        sig = vectorize.signal(entry, md)  # (bars × targets), warmup-gated
        mask = sig.to_numpy()
        rows.extend(
            {**combo, "target": target, "timestamps": list(md.index[mask[:, j]])}
            for j, target in enumerate(md.targets)
        )
        for node in iter_condition_series(entry):
            if isinstance(node, Constant):  # a bare threshold cutoff, not a signal series
                continue
            key = node.model_dump_json()  # scalarized-node identity == build_series memo key body
            if key not in value_cols:
                value_cols[key] = (render_series(node), vectorize.build_series(node, md)[0])
        entry_cols.append((combo, sig))
    return EntryListReport(
        series_end=md.index[-1],
        entries=rows,
        root_series=_root_series_frame(md, list(value_cols.values())),
        entry_flags=_entry_flags_frame(md, entry_cols),
        data_report=md.report,
    )


def _root_series_frame(
    md: MarketData,
    value_cols: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Assemble ``EntryListReport.root_series``: one value column per deduplicated root series
    node, in first-appearance order with targets inner — a deterministic layout, so identical runs
    write byte-identical CSVs. The ``@<target>`` suffix appears only when several targets run.

    ``datetime`` is the ONE reserved name here (it labels the index), so a value label colliding
    with it — or with an earlier label — takes a ``#N`` suffix. The reservation set is deliberately
    narrow: this frame carries no entry flags, so no flag name can shadow a value label and an
    external feed named ``entry`` keeps the name the caller gave it.
    """
    multi = len(md.targets) > 1

    def col(base: str, target: str) -> str:
        return f"{base}@{target}" if multi else base

    taken = {"datetime"}
    columns: dict[str, pd.Series] = {}
    for label, values in value_cols:
        for target in md.targets:
            name, n = col(label, target), 1
            while name in taken:
                n += 1
                name = f"{col(label, target)}#{n}"
            taken.add(name)
            columns[name] = values[target]
    out = pd.DataFrame(columns, index=md.index)
    out.index.name = "datetime"
    return out


def _entry_flags_frame(
    md: MarketData,
    entry_cols: list[tuple[dict[str, ParamValue], pd.DataFrame]],
) -> pd.DataFrame:
    """Assemble ``EntryListReport.entry_flags``: one 0/1 column per (combo × target), in
    combo-iteration × target order — the frame ``seikan run --entry-flags-out`` writes, so the
    layout is deterministic and identical runs emit byte-identical CSVs. Columns are named
    ``entry`` / ``entry[axis=value,...]`` with the ``@<target>`` suffix when several targets run —
    canonical names, unique by construction, so there is no collision to disambiguate."""
    multi = len(md.targets) > 1
    columns: dict[str, pd.Series] = {}
    for combo, sig in entry_cols:
        base = (
            "entry"
            if not combo
            else ("entry[" + ",".join(f"{k}={fmt_num(v)}" for k, v in combo.items()) + "]")
        )
        for target in md.targets:
            columns[f"{base}@{target}" if multi else base] = sig[target].astype(int)
    out = pd.DataFrame(columns, index=md.index)
    out.index.name = "datetime"
    return out
