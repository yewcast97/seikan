"""The library's two compute entry points over a loaded :class:`MarketData`:
:func:`compile_thesis` (the full-grid event study) and :func:`list_entries` (the firing listing);
neither loads. A host imports them, the loader (:func:`resolve_data_files`, then
:func:`load_market_data`) and :func:`canonical_dsl_hash` from the package root, ``seikan``.
Everything here is stateless: no database, no network, no home directory.
"""

from __future__ import annotations

from typing import NoReturn, cast

import pandas as pd

from seikan import dataio
from seikan.analysis.result import EntryListReport, EventStudyResult
from seikan.compiler import vectorize
from seikan.compiler.data import MarketData
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
from seikan.types import DataIssue, DataReport, EntryRow, ParamValue

__all__ = ["compile_thesis", "list_entries"]


def _refuse(md: MarketData, issues: list[DataIssue], message: str) -> NoReturn:
    """Refuse a pre-flight guard by AMENDING the loaded report — never rebuilding one, so the
    refusal can never invent a file it never read. A hand-built ``MarketData`` may carry no
    report at all (the loader always attaches one), in which case the payload carries the
    top-level errors alone — a partial report, honestly partial."""
    report = md.report
    refused: dict[str, object] = dict(report) if report is not None else {}
    refused["errors"] = [*(report["errors"] if report is not None else []), *issues]
    refused["ok"] = False
    # The no-report branch is a partial DataReport by design; the cast says so.
    raise dataio.DataError(cast("DataReport", refused), message)


def _ensure_loaded_for(md: MarketData, thesis: Thesis) -> None:
    """Refuse a ``MarketData`` loaded under another data declaration than ``thesis.data``: its
    feeds are aligned under THAT declaration's lags, window and targets, and so are the memos
    riding it, so a thesis declaring a 1-day lag measured on a lag-0 load reads every feed a bar
    early — evidence under a declaration it was never read under. The loader refuses bindings
    that do not answer the declaration it loads, so ``md.targets`` and the loaded feeds are
    ``md.spec``'s own and comparing declarations compares what was loaded. A hand-built
    ``MarketData`` (no ``spec``) is its builder's own claim."""
    if md.spec is not None and md.spec != thesis.data:
        raise ValueError(
            "the MarketData was loaded under another data declaration than this thesis's "
            "(thesis.data: targets, feeds, lags, window) — load it for this thesis with "
            "load_market_data(thesis.data, files)"
        )


def _sufficient_report(md: MarketData, thesis: Thesis) -> DataReport | None:
    """The data_report a compiled result carries — ``md.report`` itself, or a copy with this
    thesis's ``barely_sufficient`` join warning — refusing up front when the joined index cannot
    close a single observation at the longest horizon (:class:`~seikan.dataio.DataError`
    carrying the full data_report).

    ``md.report`` describes the load and is never written: the warning describes one thesis's
    horizon, and appended to the shared report it would surface in every result over that
    ``md`` — a later compile's horizon written into an earlier result."""
    errors, warnings = dataio.sufficiency_check(len(md.index), thesis.params.horizon)
    if errors:
        _refuse(md, errors, errors[0]["message"])
    # A hand-built ``MarketData`` may carry no report (the loader always attaches one).
    report = md.report
    if report is None or report["join"] is None or not warnings:
        return report
    join = report["join"]
    return {**report, "join": {**join, "warnings": [*join["warnings"], *warnings]}}


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
    :func:`list_entries`, and it must be a load of THIS thesis's data declaration (``ValueError``
    otherwise: its feeds are aligned under the lags and window it was loaded with, and the loader
    has already refused bindings that do not answer that declaration). The
    measurement-side guards live HERE: an index too short to close one observation at the
    longest horizon refuses, as does a ``volume`` read against volume-less data
    (``params.features`` included). A barely-sufficient index is recorded as a join warning in
    the RESULT's ``data_report`` — a copy of ``md.report``, which describes the load alone and is
    never written, so a returned result never changes under a later call over the same ``md``.

    The measurement: builds the entry firing mask for every target and parameter combination,
    records an overlapping raw forward-return observation at each firing bar over every measurement
    horizon (no exit condition, no position state machine), and returns the recorded observations
    plus the summary: one independently measured entry per DECLARED parameter × horizon cell —
    non-firing combos included — with overlap-honest per-cell statistics. Nothing is selected,
    ranked, or crowned; the whole grid is measured on the whole sample. The result carries the
    ``data_report`` describing everything that was read and checked.
    """
    _ensure_loaded_for(md, thesis)
    data_report = _sufficient_report(md, thesis)
    _ensure_volume_available(md, thesis, include_features=True)
    result = run_backtest(thesis, md)
    result.data_report = data_report
    return result


def list_entries(thesis: Thesis, md: MarketData) -> EntryListReport:
    """List every timestamp at which the thesis's entry condition fires over the provided series —
    the signal-listing counterpart to :func:`compile_thesis` (no backtest, no statistics, no
    horizon-runway requirement).

    Shares the event study's DSL and compiler: it consumes the same materialized ``MarketData``
    (the caller loads once with ``load_market_data(thesis.data, files)``; the evaluation memos
    ride it, so a backtest that already ran over this ``md`` leaves nothing here to rebuild) and
    builds the same entry firing mask, then reports every True bar's timestamp in ascending order
    — for each (swept signal param combo × target). Because it reuses ``vectorize.signal``, the
    listed firings are bit-identical to the bars the backtest measures at. These are raw signal
    firings, not observations: a last-bar firing is listed even though it has no next bar to fill
    at (the runner keeps only the firings it can fill, ``fire + 1 <= n_bars - 1``, so that one has
    no trades row and no ``outcome_coverage`` count). The measurement ``horizon`` does not affect
    the firing mask, so it is not part of the iteration (``iter_param_assignments`` expands only
    the entry-tree sweeps). Whether the provided series is up to date is the caller's business —
    ``series_end`` only reports where the given joined index stops.

    The report also carries the per-bar evidence in two frames, and the CLI writes each one on its
    own flag. ``root_series`` holds every deduplicated root series node (each threshold operand
    except bare constants — the operands of a Condition embedded in an event node included, since
    each is a decision root of its own — scalarized per combo, off the same memos, so no series is
    ever built twice); ``seikan run --root-series-out`` writes it, because it answers why a bar did
    or did not fire. ``entry_flags`` holds the 0/1 column per (combo × target), and ``seikan run
    --entry-flags-out`` writes it — the one CLI output a last-bar firing reaches. Flags and
    ``timestamps`` are read off the SAME ``vectorize.signal`` mask, so they agree bit for bit —
    with each other and with the backtest. What stays library-only is the raw ``entries`` ROWS:
    the per-(combo × target) timestamp lists have no CLI output, so a caller wanting firings in
    list shape rather than as a per-bar matrix comes through here.

    Like :func:`compile_thesis`, ``md`` must be a load of the thesis's data declaration.
    """
    _ensure_loaded_for(md, thesis)
    _ensure_volume_available(md, thesis, include_features=False)
    rows: list[EntryRow] = []
    value_cols: dict[str, tuple[str, pd.DataFrame]] = {}  # node canonical JSON -> (label, values)
    # (combo, boolean signal frame)
    entry_cols: list[tuple[dict[str, ParamValue], pd.DataFrame]] = []
    for combo, entry in vectorize.iter_param_assignments(thesis.entry, thesis.axes):
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


def _claim_column(name: str, taken: set[str]) -> str:
    """``name``, or ``name#N`` for the smallest ``N ≥ 2`` whose header key
    (:func:`seikan.dataio.header_key`) no earlier column holds — claimed in ``taken``.
    Deterministic in claim order."""
    claimed, n = name, 1
    while dataio.header_key(claimed) in taken:
        n += 1
        claimed = f"{name}#{n}"
    taken.add(dataio.header_key(claimed))
    return claimed


def _per_target_frame(md: MarketData, labelled: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """One column per (label × target), labels outer, targets inner, on the bar index: named
    ``label@target`` when several targets run (``label`` alone otherwise), and claimed against
    ``datetime`` and every earlier column by :func:`_claim_column`."""
    multi = len(md.targets) > 1
    taken = {"datetime"}
    columns: dict[str, pd.Series] = {}
    for label, frame in labelled:
        for target in md.targets:
            name = f"{label}@{target}" if multi else label
            columns[_claim_column(name, taken)] = frame[target]
    out = pd.DataFrame(columns, index=md.index)
    out.index.name = "datetime"
    return out


def _root_series_frame(
    md: MarketData,
    value_cols: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Assemble ``EntryListReport.root_series``: one value column per deduplicated root series
    node, in first-appearance order with targets inner — a deterministic layout, so identical runs
    write byte-identical CSVs. The ``@<target>`` suffix appears only when several targets run.

    ``datetime`` is the ONE reserved name here (it labels the index), so a value label colliding
    with it — or with an earlier label — takes a ``#N`` suffix (:func:`_claim_column`), names
    compared as the strict reader compares headers: targets ``A`` and ``a`` write ``close@A`` and
    ``close@a``, one header to that reader, so the second is ``close@a#2`` and the CSV re-reads.
    The reservation set is deliberately narrow: this frame carries no entry flags, so no flag
    name can shadow a value label and an external feed named ``entry`` keeps the name the caller
    gave it.
    """
    return _per_target_frame(md, value_cols)


def _entry_flags_frame(
    md: MarketData,
    entry_cols: list[tuple[dict[str, ParamValue], pd.DataFrame]],
) -> pd.DataFrame:
    """Assemble ``EntryListReport.entry_flags``: one 0/1 column per (combo × target), in
    combo-iteration × target order — the frame ``seikan run --entry-flags-out`` writes, so the
    layout is deterministic and identical runs emit byte-identical CSVs. Columns are named
    ``entry`` / ``entry[axis=value,...]`` with the ``@<target>`` suffix when several targets run —
    canonical names, one per combo × target, unique save in CASE: the strict reader compares
    headers case-insensitively, so targets ``A`` and ``a`` would write ``entry@A`` and ``entry@a``
    as one header it refuses. A name taken under that rule takes a ``#N`` suffix
    (:func:`_claim_column`), as in the root-series frame."""
    labelled = [
        (
            "entry"
            if not combo
            else ("entry[" + ",".join(f"{k}={fmt_num(v)}" for k, v in combo.items()) + "]"),
            sig.astype(int),
        )
        for combo, sig in entry_cols
    ]
    return _per_target_frame(md, labelled)
