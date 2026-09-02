"""``EventStudyResult``: the output of running a thesis through the forward-return event study.

Holds the per-observation DataFrame (one row per firing bar × horizon across every target ×
parameter combination — overlapping observations, censored ones flagged via
``is_open``/``exit_reason`` for charting) plus a summary dict. The summary carries NO pooled
headline scalar: there is no selected cell to describe, because the engine measures every declared
parameter × horizon cell and reports each one independently. What it carries instead is
``summary['cells']`` — ONE entry per DECLARED combo × horizon, non-firing combos included, each
with its own per-target statistics (``rot_p``/``t_hac``/``hac_se``/``n_nonoverlap`` among
them), ``conditional_buckets``, censoring ledger and decision ledger — alongside the grid-level
CSCV ``pbo`` block and the run-level geometry/provenance stamps (``n_bars``, ``index_start``/
``index_end``, ``n_hypotheses_attempted``, ``sources``). Every statistic is computed over the
CLOSED observations only. The thesis is retained.

Selection is deliberately absent here as everywhere else: nothing in this result ranks or crowns a
cell, and cross-cell multiplicity is the calling agent's to price against
``summary['n_hypotheses_attempted']``.

This module also defines ``EntryListReport`` — the signal-listing counterpart to the event study;
it evaluates only the entry firing mask and lists every firing timestamp over the provided series
(see ``api.list_entries``); it carries the TWO per-bar frames the CLI ships on flags of their own —
the root-series VALUES the entry tree thresholds on (``seikan run --root-series-out``, the why) and
the 0/1 firing matrix (``--entry-flags-out``, the whether) — and it does not ride on the backtest
result.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from seikan.dsl.schema import Thesis
from seikan.types import DataReport, EntryRow, RunSummary


@dataclass
class EventStudyResult:
    trades: pd.DataFrame  # closed + open (censored ones flagged via is_open / exit_reason="open")
    summary: RunSummary
    thesis: Thesis
    # The strict-CSV data_report for everything the run read and checked (per-file results + join
    # info + coverage warnings) — attached by ``api.compile_thesis``, surfaced by the CLI.
    data_report: DataReport | None = None

    def __repr__(self) -> str:
        """GRID FACTS ONLY — how much was declared, how much was measured, how much is censored.

        There is no headline effect size to print: reporting one number here would name a cell the
        engine never selected, which is exactly the summary-of-the-winner framing this result type
        does not have. Every read is defensive (``.get`` with a fallback, column-presence checks)
        because ``repr`` is a debugging affordance — it must survive a partially-built or drifted
        summary rather than raise inside a traceback someone is trying to read.
        """
        s = self.summary
        n_open = (
            int(self.trades["is_open"].sum())
            if not self.trades.empty and "is_open" in self.trades.columns
            else 0
        )
        return (
            f"EventStudyResult(name={self.thesis.name!r}, "
            f"cells_declared={s.get('n_hypotheses_attempted', 0)}, "
            f"observations={len(self.trades)}, open={n_open})"
        )


@dataclass
class EntryListReport:
    """Every timestamp the entry trigger fires at, ascending, per (signal combo × target).

    The signal-listing counterpart to the event study: it evaluates only the entry firing mask
    (sharing the same DSL + compiler as ``run_backtest``) and lists every True bar's timestamp —
    no forward returns, no statistics. These are raw signal firings, not observations: a last-bar
    firing is listed even though it has no next bar to fill at (observations anchor at the next
    open). The horizon axis does not affect the firing mask and is excluded from the rows.
    ``series_end`` is the last bar of the provided joined index — whether that series is up to
    date with the market is the caller's business, never checked here.

    The per-bar evidence is split in TWO frames rather than one wide matrix, because the CLI ships
    each on its own flag: ``root_series`` is what ``seikan run --root-series-out`` writes (the
    values the entry tree thresholds on, so a caller can see WHY a bar did or did not fire), while
    ``entry_flags`` is what ``seikan run --entry-flags-out`` writes (the mask itself). The flags
    earn an output of their own because a final-bar firing anchors no observation — the runner
    drops a firing it cannot fill, so it produces no trades row and no ``outcome_coverage`` count,
    and nothing in observation shape can hold it. This is the same ``vectorize.signal`` mask the
    backtest measures at, bit for bit. What no CLI output carries is ``entries`` itself: the raw
    per-(combo × target) timestamp rows below are library-only.
    """

    series_end: pd.Timestamp  # last bar of the provided joined index (no freshness claim)
    # {<swept signal param levels…>, "target": str, "timestamps": [pd.Timestamp, …] ascending
    # (empty when the condition never fires)}, one row per (combo × target).
    entries: list[EntryRow]
    # Wide per-bar VALUE frame on the joined index: one column per deduplicated root series node
    # (each threshold operand except bare constants, scalarized per combo). This is the frame the
    # root-series output CSV carries (``seikan run --root-series-out``), never printed.
    root_series: pd.DataFrame
    # Wide per-bar 0/1 frame on the joined index: one column per (combo × target). This is the
    # frame the entry-flags output CSV carries (``seikan run --entry-flags-out``), and the library's
    # handle on the very mask ``entries`` was read off, so the two can never disagree.
    entry_flags: pd.DataFrame
    # The strict-CSV data_report for everything the listing read (attached by ``api.list_entries``).
    data_report: DataReport | None = None

    def __repr__(self) -> str:
        # A row's swept-axis levels are numbers, so the sized-value check is what tells the
        # timestamp list apart from them — and it keeps this defensive the way a ``repr`` must be:
        # a drifted row missing its ``timestamps`` contributes 0 rather than raising.
        fires = sum(
            len(stamps)
            for r in self.entries
            if isinstance(stamps := r.get("timestamps", ()), (list, tuple))
        )
        return (
            f"EntryListReport(series_end={self.series_end!s}, rows={len(self.entries)}, "
            f"fires={fires})"
        )
