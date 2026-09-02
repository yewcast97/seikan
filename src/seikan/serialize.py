"""JSON-safe serialization of engine results + the CLI's caller-nominated CSV writers.

``compile_thesis`` returns pandas/numpy-typed objects; these helpers coerce them into plain JSON
values for the CLI's report document. ``serialize_result`` returns the summary dict VERBATIM — the
complete parameter × horizon grid (the ``cells`` panel with one entry per DECLARED combo × horizon,
each with its per-target reliability reads and every evidence panel) is never filtered here or
anywhere downstream; the checklist renders its per-cell results in
a separate section of the report. Every output is a caller-nominated FILE — the report itself
(``--report-out``), the per-observation trades via ``write_trades_csv`` (``--trades-out``), the
per-bar root-series value matrix via ``write_root_series_csv`` (``--root-series-out``), and the
per-bar 0/1 entry-flag matrix via ``write_entry_flags_csv`` (``--entry-flags-out``); nothing
rides stdout.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seikan.analysis.result import EventStudyResult
from seikan.types import SerializedResult

if TYPE_CHECKING:
    import pandas as pd


@contextmanager
def atomic_output(path: str) -> Iterator[str]:
    """Write-then-rename atomicity for every caller-nominated output file.

    Yields a temporary path in the TARGET's own directory (``os.replace`` is atomic only within
    one filesystem); on success the temp file replaces ``path`` in one step, on failure it is
    removed and whatever ``path`` held before — possibly a previous run's good output — survives
    intact. A crash or ENOSPC mid-write can therefore never leave a truncated report or CSV
    wearing the nominated name. Deliberately no fsync: the guarantee bought here is
    no-torn-files, not power-loss durability.
    """
    target = Path(path)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def json_safe(value: object) -> Any:
    """Coerce numpy scalars / pandas Timestamps / NaN / nested containers into JSON-safe values.

    Total by construction: anything left unrecognized degrades to ``repr``. Emission is the LAST
    step of every command, including the error envelopes, so a value the encoder cannot handle
    would raise from inside the handler itself — losing the report, the exit code, and the
    diagnosis at once. A report that says something imperfectly beats no report at all.

    The parameter is ``object`` — nothing whatever is assumed of the input, which is the point —
    while the return is deliberately ``Any``: this coercion is SHAPE-PRESERVING (every key, every
    nesting level and every non-numeric scalar survives it), so what comes back is the caller's
    own declared shape minus its JSON-illegal scalars, and callers put it straight back into the
    typed slot it came from (``serialize_result`` into ``SerializedResult.summary``, the CLI into
    its document sections). A narrower return would name a shape this function cannot see and
    force a cast at every one of those sites — stating less, checking nothing more.
    """
    if isinstance(value, dict):
        return {(k if isinstance(k, str) else str(k)): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    # pandas Timestamp / datetime -> ISO string.
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return iso()
    # numpy scalars expose ``.item()``; floats may be NaN/inf which JSON can't encode.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (ValueError, TypeError):  # a non-scalar ndarray/array-like
            return repr(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return repr(value)


def serialize_result(result: EventStudyResult) -> SerializedResult:
    """JSON-safe view of a :class:`EventStudyResult` — the statistical report ONLY.

    Returns the thesis name and the summary dict: the per-cell ``cells`` panel (one entry per
    DECLARED combo × horizon, non-firing ones included, each carrying its per-target
    ``rot_p``/``t_hac``/``hac_se``/``n_nonoverlap`` reads and ``conditional_buckets``), the
    grid-level CSCV ``pbo`` block, and the run-level geometry/provenance stamps (``n_bars``,
    ``index_start``/``index_end``,
    ``n_hypotheses_attempted``, ``sources``).

    No pooled headline count rides alongside it: a scalar ``n`` would have to be some cell's, and
    naming one cell is the selection this engine does not perform. Per-cell observation counts live
    in ``cells[i].by_target[t].n``; the recorded-row total is the trades frame's length. The
    per-observation ``trades`` frame is never embedded — the only trades channel is
    ``write_trades_csv`` (``--trades-out``).
    """
    return {
        "name": result.thesis.name,
        "summary": json_safe(result.summary),
    }


def write_trades_csv(result: EventStudyResult, path: str) -> int:
    """Write the per-observation trades frame as CSV; returns the row count.

    Every recorded observation across the WHOLE grid — the swept param levels and ``target`` label
    columns identify which cell each row belongs to, and censored rows ride along flagged by
    ``is_open``/``exit_reason``. Nothing is subset to a cell here; the caller regroups on the level
    columns.
    """
    with atomic_output(path) as tmp:
        result.trades.to_csv(tmp, index=False)
    return len(result.trades)


def write_root_series_csv(frame: pd.DataFrame, path: str) -> int:
    """Write the root-series VALUE matrix (``EntryListReport.root_series``) as CSV for
    ``--root-series-out``. No entry flags ride it — a fired bar reaches the caller as a row of the
    trades CSV instead, except a FINAL-bar firing, which anchors no observation and so has no trades
    row; that one rides ``write_entry_flags_csv`` (``--entry-flags-out``), the only CLI output that
    reports firings as such.

    ISO-8601 index labelled ``datetime``, warmup NaN as empty cells, always overwriting; returns
    the row count. The file re-reads through ``dataio.read_strict_csv`` as a series-shaped CSV in
    the ordinary case, but two shapes it can legitimately take do not: a value column that is
    entirely NaN (a transform window longer than the data), and a thesis whose every threshold
    operand is a bare constant, which has no root series at all and writes a datetime-only file with
    zero value columns. The writer gates on neither — the listing is the caller's data, and
    refusing to write it would hide exactly the degenerate thesis shape worth seeing.
    """
    return _write_indexed_csv(frame, path)


def write_entry_flags_csv(frame: pd.DataFrame, path: str) -> int:
    """Write the per-bar 0/1 entry-flag matrix (``EntryListReport.entry_flags``) as CSV for
    ``--entry-flags-out``. One integer column per (param combo × target) — named ``entry`` /
    ``entry[axis=value,...]``, carrying the ``@<target>`` suffix only when several targets run —
    beside the ISO-8601 index labelled ``datetime``. Those names are canonical and unique BY
    CONSTRUCTION (one per declared combo × target), so unlike the root-series labels they need no
    ``#N`` disambiguation. The mask is bit-identical to the backtest's: both come from
    ``vectorize.signal`` over the same compiled entry tree. It lines up BAR FOR BAR against the
    root-series CSV, which carries the identical index — but NOT against the trades rows: those are
    one per (firing bar × target × DECLARED HORIZON), and their ``entry_time`` is the next-open
    ANCHOR bar ``t+1`` while this file is indexed by the FIRING bar ``t``. Join the two on
    ``entry_bar``, the trades column that equals this file's row POSITION; a timestamp join is off
    by one bar.

    This is the ONE CLI output that carries a FINAL-bar firing. That firing anchors no observation —
    there is no ``open[t+1]`` to measure from — so the runner records nothing for it and the trades
    CSV, which is shaped per OBSERVATION, structurally cannot represent it; no coverage ledger
    counts it either. A caller asking "is this thesis firing NOW?" reads it here.

    ISO-8601 index labelled ``datetime``, always overwriting; returns the row count. The file
    ALWAYS re-reads through ``dataio.read_strict_csv`` as a series-shaped CSV — int dtype, no NaN
    (an undecidable bar is a 0, not a hole), and at least one value column, since a thesis declares
    at least one combo and at least one target. Neither degenerate shape ``write_root_series_csv``
    tolerates — an all-NaN column, or a value-column-free file — can arise here.
    """
    return _write_indexed_csv(frame, path)


def _write_indexed_csv(frame: pd.DataFrame, path: str) -> int:
    """The one datetime-indexed CSV write both per-bar listings share (atomic, like every
    output)."""
    with atomic_output(path) as tmp:
        frame.to_csv(tmp, index_label="datetime")
    return len(frame)
