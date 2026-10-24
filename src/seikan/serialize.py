"""JSON-safe serialization of engine results + the CLI's caller-nominated CSV writers.

``compile_thesis`` returns pandas/numpy-typed objects; ``json_safe`` coerces them into plain JSON
values, once per CLI document at serialization. ``serialize_result`` is the library's JSON-safe
view of a result, the summary dict VERBATIM — the complete parameter × horizon grid (the
``cells`` panel with one entry per DECLARED combo × horizon, each with its per-target
reliability reads and every evidence panel) is never filtered here or anywhere downstream; the
checklist renders its per-cell results in a separate section of the report. Every output is a
caller-nominated FILE — the report itself (``--report-out``), the per-observation trades via
``write_trades_csv`` (``--trades-out``), and
the per-bar root-series value and 0/1 entry-flag matrices via ``write_indexed_csv``
(``--root-series-out``, ``--entry-flags-out``); nothing rides stdout. The writers write the
path they are handed; the CLI hands them the temporaries of ``staged_outputs``, which publishes a
run's files together once the last of them is written.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seikan.analysis.result import EventStudyResult
from seikan.types import SerializedResult

if TYPE_CHECKING:
    import pandas as pd


@contextmanager
def staged_outputs() -> Iterator[Callable[[str], str]]:
    """Stage-then-publish for one run's caller-nominated output files.

    Yields ``stage(path)``, which returns a temporary path in the TARGET's own directory
    (``os.replace`` is atomic only within one filesystem) for the caller to write that output to.
    The temporary takes a fixed-length name of its own, never the target's extended, so a target
    named at the filesystem's length limit stages like any other.
    The target is ``path`` with its symlinks resolved, so a link is written through, never
    replaced: the file it names takes the output (created where the link dangles) and the link
    stays — the file the CLI preflight's collision checks, which follow links, compared. Nothing
    is published until the block exits cleanly; then every staged file replaces its target by one
    ``os.replace``, in staging order. A failure before that — a write error, ENOSPC, a report
    refusing its contract validation — removes every temporary and leaves every target as it
    was, so a previous run's outputs survive together and no crash mid-write leaves a truncated
    file wearing a nominated name. Publication is a sequence of renames, not one transaction: a
    rename that fails leaves the files staged before it published and the rest, the report
    (staged last) among them, as they were. Deliberately no fsync: the guarantee bought here is
    no-torn-files, not power-loss durability.
    """
    staged: list[tuple[str, Path]] = []

    def stage(path: str) -> str:
        target = Path(path).resolve()
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix="seikan-", suffix=".tmp")
        os.close(fd)
        staged.append((tmp, target))
        return tmp

    try:
        yield stage
        for tmp, target in staged:
            os.replace(tmp, target)
    finally:
        for tmp, _ in staged:
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
    result.trades.to_csv(path, index=False)
    return len(result.trades)


def write_indexed_csv(frame: pd.DataFrame, path: str) -> int:
    """Write a per-bar listing (``EntryListReport.root_series`` for ``--root-series-out``,
    ``EntryListReport.entry_flags`` for ``--entry-flags-out``) as CSV: the ISO-8601 bar index
    labelled ``datetime``, warmup NaN as empty cells, always overwriting; returns the row count.
    The files' contracts are ``contract.ROOT_SERIES_CSV`` and ``contract.ENTRY_FLAGS_CSV``."""
    frame.to_csv(path, index_label="datetime")
    return len(frame)
