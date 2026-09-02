"""Emitted-shape declarations: the strict-CSV data report."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

# ---- the strict-CSV data report -----------------------------------------------------------
#
# ``dataio``'s front-door verdict on everything a command read: per-file results, the cross-file
# join, and any top-level error. Carried on ``MarketData.report`` and on both result types,
# emitted by ``run`` / ``check-data`` / ``describe`` alike — and by the exit-2 error envelope,
# which is the whole point of accumulating violations instead of failing fast.


#: Every machine-readable issue code the strict loader, the join, the sufficiency guard and the
#: API pre-flight can emit — a closed vocabulary, so a typo'd code refuses at emission.
type DataIssueCode = Literal[
    "bad_timestamp",
    "barely_sufficient",
    "benchmark_coverage",
    "calendar_gap",
    "duplicate_column",
    "duplicate_timestamp",
    "empty_file",
    "external_coverage",
    "file_missing",
    "header_only",
    "insufficient_common_index",
    "integrity",
    "large_move",
    "missing_volume",
    "mixed_target_shapes",
    "nan_fraction",
    "no_value_columns",
    "non_numeric_value",
    "shape_mismatch",
    "spec_data_mismatch",
    "target_index_mismatch",
    "tz_aware_timestamp",
    "unreadable",
    "unsorted_timestamp",
]


class IssueExample(TypedDict):
    """One offending row named by a ``dataio`` error — the CSV line (row index + 2, for the
    header) and, where the check has one, the raw cell value as read."""

    csv_line: int
    value: NotRequired[str]


class DataIssue(TypedDict):
    """One error or warning record — produced by ``dataio.FileReport.error``/``warn``,
    ``dataio.sufficiency_check``, ``compiler.data``'s join checks and ``api``'s pre-flight
    guards; read by the CLI (which emits them) and by ``describe``'s refusal stubs (which name
    the codes).

    ``code`` is the machine-readable class, ``message`` the human sentence. The three optional
    fields are the per-check extras: the offending rows, the column a column-scoped check names,
    and the measured value a threshold-flavored warning reports.
    """

    code: DataIssueCode
    message: str
    examples: NotRequired[list[IssueExample]]
    column: NotRequired[str]
    value: NotRequired[float]


class DataIssueExtras(TypedDict):
    """The keyword tail of :class:`DataIssue` on its own — the ``**extra`` that
    ``dataio.FileReport.error``/``warn`` collect beside the ``code``/``message`` every record
    carries. Same three keys with the same meanings; declared separately so those two methods can
    state which extras a check may attach (``Unpack``) instead of accepting anything at all."""

    examples: NotRequired[list[IssueExample]]
    column: NotRequired[str]
    value: NotRequired[float]


class FileReportEntry(TypedDict):
    """One file's strict-read result — ``dataio.FileReport.to_dict``.

    ``ok`` is exactly "no errors"; ``sha256`` is the raw-byte digest computed BEFORE validation
    (so even a refused file carries its identity) and is None only when the file could not be
    read at all. The descriptive fields stay None for a file that never reached them.
    """

    path: str
    role: str
    ok: bool
    shape: Literal["ohlcv", "series"] | None
    n_rows: int | None
    start: str | None
    end: str | None
    sha256: str | None
    errors: list[DataIssue]
    warnings: list[DataIssue]


class JoinInfo(TypedDict):
    """The cross-file join — built by ``compiler.data.load_market_data``, read by the CLI report
    and appended to by ``api._ensure_sufficient`` (a barely-sufficient index lands here as a
    warning rather than refusing the run).

    ``n_common`` is the joined index length every geometry stamp derives from; the warnings are
    coverage facts (a sparse external feed, an incompletely covering benchmark), never refusals.
    """

    n_common: int
    start: str
    end: str
    warnings: list[DataIssue]


class DataReport(TypedDict):
    """Everything a command read and checked — ``dataio.build_data_report``.

    Carried on ``MarketData.report`` and on both result types, and emitted by every command that
    touches a file (``run``, ``check-data``, ``describe``) INCLUDING on refusal, where it is the
    exit-2 envelope's payload. ``ok`` is the conjunction of "no top-level error" and every file's
    own ``ok``; ``join`` is None for a read that joined nothing (``check-data``, ``describe``).
    """

    ok: bool
    files: list[FileReportEntry]
    join: JoinInfo | None
    errors: list[DataIssue]
