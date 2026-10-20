"""Strict CSV ingestion — the verifier's front door for untrusted external data.

Calling agents bring their own time series; seikan trusts nothing about them. Every file passes
through :func:`read_strict_csv` (one field per header column on every record, strict ISO-8601
timestamps, no format guessing, forced-float value columns, duplicate/order checks) and the
:func:`check_frame` integrity gate (OHLC invariants, positive prices, no inf) before the engine
sees a single number. All violations for a file are accumulated — never fail-fast — and every
check lands in a structured ``data_report`` the CLI emits even on refusal (exit 2), so a calling
agent can fix its data without guesswork. One exception is a structural refusal (undecodable
bytes, an unnamed or duplicate header name, a ragged record, a record the strict scan cannot
read — a field past ``_CSV_FIELD_LIMIT`` characters, usually a quote that never closes — or a
record pandas cannot read or reads differently from the file): past it no cell can be placed
under a column, so the cell-level checks do not run. Likewise a timestamp or value-cell error
stops the read before the frame-level checks (shape, the :func:`check_frame` invariants): the
cells' own errors are all reported, but a frame built from refused cells is not graded further.

Refusals vs warnings: structural defects (bad timestamps, non-numeric garbage, inverted OHLC)
refuse; market-shaped anomalies (interior NaN holes, crash-sized moves, calendar gaps) only warn —
crashes and halts are the research subject, not dirt.
"""

from __future__ import annotations

import codecs
import csv
import hashlib
import io
import itertools
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from dataclasses import field as dc_field
from fractions import Fraction
from typing import Literal, Unpack

import numpy as np
import pandas as pd

from seikan.constants import NEXT_OPEN_OFFSET, OHLC_COLUMNS, PAST_NANOSECOND_FRACTION
from seikan.types import (
    BarSpacing,
    DataIssue,
    DataIssueCode,
    DataIssueExtras,
    DataReport,
    FileReportEntry,
    IssueExample,
    JoinInfo,
)

#: |1-bar close return| beyond this warns of a possible unadjusted split (never refuses).
_LARGE_MOVE = 0.5
#: An index gap larger than this multiple of the median spacing warns (never refuses).
_GAP_FACTOR = 10
#: How many offending rows an error message names before truncating.
MAX_EXAMPLES = 5
#: Fewer than this many possible anchor bars beyond the longest horizon → barely_sufficient warning.
_MIN_COMFORT_ANCHORS = 8
#: The longest field, in characters, the record scan (:func:`_records`) reads — a longer one
#: refuses as ``unreadable``. It is the stdlib's default ``csv.field_size_limit()``, PINNED for
#: :func:`read_strict_csv`'s stages rather than read: that limit is interpreter-global, so a
#: library caller or a co-imported package (scipy's ARFF reader lifts it to the C long's maximum)
#: would otherwise decide whether the same bytes refuse or load.
_CSV_FIELD_LIMIT = 131072
#: Two of the four places pandas' tokenizer can read a record differently from ``csv.reader``
#: (:func:`_records`; a NUL byte and a byte-order mark past the file's start are the others), so
#: its frame is cross-checked: a carriage return that is not half of a ``\r\n``, and a line led by
#: spaces or tabs and then a quote — after a ``\n``: past a lone ``\r`` the check runs anyway, and
#: the first line, which opens the first 256 KiB read buffer, could put its quote at the next one
#: only behind ~262k leading spaces, one field past ``_CSV_FIELD_LIMIT`` the header stage refuses.
_LONE_CR = re.compile(rb"\r(?!\n)")
_SPACED_QUOTE = re.compile(rb'\n[ \t]+"')
#: The value-cell grammar, ASCII only: a signed decimal (``1``, ``1.``, ``.5``, ``-0``) with an
#: optional exponent, or a signed ``inf``/``infinity`` in any case (non-finite, refused later by
#: :func:`check_frame`). It is the set pandas' parser used to admit, and it gates ``float()``,
#: which alone would also read ``1_000``, Arabic-Indic (``١٢٣``) and full-width digits.
_NUMBER = r"[+-]?(?:(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|(?i:inf|infinity))"


class DataError(ValueError):
    """Input data failed strict validation — maps to CLI exit 2. Carries the structured report."""

    def __init__(
        self, report: DataReport, message: str = "input data failed strict validation"
    ) -> None:
        self.report = report
        super().__init__(message)


@dataclass
class FileReport:
    """Per-file result of the strict read: every error and warning, machine-readable."""

    path: str
    role: str = "file"
    shape: Literal["ohlcv", "series"] | None = None
    n_rows: int | None = None
    start: str | None = None
    end: str | None = None
    #: sha256 of the file's raw bytes — the report's data identity (a changed byte is a
    #: different exam). Computed before content validation, so even a refused file carries
    #: its digest; None only when the file itself could not be read.
    sha256: str | None = None
    errors: list[DataIssue] = dc_field(default_factory=list)
    warnings: list[DataIssue] = dc_field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, code: DataIssueCode, message: str, **extra: Unpack[DataIssueExtras]) -> None:
        self.errors.append({"code": code, "message": message, **extra})

    def warn(self, code: DataIssueCode, message: str, **extra: Unpack[DataIssueExtras]) -> None:
        self.warnings.append({"code": code, "message": message, **extra})

    def to_dict(self) -> FileReportEntry:
        return {
            "path": self.path,
            "role": self.role,
            "ok": self.ok,
            "shape": self.shape,
            "n_rows": self.n_rows,
            "start": self.start,
            "end": self.end,
            "sha256": self.sha256,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def build_data_report(
    files: list[FileReport],
    join: JoinInfo | None = None,
    errors: list[DataIssue] | None = None,
) -> DataReport:
    """Aggregate per-file reports (+ optional cross-file join info and top-level errors)."""
    errors = list(errors or [])
    return {
        "ok": not errors and all(f.ok for f in files),
        "files": [f.to_dict() for f in files],
        "join": join,
        "errors": errors,
    }


def check_frame(df: pd.DataFrame, *, shape: str) -> list[str]:
    """Every integrity violation of a frame the strict read built (a sorted, unique, non-empty
    ``DatetimeIndex`` over float value columns) — empty when it is clean.

    ``shape`` is ``"ohlcv"`` or ``"series"``. Interior NaN holes are tolerated (the engine censors
    them as ``no_outcome``); all-NaN columns and non-finite (inf) values are not.
    """
    v: list[str] = []
    for c in df.columns:
        col = df[c].to_numpy(dtype=float)
        if np.isnan(col).all():
            v.append(f"column {c!r} is entirely NaN")
        elif np.isinf(col).any():
            v.append(f"column {c!r} has non-finite (inf) values")

    # --- shape ambiguity ------------------------------------------------------------------
    if shape != "ohlcv":
        # A series read of a frame carrying two or more of the four OHLC names is ambiguous
        # evidence: the bound column would be broadcast into all four price slots while every
        # OHLC invariant below is skipped, so a file whose high sits under its low would load
        # clean — and a full OHLCV file demanded as a series would dodge the same checks. A lone
        # OHLC-named column (a bare `close`) is an ordinary series file.
        present = [c for c in OHLC_COLUMNS if c in df.columns]
        if len(present) >= 2:
            v.append(
                f"series-shaped read of a frame carrying {len(present)} of the four OHLC "
                f"column names ({present}) — ambiguous shape; rename the columns, or provide "
                f"and read the complete open/high/low/close set as OHLCV"
            )

    # --- OHLCV invariants -----------------------------------------------------------------
    if shape == "ohlcv":
        missing = [c for c in OHLC_COLUMNS if c not in df.columns]
        if missing:
            v.append(f"ohlcv frame missing columns {missing}")
        else:
            o, h, lo, c = df["open"], df["high"], df["low"], df["close"]
            if (h < lo).any():
                v.append("ohlcv has bars where high < low")
            if (h < o).any() or (h < c).any():
                v.append("ohlcv has bars where high < open/close")
            if (lo > o).any() or (lo > c).any():
                v.append("ohlcv has bars where low > open/close")
            if (df[list(OHLC_COLUMNS)] <= 0).to_numpy().any():
                v.append("ohlcv has non-positive prices")
            if "volume" in df.columns and (df["volume"] < 0).any():
                v.append("ohlcv has negative volume")
    return v


def _examples(
    mask: pd.Series, lines: list[int], raw: pd.Series | None = None
) -> list[IssueExample]:
    """First offending rows as ``{csv_line, value?}`` records, each row's csv line the physical
    line its record starts on (``lines``, :func:`_record_shape`)."""
    out: list[IssueExample] = []
    for pos in np.flatnonzero(mask.to_numpy())[:MAX_EXAMPLES]:
        rec: IssueExample = {"csv_line": lines[int(pos)]}
        if raw is not None:
            rec["value"] = str(raw.iloc[int(pos)])
        out.append(rec)
    return out


def read_strict_csv(
    path: str, *, role: str = "file", expected_shape: str | None = None
) -> tuple[pd.DataFrame | None, FileReport]:
    """Read one CSV under the strict contract → ``(canonical frame | None, FileReport)``.

    The canonical frame: a tz-naive, sorted, unique ``DatetimeIndex`` named ``datetime`` (from a
    column named ``datetime``, case-insensitive, or else the first column), every value column
    float64, column names lowercased. The contract, each violation a machine-readable error:

    - every record carries exactly the header's field count — any record with more (a trailing
      delimiter is one more) or fewer refuses (``field_count_mismatch``), never read as an index,
      never padded; a quoted delimiter or line break stays inside ONE field; blank lines skip
    - pandas reads exactly the records that scan counted — where that is not established (a bare
      ``\\r`` line ending, a NUL byte, a line led by spaces and then a quote, a byte-order mark
      past the file's start) its frame is cross-checked and a disagreement refuses (``unreadable``)
    - strict **ISO-8601** timestamps (``YYYY-MM-DD`` or full ISO timestamps), timezone-**naive**;
      no ``dayfirst``/format inference, ever; at most nanosecond resolution — a nonzero digit past
      the ninth fractional digit refuses (``bad_timestamp``) rather than being truncated
    - unique, ascending timestamps
    - value cells are ASCII decimal numbers (optional sign, fraction and exponent) or a signed
      ``inf``/``infinity``, each read correctly rounded — the double nearest the decimal written;
      the only missing-value markers are an empty cell or ``nan``
    - every value column named, and uniquely: an empty header name refuses (``unnamed_column`` —
      pandas invents ``Unnamed: N`` for it, a name the file never wrote) save the timestamp
      column's, whose name is never read; a repeated name refuses (``duplicate_column``); no
      empty/header-only file
    - :func:`check_frame` invariants for the detected (or ``expected_shape``) shape

    Warnings (returned, never refusing): per-column NaN fraction, |1-bar close return| >
    ``_LARGE_MOVE`` (possible unadjusted split), calendar gaps > ``_GAP_FACTOR`` × median spacing.

    The report also carries the file's raw-byte ``sha256`` — the run's data identity, computed
    before any validation so even a refused file keeps its digest.

    The file is read ONCE and every stage (digest, header sniff, record-shape scan, parse and its
    cross-check) consumes that one buffer, so the digest describes exactly the bytes the engine
    measured: a rewrite between stages cannot split the reported identity from the evidence
    actually parsed. The whole file is held in memory — pandas materialized it anyway. A load that
    binds one file under several keys reads it through one :class:`StrictCsvReader`, which keeps
    that single read for every binding of the path.

    The record scan reads under ``csv.field_size_limit()``, which is interpreter-global, so stages
    2 to 5 (the scans and the parse between them) run with it pinned to ``_CSV_FIELD_LIMIT`` — the
    stdlib default — and the caller's limit is handed back on every exit, admitted, refused or
    raised through: a field longer than 131072 characters refuses as ``unreadable`` (unless an
    earlier stage refused first — a duplicate header name is decided off the first record alone)
    whatever else in the process set the limit. The pin is a set-and-restore of process state, so
    it is NOT thread-safe: a thread using ``csv`` while a read is inside those stages reads under
    the pin, and a limit it sets meanwhile is overwritten by the restore.
    """
    return StrictCsvReader().read(path, role=role, expected_shape=expected_shape)


@dataclass(frozen=True)
class _Snapshot:
    """Stage 1's result for one path: its bytes and their sha256, or the failure to read them."""

    blob: bytes | None
    sha256: str | None
    errors: tuple[DataIssue, ...]


class StrictCsvReader:
    """Strict reads (:func:`read_strict_csv`) that measure ONE snapshot of each distinct path.

    A load can bind one file under several keys — two targets, a target and the benchmark, every
    member of a per-target feed. Read once per binding, a file rewritten between two of those
    reads was measured as two different files under one path, and the report's data identity
    (one sha256 per path) named only the last: a digest of bytes the other binding never parsed.
    A reader built with the paths a load will read (``paths``, one entry per binding) reads each
    distinct path once — its bytes, or the failure to read them — and every binding of it parses
    those same bytes under its own role and expected shape, so all of them carry the one digest
    by construction. A snapshot is held only until its path's last planned read, so a load holds
    no more file bytes than one read per binding did; a path read beyond its plan (or by a reader
    built with none) is read afresh.
    """

    def __init__(self, paths: Iterable[str] = ()) -> None:
        self._pending = Counter(str(p) for p in paths)  # planned reads still to come, per path
        self._snapshots: dict[str, _Snapshot] = {}

    def _snapshot(self, path: str) -> _Snapshot:
        snap = self._snapshots.pop(path, None)
        if snap is None:
            scratch = FileReport(path=path)
            blob = _read_blob(path, scratch)
            digest = hashlib.sha256(blob).hexdigest() if blob is not None else None
            snap = _Snapshot(blob, digest, tuple(scratch.errors))
        self._pending[path] -= 1
        if self._pending[path] > 0:
            self._snapshots[path] = snap
        return snap

    def read(
        self, path: str, *, role: str = "file", expected_shape: str | None = None
    ) -> tuple[pd.DataFrame | None, FileReport]:
        """:func:`read_strict_csv` over this reader's snapshot of ``path``."""
        rep = FileReport(path=str(path), role=role)
        snap = self._snapshot(rep.path)
        rep.errors.extend(issue.copy() for issue in snap.errors)
        blob = snap.blob
        if blob is None:
            return None, rep
        rep.sha256 = snap.sha256
        ambient = csv.field_size_limit(_CSV_FIELD_LIMIT)  # set, returning the limit it replaces
        try:
            header = _header_columns(path, blob, rep)
            lines = None if header is None else _record_shape(path, blob, len(header[1]), rep)
            if header is None or lines is None:
                return None, rep
            raw = _parse_frame(path, blob, header, rep)
            if raw is None or not _tokenizer_parity(path, blob, raw, rep):
                return None, rep
        finally:  # here, never in the _records generator, whose cleanup waits on garbage collection
            csv.field_size_limit(ambient)
        ts_col = "datetime" if "datetime" in raw.columns else raw.columns[0]
        ts = _parse_timestamps(path, raw, ts_col, lines, rep)
        data = _parse_value_columns(path, raw, ts_col, lines, rep)
        if rep.errors or ts is None:  # rep.errors is the read `rep.ok` derives from
            return None, rep
        df = pd.DataFrame(data, index=pd.DatetimeIndex(ts, name="datetime"))
        detected = _shape_and_integrity(path, df, expected_shape, rep)
        if rep.errors:
            return None, rep
        _descriptive_warnings(path, df, detected, lines, rep)
        return df, rep


def _read_blob(path: str, rep: FileReport) -> bytes | None:
    """Stage 1: the single byte read — one buffer feeds the digest, the header sniff, the
    record-shape scan, the parse and its cross-check, so the reported sha256 describes exactly
    the bytes the engine measured.

    Every ``OSError`` here (a directory, a denied read, a regular file as a parent directory, a
    name past the filesystem's limit) is a fact about the path, never an engine fault: the file
    is unreadable and the batch keeps its report."""
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except FileNotFoundError:
        rep.error("file_missing", f"{path}: file not found")
        return None
    except OSError as exc:
        rep.error("unreadable", f"{path}: {exc}")
        return None


class _RecordError(csv.Error):
    """A ``csv.Error`` (in practice a field past ``_CSV_FIELD_LIMIT``, the limit
    :func:`read_strict_csv` pins) tagged with the csv line its record starts on, so the refusal
    can name where to look."""

    def __init__(self, csv_line: int, exc: csv.Error) -> None:
        self.csv_line = csv_line
        super().__init__(f"the record starting at csv line {csv_line}: {exc}")


def _records(blob: bytes) -> Iterator[tuple[int, list[str]]]:
    """Every record pandas' C tokenizer reads from ``blob``, in order, as ``(csv_line, fields)``.

    ``csv.reader`` over the same decoding (``utf-8-sig``) and line handling (``newline=""``:
    ``\\n``, ``\\r\\n`` and a lone ``\\r`` each end a line, exactly as each ends one for the
    tokenizer), in the dialect :func:`_parse_frame` reads with — the shared default of both: ``,``
    delimits, ``"`` quotes, a doubled ``""`` inside quotes is a literal quote, no escape
    character, no initial-space skipping, and a quote that does not open a field is a literal
    character. A quoted delimiter or line break is therefore ONE field to both. ``csv_line`` is the
    1-based physical line the record STARTS on (a quoted line break spans one record over several).
    A BOM the codec leaves behind (a doubled one) the tokenizer drops itself, and so does this.

    Blank lines drop with ``skip_blank_lines`` parity (pandas' default, which :func:`_parse_frame`
    keeps): the tokenizer drops a line holding nothing but spaces and tabs — before the header as
    well as after it, so the header is the first record that survives — while a quoted blank
    (``""``, ``"  "``) is a record to it. ``csv.reader`` reads a whitespace line and a quoted blank
    alike, as one blank field, so the RAW line decides, never the fields.

    Parity is exact on ``\\n`` and ``\\r\\n`` files without a NUL byte, record for record and field
    for field, but for two seams, both where the tokenizer starts a new 256 KiB read buffer. A
    space-led line it entered as a possible blank backs up only to the start of the current
    buffer, so a record whose leading spaces all fall before the edge loses them. Every later stage
    strips a cell, so that alone reads the same value — but a quote opening the new buffer then
    opens a QUOTED field where this scan (and the tokenizer anywhere else in the file) reads a
    literal quote. And while its first line is still open (a quoted line break does not close it,
    a leading blank line does) the tokenizer drops a BOM that opens a buffer, as it drops the one
    opening the file: a U+FEFF at character 262144 (or a multiple) of an over-long header vanishes
    from that name, so ``\\ufeffclose`` and ``close``, distinct to the duplicate check, reach pandas
    as ``close`` twice: ``close`` loads the other column and the file's own becomes ``close.1``.
    Beyond that the tokenizer has defects of its own: after a blank line ended by a LONE ``\\r``
    it swallows the delimiter that opens the next record (``\\r,5,6`` reads ``5,6``, then pads),
    after any lone ``\\r`` a space-led line can re-read earlier text, and a NUL byte ends its
    field. Wherever a NUL byte, a lone ``\\r``, a space-led quote or a BOM past the file's start
    occurs, :func:`_tokenizer_parity` cross-checks pandas' frame against these records.

    A ``csv.Error`` re-raises as :class:`_RecordError` naming the line its record starts on. The
    reader obeys whatever ``csv.field_size_limit()`` holds while it runs — :func:`read_strict_csv`
    pins it to ``_CSV_FIELD_LIMIT`` around every stage that calls this.
    """
    text = io.TextIOWrapper(io.BytesIO(blob), encoding="utf-8-sig", newline="")
    raw = ""

    def lines() -> Iterator[str]:
        nonlocal raw
        for n, line in enumerate(text):
            raw = line[1:] if n == 0 and line.startswith("\ufeff") else line
            yield raw

    reader = csv.reader(lines())
    start = 1
    try:
        for fields in reader:
            # the reader pulls lines only until its record closes: `raw` is the record's last line
            if reader.line_num > start or raw.strip(" \t\r\n"):
                yield start, fields
            start = reader.line_num + 1
    except csv.Error as exc:
        raise _RecordError(start, exc) from exc


def _refuse_undecodable(path: str, blob: bytes, exc: UnicodeDecodeError, rep: FileReport) -> None:
    """Bytes that are not UTF-8: unreadable, named by their offset in the file and the csv line
    they sit on. The streaming decoder that met them counts its position from the start of its
    8 KiB read chunk (and from past a BOM), so the blob is decoded whole once more, on this
    failure path only: plain ``utf-8`` reads a BOM as U+FEFF, so ``start`` is the file offset."""
    try:
        blob.decode("utf-8")
    except UnicodeDecodeError as whole:
        exc = whole
    head = blob[: exc.start]  # a line ends at \n, \r\n or a lone \r, as for the record scan
    line = head.count(b"\n") + len(_LONE_CR.findall(head)) + 1
    rep.error("unreadable", f"{path}: {exc}", examples=[{"csv_line": line}])


def _refuse_unparsable(path: str, exc: _RecordError, rep: FileReport) -> None:
    """A record ``csv.reader`` cannot read: unreadable, named by the line the record starts on.
    With ``strict`` off the reader closes an unterminated quote at EOF, so a quote that never
    closes swallows the rest of the file — past the field limit, this error."""
    rep.error(
        "unreadable",
        f"{path}: could not be parsed as CSV ({exc}) — usually a quote that never closes, "
        "reading the rest of the file into one field",
        examples=[{"csv_line": exc.csv_line}],
    )


def _header_columns(path: str, blob: bytes, rep: FileReport) -> tuple[int, list[str]] | None:
    """Stage 2: header inspection BEFORE pandas renames a column → ``(csv_line, names)``. An
    empty file, an undecodable one, an unnamed value column or a duplicate name refuses here.
    The header is the first record :func:`_records` yields — the one pandas reads, leading blank
    lines skipped — and a name is read stripped and lower-cased, as every later stage reads it.
    pandas names an EMPTY one ``Unnamed: N`` and a repeated one ``col.1``, names the file never
    wrote, and an invented name can collide with a written one: ``datetime,,unnamed: 1`` loaded
    two columns called ``unnamed: 1`` and crashed the read (exit 4). The one name that may be
    empty is the timestamp column's — the first column, where none is named ``datetime`` (the
    index pandas writes unnamed) — because no stage reads it by name."""
    try:
        first = next(_records(blob), None)
    except UnicodeDecodeError as exc:  # decoding is lazy — it surfaces at the first read
        _refuse_undecodable(path, blob, exc, rep)
        return None
    except _RecordError as exc:
        _refuse_unparsable(path, exc, rep)
        return None
    if first is None:  # no header for pandas either: it said "No columns to parse from file"
        bare = not blob.removeprefix(codecs.BOM_UTF8)
        rep.error(
            "empty_file",
            f"{path}: file is empty" if bare else f"{path}: file holds only blank lines, no header",
        )
        return None
    line, cols = first[0], [str(c).strip().lower() for c in first[1]]
    ts_at = cols.index("datetime") if "datetime" in cols else 0
    unnamed = [i + 1 for i, c in enumerate(cols) if not c and i != ts_at]
    if unnamed:
        rep.error(
            "unnamed_column",
            f"{path}: header field(s) {unnamed} carry no name — every value column must be "
            "named (pandas would invent 'Unnamed: N', a name the file never wrote); only the "
            "timestamp column may go unnamed",
            examples=[{"csv_line": line}],
        )
    dups = sorted({c for c in cols if c and cols.count(c) > 1})
    if dups:
        rep.error(
            "duplicate_column",
            f"{path}: duplicate column name(s) {dups} — every column must be uniquely named",
        )
    return None if unnamed or dups else (line, cols)


def _record_shape(path: str, blob: bytes, n_fields: int, rep: FileReport) -> list[int] | None:
    """Stage 3: every record carries exactly the header's ``n_fields`` fields — more OR fewer
    refuses, every offender counted, before pandas reads a byte. pandas repairs both in silence:
    a short record pads with empty cells (a legal missing marker — a hole the file never had), and
    when EVERY record carries one field too many (a trailing delimiter is one) it reads the first
    field as an implicit index, so each header name reads the field one place to its right — a
    file that loads clean and wrong. ``index_col=False`` is no cure: it warns and drops the extra
    field. Once this stage passes the inference has nothing to trigger on, so
    :func:`_parse_frame` keeps pandas' defaults. A single ragged record pandas does refuse, but as
    an ``unreadable`` tokenizer error naming one line; here every offender is counted under its
    own code and the first ``MAX_EXAMPLES`` are named by line.

    Returns the csv line each data record starts on — row for row the frame's, which
    :func:`_tokenizer_parity` holds pandas to — so a later stage names an offending row by the
    line it sits on, past skipped blank lines and quoted line breaks; None on a refusal.
    """
    n_bad = 0
    counts: set[int] = set()
    examples: list[IssueExample] = []
    lines: list[int] = []
    try:
        records = _records(blob)
        next(records)  # the header stage 2 read
        for line, fields in records:
            lines.append(line)
            if len(fields) != n_fields:
                n_bad += 1
                counts.add(len(fields))
                if len(examples) < MAX_EXAMPLES:
                    examples.append({"csv_line": line, "value": str(len(fields))})
    except UnicodeDecodeError as exc:  # a byte past the header's chunk first decodes here
        _refuse_undecodable(path, blob, exc, rep)
        return None
    except _RecordError as exc:
        _refuse_unparsable(path, exc, rep)
        return None
    if n_bad:
        found = ", ".join(str(n) for n in sorted(counts)[:MAX_EXAMPLES])
        if len(counts) > MAX_EXAMPLES:
            found += ", …"
        rep.error(
            "field_count_mismatch",
            f"{path}: {n_bad} record(s) do not carry the header's {n_fields} field(s) (found "
            f"{found}) — every record needs exactly one field per header column (a trailing "
            "delimiter is one more, empty field)",
            examples=examples,
        )
        return None
    return lines


def _parse_frame(
    path: str, blob: bytes, header: tuple[int, list[str]], rep: FileReport
) -> pd.DataFrame | None:
    """Stage 4: the full read, everything as strings — no type guessing anywhere — under the
    header stage 2 read (``header``, its csv line and names). pandas must read every name that
    scan did, or the frame places cells under names the file does not give them: it refuses as
    ``unreadable``. The tokenizer can part from the scan on a header — a byte-order mark at its
    256 KiB read-buffer edge drops out of an over-long one, turning two distinct names into
    ``close`` and ``close.1``. The frame then carries the scan's names, so an unnamed timestamp
    column keeps its empty name rather than pandas' ``Unnamed: 0``."""
    line, cols = header
    try:
        raw = pd.read_csv(io.BytesIO(blob), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as exc:
        rep.error("unreadable", f"{path}: could not be parsed as CSV ({exc})")
        return None
    read = [str(c).strip().lower() for c in raw.columns]
    if len(read) != len(cols) or any(c and r != c for r, c in zip(read, cols, strict=True)):
        rep.error(
            "unreadable",
            f"{path}: pandas' CSV tokenizer read the header differently from the strict record "
            "scan — it cuts a name at a NUL byte, and at its 256 KiB read-buffer edge can drop a "
            "byte-order mark from an over-long header; re-save the file with no NUL bytes and no "
            "byte-order mark past its start",
            examples=[{"csv_line": line}],
        )
        return None
    raw.columns = cols
    if len(raw) == 0:
        rep.error("header_only", f"{path}: has a header but no data rows")
        return None
    return raw


def _tokenizer_parity(path: str, blob: bytes, raw: pd.DataFrame, rep: FileReport) -> bool:
    """Stage 5: pandas read exactly the records stage 3 counted. Where :func:`_records` cannot
    establish that — a NUL byte, a lone ``\\r``, a space-led quote, a BOM past the file's start —
    the tokenizer can part from a scan that passed: a record behind a lone-``\\r`` blank line
    loses its opening empty field and pads its end, so each value loads under the column to its
    left, clean and wrong; a quote at its 256 KiB buffer edge loads ``   "2021-01-04"`` as a clean
    timestamp that the same line anywhere else refuses; a BOM at that edge of an over-long header
    drops out of a name, so two names the duplicate check saw as distinct load as ``close`` and
    ``close.1`` (stage 4 refuses that header itself). There the frame's records are compared with
    the scan's cell by cell, each cell stripped as every later stage reads it, and a disagreement
    refuses as ``unreadable``, naming the csv line where the two readings part. Every other file
    is not re-read.
    """
    if not (
        b"\x00" in blob
        or _LONE_CR.search(blob)
        or _SPACED_QUOTE.search(blob)
        or blob.find(codecs.BOM_UTF8, 1) >= 0  # any BOM but the leading one the codec drops
    ):
        return True
    records = _records(blob)
    line, _header = next(records)  # stage 4 held pandas' header to it
    agree = isinstance(raw.index, pd.RangeIndex)
    rows = raw.itertuples(index=False, name=None)
    if agree:
        for at, fields in records:
            line, row = at, next(rows, None)
            if row is None or [f.strip() for f in fields] != [str(v).strip() for v in row]:
                agree = False
                break
        else:
            agree = next(rows, None) is None  # else pandas read a row the file does not hold
    if not agree:
        rep.error(
            "unreadable",
            f"{path}: pandas' CSV tokenizer read csv line {line} differently from the strict "
            "record scan — it misreads a record after a bare carriage-return line ending, cuts a "
            "field at a NUL byte, and at its 256 KiB read-buffer edge can open a quote that "
            "follows a line's leading spaces; re-save the file with \\n or \\r\\n line "
            "endings, no NUL bytes and no quote after leading spaces",
            examples=[{"csv_line": line}],
        )
    return agree


def _parse_timestamps(
    path: str, raw: pd.DataFrame, ts_col: str, lines: list[int], rep: FileReport
) -> pd.Series | None:
    """Stage 6: strict ISO-8601, tz-naive, at most nanosecond resolution, unique, ascending —
    violations accumulate as machine-readable errors, and nothing is ever format-guessed."""
    ts_raw = raw[ts_col].str.strip()
    ts: pd.Series | None
    try:
        ts = pd.to_datetime(ts_raw, format="ISO8601", errors="coerce")
    except (ValueError, TypeError) as exc:  # e.g. mixed timezone offsets
        ts = None
        rep.error(
            "bad_timestamp",
            f"{path}: timestamp column {ts_col!r} failed strict ISO-8601 parsing ({exc})",
        )
    if ts is not None:
        bad = ts.isna()
        # a stamp past the nanosecond would read as another instant — `…T00:00:00.0000000001` as
        # midnight, a duplicate of the bar stamped there (`constants.PAST_NANOSECOND_FRACTION`)
        past_ns = ~bad & ts_raw.str.contains(PAST_NANOSECOND_FRACTION, regex=True, na=False)
        if bad.any():
            rep.error(
                "bad_timestamp",
                f"{path}: {int(bad.sum())} timestamp(s) in column {ts_col!r} are not "
                "ISO-8601 (expected YYYY-MM-DD or an ISO timestamp; no other format is accepted)",
                examples=_examples(bad, lines, ts_raw),
            )
            ts = None
        elif getattr(ts.dt, "tz", None) is not None:
            rep.error(
                "tz_aware_timestamp",
                f"{path}: timestamps carry a timezone offset — provide timezone-naive "
                "ISO-8601 timestamps (drop the offset at export time)",
            )
            ts = None
        if past_ns.any():
            rep.error(
                "bad_timestamp",
                f"{path}: {int(past_ns.sum())} timestamp(s) in column {ts_col!r} carry "
                "fractional seconds past the nanosecond — the clock is nanosecond resolution "
                "and the rest would be truncated, reading another instant "
                "('…00:00:00.0000000001' as midnight); round them at export (zeros past the "
                "ninth digit are admitted)",
                examples=_examples(past_ns, lines, ts_raw),
            )
            ts = None
    if ts is not None:
        dup = ts.duplicated()
        if dup.any():
            rep.error(
                "duplicate_timestamp",
                f"{path}: {int(dup.sum())} duplicate timestamp(s)",
                examples=_examples(dup, lines, ts_raw),
            )
        unsorted = ts < ts.shift(1)
        if unsorted.any():
            rep.error(
                "unsorted_timestamp",
                f"{path}: timestamps are not sorted ascending",
                examples=_examples(unsorted, lines, ts_raw),
            )
    return ts


def _parse_value_columns(
    path: str, raw: pd.DataFrame, ts_col: str, lines: list[int], rep: FileReport
) -> dict[str, np.ndarray]:
    """Stage 7: forced-float value columns; an empty cell and ``nan`` are the ONLY missing
    markers — anything else outside the numeric grammar (:data:`_NUMBER`) refuses, named per
    column.

    A number is read CORRECTLY ROUNDED — the double nearest the decimal the file writes, which
    is what ``float()`` returns. ``pd.to_numeric`` is not: it read ``0.00000000000000001`` as
    ``0.0`` and missed by an ulp on nearly half of all 17-significant-digit strings, the form
    ``repr`` and every pandas-written CSV use, so a file read back did not hold the values that
    were written to it."""
    value_cols = [c for c in raw.columns if c != ts_col]
    if not value_cols:
        rep.error("no_value_columns", f"{path}: no value columns besides the timestamp")
    data: dict[str, np.ndarray] = {}
    for c in value_cols:
        s = raw[c].str.strip()
        is_number = s.str.fullmatch(_NUMBER).to_numpy(dtype=bool)
        is_missing = ((s == "") | (s.str.lower() == "nan")).to_numpy(dtype=bool)
        bad = ~(is_number | is_missing)
        if bad.any():
            rep.error(
                "non_numeric_value",
                f"{path}: column {c!r} has {int(bad.sum())} non-numeric value(s)",
                column=c,
                examples=_examples(pd.Series(bad), lines, s),
            )
        vals = np.full(len(s), np.nan)  # positional values; the frame gets the ts index
        cells = s.to_numpy(dtype=object)[is_number]
        vals[is_number] = [float(x) for x in cells]
        data[c] = vals
    return data


def _shape_and_integrity(
    path: str, df: pd.DataFrame, expected_shape: str | None, rep: FileReport
) -> Literal["ohlcv", "series"]:
    """Stage 8: shape detection/mismatch + the :func:`check_frame` integrity invariants.
    Returns the DETECTED shape (also stamped on the report), whatever was expected."""
    detected: Literal["ohlcv", "series"] = (
        "ohlcv" if all(c in df.columns for c in OHLC_COLUMNS) else "series"
    )
    if expected_shape is not None and detected != expected_shape:
        rep.error(
            "shape_mismatch",
            f"{path}: expected a {expected_shape}-shaped file but found {detected} "
            f"(columns: {list(df.columns)})",
        )
    shape = expected_shape or detected
    rep.shape = detected
    for violation in check_frame(df, shape=shape):
        rep.error("integrity", f"{path}: {violation}")
    return detected


def _descriptive_warnings(
    path: str, df: pd.DataFrame, detected: str, lines: list[int], rep: FileReport
) -> None:
    """Stage 9: descriptive facts + admit-with-warning anomalies (NaN fractions, crash-sized
    moves, calendar gaps) — market-shaped, the research subject, never dirt."""
    rep.n_rows = len(df)
    rep.start = df.index[0].isoformat()
    rep.end = df.index[-1].isoformat()
    for c in df.columns:
        frac = float(df[c].isna().mean())
        if frac > 0:
            rep.warn(
                "nan_fraction",
                f"{path}: column {c!r} is {frac:.1%} NaN (holes censor observations)",
                column=c,
                value=round(frac, 4),
            )
    if detected == "ohlcv" and len(df) > 1:
        # ffill + fill_method=None == pct_change's deprecated pad default: a bar after a hole is
        # still measured against the last pre-hole close (the hole itself stays NaN → no flag).
        move = df["close"].ffill().pct_change(fill_method=None).abs()
        big = move > _LARGE_MOVE
        if big.any():
            rep.warn(
                "large_move",
                f"{path}: {int(big.sum())} bar(s) move more than {_LARGE_MOVE:.0%} close-to-close "
                "— verify the series is split/dividend-adjusted",
                examples=_examples(big, lines, df["close"].astype(str)),
            )
    if len(df) > 2:
        # Exact tick arithmetic, as `bar_spacing` reads the clock: a Timedelta median times the
        # factor overflowed int64 nanoseconds past ~29 years of spacing and crashed the read
        # (three nanosecond-stamped rows 200 years apart), where every spacing is near the median.
        spacings = _tick_spacings(df.index)
        median = _median_ticks(spacings)
        if median > 0:
            n_gaps = sum(1 for d in spacings if d > _GAP_FACTOR * median)
            if n_gaps:
                seconds = _seconds(median, ticks_per_second(df.index))
                rep.warn(
                    "calendar_gap",
                    f"{path}: {n_gaps} gap(s) larger than {_GAP_FACTOR}× the median bar spacing "
                    f"({seconds} s)",
                )


#: Ticks per second of each resolution a ``DatetimeIndex`` carries (pandas ≥ 2 keeps the parsed
#: one: a second-stamped file builds a ``[us]`` index, a nanosecond-stamped one ``[ns]``).
_TICKS_PER_SECOND = {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}


def ticks_per_second(index: pd.DatetimeIndex) -> int:
    """How many of the index's own ticks (``asi8``) make one second — its resolution."""
    return _TICKS_PER_SECOND[index.unit]


def _tick_spacings(index: pd.DatetimeIndex) -> list[int]:
    """The spacings between consecutive bars in the index's OWN ticks, as Python integers in index
    order — exact at any span (an int64-nanosecond difference overflows past ~292 years)."""
    ticks: list[int] = index.asi8.tolist()
    return [b - a for a, b in itertools.pairwise(ticks)]


def _median_ticks(spacings: list[int]) -> Fraction:
    """The exact median of tick spacings — an even count's is the midpoint of its two middle
    ones (non-empty input)."""
    d = sorted(spacings)
    mid = len(d) // 2
    return Fraction(d[mid]) if len(d) % 2 else Fraction(d[mid - 1] + d[mid], 2)


def _seconds(ticks: Fraction, per_second: int) -> int | float:
    """A tick count in seconds: an integer where it is a whole second, else the exact quotient
    correctly rounded to a float."""
    q = ticks / per_second
    return q.numerator if q.denominator == 1 else float(q)


def bar_spacing(index: pd.DatetimeIndex) -> BarSpacing:
    """{min,median,max}_seconds between consecutive bars of an index — the clock geometry every
    horizon-in-bars is denominated in, stamped by the run summary and by every ``describe``
    profile in one vocabulary. Pure self-description: integers where the spacing is a whole
    second, floats otherwise (the exact spacing, correctly rounded), None below two bars. The
    engine still never interprets cadence — translation into "trading days" stays the caller's.

    Exact in the index's OWN resolution: its ticks are differenced as Python integers, and the
    median of an even count is the exact midpoint of its two middle spacings. A cast to int64
    nanoseconds overflowed wherever two consecutive bars lie more than ~292 years apart, which
    an ``[s]``/``[us]`` index holds without complaint — a 1700/2024 pair read −8.2e9 s."""
    if len(index) < 2:
        return {"min_seconds": None, "median_seconds": None, "max_seconds": None}
    per_second = ticks_per_second(index)
    d = _tick_spacings(index)
    return {
        "min_seconds": _seconds(Fraction(min(d)), per_second),
        "median_seconds": _seconds(_median_ticks(d), per_second),
        "max_seconds": _seconds(Fraction(max(d)), per_second),
    }


def sufficiency_check(
    n_common: int, horizons: int | list[int]
) -> tuple[list[DataIssue], list[DataIssue]]:
    """Structural sufficiency of the joined index vs the thesis horizon → ``(errors, warnings)``.

    A closed observation at the longest horizon ``h`` needs ``h + NEXT_OPEN_OFFSET + 1`` bars
    (the firing bar, the next-open fill, ``h`` bars to the exit print); below that no
    observation can ever close and the run is refused up front rather than producing an empty,
    trivially-uninformative summary. A barely-sufficient index only warns.
    """
    hs = horizons if isinstance(horizons, (list, tuple)) else [horizons]
    h_max = max(int(h) for h in hs)
    need = h_max + NEXT_OPEN_OFFSET + 1
    anchorable = n_common - (h_max + NEXT_OPEN_OFFSET)
    errors: list[DataIssue] = []
    warnings: list[DataIssue] = []
    if n_common < need:
        errors.append(
            {
                "code": "insufficient_common_index",
                "message": (
                    f"joined common index has {n_common} bars; the longest horizon ({h_max}) "
                    f"needs at least {need} (horizon + {NEXT_OPEN_OFFSET} + 1) for a single "
                    "closed observation"
                ),
            }
        )
    elif anchorable < _MIN_COMFORT_ANCHORS:
        warnings.append(
            {
                "code": "barely_sufficient",
                "message": (
                    f"only {anchorable} bars can anchor a closed observation at the "
                    f"longest horizon ({h_max}) — statistics will be extremely thin"
                ),
            }
        )
    return errors, warnings
