"""Load market data: strict CSV file(s) → aligned OHLCV DataFrames + external feeds for the
backtest core.

A load takes TWO inputs, because the DSL names its series without locating them: the
:class:`~seikan.dsl.schema.DataSpec` (which series, what each one MEANS to the thesis) and a
:class:`DataFiles` (which CSV each named series is, and which column of it, both supplied at
invocation). :func:`resolve_data_files` builds the second from a flat ``{key: path}`` mapping plus
an optional ``{key: column}`` one, and is the only thing that checks the two agree.

Each target (one CSV file) becomes a column; all targets are aligned on their common DatetimeIndex
(inner join) so a single vectorized backtest can run them together. External (alternative-data)
feeds are forward-filled onto the bar index — the invocation names the CSV **column** each key
reads (``--column KEY=COL``, needed only when the file holds several numeric columns), so one file
can hold several series. A shared (single-path) feed
is exposed as a ``Series`` broadcast to every target column at use, a per-target feed as a
(rows × targets) ``DataFrame``; an optional publication lag shifts a feed's timestamps before the
asof anchor so values aren't usable before their release.

Every referenced file is read through :func:`seikan.dataio.read_strict_csv` — the strict contract
for untrusted external data (ISO-8601 naive timestamps, forced-float columns, OHLC invariants; see
that module). ALL files are read and checked before anything else happens, so a single
:class:`~seikan.dataio.DataError` carries the complete per-file diagnosis; the loader never mutates
input (there is deliberately no high/low clamping — inconsistent OHLC refuses instead).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
import pandas as pd

from seikan.dataio import (
    MAX_EXAMPLES,
    DataError,
    FileReport,
    build_data_report,
    read_strict_csv,
)
from seikan.dsl.schema import BENCHMARK_KEY, DataSpec, Thesis
from seikan.types import DataIssue, DataIssueCode, DataReport, JoinInfo, TargetShape

_REQUIRED = ("open", "high", "low", "close")

#: An external feed covering less than this fraction of the joined bars draws a coverage warning.
_FEED_COVERAGE_WARN = 0.9


@dataclass
class MarketData:
    """Aligned OHLCV across targets (each field a DataFrame with ``targets`` columns)."""

    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame | None
    # Shared feeds are a ``Series`` (broadcast to every target column at use); per-target feeds
    # are a (rows × targets) ``DataFrame``.
    externals: dict[str, pd.Series | pd.DataFrame]
    targets: list[str]
    # Benchmark open prices on the joined index (the ``benchmark`` data key), for excess-return
    # measurement — sampled at exactly the observation anchor bars, NOT asof/ffilled like an
    # external feed (a bar the benchmark is missing yields NaN and censors the observation).
    benchmark_open: pd.Series | None = None
    # The file that answered the ``benchmark`` key, retained purely so the summary can keep
    # self-describing WHICH series every excess return was measured against. It rides the loaded
    # data rather than the DSL because the DSL knows names, never where anything lives.
    benchmark_path: str | None = None
    # Per-feed AGE series (calendar days since the feed's most recent native post-lag stamp at or
    # before each bar; NaN before the first stamp) — the ``days_since`` event-distance primitive.
    # Same shared-Series / per-target-DataFrame shape convention as ``externals``.
    externals_age: dict[str, pd.Series | pd.DataFrame] | None = None
    # "ohlcv" (price targets) or "series" (a single value column synthesized into OHLC; no
    # volume) — uniform across targets, enforced at load. Recorded on the summary so evidence
    # self-describes what was measured.
    target_shape: TargetShape = "ohlcv"
    # The structured data_report for everything this load read and checked (per-file results +
    # join info + coverage warnings) — surfaced verbatim in the CLI report.
    report: DataReport | None = None
    # The evaluation memos for ``vectorize.build_series`` / ``build_condition`` /
    # ``condition_arrays``, each keyed by the node's canonical JSON — one typed dict per channel
    # shape, so a hit needs no cast: ``series_memo`` holds a Series node's (value, init) frames,
    # ``condition_memo`` a Condition's (value, init, defined) frames, ``array_memo`` those three
    # channels as arrays. Riding the materialized data scopes the memos to exactly the bars they
    # derive from — ONE memo per CLI run, populated by whichever compute path builds first (the
    # backtest) and reused by the other (the listing). Cached frames are immutable by contract;
    # in-place per-run state follows the same doctrine as ``report``. The memos live as long as
    # this MarketData, so a long-lived instance reused across many theses accumulates entries —
    # ``clear_memo()`` is the caller's release valve.
    series_memo: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = dataclasses.field(
        default_factory=dict, repr=False, compare=False
    )
    condition_memo: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = dataclasses.field(
        default_factory=dict, repr=False, compare=False
    )
    array_memo: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = dataclasses.field(
        default_factory=dict, repr=False, compare=False
    )

    def clear_memo(self) -> None:
        """Release every memoized series/condition build — the valve for a long-lived instance."""
        self.series_memo.clear()
        self.condition_memo.clear()
        self.array_memo.clear()

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    def field(self, column: str) -> pd.DataFrame:
        """OHLCV field as a (rows × targets) DataFrame; ``volume`` requires a volume column. The
        public API (``api._ensure_volume_available``) preflights a volume read on volumeless data
        into a DataError (CLI exit 2); this ValueError is only the library-boundary backstop."""
        if column == "volume":
            if self.volume is None:
                raise ValueError("field 'volume' requires a volume column but data has none")
            return self.volume
        return getattr(self, column)

    def _feed_as_array(self, feeds: dict[str, pd.Series | pd.DataFrame], name: str) -> np.ndarray:
        """One feed dict entry as a (rows × targets) float array — a per-target DataFrame reads
        its target columns, a shared Series broadcasts to every target."""
        if name not in feeds:
            raise ValueError(f"external feed {name!r} not declared in DataSpec.external")
        feed = feeds[name]
        if isinstance(feed, pd.DataFrame):
            values: np.ndarray = feed[self.targets].to_numpy(dtype=float)
            return values
        return np.repeat(feed.to_numpy(dtype=float).reshape(-1, 1), len(self.targets), axis=1)

    def external_values(self, name: str) -> np.ndarray:
        """External feed as a (rows × targets) float array (a shared Series broadcasts)."""
        return self._feed_as_array(self.externals, name)

    def days_since_values(self, name: str) -> np.ndarray:
        """Feed age (days since the last native stamp) as a (rows × targets) float array."""
        return self._feed_as_array(self.externals_age or {}, name)


def _slice(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Inclusive [start, end] window by explicit Timestamp bounds.

    Deliberately NOT ``df.loc[start:end]``: label slicing resolves a partial spelling like
    "2024-01" to the whole month, so the evaluated interval depended on how the bound was
    SPELLED rather than on what it says. ``DataSpec`` has already enforced strict ISO-8601, and
    an explicit comparison makes the boundary the exact instant given — the interval a report
    certifies is then readable straight off the DSL.
    """
    if start is None and end is None:
        return df
    idx = df.index
    keep = np.ones(len(idx), dtype=bool)
    if start is not None:
        keep &= idx >= pd.Timestamp(start)
    if end is not None:
        keep &= idx <= pd.Timestamp(end)
    return df[keep]


def _select_value_column(
    df: pd.DataFrame,
    column: str | None,
    *,
    on_missing_column: Callable[[str, list[str]], str],
    on_ambiguous: Callable[[list[str]], str],
) -> pd.Series:
    """The one bound-or-sole-numeric column selection targets and feeds share.

    ``column`` (already lowercased at parse) picks the series when bound; unbound, the file must
    hold exactly one numeric column. The refusal MESSAGES stay each caller's own — they ride the
    exit-2 ``data_report`` and name the exact ``--column KEY=COL`` pair to fix — so they arrive
    as callables and this helper cannot reword them.
    """
    if column is not None:
        col = column.lower()
        if col not in df.columns:
            raise ValueError(on_missing_column(col, list(df.columns)))
        return df[col]
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) != 1:
        raise ValueError(on_ambiguous(numeric))
    return df[numeric[0]]


def _prepare_target(
    df: pd.DataFrame,
    path: str,
    start: str | None,
    end: str | None,
    key: str,
    column: str | None,
) -> tuple[pd.DataFrame, TargetShape]:
    """Canonical target frame → ``(frame, shape)``. A full-OHLCV file is a price target
    (``shape="ohlcv"``). A file WITHOUT the OHLC columns is a SERIES target (``shape="series"`` —
    a yield, spread, multiple, strategy index): its one value column (the one ``--column
    <key>=<col>`` bound, or the sole numeric column) is synthesized into open=high=low=close=value
    so the close-reading algebra and the next-bar anchor apply unchanged; no volume. A binding on
    an OHLCV file is rejected (ambiguous intent). ``key`` is the data key this file answers, so
    every refusal below names the flag pair a caller would have to fix rather than a file they
    might have bound under several keys. The frame arrives pre-validated by the strict reader —
    nothing here mutates values (no clamping)."""
    df = _slice(df, start, end)
    if all(c in df.columns for c in _REQUIRED):
        if column is not None:
            raise ValueError(
                f"{path}: --column {key}={column} is bound but the file is OHLCV-shaped; "
                f"drop the binding (a price target always measures its open-anchored prices, so "
                f"there is no column for it to choose)"
            )
        return df, "ohlcv"
    s = _select_value_column(
        df,
        column,
        on_missing_column=lambda col, cols: (
            f"{path}: --column {key}={column} is bound but the file has no column "
            f"{col!r}; got {cols}"
        ),
        on_ambiguous=lambda numeric: (
            f"{path}: not OHLCV-shaped and has {len(numeric)} numeric columns {numeric}; "
            f"bind --column {key}=COL to choose the series"
        ),
    )
    value = s.astype(float)
    return (
        pd.DataFrame({"open": value, "high": value, "low": value, "close": value}, index=df.index),
        "series",
    )


def _prepare_feed(
    df: pd.DataFrame,
    path: str,
    key: str,
    lag: pd.Timedelta | None = None,
    column: str | None = None,
) -> pd.Series:
    """Normalize an external feed to its NATIVE (post-lag availability) stamps.

    Returns the feed on its own timestamps — lag-shifted — ready either to asof-anchor onto the
    bars (reindex-ffill) or to derive the per-bar feed age (``_feed_age_days``, the ``days_since``
    primitive). The strict reader already guarantees naive, unique, sorted timestamps.

    The series is read from ``column`` (case-insensitive), which the invocation bound to ``key``
    with ``--column KEY=COL``. When nothing was bound the file must hold exactly one numeric
    column — so single-column feeds need no binding, but a multi-column feed must say which column
    it means. ``key`` itself is used only to write refusals a caller can act on: it is the data key
    the offending file answers, which is what a ``--column`` pair is typed against.

    As-of convention (observer purity): a feed value becomes usable at the FIRST bar whose stamp is
    >= the feed timestamp + ``lag``. The feed timestamp is therefore treated as the *availability*
    time. If a source stamps values at the period they DESCRIBE rather than when they were published
    (e.g. a daily aggregate stamped at that day's midnight, a monthly figure stamped at month-end),
    set ``lag`` to the publication delay — otherwise the value is visible one bar (or more) early.
    """
    s = _select_value_column(
        df,
        column,
        on_missing_column=lambda col, cols: (
            f"external feed {path!r}: --column {key}={column} is bound but the file has no "
            f"column {col!r}; got {cols}"
        ),
        on_ambiguous=lambda numeric: (
            f"external feed {path!r} has {len(numeric)} numeric columns {numeric}; "
            f"bind --column {key}=COL to choose one"
        ),
    )
    # Publication lag: shift timestamps to availability time before anchoring, so a value is not
    # usable on bars before its real release.
    if lag is not None and lag > pd.Timedelta(0):
        s = s.copy()
        s.index = s.index + lag
    return s


def _anchor_feed(native: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Anchor a prepared feed on the bar index: latest value at-or-before each bar (backward
    asof). The ONE anchoring expression: the reindex-ffill is the only collapse step for feeds
    finer than the bars — a sub-bar print from inside bar D is never visible to bar D's own
    stamp (a value becomes usable at the first bar stamped at-or-after its timestamp)."""
    return native.reindex(index, method="ffill")


def _feed_age_days(native: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Calendar days since the feed's most recent native stamp at or before each bar (float; NaN
    before the first stamp). The stamps are availability times (post-lag), so the age is exactly
    how stale the value a decision at that bar would read is — and 'an event happened within the
    last N days' is ``days_since <= N``."""
    stamps = pd.Series(native.index, index=native.index)
    last = stamps.reindex(target_index, method="ffill")
    return (target_index.to_series(index=target_index) - last) / pd.Timedelta(days=1)


def _frozen_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild ``df`` over a READ-ONLY buffer, so a value write through the returned frame raises
    instead of silently serving stale results out of the evaluation memos (which are keyed by
    node identity and trust these frames never to move — previously a comment, enforced here where
    the loader can enforce it). The freeze must happen BEFORE construction: under pandas
    copy-on-write, flipping ``.values`` flags after the fact freezes a lazy copy, not the block
    a later write goes through. Loader-only: a hand-built ``MarketData`` keeps whatever
    mutability its caller gave it — the immutability promise there stays the caller's."""
    arr = df.to_numpy(dtype=float)
    arr.setflags(write=False)
    return pd.DataFrame(arr, index=df.index, columns=df.columns, copy=False)


def _frozen_series(s: pd.Series) -> pd.Series:
    arr = s.to_numpy(dtype=float)
    arr.setflags(write=False)
    return pd.Series(arr, index=s.index, name=s.name, copy=False)


def _frozen(v: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return _frozen_frame(v) if isinstance(v, pd.DataFrame) else _frozen_series(v)


def _coverage(values: pd.Series | pd.DataFrame) -> float:
    """Fraction of joined bars carrying a value (min across columns for per-target frames)."""
    if isinstance(values, pd.DataFrame):
        if values.shape[1] == 0 or len(values) == 0:
            return 0.0
        return float(values.notna().mean().min())
    return float(values.notna().mean()) if len(values) else 0.0


@dataclass(frozen=True)
class DataFiles:
    """The invocation's answer to a thesis's logical data keys: which CSV each named series IS,
    and which column of that CSV.

    The DSL names its series and never locates them (see :class:`~seikan.dsl.schema.DataSpec`), so
    this is the other half of a load — built by :func:`resolve_data_files` from the ``--data
    KEY=PATH`` pairs and the ``--column KEY=COL`` ones, or by a library caller directly. Nothing
    here is validated against the thesis; resolution did that, and a hand-built instance is the
    caller's own promise.
    """

    #: target name → CSV path, in declaration order.
    targets: dict[str, str]
    #: feed name → one CSV path (shared feed) or {target → path} (per-target feed).
    feeds: dict[str, str | dict[str, str]] = dataclasses.field(default_factory=dict)
    #: The excess-return source, when ``params.benchmark == "market"`` asked for one.
    benchmark: str | None = None
    #: data key → the CSV column that answers it, for the keys an invocation bound one to. Keyed by
    #: the same FLAT key the path mapping uses — a target name, a shared feed name, or the derived
    #: ``<feed>@<target>`` of one per-target member — and NEVER ``benchmark``, which is outcome
    #: measurement and always reads its file's ``open``. A key absent here bound no column, which is
    #: the ordinary case: only a file holding several numeric columns needs one. The name is matched
    #: case-insensitively against the file's (lowercased) headers when the file is read, so a name
    #: that no column answers refuses there (exit 2), not here.
    columns: dict[str, str] = dataclasses.field(default_factory=dict)


def resolve_data_files(
    thesis: Thesis, mapping: dict[str, str], columns: dict[str, str] | None = None
) -> DataFiles:
    """Bind a thesis's declared data keys to the files an invocation supplies, and to the columns
    it reads out of them.

    The path mapping must answer ``thesis.data_keys()`` EXACTLY. A missing key would load nothing
    where a series was declared — the run would measure a thesis it does not have — and an unknown
    key is a caller who believes this thesis reads something it never mentions; either way the
    honest answer is a refusal naming both sets, not a best-effort load.

    A COLUMN for a key is OPTIONAL, so ``columns`` is checked differently: only a file holding
    several numeric columns needs one, and most keys are answered by a single-column file that
    names itself. What is refused is a binding that cannot mean anything — a key this thesis does
    not declare (nothing would ever read it, and a caller who typed one is holding a different
    thesis in their head), and the ``benchmark`` key, which measures outcomes off its file's
    ``open`` price and has no column to choose. Whether the bound NAME exists in the file is the
    loader's question, not this one's: it takes bytes to answer, so it refuses there (exit 2,
    listing the columns the file actually has) rather than here.
    """
    required = thesis.data_keys()
    missing = [key for key in required if key not in mapping]
    unknown = sorted(set(mapping) - set(required))
    if missing or unknown:
        detail = "; ".join(
            part
            for part in (
                f"missing {missing}" if missing else "",
                f"unknown {unknown}" if unknown else "",
            )
            if part
        )
        raise ValueError(
            f"the data mapping does not answer this thesis's declared series ({detail}); "
            f"it reads {required}"
        )
    columns = dict(columns or {})
    if BENCHMARK_KEY in columns:
        raise ValueError(
            f"the {BENCHMARK_KEY!r} key takes no column: the benchmark is outcome MEASUREMENT, not "
            f"a decision input — it is sampled at the observation's own anchor bars and always "
            f"reads its file's 'open' price, so there is nothing for a column to select"
        )
    stray = sorted(set(columns) - set(required))
    if stray:
        raise ValueError(
            f"the column bindings name key(s) this thesis does not declare ({stray}); "
            f"it reads {required}"
        )
    spec = thesis.data
    feeds: dict[str, str | dict[str, str]] = {}
    for name, keys in spec.feed_keys().items():
        if spec.external[name].per_target:
            feeds[name] = {t: mapping[f"{name}@{t}"] for t in spec.targets}
        else:
            feeds[name] = mapping[keys[0]]
    return DataFiles(
        targets={name: mapping[name] for name in spec.targets},
        feeds=feeds,
        benchmark=mapping[BENCHMARK_KEY] if thesis.params.benchmark == "market" else None,
        columns=columns,
    )


def _shared_feed_path(paths: str | dict[str, str], fname: str) -> str:
    """The ONE path a shared feed binds — the invocation's answer to the feed's own key.

    ``DataFiles.feeds`` carries both feed shapes in one mapping (a single path for a shared feed,
    a ``{target: path}`` map for a per-target one) and :func:`resolve_data_files` derives each
    from that feed's own ``per_target`` flag, so the shapes agree by construction. A mismatch
    means a hand-built :class:`DataFiles` disagrees with the
    :class:`~seikan.dsl.schema.DataSpec` it is being loaded against — the caller's own promise
    broken — and it is refused by name here rather than left to fail as an unreadable path deeper
    in the read.
    """
    if not isinstance(paths, str):
        raise ValueError(
            f"external feed {fname!r} is declared shared (per_target=false) but its path binding "
            f"is a per-target map {sorted(paths)}; bind ONE path under the feed's own key"
        )
    return paths


def _per_target_feed_paths(paths: str | dict[str, str], fname: str) -> dict[str, str]:
    """The ``{target → path}`` map a per-target feed binds, one derived ``<feed>@<target>`` key per
    target. The mirror of :func:`_shared_feed_path` — see it for why the shape is checked at all."""
    if not isinstance(paths, dict):
        raise ValueError(
            f"external feed {fname!r} is declared per_target but its path binding is the single "
            f"path {paths!r}; bind one path per target under '{fname}@<target>'"
        )
    return paths


def load_market_data(spec: DataSpec, files: DataFiles) -> MarketData:
    targets = files.targets  # {name -> path}; keys resolved against the spec by resolve_data_files
    reports: list[FileReport] = []

    def _read(path: str, role: str, expected_shape: str | None = None) -> pd.DataFrame | None:
        df, rep = read_strict_csv(path, role=role, expected_shape=expected_shape)
        reports.append(rep)
        return df

    # Read EVERY referenced file first, so one DataError carries the complete diagnosis.
    raw_targets = {name: _read(path, f"target:{name}") for name, path in targets.items()}
    feed_specs = dict(spec.external)
    feed_paths: dict[str, str | dict[str, str]] = files.feeds
    raw_feeds: dict[str, dict[str | None, pd.DataFrame | None]] = {}
    for fname, feed in feed_specs.items():
        paths = feed_paths[fname]
        if feed.per_target:  # one key per target, derived — cover is by construction
            member_paths = _per_target_feed_paths(paths, fname)
            raw_feeds[fname] = {
                name: _read(member_paths[name], f"external:{fname}:{name}") for name in targets
            }
        else:
            raw_feeds[fname] = {None: _read(_shared_feed_path(paths, fname), f"external:{fname}")}
    raw_benchmark = (
        _read(files.benchmark, "benchmark", expected_shape="ohlcv") if files.benchmark else None
    )
    if any(not r.ok for r in reports):
        raise DataError(build_data_report(reports))

    def _fail(code: DataIssueCode, message: str) -> NoReturn:
        raise DataError(
            build_data_report(reports, errors=[{"code": code, "message": message}]), message
        )

    frames: dict[str, pd.DataFrame] = {}
    shapes: dict[str, TargetShape] = {}
    for name, path in targets.items():
        # Each target reads the column the invocation bound to ITS key — deliberately PER KEY, so
        # a thesis over three yields a vendor happened to ship in three differently-headed files
        # is stateable (one DSL-level column name could only serve every target at once). A key
        # that bound nothing passes None, and the file must then name its own single value column.
        try:
            frames[name], shapes[name] = _prepare_target(
                raw_targets[name], path, spec.start, spec.end, name, files.columns.get(name)
            )
        except ValueError as exc:
            _fail("spec_data_mismatch", str(exc))
    if len(set(shapes.values())) > 1:
        _fail(
            "mixed_target_shapes",
            f"targets mix OHLCV and series shapes ({shapes}); a thesis's targets must share one "
            f"shape (their measurements must be commensurable)",
        )
    target_shape = next(iter(shapes.values()))

    # Targets must share ONE bar clock exactly. Intersecting instead would let a timestamp
    # missing from a single target delete that bar from EVERY target — before any coverage
    # ledger exists to record it — so trimming the rows under adverse firings would silently
    # shrink the graded pool and could turn an unmet cell into a met one. (The
    # benchmark join below already refused this trade-off for exactly this reason: it reindexes
    # WITHOUT intersecting so a hole censors rather than shrinks.) A caller with genuinely
    # mixed calendars pre-aligns the files, or carves an explicit `data.start`/`data.end`
    # window — which enters the DSL hash, making the evaluated interval part of the run's
    # identity instead of an artifact of the data.
    union: pd.DatetimeIndex | None = None
    for df in frames.values():
        union = df.index if union is None else union.union(df.index)
    if union is not None:
        mismatches: list[str] = []
        for name in targets:
            missing = union.difference(frames[name].index)
            if len(missing):
                shown = ", ".join(ts.isoformat() for ts in missing[:MAX_EXAMPLES])
                extra = len(missing) - min(len(missing), MAX_EXAMPLES)
                mismatches.append(
                    f"target {name!r}: missing {len(missing)} timestamp(s) present in other "
                    f"targets ({shown}{f', +{extra} more' if extra else ''})"
                )
        if mismatches:
            _fail(
                "target_index_mismatch",
                "targets do not share one bar clock — "
                + "; ".join(mismatches)
                + ". Every target must carry the same timestamps over the evaluated interval "
                "(an intersection would delete those bars from every target and shrink the "
                "graded pool invisibly); pre-align the files, or restrict the window with "
                "data.start / data.end",
            )
    common = union
    if common is None or len(common) < 2:
        _fail(
            "insufficient_common_index",
            "targets share fewer than 2 common timestamps after alignment",
        )
    names = list(targets)

    def field_frame(col: str) -> pd.DataFrame:
        return pd.DataFrame(
            {name: frames[name][col].reindex(common) for name in names}, columns=names
        )

    has_volume = all("volume" in frames[name].columns for name in names)
    volume = field_frame("volume") if has_volume else None

    join_warnings: list[DataIssue] = []
    externals: dict[str, pd.Series | pd.DataFrame] = {}
    externals_age: dict[str, pd.Series | pd.DataFrame] = {}
    for fname, feed in feed_specs.items():
        lag = feed.lag_timedelta
        paths = feed_paths[fname]
        try:
            if not feed.per_target:
                # A shared feed answers to its own name, so that is the key its column is bound to.
                native = _prepare_feed(
                    raw_feeds[fname][None],
                    _shared_feed_path(paths, fname),
                    fname,
                    lag,
                    files.columns.get(fname),
                )
                externals[fname] = _anchor_feed(native, common)
                externals_age[fname] = _feed_age_days(native, common)
            else:
                # Each MEMBER of a per-target feed is bound under its own derived key, so the
                # members may now name DIFFERENT columns — deliberate, and the symmetric twin of
                # the per-member ``--data <feed>@<target>=PATH`` that already exists: whoever may
                # point two members at two files may also point them at two column headers. The
                # single DSL ``column`` could only say one name for the whole family.
                member_paths = _per_target_feed_paths(paths, fname)
                natives = {
                    name: _prepare_feed(
                        raw_feeds[fname][name],
                        member_paths[name],
                        f"{fname}@{name}",
                        lag,
                        files.columns.get(f"{fname}@{name}"),
                    )
                    for name in names
                }
                externals[fname] = pd.DataFrame(
                    {name: _anchor_feed(natives[name], common) for name in names},
                    columns=names,
                )
                externals_age[fname] = pd.DataFrame(
                    {name: _feed_age_days(natives[name], common) for name in names}, columns=names
                )
        except ValueError as exc:
            _fail("spec_data_mismatch", str(exc))
        cov = _coverage(externals[fname])
        if cov < _FEED_COVERAGE_WARN:
            join_warnings.append(
                {
                    "code": "external_coverage",
                    "message": (
                        f"external feed {fname!r} covers only {cov:.1%} of the joined bars "
                        "(uncovered bars read NaN)"
                    ),
                }
            )

    # Benchmark: the targets define the timeline — reindex with NO ffill, so a bar where the
    # benchmark is missing stays NaN and censors the observations anchored there (intersecting
    # the index instead would silently shrink the sample and shift every transform).
    benchmark_open = None
    if files.benchmark:
        benchmark_open = _slice(raw_benchmark, spec.start, spec.end)["open"].reindex(common)
        cov = _coverage(benchmark_open)
        if cov < 1.0:
            join_warnings.append(
                {
                    "code": "benchmark_coverage",
                    "message": (
                        f"benchmark covers only {cov:.1%} of the joined bars — uncovered bars "
                        "censor their observations (exit_reason='no_benchmark')"
                    ),
                }
            )

    join: JoinInfo = {
        "n_common": len(common),
        "start": common[0].isoformat(),
        "end": common[-1].isoformat(),
        "warnings": join_warnings,
    }
    return MarketData(
        close=_frozen_frame(field_frame("close")),
        open=_frozen_frame(field_frame("open")),
        high=_frozen_frame(field_frame("high")),
        low=_frozen_frame(field_frame("low")),
        volume=_frozen_frame(volume) if volume is not None else None,
        externals={k: _frozen(v) for k, v in externals.items()},
        targets=names,
        benchmark_open=_frozen_series(benchmark_open) if benchmark_open is not None else None,
        benchmark_path=files.benchmark,
        externals_age={k: _frozen(v) for k, v in externals_age.items()},
        target_shape=target_shape,
        report=build_data_report(reports, join=join),
    )
