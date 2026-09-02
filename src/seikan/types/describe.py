"""Emitted-shape declarations: the describe profiles."""

from __future__ import annotations

from typing import Literal, TypedDict

from seikan.types.data_report import (
    DataReport,
)
from seikan.types.run import (
    BarSpacing,
)

# ---- the describe profiles ----------------------------------------------------------------
#
# ``describe.py``'s pure data profiling — the one non-measuring path. Every block states what the
# bytes contain and refuses explicitly rather than repairing: ``reason`` names a structural
# refusal (``insufficient_bars`` / ``endpoint_missing``) and ``ratio_reason`` the domain refusal
# of the RATIO algebras (``non_positive_endpoint``), both present-never-omitted.


class ChangeBlock(TypedDict):
    """One window's change (``describe._change_block``) or one window's per-bar dispersion
    (``describe._dispersion_block``) — the two share this shape exactly, one value per algebra.

    All three algebras are emitted, because choosing ONE would be the engine deciding what a
    series IS. ``diff`` carries whenever both endpoints are finite; ``pct``/``log`` are None with
    ``ratio_reason`` set off a positive scale. A refused block is all-None with ``reason`` set.
    """

    diff: float | None
    pct: float | None
    log: float | None
    reason: Literal["insufficient_bars", "endpoint_missing"] | None
    ratio_reason: Literal["non_positive_endpoint"] | None


class ExtremumPoint(TypedDict):
    """A trailing or full-sample extremum: its value and the bar it was attained at (ties resolve
    to the MOST RECENT bar — the ``bars_since_extremum`` convention, one tie rule engine-wide)."""

    value: float
    timestamp: str


class RangeDistance(TypedDict):
    """The last level's distance from a window extremum — ``diff`` always, ``pct`` only on a
    positive scale (None with the block's ``ratio_reason`` set otherwise)."""

    diff: float
    pct: float | None


class RangePositionBlock(TypedDict):
    """One window's range position — ``describe._range_position_block``.

    Requires the FULL window: a shorter file refuses as ``insufficient_bars`` and a hole inside
    the window as ``endpoint_missing`` (the extremum, or the rank, could be hiding in it), and a
    refusal nulls every value. ``percentile_rank`` is position within the window's OWN range, not
    valuation.
    """

    high: ExtremumPoint | None
    low: ExtremumPoint | None
    from_high: RangeDistance | None
    from_low: RangeDistance | None
    percentile_rank: float | None
    reason: Literal["insufficient_bars", "endpoint_missing"] | None
    ratio_reason: Literal["non_positive_endpoint"] | None


class FullSampleBlock(TypedDict):
    """The whole file's extremes and the drawdown/runup off them — ``describe._full_sample_block``.

    A property of the file's EXTENT: extend or trim the file and these move. A NaN last level
    refuses the distance reads (``endpoint_missing``) while the extremes themselves — observed
    facts — stay, with the missingness block beside them stating the holes.
    """

    high: ExtremumPoint | None
    low: ExtremumPoint | None
    drawdown_diff: float | None
    drawdown_pct: float | None
    runup_diff: float | None
    runup_pct: float | None
    reason: Literal["insufficient_bars", "endpoint_missing"] | None
    ratio_reason: Literal["non_positive_endpoint"] | None


class MissingnessBlock(TypedDict):
    """Pure hole counts for one series — ``describe._missingness_block``. Interior missingness is
    counted between the first and last valid bar; threshold-flavored warnings stay in the
    loader's ``data_report``."""

    n_missing: int
    n_interior_missing: int
    first_valid: str | None
    last_valid: str | None


class SeriesProfile(TypedDict):
    """One value column's complete profile — ``describe._series_blocks``. The window-keyed maps
    are keyed by the window's BAR COUNT rendered as a string, in the requested window order."""

    changes: dict[str, ChangeBlock]
    dispersion: dict[str, ChangeBlock]
    range_position: dict[str, RangePositionBlock]
    full_sample: FullSampleBlock
    missingness: MissingnessBlock


class VolumeWindow(TypedDict):
    """One window's volume read — ``describe._volume_block``. ``last_to_mean`` carries NO
    "unusual" flag: what counts as elevated is the caller's judgment, not a property of the file.
    A non-positive trailing mean refuses the ratio rather than dividing through zero."""

    mean: float | None
    last_to_mean: float | None
    reason: Literal["insufficient_bars", "endpoint_missing"] | None
    ratio_reason: Literal["non_positive_endpoint"] | None


class VolumeBlock(TypedDict):
    """The volume panel of an OHLCV file that carries a volume column — ``describe._volume_block``;
    None on every other file."""

    last: float | None
    windows: dict[str, VolumeWindow]


class LastBarBlock(TypedDict):
    """The file's final row verbatim — ``describe._last_bar_block``. Every column, NaN as null,
    never back-filled."""

    timestamp: str
    values: dict[str, float | None]


class FileProfile(TypedDict):
    """One ADMITTED file's profile — ``describe.profile_file``, emitted in argument order under
    the describe document's ``profiles``.

    An OHLCV file profiles ``close`` (the full bar rides ``last_bar``); a series-shaped file
    profiles every value column in file order. ``volume`` is None unless the file is OHLCV with a
    volume column. Bounded output by construction: no per-bar array ever rides it.
    """

    path: str
    sha256: str | None
    ok: bool
    shape: Literal["ohlcv", "series"] | None
    n_bars: int
    index_start: str
    index_end: str
    bar_spacing: BarSpacing
    last_bar: LastBarBlock
    series: dict[str, SeriesProfile]
    volume: VolumeBlock | None


class RefusalStub(TypedDict):
    """A REFUSED file's profile — ``describe._refusal_stub``: identity plus the reason codes,
    nothing invented. The full diagnosis lives in the ``data_report`` entry those codes point
    into. ``ok`` is always False, which is what distinguishes it from a :class:`FileProfile`."""

    path: str
    sha256: str | None
    ok: bool
    reason: str


class DescribeResult(TypedDict):
    """``describe.describe_files``'s return — the strict-read verdict plus one profile per file in
    ARGUMENT order (a refused file gets its stub, so the document is complete either way). The
    CLI splits it into the describe document's two sections and reads ``data_report["ok"]`` for
    the exit code."""

    data_report: DataReport
    profiles: list[FileProfile | RefusalStub]
