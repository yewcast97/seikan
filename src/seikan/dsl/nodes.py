"""The Series node vocabulary: swept numeric param aliases, the strict base model, the data
leaves, every transform, the cross-sectional trio, the operator pair, and the ``Series`` union
(defined HERE so its members' forward references rebuild against this namespace).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator
from pydantic import Field as PField

from seikan.constants import (
    MAX_DECLARED_GRID,
)


def _distinct_sweep[T](values: list[T]) -> list[T]:
    """Reject a repeated value inside one sweep axis.

    A sweep axis enumerates DISTINCT hypotheses. Repeating a value declares the same hypothesis
    twice, which adds no information but is not merely redundant — it corrupts the per-cell
    counting. The measurement loop runs the duplicate combo once per occurrence and appends its
    observations to the trades frame each time, while the per-cell panel groups that frame by
    parameter VALUE: all d copies of the axis point therefore read one d-fold-duplicated group
    and every count derived from it (``by_target.n``, the outcome ledger's
    ``n_attempted``/``n_closed``, ``episode_stats.n``) reports d times the real evidence. The
    overlap-honest ``n_nonoverlap`` is computed from the firing bars and stays truthful, so the
    inflation cannot even be caught by the gate's ``n_nonoverlap <= n`` reconciliation — it makes
    that
    bound strictly easier to satisfy. The result is fail-OPEN: ``[21, 21]`` doubles a cell's
    apparent support and can flip the ``support`` floor from unmet to met.

    Refusing here — at validation, before a byte of data is read — is the same discipline
    ``vectorize.collect_sweeps`` applies to duplicate axis NAMES, and for the same stated
    reason: a collision that miscounts silently must never reach the engine.
    """
    seen: list[T] = []
    dupes: list[T] = []
    for v in values:
        if v in seen:
            if v not in dupes:
                dupes.append(v)
        else:
            seen.append(v)
    if dupes:
        raise ValueError(
            f"sweep axis repeats the value(s) {dupes} — a swept list enumerates DISTINCT "
            "hypotheses, and a repeated value multiplies that cell's reported observation count "
            "without adding evidence; list each value once"
        )
    return values


#: A swept list: bounded length, and every value distinct.
_Sweep = (PField(min_length=1, max_length=MAX_DECLARED_GRID), AfterValidator(_distinct_sweep))

PosInt = Annotated[int, PField(gt=0)]

PosIntParam = PosInt | Annotated[list[PosInt], *_Sweep]

Ge2Int = Annotated[int, PField(ge=2)]

Ge2IntParam = Ge2Int | Annotated[list[Ge2Int], *_Sweep]

Ge3Int = Annotated[int, PField(ge=3)]

Ge3IntParam = Ge3Int | Annotated[list[Ge3Int], *_Sweep]

NonNegInt = Annotated[int, PField(ge=0)]

NonNegIntParam = NonNegInt | Annotated[list[NonNegInt], *_Sweep]

# A threshold constant accepts a scalar or a list; a list sweeps the threshold as its own named axis
# (see ``Constant.name``), taking part in the same Cartesian product as transform-window sweeps.
# NON-FINITE is rejected: JSON has no NaN/Infinity, but Python's ``json`` accepts the non-standard
# literals and pydantic floats admit them by default. A NaN threshold silently makes every
# comparison undecidable, and ``canonical_dsl_hash`` would hash a token no strict JSON parser can
# read back — so the identity of such a thesis is unrecoverable.
FiniteFloat = Annotated[float, PField(allow_inf_nan=False)]

FloatParam = FiniteFloat | Annotated[list[FiniteFloat], *_Sweep]

#: The plain shape the constrained param aliases above erase to: a numeric node param as the
#: traversal/rendering helpers below receive it — the scalar form, or the list form that sweeps it
#: (a ``window``/``periods``/``cooldown`` is int-valued, a ``constant.value`` float-valued). It
#: carries no constraint metadata and is never a field annotation; the models declare the
#: constrained aliases.
_NumericParam = int | float | list[int] | list[float]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Field(_Strict):
    type: Literal["field"] = "field"
    column: Literal["open", "high", "low", "close", "volume"] = "close"


class Constant(_Strict):
    type: Literal["constant"] = "constant"
    value: FloatParam
    # A list ``value`` sweeps the threshold as its own result axis (like a transform-window sweep);
    # ``name`` labels that axis in the result columns / summary tables and is required
    # (non-empty) in that case. A SCALAR constant refuses a name: execution ignores it, so it
    # would be a hash-visible field with no output — two identities for one measurement.
    name: str | None = None

    @model_validator(mode="after")
    def _require_name_when_swept(self) -> Constant:
        if isinstance(self.value, list) and not (isinstance(self.name, str) and self.name.strip()):
            raise ValueError(
                "a swept constant (list 'value') requires a non-empty 'name' to label its "
                "sweep axis"
            )
        if not isinstance(self.value, list) and self.name is not None:
            raise ValueError(
                "a scalar constant takes no 'name' — the name labels a sweep axis and exists "
                "only when 'value' is a list"
            )
        return self


class External(_Strict):
    type: Literal["external"] = "external"
    name: str


class Calendar(_Strict):
    # Calendar attribute of each bar's timestamp, broadcast to every target — the seasonality
    # primitive (turn-of-month, day-of-week, sell-in-May). CALENDAR-DAY arithmetic only: ``month``
    # (1-12), ``day_of_week`` (0=Monday .. 6=Sunday), ``day_of_month`` (1-31), ``days_to_month_end``
    # (calendar days remaining in the month, 0 = the month's last calendar day). All are knowable at
    # the bar itself; a "trading bars to month end" field would need the future session calendar and
    # is deliberately absent (look-ahead).
    type: Literal["calendar"] = "calendar"
    field: Literal["month", "day_of_week", "day_of_month", "days_to_month_end"]


class DaysSince(_Strict):
    # Calendar days since the named external feed's most recent NATIVE observation at or before the
    # bar (post-``lag`` availability stamps, NOT the forward-filled values) — the event-distance
    # primitive (PEAD windows: ``days_since(earnings) <= 3``; staleness guards). NaN before the
    # feed's first stamp (never fires — the standard NaN-gating contract). The feed must be declared
    # in ``data.external`` like an ``External`` reference. For SCHEDULED future events (FOMC,
    # earnings dates), feed a user-computed days-until-next-event series instead — the schedule is
    # public in advance, so the value is availability-honest; a ``days_until`` primitive over feed
    # stamps would read the future and is deliberately absent.
    type: Literal["days_since"] = "days_since"
    name: str


class EMA(_Strict):
    type: Literal["ema"] = "ema"
    input: Series
    window: PosIntParam


class ZScore(_Strict):
    type: Literal["zscore"] = "zscore"
    input: Series
    window: Ge2IntParam
    mean_type: Literal["sma", "ema"] = "sma"


class Percentile(_Strict):
    # Fraction of the trailing window strictly below the current value:
    # count(value < current) / window, in [0, (window-1)/window].
    type: Literal["percentile"] = "percentile"
    input: Series
    window: Ge2IntParam


class RollingAgg(_Strict):
    # A window aggregate of the input series — ``max`` / ``min`` / ``mean`` / ``std`` (population,
    # ddof=0). With ``window`` set: trailing-window (NaN until the window is full AND every bar in
    # it is finite — the same gate as ``percentile``). With ``window`` omitted/null:
    # EXPANDING max/min only (all-time high/low of the series so far; NaN until the first finite
    # bar) — the general-purpose peak/trough primitive; expanding ``mean``/``std`` are rejected
    # (silent regime-drift trap). Prefer the dedicated ``drawdown`` node for depth-below-peak;
    # ``binary_op(close / rolling_agg(close, N, "max"))`` remains valid. ``std`` needs ``window``
    # >= 2, so a set ``window`` is ``Ge2IntParam`` like sibling transforms. A simple moving average
    # is ``agg:"mean"``; realized vol is ``rolling_agg(change(close, kind="log"), N, "std")``.
    type: Literal["rolling_agg"] = "rolling_agg"
    input: Series
    window: Ge2IntParam | None = None
    agg: Literal["max", "min", "mean", "std"]

    @model_validator(mode="after")
    def _check_expanding(self) -> RollingAgg:
        if self.window is None and self.agg not in ("max", "min"):
            raise ValueError(
                "expanding rolling_agg (window omitted) only supports agg='max' or 'min'; "
                f"got agg={self.agg!r}"
            )
        return self


class Drawdown(_Strict):
    # Fractional depth below a peak: ``input / peak − 1`` (≤ 0). ``window`` set → trailing N-bar
    # peak; omitted → expanding (all-time) peak. ``input`` defaults to the target's close. Counts
    # as one operator level, recovering nesting budget vs composing
    # ``binary_op(close / rolling_agg(...))``. "In a ≥30% drawdown" is ``drawdown < -0.30``.
    # POSITIVE-scale domain: a non-positive peak makes the ratio meaningless (it inverts on a
    # negative scale), so those bars are NaN — undecidable, never a sign-flipped depth. For
    # signed series (yields, spreads) compose level reads with ``change(kind="diff")`` instead.
    type: Literal["drawdown"] = "drawdown"
    input: Series = PField(default_factory=Field)
    window: PosIntParam | None = None


class Runup(_Strict):
    # Fractional height above a trough: ``input / trough − 1`` (≥ 0) — the exact mirror of
    # ``drawdown``. ``window`` set → trailing N-bar trough; omitted → expanding (all-time) trough.
    # ``input`` defaults to the target's close. Reads both ways: ``runup >= 0.05`` as a recovery/
    # stabilization guard after a trough, or as an extension/overheat read near a peak. Counts as
    # one operator level. POSITIVE-scale domain like ``drawdown``: a non-positive trough → NaN.
    type: Literal["runup"] = "runup"
    input: Series = PField(default_factory=Field)
    window: PosIntParam | None = None


class BarsSinceExtremum(_Strict):
    # Bar count since the most recent bar attaining the trailing/expanding ``max``|``min`` of
    # ``input``. ``extremum="max"`` + expanding = bars since the all-time high (drawdown duration);
    # ``extremum="min"`` = bars since the trough ("no new low in K bars" = stabilization). Ties
    # reset to the MOST RECENT attaining bar (a retest of the peak restarts the duration).
    # ``window`` set → trailing N-bar extremum; omitted → expanding. ``input`` defaults to close.
    # Trailing form NaNs until the window is full AND every bar in it is finite (same gate as
    # ``percentile``/``rolling_agg``). Counts as one operator level.
    type: Literal["bars_since_extremum"] = "bars_since_extremum"
    extremum: Literal["max", "min"]
    input: Series = PField(default_factory=Field)
    window: PosIntParam | None = None


class Change(_Strict):
    # k-period change of ``input``, mirroring ``Outcome.kind``: ``pct`` = (cur/prev − 1),
    # ``log`` = ln(cur/prev), ``diff`` = cur − prev (level change — the honest form for
    # rates/spreads/multiples where a percent is dimensionally wrong). ``periods`` may be a list
    # (sweeps as ``change_periods``). ``pct`` and ``log`` require BOTH endpoints strictly
    # positive (ratio algebras invert silently on a signed scale — the same rule the forward
    # outcome applies) and NaN otherwise; ``diff`` only needs both bars finite (0 is a valid
    # level base).
    type: Literal["change"] = "change"
    input: Series
    periods: PosIntParam = 1
    kind: Literal["pct", "log", "diff"] = "pct"


class Shift(_Strict):
    # The input series ``periods`` bars ago (out[t] = in[t - periods]); the leading ``periods`` bars
    # are NaN. Backward-only (periods >= 1), so it can never read the future. Prefer ``change`` for
    # k-period pct/log/diff; ``shift`` remains for level comparisons (``close > shift(high, 1)``).
    # Depth-transparent like ``binary_op`` (plumbing, not an operator level); a list ``periods``
    # sweeps as its own ``shift_periods`` axis.
    type: Literal["shift"] = "shift"
    input: Series
    periods: PosIntParam = 1


class RollingCorr(_Strict):
    # Trailing-window Pearson correlation of two Series — a per-target TIME-axis transform (each
    # target's own history only; ranking across the targets at a bar is ``cross_rank``, basket
    # mode). ``window`` >= 3 (corr of 2 points
    # is always ±1 — degenerate). NaN unless the whole window is finite in BOTH inputs AND both
    # window stds > 0 (zero-variance → NaN, never ±1) — the standard NaN-gating contract. Counts as
    # ONE operator level with TWO children. Recipe for "corr(stock daily return, Δ option EOD
    # IV30)": declare the IV feed in ``data.external``, then
    # ``rolling_corr(change(close), change(iv30, kind="diff"), window)`` —
    # ``change`` counts as one level, so the whole recipe costs two of the five levels.
    type: Literal["rolling_corr"] = "rolling_corr"
    left: Series
    right: Series
    window: Ge3IntParam


class CrossRank(_Strict):
    # Ascending fraction-rank of the target's value among all targets' finite values at bar t:
    # (avg_rank - 1) / (k - 1), in [0, 1] (average ranks on ties; k = finite targets at t).
    # NaN where the target's own value is NaN or k < min_valid. ``cross_rank(x) >= 0.8`` IS
    # top-quintile membership — the quantile primitive; there is no separate quantile node.
    type: Literal["cross_rank"] = "cross_rank"
    input: Series
    min_valid: Ge2Int = 2


class CrossDemean(_Strict):
    # The target's value minus the cross-target mean of the finite values at bar t (self included).
    # NaN where the target's own value is NaN or fewer than min_valid targets are finite. The
    # zscore recipe: ``binary_op(cross_demean(x), "/", cross_agg(x, "std"))`` — the same 1 + d(x)
    # nesting cost as either node alone, and a zero-dispersion bar divides by 0 → NaN via the
    # ``BinaryOp`` ``/`` contract (never fires).
    type: Literal["cross_demean"] = "cross_demean"
    input: Series
    min_valid: Ge2Int = 2


class CrossAgg(_Strict):
    # A cross-target AGGREGATE at each bar, broadcast back to every target column — the breadth /
    # dispersion primitive ("70% of the group above its 200d SMA" via ``frac_positive`` of
    # ``binary_op(close - rolling_agg(close, 200, "mean"))``; a cross-sectional-vol regime via
    # ``std``). Unlike ``cross_rank``/``cross_demean`` the value is a property of the CROSS-SECTION,
    # not of the individual target, so a bar carries the VALUE (for every column) whenever at
    # least ``min_valid`` targets are finite — a target whose own value is still warming up sees
    # the group's breadth like any other; intentional, do not "fix". The FIRING latch is stricter
    # than the value: a member's warmup gates on its OWN input as well (``compiler.vectorize``),
    # so a member whose series has not yet produced a value cannot fire off the broadcast — a
    # pre-listing firing would censor as ``no_outcome`` and hard-refuse the cell it lands in.
    # ``std`` is the population std
    # (ddof=0, matching ``rolling_agg``); ``frac_positive`` is the fraction of finite values > 0,
    # in [0, 1].
    type: Literal["cross_agg"] = "cross_agg"
    input: Series
    agg: Literal["mean", "median", "std", "frac_positive"]
    min_valid: Ge2Int = 2


class BinaryOp(_Strict):
    type: Literal["binary_op"] = "binary_op"
    left: Series
    right: Series
    op: Literal["+", "-", "*", "/"]


class UnaryOp(_Strict):
    # Element-wise unary arithmetic, *transparent* for the nesting-depth cap like ``BinaryOp``.
    # Out-of-domain inputs (``log`` of a non-positive, ``sqrt`` of a negative) map to NaN — the
    # NaN-gating contract (never fires). ``abs`` unlocks the magnitude classics (Amihud
    # illiquidity ``|ret|/dollar volume``, |surprise| conditioning, |z| as an input series —
    # two-sided *conditions* stay an ``or`` of thresholds).
    type: Literal["unary_op"] = "unary_op"
    input: Series
    op: Literal["abs", "log", "sign", "sqrt", "neg"]


Series = Annotated[
    Field
    | Constant
    | External
    | Calendar
    | DaysSince
    | EMA
    | ZScore
    | Percentile
    | RollingAgg
    | Drawdown
    | Runup
    | BarsSinceExtremum
    | Change
    | Shift
    | RollingCorr
    | CrossRank
    | CrossDemean
    | CrossAgg
    | BinaryOp
    | UnaryOp,
    PField(discriminator="type"),
]


# Forward references resolve against THIS module's namespace — the unions and their
# members must rebuild where they are defined.
EMA.model_rebuild()
ZScore.model_rebuild()
Percentile.model_rebuild()
RollingAgg.model_rebuild()
Drawdown.model_rebuild()
Runup.model_rebuild()
BarsSinceExtremum.model_rebuild()
Change.model_rebuild()
Shift.model_rebuild()
RollingCorr.model_rebuild()
CrossRank.model_rebuild()
CrossDemean.model_rebuild()
CrossAgg.model_rebuild()
BinaryOp.model_rebuild()
UnaryOp.model_rebuild()
