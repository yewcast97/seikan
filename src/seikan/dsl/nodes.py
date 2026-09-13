"""The Series node vocabulary: swept numeric param aliases, the strict base model, the data
leaves, every transform, the cross-sectional trio, the operator pair, the event-anchor family
(which embeds a ``Condition``), and the ``Series`` union.

The Series and Condition vocabularies are MUTUALLY recursive (a threshold reads Series operands;
``mask``/``bars_since_event``/``event_value``/``event_agg`` read a Condition), so the models here
are NOT rebuilt in this module: :mod:`seikan.dsl.conditions` rebuilds every Series and Condition
model once both unions exist, and :mod:`seikan.dsl` imports it so nothing ever sees a half-built
model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator
from pydantic import Field as PField

from seikan.constants import (
    MAX_DECLARED_GRID,
)

if TYPE_CHECKING:
    # Annotations are strings under ``from __future__ import annotations``; the runtime name is
    # supplied to ``model_rebuild`` by ``seikan.dsl.conditions`` (see the module docstring).
    from seikan.dsl.conditions import Condition


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

#: An EW decay factor stated directly: ``0 < alpha <= 1`` (RiskMetrics 0.06; Wilder's 1/N).
UnitFloat = Annotated[float, PField(gt=0, le=1, allow_inf_nan=False)]

UnitFloatParam = UnitFloat | Annotated[list[UnitFloat], *_Sweep]

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
    # is deliberately absent (look-ahead). ``hour`` (0-23) and ``minute`` (0-59) read the stamp AS
    # THE CSV GIVES IT — the engine does not interpret whether a stamp is a bar's open or close.
    type: Literal["calendar"] = "calendar"
    field: Literal["month", "day_of_week", "day_of_month", "days_to_month_end", "hour", "minute"]


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
    # Exponentially weighted mean of ``input``, seeded at the first finite value, NaN-skipping.
    # EXACTLY ONE of ``window`` / ``alpha``: ``window`` sets ``alpha = 2/(window+1)`` with a
    # ``window``-observation warmup (the classic span form); ``alpha`` states the decay directly
    # (``0 < alpha <= 1`` — RiskMetrics 0.06; Wilder's 1/N smoother is ``alpha = 1/N`` with THIS
    # seed, not an SMA seed) with warmup ``ceil(2/alpha − 1)`` observations, so the two forms
    # agree bit-exactly where ``alpha = 2/(window+1)``. ``alpha = 1`` is the input itself. A list
    # sweeps as ``ema_window`` / ``ema_alpha`` by which field is set.
    type: Literal["ema"] = "ema"
    input: Series
    window: PosIntParam | None = None
    alpha: UnitFloatParam | None = None

    @model_validator(mode="after")
    def _check_window_xor_alpha(self) -> EMA:
        _require_window_xor_alpha("ema", self.window, self.alpha)
        return self


def _require_window_xor_alpha(kind: str, window: object, alpha: object) -> None:
    if (window is None) == (alpha is None):
        got = "both" if window is not None else "neither"
        raise ValueError(
            f"{kind} takes exactly one of 'window' or 'alpha'; got {got} — 'window' sets "
            "alpha = 2/(window+1) with a window-bar warmup, 'alpha' states the decay directly "
            "(0 < alpha <= 1, e.g. RiskMetrics 0.06) with warmup = ceil(2/alpha − 1) bars; both "
            "are seeded at the first finite value"
        )


class ZScore(_Strict):
    # ``(x − mean) / std`` over a trailing window (``sma``: two-pass population moments over the
    # last ``window`` bars) or an EW recurrence (``ema``: the West recurrence; ``window`` sets
    # ``alpha = 2/(window+1)`` or ``alpha`` states it directly — the EMA's own two forms; ``alpha``
    # is only valid with ``mean_type="ema"`` and must be < 1, since ``alpha = 1`` leaves zero EW
    # variance and the z-score could never be defined). Sweeps as ``zscore_window`` /
    # ``zscore_alpha``.
    type: Literal["zscore"] = "zscore"
    input: Series
    window: Ge2IntParam | None = None
    alpha: UnitFloatParam | None = None
    mean_type: Literal["sma", "ema"] = "sma"

    @model_validator(mode="after")
    def _check_window_xor_alpha(self) -> ZScore:
        _require_window_xor_alpha("zscore", self.window, self.alpha)
        if self.alpha is not None:
            if self.mean_type != "ema":
                raise ValueError("zscore 'alpha' is only valid with mean_type='ema'")
            alphas = self.alpha if isinstance(self.alpha, list) else [self.alpha]
            if any(a >= 1.0 for a in alphas):
                raise ValueError(
                    "zscore alpha must be < 1: alpha=1 leaves zero EW variance, so the z-score "
                    "could never be defined"
                )
        return self


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
    # ``median`` and ``mad`` (the median absolute deviation about the SAME window's median,
    # UNSCALED — multiply by 1.4826 via ``binary_op`` for the normal-consistent sigma) are the
    # robust pair: robust z = ``(x − rolling_agg(x, N, median)) / (1.4826 · rolling_agg(x, N,
    # mad))``. Note a rolling median of ``|x − rolling_median(x)|`` is NOT the MAD (each bar's
    # deviation would be taken about a different window's median); ``mad`` centres every
    # deviation on the one window it is computed over. Trailing-window only (no expanding form).
    type: Literal["rolling_agg"] = "rolling_agg"
    input: Series
    window: Ge2IntParam | None = None
    agg: Literal["max", "min", "mean", "std", "median", "mad"]

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
    # THE POPULATION MODEL, shared by the three cross nodes (documented once, here).
    # ``where`` — ELIGIBILITY: member g enters the cross-section at bar t iff ``where``'s tradable
    # signal holds for g there. A warming (``~init``) or decided-False member is EXCLUDED (its
    # input is NaN to the kernel; its own output is NaN). ANY member post-warmup UNDEFINED
    # (``init & ~defined``) voids the WHOLE bar's cross-section (NaN for every member): a
    # population whose membership is unknowable is not a population — fail-closed. The canonical
    # idiom is ``and(E, cross_rank(x, where=E) >= q)``: a once-eligible member's later ineligible
    # bars are post-init NaN, so the bare threshold reads undefined, and the outer ``and(E, …)``
    # absorbs it (Kleene F ∧ U = F). The guard does NOT absorb an undefined E (U ∧ U = U) —
    # intended; ``source_coverage`` reports that feed hole too.
    # ``group`` — point-in-time LABELS (typically a ``per_target`` feed of integer codes; a
    # ``Constant`` is legal and equals no grouping): the node reduces WITHIN each distinct finite
    # label at each bar; a NaN label excludes the member; ``min_valid`` applies PER GROUP;
    # ``cross_agg`` broadcasts each group's aggregate to that group's members only. Labels compare
    # by exact float equality — use integer codes. Nested cross nodes inside ``where``/``group``
    # are legal (a liquidity screen ``where = cross_rank(dollar_volume) >= 0.5``), each with its
    # own ``cross_breadth`` entry. ``cross_breadth.k`` counts ENTERING members (finite input ∧
    # eligible ∧ finitely labelled); a bar voided by an undefined eligibility has k = 0.
    where: Condition | None = None
    group: Series | None = None


class CrossDemean(_Strict):
    # The target's value minus the cross-target mean of the finite values at bar t (self included).
    # NaN where the target's own value is NaN or fewer than min_valid targets are finite. The
    # zscore recipe: ``binary_op(cross_demean(x), "/", cross_agg(x, "std"))`` — the same 1 + d(x)
    # nesting cost as either node alone, and a zero-dispersion bar divides by 0 → NaN via the
    # ``BinaryOp`` ``/`` contract (never fires).
    type: Literal["cross_demean"] = "cross_demean"
    input: Series
    min_valid: Ge2Int = 2
    where: Condition | None = None  # see ``CrossRank`` for the population model
    group: Series | None = None


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
    where: Condition | None = None  # see ``CrossRank`` for the population model
    group: Series | None = None


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


# ---- the event-anchor family: Series nodes that embed a Condition --------------------------
#
# Shared definition. The EVENT of a Condition E is its tradable signal ``value & init & defined``
# (``compiler.vectorize.signal``). ``s(t)`` is the latest bar ``<= t`` at which E fired — the
# current bar INCLUDED (``input[t]`` is knowable at ``t``; the exclusive forms are ``shift(·, 1)``
# on the input or ``lag(E, 1)`` on the event). A new event REPLACES the old one (latest wins).
# Before the first event the node is WARMUP (NaN, not initialized — a thesis whose event never
# fires has no anchor, which is not a data hole). After a post-warmup hole in E — a bar E could not
# decide — ``s(t)`` is UNKNOWN until the next defined event: the node reads NaN while initialized,
# so a threshold over it is undefined and lands in ``signal_coverage``, never silently False. Every
# node is a deterministic, causal function of E's history and the input series up to ``t`` — the
# same class of signal-side state ``first_true``, ``ema`` and the expanding extrema already carry:
# no fill, no P&L, no position, no future bar. ``compiler.nb`` (the event-anchor section) holds
# the kernels; ``compiler.vectorize`` states the channel rules per node.
#
# The embedded Condition is a DECISION, not a transform level: it is invisible to the nesting-depth
# count of the node that embeds it, and its own threshold operands are depth-checked as ROOTS of
# their own (``traverse.iter_condition_series`` yields them beside the entry's outer operands, so
# they also appear in ``--root-series-out``). Sweeps inside it register once, in engine order.


class Mask(_Strict):
    # The Condition as a Series: 1.0 where its tradable signal holds, 0.0 where it was decided
    # False, NaN while warming or undecidable. Depth-TRANSPARENT (plumbing, like ``shift``): it
    # converts a decision into a number and nothing else. The bridge every counting recipe rides —
    # "at least K of A/B/C" is ``mask(A) + mask(B) + mask(C) >= K``, "breadth of a condition" is
    # ``cross_agg(mask(C), mean)``, "any/all/count of C since S" is ``event_agg(S, mask(C),
    # max|min|sum)``. Its ``init`` is the condition's OWN latch, so a hole at C's first initialized
    # bar is ledgered rather than absorbed as warmup.
    type: Literal["mask"] = "mask"
    condition: Condition


class BarsSinceEvent(_Strict):
    # ``t − s(t)``: bars since the latest event (0 on the event bar). Streak length is
    # ``bars_since_event(not(C))``; a session clock on intraday bars is
    # ``bars_since_event(session_open)``. Counts ONE level (it has no Series child).
    type: Literal["bars_since_event"] = "bars_since_event"
    event: Condition


class EventValue(_Strict):
    # ``input[s(t)]``: the input's value AT the latest event — a recorded level, held until the
    # next event (later holes in the input are irrelevant; a NaN input AT the event bar reads NaN
    # until the next event — the anchor exists but the snapshot does not). The breakout-retest
    # primitive: ``L = event_value(S, shift(rolling_agg(high, N, max), 1))`` is the level that WAS
    # broken, not the moving one. Counts one level over ``input``.
    type: Literal["event_value"] = "event_value"
    event: Condition
    input: Series


class EventAgg(_Strict):
    # ``agg(input[s(t) .. t])``: the input aggregated over the bars since the latest event, the
    # event bar included — a resettable running aggregate. STRICT finite gate like ``rolling_agg``:
    # one NaN input bar inside ``[s, t]`` poisons the aggregate until the next event; an overflow
    # reads NaN. ``sum``/``max``/``min``/``mean`` (mean = sum / (t − s + 1)). Opening range is
    # ``event_agg(session_open, high, max)``; VWAP since an event is
    # ``event_agg(E, close × volume, sum) / event_agg(E, volume, sum)``; "C has held on any bar
    # since S" is ``event_agg(S, mask(C), max) > 0``. Counts one level over ``input``.
    type: Literal["event_agg"] = "event_agg"
    event: Condition
    input: Series
    agg: Literal["sum", "max", "min", "mean"]


class Native(_Strict):
    # A transform evaluated on an external feed's OWN clock — over its native prints, in print
    # order — and only then anchored onto the bars with the same backward-asof rule a raw feed
    # uses. ``rolling_agg(external(eps), 8, std)`` counts eight BARS of the forward-filled feed
    # (eight days of the same quarterly print); ``native(eps, rolling_agg(eps, 8, std))`` counts
    # eight RELEASES, and every print between two bars is seen (the asof collapse would keep the
    # last one only). Availability-honest by construction: the value at native print ``i`` is
    # computed from prints ``<= i`` and becomes usable at the first bar stamped at-or-after print
    # ``i``'s (post-``lag``) availability time. ``expr`` reads exactly one feed — ``external(name)``
    # with THIS node's name — and constants, through the single-input time transforms,
    # ``binary_op`` and ``rolling_corr``; everything that reads the bar clock (a target field, a
    # calendar attribute, a feed age, a cross-section, an embedded Condition) or another feed is
    # refused, as is a nested ``native``. Counts one level over ``expr``. The feed is evaluated
    # over its WHOLE print history (feeds are not sliced by ``data.start``/``end``; the anchored
    # bars are) — causal on print order, consistent with the raw asof anchoring and ``days_since``.
    # The engine has no session calendar: resampling the TARGET's own bars (weekly from daily)
    # still needs a caller-consolidated feed stamped at period completion.
    type: Literal["native"] = "native"
    name: str
    expr: Series

    @model_validator(mode="after")
    def _check_expr_reads_one_feed_clock(self) -> Native:
        # A local pre-order walk (this module cannot import ``traverse``): every node under
        # ``expr`` must be evaluable on the feed's print sequence alone.
        allowed = (
            f"allowed inside a native expr: external({self.name!r}), constant, the single-input "
            "time transforms (ema/zscore/percentile/rolling_agg/drawdown/runup/"
            "bars_since_extremum/change/shift/unary_op), binary_op, rolling_corr"
        )
        reads_feed = False
        stack: list[Series] = [self.expr]
        while stack:
            node = stack.pop()
            reason: str | None = None
            if isinstance(node, Field):
                reason = (
                    "a target field reads the BAR clock — and drawdown/runup/bars_since_extremum "
                    "default 'input' to close: pass 'input' explicitly"
                )
            elif isinstance(node, External):
                if node.name != self.name:
                    reason = "a native expr reads exactly one feed, on that feed's own clock"
                else:
                    reads_feed = True
            elif isinstance(node, (Calendar, DaysSince)):
                reason = "properties of the bar index"
            elif isinstance(node, (CrossRank, CrossDemean, CrossAgg)):
                reason = "ranks across the targets at a bar"
            elif isinstance(node, (Mask, BarsSinceEvent, EventValue, EventAgg)):
                reason = "embeds a Condition decided on the bar clock"
            elif isinstance(node, Native):
                reason = "does not nest"
            if reason is not None:
                raise ValueError(
                    f"native({self.name!r}) expr contains {node.type!r}: {reason}; {allowed}"
                )
            for attr in ("input", "expr", "left", "right"):
                child = getattr(node, attr, None)
                if child is not None:
                    stack.append(child)
        if not reads_feed:
            raise ValueError(
                f"native({self.name!r}) expr never reads the feed: it must contain "
                f"external({self.name!r}) — a native transform of nothing is a constant"
            )
        return self


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
    | UnaryOp
    | Mask
    | BarsSinceEvent
    | EventValue
    | EventAgg
    | Native,
    PField(discriminator="type"),
]

# No ``model_rebuild()`` here: the Series members that embed a ``Condition`` cannot resolve it from
# this module, so every Series model is rebuilt from ``seikan.dsl.conditions`` once both unions
# exist (see the module docstring).
