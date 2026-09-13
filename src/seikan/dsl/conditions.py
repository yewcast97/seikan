"""The condition vocabulary: threshold, the boolean combinators, rolling, first_true, lag, and
the ``Condition`` union — plus the ONE rebuild site for both vocabularies (see the bottom).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field as PField
from pydantic import model_validator

from seikan.dsl import nodes
from seikan.dsl.nodes import NonNegIntParam, PosInt, PosIntParam, Series, _Strict


class ThresholdCondition(_Strict):
    type: Literal["threshold"] = "threshold"
    left: Series
    op: Literal["<", "<=", ">", ">=", "==", "!="]
    right: Series


class AndCondition(_Strict):
    type: Literal["and"] = "and"
    conditions: list[Condition] = PField(min_length=2)


class OrCondition(_Strict):
    type: Literal["or"] = "or"
    conditions: list[Condition] = PField(min_length=2)


class NotCondition(_Strict):
    type: Literal["not"] = "not"
    condition: Condition


class RollingCondition(_Strict):
    type: Literal["rolling"] = "rolling"
    # A list ``window`` sweeps the trailing-window length as its own ``rolling_window`` result axis
    # (like a transform-window sweep), taking part in the same Cartesian product. Auto-named, so —
    # unlike a swept ``Constant`` — no ``name`` field is needed.
    window: PosIntParam
    # ``all``/``any`` fire when the inner condition held on every / at least one bar of the window;
    # ``count`` fires when it held on at least ``min_count`` bars — an "at least K of N" trigger
    # (e.g. ">= 3 of the last 5 bars closed down", a sustained-regime signal ``all`` can't state).
    # ``min_count`` is required for — and only valid with — ``count``; keep it scalar (the window
    # already sweeps via ``rolling_window``).
    agg: Literal["all", "any", "count"]
    min_count: PosInt | None = None
    condition: Condition

    @model_validator(mode="after")
    def _check_min_count(self) -> RollingCondition:
        if self.agg == "count":
            if self.min_count is None:
                raise ValueError(
                    "rolling agg='count' requires 'min_count' (the K of 'at least K of N')"
                )
            floor = min(self.window) if isinstance(self.window, list) else self.window
            if self.min_count > floor:
                raise ValueError(
                    f"rolling 'min_count' ({self.min_count}) exceeds the window ({floor}); the "
                    f"count condition could never fire"
                )
        elif self.min_count is not None:
            raise ValueError("rolling 'min_count' is only valid with agg='count'")
        return self


class FirstTrueCondition(_Strict):
    # Episode entry: fires only on a false→true transition of the child's TRADABLE signal
    # (``value & init``). The first True bar after warmup does NOT count as a transition (must have
    # seen an initialized False first), so a regime that is already true when the child warms up
    # does not phantom-fire. Optional ``cooldown`` suppresses re-fires for K bars after a fire
    # (0 = every transition; a list sweeps as ``first_true_cooldown``). The episode-entry
    # primitive: measure forward return from the bar a regime is first entered (deep drawdown,
    # end-of-bull risk alarm, …), not every bar inside it. Also the crossover recipe:
    # ``first_true(threshold(fast > slow))`` — the DSL has no dedicated ``cross`` condition.
    type: Literal["first_true"] = "first_true"
    condition: Condition
    cooldown: NonNegIntParam = 0


class LagCondition(_Strict):
    # The child's decision ``periods`` bars ago, all three channels shifted back together: value,
    # init AND defined read ``[t − k]`` (a hole moves with the decision it affects), and the
    # leading ``k`` bars are warmup (value False, init False, defined True). Backward-only
    # (``periods >= 1``), so it can never read the future; a list sweeps as ``lag_periods``. The
    # typed temporal primitive: strict ordering "A held on a PRIOR bar, then B now" is
    # ``and(rolling(any, lag(A, 1), N), B)`` — where ``rolling(any, A, N)`` includes the current
    # bar and fires when A and B first turn true together. A ``periods`` at or beyond the index
    # length never initializes (not refusable at parse — the index is unknown there).
    type: Literal["lag"] = "lag"
    condition: Condition
    periods: PosIntParam = 1


Condition = Annotated[
    ThresholdCondition
    | AndCondition
    | OrCondition
    | NotCondition
    | RollingCondition
    | FirstTrueCondition
    | LagCondition,
    PField(discriminator="type"),
]


# The ONE rebuild site for both vocabularies. Series and Condition are mutually recursive — every
# threshold reads Series operands, and the event-anchor nodes (``mask`` / ``bars_since_event`` /
# ``event_value`` / ``event_agg``) embed a Condition — so neither union can be completed in its
# own module. Each model resolves ``Series`` from its own module's globals and ``Condition`` from
# the namespace handed in here (``_types_namespace`` is explicit rather than frame-depth-relative,
# so the resolution does not depend on where this module happens to be imported from).
_NAMESPACE = {"Series": Series, "Condition": Condition}
for _model in (
    nodes.EMA,
    nodes.ZScore,
    nodes.Percentile,
    nodes.RollingAgg,
    nodes.Drawdown,
    nodes.Runup,
    nodes.BarsSinceExtremum,
    nodes.Change,
    nodes.Shift,
    nodes.RollingCorr,
    nodes.CrossRank,
    nodes.CrossDemean,
    nodes.CrossAgg,
    nodes.BinaryOp,
    nodes.UnaryOp,
    nodes.Mask,
    nodes.BarsSinceEvent,
    nodes.EventValue,
    nodes.EventAgg,
    nodes.Native,
    ThresholdCondition,
    AndCondition,
    OrCondition,
    NotCondition,
    RollingCondition,
    FirstTrueCondition,
    LagCondition,
):
    _model.model_rebuild(_types_namespace=_NAMESPACE)
