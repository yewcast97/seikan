"""The condition vocabulary: threshold, the boolean combinators, rolling, first_true, and the
``Condition`` union.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field as PField
from pydantic import model_validator

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


Condition = Annotated[
    ThresholdCondition
    | AndCondition
    | OrCondition
    | NotCondition
    | RollingCondition
    | FirstTrueCondition,
    PField(discriminator="type"),
]


# Forward references resolve against THIS module's namespace — the unions and their
# members must rebuild where they are defined.
ThresholdCondition.model_rebuild()
AndCondition.model_rebuild()
OrCondition.model_rebuild()
NotCondition.model_rebuild()
RollingCondition.model_rebuild()
FirstTrueCondition.model_rebuild()
