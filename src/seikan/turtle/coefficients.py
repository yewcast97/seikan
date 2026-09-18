"""The Turtle coefficients document — the second JSON the command takes.

The rules' braced defaults, the required ``equity``, the two trigger modes the rules leave to the
caller, and three simulation facts (the annualization clock, the account currency, the price
grid). Strict, frozen, ``extra="forbid"`` like the thesis DSL: an unknown key refuses rather than
silently meaning nothing, and every field is stamped resolved into the report's ``identity`` so a
document that relied on a default and one that spelled it out read as ONE coefficient set —
``canonical_coefficients_hash`` makes that identity a string.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: A price-triggered rule fires on the bar's close (executed at the next opening print) or on a
#: trade through the level (a resting stop at the venue).
type TriggerMode = Literal["close", "trade"]

#: Which N sets the stop after an add.
type StopNSource = Literal["entry", "current"]


class TurtleCoefficients(BaseModel):
    """The coefficient set, validated once. Field order is the report's ``identity.coefficients``
    order."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    #: Total money. Divided evenly across the thesis's targets: each target's budget is the
    #: sizing base and the cash cap of its own sub-account.
    equity: float = Field(gt=0)
    #: N = the Wilder ATR over this many bars.
    atr_period: int = Field(default=20, ge=1)
    #: Add one unit each time price closes this many N above the last fill.
    add_step_n: float = Field(default=0.5, gt=0)
    #: Stop adding once the position holds this many units.
    max_units: int = Field(default=3, ge=1)
    #: The stop sits this many N below the latest fill.
    stop_n: float = Field(default=2.0, gt=0)
    #: Exit on a close (or trade) below the lowest low of this many completed bars.
    exit_lookback: int = Field(default=20, ge=1)
    #: Fraction of the target's budget risked per unit between the fill and the initial stop:
    #: ``X = floor(risk_per_unit × budget / (stop_n × N))``.
    risk_per_unit: float = Field(default=0.01, gt=0, le=1)
    #: How the stop-loss triggers: ``close`` (default; sold at the next open) or ``trade``.
    stop_trigger: TriggerMode = "close"
    #: How the channel exit triggers: ``close`` (default) or ``trade``.
    exit_trigger: TriggerMode = "close"
    #: After an add, the stop uses the ATR current at the add's signal bar (``current``, the
    #: default reading of "N is recalculated only for the stop") or the entry ATR (``entry``).
    stop_n_source: StopNSource = "current"
    #: Bars per year for the annualized metrics (252 for daily bars).
    bars_per_year: int = Field(default=252, ge=1)
    #: The simulated account's currency code — a label for the money; key names are opaque.
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")
    #: Decimal places of the price grid every fill, level and stop is quantized to.
    price_precision: int = Field(default=4, ge=0, le=9)


def canonical_coefficients_hash(document: dict[str, object]) -> str:
    """The identity of a coefficients document: validate, fill defaults, sort keys, sha256 the
    compact JSON — so omitted and explicit defaults hash alike (a ``ValidationError`` propagates
    for an invalid document)."""
    resolved = TurtleCoefficients.model_validate(document).model_dump(mode="json")
    payload = json.dumps(resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
