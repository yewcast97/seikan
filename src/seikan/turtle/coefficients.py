"""The Turtle coefficients document — the second JSON the command takes.

The rules' braced defaults, the required ``equity``, the two trigger modes the rules leave to the
caller, three simulation facts (the annualization clock, the account currency, the price grid)
and the ``costs`` block: the commission schedule, the slippage, the market-impact law and the
stop-fill shock the engine prices every fill under. Strict, frozen, ``extra="forbid"`` like the
thesis DSL: an unknown key refuses rather than silently meaning nothing, and every field is
stamped resolved into the report's ``identity`` so a document that relied on a default and one
that spelled it out read as ONE coefficient set — ``canonical_coefficients_hash`` makes that
identity a string.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: A price-triggered rule fires on the bar's close (executed at the next opening print) or on a
#: trade through the level (a resting stop at the venue).
type TriggerMode = Literal["close", "trade"]

#: Which N sets the stop after an add.
type StopNSource = Literal["entry", "current"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class Commission(_Strict):
    """A per-fill commission schedule: ``max(per_share × shares + bps × notional, min_per_order)``,
    capped at ``cap_bps × notional`` when a cap is set, plus ``sell_bps × notional`` on sells (a
    stamp duty or transaction fee the cap never applies to). The defaults are a US retail-pro
    fixed schedule: half a cent a share, a dollar minimum, one percent of trade value at most."""

    #: Currency per share, every fill.
    per_share: float = Field(default=0.005, ge=0)
    #: The floor per fill.
    min_per_order: float = Field(default=1.0, ge=0)
    #: Basis points of the notional, every fill (percentage brokers).
    bps: float = Field(default=0.0, ge=0)
    #: Basis points of the notional on sells only, never capped.
    sell_bps: float = Field(default=0.0, ge=0)
    #: The cap per fill in basis points of the notional; ``null`` leaves fills uncapped.
    cap_bps: float | None = Field(default=100.0, gt=0)


class Slippage(_Strict):
    """Adverse slippage per share on every fill: ``reference × bps/1e4 + n_fraction × N``. At an
    opening print the open IS the print, so ``bps`` models auction-participation uncertainty;
    ``n_fraction`` is the Turtle-native "slippage in N"."""

    #: Basis points of the reference price.
    bps: float = Field(default=5.0, ge=0)
    #: A fraction of N, additive.
    n_fraction: float = Field(default=0.0, ge=0)


class Impact(_Strict):
    """Square-root market impact per share: ``coefficient × N × sqrt(shares / ADV)`` — the
    ``Y·σ·√(Q/ADV)`` law with N standing in for σ (N ≈ 1.6 × daily σ on a random walk, so a
    σ-calibrated Y scales by ≈ 0.6). Off by default; a positive coefficient needs a volume column
    on every target and an ``adv_window`` the rules' warmup covers."""

    #: The coefficient, in units of N; 0 switches impact off.
    coefficient: float = Field(default=0.0, ge=0)
    #: Bars in the average volume.
    adv_window: int = Field(default=20, ge=1)


class Costs(_Strict):
    """The cost model every fill is priced under."""

    commission: Commission = Commission()
    slippage: Slippage = Slippage()
    impact: Impact = Impact()
    #: The fraction of the bar's adverse continuation beyond a stop's trigger the fill gives up
    #: (0 fills at the trigger, 1 at the bar's low): the ignorance midpoint by default.
    stop_shock: float = Field(default=0.5, ge=0, le=1)


class TurtleCoefficients(_Strict):
    """The coefficient set, validated once. Field order is the report's ``identity.coefficients``
    order."""

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
    #: The trade-cost model.
    costs: Costs = Costs()

    @model_validator(mode="after")
    def _impact_window_within_warmup(self) -> TurtleCoefficients:
        # Warmup is a rule quantity (max(atr_period, exit_lookback) − 1) and no cost knob may
        # move it, so an enabled impact must have its ADV initialized by then.
        impact = self.costs.impact
        floor = max(self.atr_period, self.exit_lookback)
        if impact.coefficient > 0 and impact.adv_window > floor:
            raise ValueError(
                f"costs.impact.adv_window ({impact.adv_window}) must not exceed "
                f"max(atr_period, exit_lookback) ({floor}) when impact is enabled"
            )
        return self


def canonical_coefficients_hash(document: dict[str, object]) -> str:
    """The identity of a coefficients document: validate, fill defaults, sort keys, sha256 the
    compact JSON — so omitted and explicit defaults hash alike (a ``ValidationError`` propagates
    for an invalid document)."""
    resolved = TurtleCoefficients.model_validate(document).model_dump(mode="json")
    payload = json.dumps(resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
