"""The nautilus_trader strategy for ONE target: the Rust kernel decides, the venue executes.

Per bar the venue sees two events, the opening print (a quote one nanosecond before the bar) and
the bar itself. At the print the strategy asks the machine what to do at this open and submits it
as a MARKET order; the venue fills it whole at the print and delivers the fill right after the
handler returns, still at print time, where the fill handler feeds the machine and makes the
venue's resting sell stop match what the machine wants from now on (a command issued at print
time applies before the bar's replay; one issued DURING a replay would apply only after it, which
is why the fill handler never has to issue one then: a resting-stop fill leaves nothing to
reconcile). The bar's own replay fills that resting stop at its trigger if the range trades
through it. After the bar the registered indicators already hold it, the machine takes its close
decision, and the resting stop is reconciled once more.

Nothing here decides anything: every decision is the kernel's, every fill is the venue's, and
the strategy only carries prices between them and keeps a ledger of what happened. The venue
swallows an exception raised inside a handler (it logs it and carries on), so every handler
runs under :meth:`TurtleTargetStrategy._guard`, which records the failure for the engine to raise
after the run — an invariant broken here is a seikan bug and must never pass as a result.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import OrderSide, TimeInForce
from nautilus_trader.trading import Strategy

from seikan import _turtle
from seikan.turtle.market import OPEN_PRINT_OFFSET_NS, QuantizedBars, bar_type_for


@dataclass(frozen=True)
class FillRecord:
    """One venue fill, with the machine's state right after it."""

    bar: int
    kind: str
    reason: str | None
    at: str
    side: str
    shares: int
    price: float
    client_order_id: str
    cash_after: float
    shares_after: int
    units_after: int
    stop_after: float | None


@dataclass
class TargetSamples:
    """The per-bar samples taken after each close."""

    equity: list[float] = field(default_factory=list)
    cash: list[float] = field(default_factory=list)
    shares: list[int] = field(default_factory=list)
    units: list[int] = field(default_factory=list)
    stop: list[float] = field(default_factory=list)
    add_level: list[float] = field(default_factory=list)
    atr: list[float] = field(default_factory=list)
    channel: list[float] = field(default_factory=list)


class TurtleTargetConfig(StrategyConfig):
    """One target's strategy configuration. ``StrategyConfig`` is a PyO3 type that takes only its
    own keywords, so the base gets ``order_id_tag`` through ``__new__`` and everything else is
    attached as plain attributes."""

    def __new__(cls, *, order_id_tag: str, **_: object) -> TurtleTargetConfig:
        # The stub types the keywords on __init__, the PyO3 class takes them on __new__.
        return super().__new__(cls, order_id_tag=order_id_tag)  # type: ignore[call-arg]

    def __init__(
        self,
        *,
        order_id_tag: str,
        target: str,
        instrument: Any,
        bars: QuantizedBars,
        ts_ns: np.ndarray,
        fired: np.ndarray,
        kernel: _turtle.Coefficients,
    ) -> None:
        self.tag = order_id_tag
        self.target = target
        self.instrument = instrument
        self.bars = bars
        self.ts_ns = ts_ns
        self.fired = fired
        self.kernel = kernel


class TurtleTargetStrategy(Strategy):
    """Drives one :class:`seikan._turtle.Machine` against the venue."""

    def __init__(self, config: TurtleTargetConfig) -> None:
        super().__init__(config)
        self.settings = config
        self.machine = _turtle.Machine(config.kernel)
        self.atr = _turtle.WilderAtr(config.kernel.atr_period)
        self.channel = _turtle.LowestLowChannel(config.kernel.exit_lookback)
        self.fills: list[FillRecord] = []
        self.samples = TargetSamples()
        self.problems: list[str] = []
        self._n_bars = len(config.ts_ns)
        self._k = -1  # the last bar whose close was decided
        self._intent: dict[str, tuple[str, str | None, int]] = {}
        self._resting_coid: Any = None
        self._resting_reason: str | None = None

    @contextmanager
    def _guard(self) -> Iterator[None]:
        """Record a handler failure before the venue swallows it."""
        try:
            yield
        except Exception as exc:
            self.problems.append(f"{self.settings.target}: {type(exc).__name__}: {exc}")
            raise

    # ---- lifecycle ------------------------------------------------------------------------

    def on_start(self) -> None:
        bar_type = bar_type_for(self.settings.instrument)
        self.register_indicator_for_bars(bar_type, self.atr)
        self.register_indicator_for_bars(bar_type, self.channel)
        self.subscribe_bars(bar_type)
        self.subscribe_quotes(self.settings.instrument.id)

    # ---- the opening print --------------------------------------------------------------------

    def on_quote(self, quote: Any) -> None:
        with self._guard():
            k = self._k + 1
            if k >= self._n_bars:
                raise RuntimeError(f"an opening print after the last bar ({k})")
            expected = int(self.settings.ts_ns[k]) - OPEN_PRINT_OFFSET_NS
            if int(quote.ts_event) != expected:
                raise RuntimeError(
                    f"opening print at {quote.ts_event} for bar {k}, expected {expected}"
                )
            if self._intent:
                raise RuntimeError(f"market order(s) {list(self._intent)} never filled")
            action = self.machine.on_print(k, float(quote.ask_price))
            if action.kind == "buy":
                self._submit_market(OrderSide.BUY, action.shares, action.intent or "entry", None)
            elif action.kind == "sell":
                self._submit_market(OrderSide.SELL, action.shares, "exit", action.reason)
            else:
                self._reconcile_resting()

    def _submit_market(self, side: Any, shares: int, kind: str, reason: str | None) -> None:
        instrument = self.settings.instrument
        order = self.order_factory.market(
            instrument_id=instrument.id, order_side=side, quantity=instrument.make_qty(shares)
        )
        self._intent[str(order.client_order_id)] = (kind, reason, shares)
        self.submit_order(order)

    # ---- the close ----------------------------------------------------------------------------

    def on_bar(self, bar: Any) -> None:
        with self._guard():
            k = self._k + 1
            if k >= self._n_bars or int(bar.ts_init) != int(self.settings.ts_ns[k]):
                raise RuntimeError(f"bar at {bar.ts_init} for index {k}")
            if self._intent:
                raise RuntimeError(f"market order(s) {list(self._intent)} never filled")
            self._k = k
            n = self.atr.value if self.atr.initialized else None
            channel = self.channel.value if self.channel.initialized else None
            close = float(bar.close)
            self.machine.on_bar(
                k,
                close,
                n=n,
                channel=channel,
                fired=bool(self.settings.fired[k]),
                last_bar=k == self._n_bars - 1,
            )
            self._reconcile_resting()
            s = self.samples
            s.equity.append(self.machine.equity(close))
            s.cash.append(self.machine.cash)
            s.shares.append(self.machine.shares)
            s.units.append(self.machine.units)
            stop = self.machine.stop
            s.stop.append(float("nan") if stop is None else stop)
            level = self.machine.add_level
            s.add_level.append(float("nan") if level is None else level)
            s.atr.append(float("nan") if n is None else n)
            s.channel.append(float("nan") if channel is None else channel)

    # ---- fills ---------------------------------------------------------------------------------

    def on_order_filled(self, event: Any) -> None:
        with self._guard():
            coid = str(event.client_order_id)
            bar = self._k + 1
            price = float(event.last_px)
            shares = int(event.last_qty)
            at = (
                "open"
                if int(event.ts_event) == int(self.settings.ts_ns[bar]) - OPEN_PRINT_OFFSET_NS
                else "trigger"
            )
            if coid in self._intent:
                kind, reason, expected = self._intent.pop(coid)
                if shares != expected:
                    raise RuntimeError(
                        f"order {coid} filled {shares} of {expected} shares — a partial fill "
                        "under unlimited liquidity"
                    )
            elif self._resting_coid is not None and coid == str(self._resting_coid):
                kind, reason = "exit", self._resting_reason
                self._resting_coid = None
                self._resting_reason = None
            else:
                raise RuntimeError(f"fill for an unknown order {coid}")
            self.machine.on_fill(bar, kind, shares, price)
            self.fills.append(
                FillRecord(
                    bar=bar,
                    kind=kind,
                    reason=reason,
                    at=at,
                    side="buy" if kind != "exit" else "sell",
                    shares=shares,
                    price=price,
                    client_order_id=coid,
                    cash_after=self.machine.cash,
                    shares_after=self.machine.shares,
                    units_after=self.machine.units,
                    stop_after=self.machine.stop,
                )
            )
            # A print fill (entry/add) is delivered at print time, so the command this issues
            # lands before the bar's replay; a resting-stop fill leaves nothing to command.
            self._reconcile_resting()

    def on_order_rejected(self, event: Any) -> None:
        self.problems.append(f"rejected {event.client_order_id}: {getattr(event, 'reason', '')}")

    def on_order_denied(self, event: Any) -> None:
        self.problems.append(f"denied {event.client_order_id}: {getattr(event, 'reason', '')}")

    def on_order_modify_rejected(self, event: Any) -> None:
        self.problems.append(f"modify rejected {event.client_order_id}")

    def on_order_cancel_rejected(self, event: Any) -> None:
        self.problems.append(f"cancel rejected {event.client_order_id}")

    # ---- the resting stop ------------------------------------------------------------------

    def _reconcile_resting(self) -> None:
        """Make the venue's resting sell stop match what the machine wants from now on: submit,
        modify, cancel, or leave alone. A command is only ever issued outside a bar replay (at
        print time or after the bar), where it applies before the next event."""
        want = self.machine.resting()
        have = self.cache.order(self._resting_coid) if self._resting_coid is not None else None
        have_open = have is not None and have.is_open
        if want is None:
            if have_open:
                self.cancel_order(self._resting_coid)
            self._resting_coid = None
            self._resting_reason = None
            return
        instrument = self.settings.instrument
        trigger = instrument.make_price(want.trigger)
        quantity = instrument.make_qty(want.shares)
        if have_open and have is not None:
            same_trigger = float(have.trigger_price) == float(trigger)
            same_qty = int(have.quantity) == int(quantity)
            self._resting_reason = want.reason
            if same_trigger and same_qty:
                return
            # Cancel and resubmit rather than modify: the venue's cash account locks balance on
            # a MODIFIED sell stop (one unit of currency per share) and would later deny a buy
            # the kernel's ledger affords, while a fresh submission locks nothing.
            self.cancel_order(self._resting_coid)
        order = self.order_factory.stop_market(
            instrument_id=instrument.id,
            order_side=OrderSide.SELL,
            quantity=quantity,
            trigger_price=trigger,
            time_in_force=TimeInForce.GTC,
        )
        self._resting_coid = order.client_order_id
        self._resting_reason = want.reason
        self.submit_order(order)
