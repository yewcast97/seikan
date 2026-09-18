"""Type stub for the ``seikan._turtle`` extension module — the Turtle kernel compiled from
``crates/seikan-turtle`` (PyO3). Every name here exists on the extension and vice versa
(``tests/test_turtle_stubs.py`` pins both directions)."""

from collections.abc import Sequence

def version() -> str:
    """The kernel's version (the crate version)."""

def unit_shares(risk_per_unit: float, budget: float, stop_n: float, n: float) -> int:
    """Shares per unit: ``floor(risk_per_unit × budget / (stop_n × n))``; 0 when ``n`` is not a
    finite positive number."""

def price_below(level: float, precision: int) -> float:
    """The highest grid price strictly below ``level`` at ``precision`` decimals."""

def quantize(price: float, precision: int) -> float:
    """Round ``price`` to ``precision`` decimals."""

def first_eligible_bar(atr_period: int, exit_lookback: int) -> int:
    """``max(atr_period, exit_lookback) - 1`` — the first bar an entry can be taken on."""

def simulate_reference(
    coefficients: Coefficients,
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    fired: Sequence[bool],
) -> SimResult:
    """Run the reference simulator (idealized fills) over quantized OHLC sequences."""

class Coefficients:
    """The rule coefficients for one target, validated at construction (``ValueError``)."""

    atr_period: int
    add_step_n: float
    max_units: int
    stop_n: float
    exit_lookback: int
    risk_per_unit: float
    stop_trigger: str
    exit_trigger: str
    stop_n_source: str
    price_precision: int
    budget: float
    def __init__(
        self,
        *,
        atr_period: int,
        add_step_n: float,
        max_units: int,
        stop_n: float,
        exit_lookback: int,
        risk_per_unit: float,
        stop_trigger: str,
        exit_trigger: str,
        stop_n_source: str,
        price_precision: int,
        budget: float,
    ) -> None: ...
    def first_eligible_bar(self) -> int: ...

class WilderAtr:
    """Wilder's average true range — the Turtle N; ``value`` is NaN before initialization."""

    name: str
    period: int
    count: int
    initialized: bool
    has_inputs: bool
    value: float
    def __init__(self, period: int) -> None: ...
    def update_raw(self, high: float, low: float, close: float) -> None: ...
    def handle_bar(self, bar: object) -> None: ...
    def reset(self) -> None: ...

class LowestLowChannel:
    """The lowest low of the last ``lookback`` bars handled (the newest included)."""

    name: str
    lookback: int
    count: int
    initialized: bool
    has_inputs: bool
    value: float
    def __init__(self, lookback: int) -> None: ...
    def update_raw(self, low: float) -> None: ...
    def handle_bar(self, bar: object) -> None: ...
    def reset(self) -> None: ...

class StopOrder:
    """The one sell stop to keep resting at the venue."""

    trigger: float
    shares: int
    reason: str

class PrintAction:
    """What to submit at an opening print: ``kind`` is ``hold``/``buy``/``sell``."""

    kind: str
    shares: int
    intent: str | None
    reason: str | None

class Fill:
    """One reference-simulator fill; ``at`` is ``open`` or ``trigger``."""

    bar: int
    kind: str
    shares: int
    price: float
    at: str
    reason: str | None

class RoundTrip:
    """One position from entry to exit (``is_open`` when marked at the last bar)."""

    entry_bar: int
    exit_bar: int
    exit_reason: str
    is_open: bool
    entry_px: float
    exit_px: float
    unit_shares: int
    units: int
    shares: int
    cost_basis: float
    proceeds: float
    pnl: float
    n_adds: int
    adds_skipped_budget: int
    max_stop: float

class Ledger:
    """Everything the machine did and declined to do, counted."""

    entries: int
    adds: int
    exits_stop_close: int
    exits_stop_gap: int
    exits_stop_trade: int
    exits_channel_close: int
    exits_channel_trade: int
    adds_skipped_budget: int
    adds_filled_below_level: int
    entries_skipped_warmup: int
    entries_skipped_in_position: int
    entries_skipped_zero_size: int
    entries_skipped_budget: int
    entries_skipped_end_of_data: int
    entries_cash_capped: int
    stop_holds: int
    def to_dict(self) -> dict[str, int]: ...

class SimResult:
    """A reference-simulator run: fills, round trips, the ledger and per-bar samples."""

    fills: list[Fill]
    trips: list[RoundTrip]
    open_trip: RoundTrip | None
    ledger: Ledger
    equity: list[float]
    cash: list[float]
    shares: list[int]
    units: list[int]
    stop: list[float]
    add_level: list[float]
    atr: list[float]
    channel: list[float]
    first_eligible_bar: int

class Machine:
    """The long-only Turtle state machine for one target."""

    cash: float
    shares: int
    units: int
    in_position: bool
    stop: float | None
    add_level: float | None
    last_fill: float | None
    entry_px: float | None
    n_entry: float | None
    channel: float | None
    pending: str
    first_eligible_bar: int
    ledger: Ledger
    closed: list[RoundTrip]
    def __init__(self, coefficients: Coefficients) -> None: ...
    def on_print(self, bar: int, open_price: float) -> PrintAction: ...
    def on_fill(self, bar: int, kind: str, shares: int, price: float) -> None: ...
    def on_bar(
        self,
        bar: int,
        close: float,
        n: float | None = None,
        channel: float | None = None,
        fired: bool = False,
        last_bar: bool = False,
    ) -> None: ...
    def finish(self, last_bar: int, last_close: float) -> RoundTrip | None: ...
    def equity(self, close: float) -> float: ...
    def resting(self) -> StopOrder | None: ...
