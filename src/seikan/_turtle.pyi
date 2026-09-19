"""Type stub for the ``seikan._turtle`` extension module — the Turtle simulation engine compiled
from ``crates/seikan-turtle`` (PyO3). Every name here exists on the extension and vice versa
(``tests/test_turtle_stubs.py`` pins both directions)."""

from collections.abc import Sequence

def version() -> str:
    """The engine's version (the crate version)."""

def quantize(price: float, precision: int) -> float:
    """Round ``price`` to ``precision`` decimals."""

def simulate(
    coefficients: Coefficients,
    costs: CostModel,
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float] | None,
    fired: Sequence[bool],
) -> SimResult:
    """Run one target over quantized OHLC sequences (``volumes`` needed only when impact is
    enabled) with the per-bar firing flags. ``ValueError`` on unusable input or coefficients,
    ``RuntimeError`` on a broken bookkeeping invariant (a seikan bug)."""

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

class CostModel:
    """The trade-cost model, validated at construction (``ValueError``)."""

    per_share: float
    min_per_order: float
    bps: float
    sell_bps: float
    cap_bps: float | None
    slippage_bps: float
    slippage_n_fraction: float
    impact_coefficient: float
    adv_window: int
    stop_shock: float
    def __init__(
        self,
        *,
        per_share: float,
        min_per_order: float,
        bps: float,
        sell_bps: float,
        cap_bps: float | None,
        slippage_bps: float,
        slippage_n_fraction: float,
        impact_coefficient: float,
        adv_window: int,
        stop_shock: float,
    ) -> None: ...
    def impact_enabled(self) -> bool: ...

class Fill:
    """One fill with the engine's books right after it; ``at`` is ``open`` or ``trigger``."""

    bar: int
    kind: str
    shares: int
    price: float
    reference: float
    commission: float
    slippage: float
    shock: float
    impact: float
    at: str
    reason: str | None
    cash_after: float
    shares_after: int
    units_after: int
    stop_after: float | None

class RoundTrip:
    """One position from entry to exit (``is_open`` when marked at the last bar); ``pnl`` is net
    of commission, ``gross_pnl`` at the fill prices."""

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
    commission: float
    slippage: float
    shock: float
    impact: float
    gross_pnl: float
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
    """One target's run: fills, round trips, the ledger and per-bar samples."""

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
    commission_cum: list[float]
    slippage_cum: list[float]
    shock_cum: list[float]
    impact_cum: list[float]
    first_eligible_bar: int
