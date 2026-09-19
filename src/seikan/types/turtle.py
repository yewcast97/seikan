"""Emitted-shape declarations: the ``stock-turtle-trade-long-only`` report."""

from __future__ import annotations

from typing import TypedDict

from seikan.types.run import BarSpacing
from seikan.types.scalars import ParamValue

# ---- performance ----------------------------------------------------------------------------


class DrawdownBlock(TypedDict):
    """The deepest drawdown's geometry — ``turtle.metrics.performance``. Every timestamp is null
    when the curve never fell below its running peak; ``peak_time`` is null when the peak was
    the pre-data starting equity."""

    peak_time: str | None
    trough_time: str | None
    recovery_time: str | None
    bars_to_trough: int | None
    bars_to_recovery: int | None
    longest_drawdown_bars: int
    longest_drawdown_open: bool


class ExposureBlock(TypedDict):
    """Time and size in the market over the curve's bars."""

    bars_in_market: int
    fraction_in_market: float
    mean_gross_exposure: float
    max_gross_exposure: float


class PerformanceMetrics(TypedDict):
    """One equity curve's metrics (portfolio, benchmark or one target's sub-account) —
    ``turtle.metrics.performance``. Ratios that need a nonzero denominator are null when it is
    zero; the moments are null on fewer than three bars or zero dispersion."""

    start_equity: float
    end_equity: float
    net_pnl: float
    total_return: float
    cagr: float | None
    annualized_volatility: float | None
    sharpe: float | None
    sortino: float | None
    calmar: float | None
    max_drawdown: float
    drawdown: DrawdownBlock
    best_bar_return: float
    worst_bar_return: float
    mean_bar_return: float
    median_bar_return: float
    skewness: float | None
    kurtosis: float | None
    positive_bars_fraction: float
    exposure: ExposureBlock
    n_bars: int
    bars_per_year: int


class RelativeMetrics(TypedDict):
    """A curve's bar returns against the benchmark's — ``turtle.metrics.relative``."""

    beta: float | None
    alpha_bar: float | None
    alpha_annualized: float | None
    correlation: float | None
    r2: float | None
    tracking_error: float | None
    information_ratio: float | None
    excess_total_return: float
    excess_cagr: float | None
    up_capture: float | None
    down_capture: float | None
    n_up_bars: int
    n_down_bars: int


# ---- costs ----------------------------------------------------------------------------------


class CostsPaid(TypedDict):
    """The four cost buckets summed, in the account currency: over every fill of a run
    (``end_state.costs_paid`` / ``portfolio.costs_paid``, the cash fact) or over the closed round
    trips (``trades.costs``). ``total`` is their sum — what the same fills would have been worth
    more at their reference prices, net of commission."""

    commission: float
    slippage: float
    shock: float
    impact: float
    total: float


# ---- trades ---------------------------------------------------------------------------------


class ExitCounts(TypedDict):
    stop_close: int
    stop_gap: int
    stop_trade: int
    channel_close: int
    channel_trade: int
    end_of_data: int


class EntrySkips(TypedDict):
    warmup: int
    in_position: int
    zero_size: int
    budget: int
    end_of_data: int


class EntryCounts(TypedDict):
    taken: int
    cash_capped: int
    skipped: EntrySkips


class AddSkips(TypedDict):
    budget: int


class AddCounts(TypedDict):
    taken: int
    filled_below_level: int
    skipped: AddSkips


class TradeStats(TypedDict):
    """Round-trip statistics over CLOSED trips plus the engine's ledger counts —
    ``turtle.metrics.trade_stats``. An open end-of-data position is counted in ``n_open`` and
    ``exits.end_of_data`` and enters no other statistic. ``pnl``-based reads are NET of
    commission; ``gross_pnl`` is proceeds − cost_basis at the fill prices."""

    n_round_trips: int
    n_open: int
    n_wins: int
    n_losses: int
    n_flat: int
    win_rate: float | None
    gross_profit: float
    gross_loss: float
    net_pnl: float
    gross_pnl: float
    costs: CostsPaid
    turnover: float
    cost_bps_of_turnover: float | None
    profit_factor: float | None
    expectancy: float | None
    expectancy_ret: float | None
    avg_win: float | None
    avg_loss: float | None
    largest_win: float | None
    largest_loss: float | None
    win_loss_ratio: float | None
    avg_bars_held: float | None
    median_bars_held: float | None
    max_bars_held: int | None
    avg_units_at_exit: float | None
    max_units_reached: int
    avg_adds_per_trip: float | None
    mean_mae: float | None
    mean_mfe: float | None
    exits: ExitCounts
    entries: EntryCounts
    adds: AddCounts
    stop_holds: int


class PeriodicReturns(TypedDict):
    """Compounded returns per calendar month (``YYYY-MM``) and year (``YYYY``)."""

    monthly: dict[str, float]
    annual: dict[str, float]


# ---- cells ----------------------------------------------------------------------------------


class TargetEndState(TypedDict):
    shares: int
    units: int
    cash: float
    market_value: float
    stop: float | None
    add_level: float | None
    in_position: bool
    costs_paid: CostsPaid


class TargetPanel(TypedDict):
    """One target's sub-account inside a cell (its base is the per-target budget)."""

    metrics: PerformanceMetrics
    relative: RelativeMetrics
    trades: TradeStats
    end_state: TargetEndState


class PortfolioPanel(TypedDict):
    """The cell's whole account: every target's sub-account summed bar by bar."""

    metrics: PerformanceMetrics
    relative: RelativeMetrics
    periodic: PeriodicReturns
    trades: TradeStats
    costs_paid: CostsPaid


class TurtleCell(TypedDict):
    """One entry combo's simulation — ``cells[i]``, in declaration order, never ranked."""

    cell_id: str
    params: dict[str, ParamValue]
    portfolio: PortfolioPanel
    by_target: dict[str, TargetPanel]


# ---- run-level blocks -----------------------------------------------------------------------


class BenchmarkBlock(TypedDict):
    """Buy-and-hold of the index over the same bars with the same starting equity."""

    source: str
    construction: str
    units: float
    start_equity: float
    end_equity: float
    metrics: PerformanceMetrics
    periodic: PeriodicReturns


class SimulationBlock(TypedDict):
    """How the engine was set up and what every number is denominated in."""

    engine: str
    engine_version: str
    currency: str
    starting_equity: float
    n_targets: int
    budget_per_target: float
    budget_mode: str
    price_precision: int
    price_increment: float
    lot_size: int
    liquidity: str
    fill_conventions: dict[str, str]
    bars_per_year: int
    n_bars: int
    index_start: str
    index_end: str
    bar_spacing: BarSpacing
    first_eligible_bar: int
    thesis_params_ignored: list[str]


class CommissionBlock(TypedDict):
    per_share: float
    min_per_order: float
    bps: float
    sell_bps: float
    cap_bps: float | None


class SlippageBlock(TypedDict):
    bps: float
    n_fraction: float


class ImpactBlock(TypedDict):
    coefficient: float
    adv_window: int


class CoefficientsCostsBlock(TypedDict):
    """The resolved cost model (``turtle.coefficients.Costs``)."""

    commission: CommissionBlock
    slippage: SlippageBlock
    impact: ImpactBlock
    stop_shock: float


class CoefficientsBlock(TypedDict):
    """The resolved coefficient set (``turtle.coefficients.TurtleCoefficients``), every field."""

    equity: float
    atr_period: int
    add_step_n: float
    max_units: int
    stop_n: float
    exit_lookback: int
    risk_per_unit: float
    stop_trigger: str
    exit_trigger: str
    stop_n_source: str
    bars_per_year: int
    currency: str
    price_precision: int
    costs: CoefficientsCostsBlock
