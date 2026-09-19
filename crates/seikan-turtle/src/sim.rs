//! The simulator: the state machine driven over bar arrays with the venue's fills priced under
//! the cost model.
//!
//! Per bar, in order: (a) a resting sell stop the opening print gaps through fills at the open;
//! (b) the machine's one print action fills at the open (a buy sized against this very
//! executor's all-in cash-out); (c) a resting sell stop the bar's range trades through fills at
//! its trigger, shocked toward the low; (d) the indicators take the completed bar and the
//! machine takes its close decision. Every fill is reported to the machine with its commission
//! and cost attribution, so the machine's books are the only books.

use crate::coefficients::Coefficients;
use crate::costs::CostModel;
use crate::error::Error;
use crate::execution::{BarContext, Execution, Executor, Side};
use crate::indicators::{AverageVolume, LowestLowChannel, WilderAtr};
use crate::machine::{Fill, FillKind, Ledger, Machine, PrintAction, RoundTrip, StopOrder};

/// Quantized OHLC arrays of equal length, plus the volume when the cost model needs it.
#[derive(Clone, Copy, Debug)]
pub struct BarSeries<'a> {
    /// Opens on the price grid.
    pub open: &'a [f64],
    /// Highs on the price grid.
    pub high: &'a [f64],
    /// Lows on the price grid.
    pub low: &'a [f64],
    /// Closes on the price grid.
    pub close: &'a [f64],
    /// Volumes (required, finite and positive when impact is enabled; otherwise unread).
    pub volume: Option<&'a [f64]>,
}

/// Where a fill happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillAt {
    /// At the opening print (a market order, or a resting stop the print gapped through).
    Open,
    /// At the resting stop's trigger, inside the bar's range.
    Trigger,
}

impl FillAt {
    /// The lowercase name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Trigger => "trigger",
        }
    }
}

/// One venue fill with the machine's state right after it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimFill {
    /// The fill as the machine booked it.
    pub fill: Fill,
    /// Where it happened.
    pub at: FillAt,
    /// The exit reason of an exit, `None` for a buy.
    pub reason: Option<&'static str>,
    /// The price an ideal market would have filled at.
    pub reference: f64,
    /// Cash after the fill.
    pub cash_after: f64,
    /// Shares held after the fill.
    pub shares_after: u64,
    /// Units held after the fill.
    pub units_after: u32,
    /// The stop in force after the fill (`None` when flat).
    pub stop_after: Option<f64>,
}

/// The run: every fill, every round trip, the ledger and per-bar samples taken after each close.
#[derive(Clone, Debug, PartialEq)]
pub struct SimResult {
    /// Every fill, in bar order.
    pub fills: Vec<SimFill>,
    /// The closed round trips.
    pub trips: Vec<RoundTrip>,
    /// The position still open after the last bar, marked.
    pub open_trip: Option<RoundTrip>,
    /// The machine's counters.
    pub ledger: Ledger,
    /// Cash plus the position marked at the close.
    pub equity: Vec<f64>,
    /// Cash after the close.
    pub cash: Vec<f64>,
    /// Shares held after the close.
    pub shares: Vec<u64>,
    /// Units held after the close.
    pub units: Vec<u32>,
    /// The stop in force after each close (NaN when flat).
    pub stop: Vec<f64>,
    /// The next add level after each close (NaN when flat).
    pub add_level: Vec<f64>,
    /// The ATR after each bar (NaN before initialization).
    pub atr: Vec<f64>,
    /// The channel level in force for the NEXT bar (NaN before initialization).
    pub channel: Vec<f64>,
    /// Commission paid, cumulative through each bar's fills.
    pub commission_cum: Vec<f64>,
    /// Slippage attributed, cumulative.
    pub slippage_cum: Vec<f64>,
    /// Stop shock attributed, cumulative.
    pub shock_cum: Vec<f64>,
    /// Market impact attributed, cumulative.
    pub impact_cum: Vec<f64>,
    /// The first bar an entry can be taken on.
    pub first_eligible_bar: usize,
}

/// The running cost totals.
#[derive(Clone, Copy, Debug, Default)]
struct Totals {
    commission: f64,
    slippage: f64,
    shock: f64,
    impact: f64,
}

impl Totals {
    fn add(&mut self, fill: &Fill) {
        self.commission += fill.commission;
        self.slippage += fill.slippage;
        self.shock += fill.shock;
        self.impact += fill.impact;
    }
}

/// Run the machine over `bars` with the entry firing `fired` (one flag per bar) under `costs`.
pub fn simulate(
    c: &Coefficients,
    costs: &CostModel,
    bars: BarSeries<'_>,
    fired: &[bool],
) -> Result<SimResult, Error> {
    costs.validate()?;
    costs.check_warmup(c)?;
    let n = bars.open.len();
    if bars.high.len() != n || bars.low.len() != n || bars.close.len() != n || fired.len() != n {
        return Err(Error::Input(
            "open/high/low/close/fired must have equal lengths".into(),
        ));
    }
    if n == 0 {
        return Err(Error::Input("no bars".into()));
    }
    let volume = match (costs.impact.enabled(), bars.volume) {
        (false, _) => None,
        (true, None) => {
            return Err(Error::Input(
                "market impact is enabled but no volume series was given".into(),
            ));
        }
        (true, Some(v)) => {
            if v.len() != n {
                return Err(Error::Input("volume must have the bars' length".into()));
            }
            if let Some((t, bad)) = v
                .iter()
                .enumerate()
                .find(|(_, x)| !(x.is_finite() && **x > 0.0))
            {
                return Err(Error::Input(format!(
                    "market impact needs a finite, positive volume on every bar; bar {t} has {bad}"
                )));
            }
            Some(v)
        }
    };
    let mut m = Machine::new(c.clone())?;
    let executor = Executor::new(*costs, c.price_precision);
    let mut atr = WilderAtr::new(c.atr_period);
    let mut channel = LowestLowChannel::new(c.exit_lookback);
    let mut adv = AverageVolume::new(costs.impact.adv_window);
    let mut totals = Totals::default();
    let mut out = SimResult {
        fills: Vec::new(),
        trips: Vec::new(),
        open_trip: None,
        ledger: Ledger::default(),
        equity: Vec::with_capacity(n),
        cash: Vec::with_capacity(n),
        shares: Vec::with_capacity(n),
        units: Vec::with_capacity(n),
        stop: Vec::with_capacity(n),
        add_level: Vec::with_capacity(n),
        atr: Vec::with_capacity(n),
        channel: Vec::with_capacity(n),
        commission_cum: Vec::with_capacity(n),
        slippage_cum: Vec::with_capacity(n),
        shock_cum: Vec::with_capacity(n),
        impact_cum: Vec::with_capacity(n),
        first_eligible_bar: m.first_eligible_bar(),
    };

    for (t, &fired_t) in fired.iter().enumerate() {
        let open = bars.open[t];
        let ctx = BarContext {
            open,
            low: bars.low[t],
            n: atr.value(),
            adv: volume.and_then(|_| adv.value()),
        };
        // A fill is possible this bar only in position (exits) or with a buy pending; the rules
        // place both after every indicator's warmup, and this makes that a checked fact.
        if m.in_position() || !matches!(m.pending(), crate::machine::Pending::None) {
            executor.check_context(&ctx)?;
        }
        // (a) The venue sees the print before the strategy: a resting stop the open gaps through
        // fills at the open.
        if let Some(StopOrder {
            trigger,
            shares,
            reason,
        }) = m.resting()
            && open <= trigger
        {
            let exec = executor.market(Side::Sell, shares, open, &ctx)?;
            book(
                &mut m,
                &mut out,
                &mut totals,
                t,
                FillKind::Exit,
                &exec,
                FillAt::Open,
                Some(reason.as_str()),
            )?;
        }
        // (b) The strategy's one action at the print, filled at the open; a buy is sized against
        // this executor's own all-in cash-out (unreachable errors read as unaffordable, and
        // `check_context` above has already ruled them out for this bar).
        let cash_out = |q: u64| {
            executor
                .market(Side::Buy, q, open, &ctx)
                .map_or(f64::INFINITY, |e| e.cash_out())
        };
        match m.on_print(t, open, &cash_out) {
            PrintAction::Hold => {}
            PrintAction::Buy { shares, kind } => {
                let exec = executor.market(Side::Buy, shares, open, &ctx)?;
                if exec.cash_out() > m.cash() + 1e-9 * open {
                    return Err(Error::Bookkeeping(format!(
                        "a buy of {shares} shares sized at {} does not fit the cash {}",
                        exec.cash_out(),
                        m.cash()
                    )));
                }
                book(
                    &mut m,
                    &mut out,
                    &mut totals,
                    t,
                    kind,
                    &exec,
                    FillAt::Open,
                    None,
                )?;
            }
            PrintAction::Sell { shares, reason } => {
                let exec = executor.market(Side::Sell, shares, open, &ctx)?;
                book(
                    &mut m,
                    &mut out,
                    &mut totals,
                    t,
                    FillKind::Exit,
                    &exec,
                    FillAt::Open,
                    Some(reason.as_str()),
                )?;
            }
        }
        // (c) The bar's range against the stop resting through it: fills at the trigger, shocked
        // toward the low.
        if let Some(StopOrder {
            trigger,
            shares,
            reason,
        }) = m.resting()
            && bars.low[t] <= trigger
        {
            let exec = executor.stop(shares, trigger, &ctx)?;
            book(
                &mut m,
                &mut out,
                &mut totals,
                t,
                FillKind::Exit,
                &exec,
                FillAt::Trigger,
                Some(reason.as_str()),
            )?;
        }
        // (d) Indicators over the completed bar, then the close decision.
        atr.update_raw(bars.high[t], bars.low[t], bars.close[t]);
        channel.update_raw(bars.low[t]);
        if let Some(v) = volume {
            adv.update_raw(v[t]);
        }
        let n_now = atr.value();
        let channel_now = channel.value();
        m.on_bar(t, bars.close[t], n_now, channel_now, fired_t, t + 1 == n);
        out.equity.push(m.equity(bars.close[t]));
        out.cash.push(m.cash());
        out.shares.push(m.shares());
        out.units.push(m.units());
        out.stop.push(m.stop().unwrap_or(f64::NAN));
        out.add_level.push(m.add_level().unwrap_or(f64::NAN));
        out.atr.push(n_now.unwrap_or(f64::NAN));
        out.channel.push(channel_now.unwrap_or(f64::NAN));
        out.commission_cum.push(totals.commission);
        out.slippage_cum.push(totals.slippage);
        out.shock_cum.push(totals.shock);
        out.impact_cum.push(totals.impact);
    }
    out.open_trip = m.finish(n - 1, bars.close[n - 1]);
    out.trips = m.closed().to_vec();
    out.ledger = m.ledger().clone();
    Ok(out)
}

/// Report one priced execution to the machine and record it with the state after it.
#[allow(clippy::too_many_arguments)]
fn book(
    m: &mut Machine,
    out: &mut SimResult,
    totals: &mut Totals,
    bar: usize,
    kind: FillKind,
    exec: &Execution,
    at: FillAt,
    reason: Option<&'static str>,
) -> Result<(), Error> {
    let fill = Fill {
        bar,
        kind,
        shares: exec.shares,
        price: exec.price,
        commission: exec.commission,
        slippage: exec.slippage,
        shock: exec.shock,
        impact: exec.impact,
    };
    m.on_fill(fill)?;
    totals.add(&fill);
    out.fills.push(SimFill {
        fill,
        at,
        reason,
        reference: exec.reference,
        cash_after: m.cash(),
        shares_after: m.shares(),
        units_after: m.units(),
        stop_after: m.stop(),
    });
    Ok(())
}
