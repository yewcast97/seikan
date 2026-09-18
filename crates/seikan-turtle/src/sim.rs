//! The reference simulator: the state machine driven over bar arrays with IDEALIZED fills.
//!
//! The fill rules are exactly the ones the simulated venue was observed to apply (and that the
//! Python parity test holds it to): a market order at the opening print fills at the open; a
//! resting sell stop whose trigger the print gaps through fills at the open; a resting sell stop
//! whose trigger lies inside the bar's range fills at the trigger. Because the same [`Machine`]
//! makes every decision, a venue run and this simulator can only differ by execution — which is
//! what the parity test isolates.

use crate::coefficients::Coefficients;
use crate::indicators::{LowestLowChannel, WilderAtr};
use crate::machine::{Fill, FillKind, Ledger, Machine, PrintAction, RoundTrip, StopOrder};

/// Quantized OHLC arrays of equal length.
#[derive(Clone, Copy, Debug)]
pub struct BarSeries<'a> {
    pub open: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub close: &'a [f64],
}

/// Where a reference fill happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillAt {
    /// At the opening print (a market order, or a resting stop the print gapped through).
    Open,
    /// At the resting stop's trigger price, inside the bar's range.
    Trigger,
}

impl FillAt {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Trigger => "trigger",
        }
    }
}

/// One reference fill.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimFill {
    pub fill: Fill,
    pub at: FillAt,
    pub reason: Option<&'static str>,
}

/// The reference run: every fill, every round trip, the ledger and per-bar samples.
#[derive(Clone, Debug, PartialEq)]
pub struct SimResult {
    pub fills: Vec<SimFill>,
    pub trips: Vec<RoundTrip>,
    pub open_trip: Option<RoundTrip>,
    pub ledger: Ledger,
    pub equity: Vec<f64>,
    pub cash: Vec<f64>,
    pub shares: Vec<u64>,
    pub units: Vec<u32>,
    /// The stop in force after each close (NaN when flat).
    pub stop: Vec<f64>,
    /// The next add level after each close (NaN when flat).
    pub add_level: Vec<f64>,
    /// The ATR after each bar (NaN before initialization).
    pub atr: Vec<f64>,
    /// The channel level in force for the NEXT bar (NaN before initialization).
    pub channel: Vec<f64>,
    pub first_eligible_bar: usize,
}

/// Run the machine over `bars` with the entry firing `fired` (one flag per bar).
pub fn simulate(
    c: &Coefficients,
    bars: BarSeries<'_>,
    fired: &[bool],
) -> Result<SimResult, String> {
    let n = bars.open.len();
    if bars.high.len() != n || bars.low.len() != n || bars.close.len() != n || fired.len() != n {
        return Err("open/high/low/close/fired must have equal lengths".into());
    }
    if n == 0 {
        return Err("no bars".into());
    }
    let mut m = Machine::new(c.clone())?;
    let mut atr = WilderAtr::new(c.atr_period);
    let mut channel = LowestLowChannel::new(c.exit_lookback);
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
        first_eligible_bar: m.first_eligible_bar(),
    };

    for (t, &fired_t) in fired.iter().enumerate() {
        let open = bars.open[t];
        // (a) The venue sees the print before the strategy: a resting stop the open gaps through
        // fills at the open.
        if let Some(StopOrder {
            trigger,
            shares,
            reason,
        }) = m.resting()
            && open <= trigger
        {
            let fill = Fill {
                bar: t,
                kind: FillKind::Exit,
                shares,
                price: open,
            };
            m.on_fill(fill)?;
            out.fills.push(SimFill {
                fill,
                at: FillAt::Open,
                reason: Some(reason.as_str()),
            });
        }
        // (b) The strategy's one action at the print, filled at the open.
        match m.on_print(t, open) {
            PrintAction::Hold => {}
            PrintAction::Buy { shares, kind } => {
                let fill = Fill {
                    bar: t,
                    kind,
                    shares,
                    price: open,
                };
                m.on_fill(fill)?;
                out.fills.push(SimFill {
                    fill,
                    at: FillAt::Open,
                    reason: None,
                });
            }
            PrintAction::Sell { shares, reason } => {
                let fill = Fill {
                    bar: t,
                    kind: FillKind::Exit,
                    shares,
                    price: open,
                };
                m.on_fill(fill)?;
                out.fills.push(SimFill {
                    fill,
                    at: FillAt::Open,
                    reason: Some(reason.as_str()),
                });
            }
        }
        // (c) The bar's range against the stop resting through it: fills at the trigger.
        if let Some(StopOrder {
            trigger,
            shares,
            reason,
        }) = m.resting()
            && bars.low[t] <= trigger
        {
            let fill = Fill {
                bar: t,
                kind: FillKind::Exit,
                shares,
                price: trigger,
            };
            m.on_fill(fill)?;
            out.fills.push(SimFill {
                fill,
                at: FillAt::Trigger,
                reason: Some(reason.as_str()),
            });
        }
        // (d) Indicators over the completed bar, then the close decision.
        atr.update_raw(bars.high[t], bars.low[t], bars.close[t]);
        channel.update_raw(bars.low[t]);
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
    }
    out.open_trip = m.finish(n - 1, bars.close[n - 1]);
    out.trips = m.closed().to_vec();
    out.ledger = m.ledger().clone();
    Ok(out)
}
