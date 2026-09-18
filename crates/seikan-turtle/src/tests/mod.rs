//! Unit tests for the kernel: the price grid, the indicators (Faith's own N table), the state
//! machine (the worked example from the rules, every skip and precedence case) and the reference
//! simulator's identities.

mod indicators;
mod machine;
mod price;
mod sim;

use crate::coefficients::{Coefficients, NSource, Trigger};
use crate::sim::BarSeries;

/// The rules' defaults over a $100,000 single-target budget.
pub(crate) fn coefficients() -> Coefficients {
    Coefficients {
        atr_period: 20,
        add_step_n: 0.5,
        max_units: 3,
        stop_n: 2.0,
        exit_lookback: 20,
        risk_per_unit: 0.01,
        stop_trigger: Trigger::Close,
        exit_trigger: Trigger::Close,
        stop_n_source: NSource::Current,
        price_precision: 4,
        budget: 100_000.0,
    }
}

/// Explicit OHLC bars plus the firing flags, built bar by bar.
#[derive(Clone, Debug, Default)]
pub(crate) struct Bars {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub fired: Vec<bool>,
}

impl Bars {
    pub fn push(&mut self, o: f64, h: f64, l: f64, c: f64, fired: bool) -> &mut Self {
        self.open.push(o);
        self.high.push(h);
        self.low.push(l);
        self.close.push(c);
        self.fired.push(fired);
        self
    }

    /// `n` bars with open == close == `c` and a symmetric range of `half` — a constant true
    /// range of `2 × half`, so the ATR seeds to exactly that.
    pub fn flat(&mut self, n: usize, c: f64, half: f64) -> &mut Self {
        for _ in 0..n {
            self.push(c, c + half, c - half, c, false);
        }
        self
    }

    pub fn series(&self) -> BarSeries<'_> {
        BarSeries {
            open: &self.open,
            high: &self.high,
            low: &self.low,
            close: &self.close,
        }
    }

    pub fn len(&self) -> usize {
        self.close.len()
    }
}

/// The worked example from the rules: N = 2 throughout the ladder, entry at $50, adds at $51 and
/// $52, then a bar that closes (and trades) below the $48 stop. Bars 0..=23 are flat at 49.5
/// (range 48.5–50.5, TR 2), bar 24 fires with close 50, bar 25 opens at 50 (P0), bars 26/27 open
/// at the add levels, bar 28 breaks down to 47.5 (low 47.5), bar 29 opens at 47.
pub(crate) fn worked_example() -> Bars {
    let mut b = Bars::default();
    b.flat(24, 49.5, 1.0)
        .push(49.5, 51.0, 49.0, 50.0, true)
        .push(50.0, 52.0, 50.0, 51.0, false)
        .push(51.0, 53.0, 51.0, 52.0, false)
        .push(52.0, 53.5, 51.5, 52.5, false)
        .push(52.5, 52.5, 47.5, 47.5, false)
        .push(47.0, 47.5, 46.5, 47.0, false)
        .flat(2, 47.0, 1.0);
    b
}

pub(crate) fn close_to(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}
