//! Unit tests for the engine: the price grid, the indicators (Faith's own N table), the state
//! machine (the worked example from the rules, every skip and precedence case), the cost model,
//! the execution arithmetic and the simulator's identities.

mod costs;
mod execution;
mod indicators;
mod machine;
mod price;
mod sim;

use crate::coefficients::{Coefficients, NSource, Trigger};
use crate::costs::{Commission, CostModel, Impact, Slippage};
use crate::sim::{BarSeries, SimResult, simulate};

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

/// A liquid US-equity retail-pro account: the coefficients document's defaults.
pub(crate) fn realistic_costs() -> CostModel {
    CostModel {
        commission: Commission {
            per_share: 0.005,
            min_per_order: 1.0,
            bps: 0.0,
            sell_bps: 0.0,
            cap_bps: Some(100.0),
        },
        slippage: Slippage {
            bps: 5.0,
            n_fraction: 0.0,
        },
        impact: Impact {
            coefficient: 0.0,
            adv_window: 20,
        },
        stop_shock: 0.5,
    }
}

/// Explicit OHLC bars plus the firing flags, built bar by bar.
#[derive(Clone, Debug, Default)]
pub(crate) struct Bars {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Option<Vec<f64>>,
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

    /// A constant volume on every bar.
    pub fn with_volume(&mut self, volume: f64) -> &mut Self {
        self.volume = Some(vec![volume; self.len()]);
        self
    }

    pub fn series(&self) -> BarSeries<'_> {
        BarSeries {
            open: &self.open,
            high: &self.high,
            low: &self.low,
            close: &self.close,
            volume: self.volume.as_deref(),
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

/// Run frictionless — the rules alone.
pub(crate) fn run(c: &Coefficients, b: &Bars) -> SimResult {
    simulate(c, &CostModel::FRICTIONLESS, b.series(), &b.fired).expect("simulation runs")
}

/// Run under a cost model.
pub(crate) fn run_with(c: &Coefficients, costs: &CostModel, b: &Bars) -> SimResult {
    simulate(c, costs, b.series(), &b.fired).expect("simulation runs")
}

pub(crate) fn close_to(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

/// A tiny deterministic generator for property tests (no `rand` dependency).
pub(crate) struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// A float in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A float in `[lo, hi)`.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}
