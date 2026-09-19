//! The trade-cost model: the commission a fill is charged and the adverse distances (slippage,
//! market impact, the stop-fill shock) between a fill's reference price and its print.
//!
//! Every rate is in basis points of the reference or the notional; every money amount in the
//! account currency; `N` is the Wilder ATR in force during the bar and `ADV` the mean volume of
//! the previous `adv_window` bars — both known at the print, never look-ahead.

use crate::coefficients::Coefficients;
use crate::error::Error;
use crate::execution::Side;

/// One basis point as a fraction.
const BPS: f64 = 1e-4;

/// A per-fill commission schedule: `max(per_share × shares + bps × notional, min_per_order)`,
/// capped at `cap_bps × notional` when a cap is set, plus `sell_bps × notional` on sells (a stamp
/// duty or transaction fee the cap never applies to).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Commission {
    /// Currency per share, every fill.
    pub per_share: f64,
    /// The floor per fill.
    pub min_per_order: f64,
    /// Basis points of the notional, every fill.
    pub bps: f64,
    /// Basis points of the notional on sells only, never capped.
    pub sell_bps: f64,
    /// The cap per fill as basis points of the notional; `None` leaves the fill uncapped.
    pub cap_bps: Option<f64>,
}

impl Commission {
    /// No commission at all.
    pub const NONE: Self = Self {
        per_share: 0.0,
        min_per_order: 0.0,
        bps: 0.0,
        sell_bps: 0.0,
        cap_bps: None,
    };

    /// The commission charged on a fill of `shares` at `notional` (shares × fill price).
    pub fn charge(&self, side: Side, shares: u64, notional: f64) -> f64 {
        let mut charge = self.per_share * shares as f64 + self.bps * BPS * notional;
        charge = charge.max(self.min_per_order);
        if let Some(cap) = self.cap_bps {
            charge = charge.min(cap * BPS * notional);
        }
        if side == Side::Sell {
            charge += self.sell_bps * BPS * notional;
        }
        charge
    }

    fn validate(&self) -> Result<(), Error> {
        non_negative("commission.per_share", self.per_share)?;
        non_negative("commission.min_per_order", self.min_per_order)?;
        non_negative("commission.bps", self.bps)?;
        non_negative("commission.sell_bps", self.sell_bps)?;
        if let Some(cap) = self.cap_bps
            && !(cap.is_finite() && cap > 0.0)
        {
            return Err(Error::Coefficients(format!(
                "commission.cap_bps must be a finite number > 0 or absent, got {cap}"
            )));
        }
        Ok(())
    }
}

/// Adverse slippage per share on every fill: `reference × bps + n_fraction × N`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slippage {
    /// Basis points of the reference price.
    pub bps: f64,
    /// A fraction of N (the Turtle-native "slippage in N"), additive.
    pub n_fraction: f64,
}

impl Slippage {
    /// No slippage.
    pub const NONE: Self = Self {
        bps: 0.0,
        n_fraction: 0.0,
    };

    /// The adverse distance per share at `reference` under `n`.
    pub fn per_share(&self, reference: f64, n: f64) -> f64 {
        reference * self.bps * BPS + self.n_fraction * n
    }

    /// Whether the model reads N at all.
    pub fn reads_n(&self) -> bool {
        self.n_fraction > 0.0
    }

    fn validate(&self) -> Result<(), Error> {
        non_negative("slippage.bps", self.bps)?;
        non_negative("slippage.n_fraction", self.n_fraction)
    }
}

/// Square-root market impact per share: `coefficient × N × sqrt(shares / ADV)` — the
/// `Y·σ·√(Q/ADV)` law with N standing in for σ so the price cancels. A zero coefficient
/// switches it off; a positive one needs a volume series and an ADV initialized before the
/// first eligible bar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Impact {
    /// The impact coefficient in units of N.
    pub coefficient: f64,
    /// Bars in the average daily volume.
    pub adv_window: usize,
}

impl Impact {
    /// Impact switched off.
    pub const NONE: Self = Self {
        coefficient: 0.0,
        adv_window: 1,
    };

    /// Whether impact is modelled.
    pub fn enabled(&self) -> bool {
        self.coefficient > 0.0
    }

    /// The adverse distance per share of a `shares`-lot against `adv` under `n`.
    pub fn per_share(&self, n: f64, shares: u64, adv: f64) -> f64 {
        self.coefficient * n * (shares as f64 / adv).sqrt()
    }

    fn validate(&self) -> Result<(), Error> {
        non_negative("impact.coefficient", self.coefficient)?;
        if self.adv_window == 0 {
            return Err(Error::Coefficients("impact.adv_window must be >= 1".into()));
        }
        Ok(())
    }
}

/// The whole cost model, validated once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostModel {
    /// The commission schedule.
    pub commission: Commission,
    /// The per-fill slippage.
    pub slippage: Slippage,
    /// The market-impact law.
    pub impact: Impact,
    /// The fraction of the bar's adverse continuation beyond a stop's trigger the fill gives up
    /// (0 fills at the trigger, 1 at the bar's low).
    pub stop_shock: f64,
}

impl CostModel {
    /// Every cost switched off: fills at their references, no commission.
    pub const FRICTIONLESS: Self = Self {
        commission: Commission::NONE,
        slippage: Slippage::NONE,
        impact: Impact::NONE,
        stop_shock: 0.0,
    };

    /// Refuse a cost model outside its domains.
    pub fn validate(&self) -> Result<(), Error> {
        self.commission.validate()?;
        self.slippage.validate()?;
        self.impact.validate()?;
        if !(self.stop_shock.is_finite() && (0.0..=1.0).contains(&self.stop_shock)) {
            return Err(Error::Coefficients(format!(
                "stop_shock must be a finite number in [0, 1], got {}",
                self.stop_shock
            )));
        }
        Ok(())
    }

    /// Refuse an enabled impact whose ADV would not be initialized by the rules' first eligible
    /// bar: warmup is a rule quantity and no cost knob may move it.
    pub fn check_warmup(&self, rules: &Coefficients) -> Result<(), Error> {
        let floor = rules.atr_period.max(rules.exit_lookback);
        if self.impact.enabled() && self.impact.adv_window > floor {
            return Err(Error::Coefficients(format!(
                "impact.adv_window ({}) must not exceed max(atr_period, exit_lookback) ({floor}) \
                 when impact is enabled",
                self.impact.adv_window
            )));
        }
        Ok(())
    }

    /// Whether any fill needs N to be priced.
    pub fn reads_n(&self) -> bool {
        self.slippage.reads_n() || self.impact.enabled()
    }
}

fn non_negative(name: &str, value: f64) -> Result<(), Error> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(Error::Coefficients(format!(
            "{name} must be a finite number >= 0, got {value}"
        )))
    }
}
