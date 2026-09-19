//! The Turtle rule coefficients, validated once, and the unit-sizing arithmetic. The rules only:
//! the cost model is a separate value ([`crate::costs::CostModel`]) the machine never sees.

use std::fmt;

use crate::error::Error;

/// When a price-triggered rule fires: on the bar's CLOSE (executed at the next opening print)
/// or on a TRADE through the level (a resting stop order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trigger {
    /// Decided at the close, executed at the next opening print.
    Close,
    /// A resting sell stop through the level.
    Trade,
}

impl Trigger {
    /// Parse the lowercase name.
    pub fn parse(text: &str) -> Result<Self, Error> {
        match text {
            "close" => Ok(Self::Close),
            "trade" => Ok(Self::Trade),
            other => Err(Error::Coefficients(format!(
                "trigger must be 'close' or 'trade', got {other:?}"
            ))),
        }
    }

    /// The lowercase name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Close => "close",
            Self::Trade => "trade",
        }
    }
}

/// Which N sets the stop after an add: the ATR frozen at entry, or the ATR current at the add's
/// signal bar ("N is not recalculated for sizing, only for the stop").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NSource {
    /// The entry ATR.
    Entry,
    /// The ATR current at the add's signal bar.
    Current,
}

impl NSource {
    /// Parse the lowercase name.
    pub fn parse(text: &str) -> Result<Self, Error> {
        match text {
            "entry" => Ok(Self::Entry),
            "current" => Ok(Self::Current),
            other => Err(Error::Coefficients(format!(
                "stop_n_source must be 'entry' or 'current', got {other:?}"
            ))),
        }
    }

    /// The lowercase name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Entry => "entry",
            Self::Current => "current",
        }
    }
}

/// The rule coefficients for ONE target: the shared Turtle knobs plus this target's fixed cash
/// budget and the instrument's price grid.
#[derive(Clone, Debug, PartialEq)]
pub struct Coefficients {
    /// ATR lookback in bars (the Turtle "N").
    pub atr_period: usize,
    /// Add another unit each time price rises this many N above the last fill.
    pub add_step_n: f64,
    /// The position cap in units (X shares each).
    pub max_units: u32,
    /// The stop sits this many N below the latest fill.
    pub stop_n: f64,
    /// The exit channel: the lowest low of this many completed bars.
    pub exit_lookback: usize,
    /// Fraction of the budget risked per unit between the fill and the initial stop.
    pub risk_per_unit: f64,
    /// How the stop-loss fires.
    pub stop_trigger: Trigger,
    /// How the channel exit fires.
    pub exit_trigger: Trigger,
    /// Which N sets the stop after an add.
    pub stop_n_source: NSource,
    /// Decimal places of the instrument's price grid (increment `10^-price_precision`).
    pub price_precision: u32,
    /// This target's fixed cash budget (`equity / n_targets`): the sizing base and the cash cap.
    pub budget: f64,
}

impl Coefficients {
    /// Refuse a coefficient set the rules cannot be run under.
    pub fn validate(&self) -> Result<(), Error> {
        let positive = |name: &str, v: f64| -> Result<(), Error> {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(Error::Coefficients(format!(
                    "{name} must be a finite number > 0, got {v}"
                )))
            }
        };
        if self.atr_period == 0 {
            return Err(Error::Coefficients("atr_period must be >= 1".into()));
        }
        if self.exit_lookback == 0 {
            return Err(Error::Coefficients("exit_lookback must be >= 1".into()));
        }
        if self.max_units == 0 {
            return Err(Error::Coefficients("max_units must be >= 1".into()));
        }
        positive("add_step_n", self.add_step_n)?;
        positive("stop_n", self.stop_n)?;
        positive("risk_per_unit", self.risk_per_unit)?;
        if self.risk_per_unit > 1.0 {
            return Err(Error::Coefficients(format!(
                "risk_per_unit must be <= 1, got {}",
                self.risk_per_unit
            )));
        }
        positive("budget", self.budget)?;
        if self.price_precision > 9 {
            return Err(Error::Coefficients(format!(
                "price_precision must be <= 9, got {}",
                self.price_precision
            )));
        }
        Ok(())
    }

    /// The first bar index on which an entry can be taken: the ATR needs `atr_period` bars and
    /// the exit channel needs `exit_lookback` completed bars before the first in-position bar.
    pub fn first_eligible_bar(&self) -> usize {
        first_eligible_bar(self.atr_period, self.exit_lookback)
    }
}

impl fmt::Display for Coefficients {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Coefficients(atr_period={}, add_step_n={}, max_units={}, stop_n={}, \
             exit_lookback={}, risk_per_unit={}, stop_trigger={}, exit_trigger={}, \
             stop_n_source={}, price_precision={}, budget={})",
            self.atr_period,
            self.add_step_n,
            self.max_units,
            self.stop_n,
            self.exit_lookback,
            self.risk_per_unit,
            self.stop_trigger.as_str(),
            self.exit_trigger.as_str(),
            self.stop_n_source.as_str(),
            self.price_precision,
            self.budget
        )
    }
}

/// Shares per unit: `floor(risk_per_unit × budget / (stop_n × n))` — the cash risked between the
/// fill and the initial stop, divided by the stop distance per share. Zero when `n` is not a
/// finite positive number or the quotient is not finite. A representation-error guard keeps an
/// exact quotient (250.0 written as 249.999…) on its integer.
pub fn unit_shares(risk_per_unit: f64, budget: f64, stop_n: f64, n: f64) -> u64 {
    if !(n.is_finite() && n > 0.0) {
        return 0;
    }
    let q = risk_per_unit * budget / (stop_n * n);
    if !q.is_finite() || q <= 0.0 {
        return 0;
    }
    (q + 1e-9).floor() as u64
}

/// The first bar index on which an entry can be taken (see
/// [`Coefficients::first_eligible_bar`]): `max(atr_period, exit_lookback) - 1`.
pub fn first_eligible_bar(atr_period: usize, exit_lookback: usize) -> usize {
    atr_period.max(exit_lookback).saturating_sub(1)
}
