//! Execution: how an order becomes a fill under the cost model.
//!
//! A market order at the opening print references the open; a resting sell stop the bar trades
//! through references its trigger and is first shocked toward the bar's low; a stop the open
//! gaps through references the open like a market order. Every fill then moves the adverse
//! distance (slippage, impact) away from its reference in whole grid steps, against the
//! account, and is charged the schedule's commission on the resulting notional.

use crate::costs::CostModel;
use crate::error::Error;
use crate::price::{step_down, step_up};

/// The side of a fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// Shares bought (entries, adds).
    Buy,
    /// Shares sold (exits).
    Sell,
}

/// One priced fill: the print and its cost attribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Execution {
    /// Which side.
    pub side: Side,
    /// Shares filled.
    pub shares: u64,
    /// The price an ideal market would have filled at (the open, or the stop's trigger).
    pub reference: f64,
    /// The fill price, on the grid.
    pub price: f64,
    /// The commission charged.
    pub commission: f64,
    /// `|price − reference| × shares` less the shock and the impact: the bps and N-fraction
    /// slippage plus the grid rounding.
    pub slippage: f64,
    /// The stop-shock term in currency (zero on any print fill).
    pub shock: f64,
    /// The market-impact term in currency.
    pub impact: f64,
}

impl Execution {
    /// `shares × price`.
    pub fn notional(&self) -> f64 {
        self.shares as f64 * self.price
    }

    /// The cash a buy takes out of the account: notional plus commission.
    pub fn cash_out(&self) -> f64 {
        self.notional() + self.commission
    }

    /// The cash a sell brings in: notional minus commission.
    pub fn cash_in(&self) -> f64 {
        self.notional() - self.commission
    }
}

/// What is known about a bar when an order in it is priced: the print, the low the fill model
/// reads for a stop's continuation, and the two indicator values in force during the bar
/// (`None` before they initialize).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarContext {
    /// The opening print.
    pub open: f64,
    /// The bar's low.
    pub low: f64,
    /// The ATR after the previous bar.
    pub n: Option<f64>,
    /// The average volume after the previous bar.
    pub adv: Option<f64>,
}

/// Prices fills under one cost model on one price grid.
#[derive(Clone, Copy, Debug)]
pub struct Executor {
    costs: CostModel,
    precision: u32,
}

impl Executor {
    /// An executor for `costs` on the `10^-precision` grid.
    pub fn new(costs: CostModel, precision: u32) -> Self {
        Self { costs, precision }
    }

    /// The cost model in force.
    pub fn costs(&self) -> &CostModel {
        &self.costs
    }

    /// Refuse a context that cannot price a fill this bar: a model that reads N or the ADV
    /// while the indicator is uninitialized. Called by the simulator only on bars where a fill
    /// can happen, which the rules place after every indicator's warmup.
    pub fn check_context(&self, ctx: &BarContext) -> Result<(), Error> {
        if self.costs.reads_n() && ctx.n.is_none() {
            return Err(Error::Bookkeeping(
                "a fill priced against N before the ATR initialized".into(),
            ));
        }
        if self.costs.impact.enabled() && ctx.adv.is_none() {
            return Err(Error::Bookkeeping(
                "a fill priced against the ADV before it initialized".into(),
            ));
        }
        Ok(())
    }

    /// A market order at a print: the reference is the print itself.
    pub fn market(
        &self,
        side: Side,
        shares: u64,
        reference: f64,
        ctx: &BarContext,
    ) -> Result<Execution, Error> {
        self.fill(side, shares, reference, 0.0, ctx)
    }

    /// A resting sell stop the bar traded through: the reference is the trigger, shocked toward
    /// the bar's low by `stop_shock` of the continuation beyond it.
    pub fn stop(&self, shares: u64, trigger: f64, ctx: &BarContext) -> Result<Execution, Error> {
        let shock = self.costs.stop_shock * (trigger - ctx.low).max(0.0);
        self.fill(Side::Sell, shares, trigger, shock, ctx)
    }

    fn fill(
        &self,
        side: Side,
        shares: u64,
        reference: f64,
        shock_per_share: f64,
        ctx: &BarContext,
    ) -> Result<Execution, Error> {
        if shares == 0 {
            return Err(Error::Bookkeeping("a fill of zero shares".into()));
        }
        self.check_context(ctx)?;
        let n = ctx.n.unwrap_or(0.0);
        let slip = self.costs.slippage.per_share(reference, n);
        let impact_per_share = if self.costs.impact.enabled() {
            self.costs
                .impact
                .per_share(n, shares, ctx.adv.unwrap_or(f64::NAN))
        } else {
            0.0
        };
        let adverse = slip + shock_per_share + impact_per_share;
        if !adverse.is_finite() {
            return Err(Error::Bookkeeping(format!(
                "a non-finite adverse distance {adverse} at reference {reference}"
            )));
        }
        let price = match side {
            Side::Buy => step_up(reference, adverse, self.precision),
            Side::Sell => step_down(reference, adverse, self.precision),
        };
        if !(price.is_finite() && price > 0.0) {
            return Err(Error::Input(format!(
                "the cost model moved a fill at reference {reference} to {price}: the adverse \
                 distance {adverse} exceeds the price"
            )));
        }
        let q = shares as f64;
        let notional = q * price;
        let shock = shock_per_share * q;
        let impact = impact_per_share * q;
        let slippage = ((price - reference).abs() * q - shock - impact).max(0.0);
        Ok(Execution {
            side,
            shares,
            reference,
            price,
            commission: self.costs.commission.charge(side, shares, notional),
            slippage,
            shock,
            impact,
        })
    }
}
