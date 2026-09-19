//! How an order becomes a fill: references, adverse steps, attribution.

use crate::costs::{CostModel, Impact, Slippage};
use crate::error::Error;
use crate::execution::{BarContext, Executor, Side};
use crate::price::{price_below, quantize};
use crate::tests::{close_to, realistic_costs};

fn ctx(open: f64, low: f64) -> BarContext {
    BarContext {
        open,
        low,
        n: Some(2.0),
        adv: None,
    }
}

#[test]
fn frictionless_fills_are_their_references_bit_for_bit() {
    let x = Executor::new(CostModel::FRICTIONLESS, 9);
    let open = quantize(12_345.123_456_789, 9);
    let c = BarContext {
        open,
        low: open - 1.0,
        n: None,
        adv: None,
    };
    let buy = x.market(Side::Buy, 250, open, &c).unwrap();
    assert_eq!(buy.price.to_bits(), open.to_bits());
    assert_eq!(buy.reference.to_bits(), open.to_bits());
    assert_eq!(
        (buy.commission, buy.slippage, buy.shock, buy.impact),
        (0.0, 0.0, 0.0, 0.0)
    );
    let trigger = price_below(open, 9);
    let stop = x.stop(250, trigger, &c).unwrap();
    assert_eq!(stop.price.to_bits(), trigger.to_bits());
    assert!(close_to(stop.cash_in(), 250.0 * trigger));
}

#[test]
fn a_market_buy_steps_up_and_a_sell_steps_down() {
    let x = Executor::new(realistic_costs(), 4);
    let buy = x.market(Side::Buy, 250, 50.0, &ctx(50.0, 49.0)).unwrap();
    assert!(close_to(buy.price, 50.025)); // 5 bps of 50, exactly 250 steps
    assert!(close_to(buy.notional(), 250.0 * 50.025));
    assert!(close_to(buy.commission, 1.25));
    assert!(close_to(buy.slippage, 250.0 * 0.025));
    assert!(close_to(buy.cash_out(), 250.0 * 50.025 + 1.25));
    assert_eq!((buy.shock, buy.impact), (0.0, 0.0));
    let sell = x.market(Side::Sell, 250, 50.0, &ctx(50.0, 49.0)).unwrap();
    assert!(close_to(sell.price, 49.975));
    assert!(close_to(sell.cash_in(), 250.0 * 49.975 - 1.25));
}

#[test]
fn slippage_in_n_and_impact_add_to_the_adverse_distance() {
    let mut costs = CostModel::FRICTIONLESS;
    costs.slippage = Slippage {
        bps: 0.0,
        n_fraction: 0.1,
    };
    let x = Executor::new(costs, 4);
    let buy = x.market(Side::Buy, 250, 50.0, &ctx(50.0, 49.0)).unwrap();
    assert!(close_to(buy.price, 50.2)); // 0.1 × N 2
    costs.impact = Impact {
        coefficient: 0.5,
        adv_window: 20,
    };
    let x = Executor::new(costs, 4);
    let c = BarContext {
        open: 50.0,
        low: 49.0,
        n: Some(2.0),
        adv: Some(1000.0),
    };
    let buy = x.market(Side::Buy, 250, 50.0, &c).unwrap();
    assert!(close_to(buy.price, 50.7)); // + 0.5 × 2 × sqrt(250/1000)
    assert!(close_to(buy.impact, 250.0 * 0.5));
    assert!(close_to(buy.slippage, 250.0 * 0.2));
}

#[test]
fn a_stop_hit_inside_the_bar_is_shocked_toward_the_low() {
    let x = Executor::new(realistic_costs(), 4); // stop_shock 0.5, 5 bps
    let trigger = price_below(48.0, 4); // 47.9999
    let hit = x.stop(750, trigger, &ctx(52.5, 47.5)).unwrap();
    // Half the continuation beyond the trigger (0.24995) plus 5 bps of the trigger (0.0240),
    // rounded down to the grid: 2739.9… steps → 2740.
    assert!(close_to(hit.reference, trigger));
    assert!(close_to(hit.price, 47.9999 - 0.274));
    assert!(close_to(hit.shock, 750.0 * 0.249_95));
    assert!(close_to(hit.impact, 0.0));
    // Slippage takes the bps term and the rounding residue.
    assert!(close_to(
        hit.slippage,
        (trigger - hit.price) * 750.0 - hit.shock
    ));
    assert!(hit.slippage >= 750.0 * 0.023_999_95);
    // The reference-price identity behind every round trip.
    assert!(close_to(
        750.0 * trigger,
        hit.cash_in() + hit.commission + hit.slippage + hit.shock
    ));
    // A low AT the trigger has no continuation to give up.
    let touch = x.stop(750, trigger, &ctx(52.5, trigger)).unwrap();
    assert!(close_to(touch.shock, 0.0));
    // A gapped stop is a market order at the open: no shock term.
    let gap = x.market(Side::Sell, 750, 45.0, &ctx(45.0, 44.0)).unwrap();
    assert!(close_to(gap.shock, 0.0) && close_to(gap.price, 45.0 - 0.0225));
}

#[test]
fn refusals() {
    let x = Executor::new(realistic_costs(), 4);
    assert!(matches!(
        x.market(Side::Buy, 0, 50.0, &ctx(50.0, 49.0)),
        Err(Error::Bookkeeping(_))
    ));
    let mut costs = realistic_costs();
    costs.slippage.n_fraction = 0.1;
    let x = Executor::new(costs, 4);
    let no_n = BarContext {
        open: 50.0,
        low: 49.0,
        n: None,
        adv: None,
    };
    assert!(matches!(
        x.market(Side::Buy, 1, 50.0, &no_n),
        Err(Error::Bookkeeping(_))
    ));
    assert!(x.check_context(&no_n).is_err());
    assert!(x.check_context(&ctx(50.0, 49.0)).is_ok());
    let mut costs = CostModel::FRICTIONLESS;
    costs.impact = Impact {
        coefficient: 0.5,
        adv_window: 20,
    };
    let x = Executor::new(costs, 4);
    assert!(matches!(
        x.market(Side::Buy, 1, 50.0, &ctx(50.0, 49.0)),
        Err(Error::Bookkeeping(_))
    ));
    // A cost model that swallows the whole price is an input problem.
    let mut costs = CostModel::FRICTIONLESS;
    costs.slippage.n_fraction = 100.0;
    let x = Executor::new(costs, 4);
    assert!(matches!(
        x.market(Side::Sell, 1, 50.0, &ctx(50.0, 49.0)),
        Err(Error::Input(_))
    ));
}
