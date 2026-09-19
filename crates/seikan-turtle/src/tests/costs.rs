//! The cost model's arithmetic and domains.

use crate::costs::{Commission, CostModel, Impact, Slippage};
use crate::execution::{BarContext, Executor, Side};
use crate::tests::{Lcg, close_to, coefficients, realistic_costs};

#[test]
fn the_commission_schedule_applies_its_floor_cap_and_sell_fee() {
    let c = realistic_costs().commission;
    // 250 shares @ 50: 1.25 sits above the $1 floor and under the 1% cap.
    assert!(close_to(c.charge(Side::Buy, 250, 12_500.0), 1.25));
    // 20 shares @ 50: 0.10 is lifted to the floor.
    assert!(close_to(c.charge(Side::Buy, 20, 1_000.0), 1.0));
    // 1 share @ 0.50: the floor would be $1, the 1% cap says 0.005.
    assert!(close_to(c.charge(Side::Buy, 1, 0.5), 0.005));
    // A sell-side fee rides on top of the capped charge.
    let mut with_duty = c;
    with_duty.sell_bps = 10.0;
    assert!(close_to(with_duty.charge(Side::Buy, 250, 12_500.0), 1.25));
    assert!(close_to(
        with_duty.charge(Side::Sell, 250, 12_500.0),
        1.25 + 12.5
    ));
    // A pure bps schedule with no floor and no cap.
    let bps = Commission {
        per_share: 0.0,
        min_per_order: 0.0,
        bps: 5.0,
        sell_bps: 0.0,
        cap_bps: None,
    };
    assert!(close_to(bps.charge(Side::Sell, 7, 10_000.0), 5.0));
    assert!(close_to(Commission::NONE.charge(Side::Buy, 1000, 1e6), 0.0));
}

#[test]
fn slippage_and_impact_per_share() {
    let s = Slippage {
        bps: 5.0,
        n_fraction: 0.1,
    };
    assert!(close_to(s.per_share(50.0, 2.0), 0.025 + 0.2));
    assert!(s.reads_n() && !Slippage::NONE.reads_n());
    let i = Impact {
        coefficient: 0.5,
        adv_window: 20,
    };
    assert!(i.enabled() && !Impact::NONE.enabled());
    // 250 of an ADV of 1000: sqrt(0.25) = 0.5 → 0.5 × N 2 × 0.5.
    assert!(close_to(i.per_share(2.0, 250, 1000.0), 0.5));
}

#[test]
fn domains_are_enforced() {
    assert!(CostModel::FRICTIONLESS.validate().is_ok());
    assert!(realistic_costs().validate().is_ok());
    let mut m = realistic_costs();
    m.commission.per_share = -0.01;
    assert!(m.validate().is_err());
    let mut m = realistic_costs();
    m.commission.cap_bps = Some(0.0);
    assert!(m.validate().is_err());
    let mut m = realistic_costs();
    m.slippage.bps = f64::NAN;
    assert!(m.validate().is_err());
    let mut m = realistic_costs();
    m.impact.adv_window = 0;
    assert!(m.validate().is_err());
    let mut m = realistic_costs();
    m.stop_shock = 1.5;
    assert!(m.validate().is_err());
    let mut m = realistic_costs();
    m.stop_shock = -0.0;
    assert!(m.validate().is_ok());
}

#[test]
fn an_enabled_impact_may_not_outlast_the_rules_warmup() {
    let rules = coefficients(); // 20 / 20
    let mut m = realistic_costs();
    m.impact.adv_window = 30;
    assert!(m.check_warmup(&rules).is_ok()); // disabled: the window is unread
    m.impact.coefficient = 0.5;
    assert!(m.check_warmup(&rules).is_err());
    m.impact.adv_window = 20;
    assert!(m.check_warmup(&rules).is_ok());
    assert!(m.reads_n());
}

#[test]
fn the_all_in_cash_out_is_monotone_in_the_size() {
    // The sizing search relies on it: over random schedules, contexts and sizes, one more share
    // never costs less all in.
    let mut rng = Lcg::new(11);
    for _ in 0..400 {
        let costs = CostModel {
            commission: Commission {
                per_share: rng.range(0.0, 0.02),
                min_per_order: rng.range(0.0, 5.0),
                bps: rng.range(0.0, 20.0),
                sell_bps: rng.range(0.0, 10.0),
                cap_bps: if rng.unit() < 0.5 {
                    Some(rng.range(10.0, 200.0))
                } else {
                    None
                },
            },
            slippage: Slippage {
                bps: rng.range(0.0, 50.0),
                n_fraction: rng.range(0.0, 0.5),
            },
            impact: Impact {
                coefficient: rng.range(0.0, 2.0),
                adv_window: 20,
            },
            stop_shock: rng.range(0.0, 1.0),
        };
        let executor = Executor::new(costs, 4);
        let open = (rng.range(1.0, 500.0) * 1e4).round() / 1e4;
        let ctx = BarContext {
            open,
            low: open * 0.98,
            n: Some(rng.range(0.01, 5.0)),
            adv: Some(rng.range(100.0, 1e6)),
        };
        let q = rng.range(1.0, 5000.0) as u64;
        let a = executor
            .market(Side::Buy, q, open, &ctx)
            .unwrap()
            .cash_out();
        let b = executor
            .market(Side::Buy, q + 1, open, &ctx)
            .unwrap()
            .cash_out();
        assert!(
            b >= a,
            "cash-out fell from {a} to {b} at {q} shares under {costs:?}"
        );
    }
}
