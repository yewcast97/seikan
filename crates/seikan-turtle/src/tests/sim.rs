//! Identities of the simulator over a trend path: fills in bar order, equity as cash plus the
//! marked position, closed pnl equal to the cash change once flat — frictionless and under a
//! cost model, where the books must also reconcile with every fill's attribution.

use crate::costs::Impact;
use crate::error::Error;
use crate::machine::FillKind;
use crate::sim::simulate;
use crate::tests::{Bars, close_to, coefficients, realistic_costs, run, run_with};

/// Flat, then a climb of `up` bars, then a slide of `down` bars — guarantees an entry, adds and
/// an exit under the default coefficients.
fn trend_path(flat: usize, up: usize, down: usize) -> Bars {
    let mut b = Bars::default();
    b.flat(flat, 100.0, 1.0);
    let mut px = 100.0;
    for i in 0..up {
        let next = px + 1.5;
        b.push(px, next + 0.5, px - 0.5, next, i == 0);
        px = next;
    }
    for _ in 0..down {
        let next = px - 3.75;
        b.push(px, px + 0.5, next - 0.5, next, false);
        px = next;
    }
    b.flat(5, px, 1.0);
    b
}

#[test]
fn trend_path_enters_pyramids_and_exits() {
    let c = coefficients();
    let b = trend_path(30, 15, 12);
    let r = run(&c, &b);
    let kinds: Vec<&str> = r.fills.iter().map(|f| f.fill.kind.as_str()).collect();
    assert_eq!(kinds, vec!["entry", "add", "add", "exit"]);
    assert!(r.open_trip.is_none());
    assert_eq!(r.trips.len(), 1);
    let bars: Vec<usize> = r.fills.iter().map(|f| f.fill.bar).collect();
    assert!(bars.windows(2).all(|w| w[0] < w[1]));
    for t in 0..b.len() {
        assert!(close_to(
            r.equity[t],
            r.cash[t] + r.shares[t] as f64 * b.close[t]
        ));
        assert_eq!(r.shares[t], r.units[t] as u64 * r.trips[0].unit_shares);
        assert_eq!(r.commission_cum[t], 0.0);
    }
    let pnl: f64 = r.trips.iter().map(|t| t.pnl).sum();
    assert!(close_to(pnl, r.cash[b.len() - 1] - c.budget));
    // The 2N stop under the last add sits below the ladder's average cost: the slide gives the
    // trend back and the trip closes on the stop, as the rules' own worked example does.
    assert_eq!(r.trips[0].exit_reason.as_str(), "stop_close");
    assert!(close_to(
        r.trips[0].pnl,
        r.trips[0].proceeds - r.trips[0].cost_basis
    ));
    // Every fill records the books right after it.
    let last = r.fills.last().unwrap();
    assert_eq!((last.shares_after, last.units_after), (0, 0));
    assert!(last.stop_after.is_none() && close_to(last.cash_after, r.cash[last.fill.bar]));
    assert!(close_to(r.fills[0].reference, r.fills[0].fill.price));
}

#[test]
fn the_books_reconcile_with_every_fill_under_costs() {
    let c = coefficients();
    let mut costs = realistic_costs();
    costs.commission.sell_bps = 2.0;
    costs.slippage.n_fraction = 0.05;
    costs.impact = Impact {
        coefficient: 0.5,
        adv_window: 20,
    };
    let mut b = trend_path(30, 15, 12);
    b.with_volume(50_000.0);
    let r = run_with(&c, &costs, &b);
    assert_eq!(r.trips.len(), 1);
    assert!(r.open_trip.is_none());
    // Cash: the budget less every buy's all-in cost plus every sell's net proceeds.
    let mut cash = c.budget;
    let (mut commission, mut slippage, mut shock, mut impact) = (0.0, 0.0, 0.0, 0.0);
    let (mut bought_at_ref, mut sold_at_ref) = (0.0, 0.0);
    for f in &r.fills {
        let notional = f.fill.shares as f64 * f.fill.price;
        match f.fill.kind {
            FillKind::Entry | FillKind::Add => {
                cash -= notional + f.fill.commission;
                bought_at_ref += f.fill.shares as f64 * f.reference;
                assert!(f.fill.price > f.reference);
                assert!(f.fill.impact > 0.0 && f.fill.shock == 0.0);
            }
            FillKind::Exit => {
                cash += notional - f.fill.commission;
                sold_at_ref += f.fill.shares as f64 * f.reference;
                assert!(f.fill.price < f.reference);
            }
        }
        assert!(f.fill.commission >= 1.0);
        assert!(close_to(f.cash_after, cash));
        commission += f.fill.commission;
        slippage += f.fill.slippage;
        shock += f.fill.shock;
        impact += f.fill.impact;
    }
    let last = b.len() - 1;
    assert!(close_to(r.cash[last], cash));
    assert!(close_to(r.commission_cum[last], commission));
    assert!(close_to(r.slippage_cum[last], slippage));
    assert!(close_to(r.shock_cum[last], shock));
    assert!(close_to(r.impact_cum[last], impact));
    assert!(r.commission_cum.windows(2).all(|w| w[0] <= w[1]));
    // The round trip carries the same attribution, and net pnl is the cash change once flat.
    let t = &r.trips[0];
    assert!(close_to(t.commission, commission));
    assert!(close_to(t.slippage, slippage));
    assert!(close_to(t.shock, shock));
    assert!(close_to(t.impact, impact));
    assert!(close_to(t.gross_pnl, t.proceeds - t.cost_basis));
    assert!(close_to(t.pnl, t.gross_pnl - t.commission));
    assert!(close_to(t.pnl, r.cash[last] - c.budget));
    // The reference-price identity: what the fills would have been worth at their references
    // equals the net result plus every cost bucket.
    assert!(close_to(
        sold_at_ref - bought_at_ref,
        t.pnl + t.commission + t.slippage + t.shock + t.impact
    ));
    assert!(t.pnl < run(&c, &b).trips[0].pnl);
}

#[test]
fn an_intrabar_stop_under_costs_is_shocked_toward_the_low() {
    let mut c = coefficients();
    c.stop_trigger = crate::coefficients::Trigger::Trade;
    let costs = realistic_costs();
    let b = crate::tests::worked_example();
    let r = run_with(&c, &costs, &b);
    let exit = r.fills.last().unwrap();
    assert_eq!(exit.fill.kind, FillKind::Exit);
    assert_eq!(exit.at, crate::sim::FillAt::Trigger);
    // Costed buys fill above the open, so the ladder and its stop sit higher than the
    // frictionless 48: the resting trigger is one step under the trip's own max_stop.
    let trigger = crate::price::price_below(r.trips[0].max_stop, 4);
    assert!(r.trips[0].max_stop > 48.0);
    assert!(close_to(exit.reference, trigger));
    assert!(exit.fill.price < trigger && exit.fill.shock > 0.0);
    assert!(close_to(r.shock_cum[b.len() - 1], exit.fill.shock));
}

#[test]
fn the_simulator_refuses_ragged_or_unusable_input() {
    let c = coefficients();
    let b = trend_path(30, 5, 5);
    let mut fired = b.fired.clone();
    fired.pop();
    assert!(matches!(
        simulate(&c, &realistic_costs(), b.series(), &fired),
        Err(Error::Input(_))
    ));
    let empty = Bars::default();
    assert!(simulate(&c, &realistic_costs(), empty.series(), &[]).is_err());
    // Impact needs a volume series, finite and positive on every bar, and an ADV window the
    // rules' warmup covers.
    let mut costs = realistic_costs();
    costs.impact = Impact {
        coefficient: 0.5,
        adv_window: 20,
    };
    assert!(matches!(
        simulate(&c, &costs, b.series(), &b.fired),
        Err(Error::Input(_))
    ));
    let mut with_volume = b.clone();
    with_volume.with_volume(1000.0);
    assert!(simulate(&c, &costs, with_volume.series(), &b.fired).is_ok());
    with_volume.volume.as_mut().unwrap()[3] = 0.0;
    assert!(matches!(
        simulate(&c, &costs, with_volume.series(), &b.fired),
        Err(Error::Input(_))
    ));
    costs.impact.adv_window = 21;
    with_volume.volume.as_mut().unwrap()[3] = 1000.0;
    assert!(matches!(
        simulate(&c, &costs, with_volume.series(), &b.fired),
        Err(Error::Coefficients(_))
    ));
    // A cost model outside its domain refuses before a bar is read.
    let mut bad = realistic_costs();
    bad.stop_shock = 2.0;
    assert!(matches!(
        simulate(&c, &bad, b.series(), &b.fired),
        Err(Error::Coefficients(_))
    ));
}
