//! Identities of the reference simulator over a trend path: fills in bar order, equity as cash
//! plus the marked position, and closed pnl equal to the cash change once flat.

use crate::sim::simulate;
use crate::tests::{Bars, close_to, coefficients};

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
    let r = simulate(&c, b.series(), &b.fired).unwrap();
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
}

#[test]
fn the_simulator_refuses_ragged_input() {
    let c = coefficients();
    let b = trend_path(30, 5, 5);
    let mut fired = b.fired.clone();
    fired.pop();
    assert!(simulate(&c, b.series(), &fired).is_err());
    let empty = Bars::default();
    assert!(simulate(&c, empty.series(), &[]).is_err());
}
