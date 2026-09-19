//! The state machine through the reference simulator: the rules' worked example under every
//! trigger mode, the gap rule, sizing caps, warmup/in-position/end-of-data skips, the
//! stop-never-down guard and the channel's current-bar exclusion.

use crate::coefficients::{NSource, Trigger, first_eligible_bar, unit_shares};
use crate::machine::{ExitReason, Fill, FillKind, Machine, PrintAction};
use crate::price::price_below;
use crate::sim::{FillAt, SimResult};
use crate::tests::{Bars, Lcg, close_to, coefficients, run, worked_example};

/// (bar, kind, shares, price, at, reason) — one row per reference fill.
type FillRow = (
    usize,
    &'static str,
    u64,
    f64,
    &'static str,
    Option<&'static str>,
);

fn fills(r: &SimResult) -> Vec<FillRow> {
    r.fills
        .iter()
        .map(|f| {
            (
                f.fill.bar,
                f.fill.kind.as_str(),
                f.fill.shares,
                f.fill.price,
                f.at.as_str(),
                f.reason,
            )
        })
        .collect()
}

#[test]
fn unit_shares_follows_the_worked_example() {
    assert_eq!(unit_shares(0.01, 100_000.0, 2.0, 2.0), 250);
    assert_eq!(unit_shares(0.01, 100_000.0, 2.0, 0.0), 0);
    assert_eq!(unit_shares(0.01, 100_000.0, 2.0, f64::NAN), 0);
    assert_eq!(unit_shares(0.01, 100_000.0, 2.0, 1000.0), 0);
    assert_eq!(first_eligible_bar(20, 20), 19);
    assert_eq!(first_eligible_bar(3, 10), 9);
}

#[test]
fn invalid_coefficients_refuse() {
    let mut c = coefficients();
    c.max_units = 0;
    assert!(Machine::new(c).is_err());
    let mut c = coefficients();
    c.risk_per_unit = 1.5;
    assert!(Machine::new(c).is_err());
    let mut c = coefficients();
    c.budget = 0.0;
    assert!(Machine::new(c).is_err());
    assert!(Machine::new(coefficients()).is_ok());
}

#[test]
fn worked_example_close_mode() {
    let c = coefficients();
    let b = worked_example();
    let r = run(&c, &b);
    assert_eq!(
        fills(&r),
        vec![
            (25, "entry", 250, 50.0, "open", None),
            (26, "add", 250, 51.0, "open", None),
            (27, "add", 250, 52.0, "open", None),
            (29, "exit", 750, 47.0, "open", Some("stop_close")),
        ]
    );
    // The stop after each fill: 46 → 47 → 48, never a fourth add (the cap).
    assert!(close_to(r.stop[25], 46.0));
    assert!(close_to(r.stop[26], 47.0));
    assert!(close_to(r.stop[27], 48.0));
    assert_eq!(r.units[27], 3);
    assert_eq!(r.shares[27], 750);
    assert!(r.stop[29].is_nan() && r.shares[29] == 0);
    // Bar 28 closes below BOTH the stop (48) and the channel (48.5): the stop takes precedence.
    assert_eq!(r.ledger.exits_stop_close, 1);
    assert_eq!(r.ledger.exits_channel_close, 0);
    assert_eq!((r.ledger.entries, r.ledger.adds), (1, 3 - 1));
    let trip = &r.trips[0];
    assert_eq!((trip.entry_bar, trip.exit_bar), (25, 29));
    assert_eq!(trip.exit_reason, ExitReason::StopClose);
    assert!(close_to(trip.cost_basis, 250.0 * (50.0 + 51.0 + 52.0)));
    assert!(close_to(trip.pnl, 750.0 * 47.0 - 38_250.0));
    assert!(close_to(trip.gross_pnl, trip.pnl));
    assert_eq!(
        (trip.commission, trip.slippage, trip.shock, trip.impact),
        (0.0, 0.0, 0.0, 0.0)
    );
    assert!(close_to(trip.max_stop, 48.0));
    assert!(close_to(
        r.cash[b.len() - 1],
        100_000.0 - 38_250.0 + 35_250.0
    ));
    assert!(r.open_trip.is_none());
    assert_eq!(r.first_eligible_bar, 19);
}

#[test]
fn worked_example_trade_stop_with_close_channel() {
    let mut c = coefficients();
    c.stop_trigger = Trigger::Trade;
    let b = worked_example();
    let r = run(&c, &b);
    // Only the stop rests (48 → 47.9999); bar 28's low of 47.5 trades through it.
    assert_eq!(
        fills(&r)[3],
        (28, "exit", 750, 47.9999, "trigger", Some("stop_trade"))
    );
    assert_eq!(r.ledger.exits_stop_trade, 1);
    assert_eq!(r.trips[0].exit_bar, 28);
}

#[test]
fn worked_example_full_trade_mode_rests_the_higher_level() {
    let mut c = coefficients();
    c.stop_trigger = Trigger::Trade;
    c.exit_trigger = Trigger::Trade;
    let b = worked_example();
    let r = run(&c, &b);
    // The channel (48.5, the flat period's lows) sits above the stop (48): the resting order is
    // the channel's, one grid step below it, and it is what bar 28 trades through.
    assert_eq!(
        fills(&r)[3],
        (28, "exit", 750, 48.4999, "trigger", Some("channel_trade"))
    );
    assert_eq!(r.ledger.exits_channel_trade, 1);
}

#[test]
fn trade_mode_tags_the_stop_when_it_is_the_higher_level() {
    let mut c = coefficients();
    c.stop_trigger = Trigger::Trade;
    c.exit_trigger = Trigger::Trade;
    let mut b = worked_example();
    // One deep low inside the channel window drags the channel below the stop; bar 28's low
    // trades through the stop resting after the (single, N-shifted) add.
    b.low[10] = 40.0;
    b.low[28] = 47.0;
    let r = run(&c, &b);
    let exit = r.fills.last().unwrap();
    assert_eq!(exit.reason, Some("stop_trade"));
    assert_eq!(exit.at, FillAt::Trigger);
    assert!(close_to(
        exit.fill.price,
        price_below(r.trips[0].max_stop, 4)
    ));
}

#[test]
fn close_mode_channel_exit_when_the_stop_is_not_hit() {
    let c = coefficients();
    let mut b = Bars::default();
    b.flat(24, 49.5, 1.0)
        .push(49.5, 51.0, 49.0, 50.0, true)
        .push(50.0, 50.5, 49.5, 50.0, false)
        // Closes below the channel (48.5) but above the stop (46).
        .push(50.0, 50.0, 48.0, 48.2, false)
        .push(48.0, 48.5, 47.5, 48.0, false)
        .flat(2, 48.0, 0.5);
    let r = run(&c, &b);
    assert_eq!(
        fills(&r).last().unwrap(),
        &(27, "exit", 250, 48.0, "open", Some("channel_close"))
    );
    assert_eq!(r.ledger.exits_channel_close, 1);
}

#[test]
fn gap_through_the_stop_sells_at_the_open_in_close_mode() {
    let c = coefficients();
    let mut b = Bars::default();
    b.flat(24, 49.5, 1.0)
        .push(49.5, 51.0, 49.0, 50.0, true)
        .push(50.0, 51.0, 49.5, 50.0, false) // entry at 50, stop 46
        .push(50.0, 50.5, 49.0, 49.0, false) // closes above the stop and the channel: nothing pending
        .push(45.0, 45.5, 44.0, 44.5, false) // gaps open at 45 < 46
        .flat(2, 44.5, 0.5);
    let r = run(&c, &b);
    assert_eq!(
        fills(&r).last().unwrap(),
        &(27, "exit", 250, 45.0, "open", Some("stop_gap"))
    );
    assert_eq!(r.ledger.exits_stop_gap, 1);
}

#[test]
fn gap_through_the_stop_fills_at_the_open_in_trade_mode() {
    let mut c = coefficients();
    c.stop_trigger = Trigger::Trade;
    let mut b = Bars::default();
    b.flat(24, 49.5, 1.0)
        .push(49.5, 51.0, 49.0, 50.0, true)
        .push(50.0, 51.0, 49.5, 50.0, false)
        .push(50.0, 50.5, 49.0, 49.0, false)
        .push(45.0, 45.5, 44.0, 44.5, false)
        .flat(2, 44.5, 0.5);
    let r = run(&c, &b);
    // The venue's resting stop (45.9999) is gapped through by the print: filled at the open.
    assert_eq!(
        fills(&r).last().unwrap(),
        &(27, "exit", 250, 45.0, "open", Some("stop_trade"))
    );
    assert_eq!(r.ledger.exits_stop_trade, 1);
    assert_eq!(r.ledger.exits_stop_gap, 0);
}

#[test]
fn an_add_fills_at_whatever_the_open_is() {
    let c = coefficients();
    let mut b = Bars::default();
    b.flat(24, 49.5, 1.0)
        .push(49.5, 51.0, 49.0, 50.0, true)
        .push(50.0, 52.0, 50.0, 51.0, false) // entry 50; close 51 = add level
        .push(56.0, 57.0, 55.0, 56.0, false) // opens 5 above the level: still adds, at 56
        .push(56.0, 56.5, 55.5, 56.0, false)
        .flat(2, 56.0, 0.5);
    let r = run(&c, &b);
    assert_eq!(fills(&r)[1], (26, "add", 250, 56.0, "open", None));
    // The ladder re-anchors on the ACTUAL fill: next level 56.5, stop 56 − 2N.
    assert!(close_to(r.add_level[26], 56.0 + 0.5 * r.atr[25]));
    assert_eq!(r.ledger.adds_filled_below_level, 0);
    // An open BELOW a pending add's level still fills (the close was the signal).
    let mut b2 = Bars::default();
    b2.flat(24, 49.5, 1.0)
        .push(49.5, 51.0, 49.0, 50.0, true)
        .push(50.0, 52.0, 50.0, 51.0, false)
        .push(50.5, 51.0, 50.0, 50.5, false)
        .flat(2, 50.5, 0.5);
    let r2 = run(&c, &b2);
    assert_eq!(fills(&r2)[1], (26, "add", 250, 50.5, "open", None));
    assert_eq!(r2.ledger.adds_filled_below_level, 1);
}

#[test]
fn stop_never_moves_down_when_the_current_n_grows() {
    let mut c = coefficients();
    c.stop_n_source = NSource::Current;
    let mut b = Bars::default();
    b.flat(24, 49.5, 1.0)
        .push(49.5, 51.0, 49.0, 50.0, true)
        // Entry at 50 (stop 46); a huge range lifts N to 3, so the add at 51 would put the
        // stop at 45 — held at 46 instead.
        .push(50.0, 52.0, 30.0, 51.0, false)
        .push(51.0, 51.5, 50.5, 51.0, false)
        .flat(3, 51.0, 0.5);
    let r = run(&c, &b);
    assert!(close_to(r.atr[25], 3.0));
    assert_eq!(fills(&r)[1].1, "add");
    assert!(close_to(r.stop[26], 46.0));
    assert_eq!(r.ledger.stop_holds, 1);
    // With the entry N the same add moves the stop up to 47.
    c.stop_n_source = NSource::Entry;
    let r = run(&c, &b);
    assert!(close_to(r.stop[26], 47.0));
    assert_eq!(r.ledger.stop_holds, 0);
}

#[test]
fn budget_caps_the_entry_and_skips_an_unaffordable_add() {
    let mut c = coefficients();
    c.budget = 1_000.0;
    let mut b = Bars::default();
    b.flat(24, 50.0, 0.05) // N = 0.1 → risk X = 50 shares, but only 20 fit the budget
        .push(50.0, 50.05, 49.95, 50.0, true)
        .push(50.0, 50.15, 50.0, 50.1, false) // entry 20 @ 50 (cash 0); close 50.1 ≥ 50.05: add pending
        .push(50.1, 50.15, 50.05, 50.1, false) // 20 × 50.1 > 0 cash: skipped
        .flat(2, 50.1, 0.02);
    let r = run(&c, &b);
    assert_eq!(fills(&r), vec![(25, "entry", 20, 50.0, "open", None)]);
    assert_eq!(r.ledger.entries_cash_capped, 1);
    // Bars 26, 27 and 28 each open with the add pending and no cash for it.
    assert_eq!(r.ledger.adds_skipped_budget, 3);
    assert!(close_to(r.cash[25], 0.0));
    assert_eq!(r.open_trip.as_ref().unwrap().adds_skipped_budget, 3);
}

#[test]
fn skips_are_ledgered() {
    let c = coefficients();
    let mut b = Bars::default();
    b.flat(24, 49.5, 1.0);
    b.fired[5] = true; // warmup: before bar 19
    b.push(49.5, 51.0, 49.0, 50.0, true)
        .push(50.0, 51.0, 49.5, 50.5, true) // in position
        .push(50.5, 51.0, 50.0, 50.5, false)
        .push(50.5, 51.0, 50.0, 50.5, false);
    let r = run(&c, &b);
    assert_eq!(r.ledger.entries_skipped_warmup, 1);
    assert_eq!(r.ledger.entries_skipped_in_position, 1);
    assert_eq!(r.ledger.entries, 1);
    // A firing on the final bar has no print to execute at.
    let mut b2 = Bars::default();
    b2.flat(24, 49.5, 1.0).push(49.5, 51.0, 49.0, 50.0, true);
    let r2 = run(&c, &b2);
    assert_eq!(r2.ledger.entries_skipped_end_of_data, 1);
    assert!(r2.fills.is_empty());
    // A zero-size unit (N far too large for the budget) never enters.
    let mut b3 = Bars::default();
    b3.flat(24, 5000.0, 600.0)
        .push(5000.0, 5600.0, 4400.0, 5000.0, true)
        .flat(2, 5000.0, 600.0);
    let r3 = run(&c, &b3);
    assert_eq!(r3.ledger.entries_skipped_zero_size, 1);
    assert!(r3.fills.is_empty());
}

#[test]
fn an_open_position_at_the_end_is_marked_as_end_of_data() {
    let c = coefficients();
    let mut b = Bars::default();
    b.flat(24, 49.5, 1.0)
        .push(49.5, 51.0, 49.0, 50.0, true)
        .push(50.0, 51.0, 49.5, 50.5, false)
        .push(50.5, 51.0, 50.0, 50.8, false);
    let r = run(&c, &b);
    let open = r.open_trip.as_ref().expect("still long at the end");
    assert_eq!(open.exit_reason, ExitReason::EndOfData);
    assert!(open.is_open());
    assert_eq!((open.entry_bar, open.exit_bar, open.shares), (25, 26, 250));
    assert!(close_to(open.exit_px, 50.8));
    assert!(close_to(open.pnl, 250.0 * (50.8 - 50.0)));
    assert!(r.trips.is_empty());
    assert_eq!(r.shares[26], 250);
    assert!(close_to(r.equity[26], r.cash[26] + 250.0 * 50.8));
}

#[test]
fn the_channel_excludes_the_current_bar() {
    let mut c = coefficients();
    c.atr_period = 3;
    c.exit_lookback = 3;
    let mut b = Bars::default();
    b.flat(3, 10.0, 1.0) // lows 9; N = 2; first eligible bar 2
        .push(10.0, 11.0, 9.0, 10.0, true)
        .push(10.0, 11.0, 9.0, 10.0, false) // entry 10 (stop 6)
        .push(10.0, 10.5, 8.0, 9.5, false) // own low 8 is a new low, close 9.5 ≥ previous lows (9)
        .push(9.5, 9.6, 8.4, 8.5, false) // close 8.5 ≥ min(9, 9, 8) = 8: still no exit
        .push(8.5, 8.6, 7.8, 7.9, false) // close 7.9 < min(9, 8, 8.4) = 8: exit
        .push(7.9, 8.0, 7.5, 7.9, false)
        .flat(2, 7.9, 0.2);
    let r = run(&c, &b);
    let exit = fills(&r).into_iter().find(|f| f.1 == "exit").unwrap();
    assert_eq!(exit, (8, "exit", 250, 7.9, "open", Some("channel_close")));
    assert_eq!(fills(&r).iter().filter(|f| f.1 == "exit").count(), 1);
}

#[test]
fn a_pending_close_exit_cancels_the_resting_channel_stop() {
    // stop: close, channel: trade. Bar 27 closes below the stop → exit pending → no stop rests
    // through bar 28, whose low would otherwise have hit the channel order.
    let mut c = coefficients();
    c.exit_trigger = Trigger::Trade;
    let mut b = Bars::default();
    b.flat(24, 49.5, 1.0)
        .push(49.5, 51.0, 49.0, 50.0, true)
        .push(50.0, 51.0, 49.5, 50.0, false) // entry 50, stop 46, channel 48.5 rests (48.4999)
        .push(50.0, 50.5, 49.0, 49.0, false)
        .push(49.0, 49.0, 48.6, 45.5, false) // closes below the stop; low stays above the channel
        .push(45.0, 45.5, 44.0, 44.5, false) // sold at the open, once
        .flat(2, 44.5, 0.5);
    let r = run(&c, &b);
    let exits: Vec<_> = fills(&r).into_iter().filter(|f| f.1 == "exit").collect();
    assert_eq!(
        exits,
        vec![(28, "exit", 250, 45.0, "open", Some("stop_close"))]
    );
    assert_eq!(r.ledger.exits_stop_close, 1);
    assert_eq!(r.shares[28], 0);
}

#[test]
fn one_add_per_bar_and_the_cap() {
    let mut c = coefficients();
    c.max_units = 2;
    let mut b = Bars::default();
    b.flat(24, 49.5, 1.0)
        .push(49.5, 51.0, 49.0, 50.0, true)
        .push(50.0, 60.0, 50.0, 58.0, false) // entry 50; close 58 is 4N above: still ONE add pending
        .push(58.0, 59.0, 57.0, 58.0, false) // add 250 @ 58 → cap of 2 units reached
        .push(58.0, 65.0, 58.0, 64.0, false) // no third unit
        .flat(2, 64.0, 0.5);
    let r = run(&c, &b);
    assert_eq!(
        fills(&r),
        vec![
            (25, "entry", 250, 50.0, "open", None),
            (26, "add", 250, 58.0, "open", None)
        ]
    );
    assert_eq!(r.units[28], 2);
    assert_eq!(r.ledger.adds, 1);
    assert_eq!(r.ledger.exits_stop_close + r.ledger.exits_channel_close, 0);
}

fn free_fill(kind: FillKind, shares: u64, price: f64) -> Fill {
    Fill {
        bar: 0,
        kind,
        shares,
        price,
        commission: 0.0,
        slippage: 0.0,
        shock: 0.0,
        impact: 0.0,
    }
}

#[test]
fn fill_bookkeeping_refuses_impossible_reports() {
    let mut m = Machine::new(coefficients()).unwrap();
    assert!(m.on_fill(free_fill(FillKind::Entry, 1, 1.0)).is_err());
    assert!(m.on_fill(free_fill(FillKind::Exit, 1, 1.0)).is_err());
}

#[test]
fn the_frictionless_size_search_is_the_sizing_division() {
    // Under a frictionless cash-out the largest affordable size must be exactly the guarded
    // division `floor(cash / open + 1e-9)` the rules state, on inputs away from the boundary.
    let mut rng = Lcg::new(3);
    for _ in 0..500 {
        let open = (rng.range(1.0, 500.0) * 100.0).round() / 100.0;
        let q_true = rng.range(1.0, 3000.0).floor();
        let cash = q_true * open + rng.range(0.001, 0.999) * open;
        let mut c = coefficients();
        c.atr_period = 1;
        c.exit_lookback = 1;
        c.risk_per_unit = 1.0;
        c.stop_n = 1e-3; // X = 1000 × budget: always more than the cash affords
        c.budget = cash;
        let mut m = Machine::new(c).unwrap();
        m.on_bar(0, open, Some(1.0), Some(open - 1.0), true, false);
        let action = m.on_print(1, open, &|q| q as f64 * open);
        let expected = ((cash / open) + 1e-9).floor() as u64;
        assert_eq!(expected, q_true as u64);
        assert_eq!(
            action,
            PrintAction::Buy {
                shares: expected,
                kind: FillKind::Entry
            }
        );
        assert_eq!(m.ledger().entries_cash_capped, 1);
    }
}

#[test]
fn a_commission_that_does_not_fit_drops_one_share() {
    // 20 shares @ 50 fit $1,000 exactly; a $1 minimum commission means only 19 fit all in.
    let mut c = coefficients();
    c.atr_period = 1;
    c.exit_lookback = 1;
    c.risk_per_unit = 1.0;
    c.stop_n = 0.01;
    c.budget = 1_000.0;
    let mut m = Machine::new(c).unwrap();
    m.on_bar(0, 50.0, Some(1.0), Some(49.0), true, false);
    let with_min = |q: u64| q as f64 * 50.0 + 1.0;
    assert_eq!(
        m.on_print(1, 50.0, &with_min),
        PrintAction::Buy {
            shares: 19,
            kind: FillKind::Entry
        }
    );
    // A whole unit that does not fit all in is skipped whole, never partially filled.
    let mut m = Machine::new(coefficients()).unwrap();
    m.on_bar(19, 50.0, Some(2.0), Some(48.0), true, false);
    assert_eq!(
        m.on_print(20, 50.0, &|q| q as f64 * 50.0),
        PrintAction::Buy {
            shares: 250,
            kind: FillKind::Entry
        }
    );
    m.on_fill(free_fill(FillKind::Entry, 250, 50.0)).unwrap();
    m.on_bar(20, 51.0, Some(2.0), Some(48.0), false, false);
    let too_dear = |q: u64| q as f64 * 51.0 + 90_000.0;
    assert_eq!(m.on_print(21, 51.0, &too_dear), PrintAction::Hold);
    assert_eq!(m.ledger().adds_skipped_budget, 1);
    assert_eq!(m.position().unwrap().adds_skipped_budget, 1);
}
