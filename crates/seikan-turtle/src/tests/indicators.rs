use crate::indicators::{LowestLowChannel, WilderAtr};
use crate::tests::close_to;

/// Faith's HO03H table (The Original Turtle Trading Rules, ch. 3): high, low, close, TR, N.
const HO03H: &[(f64, f64, f64, f64, f64)] = &[
    (0.7220, 0.7124, 0.7124, 0.0096, 0.0134),
    (0.7170, 0.7073, 0.7073, 0.0097, 0.0132),
    (0.7099, 0.6923, 0.6923, 0.0176, 0.0134),
    (0.6930, 0.6800, 0.6838, 0.0130, 0.0134),
    (0.6960, 0.6736, 0.6736, 0.0224, 0.0139),
    (0.6820, 0.6706, 0.6706, 0.0114, 0.0137),
    (0.6820, 0.6710, 0.6710, 0.0114, 0.0136),
    (0.6795, 0.6720, 0.6744, 0.0085, 0.0134),
    (0.6760, 0.6550, 0.6616, 0.0210, 0.0138),
    (0.6650, 0.6585, 0.6627, 0.0065, 0.0134),
    (0.6701, 0.6620, 0.6701, 0.0081, 0.0131),
    (0.6965, 0.6750, 0.6965, 0.0264, 0.0138),
    (0.7065, 0.6944, 0.6944, 0.0121, 0.0137),
    (0.7115, 0.6944, 0.7087, 0.0171, 0.0139),
    (0.7168, 0.7100, 0.7124, 0.0081, 0.0136),
    (0.7265, 0.7120, 0.7265, 0.0145, 0.0136),
    (0.7265, 0.7098, 0.7098, 0.0167, 0.0138),
    (0.7184, 0.7110, 0.7184, 0.0086, 0.0135),
    (0.7280, 0.7200, 0.7228, 0.0096, 0.0133),
    (0.7375, 0.7227, 0.7359, 0.0148, 0.0134),
    (0.7447, 0.7310, 0.7389, 0.0137, 0.0134),
    (0.7420, 0.7140, 0.7162, 0.0280, 0.0141),
];

#[test]
fn wilder_atr_follows_faith_table() {
    // The table starts mid-series: seed the indicator with 20 bars of constant true range equal
    // to the PDN the first row implies ((19·PDN + 0.0096)/20 = 0.0134) and a previous close of
    // 0.7124, then feed the rows and check N against the published column.
    let pdn = (0.0134 * 20.0 - 0.0096) / 19.0;
    let mut atr = WilderAtr::new(20);
    for _ in 0..20 {
        atr.update_raw(0.7124 + pdn / 2.0, 0.7124 - pdn / 2.0, 0.7124);
    }
    assert!(atr.initialized());
    assert!(close_to(atr.value().unwrap(), pdn));
    let mut drift = 0.0f64;
    for (i, &(h, l, c, _tr_doc, n_doc)) in HO03H.iter().enumerate() {
        atr.update_raw(h, l, c);
        let n = atr.value().unwrap();
        drift = drift.max((n - n_doc).abs());
        assert!((n - n_doc).abs() < 1.5e-4, "row {i}: N {n} vs {n_doc}");
    }
    assert!(drift < 1.5e-4, "N drift {drift}");
    assert_eq!(atr.count(), 42);
}

#[test]
fn wilder_atr_seeds_with_the_mean_true_range_then_smooths() {
    let mut atr = WilderAtr::new(3);
    assert!(!atr.has_inputs());
    atr.update_raw(11.0, 9.0, 10.0); // TR 2 (no previous close)
    assert!(atr.has_inputs() && !atr.initialized() && atr.value().is_none());
    atr.update_raw(13.0, 10.5, 12.0); // TR max(2.5, 3, 0.5) = 3
    assert!(!atr.initialized());
    atr.update_raw(12.5, 11.5, 12.0); // TR max(1, 0.5, 0.5) = 1 -> seed (2+3+1)/3 = 2
    assert!(atr.initialized());
    assert!(close_to(atr.value().unwrap(), 2.0));
    atr.update_raw(15.0, 14.0, 14.5); // TR max(1, 3, 2) = 3 -> (2·2 + 3)/3
    assert!(close_to(atr.value().unwrap(), 7.0 / 3.0));
    atr.reset();
    assert!(!atr.has_inputs() && !atr.initialized() && atr.period() == 3);
}

#[test]
fn lowest_low_channel_covers_the_last_lookback_lows_including_the_newest() {
    let mut ch = LowestLowChannel::new(3);
    assert!(!ch.has_inputs() && !ch.initialized());
    for low in [5.0, 4.0] {
        ch.update_raw(low);
        assert!(ch.value().is_none() && !ch.initialized());
    }
    ch.update_raw(6.0);
    assert_eq!(ch.value(), Some(4.0));
    ch.update_raw(3.0);
    assert_eq!(ch.value(), Some(3.0));
    ch.update_raw(7.0);
    assert_eq!(ch.value(), Some(3.0));
    ch.update_raw(8.0);
    assert_eq!(ch.value(), Some(3.0));
    ch.update_raw(9.0);
    assert_eq!(ch.value(), Some(7.0));
    assert_eq!((ch.count(), ch.lookback()), (7, 3));
    ch.reset();
    assert!(ch.value().is_none() && ch.count() == 0);
}
