use crate::price::{price_below, quantize, step_down, step_up};
use crate::tests::close_to;

#[test]
fn quantize_rounds_to_the_grid() {
    assert!(close_to(quantize(101.352_420_563_040_28, 4), 101.3524));
    assert!(close_to(quantize(101.352_45, 4), 101.3525));
    assert!(close_to(quantize(2.5, 0), 3.0));
    assert!(close_to(quantize(48.0, 2), 48.0));
}

#[test]
fn price_below_is_the_highest_grid_price_strictly_under_the_level() {
    assert!(close_to(price_below(46.0, 4), 45.9999));
    assert!(close_to(price_below(46.000_03, 4), 46.0));
    assert!(close_to(price_below(45.999_95, 4), 45.9999));
    assert!(close_to(price_below(48.0, 2), 47.99));
    assert!(close_to(price_below(0.1 + 0.2, 4), 0.2999));
    // A level that only LOOKS off-grid through float noise still counts as on it.
    assert!(close_to(price_below(51.0 + 0.5 * 2.0, 4), 51.9999));
}

#[test]
fn a_zero_adverse_step_is_the_reference_itself_at_every_precision() {
    for precision in 0..=9 {
        for raw in [
            50.0,
            12_345.123_456_789,
            0.3,
            101.352_420_563_040_28,
            9_999.999_999_9,
        ] {
            let reference = quantize(raw, precision);
            assert_eq!(
                step_up(reference, 0.0, precision).to_bits(),
                reference.to_bits()
            );
            assert_eq!(
                step_down(reference, 0.0, precision).to_bits(),
                reference.to_bits()
            );
        }
        let trigger = price_below(48.0, precision);
        assert_eq!(
            step_down(trigger, 0.0, precision).to_bits(),
            trigger.to_bits()
        );
    }
}

#[test]
fn a_costed_step_moves_whole_increments_against_the_account() {
    assert!(close_to(step_up(50.0, 0.025, 4), 50.025)); // exactly 250 steps
    assert!(close_to(step_up(50.0, 0.025_01, 4), 50.0251)); // one step more, rounded up
    assert!(close_to(step_down(50.0, 0.025, 4), 49.975));
    assert!(close_to(step_down(47.9999, 0.249_95, 4), 47.7499));
    // A sub-nanostep distance is float noise, not a cost.
    assert!(close_to(step_up(50.0, 1e-14, 4), 50.0));
    // Precision 9 on a five-figure price: the step count is exact.
    let reference = quantize(12_345.123_456_789, 9);
    let moved = step_up(reference, 0.000_000_003, 9);
    assert!(((moved - reference) * 1e9 - 3.0).abs() < 1e-3);
}
