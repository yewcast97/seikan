use crate::price::{price_below, quantize};
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
