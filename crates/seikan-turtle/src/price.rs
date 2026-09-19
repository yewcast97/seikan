//! The instrument's price grid: every price the engine compares or reports sits on
//! `10^-precision` steps. References (quantized opens, `price_below` triggers) are ON the grid by
//! construction, and a costed fill moves a whole number of steps away from its reference, so a
//! zero-cost fill is the reference itself, bit for bit, at every precision.

/// Round `price` to the nearest grid point of `10^-precision`.
pub fn quantize(price: f64, precision: u32) -> f64 {
    let factor = 10f64.powi(precision as i32);
    (price * factor).round() / factor
}

/// The highest grid price STRICTLY below `level` — where a "trades below the level" stop rests.
/// A level already on the grid moves one increment down; a level between grid points floors.
pub fn price_below(level: f64, precision: u32) -> f64 {
    let factor = 10f64.powi(precision as i32);
    let scaled = level * factor;
    let nearest = scaled.round();
    // On-grid within a millionth of one increment: float noise, never a real offset.
    let on_grid = (nearest - scaled).abs() <= 1e-6;
    let steps = if on_grid {
        nearest - 1.0
    } else {
        scaled.floor()
    };
    steps / factor
}

/// `reference` moved `adverse` (a non-negative price distance) UP onto the grid: the reference's
/// own grid index plus `ceil(adverse × 10^precision − 1e-9)` steps — a buyer's fill. The
/// reference must be a grid price; a zero `adverse` returns the identical float.
pub fn step_up(reference: f64, adverse: f64, precision: u32) -> f64 {
    let factor = 10f64.powi(precision as i32);
    (grid_index(reference, factor) + adverse_steps(adverse, factor)) / factor
}

/// `reference` moved `adverse` DOWN onto the grid — a seller's fill (see [`step_up`]).
pub fn step_down(reference: f64, adverse: f64, precision: u32) -> f64 {
    let factor = 10f64.powi(precision as i32);
    (grid_index(reference, factor) - adverse_steps(adverse, factor)) / factor
}

/// The grid index of an on-grid price (the rounding absorbs the float noise of `k / factor`).
fn grid_index(reference: f64, factor: f64) -> f64 {
    (reference * factor).round()
}

/// Whole steps covering `adverse`, rounded against the account; a sub-nanostep distance is
/// float noise, not a cost.
fn adverse_steps(adverse: f64, factor: f64) -> f64 {
    (adverse * factor - 1e-9).ceil().max(0.0)
}
