//! The instrument's price grid: every price the kernel compares or reports sits on
//! `10^-precision` steps, exactly as the simulated venue's `Price` does, so the reference
//! simulator and the venue agree bit for bit.

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
