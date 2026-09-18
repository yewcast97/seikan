//! seikan-turtle — the Turtle position-management kernel behind
//! `seikan stock-turtle-trade-long-only`.
//!
//! The kernel is pure Rust with no market-data or venue dependency: prices arrive as `f64`s
//! already quantized to the instrument's price grid, the thesis's entry firing arrives as a
//! boolean per bar, and the kernel answers with decisions (what to do at the next opening print,
//! which sell stop to keep resting through the next bar) plus a ledger of everything it did and
//! declined to do. Two consumers drive it bit for bit identically:
//!
//! - the nautilus_trader strategy in `seikan.turtle.strategy` (through the PyO3 bindings in
//!   [`py`], built only with the `python` feature), which lets the simulated exchange execute
//!   the decisions; and
//! - [`sim::simulate`], the reference simulator, which applies the same decisions with idealized
//!   fills so that the venue's execution can be checked against the rules it is supposed to
//!   implement (the Python parity test) and the rules themselves can be unit-tested here.
//!
//! Modules: [`coefficients`] (the validated rule coefficients and unit sizing), [`price`] (the
//! price grid), [`indicators`] (Wilder ATR = the Turtle "N", the lowest-low channel),
//! [`machine`] (the long-only state machine) and [`sim`] (the reference simulator).

pub mod coefficients;
pub mod indicators;
pub mod machine;
pub mod price;
pub mod sim;

#[cfg(feature = "python")]
mod py;

#[cfg(test)]
mod tests;
