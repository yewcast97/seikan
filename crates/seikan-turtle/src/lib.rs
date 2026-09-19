//! seikan-turtle — the simulation engine behind `seikan stock-turtle-trade-long-only`.
//!
//! Pure Rust with no market-data or venue dependency: prices arrive as `f64`s already quantized
//! to the instrument's price grid, the thesis's entry firing arrives as a boolean per bar, and
//! the engine answers with every fill it made under the rules and the cost model, every round
//! trip, a ledger of what it did and declined to do, and per-bar samples of its books.
//!
//! Modules: [`coefficients`] (the validated rule coefficients and unit sizing), [`costs`] (the
//! commission schedule, slippage, market impact and the stop-fill shock), [`price`] (the price
//! grid), [`indicators`] (Wilder ATR = the Turtle "N", the lowest-low channel, the average
//! volume), [`machine`] (the long-only state machine: the rules and the books),
//! [`execution`] (how an order becomes a fill) and [`sim`] (the bar loop that drives them).
//! [`error`] is the one error type. The PyO3 bindings ([`py`], feature `python`) expose
//! `simulate` and its result types as `seikan._turtle`.

#![warn(missing_docs)]

pub mod coefficients;
pub mod costs;
pub mod error;
pub mod execution;
pub mod indicators;
pub mod machine;
pub mod price;
pub mod sim;

#[cfg(feature = "python")]
mod py;

#[cfg(test)]
mod tests;
