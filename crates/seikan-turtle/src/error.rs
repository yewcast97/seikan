//! The crate's error type: why the engine refused, in three classes a caller treats differently.

use std::fmt;

/// Why the engine refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A coefficient or cost-model value the rules cannot be run under (refused at construction
    /// or at the start of a run; an input problem).
    Coefficients(String),
    /// Malformed simulation input: ragged arrays, no bars, a missing or unusable volume series.
    Input(String),
    /// An internal invariant broken during a run — a bug in this crate, never an input problem.
    Bookkeeping(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Coefficients(message) => write!(f, "invalid coefficients: {message}"),
            Self::Input(message) => write!(f, "invalid input: {message}"),
            Self::Bookkeeping(message) => write!(f, "bookkeeping invariant broken: {message}"),
        }
    }
}

impl std::error::Error for Error {}
