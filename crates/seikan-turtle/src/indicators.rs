//! The two indicators the Turtle rules read, in the nautilus_trader indicator idiom
//! (`update_raw` / `value` / `initialized` / `has_inputs` / `count` / `reset`). The PyO3
//! bindings add `handle_bar(bar)` on top, so a nautilus strategy can register them as
//! bar-driven indicators.

use std::collections::VecDeque;

/// Wilder's average true range — the Turtle "N".
///
/// `TR_0 = H - L`, `TR_t = max(H - L, |H - C_{t-1}|, |C_{t-1} - L|)`; the first value is the
/// simple mean of the first `period` true ranges, after which
/// `N_t = ((period - 1) × N_{t-1} + TR_t) / period`. Initialized once `period` bars are in.
#[derive(Clone, Debug, PartialEq)]
pub struct WilderAtr {
    period: usize,
    count: usize,
    prev_close: Option<f64>,
    seed_sum: f64,
    value: Option<f64>,
}

impl WilderAtr {
    /// `period` must be >= 1.
    pub fn new(period: usize) -> Self {
        assert!(period >= 1, "WilderAtr period must be >= 1");
        Self {
            period,
            count: 0,
            prev_close: None,
            seed_sum: 0.0,
            value: None,
        }
    }

    pub fn update_raw(&mut self, high: f64, low: f64, close: f64) {
        let range = high - low;
        let tr = match self.prev_close {
            Some(pc) => range.max((high - pc).abs()).max((pc - low).abs()),
            None => range,
        };
        self.count += 1;
        match self.value {
            Some(n) => {
                let p = self.period as f64;
                self.value = Some(((p - 1.0) * n + tr) / p);
            }
            None => {
                self.seed_sum += tr;
                if self.count >= self.period {
                    self.value = Some(self.seed_sum / self.period as f64);
                }
            }
        }
        self.prev_close = Some(close);
    }

    pub fn value(&self) -> Option<f64> {
        self.value
    }

    pub fn initialized(&self) -> bool {
        self.value.is_some()
    }

    pub fn has_inputs(&self) -> bool {
        self.count > 0
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn period(&self) -> usize {
        self.period
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.period);
    }
}

/// The lowest low of the last `lookback` bars handled, the newest INCLUDED.
///
/// The Turtle exit compares a close with the lowest low of the bars BEFORE it, so the machine
/// reads this indicator's value as it stood before the current bar was handled; the value after
/// the current bar is the level in force during the next bar (the resting stop's channel leg).
#[derive(Clone, Debug, PartialEq)]
pub struct LowestLowChannel {
    lookback: usize,
    lows: VecDeque<f64>,
    count: usize,
}

impl LowestLowChannel {
    /// `lookback` must be >= 1.
    pub fn new(lookback: usize) -> Self {
        assert!(lookback >= 1, "LowestLowChannel lookback must be >= 1");
        Self {
            lookback,
            lows: VecDeque::with_capacity(lookback + 1),
            count: 0,
        }
    }

    pub fn update_raw(&mut self, low: f64) {
        self.lows.push_back(low);
        if self.lows.len() > self.lookback {
            self.lows.pop_front();
        }
        self.count += 1;
    }

    pub fn value(&self) -> Option<f64> {
        if self.lows.len() < self.lookback {
            return None;
        }
        self.lows.iter().copied().reduce(f64::min)
    }

    pub fn initialized(&self) -> bool {
        self.lows.len() >= self.lookback
    }

    pub fn has_inputs(&self) -> bool {
        self.count > 0
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn lookback(&self) -> usize {
        self.lookback
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.lookback);
    }
}
