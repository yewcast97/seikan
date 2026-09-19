//! The indicators the engine reads, each in one idiom: `update_raw` per completed bar,
//! `value` (`None` until initialized), `initialized` / `has_inputs` / `count` / `reset`.

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

    /// Feed one completed bar.
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

    /// The current N, once initialized.
    pub fn value(&self) -> Option<f64> {
        self.value
    }

    /// Whether `period` bars are in.
    pub fn initialized(&self) -> bool {
        self.value.is_some()
    }

    /// Whether any bar is in.
    pub fn has_inputs(&self) -> bool {
        self.count > 0
    }

    /// Bars fed so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The lookback.
    pub fn period(&self) -> usize {
        self.period
    }

    /// Forget every bar.
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

    /// Feed one completed bar's low.
    pub fn update_raw(&mut self, low: f64) {
        self.lows.push_back(low);
        if self.lows.len() > self.lookback {
            self.lows.pop_front();
        }
        self.count += 1;
    }

    /// The channel level, once `lookback` bars are in.
    pub fn value(&self) -> Option<f64> {
        if self.lows.len() < self.lookback {
            return None;
        }
        self.lows.iter().copied().reduce(f64::min)
    }

    /// Whether `lookback` bars are in.
    pub fn initialized(&self) -> bool {
        self.lows.len() >= self.lookback
    }

    /// Whether any bar is in.
    pub fn has_inputs(&self) -> bool {
        self.count > 0
    }

    /// Bars fed so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The lookback.
    pub fn lookback(&self) -> usize {
        self.lookback
    }

    /// Forget every bar.
    pub fn reset(&mut self) {
        *self = Self::new(self.lookback);
    }
}

/// The mean volume of the last `window` bars handled, the newest included — the ADV the impact
/// law divides by. The mean is recomputed over the window each time (no running-sum drift).
#[derive(Clone, Debug, PartialEq)]
pub struct AverageVolume {
    window: usize,
    volumes: VecDeque<f64>,
    count: usize,
}

impl AverageVolume {
    /// `window` must be >= 1.
    pub fn new(window: usize) -> Self {
        assert!(window >= 1, "AverageVolume window must be >= 1");
        Self {
            window,
            volumes: VecDeque::with_capacity(window + 1),
            count: 0,
        }
    }

    /// Feed one completed bar's volume.
    pub fn update_raw(&mut self, volume: f64) {
        self.volumes.push_back(volume);
        if self.volumes.len() > self.window {
            self.volumes.pop_front();
        }
        self.count += 1;
    }

    /// The average, once `window` bars are in.
    pub fn value(&self) -> Option<f64> {
        if self.volumes.len() < self.window {
            return None;
        }
        Some(self.volumes.iter().sum::<f64>() / self.window as f64)
    }

    /// Whether `window` bars are in.
    pub fn initialized(&self) -> bool {
        self.volumes.len() >= self.window
    }

    /// Whether any bar is in.
    pub fn has_inputs(&self) -> bool {
        self.count > 0
    }

    /// Bars fed so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The window.
    pub fn window(&self) -> usize {
        self.window
    }

    /// Forget every bar.
    pub fn reset(&mut self) {
        *self = Self::new(self.window);
    }
}
