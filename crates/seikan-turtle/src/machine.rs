//! The long-only Turtle state machine for ONE target.
//!
//! Bar stamps are decision times. Every decision is taken on a bar's close and executed at the
//! NEXT bar's opening print — except the trade-triggered stop and channel exit, which rest at the
//! venue as ONE sell stop and fill wherever the next bar trades through them. The only order that
//! ever rests through a bar is that sell stop, so an entry, an add and an exit never conflict
//! inside one bar.
//!
//! The rules, as implemented (see `CONTRACT.md` for the prose):
//!
//! - **Entry.** The thesis fires on bar `t` while flat and past warmup → market buy at the print
//!   of `t+1`. `N_entry` is the ATR at bar `t`; `X = floor(risk_per_unit × budget / (stop_n ×
//!   N_entry))`, capped at the print to what the target's cash affords. Initial stop
//!   `P0 − stop_n × N_entry`.
//! - **Adds** (no gap guard). While `units < max_units`, a close at or above
//!   `last_fill + add_step_n × N_entry` buys another X shares at the next print, wherever it opens
//!   — unless X shares do not fit the remaining cash, in which case the add is skipped whole and
//!   re-arms at the next close. After an add the stop is `max(stop, fill − stop_n × N_stop)`, `N_stop`
//!   being the current ATR or the entry ATR per `stop_n_source`; it never moves down.
//! - **Stop-loss / channel exit**, each `close` or `trade`. Close mode: `close < stop` →
//!   `stop_close`, else `close < channel` → `channel_close`, sold at the next print; an opening
//!   print below the stop with no exit pending sells at that open (`stop_gap`). Trade mode: one
//!   resting sell stop one grid step below the higher applicable level (`stop_trade` when the stop
//!   is the higher level, else `channel_trade`); a gapped print fills it at the open by the venue
//!   itself. The channel is the lowest low of the previous `exit_lookback` completed bars,
//!   the current bar excluded.
//! - Print precedence: pending exit → close-mode gap-through stop → pending entry/add (a pending
//!   add is dropped when the venue's resting stop already exited at the print).

use crate::coefficients::{Coefficients, NSource, Trigger, unit_shares};
use crate::price::price_below;

/// What a fill was for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillKind {
    Entry,
    Add,
    Exit,
}

impl FillKind {
    pub fn parse(text: &str) -> Result<Self, String> {
        match text {
            "entry" => Ok(Self::Entry),
            "add" => Ok(Self::Add),
            "exit" => Ok(Self::Exit),
            other => Err(format!(
                "fill kind must be 'entry', 'add' or 'exit', got {other:?}"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Entry => "entry",
            Self::Add => "add",
            Self::Exit => "exit",
        }
    }
}

/// Why a position closed (or, for `EndOfData`, was marked open at the last bar).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitReason {
    StopClose,
    StopGap,
    StopTrade,
    ChannelClose,
    ChannelTrade,
    EndOfData,
}

impl ExitReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StopClose => "stop_close",
            Self::StopGap => "stop_gap",
            Self::StopTrade => "stop_trade",
            Self::ChannelClose => "channel_close",
            Self::ChannelTrade => "channel_trade",
            Self::EndOfData => "end_of_data",
        }
    }
}

/// The open position.
#[derive(Clone, Debug, PartialEq)]
pub struct Position {
    pub entry_bar: usize,
    pub entry_px: f64,
    pub unit_shares: u64,
    pub units: u32,
    pub shares: u64,
    pub last_fill_px: f64,
    pub n_entry: f64,
    pub stop: f64,
    pub cost_basis: f64,
    pub n_adds: u32,
    pub adds_skipped_budget: u32,
}

impl Position {
    /// The next add level: `add_step_n × N_entry` above the latest fill.
    pub fn add_level(&self, add_step_n: f64) -> f64 {
        self.last_fill_px + add_step_n * self.n_entry
    }
}

/// The decision taken at a close, waiting for the next opening print.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pending {
    None,
    Enter { shares: u64, n: f64 },
    Add { level: f64, n_for_stop: f64 },
    Exit(ExitReason),
}

impl Pending {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Enter { .. } => "enter",
            Self::Add { .. } => "add",
            Self::Exit(_) => "exit",
        }
    }
}

/// What the strategy submits at an opening print.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrintAction {
    Hold,
    Buy { shares: u64, kind: FillKind },
    Sell { shares: u64, reason: ExitReason },
}

/// The one sell stop to keep resting at the venue through the next bar (trade modes only).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StopOrder {
    pub trigger: f64,
    pub shares: u64,
    pub reason: ExitReason,
}

/// One fill reported back to the machine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fill {
    pub bar: usize,
    pub kind: FillKind,
    pub shares: u64,
    pub price: f64,
}

/// Everything the machine did and declined to do, counted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ledger {
    pub entries: u64,
    pub adds: u64,
    pub exits_stop_close: u64,
    pub exits_stop_gap: u64,
    pub exits_stop_trade: u64,
    pub exits_channel_close: u64,
    pub exits_channel_trade: u64,
    pub adds_skipped_budget: u64,
    pub adds_filled_below_level: u64,
    pub entries_skipped_warmup: u64,
    pub entries_skipped_in_position: u64,
    pub entries_skipped_zero_size: u64,
    pub entries_skipped_budget: u64,
    pub entries_skipped_end_of_data: u64,
    pub entries_cash_capped: u64,
    pub stop_holds: u64,
}

/// One position from entry to exit (or to the last bar, marked, when `exit_reason` is
/// `EndOfData`).
#[derive(Clone, Debug, PartialEq)]
pub struct RoundTrip {
    pub entry_bar: usize,
    pub exit_bar: usize,
    pub exit_reason: ExitReason,
    pub entry_px: f64,
    pub exit_px: f64,
    pub unit_shares: u64,
    pub units: u32,
    pub shares: u64,
    pub cost_basis: f64,
    pub proceeds: f64,
    pub pnl: f64,
    pub n_adds: u32,
    pub adds_skipped_budget: u32,
    pub max_stop: f64,
}

impl RoundTrip {
    pub fn is_open(&self) -> bool {
        self.exit_reason == ExitReason::EndOfData
    }
}

/// What the machine is waiting to hear back about after issuing a print action.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Armed {
    Entry { n: f64 },
    Add { level: f64, n_for_stop: f64 },
    Exit(ExitReason),
}

/// The state machine for one target. Drive it in bar order: `on_print` at each opening print,
/// `on_fill` for every fill (print fills and resting-stop fills alike), `on_bar` at each close,
/// `finish` after the last bar; read `resting()` after `on_print`'s fills and after `on_bar` to
/// learn which sell stop (if any) must rest at the venue.
#[derive(Clone, Debug)]
pub struct Machine {
    c: Coefficients,
    first_eligible: usize,
    cash: f64,
    pos: Option<Position>,
    pending: Pending,
    armed: Option<Armed>,
    resting: Option<StopOrder>,
    channel_prev: Option<f64>,
    ledger: Ledger,
    closed: Vec<RoundTrip>,
}

impl Machine {
    pub fn new(c: Coefficients) -> Result<Self, String> {
        c.validate()?;
        Ok(Self {
            first_eligible: c.first_eligible_bar(),
            cash: c.budget,
            pos: None,
            pending: Pending::None,
            armed: None,
            resting: None,
            channel_prev: None,
            ledger: Ledger::default(),
            closed: Vec::new(),
            c,
        })
    }

    // ---- reads ------------------------------------------------------------------------------

    pub fn coefficients(&self) -> &Coefficients {
        &self.c
    }

    pub fn cash(&self) -> f64 {
        self.cash
    }

    pub fn position(&self) -> Option<&Position> {
        self.pos.as_ref()
    }

    pub fn shares(&self) -> u64 {
        self.pos.as_ref().map_or(0, |p| p.shares)
    }

    pub fn units(&self) -> u32 {
        self.pos.as_ref().map_or(0, |p| p.units)
    }

    pub fn in_position(&self) -> bool {
        self.pos.is_some()
    }

    pub fn stop(&self) -> Option<f64> {
        self.pos.as_ref().map(|p| p.stop)
    }

    pub fn add_level(&self) -> Option<f64> {
        self.pos.as_ref().map(|p| p.add_level(self.c.add_step_n))
    }

    pub fn pending(&self) -> Pending {
        self.pending
    }

    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    pub fn closed(&self) -> &[RoundTrip] {
        &self.closed
    }

    pub fn first_eligible_bar(&self) -> usize {
        self.first_eligible
    }

    /// The channel level in force for the NEXT bar (the lowest low of the last `exit_lookback`
    /// completed bars), once known.
    pub fn channel(&self) -> Option<f64> {
        self.channel_prev
    }

    /// Cash plus the position marked at `close`.
    pub fn equity(&self, close: f64) -> f64 {
        self.cash + self.shares() as f64 * close
    }

    /// The sell stop to keep resting at the venue from now on: the higher of the trade-mode
    /// levels (stop, channel), one grid step down, for the whole position — or nothing, when
    /// flat, when no level is trade-triggered, or when a close-mode exit is already pending (the
    /// print sells the position; a resting stop as well would sell it twice).
    pub fn resting(&self) -> Option<StopOrder> {
        self.resting
    }

    fn plan_resting(&self) -> Option<StopOrder> {
        let p = self.pos.as_ref()?;
        if matches!(self.pending, Pending::Exit(_)) {
            return None;
        }
        let mut best: Option<(f64, ExitReason)> = None;
        if self.c.stop_trigger == Trigger::Trade {
            best = Some((p.stop, ExitReason::StopTrade));
        }
        if self.c.exit_trigger == Trigger::Trade
            && let Some(ch) = self.channel_prev
            && !best.is_some_and(|(level, _)| level >= ch)
        {
            best = Some((ch, ExitReason::ChannelTrade));
        }
        best.map(|(level, reason)| StopOrder {
            trigger: price_below(level, self.c.price_precision),
            shares: p.shares,
            reason,
        })
    }

    // ---- transitions ------------------------------------------------------------------------

    /// The opening print of `bar`: the one action to submit at this open. Every fill that
    /// results is reported through [`Machine::on_fill`].
    pub fn on_print(&mut self, bar: usize, open: f64) -> PrintAction {
        let _ = bar;
        let pending = std::mem::replace(&mut self.pending, Pending::None);
        let action = match (&self.pos, pending) {
            (Some(p), Pending::Exit(reason)) => {
                self.armed = Some(Armed::Exit(reason));
                PrintAction::Sell {
                    shares: p.shares,
                    reason,
                }
            }
            // The gap rule (close-mode stop): an open below the stop sells at that open, whatever
            // else was pending. In trade mode the resting stop already did this at the venue.
            (Some(p), _) if self.c.stop_trigger == Trigger::Close && open < p.stop => {
                self.armed = Some(Armed::Exit(ExitReason::StopGap));
                PrintAction::Sell {
                    shares: p.shares,
                    reason: ExitReason::StopGap,
                }
            }
            (None, Pending::Enter { shares, n }) => {
                let affordable = ((self.cash / open) + 1e-9).floor().max(0.0) as u64;
                let filled = shares.min(affordable);
                if filled == 0 {
                    self.ledger.entries_skipped_budget += 1;
                    PrintAction::Hold
                } else {
                    if filled < shares {
                        self.ledger.entries_cash_capped += 1;
                    }
                    self.armed = Some(Armed::Entry { n });
                    PrintAction::Buy {
                        shares: filled,
                        kind: FillKind::Entry,
                    }
                }
            }
            (Some(p), Pending::Add { level, n_for_stop }) => {
                let cost = p.unit_shares as f64 * open;
                if p.units >= self.c.max_units {
                    PrintAction::Hold
                } else if cost > self.cash * (1.0 + 1e-12) {
                    self.ledger.adds_skipped_budget += 1;
                    let p = self.pos.as_mut().expect("position present");
                    p.adds_skipped_budget += 1;
                    PrintAction::Hold
                } else {
                    self.armed = Some(Armed::Add { level, n_for_stop });
                    PrintAction::Buy {
                        shares: p.unit_shares,
                        kind: FillKind::Add,
                    }
                }
            }
            // A pending add whose position the venue's resting stop already closed at this
            // print, a pending exit on a position already gone, or nothing pending at all.
            _ => PrintAction::Hold,
        };
        if matches!(action, PrintAction::Hold) {
            self.resting = self.plan_resting();
        }
        action
    }

    /// A fill at the venue: a print fill of the action [`Machine::on_print`] returned, or a fill
    /// of the resting sell stop. Updates cash, the position, the ledger and the resting plan.
    pub fn on_fill(&mut self, fill: Fill) -> Result<(), String> {
        match fill.kind {
            FillKind::Entry => {
                let Some(Armed::Entry { n }) = self.armed.take() else {
                    return Err("entry fill reported without an armed entry".into());
                };
                if self.pos.is_some() {
                    return Err("entry fill reported while a position is open".into());
                }
                let cost = fill.shares as f64 * fill.price;
                self.pos = Some(Position {
                    entry_bar: fill.bar,
                    entry_px: fill.price,
                    unit_shares: fill.shares,
                    units: 1,
                    shares: fill.shares,
                    last_fill_px: fill.price,
                    n_entry: n,
                    stop: fill.price - self.c.stop_n * n,
                    cost_basis: cost,
                    n_adds: 0,
                    adds_skipped_budget: 0,
                });
                self.cash -= cost;
                self.ledger.entries += 1;
            }
            FillKind::Add => {
                let Some(Armed::Add { level, n_for_stop }) = self.armed.take() else {
                    return Err("add fill reported without an armed add".into());
                };
                let Some(p) = self.pos.as_mut() else {
                    return Err("add fill reported while flat".into());
                };
                if fill.shares != p.unit_shares {
                    return Err(format!(
                        "add fill of {} shares, expected one unit of {}",
                        fill.shares, p.unit_shares
                    ));
                }
                let cost = fill.shares as f64 * fill.price;
                p.units += 1;
                p.shares += fill.shares;
                p.last_fill_px = fill.price;
                p.cost_basis += cost;
                p.n_adds += 1;
                if fill.price < level {
                    self.ledger.adds_filled_below_level += 1;
                }
                let new_stop = fill.price - self.c.stop_n * n_for_stop;
                if new_stop > p.stop {
                    p.stop = new_stop;
                } else {
                    self.ledger.stop_holds += 1;
                }
                self.cash -= cost;
                self.ledger.adds += 1;
            }
            FillKind::Exit => {
                let reason = match self.armed.take() {
                    Some(Armed::Exit(reason)) => reason,
                    Some(other) => {
                        return Err(format!("exit fill reported while armed for {other:?}"));
                    }
                    None => match self.resting {
                        Some(order) => order.reason,
                        None => {
                            return Err("exit fill reported with no exit armed or resting".into());
                        }
                    },
                };
                let Some(p) = self.pos.take() else {
                    return Err("exit fill reported while flat".into());
                };
                if fill.shares != p.shares {
                    return Err(format!(
                        "exit fill of {} shares, expected the whole position of {}",
                        fill.shares, p.shares
                    ));
                }
                let proceeds = fill.shares as f64 * fill.price;
                self.cash += proceeds;
                self.closed.push(RoundTrip {
                    entry_bar: p.entry_bar,
                    exit_bar: fill.bar,
                    exit_reason: reason,
                    entry_px: p.entry_px,
                    exit_px: fill.price,
                    unit_shares: p.unit_shares,
                    units: p.units,
                    shares: p.shares,
                    cost_basis: p.cost_basis,
                    proceeds,
                    pnl: proceeds - p.cost_basis,
                    n_adds: p.n_adds,
                    adds_skipped_budget: p.adds_skipped_budget,
                    max_stop: p.stop,
                });
                match reason {
                    ExitReason::StopClose => self.ledger.exits_stop_close += 1,
                    ExitReason::StopGap => self.ledger.exits_stop_gap += 1,
                    ExitReason::StopTrade => self.ledger.exits_stop_trade += 1,
                    ExitReason::ChannelClose => self.ledger.exits_channel_close += 1,
                    ExitReason::ChannelTrade => self.ledger.exits_channel_trade += 1,
                    ExitReason::EndOfData => {
                        return Err("end_of_data is a mark, never a fill".into());
                    }
                }
                self.pending = Pending::None;
            }
        }
        self.resting = self.plan_resting();
        Ok(())
    }

    /// The close of `bar`: `n` and `channel_now` are the indicator values AFTER this bar,
    /// `fired` whether the thesis fires on it, `last_bar` whether no print follows. Decides what
    /// the next print does and which stop (if any) rests through the next bar.
    pub fn on_bar(
        &mut self,
        bar: usize,
        close: f64,
        n: Option<f64>,
        channel_now: Option<f64>,
        fired: bool,
        last_bar: bool,
    ) {
        self.pending = Pending::None;
        match &self.pos {
            Some(p) => {
                if self.c.stop_trigger == Trigger::Close && close < p.stop {
                    self.pending = Pending::Exit(ExitReason::StopClose);
                } else if self.c.exit_trigger == Trigger::Close
                    && self.channel_prev.is_some_and(|ch| close < ch)
                {
                    self.pending = Pending::Exit(ExitReason::ChannelClose);
                } else if p.units < self.c.max_units {
                    let level = p.add_level(self.c.add_step_n);
                    if close >= level {
                        let n_for_stop = match self.c.stop_n_source {
                            NSource::Entry => p.n_entry,
                            NSource::Current => n.unwrap_or(p.n_entry),
                        };
                        self.pending = Pending::Add { level, n_for_stop };
                    }
                }
                if fired {
                    self.ledger.entries_skipped_in_position += 1;
                }
            }
            None => {
                if fired {
                    if last_bar {
                        self.ledger.entries_skipped_end_of_data += 1;
                    } else if bar < self.first_eligible || n.is_none() || channel_now.is_none() {
                        self.ledger.entries_skipped_warmup += 1;
                    } else {
                        let n = n.expect("checked above");
                        let shares =
                            unit_shares(self.c.risk_per_unit, self.c.budget, self.c.stop_n, n);
                        if shares == 0 {
                            self.ledger.entries_skipped_zero_size += 1;
                        } else {
                            self.pending = Pending::Enter { shares, n };
                        }
                    }
                }
            }
        }
        if last_bar {
            self.pending = Pending::None;
        }
        self.channel_prev = channel_now;
        self.resting = self.plan_resting();
    }

    /// After the last bar: the open position, if any, marked at `last_close` as an
    /// `end_of_data` round trip. Cash and the position itself are left as they are.
    pub fn finish(&self, last_bar: usize, last_close: f64) -> Option<RoundTrip> {
        let p = self.pos.as_ref()?;
        let proceeds = p.shares as f64 * last_close;
        Some(RoundTrip {
            entry_bar: p.entry_bar,
            exit_bar: last_bar,
            exit_reason: ExitReason::EndOfData,
            entry_px: p.entry_px,
            exit_px: last_close,
            unit_shares: p.unit_shares,
            units: p.units,
            shares: p.shares,
            cost_basis: p.cost_basis,
            proceeds,
            pnl: proceeds - p.cost_basis,
            n_adds: p.n_adds,
            adds_skipped_budget: p.adds_skipped_budget,
            max_stop: p.stop,
        })
    }
}
