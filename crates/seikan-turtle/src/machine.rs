//! The long-only Turtle state machine for ONE target.
//!
//! Bar stamps are decision times. Every decision is taken on a bar's close and executed at the
//! NEXT bar's opening print — except the trade-triggered stop and channel exit, which rest at the
//! venue as ONE sell stop and fill wherever the next bar trades through them. The only order that
//! ever rests through a bar is that sell stop, so an entry, an add and an exit never conflict
//! inside one bar.
//!
//! The machine holds the RULES and the books, and nothing about prices beyond the open it sizes
//! against: the venue prices every fill (see [`crate::execution`]) and reports it back through
//! [`Machine::on_fill`], commission and cost attribution included, and a buy is sized against
//! the venue's own all-in cash-out function so the machine never commits cash the fill will
//! exceed.
//!
//! The rules, as implemented (see `CONTRACT.md` for the prose):
//!
//! - **Entry.** The thesis fires on bar `t` while flat and past warmup → market buy at the print
//!   of `t+1`. `N_entry` is the ATR at bar `t`; `X = floor(risk_per_unit × budget / (stop_n ×
//!   N_entry))`, capped at the print to the largest size whose all-in cash-out fits the target's
//!   cash. Initial stop `P0 − stop_n × N_entry`.
//! - **Adds** (no gap guard). While `units < max_units`, a close at or above
//!   `last_fill + add_step_n × N_entry` buys another X shares at the next print, wherever it opens
//!   — unless X shares do not fit the remaining cash, in which case the add is skipped whole and
//!   re-arms at the next close. After an add the stop is `max(stop, fill − stop_n × N_stop)`, `N_stop`
//!   being the current ATR or the entry ATR per `stop_n_source`; it never moves down.
//! - **Stop-loss / channel exit**, each `close` or `trade`. Close mode: `close < stop` →
//!   `stop_close`, else `close < channel` → `channel_close`, sold at the next print; an opening
//!   print below the stop with no exit pending sells at that open (`stop_gap`). Trade mode: one
//!   resting sell stop one grid step below the higher applicable level (`stop_trade` when the stop
//!   is the higher level, else `channel_trade`); a gapped print fills it at the open. The channel
//!   is the lowest low of the previous `exit_lookback` completed bars, the current bar excluded.
//! - Print precedence: pending exit → close-mode gap-through stop → pending entry/add (a pending
//!   add is dropped when the resting stop already exited at the print).

use crate::coefficients::{Coefficients, NSource, Trigger, unit_shares};
use crate::error::Error;
use crate::price::price_below;

/// What a fill was for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillKind {
    /// The first unit of a position.
    Entry,
    /// A further unit.
    Add,
    /// The whole position sold.
    Exit,
}

impl FillKind {
    /// Parse the lowercase name.
    pub fn parse(text: &str) -> Result<Self, Error> {
        match text {
            "entry" => Ok(Self::Entry),
            "add" => Ok(Self::Add),
            "exit" => Ok(Self::Exit),
            other => Err(Error::Input(format!(
                "fill kind must be 'entry', 'add' or 'exit', got {other:?}"
            ))),
        }
    }

    /// The lowercase name.
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
    /// A close below the stop (close mode), sold at the next print.
    StopClose,
    /// An opening print below the stop (close mode), sold at that open.
    StopGap,
    /// The resting stop at the stop level (trade mode).
    StopTrade,
    /// A close below the channel (close mode), sold at the next print.
    ChannelClose,
    /// The resting stop at the channel level (trade mode).
    ChannelTrade,
    /// Still open after the last bar: marked, never filled.
    EndOfData,
}

impl ExitReason {
    /// The snake_case name.
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
    /// The entry fill's bar.
    pub entry_bar: usize,
    /// The entry fill's price (P0).
    pub entry_px: f64,
    /// X — the entry fill's size, repeated by every add.
    pub unit_shares: u64,
    /// Units held.
    pub units: u32,
    /// Shares held.
    pub shares: u64,
    /// The latest fill's price (the add ladder's anchor).
    pub last_fill_px: f64,
    /// The ATR at the entry signal bar.
    pub n_entry: f64,
    /// The stop in force.
    pub stop: f64,
    /// Σ shares × fill price over the buys (gross).
    pub cost_basis: f64,
    /// Commission charged on the buys so far.
    pub commission: f64,
    /// Slippage attributed to the buys so far.
    pub slippage: f64,
    /// Stop shock attributed so far (zero until the exit).
    pub shock: f64,
    /// Market impact attributed to the buys so far.
    pub impact: f64,
    /// Adds filled.
    pub n_adds: u32,
    /// Adds skipped for want of cash.
    pub adds_skipped_budget: u32,
}

impl Position {
    /// The next add level: `add_step_n × N_entry` above the latest fill.
    pub fn add_level(&self, add_step_n: f64) -> f64 {
        self.last_fill_px + add_step_n * self.n_entry
    }

    /// The round trip this position makes when it leaves at `exit_px` on `exit_bar`: with the
    /// exit fill's own costs when sold, without any when merely marked.
    fn round_trip(
        &self,
        exit_bar: usize,
        exit_reason: ExitReason,
        exit_px: f64,
        exit: Option<&Fill>,
    ) -> RoundTrip {
        let proceeds = self.shares as f64 * exit_px;
        let (commission, slippage, shock, impact) = match exit {
            Some(f) => (f.commission, f.slippage, f.shock, f.impact),
            None => (0.0, 0.0, 0.0, 0.0),
        };
        let commission = self.commission + commission;
        let gross_pnl = proceeds - self.cost_basis;
        RoundTrip {
            entry_bar: self.entry_bar,
            exit_bar,
            exit_reason,
            entry_px: self.entry_px,
            exit_px,
            unit_shares: self.unit_shares,
            units: self.units,
            shares: self.shares,
            cost_basis: self.cost_basis,
            proceeds,
            commission,
            slippage: self.slippage + slippage,
            shock: self.shock + shock,
            impact: self.impact + impact,
            gross_pnl,
            pnl: gross_pnl - commission,
            n_adds: self.n_adds,
            adds_skipped_budget: self.adds_skipped_budget,
            max_stop: self.stop,
        }
    }
}

/// The decision taken at a close, waiting for the next opening print.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pending {
    /// Nothing to do at the next print.
    None,
    /// Buy the first unit.
    Enter {
        /// X shares.
        shares: u64,
        /// The entry ATR.
        n: f64,
    },
    /// Buy another unit.
    Add {
        /// The add level the close cleared.
        level: f64,
        /// The N the new stop is set from.
        n_for_stop: f64,
    },
    /// Sell the whole position.
    Exit(ExitReason),
}

impl Pending {
    /// The lowercase name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Enter { .. } => "enter",
            Self::Add { .. } => "add",
            Self::Exit(_) => "exit",
        }
    }
}

/// What the venue executes at an opening print.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrintAction {
    /// Nothing.
    Hold,
    /// A market buy, already sized to fit the cash.
    Buy {
        /// Shares to buy.
        shares: u64,
        /// Entry or add.
        kind: FillKind,
    },
    /// A market sell of the whole position.
    Sell {
        /// Shares to sell.
        shares: u64,
        /// Why.
        reason: ExitReason,
    },
}

/// The one sell stop to keep resting at the venue through the next bar (trade modes only).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StopOrder {
    /// One grid step below the level.
    pub trigger: f64,
    /// The whole position.
    pub shares: u64,
    /// Which level rests.
    pub reason: ExitReason,
}

/// One fill reported back to the machine, priced and attributed by the venue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fill {
    /// The bar the fill belongs to.
    pub bar: usize,
    /// What it was for.
    pub kind: FillKind,
    /// Shares filled.
    pub shares: u64,
    /// The fill price.
    pub price: f64,
    /// The commission charged.
    pub commission: f64,
    /// Slippage attributed (currency).
    pub slippage: f64,
    /// Stop shock attributed (currency).
    pub shock: f64,
    /// Market impact attributed (currency).
    pub impact: f64,
}

/// Everything the machine did and declined to do, counted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ledger {
    /// Entries filled.
    pub entries: u64,
    /// Adds filled.
    pub adds: u64,
    /// Exits by a close below the stop.
    pub exits_stop_close: u64,
    /// Exits by an opening print below the stop (close mode).
    pub exits_stop_gap: u64,
    /// Exits by the resting stop at the stop level.
    pub exits_stop_trade: u64,
    /// Exits by a close below the channel.
    pub exits_channel_close: u64,
    /// Exits by the resting stop at the channel level.
    pub exits_channel_trade: u64,
    /// Adds that no longer fit the cash.
    pub adds_skipped_budget: u64,
    /// Adds whose open was below the add level.
    pub adds_filled_below_level: u64,
    /// Firings before the first eligible bar.
    pub entries_skipped_warmup: u64,
    /// Firings while already long.
    pub entries_skipped_in_position: u64,
    /// Firings whose unit sized to zero shares.
    pub entries_skipped_zero_size: u64,
    /// Firings the cash could not afford at all.
    pub entries_skipped_budget: u64,
    /// Firings on the final bar (no next open).
    pub entries_skipped_end_of_data: u64,
    /// Entries whose size the cash reduced.
    pub entries_cash_capped: u64,
    /// Adds whose recomputed stop was lower than the stop in force (held).
    pub stop_holds: u64,
}

/// One position from entry to exit (or to the last bar, marked, when `exit_reason` is
/// `EndOfData`). Notionals are gross; `pnl` is net of commission, and every slippage, shock and
/// impact dollar is already inside the fill prices.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundTrip {
    /// The entry fill's bar.
    pub entry_bar: usize,
    /// The exit fill's bar (the last bar when marked).
    pub exit_bar: usize,
    /// Why it closed.
    pub exit_reason: ExitReason,
    /// The entry fill's price (P0).
    pub entry_px: f64,
    /// The exit fill's price (the last close when marked).
    pub exit_px: f64,
    /// X.
    pub unit_shares: u64,
    /// Units held at exit.
    pub units: u32,
    /// Shares held at exit.
    pub shares: u64,
    /// Σ shares × fill price over the buys.
    pub cost_basis: f64,
    /// shares × exit_px.
    pub proceeds: f64,
    /// Commission over every fill of the trip, the exit included.
    pub commission: f64,
    /// Slippage attributed over every fill.
    pub slippage: f64,
    /// Stop shock attributed (the exit fill).
    pub shock: f64,
    /// Market impact attributed over every fill.
    pub impact: f64,
    /// proceeds − cost_basis.
    pub gross_pnl: f64,
    /// gross_pnl − commission.
    pub pnl: f64,
    /// Adds filled.
    pub n_adds: u32,
    /// Adds skipped for want of cash.
    pub adds_skipped_budget: u32,
    /// The stop in force at exit (the highest the trip reached).
    pub max_stop: f64,
}

impl RoundTrip {
    /// Whether this is the end-of-data mark rather than a fill.
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
    /// A machine over validated coefficients, flat, holding the budget in cash.
    pub fn new(c: Coefficients) -> Result<Self, Error> {
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

    /// The coefficients.
    pub fn coefficients(&self) -> &Coefficients {
        &self.c
    }

    /// Cash on hand.
    pub fn cash(&self) -> f64 {
        self.cash
    }

    /// The open position, if any.
    pub fn position(&self) -> Option<&Position> {
        self.pos.as_ref()
    }

    /// Shares held.
    pub fn shares(&self) -> u64 {
        self.pos.as_ref().map_or(0, |p| p.shares)
    }

    /// Units held.
    pub fn units(&self) -> u32 {
        self.pos.as_ref().map_or(0, |p| p.units)
    }

    /// Whether a position is open.
    pub fn in_position(&self) -> bool {
        self.pos.is_some()
    }

    /// The stop in force, when long.
    pub fn stop(&self) -> Option<f64> {
        self.pos.as_ref().map(|p| p.stop)
    }

    /// The next add level, when long.
    pub fn add_level(&self) -> Option<f64> {
        self.pos.as_ref().map(|p| p.add_level(self.c.add_step_n))
    }

    /// The decision waiting for the next print.
    pub fn pending(&self) -> Pending {
        self.pending
    }

    /// The counters.
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// The closed round trips, in exit order.
    pub fn closed(&self) -> &[RoundTrip] {
        &self.closed
    }

    /// The first bar an entry can be taken on.
    pub fn first_eligible_bar(&self) -> usize {
        self.first_eligible
    }

    /// The channel level in force for the NEXT bar (the lowest low of the last `exit_lookback`
    /// completed bars), once known.
    pub fn channel(&self) -> Option<f64> {
        self.channel_prev
    }

    /// Cash plus the position marked at `close` — no liquidation cost.
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

    // ---- sizing -----------------------------------------------------------------------------

    /// Whether a buy whose all-in cash-out is `cash_out` fits the cash: within a billionth of a
    /// share priced at the open — the same guard the sizing division carries, so a frictionless
    /// fit is exactly `q <= floor(cash / open + 1e-9)`.
    fn fits(&self, cash_out: f64, open: f64) -> bool {
        cash_out <= self.cash + 1e-9 * open
    }

    /// The largest `q <= wanted` whose all-in cash-out fits. `cash_out` is strictly increasing
    /// in `q` (the notional strictly, the commission at least weakly), so a binary search from
    /// the frictionless bound finds it.
    fn affordable(&self, wanted: u64, open: f64, cash_out: &dyn Fn(u64) -> f64) -> u64 {
        let bound = ((self.cash / open) + 1e-9).floor().max(0.0);
        let mut hi = if bound >= wanted as f64 {
            wanted
        } else {
            bound as u64
        };
        let mut lo = 0u64;
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if self.fits(cash_out(mid), open) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }

    // ---- transitions ------------------------------------------------------------------------

    /// The opening print of `bar`: the one action to execute at this open. `cash_out(q)` is the
    /// venue's all-in cost of buying `q` shares at this print (notional at the fill price plus
    /// commission), the function every buy is sized against. Every fill that results is
    /// reported through [`Machine::on_fill`].
    pub fn on_print(
        &mut self,
        bar: usize,
        open: f64,
        cash_out: &dyn Fn(u64) -> f64,
    ) -> PrintAction {
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
                let filled = self.affordable(shares, open, cash_out);
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
                let (unit, units) = (p.unit_shares, p.units);
                if units >= self.c.max_units {
                    PrintAction::Hold
                } else if !self.fits(cash_out(unit), open) {
                    self.ledger.adds_skipped_budget += 1;
                    let p = self.pos.as_mut().expect("position present");
                    p.adds_skipped_budget += 1;
                    PrintAction::Hold
                } else {
                    self.armed = Some(Armed::Add { level, n_for_stop });
                    PrintAction::Buy {
                        shares: unit,
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
    pub fn on_fill(&mut self, fill: Fill) -> Result<(), Error> {
        match fill.kind {
            FillKind::Entry => {
                let Some(Armed::Entry { n }) = self.armed.take() else {
                    return Err(Error::Bookkeeping(
                        "entry fill reported without an armed entry".into(),
                    ));
                };
                if self.pos.is_some() {
                    return Err(Error::Bookkeeping(
                        "entry fill reported while a position is open".into(),
                    ));
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
                    commission: fill.commission,
                    slippage: fill.slippage,
                    shock: fill.shock,
                    impact: fill.impact,
                    n_adds: 0,
                    adds_skipped_budget: 0,
                });
                self.cash -= cost + fill.commission;
                self.ledger.entries += 1;
            }
            FillKind::Add => {
                let Some(Armed::Add { level, n_for_stop }) = self.armed.take() else {
                    return Err(Error::Bookkeeping(
                        "add fill reported without an armed add".into(),
                    ));
                };
                let Some(p) = self.pos.as_mut() else {
                    return Err(Error::Bookkeeping("add fill reported while flat".into()));
                };
                if fill.shares != p.unit_shares {
                    return Err(Error::Bookkeeping(format!(
                        "add fill of {} shares, expected one unit of {}",
                        fill.shares, p.unit_shares
                    )));
                }
                let cost = fill.shares as f64 * fill.price;
                p.units += 1;
                p.shares += fill.shares;
                p.last_fill_px = fill.price;
                p.cost_basis += cost;
                p.commission += fill.commission;
                p.slippage += fill.slippage;
                p.shock += fill.shock;
                p.impact += fill.impact;
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
                self.cash -= cost + fill.commission;
                self.ledger.adds += 1;
            }
            FillKind::Exit => {
                let reason = match self.armed.take() {
                    Some(Armed::Exit(reason)) => reason,
                    Some(other) => {
                        return Err(Error::Bookkeeping(format!(
                            "exit fill reported while armed for {other:?}"
                        )));
                    }
                    None => match self.resting {
                        Some(order) => order.reason,
                        None => {
                            return Err(Error::Bookkeeping(
                                "exit fill reported with no exit armed or resting".into(),
                            ));
                        }
                    },
                };
                let Some(p) = self.pos.take() else {
                    return Err(Error::Bookkeeping("exit fill reported while flat".into()));
                };
                if fill.shares != p.shares {
                    return Err(Error::Bookkeeping(format!(
                        "exit fill of {} shares, expected the whole position of {}",
                        fill.shares, p.shares
                    )));
                }
                let trip = p.round_trip(fill.bar, reason, fill.price, Some(&fill));
                self.cash += trip.proceeds - fill.commission;
                self.closed.push(trip);
                match reason {
                    ExitReason::StopClose => self.ledger.exits_stop_close += 1,
                    ExitReason::StopGap => self.ledger.exits_stop_gap += 1,
                    ExitReason::StopTrade => self.ledger.exits_stop_trade += 1,
                    ExitReason::ChannelClose => self.ledger.exits_channel_close += 1,
                    ExitReason::ChannelTrade => self.ledger.exits_channel_trade += 1,
                    ExitReason::EndOfData => {
                        return Err(Error::Bookkeeping(
                            "end_of_data is a mark, never a fill".into(),
                        ));
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
    /// `end_of_data` round trip (no exit cost). Cash and the position itself are left as they are.
    pub fn finish(&self, last_bar: usize, last_close: f64) -> Option<RoundTrip> {
        self.pos
            .as_ref()
            .map(|p| p.round_trip(last_bar, ExitReason::EndOfData, last_close, None))
    }
}
