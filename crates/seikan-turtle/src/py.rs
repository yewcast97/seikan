//! The PyO3 bindings: the `seikan._turtle` extension module (feature `python`).
//!
//! Thin wrappers over the kernel types. Enumerations cross the boundary as their lowercase
//! string names (`"close"`/`"trade"`, `"entry"`/`"add"`/`"exit"`, the exit reasons), optional
//! numbers as `None`, and the indicators gain `handle_bar(bar)`, which reads a nautilus_trader
//! `Bar` through its `high`/`low`/`close` attributes (anything `float()` accepts), so a strategy
//! can register them as bar-driven indicators without this crate linking any nautilus code.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::coefficients::{self, Coefficients, NSource, Trigger};
use crate::indicators::{LowestLowChannel, WilderAtr};
use crate::machine::{Fill, FillKind, Ledger, Machine, PrintAction, RoundTrip, StopOrder};
use crate::price;
use crate::sim::{self, BarSeries, SimFill, SimResult};

fn value_error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

fn attr_f64(bar: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
    bar.getattr(name)?.extract::<f64>()
}

// ---- coefficients -------------------------------------------------------------------------

/// The rule coefficients for one target (validated at construction).
#[pyclass(
    name = "Coefficients",
    module = "seikan._turtle",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyCoefficients {
    inner: Coefficients,
}

#[pymethods]
impl PyCoefficients {
    #[new]
    #[pyo3(signature = (*, atr_period, add_step_n, max_units, stop_n, exit_lookback,
        risk_per_unit, stop_trigger, exit_trigger, stop_n_source, price_precision, budget))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        atr_period: usize,
        add_step_n: f64,
        max_units: u32,
        stop_n: f64,
        exit_lookback: usize,
        risk_per_unit: f64,
        stop_trigger: &str,
        exit_trigger: &str,
        stop_n_source: &str,
        price_precision: u32,
        budget: f64,
    ) -> PyResult<Self> {
        let inner = Coefficients {
            atr_period,
            add_step_n,
            max_units,
            stop_n,
            exit_lookback,
            risk_per_unit,
            stop_trigger: Trigger::parse(stop_trigger).map_err(value_error)?,
            exit_trigger: Trigger::parse(exit_trigger).map_err(value_error)?,
            stop_n_source: NSource::parse(stop_n_source).map_err(value_error)?,
            price_precision,
            budget,
        };
        inner.validate().map_err(value_error)?;
        Ok(Self { inner })
    }

    #[getter]
    fn atr_period(&self) -> usize {
        self.inner.atr_period
    }
    #[getter]
    fn add_step_n(&self) -> f64 {
        self.inner.add_step_n
    }
    #[getter]
    fn max_units(&self) -> u32 {
        self.inner.max_units
    }
    #[getter]
    fn stop_n(&self) -> f64 {
        self.inner.stop_n
    }
    #[getter]
    fn exit_lookback(&self) -> usize {
        self.inner.exit_lookback
    }
    #[getter]
    fn risk_per_unit(&self) -> f64 {
        self.inner.risk_per_unit
    }
    #[getter]
    fn stop_trigger(&self) -> &'static str {
        self.inner.stop_trigger.as_str()
    }
    #[getter]
    fn exit_trigger(&self) -> &'static str {
        self.inner.exit_trigger.as_str()
    }
    #[getter]
    fn stop_n_source(&self) -> &'static str {
        self.inner.stop_n_source.as_str()
    }
    #[getter]
    fn price_precision(&self) -> u32 {
        self.inner.price_precision
    }
    #[getter]
    fn budget(&self) -> f64 {
        self.inner.budget
    }

    /// The first bar index on which an entry can be taken.
    fn first_eligible_bar(&self) -> usize {
        self.inner.first_eligible_bar()
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
    }
}

// ---- indicators ---------------------------------------------------------------------------

/// Wilder's average true range — the Turtle N.
#[pyclass(name = "WilderAtr", module = "seikan._turtle")]
pub struct PyWilderAtr {
    inner: WilderAtr,
}

#[pymethods]
impl PyWilderAtr {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        if period == 0 {
            return Err(value_error("period must be >= 1".into()));
        }
        Ok(Self {
            inner: WilderAtr::new(period),
        })
    }

    #[getter]
    fn name(&self) -> String {
        format!("WilderAtr({})", self.inner.period())
    }
    #[getter]
    fn period(&self) -> usize {
        self.inner.period()
    }
    #[getter]
    fn count(&self) -> usize {
        self.inner.count()
    }
    #[getter]
    fn initialized(&self) -> bool {
        self.inner.initialized()
    }
    #[getter]
    fn has_inputs(&self) -> bool {
        self.inner.has_inputs()
    }
    /// The current N, or NaN before initialization.
    #[getter]
    fn value(&self) -> f64 {
        self.inner.value().unwrap_or(f64::NAN)
    }

    fn update_raw(&mut self, high: f64, low: f64, close: f64) {
        self.inner.update_raw(high, low, close);
    }

    /// Update from an object exposing `high`, `low` and `close` (a nautilus_trader `Bar`).
    fn handle_bar(&mut self, bar: &Bound<'_, PyAny>) -> PyResult<()> {
        let (high, low, close) = (
            attr_f64(bar, "high")?,
            attr_f64(bar, "low")?,
            attr_f64(bar, "close")?,
        );
        self.inner.update_raw(high, low, close);
        Ok(())
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!(
            "{}(value={}, count={})",
            self.name(),
            self.value(),
            self.count()
        )
    }
}

/// The lowest low of the last `lookback` bars handled (the newest included).
#[pyclass(name = "LowestLowChannel", module = "seikan._turtle")]
pub struct PyLowestLowChannel {
    inner: LowestLowChannel,
}

#[pymethods]
impl PyLowestLowChannel {
    #[new]
    fn new(lookback: usize) -> PyResult<Self> {
        if lookback == 0 {
            return Err(value_error("lookback must be >= 1".into()));
        }
        Ok(Self {
            inner: LowestLowChannel::new(lookback),
        })
    }

    #[getter]
    fn name(&self) -> String {
        format!("LowestLowChannel({})", self.inner.lookback())
    }
    #[getter]
    fn lookback(&self) -> usize {
        self.inner.lookback()
    }
    #[getter]
    fn count(&self) -> usize {
        self.inner.count()
    }
    #[getter]
    fn initialized(&self) -> bool {
        self.inner.initialized()
    }
    #[getter]
    fn has_inputs(&self) -> bool {
        self.inner.has_inputs()
    }
    /// The channel level, or NaN before initialization.
    #[getter]
    fn value(&self) -> f64 {
        self.inner.value().unwrap_or(f64::NAN)
    }

    fn update_raw(&mut self, low: f64) {
        self.inner.update_raw(low);
    }

    /// Update from an object exposing `low` (a nautilus_trader `Bar`).
    fn handle_bar(&mut self, bar: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.update_raw(attr_f64(bar, "low")?);
        Ok(())
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!(
            "{}(value={}, count={})",
            self.name(),
            self.value(),
            self.count()
        )
    }
}

// ---- data classes -------------------------------------------------------------------------

/// The one sell stop to keep resting at the venue.
#[pyclass(
    name = "StopOrder",
    module = "seikan._turtle",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyStopOrder {
    trigger: f64,
    shares: u64,
    reason: String,
}

impl From<StopOrder> for PyStopOrder {
    fn from(o: StopOrder) -> Self {
        Self {
            trigger: o.trigger,
            shares: o.shares,
            reason: o.reason.as_str().to_string(),
        }
    }
}

#[pymethods]
impl PyStopOrder {
    fn __repr__(&self) -> String {
        format!(
            "StopOrder(trigger={}, shares={}, reason={:?})",
            self.trigger, self.shares, self.reason
        )
    }
}

/// What to submit at an opening print: `kind` is `hold`, `buy` or `sell`; `intent` names the
/// fill kind of a buy (`entry`/`add`); `reason` names a sell's exit reason.
#[pyclass(
    name = "PrintAction",
    module = "seikan._turtle",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyPrintAction {
    kind: String,
    shares: u64,
    intent: Option<String>,
    reason: Option<String>,
}

impl From<PrintAction> for PyPrintAction {
    fn from(a: PrintAction) -> Self {
        match a {
            PrintAction::Hold => Self {
                kind: "hold".into(),
                shares: 0,
                intent: None,
                reason: None,
            },
            PrintAction::Buy { shares, kind } => Self {
                kind: "buy".into(),
                shares,
                intent: Some(kind.as_str().into()),
                reason: None,
            },
            PrintAction::Sell { shares, reason } => Self {
                kind: "sell".into(),
                shares,
                intent: Some(FillKind::Exit.as_str().into()),
                reason: Some(reason.as_str().into()),
            },
        }
    }
}

#[pymethods]
impl PyPrintAction {
    fn __repr__(&self) -> String {
        format!(
            "PrintAction(kind={:?}, shares={}, intent={:?}, reason={:?})",
            self.kind, self.shares, self.intent, self.reason
        )
    }
}

/// One reference-simulator fill: `at` is `open` or `trigger`.
#[pyclass(
    name = "Fill",
    module = "seikan._turtle",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFill {
    bar: usize,
    kind: String,
    shares: u64,
    price: f64,
    at: String,
    reason: Option<String>,
}

impl From<&SimFill> for PyFill {
    fn from(f: &SimFill) -> Self {
        Self {
            bar: f.fill.bar,
            kind: f.fill.kind.as_str().into(),
            shares: f.fill.shares,
            price: f.fill.price,
            at: f.at.as_str().into(),
            reason: f.reason.map(str::to_string),
        }
    }
}

#[pymethods]
impl PyFill {
    fn __repr__(&self) -> String {
        format!(
            "Fill(bar={}, kind={:?}, shares={}, price={}, at={:?}, reason={:?})",
            self.bar, self.kind, self.shares, self.price, self.at, self.reason
        )
    }
}

/// One position from entry to exit; `is_open` when marked at the last bar (`end_of_data`).
#[pyclass(
    name = "RoundTrip",
    module = "seikan._turtle",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyRoundTrip {
    entry_bar: usize,
    exit_bar: usize,
    exit_reason: String,
    is_open: bool,
    entry_px: f64,
    exit_px: f64,
    unit_shares: u64,
    units: u32,
    shares: u64,
    cost_basis: f64,
    proceeds: f64,
    pnl: f64,
    n_adds: u32,
    adds_skipped_budget: u32,
    max_stop: f64,
}

impl From<&RoundTrip> for PyRoundTrip {
    fn from(t: &RoundTrip) -> Self {
        Self {
            entry_bar: t.entry_bar,
            exit_bar: t.exit_bar,
            exit_reason: t.exit_reason.as_str().into(),
            is_open: t.is_open(),
            entry_px: t.entry_px,
            exit_px: t.exit_px,
            unit_shares: t.unit_shares,
            units: t.units,
            shares: t.shares,
            cost_basis: t.cost_basis,
            proceeds: t.proceeds,
            pnl: t.pnl,
            n_adds: t.n_adds,
            adds_skipped_budget: t.adds_skipped_budget,
            max_stop: t.max_stop,
        }
    }
}

#[pymethods]
impl PyRoundTrip {
    fn __repr__(&self) -> String {
        format!(
            "RoundTrip(entry_bar={}, exit_bar={}, exit_reason={:?}, units={}, shares={}, pnl={})",
            self.entry_bar, self.exit_bar, self.exit_reason, self.units, self.shares, self.pnl
        )
    }
}

/// Everything the machine did and declined to do, counted.
#[pyclass(
    name = "Ledger",
    module = "seikan._turtle",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyLedger {
    entries: u64,
    adds: u64,
    exits_stop_close: u64,
    exits_stop_gap: u64,
    exits_stop_trade: u64,
    exits_channel_close: u64,
    exits_channel_trade: u64,
    adds_skipped_budget: u64,
    adds_filled_below_level: u64,
    entries_skipped_warmup: u64,
    entries_skipped_in_position: u64,
    entries_skipped_zero_size: u64,
    entries_skipped_budget: u64,
    entries_skipped_end_of_data: u64,
    entries_cash_capped: u64,
    stop_holds: u64,
}

impl From<&Ledger> for PyLedger {
    fn from(l: &Ledger) -> Self {
        Self {
            entries: l.entries,
            adds: l.adds,
            exits_stop_close: l.exits_stop_close,
            exits_stop_gap: l.exits_stop_gap,
            exits_stop_trade: l.exits_stop_trade,
            exits_channel_close: l.exits_channel_close,
            exits_channel_trade: l.exits_channel_trade,
            adds_skipped_budget: l.adds_skipped_budget,
            adds_filled_below_level: l.adds_filled_below_level,
            entries_skipped_warmup: l.entries_skipped_warmup,
            entries_skipped_in_position: l.entries_skipped_in_position,
            entries_skipped_zero_size: l.entries_skipped_zero_size,
            entries_skipped_budget: l.entries_skipped_budget,
            entries_skipped_end_of_data: l.entries_skipped_end_of_data,
            entries_cash_capped: l.entries_cash_capped,
            stop_holds: l.stop_holds,
        }
    }
}

#[pymethods]
impl PyLedger {
    /// The counters as a plain dict.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (key, value) in [
            ("entries", self.entries),
            ("adds", self.adds),
            ("exits_stop_close", self.exits_stop_close),
            ("exits_stop_gap", self.exits_stop_gap),
            ("exits_stop_trade", self.exits_stop_trade),
            ("exits_channel_close", self.exits_channel_close),
            ("exits_channel_trade", self.exits_channel_trade),
            ("adds_skipped_budget", self.adds_skipped_budget),
            ("adds_filled_below_level", self.adds_filled_below_level),
            ("entries_skipped_warmup", self.entries_skipped_warmup),
            (
                "entries_skipped_in_position",
                self.entries_skipped_in_position,
            ),
            ("entries_skipped_zero_size", self.entries_skipped_zero_size),
            ("entries_skipped_budget", self.entries_skipped_budget),
            (
                "entries_skipped_end_of_data",
                self.entries_skipped_end_of_data,
            ),
            ("entries_cash_capped", self.entries_cash_capped),
            ("stop_holds", self.stop_holds),
        ] {
            d.set_item(key, value)?;
        }
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "Ledger(entries={}, adds={}, exits={})",
            self.entries,
            self.adds,
            self.exits_stop_close
                + self.exits_stop_gap
                + self.exits_stop_trade
                + self.exits_channel_close
                + self.exits_channel_trade
        )
    }
}

/// A reference-simulator run: fills, round trips, the ledger and per-bar samples (lists).
#[pyclass(
    name = "SimResult",
    module = "seikan._turtle",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PySimResult {
    fills: Vec<PyFill>,
    trips: Vec<PyRoundTrip>,
    open_trip: Option<PyRoundTrip>,
    ledger: PyLedger,
    equity: Vec<f64>,
    cash: Vec<f64>,
    shares: Vec<u64>,
    units: Vec<u32>,
    stop: Vec<f64>,
    add_level: Vec<f64>,
    atr: Vec<f64>,
    channel: Vec<f64>,
    first_eligible_bar: usize,
}

impl From<SimResult> for PySimResult {
    fn from(r: SimResult) -> Self {
        Self {
            fills: r.fills.iter().map(PyFill::from).collect(),
            trips: r.trips.iter().map(PyRoundTrip::from).collect(),
            open_trip: r.open_trip.as_ref().map(PyRoundTrip::from),
            ledger: PyLedger::from(&r.ledger),
            equity: r.equity,
            cash: r.cash,
            shares: r.shares,
            units: r.units,
            stop: r.stop,
            add_level: r.add_level,
            atr: r.atr,
            channel: r.channel,
            first_eligible_bar: r.first_eligible_bar,
        }
    }
}

// ---- the machine --------------------------------------------------------------------------

/// The long-only Turtle state machine for one target.
#[pyclass(name = "Machine", module = "seikan._turtle")]
pub struct PyMachine {
    inner: Machine,
}

#[pymethods]
impl PyMachine {
    #[new]
    fn new(coefficients: &PyCoefficients) -> PyResult<Self> {
        Ok(Self {
            inner: Machine::new(coefficients.inner.clone()).map_err(value_error)?,
        })
    }

    /// The one action to submit at the opening print of `bar`.
    fn on_print(&mut self, bar: usize, open: f64) -> PyPrintAction {
        self.inner.on_print(bar, open).into()
    }

    /// Report a fill (`kind` is `entry`, `add` or `exit`).
    fn on_fill(&mut self, bar: usize, kind: &str, shares: u64, price: f64) -> PyResult<()> {
        let kind = FillKind::parse(kind).map_err(value_error)?;
        self.inner
            .on_fill(Fill {
                bar,
                kind,
                shares,
                price,
            })
            .map_err(value_error)
    }

    /// The close of `bar`, with the indicator values after it.
    #[pyo3(signature = (bar, close, n=None, channel=None, fired=false, last_bar=false))]
    fn on_bar(
        &mut self,
        bar: usize,
        close: f64,
        n: Option<f64>,
        channel: Option<f64>,
        fired: bool,
        last_bar: bool,
    ) {
        self.inner.on_bar(bar, close, n, channel, fired, last_bar);
    }

    /// After the last bar: the open position marked at `last_close`, if any.
    fn finish(&self, last_bar: usize, last_close: f64) -> Option<PyRoundTrip> {
        self.inner
            .finish(last_bar, last_close)
            .as_ref()
            .map(PyRoundTrip::from)
    }

    /// Cash plus the position marked at `close`.
    fn equity(&self, close: f64) -> f64 {
        self.inner.equity(close)
    }

    /// The sell stop to keep resting at the venue from now on, if any.
    fn resting(&self) -> Option<PyStopOrder> {
        self.inner.resting().map(PyStopOrder::from)
    }

    #[getter]
    fn cash(&self) -> f64 {
        self.inner.cash()
    }
    #[getter]
    fn shares(&self) -> u64 {
        self.inner.shares()
    }
    #[getter]
    fn units(&self) -> u32 {
        self.inner.units()
    }
    #[getter]
    fn in_position(&self) -> bool {
        self.inner.in_position()
    }
    #[getter]
    fn stop(&self) -> Option<f64> {
        self.inner.stop()
    }
    #[getter]
    fn add_level(&self) -> Option<f64> {
        self.inner.add_level()
    }
    #[getter]
    fn last_fill(&self) -> Option<f64> {
        self.inner.position().map(|p| p.last_fill_px)
    }
    #[getter]
    fn entry_px(&self) -> Option<f64> {
        self.inner.position().map(|p| p.entry_px)
    }
    #[getter]
    fn n_entry(&self) -> Option<f64> {
        self.inner.position().map(|p| p.n_entry)
    }
    #[getter]
    fn channel(&self) -> Option<f64> {
        self.inner.channel()
    }
    #[getter]
    fn pending(&self) -> &'static str {
        self.inner.pending().as_str()
    }
    #[getter]
    fn first_eligible_bar(&self) -> usize {
        self.inner.first_eligible_bar()
    }
    #[getter]
    fn ledger(&self) -> PyLedger {
        PyLedger::from(self.inner.ledger())
    }
    #[getter]
    fn closed(&self) -> Vec<PyRoundTrip> {
        self.inner.closed().iter().map(PyRoundTrip::from).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Machine(cash={}, shares={}, units={}, pending={:?})",
            self.inner.cash(),
            self.inner.shares(),
            self.inner.units(),
            self.inner.pending().as_str()
        )
    }
}

// ---- functions ----------------------------------------------------------------------------

/// The kernel's version (the crate version).
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Shares per unit: `floor(risk_per_unit × budget / (stop_n × n))`.
#[pyfunction]
fn unit_shares(risk_per_unit: f64, budget: f64, stop_n: f64, n: f64) -> u64 {
    coefficients::unit_shares(risk_per_unit, budget, stop_n, n)
}

/// The highest grid price strictly below `level` at `precision` decimals.
#[pyfunction]
fn price_below(level: f64, precision: u32) -> f64 {
    price::price_below(level, precision)
}

/// Round `price` to `precision` decimals.
#[pyfunction]
fn quantize(price: f64, precision: u32) -> f64 {
    price::quantize(price, precision)
}

/// `max(atr_period, exit_lookback) - 1`.
#[pyfunction]
fn first_eligible_bar(atr_period: usize, exit_lookback: usize) -> usize {
    coefficients::first_eligible_bar(atr_period, exit_lookback)
}

/// Run the reference simulator over quantized OHLC sequences and the per-bar firing flags.
#[pyfunction]
fn simulate_reference(
    coefficients: &PyCoefficients,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    fired: Vec<bool>,
) -> PyResult<PySimResult> {
    let bars = BarSeries {
        open: &open,
        high: &high,
        low: &low,
        close: &close,
    };
    sim::simulate(&coefficients.inner, bars, &fired)
        .map(PySimResult::from)
        .map_err(value_error)
}

#[pymodule]
#[pyo3(name = "_turtle")]
fn seikan_turtle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(unit_shares, m)?)?;
    m.add_function(wrap_pyfunction!(price_below, m)?)?;
    m.add_function(wrap_pyfunction!(quantize, m)?)?;
    m.add_function(wrap_pyfunction!(first_eligible_bar, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_reference, m)?)?;
    m.add_class::<PyCoefficients>()?;
    m.add_class::<PyWilderAtr>()?;
    m.add_class::<PyLowestLowChannel>()?;
    m.add_class::<PyStopOrder>()?;
    m.add_class::<PyPrintAction>()?;
    m.add_class::<PyFill>()?;
    m.add_class::<PyRoundTrip>()?;
    m.add_class::<PyLedger>()?;
    m.add_class::<PySimResult>()?;
    m.add_class::<PyMachine>()?;
    Ok(())
}
