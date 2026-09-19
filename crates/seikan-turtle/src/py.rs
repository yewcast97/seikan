//! The PyO3 bindings: the `seikan._turtle` extension module (feature `python`).
//!
//! Thin wrappers over the engine: the rule coefficients and the cost model are built with
//! keyword arguments (enumerations as their lowercase names), `simulate` runs one target and
//! returns plain data classes whose optional numbers cross as `None`. An `Error::Bookkeeping`
//! maps to `RuntimeError` (a seikan bug), every other error to `ValueError`.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::coefficients::{Coefficients, NSource, Trigger};
use crate::costs::{Commission, CostModel, Impact, Slippage};
use crate::error::Error;
use crate::machine::{Ledger, RoundTrip};
use crate::price;
use crate::sim::{self, BarSeries, SimFill, SimResult};

impl From<Error> for PyErr {
    fn from(error: Error) -> Self {
        match error {
            Error::Bookkeeping(_) => PyRuntimeError::new_err(error.to_string()),
            Error::Coefficients(_) | Error::Input(_) => PyValueError::new_err(error.to_string()),
        }
    }
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
            stop_trigger: Trigger::parse(stop_trigger)?,
            exit_trigger: Trigger::parse(exit_trigger)?,
            stop_n_source: NSource::parse(stop_n_source)?,
            price_precision,
            budget,
        };
        inner.validate()?;
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

// ---- the cost model -----------------------------------------------------------------------

/// The trade-cost model (validated at construction).
#[pyclass(
    name = "CostModel",
    module = "seikan._turtle",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyCostModel {
    inner: CostModel,
}

#[pymethods]
impl PyCostModel {
    #[new]
    #[pyo3(signature = (*, per_share, min_per_order, bps, sell_bps, cap_bps, slippage_bps,
        slippage_n_fraction, impact_coefficient, adv_window, stop_shock))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        per_share: f64,
        min_per_order: f64,
        bps: f64,
        sell_bps: f64,
        cap_bps: Option<f64>,
        slippage_bps: f64,
        slippage_n_fraction: f64,
        impact_coefficient: f64,
        adv_window: usize,
        stop_shock: f64,
    ) -> PyResult<Self> {
        let inner = CostModel {
            commission: Commission {
                per_share,
                min_per_order,
                bps,
                sell_bps,
                cap_bps,
            },
            slippage: Slippage {
                bps: slippage_bps,
                n_fraction: slippage_n_fraction,
            },
            impact: Impact {
                coefficient: impact_coefficient,
                adv_window,
            },
            stop_shock,
        };
        inner.validate()?;
        Ok(Self { inner })
    }

    #[getter]
    fn per_share(&self) -> f64 {
        self.inner.commission.per_share
    }
    #[getter]
    fn min_per_order(&self) -> f64 {
        self.inner.commission.min_per_order
    }
    #[getter]
    fn bps(&self) -> f64 {
        self.inner.commission.bps
    }
    #[getter]
    fn sell_bps(&self) -> f64 {
        self.inner.commission.sell_bps
    }
    #[getter]
    fn cap_bps(&self) -> Option<f64> {
        self.inner.commission.cap_bps
    }
    #[getter]
    fn slippage_bps(&self) -> f64 {
        self.inner.slippage.bps
    }
    #[getter]
    fn slippage_n_fraction(&self) -> f64 {
        self.inner.slippage.n_fraction
    }
    #[getter]
    fn impact_coefficient(&self) -> f64 {
        self.inner.impact.coefficient
    }
    #[getter]
    fn adv_window(&self) -> usize {
        self.inner.impact.adv_window
    }
    #[getter]
    fn stop_shock(&self) -> f64 {
        self.inner.stop_shock
    }

    /// Whether market impact is modelled (a volume series is then required).
    fn impact_enabled(&self) -> bool {
        self.inner.impact.enabled()
    }

    fn __repr__(&self) -> String {
        format!("CostModel({:?})", self.inner)
    }
}

// ---- data classes -------------------------------------------------------------------------

/// One venue fill with the machine's state after it: `kind` is `entry`/`add`/`exit`, `at` is
/// `open` or `trigger`, `reason` names an exit's reason.
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
    reference: f64,
    commission: f64,
    slippage: f64,
    shock: f64,
    impact: f64,
    at: String,
    reason: Option<String>,
    cash_after: f64,
    shares_after: u64,
    units_after: u32,
    stop_after: Option<f64>,
}

impl From<&SimFill> for PyFill {
    fn from(f: &SimFill) -> Self {
        Self {
            bar: f.fill.bar,
            kind: f.fill.kind.as_str().into(),
            shares: f.fill.shares,
            price: f.fill.price,
            reference: f.reference,
            commission: f.fill.commission,
            slippage: f.fill.slippage,
            shock: f.fill.shock,
            impact: f.fill.impact,
            at: f.at.as_str().into(),
            reason: f.reason.map(str::to_string),
            cash_after: f.cash_after,
            shares_after: f.shares_after,
            units_after: f.units_after,
            stop_after: f.stop_after,
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
    commission: f64,
    slippage: f64,
    shock: f64,
    impact: f64,
    gross_pnl: f64,
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
            commission: t.commission,
            slippage: t.slippage,
            shock: t.shock,
            impact: t.impact,
            gross_pnl: t.gross_pnl,
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

/// One target's run: fills, round trips, the ledger and per-bar samples (lists).
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
    commission_cum: Vec<f64>,
    slippage_cum: Vec<f64>,
    shock_cum: Vec<f64>,
    impact_cum: Vec<f64>,
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
            commission_cum: r.commission_cum,
            slippage_cum: r.slippage_cum,
            shock_cum: r.shock_cum,
            impact_cum: r.impact_cum,
            first_eligible_bar: r.first_eligible_bar,
        }
    }
}

// ---- functions ----------------------------------------------------------------------------

/// The engine's version (the crate version).
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Round `price` to `precision` decimals.
#[pyfunction]
fn quantize(price: f64, precision: u32) -> f64 {
    price::quantize(price, precision)
}

/// Run one target over quantized OHLC sequences (and the volume, needed only when impact is
/// enabled) with the per-bar firing flags.
#[pyfunction]
#[pyo3(signature = (coefficients, costs, opens, highs, lows, closes, volumes, fired))]
#[allow(clippy::too_many_arguments)]
fn simulate(
    coefficients: &PyCoefficients,
    costs: &PyCostModel,
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Option<Vec<f64>>,
    fired: Vec<bool>,
) -> PyResult<PySimResult> {
    let bars = BarSeries {
        open: &opens,
        high: &highs,
        low: &lows,
        close: &closes,
        volume: volumes.as_deref(),
    };
    Ok(sim::simulate(&coefficients.inner, &costs.inner, bars, &fired)?.into())
}

#[pymodule]
#[pyo3(name = "_turtle")]
fn seikan_turtle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(quantize, m)?)?;
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    m.add_class::<PyCoefficients>()?;
    m.add_class::<PyCostModel>()?;
    m.add_class::<PyFill>()?;
    m.add_class::<PyRoundTrip>()?;
    m.add_class::<PyLedger>()?;
    m.add_class::<PySimResult>()?;
    Ok(())
}
