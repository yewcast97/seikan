"""The venue executes exactly what the rules say: nautilus_trader's fills equal the reference
simulator's, fill for fill, under every trigger mode and every execution edge the rules have —
and a run is deterministic."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from seikan import _turtle
from seikan.dsl.schema import Thesis
from seikan.serialize import json_safe
from seikan.turtle import TurtleCoefficients
from seikan.turtle.engine import kernel_coefficients, run_turtle
from tests._helpers import load
from tests._turtle_helpers import (
    Row,
    coefficients_doc,
    first_true_above,
    flat_rows,
    thesis_doc,
    worked_example_rows,
    write_bars,
)


def _fill_rows(fills) -> list[tuple]:
    return [(f.bar, f.kind, f.shares, f.price, f.at, f.reason) for f in fills]


def _run(tmp_path, rows_by_target: dict[str, list[Row]], entry: dict, coef: dict, **kw):
    paths = {
        t: str(write_bars(tmp_path / f"{t}.csv", rows, **kw)) for t, rows in rows_by_target.items()
    }
    first = next(iter(rows_by_target.values()))
    bench = write_bars(tmp_path / "bench.csv", flat_rows(len(first), 100.0, 1.0), **kw)
    thesis = Thesis.model_validate(thesis_doc(list(rows_by_target), entry))
    md = load(thesis, paths)
    c = TurtleCoefficients.model_validate(coef)
    return run_turtle(thesis, md, c, str(bench)), c


def _assert_parity(result, c: TurtleCoefficients) -> None:
    """Every target of every cell: venue fills == reference fills, and the sampled state agrees."""
    budget = c.equity / len(result.data.targets)
    kernel = kernel_coefficients(c, budget)
    for cell in result.cells:
        for target, run in cell.targets.items():
            bars = result.data.targets[target]
            ref = _turtle.simulate_reference(
                kernel,
                bars.open.tolist(),
                bars.high.tolist(),
                bars.low.tolist(),
                bars.close.tolist(),
                cell.cell.fired[target].tolist(),
            )
            assert _fill_rows(run.fills) == _fill_rows(ref.fills), (cell.cell.cell_id, target)
            assert run.ledger == ref.ledger.to_dict(), target
            assert np.allclose(run.samples.equity, ref.equity, rtol=0, atol=1e-9 * c.equity)
            assert run.samples.shares == ref.shares and run.samples.units == ref.units
            assert np.allclose(run.samples.stop, ref.stop, equal_nan=True)
            assert np.allclose(run.samples.add_level, ref.add_level, equal_nan=True)
            assert np.allclose(run.samples.atr, ref.atr, equal_nan=True)
            assert np.allclose(run.samples.channel, ref.channel, equal_nan=True)
            assert [(t.entry_bar, t.exit_bar, t.exit_reason, t.pnl) for t in run.trips] == [
                (t.entry_bar, t.exit_bar, t.exit_reason, t.pnl) for t in ref.trips
            ]
            assert (run.open_trip is None) == (ref.open_trip is None)
            if run.open_trip is not None and ref.open_trip is not None:
                assert run.open_trip.pnl == ref.open_trip.pnl
        assert cell.reconciliation.matched


MODES = [
    ("close", "close"),
    ("trade", "close"),
    ("close", "trade"),
    ("trade", "trade"),
]


@pytest.mark.parametrize(("stop_trigger", "exit_trigger"), MODES)
@pytest.mark.parametrize("stop_n_source", ["current", "entry"])
def test_worked_example_under_every_mode(tmp_path, stop_trigger, exit_trigger, stop_n_source):
    result, c = _run(
        tmp_path,
        {"PX": worked_example_rows()},
        first_true_above(50.0),
        coefficients_doc(
            stop_trigger=stop_trigger, exit_trigger=exit_trigger, stop_n_source=stop_n_source
        ),
    )
    _assert_parity(result, c)
    run = result.cells[0].targets["PX"]
    kinds = [f.kind for f in run.fills]
    assert kinds == ["entry", "add", "add", "exit"]
    assert [f.price for f in run.fills[:3]] == [50.0, 51.0, 52.0]
    assert [f.stop_after for f in run.fills[:3]] == [46.0, 47.0, 48.0]
    exit_fill = run.fills[3]
    if stop_trigger == "trade" and exit_trigger == "close":
        assert (exit_fill.bar, exit_fill.price, exit_fill.reason) == (28, 47.9999, "stop_trade")
        assert exit_fill.at == "trigger"
    elif exit_trigger == "trade":
        assert (exit_fill.bar, exit_fill.price, exit_fill.reason) == (28, 48.4999, "channel_trade")
    else:
        assert (exit_fill.bar, exit_fill.price, exit_fill.reason) == (29, 47.0, "stop_close")
        assert exit_fill.at == "open"


@pytest.mark.parametrize("stop_trigger", ["close", "trade"])
def test_gap_through_the_stop_fills_at_the_open(tmp_path, stop_trigger):
    rows = [
        *flat_rows(24, 49.5, 1.0),
        (49.5, 51.0, 49.0, 50.0),
        (50.0, 51.0, 49.5, 50.0),
        (50.0, 50.5, 49.0, 49.0),
        (45.0, 45.5, 44.0, 44.5),
        *flat_rows(2, 44.5, 0.5),
    ]
    result, c = _run(
        tmp_path, {"PX": rows}, first_true_above(50.0), coefficients_doc(stop_trigger=stop_trigger)
    )
    _assert_parity(result, c)
    exit_fill = result.cells[0].targets["PX"].fills[-1]
    assert (exit_fill.bar, exit_fill.price, exit_fill.at) == (27, 45.0, "open")
    assert exit_fill.reason == ("stop_gap" if stop_trigger == "close" else "stop_trade")


def test_adds_fill_wherever_the_open_is(tmp_path):
    rows = [
        *flat_rows(24, 49.5, 1.0),
        (49.5, 51.0, 49.0, 50.0),
        (50.0, 52.0, 50.0, 51.0),
        (56.0, 57.0, 55.0, 56.0),  # far above the add level: still adds at 56
        (55.9, 57.5, 55.5, 57.0),  # closes at the next level (56 + 0.5 × N_entry 2): add pending
        (55.0, 56.0, 54.5, 55.5),  # opens BELOW the level: still adds at 55
        *flat_rows(3, 55.5, 0.5),
    ]
    result, c = _run(tmp_path, {"PX": rows}, first_true_above(50.0), coefficients_doc())
    _assert_parity(result, c)
    run = result.cells[0].targets["PX"]
    assert [(f.kind, f.price) for f in run.fills] == [("entry", 50.0), ("add", 56.0), ("add", 55.0)]
    assert run.ledger["adds_filled_below_level"] == 1


def test_a_stop_raised_at_the_print_is_live_during_the_same_bar(tmp_path):
    # Trade-mode stop: the add fills at the print, the stop moves up to fill − 2N, and the SAME
    # bar's low trades through the raised stop (but not the old one).
    rows = [
        *flat_rows(24, 49.5, 1.0),
        (49.5, 51.0, 49.0, 50.0),
        (50.0, 52.0, 50.0, 51.0),  # entry 50, stop 46
        (51.0, 51.5, 46.5, 47.0),  # add at 51 → stop 47; low 46.5 trades through 46.9999
        *flat_rows(3, 47.0, 0.5),
    ]
    result, c = _run(
        tmp_path, {"PX": rows}, first_true_above(50.0), coefficients_doc(stop_trigger="trade")
    )
    _assert_parity(result, c)
    run = result.cells[0].targets["PX"]
    assert [(f.bar, f.kind, f.price, f.at) for f in run.fills] == [
        (25, "entry", 50.0, "open"),
        (26, "add", 51.0, "open"),
        (26, "exit", 46.9999, "trigger"),
    ]


def test_budget_caps_the_entry_and_skips_an_unaffordable_add(tmp_path):
    rows = [
        *flat_rows(24, 49.9, 0.05),  # N = 0.1
        (49.9, 50.0, 49.9, 50.0),  # fires; TR 0.1 keeps N = 0.1 → risk X = 50, but 20 fit $1,000
        (50.0, 50.15, 50.0, 50.1),  # entry 20 @ 50 (cash 0); close 50.1 ≥ 50.05: add pending
        (50.1, 50.15, 50.05, 50.1),  # 20 × 50.1 > 0 cash: skipped, re-armed at the close
        *flat_rows(2, 50.1, 0.02),
    ]
    result, c = _run(tmp_path, {"PX": rows}, first_true_above(50.0), coefficients_doc(equity=1000))
    _assert_parity(result, c)
    run = result.cells[0].targets["PX"]
    assert [(f.kind, f.shares, f.price) for f in run.fills] == [("entry", 20, 50.0)]
    assert run.ledger["entries_cash_capped"] == 1 and run.ledger["adds_skipped_budget"] == 3


def test_two_targets_share_the_venue_and_keep_their_own_budgets(tmp_path):
    a = worked_example_rows()
    b = [(o * 2, h * 2, lo * 2, c * 2) for o, h, lo, c in worked_example_rows()]
    b = [*b[5:], *flat_rows(5, b[-1][3], 2.0)]  # shifted so B fires later than A
    result, c = _run(tmp_path, {"A": a, "B": b}, first_true_above(50.0), coefficients_doc())
    _assert_parity(result, c)
    cell = result.cells[0]
    assert set(cell.targets) == {"A", "B"}
    assert cell.targets["A"].instrument_id == "T0.SIM"
    assert cell.targets["B"].instrument_id == "T1.SIM"
    assert [f.kind for f in cell.targets["A"].fills] == ["entry", "add", "add", "exit"]
    assert result.budget_per_target == 50_000.0
    assert cell.targets["A"].fills[0].shares == 125  # X = 1% × 50,000 / (2 × 2)


def test_an_open_position_at_the_end_is_marked(tmp_path):
    rows = [*flat_rows(24, 49.5, 1.0), (49.5, 51.0, 49.0, 50.0), (50.0, 51.0, 49.5, 50.5)]
    result, c = _run(tmp_path, {"PX": rows}, first_true_above(50.0), coefficients_doc())
    _assert_parity(result, c)
    run = result.cells[0].targets["PX"]
    assert run.open_trip is not None and run.open_trip.exit_reason == "end_of_data"
    assert run.end_shares == 250 and run.trips == []
    assert run.samples.equity[-1] == pytest.approx(100_000 + 250 * 0.5)


def test_swept_cells_each_get_their_own_venue(tmp_path):
    doc_entry = {
        "type": "first_true",
        "condition": {
            "type": "threshold",
            "left": {"type": "field", "column": "close"},
            "op": ">=",
            "right": {"type": "constant", "value": [50.0, 52.5], "name": "level"},
        },
    }
    result, c = _run(tmp_path, {"PX": worked_example_rows()}, doc_entry, coefficients_doc())
    _assert_parity(result, c)
    assert result.axes == ["level"]
    assert [cell.cell.cell_id for cell in result.cells] == ["entry[level=50]", "entry[level=52.5]"]
    first, second = result.cells
    assert [f.kind for f in first.targets["PX"].fills] == ["entry", "add", "add", "exit"]
    # The second cell enters at bar 28's open (52.5) and is stopped out the next open.
    assert [(f.bar, f.kind) for f in second.targets["PX"].fills] == [(28, "entry"), (29, "exit")]


def test_an_hourly_clock_runs_the_same_rules(tmp_path):
    result, c = _run(
        tmp_path,
        {"PX": worked_example_rows()},
        first_true_above(50.0),
        coefficients_doc(bars_per_year=252 * 7),
        freq="1h",
    )
    _assert_parity(result, c)
    assert [f.price for f in result.cells[0].targets["PX"].fills] == [50.0, 51.0, 52.0, 47.0]


def test_runs_are_deterministic(tmp_path):
    (tmp_path / "1").mkdir()
    (tmp_path / "2").mkdir()
    a, _c = _run(
        tmp_path / "1",
        {"PX": worked_example_rows()},
        first_true_above(50.0),
        coefficients_doc(stop_trigger="trade", exit_trigger="trade"),
    )
    b, _ = _run(
        tmp_path / "2",
        {"PX": worked_example_rows()},
        first_true_above(50.0),
        coefficients_doc(stop_trigger="trade", exit_trigger="trade"),
    )
    ra, rb = a.cells[0].targets["PX"], b.cells[0].targets["PX"]
    assert ra.fills == rb.fills and ra.ledger == rb.ledger
    assert json_safe(asdict(ra.samples)) == json_safe(asdict(rb.samples))  # NaN → null
    assert json_safe(a.cells[0].engine_stats) == json_safe(b.cells[0].engine_stats)
