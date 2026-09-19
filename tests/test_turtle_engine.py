"""The engine executes exactly what the rules say — the worked example under every trigger mode
and every execution edge the rules have, frictionless (bit for bit the rules' own numbers) —
and prices every fill under the cost model in closed form; a run is deterministic."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from seikan.dataio import DataError
from seikan.dsl.schema import Thesis
from seikan.serialize import json_safe
from seikan.turtle import TurtleCoefficients
from seikan.turtle.engine import run_turtle
from tests._helpers import load
from tests._turtle_helpers import (
    Row,
    coefficients_doc,
    first_true_above,
    flat_rows,
    frictionless,
    thesis_doc,
    worked_example_rows,
    write_bars,
)


def _fill_rows(fills) -> list[tuple]:
    return [(f.bar, f.kind, f.shares, f.price, f.at, f.reason) for f in fills]


def _run(tmp_path, rows_by_target: dict[str, list[Row]], entry: dict, coef: dict, **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {
        t: str(write_bars(tmp_path / f"{t}.csv", rows, **kw)) for t, rows in rows_by_target.items()
    }
    first = next(iter(rows_by_target.values()))
    bench = write_bars(tmp_path / "bench.csv", flat_rows(len(first), 100.0, 1.0), **kw)
    thesis = Thesis.model_validate(thesis_doc(list(rows_by_target), entry))
    md = load(thesis, paths)
    c = TurtleCoefficients.model_validate(coef)
    return run_turtle(thesis, md, c, str(bench)), c


def _assert_books(result, c: TurtleCoefficients) -> None:
    """Every target of every cell: the books reconcile with the fills, the samples and the
    ledger, and the cumulative cost buckets end at what was paid."""
    budget = c.equity / len(result.data.targets)
    for cell in result.cells:
        for target, run in cell.targets.items():
            bars = result.data.targets[target]
            s = run.samples
            equity = np.asarray(s.equity)
            assert np.allclose(equity, np.asarray(s.cash) + np.asarray(s.shares) * bars.close)
            cash = budget
            for f in run.fills:
                notional = f.shares * f.price
                cash += notional - f.commission if f.kind == "exit" else -(notional + f.commission)
                assert f.cash_after == pytest.approx(cash)
                assert f.commission >= 0 and f.slippage >= 0 and f.shock >= 0 and f.impact >= 0
            assert run.end_cash == pytest.approx(cash)
            n_open = 0 if run.open_trip is None else 1
            assert run.ledger["entries"] == len(run.trips) + n_open
            exits = sum(v for k, v in run.ledger.items() if k.startswith("exits_"))
            assert exits == len(run.trips)
            for bucket in ("commission", "slippage", "shock", "impact"):
                cum = np.asarray(getattr(s, f"{bucket}_cum"))
                assert np.all(np.diff(cum) >= 0) and cum[-1] == pytest.approx(
                    getattr(run.costs_paid, bucket)
                )
                assert getattr(run.costs_paid, bucket) == pytest.approx(
                    sum(getattr(f, bucket) for f in run.fills)
                )
            for t in run.trips:
                assert t.gross_pnl == pytest.approx(t.proceeds - t.cost_basis)
                assert t.pnl == pytest.approx(t.gross_pnl - t.commission)


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
    _assert_books(result, c)
    run = result.cells[0].targets["PX"]
    assert run.costs_paid.total == 0.0
    kinds = [f.kind for f in run.fills]
    assert kinds == ["entry", "add", "add", "exit"]
    assert [f.price for f in run.fills[:3]] == [50.0, 51.0, 52.0]
    assert [f.reference for f in run.fills[:3]] == [50.0, 51.0, 52.0]
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
    _assert_books(result, c)
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
    _assert_books(result, c)
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
    _assert_books(result, c)
    run = result.cells[0].targets["PX"]
    assert [(f.bar, f.kind, f.price, f.at) for f in run.fills] == [
        (25, "entry", 50.0, "open"),
        (26, "add", 51.0, "open"),
        (26, "exit", 46.9999, "trigger"),
    ]


def test_a_pyramided_trade_stop_never_starves_the_next_add(tmp_path):
    # X = 250 with a budget that leaves 13,150 of cash after two fills, against a third fill
    # costing 13,000: nothing is ever locked for the resting stop.
    result, c = _run(
        tmp_path,
        {"PX": worked_example_rows()},
        first_true_above(50.0),
        coefficients_doc(equity=38_400, risk_per_unit=0.0260417, stop_trigger="trade"),
    )
    _assert_books(result, c)
    run = result.cells[0].targets["PX"]
    assert [(f.kind, f.shares, f.price) for f in run.fills[:3]] == [
        ("entry", 250, 50.0),
        ("add", 250, 51.0),
        ("add", 250, 52.0),
    ]
    assert run.ledger["adds_skipped_budget"] == 0


def test_budget_caps_the_entry_and_skips_an_unaffordable_add(tmp_path):
    rows = [
        *flat_rows(24, 49.9, 0.05),  # N = 0.1
        (49.9, 50.0, 49.9, 50.0),  # fires; TR 0.1 keeps N = 0.1 → risk X = 50, but 20 fit $1,000
        (50.0, 50.15, 50.0, 50.1),  # entry 20 @ 50 (cash 0); close 50.1 ≥ 50.05: add pending
        (50.1, 50.15, 50.05, 50.1),  # 20 × 50.1 > 0 cash: skipped, re-armed at the close
        *flat_rows(2, 50.1, 0.02),
    ]
    result, c = _run(tmp_path, {"PX": rows}, first_true_above(50.0), coefficients_doc(equity=1000))
    _assert_books(result, c)
    run = result.cells[0].targets["PX"]
    assert [(f.kind, f.shares, f.price) for f in run.fills] == [("entry", 20, 50.0)]
    assert run.ledger["entries_cash_capped"] == 1 and run.ledger["adds_skipped_budget"] == 3


def test_two_targets_keep_their_own_budgets(tmp_path):
    a = worked_example_rows()
    b = [(o * 2, h * 2, lo * 2, c * 2) for o, h, lo, c in worked_example_rows()]
    b = [*b[5:], *flat_rows(5, b[-1][3], 2.0)]  # shifted so B fires later than A
    result, c = _run(tmp_path, {"A": a, "B": b}, first_true_above(50.0), coefficients_doc())
    _assert_books(result, c)
    cell = result.cells[0]
    assert set(cell.targets) == {"A", "B"}
    assert [f.kind for f in cell.targets["A"].fills] == ["entry", "add", "add", "exit"]
    assert result.budget_per_target == 50_000.0
    assert cell.targets["A"].fills[0].shares == 125  # X = 1% × 50,000 / (2 × 2)


def test_an_open_position_at_the_end_is_marked(tmp_path):
    rows = [*flat_rows(24, 49.5, 1.0), (49.5, 51.0, 49.0, 50.0), (50.0, 51.0, 49.5, 50.5)]
    result, c = _run(tmp_path, {"PX": rows}, first_true_above(50.0), coefficients_doc())
    _assert_books(result, c)
    run = result.cells[0].targets["PX"]
    assert run.open_trip is not None and run.open_trip.exit_reason == "end_of_data"
    assert run.end_shares == 250 and run.trips == []
    assert run.end_stop == pytest.approx(46.0) and run.end_add_level == pytest.approx(51.0)
    assert run.samples.equity[-1] == pytest.approx(100_000 + 250 * 0.5)


def test_swept_cells_each_get_their_own_run(tmp_path):
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
    _assert_books(result, c)
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
    _assert_books(result, c)
    assert [f.price for f in result.cells[0].targets["PX"].fills] == [50.0, 51.0, 52.0, 47.0]


def test_runs_are_deterministic(tmp_path):
    (tmp_path / "1").mkdir()
    (tmp_path / "2").mkdir()
    doc = coefficients_doc(costs={}, stop_trigger="trade", exit_trigger="trade")  # the defaults
    a, _c = _run(tmp_path / "1", {"PX": worked_example_rows()}, first_true_above(50.0), doc)
    b, _ = _run(tmp_path / "2", {"PX": worked_example_rows()}, first_true_above(50.0), doc)
    ra, rb = a.cells[0].targets["PX"], b.cells[0].targets["PX"]
    assert _fill_rows(ra.fills) == _fill_rows(rb.fills) and ra.ledger == rb.ledger
    assert json_safe(asdict(ra.samples)) == json_safe(asdict(rb.samples))  # NaN → null
    assert ra.costs_paid == rb.costs_paid and ra.costs_paid.total > 0.0


# ---- the cost model, in closed form on the worked example --------------------------------


def _worked(tmp_path, costs: dict, **coef):
    return _run(
        tmp_path,
        {"PX": worked_example_rows()},
        first_true_above(50.0),
        coefficients_doc(costs=costs, **coef),
    )


def test_commission_schedule_per_fill(tmp_path):
    costs = frictionless(
        commission={
            "per_share": 0.01,
            "min_per_order": 1.0,
            "bps": 0.0,
            "sell_bps": 10.0,
            "cap_bps": 100.0,
        }
    )
    result, c = _worked(tmp_path, costs)
    _assert_books(result, c)
    run = result.cells[0].targets["PX"]
    # Fills stay at their references (no slippage); 250 shares × $0.01 = 2.50 on every buy, and
    # the exit's 750 × $0.01 = 7.50 plus 10 bps of 750 × 47 = 35,250 → 35.25.
    assert [f.price for f in run.fills] == [50.0, 51.0, 52.0, 47.0]
    assert [f.commission for f in run.fills] == pytest.approx([2.5, 2.5, 2.5, 7.5 + 35.25])
    assert all(f.slippage == 0.0 and f.shock == 0.0 and f.impact == 0.0 for f in run.fills)
    trip = run.trips[0]
    assert trip.commission == pytest.approx(7.5 + 42.75)
    assert trip.gross_pnl == pytest.approx(-3_000.0)
    assert trip.pnl == pytest.approx(-3_000.0 - 50.25)
    assert run.end_cash == pytest.approx(97_000.0 - 50.25)
    assert run.costs_paid.commission == pytest.approx(50.25)
    # The floor: 21 shares fit $1,100 all in (21 × 50 + the $1 minimum), and 21 × $0.01 = 0.21
    # is lifted to that minimum (the 1% cap, 10.50 here, never binds).
    result, _ = _run(
        tmp_path / "small",
        {
            "PX": [
                *flat_rows(24, 49.9, 0.05),
                (49.9, 50.0, 49.9, 50.0),
                (50.0, 50.15, 50.0, 50.1),
                *flat_rows(3, 50.1, 0.02),
            ]
        },
        first_true_above(50.0),
        coefficients_doc(equity=1_100, costs=costs),
    )
    entry = result.cells[0].targets["PX"].fills[0]
    assert (entry.shares, entry.commission) == (21, 1.0)


def test_a_commission_that_does_not_fit_drops_one_share(tmp_path):
    # 20 shares @ 50 fit $1,000 exactly; the $1 minimum means only 19 fit all in.
    rows = [
        *flat_rows(24, 49.9, 0.05),
        (49.9, 50.0, 49.9, 50.0),
        (50.0, 50.15, 50.0, 50.1),
        *flat_rows(3, 50.1, 0.02),
    ]
    costs = frictionless(
        commission={
            "per_share": 0.0,
            "min_per_order": 1.0,
            "bps": 0.0,
            "sell_bps": 0.0,
            "cap_bps": None,
        }
    )
    result, c = _run(
        tmp_path, {"PX": rows}, first_true_above(50.0), coefficients_doc(equity=1000, costs=costs)
    )
    _assert_books(result, c)
    run = result.cells[0].targets["PX"]
    assert [(f.shares, f.commission, f.cash_after) for f in run.fills] == [
        (19, 1.0, pytest.approx(49.0))
    ]
    assert run.ledger["entries_cash_capped"] == 1


def test_slippage_in_bps_and_in_n(tmp_path):
    result, c = _worked(tmp_path, frictionless(slippage={"bps": 5.0, "n_fraction": 0.0}))
    _assert_books(result, c)
    run = result.cells[0].targets["PX"]
    # Buys step up 5 bps of the open onto the 4-decimal grid: the entry fills at 50.025, which
    # lifts the first add level to 51.025 — bar 26's close of 51 no longer clears it, so the
    # costed path pyramids once (at bar 28's open, 52 → 52.026) and exits on the same stop_close
    # bar, stepped DOWN 5 bps (47 → 46.9765). A different path, not a shifted copy.
    assert [(f.reference, f.price) for f in run.fills] == [
        (50.0, pytest.approx(50.025)),
        (52.0, pytest.approx(52.026)),
        (47.0, pytest.approx(46.9765)),
    ]
    assert [f.slippage for f in run.fills] == pytest.approx(
        [250 * 0.025, 250 * 0.026, 500 * 0.0235]
    )
    assert run.fills[0].stop_after == pytest.approx(50.025 - 4.0)
    assert all(f.commission == 0.0 and f.shock == 0.0 for f in run.fills)
    result, c = _worked(tmp_path / "n", frictionless(slippage={"bps": 0.0, "n_fraction": 0.1}))
    run = result.cells[0].targets["PX"]
    assert run.fills[0].price == pytest.approx(50.2)  # 0.1 × N 2 above the open
    assert run.fills[0].slippage == pytest.approx(250 * 0.2)


def test_the_stop_shock_is_its_own_bucket(tmp_path):
    result, c = _worked(tmp_path, frictionless(stop_shock=0.5), stop_trigger="trade")
    _assert_books(result, c)
    run = result.cells[0].targets["PX"]
    exit_fill = run.fills[-1]
    # Frictionless buys keep the trigger at 47.9999; bar 28's low is 47.5, so the fill gives up
    # half the continuation: 47.9999 − 0.24995 → 47.74995, stepped down to 47.7499.
    assert (exit_fill.at, exit_fill.reason, exit_fill.reference) == (
        "trigger",
        "stop_trade",
        47.9999,
    )
    assert exit_fill.price == pytest.approx(47.7499)
    assert exit_fill.shock == pytest.approx(750 * 0.24995)
    assert exit_fill.slippage == pytest.approx(750 * (0.25 - 0.24995))  # the grid rounding only
    assert run.costs_paid.shock == pytest.approx(exit_fill.shock)
    assert run.trips[0].shock == pytest.approx(exit_fill.shock)
    # A stop the open gaps through references the open and carries no shock term.
    rows = [
        *flat_rows(24, 49.5, 1.0),
        (49.5, 51.0, 49.0, 50.0),
        (50.0, 51.0, 49.5, 50.0),
        (50.0, 50.5, 49.0, 49.0),
        (45.0, 45.5, 44.0, 44.5),
        *flat_rows(2, 44.5, 0.5),
    ]
    result, _ = _run(
        tmp_path / "gap",
        {"PX": rows},
        first_true_above(50.0),
        coefficients_doc(costs=frictionless(stop_shock=1.0), stop_trigger="trade"),
    )
    gap = result.cells[0].targets["PX"].fills[-1]
    assert (gap.at, gap.reference, gap.price, gap.shock) == ("open", 45.0, 45.0, 0.0)


def test_market_impact_needs_volume_and_scales_with_participation(tmp_path):
    impact = frictionless(impact={"coefficient": 0.5, "adv_window": 20})
    result, c = _worked(tmp_path, impact)
    _assert_books(result, c)
    run = result.cells[0].targets["PX"]
    # 250 shares against an ADV of 1,000 (the fixture's constant volume): sqrt(0.25) = 0.5, so
    # 0.5 × N 2 × 0.5 = +0.50 on the entry.
    assert run.fills[0].price == pytest.approx(50.5)
    assert run.fills[0].impact == pytest.approx(250 * 0.5)
    assert run.fills[0].slippage == pytest.approx(0.0)
    assert result.data.targets["PX"].volume is not None
    # The exit of 750 shares moves sqrt(0.75) × 0.5 × N below its reference.
    exit_fill = run.fills[-1]
    assert exit_fill.impact > run.fills[0].impact
    assert result.cells[0].targets["PX"].costs_paid.impact == pytest.approx(
        sum(f.impact for f in run.fills)
    )
    # Without a volume column the request is refused as a data problem before the engine runs.
    with pytest.raises(DataError) as info:
        _run(
            tmp_path / "novol",
            {"PX": worked_example_rows()},
            first_true_above(50.0),
            coefficients_doc(costs=impact),
            volume=None,
        )
    assert info.value.report["errors"][0]["code"] == "spec_data_mismatch"
    # Disabled, the volume is never read (and not even carried).
    result, _ = _worked(tmp_path / "off", frictionless())
    assert result.data.targets["PX"].volume is None


def test_the_realistic_defaults_cost_money(tmp_path):
    result, c = _worked(tmp_path, {})
    _assert_books(result, c)
    run = result.cells[0].targets["PX"]
    assert c.costs.stop_shock == 0.5 and c.costs.commission.per_share == 0.005
    assert run.costs_paid.commission > 0 and run.costs_paid.slippage > 0
    assert run.costs_paid.shock == 0.0  # close-mode exits never rest a stop
    # The entry fills 5 bps above the open, so the FIRST add level (fill + 0.5 N = 51.025) is
    # not cleared by bar 26's close of 51: the costed path pyramids once less than the
    # frictionless one and exits on the same stop_close bar — a different path, not a
    # shifted copy, which is why costs are attributed at the references and never compared
    # with the frictionless equity.
    assert [(f.kind, f.reference, f.price) for f in run.fills] == [
        ("entry", 50.0, 50.025),
        ("add", 52.0, 52.026),
        ("exit", 47.0, 46.9765),
    ]
    trip = run.trips[0]
    assert trip.units == 2 and trip.exit_reason == "stop_close"
    assert trip.pnl == pytest.approx(trip.gross_pnl - trip.commission) and trip.commission > 0
    assert run.samples.equity[-1] == pytest.approx(100_000.0 + trip.pnl)
