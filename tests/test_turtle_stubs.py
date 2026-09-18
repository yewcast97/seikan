"""The ``seikan._turtle`` stub and the compiled extension agree in both directions, and the
kernel's Python surface behaves as the stub documents."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from seikan import _turtle


def _stub_tree() -> ast.Module:
    path = Path(__import__("seikan").__file__).parent / "_turtle.pyi"
    return ast.parse(path.read_text(encoding="utf-8"))


def _stub_names() -> dict[str, set[str]]:
    """Top-level functions/classes of the stub, each class mapped to its declared members."""
    out: dict[str, set[str]] = {}
    for node in _stub_tree().body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = set()
        elif isinstance(node, ast.ClassDef):
            members: set[str] = set()
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    members.add(item.target.id)
                elif isinstance(item, ast.FunctionDef) and not item.name.startswith("__"):
                    members.add(item.name)
            out[node.name] = members
    return out


def test_every_stub_name_exists_on_the_extension_and_vice_versa():
    stub = _stub_names()
    public = {n for n in dir(_turtle) if not n.startswith("_")}
    assert set(stub) == public
    for name, members in stub.items():
        obj = getattr(_turtle, name)
        if isinstance(obj, type):
            actual = {n for n in dir(obj) if not n.startswith("_")}
            assert members == actual, name


def test_coefficients_validate_and_expose_their_fields():
    c = _turtle.Coefficients(
        atr_period=20,
        add_step_n=0.5,
        max_units=3,
        stop_n=2.0,
        exit_lookback=20,
        risk_per_unit=0.01,
        stop_trigger="close",
        exit_trigger="trade",
        stop_n_source="current",
        price_precision=4,
        budget=100_000.0,
    )
    assert (c.stop_trigger, c.exit_trigger, c.stop_n_source) == ("close", "trade", "current")
    assert c.first_eligible_bar() == 19 == _turtle.first_eligible_bar(20, 20)
    with pytest.raises(ValueError, match=r"trigger"):
        _turtle.Coefficients(
            atr_period=20,
            add_step_n=0.5,
            max_units=3,
            stop_n=2.0,
            exit_lookback=20,
            risk_per_unit=0.01,
            stop_trigger="open",
            exit_trigger="close",
            stop_n_source="current",
            price_precision=4,
            budget=100_000.0,
        )
    with pytest.raises(ValueError, match="max_units"):
        _turtle.Coefficients(
            atr_period=20,
            add_step_n=0.5,
            max_units=0,
            stop_n=2.0,
            exit_lookback=20,
            risk_per_unit=0.01,
            stop_trigger="close",
            exit_trigger="close",
            stop_n_source="current",
            price_precision=4,
            budget=100_000.0,
        )


def test_indicators_follow_the_nautilus_idiom():
    atr = _turtle.WilderAtr(3)
    assert atr.name == "WilderAtr(3)" and not atr.has_inputs and math.isnan(atr.value)

    class Bar:
        def __init__(self, h: float, l: float, c: float) -> None:  # noqa: E741 - OHLC
            self.high, self.low, self.close = h, l, c

    for bar in (Bar(11.0, 9.0, 10.0), Bar(13.0, 10.5, 12.0)):
        atr.handle_bar(bar)
        assert atr.has_inputs and not atr.initialized
    atr.handle_bar(Bar(12.5, 11.5, 12.0))
    assert atr.initialized and atr.value == pytest.approx(2.0) and atr.count == 3
    atr.reset()
    assert not atr.has_inputs

    ch = _turtle.LowestLowChannel(2)
    ch.update_raw(5.0)
    assert not ch.initialized and math.isnan(ch.value)
    ch.handle_bar(Bar(6.0, 4.0, 5.0))
    assert ch.initialized and ch.value == 4.0 and ch.lookback == 2


def test_the_reference_simulator_runs_the_worked_example():
    c = _turtle.Coefficients(
        atr_period=20,
        add_step_n=0.5,
        max_units=3,
        stop_n=2.0,
        exit_lookback=20,
        risk_per_unit=0.01,
        stop_trigger="close",
        exit_trigger="close",
        stop_n_source="current",
        price_precision=4,
        budget=100_000.0,
    )
    o, h, lo, cl, fired = [], [], [], [], []
    for _ in range(24):
        o.append(49.5), h.append(50.5), lo.append(48.5), cl.append(49.5), fired.append(False)
    rows = [
        (49.5, 51.0, 49.0, 50.0, True),
        (50.0, 52.0, 50.0, 51.0, False),
        (51.0, 53.0, 51.0, 52.0, False),
        (52.0, 53.5, 51.5, 52.5, False),
        (52.5, 52.5, 47.5, 47.5, False),
        (47.0, 47.5, 46.5, 47.0, False),
        (47.0, 48.0, 46.0, 47.0, False),
    ]
    for r in rows:
        o.append(r[0]), h.append(r[1]), lo.append(r[2]), cl.append(r[3]), fired.append(r[4])
    res = _turtle.simulate_reference(c, o, h, lo, cl, fired)
    got = [(f.bar, f.kind, f.shares, f.price, f.at, f.reason) for f in res.fills]
    assert got == [
        (25, "entry", 250, 50.0, "open", None),
        (26, "add", 250, 51.0, "open", None),
        (27, "add", 250, 52.0, "open", None),
        (29, "exit", 750, 47.0, "open", "stop_close"),
    ]
    assert res.stop[25:28] == [46.0, 47.0, 48.0]
    assert res.ledger.to_dict()["exits_stop_close"] == 1 and res.ledger.entries == 1
    assert res.trips[0].pnl == pytest.approx(750 * 47.0 - 250 * (50 + 51 + 52))
    assert res.open_trip is None and res.first_eligible_bar == 19
    assert len(res.equity) == len(o)


def test_the_machine_is_driven_step_by_step():
    c = _turtle.Coefficients(
        atr_period=1,
        add_step_n=0.5,
        max_units=3,
        stop_n=2.0,
        exit_lookback=1,
        risk_per_unit=0.01,
        stop_trigger="trade",
        exit_trigger="close",
        stop_n_source="entry",
        price_precision=2,
        budget=10_000.0,
    )
    m = _turtle.Machine(c)
    assert m.pending == "none" and not m.in_position and m.cash == 10_000.0
    m.on_bar(0, 100.0, n=2.0, channel=99.0, fired=True, last_bar=False)
    assert m.pending == "enter" and m.resting() is None
    action = m.on_print(1, 101.0)
    assert (action.kind, action.intent, action.shares) == ("buy", "entry", 25)
    m.on_fill(1, "entry", 25, 101.0)
    assert m.in_position and m.shares == 25 and m.stop == pytest.approx(97.0)
    resting = m.resting()
    assert resting is not None and resting.reason == "stop_trade"
    assert resting.trigger == pytest.approx(96.99) and resting.shares == 25
    assert m.equity(102.0) == pytest.approx(10_000.0 - 25 * 101.0 + 25 * 102.0)
    m.on_bar(1, 102.0, n=2.0, channel=100.0, fired=False, last_bar=False)
    assert m.pending == "add" and m.add_level == pytest.approx(102.0)
    open_trip = m.finish(1, 102.0)
    assert open_trip is not None and open_trip.is_open and open_trip.exit_reason == "end_of_data"
    with pytest.raises(ValueError):
        m.on_fill(2, "exit", 1, 100.0)
