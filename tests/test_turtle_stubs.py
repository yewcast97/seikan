"""The ``seikan._turtle`` stub and the compiled extension agree in both directions, and the
engine's Python surface behaves as the stub documents."""

from __future__ import annotations

import ast
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


def _coefficients(**overrides: object) -> _turtle.Coefficients:
    kwargs: dict[str, object] = {
        "atr_period": 20,
        "add_step_n": 0.5,
        "max_units": 3,
        "stop_n": 2.0,
        "exit_lookback": 20,
        "risk_per_unit": 0.01,
        "stop_trigger": "close",
        "exit_trigger": "close",
        "stop_n_source": "current",
        "price_precision": 4,
        "budget": 100_000.0,
    }
    return _turtle.Coefficients(**{**kwargs, **overrides})  # type: ignore[arg-type]


def _frictionless(**overrides: object) -> _turtle.CostModel:
    kwargs: dict[str, object] = {
        "per_share": 0.0,
        "min_per_order": 0.0,
        "bps": 0.0,
        "sell_bps": 0.0,
        "cap_bps": None,
        "slippage_bps": 0.0,
        "slippage_n_fraction": 0.0,
        "impact_coefficient": 0.0,
        "adv_window": 1,
        "stop_shock": 0.0,
    }
    return _turtle.CostModel(**{**kwargs, **overrides})  # type: ignore[arg-type]


def _worked_example() -> tuple[list[float], list[float], list[float], list[float], list[bool]]:
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
    return o, h, lo, cl, fired


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
    c = _coefficients(exit_trigger="trade")
    assert (c.stop_trigger, c.exit_trigger, c.stop_n_source) == ("close", "trade", "current")
    assert c.first_eligible_bar() == 19 and c.budget == 100_000.0
    assert "Coefficients(" in repr(c)
    with pytest.raises(ValueError, match=r"trigger"):
        _coefficients(stop_trigger="open")
    with pytest.raises(ValueError, match="max_units"):
        _coefficients(max_units=0)


def test_the_cost_model_validates_and_exposes_its_fields():
    m = _frictionless(per_share=0.005, min_per_order=1.0, cap_bps=100.0, slippage_bps=5.0)
    assert (m.per_share, m.min_per_order, m.cap_bps, m.slippage_bps) == (0.005, 1.0, 100.0, 5.0)
    assert m.stop_shock == 0.0 and not m.impact_enabled() and _frictionless().cap_bps is None
    assert _frictionless(impact_coefficient=0.5, adv_window=20).impact_enabled()
    assert repr(m).startswith("CostModel(")
    with pytest.raises(ValueError, match="stop_shock"):
        _frictionless(stop_shock=1.5)
    with pytest.raises(ValueError, match="cap_bps"):
        _frictionless(cap_bps=0.0)
    with pytest.raises(ValueError, match="adv_window"):
        _frictionless(adv_window=0)


def test_simulate_runs_the_worked_example_frictionless():
    o, h, lo, cl, fired = _worked_example()
    res = _turtle.simulate(_coefficients(), _frictionless(), o, h, lo, cl, None, fired)
    got = [(f.bar, f.kind, f.shares, f.price, f.at, f.reason) for f in res.fills]
    assert got == [
        (25, "entry", 250, 50.0, "open", None),
        (26, "add", 250, 51.0, "open", None),
        (27, "add", 250, 52.0, "open", None),
        (29, "exit", 750, 47.0, "open", "stop_close"),
    ]
    assert [f.reference for f in res.fills] == [50.0, 51.0, 52.0, 47.0]
    assert [(f.cash_after, f.shares_after, f.units_after) for f in res.fills][-1] == (
        97_000.0,
        0,
        0,
    )
    assert res.fills[0].stop_after == pytest.approx(46.0) and res.fills[-1].stop_after is None
    assert res.stop[25:28] == [46.0, 47.0, 48.0]
    assert res.ledger.to_dict()["exits_stop_close"] == 1 and res.ledger.entries == 1
    trip = res.trips[0]
    assert trip.pnl == pytest.approx(750 * 47.0 - 250 * (50 + 51 + 52))
    assert trip.gross_pnl == trip.pnl and (trip.commission, trip.shock) == (0.0, 0.0)
    assert res.open_trip is None and res.first_eligible_bar == 19
    assert len(res.equity) == len(o) == len(res.commission_cum) == len(res.impact_cum)
    assert set(res.commission_cum) == {0.0} and set(res.shock_cum) == {0.0}


def test_simulate_refuses_unusable_input():
    o, h, lo, cl, fired = _worked_example()
    with pytest.raises(ValueError, match="equal lengths"):
        _turtle.simulate(_coefficients(), _frictionless(), o, h, lo, cl, None, fired[:-1])
    with pytest.raises(ValueError, match="volume"):
        _turtle.simulate(
            _coefficients(),
            _frictionless(impact_coefficient=0.5, adv_window=20),
            o,
            h,
            lo,
            cl,
            None,
            fired,
        )
    with pytest.raises(ValueError, match="adv_window"):
        _turtle.simulate(
            _coefficients(),
            _frictionless(impact_coefficient=0.5, adv_window=30),
            o,
            h,
            lo,
            cl,
            [1000.0] * len(o),
            fired,
        )
    res = _turtle.simulate(
        _coefficients(),
        _frictionless(impact_coefficient=0.5, adv_window=20),
        o,
        h,
        lo,
        cl,
        [1000.0] * len(o),
        fired,
    )
    assert res.fills[0].price == pytest.approx(50.5) and res.fills[0].impact == pytest.approx(125.0)
