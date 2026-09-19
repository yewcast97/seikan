"""The coefficients document: defaults (the rules' and the realistic cost model's), strictness,
domains, the impact-window rule, identity."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from seikan.turtle import TurtleCoefficients, canonical_coefficients_hash
from tests._turtle_helpers import coefficients_doc, frictionless


def test_defaults_are_the_rules_braced_values_and_a_realistic_cost_model():
    c = TurtleCoefficients.model_validate({"equity": 100_000.0})
    assert c.equity == 100_000.0
    assert (c.atr_period, c.add_step_n, c.max_units, c.stop_n) == (20, 0.5, 3, 2.0)
    assert (c.exit_lookback, c.risk_per_unit) == (20, 0.01)
    assert (c.stop_trigger, c.exit_trigger, c.stop_n_source) == ("close", "close", "current")
    assert (c.bars_per_year, c.currency, c.price_precision) == (252, "USD", 4)
    k = c.costs
    assert (k.commission.per_share, k.commission.min_per_order) == (0.005, 1.0)
    assert (k.commission.bps, k.commission.sell_bps, k.commission.cap_bps) == (0.0, 0.0, 100.0)
    assert (k.slippage.bps, k.slippage.n_fraction) == (5.0, 0.0)
    assert (k.impact.coefficient, k.impact.adv_window) == (0.0, 20)
    assert k.stop_shock == 0.5
    z = TurtleCoefficients.model_validate(coefficients_doc()).costs
    assert z.stop_shock == 0.0 and z.commission.cap_bps is None and z.slippage.bps == 0.0


def test_equity_is_required_and_an_int_is_a_legal_float():
    with pytest.raises(ValidationError, match="equity"):
        TurtleCoefficients.model_validate({})
    assert TurtleCoefficients.model_validate({"equity": 100000}).equity == 100_000.0


@pytest.mark.parametrize(
    "doc",
    [
        {"equity": "100000"},
        {"equity": True},
        {"equity": 0},
        {"equity": -1.0},
        {"equity": 1.0, "risk_per_unit": 1.5},
        {"equity": 1.0, "risk_per_unit": 0},
        {"equity": 1.0, "max_units": 0},
        {"equity": 1.0, "atr_period": 0},
        {"equity": 1.0, "exit_lookback": 0},
        {"equity": 1.0, "price_precision": 10},
        {"equity": 1.0, "stop_trigger": "open"},
        {"equity": 1.0, "stop_n_source": "latest"},
        {"equity": 1.0, "currency": "usd"},
        {"equity": 1.0, "bars_per_year": 0},
        {"equity": 1.0, "add_gap_skip_n": 1.0},
        {"equity": 1.0, "stop_n": "2"},
        {"equity": 1.0, "costs": {"stop_shock": 1.5}},
        {"equity": 1.0, "costs": {"stop_shock": -0.1}},
        {"equity": 1.0, "costs": {"commission": {"per_share": -0.01}}},
        {"equity": 1.0, "costs": {"commission": {"cap_bps": 0}}},
        {"equity": 1.0, "costs": {"slippage": {"bps": "5"}}},
        {"equity": 1.0, "costs": {"impact": {"adv_window": 0}}},
        {"equity": 1.0, "costs": {"impact": {"coefficient": 0.5, "adv_window": 21}}},
        {"equity": 1.0, "costs": {"participation_cap": 0.1}},
    ],
)
def test_invalid_documents_refuse(doc):
    with pytest.raises(ValidationError):
        TurtleCoefficients.model_validate(doc)


def test_the_impact_window_must_sit_inside_the_rules_warmup():
    ok = TurtleCoefficients.model_validate(
        {"equity": 1.0, "costs": {"impact": {"coefficient": 0.5, "adv_window": 20}}}
    )
    assert ok.costs.impact.adv_window == 20
    # A disabled impact may carry any window: it is never read.
    off = TurtleCoefficients.model_validate(
        {"equity": 1.0, "costs": {"impact": {"adv_window": 60}}}
    )
    assert off.costs.impact.coefficient == 0.0
    wider = TurtleCoefficients.model_validate(
        {
            "equity": 1.0,
            "exit_lookback": 55,
            "costs": {"impact": {"coefficient": 0.5, "adv_window": 55}},
        }
    )
    assert wider.costs.impact.adv_window == 55
    with pytest.raises(ValidationError, match="adv_window"):
        TurtleCoefficients.model_validate(
            {"equity": 1.0, "costs": {"impact": {"coefficient": 0.5, "adv_window": 21}}}
        )


def test_the_model_is_frozen():
    c = TurtleCoefficients.model_validate(coefficients_doc())
    with pytest.raises(ValidationError):
        c.equity = 1.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        c.costs.stop_shock = 1.0  # type: ignore[misc]


def test_canonical_hash_ignores_spelling_and_tracks_values():
    minimal = canonical_coefficients_hash({"equity": 100000})
    spelled = canonical_coefficients_hash(
        {
            "price_precision": 4,
            "currency": "USD",
            "bars_per_year": 252,
            "stop_n_source": "current",
            "exit_trigger": "close",
            "stop_trigger": "close",
            "risk_per_unit": 0.01,
            "exit_lookback": 20,
            "stop_n": 2.0,
            "max_units": 3,
            "add_step_n": 0.5,
            "atr_period": 20,
            "equity": 100000,
            "costs": {
                "stop_shock": 0.5,
                "impact": {"adv_window": 20, "coefficient": 0.0},
                "slippage": {"n_fraction": 0.0, "bps": 5.0},
                "commission": {
                    "cap_bps": 100.0,
                    "sell_bps": 0.0,
                    "bps": 0.0,
                    "min_per_order": 1.0,
                    "per_share": 0.005,
                },
            },
        }
    )
    assert minimal == spelled and len(minimal) == 64
    assert canonical_coefficients_hash({"equity": 100000, "costs": {"stop_shock": 0.5}}) == minimal
    assert canonical_coefficients_hash({"equity": 100000, "costs": {"stop_shock": 0.25}}) != minimal
    assert canonical_coefficients_hash(coefficients_doc()) != minimal  # frictionless ≠ defaults
    assert canonical_coefficients_hash(coefficients_doc(stop_trigger="trade")) != minimal
    with pytest.raises(ValidationError):
        canonical_coefficients_hash({"equity": 1.0, "extra": 1})


def test_json_schema_marks_equity_required():
    schema = TurtleCoefficients.model_json_schema()
    assert schema["required"] == ["equity"]
    assert schema["additionalProperties"] is False
    text = json.dumps(schema)
    for name in ("stop_shock", "per_share", "cap_bps", "n_fraction", "adv_window"):
        assert name in text


def test_frictionless_spells_every_field():
    z = frictionless()
    assert set(z) == {"commission", "slippage", "impact", "stop_shock"}
    assert set(z["commission"]) == {"per_share", "min_per_order", "bps", "sell_bps", "cap_bps"}
    assert set(z["slippage"]) == {"bps", "n_fraction"}
    assert set(z["impact"]) == {"coefficient", "adv_window"}
    c = TurtleCoefficients.model_validate(coefficients_doc()).costs
    assert c.commission.per_share == 0.0 and c.commission.min_per_order == 0.0
    assert c.slippage.bps == 0.0 and c.stop_shock == 0.0 and c.impact.coefficient == 0.0
