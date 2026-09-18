"""The coefficients document: defaults, strictness, domains, identity — and the package's
promise that importing it never imports the venue."""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from seikan.turtle import TurtleCoefficients, canonical_coefficients_hash
from tests._turtle_helpers import coefficients_doc


def test_defaults_are_the_rules_braced_values():
    c = TurtleCoefficients.model_validate(coefficients_doc())
    assert c.equity == 100_000.0
    assert (c.atr_period, c.add_step_n, c.max_units, c.stop_n) == (20, 0.5, 3, 2.0)
    assert (c.exit_lookback, c.risk_per_unit) == (20, 0.01)
    assert (c.stop_trigger, c.exit_trigger, c.stop_n_source) == ("close", "close", "current")
    assert (c.bars_per_year, c.currency, c.price_precision) == (252, "USD", 4)


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
    ],
)
def test_invalid_documents_refuse(doc):
    with pytest.raises(ValidationError):
        TurtleCoefficients.model_validate(doc)


def test_the_model_is_frozen():
    c = TurtleCoefficients.model_validate(coefficients_doc())
    with pytest.raises(ValidationError):
        c.equity = 1.0  # type: ignore[misc]


def test_canonical_hash_ignores_spelling_and_tracks_values():
    minimal = canonical_coefficients_hash(coefficients_doc())
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
        }
    )
    assert minimal == spelled and len(minimal) == 64
    assert canonical_coefficients_hash(coefficients_doc(stop_trigger="trade")) != minimal
    with pytest.raises(ValidationError):
        canonical_coefficients_hash({"equity": 1.0, "extra": 1})


def test_json_schema_marks_equity_required():
    schema = TurtleCoefficients.model_json_schema()
    assert schema["required"] == ["equity"]
    assert schema["additionalProperties"] is False


def test_importing_the_package_does_not_import_the_venue():
    code = (
        "import sys, seikan.turtle, seikan.cli; "
        "assert 'nautilus_trader' not in sys.modules, 'nautilus imported at package import'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
