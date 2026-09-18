"""``seikan stock-turtle-trade-long-only`` end to end through ``main``: the worked example's
report and CSVs, the fixed layer order, silence on success (at the file-descriptor level, where
the venue logs), every refusal tier, and the schema's new sections."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from seikan.cli import REPORT_SCHEMA_VERSION, TURTLE_COMMAND, main
from seikan.emitted import validate_emitted
from tests._turtle_helpers import (
    coefficients_doc,
    first_true_above,
    flat_rows,
    thesis_doc,
    worked_example_rows,
    write_bars,
)

LAYERS = [
    "seikan_version",
    "report_schema_version",
    "command",
    "identity",
    "data_report",
    "outputs",
    "simulation",
    "targets",
    "params",
    "n_cells",
    "benchmark",
    "cells",
    "turtle_roles",
]


def _run(capsys, argv: list[str]) -> tuple[int, dict | None]:
    code = main(argv)
    out = capsys.readouterr().out
    return code, (json.loads(out) if out else None)


@pytest.fixture
def inputs(tmp_path) -> dict[str, Path]:
    rows = worked_example_rows()
    px = write_bars(tmp_path / "px.csv", rows)
    bench = write_bars(
        tmp_path / "idx.csv", [(o * 2, h * 2, lo * 2, c * 2) for o, h, lo, c in rows]
    )
    thesis = tmp_path / "thesis.json"
    thesis.write_text(json.dumps(thesis_doc(["PX"], first_true_above(50.0))), encoding="utf-8")
    coef = tmp_path / "turtle.json"
    coef.write_text(json.dumps(coefficients_doc()), encoding="utf-8")
    return {"px": px, "bench": bench, "thesis": thesis, "coef": coef, "dir": tmp_path}


def _argv(inputs, *extra: str, coef: Path | None = None) -> list[str]:
    return [
        TURTLE_COMMAND,
        str(inputs["thesis"]),
        str(coef or inputs["coef"]),
        "--data",
        f"PX={inputs['px']}",
        "--data",
        f"benchmark={inputs['bench']}",
        *extra,
    ]


def test_the_worked_example_end_to_end(inputs, capfd):
    d = inputs["dir"]
    report, trades, fills, equity = (d / n for n in ("r.json", "t.csv", "f.csv", "e.csv"))
    code = main(
        _argv(
            inputs,
            "--report-out",
            str(report),
            "--trades-out",
            str(trades),
            "--fills-out",
            str(fills),
            "--equity-out",
            str(equity),
        )
    )
    captured = capfd.readouterr()
    assert code == 0 and captured.out == ""  # silent at the fd level: the venue logs nothing
    doc = json.loads(report.read_text(encoding="utf-8"))
    assert list(doc) == LAYERS
    assert doc["report_schema_version"] == REPORT_SCHEMA_VERSION == 6
    assert doc["command"] == TURTLE_COMMAND
    validate_emitted(TURTLE_COMMAND, doc)
    ident = doc["identity"]
    assert ident["name"] == "turtle-probe" and len(ident["dsl_hash"]) == 64
    assert len(ident["coefficients_hash"]) == 64
    assert ident["coefficients"]["equity"] == 100000.0 and ident["coefficients"]["max_units"] == 3
    assert len(ident["coefficients"]) == 13
    assert set(ident["data_digests"]) == {"PX", "benchmark"}
    assert all(
        len(v["sha256"]) == 64 and v["column"] is None for v in ident["data_digests"].values()
    )
    assert {"nautilus_trader", "seikan_turtle", "numpy"} <= set(ident["environment"])
    assert doc["data_report"]["ok"] and [f["role"] for f in doc["data_report"]["files"]] == [
        "target:PX",
        "benchmark",
    ]
    assert doc["outputs"] == {
        "report": {"path": str(report)},
        "trades": {"path": str(trades), "rows_written": 1},
        "fills": {"path": str(fills), "rows_written": 4},
        "equity": {"path": str(equity), "rows_written": 31},
    }
    assert doc["simulation"]["n_bars"] == 31 and doc["simulation"]["starting_equity"] == 100000.0
    assert doc["targets"] == ["PX"] and doc["params"] == [] and doc["n_cells"] == 1
    assert doc["benchmark"]["units"] == pytest.approx(100000 / 99.0)
    cell = doc["cells"][0]
    assert cell["cell_id"] == "entry" and cell["portfolio"]["metrics"]["end_equity"] == 97000.0
    assert cell["portfolio"]["trades"]["exits"]["stop_close"] == 1
    assert cell["by_target"]["PX"]["trades"]["n_round_trips"] == 1
    assert cell["reconciliation"]["matched"] is True
    assert doc["turtle_roles"]["fill_conventions"]["entry"].startswith("the thesis fires")
    t = pd.read_csv(trades)
    assert len(t) == 1 and t.loc[0, "pnl"] == -3000.0 and t.loc[0, "exit_reason"] == "stop_close"
    f = pd.read_csv(fills)
    assert f["price"].tolist() == [50.0, 51.0, 52.0, 47.0] and f["kind"].tolist()[-1] == "exit"
    e = pd.read_csv(equity)
    assert len(e) == 31 and e["equity"].iloc[-1] == 97000.0 and "benchmark" in e.columns


def test_trade_mode_and_pretty(inputs, capsys):
    coef = inputs["dir"] / "trade.json"
    coef.write_text(json.dumps(coefficients_doc(stop_trigger="trade")), encoding="utf-8")
    report, fills = inputs["dir"] / "r.json", inputs["dir"] / "f.csv"
    code, doc = _run(
        capsys,
        _argv(
            inputs, "--report-out", str(report), "--fills-out", str(fills), "--pretty", coef=coef
        ),
    )
    assert code == 0 and doc is None
    text = report.read_text(encoding="utf-8")
    assert text.startswith("{\n  ")  # indented
    parsed = json.loads(text)
    assert parsed["identity"]["coefficients"]["stop_trigger"] == "trade"
    assert parsed["cells"][0]["portfolio"]["trades"]["exits"]["stop_trade"] == 1
    f = pd.read_csv(fills)
    assert f["price"].tolist()[-1] == 47.9999 and f["at"].tolist()[-1] == "trigger"


def test_runs_are_identical(inputs, capsys):
    a, b = inputs["dir"] / "a.json", inputs["dir"] / "b.json"
    assert _run(capsys, _argv(inputs, "--report-out", str(a)))[0] == 0
    assert _run(capsys, _argv(inputs, "--report-out", str(b)))[0] == 0
    da, db = (json.loads(p.read_text(encoding="utf-8")) for p in (a, b))
    assert da.pop("outputs") != db.pop("outputs")  # only the nominated path differs
    assert da == db


def test_the_report_is_required(inputs, capsys):
    code, doc = _run(capsys, _argv(inputs, "--trades-out", str(inputs["dir"] / "t.csv")))
    assert code == 3 and doc["error"]["type"] == "usage" and doc["command"] == TURTLE_COMMAND
    assert "--report-out is required" in doc["error"]["message"]
    assert not (inputs["dir"] / "t.csv").exists()


def test_the_benchmark_binding_is_required(inputs, capsys):
    argv = [
        TURTLE_COMMAND,
        str(inputs["thesis"]),
        str(inputs["coef"]),
        "--data",
        f"PX={inputs['px']}",
        "--report-out",
        str(inputs["dir"] / "r.json"),
    ]
    code, doc = _run(capsys, argv)
    assert code == 3 and doc["error"]["type"] == "usage"
    assert "--data benchmark=PATH is required" in doc["error"]["message"]


def test_a_column_bound_to_the_benchmark_refuses(inputs, capsys):
    code, doc = _run(
        capsys,
        _argv(inputs, "--column", "benchmark=close", "--report-out", str(inputs["dir"] / "r.json")),
    )
    assert code == 3 and doc["error"]["type"] == "usage" and "benchmark" in doc["error"]["message"]


@pytest.mark.parametrize(
    ("text", "fragment", "has_records"),
    [
        (None, "coefficients file not found", False),
        ("{", "not strict JSON", False),
        ('{"equity": NaN}', "non-standard JSON literal", False),
        ('{"equity": 1, "equity": 2}', "more than once", False),
        ("[1]", "one JSON object", False),
        ('{"equity": 100000, "gap": 1}', "1 invalid coefficient", True),
        ('{"equity": "100000"}', "1 invalid coefficient", True),
        ("{}", "1 invalid coefficient", True),
        ('{"equity": 1, "max_units": 0, "stop_n": -1}', "2 invalid coefficients", True),
    ],
)
def test_invalid_coefficients_are_their_own_envelope(inputs, capsys, text, fragment, has_records):
    coef = inputs["dir"] / "bad.json"
    if text is not None:
        coef.write_text(text, encoding="utf-8")
    code, doc = _run(
        capsys, _argv(inputs, "--report-out", str(inputs["dir"] / "r.json"), coef=coef)
    )
    assert code == 3 and doc["error"]["type"] == "coefficients_invalid"
    assert fragment in doc["error"]["message"]
    assert ("errors" in doc["error"]) == has_records
    if has_records:
        assert all({"loc", "msg", "type"} <= set(r) for r in doc["error"]["errors"])


def test_an_output_naming_the_coefficients_file_refuses(inputs, capsys):
    code, doc = _run(capsys, _argv(inputs, "--report-out", str(inputs["coef"])))
    assert code == 3 and doc["error"]["type"] == "usage"
    assert "would overwrite the coefficients file" in doc["error"]["message"]
    assert json.loads(inputs["coef"].read_text(encoding="utf-8")) == coefficients_doc()


def test_an_output_naming_the_benchmark_refuses(inputs, capsys):
    code, doc = _run(capsys, _argv(inputs, "--report-out", str(inputs["bench"])))
    assert code == 3 and "--data benchmark" in doc["error"]["message"]


def test_a_short_thesis_refuses_before_loading(inputs, capsys):
    inputs["thesis"].write_text(
        json.dumps(thesis_doc(["PX"], first_true_above(50.0), direction="shortonly")),
        encoding="utf-8",
    )
    code, doc = _run(capsys, _argv(inputs, "--report-out", str(inputs["dir"] / "r.json")))
    assert code == 3 and doc["error"]["type"] == "usage" and "shortonly" in doc["error"]["message"]


def test_a_benchmark_missing_bars_is_exit_2(inputs, capsys):
    write_bars(inputs["bench"], worked_example_rows()[:-2])
    code, doc = _run(capsys, _argv(inputs, "--report-out", str(inputs["dir"] / "r.json")))
    assert code == 2 and doc["error"]["type"] == "data_invalid"
    assert doc["data_report"]["errors"][0]["code"] == "benchmark_coverage"
    assert not (inputs["dir"] / "r.json").exists()


def test_series_targets_are_exit_2(inputs, capsys):
    idx = pd.date_range("2020-01-01", periods=40, freq="1D")
    s = pd.DataFrame({"value": [float(v) for v in range(10, 50)]}, index=idx)
    s.index.name = "datetime"
    s.to_csv(inputs["px"])
    write_bars(inputs["bench"], flat_rows(40, 100.0, 1.0))
    code, doc = _run(capsys, _argv(inputs, "--report-out", str(inputs["dir"] / "r.json")))
    assert code == 2 and doc["data_report"]["errors"][0]["code"] == "spec_data_mismatch"


def test_a_thesis_that_declares_the_market_benchmark_binds_it_once(inputs, capsys):
    inputs["thesis"].write_text(
        json.dumps(thesis_doc(["PX"], first_true_above(50.0), benchmark="market")),
        encoding="utf-8",
    )
    report = inputs["dir"] / "r.json"
    code, _doc = _run(capsys, _argv(inputs, "--report-out", str(report)))
    assert code == 0
    doc = json.loads(report.read_text(encoding="utf-8"))
    assert [f["role"] for f in doc["data_report"]["files"]] == ["target:PX", "benchmark"]
    assert set(doc["identity"]["data_digests"]) == {"PX", "benchmark"}


def test_swept_cells_ride_side_by_side(inputs, capsys):
    doc = thesis_doc(
        ["PX"],
        {
            "type": "first_true",
            "condition": {
                "type": "threshold",
                "left": {"type": "field", "column": "close"},
                "op": ">=",
                "right": {"type": "constant", "value": [50.0, 52.5], "name": "level"},
            },
        },
    )
    inputs["thesis"].write_text(json.dumps(doc), encoding="utf-8")
    report, equity = inputs["dir"] / "r.json", inputs["dir"] / "e.csv"
    code, _ = _run(capsys, _argv(inputs, "--report-out", str(report), "--equity-out", str(equity)))
    assert code == 0
    parsed = json.loads(report.read_text(encoding="utf-8"))
    assert parsed["params"] == ["level"] and parsed["n_cells"] == 2
    assert [c["cell_id"] for c in parsed["cells"]] == ["entry[level=50]", "entry[level=52.5]"]
    assert [c["params"] for c in parsed["cells"]] == [{"level": 50.0}, {"level": 52.5}]
    e = pd.read_csv(equity)
    assert len(e) == 62 and list(e.columns)[:2] == ["level", "datetime"]


def test_schema_carries_the_turtle_sections_last(capsys):
    code, doc = _run(capsys, ["schema"])
    assert code == 0
    keys = list(doc)
    turtle = [
        "turtle_coefficients",
        "turtle_report",
        "turtle_trades_csv",
        "turtle_fills_csv",
        "turtle_equity_csv",
        "turtle_roles",
    ]
    assert keys[-len(turtle) :] == turtle
    assert keys.index("turtle_coefficients") == keys.index("describe_roles") + 1
    assert doc["turtle_coefficients"]["json_schema"]["required"] == ["equity"]
    assert "add_gap" not in json.dumps(doc["turtle_coefficients"])
    assert "coefficients_invalid" in doc["exit_codes"]["3"]
    assert doc["turtle_roles"]["claim"].startswith("a SIMULATION")
