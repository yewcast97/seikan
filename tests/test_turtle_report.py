"""The report sections and CSV frames assembled from a real simulation."""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from seikan.dsl.schema import Thesis
from seikan.serialize import json_safe
from seikan.turtle import TurtleCoefficients
from seikan.turtle.engine import run_turtle
from seikan.turtle.report import (
    equity_frame,
    fills_frame,
    report_sections,
    round_trips_frame,
    write_csv,
)
from tests._helpers import load
from tests._turtle_helpers import (
    coefficients_doc,
    first_true_above,
    flat_rows,
    thesis_doc,
    worked_example_rows,
    write_bars,
)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("turtle")
    rows = worked_example_rows()
    px = write_bars(tmp / "px.csv", rows)
    bench = write_bars(tmp / "idx.csv", [(o * 2, h * 2, lo * 2, c * 2) for o, h, lo, c in rows])
    thesis = Thesis.model_validate(thesis_doc(["PX"], first_true_above(50.0)))
    md = load(thesis, {"PX": str(px)})
    c = TurtleCoefficients.model_validate(coefficients_doc())
    return run_turtle(thesis, md, c, str(bench))


def test_sections_carry_the_worked_example(result):
    sections = report_sections(result)
    sim = sections.simulation
    assert (sim["engine"], sim["venue"], sim["oms_type"], sim["account_type"]) == (
        "nautilus_trader",
        "SIM",
        "NETTING",
        "CASH",
    )
    assert sim["n_bars"] == 31 and sim["first_eligible_bar"] == 19
    assert sim["instruments"] == {"PX": "T0.SIM"} and sim["budget_per_target"] == 100_000.0
    assert sim["price_increment"] == 0.0001 and sim["bar_spacing"]["median_seconds"] == 86_400
    assert sim["index_start"] == "2020-01-01T00:00:00" and sim["liquidity"] == "unlimited"
    assert sections.targets == ["PX"] and sections.params == [] and sections.n_cells == 1
    bench = sections.benchmark
    assert bench["units"] == pytest.approx(100_000 / 99.0)
    assert bench["end_equity"] == pytest.approx(100_000 / 99.0 * 94.0)
    assert bench["metrics"]["exposure"]["fraction_in_market"] == 1.0
    cell = sections.cells[0]
    assert cell["cell_id"] == "entry" and cell["params"] == {}
    port = cell["portfolio"]
    assert port["metrics"]["end_equity"] == pytest.approx(97_000.0)
    assert port["metrics"]["net_pnl"] == pytest.approx(-3_000.0)
    assert port["trades"]["n_round_trips"] == 1 and port["trades"]["exits"]["stop_close"] == 1
    assert port["trades"]["net_pnl"] == pytest.approx(-3_000.0)
    assert port["metrics"]["exposure"]["bars_in_market"] == 4  # bars 25..28
    assert port["metrics"]["drawdown"]["trough_time"] is not None
    assert list(port["periodic"]["monthly"]) == ["2020-01"]
    target = cell["by_target"]["PX"]
    assert target["metrics"]["start_equity"] == 100_000.0
    assert target["end_state"] == {
        "shares": 0,
        "units": 0,
        "cash": 97_000.0,
        "market_value": 0.0,
        "stop": None,
        "add_level": None,
        "in_position": False,
    }
    assert target["trades"]["mean_mae"] == pytest.approx(46.5 / 50.0 - 1.0)
    assert target["trades"]["mean_mfe"] == pytest.approx(53.5 / 50.0 - 1.0)
    assert cell["reconciliation"]["matched"] is True
    assert cell["engine_stats"]["stats_pnls"]["USD"]["PnL (total)"] == pytest.approx(-3_000.0)
    # Every number survives JSON with NaN as null and nothing else lost.
    text = json.dumps(json_safe(cell), allow_nan=False)
    assert "NaN" not in text and "Max Winner" in text
    assert all(
        not isinstance(v, float) or math.isfinite(v)
        for v in json_safe(cell["engine_stats"]).get("stats_returns", {}).values()
        if v is not None
    )


def test_csv_frames_and_writers(result, tmp_path):
    trips = round_trips_frame(result)
    assert list(trips.columns)[:4] == ["target", "entry_bar", "entry_time", "exit_bar"]
    assert len(trips) == 1
    row = trips.iloc[0]
    assert (row["entry_bar"], row["exit_bar"], row["exit_reason"], row["is_open"]) == (
        25,
        29,
        "stop_close",
        0,
    )
    assert row["pnl"] == pytest.approx(-3_000.0) and row["units"] == 3 and row["max_stop"] == 48.0
    assert row["avg_entry_px"] == pytest.approx(51.0) and row["bars_held"] == 4
    fills = fills_frame(result)
    assert [
        tuple(r) for r in fills[["bar", "kind", "shares", "price", "at"]].itertuples(index=False)
    ] == [
        (25, "entry", 250, 50.0, "open"),
        (26, "add", 250, 51.0, "open"),
        (27, "add", 250, 52.0, "open"),
        (29, "exit", 750, 47.0, "open"),
    ]
    assert (
        fills["notional"].tolist()[0] == 12_500.0 and fills["reason"].tolist()[-1] == "stop_close"
    )
    equity = equity_frame(result)
    assert list(equity.columns) == [
        "datetime", "equity", "benchmark", "cash", "market_value", "gross_exposure",
        "n_positions", "shares", "units", "stop", "add_level",
    ]  # fmt: skip
    assert len(equity) == 31 and equity["equity"].iloc[-1] == 97_000.0
    assert equity["n_positions"].tolist()[25:29] == [1, 1, 1, 1] and equity["stop"].iloc[27] == 48.0
    assert math.isnan(equity["stop"].iloc[0])
    for name, frame in (("t", trips), ("f", fills), ("e", equity)):
        path = tmp_path / f"{name}.csv"
        assert write_csv(frame, str(path)) == len(frame)
        back = pd.read_csv(path)
        assert list(back.columns) == list(frame.columns) and len(back) == len(frame)


def test_several_targets_suffix_the_equity_columns(tmp_path):
    rows = worked_example_rows()
    a = write_bars(tmp_path / "a.csv", rows)
    b = write_bars(tmp_path / "b.csv", [(o + 5, h + 5, lo + 5, c + 5) for o, h, lo, c in rows])
    bench = write_bars(tmp_path / "idx.csv", flat_rows(len(rows), 100.0, 1.0))
    thesis = Thesis.model_validate(thesis_doc(["A", "B"], first_true_above(50.0)))
    md = load(thesis, {"A": str(a), "B": str(b)})
    result = run_turtle(
        thesis, md, TurtleCoefficients.model_validate(coefficients_doc()), str(bench)
    )
    equity = equity_frame(result)
    assert "shares@A" in equity.columns and "add_level@B" in equity.columns
    sections = report_sections(result)
    assert sections.simulation["instruments"] == {"A": "T0.SIM", "B": "T1.SIM"}
    assert set(sections.cells[0]["by_target"]) == {"A", "B"}
    assert sections.cells[0]["portfolio"]["metrics"]["start_equity"] == 100_000.0
    assert sections.cells[0]["by_target"]["A"]["metrics"]["start_equity"] == 50_000.0
