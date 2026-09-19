"""The report sections and CSV frames assembled from a real simulation — frictionless (the
rules' numbers) and under the realistic cost model (the cost reads)."""

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

ZERO_COSTS = {"commission": 0.0, "slippage": 0.0, "shock": 0.0, "impact": 0.0, "total": 0.0}


def _result(tmp, coef: dict):
    rows = worked_example_rows()
    px = write_bars(tmp / "px.csv", rows)
    bench = write_bars(tmp / "idx.csv", [(o * 2, h * 2, lo * 2, c * 2) for o, h, lo, c in rows])
    thesis = Thesis.model_validate(thesis_doc(["PX"], first_true_above(50.0)))
    md = load(thesis, {"PX": str(px)})
    return run_turtle(thesis, md, TurtleCoefficients.model_validate(coef), str(bench))


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return _result(tmp_path_factory.mktemp("turtle"), coefficients_doc())


@pytest.fixture(scope="module")
def costed(tmp_path_factory):
    return _result(tmp_path_factory.mktemp("costed"), coefficients_doc(costs={}))


def test_sections_carry_the_worked_example(result):
    sections = report_sections(result)
    sim = sections.simulation
    assert sim["engine"] == "seikan._turtle" and sim["engine_version"].count(".") == 2
    assert sim["n_bars"] == 31 and sim["first_eligible_bar"] == 19 and sim["lot_size"] == 1
    assert sim["budget_per_target"] == 100_000.0 and sim["liquidity"] == "unlimited"
    assert sim["price_increment"] == 0.0001 and sim["bar_spacing"]["median_seconds"] == 86_400
    assert sim["index_start"] == "2020-01-01T00:00:00" and "venue" not in sim
    assert sim["fill_conventions"]["stop_trade"].startswith("stop_trigger 'trade'")
    assert sections.targets == ["PX"] and sections.params == [] and sections.n_cells == 1
    bench = sections.benchmark
    assert bench["units"] == pytest.approx(100_000 / 99.0)
    assert bench["end_equity"] == pytest.approx(100_000 / 99.0 * 94.0)
    assert bench["metrics"]["exposure"]["fraction_in_market"] == 1.0
    assert "frictionless" in bench["construction"]
    cell = sections.cells[0]
    assert set(cell) == {"cell_id", "params", "portfolio", "by_target"}
    assert cell["cell_id"] == "entry" and cell["params"] == {}
    port = cell["portfolio"]
    assert port["metrics"]["end_equity"] == pytest.approx(97_000.0)
    assert port["metrics"]["net_pnl"] == pytest.approx(-3_000.0)
    assert port["trades"]["n_round_trips"] == 1 and port["trades"]["exits"]["stop_close"] == 1
    assert port["trades"]["net_pnl"] == port["trades"]["gross_pnl"] == pytest.approx(-3_000.0)
    assert port["trades"]["costs"] == ZERO_COSTS and port["costs_paid"] == ZERO_COSTS
    assert port["trades"]["turnover"] == pytest.approx(38_250.0 + 35_250.0)
    assert port["trades"]["cost_bps_of_turnover"] == 0.0
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
        "costs_paid": ZERO_COSTS,
    }
    assert target["trades"]["mean_mae"] == pytest.approx(46.5 / 50.0 - 1.0)
    assert target["trades"]["mean_mfe"] == pytest.approx(53.5 / 50.0 - 1.0)
    # Every number survives JSON with NaN as null and nothing else lost.
    text = json.dumps(json_safe(cell), allow_nan=False)
    assert "NaN" not in text and "nautilus" not in text


def test_the_cost_reads_under_the_realistic_defaults(costed):
    sections = report_sections(costed)
    port = sections.cells[0]["portfolio"]
    trades = port["trades"]
    assert trades["n_round_trips"] == 1
    assert trades["costs"]["commission"] > 0 and trades["costs"]["slippage"] > 0
    assert trades["costs"]["shock"] == 0.0 and trades["costs"]["impact"] == 0.0
    assert trades["costs"]["total"] == pytest.approx(
        trades["costs"]["commission"] + trades["costs"]["slippage"]
    )
    assert trades["net_pnl"] == pytest.approx(trades["gross_pnl"] - trades["costs"]["commission"])
    assert trades["cost_bps_of_turnover"] == pytest.approx(
        trades["costs"]["total"] / trades["turnover"] * 1e4
    )
    # One closed trip and nothing open: the closed-trip costs ARE the costs paid.
    assert port["costs_paid"] == trades["costs"]
    assert sections.cells[0]["by_target"]["PX"]["end_state"]["costs_paid"] == trades["costs"]
    assert port["metrics"]["end_equity"] == pytest.approx(100_000.0 + trades["net_pnl"])


def test_csv_frames_and_writers(result, costed, tmp_path):
    trips = round_trips_frame(result)
    assert list(trips.columns)[:4] == ["target", "entry_bar", "entry_time", "exit_bar"]
    assert list(trips.columns)[14:22] == [
        "proceeds", "gross_pnl", "commission", "slippage", "shock", "impact", "pnl", "ret",
    ]  # fmt: skip
    assert len(trips) == 1
    row = trips.iloc[0]
    assert (row["entry_bar"], row["exit_bar"], row["exit_reason"], row["is_open"]) == (
        25,
        29,
        "stop_close",
        0,
    )
    assert row["pnl"] == pytest.approx(-3_000.0) and row["units"] == 3 and row["max_stop"] == 48.0
    assert row["gross_pnl"] == row["pnl"] and row["commission"] == 0.0
    assert row["avg_entry_px"] == pytest.approx(51.0) and row["bars_held"] == 4
    fills = fills_frame(result)
    assert [
        tuple(r)
        for r in fills[["bar", "kind", "shares", "reference", "price", "at"]].itertuples(
            index=False
        )
    ] == [
        (25, "entry", 250, 50.0, 50.0, "open"),
        (26, "add", 250, 51.0, 51.0, "open"),
        (27, "add", 250, 52.0, 52.0, "open"),
        (29, "exit", 750, 47.0, 47.0, "open"),
    ]
    assert (
        fills["notional"].tolist()[0] == 12_500.0 and fills["reason"].tolist()[-1] == "stop_close"
    )
    assert fills["side"].tolist() == ["buy", "buy", "buy", "sell"]
    assert "client_order_id" not in fills.columns
    equity = equity_frame(result)
    assert list(equity.columns) == [
        "datetime", "equity", "benchmark", "cash", "market_value", "gross_exposure",
        "n_positions", "commission_cum", "slippage_cum", "shock_cum", "impact_cum",
        "shares", "units", "stop", "add_level",
    ]  # fmt: skip
    assert len(equity) == 31 and equity["equity"].iloc[-1] == 97_000.0
    assert equity["n_positions"].tolist()[25:29] == [1, 1, 1, 1] and equity["stop"].iloc[27] == 48.0
    assert math.isnan(equity["stop"].iloc[0]) and equity["commission_cum"].sum() == 0.0
    costed_fills = fills_frame(costed)
    # The costed path pyramids once (the 5-bps entry lifts the first add level past bar 26's
    # close): two buys above their references, one sell below.
    assert (costed_fills["price"] > costed_fills["reference"]).tolist() == [True, True, False]
    assert (costed_fills["commission"] > 0).all() and (costed_fills["slippage"] > 0).all()
    costed_equity = equity_frame(costed)
    assert costed_equity["commission_cum"].is_monotonic_increasing
    assert costed_equity["commission_cum"].iloc[-1] == pytest.approx(
        costed_fills["commission"].sum()
    )
    for name, frame in (("t", trips), ("f", fills), ("e", equity)):
        path = tmp_path / f"{name}.csv"
        assert write_csv(frame, str(path)) == len(frame)
        back = pd.read_csv(path)
        assert list(back.columns) == list(frame.columns) and len(back) == len(frame)


def test_several_targets_suffix_the_equity_columns_and_sum_the_costs(tmp_path):
    rows = worked_example_rows()
    a = write_bars(tmp_path / "a.csv", rows)
    b = write_bars(tmp_path / "b.csv", [(o + 5, h + 5, lo + 5, c + 5) for o, h, lo, c in rows])
    bench = write_bars(tmp_path / "idx.csv", flat_rows(len(rows), 100.0, 1.0))
    thesis = Thesis.model_validate(thesis_doc(["A", "B"], first_true_above(50.0)))
    md = load(thesis, {"A": str(a), "B": str(b)})
    result = run_turtle(
        thesis, md, TurtleCoefficients.model_validate(coefficients_doc(costs={})), str(bench)
    )
    equity = equity_frame(result)
    assert "shares@A" in equity.columns and "add_level@B" in equity.columns
    sections = report_sections(result)
    assert sections.simulation["n_targets"] == 2
    cell = sections.cells[0]
    assert set(cell["by_target"]) == {"A", "B"}
    assert cell["portfolio"]["metrics"]["start_equity"] == 100_000.0
    assert cell["by_target"]["A"]["metrics"]["start_equity"] == 50_000.0
    paid = cell["portfolio"]["costs_paid"]
    for bucket in ("commission", "slippage", "shock", "impact", "total"):
        assert paid[bucket] == pytest.approx(
            sum(cell["by_target"][t]["end_state"]["costs_paid"][bucket] for t in ("A", "B"))
        )
    assert paid["total"] > 0 and equity["commission_cum"].iloc[-1] == pytest.approx(
        paid["commission"]
    )
