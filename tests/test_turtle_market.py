"""The simulation's data admission: quantization, the benchmark clock, every refusal, and the
nautilus_trader objects built from the admitted arrays."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seikan import _turtle
from seikan.dataio import DataError
from seikan.dsl.schema import Thesis
from seikan.turtle import TurtleCoefficients, preflight
from seikan.turtle import market as market_module
from tests._helpers import load
from tests._turtle_helpers import (
    coefficients_doc,
    first_true_above,
    flat_rows,
    thesis_doc,
    worked_example_rows,
    write_bars,
)

COEF = TurtleCoefficients.model_validate(coefficients_doc())


def _md(tmp_path, rows=None, **params):
    px = write_bars(tmp_path / "px.csv", rows or worked_example_rows())
    thesis = Thesis.model_validate(thesis_doc(["PX"], first_true_above(50.0), **params))
    return thesis, load(thesis, {"PX": str(px)})


def _codes(exc: DataError) -> list[str]:
    return [e["code"] for e in exc.report["errors"]]


def test_preflight_quantizes_and_extends_the_data_report(tmp_path):
    _thesis, md = _md(tmp_path)
    bench = write_bars(
        tmp_path / "idx.csv",
        [
            (o * 1.234_567_89, h * 1.234_567_89, lo * 1.234_567_89, c * 1.234_567_89)
            for o, h, lo, c in worked_example_rows()
        ],
    )
    data = preflight(md, str(bench), COEF)
    assert list(data.targets) == ["PX"] and data.precision == 4
    raw = md.close["PX"].to_numpy()
    assert np.array_equal(data.targets["PX"].close, [_turtle.quantize(float(v), 4) for v in raw])
    assert data.benchmark.open[0] == _turtle.quantize(49.5 * 1.234_567_89, 4)
    assert data.ts_ns[1] - data.ts_ns[0] == 86_400 * 10**9
    assert data.ts_ns[0] == pd.Timestamp("2020-01-01").value
    roles = [f["role"] for f in data.data_report["files"]]
    assert roles == ["target:PX", "benchmark"] and data.data_report["ok"]
    assert data.benchmark_path == str(bench)


def test_a_benchmark_the_thesis_declared_is_not_reported_twice(tmp_path):
    px = write_bars(tmp_path / "px.csv", worked_example_rows())
    bench = write_bars(tmp_path / "idx.csv", worked_example_rows())
    thesis = Thesis.model_validate(thesis_doc(["PX"], first_true_above(50.0), benchmark="market"))
    md = load(thesis, {"PX": str(px), "benchmark": str(bench)})
    data = preflight(md, str(bench), COEF)
    assert [f["role"] for f in data.data_report["files"]] == ["target:PX", "benchmark"]


def test_a_benchmark_missing_target_bars_refuses(tmp_path):
    _thesis, md = _md(tmp_path)
    bench = write_bars(tmp_path / "idx.csv", worked_example_rows()[:-3])
    with pytest.raises(DataError) as info:
        preflight(md, str(bench), COEF)
    assert _codes(info.value) == ["benchmark_coverage"]
    assert "3 of the 31" in info.value.report["errors"][0]["message"]
    assert not info.value.report["ok"]
    assert [f["role"] for f in info.value.report["files"]] == ["target:PX", "benchmark"]


def test_an_unreadable_benchmark_refuses_with_its_file_report(tmp_path):
    _thesis, md = _md(tmp_path)
    with pytest.raises(DataError) as info:
        preflight(md, str(tmp_path / "missing.csv"), COEF)
    assert _codes(info.value) == ["benchmark_coverage"]
    entry = info.value.report["files"][-1]
    assert entry["role"] == "benchmark" and not entry["ok"]
    assert entry["errors"][0]["code"] == "file_missing"


def test_series_shaped_targets_refuse(tmp_path):
    idx = pd.date_range("2020-01-01", periods=40, freq="1D")
    s = pd.DataFrame({"value": np.linspace(10, 20, 40)}, index=idx)
    s.index.name = "datetime"
    s.to_csv(tmp_path / "s.csv")
    thesis = Thesis.model_validate(thesis_doc(["S"], first_true_above(15.0)))
    md = load(thesis, {"S": str(tmp_path / "s.csv")})
    bench = write_bars(tmp_path / "idx.csv", flat_rows(40, 100.0, 1.0))
    with pytest.raises(DataError) as info:
        preflight(md, str(bench), COEF)
    assert _codes(info.value) == ["spec_data_mismatch"]


def test_a_hole_in_a_target_refuses(tmp_path):
    rows = worked_example_rows()
    px = write_bars(tmp_path / "px.csv", rows)
    text = px.read_text(encoding="utf-8").splitlines()
    cells = text[5].split(",")
    cells[4] = ""  # the close of one bar
    text[5] = ",".join(cells)
    px.write_text("\n".join(text) + "\n", encoding="utf-8")
    thesis = Thesis.model_validate(thesis_doc(["PX"], first_true_above(50.0)))
    md = load(thesis, {"PX": str(px)})
    bench = write_bars(tmp_path / "idx.csv", rows)
    with pytest.raises(DataError) as info:
        preflight(md, str(bench), COEF)
    assert _codes(info.value) == ["nan_fraction"]
    assert info.value.report["errors"][0]["column"] == "PX"


def test_too_few_bars_refuse(tmp_path):
    rows = worked_example_rows()[:15]
    _thesis, md = _md(tmp_path, rows)
    bench = write_bars(tmp_path / "idx.csv", rows)
    with pytest.raises(DataError) as info:
        preflight(md, str(bench), COEF)
    assert _codes(info.value) == ["insufficient_common_index"]
    assert "at least 21" in info.value.report["errors"][0]["message"]


def test_nautilus_objects_are_built_on_the_grid(tmp_path):
    _thesis, md = _md(tmp_path)
    bench = write_bars(tmp_path / "idx.csv", worked_example_rows())
    data = preflight(md, str(bench), COEF)
    instrument = market_module.instrument_for(0, "USD", 4)
    assert str(instrument.id) == "T0.SIM" and instrument.price_precision == 4
    assert str(instrument.price_increment) == "0.0001" and instrument.size_precision == 0
    assert str(market_module.bar_type_for(instrument)) == "T0.SIM-1-DAY-LAST-EXTERNAL"
    items = market_module.venue_data(instrument, data.targets["PX"], data.ts_ns)
    assert len(items) == 2 * len(data.index)
    print_, bar = items[0], items[1]
    assert print_.ts_init == bar.ts_init - market_module.OPEN_PRINT_OFFSET_NS
    assert float(print_.bid_price) == float(print_.ask_price) == float(bar.open) == 49.5
    assert int(print_.bid_size) == int(bar.volume) == market_module.UNLIMITED_LIQUIDITY
    assert [float(b.close) for b in items[1::2]][24:28] == [50.0, 51.0, 52.0, 52.5]
    assert market_module.instrument_symbol(3) == "T3"
