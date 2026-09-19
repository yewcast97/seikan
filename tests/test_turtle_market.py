"""The simulation's data admission: quantization, the benchmark clock, the volume the impact law
needs, and every refusal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seikan import _turtle
from seikan.dataio import DataError
from seikan.dsl.schema import Thesis
from seikan.turtle import TurtleCoefficients, preflight
from tests._helpers import load
from tests._turtle_helpers import (
    coefficients_doc,
    first_true_above,
    flat_rows,
    frictionless,
    thesis_doc,
    worked_example_rows,
    write_bars,
)

COEF = TurtleCoefficients.model_validate(coefficients_doc())
IMPACT = TurtleCoefficients.model_validate(
    coefficients_doc(costs=frictionless(impact={"coefficient": 0.5, "adv_window": 20}))
)


def _md(tmp_path, rows=None, volume=1000.0, **params):
    px = write_bars(tmp_path / "px.csv", rows or worked_example_rows(), volume=volume)
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
    assert data.index[0] == pd.Timestamp("2020-01-01") and len(data.index) == 31
    roles = [f["role"] for f in data.data_report["files"]]
    assert roles == ["target:PX", "benchmark"] and data.data_report["ok"]
    assert data.benchmark_path == str(bench)
    # Volume is carried only when the cost model reads it.
    assert data.targets["PX"].volume is None
    with_impact = preflight(md, str(bench), IMPACT)
    assert with_impact.targets["PX"].volume is not None
    assert set(with_impact.targets["PX"].volume.tolist()) == {1000.0}
    assert with_impact.benchmark.volume is None


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


def test_impact_needs_a_volume_column(tmp_path):
    _thesis, md = _md(tmp_path, volume=None)
    bench = write_bars(tmp_path / "idx.csv", worked_example_rows())
    assert preflight(md, str(bench), COEF).targets["PX"].volume is None  # unread: admitted
    with pytest.raises(DataError) as info:
        preflight(md, str(bench), IMPACT)
    assert _codes(info.value) == ["spec_data_mismatch"]
    assert "volume" in info.value.report["errors"][0]["message"]


def test_impact_refuses_missing_and_non_positive_volume(tmp_path):
    rows = worked_example_rows()
    px = write_bars(tmp_path / "px.csv", rows)
    text = px.read_text(encoding="utf-8").splitlines()
    hole = text[5].split(",")
    hole[5] = ""  # one volume cell blank
    text[5] = ",".join(hole)
    zero = text[7].split(",")
    zero[5] = "0"  # one zero-volume bar
    text[7] = ",".join(zero)
    px.write_text("\n".join(text) + "\n", encoding="utf-8")
    thesis = Thesis.model_validate(thesis_doc(["PX"], first_true_above(50.0)))
    md = load(thesis, {"PX": str(px)})
    bench = write_bars(tmp_path / "idx.csv", rows)
    assert preflight(md, str(bench), COEF).data_report["ok"]  # prices are whole: admitted
    with pytest.raises(DataError) as info:
        preflight(md, str(bench), IMPACT)
    assert _codes(info.value) == ["nan_fraction", "integrity"]
    assert all(e["column"] == "PX" for e in info.value.report["errors"])
