"""The documented public surface — ``import seikan`` and every name ``seikan.__all__`` promises —
exercised as a host would use it, so a name the facade drops or misroutes fails here rather
than in someone else's code."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import seikan
from tests._helpers import bars


def test_package_surface_runs_a_thesis_end_to_end(tmp_path):
    closes = (100 + np.random.RandomState(0).randn(300).cumsum()).tolist()
    px = bars(tmp_path / "px.csv", closes)
    doc = {
        "name": "surface",
        "data": {"targets": ["px"]},
        "entry": {
            "type": "threshold",
            "left": {"type": "field", "column": "close"},
            "op": "<",
            "right": {"type": "ema", "window": 10, "input": {"type": "field", "column": "close"}},
        },
        "params": {"horizon": [3, 8]},
    }
    thesis = seikan.Thesis.model_validate(doc)
    files = seikan.resolve_data_files(thesis, {"px": str(px)})
    assert isinstance(files, seikan.DataFiles)
    md = seikan.load_market_data(thesis.data, files)
    assert isinstance(md, seikan.MarketData)
    result = seikan.compile_thesis(thesis, md)
    assert isinstance(result, seikan.EventStudyResult)
    assert len(result.summary["cells"]) == 2 == result.summary["n_hypotheses_attempted"]
    listing = seikan.list_entries(thesis, md)
    assert isinstance(listing, seikan.EntryListReport)
    assert len(listing.entry_flags.columns) == 1
    payload = seikan.serialize_result(result)
    assert payload["name"] == "surface" and set(payload["summary"]) == set(result.summary)
    assert len(seikan.canonical_dsl_hash(doc)) == 64
    assert seikan.__version__ and set(seikan.__all__) >= {
        "compile_thesis",
        "list_entries",
        "canonical_dsl_hash",
        "load_market_data",
        "resolve_data_files",
    }


def test_list_entries_expands_shared_axes(tmp_path):
    closes = (100 + np.random.RandomState(1).randn(200).cumsum()).tolist()
    px = bars(tmp_path / "px.csv", closes)
    doc = {
        "name": "shared",
        "data": {"targets": ["px"]},
        "axes": {"N": [20, 60]},
        "entry": {
            "type": "threshold",
            "left": {"type": "ema", "window": {"axis": "N"}, "input": {"type": "field"}},
            "op": ">",
            "right": {
                "type": "rolling_agg",
                "window": {"axis": "N"},
                "agg": "mean",
                "input": {"type": "field"},
            },
        },
        "params": {"horizon": 5},
    }
    thesis = seikan.Thesis.model_validate(doc)
    md = seikan.load_market_data(thesis.data, seikan.resolve_data_files(thesis, {"px": str(px)}))
    listing = seikan.list_entries(thesis, md)
    assert list(listing.entry_flags.columns) == ["entry[N=20]", "entry[N=60]"]
    assert {row["N"] for row in listing.entries} == {20, 60}
    assert list(listing.root_series.columns) == [
        "ema(close,20)",
        "rolling_agg(close,20,mean)",
        "ema(close,60)",
        "rolling_agg(close,60,mean)",
    ]


def test_a_load_answers_one_data_declaration(tmp_path):
    # A MarketData holds feeds ALIGNED under the declaration it was loaded with. The review's
    # misuse: a lag-0 load handed to a thesis declaring a 1-day lag read the feed a bar early —
    # +100% where the declared lag measures 0%. Both entry points refuse a load of another
    # declaration; a thesis declaring the same data shares the load, whatever else it asks.
    px = bars(tmp_path / "px.csv", [100.0 + i for i in range(40)])
    feed = tmp_path / "f.csv"
    idx = pd.date_range("2021-01-01", periods=40, freq="1D", name="datetime")
    pd.DataFrame({"f": np.arange(40.0)}, index=idx).to_csv(feed)

    def thesis(lag, cut):
        return seikan.Thesis.model_validate(
            {
                "name": "lagged",
                "data": {"targets": ["px"], "external": {"f": {"lag": lag}}},
                "entry": {
                    "type": "threshold",
                    "left": {"type": "external", "name": "f"},
                    "op": ">",
                    "right": {"type": "constant", "value": cut},
                },
            }
        )

    loaded = thesis(0, 10.0)
    files = seikan.resolve_data_files(loaded, {"px": str(px), "f": str(feed)})
    md = seikan.load_market_data(loaded.data, files)
    late = thesis("1D", 10.0)
    for call in (seikan.compile_thesis, seikan.list_entries):
        with pytest.raises(ValueError, match="another data declaration"):
            call(late, md)
    same = thesis(0, 20.0)
    assert seikan.compile_thesis(same, md).summary["cells"]
    assert any(row["timestamps"] for row in seikan.list_entries(same, md).entries)
    # the load keeps its own copy of the declaration: a mutation of the thesis after the load
    # (its models are frozen only shallowly) cannot make the two agree
    loaded.data.external["f"] = late.data.external["f"]
    with pytest.raises(ValueError, match="another data declaration"):
        seikan.compile_thesis(loaded, md)
