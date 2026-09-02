"""The entry-listing path: ``list_entries`` (+ its two per-bar frames and their CSV writers,
``write_root_series_csv`` / ``write_entry_flags_csv``). It shares the validation DSL + compiler but
evaluates only the entry firing mask, reporting every firing bar's timestamp in ascending order and
assembling the per-bar evidence — no forward returns, no statistics, no claim that the provided
series is current. Also covers ``render_series``, the deterministic expression labels the value
columns use.

The per-bar evidence lives in TWO frames, each nominated by its own CLI flag: ``root_series`` (the
values the entry tree thresholds on — the WHY a bar did or did not fire, written by ``seikan run
--root-series-out``) and ``entry_flags`` (the 0/1 mask per combo × target — the WHETHER, written by
``--entry-flags-out``; without that flag the mask would reach a CLI caller only in observation
shape, as trades). Keeping them apart holds the reserved-name set to ``datetime`` alone, so the
collision tests below pin that an external feed named ``entry`` KEEPS its name among the values:
with no flag column in that namespace there is nothing for it to shadow — and a same-named column
ACROSS the two files is a different thing, a value there and a 0/1 decision here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from seikan import dataio
from seikan.api import compile_thesis, list_entries
from seikan.compiler import vectorize as vz
from seikan.dsl.schema import (
    EMA,
    BarsSinceExtremum,
    BinaryOp,
    Calendar,
    Change,
    Constant,
    DaysSince,
    Drawdown,
    External,
    Field,
    Percentile,
    RollingAgg,
    RollingCorr,
    Runup,
    Shift,
    Thesis,
    UnaryOp,
    ZScore,
    render_series,
)
from seikan.serialize import write_entry_flags_csv, write_root_series_csv
from tests._helpers import bars as _bars
from tests._helpers import lt as _lt

from ._helpers import load


def _thesis(entry, **params) -> Thesis:
    """The single-target thesis these tests list over. It NAMES one series, ``target`` — the file
    behind that name is the invocation's business, supplied at load time."""
    params = {
        "horizon": 1,
        **params,
    }  # horizon is irrelevant to the firing mask, but the DSL needs it
    return Thesis.model_validate(
        {"name": "listing", "data": {"targets": ["target"]}, "entry": entry, "params": params}
    )


def _index(closes: list[float]) -> pd.DatetimeIndex:
    return pd.date_range("2021-01-01", periods=len(closes), freq="1D")


def _list(path: Path, entry, **params):
    """Declare-bind-and-list — the three-step call the API requires, collapsed for these tests,
    which exercise listing semantics, not loading."""
    thesis = _thesis(entry, **params)
    return list_entries(thesis, load(thesis, {"target": str(path)}))


def test_list_entries_reports_all_firing_timestamps_ascending(tmp_path):
    # close<100 fires at exactly bars 1 and 3 — every firing is listed, in index order.
    closes = [110.0, 95.0, 120.0, 90.0, 130.0]
    path = _bars(tmp_path / "fires.csv", closes)
    report = _list(path, _lt(100.0))
    assert len(report.entries) == 1
    row = report.entries[0]
    assert row["target"] == "target"
    assert row.keys() == {"target", "timestamps"}  # no sweep, no horizon, no exit
    idx = _index(closes)
    assert row["timestamps"] == [idx[1], idx[3]]


def test_list_entries_includes_last_bar_firing(tmp_path):
    # A last-bar firing has no t+1 to fill at, so it is not a backtest observation — but it IS a
    # signal firing, and the listing reports raw firings, not observations. That is not a
    # library-only channel: `seikan run --entry-flags-out` writes the same mask, and it is the ONE
    # CLI output this bar can reach — nothing in observation shape (the trades CSV, any coverage
    # ledger) can represent a firing that anchors no observation.
    closes = [110.0, 120.0, 95.0]
    path = _bars(tmp_path / "asof.csv", closes)
    report = _list(path, _lt(100.0))
    assert report.entries[0]["timestamps"] == [_index(closes)[-1]]


def test_list_entries_never_fired_row_retained(tmp_path):
    # The condition never fires → the (combo × target) row is still present, with an empty list.
    path = _bars(tmp_path / "nofire.csv", [110.0, 120.0, 130.0])
    report = _list(path, _lt(100.0))
    assert len(report.entries) == 1
    assert report.entries[0]["timestamps"] == []


def test_list_entries_series_end_is_last_provided_bar(tmp_path):
    # series_end describes where the PROVIDED joined index stops — no freshness claim. Spelled as
    # the explicit call (bind the named series to a file, load once, hand the MarketData in) rather
    # than through _list, so the required-parameter contract is exercised visibly in this file.
    path = _bars(tmp_path / "end.csv", [100, 101, 102, 99])
    thesis = _thesis(_lt(100.0))
    md = load(thesis, {"target": str(path)})
    report = list_entries(thesis, md)
    assert report.series_end == md.index[-1]


def test_list_entries_excludes_horizon_axis(tmp_path):
    # The horizon does not affect the firing mask, so even a swept horizon never appears in the
    # rows.
    closes = (100 + np.random.RandomState(4).randn(60).cumsum()).tolist()
    path = _bars(tmp_path / "hz.csv", closes)
    ema = {"type": "ema", "window": 20, "input": {"type": "field", "column": "close"}}
    entry = {
        "type": "threshold",
        "left": {"type": "field", "column": "close"},
        "op": "<",
        "right": ema,
    }
    report = _list(path, entry, horizon=[1, 5, 20])
    assert len(report.entries) == 1  # scalar entry, single target → one row
    assert all("horizon" not in r for r in report.entries)


def test_list_entries_multi_target_swept_window(tmp_path):
    files = {}
    for sym, seed in [("AAA", 1), ("BBB", 2)]:
        rng = np.random.RandomState(seed)
        idx = pd.date_range("2020-01-01", periods=400, freq="1D")
        close = 100 + rng.randn(400).cumsum()
        pd.DataFrame(
            {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0},
            index=idx,
        ).to_csv(tmp_path / f"{sym}.csv")
        files[sym] = str(tmp_path / f"{sym}.csv")
    ema_list = {"type": "ema", "window": [10, 20], "input": {"type": "field", "column": "close"}}
    thesis = Thesis.model_validate(
        {
            "name": "listing_multi",
            "data": {"targets": list(files)},
            "entry": {
                "type": "threshold",
                "left": {"type": "field", "column": "close"},
                "op": "<",
                "right": ema_list,
            },
            "params": {"horizon": 10},
        }
    )
    report = list_entries(thesis, load(thesis, files))
    # one row per (ema_window combo × target) = 2 × 2
    assert len(report.entries) == 4
    assert {r["target"] for r in report.entries} == {"AAA", "BBB"}
    for r in report.entries:
        assert {"ema_window", "target", "timestamps"} == r.keys()  # signal axis only, never horizon
        assert r["timestamps"] == sorted(r["timestamps"])


def test_list_entries_swept_constant_axis(tmp_path):
    closes = [110.0] + [95.0] * 20
    path = _bars(tmp_path / "cs.csv", closes)
    entry = {
        "type": "threshold",
        "left": {"type": "field", "column": "close"},
        "op": "<",
        "right": {"type": "constant", "value": [100.0, 150.0, 200.0], "name": "buy_below"},
    }
    report = _list(path, entry, horizon=3)
    assert len(report.entries) == 3  # 3 swept thresholds × 1 target
    assert {r["buy_below"] for r in report.entries} == {100.0, 150.0, 200.0}
    by_cut = {r["buy_below"]: r["timestamps"] for r in report.entries}
    idx = _index(closes)
    assert by_cut[100.0] == list(idx[1:])  # bar 0 (110) misses the 100 cutoff
    assert by_cut[150.0] == list(idx) and by_cut[200.0] == list(idx)


def test_list_entries_rolling_window_axis(tmp_path):
    closes = (100 + np.random.RandomState(6).randn(200).cumsum()).tolist()
    path = _bars(tmp_path / "rw.csv", closes)
    entry = {
        "type": "rolling",
        "window": [3, 5],
        "agg": "any",
        "condition": {
            "type": "threshold",
            "left": {
                "type": "percentile",
                "window": [14, 21],
                "input": {"type": "field", "column": "close"},
            },
            "op": "<",
            "right": {"type": "constant", "value": 0.9},
        },
    }
    report = _list(path, entry, horizon=4)
    # 2 percentile windows × 2 rolling windows × 1 target
    assert len(report.entries) == 4
    assert all({"percentile_window", "rolling_window"} <= r.keys() for r in report.entries)
    assert {r["rolling_window"] for r in report.entries} == {3, 5}


def test_list_entries_runup_and_bars_since_extremum(tmp_path):
    # The left-side primitives must compile through the listing path.
    closes = [100.0, 80.0, 80.0, 80.0, 88.0, 90.0, 95.0]
    path = _bars(tmp_path / "left.csv", closes)
    entry = {
        "type": "and",
        "conditions": [
            {
                "type": "threshold",
                "op": ">=",
                "left": {"type": "runup"},
                "right": {"type": "constant", "value": 0.05},
            },
            {
                "type": "threshold",
                "op": ">=",
                "left": {"type": "bars_since_extremum", "extremum": "min"},
                "right": {"type": "constant", "value": 2},
            },
        ],
    }
    report = _list(path, entry)
    assert len(report.entries) == 1
    stamps = report.entries[0]["timestamps"]
    idx = _index(closes)
    # Bars 0-3 sit at/below the trough (runup < 0.05 → cannot fire); the last bar
    # (95: runup = 95/80 − 1 = 0.1875, well past the min) must be listed.
    assert idx[-1] in stamps
    assert not set(stamps) & set(idx[:4])
    assert stamps == sorted(stamps)


# ---- render_series (the value-column labels) ------------------------------------------------


def test_render_series_rule_table():
    close = Field()
    iv30 = External(name="iv30")
    assert render_series(close) == "close"
    assert render_series(Field(column="high")) == "high"
    assert render_series(Constant(value=100.0)) == "100"
    assert render_series(Constant(value=-0.4)) == "-0.4"
    assert render_series(iv30) == "iv30"
    assert render_series(Calendar(field="month")) == "calendar(month)"
    assert render_series(DaysSince(name="earnings")) == "days_since(earnings)"
    assert render_series(EMA(input=close, window=20)) == "ema(close,20)"
    assert (
        render_series(ZScore(input=EMA(input=close, window=20), window=50))
        == "zscore(ema(close,20),50)"
    )
    assert render_series(ZScore(input=close, window=50, mean_type="ema")) == "zscore(close,50,ema)"
    assert render_series(Percentile(input=iv30, window=80)) == "percentile(iv30,80)"
    assert (
        render_series(RollingAgg(input=close, window=20, agg="std")) == "rolling_agg(close,20,std)"
    )
    assert render_series(RollingAgg(input=close, agg="max")) == "rolling_agg(close,max)"
    assert render_series(Drawdown()) == "drawdown(close)"
    assert render_series(Drawdown(window=8)) == "drawdown(close,8)"
    assert render_series(Runup()) == "runup(close)"
    assert render_series(BarsSinceExtremum(extremum="min")) == "bars_since_extremum(close,min)"
    assert (
        render_series(BarsSinceExtremum(extremum="max", window=10))
        == "bars_since_extremum(close,max,10)"
    )
    assert render_series(Change(input=close, periods=5)) == "change(close,5)"
    assert render_series(Change(input=iv30, kind="diff")) == "change(iv30,1,diff)"
    assert render_series(Shift(input=Field(column="high"), periods=2)) == "shift(high,2)"
    assert (
        render_series(
            RollingCorr(left=Change(input=close), right=Change(input=iv30, kind="diff"), window=20)
        )
        == "rolling_corr(change(close,1),change(iv30,1,diff),20)"
    )
    assert (
        render_series(BinaryOp(left=close, right=EMA(input=close, window=20), op="/"))
        == "(close/ema(close,20))"
    )
    assert (
        render_series(UnaryOp(input=ZScore(input=close, window=50), op="abs"))
        == "abs(zscore(close,50))"
    )


# ---- the root_series / entry_flags frames -----------------------------------------------------


def test_signals_single_combo_single_target(tmp_path):
    closes = [110.0, 95.0, 120.0, 90.0, 130.0]
    path = _bars(tmp_path / "sig.csv", closes)
    report = _list(path, _lt(100.0))
    values, flags = report.root_series, report.entry_flags
    assert list(values.columns) == ["close"]  # constant cutoff excluded; no combo/target tags
    assert list(flags.columns) == ["entry"]
    assert values.index.name == flags.index.name == "datetime"
    assert list(values.index) == list(flags.index) == list(_index(closes))
    assert values["close"].tolist() == closes
    assert set(flags["entry"].unique()) <= {0, 1}
    # flags and the timestamp rows come from the same mask — bit-identical
    assert flags.index[flags["entry"] == 1].tolist() == report.entries[0]["timestamps"]


def test_signals_constant_only_thresholds_have_no_value_columns(tmp_path):
    path = _bars(tmp_path / "const.csv", [100.0, 101.0, 102.0])
    entry = {
        "type": "threshold",
        "left": {"type": "constant", "value": 1.0},
        "op": "<",
        "right": {"type": "constant", "value": 2.0},
    }
    report = _list(path, entry)
    # No threshold operand is a series, so there is nothing to show as a decision input — but the
    # condition still decides on every bar, and the flag frame says so.
    assert list(report.root_series.columns) == []
    assert list(report.entry_flags.columns) == ["entry"]
    assert report.entry_flags["entry"].tolist() == [1, 1, 1]


def test_signals_dedupe_and_swept_constant_axis(tmp_path):
    # 'close' appears as the left operand of BOTH thresholds and across all three swept-constant
    # combos — it is written exactly once; each combo keeps its own labelled entry column.
    closes = [110.0] + [95.0] * 20
    path = _bars(tmp_path / "cs2.csv", closes)
    entry = {
        "type": "and",
        "conditions": [
            {
                "type": "threshold",
                "left": {"type": "field", "column": "close"},
                "op": "<",
                "right": {"type": "constant", "value": [100.0, 150.0, 200.0], "name": "buy_below"},
            },
            {
                "type": "threshold",
                "left": {"type": "field", "column": "close"},
                "op": ">",
                "right": {"type": "constant", "value": 10.0},
            },
        ],
    }
    report = _list(path, entry, horizon=3)
    assert list(report.root_series.columns) == ["close"]
    assert list(report.entry_flags.columns) == [
        "entry[buy_below=100]",
        "entry[buy_below=150]",
        "entry[buy_below=200]",
    ]
    by_cut = {r["buy_below"]: r["timestamps"] for r in report.entries}
    for cut in (100.0, 150.0, 200.0):
        col = report.entry_flags[f"entry[buy_below={int(cut)}]"]
        assert report.entry_flags.index[col == 1].tolist() == by_cut[cut]


def test_signals_swept_window_column_per_value(tmp_path):
    closes = (100 + np.random.RandomState(9).randn(80).cumsum()).tolist()
    path = _bars(tmp_path / "sw.csv", closes)
    ema_list = {"type": "ema", "window": [10, 20], "input": {"type": "field", "column": "close"}}
    entry = {
        "type": "threshold",
        "left": {"type": "field", "column": "close"},
        "op": "<",
        "right": ema_list,
    }
    report = _list(path, entry)
    # value columns in first-appearance order (combo-major); the flags sit in their own frame, one
    # per combo, in the same iteration order
    assert list(report.root_series.columns) == ["close", "ema(close,10)", "ema(close,20)"]
    assert list(report.entry_flags.columns) == ["entry[ema_window=10]", "entry[ema_window=20]"]


def test_signals_multi_target_suffix_and_flags(tmp_path):
    files = {}
    for sym, seed in [("AAA", 1), ("BBB", 2)]:
        rng = np.random.RandomState(seed)
        idx = pd.date_range("2020-01-01", periods=400, freq="1D")
        close = 100 + rng.randn(400).cumsum()
        pd.DataFrame(
            {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0},
            index=idx,
        ).rename_axis("datetime").to_csv(tmp_path / f"{sym}.csv")
        files[sym] = str(tmp_path / f"{sym}.csv")
    ema_list = {"type": "ema", "window": [10, 20], "input": {"type": "field", "column": "close"}}
    thesis = Thesis.model_validate(
        {
            "name": "signals_multi",
            "data": {"targets": list(files)},
            "entry": {
                "type": "threshold",
                "left": {"type": "field", "column": "close"},
                "op": "<",
                "right": ema_list,
            },
            "params": {"horizon": 10},
        }
    )
    report = list_entries(thesis, load(thesis, files))
    assert list(report.root_series.columns) == [
        "close@AAA",
        "close@BBB",
        "ema(close,10)@AAA",
        "ema(close,10)@BBB",
        "ema(close,20)@AAA",
        "ema(close,20)@BBB",
    ]
    assert list(report.entry_flags.columns) == [
        "entry[ema_window=10]@AAA",
        "entry[ema_window=10]@BBB",
        "entry[ema_window=20]@AAA",
        "entry[ema_window=20]@BBB",
    ]
    for r in report.entries:  # every (combo × target) flag column agrees with its timestamp row
        col = report.entry_flags[f"entry[ema_window={r['ema_window']}]@{r['target']}"]
        assert report.entry_flags.index[col == 1].tolist() == r["timestamps"]


def test_signals_collision_fallback(tmp_path):
    # Two external feeds, named 'close' and 'entry'. Only 'close' collides — with the FIELD label
    # already claimed by the first threshold — and takes the '#2' suffix. 'entry' does NOT: the
    # canonical flag name it would otherwise yield to lives in a SEPARATE frame, so the value
    # namespace reserves nothing but the 'datetime' index label and the feed keeps the name the
    # caller gave it.
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    path = _bars(tmp_path / "cf.csv", closes)
    idx = _index(closes)
    for name in ("close", "entry"):
        pd.DataFrame({"v": [1.0] * len(idx)}, index=idx).rename_axis("datetime").to_csv(
            tmp_path / f"feed_{name}.csv"
        )
    entry = {
        "type": "and",
        "conditions": [
            {
                "type": "threshold",
                "left": {"type": "field", "column": "close"},
                "op": ">",
                "right": {"type": "constant", "value": 0.0},
            },
            {
                "type": "threshold",
                "left": {"type": "external", "name": "close"},
                "op": ">",
                "right": {"type": "constant", "value": 0.0},
            },
            {
                "type": "threshold",
                "left": {"type": "external", "name": "entry"},
                "op": ">",
                "right": {"type": "constant", "value": 0.0},
            },
        ],
    }
    thesis = Thesis.model_validate(
        {
            "name": "collide",
            "data": {"targets": ["target"], "external": {"close": {}, "entry": {}}},
            "entry": entry,
            "params": {"horizon": 1},
        }
    )
    files = {
        "target": str(path),
        "close": str(tmp_path / "feed_close.csv"),
        "entry": str(tmp_path / "feed_entry.csv"),
    }
    report = list_entries(thesis, load(thesis, files))
    assert list(report.root_series.columns) == ["close", "close#2", "entry"]
    assert report.root_series["close"].tolist() == closes  # the field, under its own label
    assert (report.root_series["close#2"] == 1.0).all()  # the shadowed feed, under the suffix
    assert (report.root_series["entry"] == 1.0).all()  # feed values, NOT a 0/1 flag
    assert list(report.entry_flags.columns) == ["entry"]  # the flag, alone in its own namespace
    assert set(report.entry_flags["entry"].unique()) <= {0, 1}


def test_write_root_series_csv_roundtrip_strict(tmp_path):
    closes = (100 + np.random.RandomState(3).randn(60).cumsum()).tolist()
    path = _bars(tmp_path / "rt.csv", closes)
    entry = {
        "type": "threshold",
        "left": {"type": "percentile", "window": 14, "input": {"type": "field", "column": "close"}},
        "op": "<",
        "right": {"type": "constant", "value": 0.5},
    }
    report = _list(path, entry)
    out = tmp_path / "root_series.csv"
    rows = write_root_series_csv(report.root_series, str(out))
    assert rows == len(report.root_series) == len(closes)
    frame, rep = dataio.read_strict_csv(str(out))
    assert rep.ok and rep.shape == "series"
    # Value columns only — the entry flag rides its OWN CSV (write_entry_flags_csv), never this one.
    assert list(frame.columns) == ["percentile(close,14)"]  # comma-bearing header survives
    values = report.root_series["percentile(close,14)"]
    assert frame["percentile(close,14)"].isna().tolist() == values.isna().tolist()
    pd.testing.assert_series_equal(
        frame["percentile(close,14)"], values.astype(float), check_names=False
    )
    # always overwrites: a second write leaves one identical file, not an appended one
    first = out.read_bytes()
    write_root_series_csv(report.root_series, str(out))
    assert out.read_bytes() == first


def test_write_root_series_csv_constant_only_thesis_writes_datetime_only(tmp_path):
    """A thesis whose every threshold operand is a bare constant has no root series at all. The
    writer does not gate on that: it emits the datetime-only file and returns the row count, because
    refusing would hide exactly the degenerate thesis shape a caller most needs to see. No roundtrip
    is claimed — the strict reader wants at least one value column."""
    path = _bars(tmp_path / "const_rt.csv", [100.0, 101.0, 102.0])
    entry = {
        "type": "threshold",
        "left": {"type": "constant", "value": 1.0},
        "op": "<",
        "right": {"type": "constant", "value": 2.0},
    }
    report = _list(path, entry)
    out = tmp_path / "empty_root_series.csv"
    assert write_root_series_csv(report.root_series, str(out)) == 3
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "datetime"
    assert len(lines) == 4  # header + one row per bar, no value columns


def test_write_entry_flags_csv_roundtrip_strict(tmp_path):
    """The DECISION-side twin of the root-series roundtrip — and a STRICTLY STRONGER promise.

    ``write_root_series_csv`` only re-reads 'in the ordinary case': an all-NaN value column (a
    window longer than the data) or a constant-only thesis (a datetime-only file) leaves bytes the
    strict reader refuses, which is why the two tests above have to carve the degenerate shape out.
    Neither shape can arise here. The flag frame is int 0/1 on EVERY bar — warmup and undecidable
    bars are a 0, not a hole, so no column can be all-NaN — and it always carries at least one
    column, since a thesis declares at least one combo and at least one target. So the strict
    re-read is an UNCONDITIONAL promise for this writer, and the root-series caveats must not be
    copied onto it.
    """
    closes = (100 + np.random.RandomState(3).randn(60).cumsum()).tolist()
    path = _bars(tmp_path / "flags_rt.csv", closes)
    entry = {
        "type": "threshold",
        "left": {
            "type": "percentile",
            "window": [14, 21],
            "input": {"type": "field", "column": "close"},
        },
        "op": "<",
        "right": {"type": "constant", "value": 0.5},
    }
    report = _list(path, entry)
    flags = report.entry_flags
    assert list(flags.columns) == ["entry[percentile_window=14]", "entry[percentile_window=21]"]
    assert all(pd.api.types.is_integer_dtype(flags[c]) for c in flags.columns)

    out = tmp_path / "entry_flags.csv"
    rows = write_entry_flags_csv(flags, str(out))
    assert rows == len(flags) == len(closes)

    frame, rep = dataio.read_strict_csv(str(out))
    assert rep.ok and rep.shape == "series"
    assert list(frame.columns) == list(flags.columns)  # bracketed combo headers survive verbatim
    for col in frame.columns:
        assert not frame[col].isna().any()  # the 13 warmup bars are 0s, not empty cells
        assert set(frame[col].unique()) <= {0.0, 1.0}  # the reader forces float; the VALUES are 0/1
        assert (frame[col] == 1).any() and (frame[col] == 0).any()  # both states exercised
    pd.testing.assert_frame_equal(frame, flags.astype(float))
    # and the file IS the mask: each column's flagged bars are exactly that combo's listed firings
    for r in report.entries:
        col = frame[f"entry[percentile_window={r['percentile_window']}]"]
        assert frame.index[col == 1].tolist() == r["timestamps"]

    # always overwrites: a second write leaves one identical file, not an appended one ...
    first = out.read_bytes()
    assert write_entry_flags_csv(flags, str(out)) == rows
    assert out.read_bytes() == first
    # ... and a pre-existing file is REPLACED whole — never merged into, never grown
    out.write_text("stale,junk\n" * 500, encoding="utf-8")
    assert write_entry_flags_csv(flags, str(out)) == rows
    assert out.read_bytes() == first


def test_listing_after_backtest_rebuilds_nothing(tmp_path, monkeypatch):
    # ONE evaluation memo per run, riding md: after compile_thesis over a given MarketData,
    # list_entries over the SAME md performs no new series/condition builds — every scalarized
    # combo tree and every root-series operand hits the memo the backtest populated (signal's
    # final value&init&defined AND re-runs, which is not a build). Counted at the private
    # workers, because the public wrappers also fire on memo hits; the swept window matters,
    # since it exercises per-combo scalarized-tree key identity across the two paths.
    closes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    path = _bars(tmp_path / "shared.csv", closes)
    entry = {
        "type": "threshold",
        "left": {"type": "ema", "input": {"type": "field", "column": "close"}, "window": [2, 3]},
        "op": ">",
        "right": {"type": "constant", "value": 3.0},
    }
    thesis = _thesis(entry, horizon=2)
    md = load(thesis, {"target": str(path)})

    calls = {"series": 0, "condition": 0}

    def counting(name, real):
        def wrapper(node, md_):
            calls[name] += 1
            return real(node, md_)

        return wrapper

    monkeypatch.setattr(vz, "_build_series", counting("series", vz._build_series))
    monkeypatch.setattr(vz, "_build_condition", counting("condition", vz._build_condition))

    compile_thesis(thesis, md)
    assert (
        md.series_memo and calls["series"] > 0 and calls["condition"] > 0
    )  # the backtest built into THIS md's memo
    snapshot, keys = dict(calls), (set(md.series_memo), set(md.condition_memo), set(md.array_memo))

    listing = list_entries(thesis, md)
    assert calls == snapshot  # zero rebuilds on the listing path
    assert (
        set(md.series_memo),
        set(md.condition_memo),
        set(md.array_memo),
    ) == keys  # and it needed nothing the backtest had not already built
    assert len(listing.entries) == 2  # sanity: both swept combos listed
    assert any(r["timestamps"] for r in listing.entries)  # and the shared mask actually fires


def test_api_boundary_revalidates_a_mutated_thesis(tmp_path):
    # The models are frozen, but freezing is shallow: nested lists/dicts stay mutable. The public
    # API revalidates a private copy at entry, so a post-validation poisoning of the targets list
    # refuses instead of driving an engine run no validator ever saw.
    path = _bars(tmp_path / "px.csv", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    thesis = _thesis(_lt(100.0))
    md = load(thesis, {"target": path})
    thesis.data.targets.append("target")  # a duplicate name — refused by DataSpec validation
    with pytest.raises(ValidationError):
        compile_thesis(thesis, md)
    with pytest.raises(ValidationError):
        list_entries(thesis, md)


def test_repeat_compile_records_barely_sufficient_warning_once(tmp_path):
    # The warning amends md.report IN PLACE (shared, describing the same load); repeat compiles
    # over one md must record it once, not once per call.
    path = _bars(tmp_path / "px.csv", [float(100 + i) for i in range(8)])
    thesis = _thesis(_lt(1000.0), horizon=3)  # 8 bars, h=3: sufficient but under the comfort floor
    md = load(thesis, {"target": path})
    compile_thesis(thesis, md)
    compile_thesis(thesis, md)
    warnings = [w for w in md.report["join"]["warnings"] if w["code"] == "barely_sufficient"]
    assert len(warnings) == 1
