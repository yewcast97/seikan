"""Load-contract tests for ``compiler/data.py`` (key binding, start/end, strict rejection, join,
reports)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seikan.dataio import DataError
from seikan.dsl.schema import Thesis

from ._helpers import files_for, load


def _ohlcv(path: Path, n: int = 10, start: str = "2021-01-01", *, closes=None) -> Path:
    idx = pd.date_range(start, periods=n, freq="1D")
    if closes is None:
        closes = np.linspace(100.0, 110.0, n)
    s = pd.Series(closes, index=idx, dtype=float)
    df = pd.DataFrame(
        {"open": s, "high": s + 1.0, "low": s - 1.0, "close": s, "volume": 1000.0},
        index=idx,
    )
    df.index.name = "datetime"
    df.to_csv(path)
    return path


def _thesis(targets: list[str] | None = None, *, benchmark: bool = False, **data) -> Thesis:
    """A thesis that NAMES the series a load needs, and asks nothing else of the data.

    Everything exercised here lives on the ``DataSpec``, but a load is bound to files through the
    whole thesis — ``data_keys()`` spans the targets, the feeds and the reserved benchmark slot —
    so these tests carry an entry condition that measures nothing and never fires anywhere in
    particular. It is here because a thesis without one is not a thesis.
    """
    # A declared feed nothing reads refuses at validation, so each declared feed is referenced
    # as a feature snapshot — these tests exercise LOADING, and a feature is the lightest read.
    features = {
        f"use_{name}": {"type": "external", "name": name} for name in data.get("external", {})
    }
    return Thesis.model_validate(
        {
            "name": "load",
            "data": {"targets": targets or ["target"], **data},
            "entry": {
                "type": "threshold",
                "left": {"type": "field", "column": "close"},
                "op": ">",
                "right": {"type": "constant", "value": 0.0},
            },
            "params": {
                "horizon": 1,
                **({"benchmark": "market"} if benchmark else {}),
                **({"features": features} if features else {}),
            },
        }
    )


def _codes(exc: DataError) -> set[str]:
    report = exc.report
    codes = {e["code"] for e in report.get("errors") or []}
    for f in report.get("files") or []:
        codes |= {e["code"] for e in f["errors"]}
    return codes


def test_mapping_must_answer_the_declared_keys_exactly(tmp_path):
    """The two halves of a load must agree before a byte is read.

    A missing key would load nothing where a series was declared — the run would measure a thesis
    it does not have — and an unknown key is a caller who believes this thesis reads something it
    never mentions. Neither is resolvable by best effort, so both refuse, naming what is missing
    and what was offered instead."""
    px = _ohlcv(tmp_path / "px.csv", n=6)
    thesis = _thesis(["AAA", "BBB"])
    with pytest.raises(ValueError, match=r"missing \['BBB'\]"):
        files_for(thesis, {"AAA": str(px)})
    with pytest.raises(ValueError, match=r"unknown \['CCC'\]"):
        files_for(thesis, {"AAA": str(px), "BBB": str(px), "CCC": str(px)})
    # the reserved benchmark slot is a declared key like any other: params.benchmark='market' is
    # what puts it in the request, and an invocation that ignores it is refused the same way
    with pytest.raises(ValueError, match=r"missing \['benchmark'\]"):
        files_for(_thesis(benchmark=True), {"target": str(px)})
    files = files_for(_thesis(benchmark=True), {"target": str(px), "benchmark": str(px)})
    assert files.targets == {"target": str(px)} and files.benchmark == str(px)


def test_column_bindings_are_checked_against_the_declared_keys(tmp_path):
    """The other half of a binding, checked by the same resolver but not by the same rule.

    A COLUMN is optional where a path is mandatory — only a file holding several numeric columns
    needs one — so nothing here demands that every key be bound. What is refused is a binding that
    cannot mean anything: a key this thesis never declares (nothing would ever read it, so the
    caller is holding a different thesis in their head), and ``benchmark``, which measures outcomes
    off its file's open price and has no column to choose. Both are decidable from the document
    alone, so they refuse here — before a byte is read — rather than after a file is opened."""
    px = _ohlcv(tmp_path / "px.csv", n=6)
    thesis = _thesis(["AAA", "BBB"])
    mapping = {"AAA": str(px), "BBB": str(px)}
    with pytest.raises(ValueError, match=r"does not declare \(\['CCC'\]\)"):
        files_for(thesis, mapping, {"CCC": "y10"})
    with pytest.raises(ValueError, match="benchmark"):
        files_for(
            _thesis(benchmark=True),
            {"target": str(px), "benchmark": str(px)},
            {"benchmark": "open"},
        )
    # …and a legal binding simply RIDES the resolution, key for key: nothing here opens the file,
    # so whether a column by that name exists is still the loader's question.
    files = files_for(thesis, mapping, {"AAA": "y10"})
    assert files.columns == {"AAA": "y10"}
    assert files_for(thesis, mapping).columns == {}  # binding nothing is the ordinary case


def test_per_target_feed_members_may_bind_different_columns(tmp_path):
    """The WIDENING per-key binding buys: each member of a per-target feed is bound under
    its own derived key, so the members may read DIFFERENT columns — here both from ONE wide CSV
    that holds a column per name. A single ``data.external.<feed>.column`` could only say one
    name for the whole family, leaving this shape unstateable; the binding is the exact symmetric
    twin of the per-member ``--data <feed>@<target>=PATH``. The assertion is on the loaded
    VALUES, because a binding that silently fell back to the file's first column would still
    produce a frame of the right shape."""
    idx = pd.date_range("2021-01-01", periods=6, freq="1D")
    for sym in ("AAA", "BBB"):
        _ohlcv(tmp_path / f"{sym}.csv", n=6)
    wide = pd.DataFrame(
        {"pe_aaa": 10.0 + np.arange(6, dtype=float), "pe_bbb": 20.0 + np.arange(6, dtype=float)},
        index=idx,
    )
    wide.index.name = "datetime"
    wide.to_csv(tmp_path / "pe.csv")

    thesis = _thesis(["AAA", "BBB"], external={"pe": {"per_target": True}})
    mapping = {
        "AAA": str(tmp_path / "AAA.csv"),
        "BBB": str(tmp_path / "BBB.csv"),
        "pe@AAA": str(tmp_path / "pe.csv"),
        "pe@BBB": str(tmp_path / "pe.csv"),
    }
    md = load(thesis, mapping, {"pe@AAA": "pe_aaa", "pe@BBB": "pe_bbb"})
    feed = md.externals["pe"]
    assert list(feed.columns) == ["AAA", "BBB"]
    np.testing.assert_allclose(feed["AAA"].to_numpy(), 10.0 + np.arange(6, dtype=float))
    np.testing.assert_allclose(feed["BBB"].to_numpy(), 20.0 + np.arange(6, dtype=float))

    # Leaving ONE member unbound refuses on that member alone, naming its derived key — the file
    # is ambiguous for it however unambiguous the invocation was about its sibling.
    with pytest.raises(DataError) as e:
        load(thesis, mapping, {"pe@AAA": "pe_aaa"})
    assert "bind --column pe@BBB=COL" in str(e.value)


def test_per_target_feed_derives_one_key_per_target(tmp_path):
    """A per-target feed is answered under keys DERIVED from the targets (``<feed>@<target>``), so
    its cover is by construction — there is no per-feed path mapping left to check against the
    target list, so a "keys must match the targets exactly" refusal would have nothing to refuse.
    A shared feed answers to its own bare name, and the two spellings never overlap."""
    idx = pd.date_range("2021-01-01", periods=6, freq="1D")
    for sym in ("AAA", "BBB"):
        _ohlcv(tmp_path / f"{sym}.csv", n=6)
    for sym, base in (("AAA", 1.0), ("BBB", 2.0)):
        feed = pd.DataFrame({"iv": base + np.arange(6, dtype=float)}, index=idx)
        feed.index.name = "datetime"
        feed.to_csv(tmp_path / f"iv_{sym}.csv")

    thesis = _thesis(["AAA", "BBB"], external={"iv": {"per_target": True}})
    assert thesis.data_keys() == ["AAA", "BBB", "iv@AAA", "iv@BBB"]
    mapping = {
        "AAA": str(tmp_path / "AAA.csv"),
        "BBB": str(tmp_path / "BBB.csv"),
        "iv@AAA": str(tmp_path / "iv_AAA.csv"),
        "iv@BBB": str(tmp_path / "iv_BBB.csv"),
    }
    assert files_for(thesis, mapping).feeds["iv"] == {
        "AAA": mapping["iv@AAA"],
        "BBB": mapping["iv@BBB"],
    }
    md = load(thesis, mapping)
    assert list(md.externals["iv"].columns) == ["AAA", "BBB"]  # one series per member
    assert md.externals["iv"].iloc[0].tolist() == [1.0, 2.0]

    # the bare feed name is NOT a key of a per-target feed — answering it leaves the derived keys
    # unanswered, which is exactly the cover mistake the derivation makes unstateable
    with pytest.raises(ValueError, match=r"missing \['iv@AAA', 'iv@BBB'\]"):
        files_for(thesis, {k: v for k, v in mapping.items() if "@" not in k} | {"iv": "iv.csv"})

    # …and a SHARED feed asks for exactly one file, under its own name
    shared = _thesis(["AAA", "BBB"], external={"iv": {}})
    assert shared.data_keys() == ["AAA", "BBB", "iv"]
    assert files_for(
        shared, {**{k: mapping[k] for k in ("AAA", "BBB")}, "iv": mapping["iv@AAA"]}
    ).feeds == {"iv": mapping["iv@AAA"]}


def test_start_end_slice(tmp_path):
    path = _ohlcv(tmp_path / "px.csv", n=10)
    md = load(_thesis(start="2021-01-04", end="2021-01-07"), {"target": path})
    assert len(md.index) == 4
    assert md.index[0] == pd.Timestamp("2021-01-04")
    assert md.index[-1] == pd.Timestamp("2021-01-07")


def test_report_attached_on_success(tmp_path):
    path = _ohlcv(tmp_path / "px.csv", n=10)
    md = load(_thesis(), {"target": path})
    report = md.report
    assert report is not None and report["ok"] is True
    assert [f["role"] for f in report["files"]] == ["target:target"]
    assert report["files"][0]["shape"] == "ohlcv"
    assert report["join"]["n_common"] == 10


def test_missing_ohlcv_columns_errors(tmp_path):
    # The benchmark must be OHLCV-shaped; a file without the OHLC columns refuses with a
    # shape_mismatch (targets missing a column are treated as series files instead).
    good = _ohlcv(tmp_path / "px.csv", n=5)
    idx = pd.date_range("2021-01-01", periods=5, freq="1D")
    bad = pd.DataFrame({"open": 1.0, "low": 1.0, "close": 1.0}, index=idx)
    bad.index.name = "datetime"
    bench = tmp_path / "bench.csv"
    bad.to_csv(bench)
    with pytest.raises(DataError) as e:
        load(_thesis(benchmark=True), {"target": good, "benchmark": bench})
    assert "shape_mismatch" in _codes(e.value)
    assert "integrity" in _codes(e.value)  # missing OHLCV columns from check_frame


def test_inverted_high_low_refuses(tmp_path):
    # A verifier never mutates evidence: inconsistent OHLC refuses, it is never clamped into shape.
    idx = pd.date_range("2021-01-01", periods=3, freq="1D")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [90.0, 90.0, 90.0],  # below open/close
            "low": [110.0, 110.0, 110.0],  # above open/close
            "close": [105.0, 105.0, 105.0],
            "volume": 1000.0,
        },
        index=idx,
    )
    df.index.name = "datetime"
    path = tmp_path / "clamp.csv"
    df.to_csv(path)
    with pytest.raises(DataError) as e:
        load(_thesis(), {"target": path})
    assert "integrity" in _codes(e.value)
    messages = "; ".join(err["message"] for f in e.value.report["files"] for err in f["errors"])
    assert "high < low" in messages


def test_non_iso_dates_refuse(tmp_path):
    path = tmp_path / "usdates.csv"
    path.write_text(
        "datetime,open,high,low,close\n01/02/2021,100,101,99,100\n01/03/2021,100,101,99,100\n",
        encoding="utf-8",
    )
    with pytest.raises(DataError) as e:
        load(_thesis(), {"target": path})
    assert "bad_timestamp" in _codes(e.value)


def test_tz_aware_timestamps_refuse(tmp_path):
    path = tmp_path / "tz.csv"
    path.write_text(
        "datetime,open,high,low,close\n"
        "2021-01-01T00:00:00+09:00,100,101,99,100\n"
        "2021-01-02T00:00:00+09:00,100,101,99,100\n",
        encoding="utf-8",
    )
    with pytest.raises(DataError) as e:
        load(_thesis(), {"target": path})
    assert "tz_aware_timestamp" in _codes(e.value)


def test_bad_files_all_reported_in_one_error(tmp_path):
    # Both a bad target AND a bad benchmark must appear in a single DataError report.
    bad_target = tmp_path / "t.csv"
    bad_target.write_text(
        "datetime,open,high,low,close\n2021-01-01,100,90,110,105\n", encoding="utf-8"
    )
    bad_bench = tmp_path / "b.csv"
    bad_bench.write_text("datetime,open,high,low,close\nnot-a-date,1,1,1,1\n", encoding="utf-8")
    with pytest.raises(DataError) as e:
        load(_thesis(benchmark=True), {"target": bad_target, "benchmark": bad_bench})
    by_role = {f["role"]: f for f in e.value.report["files"]}
    assert not by_role["target:target"]["ok"]
    assert not by_role["benchmark"]["ok"]


def test_unequal_target_indices_refuse_instead_of_intersecting(tmp_path):
    """Targets must share ONE bar clock.

    A loader that intersected instead would let a timestamp missing from one target silently
    delete that bar from EVERY target, with no coverage ledger recording it — trimming rows under
    adverse firings shrinks the graded pool invisibly. It refuses at the data boundary (exit 2),
    naming what is missing where."""
    a = _ohlcv(tmp_path / "a.csv", n=6, start="2021-01-01")
    b = _ohlcv(tmp_path / "b.csv", n=6, start="2021-01-04")  # only Jan 4–6 in common
    with pytest.raises(DataError) as e:
        load(_thesis(["a", "b"]), {"a": a, "b": b})
    assert "target_index_mismatch" in _codes(e.value)
    detail = str(e.value)
    assert "'a'" in detail and "'b'" in detail  # both sides named
    assert "2021-01-07" in detail or "2021-01-01" in detail  # a concrete missing stamp

    # One-bar overlap reports the same mismatch (it is checked BEFORE the 2-stamp floor).
    c = _ohlcv(tmp_path / "c.csv", n=3, start="2021-01-01")
    d = _ohlcv(tmp_path / "d.csv", n=3, start="2021-01-03")
    with pytest.raises(DataError) as e:
        load(_thesis(["c", "d"]), {"c": c, "d": d})
    assert "target_index_mismatch" in _codes(e.value)


def test_identical_target_indices_load_and_keep_every_bar(tmp_path):
    a = _ohlcv(tmp_path / "a.csv", n=6, start="2021-01-01")
    b = _ohlcv(tmp_path / "b.csv", n=6, start="2021-01-01")
    md = load(_thesis(["a", "b"]), {"a": a, "b": b})
    assert list(md.targets) == ["a", "b"]
    assert len(md.index) == 6  # nothing silently dropped


def test_deleting_one_bar_from_one_target_refuses(tmp_path):
    """Deletion monotonicity at the loader: removing a row from ONE target can never quietly
    produce a smaller (possibly more favourable) common pool."""
    a = _ohlcv(tmp_path / "a.csv", n=8, start="2021-01-01")
    full = pd.read_csv(a)
    holed = tmp_path / "b.csv"
    full.drop(index=4).to_csv(holed, index=False)  # one interior bar deleted
    _ohlcv(tmp_path / "a2.csv", n=8, start="2021-01-01")
    with pytest.raises(DataError) as e:
        load(_thesis(["a", "b"]), {"a": a, "b": holed})
    assert "target_index_mismatch" in _codes(e.value)
    assert "missing 1 timestamp" in str(e.value)


def test_index_mismatch_listing_is_bounded(tmp_path):
    a = _ohlcv(tmp_path / "a.csv", n=20, start="2021-01-01")
    b = _ohlcv(tmp_path / "b.csv", n=6, start="2021-01-01")  # 14 stamps missing
    with pytest.raises(DataError) as e:
        load(_thesis(["a", "b"]), {"a": a, "b": b})
    detail = str(e.value)
    assert "missing 14 timestamp" in detail
    assert "+9 more" in detail  # 5 examples shown, remainder counted


def test_case_insensitive_columns(tmp_path):
    idx = pd.date_range("2021-01-01", periods=4, freq="1D")
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0],
            "High": [101.0, 102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0, 102.0],
            "Close": [100.5, 101.5, 102.5, 103.5],
            "Volume": 1000.0,
        },
        index=idx,
    )
    df.index.name = "datetime"
    path = tmp_path / "caps.csv"
    df.to_csv(path)
    md = load(_thesis(), {"target": path})
    assert float(md.open.iloc[0, 0]) == 100.0
    assert float(md.close.iloc[-1, 0]) == 103.5
    assert md.volume is not None


def test_loaded_frames_are_read_only(tmp_path):
    # the evaluation memos are keyed by node identity and trust the frames never to move; the
    # loader now enforces it — a value write through a loaded frame raises instead of silently
    # serving stale cached series afterwards.
    _ohlcv(tmp_path / "px.csv")
    thesis = _thesis()
    md = load(thesis, {"target": str(tmp_path / "px.csv")})
    with pytest.raises(ValueError):
        md.close.iloc[0, 0] = 1e9
    with pytest.raises(ValueError):
        md.open.iloc[-1, 0] = 0.0
