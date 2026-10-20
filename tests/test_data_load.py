"""Load-contract tests for ``compiler/data.py`` (key binding, start/end, strict rejection, join,
reports)."""

from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seikan import dataio
from seikan.compiler import vectorize
from seikan.dataio import DataError
from seikan.dsl.schema import Thesis
from tests._helpers import files_for, load


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


def test_target_clocks_compare_in_exact_ticks_past_the_nanosecond_range(tmp_path):
    # The review's case: a nanosecond-stamped target beside one holding a 2500-01-01 bar (a
    # microsecond clock). pandas' union promoted both to int64 nanoseconds and crashed the load
    # (OutOfBoundsDatetime, exit 4) instead of naming the mismatch; in exact ticks it is the
    # ordinary refusal, each target's missing stamp named.
    a = tmp_path / "a.csv"
    a.write_text(
        "datetime,open,high,low,close\n"
        + "".join(f"2021-01-0{d}T00:00:00.000000000,1,1,1,1\n" for d in (1, 2, 3)),
        encoding="utf-8",
    )
    b = tmp_path / "b.csv"
    b.write_text(
        "datetime,open,high,low,close\n"
        + "".join(f"{d},1,1,1,1\n" for d in ("2021-01-01", "2021-01-02", "2500-01-01")),
        encoding="utf-8",
    )
    with pytest.raises(DataError) as e:
        load(_thesis(["a", "b"]), {"a": a, "b": b})
    assert _codes(e.value) == {"target_index_mismatch"}
    detail = str(e.value)
    assert "target 'a': missing 1 timestamp(s) present in other targets (2500-01-01T00:00:00)" in (
        detail
    )
    assert "target 'b': missing 1 timestamp(s) present in other targets (2021-01-03T00:00:00)" in (
        detail
    )


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
    # The evaluation memos are keyed by node identity and trust the frames never to move. The
    # loader's read-only buffers are a tripwire for the in-place value write — the one mutation
    # the engine itself could make by accident — which raises instead of silently serving stale
    # cached series afterwards. They are no immutability, which pandas cannot give: a replaced
    # column goes through, a caller's breach of the no-mutation contract, and the memo serves the
    # old build until `clear_memo()` releases it.
    _ohlcv(tmp_path / "px.csv")
    thesis = _thesis()
    md = load(thesis, {"target": str(tmp_path / "px.csv")})
    with pytest.raises(ValueError):
        md.close.iloc[0, 0] = 1e9
    with pytest.raises(ValueError):
        md.open.iloc[-1, 0] = 0.0
    close = thesis.entry.left
    before = vectorize.build_series(close, md)[0].to_numpy().copy()
    md.close["target"] = md.close["target"] * 2.0
    np.testing.assert_array_equal(vectorize.build_series(close, md)[0].to_numpy(), before)
    md.clear_memo()
    np.testing.assert_array_equal(vectorize.build_series(close, md)[0].to_numpy(), 2.0 * before)


def test_a_file_bound_under_several_keys_is_read_once(tmp_path, monkeypatch):
    # One file can answer several keys. Read once per binding, a file rewritten between those
    # reads was measured as two different files under one path, while the run's data identity
    # (one digest per path) named only the last. The load reads each distinct path once and every
    # binding parses that snapshot: here the bytes on disk change right after the first read, and
    # both targets still measure — and digest — the bytes that read returned.
    px = _ohlcv(tmp_path / "px.csv", n=8)
    first = px.read_bytes()
    other = _ohlcv(tmp_path / "other.csv", n=8, closes=np.linspace(50.0, 60.0, 8)).read_bytes()
    reads: list[str] = []
    real_read = dataio._read_blob

    def rewritten_after_its_read(path, rep):
        reads.append(path)
        blob = real_read(path, rep)
        if path == str(px):
            px.write_bytes(other)
        return blob

    monkeypatch.setattr(dataio, "_read_blob", rewritten_after_its_read)
    md = load(_thesis(["AAA", "BBB"]), {"AAA": str(px), "BBB": str(px)})
    assert reads == [str(px)]
    files = md.report["files"]
    assert [f["role"] for f in files] == ["target:AAA", "target:BBB"]
    assert files[0]["sha256"] == files[1]["sha256"] == hashlib.sha256(first).hexdigest()
    np.testing.assert_array_equal(md.close["AAA"], np.linspace(100.0, 110.0, 8))
    np.testing.assert_array_equal(md.close["BBB"], md.close["AAA"])
    # a path that cannot be read fails every binding of it, each under its own role
    gone = str(tmp_path / "gone.csv")
    with pytest.raises(DataError) as err:
        load(_thesis(["AAA", "BBB"]), {"AAA": gone, "BBB": gone})
    assert [(f["role"], [e["code"] for e in f["errors"]]) for f in err.value.report["files"]] == [
        ("target:AAA", ["file_missing"]),
        ("target:BBB", ["file_missing"]),
    ]


@pytest.mark.parametrize(
    ("last", "lag", "carried"),
    [
        # a nanosecond clock at its last instant: no lag fits (pandas raised OverflowError — an
        # exit 4 out of the load)
        ("2262-04-11T23:47:16.854775807", "1ns", None),
        # a microsecond stamp past 2262: a nanosecond lag promotes it out of the nanosecond range
        # (OutOfBoundsDatetime), while a lag at the stamp's own resolution carries it
        ("2300-01-01T00:00:00.000001", "1ns", 1),
    ],
)
def test_a_lag_past_the_representable_clock_refuses(tmp_path, last, lag, carried):
    px = _ohlcv(tmp_path / "px.csv")
    feed = tmp_path / "f.csv"
    pd.DataFrame({"datetime": ["2021-01-02", last], "f": [1.0, 2.0]}).to_csv(feed, index=False)
    files = {"target": str(px), "f": str(feed)}
    with pytest.raises(DataError) as err:
        load(_thesis(external={"f": {"lag": lag}}), files)
    assert _codes(err.value) == {"spec_data_mismatch"}
    message = err.value.report["errors"][0]["message"]
    assert "publication lag" in message and "'f'" in message and last[:10] in message
    if carried is not None:
        md = load(_thesis(external={"f": {"lag": carried}}), files)
        assert md.externals_native["f"].index[-1] == pd.Timestamp(last) + pd.Timedelta(days=1)


def test_load_retains_native_prints_post_lag_shared_and_per_target(tmp_path):
    # The loader keeps each feed's NATIVE prints (post-lag stamps, values untouched, read-only,
    # NOT sliced by data.start/end) beside the anchored values — what a `native` node reads.
    _ohlcv(tmp_path / "AAA.csv", n=6, start="2021-01-03")
    _ohlcv(tmp_path / "BBB.csv", n=6, start="2021-01-03")
    shared = pd.DataFrame(
        {"v": [1.0, 2.0, 3.0]}, index=pd.DatetimeIndex(["2020-12-30", "2021-01-04", "2021-01-06"])
    )
    shared.index.name = "datetime"
    shared.to_csv(tmp_path / "s.csv")
    for sym, vals in (("AAA", [10.0, 11.0]), ("BBB", [20.0, 21.0, 22.0])):
        df = pd.DataFrame(
            {"v": vals}, index=pd.date_range("2021-01-02", periods=len(vals), freq="2D")
        )
        df.index.name = "datetime"
        df.to_csv(tmp_path / f"p_{sym}.csv")
    thesis = _thesis(
        ["AAA", "BBB"],
        start="2021-01-04",
        external={"s": {"lag": 1}, "p": {"per_target": True}},
    )
    md = load(
        thesis,
        {
            "AAA": tmp_path / "AAA.csv",
            "BBB": tmp_path / "BBB.csv",
            "s": tmp_path / "s.csv",
            "p@AAA": tmp_path / "p_AAA.csv",
            "p@BBB": tmp_path / "p_BBB.csv",
        },
    )
    s = md.external_native("s")
    assert isinstance(s, pd.Series)
    # lag-shifted stamps, every print retained (the pre-start one included), values verbatim
    assert list(s.index) == list(pd.DatetimeIndex(["2020-12-31", "2021-01-05", "2021-01-07"]))
    np.testing.assert_array_equal(s.to_numpy(), [1.0, 2.0, 3.0])
    assert not s.to_numpy().flags.writeable
    p = md.external_native("p")
    assert isinstance(p, dict) and set(p) == {"AAA", "BBB"}
    np.testing.assert_array_equal(p["BBB"].to_numpy(), [20.0, 21.0, 22.0])
    assert len(p["AAA"]) == 2
    with pytest.raises(ValueError, match="external feed 'nope' has no retained native prints"):
        md.external_native("nope")


def test_feed_age_is_exact_tick_arithmetic_past_the_int64_nanosecond_span(tmp_path):
    # The review's case: a feed stamped at a nanosecond past 1700 against bars in 2026. pandas
    # formed the age as an int64 nanosecond difference, which overflowed past ~292 years and
    # crashed the load (OverflowError); in exact ticks the age is defined.
    _ohlcv(tmp_path / "target.csv", n=3, start="2026-01-01")
    pd.DataFrame(
        {"v": [1.0]}, index=pd.Index(["1700-01-01T00:00:00.000000001"], name="datetime")
    ).to_csv(tmp_path / "old.csv")
    thesis = _thesis(external={"old": {}})
    md = load(thesis, {"target": tmp_path / "target.csv", "old": tmp_path / "old.csv"})
    ages = md.days_since_values("old")[:, 0]
    np.testing.assert_array_equal(ages, [119069.0, 119070.0, 119071.0])


@pytest.mark.parametrize("units", [("us", "us"), ("us", "ns"), ("ns", "us"), ("s", "ms")])
def test_feed_age_is_the_exact_tick_quotient_rounded_once(units):
    # In any mix of clock resolutions the age is the exact difference over a day, rounded once —
    # a bar on a stamp reads 0.0, a bar before the first stamp NaN. pandas' Timedelta division
    # (this test's former reference) casts a nanosecond difference past 2**53 (~104 days) to a
    # double first, and so misses the correctly rounded age by an ulp on a fifth of those spans.
    from seikan.compiler.data import _feed_age_days

    rng = np.random.default_rng(len(units[0]) * 7 + len(units[1]))
    base = pd.Timestamp("1990-01-01").value
    for _ in range(50):
        f = np.unique(base + rng.integers(0, 4 * 10**17, 30) // 10**9 * 10**9)
        t = np.unique(np.r_[base - 10**12, f[:10], base + rng.integers(0, 10**18, 40)])
        native = pd.Series(
            1.0, index=pd.DatetimeIndex(f.astype("datetime64[ns]")).as_unit(units[1])
        )
        bars = pd.DatetimeIndex(t.astype("datetime64[ns]")).as_unit(units[0])
        stamps = pd.Series(native.index, index=native.index)
        behind = bars.to_series(index=bars) - stamps.reindex(bars, method="ffill")
        want = [np.nan if pd.isna(d) else float(Fraction(d.value, 86_400 * 10**9)) for d in behind]
        got = _feed_age_days(native, bars)
        np.testing.assert_array_equal(got.to_numpy(), want)
        assert np.isnan(got.iloc[0]) and (got.loc[native.index[:10]] == 0.0).all()


def test_feed_age_rounds_once_at_a_freshness_boundary(tmp_path):
    # The review's case: a print 120 days and 1 ns before the first bar. The tick difference was
    # cast to a double (past 2**53 ns it rounds to even) before the division rounded again, so
    # the age read 120.0; the exact quotient, rounded once, is the next double above 120.
    _ohlcv(tmp_path / "target.csv", n=3, start="2024-01-01")
    stamp = pd.Timestamp("2024-01-01") - pd.Timedelta(days=120, nanoseconds=1)
    pd.DataFrame({"datetime": [stamp.isoformat()], "f": [1.0]}).to_csv(
        tmp_path / "f.csv", index=False
    )
    md = load(
        _thesis(external={"f": {}}), {"target": tmp_path / "target.csv", "f": tmp_path / "f.csv"}
    )
    day = 86_400 * 10**9
    want = [float(Fraction((120 + k) * day + 1, day)) for k in range(3)]
    np.testing.assert_array_equal(md.days_since_values("f")[:, 0], want)
    assert want[0] == np.nextafter(120.0, np.inf)


def test_feed_anchoring_is_exact_tick_arithmetic_past_the_nanosecond_clock(tmp_path):
    # The review's case: a feed holding a print at 2500-01-01 — a microsecond clock, past the
    # nanosecond range — beside nanosecond bars in 2026. The asof anchor was pandas' reindex-ffill,
    # which promoted the feed clock to nanoseconds and refused the load (OutOfBoundsDatetime, a
    # ValueError, so spec_data_mismatch at exit 2) over a print no bar ever reads, after check-data
    # had accepted both files; the feed age beside it was already exact ticks. In exact ticks
    # every bar reads the 2026 print, and the age agrees with the position.
    from seikan.compiler.data import _feed_age_days, anchor_feed

    bars = pd.date_range("2026-01-01T00:00:00.000000001", periods=4, freq="1D")
    feed = pd.Series([2.0, 999.0], index=pd.DatetimeIndex(["2026-01-01", "2500-01-01"]), name="f")
    assert bars.unit == "ns" and feed.index.unit != "ns"
    anchored = anchor_feed(feed, bars)
    np.testing.assert_array_equal(anchored.to_numpy(), [2.0, 2.0, 2.0, 2.0])
    assert anchored.index.equals(bars) and anchored.name == "f"
    np.testing.assert_array_equal(np.floor(_feed_age_days(feed, bars).to_numpy()), [0, 1, 2, 3])
    # through the loader, the thesis the review ran: the feed reads 2 on every bar
    pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=pd.Index(bars, name="datetime"),
    ).to_csv(tmp_path / "target.csv")
    pd.DataFrame({"datetime": ["2026-01-01", "2500-01-01"], "f": [2.0, 999.0]}).to_csv(
        tmp_path / "f.csv", index=False
    )
    md = load(
        _thesis(external={"f": {}}), {"target": tmp_path / "target.csv", "f": tmp_path / "f.csv"}
    )
    np.testing.assert_array_equal(md.external_values("f")[:, 0], [2.0, 2.0, 2.0, 2.0])
    assert list(md.externals_native["f"].index) == list(feed.index)


@pytest.mark.parametrize("units", [("us", "us"), ("us", "ns"), ("ns", "us"), ("s", "ms")])
def test_feed_anchoring_is_pandas_reindex_ffill_bit_for_bit_in_range(units):
    # Wherever the int64 comparison held, the anchored value is the very one pandas' backward
    # asof (reindex-ffill) gave: the latest print at or before each bar, NaN before the first —
    # a bar ON a stamp reads that stamp's print, never the next.
    from seikan.compiler.data import anchor_feed

    rng = np.random.default_rng(len(units[0]) * 11 + len(units[1]))
    base = pd.Timestamp("1990-01-01").value
    for _ in range(50):
        f = np.unique(base + rng.integers(0, 4 * 10**17, 30) // 10**9 * 10**9)
        t = np.unique(np.r_[base - 10**12, f[:10], base + rng.integers(0, 10**18, 40)])
        native = pd.Series(
            rng.standard_normal(f.size),
            index=pd.DatetimeIndex(f.astype("datetime64[ns]")).as_unit(units[1]),
            name="v",
        )
        bars = pd.DatetimeIndex(t.astype("datetime64[ns]")).as_unit(units[0])
        got = anchor_feed(native, bars)
        pd.testing.assert_series_equal(got, native.reindex(bars, method="ffill"))
        assert np.isnan(got.iloc[0]) and (got.loc[native.index[:10]] == native.iloc[:10]).all()


def test_a_benchmark_bar_past_the_nanosecond_range_is_ignored_not_a_crash(tmp_path):
    # The review's case: a benchmark holding a 2500-01-01 bar (a microsecond clock) beside a
    # nanosecond target. pandas' reindex promoted it to int64 nanoseconds and crashed the load
    # (OutOfBoundsDatetime, exit 4) over a bar no target reads; matched in exact ticks it is
    # ignored, each bar reading the benchmark stamped at exactly its instant, NaN where none is.
    target = tmp_path / "target.csv"
    target.write_text(
        "datetime,open,high,low,close\n"
        + "".join(f"2021-01-0{d}T00:00:00.00000000{d // 3},1,1,1,1\n" for d in (1, 2, 3)),
        encoding="utf-8",
    )
    bench = tmp_path / "bench.csv"
    bench.write_text(
        "datetime,open,high,low,close\n"
        + "".join(
            f"{d},{v},{v},{v},{v}\n"
            for d, v in (("2021-01-01", 1), ("2021-01-02", 2), ("2021-01-03", 3), ("2500-01-01", 9))
        ),
        encoding="utf-8",
    )
    md = load(_thesis(benchmark=True), {"target": target, "benchmark": bench})
    assert md.index.unit == "ns"
    np.testing.assert_array_equal(md.benchmark_open.to_numpy(), [1.0, 2.0, np.nan])
    assert md.benchmark_open.index.equals(md.index)


@pytest.mark.parametrize("units", [("us", "us"), ("us", "ns"), ("ns", "us"), ("s", "ms")])
def test_benchmark_matching_is_pandas_reindex_bit_for_bit_in_range(units):
    # Wherever the int64 comparison held, the matched value is the very one pandas' reindex (no
    # fill) gave: the print stamped at exactly the bar's instant, NaN everywhere else.
    from seikan.compiler.data import _match_feed

    rng = np.random.default_rng(len(units[0]) * 11 + len(units[1]))
    base = pd.Timestamp("1990-01-01").value
    for _ in range(50):
        f = np.unique(base + rng.integers(0, 4 * 10**17, 30) // 10**9 * 10**9)
        t = np.unique(np.r_[base - 10**12, f[:10], base + rng.integers(0, 10**18, 40)])
        native = pd.Series(
            rng.standard_normal(f.size),
            index=pd.DatetimeIndex(f.astype("datetime64[ns]")).as_unit(units[1]),
            name="open",
        )
        bars = pd.DatetimeIndex(t.astype("datetime64[ns]")).as_unit(units[0])
        got = _match_feed(native, bars)
        pd.testing.assert_series_equal(got, native.reindex(bars))
        assert (got.loc[native.index[:10]] == native.iloc[:10]).all()
