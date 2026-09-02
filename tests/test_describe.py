"""End-to-end tests for ``seikan describe`` (in-process ``main()``): the document shape, the
argument-order and check-data-parity contracts, the domain-gated algebra refusals, the
refuse-never-repair rule, hand-computed values (including the most-recent-tie rule), and the
two structural invariants — byte-determinism and BOUNDED OUTPUT (no per-bar array ever rides
the document, whose size is independent of ``n_bars``).

``describe`` is a pure observer of FILES the way ``run`` is of THESES: it profiles, measures
nothing, and supports no thesis. The exit code carries check-data's meaning — 0 all admitted,
2 any refused (document still emitted) — never a judgment about the data's content.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from seikan.cli import MAX_DESCRIBE_WINDOWS, main
from tests._helpers import write_ohlcv as _write_ohlcv

# ---- fixtures -------------------------------------------------------------------------------


def _write_series(path, values, columns=("y",)):
    """A literal series-shaped CSV: one row per element of ``values`` (a scalar for one column,
    a tuple for several), None → empty cell (a NaN hole)."""
    idx = pd.date_range("2024-01-01", periods=len(values), freq="1D")
    lines = ["datetime," + ",".join(columns)]
    for ts, row in zip(idx, values, strict=True):
        cells = row if isinstance(row, (tuple, list)) else (row,)
        lines.append(
            ts.date().isoformat()
            + ","
            + ",".join("" if v is None else repr(float(v)) for v in cells)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_flat_ohlcv(path, closes, volumes=None):
    """A literal OHLCV CSV with hand-controlled closes (and optionally volumes): open = close,
    high/low bracket them, so every invariant holds by construction."""
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1D")
    cols = "datetime,open,high,low,close" + (",volume" if volumes is not None else "")
    lines = [cols]
    for i, (ts, c) in enumerate(zip(idx, closes, strict=True)):
        row = f"{ts.date().isoformat()},{c!r},{c + 1.0!r},{c - 1.0!r},{c!r}"
        if volumes is not None:
            row += f",{volumes[i]!r}"
        lines.append(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run(capsys, argv):
    code = main(argv)
    out = capsys.readouterr().out
    return code, json.loads(out)


def _series_profile(doc, i: int = 0, column: str | None = None):
    series = doc["profiles"][i]["series"]
    return series[column or next(iter(series))]


def _all_numbers(node, out):
    if isinstance(node, dict):
        for v in node.values():
            _all_numbers(v, out)
    elif isinstance(node, list):
        for v in node:
            _all_numbers(v, out)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out.append(float(node))


# ---- document shape and header ---------------------------------------------------------------


def test_document_layers_header_and_profile_shape(tmp_path, capsys):
    px = _write_ohlcv(tmp_path / "px.csv")
    code, doc = _run(capsys, ["describe", str(px)])
    assert code == 0
    # the FIXED layer order — the contract consumers parse against
    assert list(doc) == [
        "seikan_version",
        "report_schema_version",
        "command",
        "data_report",
        "profiles",
        "describe_roles",
    ]
    assert doc["report_schema_version"] == 5
    assert doc["command"] == "describe" and doc["seikan_version"]
    assert doc["data_report"]["ok"] is True
    profile = doc["profiles"][0]
    assert list(profile) == [
        "path",
        "sha256",
        "ok",
        "shape",
        "n_bars",
        "index_start",
        "index_end",
        "bar_spacing",
        "last_bar",
        "series",
        "volume",
    ]
    assert profile["ok"] is True and profile["shape"] == "ohlcv"
    assert profile["path"] == str(px)
    assert profile["n_bars"] == 400
    # the digest is the data_report's own identity, not a second hash
    assert profile["sha256"] == doc["data_report"]["files"][0]["sha256"]
    assert len(profile["sha256"]) == 64
    # run-summary vocabulary: the clock geometry is the runner's own stamp (daily bars)
    assert profile["bar_spacing"] == {
        "min_seconds": 86400,
        "median_seconds": 86400,
        "max_seconds": 86400,
    }
    assert profile["index_end"] == profile["last_bar"]["timestamp"]
    # OHLCV profiles `close` ONLY — the full bar rides last_bar
    assert list(profile["series"]) == ["close"]
    assert list(profile["last_bar"]["values"]) == ["open", "high", "low", "close", "volume"]
    blocks = profile["series"]["close"]
    assert list(blocks) == [
        "changes",
        "dispersion",
        "range_position",
        "full_sample",
        "missingness",
    ]
    # default windows, in the default order, under every windowed block
    for name in ("changes", "dispersion", "range_position"):
        assert list(blocks[name]) == ["1", "5", "21", "63", "126", "252"]


def test_files_profiled_in_argument_order_never_sorted(tmp_path, capsys):
    # File names chosen so lexical order is the REVERSE of the first argv: a sorted
    # implementation passes one direction and fails the other.
    zzz = _write_ohlcv(tmp_path / "zzz.csv", n=60, seed=1)
    aaa = _write_ohlcv(tmp_path / "aaa.csv", n=60, seed=2)
    for argv_files in ([str(zzz), str(aaa)], [str(aaa), str(zzz)]):
        code, doc = _run(capsys, ["describe", *argv_files])
        assert code == 0
        assert [p["path"] for p in doc["profiles"]] == argv_files
        assert [f["path"] for f in doc["data_report"]["files"]] == argv_files


# ---- refusals: check-data parity, stubs, exit codes ------------------------------------------


def test_refusal_parity_with_check_data(tmp_path, capsys):
    good = _write_ohlcv(tmp_path / "good.csv", n=60)
    bad = tmp_path / "bad.csv"
    bad.write_text("datetime,close\n2021-01-02,100\n2021-01-01,101\n", encoding="utf-8")  # unsorted
    argv_files = [str(good), str(bad)]
    check_code, check_doc = _run(capsys, ["check-data", *argv_files])
    desc_code, desc_doc = _run(capsys, ["describe", *argv_files])
    assert check_code == desc_code == 2
    # the SAME strict read, so the data_report is identical — dict equality, not resemblance
    assert desc_doc["data_report"] == check_doc["data_report"]
    # the document is STILL emitted: the admitted file carries its full profile...
    assert desc_doc["profiles"][0]["ok"] is True
    assert desc_doc["profiles"][0]["series"]["close"]["full_sample"]["high"] is not None
    # ...and the refused one carries exactly the stub — nothing about it is invented
    stub = desc_doc["profiles"][1]
    assert list(stub) == ["path", "sha256", "ok", "reason"]
    assert stub["ok"] is False and stub["path"] == str(bad)
    assert len(stub["sha256"]) == 64  # a refused file still keeps its byte identity
    assert "unsorted_timestamp" in stub["reason"]


def test_shape_mismatch_exits_2_with_stub(tmp_path, capsys):
    y = _write_series(tmp_path / "y.csv", [1.0, 2.0, 3.0])
    code, doc = _run(capsys, ["describe", str(y), "--shape", "ohlcv"])
    assert code == 2
    stub = doc["profiles"][0]
    assert stub["ok"] is False
    assert "shape_mismatch" in stub["reason"]


def test_missing_file_is_a_data_refusal_not_a_crash(tmp_path, capsys):
    code, doc = _run(capsys, ["describe", str(tmp_path / "nope.csv")])
    assert code == 2
    stub = doc["profiles"][0]
    assert stub == {
        "path": str(tmp_path / "nope.csv"),
        "sha256": None,
        "ok": False,
        "reason": "file_missing",
    }


def test_no_files_is_a_usage_error(capsys):
    code, doc = _run(capsys, ["describe"])
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert doc["command"] == "describe"


# ---- windows: order, refusals ----------------------------------------------------------------


def test_custom_windows_emitted_in_given_order(tmp_path, capsys):
    px = _write_ohlcv(tmp_path / "px.csv", n=100)
    code, doc = _run(capsys, ["describe", str(px), "--windows", "21,5,63"])
    assert code == 0
    blocks = _series_profile(doc)
    for name in ("changes", "dispersion", "range_position"):
        assert list(blocks[name]) == ["21", "5", "63"]  # given order, never sorted
    assert list(doc["profiles"][0]["volume"]["windows"]) == ["21", "5", "63"]


@pytest.mark.parametrize(
    "windows",
    [
        "0",  # a zero-bar window is no window
        "-3",  # negative bar count
        "abc",  # not an integer
        "5.5",  # not an INTEGER bar count
        "5,5",  # duplicate → duplicate JSON key
        "",  # empty list
        "1,,5",  # empty entry
        ",".join(str(i + 1) for i in range(MAX_DESCRIBE_WINDOWS + 1)),  # over the cap
    ],
)
def test_bad_windows_are_usage_errors(tmp_path, capsys, windows):
    px = _write_ohlcv(tmp_path / "px.csv", n=30)
    code, doc = _run(capsys, ["describe", str(px), "--windows", windows])
    assert code == 3, windows
    assert doc["error"]["type"] == "usage"
    assert doc["command"] == "describe"
    assert "--windows" in doc["error"]["message"]


def test_insufficient_window_reason_blocks_present_never_omitted(tmp_path, capsys):
    y = _write_series(tmp_path / "y.csv", [float(i + 1) for i in range(10)])
    code, doc = _run(capsys, ["describe", str(y)])
    assert code == 0
    blocks = _series_profile(doc)
    # every requested window has its entry, refused ones INCLUDED — absence would read as
    # "not applicable" when what happened is "this file cannot fill the window"
    for name in ("changes", "dispersion", "range_position"):
        assert list(blocks[name]) == ["1", "5", "21", "63", "126", "252"], name
    for w in ("21", "63", "126", "252"):
        refused = blocks["changes"][w]
        assert refused["reason"] == "insufficient_bars"
        assert refused["diff"] is None and refused["pct"] is None and refused["log"] is None
        assert blocks["dispersion"][w]["reason"] == "insufficient_bars"
        assert blocks["range_position"][w]["reason"] == "insufficient_bars"
    # a 10-bar file DOES fill the 5-bar window (and range_position needs only 10 >= 21 to fail)
    assert blocks["changes"]["5"]["reason"] is None
    assert blocks["changes"]["5"]["diff"] == pytest.approx(5.0)
    # range_position at window 10 would live; at 21 it refuses rather than silently shortening
    assert blocks["range_position"]["5"]["reason"] is None


# ---- series selection ------------------------------------------------------------------------


def test_multi_column_series_profiles_every_column_in_file_order(tmp_path, capsys):
    data = _write_series(
        tmp_path / "two.csv",
        [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0)],
        columns=("z", "a"),  # file order is NOT lexical order
    )
    code, doc = _run(capsys, ["describe", str(data), "--windows", "1,2"])
    assert code == 0
    profile = doc["profiles"][0]
    assert profile["shape"] == "series"
    assert list(profile["series"]) == ["z", "a"]  # file order, never sorted
    assert profile["volume"] is None  # a series file has no volume block
    assert profile["series"]["a"]["changes"]["1"]["diff"] == pytest.approx(10.0)
    assert profile["series"]["z"]["changes"]["1"]["diff"] == pytest.approx(1.0)
    assert list(profile["last_bar"]["values"]) == ["z", "a"]


# ---- domain gating: the ratio algebras -------------------------------------------------------


def test_zero_crossing_series_refuses_ratio_algebras_while_diff_lives(tmp_path, capsys):
    y = _write_series(tmp_path / "y.csv", [2.0, 1.0, -1.0, 0.5, 1.5])
    code, doc = _run(capsys, ["describe", str(y), "--windows", "1,2,4"])
    assert code == 0  # a zero-crossing series is DATA, not an error — only the ratios refuse
    blocks = _series_profile(doc)
    # endpoints both positive → all three algebras live (the path between them is not read)
    ch1 = blocks["changes"]["1"]
    assert ch1["diff"] == pytest.approx(1.0)
    assert ch1["pct"] == pytest.approx(2.0)
    assert ch1["ratio_reason"] is None
    # a non-positive endpoint → diff lives, pct/log refuse with the named reason
    ch2 = blocks["changes"]["2"]
    assert ch2["diff"] == pytest.approx(2.5)
    assert ch2["pct"] is None and ch2["log"] is None
    assert ch2["ratio_reason"] == "non_positive_endpoint"
    assert ch2["reason"] is None  # the block itself did not refuse — only the ratio algebras
    # dispersion over a window that crosses zero: diff std lives, ratio stds refuse
    d4 = blocks["dispersion"]["4"]
    assert d4["diff"] == pytest.approx(float(np.std(np.diff([2.0, 1.0, -1.0, 0.5, 1.5]), ddof=1)))
    assert d4["pct"] is None and d4["ratio_reason"] == "non_positive_endpoint"
    # range position: distance from a non-positive low refuses in pct, lives in diff
    rp4 = blocks["range_position"]["4"]
    assert rp4["low"]["value"] == pytest.approx(-1.0)
    assert rp4["from_low"]["diff"] == pytest.approx(2.5)
    assert rp4["from_low"]["pct"] is None
    assert rp4["from_high"]["pct"] == pytest.approx(0.0)  # last IS the window high
    assert rp4["ratio_reason"] == "non_positive_endpoint"
    assert rp4["percentile_rank"] == pytest.approx(1.0)
    # full sample: runup off a negative low refuses in pct, drawdown off the positive high lives
    fs = blocks["full_sample"]
    assert fs["runup_pct"] is None and fs["runup_diff"] == pytest.approx(2.5)
    assert fs["drawdown_pct"] == pytest.approx(1.5 / 2.0 - 1.0)
    assert fs["ratio_reason"] == "non_positive_endpoint"


def test_nan_endpoint_is_never_repaired(tmp_path, capsys):
    # The last level is a HOLE. The forbidden move is skipping back to the previous finite
    # value: every number that repair would have produced must appear NOWHERE in the document.
    y = _write_series(tmp_path / "y.csv", [7.0, 9.5, None])
    code, doc = _run(capsys, ["describe", str(y), "--windows", "1,2"])
    assert code == 0  # a NaN hole is a warning, not a refusal — the loader admits the file
    blocks = _series_profile(doc)
    for w in ("1", "2"):
        assert blocks["changes"][w]["reason"] == "endpoint_missing"
        assert blocks["changes"][w]["diff"] is None
        assert blocks["range_position"][w]["reason"] == "endpoint_missing"
    assert blocks["dispersion"]["1"]["reason"] == "insufficient_bars"  # w=1: no ddof=1 std
    assert blocks["dispersion"]["2"]["reason"] == "endpoint_missing"
    # full-sample extremes are observed facts and stay; the distance reads refuse
    fs = blocks["full_sample"]
    assert fs["high"]["value"] == pytest.approx(9.5)
    assert fs["reason"] == "endpoint_missing"
    assert fs["drawdown_pct"] is None and fs["drawdown_diff"] is None
    # the last bar reports the hole as null — never the previous finite value
    assert doc["profiles"][0]["last_bar"]["values"]["y"] is None
    # and the REPAIRED values — what 9.5-vs-7.0 would have minted — appear nowhere at all
    numbers: list[float] = []
    _all_numbers(doc, numbers)
    for repaired in (2.5, 9.5 / 7.0 - 1.0, math.log(9.5 / 7.0)):
        assert all(abs(v - repaired) > 1e-12 for v in numbers), repaired
    # missingness states the hole in pure counts: trailing, so not interior
    miss = blocks["missingness"]
    assert miss == {
        "n_missing": 1,
        "n_interior_missing": 0,
        "first_valid": "2024-01-01T00:00:00",
        "last_valid": "2024-01-02T00:00:00",
    }


# ---- hand-computed values --------------------------------------------------------------------


def test_exact_values_including_the_most_recent_tie_rule(tmp_path, capsys):
    # levels: 1, 3, 2, 3, 2.5 — the maximum (3.0) is attained TWICE; the reported timestamp
    # must be the most recent attainment (the bars_since_extremum rule).
    y = _write_series(tmp_path / "y.csv", [1.0, 3.0, 2.0, 3.0, 2.5])
    code, doc = _run(capsys, ["describe", str(y), "--windows", "1,2,5"])
    assert code == 0
    blocks = _series_profile(doc)

    ch1 = blocks["changes"]["1"]
    assert ch1["diff"] == pytest.approx(-0.5)
    assert ch1["pct"] == pytest.approx(2.5 / 3.0 - 1.0)
    assert ch1["log"] == pytest.approx(math.log(2.5 / 3.0))
    assert ch1["reason"] is None and ch1["ratio_reason"] is None
    ch2 = blocks["changes"]["2"]
    assert ch2["diff"] == pytest.approx(0.5) and ch2["pct"] == pytest.approx(0.25)

    # dispersion[2] reads the SAME three trailing levels changes[2] reads: [2.0, 3.0, 2.5]
    d2 = blocks["dispersion"]["2"]
    assert d2["diff"] == pytest.approx(float(np.std([1.0, -0.5], ddof=1)))
    assert d2["pct"] == pytest.approx(float(np.std([0.5, 2.5 / 3.0 - 1.0], ddof=1)))
    assert d2["log"] == pytest.approx(float(np.std(np.log([1.5, 2.5 / 3.0]), ddof=1)))
    # a single 1-bar change cannot carry a ddof=1 std
    assert blocks["dispersion"]["1"]["reason"] == "insufficient_bars"

    rp5 = blocks["range_position"]["5"]
    assert rp5["high"] == {"value": 3.0, "timestamp": "2024-01-04T00:00:00"}  # the LATER tie
    assert rp5["low"] == {"value": 1.0, "timestamp": "2024-01-01T00:00:00"}
    assert rp5["from_high"]["diff"] == pytest.approx(-0.5)
    assert rp5["from_high"]["pct"] == pytest.approx(2.5 / 3.0 - 1.0)
    assert rp5["from_low"]["diff"] == pytest.approx(1.5)
    assert rp5["from_low"]["pct"] == pytest.approx(1.5)
    # right-continuous empirical CDF: share of window levels <= the last (1, 2, 2.5 of 5)
    assert rp5["percentile_rank"] == pytest.approx(0.6)
    rp1 = blocks["range_position"]["1"]
    assert rp1["high"]["value"] == rp1["low"]["value"] == 2.5
    assert rp1["percentile_rank"] == pytest.approx(1.0)

    fs = blocks["full_sample"]
    assert fs["high"] == {"value": 3.0, "timestamp": "2024-01-04T00:00:00"}  # same tie rule
    assert fs["low"] == {"value": 1.0, "timestamp": "2024-01-01T00:00:00"}
    assert fs["drawdown_diff"] == pytest.approx(-0.5)
    assert fs["drawdown_pct"] == pytest.approx(2.5 / 3.0 - 1.0)
    assert fs["runup_diff"] == pytest.approx(1.5)
    assert fs["runup_pct"] == pytest.approx(1.5)

    assert blocks["missingness"]["n_missing"] == 0


def test_volume_block_hand_computed_and_absent_without_volume(tmp_path, capsys):
    with_vol = _write_flat_ohlcv(
        tmp_path / "v.csv",
        [10.0, 11.0, 12.0, 13.0, 14.0],
        volumes=[10.0, 20.0, 30.0, 40.0, 50.0],
    )
    code, doc = _run(capsys, ["describe", str(with_vol), "--windows", "2,5"])
    assert code == 0
    vol = doc["profiles"][0]["volume"]
    assert vol["last"] == pytest.approx(50.0)
    assert vol["windows"]["2"]["mean"] == pytest.approx(45.0)
    assert vol["windows"]["2"]["last_to_mean"] == pytest.approx(50.0 / 45.0)
    assert vol["windows"]["5"]["mean"] == pytest.approx(30.0)
    assert vol["windows"]["5"]["last_to_mean"] == pytest.approx(5.0 / 3.0)
    assert vol["windows"]["2"]["reason"] is None

    # an OHLCV file WITHOUT a volume column carries volume: null — never a zero-filled block
    no_vol = _write_flat_ohlcv(tmp_path / "nv.csv", [10.0, 11.0, 12.0])
    code, doc = _run(capsys, ["describe", str(no_vol), "--windows", "1,2"])
    assert code == 0
    assert doc["profiles"][0]["shape"] == "ohlcv"
    assert doc["profiles"][0]["volume"] is None


def test_single_row_file(tmp_path, capsys):
    one = _write_flat_ohlcv(tmp_path / "one.csv", [10.5], volumes=[1000.0])
    code, doc = _run(capsys, ["describe", str(one), "--windows", "1,5"])
    assert code == 0
    profile = doc["profiles"][0]
    assert profile["n_bars"] == 1
    assert profile["index_start"] == profile["index_end"]
    # below two bars there is no spacing to describe — null, never a guess
    assert profile["bar_spacing"] == {
        "min_seconds": None,
        "median_seconds": None,
        "max_seconds": None,
    }
    blocks = profile["series"]["close"]
    # a 1-bar change needs two bars; a 1-bar range needs exactly one and LIVES
    assert blocks["changes"]["1"]["reason"] == "insufficient_bars"
    rp1 = blocks["range_position"]["1"]
    assert rp1["reason"] is None
    assert rp1["high"]["value"] == rp1["low"]["value"] == 10.5
    assert rp1["from_high"]["diff"] == pytest.approx(0.0)
    assert rp1["percentile_rank"] == pytest.approx(1.0)
    assert blocks["range_position"]["5"]["reason"] == "insufficient_bars"
    assert blocks["full_sample"]["drawdown_diff"] == pytest.approx(0.0)
    vol = profile["volume"]
    assert vol["windows"]["1"]["last_to_mean"] == pytest.approx(1.0)
    assert vol["windows"]["5"]["reason"] == "insufficient_bars"


# ---- structural invariants -------------------------------------------------------------------


def test_document_is_byte_deterministic(tmp_path, capsys):
    px = _write_ohlcv(tmp_path / "px.csv")
    code = main(["describe", str(px)])
    first = capsys.readouterr().out
    assert code == 0
    code = main(["describe", str(px)])
    second = capsys.readouterr().out
    assert code == 0
    assert first == second  # same bytes, always — no RNG, no set/dict-order dependence


def test_document_carries_no_judgment_vocabulary(tmp_path, capsys):
    px = _write_ohlcv(tmp_path / "px.csv")
    y = _write_series(tmp_path / "y.csv", [2.0, 1.0, -1.0, 0.5, 1.5])
    _code, doc = _run(capsys, ["describe", str(px), str(y)])
    blob = json.dumps(doc).lower()
    for term in (
        "oversold",
        "overbought",
        "bullish",
        "bearish",
        "signal",
        "buy",
        "sell",
        "verdict",
        "undervalued",
        "overvalued",
    ):
        assert term not in blob, term


def _shape_tree(node):
    """The document's structure with every scalar collapsed to its kind — what must be
    IDENTICAL across file lengths if no per-bar payload leaks in."""
    if isinstance(node, dict):
        return {k: _shape_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_shape_tree(v) for v in node]
    if node is None:
        return "null"
    if isinstance(node, bool):
        return "bool"
    if isinstance(node, (int, float)):
        return "num"
    return "str"


def _max_list_len(node) -> int:
    if isinstance(node, dict):
        return max((_max_list_len(v) for v in node.values()), default=0)
    if isinstance(node, list):
        return max([len(node), *(_max_list_len(v) for v in node)])
    return 0


def test_bounded_output_document_is_independent_of_n_bars(tmp_path, capsys):
    # Two file lengths, both long enough to fill every default window: the documents must have
    # IDENTICAL structure (paths differ in values, never in shape), and no list anywhere may
    # exceed a small constant — a per-bar array leaking in fails both assertions at once.
    short = _write_ohlcv(tmp_path / "short.csv", n=300, seed=3)
    long = _write_ohlcv(tmp_path / "long.csv", n=900, seed=3)
    _code, doc_short = _run(capsys, ["describe", str(short)])
    _code, doc_long = _run(capsys, ["describe", str(long)])
    assert doc_short["profiles"][0]["n_bars"] == 300
    assert doc_long["profiles"][0]["n_bars"] == 900
    assert _shape_tree(doc_short) == _shape_tree(doc_long)
    assert _max_list_len(doc_long) <= 32
    assert _max_list_len(doc_short) == _max_list_len(doc_long)


# ---- pretty, roles ---------------------------------------------------------------------------


def test_pretty_changes_whitespace_only(tmp_path, capsys):
    px = _write_ohlcv(tmp_path / "px.csv", n=60)
    _code, plain = _run(capsys, ["describe", str(px)])
    code = main(["describe", str(px), "--pretty"])
    pretty_out = capsys.readouterr().out
    assert code == 0
    assert "\n  " in pretty_out  # actually indented
    assert json.loads(pretty_out) == plain


def test_describe_roles_stamped_identically_in_schema(tmp_path, capsys):
    px = _write_ohlcv(tmp_path / "px.csv", n=60)
    _code, doc = _run(capsys, ["describe", str(px)])
    roles = doc["describe_roles"]
    assert set(roles) == {"claim", "caveats", "scope_boundary"}
    assert "MEASURES NOTHING" in roles["claim"]
    assert "supports no thesis" in roles["claim"]
    assert set(roles["caveats"]) == {
        "percentile_rank",
        "dispersion",
        "volume",
        "drawdown",
        "windows",
    }
    assert "NOT valuation" in roles["caveats"]["percentile_rank"]
    assert "never annualized" in roles["caveats"]["dispersion"]
    assert "BARS" in roles["caveats"]["windows"]
    assert "pure observer of FILES" in roles["scope_boundary"]
    # the identical map rides `seikan schema` — one shape everywhere, like metric_roles
    _code, schema_doc = _run(capsys, ["schema"])
    assert schema_doc["describe_roles"] == roles
