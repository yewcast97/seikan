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

import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seikan.cli import MAX_DESCRIBE_WINDOWS, main
from seikan.dataio import read_strict_csv
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


def test_quotient_past_the_double_range_reads_the_exact_log_and_names_the_pct_refusal(
    tmp_path, capsys
):
    # The review's probe: 1e200 → 1e-200 is a quotient of 1e-400, which underflows to 0.0, so
    # math.log(b/a) raised a domain error and the whole command died at exit 4. ln(b/a) itself
    # is the finite −921.03…; the scale-safe log reads it as ln(b) − ln(a) exactly there.
    down = _write_series(tmp_path / "down.csv", [1e200, 1e-200])
    code, doc = _run(capsys, ["describe", str(down), "--windows", "1"])
    assert code == 0
    blocks = _series_profile(doc)
    ch1 = blocks["changes"]["1"]
    assert ch1["log"] == math.log(1e-200) - math.log(1e200) == -921.0340371976183
    assert ch1["pct"] == -1.0  # b/a underflows to 0.0, and −1.0 IS the correctly rounded pct
    assert ch1["diff"] == -1e200
    assert ch1["reason"] is None and ch1["ratio_reason"] is None
    assert blocks["full_sample"]["drawdown_pct"] == -1.0
    # Reversed, the quotient 1e400 OVERFLOWS: log still reads exactly, but pct = b/a − 1 has no
    # double to be — so it is a null that says why (it used to be an inf serialized to a
    # reasonless null), here and in every block that reads the same ratio.
    up = _write_series(tmp_path / "up.csv", [1e-200, 1e200])
    code, doc = _run(capsys, ["describe", str(up), "--windows", "1,2"])
    assert code == 0
    blocks = _series_profile(doc)
    ch1 = blocks["changes"]["1"]
    assert ch1["log"] == 921.0340371976183 and ch1["diff"] == 1e200
    assert ch1["pct"] is None and ch1["ratio_reason"] == "not_representable"
    assert ch1["reason"] is None  # the block did not refuse — only its pct read
    rp2 = blocks["range_position"]["2"]
    assert rp2["from_low"] == {"diff": 1e200, "pct": None}
    assert rp2["from_high"] == {"diff": 0.0, "pct": 0.0}
    assert rp2["reason"] is None and rp2["ratio_reason"] == "not_representable"
    fs = blocks["full_sample"]
    assert fs["runup_pct"] is None and fs["runup_diff"] == 1e200
    assert fs["drawdown_pct"] == 0.0
    assert fs["ratio_reason"] == "not_representable"


def test_subnormal_quotient_reads_the_exact_log_not_its_rounded_one(tmp_path, capsys):
    # 3e20 → 1e-300 is a quotient of 3.3e-321 — SUBNORMAL, so ln of it was finite and read
    # without complaint, but rounding to the subnormal grid had already cost it ~5e-4:
    # −737.92536 where ln(b) − ln(a) is −737.92584. The rule reads the exact value wherever the
    # quotient is not a normal double, the subnormal band included — changes and dispersion.
    y = _write_series(tmp_path / "y.csv", [3e20, 1e-300, 1e-300])
    code, doc = _run(capsys, ["describe", str(y), "--windows", "2"])
    assert code == 0
    blocks = _series_profile(doc)
    exact = math.log(1e-300) - math.log(3e20)
    assert exact == -737.9258420467627 != math.log(1e-300 / 3e20)
    ch2 = blocks["changes"]["2"]
    assert ch2["log"] == exact
    assert ch2["pct"] == -1.0 and ch2["ratio_reason"] is None
    # dispersion reads its changes through the ARRAY twin, so its expectation is np.log's
    np_exact = np.log(1e-300) - np.log(3e20)
    assert np_exact != np.log(1e-300 / 3e20)
    want = float(np.std([np_exact, np.log(1e-300 / 1e-300)], ddof=1))
    assert blocks["dispersion"]["2"]["log"] == want


def test_dispersion_pct_past_the_double_range_refuses_by_name_while_log_lives(tmp_path, capsys):
    # 1e-300 → 1e10 is a 1-bar ratio of 1e310 — a pct CHANGE with no double to be, the one thing
    # a pct std refuses on: it has none either (it warned and emitted a reasonless null), while
    # the log changes stay exact and their std finite.
    y = _write_series(tmp_path / "y.csv", [1e-300, 1e10, 1e10])
    code, doc = _run(capsys, ["describe", str(y), "--windows", "2"])
    assert code == 0
    blocks = _series_profile(doc)
    d2 = blocks["dispersion"]["2"]
    assert d2["pct"] is None and d2["ratio_reason"] == "not_representable"
    assert d2["reason"] is None
    log_changes = [np.log(1e10) - np.log(1e-300), np.log(1e10 / 1e10)]  # the array twin's log
    assert d2["log"] == float(np.std(log_changes, ddof=1))
    assert d2["diff"] == float(np.std(np.diff([1e-300, 1e10, 1e10]), ddof=1))
    ch2 = blocks["changes"]["2"]
    assert ch2["log"] == math.log(1e10) - math.log(1e-300)
    assert ch2["pct"] is None and ch2["ratio_reason"] == "not_representable"


def test_dispersion_pct_std_keeps_its_value_where_its_squares_leave_the_range(tmp_path, capsys):
    # Every 1-bar pct here is a double (~1e155, then 0.0), but np.std squares deviations of
    # ~5e154, which overflow: the pct std read inf and refused as `not_representable` while the
    # std itself (~7.07e154) is an ordinary double — and while a diff std over the same kind of
    # overflow was rescaled and computed. One rule for every dispersion std: scaled by a power
    # of two, every bit of the unscaled std, scaled; in range, np.std's, bit for bit.
    from seikan.describe import _dispersion_block

    # 1 → 3·2**600 is a 1-bar pct of exactly 3·2**600 (the −1 rounds away under its ulp), whose
    # square is past the range: the std of [3·2**600, 0.0] is the std of [3, 0], scaled
    d2 = _dispersion_block(np.array([1.0, 3.0 * 2.0**600, 3.0 * 2.0**600]), 2)
    assert d2["pct"] == float(np.std([3.0, 0.0], ddof=1)) * 2.0**600
    assert d2["reason"] is None and d2["ratio_reason"] is None
    walk = 100.0 * np.exp(np.cumsum(np.random.RandomState(5).normal(0.0, 0.02, 300)))
    for w in (2, 5, 21, 63, 252):
        tail = walk[-(w + 1) :]
        want = float(np.std(tail[1:] / tail[:-1] - 1.0, ddof=1))
        assert _dispersion_block(walk, w)["pct"] == want, w
    # end to end: computed, never refused, while every 1-bar ratio is a double — the alternating
    # file's pct changes are 1e200 and −1, a std of (1e200 + 1)/√3
    for name, levels, want in (
        ("z", [1e-150, 1e5, 1e5], 1e155 / math.sqrt(2.0)),
        ("alt", [1.0, 1e200, 1.0, 1e200, 1.0], 1e200 / math.sqrt(3.0)),
    ):
        w = len(levels) - 1
        y = _write_series(tmp_path / f"{name}.csv", levels)
        code, doc = _run(capsys, ["describe", str(y), "--windows", str(w)])
        assert code == 0
        blocks = _series_profile(doc)
        dw = blocks["dispersion"][str(w)]
        assert dw["pct"] == pytest.approx(want, rel=1e-15), name
        assert dw["reason"] is None and dw["ratio_reason"] is None
        assert dw["log"] == float(np.std(np.log(np.divide(levels[1:], levels[:-1])), ddof=1))


def test_levels_farther_apart_than_the_double_range_refuse_their_blocks_by_name(tmp_path, capsys):
    # Two FINITE levels of opposite sign can sit farther apart than the double range holds
    # (1.7e308 → −1.7e308 is a change of −3.4e308): `diff`, the algebra that rides every block
    # that did not refuse, has no double to be. A range distance's `diff` is never null, so
    # this file crashed describe at exit 4 on its own emitted contract; elsewhere the overflow
    # rode out as a reasonless null. Each such block now refuses WHOLE, by name.
    crash = _write_series(tmp_path / "crash.csv", [1.7e308, -1.7e308])
    code, doc = _run(capsys, ["describe", str(crash), "--windows", "1,2"])
    assert code == 0
    blocks = _series_profile(doc)
    refused_change = {
        "diff": None,
        "pct": None,
        "log": None,
        "reason": "not_representable",
        "ratio_reason": None,
    }
    assert blocks["changes"]["1"] == refused_change
    assert blocks["range_position"]["2"] == {
        "high": None,
        "low": None,
        "from_high": None,
        "from_low": None,
        "percentile_rank": None,
        "reason": "not_representable",
        "ratio_reason": None,
    }
    assert blocks["range_position"]["1"]["reason"] is None  # one bar: no distance to overflow
    fs = blocks["full_sample"]  # the distance reads refuse; the observed extremes stay
    assert fs["high"]["value"] == 1.7e308 and fs["low"]["value"] == -1.7e308
    assert fs["drawdown_diff"] is None and fs["runup_diff"] is None
    assert fs["drawdown_pct"] is None and fs["runup_pct"] is None
    assert fs["reason"] == "not_representable" and fs["ratio_reason"] is None
    # Only the windows that span the gap refuse: the 1-bar reads beside them still ride.
    span = _write_series(tmp_path / "span.csv", [-1e308, 1e308, 1e308])
    code, doc = _run(capsys, ["describe", str(span), "--windows", "1,2,3"])
    assert code == 0
    blocks = _series_profile(doc)
    assert blocks["changes"]["1"]["diff"] == 0.0 and blocks["changes"]["1"]["log"] == 0.0
    assert blocks["changes"]["2"] == refused_change
    assert blocks["dispersion"]["2"] == refused_change  # a 1-bar change of 2e308
    assert blocks["range_position"]["2"]["from_low"] == {"diff": 0.0, "pct": 0.0}
    assert blocks["range_position"]["3"]["reason"] == "not_representable"
    assert blocks["full_sample"]["reason"] == "not_representable"
    # A dispersion std can leave the range over changes that do not: ±1.7e308 has a std of
    # 2.4e308. The block refuses whole — the log std with it, finite as it is where 1e-300
    # keeps every level positive — while the changes it reads ride, each a double.
    for i, levels in enumerate(([0.0, 1.7e308, 0.0], [1e-300, 1.7e308, 1e-300])):
        y = _write_series(tmp_path / f"std{i}.csv", levels)
        code, doc = _run(capsys, ["describe", str(y), "--windows", "1,2"])
        assert code == 0
        blocks = _series_profile(doc)
        assert blocks["dispersion"]["2"] == refused_change
        assert blocks["changes"]["1"]["diff"] == levels[2] - levels[1]
        assert blocks["changes"]["1"]["reason"] is None
        assert blocks["full_sample"]["drawdown_diff"] == levels[2] - levels[1]


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


def test_ordinary_log_reads_are_the_direct_form_bit_for_bit(tmp_path, capsys):
    # The scale-safe log reads ln(b) − ln(a) ONLY where the quotient left the normal double
    # range: that form rounds differently from ln(b/a) on almost every ordinary pair, so on a
    # random walk every changes[N].log and dispersion[N].log carries the direct form's exact
    # bits — and the difference of logs must visibly disagree with them, or this pins nothing.
    rng = np.random.default_rng(11)
    walk = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 300)))
    y = _write_series(tmp_path / "y.csv", walk)
    windows = (2, 3, 5, 8, 13, 21, 34, 55, 89, 144)
    code, doc = _run(capsys, ["describe", str(y), "--windows", ",".join(map(str, windows))])
    assert code == 0
    blocks = _series_profile(doc)
    frame, _report = read_strict_csv(str(y))  # the levels describe read, bit for bit
    assert frame is not None
    x = frame["y"].to_numpy()
    n_change_moves = n_dispersion_moves = 0
    for w in windows:
        a, b, tail = float(x[-1 - w]), float(x[-1]), x[-(w + 1) :]
        log = blocks["changes"][str(w)]["log"]
        assert log == math.log(b / a)
        n_change_moves += log != math.log(b) - math.log(a)
        std = blocks["dispersion"][str(w)]["log"]
        assert std == float(np.std(np.log(tail[1:] / tail[:-1]), ddof=1))
        n_dispersion_moves += std != float(np.std(np.log(tail[1:]) - np.log(tail[:-1]), ddof=1))
    assert n_change_moves > 0 and n_dispersion_moves > 0


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


def test_volume_mean_of_an_overflowing_window_is_exact_never_a_zero_ratio(tmp_path, capsys):
    # Volumes alternating 1.7e308 / 1.0e308: the trailing SUM overflows, and the mean read inf —
    # null, beside a `last_to_mean` of exactly 0.0 (last / inf) and no reason at all. The mean of
    # finite volumes lies between their extremes; summed pre-scaled it is exact, and so the ratio.
    path = _write_flat_ohlcv(
        tmp_path / "v.csv", [10.0 + i for i in range(30)], volumes=[1.7e308, 1.0e308] * 15
    )
    code, doc = _run(capsys, ["describe", str(path), "--windows", "5"])
    assert code == 0
    w5 = doc["profiles"][0]["volume"]["windows"]["5"]
    assert w5["mean"] == pytest.approx(1.28e308, rel=1e-15)  # (3·1.0 + 2·1.7)e308 / 5
    assert w5["last_to_mean"] == pytest.approx(1.0 / 1.28, rel=1e-15)
    assert w5["reason"] is None and w5["ratio_reason"] is None
    # A CONSTANT overflowing window's mean IS its value: the pre-scaled mean of seven 1.7e308s
    # rounds an ulp off it on the way back (1.7000000000000001e308, a ratio of 0.999…), and is
    # held in [min, max] like any convex combination — exact, and the ratio exactly 1.0.
    path = _write_flat_ohlcv(tmp_path / "c.csv", [10.0 + i for i in range(30)], [1.7e308] * 30)
    code, doc = _run(capsys, ["describe", str(path), "--windows", "7"])
    assert code == 0
    w7 = doc["profiles"][0]["volume"]["windows"]["7"]
    assert w7["mean"] == 1.7e308 and w7["last_to_mean"] == 1.0


def test_an_all_equal_window_reads_its_own_mean_and_an_exact_zero_std(tmp_path, capsys):
    # The computed mean of bit-identical values carries its sum's rounding — 21 copies of
    # 1234.56, summed and divided back, read 1234.5600000000002 — so a held volume's ratio read
    # 0.9999999999999998, and the direct variance of identical changes was that residue, not
    # zero: the 63 log changes of 2**0..2**63 (ln 2 each) read a std of 2.238e-16. Constancy is
    # decided before either is computed; a varying window keeps the direct form's bits.
    from seikan.describe import _dispersion_block

    path = _write_flat_ohlcv(tmp_path / "v.csv", [10.0 + i for i in range(30)], [1234.56] * 30)
    code, doc = _run(capsys, ["describe", str(path), "--windows", "21"])
    assert code == 0
    w21 = doc["profiles"][0]["volume"]["windows"]["21"]
    assert w21["mean"] == 1234.56 and w21["last_to_mean"] == 1.0
    doubling = np.array([2.0**k for k in range(64)])
    assert float(np.std(np.log(doubling[1:] / doubling[:-1]), ddof=1)) > 0.0  # the residue
    d63 = _dispersion_block(doubling, 63)
    assert d63["log"] == 0.0 and d63["pct"] == 0.0 and d63["reason"] is None
    assert d63["diff"] == float(np.std(np.diff(doubling), ddof=1))
    # end to end, on levels every parser reads exactly (integers under 2**53): 30 changes of
    # ln 2 carry the residue too
    assert float(np.std(np.full(30, math.log(2.0)), ddof=1)) > 0.0
    y = _write_series(tmp_path / "y.csv", doubling[:31])
    code, doc = _run(capsys, ["describe", str(y), "--windows", "30"])
    assert code == 0
    d30 = _series_profile(doc)["dispersion"]["30"]
    assert d30["log"] == 0.0 and d30["pct"] == 0.0 and d30["reason"] is None


def test_dispersion_diff_std_keeps_its_value_where_its_squares_leave_the_range(tmp_path, capsys):
    # A VARYING series whose squared 1-bar changes leave the double range: past it the std read
    # inf — a null beside finite endpoints, with no reason (2**520, 1e160); below it they lost
    # their bits (2**-536 read 9.1647e-162 where the std is 9.1106e-162) or vanished into the 0.0
    # only an all-equal series owns (1e-170, 2**-1000). Scaled by a power of two the std is
    # computed in range — every bit of the unscaled std, scaled — and in range it is np.std's.
    # (The exact reads call the block directly: the CSV parse itself may land a 17-digit level
    # one ulp off its repr.)
    from seikan.describe import _dispersion_block

    levels = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 7.0])
    unscaled = float(np.std(np.diff(levels), ddof=1))
    for k in (-500, 520, -536, -1000):
        assert _dispersion_block(levels * 2.0**k, 5)["diff"] == unscaled * 2.0**k, k
    # an all-equal change keeps its exact 0.0 down there, and an ordinary walk np.std's bits
    assert _dispersion_block(np.arange(6.0) * 2.0**-1000, 5)["diff"] == 0.0
    # One step of the smallest subnormal, then a hold: the changes [u, 0, 0, 0, 0] vary, but
    # their std (~0.447u) rounds to 0.0 on the subnormal grid — the all-equal answer. It floors
    # at the smallest subnormal instead, as the kernels' lifted std does.
    u = 5e-324
    assert _dispersion_block(np.array([0.0, u, u, u, u, u]), 5)["diff"] == u
    assert _dispersion_block(np.full(6, u), 5)["diff"] == 0.0
    walk = 100.0 + np.random.RandomState(3).randn(300).cumsum()
    for w in (2, 5, 21, 63, 252):
        tail = walk[-(w + 1) :]
        assert _dispersion_block(walk, w)["diff"] == float(np.std(np.diff(tail), ddof=1)), w
    # end to end: never a null beside finite endpoints, never a zero for a varying series
    for i, scale in enumerate((1e160, 1e-170)):
        y = _write_series(tmp_path / f"y{i}.csv", levels * scale)
        code, doc = _run(capsys, ["describe", str(y), "--windows", "5"])
        d5 = _series_profile(doc)["dispersion"]["5"]
        assert code == 0 and d5["reason"] is None
        assert d5["diff"] == pytest.approx(unscaled * scale, rel=1e-14), scale


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


def _imported_modules(module) -> set[str]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    return names


def test_describe_shares_the_log_ratio_rule_through_a_leaf_never_the_engine():
    # The containment claim, pinned on describe's OWN imports: it reads the engine's log-ratio
    # rule from the leaf `seikan.numeric`, never from compiler/ (nor analysis/, gate/, numba) —
    # and the leaf stays a leaf, so sharing the rule cannot drag the engine in.
    import seikan.describe
    import seikan.numeric

    engine = ("seikan.compiler", "seikan.analysis", "seikan.gate", "numba")
    assert "seikan.numeric" in _imported_modules(seikan.describe)
    assert not {m for m in _imported_modules(seikan.describe) if m.startswith(engine)}
    # `math` + `numpy` and nothing else — exactly what its module note and CONTRACT's module map
    # say the leaf imports.
    assert _imported_modules(seikan.numeric) <= {"__future__", "math", "numpy", "numpy.typing"}


def test_the_normal_range_edge_has_one_definition():
    # The smallest normal double decides every out-of-range rule — the log ratio's quotient, the
    # kernels' variance tests, the pool statistics', CSCV's split moments, describe's std — and it
    # is DEFINED once, in the leaf: each layer reads the same object, so no copy can drift.
    import sys

    import seikan.describe
    import seikan.numeric
    from seikan.analysis import _cscv, _pools
    from seikan.compiler import nb

    assert seikan.numeric.MIN_NORMAL == sys.float_info.min == 2.0**-1022
    for module in (nb, _pools, _cscv, seikan.describe):
        assert module.MIN_NORMAL is seikan.numeric.MIN_NORMAL, module.__name__
    src = Path(seikan.numeric.__file__).parent
    defining = sorted(
        str(p.relative_to(src))
        for p in src.rglob("*.py")
        if any(s in p.read_text(encoding="utf-8") for s in (".tiny", "float_info.min"))
    )
    assert defining == ["numeric.py"]


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
