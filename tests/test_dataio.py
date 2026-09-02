"""The strict-CSV front door (``dataio.read_strict_csv`` + ``sufficiency_check``).

Table-driven rejection matrix for untrusted external data, plus the warning (never-refuse) cases —
crash-sized moves and NaN holes are the research subject, not dirt.
"""

from __future__ import annotations

import pytest

from seikan.dataio import read_strict_csv, sufficiency_check

_GOOD = (
    "datetime,open,high,low,close,volume\n"
    "2021-01-01,100,101,99,100,1000\n"
    "2021-01-04,101,102,100,101,1100\n"
    "2021-01-05,102,103,101,102,900\n"
)


def _write(tmp_path, text, name="f.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def _run(tmp_path, text, **kw):
    df, rep = read_strict_csv(_write(tmp_path, text), **kw)
    return df, rep


def test_clean_ohlcv_passes_with_metadata(tmp_path):
    df, rep = _run(tmp_path, _GOOD)
    assert rep.ok and rep.shape == "ohlcv" and rep.n_rows == 3
    assert rep.start == "2021-01-01T00:00:00"
    assert df.index.name == "datetime" and list(df.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert df["open"].dtype == "float64"


def test_clean_series_passes(tmp_path):
    df, rep = _run(tmp_path, "datetime,pe\n2021-01-01,12.5\n2021-01-02,13.0\n")
    assert rep.ok and rep.shape == "series"
    assert list(df.columns) == ["pe"]


REJECTIONS = [
    ("empty_file", ""),
    ("header_only", "datetime,close\n"),
    ("duplicate_column", "datetime,close,Close\n2021-01-01,1,2\n"),
    ("bad_timestamp", "datetime,close\n01/02/2021,100\n01/03/2021,100\n"),  # US-style
    ("bad_timestamp", "datetime,close\n02.01.2021,100\n03.01.2021,100\n"),  # EU-style
    (
        "tz_aware_timestamp",
        "datetime,close\n2021-01-01T00:00:00+09:00,100\n2021-01-02T00:00:00+09:00,101\n",
    ),
    ("duplicate_timestamp", "datetime,close\n2021-01-01,100\n2021-01-01,101\n"),
    ("unsorted_timestamp", "datetime,close\n2021-01-02,100\n2021-01-01,101\n"),
    ("non_numeric_value", "datetime,close\n2021-01-01,abc\n2021-01-02,100\n"),
    ("non_numeric_value", 'datetime,close\n2021-01-01,"1,234"\n2021-01-02,100\n'),  # thousands sep
    ("no_value_columns", "datetime\n2021-01-01\n"),
    ("integrity", "datetime,open,high,low,close\n2021-01-01,100,90,110,105\n"),  # high < low
    ("integrity", "datetime,open,high,low,close\n2021-01-01,-1,1,-2,0.5\n"),  # non-positive price
    (
        "integrity",
        "datetime,open,high,low,close,volume\n2021-01-01,100,101,99,100,-5\n",
    ),  # neg volume
    ("integrity", "datetime,close\n2021-01-01,nan\n2021-01-02,nan\n"),  # entirely NaN column
    ("integrity", "datetime,close\n2021-01-01,inf\n2021-01-02,1\n"),  # non-finite
]


@pytest.mark.parametrize("code,text", REJECTIONS)
def test_rejections(tmp_path, code, text):
    df, rep = _run(tmp_path, text)
    assert df is None and not rep.ok
    assert code in {e["code"] for e in rep.errors}, rep.errors


def test_unpadded_iso_ordering_accepted(tmp_path):
    # `2021-1-1` is unambiguous (ISO year-month-day ordering is fixed) — the strictness guarantee
    # is "no ambiguous format is ever guessed", not padding pedantry.
    _df, rep = _run(tmp_path, "datetime,close\n2021-1-1,100\n2021-1-2,101\n")
    assert rep.ok and rep.start == "2021-01-01T00:00:00"


def test_error_names_file_column_and_line(tmp_path):
    _, rep = _run(tmp_path, "datetime,close\n2021-01-01,100\n2021-01-02,oops\n")
    err = next(e for e in rep.errors if e["code"] == "non_numeric_value")
    assert err["column"] == "close"
    assert err["examples"][0]["csv_line"] == 3
    assert err["examples"][0]["value"] == "oops"


def test_all_violations_accumulate_in_one_report(tmp_path):
    # A duplicate timestamp AND a non-numeric value must BOTH be reported, not just the first.
    _, rep = _run(tmp_path, "datetime,close\n2021-01-01,100\n2021-01-01,xyz\n")
    codes = {e["code"] for e in rep.errors}
    assert {"duplicate_timestamp", "non_numeric_value"} <= codes


def test_expected_shape_mismatch(tmp_path):
    _, rep = _run(tmp_path, "datetime,pe\n2021-01-01,12\n2021-01-02,13\n", expected_shape="ohlcv")
    assert "shape_mismatch" in {e["code"] for e in rep.errors}


def test_sha256_digest_is_the_data_identity(tmp_path):
    # Every readable file carries its raw-byte sha256 (the report's data identity)…
    _df, rep = _run(tmp_path, "datetime,pe\n2021-01-01,12.5\n2021-01-02,13.0\n")
    assert isinstance(rep.sha256, str) and len(rep.sha256) == 64
    assert rep.to_dict()["sha256"] == rep.sha256
    # …and one changed byte is a different identity.
    _df, rep2 = _run(tmp_path, "datetime,pe\n2021-01-01,12.5\n2021-01-02,13.1\n")
    assert rep2.sha256 != rep.sha256


def test_sha256_survives_content_refusal(tmp_path):
    # The digest is computed BEFORE validation: a content-invalid file still keeps its identity
    # (the report must say exactly WHICH bytes were refused).
    _df, rep = _run(tmp_path, "datetime,close\n2021-01-02,100\n2021-01-01,101\n")  # unsorted
    assert not rep.ok
    assert isinstance(rep.sha256, str) and len(rep.sha256) == 64


def test_file_is_read_exactly_once(tmp_path, monkeypatch):
    # Structural TOCTOU regression: digest, header sniff and parse all consume ONE buffer, so a
    # rewrite between stages cannot make the reported identity describe different bytes than the
    # ones measured. Counting opens is the only way to assert the property structurally.
    import builtins

    target = _write(tmp_path, _GOOD)
    real_open = builtins.open
    opens: list[str] = []

    def counting_open(file, *a, **kw):
        if str(file) == target:
            opens.append(str(file))
        return real_open(file, *a, **kw)

    monkeypatch.setattr(builtins, "open", counting_open)
    _df, rep = read_strict_csv(target)
    assert rep.ok
    assert len(opens) == 1, (
        f"file opened {len(opens)}× — the digest may not describe the parsed bytes"
    )


def test_sha256_equals_digest_of_the_parsed_bytes(tmp_path):
    import hashlib

    target = _write(tmp_path, _GOOD)
    _df, rep = read_strict_csv(target)
    with open(target, "rb") as fh:
        assert rep.sha256 == hashlib.sha256(fh.read()).hexdigest()


def test_sha256_none_when_unreadable(tmp_path):
    _df, rep = read_strict_csv(str(tmp_path / "nope.csv"))
    assert rep.sha256 is None
    assert rep.to_dict()["sha256"] is None


def test_missing_file(tmp_path):
    df, rep = read_strict_csv(str(tmp_path / "nope.csv"))
    assert df is None and {e["code"] for e in rep.errors} == {"file_missing"}


def test_utf8_bom_tolerated(tmp_path):
    p = tmp_path / "bom.csv"
    p.write_bytes(b"\xef\xbb\xbf" + _GOOD.encode())
    df, rep = read_strict_csv(str(p))
    assert rep.ok and "open" in df.columns


# ---- warnings: market-shaped anomalies never refuse -----------------------------------------


def test_nan_hole_warns_but_passes(tmp_path):
    df, rep = _run(tmp_path, "datetime,pe\n2021-01-01,12\n2021-01-02,\n2021-01-03,13\n")
    assert rep.ok
    assert "nan_fraction" in {w["code"] for w in rep.warnings}
    assert df["pe"].isna().sum() == 1


def test_crash_bar_warns_but_passes(tmp_path):
    text = (
        "datetime,open,high,low,close\n"
        "2021-01-01,100,101,99,100\n"
        "2021-01-04,100,101,30,40\n"  # -60% day: legitimate research subject
        "2021-01-05,40,41,39,40\n"
    )
    _df, rep = _run(tmp_path, text)
    assert rep.ok
    assert "large_move" in {w["code"] for w in rep.warnings}


def test_calendar_gap_warns_but_passes(tmp_path):
    text = (
        "datetime,pe\n"
        + "".join(f"2021-01-{d:02d},1\n" for d in range(1, 11))
        + "2021-12-01,1\n"  # ~11-month hole
    )
    _, rep = _run(tmp_path, text)
    assert rep.ok
    assert "calendar_gap" in {w["code"] for w in rep.warnings}


# ---- structural sufficiency -----------------------------------------------------------------


def test_sufficiency_refuses_when_no_observation_can_close():
    errors, warnings = sufficiency_check(10, [5, 20])  # needs 20 + 1 + 1 = 22 bars
    assert errors and errors[0]["code"] == "insufficient_common_index"
    assert not warnings


def test_sufficiency_warns_when_barely_enough():
    errors, warnings = sufficiency_check(25, 20)  # 4 anchor bars < 8
    assert not errors
    assert warnings and warnings[0]["code"] == "barely_sufficient"


def test_sufficiency_ok_on_ample_data():
    errors, warnings = sufficiency_check(500, [5, 20])
    assert not errors and not warnings


def test_partial_ohlc_series_read_refuses_as_ambiguous(tmp_path):
    # 2-3 of the four OHLC names without the full set: the series read would broadcast a bound
    # column into all four price slots while every OHLC invariant is skipped — refused, naming
    # the columns. A lone OHLC-named column stays an ordinary series file.
    partial = "datetime,open,close\n2021-01-01,100,101\n2021-01-04,101,102\n"
    df, rep = _run(tmp_path, partial)
    assert df is None and not rep.ok
    assert any("ambiguous shape" in e["message"] for e in rep.errors)
    lone = "datetime,close\n2021-01-01,100\n2021-01-04,101\n"
    df, rep = _run(tmp_path, lone)
    assert df is not None and rep.ok


def test_full_ohlcv_demanded_as_series_refuses_as_ambiguous(tmp_path):
    # `--shape series` over a genuine OHLCV file would dodge every OHLC invariant the same way.
    df, rep = _run(tmp_path, _GOOD, expected_shape="series")
    assert df is None and not rep.ok
    assert any("ambiguous shape" in e["message"] for e in rep.errors)
