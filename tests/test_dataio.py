"""The strict-CSV front door (``dataio.read_strict_csv`` + ``sufficiency_check``).

Table-driven rejection matrix for untrusted external data, plus the warning (never-refuse) cases —
crash-sized moves and NaN holes are the research subject, not dirt.
"""

from __future__ import annotations

import csv
import io

import numpy as np
import pandas as pd
import pytest

from seikan import dataio
from seikan.dataio import _CSV_FIELD_LIMIT, MAX_EXAMPLES, read_strict_csv, sufficiency_check

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
    # record shape: every record carries exactly the header's field count, more OR fewer refuses
    (
        "field_count_mismatch",
        "datetime,close\n2021-01-01,2021-01-04,100\n2021-01-02,2021-01-05,101\n",
    ),  # header + 1 on EVERY record: pandas read an implicit index and loaded it clean, shifted
    ("field_count_mismatch", "datetime,close\n2021-01-01,100,\n2021-01-02,101,\n"),  # trailing
    ("field_count_mismatch", "datetime,a,b\n2021-01-01,1,2\n2021-01-02,3\n"),  # short: was padded
    ("field_count_mismatch", "datetime,close\n2021-01-01,100\n2021-01-02,101,7\n"),  # one ragged
    ("bad_timestamp", "datetime,close\n01/02/2021,100\n01/03/2021,100\n"),  # US-style
    ("bad_timestamp", "datetime,close\n02.01.2021,100\n03.01.2021,100\n"),  # EU-style
    (
        "bad_timestamp",
        "datetime,close\n2021-01-01T00:00:00.0000000001,100\n2021-01-02,101\n",
    ),  # a digit past the nanosecond: truncated, it read as midnight
    (
        "tz_aware_timestamp",
        "datetime,close\n2021-01-01T00:00:00+09:00,100\n2021-01-02T00:00:00+09:00,101\n",
    ),
    ("duplicate_timestamp", "datetime,close\n2021-01-01,100\n2021-01-01,101\n"),
    ("unsorted_timestamp", "datetime,close\n2021-01-02,100\n2021-01-01,101\n"),
    ("non_numeric_value", "datetime,close\n2021-01-01,abc\n2021-01-02,100\n"),
    ("non_numeric_value", 'datetime,close\n2021-01-01,"1,234"\n2021-01-02,100\n'),  # thousands sep
    ("non_numeric_value", "datetime,close\n2021-01-01,1_000\n2021-01-02,100\n"),  # float() reads it
    ("non_numeric_value", "datetime,close\n2021-01-01,١٢٣\n2021-01-02,100\n"),  # Arabic-Indic
    ("non_numeric_value", "datetime,close\n2021-01-01,１２３\n2021-01-02,100\n"),  # full-width
    ("non_numeric_value", "datetime,close\n2021-01-01,0x10\n2021-01-02,100\n"),  # hex
    ("non_numeric_value", "datetime,close\n2021-01-01,-nan\n2021-01-02,100\n"),  # not a marker
    ("non_numeric_value", "datetime,close\n2021-01-01,1e\n2021-01-02,100\n"),  # no exponent
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


def test_a_fractional_digit_past_the_nanosecond_refuses_and_trailing_zeros_admit(tmp_path):
    # The parsed clock is nanosecond resolution and the ISO parser TRUNCATES further digits, so
    # `.0000000001` read as midnight — admitted as another instant, or as a duplicate of the bar
    # stamped there. A nonzero digit past the ninth refuses, naming the cause; zeros past it name
    # the same instant and admit.
    df, rep = _run(
        tmp_path,
        "datetime,close\n2021-01-01T00:00:00,100\n2021-01-01T00:00:00.0000000001,101\n"
        "2021-01-01T00:00:00.9999999999,102\n",
    )
    assert df is None and [e["code"] for e in rep.errors] == ["bad_timestamp"]
    (err,) = rep.errors
    assert "past the nanosecond" in err["message"] and "2 timestamp(s)" in err["message"]
    assert [x["csv_line"] for x in err["examples"]] == [3, 4]
    df, rep = _run(
        tmp_path,
        "datetime,close\n2021-01-01T00:00:00.1234567890,100\n"
        "2021-01-01T00:00:00.123456790000,101\n2021-01-01T00:00:01.5,102\n",
    )
    assert rep.ok and df is not None
    assert list(df.index) == [
        pd.Timestamp("2021-01-01T00:00:00.123456789"),
        pd.Timestamp("2021-01-01T00:00:00.12345679"),
        pd.Timestamp("2021-01-01T00:00:01.5"),
    ]


def test_value_cells_read_correctly_rounded(tmp_path):
    # pandas' parser is not correctly rounded: it read 0.00000000000000001 as 0.0 and missed by an
    # ulp on nearly half of all 17-significant-digit strings — the form repr and every
    # pandas-written CSV use — so a file read back did not hold the values written to it. Every
    # cell now reads as the double nearest its decimal, which is float()'s value.
    rng = np.random.default_rng(3)
    vals = np.concatenate(
        [rng.standard_normal(2000) * 10.0 ** rng.integers(-30, 30, 2000), rng.random(2000)]
    )
    written = [repr(float(v)) for v in vals] + [f"{v:.15f}" for v in vals[:1000]]
    written += ["0.00000000000000001", "123456789.123456789", "1.e5", "+.5", "-0", "1E+05"]
    written += ["1e-400", "7e00000000000000000000000001", "000.12345678901234567890123"]
    idx = pd.date_range("2021-01-01", periods=len(written), freq="1min")
    lines = ["datetime,y"] + [f"{t.isoformat()},{w}" for t, w in zip(idx, written, strict=True)]
    df, rep = _run(tmp_path, "\n".join(lines) + "\n")
    assert rep.ok and df is not None
    got = df["y"].to_numpy()
    want = np.array([float(w) for w in written])
    assert np.array_equal(got, want)
    assert got[: len(vals)].tolist() == vals.tolist()  # every repr round-trips, bit for bit
    assert got[len(vals) + 1000] == 1e-17 and np.signbit(got[len(vals) + 1004])
    # the old reading missed on this very file
    assert (pd.to_numeric(pd.Series(written)).to_numpy() != want).sum() > 1000
    # inf/infinity in any case still parse, and still refuse as non-finite — never as text
    for cell in ("inf", "-Infinity", "INF", "1e400"):
        df, rep = _run(tmp_path, f"datetime,close\n2021-01-01,{cell}\n2021-01-02,1\n")
        assert df is None and [e["code"] for e in rep.errors] == ["integrity"], cell
    # the only missing markers are an empty cell and nan, in any case
    df, rep = _run(tmp_path, "datetime,y\n2021-01-01,\n2021-01-02,NaN\n2021-01-03, 2.5 \n")
    assert rep.ok and df is not None
    assert np.isnan(df["y"].iloc[0]) and np.isnan(df["y"].iloc[1]) and df["y"].iloc[2] == 2.5


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
    # Structural TOCTOU regression: digest, header sniff, record-shape scan, parse and its
    # cross-check all consume ONE buffer, so a rewrite between stages cannot make the reported
    # identity describe different bytes than the ones measured. Counting opens is the only way to
    # assert the property structurally.
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


def test_a_reader_measures_one_snapshot_of_each_path(tmp_path):
    # A load binding one file under several keys reads it through one StrictCsvReader: every
    # planned read of the path parses the FIRST read's bytes — a rewrite in between changes
    # nothing they measure or the digest they report — each under its own role and expected
    # shape, so a series file bound as a target admits while the same bytes bound where an
    # OHLCV file is expected refuse. Past its plan a path is read afresh: no snapshot outlives
    # the reads it was kept for.
    path = _write(tmp_path, "datetime,pe\n2021-01-01,12.5\n2021-01-02,13.0\n")
    reader = dataio.StrictCsvReader([path, path, path])
    df1, rep1 = reader.read(path, role="target:A")
    _write(tmp_path, "datetime,pe\n2021-01-01,99.0\n2021-01-02,98.0\n")
    df2, rep2 = reader.read(path, role="target:B")
    _df, rep3 = reader.read(path, role="benchmark", expected_shape="ohlcv")
    assert rep1.ok and rep2.ok and df1 is not None and df2 is not None
    pd.testing.assert_frame_equal(df1, df2)
    assert rep1.sha256 == rep2.sha256 == rep3.sha256
    assert not rep3.ok and "shape_mismatch" in {e["code"] for e in rep3.errors}
    assert (rep1.role, rep2.role, rep3.role) == ("target:A", "target:B", "benchmark")
    df4, rep4 = reader.read(path)
    assert rep4.sha256 != rep1.sha256 and df4 is not None and df4["pe"].iloc[0] == 99.0


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


@pytest.mark.parametrize(
    "n_before,bom,eol",
    [
        (10, b"", "\n"),
        (10, b"\xef\xbb\xbf", "\r\n"),
        (5000, b"", "\r"),
        (25000, b"\xef\xbb\xbf", "\n"),
    ],
)  # inside the decoder's first 8 KiB chunk, past it, and past pandas' 256 KiB read buffer
def test_an_undecodable_byte_is_named_by_its_file_offset_and_line(tmp_path, n_before, bom, eol):
    # A cp1252 export whose first accented byte sits deep in the file: the streaming decoder counts
    # its position from the start of its 8 KiB read chunk (and from past a BOM), pandas' from its
    # 256 KiB buffer, so the codec's own "position" named some other byte. The refusal names the
    # byte's offset in the FILE and the csv line it sits on.
    rows = "".join(f"2021-01-01,{i}{eol}" for i in range(n_before)).encode()
    blob = (
        bom + f"datetime,close{eol}".encode() + rows + f"2021-02-01,caf\xe9{eol}".encode("cp1252")
    )
    p = tmp_path / "f.csv"
    p.write_bytes(blob + rows)
    df, rep = read_strict_csv(str(p))
    assert df is None
    (err,) = rep.errors
    assert err["code"] == "unreadable"
    assert err["message"] == (
        f"{p}: 'utf-8' codec can't decode byte 0xe9 in position {len(blob) - 1 - len(eol)}: "
        "invalid continuation byte"
    )
    assert err["examples"] == [{"csv_line": n_before + 2}]


# ---- record shape: exactly one field per header column ---------------------------------------


def test_header_plus_one_on_every_record_refuses_instead_of_shifting(tmp_path):
    # Every record one field longer than the header: pandas read the first field as an implicit
    # index, so `datetime` read the SECOND date of each record and the first was discarded — the
    # file loaded clean and wrong. Now nothing loads, and every offending line is named.
    text = "datetime,close\n2021-01-01,2021-01-04,100\n2021-01-02,2021-01-05,101\n"
    df, rep = _run(tmp_path, text)
    assert df is None and rep.n_rows is None and rep.shape is None
    (err,) = rep.errors
    assert err["code"] == "field_count_mismatch"
    assert "2 record(s) do not carry the header's 2 field(s) (found 3)" in err["message"]
    assert err["examples"] == [{"csv_line": 2, "value": "3"}, {"csv_line": 3, "value": "3"}]


def test_one_ragged_record_is_a_field_count_mismatch_not_unreadable(tmp_path):
    # pandas refuses a lone ragged record itself, but as a tokenizer error filed under
    # `unreadable`; the record-shape stage reaches it first, under the precise code.
    _, rep = _run(tmp_path, "datetime,close\n2021-01-01,100\n2021-01-02,101,7\n2021-01-03,102\n")
    (err,) = rep.errors
    assert err["code"] == "field_count_mismatch"
    assert err["examples"] == [{"csv_line": 3, "value": "3"}]


def test_field_count_mismatch_counts_every_offender_and_caps_what_it_names(tmp_path):
    # Seven offenders with six distinct counts, more AND fewer than the header's three: the total
    # is exact, the examples are the first MAX_EXAMPLES in file order, the counts are the first
    # MAX_EXAMPLES in sorted order — a garbage file still gets a bounded message.
    text = (
        "datetime,a,b\n"
        "2021-01-01,1,2\n"
        "2021-01-02,1\n"  # 2 fields
        "2021-01-03,1,2,3\n"  # 4
        "2021-01-04\n"  # 1
        "2021-01-05,1,2\n"
        "2021-01-06,1,2,3,4\n"  # 5
        "2021-01-07,1,2,3,4,5\n"  # 6
        "2021-01-08,1,2,3,4,5,6\n"  # 7
        "2021-01-09,1\n"  # 2
    )
    _, rep = _run(tmp_path, text)
    (err,) = rep.errors
    assert err["code"] == "field_count_mismatch"
    assert (
        "7 record(s) do not carry the header's 3 field(s) (found 1, 2, 4, 5, 6, …)"
        in (err["message"])
    )
    assert MAX_EXAMPLES == 5
    assert err["examples"] == [
        {"csv_line": 3, "value": "2"},
        {"csv_line": 4, "value": "4"},
        {"csv_line": 5, "value": "1"},
        {"csv_line": 7, "value": "5"},
        {"csv_line": 8, "value": "6"},
    ]


def test_quoted_delimiter_and_line_break_stay_one_field(tmp_path):
    # csv.reader parity with pandas' tokenizer: a quoted comma or line break is part of ONE field,
    # so the stage counts exactly the fields pandas will read — and admits.
    text = 'datetime,"yield, 10y","spread\n(bp)"\n2021-01-01,"1.5",120\n"2021-01-04",1.6,"118"\n'
    df, rep = _run(tmp_path, text)
    assert rep.ok, rep.errors
    assert list(df.columns) == ["yield, 10y", "spread\n(bp)"]
    assert df["yield, 10y"].tolist() == [1.5, 1.6]
    assert df["spread\n(bp)"].tolist() == [120.0, 118.0]
    # an offender is named by the physical line its record STARTS on: the quoted break spans the
    # record over lines 3-4, and the clean record after it is line 5
    _, rep = _run(tmp_path, 'datetime,close\n2021-01-01,1\n2021-01-02,"2\n",9\n2021-01-03,3\n')
    (err,) = rep.errors
    assert err["code"] == "field_count_mismatch"
    assert err["examples"] == [{"csv_line": 3, "value": "3"}]


def test_blank_lines_skip_with_pandas_parity(tmp_path):
    # pandas' skip_blank_lines drops a line of nothing but spaces and tabs, before the header too,
    # and the stage drops exactly those: interior (and leading, and trailing) blanks still admit.
    text = "\n \ndatetime,close\n2021-01-01,100\n\n2021-01-02,101\n   \n\t\r\n2021-01-03,102\n \t "
    df, rep = _run(tmp_path, text)
    assert rep.ok, rep.errors
    assert df["close"].tolist() == [100.0, 101.0, 102.0]
    # a QUOTED blank is a record to pandas (one field — a short row it would pad), so it is one
    # here too, named on its own line
    _, rep = _run(tmp_path, 'datetime,close\n2021-01-01,100\n""\n2021-01-02,101\n')
    (err,) = rep.errors
    assert err["code"] == "field_count_mismatch"
    assert err["examples"] == [{"csv_line": 3, "value": "1"}]


def test_header_is_the_first_record_pandas_reads(tmp_path):
    # Leading blank lines skip before the header, so the duplicate-name sniff inspects the header
    # pandas actually uses: it used to inspect the blank line, and the case-folded duplicate went
    # on to crash the read.
    df, rep = _run(tmp_path, "\n  \ndatetime,close,Close\n2021-01-01,1,2\n")
    assert df is None and [e["code"] for e in rep.errors] == ["duplicate_column"]
    # with nothing but blank lines there is no header for pandas either ("No columns to parse
    # from file", which was `unreadable`): empty of records, and said so apart from a 0-byte file
    for text, says in (
        ("", "file is empty"),
        ("\n  \n\t\r\n", "file holds only blank lines, no header"),
    ):
        path = _write(tmp_path, text)
        df, rep = read_strict_csv(path)
        assert df is None and rep.errors == [{"code": "empty_file", "message": f"{path}: {says}"}]


def test_a_field_past_the_csv_field_limit_is_unreadable_not_a_crash(tmp_path):
    # csv.reader refuses a field longer than the limit the scan pins (_CSV_FIELD_LIMIT); in the
    # header or in a record, that is an unreadable file naming the line the record starts on, never
    # an uncaught csv.Error.
    big = "9" * (_CSV_FIELD_LIMIT + 1)
    for line, text in ((1, f"datetime,{big}\n2021-01-01,1\n"), (2, f"datetime,close\n1,{big}\n")):
        df, rep = _run(tmp_path, text)
        assert df is None and [e["code"] for e in rep.errors] == ["unreadable"]
        assert "field larger than field limit" in rep.errors[0]["message"]
        assert rep.errors[0]["examples"] == [{"csv_line": line}]


def test_an_unclosed_quote_is_named_by_the_line_it_opens_on(tmp_path):
    # csv.reader closes an unterminated quote at EOF, so the quote reads the rest of the file as
    # one field. A short rest is one record short of fields, named where the quote opens (pandas
    # said only "EOF inside string starting at row 2", counting lines from 0); a rest past the
    # field limit is unreadable, again named by that line and pointing at the quote.
    tail = "".join(f"2021-02-{d:02d},{d}\n" for d in range(1, 29))
    _, rep = _run(tmp_path, 'datetime,close\n2021-01-01,1\n"2021-01-02,2\n' + tail)
    (err,) = rep.errors
    assert err["code"] == "field_count_mismatch"
    assert err["examples"] == [{"csv_line": 3, "value": "1"}]
    past = tail * (_CSV_FIELD_LIMIT // len(tail) + 1)
    _, rep = _run(tmp_path, 'datetime,close\n2021-01-01,1\n"2021-01-02,2\n' + past)
    (err,) = rep.errors
    assert err["code"] == "unreadable"
    assert err["examples"] == [{"csv_line": 3}]
    assert "the record starting at csv line 3" in err["message"]
    assert "a quote that never closes" in err["message"]


def test_the_field_limit_is_pinned_not_read_from_the_process(tmp_path, monkeypatch):
    # csv.field_size_limit() is interpreter-global, and the scan used to read whatever it held:
    # once a co-imported package raised it (scipy's ARFF reader does), an unclosed quote read as a
    # field_count_mismatch and an over-long decimal or header name loaded; a lowered one refused
    # a clean file. The scan pins the stdlib default for read_strict_csv's stages, so the bytes
    # alone decide, and hands the caller's limit back on every exit: admitted, refused, raised.
    # The lone-\r file is clean too, but its endings send the tokenizer cross-check (stage 5) back
    # through the record scan, so a pin released before that stage refuses it under the lowered
    # limit — every stage that scans is covered, not only the two ahead of the parse.
    tail = "".join(f"2021-02-{d:02d},{d}\n" for d in range(1, 29))
    past = tail * (_CSV_FIELD_LIMIT // len(tail) + 1)  # what the unclosed quote reads as one field
    texts = {
        "unclosed": 'datetime,close\n2021-01-01,1\n"2021-01-02,2\n' + past,
        "decimal": f"datetime,close\n2021-01-01,1.{'0' * _CSV_FIELD_LIMIT}\n2021-01-02,2\n",
        "header": f"datetime,{'c' * (_CSV_FIELD_LIMIT + 1)}\n2021-01-01,1\n2021-01-02,2\n",
        "clean": "datetime,close\n2021-01-01,1\n2021-01-02,2\n",
        "lone_cr": "datetime,close\r2021-01-01,1\r2021-01-02,2\r",
    }

    def verdicts():
        out, limit = {}, csv.field_size_limit()
        for name, text in texts.items():
            df, rep = _run(tmp_path, text)
            assert csv.field_size_limit() == limit, name  # the caller's limit, handed back
            out[name] = (None if df is None else df.reset_index().to_dict("list"), rep.to_dict())
        return out

    ambient = csv.field_size_limit()
    pinned = verdicts()
    assert {name: [e["code"] for e in rep["errors"]] for name, (_, rep) in pinned.items()} == {
        "unclosed": ["unreadable"],
        "decimal": ["unreadable"],
        "header": ["unreadable"],
        "clean": [],
        "lone_cr": [],
    }
    try:
        for limit in (2**31 - 1, 4):  # raised (the most a C long holds on every platform), lowered
            csv.field_size_limit(limit)
            assert verdicts() == pinned, limit
        # an exception raised through the stages hands the limit back too
        monkeypatch.setattr(dataio, "_parse_frame", lambda *_: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            _run(tmp_path, texts["clean"])
        assert csv.field_size_limit() == 4
    finally:
        csv.field_size_limit(ambient)


def test_a_doubled_bom_is_dropped_as_pandas_drops_it(tmp_path):
    # The codec drops one BOM and pandas' tokenizer the next, so a doubled BOM ahead of a blank
    # line leaves a line pandas skips as blank — and the scan agrees, finding the same header.
    for text in ("\ufeff\ufeff\ndatetime,close\n", "\ufeff\ufeffdatetime,close\n"):
        df, rep = _run(tmp_path, text + "2021-01-01,1\n2021-01-02,2\n")
        assert rep.ok, rep.errors
        assert list(df.columns) == ["close"] and df["close"].tolist() == [1.0, 2.0]


def test_a_record_pandas_reads_differently_after_a_bare_carriage_return_refuses(tmp_path):
    # Past a blank line ended by a LONE \r, pandas' tokenizer swallows the delimiter that opens the
    # next record and pads its end: `,2022,5` read as `2022,5,` — a record the scan had counted
    # right (datetime empty, a=2022, b=5) loaded as datetime=2022, a=5, b=NaN, exit 0. The frame
    # is cross-checked against the scan wherever a lone \r occurs, and the line is named.
    for blank in ("\r", " \t\r"):
        for eol in ("\n", "\r\n"):
            text = f"datetime,a,b{eol}2021-01-01,1,2{eol}{blank},2022,5{eol}"
            df, rep = _run(tmp_path, text)
            assert df is None, (blank, eol)
            (err,) = rep.errors
            assert err["code"] == "unreadable"
            assert err["examples"] == [{"csv_line": 4}]
    # a space-led record after a lone \r: pandas re-read the header as a data row (refused, but as
    # a bad timestamp `datetime`); now the line the readings part on is named
    _, rep = _run(tmp_path, "datetime,close\r 2021-01-01,1\r")
    (err,) = rep.errors
    assert err["code"] == "unreadable" and err["examples"] == [{"csv_line": 2}]


def test_a_nul_byte_pandas_cuts_a_field_at_refuses(tmp_path):
    # pandas' tokenizer ends a field at a NUL byte: `1\x002` loaded as close=1.0, exit 0.
    df, rep = _run(tmp_path, "datetime,close\n2021-01-01,1\x002\n2021-01-02,3\n")
    assert df is None
    (err,) = rep.errors
    assert err["code"] == "unreadable" and err["examples"] == [{"csv_line": 2}]


def test_a_quote_pandas_opens_at_its_read_buffer_edge_refuses(tmp_path):
    # pandas' tokenizer reads 256 KiB at a time, and a space-led line it entered as a possible
    # blank backs up only to the start of the current buffer: with every leading space before the
    # edge and a quote opening the next buffer, it opens a QUOTED field where csv.reader — and
    # pandas anywhere else in the file — reads a literal quote. `   "<date>",5` loaded
    # quote-stripped, a clean timestamp at exit 0. A space-led quote anywhere now joins the
    # cross-check and the disagreement refuses; one byte off the edge the readings agree, and the
    # same line refuses as the bad timestamp it is, exactly as before.
    edge = 256 * 1024
    head = "datetime,close\n"
    dates = pd.date_range("1990-01-01", periods=20200, freq="D").strftime("%Y-%m-%d")
    for shift, code in ((0, "unreadable"), (1, "bad_timestamp")):
        n, extra = divmod(edge + shift - 3 - len(head), 13)  # 13-byte `<date>,1\n` filler rows
        rows = [f"{d},1\n" for d in dates[:n]]
        rows[-1] = f"{dates[n - 1]},1{'0' * extra}\n"
        text = head + "".join(rows) + f'   "{dates[n]}",5\n{dates[n + 1]},6\n'
        assert text.index('"') == edge + shift
        df, rep = _run(tmp_path, text)
        assert df is None, shift
        (err,) = rep.errors
        assert err["code"] == code
        assert err["examples"][0]["csv_line"] == n + 2


def test_a_bom_pandas_drops_at_its_read_buffer_edge_refuses(tmp_path):
    # Until its first line closes, pandas' tokenizer drops a BOM that opens a 256 KiB read buffer,
    # as it drops one opening the file. With a U+FEFF at character 262144 of an over-long header,
    # `\ufeffclose` — distinct from `close` to the duplicate check — reached pandas as a second
    # `close`: `close` loaded the other column's values and the file's own `close` became
    # `close.1`, exit 0. A BOM past the file's start now joins the cross-check and the header
    # disagreement refuses; one character off the edge the readings agree and the file loads
    # exactly as before, the BOM kept in its name.
    edge = 256 * 1024
    pads = ["a" * 87378, "b" * 87377, "c" * 87377]  # each under the csv field limit (131072)
    for lead in ("", "\ufeff"):  # the codec drops a leading BOM before the tokenizer counts
        for shift in (0, 1):
            head = ",".join(["datetime", *pads[:2], pads[2] + "c" * shift]) + ","
            text = head + "\ufeffclose,close\n2021-01-01,1,2,3,4,5\n2021-01-04,1,2,3,4,6\n"
            assert text.index("\ufeff") == edge + shift
            df, rep = _run(tmp_path, lead + text)
            if shift == 0:
                assert df is None, lead
                (err,) = rep.errors
                assert err["code"] == "unreadable" and err["examples"] == [{"csv_line": 1}]
            else:
                assert rep.ok, rep.errors
                assert list(df.columns)[-2:] == ["\ufeffclose", "close"]
                assert df["\ufeffclose"].tolist() == [4.0, 4.0]
                assert df["close"].tolist() == [5.0, 6.0]


def test_bare_carriage_return_files_pandas_reads_faithfully_still_admit(tmp_path):
    # The cross-check refuses a disagreement, never the line ending: a classic-Mac file with blank
    # and whitespace lines and quoted cells, and an empty first cell after a record's lone \r
    # (pandas reads that one right), load exactly as before.
    df, rep = _run(tmp_path, 'datetime,close\r2021-01-01,1\r\r2021-01-02,"2"\r \r2021-01-03,3\r')
    assert rep.ok, rep.errors
    assert df["close"].tolist() == [1.0, 2.0, 3.0]
    df, rep = _run(tmp_path, "a,datetime\r1,2021-01-01\r,2021-01-02\r")
    assert rep.ok, rep.errors
    assert df["a"].tolist()[0] == 1.0 and np.isnan(df["a"].tolist()[1])


def test_record_scan_matches_pandas_tokenizer_on_random_text():
    # The stage's premise, checked rather than asserted: wherever every record carries the
    # header's field count, pandas reads exactly those records — no implicit index, no padding —
    # or refuses an unterminated quote (which csv.reader closes at EOF). Random text over every
    # character the dialect, the codec or the tokenizer treats specially. On \n and \r\n text
    # without a NUL byte the readings agree exactly, header names and interior BOMs included (text
    # this short never meets the tokenizer's 256 KiB buffer edge, where a space-led quote or a BOM
    # in an over-long header can part them); past a lone \r or a NUL byte they do part, and the
    # cross-check refuses every text where they did.
    from seikan.dataio import _LONE_CR, FileReport, _records, _tokenizer_parity

    tokens = [",", '"', '""', "\n", "\r\n", "\r", " ", "\t", "a", "1", "\x00", "\ufeff"]
    rng = np.random.RandomState(0)
    n_trusted = n_parted = 0
    for _ in range(4000):
        blob = "".join(rng.choice(tokens, size=rng.randint(0, 25))).encode()
        records = [fields for _line, fields in _records(blob)]
        if not records or any(len(f) != len(records[0]) for f in records[1:]):
            continue
        names = [h.strip().lower() for h in records[0]]
        if len(set(names)) < len(names):
            continue  # a duplicate header name refuses before pandas reads
        trusted = b"\x00" not in blob and _LONE_CR.search(blob) is None
        try:
            raw = pd.read_csv(
                io.BytesIO(blob), dtype=str, keep_default_na=False, encoding="utf-8-sig"
            )
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            assert not trusted or "EOF inside string" in str(exc), blob
            continue
        same = (
            isinstance(raw.index, pd.RangeIndex)
            and raw.shape[1] == len(records[0])
            and all(h in ("", c) for h, c in zip(records[0], raw.columns, strict=True))
            and raw.values.tolist() == records[1:]
        )
        if trusted:
            n_trusted += 1
            assert same, blob
        elif not same:
            n_parted += 1
            raw.columns = [str(c).strip().lower() for c in raw.columns]  # as _parse_frame does
            assert not _tokenizer_parity("f.csv", blob, raw, FileReport(path="f.csv")), blob
    assert n_trusted > 500 and n_parted > 20  # both properties exercised, not vacuously skipped


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
    gaps = [w for w in rep.warnings if w["code"] == "calendar_gap"]
    assert len(gaps) == 1
    assert gaps[0]["message"].endswith(
        ": 1 gap(s) larger than 10× the median bar spacing (86400 s)"
    )


def test_calendar_gap_is_read_in_exact_ticks_at_any_span(tmp_path):
    # The review's case: three nanosecond-stamped rows 200 years apart. The Timedelta median
    # times the factor overflowed int64 nanoseconds and crashed the read (OverflowError); every
    # spacing sits near the median, so the file loads clean.
    text = (
        "datetime,x\n"
        "1700-01-01T00:00:00.000000001,1\n"
        "1900-01-01T00:00:00.000000001,2\n"
        "2100-01-01T00:00:00.000000001,3\n"
    )
    df, rep = _run(tmp_path, text)
    assert rep.ok and df is not None and df.index.unit == "ns"
    assert "calendar_gap" not in {w["code"] for w in rep.warnings}
    # a gap past ten times a sub-second median still warns, its spacing stated exactly
    stamps = ["2021-01-01T00:00:00", "2021-01-01T00:00:00.5", "2021-01-01T00:00:01", "2021-01-02"]
    _, rep = _run(tmp_path, "datetime,x\n" + "".join(f"{t},1\n" for t in stamps))
    assert [w["message"].split(": ", 1)[1] for w in rep.warnings] == [
        "1 gap(s) larger than 10× the median bar spacing (0.5 s)"
    ]


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


# ---- bar spacing ---------------------------------------------------------------------------


def _spacing(stamps: list[str]) -> tuple:
    s = dataio.bar_spacing(pd.DatetimeIndex(pd.to_datetime(stamps, format="ISO8601")))
    return s["min_seconds"], s["median_seconds"], s["max_seconds"]


def test_bar_spacing_is_exact_in_the_index_resolution(tmp_path):
    # The stamp was computed on int64 NANOSECONDS, which overflowed wherever two consecutive bars
    # lie more than ~292 years apart — spans an [s]/[us] index (pandas ≥ 2 keeps the parsed
    # resolution) holds without complaint: a 1700/2024 pair read −8222340873.709552 s.
    assert _spacing(["1700-01-01", "2024-01-01"]) == (10_224_403_200,) * 3
    assert _spacing(["1500-01-01", "1500-01-02", "2000-01-01"]) == (
        86400,
        7_889_227_200,
        15_778_368_000,
    )
    # whole seconds stay integers, fractions are the exact spacing correctly rounded, and an even
    # count's median is the exact midpoint of its two middle spacings
    assert _spacing(
        ["2020-01-01", "2020-01-01T23:59:59.5", "2020-01-02T23:59:59.25", "2020-01-03T23:59:59"]
    ) == (86399.5, 86399.75, 86399.75)
    assert _spacing(
        ["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:03.000000001"]
    ) == (1, 1.5000000005, 2.000000001)
    daily = pd.date_range("2021-01-01", periods=4, freq="1D")
    for unit in ("s", "ms", "us", "ns"):
        s = dataio.bar_spacing(daily.as_unit(unit))
        assert s == {"min_seconds": 86400, "median_seconds": 86400, "max_seconds": 86400}, unit
        assert all(type(v) is int for v in s.values()), unit
    # end to end: the strict read admits the old stamps, and the stamp states them
    df, rep = _run(tmp_path, "datetime,y\n1700-01-01,1.0\n2024-01-01,2.0\n2024-01-02,3.0\n")
    assert rep.ok and df is not None
    assert dataio.bar_spacing(df.index) == {
        "min_seconds": 86400,
        "median_seconds": 5_112_244_800,
        "max_seconds": 10_224_403_200,
    }
