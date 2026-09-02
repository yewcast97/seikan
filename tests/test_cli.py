"""End-to-end CLI tests (in-process ``main()``): the report document, every exit code, the honesty
invariant (a run whose every cell is unmet still writes the whole grid), lazy compute, determinism,
and the self-serve ``schema`` output.

There is ONE measuring subcommand, ``seikan run``, and it is SILENT on success: the outputs are the
four files the caller nominates by flag — including ``--entry-flags-out``, which writes the per-bar
firing mask — and stdout carries an error envelope or nothing at all. So the two helpers below
split by channel — ``_run_report`` reads the
``--report-out`` FILE back (asserting the empty stdout for free, on every report test in this
file), while ``_run`` keeps parsing stdout for the error-envelope tiers.

The exit code describes how far the RUN got, never how the evidence looked. A completed ``run`` is
exit 0 whatever its cells report — the per-cell results live in ``gate.cells`` — so every assertion
below that touches an outcome reads the JSON, never the status.

A thesis NAMES its series and never locates them, so every ``run`` invocation here carries the
``--data KEY=PATH`` pairs that answer its declared keys (``_data`` / ``_target`` build the argv
fragment). The mapping is part of the REQUEST, not of the document: it is written beside each
thesis rather than inside it, and a run whose pairs do not answer the thesis exactly is a usage
refusal before a byte of CSV is read.
"""

from __future__ import annotations

import io
import json
import typing
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seikan.cli import REPORT_SCHEMA_VERSION, main
from tests._helpers import unmet_checks as _unmet_cell_checks
from tests._helpers import unmet_run_doc as _unmet_run_checks
from tests._helpers import write_ohlcv as _write_ohlcv

# ---- fixtures -------------------------------------------------------------------------------


def _write_thesis(path: Path, **params) -> Path:
    """The workhorse single-target thesis. It declares the logical key ``target`` and nothing about
    where that series lives — the invocation says which CSV answers it (``_target`` below)."""
    dsl = {
        "name": "cli-e2e",
        "data": {"targets": ["target"]},
        "entry": {
            "type": "threshold",
            "left": {
                "type": "percentile",
                "window": 14,
                "input": {"type": "field", "column": "close"},
            },
            "op": "<",
            "right": {"type": "constant", "value": 0.35},
        },
        "params": {"horizon": 5, **params},
    }
    path.write_text(json.dumps(dsl), encoding="utf-8")
    return path


_SPIKE_CUT = 105.0


def _write_spiky_ohlcv(path: Path, fired, n: int = 120, seed: int = 0) -> Path:
    """An OHLCV CSV whose ``close`` is engineered so that ``close > 105`` is true on EXACTLY the
    bars in ``fired`` and false everywhere else.

    ``_write_ohlcv`` + a percentile threshold cannot promise either half of that. WHICH bars clear
    a percentile is whatever the seed drew, so a test naming the final bar as its seam would be
    resting that seam on an accident, and a mask that quietly became all-ones (or all-zeros) would
    keep passing a set equality that had stopped comparing anything. Here the baseline wanders
    inside [98, 102] and a fired bar is lifted by +10 into [108, 112]: 105 separates the two
    families with room to spare, every close is still distinct (returns vary — no zero-variance
    panel), and the caller CHOOSES the mask, so sparsity and a final-bar firing are properties the
    test can assert as preconditions rather than hope for.
    """
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="1D")
    close = 100.0 + rng.uniform(-2.0, 2.0, n)
    close[sorted(fired)] += 10.0
    o = pd.Series(close).shift(1).bfill().to_numpy()
    df = pd.DataFrame(
        {
            "open": o,
            "high": np.maximum(o, close) * 1.001,
            "low": np.minimum(o, close) * 0.999,
            "close": close,
            "volume": 1000.0,
        },
        index=idx,
    )
    df.index.name = "datetime"
    df.to_csv(path)
    return path


def _write_spiky_thesis(path: Path, **params) -> Path:
    """The thesis the spiky fixture was engineered against: a bare ``close > 105`` cut on a Field
    leaf, so there is no warmup window and the entry mask is EXACTLY the spikes — nothing between
    the constructed mask and the measured one to explain a disagreement away."""
    dsl = {
        "name": "cli-e2e",
        "data": {"targets": ["target"]},
        "entry": {
            "type": "threshold",
            "left": {"type": "field", "column": "close"},
            "op": ">",
            "right": {"type": "constant", "value": _SPIKE_CUT},
        },
        "params": {"horizon": 5, **params},
    }
    path.write_text(json.dumps(dsl), encoding="utf-8")
    return path


def _scattered_with_final_bar(n: int = 120, period: int = 7, offset: int = 3) -> list[int]:
    """Every ``period``-th bar from ``offset``, PLUS the final bar — sparse (well under half the
    index), scattered (no two fired bars adjacent, so the mask is not a contiguous block that a
    slice-off-by-one would reproduce) and true on bar ``n-1``, which is the seam the entry-flags
    CSV exists for."""
    return sorted({*range(offset, n - 1, period), n - 1})


def _write_series(path: Path, n: int = 400, seed: int = 0) -> Path:
    """A single-column (series-shaped) target CSV — no volume column is synthesized."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="1D")
    df = pd.DataFrame({"y": 3.0 + np.cumsum(rng.randn(n) * 0.01)}, index=idx)
    df.index.name = "datetime"
    df.to_csv(path)
    return path


def _write_multi_series(path: Path, names=("y2", "y10"), n: int = 400, seed: int = 0) -> Path:
    """A series-shaped CSV holding SEVERAL value columns — a yield curve, a vendor's whole feed
    file, the shape ``--column KEY=COL`` exists for.

    Nothing about such a file says which of its columns a given key MEANS, and nothing about the
    thesis can say it either (the document names a series, never a header), so a key answered by
    one of these must be bound to a column or refused. Each column wanders around its own level so
    a run that read the wrong one is visible in the numbers, not just in the shape.
    """
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="1D")
    df = pd.DataFrame(
        {name: 3.0 + i + np.cumsum(rng.randn(n) * 0.01) for i, name in enumerate(names)},
        index=idx,
    )
    df.index.name = "datetime"
    df.to_csv(path)
    return path


def _write_feed_thesis(path: Path, feeds: list[str], cut: float = 4.0, **params) -> Path:
    """A thesis over one OHLCV target plus external feeds, so the request has keys BESIDE the
    target to bind columns to. Two feeds are compared against each other, one against ``cut`` —
    either way every declared feed is genuinely read, so a column bound to one of them changes
    what the run measures rather than riding along unused."""
    right = (
        {"type": "external", "name": feeds[1]}
        if len(feeds) > 1
        else {"type": "constant", "value": cut}
    )
    dsl = {
        "name": "cli-e2e",
        "data": {"targets": ["target"], "external": {name: {} for name in feeds}},
        "entry": {
            "type": "threshold",
            "left": {"type": "external", "name": feeds[0]},
            "op": ">",
            "right": right,
        },
        "params": {"horizon": 5, **params},
    }
    path.write_text(json.dumps(dsl), encoding="utf-8")
    return path


def _data(mapping: dict) -> list[str]:
    """The ``--data KEY=PATH`` argv fragment binding a thesis's declared series to files.

    One pair per key the thesis declares — the invocation's half of a load, and the only half that
    knows about the filesystem. Spelled out at each call site rather than hidden in a fixture,
    because the pair set IS part of the request under test: a run that answers a key it does not
    declare, or leaves one unanswered, is refused before any CSV is opened.
    """
    return [arg for key, path in mapping.items() for arg in ("--data", f"{key}={path}")]


def _target(path) -> list[str]:
    """The one-target shorthand: every ``_write_thesis`` document declares the single key
    ``target``, so its whole mapping is which CSV that name points at."""
    return _data({"target": path})


def _columns(mapping: dict) -> list[str]:
    """The ``--column KEY=COL`` argv fragment saying which column of its file each key reads.

    The twin of ``_data`` over the SAME flat key namespace, and OPTIONAL where that one is
    mandatory: only a key whose file holds several numeric columns needs a pair here, so most
    invocations in this file carry none at all."""
    return [arg for key, column in mapping.items() for arg in ("--column", f"{key}={column}")]


def _run(capsys, argv) -> tuple[int, dict]:
    """Run and parse the JSON envelope stdout carries. For the ERROR tiers: a successful ``run``
    prints nothing, so a report test routed through here would parse an empty string."""
    code = main(argv)
    out = capsys.readouterr().out
    return code, json.loads(out)


def _run_report(capsys, tmp_path, argv, name: str = "report.json") -> tuple[int, dict]:
    """Run with ``--report-out`` appended and read the report FILE back.

    Silence on success is the contract, so it is asserted HERE rather than test by test: every
    report assertion in this file pins it for free. Error paths write no report file and DO print
    an envelope — those stay on ``_run``.
    """
    out = tmp_path / name
    code = main([*argv, "--report-out", str(out)])
    assert capsys.readouterr().out == ""
    return code, json.loads(out.read_text(encoding="utf-8"))


# ---- run: the report -------------------------------------------------------------------------


def test_run_writes_complete_report_and_exits_0(tmp_path, capsys):
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    code, doc = _run_report(capsys, tmp_path, ["run", str(thesis), *_target(px)])
    assert code == 0  # the run completed; the exit code is not a verdict
    assert doc["command"] == "run" and doc["seikan_version"]
    assert doc["report_schema_version"] == 5
    ident = doc["identity"]
    assert "command" not in ident and "report_schema_version" not in ident  # they live at top level
    assert ident["name"] == "cli-e2e"
    assert len(ident["dsl_hash"]) == 64
    assert ident["thresholds_canonical"] is True
    assert set(ident["thresholds_provenance"]) == {
        "thesis_min_trades",
        "thesis_min_n_nonoverlap",
        "thesis_max_concentration",
        "thesis_max_hypotheses",
    }
    assert doc["data_report"]["ok"] is True
    assert doc["summary"]["statistics_version"] == 4  # the estimator revision is stamped
    assert doc["summary"]["gate_evidence_basis"] == "full_sample"
    assert doc["summary"]["cells"]  # the full grid is present
    assert doc["summary"]["rotation"]["n_shifts"] > 0  # the null's resolution is stamped
    assert "trades" not in doc  # no time series ever rides the report


def test_report_layers_in_fixed_order(tmp_path, capsys):
    # The layering is FIXED (no variants): the uniform {seikan_version, report_schema_version,
    # command} header, then identity → data_report → outputs → summary → gate → metric_roles.
    # Layer order is the contract consumers parse against. There is no top-level pooled "n" (a
    # grid has no headline) and no flat trades_* keys — `outputs` owns every nominated file.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    _code, doc = _run_report(capsys, tmp_path, ["run", str(thesis), *_target(px)])
    assert list(doc) == [
        "seikan_version",
        "report_schema_version",
        "command",
        "identity",
        "data_report",
        "outputs",
        "summary",
        "gate",
        "metric_roles",
    ]
    assert "n" not in doc
    assert "trades_rows_written" not in doc and "trades_path" not in doc
    # only what was nominated is enumerated — and the report itself always is, since the document
    # exists only under --report-out
    assert doc["outputs"] == {"report": {"path": str(tmp_path / "report.json")}}
    # data identity: one entry per LOGICAL key, each naming the file that answered it, the column
    # read out of it and the sha256 of the bytes read, mirroring the data_report's per-file
    # digests. `column` is null here because this run bound none — the stamp is ALWAYS present, so
    # "the file named its own single value column" is a stated fact rather than a missing key.
    digests = doc["identity"]["data_digests"]
    by_path = {f["path"]: f["sha256"] for f in doc["data_report"]["files"]}
    assert digests == {"target": {"path": str(px), "column": None, "sha256": by_path[str(px)]}}
    assert all(len(d["sha256"]) == 64 for d in digests.values())
    # a changed data byte is a visibly different exam (a trailing blank line leaves the parsed
    # frame — and every statistic — identical; only the identity moves)
    px.write_bytes(px.read_bytes() + b"\n")
    _code, doc2 = _run_report(capsys, tmp_path, ["run", str(thesis), *_target(px)])
    assert doc2["identity"]["data_digests"] != digests
    assert doc2["summary"] == doc["summary"]


def test_outputs_enumerates_every_nominated_file_in_nomination_order(tmp_path, capsys):
    # `outputs` is keyed report → trades → root_series → entry_flags whatever order the flags
    # arrived in, and each entry names the path actually written plus (for the CSVs) the rows that
    # landed in it. The report is written LAST but listed FIRST: it VOUCHES for the other files.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    report, trades, root_series, flags = (
        tmp_path / "r.json",
        tmp_path / "trades.csv",
        tmp_path / "rs.csv",
        tmp_path / "ef.csv",
    )
    code = main(
        [
            "run",
            str(thesis),
            *_target(px),
            "--entry-flags-out",
            str(flags),  # flag order deliberately not the key order
            "--root-series-out",
            str(root_series),
            "--trades-out",
            str(trades),
            "--report-out",
            str(report),
        ]
    )
    assert code == 0
    assert capsys.readouterr().out == ""  # silent on success, however many outputs were nominated
    assert report.exists() and trades.exists() and root_series.exists() and flags.exists()
    doc = json.loads(report.read_text(encoding="utf-8"))
    outputs = doc["outputs"]
    assert list(outputs) == ["report", "trades", "root_series", "entry_flags"]
    assert outputs["report"] == {"path": str(report)}  # the report has no row count of its own
    assert outputs["trades"]["path"] == str(trades)
    assert outputs["root_series"]["path"] == str(root_series)
    assert outputs["entry_flags"]["path"] == str(flags)
    assert outputs["trades"]["rows_written"] > 0
    assert outputs["root_series"]["rows_written"] == doc["summary"]["n_bars"]
    # the flags matrix is a per-BAR view like the root series, never a per-observation one: its row
    # count is the index geometry, not the firing count (that is what makes it able to carry a
    # final-bar firing at all).
    assert outputs["entry_flags"]["rows_written"] == doc["summary"]["n_bars"]
    assert len(pd.read_csv(trades)) == outputs["trades"]["rows_written"]
    assert len(pd.read_csv(root_series)) == outputs["root_series"]["rows_written"]
    assert len(pd.read_csv(flags)) == outputs["entry_flags"]["rows_written"]


def test_identity_stamps_exactly_the_four_threshold_knobs(tmp_path, capsys):
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    _code, doc = _run_report(capsys, tmp_path, ["run", str(thesis), *_target(px)])
    thresholds = doc["identity"]["thresholds"]
    assert set(thresholds) == {
        "thesis_min_trades",
        "thesis_min_n_nonoverlap",
        "thesis_max_concentration",
        "thesis_max_hypotheses",
    }
    assert thresholds["thesis_min_trades"] == 30
    assert thresholds["thesis_min_n_nonoverlap"] == 8
    assert thresholds["thesis_max_concentration"] == 0.6
    assert thresholds["thesis_max_hypotheses"] == 64


def test_gate_section_shape_carries_no_verdict(tmp_path, capsys):
    # There is no scalar verdict: cells are graded independently, so there is no single
    # answer to report and nothing for a caller to branch on but the per-cell results.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json", horizon=[5, 10])
    _code, doc = _run_report(capsys, tmp_path, ["run", str(thesis), *_target(px)])
    gate = doc["gate"]
    assert set(gate) == {"policy_version", "n_cells", "n_met", "run_checks", "cells"}
    assert gate["policy_version"] == 3
    assert "verdict" not in json.dumps(gate)
    assert len(gate["run_checks"]) == 3
    assert [c["name"] for c in gate["run_checks"]] == [
        "evidence_complete",
        "source_coverage",
        "search_cap",
    ]
    assert gate["n_cells"] == len(gate["cells"]) == doc["summary"]["n_hypotheses_attempted"] == 2
    assert gate["n_met"] == sum(1 for c in gate["cells"] if c["met"])
    for cell in gate["cells"]:
        assert set(cell) == {"cell_id", "params", "met", "checks"}
        assert [c["name"] for c in cell["checks"]] == [
            "cell_evidence",
            "outcome_coverage",
            "signal_coverage",
            "support",
            "concentration",
        ]


def test_gate_cells_align_positionally_with_summary_cells(tmp_path, capsys):
    # gate["cells"][i] grades summary["cells"][i]; identity is the params plus the position,
    # never the rendered label.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json", horizon=[5, 10])
    _code, doc = _run_report(capsys, tmp_path, ["run", str(thesis), *_target(px)])
    summary_cells, gate_cells = doc["summary"]["cells"], doc["gate"]["cells"]
    assert len(summary_cells) == len(gate_cells)
    for s_cell, g_cell in zip(summary_cells, gate_cells, strict=True):
        assert s_cell["cell_id"] == g_cell["cell_id"]
        assert s_cell["params"] == g_cell["params"]
        assert "horizon" in s_cell["params"]  # the measurement window is always named
    assert [c["params"]["horizon"] for c in summary_cells] == [5, 10]


def test_run_exits_0_when_every_cell_is_unmet(tmp_path, capsys):
    # An unattainable (but stricter-than-canonical, so constructible) support floor leaves every
    # cell unmet. The run still COMPLETED, so the exit code is 0 and the report is complete — every
    # declared parameter × horizon cell, with the reason named per cell instead of hidden.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json", horizon=[5, 10])
    code, doc = _run_report(
        capsys, tmp_path, ["run", str(thesis), *_target(px), "--min-n-nonoverlap", "10000"]
    )
    assert code == 0
    assert doc["gate"]["n_cells"] == 2 and doc["gate"]["n_met"] == 0
    for cell in doc["gate"]["cells"]:
        assert cell["met"] is False
        assert "support" in _unmet_cell_checks(cell)
        assert len(cell["checks"]) == 5  # no short-circuit: every check still reported
    cells = doc["summary"]["cells"]
    assert {c["params"]["horizon"] for c in cells} == {5, 10}  # both horizon cells, met or not
    assert all(c["by_target"]["target"]["n"] > 0 for c in cells)  # the evidence is never filtered
    assert len(cells) == 2


def test_report_metric_roles_compact_with_claim(tmp_path, capsys):
    # The report carries the COMPACT role map (the prose lives in `seikan schema`): the exact
    # claim, per-check consumed fields, evidence-only list, scope boundary.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    _code, doc = _run_report(capsys, tmp_path, ["run", str(thesis), *_target(px)])
    roles = doc["metric_roles"]
    assert set(roles) == {
        "claim",
        "run_checks",
        "cell_checks",
        "evidence_only",
        "metric_classes",
        "caveats",
        "scope_boundary",
    }
    claim = roles["claim"]
    assert "NO significance claim" in claim
    assert "positive-expected-return" in claim
    assert "n_hypotheses_attempted" in claim  # multiplicity is handed to the caller, explicitly
    assert set(roles["run_checks"]) == {"evidence_complete", "source_coverage", "search_cap"}
    # the per-metric caveats TRAVEL WITH the report — one short sentence each, so the agent
    # holding a number also holds the reason not to over-trust it
    assert set(roles["caveats"]) >= {
        "rot_p",
        "t_hac",
        "boot",
        "pbo",
        "mean_ret",
        "concentration",
        "mean_ret_raw",
        "edge_ratio",
        "profit_factor",
    }
    # the machine-readable class map: a fixed three-value vocabulary covering every caveat key
    classes = roles["metric_classes"]
    assert set(classes.values()) == {"descriptive", "inference", "integrity"}
    assert set(roles["caveats"]) <= set(classes)
    assert classes["rot_p"] == "inference" and classes["rot_n_null"] == "integrity"
    assert classes["mean_ret"] == "descriptive" and classes["n_nonoverlap"] == "integrity"
    # the known-invalid iid pair and the derivable convenience reads are gone, not reclassed
    for retired in ("t_iid", "p_iid", "sharpe", "firing_rate", "median_ret", "nominal"):
        assert retired not in classes and retired not in classes.values()
    assert all(
        isinstance(v, str) and "\n" not in v and len(v) < 300 for v in roles["caveats"].values()
    )
    assert "anti-conservative" in roles["caveats"]["t_hac"]
    assert "gross of costs" in roles["caveats"]["mean_ret"]
    assert set(roles["cell_checks"]) == {
        "cell_evidence",
        "outcome_coverage",
        "signal_coverage",
        "support",
        "concentration",
    }
    assert isinstance(roles["evidence_only"], list)
    # no inferential surface may appear as a role anywhere
    blob = json.dumps(roles)
    for absent in ("oos_confirmation", "selection_integrity", "sign_p", "fw_p", "p_rw", "p_fdr"):
        assert absent not in blob


def test_metric_classes_cover_every_reported_metric(tmp_path, capsys):
    # The class map is TOTAL over the report's metric surface: every by_target panel key, every cell
    # evidence block and every run-level metric stamp appears in metric_classes — so a consuming
    # agent never meets a number the map cannot classify. Blocks class as a whole; episode_stats.n
    # is the one sanctioned dotted override (the reconciliation count cell_evidence re-checks).
    px = _write_ohlcv(tmp_path / "px.csv")
    bench = _write_ohlcv(tmp_path / "bench.csv", seed=3)
    thesis = _write_thesis(tmp_path / "t.json", benchmark="market")
    _code, doc = _run_report(
        capsys, tmp_path, ["run", str(thesis), *_data({"target": px, "benchmark": bench})]
    )
    classes = doc["metric_roles"]["metric_classes"]
    labels = {"target", "cell_id", "params"} | set(doc["summary"]["params"])
    for cell in doc["summary"]["cells"]:
        for panel in cell["by_target"].values():
            for key in panel:
                assert key in classes, f"by_target key {key!r} has no metric class"
        for key in set(cell) - labels - {"by_target", "pooled"}:
            assert key in classes, f"cell block {key!r} has no metric class"
    for stamp in (
        "rotation",
        "pbo",
        "baseline",
        "bar_spacing",
        "sources",
        "cross_breadth",
        "n_hypotheses_attempted",
        "n_bars",
    ):
        assert stamp in classes
    # the basket-only surfaces are classed too, statically
    assert classes["pooled"] == "descriptive" and classes["member_share"] == "descriptive"
    assert classes["episode_stats.n"] == "integrity"


def test_run_trades_out(tmp_path, capsys):
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out_csv = tmp_path / "trades.csv"
    code, doc = _run_report(
        capsys, tmp_path, ["run", str(thesis), *_target(px), "--trades-out", str(out_csv)]
    )
    assert code == 0
    trades_out = doc["outputs"]["trades"]
    assert trades_out["rows_written"] > 0
    assert trades_out["path"] == str(out_csv)
    trades = pd.read_csv(out_csv)
    assert {"target", "ret"} <= set(trades.columns)
    # The ISO times are the record: no epoch-ns duplicates of them ride the CSV.
    assert "entry_ts" not in trades.columns and "exit_ts" not in trades.columns
    assert len(trades) == trades_out["rows_written"]


def test_trades_only_run_is_silent_and_writes_no_report(tmp_path, capsys):
    # Any ONE output flag suffices, and nothing else is produced: a trades-only run leaves the
    # trades CSV and not a byte more — no report file, no stdout.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out_csv = tmp_path / "trades.csv"
    code = main(["run", str(thesis), *_target(px), "--trades-out", str(out_csv)])
    assert code == 0
    assert capsys.readouterr().out == ""
    assert len(pd.read_csv(out_csv)) > 0
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".json"] == ["t.json"]


def test_include_trades_flag_is_a_usage_error(capsys):
    # Trades never ride stdout, so there is no flag that embeds them — and an argparse usage error
    # surfaces as a clean exit-3 `usage` envelope, not a bare SystemExit.
    code, doc = _run(capsys, ["run", "whatever.json", "--include-trades"])
    assert code == 3
    assert doc["error"]["type"] == "usage"


def test_run_report_is_deterministic_and_overwrites(tmp_path, capsys):
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "report.json"
    main(["run", str(thesis), *_target(px), "--report-out", str(out)])
    first = out.read_bytes()
    main(["run", str(thesis), *_target(px), "--report-out", str(out)])
    assert capsys.readouterr().out == ""
    assert out.read_bytes() == first  # byte-identical: same inputs → same report, always rewritten


def test_benchmarked_run_report_is_deterministic(tmp_path, capsys):
    # The benchmark path carries its own extra frames (the retained raw/bench legs), so it gets
    # its own byte pin: same inputs → the same excess report, leg attribution included.
    px = _write_ohlcv(tmp_path / "px.csv")
    bench = _write_ohlcv(tmp_path / "bench.csv", seed=3)
    thesis = _write_thesis(tmp_path / "t.json", benchmark="market")
    out = tmp_path / "report.json"
    argv = [
        "run",
        str(thesis),
        *_data({"target": px, "benchmark": bench}),
        "--report-out",
        str(out),
    ]
    main(argv)
    first = out.read_bytes()
    main(argv)
    assert capsys.readouterr().out == ""
    assert out.read_bytes() == first


# ---- hash: the canonical identity, no Python required ------------------------------------------


def test_hash_emits_identity_and_data_keys(tmp_path, capsys):
    # The document is the standard header + {name, dsl_hash, data_keys}: dsl_hash is the SAME
    # identity a run report stamps for the same thesis, and data_keys is the exact set --data
    # must answer — the benchmark key included exactly when params.benchmark asks for one.
    px = _write_ohlcv(tmp_path / "px.csv")
    bench = _write_ohlcv(tmp_path / "bench.csv", seed=3)
    thesis = _write_thesis(tmp_path / "t.json", benchmark="market")
    code, doc = _run(capsys, ["hash", str(thesis)])
    assert code == 0
    assert list(doc) == [
        "seikan_version",
        "report_schema_version",
        "command",
        "name",
        "dsl_hash",
        "data_keys",
    ]
    assert doc["command"] == "hash" and doc["name"] == "cli-e2e"
    assert doc["data_keys"] == ["target", "benchmark"]
    _c, report = _run_report(
        capsys, tmp_path, ["run", str(thesis), *_data({"target": px, "benchmark": bench})]
    )
    assert doc["dsl_hash"] == report["identity"]["dsl_hash"]
    # an unbenchmarked thesis drops the reserved key from the request
    plain = _write_thesis(tmp_path / "p.json")
    _c, plain_doc = _run(capsys, ["hash", str(plain)])
    assert plain_doc["data_keys"] == ["target"]


def _stdin_bytes(monkeypatch, payload: bytes) -> None:
    """Feed `-` mode raw BYTES: `_cmd_hash` reads `sys.stdin.buffer` and decodes UTF-8 itself
    (locale-proof — the same bytes must hash identically wherever they are piped from)."""
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8"))


def test_hash_reads_stdin_and_normalizes_defaults(tmp_path, capsys, monkeypatch):
    # `-` reads the document from stdin, and the hash is CANONICAL: the omitted-default and
    # explicit-default spellings of one thesis share one identity (the gate.py discipline,
    # reachable without writing Python).
    thesis = _write_thesis(tmp_path / "t.json")
    _c, from_file = _run(capsys, ["hash", str(thesis)])
    _stdin_bytes(monkeypatch, thesis.read_bytes())
    _c, from_stdin = _run(capsys, ["hash", "-"])
    assert from_stdin == from_file
    explicit = json.loads(thesis.read_text(encoding="utf-8"))
    explicit["target_mode"] = "conjunction"  # the default, spelled out
    explicit["params"]["direction"] = "longonly"
    _stdin_bytes(monkeypatch, json.dumps(explicit).encode("utf-8"))
    _c, explicit_doc = _run(capsys, ["hash", "-"])
    assert explicit_doc["dsl_hash"] == from_file["dsl_hash"]


def test_hash_stdin_decodes_utf8_regardless_of_locale(tmp_path, capsys, monkeypatch):
    # A non-ASCII thesis name must hash to ONE identity through both mouths: the stdin branch
    # reads sys.stdin.buffer and decodes UTF-8 itself, so a host piping under a C/ASCII locale
    # cannot be handed a second, mojibake identity at exit 0.
    thesis = _write_thesis(tmp_path / "t.json")
    doc = json.loads(thesis.read_text(encoding="utf-8"))
    doc["name"] = "café—momentum"
    thesis.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    _c, from_file = _run(capsys, ["hash", str(thesis)])
    assert from_file["name"] == "café—momentum"
    _stdin_bytes(monkeypatch, thesis.read_bytes())
    _c, from_stdin = _run(capsys, ["hash", "-"])
    assert from_stdin == from_file


def test_hash_invalid_thesis_is_the_ordinary_dsl_invalid_envelope(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "x", "exit": "sma_cross"}), encoding="utf-8")
    code, doc = _run(capsys, ["hash", str(bad)])
    assert code == 3
    assert doc["command"] == "hash" and doc["error"]["type"] == "dsl_invalid"
    # a missing file is the same tier — an unreadable request, never an internal error
    code, doc = _run(capsys, ["hash", str(tmp_path / "absent.json")])
    assert code == 3 and doc["error"]["type"] == "dsl_invalid"


def test_hash_is_deterministic(tmp_path, capsys):
    thesis = _write_thesis(tmp_path / "t.json")
    code_a, doc_a = _run(capsys, ["hash", str(thesis)])
    code_b, doc_b = _run(capsys, ["hash", str(thesis)])
    assert (code_a, doc_a) == (code_b, doc_b) == (0, doc_a)


def test_basket_run_report_is_deterministic_and_overwrites(tmp_path, capsys):
    # The basket twin of the pin above, through the CLI seam: the pooled reads (the pooled
    # per-cell block with its common-shift rotation null and cross-member-merged bootstrap,
    # member_share, the pooled baseline row) must introduce no positional or clock-keyed
    # state — same inputs, same bytes, always rewritten.
    a = _write_ohlcv(tmp_path / "a.csv", seed=1)
    b = _write_ohlcv(tmp_path / "b.csv", seed=2)
    dsl = {
        "name": "cli-basket",
        "target_mode": "basket",
        "data": {"targets": ["A", "B"]},
        "entry": {
            "type": "threshold",
            "left": {
                "type": "percentile",
                "window": 14,
                "input": {"type": "field", "column": "close"},
            },
            "op": "<",
            "right": {"type": "constant", "value": 0.35},
        },
        "params": {"horizon": 5},
    }
    thesis = tmp_path / "bt.json"
    thesis.write_text(json.dumps(dsl), encoding="utf-8")
    members = _data({"A": a, "B": b})
    out = tmp_path / "report.json"
    assert main(["run", str(thesis), *members, "--report-out", str(out)]) == 0
    first = out.read_bytes()
    assert main(["run", str(thesis), *members, "--report-out", str(out)]) == 0
    assert capsys.readouterr().out == ""
    assert out.read_bytes() == first
    doc = json.loads(first)
    assert doc["summary"]["target_mode"] == "basket"  # the twin exercises the basket path
    assert "pooled" in doc["summary"]["cells"][0]


# ---- run: nominating outputs ------------------------------------------------------------------


def test_run_without_any_output_flag_is_a_usage_error(tmp_path, capsys):
    # A request that nominates nothing is malformed whatever its thesis says.
    thesis = _write_thesis(tmp_path / "t.json")
    code, doc = _run(capsys, ["run", str(thesis)])
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert doc["command"] == "run"
    message = doc["error"]["message"]
    for flag in ("--report-out", "--trades-out", "--root-series-out", "--entry-flags-out"):
        assert flag in message  # the refusal names every way to satisfy it


def test_run_without_output_flag_refuses_before_reading_the_thesis(tmp_path, capsys):
    # The zero-flag check fires BEFORE the thesis file is opened, so a nonexistent path is still
    # `usage` and never `dsl_invalid` — the cheap refusal comes first, and the caller is told about
    # the malformed REQUEST rather than sent chasing a file the run never needed.
    code, doc = _run(capsys, ["run", str(tmp_path / "does-not-exist.json")])
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert doc["command"] == "run"


@pytest.mark.parametrize(
    "argv",
    [
        ["explain", "t.json"],
        ["simulate", "t.json", "o.csv"],
    ],
)
def test_an_unknown_subcommand_is_a_usage_error(capsys, argv):
    # A name this CLI does not define fails loudly at argparse's invalid-choice hook rather than
    # falling through to a default: `command` is null because no subparser was ever resolved, and
    # the refusal still arrives as the ordinary JSON envelope on stdout.
    code, doc = _run(capsys, argv)
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert doc["command"] is None
    assert argv[0] in doc["error"]["message"]


def test_run_refuses_an_empty_output_path(tmp_path, capsys):
    # argparse DID receive --report-out, so an empty value is a nomination named unusably, not an
    # absent flag. Collapsing the two would exit 0 having silently skipped an output the caller
    # asked for — the one thing exit 0 promises it never does.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    trades = tmp_path / "tr.csv"
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *_target(px),
            "--report-out",
            "",
            "--trades-out",
            str(trades),
        ],
    )
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert "--report-out" in doc["error"]["message"]
    assert not trades.exists()  # refused before any compute, so no sibling output landed either


def test_run_refuses_two_flags_naming_one_file(tmp_path, capsys):
    # The writes are sequential overwrites, so honoring both would destroy the earlier output and
    # still enumerate it in `outputs` with a rows_written describing content that is not on disk.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    both = tmp_path / "both.out"
    code, doc = _run(
        capsys,
        ["run", str(thesis), *_target(px), "--report-out", str(both), "--trades-out", str(both)],
    )
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert "--report-out" in doc["error"]["message"]
    assert "--trades-out" in doc["error"]["message"]
    assert not both.exists()


def test_run_refuses_root_series_and_entry_flags_naming_one_file(tmp_path, capsys):
    # The collision the two per-bar outputs make easiest to type: same shape, same row count, one
    # letter apart on the command line. They come off ONE listing call and are written sequentially,
    # so honoring both would leave the FLAGS where the caller asked for values while `outputs` went
    # on enumerating two files and two row counts. Refused up front, before any compute.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    both = tmp_path / "per_bar.csv"
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *_target(px),
            "--root-series-out",
            str(both),
            "--entry-flags-out",
            str(both),
        ],
    )
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert "--root-series-out" in doc["error"]["message"]
    assert "--entry-flags-out" in doc["error"]["message"]
    assert not both.exists()


@pytest.mark.parametrize(
    "flag", ["--report-out", "--trades-out", "--root-series-out", "--entry-flags-out"]
)
def test_run_refuses_an_output_that_would_overwrite_its_own_input(tmp_path, capsys, flag):
    # The engine refuses rather than mutates everywhere else — the loader will not even clamp an
    # OHLC violation — so the CLI does not get to be the one writer that destroys its own evidence,
    # least of all while identity.data_digests stamps a sha256 for the bytes it deleted.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    before = px.read_bytes()
    code, doc = _run(capsys, ["run", str(thesis), *_target(px), flag, str(px)])
    assert code == 3
    assert doc["error"]["type"] == "usage"
    # The refusal quotes the file the way this invocation named it — by its --data key, not by a
    # DSL field, because the document holds no path to point at.
    assert "--data target" in doc["error"]["message"]
    assert px.read_bytes() == before


@pytest.mark.parametrize(
    "flag", ["--report-out", "--trades-out", "--root-series-out", "--entry-flags-out"]
)
def test_run_refuses_an_output_that_would_overwrite_the_thesis_file(tmp_path, capsys, flag):
    # The thesis is the run's PRIMARY input and the bytes identity.dsl_hash is computed over, so an
    # output landing on it destroys the definition of the exam being reported. It is also the
    # easiest collision to type by accident: the output flag sits on the same command line as the
    # thesis argument.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    before = thesis.read_bytes()
    code, doc = _run(capsys, ["run", str(thesis), *_target(px), flag, str(thesis)])
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert "thesis file" in doc["error"]["message"]
    assert thesis.read_bytes() == before


@pytest.mark.parametrize("mask", ["sparse", "every-bar"])
def test_a_final_bar_firing_rides_the_entry_flags_csv(tmp_path, capsys, mask):
    # The documented seam, pinned from BOTH sides so the docs cannot silently rot into a lie. The
    # root-series CSV carries no 0/1 flags, on the grounds that a fired bar becomes a trades row
    # instead — true for every bar EXCEPT the last, which has no next open to anchor at and so opens
    # no observation. Nothing in OBSERVATION shape can represent that firing, so no
    # observation-shaped output can carry it to a CLI caller at all.
    #
    # --entry-flags-out resolves the seam without touching the engine: the flag matrix is shaped per
    # BAR, so the final bar simply gets its 1 — which is exactly what "is my thesis firing NOW?"
    # reads. The runner drops a firing it cannot fill, which is correct, and the assertions
    # below pin both halves at once: trades stop one short, the flags do not.
    #
    # The SPARSE case is what gives the mask equality teeth. An every-bar fixture makes the headline
    # assertion "all bars == all bars", which a writer emitting a constant-1 column of the right
    # length would satisfy; on a scattered 18-of-120 mask that same writer names 102 bars that never
    # fired. The every-bar case is kept alongside it for the degenerate boundary (a mask with no
    # zeros at all, where the final-bar firing is the LAST of many rather than an isolated one), and
    # the fixture preconditions below are asserted against the mask the engine actually produced —
    # so if either fixture ever stops being the shape its name claims, this test fails rather than
    # quietly degenerating.
    from seikan.api import list_entries
    from seikan.dsl.schema import Thesis

    from ._helpers import load

    n = 120
    fired = _scattered_with_final_bar(n) if mask == "sparse" else list(range(n))
    px = _write_spiky_ohlcv(tmp_path / "px.csv", fired, n=n)
    thesis = _write_spiky_thesis(tmp_path / "t.json")
    trades, flags = tmp_path / "tr.csv", tmp_path / "ef.csv"
    code, _doc = _run_report(
        capsys,
        tmp_path,
        [
            "run",
            str(thesis),
            *_target(px),
            "--trades-out",
            str(trades),
            "--entry-flags-out",
            str(flags),
        ],
    )
    assert code == 0

    t = Thesis.model_validate(json.loads(thesis.read_text(encoding="utf-8")))
    listing = list_entries(t, load(t, {"target": px}))
    realized = set(np.flatnonzero(listing.entry_flags.to_numpy().ravel() == 1).tolist())
    assert realized == set(fired), "the engine's mask must be the one the fixture engineered"
    # FIXTURE PRECONDITIONS — the properties every assertion below depends on, stated so they
    # cannot rot: the final bar fires, and (sparse case) the mask is neither everything nor a
    # contiguous tail block that an off-by-one slice would reproduce.
    assert n - 1 in realized
    if mask == "sparse":
        assert 0 < len(realized) < n // 3
        ordered = sorted(realized)
        assert max(b - a for a, b in pairwise(ordered)) > 1

    fires = listing.entries[0]["timestamps"]
    assert fires[-1] == listing.series_end  # the last bar really did fire
    assert int(listing.entry_flags.iloc[-1].iloc[0]) == 1  # and the library reports it

    # entry_bar is the SIGNAL bar's position (entry_time is the ANCHOR bar t+1, so the two files
    # join on entry_bar, never on the timestamp). One horizon is declared here, so the trades frame
    # holds exactly one row per firing it could fill.
    frame = pd.read_csv(trades)
    entered = {int(b) for b in frame["entry_bar"]}
    assert entered == realized - {n - 1}
    assert len(frame) == len(entered)
    assert n - 1 not in entered  # the seam, stated from the observation side

    # ...and the decision-side file carries it. One row per BAR (not per observation), so the count
    # is the index geometry and the last row is the firing the trades CSV structurally cannot hold.
    flagged = pd.read_csv(flags, parse_dates=["datetime"])
    assert list(flagged.columns) == ["datetime", "entry"]
    assert len(flagged) == len(listing.root_series) == n
    assert int(flagged["entry"].iloc[-1]) == 1
    # bit-identical to the library's mask, stated as the set of bars rather than a count
    assert list(flagged.loc[flagged["entry"] == 1, "datetime"]) == list(fires)


def test_input_clobber_refusal_sees_through_the_path_spelling(tmp_path, capsys):
    # Resolved comparison, not string equality: a caller who spells the input differently is not
    # thereby allowed to overwrite it.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    (tmp_path / "sub").mkdir()
    aliased = tmp_path / "sub" / ".." / "px.csv"
    before = px.read_bytes()
    code, doc = _run(capsys, ["run", str(thesis), *_target(px), "--trades-out", str(aliased)])
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert px.read_bytes() == before


def test_run_refuses_an_output_that_would_overwrite_the_benchmark(tmp_path, capsys):
    # EVERY mapped file is protected, not just the targets: the benchmark is measurement evidence
    # too, and it reaches the run the same way they do — as a `--data` pair, under the reserved key
    # `params.benchmark: "market"` adds to the request. The guard reads the resolved mapping rather
    # than a list of DSL fields, so a key added to a thesis is covered the day it is added.
    px = _write_ohlcv(tmp_path / "px.csv")
    bench = _write_ohlcv(tmp_path / "bench.csv", seed=7)
    thesis = tmp_path / "t.json"
    dsl = json.loads(_write_thesis(thesis).read_text(encoding="utf-8"))
    dsl["params"]["benchmark"] = "market"
    thesis.write_text(json.dumps(dsl), encoding="utf-8")
    before = bench.read_bytes()
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *_data({"target": px, "benchmark": bench}),
            "--root-series-out",
            str(bench),
        ],
    )
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert "--data benchmark" in doc["error"]["message"]
    assert bench.read_bytes() == before


# ---- run: locating the series the thesis names (--data KEY=PATH) -------------------------------
#
# The document says WHICH series it measures; the invocation says WHERE they are. Everything below
# is a malformed or unanswerable REQUEST, so every refusal is `usage` (exit 3) and lands before a
# byte of CSV is read — a run that loaded nothing for a declared series would be measuring a thesis
# it does not have, and one that loaded a file for a key the thesis never mentions would be
# answering a question nobody asked.


def test_run_refuses_a_data_pair_that_is_not_key_equals_path(tmp_path, capsys):
    # A bare path is the natural typo: a caller reaching for the flag reaches for a filename.
    # Guessing which key it meant would be inventing a declaration, so the pair is refused and the
    # caller told what a pair is.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "r.json"
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            "--data",
            str(px),
            "--report-out",
            str(out),
        ],
    )
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert "is not a KEY=PATH pair" in doc["error"]["message"]
    assert not out.exists()


@pytest.mark.parametrize(
    "pair, missing",
    [("=px.csv", "an empty key"), ("target=", "an empty path")],
    ids=["empty-key", "empty-path"],
)
def test_run_refuses_a_data_pair_missing_one_of_its_halves(tmp_path, capsys, pair, missing):
    # Half a pair is not a pair: a key with no path names a series and locates nothing, and a path
    # with no key locates a file the run cannot attribute to anything the thesis declares. Neither
    # can be repaired by guessing, so neither is.
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "r.json"
    code, doc = _run(capsys, ["run", str(thesis), "--data", pair, "--report-out", str(out)])
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert missing in doc["error"]["message"]
    assert not out.exists()


def test_run_refuses_one_data_key_named_twice(tmp_path, capsys):
    # Last-wins would leave the caller not knowing which file the run read — and the report would
    # stamp a digest for one of them as if the other had never been offered. Two pairs naming one
    # series is a question about the request, so it is asked back rather than resolved.
    px = _write_ohlcv(tmp_path / "px.csv")
    other = _write_ohlcv(tmp_path / "other.csv", seed=3)
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "r.json"
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *_data({"target": px}),
            "--data",
            f"target={other}",
            "--report-out",
            str(out),
        ],
    )
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert "names 'target' twice" in doc["error"]["message"]
    assert str(px) in doc["error"]["message"] and str(other) in doc["error"]["message"]
    assert not out.exists()


def test_run_refuses_a_mapping_that_leaves_a_declared_key_unanswered(tmp_path, capsys):
    # No --data at all is the extreme case of the same refusal, and it is worth stating: a thesis
    # that names its series cannot be run by naming nothing, however complete the document is on
    # its own terms.
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "r.json"
    code, doc = _run(capsys, ["run", str(thesis), "--report-out", str(out)])
    assert code == 3
    assert doc["error"]["type"] == "usage"
    message = doc["error"]["message"]
    assert "missing ['target']" in message
    assert "it reads ['target']" in message  # both sets are named, not just the complaint
    assert not out.exists()


def test_run_refuses_a_mapping_that_answers_an_undeclared_key(tmp_path, capsys):
    # An unknown key is a caller who believes this thesis reads something it never mentions —
    # usually a stale pair left behind by an edit to the document. Loading it and
    # ignoring it would confirm the belief; refusing names both sets and corrects it.
    px = _write_ohlcv(tmp_path / "px.csv")
    spare = _write_ohlcv(tmp_path / "spare.csv", seed=4)
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "r.json"
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *_data({"target": px, "spare": spare}),
            "--report-out",
            str(out),
        ],
    )
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert "unknown ['spare']" in doc["error"]["message"]
    assert not out.exists()


def test_a_data_pair_splits_on_the_first_equals_only(tmp_path, capsys):
    # A path may legitimately contain '='; a key may not (the DSL refuses it in a target or feed
    # name for exactly this reason), so the separator is never ambiguous and the remainder of the
    # token is the path verbatim.
    px = _write_ohlcv(tmp_path / "px=1.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    _code, doc = _run_report(capsys, tmp_path, ["run", str(thesis), *_target(px)])
    assert doc["identity"]["data_digests"]["target"]["path"] == str(px)


def _per_target_feed_thesis(tmp_path: Path) -> tuple[Path, dict]:
    """Two targets and ONE per-target external feed — so the request carries the derived
    ``<feed>@<target>`` keys beside the plain target names, four names in one flat namespace,
    exactly as they are typed."""
    mapping = {
        "AAA": _write_ohlcv(tmp_path / "AAA.csv", seed=1),
        "BBB": _write_ohlcv(tmp_path / "BBB.csv", seed=2),
        "iv@AAA": _write_series(tmp_path / "iv_a.csv", seed=3),
        "iv@BBB": _write_series(tmp_path / "iv_b.csv", seed=4),
    }
    dsl = {
        "name": "cli-e2e",
        "data": {"targets": ["AAA", "BBB"], "external": {"iv": {"per_target": True}}},
        "entry": {
            "type": "threshold",
            "left": {"type": "external", "name": "iv"},
            "op": ">",
            "right": {"type": "constant", "value": 0.0},
        },
        "params": {"horizon": 5},
    }
    thesis = tmp_path / "t.json"
    thesis.write_text(json.dumps(dsl), encoding="utf-8")
    return thesis, mapping


def test_a_per_target_feed_is_answered_under_its_derived_keys(tmp_path, capsys):
    # The derived keys are not decoration: they are what the caller types, one file per (feed ×
    # target), and the run reads exactly the four files it was handed. Dropping one derived key
    # refuses like any other unanswered series — per-target COVER is by construction in the DSL,
    # so the only way to under-supply a per-target feed is at the invocation, and it is caught here.
    thesis, mapping = _per_target_feed_thesis(tmp_path)
    out = tmp_path / "ef.csv"
    assert main(["run", str(thesis), *_data(mapping), "--entry-flags-out", str(out)]) == 0
    assert capsys.readouterr().out == ""
    assert list(pd.read_csv(out).columns) == ["datetime", "entry@AAA", "entry@BBB"]

    partial = {key: path for key, path in mapping.items() if key != "iv@BBB"}
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *_data(partial),
            "--entry-flags-out",
            str(tmp_path / "again.csv"),
        ],
    )
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert "iv@BBB" in doc["error"]["message"]


def test_run_refuses_an_output_that_would_overwrite_a_mapped_feed(tmp_path, capsys):
    # The clobber guard walks the RESOLVED mapping, so it protects every file the invocation bound
    # — a per-target feed's CSV as surely as a target's — and quotes the one thing that identifies
    # it here: the key the caller typed. There is no DSL field left to name instead, and that is an
    # improvement: the guard cannot fall behind a thesis that grows a new kind of key.
    thesis, mapping = _per_target_feed_thesis(tmp_path)
    victim = mapping["iv@BBB"]
    before = victim.read_bytes()
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *_data(mapping),
            "--entry-flags-out",
            str(victim),
        ],
    )
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert "--data iv@BBB" in doc["error"]["message"]
    assert victim.read_bytes() == before


def test_data_digests_are_keyed_by_the_logical_key(tmp_path, capsys):
    # identity.data_digests is keyed {key: {path, column, sha256}}. The KEY is what the
    # thesis declares and what two runs can be compared on; the path is only where this invocation
    # happened to find the bytes, and the column only which of them answered. All three ride the
    # entry — "the series this thesis calls `benchmark` was this file, read out of this column, and
    # it hashed to this" — so a re-run over re-pulled data is a legible diff instead of a renamed
    # row. `column` is present even when nothing was bound (null): a stamp that appeared only
    # sometimes would read as a different document rather than as a different answer.
    px = _write_ohlcv(tmp_path / "px.csv")
    bench = _write_ohlcv(tmp_path / "bench.csv", seed=7)
    thesis = tmp_path / "t.json"
    dsl = json.loads(_write_thesis(thesis).read_text(encoding="utf-8"))
    dsl["params"]["benchmark"] = "market"
    thesis.write_text(json.dumps(dsl), encoding="utf-8")
    _code, doc = _run_report(
        capsys, tmp_path, ["run", str(thesis), *_data({"target": px, "benchmark": bench})]
    )
    by_path = {f["path"]: f["sha256"] for f in doc["data_report"]["files"]}
    assert doc["identity"]["data_digests"] == {
        "target": {"path": str(px), "column": None, "sha256": by_path[str(px)]},
        "benchmark": {"path": str(bench), "column": None, "sha256": by_path[str(bench)]},
    }
    assert doc["summary"]["benchmark_source"] == str(bench)  # the resolved CSV, named by path

    # ONE file answering TWO keys is TWO entries. A path-keyed shape would make it a single row
    # whose series was unrecoverable; keyed by the logical key, both series say what they read.
    _code, doc = _run_report(
        capsys,
        tmp_path,
        ["run", str(thesis), *_data({"target": px, "benchmark": px})],
        name="same.json",
    )
    digests = doc["identity"]["data_digests"]
    assert set(digests) == {"target", "benchmark"}
    assert (
        digests["target"]
        == digests["benchmark"]
        == {
            "path": str(px),
            "column": None,
            "sha256": by_path[str(px)],
        }
    )


def test_two_keys_answered_by_one_file_keep_their_own_columns(tmp_path, capsys):
    # The shape the sha256/column split exists for: one wide CSV answers TWO feed keys, each
    # reading a different column of it. The digests share a hash — it is a property of the file's
    # bytes, so a single lookup by path is correct — while each entry keeps the column that made it
    # a distinct SERIES. Collapsing them (one row per file, or a hash per key) would either lose
    # which column each key read or claim two files where there is one.
    px = _write_ohlcv(tmp_path / "px.csv")
    wide = _write_multi_series(tmp_path / "wide.csv", ("fast", "slow"))
    thesis = _write_feed_thesis(tmp_path / "t.json", ["fast", "slow"])
    _code, doc = _run_report(
        capsys,
        tmp_path,
        [
            "run",
            str(thesis),
            *_data({"target": px, "fast": wide, "slow": wide}),
            *_columns({"fast": "fast", "slow": "slow"}),
        ],
    )
    by_path = {f["path"]: f["sha256"] for f in doc["data_report"]["files"]}
    digests = doc["identity"]["data_digests"]
    assert digests["fast"] == {"path": str(wide), "column": "fast", "sha256": by_path[str(wide)]}
    assert digests["slow"] == {"path": str(wide), "column": "slow", "sha256": by_path[str(wide)]}
    assert digests["target"]["column"] is None  # the OHLCV target bound none, and says so


# ---- run: choosing WHICH column answers a key (--column KEY=COL) --------------------------------
#
# The second half of locating a series, and it sits outside the document for the same reason the
# path does: a column name is a property of the FILE that happens to answer a key — one vendor ships
# three yields in one CSV under its own spellings, another ships each in a file of its own — so were
# it in the DSL, re-shaping a CSV would make the same exam a DIFFERENT document.
#
# The binding is OPTIONAL, which splits the refusals across two tiers. What is decidable from the
# REQUEST alone is exit 3 before a byte is read (a malformed pair, a key the thesis never declares,
# the `benchmark` key, which measures outcomes off its file's open and chooses nothing). What takes
# BYTES to decide is exit 2 like every other spec/data disagreement: whether a file needs a binding
# at all, and whether the name it was given is a column that file has.


def test_run_refuses_a_column_pair_that_is_not_key_equals_col(tmp_path, capsys):
    # The bare column name is this flag's natural typo, exactly as the bare path is --data's: a
    # caller who has one file in mind types the header alone. Guessing which of the thesis's keys
    # it meant would be inventing the binding, so the pair is refused and the caller told what a
    # pair is.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "r.json"
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *_target(px),
            "--column",
            "close",
            "--report-out",
            str(out),
        ],
    )
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert "is not a KEY=COL pair" in doc["error"]["message"]
    assert not out.exists()


@pytest.mark.parametrize(
    "pair, missing",
    [("=y10", "an empty key"), ("target=", "an empty column")],
    ids=["empty-key", "empty-column"],
)
def test_run_refuses_a_column_pair_missing_one_of_its_halves(tmp_path, capsys, pair, missing):
    # Half a pair is not a pair, on this flag for the same reason as on --data: a key with no
    # column selects nothing, and a column with no key selects out of a file the run cannot
    # attribute to any declared series.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "r.json"
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *_target(px),
            "--column",
            pair,
            "--report-out",
            str(out),
        ],
    )
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert missing in doc["error"]["message"]
    assert not out.exists()


def test_run_refuses_one_column_key_named_twice(tmp_path, capsys):
    # Last-wins would leave the caller not knowing which series the run measured — and the report
    # would stamp one column into the digests as if the other had never been offered. One key reads
    # one column, so two pairs naming one key is asked back rather than resolved.
    curve = _write_multi_series(tmp_path / "curve.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "r.json"
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *_target(curve),
            "--column",
            "target=y2",
            "--column",
            "target=y10",
            "--report-out",
            str(out),
        ],
    )
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert "names 'target' twice" in doc["error"]["message"]
    assert "'y2'" in doc["error"]["message"] and "'y10'" in doc["error"]["message"]
    assert not out.exists()


def test_run_refuses_a_column_bound_to_a_key_the_thesis_never_declares(tmp_path, capsys):
    # Nothing would ever read this binding, so accepting it would let a caller believe they had
    # chosen a series when they had chosen nothing — usually a pair left behind by an edit to the
    # document. Both sets are named, as they are for an unknown --data key.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "r.json"
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *_target(px),
            *_columns({"spare": "y10"}),
            "--report-out",
            str(out),
        ],
    )
    assert code == 3
    assert doc["error"]["type"] == "usage"
    message = doc["error"]["message"]
    assert "['spare']" in message  # the stray binding
    assert "it reads ['target']" in message  # …and the keys that do exist
    assert not out.exists()


def test_run_refuses_a_column_bound_to_the_benchmark(tmp_path, capsys):
    # The benchmark is outcome MEASUREMENT, not a decision input: it is sampled at the
    # observation's own anchor bars and always reads its file's `open`, so there is nothing for a
    # column to select. The key is declared — it is in the mapping right beside the target — which
    # is exactly why the refusal is its own sentence rather than the unknown-key one.
    px = _write_ohlcv(tmp_path / "px.csv")
    bench = _write_ohlcv(tmp_path / "bench.csv", seed=7)
    thesis = tmp_path / "t.json"
    dsl = json.loads(_write_thesis(thesis).read_text(encoding="utf-8"))
    dsl["params"]["benchmark"] = "market"
    thesis.write_text(json.dumps(dsl), encoding="utf-8")
    out = tmp_path / "r.json"
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *_data({"target": px, "benchmark": bench}),
            *_columns({"benchmark": "open"}),
            "--report-out",
            str(out),
        ],
    )
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert "'benchmark' key takes no column" in doc["error"]["message"]
    assert not out.exists()


def test_run_refuses_a_column_binding_on_an_ohlcv_target(tmp_path, capsys):
    # A price target always measures its open-anchored prices, so a binding on one is a request the
    # engine cannot honour in any reading — `--column target=close` does not mean "measure closes"
    # (that would be a different anchor, which no flag may set). It takes bytes to know the file is
    # OHLCV-shaped, so this is exit 2 with the rest of the spec/data disagreements, and the message
    # quotes the binding back so the caller knows which argv token to delete.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "r.json"
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *_target(px),
            *_columns({"target": "close"}),
            "--report-out",
            str(out),
        ],
    )
    assert code == 2
    assert doc["error"]["type"] == "data_invalid"
    assert "--column target=close is bound but the file is OHLCV-shaped" in doc["error"]["message"]
    assert not out.exists()


def test_run_refuses_a_multi_column_series_file_with_no_binding(tmp_path, capsys):
    # A yield-curve CSV holds several series and says nothing about which of them this key IS.
    # Picking one (the first, the alphabetical) would measure a series nobody asked for, so the run
    # refuses and names the pair that answers it — the caller's fix is an invocation edit, and
    # there is no DSL field left to point them at instead.
    curve = _write_multi_series(tmp_path / "curve.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "r.json"
    code, doc = _run(capsys, ["run", str(thesis), *_target(curve), "--report-out", str(out)])
    assert code == 2
    assert doc["error"]["type"] == "data_invalid"
    assert "bind --column target=COL" in doc["error"]["message"]
    assert not out.exists()


def test_run_refuses_a_column_name_the_file_does_not_have(tmp_path, capsys):
    # The one refusal that cannot be decided from the request: whether a NAME answers a column is a
    # fact about bytes. So the check happens where the file is read (exit 2), and the message lists
    # the columns the file actually has — a typo is then a one-line fix rather than a second run to
    # find out what was in there.
    curve = _write_multi_series(tmp_path / "curve.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "r.json"
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *_target(curve),
            *_columns({"target": "y30"}),
            "--report-out",
            str(out),
        ],
    )
    assert code == 2
    assert doc["error"]["type"] == "data_invalid"
    message = doc["error"]["message"]
    assert "--column target=y30 is bound" in message  # the pair to fix, not merely the file
    assert "has no column 'y30'" in message
    assert "['y2', 'y10']" in message  # what it does have, in file order
    assert not out.exists()


def test_a_misspelled_column_names_the_key_whose_binding_is_wrong(tmp_path, capsys):
    # The refusal names the PAIR, not just the file, and this is the shape that makes the
    # difference load-bearing: one vendor CSV answers two feed keys, each reading its own column,
    # so "wide.csv has no column 'slwo'" would leave a caller staring at two --column pairs aimed
    # at one path with nothing to say which of them to edit. Naming the key turns the refusal back
    # into the token that has to change — the same service the OHLCV and unbound refusals already
    # performed, and the one the loader's own contract promises for all three.
    px = _write_ohlcv(tmp_path / "px.csv")
    wide = _write_multi_series(tmp_path / "wide.csv", ("fast", "slow"))
    thesis = _write_feed_thesis(tmp_path / "t.json", ["fast", "slow"])
    out = tmp_path / "r.json"
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *_data({"target": px, "fast": wide, "slow": wide}),
            *_columns({"fast": "fast", "slow": "slwo"}),
            "--report-out",
            str(out),
        ],
    )
    assert code == 2
    message = doc["error"]["message"]
    assert "--column slow=slwo is bound" in message  # the offending key, out of two on one file
    assert "fast" in message  # …and the columns that file does hold, one of them the sibling's
    assert not out.exists()


def test_naming_the_sole_column_of_a_single_column_file_is_legal(tmp_path, capsys):
    # A binding is needed only by an ambiguous file, but it is never FORBIDDEN by an unambiguous
    # one: a caller who writes the pair out for every key (a script that always emits both halves)
    # is saying something true, and refusing it would make the flag's optionality a trap.
    series = _write_series(tmp_path / "y.csv")  # one value column, named 'y'
    thesis = _write_thesis(tmp_path / "t.json")
    _code, doc = _run_report(
        capsys, tmp_path, ["run", str(thesis), *_target(series), *_columns({"target": "y"})]
    )
    assert doc["identity"]["data_digests"]["target"]["column"] == "y"
    assert doc["summary"]["target_shape"] == "series"


def test_a_column_binding_matches_the_header_case_insensitively(tmp_path, capsys):
    # The strict reader lowercases every header it admits, so a binding is lowercased once at the
    # door and matching is case-insensitive: a caller reading 'Y10' off a vendor's file must not
    # have to know what the reader did to it. The digests then stamp the ONE spelling that actually
    # matched — a report echoing "Y10" for a run that read `y10` would be describing a file that
    # does not exist.
    curve = _write_multi_series(tmp_path / "curve.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    _code, doc = _run_report(
        capsys, tmp_path, ["run", str(thesis), *_target(curve), *_columns({"target": "Y10"})]
    )
    assert doc["identity"]["data_digests"]["target"]["column"] == "y10"
    assert doc["summary"]["n_bars"] == 400


def test_a_feed_read_through_a_binding_measures_the_same_numbers_as_a_lone_file(tmp_path, capsys):
    # The end-to-end claim, and the reason the binding is a locating fact rather than a semantic
    # one: ONE thesis, run against a two-column CSV with the column bound and against a
    # single-column CSV holding the same series, must produce the SAME summary — every statistic,
    # not merely the same shape. Only the identity layer differs, and it differs in exactly the two
    # facts that did change: which file the bytes came from, and which column of it answered.
    px = _write_ohlcv(tmp_path / "px.csv")
    rng = np.random.RandomState(11)
    idx = pd.date_range("2018-01-01", periods=400, freq="1D")
    iv = 4.0 + np.cumsum(rng.randn(400) * 0.01)
    wide = pd.DataFrame({"noise": np.arange(400.0), "iv": iv}, index=idx)
    wide.index.name = "datetime"
    wide.to_csv(tmp_path / "wide.csv")
    lone = pd.DataFrame({"iv": iv}, index=idx)
    lone.index.name = "datetime"
    lone.to_csv(tmp_path / "lone.csv")
    thesis = _write_feed_thesis(tmp_path / "t.json", ["iv"], cut=4.0)

    _code, bound = _run_report(
        capsys,
        tmp_path,
        [
            "run",
            str(thesis),
            *_data({"target": px, "iv": tmp_path / "wide.csv"}),
            *_columns({"iv": "iv"}),
        ],
        name="bound.json",
    )
    _code, alone = _run_report(
        capsys,
        tmp_path,
        [
            "run",
            str(thesis),
            *_data({"target": px, "iv": tmp_path / "lone.csv"}),
        ],
        name="alone.json",
    )
    assert bound["summary"] == alone["summary"]
    assert bound["summary"]["cells"][0]["by_target"]["target"]["n"] > 0  # not a vacuous equality
    assert bound["identity"]["dsl_hash"] == alone["identity"]["dsl_hash"]  # one document, twice
    assert bound["identity"]["data_digests"]["iv"]["column"] == "iv"
    assert alone["identity"]["data_digests"]["iv"]["column"] is None


# ---- run: the error tiers ----------------------------------------------------------------------


def test_thesis_file_missing_exits_3(tmp_path, capsys):
    code, doc = _run(
        capsys,
        ["run", str(tmp_path / "nope.json"), "--report-out", str(tmp_path / "r.json")],
    )
    assert code == 3
    assert doc["error"]["type"] == "dsl_invalid"


def test_run_bad_data_exits_2_with_report(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "datetime,open,high,low,close\n2021-01-01,100,90,110,105\n2021-01-02,100,101,99,100\n",
        encoding="utf-8",
    )
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "r.json"
    code, doc = _run(capsys, ["run", str(thesis), *_target(bad), "--report-out", str(out)])
    assert code == 2
    assert doc["error"]["type"] == "data_invalid"
    codes = {e["code"] for f in doc["data_report"]["files"] for e in f["errors"]}
    assert "integrity" in codes
    assert not out.exists()  # a refused run vouches for nothing


def test_run_report_insufficient_bars_exits_2(tmp_path, capsys):
    px = _write_ohlcv(tmp_path / "short.csv", n=6)
    thesis = _write_thesis(tmp_path / "t.json", horizon=20)
    code, doc = _run(
        capsys, ["run", str(thesis), *_target(px), "--report-out", str(tmp_path / "r.json")]
    )
    assert code == 2
    assert any(e["code"] == "insufficient_common_index" for e in doc["data_report"]["errors"])


def test_root_series_only_skips_the_horizon_runway_requirement(tmp_path, capsys):
    # LAZY COMPUTE: the horizon-runway check belongs to the MEASUREMENT, and a root-series-only run
    # measures nothing — a signal series is perfectly well defined on an index too short to close a
    # single observation. The same thesis that exits 2 on the report path (above) writes its CSV
    # here, warmup NaNs and all.
    px = _write_ohlcv(tmp_path / "short.csv", n=6)
    thesis = _write_thesis(tmp_path / "t.json", horizon=20)
    out = tmp_path / "rs.csv"
    code = main(["run", str(thesis), *_target(px), "--root-series-out", str(out)])
    assert code == 0
    assert capsys.readouterr().out == ""
    frame = pd.read_csv(out)
    assert list(frame.columns) == ["datetime", "percentile(close,14)"]
    assert len(frame) == 6  # every bar of the provided index, none of them measurable


def test_entry_flags_only_skips_the_horizon_runway_requirement(tmp_path, capsys):
    # LAZY COMPUTE, the decision-side half: the runway check belongs to the MEASUREMENT, and an
    # entry-flags-only run measures nothing. The firing mask is well defined on an index far too
    # short to close one observation — and "is it firing?" is precisely the question a caller with
    # six bars of fresh data is asking, so refusing it for want of a forward window nobody wanted
    # would refuse a legitimate request on the strength of work that was never nominated.
    px = _write_ohlcv(tmp_path / "short.csv", n=6)
    thesis = _write_thesis(tmp_path / "t.json", horizon=20)
    out = tmp_path / "ef.csv"
    code = main(["run", str(thesis), *_target(px), "--entry-flags-out", str(out)])
    assert code == 0
    assert capsys.readouterr().out == ""
    frame = pd.read_csv(out)
    assert list(frame.columns) == ["datetime", "entry"]
    assert len(frame) == 6  # every bar of the provided index, none of them measurable
    assert set(frame["entry"].unique()) <= {0, 1}


def test_report_plus_entry_flags_on_a_short_index_exits_2_and_writes_no_flags(tmp_path, capsys):
    # WHICH PATH REFUSES, when the two kinds of output disagree about what the data must support.
    # Alone, --entry-flags-out succeeds on this six-bar index (the test above); alone, --report-out
    # exits 2 for want of a horizon runway (test_run_report_insufficient_bars_exits_2). Nominated
    # TOGETHER, the backtest runs FIRST — so the refusal a caller sees always comes from the STRICT
    # path, whose checks are a superset of the listing's, and never depends on flag order.
    #
    # The state this forecloses is the interesting one: a run that exits 2 having already written a
    # flags CSV. That file would be an output of a run that failed — enumerated by no report (the
    # report is written last, so a refused run vouches for nothing) and described by no exit code,
    # since 2 says the data was refused, not "one of the two nominated outputs landed anyway". A run
    # that refuses leaves the filesystem exactly as it found it, whichever outputs were nominated.
    px = _write_ohlcv(tmp_path / "short.csv", n=6)
    thesis = _write_thesis(tmp_path / "t.json", horizon=20)
    report, flags = tmp_path / "r.json", tmp_path / "ef.csv"
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *_target(px),
            "--entry-flags-out",
            str(flags),
            "--report-out",
            str(report),
        ],
    )
    assert code == 2
    assert doc["error"]["type"] == "data_invalid"
    assert any(e["code"] == "insufficient_common_index" for e in doc["data_report"]["errors"])
    assert not flags.exists()
    assert not report.exists()


def test_run_loads_market_data_exactly_once_across_all_four_outputs(tmp_path, capsys, monkeypatch):
    # ONE load per run: the CLI materializes MarketData once, after the exit-3 request validation,
    # and compile_thesis / list_entries consume it as a required parameter — neither loads. Counted
    # at BOTH seams: the cli-namespace binding proves the CLI called the loader exactly once, and
    # the strict-CSV reader inside the loader proves nothing anywhere reloaded — an internal load
    # inside api.py would be invisible to the first counter alone.
    import seikan.cli as cli_mod
    import seikan.compiler.data as data_mod

    loads: list = []
    reads: list = []
    real_load, real_read = cli_mod.load_market_data, data_mod.read_strict_csv

    def counting_load(spec, files):
        loads.append(spec)
        return real_load(spec, files)

    def counting_read(*args, **kwargs):
        reads.append(args)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(cli_mod, "load_market_data", counting_load)
    monkeypatch.setattr(data_mod, "read_strict_csv", counting_read)

    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    outs = {
        "--report-out": tmp_path / "r.json",
        "--trades-out": tmp_path / "tr.csv",
        "--root-series-out": tmp_path / "rs.csv",
        "--entry-flags-out": tmp_path / "ef.csv",
    }
    argv = ["run", str(thesis), *_target(px)]
    for flag, path in outs.items():
        argv += [flag, str(path)]
    code = main(argv)
    assert code == 0
    assert capsys.readouterr().out == ""
    assert len(loads) == 1  # the CLI loaded once...
    assert len(reads) == 1  # ...and nothing underneath re-read the one target file
    for path in outs.values():
        assert path.exists()  # the single load served all four outputs


def test_run_invalid_dsl_exits_3(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    dsl = json.loads(_write_thesis(tmp_path / "t.json").read_text(encoding="utf-8"))
    dsl["exit"] = {"type": "threshold"}  # unknown key under extra="forbid"
    bad.write_text(json.dumps(dsl), encoding="utf-8")
    code, doc = _run(capsys, ["run", str(bad), "--report-out", str(tmp_path / "r.json")])
    assert code == 3
    assert doc["error"]["type"] == "dsl_invalid"


@pytest.mark.parametrize(
    "key, value",
    [
        ("n_rotations", 1000),  # the null uses every non-identity shift: no budget to cap
        ("oos_fraction", 0.3),  # there is no holdout to size
        ("selection_mode", "in_sample"),
        ("oos_fraction", None),
    ],
)
def test_run_unknown_params_keys_exit_3(tmp_path, capsys, key, value):
    """A knob no mechanism here owns is an UNKNOWN key under ``extra="forbid"``: a thesis
    carrying one is invalid input, never a silently ignored key that leaves the caller believing
    a holdout was reserved or a rotation count honored."""
    bad = tmp_path / "bad.json"
    dsl = json.loads(_write_thesis(tmp_path / "t.json").read_text(encoding="utf-8"))
    dsl["params"][key] = value
    bad.write_text(json.dumps(dsl), encoding="utf-8")
    code, doc = _run(capsys, ["run", str(bad), "--report-out", str(tmp_path / "r.json")])
    assert code == 3
    assert doc["error"]["type"] == "dsl_invalid"
    assert key in json.dumps(doc["error"]["errors"])


def test_run_unparseable_json_exits_3(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    code, _doc = _run(capsys, ["run", str(bad), "--report-out", str(tmp_path / "r.json")])
    assert code == 3


@pytest.mark.parametrize(
    "params, data, entry",
    [
        ({"benchmark": "cross_mean"}, None, None),  # _check_benchmark_consistency
        (None, {"targets": ["benchmark"]}, None),  # _check_key_namespace
        ({"features": {"vol": {"type": "external", "name": "undeclared"}}}, None, None),
        (
            None,
            None,
            {
                "type": "threshold",
                "left": {"type": "external", "name": "nope"},
                "op": "<",
                "right": {"type": "constant", "value": 1.0},
            },
        ),
    ],
)
def test_custom_dsl_validator_errors_emit_the_exit_3_envelope(
    tmp_path, capsys, params, data, entry
):
    """Every DSL rule expressed as a custom `model_validator` must land on exit 3 WITH a
    report. Pydantic attaches the original exception object to each such error's `ctx`, which is
    not JSON serializable — emit the envelope naively and TypeError escapes the handler, killing
    the process with a traceback instead of a report. `extra="forbid"` errors, reported natively
    by pydantic-core, carry no such `ctx` and are the easy half; these are the other half."""
    dsl = json.loads(_write_thesis(tmp_path / "t.json").read_text(encoding="utf-8"))
    if params:
        dsl["params"].update(params)
    if data:
        dsl["data"] = data
    if entry:
        dsl["entry"] = entry
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(dsl), encoding="utf-8")
    code, doc = _run(capsys, ["run", str(bad), "--report-out", str(tmp_path / "r.json")])
    assert code == 3
    assert doc["error"]["type"] == "dsl_invalid"
    assert doc["error"]["errors"], "the refusal must say what was wrong"
    json.dumps(doc)  # the envelope is fully serializable


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_run_non_finite_json_literal_exits_3(tmp_path, capsys, literal):
    """Python's json decoder accepts these as an extension; JSON proper has no such literals.
    A non-finite threshold decides nothing, and `canonical_dsl_hash` would name the run with a
    token no strict parser can read back."""
    dsl = json.loads(_write_thesis(tmp_path / "t.json").read_text(encoding="utf-8"))
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(dsl).replace('"value": 0.35', f'"value": {literal}'), encoding="utf-8"
    )
    code, doc = _run(capsys, ["run", str(bad), "--report-out", str(tmp_path / "r.json")])
    assert code == 3
    assert doc["error"]["type"] == "dsl_invalid"


@pytest.mark.parametrize(
    "bound", ["01/02/2024", "2024-13-01", "2024-01-01T00:00:00+09:00", "not-a-date"]
)
def test_run_malformed_data_bounds_exit_3_not_4(tmp_path, capsys, bound):
    """Left to reach pandas label slicing, an ambiguous spelling would be silently INTERPRETED and
    an unparseable one would escape as an uncaught exception (exit 4 — an internal bug) rather
    than a DSL refusal. Both are refused at the document."""
    dsl = json.loads(_write_thesis(tmp_path / "t.json").read_text(encoding="utf-8"))
    dsl["data"]["start"] = bound
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(dsl), encoding="utf-8")
    code, doc = _run(capsys, ["run", str(bad), "--report-out", str(tmp_path / "r.json")])
    assert code == 3
    assert doc["error"]["type"] == "dsl_invalid"


def test_run_reserved_feature_name_exits_3_not_4(tmp_path, capsys):
    """A feature named `is_open` would duplicate the trades column and crash the boolean index —
    an exit-4 internal error out of ordinary (if wrong) caller input — so the name is reserved and
    the document refused."""
    dsl = json.loads(_write_thesis(tmp_path / "t.json").read_text(encoding="utf-8"))
    dsl["params"]["features"] = {"is_open": {"type": "field", "column": "close"}}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(dsl), encoding="utf-8")
    code, doc = _run(capsys, ["run", str(bad), "--report-out", str(tmp_path / "r.json")])
    assert code == 3
    assert doc["error"]["type"] == "dsl_invalid"


def test_run_over_cap_grid_exits_3_without_reading_data(tmp_path, capsys):
    """The declared grid is refused BEFORE any data is read — pointing a --data pair at a
    nonexistent CSV proves the ordering (a data-first path would exit 2 on the missing file, and
    the DSL refusal precedes even the resolution of the mapping)."""
    dsl = {
        "name": "cli-e2e",
        "data": {"targets": ["target"]},
        "entry": {
            "type": "threshold",
            "left": {"type": "field", "column": "close"},
            "op": "<",
            "right": {"type": "constant", "value": [float(i) for i in range(13)], "name": "k"},
        },
        "params": {"horizon": [1, 2, 3, 4, 5]},  # 13 × 5 = 65 > 64
    }
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(dsl), encoding="utf-8")
    code, doc = _run(
        capsys,
        [
            "run",
            str(bad),
            *_target(tmp_path / "does-not-exist.csv"),
            "--report-out",
            str(tmp_path / "r.json"),
        ],
    )
    assert code == 3
    assert doc["error"]["type"] == "dsl_invalid"


@pytest.mark.parametrize("axis_name", ["horizon", "target", "ret", "vol_14"])
def test_run_colliding_sweep_axis_exits_3_without_reading_data(tmp_path, capsys, axis_name):
    # A swept-constant name colliding with a reserved axis (target/horizon), a trade column (ret),
    # or a default feature (vol_14) is refused at PARSE time (exit 3), before any CSV is read —
    # pointing the --data pair at a nonexistent file proves the ordering (a data-first path would
    # exit 2). Raising the same collision inside collect_sweeps / the runner — after a data load —
    # would surface it as exit 4, an internal error, for what is a malformed request.
    dsl = {
        "name": "cli-e2e",
        "data": {"targets": ["target"]},
        "entry": {
            "type": "threshold",
            "left": {"type": "field", "column": "close"},
            "op": "<",
            "right": {"type": "constant", "value": [0.3, 0.35], "name": axis_name},
        },
        "params": {"horizon": 5},
    }
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(dsl), encoding="utf-8")
    code, doc = _run(
        capsys,
        [
            "run",
            str(bad),
            *_target(tmp_path / "does-not-exist.csv"),
            "--report-out",
            str(tmp_path / "r.json"),
        ],
    )
    assert code == 3
    assert doc["error"]["type"] == "dsl_invalid"


def test_run_duplicate_sweep_axis_exits_3_without_reading_data(tmp_path, capsys):
    dsl = {
        "name": "cli-e2e",
        "data": {"targets": ["target"]},
        "entry": {
            "type": "threshold",
            "left": {
                "type": "ema",
                "window": [5, 10],
                "input": {"type": "field", "column": "close"},
            },
            "op": "<",
            "right": {"type": "constant", "value": [1.0, 2.0], "name": "ema_window"},
        },
        "params": {"horizon": 5},
    }
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(dsl), encoding="utf-8")
    code, doc = _run(
        capsys,
        [
            "run",
            str(bad),
            *_target(tmp_path / "does-not-exist.csv"),
            "--report-out",
            str(tmp_path / "r.json"),
        ],
    )
    assert code == 3
    assert doc["error"]["type"] == "dsl_invalid"


def test_run_volume_on_series_data_exits_2(tmp_path, capsys):
    # A series-shaped target synthesizes no volume; an entry reading field 'volume' is a missing
    # INPUT (exit 2, data_invalid), preflighted by the API — not a runner crash (exit 4).
    data = _write_series(tmp_path / "y.csv")
    dsl = {
        "name": "cli-e2e",
        "data": {"targets": ["target"]},
        "entry": {
            "type": "threshold",
            "left": {"type": "field", "column": "volume"},
            "op": ">",
            "right": {"type": "constant", "value": 0.0},
        },
        "params": {"horizon": 5},
    }
    thesis = tmp_path / "t.json"
    thesis.write_text(json.dumps(dsl), encoding="utf-8")
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *_target(data),
            "--report-out",
            str(tmp_path / "r.json"),
        ],
    )
    assert code == 2
    assert doc["error"]["type"] == "data_invalid"
    assert "missing_volume" in json.dumps(doc["data_report"])


def test_root_series_volume_on_series_data_exits_2(tmp_path, capsys):
    # The ENTRY tree is read on the listing path too, so the same missing input refuses there —
    # lazy compute drops the measurement's checks, never the ones its own work needs.
    data = _write_series(tmp_path / "y.csv")
    dsl = {
        "name": "cli-e2e",
        "data": {"targets": ["target"]},
        "entry": {
            "type": "threshold",
            "left": {"type": "field", "column": "volume"},
            "op": ">",
            "right": {"type": "constant", "value": 0.0},
        },
        "params": {"horizon": 5},
    }
    thesis = tmp_path / "t.json"
    thesis.write_text(json.dumps(dsl), encoding="utf-8")
    out = tmp_path / "rs.csv"
    code, doc = _run(capsys, ["run", str(thesis), *_target(data), "--root-series-out", str(out)])
    assert code == 2
    assert doc["error"]["type"] == "data_invalid"
    assert not out.exists()  # the preflight refuses before any CSV is written


def test_entry_flags_volume_on_series_data_exits_2(tmp_path, capsys):
    # The flags output rides the SAME listing call, so it inherits the same refusal: the entry tree
    # is what both frames are built from, and an input it reads but the data cannot supply is a
    # missing INPUT (exit 2) however cheap the requested output is. Lazy compute drops the
    # measurement's checks, never the ones the listing's own work needs.
    data = _write_series(tmp_path / "y.csv")
    dsl = {
        "name": "cli-e2e",
        "data": {"targets": ["target"]},
        "entry": {
            "type": "threshold",
            "left": {"type": "field", "column": "volume"},
            "op": ">",
            "right": {"type": "constant", "value": 0.0},
        },
        "params": {"horizon": 5},
    }
    thesis = tmp_path / "t.json"
    thesis.write_text(json.dumps(dsl), encoding="utf-8")
    out = tmp_path / "ef.csv"
    code, doc = _run(capsys, ["run", str(thesis), *_target(data), "--entry-flags-out", str(out)])
    assert code == 2
    assert doc["error"]["type"] == "data_invalid"
    assert "missing_volume" in json.dumps(doc["data_report"])
    assert not out.exists()  # the preflight refuses before any CSV is written


def _volume_feature_thesis(tmp_path: Path) -> tuple[Path, list[str]]:
    """A thesis whose ENTRY needs no volume but whose ``params.features`` reads it, on
    series-shaped data that synthesizes none — the discriminator between the two API paths.

    Returns the thesis beside the ``--data`` fragment that locates its one series, since the
    document names the series and nothing more."""
    data = _write_series(tmp_path / "y.csv")
    dsl = {
        "name": "cli-e2e",
        "data": {"targets": ["target"]},
        "entry": {
            "type": "threshold",
            "left": {"type": "field", "column": "close"},
            "op": ">",
            "right": {"type": "constant", "value": 0.0},
        },
        "params": {"horizon": 5, "features": {"vol": {"type": "field", "column": "volume"}}},
    }
    thesis = tmp_path / "t.json"
    thesis.write_text(json.dumps(dsl), encoding="utf-8")
    return thesis, _target(data)


def test_run_volume_feature_on_series_data_exits_2(tmp_path, capsys):
    # The same refusal when a params.features series (not the entry) reads volume.
    thesis, series = _volume_feature_thesis(tmp_path)
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *series,
            "--report-out",
            str(tmp_path / "r.json"),
        ],
    )
    assert code == 2
    assert "missing_volume" in json.dumps(doc["data_report"])


def test_root_series_only_skips_the_features_volume_check(tmp_path, capsys):
    # LAZY COMPUTE, the other half: params.features exist only to bucket MEASUREMENTS, and the
    # listing path builds none — so it checks the entry tree's inputs alone (include_features
    # False) and the thesis that refuses on the report path (above) writes its CSV here.
    thesis, series = _volume_feature_thesis(tmp_path)
    out = tmp_path / "rs.csv"
    code = main(["run", str(thesis), *series, "--root-series-out", str(out)])
    assert code == 0
    assert capsys.readouterr().out == ""
    assert list(pd.read_csv(out).columns) == ["datetime", "close"]


def test_entry_flags_only_skips_the_features_volume_check(tmp_path, capsys):
    # The same lazy-compute boundary for the flags frame: params.features exist only to bucket
    # MEASUREMENTS, and the listing path builds none whichever of its two frames was nominated. So
    # the thesis that refuses on the report path writes its firing mask here, off the entry tree
    # alone.
    thesis, series = _volume_feature_thesis(tmp_path)
    out = tmp_path / "ef.csv"
    code = main(["run", str(thesis), *series, "--entry-flags-out", str(out)])
    assert code == 0
    assert capsys.readouterr().out == ""
    frame = pd.read_csv(out)
    assert list(frame.columns) == ["datetime", "entry"]
    assert set(frame["entry"].unique()) <= {0, 1}


# ---- thresholds: the sealed four ---------------------------------------------------------------


def test_threshold_flag_overrides_env(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SEIKAN_THESIS_MIN_TRADES", "99")
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    _code, doc = _run_report(capsys, tmp_path, ["run", str(thesis), *_target(px)])
    assert doc["identity"]["thresholds"]["thesis_min_trades"] == 99  # env respected
    _code, doc = _run_report(
        capsys, tmp_path, ["run", str(thesis), *_target(px), "--min-trades", "45"]
    )
    assert doc["identity"]["thresholds"]["thesis_min_trades"] == 45  # flag wins (stricter)


@pytest.mark.parametrize("flag", ["--min-oos-n-eff", "--oos-alpha", "--gate-profile"])
def test_unknown_threshold_flags_are_usage_errors(capsys, flag):
    # There is no holdout, no episode sign test and no profile system, so there is no knob for any
    # of them. An argparse usage error is the loud failure (exit 3, `usage` envelope) — never a
    # spelling silently accepted and then honoured by nothing. argparse rejects the unknown flag
    # before the run is entered at all, so no output nomination can rescue it.
    code, doc = _run(capsys, ["run", "whatever.json", flag, "0.5"])
    assert code == 3
    assert doc["error"]["type"] == "usage"


@pytest.mark.parametrize(
    "var",
    [
        "SEIKAN_THESIS_OOS_ALPHA",  # no episode sign test to size
        "SEIKAN_THESIS_MIN_OOS_N_EFF",  # no holdout tail to put a floor under
        "SEIKAN_GATE_PROFILE",  # no profile system to select
        "SEIKAN_THESIS_MIN_PSR",  # PSR rides as evidence; no check reads it
        "SEIKAN_THESIS_MIN_N_EFF",  # the retired v2 knob name — renamed N_NONOVERLAP in v3
        "SEIKAN_MAX_P_VALUE",  # a plain typo (missing THESIS_)
    ],
)
def test_unknown_seikan_env_vars_exit_3(tmp_path, capsys, monkeypatch, var):
    """The SEIKAN_ namespace is owned. The known set is derived from the settings model's
    fields, so every knob the model does not define refuses for free: a caller exporting one
    believes an exam is active that this build does not run, and that must never pass
    silently."""
    monkeypatch.setenv(var, "0.5")
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    code, doc = _run(
        capsys, ["run", str(thesis), *_target(px), "--report-out", str(tmp_path / "r.json")]
    )
    assert code == 3
    assert doc["error"]["type"] == "thresholds_invalid"
    assert var.lower() in json.dumps(doc["error"]["errors"]).lower()


def test_unknown_seikan_env_var_also_fails_a_run_that_asks_for_no_report(
    tmp_path, capsys, monkeypatch
):
    # Thresholds are constructed on EVERY run, whichever outputs were nominated: whether a
    # SEIKAN_* var is legal cannot depend on what the caller asked seikan to write. Lazy compute
    # governs COMPUTE, never request validation — and the refusal lands before the CSV is written.
    monkeypatch.setenv("SEIKAN_THESIS_MIN_PSR", "0.5")
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "trades.csv"
    code, doc = _run(capsys, ["run", str(thesis), *_target(px), "--trades-out", str(out)])
    assert code == 3
    assert doc["error"]["type"] == "thresholds_invalid"
    assert not out.exists()


def test_out_of_domain_threshold_flag_exits_3(tmp_path, capsys):
    # A nonsense exam (a mass share above 1) must refuse at construction, never produce a
    # vacuously met cell.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    code, doc = _run(
        capsys,
        [
            "run",
            str(thesis),
            *_target(px),
            "--max-concentration",
            "2",
            "--report-out",
            str(tmp_path / "r.json"),
        ],
    )
    assert code == 3
    assert doc["error"]["type"] == "thresholds_invalid"


@pytest.mark.parametrize(
    "flags",
    [
        ["--min-trades", "5"],
        ["--min-n-nonoverlap", "4"],
        ["--max-concentration", "0.9"],
        ["--max-hypotheses", "128"],
    ],
)
def test_looser_than_canonical_threshold_exits_3(tmp_path, capsys, flags):
    # The canonical exam is the floor: a LOOSER knob can never construct an exam, so a cell
    # reported as met always means at-least-canonical rigor. Exploration loses nothing by
    # this seal — an unmet cell still carries its complete statistics.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    code, doc = _run(
        capsys, ["run", str(thesis), *_target(px), *flags, "--report-out", str(tmp_path / "r.json")]
    )
    assert code == 3, flags
    assert doc["error"]["type"] == "thresholds_invalid"


def test_in_gate_threshold_revalidation_maps_to_exit_3(tmp_path, capsys):
    # evaluate_gate revalidates the thresholds it is handed. That is unreachable through the
    # CLI — `_thresholds` built them through the same constructor — but if it ever fires it is
    # a thresholds problem, so it must land on the tiered exit-3 envelope rather than main()'s
    # generic ValidationError → dsl_invalid branch.
    from pydantic import ValidationError

    import seikan.cli as cli_mod

    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "r.json"

    def _boom(_summary, _thresholds=None):
        raise ValidationError.from_exception_data(
            "GateThresholds",
            [
                {
                    "type": "less_than_equal",
                    "loc": ("thesis_max_concentration",),
                    "input": 0.9,
                    "ctx": {"le": 0.6},
                }
            ],
        )

    monkey = pytest.MonkeyPatch()
    monkey.setattr(cli_mod, "evaluate_gate", _boom)
    try:
        code, doc = _run(capsys, ["run", str(thesis), *_target(px), "--report-out", str(out)])
    finally:
        monkey.undo()
    assert code == 3
    assert doc["error"]["type"] == "thresholds_invalid"
    assert not out.exists()  # the report is written last: a failed grading writes no document


def test_non_canonical_thresholds_stamped(tmp_path, capsys):
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    _code, doc = _run_report(
        capsys, tmp_path, ["run", str(thesis), *_target(px), "--max-concentration", "0.3"]
    )
    assert doc["identity"]["thresholds_canonical"] is False  # identity fact, not in the gate
    assert "thresholds_canonical" not in doc["gate"]
    assert doc["identity"]["thresholds"]["thesis_max_concentration"] == 0.3  # snapshot names it


def test_max_hypotheses_flag(tmp_path, capsys):
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    _code, doc = _run_report(
        capsys, tmp_path, ["run", str(thesis), *_target(px), "--max-hypotheses", "4"]
    )
    assert doc["identity"]["thresholds"]["thesis_max_hypotheses"] == 4


def test_unmet_search_cap_leaves_every_cell_unmet_at_exit_0(tmp_path, capsys):
    # Run-level evidence is the ground every per-cell read stands on, so an unmet run-level
    # check leaves EVERY cell unmet — a caller reading cells[i].met never has to AND the
    # sections itself. The run still completed, so the exit code stays 0.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json", horizon=[5, 10, 15])
    code, doc = _run_report(
        capsys, tmp_path, ["run", str(thesis), *_target(px), "--max-hypotheses", "2"]
    )
    assert code == 0
    assert _unmet_run_checks(doc) == ["search_cap"]
    assert doc["gate"]["n_cells"] == 3 and doc["gate"]["n_met"] == 0
    for cell in doc["gate"]["cells"]:
        assert cell["met"] is False
    assert len(doc["summary"]["cells"]) == 3  # the record is complete regardless


# ---- run --root-series-out --------------------------------------------------------------------


def test_root_series_writes_values_only_and_prints_nothing(tmp_path, capsys):
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "rs.csv"
    code = main(["run", str(thesis), *_target(px), "--root-series-out", str(out)])
    assert code == 0
    assert capsys.readouterr().out == ""  # silent on success: the CSV is the whole output
    frame = pd.read_csv(out)
    # VALUE columns only — the per-bar DECISION INPUT view. No 0/1 entry flags ride this CSV:
    # firings ride --trades-out, in observation shape, and --entry-flags-out per bar.
    assert list(frame.columns) == ["datetime", "percentile(close,14)"]
    assert frame["percentile(close,14)"].notna().any()  # warmup aside, the series has values


def test_root_series_overwrites_existing_file(tmp_path, capsys):
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "rs.csv"
    out.write_text("stale junk, not a csv\n", encoding="utf-8")
    code = main(["run", str(thesis), *_target(px), "--root-series-out", str(out)])
    capsys.readouterr()
    assert code == 0
    assert list(pd.read_csv(out).columns) == ["datetime", "percentile(close,14)"]


def test_root_series_bad_data_exits_2(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("datetime,close\nnot-a-date,1\n", encoding="utf-8")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "rs.csv"
    code, doc = _run(capsys, ["run", str(thesis), *_target(bad), "--root-series-out", str(out)])
    assert code == 2
    assert doc["error"]["type"] == "data_invalid"  # errors still emit the JSON envelope
    assert not out.exists()  # nothing half-written


def test_root_series_unwritable_out_is_preflighted_exit_3(tmp_path, capsys):
    # An unwritable nominated output path is refused BEFORE building signals — exit 3, a
    # `usage`-class invalid request naming the path, not an exit-4 internal error after the work.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "no_such_dir" / "rs.csv"
    code, doc = _run(capsys, ["run", str(thesis), *_target(px), "--root-series-out", str(out)])
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert str(out) in doc["error"]["message"]
    assert not out.exists()


# ---- run --entry-flags-out ---------------------------------------------------------------------


def _swept_constant_thesis(tmp_path: Path) -> tuple[Path, dict]:
    """One target, a swept threshold cutoff — so the flags frame carries one column per COMBO, each
    named by the axis it varies (``entry[cut=...]``).

    Each builder here returns its thesis WITH the mapping that locates its series: the document
    names them, so a caller that holds only the thesis holds half a run."""
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = tmp_path / "t.json"
    dsl = json.loads(_write_thesis(thesis).read_text(encoding="utf-8"))
    dsl["entry"]["right"] = {"type": "constant", "value": [0.2, 0.35, 0.5], "name": "cut"}
    thesis.write_text(json.dumps(dsl), encoding="utf-8")
    return thesis, {"target": px}


def _multi_target_thesis(tmp_path: Path) -> tuple[Path, dict]:
    """Two targets on ONE bar clock, no swept axis — so the flags frame carries one column per
    TARGET and every name takes the ``@<target>`` suffix."""
    mapping = {
        sym: _write_ohlcv(tmp_path / f"{sym}.csv", seed=seed)
        for sym, seed in (("AAA", 1), ("BBB", 2))
    }
    thesis = tmp_path / "t.json"
    dsl = json.loads(_write_thesis(thesis).read_text(encoding="utf-8"))
    dsl["data"] = {"targets": list(mapping)}
    thesis.write_text(json.dumps(dsl), encoding="utf-8")
    return thesis, mapping


def _sparse_final_bar_thesis(tmp_path: Path) -> tuple[Path, dict]:
    """A SPARSE mask (18 scattered bars of 120) that ALSO fires on the final bar, over TWO declared
    horizons — the fixture that makes "less the final one" a claim with content rather than a
    subtraction of an element that was never in the set."""
    n = 120
    px = _write_spiky_ohlcv(tmp_path / "px.csv", _scattered_with_final_bar(n), n=n)
    return _write_spiky_thesis(tmp_path / "t.json", horizon=[5, 10]), {"target": px}


def _quiet_final_bar_thesis(tmp_path: Path) -> tuple[Path, dict]:
    """The other branch of "less AT MOST the final one": a percentile threshold on the random-walk
    fixture, which fires on about half the bars but not the last — so here the two files agree on
    the bar set exactly, and the warmup bars (percentile undefined → 0) are exercised too."""
    px = _write_ohlcv(tmp_path / "px.csv")
    return _write_thesis(tmp_path / "t.json"), {"target": px}


def test_entry_flags_unwritable_out_is_preflighted_exit_3(tmp_path, capsys):
    # Every nominated path is preflighted, the fourth flag included: an unwritable one is refused
    # BEFORE the listing is built — exit 3, a `usage`-class invalid request naming the path, not an
    # exit-4 internal error after the work.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "no_such_dir" / "ef.csv"
    code, doc = _run(capsys, ["run", str(thesis), *_target(px), "--entry-flags-out", str(out)])
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert str(out) in doc["error"]["message"]
    assert not out.exists()


def test_entry_flags_csv_is_deterministic_and_overwrites(tmp_path, capsys):
    # Two identical runs, two identical files: the column layout is combo-iteration × target order,
    # so nothing about the frame is dict- or set-ordered. And the path is ALWAYS overwritten
    # wholesale — a stale file at the nominated path is replaced, never appended to, so a caller
    # never reads yesterday's mask believing it is today's.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "ef.csv"
    assert main(["run", str(thesis), *_target(px), "--entry-flags-out", str(out)]) == 0
    first = out.read_bytes()
    assert main(["run", str(thesis), *_target(px), "--entry-flags-out", str(out)]) == 0
    assert capsys.readouterr().out == ""
    assert out.read_bytes() == first

    out.write_text("stale junk, not a csv\n", encoding="utf-8")
    assert main(["run", str(thesis), *_target(px), "--entry-flags-out", str(out)]) == 0
    capsys.readouterr()
    assert out.read_bytes() == first  # replaced wholesale, no trace of the stale bytes


@pytest.mark.parametrize(
    "build, expected",
    [
        (_swept_constant_thesis, ["entry[cut=0.2]", "entry[cut=0.35]", "entry[cut=0.5]"]),
        (_multi_target_thesis, ["entry@AAA", "entry@BBB"]),
    ],
    ids=["swept-constant", "multi-target"],
)
def test_entry_flags_csv_header_is_the_library_frames_columns(tmp_path, capsys, build, expected):
    # The CSV is the library frame verbatim — the same column names in the same order, behind the
    # ISO-8601 index labelled `datetime` — because those names are the ONLY thing a caller has to
    # line a flag column up against the combo it grades. Both naming families are exercised:
    # `entry[axis=value]` for a swept entry axis, the `@<target>` suffix for a multi-target regime.
    # No `#N` disambiguation appears in either: one column per declared combo × target, and no two
    # combos are equal, so the names are unique by construction (unlike the root-series labels,
    # where two rendered expressions can legitimately collide).
    from seikan.api import list_entries
    from seikan.dsl.schema import Thesis

    from ._helpers import load

    thesis, mapping = build(tmp_path)
    out = tmp_path / "ef.csv"
    code = main(["run", str(thesis), *_data(mapping), "--entry-flags-out", str(out)])
    assert code == 0
    assert capsys.readouterr().out == ""

    t = Thesis.model_validate(json.loads(thesis.read_text(encoding="utf-8")))
    frame = list_entries(t, load(t, mapping)).entry_flags
    assert list(frame.columns) == expected  # several (combo × target) columns, canonically named
    written = pd.read_csv(out)
    assert list(written.columns) == ["datetime", *frame.columns]
    values = written.drop(columns=["datetime"])
    # 0/1 INTEGERS throughout: no NaN anywhere (an undecidable bar is a 0, not a hole), so pandas
    # never widens a column to float — the shape the root-series CSV cannot promise.
    assert all(str(dt) == "int64" for dt in values.dtypes)
    assert set(np.unique(values.to_numpy())) <= {0, 1}
    for name, column in frame.items():
        assert written[name].tolist() == column.astype(int).tolist()


@pytest.mark.parametrize(
    "build, final_fires",
    [(_sparse_final_bar_thesis, True), (_quiet_final_bar_thesis, False)],
    ids=["final-bar-fires", "final-bar-quiet"],
)
def test_flagged_bars_less_the_final_one_are_exactly_the_trades_entry_bars(
    tmp_path, capsys, build, final_fires
):
    # The bit-identical seam, as an EQUALITY OF BAR SETS rather than a count: both files read the
    # same vectorize.signal mask, so the flagged bars ARE the trades frame's entry bars, less at
    # most the final one (which anchors no observation). A count-only check would sit there happily
    # while the two files disagreed about WHICH bars fired.
    #
    # BOTH branches of "at most" are exercised, because on a fixture whose final bar never fires the
    # subtraction removes an element that was not in the set and the whole clause is a no-op — the
    # named seam would go untested while the test kept passing. `final_fires` says which branch each
    # fixture is, and it is asserted below against the mask the run actually wrote, so a fixture
    # that drifts out of its branch fails here instead of silently emptying the claim.
    #
    # The join is POSITIONAL on `entry_bar`, never on the timestamp: the flags frame is indexed by
    # the SIGNAL bar t while the trades frame's `entry_time` is the ANCHOR bar t+1, so a timestamp
    # join is off by one bar. And a multi-horizon thesis writes one trades row PER HORIZON, so the
    # comparison is between SETS of entry_bar; the row count is pinned separately, against the
    # horizon count the file itself declares.
    thesis, mapping = build(tmp_path)
    trades, flags = tmp_path / "tr.csv", tmp_path / "ef.csv"
    code = main(
        [
            "run",
            str(thesis),
            *_data(mapping),
            "--trades-out",
            str(trades),
            "--entry-flags-out",
            str(flags),
        ]
    )
    assert code == 0
    assert capsys.readouterr().out == ""
    flagged = pd.read_csv(flags)
    # one row per bar, so the row POSITION is the bar position `entry_bar` records
    fired = set(flagged.index[flagged["entry"] == 1])
    final = len(flagged) - 1
    # FIXTURE PRECONDITIONS: sparse with a real margin (a mask drifting to all-ones would make the
    # equality below compare "every bar" against "every bar"), scattered rather than one contiguous
    # run, and on the branch this parametrization claims.
    assert 0 < len(fired) < len(flagged) // 2
    ordered = sorted(fired)
    assert max(b - a for a, b in pairwise(ordered)) > 1
    assert (final in fired) is final_fires

    frame = pd.read_csv(trades)
    entered = {int(b) for b in frame["entry_bar"]}
    expected = fired - {final}
    assert len(expected) == len(fired) - (1 if final_fires else 0)  # the subtraction bites, or not
    assert entered == expected  # drop or invent one bar and this breaks
    # ONE row per (firing bar × target × horizon). The trades frame spells out an axis column only
    # for a SWEPT param, so a scalar-horizon thesis carries no `horizon` column at all and lands one
    # row per firing; the swept one lands as many as it declared.
    n_horizons = frame["horizon"].nunique() if "horizon" in frame.columns else 1
    assert len(frame) == len(entered) * n_horizons


def test_constant_only_thesis_still_writes_a_flag_column_that_re_reads_strictly(tmp_path, capsys):
    # The promise the root-series CSV explicitly cannot make. A thesis whose every threshold operand
    # is a bare constant has NO root series at all — that file comes out datetime-only and the
    # strict reader refuses it for want of a value column — but the DECISION is still defined on
    # every bar, so the flags file carries the bare `entry` column and re-reads as a series-shaped
    # strict CSV. Neither degenerate shape (a value-column-free frame, an all-NaN column) can arise
    # here: a thesis declares at least one combo and at least one target, and an undecidable bar is
    # a 0, never a hole. Both files are written from one run, so the asymmetry is the assertion.
    from seikan import dataio

    px = _write_ohlcv(tmp_path / "px.csv", n=60)
    thesis = tmp_path / "t.json"
    dsl = json.loads(_write_thesis(thesis).read_text(encoding="utf-8"))
    dsl["entry"] = {
        "type": "threshold",
        "left": {"type": "constant", "value": 1.0},
        "op": ">",
        "right": {"type": "constant", "value": 0.0},
    }
    thesis.write_text(json.dumps(dsl), encoding="utf-8")
    root_series, flags = tmp_path / "rs.csv", tmp_path / "ef.csv"
    code = main(
        [
            "run",
            str(thesis),
            *_target(px),
            "--root-series-out",
            str(root_series),
            "--entry-flags-out",
            str(flags),
        ]
    )
    assert code == 0
    assert capsys.readouterr().out == ""

    rs_lines = root_series.read_text(encoding="utf-8").splitlines()
    assert rs_lines[0] == "datetime"  # no value columns whatever
    assert len(rs_lines) == 61  # header + one row per bar
    _rs_frame, rs_rep = dataio.read_strict_csv(str(root_series))
    assert not rs_rep.ok  # the strict reader wants at least one value column

    ef_lines = flags.read_text(encoding="utf-8").splitlines()
    assert ef_lines[0] == "datetime,entry"
    assert len(ef_lines) == 61
    frame, rep = dataio.read_strict_csv(str(flags))
    assert rep.ok and rep.shape == "series"
    assert (frame["entry"] == 1.0).all()  # 1 > 0 on every bar, warmup included


# ---- check-data -----------------------------------------------------------------------------


def test_check_data_pass_and_fail(tmp_path, capsys):
    good = _write_ohlcv(tmp_path / "good.csv")
    code, doc = _run(capsys, ["check-data", str(good), "--shape", "ohlcv"])
    assert code == 0 and doc["data_report"]["ok"] is True
    # the uniform header rides check-data too (the lone non-error command whose header would
    # otherwise go unpinned)
    assert doc["command"] == "check-data"
    assert doc["report_schema_version"] == 5 and doc["seikan_version"]

    bad = tmp_path / "bad.csv"
    bad.write_text("datetime,close\n2021-01-02,100\n2021-01-01,101\n", encoding="utf-8")
    code, doc = _run(capsys, ["check-data", str(good), str(bad)])
    assert code == 2
    by_path = {f["path"]: f for f in doc["data_report"]["files"]}
    assert by_path[str(good)]["ok"] is True
    assert not by_path[str(bad)]["ok"]


# ---- schema ---------------------------------------------------------------------------------


def test_schema_emits_self_serve_reference(capsys):
    code, doc = _run(capsys, ["schema"])
    assert code == 0
    assert "properties" in doc["dsl_json_schema"]
    fields = {t["field"] for t in doc["thresholds"]}
    assert fields == {
        "thesis_min_trades",
        "thesis_min_n_nonoverlap",
        "thesis_max_concentration",
        "thesis_max_hypotheses",
    }
    assert all(t["env_var"].startswith("SEIKAN_") for t in doc["thresholds"])
    by_field = {t["field"]: t for t in doc["thresholds"]}
    assert by_field["thesis_max_hypotheses"]["cli_flag"] == "--max-hypotheses"
    assert by_field["thesis_max_hypotheses"]["default"] == 64
    assert by_field["thesis_max_concentration"]["default"] == 0.6
    assert all("choices" not in t for t in doc["thresholds"])  # no enumerated profile knob
    assert "gate_profiles" not in doc  # there is no profile system, under this or any other key
    assert doc["report_schema_version"] == 5
    assert "ISO-8601" in doc["csv_format"]["timestamp_format"]


def test_schema_documents_the_root_series_csv(capsys):
    # The CSV contract is keyed `root_series_csv`, after the flag that writes it — never
    # `list_entries_csv`, after a subcommand that does not exist. The block states what the FILE
    # holds (no entry flags), and its `no_entry_flags` text must route the reader onward: firings
    # in observation shape ride --trades-out, the raw mask rides --entry-flags-out, and only the
    # second of those can carry a final-bar firing. A text naming --trades-out ALONE would send a
    # caller looking for the live signal to the one file that structurally cannot hold it.
    _code, doc = _run(capsys, ["schema"])
    assert "list_entries_csv" not in doc
    csv_doc = doc["root_series_csv"]
    assert "--root-series-out" in csv_doc["command"]
    assert "NO 0/1 entry-flag columns" in csv_doc["no_entry_flags"]
    assert "--trades-out" in csv_doc["no_entry_flags"]  # and where the firings ride instead
    assert "--entry-flags-out" in csv_doc["no_entry_flags"]  # ...and where the 0/1 flags ride
    assert "FINAL" in csv_doc["no_entry_flags"]  # the seam is named, not left to be inferred


def test_schema_documents_the_entry_flags_csv_between_root_series_and_exit_codes(capsys):
    # The fourth output gets its own contract block, positioned with the other CSV contracts: an
    # agent reads csv_format → root_series_csv → entry_flags_csv → exit_codes in one pass, so the
    # decision-side pair sits together rather than one of them turning up after the exit codes.
    _code, doc = _run(capsys, ["schema"])
    keys = list(doc)
    assert keys.index("entry_flags_csv") == keys.index("root_series_csv") + 1
    assert keys.index("entry_flags_csv") == keys.index("exit_codes") - 1
    csv_doc = doc["entry_flags_csv"]
    assert set(csv_doc) == {"command", "rows", "flag_columns", "relation_to_trades", "roundtrip"}
    assert "--entry-flags-out" in csv_doc["command"]
    assert "datetime" in csv_doc["rows"]
    # the two naming families, and the absence of the root-series `#N` disambiguation
    assert "entry[axis=value,...]" in csv_doc["flag_columns"]
    assert "@<target>" in csv_doc["flag_columns"]
    # the seam this output exists for, stated where a caller reading the contract will meet it
    assert "FINAL" in csv_doc["relation_to_trades"]
    assert "outcome_coverage" in csv_doc["relation_to_trades"]
    # and the promise root_series_csv explicitly withholds: this one ALWAYS re-reads
    assert "always re-reads" in csv_doc["roundtrip"]
    assert "unless" in doc["root_series_csv"]["roundtrip"]  # the caveated twin, for contrast


def test_schema_documents_the_trades_csv_between_csv_format_and_root_series(capsys):
    # Every output CSV carries its own contract block, positioned so the three of them read in
    # nomination order: trades → root_series → entry_flags.
    _code, doc = _run(capsys, ["schema"])
    keys = list(doc)
    assert keys.index("trades_csv") == keys.index("csv_format") + 1
    assert keys.index("trades_csv") == keys.index("root_series_csv") - 1
    csv_doc = doc["trades_csv"]
    assert set(csv_doc) == {"command", "rows", "columns", "join", "derived_views"}
    assert "--trades-out" in csv_doc["command"]
    # The episode-ledger derivation note — cells[*].episodes is a deterministic derivable
    # function of this CSV (same frozen overlap merge), which is WHY
    # there is no --episodes-out flag; the CSV is the un-truncated rebuild path past the cap.
    assert "episodes" in csv_doc["derived_views"]
    assert "--episodes-out" in csv_doc["derived_views"]  # the absent flag is named, on purpose
    assert "overlap merge" in csv_doc["derived_views"]
    assert "cap" in csv_doc["derived_views"]
    cols = csv_doc["columns"]
    assert set(cols) == {
        "<swept axes>",
        "horizon",
        "target",
        "entry_time",
        "exit_time",
        "entry_bar",
        "entry_px",
        "exit_px",
        "ret",
        "ret_raw",
        "ret_bench",
        "pre_ret",
        "mae",
        "mfe",
        "bars_to_positive",
        "bars_to_trough",
        "exit_reason",
        "is_open",
        "<features>",
    }
    # the join rule and the off-by-one seam are stated where a consumer will meet them
    assert "row position" in cols["entry_bar"] and "timestamps" in cols["entry_bar"]
    for reason in ("horizon", "open", "no_outcome", "no_benchmark"):
        assert reason in cols["exit_reason"]
    # the no-epoch-ns proof: the epoch-ns twins appear only as the explicit absence notice
    assert "entry_ts" not in json.dumps(cols)
    assert "no epoch-ns" in csv_doc["join"] and "entry_ts" in csv_doc["join"]


def test_schema_report_fields_dictionary_is_schema_side_only(tmp_path, capsys):
    # The OUTPUT-side twin of dsl_json_schema. Schema-side only: a caller caches
    # `seikan schema` once and holds every definition, while the report stays lean
    # — the same split that keeps METRIC_ROLES_DOC out of the report.
    _code, doc = _run(capsys, ["schema"])
    keys = list(doc)
    assert keys.index("report_fields") == keys.index("gate_contract") + 1
    rf = doc["report_fields"]
    assert set(rf) == {"identity", "conventions", "run", "cells"}
    assert set(rf["conventions"]) == {"alignment", "nulls", "units", "grid", "caveats"}
    assert "POSITIONALLY" in rf["conventions"]["alignment"]
    assert "no rollup" in rf["conventions"]["grid"]
    assert "summary.outcome.units" in rf["conventions"]["units"]
    assert "by_target.boot" in rf["cells"] and "by_target.subperiods" in rf["cells"]
    assert "bar_spacing" in rf["run"] and "outcome" in rf["run"] and "params / targets" in rf["run"]
    assert (
        "Pearson"
        in rf["cells"][
            "by_target.win_loss_ratio / std_ret / skewness / kurtosis / tail_ratio / cvar_5"
        ]
    )
    # ...and neither output dictionary rides the run report
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    _c, run_doc = _run_report(capsys, tmp_path, ["run", str(thesis), *_target(px)])
    assert "report_fields" not in run_doc and "trades_csv" not in run_doc


def test_schema_describe_keys_in_order(tmp_path, capsys):
    # The describe subcommand's two contract keys. `describe_report`
    # (the field dictionary) sits right after `report_fields` — the two output-side references
    # together — and `describe_roles` (the compact map every describe document stamps) right
    # after `metric_roles`, mirroring how the run report's compact map is placed. The dictionary
    # is schema-side ONLY; the roles ride the describe document identically (pinned from the
    # describe side in test_describe.py).
    _code, doc = _run(capsys, ["schema"])
    keys = list(doc)
    assert keys.index("describe_report") == keys.index("report_fields") + 1
    assert keys.index("describe_roles") == keys.index("metric_roles") + 1
    dr = doc["describe_report"]
    assert set(dr) == {"document", "blocks", "profile"}
    assert "check-data" in dr["document"]["exit_codes"]  # the parity contract is named
    assert "argument" in dr["document"]["order"].lower()  # ...and the never-sorted rule
    assert "n_bars" in dr["document"]["bounded_output"]
    roles = doc["describe_roles"]
    assert set(roles) == {"claim", "caveats", "scope_boundary"}
    assert "MEASURES NOTHING" in roles["claim"]
    # a run report carries metric_roles but NEITHER describe key — they belong to schema and
    # to the describe document respectively
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    _c, run_doc = _run_report(capsys, tmp_path, ["run", str(thesis), *_target(px)])
    assert "describe_report" not in run_doc and "describe_roles" not in run_doc


def test_schema_gate_contract_is_the_per_cell_checklist(capsys):
    _code, doc = _run(capsys, ["schema"])
    contract = doc["gate_contract"]
    assert set(contract) == {
        "contract",
        "claim",
        "evidence_basis",
        "thresholds",
        "run_checks",
        "cell_checks",
        "evidence_only",
        "metric_roles_rationale",
    }
    assert set(contract["run_checks"]) == {"evidence_complete", "source_coverage", "search_cap"}
    assert set(contract["cell_checks"]) == {
        "cell_evidence",
        "outcome_coverage",
        "signal_coverage",
        "support",
        "concentration",
    }
    # the claim disclaims exactly what an inferential exam would assert
    assert "NO significance claim" in contract["claim"]
    assert "not a verdict" in contract["claim"]
    assert "FULL SAMPLE" in contract["evidence_basis"]
    # 'open' censoring is structural, there being no embargo — the contract says so
    assert "ALLOWED" in contract["cell_checks"]["outcome_coverage"]
    # the contract names no inferential check under any spelling, and there is no flat `checks`
    # list — that shape described a single verdict, and there is none
    assert "checks" not in contract
    for absent in ("oos_confirmation", "selection_integrity"):
        assert absent not in json.dumps(contract)
    # the remaining multiplicity input is named where the caller must read it
    assert "n_hypotheses_attempted" in contract["run_checks"]["search_cap"]


def test_schema_documents_every_exit_code(capsys):
    # Agents self-serve the code meanings from here, so the block must cover exactly the codes the
    # CLI can emit — and code 0 must say in words that it is not a verdict.
    _code, doc = _run(capsys, ["schema"])
    codes = doc["exit_codes"]
    assert set(codes) == {"0", "2", "3", "4"}
    assert "NOT a verdict" in codes["0"]
    assert codes["2"].startswith("input data")
    assert codes["4"] == "internal error"


def test_exit_zero_text_scopes_the_stdout_silence_to_run(tmp_path, capsys):
    # `exit_codes` covers EVERY subcommand, so it must not promise silence globally: `check-data`
    # and `schema` both exit 0 with a document on stdout. Agents self-serve from this block, and one
    # that read a universal silence claim would discard check-data's data_report unread — or treat
    # the non-empty stdout as a stale-binary signal against a perfectly current build.
    px = _write_ohlcv(tmp_path / "px.csv")
    check_code, check_doc = _run(capsys, ["check-data", str(px)])
    assert check_code == 0 and check_doc["data_report"]["ok"]  # exit 0 WITH stdout content

    _code, doc = _run(capsys, ["schema"])
    text = doc["exit_codes"]["0"]
    assert "check-data" in text and "schema" in text  # the exceptions are named, not implied
    assert "`run`" in text  # and the silence is attributed to the command that actually has it


def test_schema_metric_roles_is_the_compact_map_prose_rides_gate_contract(tmp_path, capsys):
    # schema's `metric_roles` is the IDENTICAL compact map the run report
    # stamps (one shape everywhere) — the PROSE rationale lives under
    # gate_contract.metric_roles_rationale, so no field is a dict in one command and a list in
    # another.
    _code, schema_doc = _run(capsys, ["schema"])
    roles = schema_doc["metric_roles"]
    assert set(roles) == {
        "claim",
        "run_checks",
        "cell_checks",
        "evidence_only",
        "metric_classes",
        "caveats",
        "scope_boundary",
    }
    assert isinstance(roles["run_checks"], dict) and isinstance(roles["cell_checks"], dict)
    assert any("anti-conservative" in e for e in roles["evidence_only"])
    # the compact map is byte-identical to the one the run report carries
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    _c, run_doc = _run_report(capsys, tmp_path, ["run", str(thesis), *_target(px)])
    assert roles == run_doc["metric_roles"]
    # the prose rationale (lists of sentences) lives under gate_contract, keyed distinctly
    rationale = schema_doc["gate_contract"]["metric_roles_rationale"]
    assert "no significance claim" in rationale["claim"].lower()
    assert any("anti-conservative" in e for e in rationale["evidence_only"])
    # the long caveat mechanics stay schema-side; the compact one-liners ride every report
    assert set(rationale["caveats"]) == set(roles["caveats"])
    assert "sqrt(2/3)" in rationale["caveats"]["t_hac"]


def test_schema_immune_to_broken_env(capsys, monkeypatch):
    # `schema` reports CLASS defaults from model_fields — a polluted or broken SEIKAN_* env
    # neither crashes it nor masquerades as the default exam.
    monkeypatch.setenv("SEIKAN_THESIS_MIN_TRADES", "99")
    monkeypatch.setenv("SEIKAN_BOGUS_VAR", "1")
    code, doc = _run(capsys, ["schema"])
    assert code == 0
    by_field = {t["field"]: t for t in doc["thresholds"]}
    assert by_field["thesis_min_trades"]["default"] == 30  # the class default, not the env value


def test_schema_markdown_is_pinned_to_the_live_contract(capsys):
    # The shipped agent-facing guide is hand-maintained and read out of the wheel verbatim, so
    # the stamps and field lists it states are pinned to the code — the one place a stale
    # version number or a five-point quantile block could otherwise ship unnoticed.
    from seikan.analysis.stats import STATISTICS_VERSION
    from seikan.gate import POLICY_VERSION
    from seikan.types import PboBlock, ReportIdentity

    assert main(["schema", "--markdown"]) == 0
    guide = capsys.readouterr().out
    assert f"`report_schema_version` **{REPORT_SCHEMA_VERSION}**" in guide
    assert f"`gate.policy_version` ({POLICY_VERSION})" in guide
    assert STATISTICS_VERSION == 4  # the guide describes the v4 estimators; bump both together
    assert "{p05, p10, p25, p50, p75, p90, p95}" in guide and "{p10, p25" not in guide
    for field in typing.get_type_hints(ReportIdentity):
        assert f"`{field}`" in guide, field
    for field in typing.get_type_hints(PboBlock):
        assert field in guide, field
    for retired in ("stats_table", "by_param", "t_iid", "bars_held", "n_stats_rows"):
        # named only as retired — inside the "Not in the report" list or a changelog aside
        assert guide.count(f"`{retired}`") <= 2, retired
    assert "4 = internal" in guide


def test_schema_markdown(capsys):
    code = main(["schema", "--markdown"])
    out = capsys.readouterr().out
    assert code == 0
    assert "#" in out  # markdown headings; content is the DSL guide


def test_version_field_present_everywhere(tmp_path, capsys):
    code, doc = _run(capsys, ["schema"])
    import seikan

    assert code == 0
    assert doc["seikan_version"] == seikan.__version__


# ---- the uniform envelope: usage errors, preflight, provenance -------------------------------


def test_usage_errors_emit_exit_3_envelopes(capsys):
    # Every argparse usage error is a clean exit-3 `usage` envelope on stdout (uniform top-level
    # header). `command` is the subcommand when the failing parser knows it, else null.
    cases = [
        ([], None),  # no subcommand
        (["run"], "run"),  # missing positional
        (["run", "t.json", "--min-trades", "x"], "run"),  # bad type= conversion
        (["frobnicate"], None),  # invalid subcommand choice
    ]
    for argv, expected_command in cases:
        code, doc = _run(capsys, argv)
        assert code == 3, argv
        assert set(doc["error"]) == {"type", "message"}
        assert doc["error"]["type"] == "usage" and doc["error"]["message"]
        assert doc["command"] == expected_command
        assert doc["report_schema_version"] == 5 and doc["seikan_version"]


def test_help_and_version_are_conventional_non_json_exit_0(capsys):
    # --help/--version bypass argparse's error() hook, so they stay conventional: SystemExit(0)
    # and plain text on stdout, never a JSON envelope, never swallowed into an exit-4 report.
    for argv in (["--version"], ["--help"]):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert out.strip() and not out.strip().startswith("{")


def test_trades_out_preflight_missing_dir_exit_3(tmp_path, capsys):
    # An unwritable --trades-out path is refused BEFORE the O(grid × length) run — exit 3 usage,
    # naming the path, and nothing created.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "no_such_dir" / "trades.csv"
    code, doc = _run(capsys, ["run", str(thesis), *_target(px), "--trades-out", str(out)])
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert str(out) in doc["error"]["message"]
    assert not out.exists()


def test_report_out_preflight_missing_dir_exit_3(tmp_path, capsys):
    # Every nominated path is preflighted, in flag order — the report's included, so a caller
    # learns the directory is missing before paying for the measurement that would fill it.
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "no_such_dir" / "report.json"
    code, doc = _run(capsys, ["run", str(thesis), *_target(px), "--report-out", str(out)])
    assert code == 3
    assert doc["error"]["type"] == "usage"
    assert str(out) in doc["error"]["message"]
    assert not out.exists()


def test_trades_out_preflight_leaves_existing_file_untouched(tmp_path, capsys):
    # The preflight opens an existing output for APPEND (never truncates); a run that then fails on
    # the data (exit 2) leaves the file's bytes intact.
    bad = tmp_path / "bad.csv"
    bad.write_text("datetime,close\nnot-a-date,1\n", encoding="utf-8")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "trades.csv"
    out.write_text("PRECIOUS", encoding="utf-8")
    code, _doc = _run(capsys, ["run", str(thesis), *_target(bad), "--trades-out", str(out)])
    assert code == 2  # data error, after the preflight passed
    assert out.read_text(encoding="utf-8") == "PRECIOUS"


def test_thresholds_provenance_records_default_env_cli(tmp_path, capsys, monkeypatch):
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    _c, doc = _run_report(capsys, tmp_path, ["run", str(thesis), *_target(px)])
    assert set(doc["identity"]["thresholds_provenance"].values()) == {"default"}
    monkeypatch.setenv("SEIKAN_THESIS_MIN_TRADES", "99")
    _c, doc = _run_report(capsys, tmp_path, ["run", str(thesis), *_target(px)])
    prov = doc["identity"]["thresholds_provenance"]
    assert prov["thesis_min_trades"] == "env" and prov["thesis_min_n_nonoverlap"] == "default"
    _c, doc = _run_report(
        capsys, tmp_path, ["run", str(thesis), *_target(px), "--min-trades", "45"]
    )
    prov = doc["identity"]["thresholds_provenance"]
    assert prov["thesis_min_trades"] == "cli"  # a flag wins over the env var


def test_emit_allow_nan_backstop_raises_on_non_finite(monkeypatch):
    # json_safe maps every non-finite float to null, so this never fires in practice — but if it
    # regressed, _emit must RAISE rather than write invalid `NaN` JSON. Neuter json_safe to prove
    # the json.dumps(allow_nan=False) backstop is real.
    from seikan import cli as cli_mod

    monkeypatch.setattr(cli_mod, "json_safe", lambda d: d)
    with pytest.raises(ValueError):
        cli_mod._emit({"x": float("nan")})


def test_duplicate_json_key_refuses_as_dsl_invalid(tmp_path, capsys):
    # Python's decoder keeps the LAST occurrence of a repeated object key, so `extra="forbid"`
    # never sees the shadowed spelling — the parse refuses instead, naming the key.
    thesis = _write_thesis(tmp_path / "t.json")
    text = thesis.read_text(encoding="utf-8")
    dup = tmp_path / "dup.json"
    dup.write_text('{"name": "shadowed", ' + text.removeprefix("{"), encoding="utf-8")
    code, doc = _run(capsys, ["hash", str(dup)])
    assert code == 3
    assert doc["error"]["type"] == "dsl_invalid"
    assert "'name'" in doc["error"]["message"] and "more than once" in doc["error"]["message"]


def test_duplicate_json_key_at_depth_refuses(tmp_path, capsys, monkeypatch):
    # The hook fires at EVERY object depth, and through the stdin mouth too.
    thesis = _write_thesis(tmp_path / "t.json")
    doc = json.loads(thesis.read_text(encoding="utf-8"))
    horizon = json.dumps(doc["params"]["horizon"])
    nested = json.dumps(doc).replace(
        f'"horizon": {horizon}', f'"horizon": {horizon}, "horizon": {horizon}', 1
    )
    assert nested.count('"horizon"') == 2  # the seam actually landed
    _stdin_bytes(monkeypatch, nested.encode("utf-8"))
    code, envelope = _run(capsys, ["hash", "-"])
    assert code == 3
    assert envelope["error"]["type"] == "dsl_invalid"
    assert "'horizon'" in envelope["error"]["message"]


def test_identity_environment_stamps_the_numeric_stack(tmp_path, capsys):
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    _code, doc = _run_report(capsys, tmp_path, ["run", str(thesis), *_target(px)])
    env = doc["identity"]["environment"]
    assert set(env) == {"python", "numpy", "pandas", "scipy", "numba"}
    assert all(isinstance(v, str) and v for v in env.values())


def test_outputs_leave_no_tmp_residue(tmp_path, capsys):
    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    code, _doc = _run_report(
        capsys,
        tmp_path,
        [
            "run",
            str(thesis),
            *_target(px),
            "--trades-out",
            str(tmp_path / "trades.csv"),
            "--root-series-out",
            str(tmp_path / "roots.csv"),
            "--entry-flags-out",
            str(tmp_path / "flags.csv"),
        ],
    )
    assert code == 0
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_report_write_leaves_previous_report_intact(tmp_path, capsys, monkeypatch):
    # Atomicity: the report is serialized (and contract-validated) into a temp file and renamed
    # over the target in one step, so a mid-write failure can never leave a truncated document
    # wearing the nominated name — the previous run's good report survives.
    import os as os_mod

    px = _write_ohlcv(tmp_path / "px.csv")
    thesis = _write_thesis(tmp_path / "t.json")
    out = tmp_path / "report.json"
    out.write_text("PRECIOUS", encoding="utf-8")
    real_replace = os_mod.replace

    def _boom(src, dst):
        if str(dst) == str(out):
            raise OSError(28, "No space left on device")
        return real_replace(src, dst)

    monkeypatch.setattr("seikan.serialize.os.replace", _boom)
    code, doc = _run(capsys, ["run", str(thesis), *_target(px), "--report-out", str(out)])
    assert code == 4  # a mid-run write failure is the documented internal tier
    assert doc["error"]["type"] == "internal"
    assert out.read_text(encoding="utf-8") == "PRECIOUS"
    assert not list(tmp_path.glob("*.tmp"))
