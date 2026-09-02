"""The emitted-document contract has runtime teeth — ``seikan.emitted`` through the CLI seam.

Every OTHER test in this suite already exercises the happy path for free: ``cli._dumps``
validates each success document against the ``types.py`` TypedDicts before writing, so a builder
key no TypedDict declares — or a nullability the schema missed — fails loudly everywhere. What
this file pins is the machinery itself: that violations actually REFUSE (deep extra key, missing
section, wrong type, numeric-string coercion), that the refusal reaches the caller as the exit-4
``internal`` envelope and never as a mis-classed ``dsl_invalid``, that a parsed-back document
still validates (the bytes written are the bytes that validated), and that the document roots
stay a subset of the assembly shape the builders write against.
"""

from __future__ import annotations

import copy
import json
import typing

import numpy as np
import pandas as pd
import pytest

import seikan.cli as cli_module
from seikan.cli import main
from seikan.emitted import (
    CheckDataDocument,
    DescribeDocument,
    HashDocument,
    ReportContractError,
    RunReportDocument,
    validate_emitted,
    validate_summary,
)
from seikan.types import EmittedDocument

# ---- fixtures: one tiny real run, documents parsed back off the wire ------------------------


def _write_ohlcv(path, n_bars: int = 400, seed: int = 3) -> None:
    rng = np.random.RandomState(seed)
    px = np.full(n_bars, 100.0)
    pos = 10
    while pos < n_bars - 5:
        px[pos] = 94.0
        px[pos + 1 : pos + 4] = (96.5, 98.0, 99.5)
        pos += int(rng.randint(5, 10))
    px = px * (1.0 + rng.normal(0.0, 0.002, size=n_bars))
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="1D")
    s = pd.Series(px, index=idx)
    df = pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 1000.0}, index=idx)
    df.index.name = "datetime"
    df.to_csv(path)


def _thesis(path) -> None:
    path.write_text(
        json.dumps(
            {
                "name": "contract-probe",
                "data": {"targets": ["target"]},
                "entry": {
                    "type": "threshold",
                    "left": {"type": "field", "column": "close"},
                    "op": "<",
                    "right": {"type": "constant", "value": 95.5},
                },
                "params": {"horizon": 3},
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def run_doc(tmp_path, capsys) -> dict:
    px = tmp_path / "px.csv"
    thesis = tmp_path / "t.json"
    report = tmp_path / "report.json"
    _write_ohlcv(px)
    _thesis(thesis)
    code = main(
        [
            "run",
            str(thesis),
            "--data",
            f"target={px}",
            "--report-out",
            str(report),
        ]
    )
    assert code == 0 and capsys.readouterr().out == ""
    return json.loads(report.read_text(encoding="utf-8"))


# ---- the bytes written are the bytes that validated -----------------------------------------


def test_a_written_report_parses_back_into_a_valid_document(run_doc):
    # The seam validated this exact payload before writing; the parsed-back JSON must validate
    # too, or the wire format and the contract have diverged (a Timestamp that serialized to
    # something json round-trips differently, a numeric key surviving as a number, …).
    validate_emitted("run", run_doc)


def test_the_other_success_documents_validate_off_the_wire(tmp_path, capsys):
    px = tmp_path / "px.csv"
    thesis = tmp_path / "t.json"
    _write_ohlcv(px)
    _thesis(thesis)

    assert main(["hash", str(thesis)]) == 0
    validate_emitted("hash", json.loads(capsys.readouterr().out))

    assert main(["check-data", str(px)]) == 0
    validate_emitted("check-data", json.loads(capsys.readouterr().out))

    assert main(["describe", str(px)]) == 0
    validate_emitted("describe", json.loads(capsys.readouterr().out))


# ---- violations refuse ----------------------------------------------------------------------


def test_a_deep_extra_key_refuses(run_doc):
    # extra="forbid" must propagate from the document ROOT into every nested TypedDict — this is
    # the drift detector, and this probe (four levels down, inside a per-target evidence block)
    # is also the regression pin on pydantic's config-propagation behavior.
    doc = copy.deepcopy(run_doc)
    doc["summary"]["cells"][0]["by_target"]["target"]["boot"]["stray"] = 1
    with pytest.raises(ReportContractError, match="stray"):
        validate_emitted("run", doc)


def test_a_missing_required_section_refuses(run_doc):
    doc = copy.deepcopy(run_doc)
    del doc["gate"]
    with pytest.raises(ReportContractError, match="gate"):
        validate_emitted("run", doc)


def test_a_wrong_typed_count_refuses(run_doc):
    doc = copy.deepcopy(run_doc)
    doc["summary"]["n_bars"] = "400"  # strict: a numeric STRING must not coerce into a count
    with pytest.raises(ReportContractError, match="n_bars"):
        validate_emitted("run", doc)


def test_a_bool_does_not_coerce_into_a_number(run_doc):
    doc = copy.deepcopy(run_doc)
    doc["summary"]["cells"][0]["by_target"]["target"]["mean_ret"] = True
    with pytest.raises(ReportContractError):
        validate_emitted("run", doc)


def test_null_is_legal_exactly_where_the_schema_says(run_doc):
    # NaN → null nullability is part of the contract: a stat field admits null…
    doc = copy.deepcopy(run_doc)
    doc["summary"]["cells"][0]["by_target"]["target"]["mean_ret"] = None
    validate_emitted("run", doc)
    # …a count does not.
    doc["summary"]["cells"][0]["by_target"]["target"]["n"] = None
    with pytest.raises(ReportContractError):
        validate_emitted("run", doc)


def test_an_unknown_document_kind_refuses(run_doc):
    with pytest.raises(ReportContractError, match="no emitted-document contract"):
        validate_emitted("schema", run_doc)


def test_a_drifted_metric_roles_blob_refuses(run_doc):
    # The verbatim blobs are equality-checked against their contract.py source — stronger than
    # deep validation for a constant.
    doc = copy.deepcopy(run_doc)
    doc["metric_roles"]["claim"] = "tampered"
    with pytest.raises(ReportContractError, match="metric_roles"):
        validate_emitted("run", doc)


# ---- the library seam validates the engine's own output ---------------------------------------


def test_compile_thesis_validates_the_summary_it_returns(run_doc):
    # `api.compile_thesis` runs the RunSummary adapter over its own output before handing it
    # back, so the library path is checked exactly like the CLI document is — the report the
    # fixture wrote proves the in-memory summary passed. Directly: the emitted summary validates,
    # a deep extra key refuses, and a retired key (the schema-v4 breakdown) refuses too.
    validate_summary(run_doc["summary"])
    drifted = copy.deepcopy(run_doc["summary"])
    drifted["cells"][0]["by_target"]["target"]["stray"] = 1.0
    with pytest.raises(ReportContractError, match="stray"):
        validate_summary(drifted)
    retired = copy.deepcopy(run_doc["summary"])
    retired["stats_table"] = []
    with pytest.raises(ReportContractError, match="stats_table"):
        validate_summary(retired)


def test_summary_vocabularies_are_closed(run_doc):
    # The closed Literal vocabularies are enforced by the strict adapter: a reason outside its
    # set, a stamp outside its set, and a provenance outside {default, env, cli} all refuse.
    for path, value in (
        (("cells", 0, "by_target", "target", "boot", "reason"), "made_up"),
        (("gate_evidence_basis",), "holdout"),
        (("target_shape",), "bars"),
        (("pbo", "reason"), "unlucky"),
    ):
        doc = copy.deepcopy(run_doc["summary"])
        node = doc
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        with pytest.raises(ReportContractError, match=str(path[-1])):
            validate_summary(doc)
    ident = copy.deepcopy(run_doc)
    ident["identity"]["thresholds_provenance"]["thesis_min_trades"] = "config_file"
    with pytest.raises(ReportContractError, match="thresholds_provenance"):
        validate_emitted("run", ident)


# ---- the refusal reaches the caller as exit 4, never as a caller-blaming class --------------


def test_contract_violation_exits_4_with_the_internal_envelope(tmp_path, capsys, monkeypatch):
    # Drift the blob the CLI embeds away from the source emitted.py compares against: the seam
    # must refuse, and the refusal must ride the catch-all to the exit-4 `internal` envelope —
    # NOT the exit-3 `dsl_invalid` branch, which would blame the caller's thesis for the
    # verifier's own bug. (A raw pydantic ValidationError escaping would do exactly that, which
    # is why ReportContractError is a RuntimeError.)
    px = tmp_path / "px.csv"
    thesis = tmp_path / "t.json"
    report = tmp_path / "report.json"
    _write_ohlcv(px)
    _thesis(thesis)
    monkeypatch.setattr(cli_module, "METRIC_ROLES", {"claim": "drifted"})
    code = main(
        [
            "run",
            str(thesis),
            "--data",
            f"target={px}",
            "--report-out",
            str(report),
        ]
    )
    assert code == 4
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["error"]["type"] == "internal"
    assert "metric_roles" in envelope["error"]["message"]
    assert not report.exists()  # no out-of-contract artifact lands on disk


# ---- the roots stay anchored to the assembly shape ------------------------------------------


def test_document_roots_are_subsets_of_the_assembly_shape():
    # EmittedDocument is the all-NotRequired shape the builders write against; each command root
    # restates its sections as REQUIRED but may never invent a key the assembly shape lacks.
    assembly = set(typing.get_type_hints(EmittedDocument))
    for root in (RunReportDocument, HashDocument, CheckDataDocument, DescribeDocument):
        assert set(typing.get_type_hints(root)) <= assembly, root.__name__


def test_report_fields_documents_every_declared_panel_field():
    # types.py DECLARES the emitted shapes (machine-checked at emission); contract.py's
    # REPORT_FIELDS DOCUMENTS them (emitted by `seikan schema`); nothing else ties the two, so a
    # field added to a TypedDict but never documented would drift silently. Weak-but-real pin:
    # every declared field name of the load-bearing panels must appear somewhere in the
    # REPORT_FIELDS text (its keys are prose-compound, so set equality is not expressible).
    import json
    import typing

    from seikan import types as t
    from seikan.contract import DESCRIBE_REPORT, REPORT_FIELDS

    text = json.dumps(REPORT_FIELDS)
    for td_name in (
        "RunSummary",
        "SummaryCell",
        "CellTargetPanel",
        "CellPooledPanel",
        "PboBlock",
        "EpisodeProfileBlock",
        "EpisodeStatsBlock",
        "ReportIdentity",
    ):
        hints = typing.get_type_hints(getattr(t, td_name))
        missing = [k for k in hints if k not in text]
        assert not missing, f"{td_name} fields {missing} undocumented in REPORT_FIELDS"
    describe_text = json.dumps(DESCRIBE_REPORT)
    for td_name in ("FileProfile", "SeriesProfile"):
        hints = typing.get_type_hints(getattr(t, td_name))
        missing = [k for k in hints if k not in describe_text]
        assert not missing, f"{td_name} fields {missing} undocumented in DESCRIBE_REPORT"
