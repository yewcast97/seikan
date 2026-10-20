"""The emitted-document contract has runtime teeth — ``seikan.emitted`` through the CLI seam.

Every OTHER test in this suite already exercises the happy path for free: ``cli._dumps``
validates each success document against the ``seikan.types`` TypedDicts before writing, so a
builder key no TypedDict declares — or a nullability the schema missed — fails loudly
everywhere. What this file pins is the machinery itself: that violations actually REFUSE (deep
extra key, missing section, wrong type, numeric-string coercion), that the refusal reaches the
caller as the exit-4 ``internal`` envelope and never as a mis-classed ``dsl_invalid``, and that a
parsed-back document still validates (the bytes written are the bytes that validated).
"""

from __future__ import annotations

import copy
import json
import typing

import numpy as np
import pandas as pd
import pytest

import seikan.cli as cli_module
import seikan.compiler.runner as runner_module
from seikan.cli import main
from seikan.emitted import (
    ReportContractError,
    validate_emitted,
    validate_summary,
)

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


@pytest.fixture
def describe_doc(tmp_path, capsys) -> dict:
    px = tmp_path / "px.csv"
    _write_ohlcv(px)
    assert main(["describe", str(px), "--windows", "1,5"]) == 0
    return json.loads(capsys.readouterr().out)


def test_ratio_reason_is_a_closed_vocabulary_and_whole_refusals_need_none(describe_doc):
    doc = copy.deepcopy(describe_doc)
    block = doc["profiles"][0]["series"]["close"]["changes"]["1"]
    block["pct"], block["ratio_reason"] = None, "not_representable"  # a pct past the range
    validate_emitted("describe", doc)
    block["ratio_reason"] = "overflow"  # …and the vocabulary stays closed around it
    with pytest.raises(ReportContractError, match="ratio_reason"):
        validate_emitted("describe", doc)
    # a block refused WHOLE is all-null under its structural reason, no ratio_reason owed —
    # and that vocabulary is closed too
    block.update(diff=None, pct=None, log=None, reason="insufficient_bars", ratio_reason=None)
    validate_emitted("describe", doc)
    block["reason"] = "overflow"
    with pytest.raises(ReportContractError, match="'endpoint_missing' or 'not_representable'"):
        validate_emitted("describe", doc)


# ---- validate_summary: the engine summary's own contract ----------------------------------------


def test_validate_summary_holds_a_summary_to_the_run_root(run_doc):
    # A library caller's check of an engine summary, without a document: the emitted summary
    # validates, a deep extra key refuses, and a retired key (the schema-v4 breakdown) too.
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


def _patch_pbo(summary: dict, fields: dict) -> None:
    """Write ``fields`` into a summary's CSCV block: the search burden at the summary level,
    every other key into the ``pbo`` block."""
    for key, value in fields.items():
        (summary if key == "n_hypotheses_attempted" else summary["pbo"])[key] = value


# The fixture's block is a one-candidate `single_combo`: 1 / 1 / 1 under a burden of 1, nothing
# split (0 / 0) and n_candidates_min 0 — so each case below breaks exactly ONE link (a case that
# raises a rung raises the burden with it wherever the declared rung must stay equal to it; a
# negative n_combos necessarily breaks the per-split link too, and the population ledger
# reports first).
@pytest.mark.parametrize(
    ("ledger", "message"),
    [
        (
            {
                "n_combos": 3,
                "n_combos_scoreable": 2,
                "n_combos_declared": 2,
                "n_hypotheses_attempted": 2,
            },
            "population ledger .* violated: 3 / 2 / 2 / 2",
        ),
        (
            {
                "n_combos": 1,
                "n_combos_scoreable": 3,
                "n_combos_declared": 2,
                "n_hypotheses_attempted": 2,
            },
            "population ledger .* violated: 1 / 3 / 2 / 2",
        ),
        ({"n_combos_declared": 2}, "population ledger .* violated: 1 / 1 / 2 / 1"),
        ({"n_hypotheses_attempted": 2}, "population ledger .* violated: 1 / 1 / 1 / 2"),
        ({"n_combos": -1}, "population ledger .* violated: -1 / 1 / 1 / 1"),
        ({"n_splits": 7, "n_splits_attempted": 6}, "per-split ledger .* violated: 7 / 6, 0 / 1"),
        ({"n_splits": -1, "n_splits_attempted": -1}, "per-split ledger .* violated: -1 / -1"),
        ({"n_candidates_min": 2}, "per-split ledger .* violated: 0 / 0, 2 / 1"),
        ({"n_candidates_min": -1}, "per-split ledger .* violated: 0 / 0, -1 / 1"),
    ],
    ids=[
        "distinct_over_scoreable",
        "scoreable_over_declared",
        "declared_over_the_burden",
        "burden_over_the_declared",
        "negative_distinct",
        "splits_over_attempted",
        "negative_splits",
        "candidates_over_combos",
        "negative_candidates_min",
    ],
)
def test_an_impossible_pbo_ledger_refuses_at_both_seams(run_doc, ledger, message):
    # Every count here is a well-typed non-null int, so the shape validation passes it — only
    # the cross-field ledger arithmetic (the AfterValidator on both summary-carrying roots) can
    # see that the block is internally impossible, and it must refuse at the library seam and
    # at the emission seam alike. The declared rung is the summary's own search burden: a block
    # whose declared grid disagrees with `n_hypotheses_attempted` ranked a different search.
    # Both refusals name where the broken rule is declared — the ledger lives in seikan.emitted,
    # not in the seikan.types shape.
    summary = copy.deepcopy(run_doc["summary"])
    _patch_pbo(summary, ledger)
    with pytest.raises(ReportContractError, match=message) as refused:
        validate_summary(summary)
    assert "declared in seikan.types/seikan.emitted" in str(refused.value)
    doc = copy.deepcopy(run_doc)
    _patch_pbo(doc["summary"], ledger)
    with pytest.raises(ReportContractError, match=message) as refused:
        validate_emitted("run", doc)
    assert "declared in seikan.types/seikan.emitted" in str(refused.value)


@pytest.mark.parametrize(
    ("verdict", "message"),
    [
        ({"pbo": 0.5}, r"verdict pairing .* violated: pbo 0\.5 beside reason 'single_combo'"),
        ({"reason": None}, "verdict pairing .* violated: pbo None beside reason None"),
    ],
    ids=["a_number_beside_a_reason", "a_reasonless_null"],
)
def test_a_pbo_verdict_out_of_its_pairing_refuses_at_both_seams(run_doc, verdict, message):
    # `pbo` and `reason` are each well-typed on their own (a float or null; a reason or null),
    # so only the pairing can see a block that scored AND declined, or declined without saying
    # why — a refusal is explicit, never a bare null. The fixture's block is a declined
    # `single_combo` with its ledger intact, so each case breaks the pairing alone.
    summary = copy.deepcopy(run_doc["summary"])
    _patch_pbo(summary, verdict)
    with pytest.raises(ReportContractError, match=message):
        validate_summary(summary)
    doc = copy.deepcopy(run_doc)
    _patch_pbo(doc["summary"], verdict)
    with pytest.raises(ReportContractError, match=message):
        validate_emitted("run", doc)


def test_a_scored_pbo_at_equality_on_every_link_validates(run_doc):
    # The ledgers are `<=` chains, not strict: an all-distinct, all-scoreable grid whose every
    # offered split scored over the full roster sits at equality on every link and is legal —
    # the declared rung equal to the burden, as the runner always stamps it, and the block
    # SCORED (a number beside no reason), the pairing's other legal arm.
    doc = copy.deepcopy(run_doc)
    _patch_pbo(
        doc["summary"],
        {
            "pbo": 0.5,
            "reason": None,
            "n_combos": 2,
            "n_combos_scoreable": 2,
            "n_combos_declared": 2,
            "n_hypotheses_attempted": 2,
            "n_splits": 6,
            "n_splits_attempted": 6,
            "n_candidates_min": 2,
        },
    )
    validate_summary(doc["summary"])
    validate_emitted("run", doc)


# ---- the refusal reaches the caller as exit 4, never as a caller-blaming class --------------


def test_contract_violation_exits_4_with_the_internal_envelope(tmp_path, capsys, monkeypatch):
    # Drift an identity stamp off its declared type (`environment` versions are strings): the
    # seam must refuse, and the refusal must ride the catch-all to the exit-4 `internal`
    # envelope — NOT the exit-3 `dsl_invalid` branch, which would blame the caller's thesis for
    # the verifier's own bug. (A raw pydantic ValidationError escaping would do exactly that,
    # which is why ReportContractError is a RuntimeError.)
    px = tmp_path / "px.csv"
    thesis = tmp_path / "t.json"
    report = tmp_path / "report.json"
    _write_ohlcv(px)
    _thesis(thesis)
    monkeypatch.setattr(cli_module, "_environment", lambda: {"python": 3.13})
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
    assert "environment" in envelope["error"]["message"]
    assert not report.exists()  # no out-of-contract artifact lands on disk


def test_an_impossible_pbo_ledger_from_the_engine_exits_4(tmp_path, capsys, monkeypatch):
    # The ledger check is a statement about the ENGINE: a CSCV block whose distinct count
    # outgrew its scoreable count must ride the same catch-all to the exit-4 `internal` envelope
    # as any other contract violation, and leave no report on disk.
    real = runner_module.cscv_pbo

    def drifted(*args, **kwargs):
        out = real(*args, **kwargs)
        return {**out, "n_combos": out["n_combos_scoreable"] + 1}

    px = tmp_path / "px.csv"
    thesis = tmp_path / "t.json"
    report = tmp_path / "report.json"
    _write_ohlcv(px)
    _thesis(thesis)
    monkeypatch.setattr(runner_module, "cscv_pbo", drifted)
    code = main(["run", str(thesis), "--data", f"target={px}", "--report-out", str(report)])
    assert code == 4
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["error"]["type"] == "internal"
    assert "population ledger" in envelope["error"]["message"]
    assert not report.exists()


# ---- the declared shapes stay documented ---------------------------------------------------


def test_report_fields_documents_every_declared_panel_field():
    # seikan.types DECLARES the emitted shapes (machine-checked at emission); contract.py's
    # REPORT_FIELDS DOCUMENTS them (emitted by `seikan schema`); nothing else ties the two, so a
    # field added to a TypedDict but never documented would drift silently. Weak-but-real pin:
    # every declared field name of the load-bearing panels must appear somewhere in the
    # REPORT_FIELDS text (its keys are prose-compound, so set equality is not expressible).
    import json

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
