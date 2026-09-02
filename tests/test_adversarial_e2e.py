"""The adversarial contract, end to end through the CLI.

Threat model: the calling agent controls the thesis DSL and the CSVs, and wants a clean-looking
report for a worthless strategy. Every test here starts from a GOLDEN run whose cell meets the
checklist and applies one manipulation, asserting that the manipulation degrades the run and never
the other way. The invariant under test is deletion monotonicity — removing or corrupting input
can only ever preserve or create an UNMET check.

The exit code is not the assertion. A completed run is exit 0 whatever its cells report, so each
test names the specific check that must flip inside the emitted JSON; the tiers below are about
runs that could not produce a report at all: 2 data invalid · 3 DSL/thresholds invalid · 4
internal error (never expected here — an exit 4 from caller input is a bug in the verifier).

Every attack runs through ``seikan run --report-out``: the report is a FILE and a completed run
prints NOTHING on stdout, so the two channels are read separately — a success is read off the file
(with an empty stdout asserted alongside it, since a report leaking onto stdout would be a contract
regression this suite is well placed to catch), a refusal off the stdout envelope.

The one exception is the stamp-tampering block at the bottom: those attacks mutate a SUMMARY the
engine already emitted, which no CLI invocation can express — the CLI computes its own summary —
so they drive the same library seam the CLI itself uses (``run_backtest`` → ``evaluate_gate``)
and tamper in between.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seikan.cli import REPORT_SCHEMA_VERSION, main
from tests._helpers import unmet_cell_doc as _unmet_cell
from tests._helpers import unmet_run_doc as _unmet_run

from ._helpers import load

# ---- the golden run ----------------------------------------------------------------------


def _planted_dip_px(n_bars: int, gap_lo: int, gap_hi: int, *, seed: int = 0) -> np.ndarray:
    """~100-level path with aperiodic one-bar dips to ~94 and a three-bar climb back — a
    planted, spacing-irregular edge a `close < 95.5` entry captures at h=3."""
    rng = np.random.RandomState(seed)
    px = np.full(n_bars, 100.0)
    pos = 10
    while pos < n_bars - 5:
        px[pos] = 94.0
        px[pos + 1 : pos + 4] = (96.5, 98.0, 99.5)
        pos += int(rng.randint(gap_lo, gap_hi + 1))
    return px * (1.0 + rng.normal(0.0, 0.002, size=n_bars))


N_BARS = 1000
_IDX = pd.date_range("2015-01-02", periods=N_BARS, freq="1D")


def _write_px(path: Path, px: np.ndarray, *, holes: dict[str, list[int]] | None = None) -> Path:
    s = pd.Series(px, index=_IDX)
    df = pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 1000.0}, index=_IDX)
    for column, bars in (holes or {}).items():
        for bar in bars:
            df.iloc[bar, df.columns.get_loc(column)] = np.nan
    df.index.name = "datetime"
    df.to_csv(path)
    return path


def _write_thesis(path: Path, dsl: dict) -> Path:
    path.write_text(json.dumps(dsl), encoding="utf-8")
    return path


def _dsl(**params) -> dict:
    return {
        "name": "adversarial",
        "data": {"targets": ["target"]},
        "entry": {
            "type": "threshold",
            "left": {"type": "field", "column": "close"},
            "op": "<",
            "right": {"type": "constant", "value": 95.5},
        },
        "params": {"horizon": 3, **params},
    }


def _run(
    capsys,
    tmp_path,
    argv,
    data: dict | None = None,
    columns: dict | None = None,
) -> tuple[int, dict]:
    """Run ``seikan run`` with a report file appended and return (exit code, the JSON document).

    The two channels are disjoint, so which one carries the document
    is decided by the exit code: a completed run writes the report to the nominated file and leaves
    stdout EMPTY (asserted here — every test in this suite would otherwise keep passing if the
    report regressed back onto stdout), while a refusal writes only its envelope to stdout and no
    file at all. The report path is always nominated: a run that nominates nothing is a usage error
    before the thesis is read, which would mask every manipulation this suite plants; ``argv`` may
    still carry further output flags, and the silence assertion then covers those files too.

    ``data`` is the invocation's answer to the series the thesis NAMES, spelled onto the command
    line as the ``--data KEY=PATH`` pairs the CLI requires — an attack that swaps a file swaps it
    here, where a caller would. ``columns`` is the other half of the same answer (``--column
    KEY=COL``), optional because only a key whose file holds several numeric columns needs one: an
    attack that re-points a key at a different SERIES inside one file plants it here.
    """
    report = tmp_path / "report.json"
    pairs = [arg for key, path in (data or {}).items() for arg in ("--data", f"{key}={path}")]
    pairs += [arg for key, col in (columns or {}).items() for arg in ("--column", f"{key}={col}")]
    code = main([*argv, *pairs, "--report-out", str(report)])
    out = capsys.readouterr().out
    if code == 0:
        assert out == "", "a completed run prints nothing on stdout"
        return code, json.loads(report.read_text(encoding="utf-8"))
    return code, json.loads(out)


@pytest.fixture
def golden(tmp_path, capsys):
    """A run whose single declared cell clears the whole checklist — the baseline no attack may
    leave standing."""
    px = _write_px(tmp_path / "px.csv", _planted_dip_px(N_BARS, 5, 9))
    thesis = _write_thesis(tmp_path / "t.json", _dsl())
    code, doc = _run(capsys, tmp_path, ["run", str(thesis)], {"target": px})
    assert code == 0
    assert doc["gate"]["n_met"] == 1, (_unmet_run(doc), _unmet_cell(doc))
    return tmp_path


# ---- the attacks ------------------------------------------------------------------------


def test_golden_baseline_meets_its_cell(golden, capsys):
    code, doc = _run(capsys, golden, ["run", str(golden / "t.json")], {"target": golden / "px.csv"})
    assert code == 0
    assert doc["gate"]["policy_version"] == 3
    # The envelope version is the CLI's own constant (its literal pin lives in test_cli.py);
    # this suite pins the CONTRACT stamps — the estimator and checklist revisions — literally.
    assert doc["report_schema_version"] == REPORT_SCHEMA_VERSION
    assert doc["command"] == "run"
    assert doc["summary"]["statistics_version"] == 4
    assert doc["summary"]["target_mode"] == "conjunction"
    assert doc["gate"]["n_cells"] == doc["gate"]["n_met"] == 1
    assert doc["gate"]["cells"][0]["met"] is True
    assert "verdict" not in json.dumps(doc["gate"])


def test_deleting_a_bar_from_one_target_exits_2(golden, capsys):
    """Unequal target clocks refuse rather than intersect: silently intersecting them would let a
    row deleted from one target delete that bar from EVERY target, with no ledger recording it."""
    px = _planted_dip_px(N_BARS, 5, 9)
    a = _write_px(golden / "a.csv", px)
    full = pd.read_csv(a)
    b = golden / "b.csv"
    full.drop(index=range(400, 410)).to_csv(b, index=False)
    thesis = _write_thesis(
        golden / "multi.json",
        {
            **_dsl(),
            "data": {"targets": ["a", "b"]},
        },
    )
    code, doc = _run(capsys, golden, ["run", str(thesis)], {"a": a, "b": b})
    assert code == 2
    assert doc["error"]["type"] == "data_invalid"
    assert "target_index_mismatch" in json.dumps(doc["data_report"])


def test_hole_absorbed_by_a_decisive_sibling_leaves_source_coverage_unmet_at_exit_0(golden, capsys):
    """Kleene `F∧U = F`: the root condition stays DEFINED, so the per-cell decision ledger reads
    clean. Only the run-level per-source panel sees it — and that unmet check leaves the cell
    unmet."""
    px = _planted_dip_px(N_BARS, 5, 9)
    holed = _write_px(golden / "px2.csv", px, holes={"high": [500]})
    dsl = _dsl()
    dsl["entry"] = {
        "type": "and",
        "conditions": [
            dsl["entry"],
            {
                "type": "threshold",
                "left": {"type": "field", "column": "high"},
                "op": ">",
                "right": {"type": "constant", "value": 0.0},
            },
        ],
    }
    code, doc = _run(
        capsys, golden, ["run", str(_write_thesis(golden / "k.json", dsl))], {"target": holed}
    )
    assert code == 0
    assert _unmet_run(doc) == ["source_coverage"]
    assert _unmet_cell(doc) == []  # every per-cell read is clean
    assert doc["gate"]["cells"][0]["met"] is False  # the unmet run-level check leaves it unmet
    assert doc["summary"]["cells"][0]["signal_coverage"]["target"]["n_undefined"] == 0
    assert doc["summary"]["sources"]["target"]["by_source"]["field:high"]["n_missing"] == 1


def test_interior_hole_leaves_source_coverage_unmet_with_exit_0(golden, capsys):
    """A hole is a hole wherever it sits: the evaluated interval is one undivided span, so no
    region belongs to no pool."""
    px = _planted_dip_px(N_BARS, 5, 9)
    holed = _write_px(golden / "px3.csv", px, holes={"close": [700]})
    code, doc = _run(
        capsys, golden, ["run", str(_write_thesis(golden / "e.json", _dsl()))], {"target": holed}
    )
    assert code == 0
    assert _unmet_run(doc) == ["source_coverage"]
    assert doc["gate"]["n_met"] == 0


def test_deleting_the_losing_decision_bars_leaves_signal_coverage_unmet_with_exit_0(golden, capsys):
    """The headline attack: hole the inputs under the worst firings so they never fire. The
    outcome ledger cannot see it (those bars produced no observation to censor) — the decision
    ledger can.

    ``--entry-flags-out`` is nominated alongside the report to pin what the attack looks like in
    the ONE output that reports firings AS SUCH: a suppressed bar is a plain 0, spelled exactly
    like a bar that simply did not fire, and the matrix stays valid 0/1 throughout — no NaN marks
    the deletion. So the firing RECORD reads clean after the inputs under the worst firings are
    deleted, and `signal_coverage` is the only thing standing between that file and a caller who
    believes it."""
    px = _planted_dip_px(N_BARS, 5, 9)
    dips = [int(i) for i in np.flatnonzero(px < 95.5)]
    holed = dips[: len(dips) // 3]
    px_path = _write_px(golden / "px4.csv", px, holes={"close": holed})
    flags = golden / "flags.csv"
    thesis = _write_thesis(golden / "d.json", _dsl())
    code, doc = _run(
        capsys,
        golden,
        ["run", str(thesis), "--entry-flags-out", str(flags)],
        {"target": px_path},
    )
    assert code == 0
    assert "signal_coverage" in _unmet_cell(doc)
    assert doc["gate"]["cells"][0]["met"] is False
    assert doc["summary"]["cells"], "an unmet checklist never filters the evidence"
    # `outputs` enumerates both nominated files in nomination order, report first however the flags
    # were ordered on the command line.
    assert list(doc["outputs"]) == ["report", "entry_flags"]
    assert doc["outputs"]["entry_flags"] == {"path": str(flags), "rows_written": N_BARS}
    fired = pd.read_csv(flags, index_col="datetime")["entry"]
    assert set(fired.unique()) <= {0, 1}  # a deleted decision input is a 0, never a hole
    assert fired.iloc[holed].eq(0).all()
    assert int(fired.sum()) == len(dips) - len(holed)  # the attack is invisible in this file


def test_censoring_the_outcome_leg_leaves_outcome_coverage_unmet_with_exit_0(golden, capsys):
    """The outcome-side twin: hole the MEASURED leg (`open`) so adverse observations censor as
    `no_outcome` and every closed-only statistic silently skips them. The decision inputs stay
    intact, so the refusal is localized to the one check that reads the censoring ledger."""
    px = _planted_dip_px(N_BARS, 5, 9)
    dips = np.flatnonzero(px < 95.5)
    exits = [int(b) for b in dips[::5] + 4 if b < N_BARS]
    holed = _write_px(golden / "px5.csv", px, holes={"open": exits})
    code, doc = _run(
        capsys, golden, ["run", str(_write_thesis(golden / "o.json", _dsl()))], {"target": holed}
    )
    assert code == 0
    assert _unmet_run(doc) == []
    assert _unmet_cell(doc) == ["outcome_coverage"]
    reasons = doc["summary"]["cells"][0]["outcome_coverage"]["target"]["exit_reasons"]
    assert reasons["no_outcome"] > 0


def test_over_cap_grid_exits_3_before_reading_data(golden, capsys):
    """An over-cap declared grid can never be measured under any legal thresholds, so it refuses
    without pricing the sweep. The file the invocation names is deliberately nonexistent."""
    dsl = _dsl()
    dsl["entry"]["right"] = {
        "type": "constant",
        "value": [float(i) for i in range(13)],
        "name": "k",
    }
    dsl["params"]["horizon"] = [1, 2, 3, 4, 5]  # 13 × 5 = 65
    code, doc = _run(
        capsys,
        golden,
        ["run", str(_write_thesis(golden / "g.json", dsl))],
        {"target": Path("/nonexistent/never-read.csv")},
    )
    assert code == 3 and doc["error"]["type"] == "dsl_invalid"


def test_non_finite_json_literal_exits_3(golden, capsys):
    px = _write_px(golden / "px6.csv", _planted_dip_px(N_BARS, 5, 9))
    raw = json.dumps(_dsl()).replace('"value": 95.5', '"value": NaN')
    (golden / "n.json").write_text(raw, encoding="utf-8")
    code, doc = _run(capsys, golden, ["run", str(golden / "n.json")], {"target": px})
    assert code == 3 and doc["error"]["type"] == "dsl_invalid"


def test_reserved_feature_name_exits_3(golden, capsys):
    px = _write_px(golden / "px7.csv", _planted_dip_px(N_BARS, 5, 9))
    dsl = _dsl(features={"ret": {"type": "field", "column": "close"}})
    code, doc = _run(
        capsys, golden, ["run", str(_write_thesis(golden / "f.json", dsl))], {"target": px}
    )
    assert code == 3 and doc["error"]["type"] == "dsl_invalid"


def test_loosened_thresholds_exit_3(golden, capsys):
    """The canonical checklist is the floor: a looser knob is not a result, it is invalid input,
    so a cell reported as met always means at-least-canonical rigor."""
    code, doc = _run(
        capsys,
        golden,
        ["run", str(golden / "t.json"), "--max-concentration", "0.95"],
        {"target": golden / "px.csv"},
    )
    assert code == 3 and doc["error"]["type"] == "thresholds_invalid"


def test_a_thesis_declaring_a_holdout_exits_3(golden, capsys):
    """There is no split to choose: a thesis carrying `oos_fraction` (or `selection_mode`) is
    asking for a mechanism this build does not have, and `extra="forbid"` refuses it rather than
    letting the caller believe a tail was reserved."""
    px = _write_px(golden / "px8.csv", _planted_dip_px(N_BARS, 5, 9))
    dsl = _dsl(oos_fraction=0.25)
    code, doc = _run(
        capsys, golden, ["run", str(_write_thesis(golden / "h.json", dsl))], {"target": px}
    )
    assert code == 3 and doc["error"]["type"] == "dsl_invalid"
    assert "oos_fraction" in json.dumps(doc["error"]["errors"])


def test_the_complete_report_is_emitted_however_many_cells_are_unmet(golden, capsys):
    """The honesty invariant across the whole attack surface: the report carries every declared
    cell, with every check reported for each, no matter how the checklist came out. There is no
    refusal exit to hide behind — reporting and grading are strictly separable."""
    px = _write_px(golden / "px9.csv", _planted_dip_px(N_BARS, 5, 9))
    dsl = _dsl(horizon=[3, 5, 7])
    thesis = _write_thesis(golden / "r.json", dsl)
    code, doc = _run(
        capsys, golden, ["run", str(thesis), "--min-n-nonoverlap", "10000"], {"target": px}
    )
    assert code == 0
    assert doc["gate"]["n_cells"] == 3 and doc["gate"]["n_met"] == 0
    assert len(doc["gate"]["run_checks"]) == 3
    for i, cell in enumerate(doc["gate"]["cells"]):
        assert len(cell["checks"]) == 5  # no short-circuit
        assert _unmet_cell(doc, i) == ["support"]
    assert len(doc["summary"]["cells"]) == 3
    assert all(c["by_target"] for c in doc["summary"]["cells"])
    assert doc["identity"]["dsl_hash"] and doc["identity"]["data_digests"]


# ---- basket attacks ---------------------------------------------------------------------
#
# The planted TWO-MEMBER basket: member A's events rebound ≈ +3.6%, member B's fade ≈ −3.0%, in
# comparable |return| masses — so the pooled panel meets the basket rubric (small positive
# pooled mean, member shares ≈ 0.54/0.46) while member B alone would sit below a per-member floor.
# The entry reads a per-member decision FEED (1.0 everywhere, 0.0 on event bars), which is what
# makes the feed the decision leaf an attacker holes. Events are SPARSE (gaps 6–10 > h) on
# purpose: a basket entry firing nearly every bar chains ALL observations into ONE merged
# episode cluster and concentration refuses at max_cluster_share_abs=1.0 — correct pre-existing
# doctrine, but it would make every fixture here fail for the wrong reason.

_POOLED_KEYS = {
    "n",
    "n_nonoverlap",
    "mean_ret",
    "mean_ret_raw",
    "mean_ret_bench",
    "benchmark_regression",
    "hit_rate",
    "win_loss_ratio",
    "profit_factor",
    "std_ret",
    "skewness",
    "kurtosis",
    "tail_ratio",
    "cvar_5",
    "t_hac",
    "hac_se",
    "rot_p",
    "rot_n_null",
    "concentration",
    "member_share",
    "boot",
    "subperiods",
    "ret_quantiles",
    "worst_ret",
    "best_ret",
    "mae_quantiles",
    "mfe_quantiles",
    "edge_ratio",
    "timing",
}


def _event_bars(gap_lo: int, gap_hi: int, *, start: int, seed: int) -> list[int]:
    rng = np.random.RandomState(seed)
    bars, pos = [], start
    while pos < N_BARS - 8:
        bars.append(pos)
        pos += int(rng.randint(gap_lo, gap_hi + 1))
    return bars


def _write_member(
    px_path: Path, sig_path: Path, events, steps, *, seed: int, sig_holes=None
) -> None:
    """One basket member: a ~100-level price with a planted 3-bar response after each event bar
    (fill open[t+1] → exit open[t+4] at h=3), and its decision feed — holed where asked."""
    rng = np.random.RandomState(seed)
    px = np.full(N_BARS, 100.0)
    for t in events:
        px[t + 1 : t + 5] = 100.0 * np.asarray(steps)
    _write_px(px_path, px * (1.0 + rng.normal(0.0, 0.002, size=N_BARS)))
    sig = np.ones(N_BARS)
    sig[events] = 0.0
    if sig_holes is not None:
        sig[list(sig_holes)] = np.nan
    feed = pd.DataFrame({"sig": pd.Series(sig, index=_IDX)})
    feed.index.name = "datetime"
    feed.to_csv(sig_path)


def _basket_dsl_dict(
    tmp_path: Path, *, b_sig_holes=None, b_steps=(1.000, 0.990, 0.980, 0.970)
) -> dict:
    _write_member(
        tmp_path / "bkA.csv",
        tmp_path / "bk_sigA.csv",
        _event_bars(6, 10, start=12, seed=0),
        (1.000, 1.012, 1.024, 1.036),
        seed=2,
    )
    _write_member(
        tmp_path / "bkB.csv",
        tmp_path / "bk_sigB.csv",
        _event_bars(6, 10, start=15, seed=1),
        b_steps,
        seed=3,
        sig_holes=b_sig_holes,
    )
    return {
        "name": "planted_basket",
        "target_mode": "basket",
        "data": {
            "targets": ["A", "B"],
            "external": {"sig": {"per_target": True}},
        },
        "entry": {
            "type": "threshold",
            "left": {"type": "external", "name": "sig"},
            "op": "<",
            "right": {"type": "constant", "value": 0.5},
        },
        "params": {"horizon": 3},
    }


def _basket_files(tmp_path: Path) -> dict[str, Path]:
    """The invocation's answer for the planted basket: each member, and each member's own decision
    feed under the derived ``sig@<target>`` key a per-target feed declares."""
    return {
        "A": tmp_path / "bkA.csv",
        "B": tmp_path / "bkB.csv",
        "sig@A": tmp_path / "bk_sigA.csv",
        "sig@B": tmp_path / "bk_sigB.csv",
    }


def test_basket_member_hole_leaves_source_coverage_unmet_naming_the_member(tmp_path, capsys):
    """The decision-input attack, basket edition: hole ONE member's decision feed over an
    interior stretch. The pooled panel keeps looking healthy (the surviving rows are untouched),
    so the refusal must come from the coverage layers — and it must NAME the holed member, since
    per-target fail-closed coverage is exactly what basket mode retains per member. The report
    stays complete: every pooled block is emitted, because refusing is never filtering."""
    # control: the same basket with clean feeds clears the whole checklist
    thesis = _write_thesis(tmp_path / "bk.json", _basket_dsl_dict(tmp_path))
    code, doc = _run(capsys, tmp_path, ["run", str(thesis)], _basket_files(tmp_path))
    assert code == 0
    assert doc["summary"]["target_mode"] == "basket"
    assert doc["gate"]["n_met"] == 1, (_unmet_run(doc), _unmet_cell(doc))

    # the attack: a 20-bar interior outage in member B's decision feed
    holed = _write_thesis(
        tmp_path / "bk_holed.json", _basket_dsl_dict(tmp_path, b_sig_holes=range(300, 320))
    )
    code, doc = _run(capsys, tmp_path, ["run", str(holed)], _basket_files(tmp_path))
    assert code == 0
    assert _unmet_run(doc) == ["source_coverage"]
    src = next(c for c in doc["gate"]["run_checks"] if c["name"] == "source_coverage")
    assert "B: decision input external:sig" in src["detail"]  # the member is named
    # the hole sits under B's OWN threshold, so B's three-valued ledger sees it too — while A,
    # whose feed is whole, stays clean (no cross node couples the members' decisions here)
    assert "signal_coverage" in _unmet_cell(doc)
    sig = doc["summary"]["cells"][0]["signal_coverage"]
    assert sig["B"]["n_undefined"] == 20 and sig["A"]["n_undefined"] == 0
    assert doc["gate"]["cells"][0]["met"] is False
    # the complete report is still emitted, pooled blocks included — nothing was withheld
    cell = doc["summary"]["cells"][0]
    assert set(cell["pooled"]) == _POOLED_KEYS
    assert cell["pooled"]["n"] > 0 and cell["by_target"]
    assert doc["summary"]["baseline"][0]["pooled"]["n_eligible"] > 0


# ---- stamp tampering (library seam: run_backtest → tamper → evaluate_gate) --------------


def _basket_summary(tmp_path, **dsl_kwargs) -> dict:
    from seikan.compiler.runner import run_backtest
    from seikan.dsl.schema import Thesis

    thesis = Thesis.model_validate(_basket_dsl_dict(tmp_path, **dsl_kwargs))
    return run_backtest(thesis, load(thesis, _basket_files(tmp_path))).summary


def _conjunction_summary(tmp_path) -> dict:
    """A TWO-target conjunction golden (both members carry the planted dip edge), so the
    basket-rewrite tamper below isolates the missing-pooled-panel refusal instead of tripping
    the <2-targets refusal first."""
    from seikan.compiler.runner import run_backtest
    from seikan.dsl.schema import Thesis

    _write_px(tmp_path / "cjA.csv", _planted_dip_px(N_BARS, 5, 9, seed=0))
    _write_px(tmp_path / "cjB.csv", _planted_dip_px(N_BARS, 5, 9, seed=1))
    thesis = Thesis.model_validate(
        {
            "name": "conjunction_pair",
            "data": {"targets": ["A", "B"]},
            "entry": {
                "type": "threshold",
                "left": {"type": "field", "column": "close"},
                "op": "<",
                "right": {"type": "constant", "value": 95.5},
            },
            "params": {"horizon": 3},
        }
    )
    files = {"A": tmp_path / "cjA.csv", "B": tmp_path / "cjB.csv"}
    return run_backtest(thesis, load(thesis, files)).summary


def _grade(summary):
    from seikan.gate import GateThresholds, evaluate_gate

    return evaluate_gate(summary, GateThresholds())


def _cell_fails(report, index: int = 0) -> list[str]:
    return [c.name for c in report.cells[index].checks if not c.met]


def test_deleting_the_target_mode_stamp_refuses_everywhere_it_dispatches(tmp_path):
    """The stamp selects the rubric, so stripping it is the bypass every dispatch must refuse:
    grading under an ASSUMED mode would let a caller pick whichever rubric flatters the summary.
    ``evidence_complete`` refuses at the run level, and every mode-dispatching cell check
    (cell_evidence, support, concentration) refuses independently — no check trusts another to
    have caught it."""
    summary = _basket_summary(tmp_path)
    assert _grade(summary).cells[0].met is True  # the honest run is met

    tampered = copy.deepcopy(summary)
    del tampered["target_mode"]
    report = _grade(tampered)
    ec = next(c for c in report.run_checks if c.name == "evidence_complete")
    assert not ec.met and "target_mode" in ec.detail
    assert set(_cell_fails(report)) == {"cell_evidence", "support", "concentration"}
    assert report.cells[0].met is False


def test_rewriting_the_stamp_can_only_preserve_or_create_refusals(tmp_path):
    """Rewriting the stamp swaps the RUBRIC out from under the evidence, and both directions
    must degrade, never flatter:

    - basket → "conjunction": the ``pooled`` panel the runner writes ONLY in basket mode is
      still on every cell, and under a conjunction stamp that key is the WITNESS of the tamper —
      cell_evidence refuses it as a restamped basket on every cell, before any number is read.
      The per-member floors the honest basket rubric never applies then grade each member alone
      too, and the planted fading member refuses support beside it — a caller cannot relabel a
      basket to hide behind the healthier rubric, because there isn't one.
    - conjunction → "basket": the summary never formed a pooled panel, so cell_evidence refuses
      on the missing panel and both pooled-reading checks refuse beside it. A conjunction run
      cannot be re-stamped into the pooled rubric it carries no evidence for."""
    basket = _basket_summary(tmp_path)
    assert _grade(basket).cells[0].met is True

    as_conjunction = copy.deepcopy(basket)
    as_conjunction["target_mode"] = "conjunction"
    report = _grade(as_conjunction)
    assert set(_cell_fails(report)) == {"cell_evidence", "support"}
    ce = next(c for c in report.cells[0].checks if c.name == "cell_evidence")
    assert "restamped basket" in ce.detail  # the pooled key the tamper could not shed
    support = next(c for c in report.cells[0].checks if c.name == "support")
    assert "B: mean_ret=" in support.detail  # the fading member the honest rubric never floors
    assert report.cells[0].met is False

    conjunction = _conjunction_summary(tmp_path)
    assert _grade(conjunction).cells[0].met is True

    as_basket = copy.deepcopy(conjunction)
    as_basket["target_mode"] = "basket"
    report = _grade(as_basket)
    assert [c.name for c in report.run_checks if not c.met] == []  # 2 targets, pct: readable
    assert set(_cell_fails(report)) == {"cell_evidence", "support", "concentration"}
    ce = next(c for c in report.cells[0].checks if c.name == "cell_evidence")
    assert "pooled panel" in ce.detail  # the refusal names the evidence the rubric requires
    assert report.cells[0].met is False


def test_restamping_cannot_launder_a_one_name_basket_through_the_per_member_rubric(tmp_path):
    """The pin on the case the fading-member restamp above does not cover: a basket whose every
    member is INDIVIDUALLY healthy (clears the n / n_nonoverlap / mean floors, spreads its own
    return
    mass) but whose pooled mass rides mostly one name, so the ONLY refusal is the basket-only
    ``member_share`` ceiling. Were a stray ``pooled`` ignored, flipping the one stamp field to
    "conjunction" would drop the one detector that fires and every per-member read would pass —
    a fully CLEAN report for the summary the honest rubric refused. The pooled key the tamper
    cannot shed is the witness instead: cell_evidence refuses it as a restamped basket, so the
    flip trades one refusal for another and the hole stays shut."""
    # Member B's events drift +1.0% against A's +3.6% — both positive, both dense, so every
    # member clears every per-member read while A carries ~78% of the pooled |return| mass.
    summary = _basket_summary(tmp_path, b_steps=(1.000, 1.003, 1.006, 1.010))
    cell = summary["cells"][0]
    for tgt in ("A", "B"):  # the premise: individually healthy members, asserted not assumed
        st = cell["by_target"][tgt]
        assert st["n"] >= 30 and st["n_nonoverlap"] >= 8 and st["mean_ret"] > 0
        assert st["concentration"]["top_share_abs"] <= 0.6
    assert cell["episode_stats"]["max_cluster_share_abs"] <= 0.6
    assert cell["pooled"]["member_share"]["max_member_share_abs"] > 0.6  # the one breach

    report = _grade(summary)
    assert _cell_fails(report) == ["concentration"]  # refused ONLY by the basket-only detector
    conc = next(c for c in report.cells[0].checks if c.name == "concentration")
    assert "one member carries the basket's mass" in conc.detail
    assert report.cells[0].met is False

    laundered = copy.deepcopy(summary)
    laundered["target_mode"] = "conjunction"
    report = _grade(laundered)
    assert report.cells[0].met is False  # STILL refused — the tamper bought nothing
    assert _cell_fails(report) == ["cell_evidence"]  # …now via the restamp witness
    ce = next(c for c in report.cells[0].checks if c.name == "cell_evidence")
    assert "restamped basket" in ce.detail
