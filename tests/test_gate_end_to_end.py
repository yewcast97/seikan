"""Closed-loop reporting: synthetic data → ``run_backtest`` → ``evaluate_gate``.

The gate unit tests grade hand-built summaries; these tests certify the REAL contract — the
summary the engine actually emits is graded (cell by cell) by the real checklist
end-to-end, with no hand-shaped evidence in between. The planted series are aperiodic (random
event spacing) so the circular-shift rotation null is not degenerate, and the planted edge is
strong and consistent so the canonical thresholds are clearable by construction.

There is no verdict to assert here. Every test names the CELL (or the run-level check) whose
result it is about, and the sabotage tests additionally pin that the other checks hold — a
manipulation that flips an unrelated check is a manipulation nobody has actually localized.

The ``target_mode`` stamp dispatches the rubric: conjunction runs grade per-member floors, basket
runs grade the per-cell POOLED panel instead. The basket section at
the bottom drives a planted two-member basket through the same closed loop, and the stripping
tests pin the evidence/gate seam — evidence-only blocks may vanish without moving a single check
result, gated pooled fields may not.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

from seikan.compiler.runner import run_backtest
from seikan.dsl.schema import Thesis
from seikan.gate import GateThresholds, evaluate_gate
from seikan.serialize import json_safe
from tests._helpers import keys_deep as _keys_deep
from tests._helpers import unmet_cell_report as _unmet_cell
from tests._helpers import unmet_run_report as _unmet_run

from ._helpers import load

# ---- planted series ---------------------------------------------------------------------------


def _planted_dip_px(n_bars: int, gap_lo: int, gap_hi: int, *, seed: int = 0) -> np.ndarray:
    """A ~100-level price path with aperiodic one-bar dips to ~94; the three bars after each dip
    climb back (96.5 / 98 / 99.5). A ``close < 95.5`` entry fires exactly on the dip bars, and
    an h=3 forward return from the next open is ≈ +3.6% — a planted, spacing-irregular edge."""
    rng = np.random.RandomState(seed)
    px = np.full(n_bars, 100.0)
    pos = 10
    while pos < n_bars - 5:
        px[pos] = 94.0
        px[pos + 1 : pos + 4] = (96.5, 98.0, 99.5)
        pos += int(rng.randint(gap_lo, gap_hi + 1))
    return px * (1.0 + rng.normal(0.0, 0.002, size=n_bars))


def _dip_then_fade_px(
    n_bars: int, gap_lo: int = 18, gap_hi: int = 26, *, seed: int = 0
) -> np.ndarray:
    """The same dip and three-bar rebound, followed by a nine-bar fade back down to ~96.

    One firing mask, two OPPOSITE truths depending on the measurement window: an h=3 observation
    (fill 96.5 → exit 100) captures the rebound at ≈ +3.6%, while an h=12 observation (fill 96.5 →
    exit 96.0) runs through the fade at ≈ −0.5%. The fade floor stays above the 95.5 trigger, so
    both cells read the SAME entries — the only thing that differs between them is the horizon.
    """
    rng = np.random.RandomState(seed)
    px = np.full(n_bars, 100.0)
    pos = 20
    while pos < n_bars - 20:
        px[pos] = 94.0
        px[pos + 1 : pos + 4] = (96.5, 98.0, 99.5)
        px[pos + 4] = 100.0
        px[pos + 5 : pos + 14] = 96.0
        pos += int(rng.randint(gap_lo, gap_hi + 1))
    return px * (1.0 + rng.normal(0.0, 0.002, size=n_bars))


def _write_px(path, px: np.ndarray) -> None:
    idx = pd.date_range("2015-01-02", periods=px.shape[0], freq="1D")
    s = pd.Series(px, index=idx)
    df = pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 1000.0}, index=idx)
    df.index.name = "datetime"
    df.to_csv(path)


def _planted_dip_series(path, n_bars: int, gap_lo: int, gap_hi: int, *, seed: int = 0):
    _write_px(path, _planted_dip_px(n_bars, gap_lo, gap_hi, seed=seed))


DIP_ENTRY = {
    "type": "threshold",
    "left": {"type": "field", "column": "close"},
    "op": "<",
    "right": {"type": "constant", "value": 95.5},
}


def _run_entry(path, entry, horizon=3, **params) -> dict:
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": entry,
            "params": {"horizon": horizon, **params},
        }
    )
    return run_backtest(thesis, load(thesis, {"target": path})).summary


def _run(path, direction: str = "longonly", horizon=3) -> dict:
    return _run_entry(path, DIP_ENTRY, horizon=horizon, direction=direction)


# ---- readers ----------------------------------------------------------------------------------


def _check(cell, name):
    return next(c for c in cell.checks if c.name == name)


def _run_check(report, name):
    return next(c for c in report.run_checks if c.name == name)


def _by_horizon(summary, report) -> dict[int, object]:
    """Graded cells keyed by horizon — the gate's cells are index-aligned with the summary's."""
    return {
        s_cell["params"]["horizon"]: g_cell
        for s_cell, g_cell in zip(summary["cells"], report.cells, strict=True)
    }


# ---- the golden: one run, cells graded independently -------------------------------------------


def test_golden_run_meets_the_edge_cell_and_not_the_noise_cell(tmp_path):
    # ONE firing mask, two horizons, two opposite truths. The engine grades each cell alone: the
    # h=3 cell measures the planted rebound and clears the whole checklist, the h=12 cell measures
    # the fade that follows and leaves support unmet on its negative mean. Neither result
    # influences the
    # other, and nothing here selects the winner — that is the caller's job.
    path = tmp_path / "px.csv"
    _write_px(path, _dip_then_fade_px(3000))
    summary = _run_entry(path, DIP_ENTRY, horizon=[3, 12])
    report = evaluate_gate(summary, GateThresholds())

    assert _unmet_run(report) == []
    assert summary["n_hypotheses_attempted"] == 2
    assert len(summary["cells"]) == len(report.cells) == 2
    cells = _by_horizon(summary, report)

    edge = cells[3]
    assert edge.met is True, f"the planted edge cell is unmet on: {_unmet_cell(edge)}"

    noise = cells[12]
    assert noise.met is False
    assert _unmet_cell(noise) == ["support"]
    assert "mean_ret=" in _check(noise, "support").detail  # the refusal names the sign it read

    doc = report.to_dict()
    assert doc["n_cells"] == 2 and doc["n_met"] == 1
    # The whole evidence contract is on record end-to-end, exactly.
    assert summary["statistics_version"] == 4
    assert summary["gate_evidence_basis"] == "full_sample"
    assert summary["target_mode"] == "conjunction"  # the stamp the checklist dispatched on
    assert summary["n_bars"] == 3000
    assert summary["index_start"] and summary["index_end"]
    assert set(summary["sources"]) == set(summary["targets"])
    # the evidence blocks ride the graded report: the run-level base rate per declared
    # horizon, and the per-cell narrative panels — while `pooled` stays basket-only
    assert [e["horizon"] for e in summary["baseline"]] == [3, 12]
    for cell in summary["cells"]:
        assert {
            "episodes",
            "conditional_buckets",
            "bucket_monotonicity",
            "feature_association",
        } <= cell.keys()
        assert "pooled" not in cell  # a conjunction run forms no pooled read


def test_the_gate_report_carries_no_verdict(tmp_path):
    # There is no scalar verdict under any spelling: a grid has no single result, and publishing
    # one would invite reading the best cell as the run's answer. The retired pass/fail keys must
    # not resurface either — `met` states a neutral threshold comparison held, nothing more.
    path = tmp_path / "px.csv"
    _planted_dip_series(path, 1000, 5, 9)
    doc = evaluate_gate(_run(path), GateThresholds()).to_dict()
    assert set(doc) == {"policy_version", "n_cells", "n_met", "run_checks", "cells"}
    assert doc["policy_version"] == 3
    assert "verdict" not in json.dumps(doc)
    assert not {"passed", "n_passed"} & set(_keys_deep(doc))
    assert [c["name"] for c in doc["run_checks"]] == [
        "evidence_complete",
        "source_coverage",
        "search_cap",
    ]
    assert [c["name"] for c in doc["cells"][0]["checks"]] == [
        "cell_evidence",
        "outcome_coverage",
        "signal_coverage",
        "support",
        "concentration",
    ]


def test_every_declared_cell_is_on_the_record_including_non_firing_ones(tmp_path):
    # The honesty invariant: a declared hypothesis that never fired is exactly the one a reader
    # most needs to see, because its absence is what makes a surviving cell look inevitable. It
    # appears with an explicit zero record and goes unmet on scarcity — never by omission.
    path = tmp_path / "px.csv"
    _write_px(path, _dip_then_fade_px(3000))
    entry = {
        "type": "threshold",
        "left": {"type": "field", "column": "close"},
        "op": "<",
        "right": {"type": "constant", "value": [95.5, 90.0], "name": "trigger"},
    }
    summary = _run_entry(path, entry)
    report = evaluate_gate(summary, GateThresholds())
    assert len(summary["cells"]) == summary["n_hypotheses_attempted"] == 2

    silent = next(c for c in summary["cells"] if c["params"]["trigger"] == 90.0)
    assert silent["by_target"]["target"]["n"] == 0
    assert silent["outcome_coverage"]["target"]["n_attempted"] == 0
    graded = report.cells[summary["cells"].index(silent)]
    # Its ledgers still RECONCILE (0 == 0) — the cell is complete evidence of nothing happening,
    # which is a different claim from evidence being absent.
    assert _check(graded, "cell_evidence").met is True
    assert graded.met is False
    assert set(_unmet_cell(graded)) == {"support", "concentration"}
    assert "n=0" in _check(graded, "support").detail
    assert report.to_dict()["n_met"] == 1


def test_an_unmet_run_level_check_leaves_every_cell_unmet(tmp_path):
    # Run-level evidence is the ground every per-cell read stands on, so one unmet run-level
    # check leaves EVERY cell unmet — a caller reading cells[i].met gets the complete answer
    # without ANDing sections itself. Here every per-cell check holds and the cells are still
    # unmet.
    path = tmp_path / "px.csv"
    _planted_dip_series(path, 1000, 5, 9)
    summary = _run_entry(path, DIP_ENTRY, horizon=[3, 5])
    report = evaluate_gate(summary, GateThresholds(thesis_max_hypotheses=1))
    assert _unmet_run(report) == ["search_cap"]
    assert len(report.cells) == 2
    for cell in report.cells:
        assert _unmet_cell(cell) == []  # every per-cell check holds
        assert cell.met is False  # and the cell is still unmet
    assert report.to_dict()["n_met"] == 0


def test_rare_firings_leave_support_unmet_on_scarcity(tmp_path):
    # ~20 rare episodes. Low power is evidence scarcity, not a reason for a weaker exam — there
    # is no alternate checklist to fall into, so the cell simply falls short of its observation
    # floor.
    path = tmp_path / "px.csv"
    _planted_dip_series(path, 1000, 40, 60)
    summary = _run(path)
    report = evaluate_gate(summary, GateThresholds())
    cell = report.cells[0]
    assert _unmet_cell(cell) == ["support"]
    assert "n=" in _check(cell, "support").detail  # names the scarce count
    assert summary["cells"]  # complete grid still on record


def test_wrong_side_of_the_edge_leaves_support_unmet(tmp_path):
    # SHORT the planted rebound: identical firings, sign-flipped measured edge. `mean_ret > 0` is
    # a sign read on the realized sample — deliberately not a test — and it is what refuses here.
    path = tmp_path / "px.csv"
    _planted_dip_series(path, 1000, 5, 9)
    summary = _run(path, direction="shortonly")
    report = evaluate_gate(summary, GateThresholds())
    assert _unmet_cell(report.cells[0]) == ["support"]
    assert summary["cells"][0]["by_target"]["target"]["mean_ret"] < 0
    assert summary["cells"]  # complete grid still on record


def test_edge_cell_survives_stricter_thresholds(tmp_path):
    # Stricter-than-canonical overrides are the only overrides that construct; the planted edge
    # is far from the boundary and clears a moderately stricter checklist too.
    path = tmp_path / "px.csv"
    _planted_dip_series(path, 1000, 5, 9)
    summary = _run(path)
    strict = GateThresholds(thesis_min_n_nonoverlap=12, thesis_max_concentration=0.3)
    report = evaluate_gate(summary, strict)
    cell = report.cells[0]
    assert cell.met is True, f"the stricter checklist failed on: {_unmet_cell(cell)}"
    assert strict.is_canonical() is False  # the CLI stamps this into identity, not the gate


# ---- censoring: what is structural vs what is a hole -------------------------------------------


def test_end_of_data_right_censoring_is_allowed(tmp_path):
    # With no holdout there is no embargo and no tail, so a forward window running past the last
    # bar is the calendar, not a data defect. The cell carries an `open` firing and is still met.
    px = _dip_then_fade_px(3000)
    last_dip = int(np.flatnonzero(px < 95.5)[-1])
    path = tmp_path / "px.csv"
    _write_px(path, px[: last_dip + 3])  # the final firing's h=3 window runs off the end
    summary = _run(path)
    ledger = summary["cells"][0]["outcome_coverage"]["target"]
    assert ledger["exit_reasons"]["open"] == 1
    assert ledger["exit_reasons"]["no_outcome"] == 0
    report = evaluate_gate(summary, GateThresholds())
    cell = report.cells[0]
    assert cell.met is True, f"structural censoring refused on: {_unmet_cell(cell)}"
    assert "ALLOWED" in _check(cell, "outcome_coverage").threshold


def test_censored_outcomes_leave_outcome_coverage_unmet(tmp_path):
    # The missingness attack, end-to-end: NaN holes at outcome endpoints censor observations
    # (`no_outcome`) which every closed-only statistic silently skips, so a file that deleted
    # adverse outcomes could otherwise leave a clean-looking cell. The holes sit in the MEASURED
    # leg (`open`) only, so the decision inputs stay clean and the refusal is localized.
    n_bars = 1000
    px = _planted_dip_px(n_bars, 5, 9)
    dips = np.flatnonzero(px < 95.5)
    opens = px.copy()
    opens[dips[::5] + 4] = np.nan  # each firing's h=3 exit anchor (fill = dip+1, exit = dip+4)
    idx = pd.date_range("2015-01-02", periods=n_bars, freq="1D")
    s = pd.Series(px, index=idx)
    df = pd.DataFrame(
        {"open": pd.Series(opens, index=idx), "high": s, "low": s, "close": s, "volume": 1000.0},
        index=idx,
    )
    df.index.name = "datetime"
    path = tmp_path / "px.csv"
    df.to_csv(path)

    summary = _run(path)
    assert summary["cells"][0]["outcome_coverage"]["target"]["exit_reasons"]["no_outcome"] > 0
    report = evaluate_gate(summary, GateThresholds())
    assert _unmet_run(report) == []  # the decision side is untouched
    cell = report.cells[0]
    assert _unmet_cell(cell) == ["outcome_coverage"]
    assert "no_outcome" in _check(cell, "outcome_coverage").detail
    assert summary["cells"]  # the refusal never filters the evidence


def test_benchmark_hole_leaves_outcome_coverage_unmet(tmp_path):
    # A benchmark hole censors the observation as `no_benchmark` — distinguishable from a target
    # hole and, like it, never confusable with terminal `open` censoring the checklist allows.
    n_bars = 1000
    px = _planted_dip_px(n_bars, 5, 9)
    idx = pd.date_range("2015-01-02", periods=n_bars, freq="1D")
    path = tmp_path / "px.csv"
    _write_px(path, px)
    bench_open = np.full(n_bars, 50.0)
    bench_open[np.flatnonzero(px < 95.5)[::5] + 4] = np.nan
    bench = pd.DataFrame(
        {
            "open": pd.Series(bench_open, index=idx),
            "high": 50.0,
            "low": 50.0,
            "close": 50.0,
            "volume": 1.0,
        },
        index=idx,
    )
    bench.index.name = "datetime"
    bench.to_csv(tmp_path / "bench.csv")
    thesis = Thesis.model_validate(
        {
            "name": "benchmarked_dip",
            "data": {"targets": ["target"]},
            "entry": DIP_ENTRY,
            "params": {"horizon": 3, "benchmark": "market"},
        }
    )
    # `benchmark: "market"` adds the reserved `benchmark` key to the thesis's declared series
    files = {"target": path, "benchmark": tmp_path / "bench.csv"}
    summary = run_backtest(thesis, load(thesis, files)).summary
    reasons = summary["cells"][0]["outcome_coverage"]["target"]["exit_reasons"]
    assert reasons["no_benchmark"] > 0 and reasons["open"] == 0
    report = evaluate_gate(summary, GateThresholds())
    cell = report.cells[0]
    assert _unmet_cell(cell) == ["outcome_coverage"]
    assert "no_benchmark" in _check(cell, "outcome_coverage").detail


# ---- the decision side: pooled ledger + raw sources ---------------------------------------------


def _dip_thesis_on_feed():
    """The golden dip thesis, but the entry reads an EXTERNAL feed — so holing the feed is a
    decision-input attack (it suppresses firings) rather than an outcome attack."""
    return Thesis.model_validate(
        {
            "name": "planted_dip_rebound_feed",
            "data": {"targets": ["target"], "external": {"px": {}}},
            "entry": {
                "type": "threshold",
                "left": {"type": "external", "name": "px"},
                "op": "<",
                "right": {"type": "constant", "value": 95.5},
            },
            "params": {"horizon": 3},
        }
    )


def _write_feed(path, values, n_bars):
    idx = pd.date_range("2015-01-02", periods=n_bars, freq="1D")
    df = pd.DataFrame({"px": pd.Series(values, index=idx, dtype=float)})
    df.index.name = "datetime"
    df.to_csv(path)


def test_deleting_losing_decision_inputs_cannot_improve_a_cell(tmp_path):
    # THE decision-input attack, end-to-end. A NaN in an entry operand does not censor an
    # outcome — it suppresses the FIRING, so without a ledger recording it the adverse rows would
    # simply cease to exist: an attacker (or a vendor outage) could delete the losers and keep the
    # winners. The enforced asymmetry is that deleting data can only ever REFUSE.
    n_bars = 1000
    px = _planted_dip_px(n_bars, 5, 9)
    path, feed = tmp_path / "px.csv", tmp_path / "feed.csv"
    _write_px(path, px)
    _write_feed(feed, px, n_bars)

    thesis, files = _dip_thesis_on_feed(), {"target": path, "px": feed}
    control = run_backtest(thesis, load(thesis, files))
    assert evaluate_gate(control.summary, GateThresholds()).cells[0].met is True

    # Now hole the decision feed at the firing bars whose forward return was WORST — exactly
    # the rows an adversary would want gone.
    losers = control.trades.nsmallest(len(control.trades) // 3, "ret")["entry_bar"].to_numpy()
    attacked = px.copy()
    attacked[losers] = np.nan
    _write_feed(feed, attacked, n_bars)
    summary = run_backtest(thesis, load(thesis, files)).summary

    report = evaluate_gate(summary, GateThresholds())
    cell = report.cells[0]
    assert cell.met is False, "deleting losing decision inputs must never pass"
    # The two decision-side layers both see it: the pooled three-valued ledger (the operand is
    # the root condition here, so the hole leaves it undecidable) and the raw source panel.
    assert _unmet_cell(cell) == ["signal_coverage"]
    assert "undecidable" in _check(cell, "signal_coverage").detail
    assert _unmet_run(report) == ["source_coverage"]
    # With the losers gone the pool looks HEALTHY to every other read — support and
    # concentration both pass — which is exactly why the decision-side layers have to catch it.
    assert _check(cell, "support").met and _check(cell, "concentration").met
    assert summary["cells"]  # the refusal never filters the evidence


def test_not_over_a_decision_hole_neither_fires_nor_meets(tmp_path):
    # The phantom fire: NaN → comparison False → `not` negates it to True → a FIRING manufactured
    # from missing data. It must neither fire nor be silently tolerated.
    n_bars = 1000
    px = _planted_dip_px(n_bars, 5, 9)
    holed = px.copy()
    holed[np.arange(300, 340)] = np.nan  # a 40-bar feed outage
    path, feed = tmp_path / "px.csv", tmp_path / "feed.csv"
    _write_px(path, px)
    _write_feed(feed, holed, n_bars)
    thesis = Thesis.model_validate(
        {
            "name": "not_over_hole",
            "data": {"targets": ["target"], "external": {"px": {}}},
            "entry": {
                "type": "not",
                "condition": {
                    "type": "threshold",
                    "left": {"type": "external", "name": "px"},
                    "op": ">",
                    "right": {"type": "constant", "value": 95.5},
                },
            },
            "params": {"horizon": 3},
        }
    )
    result = run_backtest(thesis, load(thesis, {"target": path, "px": feed}))
    fired = set(result.trades["entry_bar"])
    assert not (fired & set(range(300, 340))), "a hole must never manufacture a firing"
    report = evaluate_gate(result.summary, GateThresholds())
    assert _check(report.cells[0], "signal_coverage").met is False
    assert report.cells[0].met is False


def test_a_never_available_or_leaf_refuses_source_coverage_end_to_end(tmp_path):
    # The v3 branch's reason for existing, run end-to-end: `dip OR (iv > 0)` where the iv feed's
    # stamps all land AFTER the run window — post-asof-ffill it is NaN on every bar. Kleene
    # absorption (T∨U = T) lets the dip sibling fire the root, the never-warming branch stays
    # vacuously defined (signal_coverage clean), and support happily grades the sibling's
    # firings — the pre-v3 "the run cannot fire on it anyway, so support refuses" rationale,
    # defeated. Only the source panel records the leaf, so only source_coverage can refuse it.
    n_bars = 1000
    path, feed = tmp_path / "px.csv", tmp_path / "iv.csv"
    _planted_dip_series(path, n_bars, 5, 9)
    late_idx = pd.date_range("2020-01-01", periods=10, freq="1D")  # entirely after the target
    df = pd.DataFrame({"iv": pd.Series(np.ones(10), index=late_idx)})
    df.index.name = "datetime"
    df.to_csv(feed)
    thesis = Thesis.model_validate(
        {
            "name": "never_available_or_leaf",
            "data": {"targets": ["target"], "external": {"iv": {}}},
            "entry": {
                "type": "or",
                "conditions": [
                    DIP_ENTRY,
                    {
                        "type": "threshold",
                        "left": {"type": "external", "name": "iv"},
                        "op": ">",
                        "right": {"type": "constant", "value": 0.0},
                    },
                ],
            },
            "params": {"horizon": 3},
        }
    )
    summary = run_backtest(thesis, load(thesis, {"target": path, "iv": feed})).summary
    # the runner's ledger stays factual: zero post-warmup holes, a null first_available
    leaf = summary["sources"]["target"]["by_source"]["external:iv"]
    assert leaf == {"n_missing": 0, "first_available": None}
    report = evaluate_gate(summary, GateThresholds())
    cell = report.cells[0]
    # the sibling fired the root and every other read is clean — support included
    assert _check(cell, "support").met and _check(cell, "signal_coverage").met
    assert _unmet_run(report) == ["source_coverage"]
    coverage = next(c for c in report.run_checks if c.name == "source_coverage")
    assert "external:iv" in coverage.detail and "never available" in coverage.detail
    assert cell.met is False  # the run-level refusal reaches every cell


def test_a_feed_that_starts_late_is_warmup_not_a_hole(tmp_path):
    # The fail-closed contract must not punish ordinary warmup: a feed with a LEADING gap is
    # uninitialized, not undecidable, and the cell is still met.
    n_bars = 1000
    px = _planted_dip_px(n_bars, 5, 9)
    late = px.copy()
    late[:120] = np.nan  # the feed simply starts later than the price series
    path, feed = tmp_path / "px.csv", tmp_path / "feed.csv"
    _write_px(path, px)
    _write_feed(feed, late, n_bars)
    thesis = _dip_thesis_on_feed()
    summary = run_backtest(thesis, load(thesis, {"target": path, "px": feed})).summary
    for entry in summary["cells"][0]["signal_coverage"].values():
        assert entry["n_undefined"] == 0
    report = evaluate_gate(summary, GateThresholds())
    assert _unmet_run(report) == []
    cell = report.cells[0]
    assert cell.met is True, f"a late-starting feed refused on: {_unmet_cell(cell)}"


def _hole_field(path, px, column, bar, n_bars=1000):
    """The golden price series with ONE interior NaN in a single OHLC column."""
    idx = pd.date_range("2015-01-02", periods=n_bars, freq="1D")
    s = pd.Series(px, index=idx)
    df = pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 1000.0}, index=idx)
    df.iloc[bar, df.columns.get_loc(column)] = np.nan
    df.index.name = "datetime"
    df.to_csv(path)
    return path


_AND_ENTRY = {
    "type": "and",
    "conditions": [
        DIP_ENTRY,
        {
            "type": "threshold",
            "left": {"type": "field", "column": "high"},
            "op": ">",
            "right": {"type": "constant", "value": 0.0},
        },
    ],
}


def test_kleene_absorbed_operand_hole_leaves_source_coverage_unmet(tmp_path):
    """The headline hole class: ``and`` recovers a verdict from a decisive child
    (``F∧U = F``), so holing the OTHER operand leaves the root condition fully DEFINED and the
    per-cell pooled counts read a clean zero. The bar is decided — what is unknowable is whether
    it would have decided the same way with the data present, which only the raw source panel
    can see."""
    px = _planted_dip_px(1000, 5, 9)
    clean = tmp_path / "clean.csv"
    _write_px(clean, px)
    assert evaluate_gate(_run_entry(clean, _AND_ENTRY), GateThresholds()).cells[0].met is True

    bar = 500
    assert px[bar] > 95.5, "the holed bar must be one the close operand decides False"
    holed = _hole_field(tmp_path / "holed.csv", px, "high", bar)
    summary = _run_entry(holed, _AND_ENTRY)

    # The pooled root channel is BLIND here — this is the regression, stated as an assertion.
    assert summary["cells"][0]["signal_coverage"]["target"]["n_undefined"] == 0
    src = summary["sources"]["target"]
    assert src["n_missing"] == 1
    assert src["by_source"]["field:high"]["n_missing"] == 1

    report = evaluate_gate(summary, GateThresholds())
    assert _unmet_run(report) == ["source_coverage"]
    assert "field:high" in _run_check(report, "source_coverage").detail
    cell = report.cells[0]
    assert _unmet_cell(cell) == []  # every per-cell read is clean; only the run level sees it
    assert cell.met is False  # and the unmet run-level check still leaves the cell unmet


def test_interior_hole_leaves_source_coverage_unmet(tmp_path):
    """A plain interior hole in a decision leaf. It is a hole wherever it sits — the evaluated
    interval is ONE undivided span, so there is no band that belongs to no pool."""
    px = _planted_dip_px(1000, 5, 9)
    holed = _hole_field(tmp_path / "px.csv", px, "close", 700)
    summary = _run_entry(holed, DIP_ENTRY)
    assert summary["sources"]["target"]["n_missing"] == 1
    report = evaluate_gate(summary, GateThresholds())
    assert _unmet_run(report) == ["source_coverage"]
    assert report.cells[0].met is False


def test_nan_skipping_transform_hole_leaves_source_coverage_unmet(tmp_path):
    """``ema`` SKIPS NaNs and carries its running state across a hole, emitting a finite value
    on the next bar — so the contaminated output reads as perfectly decided downstream. The
    source read sits under the transform, where no kernel can launder it."""
    px = _planted_dip_px(1000, 5, 9)
    entry = {
        "type": "threshold",
        "left": {"type": "ema", "window": 5, "input": {"type": "field", "column": "close"}},
        "op": "<",
        "right": {"type": "constant", "value": 97.0},
    }
    holed = _hole_field(tmp_path / "px.csv", px, "close", 400)
    summary = _run_entry(holed, entry)
    assert summary["sources"]["target"]["n_missing"] == 1
    report = evaluate_gate(summary, GateThresholds())
    check = _run_check(report, "source_coverage")
    assert not check.met and "field:close" in check.detail


def test_sparse_and_late_feeds_stay_legitimate(tmp_path):
    """The contract must not punish honest feed design. Interior SPARSENESS is the asof/ffill
    semantics a weekly feed on daily bars relies on (the joined value stays finite), and a feed
    that merely STARTS LATE is warmup — its ``first_available`` is reported as evidence rather
    than counted as a hole."""
    n_bars = 1000
    px = _planted_dip_px(n_bars, 5, 9)
    path, feed = tmp_path / "px.csv", tmp_path / "feed.csv"
    _write_px(path, px)

    # Sparse stamping: only every 5th bar carries a stamp; ffill covers the rest.
    idx = pd.date_range("2015-01-02", periods=n_bars, freq="1D")
    sparse = pd.DataFrame({"px": pd.Series(px, index=idx)}).iloc[::5]
    sparse.index.name = "datetime"
    sparse.to_csv(feed)
    thesis, files = _dip_thesis_on_feed(), {"target": path, "px": feed}
    summary = run_backtest(thesis, load(thesis, files)).summary
    assert summary["sources"]["target"]["n_missing"] == 0

    # Late start: leading unavailability is warmup, and the first available bar is on record.
    late = px.copy()
    late[:120] = np.nan
    _write_feed(feed, late, n_bars)
    summary = run_backtest(thesis, load(thesis, files)).summary
    src = summary["sources"]["target"]
    assert src["n_missing"] == 0
    assert src["by_source"]["external:px"]["first_available"].startswith("2015-05-02")
    report = evaluate_gate(summary, GateThresholds())
    assert _unmet_run(report) == [] and report.cells[0].met is True


def test_source_panel_geometry_is_data_independent(tmp_path):
    """``n_bars`` is pure geometry — the evaluated interval's length. Only the COUNTS may move
    under a data mutation, never the denominator."""
    n_bars = 1000
    px = _planted_dip_px(n_bars, 5, 9)
    _write_px(tmp_path / "a.csv", px)
    base = _run_entry(tmp_path / "a.csv", DIP_ENTRY)["sources"]["target"]
    holed = _hole_field(tmp_path / "b.csv", px, "close", 400)
    after = _run_entry(holed, DIP_ENTRY)["sources"]["target"]
    assert base["n_bars"] == after["n_bars"] == n_bars
    assert base["n_missing"] == 0 and after["n_missing"] == 1


# ---- the honesty invariant ---------------------------------------------------------------------


def test_summary_verbatim_regardless_of_cell_outcomes(tmp_path):
    # Met or unmet, the same evaluate_gate input summary is untouched: the gate renders a
    # per-cell result ABOUT the summary and can never filter or edit it.
    path = tmp_path / "px.csv"
    _write_px(path, _dip_then_fade_px(3000))
    summary = _run_entry(path, DIP_ENTRY, horizon=[3, 12])
    before = json.dumps(json_safe(summary), sort_keys=True)  # NaN-safe structural snapshot
    mixed = evaluate_gate(summary, GateThresholds())
    none_pass = evaluate_gate(summary, GateThresholds(thesis_min_n_nonoverlap=10000))
    assert mixed.to_dict()["n_met"] == 1 and none_pass.to_dict()["n_met"] == 0
    assert json.dumps(json_safe(summary), sort_keys=True) == before


def test_no_check_reads_the_evidence_riders(tmp_path):
    # "No check reads them", asserted mechanically: DELETE every evidence-only rider — the
    # attribution legs, profit_factor, best_ret, edge_ratio, the excursion means, and for good
    # measure the whole quantile/boot/subperiod blocks — from a real emitted summary and the
    # gate's verdict document is DEEP-EQUAL to the one graded on the full summary. A future
    # check that quietly starts consuming a rider fails here, by construction.
    path = tmp_path / "px.csv"
    _write_px(path, _dip_then_fade_px(3000))
    summary = _run_entry(path, DIP_ENTRY, horizon=[3, 12])
    graded_full = json_safe(evaluate_gate(summary, GateThresholds()).to_dict())
    stripped = json.loads(json.dumps(json_safe(summary)))  # deep copy, NaN → null like the wire
    riders = {
        "mean_ret_raw",
        "mean_ret_bench",
        "benchmark_regression",
        "profit_factor",
        "win_loss_ratio",
        "std_ret",
        "skewness",
        "kurtosis",
        "tail_ratio",
        "cvar_5",
        "best_ret",
        "edge_ratio",
        "worst_ret",
        "ret_quantiles",
        "mae_quantiles",
        "mfe_quantiles",
        "boot",
        "subperiods",
        "timing",
    }
    for cell in stripped["cells"]:
        for panel in cell["by_target"].values():
            for key in riders:
                panel.pop(key, None)
        # The whole episode-deduplicated twin is a rider too: the gate reads episode_stats,
        # never its profile.
        cell.pop("episode_profile", None)
    graded_stripped = json_safe(evaluate_gate(stripped, GateThresholds()).to_dict())
    assert graded_stripped == graded_full


# ---- basket mode: the pooled rubric, end-to-end -------------------------------------------------
#
# A planted TWO-MEMBER basket whose members disagree: member A's events rebound ≈ +3.6%, member
# B's fade ≈ −3.0%, in comparable |return| magnitudes and counts. The pooled panel therefore
# carries a small positive edge spread across BOTH members' masses (share ≈ 0.54/0.46), so the
# basket rubric grades the cell met while the per-member floor conjunction would apply refuses B —
# the anti-weakest doctrine, exercised on a real emitted summary rather than a hand-built one.
#
# The entry reads a per-member decision FEED (1.0 everywhere, 0.0 on event bars), so the price
# path is free to carry any planted response. Events are SPARSE (gaps 6–10 > h) on purpose: a
# basket entry that fires (for some member) nearly every bar chains ALL observations into ONE
# merged episode cluster, and concentration then refuses at max_cluster_share_abs=1.0 — correct,
# pre-existing doctrine, pinned separately below.

N_BASKET_BARS = 1200


def _planted_event_bars(n_bars: int, gap_lo: int, gap_hi: int, *, start: int, seed: int):
    """Aperiodic event bars; the gap floor keeps consecutive planted 4-bar responses disjoint."""
    rng = np.random.RandomState(seed)
    bars, pos = [], start
    while pos < n_bars - 8:
        bars.append(pos)
        pos += int(rng.randint(gap_lo, gap_hi + 1))
    return bars


def _write_basket_member(px_path, sig_path, events, steps, *, seed: int) -> None:
    """One basket member: a ~100-level price carrying a planted 3-bar response after each event
    bar (fill open[t+1] → exit open[t+4] under h=3), and the member's decision feed."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2015-01-02", periods=N_BASKET_BARS, freq="1D")
    px = np.full(N_BASKET_BARS, 100.0)
    for t in events:
        px[t + 1 : t + 5] = 100.0 * np.asarray(steps)
    _write_px(px_path, px * (1.0 + rng.normal(0.0, 0.002, size=N_BASKET_BARS)))
    sig = np.ones(N_BASKET_BARS)
    sig[events] = 0.0
    feed = pd.DataFrame({"sig": pd.Series(sig, index=idx)})
    feed.index.name = "datetime"
    feed.to_csv(sig_path)


def _basket_summary(tmp_path) -> dict:
    _write_basket_member(
        tmp_path / "A.csv",
        tmp_path / "sigA.csv",
        _planted_event_bars(N_BASKET_BARS, 6, 10, start=12, seed=0),
        (1.000, 1.012, 1.024, 1.036),
        seed=2,  # rebounds: ≈ +3.6% per observation
    )
    _write_basket_member(
        tmp_path / "B.csv",
        tmp_path / "sigB.csv",
        _planted_event_bars(N_BASKET_BARS, 6, 10, start=15, seed=1),
        (1.000, 0.990, 0.980, 0.970),
        seed=3,  # fades: ≈ −3.0% per observation
    )
    thesis = Thesis.model_validate(
        {
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
    )
    # a per-target feed answers to one derived key per member, `sig@A` / `sig@B`
    files = {
        "A": tmp_path / "A.csv",
        "B": tmp_path / "B.csv",
        "sig@A": tmp_path / "sigA.csv",
        "sig@B": tmp_path / "sigB.csv",
    }
    return run_backtest(thesis, load(thesis, files)).summary


def test_basket_golden_grades_the_pooled_panel_not_the_members(tmp_path):
    summary = _basket_summary(tmp_path)
    report = evaluate_gate(summary, GateThresholds())

    assert summary["target_mode"] == "basket"
    assert _unmet_run(report) == []
    cell = report.cells[0]
    assert cell.met is True, f"the planted basket cell failed on: {_unmet_cell(cell)}"

    # the anti-weakest pin, end-to-end: member B's own mean is NEGATIVE and the cell still
    # is met, because basket support reads the pooled panel and never a member's own floor —
    # `by_target` is attribution evidence, read by no check
    s_cell = summary["cells"][0]
    assert s_cell["by_target"]["B"]["mean_ret"] < 0 < s_cell["by_target"]["A"]["mean_ret"]
    assert s_cell["pooled"]["mean_ret"] > 0
    assert _check(cell, "support").observed == {
        "pooled": {k: s_cell["pooled"][k] for k in ("n", "n_nonoverlap", "mean_ret")}
    }
    # …and the pool is genuinely both members' mass, not one name wearing a basket
    shares = s_cell["pooled"]["member_share"]["by_target"]
    assert max(shares.values()) < 0.6
    assert _check(cell, "concentration").observed["max_member_share_abs"] == max(shares.values())
    # the basket baseline carries the pooled base-rate row beside the per-member ones
    assert set(summary["baseline"][0]["by_target"]) == {"A", "B"}
    assert summary["baseline"][0]["pooled"]["n_eligible"] > 0


def test_an_every_bar_basket_entry_is_one_merged_episode(tmp_path):
    # The domain fact behind every sparse basket fixture in this file: an entry that fires on
    # every bar chains ALL observations — across bars AND members — into ONE overlap-merged
    # episode cluster, and concentration correctly refuses at max_cluster_share_abs=1.0. A
    # basket thesis that is "always on" has one episode of evidence, however many rows it wrote;
    # `first_true` (with a cooldown) is how a caller makes episodes countable.
    _planted_dip_series(tmp_path / "a.csv", 400, 5, 9, seed=0)
    _planted_dip_series(tmp_path / "b.csv", 400, 5, 9, seed=1)
    thesis = Thesis.model_validate(
        {
            "name": "always_on",
            "target_mode": "basket",
            "data": {"targets": ["A", "B"]},
            "entry": {
                "type": "threshold",
                "left": {"type": "field", "column": "close"},
                "op": "<",
                "right": {"type": "constant", "value": 1e9},
            },
            "params": {"horizon": 3},
        }
    )
    files = {"A": tmp_path / "a.csv", "B": tmp_path / "b.csv"}
    summary = run_backtest(thesis, load(thesis, files)).summary
    ep = summary["cells"][0]["episode_stats"]
    assert ep["n_clusters"] == 1 and ep["max_cluster_share_abs"] == 1.0
    report = evaluate_gate(summary, GateThresholds())
    cell = report.cells[0]
    assert "concentration" in _unmet_cell(cell)
    assert "max_cluster_share_abs=1.00" in _check(cell, "concentration").detail
    assert "one merged episode cluster" in _check(cell, "concentration").detail


# ---- the evidence/gate seam: what may vanish and what may not ----------------------------------


def _strip_evidence_only(summary: dict) -> dict:
    """Remove every evidence-only block: the run-level baseline, the per-cell narrative panels,
    and (where present) the pooled panel's non-gated fields."""
    stripped = copy.deepcopy(summary)
    stripped.pop("baseline")
    for cell in stripped["cells"]:
        for key in (
            "episodes",
            "episode_profile",
            "conditional_buckets",
            "bucket_monotonicity",
            "feature_association",
        ):
            cell.pop(key)
        if "pooled" in cell:
            for key in (
                "boot",
                "subperiods",
                "ret_quantiles",
                "mae_quantiles",
                "mfe_quantiles",
                "t_hac",
                "hac_se",
                "rot_p",
                "rot_n_null",
                "benchmark_regression",
                "win_loss_ratio",
                "std_ret",
                "skewness",
                "kurtosis",
                "tail_ratio",
                "cvar_5",
                "timing",
            ):
                cell["pooled"].pop(key)
    return stripped


def test_stripping_evidence_only_blocks_changes_no_check_result(tmp_path):
    # The seam, pinned from both sides of the mode dispatch: everything the contract
    # calls evidence-only — baseline, the episode ledger, per-cell buckets/monotonicity, feature
    # association, and the pooled panel's non-gated reads — can be deleted wholesale without
    # moving a single check result, run-level or per-cell, in EITHER mode. A check whose result
    # moved would have been reading a panel the contract says no check reads.
    path = tmp_path / "px.csv"
    _write_px(path, _dip_then_fade_px(3000))
    conjunction = _run_entry(path, DIP_ENTRY, horizon=[3, 12])
    basket = _basket_summary(tmp_path)
    for summary in (conjunction, basket):
        base = evaluate_gate(summary, GateThresholds()).to_dict()
        stripped = evaluate_gate(_strip_evidence_only(summary), GateThresholds()).to_dict()
        assert stripped == base
    # the pin is not vacuous: one mode graded a mixed grid, the other a fully met basket cell
    assert evaluate_gate(conjunction, GateThresholds()).to_dict()["n_met"] == 1
    assert evaluate_gate(basket, GateThresholds()).to_dict()["n_met"] == 1


def test_stripping_a_gated_pooled_field_refuses(tmp_path):
    # The other side of the seam: the pooled fields the basket rubric GRADES may not vanish.
    # Deleting any of them flips the named dispatching check to a refusal — and only ever that
    # direction: no unmet check can come out met because evidence went missing.
    summary = _basket_summary(tmp_path)
    base = evaluate_gate(summary, GateThresholds())
    assert base.cells[0].met is True
    gated = {
        "n": {"cell_evidence", "support"},
        "n_nonoverlap": {"cell_evidence", "support"},
        "mean_ret": {"support"},
        "concentration": {"concentration"},
        "member_share": {"concentration"},
    }
    for field, expect_unmet in gated.items():
        tampered = copy.deepcopy(summary)
        tampered["cells"][0]["pooled"].pop(field)
        report = evaluate_gate(tampered, GateThresholds())
        cell = report.cells[0]
        assert cell.met is False, f"stripping pooled.{field} must refuse"
        assert set(_unmet_cell(cell)) == expect_unmet, (
            f"stripping pooled.{field} refused on {_unmet_cell(cell)}"
        )
        assert _unmet_run(report) == []  # the refusal is localized to the cell that lost it
