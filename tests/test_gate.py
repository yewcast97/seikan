"""The per-cell checklist (``gate.evaluate_gate``) + the DSL hash discipline.

The gate is a pure function over one engine summary dict — no storage, no models, no agent. Its
contract: EVERY check evaluated and reported (no short-circuit), every DECLARED cell graded
INDEPENDENTLY, and the summary itself never filtered. The checklist under test here: three
run-level checks (``evidence_complete``, ``source_coverage``, ``search_cap``) plus five per-cell
checks (``cell_evidence``, ``outcome_coverage``, ``signal_coverage``, ``support``,
``concentration``), NO scalar verdict and no inferential exam anywhere — a cell's ``met`` is a
completeness / support / concentration result, not a significance claim. The summary's
``target_mode`` stamp selects the rubric (conjunction: per-target floors and ceilings, weakest
target decides; basket: the pooled panel is graded as ONE evidence pool) and a missing or
garbage stamp refuses fail-closed everywhere it would have dispatched. There is no in-sample/
out-of-sample split, so end-of-data right-censoring (``open``) is structural and allowed while an
in-bounds hole (``no_outcome``/``no_benchmark``) refuses; the fail-closed decision-side contract
holds in both layers (the per-cell undefined-decision ledger and the run-level per-source
availability panel) and stays PER-TARGET in both modes; the commensurability guard, strict
numeric hygiene (NaN / ±inf / non-integral counts refuse), and the canonical-floor thresholds
apply uniformly.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from seikan.analysis.stats import STATISTICS_VERSION
from seikan.gate import POLICY_VERSION, GateThresholds, canonical_dsl_hash, evaluate_gate
from tests._helpers import keys_deep as _keys_deep

#: The run-level checks, in report order.
_RUN_NAMES = ["evidence_complete", "source_coverage", "search_cap"]

#: The per-cell checks, in report order — the same five for every cell.
_CELL_NAMES = [
    "cell_evidence",
    "outcome_coverage",
    "signal_coverage",
    "support",
    "concentration",
]

#: The joined index length every geometry read is verified against.
_N_BARS = 600


def _tgt_cell(**over) -> dict:
    """One per-target entry of a cell's ``by_target`` panel that clears every read."""
    c = {
        "n": 42,
        "n_nonoverlap": 14,
        "mean_ret": 0.05,
        "hit_rate": 0.6,
        "t_hac": 2.4,
        "hac_se": 0.02,
        "rot_p": 0.03,
        "concentration": {"top_share_abs": 0.3, "n_top": 3, "top_frac": 0.05},
        # Evidence riders — no check reads either (the regressions below prove it).
        "boot": {
            "method": "episode_percentile",
            "ci_level": 0.95,
            "n_boot": 2000,
            "n_episodes": 12,
            "ci_lo": 0.01,
            "ci_hi": 0.09,
            "boot_se": 0.02,
            "reason": None,
        },
        "subperiods": [
            {
                "start": "2020-01-01T00:00:00",
                "end": "2020-10-14T00:00:00",
                "n": 14,
                "mean_ret": 0.04,
            },
            {
                "start": "2020-10-15T00:00:00",
                "end": "2021-08-03T00:00:00",
                "n": 14,
                "mean_ret": 0.05,
            },
            {
                "start": "2021-08-04T00:00:00",
                "end": "2022-05-24T00:00:00",
                "n": 14,
                "mean_ret": 0.06,
            },
        ],
    }
    c.update(over)
    return c


def _cov(n: int = 42, **over) -> dict:
    """One clean censoring-ledger entry — ONE pool (no holdout, so no in_sample/oos/embargo
    split), fully measured: every firing closed at its horizon."""
    c = {
        "n_attempted": n,
        "n_closed": n,
        "exit_reasons": {"horizon": n, "open": 0, "no_outcome": 0, "no_benchmark": 0},
    }
    c.update(over)
    return c


def _sig(**over) -> dict:
    """One clean undefined-decision ledger entry: every post-warmup decision bar decidable.
    ``n_bars`` is pure geometry, so it always equals the summary's ``n_bars``."""
    c = {"n_bars": _N_BARS, "n_undefined": 0}
    c.update(over)
    return c


def _cell(
    targets: tuple[str, ...] = ("AAA",),
    n: int = 42,
    n_nonoverlap: int | None = None,
    mean_ret: float = 0.05,
    top_share: float = 0.3,
    cluster_share: float = 0.3,
    cell_id: str = "zscore_window=20,horizon=21",
    params: dict | None = None,
    **over,
) -> dict:
    """One declared parameter × horizon cell whose panels reconcile by construction.

    ``n`` drives the graded count, the ledger's attempted/closed counts and the episode panel's
    per-target total together — the reconciliation ``cell_evidence`` enforces — so a test that
    wants to break ONE check moves that check's own field and nothing else.
    """
    tg = list(targets)
    n_nonoverlap = min(14, n) if n_nonoverlap is None else n_nonoverlap
    c = {
        "cell_id": cell_id,
        "params": params if params is not None else {"zscore_window": 20, "horizon": 21},
        "by_target": {
            t: _tgt_cell(
                n=n,
                n_nonoverlap=n_nonoverlap,
                mean_ret=mean_ret,
                concentration={"top_share_abs": top_share, "n_top": 3, "top_frac": 0.05},
            )
            for t in tg
        },
        "episode_stats": {
            "n": n * len(tg),
            "n_clusters": 12,
            "largest_cluster_n": 5,
            "largest_cluster_share_abs": cluster_share,
            "largest_cluster_start": None,
            "max_cluster_share_abs": cluster_share,
        },
        "outcome_coverage": {t: _cov(n) for t in tg},
        "signal_coverage": {t: _sig() for t in tg},
    }
    c.update(over)
    return c


def _sources(**over) -> dict:
    """One clean per-source availability panel: every raw decision leaf the entry tree reads is
    available across the whole evaluated interval."""
    c = {
        "n_bars": _N_BARS,
        "n_missing": 0,
        "by_source": {
            "field:close": {"n_missing": 0, "first_available": "2020-01-01T00:00:00"},
            "external:iv30": {"n_missing": 0, "first_available": "2020-01-01T00:00:00"},
        },
    }
    c.update(over)
    return c


def _summary(cells: list | None = None, targets: tuple[str, ...] = ("AAA",), **over) -> dict:
    """A summary whose every declared cell is met under canonical thresholds."""
    tg = list(targets)
    cells = [_cell(targets=tuple(tg))] if cells is None else cells
    s = {
        "targets": tg,
        "statistics_version": STATISTICS_VERSION,
        "gate_evidence_basis": "full_sample",
        # The measurement-algebra stamp — ALWAYS the explicit dict the runner writes (the default
        # run stamps pct-on-target). A summary that OMITS it, or carries a null in its place, is
        # drifted input — every reported claim would be denominated in nothing.
        "outcome": {"series": "target", "kind": "pct", "units": "fraction"},
        # The rubric-selection stamp — the runner ALWAYS writes it, and the checklist dispatches
        # on it: a summary without it refuses fail-closed.
        "target_mode": "conjunction",
        "n_bars": _N_BARS,
        "index_start": "2020-01-01T00:00:00",
        "index_end": "2022-05-24T00:00:00",
        "n_hypotheses_attempted": len(cells),
        "sources": {t: _sources() for t in tg},
        "cells": cells,
        # Evidence-only riders the checklist must never read (regressions below prove it).
        "direction": "longonly",
        "benchmark": None,
        "benchmark_source": None,
        "target_shape": "ohlcv",
        "rotation": {"n_shifts": 599, "p_resolution": 1 / 600},
        "bar_spacing": {"min_seconds": 86400, "median_seconds": 86400, "max_seconds": 259200},
        "pbo": {"pbo": 0.2, "reason": None, "prob_oos_loss": 0.0},
        # No run-level pooled conditional_buckets / bucket_monotonicity pair: pooled conditioning
        # depends on grid composition, so the fixture models the summary the runner actually
        # emits, which carries neither.
    }
    s.update(over)
    return s


def _two_targets(cells: list | None = None, **over) -> dict:
    """A two-target fully met summary — every run-level and per-cell panel covers both."""
    tg = ("AAA", "BBB")
    cells = [_cell(targets=tg)] if cells is None else cells
    return _summary(cells=cells, targets=tg, **over)


def _pooled(
    targets: tuple[str, ...] = ("AAA", "BBB"),
    n: int = 84,
    n_nonoverlap: int = 20,
    mean_ret: float = 0.05,
    top_share: float = 0.3,
    **over,
) -> dict:
    """One pooled cross-target panel that clears every basket read — the full field
    set (n, n_nonoverlap, mean_ret, hit_rate, t_hac, hac_se, rot_p, concentration, member_share,
    boot,
    subperiods, ret_quantiles, worst_ret, mae_quantiles, mfe_quantiles). Equal member shares by
    default, so ``max_member_share_abs`` sits under the sealed ceiling."""
    tg = list(targets)
    share = 1.0 / len(tg)
    p = {
        "n": n,
        "n_nonoverlap": n_nonoverlap,
        "mean_ret": mean_ret,
        "hit_rate": 0.6,
        "t_hac": 2.1,
        "hac_se": 0.02,
        "rot_p": 0.04,
        "concentration": {"top_share_abs": top_share, "n_top": 5, "top_frac": 0.05},
        "member_share": {
            "by_target": dict.fromkeys(tg, share),
            "max_member_share_abs": share,
        },
        # Pooled evidence riders — no check reads any of them (regressions below prove it).
        "boot": {
            "method": "episode_percentile",
            "ci_level": 0.95,
            "n_boot": 2000,
            "n_episodes": 15,
            "ci_lo": 0.01,
            "ci_hi": 0.08,
            "boot_se": 0.015,
            "reason": None,
        },
        "subperiods": [
            {
                "start": "2020-01-01T00:00:00",
                "end": "2020-10-14T00:00:00",
                "n": 28,
                "mean_ret": 0.04,
            },
            {
                "start": "2020-10-15T00:00:00",
                "end": "2021-08-03T00:00:00",
                "n": 28,
                "mean_ret": 0.05,
            },
            {
                "start": "2021-08-04T00:00:00",
                "end": "2022-05-24T00:00:00",
                "n": 28,
                "mean_ret": 0.06,
            },
        ],
        "ret_quantiles": {"p10": -0.03, "p25": -0.005, "p50": 0.03, "p75": 0.08, "p90": 0.13},
        "worst_ret": -0.18,
        "mae_quantiles": {
            "n": 84,
            "p10": -0.09,
            "p25": -0.05,
            "p50": -0.02,
            "p75": -0.01,
            "p90": 0.0,
            "worst": -0.22,
        },
        "mfe_quantiles": {
            "n": 84,
            "p10": 0.0,
            "p25": 0.01,
            "p50": 0.03,
            "p75": 0.06,
            "p90": 0.11,
            "best": 0.24,
        },
    }
    p.update(over)
    return p


def _basket_cell(
    targets: tuple[str, ...] = ("AAA", "BBB"),
    n: int = 42,
    pooled: dict | None = None,
    **over,
) -> dict:
    """A basket cell: the conjunction fixture plus the pooled panel, reconciling by
    construction (``pooled.n == n × len(targets) == episode_stats.n``)."""
    tg = tuple(targets)
    cell = _cell(targets=tg, n=n, **over)
    cell["pooled"] = _pooled(targets=tg, n=n * len(tg)) if pooled is None else pooled
    return cell


def _basket(cells: list | None = None, targets: tuple[str, ...] = ("AAA", "BBB"), **over) -> dict:
    """A basket-mode summary whose every declared cell is met under canonical thresholds."""
    tg = tuple(targets)
    cells = [_basket_cell(targets=tg)] if cells is None else cells
    return _summary(cells=cells, targets=tg, target_mode="basket", **over)


def _run(report) -> dict:
    return {c.name: c for c in report.run_checks}


def _cell_checks(report, i: int = 0) -> dict:
    return {c.name: c for c in report.cells[i].checks}


# ---- checklist contract ---------------------------------------------------------------------


def test_fully_met_summary_grades_every_cell_and_reports_both_checklists():
    report = evaluate_gate(_summary())
    assert [c.name for c in report.run_checks] == _RUN_NAMES
    assert len(report.cells) == 1
    assert [c.name for c in report.cells[0].checks] == _CELL_NAMES
    assert report.cells[0].met is True


def test_every_declared_cell_carries_the_same_five_checks_in_order():
    cells = [_cell(cell_id=f"horizon={h}", params={"horizon": h}) for h in (5, 10, 21)]
    report = evaluate_gate(_summary(cells=cells))
    assert len(report.cells) == 3
    for graded in report.cells:
        assert [c.name for c in graded.checks] == _CELL_NAMES


def test_report_serializes_to_the_declared_shape():
    doc = evaluate_gate(_summary()).to_dict()
    assert set(doc) == {"policy_version", "n_cells", "n_met", "run_checks", "cells"}
    assert doc["policy_version"] == POLICY_VERSION == 3
    assert doc["n_cells"] == 1
    assert doc["n_met"] == 1
    assert [c["name"] for c in doc["run_checks"]] == _RUN_NAMES
    assert set(doc["cells"][0]) == {"cell_id", "params", "met", "checks"}
    assert doc["cells"][0]["cell_id"] == "zscore_window=20,horizon=21"
    assert doc["cells"][0]["params"] == {"zscore_window": 20, "horizon": 21}
    assert all(
        {"name", "met", "observed", "threshold", "detail"} == set(c)
        for c in doc["run_checks"] + doc["cells"][0]["checks"]
    )


def test_no_verdict_or_pass_fail_key_anywhere_in_the_report():
    # There is no scalar verdict: exit 0 says the run finished, and the per-cell `met` flags
    # are the whole result. A stray "verdict" anywhere would invite the reading that one number
    # certifies the run — and the retired pass/fail vocabulary must not resurface as keys either:
    # `met` states that a neutral threshold comparison held, and the judgement stays the caller's.
    doc = evaluate_gate(_summary(cells=[_cell(), _cell(n=10)])).to_dict()
    assert not {"verdict", "passed", "n_passed"} & set(_keys_deep(doc))


def test_the_checklist_defines_exactly_these_checks():
    # The whole checklist is three run-level checks plus five per-cell checks, and nothing else.
    # In particular there is no `selection_integrity` exam — nothing is selected, every declared
    # cell is graded independently — and no `oos_confirmation` exam, since there is no holdout to
    # confirm against.
    report = evaluate_gate(_summary())
    names = [c.name for c in report.run_checks] + [c.name for c in report.cells[0].checks]
    assert names == _RUN_NAMES + _CELL_NAMES
    assert "selection_integrity" not in names
    assert "oos_confirmation" not in names


def test_no_short_circuit_every_check_reported_on_multiple_unmet_checks():
    # Break a run-level check and several of one cell's checks at once — both checklists must
    # still arrive whole.
    cell = _cell(n=10, top_share=0.9)
    cell["signal_coverage"]["AAA"]["n_undefined"] = 4
    report = evaluate_gate(
        _summary(
            cells=[cell],
            sources={
                "AAA": _sources(
                    n_missing=2,
                    by_source={
                        "field:close": {"n_missing": 2, "first_available": "2020-01-01T00:00:00"},
                    },
                )
            },
        )
    )
    assert [c.name for c in report.run_checks] == _RUN_NAMES
    assert [c.name for c in report.cells[0].checks] == _CELL_NAMES
    checks = _cell_checks(report)
    assert not _run(report)["source_coverage"].met
    assert not checks["support"].met
    assert not checks["concentration"].met
    assert not checks["signal_coverage"].met
    assert checks["outcome_coverage"].met  # untouched checks still evaluated + reported
    assert checks["cell_evidence"].met


def test_empty_summary_reports_the_run_checklist_and_no_cells():
    report = evaluate_gate({})
    assert [c.name for c in report.run_checks] == _RUN_NAMES
    assert report.cells == []
    assert not _run(report)["evidence_complete"].met
    assert not _run(report)["source_coverage"].met
    doc = report.to_dict()
    assert doc["n_cells"] == 0 and doc["n_met"] == 0


def test_no_applicable_field_in_any_serialized_check():
    # There is no `applicable` field: every check is structurally always applicable, so such a
    # field would carry no information. It must not appear anywhere in the report.
    doc = evaluate_gate(_summary()).to_dict()
    assert "applicable" not in set(_keys_deep(doc))
    check_keys = {"name", "met", "observed", "threshold", "detail"}
    for c in doc["run_checks"]:
        assert set(c) == check_keys
    for cell in doc["cells"]:
        for c in cell["checks"]:
            assert set(c) == check_keys


# ---- per-cell independence (the identity's whole point) -------------------------------------


def test_a_unmet_cell_never_drags_down_its_siblings():
    # Cell B's support floor is broken; cell A is untouched. Cells are graded ALONE, so the
    # report must say "one of two met" — not refuse the run.
    ok = _cell(cell_id="horizon=21", params={"horizon": 21})
    bad = _cell(cell_id="horizon=5", params={"horizon": 5}, n=10)
    report = evaluate_gate(_summary(cells=[ok, bad]))
    assert report.cells[0].met is True
    assert report.cells[1].met is False
    assert not _cell_checks(report, 1)["support"].met
    assert _cell_checks(report, 0)["support"].met
    assert report.to_dict()["n_met"] == 1
    assert report.to_dict()["n_cells"] == 2


def test_cells_are_index_aligned_with_the_summary_panel():
    cells = [_cell(cell_id=f"horizon={h}", params={"horizon": h}) for h in (5, 10, 21)]
    report = evaluate_gate(_summary(cells=cells))
    assert [c.cell_id for c in report.cells] == ["horizon=5", "horizon=10", "horizon=21"]
    assert [c.params["horizon"] for c in report.cells] == [5, 10, 21]


def test_nothing_is_summed_across_cells():
    # Horizon siblings legitimately repeat the same combo-keyed signal_coverage counts. Summing
    # them would manufacture a breach out of a clean run; each cell is graded on its own numbers.
    cells = [
        _cell(cell_id="horizon=5", params={"horizon": 5}),
        _cell(cell_id="horizon=21", params={"horizon": 21}),
    ]
    report = evaluate_gate(_summary(cells=cells))
    assert all(c.met for c in report.cells)
    assert report.to_dict()["n_met"] == 2


# ---- an unmet run-level check leaves EVERY cell unmet ---------------------------------------


def test_an_unmet_run_level_check_leaves_every_cell_unmet_while_reporting_each_honestly():
    # The run's evidence is the ground every per-cell read stands on, so a run-level breach
    # leaves all of them unmet — and each cell still reports its own five checks truthfully
    # (they held).
    cells = [
        _cell(cell_id="horizon=5", params={"horizon": 5}),
        _cell(cell_id="horizon=21", params={"horizon": 21}),
    ]
    report = evaluate_gate(
        _summary(
            cells=cells,
            sources={
                "AAA": _sources(
                    n_missing=1,
                    by_source={
                        "field:close": {"n_missing": 1, "first_available": "2020-01-01T00:00:00"},
                    },
                )
            },
        )
    )
    assert not _run(report)["source_coverage"].met
    assert report.to_dict()["n_met"] == 0
    for i in range(2):
        assert report.cells[i].met is False
        assert all(c.met for c in report.cells[i].checks)  # the cell's OWN checks are clean


def test_an_unmet_run_level_check_still_reports_the_cell_panel():
    # A caller must be able to read every cell's evidence even when the run cannot be certified —
    # the checklist renders a result about the summary, it never filters it.
    report = evaluate_gate(_summary(statistics_version=STATISTICS_VERSION - 1))
    assert not _run(report)["evidence_complete"].met
    assert len(report.cells) == 1
    assert [c.name for c in report.cells[0].checks] == _CELL_NAMES
    assert report.cells[0].met is False


# ---- evidence_complete (run level) -----------------------------------------------------------


def test_evidence_complete_requires_matching_statistics_version():
    # A summary from a different estimator revision must refuse ungraded, never be graded by
    # the wrong rubric.
    check = _run(evaluate_gate(_summary(statistics_version=STATISTICS_VERSION - 1)))[
        "evidence_complete"
    ]
    assert not check.met
    assert "statistics_version" in check.detail
    s = _summary()
    del s["statistics_version"]
    assert not _run(evaluate_gate(s))["evidence_complete"].met


def test_evidence_complete_requires_the_full_sample_basis():
    check = _run(evaluate_gate(_summary(gate_evidence_basis="in_sample")))["evidence_complete"]
    assert not check.met
    assert "gate_evidence_basis" in check.detail
    s = _summary()
    del s["gate_evidence_basis"]
    assert not _run(evaluate_gate(s))["evidence_complete"].met


def test_evidence_complete_requires_a_string_target_list():
    s = _summary()
    del s["targets"]
    check = _run(evaluate_gate(s))["evidence_complete"]
    assert not check.met
    assert "targets" in check.detail
    assert not _run(evaluate_gate(_summary(targets=[])))["evidence_complete"].met


def test_mixed_type_target_names_refuse_without_crashing():
    # Target names index every panel and are sorted in half a dozen places, so a non-string name
    # would raise TypeError out of `sorted` — a CRASH where the contract promises a refusal.
    report = evaluate_gate(_summary(targets=["AAA", 3]))
    assert [c.name for c in report.run_checks] == _RUN_NAMES  # the checklist still ran whole
    check = _run(report)["evidence_complete"]
    assert not check.met
    assert "list of strings" in check.detail
    assert report.cells[0].met is False


def test_evidence_complete_requires_a_readable_outcome_stamp():
    s = _summary()
    del s["outcome"]
    check = _run(evaluate_gate(s))["evidence_complete"]
    assert not check.met
    assert "outcome stamp" in check.detail
    check = _run(evaluate_gate(_summary(outcome={"series": "target", "kind": "bogus"})))[
        "evidence_complete"
    ]
    assert not check.met
    assert "unreadable" in check.detail
    assert not _run(evaluate_gate(_summary(outcome="pct")))["evidence_complete"].met


def test_null_outcome_stamp_refuses_as_drifted_input():
    # The runner always stamps the explicit dict, so `outcome: None` — a summary describing itself
    # as measured in nothing at all — can only be stripped or hand-edited input. It refuses
    # fail-closed rather than being graded under an ASSUMED algebra.
    check = _run(evaluate_gate(_summary(outcome=None)))["evidence_complete"]
    assert not check.met
    assert "null" in check.detail


def test_partial_outcome_stamp_refuses():
    # A stamp without a string series names no measured object — same drifted-input refusal.
    check = _run(evaluate_gate(_summary(outcome={"kind": "pct"})))["evidence_complete"]
    assert not check.met
    assert "series" in check.detail


@pytest.mark.parametrize("kind", ["pct", "log", "diff"])
def test_evidence_complete_accepts_every_declared_algebra(kind):
    s = _summary(outcome={"series": "target", "kind": kind})
    assert _run(evaluate_gate(s))["evidence_complete"].met


def test_evidence_complete_requires_a_countable_declared_grid():
    check = _run(evaluate_gate(_summary(n_hypotheses_attempted=None)))["evidence_complete"]
    assert not check.met
    assert "n_hypotheses_attempted" in check.detail
    s = _summary()
    del s["n_hypotheses_attempted"]
    assert not _run(evaluate_gate(s))["evidence_complete"].met
    check = _run(evaluate_gate(_summary(n_hypotheses_attempted=0)))["evidence_complete"]
    assert not check.met
    assert "declared no hypothesis" in check.detail


def test_evidence_complete_requires_countable_geometry():
    s = _summary()
    del s["n_bars"]
    check = _run(evaluate_gate(s))["evidence_complete"]
    assert not check.met
    assert "n_bars" in check.detail
    check = _run(evaluate_gate(_summary(n_bars=0)))["evidence_complete"]
    assert not check.met
    assert "empty index" in check.detail


def test_evidence_complete_requires_every_declared_cell_on_the_record():
    # The honesty invariant of this identity: non-firing combos are reported, never dropped, so a
    # report short of its own declared grid has silently deleted hypotheses from the burden it
    # declares.
    check = _run(evaluate_gate(_summary(cells=[_cell()], n_hypotheses_attempted=4)))[
        "evidence_complete"
    ]
    assert not check.met
    assert "n_hypotheses_attempted=4" in check.detail
    check = _run(evaluate_gate(_summary(cells=[_cell(), _cell()], n_hypotheses_attempted=1)))[
        "evidence_complete"
    ]
    assert not check.met


def test_evidence_complete_requires_a_sources_panel_covering_the_regime():
    s = _summary()
    del s["sources"]
    check = _run(evaluate_gate(s))["evidence_complete"]
    assert not check.met
    assert "sources panel" in check.detail
    s = _two_targets(sources={"AAA": _sources()})  # BBB silently dropped
    check = _run(evaluate_gate(s))["evidence_complete"]
    assert not check.met
    assert "BBB" in check.detail
    s = _summary(sources={"AAA": _sources(), "ZZZ": _sources()})  # an unexpected target
    assert not _run(evaluate_gate(s))["evidence_complete"].met


def test_evidence_complete_rejects_non_string_source_panel_keys():
    check = _run(evaluate_gate(_summary(sources={7: _sources()})))["evidence_complete"]
    assert not check.met
    assert "non-string" in check.detail


def test_unreadable_cells_panel_yields_no_graded_cells_without_raising():
    for bad in ("nope", {"AAA": {}}, None, 7):
        s = _summary(n_hypotheses_attempted=1)
        s["cells"] = bad
        report = evaluate_gate(s)
        assert report.cells == []
        doc = report.to_dict()
        assert doc["n_cells"] == 0 and doc["n_met"] == 0
        assert doc["cells"] == []
        check = _run(report)["evidence_complete"]
        assert not check.met
        assert "cells panel" in check.detail


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 3.5, None, "4"])
def test_evidence_complete_uncountable_grid_size_refuses(bad):
    # int(1.9) == 1 truncation is drifted input, not a countable search size — and int(inf) used
    # to raise OverflowError, a crash where the contract promises a refusal.
    check = _run(evaluate_gate(_summary(cells=[_cell()], n_hypotheses_attempted=bad)))[
        "evidence_complete"
    ]
    assert not check.met


@pytest.mark.parametrize("bad", [math.nan, math.inf, 1.5, None])
def test_evidence_complete_uncountable_n_bars_refuses(bad):
    assert not _run(evaluate_gate(_summary(n_bars=bad)))["evidence_complete"].met


# ---- source_coverage (run level, fail-closed availability) -----------------------------------


def test_source_coverage_met_on_a_fully_available_thesis():
    check = _run(evaluate_gate(_summary()))["source_coverage"]
    assert check.met


def test_source_coverage_refuses_a_missing_decision_input():
    # The hole classes the per-cell three-valued ledger structurally cannot see: an operand
    # absorbed by a decisive sibling (Kleene F∧U = F) or a value laundered through a NaN-skipping
    # recursive kernel. Either shows up here as an unavailable SOURCE, with every pooled read
    # perfectly clean.
    report = evaluate_gate(
        _summary(
            sources={
                "AAA": _sources(
                    n_missing=3,
                    by_source={
                        "field:close": {"n_missing": 0, "first_available": "2020-01-01T00:00:00"},
                        "external:iv30": {"n_missing": 3, "first_available": "2020-01-01T00:00:00"},
                    },
                )
            }
        )
    )
    check = _run(report)["source_coverage"]
    assert not check.met
    assert "external:iv30" in check.detail
    # …and the per-cell ledgers are none the wiser, which is exactly why this layer exists.
    assert _cell_checks(report)["signal_coverage"].met
    assert _cell_checks(report)["outcome_coverage"].met
    assert report.cells[0].met is False


def test_a_late_starting_source_is_warmup_not_a_hole():
    # The observer had nothing to read yet — exactly like a transform's warmup window. The panel
    # records `first_available` as evidence and the check is met.
    s = _summary(
        sources={
            "AAA": _sources(
                by_source={
                    "field:close": {"n_missing": 0, "first_available": "2020-01-01T00:00:00"},
                    "external:iv30": {"n_missing": 0, "first_available": "2021-06-30T00:00:00"},
                }
            )
        }
    )
    check = _run(evaluate_gate(s))["source_coverage"]
    assert check.met
    obs = check.observed["by_target"]["AAA"]["by_source"]["external:iv30"]
    assert obs["first_available"] == "2021-06-30T00:00:00"


def test_a_never_available_source_refuses():
    # Warmup requires a start to EXIST (policy v3). The old rationale — "the run cannot fire on
    # it anyway, so support refuses" — is defeated by a decisive sibling: `X or <never-available
    # leaf>` fires through Kleene absorption (T∨U = T), support grades the sibling's firings,
    # and the missing leaf leaves no trace anywhere else. The panel records the fact factually
    # (zero post-warmup holes, a null first_available); the REFUSAL is this check's.
    s = _summary(
        sources={
            "AAA": _sources(
                by_source={
                    "external:iv30": {"n_missing": 0, "first_available": None},
                }
            )
        }
    )
    check = _run(evaluate_gate(s))["source_coverage"]
    assert not check.met
    assert "external:iv30" in check.detail and "never available" in check.detail
    # the null is still echoed verbatim into observed — evidence and refusal, not either/or
    obs = check.observed["by_target"]["AAA"]["by_source"]["external:iv30"]
    assert obs["first_available"] is None


@pytest.mark.parametrize("bad", [7, {}, [], 1.5, True])
def test_a_garbage_first_available_refuses(bad):
    # first_available is a timestamp string or an explicit null — anything else is drifted
    # input, refused fail-closed like every other unreadable field here.
    s = _summary(
        sources={
            "AAA": _sources(
                by_source={
                    "external:iv30": {"n_missing": 0, "first_available": bad},
                }
            )
        }
    )
    check = _run(evaluate_gate(s))["source_coverage"]
    assert not check.met
    assert "neither a timestamp string nor null" in check.detail


def test_a_missing_first_available_key_refuses():
    # The runner always writes the key, so its absence is drifted input — whether the input
    # ever existed cannot be verified from a ledger that does not say.
    s = _summary(
        sources={
            "AAA": _sources(
                by_source={
                    "external:iv30": {"n_missing": 0},
                }
            )
        }
    )
    check = _run(evaluate_gate(s))["source_coverage"]
    assert not check.met
    assert "first_available is absent" in check.detail


def test_a_never_available_basket_member_source_refuses_naming_the_member():
    # Per-target in basket too: the refusal names the member whose leaf never existed, and the
    # healthy member's panel refuses nothing.
    s = _two_targets(
        sources={
            "AAA": _sources(),
            "BBB": _sources(
                by_source={
                    "external:iv30": {"n_missing": 0, "first_available": None},
                }
            ),
        }
    )
    check = _run(evaluate_gate(s))["source_coverage"]
    assert not check.met
    assert "BBB" in check.detail and "never available" in check.detail
    assert "AAA:" not in check.detail


def test_source_coverage_requires_an_entry_per_target():
    s = _two_targets(sources={"AAA": _sources()})
    check = _run(evaluate_gate(s))["source_coverage"]
    assert not check.met
    assert "BBB" in check.detail


def test_source_panel_must_span_the_run_interval():
    s = _summary(sources={"AAA": _sources(n_bars=500)})
    check = _run(evaluate_gate(s))["source_coverage"]
    assert not check.met
    assert "different interval" in check.detail


def test_source_union_cannot_exceed_the_sum_of_its_parts():
    # n_missing is a UNION over sources, so it can never outnumber the per-source total.
    s = _summary(
        sources={
            "AAA": _sources(
                n_missing=5,
                by_source={
                    "field:close": {"n_missing": 2, "first_available": "2020-01-01T00:00:00"}
                },
            )
        }
    )
    check = _run(evaluate_gate(s))["source_coverage"]
    assert not check.met
    assert "per-source total" in check.detail


@pytest.mark.parametrize(
    "bad",
    [
        {"n_bars": _N_BARS, "n_missing": _N_BARS + 1, "by_source": {}},  # union exceeds interval
        {"n_bars": _N_BARS, "n_missing": math.nan, "by_source": {}},  # uncountable
        {"n_bars": _N_BARS, "n_missing": None, "by_source": {}},  # the json_safe spelling
        {"n_bars": _N_BARS, "n_missing": -1, "by_source": {}},  # negative
        {"n_bars": math.inf, "n_missing": 0, "by_source": {}},  # non-finite denominator
        {"n_bars": _N_BARS, "n_missing": 1.5, "by_source": {}},  # non-integral
        {"n_bars": _N_BARS, "n_missing": 0, "by_source": "nope"},  # unreadable leaf map
    ],
)
def test_source_panel_uncountable_reads_refuse(bad):
    assert not _run(evaluate_gate(_summary(sources={"AAA": bad})))["source_coverage"].met


def test_source_panel_non_string_source_key_refuses():
    s = _summary(
        sources={
            "AAA": _sources(
                by_source={7: {"n_missing": 0, "first_available": None}},
            )
        }
    )
    assert not _run(evaluate_gate(s))["source_coverage"].met


@pytest.mark.parametrize("bad", [math.nan, None, -1, 1.5, _N_BARS + 1])
def test_per_source_counts_must_be_counts_within_the_interval(bad):
    s = _summary(
        sources={
            "AAA": _sources(
                by_source={
                    "field:close": {"n_missing": bad, "first_available": "2020-01-01T00:00:00"}
                },
            )
        }
    )
    assert not _run(evaluate_gate(s))["source_coverage"].met


def test_source_coverage_needs_the_runs_geometry():
    s = _summary()
    del s["n_bars"]
    check = _run(evaluate_gate(s))["source_coverage"]
    assert not check.met
    assert "n_bars" in check.detail


def test_source_coverage_has_no_knob():
    # Unconditional by construction: there is no threshold for how much of the thesis may be
    # unevaluable.
    assert "no knob" in _run(evaluate_gate(_summary()))["source_coverage"].threshold


# ---- search_cap (run level) -------------------------------------------------------------------


def test_search_cap_applies_to_every_thesis():
    check = _run(evaluate_gate(_summary()))["search_cap"]
    assert check.met


def test_search_cap_gates_the_declared_grid():
    # The declared grid is the honest search burden: combos that never fired still count, and it
    # is the ONLY multiplicity input this policy carries.
    at_cap = _summary(cells=[_cell()] * 64, n_hypotheses_attempted=64)
    report = evaluate_gate(at_cap)
    assert _run(report)["search_cap"].met  # boundary inclusive
    assert report.to_dict()["n_met"] == 64
    over = _summary(cells=[_cell()] * 65, n_hypotheses_attempted=65)
    report = evaluate_gate(over)
    check = _run(report)["search_cap"]
    assert not check.met
    assert "narrow the declared sweep" in check.detail
    assert report.to_dict()["n_met"] == 0  # a run-level breach leaves every cell unmet


def test_search_cap_missing_or_fractional_count_refuses():
    check = _run(evaluate_gate(_summary(n_hypotheses_attempted=None)))["search_cap"]
    assert not check.met
    assert not _run(evaluate_gate(_summary(n_hypotheses_attempted=1.9)))["search_cap"].met
    assert not _run(evaluate_gate(_summary(n_hypotheses_attempted=math.inf)))["search_cap"].met


def test_search_cap_not_disableable_by_construction():
    # The search budget is part of the checklist — None is rejected at construction time.
    with pytest.raises(ValidationError):
        GateThresholds(thesis_max_hypotheses=None)


def test_a_stricter_search_cap_binds():
    s = _summary(cells=[_cell()] * 9, n_hypotheses_attempted=9)
    report = evaluate_gate(s, GateThresholds(thesis_max_hypotheses=8))
    assert not _run(report)["search_cap"].met
    assert report.to_dict()["n_met"] == 0


# ---- cell_evidence (per cell: shape, ledger arithmetic, cross-panel reconciliation) ----------


def test_cell_evidence_met_on_reconciling_panels():
    check = _cell_checks(evaluate_gate(_summary()))["cell_evidence"]
    assert check.met


def test_cell_evidence_requires_dict_params():
    report = evaluate_gate(_summary(cells=[_cell(params="horizon=21")]))
    check = _cell_checks(report)["cell_evidence"]
    assert not check.met
    assert "params is not a dict" in check.detail


@pytest.mark.parametrize("panel", ["by_target", "outcome_coverage", "signal_coverage"])
def test_cell_evidence_requires_each_panel(panel):
    cell = _cell()
    del cell[panel]
    check = _cell_checks(evaluate_gate(_summary(cells=[cell])))["cell_evidence"]
    assert not check.met
    assert panel in check.detail


@pytest.mark.parametrize("panel", ["by_target", "outcome_coverage", "signal_coverage"])
def test_cell_evidence_requires_exact_target_coverage(panel):
    # A silently dropped regime target must leave this check unmet, never escape notice by absence.
    cells = [_cell(targets=("AAA", "BBB"))]
    cells[0][panel] = {k: v for k, v in cells[0][panel].items() if k == "AAA"}
    check = _cell_checks(evaluate_gate(_two_targets(cells=cells)))["cell_evidence"]
    assert not check.met
    assert "BBB" in check.detail


@pytest.mark.parametrize("panel", ["by_target", "outcome_coverage", "signal_coverage"])
def test_cell_evidence_rejects_an_unexpected_target(panel):
    cell = _cell()
    cell[panel] = {**cell[panel], "ZZZ": cell[panel]["AAA"]}
    check = _cell_checks(evaluate_gate(_summary(cells=[cell])))["cell_evidence"]
    assert not check.met
    assert "ZZZ" in check.detail


@pytest.mark.parametrize("panel", ["by_target", "outcome_coverage", "signal_coverage"])
def test_cell_evidence_rejects_non_string_panel_keys(panel):
    cell = _cell()
    cell[panel] = {7: cell[panel]["AAA"]}
    report = evaluate_gate(_summary(cells=[cell]))
    check = _cell_checks(report)["cell_evidence"]
    assert not check.met
    assert "non-string" in check.detail
    assert report.cells[0].met is False


def test_cell_evidence_ledger_exit_reasons_must_sum_to_attempted():
    cell = _cell()
    cell["outcome_coverage"]["AAA"]["exit_reasons"]["horizon"] = 40  # 40 != n_attempted 42
    check = _cell_checks(evaluate_gate(_summary(cells=[cell])))["cell_evidence"]
    assert not check.met
    assert "does not account for every firing" in check.detail


def test_cell_evidence_closed_must_equal_the_horizon_count():
    cell = _cell()
    cell["outcome_coverage"]["AAA"]["n_closed"] = 40  # horizon says 42
    check = _cell_checks(evaluate_gate(_summary(cells=[cell])))["cell_evidence"]
    assert not check.met
    assert "inconsistent ledger" in check.detail


def test_cell_evidence_rejects_unknown_exit_reasons():
    # A firing parked under an unrecognized reason is a firing no reader downstream can classify.
    cell = _cell()
    cell["outcome_coverage"]["AAA"]["exit_reasons"]["mystery"] = 0
    check = _cell_checks(evaluate_gate(_summary(cells=[cell])))["cell_evidence"]
    assert not check.met
    assert "unknown exit reason" in check.detail


def test_cell_evidence_support_panel_must_agree_with_the_coverage_ledger():
    # The internally impossible summary: one closed firing in the ledger, but the graded panel
    # claims 42. The two are reconciled against EACH OTHER, not each only against itself.
    cell = _cell()
    cell["outcome_coverage"]["AAA"] = _cov(1)
    check = _cell_checks(evaluate_gate(_summary(cells=[cell])))["cell_evidence"]
    assert not check.met
    assert "different pools" in check.detail


def test_cell_evidence_n_nonoverlap_cannot_exceed_n():
    cell = _cell()
    cell["by_target"]["AAA"]["n_nonoverlap"] = 99
    check = _cell_checks(evaluate_gate(_summary(cells=[cell])))["cell_evidence"]
    assert not check.met
    assert "n_nonoverlap=99" in check.detail


def test_cell_evidence_episode_panel_must_total_the_per_target_counts():
    # The concentration panel and the support panel must describe ONE pool.
    cell = _cell(targets=("AAA", "BBB"))
    cell["episode_stats"]["n"] = 42  # should be 84
    check = _cell_checks(evaluate_gate(_two_targets(cells=[cell])))["cell_evidence"]
    assert not check.met
    assert "per-target total" in check.detail


def test_cell_evidence_requires_an_episode_panel():
    cell = _cell()
    del cell["episode_stats"]
    check = _cell_checks(evaluate_gate(_summary(cells=[cell])))["cell_evidence"]
    assert not check.met
    assert "episode_stats" in check.detail


def test_cell_evidence_signal_ledger_must_span_the_whole_index():
    # n_bars is pure geometry (the joined index length), not a property of the data.
    cell = _cell()
    cell["signal_coverage"]["AAA"]["n_bars"] = 500
    check = _cell_checks(evaluate_gate(_summary(cells=[cell])))["cell_evidence"]
    assert not check.met
    assert "spans the whole index" in check.detail


def test_cell_evidence_needs_the_runs_geometry_to_verify_the_signal_ledger():
    s = _summary()
    del s["n_bars"]
    check = _cell_checks(evaluate_gate(s))["cell_evidence"]
    assert not check.met
    assert "no countable n_bars" in check.detail


@pytest.mark.parametrize("field", ["n_attempted", "n_closed"])
@pytest.mark.parametrize("bad", [math.nan, None, math.inf, 1.5, -1])
def test_cell_evidence_uncountable_ledger_counts_refuse(field, bad):
    cell = _cell()
    cell["outcome_coverage"]["AAA"][field] = bad
    check = _cell_checks(evaluate_gate(_summary(cells=[cell])))["cell_evidence"]
    assert not check.met
    assert "drifted input" in check.detail


@pytest.mark.parametrize("bad", [math.nan, None, math.inf, 1.5, -1])
def test_cell_evidence_uncountable_graded_counts_refuse(bad):
    # Both spellings matter: the report embeds a json_safe twin in which NaN became null.
    cell = _cell()
    cell["by_target"]["AAA"]["n"] = bad
    assert not _cell_checks(evaluate_gate(_summary(cells=[cell])))["cell_evidence"].met
    cell = _cell()
    cell["by_target"]["AAA"]["n_nonoverlap"] = bad
    assert not _cell_checks(evaluate_gate(_summary(cells=[cell])))["cell_evidence"].met
    cell = _cell()
    cell["episode_stats"]["n"] = bad
    assert not _cell_checks(evaluate_gate(_summary(cells=[cell])))["cell_evidence"].met


def test_cell_evidence_without_a_usable_regime():
    check = _cell_checks(evaluate_gate(_summary(targets=["AAA", 3])))["cell_evidence"]
    assert not check.met
    assert "target list" in check.detail


# ---- malformed cells go unmet alone, never raise -------------------------------------------------


@pytest.mark.parametrize("junk", ["not a dict", 7, None, ["by_target"]])
def test_a_malformed_cell_entry_leaves_only_itself_unmet(junk):
    report = evaluate_gate(_summary(cells=[_cell(), junk], n_hypotheses_attempted=2))
    assert report.cells[0].met is True
    assert report.cells[1].met is False
    check = _cell_checks(report, 1)["cell_evidence"]
    assert not check.met
    assert "not a dict" in check.detail
    # Even an unreadable entry gets a positional label so cells[i] stays alignable with the
    # summary panel it grades.
    assert report.cells[1].cell_id == "<unreadable cell[1]>"
    assert report.cells[1].params == {}
    assert report.to_dict()["n_met"] == 1


def test_an_empty_cell_entry_leaves_every_check_it_cannot_supply_unmet():
    report = evaluate_gate(_summary(cells=[_cell(), {}], n_hypotheses_attempted=2))
    checks = _cell_checks(report, 1)
    assert [c.name for c in report.cells[1].checks] == _CELL_NAMES  # nothing short-circuits
    assert not checks["cell_evidence"].met
    assert not checks["outcome_coverage"].met
    assert not checks["signal_coverage"].met
    assert not checks["support"].met
    assert not checks["concentration"].met
    assert report.cells[0].met is True


def test_a_declared_cell_that_never_fired_is_reported_and_leaves_support_unmet():
    # The identity's honesty invariant: a non-firing combo is on the record with explicit zero /
    # NaN evidence, and it goes unmet on that evidence rather than vanishing.
    empty = _cell(
        cell_id="horizon=5",
        params={"horizon": 5},
        n=0,
        n_nonoverlap=0,
        mean_ret=math.nan,
        top_share=math.nan,
        cluster_share=math.nan,
    )
    report = evaluate_gate(_summary(cells=[_cell(), empty], n_hypotheses_attempted=2))
    checks = _cell_checks(report, 1)
    assert checks["cell_evidence"].met  # zero everywhere still reconciles
    assert checks["outcome_coverage"].met
    assert not checks["support"].met
    assert not checks["concentration"].met
    assert report.cells[1].met is False
    assert report.cells[0].met is True


# ---- outcome_coverage (per cell, fail-closed missingness) ------------------------------------


def test_outcome_coverage_met_on_a_fully_measured_cell():
    check = _cell_checks(evaluate_gate(_summary()))["outcome_coverage"]
    assert check.met


def test_open_firings_are_structurally_allowed():
    # With no holdout there is no embargo and no tail: a forward window running past the last bar
    # is geometry every cell near the index end must exhibit. Refusing it would refuse the
    # calendar, not a data defect.
    cell = _cell()
    cell["outcome_coverage"]["AAA"] = {
        "n_attempted": 48,
        "n_closed": 42,
        "exit_reasons": {"horizon": 42, "open": 6, "no_outcome": 0, "no_benchmark": 0},
    }
    report = evaluate_gate(_summary(cells=[cell]))
    assert _cell_checks(report)["outcome_coverage"].met
    assert _cell_checks(report)["cell_evidence"].met  # n still reconciles with n_closed
    assert report.cells[0].met is True
    assert "ALLOWED" in _cell_checks(report)["outcome_coverage"].threshold


def test_no_outcome_firings_refuse():
    # A NaN hole that deletes outcomes can hide adverse results — the engine's closed-only
    # statistics silently skip such rows, so the checklist must refuse them.
    cell = _cell()
    cell["outcome_coverage"]["AAA"] = {
        "n_attempted": 45,
        "n_closed": 42,
        "exit_reasons": {"horizon": 42, "open": 0, "no_outcome": 3, "no_benchmark": 0},
    }
    check = _cell_checks(evaluate_gate(_summary(cells=[cell])))["outcome_coverage"]
    assert not check.met
    assert "no_outcome" in check.detail


def test_no_benchmark_firings_refuse():
    cell = _cell()
    cell["outcome_coverage"]["AAA"] = {
        "n_attempted": 43,
        "n_closed": 42,
        "exit_reasons": {"horizon": 42, "open": 0, "no_outcome": 0, "no_benchmark": 1},
    }
    check = _cell_checks(evaluate_gate(_summary(cells=[cell])))["outcome_coverage"]
    assert not check.met
    assert "no_benchmark" in check.detail


def test_outcome_coverage_missing_panel_fails_closed():
    cell = _cell()
    del cell["outcome_coverage"]
    check = _cell_checks(evaluate_gate(_summary(cells=[cell])))["outcome_coverage"]
    assert not check.met
    assert "outcome_coverage" in check.detail


def test_outcome_coverage_requires_exact_target_coverage():
    cell = _cell(targets=("AAA", "BBB"))
    del cell["outcome_coverage"]["BBB"]
    check = _cell_checks(evaluate_gate(_two_targets(cells=[cell])))["outcome_coverage"]
    assert not check.met
    assert "BBB" in check.detail


@pytest.mark.parametrize("bad", [math.nan, None, math.inf, 2.5, -1])
def test_outcome_coverage_uncountable_exit_reason_counts_refuse(bad):
    cell = _cell()
    cell["outcome_coverage"]["AAA"]["exit_reasons"]["no_outcome"] = bad
    check = _cell_checks(evaluate_gate(_summary(cells=[cell])))["outcome_coverage"]
    assert not check.met
    assert "drifted input" in check.detail


def test_outcome_coverage_one_unmet_target_sinks_the_cell():
    cell = _cell(targets=("AAA", "BBB"))
    cell["outcome_coverage"]["BBB"] = {
        "n_attempted": 45,
        "n_closed": 42,
        "exit_reasons": {"horizon": 42, "open": 0, "no_outcome": 3, "no_benchmark": 0},
    }
    check = _cell_checks(evaluate_gate(_two_targets(cells=[cell])))["outcome_coverage"]
    assert not check.met
    assert "BBB" in check.detail
    assert "AAA" not in check.detail


# ---- signal_coverage (per cell, the undefined-decision contract) ------------------------------


def test_signal_coverage_met_on_a_fully_decidable_cell():
    check = _cell_checks(evaluate_gate(_summary()))["signal_coverage"]
    assert check.met


def test_one_undecidable_decision_bar_refuses():
    # The outcome ledger only accounts for bars that FIRED, so a missing decision input that
    # SUPPRESSES a firing leaves no trace there — this is the only per-cell check that can see it.
    cell = _cell()
    cell["signal_coverage"]["AAA"]["n_undefined"] = 1
    report = evaluate_gate(_summary(cells=[cell]))
    checks = _cell_checks(report)
    assert not checks["signal_coverage"].met
    assert "undecidable" in checks["signal_coverage"].detail
    assert checks["outcome_coverage"].met  # …and the outcome ledger is none the wiser


def test_undefined_cannot_exceed_the_index():
    cell = _cell()
    cell["signal_coverage"]["AAA"]["n_undefined"] = _N_BARS + 1
    check = _cell_checks(evaluate_gate(_summary(cells=[cell])))["signal_coverage"]
    assert not check.met
    assert "exceeds n_bars" in check.detail


def test_signal_coverage_missing_panel_refuses_twice():
    cell = _cell()
    del cell["signal_coverage"]
    checks = _cell_checks(evaluate_gate(_summary(cells=[cell])))
    assert not checks["signal_coverage"].met
    assert not checks["cell_evidence"].met  # the panel is part of the cell's evidence contract


def test_signal_coverage_must_cover_the_regime_exactly():
    cell = _cell(targets=("AAA", "BBB"))
    del cell["signal_coverage"]["BBB"]
    check = _cell_checks(evaluate_gate(_two_targets(cells=[cell])))["signal_coverage"]
    assert not check.met
    assert "BBB" in check.detail


@pytest.mark.parametrize(
    "bad",
    [
        {"n_bars": _N_BARS, "n_undefined": math.nan},
        {"n_bars": _N_BARS, "n_undefined": None},
        {"n_bars": _N_BARS, "n_undefined": -1},
        {"n_bars": _N_BARS, "n_undefined": 1.5},
        {"n_bars": math.inf, "n_undefined": 0},
        {"n_bars": None, "n_undefined": 0},
    ],
)
def test_signal_coverage_uncountable_reads_refuse(bad):
    cell = _cell()
    cell["signal_coverage"]["AAA"] = bad
    assert not _cell_checks(evaluate_gate(_summary(cells=[cell])))["signal_coverage"].met


def test_signal_coverage_has_no_knob():
    assert "no knob" in _cell_checks(evaluate_gate(_summary()))["signal_coverage"].threshold


# ---- support (count floors + positive sign, per target, NON-inferential) ----------------------


def test_support_met_on_the_canonical_floors():
    check = _cell_checks(evaluate_gate(_summary()))["support"]
    assert check.met


def test_support_raw_n_floor():
    check = _cell_checks(evaluate_gate(_summary(cells=[_cell(n=10)])))["support"]
    assert not check.met
    assert "n=10" in check.detail


def test_support_n_nonoverlap_floor():
    check = _cell_checks(evaluate_gate(_summary(cells=[_cell(n_nonoverlap=5)])))["support"]
    assert not check.met
    assert "n_nonoverlap=5" in check.detail


def test_support_requires_a_positive_mean():
    check = _cell_checks(evaluate_gate(_summary(cells=[_cell(mean_ret=-0.01)])))["support"]
    assert not check.met
    assert "mean_ret" in check.detail
    # Exactly zero is not positive either.
    assert not _cell_checks(evaluate_gate(_summary(cells=[_cell(mean_ret=0.0)])))["support"].met


def test_support_ignores_the_uncalibrated_statistics():
    # Deliberate: no t-statistic and no p-value gates. The overlap HAC understates its SE and the
    # rotation null over-certifies under signal-aligned volatility regimes, so both ride as
    # evidence and nothing reads them.
    cell = _cell()
    cell["by_target"]["AAA"].update(t_hac=-9.0, hac_se=math.nan, rot_p=0.99)
    report = evaluate_gate(_summary(cells=[cell], pbo={"pbo": 0.99, "prob_oos_loss": 0.99}))
    assert _cell_checks(report)["support"].met
    assert report.cells[0].met is True


def test_checks_never_read_the_bootstrap_and_spacing_evidence_blocks():
    # boot / subperiods / bar_spacing are EVIDENCE-ONLY: corrupt all three arbitrarily in an
    # otherwise fully met summary and every check still comes out met — the moment one goes
    # unmet here, a check
    # has started reading evidence, which is a policy change, not a fix.
    cell = _cell()
    cell["by_target"]["AAA"].update(boot="garbage", subperiods=[{"n": math.nan}])
    report = evaluate_gate(_summary(cells=[cell], bar_spacing={"min_seconds": "not a number"}))
    assert report.cells[0].met is True
    assert all(c.met for c in report.run_checks)


def test_checks_never_read_the_baseline_and_episode_evidence_blocks():
    # baseline / episodes / per-cell conditional_buckets / feature_association are EVIDENCE-ONLY:
    # corrupt all of them arbitrarily in an otherwise fully met summary and every check still
    # comes out met — the moment one goes unmet here, a check has started reading evidence,
    # which is a policy change, not a fix.
    cell = _cell()
    cell.update(
        episodes="garbage",
        conditional_buckets=7,
        bucket_monotonicity=[],
        feature_association={"ret_5": math.nan},
    )
    report = evaluate_gate(_summary(cells=[cell], baseline="not a list"))
    assert report.cells[0].met is True
    assert all(c.met for c in report.run_checks)


def test_support_claim_language_is_not_a_significance_claim():
    check = _cell_checks(evaluate_gate(_summary()))["support"]
    assert "NOT a significance claim" in check.threshold


def test_one_target_short_of_the_floors_sinks_the_cell():
    cell = _cell(targets=("AAA", "BBB"))
    cell["by_target"]["BBB"]["mean_ret"] = -0.01
    check = _cell_checks(evaluate_gate(_two_targets(cells=[cell])))["support"]
    assert not check.met
    assert "BBB" in check.detail
    assert "AAA" not in check.detail


@pytest.mark.parametrize("bad", [math.nan, None, math.inf, 2.5])
def test_support_uncountable_counts_fail_closed(bad):
    # float(NaN) < floor is False — the classic fail-open. A NaN count must refuse, never sail
    # through;
    # 2.5 observations is drifted input, not "2".
    cell = _cell()
    cell["by_target"]["AAA"]["n"] = bad
    assert not _cell_checks(evaluate_gate(_summary(cells=[cell])))["support"].met
    cell = _cell()
    cell["by_target"]["AAA"]["n_nonoverlap"] = bad
    assert not _cell_checks(evaluate_gate(_summary(cells=[cell])))["support"].met


@pytest.mark.parametrize("bad", [math.nan, None, math.inf, -math.inf])
def test_support_unreadable_mean_fails_closed(bad):
    # mean_ret is a magnitude, not a count — any non-integral value is legitimate — but a NaN or
    # ±inf mean is unreadable, and `NaN <= 0` being False must never buy a pass.
    cell = _cell()
    cell["by_target"]["AAA"]["mean_ret"] = bad
    assert not _cell_checks(evaluate_gate(_summary(cells=[cell])))["support"].met


def test_support_missing_panel_refuses():
    cell = _cell()
    del cell["by_target"]
    check = _cell_checks(evaluate_gate(_summary(cells=[cell])))["support"]
    assert not check.met
    assert "by_target" in check.detail


def test_a_stricter_support_floor_binds():
    report = evaluate_gate(_summary(), GateThresholds(thesis_min_trades=100))
    assert not _cell_checks(report)["support"].met
    assert report.to_dict()["n_met"] == 0


# ---- concentration (per cell, universal ceiling) ----------------------------------------------


def test_concentration_applies_to_every_cell():
    check = _cell_checks(evaluate_gate(_summary()))["concentration"]
    assert check.met


def test_concentration_ceiling_refuses_a_one_episode_edge():
    check = _cell_checks(evaluate_gate(_summary(cells=[_cell(top_share=0.75)])))["concentration"]
    assert not check.met
    assert "one episode" in check.detail


def test_concentration_reads_the_cell_cluster_mass():
    check = _cell_checks(evaluate_gate(_summary(cells=[_cell(cluster_share=0.85)])))[
        "concentration"
    ]
    assert not check.met
    assert "one merged episode cluster" in check.detail


def test_concentration_reads_every_target():
    # A target riding one whale event is refused by the SAME ceiling as any other.
    cell = _cell(targets=("AAA", "BBB"))
    cell["by_target"]["BBB"]["concentration"] = {"top_share_abs": 0.9}
    check = _cell_checks(evaluate_gate(_two_targets(cells=[cell])))["concentration"]
    assert not check.met
    assert "BBB" in check.detail


def test_concentration_ceiling_not_disableable_by_construction():
    # A checklist without a one-episode detector is no checklist: the ceiling is a required,
    # bounded field — None is rejected at construction time, never a vacuous pass at grade time.
    with pytest.raises(ValidationError):
        GateThresholds(thesis_max_concentration=None)


def test_concentration_missing_reads_refuse():
    cell = _cell()
    del cell["by_target"]["AAA"]["concentration"]
    assert not _cell_checks(evaluate_gate(_summary(cells=[cell])))["concentration"].met
    cell = _cell()
    cell["episode_stats"] = {"n": 42}  # no cluster-mass read
    assert not _cell_checks(evaluate_gate(_summary(cells=[cell])))["concentration"].met


@pytest.mark.parametrize("bad", [1.5, -0.1, math.nan, None, math.inf])
def test_concentration_out_of_range_shares_refuse(bad):
    # A mass share outside [0,1] is drifted input, not a number to compare.
    assert not _cell_checks(evaluate_gate(_summary(cells=[_cell(top_share=bad)])))[
        "concentration"
    ].met
    assert not _cell_checks(evaluate_gate(_summary(cells=[_cell(cluster_share=bad)])))[
        "concentration"
    ].met


def test_concentration_boundary_is_inclusive():
    assert _cell_checks(evaluate_gate(_summary(cells=[_cell(top_share=0.6, cluster_share=0.6)])))[
        "concentration"
    ].met


def test_concentration_refuses_incommensurable_cluster_mass():
    # diff outcomes measure each target in its own level units — a cross-target mass share
    # certifies nothing, so the read refuses instead of comparing incomparables.
    s = _two_targets(outcome={"series": "target", "kind": "diff"})
    check = _cell_checks(evaluate_gate(s))["concentration"]
    assert not check.met
    assert "diff" in check.detail
    # A single-target diff run stays commensurable — no refusal from this guard.
    single = _summary(outcome={"series": "target", "kind": "diff"})
    assert _cell_checks(evaluate_gate(single))["concentration"].met


def test_stripping_the_outcome_stamp_cannot_bypass_the_commensurability_guard():
    # Reading `outcome.get("kind")` through a "missing ⇒ commensurable" default would let DELETING
    # the stamp buy a diff-outcome multi-target run a free pass on the cross-target mass read —
    # the one thing the guard exists to prevent. A missing stamp refuses instead.
    s = _two_targets()
    del s["outcome"]
    report = evaluate_gate(s)
    check = _cell_checks(report)["concentration"]
    assert not check.met
    assert "no outcome stamp" in check.detail
    # …and the missing stamp itself leaves run-level evidence unmet.
    assert not _run(report)["evidence_complete"].met


def test_unreadable_outcome_kind_refuses_rather_than_assuming_commensurable():
    s = _two_targets(outcome={"series": "target", "kind": "bogus"})
    report = evaluate_gate(s)
    assert not _cell_checks(report)["concentration"].met
    assert not _run(report)["evidence_complete"].met


def test_null_outcome_stamp_refuses_the_commensurability_read():
    # Accepting a null stamp would be the stamp-stripping bypass one spelling over — a
    # multi-target run with a null stamp cannot certify its cross-target mass reads either.
    report = evaluate_gate(_two_targets(outcome=None))
    assert not _cell_checks(report)["concentration"].met
    assert not _run(report)["evidence_complete"].met


def test_a_stricter_concentration_ceiling_binds():
    report = evaluate_gate(_summary(), GateThresholds(thesis_max_concentration=0.2))
    assert not _cell_checks(report)["concentration"].met


# ---- target-mode dispatch: the stamp selects the rubric --------------------------------------


def test_a_clean_basket_summary_meets_every_check():
    report = evaluate_gate(_basket())
    assert all(c.met for c in report.run_checks)
    assert report.cells[0].met is True
    assert all(c.met for c in report.cells[0].checks)


def test_evidence_complete_observes_the_target_mode():
    assert _run(evaluate_gate(_summary()))["evidence_complete"].observed["target_mode"] == (
        "conjunction"
    )
    assert _run(evaluate_gate(_basket()))["evidence_complete"].observed["target_mode"] == ("basket")


def test_missing_target_mode_stamp_refuses_everywhere_it_dispatches():
    # The stamp SELECTS the rubric, so stripping it must refuse in evidence_complete AND in
    # every check that dispatches on it — grading under an assumed mode would be the
    # stamp-stripping bypass one field over (the outcome-stamp precedent).
    s = _summary()
    del s["target_mode"]
    report = evaluate_gate(s)
    assert not _run(report)["evidence_complete"].met
    assert "target_mode" in _run(report)["evidence_complete"].detail
    checks = _cell_checks(report)
    assert not checks["cell_evidence"].met
    assert not checks["support"].met
    assert not checks["concentration"].met
    # The per-target coverage ledgers do not dispatch on the mode and still grade cleanly.
    assert checks["outcome_coverage"].met
    assert checks["signal_coverage"].met
    assert report.cells[0].met is False


@pytest.mark.parametrize(
    "garbage",
    ["pooled", "BASKET", "Conjunction", "", None, 7, True, {"mode": "basket"}, ["basket"]],
)
def test_garbage_target_mode_stamp_refuses(garbage):
    # Only the two exact rubric names are readable — any other spelling is an unanswerable
    # question, never a default.
    report = evaluate_gate(_summary(target_mode=garbage))
    assert not _run(report)["evidence_complete"].met
    checks = _cell_checks(report)
    assert not checks["cell_evidence"].met
    assert not checks["support"].met
    assert not checks["concentration"].met
    assert report.cells[0].met is False


def test_basket_below_two_targets_refuses_as_drifted_input():
    # Validation refuses a one-member basket upstream, so a summary carrying one can only be
    # drifted input — the gate re-refuses, never trusts.
    check = _run(evaluate_gate(_basket(targets=("AAA",))))["evidence_complete"]
    assert not check.met
    assert "basket" in check.detail
    assert "validation" in check.detail
    assert "drifted input" in check.detail


def test_basket_with_a_diff_outcome_refuses_as_drifted_input():
    s = _basket(outcome={"series": "target", "kind": "diff"})
    report = evaluate_gate(s)
    check = _run(report)["evidence_complete"]
    assert not check.met
    assert "diff" in check.detail
    assert "drifted input" in check.detail
    # …and the commensurability guard refuses the mass read too (defense-in-depth).
    assert not _cell_checks(report)["concentration"].met


def test_conjunction_with_cross_mean_benchmark_refuses_evidence_complete():
    # The mirror of the basket-side re-refusals: validation mode-gates cross_mean exactly like
    # basket+diff, so a conjunction-stamped summary claiming `benchmark: "cross_mean"` can only
    # be a RESTAMPED basket — restamping is precisely the one-field tamper that would otherwise
    # swap the pooled rubric for the per-target one it was refused under. Run-level, so it
    # leaves every cell unmet.
    report = evaluate_gate(_summary(benchmark="cross_mean"))
    check = _run(report)["evidence_complete"]
    assert not check.met
    assert "cross_mean" in check.detail
    assert "restamped basket" in check.detail
    assert report.cells[0].met is False
    assert report.to_dict()["n_met"] == 0
    # …and under a basket stamp the same rider is the legal, validated configuration.
    assert _run(evaluate_gate(_basket(benchmark="cross_mean")))["evidence_complete"].met


def test_a_starving_member_with_a_healthy_pool_meets_basket_support():
    # THE anti-weakest pin: basket floors read the POOLED panel, never per member — a thin
    # member does not sink a basket cell, because the claim is about the pool, not any name in
    # it. (Under conjunction the same numbers would fail on BBB.)
    cell = _basket_cell()
    cell["by_target"]["BBB"] = _tgt_cell(n=3, n_nonoverlap=1, mean_ret=-0.02)
    cell["outcome_coverage"]["BBB"] = _cov(3)
    cell["episode_stats"]["n"] = 45  # 42 + 3
    cell["pooled"].update(n=45, n_nonoverlap=15)
    report = evaluate_gate(_basket(cells=[cell]))
    checks = _cell_checks(report)
    assert checks["support"].met
    assert checks["cell_evidence"].met  # the panels still reconcile
    assert report.cells[0].met is True


def test_basket_support_threshold_names_the_pool_not_the_members():
    check = _cell_checks(evaluate_gate(_basket()))["support"]
    assert "ONE evidence pool" in check.threshold
    assert "never per member" in check.threshold
    assert "NOT a significance claim" in check.threshold
    assert check.observed == {"pooled": {"n": 84, "n_nonoverlap": 20, "mean_ret": 0.05}}


def test_basket_support_floors_read_the_pooled_panel():
    cell = _basket_cell(pooled=_pooled(n=84, n_nonoverlap=20, mean_ret=-0.01))
    check = _cell_checks(evaluate_gate(_basket(cells=[cell])))["support"]
    assert not check.met
    assert "pooled: mean_ret" in check.detail
    cell = _basket_cell(n=10)  # pooled.n = 20 < 30; the members' own 10s are not the read
    check = _cell_checks(evaluate_gate(_basket(cells=[cell])))["support"]
    assert not check.met
    assert "pooled: n=20" in check.detail
    cell = _basket_cell(pooled=_pooled(n_nonoverlap=5))
    check = _cell_checks(evaluate_gate(_basket(cells=[cell])))["support"]
    assert not check.met
    assert "pooled: n_nonoverlap=5" in check.detail


def test_basket_cell_without_a_pooled_panel_refuses_support_and_evidence():
    cell = _cell(targets=("AAA", "BBB"))  # a conjunction-shaped cell inside a basket run
    report = evaluate_gate(_basket(cells=[cell]))
    checks = _cell_checks(report)
    assert not checks["cell_evidence"].met
    assert "pooled" in checks["cell_evidence"].detail
    assert not checks["support"].met
    assert "pooled" in checks["support"].detail
    assert not checks["concentration"].met
    assert report.cells[0].met is False


@pytest.mark.parametrize("bad", [math.nan, math.inf, 1.5, -1, None])
@pytest.mark.parametrize("field", ["n", "n_nonoverlap"])
def test_uncountable_pooled_counts_refuse(field, bad):
    cell = _basket_cell()
    cell["pooled"][field] = bad
    checks = _cell_checks(evaluate_gate(_basket(cells=[cell])))
    assert not checks["cell_evidence"].met
    assert not checks["support"].met


@pytest.mark.parametrize("bad", [math.nan, None, math.inf, -math.inf])
def test_unreadable_pooled_mean_fails_closed(bad):
    cell = _basket_cell()
    cell["pooled"]["mean_ret"] = bad
    assert not _cell_checks(evaluate_gate(_basket(cells=[cell])))["support"].met


def test_pooled_n_must_total_the_member_counts():
    cell = _basket_cell()
    cell["pooled"]["n"] = 83  # sum of by_target.n is 84
    check = _cell_checks(evaluate_gate(_basket(cells=[cell])))["cell_evidence"]
    assert not check.met
    assert "pooled.n=83" in check.detail
    assert "different pools" in check.detail


def test_pooled_n_nonoverlap_cannot_exceed_pooled_n():
    cell = _basket_cell(pooled=_pooled(n_nonoverlap=99))
    check = _cell_checks(evaluate_gate(_basket(cells=[cell])))["cell_evidence"]
    assert not check.met
    assert "pooled.n_nonoverlap=99" in check.detail


def test_pooled_n_nonoverlap_is_bounded_by_the_bar_clock():
    # Same-bar firings across members collapse to ONE greedy episode, so pooled.n_nonoverlap can
    # never
    # outrun the index even though pooled.n legitimately can (up to n_bars × members).
    cell = _basket_cell(n=350, pooled=_pooled(n=700, n_nonoverlap=601))
    check = _cell_checks(evaluate_gate(_basket(cells=[cell])))["cell_evidence"]
    assert not check.met
    assert f"n_bars={_N_BARS}" in check.detail


def test_pooled_n_cannot_exceed_bars_times_members():
    # Nothing short-circuits: the geometric bound is reported alongside the sum mismatch.
    cell = _basket_cell(pooled=_pooled(n=1300, n_nonoverlap=20))
    check = _cell_checks(evaluate_gate(_basket(cells=[cell])))["cell_evidence"]
    assert not check.met
    assert "at most once per bar" in check.detail


def test_a_basket_cell_exactly_filling_the_bar_clock_still_reconciles():
    # Boundary inclusive: every member firing on every bar is legitimate geometry.
    cell = _basket_cell(n=_N_BARS, pooled=_pooled(n=2 * _N_BARS, n_nonoverlap=_N_BARS))
    check = _cell_checks(evaluate_gate(_basket(cells=[cell])))["cell_evidence"]
    assert check.met


def test_per_member_concentration_is_replaced_by_the_pooled_read_in_basket():
    # THE replacement pin: a member's own top_share_abs at 0.9 does not sink a basket cell whose
    # POOLED mass is spread — the per-target layer is replaced, not stacked.
    cell = _basket_cell()
    for tgt in ("AAA", "BBB"):
        cell["by_target"][tgt]["concentration"] = {
            "top_share_abs": 0.9,
            "n_top": 3,
            "top_frac": 0.05,
        }
    report = evaluate_gate(_basket(cells=[cell]))
    check = _cell_checks(report)["concentration"]
    assert check.met
    assert report.cells[0].met is True


def test_pooled_top_share_breach_leaves_basket_concentration_unmet():
    cell = _basket_cell(pooled=_pooled(top_share=0.75))
    check = _cell_checks(evaluate_gate(_basket(cells=[cell])))["concentration"]
    assert not check.met
    assert "the basket's edge is one episode" in check.detail


def test_member_mass_breach_leaves_basket_concentration_unmet():
    pooled = _pooled()
    pooled["member_share"] = {"by_target": {"AAA": 0.8, "BBB": 0.2}, "max_member_share_abs": 0.8}
    cell = _basket_cell(pooled=pooled)
    check = _cell_checks(evaluate_gate(_basket(cells=[cell])))["concentration"]
    assert not check.met
    assert "one member carries the basket's mass" in check.detail
    assert "mostly one name" in check.detail


def test_missing_member_share_refuses_basket_concentration():
    pooled = _pooled()
    del pooled["member_share"]
    cell = _basket_cell(pooled=pooled)
    check = _cell_checks(evaluate_gate(_basket(cells=[cell])))["concentration"]
    assert not check.met
    assert "a one-name basket cannot be ruled out" in check.detail


def test_cluster_share_breach_leaves_basket_concentration_unmet():
    cell = _basket_cell(cluster_share=0.85)
    check = _cell_checks(evaluate_gate(_basket(cells=[cell])))["concentration"]
    assert not check.met
    assert "one merged episode cluster" in check.detail


def test_basket_concentration_boundary_is_inclusive():
    pooled = _pooled(top_share=0.6)
    pooled["member_share"]["max_member_share_abs"] = 0.6
    cell = _basket_cell(pooled=pooled, cluster_share=0.6)
    assert _cell_checks(evaluate_gate(_basket(cells=[cell])))["concentration"].met


@pytest.mark.parametrize("bad", [1.5, -0.1, math.nan, None, math.inf])
def test_pooled_out_of_range_shares_refuse(bad):
    cell = _basket_cell(pooled=_pooled(top_share=bad))
    assert not _cell_checks(evaluate_gate(_basket(cells=[cell])))["concentration"].met
    pooled = _pooled()
    pooled["member_share"]["max_member_share_abs"] = bad
    cell = _basket_cell(pooled=pooled)
    assert not _cell_checks(evaluate_gate(_basket(cells=[cell])))["concentration"].met


def test_a_pooled_key_on_a_conjunction_cell_refuses_as_a_restamped_basket():
    # The runner writes `pooled` ONLY in basket mode — absent, not null, on conjunction cells — so
    # a conjunction-stamped cell carrying one is never stray noise, it is the SIGNATURE of a
    # restamped basket: flip the one stamp field on a refused basket run and the pooled rubric
    # (member-mass ceiling, pooled floors) silently swaps for the per-target one it was refused
    # under, which is exactly the tamper that ignoring a stray `pooled` would freeze open.
    # Refusing costs ZERO honest refusals, because no legitimate summary can exhibit the
    # configuration. The panel is refused for EXISTING, never graded: junk and healthy numbers
    # alike buy the same cell_evidence refusal, and support/concentration never read it —
    # conjunction has not quietly grown a third rubric.
    for pooled in (
        "garbage",
        _pooled(n=1, n_nonoverlap=99, mean_ret=-1.0, top_share=0.99),
        _pooled(),
    ):
        cell = _cell(targets=("AAA", "BBB"))
        cell["pooled"] = pooled
        report = evaluate_gate(_two_targets(cells=[cell]))
        checks = _cell_checks(report)
        assert not checks["cell_evidence"].met
        assert "restamped basket" in checks["cell_evidence"].detail
        assert checks["support"].met  # the pooled numbers are still read by no check
        assert checks["concentration"].met
        assert report.cells[0].met is False


def test_stripping_pooled_non_gated_fields_changes_no_result():
    # Evidence-only pin, pooled edition: everything beyond {n, n_nonoverlap, mean_ret,
    # concentration.top_share_abs, member_share.max_member_share_abs} is evidence and no check
    # reads it — member_share.by_target included (attribution, never graded).
    cell = _basket_cell()
    for k in (
        "hit_rate",
        "t_hac",
        "hac_se",
        "rot_p",
        "boot",
        "subperiods",
        "ret_quantiles",
        "worst_ret",
        "mae_quantiles",
        "mfe_quantiles",
    ):
        del cell["pooled"][k]
    del cell["pooled"]["member_share"]["by_target"]
    report = evaluate_gate(_basket(cells=[cell]))
    assert report.cells[0].met is True
    assert all(c.met for c in report.run_checks)


def test_basket_outcome_coverage_stays_per_target_naming_the_member():
    # Coverage is per-target in BOTH modes: a hole in one member corrupts every member's
    # cross-sectional reads, so the refusal names the member — pooling the ledgers would let a
    # healthy sibling launder a holed member through.
    cell = _basket_cell()
    cell["outcome_coverage"]["BBB"] = {
        "n_attempted": 45,
        "n_closed": 42,
        "exit_reasons": {"horizon": 42, "open": 0, "no_outcome": 3, "no_benchmark": 0},
    }
    check = _cell_checks(evaluate_gate(_basket(cells=[cell])))["outcome_coverage"]
    assert not check.met
    assert "BBB" in check.detail
    assert "AAA" not in check.detail


def test_basket_signal_coverage_stays_per_target_naming_the_member():
    cell = _basket_cell()
    cell["signal_coverage"]["BBB"]["n_undefined"] = 2
    check = _cell_checks(evaluate_gate(_basket(cells=[cell])))["signal_coverage"]
    assert not check.met
    assert "BBB" in check.detail
    assert "AAA" not in check.detail


def test_basket_source_coverage_stays_per_target_naming_the_member():
    s = _basket(
        sources={
            "AAA": _sources(),
            "BBB": _sources(
                n_missing=2,
                by_source={
                    "field:close": {"n_missing": 2, "first_available": "2020-01-01T00:00:00"},
                },
            ),
        }
    )
    check = _run(evaluate_gate(s))["source_coverage"]
    assert not check.met
    assert "BBB" in check.detail


# ---- threshold settings (four knobs, canonical-floor, sealed, stamped) ------------------------


def test_thresholds_snapshot_roundtrip():
    t = GateThresholds()
    snap = t.snapshot()
    assert snap == {
        "thesis_min_trades": 30,
        "thesis_min_n_nonoverlap": 8,
        "thesis_max_concentration": 0.6,
        "thesis_max_hypotheses": 64,
    }
    assert GateThresholds(**snap).snapshot() == snap


@pytest.mark.parametrize(
    "field,value",
    [
        ("thesis_min_trades", 0),
        ("thesis_min_trades", -5),
        ("thesis_min_n_nonoverlap", 1),
        ("thesis_max_concentration", 0.0),
        ("thesis_max_concentration", 1.5),
        ("thesis_max_hypotheses", 0),
    ],
)
def test_threshold_domains_are_sealed(field, value):
    # "Met" must never be manufacturable by a nonsense threshold: out-of-domain values are
    # rejected at construction (CLI: exit 3, thresholds_invalid), never accepted into a vacuous
    # checklist.
    with pytest.raises(ValidationError):
        GateThresholds(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("thesis_min_trades", 5),
        ("thesis_min_n_nonoverlap", 4),
        ("thesis_max_concentration", 0.9),
        ("thesis_max_hypotheses", 128),
    ],
)
def test_canonical_thresholds_are_the_floor(field, value):
    # The party being graded cannot bend the checklist: any value LOOSER than the canonical
    # default refuses at construction (CLI: exit 3, thresholds_invalid), so a cell reported as
    # met always means at-least-canonical rigor — and an unmet cell still carries its complete
    # statistics, so exploration loses nothing.
    with pytest.raises(ValidationError):
        GateThresholds(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("thesis_min_trades", 100),
        ("thesis_min_n_nonoverlap", 12),
        ("thesis_max_concentration", 0.4),
        ("thesis_max_hypotheses", 8),
    ],
)
def test_stricter_than_canonical_constructs(field, value):
    t = GateThresholds(**{field: value})
    assert getattr(t, field) == value
    assert t.is_canonical() is False  # exact-default stamp: stricter is still non-canonical


def test_unknown_seikan_env_var_is_a_hard_error(monkeypatch):
    # A typo'd threshold var means the caller believes a checklist ran that didn't — refuse loudly
    # instead of silently falling back to defaults (the SEIKAN_ namespace is owned).
    monkeypatch.setenv("SEIKAN_MAX_P_VALUE", "0.2")  # typo: missing THESIS_
    with pytest.raises(ValidationError):
        GateThresholds()


@pytest.mark.parametrize(
    "var",
    [
        "SEIKAN_THESIS_OOS_ALPHA",
        "SEIKAN_THESIS_MIN_OOS_N_EFF",
        "SEIKAN_GATE_PROFILE",
        "SEIKAN_THESIS_MIN_PSR",
        "SEIKAN_THESIS_MIN_N_EFF",
    ],  # last: the retired v2 knob name
)
def test_unsupported_threshold_env_vars_are_hard_errors(monkeypatch, var):
    # The known set is derived from `model_fields`, so every knob the model does not define is
    # refused for free: an environment carrying one is asking for an exam this build does not
    # run, which is exactly what must not pass silently.
    monkeypatch.setenv(var, "0.05")
    with pytest.raises(ValidationError):
        GateThresholds()


def test_thresholds_canonical_flag():
    # is_canonical feeds the CLI's identity.thresholds_canonical stamp (an identity fact, never
    # part of a checklist result; see test_cli).
    assert GateThresholds().is_canonical() is True
    assert GateThresholds(thesis_min_trades=50).is_canonical() is False
    assert GateThresholds(thesis_max_concentration=0.3).is_canonical() is False


def test_thresholds_are_frozen():
    t = GateThresholds()
    with pytest.raises(ValidationError):
        t.thesis_min_trades = 99


def test_evaluate_gate_revalidates_a_loosened_subclass():
    # The party being graded cannot bend the checklist even at the Python boundary: a subclass
    # that un-freezes itself and loosens every knob is reconstructed through the sealed
    # constructor at `evaluate_gate` entry.
    class Loosened(GateThresholds):
        model_config = SettingsConfigDict(env_prefix="SEIKAN_", extra="forbid", frozen=False)

    t = Loosened()
    object.__setattr__(t, "thesis_min_trades", 0)
    object.__setattr__(t, "thesis_max_concentration", 0.99)
    assert t.thesis_min_trades == 0  # the mutation "worked" on the object…
    with pytest.raises(ValidationError):  # …and buys nothing at the gate
        evaluate_gate(_summary(), t)


def test_revalidation_prefers_explicit_fields_over_the_environment(monkeypatch):
    # Reconstruction passes every field explicitly, so a polluted environment cannot perturb the
    # checklist that was actually stamped into the report.
    t = GateThresholds(thesis_min_trades=45)
    monkeypatch.setenv("SEIKAN_THESIS_MIN_TRADES", "30")
    report = evaluate_gate(_summary(), t)
    # 42 < the caller's stricter 45 floor — the env's laxer 30 did not take over.
    assert not _cell_checks(report)["support"].met


# ---- canonical dsl hash -----------------------------------------------------------------------

_DSL = {
    "name": "t",
    "data": {"targets": ["target"]},
    "entry": {
        "type": "threshold",
        "left": {"type": "field", "column": "close"},
        "op": "<",
        "right": {"type": "constant", "value": 100.0},
    },
    "params": {"horizon": 5},
}


def test_hash_fills_defaults_and_sorts_keys():
    explicit = {
        "params": {"horizon": 5, "direction": "longonly"},  # explicit default + reordered keys
        "entry": _DSL["entry"],
        "data": {"targets": ["target"], "external": {}},  # explicit default beside the names
        "name": "t",
    }
    assert canonical_dsl_hash(_DSL) == canonical_dsl_hash(explicit)


def test_hash_changes_with_the_rules():
    other = {**_DSL, "params": {"horizon": 10}}
    assert canonical_dsl_hash(_DSL) != canonical_dsl_hash(other)


def test_hash_outcome_omitted_and_empty_share_one_identity():
    # `outcome` is non-optional with a default factory: omitted, `{}` and the spelled-out default
    # all canonicalize to the SAME filled object — one measurement, one identity. An explicit
    # `null` is a second spelling of the same default and refuses instead of forking the hash.
    empty = {**_DSL, "params": {"horizon": 5, "outcome": {}}}
    spelled = {**_DSL, "params": {"horizon": 5, "outcome": {"series": "target", "kind": "pct"}}}
    assert canonical_dsl_hash(_DSL) == canonical_dsl_hash(empty) == canonical_dsl_hash(spelled)
    with pytest.raises(ValidationError):
        canonical_dsl_hash({**_DSL, "params": {"horizon": 5, "outcome": None}})


def test_hash_rejects_invalid_dsl():
    with pytest.raises(ValidationError):
        canonical_dsl_hash({**_DSL, "exit": {"type": "threshold"}})  # extra="forbid"


def test_hash_rejects_n_rotations_as_an_unknown_key():
    # `n_rotations` is not a DSL field — a thesis carrying it is invalid input (extra="forbid"),
    # never silently accepted under an identity that does not describe it.
    with pytest.raises(ValidationError):
        canonical_dsl_hash({**_DSL, "params": {"horizon": 5, "n_rotations": 1000}})


@pytest.mark.parametrize(
    "params",
    [
        {"horizon": 5, "oos_fraction": 0.3},
        {"horizon": 5, "selection_mode": "in_sample"},
    ],
)
def test_hash_rejects_the_holdout_knobs(params):
    # There is no split, so a thesis declaring one describes a run this build cannot perform:
    # extra="forbid" makes it exit-3 invalid input rather than a silently ignored key that would
    # hash to the same identity as an honest thesis.
    with pytest.raises(ValidationError):
        canonical_dsl_hash({**_DSL, "params": params})


@pytest.mark.parametrize(
    "data",
    [
        {"targets": ["target"], "target_column": "y10"},
        {"targets": ["target"], "external": {"iv": {"column": "iv30"}}},
    ],
    ids=["target-column", "feed-column"],
)
def test_hash_rejects_the_column_fields_bound_at_the_invocation(data):
    # WHICH column of a CSV answers a key is bound at invocation (`--column KEY=COL`), like which
    # file does — so a thesis carrying either column field describes a document this build cannot
    # hash. Refusing it is the point: under a silently-ignored key the SAME hash would stand for
    # two different exams, one of them reading a column the run never honoured.
    with pytest.raises(ValidationError):
        canonical_dsl_hash({**_DSL, "data": data})


# ---- index geometry bounds the OUTCOME side, not just the decision side -------------------


def test_outcome_ledger_counts_cannot_exceed_the_index_length():
    """A target fires at most once per bar, so no ledger count can outrun ``n_bars``.

    Without this bound the outcome side was asymmetric with the decision side (which enforces
    ``n_undefined <= n_bars`` and ``sources.n_missing <= n_bars``): a forged ledger claiming more
    firings than the index has bars reconciled internally — every count inflated together, and
    ``n_nonoverlap <= n`` only gets easier — so an internally impossible cell graded as fully met.
    """
    report = evaluate_gate(_summary(cells=[_cell(n=_N_BARS + 1)]))
    ev = _cell_checks(report)["cell_evidence"]
    assert ev.met is False
    assert f"n_bars={_N_BARS}" in ev.detail
    assert report.cells[0].met is False
    assert report.to_dict()["n_met"] == 0


def test_a_cell_exactly_filling_the_index_is_still_met():
    """The bound is ``<=``, not ``<`` — a condition firing on every bar is legitimate, so the
    check must not manufacture a refusal at the boundary."""
    report = evaluate_gate(_summary(cells=[_cell(n=_N_BARS)]))
    assert _cell_checks(report)["cell_evidence"].met is True


# ---- support / concentration grade the REGIME, not whichever panel is present --------------


def test_support_refuses_a_target_dropped_from_the_by_target_panel():
    """The loop is anchored to the DECLARED targets, not to ``by_target``: anchoring it to the
    panel would let a dropped regime target go ungraded, and ``support`` would report "every
    target carries readable support" over a regime it had not seen all of. Deleting a target must
    REFUSE (fail-closed), never quietly shrink the exam."""
    s = _two_targets()
    dropped = s["targets"][1]
    s["cells"][0]["by_target"].pop(dropped)
    support = _cell_checks(evaluate_gate(s))["support"]
    assert support.met is False
    assert dropped in support.detail


def test_concentration_refuses_a_target_dropped_from_the_by_target_panel():
    s = _two_targets()
    dropped = s["targets"][1]
    s["cells"][0]["by_target"].pop(dropped)
    conc = _cell_checks(evaluate_gate(s))["concentration"]
    assert conc.met is False
    assert dropped in conc.detail
