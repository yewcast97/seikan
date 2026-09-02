"""The three run-level checks: evidence_complete, source_coverage, search_cap."""

from __future__ import annotations

from collections.abc import Mapping

from seikan.analysis.stats import STATISTICS_VERSION
from seikan.gate._model import GateCheck
from seikan.gate._read import (
    _as_dict,
    _detail,
    _int,
    _mode,
    _str_keyed,
    _targets,
)
from seikan.settings import GateThresholds
from seikan.types import OUTCOME_KINDS, TARGET_MODES


def _check_evidence_complete(s: Mapping[str, object], t: GateThresholds) -> GateCheck:
    """The summary must carry the evidence the rest of the checklist grades, under the contract
    this gate was built against: the version stamps must match (``statistics_version`` == this
    build's, ``gate_evidence_basis == "full_sample"`` — a summary from a different estimator
    revision refuses rather than being graded by the wrong rubric), the target list must be a
    non-empty list of STRINGS (a non-string name indexes no panel), the ``outcome`` stamp — the
    algebra every reported number is denominated in — must be the explicit ``{series, kind}``
    dict the runner always writes (a null stamp is stripped input and refuses), the
    ``target_mode`` stamp must be one of the two rubric names —
    "conjunction" or "basket" — because it selects the rubric every cross-target read is graded
    under (a missing or garbage stamp refuses fail-closed, and a basket stamp over fewer than
    two targets or a ``diff`` outcome refuses as drifted input: validation refuses both
    upstream, and the gate re-refuses rather than trusts), the geometry
    (``n_bars``) and the declared grid (``n_hypotheses_attempted``) must be countable and at
    least one, the per-target ``sources`` panel must be string-keyed and cover the regime
    EXACTLY, and ``cells`` must be a list holding EXACTLY the declared grid. That last one is the
    honesty invariant of this identity: the panel carries one entry per declared combo × horizon
    INCLUDING those that never fired, so a report short of ``n_hypotheses_attempted`` cells has
    silently dropped hypotheses from the record and is drifted input, not evidence."""
    targets = _targets(s)
    mode = _mode(s)
    stats_v = s.get("statistics_version")
    basis = s.get("gate_evidence_basis")
    outcome = s.get("outcome")
    n_att = _int(s.get("n_hypotheses_attempted"))
    n_bars = _int(s.get("n_bars"))
    cells = s.get("cells")
    src = _str_keyed(s.get("sources"))
    threshold = (
        f"statistics_version=={STATISTICS_VERSION}; gate_evidence_basis==full_sample; "
        "targets a non-empty list of strings; outcome an explicit {series, kind} dict with a "
        "string series and kind in {pct,log,diff} (null refuses); "
        "target_mode in {conjunction,basket} — the stamp selects the rubric, a missing or "
        "garbage stamp refuses, and basket requires >=2 targets and a non-diff outcome "
        "(refused at validation upstream; the gate re-refuses, never trusts); "
        "n_hypotheses_attempted>=1; n_bars>=1; sources string-keyed and covering targets "
        "exactly; cells a list with len(cells)==n_hypotheses_attempted (every declared cell on "
        "the record, non-firing combos included)"
    )
    unmet: list[str] = []
    if stats_v != STATISTICS_VERSION:
        unmet.append(
            f"statistics_version={stats_v!r} does not match this gate's expected "
            f"{STATISTICS_VERSION} — the summary was produced by a different estimator revision "
            "and cannot be graded by this checklist; re-run the engine"
        )
    if basis != "full_sample":
        unmet.append(
            f"gate_evidence_basis={basis!r} — every cell is graded on full-sample evidence; a "
            "summary describing any other basis was produced under a different contract"
        )
    if targets is None:
        unmet.append(
            "summary lacks a usable target list — targets must be a non-empty list of strings "
            "(a non-string name indexes no panel and cannot be verified); drifted input"
        )
    if "outcome" not in s:
        unmet.append(
            "summary carries no outcome stamp — the declared measurement algebra is what every "
            "reported claim is denominated in, so an unstamped summary cannot be graded"
        )
    elif (
        not isinstance(outcome, dict)
        or outcome.get("kind") not in OUTCOME_KINDS
        or not isinstance(outcome.get("series"), str)
        or not outcome.get("series")
    ):
        unmet.append(
            f"outcome stamp is unreadable ({s.get('outcome')!r}) — must be the explicit dict "
            "the runner always stamps ({series, kind} with a non-empty string series and kind "
            f"in {list(OUTCOME_KINDS)}); a null or partial stamp is stripped input, and "
            "both refuse; drifted input"
        )
    if mode is None:
        unmet.append(
            f"target_mode stamp missing or unreadable ({s.get('target_mode')!r} — not in "
            f"{list(TARGET_MODES)}) — the stamp SELECTS the rubric (conjunction: the weakest "
            "target decides; basket: the pooled panel is graded), so an unstamped summary "
            "cannot be graded under an assumed mode; drifted input"
        )
    elif mode == "basket":
        if targets is not None and len(targets) < 2:
            unmet.append(
                f"target_mode='basket' with {len(targets)} target(s) — a basket of one is "
                "degenerate and refused at validation; a summary carrying it is drifted input"
            )
        if isinstance(outcome, dict) and outcome.get("kind") == "diff":
            unmet.append(
                "target_mode='basket' with outcome kind='diff' — pooled level-unit returns "
                "have no common unit the engine can certify, and the combination is refused "
                "at validation; a summary carrying it is drifted input"
            )
    # Deliberately NOT collapsed into `elif mode == "conjunction" and s.get(...)` (ruff SIM102):
    # `TargetMode` holds exactly two values, so after the two arms above a type checker proves the
    # mode test always true and reports the collapsed form as a redundant expression. The arm has
    # to keep NAMING its mode all the same — an `else` would hand this conjunction-specific
    # refusal to any third mode added later, which is the ambient coupling this file exists to
    # refuse.
    elif mode == "conjunction":  # noqa: SIM102
        # The mirror of the basket-side re-refusals: validation mode-gates cross_mean exactly
        # like basket+diff, so a conjunction-stamped summary claiming it can only be a
        # restamped basket — and restamping is precisely the one-field tamper that would
        # otherwise swap the pooled rubric for the per-target one it was refused under.
        if s.get("benchmark") == "cross_mean":
            unmet.append(
                "target_mode='conjunction' with benchmark='cross_mean' — the cross-target "
                "mean couples targets in the outcome, which conjunction forbids and "
                "validation refuses; a summary carrying both is a restamped basket; "
                "drifted input"
            )
    if n_att is None:
        unmet.append(
            "summary carries no countable n_hypotheses_attempted — the declared grid size is "
            "unrecorded, so cross-cell multiplicity cannot be priced"
        )
    elif n_att < 1:
        unmet.append(
            f"n_hypotheses_attempted={n_att} — a run that declared no hypothesis measured "
            "nothing; drifted summary input"
        )
    if n_bars is None:
        unmet.append(
            "summary carries no countable n_bars — the joined index length is the denominator "
            "every coverage read is verified against; drifted input"
        )
    elif n_bars < 1:
        unmet.append(f"n_bars={n_bars} — an empty index measures nothing; drifted summary input")
    sources_raw = s.get("sources")
    if isinstance(sources_raw, dict) and src is None:
        unmet.append(
            "sources is keyed by a non-string target name — it indexes no regime target; "
            "drifted input"
        )
    if not isinstance(sources_raw, dict):
        unmet.append(
            "summary lacks a sources panel — without per-source availability, decision inputs "
            "suppressed at the source cannot be ruled out"
        )
    elif src is not None and targets and set(src) != set(targets):
        missing = sorted(set(targets) - set(src))
        extra = sorted(set(src) - set(targets))
        unmet.append(
            f"sources does not cover the regime exactly (missing={missing}, unexpected={extra})"
        )
    if not isinstance(cells, list):
        unmet.append(
            f"summary carries no cells panel (cells={type(cells).__name__}) — the per-cell "
            "record IS the report under this policy; drifted input"
        )
    elif n_att is not None and len(cells) != n_att:
        unmet.append(
            f"cells holds {len(cells)} entr(ies) but n_hypotheses_attempted={n_att} — every "
            "declared combo × horizon must be on the record, non-firing ones included; a report "
            "missing declared cells has dropped hypotheses from the search burden it declares"
        )
    observed = {
        "statistics_version": stats_v,
        "gate_evidence_basis": basis,
        "target_mode": s.get("target_mode"),
        "targets": sorted(targets) if targets else None,
        "sources_covered": sorted(src) if src else None,
        "n_hypotheses_attempted": n_att,
        "n_bars": n_bars,
        "n_cells": len(cells) if isinstance(cells, list) else None,
    }
    return GateCheck(
        name="evidence_complete",
        met=not unmet,
        observed=observed,
        threshold=threshold,
        detail=(
            _detail(
                "stamps match and the summary carries every declared cell with its "
                "geometry and "
                "source panels",
                unmet,
            )
        ),
    )


def _check_source_coverage(s: Mapping[str, object], t: GateThresholds) -> GateCheck:
    """Fail-closed availability contract over the RAW decision inputs — run-level because it is
    combo-independent: the entry tree's leaves (``Field``/``External``/``DaysSince``) either had
    data on a bar or they did not, whichever parameters read them.

    This is the layer the per-cell three-valued ``signal_coverage`` ledger structurally cannot
    see. Two ways a hole reaches a decision while the root condition reads perfectly DEFINED:

    - **Kleene absorption.** ``and``/``or`` recover a decision from a decisive child (``F∧U = F``,
      ``T∨U = T``), so a hole in one operand leaves the root fully decided. The bar is decided;
      what is missing is whether it would have decided the SAME WAY with the data present.
    - **Recursive kernels launder state.** ``ema``/``zscore_ema``/expanding aggregates/
      ``bars_since_extremum`` SKIP NaNs and carry their running state across a hole, then emit a
      finite value on the next bar. Definedness is minted at the threshold comparison from
      operand finiteness, so the contaminated value reads as perfectly decided.

    Counting availability at the source puts no operator between the hole and the count. A source
    that merely STARTS LATE is warmup, not a hole (its ``first_available`` is reported as
    evidence) — the observer had nothing to read yet, exactly as a transform's warmup window is
    not a hole. But warmup requires a start to EXIST: a leaf whose ``first_available`` is null
    never became available at all, and every decision the run took was taken without an input
    the thesis declares it reads. Such a leaf decides nothing itself and so refuses nowhere
    downstream — a decisive sibling (``T∨U = T``) settles the root over it, ``support`` grades
    the sibling's firings — which makes this branch the ONE place it can refuse (v3; it was the
    one hole size that passed). ``n_bars`` is pure geometry (the joined index length), so
    ``n_missing <= n_bars`` and "the union cannot outnumber its parts" are verifiable arithmetic
    no property of the data can bend. Unconditional: there is no threshold knob for how much of
    the thesis may be unevaluable."""
    threshold = (
        "per target: sources.n_missing == 0 — every raw decision input the entry tree reads is "
        "available on every bar of the evaluated interval after its own first available bar (a "
        "series that merely starts late is warmup) — and first_available is non-null: an input "
        "that never became available at all refuses, because warmup requires a start to exist. "
        "With sources.n_bars == summary.n_bars, every per-source count in 0..n_bars, and the "
        "union no larger than the sum of parts. Unconditional; no knob"
    )
    targets = _targets(s)
    tset = set(targets) if targets else None
    sources = _str_keyed(s.get("sources")) or {}
    run_bars = _int(s.get("n_bars"))
    unmet: list[str] = []
    if tset is None:
        unmet.append(
            "summary lacks a usable target list (non-empty, all strings) — source coverage "
            "cannot be verified"
        )
    elif not sources:
        unmet.append(
            "summary lacks a sources panel — without per-source availability, firings "
            "suppressed by a hole under the decision inputs cannot be ruled out"
        )
    if run_bars is None:
        unmet.append(
            "summary carries no countable n_bars — the geometry every availability count is "
            "verified against is unrecorded; drifted input"
        )
    by_target: dict[str, dict[str, object] | None] = {}
    observed: dict[str, object] = {"n_bars": run_bars, "by_target": by_target}
    for tgt in sorted(tset or set(sources)):
        entry = sources.get(tgt)
        entry = entry if isinstance(entry, dict) else None
        if entry is None:
            unmet.append(
                f"{tgt}: no per-source availability entry — decision inputs suppressed at the "
                "source cannot be ruled out for this target"
            )
            by_target[tgt] = None
            continue
        s_bars = _int(entry.get("n_bars"))
        s_missing = _int(entry.get("n_missing"))
        by_source = _str_keyed(entry.get("by_source"))
        per_by_source: dict[str, dict[str, object]] = {}
        per: dict[str, object] = {
            "n_bars": s_bars,
            "n_missing": s_missing,
            "by_source": per_by_source,
        }
        by_target[tgt] = per
        if by_source is None:
            unmet.append(
                f"{tgt}.sources: by_source is missing or keyed by a non-string source name — "
                "drifted input"
            )
        if s_bars is None or s_bars < 0 or s_missing is None or s_missing < 0:
            unmet.append(
                f"{tgt}.sources: uncountable or negative availability counts — drifted input"
            )
            continue
        if run_bars is not None and s_bars != run_bars:
            unmet.append(
                f"{tgt}.sources: n_bars={s_bars} != the summary's n_bars={run_bars} — the "
                "availability panel spans a different interval than the run; drifted input"
            )
        if s_missing > s_bars:
            unmet.append(
                f"{tgt}.sources: n_missing={s_missing} exceeds the interval's n_bars={s_bars} — "
                "drifted input"
            )
        per_source_total = 0
        leaves = by_source or {}
        for label in sorted(leaves):
            leaf = leaves[label]
            leaf = _as_dict(leaf)
            n_miss = _int(leaf.get("n_missing"))
            per_by_source[label] = {
                "n_missing": n_miss,
                "first_available": leaf.get("first_available"),
            }
            # The never-available branch (v3), fail-closed on garbage like every read here: a
            # missing key or a non-string non-null value is drifted input, an explicit null is
            # the runner's honest "no bar ever carried this input" — and the refusal, not the
            # ledger, is this check's to own. No timestamp is ever parsed: a string is a start.
            if "first_available" not in leaf:
                unmet.append(
                    f"{tgt}.sources[{label}]: first_available is absent — whether this input "
                    "ever existed cannot be verified; drifted input"
                )
            elif leaf.get("first_available") is None:
                unmet.append(
                    f"{tgt}: decision input {label} was never available on any bar of the "
                    "evaluated interval (first_available is null) — a never-available input "
                    "decides nothing itself, yet a decisive sibling settles the root over it "
                    "(T∨U = T) and every firing was taken without it; supply the series or "
                    "remove the leaf from the entry"
                )
            elif not isinstance(leaf.get("first_available"), str):
                unmet.append(
                    f"{tgt}.sources[{label}]: first_available is neither a timestamp string "
                    "nor null — drifted input"
                )
            if n_miss is None or n_miss < 0 or n_miss > s_bars:
                unmet.append(
                    f"{tgt}.sources[{label}]: n_missing is not a count within 0..{s_bars} — "
                    "drifted input"
                )
                continue
            per_source_total += n_miss
            if n_miss > 0:
                unmet.append(
                    f"{tgt}: decision input {label} is missing on {n_miss} of {s_bars} bar(s) "
                    "of the evaluated interval — a hole under the decision inputs suppresses "
                    "firings (and can survive a decisive sibling or a NaN-skipping transform), "
                    "so adverse firings may have been deleted; repair the series or exclude "
                    "the region with data.start / data.end"
                )
        if by_source is not None and s_missing > per_source_total:
            unmet.append(
                f"{tgt}.sources: n_missing={s_missing} exceeds the per-source total "
                f"{per_source_total} — the union cannot outnumber its parts; drifted input"
            )
    return GateCheck(
        name="source_coverage",
        met=not unmet,
        observed=observed,
        threshold=threshold,
        detail=(
            _detail(
                "every raw decision input is available across the evaluated interval "
                "(warmup "
                "excluded), "
                "so no firing was suppressed by a missing input",
                unmet,
            )
        ),
    )


def _check_search_cap(s: Mapping[str, object], t: GateThresholds) -> GateCheck:
    """Universal structural cap on the DECLARED grid (``n_hypotheses_attempted`` ≤ cap). The
    attempted count is the honest search burden — every declared combo × horizon, whether or not
    it fired — and it is the ONLY multiplicity input this policy carries: the gate grades cells
    independently and takes no cross-cell correction, so the caller prices its own selection
    against this number. The cap bounds how much single-run mining can be declared at all.
    Non-optional by construction."""
    cap = int(t.thesis_max_hypotheses)
    n_att = _int(s.get("n_hypotheses_attempted"))
    if n_att is None:
        return GateCheck(
            name="search_cap",
            met=False,
            observed=None,
            threshold=cap,
            detail=(
                "summary carries no countable n_hypotheses_attempted — the declared grid "
                "size "
                "cannot be certified"
            ),
        )
    ok = n_att <= cap
    return GateCheck(
        name="search_cap",
        met=ok,
        observed=n_att,
        threshold=cap,
        detail=(
            "declared search grid within the cap"
            if ok
            else f"n_hypotheses_attempted={n_att} exceeds the search cap {cap} — narrow the "
            "declared sweep (the cap is structural: non-firing combos cannot shrink it)"
        ),
    )


_RUN_CHECKS = (
    _check_evidence_complete,
    _check_source_coverage,
    _check_search_cap,
)
