"""The five per-cell checks and their graded sections."""

from __future__ import annotations

from collections.abc import Mapping

from seikan.gate._model import GateCheck
from seikan.gate._read import (
    _as_dict,
    _cell_panel,
    _detail,
    _incommensurable_pool,
    _int,
    _int_ledger,
    _mode,
    _num,
    _prob,
    _str_keyed,
    _targets,
)
from seikan.settings import GateThresholds
from seikan.types import EXIT_REASONS


def _cell_panels(
    cell: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """``(by_target, outcome_coverage, signal_coverage)`` as string-keyed dicts, empty when
    unreadable — the shared reader every per-cell check uses, so a malformed panel produces the
    same refusal in each of them instead of a different exception in one."""
    return (
        _str_keyed(cell.get("by_target")) or {},
        _str_keyed(cell.get("outcome_coverage")) or {},
        _str_keyed(cell.get("signal_coverage")) or {},
    )


def _grade_outcome_ledger(cov: dict[str, object], run_bars: int | None, unmet: list[str]) -> bool:
    """Per-target exit-reason ledger arithmetic (shape only — which reasons may be NONZERO is
    ``outcome_coverage``'s semantic read). Appends into ``unmet`` in ledger order; returns False
    when any row was uncountable (the cross-panel total is then unverifiable)."""
    readable = True
    for tgt in sorted(cov):
        entry = cov.get(tgt)
        entry = _as_dict(entry)
        n_att = _int(entry.get("n_attempted"))
        n_closed = _int(entry.get("n_closed"))
        reasons = entry.get("exit_reasons")
        reasons = _as_dict(reasons)
        ledger, _bad = _int_ledger(reasons)
        unknown = sorted(k for k in reasons if k not in EXIT_REASONS)
        if unknown:
            # A firing parked under an unrecognized reason is a firing the ledger's arithmetic
            # cannot account for — and its semantics are unknown to every reader downstream.
            unmet.append(
                f"outcome_coverage[{tgt}]: unknown exit reason(s) {unknown} outside "
                f"{list(EXIT_REASONS)} — drifted input"
            )
        if n_att is None or n_att < 0 or n_closed is None or n_closed < 0 or ledger is None:
            unmet.append(
                f"outcome_coverage[{tgt}]: uncountable or negative ledger counts — drifted input"
            )
            readable = False
            continue
        if sum(ledger.values()) != n_att:
            unmet.append(
                f"outcome_coverage[{tgt}]: exit reasons sum to {sum(ledger.values())} but "
                f"n_attempted={n_att} — the ledger does not account for every firing"
            )
        if n_closed != ledger["horizon"]:
            unmet.append(
                f"outcome_coverage[{tgt}]: n_closed={n_closed} != horizon count "
                f"{ledger['horizon']} — inconsistent ledger"
            )
        # Index geometry bounds the OUTCOME side exactly as it bounds the decision side
        # (`signal_coverage`'s n_undefined <= n_bars, `sources`' n_missing <= n_bars): a target
        # fires at most once per bar, so the runner writes at most one ledger row per bar per
        # target. Without this, a ledger could claim more firings than the index has bars and
        # still reconcile internally — the counts inflate together, and `n_nonoverlap <= n` only
        # gets
        # easier — so the impossibility would sail past every relative check.
        if run_bars is not None and n_att > run_bars:
            unmet.append(
                f"outcome_coverage[{tgt}]: n_attempted={n_att} exceeds the index's "
                f"n_bars={run_bars} — a target fires at most once per bar; drifted input"
            )
    return readable


def _grade_by_target(
    by_tgt: dict[str, object],
    cov: dict[str, object],
    run_bars: int | None,
    by_target_n: dict[str, int | None],
    unmet: list[str],
) -> tuple[int, bool]:
    """Per-target support-panel counts + their reconciliation against the coverage ledger.
    Returns ``(per-target n total, readable)``."""
    total = 0
    readable = True
    for tgt in sorted(by_tgt):
        st = by_tgt.get(tgt)
        st = _as_dict(st)
        n = _int(st.get("n"))
        n_nonoverlap = _int(st.get("n_nonoverlap"))
        by_target_n[tgt] = n
        if n is None or n < 0:
            unmet.append(f"by_target[{tgt}]: n is not a count — drifted input")
            readable = False
            continue
        if run_bars is not None and n > run_bars:
            # The same geometric bound the ledger carries, applied where `support` reads. Graded
            # independently so a cell whose panels disagree still cannot claim impossible support.
            unmet.append(
                f"by_target[{tgt}]: n={n} exceeds the index's n_bars={run_bars} — a target fires "
                "at most once per bar; drifted input"
            )
        total += n
        if n_nonoverlap is None or n_nonoverlap < 0:
            unmet.append(f"by_target[{tgt}]: n_nonoverlap is not a count — drifted input")
        elif n_nonoverlap > n:
            unmet.append(
                f"by_target[{tgt}]: n_nonoverlap={n_nonoverlap} exceeds n={n} — "
                "non-overlapping windows cannot outnumber observations; drifted input"
            )
        pool = cov.get(tgt)
        pool = _as_dict(pool)
        n_closed = _int(pool.get("n_closed"))
        if n_closed is not None and n != n_closed:
            unmet.append(
                f"{tgt}: by_target n={n} but its coverage ledger closed {n_closed} firing(s) — "
                "the panels describe different pools; drifted input"
            )
    return total, readable


def _grade_signal_geometry(sig: dict[str, object], run_bars: int | None, unmet: list[str]) -> None:
    """``signal_coverage[t].n_bars`` must equal the run's — the decision ledger spans the whole
    index by construction (pure geometry, never a property of the data)."""
    for tgt in sorted(sig):
        entry = sig.get(tgt)
        entry = _as_dict(entry)
        n_bars = _int(entry.get("n_bars"))
        if n_bars is None or n_bars < 0:
            unmet.append(f"signal_coverage[{tgt}]: n_bars is not a count — drifted input")
        elif run_bars is None:
            unmet.append(
                f"signal_coverage[{tgt}]: the summary carries no countable n_bars to verify "
                f"n_bars={n_bars} against — drifted input"
            )
        elif n_bars != run_bars:
            unmet.append(
                f"signal_coverage[{tgt}]: n_bars={n_bars} != the summary's n_bars={run_bars} — "
                "the decision ledger spans the whole index by construction; drifted input"
            )


def _grade_episode_total(
    cell: dict[str, object],
    by_target_n_total: int,
    n_total_readable: bool,
    observed: dict[str, object],
    unmet: list[str],
) -> None:
    """``episode_stats.n`` must equal the per-target total — the concentration panel and the
    support panel must describe ONE pool."""
    ep = cell.get("episode_stats")
    ep = ep if isinstance(ep, dict) else None
    if ep is None:
        unmet.append(
            "cell lacks an episode_stats panel — the cross-target episode-cluster read cannot "
            "be verified; drifted input"
        )
    else:
        ep_n = _int(ep.get("n"))
        observed["episode_stats_n"] = ep_n
        if ep_n is None or ep_n < 0:
            unmet.append("episode_stats.n is not a count — drifted input")
        elif n_total_readable and ep_n != by_target_n_total:
            unmet.append(
                f"episode_stats.n={ep_n} != the per-target total {by_target_n_total} — the "
                "concentration panel and the support panel describe different pools; drifted "
                "input"
            )


def _grade_pooled(
    cell: dict[str, object],
    mode: str | None,
    targets: list[str] | None,
    run_bars: int | None,
    by_target_n_total: int,
    n_total_readable: bool,
    observed: dict[str, object],
    unmet: list[str],
) -> None:
    """The mode-dispatched pooled-panel contract: a basket cell must carry one that reconciles
    with the member panels and the bar clock; a conjunction cell carrying one is the signature
    of a restamped basket and refuses."""
    if mode == "conjunction" and "pooled" in cell:
        # The runner writes `pooled` ONLY in basket mode — absent, not null, on
        # conjunction cells — so a conjunction-stamped cell carrying one is not stray noise but
        # the signature of a RESTAMPED basket: flip the one stamp field on a refused basket run
        # and the pooled rubric (member-mass ceiling, pooled floors) silently swaps for the
        # per-target one it was refused under. Refusing here costs zero honest refusals, since
        # no legitimate summary can exhibit the configuration.
        unmet.append(
            "cell carries a pooled panel under a conjunction stamp — the "
            "runner writes pooled only in basket mode (absent, not null, on conjunction "
            "cells), so a conjunction-stamped summary carrying one is a restamped basket; "
            "drifted input"
        )
    if mode == "basket":
        # The pooled panel is what basket `support`/`concentration` grade, so a basket cell
        # without one — or one whose pooled counts contradict the member panels or the bar
        # clock — is drifted input.
        pooled = cell.get("pooled")
        if not isinstance(pooled, dict):
            unmet.append(
                f"cell lacks a pooled panel ({type(pooled).__name__}) — basket cells are "
                "graded on the pooled cross-target block, so a basket cell without one "
                "carries no gradable evidence; drifted input"
            )
        else:
            p_n = _int(pooled.get("n"))
            p_eff = _int(pooled.get("n_nonoverlap"))
            observed["pooled"] = {"n": p_n, "n_nonoverlap": p_eff}
            if p_n is None or p_n < 0:
                unmet.append("pooled.n is not a count — drifted input")
            else:
                if n_total_readable and p_n != by_target_n_total:
                    unmet.append(
                        f"pooled.n={p_n} != the per-target total {by_target_n_total} — the "
                        "pooled panel and the member panels describe different pools; drifted "
                        "input"
                    )
                if run_bars is not None and targets and p_n > run_bars * len(targets):
                    unmet.append(
                        f"pooled.n={p_n} exceeds n_bars × len(targets) = "
                        f"{run_bars * len(targets)} — each member fires at most once per bar; "
                        "drifted input"
                    )
            if p_eff is None or p_eff < 0:
                unmet.append("pooled.n_nonoverlap is not a count — drifted input")
            else:
                if p_n is not None and p_n >= 0 and p_eff > p_n:
                    unmet.append(
                        f"pooled.n_nonoverlap={p_eff} exceeds pooled.n={p_n} — "
                        "non-overlapping windows cannot outnumber observations; drifted input"
                    )
                if run_bars is not None and p_eff > run_bars:
                    unmet.append(
                        f"pooled.n_nonoverlap={p_eff} exceeds the index's n_bars={run_bars} — the "
                        "greedy non-overlapping count collapses same-bar cross-member "
                        "firings, so it is bounded by the bar clock; drifted input"
                    )


def _check_cell_evidence(cell: object, s: Mapping[str, object], t: GateThresholds) -> GateCheck:
    """The cell must carry the evidence its own checks grade, and its panels must AGREE.

    Shape: the entry is a dict with a dict ``params`` (the cell's identity — axes plus the
    horizon, always present), and its three per-target panels (``by_target``,
    ``outcome_coverage``, ``signal_coverage``) are string-keyed and cover the regime EXACTLY (a
    silently dropped target must fail here, not pass by absence).

    Arithmetic: every count is countable and non-negative, the exit reasons sum to the pool's
    ``n_attempted``, and ``n_closed == exit_reasons["horizon"]`` — a ledger anyone can re-check.

    Reconciliation: the panels describe the SAME rows, so ``support`` reading ``by_target[t].n``
    and ``outcome_coverage`` counting the closed firings that produced it must agree;
    ``n_nonoverlap <=
    n`` (non-overlapping windows cannot outnumber observations); the cell's ``episode_stats.n`` is
    the per-target total (the concentration panel and the support panel must describe one pool);
    and ``signal_coverage[t].n_bars`` is the run's ``n_bars`` (the decision ledger spans the whole
    index — pure geometry, not a property of the data). A summary claiming 42 observations over a
    ledger holding one is internally impossible, and an impossible summary is drifted input, not
    something to grade.

    Basket cells (``target_mode == "basket"``) additionally carry the ``pooled`` cross-target
    panel their own ``support``/``concentration`` reads grade, and it must reconcile with the
    member panels: ``pooled.n == sum of by_target.n`` (one pool, fully attributed),
    ``pooled.n_nonoverlap <= pooled.n``, ``pooled.n_nonoverlap <= n_bars`` (the greedy
    non-overlapping count
    collapses same-bar cross-member firings, so it is bounded by the bar clock), and
    ``pooled.n <= n_bars × len(targets)`` (each member fires at most once per bar). Conjunction
    cells are graded by the per-target rules — and a ``pooled`` key under a conjunction stamp
    REFUSES, because the runner writes pooled only in basket mode: the configuration is the
    signature of a restamped basket, and refusing it costs zero honest refusals. A missing or
    unreadable ``target_mode`` stamp refuses: whether a pooled panel is part of the cell's
    evidence contract is then undeterminable."""
    run_bars = _int(s.get("n_bars"))
    mode = _mode(s)
    threshold = (
        "cell is a dict with dict params; by_target, outcome_coverage and signal_coverage "
        "string-keyed and covering summary targets exactly; all counts countable and "
        "non-negative; per target sum(exit_reasons)==n_attempted and n_closed==exit_reasons"
        "[horizon]; panels reconcile (by_target.n==outcome_coverage.n_closed, n_nonoverlap<=n, "
        "episode_stats.n==sum of by_target.n, signal_coverage.n_bars==summary.n_bars); "
        "target_mode readable (the stamp says whether a pooled panel is part of the cell's "
        "contract); basket cells additionally carry a pooled dict that reconciles "
        "(pooled.n==sum of by_target.n, pooled.n_nonoverlap<=pooled.n, "
        "pooled.n_nonoverlap<=n_bars, "
        "pooled.n<=n_bars×len(targets)); a pooled key on a conjunction cell refuses (a "
        "restamped basket)"
    )
    unmet: list[str] = []
    by_target_n: dict[str, int | None] = {}
    observed: dict[str, object] = {
        "params": None,
        "by_target_n": by_target_n,
        "episode_stats_n": None,
    }
    if not isinstance(cell, dict):
        return GateCheck(
            name="cell_evidence",
            met=False,
            observed={"type": type(cell).__name__},
            threshold=threshold,
            detail=(
                f"cell entry is not a dict ({type(cell).__name__}) — the per-cell record is "
                "unreadable; drifted input"
            ),
        )
    params = cell.get("params")
    if not isinstance(params, dict):
        unmet.append(
            f"params is not a dict ({type(params).__name__}) — the cell does not name the "
            "parameter assignment it measured; drifted input"
        )
    else:
        observed["params"] = params
    targets = _targets(s)
    tset = set(targets) if targets else None
    by_tgt, cov, sig = _cell_panels(cell)
    if tset is None:
        unmet.append(
            "summary lacks a usable target list (non-empty, all strings) — the cell's panels "
            "cannot be verified against a regime"
        )
    if mode is None:
        unmet.append(
            "summary carries no readable target_mode stamp — whether this cell's evidence "
            "contract includes a pooled panel is undeterminable, so the cell cannot be "
            "verified against either rubric; drifted input"
        )
    for pname, raw, panel in (
        ("by_target", cell.get("by_target"), by_tgt),
        ("outcome_coverage", cell.get("outcome_coverage"), cov),
        ("signal_coverage", cell.get("signal_coverage"), sig),
    ):
        if not isinstance(raw, dict):
            unmet.append(
                f"cell lacks a {pname} panel ({type(raw).__name__}) — the cell's evidence is "
                "incomplete; drifted input"
            )
            continue
        if _str_keyed(raw) is None:
            unmet.append(
                f"{pname} is keyed by a non-string target name — it indexes no regime target; "
                "drifted input"
            )
            continue
        if tset is not None and set(panel) != tset:
            missing = sorted(tset - set(panel))
            extra = sorted(set(panel) - tset)
            unmet.append(
                f"{pname} does not cover the regime exactly (missing={missing}, unexpected={extra})"
            )
    # The five graded sections below share the unmet list, in this order — append order is
    # emitted text.
    readable_cov = _grade_outcome_ledger(cov, run_bars, unmet)
    by_target_n_total, readable_bt = _grade_by_target(by_tgt, cov, run_bars, by_target_n, unmet)
    n_total_readable = bool(cov) and bool(by_tgt) and readable_cov and readable_bt
    _grade_signal_geometry(sig, run_bars, unmet)
    _grade_episode_total(cell, by_target_n_total, n_total_readable, observed, unmet)
    _grade_pooled(
        cell, mode, targets, run_bars, by_target_n_total, n_total_readable, observed, unmet
    )
    return GateCheck(
        name="cell_evidence",
        met=not unmet,
        observed=observed,
        threshold=threshold,
        detail=(
            _detail(
                "the cell carries every panel for every target and they reconcile",
                unmet,
            )
        ),
    )


def _check_cell_outcome_coverage(
    cell: object, s: Mapping[str, object], t: GateThresholds
) -> GateCheck:
    """Fail-closed missingness contract: no unmeasured firing may sit in a graded cell.

    The engine censors a NaN outcome endpoint (``no_outcome``) or a benchmark hole
    (``no_benchmark``) and every statistic silently skips the row — which is exactly how a vendor
    outage, a stale outcome feed, or an adversarial file could delete adverse outcomes and leave a
    clean-looking cell. This reads the cell's per-target censoring ledger and refuses ANY
    ``no_outcome``/``no_benchmark`` firing (missing-at-random is never assumed).

    ``open`` is ALLOWED. With no holdout there is no embargo and no tail: a firing whose forward
    window runs past the last bar is right-censored by the data's end, which is structural
    geometry every cell near the index end must exhibit. Refusing it would refuse the calendar,
    not a data defect. What it is NOT allowed to hide is an in-bounds hole — those classify as
    ``no_outcome``/``no_benchmark`` upstream and refuse here.

    Reads are independent of every other check (no check relies on another having run); a missing
    or uncountable ledger refuses."""
    threshold = (
        "per target: exit_reasons.no_outcome == 0 and exit_reasons.no_benchmark == 0. "
        "exit_reasons.open is ALLOWED at any count — with no holdout and no embargo, a forward "
        "window running past the data end is structural right-censoring, not a data hole"
    )
    targets = _targets(s)
    tset = set(targets) if targets else None
    cov = _cell_panel(cell, "outcome_coverage")
    unmet: list[str] = []
    if tset is None:
        unmet.append(
            "summary lacks a usable target list (non-empty, all strings) — outcome coverage "
            "cannot be verified"
        )
    elif not cov:
        unmet.append(
            "cell lacks outcome_coverage — without the censoring ledger, deleted or unmeasured "
            "outcomes cannot be ruled out"
        )
    elif set(cov) != tset:
        missing = sorted(tset - set(cov))
        extra = sorted(set(cov) - tset)
        unmet.append(
            f"outcome_coverage does not cover the regime exactly "
            f"(missing={missing}, unexpected={extra})"
        )
    by_target: dict[str, dict[str, int | None]] = {}
    observed: dict[str, object] = {"by_target": by_target}
    for tgt in sorted(tset or set(cov)):
        entry = cov.get(tgt)
        entry = _as_dict(entry)
        reasons = entry.get("exit_reasons")
        reasons = _as_dict(reasons)
        by_target[tgt] = {k: _int(reasons.get(k)) for k in EXIT_REASONS}
        ledger, bad = _int_ledger(reasons)
        if ledger is None:
            unmet.append(f"{tgt}: uncountable or negative exit-reason counts {bad} — drifted input")
            continue
        if ledger["no_outcome"] > 0:
            unmet.append(
                f"{tgt}: {ledger['no_outcome']} no_outcome firing(s) — an unmeasured outcome "
                "inside a graded cell can hide adverse results (informative missingness); "
                "repair the outcome series"
            )
        if ledger["no_benchmark"] > 0:
            unmet.append(
                f"{tgt}: {ledger['no_benchmark']} no_benchmark firing(s) — a benchmark hole "
                "censors observations inside a graded cell; repair the benchmark series"
            )
    return GateCheck(
        name="outcome_coverage",
        met=not unmet,
        observed=observed,
        threshold=threshold,
        detail=(
            _detail(
                "every firing is either fully measured or right-censored at the data end",
                unmet,
            )
        ),
    )


def _check_cell_signal_coverage(
    cell: object, s: Mapping[str, object], t: GateThresholds
) -> GateCheck:
    """Fail-closed UNDEFINED-DECISION contract — the decision-side twin of ``outcome_coverage``.

    ``outcome_coverage`` can only account for bars that FIRED. A missing decision input does not
    censor an outcome; it suppresses the firing itself (a NaN operand compares False), so an
    adverse firing could be deleted by holing its inputs and NO outcome ledger would record the
    suppression — deleting data would improve the result. The engine evaluates conditions
    three-valued and reports, per target, how many post-warmup bars were UNDECIDABLE
    (``init & ~defined``); this check refuses any of them.

    This is the POOLED layer — the root condition's decidability. The raw inputs underneath it are
    graded once, run-level, by ``source_coverage``, which catches the two hole classes this
    channel cannot see: Kleene absorption (a decisive sibling settles the root while an operand is
    holed) and a NaN-skipping recursive kernel laundering state carried across a hole into a
    finite value.

    ``n_bars`` is pure geometry (the joined index length), so ``n_undefined <= n_bars`` is
    verifiable arithmetic no property of the data can bend. Unconditional: there is no threshold
    knob for how much of the thesis may be unevaluable. The ledger is keyed by COMBO upstream, so
    horizon siblings legitimately repeat the same counts — each cell is graded alone and nothing
    is summed across cells."""
    threshold = (
        "per target: n_undefined == 0 (no post-warmup undecidable decision bar) and "
        "n_undefined <= n_bars. Unconditional; no knob"
    )
    targets = _targets(s)
    tset = set(targets) if targets else None
    sig = _cell_panel(cell, "signal_coverage")
    unmet: list[str] = []
    if tset is None:
        unmet.append(
            "summary lacks a usable target list (non-empty, all strings) — signal coverage "
            "cannot be verified"
        )
    elif not sig:
        unmet.append(
            "cell lacks signal_coverage — without the undefined-decision ledger, firings "
            "suppressed by missing inputs cannot be ruled out"
        )
    elif set(sig) != tset:
        missing = sorted(tset - set(sig))
        extra = sorted(set(sig) - tset)
        unmet.append(
            f"signal_coverage does not cover the regime exactly "
            f"(missing={missing}, unexpected={extra})"
        )
    by_target: dict[str, dict[str, int | None]] = {}
    observed: dict[str, object] = {"by_target": by_target}
    for tgt in sorted(tset or set(sig)):
        entry = sig.get(tgt)
        entry = _as_dict(entry)
        n_bars = _int(entry.get("n_bars"))
        n_undef = _int(entry.get("n_undefined"))
        by_target[tgt] = {"n_bars": n_bars, "n_undefined": n_undef}
        if n_bars is None or n_bars < 0 or n_undef is None or n_undef < 0:
            unmet.append(f"{tgt}: uncountable or negative signal-coverage counts — drifted input")
            continue
        if n_undef > n_bars:
            unmet.append(f"{tgt}: n_undefined={n_undef} exceeds n_bars={n_bars} — drifted input")
        elif n_undef > 0:
            unmet.append(
                f"{tgt}: {n_undef} undecidable decision bar(s) of {n_bars} — a missing input "
                "suppresses firings before any outcome ledger can see them, so adverse firings "
                "may have been deleted; repair the decision series"
            )
    return GateCheck(
        name="signal_coverage",
        met=not unmet,
        observed=observed,
        threshold=threshold,
        detail=(
            _detail(
                "every decision bar is decidable, so no firing was suppressed by a missing input",
                unmet,
            )
        ),
    )


def _support_basket(cell: object, t: GateThresholds) -> GateCheck:
    """The basket support rubric: the three sealed floors read from the POOLED panel — the
    members form one evidence pool, and a thin member does not sink the cell."""
    threshold = (
        f"pooled: n>={t.thesis_min_trades} & n_nonoverlap>={t.thesis_min_n_nonoverlap} & "
        f"mean_ret>0 "
        "(basket: the members form ONE evidence pool — floors read the pooled panel, "
        "never per member; a descriptive floor, NOT a significance claim)"
    )
    pooled = cell.get("pooled") if isinstance(cell, dict) else None
    pooled = pooled if isinstance(pooled, dict) else None
    if pooled is None:
        return GateCheck(
            name="support",
            met=False,
            observed=None,
            threshold=threshold,
            detail=(
                "no usable pooled panel to check (pooled missing or not a dict) — "
                "basket "
                "support is a claim about the pool, and a cell without the pooled panel "
                "carries no gradable support"
            ),
        )
    n, n_nonoverlap = _int(pooled.get("n")), _int(pooled.get("n_nonoverlap"))
    mean_ret = _num(pooled.get("mean_ret"))
    p_observed: dict[str, dict[str, int | float | None]] = {
        "pooled": {"n": n, "n_nonoverlap": n_nonoverlap, "mean_ret": mean_ret}
    }
    p_unmet: list[str] = []
    if n is None or n < t.thesis_min_trades:
        p_unmet.append(f"pooled: n={n} (<{t.thesis_min_trades} observations)")
    elif n_nonoverlap is None or n_nonoverlap < t.thesis_min_n_nonoverlap:
        p_unmet.append(
            f"pooled: n_nonoverlap={n_nonoverlap} "
            f"(<{t.thesis_min_n_nonoverlap} non-overlapping episodes)"
        )
    elif mean_ret is None or mean_ret <= 0:
        p_unmet.append(f"pooled: mean_ret={mean_ret} (<=0 or missing)")
    return GateCheck(
        name="support",
        met=not p_unmet,
        observed=p_observed,
        threshold=threshold,
        detail=(_detail("the pooled panel carries readable support", p_unmet)),
    )


def _support_conjunction(cell: object, s: Mapping[str, object], t: GateThresholds) -> GateCheck:
    """The conjunction support rubric: the same floors per target, the weakest deciding —
    targets are the thesis's regime, so one failing target sinks the cell."""
    # conjunction — the per-target rubric: the weakest target decides.
    threshold = (
        f"per target: n>={t.thesis_min_trades} & n_nonoverlap>={t.thesis_min_n_nonoverlap} & "
        f"mean_ret>0 "
        "(the weakest target decides; a descriptive floor, NOT a significance claim)"
    )
    by_tgt = _cell_panel(cell, "by_target")
    if not by_tgt:
        return GateCheck(
            name="support",
            met=False,
            observed=None,
            threshold=threshold,
            detail=(
                "no usable per-target panel to check (by_target missing, empty, or keyed "
                "by a "
                "non-string target name)"
            ),
        )
    unmet: list[str] = []
    observed: dict[str, dict[str, int | float | None]] = {}
    # Grade the REGIME, not the panel that happens to be present. Anchoring the loop to
    # ``by_target`` would let a target dropped from the panel go ungraded — this check would
    # report "every target carries readable support" over a regime it never saw all of. A
    # missing target reads as an absent entry below and goes unmet on an uncountable ``n``,
    # which is the fail-closed direction: deleting evidence can only ever refuse.
    for tgt in sorted(_targets(s) or by_tgt):
        st = by_tgt.get(tgt)
        st = _as_dict(st)
        n, n_nonoverlap = _int(st.get("n")), _int(st.get("n_nonoverlap"))
        mean_ret = _num(st.get("mean_ret"))
        observed[tgt] = {"n": n, "n_nonoverlap": n_nonoverlap, "mean_ret": mean_ret}
        if n is None or n < t.thesis_min_trades:
            unmet.append(f"{tgt}: n={n} (<{t.thesis_min_trades} observations)")
        elif n_nonoverlap is None or n_nonoverlap < t.thesis_min_n_nonoverlap:
            unmet.append(
                f"{tgt}: n_nonoverlap={n_nonoverlap} "
                f"(<{t.thesis_min_n_nonoverlap} non-overlapping episodes)"
            )
        elif mean_ret is None or mean_ret <= 0:
            unmet.append(f"{tgt}: mean_ret={mean_ret} (<=0 or missing)")
    return GateCheck(
        name="support",
        met=not unmet,
        observed=observed,
        threshold=threshold,
        detail=_detail("every target carries readable support", unmet),
    )


def _check_cell_support(cell: object, s: Mapping[str, object], t: GateThresholds) -> GateCheck:
    """The cell must carry enough evidence to be worth reading, under the rubric the summary's
    ``target_mode`` stamp selects — the SAME three sealed floors either way (a raw observation
    floor, a non-overlapping-episode floor on ``n_nonoverlap`` — the greedy non-overlap count, since
    overlapping forward returns inflate the raw one — and a positive mean), read from different
    panels:

    - **conjunction** — every target individually: targets are the thesis's regime, a
      conjunction, so one failing target sinks the cell (it doesn't hold everywhere it claims
      to).
    - **basket** — the members form ONE evidence pool, so the floors read the POOLED panel
      (``pooled.{n, n_nonoverlap, mean_ret}``) and never a member's own: a thin member does not
      sink a
      basket cell, because the basket claim is about the pool, not about any name in it.
    - a missing or unreadable stamp REFUSES — grading under an assumed rubric is the
      stamp-stripping bypass one field over.

    Deliberately NOT an inferential claim. No t-statistic and no p-value gates here: the nominal
    per-cell statistics are known-uncalibrated on overlapping pools (the rotation null
    over-certifies under signal-aligned volatility regimes; the overlap HAC understates its SE)
    — and the pooled panel's reliability reads inherit the same caveats — so they ride in the
    summary as EVIDENCE and this check reads none of them. ``mean_ret > 0`` is a sign read on
    the realized sample, not a test — passing it certifies no positive expected return.

    All reads are strict: a NaN, ±inf, or non-integral count is drifted input and refuses, never
    sails past a comparison."""
    mode = _mode(s)
    if mode is None:
        return GateCheck(
            name="support",
            met=False,
            observed=None,
            threshold=(
                "target_mode selects the rubric: conjunction — per-target floors, the "
                "weakest "
                "target decides; basket — the same floors on the pooled panel"
            ),
            detail=(
                f"target_mode stamp missing or unreadable ({s.get('target_mode')!r}) — "
                f"which "
                "support rubric applies is undeterminable, and grading under an assumed "
                "mode "
                "would be the stamp-stripping bypass; drifted input"
            ),
        )
    if mode == "basket":
        return _support_basket(cell, t)
    return _support_conjunction(cell, s, t)


def _concentration_basket(cell: object, s: Mapping[str, object], max_conc: float) -> GateCheck:
    """The basket concentration rubric: the pooled top-share read replaces the per-target
    layer, the member-mass ceiling joins it (the one-name-basket detector), and the
    episode-cluster ceiling stays."""
    p_unmet: list[str] = []
    p_observed: dict[str, float | None] = {
        "pooled_top_share_abs": None,
        "max_member_share_abs": None,
        "max_cluster_share_abs": None,
    }
    pooled = cell.get("pooled") if isinstance(cell, dict) else None
    pooled = _as_dict(pooled)
    conc = pooled.get("concentration")
    conc = _as_dict(conc)
    share_p = _prob(conc.get("top_share_abs"))
    p_observed["pooled_top_share_abs"] = share_p
    if share_p is None:
        p_unmet.append(
            "pooled concentration missing or out of [0,1] — the basket's one-episode "
            "detector cannot be certified"
        )
    elif share_p > max_conc:
        p_unmet.append(f"pooled top_share_abs={share_p:.2f} — the basket's edge is one episode")
    ms = pooled.get("member_share")
    ms = _as_dict(ms)
    m_share = _prob(ms.get("max_member_share_abs"))
    p_observed["max_member_share_abs"] = m_share
    if m_share is None:
        p_unmet.append("member-mass decomposition missing — a one-name basket cannot be ruled out")
    elif m_share > max_conc:
        p_unmet.append(
            f"max_member_share_abs={m_share:.2f} — one member carries the basket's mass; "
            "the basket claim is mostly one name"
        )
    incommensurable, reason = _incommensurable_pool(s)
    if incommensurable:
        p_unmet.append(reason)
    else:
        ep = cell.get("episode_stats") if isinstance(cell, dict) else None
        ep = _as_dict(ep)
        cluster_share = _prob(ep.get("max_cluster_share_abs"))
        p_observed["max_cluster_share_abs"] = cluster_share
        if cluster_share is None:
            p_unmet.append(
                "episode_stats.max_cluster_share_abs missing or out of [0,1] — "
                "cross-target episode-cluster concentration cannot be certified"
            )
        elif cluster_share > max_conc:
            p_unmet.append(
                f"max_cluster_share_abs={cluster_share:.2f} — one merged episode cluster "
                "carries the mass"
            )
    return GateCheck(
        name="concentration",
        met=not p_unmet,
        observed=p_observed,
        threshold=max_conc,
        detail=(
            _detail(
                "return mass spread across episodes, members, and clusters",
                p_unmet,
                suffix="; widen universe/history",
            )
        ),
    )


def _concentration_conjunction(cell: object, s: Mapping[str, object], max_conc: float) -> GateCheck:
    """The conjunction concentration rubric: every regime target's top-share read plus the
    merged episode-cluster ceiling."""
    # conjunction — the per-target layer plus the episode-cluster ceiling.
    unmet: list[str] = []
    by_target: dict[str, float | None] = {}
    observed: dict[str, object] = {"by_target": by_target, "max_cluster_share_abs": None}
    by_tgt = _cell_panel(cell, "by_target")
    if not by_tgt:
        unmet.append(
            "no usable per-target panel — per-target concentration cannot be certified "
            "(missing, empty, or keyed by a non-string target name)"
        )
    # The regime decides the loop, not the panel — a target missing from ``by_target`` must be
    # refused as unverified, never skipped into an affirmative "mass is spread" detail.
    for tgt in sorted(_targets(s) or by_tgt):
        st = by_tgt.get(tgt)
        st = _as_dict(st)
        conc = st.get("concentration")
        conc = _as_dict(conc)
        share_t = _prob(conc.get("top_share_abs"))
        by_target[tgt] = share_t
        if share_t is None:
            unmet.append(f"{tgt}: per-target concentration missing or out of [0,1]")
        elif share_t > max_conc:
            unmet.append(f"{tgt}: top_share_abs={share_t:.2f} — this target's edge is one episode")
    incommensurable, reason = _incommensurable_pool(s)
    if incommensurable:
        unmet.append(reason)
    else:
        ep = cell.get("episode_stats") if isinstance(cell, dict) else None
        ep = _as_dict(ep)
        cluster_share = _prob(ep.get("max_cluster_share_abs"))
        observed["max_cluster_share_abs"] = cluster_share
        if cluster_share is None:
            unmet.append(
                "episode_stats.max_cluster_share_abs missing or out of [0,1] — cross-target "
                "episode-cluster concentration cannot be certified"
            )
        elif cluster_share > max_conc:
            unmet.append(
                f"max_cluster_share_abs={cluster_share:.2f} — one merged episode cluster carries "
                "the mass"
            )
    return GateCheck(
        name="concentration",
        met=not unmet,
        observed=observed,
        threshold=max_conc,
        detail=(
            _detail(
                "return mass spread across episodes, targets, and clusters",
                unmet,
                suffix="; widen universe/history",
            )
        ),
    )


def _check_cell_concentration(
    cell: object, s: Mapping[str, object], t: GateThresholds
) -> GateCheck:
    """One sealed ceiling, dispatched by the ``target_mode`` stamp — the one-episode and
    one-name detectors for EVERY cell. Any missing read refuses: a cell without its detectors
    has not been checked for the failure mode that matters most on overlapping pools.

    **conjunction**: two layers — the top-5% |return|-mass share
    of every regime target's pool (a non-binding target cannot ride one whale event through the
    regime claim), and ``episode_stats.max_cluster_share_abs`` — the mass share of the largest
    merged cross-target episode cluster over the cell's closed rows (a crisis smeared across
    rows AND targets is still one episode).

    **basket**: the pooled read REPLACES the per-target layer (the basket is graded as one
    pool, so ``pooled.concentration.top_share_abs`` is the top-share detector), the
    episode-cluster ceiling stays ("not one crisis"), and a THIRD read joins them:
    ``pooled.member_share.max_member_share_abs`` — the one-name-basket detector ("not one
    name"), same sealed ceiling, no new knob. A basket whose member-mass decomposition is
    absent has not been checked for the failure mode that renames a single-name bet a basket.

    A missing or unreadable ``target_mode`` stamp refuses — which layer applies is
    undeterminable. A ``diff``-outcome multi-target run refuses the cross-target mass read
    outright in BOTH modes: level units from different series are not commensurable, so a mass
    share across them certifies nothing — and a MISSING or unreadable outcome stamp refuses the
    same way, because stripping the stamp would otherwise bypass the guard (applied in basket
    as defense-in-depth even though ``evidence_complete`` already refuses basket+diff)."""
    max_conc = float(t.thesis_max_concentration)
    mode = _mode(s)
    if mode is None:
        return GateCheck(
            name="concentration",
            met=False,
            observed=None,
            threshold=max_conc,
            detail=(
                f"target_mode stamp missing or unreadable ({s.get('target_mode')!r}) — "
                f"whether "
                "the mass read is per-target (conjunction) or pooled (basket) is "
                "undeterminable; "
                "drifted input"
            ),
        )
    if mode == "basket":
        return _concentration_basket(cell, s, max_conc)
    return _concentration_conjunction(cell, s, max_conc)


_CELL_CHECKS = (
    _check_cell_evidence,
    _check_cell_outcome_coverage,
    _check_cell_signal_coverage,  # the two fail-closed data-integrity ledgers sit together
    _check_cell_support,
    _check_cell_concentration,
)
