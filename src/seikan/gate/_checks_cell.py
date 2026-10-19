"""The five per-cell checks and their graded sections."""

from __future__ import annotations

import math
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
from seikan.types import EXIT_REASONS, top_share_count

#: How far a basket's member shares may sum from 1. Each share is the correctly rounded exact
#: quotient of its member's |return| mass over the pool's — quotients that sum to exactly 1 — and
#: those roundings enter the sum weighted by the shares themselves, so the shares' exact sum is
#: within 2**-53 of 1 whatever the member count; ``math.fsum`` rounds it once more. 2**-50 holds
#: all of it.
_SHARE_SUM_TOLERANCE = 2.0**-50


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


def _grade_outcome_ledger(
    cov: dict[str, object],
    run_bars: int | None,
    closable: int | None,
    openable: int | None,
    unmet: list[str],
) -> bool:
    """Per-target exit-reason ledger arithmetic (shape only — which reasons may be NONZERO is
    ``outcome_coverage``'s semantic read) and the index geometry that bounds it: a target fires
    at most once per bar and a firing on the final bar has no next open to fill at, so the ledger
    holds at most ``n_bars − 1`` firings, of which at most ``closable`` close inside the index and
    at most ``openable`` — the last ``horizon`` anchor bars — run past its end. Appends into
    ``unmet`` in ledger order; returns False when any row was uncountable (the cross-panel total
    is then unverifiable)."""
    readable = True
    for tgt in sorted(cov):
        entry = cov.get(tgt)
        entry = _as_dict(entry)
        n_att = _int(entry.get("n_attempted"))
        n_closed = _int(entry.get("n_closed"))
        reasons = entry.get("exit_reasons")
        reasons = _as_dict(reasons)
        ledger, _bad = _int_ledger(reasons)
        unknown = sorted((k for k in reasons if k not in EXIT_REASONS), key=str)
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
        # (`signal_coverage`'s n_undefined <= n_bars, `sources`' n_missing <= n_bars). Without
        # it a ledger could claim more firings than the index can anchor, or more closed ones
        # than its horizon closes, and still reconcile internally — the counts inflate together,
        # and `n_nonoverlap <= n` only gets easier — so the impossibility would sail past every
        # relative check: 30 closed observations of horizon 1 over 31 bars, where 29 close, met
        # the support floor.
        if run_bars is not None and n_att > run_bars - 1:
            unmet.append(
                f"outcome_coverage[{tgt}]: n_attempted={n_att} exceeds {run_bars - 1}, the "
                f"firings an index of n_bars={run_bars} can anchor — a target fires at most once "
                "per bar, and the final bar has no next open to fill at; drifted input"
            )
        if closable is not None and n_closed > closable:
            unmet.append(
                f"outcome_coverage[{tgt}]: n_closed={n_closed} exceeds {closable}, the most "
                "observations of the cell's horizon that close inside the index (a firing at bar "
                "t closes at bar t + 1 + horizon); drifted input"
            )
        if openable is not None and ledger["open"] > openable:
            # The complement: only the last `horizon` anchor bars run past the index end, so a
            # ledger parking more firings under `open` — the one reason outcome_coverage allows
            # — relabels deleted outcomes as the calendar: 100 open of horizon 21 over 600 bars
            # met every check where 21 can be.
            unmet.append(
                f"outcome_coverage[{tgt}]: open={ledger['open']} exceeds {openable}, the most "
                "firings whose window of the cell's horizon runs past the index end (a firing is "
                "right-censored only from bar n_bars − 1 − horizon on); drifted input"
            )
    return readable


def _grade_top_share(panel: dict[str, object], n: int | None, label: str, unmet: list[str]) -> None:
    """A panel's ``concentration.top_share_abs`` must be at least its own count's share of the
    pool it is read off: the largest ``k`` of ``n`` magnitudes carry at least ``k / n`` of their
    total mass, ``k`` the emission's own top count (:func:`top_share_count`). Exact: the share is
    the correctly rounded exact quotient of exact masses, ``k / n`` the correctly rounded
    quotient of its integers, and rounding is monotone. The review's tamper: 0.0 beside 100
    observations met the ceiling. A null share (a massless or overflowed pool) is
    ``concentration``'s to refuse."""
    share = _prob(_as_dict(panel.get("concentration")).get("top_share_abs"))
    if share is None or n is None or n < 1:
        return
    k = top_share_count(n)
    if share < k / n:
        unmet.append(
            f"{label}: concentration.top_share_abs={share} is below {k}/{n}={k / n}, the least "
            f"share the largest {k} of {n} observations can carry; drifted input"
        )


def _grade_mean_range(panel: dict[str, object], label: str, unmet: list[str]) -> None:
    """A panel's ``mean_ret`` must lie within its pool's extrema, ``[worst_ret, best_ret]``. Exact:
    the mean is the pool's exact sum over its count, rounded once, the extrema are the same
    pool's rows, and rounding is monotone — on a ``cross_mean`` pool too, whose rows are their
    exact excesses rounded once. The review's tamper: a mean of 0.02 beside a range of
    [−0.2, −0.01] met every check. A null mean is ``support``'s to refuse."""
    mean = _num(panel.get("mean_ret"))
    if mean is None:
        return
    worst, best = _num(panel.get("worst_ret")), _num(panel.get("best_ret"))
    if worst is None or best is None or not worst <= mean <= best:
        unmet.append(
            f"{label}: mean_ret={mean} lies outside [worst_ret, best_ret] = [{worst}, {best}] — "
            "a mean is its pool's exact sum over its count, rounded once, which never leaves the "
            "pool's range; drifted input"
        )


def _grade_by_target(
    by_tgt: dict[str, object],
    cov: dict[str, object],
    closable: int | None,
    windows: int | None,
    by_target_n: dict[str, int | None],
    unmet: list[str],
) -> tuple[int, bool, list[int] | None]:
    """Per-target support-panel counts + their reconciliation against the coverage ledger, the
    index geometry — the most observations of the cell's horizon that close inside the index
    (``closable``) and the most non-overlapping windows among them (``windows``) — and the
    panel's own top-share floor and mean range (:func:`_grade_top_share`,
    :func:`_grade_mean_range`). Returns ``(per-target n total,
    readable, the members' n_nonoverlap counts)``, the counts ``None`` where any member's is
    unreadable."""
    total = 0
    readable = True
    member_windows: list[int] | None = []
    for tgt in sorted(by_tgt):
        st = by_tgt.get(tgt)
        st = _as_dict(st)
        n = _int(st.get("n"))
        n_nonoverlap = _int(st.get("n_nonoverlap"))
        by_target_n[tgt] = n
        if n_nonoverlap is None or n_nonoverlap < 0:
            member_windows = None
        elif member_windows is not None:
            member_windows.append(n_nonoverlap)
        if n is None or n < 0:
            unmet.append(f"by_target[{tgt}]: n is not a count — drifted input")
            readable = False
            continue
        if closable is not None and n > closable:
            # The same geometric bound the ledger carries, applied where `support` reads. Graded
            # independently so a cell whose panels disagree still cannot claim impossible support.
            unmet.append(
                f"by_target[{tgt}]: n={n} exceeds {closable}, the most observations of the "
                "cell's horizon that close inside the index; drifted input"
            )
        total += n
        _grade_top_share(st, n, f"by_target[{tgt}]", unmet)
        _grade_mean_range(st, f"by_target[{tgt}]", unmet)
        if n_nonoverlap is None or n_nonoverlap < 0:
            unmet.append(f"by_target[{tgt}]: n_nonoverlap is not a count — drifted input")
        elif n_nonoverlap > n:
            unmet.append(
                f"by_target[{tgt}]: n_nonoverlap={n_nonoverlap} exceeds n={n} — "
                "non-overlapping windows cannot outnumber observations; drifted input"
            )
        elif windows is not None and n_nonoverlap > windows:
            unmet.append(
                f"by_target[{tgt}]: n_nonoverlap={n_nonoverlap} exceeds {windows}, the most "
                "non-overlapping windows of the cell's horizon that close inside the index; "
                "drifted input"
            )
        pool = cov.get(tgt)
        pool = _as_dict(pool)
        n_closed = _int(pool.get("n_closed"))
        if n_closed is not None and n != n_closed:
            unmet.append(
                f"{tgt}: by_target n={n} but its coverage ledger closed {n_closed} firing(s) — "
                "the panels describe different pools; drifted input"
            )
    return total, readable, member_windows


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


def _grade_episode_stats(
    cell: dict[str, object],
    by_target_n_total: int,
    n_total_readable: bool,
    observed: dict[str, object],
    unmet: list[str],
) -> None:
    """``episode_stats.n`` must equal the per-target total — the concentration panel and the
    support panel must describe ONE pool — ``n_clusters`` must be a count the pool can hold (at
    most ``n``, and 0 exactly on an empty pool: every closed row lies in one cluster), and the
    ceiling read ``max_cluster_share_abs``, the largest |return|-mass share held by any cluster,
    must agree with its own panel: it is at least ``largest_cluster_share_abs`` (the
    largest-BY-COUNT cluster's share, one of the shares it is the maximum of) and at least
    ``1 / n_clusters`` (the largest of ``n_clusters`` shares of one mass is at least their mean —
    exactly 1 with one cluster). Both rules are exact: every share is the correctly rounded exact
    quotient of exact masses, as ``1 / n_clusters`` is of its integers, and rounding is monotone
    — the quotient of two separately rounded masses was not (0.33333333333333326 for a third,
    which this floor would have refused off an honest pool). The review's tampers: one cluster
    at ``largest_cluster_share_abs`` 1.0 beside a ``max_cluster_share_abs`` of 0.5, and two
    clusters at 0.30 each where the other holds 0.70. A null maximum (a pool with no mass, an
    overflowed one, incommensurable units) is ``concentration``'s to refuse; the ``episodes``
    ledger, evidence-only and truncated at its cap, is not read."""
    ep = cell.get("episode_stats")
    if not isinstance(ep, dict):
        unmet.append(
            "cell lacks an episode_stats panel — the cross-target episode-cluster read cannot "
            "be verified; drifted input"
        )
        return
    ep_n = _int(ep.get("n"))
    observed["episode_stats_n"] = ep_n
    if ep_n is None or ep_n < 0:
        unmet.append("episode_stats.n is not a count — drifted input")
    elif n_total_readable and ep_n != by_target_n_total:
        unmet.append(
            f"episode_stats.n={ep_n} != the per-target total {by_target_n_total} — the "
            "concentration panel and the support panel describe different pools; drifted input"
        )
    n_clusters = _int(ep.get("n_clusters"))
    if n_clusters is None or n_clusters < 0:
        unmet.append(
            f"episode_stats.n_clusters={ep.get('n_clusters')!r} is not a count — drifted input"
        )
    elif ep_n is not None and ep_n >= 0 and (n_clusters > ep_n or (n_clusters == 0) != (ep_n == 0)):
        unmet.append(
            f"episode_stats.n_clusters={n_clusters} beside n={ep_n} — every closed row lies in "
            "exactly one cluster, so a pool of n rows holds between 1 and n clusters (none when "
            "empty); drifted input"
        )
    top = _prob(ep.get("max_cluster_share_abs"))
    if top is None:
        return
    largest = _prob(ep.get("largest_cluster_share_abs"))
    if largest is not None and largest > top:
        unmet.append(
            f"episode_stats.max_cluster_share_abs={top} is below largest_cluster_share_abs="
            f"{largest} — the largest share held by any cluster cannot be below the "
            "largest-by-count cluster's; drifted input"
        )
    if n_clusters is not None and n_clusters >= 1 and top < 1.0 / n_clusters:
        unmet.append(
            f"episode_stats.max_cluster_share_abs={top} is below 1/n_clusters={1.0 / n_clusters} "
            f"— the largest of {n_clusters} share(s) of one mass is at least their mean (exactly "
            "1 with one cluster); drifted input"
        )


def _grade_pooled(
    cell: dict[str, object],
    mode: str | None,
    targets: list[str] | None,
    closable: int | None,
    windows: int | None,
    by_target_n_total: int,
    n_total_readable: bool,
    member_windows: list[int] | None,
    observed: dict[str, object],
    unmet: list[str],
) -> None:
    """The mode-dispatched pooled-panel contract: a basket cell must carry one that reconciles
    with the member panels — their total ``n``, and their ``n_nonoverlap`` counts
    (``member_windows``), between whose largest and whose total the pooled count lies — and the
    bar clock (``closable``, the most observations of the cell's horizon that close inside the
    index per member; ``windows``, the most non-overlapping windows among them); a conjunction
    cell carrying one is the signature of a restamped basket and refuses."""
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
                if closable is not None and targets and p_n > closable * len(targets):
                    unmet.append(
                        f"pooled.n={p_n} exceeds len(targets) × {closable} = "
                        f"{closable * len(targets)}, the most observations of the cell's horizon "
                        "that close inside the index across the members — each fires at most "
                        "once per bar; drifted input"
                    )
                _grade_top_share(pooled, p_n, "pooled", unmet)
                _grade_mean_range(pooled, "pooled", unmet)
            if p_eff is None or p_eff < 0:
                unmet.append("pooled.n_nonoverlap is not a count — drifted input")
            else:
                if p_n is not None and p_n >= 0 and p_eff > p_n:
                    unmet.append(
                        f"pooled.n_nonoverlap={p_eff} exceeds pooled.n={p_n} — "
                        "non-overlapping windows cannot outnumber observations; drifted input"
                    )
                if windows is not None and p_eff > windows:
                    unmet.append(
                        f"pooled.n_nonoverlap={p_eff} exceeds {windows}, the most non-overlapping "
                        "windows of the cell's horizon that close inside the index — the greedy "
                        "count collapses same-bar cross-member firings, so the bar clock bounds "
                        "it; drifted input"
                    )
                # Greedy earliest-start on equal windows is the largest disjoint set, so the
                # pooled count lies between the members': a member's disjoint windows are a
                # disjoint selection of the pool's, and a disjoint pooled selection splits into
                # disjoint selections within its members. The review's 8 beside members at 1 and
                # 1 — two pooled — met the support floor.
                if member_windows and p_eff < max(member_windows):
                    unmet.append(
                        f"pooled.n_nonoverlap={p_eff} is below {max(member_windows)}, the largest "
                        "member's own count — a member's disjoint windows are a disjoint "
                        "selection of the pool's; drifted input"
                    )
                if member_windows and p_eff > sum(member_windows):
                    unmet.append(
                        f"pooled.n_nonoverlap={p_eff} exceeds {sum(member_windows)}, the members' "
                        "total — a disjoint pooled selection splits into disjoint selections "
                        "within its members, and the greedy count on equal windows is the "
                        "largest; drifted input"
                    )
            member_share = pooled.get("member_share")
            if isinstance(member_share, dict):
                _grade_member_share(member_share, targets, unmet)


def _grade_member_share(
    member_share: dict[str, object], targets: list[str] | None, unmet: list[str]
) -> None:
    """``pooled.member_share``'s one-name read must be the maximum of the decomposition it is
    read off, and the decomposition must conserve the mass it attributes: ``by_target`` is
    string-keyed over the targets exactly, and where ``max_member_share_abs`` is a share, every
    member's share is one, their maximum EQUALS it — the runner writes ``max(by_target.values())``
    — and they sum to 1 within rounding (:data:`_SHARE_SUM_TOLERANCE`; the runner divides each
    member's exact |return| mass by the pool's). 0.5 beside ``{AAA: 0.5, BBB: 0.0}`` agreed with
    its maximum and met every check with half the pool's mass gone from its own breakdown. A
    missing panel or a null maximum (a pool with no mass) is ``concentration``'s to refuse."""
    shares = _str_keyed(member_share.get("by_target"))
    if shares is None:
        unmet.append(
            "pooled.member_share.by_target is not a string-keyed dict — the decomposition the "
            "one-name read is taken from is missing; drifted input"
        )
        return
    if targets is not None and set(shares) != set(targets):
        unmet.append(
            "pooled.member_share.by_target does not cover the regime exactly "
            f"(missing={sorted(set(targets) - set(shares))}, "
            f"unexpected={sorted(set(shares) - set(targets))})"
        )
        return
    top = _prob(member_share.get("max_member_share_abs"))
    if top is None:
        return
    readable = [v for v in map(_prob, shares.values()) if v is not None]
    if not readable or len(readable) != len(shares):
        unmet.append(
            f"pooled.member_share.by_target holds a value that is no share in [0,1] beside "
            f"max_member_share_abs={top} — drifted input"
        )
        return
    if max(readable) != top:
        unmet.append(
            f"pooled.member_share.max_member_share_abs={top} is not the largest by_target share "
            f"({max(readable)}) — the one-name read and its decomposition disagree; drifted input"
        )
    if abs(math.fsum(readable) - 1.0) > _SHARE_SUM_TOLERANCE:
        unmet.append(
            f"pooled.member_share.by_target shares sum to {math.fsum(readable)} — the "
            "decomposition does not conserve the pool's mass (each share is one rounded quotient "
            "of exact masses, so they sum to 1 within rounding); drifted input"
        )


def _check_cell_evidence(cell: object, s: Mapping[str, object], t: GateThresholds) -> GateCheck:
    """The cell must carry the evidence its own checks grade, and its panels must AGREE.

    Shape: the entry is a dict with a dict ``params`` (the cell's identity — axes plus the
    horizon, always present) naming a countable ``horizon >= 1`` (a cell that does not name the
    window it measured is not reproducible), and its three per-target panels (``by_target``,
    ``outcome_coverage``, ``signal_coverage``) are string-keyed and cover the regime EXACTLY (a
    silently dropped target must fail here, not pass by absence).

    Arithmetic: every count is countable and non-negative, the exit reasons sum to the pool's
    ``n_attempted``, and ``n_closed == exit_reasons["horizon"]`` — a ledger anyone can re-check —
    and the index's geometry holds it: a firing fills at the next bar's open and closes
    ``horizon`` bars later, so at most ``n_bars − 1`` firings anchor an observation (the final
    bar has no next open) and at most ``max(0, n_bars − 1 − horizon)`` of them close inside the
    index, while at most ``min(horizon, n_bars − 1)`` of them run past its end (``open``, the
    last ``horizon`` anchor bars). Bounded by ``n_bars`` alone, 30 closed observations of
    horizon 1 over 31 bars — where 29 close, and the run read 29 closed and one open —
    reconciled everywhere and met the 30-observation support floor; with the closed count bounded
    alone, 100 open observations of horizon 21 over 600 bars, beside 42 closed, met every check
    where 21 can be open — the same 100 under ``no_outcome`` refuse.

    Reconciliation: the panels describe the SAME rows, so ``support`` reading ``by_target[t].n``
    and ``outcome_coverage`` counting the closed firings that produced it must agree (and that
    ``n`` is held to the same closed ceiling); ``n_nonoverlap <= n`` (non-overlapping windows
    cannot outnumber observations) and ``n_nonoverlap <= max(0, (n_bars − 2) // horizon)`` — the
    greedy count's windows sit ``horizon`` bars apart among the closed observations: 40 windows
    of 21 bars in a 600-bar index claimed more than the 28 that fit; each panel's
    ``concentration.top_share_abs`` is at least ``k / n`` for its own ``n`` and the emission's
    top count ``k = max(1, ⌈n / 20⌉)`` (:func:`_grade_top_share` — the largest ``k`` of ``n``
    observations carry at least ``k / n`` of the mass; 0.0 beside 100 observations met the
    ceiling); each panel's ``mean_ret`` lies within its ``[worst_ret, best_ret]``
    (:func:`_grade_mean_range` — a mean is its pool's exact sum over its count, rounded once;
    0.02 beside a range of [−0.2, −0.01] met every check); the cell's
    ``episode_stats.n`` is the per-target total (the concentration panel and the support panel
    must describe one pool), its ``n_clusters`` a count the pool can hold, and its
    ``max_cluster_share_abs`` agrees with its own panel — at least ``largest_cluster_share_abs``
    and at least ``1 / n_clusters`` (:func:`_grade_episode_stats`); and
    ``signal_coverage[t].n_bars`` is the run's ``n_bars`` (the
    decision ledger spans the whole index — pure geometry, not a property of the data). A summary
    claiming 42 observations over a ledger holding one is internally impossible, and an
    impossible summary is drifted input, not something to grade.

    Basket cells (``target_mode == "basket"``) additionally carry the ``pooled`` cross-target
    panel their own ``support``/``concentration`` reads grade, and it must reconcile with the
    member panels: ``pooled.n == sum of by_target.n`` (one pool, fully attributed),
    ``pooled.n_nonoverlap <= pooled.n``, ``pooled.n_nonoverlap <= max(0, (n_bars − 2) //
    horizon)`` (the greedy non-overlapping count collapses same-bar cross-member firings, so the
    bar clock's windows bound it as they bound each target's), ``pooled.n_nonoverlap`` between
    the largest member's ``n_nonoverlap`` and the members' total (greedy earliest-start on equal
    windows is the largest disjoint set: a member's disjoint windows are a disjoint selection of
    the pool's, and a disjoint pooled selection splits into disjoint selections within its
    members — the review's 8 beside members at 1 and 1, two pooled, met the support floor),
    ``pooled.n <= len(targets) ×
    max(0, n_bars − 1 − horizon)`` (each member fires at most once per bar and closes at most
    that many), ``pooled.concentration.top_share_abs`` at least ``k / pooled.n`` (the same
    floor, over the pool the basket rubric grades), ``pooled.mean_ret`` within its pool's
    extrema (the same range), and a ``member_share`` panel's
    one-name read is the largest share of its ``by_target`` decomposition, string-keyed over the
    targets exactly and summing to 1 within rounding (:func:`_grade_member_share` —
    ``max_member_share_abs`` 0.5 beside shares ``{AAA: 1.0, BBB: 0.0}`` met every check, the
    ceiling grading a number its own breakdown contradicts, and so did 0.5 beside
    ``{AAA: 0.5, BBB: 0.0}``, half the mass gone from the breakdown). Conjunction
    cells are graded by the per-target rules — and a ``pooled`` key under a conjunction stamp
    REFUSES, because the runner writes pooled only in basket mode: the configuration is the
    signature of a restamped basket, and refusing it costs zero honest refusals. A missing or
    unreadable ``target_mode`` stamp refuses: whether a pooled panel is part of the cell's
    evidence contract is then undeterminable."""
    run_bars = _int(s.get("n_bars"))
    mode = _mode(s)
    threshold = (
        "cell is a dict with dict params naming a countable horizon>=1; by_target, "
        "outcome_coverage and signal_coverage string-keyed and covering summary targets exactly; "
        "all counts countable and non-negative; per target sum(exit_reasons)==n_attempted, "
        "n_closed==exit_reasons[horizon], n_attempted<=n_bars-1, "
        "n_closed<=max(0,n_bars-1-horizon) and exit_reasons.open<=min(horizon,n_bars-1) (a "
        "firing fills at the next bar's open and closes horizon bars later, so only the last "
        "horizon anchor bars are right-censored); panels reconcile "
        "(by_target.n==outcome_coverage.n_closed, "
        "n<=max(0,n_bars-1-horizon), n_nonoverlap<=n, n_nonoverlap<=max(0,(n_bars-2)//horizon), "
        "concentration.top_share_abs>=k/n for k=max(1,ceil(n/20)) (the largest k of n "
        "observations carry at least k/n of the mass), worst_ret<=mean_ret<=best_ret (a mean "
        "is its pool's exact sum over its count, rounded once), "
        "episode_stats.n==sum of by_target.n, episode_stats.n_clusters a count in 0..n (0 iff "
        "n==0), episode_stats.max_cluster_share_abs>=largest_cluster_share_abs and "
        ">=1/n_clusters (==1 with one cluster), signal_coverage.n_bars==summary.n_bars); "
        "target_mode readable (the stamp says whether a pooled panel is part of the cell's "
        "contract); basket cells additionally carry a pooled dict that reconciles "
        "(pooled.n==sum of by_target.n, pooled.n_nonoverlap<=pooled.n, "
        "pooled.n_nonoverlap<=max(0,(n_bars-2)//horizon), "
        "max(by_target.n_nonoverlap)<=pooled.n_nonoverlap<=sum(by_target.n_nonoverlap), "
        "pooled.n<=len(targets)×max(0,n_bars-1-horizon), "
        "pooled.concentration.top_share_abs>=k/pooled.n, "
        "pooled.worst_ret<=pooled.mean_ret<=pooled.best_ret, member_share.by_target string-keyed "
        "over the targets "
        "exactly and, where max_member_share_abs is a share, every share in [0,1] with "
        "max==max_member_share_abs and sum==1 within 2**-50); a pooled key on a conjunction "
        "cell refuses (a restamped basket)"
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
    horizon: int | None = None
    if not isinstance(params, dict):
        unmet.append(
            f"params is not a dict ({type(params).__name__}) — the cell does not name the "
            "parameter assignment it measured; drifted input"
        )
    else:
        observed["params"] = params
        horizon = _int(params.get("horizon"))
        if horizon is None or horizon < 1:
            unmet.append(
                f"params.horizon={params.get('horizon')!r} is not a count >= 1 — the cell does "
                "not name the window it measured; drifted input"
            )
            horizon = None
    # A firing at bar t fills at the next bar's open and closes `horizon` bars later, so an index
    # of n_bars anchors firings on bars 0 … n_bars − 2 and closes those on bars
    # 0 … n_bars − 2 − horizon: per target at most `closable` closed observations, whose greedy
    # non-overlapping windows sit `horizon` bars apart — at most `windows` of them — and at most
    # `openable` right-censored ones, the anchor bars n_bars − 1 − horizon … n_bars − 2.
    closable: int | None = None
    windows: int | None = None
    openable: int | None = None
    if run_bars is not None and horizon is not None:
        closable = max(0, run_bars - 1 - horizon)
        windows = max(0, (run_bars - 2) // horizon)
        openable = run_bars - 1 - closable
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
    readable_cov = _grade_outcome_ledger(cov, run_bars, closable, openable, unmet)
    by_target_n_total, readable_bt, member_windows = _grade_by_target(
        by_tgt, cov, closable, windows, by_target_n, unmet
    )
    n_total_readable = bool(cov) and bool(by_tgt) and readable_cov and readable_bt
    _grade_signal_geometry(sig, run_bars, unmet)
    _grade_episode_stats(cell, by_target_n_total, n_total_readable, observed, unmet)
    _grade_pooled(
        cell,
        mode,
        targets,
        closable,
        windows,
        by_target_n_total,
        n_total_readable,
        member_windows if n_total_readable else None,
        observed,
        unmet,
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
    not a data defect (``cell_evidence`` holds the count to the ``min(horizon, n_bars − 1)``
    anchor bars the calendar can censor). What it is NOT allowed to hide is an in-bounds hole —
    those classify as ``no_outcome``/``no_benchmark`` upstream and refuse here.

    Reads are independent of every other check (no check relies on another having run); a missing
    or uncountable ledger refuses."""
    threshold = (
        "per target: exit_reasons.no_outcome == 0 and exit_reasons.no_benchmark == 0. "
        "exit_reasons.open is ALLOWED — with no holdout and no embargo, a forward window running "
        "past the data end is structural right-censoring, not a data hole (cell_evidence holds "
        "the count to the min(horizon, n_bars − 1) anchor bars the calendar can censor)"
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
    graded once, run-level, by ``source_coverage``, which catches the two hole classes this channel
    cannot see: absorption (every completion of a hole decides the node the same way — a decisive
    sibling settles the root while an operand is holed, or a ``first_true`` hole's completions
    re-converge before they disagree on any fire, e.g. one on a cooldown bar other than the
    cooldown's last) and a hole inside a windowed transform's warmup, which only delays its first
    value.

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
                f"{tgt}: {n_undef} undecidable decision bar(s) of {n_bars} — a missing or "
                "undefined input (a data hole, x/0, an overflow, a zero-dispersion window) "
                "suppresses firings before any outcome ledger can see them, so adverse firings "
                "may have been deleted; repair the decision series or guard the operation"
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
