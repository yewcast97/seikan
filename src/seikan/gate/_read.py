"""Strict readers over drifted summary input: every numeric/panel read narrows through these
funnels, refusing rather than crashing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import SupportsFloat, cast

from seikan.types import (
    EXIT_REASONS,
    OUTCOME_KINDS,
    TARGET_MODES,
    ExitReason,
    TargetMode,
)


def _num(value: object) -> float | None:
    """float(value), with None/NaN/±inf/garbage collapsed to None (a NaN would sail past
    comparisons and an inf would sail THROUGH them; drifted non-finite input must produce a
    refusal, never a crash or a certified impossibility)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        # The conversion is duck-typed BY DESIGN — the ``except`` below IS the contract for
        # everything that does not convert (a dict, a list, a non-numeric string), which is why
        # the parameter is an untrusted ``object``. The cast states that intent to the checker;
        # nothing about it changes what a non-convertible value does here.
        v = float(cast("SupportsFloat", value))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _int(value: object) -> int | None:
    """The integer twin of :func:`_num` — None on missing/non-finite/garbage/NON-INTEGRAL input.
    A count of 1.9 is drifted input, not "1": truncation would certify a search size the summary
    never actually recorded."""
    v = _num(value)
    if v is None:
        return None
    try:
        if v != int(v):
            return None
        return int(v)
    except (OverflowError, ValueError):  # _num already blocks inf; belt-and-braces
        return None


def _prob(value: object) -> float | None:
    """The probability twin of :func:`_num` — additionally None outside [0, 1]. A p-value or
    mass share of 1.5 is drifted input, not a number to compare."""
    v = _num(value)
    if v is None or not (0.0 <= v <= 1.0):
        return None
    return v


def _targets(s: Mapping[str, object]) -> list[str] | None:
    """The summary's target list, or None when it is missing/empty/not all strings.

    Target names index every panel and are ``sorted()`` in half a dozen places to build stable
    ``observed`` payloads and set differences. A drifted summary carrying a non-string name
    (``["AAA", 3]``) would raise ``TypeError`` inside the comparison — a CRASH, when the
    contract is that drifted input refuses with a detail. Type-checking once, here, keeps every
    downstream sort total."""
    t = s.get("targets")
    if not isinstance(t, (list, tuple)) or not t:
        return None
    if not all(isinstance(x, str) for x in t):
        return None
    return list(t)


def _mode(s: Mapping[str, object]) -> TargetMode | None:
    """The summary's ``target_mode`` stamp, or None unless it is EXACTLY one of the two rubric
    names.

    The stamp SELECTS how cross-target evidence is graded (conjunction: the weakest target
    decides; basket: the pooled panel is graded), so a missing or garbage stamp is never a
    default — it is an unanswerable question, and every check that dispatches on it refuses
    fail-closed (the outcome-stamp precedent: an assumed mode is the stamp-stripping bypass one
    field over)."""
    m = s.get("target_mode")
    return m if m in TARGET_MODES else None


def _cell_panel(cell: object, key: str) -> dict[str, object]:
    """A string-keyed panel off a cell entry — ``{}`` when the entry is not a dict or the panel
    is missing/non-string-keyed. The four per-cell checks read their panels through this."""
    if not isinstance(cell, dict):
        return {}
    return _str_keyed(cell.get(key)) or {}


def _int_ledger(
    reasons: dict[str, object],
) -> tuple[dict[str, int] | None, list[ExitReason]]:
    """Parse an exit-reason dict into countable non-negative ints, or ``(None, bad_keys)``.

    The one place the ledger's numeric hygiene is established: past a ``None`` return every
    reason count is a genuine int — the narrowing the two casts this replaces used to assert."""
    counts = {k: _int(reasons.get(k)) for k in EXIT_REASONS}
    bad = sorted(k for k, v in counts.items() if v is None or v < 0)
    if bad:
        return None, bad
    return {k: v for k, v in counts.items() if v is not None}, []


def _detail(ok: str, unmet: list[str], suffix: str = "") -> str:
    """The uniform check detail: the ok sentence when nothing is unmet, else every reason joined
    in APPEND order — the order is emitted text, so helpers that collect ``unmet`` preserve it."""
    return ok if not unmet else "; ".join(unmet) + suffix


def _as_dict(value: object) -> dict[str, object]:
    """Collapse a maybe-dict read to ``{}`` — a drifted summary can carry anything under a key a
    check reads, and every panel read narrows through this one funnel instead of respelling the
    isinstance dance."""
    return value if isinstance(value, dict) else {}


def _str_keyed(value: object) -> dict[str, object] | None:
    """A per-target/per-source panel dict, or None unless every key is a string (same total-order
    discipline as :func:`_targets` — panel keys are sorted just as often)."""
    if not isinstance(value, dict):
        return None
    if not all(isinstance(k, str) for k in value):
        return None
    return value


def _incommensurable_pool(s: Mapping[str, object]) -> tuple[bool, str]:
    """``(incommensurable, reason)`` for the summary's pooled CROSS-TARGET mass reads.

    A ``diff`` outcome measures each target in its OWN level units (bp, index points, ratio
    turns), so cluster means and |return|-mass shares across several targets compare
    incomparables — a high-scale target dominates every pooled read. pct/log are commensurable.

    Fails CLOSED on an unreadable stamp. The runner ALWAYS writes the explicit ``outcome`` dict,
    so a multi-target summary that arrives without it, with ``None``, or with a kind outside the
    vocabulary is drifted input. Reading an absent stamp as "commensurable" would let STRIPPING
    it bypass this guard entirely, and accepting a null is the same bypass one spelling over."""
    targets = s.get("targets")
    multi = isinstance(targets, (list, tuple)) and len(targets) > 1
    if not multi:
        return False, ""
    if "outcome" not in s:
        return True, (
            "summary carries no outcome stamp — the measurement algebra is unrecorded, so "
            "cross-target |return|-mass shares cannot be certified commensurable; drifted input"
        )
    outcome = s.get("outcome")
    kind = outcome.get("kind") if isinstance(outcome, dict) else None
    if kind not in OUTCOME_KINDS:
        return True, (
            f"outcome stamp is unreadable ({outcome!r} — kind not in {list(OUTCOME_KINDS)}); "
            "a null stamp is stripped input, and the measurement algebra "
            "is unverifiable either way; drifted input"
        )
    if kind == "diff":
        return True, (
            "outcome kind='diff' across multiple targets — cross-target |return|-mass shares "
            "mix incomparable level units; run per-target or use a pct/log outcome"
        )
    return False, ""
