"""Per-source availability for the runner: the raw decision inputs, counted leaf by leaf.

The root condition's ``defined`` channel answers "was the condition DECIDED?", which is strictly
weaker than "did the observer have its inputs?" — a hole is absorbed wherever every completion of
it decides the node the same way (a decisive sibling settles the root, Kleene F∧U = F; a
``first_true`` hole whose completions re-converge before they disagree on any fire), and one
inside a windowed transform's warmup only delays its first value. These functions read the raw
leaves DIRECTLY, so no operator sits between a hole and the count;
the runner mounts the result as the run-level ``summary["sources"]`` panel.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from seikan.compiler.data import MarketData, _anchor_feed
from seikan.dsl.schema import Thesis, iter_source_leaves
from seikan.types import SourceAvailability, SourceCoverage


class LeafMasks(NamedTuple):
    """One raw decision leaf's per-bar (rows × targets) masks: ``started`` — the leaf's warmup is
    over (its first True bar is the leaf's ``first_available``) — and ``available`` — everything
    it carries into the decision arrived intact. The bar-clock kinds are one mask twice (the
    value exists); a ``native`` leaf starts where the feed's first finite print arrives and can
    be started yet unavailable."""

    available: np.ndarray
    started: np.ndarray


def _arrived(counts: np.ndarray, prints: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """(rows,) a per-print running count, read on the bars with the ONE asof expression the
    anchoring uses (``data._anchor_feed``): the count of prints arrived at or before each bar."""
    running = pd.Series(counts.astype(float), index=prints.index)
    arrived: np.ndarray = _anchor_feed(running, index).fillna(0.0).to_numpy(dtype=float)
    return arrived


def _print_ledger(prints: pd.Series, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(rows,) ``(started, hole)`` of a ``native`` leaf over ONE print history.

    A native evaluation is seeded by the feed's first FINITE print, so the leaf starts at the bar
    that print arrives at (a print stamped ``s`` arrives at the first bar stamped at-or-after
    ``s``) — leading non-finite prints are the feed's warmup, never holes. From there, ``hole``
    marks every bar whose interval ``(previous bar, bar]`` absorbed a non-finite print: the prints
    the bar-clock asof read collapses away (only an interval's LAST print reaches a bar). The first
    bar's interval is the whole history before it: a native
    read consumes every print — feeds are not sliced by ``data.start`` — so a NaN print stamped
    before the evaluated interval is one the evaluation consumed, like one inside it. The ledger
    does not model how far back the expression reaches (a windowed expression may never read an
    old print), exactly as a bar-clock leaf is counted whichever parameters read it: a refusal
    that repairing or trimming the feed clears, never a hole it could miss."""
    vals = prints.to_numpy(dtype=float)
    finite = np.isfinite(vals)
    if not finite.any():
        none = np.zeros(len(index), dtype=bool)
        return none, none.copy()
    bad = ~finite
    bad[: int(np.argmax(finite))] = False
    started = _arrived(np.cumsum(finite), prints, index) > 0
    hole = np.diff(_arrived(np.cumsum(bad), prints, index), prepend=0.0) > 0
    return started, hole


def _native_masks(md: MarketData, name: str) -> LeafMasks:
    """A ``native`` leaf: started where the feed's first finite print arrives, and available
    where, beyond that, the anchored value exists (a late feed is warmup, a sparse one is
    available after its first stamp) and no print reaching the bar was missing."""
    native = md.external_native(name)
    columns = (
        [_print_ledger(native[t], md.index) for t in md.targets]
        if isinstance(native, dict)
        else [_print_ledger(native, md.index)] * len(md.targets)
    )
    started = np.column_stack([c[0] for c in columns])
    holes = np.column_stack([c[1] for c in columns])
    return LeafMasks(started & np.isfinite(md.external_values(name)) & ~holes, started)


def source_availability(thesis: Thesis, md: MarketData) -> dict[str, LeafMasks]:
    """``{source label → its (rows × targets) masks}`` for every raw decision input the entry
    tree reads.

    The decision-side completeness contract is enforced HERE, at the sources, rather than at the
    root condition: the root's three-valued ``defined`` channel is masked wherever every completion
    of a hole decides the node the same way — an operand settled by a decisive sibling (Kleene
    ``F∧U = F``), a ``first_true`` hole whose completions re-converge before they disagree on any
    fire — and a hole inside a windowed transform's warmup only delays its first value: either way
    the root reads no undecided bar while an input is missing.

    "Available" is the finiteness of the JOINED per-bar value, which is what the decision
    actually reads. For an external feed that is the post-asof-ffill value, so ordinary sparse
    stamping (a weekly feed on daily bars) is available everywhere after its first stamp — the
    legitimate design — while a DELETED interior stamp is invisible by construction (ffill
    carries the prior value) and an explicitly stamped NaN row is not. A ``native`` leaf reads
    the prints themselves — every print between two bars, not the last one — so an explicitly
    stamped NaN print absorbed between two bars is a hole there too (:func:`_native_masks`).
    """
    out: dict[str, LeafMasks] = {}
    for kind, name in iter_source_leaves(thesis.entry):
        if kind == "native":
            out[f"native:{name}"] = _native_masks(md, name)
            continue
        if kind == "field":
            arr = md.field(name).to_numpy(dtype=float)
        elif kind == "external":
            arr = md.external_values(name)
        else:
            arr = md.days_since_values(name)
        finite = np.isfinite(arr)
        out[f"{kind}:{name}"] = LeafMasks(finite, finite)
    return out


def source_coverage(
    src_avail: dict[str, LeafMasks], index: pd.DatetimeIndex, g: int
) -> SourceCoverage:
    """One target's per-source availability ledger over the WHOLE evaluated interval.

    Counts only holes FROM a source's start on — its ``first_available``, the first bar its
    warmup is over (for a bar-clock leaf the first bar its value exists, so the count runs
    strictly after it): a series that merely starts late is warmup — the observer had nothing to
    read yet, exactly as a transform's warmup window is not a hole — and ``first_available`` is
    reported so a late start stays auditable in the evidence. The interval is the full joined
    index: every bar a decision could have been taken on is covered, including the stretches no
    cell happened to fire in. The ledger stays FACTUAL on a never-available leaf (zero
    post-warmup holes, a null ``first_available``): the refusal is
    ``gate._check_source_coverage``'s (policy v3), never the emission's — refused three times,
    trusted zero."""
    n_bars = len(index)
    union = np.zeros(n_bars, dtype=bool)
    by_source: dict[str, SourceAvailability] = {}
    for label in sorted(src_avail):
        avail = src_avail[label].available[:, g]
        started = src_avail[label].started[:, g]
        first = int(np.argmax(started)) if bool(started.any()) else None
        hole = ~avail
        if first is None:
            # Never available at all: no bar was ever post-warmup, so the hole count is
            # vacuously zero — the ledger stays factual and the null first_available below is
            # what the gate refuses.
            hole[:] = False
        else:
            hole[:first] = False
        union |= hole
        by_source[label] = {
            "n_missing": int(hole.sum()),
            "first_available": index[first].isoformat() if first is not None else None,
        }
    return {
        "n_bars": int(n_bars),
        "n_missing": int(union.sum()),
        "by_source": by_source,
    }
