"""Per-source availability for the runner: the raw decision inputs, counted leaf by leaf.

The root condition's ``defined`` channel answers "was the condition DECIDED?", which is strictly
weaker than "did the observer have its inputs?" — a decisive sibling settles the root (Kleene
F∧U = F) and a NaN-skipping recursive kernel carries state across a hole and emits a finite
value after it. These functions read the raw leaves DIRECTLY, so no operator sits between a hole
and the count; the runner mounts the result as the run-level ``summary["sources"]`` panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seikan.compiler.data import MarketData
from seikan.dsl.schema import Thesis, iter_source_leaves
from seikan.types import SourceAvailability, SourceCoverage


def source_availability(thesis: Thesis, md: MarketData) -> dict[str, np.ndarray]:
    """``{source label → (rows × targets) finite mask}`` for every raw decision input the entry
    tree reads.

    The decision-side completeness contract is enforced HERE, at the sources, rather than at the
    root condition: the root's Kleene ``defined`` channel is masked by a decisive sibling
    (``F∧U = F``), and a NaN-skipping recursive kernel (EMA, expanding max/min,
    ``bars_since_extremum``) silently carries state across a hole and emits a FINITE value
    afterwards, so a hole can alter later decisions while leaving the root fully decided.

    "Available" is the finiteness of the JOINED per-bar value, which is what the decision
    actually reads. For an external feed that is the post-asof-ffill value, so ordinary sparse
    stamping (a weekly feed on daily bars) is available everywhere after its first stamp — the
    legitimate design — while a DELETED interior stamp is invisible by construction (ffill
    carries the prior value) and an explicitly stamped NaN row is not.
    """
    out: dict[str, np.ndarray] = {}
    for kind, name in iter_source_leaves(thesis.entry):
        if kind == "field":
            arr = md.field(name).to_numpy(dtype=float)
        elif kind == "external":
            arr = md.external_values(name)
        else:
            arr = md.days_since_values(name)
        out[f"{kind}:{name}"] = np.isfinite(arr)
    return out


def source_coverage(
    src_avail: dict[str, np.ndarray], index: pd.DatetimeIndex, g: int
) -> SourceCoverage:
    """One target's per-source availability ledger over the WHOLE evaluated interval.

    Counts only holes STRICTLY AFTER a source's first available bar: a series that merely
    starts late is warmup — the observer had nothing to read yet, exactly as a transform's
    warmup window is not a hole — and ``first_available`` is reported so a late start stays
    auditable in the evidence. The interval is the full joined index: every bar a decision could
    have been taken on is covered, including the stretches no cell happened to fire in. The
    ledger stays FACTUAL on a never-available leaf (zero post-warmup holes, a null
    ``first_available``): the refusal is ``gate._check_source_coverage``'s (policy v3), never
    the emission's — refused three times, trusted zero."""
    n_bars = len(index)
    union = np.zeros(n_bars, dtype=bool)
    by_source: dict[str, SourceAvailability] = {}
    for label in sorted(src_avail):
        avail = src_avail[label][:, g]
        first = int(np.argmax(avail)) if bool(avail.any()) else None
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
