"""Emitted-shape declarations: the reliability / CSCV pass."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:  # annotations only — nothing beyond the stdlib is imported at runtime
    import numpy as np
from seikan.types.scalars import (
    CellKey,
)

# ---- the reliability / CSCV pass ----------------------------------------------------------
#
# The one place a raw ARRAY rides a pipeline dict: the runner hands the statistical layer a
# roster of (mask, forward-return) columns and gets per-cell reads back.


class ReliabilityCell(TypedDict):
    """One entry of the reliability roster — built by ``compiler.runner``, consumed by
    ``analysis.stats.reliability_summary``, ``pooled_reliability_summary`` and ``cscv_pbo``.

    ``key`` is the cell's ``ComboKey`` with the target appended (the pooled pass groups on
    ``key[:-1]``); ``mask_col`` is the (T,) firing mask and ``fwd_col`` the (T,) signed forward
    return anchored at the FIRING bar, so the two line up for the rotation null; ``h`` is the
    measurement horizon in bars.
    """

    key: CellKey
    mask_col: np.ndarray
    fwd_col: np.ndarray
    h: int


class ReliabilityRead(TypedDict):
    """One cell's overlap-aware inference — the per-cell value of
    ``reliability_summary``/``pooled_reliability_summary``.

    Merged into each cell's per-target (or pooled) panel by ``compiler.runner``. The three
    estimators are known anti-conservative and read by NO check: evidence a reader weighs, never
    a certificate. ``rot_n_null`` is the number of DEFINED shifts the cell's null was formed
    over — its own resolution floor is ``1/(1 + rot_n_null)``, which on a sparse mask sits
    above the run-level ``rotation.p_resolution``.
    """

    rot_p: float
    rot_n_null: int
    t_hac: float
    hac_se: float
    n_nonoverlap: int


class ReliabilitySummary(TypedDict):
    """The reliability pass's return — ``analysis.stats.reliability_summary`` (keyed by
    ``CellKey``) and ``pooled_reliability_summary`` (keyed by the combo prefix).

    A ``ComboKey`` IS a ``CellKey`` with nothing appended, so the one key type serves both passes.
    A cell that never fired has NO entry: the runner tracks the DECLARED grid separately, so a
    missing key reads as "no firings", never as "dropped". ``n_shifts`` is the rotation null's
    shift count, which the runner stamps as ``summary["rotation"]`` — the smallest attainable
    ``rot_p`` is ``1/(1 + n_shifts)``.
    """

    per_cell: dict[CellKey, ReliabilityRead]
    n_shifts: int


#: The one way the degradation-slope regression declines: no usable spread in the train scores.
type DegradationSlopeReason = Literal["degenerate_train_scores"]


class PboBlock(TypedDict):
    """The grid-level CSCV block — ``analysis.stats.cscv_pbo``, mounted as ``summary["pbo"]``.

    A property of the SEARCH SPACE the caller is about to select from, attached to no hypothesis
    and read by no cell's grade. ``pbo`` is None exactly when ``reason`` is set (``single_combo``
    — fewer than two DISTINCT candidates to select among, a one-combo grid or a grid of
    byte-identical clones; ``insufficient_data`` — too little block-local data even at S=4), and
    the diagnostics are NaN → null there. The population ledger reads left to right:
    ``n_combos <= n_combos_scoreable <= n_combos_declared`` — the DISTINCT candidates scored
    (byte-identical combos collapse to one before scoring), the combos admissible with at least
    one fired-and-closed pool, and the declared combo × horizon grid
    (``== n_hypotheses_attempted``). CSCV never sees a declared combo that produced no closed
    pool; the ledger is what says how far short of the declared grid the scored population fell.
    The per-split ledger sits beside it: ``n_splits_attempted`` is the C(S, S/2) the partition
    offered against the ``n_splits`` actually scored (a split whose train winner has no finite
    test score is skipped), and ``n_candidates_min`` the smallest finite candidate population any
    scored split ranked over — canonical CSCV assumes a fixed count, and this is how far the
    block-local thinning departed from it.
    """

    pbo: float | None
    reason: Literal["single_combo", "insufficient_data"] | None
    n_splits: int
    n_splits_attempted: int
    n_candidates_min: int
    n_combos: int
    n_combos_scoreable: int
    n_combos_declared: int
    blocks: int
    lambda_mean: float | None
    oos_degradation_slope: float | None
    oos_degradation_slope_reason: DegradationSlopeReason | None
    prob_oos_loss: float | None
