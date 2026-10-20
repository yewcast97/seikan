"""evaluate_gate: one uniform checklist applied to every cell independently."""

from __future__ import annotations

from collections.abc import Mapping

from seikan.gate._checks_cell import _CELL_CHECKS
from seikan.gate._checks_run import _RUN_CHECKS
from seikan.gate._model import CellReport, GateReport
from seikan.settings import GateThresholds
from seikan.types import (
    JsonValue,
)


def _cell_identity(entry: object, index: int) -> tuple[str, dict[str, JsonValue]]:
    """``(cell_id, params)`` for a summary cell entry, total over drifted input.

    Identity is the params plus the POSITION in the panel; the rendered ``cell_id`` is a label.
    A drifted entry that carries neither still gets a positional label so the caller can align
    ``gate["cells"][i]`` with ``summary["cells"][i]`` and see exactly which cell was unreadable.
    """
    raw_id = entry.get("cell_id") if isinstance(entry, dict) else None
    cell_id = raw_id if isinstance(raw_id, str) else f"<unreadable cell[{index}]>"
    raw_params = entry.get("params") if isinstance(entry, dict) else None
    params = raw_params if isinstance(raw_params, dict) else {}
    return cell_id, params


def evaluate_gate(
    summary: Mapping[str, object], thresholds: GateThresholds | None = None
) -> GateReport:
    """Run the full checklist over one engine summary → :class:`GateReport`.

    Three run-level checks are evaluated once; then every entry of ``summary["cells"]`` is graded
    independently by five per-cell checks, in panel order, so ``gate["cells"][i]`` aligns with
    ``summary["cells"][i]``. A cell's ``met`` is the conjunction of its OWN checks and all
    run-level checks — an unmet run-level check leaves every cell unmet, so a caller reading one
    cell's ``met`` never has to AND the sections itself.

    Nothing short-circuits and nothing is filtered: every check is evaluated and reported for
    every cell, and the summary itself is never modified. An unreadable ``cells`` panel yields no
    graded cells (``evidence_complete`` is already unmet); a malformed individual entry gets an
    unmet ``cell_evidence`` while its siblings grade normally. Drifted input refuses with a
    detail — this function does not raise on it.

    ``thresholds`` omitted, the exam is ``GateThresholds()``: the canonical one, tightened by
    any ``SEIKAN_*`` variable — an unknown or out-of-domain one raises
    ``pydantic.ValidationError``.
    """
    t = thresholds if thresholds is not None else GateThresholds()
    s = summary or {}
    run_checks = [check(s, t) for check in _RUN_CHECKS]
    run_all_met = all(c.met for c in run_checks)

    raw_cells = s.get("cells")
    cells: list[CellReport] = []
    if isinstance(raw_cells, list):
        for i, entry in enumerate(raw_cells):
            cell_id, params = _cell_identity(entry, i)
            checks = [check(entry, s, t) for check in _CELL_CHECKS]
            cells.append(
                CellReport(
                    cell_id=cell_id,
                    params=params,
                    met=run_all_met and all(c.met for c in checks),
                    checks=checks,
                )
            )
    return GateReport(run_checks=run_checks, cells=cells)
