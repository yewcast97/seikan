"""The policy version and the three result dataclasses the checklist reports through."""

from __future__ import annotations

from dataclasses import dataclass

from seikan.types import (
    GateCellDict,
    GateCheckDict,
    GateSection,
    JsonValue,
)

#: The gate-policy version stamped into every report. Bumped whenever a check's SEMANTICS change
#: (new check, changed branch logic, changed default meaning) or the section's emitted shape
#: moves, so two results are comparable only when their policy versions match. It names the
#: CHECKLIST semantics — ``seikan_version`` names the package and ``analysis.STATISTICS_VERSION``
#: the estimators the checklist reads. The revision history is in CHANGELOG.md.
POLICY_VERSION = 3


@dataclass(frozen=True)
class GateCheck:
    name: str
    met: bool
    #: What the check READ, echoed as it was found — an ``object``, because a drifted summary may
    #: carry anything under the key a check looked at, and reporting it verbatim is how a refusal
    #: says what it saw.
    observed: object
    #: What the check REQUIRED — a sealed knob's value or the checklist's own sentence. This side
    #: is written BY the gate and never echoed, so it is JSON by construction.
    threshold: JsonValue
    detail: str

    def to_dict(self) -> GateCheckDict:
        return {
            "name": self.name,
            "met": self.met,
            "observed": self.observed,
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CellReport:
    """One declared parameter × horizon cell's complete checklist result.

    ``met`` is the conjunction of this cell's own checks AND every run-level check, so a
    caller reading ``cells[i].met`` gets the complete answer without ANDing sections itself.
    ``cell_id`` and ``params`` mirror the summary cell they grade; identity is the params plus
    the position in the panel, never the rendered label.
    """

    cell_id: str
    params: dict[str, JsonValue]
    met: bool
    checks: list[GateCheck]

    def to_dict(self) -> GateCellDict:
        return {
            "cell_id": self.cell_id,
            "params": self.params,
            "met": self.met,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass
class GateReport:
    """The whole gate section: the run-level checks once, then every graded cell in panel order.

    There is no scalar verdict. ``n_met`` counts cells whose ``met`` is True — a number the
    caller reads alongside ``n_cells`` and ``n_hypotheses_attempted`` when it prices its own
    selection.
    """

    run_checks: list[GateCheck]
    cells: list[CellReport]

    def to_dict(self) -> GateSection:
        return {
            "policy_version": POLICY_VERSION,
            "n_cells": len(self.cells),
            "n_met": sum(1 for c in self.cells if c.met),
            "run_checks": [c.to_dict() for c in self.run_checks],
            "cells": [c.to_dict() for c in self.cells],
        }
