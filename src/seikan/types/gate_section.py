"""Emitted-shape declarations: the gate section."""

from __future__ import annotations

from typing import TypedDict

from seikan.types.scalars import (
    JsonValue,
)

# ---- the gate section ---------------------------------------------------------------------
#
# ``gate``'s dataclasses are the in-memory result; these are the dicts their ``to_dict`` methods
# render for the report's ``gate`` layer.


class GateCheckDict(TypedDict):
    """One check's result — ``gate.GateCheck.to_dict``, reported for every run-level check and for
    every cell's five, always evaluated and never short-circuited.

    ``observed`` is what the check actually read (a scalar, or a per-target payload) and
    ``threshold`` what it required (a number for the sealed ceilings, a sentence for the
    structural contracts); both are JSON payloads this layer carries without interpreting.
    ``detail`` states what held, or every unmet condition joined — a refusal always says why.

    The two payloads are typed from opposite sides of the trust boundary the gate sits on:
    ``threshold`` is written BY the checklist (a knob's value or its own sentence), so it is JSON
    by construction, while ``observed`` ECHOES what the check found in an untrusted summary — a
    drifted report may carry anything under the key a check looked at, and reporting it as found
    is how a refusal says what it saw. ``json_safe`` renders it at emit.
    """

    name: str
    met: bool
    observed: object
    threshold: JsonValue
    detail: str


class GateCellDict(TypedDict):
    """One declared cell's complete checklist result — ``gate.CellReport.to_dict``, index-aligned
    with ``summary["cells"]``.

    ``met`` is the conjunction of this cell's own checks AND every run-level check, so a caller
    reading one cell never has to AND the sections itself. ``cell_id``/``params`` mirror the
    summary cell being graded — echoed as read, since a drifted entry may carry neither (it still
    gets a positional label so the alignment holds).
    """

    cell_id: str
    params: dict[str, JsonValue]
    met: bool
    checks: list[GateCheckDict]


class GateSection(TypedDict):
    """The report's whole ``gate`` layer — ``gate.GateReport.to_dict``.

    There is no verdict key and no short-circuit, and the summary this grades is never filtered:
    ``n_met`` counts cells whose ``met`` is True, a number the caller reads alongside
    ``n_cells`` and ``n_hypotheses_attempted`` when it prices its own selection.
    """

    policy_version: int
    n_cells: int
    n_met: int
    run_checks: list[GateCheckDict]
    cells: list[GateCellDict]
