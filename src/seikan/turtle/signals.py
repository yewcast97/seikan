"""The thesis → entry-signal translation: one boolean firing series per (entry combo × target),
read off the engine's own listing so the turtle enters on exactly the bars ``seikan run``
measures at and ``--entry-flags-out`` writes.

A swept thesis declares several entry combos; each becomes one CELL of the simulation
(one backtest, reported beside the others, never ranked). The measurement horizon plays no
part — the turtle owns its exit — so combos expand over the ENTRY axes only, exactly as
``api.list_entries`` expands them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seikan.api import MarketData, list_entries
from seikan.dsl.schema import Thesis
from seikan.turtle.errors import TurtleRequestError
from seikan.types import ParamValue


@dataclass(frozen=True)
class EntryCell:
    """One entry combo: its canonical label (the entry-flags column base, ``entry`` or
    ``entry[axis=value,...]``), its parameter assignment, and the firing flags per target on the
    joined bar clock."""

    cell_id: str
    params: dict[str, ParamValue]
    fired: dict[str, np.ndarray]


def entry_cells(thesis: Thesis, md: MarketData) -> tuple[list[str], list[EntryCell]]:
    """The swept axis names and one :class:`EntryCell` per declared entry combo, in the engine's
    combo order. Refuses a thesis whose ``direction`` is not ``longonly``: the turtle buys on a
    firing, and taking a short-side alarm long is a different exam."""
    if thesis.params.direction != "longonly":
        raise TurtleRequestError(
            f"stock-turtle-trade-long-only trades LONG on the entry firing, but the thesis "
            f"declares direction {thesis.params.direction!r}"
        )
    listing = list_entries(thesis, md)
    n_targets = len(md.targets)
    multi = n_targets > 1
    columns = list(listing.entry_flags.columns)
    rows = listing.entries
    if len(columns) != len(rows) or len(rows) % n_targets:
        raise RuntimeError("the entry listing's columns and rows disagree on the combo grid")
    axes = [k for k in rows[0] if k not in ("target", "timestamps")] if rows else []
    cells: list[EntryCell] = []
    for c in range(len(rows) // n_targets):
        first = rows[c * n_targets]
        params: dict[str, ParamValue] = {}
        for k in axes:
            value = first[k]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise RuntimeError(f"entry axis {k!r} carries a non-numeric value {value!r}")
            params[k] = value
        fired: dict[str, np.ndarray] = {}
        base: str | None = None
        for j, target in enumerate(md.targets):
            column = columns[c * n_targets + j]
            label = column.rsplit("@", 1)[0] if multi else column
            if base is None:
                base = label
            elif label != base:
                raise RuntimeError(
                    f"entry-flags column {column!r} does not share the combo label {base!r}"
                )
            fired[target] = listing.entry_flags[column].to_numpy().astype(bool)
        cells.append(EntryCell(cell_id=base or "entry", params=params, fired=fired))
    return axes, cells
