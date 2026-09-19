"""The Turtle simulation behind ``seikan stock-turtle-trade-long-only``.

A separate simulation package beside the observer-pure event study: it consumes the event
study's entry mask (``api.list_entries``) as the entry signal and hands every (cell × target)
to seikan's own Rust engine (``seikan._turtle``), which holds the long-only Turtle rules, prices
every fill under the coefficients' cost model and keeps the only set of books. Python admits
the data, translates the thesis, computes the performance arithmetic and assembles the report.
Nothing here touches ``compiler/``, ``analysis/`` or ``gate/``.
"""

from seikan.turtle.coefficients import TurtleCoefficients, canonical_coefficients_hash
from seikan.turtle.errors import TurtleRequestError
from seikan.turtle.market import SimulationData, preflight
from seikan.turtle.signals import EntryCell, entry_cells

__all__ = [
    "EntryCell",
    "SimulationData",
    "TurtleCoefficients",
    "TurtleRequestError",
    "canonical_coefficients_hash",
    "entry_cells",
    "preflight",
]
