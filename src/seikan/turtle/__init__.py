"""The Turtle simulation behind ``seikan stock-turtle-trade-long-only``.

A separate simulation package beside the observer-pure event study: it consumes the engine's
entry mask (``api.list_entries``) as the entry signal and runs the long-only Turtle position
rules on nautilus_trader's simulated exchange, with the rules themselves implemented once in the
Rust kernel (``seikan._turtle``). Nothing here touches ``compiler/``, ``analysis/`` or ``gate/``.

Importing this package never imports nautilus_trader: the modules that need it (``market``'s
builders, ``strategy``, ``engine``) import it inside the functions that use it, so ``seikan
run`` and ``seikan schema`` never pay for the venue.
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
