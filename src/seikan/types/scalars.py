"""Emitted-shape declarations: scalar vocabularies (repackaged verbatim from the former
monolithic ``types.py`` — the package ``__init__`` re-exports everything, so
``from seikan.types import X`` keeps working; see its reading rules).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:  # annotations only — nothing beyond the stdlib is imported at runtime
    import pandas as pd

# ---- scalar vocabularies ------------------------------------------------------------------
#
# The small alias set the dict shapes below are built out of. The Literal ones mirror the DSL's
# own Literal fields (``dsl.schema``) so a stamp written from a validated thesis type-checks
# against the summary it is written into, and the gate's vocabulary tuples can be typed by them.

#: One swept axis's assigned value — a transform window/period, a swept ``Constant`` cutoff, or
#: the measurement horizon. Ints and floats only: the sweep axes are numeric by construction
#: (``vectorize.collect_sweeps``), and a target NAME is never a parameter.
type ParamValue = int | float

#: A cell's parameter assignment as a tuple, in ``summary["params"]`` order (the horizon last
#: when it was swept) — the runner's ``combo_key``, and the lookup key of every per-combo map.
type ComboKey = tuple[ParamValue, ...]

#: A ``ComboKey`` with the TARGET name appended — the per-(cell × target) key of the reliability
#: pass and the runner's row-group partitions.
type CellKey = tuple[ParamValue | str, ...]

#: Any value that survives ``serialize.json_safe`` — the type of a payload this package carries
#: without interpreting: a gate check's ``observed``/``threshold``, a static contract document,
#: the parsed thesis DSL. Recursive on purpose: it says "JSON", not "anything".
type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None

#: A thesis DSL document as PARSED FROM JSON, before ``dsl.schema.Thesis`` validates it — what
#: ``gate.canonical_dsl_hash`` normalizes and hashes.
type DslDocument = dict[str, JsonValue]

#: The measurement algebra (``params.outcome.kind``) every reported number is denominated in.
type OutcomeKind = Literal["pct", "log", "diff"]

#: The human-readable denomination each algebra measures in (``runner._OUTCOME_UNITS``).
type OutcomeUnits = Literal["fraction", "log", "level_diff"]

#: The declared target semantics — the stamp the checklist dispatches its rubric on.
type TargetMode = Literal["conjunction", "basket"]

#: The sign convention of every measured return (``params.direction``).
type Direction = Literal["longonly", "shortonly"]

#: The excess-return adjustment, when one was declared (``params.benchmark``).
type BenchmarkMode = Literal["market", "cross_mean"]

#: The complete exit-reason vocabulary of the censoring ledger. ``horizon`` is a closed
#: observation; the other three are censorings (``open`` = structural right-censoring at the data
#: end, the other two = data holes the checklist refuses).
type ExitReason = Literal["horizon", "open", "no_outcome", "no_benchmark"]

#: The baseline panel's exclusion vocabulary — the exit reasons MINUS ``horizon``, since a
#: baseline row has nothing to close.
type ExclusionReason = Literal["open", "no_outcome", "no_benchmark"]

#: The same vocabularies as runtime tuples, derived from the Literals above so the type and the
#: value can never disagree — the runner's ledgers and the gate's readers iterate these.
EXIT_REASONS: tuple[ExitReason, ...] = get_args(ExitReason.__value__)
EXCLUSION_REASONS: tuple[ExclusionReason, ...] = get_args(ExclusionReason.__value__)
OUTCOME_KINDS: tuple[OutcomeKind, ...] = get_args(OutcomeKind.__value__)
TARGET_MODES: tuple[TargetMode, ...] = get_args(TargetMode.__value__)

#: The checklist's evidence basis — there is exactly one: no holdout exists, every cell is
#: measured once over the whole index, and the gate verifies the stamp as a drift detector.
type EvidenceBasis = Literal["full_sample"]

#: A target's data shape (``compiler.data``): full OHLCV prices, or a single value column
#: synthesized into OHLC.
type TargetShape = Literal["ohlcv", "series"]

#: One row of ``EntryListReport.entries``: the swept signal-axis levels (caller-chosen keys),
#: ``target``, and ``timestamps`` — every bar the entry fired on, ascending, empty when it never
#: fired. Produced by ``api.list_entries``; library-only (no CLI output carries it).
type EntryRow = dict[str, ParamValue | str | list[pd.Timestamp]]
