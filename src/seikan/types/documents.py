"""Emitted-shape declarations: the emitted documents."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from seikan.types.run import (
    RunSummary,
)
from seikan.types.scalars import (
    JsonValue,
)

# ---- the emitted documents ----------------------------------------------------------------


class SerializedResult(TypedDict):
    """``serialize.serialize_result``'s return — the thesis name and the summary, JSON-safe.

    The summary rides VERBATIM (every declared cell, whatever the checklist says); the
    per-observation trades frame is never embedded — its only channel is ``write_trades_csv``.
    After ``json_safe`` the shape is unchanged except that non-finite floats are null.
    """

    name: str
    summary: RunSummary


class ThresholdsSnapshot(TypedDict):
    """The four checklist knobs actually used — ``settings.GateThresholds.snapshot``.

    Stamped into every run report's identity layer and reconstructible:
    ``GateThresholds(**snapshot())`` round-trips.
    """

    thesis_min_trades: int
    thesis_min_n_nonoverlap: int
    thesis_max_concentration: float
    thesis_max_hypotheses: int


class DataDigest(TypedDict):
    """One bound data key's provenance — an entry of the report's ``identity.data_digests``,
    built by the CLI.

    Keyed by the LOGICAL key the thesis declares (never the path), so two runs are comparable.
    ``column`` is ALWAYS present and null when the key bound none ("the file named its own" is a
    fact worth stating); ``sha256`` is looked up by PATH, so two keys answered by one file
    correctly share one digest.
    """

    path: str
    column: str | None
    sha256: str | None


#: Where a checklist knob's value came from — the SOURCE, never the value.
type ThresholdProvenance = Literal["default", "env", "cli"]


class ReportIdentity(TypedDict):
    """WHICH exam ran on WHICH inputs — the run report's ``identity`` layer, built by the CLI.

    A changed rule set (``dsl_hash``), data byte (``data_digests``), exam knob (``thresholds``) or
    knob PROVENANCE is a visibly distinct exam. ``thresholds_canonical`` lives here because it is
    an identity fact, not a checklist result; ``thresholds_provenance`` maps each knob to where
    its value came from, which the snapshot alone cannot say.
    """

    name: str
    dsl_hash: str
    data_digests: dict[str, DataDigest]
    thresholds: ThresholdsSnapshot
    thresholds_canonical: bool
    thresholds_provenance: dict[str, ThresholdProvenance]
    environment: dict[str, str]


class OutputEntry(TypedDict):
    """One file a run wrote — an entry of the report's ``outputs``, keyed in NOMINATION order.

    ``rows_written`` rides every CSV output; the report's own entry carries the path alone, since
    the report is what enumerates the others and is published last.
    """

    path: str
    rows_written: NotRequired[int]


class ThresholdDoc(TypedDict):
    """One knob's documentation row — ``seikan schema``'s ``thresholds`` list, rendered by the CLI
    straight off ``GateThresholds.model_fields`` so a dropped field disappears from the flags, the
    env-var listing and the docs at once."""

    field: str
    default: JsonValue
    env_var: str
    cli_flag: str


class ValidationRecord(TypedDict):
    """One pydantic validation failure, JSON-safe — ``cli._validation_records``. The context is
    deliberately dropped (it can carry a non-serializable original exception); the human text
    already lives in ``msg``."""

    loc: list[str | int]
    msg: str
    type: str


class ErrorEnvelope(TypedDict):
    """The machine-readable error a failing command emits on stdout — the ``error`` section of
    ``emitted.ErrorDocument``.

    ``type`` names the class the exit code reflects (``usage`` / ``dsl_invalid`` /
    ``thresholds_invalid`` / ``data_invalid`` / ``internal``); ``errors`` rides only the two
    pydantic-backed classes, which have structured records to carry.
    """

    type: Literal["usage", "data_invalid", "dsl_invalid", "thresholds_invalid", "internal"]
    message: str
    errors: NotRequired[list[ValidationRecord]]
