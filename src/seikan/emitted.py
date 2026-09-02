"""Runtime validation of the EMITTED documents — the contract in ``types.py``, with teeth.

The pipeline's dict shapes are declared once as ``TypedDict``s and, until this module, enforced
only statically: a builder writing a key no TypedDict declares, or a TypedDict promising a key no
builder writes, emitted fine and drifted silently. This module makes those declarations the
runtime schema of everything the CLI writes: one pydantic ``TypeAdapter`` per emitted document
kind, applied by ``cli._dumps`` to the exact ``json_safe``'d payload about to be serialized — the
bytes on disk are the bytes that validated.

The roots below restate each command's document with its sections REQUIRED
(``types.EmittedDocument`` stays the all-``NotRequired`` assembly shape the builders write
against; a test pins each root's keys to a subset of its annotations).
``@with_config(extra="forbid")`` on a root propagates to every nested config-less TypedDict, so
an undeclared key ANYWHERE in the tree refuses — that is the drift detector. Validation is
``strict``: numeric strings and bools do not coerce into numbers (lax coercion is exactly how
drift hides), while an ``int`` into a ``float`` field remains valid.

What is deliberately NOT validated:

- ``schema`` documents — static contract prose (``dict[str, JsonValue]`` blobs); validating
  "JSON is JSON" asserts nothing.
- Error envelopes — a validation raise inside an error handler would destroy the envelope, the
  exit code and the diagnosis at once, which is the exact failure ``json_safe``'s totality
  doctrine exists to prevent.
- ``api.compile_thesis`` returns the raw IN-MEMORY summary (NaN floats, never nulls), but it
  does not return it UNCHECKED: :func:`validate_summary` runs the same strict ``RunSummary``
  adapter over the ``json_safe``'d copy at that seam, so the engine's output is verified against
  its declared shape on the library path exactly as the CLI verifies the whole document at
  emission — the copy is discarded and the caller gets the in-memory dict.

The static blobs a document embeds verbatim (``metric_roles``, ``describe_roles``) are checked by
EQUALITY against their ``contract.py`` source instead of deep-validated — for a constant, byte
identity is the stronger drift guard.

A refusal raises :class:`ReportContractError` (a ``RuntimeError``, never the pydantic
``ValidationError`` itself): the CLI's exception ladder catches ``ValidationError`` as an exit-3
``dsl_invalid`` — a statement about the CALLER's thesis — while an emitted document violating its
own contract is a seikan bug and must fall through to the exit-4 ``internal`` envelope.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import ConfigDict, TypeAdapter, ValidationError, with_config

from seikan.contract import DESCRIBE_ROLES, METRIC_ROLES
from seikan.serialize import json_safe
from seikan.types import (
    DataReport,
    FileProfile,
    GateSection,
    JsonValue,
    OutputEntry,
    RefusalStub,
    ReportIdentity,
    RunSummary,
)


class ReportContractError(RuntimeError):
    """An emitted document violated the contract ``types.py`` declares — a seikan bug.

    Deliberately a ``RuntimeError`` so it rides the CLI's catch-all to the exit-4 ``internal``
    envelope; re-raising the pydantic ``ValidationError`` would be mis-caught as the exit-3
    ``dsl_invalid`` class, which blames the caller's thesis for the verifier's own drift.
    """


@with_config(ConfigDict(extra="forbid"))
class RunReportDocument(TypedDict):
    """The ``--report-out`` document, sections required, in the fixed layer order."""

    seikan_version: str
    report_schema_version: int
    command: Literal["run"]
    identity: ReportIdentity
    data_report: DataReport
    outputs: dict[str, OutputEntry]
    summary: RunSummary
    gate: GateSection
    metric_roles: dict[str, JsonValue]


@with_config(ConfigDict(extra="forbid"))
class HashDocument(TypedDict):
    """The ``seikan hash`` stdout document: canonical identity plus the exact data-key set."""

    seikan_version: str
    report_schema_version: int
    command: Literal["hash"]
    name: str
    dsl_hash: str
    data_keys: list[str]


@with_config(ConfigDict(extra="forbid"))
class CheckDataDocument(TypedDict):
    """The ``seikan check-data`` stdout document — emitted at exit 0 AND at exit 2, so the exit-2
    emission validates too: that refusal document IS the command's document."""

    seikan_version: str
    report_schema_version: int
    command: Literal["check-data"]
    data_report: DataReport


@with_config(ConfigDict(extra="forbid"))
class DescribeDocument(TypedDict):
    """The ``seikan describe`` stdout document — like ``check-data``, emitted at both exit tiers."""

    seikan_version: str
    report_schema_version: int
    command: Literal["describe"]
    data_report: DataReport
    profiles: list[FileProfile | RefusalStub]
    describe_roles: dict[str, JsonValue]


# One adapter per document kind, built once at import — validation cost is a few milliseconds
# against an O(grid × length) run. Keyed by the ``command`` header value ``cli._base_doc`` stamps.
_ADAPTERS: dict[str, TypeAdapter[object]] = {
    "run": TypeAdapter(RunReportDocument),
    "hash": TypeAdapter(HashDocument),
    "check-data": TypeAdapter(CheckDataDocument),
    "describe": TypeAdapter(DescribeDocument),
}


@with_config(ConfigDict(extra="forbid"))
class _SummaryRoot(TypedDict):
    """The engine summary's own contract root — the ``run`` document's ``summary`` section
    wrapped so ``extra="forbid"`` propagates into every nested TypedDict, validated on its own at
    the library seam (``api.compile_thesis``) where no document exists yet."""

    summary: RunSummary


_SUMMARY: TypeAdapter[_SummaryRoot] = TypeAdapter(_SummaryRoot)


def validate_summary(summary: RunSummary) -> None:
    """Refuse an engine summary that violates :class:`~seikan.types.RunSummary`; return nothing.

    Runs over ``json_safe(summary)`` — the emitted form, where every NaN is null — under the
    same strict, extra-forbidding rules the CLI applies to the whole document, and discards the
    validated copy (the in-memory summary the caller holds is the record). A refusal is a
    :class:`ReportContractError`: the engine produced a shape it does not declare, which is a
    seikan bug and never a statement about the caller's thesis.
    """
    try:
        _SUMMARY.validate_python({"summary": json_safe(summary)}, strict=True)
    except ValidationError as exc:
        raise ReportContractError(
            "the engine summary violates the RunSummary contract declared in seikan.types — "
            f"this is a seikan bug, not an input problem: {exc}"
        ) from exc


#: The verbatim-embedded static blobs, checked by equality: (command, document key, the
#: ``contract.py`` source constant). Normalized through ``json_safe`` so a tuple-vs-list spelling
#: difference in the constant cannot read as drift.
_VERBATIM_BLOBS: dict[str, tuple[str, dict[str, JsonValue]]] = {
    "run": ("metric_roles", METRIC_ROLES),
    "describe": ("describe_roles", DESCRIBE_ROLES),
}


def validate_emitted(command: str, safe_doc: object) -> None:
    """Refuse an emitted document that violates its declared contract; return nothing.

    ``safe_doc`` must already be ``json_safe``'d — this validates the exact payload about to be
    serialized, where every NaN has become null and every Timestamp a string. The adapter's
    return value is DISCARDED on purpose: strict mode still accepts an ``int`` into a ``float``
    field by constructing a new float, and writing the validator's copy back would silently
    rewrite emitted bytes (``20`` → ``20.0``). The original document is what gets written.
    """
    adapter = _ADAPTERS.get(command)
    if adapter is None:
        raise ReportContractError(
            f"no emitted-document contract is declared for command {command!r} — "
            "validate_emitted was called on a document kind it does not know"
        )
    try:
        adapter.validate_python(safe_doc, strict=True)
    except ValidationError as exc:
        raise ReportContractError(
            f"the {command!r} document violates the emitted contract declared in "
            f"seikan.types/seikan.emitted — this is a seikan bug, not an input problem: {exc}"
        ) from exc
    blob = _VERBATIM_BLOBS.get(command)
    if blob is not None:
        key, source = blob
        emitted = safe_doc.get(key) if isinstance(safe_doc, dict) else None
        if emitted != json_safe(source):
            raise ReportContractError(
                f"the {command!r} document's {key!r} blob does not equal its contract.py "
                "source — the verbatim-embedded constant drifted between assembly and emission"
            )
