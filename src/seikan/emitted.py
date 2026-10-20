"""Runtime validation of the EMITTED documents — the contract in ``seikan.types``, with teeth.

One root per document the CLI writes. The four success roots are validated by ``cli._dumps``
against the exact ``json_safe``'d payload about to be serialized — the bytes on disk are the
bytes that validated — and ``@with_config(extra="forbid")`` on a root propagates to every nested
config-less TypedDict, so an undeclared key ANYWHERE in the tree refuses: that is the drift
detector. Validation is ``strict``: numeric strings and bools do not coerce into numbers (lax
coercion is exactly how drift hides), while an ``int`` into a ``float`` field remains valid. The
builders write each root as one literal, so the type checker holds a builder to its root too.

Shape is not the whole contract: the CSCV ``summary.pbo`` block's verdict pairing and ledgers
(:func:`_check_pbo`), which no ``TypedDict`` can state, ride the run root's summary as an
``AfterValidator`` and refuse through the same ``ValidationError`` path.

Not validated: the ``schema`` document (static contract prose — validating "JSON is JSON"
asserts nothing) and the error envelopes (a validation raise inside an error handler would
destroy the envelope, the exit code and the diagnosis at once — the failure ``json_safe``'s
totality doctrine exists to prevent). Their roots type the builders. A library caller validates
an engine summary with :func:`validate_summary`, the run root's own ``summary`` contract.

A refusal raises :class:`ReportContractError` (a ``RuntimeError``, never the pydantic
``ValidationError`` itself): the CLI's exception ladder catches ``ValidationError`` as an exit-3
``dsl_invalid`` — a statement about the CALLER's thesis — while an emitted document violating its
own contract is a seikan bug and must fall through to the exit-4 ``internal`` envelope.
"""

from __future__ import annotations

from typing import Annotated, Literal, NotRequired, TypedDict

from pydantic import AfterValidator, ConfigDict, TypeAdapter, ValidationError, with_config

from seikan.serialize import json_safe
from seikan.types import (
    DataReport,
    ErrorEnvelope,
    FileProfile,
    GateSection,
    JsonValue,
    OutputEntry,
    RefusalStub,
    ReportIdentity,
    RunSummary,
    ThresholdDoc,
)


class ReportContractError(RuntimeError):
    """An emitted document violated the contract ``seikan.types`` declares — a seikan bug.

    Deliberately a ``RuntimeError`` so it rides the CLI's catch-all to the exit-4 ``internal``
    envelope; re-raising the pydantic ``ValidationError`` would be mis-caught as the exit-3
    ``dsl_invalid`` class, which blames the caller's thesis for the verifier's own drift.
    """


def _check_pbo(summary: RunSummary) -> RunSummary:
    """The CSCV block's cross-field invariants — what a ``TypedDict`` cannot state: the verdict
    pairing (``pbo`` is null exactly when ``reason`` is set — a declined block says why, a scored
    one names no reason), the population ledger ``0 <= n_combos <= n_combos_scoreable <=
    n_combos_declared``, closed by ``n_combos_declared == n_hypotheses_attempted`` (the runner
    hands CSCV the declared cell count, one per declared combo × horizon, so the block's top
    rung IS the summary's search burden), and the per-split ledger ``0 <= n_splits <=
    n_splits_attempted``, ``0 <= n_candidates_min <= n_combos``.

    Runs AFTER the strict shape validation (every count is an ``int`` by then; none is nullable)
    on both roots that carry a summary, the ``run`` document and :class:`_SummaryRoot`, and
    holds on EVERY path ``cscv_pbo`` returns — scored (``pbo`` a fraction over at least six
    scored splits: never NaN, so never a reasonless null) or declined, a declined block
    carrying the population it established before declining. A
    violation raises ``ValueError``, which pydantic folds into the ``ValidationError`` the seam
    re-raises as :class:`ReportContractError`: an internally impossible block is a seikan bug
    (exit 4), never a statement about the caller's thesis.
    """
    pbo = summary["pbo"]
    if (pbo["pbo"] is None) != (pbo["reason"] is not None):
        raise ValueError(
            "summary.pbo verdict pairing (pbo is null exactly when reason is set) is violated: "
            f"pbo {pbo['pbo']!r} beside reason {pbo['reason']!r}"
        )
    population = (
        pbo["n_combos"],
        pbo["n_combos_scoreable"],
        pbo["n_combos_declared"],
        summary["n_hypotheses_attempted"],
    )
    n_combos, n_scoreable, n_declared, n_attempted = population
    if not 0 <= n_combos <= n_scoreable <= n_declared == n_attempted:
        raise ValueError(
            "summary.pbo population ledger 0 <= n_combos <= n_combos_scoreable <= "
            "n_combos_declared == summary.n_hypotheses_attempted is violated: "
            + " / ".join(map(str, population))
        )
    if not (
        0 <= pbo["n_splits"] <= pbo["n_splits_attempted"]
        and 0 <= pbo["n_candidates_min"] <= n_combos
    ):
        raise ValueError(
            "summary.pbo per-split ledger 0 <= n_splits <= n_splits_attempted, 0 <= "
            f"n_candidates_min <= n_combos is violated: {pbo['n_splits']} / "
            f"{pbo['n_splits_attempted']}, {pbo['n_candidates_min']} / {n_combos}"
        )
    return summary


@with_config(ConfigDict(extra="forbid"))
class RunReportDocument(TypedDict):
    """The ``--report-out`` document, sections required, in the fixed layer order."""

    seikan_version: str
    report_schema_version: int
    command: Literal["run"]
    identity: ReportIdentity
    data_report: DataReport
    outputs: dict[str, OutputEntry]
    summary: Annotated[RunSummary, AfterValidator(_check_pbo)]
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
    """The ``seikan describe`` stdout document — like ``check-data``, emitted at both exit
    tiers."""

    seikan_version: str
    report_schema_version: int
    command: Literal["describe"]
    data_report: DataReport
    profiles: list[FileProfile | RefusalStub]
    describe_roles: dict[str, JsonValue]


@with_config(ConfigDict(extra="forbid"))
class SchemaDocument(TypedDict):
    """The ``seikan schema`` stdout document: the static contract sections, in emitted order —
    the DSL's JSON Schema, the threshold knobs, the checklist contract, the output field
    dictionaries and the CSV contracts, the exit codes, and the compact role maps the run and
    describe documents stamp."""

    seikan_version: str
    report_schema_version: int
    command: Literal["schema"]
    dsl_json_schema: dict[str, JsonValue]
    thresholds: list[ThresholdDoc]
    gate_contract: dict[str, JsonValue]
    report_fields: dict[str, JsonValue]
    describe_report: dict[str, JsonValue]
    csv_format: dict[str, JsonValue]
    trades_csv: dict[str, JsonValue]
    root_series_csv: dict[str, JsonValue]
    entry_flags_csv: dict[str, JsonValue]
    exit_codes: dict[str, JsonValue]
    metric_roles: dict[str, JsonValue]
    describe_roles: dict[str, JsonValue]


@with_config(ConfigDict(extra="forbid"))
class ErrorDocument(TypedDict):
    """Every failing command's stdout document: the header, the ``error`` envelope and — on the
    exit-2 data class — the ``data_report`` that diagnoses it. ``command`` is None for a usage
    error raised before a subcommand resolved (bare ``seikan``, an unknown subcommand)."""

    seikan_version: str
    report_schema_version: int
    command: str | None
    error: ErrorEnvelope
    data_report: NotRequired[DataReport]


#: Every document the CLI writes.
type Document = (
    RunReportDocument
    | HashDocument
    | CheckDataDocument
    | DescribeDocument
    | SchemaDocument
    | ErrorDocument
)


# One adapter per document kind, built once at import — validation cost is a few milliseconds
# against an O(grid × length) run. Keyed by the document's ``command``.
_ADAPTERS: dict[str, TypeAdapter[object]] = {
    "run": TypeAdapter(RunReportDocument),
    "hash": TypeAdapter(HashDocument),
    "check-data": TypeAdapter(CheckDataDocument),
    "describe": TypeAdapter(DescribeDocument),
}


@with_config(ConfigDict(extra="forbid"))
class _SummaryRoot(TypedDict):
    """The engine summary's own contract root — the ``run`` document's ``summary`` section
    wrapped so ``extra="forbid"`` propagates into every nested TypedDict, validated on its own by
    :func:`validate_summary` where no document exists — under the same CSCV-block check
    (:func:`_check_pbo`) as the document's own ``summary``."""

    summary: Annotated[RunSummary, AfterValidator(_check_pbo)]


_SUMMARY: TypeAdapter[_SummaryRoot] = TypeAdapter(_SummaryRoot)


def validate_summary(summary: RunSummary) -> None:
    """Refuse an engine summary that violates :class:`~seikan.types.RunSummary` or the
    cross-field invariants this seam adds to it (:func:`_check_pbo`); return nothing.

    Runs over ``json_safe(summary)`` — the emitted form, where every NaN is null — under the
    same strict, extra-forbidding rules and the same CSCV-block check the CLI applies to the
    whole document, and discards the validated copy (the in-memory summary the caller holds is
    the record). A refusal is a :class:`ReportContractError`: the engine produced a summary its
    contract does not admit, which is a seikan bug and never a statement about the caller's
    thesis.
    """
    try:
        _SUMMARY.validate_python({"summary": json_safe(summary)}, strict=True)
    except ValidationError as exc:
        raise ReportContractError(
            "the engine summary violates the RunSummary contract declared in "
            f"seikan.types/seikan.emitted — this is a seikan bug, not an input problem: {exc}"
        ) from exc


def validate_emitted(command: str, safe_doc: object) -> None:
    """Refuse an emitted document that violates its declared contract; return nothing.

    ``safe_doc`` must already be ``json_safe``'d — this validates the exact payload about to be
    serialized, where every NaN has become null and every Timestamp a string. The adapter's
    return value is DISCARDED on purpose: strict mode still accepts an ``int`` into a ``float``
    field by constructing a new float, and writing the validator's copy back would silently
    rewrite emitted bytes (``20`` → ``20.0``). The original document is what gets written.
    """
    try:
        _ADAPTERS[command].validate_python(safe_doc, strict=True)
    except ValidationError as exc:
        raise ReportContractError(
            f"the {command!r} document violates the emitted contract declared in "
            f"seikan.types/seikan.emitted — this is a seikan bug, not an input problem: {exc}"
        ) from exc
