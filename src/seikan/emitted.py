"""Runtime validation of the EMITTED documents — the contract in ``seikan.types``, with teeth.

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

Shape is not the whole contract: the seam also checks the cross-field invariants no
``TypedDict`` can state, each an ``AfterValidator`` on the annotation that carries them, so it
runs after the strict shape pass and refuses through the same ``ValidationError`` path — the
CSCV ``summary.pbo`` block's verdict pairing and ledgers (:func:`_check_pbo`), on both roots
that carry a summary, and the ``describe`` profiles' explicit refusals
(:func:`_check_refusal_reasons`).

What is deliberately NOT validated:

- ``schema`` documents — static contract prose (``dict[str, JsonValue]`` blobs); validating
  "JSON is JSON" asserts nothing.
- Error envelopes — a validation raise inside an error handler would destroy the envelope, the
  exit code and the diagnosis at once, which is the exact failure ``json_safe``'s totality
  doctrine exists to prevent.
- ``api.compile_thesis`` returns the raw IN-MEMORY summary (NaN floats, never nulls), but it
  does not return it UNCHECKED: :func:`validate_summary` runs the same strict ``RunSummary``
  adapter, CSCV-block check included, over the ``json_safe``'d copy at that seam, so the
  engine's output is verified against its declared contract on the library path exactly as the
  CLI verifies the whole document at emission — the copy is discarded and the caller gets the
  in-memory dict.

The static blobs a document embeds verbatim (``metric_roles``, ``describe_roles``) are checked by
EQUALITY against their ``contract.py`` source instead of deep-validated — for a constant, byte
identity is the stronger drift guard.

A refusal raises :class:`ReportContractError` (a ``RuntimeError``, never the pydantic
``ValidationError`` itself): the CLI's exception ladder catches ``ValidationError`` as an exit-3
``dsl_invalid`` — a statement about the CALLER's thesis — while an emitted document violating its
own contract is a seikan bug and must fall through to the exit-4 ``internal`` envelope.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict, cast

from pydantic import AfterValidator, ConfigDict, TypeAdapter, ValidationError, with_config

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
    on both roots that carry a summary, the ``run`` document and the library seam's
    :class:`_SummaryRoot`, and holds on EVERY path ``cscv_pbo`` returns — scored (``pbo`` a
    fraction over at least six scored splits: never NaN, so never a reasonless null) or
    declined, a declined block carrying the population it established before declining. A
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


def _check_refusal_reasons(
    profiles: list[FileProfile | RefusalStub],
) -> list[FileProfile | RefusalStub]:
    """Refusals are explicit, never omitted: on a block that did not refuse whole (``reason``
    null) a ``diff`` read is never null — ``diff`` rides every such block, and one with no double
    to be refuses the block by name (``not_representable``) — and a ratio read (``pct``/``log``,
    a range distance's ``pct``, ``drawdown_pct``/``runup_pct``, ``last_to_mean``) that is null
    must carry the ``ratio_reason`` saying why — for ``log`` only ``non_positive_endpoint``, the
    scale-safe log never refusing as ``not_representable``. A reasonless null reads as "nothing
    to report" — exactly how a ``pct`` or a ``diff`` past the double range used to ride out —
    and the per-field TypedDicts cannot state a cross-field rule, so the seam does. A violation is a
    builder bug in ``describe.py``: the ``ValueError`` here surfaces as the pydantic
    ``ValidationError`` :func:`validate_emitted` re-raises as :class:`ReportContractError`."""
    for profile in profiles:
        if "series" not in profile:  # a refused file's stub carries no blocks
            continue
        admitted = cast("FileProfile", profile)
        # (where, reason, ratio_reason, diff reads, ratio reads) per block
        reads: list[
            tuple[str, str | None, str | None, tuple[float | None, ...], tuple[float | None, ...]]
        ] = []
        for column, sp in admitted["series"].items():
            for kind, panel in (("changes", sp["changes"]), ("dispersion", sp["dispersion"])):
                for w, ch in panel.items():
                    if (
                        ch["reason"] is None
                        and ch["log"] is None
                        and (ch["ratio_reason"] == "not_representable")
                    ):
                        raise ValueError(
                            f"{admitted['path']}: {column}.{kind}[{w}] carries a null log under "
                            "ratio_reason not_representable — log is scale-safe and refuses only "
                            "off the positive domain"
                        )
                    reads.append(
                        (
                            f"{column}.{kind}[{w}]",
                            ch["reason"],
                            ch["ratio_reason"],
                            (ch["diff"],),
                            (ch["pct"], ch["log"]),
                        )
                    )
            for w, rp in sp["range_position"].items():
                legs = (rp["from_high"], rp["from_low"])
                reads.append(
                    (
                        f"{column}.range_position[{w}]",
                        rp["reason"],
                        rp["ratio_reason"],
                        tuple(None if d is None else d["diff"] for d in legs),
                        tuple(None if d is None else d["pct"] for d in legs),
                    )
                )
            fs = sp["full_sample"]
            reads.append(
                (
                    f"{column}.full_sample",
                    fs["reason"],
                    fs["ratio_reason"],
                    (fs["drawdown_diff"], fs["runup_diff"]),
                    (fs["drawdown_pct"], fs["runup_pct"]),
                )
            )
        if admitted["volume"] is not None:
            for w, vw in admitted["volume"]["windows"].items():
                reads.append(
                    (f"volume[{w}]", vw["reason"], vw["ratio_reason"], (), (vw["last_to_mean"],))
                )
        for where, reason, ratio_reason, diffs, ratios in reads:
            if reason is not None:  # refused whole: its nulls stand under the structural reason
                continue
            if any(v is None for v in diffs):
                raise ValueError(
                    f"{admitted['path']}: {where} carries a null diff on a block that did not "
                    "refuse whole — a diff with no double to be refuses its block by reason"
                )
            if ratio_reason is None and any(v is None for v in ratios):
                raise ValueError(
                    f"{admitted['path']}: {where} carries a null ratio read with no "
                    "ratio_reason — a refusal must be explicit, never omitted"
                )
    return profiles


@with_config(ConfigDict(extra="forbid"))
class DescribeDocument(TypedDict):
    """The ``seikan describe`` stdout document — like ``check-data``, emitted at both exit tiers;
    its ``profiles`` additionally pass :func:`_check_refusal_reasons`."""

    seikan_version: str
    report_schema_version: int
    command: Literal["describe"]
    data_report: DataReport
    profiles: Annotated[list[FileProfile | RefusalStub], AfterValidator(_check_refusal_reasons)]
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
    the library seam (``api.compile_thesis``) where no document exists yet — under the same
    CSCV-block check (:func:`_check_pbo`) as the document's own ``summary``."""

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
