"""Gate thresholds — the only configuration a stateless verifier has.

Four knobs (``thesis_min_trades``, ``thesis_min_n_nonoverlap``, ``thesis_max_concentration``,
``thesis_max_hypotheses``) parameterize the per-cell checklist. Environment variables (prefix
``SEIKAN_``, e.g. ``SEIKAN_THESIS_MIN_TRADES``) set the defaults; explicit CLI flags override
them per run. There is deliberately no config file and no home directory — the thresholds
actually used are snapshotted into every report, so a run is fully described by its inputs.

Four sealing rules (the policy is stamped AND bounded):

- **The canonical exam is the floor** — every knob's domain admits only its class default or
  STRICTER (``thesis_min_trades ≥ 30``, ``thesis_max_concentration ≤ 0.6``, …). A looser exam
  is rejected at construction time (CLI exit 3, ``thresholds_invalid``): a cell reported as
  ``met`` always means at-least-canonical rigor, and the party being graded cannot bend the
  checklist it is graded by. Nothing is lost for exploration — an unmet cell still carries its
  complete statistics, so every number is readable regardless of its checklist result.
- **Every numeric field is domain-bounded** — a nonsense exam (a negative sample floor, a
  concentration ceiling above 1) can never construct.
- **The ``SEIKAN_`` env namespace is owned** (``extra="forbid"``): an unknown ``SEIKAN_*``
  variable — typically a typo'd threshold var the caller believes is active — refuses the run
  instead of silently falling back to defaults, in any letter case, since the knobs themselves
  are read in any (``seikan_thesis_min_trades`` sets ``thesis_min_trades``).
- **The instance is FROZEN**: the floors above are constructor-time validation, so a mutable
  settings object would let a library caller loosen the exam after it had been checked.
  Assignment raises.

There is exactly ONE checklist, applied identically to every declared cell: no profiles, no
per-cell exemptions, nothing the caller can select into. What a passing cell asserts is
completeness, support and non-concentration — never significance — so these knobs size the
evidence a cell must carry, not the confidence anyone may claim from it.
"""

from __future__ import annotations

import os

from pydantic import ConfigDict, Field, TypeAdapter, model_validator, with_config
from pydantic_settings import BaseSettings, SettingsConfigDict

from seikan.constants import MAX_DECLARED_GRID
from seikan.types import ThresholdsSnapshot

#: The environment namespace the knobs are read from. pydantic-settings matches a variable's
#: name in any letter case, by lower-casing it; :func:`env_var_is_set` and the namespace check
#: below compare the same way.
ENV_PREFIX = "SEIKAN_"


def env_var(field: str) -> str:
    """A knob's environment variable, as the docs and ``seikan schema`` spell it."""
    return f"{ENV_PREFIX}{field.upper()}"


def env_var_is_set(field: str) -> bool:
    """True when the knob's variable is set, in any letter case."""
    name = env_var(field).lower()
    return any(k.lower() == name for k in os.environ)


@with_config(ConfigDict(extra="forbid"))
class _SealedSnapshot(ThresholdsSnapshot):
    """``ThresholdsSnapshot`` with an extra key refused (a plain TypedDict drops it)."""


#: The snapshot shape's validator — the seam between the settings model and the emitted
#: ``identity.thresholds`` stamp, checked at every ``snapshot()``.
_SNAPSHOT: TypeAdapter[_SealedSnapshot] = TypeAdapter(_SealedSnapshot)


class GateThresholds(BaseSettings):
    """The per-cell checklist's knobs (see ``seikan.gate`` for what each check does)."""

    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="forbid", frozen=True)

    #: Floor on a target's raw (overlapping) closed-observation count within a cell. Canonical
    #: floor: only equal-or-higher constructs.
    thesis_min_trades: int = Field(default=30, ge=30)
    #: Floor on non-overlapping observations per target within a cell. Overlapping forward
    #: returns inflate the raw count, so a cell can clear ``thesis_min_trades`` on a handful of
    #: distinct market episodes; ``n_nonoverlap`` is the greedy non-overlapping count —
    #: deliberately not called independent, which non-overlap does not certify.
    #: Canonical floor: only equal-or-higher constructs.
    thesis_min_n_nonoverlap: int = Field(default=8, ge=8)
    #: Ceiling on |return|-mass concentration, applied within each cell to EVERY regime target's
    #: ``concentration.top_share_abs`` (top-5% share) AND to that cell's episode panel's
    #: ``max_cluster_share_abs`` (largest merged cross-target episode cluster). Refuses a
    #: one-episode "edge". Non-optional by construction. Canonical ceiling: only equal-or-lower
    #: constructs.
    thesis_max_concentration: float = Field(default=0.6, gt=0.0, le=0.6, allow_inf_nan=False)
    #: Ceiling on the DECLARED grid size (``n_hypotheses_attempted`` — non-firing combos cannot
    #: shrink it). Every declared cell is measured and reported independently, so the grid is
    #: exactly the multiplicity the CALLER must price its own selection against; the cap bounds
    #: how wide a search a single run may declare at all. Non-optional by construction.
    #: Canonical ceiling: only equal-or-lower constructs.
    thesis_max_hypotheses: int = Field(default=MAX_DECLARED_GRID, ge=1, le=MAX_DECLARED_GRID)

    @model_validator(mode="after")
    def _reject_unknown_env(self) -> GateThresholds:
        # The SEIKAN_ namespace is owned: pydantic-settings only READS declared fields from the
        # environment, so ``extra="forbid"`` alone never sees a typo'd var (SEIKAN_MAX_P_VALUE,
        # missing THESIS_), whose caller believes a different exam is active than the one that
        # runs — refuse it. ``seikan_thesis_min_trade`` is as typo'd as its upper-case spelling.
        known = {env_var(name) for name in type(self).model_fields}
        known_lower = {name.lower() for name in known}
        unknown = sorted(
            k
            for k in os.environ
            if k.lower().startswith(ENV_PREFIX.lower()) and k.lower() not in known_lower
        )
        if unknown:
            raise ValueError(
                f"unknown {ENV_PREFIX}* environment variable(s): {', '.join(unknown)} — the "
                f"{ENV_PREFIX} namespace belongs to the gate thresholds (a typo'd var would "
                "silently run a different exam than intended); known vars: "
                + ", ".join(sorted(known))
            )
        return self

    def snapshot(self) -> ThresholdsSnapshot:
        """The thresholds actually used, for the report's provenance section. A pure,
        reconstructible config: ``GateThresholds(**snapshot())`` round-trips."""
        # The dump is DERIVED from the model rather than written out field by field, so a knob
        # can never be added to the exam and left out of the stamp — and it is VALIDATED against
        # the sealed snapshot shape rather than cast to it, so a knob added here without its twin
        # in ``ThresholdsSnapshot`` (or the reverse) refuses instead of drifting.
        return _SNAPSHOT.validate_python(self.model_dump(), strict=True)

    def is_canonical(self) -> bool:
        """True when every field equals its class default — the canonical exam, no env/CLI
        overrides. Compared against ``model_fields`` defaults (never a live env-reading
        instance), so a polluted environment cannot fake canonicity. Exact equality only: a
        *stricter* override is still non-canonical (consumers diff the snapshot for
        direction)."""
        return all(
            getattr(self, name) == field.default for name, field in type(self).model_fields.items()
        )
