"""The per-cell checklist — a completeness/support/concentration result about one engine summary.

``evaluate_gate`` grades EVERY declared parameter × horizon cell INDEPENDENTLY and reports each
check (``{name, met, observed, threshold, detail}``). There is no short-circuit and
no scalar verdict: a caller sees the complete checklist for every cell, exactly as it sees the
complete grid — the gate renders a per-cell result about the summary, it NEVER filters it.

The checks read the engine summary directly (the dict ``compile_thesis`` produces). The
``POLICY_VERSION`` contract:

- **The checklist is NON-INFERENTIAL.** A cell's ``met`` makes NO significance claim and
  certifies NO positive expected return. It says exactly this much: the cell's evidence is
  completely measured (every firing accounted for, every decision bar decidable, every raw
  decision input available), it clears the raw support floors on full-sample evidence, and its
  return mass is not one episode. Nothing here is a test — the nominal per-cell statistics
  (``rot_p``, ``t_hac``, ``hac_se``, ``pbo``, the episode and bucket panels) ride along as
  EVIDENCE and no check reads them.
- **One stamp selects between two rubrics.** The summary's ``target_mode`` stamp names the
  target semantics every cross-target read is graded under: ``conjunction`` (targets are the
  thesis's regime — per-target floors and ceilings, the weakest target decides) or
  ``basket`` (the members form ONE evidence pool — ``support`` reads the
  pooled floors, ``concentration`` reads the pooled top share plus the episode-cluster and
  member-mass ceilings, ``cell_evidence`` reconciles the pooled panel against the member
  panels). A missing or unreadable stamp refuses fail-closed everywhere it would have
  dispatched — grading under an assumed mode is the stamp-stripping bypass one field over. The
  coverage contracts stay per-target in BOTH modes: a hole in one member corrupts every
  member's cross-sectional reads, so per-member fail-closed is structurally required even in a
  basket.
- **Selection and multiplicity belong to the CALLER.** The engine measures and reports; it does
  not select, rank, or crown a winner. Cross-cell multiplicity is the caller's to price against
  the stamped ``n_hypotheses_attempted`` (which ``search_cap`` bounds), and cross-RUN search
  discipline is beyond a stateless verifier entirely — the identity layer (``dsl_hash``,
  ``data_digests``) makes each distinct exam visible so the caller can price it.
- **The missingness contract fails closed, on BOTH sides.** ``outcome_coverage`` reads each
  cell's per-target censoring ledger and refuses any ``no_outcome``/``no_benchmark`` firing — a
  data hole that deletes outcomes (a vendor outage, a stale feed, an adversarial file) can hide
  adverse results, so a graded cell must be completely measured. ``signal_coverage`` closes the
  same hole on the DECISION side: the outcome ledger can only account for bars that FIRED, so a
  missing input that suppresses a firing leaves no trace there — the runner's three-valued ledger
  counts post-warmup UNDECIDABLE decision bars and any of them refuses. The run-level
  ``source_coverage`` closes what the root three-valued channel structurally cannot see (Kleene
  absorption, a NaN-skipping recursive kernel laundering contaminated state into a finite value)
  by counting availability of the RAW decision leaves. Together: deleting data can only ever
  REFUSE.
- **Right-censoring at the data end is structural and ALLOWED.** With no holdout there is no
  embargo and no tail, so a trailing window running past the last bar is geometry, not a hole:
  ``open`` firings are legitimate everywhere. In-bounds NaN legs remain
  ``no_outcome``/``no_benchmark`` and still refuse.
- **Structural discipline is universal**: one concentration ceiling over every regime target's
  |return|-mass top share AND the cell's largest merged cross-target episode cluster (a
  one-episode "edge" refuses), and one cap on the DECLARED grid (``n_hypotheses_attempted`` —
  non-firing combos cannot shrink it).
- **The evidence contract is stamped and verified.** ``evidence_complete`` requires the summary's
  ``statistics_version`` to equal this build's, ``gate_evidence_basis == "full_sample"``, an
  EXPLICIT ``outcome`` stamp — the ``{series, kind}`` dict the runner always
  writes; a null stamp refuses — string-typed
  targets and panel keys, and a ``cells`` panel holding EXACTLY the declared grid — a report
  missing declared cells is drifted input, not evidence. Per cell, ``cell_evidence`` additionally
  requires the panels to AGREE: the graded ``n`` equals the ledger's closed count, ``n_nonoverlap
  <= n``,
  the episode panel's ``n`` is the per-target total, and the signal ledger spans the whole index.
  An internally impossible summary is drifted input, not something to grade.
- **Strict numeric hygiene**: every read collapses NaN, ±inf, and non-integral counts to a
  refusal — drifted input can never sail past a comparison or crash the gate.
- **Commensurability guard**: a ``diff``-outcome, multi-target run refuses the cross-target
  cluster-mass read — level-unit returns from different series cannot be mass-compared.
- **Fail closed, never crash.** An unreadable ``cells`` panel leaves ``evidence_complete`` unmet
  and yields zero graded cells; a malformed individual cell entry gets an unmet ``cell_evidence``
  while its siblings grade normally. Drifted input refuses with a detail; it never raises.

``canonical_dsl_hash`` is the identity discipline: sha256 over the defaults-filled, key-sorted
DSL, so omitted defaults and explicit-default forms hash identically and a report is bound to
exactly the rules it validated.

The package is layered: ``_read`` holds the strict readers over drifted input, ``_checks_run``
and ``_checks_cell`` the checks, ``_evaluate`` the checklist driver, ``_model`` the policy
version and result types, ``_hash`` the canonical DSL identity. This facade is the public
surface.
"""

from seikan.gate._evaluate import evaluate_gate
from seikan.gate._hash import canonical_dsl_hash
from seikan.gate._model import POLICY_VERSION, CellReport, GateCheck, GateReport
from seikan.settings import GateThresholds

__all__ = [
    "POLICY_VERSION",
    "CellReport",
    "GateCheck",
    "GateReport",
    "GateThresholds",
    "canonical_dsl_hash",
    "evaluate_gate",
]
