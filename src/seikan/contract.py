"""The CLI's static contract payloads — data only, no logic.

Everything ``seikan schema`` emits about the contract lives here: the compact ``METRIC_ROLES``
map (stamped identically into every run report), its prose rationale, the gate-contract and
CSV-contract references, and the exit-code meanings. A leaf module like ``constants.py``: it
imports nothing from the package, and only ``cli.py`` imports it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only — nothing here is imported at runtime, leaf module intact
    from seikan.types import JsonValue

#: The compact role map stamped into every run report: exactly what a passing cell claims
#: (and what the exit code does NOT claim), which summary fields each check consumes, and what is
#: evidence-only — so a calling agent never mistakes an uncalibrated evidence panel for an
#: inferential result, and never reads the exit code as a verdict. One line each; the full prose
#: rationale lives in ``seikan schema`` (``METRIC_ROLES_DOC``).
METRIC_ROLES: dict[str, JsonValue] = {
    "claim": (
        "exit 0 certifies ONLY that the run finished and every nominated output was written (this "
        "report, being one of them, is complete) — it is not a "
        "verdict and says nothing about any cell. Per-cell `met` is a completeness / support / "
        "concentration checklist over full-sample evidence, with NO significance claim and NO "
        "positive-expected-return certification: it asserts that the cell's evidence is fully "
        "measured (every firing accounted for, every decision bar decidable, every raw decision "
        "input available), that it clears the raw support floors, and that its return mass is not "
        "one episode. Nothing here is a test. Selection among cells and cross-cell multiplicity "
        "are the CALLER's, priced against n_hypotheses_attempted"
    ),
    "run_checks": {
        "evidence_complete": [
            "statistics_version",
            "gate_evidence_basis",
            "targets",
            "outcome (an explicit {series, kind} dict — a null stamp refuses)",
            "target_mode ('conjunction' | 'basket' — the stamp selects the rubric; missing or "
            "garbage refuses, and basket with <2 targets or a diff outcome refuses as drifted "
            "input)",
            "n_hypotheses_attempted",
            "n_bars",
            "cells (a list with len(cells) == n_hypotheses_attempted)",
            "sources (string-keyed, covering targets exactly)",
        ],
        "source_coverage": [
            "sources[*].{n_bars,n_missing}",
            "sources[*].by_source[*].{n_missing,first_available} (a null first_available — an "
            "input that never became available — refuses; a late START is warmup and stays "
            "evidence)",
            "n_bars (the geometry every availability count is verified against)",
        ],
        "search_cap": [
            "n_hypotheses_attempted",
        ],
    },
    "cell_checks": {
        "cell_evidence": [
            "cells[*].params",
            "cells[*].by_target[*].{n,n_nonoverlap}",
            "cells[*].outcome_coverage[*].{n_attempted,n_closed,exit_reasons}",
            "cells[*].signal_coverage[*].n_bars",
            "cells[*].episode_stats.n",
            "cells[*].pooled.{n,n_nonoverlap} (basket only — pooled.n == sum of by_target.n, "
            "pooled.n_nonoverlap <= pooled.n, pooled.n_nonoverlap <= n_bars, pooled.n <= n_bars × "
            "targets; "
            "a pooled key on a conjunction cell REFUSES as a restamped basket)",
            "targets + n_bars + target_mode (the cross-panel reconciliation references)",
        ],
        "outcome_coverage": [
            "cells[*].outcome_coverage[*].exit_reasons.{no_outcome,no_benchmark} "
            "('open' is ALLOWED at any count — end-of-data right-censoring is structural; "
            "per-target in BOTH modes)",
        ],
        "signal_coverage": [
            "cells[*].signal_coverage[*].{n_bars,n_undefined} (per-target in BOTH modes)",
        ],
        "support": [
            "conjunction: cells[*].by_target[*].{n,n_nonoverlap,mean_ret} (the weakest target "
            "decides)",
            "basket: cells[*].pooled.{n,n_nonoverlap,mean_ret} (one pool — no member is examined "
            "alone)",
        ],
        "concentration": [
            "conjunction: cells[*].by_target[*].concentration.top_share_abs",
            "basket: cells[*].pooled.concentration.top_share_abs + "
            "cells[*].pooled.member_share.max_member_share_abs (the one-name-basket detector)",
            "cells[*].episode_stats.max_cluster_share_abs (both modes)",
            "outcome.kind + targets (diff-outcome multi-target incommensurability guard)",
        ],
    },
    "evidence_only": [
        "per-cell rot_p / rotation (circular-shift null — anti-conservative under signal-aligned "
        "volatility regimes)",
        "t_hac / hac_se (event-time overlap HAC — understates the SE on overlapping pools)",
        "summary.pbo block (pbo, reason, n_splits, n_splits_attempted, n_candidates_min, "
        "n_combos, n_combos_scoreable, n_combos_declared, blocks, lambda_mean, "
        "oos_degradation_slope, oos_degradation_slope_reason, prob_oos_loss — CSCV over the "
        "scoreable grid, with the population ledger n_combos <= n_combos_scoreable <= "
        "n_combos_declared and the per-split ledger n_splits <= n_splits_attempted, "
        "n_candidates_min <= n_combos)",
        "per-cell by_target hit_rate / win_loss_ratio / std_ret / skewness / kurtosis / "
        "tail_ratio / cvar_5 (descriptive shape — the pool_moments reads over the cell's own "
        "closed pool, mounted on the by_target and pooled panels; hit_rate is the share of "
        "returns > 0, a zero return is not a hit; win_loss_ratio counts zeros on neither side)",
        "per-cell profit_factor (gross win mass over gross loss mass — Σwins/|Σlosses|; NaN, "
        "never infinity, when either side is empty; see caveats.profit_factor)",
        "per-cell by_target mean_ret_raw / mean_ret_bench (the excess mean's own legs — "
        "mean_ret ≈ mean_ret_raw − mean_ret_bench over the same closed pool; unbenchmarked "
        "runs: mean_ret_raw equals mean_ret and mean_ret_bench is null; see "
        "caveats.mean_ret_raw)",
        "per-cell by_target benchmark_regression ({n, beta, alpha, r2, reason} — per-window OLS "
        "attribution of the raw leg on the bench leg, never annualized; alpha + beta·mean_bench "
        "== mean_ret_raw is a re-checkable identity; null fields + reason when unbenchmarked, "
        "thin or degenerate; see caveats.benchmark_regression)",
        "per-cell by_target boot (episode-bootstrap CI {method, ci_level, n_boot, n_episodes, "
        "ci_lo, ci_hi, boot_se, reason} — the dependence-robust counterweight to t_hac/rot_p; "
        "see caveats.boot)",
        "per-cell by_target subperiods (three equal-bar eras [{start, end, n, mean_ret}] — "
        "descriptive era visibility; nothing selects on it)",
        "per-cell by_target ret_quantiles {p05,p10,p25,p50,p75,p90,p95} + worst_ret / best_ret "
        "(closed-return order statistics — the typical-observation read mean_ret cannot give; "
        "p05 doubles as the historical VaR(5%) cvar_5 tails off; see caveats.ret_quantiles)",
        "per-cell by_target mae_quantiles / mfe_quantiles ({n, mean, p05..p95, worst | best} — "
        "RAW post-entry excursion statistics over the full H/L of [fill, fill+h-1] plus the "
        "exit open, each block over its own finite subset; see caveats)",
        "per-cell by_target edge_ratio (mean RAW MFE over |mean RAW MAE| — the one sanctioned "
        "excursion ratio, both legs RAW; see caveats.edge_ratio)",
        "per-cell by_target timing ({n_to_positive, median_bars_to_positive, n_to_trough, "
        "median_bars_to_trough} — the timing pair aggregated the way the excursion pair is; "
        "survivors-only medians, censored at h; see caveats.timing)",
        "per-cell episode_profile (the episode-deduplicated TWIN of the row-level pool "
        "statistics — the same frozen cross-target merge as episode_stats, per-episode mean "
        "ret and extreme excursions, then the same statistic family over episodes incl. "
        "max_win_streak / max_loss_streak; n_episodes == episode_stats.n_clusters by "
        "construction; row-vs-episode divergence is the visible cluster diagnostic, reported "
        "never corrected; see caveats.episode_profile)",
        "bar_spacing (run-level {min,median,max}_seconds — the index's clock geometry)",
        "summary.cross_breadth (per cross node × combo: the per-bar entering-member count k the "
        "cross kernels reduced over — finite input, and under where/group eligible and finitely "
        "labelled — summarized: n_bars_evaluated, n_bars_below_full, "
        "k_min/k_median/k_max, first_full_bar. The effective-universe ledger: member warmup "
        "legally thins k, and this panel is what makes the thinning visible; [] outside "
        "basket)",
        "episode_stats beyond the two fields the checklist reads (n, max_cluster_share_abs)",
        "summary.baseline (run-level unconditional base rate per horizon × target — plus a "
        "pooled row in basket; the conditional-vs-base-rate comparison is the caller's, and "
        "no uplift field ever exists)",
        "per-cell episodes (the time-ordered episode ledger under episode_stats — bounded at "
        "its stated cap, with mass-conserving truncation counts)",
        "per-cell conditional_buckets / bucket_monotonicity (descriptive conditioning over "
        "the cell's own rows — bucket edges per TARGET, records aggregated by ordinal — never "
        "pooled across cells)",
        "per-cell feature_association (Spearman rho per cell × feature × target — per-target "
        "in BOTH modes, deliberately no p-value)",
        "per-cell pooled beyond the fields the checklist reads (mean_ret_raw, mean_ret_bench, "
        "benchmark_regression, hit_rate, win_loss_ratio, profit_factor, std_ret, skewness, "
        "kurtosis, tail_ratio, cvar_5, t_hac, hac_se, rot_p, boot, subperiods, ret_quantiles, "
        "worst_ret, best_ret, mae_quantiles, mfe_quantiles, edge_ratio, timing — basket "
        "evidence riders; none gate. Under cross_mean the pooled mean_ret is the FIRING "
        "subset's cross-sectional selection tilt — each closed bar's FULL cross-section demeans "
        "to zero, but the pool holds only the members that fired, so ≈ 0 appears only when "
        "firings are basket-wide — while the legs still attribute: mean_ret_bench ≈ "
        "mean_ret_raw is the basket's own drift)",
        "per-cell pooled.member_share.by_target (the full member-mass decomposition — "
        "attribution, never a ranking; only max_member_share_abs gates)",
    ],
    # ONE machine-readable class per reported metric, so a consuming agent can tell what KIND of
    # number it is holding without parsing prose: `descriptive` describes the realized sample and
    # certifies nothing; `inference` is an uncalibrated inferential estimator (directionally
    # informative, never significance); `integrity` is an accounting/reconciliation ledger whose
    # job is to make missing or tampered evidence visible — the rotation null's resolution stamp
    # and a cell's `rot_n_null` are integrity by that rule, not inference. The class is ORTHOGONAL
    # to gating: `mean_ret` and `concentration` are descriptive quantities a check happens to
    # read, and integrity is reserved for ledgers, not for "the gate looks at it". `n` /
    # `n_nonoverlap` are integrity by that rule — counts whose JOB is visibility:
    # `n_nonoverlap` is the
    # greedy non-overlapping count, carried precisely so the overlap inflation (n=400,
    # n_nonoverlap=18)
    # cannot hide — the same accounting job as `n`, and NOT an inferential estimate, though the
    # reliability layer sets its df from it. A BLOCK classes as a
    # whole; the one dotted override is `episode_stats.n`, the reconciliation count
    # `cell_evidence` re-checks, while `episode_stats` itself (the cluster profile) stays
    # descriptive.
    "metric_classes": {
        # inference — uncalibrated inferential estimators (see each one's caveat)
        "rot_p": "inference",
        "t_hac": "inference",
        "hac_se": "inference",
        "boot": "inference",
        "pbo": "inference",
        # integrity — accounting and reconciliation ledgers
        "n": "integrity",
        "n_nonoverlap": "integrity",
        "rot_n_null": "integrity",
        "rotation": "integrity",
        "exit_reasons": "integrity",
        "outcome_coverage": "integrity",
        "signal_coverage": "integrity",
        "sources": "integrity",
        "cross_breadth": "integrity",
        "episode_stats.n": "integrity",
        "data_digests": "integrity",
        "n_hypotheses_attempted": "integrity",
        "n_bars": "integrity",
        # descriptive — realized-sample description, certifying nothing
        "mean_ret": "descriptive",
        "mean_ret_raw": "descriptive",
        "mean_ret_bench": "descriptive",
        "std_ret": "descriptive",
        "hit_rate": "descriptive",
        "win_loss_ratio": "descriptive",
        "profit_factor": "descriptive",
        "skewness": "descriptive",
        "kurtosis": "descriptive",
        "tail_ratio": "descriptive",
        "cvar_5": "descriptive",
        "worst_ret": "descriptive",
        "best_ret": "descriptive",
        "ret_quantiles": "descriptive",
        "mae_quantiles": "descriptive",
        "mfe_quantiles": "descriptive",
        "edge_ratio": "descriptive",
        "concentration": "descriptive",
        "member_share": "descriptive",
        "episode_stats": "descriptive",
        "episodes": "descriptive",
        "subperiods": "descriptive",
        "baseline": "descriptive",
        "conditional_buckets": "descriptive",
        "bucket_monotonicity": "descriptive",
        "feature_association": "descriptive",
        "bar_spacing": "descriptive",
        "pooled": "descriptive",
        "benchmark_regression": "descriptive",
        "episode_profile": "descriptive",
        "timing": "descriptive",
    },
    # One honest sentence per number a reader is likely to over-trust, TRAVELLING WITH the
    # report: the compact roles say WHETHER a field gates (none of these do); the caveats say
    # WHY quoting it as certification would mislead. The long mechanics live in
    # ``METRIC_ROLES_DOC`` (schema-only).
    "caveats": {
        "rot_p": (
            "assumes shift-exchangeability — over-certifies when volatility clusters where the "
            "signal fires; one-sided, floored at the cell's own 1/(1+rot_n_null); the pooled "
            "rot_p is a COMMON-SHIFT null with the same caveats; never quote as significance"
        ),
        "t_hac": (
            "(and hac_se) the Bartlett taper understates the SE on overlapping pools — "
            "anti-conservative, ~10-12% rejection at nominal 5% under an iid-innovation null"
        ),
        "boot": (
            "the episode bootstrap assumes episode INDEPENDENCE; adjacent episodes still "
            "correlate through volatility regimes — less anti-conservative than t_hac, not "
            "calibrated"
        ),
        "pbo": (
            "CSCV over the SCOREABLE grid — only combos that fired with a closed pool enter "
            "(see the population ledger); in-block Sharpes are overlap-inflated, S can fall to "
            "6 dependent splits, and the per-split candidate count thins with block-local data "
            "(n_splits_attempted, n_candidates_min)"
        ),
        "cross_breadth": (
            "k counts ENTERING members per bar (finite input, and under where/group eligible "
            "and finitely labelled) — member warmup or a screen legally thins it (a late start "
            "is warmup, not a hole), so k_min < len(targets) states coverage, not a defect; "
            "below min_valid the node emitted no cross-section at all"
        ),
        "mean_ret": (
            "in-sample full-sample descriptive, gross of costs — no holdout and no deflation; "
            "read it against ret_quantiles.p50, because a positive mean over a negative median "
            "is a pool a few spikes carried, not a typical outcome"
        ),
        "concentration": (
            "top-5% |return|-mass share with k = max(1, ceil(0.05·n)); at n <= 20 the top set "
            "is a single observation, so thin pools read structurally elevated"
        ),
        "ret_quantiles": (
            "linear-interpolated order statistics of the closed pool — below n≈20 the outer "
            "percentiles rest on one or two observations, and overlap smears one market move "
            "across ~h rows, so these are not independent-draw estimates"
        ),
        "mae_quantiles": (
            "RAW-path excursions incl. the block's mean, never benchmark-adjusted (unlike ret "
            "when benchmarked); overlapping windows share one trough, so a single crash sets the "
            "mae of ~h neighbouring rows and the tail percentiles are not independent events"
        ),
        "mfe_quantiles": (
            "RAW-path interim marks incl. the block's mean, never benchmark-adjusted and never "
            "attainable exits (this engine has no exit rule); overlapping windows share one "
            "peak, so the tail percentiles are not independent events"
        ),
        "mean_ret_raw": (
            "(and mean_ret_bench) the excess mean's own legs, direction-signed, over the same "
            "closed pool — attribution, not two extra hypotheses; unbenchmarked: mean_ret_raw "
            "equals mean_ret and mean_ret_bench is null ('no benchmark', never zero); "
            "in-sample, gross of costs like mean_ret"
        ),
        "edge_ratio": (
            "mean RAW MFE over |mean RAW MAE|, over PAIRED rows (both legs finite — so under "
            "asymmetric holes it need not equal the quantile blocks' means' ratio); "
            "UNNORMALIZED, so never compare it across instruments; MFE is a mark, not an "
            "attainable exit"
        ),
        "profit_factor": (
            "Σwins/|Σlosses| over closed OVERLAPPING returns — one move enters the sums ~h "
            "times and no equity curve exists, so this is not a realizable P&L ratio; zero "
            "rows join neither side; null (never infinity) when either side is empty; "
            "≈ (n_wins/n_losses) × win_loss_ratio, derivable"
        ),
        "baseline": (
            "in-sample base rates over every anchor bar in the cells' own algebra/benchmark/"
            "direction — quote a conditional mean AGAINST it; uplift is never derived; under "
            "'cross_mean' the pooled BASELINE mean is ~0 by construction (unlike the pooled "
            "CELL mean, firing rows only: the selection tilt)"
        ),
        "episodes": (
            "time-ordered, never ranked by share, and BOUNDED: past the cap the ledger "
            "truncates visibly (n_omitted, omitted_share_abs), so any count read off it is a "
            "floor, not a total — reconcile against episode_stats.n_clusters"
        ),
        "conditional_buckets": (
            "per-cell, bucket edges per TARGET (pooled edges would conflate member levels with "
            "time variation); qcut over overlap-inflated rows — 'associated in this sample', "
            "never 'predicts' — and never pool across cells: a cross-cell pool's conditioning "
            "depends on grid composition"
        ),
        "feature_association": (
            "Spearman rho over overlapping rows, no p-value on purpose (an overlap-inflated p "
            "is the over-trustable number); per-target in BOTH modes — a pooled rho would "
            "conflate member levels with time variation; ~h rows per episode share one "
            "outcome, so a rank pattern can be a few episodes wearing many rows"
        ),
        "pooled": (
            "(bar × member) rows — one market move smears across members AS WELL AS across ~h "
            "overlapping horizons, so pooled n overstates the independent information TWICE; "
            "read pooled.n_nonoverlap and the cluster share before quoting any pooled count"
        ),
        "member_share": (
            "a full decomposition of the pooled |return| mass, never a ranking — by_target is "
            "attribution, not a verdict about any member; a 2-member basket reads "
            "structurally elevated (its larger member always carries >= 0.5)"
        ),
        "win_loss_ratio": (
            "average win over average loss — the payoff asymmetry beside hit_rate's frequency; "
            "zero returns join neither side (hit_rate counts them as non-hits), null (never "
            "infinity) when either side is empty, and overlap smears one move into both sides "
            "of the ratio"
        ),
        "std_ret": (
            "dispersion of OVERLAPPING rows — understates independent-information dispersion "
            "exactly as n overstates n_nonoverlap; a description of this pool, never a risk "
            "estimate"
        ),
        "skewness": (
            "(and kurtosis — Pearson, normal = 3) scipy's POPULATION moments (bias=True, not "
            "pandas' unbiased G1/G2) over an overlapping pool: one smeared move manufactures "
            "apparent tail weight, so read both as directional, not exact"
        ),
        "tail_ratio": (
            "≡ |p95/p05| of ret_quantiles — derivable, unstable when p05 sits near zero, and a "
            "spread ratio rather than a tail read unless p05 < 0 < p95; on thin pools both "
            "tails rest on one or two rows"
        ),
        "cvar_5": (
            "mean of the observations at or below ret_quantiles.p05 (its historical-VaR "
            "partner); at n <= 20 the 'tail' is a single observation, and overlap makes tail "
            "rows the same market move"
        ),
        "benchmark_regression": (
            "per-window OLS attribution of the raw leg on the bench leg — beta unitless, alpha "
            "in outcome units per h-bar window, NEVER annualized; overlapping rows inflate the "
            "fit; null fields + reason when unbenchmarked, thin or degenerate; a one-regressor "
            "attribution, not a factor model"
        ),
        "episode_profile": (
            "the same statistic family in EPISODE units (per-episode mean ret, extreme "
            "excursions) — divergence from the row-level twin IS the cluster diagnostic, "
            "reported never corrected; quantiles and streaks over few episodes rest on one or "
            "two of them, so read every field beside n_episodes"
        ),
        "timing": (
            "RAW-path medians censored at h; bars_to_positive covers only rows that ever "
            "touched positive (its n scopes it) — a survivors-only read, never a recovery "
            "probability"
        ),
    },
    "scope_boundary": (
        "the checklist prices ONE cell of ONE run — it takes no cross-cell correction, and "
        "cross-RUN search (DSL variants, re-submissions over the same data) is invisible to a "
        "stateless reporter and belongs to the calling agent; the identity layer (dsl_hash, "
        "data_digests, summary.index_start/index_end, and identity.environment — the "
        "python/numpy/pandas/scipy/numba stack the numbers were computed under) exists so the "
        "caller CAN enforce it"
    ),
}

#: The prose rationale behind ``METRIC_ROLES`` — emitted by ``seikan schema`` only (the
#: run report carries the compact map above; repeating ~3 KB of static prose in every
#: report buys nothing).
METRIC_ROLES_DOC: dict[str, JsonValue] = {
    "claim": (
        "The exit code reports how far the RUN got, never how the evidence looked: exit 0 means "
        "the run finished and every nominated output was written — the report, when one was "
        "nominated, is complete — whatever every cell's checklist says. A "
        "cell's `met` is a NON-INFERENTIAL checklist — completeness, support, and "
        "non-concentration on full-sample evidence. It makes no significance claim and certifies "
        "no positive expected return; `mean_ret > 0` inside `support` is a sign read on the "
        "realized sample, not a test. Nothing in this report selects: the engine measures every "
        "declared cell and reports each independently, so choosing among them — and pricing the "
        "multiplicity of having looked at n_hypotheses_attempted of them — is the calling "
        "agent's work, not seikan's."
    ),
    "run_checks": [
        "statistics_version + gate_evidence_basis (evidence_complete — the summary must carry "
        "the stamps this checklist was built against, or it refuses ungraded: a summary from a "
        "different estimator revision would be graded by the wrong rubric)",
        "targets + outcome + target_mode + n_bars + n_hypotheses_attempted + sources + cells "
        "(evidence_complete — targets must be a non-empty list of STRINGS (a non-string name "
        "indexes no panel), the outcome stamp (the measurement algebra every reported number is "
        "denominated in) must be the explicit {series, kind} dict the runner always stamps — a "
        "null stamp is stripped input and refuses — the target_mode "
        "stamp must be 'conjunction' or 'basket', because it SELECTS the rubric every "
        "cross-target read is graded under: a missing or garbage stamp refuses fail-closed, "
        "and a basket stamp over fewer than two targets or a diff outcome refuses as drifted "
        "input (validation refuses both upstream; the gate re-refuses, never trusts) — the "
        "geometry and declared grid must be "
        "countable and at least one, the sources panel must be string-keyed and cover the regime "
        "EXACTLY, and cells must hold EXACTLY n_hypotheses_attempted entries. That last one is "
        "this report's honesty invariant: the panel carries one entry per declared combo × "
        "horizon INCLUDING those that never fired, so a report short of the declared grid has "
        "dropped hypotheses from the search burden it declares — drifted input, not evidence)",
        "sources[*].{n_bars,n_missing} + sources[*].by_source[*].n_missing (source_coverage — "
        "the fail-closed availability contract over the RAW decision inputs, run-level because "
        "it is combo-independent: the entry tree's leaves (Field/External/DaysSince) either had "
        "data on a bar or they did not, whichever parameters read them. This is the layer the "
        "per-cell three-valued signal_coverage ledger structurally cannot see, because two hole "
        "classes decide cleanly while data is missing: an operand hole absorbed by a decisive "
        "sibling (Kleene F and U = F leaves the root DEFINED), and a hole a NaN-skipping "
        "recursive kernel (ema, expanding aggregates, bars_since_extremum) carried its state "
        "across before emitting a finite value. Counting at the source puts no operator between "
        "the hole and the count. A source that merely STARTS LATE is warmup, not a hole — its "
        "first_available is reported as evidence — but warmup requires a start to EXIST: a null "
        "first_available, an input that never became available at all, refuses. Unconditional; "
        "no knob)",
        "n_hypotheses_attempted (search_cap — the DECLARED grid, which non-firing combos cannot "
        "shrink, bounded by thesis_max_hypotheses. It is the ONLY multiplicity input this policy "
        "carries: cells are graded independently and no cross-cell correction is taken, so the "
        "cap bounds how wide a search a single run may declare and the caller prices its own "
        "selection against the number)",
    ],
    "cell_checks": [
        "cells[*].params + by_target + outcome_coverage + signal_coverage + episode_stats "
        "(+ pooled in basket) "
        "(cell_evidence — the cell must carry the evidence its own checks grade, and its panels "
        "must AGREE. Shape: the three per-target panels are string-keyed and cover the regime "
        "exactly (a silently dropped target leaves this check unmet rather than escaping notice "
        "by absence). "
        "Arithmetic: exit reasons sum to n_attempted and n_closed == exit_reasons.horizon — a "
        "ledger anyone can re-check. Reconciliation: by_target.n == outcome_coverage.n_closed, "
        "n_nonoverlap <= n (non-overlapping windows cannot outnumber observations), "
        "episode_stats.n == the "
        "per-target total (the concentration and support panels must describe one pool), and "
        "signal_coverage.n_bars == summary.n_bars (the decision ledger spans the whole index by "
        "construction). Basket cells must additionally carry the pooled dict their own rubric "
        "grades, reconciling with the member panels: pooled.n == the per-target total, "
        "pooled.n_nonoverlap <= pooled.n, pooled.n_nonoverlap <= n_bars (same-bar cross-member "
        "firings "
        "collapse in the greedy count), pooled.n <= n_bars × len(targets) (each member fires "
        "at most once per bar); a pooled key on a conjunction cell REFUSES — the runner "
        "writes pooled only in basket mode, so the configuration is the signature of a "
        "restamped basket and refusing it costs zero honest refusals; a missing target_mode "
        "stamp refuses — whether a pooled panel is part of the contract is then "
        "undeterminable. An internally impossible summary is drifted input, not something "
        "to grade)",
        "cells[*].outcome_coverage[*].exit_reasons.{no_outcome,no_benchmark} (outcome_coverage — "
        "the fail-closed missingness contract: the engine censors a NaN outcome endpoint or a "
        "benchmark hole and every statistic silently skips the row, which is exactly how a "
        "vendor outage, a stale feed, or an adversarial file could delete adverse outcomes and "
        "leave a clean-looking cell. Any such firing refuses; missing-at-random is never "
        "assumed. 'open' is ALLOWED at any count: with no holdout there is no embargo and no "
        "tail, so a forward window running past the last bar is structural right-censoring every "
        "cell near the index end must exhibit — refusing it would refuse the calendar, not a "
        "data defect. An in-bounds hole is not 'open'; it classifies as no_outcome/no_benchmark "
        "upstream and refuses here)",
        "cells[*].signal_coverage[*].{n_bars,n_undefined} (signal_coverage — the DECISION-side "
        "twin of outcome_coverage, pooled layer. The outcome ledger can only account for bars "
        "that FIRED, so a missing decision input does not censor an outcome: it suppresses the "
        "firing itself and leaves no trace there, which would mean deleting data improves a "
        "result. The engine evaluates conditions three-valued and counts post-warmup UNDECIDABLE "
        "bars (init & ~defined); any of them refuses. n_bars is pure geometry, so n_undefined <= "
        "n_bars is verifiable arithmetic no property of the data can bend. The ledger is keyed "
        "by COMBO upstream, so horizon siblings legitimately repeat the same counts — each cell "
        "is graded alone and nothing is ever summed across cells. Unconditional; no knob)",
        "cells[*].by_target[*].{n,n_nonoverlap,mean_ret} (conjunction) | "
        "cells[*].pooled.{n,n_nonoverlap,"
        "mean_ret} (basket) (support — the SAME sealed floors under either rubric: a raw "
        "observation count, a non-overlapping-window count (n_nonoverlap, the greedy "
        "count — overlapping forward returns inflate the raw one), and a positive mean. In "
        "conjunction the floors read every target individually — targets are the thesis's "
        "regime, so the weakest target decides. In basket the members form ONE evidence pool: "
        "the pooled block clears the floors and no member is examined alone — a thin member "
        "does not sink a basket cell, because the claim is about the pool, not any name in it. "
        "A missing target_mode stamp refuses. Deliberately NOT an inferential claim: no "
        "t-statistic and no p-value gates here, because the nominal statistics are "
        "known-uncalibrated on overlapping pools — they ride along as evidence and this check "
        "reads none of them)",
        "cells[*].by_target[*].concentration.top_share_abs (conjunction) | "
        "cells[*].pooled.concentration.top_share_abs + "
        "cells[*].pooled.member_share.max_member_share_abs (basket) + "
        "cells[*].episode_stats.max_cluster_share_abs (both modes) (concentration — one "
        "universal ceiling (thesis_max_concentration), dispatched by the target_mode stamp. "
        "Conjunction: every regime target's top-5% |return|-mass share (no target may ride one "
        "whale event through the regime claim) plus the mass share of the largest merged "
        "cross-target episode cluster (a crisis smeared across rows AND targets is still one "
        "episode). Basket: the pooled top share REPLACES the per-target layer — the basket is "
        "graded as one pool — the episode-cluster ceiling stays ('not one crisis'), and the "
        "member-mass ceiling joins them ('not one name': max_member_share_abs over the sealed "
        "same ceiling is the one-name-basket detector, and a missing decomposition refuses). "
        "A missing target_mode stamp refuses. A diff-outcome multi-target run refuses "
        "the cross-target mass read as incommensurable — level units from different series are "
        "not mass-comparable — and so does a MISSING or unreadable outcome stamp, since "
        "stripping it would otherwise bypass the guard)",
    ],
    "evidence_only": [
        "per-cell rot_p / rotation (the circular-shift null — KNOWN anti-conservative: it "
        "assumes shift-exchangeability and over-certifies under signal-aligned volatility "
        "regimes; no check reads it)",
        "t_hac / hac_se (event-time overlap HAC — KNOWN anti-conservative: the Bartlett taper "
        "understates the long-run variance on overlapping pools; no check reads it — re-derive a "
        "p from t_hac at df = n_nonoverlap-1 when you need one, knowing that caveat)",
        "the summary.pbo block (pbo, reason, n_splits, n_splits_attempted, n_candidates_min, "
        "n_combos, n_combos_scoreable, n_combos_declared, blocks, lambda_mean, "
        "oos_degradation_slope, oos_degradation_slope_reason, prob_oos_loss) — CSCV "
        "selection fragility over the SCOREABLE grid's symmetric block splits (the population "
        "ledger says how far short of the declared grid that fell, the per-split ledger how far "
        "the block-local thinning departed from a fixed candidate count); a grid-level "
        "descriptive read, never a per-cell result",
        "per-cell by_target hit_rate, win_loss_ratio, std_ret, skewness, kurtosis, tail_ratio, "
        "cvar_5 — distribution shape and effect size, descriptive: the pool_moments reads over "
        "the cell's own closed pool, on the by_target and pooled panels alike (tail_ratio ≡ "
        "|p95/p05| of ret_quantiles, derivable like profit_factor; cvar_5 is the mean at or "
        "below ret_quantiles.p05, its historical-VaR partner; a per-observation Sharpe is "
        "mean_ret / std_ret, a firing rate is outcome_coverage.n_attempted / n_bars — both "
        "derivable, neither emitted)",
        "per-cell profit_factor (gross win mass over gross loss mass, Σwins/|Σlosses| — the "
        "hit-weighted asymmetry partnering win_loss_ratio and derivable from it; zero rows "
        "join neither side, and either side empty yields null, never an infinity)",
        "per-cell by_target mean_ret_raw / mean_ret_bench (the excess mean's own "
        "direction-signed legs over the SAME closed pool, so mean_ret ≈ mean_ret_raw − "
        "mean_ret_bench holds by construction: attribution — '+0.8% because the target made "
        "+3.2% against a +2.4% market' vs 'lost less than a falling market' — not two extra "
        "hypotheses; unbenchmarked, mean_ret_raw equals mean_ret and mean_ret_bench is null)",
        "per-cell by_target benchmark_regression (per-window OLS attribution of the raw leg on "
        "the bench leg over PAIRED rows — beta = cov/var, alpha = mean_raw − beta·mean_bench in "
        "outcome units per h-bar window, NEVER annualized, r2 the fit's share; the identity "
        "alpha + beta·mean_bench == mean_ret_raw is a reader's to re-check, and the question it "
        "answers — is the excess mean alpha, or beta ≠ 1 riding market drift — is one the two "
        "leg means alone cannot; null fields with n and a reason when unbenchmarked "
        "(no_paired_observations), below 3 pairs, or over a constant bench leg; a one-regressor "
        "attribution, never a factor model, and the overlapping rows inflate the fit exactly as "
        "they inflate every row-level read)",
        "episode_stats beyond the two fields the checklist reads (n, max_cluster_share_abs): "
        "n_clusters, the earliest entry, and the cluster-mass profile are a regime-clustering "
        "diagnostic",
        "per-cell by_target boot (the episode-bootstrap percentile CI for the "
        "pool mean: overlap-connected [t, t+h) episodes resampled with replacement, so "
        "within-episode dependence is preserved exactly and the interval is as wide as the "
        "EPISODE count warrants; deterministic, content-seeded; below 5 episodes it reports "
        "null fields with a reason instead of a degenerate interval)",
        "per-cell by_target subperiods (n / mean_ret over three equal-bar eras "
        "of the shared index, entry-bar assignment, NO purging: era visibility, not a holdout, "
        "and nothing selects on it)",
        "bar_spacing ({min,median,max}_seconds between consecutive bars: the "
        "clock geometry every horizon-in-bars is denominated in; self-description, never "
        "interpreted by the engine)",
        "per-cell by_target ret_quantiles + worst_ret / best_ret (the closed pool's "
        "{p05,p10,p25,p50,p75,p90,p95} and its single worst and best observations: the SHAPE "
        "read a mean cannot give, so a reader can state what a typical observation looked like "
        "instead of describing the pool by its average alone — p05 doubling as the historical "
        "VaR(5%) whose lower tail cvar_5 averages, p95 its favorable mirror; no count of its "
        "own, because its pool is the cell's own n)",
        "per-cell by_target mae_quantiles / mfe_quantiles (the same seven points "
        "plus the subset's mean and extreme over the RAW post-entry excursion columns, each "
        "with its own n: the holding-period path evidence, aggregated into the report so it "
        "reaches a reader without --trades-out)",
        "per-cell by_target timing ({n_to_positive, median_bars_to_positive, n_to_trough, "
        "median_bars_to_trough} — the WHEN of the path aggregated the way the excursion pair "
        "is, so it reaches a reader without --trades-out either; medians only, both durations "
        "censored at h, and bars_to_positive covers only the rows whose path ever touched "
        "positive — a survivors-only conditional read whose own n scopes it, never a recovery "
        "probability)",
        "per-cell episode_profile (the episode-deduplicated TWIN of the row-level pool "
        "statistics: the same frozen cross-target overlap merge as episode_stats — one crisis "
        "seen through three targets is ONE episode — then one aggregate per episode (the MEAN "
        "of its rows' ret, matching the ledger's per-episode read; the extreme mae/mfe within "
        "it) and the same statistic family over the episodes: n_episodes, hit_rate, mean_ret, "
        "profit_factor, ret_quantiles, worst/best, the excursion blocks over per-episode "
        "extremes, edge_ratio, and max_win_streak / max_loss_streak — the one honest home for "
        "consecutive-outcome reads, since over overlapping ROWS a streak is a cluster artifact. "
        "Row-vs-episode divergence is the visible cluster diagnostic — the n vs n_nonoverlap "
        "doctrine "
        "applied to every pool statistic — REPORTED, never a correction: no row-level number is "
        "reweighted. Emits always; n_episodes == episode_stats.n_clusters by construction, and "
        "in basket the cross-target merge makes it the pooled episode read for free)",
        "per-cell by_target edge_ratio (mean RAW MFE over |mean RAW MAE| — the one sanctioned "
        "ratio between the excursion pools, both legs RAW so it survives a benchmark; "
        "deliberately unnormalized, so it compares nothing across instruments; the measured "
        "answer to 'would a stop have cost more than it saved?', which stays the caller's "
        "judgment)",
        "summary.baseline (the run-level unconditional base rate per horizon "
        "× target over every fillable anchor bar, same algebra/benchmark/direction as the "
        "cells, with an exclusions ledger and, in basket, a pooled row; NO uplift field ever — "
        "the conditional-vs-base-rate comparison is the caller's, and no check reads it)",
        "summary.cross_breadth (per cross node × combo, basket only — the per-bar count k of "
        "entering members the cross kernels reduced over (finite input, and under where/group "
        "eligible and finitely labelled), recomputed off the node's own memoized frames and "
        "summarized: n_bars_evaluated (k >= min_valid), "
        "n_bars_below_full (evaluated bars short of the declared basket), k_min/k_median/"
        "k_max over the evaluated bars, and first_full_bar, the warmup-tail mirror of "
        "first_available. The effective-universe ledger: member warmup legally thins the "
        "cross-section, and silence about the thinning was the failure mode — entries repeat "
        "across combos that do not move the node's input, honest repetition never a sum; "
        "read by no check)",
        "per-cell episodes (the time-ordered episode ledger under "
        "episode_stats: earliest first, never ranked by share, bounded at its cap with "
        "explicit mass-conserving truncation counts; n_total reconciles with "
        "episode_stats.n_clusters)",
        "per-cell conditional_buckets / bucket_monotonicity (per-feature "
        "mean-return-by-quantile over the CELL's own closed rows — bucket EDGES per target, "
        "records aggregated by ordinal, because one qcut over raw levels pooled across members "
        "would conflate member LEVELS with time variation — with explicit refusal reasons; "
        "there is no run-level pooled pair, because a pooled qcut's conditioning would depend "
        "on grid composition)",
        "per-cell feature_association (Spearman rho between the entry-time "
        "feature snapshot and the realized closed ret, per cell × feature × target — the time "
        "axis within ONE target, in BOTH modes; deliberately no p-value)",
        "per-cell pooled beyond the fields the checklist reads (basket only — "
        "mean_ret_raw/mean_ret_bench, hit_rate, profit_factor, t_hac/hac_se (same-bar "
        "cross-member pairs at full Bartlett weight), the common-shift rot_p, boot over "
        "cross-member-merged episodes, subperiods, ret_quantiles/worst_ret/best_ret, "
        "mae_quantiles/mfe_quantiles, edge_ratio: pooled twins of the by_target evidence "
        "riders, none gating — under cross_mean the pooled mean_ret is the FIRING subset's "
        "cross-sectional selection tilt: each closed bar's FULL cross-section demeans to zero, "
        "but the pool holds only the members that fired, so ≈ 0 appears only when firings are "
        "basket-wide (an unconditional basket-wide entry), and a selective signal's pooled mean "
        "IS its cross-sectional selection read. The LEGS still attribute: mean_ret_bench ≈ "
        "mean_ret_raw is the basket's own realized drift, a market-magnitude number and not a "
        "construction artifact)",
        "per-cell pooled.member_share.by_target (basket only — each member's share of the "
        "pooled |return| mass, a full decomposition and never a ranking; the checklist reads "
        "only max_member_share_abs)",
    ],
    "metric_classes": (
        "the compact metric_roles carries ONE machine-readable class per reported metric, so a "
        "consuming agent can tell STRUCTURALLY what kind of number it is holding instead of "
        "parsing this prose: `descriptive` describes the realized sample and certifies nothing; "
        "`inference` is an uncalibrated inferential estimator (rot_p, t_hac, boot, pbo — "
        "directionally informative, never significance); `integrity` is an accounting or "
        "reconciliation ledger whose job is to make missing or tampered evidence visible "
        "(coverage panels, counts, digests, the rotation null's resolution). The vocabulary is "
        "FIXED at those three values and the class is orthogonal to gating: what a "
        "check reads is stated in run_checks/cell_checks, and a descriptive quantity does not "
        "become 'integrity' because a check happens to read it. A block classes as a whole; the "
        "one dotted override is episode_stats.n, the reconciliation count cell_evidence "
        "re-checks, while episode_stats itself stays descriptive. (Adapted from the metric-"
        "registry tagging idea in portfolio libraries, reduced to a closed vocabulary: free-form "
        "tags invite drift, three sealed classes answer the one question a caller actually has — "
        "'may I treat this number as evidence, and of what kind?')"
    ),
    # The long mechanics behind the compact ``metric_roles.caveats`` map every report carries.
    "caveats": {
        "rot_p": (
            "The rotation null fixes the forward-return series and rotates only the firing "
            "mask, so it is valid exactly when the series looks the same wherever the mask "
            "lands (shift-exchangeability). Volatility clusters in the same stretches most "
            "signals fire — crises, earnings seasons, regime breaks — so rotated masks land in "
            "calm periods, the null distribution is too narrow, and rot_p over-certifies. It is "
            "also one-sided (right tail) and floored at the cell's own 1/(1 + rot_n_null) — the "
            "count of DEFINED shifts, which a sparse mask lifts above the run-level "
            "rotation.p_resolution. The pooled rot_p is a COMMON-SHIFT null: one shift rotates "
            "EVERY member's mask as a block, preserving per-member firing counts and the per-bar "
            "cross-sectional pattern a rank signal fixes — rotating members independently would "
            "destroy exactly the structure a basket thesis is about. With one member it reduces "
            "to the per-target null, and it inherits every caveat above."
        ),
        "t_hac": (
            "The event-time HAC's Bartlett taper downweights exactly the lags that carry the "
            "overlap covariance, so hac_se understates the long-run variance: the SE ratio "
            "approaches sqrt(2/3) ≈ 0.82 on heavily overlapping pools and Monte Carlo under an "
            "iid-innovation null rejects ~10-12% at nominal 5%. The df = n_nonoverlap − 1 "
            "reference "
            "fixes the tail's shape, not the SE's scale."
        ),
        "boot": (
            "The episode bootstrap resamples overlap-connected episodes as exchangeable units, "
            "which is honest about within-episode dependence but assumes independence BETWEEN "
            "episodes — adjacent episodes still co-move through slow volatility regimes, so the "
            "CI is less anti-conservative than t_hac, not calibrated. Below 5 episodes there is "
            "no resampling distribution worth reporting and the block says so with a reason."
        ),
        "pbo": (
            "CSCV describes the SCOREABLE grid, never one cell, and cannot see DSL variants "
            "tried across runs: only combos that fired with at least one closed observation "
            "enter, byte-identical clones collapse to one candidate, and the ledger n_combos "
            "<= n_combos_scoreable <= n_combos_declared says how far short of the declared "
            "grid the scored population fell — a PBO over survivors understates the search the "
            "caller actually ran. Its block scores are per-observation Sharpes computed on "
            "overlapping in-block pools (only block-boundary crossings are purged), so the "
            "ranking it is built on inherits overlap inflation; sparse grids fall back to S = 6 "
            "or 4 blocks, where the whole number can flip on one split (the reported `blocks` "
            "says which S was used)."
        ),
        "cross_breadth": (
            "k counts ENTERING members of a cross node per bar (finite input, and under "
            "where/group eligible and finitely labelled), recomputed off the node's own "
            "memoized frames — exactly the count the kernels floor at min_valid and then "
            "discard. Member warmup or a screen legally thins it (a late start is warmup, not "
            "a hole; "
            "post-warmup holes refuse via source_coverage), so k_min < len(targets) states "
            "COVERAGE, not a defect: what it makes visible is the effective universe drifting "
            "through time — early bars ranked among fewer members than late ones. Entries "
            "repeat across combos that do not move the node's input; they are never summed."
        ),
        "mean_ret": (
            "A full-sample in-sample descriptive: no holdout, no deflation, no cost model. The "
            "support check reads only its SIGN, and a positive sign on the realized sample is "
            "not an expected-return claim. It is also the number most easily mistaken for a "
            "typical outcome: a +3% mean can sit on a NEGATIVE p50 when two spikes carry the "
            "mass, and the concentration check does not catch it — that check reads |return| "
            "mass in one EPISODE, while a mild right skew spreads its mass thinly and still "
            "drags the mean off the median. Read mean_ret and ret_quantiles.p50 together; when "
            "they disagree in sign, the median is the honest headline and the mean is a "
            "statement about the tail."
        ),
        "concentration": (
            "top_share_abs is the |return|-mass share of the top 5% of observations with "
            "k = max(1, ceil(0.05·n)) — for n <= 20 that is the single largest observation, so "
            "thin pools read structurally elevated (n_top says what k was)."
        ),
        "ret_quantiles": (
            "Order statistics of the cell's own closed returns under numpy's default linear "
            "interpolation (so p05 agrees with the VaR(5%) read cvar_5 tails off, on the same "
            "pool). Two limits. Below n≈20 the outer points are "
            "interpolations between the extreme observations — p05 and p95 then describe one "
            "or two rows, not a tail — and even at the support floor of 30 an outer 5% point "
            "rests on ~2 observations. And the observations are OVERLAPPING: one market move "
            "is smeared across ~h rows, so these are observation-weighted descriptions of what "
            "the pool held, not independent-draw quantile estimates. Under a benchmark they "
            "are EXCESS returns, like ret itself."
        ),
        "mae_quantiles": (
            "The distribution of the per-trade maximum adverse excursion over the full H/L of "
            "[fill, fill+h-1] plus the exit open — "
            "how deep the position ran against itself before the horizon closed — plus the "
            "subset's mean beside the n that scopes it. RAW path always: under a benchmark, ret "
            "becomes excess while mae does NOT, so differences between the two are "
            "unit-mismatched and no ratio of ret against an excursion is meaningful (edge_ratio "
            "divides the two RAW excursion means by each other, which is why it alone is "
            "sanctioned). Overlapping windows share the same trough, so one crash sets the mae "
            "of ~h neighbouring rows and the lower tail has atoms rather than independent "
            "events. The block's own n can be BELOW the cell's n: a hole anywhere in the "
            "excursion window censors mae on a row whose ret still closed. Custom-outcome and "
            "series-shaped targets have no true intrabar range (open=high=low=close), so their "
            "excursions understate."
        ),
        "mfe_quantiles": (
            "The favorable mirror of mae_quantiles — the best interim mark over the same window "
            "— carrying every one of its caveats (RAW path, shared peaks across overlapping "
            "windows, its own n and mean, synthesized ranges on series targets). One more, and "
            "it is the one that misleads: an MFE is a MARK, never an exit. This engine has no "
            "exit rule, so 'the trade was up 8% at one point' describes the path the horizon "
            "measurement sat through, and reading it as a foregone gain silently assumes an "
            "exit policy no part of this report measured."
        ),
        "mean_ret_raw": (
            "(and mean_ret_bench) The excess mean's own legs, each the mean of a per-observation "
            "direction-signed column over the SAME closed pool as mean_ret, so mean_ret ≈ "
            "mean_ret_raw − mean_ret_bench holds by construction (the runner subtracts the legs "
            "per observation and the mean is linear). They are ATTRIBUTION, not two extra "
            "hypotheses: '+0.8% excess because the target made +3.2% against a +2.4% market' "
            "and '+0.8% because the target lost less than a falling market' are different "
            "stories about the same excess mean, and without the legs a reader cannot tell "
            "them apart. On an unbenchmarked run mean_ret_raw equals mean_ret and "
            "mean_ret_bench is null — no benchmark leg existed, which null states and zero "
            "would falsify. In-sample, gross of costs, uncorrected — every mean_ret caveat "
            "applies to each leg."
        ),
        "edge_ratio": (
            "mean RAW MFE over |mean RAW MAE| for one cell's own excursion pools — the measured "
            "answer to 'did the typical interim gain outrun the typical interim pain?', which is "
            "the evidence a stop-loss debate actually needs. Both legs are RAW path, so the "
            "ratio survives a benchmark unchanged; it is deliberately UNNORMALIZED (vectorbt's "
            "edge ratio scales each excursion by a volatility estimate at entry first), so it "
            "compares nothing across instruments and no cross-cell league table may be built "
            "from it. The MFE leg is a mark, never an attainable exit; overlap smears one move "
            "across ~h rows on both legs; and the ratio is formed over PAIRED rows — both "
            "legs finite — while the quantile blocks keep their own per-leg subsets, so under "
            "asymmetric holes it need not equal their means' ratio. Null when no pair survives "
            "or the paired adverse mean is exactly zero — an unbounded ratio is refused, never "
            "emitted."
        ),
        "profit_factor": (
            "Gross win mass over gross loss mass, Σwins/|Σlosses|, over the pool's closed "
            "returns — which OVERLAP: one market move enters both sums up to ~h times, and no "
            "equity curve or position book exists, so this is never a realizable P&L ratio, "
            "only the pool's mass asymmetry. It is the MASS-weighted read beside "
            "win_loss_ratio's AVERAGE-weighted one, and the two are algebraically joined: "
            "profit_factor ≈ (n_wins/n_losses) × win_loss_ratio (exact when no zero-return "
            "rows exist) — a convenience read, never independent evidence. Zero returns join "
            "neither side. When either side is empty the field is null, never an infinity: "
            "'no losses in sample' is hit_rate's fact to state, and an unbounded ratio invites "
            "exactly the certainty-shaped reading this report refuses everywhere else."
        ),
        "baseline": (
            "The unconditional base rate is measured in EXACTLY the cells' own algebra — same "
            "outcome, same benchmark leg, same direction sign — so a conditional mean quoted "
            "without it is the market wearing a costume. It is in-sample, over every fillable "
            "anchor bar, and the exclusions ledger (open / no_outcome / no_benchmark) is its "
            "honesty channel: n_eligible + sum(exclusions) == n_anchor_bars is re-checkable "
            "arithmetic. There is deliberately NO uplift or difference field — computing the "
            "comparison is the caller's act. Under benchmark 'cross_mean' the basket's pooled "
            "BASELINE mean sits at ~0 BY CONSTRUCTION: every member enters every eligible bar, "
            "each member's excess is taken against the members' own cross-sectional mean, and "
            "complete cross-sections sum to zero — an identity, not a market fact. The pooled "
            "CELL mean is NOT that identity: its pool holds only the members that FIRED, so it "
            "is the firing subset's cross-sectional selection tilt, ≈ 0 only when firings are "
            "basket-wide."
        ),
        "episodes": (
            "The ledger lists overlap-merged episodes EARLIEST FIRST — it is never ranked by "
            "share, so 'the biggest episode' requires reading all of it. It is bounded at its "
            "stated cap: past it, entries fall off the list but never off the ledger's "
            "arithmetic (n_omitted and omitted_share_abs conserve the mass), so a count read "
            "off a truncated list is a floor, not a total. n_total == episode_stats.n_clusters "
            "always — a mismatch is drifted input."
        ),
        "conditional_buckets": (
            "Per-cell, per-feature qcut buckets over the cell's own closed rows — edges per "
            "TARGET, aggregated by ordinal: one pooled qcut over raw levels would let the top "
            "bucket simply BE the high-level member, a Simpson's inversion wearing a "
            "conditioning read's clothes. The rows are OVERLAPPING, so bucket means inherit "
            "the same "
            "smearing as every pooled read: 'associated in this sample', never 'predicts'. Do "
            "not rebuild a pooled cross-cell version yourself — the same bar enters once per "
            "combo × horizon, which makes every bucket boundary depend on grid composition "
            "(dishonest conditioning, not redundancy)."
        ),
        "feature_association": (
            "Spearman between the entry-time feature snapshot and the realized closed ret, per "
            "cell × feature × target. It carries NO p-value on purpose: overlap inflates any p "
            "into exactly the over-trustable number the doctrine forbids. It stays per-target "
            "in BOTH modes, because a pooled cross-member rank correlation would conflate "
            "LEVEL differences between members with variation through TIME — a basket's "
            "pooled evidence lives in the pooled panel, not here."
        ),
        "pooled": (
            "The pooled panel's rows are (bar × member) observations: one market move smears "
            "across the basket's members AS WELL AS across ~h overlapping horizons, so "
            "pooled.n overstates the independent information TWICE. pooled.n_nonoverlap is the "
            "same "
            "greedy non-overlapping kernel as everywhere else — same-bar firings across "
            "members collapse to ONE non-overlapping window — and it, with the episode-"
            "cluster share, is the honest size of the pool. by_target attribution never "
            "grades: 'strong in NVDA, weak in AMD' is not a statement a basket run can make."
        ),
        "member_share": (
            "A FULL decomposition of the pooled |return| mass across members, never a ranking "
            "and never a verdict about any member. Structure matters: a 2-member basket's "
            "larger member always carries >= 0.5, so small baskets read structurally elevated "
            "against the same sealed ceiling — exactly as thin pools do under concentration. "
            "Only max_member_share_abs gates; by_target is attribution."
        ),
        "win_loss_ratio": (
            "Average win over average |loss| — the payoff-asymmetry partner of hit_rate's "
            "frequency: a 40% hit rate with a 3:1 payoff and a 70% hit rate with 1:2 describe "
            "opposite edges the mean alone conflates. Null, never an infinity, when either "
            "side is empty ('no losses in sample' is hit_rate's fact to state, not an unbounded "
            "ratio's), and overlap smears one market move into BOTH sides of the ratio, so it "
            "is a description of this pool, not an estimate of a payoff distribution."
        ),
        "std_ret": (
            "Sample dispersion (ddof=1) of OVERLAPPING rows: ~h rows share each move, so this "
            "understates the dispersion of the independent information exactly as n overstates "
            "n_nonoverlap — the same inflation read from the other side. A description of the "
            "realized "
            "pool, never a risk estimate, and never annualized."
        ),
        "skewness": (
            "(and kurtosis — PEARSON kurtosis, normal = 3, not excess) The shape moments that "
            "say when the mean is a poor description of the pool: heavy right skew means the "
            "mean rides a few spikes, fat tails mean the typical observation is calmer than "
            "the moments suggest. Over an overlapping pool one smeared move manufactures "
            "apparent tail weight — a single crash entering ~h rows reads as a tail, not an "
            "outlier — so both are directional reads, not estimated population moments."
        ),
        "tail_ratio": (
            "|p95 / p05| of the closed pool — the tail-asymmetry convenience read, exactly "
            "derivable from ret_quantiles and carried like profit_factor is: a convenience, "
            "never independent evidence. Unstable when p05 sits near zero (the denominator is "
            "an order statistic, not a scale), and on thin pools both tails rest on one or two "
            "rows the overlap made siblings of one move."
        ),
        "cvar_5": (
            "The mean of the observations at or below ret_quantiles.p05 — historical expected "
            "shortfall beside its VaR partner, adding what the quantile alone cannot say: how "
            "bad the tail is INSIDE the 5% cut. In-sample, gross of costs, and below n=20 the "
            "'tail' is a single observation; overlap makes tail rows the same market move, so "
            "this describes the realized worst stretch, never a loss distribution."
        ),
        "benchmark_regression": (
            "Per-window OLS of the raw leg on the bench leg over PAIRED closed rows — the "
            "attribution the two leg means cannot give: beta says how much of the raw outcome "
            "rode the benchmark (unitless), alpha what remained per h-bar window (outcome "
            "units, NEVER annualized — this engine has no calendar-return framing), r2 how "
            "much of the row variance the one regressor explains. alpha + beta·mean_bench == "
            "mean_ret_raw is an exact identity to re-check. Overlapping rows inflate the fit "
            "precisely as they inflate every row-level statistic (no HAC correction is applied "
            "to the regression — it is attribution, not inference), so r2 and beta are "
            "descriptions of this pool, never factor loadings; null fields with n and a reason "
            "when unbenchmarked, below three pairs, or over a constant bench leg."
        ),
        "episode_profile": (
            "The episode-deduplicated TWIN of the row-level statistics: the same frozen "
            "cross-target overlap merge as episode_stats clusters the rows into market "
            "episodes, each episode contributes ONE aggregate (the mean of its rows' ret — the "
            "ledger's own per-episode read; a sum would scale with overlap density and "
            "re-import the inflation — and its extreme mae/mfe), and the same statistic family "
            "is recomputed over the episodes, streaks included (their one honest home: over "
            "overlapping rows 'consecutive' is a cluster artifact). Read it AGAINST the "
            "row-level twin — hit_rate vs hit_rate, mean vs mean, tail vs tail: divergence is "
            "the cluster diagnostic, and nothing is corrected or reweighted, because choosing "
            "the exchangeable unit is the caller's modeling judgment, not this reporter's. "
            "Emits always; quantiles and streaks over few episodes rest on one or two of them "
            "(n_episodes says so), and n_episodes == episode_stats.n_clusters by construction."
        ),
        "timing": (
            "Medians of the RAW timing pair (bars_to_positive, bars_to_trough), each over its "
            "own finite subset with the count that scopes it. Both durations are right-censored "
            "at h — a path still under water at the horizon reports no bars_to_positive at all "
            "— so these are medians only (a mean of censored durations misleads), and "
            "bars_to_positive's median is a SURVIVORS-ONLY conditional read: 'the paths that "
            "recovered did so by bar K', never 'paths recover with probability'. The ratio "
            "n_to_positive / n is the caller's to form and interpret."
        ),
    },
    "scope_boundary": (
        "The checklist prices one cell of one run. It takes NO cross-cell correction — the "
        "search cap bounds the declared grid and n_hypotheses_attempted stamps it, and pricing "
        "the multiplicity of choosing among those cells is the calling agent's work. A stateless "
        "reporter equally cannot police search ACROSS runs: many DSL variants or re-submissions "
        "over the same data are invisible to it, and repeated external search invalidates any "
        "error rate anyone might attach to a single report. The identity layer (dsl_hash, "
        "per-key data_digests, summary.index_start/index_end) makes every distinct exam visible "
        "so the calling agent CAN enforce a budget; research-process discipline (how many theses "
        "were tried, pre-registration, family-level correction across runs) belongs to that "
        "agent. Never read exit 0 as a certificate over anything — not the research process, and "
        "not even one cell."
    ),
}

#: The report/summary field dictionary — the OUTPUT-side twin of ``dsl_json_schema``. Emitted by
#: ``seikan schema`` only: an agent caches the schema once and holds every definition, while the
#: report stays lean (the same split as ``METRIC_ROLES_DOC``). Inner keys are summary-relative
#: field paths; values are one-line definitions.
REPORT_FIELDS: dict[str, JsonValue] = {
    "identity": {
        "name": "the thesis's own declared name, echoed verbatim",
        "dsl_hash": (
            "canonical_dsl_hash of the normalized document (defaults filled, null-valued optional "
            "fields dropped, keys sorted) — the identity two runs compare on; paths and column "
            "names live OUTSIDE it"
        ),
        "data_digests": (
            "per declared data key: {path, column, sha256} — where THIS invocation found the "
            "bytes, which column answered the key (null when the file named its own), and the "
            "file's raw-byte digest; keyed by the LOGICAL key, never by path"
        ),
        "thresholds": "the checklist-knob snapshot actually used (canonical-as-floor)",
        "thresholds_canonical": "true iff every knob equals its class default",
        "thresholds_provenance": (
            "per knob: default|env|cli — the SOURCE, so an auditor tells a stricter-via-flag "
            "run from a stricter-via-env one"
        ),
        "environment": (
            "the numeric stack the numbers were computed under — python/numpy/pandas/scipy/"
            "numba versions; floating-point results are a property of (inputs, code, "
            "libraries), and a report reproducible only against an unstated stack is not fully "
            "reconstructible"
        ),
    },
    "conventions": {
        "alignment": (
            "summary.cells[i] and gate.cells[i] are POSITIONALLY aligned; cell_id is a rendered "
            "label, never a key — identity is params + position"
        ),
        "nulls": (
            "every non-finite number serializes as null; a target with no closed rows carries "
            "n=0 and null statistics BY CONSTRUCTION — null means 'no evidence', never 'zero' "
            "(the evidence blocks additionally carry a `reason` string when null)"
        ),
        "units": (
            "every return-valued field (mean_ret, ret, mae, mfe, pre_ret, ci_lo/ci_hi, cvar_5, "
            "ret_quantiles, ...) is denominated per summary.outcome.units — 'fraction' (pct), "
            "'log', or 'level_diff' (the measured series' own level units); benchmarked runs are "
            "EXCESS returns in the same algebra, and direction signs every measurement. "
            "ret_raw / ret_bench (and mean_ret_raw / mean_ret_bench) are the excess return's "
            "own direction-signed legs in that same algebra — ret == ret_raw − ret_bench per "
            "observation. ONE exemption: the RAW-path fields (mae, mfe, pre_ret, the "
            "mae_quantiles / mfe_quantiles blocks incl. their means) "
            "are never benchmark-adjusted, so under a benchmark they are not commensurable with "
            "ret and no difference between them means anything — edge_ratio divides the two RAW "
            "excursion means by each other, which is why it alone is sanctioned"
        ),
        "grid": (
            "the summary is a GRID: it carries no rollup, no breakdown table and no pooled "
            "headline, because a mean of per-cell means over different horizons is a number in "
            "no unit that moves with grid composition. Read a sweep axis off cells[*].params "
            "against cells[*].by_target[t] per horizon; a cell that never fired is on the "
            "record with n = 0"
        ),
        "caveats": (
            "metric_roles.caveats (in this same report) carries one honest sentence per "
            "over-trustable number — read it before quoting rot_p, t_hac, boot, pbo or "
            "mean_ret — and metric_roles.metric_classes tags every metric "
            "descriptive | inference | integrity, so what KIND of number a field is can be "
            "read structurally"
        ),
    },
    "run": {
        "statistics_version": "the estimator revision that produced every number",
        "gate_evidence_basis": "'full_sample' — no holdout exists; every cell is measured once",
        "target_mode": (
            "'conjunction' | 'basket' — which target semantics produced every cross-target "
            "read; always stamped. The checklist dispatches on it (a missing stamp refuses): "
            "conjunction grades targets as the thesis's regime, weakest target deciding; "
            "basket grades each cell's pooled cross-target panel"
        ),
        "baseline": (
            "run-level unconditional base rates, one entry per horizon in declaration order: "
            "by_target[t] = {n_anchor_bars, n_eligible, exclusions, mean_ret, std_ret, "
            "hit_rate, ret_quantiles, worst_ret, best_ret} over EVERY fillable anchor bar, "
            "same algebra/benchmark/direction as the cells; basket entries additionally carry "
            "a pooled row summing the per-target counts; n_eligible + sum(exclusions) == "
            "n_anchor_bars is re-checkable; empty pools are null, never zero; NO uplift field "
            "— the conditional-vs-base-rate comparison is the caller's (evidence-only)"
        ),
        "n_bars / index_start / index_end": "geometry and extent of the evaluated joined index",
        "bar_spacing": (
            "{min,median,max}_seconds between consecutive bars — the clock geometry a "
            "horizon-in-bars is denominated in"
        ),
        "n_hypotheses_attempted": (
            "the DECLARED combo × horizon grid, non-firing combos included — the ONLY "
            "multiplicity input in the report; nothing is corrected for it"
        ),
        "outcome": (
            "{series, kind, units} — ALWAYS explicit (never null): the "
            "measurement algebra every reported number is denominated in"
        ),
        "direction / benchmark / benchmark_source / target_shape": (
            "self-description: the sign convention, whether returns are excess, against what, "
            "and the target's data shape (ohlcv | series)"
        ),
        "rotation": (
            "{n_shifts, p_resolution}: a rot_p AT p_resolution means 'no shift beat the "
            "observation', not p ≈ 0"
        ),
        "pbo": (
            "the grid-level CSCV block {pbo, reason, n_splits, n_splits_attempted, "
            "n_candidates_min, n_combos, n_combos_scoreable, n_combos_declared, blocks, "
            "lambda_mean, oos_degradation_slope, oos_degradation_slope_reason, prob_oos_loss} "
            "— a property of the search space's SCOREABLE combos, attached to no cell; the "
            "ledger n_combos <= n_combos_scoreable <= n_combos_declared says how far short of "
            "the declared grid the scored population fell, and n_splits <= n_splits_attempted "
            "with n_candidates_min <= n_combos how far the block-local thinning departed from "
            "the fixed candidate count canonical CSCV assumes"
        ),
        "sources": (
            "per-target per-decision-leaf availability (n_missing, first_available) — the raw "
            "inputs under source_coverage"
        ),
        "cross_breadth": (
            "per (cross node × combo): {node, params, min_valid, n_bars, n_bars_evaluated, "
            "n_bars_below_full, k_min, k_median, k_max, first_full_bar} — the effective-"
            "universe ledger over the cross kernels' entering-member count k (finite input, "
            "and under where/group eligible and finitely labelled); always present, "
            "[] outside basket; evidence-only"
        ),
        "params / targets": (
            "the two grid labels: the swept ENTRY axes (horizon only when it was swept) every "
            "cell's params is keyed by, and the regime — the reference every per-target panel "
            "is verified against"
        ),
    },
    "cells": {
        "by_target.n": "closed observations in this target's pool",
        "by_target.n_nonoverlap": (
            "greedy NON-OVERLAPPING observation count — the overlap-honest sample size, "
            "and NOT an independence claim: non-overlap is not independence; "
            "every df in the layer derives from it"
        ),
        "by_target.mean_ret": (
            "mean closed return, units per summary.outcome, gross of costs, in-sample"
        ),
        "by_target.mean_ret_raw / mean_ret_bench": (
            "the excess mean's own direction-signed legs over the SAME closed pool — mean_ret "
            "≈ mean_ret_raw − mean_ret_bench; attribution, not two extra hypotheses "
            "(evidence-only). Unbenchmarked runs: mean_ret_raw == mean_ret and mean_ret_bench "
            "is null — 'no benchmark leg', never zero"
        ),
        "by_target.hit_rate": "share of closed returns > 0",
        "by_target.benchmark_regression": (
            "{n, beta, alpha, r2, reason} — per-window OLS attribution of the raw leg on the "
            "bench leg over PAIRED rows: alpha + beta·mean_ret_bench == mean_ret_raw exactly; "
            "alpha in outcome units per h-bar window, never annualized; null fields + reason "
            "when unbenchmarked (no_paired_observations), below 3 pairs, or over a constant "
            "bench leg (evidence-only)"
        ),
        "by_target.win_loss_ratio / std_ret / skewness / kurtosis / tail_ratio / cvar_5": (
            "the summarize shape/dispersion reads over the SAME closed pool, mounted per cell: "
            "payoff asymmetry (null when a side is empty), ddof=1 dispersion, the two shape "
            "moments (kurtosis is Pearson, normal = 3), tail_ratio ≡ |p95/p05| of "
            "ret_quantiles (derivable), and the mean at or below p05 — its historical-VaR "
            "partner (all evidence-only)"
        ),
        "by_target.profit_factor": (
            "gross win mass / gross loss mass (Σwins/|Σlosses|) over the closed pool — the "
            "mass-weighted asymmetry beside win_loss_ratio, derivable from it; null (never "
            "infinity) when either side is empty (evidence-only)"
        ),
        "by_target.t_hac / hac_se": (
            "event-time overlap-HAC t and SE (no p is emitted — re-derive it at "
            "df = n_nonoverlap - 1, knowing the caveat)"
        ),
        "by_target.rot_p / rot_n_null": (
            "one-sided right-tail circular-rotation p, and the number of DEFINED shifts its "
            "null was formed over — the cell's own resolution floor is 1/(1 + rot_n_null), "
            ">= summary.rotation.p_resolution"
        ),
        "by_target.concentration": (
            "{top_share_abs, n_top, top_frac}: |return|-mass share of the top 5% observations"
        ),
        "by_target.boot": (
            "{method, ci_level, n_boot, n_episodes, ci_lo, ci_hi, boot_se, reason} — "
            "episode-bootstrap percentile CI for the pool mean (evidence-only)"
        ),
        "by_target.subperiods": (
            "three equal-bar eras [{start, end, n, mean_ret}] — era visibility (evidence-only)"
        ),
        "by_target.ret_quantiles": (
            "{p05, p10, p25, p50, p75, p90, p95} of this pool's closed returns, "
            "linear-interpolated order statistics in summary.outcome units (EXCESS when "
            "benchmarked) — the typical-observation read, p05 doubling as the historical "
            "VaR(5%) cvar_5 tails off (evidence-only). No n of its own: the pool is "
            "by_target.n. Null at every point on an empty pool"
        ),
        "by_target.worst_ret / best_ret": (
            "the single worst and best closed observations of this pool, same units as "
            "mean_ret (evidence-only); null on an empty pool"
        ),
        "by_target.mae_quantiles / mfe_quantiles": (
            "{n, mean, p05, p10, p25, p50, p75, p90, p95, worst | best} over the per-trade "
            "post-entry excursions on the full H/L of [fill, fill+h-1] plus the exit open — "
            "mae <= 0, mfe >= 0, both RAW path and "
            "NEVER benchmark-adjusted (evidence-only). Their own n may be BELOW by_target.n: a "
            "hole in the excursion window censors mae/mfe on a row whose ret closed, and each "
            "block's mean covers exactly that same subset. Null at every point when n is 0"
        ),
        "by_target.edge_ratio": (
            "mean RAW MFE / |mean RAW MAE| over the two excursion pools — the one sanctioned "
            "excursion ratio (both legs RAW, so it survives a benchmark); deliberately "
            "unnormalized, never comparable across instruments; null when either pool is empty "
            "or the adverse mean is zero (evidence-only)"
        ),
        "by_target.timing": (
            "{n_to_positive, median_bars_to_positive, n_to_trough, median_bars_to_trough} — "
            "medians of the RAW timing pair, each over its own finite subset; censored at h, "
            "survivors-only (evidence-only)"
        ),
        "episode_stats": (
            "cross-target merged episode clusters over the cell's closed rows: {n, n_clusters, "
            "largest_cluster_n, largest_cluster_share_abs, largest_cluster_start, "
            "max_cluster_share_abs, mass_hhi, effective_n_clusters} — the checklist reads only "
            "n and max_cluster_share_abs; the hhi pair is the smooth evidence-only companion"
        ),
        "episodes": (
            "the time-ordered episode LEDGER under episode_stats: {entries: [{start, end, n, "
            "mean_ret, share_abs}], n_total, n_omitted, omitted_share_abs, cap} — earliest "
            "first, never ranked; truncation past the cap is explicit and mass-conserving; "
            "n_total == episode_stats.n_clusters (evidence-only)"
        ),
        "episode_profile": (
            "the episode-deduplicated TWIN of the row-level pool statistics: {n_episodes, "
            "hit_rate, mean_ret, profit_factor, ret_quantiles, worst_ret, best_ret, "
            "mae_quantiles, mfe_quantiles, edge_ratio, max_win_streak, max_loss_streak} over "
            "per-episode aggregates (mean ret; extreme excursions) under the SAME cross-target "
            "merge as episode_stats — n_episodes == episode_stats.n_clusters by construction; "
            "row-vs-episode divergence is the cluster diagnostic, reported never corrected; "
            "emits always; in basket it doubles as the pooled episode read (evidence-only)"
        ),
        "conditional_buckets / bucket_monotonicity": (
            "PER-CELL feature conditioning over the cell's own closed rows, pooled across its "
            "targets: per feature {buckets: [{bucket, n, mean_ret, hit_rate}], reason} with "
            "explicit refusal reasons, plus a per-feature Spearman {rho, sign} — there is no "
            "run-level pooled pair (evidence-only)"
        ),
        "feature_association": (
            "per feature × target {rho, n, reason} — Spearman between the entry-time feature "
            "snapshot and the realized closed ret within one target's time axis; per-target "
            "in BOTH modes; no p-value, deliberately (evidence-only)"
        ),
        "pooled": (
            "BASKET CELLS ONLY (absent — not null — on conjunction cells): the cell's one "
            "cross-target evidence pool over the concatenated (bar × member) closed rows in "
            "target-declaration order — the panel the basket rubric grades instead of "
            "per-member floors"
        ),
        "pooled.n / n_nonoverlap": (
            "closed pooled observations, and the greedy NON-OVERLAPPING count over them — the "
            "same n_nonoverlap kernel as by_target (one meaning engine-wide), so same-bar firings "
            "across members collapse to ONE non-overlapping window; the checklist "
            "reconciles both and grades them against the support floors"
        ),
        "pooled.mean_ret / hit_rate": (
            "pooled-panel twins of the by_target pair, same units; mean_ret > 0 is the basket "
            "support sign read"
        ),
        "pooled.t_hac / hac_se": (
            "event-time overlap HAC over the pooled rows, df = pooled.n_nonoverlap - 1 — same-bar "
            "cross-member pairs enter at full Bartlett weight (cluster-robust for free), same "
            "anti-conservative caveat (evidence-only)"
        ),
        "pooled.rot_p": (
            "COMMON-SHIFT rotation null — one shift rotates every member's mask as a block, "
            "preserving the per-bar cross-sectional pattern; see caveats.rot_p "
            "(evidence-only)"
        ),
        "pooled.concentration": (
            "{top_share_abs, n_top, top_frac} over the POOLED rows — the top-share read basket "
            "concentration grades INSTEAD of the per-target layer"
        ),
        "pooled.member_share": (
            "{by_target, max_member_share_abs} — each member's share of the pooled |return| "
            "mass; the checklist reads ONLY max_member_share_abs (the one-name-basket "
            "detector), by_target is attribution and never a ranking"
        ),
        "pooled.mean_ret_raw / mean_ret_bench / profit_factor / boot / subperiods / "
        "ret_quantiles / worst_ret / best_ret / mae_quantiles / mfe_quantiles / edge_ratio": (
            "pooled twins of the by_target evidence riders (boot resamples cross-member-merged "
            "episodes; every by_target caveat carries over), evidence-only — under cross_mean "
            "the pooled mean_ret is the FIRING subset's cross-sectional selection tilt (≈ 0 "
            "only when firings are basket-wide — the full cross-section demeans to zero, the "
            "firing subset need not) and the legs still attribute: mean_ret_bench ≈ "
            "mean_ret_raw is the basket's own realized drift"
        ),
        "outcome_coverage": (
            "per target {n_attempted, n_closed, exit_reasons} — 'open' is ALLOWED at any count "
            "(end-of-data right-censoring is structural); no_outcome / no_benchmark are data "
            "holes and refuse"
        ),
        "signal_coverage": (
            "per target {n_bars, n_undefined} — post-warmup bars where the entry condition was "
            "UNDECIDABLE; any > 0 refuses. n_bars == summary.n_bars always (pure geometry)"
        ),
    },
}

CSV_FORMAT: dict[str, JsonValue] = {
    "encoding": "UTF-8 (BOM tolerated)",
    "timestamp_column": "a column named 'datetime' (case-insensitive), else the first column",
    "timestamp_format": (
        "strict ISO-8601 (YYYY-MM-DD or full timestamp), timezone-NAIVE, unique, sorted "
        "ascending; no other date format is ever guessed"
    ),
    "value_columns": (
        "plain numbers; the only missing-value markers are an empty cell or 'nan'; "
        "no thousands separators, no currency symbols"
    ),
    "ohlcv_shape": (
        "columns open,high,low,close (+optional volume, others): high>=max(open,close), "
        "low<=min(open,close), prices>0, volume>=0 — violations refuse, never clamp"
    ),
    "series_shape": "one or more named numeric columns (a yield, a P/E, an index …)",
    "warnings_never_refuse": "NaN holes, crash-sized moves, calendar gaps (warned, admitted)",
}

#: The checklist contract (``seikan schema`` self-description). ONE checklist
#: — two rubrics, selected per run by the summary's stamped ``target_mode``, applied identically
#: to every declared cell — no profiles, no per-cell exemptions, nothing the caller can select
#: into. ``policy_version`` (stamped into ``gate``) names the checklist semantics; two cells'
#: results are comparable only under the same version.
GATE_CONTRACT_DOC: dict[str, JsonValue] = {
    "contract": (
        "one checklist, applied to EVERY declared parameter × horizon cell independently "
        "(the summary's target_mode stamp selects the rubric cross-target reads "
        "are graded under, and a missing stamp refuses): three run-level checks reported once "
        "in run_checks, five per-cell checks "
        "in every cells[i].checks, each {name, met, observed, threshold, detail}. "
        "No short-circuit — every check is always evaluated and always reported — and no "
        "verdict: the gate section is {policy_version, n_cells, n_met, run_checks, cells}, "
        "index-aligned with summary.cells. A cell's `met` is the conjunction of its own five "
        "checks AND all three run-level checks, so an unmet run-level check leaves every cell "
        "unmet and a caller reading cells[i].met gets the complete answer without ANDing "
        "sections itself"
    ),
    "claim": (
        "exit 0 certifies ONLY that the run completed and every nominated output was written — the "
        "report, when one was nominated, is complete — and it is not a "
        "verdict. A cell's `met` is a completeness / support / concentration checklist with NO "
        "significance claim and NO positive-expected-return certification: nothing in it is a "
        "test, and mean_ret > 0 is a sign read on the realized sample. Selection among cells and "
        "cross-cell multiplicity are the CALLER's, priced against n_hypotheses_attempted"
    ),
    "evidence_basis": (
        "FULL SAMPLE, uniformly: every cell is graded on its own rows over the whole index "
        "(gate_evidence_basis == 'full_sample', verified as a drift detector). There is no "
        "holdout, no embargo, no tail and no split — so there is nothing to shop and nothing to "
        "reserve, and equally no out-of-sample confirmation to claim. The engine measures every "
        "declared cell and reports each one; it does not select, rank, or crown a winner"
    ),
    "thresholds": (
        "canonical-as-floor: every knob constructs only at its default or STRICTER (exit 3 "
        "thresholds_invalid otherwise), so a cell reported as met always means "
        "at-least-canonical rigor and the party being graded cannot bend the checklist it is "
        "graded by. Four knobs, no optional ones: thesis_min_trades, thesis_min_n_nonoverlap, "
        "thesis_max_concentration, thesis_max_hypotheses"
    ),
    "run_checks": {
        "evidence_complete": (
            "statistics_version matches this build and gate_evidence_basis == full_sample (a "
            "summary from another estimator revision refuses ungraded rather than being graded "
            "by the wrong rubric); targets is a non-empty list of STRINGS; the outcome stamp "
            "(the measurement algebra every reported number is denominated in) is the explicit "
            "{series, kind} dict the runner always stamps — a null or partial stamp refuses as "
            "drifted input; the target_mode stamp is 'conjunction' or 'basket' — "
            "it SELECTS the rubric every cross-target read is graded under, so a missing or "
            "garbage stamp refuses fail-closed, and a basket stamp over fewer than two targets "
            "or a diff outcome refuses as drifted input (validation refuses both upstream; the "
            "gate re-refuses, never trusts); n_hypotheses_attempted and n_bars "
            "are countable and "
            ">= 1; the sources "
            "panel is string-keyed and covers the target set EXACTLY; and cells is a list "
            "holding EXACTLY n_hypotheses_attempted entries — every declared combo × horizon on "
            "the record, non-firing ones included, because a report missing declared cells has "
            "dropped hypotheses from the search burden it declares. NaN/±inf/non-integral reads "
            "refuse"
        ),
        "source_coverage": (
            "fail-closed availability contract over the RAW decision inputs, run-level because "
            "it is combo-independent: per target sources.n_missing == 0 — every leaf the entry "
            "tree reads (Field/External/DaysSince) available on every bar of the evaluated "
            "interval after its own first available bar — with sources.n_bars == summary.n_bars, "
            "every per-source count in 0..n_bars, and the union no larger than the sum of parts. "
            "This is the layer the per-cell three-valued signal_coverage ledger structurally "
            "cannot see: an operand hole absorbed by a decisive sibling (Kleene F and U = F "
            "leaves the root DEFINED) and a hole a NaN-skipping recursive kernel (ema, expanding "
            "aggregates, bars_since_extremum) carried its state across both decide cleanly while "
            "data is missing. A source that merely STARTS LATE is warmup, not a hole — its "
            "first_available is reported as evidence — but warmup requires a start to exist: a "
            "null first_available, an input that never became available at all, refuses. "
            "Unconditional, with no threshold knob"
        ),
        "search_cap": (
            "n_hypotheses_attempted (the DECLARED grid — non-firing combos cannot shrink it) "
            "<= thesis_max_hypotheses. The only multiplicity input this policy carries: cells "
            "are graded independently and no cross-cell correction is taken, so the cap bounds "
            "how wide a search one run may declare and the caller prices its own selection "
            "against the stamped number"
        ),
    },
    "cell_checks": {
        "cell_evidence": (
            "the cell entry is a dict with a dict params (its identity — the axes plus the "
            "horizon, always present); by_target, outcome_coverage and signal_coverage are "
            "string-keyed and cover the target set EXACTLY (a silently dropped target leaves "
            "this check unmet rather than escaping notice by absence); every count is countable "
            "and non-negative; the "
            "ledger arithmetic holds per target (sum(exit_reasons) == n_attempted, n_closed == "
            "exit_reasons.horizon); and the panels RECONCILE — by_target.n == "
            "outcome_coverage.n_closed, n_nonoverlap <= n, episode_stats.n == the per-target "
            "total, "
            "signal_coverage.n_bars == summary.n_bars. Basket cells additionally carry the "
            "pooled dict their rubric grades, reconciling with the member panels (pooled.n == "
            "the per-target total, pooled.n_nonoverlap <= pooled.n, pooled.n_nonoverlap <= "
            "n_bars, pooled.n "
            "<= n_bars × len(targets)); a pooled key on a conjunction cell REFUSES as a "
            "restamped basket (the runner writes pooled only in basket mode), and a "
            "missing target_mode stamp refuses. An internally impossible summary is "
            "drifted input, not something to grade"
        ),
        "outcome_coverage": (
            "fail-closed missingness contract: per target, exit_reasons.no_outcome == 0 and "
            "exit_reasons.no_benchmark == 0 — a data hole that deletes outcomes can hide adverse "
            "results, and missing-at-random is never assumed. exit_reasons.open is ALLOWED at "
            "any count: with no holdout there is no embargo and no tail, so a forward window "
            "running past the last bar is structural end-of-data right-censoring, not a data "
            "hole. An in-bounds NaN leg is never 'open' — it classifies as "
            "no_outcome/no_benchmark upstream and refuses here"
        ),
        "signal_coverage": (
            "fail-closed DECISION-side contract, the twin of outcome_coverage: per target "
            "n_undefined == 0 (no post-warmup undecidable decision bar — init & ~defined under "
            "the engine's three-valued evaluation) and n_undefined <= n_bars. The outcome ledger "
            "only accounts for bars that FIRED, so a missing input that suppresses a firing "
            "leaves no trace there; without this check, deleting the inputs under adverse "
            "firings would improve a cell unseen. The raw inputs underneath are graded once, "
            "run-level, by source_coverage. Unconditional, with no threshold knob"
        ),
        "support": (
            "the SAME sealed floors under the rubric target_mode selects: n >= "
            "thesis_min_trades AND n_nonoverlap >= thesis_min_n_nonoverlap AND mean_ret > 0. "
            "Conjunction — "
            "per target over the cell's own full-sample rows, the weakest target decides "
            "(targets are the thesis's regime). Basket — the members form ONE evidence pool: "
            "the pooled block clears the floors and no member is examined alone, so a thin "
            "member does not sink a basket cell. A missing target_mode stamp refuses. Evidence "
            "floors, deliberately NOT an inferential "
            "claim — no t-statistic or p-value gates, because the rotation rot_p and the "
            "overlap-HAC t are known anti-conservative and stay evidence-only"
        ),
        "concentration": (
            "one universal ceiling (thesis_max_concentration), dispatched by target_mode. "
            "Conjunction: every regime target's concentration.top_share_abs AND the cell's "
            "episode_stats.max_cluster_share_abs (the largest merged cross-target episode "
            "cluster's mass) — a one-episode edge refuses. Basket: "
            "pooled.concentration.top_share_abs REPLACES the per-target layer, the "
            "episode-cluster ceiling stays ('not one crisis'), and "
            "pooled.member_share.max_member_share_abs joins them ('not one name' — the "
            "one-name-basket detector; a missing member-mass decomposition refuses). A missing "
            "target_mode stamp refuses; a diff-outcome multi-target run refuses the "
            "cross-target mass read as incommensurable, and so does a missing, null, or "
            "unreadable outcome stamp"
        ),
    },
    "evidence_only": (
        "rot_p, t_hac/hac_se, the summary.pbo block (CSCV), the per-cell distribution-shape "
        "descriptives, the per-target boot "
        "episode-bootstrap CI and subperiods era panel, the run-level bar_spacing stamp and "
        "baseline panel, the "
        "episode_stats panel beyond the two fields the checklist reads, the per-cell episodes "
        "ledger, conditional_buckets/bucket_monotonicity and feature_association, and — in "
        "basket — every pooled field beyond {n, n_nonoverlap, mean_ret, "
        "concentration.top_share_abs, "
        "member_share.max_member_share_abs}, including member_share.by_target (attribution, "
        "never a ranking) all ride in the summary as EVIDENCE and no "
        "check reads them — see metric_roles (and its caveats map) for why"
    ),
    # The prose rationale behind the compact ``metric_roles`` map (which the run report and
    # ``seikan schema`` both stamp identically). It lives HERE, under its own key, so
    # ``metric_roles`` itself is never a dict in one command and a list-of-prose in another.
    "metric_roles_rationale": METRIC_ROLES_DOC,
}

#: What each exit code MEANS. They describe how far the RUN got — never how the evidence looked:
#: a completed run is exit 0 whatever its cells report, and 2/3/4 mean the run could not produce a
#: report at all.
EXIT_CODES: dict[str, JsonValue] = {
    "0": (
        "the command completed. For `run`: every nominated output was written and stdout stayed "
        "empty (the report, when one was nominated, is complete; per-cell results are inside "
        "gate.cells; the exit code is NOT a verdict and says nothing about any cell). The silence "
        "belongs to `run`, whose outputs are files — `hash`, `check-data`, `describe` and "
        "`schema` emit their own document on stdout at this same code"
    ),
    "2": "input data failed strict validation (see data_report)",
    "3": (
        "invalid request — an argparse usage error (including a run that nominates no output), an "
        "invalid thesis DSL or gate-threshold set, an invalid Turtle coefficients document, or an "
        "unusable nominated output path: one that is empty, unwritable, named by two flags at "
        "once, or names one of the run's own input files (usage / dsl_invalid / "
        "thresholds_invalid / coefficients_invalid envelope)"
    ),
    "4": "internal error",
}

#: The ``--trades-out`` CSV contract, column by column (nothing on stdout on success — the file
#: is the whole output; errors still emit the JSON envelope).
TRADES_CSV: dict[str, JsonValue] = {
    "command": (
        "seikan run <thesis.json> --trades-out <out.csv> (always overwrites; silent on success)"
    ),
    "rows": (
        "one per recorded OBSERVATION — firing bar × target × declared horizon, the WHOLE grid "
        "in one file (regroup on the leading param columns + target; there is no cell_id "
        "column). Censored firings ride along flagged by is_open/exit_reason and are excluded "
        "from every statistic. A firing on the FINAL bar anchors no observation and has NO row "
        "here — it rides --entry-flags-out"
    ),
    "columns": {
        "<swept axes>": (
            "one leading column per swept ENTRY axis, in summary.params order; absent when "
            "nothing is swept"
        ),
        "horizon": (
            "the cell's declared measurement horizon h — ALWAYS present, right after the swept "
            "axes and before target, so a row names its window whether or not the horizon was "
            "swept; with the swept axes it is the row's cell identity"
        ),
        "target": "the target the row belongs to (regime member or basket member)",
        "entry_time": (
            "ISO-8601 timestamp of the next-open ANCHOR bar t+1 — the firing bar is t; see "
            "entry_bar for the join rule"
        ),
        "exit_time": (
            "ISO-8601 timestamp of the exit bar (clamped to the final bar when censored)"
        ),
        "entry_bar": (
            "the FIRING bar's integer position on the joined index — the JOIN KEY to the "
            "entry-flags CSV's row position. Never join the two files on timestamps: entry_time "
            "is the anchor, one bar AFTER the firing the flags file marks"
        ),
        "entry_px": (
            "the measured value at the anchor (the target's open, or the outcome feed's value)"
        ),
        "exit_px": "the measured value at the exit bar; empty when censored",
        "ret": (
            "the signed forward outcome over [t+1, t+1+h], denominated per summary.outcome "
            "(excess when benchmarked, direction-signed); empty on censored rows"
        ),
        "ret_raw": (
            "the signed RAW outcome leg over the same window — equals ret when no benchmark is "
            "declared; ret == ret_raw - ret_bench per row when benchmarked; empty on censored "
            "rows"
        ),
        "ret_bench": (
            "the signed benchmark leg the excess subtraction consumed (the market series' "
            "same-window return, or the bar's basket mean under cross_mean); empty when "
            "unbenchmarked or censored"
        ),
        "pre_ret": (
            "RAW drift INTO the entry over the same h-bar window, sign-aligned — negative means "
            "the series moved AGAINST the eventual position; the leakage canary"
        ),
        "mae": (
            "worst interim adverse mark over the full H/L of [fill, fill+h-1] plus the exit "
            "open, RAW path (never "
            "benchmark-adjusted), <= 0; empty when censored"
        ),
        "mfe": (
            "best interim favorable mark over the same window, RAW path (never "
            "benchmark-adjusted), >= 0; empty when censored. A MARK, not an attainable exit — "
            "this engine has no exit rule"
        ),
        "bars_to_positive": (
            "first forward bar the measured path is back >= entry; empty if never, or censored"
        ),
        "bars_to_trough": "bars from fill to the MAE extremum; empty when censored",
        "exit_reason": (
            "horizon (closed) | open (end-of-data right-censoring — structural, the checklist "
            "allows it) | no_outcome / no_benchmark (in-bounds data holes — the checklist "
            "refuses them)"
        ),
        "is_open": (
            "True iff censored (any non-horizon exit_reason); censored rows carry no statistics"
        ),
        "<features>": (
            "one trailing column per entry-time feature snapshot, taken at the FIRING bar "
            "(defaults: ret_5, ret_20, vol_14)"
        ),
    },
    "join": (
        "join to --entry-flags-out ON entry_bar == that file's row position (the 0-based bar "
        "index of the joined index), NEVER on timestamps — a timestamp join is off by one bar "
        "(anchor vs firing). There are no epoch-ns entry_ts/exit_ts twins; the ISO times are "
        "the record"
    ),
    "derived_views": (
        "cells[*].episodes (and episode_stats) are DETERMINISTIC, DERIVABLE functions of this "
        "CSV — which is why there is no --episodes-out flag: take one cell's rows (regroup on "
        "the leading swept-axis columns; the horizon is a cell axis like any other), keep the "
        "CLOSED ones (is_open false), and greedily merge overlapping half-open "
        "[entry_time, exit_time) windows ACROSS targets — the SAME frozen overlap merge "
        "episode_stats runs. That reproduces the in-report ledger exactly, including past its "
        "cap: the report's episodes list truncates VISIBLY at `cap` entries (n_omitted / "
        "omitted_share_abs conserve the mass), while this CSV never truncates — rebuild the "
        "ledger from here when you need the entries past the cap"
    ),
}

#: The ``--root-series-out`` CSV contract (nothing on stdout on success — the file is the whole
#: output; errors still emit the JSON envelope).
ROOT_SERIES_CSV: dict[str, JsonValue] = {
    "command": (
        "seikan run <thesis.json> --root-series-out <out.csv> (always overwrites; silent on "
        "success)"
    ),
    "rows": "one per bar of the joined index; ISO-8601 'datetime' index column",
    "value_columns": (
        "one per deduplicated root series node — every Series operand of a threshold condition "
        "except bare constants, scalarized per param combo and rendered as an expression, e.g. "
        "percentile(iv30,80); '@<target>' suffix when several targets; '#N' suffix on a name "
        "collision (only the 'datetime' index name is reserved in this namespace)"
    ),
    "no_entry_flags": (
        "this CSV carries NO 0/1 entry-flag columns — it is the per-bar DECISION INPUT view, the "
        "evidence a caller reads to see why a bar did or did not fire. A bar that fired becomes a "
        "row of the --trades-out CSV instead, in observation shape; the 0/1 flags themselves ride "
        "the --entry-flags-out CSV, which is also the one output carrying a firing on the FINAL "
        "bar (it anchors no observation, so nothing in observation shape can represent it). "
        "Nominate all three flags and every view is on disk: decision inputs, observations, and "
        "the raw firing mask"
    ),
    "warmup": "transform warmup bars are empty cells (NaN)",
    "roundtrip": (
        "re-reads as a series-shaped strict CSV, unless a value column is entirely NaN (a window "
        "longer than the data) or the thesis has no root series at all (every threshold operand a "
        "bare constant, leaving a datetime-only CSV) — the strict reader refuses both"
    ),
}

#: The ``--entry-flags-out`` CSV contract (nothing on stdout on success — the file is the whole
#: output; errors still emit the JSON envelope). The DECISION-side twin of ``ROOT_SERIES_CSV``:
#: that file says what the entry tree SAW on each bar, this one says what it DECIDED.
ENTRY_FLAGS_CSV: dict[str, JsonValue] = {
    "command": (
        "seikan run <thesis.json> --entry-flags-out <out.csv> (always overwrites; silent on "
        "success)"
    ),
    "rows": "one per bar of the joined index; ISO-8601 'datetime' index column",
    "flag_columns": (
        "one 0/1 INTEGER column per (param combo × target), in combo-iteration × target order: "
        "'entry' for a thesis with no swept entry axis, else 'entry[axis=value,...]' naming the "
        "combo, with an '@<target>' suffix only when several targets run. These names are "
        "canonical and unique BY CONSTRUCTION — one column per declared combo × target and no two "
        "combos are equal — so unlike the root-series namespace, where two rendered expressions "
        "can collide, there is no '#N' disambiguation here"
    ),
    "relation_to_trades": (
        "bit-identical to the firing mask the backtest measures at (both read vectorize.signal), "
        "but the two files are ONE-TO-MANY, not row-for-row: one flagged bar × target opens one "
        "--trades-out row PER DECLARED HORIZON, so a horizon sweep multiplies the trades rows "
        "against a flags matrix that does not change shape. EXCEPTION: a firing on the FINAL bar "
        "has no next open to anchor at — it opens no observation, has no trades row at any "
        "horizon and is counted in no outcome_coverage ledger. This file is where that firing "
        "appears, and it is what answers 'is my thesis firing NOW?'. JOIN ON `entry_bar`, NEVER "
        "ON THE TIMESTAMP: this file is indexed by the FIRING bar t, while a trades row's "
        "`entry_time` is the next-open ANCHOR bar t+1, so a timestamp join is off by one bar; "
        "`entry_bar` is the trades column that equals this file's row POSITION. (The bar-for-bar "
        "twin is the --root-series-out CSV, which carries the identical index.) Note also that a "
        "root-series column may legally be named 'entry' (an external feed the caller named "
        "that), so a same-named column ACROSS the two files is a different thing — a value "
        "there, a 0/1 decision here"
    ),
    "roundtrip": (
        "always re-reads as a series-shaped strict CSV: integer 0/1 throughout, no NaN (the "
        "tradable signal is defined on every bar — warmup and undecidable bars are simply 0), and "
        "at least ONE value column whatever it is named, since a thesis declares at least one "
        "combo and at least one target (see flag_columns for the naming — a multi-target run has "
        "no column called plain 'entry' at all). The two degenerate shapes that can defeat the "
        "root-series CSV — an all-NaN value column, a datetime-only frame — cannot arise here"
    ),
}

#: The role map stamped into every ``seikan describe`` document (and emitted identically by
#: ``seikan schema``): exactly what the document claims — pure data profiling, nothing more —
#: with one honest sentence per number a reader is likely to over-trust, so a market-context
#: figure never quietly hardens into a thesis. The field-by-field dictionary is
#: ``DESCRIBE_REPORT`` (schema-side only, the ``REPORT_FIELDS`` split).
DESCRIBE_ROLES: dict[str, JsonValue] = {
    "claim": (
        "pure data profiling: `describe` states what the FILES contain — levels, changes, "
        "dispersion, range position, missingness — and MEASURES NOTHING. It runs no entry "
        "condition, opens no observation, grades no checklist and supports no thesis: nothing "
        "it emits says a series is attractive or stretched in any sense beyond its own "
        "trailing range, and no field is a recommendation of any kind. Exit 0 means every "
        "file was admitted; exit 2 means at least one was refused — the document is still "
        "emitted, with a stub profile per refused file and check-data's own data_report "
        "naming why"
    ),
    "caveats": {
        "percentile_rank": (
            "position within the trailing window's OWN range — NOT valuation: a trending "
            "series sits at its extreme by construction, so 1.0 or 0.0 reads 'at the window "
            "extreme', never 'over- or under-priced'"
        ),
        "dispersion": (
            "ddof=1 std of 1-bar changes, PER BAR and never annualized — any sqrt-time "
            "scaling is the caller's assumption about a cadence this engine does not "
            "interpret"
        ),
        "volume": (
            "last_to_mean is a plain ratio and carries NO 'unusual' flag — what counts as "
            "elevated is the caller's judgment, not a property of the file"
        ),
        "drawdown": (
            "measured from the highest bar THIS FILE contains (runup from its lowest) — "
            "extend or trim the file and the number moves; a property of the file's extent, "
            "not of the instrument"
        ),
        "windows": (
            "BARS, never days — bar_spacing states the clock ({min,median,max}_seconds "
            "between consecutive bars), and translating '21 bars' into calendar language is "
            "the caller's act"
        ),
    },
    "scope_boundary": (
        "`describe` is a pure observer of FILES the way `run` is of THESES: it profiles the "
        "bytes it was handed as of their last bar, refuses what fails the strict contract, "
        "and never repairs, ranks, selects or forecasts. Nothing it emits clears any "
        "checklist or supports any thesis; a figure quoted from it should name the file and "
        "be dated to index_end. The moment a question pairs today's description with what "
        "FOLLOWED — position paired with subsequent returns — it is a thesis, and a thesis "
        "is measured by `seikan run` or it is not measured at all"
    ),
}

#: The ``describe`` document's field dictionary — the output-side reference for the profiling
#: subcommand, exactly as ``REPORT_FIELDS`` is for ``run``. Emitted by ``seikan schema`` only;
#: the describe document itself stays lean and carries the compact ``DESCRIBE_ROLES`` above.
DESCRIBE_REPORT: dict[str, JsonValue] = {
    "document": {
        "command": (
            "seikan describe <files...> [--shape {ohlcv,series}] [--windows N,N,...] "
            "[--pretty] — one JSON document on stdout"
        ),
        "layers": (
            "seikan_version -> report_schema_version -> command -> data_report -> profiles "
            "-> describe_roles, in this FIXED order"
        ),
        "exit_codes": (
            "0 every file admitted / 2 any file refused (the document is STILL emitted — "
            "refused files carry stub profiles) / 3 usage (bad --windows, no files) / 4 "
            "internal. check-data parity: data_report is byte-equal to what `seikan "
            "check-data` would emit over the same files and --shape, because it comes from "
            "the same strict read"
        ),
        "order": (
            "profiles[i] describes the i-th file of the ARGUMENT LIST — argument order, "
            "never sorted; data_report.files aligns with it"
        ),
        "windows": (
            "one comma-separated list of BAR counts, default 1,5,21,63,126,252, at most 16, "
            "emitted in the GIVEN order under every windowed block; windows are bars, never "
            "days (bar_spacing states the clock)"
        ),
        "bounded_output": (
            "no per-bar array ever rides the document — its size is independent of n_bars; "
            "the per-bar views belong to the files themselves"
        ),
    },
    "profile": {
        "path / sha256 / ok": (
            "the file, its raw-byte digest (the same identity data_report carries), and "
            "whether it was admitted. A refused file's whole profile is the stub {path, "
            "sha256, ok: false, reason} with reason = the data_report error codes — nothing "
            "about a refused file is ever invented"
        ),
        "shape": "'ohlcv' | 'series', as detected — --shape only refuses, never converts",
        "n_bars / index_start / index_end": (
            "geometry and extent, in the run summary's vocabulary"
        ),
        "bar_spacing": (
            "{min,median,max}_seconds between consecutive bars — the run summary's own "
            "clock-geometry stamp, null below two bars"
        ),
        "last_bar": (
            "{timestamp, values: {column: value | null}} — the final row VERBATIM, every "
            "column, NaN as null and never back-filled"
        ),
        "series": (
            "the per-column profile blocks {changes, dispersion, range_position, "
            "full_sample, missingness}. An OHLCV file profiles `close` only (the full bar "
            "rides last_bar); a series-shaped file profiles EVERY value column, in file "
            "order — describe profiles the FILE, so it never asks which column you meant "
            "and has no column flag at all. Choosing the column a RUN reads is the "
            "invocation's act, and it has its own spelling there: `seikan run --column "
            "KEY=COL`"
        ),
        "volume": (
            "OHLCV files with a volume column only, else null: {last, windows: {N: {mean, "
            "last_to_mean, reason, ratio_reason}}} — the last volume, the trailing mean per "
            "window, and their plain ratio (refused with ratio_reason when the mean is not "
            "positive); no 'unusual' flag exists"
        ),
    },
    "blocks": {
        "changes[N]": (
            "{diff, pct, log, reason, ratio_reason} — the N-bar change of the last level "
            "against the level N bars earlier. diff whenever both endpoints are finite; "
            "pct/log ONLY when both are strictly positive (the positivity "
            "rule — ratio algebras through zero mint garbage, so they refuse with "
            "ratio_reason 'non_positive_endpoint' while diff lives). A NaN endpoint refuses "
            "as 'endpoint_missing' and is NEVER repaired by skipping to the previous finite "
            "value. All three algebras ride with domain-nulls because choosing one would be "
            "the engine deciding what a series IS"
        ),
        "dispersion[N]": (
            "{diff, pct, log, reason, ratio_reason} — ddof=1 std of the N trailing 1-bar "
            "changes (the same N+1 trailing levels changes[N] reads), one per algebra under "
            "the same domain gates, PER BAR and never annualized. N=1 holds a single change "
            "and cannot carry a ddof=1 std (insufficient_bars); a hole in the window is a "
            "change with a missing endpoint (endpoint_missing), never skipped"
        ),
        "range_position[N]": (
            "{high: {value, timestamp}, low: {value, timestamp}, from_high: {diff, pct}, "
            "from_low: {diff, pct}, percentile_rank, reason, ratio_reason} — trailing "
            "extremes over EXACTLY the last N bars: a shorter file refuses as "
            "insufficient_bars (the window is never silently shortened) and a hole inside "
            "the window refuses as endpoint_missing (the extremum could be hiding in it). "
            "Ties resolve to the MOST RECENT bar (the bars_since_extremum rule). "
            "percentile_rank is the right-continuous empirical CDF of the last level within "
            "the window — the share of window levels <= it, in (0, 1]"
        ),
        "full_sample": (
            "{high, low, drawdown_diff, drawdown_pct, runup_diff, runup_pct, reason, "
            "ratio_reason} — whole-file extremes over the OBSERVED (finite) levels, "
            "most-recent tie rule, with the missingness block beside them stating the "
            "holes; drawdown_pct = last/high - 1 and runup_pct = last/low - 1 (diff twins "
            "always ride), both against the ACTUAL last level — a NaN last level refuses "
            "them (endpoint_missing) rather than substituting the previous finite value"
        ),
        "missingness": (
            "{n_missing, n_interior_missing, first_valid, last_valid} — pure counts and the "
            "valid extent; threshold-flavored warnings stay in the loader's data_report"
        ),
        "reasons": (
            "refusals are EXPLICIT and PRESENT, never omitted: every windowed block carries "
            "its entry for every requested window, with reason in {insufficient_bars, "
            "endpoint_missing} when the block refuses whole, and ratio_reason in "
            "{non_positive_endpoint} when only the ratio algebras refuse while diff lives"
        ),
    },
}


# ---- the Turtle simulation (`stock-turtle-trade-long-only`) ----------------------------------

#: How each kind of fill is executed — stamped into ``simulation.fill_conventions`` and into
#: ``turtle_roles`` so a reader of either knows where every price came from.
TURTLE_FILL_CONVENTIONS: dict[str, JsonValue] = {
    "entry": (
        "the thesis fires on bar t (the engine's own tradable signal) while flat and past warmup "
        "→ a market buy at the opening print of bar t+1, filled at that bar's open; X = "
        "floor(risk_per_unit × budget / (stop_n × N_entry)) shares, N_entry the ATR at bar t, "
        "capped to what the target's cash affords"
    ),
    "add": (
        "a close at or above last_fill + add_step_n × N_entry, while under max_units → another X "
        "shares bought at the next opening print, wherever it opens (no gap guard; an open below "
        "the level still fills); skipped whole when X shares no longer fit the cash, and re-armed "
        "at the next close"
    ),
    "stop_close": (
        "stop_trigger 'close': a close below the stop → the whole position sold at the next "
        "opening print, at that open"
    ),
    "stop_gap": (
        "stop_trigger 'close': an opening print below the stop with no exit pending → sold at "
        "that open, no waiting for the close"
    ),
    "stop_trade": (
        "stop_trigger 'trade': one sell stop rests at the venue one price increment below the "
        "stop; a bar trading through it fills at the trigger, a print gapping through it fills "
        "at the open"
    ),
    "channel_close": (
        "exit_trigger 'close': a close below the lowest low of the previous exit_lookback "
        "completed bars → sold at the next opening print"
    ),
    "channel_trade": (
        "exit_trigger 'trade': the resting sell stop sits one increment below the channel when "
        "the channel is the higher level; fills at the trigger, or at the open when gapped"
    ),
    "end_of_data": (
        "a position still open after the last bar is MARKED at the last close — not a fill; it "
        "is excluded from every trade statistic and counted in n_open"
    ),
}

#: The role map stamped into every ``stock-turtle-trade-long-only`` report as ``turtle_roles``
#: (and emitted identically by ``seikan schema``): the exact claim a simulation report makes,
#: one honest sentence per number a reader is likely to over-trust, and the fill conventions.
TURTLE_ROLES: dict[str, JsonValue] = {
    "claim": (
        "a SIMULATION, not an event study: the thesis's entry firing is bought and managed under "
        "the long-only Turtle position rules on nautilus_trader's simulated exchange, with "
        "unlimited liquidity, zero commission and every fill at an opening print or a stop "
        "trigger. The report describes what that simulation did over the provided bars — one "
        "cell per declared entry combo, every cell reported, none ranked, no best cell — and "
        "certifies nothing about the future. exit 0 says only that the run finished and every "
        "nominated output was written"
    ),
    "caveats": {
        "equity_curve": (
            "one path over one sample: a single realized sequence of fills, not a distribution; "
            "every metric below is a description of that path"
        ),
        "cagr": (
            "the compounding exponent is bars_per_year / n_bars — a coefficient the caller sets, "
            "not a clock the engine reads; a short sample annualizes noise"
        ),
        "sharpe": (
            "mean / sample std of bar returns × √bars_per_year, risk-free zero; bar returns of a "
            "position-holding curve are autocorrelated and fat-tailed, so the ratio is neither "
            "normal nor comparable across bar spacings"
        ),
        "sortino": "the downside deviation is the root mean square of the negative bar returns",
        "calmar": "cagr over the deepest drawdown of one path — a two-number ratio of two extremes",
        "max_drawdown": (
            "measured on close-to-close equity marks, so an intrabar excursion below a close is "
            "invisible to it"
        ),
        "alpha_beta": (
            "an ordinary least-squares line through bar returns against the index's, alpha per "
            "bar and × bars_per_year; over a curve that is flat most bars, beta describes the "
            "in-market bars diluted by the flat ones"
        ),
        "information_ratio": (
            "mean / std of the return difference to buy-and-hold × √bars_per_year — the "
            "benchmark's own path is one sample too"
        ),
        "capture": (
            "mean strategy return over the index's up (down) bars divided by the index's mean "
            "over the same bars; null when the index never rose (fell)"
        ),
        "trade_stats": (
            "closed round trips only; an open end-of-data position is marked, counted in n_open "
            "and excluded; a handful of trips supports no rate or expectancy"
        ),
        "mae_mfe": (
            "raw excursions of the held bars' lows and highs against the ENTRY price, the exit "
            "bar included — marks, not attainable fills"
        ),
        "engine_stats": (
            "nautilus_trader's own analyzer over the same run, verbatim: its Sharpe/Sortino "
            "assume 252 periods, its PnL (total) is the account's cash change (an open position "
            "reads as its cost), and its per-position rates count positions, not round trips"
        ),
        "benchmark": (
            "buy-and-hold of the bound index from the first bar's open — a passive path with "
            "100% exposure against a strategy that is flat between signals"
        ),
        "liquidity": (
            "unlimited at every print and trigger: no partial fills, no slippage, no impact; the "
            "venue's own volume-based partial fills are switched off by construction"
        ),
        "costs": "zero commission and zero fees; the account currency is a label",
        "price_grid": (
            "every price is quantized to price_precision decimals before the run; the entry "
            "signal was computed by the event study on the unquantized data"
        ),
        "warmup": (
            "no entry before bar max(atr_period, exit_lookback) − 1; earlier firings are "
            "counted as entries.skipped.warmup, never taken"
        ),
        "sample_size": (
            "the number of round trips is the effective sample; n_bars is the calendar, not the "
            "evidence"
        ),
        "no_ranking": (
            "cells are reported in declaration order with every declared combo present; "
            "choosing among them, and the multiplicity of having looked, are the caller's"
        ),
    },
    "fill_conventions": TURTLE_FILL_CONVENTIONS,
}

#: The coefficients document (`seikan schema` also emits its JSON Schema as
#: ``turtle_coefficients.json_schema``).
TURTLE_COEFFICIENTS: dict[str, JsonValue] = {
    "command": (
        "seikan stock-turtle-trade-long-only <thesis.json> <coefficients.json> --data KEY=PATH "
        "... --data benchmark=PATH --report-out <path> [--trades-out] [--fills-out] "
        "[--equity-out] [--column KEY=COL] [--pretty]"
    ),
    "document": (
        "one JSON object, strict (unknown keys, non-finite numbers, duplicate keys and coerced "
        "types all refuse with the exit-3 coefficients_invalid envelope); every omitted field "
        "takes the rules' default, and the resolved set is stamped into identity.coefficients "
        "with its canonical hash"
    ),
    "fields": {
        "equity": "REQUIRED, > 0: total money; each target's budget is equity / n_targets",
        "atr_period": "default 20: N is the Wilder ATR over this many bars",
        "add_step_n": (
            "default 0.5: add a unit each time the close is this many N above the last fill"
        ),
        "max_units": "default 3: the position cap in units of X shares",
        "stop_n": "default 2.0: the stop sits this many N below the latest fill",
        "exit_lookback": "default 20: exit below the lowest low of this many completed bars",
        "risk_per_unit": (
            "default 0.01: X = floor(risk_per_unit × budget / (stop_n × N_entry)) — the worked "
            "example's $1,000 ÷ (2 × $2) = 250 shares"
        ),
        "stop_trigger": "default 'close' | 'trade': how the stop-loss fires (see fill_conventions)",
        "exit_trigger": "default 'close' | 'trade': how the channel exit fires",
        "stop_n_source": (
            "default 'current' | 'entry': after an add, the stop uses the ATR current at the "
            "add's signal bar (the literal reading of 'N is recalculated only for the stop') or "
            "the entry ATR; add levels and X always use the entry ATR"
        ),
        "bars_per_year": "default 252: the annualization clock of the report's metrics",
        "currency": "default 'USD': the simulated account's currency code (three capitals)",
        "price_precision": (
            "default 4 (0..9): decimals of the price grid every price is quantized to"
        ),
    },
    "rules": {
        "sizing": (
            "each target runs its own sub-account with a FIXED budget of equity / n_targets: the "
            "sizing base and the cash cap; the initial buy is capped to what the budget affords "
            "(entries.cash_capped) and an add that no longer fits is skipped whole "
            "(adds.skipped.budget)"
        ),
        "ladder": (
            "add levels are anchored on the last ACTUAL fill: last_fill + add_step_n × N_entry; "
            "one add per bar; every add is X shares — the first fill's size"
        ),
        "stop": (
            "initial P0 − stop_n × N_entry; after an add max(stop, fill − stop_n × N_stop) — it "
            "never moves down (stop_holds counts a held one)"
        ),
        "channel": (
            "the lowest low of the previous exit_lookback COMPLETED bars, the current bar "
            "excluded; whichever of stop and channel fires first closes the whole position"
        ),
        "signal": (
            "the thesis's own tradable firing per (entry combo × target), bit-identical to "
            "--entry-flags-out; a firing while in position, before warmup, on the final bar, or "
            "that sizes to zero shares is counted, never taken; direction must be longonly; "
            "params.horizon/outcome/benchmark/features are ignored (stamped in "
            "simulation.thesis_params_ignored)"
        ),
    },
}

#: The ``stock-turtle-trade-long-only`` report's field dictionary — the output-side twin of the
#: coefficients document, block by block in the report's layer order.
TURTLE_REPORT: dict[str, JsonValue] = {
    "layer_order": (
        "seikan_version → report_schema_version → command → identity → data_report → outputs → "
        "simulation → targets → params → n_cells → benchmark → cells → turtle_roles"
    ),
    "identity": {
        "name": "the thesis's declared name",
        "dsl_hash": "canonical_dsl_hash of the thesis — the same identity `seikan hash` emits",
        "coefficients": {
            "equity": "total money (required)",
            "atr_period": "the N lookback",
            "add_step_n": "the add step in N",
            "max_units": "the unit cap",
            "stop_n": "the stop distance in N",
            "exit_lookback": "the channel lookback",
            "risk_per_unit": "the sizing fraction",
            "stop_trigger": "close | trade",
            "exit_trigger": "close | trade",
            "stop_n_source": "current | entry",
            "bars_per_year": "the annualization clock",
            "currency": "the account currency",
            "price_precision": "the price grid's decimals",
            "_": "the RESOLVED set, every field (see turtle_coefficients for the semantics)",
        },
        "coefficients_hash": (
            "sha256 of the resolved coefficients (defaults filled, keys sorted) — omitted and "
            "explicit defaults share one identity"
        ),
        "data_digests": (
            "per bound key incl. benchmark: {path, column, sha256}, exactly as the run report"
        ),
        "environment": (
            "python/numpy/pandas/scipy/numba plus nautilus_trader and seikan_turtle versions"
        ),
    },
    "simulation": {
        "engine": "'nautilus_trader'",
        "engine_version": "the installed nautilus_trader distribution",
        "kernel": "'seikan._turtle' — the Rust kernel that takes every decision",
        "kernel_version": "the kernel's crate version",
        "venue": "'SIM'",
        "oms_type": (
            "'NETTING': one position per instrument, adds increase it, the exit flattens it"
        ),
        "account_type": "'CASH': no borrowing; the kernel never sizes beyond a target's budget",
        "currency": "the account currency (a label)",
        "starting_equity": "coefficients.equity",
        "n_targets": "the thesis's target count",
        "budget_per_target": "starting_equity / n_targets",
        "budget_mode": "'fixed': the budget neither grows with profits nor shrinks with losses",
        "instruments": "target → the positional venue symbol it traded under (T0.SIM, …)",
        "price_precision": "decimals of the price grid",
        "price_increment": "10^-price_precision — the step a 'trade' trigger sits below its level",
        "size_precision": "0: whole shares",
        "lot_size": "1",
        "liquidity": "'unlimited' (see turtle_roles.caveats.liquidity)",
        "commission": "0.0",
        "pre_trade_risk": (
            "who checks a buy before it is submitted: the kernel (the venue's risk engine is "
            "bypassed; its account books still apply and are reconciled)"
        ),
        "fill_conventions": "how every kind of fill was executed (the turtle_roles map)",
        "bars_per_year": "the annualization clock",
        "n_bars": "bars on the joined clock",
        "index_start": "first bar stamp",
        "index_end": "last bar stamp",
        "bar_spacing": "{min,median,max}_seconds between consecutive bars — the real clock",
        "first_eligible_bar": (
            "max(atr_period, exit_lookback) − 1: the first bar an entry can be taken on"
        ),
        "thesis_params_ignored": "the thesis parameters the simulation reads nothing from",
        "bar_type_label": (
            "the label every clock is fed under ('1-DAY-LAST-EXTERNAL'; see bar_spacing)"
        ),
    },
    "targets": "the thesis's targets in declaration order",
    "params": "the swept entry axes (the cell key columns)",
    "n_cells": "len(cells) == the declared entry combos",
    "benchmark": {
        "source": "the benchmark CSV's path",
        "construction": "how the buy-and-hold curve is built",
        "units": "starting_equity / open[0]",
        "start_equity": "starting_equity",
        "end_equity": "units × close[-1]",
        "metrics": "the index's own PerformanceMetrics over the same bars",
        "periodic": "the index's monthly and annual compounded returns",
    },
    "cells": {
        "cell_id": "'entry' or 'entry[axis=value,...]' — the entry-flags column label",
        "params": "the entry combo's axis values (identity is params + position)",
        "portfolio": {
            "metrics": (
                "PerformanceMetrics of the whole account (every sub-account summed bar by bar)"
            ),
            "relative": "RelativeMetrics of the account's bar returns against the benchmark's",
            "periodic": "monthly / annual compounded returns of the account",
            "trades": "TradeStats pooled over every target's round trips",
        },
        "by_target": {
            "metrics": "PerformanceMetrics of the target's sub-account (base = budget_per_target)",
            "relative": "the sub-account against the benchmark",
            "trades": "the target's own TradeStats",
            "end_state": {
                "shares": "shares held after the last bar",
                "units": "units held",
                "cash": "the sub-account's cash",
                "market_value": "shares × last close",
                "stop": "the stop in force, null when flat",
                "add_level": "the next add level, null when flat",
                "in_position": "shares > 0",
            },
        },
        "engine_stats": {
            "stats_pnls": "nautilus_trader's PnL statistics per currency (NaN → null)",
            "stats_returns": "nautilus_trader's return statistics (252-period assumptions)",
            "stats_general": "nautilus_trader's general statistics",
        },
        "reconciliation": {
            "n_fills_ledger": "fills the kernel recorded",
            "n_fills_engine": "filled orders the venue reports",
            "cash_change_ledger": "Σ end cash − starting_equity per the kernel",
            "cash_change_engine": "the venue's PnL (total) — its account's cash change",
            "end_cash_ledger": "Σ per-target cash per the kernel",
            "end_cash_account": "the venue account's closing total",
            "matched": "always true — a mismatch never becomes a report (exit 4)",
        },
    },
    "PerformanceMetrics": {
        "start_equity": "the curve's base (the bar before the first)",
        "end_equity": "the last mark",
        "net_pnl": "end − start",
        "total_return": "end / start − 1",
        "cagr": "(end/start)^(bars_per_year/n_bars) − 1; null when end ≤ 0",
        "annualized_volatility": "sample std of bar returns × √bars_per_year",
        "sharpe": "mean / std × √bars_per_year, risk-free zero; null at zero std",
        "sortino": "mean / downside deviation × √bars_per_year; null at zero downside",
        "calmar": "cagr / |max_drawdown|; null when either is unavailable or zero",
        "max_drawdown": "min over bars of equity / running peak − 1 (the start included), ≤ 0",
        "drawdown": {
            "peak_time": "stamp of the peak the deepest drawdown fell from (null: the start)",
            "trough_time": "stamp of the deepest trough",
            "recovery_time": "first stamp back at the peak (null: never)",
            "bars_to_trough": "bars from the peak to the trough",
            "bars_to_recovery": "bars from the trough to recovery (null: never)",
            "longest_drawdown_bars": "the longest run of bars below the running peak",
            "longest_drawdown_open": "whether that run is still open at the last bar",
        },
        "best_bar_return": "max bar return",
        "worst_bar_return": "min bar return",
        "mean_bar_return": "mean bar return",
        "median_bar_return": "median bar return",
        "skewness": "scipy population skewness of bar returns (null under 3 bars or zero std)",
        "kurtosis": "Pearson kurtosis (normal = 3)",
        "positive_bars_fraction": "bars with a positive return / n_bars",
        "exposure": {
            "bars_in_market": "bars with shares held",
            "fraction_in_market": "bars_in_market / n_bars",
            "mean_gross_exposure": "mean of market value / equity",
            "max_gross_exposure": "max of market value / equity",
        },
        "n_bars": "bars in the curve",
        "bars_per_year": "the annualization clock used",
    },
    "RelativeMetrics": {
        "beta": "cov(strategy, benchmark) / var(benchmark) over bar returns (ddof 1)",
        "alpha_bar": "mean(strategy) − beta × mean(benchmark), per bar",
        "alpha_annualized": "alpha_bar × bars_per_year",
        "correlation": "Pearson correlation of the bar returns",
        "r2": "correlation²",
        "tracking_error": "std(strategy − benchmark) × √bars_per_year",
        "information_ratio": "mean(strategy − benchmark) / std(…) × √bars_per_year",
        "excess_total_return": "total return minus the benchmark's",
        "excess_cagr": "cagr minus the benchmark's (null when either is null)",
        "up_capture": (
            "mean strategy return over the benchmark's up bars / the benchmark's mean there"
        ),
        "down_capture": "the same over the benchmark's down bars",
        "n_up_bars": "benchmark bars with a positive return",
        "n_down_bars": "benchmark bars with a negative return",
    },
    "TradeStats": {
        "n_round_trips": "closed positions (entry to exit)",
        "n_open": "positions still open after the last bar",
        "n_wins": "trips with pnl > 0",
        "n_losses": "trips with pnl < 0",
        "n_flat": "trips with pnl == 0",
        "win_rate": "n_wins / n_round_trips",
        "gross_profit": "Σ winning pnl",
        "gross_loss": "Σ losing pnl (≤ 0)",
        "net_pnl": "Σ pnl over closed trips",
        "profit_factor": "gross_profit / |gross_loss|; null with no losses",
        "expectancy": "mean pnl per trip",
        "expectancy_ret": "mean of pnl / cost_basis per trip",
        "avg_win": "mean winning pnl",
        "avg_loss": "mean losing pnl",
        "largest_win": "max pnl",
        "largest_loss": "min pnl",
        "win_loss_ratio": "avg_win / |avg_loss|",
        "avg_bars_held": "mean exit_bar − entry_bar",
        "median_bars_held": "median bars held",
        "max_bars_held": "max bars held",
        "avg_units_at_exit": "mean units held at exit",
        "max_units_reached": "the largest unit count any trip reached",
        "avg_adds_per_trip": "mean adds per trip",
        "mean_mae": "mean adverse excursion vs the entry price (min low / P0 − 1)",
        "mean_mfe": "mean favorable excursion vs the entry price (max high / P0 − 1)",
        "exits": {
            "stop_close": "exits by a close below the stop",
            "stop_gap": "exits by an opening print below the stop (close mode)",
            "stop_trade": "exits by the resting stop at the stop level",
            "channel_close": "exits by a close below the channel",
            "channel_trade": "exits by the resting stop at the channel level",
            "end_of_data": "positions marked open at the last bar (== n_open)",
        },
        "entries": {
            "taken": "entries filled",
            "cash_capped": "entries whose size the budget reduced",
            "skipped": {
                "warmup": "firings before first_eligible_bar",
                "in_position": "firings while already long",
                "zero_size": "firings whose unit sized to zero shares",
                "budget": "firings the budget could not afford at all",
                "end_of_data": "firings on the final bar (no next open)",
            },
        },
        "adds": {
            "taken": "adds filled",
            "filled_below_level": "adds whose open was below the add level",
            "skipped": {"budget": "adds that no longer fit the cash"},
        },
        "stop_holds": "adds whose recomputed stop was lower than the stop in force (held)",
    },
    "PeriodicReturns": {
        "monthly": "'YYYY-MM' → compounded return of the bars in that month",
        "annual": "'YYYY' → compounded return of the bars in that year",
    },
    "conventions": {
        "returns": "simple bar returns; the bar before the first is the starting equity",
        "nulls": (
            "every non-finite number serializes as null; a ratio with a zero denominator is null"
        ),
        "time": "ISO-8601 naive stamps, the CSV's own",
        "ranking": "none: cells ride in declaration order, every declared combo present",
    },
}

#: The ``--trades-out`` CSV of the turtle command: one row per round trip.
TURTLE_TRADES_CSV: dict[str, JsonValue] = {
    "command": (
        "seikan stock-turtle-trade-long-only … --trades-out <out.csv> (always overwrites; "
        "silent on success)"
    ),
    "rows": (
        "one per ROUND TRIP — a position from its entry fill to its exit fill — over every cell "
        "and target, closed trips first per target then the open end-of-data mark (is_open 1, "
        "exit_reason end_of_data, exit_px the last close)"
    ),
    "columns": {
        "<swept axes>": (
            "one leading column per swept entry axis, in params order; absent when none"
        ),
        "target": "the target",
        "entry_bar": "bar position of the entry fill (the print of the bar after the firing)",
        "entry_time": "that bar's stamp",
        "exit_bar": "bar position of the exit fill (or the last bar when open)",
        "exit_time": "that bar's stamp",
        "exit_reason": (
            "stop_close | stop_gap | stop_trade | channel_close | channel_trade | end_of_data"
        ),
        "is_open": "1 for the end-of-data mark, else 0",
        "entry_px": "the first fill's price (P0)",
        "avg_entry_px": "cost_basis / shares",
        "exit_px": "the exit fill's price (the last close when open)",
        "unit_shares": "X — the first fill's size, repeated by every add",
        "units": "units held at exit",
        "shares": "shares held at exit",
        "cost_basis": "Σ shares × fill price over the trip's buys",
        "proceeds": "shares × exit_px",
        "pnl": "proceeds − cost_basis",
        "ret": "pnl / cost_basis",
        "bars_held": "exit_bar − entry_bar",
        "n_adds": "adds filled in the trip",
        "adds_skipped_budget": "adds the cash could not cover during the trip",
        "max_stop": "the stop in force at exit (the highest the trip reached)",
        "mae": "min low over the held bars / entry_px − 1",
        "mfe": "max high over the held bars / entry_px − 1",
    },
}

#: The ``--fills-out`` CSV of the turtle command: one row per venue fill.
TURTLE_FILLS_CSV: dict[str, JsonValue] = {
    "command": (
        "seikan stock-turtle-trade-long-only … --fills-out <out.csv> (always overwrites; silent "
        "on success)"
    ),
    "rows": "one per venue fill, in fill order per target, over every cell",
    "columns": {
        "<swept axes>": "one leading column per swept entry axis; absent when none",
        "target": "the target",
        "bar": "the bar the fill belongs to",
        "datetime": "that bar's stamp",
        "at": "open (at the opening print) | trigger (the resting stop inside the bar)",
        "kind": "entry | add | exit",
        "reason": "the exit reason for an exit, empty otherwise",
        "side": "buy | sell",
        "shares": "shares filled",
        "price": "the fill price",
        "notional": "shares × price",
        "cash_after": "the target's cash after the fill",
        "shares_after": "shares held after the fill",
        "units_after": "units held after the fill",
        "stop_after": "the stop in force after the fill (empty when flat)",
        "client_order_id": "the venue order id",
    },
}

#: The ``--equity-out`` CSV of the turtle command: one row per bar per cell.
TURTLE_EQUITY_CSV: dict[str, JsonValue] = {
    "command": (
        "seikan stock-turtle-trade-long-only … --equity-out <out.csv> (always overwrites; silent "
        "on success)"
    ),
    "rows": (
        "n_bars per cell, cells in declaration order (the leading axis columns tell them apart)"
    ),
    "columns": {
        "<swept axes>": "one leading column per swept entry axis; absent when none",
        "datetime": "the bar stamp",
        "equity": "the account after the close: Σ cash + shares × close over the targets",
        "benchmark": "the buy-and-hold curve at the same close",
        "cash": "Σ cash",
        "market_value": "Σ shares × close",
        "gross_exposure": "market_value / equity",
        "n_positions": "targets holding shares",
        "shares[@<target>]": "shares held per target ('@<target>' only with several targets)",
        "units[@<target>]": "units held per target",
        "stop[@<target>]": "the stop in force per target (empty when flat)",
        "add_level[@<target>]": "the next add level per target (empty when flat)",
    },
}
