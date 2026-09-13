# Changelog

seikan carries three independent version stamps beside the package version: `report_schema_version`
(the emitted documents' shape — `cli.REPORT_SCHEMA_VERSION`), `statistics_version` (the meaning of
every estimator — `analysis.stats.STATISTICS_VERSION`) and `gate.policy_version` (the checklist's
semantics — `gate.POLICY_VERSION`). Each entry below names every stamp it moved. A change to a
frozen statistic is only ever a correction, and it bumps `statistics_version`; anything that
merely repackages code changes no number.

## 4.0.0 — report schema 5, statistics 4, policy 3

DSL expressiveness release: the entry vocabulary grows so a thesis can say what it means instead
of substituting a nearby question; no stamp moves and every emitted number is byte-identical to
3.0.0 for every document 3.0.0 accepted.

- CHANGED `canonical_dsl_hash`: the normalized payload now drops every `null`-valued optional field
  (`exclude_none`) beside filling defaults and sorting keys. Every `dsl_hash` moves exactly once
  (no stored theses or reports exist); from here on a new `None`-defaulted optional field is
  hash-safe like a new node type — it appears in the payload only when set — while a non-`None`
  default still moves every hash.
- ADDED the event algebra: Series nodes `mask(condition)`, `bars_since_event(event)`,
  `event_value(event, input)` and `event_agg(event, input, agg: sum|max|min|mean)`, and the
  Condition node `lag(condition, periods)` (sweeps as `lag_periods`). An event is a condition's
  tradable signal; `s(t)` is the latest event bar ≤ t (current bar included, latest wins); before
  the first event a node is warmup, after a post-warmup hole in the event condition it reads NaN
  while initialized (→ `signal_coverage`). `mask` is depth-transparent; the three event nodes
  count one level over their input; an embedded condition adds no depth and its threshold operands
  are depth-checked as roots of their own and listed by `--root-series-out`. Sweeps inside an
  embedded condition register once, in engine order (the condition before the node's input).
  `render_condition` joins `render_series` on the `dsl.schema` facade; `iter_condition_series`
  now yields embedded-condition operands beside the outer ones.
- ADDED `native(name, expr)`: a transform evaluated on an external feed's own clock (over its
  native post-lag prints, every print seen) and only then asof-anchored onto the bars — windows
  count releases, not bars. `expr` reads exactly that feed plus constants through the single-input
  time transforms, `binary_op` and `rolling_corr`; anything on the bar clock refuses by name.
  Counts one level over `expr`. `MarketData.externals_native` / `external_native(name)` retain
  each feed's native prints (a hand-built `MarketData` must supply them to serve a `native` node).
- ADDED the cross-sectional population model: optional `where` (a Condition — eligibility;
  a warming or decided-False member is excluded, any member post-warmup undefined voids the
  whole bar, fail-closed) and `group` (a Series of point-in-time labels — reduce within each
  finite label, `min_valid` per group, `cross_agg` broadcasts per group) on `cross_rank` /
  `cross_demean` / `cross_agg`. The canonical idiom is `and(E, cross_rank(x, where=E) >= q)`.
  Grouping wraps the existing kernels per label (`nb.cross_grouped_apply_nb`; one label is
  bit-exact with the ungrouped result). `cross_breadth.k` now counts ENTERING members (finite
  input, eligible, finitely labelled) — identical to before for nodes without selectors; no
  report field moves. Sweep axes inside a cross node register input → where → group.
- ADDED estimator forms: `ema` / `zscore` take exactly one of `window` or `alpha` (an explicit
  EW decay, `0 < alpha ≤ 1`, warmup `ceil(2/alpha − 1)`; bit-exact with the window form at
  `alpha = 2/(window+1)`; sweeps as `ema_alpha` / `zscore_alpha`; `zscore` alpha needs
  `mean_type: "ema"` and `alpha < 1`); `rolling_agg` gains `median` and `mad` (unscaled median
  absolute deviation about the same window's median; trailing only); `calendar` gains `hour` and
  `minute`. `nb.ema_params(window, alpha)` is the one `(alpha, warmup)` derivation; the EW
  kernels take that pair. The `window` form's payload is unchanged, so an existing `ema` /
  `zscore` document's hash does not move beyond the `exclude_none` change above.
- ADDED shared sweep axes: a top-level `"axes": {"N": [20, 60]}` declares one hypothesis axis
  that any sweepable ENTRY param reads with `{"axis": "N"}` (transform `window`/`periods`/
  `alpha`, `constant.value`, `rolling.window`, `first_true.cooldown`, `lag.periods`; never
  `params.horizon`, never inside `params.features`). Recorded once in `summary.params` /
  `cells[].params` / the trades CSV and counted once toward the cap; every axis value is
  re-validated against every referencing param at parse; an undeclared reference, an
  unreferenced axis, an empty `axes` and a name colliding with an auto-named axis, a swept
  constant's name or a reserved column refuse at parse. Library signatures:
  `collect_sweeps(entry, axes=None)`, `iter_param_assignments(entry, axes=None)`,
  `declared_grid_size(entry, horizon, axes=None)`.
- Depth accounting: `bars_since_event` / `event_value` / `event_agg` / `native` each count one
  level (over their input / expr); `mask` is transparent; an embedded Condition adds nothing to
  its embedding node and its operands are roots of their own; a cross node's `group` counts as a
  second child, its `where` operands are roots. `MAX_SERIES_NESTING` stays 5.
- NaN-gating contract, unchanged in kind: every new node yields NaN (→ an undefined threshold →
  `signal_coverage`) on a post-warmup hole, and raw leaves under an embedded condition or a
  native expr still flow to `source_coverage`. No stamp moves: `report_schema_version` 5,
  `statistics_version` 4, `policy_version` 3; every number a 3.0.0 document produced is
  byte-identical (verified by diffing full conjunction and basket reports before/after — only
  `identity.dsl_hash` differs, per the `exclude_none` change).

## 3.0.0 — report schema 5, statistics 4, policy 3

Audit release: statistically inappropriate or degenerate report content removed, every surviving
number byte-identical to 2.0.0.

- REMOVED `summary.stats_table`, `summary.by_target`, `summary.by_param` and `n_stats_rows` — the
  legacy fired-pools-only breakdown (a duplicate of `cells[].by_target`, the panel that also
  carries the cells that never fired) and its rollups, which averaged per-cell means across
  horizons and hypotheses: a number in no unit that moved with grid composition, and the one
  cross-cell aggregate in an engine that forms none. `params` and `targets` stay as run-level
  stamps.
- REMOVED `t_iid` / `p_iid` (an iid one-sample t-test on observations that overlap by
  construction; the report itself tagged it known-invalid) together with the `nominal` metric
  class — `metric_classes` is now the three-value vocabulary `descriptive | inference |
  integrity`.
- REMOVED the derivable per-cell `sharpe` (`mean_ret / std_ret`), `firing_rate`
  (`outcome_coverage[t].n_attempted / n_bars`), `median_ret` (`ret_quantiles.p50`),
  `mean_mae` / `mean_mfe` (the excursion blocks' own `mean`) and the constant
  `mean_bars_held` / `median_bars_held` / `max_bars_held` (identically the horizon).
- Trades CSV: the constant `bars_held` column is replaced by an ALWAYS-present `horizon` column
  right after the swept axes and before `target`, so a row names its measurement window whether
  or not the horizon was swept.
- ADDED integrity reads: `rot_n_null` on every per-target and pooled panel (the number of
  DEFINED shifts a cell's rotation null was formed over — its own p floor is `1/(1 + rot_n_null)`,
  which on a sparse mask sits above the run-level `rotation.p_resolution`), and
  `pbo.n_splits_attempted` / `pbo.n_candidates_min` (the C(S, S/2) the partition offered against
  the splits scored, and the smallest finite candidate population any scored split ranked over —
  canonical CSCV assumes a fixed count, and this is how far the block-local thinning departed).
- The basket cell's `pooled.member_share` now sits last in the pooled block (key order only).
- Caveats state the tie rule shared by `hit_rate` (a zero return is not a hit) and
  `win_loss_ratio` / `profit_factor` (zeros join neither side), that `skewness` / `kurtosis` are
  scipy's population moments (`bias=True`), that `tail_ratio` is a spread ratio unless
  `p05 < 0 < p95`, and that the top-5% concentration set is a single observation at `n ≤ 20`.
- `newey_west_mean` returns `(t, se)` — the p-value it computed was never emitted, and a consumer
  derives one at `df = n_nonoverlap − 1`.
- Episode timestamps (`episode_stats.largest_cluster_start`, `episodes.entries[*].start` /
  `end`) render as plain ISO-8601 seconds like every other timestamp in the report; they carried
  a fractional-second suffix before.
- Library: `api.compile_thesis` validates the summary it returns against the `RunSummary`
  contract; the closed vocabularies (block reasons, stamps, issue codes, threshold provenance,
  error types) are typed Literals; `MarketData.cache` became three typed memos with
  `clear_memo()`; the facades (`seikan.gate`, `seikan.analysis.stats`, `seikan.dsl.schema`,
  `seikan.types`) export public names only; `seikan` itself exports `canonical_dsl_hash`;
  `dataio.bar_spacing` is the one clock-geometry stamp; the path/source kernels in
  `compiler.paths` / `compiler.sources` are public.

## 2.0.0 — report schema 4, statistics 4, policy 3

- Report schema v4: `identity` gained `environment` (the python/numpy/pandas/scipy/numba versions
  the numbers were computed under) and the `pbo` block gained `oos_degradation_slope_reason`.
- Statistics v4 (corrections, numerical and domain): the variance family is computed CENTERED
  (zscore-SMA two-pass, zscore-EMA via the West recurrence, cross_agg std two-pass) where the
  one-pass E[x²] − E[x]² forms cancelled catastrophically at large input levels; every kernel and
  binary op enforces the finite-or-NaN invariant; `change(kind="pct")` and `drawdown` / `runup`
  are positive-domain; the event-time HAC uses the n − 1 covariance divisor (the sparse-pool
  reduction to the iid SE is exact) with a canonical (bar, ret) tie order; the episode panels
  sort stably with a target tie-break and the episode bootstrap's content-keyed seed hashes the
  same canonical order; CSCV split scores run on SHIFTED moments and `oos_degradation_slope` is
  a centered OLS that nulls with a reason on degenerate train scores; zero-dispersion pools null
  their skewness / kurtosis instead of minting unreliable moments.

## Earlier stamps

- Report schema v3: `n_eff` renamed `n_nonoverlap` everywhere it is emitted, with the knob
  `thesis_min_n_nonoverlap` and its flag / env twins (same greedy non-overlapping count; non-overlap
  is not independence, and the old name claimed it was); the summary gained `cross_breadth`; the
  `pbo` block gained `n_combos_scoreable` / `n_combos_declared`; `episode_stats` gained
  `mass_hhi` / `effective_n_clusters`; multi-target bucket records are labelled `q1..qk`.
- Report schema v2 / policy v2: the gate section's booleans renamed `passed` / `n_passed` →
  `met` / `n_met`; no check's branch logic changed.
- Policy v3: `source_coverage` refuses a leaf whose `first_available` is null (an input never
  available on any bar — the one hole size that passed, since a decisive sibling settles the root
  over it), and the support floor reads the renamed `n_nonoverlap`.
- Statistics v3 (corrections): the excursion / trough window stops reading the exit bar's
  high / low (full H/L over `[fill, fill+h−1]` plus only the exit print); CSCV test ranks are
  midranks, byte-identical candidate combos collapse to one before scoring, and the block carries
  the population ledger `n_combos <= n_combos_scoreable <= n_combos_declared`; `edge_ratio` pairs
  its rows; conditional-bucket edges are per target; `episode_stats` added `mass_hhi` /
  `effective_n_clusters`.
- Statistics v2 (additions, no v1 number changed): the p05 / p95 quantile tails, the per-cell
  shape reads, `benchmark_regression`, `timing` and the episode-deduplicated `episode_profile`.
