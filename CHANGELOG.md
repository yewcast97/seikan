# Changelog

seikan carries three independent version stamps beside the package version: `report_schema_version`
(the emitted documents' shape — `cli.REPORT_SCHEMA_VERSION`), `statistics_version` (the meaning of
every estimator — `analysis.stats.STATISTICS_VERSION`) and `gate.policy_version` (the checklist's
semantics — `gate.POLICY_VERSION`). Each entry below names every stamp it moved. A change to a
frozen statistic is only ever a correction, and it bumps `statistics_version`; anything that
merely repackages code changes no number.

## Unreleased — report schema 5, statistics 5, policy 3

Correction release for six defects an external review found in 4.0.0, each a way the engine could
manufacture evidence: warmup read as a truth value, rounding residue read as dispersion or as a
sign, a missing input no ledger saw, a count of evaluations that never happened. Every fix lands in
the fail-closed direction and none touches observer purity, the per-cell reporting doctrine or a
sealed threshold. `statistics_version` moves 4 → 5 (summaries compare only within a version, and
the gate refuses a summary stamped otherwise); `report_schema_version` and `policy_version` do not
move — no key, field or check rule changes, and the one new value is a `sources` label
(`native:<feed>`). Every number below is listed with the inputs that move it; every other summary
number is byte-identical to 4.0.0 (verified by diffing conjunction, market-basket and
cross_mean-basket summaries before and after). Reworded texts ride every report and
`seikan schema`: the embedded `metric_roles` (caveats `baseline` / `cross_breadth` / `rot_p`, the
`cross_breadth` and pooled-riders `evidence_only` lines), the `source_coverage` threshold (holes
count "from its own first_available on", was "after its own first available bar"), the
`signal_coverage` unmet wording and the native remedy; `seikan schema` also rewords
`report_fields.run.sources` / `cross_breadth` and `gate_contract.run_checks.source_coverage`.

- CHANGED the decision layer's warmup rules (`compiler.vectorize`). A still-warming condition node
  has no truth value (`~init ⇒ ~value`, beside `~init ⇒ defined`), and negation waits for its whole
  operand: `not` is initialized only where every condition under its operand is past warmup
  (`vectorize.fully_initialized`, per member column, through `rolling` / `lag` / `first_true` and
  into conditions embedded in operand series). In 4.0.0 `or(not(ema(close, 40) > 0), close < 0)`
  fired — and closed observations — on all 39 warmup bars; `not(or(A_warming, B_false))`, a warming
  `rolling(any)` under an `or`, a phantom True absorbing an `or` over a live hole, and a late
  member's `cross_agg` broadcast through an `or` leaked the same way. `mask(C)` is a number only
  once C is fully initialized — NaN and warmup while any branch warms, its live-branch Trues
  included, since a series of those Trues alone would hand every NaN-skipping reader a
  value-selected history — so a history read over it starts there and a cross-section thins the
  member meanwhile (`cross_breadth` records the `k`). Every Series computed from other Series
  initializes only at a finite value where its inputs are initialized — at the bar a lagged read
  (`shift`, `change`) reaches back to (`_derived_latch`; identical to the finiteness latch for
  plain data) — and a `cross_agg` broadcast waits for the member's own first value as well as its
  init. A Series computed from an input that is initialized with no value yet — an event node whose
  anchor became unknowable before its first event (a domain NaN in E), a mask undecided on its
  first live bar — ledgers that stretch on `signal_coverage` as the input's direct readers do
  (4.0.0 absorbed it as the derived series' warmup, so a domain NaN deleted firings unseen; new
  refusals only). Moves: firings and every cell read of theses with a `not` / `mask` over a
  staggered-warmup operand, a partially warm `rolling` under an `or`, a late member's `cross_agg`
  broadcast read through an `or` / `lag`, directly or through any transform (removals where the
  condition tree reads them directly; a history read or a cross-section over such a `mask` starts
  later and can move either way, and so does that cross node's `cross_breadth` panel — its `k`
  counts the member only once its mask is live); a `not` / `mask` over a partially warm operand
  neither fires nor ledgers there. Signal-level reads (`first_true`, event anchors, cross `where`)
  keep reading the tradable signal — where a branch used to leak a phantom True they now see the
  live branch's real transitions and can GAIN firings (documented residual).
- CHANGED zero dispersion to exact. A window, cross-section or since-event run of bit-identical
  values has its mean equal to the value, a `std` / `mad` / `cross_demean` of 0.0 and an undefined
  `zscore` / `rolling_corr` (4.0.0: `zscore_sma([0.1]*3) == -1`, `[100.1]*3 == +1`,
  `rolling_corr(x, x) == 1`, a constant decimal feed fired `zscore(F, 3) < -0.5` on 294 of 300
  bars); `ema` holds each update inside the OPEN interval its exact value lies in (a flat feed
  drifted off itself; 6% of move-then-hold stretches settled on the wrong side, and 3–23% landed ON
  the held value — every series for windows 3–5 — so `x <= ema(x)` fired on held bars exact
  arithmetic keeps above the mean, and a crossover recipe fired a second time after a hold);
  `zscore` (`ema`) clamps its mean the same way and, once a held input's mean can no longer move,
  carries the deviation exactly, never rounding it to zero — it decays on the held stretch, keeping
  the sign exact arithmetic gives it, instead of locking at `±1/√(1−α)` (a held stretch's `z > 0`
  had read against exact arithmetic on ~10% of held bars; at `alpha ≥ 0.5` a long hold would
  otherwise turn undecidable); `event_agg` `mean` of an all-equal run is its value. A post-warmup
  zero-dispersion bar is undecidable (→ `signal_coverage`, the gate now says "missing or undefined
  input"; a cross node over such an input no longer counts the member there, so its `cross_breadth`
  panel moves too). Statistics: `t_hac` / `hac_se` null on a near-constant pool (4.0.0 read ~1e16);
  `std_ret` exactly 0.0 on a constant pool (by_target, pooled, baseline); `benchmark_regression`
  declines a near-constant bench leg and fits `beta` 0.0 (alpha the raw leg's mean, `r2` null —
  nothing to explain) on a near-constant raw leg; CSCV split scores are NaN on a split pool that is
  constant (per-block extrema), whose largest deviation is within ~10 ulps of its mean (the HAC's
  gate; a single-level one-ulp pool scored ~1e15 and won every split, `pbo` 0.0) or that varies
  only within its accumulation error (4.0.0 scored ~30% of such splits at ±1e7), moving `pbo` and
  its diagnostics where a candidate had such splits; `boot` is degenerate exactly — `boot_se` 0.0 —
  on a constant pool (its interval on the pool's arithmetic mean) and on any pool whose resampled
  means are bit-identical on every draw; a constant pool has no `bucket_monotonicity` entry.
  Feature snapshots (`params.features`) read the same kernels, so a feature over a bit-identical
  window moves with them — the trades CSV feature columns, `conditional_buckets`,
  `bucket_monotonicity` and `feature_association` (a `zscore` feature over a held decimal level
  reads NaN, not ±1, and leaves the buckets); a `mask` feature is likewise NaN until its condition
  is fully initialized. Every window, cross-section and pool that varies is computed as before, bit
  for bit; the EW recurrences differ only on the steps where a rounded update left its exact
  interval or a held input's deviation is carried (rounding-scale steps tick data with repeated
  values can take too).
- ADDED `native:<feed>` to the `sources` panel: a feed a `native` node reads on its own clock
  starts where its first finite print arrives and is missing on every bar whose interval absorbed a
  non-finite print, or whose anchored value is missing (a NaN print stays missing until the next
  finite print arrives, as on the `external` leaf) — the first bar's interval being the whole
  pre-interval print history, which `data.start` cannot exclude (every consumed print counts,
  whatever the expression's reach). In 4.0.0 a NaN print between two bars was skipped by a native
  `ema` and seen by no ledger; such runs now refuse on `source_coverage` (remedy: repair or trim
  the feed; `data.end` can end the interval before a hole inside it). The bar-clock leaves are
  counted as before; holes count from each leaf's `first_available` on.
- CHANGED the rotation null's tie handling: the coincidence count `den` is an exact integer and
  ties are judged within the FFT's rounding (an absolute band, `16·ε·log2(T)·‖f‖₂·√k / den`).
  `rot_p` moves up wherever shifts tie the observed mean exactly — a forward series of identical
  returns now reads exactly 1 (4.0.0: as low as 0.016), and outcomes whose values are exactly
  representable (count series, a `diff` on a binary-exact tick such as 0.25) count their exact ties
  (4.0.0 read p up to ~0.1 low); a `diff` of decimal-quoted levels counts its tick-level ties only
  where the quotes' representation residue (~ε·level) sits inside the band, which grows with the
  series length (~`log2(T)·√(T/k)`) — and reads them as variation above that, as 4.0.0 did (with
  0.01 ticks from roughly 50 on a year of daily bars to a few hundred on ten years; 5-decimal FX
  quotes always). A `cross_mean` basket-wide pool counts its ties only where the demeaning's
  residue (~ε·|raw|) sits inside the band: ordinary dispersed baskets now read 1 (4.0.0: coin
  flips), while legs equal but for their last bits (one instrument in two quote units) still read a
  coin flip, as in 4.0.0 — the pool's reads other than `mean_ret` keep ordinary rounding. Dispersed
  returns are byte-identical.
- CHANGED the basket `cross_mean` pooled `mean_ret` (and the baseline's pooled `mean_ret`) to be
  summed from the raw legs (`analysis.stats.cross_mean_pooled_mean`): a bar on which every member
  closed, or whose legs are all equal, contributes exactly 0, in any declaration order. In 4.0.0 a
  basket-wide firing read ±1e-19 with its sign set by the targets' order, and `support` flipped
  with it; the baseline's "≈ 0 by construction" is now exactly 0.0. Every `cross_mean` pooled
  `mean_ret` is this exact sum, so it can move in its last bits on any basket; legs so large the
  sum leaves the double range read NaN (refused), never a crash. The `cross_mean` benchmark leg
  takes an all-equal bar's value exactly, so on a basket whose legs are bit-identical (a shared or
  duplicated outcome feed) every per-row excess is exactly 0.0 and every read of it moves to its
  exact value: by_target / pooled / baseline means, `hit_rate`, `std_ret`, quantiles, `t_hac` /
  `hac_se` (null), `rot_p` (1.0), `boot`, `subperiods`, `concentration` and `member_share` (null —
  the gate's concentration check reads them missing), `bucket_monotonicity` (entries dropped),
  `conditional_buckets` (bucket means and hit rates 0.0), `feature_association` (null,
  `no_rank_variation`), the shape reads (`skewness` / `kurtosis` / `win_loss_ratio` /
  `profit_factor` null; `tail_ratio`, `cvar_5`, `worst_ret` / `best_ret` exact), `episode_profile`,
  `episode_stats` (cluster-mass fields null) and the `episodes` ledger (`share_abs` null),
  `benchmark_regression` (last bits), the grid-level `pbo` block (every split pool constant: null,
  `reason: insufficient_data`, after the block fallbacks), `mean_ret_bench`, and the trades `ret` /
  `ret_bench` columns. Market / unbenchmarked pools and conjunction runs are unchanged.
- CHANGED `cross_breadth`'s `k` to the members a cross kernel actually reduced over
  (`vectorize.cross_reduced`: the entering members of every population whose reduction produced a
  value — a member whose own demeaned value overflowed still counts). Moves only where they differ:
  grouped nodes with groups below `min_valid` (4.0.0 reported 80 bars evaluated and a full basket
  from bar 0 for a node that never produced a value) and populations whose reduction overflowed.
  This change leaves ungrouped panels on ordinary data byte-identical; the decision-layer and
  zero-dispersion changes above move the inputs some cross nodes reduce over, and their panels with
  them.
- Library: `vectorize.fully_initialized`, `vectorize.cross_reduced`,
  `analysis.stats.cross_mean_pooled_mean`, `sources.LeafMasks` (`source_availability` now returns
  `{label: LeafMasks}`), `MarketData.full_memo` (cleared by `clear_memo()`),
  `nb.mask_apply_nb(value, init, defined, full)`, and the public walks
  `dsl.schema.iter_source_leaves` / `series_source_leaves`, which also yield `("native", feed)` for
  a `native` node beside the `("external", feed)` leaf under its `expr`.

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
