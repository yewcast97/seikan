# Changelog

seikan carries three independent version stamps beside the package version: `report_schema_version`
(the emitted documents' shape — `cli.REPORT_SCHEMA_VERSION`), `statistics_version` (the meaning of
every estimator — `analysis.stats.STATISTICS_VERSION`) and `gate.policy_version` (the checklist's
semantics — `gate.POLICY_VERSION`). Each entry below names every stamp it moved. A change to a
frozen statistic is only ever a correction, and it bumps `statistics_version`; anything that
merely repackages code changes no number.

## Unreleased — report schema 5, statistics 5, policy 3

Correction release for six defects an external review found in 4.0.0 and for eight more 4.0.0
defects that a second review round of the unreleased v5 build found, each a way the engine could
manufacture evidence. The first six: warmup read as a truth value, rounding residue read as
dispersion or as a sign, a missing input no ledger saw, a count of evaluations that never happened.
The second eight, each older than the first round (the before-values quoted for them are 4.0.0's and
the first-round build's alike): a data hole read as a known `first_true` decision, a basket's
rotation null formed without a member whose only firing was the final bar, rounding residue read as
a `cross_mean` full bar's sum, a ragged CSV read as clean data, an out-of-range intermediate read as
a value or crashing the report (`x / inf` as 0, a subnormal variance as zero dispersion, an
overflowed episode mean failing the summary's own contract), an unrepresentable quotient read as a
missing log, a parameter the schema admitted that no kernel could execute, and a declined CSCV block
that claimed no candidate ever existed. Every fix lands in the fail-closed direction — where a
ledger now passes or a cell now fires, the decision or the value was in fact known (an absorbed
`first_true` hole, a correlation an overflow read as 0, the exact log an unrepresentable quotient
hid) — and none touches observer purity, the per-cell reporting doctrine or a sealed threshold.
`statistics_version` moves 4 → 5 (summaries compare only within a version, and the gate refuses a
summary stamped otherwise); statistics 5 is still unreleased, so the second round folds into it and
no stamp moves again. `report_schema_version` and `policy_version` do not move — no report key or
check rule changes; each new value joins an existing field's vocabulary: the `sources` label
`native:<feed>`, the `data_report` issue code `field_count_mismatch`, and `not_representable` as a
describe `ratio_reason`, a describe block `reason`, a `benchmark_regression` `reason` and a `boot`
`reason`; and the two fields that become nullable, `episodes.entries[].mean_ret` and
`conditional_buckets.<feature>.buckets[].mean_ret`, are null only where a mean's finite returns
sum past the double range, which the previous build crashed on (exit 4) or, for a bucket, never
computed over that bin, having refused the feature (`insufficient_distinct_values`) or merged its
bins over edges it overflowed or could not label — so the widening admits only values the previous
build never emitted, and every document it did emit stays valid. The DSL gains parameter bounds
(every bar-count parameter at most 2^31 − 1, the EW `alpha` at least 2^-30): a thesis past them now
refuses at parse (exit 3), `seikan schema`'s DSL JSON Schema carries them as `maximum` / `minimum`,
and every `dsl_hash` is unchanged. Every number below is listed with the inputs that move it; every
other summary number is byte-identical to 4.0.0 (first round: verified by diffing conjunction,
market-basket and cross_mean-basket summaries before and after; second round: 49 CLI fixtures —
`run`, `describe`, `check-data` and `schema` — diffed bitwise against the first round's build, ten
of them designated controls: the four `describe` / `check-data` controls are byte-identical and five
of the six `run` controls differ only in the reworded `metric_roles` line below, the sixth — a
`pct`-outcome run over 1e±198 levels — also moving its default `vol_14` feature and dropping its
pct-outcome RuntimeWarning, as the log-ratio entry states).

Reworded texts ride every report and `seikan schema`: the embedded `metric_roles` (caveats
`baseline` / `cross_breadth` / `rot_p`, the `cross_breadth` and pooled-riders `evidence_only` lines
— the pooled-riders line reworded again in the second round, which changes no other text a run
report carries), the `source_coverage` threshold (holes count "from its own first_available on",
was "after its own first available bar"), the `signal_coverage` unmet wording and the native
remedy. `seikan schema` also rewords `report_fields.conventions.nulls`, `report_fields.run.sources`
/ `cross_breadth` / `pbo`, `report_fields.cells.by_target.benchmark_regression` / `by_target.boot` /
`by_target.rot_p / rot_n_null`, `report_fields.cells.episodes` / `episode_profile` /
`conditional_buckets / bucket_monotonicity` / `pooled` / `pooled.rot_p` / `pooled.t_hac / hac_se`
and the pooled-riders key (`pooled.mean_ret_raw / … / edge_ratio`),
`gate_contract.run_checks.source_coverage`, `gate_contract.metric_roles_rationale` (caveats
`baseline` / `benchmark_regression` / `boot` / `conditional_buckets` / `cross_breadth` /
`episode_profile` / `pbo` / `rot_p`, the `benchmark_regression` / `boot` / `cross_breadth` /
`episode_profile` / pooled-riders `evidence_only` lines and the `source_coverage` line of
`run_checks`), `trades_csv.derived_views` (the `cross_mean` full-bar rule and the (entry_time,
target) row order of the ledger rebuild), `entry_flags_csv.relation_to_trades` (the rotation nulls
still see a final-bar firing), `describe_report.blocks` (`changes[N]`, `dispersion[N]`,
`range_position[N]`, `full_sample`, `reasons`) and `csv_format.encoding`, and adds the entry
`csv_format.record_shape`. The DSL guide (`seikan schema --markdown`) restates the corrections where
they apply — in the second round the data contract's record shape, the zero-dispersion passage (its
normal-range qualifier, the out-of-range rule, where an overflow lands and the EW z-score's
residual), `ema` (the `alpha` floor and where the two EW forms' warmups agree), `change` and the
outcome's `log` (scale-safe), `rolling_corr`, `first_true` (hole completions), the final-bar firing,
the bar-count ceiling, the trades-CSV ledger rebuild, the `pooled` panel (the `cross_mean` identity
and its reach, the masks its `rot_p` rotates), `episodes` and `conditional_buckets` (null means, bin
labels, spanning features), `rot_n_null`, `boot`, `source_coverage`, `summary.pbo` and `describe`.

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
  otherwise turn undecidable; a hold whose mean lands ON the held value from the bar the input moved
  carries nothing, a residual the out-of-range entry below documents); `event_agg` `mean` of an
  all-equal run is its value. A post-warmup zero-dispersion bar is undecidable (→ `signal_coverage`,
  the gate now says "missing or undefined input"; a cross node over such an input no longer counts
  the member there, so its `cross_breadth` panel moves too). Statistics: `t_hac` / `hac_se` null on
  a near-constant pool (4.0.0 read ~1e16); `std_ret` exactly 0.0 on a constant pool (by_target,
  pooled, baseline); `benchmark_regression` declines a near-constant bench leg and fits `beta` 0.0
  (alpha the raw leg's mean, `r2` null — nothing to explain) on a near-constant raw leg; CSCV split
  scores are NaN on a split pool that is constant (per-block extrema), whose largest deviation is
  within ~10 ulps of its mean (the HAC's gate; a single-level one-ulp pool scored ~1e15 and won
  every split, `pbo` 0.0) or that varies only within its accumulation error (4.0.0 scored ~30% of
  such splits at ±1e7), moving `pbo` and its diagnostics where a candidate had such splits; `boot`
  is degenerate exactly — `boot_se` 0.0 — on a constant pool (its interval on the pool's arithmetic
  mean) and on any pool whose resampled means are bit-identical on every draw; a constant pool has
  no `bucket_monotonicity` entry. Feature snapshots (`params.features`) read the same kernels, so a
  feature over a bit-identical window moves with them — the trades CSV feature columns,
  `conditional_buckets`, `bucket_monotonicity` and `feature_association` (a `zscore` feature over a
  held decimal level reads NaN, not ±1, and leaves the buckets); a `mask` feature is likewise NaN
  until its condition is fully initialized. Every window, cross-section and pool that varies is
  computed as before, bit for bit, while its variances (and, for `rolling_corr`, their product) are
  normal doubles (see the out-of-range entry below); the EW recurrences differ only on the steps
  where a rounded update left its exact interval or a held input's deviation is carried
  (rounding-scale steps tick data with repeated values can take too), and the EW z-score where its
  plain recurrence left the normal range.
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
  quotes always). A `cross_mean` basket-wide entry counts its ties only where the demeaning's
  residue (~ε·|raw|) sits inside the band: ordinary dispersed baskets now read 1 (4.0.0: coin
  flips), while legs equal but for their last bits (one instrument in two quote units) still read a
  coin flip, as in 4.0.0 — the pool's other bar-closed reads (`boot`, `subperiods`, the `episodes`
  ledger and `episode_profile` means, the HAC), the grid-level CSCV split means and the
  conditional-bucket means of a full bar held whole in one bucket take the identity exactly (see the
  `cross_mean` full-bar entry below), while `rot_p` takes it only within its tie band. Dispersed
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
- CHANGED `first_true` over a holed child to carry every completion of the holes
  (`nb.first_true_1d`). A post-warmup hole in the child reset the transition machine, tainted
  `max(1, cooldown) + 1` bars and then declared the decision known again — but the completions can
  disagree for much longer (a hole that may have shifted a cooldown's phase keeps them apart for
  good), and the kernel fired one completion's phase as if it were known, while an `or` sibling
  True over the short taint absorbed every undecided bar (T ∨ U = T) and left no ledger trace: the
  review's `or(first_true(C, cooldown=3), H)` — C alternating with a hole at bar 5, H True on bars
  5–8 — read 0 undecidable bars of 60 and fired bars 11, 15, …, 59 of one phase; it now reads 26
  and fires only the bars every completion agrees on. The kernel carries the set of machine states
  every completion can be in (warmup resets it to one state, a hole in warmup is warmup, at most
  `4 + cooldown` states live, O(1) amortized per bar): a bar fires where every completion fires and
  is decided exactly where all of them agree. Moves, only on a `first_true` whose child has a
  post-warmup hole and on every node that reads it: its definedness, in both directions — up where
  the completions disagree past the old taint (one hole in a periodic child under a cooldown can
  leave the rest of the series undecidable, a new refusal), down where the taint over-marked (the
  hole bar is decided False when no completion could fire on it — all cooling down, the bar before
  it True, or the child not yet past its first False — and at cooldown 0 the bar after a hole is
  undecided only if it is True; a target with a 12-bar data gap under
  `first_true(close > rolling_agg(close, 5, mean))` reads 14 undecidable bars where it read 18 at
  cooldown 2, 12 where it read 20 at cooldown 4), and a hole whose completions re-converge before
  they disagree on any fire is absorbed like `F ∧ U = F`, so a cell whose only undecided bars were
  such holes now meets `signal_coverage` (a raw-leaf hole there still refuses on `source_coverage`;
  a NaN computed inside the expression is ledgered nowhere, as an absorbed one already was); its
  firings at and after a hole (a fire every completion agrees on is emitted where the reset machine
  had forgotten `seen_false` or the cooldown, a fire only one completion makes is not); and every
  statistic of the cells whose firings moved, `pbo` with them. A child without a post-warmup hole
  is one completion — the old machine — and bit-identical, as is every node that reads no holed
  `first_true`.
- CHANGED the basket's pooled rotation roster (`compiler.runner._measure_grid`): a fired member's
  COMPLETE mask joins the common-shift null before its firings are filtered to the fillable ones. A
  member whose only firing was the final bar — no next open, so no row and no `outcome_coverage`
  count — was dropped from the roster, so the common shift rotated a cross-sectional firing pattern
  the thesis never produced, against the roster's own doctrine (every FIRED member, closed rows or
  not; a member that also fired elsewhere already passed its complete mask). Moves, only on basket
  cells (unbenchmarked, `market` or `cross_mean`) where some member's only firing is the final bar:
  `pooled.rot_p` (a two-member basket over 80 bars at h = 5: 0.2405 → 0.25) and `pooled.rot_n_null`
  where the rotated firing defines a shift the other members leave undefined (78 → 79 there) — a
  null `pooled.rot_p` becomes a number where the other members alone left every shift undefined,
  and a cell whose every fired member fired only on the final bar now reports its defined-shift
  count beside a null `rot_p`, as an all-censored cell already did. No check reads either field, so
  no verdict moves; every other pooled read, every `by_target` panel (its null still admits only
  members with closed rows), `pbo`, the baseline, the trades CSV and every conjunction run are
  byte-identical.
- CHANGED the basket `cross_mean` reads that sum whole bars — the pooled panel's, the grid-level
  CSCV split means and the conditional-bucket means — to take the zero-sum identity the pooled
  `mean_ret` already reads. The first round re-summed only that mean from the raw legs; every other
  read that sums whole bars still added the rounded per-row excesses, so a FULL bar (every member
  closed in the cell) contributed its rounding residue to the bootstrap, the era means, the
  `episodes` ledger and `episode_profile` means, the pooled event-time HAC (whose long-run variance
  reads the rows only through per-bar deviation sums, same-bar pairs at weight 1), the CSCV split
  scores and the bucket means — the last two ranking residue in the targets' declaration order. The
  review's basket — members priced 2^t, 4^t and 16^t at h = 1, every bar full — read a zero-width
  `boot` interval at 5.9e-16, era and ledger means of 5.9e-16, `episode_profile.hit_rate` 1.0 with a
  win streak across every episode, and a `hac_se` that was residue or null by the series length
  (1.09e-8 at 12 bars, null at 11 or 13); a three-member grid firing every member on the same bars
  read `pbo` 0.29 to 0.34 over its six target orders. `runner._cells_panel` now hands the pooled
  reads a per-row pool value — 0.0 on a full bar's rows, the row's `ret` everywhere else; internal,
  never a trades column: `boot` adds it to its episode sums (episodes, the canonical (bar, ret)
  order, the seed and so the draws stay on `ret`), the `subperiods` means, the `episodes` ledger
  means and `episode_profile`'s per-episode means read it (and everything over them: `hit_rate`,
  `mean_ret`, `profit_factor`, quantiles, `worst_ret` / `best_ret`, the streaks), and the pooled
  `hac_se` is re-read over it, `t_hac` the exact `mean_ret` over that SE. A CSCV split holding a
  full bar takes its MEAN from the same bar-closed sums (a block holds whole bars), its sd,
  dispersion gates and rounding floor staying on the rows; a full bar whose rows all land in ONE
  bucket ordinal enters that bucket's `mean_ret` at 0.0, while a bar the per-target edges split
  across ordinals, or leave partly unbucketed, keeps each row's `ret` (a bucket's `n` and `hit_rate`
  stay the rows' own). Moves, only on `cross_mean` baskets holding a full bar: a pool made only of
  full bars reads `boot` 0.0 / 0.0 with `boot_se` 0.0, era, ledger and per-episode means 0.0
  (`hit_rate` 0.0, both streaks 0, `profit_factor` null — no win and no loss) and `t_hac` / `hac_se`
  null (no long-run variance, as on identical legs); a mixed pool moves `t_hac` / `hac_se`, `boot`,
  the era means and a mixed episode's mean at rounding scale, while an episode made only of full
  bars goes from a residue win or loss to exactly 0, so the profile's `hit_rate`, streaks, quantiles
  and `profit_factor` can move by whole episodes (the power basket with members entering at bars 6,
  11 and 21 of 30: `hit_rate` 1.0 → 15/22, `max_win_streak` 22 → 15). On a grid whose cells are all
  full every split the `pbo` block scores is a full tie (λ = 0), so `pbo`, `lambda_mean` and
  `prob_oos_loss` read 0.0 whenever it scores (0.0 in all six orders of the three-member grid
  above); every train score tied, each split goes to the first-declared candidate and a split whose
  winner has no test pool is skipped, so a sparse all-full grid can score fewer splits, fall back to
  a coarser S (`blocks`) or decline `insufficient_data` (`pbo` null), in any target order alike. On
  a mixed grid a split holding only full rows — every split of an all-full candidate, some of a
  partly-full one — now scores exactly 0 where its residue's sign counted as a test loss and ordered
  it against another candidate's residue, which can move any field of the block, on a grid with no
  all-full candidate too (three two-decimal random walks, levels 30 / 34 / 38 at h 2 / 5: `pbo`
  0.157 → 0.086, `prob_oos_loss` 0.914 → 0.8); a partly-full candidate's other splits move their
  means in the last bits. A bucket of whole full bars alone reads `mean_ret` exactly 0.0, and a
  feature whose every bucket is such (an all-full pool under a basket-wide feature — a `calendar`
  field, a shared `external` feed) has no `bucket_monotonicity` entry; any other bucket holding a
  whole full bar moves its `mean_ret` in the last bits, and `bucket_monotonicity` moves where the
  exact zeros reorder the buckets (two all-full buckets of a mixed pool now tie). Byte-identical:
  every conjunction run, every `market` or unbenchmarked basket, every `cross_mean` grid without a
  full bar (its HAC, `pbo` and buckets included), every `by_target` panel, every row-level read
  (`std_ret` and the shape reads, quantiles, row `hit_rate`, `win_loss_ratio`, `profit_factor`,
  `concentration`, `member_share`, `episode_stats`, a bucket's `n` and `hit_rate`), `rot_p`,
  `n_nonoverlap`, `feature_association`, the baseline, every gate verdict and the trades CSV.
  Documented limit, unchanged: `rot_p` takes the identity only within its tie band (the
  rotation-null entry above).
- CHANGED the strict CSV reader (`dataio.read_strict_csv`, behind `run`, `check-data`, `describe`
  and `load_market_data` alike) to check record shape before pandas parses: every record must carry
  exactly the header's field count, and a record with more or fewer refuses (exit 2) under the new
  `data_report` issue code `field_count_mismatch` — every offender counted, the first five named by
  the csv line the record starts on, with its field count as `value`. pandas repairs record shape
  in silence: when EVERY record carries one field too many (a leading row number, a trailing
  delimiter on every data row) it reads the first field as an implicit index, so each header name
  reads the field one place to its right — `datetime,close` over `2021-01-01,2021-01-04,100` loaded
  as 2021-01-04 → 100, and `seikan run` measured it at exit 0 — and a short record was padded into
  a NaN hole the file never had. Moves: such files refuse as `field_count_mismatch` where they
  loaded or refused as `bad_timestamp` / `non_numeric_value`; so do a single ragged later record
  (was `unreadable`, naming one line) and a quoted blank line (was `bad_timestamp`); the header is
  the first non-blank line, for the duplicate-name check too (a blank first line before a
  case-folded duplicate header crashed with exit 4, now `duplicate_column`; a file of only blank
  lines is `empty_file`, was `unreadable`); an unterminated quote, which swallows the rest of the
  file into one field, refuses naming the line it opens on — `field_count_mismatch` where that
  leaves the record off the header's field count, `unreadable` once it swallows more than 131072
  characters — and otherwise as before (`unreadable`, pandas' EOF-inside-string message naming a
  0-based row); a field past that limit refuses as `unreadable` naming its record's line (it crashed
  in the header; in a data record pandas parsed it, so a long decimal literal loaded and a long
  integer literal refused as `non_numeric_value`) unless an earlier refusal, a duplicate header
  name, comes first — the limit is the stdlib default, now pinned for the read and handed back on
  every exit, so a process that moved `csv.field_size_limit()` (the `sys.maxsize` idiom, scipy's
  ARFF reader) gets the CLI's verdict on the same bytes (the pin is interpreter-global, so not
  thread-safe against another thread's `csv` use); an undecodable byte is named by its offset in the
  file and its csv line (its position was 3 off after a BOM, and past the first ~8 KiB came as
  pandas' parse error, wrong past 256 KiB). Where `csv.reader` and pandas' C tokenizer can read the
  same bytes differently — a lone `\r` line ending, a NUL byte, a line led by spaces and then a
  quote, a byte-order mark past the file's start — the parsed frame is cross-checked cell by cell
  against the scanned records and a disagreement refuses as `unreadable` naming the line; each of
  these loaded at exit 0 (a NUL truncated a cell, `1\x002` to 1.0 and a `clo\x00se` header to `clo`;
  a delimiter-led record after a `\r`-ended blank line minted a row; a space-led quote or a BOM at
  pandas' 256 KiB read-buffer edge read a clean row, or slipped a duplicate column past the
  duplicate-name check). Every well-formed `\n` / `\r\n` file whose fields stay within 131072
  characters — blank lines, quoted delimiters and line breaks, a leading BOM and right-aligned
  columns included — reads byte-identically.
- CHANGED every layer to read an out-of-range intermediate as no value. `x / inf` is a FINITE 0.0,
  so sanitizing a result by finiteness let an overflowed divisor through: `rolling_corr(x, x)` read
  0.0 at 1e80 scale (its variance product overflowed), a `zscore` (`sma`) read 0.0 off an overflowed
  window variance, and the pools read `t_hac` 0.0 beside an infinite SE, mass shares of 0.0 off an
  overflowed |return| total (which the `concentration` ceiling PASSED), `rot_p` at its floor for an
  infinite observed mean, `cvar_5` as the whole pool's mean (+2.33e307 for `[−1e308, 8e307, 9e307]`)
  and a one-sided bootstrap interval. Below the normal range the opposite loss: a variance of
  deviations under ~1.5e-154 keeps few or no significant bits, and ordinary data reaches it — an
  `ema` of a change decays geometrically through it over a held stretch — so a varying window read
  the 0.0 an all-equal window owns, `rolling_corr` read NaN (1e-85 scale) or a value off in its
  third digit, the EW z-score NaN at `alpha ≥ 0.5` or a collapsed value below it, and a pool HAC's t
  off by 0.7, or a ZeroDivisionError out of the run. Now an overflowed variance, sum, mean or
  quantile is NaN → null wherever it is divided by or read — in the kernels it lands where an `x /
  0` would, undecidable once the series has initialized and its warmup before (a `zscore` whose
  every window overflows, levels near 1e160, never initializes: it never fires and no ledger counts
  it); a VARYING window whose variance falls below the normal range is recomputed on its values ×
  2^600 (exact, in the same summation order), so `std` reads its exact value floored at the smallest
  subnormal (`std > 0` holds on every varying window) and the SMA `zscore` its exact z;
  `rolling_corr` takes the product of two normal variances exactly (exponent extraction,
  `nb._sqrt_product`), `corr(x, x)` exactly 1.0 from 1e150 down to subnormal values; the EW
  `zscore`'s West recurrence runs in a frame lifted by 2^600 once its variance nears the range,
  reading the recurrence's value in an unbounded exponent range at every alpha (only a held input's
  decay whose exact |z| is already under 2^-158/α is floored at the smallest subnormal, sign kept; a
  hold whose mean lands ON the held value from the bar the input moved carries nothing, its exact
  deviation unknown, and reads z 0.0 as 4.0.0 did — undefined once its variance rounds to 0.0 at α ≥
  0.5, ~490 held bars at α 0.75 off a few-ulp step at unit levels, a decided 0.0 below α 0.5: a
  documented residual); and the pools null a VARYING pool's variance that is not a normal double, on
  either side (a constant pool keeps its exact 0.0). Moves, in the kernels: `rolling_corr`, `zscore`
  (both means) and `rolling_agg` / `cross_agg` `std` wherever a variance (or, for `rolling_corr`,
  their product) leaves the normal range, with firings, `signal_coverage` and every read downstream
  — on a feed that moves 100 bars and holds 1,900, `zscore(e, 20)` of `e = ema(change(feed), 5)`
  read 982 undecidable bars and now 63 (the windows the decay has settled into), and the documented
  guard `and(rolling_agg(e, 20, std) > 0, zscore(e, 2, mean_type: ema) …)` read 16 and now 0. In the
  pools, each read is null where its operand left the range (returns reach it only off input data at
  that scale): `t_hac` / `hac_se`, `std_ret` (pool and baseline), `boot` (`ci_lo` / `ci_hi` null
  together with `boot_se` when an episode sum or a draw overflowed — it read one finite bound beside
  a null one — and `boot_se` alone when the draws' variance is not a normal double), `skewness` /
  `kurtosis`, `win_loss_ratio` / `profit_factor` / `edge_ratio` / `tail_ratio`, `cvar_5`, the mass
  shares (`concentration.top_share_abs`, the `episode_stats` shares, `mass_hhi` /
  `effective_n_clusters`, the `episodes` `share_abs` / `omitted_share_abs`, `pooled.member_share` —
  the concentration check now refuses where it passed), `rot_p` (an infinite observed mean, or a
  defined shift whose rotation overflowed, which it dropped like an undefined one: 1/7 over the
  survivors where exact arithmetic gives 2/9 — beside a null `rot_p`, `rot_n_null` now counts every
  defined shift, where it counted only those the overflow spared), `bucket_monotonicity` (no entry),
  the `episode_profile` reads over an overflowed episode mean, and the CSCV split Sharpe (NaN,
  moving `pbo` and its diagnostics). ADDED the `boot` `reason` VALUE `not_representable` for both of
  its nulls (they carried `reason` null), and a constant pool whose mean's sum overflows reads the
  value itself (`ci_lo == ci_hi`, `boot_se` 0.0) where it read a null interval with no reason. An
  `episodes` entry's or a conditional bucket's `mean_ret` whose finite returns sum past the range is
  null, the two fields now nullable: such a mean failed the summary's own contract (exit 4;
  `ReportContractError` out of `api.compile_thesis`) on input the strict loader admits — a 1e-300 ↔
  1e8 OHLCV quote, a `diff` outcome on a series near ±1.7e308. `conditional_buckets` reads each
  row's bucket by qcut's bin index, never its interval label: pandas renders an edge through
  x·10^digits, so an edge past ~1.8e305 read `inf` and colliding labels merged the bins they named
  (a 1e-300 ↔ 1e8 OHLCV target's default `ret_5`: `(-1.001, inf]` 71 / `(inf, inf]` 71 over true
  bins of 71 / 35 / 36) and one under ~1e-306 in magnitude refused the feature as
  `insufficient_distinct_values`, and a declared feature spanning more than the double range (a feed
  alternating ±1.7e308), whose interpolated edges overflowed, is now bucketed exactly off its halved
  values. A one-target panel keeps pandas' interval labels wherever every edge prints distinct and
  finite, and reads the ordinals `q1..qk` (the multi-target labels) otherwise; a panel such an edge
  touched reads its true bins, `bucket_monotonicity` with them (on one target it only gains buckets;
  across targets it can also newly refuse `insufficient_distinct_values` where the members' true bin
  counts differ). The pool layer decides every such read by finiteness without a numpy
  RuntimeWarning: a default run printed them, and `python -W error -m seikan run` at overflow scale
  exited 4 (`internal`), where it now exits 0. ADDED the `benchmark_regression` `reason` VALUE
  `not_representable`: the fit declines where a varying bench leg's variance is not a normal double
  or `beta` / `alpha` overflow (it read a `beta` of 0.0 off an overflowed bench variance — `alpha`
  the raw leg's mean, `reason` null — an infinite `beta`, `no_benchmark_variation` for a varying
  bench whose variance vanished, and a reasonless NaN fit at 1e170 scale), its `r2` takes the
  variance product exponent-safely, and `r2` alone is null off a raw variance that is not a normal
  double. `describe`'s `volume.windows[N].mean` is overflow-safe (it read null beside `last_to_mean`
  0.0; a constant 1.7e308 volume now reads its value, `last_to_mean` 1.0), and every `dispersion[N]`
  std — `diff`, `pct` and `log` by one rule — is scale-safe, squared deviations past the range
  rescaled by a power of two and a varying series' std floored at the smallest subnormal: it read a
  reasonless null wherever the squared deviations overflowed (levels 1e-150, 1e5, 1e5 at window 2
  now read a `pct` std of 7.07e154) and a `diff` std of 0.0 on a varying series at 1e-170 scale; a
  std with no double to be refuses by name (the log-ratio entry below). Every window whose variances
  (and, for `rolling_corr`, their product) are normal doubles, and every EW z-score whose plain
  recurrence stays in that range, keeps every bit.
- CHANGED the log ratio to be scale-safe, one rule everywhere (`seikan.numeric`). `ln(b/a)` of two
  positive finite levels is always a representable double (|ln| < 1455), but the quotient it was
  read through need not be: 1e200 → 1e-200 is a quotient of 1e-400, which underflows to 0.0, so
  `seikan describe` crashed (a math domain error, exit 4), the runner's `log` outcome read −inf and
  censored a measurable window as `no_outcome` (which `outcome_coverage` then refused), and
  `change(kind: "log")` read an undecidable bar; a subnormal quotient read a finite but wrong log
  (3e20 → 1e-300: −737.92536 for the exact −737.92584). The log is now `ln(b) − ln(a)` wherever the
  quotient is not a finite positive normal double, and the direct `ln(b/a)`, bit for bit,
  everywhere else — in `change(kind: "log")`, the outcome measurement and describe's `changes` /
  `dispersion` log reads. Moves, only where a quotient leaves the normal range (|ln| beyond ~708,
  which ordinary price levels never reach): `log`-outcome `ret` / `ret_raw`, `pre_ret`, `mae` /
  `mfe` and `bars_to_positive` close with the exact log where they were `no_outcome` / NaN (an
  OHLCV target whose intrabar extreme leaves the range moves its `mae` / `mfe` alone); a `market`
  leg past the range closes as `horizon` instead of `no_benchmark`; under `cross_mean` a member's
  log leg past the range no longer censors its whole bar (that member `no_outcome`, every other
  `no_benchmark`) — every member closes as `horizon`, the bar mean carrying that leg's exact log in
  its 1/k share; `change(kind: "log")` values move from NaN to finite, with thresholds, firings,
  `signal_coverage` and the default `vol_14` feature (its trades column and conditioning reads, a
  `pct`-outcome run's included); and with the rows, every statistic of the cells that hold them,
  their `outcome_coverage` counts, the baseline and its exclusions, `pbo` and those cells' gate
  verdicts (`outcome_coverage` can turn met, the sample-dependent checks either way). A subnormal
  quotient's log moves from one finite value to the exact one, exit reasons unchanged. `describe`
  exits 0 on such files with the exact log (1e200 → 1e-200: −921.034…, `pct` −1.0) and refuses by
  name every read with no double to be: a `pct` past the range — a change's ratio, or a 1-bar ratio
  a `dispersion` std reads — is null with the new `ratio_reason` VALUE `not_representable`
  (`changes`, `dispersion`, `range_position`, `full_sample` — a reasonless null before), and a
  `diff` with none (finite levels of opposite sign more than ~1.8e308 apart, which crashed
  `range_position` out of the emission validator with exit 4, or a dispersion std past the range)
  refuses its whole block with the new block `reason` VALUE `not_representable` (`full_sample`
  keeping its extremes); the describe emission seam now refuses a null `diff`, or a null ratio read
  without its `ratio_reason`, on a block that did not refuse whole (exit 4, a seikan bug). A `pct`
  or `diff` outcome past the range still censors as `no_outcome`, now without a RuntimeWarning.
  Everything else is byte-identical.
- CHANGED the DSL's parameter domain to its executable domain (`constants.MAX_BAR_COUNT_PARAM` =
  2^31 − 1, `constants.MIN_EW_ALPHA` = 2^-30). The schema admitted parameters no kernel can run: an
  EW `alpha` of 1e-19 (only `0 < alpha ≤ 1` was checked) implied a 2e19-bar warmup no machine
  integer holds and died in numba, and `alpha: 5e-324` an infinite one (OverflowError) — exit 4,
  the internal-bug class, on caller input; the numba kernels' bar counts (an `ema` / EW `zscore` /
  `bars_since_extremum` `window`, a `first_true.cooldown`) crashed the same way at 10^20, from 2^63
  to 2^64 hung on which numba specialization the process had compiled first (`ema`
  `window: [2^63, 20]` ran, `[20, 2^63]` crashed) and ran below that; `rolling.window` / `min_count`
  crashed in pandas from 2^63; every other bar count (the `rolling_agg`, `percentile`, `drawdown`,
  `runup`, SMA `zscore` and `rolling_corr` windows, `shift` / `change` / `lag` `periods`) ran at any
  size, 10^20 included; and a past-ceiling `horizon` exited 2 on a measuring run and 0 on a flags-
  or root-series-only one, at any size. A past-ceiling parameter that ran left its node all warmup —
  zero firings where it gated and the real firings where it did not (a `features` snapshot, an `or`
  sibling) — and a `cooldown` of 10^12 fired its first edge. Every bar-count parameter — a transform
  `window` / `periods`, `rolling.window` / `min_count`, `first_true.cooldown`, `lag.periods`,
  `params.horizon`, and every shared-axis value one of them reads (a cross node's `min_valid`
  shares the type; its own floor binds first) — is now at most 2^31 − 1 and the EW `alpha` at least
  2^-30 (exactly the window form at `window = 2^31 − 1`), so a thesis past them refuses at parse
  (exit 3, `dsl_invalid`) wherever the value sits, and `seikan schema`'s DSL JSON Schema carries
  `maximum: 2147483647` on every bar-count field and `minimum: 9.313225746154785e-10` on `alpha`
  (was `exclusiveMinimum: 0`). Within the ceiling a parameter longer than the sample keeps its
  ordinary outcome (all warmup; a cooldown fires its first edge and holds every later one; a
  horizon the index cannot close is insufficient data, exit 2, on a measuring run), so a stored
  "fire once" `cooldown: 10**12` restates as `cooldown: 2147483647`. Every thesis inside the domain
  gives byte-identical reports, trades, flags and root series under the same `dsl_hash`; no
  statistic moves. `nb.ema_params` now raises a `ValueError` naming the domain for a span past the
  ceiling or not finite and for a window that is no bar count (below 1, fractional), where it
  returned a pair the kernel crashed on, ran all-warmup or silently truncated (window 2.5 → alpha
  2/3.5); a whole window of 1 or more returns exactly the same pair. A documentation correction
  rides along: the two EW forms' warmups agree bit-exactly only for windows below 5,935,198 bars
  (past that the alpha form can warm one bar later, same values).
- CHANGED a declined CSCV `pbo` block to carry its real candidate population. The ledger
  `n_combos <= n_combos_scoreable <= n_combos_declared` says how far short of the declared grid the
  scored population fell, but the declined template was built before the two lower counts were
  known, and two declined paths returned it as is — a sample under 4 bars, and a 4–7-bar sample no
  block partition fits: a thin grid of two distinct candidates read 0 / 0 / 2, a ledger claiming no
  candidate ever existed, while an 8-bar sample declining in the split loop read 2 / 2 / 2.
  Admissibility and the duplicate collapse read the cells alone, never the sample length, so they
  now run first and seed every declined block (`n_combos` is the DISTINCT candidates entering
  CSCV). Moves, only on runs under 8 bars with a non-empty CSCV population: `summary.pbo.n_combos`
  / `n_combos_scoreable` (under 4 bars any run with a fired-and-closed candidate — an unswept
  thesis 0 / 0 / 1 → 1 / 1 / 1, a clone grid 0 / 0 / 2 → 1 / 2 / 2; at 4–7 bars a grid of two or
  more distinct candidates, 0 / 0 / 2 → 2 / 2 / 2); `reason`, `pbo` and every other field are
  unchanged on every input. The emission seam (`seikan.emitted`, on the `run` document and on
  `validate_summary` / `api.compile_thesis`) now checks the block's cross-field invariants no
  TypedDict can state — `pbo` null exactly when `reason` is set,
  `0 <= n_combos <= n_combos_scoreable <= n_combos_declared == n_hypotheses_attempted`,
  `0 <= n_splits <= n_splits_attempted`, `0 <= n_candidates_min <= n_combos` — and refuses a
  violation as a seikan bug (`ReportContractError`, exit 4; no report written).
- Library: `vectorize.fully_initialized`, `vectorize.cross_reduced`,
  `analysis.stats.cross_mean_pooled_mean`, `analysis.stats.episode_bootstrap_ci` / `episode_ledger`
  / `episode_profile` (an optional keyword-only `rets_pool`: row-aligned values the episode sums
  add, default the rows' `ret`; in the bootstrap they break only the ties left in its canonical
  (bar, ret) order, so row order stays irrelevant), `sources.LeafMasks` (`source_availability` now
  returns `{label: LeafMasks}`), `MarketData.full_memo` (cleared by `clear_memo()`),
  `nb.mask_apply_nb(value, init, defined, full)`, and the public walks
  `dsl.schema.iter_source_leaves` / `series_source_leaves`, which also yield `("native", feed)` for
  a `native` node beside the `("external", feed)` leaf under its `expr`; the leaf module
  `seikan.numeric` (`log_ratio`, which takes numpy scalars without a warning, and its elementwise
  twin `log_ratio_array`, which always returns an array — the one scale-safe log-ratio rule,
  importing nothing from the package; and `MIN_NORMAL`, the smallest normal double, defined there
  once for every layer); `constants.MAX_BAR_COUNT_PARAM` / `MIN_EW_ALPHA`; `nb.ema_params` raising
  `ValueError` off its domain (above); `analysis.stats.newey_west_mean` returning `(NaN, NaN)` where
  the mean's variance is not a normal double; `analysis.stats.cscv_pbo` / `cell_conditional_buckets`
  taking an optional keyword-only `full_bars` (the `cross_mean` full-bar masks their split and
  bucket means read: per combo for CSCV, basket only; one bar-axis mask read through `entry_bar` for
  the buckets); `seikan.types.BlockReason` / `RatioReason` (the describe refusal vocabularies the
  `ChangeBlock` / `RangePositionBlock` / `FullSampleBlock` fields read), `DataIssueCode` with
  `field_count_mismatch` (its `IssueExample.csv_line` the line the record starts on, `value` its
  field count), `BenchmarkRegressionReason` and `EpisodeBootstrapCI.reason` with
  `not_representable`, and `EpisodeLedgerEntry` / `BucketRecord` `mean_ret` as `float | None`; and
  `validate_summary` refusing an impossible `pbo` block (its message now names
  `seikan.types/seikan.emitted`, as the document seam's did).

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
