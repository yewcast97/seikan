## seikan

A **stateless CLI forward-return event-study engine and per-hypothesis reporter**, consumed by
other agents (e.g. an AI coding agent): thesis DSL JSON + strict CSV time series in → the outputs
the caller NOMINATES, one flag each, written as FILES. It backtests a thesis as an **observer-pure
forward-return event study** (entry condition + horizon; no exit, no portfolio simulation),
measures EVERY declared parameter × horizon cell over the full sample, applies ONE uniform
checklist to EACH CELL INDEPENDENTLY, and reports each cell's result beside its complete
statistics in ONE complete JSON report; the per-bar series the entry tree thresholds on ride their
own CSV, so a caller can see why a bar did or did not fire, and the per-bar firing mask itself rides
another. A thesis declares one of two **target modes** — `conjunction` (the default: the targets
are the thesis's regime, measured side by side with the weakest deciding) or `basket` (the targets
form ONE cross-section per bar: cross-sectional signal nodes and the `cross_mean` benchmark become
legal, and each cell is graded on a pooled cross-target panel instead of per-member floors). No
database, no home directory, no network, no persistence — a run is fully described by
its inputs, and two identical runs emit byte-identical reports. Observer purity here is
COMPUTATIONAL and CONDITIONAL on point-in-time inputs: the engine enforces the as-of convention
it declares (feed timestamps are availability times; `lag` models publication delay), and cannot
verify that a file's timestamps are availability times, true vintages, or a survivorship-free
universe — that honesty arrives with the data and stays the caller's.

**Seikan does not select, rank, or crown a winner.** There is no best cell, no headline scalar, no
verdict, and no search-adjusted statistic. Every declared hypothesis — including combos that never
fired — is measured and reported with its own nominal numbers. Choosing among the cells, and
pricing the multiplicity of having looked at `n_hypotheses_attempted` of them (across runs, DSLs
and data windows, which a single stateless run cannot see at all), is the CALLING AGENT's work.
The exit code is NOT a verdict: it reports how far the RUN got.

### the CLI contract

```
seikan run <thesis.json>         full-grid event study + the per-cell checklist, written to
    --data KEY=PATH ...          the files the caller NOMINATES. The thesis NAMES its series;
    [--column KEY=COL ...]       this invocation LOCATES and SHAPES them. At least ONE output
                                 flag is required (none → exit 3 `usage`, refused before the
                                 thesis file is read); each path is ALWAYS overwritten. SILENT
                                 on success: exit 0 prints NOTHING on stdout — the outputs are
                                 the files, and only an error ever puts JSON there.
  --data KEY=PATH                LOCATES one declared series: KEY is a target name, an external
                                 feed name (`FEED@TARGET` for one member of a per-target feed),
                                 or the reserved `benchmark`; repeat once per key. The set must
                                 answer the thesis's declared keys EXACTLY — a missing, unknown
                                 or duplicated key is exit 3 `usage`, decided before a byte of
                                 data is read.
  --column KEY=COL               SHAPES it: WHICH column of that key's CSV the key reads, over
                                 the SAME flat key namespace, matched case-insensitively.
                                 OPTIONAL per key — only a file holding several numeric columns
                                 needs one — and REFUSED for `benchmark`, which always reads
                                 its file's `open`. The members of one per-target feed may bind
                                 DIFFERENT columns.
  --report-out <path>            the complete statistical report; no time series ride it.
  --trades-out <path>            one row per observation — firing × target × horizon —
                                 with the post-entry path columns (mae/mfe, the RAW-path
                                 adverse and favorable excursions, plus the timing pair).
  --root-series-out <path>       per-bar VALUES of the root series nodes the entry tree
                                 thresholds on — WHY a bar did or did not fire. NO 0/1
                                 flags ride it: whether a bar fired is --trades-out's
                                 business (in observation shape) and --entry-flags-out's.
  --entry-flags-out <path>       the per-bar 0/1 firing matrix, ONE integer column per
                                 param combo × target, bit-identical to the backtest's
                                 mask. The ONLY output that can carry a FINAL-BAR firing,
                                 which anchors no observation and so has no trades row.
  --min-trades N --min-n-nonoverlap N   the four checklist knobs, stricter-only (canonical-as-
  --max-concentration X          floor). Each flag is its GateThresholds knob minus the
  --max-hypotheses N             `thesis_` prefix, snake_case → --kebab-case
                                 (thesis_min_trades → --min-trades); env twin:
                                 SEIKAN_<KNOB> (SEIKAN_THESIS_MIN_TRADES). See the
                                 naming RULE under the sealed-policy paragraph below.
seikan hash <thesis.json|->      validate the thesis DSL and emit its canonical identity on
                                 stdout: `{name, dsl_hash, data_keys}` after the standard
                                 header. `dsl_hash` is `canonical_dsl_hash` (defaults
                                 filled, `null`-valued optional fields dropped, keys
                                 sorted — the same identity a run report stamps), and
                                 `data_keys` is the EXACT key set a `run`'s
                                 --data must answer, `benchmark` included when
                                 `params.benchmark == "market"` — so a host can assemble
                                 the whole invocation from this one document, no Python
                                 required. `-` reads the thesis from stdin. Touches no
                                 data file; an invalid document is the ordinary exit 3
                                 `dsl_invalid` envelope.
seikan check-data <files...>     pre-flight CSVs against the strict data contract. Exit 0/2.
                                 `--shape ohlcv|series` demands a shape instead of detecting
                                 it; `--pretty` indents the document (every subcommand takes
                                 it, `run` applying it to the report file and the envelopes).
seikan describe <files...>       pure data profiling for market-context prose: per-file levels,
                                 changes (all three algebras, domain-nulled), dispersion, range
                                 position, whole-file extremes, missingness, volume — bounded
                                 output, no per-bar arrays, and it MEASURES NOTHING (no entry
                                 condition, no forward observation, no checklist; describing is
                                 not evidence). Files profiled in argument order; `--windows`
                                 is trailing BAR counts (default 1,5,21,63,126,252, at most 16);
                                 `--shape` and `--pretty` as for check-data.
                                 One JSON document on stdout; exit 0 all admitted / 2 any
                                 refused (document still emitted, check-data parity) / 3 usage.
seikan schema [--markdown]       DSL JSON Schema + thresholds + the checklist contract + the
                                 report-field dictionary (`report_fields`) + the describe-side
                                 dictionaries (`describe_report` / `describe_roles`) + the input
                                 CSV contract and the three output-CSV contracts (`trades_csv` /
                                 `root_series_csv` / `entry_flags_csv`) + exit codes.
```

A thesis NAMES its series; the invocation LOCATES and SHAPES them. `data.targets` and
`data.external` declare logical keys and nothing else — no path, no column header — and the two
binding flags answer those keys over ONE flat namespace: a target name, a shared feed name,
`<feed>@<target>` for one member of a per-target feed, and the reserved `benchmark` when
`params.benchmark == "market"` asks for an excess-return source. `--data` says which CSV answers a
key; `--column` says which of that CSV's columns the key reads. Both are properties of whatever
file happens to answer the key — one vendor ships three yields in a single CSV under its own
spellings, another ships each in a file of its own — and neither is a property of the QUESTION: a
path or a header inside the document would let a re-pulled file or a renamed header move a
`dsl_hash` that nothing about the thesis had moved, and re-measuring the same thesis next month
would mean rewriting it. Neither lives in the DSL; what a feed entry fixes is the SEMANTIC read,
which is the only part a thesis is entitled to fix (`per_target`, `lag`). A key's NAME is opaque
to the engine the same way: `NVDA` is a label, not a ticker. What a name denotes — an instrument,
a tenor, a spread — and whether the bound bytes really are that thing is the CALLER's knowledge,
never the engine's.

The two flags are CHECKED differently, because they answer differently-shaped questions. The path
mapping must answer the declared keys EXACTLY: a missing key would load nothing where a series was
declared, so the run would measure a thesis it does not have; an unknown key is a caller holding a
different thesis in their head; a duplicated one would make the run choose which file it read, and
choosing silently answers a question nobody asked. All three are exit 3, decided before a byte of
data is touched. A column binding is OPTIONAL, because most keys are answered by a file that names
its own single value column and only a multi-column file needs one; what is refused ON SIGHT is a
binding that could not mean anything — a key this thesis never declares, and `benchmark`, which is
outcome MEASUREMENT sampled at the observation's own anchor bars and always reads its file's
`open`, leaving nothing there for a column to select. Whether the bound NAME answers these
particular bytes is a question only the bytes can answer, so it refuses one tier down instead
(exit 2, `spec_data_mismatch`): a binding aimed at an OHLCV target, which measures its
open-anchored prices and had no column to choose in the first place; a name no column carries, the
refusal listing the ones the file does hold; or a multi-column file with no binding at all, told to
bind one and which names are available. The same code refuses a feed whose declared `lag` moves
its stamps past the datetime range its clock represents — the thesis's lag and the file's stamps
cannot be combined (pandas' overflow there escaped the load as exit 4). COL is lowercased at parse
and matched case-insensitively
— the strict reader lowercases every header it admits — so the ONE spelling that reaches the loader
is also the one the report stamps. Binding per KEY is also WIDER than one document field could be:
the members of a single per-target feed may bind DIFFERENT columns, which a single DSL field could
not express. `check-data` and `describe` take no column flag and are not
missing one: they profile FILES, not bindings — a series file is profiled column by column —
because choosing the column a MEASUREMENT reads is the invocation's act, and it belongs to the
subcommand that measures.

Success is SILENT because the outputs ARE the files: a report on stdout would make every caller
buffer and re-parse a document it was going to write down anyway, and would force a result and an
error envelope to share one channel. A `run` puts an error envelope on stdout or nothing at all,
so a caller branches on the exit code and then reads the paths it itself
named — the silence belongs to `run`, whose outputs are files, and NOT to `hash`,
`check-data`, `describe` or `schema`, which still emit their document there at exit 0. A
run that nominates NO output is refused BEFORE the thesis file is read — it can produce nothing
whatever the thesis says, so the refusal costs nothing and its `usage` type never depends on data
seikan has not looked at yet. The same refusal covers every nomination that cannot be honored AS
GIVEN: an empty path, two flags naming ONE file (the writes are sequential, so the later would
destroy the earlier while the report went on enumerating both), and an output aimed at one of the
RUN's own inputs — the thesis document itself, or any CSV a `--data` pair pointed a key at
(resolved, so a different spelling of the same file does not slip through: by spelling first —
`..`, symlinks — and then by the filesystem itself, which reports one device and inode for two
paths that are one file, so neither `PRICES.csv` for `prices.csv` on a case-insensitive volume
nor a hard link gets past it).
That last one is the loader's refuse-never-mutate rule reaching the write side: an engine that will
not clamp an OHLC violation does not get to overwrite the evidence it is measuring and then stamp a
`data_digests` sha256 for bytes it deleted. Every path argument — the thesis, each `--data`
path, the four outputs, and the files `hash`, `check-data` and `describe` read — takes a leading
`~` as the caller's home, expanded ONCE as the command line is parsed, so the loader, these
checks, the writes and the report's `data_digests` / `outputs` paths all read the one file it
names (`./~` is a directory literally named `~`); expanded by the checks alone, an input spelled
`~/prices.csv` or `./~/prices.csv` was read from one file while the checks protected another, and
a report nominated at the file read overwrote it. Compute is likewise LAZY, matched to what was
nominated: `--report-out` or `--trades-out` runs the full-grid event study, while a run asking ONLY for the
listing outputs (`--root-series-out` and/or `--entry-flags-out`) takes the cheaper listing path — no
backtest, and no requirement that the index be long enough to close one observation at the longest
horizon. Both listings are DECISION-side artifacts no forward window ever touches, so demanding the
horizon runway for them would refuse a legitimate request on the strength of a measurement nobody
asked for. When both kinds are nominated the backtest runs FIRST, so every data refusal surfaces
from the strict path, whose checks are a superset of the listing's — and the market data is loaded
exactly ONCE per run, after request validation, with the one materialized `MarketData` shared by
both compute paths. That `MarketData` carries the run's one evaluation memo (`md.cache`), so every
series and condition is built at most once per run, whichever path asks first — a listing after a
backtest rebuilds nothing.

Tiered exit codes describe how far the RUN got, never how the evidence looked: **0** run
completed — every nominated output was written, and the report, when one was nominated, is
complete (per-cell results live in `gate.cells`) · **2** input data
failed strict validation (`data_report` says exactly what and where — including a column binding
these particular bytes cannot honor) · **3** invalid request — an argparse usage error (INCLUDING a
run that nominates no output at all, a `--data` mapping that does not answer the thesis's declared
keys exactly, and a `--column` bound to a key the thesis never declares or to `benchmark`), an
invalid thesis DSL or gate-threshold set, or an unusable nominated output path — empty, unwritable,
named by two flags at once, or naming one of the run's own inputs — the last group checked before
the O(grid × length) work rather than after it (`usage` / `dsl_invalid` / `thresholds_invalid`
envelopes) · **4** internal error. Every seikan-DETECTED error prints a machine-readable JSON
envelope on stdout (`--help`/`--version` are conventional non-JSON exit 0, and a SIGINT is Python's
default with no envelope); humans get one line on stderr. Nothing a checklist reports can move the
exit code.

The DSL holds neither locating fact — not a path, not a column name. `data.targets` is a list of
NAMES, a feed declares
`per_target` and `lag` rather than a file, and there is no benchmark path — the run's binding is
stamped into `identity.data_digests` as `key → {path, column, sha256}` instead, which is where a
locating fact belongs. A thesis that names a path or a column anyway is exit 3 `dsl_invalid` under
`extra="forbid"`. The refusal is the point rather than the cost: keeping those facts out of the
hashed document is the whole reason they are not in it.

Every JSON document — the `run` report, `hash`, `check-data`, `describe`, `schema`, and
every error
envelope — opens
with the same top-level header `{seikan_version, report_schema_version, command}`
(`report_schema_version` **5** — v5 REMOVED the legacy `stats_table`/`by_target`/`by_param`/
`n_stats_rows` grid breakdown (a fired-pools-only duplicate of `cells[].by_target` whose rollups
averaged per-cell means across horizons), the nominal `t_iid`/`p_iid` pair and the derivable
`sharpe`/`firing_rate`, renamed the trades column `bars_held` → `horizon` (always present), and
ADDED the integrity reads `rot_n_null` per panel and `pbo.n_splits_attempted`/`n_candidates_min`;
every surviving number is byte-identical to v4); an error envelope then carries
`error: {type, message, errors?}` with `type` ∈ {usage, data_invalid, dsl_invalid,
thresholds_invalid, internal}. The `run` report —
the file `--report-out` nominates, never stdout — has ONE FIXED layer order after the header, no
variants, so a reader never has to discover which keys this particular invocation happened to
produce: `identity` (`name`, `dsl_hash`, `data_digests` — one entry per data key the thesis
declares, `{path, column, sha256}` — the `thresholds` snapshot actually used,
`thresholds_canonical` (True iff every knob equals its class default) and
`thresholds_provenance` (per knob: `default`|`env`|`cli` — the SOURCE, so an auditor tells a
stricter-via-flag run from a stricter-via-env one), and `environment` — the numeric stack
(python/numpy/pandas/scipy/numba versions): floating-point results are a property of
(inputs, code, libraries), and a report reproducible only against an unstated stack is not
fully reconstructible) → `data_report` → `outputs` → `summary`
(verbatim) → `gate: {policy_version, n_cells, n_met, run_checks, cells}` → compact `metric_roles`
(the exact claim + which fields each check consumes + what is evidence-only + a machine-readable
`metric_classes` map — EVERY reported metric tagged `descriptive` | `inference` |
`integrity`, a fixed three-value vocabulary, so a consuming agent can tell STRUCTURALLY what kind
of number it is holding: descriptive describes the realized sample, inference is an uncalibrated
inferential estimator (rot_p, t_hac, boot, pbo), integrity is an accounting/reconciliation ledger
(counts, coverage panels, digests, the rotation null's resolution) — the class is orthogonal to
gating — plus a per-metric `caveats` map — one honest sentence per over-trustable number, 27
keys: rot_p, t_hac, boot, pbo, cross_breadth, mean_ret, mean_ret_raw, concentration,
ret_quantiles, mae_quantiles, mfe_quantiles, edge_ratio, profit_factor, baseline, episodes,
conditional_buckets, feature_association, pooled, member_share, win_loss_ratio,
std_ret, skewness, tail_ratio, cvar_5, benchmark_regression, episode_profile, timing —
travelling WITH the
report so the caveat rides beside the number it qualifies; the prose rationale lives in
`seikan schema` under `gate_contract.metric_roles_rationale`, and `metric_roles` itself is the SAME
compact object in the report and in `seikan schema`). `data_digests` is keyed by the LOGICAL key
and never by path: the key is what the thesis DECLARES and what two runs can be compared on, while
the path is only where this particular invocation found the bytes — so "the series this thesis
calls NVDA was that file, read out of that column, and it hashed to this" survives a re-pull as a
legible diff instead of a renamed row. Its `column` is ALWAYS present, and null when the key bound
none: a stamp that appears only sometimes reads as a different DOCUMENT rather than as a different
answer, and "nothing was bound, so the file named its own column" is a fact about the run worth
saying out loud. The sha256, being a property of the FILE's bytes, is shared by two keys that one
CSV happens to answer — each keeping its own column, which is precisely what the digest could not
say while it was keyed by path. The load reads each distinct path ONCE and every key bound to it
parses that one snapshot, so the shared digest names the bytes each of those keys measured (read
once per key, a file rewritten between the reads was two files under one path, and the digest named
the last). `outputs` is ONE block keyed in
nomination order — `report` → `trades` → `root_series` → `entry_flags` — holding only the outputs
this run actually nominated, each with its `path` plus, for each CSV, `rows_written`; a flat
`trades_rows_written`/`trades_path` pair could only ever describe one of four
files and say nothing about the three it did not name. It always contains `report`, since the
document carrying it exists only because `--report-out` asked for it — and the report is written
LAST, after every other nominated output has landed, so a document enumerating files that never
appeared is not a state this CLI can reach. There is no top-level `n`: a scalar observation count
would have to be some cell's, and naming one cell is the selection this engine does not perform.
`policy_version` (**4** — v4: a cell names its horizon, which bounds its non-overlapping
counts, no two cells name one assignment, and a basket's one-name read is the largest share of
its member decomposition; v3: a never-available source refuses, and the support floor reads the renamed
`n_nonoverlap`) names the checklist semantics (two cell results compare only
under the same version); `summary.statistics_version` (**5** — v2 ADDED evidence: the p05/p95
quantile tails, the promoted per-cell shape reads, `benchmark_regression`, `timing` and the
episode-deduplicated `episode_profile`; v3 CORRECTED: the excursion/trough window stops reading
the exit bar's high/low — full H/L over `[fill, fill+h−1]` plus only the exit print — CSCV
gained midrank ties, duplicate-candidate collapse and the declared/scoreable/distinct ledger,
`edge_ratio` pairs its rows, bucket edges are per target, and `episode_stats` added
`mass_hhi`/`effective_n_clusters`; v4 CORRECTED, numerically and in domain: the variance
family is computed CENTERED — zscore-SMA two-pass, zscore-EMA via the West recurrence,
cross_agg std two-pass — where the one-pass E[x²]−E[x]² forms cancelled catastrophically at
large input levels; every kernel and binary op enforces the finite-or-NaN invariant, so an
overflow from finite inputs is NaN instead of latching warmup or firing a threshold (undecidable
once its series has initialized, warmup before — see "Finite or NaN, everywhere" below);
`change(kind="pct")` and `drawdown`/`runup` are positive-domain (the ratio algebras inverted
silently on signed series — the outcome-side rule, now on the decision side too); the event-time HAC
uses the n−1 covariance divisor, making the sparse-pool reduction to the iid SE exact, with a
canonical (bar, ret) tie order; the episode panels sort stably with a target tie-break and the
episode bootstrap's content-keyed seed hashes the same canonical order, so tied-bar row order is
genuinely irrelevant; CSCV split scores run on SHIFTED moments (level-safe) and
`oos_degradation_slope` is a centered OLS that nulls with `oos_degradation_slope_reason` on
degenerate train scores; and zero-dispersion pools null their skewness/kurtosis instead of minting
unreliable moments; v5 CORRECTED six ways evidence could be manufactured: warmup has no truth value
and negation waits for its whole operand — `mask` with it, and a series computed from others
initializes only where its inputs are — zero dispersion is exact in every kernel,
recurrence and pool read that divides by it, a `native` feed's prints are their own `sources`
leaf, the rotation null judges ties within the FFT's rounding, the `cross_mean` pooled mean and
the bootstrap, era, episode and HAC reads beside it keep its zero-excess identities exactly, and
`cross_breadth`'s `k` counts the members a kernel actually reduced over; later review rounds
corrected more — among them correctly rounded CSV values, exactly rounded graded means and the
decision-side means beside them (`rolling_agg` / `cross_agg` `mean` and the cross-section
centre), exact per-episode means, each cross-section member's exact deviation (`cross_demean`,
the `cross_mean` excess), a HAC summed per bar about the pool's offset, exact `cross_mean` and
`event_agg` sums (every `cross_mean` bar-closed sum the bar's exact excess sum), window and pool
moments — the shape moments included — about their own offset, a `cvar_5` over the worst
⌈n/20⌉ observations, a signed log ratio (`ln(a/b)` is `−ln(b/a)` bit for bit), the regression's
alpha on the exact leg means and a fit of every varying raw leg, an EW z-score on a compensated
mean, correlations clipped to their range, warmup that follows the bars a lagged or snapshot
operand reads, cross-sections that enter only initialized inputs and labels, cross-target value
reads withheld where a `diff` outcome's level units differ, a rotation transform run in a unit
frame, an overflow-safe median, and clock arithmetic in exact ticks)
names the engine's estimator revision the same way.

The summary is a GRID, not a headline. Run-level keys are geometry and provenance only —
`n_bars`, `index_start`/`index_end`, `bar_spacing` ({min,median,max}_seconds between consecutive
bars — the clock geometry a horizon-in-bars is denominated in, exact in the index's own
resolution), `n_hypotheses_attempted`,
`direction`, `benchmark`/`benchmark_source`, `outcome` (ALWAYS the explicit
`{series, kind, units}` dict — never null; `units` derives from the algebra:
pct→`fraction`, log→`log`, diff→`level_diff`), `target_shape`, `target_mode` (ALWAYS
stamped — `conjunction` | `basket`, the target semantics every cross-target read was
produced under; the checklist dispatches its rubric on it), `statistics_version`,
`gate_evidence_basis`, `params`/`targets` (the swept entry axes and the regime — the two grid
labels), `rotation`, the per-target `sources` availability panel,
`cross_breadth` (the effective-universe ledger, one entry per cross node × combo summarizing the
per-bar count `k` of members the cross kernels reduce over — entering (a finite, initialized input,
and under the node's `where`/`group` selectors eligible with a finite, initialized label) in a
population that cleared `min_valid` and whose reduction produced a value (one that overflowed
counts nothing): the whole
cross-section, or each group on its own — always present, `[]` outside
basket; evidence-only), the
grid-level CSCV `pbo` block (ONE nested object: `{pbo, reason, n_splits, n_splits_attempted,
n_candidates_min, n_combos, n_combos_scoreable, n_combos_declared, blocks, lambda_mean,
oos_degradation_slope, oos_degradation_slope_reason, prob_oos_loss}` — the ledger `n_combos <=
n_combos_scoreable <= n_combos_declared` says how far short of the declared grid the scored
population fell (`n_combos` is the DISTINCT candidates entering CSCV, the population a scored
block ranks over; the ledger reads the cells alone, so a declined block, whatever its reason,
carries the real counts, never placeholder zeros), and `n_splits <= n_splits_attempted` with
`n_candidates_min` how far the block-local thinning departed from the fixed candidate count
canonical CSCV assumes — both ledgers re-checked at emission (every count non-negative, each
chain ordered, `n_combos_declared == n_hypotheses_attempted`) beside the verdict pairing (`pbo`
null exactly when `reason` is set), where a violation is a seikan bug (exit 4); its split
score is mode-aware — the weakest-target per-observation Sharpe under conjunction, under basket
the per-observation Sharpe of ONE column, the members' kept rows pooled (so candidates holding the
same returns in every block tie however the returns fall across members — scored off per-member
moments, two such candidates read `pbo` 0.343 for a full tie), whose mean reads each `cross_mean`
bar at its exact excess sum — see the pooled panel), and `baseline` (the run-level unconditional base rate per
horizon × target over every fillable anchor bar, same algebra/benchmark/direction as the cells,
with an `exclusions` ledger pinned by `n_eligible + Σexclusions == n_anchor_bars`, a `pooled`
row per horizon in basket, and deliberately NO uplift field — the conditional-vs-base-rate
comparison is the caller's). There is no rollup and no breakdown table: a mean of per-cell
means over different horizons is a number in no unit that moves with grid composition, so a
sweep axis is read off `cells[*].params` against `cells[*].by_target[t]` per horizon.
Conditioning is PER-CELL and never run-level: a pooled qcut would re-enter the same bar once per
combo × horizon, so adding a cell would change every other cell's conditioning. The
report's spine is
`summary.cells`: ONE entry per declared combo × horizon, in declaration order, INCLUDING those
that never fired (an explicit zero/NaN record), each carrying `cell_id`, `params` (the axes plus
the horizon, always present), `by_target` (n / n_nonoverlap / mean_ret / mean_ret_raw / mean_ret_bench —
the excess mean's own direction-signed legs over the same closed pool, so `mean_ret ≈
mean_ret_raw − mean_ret_bench` and a reader can tell "+3.2% against a +2.4% market" from "lost
less than a falling market"; unbenchmarked runs: `mean_ret_raw` equals `mean_ret` and
`mean_ret_bench` is null — `benchmark_regression` — {n, beta, alpha, r2, reason}, the legs'
per-window OLS attribution: is the excess mean alpha, or beta ≠ 1 riding market drift; `alpha +
beta·mean_ret_bench == mean_ret_raw` to the rounding of one product and one difference (alpha
reads the exactly rounded leg means), alpha per h-bar window and never annualized, null
fields + reason when unbenchmarked/thin/degenerate/out of range — hit_rate / win_loss_ratio /
profit_factor (Σwins/|Σlosses|, null never infinity when a side is empty) / std_ret / skewness /
kurtosis (Pearson, normal = 3) / tail_ratio (≡ |p95/p05|, derivable) / cvar_5 (the mean of the
worst ⌈n/20⌉ observations, the expected shortfall beside `ret_quantiles.p05`) — the
`pool_moments` shape reads over the pool's own closed rows — t_hac / hac_se / rot_p /
rot_n_null (the defined-shift count the cell's null was formed over; its own p floor is
`1/(1 + rot_n_null)`) / concentration, plus the
evidence blocks `boot` — the per-target
episode-bootstrap percentile CI — and `subperiods` — the pool's n / mean over three equal-bar
eras — and `ret_quantiles` {p05,p10,p25,p50,p75,p90,p95} + `worst_ret` / `best_ret` over
the closed returns, and `mae_quantiles` / `mfe_quantiles` — the same seven points plus the
subset's mean and worst/best over the RAW post-entry excursions, each carrying its OWN n, which
can sit below the cell's n because an excursion-window hole censors the path column on a row
whose `ret` closed — and `edge_ratio`, mean RAW MFE over |mean RAW MAE|, the one sanctioned
excursion ratio, and `timing` — {n_to_positive, median_bars_to_positive, n_to_trough,
median_bars_to_trough}, the timing pair aggregated the way the excursion pair is: survivors-only
medians censored at h),
`episode_stats`, the evidence blocks — `episodes` (the time-ordered,
mass-conserving episode ledger, capped at 32 with explicit truncation and `n_total ==
episode_stats.n_clusters`), `episode_profile` (the episode-deduplicated TWIN of the row-level
pool statistics: the SAME frozen cross-target merge as `episode_stats`, one aggregate per
episode — the mean of its rows' `ret` (a `cross_mean` bar's rows at their exact excess sum, 0 on
a full bar, see `pooled`), the ledger's own per-episode read, plus the extreme
`mae`/`mfe` within it — then the same statistic family over the episodes: n_episodes / hit_rate
/ mean_ret / profit_factor / ret_quantiles / worst_ret / best_ret / mae_quantiles /
mfe_quantiles / edge_ratio / max_win_streak / max_loss_streak, the streak pair's one honest
home since over overlapping ROWS "consecutive" is a cluster artifact; `n_episodes ==
episode_stats.n_clusters` by construction, it emits ALWAYS — descriptive statistics do not
refuse on thinness — and row-vs-episode divergence is the visible cluster diagnostic, reported
never corrected), `conditional_buckets`/`bucket_monotonicity` (per-cell, pooled across
the cell's own targets with explicit refusal reasons) and `feature_association` (per-target
Spearman, no p-value) — where a `diff` outcome's rows span several targets, each in its own level
units, every value these blocks pool ACROSS targets is null (the episode means, |return|-mass
shares and excursion extremes, an ordinal bucket's `mean_ret` and with it the monotonicity read;
the counts, the timing and every per-target panel stand) — plus, on BASKET cells only (absent, not null, on conjunction cells),
`pooled`: the cell's one cross-target evidence pool over the concatenated (bar × member) closed
rows (n / n_nonoverlap / mean_ret / mean_ret_raw / mean_ret_bench / benchmark_regression / hit_rate /
win_loss_ratio / profit_factor / std_ret / skewness / kurtosis / tail_ratio / cvar_5 / t_hac /
hac_se / rot_p / rot_n_null — the one pair NOT read over the closed rows: the COMMON-SHIFT null
rotates every FIRED member's complete mask as one block, closed rows or not, so a member whose
every firing was censored, or whose only firing is the final bar, still moves it (see
`gate_contract.metric_roles_rationale.caveats.rot_p`) — concentration / `member_share`
— the full per-member mass decomposition plus `max_member_share_abs` — boot / subperiods /
ret_quantiles / worst_ret / best_ret / mae_quantiles / mfe_quantiles / edge_ratio / timing —
under
`cross_mean` the pooled `mean_ret` is the FIRING subset's cross-sectional selection tilt — the
full cross-section demeans to zero each bar, the firing subset need not — and EXACTLY: the
pooled mean is the subset's exact excess sum over `n`, summed from the raw legs (an
integer-weighted sum of the legs, `N·[closed] − k_t` per leg, computed exactly), never rounding
residue whose sign the targets' declaration order would set, so it is 0 exactly where the
selected excesses cancel — always on a full bar (`S − 1.0·S`) and where every member's leg is the
same value (a shared or duplicated outcome feed: every per-row excess is then exactly 0, and
every read of it takes its exact value — means, quantiles and std 0.0, `rot_p` 1.0, `t_hac` /
`hac_se`, the shape reads, `concentration` and `member_share` null); with distinct legs each
per-row excess is that row's own rounded value, and the pooled reads that SUM whole bars read every
bar at its EXACT excess sum instead — its numerator `N·Σe` (the same weighted terms) carried
exactly, as the few doubles that sum to it, on the bar's first closed row and zeros on the rest
(`cross_mean_bar_numerators`), each such read summing its bars' numerators in ONE exactly
rounded sum over `N` times its row count: 0 on a full bar, and 0 wherever the excesses it holds
cancel, across bars too (from the trades exactly on a full bar — a bar on which every member has
a closed row in the cell, at 0 — and to rounding elsewhere, a partial bar's sum needing the legs
of members with no row), so the bootstrap's episode sums (same draws, same seed), the
`subperiods` means and the `episodes` ledger and `episode_profile` means (with the profile's hit
rate, streaks and quantiles over them) never read the rows' residue — three of six members firing
on bars whose selected excesses cancel read a positive bootstrap interval and positive eras
beside a `mean_ret` of exactly 0, and one member firing per bar, three bars to an episode whose
excesses cancel, read every episode at 9.9e-17 (each bar's sum rounded over `N` before the
episode added it: a hit rate of 1, a win streak of 8), where each now reads 0; the HAC reads the same
sums: its long-run variance reads the rows only through per-bar deviation sums (same-bar pairs
at weight 1), and under `cross_mean` each bar's is its closed members' EXACT excess sum around the
exact `mean_ret`, on every pool — so `t_hac` is `mean_ret` over `hac_se` on every pool, and a
pool whose every bar's excess sum is exactly 0 (every bar full at a zero mean, or a firing subset
whose excesses cancel on each bar, where the rounded rows' bar sums are residue — a `t_hac` of
5.43 beside a `mean_ret` of exactly 0) has no long-run variance: `t_hac` / `hac_se` null, as on
identical legs; the grid-level CSCV reads them too — a block holds whole bars (its purge reads
the bar and the cell's horizon alone), so every split's MEAN is the exactly rounded sum of its
rows' bar numerators over `N` times its row count, over the rows' own sd, and every split a grid of all-full cells scores is a full tie (`pbo`,
`lambda_mean` and `prob_oos_loss` 0.0 whenever the block scores, in any target order, where the
rounded rows ranked their residue — every train score tied, the first-declared candidate wins
each split, so a sparse such grid can score fewer splits, fall back to a coarser S or decline
`insufficient_data`); so do the cell's `conditional_buckets` means over a bar whose rows all land
in ONE bucket (bucketed by a basket-wide feature — a `calendar` field, a shared `external` feed —
an all-full pool reads every bucket mean 0.0 and no `bucket_monotonicity`, in any declaration
order), while a bar the per-target edges split across buckets, or partly unbucketed, keeps each
row's own excess there (no bucket holds its whole sum); `rot_p` counts
the identity's ties only where the demeaning's residue (~ε·|raw|) sits inside its FFT tie band
(a basket-wide ENTRY on ordinary dispersed legs reads 1 — every member fires on the same bars,
so every rotated pool is full too — while legs equal but for their last bits, one instrument in
two quote units, still read a coin flip), and row-level reads (`std_ret` and the shape reads,
quantiles, the row hit rate, `profit_factor`, |ret| mass shares, a bucket's `n` and
`hit_rate`) keep each row's ordinary rounding, its excess being its own — and the LEGS still
attribute: `mean_ret_bench` ≈
`mean_ret_raw`
is the basket's own realized drift), the panel the
basket rubric grades
while `by_target` stays as attribution — `episode_profile`, being the cross-target merge
already, doubles as the pooled episode read — and `outcome_coverage` and `signal_coverage`.
`len(summary.cells) == n_hypotheses_attempted` holds by construction — the panel is driven off the
DECLARED grid recorded before anything about firing is known, never off the trades frame, because
a non-firing hypothesis is precisely the one whose absence makes a surviving cell look inevitable.
`gate.cells[i]` is index-aligned with `summary.cells[i]`.

**The checklist contract (policy v4 — ONE checklist, applied to every cell independently)**:
three run-level checks reported once in `run_checks`, five per-cell checks in every
`cells[i].checks`, each `{name, met, observed, threshold, detail}`. A cell's
`met` is the conjunction of its own five checks AND all three run-level checks — an unmet
run-level check leaves every cell unmet, so a caller reading `cells[i].met` never has to AND the
sections itself — and `n_met` counts on that definition. There is no pass/fail vocabulary
anywhere in the section: `met` states that a neutral threshold comparison held, and the
judgement stays the caller's. ONE stamp selects between two rubrics: the
summary's `target_mode` names the target semantics every cross-target read is graded under —
conjunction (per-target floors and ceilings, the weakest target decides) or basket (the members
form one evidence pool: `support` reads the pooled floors,
`concentration` the pooled ceilings, `cell_evidence` reconciles the pooled panel against the
member panels). A missing or unreadable stamp refuses fail-closed everywhere it would have
dispatched — grading under an assumed mode is the stamp-stripping bypass one field over.

What a per-cell `met` MEANS, exactly: the cell's evidence is completely measured (every firing
accounted for, every decision bar decidable, every raw decision input available), it clears the
raw support floors on FULL-SAMPLE evidence — per target under conjunction, on the pooled panel
under basket — and its return mass is not one episode (nor, in basket, one member). **It is NOT
an inferential claim.** There is no significance claim and no positive-expected-return
certification; `mean_ret > 0` inside `support` is a sign read on the realized sample, not a test.
Nothing in the checklist is a test at all — there is no holdout, no embargo, no tail and no sign
test, so there is nothing to shop and nothing to reserve, and equally no out-of-sample
confirmation anyone may claim. Exit 0 certifies only that the run finished and every nominated
output was written.

The three RUN-LEVEL checks: `evidence_complete` (the summary must carry the evidence this
checklist grades, under the contract it was built against — `statistics_version` == current,
`gate_evidence_basis == "full_sample"`, a target list that is a non-empty list of STRINGS (a
non-string name indexes no panel and would crash `sorted`), an EXPLICIT `outcome` stamp — the
`{series, kind}` dict the runner always writes, naming the algebra every reported
number is denominated in; a null stamp is stripped input and refuses — a
readable `target_mode` stamp in {conjunction, basket}, since the stamp SELECTS the rubric (a
missing or garbage stamp refuses fail-closed, and a basket stamp over fewer than two targets or
a `diff` outcome refuses as drifted input: DSL validation refuses both upstream, and the gate
re-refuses rather than trusts) —
countable `n_bars` and
`n_hypotheses_attempted` ≥ 1, a string-keyed `sources` panel covering the targets EXACTLY, and a
`cells` list holding EXACTLY `n_hypotheses_attempted` entries, no two naming the same `params` —
a report short of the declared grid has dropped hypotheses from the search burden it declares,
and one holding a cell twice stands one in for a missing hypothesis at the right count (every
check met on the review's copy), which is drifted input, not evidence); `source_coverage` (the fail-closed availability contract over the RAW decision inputs,
run-level because it is combo-independent: per target `sources.n_missing == 0` — every leaf the
entry tree reads (`Field`/`External`/`DaysSince`, plus `native:<feed>` for a feed a `Native` node
reads on its own clock — every print, so a NaN print absorbed between two bars is a hole the
asof-anchored `external` leaf never sees, and one stamped after the feed's first finite print
but before the first bar is a hole on it, since a native read consumes the whole print history —
counted whatever the expression's reach, as every leaf is counted whichever parameters read it)
available on every bar from its own
`first_available` on — with `sources.n_bars == summary.n_bars`, every per-source count in `0..n_bars`,
and the union no larger than the sum of parts; a source that merely STARTS LATE is warmup, not a
hole, and `first_available` is evidence, never a refusal; no knob); and `search_cap`
(`n_hypotheses_attempted` — the DECLARED grid, which non-firing combos cannot shrink — ≤ 64, the
ONLY multiplicity input this policy carries).

The five PER-CELL checks: `cell_evidence` (the entry is a dict with a dict `params` naming a
countable `horizon ≥ 1`; `by_target`,
`outcome_coverage` and `signal_coverage` are string-keyed and cover the targets EXACTLY — a
silently dropped target leaves this check unmet rather than escaping notice by absence; every
count is countable and
non-negative; the ledger arithmetic holds per target (`sum(exit_reasons) == n_attempted`,
`n_closed == exit_reasons["horizon"]`); and the panels RECONCILE — `by_target[t].n ==
outcome_coverage[t].n_closed`, `n_nonoverlap ≤ n`, `n_nonoverlap ≤ max(0, ⌊(n_bars − 2) /
horizon⌋)` (a firing fills at the next bar's open and closes `horizon` bars later: 40 windows of
21 bars claimed in a 600-bar index, where 28 fit, is no run's output), `episode_stats.n` == the per-target total,
`signal_coverage[t].n_bars == summary.n_bars`: an internally impossible cell is drifted input, not
something to grade. Basket cells must ADDITIONALLY carry the `pooled` dict their own rubric
grades, reconciling with the member panels: `pooled.n ==` the per-target total (one pool, fully
attributed), `pooled.n_nonoverlap ≤ pooled.n`, `pooled.n_nonoverlap ≤ max(0, ⌊(n_bars − 2) /
horizon⌋)` (same-bar cross-member firings collapse in the greedy count, so the bar clock's windows
bound it), `pooled.n ≤ n_bars ×
len(targets)` (each member fires at most once per bar), and a `member_share.max_member_share_abs`
that is a share is the largest of its `by_target` shares, string-keyed over the targets exactly
(0.5 beside shares `{AAA: 1.0, BBB: 0.0}` met every check); a `pooled` key on a conjunction cell
REFUSES — the runner writes pooled only in basket mode (absent, not null, on conjunction
cells), so the configuration is the signature of a RESTAMPED basket and refusing it costs zero
honest refusals (`evidence_complete` re-refuses `benchmark='cross_mean'` under a conjunction
stamp by the same logic); a missing `target_mode` stamp refuses — whether a pooled panel is
part of the cell's contract is then undeterminable); `outcome_coverage` (fail-closed missingness on the
OUTCOME side: per target
`no_outcome == 0` and `no_benchmark == 0` — a data hole that deletes outcomes can hide adverse
results, and missing-at-random is never assumed. **`open` is ALLOWED at any count**: with no
holdout there is no embargo and no tail, so a forward window running past the last bar is
structural end-of-data right-censoring every cell near the index end must exhibit, and refusing it
would refuse the calendar rather than a data defect. A window that closes within the data and
reads a NaN leg is never `open` — it classifies as `no_outcome`/`no_benchmark` upstream and
refuses here; one running past the last bar is `open` even on a NaN fill print, as no outcome of
it was ever measurable for a hole to delete); `signal_coverage` (the
fail-closed DECISION-side twin, pooled layer: per target `n_undefined == 0` and `n_undefined ≤
n_bars`. The outcome ledger can only account for bars that FIRED, so a missing decision input
suppresses the firing itself and leaves no trace there — without this check, deleting the inputs
under adverse firings would improve a cell unseen. The raw inputs underneath are graded once,
run-level, by `source_coverage`, which catches the two hole classes this channel structurally
cannot see: absorption — every completion of a hole decides the node the same way, so the root
reads DEFINED while an input is holed (Kleene `F∧U = F`; a `first_true` hole whose completions
re-converge before they disagree on any fire, e.g. one on a cooldown bar other than the
cooldown's last) — and a hole inside a windowed transform's warmup, which only delays its first
value (the bars between read as warmup). A transform carrying state across bars has no value
from a hole in its input on, so that hole reaches this ledger ("Warmup is no evidence"). A NaN
computed INSIDE the expression (`log` of a non-positive value, `x/0`) is no raw leaf's: absorbed,
it moves no decision; inside a warmup, it only delays the first value — ledgered nowhere either
way. No knob. BOTH coverage checks — and `source_coverage` — stay PER-TARGET in BOTH modes: a
hole in one basket member corrupts every member's cross-sectional reads, so per-member
fail-closed is structurally required even in a basket); `support` (the same
three sealed floors — `n ≥ 30`, `n_nonoverlap ≥ 8`, `mean_ret > 0` — read from the panel the
`target_mode` stamp selects: under conjunction per target, over the cell's OWN full-sample rows,
the weakest target deciding, since the targets are then the thesis's regime; under basket on
`pooled.{n, n_nonoverlap, mean_ret}` — the members form ONE evidence pool, floors read the pooled
panel and never per member, so a thin member does not sink a basket cell, because the basket
claim is about the pool, not about any name in it — deliberately NOT an inferential claim
either way, no t-statistic and no
p-value gates); and `concentration` (one universal ceiling, dispatched by the same stamp: under
conjunction over every regime target's
`concentration.top_share_abs` AND the cell's `episode_stats.max_cluster_share_abs` (the largest
|return|-mass share held by any one cluster — distinct from the `largest_cluster_n` /
`largest_cluster_share_abs` / `largest_cluster_start` triple, which describes the largest cluster
BY ROW COUNT and is evidence only), the largest
merged cross-target episode cluster — a one-episode edge refuses; under basket the pooled read
REPLACES the per-target layer (`pooled.concentration.top_share_abs`), the episode-cluster
ceiling stays ("not one crisis"), and `pooled.member_share.max_member_share_abs` joins them —
the one-name-basket detector ("not one name"), same sealed ceiling, no new knob. A
`diff`-outcome multi-target
run refuses the cross-target mass read as incommensurable, as does a MISSING, NULL, or unreadable
outcome stamp, since stripping it would otherwise bypass the guard and accepting a null is the
same bypass one spelling over — retained in basket as defense-in-depth even though
`evidence_complete` already refuses basket+diff).

Fail-closed, never crash: an unreadable `cells` panel leaves `evidence_complete` unmet and yields
`n_cells` 0 / `n_met` 0 / `gate.cells == []` — and the CLI still exits 0; a malformed
INDIVIDUAL cell entry gets an unmet `cell_evidence` with a detail while its siblings grade
normally; NaN / ±inf / non-integral reads refuse per read. Nothing short-circuits: every check is
always evaluated and always reported.

The uncalibrated statistics are EVIDENCE, never inputs: `rot_p` (the rotation null over-certifies
under signal-aligned volatility regimes), `t_hac`/`hac_se` (the overlap HAC understates its SE),
the per-cell `boot` episode-bootstrap CI (the dependence-robust counterweight to those two —
resamples overlap-connected episodes, so it is honest about clustered pools, but it assumes
episode independence and is itself uncalibrated), the `subperiods` era panel, `bar_spacing`,
the order-statistic blocks (`ret_quantiles`/`worst_ret`/`best_ret` — the median-against-mean
read that exposes a pool a few spikes carried, which `concentration` structurally cannot, since
mild right skew concentrates no mass in one episode — and `mae_quantiles`/`mfe_quantiles` with
their subset means, RAW
path and so NOT commensurable with a benchmarked excess `ret`, tied together by overlapping
windows sharing one trough and one peak, and in the MFE's case a MARK rather than an attainable
exit, this engine having no exit rule), `edge_ratio` (the one sanctioned ratio BETWEEN the two
RAW excursion pools — deliberately unnormalized, so it compares nothing across instruments), the
per-cell attribution legs `mean_ret_raw`/`mean_ret_bench` (the excess mean decomposed over its
own pool — attribution, never two extra hypotheses; unbenchmarked, `mean_ret_raw` equals
`mean_ret` and `mean_ret_bench` is null, and the
trades CSV carries the per-row `ret_raw`/`ret_bench` twins), `benchmark_regression` (the legs'
per-window OLS attribution — beta/alpha/r2 over paired rows, never annualized, null-with-reason
when unbenchmarked/thin/degenerate; overlap inflates the fit exactly as it inflates every
row-level read), `profit_factor` (derivable ≈
(n_wins/n_losses) × win_loss_ratio; null, never an infinity, when a side is empty), the CSCV
`pbo` block, the per-cell distribution-shape descriptives — `win_loss_ratio`, `std_ret`,
`skewness`/`kurtosis` (scipy's population moments, bias=True), `tail_ratio` (≡ |p95/p05|,
derivable), `cvar_5` — riding the cell panels (a per-observation Sharpe is `mean_ret/std_ret`
and a firing rate `outcome_coverage.n_attempted/n_bars`, both derivable and neither emitted),
the `timing` block (the
timing pair's survivors-only medians, censored at h), the episode panel beyond the
two fields the checklist reads, `baseline` (with its exclusions ledger
and the BASELINE cross_mean identity — the pooled CELL mean is the selection tilt, not that
identity), the per-cell `episodes` ledger, the per-cell
`episode_profile` (the episode-deduplicated twin — see below), the per-cell
`conditional_buckets`/`bucket_monotonicity` and `feature_association`, the pooled panel beyond
the fields the basket rubric reads (mean_ret_raw/mean_ret_bench, benchmark_regression,
hit_rate, win_loss_ratio, profit_factor, the shape reads, t_hac/hac_se, rot_p, boot,
subperiods, the quantile blocks with worst/best, edge_ratio, timing), and
`pooled.member_share.by_target` (the full member-mass decomposition —
attribution, never a ranking; only `max_member_share_abs` gates, and `cell_evidence` reads
`by_target` only to reconcile that maximum) — all ride in the summary and no check grades them — `metric_roles` says so explicitly, and
its `caveats` map states each one's failure mode in the report itself. Cross-target STATISTICS
exist ONLY in basket mode, as the declared, graded `pooled` block — conjunction forms none — and
NEITHER mode selects among members: nothing anywhere in the report ranks the targets or crowns
one. There is no
search-adjusted statistic anywhere: no `fw_p`, no winner's-curse deflation,
no Romano-Wolf, no cross-cell BH-FDR. A correction computed inside one run only ever sees that
run's grid and silently understates a caller who searched across runs; the multiplicity belongs
to the caller, priced against the stamped `n_hypotheses_attempted`.

**The cluster-twist map.** Overlapping observations mean one market episode enters every
row-level statistic up to ~h times (× members in basket) — one crisis can be the whole tail of
`mae_quantiles`, most of `hit_rate`, and `profit_factor`'s loss mass. The calibration for that
is never a reweighting (choosing the exchangeable unit is the caller's modeling judgment); it is
the twin-view discipline — every row-level read has a stated dependence-aware counterpart, and
the divergence between the two IS the diagnostic:

| row-level read | how one episode inflates it | the honest counterpart |
|---|---|---|
| `n`, `hit_rate`, `mean_ret`, `profit_factor`, `win_loss_ratio` | one move enters ~h× (× members) | `episode_profile`'s counts/hit/PF/mean; `n_nonoverlap` |
| `ret_quantiles`/`worst_ret`/`best_ret`, `skewness`/`kurtosis`, `tail_ratio`, `cvar_5`, `std_ret` | the "tail" is one smeared move | `episode_profile.ret_quantiles`/worst/best; the shape caveats (no episode moments — too few episodes to support them) |
| `mae_quantiles`/`mfe_quantiles`, `edge_ratio` | overlapping windows share one trough/peak | `episode_profile`'s per-episode extremes and its own `edge_ratio` |
| row-level streaks | consecutive rows ≈ one episode | never emitted — `episode_profile.max_win_streak`/`max_loss_streak` are the honest form |
| `feature_association`, `conditional_buckets`/`bucket_monotonicity` | ~h rows per episode share one outcome | caveat only: features VARY within an episode, so a per-episode aggregate would manufacture a snapshot nobody observed |
| `benchmark_regression`, `timing` | the same smearing on the fit / the medians | caveat only (a per-episode regression over few episodes is noise — stated, not computed) |
| `concentration`, `episode_stats`, `boot`, `pbo` | — | already the dependence-aware layer |

The policy is sealed as well as stamped: **the canonical checklist is the floor** — every knob
constructs only at its default or STRICTER, so a loosened exam is exit 3 (`thresholds_invalid`),
same as a nonsense value — and the `SEIKAN_*` env namespace is owned: an unknown/typo'd `SEIKAN_*`
var — any name that is not one of the four knobs, `SEIKAN_GATE_PROFILE` and
`SEIKAN_THESIS_OOS_ALPHA` included — is a hard error, never silently ignored. The knob NAMING is
a rule, not a coincidence: a knob's CLI flag is its `GateThresholds` field name minus the
`thesis_` prefix with snake_case turned into `--kebab-case` (`thesis_min_trades` →
`--min-trades`), and its env var is the full field name upper-cased under the `SEIKAN_` prefix
(`SEIKAN_THESIS_MIN_TRADES`) — so a host derives the whole flag surface from `seikan schema`'s
`thresholds` listing instead of reverse-engineering the CLI. Scope boundary (stamped into
`metric_roles`): the checklist prices ONE cell of ONE run. It takes no cross-cell correction, and
cross-run/cross-DSL search discipline is invisible to a stateless reporter — but the identity
layer (`dsl_hash`, the per-key `data_digests`, `summary.index_start`/`index_end`) makes every
distinct exam visible so the caller can enforce its own budget. Deployment judgment is likewise the caller's.

### non-negotiable invariants

- **The engine's statistical mechanics are frozen** (`compiler/` + `analysis/` — the EVENT-TIME
  HAC t (Bartlett on actual entry-bar distances, df=n_nonoverlap−1), the circular-shift rotation null,
  `n_nonoverlap`, CSCV PBO, episode clustering and |return|-mass concentration, the descriptive panels).
  Any change here changes what a report MEANS — two reports compare only under the same
  `statistics_version`. Repackage, never rewrite; a
  correctness fix must bump `analysis.stats.STATISTICS_VERSION` (+ `gate.POLICY_VERSION` when the
  checklist's meaning or its emitted shape moves), and a reader-visible change to any emitted
  document's shape or key vocabulary bumps `cli.REPORT_SCHEMA_VERSION` — never land silently.
  Those constants are where the numbers are
  defined. The known rotation/HAC calibration caveats are
  documented in `stats.py` and `metric_roles`. Note what the checklist deliberately does NOT have:
  a false-activation rate. It makes no inferential claim, so there is nothing to calibrate — a
  null run's cells can and sometimes will come out met, and `met` is not significance. Calibrating
  the evidence-only ESTIMATORS (`rot_p`, `t_hac`) under dependent/heteroskedastic nulls remains
  open statistical work.
- **The checklist never filters the summary.** The engine summary is embedded in the report
  VERBATIM — the complete parameter × horizon grid (`cells` with every declared hypothesis
  including non-firing ones, each with every per-target panel and reliability read) —
  whatever any cell's result is. The per-cell results are a separate `gate` section in which EVERY
  check is reported for EVERY cell (no short-circuit), and they cannot move the exit code. See
  `gate/` and the honesty invariant comment in `cli.py`.
- **Strict CSV is never relaxed.** ISO-8601 naive timestamps only (no format guessing — ambiguous
  dates are rejected, not interpreted; `data.start`/`data.end` obey the same strict parse, so a
  malformed bound is exit 3 rather than a silently reinterpreted window), at most nanosecond
  resolution (a nonzero fractional digit past the ninth refuses as `bad_timestamp` — the parser
  would truncate it into another instant, an availability moved onto an earlier bar — and as an
  exit-3 DSL refusal in a `data.start`/`data.end` bound), values
  under an explicit ASCII numeric grammar (sign, digits, fraction, exponent; `inf`/`infinity`,
  refused later as non-finite) read CORRECTLY ROUNDED — the double nearest the decimal written,
  so every repr a writer emits reads back bit for bit, where pandas' own parser missed by an ulp
  on nearly half of all 17-digit strings and read `0.00000000000000001` as 0 — and OHLC
  invariants refuse rather than clamp (the loader never mutates evidence). Every value column is
  named, uniquely: an empty header name refuses (`unnamed_column` — pandas invents `Unnamed: N`
  for it, and `datetime,,unnamed: 1` loaded two columns under one name and crashed the read, exit
  4), save the timestamp column's, which no stage reads; a repeated name refuses
  (`duplicate_column`), and pandas must read every name the strict scan does. Every record
  carries exactly one field per header column, checked by a `csv.reader` pass in pandas' own
  dialect before pandas reads the file: any record with more fields (a trailing delimiter is one
  more) or fewer refuses as `field_count_mismatch`, counting every offender and naming the first
  few csv lines with their counts — pandas would otherwise pad a short record with empty cells
  (a hole the file never had) and, when EVERY record runs one long, read the first field as an
  implicit index, so each header name reads the field one place to its right and the file loads
  clean and wrong. A quoted delimiter or line break stays one field; blank lines skip as they do
  for pandas. Where pandas' tokenizer itself can read a record differently from that pass (after
  a bare `\r` line ending, at a NUL byte, at a quote after a line's leading spaces, which its
  256 KiB read-buffer edge can open, at a byte-order mark past the file's start, which that edge
  can drop from an over-long header) its frame is cross-checked cell by cell, and a disagreement
  refuses as `unreadable` naming the csv line. **Multi-target
  runs require ONE bar clock**: unequal target indices refuse (exit 2, naming what is missing
  where) instead of being intersected — an intersection deletes a bar from EVERY target before any
  coverage ledger exists, so trimming rows under adverse firings would shrink the graded pool
  invisibly.
  Basket mode leans on that clock DOUBLY: a bar's cross-section — the thing `cross_rank` ranks
  and `cross_mean` averages — only exists because every member stamps the same bars, so the
  union-and-refuse rule is the basket's precondition, not merely its hygiene.
  Market-shaped anomalies (NaN holes, crash bars, gaps) warn but admit — crashes are the research
  subject. See `dataio.py`.
- **Refuse invalid input early and cheaply.** Non-finite numeric literals (`NaN`/`Infinity` —
  Python's JSON decoder accepts them, JSON proper has none) refuse at parse; a `params.features`
  key colliding with a trade column (`ret`, `is_open`, `target`, …) refuses at validation instead
  of silently overwriting the engine's own evidence or crashing a boolean index; and a DECLARED
  grid above the sealed search cap refuses at validation, BEFORE any data is read — such a grid can
  never meet `search_cap` under any legal thresholds, so the engine owes it neither a report nor
  the O(grid × length) work of producing one.
- **Hash discipline**: `canonical_dsl_hash` fills defaults, drops every `null`-valued optional
  field and sorts keys, so omitted-default, explicit-default and explicit-`null` DSLs share one
  identity. New node types and `None`-defaulted optional fields are hash-safe (they appear only
  when used); a non-`None` default still moves every stored hash, and callers keeping reports must
  re-validate rather than assume a stored hash still names the same thesis. `target_mode` remains
  the discipline's one DELIBERATE exception: a mode is a property of the whole thesis, not of any
  node, so it has to be a non-`None`-defaulted field, and that cost is paid openly instead of being
  smuggled in.
  REMOVING a field moves every stored hash the same way (the
  normalization fills defaults over whatever the model still declares). File paths and column names
  sit OUTSIDE the hashed document for exactly this reason: the same exam over re-pulled or
  re-shaped data is ONE document rather than a new one every time the data layer moves, with the
  locating facts stamped in `identity.data_digests` where they
  belong.
- **Statelessness**: no SQLite, no `$SEIKAN_HOME`, no config file. Thresholds come from `SEIKAN_*`
  env vars overridden by CLI flags, and the snapshot used is stamped into every report. The one
  on-disk artifact is numba's JIT cache (see dev commands — `NUMBA_CACHE_DIR` under a read-only
  install): a compilation cache, never run state, and its presence, absence or location changes
  no number in any report.
- **Warmup is no evidence.** A condition node that is still warming has no truth value
  (`~init ⇒ ~value`, beside `~init ⇒ defined`), so a warming branch can neither fire an `or`
  that a live sibling initialized nor absorb it over the sibling's hole. Negation (`not`, and the
  `mask`, its numeric spelling) is initialized only where EVERY condition under its operand is
  past warmup: `or` initializes on any branch, and its False while a branch warms is the
  doctrinal no-signal, not a decided False. Its True there IS decided (T∨U = T) and the `or`
  fires on it, but `mask` stays warmup (NaN) until every branch is live: a series of those Trues
  alone, the Falses still unknown, would hand every reader that skips NaN (an `ema`, an expanding
  aggregate, a cross-section's breadth) a value-selected history. So a history read over a mask
  begins where every branch is live, and a cross-section thins a member while its mask warms, as
  it thins a late-listed member (`cross_breadth` records the `k`); the condition-level reads
  (`rolling`, `first_true`, the event nodes) still see the live branch's firings. A Series
  computed from other Series initializes only at a finite value on a bar where every input is
  initialized (for a lagged read — `shift`, `change` — where the bar it reads back to was; for an
  event snapshot — `event_value`, `event_agg` — where its event bar was, or the current bar while
  a hole leaves that bar unknown, so a value an input carried during its own warmup never becomes
  usable without a new event), or
  where every input is initialized and one has produced no value since its own init (an event
  node whose anchor became unknowable before its first event, a mask undecided on its first live
  bar — a hole it then ledgers, as the input's direct readers do, never its own warmup; a value
  the input carried before its init is not its own and settles nothing): identical to its own
  finiteness latch for plain data, and what keeps a reader of a `cross_agg` broadcast into a
  member whose own input has not started from latching early (the broadcast itself waits for the
  member's own first finite value since its init — a value carried before the init, beside an
  init that arrived later, is neither). A transform that carries state across bars — `ema` below decay 1
  (at `alpha = 1` it is its input, hole for hole), `zscore` over an EW mean, and the expanding
  (`window`-less) `rolling_agg` / `drawdown` / `runup` / `bars_since_extremum` — has no value
  from its input's first post-init hole on: its kernel skips a NaN, and the state it carried
  across one (a peak the hidden bar may have set, a mean it would have moved) read as decided on
  every later bar, a domain NaN (`log` of a non-positive value) ledgered nowhere —
  `expanding_max(mask(log(z) > 0)) < 0.5` fired on every bar after a `z` of −1, as after a 1,
  where a 2 stops it. So it is NaN and initialized from there: undecidable to a
  threshold over it, a hole to a series computed from it (a reader still warming when it dies
  warms on), while a trailing window or a lag is decided again once it reads past the hole. A
  cross node enters a member only with an initialized input, and
  groups it only by an initialized label: a value either carries before its own init enters no
  cross-section, so a member whose input reads a broadcast its own data has not joined is neither
  ranked, demeaned nor averaged against its siblings (nor counted in their `cross_breadth` `k`),
  and a rank or deviation whose label has had no value since its init ledgers that stretch, like
  every reader of the label. A feature snapshot reads the same channel: a firing on a bar where
  the feature's series has not initialized snapshots NaN, never the value it carries there, and
  enters no association or bucket. Negation's full warmup follows the same bars: a condition embedded
  in a `shift`ed or `change`d operand must be past warmup on the bar the lag reads, one embedded
  in an event snapshot's input on its event bar (`not(shift(bars_since_event(or(A, B)), 20) >
  1000)` waits for A twenty bars longer, as `lag(not(…), 20)` does). Over thresholds on
  window-local operands,
  `and`/`or`/`not`/`mask`, `rolling` and `lag`, lengthening a hole-free warmup can therefore only
  remove firings (a path-dependent transform — `ema`, an expanding aggregate, a cross-section —
  reads a different history when its input starts later, which is no claim about warmup).
  Signal-level reads (`first_true`, the event anchors, a cross node's `where`) read the tradable
  signal itself, for which a warming branch is simply not firing: `bars_since_event(or(A, B))`
  counts B's events while A warms, where `not(rolling(any, or(A, B), N))` waits for A. A `not` /
  `mask` over a partially warm operand is warmup — it neither fires nor ledgers there (the raw
  leaves stay under `source_coverage`).
- **Finite or NaN, everywhere.** Every series value the engine produces is finite or NaN: kernels
  and binary ops sanitize results whose arithmetic can overflow from finite inputs, and
  definedness/warmup latching are minted from `isfinite` — so an overflow never latches warmup and
  can never fire a signal. Where it lands follows the warmup latch ("Warmup is no evidence"), never
  the overflow: on a bar after its series initialized the NaN is in the undecidable ledger
  (`signal_coverage`); before, the bar is the series' warmup exactly as a leading `x / 0` is, and no
  ledger counts it. A series initializes no earlier than its first finite value, so one that
  overflows on every bar (a `zscore` of levels near 1e160) is warmup throughout: its threshold never
  fires, and no coverage ledger counts it. The exception is a series the latch initializes before
  any value — an event node whose anchor became unknowable before its first event, a mask undecided
  on its first live bar, a stateful transform whose input holes before its first value, and a
  series computed from such an input: it initializes there, finite or not, and every overflowed bar
  after it is undecidable. A `cross_agg` is never one — its broadcast
  waits for its own first finite value as well as for the member's own first finite value since
  its init — so one that
  overflows on every bar is warmup throughout, whatever its input. The ratio transforms
  (`change(kind="pct")`, `drawdown`, `runup`) are POSITIVE-domain like the pct/log outcome
  algebras: off a positive scale they invert silently, so a non-positive base censors as NaN
  instead of minting a sign-flipped number. Signed series belong to `diff`. The LOG ratio is
  scale-safe, one rule for `change(kind="log")`, the `log` outcome (benchmark leg, `pre_ret`,
  excursions included) and `describe`'s `log` reads (`seikan.numeric`). It is SIGNED: `ln(hi/lo)`
  of the larger level over the smaller, negated where the level fell, so `ln(a/b)` is `−ln(b/a)`
  bit for bit and a move and its reverse cancel exactly (a cycle through three or more levels only
  to rounding, as any floating-point sum) — read in its own direction, each quotient rounded on
  its own, and 100 → 110 → 100 summed to 2e-17, a positive `mean_ret` that met the support floor.
  `ln(b/a)` of two positive finite levels is always representable even where the quotient is not,
  so it reads `ln hi − ln lo` exactly where that quotient overflowed (1e200 →
  1e-200 is −921.03…, never `ln(0)`), and the direct form — bit for bit — everywhere else. A
  `pct` (or `diff`) past the double range has no representable value: NaN (a `no_outcome`
  observation, a NaN decision operand as above); a `describe` `pct` past the range (for a dispersion
  std, any one 1-bar ratio past it) is null with `ratio_reason: "not_representable"`, and a
  `describe` `diff` past it — levels of opposite sign ~1.8e308 apart — refuses its whole block as
  `reason: "not_representable"` (`diff` rides every block that did not refuse, so it is never a
  reasonless null). A value that IS representable is never lost to an intermediate's overflow
  either: an even median's middle pair (`rolling_agg` / `cross_agg` `median`, and `mad`) meets at
  numpy's own `(lo + hi) / 2` wherever that is finite, else at the halves' sum — exact there — so
  levels past half the double range read their median (a constant window its value and a MAD of
  0.0), never the NaN their overflowed sum read beside a mean of 1e308. Nor is a cross-section's
  exact sum or mean: a row whose largest member reaches `2**996` is summed in a frame scaled by
  `2**-28`, chosen off the members' magnitudes, so `cross_agg` `mean`, the `cross_mean` bench leg
  and every exact deviation read the same in any member order (summed directly,
  `[1e308, 1e308, −1e308, −1e308]` read NaN in that order and 0 in another), and each quotient
  is divided back in the input frame wherever its numerator is representable there; a row that
  frame cannot hold exactly — a member below `2**-1046` beside one past `2**996` — is NaN,
  undecidable, never a value read off a rounded member (`cross_demean` read
  `[1e308, −1e308, 1e-320, 0]`'s last two deviations, 7.5e-321 and −2.5e-321, as 0.0). A
  window's running sum (`rolling_agg` `mean`) is NaN where it leaves the range on the way, in the
  order its bars arrived.
- **Zero dispersion is exact.** The floating-point mean of a constant DECIMAL series is not the
  constant, and its rounding residue (~1e-17) used to read as dispersion — a z-score of exactly
  ±1, a correlation of 1, a flat feed drifting off its own EMA, an EW z-score locked at
  `±1/√(1−α)` on a held stretch, `|t_hac|` ~ 1e16, a CSCV split Sharpe of ±1e7. Every read that
  can meet it gives an all-equal (bit-identical) input its exact answer instead. In the kernels: a
  window, cross-section or since-event run of equal values has its mean equal to the value, a std
  / `mad` / demeaned value of 0.0 and an undefined (NaN) z-score / correlation, and the EW
  recurrences never leave the interval their exact update lies in (a constant input stays exactly
  itself; a held input is approached strictly from its own side — one ulp short of the value at
  the closest, never on it, so `x > ema(x)` keeps one answer through a hold — and its EW z-score
  decays instead of locking, keeping one sign: its mean is compensated — carried as two doubles,
  each update taken with TwoSum — so a deviation at the level's ulp keeps its bits, a step whose
  update rounds the mean onto the new value keeps its gap, a hold onto the COMPUTED mean reads a
  decided 0.0, and the z never passes its bound `√((1−α)/α)` — a plain mean read `[1, 1,
  1 + 2**-52]` at α 0.5 as z 2, off a mean rounded back onto 1, where the exact z is 1). That
  answer and that sign are exact arithmetic's wherever the exact mean sits off the held value by
  more than the rounding of its steps. Within it they are the rounding's: `[0.1, 0.3,
  −0.09999999999999999]` then 0.05 held has, at window 3, an exact mean of exactly 0.05 from bar 2
  on — neither side of the hold — while the `ema` sits one ulp above it (bar 1's update rounded up
  at a tie), so `x < ema(x)` holds on all 157 held bars, and the z-score reads 5.9e-17, so
  `z > 0` does. Closing that needs the exact mean, a rational that grows with every step.
  In the statistics: a constant pool has `std_ret` 0.0, no bucket-monotonicity rank and a degenerate
  bootstrap (`boot_se` 0.0, `ci_lo == ci_hi` on the pool's own arithmetic mean — the value itself
  where that mean's sum overflowed — as for any pool whose resampled means are bit-identical on
  every draw), and the reads that divide by dispersion stand down where the spread is within
  rounding — the HAC nulls (largest deviation within ~10 ulps of the mean, the skewness/kurtosis
  gate, or every BAR's deviation sum exactly 0 about the exact mean: a pooled pool whose bars all
  sit at its mean, every row on one bar among them; skewness/kurtosis are otherwise taken about
  the pool's offset, translation-invariant — about scipy's own rounded mean the symmetric `1 + k·eps`,
  k = 0..21, read skewness 0.666 for 0), the benchmark regression declines a (near-)constant
  bench leg (and fits `beta` exactly 0 on a constant raw leg, while a raw leg that varies is
  fitted however small its spread — its deviations about its offset are exact), and a CSCV
  split scores NaN when its pool is exactly constant (the per-block extrema) or within the HAC's
  ~10 ulps. The pool MEANS the checklist grades, and their direct twins — the legs,
  the eras, the baseline, the HAC's `t` numerator — are the exactly rounded sum over `n` (`math.fsum`):
  the exact sum's sign, exactly 0.0 where it is 0 (a pairwise float sum lost a term a larger one
  swallowed before its cancelling partner arrived — an exact zero read as a positive mean), and a
  constant pool's own value; a `cross_mean` basket's pooled `mean_ret` is its exact excess sum
  from the raw legs, and the bar-closed sums beside it — the CSCV split means and the bucket
  means over whole bars among them — read each bar at its exact excess sum (see the pooled
  panel). So are the episode panels' per-episode means and `episode_profile.mean_ret` over them,
  whose signs the hit rate and the streaks count — summed pairwise, eight episodes of
  `[−0.5, −2**-55, 0.5, 2**-55]` read 6.9e-18 each, a hit rate of 1 and a win streak of 8 beside
  a `mean_ret` of 0 — and the conditional buckets' means, whose order `bucket_monotonicity`
  ranks: summed in floating point, four buckets holding `[−0.3, −0.2, −0.1, 0.1, 0.2, 0.3]` in
  four row orders read four rising means and a `rho` of 1. The bootstrap's draws stay the
  arithmetic mean, save that each episode's sum is exactly rounded too — summed in floating
  point, eight episodes of `[1, −2**54, −1, 2**54]` read a zero-width interval at 0.25 beside a
  `mean_ret` of 0; a draw's sum over its episodes' sums stays ordinary arithmetic. A CSCV
  split's mean is its rows' exactly rounded sum over their count, one `math.fsum` over its
  blocks' exact parts — rebuilt from the column's mean, an exact 0 read −2.6e-18 and two
  candidates cycling through ±0.3 read `pbo` 2/70 and `prob_oos_loss` 1/70 for 1/70 and 0 —
  and its block moments are exactly rounded sums (a column's returns about its exactly rounded
  mean, each block's deviations from the block's mean squared), the same doubles in any row
  order: two candidates holding the same returns in every block, in other orders, ranked apart
  by row-order rounding (`pbo` 0.571 for a grid of full ties). A split's variance merges its
  blocks' moments (`Σ m2_b + Σ n_b·(m_b − m)²`, every term a square, ordinary arithmetic). The HAC's per-bar deviation
  sums are exact about the pool's EXACT mean, each rounded once (the returns read as integers over
  one power-of-two denominator), so a bar at the mean reads 0 and a pool and its translate read
  one `hac_se`: rounded per row about the pool's range midpoint, three members alternating
  `[0.8, −0.4, −0.4]` and `[0, 0, 0]` beside a fourth at `2**-52` — every bar exactly at the
  mean — read a `t_hac` of 35.7, and about the reported mean returns at `1 + k·2**-52` read an SE
  30% high. The |return|-mass shares (`top_share_abs`, the episode shares, `member_share`)
  divide exactly rounded masses, so no
  share passes 1 — summed in floating point, five masses among 95 zeros read a top share of
  1.0000000000000002 — and `mass_hhi` / `effective_n_clusters` are the correctly rounded values
  of exact arithmetic over the cluster masses, the latter within `[1, n_clusters]`.
  The means a DECISION reads are exactly rounded the same way: `rolling_agg` and `cross_agg` `mean`,
  and the `cross_mean` bench leg (`nb.cross_means`, `cross_agg`'s own kernel), are the window's or
  the bar's exactly rounded sum over its count, in any bar or member order (summed pairwise, `[1, −2**-54, −1, 2**-54, 0]`
  read a mean of 1.1e-17 in one member order and 0 in another, so `cross_agg(x, mean) > 0` fired on
  the order the targets were declared in); a deviation from a cross-section's mean that is READ —
  `cross_demean` and the `cross_mean` excess — is the member's exact deviation, its numerator
  `k·x − S` summed exactly and rounded once over `k` (`nb.cross_deviations`), so a member AT the
  mean reads exactly 0.0 (about the rounded centre, `[0, 1, 1, −1 − 2**-52, −1 + 2**-52]`'s first
  member read 2.2e-17 and `x == 0 AND cross_demean(x) > 0` fired); a window's variance is still
  taken about its direct two-pass centre (below), so a z-score within the rounding of its window's
  spread of 0 can read 0 or the other sign.
  A post-warmup zero-dispersion bar
  is UNDECIDABLE like `x / 0`: its NaN lands in `signal_coverage`, never fires — guard a z-score
  of a series that can sit flat with `rolling_agg(x, N, std) > 0`. "Equal" means bit-identical: a
  spread built from decimal levels moving in lockstep carries residue at the operands' scale,
  which no kernel can tell from variation. Every window, cross-section and pool that varies is
  the direct two-pass form about its OWN offset — the window's first value, a cross-section's or a
  pool's range midpoint (`numeric.midrange`: the pools' `std_ret`, `boot_se` and baseline std, the
  benchmark regression's second moments, `describe`'s dispersion stds)
  — bit for bit, while its
  variances and, for `rolling_corr`, their product are normal doubles (the next bullet): centered
  on the rounded mean of the raw values, a spread at its level's ulp was lost to that mean's
  rounding (`[2**52, 2**52 + 1]` read a window std of 0.707 for 0.5, a two-point z-score of
  1.414, past its bound of 1, and a ddof=1 `std_ret` of 1.0 for 0.707; legs clustered 31 and 62
  ulps apart read a regression beta of 1.9984 for 2). The pool means stay the exactly rounded
  sum, and `alpha` stays `mean_raw − beta·mean_bench` on them. A `cross_mean` excess is the
  member's exact deviation, `cross_demean`'s (above): legs `[2**52, 2**52 + 1, 2**52 + 1]` read
  excesses `[0, 1, 1]` for `[−2/3, 1/3, 1/3]` about their rounded mean, two selected rows
  "beating the basket" by 1 beside an exact pooled mean of 1/3; its bench leg is the bar's exactly
  rounded mean (above), so `ret == ret_raw − ret_bench` holds to that rounding (exactly under
  `market`) — about the bar's offset, `[0, .5, .5, −.5 − 2**-53, −.5 + 2**-53]` read a bench leg
  of −1.1e-17 for an exact 0, and a member's regression on it beta 4 for 5 — and a bar of finite
  legs always has its mean (legs whose plain sum overflows are summed in a scaled frame). `ema`
  differs from its plain recurrence only on the steps where a rounded update left its exact
  interval — rounding-scale steps a varying series with repeated values (tick data) can take too;
  the EW z-score reads the plain West recurrence's value to its rounding (the compensated mean),
  lifted where that recurrence leaves the normal range (the next bullet).
- **Out-of-range dispersion is no dispersion, nor a lost one.** The finite-or-NaN line holds for
  INTERMEDIATES too, not only results: `x / inf` is a finite 0.0, so no check on a result can see a
  quotient whose divisor overflowed. A window variance that OVERFLOWED (deviations past ~1.3e154)
  read `rolling_corr(x, x)` as 0 at 1e80 and a z-score as 0: it is NaN wherever it is divided by or
  reported, and that NaN lands where any overflow's does ("Finite or NaN, everywhere"):
  undecidable on a bar after its series initialized, the series' warmup before — like a leading
  `x / 0`, a z-score of plain data whose every window overflows, its first included, never
  initializes, never fires and is counted by no ledger (`source_coverage` included: its raw
  leaves are all there). One that fell BELOW the normal range (deviations under
  ~1.5e-154) is no overflow but a precision loss, and ordinary data reaches it — an `ema` of a
  change decays geometrically through it over a held stretch: its squares lost their bits (a
  z-score past its bound `√(w−1)`, a correlation off in its third digit) or vanished into the `0.0`
  an all-equal window owns (the guard `std > 0` False on moving data). Such a VARYING window is
  recomputed on its values lifted by `2**600` — exact, scaling up by a power of two never rounds —
  in the direct form's own summation order, so its `zscore` (`mean_type: "sma"`), `rolling_corr`
  and `std` (`rolling_agg`, `cross_agg`) are the values the direct form has in an unbounded
  exponent range; a `std` under the smallest subnormal floors there, so `std > 0` holds on every
  varying window, exactly where the SMA z-score it guards is defined. The correlation's variance
  PRODUCT of two normal variances is taken in an unbounded exponent range too, so
  `rolling_corr(x, x)` is exactly 1 at 1e80, at 1e-85 and on subnormal values as at 1; and a
  correlation is clipped to `[−1, 1]`, the regression's `r2` to at most 1 — an exactly linear
  window's or pool's rounding read 1.0000000000000002 (`rolling_corr(x, 3x + 7, 3) > 1` fired on
  it). The EW
  `zscore` carries its variance in a recurrence, which runs in a frame lifted by `2**600` once that
  variance nears the range — exact, the same doctrine — so its z too is the recurrence's value in
  an unbounded exponent range: never NaN for want of range, nor collapsed toward 0 off a floored
  variance (which decided thresholds the exact z does not), nor past its bound `√((1−α)/α)`; past
  its warmup, save an overflow carried forward, it is defined on every window `std > 0` admits, a
  hold included however long. Out of the frame's reach is only a held input's decay, whose z exact
  arithmetic has sent under `2**-158/α` by then: its variance and deviation are floored at the
  smallest subnormal there, sign kept — decided, imprecise in magnitude. A plain mean left a
  residual on holds that the compensated one closes: a step of a few ulps whose update landed the
  mean ON the new value (the LANDING GAP — an `ema` converging onto a held price), or an input
  moved onto the mean, read z 0.0 off a deviation it could not know, and at α ≥ 0.5 turned
  undefined once that hold's variance rounded to 0.0 (~490 held bars after a two-ulp step at α
  0.75) — `signal_coverage` refused holds exact arithmetic decides. The pool statistics read
  measured returns, which reach that range only off input data at that scale, and hold the
  undecidable line on both sides: a share, ratio, `t_hac`, CSCV split Sharpe, episode-mean read,
  regression read or `cvar_5` read off a sum, mean, quantile or variance that overflowed, and a
  `std_ret`, `boot_se`, `t_hac`, skewness/kurtosis, split Sharpe or `r2`/beta read off a variance
  (or the power of it a moment divides by) outside the normal range, is null —
  `benchmark_regression` declines with reason `not_representable`, an overflowed |return| mass
  fails the concentration check instead of passing it on 0.0 shares, an episode's or a bucket's
  mean whose sum overflowed is null itself (the `episodes` ledger entry, the conditional bucket —
  which leaves `bucket_monotonicity` null), and an episode sum that overflowed nulls both `boot`
  bounds with `boot_se`, never one bound, with reason `not_representable` (the reason a
  `boot_se` nulled alone off the draws' variance carries too). A conditional bucket whose
  feature edge is past what an interval label prints (~1.8e305, or under ~1e-306 in magnitude)
  keeps its true bin under its ordinal `q1..qk`, never merged with a colliding label, and a
  feature whose values span more than the double range (its quantile interpolation overflowing
  across zero) is bucketed off its halved values — exact there, every such value lying past
  2**970 in magnitude — never cut on an infinite edge. None of these reads warns: an overflow
  there is an expected input, never a warning (an exception under `python -W error`). The
  rotation null meets neither edge: its transform runs on each forward column scaled by a power
  of two into a UNIT FRAME (largest magnitude in `[0.5, 1)`; a basket's members brought into
  their largest member's frame exactly), where nothing can leave the double range and the p,
  scale-free, is never scaled back — a column near the ceiling reads exact arithmetic's p where
  its overflowed transform read null, and a subnormal one its ties (every rotation of constant
  `1e-323` returns reads 1.0, where the transform's few-bit arithmetic read 0.05). Every column
  the raw transform carried reads bit-identically (the FFT is exactly equivariant under such a
  scaling).
  `describe` reports numbers it never divides by, so it computes them instead: its dispersion stds
  (`diff`, `pct` and `log` alike) and its trailing volume mean are scaled by a power of two where
  their squares or sum leave the range (a varying series never reads std 0.0, `last_to_mean` never
  `x / inf`, and finite levels never a null std while their 1-bar changes and the std itself are
  representable doubles). Every window and pool whose variances and, for `rolling_corr`, their
  product are normal doubles, and whose sums and means stay finite, keeps its direct form's bits
  (a window's or a pool's: the two-pass form about its offset), and every EW z-score whose
  recurrence stays in the normal range reads that recurrence's value to its rounding.
- **Strict, frozen DSL.** Types are strict (no str/bool/float coerces to an int, no str to a
  bool; int stays legal for a float field — JSON has one number type), instances are frozen
  (`model_copy(update=...)` is the derive path), a duplicate JSON object key at any depth
  refuses at parse (last-key-wins made the effective document parser-dependent), a string
  feed `lag` is read by seikan's own exact grammar — one or more `<number><unit>` terms (`"36h"`,
  `"1h30m"`, `"1.5 days"`), refused unless a whole number of nanoseconds, with clock, ISO-8601
  and bare-number forms refused — never pandas' parser, which read `"nan"` as NaT (a declared
  publication delay silently degraded to zero, the look-ahead direction), truncated past the
  nanosecond, read `"5"` as five nanoseconds and `"1,5h"` as fifteen hours, and was a nanosecond
  short on `"0.57 days"`, and the public API (`compile_thesis`/`list_entries`) revalidates a
  private copy at the boundary because freezing is shallow. The schema's domain is the
  EXECUTABLE domain: every bar-count parameter (windows, periods, `cooldown`, `min_count`,
  `horizon`, shared-axis values read by them — and `min_valid`, a member count sharing their
  integer type) is at most `MAX_BAR_COUNT_PARAM` = 2^31 − 1 and an EW `alpha` at least
  `MIN_EW_ALPHA` = 2^-30 (its warmup span is then within the same ceiling), so an ema
  `window: 10**20` or `alpha: 1e-19` refuses at parse (exit 3) instead of crashing a numba
  kernel (exit 4); `nb.ema_params` re-refuses an over-ceiling span, and a window that is no
  bar count (below 1, fractional), with a `ValueError` for library callers. A shared int axis
  value read by a float param (`constant.value`) must be one a double holds exactly: the param
  executes the double, so `2**63 − 1` would run `2**63`'s threshold under its own label, and
  refuses at parse instead.
- **One spelling per meaning (canonical identity).** `params.outcome` is non-optional with a
  default factory (omitted ≡ `{}` ≡ the spelled-out default — one identity; explicit `null`
  refuses); `params.features` refuses the empty dict; a scalar constant refuses a `name` (the
  name labels a sweep axis and exists only for a list value); and a declared external feed
  nothing references — entry tree, features, or `outcome.series` — refuses at validation, since
  it demanded a file and moved the identity without changing the measurement.
- **Ambiguous shape refuses.** A series read of a file carrying two or more of the four OHLC
  column names (without, or even with, the full set when a series read is demanded) refuses:
  it would be broadcast into all four price slots while every OHLC integrity invariant is
  skipped. A lone OHLC-named column (a bare `close`) is an ordinary series file.
- **Atomic outputs.** Every nominated output writes to a temp file in the target's own
  directory and lands via one `os.replace`, so a crash or ENOSPC mid-write can never leave a
  truncated file wearing the nominated name; a symlinked nomination is written through — the
  file it names takes the output (created where the link dangles) and the link stays; the
  report is still written LAST, after every other nominated output, and
  serialization/contract-validation still precede any file side effect.

### module map (`src/seikan/`)

- `cli.py` / `__main__.py` — argparse CLI, the five subcommands (`run`, `hash`, `check-data`,
  `describe`, `schema`),
  tiered exit codes, the caller-nominated outputs (`_preflight_outputs` refuses an unusable path,
  and two nominations — or a nomination and an input — the filesystem resolves to one file,
  before the run, not after it) and the JSON report assembly (the honesty invariant lives here: the
  summary is embedded before the checklist runs, and `_cmd_run` returns `EXIT_OK` unconditionally,
  so the only thing a checklist result moves is the content of `gate`).
- `describe.py` — pure data profiling behind `seikan describe`: plain numpy/pandas, no numba, no
  RNG, and nothing imported from `compiler/`, `analysis/` or `gate/` (the clock-geometry stamp it
  shares with the run summary is `dataio.bar_spacing`, which lives with the data layer, and the
  log-ratio rule and normal-range edge it shares with the engine live in the leaf `numeric.py`).
  Admission is the SAME strict read `check-data` performs; refusals get stub profiles, never
  guesses; bounded output — no per-bar array ever rides the document.
- `contract.py` — the static contract payloads, data only (imports nothing from the package):
  `METRIC_ROLES` (compact, with its `caveats` map, stamped into every report) and
  `METRIC_ROLES_DOC` / `GATE_CONTRACT_DOC` / `REPORT_FIELDS` (the report/summary field dictionary
  — the output-side twin of the DSL JSON schema) / `DESCRIBE_ROLES` / `DESCRIBE_REPORT` (the
  describe-side claim/caveats and field dictionary) / `CSV_FORMAT` / `TRADES_CSV` /
  `ROOT_SERIES_CSV` / `ENTRY_FLAGS_CSV` / `EXIT_CODES` — the prose `seikan schema` emits.
- `constants.py` — structural names and bounds shared across the layers without a cycle:
  `TRADE_COLUMNS`, the reserved sweep/feature namespaces, `DEFAULT_FEATURE_NAMES`,
  `MAX_DECLARED_GRID` (the sealed search cap's validation-side twin).
- `numeric.py` — scale-safe numeric rules shared across the layers, a leaf (`math` + `numpy`,
  no numba, nothing from the package, so `describe.py` can share them with `compiler/` and
  `analysis/`): `log_ratio` / `log_ratio_array`, the one log-ratio rule of the decision side, the
  outcome algebra and `describe`, `midrange`, the offset the pools' and `describe`'s dispersions
  are taken about, and `MIN_NORMAL`, the smallest normal double — the one edge of
  the normal range the kernels' variance tests, the pool statistics and `describe`'s std read.
- `types/` — the emitted-shape declarations (TypedDicts), one module per report section behind
  the package facade; `emitted.py` validates every success document against them at emission,
  so the bytes written are the bytes that type-checked, and also checks the cross-field
  invariants no TypedDict can state (the `summary.pbo` verdict pairing and ledgers; `describe`'s
  explicit refusals — no null `diff` on a block that did not refuse, a reason beside every null
  ratio read).
- `gate/` — a package behind one facade (`from seikan.gate import …` — the policy doctrine
  rides its `__init__` docstring): `_hash` holds `canonical_dsl_hash` (re-exported by `api.py`,
  the supported import site; the `seikan hash` subcommand is the same identity without Python),
  `_model` the policy version + result dataclasses, `_read` the strict readers over drifted
  input, `_checks_run`/`_checks_cell` the checks, `_evaluate` `evaluate_gate`: the per-cell
  checklist (3 run-level + 5
  per-cell checks, each `{name, met, observed, threshold, detail}`;
  `POLICY_VERSION` 4 stamped into every report; the thresholds handed in are REVALIDATED by
  reconstruction at entry, so a mutated or loosened object cannot bend the checklist at the library
  boundary). The policy contract: stamped evidence verification (`statistics_version` +
  `gate_evidence_basis == "full_sample"` + a readable `target_mode` stamp — the rubric selector,
  refusing fail-closed when missing — + an exactly-sized `cells` panel of distinct `params` + a string-keyed
  `sources` panel), fail-closed missingness on BOTH sides (outcomes via `outcome_coverage` —
  end-of-data `open` allowed, in-bounds holes refused — and decisions via the pooled
  `signal_coverage` plus the run-level `source_coverage`; per-target in BOTH modes), support
  floors and the
  universal concentration ceiling read from the mode's panel (per-target under conjunction,
  pooled — including the member-mass ceiling — under basket, with the pooled panel reconciled in
  `cell_evidence`), and the declared-grid `search_cap`. No inferential check exists
  at any level. `GateReport.to_dict()` emits `{policy_version, n_cells, n_met, run_checks,
  cells}` — no verdict key. Drifted summary input refuses with a detail, never crashes (an
  integer past the double range, which JSON admits and `float()` raises on, included).
- `settings.py` — `GateThresholds` (pydantic-settings, FROZEN, `SEIKAN_` env prefix; FOUR knobs:
  `thesis_min_trades`, `thesis_min_n_nonoverlap`, `thesis_max_concentration`, `thesis_max_hypotheses` —
  no optional knobs, no profile field, no alpha). Sealed canonical-as-floor: every knob's domain
  admits only its default or stricter (a loosened exam is exit 3), unknown `SEIKAN_*` env vars
  refuse (the known set is derived from `model_fields`, so a deleted knob's env var is enforced for
  free), `is_canonical()` feeds the report's `identity.thresholds_canonical` stamp.
- `dataio.py` — the strict-CSV front door: `read_strict_csv` (all violations accumulated, except
  past a structural refusal — undecodable bytes, a duplicate header name, a ragged record, a
  record the strict scan cannot read (a field past 131072 characters, usually a quote that never
  closes), a record pandas' tokenizer cannot read or reads differently from the strict scan —
  where no cell can be placed under a column, so the cell-level checks do not run — and past a
  timestamp or value-cell error, which stops the frame-level shape and integrity checks;
  machine-readable `FileReport`s — each carrying the file's raw-byte `sha256`, the report's data
  identity), `check_frame` integrity invariants, `sufficiency_check` (an index too short to close
  one observation refuses up front), `DataError` → exit 2.
- `api.py` — the caller binds the thesis's declared keys to the files it has
  (`resolve_data_files(thesis, {key: path}, {key: column})` → a `DataFiles`), loads ONCE via
  `load_market_data(thesis.data, files)` (both re-exported here — this module is the public API's
  one home and the loader its front door; `canonical_dsl_hash` is re-exported here too, so a host
  computing a thesis's identity imports it from the public home, never from `gate`) and hands the
  materialized `MarketData` — a load of the thesis's own data declaration: both entry points
  refuse (`ValueError`) one loaded under another `thesis.data`, its feeds aligned under other lags
  and window — to `compile_thesis(thesis, md)`
  (sufficiency → `run_backtest`; never loads; the result's `data_report` is its own — the load's
  report, or a copy carrying this horizon's `barely_sufficient` warning, `md.report` never
  written) and/or
  `list_entries(thesis, md)` (every firing timestamp of the full entry mask, bit-identical to the backtest's,
  plus TWO per-bar frames instead of one wide one, because they answer two different questions and
  the CLI nominates them one flag each: `root_series` — the deduplicated root-series value columns,
  each threshold operand except bare constants, labelled by `render_series` — is what
  `--root-series-out` writes, since it answers WHY a bar did or did not fire; `entry_flags` — the
  0/1 column per combo × target, canonically named, unique as the strict reader compares headers
  (case-insensitively: of targets `A` and `a` the second takes a `#N` suffix) — is what
  `--entry-flags-out` writes, and it answers WHETHER it fired. Keeping the two apart is what lets
  the value frame reserve nothing but `datetime`, so an external feed named `entry` keeps its name.
  `entry_flags` is also the ONE output that can express a final-bar firing: it anchors no
  observation, so the runner correctly drops it from the OBSERVATIONS and no trades row exists to
  carry it — a caller asking "is this firing NOW?" reads the last row of the entry-flags CSV, or
  `entries` through the library. No freshness claim — `series_end` just reports where the provided
  index stops).
- `serialize.py` — JSON-safe views (`serialize_result` returns the summary verbatim, never trades)
  + the caller-nominated CSV writers (`write_trades_csv` for `--trades-out`,
  `write_root_series_csv` for `--root-series-out`, `write_entry_flags_csv` for
  `--entry-flags-out`); nothing it produces rides stdout.
- `dsl/` — the thesis DSL (pydantic, `extra="forbid"`, strict, frozen; a `DataSpec` NAMES
  series and locates none of them, and `BacktestParams` carries no sampling knob of any kind, so
  the DSL can express neither where its data lives nor a partition of it that some cells see and
  others do not). Layered behind the `schema.py` facade: `nodes` (the Series vocabulary and its
  union — the data leaves, the time transforms, the cross-sectional trio with its `where`/`group`
  population selectors, the operator pair, the event-anchor family `mask` / `bars_since_event` /
  `event_value` / `event_agg` that embeds a Condition, `native` for feed-clock transforms, and
  the `AxisRef` shared-axis reference), `conditions` (threshold, the combinators, `rolling`,
  `first_true`, `lag` — and the ONE rebuild site for both mutually recursive vocabularies),
  `thesis_model` (feed/data/outcome/params/`axes`/`Thesis`), `traverse` (the tree walks, in a
  direct layer and a crossing layer that reaches into embedded conditions; one sweep-scan state
  mirrors the engine resolver) and `render` (presentation-only labels for Series and
  Conditions).
  `compiler/` — data loading (`resolve_data_files` is where the invocation's `--data` /
  `--column` pairs meet the thesis's declared keys, and the only thing that checks the two
  agree; the loader reads each distinct path once, every key bound to it parsing that snapshot,
  and returns frames over READ-ONLY buffers — a tripwire for the in-place value write, which raises
  instead of serving stale memo results; pandas cannot freeze a frame against a replaced column, a
  breach of the no-mutation contract that `MarketData.clear_memo()` resets — and retains each feed's native post-lag prints as `MarketData.externals_native`
  for the `native` node) + numpy/numba transform kernels (`nb.py`, `vectorize.py`,
  `transforms.py`) + the pure path/source leaves (`paths.py`, `sources.py`) + the runner, which
  measures every declared cell once over the whole index and builds the per-cell panel off the
  DECLARED grid.
  `analysis/` — the statistical layer, one module per estimator family behind the `stats.py`
  facade (`_pools`, `_conditioning`, `_hac`, `_cscv`, `_episodes`, `_bootstrap`,
  `_baseline`, `_rotation` — every statistic NOMINAL and per-cell) and result types. The
  statistical doctrine — what the checklist reads vs what is evidence-only, observer purity,
  sweep semantics — lives in these modules' own docstrings and module headers; read them before
  touching any of the three.
- `reference/dsl-schema.md` — the agent-facing DSL guide (`seikan schema --markdown`).

### dev commands

```bash
uv sync                   # install (numpy/numba/scipy/pandas/pydantic only — no services)
uv run ruff format --check src tests   # formatting is enforced, never a matter of taste
uv run ruff check src tests            # lint
uv run mypy                            # strict, over the whole package
uv run pytest             # the WHOLE suite always runs; zero skips
                          # (numba cold-compiles on the first backtest test)
uv run seikan schema      # the machine-readable self-description
uv build --wheel          # the distribution a host installs (read-only is fine — see the
                          # NUMBA_CACHE_DIR note below)
```

There is no blessed bundler spec in this repo. A host freezing the CLI (e.g. with PyInstaller)
owes three things: ONEDIR, not onefile — a host service invokes the CLI repeatedly and onefile
would re-extract numpy/scipy/numba on every call; ship the `seikan` package as plain `.py` source
(PyInstaller's `module_collection_mode`), because numba's `cache=True` kernels need a real source
file to locate and validate their on-disk cache entries; and a writable cache location per the
note below.

Note: the first `seikan run` on a machine pays numba's JIT compile (seconds); kernels are
`cache=True`, so later runs reuse the on-disk cache. By default numba writes that cache next to
the kernel sources (`src/seikan/compiler/__pycache__/`; inside a bundle, next to the shipped
sources). Under a READ-ONLY install — a wheel installed read-only, a sandboxed venv — numba
refuses to compile a cacheable kernel with nowhere writable to put it, so the host must point
`NUMBA_CACHE_DIR` at a writable directory. That variable is part of this convention: the ONE
environment requirement the engine imposes outside the `SEIKAN_*` namespace. The cache is a
compilation artifact, never run state — its presence, absence or location changes no number in
any report.

### deferred (later phases)

Richer report visualization. Thesis lifecycle (draft/active/dormant), data acquisition, source
credibility, and scheduling are deliberately OUT — they belong to the calling agent, as does
selection among the reported cells and the multiplicity that selection carries.

Two documented LIMITATIONS deferred by decision (2026-08), not oversights — both produce
conservative, fully-ledgered results today, never wrong ones:

- **Evaluation vs materialization bounds.** `data.start`/`data.end` are MATERIALIZATION bounds:
  they slice the raw data before transforms and forward outcomes are built, so a bounded study
  loses left-edge sample to signal warmup (ledgered as warmup, never a hole) and right-censors
  its final horizons (ledgered as `open`, which the checklist allows). There is currently no
  way to say "warm the signals on pre-window history" or "close the final windows on
  post-window data" — provide more history and accept measurement from the start bound, or
  pre-trim the CSVs (the invocation locates files, so trimming is the caller's act). Separating
  the two windows is a future versioned change with a large geometry blast radius (rotation
  null, PBO blocks, baseline, coverage ledgers all re-derive).
- **Feed staleness at the OUTCOME.** An external feed is anchored backward-as-of with NO age
  bound (the legitimate design for sparse stamping — a weekly feed on daily bars). The hazard
  worth knowing: a DEAD feed (no new prints) serving as a `params.outcome.series` yields equal
  endpoints, which measure as finite 0.0 returns and close as `"horizon"` — not `no_outcome`.
  Decision-side staleness is caller-expressible today (`days_since(feed) <= N`); the outcome
  side has no expressible bound, so whether an outcome feed is alive over the measured window
  is the caller's data honesty, exactly like vintages. An optional per-feed `max_age`
  (censoring stale joined values fail-closed) is the designed future fix.

Three more, deferred with the 4.0.0 DSL expressiveness release (2026-09-13), each with an
honest workaround today:

- **Target-clock resampling.** `native(feed, expr)` evaluates on a FEED's own clock, but the
  engine has no session calendar, so resampling the TARGET's own bars (a weekly close from daily
  bars) still needs a caller-consolidated feed stamped at period completion; likewise the LAST
  trading day of a month needs a schedule feed (the first is `change(calendar(month), 1, diff)
  != 0`).
- **Applicability vs missing data.** A basket member that leaves the universe and stops trading
  is still a raw-source hole to `source_coverage` — telling "inapplicable" from "missing" is a
  versioned ledger-contract change; today the `where` eligibility selector keeps such a member
  out of the cross-section, and its hole is reported, never absorbed.
- **Reference universes and member weights.** A cross-section is always the declared targets:
  a rank against an OUTSIDE universe is supplied as a feed of breakpoints; a weighted mean is
  the exact composition `cross_agg(x·w, mean) / cross_agg(w, mean)`.

Two pieces of CALLER discipline worth naming without legislating: a report is a FILE artifact —
reference it by path and identity (`dsl_hash`, `data_digests`), never re-type its numbers through
a model, because a transcription is a fork of the evidence; and whether a host treats an
unavailable engine as fatal or degrades to running without measurement is the host's policy — the
engine neither requires nor forbids it, promising only the tiered exit codes and the one envelope.
