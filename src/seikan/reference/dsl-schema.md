# seikan thesis DSL — schema reference

A `Thesis` is a pydantic model (run it with `seikan run <thesis.json> --report-out <report.json>
--data KEY=PATH [--column KEY=COL] …`; the machine-readable JSON Schema is `seikan schema`). Time
series are referenced by **logical key** — a name the thesis declares — and the invocation supplies
both facts the document deliberately withholds: `--data` names the CSV behind each key, and
`--column` names which column of that CSV the key reads, needed only where the file holds more than
one numeric series. Every file must satisfy the strict data contract (`seikan schema` →
`csv_format`; pre-flight with `seikan check-data`). A target is an **OHLCV** file (a price) or a
**series** file (a value column — a yield, spread, valuation multiple, strategy index; see
DataSpec); an external feed is one numeric **column** of a CSV, and WHICH column that is arrives
with the run, never with the thesis.

seikan is a **pure observer**: a thesis is an `entry` firing condition + a measurement `horizon`.
There is **no exit condition** — every bar `entry` fires opens an independent, overlapping
forward observation measured over `horizon` bars (sign set by `direction`; WHAT is measured is set
by `params.outcome` — the target's pct return by default). No stop-loss / take-profit / position
state machine.

It is also a **per-hypothesis reporter**: every parameter × horizon cell your sweep declares is
measured over the whole sample and reported independently, including combos that never fired.
seikan does not select, rank, or crown a winner, and no statistic it emits is corrected for the
size of your grid — **you** choose among the cells and own the multiplicity of having looked at
`n_hypotheses_attempted` of them. See "Reading the result" below before you act on any number.

## Top level

```jsonc
{
  "name": "iv-mean-reversion",
  "description": "optional, free text",
  "target_mode": "conjunction",  // or "basket" — how multiple targets relate; defaults to conjunction
  "data":   { /* DataSpec */ },
  "entry":  { /* Condition — the firing signal */ },
  "params": { /* BacktestParams — horizon defaults to 1; set it explicitly */ }
}
```

`target_mode` declares how a multi-target thesis's targets relate — `"conjunction"` (the
default: the targets are the thesis's regime, each measured on its own) or `"basket"` (the
targets form one cross-section per bar, graded as one pool). See "Target modes — conjunction vs
basket" below before declaring a basket.

**The document is STRICT, on purpose.** Unknown keys refuse (`extra="forbid"`), and so do lax
type spellings: `"window": "3"`, `"window": true` and `"window": 3.0` all refuse rather than
coerce to `3` (an integer literal in a float position stays legal — JSON has one number type),
`"per_target": "false"` refuses rather than becoming `false`, and a repeated JSON object key at
ANY depth refuses at parse (Python's last-key-wins would make the effective document depend on
the parser). Validated documents are also frozen: field assignment after validation raises.

## DataSpec

Series are **named here and located at invocation**. The DSL declares logical keys; `seikan run`
binds each key to a CSV with a repeated `--data KEY=PATH` (relative paths resolve against the
working directory of the invocation), and to a column of that CSV with a repeated `--column
KEY=COL` wherever the file holds several numeric series. A path inside the document would make the
same exam over re-pulled data a *different* document — `dsl_hash` would move for a reason that has
nothing to do with what is being asked — and a column NAME is that same fact one level in: it is a
property of the file that happens to answer a key (this vendor ships three yields in one CSV under
its own headers, that one ships each in a file of its own), never a property of the question, so a
column in the document would let a renamed header or a split file move a hash the thesis never
moved. So the thesis says WHICH series it measures, and the run says where they are and which
column of each to read.

- `targets`: [str] — the target keys, backtested together (one column each). At least one.
- `start` / `end`: optional ISO date strings to clip the window.
- **Target shape, and the one binding a target may need** (no DSL field of its own — the shape is
  read off the file, and the column is bound at invocation). A file WITHOUT open/high/low/close
  columns loads as a **series** target: its value column is measured directly
  (open=high=low=close=value synthesized so the close-reading algebra applies; no volume, so
  `field:"volume"` is unavailable). A series file holding ONE numeric column names itself and needs
  nothing. One holding several says nothing about which of them the target IS, so the run must:
  `--column <target>=<col>`. Leave it out and the load refuses (exit 2) naming the columns the file
  actually holds; misspell it and it refuses the same way, listing them again rather than guessing
  at which one you meant. An **OHLCV** target takes no binding at all — a price target always
  measures its open-anchored prices, so there is no column for one to choose — and a binding on an
  OHLCV file is an error rather than a no-op, because a caller who typed it meant something the
  file cannot do. All targets of one thesis must share one shape (never mix a price with a yield —
  their measurements aren't commensurable).
- `external`: {feed_name: ExternalFeed} — alternative-data feeds (numeric). Reference a feed in a
  condition with `{"type": "external", "name": "<feed_name>"}`. Each entry configures only the
  SEMANTIC read — what the series MEANS to this thesis, which is the only thing the document is
  entitled to fix; the file and the column both arrive with the invocation, like every other
  locating fact:
  - `{}` — one series shared by (broadcast to) every target, answering to the feed's own key.
  - `{"lag": L}` — the same shared series, carrying a publication lag (below).
  - `{"per_target": true, "lag": L}` — **per-target** series (e.g. each ticker's own IV). It
    answers to one derived key per target, `<feed>@<target>`; cover is by construction, so there is
    nothing to keep in sync with the target list. `lag` is optional here too.

  A feed whose CSV holds several numeric columns says which one it reads at INVOCATION: `--column
  <feed>=<col>` for a shared feed, `--column <feed>@<target>=<col>` for one member of a per-target
  feed — and the members may name DIFFERENT columns, which one DSL field could not say
  (per-ticker files sourced from different vendors rarely agree on a header). A single-column file
  needs no binding; a multi-column one without a binding refuses at load (exit 2) naming the
  columns it holds, and so does a name no column of the file answers. Matching is
  case-insensitive.

  `lag` (default 0) shifts the feed's timestamps forward before it is anchored on the bar index, so a
  value is not usable before its real release date. An **int is calendar days**; a string is a pandas
  Timedelta like `"36h"` (careful: `"1m"` is one *minute*; and `"nan"`/`"NaT"` refuse — a lag that
  parses to NaT would silently behave as zero, the look-ahead direction). Macro prints publish with a
  delay — model it, e.g. CPI `"lag": 14`.

**Every feed name referenced by the entry condition must be a key of `data.external`** — validation
fails upfront (naming the missing feeds) otherwise. The CONVERSE holds too: a declared feed that
nothing references (entry tree, `params.features`, or `params.outcome.series`) refuses — an unused
declaration still demands a file at invocation and moves the `dsl_hash` without changing the
measurement.

### Data keys

The run's keys live in ONE flat namespace, because that is the namespace you type on a command
line — and it is the namespace BOTH invocation flags address: one key names the file that answers
it (`--data KEY=PATH`) and the column read out of that file (`--column KEY=COL`), so learning the
keys once is learning both flags. In resolution order they are: every target name; then per feed,
its own name (shared) or `<feed>@<target>` for each target (per-target); then `benchmark` when
`params.benchmark` is `"market"` — a reserved key naming the **OHLCV** series excess returns are
measured against. Unlike an external feed the benchmark is *outcome measurement*, not a decision
input: its open price is sampled at exactly the observation anchor bars (never forward-filled, no
lag) — bars where it is missing censor the observation (`exit_reason="no_benchmark"`). It therefore
takes a path like every other key and a column NEVER: there is nothing for one to select when the
answer is always that file's `open`, so `--column benchmark=…` is refused (exit 3) rather than
quietly ignored.

Validation refuses a target or feed name that is empty, carries surrounding whitespace, contains
`=` or `@` (the pair separator and the per-target derivation), is the reserved word `benchmark`, or
is claimed by both a target and a feed. And `seikan run` refuses a `--data` set that does not answer
the declared keys **exactly** — a missing key would measure a thesis you do not have, an unknown one
means you believe this thesis reads something it never mentions. `--column` is judged by the same
standard held the other way round: it is OPTIONAL per key (most files name their own single value
column), but a binding that could never be read is refused on sight (exit 3) — a key this thesis
does not declare, or `benchmark`. Whether the NAME you bound exists in the file takes bytes to
answer, so that refusal comes from the loader instead (exit 2, listing the file's columns).

```
seikan run thesis.json --report-out report.json \
  --data NVDA=data/nvda_ohlcv.csv \
  --data AMD=data/amd_ohlcv.csv \
  --data iv30@NVDA=data/nvda_options.csv --column iv30@NVDA=iv30 \
  --data iv30@AMD=data/amd_options.csv   --column iv30@AMD=iv_30d \
  --data benchmark=data/spy_ohlcv.csv
```

Read that run line as two answers rather than one: five keys the thesis declared, five files that
answer them, and two column bindings for the only two files that needed one — the option panels
come from different sources and spell the same quantity differently, which is a fact about those
files and about nothing else. The price targets and the benchmark bind no column at all: an OHLCV
file holds nothing for a binding to choose between.

### The CSV files you reference (the strict data contract)

seikan is a verifier for **untrusted external data**: every referenced file is checked before the
engine sees a number, and a violation exits 2 with a machine-readable `data_report` naming file /
column / csv line. Pre-flight with `seikan check-data <files…>`. The contract:

- **Timestamps**: a column named `datetime` (case-insensitive), else the first column. Strict
  **ISO-8601** (`YYYY-MM-DD` or a full timestamp), **timezone-naive**, unique, sorted ascending.
  No other date format is ever guessed (`01/02/2021` is rejected, not interpreted).
- **Values**: plain numbers, forced to float. The only missing-value markers are an empty cell or
  `nan`. No thousands separators, no currency symbols, no `inf`.
- **OHLCV shape** (columns `open,high,low,close` + optional `volume`): `high ≥ max(open,close)`,
  `low ≤ min(open,close)`, prices > 0, volume ≥ 0. Violations **refuse — never clamp** (a verifier
  does not mutate evidence).
- **Series shape**: one or more named numeric columns — but at most ONE of the four OHLC names
  (`open`/`high`/`low`/`close`): a series read of a file carrying two or more of them refuses as
  ambiguous (it would be broadcast into all four price slots while every OHLC invariant above is
  skipped). A lone `close` column is an ordinary series file; a full OHLC set is an OHLCV file.
- Warned but admitted (crashes are the research subject, not dirt): interior NaN holes,
  |1-bar move| > 50% (check your split adjustment), large calendar gaps.

Macro series are usually indexed by their *period* date, not their release date — add a `lag` to the
external entry (e.g. CPI ≈ 14 days) to avoid lookahead bias. The thesis core forward-fills external
feeds onto the target's bar index, so mixing cadences (a daily target with a monthly feed) is fine.

For a multi-target thesis whose conditions use a per-ticker feed, declare it per-target and answer
one `--data iv30@<ticker>=…` per ticker:

```jsonc
"external": { "iv30": { "per_target": true } }
```

A feed that lives in a multi-column CSV is declared exactly like one that does not (`"skew": {}` —
the entry says nothing about the shape of the file), and the run says which column it reads:

```
seikan run thesis.json --report-out report.json \
  --data SPY=data/spy_ohlcv.csv \
  --data skew=data/options_panel.csv --column skew=iv_skew
```

The thesis is the same document whether the vendor ships one column or twenty, and stays the same
document when they re-head the file next quarter: only the invocation changes.

## Target modes — conjunction vs basket

`target_mode` routes two different questions. **Conjunction** (the default) answers "does this
claim hold for EACH of these?": the targets are the thesis's regime, measured side by side with
the weakest deciding — a mode a single-target thesis is always in, and the one that forms no
cross-target statistic at all. **Basket** answers "does position WITHIN
this group predict what follows?": the targets form ONE cross-section per bar, the
cross-sectional Series nodes (`cross_rank` / `cross_demean` / `cross_agg`) and
`benchmark: "cross_mean"` become legal, and every cell gains a **pooled** cross-target
statistics block which the checklist grades instead of per-member floors. A basket thesis is
about the RANKING RULE, not its current top name — "the top momentum quintile of this group
outperforms the group", never "buy whichever ticker ranks first today". The engine still selects
among members never: `by_target` stays in the report as attribution, read by no check.

Both modes stand on the same ONE-bar-clock rule: multi-target runs refuse unequal target indices
(exit 2, naming what is missing where) rather than intersecting them — and basket leans on that
clock doubly, because a bar's cross-section only exists where every member stamps the same bars.

What validation refuses (exit 3, `dsl_invalid`), with the exact messages:

- basket below two targets — "target_mode='basket' requires >= 2 targets (data.targets); got N — a
  basket of one is degenerate; use target_mode='conjunction'".
- a cross node outside basket — "cross-sectional node(s) [...] require target_mode='basket' —
  conjunction declares the targets an independent regime and forms no cross-target statistic;
  declare target_mode='basket' to rank within the group". `params.features` are scanned too: a
  cross-sectional feature snapshot couples the targets the same way an entry operand does.
- an unsatisfiable definedness floor — "cross-sectional node '<type>' has min_valid=M but only N
  targets are declared; it could never be defined".
- basket + `outcome.kind: "diff"` — "target_mode='basket' cannot be combined with outcome kind
  'diff': the pooled cross-target statistics average returns across members, which needs a
  common unit the engine cannot certify for level changes (bp, index points, ratio turns);
  pct/log are scale-free — use one of those or target_mode='conjunction'".
- `cross_mean` outside basket — "params.benchmark='cross_mean' requires target_mode='basket': it
  demeans each member's forward return by the basket mean, coupling the targets in the OUTCOME
  exactly as cross nodes couple them in the signal — which conjunction forbids".

The converse is NOT required: a basket thesis need not carry a cross node — the mode alone
changes the statistical read (the pooled per-cell block, the pooled floors) — and
`benchmark: "market"` stays legal in basket.

**Dense always-on basket signals: wrap the rank threshold in `first_true`.** A bare
`cross_rank(x) >= 0.8` entry fires (for some member) on nearly EVERY bar — SOME member is always
in the top quintile — so every overlapping observation window chains into ONE transitively
merged episode cluster, `episode_stats.max_cluster_share_abs` reads 1.0, and the `concentration`
check refuses every cell. That is the checklist working, not a bug: an always-on signal measures
membership churn smeared into one endless episode, not a decision anyone could time — the same
property any dense conjunction signal has, with the same doctrine answer. The
episode-entry primitive `first_true` (fire on ENTERING the quintile, optional `cooldown`) is
both the honest measurement and the sparse form the episode machinery is built for — see the
third worked example below.

## Series nodes (the operands of conditions)

Leaves:
- `{"type": "field", "column": "open|high|low|close|volume"}` (default `close`)
- `{"type": "constant", "value": 1.23}` — `value` may also be a **list** to sweep the threshold as
  its own axis (like a transform-window sweep); then `name` is **required** to label it, e.g.
  `{"type": "constant", "value": [-0.20, -0.30, -0.40], "name": "dd_thresh"}`. The `name` must be
  unique and not `target`/`horizon`/a trade-or-feature column. A SCALAR `value` refuses a `name`:
  execution never reads it, so it would move the hash without moving the measurement.
- `{"type": "external", "name": "<feed_name>"}`
- `{"type": "calendar", "field": "month|day_of_week|day_of_month|days_to_month_end"}` — the bar
  timestamp's calendar attribute (month 1-12; day_of_week 0=Mon..6=Sun; days_to_month_end =
  calendar days left in the month, 0 = last calendar day). The seasonality primitive:
  turn-of-month = `or(day_of_month >= 25, day_of_month <= 3)`. Calendar-day arithmetic only.
- `{"type": "days_since", "name": "<feed_name>"}` — calendar days since the feed's most recent
  NATIVE observation (its real stamps + `lag`, not the forward-filled values); NaN before the
  first stamp. The event-distance primitive: PEAD window = `days_since(earnings) <= 3`; also a
  staleness guard on any feed. For SCHEDULED future events (FOMC), feed a user-computed
  days-until-next-event series instead — a `days_until` over stamps would read the future.

Transforms (the ONE operator family — `window`/`periods` may be a **list** to sweep a grid).
There is **no Indicator family** (no RSI/ADX/ATR/OBV/VWAP/ROC); compose what you need from these
primitives. Classic technical indicators are deliberately absent — too widely used to be
informative once everyone already trades them.

- `ema` (input, window) — exponential moving average of any Series.
- `zscore` (input, window, mean_type `"sma"|"ema"` default `"sma"`).
- `percentile` (input, window) — fraction of the window strictly below the current value
  `count(value < current)/window` in `[0, (window−1)/window]` (the current bar sits in its own
  window, so exactly 1 is unattainable).
- `change` (input, periods=1, kind `"pct"|"log"|"diff"` default `"pct"`) — k-period change,
  mirroring `Outcome.kind`: `pct` = (cur/prev − 1), `log` = ln(cur/prev), `diff` = cur − prev
  (level change — the honest form for rates/spreads/multiples). `pct` and `log` require BOTH
  endpoints strictly positive and are NaN otherwise (ratio algebras invert silently on a signed
  scale — the same rule the forward outcome applies); signed series belong to `diff`. A list
  `periods` sweeps as `change_periods`. Prefer `change(kind:"diff")` over composing
  `binary_op(x, shift(x), "-")`.
- `shift` (periods=1) — the input `periods` bars ago (backward-only; leading bars NaN). Prefer
  `change` for k-period pct/log/diff; `shift` remains for level comparisons
  (`close > shift(high, 1)`). Free for the nesting limit, like `binary_op`.
- `rolling_agg` (window, `agg`: `"max"|"min"|"mean"|"std"`) — trailing-window aggregate when
  `window` is set (std is population, ddof=0; NaN until the window is full and finite). **Omit
  `window`** for an EXPANDING (all-time) `max`/`min` only — expanding `mean`/`std` are rejected.
  Prefer the dedicated `drawdown` node for depth-below-peak; the composed form
  `binary_op(close ÷ rolling_agg(close, N, "max"))` remains valid. A **simple moving average** is
  `agg:"mean"`; **realized vol** is `rolling_agg(change(close, kind="log"), N, "std")`.
- `drawdown` (optional `input` defaulting to close, optional `window`) — fractional depth below a
  peak: `input / peak − 1` (≤ 0). POSITIVE-scale domain: a non-positive peak → NaN (the ratio
  inverts on a negative scale; signed series compose level reads with `change(kind:"diff")`
  instead). Omit `input` to read the target's close. `window` set → trailing
  N-bar peak; omit `window` → expanding (all-time) peak. "In a ≥30% drawdown" is
  `{"type": "threshold", "left": {"type": "drawdown"}, "op": "<", "right": {"type": "constant", "value": -0.30}}`.
  Counts as one operator level (recovers nesting budget vs composing `binary_op` + `rolling_agg`).
- `runup` (optional `input` defaulting to close, optional `window`) — fractional height above a
  trough: `input / trough − 1` (≥ 0), the exact mirror of `drawdown` — positive-scale domain
  included (a non-positive trough → NaN). Reads both ways: recovery/
  stabilization after a trough (`runup >= 0.05`) or extension/overheat near a peak.
- `bars_since_extremum` (`extremum`: `"max"|"min"`, optional `input` defaulting to close, optional
  `window`) — bar count since the most recent bar attaining the trailing/expanding max|min. Ties
  reset to the **most recent** attaining bar. `extremum:"max"` = drawdown duration (bars since the
  peak); `extremum:"min"` = stabilization ("no new low in K bars"). Trailing form NaNs until the
  window is full and finite. Example duration gate:
  `{"type": "threshold", "left": {"type": "bars_since_extremum", "extremum": "max"}, "op": ">=", "right": {"type": "constant", "value": 60}}`.
- `rolling_corr` (`left`, `right`, `window` ≥ 3) — trailing-window Pearson correlation of two
  Series. Per-target TIME-axis only (ranking across the targets at a bar is `cross_rank`, basket
  mode). NaN unless the whole window
  is finite in BOTH inputs AND both window stds > 0 (zero-variance → NaN, never ±1). Counts as one
  operator level with two children. Recipe for "corr(stock daily return, Δ option EOD IV30)":
  declare the IV feed in `data.external`, then
  `rolling_corr(change(close), change(iv30, kind="diff"), window)`.
  A list `window` sweeps as `rolling_corr_window`.

Cross-sectional transforms (**basket mode ONLY** — see "Target modes" above; using one under
conjunction is exit 3). Unlike the trailing transforms, which operate along each target's own
TIME axis, these operate ACROSS the targets at each bar — the primitives for relative-value
statements ("the cheapest tier within the group outperforms"). A bar's cross-section is defined
only where at least `min_valid` (default 2) targets hold finite values; elsewhere the node is NaN
and the bar simply doesn't fire. `min_valid` is deliberately NOT sweepable — a swept definedness
floor sweeps the SAMPLE, not the hypothesis. Each node counts as one operator level.

- `cross_rank` (input, min_valid=2) — ascending fraction-rank of the target's value among all
  targets' finite values at the bar: `(avg_rank − 1)/(k − 1)` in `[0, 1]`, average ranks on ties;
  NaN where the target's own value is NaN or fewer than `min_valid` targets are finite.
  `cross_rank(x) >= 0.8` IS top-quintile membership — the quantile primitive; there is no
  separate quantile node.
- `cross_demean` (input, min_valid=2) — the target's value minus the cross-target mean of the
  finite values at the bar (self included); same NaN gating. The cross-sectional ZSCORE recipe:
  `binary_op(cross_demean(x), "/", cross_agg(x, "std"))` — the same nesting cost as either node
  alone, and a zero-dispersion bar divides by zero → NaN via the `binary_op` `/` contract (never
  fires).
- `cross_agg` (input, `agg`: `"mean"|"median"|"std"|"frac_positive"`, min_valid=2) — a
  cross-target AGGREGATE at each bar, broadcast back to every target column — the breadth /
  dispersion primitive ("70% of the group above its 200d SMA" via `frac_positive` of
  `binary_op(close - rolling_agg(close, 200, "mean"))`; a cross-sectional-vol regime via `std`).
  Unlike the other two, the value is a property of the CROSS-SECTION, not of the individual
  target, so a bar carries the VALUE (for every column) wherever `min_valid` targets are finite
  even when the target's own value is still warming up — intentional, do not "fix". The FIRING
  latch is stricter: a member cannot fire off the broadcast before its own series has ever
  produced a value (its warmup gates on its own input as well), so a staggered-start basket
  measures each member only over the history it actually has. `std` is the population std
  (ddof=0, matching `rolling_agg`); `frac_positive` is the fraction of finite values > 0, in
  `[0, 1]`.

Combinators:
- `{"type": "binary_op", "left": <Series>, "right": <Series>, "op": "+|-|*|/"}` — element-wise
  arithmetic of two series (e.g. a fast−slow EMA spread, or a normalized ratio). Division by zero
  yields NaN (the bar simply doesn't fire). `binary_op` is "free" for the nesting limit below.
- `{"type": "unary_op", "input": <Series>, "op": "abs|log|sign|sqrt|neg"}` — element-wise unary
  arithmetic; out-of-domain (log of a non-positive, sqrt of a negative) → NaN. `abs` unlocks the
  magnitude classics (Amihud illiquidity `|ret|/dollar-volume`, |surprise| conditioning); for a
  two-sided *condition* prefer `or(z > 2, z < -2)`. Also free for the nesting limit.

**Nesting limit:** operators may nest at most **five levels** deep. Each transform (`ema`,
`zscore`, `percentile`, `rolling_agg`, `drawdown`, `runup`, `bars_since_extremum`, `change`,
`rolling_corr`, the cross-sectional nodes) counts one level, so a five-stage pipeline like
`zscore(ema(percentile(change(rolling_agg(...)))))` is the deepest shape allowed and a sixth
wrap is rejected at validation. `binary_op` / `unary_op` / `shift` are transparent (they don't
count as a level), so `binary_op(ema(x), ema(y))` costs one level, not two; `rolling_corr`
counts one level over its deeper child. Depth is a budget, not a goal — every extra level is
another window a reader must audit, so prefer the shallowest expression that says the thesis.

## Condition nodes

- `{"type": "threshold", "left": <Series>, "op": "<|<=|>|>=|==|!=", "right": <Series>}`
- `{"type": "and|or", "conditions": [<Condition>, <Condition>, ...]}`  (min 2)
- `{"type": "not", "condition": <Condition>}`
- `{"type": "rolling", "window": N, "agg": "all|any|count", "condition": <Condition>}` — fires when
  the inner condition held on every (`all`) / at least one (`any`) / at least `min_count` (`count`)
  bar of the trailing window. `count` **requires** `min_count` (the K of "at least K of N", e.g. `>= 3
  of the last 5 bars closed down" = `{"agg": "count", "window": 5, "min_count": 3, "condition":
  <close-down>}`) — a sustained-regime trigger `all`/`any` can't state. `window` may also be a **list** to
  sweep the trailing-window length as its own `rolling_window` axis (like a transform-window sweep),
  e.g. `"window": [3, 5, 10]`; keep `min_count` scalar. Prefer `bars_since_extremum(max) >= N` for
  true duration-since-peak; `rolling` + `agg:"all"` over a drawdown threshold remains valid as
  "sustained depth for ≥ M bars".
- `{"type": "first_true", "condition": <Condition>, "cooldown": K}` — **episode entry**: fires only
  on a false→true transition of the child's tradable signal. The first True after warmup does NOT
  count (must have seen an initialized False first). Optional `cooldown`
  (default 0) suppresses re-fires for K bars after a fire; a list sweeps as `first_true_cooldown`.
  The episode-entry primitive: measure forward return from the bar a regime is first entered
  (deep drawdown, end-of-bull risk alarm, …), not every bar inside it. Also the **crossover recipe**:
  `first_true(threshold(fast > slow))` (there is no dedicated crossover condition). Sparse episodes
  legitimately leave the per-cell `support` floors unmet (`n >= 30`, `n_nonoverlap >= 8` — per target in
  conjunction, on the pooled panel in basket) — that is
  the checklist working, not a bug: the pool is too thin to carry a support claim, there is no
  alternate rare-event exam, and nothing is lost by it because the cell's complete
  statistics (episode clusters, concentration, `rot_p`, `t_hac`, its coverage ledgers) are reported
  either way for you to read.

### Thesis recipes (archetype-neutral engine)

The engine validates **any** observer-pure thesis (entry + horizon; no exit) — per-target
time-series under conjunction, cross-sectional under `target_mode: "basket"`. Two worked recipes
here; the basket archetype is the third full example at the bottom of this guide:

#### (a) Deep-drawdown rebound / mean-reversion

Observer-native pattern for "stock endured a material drawdown, looks cheap, then rebounds":

1. **Depth** — `drawdown` (trailing N-bar or expanding) `< -(X)` e.g. `-0.30`.
2. **Duration** — `bars_since_extremum` with `extremum:"max"` `>= N` (bars since the peak), or
   wrap depth with `rolling` `agg:"all"` over M bars (sustained depth).
3. **Recovery / stabilization guard** (optional but recommended) — `runup >= r` (bounced off the
   trough) and/or `bars_since_extremum` with `extremum:"min"` `>= K` (no new low for K bars).
4. **Episode entry** — wrap with `first_true` so each drawdown episode contributes one observation.
5. **Valuation** (optional) — `and` with a `threshold` on an `external` PE/PB feed (CSV; set
   publication `lag`). No built-in valuation ratios.
6. **Horizon** — long forward window in **bars** (e.g. ~21×months on daily); convert calendar
   months yourself — windows are bar-indexed.
7. **Benchmark** — `params.benchmark: "market"` + a `benchmark` data key so long-horizon results are
   excess-of-market, not beta. Direction defaults to `longonly`.

Post-entry path evidence rides the per-observation trades frame (`--trades-out`) as `mae` / `mfe`
(the RAW-path worst adverse and best favorable interim marks), `bars_to_positive` and
`bars_to_trough` — read them for your own deployment judgment; the excursion
pair is also aggregated per (cell × target) as `mae_quantiles` / `mfe_quantiles`, so the path read
does not require the CSV. There is no aggregate path-risk panel and no cost model: those would be
winner-only summaries, and there is no winner to summarize. The decision side has two files of its
own, split by the question they answer. `--root-series-out` writes the per-bar VALUES of every root
series in the entry tree (each threshold operand except bare constants, one column per deduplicated
node, `@<target>` families when several targets) — the *why* a bar did or did not fire.
`--entry-flags-out` writes the per-bar 0/1 firing matrix — one integer column per parameter combo ×
target, named `entry` / `entry[axis=value,…]` with an `@<target>` suffix when several targets run,
bit-identical to the backtest's own mask — the *whether*. A fired bar also reaches you through the
trades CSV, in observation shape.

One firing reaches only the flags file, and it is the one you are most likely to want: a firing on
the **final bar** has no next open to anchor at, so it opens no observation, and a frame in
observation shape has nowhere to put it (the runner drops it from the observations by design, and
no `outcome_coverage` count records it either). So if you are asking "is my thesis firing right
now?", read the last row of the `--entry-flags-out` CSV — never infer it from an empty trades tail.
The library equivalent is `api.list_entries`, whose `entries` / `entry_flags` report raw firings
rather than observations.

#### (b) End-of-bull / exit risk-alarm

Observer-native pattern for "synthetic warning features fire → leave before the peak":

1. **Feeds** — declare each warning feature in `data.external` (e.g. reverse IV skew, long leverage,
   opinion heat) with honest publication `lag`. Normalize via `zscore` / `percentile` / `change`.
2. **Composite** — either an `and` of thresholds, or a weighted sum via transparent `binary_op`
   (stays within the 5-level nesting cap) then threshold the score.
3. **Episode entry** — wrap with `first_true` so each alarm episode contributes one observation.
4. **Direction** — `params.direction: "shortonly"` so the measured edge is positive when price
   falls / underperforms after the alarm (the gate requires a positive edge).
5. **Horizon** — sweep the forward window as a response curve (e.g. `[5, 21, 63]`).
6. **Benchmark** (optional) — `params.benchmark: "market"` (+ `--data benchmark=…`) to strip beta from the excess underperformance.

All windows (`horizon`, `rolling.window`, transform windows) are **bar counts**, not calendar
months. Alternative-data feeds are external CSV files, not DSL builtins.

## BacktestParams (set `horizon` explicitly)

`horizon` (**defaults to 1 — set it explicitly**; the default is only a neutral fallback for the
immediate next-period return. The forward measurement window in bars; a **list** sweeps it as its own
`horizon` axis → a return response curve, e.g. `[1, 5, 10, 20]`), `direction`
(`longonly`|`shortonly`, default `longonly` — the sign of the measured return), `outcome`
(what a firing measures — see below; omitted = the target's pct return), `benchmark`
(`"market"`|`"cross_mean"` (basket only)|omitted — see below), `features` (see below). That is the
whole model. Returns are raw
`exit/entry - 1` unless a `benchmark` makes them excess returns.

**There is no sampling knob of any kind.** Every declared parameter × horizon cell is measured over
the WHOLE index and reported independently, so the DSL cannot express a partition of the data that
some cells see and others do not. The circular-shift rotation null likewise always uses every
non-identity shift, so its resolution is a property of the series length, never a choice.

**Keys that do not exist — a thesis carrying any of them is rejected (exit 3, `dsl_invalid`)**,
because `extra="forbid"` applies to every DSL model:

- `oos_fraction` / `selection_mode` — there is no in-sample/out-of-sample split: no
  holdout, no embargo, no cut timestamp, no tail and no sign test. Every cell is graded on
  full-sample evidence, and nothing seikan reports is an out-of-sample confirmation. There is no
  replacement key, because the thing they would parameterize — selecting a winner in-sample and
  confirming it on a reserved tail — is not something this engine does.
- `n_rotations` — the rotation null uses every non-identity shift; there is no cap to set, so its
  resolution is a property of the series length rather than a choice.
- `exit`, `fill_timing`, `init_cash`, `fees`, `slippage` — observer purity: no exit rule, one
  tradable anchor, no portfolio simulation, no cost model.

**Outcome (`params.outcome`)** — WHAT each firing measures:
- `{"series": "target" (default) | "<declared feed name>", "kind": "pct" (default) | "log" | "diff"}`.
  Omitting the field and writing `{}` are the SAME document (both canonicalize to the filled
  default and hash identically); an explicit `"outcome": null` refuses — it would be a second
  spelling of the same default, splitting one thesis into two identities.
- `kind: "diff"` measures the level CHANGE `b − a` in the series' own units — the honest form for
  yields/spreads/multiples, where a percent of a near-zero or sign-crossing level is meaningless
  ("the 10y falls 50bp" is `direction: "shortonly"` + `outcome: {"kind": "diff"}` on a yield
  target; `mean_ret` is then in level units, not a percent).
- `series: "<feed>"` measures a FEED's forward evolution instead of the target's price — "when the
  target dips, implied vol rises over the next 5 bars" is `outcome: {"series": "iv", "kind":
  "diff"}`. The feed must be declared in `data.external`; a window the feed can't cover censors as
  `exit_reason="no_outcome"`.
- `pct` and `log` are RATIO algebras and require **both endpoints strictly positive**: off a
  positive scale neither fails loudly (a percent change through zero returns a finite number with
  an INVERTED sign; `ln` of a negative/negative ratio is finite too), so a non-positive endpoint
  censors the observation as `exit_reason="no_outcome"` instead of minting a garbage return — which
  the per-cell `outcome_coverage` check then refuses. `diff` is untouched (a yield crossing zero is
  a real move, not a domain error). OHLCV targets cannot reach this; series targets and feed
  outcomes can.
- The anchor is always the next bar (`t+1 → t+1+h`) — no same-bar look-ahead, whatever the outcome.
  `benchmark: "market"` works with `pct` and `log` (the benchmark leg is measured in the OUTCOME'S
  OWN algebra, so a `log` outcome yields the true log-excess `ln(tgt ratio) − ln(bench ratio)`) but
  is **refused with `kind: "diff"`** at validation: a diff outcome is in the target's own level
  units and subtracting a benchmark RETURN from it is incommensurable however it is computed. The
  summary self-describes via `outcome` / `target_shape`.

**Benchmark (excess returns)** — at horizons of ~20+ bars a raw forward return mostly reflects
whether the MARKET went up, not the thesis; long-horizon theses should set `benchmark`:
- `"market"` — each observation becomes the EXCESS over the same-window return of the
  `benchmark` series: `(tgt[t+1+h]/tgt[t+1] − 1) − (bench[t+1+h]/bench[t+1] − 1)` (sign
  applied last; under `shortonly` you profit when the target *underperforms* the benchmark).
- `"cross_mean"` (**basket mode ONLY** — validation refuses it under conjunction) — each member's
  observation becomes the EXCESS over the basket's own same-window mean forward return, all
  declared members, self included, measured in the outcome's own algebra — the relative-value
  read "did this member beat the basket?". It takes no `benchmark` key (supplying one
  beside it is an error), and it works under `pct` and `log` (`diff` never reaches it — basket
  refuses `diff` outright). Its missingness is FAIL-CLOSED at whole-bar granularity: any member's
  leg non-finite at a bar censors the WHOLE bar's benchmark leg, so every member's firing there
  exits as `no_benchmark` — never a partial-basket mean, which would quietly demean the surviving
  members by a DIFFERENT basket while their rows escape the censoring ledger. Note the identity
  it implies — and its boundary: under `cross_mean` the POOLED BASELINE mean is ≈ 0 by
  construction (every member enters every eligible bar, and complete cross-sections demean to
  zero — an identity, not a finding), while the pooled CELL mean is NOT that identity: it holds
  only the members that FIRED, so it reads the firing subset's cross-sectional selection tilt.
When set, EVERY reported statistic (mean / hit rate / HAC / rotation null / buckets / PBO) is an
excess-return statistic, and the summary records `benchmark` / `benchmark_source`.

**Measurement** — every bar `entry` fires opens an OVERLAPPING observation; its return is the forward
return over `horizon` bars. Firing bars whose horizon runs past the data end are right-censored
(`exit_reason="open"`, `is_open=true`, excluded from the stats). There is no exit condition.

**Fill timing** — a firing bar `t`'s observation is always anchored at `open[t+1]`→`open[t+1+h]`
(the next bar's open — the only tradable convention; there is no same-bar-close mode).

**Conditional features** — `features`: `{name: <Series>}` snapshots an extra entry-time series per
observation (any **scalar-param** Series, including `external` feeds — declare them in `data.external`)
so each cell buckets its returns by it (the per-cell `conditional_buckets`),
testing whether the edge is regime-conditional. Defaults to built-in momentum (`ret_5`, `ret_20`)
+ realized-vol (`vol_14` = `rolling_agg(change(close, kind="log"), 14, "std")`) snapshots when
unset. When given, the dict must be NON-EMPTY — `{}` would behave exactly like the omitted field
while hashing differently, so omit the field entirely for the defaults. Features are scanned by the cross-node mode rule exactly as the entry is: a
cross-sectional feature snapshot requires `target_mode: "basket"`.

## Reading the result (you do the selecting)

`seikan run` measures **every declared parameter × horizon cell** on the full sample and
reports each one independently. It does **not** select, rank, or crown a winner: there is no best
cell, no headline scalar, no verdict, and no search-adjusted statistic anywhere in the report.
Choosing among the cells — and pricing the multiplicity of having looked at
`n_hypotheses_attempted` of them, across this run and every other run you made — is **your** job as
the calling agent. A stateless reporter cannot see your other runs, so it does not pretend to
correct for them.

The report is a FILE you nominate: `--report-out <path>` (always overwritten). A successful run
prints **nothing** on stdout — only an error ever emits a JSON envelope there — so read the report
off disk, not off the pipe. It is `report_schema_version` **5**, in a FIXED layer order (there are
no layering variants to branch on):

`seikan_version` / `report_schema_version` / `command` (the header every seikan JSON document
opens with, `command` = `"run"`) → `identity` (`name`, `dsl_hash`, per-key `data_digests` — each
`{path, column, sha256}`, keyed by the logical key the thesis declares, with `column` ALWAYS
present and `null` where that key bound none, because "no column was bound, so the file named its
own" is a fact about the run and not an absence to infer (two keys answered by ONE file share its
`sha256` — a property of the bytes — while keeping their own column) — the `thresholds` snapshot
actually used, `thresholds_canonical`, per-knob `thresholds_provenance`
— `default` | `env` | `cli`, the SOURCE, so a stricter-via-flag run is distinguishable from a
stricter-via-env one — and `environment`, the python/numpy/pandas/scipy/numba versions the
numbers were computed under, since a report reproducible only against an unstated numeric
stack is not fully reconstructible) → `data_report` → `outputs` → `summary` (verbatim) →
`gate` → `metric_roles`.

`outputs` names every file this run wrote, keyed in nomination order `report` → `trades` →
`root_series` → `entry_flags` and holding only the ones you asked for: `{"path": …}` for the report,
`{"path": …, "rows_written": N}` for each CSV. A report you are reading always contains its own
`report` entry, so the document says exactly which artifacts of the run exist beside it.

**The exit code is not a verdict.** Exit 0 means the run finished and every nominated output was
written — the report, when you asked for one, is complete — whatever every cell says. (2 = data
invalid, 3 = invalid request — a usage error, including a run that nominates no output at all, an
invalid thesis DSL or gate-threshold set, or an unusable nominated output path: empty, unwritable,
named by two flags at once, or naming one of your own input CSVs — nominate four DISTINCT paths,
none of them a file the thesis reads, 4 = internal. `seikan schema` → `exit_codes` is the
machine-readable list.)

### The trades CSV (`--trades-out`)

One row per recorded OBSERVATION — firing bar × target × declared horizon, the whole grid in one
file. Regroup rows on the leading param columns + `target` (there is no `cell_id` column).
Machine-readable contract: `seikan schema` → `trades_csv` — whose `derived_views` note records
that each cell's `episodes` ledger (and `episode_stats`) is a deterministic derivable function of
this CSV, which is why there is no `--episodes-out` flag: the in-report ledger truncates visibly
at its cap, while this CSV never truncates, so rebuild past-the-cap entries from here. The
columns:

| column | meaning |
|---|---|
| *swept axes* | one leading column per swept axis, in `summary.params` order — the row's cell identity |
| `target` | the target the row belongs to (regime member or basket member) |
| `entry_time` | ISO timestamp of the next-open ANCHOR bar `t+1` (the firing bar is `t`) |
| `exit_time` | ISO timestamp of the exit bar (clamped to the final bar when censored) |
| `entry_bar` | the FIRING bar's integer position — the JOIN KEY to the entry-flags CSV's row position; never join on timestamps (off by one: anchor vs firing) |
| `horizon` | the cell's declared measurement horizon `h` — ALWAYS present, right after the swept axes and before `target`, so a row names its window whether or not the horizon was swept; with the swept axes it is the row's cell identity |
| `entry_px` / `exit_px` | the measured value at anchor / exit (`exit_px` empty when censored) |
| `ret` | the signed forward outcome, denominated per `summary.outcome` (excess when benchmarked); empty on censored rows |
| `ret_raw` | the signed RAW outcome leg — equals `ret` when unbenchmarked; `ret == ret_raw − ret_bench` per row when benchmarked; empty on censored rows |
| `ret_bench` | the signed benchmark leg the excess subtraction consumed (the market series' same-window return, or the bar's basket mean under `cross_mean`); empty when unbenchmarked or censored |
| `pre_ret` | RAW drift INTO entry over the same `h`-bar window, sign-aligned — the leakage canary |
| `mae` | worst interim adverse mark, RAW path, ≤ 0; empty when censored |
| `mfe` | best interim favorable mark over the same window, RAW path, ≥ 0; empty when censored. A MARK, never an attainable exit — there is no exit rule |
| `bars_to_positive` / `bars_to_trough` | recovery / trough timing over the window; empty when censored |
| `exit_reason` | `horizon` (closed) · `open` (end-of-data censoring, structural) · `no_outcome` / `no_benchmark` (data holes — the checklist refuses them) |
| `is_open` | True iff censored; censored rows are excluded from every statistic |
| *features* | trailing entry-time feature snapshots at the FIRING bar (defaults `ret_5`, `ret_20`, `vol_14`) |

(There are no epoch-nanosecond twins of `entry_time`/`exit_time`: the ISO times are the record,
and int64 nanosecond counts silently round in IEEE-754 consumers.)

### The two panels you read together

- **`summary.cells`** — ONE entry per declared combo × horizon, in declaration order, **including
  combos and horizons that never fired** (an explicit zero/NaN record). `len(summary.cells) ==
  summary.n_hypotheses_attempted`, always. Each entry:
  - `cell_id` — a rendered label (`"zscore_window=20,horizon=21"`; swept axes first, horizon last).
    A convenience for humans: a cell's real identity is its `params` plus its **position** in the
    list, so never key off the label.
  - `params` — the parameter assignment, with `horizon` **always present** even when you did not
    sweep it.
  - `by_target` — per target: `n`, `n_nonoverlap` (non-overlapping episodes — non-overlap is not independence; compare to `n`
    to see the overlap inflation), `mean_ret`, `mean_ret_raw` / `mean_ret_bench` (the excess
    mean's own direction-signed legs over the same closed pool — `mean_ret ≈ mean_ret_raw −
    mean_ret_bench`, so "+3.2% against a +2.4% market" and "lost less than a falling market" are
    distinguishable; unbenchmarked, `mean_ret_raw` equals `mean_ret` and `mean_ret_bench` is
    null), `hit_rate`, `profit_factor` (Σwins/|Σlosses|; null — never an infinity — when either
    side is empty), `t_hac`, `hac_se`, `rot_p`,
    `concentration` (top-5% |return|-mass share), `boot` (the episode-bootstrap percentile CI —
    see the statistics section), `subperiods` (the pool's `n` / `mean_ret`
    over three equal-bar eras of the shared index — era visibility),
    `ret_quantiles` + `worst_ret` / `best_ret` (the closed pool's seven order statistics and its
    single worst and best observations), `mae_quantiles` / `mfe_quantiles` (the same seven points
    plus the subset's `mean` and `worst` / `best` over the RAW post-entry excursions, each block
    carrying its own `n` — see the statistics section) and `edge_ratio` (mean RAW MFE over
    |mean RAW MAE| — the ONE sanctioned excursion ratio, both legs RAW; see the statistics
    section). A target with no closed rows still gets a full entry with `n = 0`, null
    statistics and null evidence blocks (`boot` says why in its `reason`). In basket mode this
    panel REMAINS — as attribution, read by no check; the graded panel is `pooled` below.
  - `pooled` — **basket cells only** (absent — not null — on conjunction cells): the cell's ONE
    cross-target evidence pool over the concatenated (bar × member) closed rows in
    target-declaration order, mirroring `by_target[t]`'s shape so there is one mental model:
    `n` (== the sum of `by_target.n`), `n_nonoverlap` (the same greedy non-overlapping kernel over the
    concatenated entry bars — same-bar cross-member firings collapse to ONE non-overlapping window),
    `mean_ret`, `mean_ret_raw` / `mean_ret_bench` (the pooled attribution legs — under
    `cross_mean` the pooled `mean_ret` is the FIRING subset's cross-sectional selection tilt:
    each closed bar's FULL cross-section demeans to zero, but the pool holds only the members
    that fired, so ≈ 0 appears only when firings are basket-wide, and a selective signal's
    pooled mean IS its cross-sectional selection read; the LEGS still attribute —
    `mean_ret_bench` ≈ `mean_ret_raw` is the basket's own realized drift), `hit_rate`,
    `profit_factor`, `t_hac` /
    `hac_se` (the same event-time HAC on the pooled rows;
    same-bar pairs enter at Bartlett weight 1, so one market move seen through several members is
    one cluster), `rot_p` (a COMMON-SHIFT rotation null — one shift rotates every member's mask
    as a block), `concentration`, `member_share` (`by_target` — each member's share of the pooled
    |return| mass, a full decomposition and never a ranking — plus `max_member_share_abs`, the
    one read the checklist gates), `boot` (the episode bootstrap over the cross-member-merged
    episodes), `subperiods`, `ret_quantiles` / `worst_ret` / `best_ret`, `mae_quantiles` /
    `mfe_quantiles`, and `edge_ratio`. This is the panel the checklist grades in basket mode.
  - `episode_stats` — the cell's closed rows clustered into market EPISODES by merging overlapping
    `[entry, exit)` windows **across targets**, so one crisis seen through three targets is one
    episode. `max_cluster_share_abs` — the largest |return|-mass share held by ANY single
    cluster — is the one-episode detector the checklist reads; `largest_cluster_n`,
    `largest_cluster_share_abs` and `largest_cluster_start` describe the largest cluster BY
    ROW COUNT (its size, its mass share, its earliest entry) — a different "largest", so the
    two shares need not agree — and `mass_hhi` / `effective_n_clusters` are the ceiling's
    smooth companions. (In basket mode the same cross-target merge makes it the pooled episode
    read for free.)
  - `episodes` — the TIME-ORDERED episode ledger: the narrative companion to
    `episode_stats`, same frozen overlap merge, entries earliest first — never ranked by share —
    each `{start, end, n, mean_ret, share_abs}`, capped at 32 with mass-conserving truncation
    (`n_omitted` + `omitted_share_abs`; listed + omitted shares ≈ 1), and `n_total ==
    episode_stats.n_clusters` by construction. A count read off a truncated ledger is a floor.
  - `conditional_buckets` / `bucket_monotonicity` — per-feature qcut buckets (q=4) over the
    CELL's own closed rows — bucket EDGES per target, records aggregated by ordinal (labels
    `q1..qk` on multi-target cells; one pooled qcut over raw levels would conflate member
    LEVELS with time variation) — never across cells (see "Not in the report"). Refusals are
    explicit, never absent:
    `no_closed_observations` / `insufficient_observations` (below 20 valid rows) /
    `insufficient_distinct_values`. `bucket_monotonicity` carries a Spearman `rho` + `sign` per
    feature that bucketed with ≥ 3 populated buckets.
  - `feature_association` — per (feature × target): Spearman `rho` between the entry-time
    feature snapshot and the realized closed return, with its `n` and an explicit `reason`
    (`insufficient_observations` below 10 pairs; `no_rank_variation` on constant input).
    Per-target in BOTH modes, and deliberately WITHOUT a p-value — an overlap-inflated p is
    exactly the over-trustable number the doctrine forbids. "Associated in this sample", never
    "predicts".
  - `outcome_coverage` — the per-target censoring ledger over the cell's FULL rows (censored ones
    included): `n_attempted`, `n_closed`, and all four `exit_reasons` (`horizon` / `open` /
    `no_outcome` / `no_benchmark`), zeros reported, so `sum(exit_reasons) == n_attempted` is
    arithmetic you can re-check.
  - `signal_coverage` — per target `n_bars` + `n_undefined` (post-warmup bars where the entry
    condition was UNDECIDABLE because an input was missing). Keyed by combo upstream, so horizon
    siblings repeat the same numbers on purpose. Never sum across cells.
- **`gate.cells`** — index-aligned with `summary.cells` (`gate.cells[i]` grades
  `summary.cells[i]`). Each carries `cell_id`, `params`, `met`, and `checks` — five
  `{name, met, observed, threshold, detail}` records, always all five, never
  short-circuited. Alongside: `gate.policy_version` (3), `gate.n_cells`, `gate.n_met`, and
  `gate.run_checks` (three run-level records reported once).

A cell's `met` already includes the run-level checks — an unmet run-level check leaves every cell
unmet — so you read `gate.cells[i].met` and never have to AND the sections yourself.

### What `met` means, and what it does not

A cell reported `met` asserts exactly three things — the second and third read from the panel the
summary's `target_mode` stamp selects:

1. **Completeness** — every firing is accounted for, every decision bar was decidable, and every
   raw decision input was available. Nothing was silently deleted from the evidence. (Per target
   in BOTH modes: a hole in one basket member corrupts every member's cross-sectional reads.)
2. **Support** — the same three sealed floors either way: `n >= 30` closed observations,
   `n_nonoverlap >= 8` non-overlapping episodes, and a positive realized `mean_ret`. Under **conjunction**,
   per target — the weakest target decides (targets are your thesis's regime). Under **basket**,
   on the `pooled` panel — the members form ONE evidence pool, floors read the pooled panel and
   never a member's own; a thin member does not sink a basket cell, because the basket claim is
   about the pool, not about any name in it.
3. **Non-concentration** — one sealed ceiling (0.6). Under **conjunction**: no regime target's
   return mass, and no single merged cross-target episode cluster, exceeds it. Under **basket**:
   the pooled top share replaces the per-target layer, the episode-cluster ceiling stays ("not
   one crisis"), and `pooled.member_share.max_member_share_abs` joins them — the one-name-basket
   detector ("not one name"), same ceiling, no new knob. A one-episode "edge" refuses either way.

**It is NOT an inferential claim.** No significance is asserted, no positive expected return is
certified, and nothing in the checklist is a statistical test — `mean_ret > 0` is a sign read on
the realized sample. There is no holdout, no embargo, no tail and no out-of-sample confirmation, so
no cell's `met` can be quoted as one. Treat `met` as "this cell's evidence is complete and
substantial enough to be worth your attention", never as "this edge is real".

Correspondingly, **an unmet cell is not a rejected thesis** — read its `detail` strings. An unmet
`support` on a sparse `first_true` pool means the pool is thin, which you already knew and which is
honest, not a defect. An unmet `outcome_coverage` or `signal_coverage` means your DATA has a hole
under the firings and the numbers cannot be trusted until you repair it. Either way the cell's
complete statistics are still in `summary.cells[i]` — a checklist result never filters the summary.

### The three run-level checks

- `evidence_complete` — the summary carries the stamps this checklist was built against
  (`statistics_version` matching this build's, `gate_evidence_basis == "full_sample"`, an
  EXPLICIT `outcome` stamp —
  the `{series, kind}` dict the runner always writes; a null stamp refuses — a
  readable `target_mode` stamp in `{conjunction, basket}` — the stamp SELECTS the rubric every
  cross-target read is graded under, so a missing or garbage stamp refuses fail-closed rather
  than being graded under an assumed mode, and a basket stamp over fewer than two targets or a
  `diff` outcome refuses as drifted input (validation refuses both upstream; the gate re-refuses,
  never trusts) — string-typed targets, countable `n_bars`/`n_hypotheses_attempted`), a
  string-keyed `sources` panel covering the targets EXACTLY, and a `cells` panel holding
  EXACTLY the declared grid. A report short of `n_hypotheses_attempted` cells has dropped
  hypotheses from the search burden it declares, which is drifted input, not evidence.
- `source_coverage` — fail-closed availability over the RAW decision inputs (`summary.sources`):
  per target `n_missing == 0` for every leaf the entry tree reads (a price field, an external feed,
  a `days_since` age) over the evaluated interval, after its own first available bar. This catches
  what the per-cell `signal_coverage` structurally cannot see: an `and`/`or` operand hole settled by
  a decisive sibling (`F ∧ U = F` leaves the root DEFINED), and a hole a NaN-skipping recursive
  transform (`ema`, expanding `rolling_agg`, `bars_since_extremum`) carried its state across before
  emitting a finite value. **A feed that merely starts late is WARMUP, not a hole** — its
  `first_available` is reported as evidence and never refuses; ordinary sparse stamping (a weekly
  feed on daily bars) is available throughout after its first stamp.
- `search_cap` — `n_hypotheses_attempted` (the DECLARED grid; non-firing combos cannot shrink it)
  ≤ 64. This is the only multiplicity input the policy carries, and a grid above the cap is refused
  at DSL validation (exit 3) before any data is read.

### The five per-cell checks

- `cell_evidence` — the cell's panels are present, string-keyed, cover your targets exactly, and
  RECONCILE with each other (`by_target.n == outcome_coverage.n_closed`, `n_nonoverlap <= n`,
  `episode_stats.n` == the per-target total, `signal_coverage.n_bars == summary.n_bars`). Basket
  cells must ADDITIONALLY carry the `pooled` dict their own rubric grades, reconciling with the
  member panels: `pooled.n ==` the sum of `by_target.n` (one pool, fully attributed),
  `pooled.n_nonoverlap <= pooled.n`, `pooled.n_nonoverlap <= n_bars` (the greedy count collapses same-bar
  cross-member firings, so it is bounded by the bar clock), and `pooled.n <= n_bars ×
  len(targets)` (each member fires at most once per bar). A `pooled` key on a conjunction cell
  REFUSES: the engine writes `pooled` only in basket mode, so a conjunction-stamped report
  carrying one can only be a basket report whose mode stamp was rewritten — the checklist reads
  it as drifted input rather than quietly grading the friendlier rubric (a conjunction stamp
  beside `benchmark: "cross_mean"` refuses the same way). An internally impossible cell is
  drifted input, not something to grade.
- `outcome_coverage` — per target `no_outcome == 0` and `no_benchmark == 0`. A hole that deletes
  outcomes can hide adverse results, and missing-at-random is never assumed. **`open` is ALLOWED at
  any count**: with no holdout there is no embargo and no tail, so a forward window running past
  the last bar is structural end-of-data right-censoring that every cell near the index end must
  exhibit. An in-bounds NaN leg is never `open` — it classifies as `no_outcome`/`no_benchmark` and
  refuses.
- `signal_coverage` — per target `n_undefined == 0`. A missing entry-condition input does not
  censor an outcome, it suppresses the FIRING, which the outcome ledger structurally cannot see;
  without this check, deleting the inputs under adverse firings would improve a cell unseen.
  (Both coverage checks, like `source_coverage`, stay PER-TARGET in BOTH modes: a hole in one
  basket member corrupts every member's ranks, so per-member fail-closed is structurally
  required in a basket too.)
- `support` — the floors above (`thesis_min_trades`, `thesis_min_n_nonoverlap`, `mean_ret > 0`),
  dispatched by the `target_mode` stamp. Conjunction's threshold reads "per target:
  n>=30 & n_nonoverlap>=8 & mean_ret>0 (the weakest target decides; a descriptive floor, NOT a
  significance claim)"; basket's reads "pooled: n>=30 & n_nonoverlap>=8 & mean_ret>0 (basket: the
  members form ONE evidence pool — floors read the pooled panel, never per member; a descriptive
  floor, NOT a significance claim)". A basket cell without a usable `pooled` panel goes unmet —
  basket support is a claim about the pool, and a cell without the pooled panel carries no
  gradable support. Deliberately NOT inferential either way: no t-statistic and no p-value
  gates, because `rot_p` and `t_hac` are known anti-conservative and stay evidence-only.
- `concentration` — one sealed ceiling, dispatched by the `target_mode` stamp. Conjunction:
  every target's `top_share_abs` AND the cell's `episode_stats.max_cluster_share_abs` under the
  ceiling. Basket: the pooled read REPLACES the per-target layer (`pooled.concentration
  .top_share_abs` — "the basket's edge is one episode" when breached), the episode-cluster
  ceiling stays, and `pooled.member_share.max_member_share_abs` joins them ("one member carries
  the basket's mass; the basket claim is mostly one name"). A `diff`-outcome multi-target run
  refuses the cross-target mass read as incommensurable (level units from different series are
  not mass-comparable), as does a missing, null, or unreadable outcome stamp — retained in
  basket as defense-in-depth even though `evidence_complete` already refuses basket+diff.

**The canonical checklist is the floor**: every threshold constructs only at its default or
stricter (`--min-trades 40` is fine, a loosened value is exit 3, `thresholds_invalid`), and an
unknown `SEIKAN_*` env var is a hard error. `identity.thresholds_canonical` stamps whether the
knobs were the canonical set. The full check semantics are in `seikan schema` → `gate_contract`;
the report/summary field dictionary is `seikan schema` → `report_fields`, and the trades CSV's
column contract is `seikan schema` → `trades_csv`.

### The statistics — all of them evidence-only

Overlapping forward returns are autocorrelated, so no classical iid t-test is emitted at all —
it would be invalid by construction. Every statistic below is NOMINAL and per-cell: none is corrected for the size of your grid, and
**no check reads any of them**. They inform your judgment; they certify nothing. The same
caveats travel IN every report as the compact `metric_roles.caveats` map — one sentence per
number a reader is likely to over-trust — so an agent holding a report does not need this guide
open to quote a statistic honestly.

- `summary.baseline` — the run-level UNCONDITIONAL base rate, one entry per
  horizon in declaration order, per target: `{n_anchor_bars, n_eligible, exclusions, mean_ret,
  std_ret, hit_rate, ret_quantiles, worst_ret, best_ret}` over EVERY fillable anchor bar, under
  the same next-open anchor, algebra, benchmark leg and direction sign as the cells. **This is
  the number every conditional mean must be read against**: "+0.9% on firing bars vs +0.4% on
  all bars" is a finding; "+0.9%" alone is the market wearing a costume. In-sample and
  unconditional; the `exclusions` ledger (the exit-reason vocabulary minus `horizon`) is the
  honesty channel, with `n_eligible + Σexclusions == n_anchor_bars` re-checkable arithmetic; an
  empty pool is all-null, never zeros; and there is deliberately NO cell-vs-baseline uplift
  field anywhere — the comparison is yours. In basket mode each horizon entry also carries a
  `pooled` row over the concatenated (bar × member) eligible observations, whose counts are the
  per-target sums — and under `benchmark: "cross_mean"` that pooled baseline mean is ≈ 0 BY
  CONSTRUCTION (an identity, not a finding).
- `summary.cross_breadth` (basket only; `[]` elsewhere) — the effective-universe ledger, one
  entry per cross node × combo: `{node, params, min_valid, n_bars, n_bars_evaluated,
  n_bars_below_full, k_min, k_median, k_max, first_full_bar}` over the per-bar count `k` of
  finite member inputs the cross kernels reduce over. Member warmup legally thins `k` (a late
  start is warmup, not a hole), so `k_min < len(targets)` states coverage, not a defect — what
  the panel makes visible is the effective universe drifting through time: early bars ranked
  among fewer members than late ones. Entries repeat across combos that do not move the node's
  input; they are never summed, and no check reads any of it.
- `rot_p` (per cell, in `summary.cells[i].by_target[t]`) — the
  circular-shift rotation null: the forward-return series is fixed and only the firing mask
  rotates, preserving the firing count, temporal clustering and overlap exactly. It answers "was
  THIS mask's timing informative against its own null?" and nothing else — it is **not** a survival
  probability over your search. Beside it, `rot_n_null` is the number of DEFINED shifts the
  cell's null was formed over, so the cell's own smallest attainable value is
  `1/(1+rot_n_null)` (≥ `summary.rotation.p_resolution` = `1/(1+n_shifts)`, the run-level floor
  over a fully defined null); a `rot_p` sitting at its floor means "no shift beat the
  observation", not "p ≈ 0". KNOWN CAVEAT: it assumes shift-exchangeability and over-certifies when volatility
  clusters in the same stretches your signal fires.
- `t_hac` / `hac_se` — the EVENT-TIME Newey-West t: Bartlett weights on actual entry-bar distances,
  so a sparse pool degrades to the iid SE instead of a collapsed one. df = `n_nonoverlap` − 1; the derived
  p is not emitted, re-derive it if you need one. KNOWN CAVEAT: anti-conservative on heavily
  overlapping pools (the taper understates the long-run variance).
- `n_nonoverlap` — the greedy non-overlapping observation count. Compare to `n` to see how much of your
  sample is really one event.
- The basket's `pooled` reliability reads (basket cells only) — same estimators,
  pooled rows, three things to hold onto when quoting them: (1) `pooled.n` counts (bar × member)
  rows, so one market move smears across members AS WELL AS across ~h overlapping horizons —
  pooled `n` overstates the non-overlapping information TWICE; quote `pooled.n_nonoverlap` beside it, never
  a member count summed as if independent. (2) Same-bar firings across members are ONE market
  event: the greedy `n_nonoverlap` kernel collapses them to one window, and the pooled HAC prices their
  cross terms at Bartlett weight 1 — full covariance, never independent evidence. (3)
  `pooled.member_share.by_target` is ATTRIBUTION — a full decomposition of the pooled |return|
  mass, never a ranking, and read by no check (only `max_member_share_abs` gates); a 2-member
  basket reads structurally elevated, its larger member always carrying ≥ 0.5. The pooled
  `rot_p` is a COMMON-SHIFT null — one shift rotates every member's mask as a block, preserving
  per-member counts AND the per-bar cross-sectional pattern a rank signal fixes — and it
  inherits every per-target `rot_p` caveat; never quote it as significance.
- `by_target[t].boot` — the EPISODE-BOOTSTRAP percentile CI for the pool mean:
  the cell's closed observations cluster into overlap-connected `[t, t+h)` episodes (per target),
  2000 draws resample the episodes with replacement, and the block carries `{method, ci_level,
  n_boot, n_episodes, ci_lo, ci_hi, boot_se, reason}`. This is the dependence-robust counterweight
  to `t_hac`/`rot_p`: within-episode dependence is preserved exactly, so the interval is as wide
  as the EPISODE count warrants — a clustered pool with `n = 80` but 6 episodes gets an honestly
  wide CI. Deterministic (content-seeded); fewer than 5 episodes → null fields with a `reason`
  instead of a degenerate interval. KNOWN CAVEAT: it assumes episode INDEPENDENCE — adjacent
  episodes still correlate through slow volatility regimes — so it is less anti-conservative than
  `t_hac`, not calibrated.
- `by_target[t].subperiods` — the pool's `{start, end, n, mean_ret}` over three
  equal-bar eras of the shared index (entry-bar assignment, no purging). Era VISIBILITY, not a
  holdout: it shows whether an edge lived in one era, and nothing selects on it.
- `by_target[t].ret_quantiles` + `worst_ret` / `best_ret` — the closed pool's
  `{p05, p10, p25, p50, p75, p90, p95}` (linear interpolation; `p05` doubles as the historical
  VaR(5%) that `cvar_5` tails off) and its single worst and best observations. Read `p50` AGAINST `mean_ret`: a positive mean sitting over
  a negative
  median is a pool a few spikes carried, not a typical outcome — and `concentration` does not catch
  it, because that check reads |return| MASS in one EPISODE while a mild right skew spreads its
  mass thinly and still drags the mean off the median. KNOWN CAVEAT: below n ≈ 20 the outer points
  interpolate between one or two observations, and the observations OVERLAP (one market move
  smeared across ~h rows), so these describe what the pool HELD, not independent-draw quantile
  estimates. Under a benchmark they are excess returns, like `ret` itself. No `n` of their own —
  the pool is `by_target[t].n`.
- `by_target[t].mean_ret_raw` / `mean_ret_bench` — the excess mean's own direction-signed legs
  over the SAME closed pool (`mean_ret ≈ mean_ret_raw − mean_ret_bench`; the trades CSV carries
  the per-row twins `ret_raw` / `ret_bench`). Attribution, never two extra hypotheses: "+0.8%
  excess because the target made +3.2% against a +2.4% market" and "+0.8% because the target lost
  less than a falling market" are different stories the excess mean alone cannot tell apart.
  Unbenchmarked, `mean_ret_raw` equals `mean_ret` and `mean_ret_bench` is null.
- `by_target[t].profit_factor` — gross win mass over gross loss mass (`Σwins / |Σlosses|`) of the
  closed pool: the MASS-weighted asymmetry beside `win_loss_ratio`'s average-weighted one, and
  derivable from it (≈ `(n_wins/n_losses) × win_loss_ratio`). Zero returns join neither side;
  either side empty is null, never an infinity. Overlap smears one move into both sums, and no
  equity curve exists — this is not a realizable P&L ratio.
- `by_target[t].mae_quantiles` / `mfe_quantiles` — the same seven points plus the subset's `mean`,
  `worst` / `best` and the block's OWN `n`, over the per-trade post-entry excursions on
  the full H/L of `[fill, fill+h−1]` plus the exit open (the exit bar's own high/low print
  after the exit): how deep the position ran against itself (`mae ≤ 0`) and how far the interim
  gain reached (`mfe ≥ 0`) before the horizon closed — the path read, aggregated, without opening
  `--trades-out`. Their `n` can sit BELOW `by_target[t].n`: a hole anywhere in the
  excursion window censors `mae`/`mfe` on a row whose `ret` still closed. KNOWN CAVEATS: both are
  RAW path, so under a benchmark `ret` is EXCESS while these are not — `ret` is not commensurable
  with either, and no ratio of `ret` against an excursion means anything (the sanctioned
  `edge_ratio` below divides the two RAW excursion means by EACH OTHER, which is why it alone
  survives a benchmark); overlapping windows share one trough
  and one peak, so a single crash sets the excursion of ~h neighbouring rows and the tail
  percentiles are not independent events; and an MFE is a MARK, never an attainable exit — this
  engine has no exit rule, so reading "it was up 8% at one point" as a foregone gain assumes an
  exit policy nothing in this report measured. Series-shaped and feed outcomes have no true
  intrabar range (open=high=low=close), so their excursions understate.
- `by_target[t].edge_ratio` — `mean(mfe pool) / |mean(mae pool)|`, the one sanctioned ratio
  BETWEEN the excursion pools: above 1, the typical interim gain outran the typical interim pain —
  the measured input to a stop-loss judgment, which stays yours. Deliberately UNNORMALIZED (no
  volatility scaling), so it compares nothing across instruments; each leg's mean covers its own
  finite subset; null when either pool is empty or the adverse mean is exactly zero.
- `summary.bar_spacing` — `{min,median,max}_seconds` between consecutive bars:
  the clock geometry every horizon-in-bars is denominated in, so "horizon 21" can be translated
  into calendar language without guessing.
- `summary.pbo` — ONE nested block `{pbo, reason, n_splits, n_splits_attempted,
  n_candidates_min, n_combos, n_combos_scoreable, n_combos_declared, blocks, lambda_mean,
  oos_degradation_slope, oos_degradation_slope_reason, prob_oos_loss}`; there are no flat
  top-level `pbo_*` keys. `n_splits_attempted` is the C(S, S/2) the partition offered against
  the `n_splits` actually scored (a split whose train winner has no finite test score is
  skipped) and `n_candidates_min` the smallest finite candidate population any scored split
  ranked over — canonical CSCV assumes a fixed count, and this is how far the block-local
  thinning departed from it; `oos_degradation_slope_reason` is null, or
  `degenerate_train_scores` when the slope is null because the train scores carried no spread.
  `summary.pbo.pbo` is the CSCV probability of backtest overfitting over every symmetric block
  train/test split (block-local windows pre-purged; adaptive S = 8 → 6 → 4). It ranges over the
  **scoreable** grid — only combos that fired with at least one closed observation, with
  byte-identical clones collapsed to one candidate; the ledger
  `n_combos <= n_combos_scoreable <= n_combos_declared` says how far short of the declared grid
  the scored population fell. It is attached to no
  cell: it is a property of the **search space you are about to select from** — "if I pick the best
  cell off this grid, how often would that pick fail to travel to data it never saw?" — and it is
  the single most relevant number to read *before* you select. Its split score mirrors the
  statistic a caller of the declared mode selects on: under conjunction each train-winner is
  scored by its WEAKEST target's per-observation Sharpe (the weakest-target rule the regime
  implies), under basket by the POOLED per-observation Sharpe over the concatenated member
  observations — no min, because a basket caller selects on the pooled read.
- A sweep-axis read — is an effect a plateau across a swept axis or a lone spike, the honest
  analogue of family-wise machinery this engine does not carry — is yours to form from
  `cells[*].params` against `cells[*].by_target[t].mean_ret`, PER HORIZON. There is no rollup
  and no breakdown table: a mean of per-cell means over different horizons is a number in no
  unit that moves with grid composition. A per-observation Sharpe is `mean_ret / std_ret` and a
  firing rate `outcome_coverage[t].n_attempted / n_bars`, both derivable from the panel and
  neither emitted.
- `cells[i].conditional_buckets` / `bucket_monotonicity` (PER CELL; there is no run-level pooled
  pair — see "Not in the report") — returns bucketed by each entry-time
  feature over the cell's own closed rows, with a Spearman `rho` + `sign` per feature (alphalens
  mean-return-by-quantile in one number): does the edge strengthen monotonically with the
  feature? Pooled across the cell's TARGETS — which in basket mode already IS the pooled
  conditioning read — but never across cells, and overlap-inflated like every trades-pool
  statistic: "associated in this sample" is the entire claim a bucket pattern supports. Do not
  pool buckets back across cells yourself: a pooled qcut conditions on grid composition.
- `cells[i].episodes` — the time-ordered episode ledger under the shares the
  `concentration` check reads: which calendar episodes the cell's edge actually was, earliest
  first, bounded at 32 with visible mass-conserving truncation. A count read off a truncated
  ledger is a floor, and `n_total == episode_stats.n_clusters` is a reconciliation you can
  re-check.
- `cells[i].feature_association` — per (feature × target) Spearman `rho` + `n`
  with explicit refusal reasons and deliberately no p-value. Per-target in BOTH modes: a pooled
  cross-member rank correlation would conflate level differences between members with variation
  through time — an attribution artifact wearing an association's clothes.
- Pool shape per cell: `hit_rate`, `win_loss_ratio`, `skewness` — how badly a positive `mean_ret`
  is being carried by a few observations. (For the one-episode read proper, use the cell's
  `by_target[t].concentration.top_share_abs` and `episode_stats.max_cluster_share_abs`, which the
  checklist itself gates on.)
- Run-level provenance you should record: `summary.n_bars`, `index_start`/`index_end`,
  `bar_spacing`, `n_hypotheses_attempted`, `direction`, `benchmark`/`benchmark_source`, `outcome`
  (ALWAYS the explicit `{series, kind, units}` dict — `units` is `fraction`
  for pct, `log` for log, `level_diff` for diff, and it is the denomination of every return-valued
  number in the report), `target_shape`, `target_mode` (ALWAYS stamped —
  which target semantics produced every cross-target read; say which mode a number came from
  when you quote it),
  plus `identity.dsl_hash` and per-key `identity.data_digests`. **This is the bookkeeping that
  makes your own multiplicity discipline possible**: every distinct exam is visible, so you can
  count how many you ran over the same data before you believe any of them.

**Not in the report, do not look for them** (and do not reconstruct them client-side from the
per-cell numbers): `fw_p`, `p_fdr`, `p_rw`/`romano_wolf`, `mean_ret_deflated`/`deflation`,
`best_cell`, `best_cell_by_target`, `binding_target`, `psr`/`dsr`/`sr`, `edge_stability`,
`path_risk`, `cost_robustness`, `lag_sensitivity`, `horizon_coherence`, `pre_entry_drift`,
`null_p`, `in_sample*` / `out_of_sample*`, `oos_fraction`, `oos_cut_time`, `oos_mean`,
`selection_mode` (note that `oos_degradation_slope` and `prob_oos_loss` DO exist — they are
CSCV/PBO companions naming that method's own internal block splits, nothing to do with a holdout,
and they ride INSIDE the nested `summary.pbo` block, never as flat top-level keys), every
pooled top-level headline (`mean_ret`, `n`,
`hit_rate`, `exit_reasons`, …), the retired grid breakdown and its rollups (`stats_table`,
`by_target`, `by_param`, `n_stats_rows` — a fired-pools-only duplicate of `cells[].by_target`
whose rollups averaged per-cell means across horizons), the classical iid pair `t_iid`/`p_iid`
(invalid by construction on overlapping pools), the derivable `sharpe`/`firing_rate`/`median_ret`
(`mean_ret/std_ret`, `n_attempted/n_bars`, `ret_quantiles.p50`), and any run-level pooled
`summary.conditional_buckets` / `bucket_monotonicity`: conditioning is per-cell, because a
pooled qcut re-enters the same bar once per combo × horizon, so adding a cell would change every
other cell's "conditioning" numbers — grid-composition-dependent conditioning is dishonest, not
merely redundant. Pooled views stay rebuildable from `--trades-out`, by you, on your own
responsibility. A search correction computed inside one run only sees that run's
grid and understates a caller who searched across runs; the multiplicity is yours, priced
against `n_hypotheses_attempted`.

Targets relate the way `target_mode` declares, and in NEITHER mode are they a search axis. Under
**conjunction** they are your thesis's regime: the engine forms no cross-target statistic and
takes no cross-target correction, it reports each target side by side and lets the weakest speak
for itself — wherever a per-cell rule applies (`support`, `concentration`), one target short of
the rule leaves the cell unmet; drop or replace the target, don't cherry-pick the winners. Under
**basket** they are ONE cross-section: the engine forms exactly the cross-target statistics the
declared `pooled` block carries and grades that pool — and it still selects among members NEVER.
`by_target` is attribution, not a per-member verdict ("strong in NVDA, weak in AMD" is not a
sentence a basket run can support), and a re-run with a reshuffled member set is a NEW thesis
you must count against your own search budget, not a tuning knob. In this guide "basket" is
exclusively the mode's name.

## Example — deep-drawdown rebound with valuation cheapness, forward-return response curve

```jsonc
{
  "name": "deep-dd-cheap-pe-rebound",
  "data": {
    "targets": ["NVDA"],
    "external": { "pe": { "lag": 1 } }
  },
  "entry": {
    "type": "first_true",
    "condition": {
      "type": "and",
      "conditions": [
        { "type": "threshold",
          "left":  { "type": "drawdown" },
          "op": "<",
          "right": { "type": "constant", "value": -0.30 } },
        { "type": "threshold",
          "left":  { "type": "bars_since_extremum", "extremum": "max" },
          "op": ">=",
          "right": { "type": "constant", "value": 60 } },
        { "type": "threshold",
          "left":  { "type": "runup" },
          "op": ">=",
          "right": { "type": "constant", "value": 0.05 } },
        { "type": "threshold",
          "left":  { "type": "percentile",
                     "input": { "type": "external", "name": "pe" },
                     "window": 252 },
          "op": "<=",
          "right": { "type": "constant", "value": 0.20 } }
      ]
    }
  },
  "params": {
    "direction": "longonly",
    "horizon": [21, 63, 126],
    "benchmark": "market",
    "features": { "pe_level": { "type": "external", "name": "pe" } }
  }
}
```

Three keys — `NVDA`, `pe`, and the `benchmark` that `"benchmark": "market"` conscripts — so three
`--data` pairs, plus one `--column` because the fundamentals vendor ships every ratio in one file:

```
seikan run deep-dd.json --report-out report.json \
  --data NVDA=data/nvda_ohlcv.csv \
  --data pe=data/nvda_fundamentals.csv --column pe=pe_ratio \
  --data benchmark=data/spy_ohlcv.csv
```

## Example — end-of-bull risk alarm (shortonly, synthetic external-feed composite)

```jsonc
{
  "name": "eob-iv-skew-leverage-heat-alarm",
  "data": {
    "targets": ["SPY"],
    "external": {
      "iv_skew": { "lag": 0 },
      "leverage": { "lag": 1 },
      "heat": { "lag": 0 }
    }
  },
  "entry": {
    "type": "first_true",
    "condition": {
      "type": "threshold",
      "left": {
        "type": "binary_op",
        "op": "+",
        "left": {
          "type": "binary_op",
          "op": "+",
          "left": {
            "type": "zscore",
            "input": { "type": "external", "name": "iv_skew" },
            "window": 63
          },
          "right": {
            "type": "zscore",
            "input": { "type": "external", "name": "leverage" },
            "window": 63
          }
        },
        "right": {
          "type": "zscore",
          "input": { "type": "external", "name": "heat" },
          "window": 63
        }
      },
      "op": ">",
      "right": { "type": "constant", "value": 3.0, "name": "alarm_score" }
    }
  },
  "params": {
    "direction": "shortonly",
    "horizon": [5, 21, 63],
    "benchmark": "market"
  }
}
```

Here all three warning feeds live in ONE sentiment panel under three headers — the multi-column
case, stated where the file is named:

```
seikan run eob-alarm.json --report-out report.json \
  --data SPY=data/spy_ohlcv.csv \
  --data iv_skew=data/sentiment_panel.csv  --column iv_skew=reverse_iv_skew \
  --data leverage=data/sentiment_panel.csv --column leverage=long_leverage \
  --data heat=data/sentiment_panel.csv     --column heat=opinion_heat \
  --data benchmark=data/acwi_ohlcv.csv
```

Three keys answered by one file, each reading its own column: the digests panel records all three
with the same `path` and `sha256` and three different `column` values, which is exactly what
happened. Note the benchmark is a BROADER index than the target — benchmarking SPY against SPY
would make every excess return identically zero.

## Example — multi-factor basket: top momentum quintile vs the basket mean

A basket thesis about the RANKING RULE: "entering the top 21-bar-momentum quintile of this group
precedes outperformance of the group". Note the `first_true` wrap — a bare `cross_rank(...) >=
0.8` fires on nearly every bar, because SOME member is always in the top quintile, and an
always-on basket signal chains into one merged episode cluster that leaves `concentration` unmet
in every cell (see "Target modes" above). Firing on ENTERING the quintile, with a cooldown, is the
honest measurement.

```jsonc
{
  "name": "semis-21d-momentum-top-quintile",
  "target_mode": "basket",
  "data": { "targets": ["NVDA", "AMD", "AVGO", "TSM", "MU"] },
  "entry": {
    "type": "first_true",
    "cooldown": 10,
    "condition": {
      "type": "threshold",
      "left": {
        "type": "cross_rank",
        "input": { "type": "change", "input": { "type": "field" }, "periods": 21 }
      },
      "op": ">=",
      "right": { "type": "constant", "value": 0.8 }
    }
  },
  "params": {
    "direction": "longonly",
    "horizon": [5, 21],
    "benchmark": "cross_mean"
  }
}
```

This declares a two-cell grid (one combo × two horizons). Each cell carries the graded `pooled`
block beside the per-member `by_target` attribution; `benchmark: "cross_mean"` makes every
return "excess over the basket", so the POOLED baseline mean sits at ≈ 0 by construction and the
per-target baseline rows show which members carried the group. Read the pooled `n_nonoverlap` against
the pooled `n` — five members firing on the same bar are one market event, not five.

## `seikan describe` — market context without a thesis

`seikan describe <files...> [--shape {ohlcv,series}] [--windows N,N,...] [--pretty]` profiles
data files and **measures nothing**: it is a pure observer of FILES the way `run` is of THESES.
Use it to source a market-context paragraph or a daily note — where the price sits in its range,
what realized dispersion has done, how deep the drawdown from the file's high runs — without
hand-computing context (a quality hazard) and without opening a single forward observation.
It runs no entry condition, grades no checklist, supports no thesis and clears nothing. **The
moment a question pairs today's description with what FOLLOWED, it is a thesis, and theses go
through `seikan run`** — in basket mode when the question is cross-sectional.

Mechanics: files are profiled in ARGUMENT ORDER, never sorted. `--windows` is one
comma-separated list of trailing windows in BARS (default `1,5,21,63,126,252`, at most 16; a bad
list is exit 3 `usage`). There is deliberately no column flag here, and none on `check-data`
either: both profile the FILE rather than a binding, so neither ever has to ask which column you
meant — an OHLCV file profiles `close` (the full final bar rides `last_bar`), a series-shaped file
profiles EVERY value column, in file order. Choosing the column a MEASUREMENT reads is the
invocation's act and has its own spelling there (`seikan run --column KEY=COL`), against keys a
thesis declared; a description answers to no thesis and so has no keys to bind. Exit codes: 0 all
files admitted ·
2 any refused (the document is STILL emitted, with a `{path, sha256, ok: false, reason}` stub
per refused file) · 3 usage · 4 internal. One JSON document on stdout — unlike `run`, whose
outputs are files — in the fixed order `seikan_version` → `report_schema_version` → `command` →
`data_report` → `profiles` → `describe_roles`. Admission is the SAME strict read `check-data`
performs, so `data_report` is byte-equal to check-data's over the same files.

Each admitted profile: identity + geometry in the run summary's own vocabulary (`path`,
`sha256`, `shape`, `n_bars`, `index_start`/`index_end`, `bar_spacing`), `last_bar` (the final
row verbatim, NaN as null, never back-filled), and per series: `changes` (the w-bar endpoint
change in ALL THREE algebras with domain-nulls — `pct`/`log` refuse non-positive endpoints
(`ratio_reason: "non_positive_endpoint"`), a NaN endpoint refuses as `endpoint_missing` and is
NEVER repaired by walking back to the previous finite value), `dispersion` (ddof=1 std of 1-bar
changes, per bar, NEVER annualized), `range_position` (trailing high/low with timestamps —
most-recent tie rule — distance from each, and `percentile_rank` within the window; requires
the full window, `insufficient_bars` otherwise, never silently shortened), `full_sample`
(whole-file extremes plus `drawdown_pct = last/high − 1` and `runup_pct`, with diff twins),
`missingness` (pure counts), and — OHLCV with volume only — `volume` (last, trailing mean,
`last_to_mean`; otherwise null).

How to read it honestly (the `describe_roles.caveats` travel in the document): `percentile_rank`
is position within the window's OWN range, NOT valuation — a trending series sits at its extreme
by construction; `dispersion` is per-bar, and any √t scaling is YOUR cadence assumption;
`last_to_mean` carries no "unusual" flag — what counts as elevated is your judgment; drawdown is
a property of the file's EXTENT — extend or trim the file and it moves; windows are BARS, never
days — `bar_spacing` states the clock, and translating "21 bars" into calendar language is your
act. Date anything you quote from it to the data's `index_end`.

What it will never contain: no per-bar arrays (output size is independent of `n_bars` — the
bounded-output invariant), no forward returns, no signals, no verdicts, no "oversold" /
"stretched" / "attractive" in any spelling, no ranking of the files it was handed, and no field
a thesis could cite as evidence. It describes what IS; a measurement states what FOLLOWED.
