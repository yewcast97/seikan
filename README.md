# seikan

A **stateless CLI forward-return event-study engine and per-hypothesis reporter** for
trading/investment theses, built to be driven by other agents (or humans). You bring a thesis
DSL (JSON) and time series (strict CSV); seikan runs an observer-pure forward-return event
study — entry condition + measurement horizon, no exit rule, no portfolio simulation — over
**every parameter × horizon cell your sweep declares**, and writes one complete JSON report.

## What makes it different

- **The full grid, always.** Every declared hypothesis is measured on the full sample and
  reported — including combos that never fired. `len(summary.cells) == n_hypotheses_attempted`
  holds by construction, so a surviving cell can never look inevitable by its siblings' absence.
- **No winner.** seikan does not select, rank, or crown a best cell: no headline scalar, no
  verdict, no search-adjusted statistic. Selection — and the multiplicity cost of having
  looked — stays with the caller, priced against the stamped grid size.
- **A checklist, not a test.** Each cell gets a completeness / support / concentration checklist
  with no significance claim and no positive-expected-return certification; `met` states that a
  neutral threshold comparison held, and the judgement stays the caller's.
- **Observer purity, honestly scoped.** The engine enforces its as-of convention computationally
  (feed timestamps are availability times; `lag` models publication delay) and is explicit about
  what it cannot verify: that your timestamps, vintages and universe are honest arrives with the
  data and stays the caller's burden.
- **Fail-closed ledgers on both sides of every firing.** A missing outcome refuses; undecidable
  decision bars and unavailable raw inputs are counted and refuse — deleting data can only ever
  leave a cell unmet, never improve it.
- **Overlap-honest evidence.** Event-time HAC, a rotation null, an episode bootstrap, CSCV PBO,
  episode clustering and an episode-deduplicated twin view ride every cell as evidence — each
  tagged in the report's machine-readable `metric_roles.metric_classes` with a per-metric caveat —
  and none of them gates anything.
- **Deterministic and stateless.** No database, no network, no home directory; identical inputs
  produce byte-identical reports.

## Install

```bash
uv sync            # or: uv tool install .   (Python >= 3.13)
```

## Use

```bash
seikan schema                      # machine-readable self-description of the whole contract
seikan hash thesis.json            # canonical identity + the exact data keys a run must bind
seikan check-data px.csv           # pre-flight a CSV against the strict data contract
seikan describe px.csv             # pure data profiling — measures nothing
seikan run thesis.json --data PX=px.csv --report-out report.json
```

A minimal thesis:

```json
{
  "name": "deep-drawdown-rebound",
  "data": { "targets": ["PX"] },
  "entry": {
    "type": "first_true",
    "condition": {
      "type": "and",
      "conditions": [
        { "type": "threshold", "left": { "type": "drawdown" }, "op": "<",
          "right": { "type": "constant", "value": -0.4 } },
        { "type": "threshold", "left": { "type": "bars_since_extremum", "extremum": "min" },
          "op": ">=", "right": { "type": "constant", "value": 20 } }
      ]
    }
  },
  "params": { "horizon": [21, 63], "benchmark": "market" }
}
```

Exit codes describe how far the RUN got, never how the evidence looked:

| code | meaning |
|---|---|
| 0 | the command completed — for `run`, every nominated output was written (the per-cell results live inside `gate.cells`; exit 0 is not a verdict) |
| 2 | input data failed strict validation (see `data_report`) |
| 3 | invalid request — a usage error, an invalid thesis DSL or threshold set, or an unusable output path |
| 4 | internal error |

It declares two data keys — the target `PX`, and `benchmark`, which `params.benchmark: "market"`
brings in — so it runs as
`seikan run thesis.json --data PX=px.csv --data benchmark=spy.csv --report-out report.json`.
The document names its series and never locates them: paths and column bindings belong to the
invocation and are stamped into the report, so re-pulled or re-shaped data next month re-runs the
same exam under the same identity. This thesis declares a two-cell grid (one entry combo × two
horizons), so the report carries two entries in `summary.cells`, each graded on its own.

## Development

```bash
uv run ruff format --check src tests   # formatting is enforced
uv run ruff check src tests            # lint
uv run mypy                            # strict, over the whole package
uv run pytest                          # the whole suite, zero skips
```

## Where the depth lives

`CONTRACT.md` (repo root) is the working contract: the CLI and data-binding semantics, the
report layout, the per-cell checklist, the frozen statistical mechanics and the strict-CSV
doctrine. `seikan schema` emits the same contract machine-readably — the DSL JSON Schema,
thresholds, CSV contracts, exit codes and field dictionaries.
