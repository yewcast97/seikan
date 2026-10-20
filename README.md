<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/seikan-lockup-white.svg">
    <img src="assets/seikan-lockup.svg" alt="seikan" width="320">
  </picture>
</p>

seikan is a stateless command-line engine for forward-return event studies of trading and
investment theses, built to be driven by agents or humans. A thesis (JSON) declares an entry
condition, the parameters it sweeps and the horizons to measure — no exit rule, no portfolio
simulation; seikan measures every declared cell over strict CSV data and writes one JSON report.

## Key features

### Every cell, always

Every declared parameter × horizon cell is measured on the full sample and reported, combos that
never fired included: `len(summary.cells) == n_hypotheses_attempted` holds by construction.

### No winner

seikan selects, ranks and crowns nothing — no headline scalar, no verdict. Choosing among the
cells, and pricing the multiplicity of having declared them, stays with the caller, against the
stamped grid size.

### A checklist, not a test

Each cell gets a completeness, support and concentration checklist. It makes no significance claim
and certifies no positive expected return.

### Observer-pure measurement

A firing fills at the next bar's open; feed timestamps are availability times, and `lag` models
publication delay. What the engine cannot verify — that timestamps, vintages and universes are
honest — arrives with the data and stays the caller's.

### Fail-closed ledgers

A missing outcome, an undecidable decision bar and an unavailable raw input are each counted, and
each refuses the cell: deleting data can only leave a cell unmet, never improve it.

### Overlap-honest evidence

An event-time HAC, a rotation null, an episode bootstrap and an episode-deduplicated profile ride
every cell, and CSCV PBO the grid, as evidence — each classed in the report's `metric_roles` beside
its caveat; none of them gates anything.

### Deterministic and stateless

No database, no network, no home directory: identical inputs produce byte-identical reports.

## Install

```bash
uv sync            # or: uv tool install .   (Python >= 3.13)
```

## Quick start

```bash
seikan schema                      # the whole contract, machine-readable
seikan hash thesis.json            # canonical identity + the data keys a run must bind
seikan check-data px.csv           # pre-flight a CSV against the strict data contract
seikan describe px.csv             # profile data files; measures nothing
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

Its two horizons make two cells, each graded on its own. It names two data keys — the target `PX`
and the `benchmark` that `"market"` asks for — and the invocation binds them to files, so the
paths are stamped into the report rather than written into the thesis:

```bash
seikan run thesis.json --data PX=px.csv --data benchmark=spy.csv --report-out report.json
```

Exit codes describe how far the run got, never how the evidence looked:

| code | meaning |
|---|---|
| 0 | completed — for `run`, every nominated output was written; per-cell results live in `gate.cells` |
| 2 | input data failed strict validation (see `data_report`) |
| 3 | invalid request — a usage error, an invalid thesis or threshold set, or an unusable output path |
| 4 | internal error |

## Development

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy
uv run pytest
```

CI (`.github/workflows/ci.yml`) runs these on every push to `main` and every pull request, then
builds the wheel and checks it ships its package data. `pre-commit install` runs the ruff checks at
commit time.

## Documentation

- [`CONTRACT.md`](CONTRACT.md) — the working contract: the CLI and data binding, the report
  layout, the checklist, the statistical mechanics and the strict-CSV rules.
- `seikan schema` — the same contract, machine-readable: the DSL JSON Schema, the thresholds, the
  CSV contracts, the exit codes and the field dictionaries.
- `seikan schema --markdown` — the DSL guide.
