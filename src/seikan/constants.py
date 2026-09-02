"""Structural names and bounds shared across the DSL, the compiler and the settings layer.

A leaf module by design: it imports nothing from the package, so ``dsl.schema`` (which must
reject a colliding feature name at VALIDATION time, long before a runner exists) and
``compiler.runner`` (which writes those very columns) can agree on one source of truth without a
cycle. Everything here is a NAME or a STRUCTURAL bound — never a statistical threshold; those live
in :mod:`seikan.settings` and are stamped into every report.
"""

from __future__ import annotations

#: The columns ``compiler.runner`` writes into the trades frame. A caller-chosen name colliding
#: with one of these (a swept constant's ``name``, or a ``params.features`` key) would overwrite
#: the engine's own record — silently destroying ``ret`` or crashing the ``is_open`` boolean
#: index — so both namespaces are validated against this tuple.
TRADE_COLUMNS: tuple[str, ...] = (
    "entry_time",
    "exit_time",
    "entry_bar",
    "entry_px",
    "exit_px",
    "ret",
    "ret_raw",
    "ret_bench",
    "pre_ret",
    "mae",
    "mfe",
    "bars_to_positive",
    "bars_to_trough",
    "exit_reason",
    "is_open",
)

#: The sweep-axis levels the engine names itself (``target`` is the regime column, ``horizon`` the
#: measurement axis). A swept constant may not claim either.
RESERVED_SWEEP_LEVELS: frozenset[str] = frozenset({"target", "horizon"})

#: Names a ``params.features`` key may not take: the reserved sweep levels plus every trade
#: column. Features are written into the trades frame beside the engine's own columns, so a
#: collision either overwrites evidence or duplicates a column name (a duplicate makes
#: ``trades["ret"]`` a DataFrame and the statistics read a 2-D array).
RESERVED_FEATURE_NAMES: frozenset[str] = RESERVED_SWEEP_LEVELS | frozenset(TRADE_COLUMNS)

#: The default entry-time feature snapshot names ``compiler.runner._FEATURE_NODES`` builds when
#: ``params.features`` is unset (short/medium momentum + realized vol). Declared here so
#: ``dsl.schema`` can refuse a sweep axis that would shadow one at VALIDATION time; the runner
#: asserts its ``_FEATURE_NODES`` keys still equal this tuple.
DEFAULT_FEATURE_NAMES: tuple[str, ...] = ("ret_5", "ret_20", "vol_14")

#: The sealed ceiling on the DECLARED hypothesis grid (combos × horizons). This is the structural
#: search cap in its two enforcement sites: ``settings.GateThresholds.thesis_max_hypotheses``
#: cannot be constructed above it, and ``dsl.schema.Thesis`` refuses a grid larger than it BEFORE
#: any data is read — an over-cap grid fails the search cap in EVERY cell under any legal
#: thresholds, so the engine owes it no report (and no CPU).
MAX_DECLARED_GRID = 64

#: The ceiling on Series operator-nesting depth, enforced by ``dsl.schema.Thesis`` at PARSE time
#: (exit 3) on both the entry tree and ``params.features``. Counted levels are the transforms
#: (ema, zscore, percentile, rolling_agg, drawdown, runup, bars_since_extremum, change,
#: rolling_corr, the cross-sectional nodes); ``binary_op``/``unary_op``/``shift`` are transparent
#: plumbing. A STRUCTURAL bound like ``MAX_DECLARED_GRID`` — it bounds what the language can say
#: (and what a reader must audit), never what any reported number means.
MAX_SERIES_NESTING = 5
