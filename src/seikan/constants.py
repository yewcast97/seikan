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

#: The four price columns that make a file OHLCV-shaped — ``dataio``'s shape detection and OHLC
#: invariants, and the loader's target frames.
OHLC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")

#: The names the engine writes beside a combo's sweep levels: ``target`` (the regime column) and
#: ``horizon`` (the measurement axis) in the trades frame and the cell params, and ``timestamps``
#: (the firing list) in each ``list_entries`` row ``{**combo, target, timestamps}``. A swept
#: constant or shared axis may not claim one — the row would overwrite its value.
RESERVED_SWEEP_LEVELS: frozenset[str] = frozenset({"target", "horizon", "timestamps"})

#: Names a ``params.features`` key may not take: ``target``, ``horizon`` and every trade column.
#: Features are written into the trades frame beside the engine's own columns, so a collision
#: either overwrites evidence or duplicates a column name (a duplicate makes ``trades["ret"]`` a
#: DataFrame and the statistics read a 2-D array).
RESERVED_FEATURE_NAMES: frozenset[str] = frozenset({"target", "horizon", *TRADE_COLUMNS})

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

#: The ceiling on every BAR-COUNT parameter a thesis declares — a transform ``window`` /
#: ``periods``, ``rolling.window`` / ``min_count``, ``first_true.cooldown``, ``lag.periods`` and
#: ``params.horizon`` — enforced by the ``dsl.nodes`` integer aliases at PARSE time (exit 3), shared
#: axes included (``Thesis._check_shared_axes`` re-validates every value through the same aliases).
#: A cross node's ``min_valid`` carries it too, only because it shares the ``Ge2Int`` alias: it is a
#: MEMBER count, and its own floor (no more than the declared targets) binds long before.
#: Unbounded, a well-typed thesis was not an executable one: an ema ``window: 10**20`` reached a
#: numba kernel as a Python int no machine integer holds and died in the compiler (a
#: ``TypingError``, exit 4 — the internal-bug class — from caller input). ``2**31 − 1`` fits every
#: integer type the kernels and pandas windows compile for (int32 included) with the index
#: arithmetic the engine adds to it (``t + 1 + h``) still far inside int64, and no series
#: approaches it. A STRUCTURAL bound like ``MAX_DECLARED_GRID``: a parameter within it but longer
#: than the sample is legal and keeps its ordinary outcome — a window, ``periods`` or lag is all
#: warmup and never emits, a cooldown fires its first edge and holds every later one, and a
#: horizon the common index cannot close refuses a measuring run as insufficient data (exit 2).
MAX_BAR_COUNT_PARAM = 2**31 - 1

#: The smallest EW decay ``alpha`` the DSL admits (``ema`` / ``zscore``):
#: ``2 / (MAX_BAR_COUNT_PARAM + 1)``, which is ``2**-30`` exactly, so the equivalent span
#: ``ceil(2/alpha − 1)`` an alpha warms over (``compiler.nb.ema_params``) is at most
#: ``MAX_BAR_COUNT_PARAM`` — and ``alpha = MIN_EW_ALPHA`` is the window form at
#: ``window = MAX_BAR_COUNT_PARAM``, bit for bit. Below it the span passes the ceiling. Above
#: ~1.08e-19 (``2**-63``, a span of 2**64) the run was all warmup — from ~2.17e-19 (``2**-62``, a
#: span of 2**63) down only while no signed specialization of the same kernel had been compiled
#: first in the process, else "int too big to convert" — and at or below it the span was one no
#: machine integer holds: ``1e-19`` implied a 2e19-bar warmup (a numba ``TypingError``) and the
#: subnormal ``5e-324`` an infinite one (an ``OverflowError`` out of ``math.ceil``), both exit 4
#: from a thesis the schema had accepted.
MIN_EW_ALPHA = 2.0 / (MAX_BAR_COUNT_PARAM + 1)

#: The ceiling on Series operator-nesting depth, enforced by ``dsl.schema.Thesis`` at PARSE time
#: (exit 3) on both the entry tree and ``params.features``. Counted levels are the transforms
#: (ema, zscore, percentile, rolling_agg, drawdown, runup, bars_since_extremum, change,
#: rolling_corr, the cross-sectional nodes, and the event-anchor nodes ``bars_since_event`` /
#: ``event_value`` / ``event_agg``); ``binary_op``/``unary_op``/``shift``/``mask`` are transparent
#: plumbing, and a Condition embedded in an event node is a decision rather than a level — its
#: threshold operands are checked as roots of their own. A STRUCTURAL bound like
#: ``MAX_DECLARED_GRID`` — it bounds what the language can say (and what a reader must audit),
#: never what any reported number means.
MAX_SERIES_NESTING = 5

#: A seconds fraction with a nonzero digit past the ninth. The clock is nanosecond resolution and
#: pandas TRUNCATES the rest, so such a stamp reads as another instant (``…T00:00:00.0000000001``
#: as midnight). Zeros past the ninth digit name the same instant and are admitted. One rule for
#: every stamp seikan reads: a CSV index (``dataio``) and the ``data.start``/``data.end`` bounds.
PAST_NANOSECOND_FRACTION = r"\.\d{9}\d*[1-9]"
