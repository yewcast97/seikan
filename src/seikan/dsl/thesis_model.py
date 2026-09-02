"""The document spine: the reserved data keys, ``ExternalFeed``, ``DataSpec``, ``Outcome``,
``BacktestParams`` and ``Thesis`` with its cross-field validators.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import Field as PField
from pydantic import model_validator

from seikan.constants import (
    DEFAULT_FEATURE_NAMES,
    MAX_DECLARED_GRID,
    MAX_SERIES_NESTING,
    RESERVED_FEATURE_NAMES,
    RESERVED_SWEEP_LEVELS,
    TRADE_COLUMNS,
)

if TYPE_CHECKING:
    import pandas as pd
from seikan.dsl.conditions import Condition
from seikan.dsl.nodes import PosIntParam, Series, _Strict
from seikan.dsl.traverse import (
    _iter_sweep_axis_names,
    _series_depth,
    _series_external_names,
    _series_has_sweep,
    declared_grid_size,
    iter_condition_series,
    iter_external_names,
    series_cross_nodes,
)

#: The data key naming the excess-return source ``params.benchmark = "market"`` measures against.
#: A DEDICATED slot, not an external feed: external feeds are asof-anchored/lagged *decision*
#: inputs, while the benchmark is *outcome measurement* — its open price is sampled at exactly the
#: observation's anchor bars (open[t+1] / open[t+1+h]) on the joined index, no ffill, no lag. A bar
#: where the benchmark is missing censors the observation (exit_reason "no_benchmark") rather than
#: shifting the timeline.
BENCHMARK_KEY = "benchmark"

#: Names no target or feed may take, because the run's key namespace is flat and these are spoken
#: for. Exactly one key is reserved; the tuple exists so adding a second stays one edit.
RESERVED_DATA_KEYS = (BENCHMARK_KEY,)


class ExternalFeed(_Strict):
    # Structured external-feed entry. A feed is a logical KEY, never a file: the CSV behind it is
    # named at invocation (``seikan run --data <feed>=<path>``), and so is the COLUMN read out of
    # that CSV (``seikan run --column <feed>=<col>``, or ``--column <feed>@<target>=<col>`` for one
    # member of a per-target feed). What an entry configures is the SEMANTIC read — what the series
    # MEANS to this thesis, which is the only thing the document is entitled to fix. ``per_target``
    # gives each target its own series (the invocation then answers one path per target under the
    # derived keys ``<feed>@<target>``) where the default broadcasts ONE series across every
    # target; ``lag`` shifts the feed's timestamps forward by a calendar duration before the asof
    # anchor, modelling publication delay (an int is days; a string is a pandas Timedelta like
    # "36h" — note "1m" is one minute). A column NAME is not semantic: it is a property of the file
    # that happens to answer this key — one vendor ships three series in one CSV under its own
    # spellings, another ships each in its own file — so a name here would let re-shaping the CSV
    # turn the same exam into a DIFFERENT document, which is exactly why paths do not live here
    # either. Feed timestamps are treated as AVAILABILITY times; if the source stamps values at
    # the period they describe (a daily aggregate stamped that day's midnight, a month-end figure),
    # set ``lag`` to the real publication delay or the value leaks into bars that predate its
    # release.

    per_target: bool = False
    lag: str | int = 0

    @model_validator(mode="after")
    def _check_lag(self) -> ExternalFeed:
        _ = self.lag_timedelta  # parse now so a bad lag fails at model_validate time
        return self

    @property
    def lag_timedelta(self) -> pd.Timedelta:
        import pandas as pd  # local: keep the DSL module import-light

        td = pd.Timedelta(days=self.lag) if isinstance(self.lag, int) else pd.Timedelta(self.lag)
        if pd.isna(td):
            # pd.Timedelta("nan"/"NaT") returns NaT rather than raising, and NaT compares False
            # against every bound — downstream it would silently skip the timestamp shift, turning
            # a declared publication delay into zero lag (the look-ahead direction).
            raise ValueError(
                f"external feed lag must be a finite duration, got {self.lag!r} (parses to NaT)"
            )
        if td < pd.Timedelta(0):
            raise ValueError(f"external feed lag must be >= 0, got {self.lag!r}")
        return td


class DataSpec(_Strict):
    # WHICH series this thesis measures, NAMED but not located. ``targets`` lists the logical keys
    # to backtest together (each becomes a column); the CSV behind every key — and WHICH column of
    # that CSV answers it — is supplied at invocation (``seikan run --data <key>=<path> --column
    # <key>=<col>``). A file path inside the document would make the same exam over re-pulled data
    # a DIFFERENT document — ``dsl_hash`` would move for a reason that has nothing to do with what
    # is being asked, and a thesis could not be re-measured next month without being rewritten —
    # so the DSL names a series and the invocation locates it. A column name is that same fact one
    # level in: it is a property of the FILE that happens to answer a key (this vendor ships three
    # yields in one CSV under its own spellings, that one ships each in a file of its own), never a
    # property of the thesis, so while it lived here the same exam over a RE-SHAPED CSV was
    # likewise a different document — a column rename or a split file moved a hash the question
    # never moved.
    # All time series are STRICT CSV files with an ISO-8601 timezone-naive datetime index (see
    # ``dataio.read_strict_csv`` — no format guessing, ever): a full-OHLCV file is a
    # price target; a file WITHOUT open/high/low/close is a SERIES target (a yield, a spread, a
    # valuation multiple, a strategy index) — its single value column is measured directly
    # (open=high=low=close=value is synthesized so the close-reading algebra applies; no volume).
    # A multi-column series file needs ``--column <target>=<col>`` to say which of its columns the
    # target IS; an OHLCV target takes no column binding at all, since a price target always
    # measures its open-anchored prices. All targets of one thesis must share
    # one shape — mixing a price target with a series target is rejected at load.
    targets: list[str]
    start: str | None = None
    end: str | None = None
    # Alternative-data feeds (strict CSV), keyed by feed name. Each entry configures the SEMANTIC
    # read — one series or one series per target, and what publication lag it carries; the file and
    # the column read out of it arrive with the invocation, like every other locating fact.
    external: dict[str, ExternalFeed] = PField(default_factory=dict)

    @model_validator(mode="after")
    def _check_key_namespace(self) -> DataSpec:
        # The run's data keys live in ONE flat namespace, because that is the namespace a caller
        # types on a command line: a target key, a feed key, a derived ``<feed>@<target>`` key and
        # the reserved ``benchmark`` key all compete for the same ``--data KEY=PATH`` slot. Two
        # declarations answering to one key would let a single ``--data`` pair silently stand in
        # for both, so the collision is refused here rather than resolved by precedence. It is
        # also the namespace ``--column KEY=COL`` addresses: one key names both the file that
        # answers it and the column read out of that file, so a collision left standing here would
        # misdirect two flags rather than one — and a column bound for the series a caller meant,
        # applied to the series they did not, is a silently different measurement.
        if not self.targets:
            raise ValueError("data.targets must name at least one target series")
        seen: dict[str, str] = {}
        # Typed ``Sequence[object]`` because that is how this check reads them — a name is a
        # candidate --data key until it has been shown to be a string, and the isinstance below is
        # the check that shows it. Naming the declared element type here instead would make that
        # first clause statically dead.
        groups: tuple[tuple[str, Sequence[object]], ...] = (
            ("target", self.targets),
            ("external feed", list(self.external)),
        )
        for kind, names in groups:
            for name in names:
                if not isinstance(name, str) or not name or name != name.strip():
                    raise ValueError(
                        f"{kind} name {name!r} must be a non-empty string with no surrounding "
                        "whitespace — it is typed as a --data key"
                    )
                if "=" in name or "@" in name:
                    raise ValueError(
                        f"{kind} name {name!r} may not contain '=' or '@': '=' separates a "
                        "--data KEY=PATH pair and '@' derives a per-target feed key"
                    )
                if name in RESERVED_DATA_KEYS:
                    raise ValueError(
                        f"{kind} name {name!r} is reserved: the '{name}' key names the "
                        "excess-return source params.benchmark='market' asks for"
                    )
                if name in seen:
                    raise ValueError(
                        f"{kind} name {name!r} is already declared as a {seen[name]} — one data "
                        "key answers one series"
                    )
                seen[name] = kind
        return self

    @model_validator(mode="after")
    def _check_start_end(self) -> DataSpec:
        # The evaluated interval is part of what a report certifies, so its bounds obey the same
        # strict ISO-8601 timezone-naive discipline as the CSV index itself (dataio) — no format
        # guessing, ever. Handing them to pandas label slicing verbatim would silently INTERPRET
        # an ambiguous "01/02/2024", and an unparseable bound would escape as an uncaught
        # exception (exit 4) instead of a DSL refusal (exit 3).
        import pandas as pd  # local: keep the DSL module import-light

        for field in ("start", "end"):
            raw = getattr(self, field)
            if raw is None:
                continue
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"data.{field} must be a non-empty ISO-8601 timestamp string")
            try:
                ts = pd.to_datetime(raw, format="ISO8601")
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"data.{field}={raw!r} is not a strict ISO-8601 timestamp "
                    f"(e.g. '2024-01-31' or '2024-01-31T09:30:00'): {exc}"
                ) from exc
            if pd.isna(ts):
                raise ValueError(f"data.{field}={raw!r} parses to NaT")
            if ts.tzinfo is not None:
                raise ValueError(
                    f"data.{field}={raw!r} carries a timezone; the data contract is "
                    "timezone-naive timestamps only"
                )
        if (
            self.start is not None
            and self.end is not None
            and pd.to_datetime(self.start, format="ISO8601")
            >= pd.to_datetime(self.end, format="ISO8601")
        ):
            raise ValueError(
                f"data.start ({self.start!r}) must be strictly before data.end ({self.end!r})"
            )
        return self

    def feed_keys(self) -> dict[str, list[str]]:
        """{feed name → the data key(s) that answer it}. A per-target feed derives one key per
        target (``<feed>@<target>``, in target-declaration order); a shared feed answers to its own
        name. Per-target cover is by construction here — there is no mapping to check against the
        targets, because the keys ARE derived from them."""
        return {
            name: ([f"{name}@{t}" for t in self.targets] if feed.per_target else [name])
            for name, feed in self.external.items()
        }


class Outcome(_Strict):
    # WHAT a firing measures (the observation's outcome). The default (``params.outcome`` omitted,
    # or spelled ``{}`` — both canonicalize to the same filled object, one identity)
    # is the target's forward PERCENT return on the next-open anchor. ``series`` picks the measured
    # series: ``"target"`` = the target's own value column; a declared external feed name = that
    # feed's forward evolution (e.g. realized vol, a credit spread, an IV index — "when X fires,
    # Y moves"). ``kind`` picks the measurement algebra: ``pct`` = (b/a − 1), ``log`` = ln(b/a)
    # (both need a positive-scale series), ``diff`` = (b − a) — the honest form for
    # rates/spreads/multiples, where a percent of a near-zero or sign-crossing level is meaningless
    # (a 10y yield falling 4.0 → 3.5 is −0.5 in ``diff``, not −12.5%). The anchor stays next-bar
    # (off=+1) for every outcome — no same-bar look-ahead; ``direction`` still sets the sign, so
    # ``shortonly`` + ``diff`` profits when the level FALLS. A NaN window in a feed outcome
    # censors the observation as ``exit_reason="no_outcome"``.

    series: str = "target"
    kind: Literal["pct", "log", "diff"] = "pct"


class BacktestParams(_Strict):
    # Observer-native forward-return event study: every bar where ``entry`` fires opens an
    # independent, OVERLAPPING forward-return observation measured over ``horizon`` bars. There is
    # no exit condition and no one-position-at-a-time state machine. Returns are raw measurements
    # (exit/entry - 1): there is no fee/slippage model and no equity curve. `direction`
    # sets the sign of the measured return (longonly = the forward return, shortonly = its
    # negative).
    # `horizon` is the forward measurement window in bars (default 1 = the immediate next-period
    # forward return); a list sweeps it as its own result axis ("horizon"), yielding a return
    # response curve (e.g. [1, 5, 10, 20]) — set it explicitly, the default is only a neutral
    # fallback.
    # A firing bar t's observation is always anchored at the NEXT bar's open —
    # open[t+1]→open[t+1+h] — the only tradable convention (a same-bar close[t]→close[t+h]
    # measurement would read from a price the decision itself consumed). `features` names extra
    # entry-time series snapshots (any scalar-param Series, incl. externals) used for conditional
    # bucketing of returns — defaults to built-in momentum + volatility snapshots when unset.
    # There is no sampling knob of any kind here: every declared parameter × horizon cell is
    # measured over the WHOLE index and reported independently, so the DSL cannot express a
    # partition of the data that some cells see and others do not. The circular-shift rotation
    # null likewise always uses every non-identity shift (a capped subsample has residue
    # aliasing), so its resolution is a property of the series length, never a choice.
    # `benchmark` switches the measured forward return from raw to EXCESS: "market" subtracts the
    # same-window return of the ``benchmark`` key's series (open[t+1]→open[t+1+h], the same
    # next-open anchor). Without it, long-horizon (20-60 bar) raw returns are dominated by market
    # beta. Under `shortonly` the excess is sign·(tgt_ret − bench_ret) — profits when the target
    # UNDERPERFORMS the benchmark (the hedged-short reading). Every downstream statistic
    # (rotation null, HAC,
    # conditional buckets, PBO) then describes excess returns; the summary records the mode under
    # "benchmark". "cross_mean" (basket mode ONLY — ``Thesis`` validation enforces it) subtracts
    # the same-window mean forward return of ALL declared members, self included, measured in the
    # outcome's own algebra — the relative-value read "did this member beat the basket?". It
    # couples the targets in the OUTCOME exactly as cross nodes couple them in the signal, which
    # is why conjunction refuses it. Its missingness is fail-closed: any member's leg non-finite
    # at a bar censors the WHOLE bar's benchmark leg, so every member's firing there exits as
    # ``no_benchmark`` — a hole in one member never silently reshapes the others' benchmark.
    direction: Literal["longonly", "shortonly"] = "longonly"
    horizon: PosIntParam = PField(default=1)
    features: dict[str, Series] | None = PField(default=None)
    benchmark: Literal["market", "cross_mean"] | None = None
    # Non-optional with a default factory, so the field has ONE canonical spelling per meaning:
    # omitted and ``{}`` both validate to the defaults-filled object and hash identically, and an
    # explicit ``null`` refuses (it would be a second spelling of the same default, splitting one
    # thesis into two identities).
    outcome: Outcome = PField(default_factory=Outcome)

    @model_validator(mode="after")
    def _check_features_scalar(self) -> BacktestParams:
        for name, node in (self.features or {}).items():
            if _series_has_sweep(node):
                raise ValueError(
                    f"feature {name!r} must use scalar params (no list sweeps); features are "
                    f"grouping variables for conditional analysis, not swept result axes"
                )
        return self

    @model_validator(mode="after")
    def _check_feature_names(self) -> BacktestParams:
        # Feature snapshots are written into the trades frame BESIDE the engine's own columns,
        # so a colliding name either overwrites evidence (a feature named `ret` replaced the
        # measured forward return) or duplicates a column — after which `trades["ret"]` is a
        # DataFrame, the statistics read a 2-D array, and `trades["is_open"]` crashes the
        # boolean index. The runner cannot catch this: by the time it sees the name, the
        # collision has already happened.
        # An EMPTY dict is refused rather than aliased: `{}` behaves exactly like the omitted
        # field (the built-in snapshots), so admitting it would give one meaning two identities —
        # the None-vs-{} hash split. Omit the field for the defaults.
        if self.features is not None and not self.features:
            raise ValueError(
                "params.features, when given, must be non-empty — omit the field entirely for "
                "the default momentum/volatility snapshots"
            )
        # ``Iterable[object]`` like ``DataSpec._check_key_namespace``: the isinstance below is what
        # establishes these keys are strings, so it stays a live check, not a dead clause.
        feature_names: Iterable[object] = self.features or {}
        for name in feature_names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("feature names must be non-empty strings")
            if name in RESERVED_FEATURE_NAMES:
                raise ValueError(
                    f"feature name {name!r} collides with a reserved result column "
                    f"(the trades frame's own fields plus 'target'/'horizon'); rename the "
                    f"feature. Reserved: {sorted(RESERVED_FEATURE_NAMES)}"
                )
        return self


class Thesis(_Strict):
    # An observer-native event-study thesis: ``entry`` is the firing condition (the belief that a
    # signal precedes a forward return); there is no exit condition. Every firing bar opens an
    # overlapping forward-return observation over ``params.horizon`` bars (see BacktestParams).
    name: str
    description: str | None = None
    # How the targets relate. ``conjunction`` (the default): the targets are the thesis's REGIME,
    # measured side by side with the weakest deciding, and no cross-target statistic is formed at
    # all. ``basket`` declares the targets ONE
    # cross-section per bar: the cross nodes (cross_rank/cross_demean/cross_agg) and
    # ``benchmark: "cross_mean"`` become legal, and every cell gains a POOLED cross-target
    # statistics block the checklist grades instead of per-member floors. The default is the only
    # legal value below 2 targets (a basket of one is degenerate).
    target_mode: Literal["conjunction", "basket"] = "conjunction"
    data: DataSpec
    entry: Condition
    params: BacktestParams = PField(default_factory=BacktestParams)

    @model_validator(mode="after")
    def _check_external_feeds_declared(self) -> Thesis:
        used = set(iter_external_names(self.entry))
        for node in (self.params.features or {}).values():
            used |= set(_series_external_names(node))
        missing = sorted(used - self.data.external.keys())
        if missing:
            raise ValueError(
                f"conditions reference external feed(s) {missing} not declared in "
                f"data.external (declared: {sorted(self.data.external)})"
            )
        # The converse refuses too: a declared feed nothing reads still demands a file at
        # invocation, is loaded, digested and stamped into the identity — moving the dsl_hash
        # without changing the measurement. ``outcome.series`` counts as a use only when it names
        # a DECLARED feed: the spelling "target" always means the target itself, never a feed
        # that happens to carry that name.
        outcome_series = self.params.outcome.series
        if outcome_series != "target" and outcome_series in self.data.external:
            used.add(outcome_series)
        unused = sorted(self.data.external.keys() - used)
        if unused:
            raise ValueError(
                f"external feed(s) {unused} are declared in data.external but never referenced "
                f"by the entry condition, params.features, or params.outcome.series — remove the "
                f"declaration or reference the feed"
            )
        return self

    def data_keys(self) -> list[str]:
        """Every logical data key this thesis needs a CSV for, in resolution order: each target,
        then each external feed (its own name, or ``<feed>@<target>`` per target when the feed is
        per-target), then ``benchmark`` when ``params.benchmark`` asks for a market source.

        This list IS the run's request: ``seikan run --data KEY=PATH`` must answer it exactly, and
        a thesis that says which series it reads without saying where they live is what makes the
        same exam re-runnable over re-pulled data."""
        keys = list(self.data.targets)
        for derived in self.data.feed_keys().values():
            keys.extend(derived)
        if self.params.benchmark == "market":
            keys.append(BENCHMARK_KEY)
        return keys

    @model_validator(mode="after")
    def _check_target_mode(self) -> Thesis:
        # basket pools the targets into ONE cross-section per bar, so it needs a cross-section
        # to pool. Note the converse is NOT required: a basket thesis need not carry a cross
        # node — the mode alone changes the statistical read (pooled per-cell block, pooled
        # floors), so declaring it over plain per-target signals is meaningful.
        if self.target_mode == "basket" and len(self.data.targets) < 2:
            raise ValueError(
                f"target_mode='basket' requires >= 2 targets (data.targets); got "
                f"{len(self.data.targets)} — a basket of one is degenerate; use "
                f"target_mode='conjunction'"
            )
        return self

    @model_validator(mode="after")
    def _check_cross_nodes_mode(self) -> Thesis:
        # A cross-sectional transform ranks/demeans/aggregates ACROSS targets at each bar —
        # exactly the coupling conjunction declares the targets do NOT have — so cross nodes
        # require target_mode='basket' (which _check_target_mode already holds to >= 2 targets)
        # and a satisfiable min_valid. Features are scanned too: a cross-sectional feature
        # snapshot couples the targets the same way an entry operand does.
        series_iter = list(iter_condition_series(self.entry))
        series_iter.extend((self.params.features or {}).values())
        cross_nodes = [n for s in series_iter for n in series_cross_nodes(s)]
        if not cross_nodes:
            return self
        if self.target_mode != "basket":
            kinds = sorted({n.type for n in cross_nodes})
            raise ValueError(
                f"cross-sectional node(s) {kinds} require target_mode='basket' — conjunction "
                "declares the targets an independent regime and forms no cross-target "
                "statistic; declare target_mode='basket' to rank within the group"
            )
        n_targets = len(self.data.targets)
        for node in cross_nodes:
            if node.min_valid > n_targets:
                raise ValueError(
                    f"cross-sectional node {node.type!r} has min_valid={node.min_valid} but only "
                    f"{n_targets} targets are declared; it could never be defined"
                )
        return self

    @model_validator(mode="after")
    def _check_outcome_consistency(self) -> Thesis:
        outcome = self.params.outcome
        if outcome.series != "target" and outcome.series not in self.data.external:
            raise ValueError(
                f"params.outcome.series {outcome.series!r} is not a declared external feed "
                f"(declared: {sorted(self.data.external)}); use 'target' or declare the feed in "
                f"data.external"
            )
        if self.params.benchmark == "market" and outcome.kind == "diff":
            raise ValueError(
                "params.benchmark='market' cannot be combined with outcome kind 'diff': a diff "
                "outcome is in the target's own LEVEL units (bp, index points, ratio turns) and "
                "subtracting a benchmark RETURN from it is incommensurable by construction; "
                "drop the benchmark or use outcome kind 'pct'/'log'"
            )
        if self.target_mode == "basket" and outcome.kind == "diff":
            raise ValueError(
                "target_mode='basket' cannot be combined with outcome kind 'diff': the pooled "
                "cross-target statistics average returns across members, which needs a common "
                "unit the engine cannot certify for level changes (bp, index points, ratio "
                "turns); pct/log are scale-free — use one of those or target_mode='conjunction'"
            )
        return self

    @model_validator(mode="after")
    def _check_benchmark_consistency(self) -> Thesis:
        # ``params.benchmark='market'`` needs no matching source FIELD to agree with: it simply
        # adds the reserved ``benchmark`` key to ``data_keys()``, and an invocation that does not
        # answer that key is refused at resolution. There is no stale-source refusal ("declared
        # but unused") to make either — an unused declaration is not expressible.
        if self.params.benchmark == "cross_mean" and self.target_mode != "basket":
            raise ValueError(
                "params.benchmark='cross_mean' requires target_mode='basket': it demeans each "
                "member's forward return by the basket mean, coupling the targets in the "
                "OUTCOME exactly as cross nodes couple them in the signal — which conjunction "
                "forbids"
            )
        return self

    @model_validator(mode="after")
    def _check_declared_grid(self) -> Thesis:
        # The gate's search cap is structural: a declared grid above it fails the search cap in
        # EVERY cell under any legal thresholds (`settings` admits no looser ceiling), so no
        # per-cell result it could produce would be readable evidence. Enforcing it here — before
        # a single CSV is read — makes that refusal cost nothing, where otherwise the runner
        # prices the entire grid first (rotation nulls and per-cell panels over every combo ×
        # horizon × target) only to hand the gate a summary whose search cap fails every cell
        # in it. The engine owes an impossible exam no report.
        size = declared_grid_size(self.entry, self.params.horizon)
        if size > MAX_DECLARED_GRID:
            raise ValueError(
                f"declared hypothesis grid is {size} (swept entry params × horizons) but the "
                f"sealed search cap is {MAX_DECLARED_GRID} — a grid this wide fails the search "
                "cap in every cell under any legal thresholds; narrow the sweep"
            )
        return self

    @model_validator(mode="after")
    def _check_sweep_axis_names(self) -> Thesis:
        # The runner's vectorize.collect_sweeps rejects reserved (target/horizon) and duplicate
        # sweep-axis names, and the runner rejects trade/feature-column collisions — but both fire
        # only AFTER a data load, surfacing as exit 4 (internal). Reproduce the SAME refusals here,
        # at parse time (exit 3), off the exact axis names the engine will assign. The runtime
        # checks stay as library-boundary backstops for a model_construct-built thesis.
        names = _iter_sweep_axis_names(self.entry)
        seen: set[str] = set()
        for lvl in names:
            if lvl in RESERVED_SWEEP_LEVELS:
                raise ValueError(
                    f"sweep axis name {lvl!r} is reserved (the engine names the target and horizon "
                    f"axes itself); rename the swept constant's 'name'"
                )
            if lvl in seen:
                raise ValueError(
                    f"duplicate sweep axis name {lvl!r}; each swept constant 'name' must be unique "
                    f"and must not collide with a transform axis (e.g. 'ema_window')"
                )
            seen.add(lvl)
        features = (
            set(self.params.features)
            if self.params.features is not None
            else set(DEFAULT_FEATURE_NAMES)
        )
        collisions = sorted(seen & (set(TRADE_COLUMNS) | features))
        if collisions:
            raise ValueError(
                f"sweep axis name(s) {collisions} collide with reserved trade/feature columns "
                f"(the trades frame's own fields plus the entry-time feature snapshots); rename "
                f"the swept constant's 'name'"
            )
        return self

    @model_validator(mode="after")
    def _check_series_nesting_depth(self) -> Thesis:
        # Operators may nest at most ``MAX_SERIES_NESTING`` (5) levels deep; deeper is rejected.
        # binary_op/unary_op/shift are transparent (do not count as a level) — see _series_depth.
        for series in iter_condition_series(self.entry):
            depth = _series_depth(series)
            if depth > MAX_SERIES_NESTING:
                raise ValueError(
                    f"series {series.type!r} nests {depth} operator levels deep; the maximum is "
                    f"{MAX_SERIES_NESTING} (each transform counts one level; "
                    f"binary_op/unary_op/shift are free). Flatten or split the expression."
                )
        for name, node in (self.params.features or {}).items():
            depth = _series_depth(node)
            if depth > MAX_SERIES_NESTING:
                raise ValueError(
                    f"feature {name!r} series {node.type!r} nests {depth} operator levels deep; "
                    f"the maximum is {MAX_SERIES_NESTING} (each transform counts one level; "
                    f"binary_op/unary_op/shift are free). Flatten or split the expression."
                )
        return self


# Forward references resolve against THIS module's namespace — the unions and their
# members must rebuild where they are defined.
BacktestParams.model_rebuild()
Thesis.model_rebuild()
