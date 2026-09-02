from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from seikan.dsl.schema import (
    EMA,
    AndCondition,
    BinaryOp,
    Constant,
    External,
    ExternalFeed,
    Field,
    FirstTrueCondition,
    NotCondition,
    Percentile,
    Thesis,
    ThresholdCondition,
    ZScore,
    declared_grid_size,
)

NESTED_THESIS = {
    "name": "ema_cross_with_percentile_filter",
    "description": "EMA10/30 golden cross (first_true of a threshold) with a percentile<0.7 filter",
    "data": {"targets": ["target"], "start": "2020-01-01"},
    "entry": {
        "type": "and",
        "conditions": [
            {
                "type": "first_true",
                "condition": {
                    "type": "threshold",
                    "left": {
                        "type": "ema",
                        "input": {"type": "field", "column": "close"},
                        "window": 10,
                    },
                    "op": ">",
                    "right": {
                        "type": "ema",
                        "input": {"type": "field", "column": "close"},
                        "window": 30,
                    },
                },
            },
            {
                "type": "threshold",
                "left": {
                    "type": "percentile",
                    "window": 14,
                    "input": {"type": "field", "column": "close"},
                },
                "op": "<",
                "right": {"type": "constant", "value": 0.7},
            },
        ],
    },
    "params": {"horizon": 20},
}


def test_parses_nested_discriminated_union():
    t = Thesis.model_validate(NESTED_THESIS)
    assert t.name == "ema_cross_with_percentile_filter"
    assert isinstance(t.entry, AndCondition)
    # There is no dedicated CrossCondition — a crossover is spelled
    # first_true(threshold(fast > slow)).
    assert isinstance(t.entry.conditions[0], FirstTrueCondition)
    assert isinstance(t.entry.conditions[0].condition, ThresholdCondition)
    assert isinstance(t.entry.conditions[0].condition.left, EMA)
    assert isinstance(t.entry.conditions[0].condition.left.input, Field)
    assert isinstance(t.entry.conditions[1], ThresholdCondition)
    assert isinstance(t.entry.conditions[1].right, Constant)
    assert t.params.horizon == 20


def test_json_roundtrip_is_lossless():
    t1 = Thesis.model_validate(NESTED_THESIS)
    t2 = Thesis.model_validate_json(t1.model_dump_json())
    assert t1 == t2


def test_not_condition_negates():
    t = Thesis.model_validate(
        {
            "name": "always_in",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "not",
                "condition": {
                    "type": "threshold",
                    "left": {"type": "drawdown", "window": 14},
                    "op": ">",
                    "right": {"type": "constant", "value": 100.0},
                },
            },
        }
    )
    assert isinstance(t.entry, NotCondition)
    assert isinstance(t.entry.condition, ThresholdCondition)


def _thesis_with_entry_left(left: dict, external: dict | None = None) -> dict:
    # A declared feed nothing references refuses at validation, so the helper declares none by
    # default — callers whose `left` reads a feed pass its declaration explicitly.
    return {
        "name": "x",
        "data": {"targets": ["target"], "external": external or {}},
        "entry": {
            "type": "threshold",
            "left": left,
            "op": ">",
            "right": {"type": "constant", "value": 0.0},
        },
    }


# Reusable shapes for the MAX_SERIES_NESTING (5) boundary tests. Validation never reads data,
# so the windows are arbitrary legal values.
_FIELD_CLOSE = {"type": "field", "column": "close"}
# percentile(ema(percentile(ema(field)))) — a 4-transform chain (depth 4).
_CHAIN4 = {
    "type": "percentile",
    "window": 14,
    "input": {
        "type": "ema",
        "window": 5,
        "input": {
            "type": "percentile",
            "window": 14,
            "input": {"type": "ema", "window": 10, "input": _FIELD_CLOSE},
        },
    },
}
# ema(percentile(ema(percentile(ema(field))))) — depth 5, exactly at the cap.
_D5 = {"type": "ema", "window": 5, "input": _CHAIN4}


def test_ema_can_wrap_a_transform():
    # EMA of Percentile (depth 2) is allowed — bounded pipelining.
    t = Thesis.model_validate(
        _thesis_with_entry_left(
            {
                "type": "ema",
                "window": 10,
                "input": {
                    "type": "percentile",
                    "window": 14,
                    "input": {"type": "field", "column": "close"},
                },
            }
        )
    )
    assert isinstance(t.entry.left, EMA)
    assert isinstance(t.entry.left.input, Percentile)


def test_transform_can_wrap_field_or_transform():
    # ZScore over a Field and over another transform are both allowed.
    t1 = Thesis.model_validate(
        _thesis_with_entry_left(
            {"type": "zscore", "window": 20, "input": {"type": "field", "column": "close"}}
        )
    )
    assert isinstance(t1.entry.left.input, Field)
    t2 = Thesis.model_validate(
        _thesis_with_entry_left(
            {
                "type": "zscore",
                "window": 20,
                "input": {
                    "type": "percentile",
                    "window": 14,
                    "input": {"type": "field", "column": "close"},
                },
            }
        )
    )
    assert isinstance(t2.entry.left.input, Percentile)


def test_binary_op_parses_and_roundtrips():
    t = Thesis.model_validate(
        _thesis_with_entry_left(
            {
                "type": "binary_op",
                "op": "-",
                "left": {"type": "ema", "input": {"type": "field", "column": "close"}, "window": 5},
                "right": {
                    "type": "ema",
                    "input": {"type": "field", "column": "close"},
                    "window": 20,
                },
            }
        )
    )
    assert isinstance(t.entry.left, BinaryOp)
    assert t.entry.left.op == "-"
    assert Thesis.model_validate_json(t.model_dump_json()) == t


def test_nesting_depth_capped_at_five():
    field = {"type": "field", "column": "close"}
    ema6 = field
    for _ in range(6):
        ema6 = {"type": "ema", "window": 5, "input": ema6}
    too_deep = (
        ema6,  # six stacked transforms over a leaf
        {"type": "zscore", "window": 10, "input": _D5},  # one wrap over a chain at the cap
        # transparent binary_op between the cap chain and the extra wrap — still 6 levels
        {
            "type": "ema",
            "window": 5,
            "input": {
                "type": "binary_op",
                "op": "+",
                "left": _D5,
                "right": {"type": "constant", "value": 1.0},
            },
        },
    )
    for bad in too_deep:
        with pytest.raises(ValidationError, match="nests"):
            Thesis.model_validate(_thesis_with_entry_left(bad))


def test_nesting_depth_allows_boundary_cases():
    ema_percentile = {
        "type": "ema",
        "window": 10,
        "input": {
            "type": "percentile",
            "window": 14,
            "input": {"type": "field", "column": "close"},
        },
    }
    # binary_op is transparent: binary_op(ema(percentile), ema(percentile)) = depth 2, allowed.
    Thesis.model_validate(
        _thesis_with_entry_left(
            {"type": "binary_op", "op": "-", "left": ema_percentile, "right": ema_percentile}
        )
    )
    # ema(ema(close)) = depth 2, allowed.
    Thesis.model_validate(
        _thesis_with_entry_left(
            {
                "type": "ema",
                "window": 5,
                "input": {
                    "type": "ema",
                    "window": 10,
                    "input": {"type": "field", "column": "close"},
                },
            }
        )
    )
    # binary_op stays transparent at the cap: both arms at depth 5 is still depth 5, allowed.
    Thesis.model_validate(
        _thesis_with_entry_left({"type": "binary_op", "op": "-", "left": _D5, "right": _D5})
    )
    # Five stacked transforms over a leaf — exactly at the cap, allowed.
    field = {"type": "field", "column": "close"}
    ema5 = field
    for _ in range(5):
        ema5 = {"type": "ema", "window": 5, "input": ema5}
    Thesis.model_validate(_thesis_with_entry_left(ema5))


def test_external_in_threshold():
    t = Thesis.model_validate(
        {
            "name": "rate_vs_shipping",
            "data": {
                "targets": ["target"],
                "external": {"rate": {}, "shipping": {}},
            },
            "entry": {
                "type": "threshold",
                "left": {
                    "type": "zscore",
                    "window": 60,
                    "input": {"type": "external", "name": "rate"},
                },
                "op": ">",
                "right": {
                    "type": "zscore",
                    "window": 60,
                    "input": {"type": "external", "name": "shipping"},
                },
            },
        }
    )
    assert set(t.data.external) == {"rate", "shipping"}
    assert isinstance(t.entry.left.input, External)
    assert t.entry.left.input.name == "rate"


def test_external_feed_entry_configures_the_semantic_read_only():
    # What a feed entry may configure is what its series MEANS to this thesis — answered once or
    # once per target (`per_target`), carrying what publication delay (`lag`). Both survive the
    # JSON round trip a stored thesis makes, because both are properties of the QUESTION and
    # nothing about them changes when the data behind the key is re-pulled or re-shaped.
    shared = ExternalFeed()
    assert shared.per_target is False and shared.lag == 0
    per_target = ExternalFeed(per_target=True, lag=14)
    assert per_target.per_target is True and per_target.lag == 14
    assert ExternalFeed.model_validate_json(per_target.model_dump_json()) == per_target
    with pytest.raises(ValidationError):  # extra="forbid"
        ExternalFeed(per_targett=True)


def test_external_feed_column_is_refused_as_an_unknown_key():
    # `column` is not a DSL field: which column of a CSV answers a key is a property of the file
    # that happens to answer it, not of the thesis, so it is bound at invocation (`--column
    # KEY=COL`) exactly as the file itself is (`--data KEY=PATH`). Were it carried here, re-shaping
    # a CSV — a renamed header, three series split into three files — would make the same exam a
    # DIFFERENT document. extra="forbid" is what makes the exclusion honest: a thesis carrying the
    # field is refused (exit 3) rather than silently accepted under an identity that does not
    # measure what it says.
    with pytest.raises(ValidationError, match="column"):
        ExternalFeed(column="iv_skew")
    # …and inside a whole document, where the refusal must point at the field rather than at the
    # feed that carries it — the match is what keeps this test from passing on any refusal at all.
    with pytest.raises(ValidationError, match=r"data\.external\.skew\.column"):
        Thesis.model_validate(
            {
                "name": "skew",
                "data": {
                    "targets": ["NVDA"],
                    "external": {"skew": {"column": "iv_skew"}},
                },
                "entry": {
                    "type": "threshold",
                    "left": {"type": "external", "name": "skew"},
                    "op": "<",
                    "right": {"type": "constant", "value": 0.0},
                },
            }
        )


def test_data_target_column_is_refused_as_an_unknown_key():
    # The target side of the same rule: a multi-column series file says which column each TARGET
    # is through `--column <target>=COL`, which additionally lets two targets read two
    # differently-headed files — something a single `data.target_column` could never say, since
    # it would have to serve every target at once.
    with pytest.raises(ValidationError, match=r"data\.target_column"):
        Thesis.model_validate(
            {
                "name": "curve",
                "data": {"targets": ["curve"], "target_column": "y10"},
                "entry": {
                    "type": "threshold",
                    "left": {"type": "field", "column": "close"},
                    "op": ">",
                    "right": {"type": "constant", "value": 0.0},
                },
            }
        )


def test_and_requires_min_two_conditions():
    with pytest.raises(ValidationError):
        Thesis.model_validate(
            {
                "name": "bad",
                "data": {"targets": ["target"]},
                "entry": {
                    "type": "and",
                    "conditions": [
                        {
                            "type": "threshold",
                            "left": {"type": "drawdown", "window": 14},
                            "op": "<",
                            "right": {"type": "constant", "value": -0.30},
                        }
                    ],
                },
            }
        )


def test_zscore_window_must_be_at_least_two():
    with pytest.raises(ValidationError):
        Thesis.model_validate(
            _thesis_with_entry_left(
                {
                    "type": "zscore",
                    "window": 1,
                    "input": {"type": "external", "name": "rate"},
                }
            )
        )


def test_json_schema_generation_works():
    schema = Thesis.model_json_schema()
    assert "$defs" in schema or "properties" in schema
    assert json.dumps(schema)


# ---- multi-target / multi-param / direction (backtest engine) --------------


def test_list_valued_window_is_a_sweep():
    ema = EMA(input=Field(column="close"), window=[10, 20, 30])
    assert ema.window == [10, 20, 30]
    assert EMA(input=Field(column="close"), window=10).window == 10


def test_list_window_elements_keep_constraint():

    with pytest.raises(ValidationError):
        ZScore(input=External(name="r"), window=[20, 1])  # 1 < 2


def test_swept_constant_requires_name():
    assert Constant(value=60.0).name is None  # scalar: no name needed (unchanged behavior)
    assert Constant(value=[55, 60, 65], name="thresh").value == [
        55.0,
        60.0,
        65.0,
    ]  # swept + name: ok
    with pytest.raises(ValidationError):
        Constant(value=[55, 60, 65])  # swept value, no name to label the axis
    with pytest.raises(ValidationError):
        Constant(value=[55, 60], name="   ")  # swept value, blank name


def test_swept_constant_keeps_min_length():
    with pytest.raises(ValidationError):
        Constant(value=[], name="x")  # list min_length >= 1


def test_horizon_scalar_list_or_default():
    from seikan.dsl.schema import BacktestParams

    assert BacktestParams().horizon == 1  # default = the immediate next-period forward return
    assert BacktestParams(horizon=10).horizon == 10
    assert BacktestParams(horizon=[5, 10, 20]).horizon == [5, 10, 20]


def test_horizon_keeps_positivity_constraint():
    from seikan.dsl.schema import BacktestParams

    for bad in (0, -1, [5, 0], []):  # scalar gt=0, list elements gt=0, list min_length>=1
        with pytest.raises(ValidationError):
            BacktestParams(horizon=bad)


def test_horizon_list_json_roundtrip():
    from seikan.dsl.schema import BacktestParams

    p = BacktestParams(horizon=[3, 6])
    assert BacktestParams.model_validate_json(p.model_dump_json()).horizon == [3, 6]


def test_sampling_knobs_are_rejected_as_unknown_keys():
    """The DSL cannot express a partition of the data that some cells see and others do not.

    Every declared parameter × horizon cell is measured over the WHOLE index and reported
    independently, so there is no holdout to size (`oos_fraction`) and no selection pass to
    aim (`selection_mode`) — and no rotation budget to cap (`n_rotations`), because the null
    uses every non-identity shift. All three are UNKNOWN keys under ``extra="forbid"``: a thesis
    carrying one is invalid input, never a silently ignored key that leaves the caller believing
    a tail was reserved.
    """
    from seikan.dsl.schema import BacktestParams

    for key, values in (
        ("oos_fraction", (0.3, 0.25, None)),
        ("selection_mode", ("in_sample", "full_sample", None)),
        ("n_rotations", (100, 1000)),
    ):
        assert key not in BacktestParams.model_fields
        for value in values:
            with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
                BacktestParams(**{key: value})


def test_backtest_params_surface_is_exactly_the_observer_knobs():
    """A pin on the whole field set: a mechanism the observer does not have must leave no field
    behind, and adding a default-valued field would change every stored dsl_hash, so the surface
    itself is worth asserting rather than only its individual members."""
    from seikan.dsl.schema import BacktestParams

    assert set(BacktestParams.model_fields) == {
        "direction",
        "horizon",
        "features",
        "benchmark",
        "outcome",
    }


def test_direction_field_longonly_or_shortonly():
    from seikan.dsl.schema import BacktestParams

    assert BacktestParams().direction == "longonly"
    assert BacktestParams(direction="shortonly").direction == "shortonly"
    for bad in ("both", "sideways"):  # 'both' is a position concept, foreign to the observer
        with pytest.raises(ValidationError):
            BacktestParams(direction=bad)


# ---- DSL features ----------------------------------------------------------


def test_features_reject_sweep_keep_scalar():
    from seikan.dsl.schema import BacktestParams

    scalar = {"type": "ema", "window": 20, "input": {"type": "field", "column": "close"}}
    sweep = {"type": "ema", "window": [10, 20], "input": {"type": "field", "column": "close"}}
    assert BacktestParams(features={"m": scalar}).features["m"].type == "ema"
    with pytest.raises(ValidationError):
        BacktestParams(features={"m": sweep})  # features are grouping vars, not swept axes


def test_thesis_feature_external_must_be_declared():
    from seikan.dsl.schema import Thesis

    zs_ext = {"type": "zscore", "window": 20, "input": {"type": "external", "name": "rate"}}
    base = {
        "name": "t",
        "data": {"targets": ["target"]},
        "entry": {
            "type": "threshold",
            "left": {"type": "field", "column": "close"},
            "op": "<",
            "right": {"type": "constant", "value": 1},
        },
    }
    with pytest.raises(ValidationError):
        Thesis.model_validate(
            {**base, "params": {"features": {"z": zs_ext}}}
        )  # 'rate' not declared
    ok = Thesis.model_validate(
        {
            **base,
            "data": {"targets": ["target"], "external": {"rate": {}}},
            "params": {"features": {"z": zs_ext}},
        }
    )
    assert ok.params.features["z"].type == "zscore"


def test_thesis_feature_depth_capped():
    from seikan.dsl.schema import Thesis

    base = {
        "name": "t",
        "data": {"targets": ["target"]},
        "entry": {
            "type": "threshold",
            "left": {"type": "field", "column": "close"},
            "op": "<",
            "right": {"type": "constant", "value": 1},
        },
    }
    # A depth-5 feature is accepted — params.features shares the entry tree's boundary exactly.
    Thesis.model_validate({**base, "params": {"features": {"d5": _D5}}})
    # zscore over a chain at the cap — 6 operator levels > 5, refused via the features branch.
    deep = {"type": "zscore", "window": 10, "input": _D5}
    with pytest.raises(ValidationError, match="nests 6 operator levels"):
        Thesis.model_validate({**base, "params": {"features": {"d": deep}}})


# ---- external feed entries (shared / per-target / lag) ----------------------


def test_external_feed_forms_parse():
    from seikan.dsl.schema import DataSpec, ExternalFeed

    # Every entry is an object — there is no bare-path shorthand, because the DSL holds no paths —
    # and what it configures is the SEMANTIC read: how many series the key stands for and what
    # publication delay each carries, never where they live or which column of a CSV they are.
    # `{}` is the whole declaration of an ordinary shared feed: one series, broadcast to every
    # target, its file (and column, if its file needs one named) supplied at invocation.
    # `per_target` is the other shape: one series per target, answered under the derived
    # `<feed>@<target>` keys.
    spec = DataSpec(
        targets=["AAA", "BBB"],
        external={"cpi": {"lag": 14}, "iv30": {"per_target": True}},
    )
    assert isinstance(spec.external["cpi"], ExternalFeed)
    assert spec.external["cpi"].lag == 14 and spec.external["cpi"].per_target is False
    assert spec.external["iv30"].per_target is True and spec.external["iv30"].lag == 0


def test_external_feed_lag_validation():
    import pandas as pd

    from seikan.dsl.schema import ExternalFeed

    assert ExternalFeed(lag=3).lag_timedelta == pd.Timedelta(days=3)
    assert ExternalFeed(lag="36h").lag_timedelta == pd.Timedelta(hours=36)
    assert ExternalFeed().lag_timedelta == pd.Timedelta(0)
    with pytest.raises(ValidationError):
        ExternalFeed(lag=-1)
    with pytest.raises(ValidationError):
        ExternalFeed(lag="garbage")
    with pytest.raises(ValidationError):
        ExternalFeed(lag="")
    # pd.Timedelta("nan"/"NaT") returns NaT instead of raising, and NaT compares False against
    # every bound — unrejected it would silently skip the timestamp shift downstream, degrading a
    # declared publication delay to zero (the look-ahead direction).
    with pytest.raises(ValidationError, match="NaT"):
        ExternalFeed(lag="nan")
    with pytest.raises(ValidationError, match="NaT"):
        ExternalFeed(lag="NaT")


def test_external_feed_rejects_unknown_keys():
    from seikan.dsl.schema import ExternalFeed

    with pytest.raises(ValidationError):
        ExternalFeed(target="AAA")


def test_structured_external_json_roundtrip_is_lossless():
    thesis = dict(
        _thesis_with_entry_left({"type": "external", "name": "rate"}),
        data={
            "targets": ["target"],
            "external": {"rate": {"lag": 14}},
        },
    )
    t1 = Thesis.model_validate(thesis)
    t2 = Thesis.model_validate_json(t1.model_dump_json())
    assert t1 == t2
    assert t2.data.external["rate"].lag == 14


# ---- the data key namespace (targets, feeds, and the reserved benchmark key) ----
#
# A thesis NAMES its series and never locates them, so the names are what an invocation types:
# `seikan run --data KEY=PATH`. That makes the key namespace part of the DSL's surface — flat,
# because a command line is flat — and the validator below owns it. Every refusal here is a name
# that could not be typed, or could be typed for two different series at once.


def test_targets_must_name_at_least_one_series():
    from seikan.dsl.schema import DataSpec

    # There is no implicit target to fall back on: the document holds no paths, so the list IS
    # the declaration of what gets measured. An empty one declares a thesis with nothing to
    # measure.
    with pytest.raises(ValidationError, match="must name at least one target"):
        DataSpec(targets=[])
    with pytest.raises(ValidationError):
        DataSpec()  # the field is required — no default empty list either


def test_data_key_names_must_be_typeable():
    from seikan.dsl.schema import DataSpec

    # A key with surrounding whitespace cannot be typed back: `--data ' AAA'=x.csv` either loses
    # the space to the shell or carries it into a key the DSL spells differently, and an empty one
    # names nothing at all. Both are refused where they are declared rather than where they fail.
    for bad in ("", " AAA", "AAA "):
        with pytest.raises(ValidationError, match="typed as a --data key"):
            DataSpec(targets=[bad])
    with pytest.raises(ValidationError, match="typed as a --data key"):
        DataSpec(targets=["AAA"], external={" iv": {}})


def test_data_key_names_may_not_contain_the_two_separators():
    from seikan.dsl.schema import DataSpec

    # `=` separates a `--data KEY=PATH` pair and `@` derives a per-target feed key, so a name
    # carrying either is ambiguous by construction — `--data a=b=c.csv` and a feed literally named
    # `iv@AAA` would both parse into something the caller did not write.
    for bad in ("A=B", "iv@AAA"):
        with pytest.raises(ValidationError, match="may not contain"):
            DataSpec(targets=[bad])
        with pytest.raises(ValidationError, match="may not contain"):
            DataSpec(targets=["AAA"], external={bad: {}})


def test_benchmark_is_a_reserved_data_key():
    from seikan.dsl.schema import BENCHMARK_KEY, RESERVED_DATA_KEYS, DataSpec

    # The excess-return source `params.benchmark='market'` asks for is a DEDICATED slot in the same
    # flat namespace, so nothing else may answer to its name — a target called "benchmark" would
    # make one `--data benchmark=...` pair stand for both a measured series and the thing it is
    # measured against.
    assert RESERVED_DATA_KEYS == (BENCHMARK_KEY,) == ("benchmark",)
    with pytest.raises(ValidationError, match="is reserved"):
        DataSpec(targets=["benchmark"])
    with pytest.raises(ValidationError, match="is reserved"):
        DataSpec(targets=["AAA"], external={"benchmark": {}})


def test_a_target_and_a_feed_may_not_share_one_key():
    from seikan.dsl.schema import DataSpec

    # One key answers one series. Two declarations under one name would let a single `--data` pair
    # silently stand in for both, which is a collision to refuse rather than resolve by precedence:
    # whichever way it resolved, the run would read a file for a series the caller thinks it did
    # not supply.
    with pytest.raises(ValidationError, match="already declared as a target"):
        DataSpec(targets=["AAA", "BBB"], external={"AAA": {}})
    DataSpec(targets=["AAA"], external={"AAA_iv": {}})  # merely similar names are fine


def test_feed_keys_derives_one_key_per_target_for_a_per_target_feed():
    from seikan.dsl.schema import DataSpec

    # Per-target COVER is by construction here: the keys are derived from the target list, so
    # there is no mapping left to check against it and no way to declare a feed that covers some
    # targets and not others. A shared feed answers to its own name, once.
    spec = DataSpec(
        targets=["AAA", "BBB"],
        external={"cpi": {}, "iv30": {"per_target": True}},
    )
    assert spec.feed_keys() == {
        "cpi": ["cpi"],
        "iv30": ["iv30@AAA", "iv30@BBB"],
    }
    assert DataSpec(targets=["AAA"]).feed_keys() == {}


def test_data_keys_is_the_runs_request_in_resolution_order():
    # `data_keys()` IS what `seikan run --data KEY=PATH` must answer, so its ORDER is contract:
    # every target in declaration order, then each feed (its own name, or the derived
    # `<feed>@<target>` keys in target order), then the reserved benchmark key when — and only
    # when — `params.benchmark` asks for a market source.
    base = {
        "name": "keys",
        "data": {
            "targets": ["AAA", "BBB"],
            "external": {"cpi": {}, "iv30": {"per_target": True}},
        },
        "entry": {
            "type": "threshold",
            "left": {"type": "external", "name": "cpi"},
            "op": ">",
            "right": {"type": "external", "name": "iv30"},
        },
    }
    assert Thesis.model_validate(base).data_keys() == [
        "AAA",
        "BBB",
        "cpi",
        "iv30@AAA",
        "iv30@BBB",
    ]
    with_market = Thesis.model_validate({**base, "params": {"benchmark": "market"}})
    assert with_market.data_keys() == [
        "AAA",
        "BBB",
        "cpi",
        "iv30@AAA",
        "iv30@BBB",
        "benchmark",
    ]


def test_data_keys_adds_the_benchmark_key_only_for_a_market_source():
    # `cross_mean` reads no file — it demeans by the basket's own members — so it adds no key.
    # The benchmark slot is not a mode flag: it is a series the invocation has to supply, and it
    # appears in the request exactly when a file is needed for it.
    plain = Thesis.model_validate(_th(_SIMPLE_ENTRY))
    assert plain.data_keys() == ["target"]
    market = Thesis.model_validate(_th(_SIMPLE_ENTRY) | {"params": {"benchmark": "market"}})
    assert market.data_keys() == ["target", "benchmark"]
    cross = Thesis.model_validate(
        _multi_target_thesis(_SIMPLE_ENTRY, n_targets=2) | {"params": {"benchmark": "cross_mean"}}
    )
    assert cross.data_keys() == ["t0", "t1"]


def test_path_fields_are_unknown_keys():
    """Paths are no part of the document, and `extra="forbid"` is what makes that a BREAK rather
    than a silent reinterpretation. A thesis carrying `data.path`, `data.paths`, `data.benchmark` or
    a feed's `path`/`paths` is invalid input — never a key quietly ignored while the run reads
    files the caller never named, or measures a benchmark it thinks it declared. The bare-string
    feed shorthand goes the same way: `{"iv": "iv.csv"}` is not a feed at all, because a feed
    entry is an object of read options and a string is a location.

    (With no paths inside the hashed document, the same exam over re-pulled data is the SAME
    document, and its `dsl_hash` is stable across them.)
    """
    from seikan.dsl.schema import DataSpec

    for bad in (
        {"path": "a.csv"},
        {"paths": {"AAA": "a.csv"}},
        {"benchmark": "spx.csv"},
    ):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            DataSpec(targets=["AAA"], **bad)
    for bad in ({"path": "iv.csv"}, {"paths": {"AAA": "iv.csv"}}):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            DataSpec(targets=["AAA"], external={"iv": bad})
    with pytest.raises(ValidationError):
        DataSpec(targets=["AAA"], external={"iv": "iv.csv"})


# ---- upfront feed-completeness validation ------------------------------------


def _thesis_with_entry(entry: dict, external: dict | None = None) -> dict:
    return {
        "name": "x",
        "data": {"targets": ["target"], "external": external or {}},
        "entry": entry,
    }


def _vix_threshold() -> dict:
    return {
        "type": "threshold",
        "left": {"type": "external", "name": "vix"},
        "op": ">",
        "right": {"type": "constant", "value": 20.0},
    }


def test_missing_external_feed_fails_at_validate_time():
    with pytest.raises(ValidationError, match="vix"):
        Thesis.model_validate(_thesis_with_entry(_vix_threshold()))
    # declaring the feed fixes it
    Thesis.model_validate(_thesis_with_entry(_vix_threshold(), external={"vix": {}}))


def test_missing_external_feed_via_transform_input():
    entry = {
        "type": "threshold",
        "left": {"type": "percentile", "window": 60, "input": {"type": "external", "name": "iv30"}},
        "op": ">",
        "right": {"type": "constant", "value": 0.8},
    }
    with pytest.raises(ValidationError, match="iv30"):
        Thesis.model_validate(_thesis_with_entry(entry))


def test_missing_external_feed_via_binary_op_and_nested_transform():
    # Feeds referenced inside a binary_op (and under a transform) are discovered recursively.
    entry = {
        "type": "threshold",
        "left": {
            "type": "binary_op",
            "op": "-",
            "left": {"type": "external", "name": "a"},
            "right": {"type": "zscore", "window": 5, "input": {"type": "external", "name": "b"}},
        },
        "op": ">",
        "right": {"type": "constant", "value": 0.0},
    }
    with pytest.raises(ValidationError, match=r"a.*b|'a', 'b'"):
        Thesis.model_validate(_thesis_with_entry(entry))
    # declaring both feeds fixes it
    Thesis.model_validate(_thesis_with_entry(entry, external={"a": {}, "b": {}}))


def test_missing_external_feed_nested_and_first_true_rolling():
    # There is no dedicated CrossCondition — a crossover-style firing condition is spelled
    # first_true(threshold(...)), and external-feed discovery must walk through it.
    first_true_spread = {
        "type": "first_true",
        "condition": {
            "type": "threshold",
            "left": {"type": "external", "name": "spread"},
            "op": ">",
            "right": {"type": "constant", "value": 0.0},
        },
    }
    nested = {
        "type": "and",
        "conditions": [
            {
                "type": "not",
                "condition": {
                    "type": "rolling",
                    "window": 5,
                    "agg": "any",
                    "condition": first_true_spread,
                },
            },
            {"type": "or", "conditions": [_vix_threshold(), _vix_threshold()]},
        ],
    }
    with pytest.raises(ValidationError, match=r"spread.*vix|'spread', 'vix'"):
        Thesis.model_validate(_thesis_with_entry(nested))


def test_unknown_exit_key_is_rejected():
    # Unknown keys, `exit` among them, are rejected under extra="forbid".
    thesis = _thesis_with_entry(
        {
            "type": "threshold",
            "left": {"type": "drawdown", "window": 14},
            "op": "<",
            "right": {"type": "constant", "value": -0.30},
        }
    )
    thesis["exit"] = _vix_threshold()
    with pytest.raises(ValidationError):
        Thesis.model_validate(thesis)


def test_declared_but_unused_external_feed_refuses():
    # An unused declaration still demands a file at invocation and moves the identity without
    # changing the measurement — refused, naming the feed.
    with pytest.raises(ValidationError, match=r"\['spare'\].*never referenced"):
        Thesis.model_validate(
            _thesis_with_entry(
                {
                    "type": "threshold",
                    "left": {"type": "drawdown", "window": 14},
                    "op": "<",
                    "right": {"type": "constant", "value": -0.30},
                },
                external={"spare": {}},
            )
        )


# ---- rolling_corr (trailing-window Pearson correlation of two series) ------


def test_rolling_corr_parses_and_roundtrips():
    from seikan.dsl.schema import RollingCorr

    entry = {
        "type": "threshold",
        "left": {
            "type": "rolling_corr",
            "left": {
                "type": "change",
                "input": {"type": "field", "column": "close"},
                "kind": "pct",
            },
            "right": {"type": "external", "name": "iv30"},
            "window": 20,
        },
        "op": ">",
        "right": {"type": "constant", "value": 0.5},
    }
    t = Thesis.model_validate(_th(entry, data={"targets": ["target"], "external": {"iv30": {}}}))
    assert isinstance(t.entry.left, RollingCorr)
    assert Thesis.model_validate(json.loads(t.model_dump_json())) == t


def test_rolling_corr_window_must_be_at_least_three():
    from seikan.dsl.schema import RollingCorr

    RollingCorr(left=Field(), right=Field(column="volume"), window=3)  # boundary ok
    with pytest.raises(ValidationError):
        RollingCorr(left=Field(), right=Field(column="volume"), window=2)


def test_rolling_corr_counts_as_one_operator_level_with_two_children():
    # rolling_corr counts one level over its deeper child: over a 4-transform chain it reads
    # depth 5 (ok, at the cap); wrapping the chain once more is depth 6 (rejected).
    ok = {
        "type": "threshold",
        "left": {"type": "rolling_corr", "left": _CHAIN4, "right": {"type": "field"}, "window": 10},
        "op": ">",
        "right": {"type": "constant", "value": 0.0},
    }
    Thesis.model_validate(_th(ok))
    deep = {
        "type": "threshold",
        "left": {"type": "rolling_corr", "left": _D5, "right": {"type": "field"}, "window": 10},
        "op": ">",
        "right": {"type": "constant", "value": 0.0},
    }
    with pytest.raises(ValidationError, match="nests"):
        Thesis.model_validate(_th(deep))


def test_rolling_corr_external_feed_discovery_via_left_and_right():
    entry_left_missing = {
        "type": "threshold",
        "left": {
            "type": "rolling_corr",
            "left": {"type": "external", "name": "a"},
            "right": {"type": "field"},
            "window": 10,
        },
        "op": ">",
        "right": {"type": "constant", "value": 0.0},
    }
    with pytest.raises(ValidationError, match="a"):
        Thesis.model_validate(_th(entry_left_missing))
    entry_right_missing = {
        "type": "threshold",
        "left": {
            "type": "rolling_corr",
            "left": {"type": "field"},
            "right": {"type": "external", "name": "b"},
            "window": 10,
        },
        "op": ">",
        "right": {"type": "constant", "value": 0.0},
    }
    with pytest.raises(ValidationError, match="b"):
        Thesis.model_validate(_th(entry_right_missing))
    Thesis.model_validate(
        _th(entry_left_missing, data={"targets": ["target"], "external": {"a": {}}})
    )


def test_rolling_corr_window_sweep_registers_as_rolling_corr_window():
    from seikan.compiler.vectorize import collect_sweeps

    entry = {
        "type": "threshold",
        "left": {
            "type": "rolling_corr",
            "left": {"type": "field"},
            "right": {"type": "field", "column": "volume"},
            "window": [10, 20, 30],
        },
        "op": ">",
        "right": {"type": "constant", "value": 0.0},
    }
    t = Thesis.model_validate(_th(entry))
    assert collect_sweeps(t.entry) == [("rolling_corr_window", [10, 20, 30])]


# ---- benchmark (excess-return) fields ----------------------------------------


def _bench_thesis(
    *,
    data_extra: dict | None = None,
    params_extra: dict | None = None,
    targets: list[str] | None = None,
    target_mode: str | None = None,
) -> dict:
    data: dict = {"targets": targets or ["target"]}
    data.update(data_extra or {})
    thesis: dict = {
        "name": "b",
        "data": data,
        "entry": {
            "type": "threshold",
            "left": {"type": "drawdown", "window": 14},
            "op": "<",
            "right": {"type": "constant", "value": -0.30},
        },
        "params": {"horizon": 20, **(params_extra or {})},
    }
    if target_mode is not None:
        thesis["target_mode"] = target_mode
    return thesis


def test_benchmark_market_asks_for_the_reserved_key_and_nothing_else():
    # `market` is a mode, not a source declaration: it says the measurement is an excess return
    # and adds the reserved `benchmark` key to what the invocation must answer. There is no field
    # for it to agree with, so cross-field refusals ('market' without a source, a source without
    # 'market') are simply unexpressible — an invocation that does not answer the key is refused
    # at resolution instead, where the file actually would be read.
    t = Thesis.model_validate(_bench_thesis(params_extra={"benchmark": "market"}))
    assert t.params.benchmark == "market"
    assert t.data_keys() == ["target", "benchmark"]
    assert Thesis.model_validate(_bench_thesis()).data_keys() == ["target"]


def test_benchmark_cross_mean_requires_basket_mode():
    # cross_mean couples the targets in the OUTCOME exactly as cross nodes couple them in the
    # signal — conjunction refuses it (even multi-target); basket accepts it, its >= 2 targets
    # already held by the mode itself.
    with pytest.raises(ValidationError, match="requires target_mode='basket'"):
        Thesis.model_validate(
            _bench_thesis(targets=["a", "b"], params_extra={"benchmark": "cross_mean"})
        )
    Thesis.model_validate(
        _bench_thesis(
            targets=["a", "b"],
            params_extra={"benchmark": "cross_mean"},
            target_mode="basket",
        )
    )


def test_benchmark_rejects_unknown_values():
    # The literal is the whole surface: no alias, casing variant, or invented mode sails through.
    for bad in ("median", "sector", "CROSS_MEAN"):
        with pytest.raises(ValidationError):
            Thesis.model_validate(_bench_thesis(params_extra={"benchmark": bad}))


def test_benchmark_fields_json_roundtrip():
    t = Thesis.model_validate(_bench_thesis(params_extra={"benchmark": "market"}))
    assert Thesis.model_validate(json.loads(t.model_dump_json())) == t


# ---- target modes + cross-sectional transforms (CrossRank / CrossDemean / CrossAgg) ----


def _cross_entry(kind: str = "cross_rank", min_valid: int | None = None) -> dict:
    node: dict = {
        "type": kind,
        "input": {"type": "change", "periods": 5, "input": {"type": "field", "column": "close"}},
    }
    if kind == "cross_agg":
        node["agg"] = "frac_positive"
    if min_valid is not None:
        node["min_valid"] = min_valid
    return {
        "type": "threshold",
        "left": node,
        "op": "<=",
        "right": {"type": "constant", "value": 0.34},
    }


def _multi_target_thesis(
    entry: dict, n_targets: int = 3, features: dict | None = None, target_mode: str = "basket"
) -> dict:
    targets = [f"t{i}" for i in range(n_targets)]
    thesis: dict = {
        "name": "x",
        "target_mode": target_mode,
        "data": {"targets": targets},
        "entry": entry,
    }
    if features is not None:
        thesis["params"] = {"features": features}
    return thesis


def test_cross_sectional_parses_and_roundtrips():
    for kind in ("cross_rank", "cross_demean", "cross_agg"):
        t = Thesis.model_validate(_multi_target_thesis(_cross_entry(kind, min_valid=2)))
        assert t.target_mode == "basket"
        assert Thesis.model_validate(json.loads(t.model_dump_json())) == t


def test_cross_nodes_require_basket_mode():
    # The gate is the MODE, not the target count: even a multi-target conjunction refuses a
    # cross node — conjunction declares the targets independent, and a node coupling them
    # contradicts that declaration. The message must name the mode that would make it legal.
    for kind in ("cross_rank", "cross_demean", "cross_agg"):
        with pytest.raises(ValidationError, match="require target_mode='basket'"):
            Thesis.model_validate(
                _multi_target_thesis(_cross_entry(kind), target_mode="conjunction")
            )


def test_basket_requires_two_targets():
    with pytest.raises(ValidationError, match="a basket of one is degenerate") as exc:
        Thesis.model_validate(_multi_target_thesis(_SIMPLE_ENTRY, n_targets=1))
    assert "data.targets" in str(exc.value)  # the refusal names the field that would widen it
    Thesis.model_validate(_multi_target_thesis(_SIMPLE_ENTRY, n_targets=2))  # boundary ok


def test_basket_without_cross_node_is_valid():
    # Basket does not require a cross node: the mode alone changes the statistical read (the
    # pooled per-cell block and pooled floors), so declaring it over plain per-target signals
    # is meaningful, not vacuous.
    t = Thesis.model_validate(_multi_target_thesis(_SIMPLE_ENTRY))
    assert t.target_mode == "basket"


def test_market_benchmark_stays_legal_in_basket():
    # Only cross_mean is mode-gated; stripping beta against a market series is orthogonal to
    # how the targets relate to each other.
    t = Thesis.model_validate(
        _multi_target_thesis(_SIMPLE_ENTRY)
        | {"data": {"targets": ["a", "b"]}, "params": {"benchmark": "market"}}
    )
    assert t.params.benchmark == "market" and t.target_mode == "basket"
    assert t.data_keys() == ["a", "b", "benchmark"]


def test_min_valid_is_not_sweepable():
    # A swept definedness floor sweeps the SAMPLE, not the hypothesis — min_valid is a plain int.
    from seikan.dsl.schema import CrossAgg, CrossDemean, CrossRank

    with pytest.raises(ValidationError):
        CrossRank(input=Field(column="close"), min_valid=[2, 3])
    with pytest.raises(ValidationError):
        CrossDemean(input=Field(column="close"), min_valid=[2, 3])
    with pytest.raises(ValidationError):
        CrossAgg(input=Field(column="close"), agg="std", min_valid=[2, 3])


def test_cross_sectional_min_valid_must_be_satisfiable():
    with pytest.raises(ValidationError, match="min_valid=4"):
        Thesis.model_validate(_multi_target_thesis(_cross_entry(min_valid=4), n_targets=3))
    Thesis.model_validate(
        _multi_target_thesis(_cross_entry(min_valid=3), n_targets=3)
    )  # boundary ok


def test_cross_sectional_min_valid_floor_is_two():
    with pytest.raises(ValidationError):
        Thesis.model_validate(_multi_target_thesis(_cross_entry(min_valid=1)))


def test_cross_sectional_in_features_is_mode_checked():
    # params.features is scanned like the entry: a cross-sectional feature snapshot couples the
    # targets exactly as an entry operand does, so conjunction refuses it there too.
    feature = {
        "mom_rank": {
            "type": "cross_rank",
            "input": {
                "type": "change",
                "periods": 20,
                "input": {"type": "field", "column": "close"},
            },
        }
    }
    with pytest.raises(ValidationError, match="require target_mode='basket'"):
        Thesis.model_validate(
            _multi_target_thesis(_SIMPLE_ENTRY, features=feature, target_mode="conjunction")
        )
    Thesis.model_validate(_multi_target_thesis(_SIMPLE_ENTRY, features=feature))


def test_cross_sectional_counts_toward_depth_cap():
    # cross_rank counts one level like any transform: cross_rank(zscore(field)) is 2 levels, and
    # three more wraps reach the cap at 5 — allowed. A sixth wrap is rejected, and the reported
    # depth of 6 pins that cross_rank was COUNTED (if it were free, `six` would read 5 and pass).
    two = {
        "type": "cross_rank",
        "input": {"type": "zscore", "input": {"type": "field"}, "window": 10},
    }
    five = {
        "type": "ema",
        "window": 5,
        "input": {
            "type": "percentile",
            "window": 14,
            "input": {"type": "ema", "window": 10, "input": two},
        },
    }
    Thesis.model_validate(
        _multi_target_thesis(
            {
                "type": "threshold",
                "left": five,
                "op": ">",
                "right": {"type": "constant", "value": 0.8},
            }
        )
    )
    six = {"type": "percentile", "input": five, "window": 14}
    with pytest.raises(ValidationError, match="nests 6 operator levels"):
        Thesis.model_validate(
            _multi_target_thesis(
                {
                    "type": "threshold",
                    "left": six,
                    "op": ">",
                    "right": {"type": "constant", "value": 0.8},
                }
            )
        )


def test_zscore_recipe_depth():
    # The documented zscore recipe — binary_op(cross_demean(x), "/", cross_agg(x, "std")) — costs
    # the same 1 + d(x) levels as either node alone (binary_op is transparent), so over a depth-4
    # x it sits exactly at the cap; one more real level inside x tips both arms to 6 and refuses.
    x4 = {
        "type": "change",
        "periods": 5,
        "input": {
            "type": "ema",
            "window": 10,
            "input": {
                "type": "percentile",
                "window": 14,
                "input": {
                    "type": "ema",
                    "window": 5,
                    "input": {"type": "field", "column": "close"},
                },
            },
        },
    }
    recipe = {
        "type": "binary_op",
        "op": "/",
        "left": {"type": "cross_demean", "input": x4},
        "right": {"type": "cross_agg", "agg": "std", "input": x4},
    }
    Thesis.model_validate(
        _multi_target_thesis(
            {
                "type": "threshold",
                "left": recipe,
                "op": ">",
                "right": {"type": "constant", "value": 1.0},
            }
        )
    )
    x5 = {
        "type": "change",
        "periods": 5,
        "input": {
            "type": "ema",
            "window": 10,
            "input": {
                "type": "percentile",
                "window": 14,
                "input": {
                    "type": "ema",
                    "window": 5,
                    "input": {
                        "type": "ema",
                        "window": 5,
                        "input": {"type": "field", "column": "close"},
                    },
                },
            },
        },
    }
    deep = {
        "type": "binary_op",
        "op": "/",
        "left": {"type": "cross_demean", "input": x5},
        "right": {"type": "cross_agg", "agg": "std", "input": x5},
    }
    with pytest.raises(ValidationError, match="nests 6 operator levels"):
        Thesis.model_validate(
            _multi_target_thesis(
                {
                    "type": "threshold",
                    "left": deep,
                    "op": ">",
                    "right": {"type": "constant", "value": 1.0},
                }
            )
        )


def test_cross_sectional_external_discovery_walks_input():
    # An External inside a cross node's input must be declared — _iter_child_series covers cross
    # nodes.
    entry = {
        "type": "threshold",
        "left": {"type": "cross_rank", "input": {"type": "external", "name": "pe"}},
        "op": "<=",
        "right": {"type": "constant", "value": 0.2},
    }
    with pytest.raises(ValidationError, match="external feed"):
        Thesis.model_validate(_multi_target_thesis(entry))


def test_basket_refuses_diff_outcome():
    # Pooled cross-target statistics average returns across members — that needs a common unit
    # the engine cannot certify for level changes (bp, index points); pct/log are scale-free.
    with pytest.raises(ValidationError, match="common unit"):
        Thesis.model_validate(
            _multi_target_thesis(_SIMPLE_ENTRY) | {"params": {"outcome": {"kind": "diff"}}}
        )
    # conjunction multi-target + diff stays valid: nothing pools, so nothing needs a common unit.
    Thesis.model_validate(
        _multi_target_thesis(_SIMPLE_ENTRY, target_mode="conjunction")
        | {"params": {"outcome": {"kind": "diff"}}}
    )
    # …and the ratio algebras stay legal in basket.
    Thesis.model_validate(
        _multi_target_thesis(_SIMPLE_ENTRY) | {"params": {"outcome": {"kind": "log"}}}
    )


# ---- target_mode hash discipline (canonical_dsl_hash lives in gate.py) ------


def test_omitted_target_mode_hashes_as_conjunction():
    """``canonical_dsl_hash`` normalizes with defaults filled, so a thesis that never mentions
    ``target_mode`` and one spelling the default explicitly are ONE identity — the same
    discipline the gate's hash-stability pins hold for every other default."""
    from seikan.gate import canonical_dsl_hash

    base = _th(_SIMPLE_ENTRY)
    assert canonical_dsl_hash(base) == canonical_dsl_hash({**base, "target_mode": "conjunction"})


def test_basket_mode_hashes_differently():
    """The mode changes what every number means (pooled grading, cross_mean legality), so two
    theses differing only in ``target_mode`` are distinct hypotheses with distinct identities."""
    from seikan.gate import canonical_dsl_hash

    conj = _multi_target_thesis(_SIMPLE_ENTRY, target_mode="conjunction")
    basket = _multi_target_thesis(_SIMPLE_ENTRY, target_mode="basket")
    assert canonical_dsl_hash(conj) != canonical_dsl_hash(basket)


# ---- Phase-1 algebra nodes (calendar / days_since / shift / unary_op / drawdown.input) ----


def _th(entry: dict, data: dict | None = None) -> dict:
    return {"name": "x", "data": data or {"targets": ["target"]}, "entry": entry}


def test_calendar_and_days_since_parse_and_roundtrip():
    entry = {
        "type": "and",
        "conditions": [
            {
                "type": "threshold",
                "left": {"type": "calendar", "field": "day_of_month"},
                "op": ">=",
                "right": {"type": "constant", "value": 25.0},
            },
            {
                "type": "threshold",
                "left": {"type": "days_since", "name": "earn"},
                "op": "<=",
                "right": {"type": "constant", "value": 3.0},
            },
        ],
    }
    t = Thesis.model_validate(_th(entry, data={"targets": ["target"], "external": {"earn": {}}}))
    assert Thesis.model_validate(json.loads(t.model_dump_json())) == t


def test_calendar_rejects_unknown_field():
    with pytest.raises(ValidationError):
        Thesis.model_validate(
            _th(
                {
                    "type": "threshold",
                    "left": {"type": "calendar", "field": "bars_to_month_end"},
                    "op": "==",
                    "right": {"type": "constant", "value": 0.0},
                }
            )
        )


def test_days_since_requires_declared_feed():
    # DaysSince references a feed by name exactly like External — undeclared fails at validate time.
    with pytest.raises(ValidationError, match="earn"):
        Thesis.model_validate(
            _th(
                {
                    "type": "threshold",
                    "left": {"type": "days_since", "name": "earn"},
                    "op": "<=",
                    "right": {"type": "constant", "value": 3.0},
                }
            )
        )


def test_shift_and_unary_are_depth_transparent():
    # zscore(x - shift(x)) is one operator level (shift/binary_op are plumbing) — allowed even
    # wrapped once more; and a chain of FIVE real operator levels threaded with shift/unary_op/
    # binary_op plumbing at every stage is still accepted (if the plumbing counted, this chain
    # would read >= 7 levels and reject — acceptance pins the transparency at the new cap). A
    # sixth real level rejects.
    diff = {
        "type": "binary_op",
        "op": "-",
        "left": {"type": "external", "name": "pmi"},
        "right": {"type": "shift", "input": {"type": "external", "name": "pmi"}, "periods": 1},
    }
    z = {"type": "zscore", "input": diff, "window": 12}  # 1 real level
    absz = {"type": "unary_op", "op": "abs", "input": z}  # still 1 (plumbing)
    data = {"targets": ["target"], "external": {"pmi": {}}}
    Thesis.model_validate(
        _th(
            {
                "type": "threshold",
                "left": absz,
                "op": ">",
                "right": {"type": "constant", "value": 2.0},
            },
            data=data,
        )
    )
    ema_absz = {"type": "ema", "input": absz, "window": 5}  # 2 levels — still fine
    Thesis.model_validate(
        _th(
            {
                "type": "threshold",
                "left": ema_absz,
                "op": ">",
                "right": {"type": "constant", "value": 2.0},
            },
            data=data,
        )
    )
    p3 = {"type": "percentile", "input": ema_absz, "window": 10}  # 3 real levels
    shifted = {"type": "shift", "input": p3, "periods": 1}  # still 3 (plumbing)
    e4 = {"type": "ema", "input": shifted, "window": 5}  # 4
    neg = {"type": "unary_op", "op": "neg", "input": e4}  # still 4 (plumbing)
    p5 = {"type": "percentile", "input": neg, "window": 14}  # 5 — at the cap
    Thesis.model_validate(
        _th(
            {
                "type": "threshold",
                "left": p5,
                "op": ">",
                "right": {"type": "constant", "value": 0.9},
            },
            data=data,
        )
    )
    six = {"type": "ema", "input": p5, "window": 5}  # 6 real levels — rejected
    with pytest.raises(ValidationError, match="nests 6 operator levels"):
        Thesis.model_validate(
            _th(
                {
                    "type": "threshold",
                    "left": six,
                    "op": ">",
                    "right": {"type": "constant", "value": 0.9},
                },
                data=data,
            )
        )


def test_shift_periods_must_be_positive():
    # Backward-only by construction: periods >= 1 (a zero/negative shift could read the future).
    with pytest.raises(ValidationError):
        Thesis.model_validate(
            _th(
                {
                    "type": "threshold",
                    "left": {"type": "shift", "input": {"type": "field"}, "periods": 0},
                    "op": ">",
                    "right": {"type": "constant", "value": 0.0},
                }
            )
        )


def test_shift_external_discovery_walks_input():
    with pytest.raises(ValidationError, match="pmi"):
        Thesis.model_validate(
            _th(
                {
                    "type": "threshold",
                    "left": {
                        "type": "shift",
                        "input": {"type": "external", "name": "pmi"},
                        "periods": 3,
                    },
                    "op": ">",
                    "right": {"type": "constant", "value": 50.0},
                }
            )
        )


def test_drawdown_input_parses_and_default_close_fills():
    from seikan.dsl.schema import Drawdown

    with_input = Thesis.model_validate(
        _th(
            {
                "type": "threshold",
                "left": {
                    "type": "drawdown",
                    "window": 14,
                    "input": {"type": "external", "name": "iv"},
                },
                "op": "<",
                "right": {"type": "constant", "value": -0.1},
            },
            data={"targets": ["target"], "external": {"iv": {}}},
        )
    )
    assert Thesis.model_validate(json.loads(with_input.model_dump_json())) == with_input
    short = Drawdown(window=14)
    dump = short.model_dump(mode="json")
    assert dump["input"] == {"type": "field", "column": "close"}


def test_drawdown_input_external_discovery():
    with pytest.raises(ValidationError, match="iv"):
        Thesis.model_validate(
            _th(
                {
                    "type": "threshold",
                    "left": {
                        "type": "drawdown",
                        "window": 14,
                        "input": {"type": "external", "name": "iv"},
                    },
                    "op": "<",
                    "right": {"type": "constant", "value": -0.1},
                }
            )
        )


# ---- params.outcome (WHAT is measured; the column it is READ from is bound at invocation) -----


def test_outcome_parses_and_roundtrips():
    t = Thesis.model_validate(
        _th(
            {
                "type": "threshold",
                "left": {"type": "field"},
                "op": ">",
                "right": {"type": "constant", "value": 3.0},
            }
        )
        | {"params": {"horizon": 5, "outcome": {"kind": "diff"}}}
    )
    assert t.params.outcome.kind == "diff" and t.params.outcome.series == "target"
    assert Thesis.model_validate(json.loads(t.model_dump_json())) == t


def test_outcome_feed_must_be_declared():
    with pytest.raises(ValidationError, match="not a declared external feed"):
        Thesis.model_validate(
            _th(
                {
                    "type": "threshold",
                    "left": {"type": "field"},
                    "op": "<",
                    "right": {"type": "constant", "value": 100.0},
                }
            )
            | {"params": {"outcome": {"series": "iv", "kind": "diff"}}}
        )
    Thesis.model_validate(
        _th(
            {
                "type": "threshold",
                "left": {"type": "field"},
                "op": "<",
                "right": {"type": "constant", "value": 100.0},
            },
            data={"targets": ["target"], "external": {"iv": {}}},
        )
        | {"params": {"outcome": {"series": "iv", "kind": "diff"}}}
    )


def test_market_benchmark_refuses_only_diff_outcomes():
    # The benchmark leg is measured in the SAME algebra as the outcome, so both
    # RATIO algebras are dimensionally consistent — a log outcome yields a true log-excess.
    # `diff` stays refused: level units minus a benchmark RETURN is incommensurable however it
    # is computed.
    base = _th(
        {
            "type": "threshold",
            "left": {"type": "field"},
            "op": "<",
            "right": {"type": "constant", "value": 100.0},
        },
        data={"targets": ["target"]},
    )
    with pytest.raises(ValidationError, match="incommensurable by construction"):
        Thesis.model_validate(
            base | {"params": {"benchmark": "market", "outcome": {"kind": "diff"}}}
        )
    Thesis.model_validate(base | {"params": {"benchmark": "market", "outcome": {"kind": "pct"}}})
    Thesis.model_validate(base | {"params": {"benchmark": "market", "outcome": {"kind": "log"}}})


def test_outcome_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        Thesis.model_validate(
            _th(
                {
                    "type": "threshold",
                    "left": {"type": "field"},
                    "op": ">",
                    "right": {"type": "constant", "value": 0.0},
                }
            )
            | {"params": {"outcome": {"kind": "diff", "anchor": "same_close"}}}
        )


# ---- left-side mean-reversion primitives (runup / bars_since_extremum) --------


def test_runup_and_bars_since_extremum_parse_and_fill_defaults():
    from seikan.dsl.schema import BarsSinceExtremum, Runup

    ru = Runup()
    dump = ru.model_dump(mode="json")
    assert dump == {
        "type": "runup",
        "input": {"type": "field", "column": "close"},
        "window": None,
    }
    bse = BarsSinceExtremum(extremum="max")
    dump = bse.model_dump(mode="json")
    assert dump == {
        "type": "bars_since_extremum",
        "extremum": "max",
        "input": {"type": "field", "column": "close"},
        "window": None,
    }
    # Explicit non-defaults survive
    ru2 = Runup(window=20, input={"type": "field", "column": "close"})
    assert ru2.model_dump(mode="json")["window"] == 20
    assert ru2.model_dump(mode="json")["input"] == {"type": "field", "column": "close"}


def test_runup_bars_since_roundtrip_in_thesis():
    thesis = Thesis.model_validate(
        {
            "name": "left",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "and",
                "conditions": [
                    {
                        "type": "threshold",
                        "op": ">=",
                        "left": {"type": "runup", "window": 60},
                        "right": {"type": "constant", "value": 0.05},
                    },
                    {
                        "type": "threshold",
                        "op": ">=",
                        "left": {"type": "bars_since_extremum", "extremum": "min"},
                        "right": {"type": "constant", "value": 5},
                    },
                ],
            },
            "params": {"horizon": 20},
        }
    )
    again = Thesis.model_validate(thesis.model_dump(mode="json"))
    assert again.entry.conditions[0].left.type == "runup"
    assert again.entry.conditions[1].left.extremum == "min"


def test_runup_counts_as_one_operator_level():
    # runup over a 4-transform chain = 5 levels (ok, at the cap); ema of that = 6 (reject).
    ok = {
        "type": "threshold",
        "op": ">",
        "left": {"type": "runup", "input": _CHAIN4},
        "right": {"type": "constant", "value": 0.05},
    }
    Thesis.model_validate(_th(ok))
    deep = {
        "type": "threshold",
        "op": ">",
        "left": {
            "type": "ema",
            "window": 5,
            "input": {"type": "runup", "input": _CHAIN4},
        },
        "right": {"type": "constant", "value": 0.05},
    }
    with pytest.raises(ValidationError, match="nests"):
        Thesis.model_validate(_th(deep))


def test_first_true_parses_and_rejects_negative_cooldown():
    entry = {
        "type": "first_true",
        "condition": {
            "type": "threshold",
            "left": {"type": "drawdown"},
            "op": "<",
            "right": {"type": "constant", "value": -0.3},
        },
        "cooldown": 5,
    }
    t = Thesis.model_validate(_th(entry) | {"params": {"horizon": 20}})
    assert t.entry.type == "first_true" and t.entry.cooldown == 5
    assert Thesis.model_validate(json.loads(t.model_dump_json())) == t
    with pytest.raises(ValidationError):
        Thesis.model_validate(_th({**entry, "cooldown": -1}) | {"params": {"horizon": 20}})


def test_expanding_rolling_agg_rejects_mean():
    with pytest.raises(ValidationError, match="expanding rolling_agg"):
        Thesis.model_validate(
            _th(
                {
                    "type": "threshold",
                    "left": {
                        "type": "rolling_agg",
                        "input": {"type": "field", "column": "close"},
                        "agg": "mean",
                    },
                    "op": ">",
                    "right": {"type": "constant", "value": 0.0},
                }
            )
            | {"params": {"horizon": 5}}
        )
    # Expanding max is allowed (window omitted).
    ok = Thesis.model_validate(
        _th(
            {
                "type": "threshold",
                "left": {
                    "type": "rolling_agg",
                    "input": {"type": "field", "column": "close"},
                    "agg": "max",
                },
                "op": ">",
                "right": {"type": "constant", "value": 0.0},
            }
        )
        | {"params": {"horizon": 5}}
    )
    assert ok.entry.left.window is None and ok.entry.left.agg == "max"


def test_rolling_min_count_exceeds_window_rejected():
    with pytest.raises(ValidationError, match="min_count"):
        Thesis.model_validate(
            _th(
                {
                    "type": "rolling",
                    "window": 5,
                    "agg": "count",
                    "min_count": 6,
                    "condition": {
                        "type": "threshold",
                        "left": {"type": "field", "column": "close"},
                        "op": "<",
                        "right": {"type": "constant", "value": 100.0},
                    },
                }
            )
            | {"params": {"horizon": 5}}
        )


# ---- reserved feature names + the pre-flight declared-grid cap ----


_SIMPLE_ENTRY = {
    "type": "threshold",
    "left": {"type": "field", "column": "close"},
    "op": "<",
    "right": {"type": "constant", "value": 95.5},
}


@pytest.mark.parametrize(
    "name",
    [
        "ret",
        "ret_raw",
        "ret_bench",
        "is_open",
        "target",
        "entry_bar",
        "mae",
        "horizon",
        "exit_reason",
        "pre_ret",
    ],
)
def test_reserved_feature_names_are_rejected(name):
    """A feature is written into the trades frame beside the engine's own columns, so a name the
    engine owns must refuse: `ret` would silently overwrite the measured forward return, and
    `is_open` would duplicate the column and crash the boolean index with an uncaught
    exception (exit 4)."""
    with pytest.raises(ValidationError, match="reserved result column"):
        Thesis.model_validate(
            _th(_SIMPLE_ENTRY)
            | {"params": {"features": {name: {"type": "field", "column": "close"}}}}
        )


def test_ordinary_feature_names_still_validate():
    t = Thesis.model_validate(
        _th(_SIMPLE_ENTRY)
        | {"params": {"features": {"my_vol": {"type": "field", "column": "close"}}}}
    )
    assert set(t.params.features) == {"my_vol"}


def test_declared_grid_over_the_cap_refuses_before_any_data_is_read():
    """The cap is structural — an over-cap grid fails the search cap in every cell under any
    legal thresholds, so it must refuse without pricing the whole sweep first. The refusal lands at
    ``model_validate``, on a document that names its series and locates none of them: no file has
    been bound to a data key yet, so an over-cap grid cannot even reach the question of what its
    CSVs contain."""
    entry = {
        "type": "threshold",
        "left": {"type": "field", "column": "close"},
        "op": "<",
        "right": {"type": "constant", "value": [float(i) for i in range(13)], "name": "k"},
    }
    with pytest.raises(ValidationError, match="fails the search cap"):
        Thesis.model_validate(
            {"name": "x", "data": {"targets": ["target"]}, "entry": entry}
            | {"params": {"horizon": [1, 2, 3, 4, 5]}}  # 13 × 5 = 65
        )


def test_declared_grid_exactly_at_the_cap_validates():
    entry = {
        "type": "threshold",
        "left": {"type": "field", "column": "close"},
        "op": "<",
        "right": {"type": "constant", "value": [float(i) for i in range(16)], "name": "k"},
    }
    t = Thesis.model_validate(_th(entry) | {"params": {"horizon": [1, 2, 3, 4]}})  # 16 × 4 = 64
    assert declared_grid_size(t.entry, t.params.horizon) == 64


def test_a_single_swept_axis_over_the_cap_refuses_on_its_own_bound():
    """One axis longer than the whole cap cannot possibly fit the product, so the per-axis
    `max_length` sanity bound catches it first. Same tier (exit 3), different message."""
    with pytest.raises(ValidationError, match="at most 64 items"):
        Thesis.model_validate(_th(_SIMPLE_ENTRY) | {"params": {"horizon": list(range(1, 66))}})


def test_swept_horizon_counts_toward_the_product():
    # 33 horizons × 2 constants = 66: each axis is individually legal; the PRODUCT is not.
    entry = {
        "type": "threshold",
        "left": {"type": "field", "column": "close"},
        "op": "<",
        "right": {"type": "constant", "value": [94.0, 95.5], "name": "k"},
    }
    with pytest.raises(ValidationError, match="fails the search cap"):
        Thesis.model_validate(_th(entry) | {"params": {"horizon": list(range(1, 34))}})


def test_rolling_and_first_true_sweeps_count_toward_the_grid():
    """The walker must see the conditions' OWN swept params, not just the Series operands."""
    entry = {
        "type": "first_true",
        "cooldown": list(range(1, 10)),
        "condition": {
            "type": "rolling",
            "window": list(range(2, 11)),
            "agg": "all",
            "condition": _SIMPLE_ENTRY,
        },
    }
    # 9 cooldowns × 9 windows = 81 > 64
    with pytest.raises(ValidationError, match="fails the search cap"):
        Thesis.model_validate(_th(entry))


# ---- a sweep axis enumerates DISTINCT hypotheses -------------------------------------------


def test_duplicate_values_in_a_sweep_axis_refuse():
    """A repeated value declares the same hypothesis twice — zero new information, but it does
    not merely waste a cell: the measurement loop runs the duplicate combo once per occurrence
    and appends its observations to the trades frame each time, while the per-cell panel groups
    that frame by parameter VALUE. All copies then read one d-fold-duplicated group, so every
    count derived from it (``by_target.n``, the ledger's attempted/closed counts,
    ``episode_stats.n``) reports d times the real evidence while the overlap-honest ``n_nonoverlap``
    stays truthful — inflation the gate's ``n_nonoverlap <= n`` reconciliation cannot catch,
    because it
    only makes that bound easier. ``horizon: [21, 21]`` doubles a cell's apparent support and
    can flip the ``support`` floor from refused to passed: fail-OPEN, so it refuses here,
    before a byte of data is read."""
    for horizon in ([21, 21], [21, 34, 21]):
        with pytest.raises(ValidationError, match="repeats the value"):
            Thesis.model_validate({**NESTED_THESIS, "params": {"horizon": horizon}})


def test_duplicate_values_refuse_on_every_sweep_axis_kind():
    """The rule lives on the swept-list type itself, so transform windows, swept threshold
    constants and the horizon axis are all covered by one guard — no axis kind is exempt."""
    # a transform window axis
    with pytest.raises(ValidationError, match="repeats the value"):
        Thesis.model_validate(
            _thesis_with_entry_left(
                {
                    "type": "zscore",
                    "window": [20, 20],
                    "input": {"type": "field", "column": "close"},
                }
            )
        )
    # a swept threshold constant axis
    with pytest.raises(ValidationError, match="repeats the value"):
        Thesis.model_validate(
            {
                "name": "dup_const",
                "data": {"targets": ["target"]},
                "entry": {
                    "type": "threshold",
                    "left": {"type": "field", "column": "close"},
                    "op": "<",
                    "right": {"type": "constant", "name": "thr", "value": [95.5, 95.5]},
                },
            }
        )


def test_distinct_sweep_values_are_untouched():
    """The guard rejects repetition only — ordinary grids, including unsorted ones and a
    single-element list, still validate."""
    for horizon in ([21, 34], [34, 21], [21]):
        t = Thesis.model_validate({**NESTED_THESIS, "params": {"horizon": horizon}})
        assert t.params.horizon == horizon


# ---- sweep-axis-name collisions (parse-time, exit 3 not exit 4) ----------------------------------


def _thesis_swept_constant(
    name: str, *, left: dict | None = None, features: dict | None = None
) -> dict:
    d = {
        "name": "axis",
        "data": {"targets": ["target"]},
        "entry": {
            "type": "threshold",
            "left": left or {"type": "field", "column": "close"},
            "op": "<",
            "right": {"type": "constant", "value": [0.3, 0.35], "name": name},
        },
        "params": {"horizon": 5},
    }
    if features is not None:
        d["params"]["features"] = features
    return d


@pytest.mark.parametrize("reserved", ["target", "horizon"])
def test_thesis_rejects_reserved_sweep_axis_name(reserved):
    with pytest.raises(ValidationError, match="reserved"):
        Thesis.model_validate(_thesis_swept_constant(reserved))


def test_thesis_rejects_duplicate_sweep_axis_name():
    # A swept constant named after a co-occurring swept transform axis miscounts silently.
    left = {"type": "percentile", "window": [14, 21], "input": {"type": "field", "column": "close"}}
    with pytest.raises(ValidationError, match="duplicate sweep axis"):
        Thesis.model_validate(_thesis_swept_constant("percentile_window", left=left))


@pytest.mark.parametrize("col", ["ret", "ret_raw", "ret_bench", "mae", "is_open"])
def test_thesis_rejects_sweep_axis_colliding_with_trade_column(col):
    with pytest.raises(ValidationError, match="trade/feature"):
        Thesis.model_validate(_thesis_swept_constant(col))


def test_thesis_rejects_sweep_axis_colliding_with_default_feature():
    with pytest.raises(ValidationError, match="trade/feature"):
        Thesis.model_validate(_thesis_swept_constant("vol_14"))


def test_thesis_rejects_sweep_axis_colliding_with_declared_feature():
    d = _thesis_swept_constant("myfeat", features={"myfeat": {"type": "field", "column": "close"}})
    with pytest.raises(ValidationError, match="trade/feature"):
        Thesis.model_validate(d)


def test_thesis_accepts_a_distinct_swept_axis_name():
    # The happy path: a uniquely named swept constant beside a swept transform validates.
    left = {"type": "percentile", "window": [14, 21], "input": {"type": "field", "column": "close"}}
    t = Thesis.model_validate(_thesis_swept_constant("cutoff", left=left))
    assert t.entry.right.name == "cutoff"


# ---- strict types + frozen instances -----------------------------------------------------------


def _minimal_thesis_doc(**params) -> dict:
    return {
        "name": "strictness",
        "data": {"targets": ["target"]},
        "entry": {
            "type": "threshold",
            "left": {"type": "field", "column": "close"},
            "op": ">",
            "right": {"type": "constant", "value": 0.0},
        },
        "params": {"horizon": 5, **params},
    }


@pytest.mark.parametrize("window", ["3", True, 3.0])
def test_strict_types_refuse_coerced_int_params(window):
    # `"window": "3"` / `true` / `3.0` each spell a different document than the int the lax mode
    # would fabricate — strict types refuse instead of reinterpreting.
    with pytest.raises(ValidationError):
        EMA.model_validate({"type": "ema", "input": {"type": "field"}, "window": window})


@pytest.mark.parametrize("per_target", ["false", "true", 0, 1])
def test_strict_types_refuse_coerced_bools(per_target):
    with pytest.raises(ValidationError):
        ExternalFeed.model_validate({"per_target": per_target})


def test_strict_types_refuse_string_horizon():
    with pytest.raises(ValidationError):
        Thesis.model_validate(_minimal_thesis_doc(horizon="5"))


def test_strict_types_keep_int_for_float_legal():
    # JSON has one number type: an integer literal in a float position is the same claim.
    assert Constant.model_validate({"type": "constant", "value": 30}).value == 30.0


def test_validated_models_are_frozen():
    thesis = Thesis.model_validate(_minimal_thesis_doc())
    with pytest.raises(ValidationError):
        thesis.name = "renamed"
    with pytest.raises(ValidationError):
        thesis.params.horizon = -2
    # model_copy(update=...) is the sanctioned derive path; the original stays intact.
    derived = thesis.model_copy(update={"name": "derived"})
    assert derived.name == "derived" and thesis.name == "strictness"


# ---- canonical identity: one spelling per meaning ----------------------------------------------


def test_empty_features_dict_refuses():
    # `{}` would behave exactly like the omitted field (the built-in snapshots) while hashing
    # differently — one meaning, two identities. Omit the field for the defaults.
    with pytest.raises(ValidationError, match="non-empty"):
        Thesis.model_validate(_minimal_thesis_doc(features={}))


def test_scalar_constant_refuses_a_name():
    with pytest.raises(ValidationError, match="scalar constant takes no 'name'"):
        Constant.model_validate({"type": "constant", "value": 0.35, "name": "cut"})
    # The swept form still REQUIRES one — the name labels a real sweep axis there.
    with pytest.raises(ValidationError, match="requires a non-empty 'name'"):
        Constant.model_validate({"type": "constant", "value": [0.3, 0.35]})
    assert (
        Constant.model_validate({"type": "constant", "value": [0.3, 0.35], "name": "cut"}).name
        == "cut"
    )


def test_feed_used_only_as_outcome_series_is_referenced():
    # outcome.series naming a declared feed IS a use — the feed feeds the measurement.
    doc = _minimal_thesis_doc(outcome={"series": "vol", "kind": "diff"})
    doc["data"] = {"targets": ["target"], "external": {"vol": {}}}
    Thesis.model_validate(doc)


def test_feed_used_only_via_days_since_is_referenced():
    doc = {
        "name": "staleness",
        "data": {"targets": ["target"], "external": {"cpi": {}}},
        "entry": {
            "type": "threshold",
            "left": {"type": "days_since", "name": "cpi"},
            "op": "<=",
            "right": {"type": "constant", "value": 35.0},
        },
        "params": {"horizon": 5},
    }
    Thesis.model_validate(doc)


def test_feed_named_target_is_not_marked_used_by_the_default_outcome():
    # outcome.series == "target" always means the target itself, never a feed that happens to
    # carry that name — such a feed is unused and refuses.
    doc = _minimal_thesis_doc()
    doc["data"] = {"targets": ["px"], "external": {"target": {}}}
    with pytest.raises(ValidationError, match="never referenced"):
        Thesis.model_validate(doc)
