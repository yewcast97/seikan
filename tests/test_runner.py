"""End-to-end runner tests for the observer-native forward-return event study: per-firing-bar
overlapping observations, the horizon response curve, censoring, the next-open fill anchor,
external feeds, direction, the multi-target × multi-param × multi-horizon breakdown, the per-cell
panel over the DECLARED grid, and the per-cell overlap-aware reliability stats.

The runner is a per-hypothesis REPORTER: it selects nothing, crowns nothing and corrects nothing
for the size of the grid, so every assertion here reads a cell's own numbers or a run-level
geometry stamp — never a headline, a winner, or a holdout.

Two target modes: ``conjunction`` (the default) measures every target on its own — the targets are
the thesis's regime and the weakest one speaks for itself; ``target_mode: "basket"`` makes the
targets ONE cross-section per bar, so cross-sectional Series nodes are legal, ``benchmark:
"cross_mean"`` is available, and every cell carries a pooled cross-target block. The basket
sections at the bottom of this file pin that the pooled reads obey the same doctrines as
everything above: reconciling arithmetic, grid-composition independence, determinism, and
explicit refusals over silent absences.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from seikan.analysis.stats import benchmark_regression, episode_profile, pool_moments
from seikan.compiler.runner import run_backtest
from seikan.dsl.schema import Thesis
from seikan.serialize import json_safe
from tests._helpers import bars as _bars
from tests._helpers import lt as _lt

from ._helpers import load

_SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample.csv"


def _ohlc_bars(path: Path, rows: list[list[float]]) -> Path:
    """Write bars with distinct open/high/low/close (rows are [open, high, low, close])."""
    idx = pd.date_range("2021-01-01", periods=len(rows), freq="1D")
    df = pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"])
    df["volume"] = 1000.0
    df.index.name = "datetime"
    df.to_csv(path)
    return path


def _obs(path: Path, entry, closes, *, horizon=1, **params) -> pd.DataFrame:
    _bars(path, closes)
    params = {"horizon": horizon, **params}
    thesis = Thesis.model_validate(
        {"name": "t", "data": {"targets": ["target"]}, "entry": entry, "params": params}
    )
    return run_backtest(thesis, load(thesis, {"target": path})).trades


def _ohlc_run(path: Path, rows, entry, *, horizon=1, **params):
    """Run a hand-built distinct-OHLC thesis and return the FULL result (trades AND summary)."""
    _ohlc_bars(path, rows)
    params = {"horizon": horizon, **params}
    thesis = Thesis.model_validate(
        {"name": "s", "data": {"targets": ["target"]}, "entry": entry, "params": params}
    )
    return run_backtest(thesis, load(thesis, {"target": path}))


def _ohlc_obs(path: Path, rows, entry, *, horizon=1, **params) -> pd.DataFrame:
    return _ohlc_run(path, rows, entry, horizon=horizon, **params).trades


def _gt(v):
    return {
        "type": "threshold",
        "left": {"type": "field", "column": "close"},
        "op": ">",
        "right": {"type": "constant", "value": v},
    }


def _closed_n(summary: dict, cell_index: int = 0) -> int:
    """The total closed-observation count of one declared cell, across every target."""
    return sum(st["n"] for st in summary["cells"][cell_index]["by_target"].values())


# ---- forward-return measurement semantics ----------------------------------
# every observation anchors at the NEXT bar's open (open[t+1]→open[t+1+h]) — the only tradable
# convention. There is no `same_close` alternative: measuring from the close the decision itself
# consumed is not a fill anyone could have taken.


def test_unknown_fill_timing_key_is_rejected(tmp_path):
    # Unknown fields, `fill_timing` among them, are rejected under extra="forbid".
    _bars(tmp_path / "a.csv", [110, 95, 102, 108, 100])
    with pytest.raises(ValidationError):
        Thesis.model_validate(
            {
                "name": "s",
                "data": {"targets": ["target"]},
                "entry": _lt(100.0),
                "params": {"horizon": 2, "fill_timing": "same_close"},
            }
        )


def test_holdout_keys_are_rejected(tmp_path):
    # There is no split, so the DSL has no `oos_fraction` / `selection_mode`; under
    # extra="forbid" a thesis carrying either is invalid input, never silently ignored.
    _bars(tmp_path / "a.csv", [110, 95, 102, 108, 100])
    base = {"name": "s", "data": {"targets": ["target"]}, "entry": _lt(100.0)}
    for bad in ({"oos_fraction": 0.3}, {"selection_mode": "in_sample"}):
        with pytest.raises(ValidationError):
            Thesis.model_validate({**base, "params": {"horizon": 2, **bad}})


def test_next_open_fill_semantics(tmp_path):
    # Firing bar 1 fills at open[2], horizon 2 exits at open[4].
    tr = _obs(tmp_path / "a.csv", _lt(100.0), [110, 95, 102, 108, 100], horizon=2)
    assert len(tr) == 1
    row = tr.iloc[0]
    assert row.entry_px == 102.0 and row.exit_px == 100.0
    assert row.horizon == 2 and row.exit_reason == "horizon" and not row.is_open
    assert row.ret == pytest.approx(100.0 / 102.0 - 1.0)


def test_overlapping_observations_every_firing_bar(tmp_path):
    # entry fires on three CONSECUTIVE bars (1,2,3); horizon 2 → three independent OVERLAPPING
    # observations (a one-position state machine would gate these to fewer). Fills are
    # next-open: bar1→open[2]=96, bar2→open[3]=97, bar3→open[4]=130.
    tr = _obs(tmp_path / "ov.csv", _lt(100.0), [110, 95, 96, 97, 130, 130, 131], horizon=2)
    closed = tr[~tr["is_open"]]
    assert len(closed) == 3
    assert sorted(closed["entry_px"]) == [96.0, 97.0, 130.0]
    # bar1→open[4]=130, bar2→open[5]=130, bar3→open[6]=131
    assert set(closed["horizon"]) == {2}


def test_horizon_past_end_is_censored(tmp_path):
    # firing bars whose horizon runs past the data end are right-censored: flagged open, NaN ret,
    # excluded from the statistics. (Bar 3 fires but has no next bar to fill at — not recorded.)
    _bars(tmp_path / "cen.csv", [110, 95, 96, 97])
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": _lt(100.0),
            "params": {"horizon": 5},
        }
    )
    res = run_backtest(thesis, load(thesis, {"target": tmp_path / "cen.csv"}))
    tr = res.trades
    assert len(tr) == 2 and int(tr["is_open"].sum()) == 2
    assert (tr["exit_reason"] == "open").all()
    assert tr["ret"].isna().all()
    # Nothing closed → the declared cell is still reported, with an explicit zero and the two
    # censored firings visible in its ledger. A cell that measured nothing is evidence too.
    cell = res.summary["cells"][0]
    assert _closed_n(res.summary) == 0
    assert cell["outcome_coverage"]["target"]["n_attempted"] == 2
    assert cell["outcome_coverage"]["target"]["exit_reasons"]["open"] == 2


def test_no_signal_no_observations(tmp_path):
    tr = _obs(tmp_path / "f.csv", _lt(0.0), [100, 101, 102], horizon=1)
    assert len(tr) == 0 and list(tr.columns)  # schema columns still present


def test_no_observation_before_signal_initialized(tmp_path):
    rng = np.random.RandomState(0)
    closes = (100 + rng.randn(200).cumsum()).tolist()
    ema = {"type": "ema", "window": 20, "input": {"type": "field", "column": "close"}}
    entry = {
        "type": "threshold",
        "left": {"type": "field", "column": "close"},
        "op": "<",
        "right": ema,
    }
    tr = _obs(tmp_path / "e.csv", entry, closes, horizon=1)
    thesis = Thesis.model_validate(
        {"name": "t", "data": {"targets": ["target"]}, "entry": entry, "params": {"horizon": 1}}
    )
    md = load(thesis, {"target": tmp_path / "e.csv"})
    cutoff = md.index[19]  # EMA(20) initializes after 20 obs
    assert (tr["entry_time"] >= cutoff).all()


def test_rolling_negation_stays_gated_during_warmup(tmp_path):
    # Regression: `not(rolling(all, <warming condition>))` must NOT fire while the inner transform
    # is still warming up. The inner threshold is False during EMA(20)'s warmup, so rolling-all is
    # False and `not` is True — the rolling must stay UN-initialized (gated) until its inner is
    # initialized across the whole window, else the negation leaks observations onto warmup bars.
    rng = np.random.RandomState(1)
    closes = (100 + rng.randn(200).cumsum()).tolist()
    ema = {"type": "ema", "window": 20, "input": {"type": "field", "column": "close"}}
    inner = {
        "type": "threshold",
        "left": {"type": "field", "column": "close"},
        "op": "<",
        "right": ema,
    }
    entry = {
        "type": "not",
        "condition": {"type": "rolling", "window": 3, "agg": "all", "condition": inner},
    }
    tr = _obs(tmp_path / "rollneg.csv", entry, closes, horizon=1)
    thesis = Thesis.model_validate(
        {"name": "t", "data": {"targets": ["target"]}, "entry": entry, "params": {"horizon": 1}}
    )
    md = load(thesis, {"target": tmp_path / "rollneg.csv"})
    assert len(tr) > 0  # the negation does fire post-warmup — the test isn't vacuous
    assert (tr["entry_time"] >= md.index[19]).all()  # but never during EMA(20)'s warmup


# ---- next-open execution (the fill anchor) ----------------------------------


def test_next_open_fills_forward_at_open(tmp_path):
    # A firing bar t's observation runs open[t+1]→open[t+1+h]. entry close<100
    # fires at bar 1 → entry fills at open[2]=102 (idx[2]); horizon 1 → exit at open[3]=100
    # (idx[3]).
    rows = [
        [110, 110, 110, 110],
        [96, 96, 94, 95],
        [102, 109, 102, 108],
        [100, 100, 100, 100],
        [100, 100, 100, 100],
    ]
    tr = _ohlc_obs(tmp_path / "no.csv", rows, _lt(100.0), horizon=1)
    closed = tr[~tr["is_open"]]
    assert len(closed) == 1
    row = closed.iloc[0]
    idx = pd.date_range("2021-01-01", periods=len(rows), freq="1D")
    assert row.entry_time == idx[2] and row.entry_px == pytest.approx(102.0)
    assert row.exit_time == idx[3] and row.exit_px == pytest.approx(100.0)
    assert row.horizon == 1 and row.exit_reason == "horizon"
    assert row.ret == pytest.approx(100.0 / 102.0 - 1.0)


def test_next_open_anchors_one_bar_after_firing(tmp_path):
    # open==close here (via _bars): firing at bar 1, horizon 2 → entry at index[2], exit at
    # index[4].
    tr = _obs(tmp_path / "no2.csv", _lt(100.0), [110, 95, 102, 108, 100], horizon=2)
    assert len(tr) == 1
    row = tr.iloc[0]
    assert row.entry_time == pd.Timestamp("2021-01-03")  # index[2], one bar after the firing
    assert row.entry_px == pytest.approx(102.0) and row.exit_px == pytest.approx(100.0)
    assert row.horizon == 2 and row.ret == pytest.approx(100.0 / 102.0 - 1.0)


# ---- horizon response curve ------------------------------------------------


def test_horizon_sweep_is_response_curve(tmp_path):
    closes = (100 + np.random.RandomState(4).randn(200).cumsum()).tolist()
    _bars(tmp_path / "hz.csv", closes)
    ema = {"type": "ema", "window": 20, "input": {"type": "field", "column": "close"}}
    thesis = Thesis.model_validate(
        {
            "name": "hz",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "threshold",
                "left": {"type": "field", "column": "close"},
                "op": "<",
                "right": ema,
            },
            "params": {"horizon": [1, 5, 20]},
        }
    )
    res = run_backtest(thesis, load(thesis, {"target": tmp_path / "hz.csv"}))
    s = res.summary
    assert s["params"] == ["horizon"]
    # each declared horizon is one hypothesis and one cell — the response curve IS the grid
    assert s["n_hypotheses_attempted"] == 3 == len(s["cells"])
    assert [c["params"]["horizon"] for c in s["cells"]] == [1, 5, 20]
    assert set(res.trades["horizon"]) == {1, 5, 20}  # every row names its measurement window


# ---- sample-data end to end + per-cell panel + reliability ------------------


@pytest.fixture()
def thesis() -> Thesis:
    ema = {"type": "ema", "window": 20, "input": {"type": "field", "column": "close"}}
    return Thesis.model_validate(
        {
            "name": "ema_meanrev",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "threshold",
                "left": {"type": "field", "column": "close"},
                "op": "<",
                "right": ema,
            },
            "params": {"horizon": 10},
        }
    )


def test_run_backtest_end_to_end(thesis):
    res = run_backtest(thesis, load(thesis, {"target": _SAMPLE}))
    for col in (
        "target",
        "entry_time",
        "exit_time",
        "entry_px",
        "exit_px",
        "horizon",
        "ret",
        "exit_reason",
        "is_open",
    ):
        assert col in res.trades.columns
    assert "ret_5" in res.trades.columns and "ret_20" in res.trades.columns
    # No epoch-ns twins: the ISO times are the record, and an int64 nanosecond count silently
    # rounds in any IEEE-754 consumer.
    assert "entry_ts" not in res.trades.columns and "exit_ts" not in res.trades.columns

    s = res.summary
    assert (res.trades["horizon"] == 10).all()  # horizon is the measurement window, always named
    assert list(res.trades.columns[:2]) == ["horizon", "target"]  # unswept: still the identity
    # single target, scalar horizon → one declared cell
    assert s["params"] == [] and s["targets"] == ["target"]
    assert s["n_hypotheses_attempted"] == 1 and len(s["cells"]) == 1

    cell = s["cells"][0]
    assert cell["params"] == {"horizon": 10} and cell["cell_id"] == "horizon=10"
    st = cell["by_target"]["target"]
    assert set(st) == {
        "n",
        "n_nonoverlap",
        "mean_ret",
        "mean_ret_raw",
        "mean_ret_bench",
        "benchmark_regression",
        "hit_rate",
        "win_loss_ratio",
        "profit_factor",
        "std_ret",
        "skewness",
        "kurtosis",
        "tail_ratio",
        "cvar_5",
        "t_hac",
        "hac_se",
        "rot_p",
        "rot_n_null",
        "concentration",
        "boot",
        "subperiods",
        # evidence: distribution shape and holding-period path
        "ret_quantiles",
        "worst_ret",
        "best_ret",
        "mae_quantiles",
        "mfe_quantiles",
        "edge_ratio",
        "timing",
    }
    assert st["n"] > 0 and math.isfinite(st["mean_ret"]) and 0.0 <= st["hit_rate"] <= 1.0
    assert st["n_nonoverlap"] <= st["n"]  # overlap can only shrink the effective sample
    assert 0.0 <= st["rot_p"] <= 1.0
    # the attribution legs: unbenchmarked run, so the raw leg IS the return and the bench leg is
    # null — "no benchmark", never a measured zero
    assert st["mean_ret_raw"] == st["mean_ret"] and math.isnan(st["mean_ret_bench"])
    # the bootstrap / subperiod blocks reconcile with the pool they describe
    assert st["boot"]["method"] == "episode_percentile"
    assert sum(seg["n"] for seg in st["subperiods"]) == st["n"]
    # the quantile blocks: seven points each, ordered, and the excursion pools carry their own
    # count and subset mean
    assert set(st["ret_quantiles"]) == {"p05", "p10", "p25", "p50", "p75", "p90", "p95"}
    assert set(st["mae_quantiles"]) == {
        "n",
        "mean",
        "p05",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "worst",
    }
    assert set(st["mfe_quantiles"]) == {
        "n",
        "mean",
        "p05",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "best",
    }
    assert st["mae_quantiles"]["n"] <= st["n"] and st["mfe_quantiles"]["n"] <= st["n"]
    assert st["worst_ret"] <= st["best_ret"]
    assert set(cell["outcome_coverage"]["target"]["exit_reasons"]) == {
        "horizon",
        "open",
        "no_outcome",
        "no_benchmark",
    }

    # no HAC p, no iid pair, and no search-adjusted statistics anywhere: the caller derives a p
    # at df = n_nonoverlap − 1 if it wants one and prices multiplicity itself
    for absent in ("p_hac", "t_iid", "p_iid", "p_fdr", "p_rw", "sharpe", "firing_rate"):
        assert absent not in st


def test_returns_are_raw_open_to_open(thesis):
    tr = run_backtest(thesis, load(thesis, {"target": _SAMPLE})).trades
    closed = tr[~tr["is_open"]]
    expected = closed["exit_px"] / closed["entry_px"] - 1.0
    assert (closed["ret"] - expected).abs().max() < 1e-9


def test_direction_recorded_in_summary(thesis):
    s = run_backtest(thesis, load(thesis, {"target": _SAMPLE})).summary
    assert s["direction"] == "longonly"


# ---- the per-cell panel over the DECLARED grid ------------------------------


def _noise_px(path: Path, n: int = 900, seed: int = 0) -> Path:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="1D")
    s = pd.Series(100 * np.exp(np.cumsum(rng.randn(n) * 0.01)), index=idx)
    o = s.shift(1).bfill()
    df = pd.DataFrame(
        {
            "open": o,
            "high": np.maximum(s * 1.01, o),
            "low": np.minimum(s * 0.99, o),
            "close": s,
            "volume": 1000.0,
        },
        index=idx,
    )
    df.index.name = "datetime"
    df.to_csv(path)
    return path


def _dead_combo_thesis() -> Thesis:
    """A 3 × 2 grid whose ``cut=-1.0`` combos can NEVER fire (a percentile is always ≥ 0)."""
    return Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "threshold",
                "left": {
                    "type": "percentile",
                    "window": 14,
                    "input": {"type": "field", "column": "close"},
                },
                "op": "<",
                "right": {"type": "constant", "value": [0.30, 0.35, -1.0], "name": "cut"},
            },
            "params": {"horizon": [5, 10]},
        }
    )


def test_cells_cover_the_declared_grid_including_combos_that_never_fire(tmp_path):
    # THE honesty invariant of this identity. A groupby over the trades frame cannot invent a row
    # for a combo with no observations, and the reliability pass only ever sees cells that fired —
    # so a panel driven off either would silently DELETE every non-firing hypothesis, which is
    # precisely the hypothesis whose absence makes a surviving cell look inevitable.
    _noise_px(tmp_path / "px.csv")
    thesis = _dead_combo_thesis()
    s = run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"})).summary
    assert s["n_hypotheses_attempted"] == 6 == len(s["cells"])
    by_id = {c["cell_id"]: c for c in s["cells"]}
    assert set(by_id) == {
        "cut=0.3,horizon=5",
        "cut=0.3,horizon=10",
        "cut=0.35,horizon=5",
        "cut=0.35,horizon=10",
        "cut=-1,horizon=5",
        "cut=-1,horizon=10",
    }
    # the two dead combos are on the record as EXPLICIT zeros, not as absences
    for cid in ("cut=-1,horizon=5", "cut=-1,horizon=10"):
        dead = by_id[cid]
        assert dead["params"]["cut"] == -1.0
        st = dead["by_target"]["target"]
        assert st["n"] == 0 and st["n_nonoverlap"] == 0
        assert math.isnan(st["mean_ret"]) and math.isnan(st["concentration"]["top_share_abs"])
        assert dead["outcome_coverage"]["target"]["n_attempted"] == 0
        assert dead["episode_stats"]["n"] == 0
        # signal_coverage is combo geometry, not a firing count — it is reported regardless
        assert dead["signal_coverage"]["target"]["n_bars"] == s["n_bars"]
        # the evidence blocks keep the SAME keys on a dead cell — nulls with a reason, and
        # the subperiod eras are run geometry, so their timestamps are real even here
        assert st["boot"]["reason"] == "no_observations" and st["boot"]["ci_lo"] is None
        assert len(st["subperiods"]) == 3
        assert all(seg["n"] == 0 and seg["mean_ret"] is None for seg in st["subperiods"])
        assert all(seg["start"] and seg["end"] for seg in st["subperiods"])
    # and the live combos did measure something, so the test is not vacuous
    assert by_id["cut=0.3,horizon=5"]["by_target"]["target"]["n"] > 0


def test_boot_blocks_are_grid_composition_independent(tmp_path):
    # Content-seeded bootstrap: adding a combo to the grid changes no sibling
    # cell's draws. Positional seeding would re-jitter every cell whenever the sweep grew — the
    # same doctrine "adding a cell changes no other cell's numbers" the rotation/HAC pass keeps.
    _noise_px(tmp_path / "px.csv")

    def _sweep(cuts):
        return Thesis.model_validate(
            {
                "name": "t",
                "data": {"targets": ["target"]},
                "entry": {
                    "type": "threshold",
                    "left": {
                        "type": "percentile",
                        "window": 14,
                        "input": {"type": "field", "column": "close"},
                    },
                    "op": "<",
                    "right": {"type": "constant", "value": cuts, "name": "cut"},
                },
                "params": {"horizon": [5, 10]},
            }
        )

    t2, t3 = _sweep([0.30, 0.35]), _sweep([0.30, 0.35, 0.40])
    s2 = run_backtest(t2, load(t2, {"target": tmp_path / "px.csv"})).summary
    s3 = run_backtest(t3, load(t3, {"target": tmp_path / "px.csv"})).summary
    by3 = {c["cell_id"]: c for c in s3["cells"]}
    for c2 in s2["cells"]:
        assert (
            by3[c2["cell_id"]]["by_target"]["target"]["boot"] == c2["by_target"]["target"]["boot"]
        )


def test_bar_spacing_stamp_daily_and_irregular(tmp_path):
    px = _bars(tmp_path / "px.csv", [100, 101, 102, 103, 104, 105])
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": _lt(1000.0),
            "params": {"horizon": 1},
        }
    )
    s = run_backtest(thesis, load(thesis, {"target": px})).summary
    assert s["bar_spacing"] == {"min_seconds": 86400, "median_seconds": 86400, "max_seconds": 86400}

    # A calendar gap widens max_seconds; min/median stay the modal day. Pure self-description —
    # the engine still never interprets cadence.
    idx = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-06", "2021-01-07"])
    px = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=idx)
    df = pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0}, index=idx)
    df.index.name = "datetime"
    df.to_csv(tmp_path / "gap.csv")
    thesis2 = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": _lt(1000.0),
            "params": {"horizon": 1},
        }
    )
    s2 = run_backtest(thesis2, load(thesis2, {"target": tmp_path / "gap.csv"})).summary
    assert s2["bar_spacing"] == {
        "min_seconds": 86400,
        "median_seconds": 86400,
        "max_seconds": 3 * 86400,
    }

    from seikan.dataio import bar_spacing

    assert bar_spacing(pd.DatetimeIndex([])) == {
        "min_seconds": None,
        "median_seconds": None,
        "max_seconds": None,
    }
    assert bar_spacing(pd.to_datetime(["2021-01-01"]))["median_seconds"] is None


def test_cell_count_equals_the_declared_grid_across_shapes(tmp_path):
    # len(cells) == n_hypotheses_attempted by construction, whatever the grid's shape — including
    # the degenerate one-cell run whose horizon was never swept.
    _noise_px(tmp_path / "px.csv", n=400)
    grids = [
        ({"value": 0.35}, 5, 1),  # scalar constants carry no name — nothing to label
        ({"value": [0.30, 0.35], "name": "cut"}, 5, 2),
        ({"value": [0.30, 0.35], "name": "cut"}, [3, 5, 8], 6),
    ]
    for const, horizon, expected in grids:
        thesis = Thesis.model_validate(
            {
                "name": "t",
                "data": {"targets": ["target"]},
                "entry": {
                    "type": "threshold",
                    "left": {
                        "type": "percentile",
                        "window": 14,
                        "input": {"type": "field", "column": "close"},
                    },
                    "op": "<",
                    "right": {"type": "constant", **const},
                },
                "params": {"horizon": horizon},
            }
        )
        s = run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"})).summary
        assert s["n_hypotheses_attempted"] == expected == len(s["cells"])


def test_cell_identity_carries_the_horizon_even_when_unswept(tmp_path):
    # A cell that does not name its measurement window is not reproducible from the report, so the
    # horizon rides in `params` (and last in `cell_id`) whether or not it was a sweep axis.
    _noise_px(tmp_path / "px.csv", n=400)
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "threshold",
                "left": {
                    "type": "percentile",
                    "window": [14, 21],
                    "input": {"type": "field", "column": "close"},
                },
                "op": "<",
                "right": {"type": "constant", "value": 0.35},
            },
            "params": {"horizon": 7},
        }
    )
    s = run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"})).summary
    assert s["params"] == ["percentile_window"]  # horizon is not a SWEEP axis here…
    assert [c["cell_id"] for c in s["cells"]] == [
        "percentile_window=14,horizon=7",
        "percentile_window=21,horizon=7",
    ]
    for c in s["cells"]:  # …but it is always part of the cell's identity
        assert c["params"]["horizon"] == 7


def test_cell_order_and_ids_are_deterministic_across_runs(tmp_path):
    # Two identical runs must emit byte-identical cell identities in byte-identical order: the
    # panel is the report's spine, and a caller aligning gate["cells"][i] with summary["cells"][i]
    # depends on the position being stable.
    _noise_px(tmp_path / "px.csv", n=500)
    thesis = _dead_combo_thesis()
    md = load(thesis, {"target": tmp_path / "px.csv"})
    a = run_backtest(thesis, md).summary["cells"]
    b = run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"})).summary["cells"]
    assert [c["cell_id"] for c in a] == [c["cell_id"] for c in b]
    assert [c["params"] for c in a] == [c["params"] for c in b]
    assert [c["by_target"]["target"]["n"] for c in a] == [c["by_target"]["target"]["n"] for c in b]


def test_cell_panels_reconcile_with_the_trades_frame(tmp_path):
    # Every panel of a cell describes the SAME rows, and each is arithmetic a reader re-checks
    # against the frame: the graded n is the ledger's closed count, the reasons sum to the
    # attempted count, and the episode panel spans the per-target total.
    _noise_px(tmp_path / "px.csv")
    thesis = _dead_combo_thesis()
    res = run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"}))
    s = res.summary
    for cell in s["cells"]:
        rows = res.trades
        for axis, value in cell["params"].items():
            if axis in rows.columns:
                rows = rows[rows[axis] == value]
        for tgt in s["targets"]:
            rows_t = rows[rows["target"] == tgt]
            cov = cell["outcome_coverage"][tgt]
            assert cov["n_attempted"] == len(rows_t)
            assert sum(cov["exit_reasons"].values()) == cov["n_attempted"]
            assert cov["n_closed"] == cov["exit_reasons"]["horizon"]
            assert cov["n_closed"] == int((~rows_t["is_open"]).sum())
            st = cell["by_target"][tgt]
            assert st["n"] == cov["n_closed"]
            assert st["n_nonoverlap"] <= st["n"]
        assert cell["episode_stats"]["n"] == sum(st["n"] for st in cell["by_target"].values())


def test_by_target_quantile_blocks_are_the_trades_frame_read_back(tmp_path):
    # The quantile blocks are AGGREGATIONS OF ROWS A READER ALREADY HAS, so every point must
    # be re-derivable from the frame the same run wrote — otherwise the panel is a second opinion
    # about the same pool instead of a summary of it — all SEVEN points, p05 and p95 included;
    # the excursion blocks carry their OWN n because a window hole can censor `mae`/`mfe` on a
    # row whose `ret` closed, and that count must be the frame's.
    _noise_px(tmp_path / "px.csv")
    thesis = _dead_combo_thesis()
    res = run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"}))
    s = res.summary
    graded = 0
    for cell in s["cells"]:
        rows = res.trades
        for axis, value in cell["params"].items():
            if axis in rows.columns:
                rows = rows[rows[axis] == value]
        for tgt in s["targets"]:
            st = cell["by_target"][tgt]
            if st["n"] == 0:
                continue
            graded += 1
            closed = rows[(rows["target"] == tgt) & (~rows["is_open"])]
            pool = closed["ret"].to_numpy(dtype=float)
            assert pool.size == st["n"]
            q = st["ret_quantiles"]
            points = ["p05", "p10", "p25", "p50", "p75", "p90", "p95"]
            expected = np.percentile(pool, [5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0])
            assert [q[k] for k in points] == pytest.approx(expected)
            assert q["p05"] <= q["p10"] <= q["p25"] <= q["p50"] <= q["p75"] <= q["p90"] <= q["p95"]
            assert st["worst_ret"] == pytest.approx(float(pool.min()))
            for name, tail in (("mae", "worst"), ("mfe", "best")):
                col = closed[name].to_numpy(dtype=float)
                finite = col[np.isfinite(col)]
                blk = st[f"{name}_quantiles"]
                assert blk["n"] == finite.size
                assert blk["n"] <= st["n"]  # a censored path row is dropped, never zero-filled
                assert [blk["p10"], blk["p25"], blk["p50"], blk["p75"], blk["p90"]] == (
                    pytest.approx(np.percentile(finite, [10.0, 25.0, 50.0, 75.0, 90.0]))
                )
                extreme = float(finite.min()) if tail == "worst" else float(finite.max())
                assert blk[tail] == pytest.approx(extreme)
    assert graded == 4  # 2 live cuts × 2 horizons; the dead cut contributes no pool


def test_panel_shape_reads_are_pool_moments_over_the_same_pool(tmp_path):
    # The shape/dispersion keys are `pool_moments` over the cell's closed slice of the trades
    # frame — exact, not approximate. Anything else would be a second opinion about one pool.
    _noise_px(tmp_path / "px.csv")
    thesis = _dead_combo_thesis()
    res = run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"}))
    s = res.summary
    promoted = ("win_loss_ratio", "std_ret", "skewness", "kurtosis", "tail_ratio", "cvar_5")
    graded = 0
    for cell in s["cells"]:
        rows = res.trades
        for axis, value in cell["params"].items():
            if axis in rows.columns:
                rows = rows[rows[axis] == value]
        for tgt in s["targets"]:
            st = cell["by_target"][tgt]
            if st["n"] == 0:
                continue
            graded += 1
            closed = rows[(rows["target"] == tgt) & (~rows["is_open"])]
            ref = pool_moments(closed["ret"].to_numpy(dtype=float))
            for key in promoted:
                if math.isnan(ref[key]):
                    assert math.isnan(st[key])
                else:
                    assert st[key] == ref[key]  # the same kernel over the same rows — exact
    assert graded == 4


def test_episode_profile_is_the_trades_frame_read_back(tmp_path):
    # Like the quantile blocks, the episode profile is an aggregation of rows a reader already
    # has: recomputing it from the cell's closed rows must reproduce the emitted block exactly
    # (compared through json_safe, since NaN != NaN in a raw dict comparison).
    _noise_px(tmp_path / "px.csv")
    thesis = _dead_combo_thesis()
    res = run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"}))
    checked = 0
    for cell in res.summary["cells"]:
        rows = res.trades
        for axis, value in cell["params"].items():
            if axis in rows.columns:
                rows = rows[rows[axis] == value]
        closed = rows[~rows["is_open"]] if not rows.empty else rows
        assert json_safe(cell["episode_profile"]) == json_safe(episode_profile(closed))
        assert cell["episode_profile"]["n_episodes"] == cell["episode_stats"]["n_clusters"]
        checked += 1
    assert checked == len(res.summary["cells"])


def test_benchmark_regression_panel_reads_the_frame_legs_back(tmp_path):
    # Benchmarked run: the panel block must equal a from-scratch regression over the frame's
    # own row-aligned legs, and the alpha identity must reconcile with the panel's leg means.
    tgt = 100 + np.random.RandomState(21).randn(160).cumsum()
    bench = 50 + np.random.RandomState(22).randn(160).cumsum() * 0.3
    res = _bench_run(tmp_path, tgt, bench, benchmark="market")
    st = res.summary["cells"][0]["by_target"]["target"]
    closed = res.trades[~res.trades["is_open"]]
    ref = benchmark_regression(
        closed["ret_raw"].to_numpy(dtype=float), closed["ret_bench"].to_numpy(dtype=float)
    )
    assert st["benchmark_regression"] == ref
    blk = st["benchmark_regression"]
    assert blk["reason"] is None and blk["n"] == st["n"]
    assert blk["alpha"] + blk["beta"] * st["mean_ret_bench"] == pytest.approx(
        st["mean_ret_raw"], abs=1e-15
    )
    # unbenchmarked: the all-NaN bench leg declines with the stamped reason, never a zero fit
    raw = _bench_run(tmp_path, tgt)
    blk_raw = raw.summary["cells"][0]["by_target"]["target"]["benchmark_regression"]
    assert blk_raw == {
        "n": 0,
        "beta": None,
        "alpha": None,
        "r2": None,
        "reason": "no_paired_observations",
    }


def test_horizon_siblings_repeat_the_same_signal_coverage(tmp_path):
    # The undefined-decision ledger is keyed by COMBO: the entry condition is what a missing input
    # renders undecidable, and the measurement horizon has no say in it. Horizon siblings therefore
    # legitimately report identical counts — each cell is graded alone, nothing is summed across.
    _noise_px(tmp_path / "px.csv", n=400)
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "threshold",
                "left": {
                    "type": "percentile",
                    "window": 14,
                    "input": {"type": "field", "column": "close"},
                },
                "op": "<",
                "right": {"type": "constant", "value": 0.35},
            },
            "params": {"horizon": [3, 9]},
        }
    )
    s = run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"})).summary
    a, b = s["cells"]
    assert a["params"]["horizon"] != b["params"]["horizon"]
    assert a["signal_coverage"] == b["signal_coverage"]


def test_statistics_version_and_basis_are_stamped(tmp_path):
    _noise_px(tmp_path / "px.csv", n=400)
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "threshold",
                "left": {
                    "type": "percentile",
                    "window": 14,
                    "input": {"type": "field", "column": "close"},
                },
                "op": "<",
                "right": {"type": "constant", "value": 0.35},
            },
            "params": {"horizon": 5},
        }
    )
    s = run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"})).summary
    assert s["statistics_version"] == 4
    # Every cell is measured once over the whole index — stamped so a consumer verifies the basis
    # it is reading instead of assuming it.
    assert s["gate_evidence_basis"] == "full_sample"
    # the rotation stamp: every non-identity shift; no "requested" knob and no selection "basis"
    assert set(s["rotation"]) == {"n_shifts", "p_resolution"}
    assert s["rotation"]["n_shifts"] == s["n_bars"] - 1
    assert s["rotation"]["p_resolution"] == pytest.approx(1.0 / (1.0 + s["rotation"]["n_shifts"]))


def test_geometry_stamps_describe_the_evaluated_interval(tmp_path):
    # The bookkeeping any cross-run search discipline needs and a stateless verifier cannot itself
    # perform: two runs over different data windows must be tellable apart from the report alone.
    _noise_px(tmp_path / "px.csv", n=400)
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": _gt(0.0),
            "params": {"horizon": 5},
        }
    )
    md = load(thesis, {"target": tmp_path / "px.csv"})
    s = run_backtest(thesis, md).summary
    assert s["n_bars"] == len(md.index) == 400
    assert s["index_start"] == md.index[0].isoformat()
    assert s["index_end"] == md.index[-1].isoformat()


_FORBIDDEN_SUMMARY_KEYS = (
    # pooled headline scalars spread across the top level — a grid has no single result, and
    # publishing one invites reading the best cell as the run's answer
    "n",
    "n_valid",
    "mean_ret",
    "median_ret",
    "std_ret",
    "hit_rate",
    "t_iid",
    "p_iid",
    "win_loss_ratio",
    "skewness",
    "kurtosis",
    "tail_ratio",
    "cvar_5",
    "mean_bars_held",
    "median_bars_held",
    "max_bars_held",
    "exit_reasons",
    # the retired grid breakdown (schema v5): a fired-pools-only duplicate of cells[].by_target,
    # rollups that averaged per-cell means across horizons, and the derivable per-cell twins
    "stats_table",
    "by_target",
    "by_param",
    "n_stats_rows",
    "sharpe",
    "firing_rate",
    # the CSCV/PBO family is emitted as ONE nested summary["pbo"] block, never as flat top-level
    # keys (bare "pbo" is absent from this list — it names the nested block)
    "pbo_reason",
    "pbo_n_splits",
    "pbo_n_combos",
    "pbo_blocks",
    "lambda_mean",
    "oos_degradation_slope",
    "oos_degradation_slope_reason",
    "prob_oos_loss",
    # search-adjusted statistics: multiplicity is the caller's to price, against
    # n_hypotheses_attempted
    "fw_p",
    "n_hypotheses",
    "romano_wolf",
    "deflation",
    "mean_ret_deflated",
    # selection: the engine crowns nobody
    "best_cell",
    "best_cell_by_target",
    "binding_target",
    "selection_mode",
    "selection_mode_requested",
    # an in-sample / out-of-sample split and everything that would read it
    "oos_fraction",
    "oos_cut_time",
    "oos_mean",
    "in_sample",
    "out_of_sample",
    "out_of_sample_by_target",
    "out_of_sample_pooled",
    "in_sample_by_target",
    "in_sample_episode_stats",
    # panels that belong to the per-cell record (a top-level copy would describe a winner)
    "outcome_coverage",
    "signal_coverage",
    "episode_stats",
    "concentration",
    # run-level pooled conditional panels: a pooled qcut re-cuts the same bar once per
    # combo × horizon, so its numbers move with GRID COMPOSITION — the panels live per cell over
    # the cell's own closed rows, and a run-level one would be exactly that
    # composition-dependent read
    "conditional_buckets",
    "bucket_monotonicity",
    # winner-only evidence panels
    "psr",
    "psr_worst_target",
    "sr",
    "sr_se",
    "dsr",
    "expected_max_sr",
    "dsr_n_trials",
    "n_nonoverlap_required",
    "n_eff_required",
    "edge_stability",
    "pre_entry_drift",
    "path_risk",
    "cost_robustness",
    "lag_sensitivity",
    "horizon_coherence",
    "mean_ret_ci95",
    "null_mean",
    "null_std",
    "null_p",
    # the retired v2 spelling of the non-overlap count — renamed n_nonoverlap in schema v3,
    # so the old key reappearing anywhere would be a silent contract fork
    "n_eff",
)


def test_forbidden_summary_keys_never_appear(tmp_path):
    # A pin, not a smoke test: each of these names a selected cell, a search correction, or a
    # holdout — concepts this identity does not have. One appearing means the report has started
    # answering a question the engine does not ask.
    _noise_px(tmp_path / "px.csv", n=500)
    thesis = _dead_combo_thesis()
    s = run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"})).summary
    leaked = sorted(k for k in _FORBIDDEN_SUMMARY_KEYS if k in s)
    assert leaked == []


def test_summary_is_json_safe_through_serialize(tmp_path):
    from seikan.serialize import serialize_result

    _noise_px(tmp_path / "px.csv", n=500)
    thesis = _dead_combo_thesis()
    payload = serialize_result(run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"})))

    def ok(o) -> bool:
        if isinstance(o, dict):
            return all(isinstance(k, str) and ok(v) for k, v in o.items())
        if isinstance(o, list):
            return all(ok(v) for v in o)
        return o is None or isinstance(o, (str, int, float))

    assert ok(payload["summary"])  # NaN → None, no numpy/Timestamp survives serialization
    cells = payload["summary"]["cells"]
    assert len(cells) == payload["summary"]["n_hypotheses_attempted"]
    # the non-firing cells serialize as explicit nulls, never as dropped entries
    dead = [c for c in cells if c["by_target"]["target"]["n"] == 0]
    assert dead and all(c["by_target"]["target"]["mean_ret"] is None for c in dead)


def test_dead_cell_quantile_blocks_serialize_as_explicit_nulls(tmp_path):
    # The quantile blocks obey the same rule as every other evidence panel: a cell that never fired
    # keeps ALL FOUR keys and fills them with nulls, because a dropped block reads as "not
    # applicable to this cell" while a zero reads as a measured flat path. An empty excursion pool
    # says so twice — `n == 0` AND null at every point — so a consumer never has to infer emptiness
    # from a quantile it cannot tell apart from a real one.
    from seikan.serialize import serialize_result

    _noise_px(tmp_path / "px.csv", n=500)
    thesis = _dead_combo_thesis()
    payload = serialize_result(run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"})))
    dead = [c for c in payload["summary"]["cells"] if c["by_target"]["target"]["n"] == 0]
    assert len(dead) == 2  # the cut=-1.0 horizon siblings, which a percentile can never satisfy
    points = ("p05", "p10", "p25", "p50", "p75", "p90", "p95")
    for cell in dead:
        st = cell["by_target"]["target"]
        assert set(st["ret_quantiles"]) == set(points)
        assert all(st["ret_quantiles"][p] is None for p in points)
        assert st["worst_ret"] is None
        for name, tail in (("mae_quantiles", "worst"), ("mfe_quantiles", "best")):
            blk = st[name]
            assert blk["n"] == 0
            assert all(blk[p] is None for p in points)
            assert blk[tail] is None
        # the v2 evidence obeys the same doctrine on a dead cell: null-with-reason, zero-count
        # blocks, and the explicit zero episode profile — never an omitted key
        assert st["benchmark_regression"] == {
            "n": 0,
            "beta": None,
            "alpha": None,
            "r2": None,
            "reason": "no_paired_observations",
        }
        assert st["timing"]["n_to_positive"] == 0 and st["timing"]["n_to_trough"] == 0
        assert st["timing"]["median_bars_to_positive"] is None
        for key in ("win_loss_ratio", "std_ret", "skewness", "kurtosis", "tail_ratio", "cvar_5"):
            assert st[key] is None
        prof = cell["episode_profile"]
        assert prof["n_episodes"] == 0
        assert prof["max_win_streak"] == prof["max_loss_streak"] == 0
        assert prof["hit_rate"] is None and prof["mean_ret"] is None


def test_grid_level_evidence_rides_the_run_not_a_cell(tmp_path):
    # CSCV/PBO is a property of the SEARCH SPACE the caller is about to select from, attached to
    # no hypothesis and read by no cell's checklist.
    _noise_px(tmp_path / "px.csv")
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "threshold",
                "left": {
                    "type": "percentile",
                    "window": 14,
                    "input": {"type": "field", "column": "close"},
                },
                "op": "<",
                # The cut pair must STRADDLE an attainable percentile value (the window-14 grid
                # is k/14, so 4/14 ≈ 0.286 sits between these): cuts that share a grid gap —
                # 0.30 with 0.35, say — produce byte-identical masks, and the v3 duplicate
                # collapse would rightly report 3 distinct candidates for the 6 declared.
                "right": {"type": "constant", "value": [0.25, 0.35], "name": "cut"},
            },
            "params": {"horizon": [5, 10, 20]},
        }
    )
    s = run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"})).summary
    assert set(s["pbo"]) == {
        "pbo",
        "reason",
        "n_splits",
        "n_splits_attempted",
        "n_candidates_min",
        "n_combos",
        "n_combos_scoreable",
        "n_combos_declared",
        "blocks",
        "lambda_mean",
        "oos_degradation_slope",
        "oos_degradation_slope_reason",
        "prob_oos_loss",
    }
    # The population ledger: every declared combo fired and closed here and no two are
    # byte-identical, so declared == scoreable == distinct.
    assert s["pbo"]["n_combos"] == 6  # 2 cuts × 3 horizons
    assert s["pbo"]["n_combos_scoreable"] == 6
    assert s["pbo"]["n_combos_declared"] == 6 == s["n_hypotheses_attempted"]
    assert all("pbo" not in cell for cell in s["cells"])


def test_unswept_thesis_pbo_is_single_combo(tmp_path):
    _noise_px(tmp_path / "px.csv")
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "threshold",
                "left": {
                    "type": "percentile",
                    "window": 14,
                    "input": {"type": "field", "column": "close"},
                },
                "op": "<",
                "right": {"type": "constant", "value": 0.35},
            },
            "params": {"horizon": 5},
        }
    )
    s = run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"})).summary
    # a one-combo grid has no selection to overfit
    assert s["pbo"]["pbo"] is None and s["pbo"]["reason"] == "single_combo"
    assert s["pbo"]["n_combos_declared"] == 1 == s["n_hypotheses_attempted"]


def test_n_hypotheses_attempted_counts_the_declared_grid(tmp_path):
    # The attempted count is the DECLARED grid (combos × horizons) — a cut that never fires still
    # counts, so the search burden the caller must price cannot be shrunk by non-firing combos.
    _noise_px(tmp_path / "px.csv")
    thesis = _dead_combo_thesis()
    res = run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"}))
    s = res.summary
    assert s["n_hypotheses_attempted"] == 6  # 3 cuts × 2 horizons declared
    populated = sum(1 for c in s["cells"] if any(st["n"] > 0 for st in c["by_target"].values()))
    assert populated == 4  # only 2 cuts ever fire → 4 populated cells
    assert len(s["cells"]) == 6  # …but every declared cell is still reported


# ---- external feeds, direction, multi-target × multi-param -----------------


def test_external_feed_end_to_end(tmp_path):
    idx = pd.date_range("2020-01-01", periods=160, freq="D")
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, len(idx)))
    pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=idx
    ).to_csv(tmp_path / "px.csv")
    rate = pd.Series(np.sin(np.arange(len(idx)) / 3.0), index=idx, name="rate")
    rate.to_frame().to_csv(tmp_path / "rate.csv")
    zscore = {"type": "zscore", "window": 20, "input": {"type": "external", "name": "rate"}}
    thesis = Thesis.model_validate(
        {
            "name": "ext_meanrev",
            "data": {"targets": ["target"], "external": {"rate": {}}},
            "entry": {
                "type": "threshold",
                "left": zscore,
                "op": "<",
                "right": {"type": "constant", "value": -0.5},
            },
            "params": {"horizon": 10},
        }
    )
    files = {"target": tmp_path / "px.csv", "rate": tmp_path / "rate.csv"}
    res = run_backtest(thesis, load(thesis, files))
    s = res.summary
    assert _closed_n(s) > 0
    reasons = s["cells"][0]["outcome_coverage"]["target"]["exit_reasons"]
    assert reasons["no_outcome"] == 0 and reasons["no_benchmark"] == 0  # complete series
    # the feed is a raw decision leaf: its availability is accounted for run-level
    assert "external:rate" in s["sources"]["target"]["by_source"]


def test_rolling_corr_entry_end_to_end(tmp_path):
    # A thesis entry keyed on rolling_corr(close change, an external feed) compiles, fires, and
    # produces a full per-cell record (bars_to_trough / mae in the trades frame, the overlap-aware
    # reads in the cell) — no different from any other threshold-on-a-Series entry.
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    close = 100 + np.cumsum(np.random.default_rng(3).normal(0, 1, len(idx)))
    pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=idx
    ).to_csv(tmp_path / "px.csv")
    feed = pd.Series(np.random.default_rng(4).normal(0, 1, len(idx)).cumsum(), index=idx, name="f")
    feed.to_frame().to_csv(tmp_path / "f.csv")

    entry = {
        "type": "threshold",
        "left": {
            "type": "rolling_corr",
            "left": {"type": "change", "input": {"type": "field", "column": "close"}},
            "right": {"type": "external", "name": "f"},
            "window": 20,
        },
        "op": ">",
        "right": {"type": "constant", "value": 0.1},
    }
    thesis = Thesis.model_validate(
        {
            "name": "corr",
            "data": {"targets": ["target"], "external": {"f": {}}},
            "entry": entry,
            "params": {"horizon": 5},
        }
    )
    files = {"target": tmp_path / "px.csv", "f": tmp_path / "f.csv"}
    res = run_backtest(thesis, load(thesis, files))
    assert "bars_to_trough" in res.trades.columns and "mae" in res.trades.columns
    st = res.summary["cells"][0]["by_target"]["target"]
    assert st["n"] > 0 and 0.0 <= st["rot_p"] <= 1.0


def test_per_target_external_matches_shared_form(tmp_path):
    idx = pd.date_range("2020-01-01", periods=160, freq="D")
    for sym, seed in [("AAA", 1), ("BBB", 2)]:
        close = 100 + np.cumsum(np.random.default_rng(seed).normal(0, 1, len(idx)))
        pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0},
            index=idx,
        ).to_csv(tmp_path / f"{sym}.csv")
    rate = pd.Series(np.sin(np.arange(len(idx)) / 3.0), index=idx, name="rate")
    rate.to_frame().to_csv(tmp_path / "rate.csv")

    files = {"AAA": str(tmp_path / "AAA.csv"), "BBB": str(tmp_path / "BBB.csv")}
    zscore = {"type": "zscore", "window": 20, "input": {"type": "external", "name": "rate"}}
    rate_path = str(tmp_path / "rate.csv")

    def run(external, feeds):
        thesis = Thesis.model_validate(
            {
                "name": "ext",
                "data": {"targets": list(files), "external": external},
                "entry": {
                    "type": "threshold",
                    "left": zscore,
                    "op": "<",
                    "right": {"type": "constant", "value": -0.5},
                },
                "params": {"horizon": 10},
            }
        )
        return run_backtest(thesis, load(thesis, {**files, **feeds})).trades

    shared = run({"rate": {}}, {"rate": rate_path})
    per_target = run({"rate": {"per_target": True}}, {"rate@AAA": rate_path, "rate@BBB": rate_path})
    assert len(shared) > 0
    pd.testing.assert_frame_equal(shared, per_target)


def test_lagged_external_shifts_entries(tmp_path):
    # Flat prices, a step feed crossing the entry threshold at day 10: a 2-day lag delays the first
    # firing from the step date to step date + 2 days (next-open → entry_time == firing bar + 1).
    idx = pd.date_range("2021-01-01", periods=30, freq="1D")
    close = pd.Series(100.0, index=idx)
    pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=idx
    ).to_csv(tmp_path / "px.csv")
    step = pd.Series(np.where(np.arange(30) >= 10, 1.0, 0.0), index=idx, name="flag")
    step.to_frame().to_csv(tmp_path / "flag.csv")

    def first_entry(external):
        thesis = Thesis.model_validate(
            {
                "name": "lag",
                "data": {"targets": ["target"], "external": external},
                "entry": {
                    "type": "threshold",
                    "left": {"type": "external", "name": "flag"},
                    "op": ">",
                    "right": {"type": "constant", "value": 0.5},
                },
                "params": {"horizon": 5},
            }
        )
        files = {"target": tmp_path / "px.csv", "flag": flag_path}
        trades = run_backtest(thesis, load(thesis, files)).trades
        return trades["entry_time"].min()

    flag_path = str(tmp_path / "flag.csv")
    assert first_entry({"flag": {}}) == idx[11]
    assert first_entry({"flag": {"lag": 2}}) == idx[13]


def test_external_feed_bound_to_a_column_matches_that_series_alone_in_its_own_file(tmp_path):
    # The whole claim of the column binding: WHICH column a feed reads is a fact about the file,
    # never about the thesis — so ONE thesis, run against a two-column CSV with `rate` bound and
    # against a one-column CSV holding the same numbers, must produce the same trades down to the
    # byte. If the binding ever read the wrong column (or the position instead of the name) the
    # frames diverge, and a vendor re-shaping its CSV would silently become a different exam.
    idx = pd.date_range("2020-01-01", periods=160, freq="D")
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, len(idx)))
    pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=idx
    ).to_csv(tmp_path / "px.csv")
    rate = np.sin(np.arange(len(idx)) / 3.0)
    multi = pd.DataFrame({"noise": np.arange(len(idx), dtype=float), "rate": rate}, index=idx)
    multi.index.name = "datetime"
    multi.to_csv(tmp_path / "multi.csv")
    single = pd.DataFrame({"rate": rate}, index=idx)
    single.index.name = "datetime"
    single.to_csv(tmp_path / "single.csv")
    zscore = {"type": "zscore", "window": 20, "input": {"type": "external", "name": "rate"}}
    thesis = Thesis.model_validate(
        {
            "name": "ext",
            "data": {"targets": ["target"], "external": {"rate": {}}},
            "entry": {
                "type": "threshold",
                "left": zscore,
                "op": "<",
                "right": {"type": "constant", "value": -0.5},
            },
            "params": {"horizon": 10},
        }
    )

    def run(feed_path, columns=None):
        files = {"target": tmp_path / "px.csv", "rate": feed_path}
        return run_backtest(thesis, load(thesis, files, columns)).trades

    named = run(tmp_path / "multi.csv", {"rate": "rate"})
    single_col = run(tmp_path / "single.csv")  # a file with one value column names itself
    assert len(named) > 0
    pd.testing.assert_frame_equal(named, single_col)


def test_external_feed_multi_column_without_a_binding_refuses(tmp_path):
    # A file holding several numeric columns says nothing about which of them the feed IS, and
    # picking one (the first, the alphabetical) would measure a series nobody asked for. The
    # refusal names the flag pair that answers it, since the caller's fix is an invocation edit
    # there being no DSL field to reach for.
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    close = pd.Series(100.0, index=idx)
    pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=idx
    ).to_csv(tmp_path / "px.csv")
    multi = pd.DataFrame(
        {"a": np.arange(40, dtype=float), "b": np.arange(40, dtype=float)}, index=idx
    )
    multi.index.name = "datetime"
    multi.to_csv(tmp_path / "multi.csv")
    thesis = Thesis.model_validate(
        {
            "name": "amb",
            "data": {"targets": ["target"], "external": {"f": {}}},
            "entry": {
                "type": "threshold",
                "left": {"type": "external", "name": "f"},
                "op": ">",
                "right": {"type": "constant", "value": 0.5},
            },
            "params": {"horizon": 5},
        }
    )
    files = {"target": tmp_path / "px.csv", "f": tmp_path / "multi.csv"}
    with pytest.raises(ValueError, match=r"bind --column f=COL"):
        load(thesis, files)
    # …and the binding the message asks for is exactly what makes the same files load.
    md = load(thesis, files, {"f": "b"})
    np.testing.assert_allclose(md.externals["f"].to_numpy(), np.arange(40, dtype=float))


def test_shortonly_flips_the_sign(tmp_path):
    # Same firing, same fills: shortonly return is the negative of the longonly forward return.
    closes = (100 + np.random.RandomState(1).randn(160).cumsum()).tolist()
    ema = {"type": "ema", "window": 10, "input": {"type": "field", "column": "close"}}
    entry = {
        "type": "threshold",
        "left": {"type": "field", "column": "close"},
        "op": ">",
        "right": ema,
    }
    long_tr = _obs(tmp_path / "lo.csv", entry, closes, horizon=15, direction="longonly")
    short_tr = _obs(tmp_path / "sh.csv", entry, closes, horizon=15, direction="shortonly")
    assert len(long_tr) > 0 and (long_tr["horizon"] == 15).all()
    lc = long_tr[~long_tr["is_open"]].reset_index(drop=True)
    sc = short_tr[~short_tr["is_open"]].reset_index(drop=True)
    assert (lc["ret"] + sc["ret"]).abs().max() < 1e-9


def test_multi_target_multi_param(tmp_path):
    files = {}
    for sym, seed in [("AAA", 1), ("BBB", 2)]:
        rng = np.random.RandomState(seed)
        idx = pd.date_range("2020-01-01", periods=400, freq="1D")
        close = 100 + rng.randn(400).cumsum()
        pd.DataFrame(
            {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0},
            index=idx,
        ).to_csv(tmp_path / f"{sym}.csv")
        files[sym] = str(tmp_path / f"{sym}.csv")

    def ema(w):
        return {"type": "ema", "window": w, "input": {"type": "field", "column": "close"}}

    thesis = Thesis.model_validate(
        {
            "name": "ema_cross",
            "data": {"targets": list(files)},
            "entry": {
                "type": "first_true",
                "condition": {
                    "type": "threshold",
                    "op": ">",
                    "left": ema([10, 20]),
                    "right": ema(50),
                },
            },
            "params": {"horizon": 10},
        }
    )
    res = run_backtest(thesis, load(thesis, files))
    s = res.summary
    # one swept fast window (the entry side is the only side there is), over two targets
    assert s["params"] == ["ema_window"]
    assert set(s["targets"]) == {"AAA", "BBB"}
    assert {c["params"]["ema_window"] for c in s["cells"]} <= {10, 20}
    cell, st = s["cells"][0], s["cells"][0]["by_target"]["AAA"]
    for k in ("n", "mean_ret", "std_ret", "win_loss_ratio", "skewness", "rot_p", "t_hac"):
        assert k in st
    # a per-observation Sharpe is DERIVABLE (mean_ret / std_ret, un-annualized) from the panel's
    # own reads over that cell's closed returns — which is why none is emitted
    cell_rets = res.trades[
        (res.trades["ema_window"] == cell["params"]["ema_window"])
        & (res.trades["target"] == "AAA")
        & (~res.trades["is_open"])
    ]["ret"]
    if len(cell_rets) > 1 and cell_rets.std(ddof=1) > 0:
        assert st["mean_ret"] / st["std_ret"] == pytest.approx(
            cell_rets.mean() / cell_rets.std(ddof=1)
        )

    # Targets are the REGIME, not a search axis: every declared cell reports EVERY target side by
    # side (a target with no closed rows still gets an explicit entry), so the weakest one speaks
    # for itself instead of being folded away.
    assert s["n_hypotheses_attempted"] == 2 == len(s["cells"])
    for cell in s["cells"]:
        assert set(cell["by_target"]) == {"AAA", "BBB"}
        assert set(cell["outcome_coverage"]) == {"AAA", "BBB"}
        assert set(cell["signal_coverage"]) == {"AAA", "BBB"}
    assert set(s["sources"]) == {"AAA", "BBB"}


def test_window_and_horizon_sweep_combined(tmp_path):
    closes = (100 + np.random.RandomState(3).randn(200).cumsum()).tolist()
    _bars(tmp_path / "combo.csv", closes)
    ema_list = {"type": "ema", "window": [10, 20], "input": {"type": "field", "column": "close"}}
    thesis = Thesis.model_validate(
        {
            "name": "combo",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "threshold",
                "left": {"type": "field", "column": "close"},
                "op": "<",
                "right": ema_list,
            },
            "params": {"horizon": [3, 6]},
        }
    )
    res = run_backtest(thesis, load(thesis, {"target": tmp_path / "combo.csv"}))
    s = res.summary
    # the transform axis is discovered first, horizon is appended last
    assert s["params"] == ["ema_window", "horizon"]
    assert {c["params"]["horizon"] for c in s["cells"]} == {3, 6}
    assert {c["params"]["ema_window"] for c in s["cells"]} <= {10, 20}
    assert [c["cell_id"] for c in s["cells"]] == [
        "ema_window=10,horizon=3",
        "ema_window=10,horizon=6",
        "ema_window=20,horizon=3",
        "ema_window=20,horizon=6",
    ]


# ---- DSL-specified features + per-cell conditional buckets ------------------
# There is no RUN-LEVEL pooled `conditional_buckets`/`bucket_monotonicity` under any spelling — a
# pooled qcut re-enters the same bar once per combo × horizon, so its "conditioning" numbers would
# move whenever the grid composition did. The panels live PER CELL over the cell's own closed
# rows, which nothing outside the cell can move. Refusals are explicit
# (`no_closed_observations` / `insufficient_observations` / `insufficient_distinct_values`,
# BUCKET_MIN_N=20), never absences.

_BUCKET_REASONS = (
    None,
    "no_closed_observations",
    "insufficient_observations",
    "insufficient_distinct_values",
)


def test_dsl_feature_conditional_buckets(tmp_path):
    closes = (100 + np.random.RandomState(7).randn(300).cumsum()).tolist()
    _bars(tmp_path / "feat.csv", closes)
    ema = {"type": "ema", "window": 20, "input": {"type": "field", "column": "close"}}
    momentum = {
        "type": "change",
        "periods": 10,
        "kind": "pct",
        "input": {"type": "field", "column": "close"},
    }
    thesis = Thesis.model_validate(
        {
            "name": "feat",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "threshold",
                "left": {"type": "field", "column": "close"},
                "op": "<",
                "right": ema,
            },
            "params": {"horizon": 10, "features": {"momentum": momentum}},
        }
    )
    res = run_backtest(thesis, load(thesis, {"target": tmp_path / "feat.csv"}))
    assert "momentum" in res.trades.columns
    assert "conditional_buckets" not in res.summary  # never at the run level, only per cell
    for cell in res.summary["cells"]:
        panel = cell["conditional_buckets"]
        # params.features REPLACES the defaults, so the declared feature is the whole panel
        assert set(panel) == {"momentum"}
        blk = panel["momentum"]
        assert blk["reason"] is None and len(blk["buckets"]) >= 2
        for b in blk["buckets"]:
            assert {"bucket", "n", "mean_ret", "hit_rate"} <= b.keys()
        # the bucketed rows are the cell's own closed rows (a warmup-NaN snapshot drops out),
        # never a pool any other cell contributes to
        assert sum(b["n"] for b in blk["buckets"]) <= cell["by_target"]["target"]["n"]


def test_default_features_still_bucketed(thesis):
    res = run_backtest(thesis, load(thesis, {"target": _SAMPLE}))
    for col in ("ret_5", "ret_20", "vol_14"):
        assert col in res.trades.columns
    assert "conditional_buckets" not in res.summary
    cell = res.summary["cells"][0]
    assert set(cell["conditional_buckets"]) == {"ret_5", "ret_20", "vol_14"}
    # this fixture's pool is deep (n >> BUCKET_MIN_N), so every default feature actually buckets
    for blk in cell["conditional_buckets"].values():
        assert blk["reason"] is None
        assert len(blk["buckets"]) >= 2


def test_bucket_monotonicity_rides_each_cell(tmp_path):
    # The alphalens-style monotonicity read (evidence-only) rides the cell whose buckets it
    # describes — and only for features that actually bucketed with a rank signal to report.
    closes = (100 + np.random.RandomState(21).randn(400).cumsum()).tolist()
    _bars(tmp_path / "mono.csv", closes)
    below_ema = {
        "type": "threshold",
        "left": {"type": "field", "column": "close"},
        "op": "<",
        "right": {"type": "ema", "window": 10, "input": {"type": "field", "column": "close"}},
    }
    thesis = Thesis.model_validate(
        {"name": "m", "data": {"targets": ["target"]}, "entry": below_ema, "params": {"horizon": 5}}
    )
    s = run_backtest(thesis, load(thesis, {"target": tmp_path / "mono.csv"})).summary
    assert "bucket_monotonicity" in s["cells"][0] and "bucket_monotonicity" not in s
    graded = 0
    for cell in s["cells"]:
        for blk in cell["conditional_buckets"].values():
            assert blk["reason"] in _BUCKET_REASONS
        for name, mono in cell["bucket_monotonicity"].items():
            graded += 1
            assert name in cell["conditional_buckets"]
            assert cell["conditional_buckets"][name]["reason"] is None  # only bucketed features
            assert -1.0 <= mono["rho"] <= 1.0 and mono["sign"] in (-1, 0, 1)
    assert graded > 0  # the dense fixture does produce a rank read — nothing here is vacuous


def test_multi_target_bucket_edges_aggregate_by_ordinal(tmp_path):
    # The runner's rows carry `target`, so a two-target cell's bucket panel takes the
    # per-target-edges arm: aggregated records are labelled by ORDINAL (q1..qk), never by an
    # interval string, which per-target edges make target-relative. (The stats-level tests pin
    # the aggregation arithmetic; this pins the runner wiring.)
    n = 80
    idx = pd.date_range("2021-01-01", periods=n, freq="1D")
    rng = np.random.RandomState(21)
    a = 100 + rng.randn(n).cumsum()
    b = 5000 + 50 * rng.randn(n).cumsum()
    files = {
        "A": _write_bars(tmp_path / "A.csv", a, idx),
        "B": _write_bars(tmp_path / "B.csv", b, idx),
    }
    thesis = Thesis.model_validate(
        {"name": "mt", "data": {"targets": ["A", "B"]}, "entry": _lt(1e9), "params": {"horizon": 1}}
    )
    res = run_backtest(thesis, load(thesis, files))
    cell = res.summary["cells"][0]
    bucketed = [blk for blk in cell["conditional_buckets"].values() if blk["reason"] is None]
    assert bucketed  # the default features bucket on an 80-bar two-member pool
    for blk in bucketed:
        assert [r["bucket"] for r in blk["buckets"]] == [
            f"q{i + 1}" for i in range(len(blk["buckets"]))
        ]


# ---- cross_breadth: the effective-universe ledger -------------------------------------------
# The cross kernels reduce over the per-bar finite-member count k and discard it; member warmup
# legally thins the cross-section (a late start is warmup, not a hole), so without this panel
# the effective universe could drift silently through time. Evidence-only; read by no check.


def _breadth_basket(tmp_path, entry, n=60, late_from=0, horizon=2):
    idx = pd.date_range("2021-01-01", periods=n, freq="1D")
    rng = np.random.RandomState(3)
    series = {
        "A": 100 + rng.randn(n).cumsum(),
        "B": 90 + rng.randn(n).cumsum(),
        "C": 80 + rng.randn(n).cumsum(),
    }
    if late_from:
        series["C"][:late_from] = np.nan  # a late-starting member: warmup, legal
    files = {
        name: _write_bars(tmp_path / f"{name}.csv", vals, idx) for name, vals in series.items()
    }
    thesis = Thesis.model_validate(
        {
            "name": "breadth",
            "target_mode": "basket",
            "data": {"targets": ["A", "B", "C"]},
            "entry": entry,
            "params": {"horizon": horizon},
        }
    )
    return run_backtest(thesis, load(thesis, files)).summary


_RANK_ENTRY = {
    "type": "threshold",
    "left": {"type": "cross_rank", "input": {"type": "field", "column": "close"}},
    "op": ">=",
    "right": {"type": "constant", "value": 0.5},
}


def test_cross_breadth_reports_a_late_starting_member(tmp_path):
    # One member NaN for its first W bars: the cross-section runs at k=2 there and k=3 after —
    # the panel says so, bar for bar, where the kernels used to discard it.
    W = 15
    s = _breadth_basket(tmp_path, _RANK_ENTRY, n=60, late_from=W)
    panel = s["cross_breadth"]
    assert len(panel) == 1
    entry = panel[0]
    assert entry["node"] == "cross_rank(close)"
    assert entry["params"] == {}  # nothing swept
    assert entry["min_valid"] == 2 and entry["n_bars"] == 60
    assert entry["n_bars_evaluated"] == 60  # k >= 2 everywhere: the node always evaluated
    assert entry["n_bars_below_full"] == W
    assert entry["k_min"] == 2 and entry["k_max"] == 3
    assert entry["first_full_bar"].startswith("2021-01-16")  # bar W, the late member's start


def test_cross_breadth_full_basket_is_flat(tmp_path):
    # No warmup anywhere: every bar carries the whole declared basket and the ledger says so.
    s = _breadth_basket(tmp_path, _RANK_ENTRY, n=40)
    entry = s["cross_breadth"][0]
    assert entry["n_bars_evaluated"] == 40 and entry["n_bars_below_full"] == 0
    assert entry["k_min"] == entry["k_max"] == 3 and entry["k_median"] == 3.0
    assert entry["first_full_bar"] == s["index_start"]


def test_cross_breadth_is_keyed_per_combo_when_the_input_sweeps(tmp_path):
    # A swept input moves the node's warmup, so breadth is a per-combo fact: two entries with
    # distinct params, the longer window thinning more bars.
    entry = {
        "type": "threshold",
        "left": {
            "type": "cross_rank",
            "input": {
                "type": "change",
                "periods": [5, 21],
                "input": {"type": "field", "column": "close"},
            },
        },
        "op": ">=",
        "right": {"type": "constant", "value": 0.5},
    }
    s = _breadth_basket(tmp_path, entry, n=60)
    panel = s["cross_breadth"]
    assert len(panel) == 2
    by_params = {tuple(e["params"].items()): e for e in panel}
    assert len(by_params) == 2  # distinct combos, distinct entries
    short, long_ = sorted(panel, key=lambda e: list(e["params"].values()))
    # change(close, p) is NaN for the first p bars of every member, so the node evaluates on
    # fewer bars under the longer window — the sweep moved the SAMPLE the cross read stood on
    assert short["n_bars_evaluated"] == 60 - 5
    assert long_["n_bars_evaluated"] == 60 - 21


def test_conjunction_run_emits_an_empty_cross_breadth_panel(tmp_path):
    # Cross nodes are basket-gated by validation, so a conjunction run's panel is [] by
    # construction — always present, never absent (absence would carry no signal).
    _noise_px(tmp_path / "px.csv")
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": _lt(1e9),
            "params": {"horizon": 5},
        }
    )
    s = run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"})).summary
    assert s["cross_breadth"] == []


def test_cross_breadth_deduplicates_identical_nodes(tmp_path):
    # The same cross node twice in one entry tree shares one memoized frame and one breadth
    # profile — one entry, not two.
    entry = {"type": "and", "conditions": [_RANK_ENTRY, _RANK_ENTRY]}
    s = _breadth_basket(tmp_path, entry, n=40)
    assert len(s["cross_breadth"]) == 1


# ---- external feed as-of anchoring (observer purity) ------------------------


def test_sub_bar_external_feed_never_leaks_into_its_own_bar(tmp_path):
    # Regression for a look-ahead: an intraday feed print from INSIDE day D must not be visible to
    # the day-D bar (stamped D 00:00) — a sub-bar resample into left-labeled bins would leak it.
    # A value becomes usable at the first bar stamped at-or-after its timestamp.
    from seikan.compiler.data import _anchor_feed, _prepare_feed
    from seikan.dataio import read_strict_csv

    bars = pd.date_range("2021-01-01", periods=10, freq="1D")
    stamps, vals = [], []
    for i, day in enumerate(bars):  # two prints per day: finer than the bars, so sub-bar
        stamps += [day + pd.Timedelta(hours=6), day + pd.Timedelta(hours=18)]
        vals += [float(i), float(i) + 0.5]
    feed = pd.DataFrame({"x": vals}, index=pd.DatetimeIndex(stamps, name="datetime"))
    feed.to_csv(tmp_path / "feed.csv")

    df, rep = read_strict_csv(str(tmp_path / "feed.csv"), role="external")
    assert rep.ok and df is not None
    anchored = _anchor_feed(_prepare_feed(df, str(tmp_path / "feed.csv"), "feed"), bars)
    assert np.isnan(anchored.iloc[0])  # nothing published at-or-before day 0's stamp
    for i in range(1, 10):  # bar D sees day D-1's last print — never day D's own prints
        assert anchored.iloc[i] == pytest.approx((i - 1) + 0.5)


# ---- swept constants -------------------------------------------------------


def test_constant_sweep_summary_rollups(tmp_path):
    closes = [110.0] + [95.0] * 20
    _bars(tmp_path / "cs.csv", closes)
    thesis = Thesis.model_validate(
        {
            "name": "thresh_sweep",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "threshold",
                "left": {"type": "field", "column": "close"},
                "op": "<",
                "right": {"type": "constant", "value": [100.0, 150.0, 200.0], "name": "buy_below"},
            },
            "params": {"horizon": 3},
        }
    )
    res = run_backtest(thesis, load(thesis, {"target": tmp_path / "cs.csv"}))
    s = res.summary
    assert s["params"] == ["buy_below"]  # the swept constant is the only axis (horizon is scalar)
    assert {c["params"]["buy_below"] for c in s["cells"]} == {100.0, 150.0, 200.0}
    assert len(s["cells"]) == 3 and "buy_below" in res.trades.columns
    # integral float levels render compactly in the cell label (100.0 → "100")
    assert [c["cell_id"] for c in s["cells"]] == [
        "buy_below=100,horizon=3",
        "buy_below=150,horizon=3",
        "buy_below=200,horizon=3",
    ]


def test_constant_and_window_sweep_combine_into_one_grid(tmp_path):
    closes = (100 + np.random.RandomState(5).randn(200).cumsum()).tolist()
    _bars(tmp_path / "cw.csv", closes)
    thesis = Thesis.model_validate(
        {
            "name": "combo_const",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "threshold",
                "left": {
                    "type": "percentile",
                    "window": [14, 21],
                    "input": {"type": "field", "column": "close"},
                },
                "op": "<",
                "right": {"type": "constant", "value": [0.40, 0.50], "name": "pctl_floor"},
            },
            "params": {"horizon": 4},
        }
    )
    s = run_backtest(thesis, load(thesis, {"target": tmp_path / "cw.csv"})).summary
    # transform axis first, constant labelled by name
    assert s["params"] == ["percentile_window", "pctl_floor"]
    assert {c["params"]["pctl_floor"] for c in s["cells"]} <= {0.40, 0.50}
    assert s["n_hypotheses_attempted"] == 4 == len(s["cells"])


def test_rolling_window_sweep_is_a_grid_axis(tmp_path):
    # A `rolling` condition's list `window` fans out as the auto-named `rolling_window` axis,
    # combining with an inner transform sweep into one Cartesian grid that flows through to the
    # summary. `any`
    # over a high percentile floor fires near every bar, so all four combos populate the table.
    closes = (100 + np.random.RandomState(6).randn(200).cumsum()).tolist()
    _bars(tmp_path / "rw.csv", closes)
    thesis = Thesis.model_validate(
        {
            "name": "roll_sweep",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "rolling",
                "window": [3, 5],
                "agg": "any",
                "condition": {
                    "type": "threshold",
                    "left": {
                        "type": "percentile",
                        "window": [14, 21],
                        "input": {"type": "field", "column": "close"},
                    },
                    "op": "<",
                    "right": {"type": "constant", "value": 0.9},
                },
            },
            "params": {"horizon": 4},
        }
    )
    res = run_backtest(thesis, load(thesis, {"target": tmp_path / "rw.csv"}))
    s = res.summary
    # inner transform axis first, rolling after
    assert s["params"] == ["percentile_window", "rolling_window"]
    assert {c["params"]["rolling_window"] for c in s["cells"]} == {3, 5}
    assert len(s["cells"]) == 4  # 2 percentile windows × 2 rolling windows
    assert "rolling_window" in res.trades.columns


# ---- rolling_agg + rolling count: pullback expressiveness -------------------


def _rolling_max(window):
    return {
        "type": "rolling_agg",
        "agg": "max",
        "window": window,
        "input": {"type": "field", "column": "close"},
    }


def _drawdown_from_high(frac, window=5):
    # close < frac × (rolling window high) — i.e. more than (1-frac) below the N-bar high.
    return {
        "type": "threshold",
        "op": "<",
        "left": {
            "type": "binary_op",
            "op": "/",
            "left": {"type": "field", "column": "close"},
            "right": _rolling_max(window),
        },
        "right": {"type": "constant", "value": frac},
    }


def test_rolling_agg_drawdown_from_high(tmp_path):
    # The canonical pullback measure `rank` can't state: how far below the trailing high. A 10% drop
    # below the 5-bar high fires only at the dip bar; a flat run never does.
    tr = _obs(
        tmp_path / "dd.csv",
        _drawdown_from_high(0.95, 5),
        [100, 100, 100, 100, 100, 90, 100, 100],
        horizon=1,
    )
    closed = tr[~tr["is_open"]]
    assert len(closed) == 1  # only bar 5 (close 90 vs 5-bar high 100 = 0.90 < 0.95)
    idx = pd.date_range("2021-01-01", periods=8, freq="1D")
    assert closed.iloc[0]["entry_time"] == idx[6]  # fires bar 5 → fills next open (bar 6)


_DOWN = {
    "type": "threshold",
    "op": "<",
    "right": {"type": "constant", "value": 0.0},
    "left": {
        "type": "change",
        "periods": 1,
        "kind": "pct",
        "input": {"type": "field", "column": "close"},
    },
}


def _count_down(k, window=5):
    return {"type": "rolling", "window": window, "agg": "count", "min_count": k, "condition": _DOWN}


def test_rolling_count_at_least_k_of_n(tmp_path):
    # "at least K of the last N bars closed down" — an OR-of-thresholds `all`/`any` can't state.
    # down flags (change(kind=pct)<0): [_, T, T, F, T, T, F]; the rolling only initializes once the
    # inner (change) is warm across the whole window (bar >= 5).
    closes = [100, 99, 98, 101, 97, 96, 105, 106, 107]
    k3 = _obs(tmp_path / "k3.csv", _count_down(3, 5), closes, horizon=1)
    k4 = _obs(tmp_path / "k4.csv", _count_down(4, 5), closes, horizon=1)
    idx = pd.date_range("2021-01-01", periods=len(closes), freq="1D")
    # bar 5's window[1..5]=4 down (fires K=3 and K=4); bar 6's window[2..6]=3 down (fires K=3 only).
    # Each firing bar fills at the NEXT open: bar 5 → idx[6], bar 6 → idx[7].
    assert sorted(k3[~k3["is_open"]]["entry_time"]) == [idx[6], idx[7]]
    assert sorted(k4[~k4["is_open"]]["entry_time"]) == [idx[6]]  # only the 4-down window (bar 5)


def test_rolling_count_requires_min_count(tmp_path):
    # agg='count' without min_count, and min_count on a non-count agg, both fail model validation.
    with pytest.raises(ValueError, match="min_count"):
        Thesis.model_validate(
            {
                "name": "t",
                "data": {"targets": ["target"]},
                "entry": {"type": "rolling", "window": 5, "agg": "count", "condition": _DOWN},
                "params": {"horizon": 1},
            }
        )
    with pytest.raises(ValueError, match="only valid with agg='count'"):
        Thesis.model_validate(
            {
                "name": "t",
                "data": {"targets": ["target"]},
                "entry": {
                    "type": "rolling",
                    "window": 5,
                    "agg": "any",
                    "min_count": 2,
                    "condition": _DOWN,
                },
                "params": {"horizon": 1},
            }
        )


def test_rolling_agg_window_sweep_is_a_grid_axis(tmp_path):
    # A swept rolling_agg window fans out as the auto-named `rolling_agg_window` axis (the SMA-cross
    # response curve): close below its 5- vs 10-bar mean.
    closes = (100 + np.random.RandomState(9).randn(200).cumsum()).tolist()
    _bars(tmp_path / "ra.csv", closes)
    entry = {
        "type": "threshold",
        "op": "<",
        "left": {"type": "field", "column": "close"},
        "right": {
            "type": "rolling_agg",
            "agg": "mean",
            "window": [5, 10],
            "input": {"type": "field", "column": "close"},
        },
    }
    thesis = Thesis.model_validate(
        {"name": "sma", "data": {"targets": ["target"]}, "entry": entry, "params": {"horizon": 4}}
    )
    s = run_backtest(thesis, load(thesis, {"target": tmp_path / "ra.csv"})).summary
    assert s["params"] == ["rolling_agg_window"]
    assert [c["params"]["rolling_agg_window"] for c in s["cells"]] == [5, 10]


# ---- pre-entry drift (the trades frame's backward half) ---------------------


def test_pre_ret_negative_on_a_decline(tmp_path):
    # A signal firing throughout a strict decline enters after a fall → the sign-aligned pre-entry
    # drift is negative (the pullback/rebound setup, not momentum continuation). It rides the
    # trades frame as evidence for the caller; no summary panel describes a selected cell.
    closes = list(np.linspace(120.0, 81.0, 40))
    tr_path = tmp_path / "dec.csv"
    _bars(tr_path, closes)
    thesis = Thesis.model_validate(
        {
            "name": "dec",
            "data": {"targets": ["target"]},
            "entry": _lt(100.0),
            "params": {"horizon": 3},
        }
    )
    trades = run_backtest(thesis, load(thesis, {"target": tr_path})).trades
    pre = trades["pre_ret"].dropna()
    assert len(pre) > 0 and pre.mean() < 0


def test_pre_ret_excludes_pre_window_before_data_start(tmp_path):
    # Firing bars whose pre-window (h bars before the fill) precedes the data start carry NaN
    # pre_ret — even though they are perfectly valid FORWARD observations.
    tr_path = tmp_path / "early.csv"
    _bars(
        tr_path, [95, 96, 130, 130, 130, 130]
    )  # fires bars 0 & 1; both fill within h=3 of the start
    thesis = Thesis.model_validate(
        {
            "name": "e",
            "data": {"targets": ["target"]},
            "entry": _lt(100.0),
            "params": {"horizon": 3},
        }
    )
    res = run_backtest(thesis, load(thesis, {"target": tr_path}))
    closed = res.trades[~res.trades["is_open"]]
    assert len(closed) == 2 and closed["ret"].notna().all()  # two valid closed forward observations
    assert closed["pre_ret"].isna().all()  # but neither has a full pre-window → NaN


def test_pullback_thesis_end_to_end(tmp_path):
    # The composed pullback thesis: a drawdown-from-high AND a K-of-N down-day count, swept over a
    # horizon response curve — it compiles, fires, and reports every declared cell.
    cycle = [100.0, 98.0, 96.0, 94.0, 92.0, 110.0]
    closes = cycle * 12  # repeated deep-V pullbacks
    _bars(tmp_path / "pb.csv", closes)
    entry = {"type": "and", "conditions": [_drawdown_from_high(0.95, 5), _count_down(3, 5)]}
    thesis = Thesis.model_validate(
        {
            "name": "pullback",
            "data": {"targets": ["target"]},
            "entry": entry,
            "params": {"horizon": [1, 5, 10]},
        }
    )
    res = run_backtest(thesis, load(thesis, {"target": tmp_path / "pb.csv"}))
    s = res.summary
    assert s["params"] == ["horizon"]
    assert s["n_hypotheses_attempted"] == 3 == len(s["cells"])
    assert _closed_n(s) > 0
    assert res.trades["pre_ret"].notna().any()  # the backward evidence column is populated
    # the conditional panels ride each cell — never the run level
    assert "bucket_monotonicity" not in s and "conditional_buckets" not in s
    for cell in s["cells"]:
        assert set(cell["conditional_buckets"]) == {"ret_5", "ret_20", "vol_14"}


# ---- drawdown / first_true rebound primitives --------------------------------


def test_drawdown_node_trailing_fires_like_composed_ratio(tmp_path):
    # Native ``drawdown`` with window=5 at depth < -0.05 ≡ close/rolling_max < 0.95.
    tr = _obs(
        tmp_path / "ddn.csv",
        {
            "type": "threshold",
            "op": "<",
            "left": {"type": "drawdown", "window": 5},
            "right": {"type": "constant", "value": -0.05},
        },
        [100, 100, 100, 100, 100, 90, 100, 100],
        horizon=1,
    )
    closed = tr[~tr["is_open"]]
    assert len(closed) == 1
    idx = pd.date_range("2021-01-01", periods=8, freq="1D")
    assert closed.iloc[0]["entry_time"] == idx[6]


def test_drawdown_expanding_all_time_peak(tmp_path):
    # Expanding drawdown from the all-time high (no window): after 100→70 the depth is -0.30.
    closes = [100, 110, 105, 70, 80, 90, 100]
    tr = _obs(
        tmp_path / "ath.csv",
        {
            "type": "threshold",
            "op": "<=",
            "left": {"type": "drawdown"},
            "right": {"type": "constant", "value": -0.30},
        },
        closes,
        horizon=1,
    )
    closed = tr[~tr["is_open"]]
    idx = pd.date_range("2021-01-01", periods=len(closes), freq="1D")
    # Only bar 3 (70 vs ATH 110 ≈ -0.364) qualifies; fills next open.
    assert list(closed["entry_time"]) == [idx[4]]


def test_expanding_rolling_agg_max_and_rejects_mean(tmp_path):
    _bars(tmp_path / "ex.csv", [10, 20, 15, 25])
    ok = Thesis.model_validate(
        {
            "name": "ath",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "threshold",
                "op": ">=",
                "left": {"type": "field", "column": "close"},
                "right": {
                    "type": "rolling_agg",
                    "agg": "max",
                    "input": {"type": "field", "column": "close"},
                },
            },
            "params": {"horizon": 1},
        }
    )
    res = run_backtest(ok, load(ok, {"target": tmp_path / "ex.csv"}))
    # close == expanding max on new highs (bars 0,1,3) and never on bar 2 (15 < 20)
    closed = res.trades[~res.trades["is_open"]]
    idx = pd.date_range("2021-01-01", periods=4, freq="1D")
    # fires 0,1,3 → fills at 1,2, (3's fill is open/censored at end for h=1? bar3+1 out of range)
    assert idx[1] in set(closed["entry_time"]) and idx[2] in set(closed["entry_time"])
    with pytest.raises(ValueError, match="expanding rolling_agg"):
        Thesis.model_validate(
            {
                "name": "bad",
                "data": {"targets": ["target"]},
                "entry": {
                    "type": "threshold",
                    "op": ">",
                    "left": {
                        "type": "rolling_agg",
                        "agg": "mean",
                        "input": {"type": "field", "column": "close"},
                    },
                    "right": {"type": "constant", "value": 0},
                },
                "params": {"horizon": 1},
            }
        )


def test_first_true_episode_entry_not_every_bar(tmp_path):
    # Regime close < 100 holds for many bars; first_true fires once per episode.
    closes = [110, 90, 90, 90, 110, 85, 85, 100]
    regime = _lt(100.0)
    every = _obs(tmp_path / "every.csv", regime, closes, horizon=1)
    once = _obs(
        tmp_path / "once.csv",
        {"type": "first_true", "condition": regime},
        closes,
        horizon=1,
    )
    assert len(every[~every["is_open"]]) > len(once[~once["is_open"]])
    idx = pd.date_range("2021-01-01", periods=len(closes), freq="1D")
    # Episodes start at bars 1 and 5 → fills at idx[2], idx[6]
    assert sorted(once[~once["is_open"]]["entry_time"]) == [idx[2], idx[6]]


def test_first_true_warmup_no_phantom_on_already_true_regime(tmp_path):
    # EMA(14) warms up while already below 100 — first_true must not fire at the latch bar.
    closes = [50.0] * 30 + [120.0] * 5 + [50.0] * 10
    entry = {
        "type": "first_true",
        "condition": {
            "type": "threshold",
            "left": {"type": "ema", "window": 14, "input": {"type": "field", "column": "close"}},
            "op": "<",
            "right": {"type": "constant", "value": 100},
        },
    }
    tr = _obs(tmp_path / "warm.csv", entry, closes, horizon=1)
    # Just assert it runs and never fires before EMA init (~bar 14)
    closed = tr[~tr["is_open"]]
    if len(closed):
        idx = pd.date_range("2021-01-01", periods=len(closes), freq="1D")
        assert closed["entry_time"].min() >= idx[14]


def test_drawdown_window_and_first_true_cooldown_sweep(tmp_path):
    # Sweep plumbing: drawdown.window and first_true.cooldown must each become a grid axis with
    # distinct per-combo results (a missed traversal would silently pin one axis).
    cycle = [100.0, 95.0, 90.0, 85.0, 120.0]
    closes = cycle * 20
    _bars(tmp_path / "sw.csv", closes)
    entry = {
        "type": "first_true",
        "cooldown": [0, 3],
        "condition": {
            "type": "threshold",
            "op": "<",
            "left": {"type": "drawdown", "window": [3, 5]},
            "right": {"type": "constant", "value": -0.08},
        },
    }
    thesis = Thesis.model_validate(
        {"name": "sw", "data": {"targets": ["target"]}, "entry": entry, "params": {"horizon": 2}}
    )
    s = run_backtest(thesis, load(thesis, {"target": tmp_path / "sw.csv"})).summary
    assert set(s["params"]) == {"drawdown_window", "first_true_cooldown"}
    assert {c["params"]["drawdown_window"] for c in s["cells"]} == {3, 5}
    assert {c["params"]["first_true_cooldown"] for c in s["cells"]} == {0, 3}
    assert len(s["cells"]) == 4


def test_deep_drawdown_rebound_thesis_end_to_end(tmp_path):
    # first_true(rolling-all of deep drawdown) + long horizon — full pipeline incl. entry listing.
    from seikan.api import list_entries

    # Repeated: grind down from a peak into a deep trough, then rebound.
    cycle = [100.0, 98.0, 95.0, 90.0, 80.0, 75.0, 70.0, 85.0, 95.0, 105.0]
    closes = cycle * 15
    _bars(tmp_path / "rebound.csv", closes)
    deep = {
        "type": "threshold",
        "op": "<",
        "left": {"type": "drawdown", "window": 8},
        "right": {"type": "constant", "value": -0.20},
    }
    entry = {
        "type": "first_true",
        "condition": {
            "type": "rolling",
            "window": 3,
            "agg": "all",
            "condition": deep,
        },
    }
    thesis = Thesis.model_validate(
        {
            "name": "deep-dd-rebound",
            "data": {"targets": ["target"]},
            "entry": entry,
            "params": {"horizon": [5, 10]},
        }
    )
    md = load(thesis, {"target": tmp_path / "rebound.csv"})
    res = run_backtest(thesis, md)
    s = res.summary
    assert s["params"] == ["horizon"]
    assert len(s["cells"]) == 2 and _closed_n(s) > 0
    # The entry listing uses the same DSL/compiler — and the same one materialized MarketData.
    report = list_entries(thesis, md)
    assert report.series_end is not None and isinstance(report.entries, list)


# ---- post-entry MAE + left-side duration/runup nodes ------------------------


def test_mae_long_dip_then_recover(tmp_path):
    # Hand-computed: fire when close < 100 at bar 1 (95). Fill at open[2]=100.
    # Horizon 3: path opens 100, 90, 80, 110 with lows matching opens (flat bars).
    # MAE = min(low[2..4] + the exit print open[5]) / 100 - 1. With distinct OHLC:
    #   bar:     0          1          2          3          4          5
    #   o/h/l/c: 110/110/110/110, 95/95/95/95, 100/100/100/100, 90/90/85/90, 80/80/80/80,
    #            110/110/110/110
    # Fill at bar 2 open=100; lows [100, 85, 80] + exit print 110 → min=80 → mae = 80/100-1
    # = -0.20
    rows = [
        [110, 110, 110, 110],
        [95, 95, 95, 95],
        [100, 100, 100, 100],
        [90, 90, 85, 90],
        [80, 80, 80, 80],
        [110, 110, 110, 110],
    ]
    tr = _ohlc_obs(tmp_path / "mae_long.csv", rows, _lt(100.0), horizon=3)
    closed = tr[~tr["is_open"]]
    assert len(closed) == 1
    assert closed.iloc[0]["mae"] == pytest.approx(-0.20)
    assert closed.iloc[0]["ret"] == pytest.approx(110.0 / 100.0 - 1.0)
    # Path back above entry at j=3 (open[5]=110)
    assert closed.iloc[0]["bars_to_positive"] == pytest.approx(3.0)
    # Trough of the adverse (low) path is at j=2 (low=80)
    assert closed.iloc[0]["bars_to_trough"] == pytest.approx(2.0)


def test_mae_shortonly_uses_high_frame(tmp_path):
    # Short: adverse = high. Fire close>100 at bar 1 (110). Fill open[2]=100.
    # h=2: highs over [2..3] = 100, 120 + the exit print open[4]=90 → max=120
    # → mae = -(120/100-1) = -0.20
    rows = [
        [90, 90, 90, 90],
        [110, 110, 110, 110],
        [100, 100, 100, 100],
        [95, 120, 95, 95],
        [90, 105, 90, 90],
    ]
    tr = _ohlc_obs(tmp_path / "mae_short.csv", rows, _gt(100.0), horizon=2, direction="shortonly")
    closed = tr[~tr["is_open"]]
    assert len(closed) == 1
    assert closed.iloc[0]["mae"] == pytest.approx(-0.20)


def test_mae_censored_open_is_nan(tmp_path):
    # Fire on last-but-one bar so fill exists but horizon runs past end → open, mae NaN.
    tr = _obs(tmp_path / "mae_open.csv", _lt(100.0), [110, 90, 100], horizon=2)
    assert len(tr) == 1 and bool(tr.iloc[0]["is_open"])
    assert math.isnan(tr.iloc[0]["mae"])


def test_mae_nan_hole_in_window_invalidates(tmp_path):
    # THE P2 regression: a single NaN low INSIDE the forward window must invalidate the MAE
    # (any-NaN → maximum filter over the indicator); a minimum filter would catch only all-NaN
    # windows and certify an incomplete adverse path as finite.
    rows = [
        [110, 111, 109, 110],
        [95, 96, 94, 95],  # fires (close < 100)
        [100, 101, 99, 100],  # fill bar
        [101, 102, float("nan"), 101],  # NaN low inside the MAE window
        [102, 103, 101, 102],
        [103, 104, 102, 103],
    ]
    tr = _ohlc_obs(tmp_path / "mae_hole.csv", rows, _lt(100.0), horizon=3)
    closed = tr[~tr["is_open"]]
    assert len(closed) == 1
    assert closed.iloc[0]["exit_reason"] == "horizon"  # the RETURN leg is intact…
    assert math.isnan(closed.iloc[0]["mae"])  # …but the adverse path is incomplete → NaN


def test_mae_nan_hole_short_side(tmp_path):
    rows = [
        [100, 101, 99, 100],
        [105, 106, 104, 105],  # fires (close > 100)
        [100, 101, 99, 100],  # fill bar
        [101, float("nan"), 100, 101],  # NaN high inside the short-side MAE window
        [102, 103, 101, 102],
    ]
    tr = _ohlc_obs(tmp_path / "mae_hole_s.csv", rows, _gt(100.0), horizon=2, direction="shortonly")
    closed = tr[~tr["is_open"]]
    assert len(closed) == 1
    assert math.isnan(closed.iloc[0]["mae"])


def test_mae_clean_window_stays_finite_next_to_a_hole(tmp_path):
    # The indicator filter is window-local: a firing whose window ends BEFORE the hole keeps a
    # finite MAE (the hole must not smear backwards beyond the windows it actually touches).
    rows = [
        [110, 111, 109, 110],
        [95, 96, 94, 95],  # fires; h=1 window = bars 2..3 (clean)
        [100, 101, 99, 100],
        [101, 102, 100, 101],
        [101, 102, float("nan"), 101],  # hole after the window
        [102, 103, 101, 102],
    ]
    tr = _ohlc_obs(tmp_path / "mae_clean.csv", rows, _lt(100.0), horizon=1)
    closed = tr[~tr["is_open"]]
    assert len(closed) == 1
    assert closed.iloc[0]["mae"] == pytest.approx(99.0 / 100.0 - 1.0)


# ---- post-entry MFE: the favorable mirror ------------------------------------
# An MFE is a MARK, never an attainable exit — this engine has no exit rule, so the interim peak
# is what the position was worth on paper, not what any strategy could have taken. Every test here
# reads it off the RAW path, which is why none of them runs under a benchmark.


def test_mfe_long_peak_then_fade(tmp_path):
    # Hand-computed mirror of test_mae_long_dip_then_recover. Fire when close < 100 at bar 1 (95).
    # Fill at open[2] = 100. Horizon 3 → full H/L over bars 2..4 plus ONLY the exit print
    # open[5] (the exit bar's own high/low never enter):
    #   bar:     0                1              2                    3
    #   o/h/l/c: 110/110/110/110, 95/95/95/95,   100/100/100/100,     110/120/110/110
    #   bar:     4                5
    #   o/h/l/c: 105/105/105/105, 90/90/90/90
    # highs [100, 120, 105] + exit print 90 → max 120 → mfe = 120/100 − 1 = +0.20
    # lows  [100, 110, 105] + exit print 90 → min  90 → mae =  90/100 − 1 = −0.10
    # The give-back read the pair exists for: a +20% paper mark closed at −10%, and no ratio
    # between them is a strategy — there was never an exit rule to take the peak with.
    rows = [
        [110, 110, 110, 110],
        [95, 95, 95, 95],
        [100, 100, 100, 100],
        [110, 120, 110, 110],
        [105, 105, 105, 105],
        [90, 90, 90, 90],
    ]
    tr = _ohlc_obs(tmp_path / "mfe_long.csv", rows, _lt(100.0), horizon=3)
    closed = tr[~tr["is_open"]]
    assert len(closed) == 1
    row = closed.iloc[0]
    assert row["mfe"] == pytest.approx(0.20)
    assert row["mae"] == pytest.approx(-0.10)
    assert row["ret"] == pytest.approx(90.0 / 100.0 - 1.0)
    assert row["mae"] <= row["ret"] <= row["mfe"]


def test_mfe_shortonly_uses_low_frame(tmp_path):
    # Short: FAVORABLE = low (the adverse frame's mirror — mae reads high here). Fire close > 100
    # at bar 1 (110), fill open[2] = 100.
    # lows  over [2..3]: 100, 80 + exit print open[4]=90 → min  80 → mfe = −(80/100 − 1) = +0.20
    # highs over [2..3]: 100, 105 + exit print 90       → max 105 → mae = −(105/100 − 1) = −0.05
    rows = [
        [90, 90, 90, 90],
        [110, 110, 110, 110],
        [100, 100, 100, 100],
        [95, 105, 80, 95],
        [90, 95, 90, 90],
    ]
    tr = _ohlc_obs(tmp_path / "mfe_short.csv", rows, _gt(100.0), horizon=2, direction="shortonly")
    closed = tr[~tr["is_open"]]
    assert len(closed) == 1
    row = closed.iloc[0]
    assert row["mfe"] == pytest.approx(0.20)
    assert row["mae"] == pytest.approx(-0.05)
    assert row["ret"] == pytest.approx(0.10)  # −(90/100 − 1)


def test_mfe_censored_open_is_nan(tmp_path):
    # A right-censored firing has no measured window at all, so BOTH excursion marks are NaN — the
    # path columns never describe an observation the return column refused to record.
    tr = _obs(tmp_path / "mfe_open.csv", _lt(100.0), [110, 90, 100], horizon=2)
    assert len(tr) == 1 and bool(tr.iloc[0]["is_open"])
    assert math.isnan(tr.iloc[0]["mfe"]) and math.isnan(tr.iloc[0]["mae"])


@pytest.mark.parametrize(
    "direction,censored,intact",
    [("longonly", "mfe", "mae"), ("shortonly", "mae", "mfe")],
)
def test_a_window_hole_censors_only_the_frame_it_sits_in(tmp_path, direction, censored, intact):
    # ONE hole in the `high` column, read from both sides: long reads high as its FAVORABLE frame
    # (mfe dies, mae lives), short reads it as its ADVERSE frame (mae dies, mfe lives). The
    # return leg is measured off `open` and closes cleanly either way — which is exactly why the
    # excursion blocks carry their own `n`: a censored path column subtracts from THEIR pool while
    # the cell's `n` is untouched, and equating the two counts would silently zero-fill a hole.
    # With h=1 the window's H/L part is the FILL bar alone — the exit bar contributes only its
    # open — so the hole censors exactly the window whose H/L part CONTAINS bar 5.
    rows = [
        [90, 91, 89, 90],
        [91, 92, 90, 91],
        [92, 93, 91, 92],
        [93, 94, 92, 93],
        [94, 95, 93, 94],
        [95, float("nan"), 94, 95],  # the hole
        [96, 97, 95, 96],
        [97, 98, 96, 97],
    ]
    res = _ohlc_run(tmp_path / "hole.csv", rows, _lt(100.0), horizon=1, direction=direction)
    tr = res.trades.sort_values("entry_bar")
    closed = tr[~tr["is_open"]]
    # fires on every bar (every close < 100); the last firing cannot fill, and the fill at bar 7
    # has no forward bar left → 6 closed rows, every one of them a genuine "horizon" exit
    assert len(closed) == 6
    assert set(closed["exit_reason"]) == {"horizon"}
    by_fill = {int(r["entry_bar"]) + 1: r for _, r in closed.iterrows()}
    # only the window whose H/L part contains bar 5 loses the holed frame's mark…
    assert math.isnan(by_fill[5][censored])
    # …its neighbours do not: fill 4's H/L part ends a bar EARLIER and its exit-bar read is the
    # (finite) open[5], never the holed high — the hole sits in a part of the exit bar the v3
    # window no longer contains — while fill 6 starts after the hole. Window-local, no smear.
    assert math.isfinite(by_fill[4][censored])
    assert math.isfinite(by_fill[3][censored]) and math.isfinite(by_fill[6][censored])
    # the other frame is whole, so its column survives on every closed row
    assert np.isfinite(closed[intact].to_numpy(dtype=float)).all()

    st = res.summary["cells"][0]["by_target"]["target"]
    assert st["n"] == 6
    assert st[f"{censored}_quantiles"]["n"] == 5 < st["n"]  # the pool IS a strict subset
    assert st[f"{intact}_quantiles"]["n"] == 6


def test_mfe_feed_outcome_uses_the_measured_series(tmp_path):
    # A feed outcome has no OHLC frames to mirror, so BOTH excursion marks read the measured series
    # itself: mfe = the window's max feed value, mae = its min, both against the anchor value.
    _bars(tmp_path / "px.csv", [110, 95, 102, 99, 100, 99, 110])
    idx = pd.date_range("2021-01-01", periods=7, freq="1D")
    pd.DataFrame({"iv": [20.0, 21.0, 22.0, 25.0, 24.0, 23.0, 26.0]}, index=idx).to_csv(
        tmp_path / "iv.csv"
    )
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"], "external": {"iv": {}}},
            "entry": _lt(100.0),  # fires at bars 1 (95), 3 (99), 5 (99)
            "params": {"horizon": 2, "outcome": {"series": "iv", "kind": "diff"}},
        }
    )
    files = {"target": tmp_path / "px.csv", "iv": tmp_path / "iv.csv"}
    res = run_backtest(thesis, load(thesis, files))
    closed = res.trades[~res.trades["is_open"]].sort_values("entry_time")
    # bar 1 → anchor iv[2]=22 over iv[2..4] = {22, 25, 24}: peak 25 → +3, trough 22 → 0, ret 24−22
    # bar 3 → anchor iv[4]=24 over iv[4..6] = {24, 23, 26}: peak 26 → +2, trough 23 → −1, ret +2
    assert list(closed["ret"]) == pytest.approx([2.0, 2.0])
    assert list(closed["mfe"]) == pytest.approx([3.0, 2.0])
    assert list(closed["mae"]) == pytest.approx([0.0, -1.0])
    # the second row's peak IS its exit mark — a feed outcome has no intra-bar H/L, so the exit
    # "print" is the measured series itself and the v3 split window is IDENTICAL to the old
    # inclusive one — while the first row's peak is an interim mark it gave back, in the units
    # of the FEED, never the target's
    assert res.summary["outcome"] == {"series": "iv", "kind": "diff", "units": "level_diff"}


@pytest.mark.parametrize("direction", ["longonly", "shortonly"])
def test_excursions_bracket_the_return_row_wise(tmp_path, direction):
    # The two structural sign/ordering facts, over a whole noise run rather than one hand fixture:
    # (1) mae <= 0 <= mfe wherever both are defined — the fill bar's own low/high sit on either
    #     side of the fill price, so neither mark can cross the entry;
    # (2) mae <= ret <= mfe, because the exit print open[fill+h] — the mark the return is
    #     measured at — is itself one of the marks the extremum ranges over (the H/L part stops
    #     a bar earlier; the exit bar's own high/low never enter).
    # (2) deliberately does NOT hold under a benchmark and is NOT asserted for one: `ret` becomes
    # an EXCESS return there while `mae`/`mfe` stay RAW, so the three stop being commensurable and
    # a bracket between them would be an arithmetic accident, not an invariant.
    _noise_px(tmp_path / "px.csv", n=400)
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": _gt(0.0),  # every bar fires: the property is about the path, not the signal
            "params": {"horizon": 5, "direction": direction},
        }
    )
    res = run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"}))
    closed = res.trades[~res.trades["is_open"]]
    mae = closed["mae"].to_numpy(dtype=float)
    mfe = closed["mfe"].to_numpy(dtype=float)
    ret = closed["ret"].to_numpy(dtype=float)
    defined = np.isfinite(mae) & np.isfinite(mfe)
    assert defined.sum() > 300 and defined.all()  # no holes in this fixture: nothing is vacuous
    assert (mae[defined] <= 0.0).all() and (mfe[defined] >= 0.0).all()
    assert (mae[defined] <= ret[defined]).all() and (ret[defined] <= mfe[defined]).all()
    assert (mfe[defined] > 0.0).any() and (mae[defined] < 0.0).any()


# ---- the v3 excursion window: post-exit prices never enter ----------------------------------
# The path a fill is marked over is the full H/L of [fill, fill+h−1] plus ONLY the exit print
# open[fill+h] — the exit bar's own high/low print AFTER the exit and belong to the next holding
# period. Pre-v3 the window was [fill, fill+h] inclusive, so an exit-bar spike could rewrite
# mae/mfe/bars_to_trough off prices the trade never held through.


def test_excursions_ignore_the_exit_bars_high_low(tmp_path):
    # THE AUDIT FIXTURE: a dead-flat path whose EXIT bar spikes to high=1000 / low=1 after
    # opening flat at 100. The trade never held through that spike — ret == 0 — and the marks
    # must say so: the pre-v3 inclusive window read mfe=+9.0 / mae=−0.99 off post-exit prices.
    rows = [
        [100, 100, 100, 100],
        [95, 95, 95, 95],  # fires (close < 100)
        [100, 100, 100, 100],  # fill bar
        [100, 100, 100, 100],
        [100, 1000, 1, 100],  # exit bar: open 100, then a spike the trade never held
        [100, 100, 100, 100],
    ]
    tr = _ohlc_obs(tmp_path / "exit_spike.csv", rows, _lt(100.0), horizon=2)
    row = tr[~tr["is_open"]].sort_values("entry_bar").iloc[0]  # fill 2, exit print open[4]=100
    assert row["ret"] == pytest.approx(0.0)
    assert row["mfe"] == pytest.approx(0.0)
    assert row["mae"] == pytest.approx(0.0)


def test_trough_attained_at_the_exit_print_reports_h(tmp_path):
    # The exit print is a real member of the path: when it IS the worst mark, mae reads off it
    # and bars_to_trough == h — "the trough was the exit itself".
    rows = [
        [100, 100, 100, 100],
        [95, 95, 95, 95],  # fires
        [100, 101, 99, 100],  # fill bar: interim low 99
        [98, 99, 97, 98],  # interim low 97
        [90, 95, 90, 92],  # exit bar: open 90, below every interim low
    ]
    tr = _ohlc_obs(tmp_path / "trough_exit.csv", rows, _lt(100.0), horizon=2)
    row = tr[~tr["is_open"]].sort_values("entry_bar").iloc[0]
    assert row["mae"] == pytest.approx(90.0 / 100.0 - 1.0)
    assert row["ret"] == pytest.approx(90.0 / 100.0 - 1.0)  # the trough IS the exit print
    assert row["bars_to_trough"] == pytest.approx(2.0)  # j == h


def test_trough_tie_with_the_exit_print_resolves_to_the_path_index(tmp_path):
    # First attainment: when the exit print only TIES the interim trough, the trough was reached
    # on the path first — the comparison is strict, so j stays the path index.
    rows = [
        [100, 100, 100, 100],
        [95, 95, 95, 95],  # fires
        [100, 101, 99, 100],  # fill bar
        [98, 99, 90, 98],  # interim low 90 — the first attainment
        [90, 95, 90, 92],  # exit print open 90 ties it
    ]
    tr = _ohlc_obs(tmp_path / "trough_tie.csv", rows, _lt(100.0), horizon=2)
    row = tr[~tr["is_open"]].sort_values("entry_bar").iloc[0]
    assert row["mae"] == pytest.approx(-0.10)
    assert row["bars_to_trough"] == pytest.approx(1.0)  # bar 3, never the exit


def test_exit_bar_high_low_hole_no_longer_censors_the_excursion(tmp_path):
    # A NaN in the exit bar's high/low sits OUTSIDE the v3 window — only its open is read — so
    # the excursion survives; the pre-v3 inclusive window died on it.
    rows = [
        [100, 100, 100, 100],
        [95, 95, 95, 95],  # fires
        [100, 101, 99, 100],  # fill bar
        [98, 99, 97, 98],
        [96, float("nan"), float("nan"), 96],  # exit bar: holed H/L, finite open
        [100, 100, 100, 100],
    ]
    tr = _ohlc_obs(tmp_path / "exit_hole.csv", rows, _lt(100.0), horizon=2)
    row = tr[~tr["is_open"]].sort_values("entry_bar").iloc[0]
    assert row["exit_reason"] == "horizon"
    assert row["mae"] == pytest.approx(96.0 / 100.0 - 1.0)  # the exit print IS the trough
    assert row["mfe"] == pytest.approx(101.0 / 100.0 - 1.0)  # interim high on the fill bar
    assert row["bars_to_trough"] == pytest.approx(2.0)


def test_h1_excursions_are_fill_bar_extremes_plus_exit_print(tmp_path):
    # h=1: the H/L part is the FILL bar alone; the next bar contributes only its open.
    rows = [
        [100, 100, 100, 100],
        [95, 95, 95, 95],  # fires
        [100, 104, 97, 100],  # fill bar: the whole interim path
        [102, 150, 50, 102],  # exit bar: open 102; its own H/L never enter
        [100, 100, 100, 100],
    ]
    tr = _ohlc_obs(tmp_path / "h1.csv", rows, _lt(100.0), horizon=1)
    row = tr[~tr["is_open"]].sort_values("entry_bar").iloc[0]
    assert row["mae"] == pytest.approx(97.0 / 100.0 - 1.0)
    assert row["mfe"] == pytest.approx(104.0 / 100.0 - 1.0)
    assert row["ret"] == pytest.approx(102.0 / 100.0 - 1.0)
    assert row["bars_to_trough"] == pytest.approx(0.0)  # the fill bar's own low


def test_runup_and_bars_since_extremum_nodes_fire(tmp_path):
    # Expanding runup >= 0.05 after a trough, AND bars_since min >= 2 (stabilization).
    # bars_since_min first clears 2 at bar 5 (90); fill next open needs a bar after that.
    closes = [100, 80, 80, 80, 88, 90, 95, 100]
    entry = {
        "type": "and",
        "conditions": [
            {
                "type": "threshold",
                "op": ">=",
                "left": {"type": "runup"},
                "right": {"type": "constant", "value": 0.05},
            },
            {
                "type": "threshold",
                "op": ">=",
                "left": {"type": "bars_since_extremum", "extremum": "min"},
                "right": {"type": "constant", "value": 2},
            },
        ],
    }
    tr = _obs(tmp_path / "ru.csv", entry, closes, horizon=1)
    closed = tr[~tr["is_open"]]
    idx = pd.date_range("2021-01-01", periods=len(closes), freq="1D")
    # bar 5 (90): runup=0.125, since-min=2 → fill at idx[6]
    assert idx[6] in set(closed["entry_time"])


def test_bars_since_extremum_window_sweep(tmp_path):
    _bars(tmp_path / "bse.csv", [10, 9, 8, 7, 8, 9, 10] * 5)
    thesis = Thesis.model_validate(
        {
            "name": "bse",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "threshold",
                "op": ">=",
                "left": {"type": "bars_since_extremum", "extremum": "max", "window": [3, 5]},
                "right": {"type": "constant", "value": 2},
            },
            "params": {"horizon": 1},
        }
    )
    s = run_backtest(thesis, load(thesis, {"target": tmp_path / "bse.csv"})).summary
    assert "bars_since_extremum_window" in s["params"]
    assert {c["params"]["bars_since_extremum_window"] for c in s["cells"]} == {3, 5}


def test_constant_sweep_name_colliding_with_trade_column_rejected_at_validation():
    # A swept constant whose name collides with a trades-frame column is refused at PARSE time
    # (the runner's reserved-column check is a library backstop for model_construct paths).
    with pytest.raises(ValueError, match="reserved trade/feature columns"):
        Thesis.model_validate(
            {
                "name": "bad_axis",
                "data": {"targets": ["target"]},
                "entry": {
                    "type": "threshold",
                    "left": {"type": "field", "column": "close"},
                    "op": "<",
                    "right": {"type": "constant", "value": [90.0, 100.0], "name": "ret"},
                },
                "params": {"horizon": 3},
            }
        )


def test_horizon_reserved_as_sweep_axis_name_rejected_at_validation():
    # A swept constant named "horizon" collides with the horizon axis → rejected at PARSE time
    # (collect_sweeps keeps the same refusal as a runtime backstop).
    with pytest.raises(ValueError, match="reserved"):
        Thesis.model_validate(
            {
                "name": "bad_horizon",
                "data": {"targets": ["target"]},
                "entry": {
                    "type": "threshold",
                    "left": {"type": "field", "column": "close"},
                    "op": "<",
                    "right": {"type": "constant", "value": [90.0, 100.0], "name": "horizon"},
                },
                "params": {"horizon": 3},
            }
        )


def test_runner_reserved_column_backstop_survives_a_forged_thesis(tmp_path):
    # The parse-time validator refuses a sweep axis colliding with a trade column, but the runner
    # keeps its OWN reserved-column check as a library backstop for a model_construct/forged thesis
    # that bypassed validation. Forge a validated swept constant's name to a trade column and
    # confirm run_backtest still refuses at the boundary.
    _bars(tmp_path / "col.csv", [110.0] + [95.0] * 10)
    valid = Thesis.model_validate(
        {
            "name": "forge",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "threshold",
                "left": {"type": "field", "column": "close"},
                "op": "<",
                "right": {"type": "constant", "value": [90.0, 100.0], "name": "cutoff"},
            },
            "params": {"horizon": 3},
        }
    )
    forged = valid.model_copy(deep=True)
    object.__setattr__(forged.entry.right, "name", "ret")  # collides with the trades 'ret' column
    with pytest.raises(ValueError, match="reserved trade/feature columns"):
        run_backtest(forged, load(forged, {"target": tmp_path / "col.csv"}))


def test_entry_bar_column_matches_firing_bars(tmp_path):
    # entry_bar is the FIRING bar index (fill = entry_bar + 1) — the basis the event-time HAC
    # and nonoverlap_count share, carried so every trades sub-frame keeps real bar distances.
    tr = _obs(tmp_path / "eb.csv", _lt(100.0), [110, 95, 102, 108, 100, 96, 103, 108], horizon=2)
    assert "entry_bar" in tr.columns
    idx = pd.date_range("2021-01-01", periods=8, freq="1D")
    for _, row in tr.iterrows():
        assert idx[int(row["entry_bar"]) + 1] == row["entry_time"]  # next-open fill anchor


# ---- benchmark-adjusted (excess) forward returns -----------------------------


def _write_bars(path: Path, opens: np.ndarray, idx: pd.DatetimeIndex) -> str:
    df = pd.DataFrame(
        {"open": opens, "high": opens, "low": opens, "close": opens, "volume": 1000.0}, index=idx
    )
    df.index.name = "datetime"
    df.to_csv(path)
    return str(path)


def _bench_run(tmp_path, tgt_opens, bench_opens=None, *, entry=None, horizon=5, **params):
    n = len(tgt_opens)
    idx = pd.date_range("2021-01-01", periods=n, freq="1D")
    files = {"target": _write_bars(tmp_path / "tgt.csv", np.asarray(tgt_opens, float), idx)}
    if bench_opens is not None:
        files["benchmark"] = _write_bars(
            tmp_path / "bench.csv", np.asarray(bench_opens, float), idx
        )
    entry = entry or _lt(1e9)  # fire every bar
    thesis = Thesis.model_validate(
        {
            "name": "b",
            "data": {"targets": ["target"]},
            "entry": entry,
            "params": {"horizon": horizon, **params},
        }
    )
    return run_backtest(thesis, load(thesis, files))


def test_flat_benchmark_excess_equals_raw(tmp_path):
    closes = 100 + np.random.RandomState(5).randn(120).cumsum()
    raw = _bench_run(tmp_path, closes)
    exc = _bench_run(tmp_path, closes, np.full(120, 50.0), benchmark="market")
    np.testing.assert_allclose(
        raw.trades["ret"].to_numpy(dtype=float),
        exc.trades["ret"].to_numpy(dtype=float),
        equal_nan=True,
    )
    assert raw.summary["benchmark"] is None
    assert exc.summary["benchmark"] == "market"
    assert exc.summary["benchmark_source"].endswith("bench.csv")


def test_benchmark_equal_to_target_zeroes_every_return(tmp_path):
    closes = 100 + np.random.RandomState(6).randn(120).cumsum()
    res = _bench_run(tmp_path, closes, closes, benchmark="market")
    closed = res.trades[~res.trades["is_open"]]
    assert len(closed) > 0
    np.testing.assert_allclose(closed["ret"].to_numpy(dtype=float), 0.0, atol=1e-12)


def test_market_excess_matches_hand_computation(tmp_path):
    tgt = np.array([100.0, 102.0, 104.0, 100.0, 108.0, 112.0, 110.0, 118.0, 121.0, 125.0])
    bench = np.array([50.0, 50.5, 51.0, 50.0, 52.0, 53.0, 52.5, 54.0, 54.5, 55.0])
    res = _bench_run(tmp_path, tgt, bench, horizon=3, benchmark="market")
    closed = res.trades[~res.trades["is_open"]].sort_values("entry_time")
    idx = pd.date_range("2021-01-01", periods=len(tgt), freq="1D")
    for e_ts, ret in zip(closed["entry_time"], closed["ret"], strict=True):
        e = idx.get_loc(e_ts)  # the next-open anchor bar
        want = (tgt[e + 3] / tgt[e] - 1.0) - (bench[e + 3] / bench[e] - 1.0)
        assert ret == pytest.approx(want, abs=1e-12)


def test_log_outcome_yields_a_true_log_excess(tmp_path):
    """The benchmark leg is measured in the SAME algebra as the outcome: ln(tgt ratio) −
    ln(bench ratio). A leg hard-coded to PERCENT would subtract a percent from a log."""
    tgt = np.array([100.0, 102.0, 104.0, 100.0, 108.0, 112.0, 110.0, 118.0, 121.0, 125.0])
    bench = np.array([50.0, 50.5, 51.0, 50.0, 52.0, 53.0, 52.5, 54.0, 54.5, 55.0])
    res = _bench_run(tmp_path, tgt, bench, horizon=3, benchmark="market", outcome={"kind": "log"})
    closed = res.trades[~res.trades["is_open"]].sort_values("entry_time")
    assert len(closed) > 0
    idx = pd.date_range("2021-01-01", periods=len(tgt), freq="1D")
    for e_ts, ret in zip(closed["entry_time"], closed["ret"], strict=True):
        e = idx.get_loc(e_ts)
        want = math.log(tgt[e + 3] / tgt[e]) - math.log(bench[e + 3] / bench[e])
        assert ret == pytest.approx(want, abs=1e-12)


def test_benchmark_legs_attribute_the_excess_return(tmp_path):
    """The two legs the excess subtraction consumed are RETAINED, not discarded: per closed row
    ret == ret_raw − ret_bench bit-exactly (they are the very floats `_forward` subtracted), and
    the panel's mean decomposes the same way over the same pool — so a reader can tell "+0.8%
    because the target made +3.2% against a +2.4% market" from "lost less than a falling
    market", which the excess mean alone cannot say."""
    tgt = 100 + np.random.RandomState(11).randn(120).cumsum()
    bench = 50 + np.random.RandomState(12).randn(120).cumsum() * 0.3
    res = _bench_run(tmp_path, tgt, bench, benchmark="market")
    closed = res.trades[~res.trades["is_open"]]
    assert len(closed) > 0
    np.testing.assert_array_equal(
        closed["ret"].to_numpy(dtype=float),
        closed["ret_raw"].to_numpy(dtype=float) - closed["ret_bench"].to_numpy(dtype=float),
    )
    # the raw leg is the hand-computed unbenchmarked return; the bench leg is the market's own
    raw = _bench_run(tmp_path, tgt)
    raw_closed = raw.trades[~raw.trades["is_open"]]
    np.testing.assert_allclose(
        closed.sort_values("entry_time")["ret_raw"].to_numpy(dtype=float),
        raw_closed.sort_values("entry_time")["ret"].to_numpy(dtype=float)[: len(closed)],
        atol=1e-12,
    )
    # the mean identity holds in the cells panel (linearity of the mean over one shared pool —
    # approximate only through float summation order)
    st = res.summary["cells"][0]["by_target"]["target"]
    assert st["mean_ret"] == pytest.approx(st["mean_ret_raw"] - st["mean_ret_bench"], abs=1e-12)
    # censored rows carry no legs: the legs attribute `ret`, and `ret`'s pool is the closed rows
    censored = res.trades[res.trades["is_open"]]
    assert len(censored) > 0  # every-bar entry guarantees terminal right-censoring
    assert censored["ret_raw"].isna().all() and censored["ret_bench"].isna().all()


def test_unbenchmarked_run_raw_leg_is_the_return_and_bench_leg_is_null(tmp_path):
    # No benchmark declared → ret_raw IS ret (same floats, not approximately) and ret_bench is
    # all-null in the frame and the panels alike: "no benchmark leg", never a measured zero.
    closes = 100 + np.random.RandomState(13).randn(120).cumsum()
    res = _bench_run(tmp_path, closes)
    closed = res.trades[~res.trades["is_open"]]
    assert len(closed) > 0
    assert (closed["ret_raw"] == closed["ret"]).all()
    assert res.trades["ret_bench"].isna().all()
    st = res.summary["cells"][0]["by_target"]["target"]
    assert st["mean_ret_raw"] == st["mean_ret"] and math.isnan(st["mean_ret_bench"])


def test_diff_outcome_with_benchmark_is_refused_at_the_library_boundary(tmp_path):
    """The DSL refuses this pairing; the runner refuses it again so a hand-built
    ``model_construct`` thesis cannot reach the excess arithmetic and subtract a return from a
    level."""
    tgt = np.array([100.0, 102.0, 104.0, 100.0, 108.0, 112.0])
    bench = np.array([50.0, 50.5, 51.0, 50.0, 52.0, 53.0])
    idx = pd.date_range("2021-01-01", periods=len(tgt), freq="1D")
    files = {
        "target": _write_bars(tmp_path / "tgt.csv", tgt, idx),
        "benchmark": _write_bars(tmp_path / "bench.csv", bench, idx),
    }
    valid = Thesis.model_validate(
        {
            "name": "b",
            "data": {"targets": ["target"]},
            "entry": _lt(1e9),
            "params": {"horizon": 2, "benchmark": "market", "outcome": {"kind": "pct"}},
        }
    )
    # Bypass validation the way a library caller could, then flip the algebra underneath it.
    forged = valid.model_copy(deep=True)
    object.__setattr__(forged.params.outcome, "kind", "diff")
    with pytest.raises(ValueError, match="incommensurable"):
        run_backtest(forged, load(forged, files))


def test_shortonly_market_excess_flips_sign(tmp_path):
    closes = 100 + np.random.RandomState(7).randn(100).cumsum()
    bench = 50 + np.random.RandomState(8).randn(100).cumsum() * 0.3
    lo = _bench_run(tmp_path, closes, bench, benchmark="market", direction="longonly")
    sh = _bench_run(tmp_path, closes, bench, benchmark="market", direction="shortonly")
    np.testing.assert_allclose(
        sh.trades["ret"].to_numpy(dtype=float),
        -lo.trades["ret"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_missing_benchmark_bars_censor_as_no_benchmark(tmp_path):
    n = 60
    idx = pd.date_range("2021-01-01", periods=n, freq="1D")
    closes = 100 + np.arange(n, dtype=float)
    files = {"target": _write_bars(tmp_path / "tgt.csv", closes, idx)}
    # benchmark file MISSING the final 10 calendar days: reindex (no ffill) leaves NaN there
    bench = pd.DataFrame(
        {"open": 50.0, "high": 50.0, "low": 50.0, "close": 50.0, "volume": 1.0}, index=idx[:-10]
    )
    bench.index.name = "datetime"
    bench.to_csv(tmp_path / "bench.csv")
    files["benchmark"] = str(tmp_path / "bench.csv")

    thesis = Thesis.model_validate(
        {
            "name": "b",
            "data": {"targets": ["target"]},
            "entry": _lt(1e9),
            "params": {"horizon": 5, "benchmark": "market"},
        }
    )
    res = run_backtest(thesis, load(thesis, files))
    reasons = res.trades["exit_reason"].value_counts().to_dict()
    # windows reaching into the benchmark-less zone (but not past the data end) are "no_benchmark";
    # windows past the data end stay "open"
    assert reasons.get("no_benchmark", 0) == 10
    assert reasons.get("open", 0) == 5
    censored = res.trades[res.trades["exit_reason"] == "no_benchmark"]
    assert censored["is_open"].all()
    # the cell's ledger carries both censoring classes distinctly, over its FULL rows
    cov = res.summary["cells"][0]["outcome_coverage"]["target"]
    assert cov["n_attempted"] == len(res.trades)
    assert cov["exit_reasons"]["no_benchmark"] == 10 and cov["exit_reasons"]["open"] == 5
    assert cov["n_closed"] == len(res.trades) - 15  # excluded from the closed stats


def test_benchmark_mode_interior_target_hole_is_no_outcome(tmp_path):
    # Under `benchmark: "market"` the TARGET leg's own in-bounds NaN is "no_outcome", not "open":
    # labelling it "open" would dress a hole that deletes adverse target outcomes as harmless
    # terminal right-censoring. An in-data hole is "no_outcome" on every path; only running past
    # the data end is "open".
    n = 60
    idx = pd.date_range("2021-01-01", periods=n, freq="1D")
    closes = 100 + np.arange(n, dtype=float)
    opens = closes.copy()
    opens[30] = np.nan  # interior hole in the measured (open) leg — nothing to do with the end
    df = pd.DataFrame(
        {"open": opens, "high": closes + 1, "low": closes - 1, "close": closes, "volume": 1.0},
        index=idx,
    )
    df.index.name = "datetime"
    df.to_csv(tmp_path / "tgt.csv")
    bench = pd.DataFrame(
        {"open": 50.0, "high": 50.0, "low": 50.0, "close": 50.0, "volume": 1.0}, index=idx
    )
    bench.index.name = "datetime"
    bench.to_csv(tmp_path / "bench.csv")
    thesis = Thesis.model_validate(
        {
            "name": "b",
            "data": {"targets": ["target"]},
            "entry": _lt(1e9),
            "params": {"horizon": 5, "benchmark": "market"},
        }
    )
    files = {"target": tmp_path / "tgt.csv", "benchmark": tmp_path / "bench.csv"}
    res = run_backtest(thesis, load(thesis, files))
    reasons = res.trades["exit_reason"].value_counts().to_dict()
    # bar 30 is an entry anchor (fill NaN) and the exit of the window opened at bar 25 → both
    # unmeasurable INSIDE the data, so both are no_outcome. Only the last h bars are "open".
    assert reasons.get("no_outcome", 0) == 2
    assert reasons.get("open", 0) == 5
    assert reasons.get("no_benchmark", 0) == 0  # the benchmark is complete; the target is not


# ---- outcome generalization (params.outcome) + series targets ----------------


def _series_bars(path: Path, values: list[float], column: str = "y") -> Path:
    idx = pd.date_range("2021-01-01", periods=len(values), freq="1D")
    df = pd.DataFrame({column: pd.Series(values, index=idx, dtype=float)})
    df.index.name = "datetime"
    df.to_csv(path)
    return path


def test_series_target_loads_and_synthesizes_ohlc(tmp_path):
    _series_bars(tmp_path / "y.csv", [3.0, 3.1, 3.2, 3.0, 2.9])
    thesis = Thesis.model_validate(
        {"name": "t", "data": {"targets": ["target"]}, "entry": _gt(0.0)}
    )
    md = load(thesis, {"target": tmp_path / "y.csv"})
    assert md.target_shape == "series"
    assert md.volume is None
    np.testing.assert_array_equal(md.open.to_numpy(), md.close.to_numpy())
    np.testing.assert_array_equal(md.high.to_numpy(), md.close.to_numpy())


def _measured(tmp_path, values, kind, *, entry_at=-1e9, horizon=1):
    """Run a one-target series thesis and return its trades frame."""
    path = _series_bars(tmp_path / "y.csv", values)
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": _gt(entry_at),
            "params": {"horizon": horizon, "outcome": {"kind": kind}},
        }
    )
    return run_backtest(thesis, load(thesis, {"target": path})).trades


def test_pct_outcome_censors_non_positive_endpoints(tmp_path):
    """A percent change across zero returns a FINITE number with an inverted sign, which would
    pass for a real return — so it is censored as `no_outcome`, which the per-cell missingness
    contract refuses."""
    # 1.0 → -0.5 → -1.0 → 2.0: every window touching a non-positive level is unmeasurable.
    trades = _measured(tmp_path, [1.0, -0.5, -1.0, 2.0, 3.0], "pct")
    by_bar = {int(r.entry_bar): r for r in trades.itertuples()}
    # bar 0 fills at open[1] = -0.5 (non-positive denominator) → censored, not a -150% "return"
    assert by_bar[0].exit_reason == "no_outcome" and math.isnan(by_bar[0].ret)
    # bar 1 measures -1.0 → 2.0: a negative denominator with a positive numerator
    assert by_bar[1].exit_reason == "no_outcome"
    # a fully positive window still measures normally
    assert by_bar[2].exit_reason == "horizon"
    np.testing.assert_allclose(by_bar[2].ret, 3.0 / 2.0 - 1.0)


def test_log_outcome_censors_two_negative_endpoints(tmp_path):
    """The subtle half: a single negative endpoint makes ln() NaN on its own, but TWO negative
    endpoints give a positive ratio and a finite, meaningless value (−4 → −2 read as +0.69)."""
    # The measured window is open[t+1] → open[t+2], so bar 0 spans -4.0 → -2.0: a POSITIVE
    # ratio, which ln() happily reports as +0.69 — a 50% "gain" while the level fell.
    trades = _measured(tmp_path, [10.0, -4.0, -2.0, 5.0, 20.0], "log")
    by_bar = {int(r.entry_bar): r for r in trades.itertuples()}
    assert by_bar[0].exit_reason == "no_outcome" and math.isnan(by_bar[0].ret)
    assert by_bar[1].exit_reason == "no_outcome"  # -2.0 → 5.0, sign change
    assert by_bar[2].exit_reason == "horizon"
    np.testing.assert_allclose(by_bar[2].ret, math.log(20.0 / 5.0))


def test_diff_outcome_still_measures_signed_levels(tmp_path):
    """`diff` exists precisely for series that cross zero — it must be untouched."""
    trades = _measured(tmp_path, [1.0, -0.5, -1.0, 2.0], "diff")
    by_bar = {int(r.entry_bar): r for r in trades.itertuples()}
    assert by_bar[0].exit_reason == "horizon"
    np.testing.assert_allclose(by_bar[0].ret, -1.0 - (-0.5))
    assert by_bar[1].exit_reason == "horizon"
    np.testing.assert_allclose(by_bar[1].ret, 2.0 - (-1.0))


def test_sign_flipping_an_endpoint_can_only_censor(tmp_path):
    """Deletion monotonicity for the outcome domain: making an endpoint non-positive removes
    an observation; it can never mint one."""
    good = _measured(tmp_path, [1.0, 2.0, 4.0, 8.0], "pct")
    assert set(good["exit_reason"]) <= {"horizon", "open"}  # only terminal censoring
    flipped = _measured(tmp_path, [1.0, -2.0, 4.0, 8.0], "pct")
    n_closed_before = int((good["exit_reason"] == "horizon").sum())
    n_closed_after = int((flipped["exit_reason"] == "horizon").sum())
    assert n_closed_after < n_closed_before


def test_series_target_multi_column_needs_the_invocation_to_bind_one(tmp_path):
    # A yield-curve CSV holds several series; ONE thesis measures whichever of them the invocation
    # binds. Unbound, the file is ambiguous and refuses (naming the pair that resolves it) — and
    # the SAME document, bound to the other column, measures the other series. That is the point
    # of binding at the invocation: the choice of column never perturbs the thesis's identity.
    idx = pd.date_range("2021-01-01", periods=4, freq="1D")
    pd.DataFrame({"y2": [1.0] * 4, "y10": [3.0, 3.1, 3.0, 2.9]}, index=idx).to_csv(
        tmp_path / "curve.csv"
    )
    files = {"target": tmp_path / "curve.csv"}
    thesis = Thesis.model_validate(
        {"name": "t", "data": {"targets": ["target"]}, "entry": _gt(0.0)}
    )
    with pytest.raises(ValueError, match=r"bind --column target=COL"):
        load(thesis, files)
    md = load(thesis, files, {"target": "y10"})
    np.testing.assert_allclose(md.close["target"].to_numpy(), [3.0, 3.1, 3.0, 2.9])
    other = load(thesis, files, {"target": "y2"})
    np.testing.assert_allclose(other.close["target"].to_numpy(), [1.0] * 4)


def test_a_column_binding_on_an_ohlcv_target_refuses(tmp_path):
    # A price target always measures its open-anchored prices, so there is no column for a binding
    # to choose and the pair can only be a mistaken request. The refusal quotes the binding back
    # verbatim rather than describing it, because the caller's fix is to delete that argv token.
    _bars(tmp_path / "px.csv", [100.0, 101.0, 102.0])
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": _gt(0.0),
        }
    )
    with pytest.raises(ValueError, match=r"--column target=close is bound but the file is OHLCV"):
        load(thesis, {"target": tmp_path / "px.csv"}, {"target": "close"})


def test_mixed_target_shapes_rejected(tmp_path):
    _bars(tmp_path / "px.csv", [100.0, 101.0, 102.0])
    _series_bars(tmp_path / "y.csv", [3.0, 3.1, 3.2])
    thesis = Thesis.model_validate(
        {"name": "t", "data": {"targets": ["px", "y"]}, "entry": _gt(0.0)}
    )
    with pytest.raises(ValueError, match="mix OHLCV and series"):
        load(thesis, {"px": tmp_path / "px.csv", "y": tmp_path / "y.csv"})


def test_diff_outcome_on_series_target_hand_computed(tmp_path):
    # yield target, diff outcome, longonly: firing bar t anchors at value[t+1], measures
    # value[t+1+h] − value[t+1] — in the series' own units, no percent anywhere.
    vals = [3.0, 3.5, 3.2, 3.4, 3.9, 3.1]
    _series_bars(tmp_path / "y.csv", vals)
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": _gt(3.45),  # fires at bars 1 (3.5) and 4 (3.9)
            "params": {"horizon": 2, "outcome": {"kind": "diff"}},
        }
    )
    res = run_backtest(thesis, load(thesis, {"target": tmp_path / "y.csv"}))
    closed = res.trades[~res.trades["is_open"]]
    assert len(closed) == 1  # bar 4's window runs past the data end (censored "open")
    row = closed.iloc[0]
    assert row.entry_px == 3.2 and row.exit_px == 3.9
    assert row.ret == pytest.approx(3.9 - 3.2)
    assert res.summary["outcome"] == {"series": "target", "kind": "diff", "units": "level_diff"}
    assert res.summary["target_shape"] == "series"


def test_log_outcome_hand_computed(tmp_path):
    _bars(tmp_path / "px.csv", [110.0, 95.0, 102.0, 108.0, 100.0])
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": _lt(100.0),  # fires at bar 1
            "params": {"horizon": 2, "outcome": {"kind": "log"}},
        }
    )
    res = run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"}))
    row = res.trades[~res.trades["is_open"]].iloc[0]
    assert row.ret == pytest.approx(math.log(100.0 / 102.0))
    assert res.summary["outcome"] == {"series": "target", "kind": "log", "units": "log"}


def test_feed_outcome_measures_the_feed_not_the_target(tmp_path):
    # "when the target dips, the IV feed rises": the observation measures the FEED's forward diff.
    _bars(tmp_path / "px.csv", [110, 95, 102, 99, 100, 99, 110])
    idx = pd.date_range("2021-01-01", periods=7, freq="1D")
    pd.DataFrame({"iv": [20.0, 21.0, 22.0, 25.0, 24.0, 23.0, 26.0]}, index=idx).to_csv(
        tmp_path / "iv.csv"
    )
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"], "external": {"iv": {}}},
            "entry": _lt(100.0),  # fires at bars 1 (95), 3 (99), 5 (99)
            "params": {"horizon": 1, "outcome": {"series": "iv", "kind": "diff"}},
        }
    )
    files = {"target": tmp_path / "px.csv", "iv": tmp_path / "iv.csv"}
    res = run_backtest(thesis, load(thesis, files))
    closed = res.trades[~res.trades["is_open"]].sort_values("entry_time")
    # bar 1 → iv[2]..iv[3]: 25−22 = 3; bar 3 → iv[4]..iv[5]: 23−24 = −1; bar 5 runs past the end
    assert list(closed["ret"]) == pytest.approx([3.0, -1.0])
    assert closed.iloc[0]["entry_px"] == 22.0  # the measured value at the anchor is the FEED's


def test_feed_outcome_hole_censors_as_no_outcome(tmp_path):
    # A firing whose anchor precedes the feed's first stamp has no measurable outcome — censored
    # distinctly ("no_outcome"), not conflated with running past the data end ("open").
    _bars(tmp_path / "px.csv", [95, 96, 102, 108, 99, 100, 110])
    idx = pd.date_range("2021-01-01", periods=7, freq="1D")
    iv = pd.DataFrame({"iv": [np.nan, np.nan, np.nan, 25.0, 24.0, 23.0, 26.0]}, index=idx).dropna()
    iv.to_csv(tmp_path / "iv.csv")
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"], "external": {"iv": {}}},
            "entry": _lt(100.0),  # fires at bars 0 (95), 1 (96), 4 (99)
            "params": {"horizon": 1, "outcome": {"series": "iv", "kind": "diff"}},
        }
    )
    files = {"target": tmp_path / "px.csv", "iv": tmp_path / "iv.csv"}
    res = run_backtest(thesis, load(thesis, files))
    reasons = res.trades.sort_values("entry_time")["exit_reason"].tolist()
    # bar 0 anchors at bar 1 (feed not yet stamped) → no_outcome; bar 1 anchors at bar 2 (still
    # unstamped) → no_outcome; bar 4 anchors at iv[5]=23, exits iv[6]=26 → closed "horizon".
    assert reasons == ["no_outcome", "no_outcome", "horizon"]


def test_default_run_summary_outcome_stamp_is_always_explicit(tmp_path):
    # The default run spells its algebra out rather than stamping a None a consumer would have to
    # KNOW meant "pct on target" — no null carries meaning.
    px = _bars(tmp_path / "px.csv", [110, 95, 102, 108, 100])
    tr_thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": _lt(100.0),
            "params": {"horizon": 1},
        }
    )
    res = run_backtest(tr_thesis, load(tr_thesis, {"target": px}))
    assert res.summary["outcome"] == {"series": "target", "kind": "pct", "units": "fraction"}
    assert res.summary["target_shape"] == "ohlcv"


# ---- the two fail-closed data-integrity ledgers ------------------------------


def _hole_thesis(tmp_path, feed_values, horizon=5):
    """A single-combo, single-horizon thesis whose entry reads an external feed (so a NaN in the
    feed is a DECISION-input hole, not an outcome hole)."""
    n = len(feed_values)
    idx = pd.date_range("2015-01-01", periods=n, freq="1D")
    closes = 100 + np.arange(n, dtype=float)
    df = pd.DataFrame(
        {"open": closes, "high": closes + 1, "low": closes - 1, "close": closes, "volume": 1.0},
        index=idx,
    )
    df.index.name = "datetime"
    df.to_csv(tmp_path / "px.csv")
    feed = pd.DataFrame({"x": pd.Series(feed_values, index=idx, dtype=float)})
    feed.index.name = "datetime"
    feed.to_csv(tmp_path / "x.csv")
    return Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"], "external": {"x": {}}},
            "entry": {
                "type": "threshold",
                "left": {"type": "external", "name": "x"},
                "op": ">",
                "right": {"type": "constant", "value": 0.0},
            },
            "params": {"horizon": horizon},
        }
    )


def _hole_files(tmp_path) -> dict[str, Path]:
    """The two files ``_hole_thesis`` writes, under the keys its DSL names them by."""
    return {"target": tmp_path / "px.csv", "x": tmp_path / "x.csv"}


def test_signal_coverage_denominator_is_pure_index_geometry(tmp_path):
    # `n_bars` is the joined index length — pure geometry no property of the data can shrink —
    # which is what makes `n_undefined <= n_bars` arithmetic a reader re-checks rather than a
    # claim. A clean feed leaves n_undefined == 0.
    n = 400
    thesis = _hole_thesis(tmp_path, [1.0] * n)
    s = run_backtest(thesis, load(thesis, _hole_files(tmp_path))).summary
    for cell in s["cells"]:
        sig = cell["signal_coverage"]
        assert set(sig) == set(s["targets"])
        assert sig["target"]["n_bars"] == n == s["n_bars"]
        assert sig["target"]["n_undefined"] == 0


def test_signal_coverage_counts_every_undecidable_decision_bar(tmp_path):
    # A hole in the decision feed suppresses a firing and leaves NO trace in the outcome ledger
    # (which can only account for bars that FIRED) — this is the ledger that sees it. There is no
    # pool to attribute it to: one cell, one interval, one count.
    n = 400
    values = [1.0] * n
    values[10] = np.nan
    values[395] = np.nan
    thesis = _hole_thesis(tmp_path, values)
    result = run_backtest(thesis, load(thesis, _hole_files(tmp_path)))
    s = result.summary
    assert s["cells"][0]["signal_coverage"]["target"]["n_undefined"] == 2
    # …and the suppressed firings are invisible to the outcome ledger, which is the whole point.
    assert 10 not in set(result.trades["entry_bar"])
    assert 395 not in set(result.trades["entry_bar"])


def test_sources_panel_is_always_present_and_run_level(tmp_path):
    # Combo-independent by construction (the leaf SET is a property of the entry tree), so the
    # per-source availability panel is a run-level stamp — and it is built whether or not anything
    # fired, because a run whose sources are holed must say so even when the holes suppressed
    # every firing.
    thesis = _hole_thesis(tmp_path, [1.0] * 200)
    s = run_backtest(thesis, load(thesis, _hole_files(tmp_path))).summary
    assert set(s["sources"]) == set(s["targets"])
    panel = s["sources"]["target"]
    assert panel["n_bars"] == s["n_bars"] and panel["n_missing"] == 0
    # only the leaves the ENTRY TREE reads are decision inputs — the price frame here is measured,
    # never decided on, and the feature snapshots are evidence, so neither appears
    assert set(panel["by_source"]) == {"external:x"}
    for leaf in panel["by_source"].values():
        assert leaf["n_missing"] == 0 and isinstance(leaf["first_available"], str)
    # a thesis that NEVER fires still carries the panel
    never = _hole_thesis(tmp_path, [-1.0] * 200)
    s_never = run_backtest(never, load(never, _hole_files(tmp_path))).summary
    assert _closed_n(s_never) == 0
    assert s_never["sources"]["target"]["n_bars"] == s_never["n_bars"]


def test_sources_panel_counts_a_hole_after_first_availability_only(tmp_path):
    # A source that merely STARTS LATE is warmup, not a hole (its `first_available` is evidence);
    # a hole strictly after it is counted. The external feed's availability is its post-asof-ffill
    # value, so the leading unstamped bars are the late start.
    n = 200
    values = [np.nan] * 20 + [1.0] * (n - 20)
    thesis = _hole_thesis(tmp_path, values)
    md = load(thesis, _hole_files(tmp_path))
    s = run_backtest(thesis, md).summary
    leaf = s["sources"]["target"]["by_source"]["external:x"]
    assert leaf["n_missing"] == 0  # the late start is warmup, not missingness
    assert leaf["first_available"] == md.index[20].isoformat()
    assert s["sources"]["target"]["n_missing"] == 0


# ---- cross-sectional entry (basket mode: CrossRank across targets) -----------
# The cross family ranks WITHIN the declared basket at each bar, so it is legal only under
# `target_mode: "basket"` — the mode that declares the targets one cross-section rather than a
# regime conjunction.


def test_cross_rank_bottom_tercile_end_to_end(tmp_path):
    # Three deterministic targets: A declines, B drifts up, C rises fastest — so A is always the
    # bottom of the change(close, 5) cross-section (rank 0.0), B the middle (0.5), C the top
    # (1.0). The entry `cross_rank(change(close, 5)) <= 0.34` must therefore fire ONLY on A, on
    # every bar where the 5-bar change is defined for all three (k = 3 >= min_valid).
    n = 40
    idx = pd.date_range("2021-01-01", periods=n, freq="1D")
    slopes = {"A": -0.5, "B": 0.2, "C": 0.5}
    files = {}
    for sym, slope in slopes.items():
        close = 100.0 + slope * np.arange(n)
        _write_bars(tmp_path / f"{sym}.csv", close, idx)
        files[sym] = str(tmp_path / f"{sym}.csv")

    thesis = Thesis.model_validate(
        {
            "name": "bottom_tercile",
            "target_mode": "basket",
            "data": {"targets": list(files)},
            "entry": {
                "type": "threshold",
                "left": {
                    "type": "cross_rank",
                    "input": {
                        "type": "change",
                        "periods": 5,
                        "input": {"type": "field", "column": "close"},
                    },
                    "min_valid": 3,
                },
                "op": "<=",
                "right": {"type": "constant", "value": 0.34},
            },
            "params": {"horizon": 3},
        }
    )
    res = run_backtest(thesis, load(thesis, files))
    trades = res.trades

    # only the bottom target fires, on every bar from change(5) warmup (bar 5) to the last
    # anchorable bar; entry_time is the NEXT-open anchor bar t+1, so the firing bar is one earlier
    assert set(trades["target"]) == {"A"}
    fire_bars = sorted(idx.get_loc(ts) - 1 for ts in trades["entry_time"])
    assert fire_bars == list(range(5, n - 1))  # t = n-1 has no open[t+1] anchor

    # closed observations measure A's open[e] -> open[e+3] (e = the anchor bar); A moves -0.5/bar
    # so every return is a hand-checkable negative number
    closed = trades[~trades["is_open"]]
    a_open = 100.0 + slopes["A"] * np.arange(n)
    for e, ret in zip((idx.get_loc(ts) for ts in closed["entry_time"]), closed["ret"], strict=True):
        assert ret == pytest.approx(a_open[e + 3] / a_open[e] - 1.0)

    # the summary says which target semantics produced it — the checklist dispatches on this
    assert res.summary["target_mode"] == "basket"
    # non-firing members still hold their place in every per-cell panel (basket never narrows
    # the declared cross-section any more than conjunction narrows the regime)
    assert set(res.summary["cells"][0]["by_target"]) == {"A", "B", "C"}


def test_cross_rank_min_valid_censors_thin_cross_section(tmp_path):
    # Two targets ranked on a per-target external feed with one member's values holed over an
    # INTERIOR stretch: the cross-section drops to k = 1 < min_valid = 2 there, so cross_rank is
    # NaN for EVERY member — the threshold mints defined=False and the stretch fires nothing.
    # The three-valued ledger must see it on BOTH siblings: a hole in one member censors every
    # member's rank, and each target's
    # `signal_coverage` counts the undecidable bars — the fail-closed behavior that keeps a
    # vendor outage in one name from silently thinning the whole basket's evidence.
    n = 60
    hole = range(20, 40)
    idx = pd.date_range("2021-01-01", periods=n, freq="1D")
    files = {}
    for sym in ("AAA", "BBB"):
        _write_bars(tmp_path / f"{sym}.csv", 100 + np.arange(n, dtype=float), idx)
        files[sym] = str(tmp_path / f"{sym}.csv")
    mom_a = pd.DataFrame({"mom": np.arange(n, dtype=float)}, index=idx)
    mom_a.index.name = "datetime"
    mom_a.to_csv(tmp_path / "mom_a.csv")
    vals = np.arange(n, dtype=float)
    vals[list(hole)] = np.nan
    mom_b = pd.DataFrame({"mom": vals}, index=idx)
    mom_b.index.name = "datetime"
    mom_b.to_csv(tmp_path / "mom_b.csv")

    thesis = Thesis.model_validate(
        {
            "name": "thin_cross_section",
            "target_mode": "basket",
            "data": {
                "targets": list(files),
                "external": {"mom": {"per_target": True}},
            },
            "entry": {
                "type": "threshold",
                "left": {"type": "cross_rank", "input": {"type": "external", "name": "mom"}},
                "op": "<=",
                "right": {"type": "constant", "value": 1.0},
            },
            "params": {"horizon": 2},
        }
    )
    # a per-target feed answers to one derived key per member
    files |= {"mom@AAA": str(tmp_path / "mom_a.csv"), "mom@BBB": str(tmp_path / "mom_b.csv")}
    res = run_backtest(thesis, load(thesis, files))
    # the holed stretch fires nothing for ANYONE — k = 1 < min_valid censors the whole bar
    assert not (set(res.trades["entry_bar"].astype(int)) & set(hole))
    # …and both siblings' three-valued ledgers count the stretch: the hole sits in BBB's feed,
    # but the RANK it corrupts is everyone's (BBB undefined on its own NaN, AAA on the thin k)
    sig = res.summary["cells"][0]["signal_coverage"]
    assert sig["AAA"]["n_undefined"] == len(hole)
    assert sig["BBB"]["n_undefined"] == len(hole)
    # the raw source panel localizes WHOSE feed carried the hole — only BBB's
    assert res.summary["sources"]["BBB"]["by_source"]["external:mom"]["n_missing"] == len(hole)
    assert res.summary["sources"]["AAA"]["by_source"]["external:mom"]["n_missing"] == 0


def test_cross_agg_member_cannot_fire_before_its_own_series_exists(tmp_path):
    # The staggered-listing seam, end to end (the unit-level init pin lives in test_vectorize):
    # CrossAgg broadcasts the group's aggregate into EVERY column wherever k >= min_valid, so a
    # member whose own series lists LATE still SEES the basket's breadth from bar 0 — the
    # sanctioned value semantics — but its warmup latch is gated by its OWN input, so it cannot
    # fire before its own data begins. Before the gate, the late member fired off the broadcast
    # into its own missing prices and every such firing censored `no_outcome`, hard-refusing the
    # cell for what is ordinary listing geometry, not a data hole.
    n, K, h = 60, 20, 3
    idx = pd.date_range("2021-01-01", periods=n, freq="1D")
    values = {
        "A": 10.0 + 0.1 * np.arange(n),
        "B": 20.0 + 0.1 * np.arange(n),
        "C": np.concatenate([np.full(K, np.nan), 30.0 + 0.1 * np.arange(n - K)]),
    }
    files = {}
    for sym, vals in values.items():
        df = pd.DataFrame({"y": pd.Series(vals, index=idx)})
        df.index.name = "datetime"
        df.to_csv(tmp_path / f"{sym}.csv")
        files[sym] = str(tmp_path / f"{sym}.csv")
    thesis = Thesis.model_validate(
        {
            "name": "staggered_listing",
            "target_mode": "basket",
            "data": {"targets": list(files)},
            "entry": {
                "type": "threshold",
                "left": {
                    "type": "cross_agg",
                    "agg": "mean",
                    "min_valid": 2,
                    "input": {"type": "field", "column": "close"},
                },
                "op": ">",
                "right": {"type": "constant", "value": 0.0},
            },
            "params": {"horizon": h},
        }
    )
    res = run_backtest(thesis, load(thesis, files))
    trades = res.trades
    # every member fires — the aggregate is defined from bar 0 (A and B keep k = 2 >= min_valid)
    assert set(trades["target"]) == {"A", "B", "C"}
    by_tgt = {t: g["entry_bar"].astype(int) for t, g in trades.groupby("target")}
    # …the on-time members from bar 0, the late member from ITS first finite bar exactly: not
    # bar 0 (the broadcast is not its warmup) and not later (the aggregate was already defined)
    assert int(by_tgt["A"].min()) == 0 and int(by_tgt["B"].min()) == 0
    assert int(by_tgt["C"].min()) == K
    # C's firings all anchor inside its own data, so NOTHING censors as an in-data hole — firing C
    # into its missing prices would fill this ledger with no_outcome instead
    cov = res.summary["cells"][0]["outcome_coverage"]
    for tgt in ("A", "B", "C"):
        assert cov[tgt]["exit_reasons"]["no_outcome"] == 0
    # and the leading stretch is C's WARMUP (vacuously defined), never an undecidable hole
    sig = res.summary["cells"][0]["signal_coverage"]
    assert all(sig[t]["n_undefined"] == 0 for t in ("A", "B", "C"))


def test_cross_mean_excess_is_antisymmetric_for_two_targets(tmp_path):
    # `benchmark: "cross_mean"` (basket-only) demeans each member's signed forward return by the
    # basket mean at every bar, so with two members the excesses are EXACTLY opposite bar by bar:
    # r_A − m = −(r_B − m). Clean data → every leg finite → the fail-closed whole-bar censoring
    # never triggers here, and the arithmetic is the engine's own.
    n = 100
    idx = pd.date_range("2021-01-01", periods=n, freq="1D")
    rng = np.random.RandomState(9)
    a = 100 + rng.randn(n).cumsum()
    b = 80 + rng.randn(n).cumsum()
    files = {
        "A": _write_bars(tmp_path / "A.csv", a, idx),
        "B": _write_bars(tmp_path / "B.csv", b, idx),
    }
    thesis = Thesis.model_validate(
        {
            "name": "cm",
            "target_mode": "basket",
            "data": {"targets": ["A", "B"]},
            "entry": _lt(1e9),
            "params": {"horizon": 5, "benchmark": "cross_mean"},
        }
    )
    res = run_backtest(thesis, load(thesis, files))
    closed = res.trades[~res.trades["is_open"]]
    ra = closed[closed["target"] == "A"].sort_values("entry_time")["ret"].to_numpy(dtype=float)
    rb = closed[closed["target"] == "B"].sort_values("entry_time")["ret"].to_numpy(dtype=float)
    np.testing.assert_allclose(ra, -rb, atol=1e-12)
    # …and match the hand-computed demeaned forward returns (anchor e = firing bar + 1)
    fa = a[6:] / a[1:-5] - 1.0
    fb = b[6:] / b[1:-5] - 1.0
    np.testing.assert_allclose(ra, (fa - (fa + fb) / 2.0)[: len(ra)], atol=1e-12)
    # the benchmark self-description names the mode-gated literal; there is no external source
    assert res.summary["benchmark"] == "cross_mean"
    assert res.summary["benchmark_source"] is None
    # the legs attribute the demeaning exactly like the market case: each closed row's bench leg
    # is its bar's basket mean of the signed raw legs, ret == ret_raw − ret_bench bit-exactly,
    # and the pooled mean_ret_bench is the pooled raw mean — THIS fixture fires basket-wide
    # (_lt(1e9): every member, every bar), the one case the pooled excess mean is ≈ 0 by
    # construction; a SELECTIVE signal's pooled mean is the firing subset's tilt, not ≈ 0
    np.testing.assert_array_equal(
        closed["ret"].to_numpy(dtype=float),
        closed["ret_raw"].to_numpy(dtype=float) - closed["ret_bench"].to_numpy(dtype=float),
    )
    bench_a = closed[closed["target"] == "A"].sort_values("entry_time")["ret_bench"]
    bench_b = closed[closed["target"] == "B"].sort_values("entry_time")["ret_bench"]
    np.testing.assert_allclose(
        bench_a.to_numpy(dtype=float), bench_b.to_numpy(dtype=float)[: len(bench_a)], atol=0
    )  # one bar, ONE benchmark leg — both members subtract the same basket mean
    pooled = res.summary["cells"][0]["pooled"]
    # ≈ 0 BECAUSE the firings are basket-wide (complete cross-sections demean to zero) —
    # never an identity of cross_mean itself: a selective pool's mean is its selection tilt
    assert pooled["mean_ret"] == pytest.approx(0.0, abs=1e-12)
    assert pooled["mean_ret_raw"] == pytest.approx(pooled["mean_ret_bench"], abs=1e-12)


def test_cross_mean_member_hole_censors_the_whole_bar(tmp_path):
    # Fail-closed missingness for the basket's own benchmark, mirroring the market-benchmark pin
    # above (test_missing_benchmark_bars_censor_as_no_benchmark) with the hole in a MEMBER file:
    # the basket mean is over ALL declared members' legs, so one member's NaN leg kills the WHOLE
    # bar — the HEALTHY member's firing there censors as `no_benchmark`, never closes against a
    # partial-basket mean. This test FAILS if `_forward`'s `fwd.mean` were replaced by
    # `np.nanmean`: the surviving member would be quietly demeaned by a different (one-member)
    # basket, its rows would close as `horizon`, and no row would ever carry `no_benchmark`.
    n, h = 60, 5
    holes = [30, 31]
    idx = pd.date_range("2021-01-01", periods=n, freq="1D")
    rng = np.random.RandomState(13)
    a = 100 + rng.randn(n).cumsum()
    b = 80 + rng.randn(n).cumsum()
    _write_bars(tmp_path / "A.csv", a, idx)
    # member B: the measured (open) leg holed at two interior bars; close stays intact, so the
    # entry still fires everywhere and the hole is purely an OUTCOME-side event
    b_df = pd.DataFrame({"open": b, "high": b, "low": b, "close": b, "volume": 1000.0}, index=idx)
    b_df.iloc[holes, b_df.columns.get_loc("open")] = np.nan
    b_df.index.name = "datetime"
    b_df.to_csv(tmp_path / "B.csv")
    thesis = Thesis.model_validate(
        {
            "name": "cm_hole",
            "target_mode": "basket",
            "data": {"targets": ["A", "B"]},
            "entry": _lt(1e9),
            "params": {"horizon": h, "benchmark": "cross_mean"},
        }
    )
    files = {"A": str(tmp_path / "A.csv"), "B": str(tmp_path / "B.csv")}
    res = run_backtest(thesis, load(thesis, files))
    trades = res.trades
    # a hole at open[m] kills the anchors reading it: the fill at t+1 == m and the exit landing
    # at t+1+h == m — four firing bars in all for two holed bars
    affected = sorted({m - 1 for m in holes} | {m - 1 - h for m in holes})
    a_rows = trades[trades["target"] == "A"].set_index("entry_bar")
    for t in affected:
        row = a_rows.loc[t]
        assert row["exit_reason"] == "no_benchmark"  # the whole bar dies, healthy member included
        assert math.isnan(row["ret"])  # excluded from every closed statistic
    # the holed member's own leg is unmeasurable there — an in-data hole, distinctly `no_outcome`
    b_rows = trades[trades["target"] == "B"].set_index("entry_bar")
    for t in affected:
        assert b_rows.loc[t]["exit_reason"] == "no_outcome"
    # the per-target ledgers carry the censoring classes distinctly — and n excludes the bar
    cov = res.summary["cells"][0]["outcome_coverage"]
    assert cov["A"]["exit_reasons"]["no_benchmark"] == len(affected) > 0
    assert cov["A"]["exit_reasons"]["no_outcome"] == 0
    assert cov["B"]["exit_reasons"]["no_outcome"] == len(affected)
    assert cov["B"]["exit_reasons"]["no_benchmark"] == 0
    assert res.summary["cells"][0]["by_target"]["A"]["n"] == cov["A"]["exit_reasons"]["horizon"]
    # clean bars still demean against the FULL two-member basket: retA == −retB exactly
    closed = trades[~trades["is_open"]]
    ra = closed[closed["target"] == "A"].sort_values("entry_bar")["ret"].to_numpy(dtype=float)
    rb = closed[closed["target"] == "B"].sort_values("entry_bar")["ret"].to_numpy(dtype=float)
    np.testing.assert_allclose(ra, -rb, atol=1e-12)


# ---- basket library-boundary backstops ---------------------------------------
# Every mode-gating rule the DSL enforces is re-raised inside `run_backtest`, so a hand-built
# `model_construct`/`__setattr__` thesis cannot slide a refused pairing past validation and into
# the measurement arithmetic. Modeled on the market+diff backstop above
# (test_diff_outcome_with_benchmark_is_refused_at_the_library_boundary): forge the one field
# validation would have refused, and the runner refuses it again.


def test_basket_diff_outcome_is_refused_at_the_library_boundary(tmp_path):
    # Basket + diff pools level-unit returns across members the engine cannot certify
    # commensurable; the DSL refuses the pairing, and the runner re-refuses the forged spelling.
    idx = pd.date_range("2021-01-01", periods=40, freq="1D")
    files = {
        "A": _write_bars(tmp_path / "A.csv", 100 + np.arange(40, dtype=float), idx),
        "B": _write_bars(tmp_path / "B.csv", 80 + np.arange(40, dtype=float), idx),
    }
    valid = Thesis.model_validate(
        {
            "name": "b",
            "target_mode": "basket",
            "data": {"targets": ["A", "B"]},
            "entry": _lt(1e9),
            "params": {"horizon": 2, "outcome": {"kind": "pct"}},
        }
    )
    forged = valid.model_copy(deep=True)
    object.__setattr__(forged.params.outcome, "kind", "diff")
    with pytest.raises(ValueError, match="incommensurable"):
        run_backtest(forged, load(forged, files))


def test_cross_mean_outside_basket_is_refused_at_the_library_boundary(tmp_path):
    # cross_mean couples the targets in the OUTCOME exactly as cross nodes couple them in the
    # signal — legal only where the mode declares one cross-section, and a single-target
    # conjunction has no basket to demean against.
    idx = pd.date_range("2021-01-01", periods=40, freq="1D")
    px = _write_bars(tmp_path / "tgt.csv", 100 + np.arange(40, dtype=float), idx)
    valid = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": _lt(1e9),
            "params": {"horizon": 2},
        }
    )
    forged = valid.model_copy(deep=True)
    object.__setattr__(forged.params, "benchmark", "cross_mean")
    with pytest.raises(ValueError, match="cross_mean"):
        run_backtest(forged, load(forged, {"target": px}))


def test_cross_nodes_outside_basket_are_refused_at_the_library_boundary(tmp_path):
    # The transforms-layer >= 2-columns check cannot see the MODE, so without this backstop a
    # forged conjunction thesis could rank across a regime's targets — the exact coupling
    # conjunction exists to forbid. The runner scans the entry (and features) for cross nodes
    # and refuses them under any stamp but basket.
    idx = pd.date_range("2021-01-01", periods=40, freq="1D")
    files = {
        "A": _write_bars(tmp_path / "A.csv", 100 + np.arange(40, dtype=float), idx),
        "B": _write_bars(tmp_path / "B.csv", 80 - np.arange(40, dtype=float) * 0.1, idx),
    }
    valid = Thesis.model_validate(
        {
            "name": "x",
            "target_mode": "basket",
            "data": {"targets": ["A", "B"]},
            "entry": {
                "type": "threshold",
                "left": {"type": "cross_rank", "input": {"type": "field", "column": "close"}},
                "op": "<=",
                "right": {"type": "constant", "value": 0.5},
            },
            "params": {"horizon": 2},
        }
    )
    forged = valid.model_copy(deep=True)
    object.__setattr__(forged, "target_mode", "conjunction")
    with pytest.raises(ValueError, match="cross-sectional node"):
        run_backtest(forged, load(forged, files))


# ---- basket mode: the per-cell POOLED cross-target block ---------------------
# The panel the basket checklist grades INSTEAD of per-member floors. `by_target` stays exactly
# as it is in conjunction (attribution evidence, read by no check); `pooled` is additive and
# reconciles with everything it sits beside.

#: The by_target per-pool key set — UNCHANGED from conjunction (pinned in
#: test_run_backtest_end_to_end); the basket must add a panel, never rewrite one.
_BY_TARGET_KEYS = {
    "n",
    "n_nonoverlap",
    "mean_ret",
    "mean_ret_raw",
    "mean_ret_bench",
    "benchmark_regression",
    "hit_rate",
    "win_loss_ratio",
    "profit_factor",
    "std_ret",
    "skewness",
    "kurtosis",
    "tail_ratio",
    "cvar_5",
    "t_hac",
    "hac_se",
    "rot_p",
    "rot_n_null",
    "concentration",
    "boot",
    "subperiods",
    "ret_quantiles",
    "worst_ret",
    "best_ret",
    "mae_quantiles",
    "mfe_quantiles",
    "edge_ratio",
    "timing",
}

#: The pooled panel mirrors by_target's shape (one mental model) plus the member decomposition.
_POOLED_KEYS = _BY_TARGET_KEYS | {"member_share"}


def _basket_files(tmp_path, n=420) -> dict[str, str]:
    files = {}
    for i, sym in enumerate(("AAA", "BBB")):
        _noise_px(tmp_path / f"{sym}.csv", n=n, seed=10 + i)
        files[sym] = str(tmp_path / f"{sym}.csv")
    return files


def _pctl_thesis(targets, *, target_mode, horizon, cuts=0.35):
    right = (
        {"type": "constant", "value": cuts, "name": "cut"}
        if isinstance(cuts, list)
        else {"type": "constant", "value": cuts}
    )
    return Thesis.model_validate(
        {
            "name": "b",
            "target_mode": target_mode,
            "data": {"targets": list(targets)},
            "entry": {
                "type": "threshold",
                "left": {
                    "type": "percentile",
                    "window": 14,
                    "input": {"type": "field", "column": "close"},
                },
                "op": "<",
                "right": right,
            },
            "params": {"horizon": horizon},
        }
    )


def test_basket_cells_carry_a_reconciled_pooled_block(tmp_path):
    files = _basket_files(tmp_path)
    thesis = _pctl_thesis(list(files), target_mode="basket", horizon=[5, 10])
    res = run_backtest(thesis, load(thesis, files))
    s = res.summary
    assert s["target_mode"] == "basket"
    assert len(s["cells"]) == 2
    for cell in s["cells"]:
        pooled = cell["pooled"]
        assert set(pooled) == _POOLED_KEYS
        # one pool, three descriptions: the pooled count, the member panels' sum, and the
        # cross-target episode merge all describe the SAME closed rows
        assert pooled["n"] == sum(st["n"] for st in cell["by_target"].values())
        assert pooled["n"] == cell["episode_stats"]["n"]
        assert 0 < pooled["n_nonoverlap"] <= pooled["n"]
        # hand-check: pooled.mean_ret IS the concatenated closed-rows mean off the trades frame
        rows = res.trades
        for axis, value in cell["params"].items():
            if axis in rows.columns:
                rows = rows[rows[axis] == value]
        closed = rows[~rows["is_open"]]
        pool = closed["ret"].to_numpy(dtype=float)
        assert pool.size == pooled["n"] > 0
        assert pooled["mean_ret"] == pytest.approx(float(pool.mean()))
        assert pooled["hit_rate"] == pytest.approx(float((pool > 0).mean()))
        # member_share is a FULL decomposition (every declared member, never ranked): shares sum
        # to 1, the max is the max, and each share is the member's |return| mass off the frame
        shares = pooled["member_share"]["by_target"]
        assert set(shares) == set(s["targets"])
        assert sum(shares.values()) == pytest.approx(1.0)
        assert pooled["member_share"]["max_member_share_abs"] == pytest.approx(max(shares.values()))
        total_mass = float(np.abs(pool).sum())
        for tgt in s["targets"]:
            mass_t = float(
                np.abs(closed[closed["target"] == tgt]["ret"].to_numpy(dtype=float)).sum()
            )
            assert shares[tgt] == pytest.approx(mass_t / total_mass)
        # by_target keys UNCHANGED from conjunction — pooling adds a panel, it rewrites nothing
        for st in cell["by_target"].values():
            assert set(st) == _BY_TARGET_KEYS


def test_conjunction_cells_never_carry_a_pooled_key(tmp_path):
    # Absence, not null: `pooled` is the basket rubric's panel, and a conjunction run forms no
    # pooled read at all — a null block would invite reading a pooled headline into a mode whose
    # doctrine forbids one. Walked over EVERY declared cell, dead combos included.
    files = _basket_files(tmp_path)
    thesis = _pctl_thesis(
        list(files), target_mode="conjunction", horizon=[5, 10], cuts=[0.35, -1.0]
    )
    s = run_backtest(thesis, load(thesis, files)).summary
    assert s["target_mode"] == "conjunction"
    assert len(s["cells"]) == 4
    for cell in s["cells"]:
        assert "pooled" not in cell


def test_basket_pooled_block_is_horizon_composition_independent(tmp_path):
    # Adding a horizon to the sweep changes nothing another cell already measured — the pooled
    # panel obeys the same grid-composition doctrine as every per-cell read (the content-keyed
    # bootstrap seed included).
    from seikan.serialize import json_safe

    files = _basket_files(tmp_path)
    one = _pctl_thesis(list(files), target_mode="basket", horizon=[5])
    two = _pctl_thesis(list(files), target_mode="basket", horizon=[5, 21])
    s1 = run_backtest(one, load(one, files)).summary
    s2 = run_backtest(two, load(two, files)).summary
    c1 = next(c for c in s1["cells"] if c["params"]["horizon"] == 5)
    c2 = next(c for c in s2["cells"] if c["params"]["horizon"] == 5)
    assert json_safe(c1["pooled"]) == json_safe(c2["pooled"])


def test_basket_run_is_deterministic(tmp_path):
    # Two identical basket runs emit deep-equal summaries: the pooled additions (member_share,
    # the pooled rotation null, the cross-member-merged bootstrap) introduce no positional or
    # clock-keyed state — same-bar cross-member tie order is fixed as target-declaration order.
    import json

    from seikan.serialize import json_safe

    files = _basket_files(tmp_path)
    thesis = _pctl_thesis(list(files), target_mode="basket", horizon=[5, 10])
    a = run_backtest(thesis, load(thesis, files)).summary
    b = run_backtest(thesis, load(thesis, files)).summary
    assert json.dumps(json_safe(a), sort_keys=True) == json.dumps(json_safe(b), sort_keys=True)


def test_mode_flip_changes_only_the_additive_blocks(tmp_path):
    # THE conjunction-byte-identity pin, read from the basket side: flipping the declared mode on
    # the SAME thesis (no cross nodes, no benchmark) over the SAME data changes NOTHING the
    # conjunction engine measured — basket is ADDITIVE. Every per-target number, every ledger,
    # every rollup and every run stamp is dict-equal across the modes; the only differences are
    # the stamp itself, the per-cell `pooled` block, and the baseline's `pooled` row.
    import json

    from seikan.serialize import json_safe

    def _eq(x, y):
        return json.dumps(json_safe(x), sort_keys=True) == json.dumps(json_safe(y), sort_keys=True)

    files = _basket_files(tmp_path)
    tc = _pctl_thesis(list(files), target_mode="conjunction", horizon=[5, 10])
    tb = _pctl_thesis(list(files), target_mode="basket", horizon=[5, 10])
    sc = run_backtest(tc, load(tc, files)).summary
    sb = run_backtest(tb, load(tb, files)).summary
    assert sc["target_mode"] == "conjunction" and sb["target_mode"] == "basket"
    assert set(sc) == set(sb)  # no summary key appears or vanishes with the mode
    # every shared run-level key (params, targets, rotation, sources, geometry and
    # self-description stamps, …) is byte-equal across the modes
    for key in sorted(set(sc) - {"target_mode", "cells", "baseline", "pbo"}):
        assert _eq(sc[key], sb[key]), key
    # pbo: both members fire AND close in every combo (asserted below), so basket's ≥1-member
    # admissibility relaxation changes nothing and the STRUCTURAL fields agree exactly. The
    # SCORED fields legitimately move with the mode and are deliberately not pinned equal: the
    # split score mirrors the statistic the declared mode's caller selects on (weakest-member
    # Sharpe under conjunction, pooled Sharpe under basket — `cscv_pbo(..., mode=...)`), so
    # asserting equal scores would assert the two exams are one exam.
    for key in (
        "reason",
        "n_splits",
        "n_combos",
        "n_combos_scoreable",
        "n_combos_declared",
        "blocks",
    ):
        assert sc["pbo"][key] == sb["pbo"][key]
    assert sc["pbo"]["reason"] is None
    assert len(sc["cells"]) == len(sb["cells"]) == 2
    for cc, cb in zip(sc["cells"], sb["cells"], strict=True):
        for st in cc["by_target"].values():
            assert st["n"] > 0  # the pbo premise: no member absent from any combo
        assert set(cb) == set(cc) | {"pooled"}  # the pooled block is the ONLY per-cell addition
        for key in cc:  # by_target / episode_stats / outcome_coverage / signal_coverage / …
            assert _eq(cc[key], cb[key]), key
    for ec, eb in zip(sc["baseline"], sb["baseline"], strict=True):
        assert set(eb) == set(ec) | {"pooled"}  # the pooled row is the ONLY baseline addition
        assert ec["horizon"] == eb["horizon"]
        assert _eq(ec["by_target"], eb["by_target"])


def test_pooled_rotation_survives_an_all_censored_member(tmp_path):
    # The separate `pooled_cells` roster's pin: every FIRED member's mask reaches the pooled
    # rotation null, closed rows or not. A member whose only firings sit within h bars of the
    # data end contributes ZERO closed rows — `rel_cells` (closed-only by construction) never
    # holds it, which is exactly why the pooled read cannot be driven off `rel_cells`: the
    # censored member still shaped the basket's cross-sectional firing pattern, and the common
    # shift must preserve what it rotates. The run completes, the member is on the record with
    # an explicit zero, and the pooled statistics stay finite off the surviving member's rows.
    n, h = 80, 5
    idx = pd.date_range("2021-01-01", periods=n, freq="1D")
    rng = np.random.RandomState(17)
    a = 100.0 + rng.uniform(-2.0, 2.0, n)
    a[[10, 18, 26, 34, 42]] += 10.0  # sparse interior events, all closeable
    b = 100.0 + rng.uniform(-2.0, 2.0, n)
    b[[76, 77]] += 10.0  # B's ONLY threshold crossings: both windows run past the data end
    files = {
        "A": _write_bars(tmp_path / "A.csv", a, idx),
        "B": _write_bars(tmp_path / "B.csv", b, idx),
    }
    thesis = Thesis.model_validate(
        {
            "name": "censored_member",
            "target_mode": "basket",
            "data": {"targets": ["A", "B"]},
            "entry": _gt(105.0),
            "params": {"horizon": h},
        }
    )
    res = run_backtest(thesis, load(thesis, files))
    cell = res.summary["cells"][0]
    # B fired twice, closed nothing — a full by_target entry with an explicit zero, never absent
    assert cell["outcome_coverage"]["B"] == {
        "n_attempted": 2,
        "n_closed": 0,
        "exit_reasons": {"horizon": 0, "open": 2, "no_outcome": 0, "no_benchmark": 0},
    }
    st_b = cell["by_target"]["B"]
    assert set(st_b) == _BY_TARGET_KEYS
    assert st_b["n"] == 0 and st_b["n_nonoverlap"] == 0 and math.isnan(st_b["mean_ret"])
    # the pooled block rides the surviving member's closed rows and stays FINITE — the censored
    # member's presence in the rotation roster must not NaN the cell's pooled read
    pooled = cell["pooled"]
    assert pooled["n"] == cell["by_target"]["A"]["n"] == 5
    assert pooled["n_nonoverlap"] == 5  # A's events sit > h bars apart
    assert math.isfinite(pooled["rot_p"]) and 0.0 <= pooled["rot_p"] <= 1.0
    assert math.isfinite(pooled["mean_ret"])
    assert pooled["member_share"]["by_target"]["A"] == pytest.approx(1.0)


# ---- run-level unconditional baseline (evidence-only) ------------------------
# The base rate every conditional mean must be read against: the SAME measurement with the entry
# condition removed, per horizon × target — plus, in basket mode, one pooled (bar × member) row.
# Unconditional by construction (it exists whether or not anything fired), arithmetic a reader
# re-checks (`n_eligible + Σexclusions == n_anchor_bars`), and carrying NO uplift field ever —
# the cell-vs-baseline comparison is the caller's.

_BASELINE_ROW_KEYS = {
    "n_anchor_bars",
    "n_eligible",
    "exclusions",
    "mean_ret",
    "std_ret",
    "hit_rate",
    "ret_quantiles",
    "worst_ret",
    "best_ret",
}


def _never_fires_thesis(targets, horizon, *, target_mode="conjunction"):
    """An entry that can never fire: the baseline panel is UNCONDITIONAL, so it must be complete
    even on a run that produced zero observations."""
    return Thesis.model_validate(
        {
            "name": "bl",
            "target_mode": target_mode,
            "data": {"targets": list(targets)},
            "entry": _lt(0.0),
            "params": {"horizon": horizon},
        }
    )


def test_baseline_arithmetic_reconciles_per_horizon_and_target(tmp_path):
    # On a clean complete file the ONLY exclusion is terminal geometry: anchors are the fills
    # t+1 for firing bars t = 0..n_bars−2, and an anchor is "open" where t+1+h > n_bars−1 —
    # exactly h of them per target. Everything else is eligible, and the row's arithmetic
    # (`n_eligible + Σexclusions == n_anchor_bars`) closes without remainder.
    files = {}
    for i, sym in enumerate(("AAA", "BBB")):
        _noise_px(tmp_path / f"{sym}.csv", n=400, seed=30 + i)
        files[sym] = str(tmp_path / f"{sym}.csv")
    thesis = _never_fires_thesis(list(files), [3, 7])
    s = run_backtest(thesis, load(thesis, files)).summary
    assert _closed_n(s) == 0  # nothing fired — the baseline is unconditional evidence anyway
    assert [e["horizon"] for e in s["baseline"]] == [3, 7]
    for entry in s["baseline"]:
        h = entry["horizon"]
        assert "pooled" not in entry  # the pooled row is basket-only
        assert set(entry["by_target"]) == {"AAA", "BBB"}
        for row in entry["by_target"].values():
            assert set(row) == _BASELINE_ROW_KEYS
            assert row["n_anchor_bars"] == s["n_bars"] - 1
            assert row["n_eligible"] + sum(row["exclusions"].values()) == row["n_anchor_bars"]
            # the clean-file exclusion breakdown is hand-derivable: h open anchors, no holes
            assert row["exclusions"] == {"open": h, "no_outcome": 0, "no_benchmark": 0}
            assert row["n_eligible"] == s["n_bars"] - 1 - h
            assert math.isfinite(row["mean_ret"]) and 0.0 <= row["hit_rate"] <= 1.0
            q = row["ret_quantiles"]
            assert q["p10"] <= q["p25"] <= q["p50"] <= q["p75"] <= q["p90"]
            assert row["worst_ret"] <= q["p10"] and q["p90"] <= row["best_ret"]


def test_baseline_flat_benchmark_parity_zeroes_the_base_rate(tmp_path):
    # A benchmark equal to the target zeroes every excess return, so the unconditional base rate
    # is EXACTLY zero — the baseline rides the same measurement (same benchmark leg) as the cells.
    closes = 100 + np.random.RandomState(6).randn(120).cumsum()
    res = _bench_run(tmp_path, closes, closes, benchmark="market")
    row = res.summary["baseline"][0]["by_target"]["target"]
    assert row["n_eligible"] > 0
    assert row["mean_ret"] == pytest.approx(0.0, abs=1e-12)
    assert row["worst_ret"] == pytest.approx(0.0, abs=1e-12)
    assert row["best_ret"] == pytest.approx(0.0, abs=1e-12)


def test_baseline_shortonly_flips_the_sign(tmp_path):
    # Direction rides the baseline exactly as it rides the cells: the shortonly base rate is the
    # longonly one with every observation's sign flipped — mirrored tails included.
    closes = 100 + np.random.RandomState(7).randn(150).cumsum()
    lo = _bench_run(tmp_path, closes).summary["baseline"][0]["by_target"]["target"]
    sh = _bench_run(tmp_path, closes, direction="shortonly")
    sh = sh.summary["baseline"][0]["by_target"]["target"]
    assert sh["n_eligible"] == lo["n_eligible"]
    assert sh["exclusions"] == lo["exclusions"]
    assert sh["mean_ret"] == pytest.approx(-lo["mean_ret"])
    assert sh["worst_ret"] == pytest.approx(-lo["best_ret"])
    assert sh["best_ret"] == pytest.approx(-lo["worst_ret"])
    assert sh["ret_quantiles"]["p10"] == pytest.approx(-lo["ret_quantiles"]["p90"])
    assert sh["ret_quantiles"]["p90"] == pytest.approx(-lo["ret_quantiles"]["p10"])


def test_baseline_is_grid_composition_independent(tmp_path):
    # The baseline describes the DATA under the declared horizons — adding a swept entry value
    # changes no baseline number, the same doctrine as every per-cell read. (A new HORIZON adds
    # a new baseline entry, because a horizon is a new measurement window, not a new search arm.)
    import json

    from seikan.serialize import json_safe

    _noise_px(tmp_path / "px.csv")

    def _sweep(cuts):
        return Thesis.model_validate(
            {
                "name": "t",
                "data": {"targets": ["target"]},
                "entry": {
                    "type": "threshold",
                    "left": {
                        "type": "percentile",
                        "window": 14,
                        "input": {"type": "field", "column": "close"},
                    },
                    "op": "<",
                    "right": {"type": "constant", "value": cuts, "name": "cut"},
                },
                "params": {"horizon": [5, 10]},
            }
        )

    t2, t3 = _sweep([0.30, 0.35]), _sweep([0.30, 0.35, 0.40])
    s2 = run_backtest(t2, load(t2, {"target": tmp_path / "px.csv"})).summary
    s3 = run_backtest(t3, load(t3, {"target": tmp_path / "px.csv"})).summary
    assert json.dumps(json_safe(s2["baseline"]), sort_keys=True) == json.dumps(
        json_safe(s3["baseline"]), sort_keys=True
    )


def test_baseline_deletion_monotonicity_under_an_interior_hole(tmp_path):
    # Punching an interior NaN hole into one target's measured leg can only ever GROW that
    # target's exclusions and SHRINK its eligible pool — never mint eligibility — and the other
    # target's row must not move at all. One open[m] hole kills exactly the two anchors that
    # read it (the fill at m and the exit landing on m), both classified `no_outcome` because
    # the hole is in-data, not terminal geometry.
    from seikan.serialize import json_safe

    for sym, seed in (("AAA", 40), ("BBB", 41)):
        _noise_px(tmp_path / f"{sym}.csv", n=400, seed=seed)
    clean_files = {"AAA": str(tmp_path / "AAA.csv"), "BBB": str(tmp_path / "BBB.csv")}
    thesis = _never_fires_thesis(list(clean_files), [3, 7])
    s_clean = run_backtest(thesis, load(thesis, clean_files)).summary

    holed_px = pd.read_csv(tmp_path / "BBB.csv", index_col="datetime")
    holed_px.iloc[200, holed_px.columns.get_loc("open")] = np.nan
    holed_px.to_csv(tmp_path / "BBB_holed.csv")
    # the SAME thesis, re-measured over a different BBB file: what moved is the data, not the exam
    holed_files = {"AAA": str(tmp_path / "AAA.csv"), "BBB": str(tmp_path / "BBB_holed.csv")}
    s_holed = run_backtest(thesis, load(thesis, holed_files)).summary

    for e_holed, e_clean in zip(s_holed["baseline"], s_clean["baseline"], strict=True):
        hb, cb = e_holed["by_target"]["BBB"], e_clean["by_target"]["BBB"]
        assert hb["exclusions"]["no_outcome"] == 2 > cb["exclusions"]["no_outcome"]
        assert hb["exclusions"]["open"] == cb["exclusions"]["open"]  # geometry is untouched
        assert hb["n_eligible"] == cb["n_eligible"] - 2
        # the conservation pin holds THROUGH the deletion — nothing leaks out of the ledger
        assert hb["n_eligible"] + sum(hb["exclusions"].values()) == hb["n_anchor_bars"]
        # the hole is localized: the clean sibling's row is untouched to the last digit
        assert json_safe(e_holed["by_target"]["AAA"]) == json_safe(e_clean["by_target"]["AAA"])


def test_basket_baseline_pooled_row_is_the_per_target_sum(tmp_path):
    # The basket's pooled base rate concatenates the (bar × member) eligible observations, so
    # every COUNT is the per-target sum (hand-averaging per-target baselines would mis-weight
    # under holes) and the conservation pin closes on the pooled row too.
    files = _basket_files(tmp_path)
    thesis = _never_fires_thesis(list(files), [4], target_mode="basket")
    s = run_backtest(thesis, load(thesis, files)).summary
    entry = s["baseline"][0]
    pooled = entry["pooled"]
    rows = list(entry["by_target"].values())
    assert pooled["n_anchor_bars"] == sum(r["n_anchor_bars"] for r in rows)
    assert pooled["n_eligible"] == sum(r["n_eligible"] for r in rows)
    for k in ("open", "no_outcome", "no_benchmark"):
        assert pooled["exclusions"][k] == sum(r["exclusions"][k] for r in rows)
    assert pooled["n_eligible"] + sum(pooled["exclusions"].values()) == pooled["n_anchor_bars"]
    # the pooled mean is the eligible-count-weighted member mean — concatenation, not averaging
    want = sum(r["mean_ret"] * r["n_eligible"] for r in rows) / sum(r["n_eligible"] for r in rows)
    assert pooled["mean_ret"] == pytest.approx(want)


# ---- per-cell episode ledger + feature association ----------------------------


def test_episode_ledger_reconciles_orders_and_conserves_mass_past_the_cap(tmp_path):
    # The time-ordered narrative companion to episode_stats: `n_total` IS `n_clusters` (same
    # frozen merge, by construction), entries are earliest-first (never ranked — ranking would
    # crown episodes the way the engine refuses to crown cells), and truncation past the cap is
    # explicit and mass-conserving, so a count read off a truncated ledger is a floor.
    from seikan.analysis.stats import EPISODE_LIST_MAX

    cycle = [100.0, 98.0, 96.0, 94.0, 92.0, 110.0]
    _bars(tmp_path / "cap.csv", cycle * 60)  # enough distinct episodes to overflow the cap
    thesis = Thesis.model_validate(
        {
            "name": "cap",
            "data": {"targets": ["target"]},
            "entry": _drawdown_from_high(0.95, 5),
            "params": {"horizon": 1},
        }
    )
    md = load(thesis, {"target": tmp_path / "cap.csv"})
    cell = run_backtest(thesis, md).summary["cells"][0]
    ep = cell["episodes"]
    assert ep["cap"] == EPISODE_LIST_MAX
    assert ep["n_total"] == cell["episode_stats"]["n_clusters"] > EPISODE_LIST_MAX
    assert len(ep["entries"]) == EPISODE_LIST_MAX
    assert ep["n_omitted"] == ep["n_total"] - EPISODE_LIST_MAX
    starts = [e["start"] for e in ep["entries"]]
    assert starts == sorted(starts)  # earliest first — ISO stamps sort lexicographically
    for e in ep["entries"]:
        assert set(e) == {"start", "end", "n", "mean_ret", "share_abs"}
        assert e["start"] <= e["end"] and e["n"] >= 1
    # mass conservation: listed shares + the omitted remainder ≈ 1, so no episode vanishes
    assert sum(e["share_abs"] for e in ep["entries"]) + ep["omitted_share_abs"] == pytest.approx(
        1.0
    )


def test_dead_cell_evidence_blocks_are_explicit_refusals(tmp_path):
    # A cell that never fired keeps EVERY evidence block, each stating its emptiness
    # explicitly: the episode ledger is the empty ledger (never null), every bucket panel names
    # `no_closed_observations`, and every association names `insufficient_observations` at n=0.
    from seikan.analysis.stats import EPISODE_LIST_MAX

    _noise_px(tmp_path / "px.csv", n=500)
    thesis = _dead_combo_thesis()
    s = run_backtest(thesis, load(thesis, {"target": tmp_path / "px.csv"})).summary
    dead = [c for c in s["cells"] if c["by_target"]["target"]["n"] == 0]
    assert len(dead) == 2  # the cut=-1.0 horizon siblings
    for cell in dead:
        assert cell["episodes"] == {
            "entries": [],
            "n_total": 0,
            "n_omitted": 0,
            "omitted_share_abs": 0.0,
            "cap": EPISODE_LIST_MAX,
        }
        for blk in cell["conditional_buckets"].values():
            assert blk == {"buckets": [], "reason": "no_closed_observations"}
        assert cell["bucket_monotonicity"] == {}
        for by_tgt in cell["feature_association"].values():
            for rec in by_tgt.values():
                assert rec == {"rho": None, "n": 0, "reason": "insufficient_observations"}


def test_feature_association_covers_every_feature_and_target(tmp_path):
    # Spearman(entry-time snapshot, realized closed return) per (cell × feature × TARGET) — per
    # target in BOTH modes, because pooling ranks across members would conflate level differences
    # with time variation. Every declared pair gets an entry: a measured rho, or an explicit
    # refusal from the fixed vocabulary — never an absence.
    files = {}
    for i, sym in enumerate(("AAA", "BBB")):
        _noise_px(tmp_path / f"{sym}.csv", n=500, seed=50 + i)
        files[sym] = str(tmp_path / f"{sym}.csv")
    thesis = _pctl_thesis(list(files), target_mode="conjunction", horizon=[5])
    s = run_backtest(thesis, load(thesis, files)).summary
    measured = 0
    for cell in s["cells"]:
        fa = cell["feature_association"]
        assert set(fa) == {"ret_5", "ret_20", "vol_14"}  # the default snapshots
        for by_tgt in fa.values():
            assert set(by_tgt) == {"AAA", "BBB"}
            for rec in by_tgt.values():
                assert set(rec) == {"rho", "n", "reason"}
                assert rec["reason"] in (None, "insufficient_observations", "no_rank_variation")
                if rec["reason"] is None:
                    measured += 1
                    assert -1.0 <= rec["rho"] <= 1.0 and rec["n"] >= 10
                else:
                    assert rec["rho"] is None
    assert measured > 0  # the dense fixture measures real associations — nothing is vacuous


def test_pbo_scores_a_level_heavy_diff_outcome_end_to_end(tmp_path):
    # A diff outcome on a series-shaped target at level 1e9: the CSCV split scores used to
    # cancel their one-pass variance to zero there and refuse as "insufficient_data"; the
    # shifted-moment scoring gives the grid a real pbo block through the public runner.
    rng = np.random.RandomState(31)
    vals = list(1e9 + rng.normal(0.0, 1.0, 400).cumsum() * 0.01)
    _series_bars(tmp_path / "y.csv", vals)
    thesis = Thesis.model_validate(
        {
            "name": "t",
            "data": {"targets": ["target"]},
            "entry": {
                "type": "threshold",
                "left": {
                    "type": "percentile",
                    "window": 14,
                    "input": {"type": "field", "column": "close"},
                },
                "op": "<",
                "right": {"type": "constant", "value": [0.3, 0.5], "name": "cut"},
            },
            "params": {"horizon": [3, 5], "outcome": {"kind": "diff"}},
        }
    )
    res = run_backtest(thesis, load(thesis, {"target": tmp_path / "y.csv"}))
    pbo = res.summary["pbo"]
    assert pbo["reason"] is None
    assert 0.0 <= pbo["pbo"] <= 1.0 and np.isfinite(pbo["lambda_mean"])
