"""Statistics tests with known inputs.

Every statistic in ``analysis.stats`` is NOMINAL and per-cell: nothing here selects, ranks or
search-corrects a cell, so these tests pin measurement arithmetic and independence — never a
winner, a family-wise correction, or an inferential verdict.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from seikan.analysis._conditioning import _bucket_monotonicity
from seikan.analysis._cscv import _degradation_slope
from seikan.analysis._hac import _bar_deviation_sums
from seikan.analysis._pools import _near_constant
from seikan.analysis._rotation import _fwd_ffts, _rotation_num_den, _tail_p
from seikan.analysis.stats import (
    ASSOC_MIN_N,
    BUCKET_MIN_N,
    EPISODE_LIST_MAX,
    STATISTICS_VERSION,
    BarNumerators,
    baseline_summary,
    benchmark_regression,
    cell_conditional_buckets,
    concentration,
    conditional_buckets,
    cross_mean_bar_numerators,
    cross_mean_pooled_hac,
    cross_mean_pooled_mean,
    cscv_pbo,
    edge_ratio,
    episode_bootstrap_ci,
    episode_ledger,
    episode_profile,
    episode_stats,
    exact_mean,
    feature_outcome_association,
    newey_west_mean,
    nonoverlap_count,
    overlap_clusters,
    pool_moments,
    pool_quantiles,
    pooled_reliability_summary,
    profit_factor,
    reliability_summary,
    subperiod_edges,
    subperiod_means,
    timing_summary,
)


def _episodes(rets, reasons=None, feature=None) -> pd.DataFrame:
    n = len(rets)
    data = {
        "ret": rets,
        "exit_reason": reasons if reasons is not None else ["horizon"] * n,
    }
    if feature is not None:
        data["feat"] = feature
    return pd.DataFrame(data)


def test_statistics_version_is_stamped():
    # v5: corrections — warmup has no truth value and negation waits for its whole operand, exact
    # zero dispersion, native-clock print holes, rotation-null ties within the FFT's rounding, the
    # exact cross_mean pooled mean, and the breadth k a cross kernel actually reduced over.
    assert STATISTICS_VERSION == 5


# ---- pool_moments: one pool's moment and shape reads ------------------------


def test_pool_moments_basic():
    rets = [0.01, -0.02, 0.03, 0.04, -0.01]
    s = pool_moments(np.asarray(rets))
    assert s["n"] == 5
    assert s["mean_ret"] == pytest.approx(np.mean(rets))
    assert s["std_ret"] == pytest.approx(np.std(rets, ddof=1))
    assert s["hit_rate"] == pytest.approx(3 / 5)
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    assert s["win_loss_ratio"] == pytest.approx(np.mean(wins) / abs(np.mean(losses)))
    assert s["skewness"] == pytest.approx(sps.skew(np.asarray(rets, dtype=float)))
    # no iid t-test, no equity-curve framing and no iid bootstrap CI: the observer measures a
    # pool of OVERLAPPING observations, on which the classical pair is invalid by construction
    assert not (
        {"t_iid", "p_iid", "total_ret", "ann_factor", "mean_ret_annualized", "boot_ci_lo"}
        & s.keys()
    )


def test_pool_moments_empty():
    s = pool_moments(np.empty(0))
    assert s["n"] == 0
    assert all(np.isnan(v) for k, v in s.items() if k != "n")


def test_pool_moments_drops_non_finite_returns():
    s = pool_moments(np.array([0.01, float("nan"), 0.03, float("inf")]))
    assert s["n"] == 2
    assert s["mean_ret"] == pytest.approx(0.02)


def test_pool_moments_zero_returns_are_non_hits_and_join_no_ratio_side():
    # The tie rule, stated once for all three sibling reads: a zero return is not a hit, and it
    # joins neither side of win_loss_ratio (nor of profit_factor) — so hit_rate's complement is
    # NOT the loss rate the ratios' denominators describe.
    rets = np.array([0.01, -0.01, 0.0, 0.0, 0.02])
    s = pool_moments(rets)
    assert s["hit_rate"] == pytest.approx(2 / 5)
    assert s["win_loss_ratio"] == pytest.approx(0.015 / 0.01)
    assert profit_factor(rets) == pytest.approx(0.03 / 0.01)


def test_pool_moments_shape_metrics_known_answers():
    # kurtosis (Pearson, population moments) of [1,2,3,4]: m2=1.25, m4=2.5625 → 1.64
    s = pool_moments(np.array([1.0, 2.0, 3.0, 4.0]))
    assert s["kurtosis"] == pytest.approx(1.64)
    rets = np.linspace(-0.05, 0.05, 100)
    s = pool_moments(rets)
    p5, p95 = np.percentile(rets, [5, 95])
    assert s["tail_ratio"] == pytest.approx(abs(p95 / p5))
    assert s["cvar_5"] == pytest.approx(np.mean(np.sort(rets)[:5]))
    # the moments are scipy's POPULATION estimators (bias=True), which pandas' unbiased G1/G2
    # would not reproduce — the convention the caveat states
    pool = np.random.RandomState(9).normal(0.0, 0.01, 50)
    s = pool_moments(pool)
    assert s["skewness"] == pytest.approx(sps.skew(pool, bias=True))
    assert s["kurtosis"] == pytest.approx(sps.kurtosis(pool, fisher=False, bias=True))


def test_pool_moments_cvar_5_is_the_mean_of_the_worst_twentieth():
    # The realized 5% expected shortfall is a COUNT, ⌈n/20⌉ — `concentration`'s — never the
    # observations at or below p05: a tie AT p05 joined that tail whole, and 99 zeros beside one
    # −0.5 read −0.005 for a worst 5% of −0.1.
    assert pool_moments(np.array([-0.5] + [0.0] * 99))["cvar_5"] == -0.1
    # one observation through n = 20, two from n = 21
    assert pool_moments(np.arange(20.0))["cvar_5"] == 0.0
    assert pool_moments(np.arange(21.0))["cvar_5"] == 0.5
    # p05 is never read: across a gap past the double range it interpolated to +inf, and
    # `v <= inf` took the WHOLE pool's mean, +2.3e307, for a worst observation of −1e308
    assert pool_moments(np.array([-1e308, 8e307, 9e307]))["cvar_5"] == -1e308
    # ...and a worst twentieth whose sum leaves the range has no mean: NaN → null
    assert np.isnan(pool_moments(np.array([-1.7e308, -1e308] + [0.0] * 19))["cvar_5"])


def test_pool_moments_tail_reads_describe_a_one_observation_pool():
    # Both tail reads are defined at n = 1 — p05 = p95 = the observation, as pool_quantiles
    # reports them, and the worst twentieth is the observation itself — but sat behind the std's
    # n > 1 guard: a one-row pool read null beside its own p05 and p95.
    s = pool_moments(np.array([-0.1]))
    assert s["tail_ratio"] == 1.0 and s["cvar_5"] == -0.1
    assert pool_quantiles(np.array([-0.1]))["p05"] == -0.1
    assert np.isnan(pool_moments(np.array([0.0]))["tail_ratio"])  # p05 = 0: no ratio


def test_profit_factor_arithmetic_and_refusals():
    # Σwins/|Σlosses|: zero-return rows join NEITHER side, and either side empty is NaN — the
    # win_loss_ratio no-ratio-to-form rule, and this engine never emits an infinity.
    rets = np.array([0.04, -0.01, 0.02, -0.02, 0.0])
    assert profit_factor(rets) == pytest.approx(0.06 / 0.03)
    # joined to win_loss_ratio by (n_wins / n_losses) — derivable, so the caveat can say so
    s = pool_moments(rets)
    assert profit_factor(rets) == pytest.approx(s["win_loss_ratio"] * (2 / 2))
    assert np.isnan(profit_factor(np.array([0.01, 0.02])))  # no losses → NaN, never +inf
    assert np.isnan(profit_factor(np.array([-0.01, -0.02])))  # no wins → NaN, never 0
    assert np.isnan(profit_factor(np.array([0.0, 0.0])))
    assert np.isnan(profit_factor(np.array([])))
    assert profit_factor(np.array([0.01, np.nan, -0.02])) == pytest.approx(0.5)


def test_edge_ratio_arithmetic_and_refusals():
    # mean RAW MFE over |mean RAW MAE| over PAIRED rows — mae is sign-negative, hence the abs;
    # no surviving pair or a zero paired adverse mean refuses as NaN (an unbounded ratio is
    # never emitted). Inputs are row-aligned; the ratio filters the pairs itself.
    mae = np.array([-0.02, -0.04, -0.06])
    mfe = np.array([0.03, 0.06, 0.03])
    assert edge_ratio(mae, mfe) == pytest.approx(0.04 / 0.04)
    assert np.isnan(edge_ratio(np.array([]), np.array([])))
    assert np.isnan(edge_ratio(np.full(3, np.nan), mfe))
    assert np.isnan(edge_ratio(mae, np.full(3, np.nan)))
    assert np.isnan(edge_ratio(np.zeros(3), mfe))  # never went against: no ratio
    # THE PAIRING PIN: a hole in EITHER leg drops the row from BOTH legs — an mae finite on a
    # censored-mfe row contributes nothing to the ratio (pre-v3 each leg filtered its own
    # subset, so the two means could cover trades that never shared a path).
    mae_holed = np.array([-0.02, np.nan, -0.04])
    mfe_holed = np.array([0.03, 0.06, np.nan])
    assert edge_ratio(mae_holed, mfe_holed) == pytest.approx(0.03 / 0.02)


def test_ratio_reads_refuse_an_overflowed_operand():
    # Finite returns whose SUM overflows: `x / inf` was a FINITE 0.0 — a win/loss ratio, a
    # profit factor and an edge ratio of exactly zero, a reader takes at face value. An operand
    # past the double range is no evidence: NaN → null.
    # (No errstate around the engine: its overflows are expected inputs, never warnings.)
    rets = np.array([1.0, -1e308, -1e308])  # the loss mass (and so its mean) overflows to -inf
    assert np.isnan(pool_moments(rets)["win_loss_ratio"])
    assert np.isnan(profit_factor(rets))
    assert np.isnan(edge_ratio(np.array([-1e308, -1e308]), np.array([1.0, 1.0])))
    # |p95 / p05| where p05 interpolates across ±1.7e308 and overflows to +inf
    tail = np.array([-1.7e308] + [1.7e308] * 9)
    with np.errstate(over="ignore"):
        assert np.isinf(np.percentile(tail, 5.0))  # the premise
    assert np.isnan(pool_moments(tail)["tail_ratio"])


def test_empty_pool_moments_are_complete():
    # The all-NaN record and a populated pool answer the SAME key set: an empty pool is a full
    # row of nulls, never a shorter document — the null-not-missing doctrine at the pool level.
    assert set(pool_moments(np.empty(0))) == set(pool_moments(np.array([0.01, -0.02, 0.03])))


# ---- pool_quantiles: the seven-point shape read ------------------------------


def test_pool_quantiles_hand_computed_linear_interpolation():
    # The interpolation rule is part of the contract, not an implementation detail: the shoulders
    # are the same rule `pool_moments`' p5/p95 and the episode bootstrap's CI use, or two blocks
    # of one report would disagree about the same pool.
    # np.percentile's default linear rule on [1..10] (index = p/100 · 9):
    #   p05 → 0.45 → 1 + 0.45·(2−1) = 1.45     p10 → 0.9  → 1 + 0.9·(2−1)  = 1.9
    #   p25 → 2.25 → 3 + 0.25·(4−3) = 3.25     p50 → 4.5  → 5 + 0.5·(6−5)  = 5.5
    #   p75 → 6.75 → 7 + 0.75·(8−7) = 7.75     p90 → 8.1  → 9 + 0.1·(10−9) = 9.1
    #   p95 → 8.55 → 9 + 0.55·(10−9) = 9.55 (approx — 0.55·1 is inexact in binary)
    q = pool_quantiles(np.arange(1, 11, dtype=float))
    assert q == {
        "p05": 1.45,
        "p10": 1.9,
        "p25": 3.25,
        "p50": 5.5,
        "p75": 7.75,
        "p90": 9.1,
        "p95": pytest.approx(9.55),
    }


def test_pool_quantiles_single_observation_is_that_observation_seven_times():
    # An n=1 pool has no shape to read: every point IS the one observation. The block reports it
    # rather than refusing, because the count that qualifies it rides alongside (`mae_quantiles.n`)
    # or IS the pool (`ret_quantiles` has no n of its own — it is the cell's n).
    q = pool_quantiles(np.array([0.042]))
    assert q == {
        "p05": 0.042,
        "p10": 0.042,
        "p25": 0.042,
        "p50": 0.042,
        "p75": 0.042,
        "p90": 0.042,
        "p95": 0.042,
    }


def test_pool_quantiles_empty_pool_is_seven_nans_never_zero():
    # "No evidence" must serialize to null, never to 0.0 — a zero quantile reads as a measured
    # outcome and would let a cell that fired nothing look like a cell that fired flat.
    q = pool_quantiles(np.array([]))
    assert set(q) == {"p05", "p10", "p25", "p50", "p75", "p90", "p95"}
    assert all(np.isnan(v) for v in q.values())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_pool_quantiles_drops_non_finite_values(bad):
    # THE property the excursion blocks depend on: an `mae`/`mfe` censored by a window hole is NaN
    # on a row whose `ret` closed, so a poisoned pool must read exactly like the pool without it —
    # not NaN throughout (one hole erasing the block), and not zero-filled (a hole minting a mark).
    # ±inf gets the same treatment: it is unmeasurable, not an extreme observation.
    clean = np.array([1.0, 2.0, 3.0, 4.0])
    poisoned = np.array([1.0, bad, 2.0, 3.0, bad, 4.0])
    assert pool_quantiles(poisoned) == pool_quantiles(clean)
    assert pool_quantiles(poisoned)["p50"] == pytest.approx(2.5)


def test_pool_quantiles_points_are_ordered():
    # Order statistics are ordered by construction; pinning it is what makes a reader's "the p10
    # shoulder" reading of the block safe without re-sorting it.
    rng = np.random.RandomState(19)
    for _ in range(20):
        q = pool_quantiles(rng.randn(int(rng.randint(2, 200))) * 0.03)
        assert q["p05"] <= q["p10"] <= q["p25"] <= q["p50"] <= q["p75"] <= q["p90"] <= q["p95"]


def test_pool_quantiles_interpolated_past_the_double_range_are_not_finite():
    # p05 and p10 interpolate between -1.7e308 and 1.7e308, a gap past the double range: the
    # computed points are ±inf (p10's true 1.36e308 read -inf) → null, never a finite point —
    # and silently, the overflow an expected input. Points between equal neighbours stand.
    q = pool_quantiles(np.array([-1.7e308] + [1.7e308] * 9))
    assert not np.isfinite(q["p05"]) and not np.isfinite(q["p10"])
    assert q["p25"] == q["p50"] == q["p95"] == 1.7e308


# ---- baseline_summary: the statistical half of a baseline row ----------------


def test_baseline_summary_hand_computed_and_key_set_pins_the_seam():
    # The key set IS the seam contract: baseline_summary owns exactly the fields the return
    # array can carry; n_anchor_bars and the exclusions breakdown are the RUNNER'S count fields
    # (properties of the anchor geometry), mounted beside these — never computed here.
    rets = np.array([0.01, -0.02, 0.03, 0.04])
    b = baseline_summary(rets)
    assert set(b) == {
        "n_eligible",
        "mean_ret",
        "std_ret",
        "hit_rate",
        "ret_quantiles",
        "worst_ret",
        "best_ret",
    }
    assert b["n_eligible"] == 4
    assert b["mean_ret"] == pytest.approx(0.015)
    assert b["std_ret"] == pytest.approx(float(np.std(rets, ddof=1)))
    assert b["hit_rate"] == pytest.approx(0.75)
    assert b["ret_quantiles"] == pool_quantiles(rets)  # same interpolation rule, p50 = median
    assert b["worst_ret"] == pytest.approx(-0.02)
    assert b["best_ret"] == pytest.approx(0.04)


def test_baseline_summary_empty_pool_is_all_null_never_zeros():
    # A zero base rate is a MEASURED outcome; a pool with no observations measured nothing.
    b = baseline_summary(np.array([]))
    assert b["n_eligible"] == 0  # the one honest zero: a count, not a statistic
    for k in ("mean_ret", "std_ret", "hit_rate", "worst_ret", "best_ret"):
        assert np.isnan(b[k])
    assert all(np.isnan(v) for v in b["ret_quantiles"].values())


def test_baseline_summary_drops_non_finite_and_counts_only_what_it_described():
    dirty = np.array([0.01, float("nan"), 0.03, float("inf"), float("-inf")])
    assert baseline_summary(dirty) == baseline_summary(np.array([0.01, 0.03]))
    assert baseline_summary(dirty)["n_eligible"] == 2


def test_baseline_summary_single_observation_has_no_dispersion():
    b = baseline_summary(np.array([0.02]))
    assert b["n_eligible"] == 1 and b["mean_ret"] == pytest.approx(0.02)
    assert np.isnan(b["std_ret"])  # one observation has no dispersion — null, never 0.0
    assert b["worst_ret"] == pytest.approx(0.02) and b["best_ret"] == pytest.approx(0.02)


def test_baseline_summary_is_pool_agnostic_under_concatenation():
    # The basket pooled row is the SAME function over the concatenated (bar × member) eligible
    # observations — nothing in it knows which pool it is describing, so concatenation order
    # cannot move any field.
    a = np.array([0.01, -0.02, 0.03])
    b = np.array([0.005, 0.02])
    pooled = baseline_summary(np.concatenate([a, b]))
    swapped = baseline_summary(np.concatenate([b, a]))
    assert pooled == swapped
    assert pooled["n_eligible"] == 5


# ---- conditional buckets + monotonicity ------------------------------------


def test_conditional_buckets_monotone():
    # ret increases with feature → bucket mean_ret should be increasing
    feat = np.linspace(0, 1, 100)
    rets = feat * 0.1
    ep = _episodes(list(rets), feature=list(feat))
    table = conditional_buckets(ep, "feat", q=4)
    assert len(table) == 4
    assert table["mean_ret"].is_monotonic_increasing
    assert table["n"].sum() == 100


def test_conditional_buckets_missing_feature_raises():
    with pytest.raises(KeyError):
        conditional_buckets(_episodes([0.01, 0.02]), "nope")


def _recs(means: list[float], n: int = 10) -> list[dict]:
    return [{"bucket": str(i), "n": n, "mean_ret": m, "hit_rate": 0.5} for i, m in enumerate(means)]


def test_bucket_monotonicity_increasing_is_rho_one():
    mono = _bucket_monotonicity(_recs([-0.02, 0.0, 0.01, 0.03]))
    assert mono is not None and mono["rho"] == pytest.approx(1.0) and mono["sign"] == 1


def test_bucket_monotonicity_decreasing_is_rho_negative_one():
    mono = _bucket_monotonicity(_recs([0.03, 0.01, 0.0, -0.02]))
    assert mono["rho"] == pytest.approx(-1.0) and mono["sign"] == -1


def test_bucket_monotonicity_non_monotone_is_weak():
    mono = _bucket_monotonicity(_recs([0.0, 0.05, -0.01, 0.04]))  # zig-zag → weak rank signal
    assert mono is not None and abs(mono["rho"]) < 1.0


def test_bucket_monotonicity_needs_three_buckets():
    assert _bucket_monotonicity(_recs([0.01, 0.03])) is None
    # empty / all-empty buckets don't count toward the three
    assert _bucket_monotonicity([{"n": 0, "mean_ret": 0.01}] * 4) is None


def test_bucket_monotonicity_constant_means_is_none():
    # a degenerate cell (all bucket means equal) has no rank signal — None, and no scipy warning
    assert _bucket_monotonicity(_recs([0.01, 0.01, 0.01, 0.01])) is None


def test_bucket_monotonicity_of_an_overflowed_bucket_mean_is_none():
    # A populated bucket whose return sum overflowed reads mean = inf (or NaN off inf − inf),
    # which ranked it above every finite bucket (rho 0.4 here) whatever its true place. Its place
    # is unknown, and so is the rank signal — dropping it would read the rest's rho 1.0 as if
    # the bucket were empty. An EMPTY bucket still just drops out.
    for overflowed in (float("inf"), float("-inf"), float("nan")):
        assert _bucket_monotonicity(_recs([0.01, overflowed, 0.02, 0.03])) is None
    empty = {"bucket": "e", "n": 0, "mean_ret": float("nan"), "hit_rate": float("nan")}
    mono = _bucket_monotonicity([*_recs([0.01, 0.02]), empty, *_recs([0.03])])
    assert mono == {"rho": 1.0, "sign": 1}


def test_bucket_monotonicity_json_safe():
    mono = _bucket_monotonicity(_recs([-0.01, 0.0, 0.02]))
    assert isinstance(mono["rho"], float) and isinstance(mono["sign"], int)
    assert not isinstance(mono["rho"], np.generic) and not isinstance(mono["sign"], np.generic)


# ---- cell_conditional_buckets: per-cell conditioning, explicit refusals ----------


def test_cell_conditional_buckets_monotone_plant():
    feat = np.linspace(0.0, 1.0, 100)
    trades = _episodes(list(feat * 0.1), feature=list(feat))
    out = cell_conditional_buckets(trades, ["feat"])
    assert set(out) == {"conditional_buckets", "bucket_monotonicity"}  # the two keys it mounts
    block = out["conditional_buckets"]["feat"]
    assert block["reason"] is None
    assert len(block["buckets"]) == 4
    assert sum(b["n"] for b in block["buckets"]) == 100
    means = [b["mean_ret"] for b in block["buckets"]]
    assert means == sorted(means)  # ascending feature order carries the planted monotone edge
    mono = out["bucket_monotonicity"]["feat"]
    assert mono["rho"] == pytest.approx(1.0) and mono["sign"] == 1


def test_cell_conditional_buckets_refusal_no_closed_observations():
    empty = pd.DataFrame(columns=["ret", "bars_held", "exit_reason", "feat"])
    out = cell_conditional_buckets(empty, ["feat"])
    assert out["conditional_buckets"]["feat"] == {
        "buckets": [],
        "reason": "no_closed_observations",
    }
    assert out["bucket_monotonicity"] == {}
    # an all-NaN feature column carries no observation of the feature either
    t = _episodes([0.01] * 25, feature=[float("nan")] * 25)
    assert (
        cell_conditional_buckets(t, ["feat"])["conditional_buckets"]["feat"]["reason"]
        == "no_closed_observations"
    )


def test_cell_conditional_buckets_refusal_insufficient_observations():
    t = _episodes([0.01] * (BUCKET_MIN_N - 1), feature=[float(i) for i in range(BUCKET_MIN_N - 1)])
    out = cell_conditional_buckets(t, ["feat"])
    assert out["conditional_buckets"]["feat"] == {
        "buckets": [],
        "reason": "insufficient_observations",
    }
    # the floor is BUCKET_MIN_N = 20, inclusive: exactly 20 valid rows bucket
    assert BUCKET_MIN_N == 20
    t20 = _episodes(
        list(np.linspace(0.0, 0.1, BUCKET_MIN_N)),
        feature=[float(i) for i in range(BUCKET_MIN_N)],
    )
    assert cell_conditional_buckets(t20, ["feat"])["conditional_buckets"]["feat"]["reason"] is None


def test_cell_conditional_buckets_refusal_insufficient_distinct_values():
    t = _episodes([0.01] * 40, feature=[7.0] * 40)  # constant feature — no quantile split exists
    out = cell_conditional_buckets(t, ["feat"])
    assert out["conditional_buckets"]["feat"] == {
        "buckets": [],
        "reason": "insufficient_distinct_values",
    }
    assert "feat" not in out["bucket_monotonicity"]  # a refused feature grades nothing


def test_cell_conditional_buckets_per_feature_reasons_are_independent():
    feat = np.linspace(0.0, 1.0, 60)
    t = _episodes(list(feat * 0.1), feature=list(feat))
    t["flat"] = 1.0
    out = cell_conditional_buckets(t, ["feat", "flat"])
    assert out["conditional_buckets"]["feat"]["reason"] is None
    assert out["conditional_buckets"]["flat"]["reason"] == "insufficient_distinct_values"
    # every requested feature has an entry: refusals are explicit, never absent
    assert set(out["conditional_buckets"]) == {"feat", "flat"}


def test_cell_conditional_buckets_per_target_edges_defeat_simpsons_inversion():
    # Two members, BOTH with a falling within-member edge, the second sitting two orders of
    # magnitude higher in the feature's LEVEL. A pooled qcut splits by member (its top buckets
    # simply ARE member B) and points UP from bottom to top bucket — Simpson's inversion wearing
    # a conditioning read's clothes. Per-target edges read the true within-member direction.
    feat_a = np.linspace(0.0, 1.0, 24)
    feat_b = 100.0 + np.linspace(0.0, 1.0, 24)
    frame = pd.DataFrame(
        {
            "target": ["A"] * 24 + ["B"] * 24,
            "feat": np.concatenate([feat_a, feat_b]),
            "ret": np.concatenate([0.10 - 0.08 * feat_a, 0.30 - 0.08 * (feat_b - 100.0)]),
        }
    )
    # the CONTRAST: the frozen pooled kernel reads the cross-member level artifact as a lift
    pooled_means = list(conditional_buckets(frame, "feat")["mean_ret"])
    assert pooled_means[-1] > pooled_means[0]
    # the panel's per-target edges read the true falling edge, member by member
    out = cell_conditional_buckets(frame, ["feat"])
    block = out["conditional_buckets"]["feat"]
    assert block["reason"] is None
    assert [b["bucket"] for b in block["buckets"]] == ["q1", "q2", "q3", "q4"]
    means = [b["mean_ret"] for b in block["buckets"]]
    assert means == sorted(means, reverse=True)
    mono = out["bucket_monotonicity"]["feat"]
    assert mono["rho"] == pytest.approx(-1.0) and mono["sign"] == -1


def test_cell_conditional_buckets_refuse_a_member_that_cannot_match_bucket_counts():
    # A member that cannot form the siblings' bucket count refuses the WHOLE feature: a panel
    # that quietly dropped the member's rows would move with data quality, not with the
    # hypothesis (no silent partial pools).
    frame = pd.DataFrame(
        {
            "target": ["A"] * 30 + ["B"] * 2,
            "feat": [*np.linspace(0.0, 1.0, 30), 5.0, 5.0],  # B: one distinct value → 1 bucket
            "ret": [0.01] * 32,
        }
    )
    out = cell_conditional_buckets(frame, ["feat"])
    assert out["conditional_buckets"]["feat"] == {
        "buckets": [],
        "reason": "insufficient_distinct_values",
    }
    # …and a member that buckets, but to a DIFFERENT populated count, refuses the same way
    frame2 = pd.DataFrame(
        {
            "target": ["A"] * 30 + ["B"] * 12,
            "feat": [*np.linspace(0.0, 1.0, 30), *np.repeat([1.0, 2.0], 6)],  # B: 2, A: 4
            "ret": [0.01] * 42,
        }
    )
    assert (
        cell_conditional_buckets(frame2, ["feat"])["conditional_buckets"]["feat"]["reason"]
        == "insufficient_distinct_values"
    )


def test_cell_conditional_buckets_aggregates_ordinals_across_targets():
    # Hand-computed two-member aggregation: per-member quartiles of 12 rows each → 3-row
    # buckets; ordinal i pools its 6 union rows with exact row-level statistics.
    feat = np.repeat([1.0, 2.0, 3.0, 4.0], 3)
    ret_a = np.repeat([0.04, 0.03, 0.02, 0.01], 3)
    ret_b = np.repeat([0.08, 0.06, 0.04, -0.02], 3)
    frame = pd.DataFrame(
        {
            "target": ["A"] * 12 + ["B"] * 12,
            "feat": np.concatenate([feat, 100.0 + feat]),
            "ret": np.concatenate([ret_a, ret_b]),
        }
    )
    block = cell_conditional_buckets(frame, ["feat"])["conditional_buckets"]["feat"]
    assert block["reason"] is None
    assert [b["bucket"] for b in block["buckets"]] == ["q1", "q2", "q3", "q4"]
    assert [b["n"] for b in block["buckets"]] == [6, 6, 6, 6]
    assert [b["mean_ret"] for b in block["buckets"]] == pytest.approx([0.06, 0.045, 0.03, -0.005])
    assert [b["hit_rate"] for b in block["buckets"]] == pytest.approx([1.0, 1.0, 1.0, 0.5])


def test_cell_conditional_buckets_withhold_means_across_incommensurable_targets():
    # Under a `diff` outcome each target is in its own level units: a record pooling both
    # members' rows by ordinal has no unit for its mean — null, and so no monotonicity ranks the
    # records — while its n and hit_rate (a count, a share of signs) stand. One participating
    # target's records keep their means.
    feat = np.repeat([1.0, 2.0, 3.0, 4.0], 3)
    frame = pd.DataFrame(
        {
            "target": ["A"] * 12 + ["B"] * 12,
            "feat": np.concatenate([feat, 100.0 + feat]),
            "ret": np.concatenate(
                [np.repeat([0.04, 0.03, 0.02, 0.01], 3), np.repeat([0.08, 0.06, 0.04, -0.02], 3)]
            ),
        }
    )
    plain = cell_conditional_buckets(frame, ["feat"])
    held = cell_conditional_buckets(frame, ["feat"], commensurable=False)
    pairs = zip(
        plain["conditional_buckets"]["feat"]["buckets"],
        held["conditional_buckets"]["feat"]["buckets"],
        strict=True,
    )
    for p, h in pairs:
        assert math.isnan(h["mean_ret"]) and (h["n"], h["hit_rate"]) == (p["n"], p["hit_rate"])
    assert plain["bucket_monotonicity"] and held["bucket_monotonicity"] == {}
    one = frame[frame["target"] == "A"]
    assert cell_conditional_buckets(one, ["feat"], min_n=10, commensurable=False) == (
        cell_conditional_buckets(one, ["feat"], min_n=10)
    )


def test_episode_panels_withhold_value_reads_across_incommensurable_targets():
    # The same rule over the cross-target episode merge: over two targets' rows every mean,
    # |return|-mass share and excursion extreme is NaN — the profile reads as over unknown
    # means, zero streaks and empty excursion blocks — beside the real counts and extents; one
    # target's rows keep every value.
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2022-01-03")
    starts = (0, 0, 10, 10, 20)
    rows = pd.DataFrame(
        {
            "entry_time": [base + d * day for d in starts],
            "exit_time": [base + (d + 3) * day for d in starts],
            "ret": [0.25, -2.0, 0.5, -1.0, 0.125],
            "target": ["rate", "index", "rate", "index", "rate"],
            "mae": [-0.5, -3.0, -0.25, -2.0, -0.125],
            "mfe": [0.5, 1.0, 0.75, 0.5, 0.25],
        }
    )
    stats = episode_stats(rows, commensurable=False)
    assert (stats["n"], stats["n_clusters"], stats["largest_cluster_n"]) == (5, 3, 2)
    assert math.isnan(stats["max_cluster_share_abs"]) and math.isnan(stats["mass_hhi"])
    ledger = episode_ledger(rows, cap=2, commensurable=False)
    assert [e["n"] for e in ledger["entries"]] == [2, 2] and ledger["n_omitted"] == 1
    assert all(math.isnan(e["mean_ret"]) and math.isnan(e["share_abs"]) for e in ledger["entries"])
    assert math.isnan(ledger["omitted_share_abs"])
    prof = episode_profile(rows, commensurable=False)
    assert prof["n_episodes"] == 3 and math.isnan(prof["mean_ret"]) and math.isnan(prof["hit_rate"])
    assert prof["max_win_streak"] == prof["max_loss_streak"] == 0
    assert prof["mae_quantiles"]["n"] == prof["mfe_quantiles"]["n"] == 0
    assert math.isnan(prof["edge_ratio"])
    rate = rows[rows["target"] == "rate"]
    assert episode_profile(rate, commensurable=False)["mean_ret"] == exact_mean(
        np.array([0.25, 0.5, 0.125])
    )
    assert episode_stats(rate, commensurable=False)["max_cluster_share_abs"] == 0.5 / 0.875


def test_cell_conditional_buckets_single_target_column_is_identical():
    # A constant `target` column routes through the same verbatim kernel path as no column at
    # all: single-target cells are bit-identical to the pre-v3 pooled read, interval labels
    # included.
    feat = np.linspace(0.0, 1.0, 40)
    bare = _episodes(list(feat * 0.1), feature=list(feat))
    tagged = bare.copy()
    tagged["target"] = "T"
    assert cell_conditional_buckets(bare, ["feat"]) == cell_conditional_buckets(tagged, ["feat"])
    labels = [
        b["bucket"]
        for b in cell_conditional_buckets(tagged, ["feat"])["conditional_buckets"]["feat"][
            "buckets"
        ]
    ]
    assert all("," in label for label in labels)  # interval labels, never ordinals


def test_cell_conditional_bucket_means_past_the_double_range_are_nan():
    # Finite returns of 1e308 and 1.5e308 in the lower three buckets: each one's sum overflows,
    # and its mean read ±inf — a value the emitted contract (a plain float) then refused, and a
    # sign the overflow picked. NaN → null instead, on the one-target path and on the ordinal
    # path alike, while a bucket whose sum stays in range keeps its bits.
    feat = np.arange(24, dtype=float)
    rets = np.where(feat < 18, np.where(feat % 2 == 0, 1e308, 1.5e308), 0.25)
    one = cell_conditional_buckets(_episodes(list(rets), feature=list(feat)), ["feat"])
    two = cell_conditional_buckets(
        pd.DataFrame(
            {
                "target": ["A"] * 24 + ["B"] * 24,
                "feat": np.concatenate([feat, 100.0 + feat]),
                "ret": np.concatenate([rets, rets]),
            }
        ),
        ["feat"],
    )
    for out, n in ((one, 6), (two, 12)):
        block = out["conditional_buckets"]["feat"]
        assert block["reason"] is None and [b["n"] for b in block["buckets"]] == [n] * 4
        means = [b["mean_ret"] for b in block["buckets"]]
        assert all(np.isnan(m) for m in means[:3]) and means[3] == 0.25
        assert out["bucket_monotonicity"] == {}  # an unknown mean, so no rank signal either


@pytest.mark.parametrize(
    "feat",
    [np.linspace(2e306, 1.7e308, 24), np.linspace(1e-320, 5e-307, 24)],
    ids=["ceiling", "floor"],
)
def test_ordinal_buckets_pandas_cannot_label_keep_their_bins(feat):
    # Per-target edges pandas cannot render: it rounds each interval LABEL's edges through
    # x·10**precision. Past ~1.8e305 every label read "(inf, inf]" (and warned), and the
    # categorical's codes, built off those colliding labels, merged all four quartiles into ONE
    # bucket; under ~1e-306 in magnitude a break read NaN and the labelled qcut raised. Both
    # were refused as `insufficient_distinct_values`. The ordinal path reads the bin indices
    # alone: four buckets of six rows per member, silently.
    frame = pd.DataFrame(
        {
            "target": ["A"] * 24 + ["B"] * 24,
            "feat": np.concatenate([feat, feat]),
            "ret": np.tile(np.repeat([0.01, 0.02, 0.03, 0.04], 6), 2),
        }
    )
    out = cell_conditional_buckets(frame, ["feat"])
    block = out["conditional_buckets"]["feat"]
    assert block["reason"] is None
    assert [b["n"] for b in block["buckets"]] == [12, 12, 12, 12]
    assert [b["mean_ret"] for b in block["buckets"]] == pytest.approx([0.01, 0.02, 0.03, 0.04])
    assert out["bucket_monotonicity"]["feat"] == {"rho": pytest.approx(1.0), "sign": 1}


@pytest.mark.parametrize(
    ("feat", "true_n"),
    [
        # Edges past ~1.8e305 rendered "inf": "(-1.001, inf]" 30 / "(inf, inf]" 30 over three
        # true bins, the two "(inf, inf]" labels' bins merged by the categorical (and warned).
        (np.r_[np.full(30, -1.0), np.linspace(1e306, 1.7e308, 30)], [30, 15, 15]),
        # Edges under ~1e-306 in magnitude: 10**digits overflowed, the breaks read NaN, no
        # interval could be built, and the qcut's ValueError refused a feature with four bins.
        (np.linspace(1e-320, 5e-307, 40), [10, 10, 10, 10]),
        # ONE edge past the ceiling, the labels still distinct: pandas' "(0.77, inf]" names a
        # bin it cannot print — the rendered breaks must be finite, not merely distinct.
        (np.r_[np.linspace(0.0, 1.0, 39), 1e306], [10, 10, 10, 10]),
    ],
    ids=["ceiling", "floor", "top_edge"],
)
def test_single_target_buckets_pandas_cannot_label_keep_their_bins(feat, true_n):
    # The single-target arm passes the frozen kernel's records through, interval labels and all
    # — but its membership came off pandas' rendered labels, which leave the double range at
    # both ends of the feature scale. The bins are read by index now; where no interval string
    # prints them faithfully, each populated bin takes its ordinal, the multi-target arm's
    # label. Silent: pandas' rendering runs under errstate (the suite's -W error).
    ret = np.linspace(-0.05, 0.05, feat.size)  # ascending with the feature: a planted rank
    out = cell_conditional_buckets(pd.DataFrame({"feat": feat, "ret": ret}), ["feat"])
    block = out["conditional_buckets"]["feat"]
    assert block["reason"] is None
    assert [(b["bucket"], b["n"]) for b in block["buckets"]] == [
        (f"q{i + 1}", n) for i, n in enumerate(true_n)
    ]
    if len(true_n) >= 3:
        assert out["bucket_monotonicity"]["feat"] == {"rho": pytest.approx(1.0), "sign": 1}
    table = conditional_buckets(pd.DataFrame({"feat": feat, "ret": ret}), "feat")
    assert list(table.index.astype(str)) == [f"q{i + 1}" for i in range(len(true_n))]


def test_single_target_buckets_pandas_can_label_keep_its_intervals():
    # The ordinary twin: wherever pandas prints the edges, the kernel's table IS pandas' qcut
    # grouping — the interval categorical as the index, byte for byte.
    feat = np.r_[np.full(10, 1e-300), np.linspace(1e305, 1.7e305, 30)]
    frame = pd.DataFrame({"feat": feat, "ret": np.linspace(-1.0, 1.0, feat.size)})
    table = conditional_buckets(frame, "feat")
    expected = frame.groupby(pd.qcut(frame["feat"], q=4, duplicates="drop"), observed=True)["ret"]
    assert table.index.equals(expected.size().index)
    assert list(table["n"]) == list(expected.size()) == [10, 10, 10, 10]
    assert str(table.index[-1]).endswith(", 1.7e+305]")  # an edge just short of the ceiling


@pytest.mark.parametrize("n_targets", [1, 2], ids=["one_target", "two_targets"])
def test_buckets_of_a_feature_spanning_the_double_range_are_its_true_bins(n_targets):
    # A feature whose values span more than the double range: pandas interpolates a quantile as
    # a + (b − a)·t, and b − a overflows between the two order statistics either side of zero,
    # so the median edge read -inf (with a warning) and both arms cut their bins on it — 1/17/6
    # where the true bins are 6/6/6/6, a non-monotone panel with a rank read. Such a pool holds
    # only values past 2**970 in magnitude, so its halved twin's bins are exactly its own: the
    # true bins, labelled by ordinal (no interval label prints an edge there), silently.
    rng = np.random.default_rng(3)
    feat = np.r_[-1.7e308 * rng.uniform(0.95, 1, 12), 1.7e308 * rng.uniform(0.95, 1, 12)]
    rng.shuffle(feat)
    ret = np.argsort(np.argsort(feat)) / 100.0  # the feature's rank: ascending with it
    with np.errstate(over="ignore", invalid="ignore"):  # the premise: pandas' edge overflows
        assert not np.isfinite(pd.qcut(feat, 4, retbins=True, labels=False)[1]).all()
    true = pd.qcut(feat * 2.0**-500, 4, labels=False)  # the same pool, well inside the range
    true_means = [ret[true == i].mean() for i in range(4)]
    frame = pd.DataFrame(
        {
            "target": np.repeat(["A", "B"][:n_targets], feat.size),
            "feat": np.tile(feat, n_targets),
            "ret": np.tile(ret, n_targets),
        }
    )
    out = cell_conditional_buckets(frame, ["feat"])
    block = out["conditional_buckets"]["feat"]
    assert block["reason"] is None
    assert [(b["bucket"], b["n"]) for b in block["buckets"]] == [
        (f"q{i + 1}", 6 * n_targets) for i in range(4)
    ]
    assert [b["mean_ret"] for b in block["buckets"]] == pytest.approx(true_means)
    assert out["bucket_monotonicity"]["feat"] == {"rho": pytest.approx(1.0), "sign": 1}


def _full_bar_frame(order, feat, n_bars=48, seed=41):
    """A ``cross_mean`` pool whose every bar is FULL: members A, B, C each hold one row on every
    bar 0..n_bars-1, its excess the member's leg less the bar's basket mean — three rounded
    values whose exact sum is 0 and whose rounded sum is residue. Rows run in ``order``'s
    target-then-bar order (the runner's frame order); ``feat[member]`` is the member's
    entry-time feature column over the bars. Returns the frame and the (bars × A, B, C) legs."""
    rng = np.random.RandomState(seed)
    raw = np.round(rng.normal(0.0, 0.02, (n_bars, 3)), 4)
    excess = dict(zip("ABC", (raw - raw.mean(axis=1, keepdims=True)).T, strict=True))
    bars = np.arange(n_bars)
    frame = pd.DataFrame(
        {
            "target": np.repeat(list(order), n_bars),
            "entry_bar": np.tile(bars, len(order)),
            "feat": np.concatenate([feat[m] for m in order]),
            "ret": np.concatenate([excess[m] for m in order]),
        }
    )
    return frame, raw


def _bar_pool(frame, raw, members="ABC"):
    """The rows' bar numerators (``cross_mean_bar_numerators``) row-aligned with ``frame``: each
    bar's exact numerator over the members holding a row in ``frame`` on its first such member
    (in ``members`` order), zeros on the others — what the runner hands the buckets as
    ``rets_pool``, over ``pool_scale`` 3."""
    col = {m: g for g, m in enumerate(members)}
    bars = frame["entry_bar"].to_numpy(dtype=np.int64)
    member = frame["target"].map(col).to_numpy(dtype=np.int64)
    closed = np.zeros(raw.shape, dtype=bool)
    closed[bars, member] = True
    return cross_mean_bar_numerators(raw, closed).at(bars, member)


def test_cell_buckets_read_a_whole_full_bar_at_its_exact_zero_in_any_declaration_order():
    # An all-full cross_mean pool bucketed by a basket-wide feature (the bar's weekday, the same
    # for every member): every member's quartile edges agree, so every bar's rows land in ONE
    # ordinal and every record is whole bars — whose exact mean is 0.0. Averaged over the
    # rounded rows it is residue (~1e-19), and the residue ranks: a bucket_monotonicity whose
    # rho (pairwise sums) moved with nothing but the order the targets were declared in. The
    # counts and hit rates stay row-level: each row's excess is its own.
    import itertools

    dow = np.arange(48) % 4
    feat = {m: dow.astype(float) for m in "ABC"}
    plain_reads = set()
    for order in itertools.permutations("ABC"):
        frame, raw = _full_bar_frame(order, feat)
        plain = cell_conditional_buckets(frame, ["feat"])
        out = cell_conditional_buckets(
            frame, ["feat"], rets_pool=_bar_pool(frame, raw, order), pool_scale=3
        )
        block = out["conditional_buckets"]["feat"]
        plain_block = plain["conditional_buckets"]["feat"]
        assert block["reason"] is None and len(block["buckets"]) == 4
        assert [b["mean_ret"] for b in block["buckets"]] == [0.0] * 4, order
        assert out["bucket_monotonicity"] == {}, order  # all means equal: no rank signal
        assert [(b["bucket"], b["n"], b["hit_rate"]) for b in block["buckets"]] == [
            (b["bucket"], b["n"], b["hit_rate"]) for b in plain_block["buckets"]
        ]
        assert 0.0 < block["buckets"][0]["hit_rate"] < 1.0
        plain_reads.add(
            (
                tuple(b["mean_ret"] for b in plain_block["buckets"]),
                str(plain["bucket_monotonicity"]),
            )
        )
    # the premise: over the rounded rows the means are residue, and the residue ranks
    assert any(m != 0.0 for means, _ in plain_reads for m in means)
    assert all(mono != "{}" for _, mono in plain_reads)


def test_cell_buckets_bars_split_across_ordinals_are_byte_identical():
    # A PER-MEMBER feature splits every bar: A's feature ascends with the bar and B's descends,
    # so A's quartile j and B's 3 - j never coincide — no record holds a bar's whole sum, and
    # every row keeps its own return: the panel is the plain read, byte for byte.
    rng = np.random.RandomState(43)
    bars = np.arange(48, dtype=float)
    feat = {"A": bars, "B": -bars, "C": rng.normal(size=48)}
    frame, raw = _full_bar_frame("ABC", feat)
    out = cell_conditional_buckets(frame, ["feat"], rets_pool=_bar_pool(frame, raw), pool_scale=3)
    assert out == cell_conditional_buckets(frame, ["feat"])
    assert out["conditional_buckets"]["feat"]["reason"] is None
    # the rows' own returns as the pool values ARE the plain read; a misaligned pool is refused
    plain = cell_conditional_buckets(frame, ["feat"])
    assert cell_conditional_buckets(frame, ["feat"], rets_pool=frame["ret"].to_numpy()) == plain
    with pytest.raises(ValueError, match="row-aligned"):
        cell_conditional_buckets(frame, ["feat"], rets_pool=np.zeros(3))


def test_cell_buckets_read_every_bar_held_whole_at_its_exact_sum():
    # One pool, every case at once. Every member keeps ten rows of each feature value 0..3, so
    # the per-target quartile edges agree and a row's ordinal IS its value. Bars 0..39 are full:
    # most land whole in one ordinal, but bars 4 and 9 are SPLIT (A and B swap their values 0
    # and 1) and bars 6 and 10 are PARTLY UNBUCKETED (C's feature missing on 6, A's and B's on
    # 10); bar 40 holds A and B only and bar 41 C only (partial bars, each whole in ordinal 2).
    # Every bar held whole reads its exact excess sum in its record's mean (0.0 for a full bar),
    # every other row its own return — and n and hit_rate are the rows' own throughout.
    n_bars = 42
    value = (np.arange(n_bars) % 4).astype(float)
    value[40] = value[41] = 2.0
    feat = {m: value.copy() for m in "ABC"}
    for m in "AB":
        feat[m][4], feat[m][9] = value[9], value[4]
        feat[m][10] = np.nan
    feat["C"][6] = np.nan
    frame, raw = _full_bar_frame("ABC", feat, n_bars=n_bars)
    bar, tgt = frame["entry_bar"].to_numpy(), frame["target"].to_numpy()
    frame = frame[~(((bar == 40) & (tgt == "C")) | ((bar == 41) & (tgt != "C")))]
    pool = _bar_pool(frame, raw)
    valid_rows = frame["feat"].notna().to_numpy()
    valid = frame[valid_rows]
    for _, rows in valid.groupby("target"):  # the design: each member's ordinal is its value
        codes = pd.qcut(rows["feat"], 4, labels=False, duplicates="drop").to_numpy()
        assert (codes == rows["feat"].to_numpy()).all()

    out = cell_conditional_buckets(frame, ["feat"], rets_pool=pool, pool_scale=3)
    plain = cell_conditional_buckets(frame, ["feat"])
    block = out["conditional_buckets"]["feat"]["buckets"]
    whole = valid["entry_bar"].isin(sorted(set(range(42)) - {4, 9, 6, 10})).to_numpy()
    # a bucket's exact numerator: its whole bars' exact parts, every other row's return times 3
    numerators = [
        3 * Fraction(float(r)) if not w else sum(map(Fraction, parts))
        for r, w, parts in zip(valid["ret"], whole, pool[valid_rows].tolist(), strict=True)
    ]
    ordinal = valid["feat"].to_numpy()
    assert [b["mean_ret"] for b in block] == [
        float(sum(n for n, o in zip(numerators, ordinal, strict=True) if o == i))
        / (3 * int((ordinal == i).sum()))
        for i in range(4)
    ]
    assert [(b["n"], b["hit_rate"]) for b in block] == [
        (b["n"], b["hit_rate"]) for b in plain["conditional_buckets"]["feat"]["buckets"]
    ]
    # value 3 holds whole full bars alone: exactly 0.0; the other three hold a kept row
    assert block[3]["mean_ret"] == 0.0
    assert all(b["mean_ret"] != 0.0 for b in block[:3])
    # the partial bars enter ordinal 2 at their exact sums, never their rows' rounded sum
    for b in (40, 41):
        rows = frame["entry_bar"].to_numpy() == b
        legs = [Fraction(float(x)) for x in raw[b]]
        exact = sum(legs["ABC".index(m)] - sum(legs) / 3 for m in frame["target"][rows])
        assert sum(map(Fraction, pool[rows].ravel().tolist())) == 3 * exact


# ---- feature_outcome_association: per-(cell × feature × target) Spearman ----------


def test_feature_outcome_association_planted_monotone_is_rho_one():
    vals = np.arange(20, dtype=float)
    out = feature_outcome_association(vals, vals**3)  # monotone but nonlinear: rank rho is 1
    assert out["rho"] == pytest.approx(1.0) and out["n"] == 20 and out["reason"] is None


def test_feature_outcome_association_planted_inverse_is_rho_minus_one():
    vals = np.arange(15, dtype=float)
    out = feature_outcome_association(vals, -vals)
    assert out["rho"] == pytest.approx(-1.0) and out["n"] == 15 and out["reason"] is None


def test_feature_outcome_association_floor_is_assoc_min_n_inclusive():
    assert ASSOC_MIN_N == 10
    nine = np.arange(9, dtype=float)
    assert feature_outcome_association(nine, nine) == {
        "rho": None,
        "n": 9,
        "reason": "insufficient_observations",
    }
    ten = np.arange(10, dtype=float)
    assert feature_outcome_association(ten, ten)["reason"] is None


def test_feature_outcome_association_drops_non_finite_pairs():
    # pairwise-finite: a NaN/inf on EITHER side drops the pair, and n counts what was ranked
    vals = np.concatenate([np.arange(10, dtype=float), [float("nan"), 5.0, float("inf")]])
    rets = np.concatenate([np.arange(10, dtype=float) * 0.1, [0.5, float("nan"), 0.7]])
    out = feature_outcome_association(vals, rets)
    assert out["n"] == 10 and out["rho"] == pytest.approx(1.0) and out["reason"] is None
    # dropping below the floor refuses with the count it actually had
    short = feature_outcome_association(
        np.array([1.0, 2.0, float("nan")]), np.array([0.1, 0.2, 0.3])
    )
    assert short == {"rho": None, "n": 2, "reason": "insufficient_observations"}


def test_feature_outcome_association_constant_input_is_no_rank_variation():
    # rho = 0 would claim a MEASURED absence of association; an undefined rank statistic is a
    # refusal with a reason instead.
    const = np.full(12, 3.0)
    varying = np.arange(12, dtype=float)
    assert feature_outcome_association(const, varying) == {
        "rho": None,
        "n": 12,
        "reason": "no_rank_variation",
    }
    assert feature_outcome_association(varying, const) == {
        "rho": None,
        "n": 12,
        "reason": "no_rank_variation",
    }


def test_feature_outcome_association_emits_no_p_value():
    # Deliberate: an overlap-inflated Spearman p is exactly the over-trustable number the
    # doctrine forbids, and there is no honest event-time correction to put in its place.
    out = feature_outcome_association(
        np.arange(30, dtype=float), np.random.RandomState(1).randn(30)
    )
    assert set(out) == {"rho", "n", "reason"}


# ---- event-time Newey-West HAC ---------------------------------------------


def test_newey_west_h1_matches_ols_se():
    # h=1 windows cannot overlap — the event-time HAC reduces to the plain OLS SE of the mean.
    rng = np.random.RandomState(0)
    rets = rng.normal(0.01, 0.02, size=200)
    t0, se0 = newey_west_mean(rets, np.arange(200), h=1)
    expected_se = np.sqrt(np.var(rets, ddof=1) / rets.shape[0])  # HAC lag-0 = sample var / n
    assert se0 == pytest.approx(expected_se, rel=1e-9)
    assert t0 == pytest.approx(rets.mean() / expected_se, rel=1e-9)


def test_newey_west_widens_se_under_overlap():
    # Overlapping window-sums induce strong positive autocorrelation (MA): on a contiguous
    # every-bar pool the event-time HAC SE (bar gap = event lag) must be materially wider than
    # the iid SE, so the honest t is much smaller than the naive t.
    rng = np.random.RandomState(1)
    w, T = 20, 600
    x = rng.normal(0.001, 1.0, T + w)
    rets = np.array([x[i : i + w].mean() for i in range(T)])  # overlapping → MA(w-1)
    bars = np.arange(T)
    _, se_iid = newey_west_mean(rets, bars, h=1)
    t_hac, se_hac = newey_west_mean(rets, bars, h=w)
    t_iid, _ = newey_west_mean(rets, bars, h=1)
    assert se_hac > 1.5 * se_iid
    assert abs(t_hac) < abs(t_iid)


def test_newey_west_contiguous_matches_classic_bartlett():
    # On a contiguous every-bar pool (bar gap d = event lag k) the event-time estimator must
    # reproduce the classic overlap HAC with truncation lag h-1 EXACTLY: w_k = 1 - k/h both ways.
    rng = np.random.RandomState(7)
    rets = rng.normal(0.002, 0.01, size=120)
    h = 12
    t_new, se_new = newey_west_mean(rets, np.arange(rets.size), h=h)
    dev = rets - rets.mean()
    n = rets.size
    s = float(dev @ dev) / (n - 1)
    for k in range(1, min(h - 1, n - 1) + 1):
        s += 2.0 * (1.0 - k / h) * float(dev[k:] @ dev[:-k]) / (n - 1)
    se_classic = np.sqrt(s / n)
    assert se_new == pytest.approx(se_classic, rel=1e-12)
    assert t_new == pytest.approx(rets.mean() / se_classic, rel=1e-12)


def test_newey_west_sparse_independent_events_keep_iid_se():
    # THE P0 regression: on a sparse pool whose events sit further apart than the horizon there
    # is NO overlap — every cross term must drop and the SE must equal the iid SE exactly. The
    # ordinal-lag estimator collapsed this SE to ~0.23x (Bartlett with lag ≈ n → Σdev ≈ 0),
    # fabricating significance on independent returns.
    rng = np.random.RandomState(11)
    n, h = 12, 60
    rets = rng.standard_normal(n)
    bars = np.arange(n) * 500  # events years apart in bar time
    _t_ev, se_ev = newey_west_mean(rets, bars, h=h)
    iid_se = rets.std(ddof=1) / np.sqrt(n)  # the ddof=1 divisor makes the reduction EXACT
    assert se_ev == pytest.approx(iid_se, rel=1e-12)
    # And in distribution: the median SE ratio over many draws sits at ~1, not ~0.23.
    ratios = []
    for seed in range(300):
        r = np.random.RandomState(seed).standard_normal(n)
        _, se = newey_west_mean(r, bars, h=h)
        ratios.append(se / (r.std(ddof=1) / np.sqrt(n)))
    assert 0.9 < float(np.median(ratios)) < 1.1


def test_newey_west_partial_overlap_weights_actual_bar_gap():
    # Two events d bars apart under horizon h get weight 1 - d/h — not the ordinal weight
    # 1 - 1/h a contiguous pool would give its first lag.
    rets = np.array([0.01, -0.02, 0.015, 0.005])
    h = 10
    bars = np.array([0, 7, 40, 80])  # only the first pair overlaps (d=7 < h)
    _, se = newey_west_mean(rets, bars, h=h)
    dev = rets - rets.mean()
    n = rets.size
    s = float(dev @ dev) / (n - 1) + 2.0 * (1.0 - 7.0 / h) * float(dev[0] * dev[1]) / (n - 1)
    assert se == pytest.approx(np.sqrt(s / n), rel=1e-12)


def test_newey_west_unsorted_bars_are_sorted_internally():
    rng = np.random.RandomState(5)
    rets = rng.standard_normal(30)
    bars = np.arange(30) * 3
    perm = rng.permutation(30)
    t_a, se_a = newey_west_mean(rets, bars, h=10)
    t_b, se_b = newey_west_mean(rets[perm], bars[perm], h=10)
    assert (t_a, se_a) == pytest.approx((t_b, se_b), rel=1e-12)


def _hac_exact(rets: np.ndarray, bars: np.ndarray, h: int) -> tuple[float, float]:
    """``(t, se)`` of the event-time HAC in exact rational arithmetic — the estimator's own
    definition, ``Σ_i Σ_j w(|b_i − b_j|)·dev_i·dev_j / (n − 1) / n`` about the exact mean, with
    only the final square root and quotient rounded."""
    n = rets.size
    q = [Fraction(r) for r in rets.tolist()]
    mean = sum(q) / n
    dev = [x - mean for x in q]
    s = Fraction(0)
    for i in range(n):
        for j in range(n):
            w = 1 - Fraction(abs(int(bars[i]) - int(bars[j])), h)
            if w > 0:
                s += w * dev[i] * dev[j]
    se = math.sqrt(s / (n - 1) / n)
    return float(mean) / se, se


def test_newey_west_sums_each_bar_before_the_quadratic_form():
    # Ten members fire together on 40 bars, two apart, at h = 1: five rows of +0.5, four of −0.5
    # and one of −0.5 + k·2**−24 (k cycling 1..4), so each bar's deviations cancel to
    # (k − 2.5)·2**−24. Accumulated row by row, the squared deviations (~0.25 each) and the
    # same-bar cross terms that cancel them were summed apart, and the cancellation destroyed the
    # combined variance: SE 3.7252902984619143e-10 and t 40.0, where the estimator's exact value
    # is SE 1.05499077648638e-9, t 14.124446891825533 — a 65% understatement. The rows enter
    # through their per-bar sums now, exactly rounded.
    rets, bars = [], []
    for b in range(40):
        rets += [0.5] * 5 + [-0.5] * 4 + [-0.5 + (b % 4 + 1) * 2.0**-24]
        bars += [2 * b] * 10
    rets_a, bars_a = np.array(rets), np.array(bars)
    t, se = newey_west_mean(rets_a, bars_a, 1)
    t_exact, se_exact = _hac_exact(rets_a, bars_a, 1)
    assert (t, se) == (t_exact, se_exact) == (14.124446891825533, 1.05499077648638e-9)
    # an ordinary overlapping basket pool agrees with the exact estimator to rounding
    rng = np.random.RandomState(9)
    bars_b = np.sort(rng.randint(0, 60, 90))
    rets_b = rng.normal(0.001, 0.02, 90)
    t, se = newey_west_mean(rets_b, bars_b, 7)
    t_exact, se_exact = _hac_exact(rets_b, bars_b, 7)
    assert t == pytest.approx(t_exact, rel=1e-13) and se == pytest.approx(se_exact, rel=1e-13)


def test_newey_west_same_bar_sum_keeps_the_centre_a_rounded_deviation_lost():
    # mean = 2**-53 / 5. Each row's `r − mean` rounded to ±0.5, so the two-row bars summed to 0
    # for their exact −2·mean and the SE read 1.986e-17 for the exact 2.432e-17. One fsum over a
    # bar's raw returns and its copies of −mean keeps it: the estimator's exact value.
    rets, bars = np.array([0.5, -0.5, 0.5, -0.5, 2.0**-53]), np.array([0, 0, 3, 3, 6])
    t, se = newey_west_mean(rets, bars, 1)
    assert (t, se) == _hac_exact(rets, bars, 1) == (0.9128709291752769, 2.432376777795247e-17)
    # the bar's rows enter in any order
    assert newey_west_mean(rets[[1, 0, 3, 2, 4]], bars, 1) == (t, se)


def test_newey_west_same_bar_sum_past_the_double_range_is_null_never_a_crash():
    # Finite returns whose same-bar sum leaves the double range: fsum raised OverflowError out
    # of the run (exit 4 on valid input) where every other unrepresentable estimate is null.
    rets, bars = np.array([-1e308, 8e307, 8e307, -1e308]), np.array([0, 1, 1, 2])
    t, se = newey_west_mean(rets, bars, 1)
    assert np.isnan(t) and np.isnan(se)
    # the basket pooled read that reaches it: members A and B fire together on bar 1
    T = 50
    fwd_a, fwd_b, ma, mb = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    fwd_a[[0, 1]], fwd_b[[1, 2]] = [-1e308, 8e307], [8e307, -1e308]
    ma[[0, 1]], mb[[1, 2]] = 1.0, 1.0
    cells = [
        {"key": ("c", "A"), "mask_col": ma, "fwd_col": fwd_a, "h": 1},
        {"key": ("c", "B"), "mask_col": mb, "fwd_col": fwd_b, "h": 1},
    ]
    read = pooled_reliability_summary(cells, T)["per_cell"][("c",)]
    assert np.isnan(read["t_hac"]) and np.isnan(read["hac_se"])
    # a deviation past the range reads quietly in the zero-dispersion gate: no overflow warning
    t, se = newey_west_mean(np.array([1.7e308, -1.7e308, 1.7e308]), np.arange(3), 2)
    assert np.isnan(t) and np.isnan(se)


def test_newey_west_bar_deviation_sums_are_exact_about_the_exact_mean():
    # Each bar's Σ(r − μ) about the pool's EXACT mean μ: its exact value, rounded once — no row's
    # deviation rounded before the bar sums it, and no centre rounded either, at any level.
    rng = np.random.RandomState(21)
    for level in (0.0, 1.0, 1e6):
        rets = level + rng.normal(0.0, 0.02, 400) * 10.0 ** rng.randint(-4, 5, 400)
        bars = rng.randint(0, 90, 400)
        assert (np.bincount(bars) > 1).any() and (np.bincount(bars) == 1).any()
        mu = sum(map(Fraction, rets.tolist())) / rets.size
        u, d = _bar_deviation_sums(rets, bars)
        assert u.tolist() == sorted(set(bars.tolist()))
        for a, got in zip(u.tolist(), d.tolist(), strict=True):
            exact = sum(Fraction(v) - mu for v in rets[bars == a].tolist())
            assert got == float(exact), (level, a)


def test_newey_west_members_cancelling_within_each_bar_have_no_long_run_variance():
    # The review's basket: three members alternate [0.8, −0.4, −0.4] and [0, 0, 0] over 40 bars
    # beside a fourth at 2**−52 (h = 1), so every bar sits exactly at the mean 2**−54. Each row's
    # deviation was rounded about the pool's range midpoint before its bar summed it, and the bars
    # read ±5.6e-17 — an SE of 1.6e-18 and t 35.7 where the estimator has no variance at all.
    # The bar sums are exact about the exact mean: no estimate, on the kernel and on the basket's
    # pooled read. With the fourth at 2**−51 on every other bar the bars vary for real (±2**−53)
    # and the SE is the estimator's exact value; it read 27% high.
    def basket(odd):
        cols = np.zeros((50, 4))
        cols[0:40:2, :3] = [0.8, -0.4, -0.4]
        cols[:40, 3] = np.where(np.arange(40) % 2 == 0, 2.0**-52, odd)
        return cols

    for odd, exact_t in ((2.0**-52, None), (2.0**-51, 18.914280319377735)):
        cols = basket(odd)
        rets, bars = cols[:40].ravel(), np.repeat(np.arange(40), 4)
        t, se = newey_west_mean(rets, bars, 1)
        cells = [
            {"key": ("c", m), "mask_col": np.arange(50) < 40, "fwd_col": cols[:, g], "h": 1}
            for g, m in enumerate("ABCD")
        ]
        read = pooled_reliability_summary(cells, 50)["per_cell"][("c",)]
        if exact_t is None:
            assert exact_mean(rets) == 2.0**-54
            assert np.isnan(t) and np.isnan(se)
            assert np.isnan(read["t_hac"]) and np.isnan(read["hac_se"])
        else:
            t_exact, se_exact = _hac_exact(rets, bars, 1)
            assert t_exact == exact_t
            assert se == pytest.approx(se_exact, rel=1e-13) and t == exact_mean(rets) / se
            assert (read["t_hac"], read["hac_se"]) == (t, se)


def test_newey_west_deviations_are_summed_about_the_pools_offset():
    # The review's pool: 10,000 returns at 1 + k·2**-52 (k alternating 0 and 1, the last 32),
    # three bars apart at h = 1, so the estimate is the iid std/√n. The reported mean rounds at
    # the LEVEL's ulp, as large as the deviations themselves, and about it the HAC read an SE of
    # 1.714e-18 for the exact 1.312e-18 — while the same pool shifted onto 0, whose mean is
    # representable, read the exact value: the estimate hung on the level. Summed exactly about
    # the exact mean, the pool and its shift read the same SE, the exact one to rounding.
    u = np.tile([0.0, 1.0], 5000)
    u[-1] = 32.0
    rets = 1.0 + 2.0**-52 * u
    bars = 3 * np.arange(rets.size)
    t, se = newey_west_mean(rets, bars, 1)
    q = [Fraction(r) for r in rets.tolist()]
    mu = sum(q) / len(q)
    se_exact = math.sqrt(sum((x - mu) ** 2 for x in q) / (len(q) - 1) / len(q))
    assert se == pytest.approx(se_exact, rel=1e-15)
    assert newey_west_mean(rets - 1.0, bars, 1)[1] == se
    assert se == pytest.approx(float(np.std(rets - 1.0, ddof=1)) / 100.0, rel=1e-15)
    assert t == exact_mean(rets) / se  # t stays the reported mean over the SE


def test_newey_west_pool_whose_bars_sit_at_its_mean_is_null():
    # Every row on ONE bar: the one deviation sum Σr − n·mean is exactly 0 for the exact mean,
    # and the quadratic form read the centre's rounding — (1.03e16, 2.27e-17) for rows that vary.
    t, se = newey_west_mean(np.array([0.1, 0.2, 0.4]), np.zeros(3, dtype=int), 5)
    assert np.isnan(t) and np.isnan(se)
    # Two bars, each at the pool's mean however its rows vary: the same residue.
    rets = np.array([0.1, 0.2, 0.4, 0.4, 0.1, 0.2])
    t, se = newey_west_mean(rets, np.array([0, 0, 0, 5, 5, 5]), 1)
    assert np.isnan(t) and np.isnan(se)
    # bars whose means genuinely differ still read
    shifted = rets + np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0])
    t, se = newey_west_mean(shifted, np.array([0, 0, 0, 5, 5, 5]), 1)
    assert np.isfinite(t) and se > 0.0


def test_exact_mean_keeps_the_exact_sum_sign_and_zero():
    # np.mean sums pairwise in floating point, and a large term swallows a small one before its
    # cancelling partner arrives: the 49 diff returns of [0, 2**54, 0, −1, 0] × 10 sum to exactly
    # 0 and np.mean read 1/49. The exactly rounded sum keeps the exact sum's sign and its zero.
    levels = np.array([0.0] + [0.0, 2.0**54, 0.0, -1.0, 0.0] * 10)
    rets = np.diff(levels[1:])  # the h = 1 diff returns open[t+2] − open[t+1], t = 0..48
    assert sum(Fraction(r) for r in rets.tolist()) == 0
    assert float(np.mean(rets)) == 1 / 49  # the defect
    assert exact_mean(rets) == 0.0
    assert pool_moments(rets)["mean_ret"] == 0.0
    assert baseline_summary(rets)["mean_ret"] == 0.0
    assert subperiod_means(rets, np.arange(49), np.array([0, 49]))[0]["mean_ret"] == 0.0
    # a residue of the exact sum keeps its sign
    assert exact_mean(np.array([2.0**54, 1.0, -(2.0**54)])) == 1 / 3
    assert exact_mean(np.array([2.0**54, -1.0, -(2.0**54)])) == -1 / 3
    # an all-equal pool's mean is its value; empty, non-finite and an overflowed sum are NaN
    assert exact_mean(np.full(13, 0.3)) == 0.3 != float(np.mean(np.full(13, 0.3)))
    assert math.isnan(exact_mean(np.empty(0)))
    assert math.isnan(exact_mean(np.array([1.0, np.inf])))
    assert math.isnan(exact_mean(np.array([1.7e308, 1.7e308, 1.0])))
    # an ordinary pool reads the correctly rounded sum over n
    v = np.random.RandomState(2).normal(0.0, 0.01, 1000)
    assert exact_mean(v) == math.fsum(v.tolist()) / 1000


def test_newey_west_emits_no_p_value():
    # The engine never computes a p from t_hac: on an overlapping pool the row count wildly
    # overstates the information, so any p is the CONSUMER's to derive at df = n_nonoverlap − 1,
    # caveat in hand. The kernel returns exactly (t, se).
    rets = np.random.RandomState(3).randn(200) * 0.01 + 0.002
    out = newey_west_mean(rets, np.arange(200), h=5)
    assert len(out) == 2 and all(np.isfinite(v) for v in out)
    assert newey_west_mean(rets[:1], np.arange(1), h=5) == (pytest.approx(np.nan, nan_ok=True),) * 2


def test_nonoverlap_count_packs_greedily():
    assert nonoverlap_count(np.array([0, 1, 2, 10, 11, 20]), horizon=5) == 3  # 0, 10, 20
    assert nonoverlap_count(np.array([]), horizon=5) == 0
    assert nonoverlap_count(np.array([0, 5, 10]), horizon=5) == 3  # exactly non-overlapping


# ---- the circular-shift rotation null (per-cell, no cross-cell statistic) ---


def test_tail_p_is_the_corrected_right_tail_fraction():
    # (1 + #{null >= observed}) / (1 + #null) — the small-sample correction that makes the
    # smallest attainable p 1/(1+n_shifts) rather than 0.
    null = np.array([0.0, 1.0, 2.0, 3.0])
    assert _tail_p(2.0, null) == (pytest.approx(3.0 / 5.0), 4)
    assert _tail_p(10.0, null) == (pytest.approx(1.0 / 5.0), 4)  # nothing beat it — the floor
    p, n_null = _tail_p(float("nan"), null)
    assert np.isnan(p) and n_null == 4
    p, n_null = _tail_p(1.0, np.array([np.nan, np.nan]))
    assert np.isnan(p) and n_null == 0
    # an UNDEFINED shift (NaN) drops out of the null, so the count — and with it the p's own
    # floor 1/(1 + n_null) — is what the reported rot_n_null carries
    assert _tail_p(2.0, np.array([0.0, np.nan, 3.0])) == (pytest.approx(2.0 / 3.0), 2)


def test_tail_p_tolerance_counts_near_ties_as_ties():
    # A null value within the two error bounds below the observed one is a tie the arithmetic
    # cannot resolve: it counts as >= (conservative — a band only raises p). The tolerances ride
    # the same NaN filter as the null, and zero tolerance is the exact comparison.
    null = np.array([1.0 - 3e-16, np.nan, 1.0 - 1e-9, 2.0])
    assert _tail_p(1.0, null) == (pytest.approx(2.0 / 4.0), 3)
    tol = np.array([2e-16, 5.0, 2e-16, 2e-16])
    assert _tail_p(1.0, null, 2e-16, tol) == (pytest.approx(3.0 / 4.0), 3)
    assert _tail_p(1.0, null, 0.0, np.zeros(4)) == _tail_p(1.0, null)


def test_tail_p_of_an_overflowed_observed_mean_is_nan_never_the_floor():
    # An observed mean past the double range sat above every finite null value: the p FLOOR,
    # the strongest evidence the rotation can give, minted from an overflow. No observation: NaN.
    for observed in (float("inf"), float("-inf")):
        p, n_null = _tail_p(observed, np.array([0.1, 0.2, 0.3]))
        assert np.isnan(p) and n_null == 3


def test_rotation_null_near_the_ceiling_reads_its_exact_p_in_the_unit_frame():
    # One firing on a forward column holding 5e307 and 1e307: the raw transform's partial sums
    # (~T × 5e307) left the double range at two DEFINED shifts — one of them the shift onto
    # 5e307, the only one that beats the observed 1e307 — and the read was voided (null, never
    # the 1/7 of its survivors). In the column's unit frame nothing leaves the range: exact
    # arithmetic's 2/9 over all 8 defined shifts, as on the same column far below the ceiling.
    fwd = np.array([-0.4, -0.75, 0.2, 5e307, -1.6, 2.0, -1.0, -0.2, 1e307])
    mask = np.zeros(9)
    mask[8] = 1.0
    for column in (fwd, fwd * 1e-300):
        read = reliability_summary(
            [{"key": ("t",), "mask_col": mask, "fwd_col": column, "h": 1}], 9, ["t"]
        )["per_cell"][("t",)]
        assert read["rot_p"] == 2 / 9 and read["rot_n_null"] == 8


def test_a_subnormal_constant_column_ties_every_rotation():
    # The review's case: forward returns of 1e-323 on every finite bar, so every rotation's mean
    # IS the observed one. The raw transform ran on values of a few bits, its means came out as
    # rounding noise, and rot_p read 0.05 — the floor — where the same mask over unit-scale
    # returns read 1.0. The unit frame runs it on 0.5.
    T = 20
    mask = np.zeros(T)
    mask[[16, 18, 19]] = 1.0
    for c in (1e-323, 5e-324, 1e-310, 1.0, 1.7e308):
        fwd = np.full(T, c)
        fwd[18:] = np.nan
        read = _one_cell_read(mask, fwd)
        assert read["rot_p"] == 1.0 and read["rot_n_null"] == T - 1, c


def test_rotation_den_is_an_exact_count():
    # den counts coincidences of the rotated mask with finite forward bars: an integer the FFT
    # only approximates, returned rounded so it divides exactly.
    rng = np.random.RandomState(8)
    fwd = rng.normal(0.0, 0.01, 2999)
    fwd[rng.rand(2999) < 0.05] = np.nan
    mask = (rng.rand(2999) < 0.3).astype(float)
    _num, den = _rotation_num_den(mask, *_fwd_ffts(fwd), fwd.shape[0])
    np.testing.assert_array_equal(den, np.round(den))
    finite = np.isfinite(fwd)
    for tau in (0, 1, 777, 2998):
        assert den[tau] == float(np.sum(np.roll(mask > 0, tau) & finite))


@pytest.mark.parametrize("c", [0.1, float(np.log(2.0)), 0.3, 0.013])
def test_constant_return_pool_rotation_p_is_one(c):
    # Every rotated mask of a constant-return pool selects the SAME conditional mean: the whole
    # null ties the observed value and rot_p is exactly 1. The FFT's rounding residue (~1e-16)
    # used to break those ties at random — a constant pool read rot_p as low as 0.016.
    rng = np.random.RandomState(4)
    T = 61
    fwd = np.full(T, c)
    for _ in range(5):
        mask = np.zeros(T)
        mask[rng.choice(T, int(rng.randint(5, 30)), replace=False)] = 1.0
        cells = [{"key": ("k", "t"), "mask_col": mask, "fwd_col": fwd, "h": 1}]
        read = reliability_summary(cells, T, ["t"])["per_cell"][("k", "t")]
        assert read["rot_p"] == 1.0 and read["rot_n_null"] == T - 1
        pooled = pooled_reliability_summary(
            [*cells, {"key": ("k", "u"), "mask_col": np.roll(mask, 3), "fwd_col": fwd, "h": 1}], T
        )["per_cell"][("k",)]
        assert pooled["rot_p"] == 1.0 and pooled["rot_n_null"] == T - 1


def _one_cell_read(mask, fwd):
    T = fwd.shape[0]
    cells = [{"key": ("k", "t"), "mask_col": mask, "fwd_col": fwd, "h": 1}]
    return reliability_summary(cells, T, ["t"])["per_cell"][("k", "t")]


def _exact_corr(members, weights=None):
    """Exact INTEGER circular cross-correlations: ``den[tau]`` = the finite hits of every member's
    rotated mask, ``num[tau]`` the sum of integer ``weights[g]`` over those hits."""
    T = members[0][1].shape[0]
    den = np.zeros(T, dtype=np.int64)
    num = np.zeros(T, dtype=np.int64)
    for g, (m, f) in enumerate(members):
        v2 = np.tile(np.isfinite(f).astype(np.int64), 2)
        w = None if weights is None else np.tile(weights[g], 2)
        for i in np.flatnonzero(m > 0):
            den += v2[i : i + T]
            if w is not None:
                num += w[i : i + T]
    return num, den


def _random_mask(rng, T, k):
    m = np.zeros(T)
    m[rng.choice(T, k, replace=False)] = 1.0
    return m


@pytest.mark.parametrize("c", [0.1, 0.013, 0.3])
def test_holed_constant_pool_rotation_p_is_one(c):
    # Constant finite returns between NaN holes: den differs per shift, so each shift's mean is
    # num/den with its OWN rounding — still an exact tie with the observed c. Exact counts alone
    # (rint) do not restore these ties; the tie band does.
    rng = np.random.RandomState(12)
    T = 61
    checked = 0
    for p_nan in (0.05, 0.3, 0.7):
        for k in (3, 30):
            fwd = np.full(T, c)
            fwd[rng.rand(T) < p_nan] = np.nan
            mask = _random_mask(rng, T, k)
            _num, den = _exact_corr([(mask, fwd)])
            if den[0] == 0:
                continue
            read = _one_cell_read(mask, fwd)
            assert read["rot_p"] == 1.0, (p_nan, k, read["rot_p"])
            assert read["rot_n_null"] == int(np.sum(den[1:] > 0))
            checked += 1
    assert checked >= 4


@pytest.mark.parametrize(("c", "k"), [(0.1, 5), (0.1, 100), (1e-4, 10005), (12.5, 10005)])
def test_long_constant_pool_rotation_p_is_one(c, k):
    # The FFT's error grows with T and the firing count: a long PRIME-length series (pocketfft's
    # Bluestein path, less accurate than a power of two) under a sparse and a near-full mask.
    rng = np.random.RandomState(11)
    T = 10007
    read = _one_cell_read(_random_mask(rng, T, k), np.full(T, c))
    assert read["rot_p"] == 1.0 and read["rot_n_null"] == T - 1


def test_quantized_pool_rotation_p_is_the_exact_tie_aware_p():
    # A `diff` outcome over a step series (a policy rate): exact 0.0 almost everywhere, a few
    # ±0.25 / +0.5 — a tie-heavy pool whose values are exact multiples of 0.25, so the oracle
    # compares integer sums (mean_tau >= mean_0  <=>  q_tau·den_0 >= q_0·den_tau).
    rng = np.random.RandomState(16)
    T = 500
    u = rng.rand(T)
    fwd = np.where(u < 0.0025, 0.25, np.where(u < 0.005, -0.25, 0.0))
    fwd[rng.rand(T) < 0.01] = 0.5
    q = np.rint(fwd * 4.0).astype(np.int64)
    for k in (10, 25, 125):
        mask = _random_mask(rng, T, k)
        num, den = _exact_corr([(mask, fwd)], [q])
        want = (1.0 + int(np.sum(num[1:] * den[0] >= num[0] * den[1:]))) / T
        read = _one_cell_read(mask, fwd)
        assert (read["rot_p"], read["rot_n_null"]) == (want, T - 1), (k, read["rot_p"], want)


def test_pooled_constant_pool_of_many_holed_members_rotation_p_is_one():
    # The pooled common-shift null sums num/den over MANY members, each with its own holes: the
    # members' bands must ADD across the group for every pooled mean to tie the observed c.
    rng = np.random.RandomState(13)
    T, G, c = 61, 1000, 0.1
    members = []
    for _ in range(G):
        fwd = np.full(T, c)
        fwd[rng.rand(T) < 0.3] = np.nan
        members.append((_random_mask(rng, T, int(rng.randint(2, 20))), fwd))
    cells = [
        {"key": ("k", f"t{g}"), "mask_col": m, "fwd_col": f, "h": 1}
        for g, (m, f) in enumerate(members)
    ]
    read = pooled_reliability_summary(cells, T)["per_cell"][("k",)]
    _num, den = _exact_corr(members)
    assert read["rot_p"] == 1.0 and read["rot_n_null"] == int(np.sum(den[1:] > 0))


def test_rotation_tie_band_does_not_swallow_a_resolvable_gap():
    # One firing bar lifted by delta: the k shifts landing a firing on it (the observed among
    # them) tie exactly, every other shift sits delta/k BELOW — a gap ~25x the band and ~1000x
    # the FFT's actual error. Exact p = k/T: a band inflated past the gap would read 1, a missing
    # band would break the k-1 null ties.
    rng = np.random.RandomState(5)
    T, k, c = 512, 16, 0.01
    fire = rng.choice(T, k, replace=False)
    mask = np.zeros(T)
    mask[fire] = 1.0
    fwd = np.full(T, c)
    fwd[fire[0]] = c + c * 2.0**-36
    read = _one_cell_read(mask, fwd)
    assert read["rot_p"] == k / T and read["rot_n_null"] == T - 1


def test_rotation_unit_frame_bounds_the_column_and_keeps_its_norm():
    # The transform runs on fwd · 2**-e, e the exponent of its largest finite magnitude, so that
    # magnitude lies in [0.5, 1): the 2-norm the tie band scales with is finite and positive at
    # every scale (sqrt(x·x) of the raw column overflowed past ~1.3e154 and vanished below
    # ~1e-162), NaN bars stay NaN, and a column of zeros sets no frame.
    from seikan.analysis._rotation import _ffts_for

    for c in (5e-324, 1e-323, 1e-170, 0.01, 1e200, 1.7e308):
        fwd = np.full(100, c)
        fwd[7] = np.nan
        _f, valid, f_norm, e = _ffts_for(fwd, {})
        m, e_c = math.frexp(c)
        assert e == e_c and f_norm == pytest.approx(m * math.sqrt(99), rel=1e-15), c
        assert valid[0] == 99.0
    assert _ffts_for(np.zeros(5), {})[2:] == (0.0, 0)
    assert _ffts_for(np.full(5, np.nan), {})[2:] == (0.0, 0)


@pytest.mark.parametrize("c", [0.1 * 1e-170, 1e200])
def test_rotation_tie_band_survives_float_extremes_on_the_read_path(c):
    # The read path must take the scale-safe norm: a plain sqrt(x·x) underflows to a zero band
    # (~1e-171) or overflows to the exact fallback (~1e200), and the residue gets its coin flips
    # back. Many masks, because one mask may leave every computed mean bit-identical — and then
    # even a band-less read says 1.0.
    rng = np.random.RandomState(0)
    T = 61
    fwd = np.full(T, c)
    for _ in range(60):
        read = _one_cell_read(_random_mask(rng, T, int(rng.randint(3, 50))), fwd)
        assert read["rot_p"] == 1.0 and read["rot_n_null"] == T - 1, c


def test_rotation_p_is_scale_invariant_at_float_extremes():
    # Exact power-of-two rescalings of dispersed returns move no comparison: an infinite band
    # would count every shift as a tie (rot_p 1.0), a zero one would change nothing here.
    rng = np.random.RandomState(9)
    T = 61
    mask = _random_mask(rng, T, 20)
    fwd = rng.normal(0.001, 0.01, T)
    base = _one_cell_read(mask, fwd)["rot_p"]
    assert base < 0.9
    for s in (2.0**-560, 2.0**520, 2.0**660):  # ~1e-169, ~3e156, ~5e198
        assert _one_cell_read(mask, fwd * s)["rot_p"] == base, s


def test_rotation_tie_band_leaves_real_dispersion_untouched():
    # On returns with genuine dispersion the band (~1e-15 here) sits far below the null's spacing:
    # rot_p equals the exact-comparison tail p of the brute-force null.
    rng = np.random.RandomState(9)
    T = 400
    fwd = rng.normal(0.001, 0.01, T)
    mask = np.zeros(T)
    mask[rng.choice(T, 40, replace=False)] = 1.0
    brute = np.array([fwd[np.roll(mask > 0, tau)].mean() for tau in range(T)])
    read = reliability_summary(
        [{"key": ("k", "t"), "mask_col": mask, "fwd_col": fwd, "h": 1}], T, ["t"]
    )["per_cell"][("k", "t")]
    assert read["rot_p"] == _tail_p(brute[0], brute[1:])[0]


def _rotation_means(mask, fwd):
    """Test-side composition of the production spectra pipeline: the conditional mean under
    every circular shift, ``num[tau] / den[tau]`` off the two memoized cross-correlation
    channels (the production path never forms the division per cell — it sums the channels
    across members first)."""
    fwd = np.asarray(fwd, dtype=float)
    num, den = _rotation_num_den(mask, *_fwd_ffts(fwd), fwd.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.where(den > 0.5, num / den, np.nan)
    return float(means[0]), means


def test_rotation_means_observed_is_identity_shift():
    rng = np.random.RandomState(2)
    fwd = rng.normal(0, 0.01, 100)
    mask = (rng.rand(100) < 0.3).astype(float)
    observed, means = _rotation_means(mask, fwd)
    assert observed == pytest.approx(fwd[mask > 0].mean())
    assert means.shape == (100,) and means[0] == pytest.approx(observed)


def test_reliability_summary_returns_only_per_cell_nominal_reads():
    # The shape: a dict of independent per-cell reads plus the shift count, and NOTHING
    # describing the grid as a whole. No fw_p, no best_combo, no deflation, no stepdown, no FDR —
    # a search correction computed inside one run only ever sees that run's grid, so multiplicity
    # is the caller's to price (against n_hypotheses_attempted).
    rng = np.random.RandomState(0)
    T = 300
    fwd = rng.normal(0.0, 0.01, T)
    cells = []
    for k in range(3):
        m = np.zeros(T)
        m[rng.choice(T, 30, replace=False)] = 1.0
        cells.append({"key": (k, "t"), "mask_col": m, "fwd_col": fwd, "h": 1})
    rel = reliability_summary(cells, T, ["t"])
    assert set(rel) == {"per_cell", "n_shifts"}
    assert set(rel["per_cell"]) == {(0, "t"), (1, "t"), (2, "t")}
    for read in rel["per_cell"].values():
        assert set(read) == {"rot_p", "rot_n_null", "t_hac", "hac_se", "n_nonoverlap"}
        assert 0.0 <= read["rot_p"] <= 1.0
        assert isinstance(read["n_nonoverlap"], int)


def test_reliability_summary_empty_grid_is_the_zero_shape():
    assert reliability_summary([], 400, ["t"]) == {"per_cell": {}, "n_shifts": 0}
    # too short to rotate against: same total, no exception
    m = np.array([1.0, 0.0])
    fwd = np.array([0.01, 0.02])
    assert reliability_summary(
        [{"key": ("c", "t"), "mask_col": m, "fwd_col": fwd, "h": 1}], 2, ["t"]
    ) == {"per_cell": {}, "n_shifts": 0}


def test_rotation_null_detects_edge_and_clears_noise():
    # A signal aligned with high forward returns is significant against its OWN rotation null; a
    # random signal of the same count is not. (The forward-return series is fixed; only the firing
    # mask rotates.) Each cell is its own combo with a single target "t".
    rng = np.random.RandomState(0)
    T = 400
    fwd = rng.normal(0.0, 0.01, T)
    edge = (fwd > np.quantile(fwd, 0.8)).astype(float)
    noise = np.zeros(T)
    noise[rng.choice(T, int(edge.sum()), replace=False)] = 1.0
    cells = [
        {"key": ("edge", "t"), "mask_col": edge, "fwd_col": fwd, "h": 1},
        {"key": ("noise", "t"), "mask_col": noise, "fwd_col": fwd, "h": 1},
    ]
    rel = reliability_summary(cells, T, ["t"])
    assert rel["per_cell"][("edge", "t")]["rot_p"] < 0.05
    assert rel["per_cell"][("edge", "t")]["rot_p"] < rel["per_cell"][("noise", "t")]["rot_p"]
    # Resolution transparency: the shift count actually used is reported (min achievable
    # p = 1/(1+n_shifts)); the null always uses every non-identity shift — T-1 = 399.
    assert rel["n_shifts"] == 399


def test_rotation_grid_uses_every_shift_no_periodic_aliasing():
    # THE aliasing trap the every-shift null avoids: with a period-5 signal at T=5000, an
    # evenly-spaced 1000-shift grid has residues mod 5 of [0,333,333,333,1] — the aligned phase
    # is nearly absent from the null, so a purely periodic artifact certifies at ~1/1000. All
    # 4999 shifts are the full randomization set: ~1/5 of them realign the mask with the "good"
    # bars, so rot_p sits near 0.2.
    T = 5000
    mask = (np.arange(T) % 5 == 0).astype(float)
    fwd = np.where(np.arange(T) % 5 == 0, 1.0, -0.25)  # circular-stationary, period 5
    rel = reliability_summary(
        [{"key": ("periodic", "t"), "mask_col": mask, "fwd_col": fwd, "h": 1}], T, ["t"]
    )
    assert rel["n_shifts"] == T - 1
    rot_p = rel["per_cell"][("periodic", "t")]["rot_p"]
    assert rot_p > 0.15  # the periodic null is NOT significant once every phase is priced


def test_per_cell_reads_do_not_depend_on_the_rest_of_the_grid():
    # THE independence invariant: every cell is measured on its OWN mask against its OWN null,
    # so adding hypotheses to the grid cannot move a cell's numbers. A max-over-combos search null
    # would move every one of them, which is exactly why such a grid cannot be read cell by cell.
    rng = np.random.RandomState(4)
    T = 400
    fwd = rng.normal(0.0, 0.01, T)
    lone = np.zeros(T)
    lone[rng.choice(T, 40, replace=False)] = 1.0
    cells = [{"key": ("a", "t"), "mask_col": lone, "fwd_col": fwd, "h": 5}]
    alone = reliability_summary(cells, T, ["t"])["per_cell"][("a", "t")]
    for k in range(10):
        m = np.zeros(T)
        m[rng.choice(T, 40, replace=False)] = 1.0
        cells.append({"key": (f"n{k}", "t"), "mask_col": m, "fwd_col": fwd, "h": 5})
    crowded = reliability_summary(cells, T, ["t"])["per_cell"][("a", "t")]
    assert crowded == alone


def test_per_cell_n_nonoverlap_and_hac_match_direct_recomputation():
    # The per-cell reads ARE the documented kernels applied to that cell's closed rows: no
    # studentization, no pooling, no adjustment sits between them and the report.
    rng = np.random.RandomState(6)
    T, h = 500, 10
    fwd = rng.normal(0.001, 0.01, T)
    fwd[-h:] = np.nan  # trailing windows run past the data end
    mask = np.zeros(T)
    mask[rng.choice(T, 60, replace=False)] = 1.0
    rel = reliability_summary(
        [{"key": ("c", "t"), "mask_col": mask, "fwd_col": fwd, "h": h}], T, ["t"]
    )
    read = rel["per_cell"][("c", "t")]
    closed = np.flatnonzero((mask > 0) & np.isfinite(fwd))
    assert read["n_nonoverlap"] == nonoverlap_count(closed, h)
    t_ref, se_ref = newey_west_mean(fwd[closed], closed, h)
    assert read["t_hac"] == pytest.approx(t_ref)
    assert read["hac_se"] == pytest.approx(se_ref)
    # the defined-shift count is the cell's own resolution: a 60-firing mask over a 10-bar NaN
    # tail never lands EVERY firing in the tail, so every non-identity shift is defined
    assert read["rot_n_null"] == T - 1


# ---- pooled_reliability_summary: the basket-mode pooled read --------------------


def _member_cells(rng, T=400, h=5, n_fire=40, combo="c", targets=("A", "B")):
    """Basket members of ONE combo: per-target fwd columns (NaN tails) and firing masks."""
    cells = []
    for tgt in targets:
        fwd = rng.normal(0.001, 0.01, T)
        fwd[-h:] = np.nan  # trailing windows past the data end stay censored
        mask = np.zeros(T)
        mask[rng.choice(T - h, n_fire, replace=False)] = 1.0
        cells.append({"key": (combo, tgt), "mask_col": mask, "fwd_col": fwd, "h": h})
    return cells


def test_pooled_reliability_matches_direct_kernels_on_concatenated_rows():
    # The pooled reads ARE the documented kernels applied to the concatenated member rows —
    # no reweighting, no averaging of per-member statistics, nothing in between.
    rng = np.random.RandomState(21)
    T, h = 400, 5
    cells = _member_cells(rng, T=T, h=h)
    out = pooled_reliability_summary(cells, T)
    assert set(out) == {"per_cell", "n_shifts"} and out["n_shifts"] == T - 1
    read = out["per_cell"][("c",)]
    assert set(read) == {"rot_p", "rot_n_null", "t_hac", "hac_se", "n_nonoverlap"}
    bars_parts, rets_parts = [], []
    for c in cells:  # cells-list order — the tie order the runner fixes as declaration order
        fwd = np.asarray(c["fwd_col"], dtype=float)
        t = np.flatnonzero((np.asarray(c["mask_col"]) > 0) & np.isfinite(fwd))
        bars_parts.append(t)
        rets_parts.append(fwd[t])
    bars = np.concatenate(bars_parts)
    rets = np.concatenate(rets_parts)
    n_nonoverlap = nonoverlap_count(bars, h)
    t_ref, se_ref = newey_west_mean(rets, bars, h)
    assert read["n_nonoverlap"] == n_nonoverlap
    assert read["t_hac"] == pytest.approx(t_ref, rel=1e-12)
    assert read["hac_se"] == pytest.approx(se_ref, rel=1e-12)
    assert 0.0 <= read["rot_p"] <= 1.0


def test_pooled_same_bar_cross_member_firings_collapse_to_one_n_nonoverlap():
    # Two members, one market event: ONE independent window, never two — the greedy kernel
    # collapses duplicate bars automatically, which is why n_nonoverlap keeps one meaning
    # engine-wide.
    T, h = 60, 5
    fwd_a = np.full(T, 0.01)
    fwd_b = np.full(T, 0.02)
    m = np.zeros(T)
    m[10] = 1.0
    cells = [
        {"key": ("c", "A"), "mask_col": m, "fwd_col": fwd_a, "h": h},
        {"key": ("c", "B"), "mask_col": m.copy(), "fwd_col": fwd_b, "h": h},
    ]
    assert pooled_reliability_summary(cells, T)["per_cell"][("c",)]["n_nonoverlap"] == 1
    m2 = np.zeros(T)
    m2[10] = 1.0
    m2[30] = 1.0
    cells2 = [
        {"key": ("c", "A"), "mask_col": m2, "fwd_col": fwd_a, "h": h},
        {"key": ("c", "B"), "mask_col": m, "fwd_col": fwd_b, "h": h},
    ]
    # bars {10, 30, 10} → the duplicate bar-10 firing adds nothing
    assert pooled_reliability_summary(cells2, T)["per_cell"][("c",)]["n_nonoverlap"] == 2


def test_pooled_same_bar_hac_pair_enters_at_bartlett_weight_one():
    # Three pooled rows: members A and B both fire at bar 0 (d=0 → weight exactly 1), A fires
    # again at bar 100 (d=100 ≥ h → weight 0). Long-run variance = gamma_0 + 2·1·dev_i·dev_j/n
    # — the same-bar cross term at FULL weight: one market move seen through two members is one
    # cluster, priced at full covariance.
    T, h = 200, 5
    a0, a1, b0 = 0.02, -0.01, 0.03
    fwd_a = np.zeros(T)
    fwd_a[0], fwd_a[100] = a0, a1
    fwd_b = np.zeros(T)
    fwd_b[0] = b0
    ma = np.zeros(T)
    ma[0], ma[100] = 1.0, 1.0
    mb = np.zeros(T)
    mb[0] = 1.0
    cells = [
        {"key": ("c", "A"), "mask_col": ma, "fwd_col": fwd_a, "h": h},
        {"key": ("c", "B"), "mask_col": mb, "fwd_col": fwd_b, "h": h},
    ]
    read = pooled_reliability_summary(cells, T)["per_cell"][("c",)]
    rets = np.array([a0, b0, a1])  # canonical (bar, ret) order: bar-0 ties sort by return
    mean = rets.mean()
    dev = rets - mean
    n = 3
    s = float(dev @ dev) / (n - 1) + 2.0 * 1.0 * dev[0] * dev[1] / (n - 1)
    se = np.sqrt(s / n)
    assert read["n_nonoverlap"] == 2  # bars {0, 100}: the same-bar duplicate collapses
    assert read["hac_se"] == pytest.approx(se, rel=1e-12)
    assert read["t_hac"] == pytest.approx(mean / se, rel=1e-12)


def test_pooled_common_shift_rotation_matches_brute_force_roll_of_every_member():
    """THE common-shift definition: one tau rotates EVERY member's mask as a block, and the
    pooled conditional mean is the ratio of pooled sums — never a mean of per-member means.

    The doctrine the common shift encodes: rotating every mask by the SAME tau rolls the
    cross-sectional per-bar firing count as a block, so a rank signal's "exactly k members fire
    together" pattern survives every shift — independent per-member shifts would scramble it
    into masks the signal could never emit. The np.roll brute force below IS that block
    rotation, so matching it at every tau is what pins the property in engine code (asserting it
    over bare np.roll calls on both sides would exercise no engine code at all — a pure-numpy
    tautology)."""
    rng = np.random.RandomState(31)
    T, h = 97, 4
    cells = _member_cells(rng, T=T, h=h, n_fire=20)
    masks = [np.asarray(c["mask_col"]) > 0 for c in cells]
    fwds = [np.asarray(c["fwd_col"], dtype=float) for c in cells]
    brute = np.empty(T)
    for tau in range(T):
        num = 0.0
        den = 0.0
        for m, f in zip(masks, fwds, strict=True):
            rolled = np.roll(m, tau)
            fin = np.isfinite(f)
            num += float(np.sum(np.where(rolled & fin, f, 0.0)))
            den += float(np.sum(rolled & fin))
        brute[tau] = num / den if den > 0.5 else float("nan")
    # helper level: the pooled num/den arrays are exactly the summed member cross-correlations
    num_sum = np.zeros(T)
    den_sum = np.zeros(T)
    for m, f in zip(masks, fwds, strict=True):
        num_c, den_c = _rotation_num_den(m, *_fwd_ffts(f), T)
        num_sum += num_c
        den_sum += den_c
    means = np.where(den_sum > 0.5, num_sum / den_sum, np.nan)
    np.testing.assert_allclose(means, brute, rtol=1e-9, atol=1e-12)
    # end to end: rot_p is the tail p of the brute-force common-shift null
    read = pooled_reliability_summary(cells, T)["per_cell"][("c",)]
    p_brute, n_null_brute = _tail_p(brute[0], brute[1:])
    assert read["rot_p"] == pytest.approx(p_brute) and read["rot_n_null"] == n_null_brute


def test_pooled_g1_reduces_exactly_to_the_per_target_read():
    # A one-member basket group IS its member: same spectra, same division, same kernels on the
    # same rows — EXACT equality, not approximation.
    rng = np.random.RandomState(41)
    T, h = 300, 5
    fwd = rng.normal(0.0, 0.01, T)
    fwd[-h:] = np.nan
    mask = np.zeros(T)
    mask[rng.choice(T - h, 35, replace=False)] = 1.0
    cells = [{"key": ("k", "t"), "mask_col": mask, "fwd_col": fwd, "h": h}]
    per_target = reliability_summary(cells, T, ["t"])["per_cell"][("k", "t")]
    pooled = pooled_reliability_summary(cells, T)["per_cell"][("k",)]
    assert pooled == per_target


def test_pooled_groups_are_independent_of_the_rest_of_the_grid():
    # The same independence invariant as the per-target pass: grouping is structural, so adding
    # a combo to the grid changes no other combo's pooled numbers.
    rng = np.random.RandomState(43)
    T = 250
    cells_a = _member_cells(rng, T=T, h=5, combo="a")
    alone = pooled_reliability_summary(cells_a, T)["per_cell"][("a",)]
    cells_b = _member_cells(rng, T=T, h=8, n_fire=25, combo="b")
    both = pooled_reliability_summary(cells_a + cells_b, T)
    assert both["per_cell"][("a",)] == alone
    assert set(both["per_cell"]) == {("a",), ("b",)}


def test_pooled_reliability_empty_and_short_guards_mirror_reliability_summary():
    assert pooled_reliability_summary([], 400) == {"per_cell": {}, "n_shifts": 0}
    m = np.array([1.0, 0.0])
    fwd = np.array([0.01, 0.02])
    assert pooled_reliability_summary(
        [{"key": ("c", "t"), "mask_col": m, "fwd_col": fwd, "h": 1}], 2
    ) == {"per_cell": {}, "n_shifts": 0}


def test_pooled_rotation_over_opposite_members_at_the_ceiling_ties_every_shift():
    # Two members firing on one bar whose forward returns are +1.7e308 and -1.7e308: each raw
    # transform overflowed, to +inf and to -inf, and the common-shift null summed them — inf − inf,
    # a null read. In the unit frame each numerator is finite and the pair cancels exactly at
    # every shift: every rotation's pooled mean is the observed 0, a tie — rot_p 1.0, silently.
    T = 16
    m = np.zeros(T)
    m[3] = 1.0
    cells = []
    for tgt, v in (("A", 1.7e308), ("B", -1.7e308)):
        fwd = np.zeros(T)
        fwd[3] = v
        cells.append({"key": ("c", tgt), "mask_col": m, "fwd_col": fwd, "h": 1})
    read = pooled_reliability_summary(cells, T)["per_cell"][("c",)]
    assert read["rot_p"] == 1.0 and read["rot_n_null"] == T - 1 and read["n_nonoverlap"] == 1


def test_pooled_rotation_frames_members_of_different_scales_together():
    # The common frame is the largest member's — a member of zeros sets none. A subnormal member
    # (small integer multiples of 2**-1074, ties galore) pooled beside a member of zeros reads the
    # rot_p of the same pool at unit scale; framed at the zero member's exponent 0 instead, its
    # rotation means ran at their raw subnormal scale, where distinct means round together.
    rng = np.random.RandomState(4)
    T = 40
    for _ in range(20):
        masks = [(rng.rand(T) < 0.3).astype(float) for _ in range(2)]
        ints = rng.randint(1, 9, T).astype(float)
        ints[rng.rand(T) < 0.2] = np.nan
        reads = []
        for column in (np.ldexp(ints, -1074), ints):
            cells = [
                {"key": ("c", "A"), "mask_col": masks[0], "fwd_col": column, "h": 1},
                {"key": ("c", "B"), "mask_col": masks[1], "fwd_col": np.zeros(T), "h": 1},
            ]
            reads.append(pooled_reliability_summary(cells, T)["per_cell"][("c",)]["rot_p"])
        assert reads[0] == reads[1]


@pytest.mark.parametrize("k", [-1008, -600, -7, 3, 600, 1021])
def test_rotation_p_is_bit_identical_under_power_of_two_rescaling(k):
    # The FFT is bit-exactly equivariant under a power-of-two scaling that keeps its arithmetic in
    # range, and the unit frame removes the scale before the transform: every rescaling of a
    # column that stays exactly representable reads the same rot_p — dispersed returns, tick
    # ties and constants alike, per target and pooled.
    rng = np.random.RandomState(17)
    T = 97
    columns = [
        np.clip(rng.normal(0.001, 0.01, T), 1e-4, None) * rng.choice([-1.0, 1.0], T),
        np.round(rng.normal(0.0, 0.01, T) * 400) / 400 + 0.25,
        np.full(T, 0.013),
    ]
    masks = [_random_mask(rng, T, n) for n in (5, 30, 70)]
    for f in columns:
        f = f.copy()
        f[-3:] = np.nan
        for mask in masks:
            assert _one_cell_read(mask, np.ldexp(f, k))["rot_p"] == _one_cell_read(mask, f)["rot_p"]
    cells = [
        {"key": ("c", str(g)), "mask_col": masks[g], "fwd_col": columns[g], "h": 1}
        for g in range(3)
    ]
    scaled = [{**c, "fwd_col": np.ldexp(c["fwd_col"], k)} for c in cells]
    base = pooled_reliability_summary(cells, T)["per_cell"][("c",)]
    assert pooled_reliability_summary(scaled, T)["per_cell"][("c",)]["rot_p"] == base["rot_p"]


# ---- CSCV → PBO (grid-level evidence, attached to no hypothesis) ------------


def _pbo_cells(rng, T, n_combos, edge_combo=None, n_fire=60, h=1):
    """Synthetic single-target cells sharing one fwd series (like the real engine)."""
    fwd = rng.normal(0.0, 0.01, T)
    signal_bars = rng.choice(T - h - 1, n_fire, replace=False)
    if edge_combo is not None:
        fwd[signal_bars] += 0.02  # bars the edge combo fires on are genuinely shifted
    cells = []
    for k in range(n_combos):
        m = np.zeros(T)
        if edge_combo is not None and k == edge_combo:
            m[signal_bars] = 1.0
        else:
            m[rng.choice(T - h - 1, n_fire, replace=False)] = 1.0
        cells.append({"key": (k, "t"), "mask_col": m, "fwd_col": fwd, "h": h})
    return cells


def test_cscv_pbo_noise_grid_is_a_coin_flip():
    rng = np.random.RandomState(5)
    cells = _pbo_cells(rng, 800, 12)
    out = cscv_pbo(cells, 800, ["t"], n_combos_declared=12)
    assert out["reason"] is None and out["n_splits"] == 70
    # pure noise: the train-winner's OOS rank is uniform → PBO near one half
    assert 0.25 <= out["pbo"] <= 0.75


def test_cscv_pbo_planted_edge_travels():
    rng = np.random.RandomState(6)
    cells = _pbo_cells(rng, 800, 12, edge_combo=0)
    out = cscv_pbo(cells, 800, ["t"], n_combos_declared=12)
    assert out["pbo"] <= 0.1  # the winner keeps winning out of sample
    assert out["prob_oos_loss"] <= 0.1
    assert out["lambda_mean"] > 0


def test_cscv_pbo_single_combo_and_empty_guards():
    rng = np.random.RandomState(7)
    cells = _pbo_cells(rng, 400, 1)
    out = cscv_pbo(cells, 400, ["t"], n_combos_declared=1)
    assert out["pbo"] is None and out["reason"] == "single_combo"
    out = cscv_pbo([], 400, ["t"], n_combos_declared=0)
    assert out["pbo"] is None and out["reason"] == "insufficient_data"


def test_cscv_pbo_six_bar_sample_declines_beside_the_real_candidate_counts():
    # The review's scenario: six bars, h=1, two distinct candidates that both fired. No block
    # partition fits six bars (S=4 needs eight), so the block declines — but the population is a
    # property of the cells, not of the partition: 2 distinct / 2 scoreable / 2 declared. The
    # pre-fix block reported 0 / 0 / 2, a ledger claiming no candidate ever existed.
    T = 6
    fwd = np.array([0.01, -0.02, 0.03, 0.01, np.nan, np.nan])
    cells = []
    for k in range(2):
        m = np.zeros(T)
        m[[k, k + 2]] = 1.0  # distinct masks per combo: two candidates, not one cloned
        cells.append({"key": (k, "t"), "mask_col": m, "fwd_col": fwd, "h": 1})
    out = cscv_pbo(cells, T, ["t"], n_combos_declared=2)
    assert out["pbo"] is None and out["reason"] == "insufficient_data"
    assert (out["n_combos"], out["n_combos_scoreable"], out["n_combos_declared"]) == (2, 2, 2)
    # Nothing was split: no partition was offered, so none was attempted.
    assert (out["n_splits"], out["n_splits_attempted"], out["n_candidates_min"]) == (0, 0, 0)
    assert out["blocks"] == 8


def _thin_ledger_cells(T):
    """A two-target conjunction grid whose population ledger cannot be 2/2/2 by coincidence:
    combos 0 and 1 are distinct, combo 2 is a byte-identical clone of combo 0, and combo 3 has no
    B pool (inadmissible under conjunction) — 2 distinct of 3 scoreable, declared as 5 (one more
    combo never fired, so it never reaches CSCV). Every firing sits in the first three bars."""
    fwd = np.linspace(0.01, 0.02, T)

    def _mask(*bars):
        m = np.zeros(T)
        m[list(bars)] = 1.0
        return m

    def _cell(k, tgt, mask):
        return {"key": (k, tgt), "mask_col": mask, "fwd_col": fwd, "h": 1}

    return [
        _cell(0, "A", _mask(0)),
        _cell(0, "B", _mask(1)),
        _cell(1, "A", _mask(1)),
        _cell(1, "B", _mask(0)),
        _cell(2, "A", _mask(0)),
        _cell(2, "B", _mask(1)),
        _cell(3, "A", _mask(0, 2)),
    ]


@pytest.mark.parametrize(
    ("T", "blocks", "n_splits_attempted"),
    [(3, 8, 0), (6, 8, 0), (8, 4, 6)],
    ids=["under_four_bars", "no_partition_fits", "every_split_purged"],
)
def test_cscv_pbo_every_declined_path_carries_the_population_ledger(T, blocks, n_splits_attempted):
    # The three ways a block with candidates declines — a sample under 4 bars, a sample no
    # partition fits (4-7 bars), and a partition whose every split purged away (S=4 over 8 bars:
    # two-bar blocks cannot hold an off+h=2 window) — must all carry the SAME established
    # population: the ledger reads the cells, never the sample length.
    out = cscv_pbo(_thin_ledger_cells(T), T, ["A", "B"], n_combos_declared=5)
    assert out["pbo"] is None and out["reason"] == "insufficient_data"
    assert (out["n_combos"], out["n_combos_scoreable"], out["n_combos_declared"]) == (2, 3, 5)
    assert out["n_splits"] == 0 and out["n_candidates_min"] == 0
    assert out["n_splits_attempted"] == n_splits_attempted and out["blocks"] == blocks
    for key in ("lambda_mean", "oos_degradation_slope", "prob_oos_loss"):
        assert np.isnan(out[key])
    assert out["oos_degradation_slope_reason"] is None


def test_cscv_pbo_purges_boundary_crossing_windows():
    # 80 bars; the adaptive S=8→6→4 fallback tries block widths 10, ~13.3, 20 in turn. A window of
    # off+h=26 bars exceeds even the widest candidate block (20), so EVERY firing purges away at
    # every adaptive S — insufficient data, never a leaked score, regardless of which S is tried.
    T, h = 80, 25
    fwd = np.random.RandomState(8).normal(0, 0.01, T)
    cells = []
    for k in range(2):
        m = np.zeros(T)
        # Distinct masks per combo (offset by k): identical masks over one fwd are ONE candidate
        # and would refuse as single_combo before any purge arithmetic runs.
        m[5 + k + 10 * np.arange(7)] = 1.0
        cells.append({"key": (k, "t"), "mask_col": m, "fwd_col": fwd, "h": h})
    out = cscv_pbo(cells, T, ["t"], n_combos_declared=2)
    assert out["pbo"] is None and out["reason"] == "insufficient_data"


def test_cscv_pbo_falls_back_to_fewer_blocks():
    # T=96 → S=8 blocks of width 12, S=6 of width 16. With off+h=12, every firing purges at S=8
    # (window span equals block width ⇒ no t satisfies t+off+h < block_end). At S=6 the wider
    # blocks keep early-in-block firings, so the adaptive fallback scores and reports blocks=6.
    T, h, off = 96, 11, 1
    assert off + h == T // 8
    rng = np.random.RandomState(11)
    fwd = rng.normal(0.002, 0.01, T)
    # All combos fire in the first 3 bars of every 16-bar region (survive S=6 width-16;
    # purge entirely at S=8 width-12). Distinct early slots so each combo has its own pool.
    cells = []
    for k, offsets in enumerate(((0, 1), (1, 2), (0, 2))):
        m = np.zeros(T)
        for b in range(0, T, 16):
            for o in offsets:
                m[b + o] = 1.0
        cells.append({"key": (k, "t"), "mask_col": m, "fwd_col": fwd, "h": h})
    # Plant a mild edge on combo 0's fire bars only.
    edge = cells[0]["mask_col"] > 0
    fwd = fwd.copy()
    fwd[edge] += 0.02
    cells[0]["fwd_col"] = fwd
    for c in cells[1:]:
        c["fwd_col"] = fwd
    out = cscv_pbo(cells, T, ["t"], n_combos_declared=3)
    assert out["reason"] is None
    assert out["blocks"] == 6
    assert out["n_combos"] >= 2
    assert isinstance(out["pbo"], float) and 0.0 <= out["pbo"] <= 1.0
    assert out["n_splits"] >= 2


# ---- CSCV target modes: the score mirrors the caller's selection --------------------


def test_cscv_pbo_default_mode_is_conjunction():
    # The default path IS the conjunction path — the conjunction tests above pin its numbers;
    # this pins that omitting the mode selects it.
    rng = np.random.RandomState(5)
    cells = _pbo_cells(rng, 800, 12)
    assert cscv_pbo(cells, 800, ["t"], n_combos_declared=12) == cscv_pbo(
        cells, 800, ["t"], mode="conjunction", n_combos_declared=12
    )


def test_cscv_pbo_unknown_mode_refuses():
    with pytest.raises(ValueError):
        cscv_pbo([], 400, ["t"], mode="both", n_combos_declared=0)


def test_cscv_pbo_basket_score_is_the_pooled_per_observation_sharpe():
    # HAND-CHECK by construction: members fire on DISJOINT bars, so a basket combo's pooled
    # (bar × member) pool IS the union pool of one synthetic target. Basket scoring of the
    # two-member grid must therefore reproduce single-target conjunction scoring of the union
    # grid (min over ONE target = identity = that pool's per-observation Sharpe) — which is the
    # statement "the basket score is the pooled Sharpe, no weakest-member min".
    rng = np.random.RandomState(9)
    T, h = 800, 1
    fwd_a = rng.normal(0.0, 0.01, T)
    fwd_b = rng.normal(0.0, 0.01, T)
    basket_cells, union_cells = [], []
    for k in range(6):
        bars = rng.choice(T - 2, 80, replace=False)
        ma = np.zeros(T)
        ma[bars[:40]] = 1.0
        mb = np.zeros(T)
        mb[bars[40:]] = 1.0
        basket_cells += [
            {"key": (k, "A"), "mask_col": ma, "fwd_col": fwd_a, "h": h},
            {"key": (k, "B"), "mask_col": mb, "fwd_col": fwd_b, "h": h},
        ]
        mu = np.zeros(T)
        mu[bars] = 1.0
        fwd_u = np.where(ma > 0, fwd_a, fwd_b)  # each member's own fwd at its own firing bars
        union_cells.append({"key": (k, "P"), "mask_col": mu, "fwd_col": fwd_u, "h": h})
    basket = cscv_pbo(basket_cells, T, ["A", "B"], mode="basket", n_combos_declared=6)
    union = cscv_pbo(union_cells, T, ["P"], n_combos_declared=6)
    assert basket["reason"] is None
    for key in (
        "reason",
        "n_splits",
        "n_combos",
        "n_combos_scoreable",
        "n_combos_declared",
        "blocks",
    ):
        assert basket[key] == union[key]
    for key in ("pbo", "lambda_mean", "oos_degradation_slope", "prob_oos_loss"):
        assert basket[key] == pytest.approx(union[key])


def test_cscv_pbo_basket_admits_partial_combos_and_zero_fills_absent_members():
    rng = np.random.RandomState(10)
    T, h = 800, 1
    fwd_a = rng.normal(0.0, 0.01, T)
    fwd_b = rng.normal(0.0, 0.01, T)

    def _cell(k, tgt, fwd):
        m = np.zeros(T)
        m[rng.choice(T - 2, 60, replace=False)] = 1.0
        return {"key": (k, tgt), "mask_col": m, "fwd_col": fwd, "h": h}

    cells = [_cell(0, "A", fwd_a), _cell(0, "B", fwd_b), _cell(1, "A", fwd_a)]
    # conjunction: combo 1 has no B pool → inadmissible → one full combo → no selection to score
    conj = cscv_pbo(cells, T, ["A", "B"], n_combos_declared=2)
    assert conj["pbo"] is None and conj["reason"] == "single_combo" and conj["n_combos"] == 1
    assert conj["n_combos_scoreable"] == 1  # combo 1 is inadmissible under conjunction
    # basket: ≥1 member present admits combo 1; the absent member zero-fills
    bask = cscv_pbo(cells, T, ["A", "B"], mode="basket", n_combos_declared=2)
    assert bask["reason"] is None and bask["n_combos"] == 2 == bask["n_combos_scoreable"]
    # zero-filled absence == a member that never fired: EXACTLY the same numbers
    never = {"key": (1, "B"), "mask_col": np.zeros(T), "fwd_col": fwd_b, "h": h}
    assert bask == cscv_pbo([*cells, never], T, ["A", "B"], mode="basket", n_combos_declared=2)


def _one_member_edge_cells(T=800, h=1, seed=12):
    """An 8-combo two-member grid whose edge lives in ONE member's pool (combo 0's A mask fires
    on the genuinely shifted bars; every other mask is noise) — the fixture the two mode-contrast
    pins below share EXACTLY, so they read one grid through the two scores."""
    rng = np.random.RandomState(seed)
    fwd_a = rng.normal(0.0, 0.01, T)
    fwd_b = rng.normal(0.0, 0.01, T)
    edge_bars = rng.choice(T - 2, 60, replace=False)
    fwd_a[edge_bars] += 0.02
    cells = []
    for k in range(8):
        ma = np.zeros(T)
        if k == 0:
            ma[edge_bars] = 1.0
        else:
            ma[rng.choice(T - 2, 60, replace=False)] = 1.0
        mb = np.zeros(T)
        mb[rng.choice(T - 2, 60, replace=False)] = 1.0
        cells += [
            {"key": (k, "A"), "mask_col": ma, "fwd_col": fwd_a, "h": h},
            {"key": (k, "B"), "mask_col": mb, "fwd_col": fwd_b, "h": h},
        ]
    return cells, T


def test_cscv_pbo_basket_pools_across_members_instead_of_taking_the_weakest():
    # The edge lives in ONE member's pool. The conjunction score (weakest member) never sees it;
    # the basket's pooled per-observation Sharpe does, and the pooled winner travels.
    cells, T = _one_member_edge_cells()
    bask = cscv_pbo(cells, T, ["A", "B"], mode="basket", n_combos_declared=8)
    assert bask["reason"] is None
    assert bask["pbo"] <= 0.1  # the one-member edge dominates the POOL and keeps winning OOS


def test_cscv_pbo_conjunction_stays_blind_to_a_one_member_edge():
    # The MIRROR pin over the SAME grid: the conjunction (default) score is the WEAKEST member's
    # Sharpe, and the edge member's noise sibling B drags combo 0 back into the pack — so the
    # conjunction read must NOT see the edge the basket read travels on. Kills the mutant where
    # basket scoring leaks into the conjunction path (a leak would reproduce the sibling test's
    # pbo <= 0.1 here); every combo carries both members, so admissibility is identical and the
    # contrast is PURELY the score.
    cells, T = _one_member_edge_cells()
    conj = cscv_pbo(cells, T, ["A", "B"], n_combos_declared=8)  # default mode == conjunction
    assert conj["reason"] is None
    assert conj["pbo"] >= 0.25  # coin-flip territory: the one-member edge does not travel


def test_cscv_pbo_identical_candidates_collapse_to_single_combo():
    # A grid of byte-identical clones is ONE candidate wearing many names: pre-v3 the strict
    # test rank handed the train winner rank 1 (worst) for tying with its own copies, reading
    # PBO = 1.0 — maximal overfitting — off a grid with NO selection in it.
    rng = np.random.RandomState(13)
    T = 800
    fwd = rng.normal(0.0, 0.01, T)
    m = np.zeros(T)
    m[rng.choice(T - 2, 60, replace=False)] = 1.0
    cells = [{"key": (k, "t"), "mask_col": m, "fwd_col": fwd, "h": 1} for k in range(5)]
    out = cscv_pbo(cells, T, ["t"], n_combos_declared=5)
    assert out["pbo"] is None and out["reason"] == "single_combo"
    assert out["n_combos"] == 1  # one distinct candidate
    assert out["n_combos_scoreable"] == 5  # every clone had a pool
    assert out["n_combos_declared"] == 5


def test_cscv_pbo_midrank_full_test_tie_is_lambda_zero():
    # Two DISTINCT masks whose pools carry the same values in every block: every split's train
    # AND test scores tie exactly, the midrank puts the winner at ω = 0.5, λ = 0 — a split that
    # cannot distinguish candidates is not evidence of overfitting. The pre-v3 strict rank read
    # this grid as pbo = 1.0.
    T, h = 400, 1
    fwd = np.zeros(T)
    vals = (0.01, -0.02, 0.03)
    cells = []
    for k in range(2):
        m = np.zeros(T)
        for b in range(8):  # S=8 blocks of 50 bars; every firing survives the purge
            for j, v in enumerate(vals):
                bar = b * 50 + 5 + 10 * j + k  # combo k's bars offset by one — distinct masks
                m[bar] = 1.0
                fwd[bar] = v
        cells.append({"key": (k, "t"), "mask_col": m, "fwd_col": fwd, "h": h})
    out = cscv_pbo(cells, T, ["t"], n_combos_declared=2)
    assert out["reason"] is None and out["n_combos"] == 2
    assert out["pbo"] == 0.0
    assert out["lambda_mean"] == 0.0


def test_cscv_pbo_candidates_with_the_same_block_returns_tie_in_any_row_order():
    # The review's case: two candidates holding the same six returns in each of eight blocks, in
    # other row orders. Every split ties in exact arithmetic, but the block sums ran in row order
    # and their last bits ranked the candidates: pbo 0.571, lambda_mean -0.248. Exact block sums
    # are the same doubles in any order, so every split is a full tie.
    T, h = 800, 1
    vals = np.array([-0.04, -0.02, 0.0, 0.02, 0.04, 0.06])
    # each block's six returns in row order, as indices into `vals` (the review's seed 8)
    orders = (
        "035124 350124 530142 325014 254130 154203 025143 013452",
        "054321 105342 402513 241305 234150 153402 421035 251043",
    )
    cells = []
    for k, order in enumerate(orders):
        mask, fwd = np.zeros(T), np.full(T, np.nan)
        for b, perm in enumerate(order.split()):  # S=8 blocks of 100 bars, nothing purged
            bars = b * 100 + 2 + 10 * np.arange(6)
            mask[bars] = 1.0
            fwd[bars] = vals[[int(i) for i in perm]]
        cells.append({"key": (k, "t"), "mask_col": mask, "fwd_col": fwd, "h": h})
    out = cscv_pbo(cells, T, ["t"], n_combos_declared=2)
    assert out["reason"] is None and out["n_combos"] == 2 and out["n_splits"] == 70
    assert out["pbo"] == 0.0
    assert out["lambda_mean"] == 0.0


def test_cscv_pbo_duplicate_collapse_leaves_a_planted_edge_unchanged():
    # Cloning a noise combo must not move any scored number — the clone collapses away and only
    # the population ledger says it was declared. Also the ledger inequality every block obeys.
    rng = np.random.RandomState(6)
    cells = _pbo_cells(rng, 800, 12, edge_combo=0)
    base = cscv_pbo(cells, 800, ["t"], n_combos_declared=12)
    clone = {**cells[3], "key": (12, "t")}
    out = cscv_pbo([*cells, clone], 800, ["t"], n_combos_declared=13)
    assert out["n_combos"] == 12 == base["n_combos"]
    assert out["n_combos_scoreable"] == 13 and out["n_combos_declared"] == 13

    def scored(d):
        return {k: v for k, v in d.items() if not k.startswith("n_combos")}

    assert scored(out) == scored(base)
    assert out["n_combos"] <= out["n_combos_scoreable"] <= out["n_combos_declared"]


# ---- episode clustering + return-mass concentration ------------------------


def test_concentration_hand_computed():
    rets = np.array([0.5] + [0.01] * 19)  # one whale among 20 → top 5% = 1 obs
    out = concentration(rets)
    assert out["n_top"] == 1
    assert out["top_share_abs"] == pytest.approx(0.5 / (0.5 + 0.19))
    assert np.isnan(concentration(np.array([]))["top_share_abs"])


def test_concentration_of_an_overflowed_mass_is_nan_never_a_passing_zero():
    # 100 returns of 3e306: the top-5 mass (1.5e307) is finite, the total overflows, and the
    # share read 0.0 — "no concentration", which the ceiling check passed. NaN refuses instead.
    out = concentration(np.full(100, 3e306))
    assert out["n_top"] == 5 and np.isnan(out["top_share_abs"])


def test_concentration_share_never_passes_one():
    # The review's case: five masses among 95 zeros. Summed pairwise, the total's eight-way
    # accumulation rounded the five otherwise than their own sum, and the top 5% held
    # 1.0000000000000002 of it. A subset's exact sum never exceeds the whole's.
    top = [0.8605673135181114, 0.6019700926291229, 0.43127194749784414, 0.29344776720559157]
    out = concentration(np.r_[top, 0.2448757820521963, np.zeros(95)])
    assert out["n_top"] == 5 and out["top_share_abs"] == 1.0


def test_episode_stats_overlapping_vs_disjoint():
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2021-01-01")
    # Two overlapping [entry, exit) windows merge; a later disjoint window is its own cluster.
    # |ret| mass of the 2-obs cluster: (0.10+0.30)/(0.10+0.30+0.05) = 0.8
    overlap = pd.DataFrame(
        {
            "entry_time": [base + d * day for d in (0, 3, 20)],
            "exit_time": [base + d * day for d in (10, 12, 25)],
            "ret": [0.10, 0.30, 0.05],
        }
    )
    out = episode_stats(overlap)
    assert out["n"] == 3 and out["n_clusters"] == 2
    assert out["largest_cluster_n"] == 2
    assert out["largest_cluster_share_abs"] == pytest.approx(0.40 / 0.45)
    assert out["largest_cluster_start"].startswith("2021-01-01T00:00:00")
    assert out["max_cluster_share_abs"] == pytest.approx(0.40 / 0.45)
    # |return|-mass Herfindahl over the shares {8/9, 1/9}: 65/81, effective count its inverse
    assert out["mass_hhi"] == pytest.approx(65.0 / 81.0)
    assert out["effective_n_clusters"] == pytest.approx(81.0 / 65.0)

    disjoint = pd.DataFrame(
        {
            "entry_time": [base + d * day for d in (0, 10)],
            "exit_time": [base + d * day for d in (5, 15)],
            "ret": [0.02, -0.03],
        }
    )
    d = episode_stats(disjoint)
    assert d["n_clusters"] == 2 and d["largest_cluster_n"] == 1
    # max_cluster_share_abs is largest-by-MASS, not largest-by-count: the first (tie-broken)
    # largest-by-count cluster holds 0.4 of the mass, but the -0.03 cluster holds 0.6.
    assert d["largest_cluster_share_abs"] == pytest.approx(0.4)
    assert d["max_cluster_share_abs"] == pytest.approx(0.6)
    assert d["mass_hhi"] == pytest.approx(0.4**2 + 0.6**2)
    assert d["effective_n_clusters"] == pytest.approx(1.0 / 0.52)


def test_episode_stats_empty_guards():
    empty = episode_stats(pd.DataFrame())
    assert empty["n"] == 0 and empty["n_clusters"] == 0
    assert np.isnan(empty["max_cluster_share_abs"])
    assert np.isnan(empty["mass_hhi"]) and np.isnan(empty["effective_n_clusters"])
    missing = episode_stats(
        pd.DataFrame({"ret": [0.01], "exit_time": [pd.Timestamp("2021-01-06")]})
    )
    assert missing["n"] == 0


def test_episode_stats_mass_hhi_hand_computed():
    # THE BLIND SPOT the hhi pair exists for: two episodes carrying 50% of the |return| mass
    # each sail under a 0.60 single-share ceiling, while hhi = 0.5 / effective = 2.0 says the
    # pool is exactly two effective episodes. A zero-mass pool refuses both as NaN (the shares'
    # own domain rule).
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2021-03-01")
    frame = pd.DataFrame(
        {
            "entry_time": [base, base + 20 * day],
            "exit_time": [base + 5 * day, base + 25 * day],
            "ret": [0.05, -0.05],
        }
    )
    out = episode_stats(frame)
    assert out["max_cluster_share_abs"] == pytest.approx(0.5)  # under any 0.60 ceiling
    assert out["mass_hhi"] == pytest.approx(0.5)
    assert out["effective_n_clusters"] == pytest.approx(2.0)
    zero = episode_stats(frame.assign(ret=[0.0, 0.0]))
    assert np.isnan(zero["mass_hhi"]) and np.isnan(zero["effective_n_clusters"])


def test_episode_shares_and_the_hhi_pair_hold_their_bounds():
    # A zero-mass episode (three rows of 0.0) before one of [0.1, 0.1, 0.1, 0.1, 0.2]: summed
    # pairwise, the pool's mass rounded below the episode's own, and the share read
    # 1.0000000000000002, the hhi 1.0000000000000004, the effective count 0.9999999999999996 and
    # the ledger's omitted share past a cap of one 1.0000000000000002. Five equal episodes of 0.3
    # read an effective count of 5.000000000000001, past n_clusters. Exact masses and the exact
    # hhi pair hold every bound.
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2021-01-01")
    days = [0, 1, 2, 200, 201, 202, 203, 204]
    frame = pd.DataFrame(
        {
            "entry_time": [base + d * day for d in days],
            "exit_time": [base + (d + 5) * day for d in days],
            "ret": [0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.2],
        }
    )
    out = episode_stats(frame)
    assert out["n_clusters"] == 2
    assert out["max_cluster_share_abs"] == out["largest_cluster_share_abs"] == 1.0
    assert out["mass_hhi"] == 1.0 and out["effective_n_clusters"] == 1.0
    assert [e["share_abs"] for e in episode_ledger(frame)["entries"]] == [0.0, 1.0]
    assert episode_ledger(frame, cap=1)["omitted_share_abs"] == 1.0
    equal = pd.DataFrame(
        {
            "entry_time": [base + 10 * i * day for i in range(5)],
            "exit_time": [base + (10 * i + 2) * day for i in range(5)],
            "ret": [0.3] * 5,
        }
    )
    eq = episode_stats(equal)
    assert eq["n_clusters"] == 5 and eq["mass_hhi"] == 0.2 and eq["effective_n_clusters"] == 5.0


def test_episode_shares_of_an_overflowed_mass_are_nan_never_a_passing_zero():
    # Two disjoint episodes of 1e308 each: the total |return| mass overflows, and every share
    # read `x / inf = 0.0` — a max_cluster_share_abs of zero, which the one-episode ceiling
    # passes. No share of an overflowed total exists: NaN, which the checklist refuses.
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2021-03-01")
    frame = pd.DataFrame(
        {
            "entry_time": [base, base + 20 * day],
            "exit_time": [base + 5 * day, base + 25 * day],
            "ret": [1e308, 1e308],
        }
    )
    out = episode_stats(frame)
    ledger = episode_ledger(frame)
    assert out["n_clusters"] == 2
    for key in ("largest_cluster_share_abs", "max_cluster_share_abs", "mass_hhi"):
        assert np.isnan(out[key]), key
    assert np.isnan(out["effective_n_clusters"])
    assert [np.isnan(e["share_abs"]) for e in ledger["entries"]] == [True, True]


def test_overlap_clusters_chain_merge_vs_greedy_count():
    # Chain A∩B, B∩C with A∦C: ONE merged cluster, but TWO greedy non-overlapping windows —
    # clusters are the more conservative independent-unit count (n_clusters <= n_nonoverlap always).
    entry = np.array([0, 9, 19])
    exit_ = np.array([10, 20, 30])
    assert overlap_clusters(entry, exit_) == [[0, 1, 2]]
    assert nonoverlap_count(entry, horizon=10) == 2
    # Half-open tie rule: a window starting exactly at the running end is a NEW cluster.
    assert overlap_clusters(np.array([0, 10]), np.array([10, 20])) == [[0], [1]]
    assert overlap_clusters(np.array([]), np.array([])) == []


def test_episode_clusters_never_outnumber_independent_windows():
    # Transitive merging is coarser than greedy non-overlap, so a cell's episode count is weakly
    # stricter than its n_nonoverlap — the property the per-cell concentration read leans on.
    rng = np.random.RandomState(11)
    h = 12
    for _ in range(20):
        n = int(rng.randint(1, 40))
        entry = np.sort(rng.randint(0, 500, size=n))
        exit_ = entry + h
        assert len(overlap_clusters(entry, exit_)) <= nonoverlap_count(entry, horizon=h)


# ---- episode_ledger: the time-ordered per-cell episode list --------------------


def test_episode_ledger_matches_episode_stats_and_hand_computed_shares():
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2021-01-01")
    # Cluster 1: [0,10) and [3,12) chain-merge (episode end = the LATER exit, day 12);
    # cluster 2: [20,25). |ret| shares: 0.40/0.45 and 0.05/0.45.
    trades = pd.DataFrame(
        {
            "entry_time": [base + d * day for d in (0, 3, 20)],
            "exit_time": [base + d * day for d in (10, 12, 25)],
            "ret": [0.10, 0.30, -0.05],
        }
    )
    led = episode_ledger(trades)
    assert led["n_total"] == episode_stats(trades)["n_clusters"] == 2  # the reconciliation pin
    assert led["n_omitted"] == 0 and led["omitted_share_abs"] == 0.0
    assert led["cap"] == EPISODE_LIST_MAX == 32
    first, second = led["entries"]
    assert first["start"].startswith("2021-01-01") and first["end"].startswith("2021-01-13")
    assert first["n"] == 2 and first["mean_ret"] == pytest.approx(0.20)
    assert first["share_abs"] == pytest.approx(0.40 / 0.45)
    assert second["start"].startswith("2021-01-21") and second["end"].startswith("2021-01-26")
    assert second["n"] == 1 and second["mean_ret"] == pytest.approx(-0.05)
    assert second["share_abs"] == pytest.approx(0.05 / 0.45)
    # mass conservation: listed + omitted shares cover the pool exactly
    assert sum(e["share_abs"] for e in led["entries"]) + led["omitted_share_abs"] == (
        pytest.approx(1.0)
    )


def test_episode_ledger_truncation_is_time_ordered_and_mass_conserving():
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2020-01-01")
    # 8 disjoint episodes; the LAST carries most of the mass — a share-ranked ledger would list
    # it first, a time-ordered one must OMIT it under cap=4 and account for it in the remainder.
    rets = [0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.90]
    trades = pd.DataFrame(
        {
            "entry_time": [base + 10 * i * day for i in range(8)],
            "exit_time": [base + (10 * i + 3) * day for i in range(8)],
            "ret": rets,
        }
    )
    led = episode_ledger(trades, cap=4)
    assert led["cap"] == 4 and led["n_total"] == 8
    assert len(led["entries"]) == 4 and led["n_omitted"] == 4
    starts = [e["start"] for e in led["entries"]]
    assert starts == sorted(starts)  # earliest first, never ranked by share
    total = sum(abs(r) for r in rets)
    assert [e["share_abs"] for e in led["entries"]] == [
        pytest.approx(abs(r) / total) for r in rets[:4]
    ]
    assert led["omitted_share_abs"] == pytest.approx(sum(abs(r) for r in rets[4:]) / total)
    assert sum(e["share_abs"] for e in led["entries"]) + led["omitted_share_abs"] == (
        pytest.approx(1.0)
    )
    # the whale stayed in the omitted mass — truncation promoted nothing
    assert all(e["share_abs"] < 0.5 for e in led["entries"])


def test_episode_ledger_dead_cell_is_the_explicit_empty_block():
    assert episode_ledger(pd.DataFrame()) == {
        "entries": [],
        "n_total": 0,
        "n_omitted": 0,
        "omitted_share_abs": 0.0,
        "cap": 32,
    }
    # NaN rows drop exactly as in episode_stats — a fully-censored frame is a dead cell too
    censored = pd.DataFrame(
        {
            "entry_time": [pd.Timestamp("2021-01-01")],
            "exit_time": [pd.Timestamp("2021-01-05")],
            "ret": [float("nan")],
        }
    )
    assert episode_ledger(censored)["n_total"] == 0


def test_episode_ledger_reconciles_with_episode_stats_on_random_pools():
    # n_total == episode_stats.n_clusters is BY CONSTRUCTION (same dropna, same sort, same
    # frozen half-open merge) — pinned over random pools so the pipelines cannot drift apart.
    rng = np.random.RandomState(23)
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2019-06-01")
    for _ in range(10):
        n = int(rng.randint(1, 60))
        entry_d = np.sort(rng.randint(0, 300, size=n))
        width = rng.randint(1, 15, size=n)
        trades = pd.DataFrame(
            {
                "entry_time": [base + int(d) * day for d in entry_d],
                "exit_time": [base + int(d + w) * day for d, w in zip(entry_d, width, strict=True)],
                "ret": rng.randn(n) * 0.02,
            }
        )
        led = episode_ledger(trades)
        assert led["n_total"] == episode_stats(trades)["n_clusters"]
        assert len(led["entries"]) + led["n_omitted"] == led["n_total"]
        assert len(led["entries"]) <= EPISODE_LIST_MAX
        if led["n_omitted"] == 0:
            assert led["omitted_share_abs"] == 0.0


# ---- episode_profile: the episode-deduplicated twin (evidence-only) ----------


def test_episode_profile_hand_computed_three_episodes():
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2021-01-01")
    # Episodes: [0,10) and [3,12) chain-merge; [20,25); [30,34). Per-episode ret MEANS (the
    # ledger's own aggregate): (0.10+0.30)/2 = 0.20, −0.05, 0.08 → hit 2/3, PF over the means
    # (0.20+0.08)/0.05 = 5.6, and the time-ordered sign walk +,−,+ gives 1/1 streaks.
    # Per-episode extremes: mae min per cluster = −0.06 / −0.01 / −0.03, mfe max = 0.35 / 0.01
    # / 0.09 — one crisis smeared across two rows contributes ONE worst excursion.
    trades = pd.DataFrame(
        {
            "entry_time": [base + d * day for d in (0, 3, 20, 30)],
            "exit_time": [base + d * day for d in (10, 12, 25, 34)],
            "ret": [0.10, 0.30, -0.05, 0.08],
            "mae": [-0.02, -0.06, -0.01, -0.03],
            "mfe": [0.05, 0.35, 0.01, 0.09],
        }
    )
    prof = episode_profile(trades)
    assert prof["n_episodes"] == episode_stats(trades)["n_clusters"] == 3
    assert prof["hit_rate"] == pytest.approx(2 / 3)
    assert prof["mean_ret"] == pytest.approx((0.20 - 0.05 + 0.08) / 3)
    assert prof["profit_factor"] == pytest.approx((0.20 + 0.08) / 0.05)
    assert prof["worst_ret"] == pytest.approx(-0.05)
    assert prof["best_ret"] == pytest.approx(0.20)
    assert prof["ret_quantiles"]["p50"] == pytest.approx(0.08)
    assert prof["max_win_streak"] == 1 and prof["max_loss_streak"] == 1
    mae_b, mfe_b = prof["mae_quantiles"], prof["mfe_quantiles"]
    assert mae_b["n"] == mfe_b["n"] == 3
    assert mae_b["worst"] == pytest.approx(-0.06)
    assert mae_b["mean"] == pytest.approx((-0.06 - 0.01 - 0.03) / 3)
    assert mfe_b["best"] == pytest.approx(0.35)
    assert prof["edge_ratio"] == pytest.approx(
        np.mean([0.35, 0.01, 0.09]) / abs(np.mean([-0.06, -0.01, -0.03]))
    )


def test_episode_profile_deduplicates_what_row_level_statistics_inflate():
    # THE point of the block: ~h overlapping rows of one losing crisis beside one clean winning
    # day. Row-level hit_rate reads 1/7 and the row PF is drowned; episode-level reads 1/2 and
    # the PF is the two events' own ratio — the divergence IS the cluster diagnostic.
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2022-01-01")
    crisis_entries = [base + d * day for d in range(0, 6)]  # 6 overlapping [d, d+8) rows
    trades = pd.DataFrame(
        {
            "entry_time": [*crisis_entries, base + 40 * day],
            "exit_time": [*[e + 8 * day for e in crisis_entries], base + 44 * day],
            "ret": [-0.05, -0.06, -0.04, -0.05, -0.06, -0.04, 0.03],
        }
    )
    prof = episode_profile(trades)
    row_hit = float(np.mean(trades["ret"].to_numpy() > 0))
    assert row_hit == pytest.approx(1 / 7)
    assert prof["n_episodes"] == 2
    assert prof["hit_rate"] == pytest.approx(1 / 2)
    crisis_mean = float(np.mean([-0.05, -0.06, -0.04, -0.05, -0.06, -0.04]))
    assert prof["profit_factor"] == pytest.approx(0.03 / abs(crisis_mean))
    assert prof["max_loss_streak"] == 1  # one crisis, however many rows it wore


def test_episode_profile_streaks_walk_time_order_and_zero_breaks_both():
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2020-01-01")
    # Disjoint episodes with ret means +,+,0,−,−,−,+ → max_win 2, max_loss 3; the zero-mean
    # episode joins neither side, exactly like a zero row in profit_factor.
    rets = [0.01, 0.02, 0.0, -0.01, -0.02, -0.01, 0.03]
    trades = pd.DataFrame(
        {
            "entry_time": [base + 10 * i * day for i in range(7)],
            "exit_time": [base + (10 * i + 3) * day for i in range(7)],
            "ret": rets,
        }
    )
    prof = episode_profile(trades)
    assert prof["n_episodes"] == 7
    assert prof["max_win_streak"] == 2 and prof["max_loss_streak"] == 3


def test_episode_profile_reads_nothing_off_an_overflowed_episode_mean():
    # One episode of two overlapping rows, 1.2e308 and 1.3e308: its true mean is 1.25e308, but
    # its row SUM overflows, and a pairwise mean read inf. The profit factor (6.0 here) and the
    # quantiles were read WITHOUT it — the pool filters drop a non-finite value — the worst
    # return beside it, and the hit rate and the win streak off its overflowed sign. Its value
    # and sign are unknown, so every read over the episode means is null and it breaks both
    # streaks (2, not 3). (Two EQUAL rows read their value: an all-equal pool's exact mean.)
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2021-03-01")
    starts = [0, 1, 20, 40, 60, 80]
    trades = pd.DataFrame(
        {
            "entry_time": [base + d * day for d in starts],
            "exit_time": [base + (d + 5) * day for d in starts],
            "ret": [1.2e308, 1.3e308, 1.0, 2.0, -1.0, 3.0],
        }
    )
    prof = episode_profile(trades)
    assert prof["n_episodes"] == 5
    for key in ("hit_rate", "mean_ret", "profit_factor", "worst_ret", "best_ret"):
        assert np.isnan(prof[key]), key
    assert all(np.isnan(v) for v in prof["ret_quantiles"].values())
    assert prof["max_win_streak"] == 2 and prof["max_loss_streak"] == 1


def test_episode_ledger_mean_past_the_double_range_is_nan_never_a_signed_inf():
    # The ledger's per-episode mean over finite rows whose sum leaves the double range. Read off
    # the overflow it was `inf` — into a field the emitted contract typed plain `float` (the
    # run died at the seam) — and not even sign-safe: 1e308 + 1e308 is +inf before three rows of
    # -1.7e308 bring the true sum (and mean, -6.2e307) below zero. NaN → null, the unknown mean
    # episode_profile already reads nothing off; an episode in range keeps its bits.
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2021-03-01")
    starts = [0, 1, 2, 3, 4, 40]
    trades = pd.DataFrame(
        {
            "entry_time": [base + d * day for d in starts],
            "exit_time": [base + (d + 5) * day for d in starts],
            "ret": [1e308, 1e308, -1.7e308, -1.7e308, -1.7e308, 0.125],
        }
    )
    ledger = episode_ledger(trades)
    assert [e["n"] for e in ledger["entries"]] == [5, 1]
    assert np.isnan(ledger["entries"][0]["mean_ret"]) and ledger["entries"][1]["mean_ret"] == 0.125
    prof = episode_profile(trades)
    assert np.isnan(prof["mean_ret"]) and prof["max_win_streak"] == 1
    # a full bar's exact 0 through `rets_pool` stays exactly 0 beside it
    zeros = episode_ledger(trades, rets_pool=np.array([0.0] * 5 + [0.125]))
    assert [e["mean_ret"] for e in zeros["entries"]] == [0.0, 0.125]


def test_episode_means_are_exactly_rounded_sums():
    # The review's episodes: four overlapping rows [−0.5, −2**−55, 0.5, 2**−55] each, summing to
    # exactly 0. Summed pairwise each mean read 6.9e-18 — eight "winning" episodes, a hit rate of
    # 1 and a win streak of 8 beside a row mean_ret of exactly 0. The exactly rounded sum reads
    # every episode at 0: no win, no streak.
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2022-01-03")
    rows = [-0.5, -(2.0**-55), 0.5, 2.0**-55]
    starts = [10 * e + i for e in range(8) for i in range(4)]
    trades = pd.DataFrame(
        {
            "entry_time": [base + d * day for d in starts],
            "exit_time": [base + (d + 5) * day for d in starts],
            "ret": rows * 8,
        }
    )
    assert float(np.mean(rows)) > 0.0  # the premise
    assert [e["mean_ret"] for e in episode_ledger(trades)["entries"]] == [0.0] * 8
    prof = episode_profile(trades)
    assert prof["n_episodes"] == 8 and prof["hit_rate"] == 0.0 and prof["mean_ret"] == 0.0
    assert prof["max_win_streak"] == 0 and prof["max_loss_streak"] == 0


def test_episode_profile_merges_across_targets_exactly_like_episode_stats():
    # Two targets riding the same market window are ONE episode — the same cross-target merge
    # doctrine as episode_stats, so a crisis seen through three names cannot read as three
    # episodes here either.
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2021-03-01")
    trades = pd.DataFrame(
        {
            "entry_time": [base, base + 1 * day],
            "exit_time": [base + 5 * day, base + 6 * day],
            "ret": [0.04, 0.02],
            "target": ["AAA", "BBB"],
        }
    )
    prof = episode_profile(trades)
    assert prof["n_episodes"] == 1
    assert prof["mean_ret"] == pytest.approx(0.03)  # ONE episode aggregating both members' rows


def test_episode_profile_excursion_subsets_track_censored_paths():
    # A whole episode whose path columns were censored contributes nothing to the excursion
    # pools — their n sits below n_episodes and the block says so, the same subset doctrine as
    # the row-level mae/mfe blocks.
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2021-05-01")
    trades = pd.DataFrame(
        {
            "entry_time": [base, base + 20 * day, base + 40 * day],
            "exit_time": [base + 3 * day, base + 23 * day, base + 43 * day],
            "ret": [0.02, 0.03, 0.01],
            "mae": [-0.01, float("nan"), -0.05],
            "mfe": [0.03, float("nan"), float("nan")],
        }
    )
    prof = episode_profile(trades)
    assert prof["n_episodes"] == 3
    assert prof["mae_quantiles"]["n"] == 2 and prof["mfe_quantiles"]["n"] == 1
    assert prof["mae_quantiles"]["worst"] == pytest.approx(-0.05)
    # the ratio PAIRS at the episode level: the mae-only third episode feeds its own quantile
    # block above but never the ratio — only the first episode carries both legs
    assert prof["edge_ratio"] == pytest.approx(0.03 / 0.01)


def test_episode_profile_is_invariant_to_tied_row_order():
    # Rows sharing one entry bar land in one cluster whichever order the frame holds them in —
    # per-episode mean/min/max are order-free, so the whole block must be.
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2021-07-01")
    frame = pd.DataFrame(
        {
            "entry_time": [base, base, base + 20 * day],
            "exit_time": [base + 4 * day, base + 4 * day, base + 24 * day],
            "ret": [0.02, -0.01, 0.05],
            "mae": [-0.02, -0.04, -0.01],
            "mfe": [0.03, 0.01, 0.06],
        }
    )
    shuffled = frame.iloc[[1, 2, 0]].reset_index(drop=True)
    assert episode_profile(frame) == episode_profile(shuffled)


def test_episode_means_sum_the_pool_values_while_shares_keep_ret():
    # `rets_pool` (row-aligned with the frame) is what an episode MEAN sums: a `cross_mean`
    # basket's full bar reads 0.0 on each of its rows — the identity its excesses' sum holds —
    # where its rounded rows average to residue (a "win" every time, a streak of them). The
    # |return| shares stay on `ret`, and the values follow their rows by POSITION through the
    # sort, whatever index labels the frame carries.
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2021-07-01")
    frame = pd.DataFrame(
        {
            "entry_time": [base + 20 * day, base, base, base],
            "exit_time": [base + 21 * day, base + day, base + day, base + day],
            "ret": [0.05, -5.333333333333333, -3.333333333333333, 8.666666666666668],
            "target": ["A", "C", "B", "A"],
        }
    )
    pool = np.array([0.05, 0.0, 0.0, 0.0])
    plain_led, plain_prof = episode_ledger(frame), episode_profile(frame)
    assert plain_led["entries"][0]["mean_ret"] != 0.0  # the premise: residue
    assert plain_prof["hit_rate"] == 1.0 and plain_prof["max_win_streak"] == 2
    led = episode_ledger(frame, rets_pool=pool)
    assert [e["mean_ret"] for e in led["entries"]] == [0.0, 0.05]
    assert [e["share_abs"] for e in led["entries"]] == [
        e["share_abs"] for e in plain_led["entries"]
    ]
    prof = episode_profile(frame, rets_pool=pool)
    assert prof["n_episodes"] == 2 and prof["hit_rate"] == 0.5 and prof["worst_ret"] == 0.0
    assert prof["max_win_streak"] == 1 and prof["max_loss_streak"] == 0
    perm = [2, 0, 3, 1]  # labels left non-positional on purpose
    assert episode_ledger(frame.iloc[perm], rets_pool=pool[perm]) == led
    assert episode_profile(frame.iloc[perm], rets_pool=pool[perm]) == prof
    # omitting it IS `rets_pool=ret`, byte for byte; a misaligned pool is refused
    assert episode_ledger(frame, rets_pool=frame["ret"].to_numpy()) == plain_led
    assert episode_profile(frame, rets_pool=frame["ret"].to_numpy()) == plain_prof
    with pytest.raises(ValueError, match="row-aligned"):
        episode_profile(frame, rets_pool=pool[:3])


def test_episode_profile_dead_cell_is_the_explicit_zero_block():
    for dead in (
        pd.DataFrame(),
        pd.DataFrame(
            {
                "entry_time": [pd.Timestamp("2021-01-01")],
                "exit_time": [pd.Timestamp("2021-01-05")],
                "ret": [float("nan")],
            }
        ),
    ):
        prof = episode_profile(dead)
        assert prof["n_episodes"] == 0
        assert prof["max_win_streak"] == 0 and prof["max_loss_streak"] == 0
        assert np.isnan(prof["hit_rate"]) and np.isnan(prof["mean_ret"])
        assert np.isnan(prof["profit_factor"]) and np.isnan(prof["edge_ratio"])
        assert prof["mae_quantiles"]["n"] == 0 and np.isnan(prof["mae_quantiles"]["worst"])
        assert all(np.isnan(v) for v in prof["ret_quantiles"].values())


def test_episode_profile_reconciles_with_episode_stats_on_random_pools():
    # n_episodes == episode_stats.n_clusters is BY CONSTRUCTION (verbatim pipeline) — pinned
    # over random pools so the two cannot drift apart; the frames carry no excursion columns,
    # which also pins the defensive read (absent column → empty excursion pools, never a crash).
    rng = np.random.RandomState(41)
    day = pd.Timedelta(days=1)
    base = pd.Timestamp("2018-02-01")
    for _ in range(10):
        n = int(rng.randint(1, 60))
        entry_d = np.sort(rng.randint(0, 300, size=n))
        width = rng.randint(1, 15, size=n)
        trades = pd.DataFrame(
            {
                "entry_time": [base + int(d) * day for d in entry_d],
                "exit_time": [base + int(d + w) * day for d, w in zip(entry_d, width, strict=True)],
                "ret": rng.randn(n) * 0.02,
            }
        )
        prof = episode_profile(trades)
        assert prof["n_episodes"] == episode_stats(trades)["n_clusters"]
        assert prof["mae_quantiles"]["n"] == 0 and prof["mfe_quantiles"]["n"] == 0
        assert max(prof["max_win_streak"], prof["max_loss_streak"]) <= prof["n_episodes"]


# ---- benchmark_regression: paired-leg attribution (evidence-only) ------------


def test_cross_mean_pooled_mean_full_bars_are_exactly_zero_in_any_member_order():
    # A bar on which every member closed contributes S − 1.0·S: exactly zero, in any column order.
    rng = np.random.RandomState(14)
    raw = np.round(rng.normal(0.001, 0.02, (300, 5)), 6)
    full = np.ones_like(raw, dtype=bool)
    for perm in ([0, 1, 2, 3, 4], [4, 2, 0, 3, 1], [1, 0, 4, 3, 2]):
        assert cross_mean_pooled_mean(raw[:, perm], full[:, perm]) == 0.0
    # the per-row route it replaces carries residue whose sign depends on the order
    excess = raw - raw.mean(axis=1, keepdims=True)
    assert float(np.mean(excess)) != 0.0


def test_cross_mean_pooled_mean_partial_pool_is_the_excess_mean():
    # A selective pool: the mean of the closed rows' excess legs, to rounding, and bit-identical
    # under any member order; an empty pool is NaN.
    rng = np.random.RandomState(15)
    raw = rng.normal(0.0, 0.02, (200, 4))
    closed = rng.rand(200, 4) < 0.3
    excess = raw - raw.mean(axis=1, keepdims=True)
    want = float(np.mean(excess[closed]))
    got = cross_mean_pooled_mean(raw, closed)
    assert got == pytest.approx(want, rel=1e-12)
    perm = [2, 0, 3, 1]
    assert cross_mean_pooled_mean(raw[:, perm], closed[:, perm]) == got
    assert np.isnan(cross_mean_pooled_mean(raw, np.zeros_like(closed)))


def test_cross_mean_pooled_mean_identical_rows_are_exactly_zero_whatever_closed():
    # Every member's leg the same value on a bar (a shared outcome feed): each excess is exactly
    # zero whichever members closed; one leg off it is ordinary variation again.
    rng = np.random.RandomState(16)
    x = np.round(rng.normal(0.0, 0.03, 400), 4)
    raw = np.repeat(x[:, None], 7, axis=1)
    closed = rng.rand(400, 7) < 0.4
    closed[0] = [True, False, False, False, False, False, False]
    assert cross_mean_pooled_mean(raw, closed) == 0.0
    raw2 = raw.copy()
    raw2[0, 6] += 0.01
    np.testing.assert_allclose(
        cross_mean_pooled_mean(raw2, closed), (-0.01 / 7) / closed.sum(), rtol=1e-9
    )


def test_cross_mean_pooled_mean_partial_basket_cancels_exactly():
    # A 22-member bar of 7 zeros, 7 twos and 8 ones has mean 1; its 15 firing members (the zeros,
    # the twos, one of the ones) have excesses summing to exactly 0. The per-bar form rounded the
    # membership fraction before the cancellation — 15 − fl(15/22)·22 = 1.8e-15 — and a pool of
    # such bars read a positive mean_ret the support floor passed. The weighted legs sum exactly.
    raw = np.array([[0.0] * 7 + [2.0] * 7 + [1.0] * 8] * 30)
    closed = np.array([[True] * 15 + [False] * 7] * 30)
    assert 15.0 - (15 / 22) * 22.0 != 0.0  # the defect's arithmetic
    assert cross_mean_pooled_mean(raw, closed) == 0.0
    for perm in (np.arange(22)[::-1], np.random.RandomState(3).permutation(22)):
        assert cross_mean_pooled_mean(raw[:, perm], closed[:, perm]) == 0.0
    # a dispersed partial pool: the exact weighted numerator, correctly rounded, over N·n
    rng = np.random.RandomState(17)
    raw = rng.normal(0.0, 0.02, (300, 9)) * 10.0 ** rng.randint(-3, 4, (300, 9))
    closed = rng.rand(300, 9) < 0.35
    k = closed.sum(axis=1, keepdims=True)
    weights = (9 * closed - k).ravel().tolist()
    exact = sum(
        (w * Fraction(x) for w, x in zip(weights, raw.ravel().tolist(), strict=True) if w),
        Fraction(0),
    )
    got = cross_mean_pooled_mean(raw, closed)
    assert got == float(exact) / (9 * int(closed.sum()))
    assert got == pytest.approx(float(np.mean((raw - raw.mean(axis=1, keepdims=True))[closed])))


def test_cross_mean_pooled_mean_overflow_is_refusable_not_a_crash():
    # Finite legs whose exact pooled sum leaves the double range: the per-row route read inf
    # (refused by the gate as non-finite); the exact route must not raise out of the run.
    assert np.isnan(cross_mean_pooled_mean(np.array([[1.5e308, 0.0]] * 3), np.array([[1, 0]] * 3)))
    # a bar whose own sum overflows
    raw = np.array([[1.5e308, 1.5e308, -1.0]] * 2)
    assert np.isnan(cross_mean_pooled_mean(raw, np.array([[1, 0, 0]] * 2, dtype=bool)))


def test_cross_mean_bar_numerators_carry_each_bar_s_numerator_exactly():
    # A bar's numerator — N times its closed members' excess sum — is the exact sum of its
    # weighted legs, which one double cannot hold once it needs more than 53 bits: legs spanning
    # binades take two parts. The parts sum exactly to it on every bar and sit on the bar's first
    # closed member (zeros on its others); a bar no member closed is zeros, one whose sum leaves
    # the range NaN.
    rng = np.random.RandomState(9)
    raw = rng.uniform(0.01, 10.0, (40, 4))
    closed = rng.random_sample((40, 4)) < 0.5
    closed[0] = False
    raw[1], closed[1] = [1e308, -1e308, 1e308, 0.5], [True, False, False, False]
    num = cross_mean_bar_numerators(raw, closed)
    assert num.parts.shape[1] >= 2
    for t in range(2, 40):
        legs = [Fraction(float(x)) for x in raw[t]]
        exact = sum(4 * legs[g] - sum(legs) for g in np.flatnonzero(closed[t]))
        assert sum(map(Fraction, num.parts[t].tolist())) == exact, t
    assert (num.parts[0] == 0.0).all() and np.isnan(num.parts[1]).all()
    bars, members = np.nonzero(closed)
    rows = num.at(bars, members)
    first = members == np.argmax(closed, axis=1)[bars]
    np.testing.assert_array_equal(rows[first], num.parts[bars[first]])
    assert (rows[~first] == 0.0).all()


def _pooled_hac(excess, raw, closed, mean, h):
    """``cross_mean_pooled_hac`` over the bar numerators of ``raw`` / ``closed`` — the runner's
    call."""
    return cross_mean_pooled_hac(excess, closed, cross_mean_bar_numerators(raw, closed), mean, h)


def _cross_mean_hac_se_exact(raw, closed, h):
    """The pooled ``cross_mean`` HAC's SE in exact arithmetic: the Bartlett form over each bar's
    exact excess deviation sum ``Σ_{closed g} e_tg − k_t·μ`` about the exact pooled mean ``μ``,
    over ``(n − 1)·n``, only the root rounded."""
    n_members = raw.shape[1]
    sums, ks = {}, {}
    for t in np.flatnonzero(closed.any(axis=1)).tolist():
        legs = [Fraction(float(x)) for x in raw[t]]
        sums[t] = sum(legs[g] - sum(legs) / n_members for g in np.flatnonzero(closed[t]))
        ks[t] = int(closed[t].sum())
    n = sum(ks.values())
    mu = sum(sums.values()) / n
    dev = {t: sums[t] - ks[t] * mu for t in sums}
    s = sum(
        (1 - Fraction(abs(a - b), h)) * dev[a] * dev[b] for a in dev for b in dev if abs(a - b) < h
    )
    return math.sqrt(s / (n - 1) / n)


def test_cross_mean_pooled_hac_reads_the_exact_bar_deviation_sums():
    # Each bar's deviation sum is its exact numerator over N less its closed rows' share of the
    # exact pooled mean, rounded once: the SE is the estimator's exact value to rounding, and no
    # member order moves a bit of it.
    rng = np.random.RandomState(18)
    raw = rng.normal(0.0, 0.02, (120, 5)) * 10.0 ** rng.randint(-3, 4, (120, 5))
    closed = rng.rand(120, 5) < 0.4
    closed[:10] = True  # ten full bars beside the partial ones
    excess = raw - raw.mean(axis=1, keepdims=True)
    mean = cross_mean_pooled_mean(raw, closed)
    for h in (1, 4):
        t, se = _pooled_hac(excess, raw, closed, mean, h)
        assert se == pytest.approx(_cross_mean_hac_se_exact(raw, closed, h), rel=1e-14)
        assert t == mean / se
        for perm in ([4, 2, 0, 3, 1], [1, 0, 4, 3, 2]):
            moved = _pooled_hac(excess[:, perm], raw[:, perm], closed[:, perm], mean, h)
            assert moved == (t, se)
    # a bar whose numerator leaves the double range: no estimate
    big = raw.copy()
    big[50] = [1e308, -1e308, 1e308, 0.0, 0.0]
    big_closed = closed.copy()
    big_closed[50] = [True, False, False, False, False]
    t, se = _pooled_hac(excess, big, big_closed, mean, 1)
    assert np.isnan(t) and np.isnan(se)


def test_cross_mean_pooled_hac_has_no_variance_where_every_bar_sum_is_exactly_zero():
    # Every bar's excess sum exactly 0 — all-full bars, identical legs, a firing subset whose
    # excesses cancel — is no long-run variance: null, in any member order. Read off the rounded
    # rows, each bar's sum was residue, and the HAC divided it.
    rng = np.random.RandomState(19)
    raw = np.round(rng.normal(0.001, 0.02, (60, 4)), 6)
    full = np.ones_like(raw, dtype=bool)
    for perm in ([0, 1, 2, 3], [3, 1, 0, 2]):
        r, c = raw[:, perm], full[:, perm]
        excess = r - r.mean(axis=1, keepdims=True)
        t, se = _pooled_hac(excess, r, c, cross_mean_pooled_mean(r, c), 1)
        assert np.isnan(t) and np.isnan(se), perm
    same = np.repeat(raw[:, :1], 4, axis=1)
    t, se = _pooled_hac(np.zeros_like(same), same, full, 0.0, 1)
    assert np.isnan(t) and np.isnan(se)
    # the 22-member bar whose 15 firing members' excesses sum to exactly 0, on every bar
    raw22 = np.array([[0.0] * 7 + [2.0] * 7 + [1.0] * 8] * 30)
    closed22 = np.array([[True] * 15 + [False] * 7] * 30)
    t, se = _pooled_hac(raw22 - 1.0, raw22, closed22, 0.0, 2)
    assert np.isnan(t) and np.isnan(se)
    # the review's basket: members (a, b, a, b); only the first two fire, three bars apart, and
    # every other firing bar they move while their twins mirror them — each firing bar's two
    # excesses cancel exactly, its two rounded rows do not
    a, b = math.log(0.99), math.log(0.98)
    raw = np.zeros((90, 4))
    raw[0::6] = [a, b, a, b]
    closed = np.zeros((90, 4), dtype=bool)
    closed[0::3, :2] = True
    excess = raw - raw.mean(axis=1, keepdims=True)
    assert (excess[closed].reshape(-1, 2).sum(axis=1) != 0.0).any()  # the premise
    mean = cross_mean_pooled_mean(raw, closed)
    assert mean == 0.0
    t, se = _pooled_hac(excess, raw, closed, mean, 1)
    assert np.isnan(t) and np.isnan(se)
    # legs past the double range: no estimate, never a warning
    big = np.array([[1.5e308, 0.0, 0.0]] * 4)
    closed_big = np.array([[True, True, False]] * 4)
    excess_big = big - big.mean(axis=1, keepdims=True)
    t, se = _pooled_hac(excess_big, big, closed_big, cross_mean_pooled_mean(big, closed_big), 1)
    assert np.isnan(t) and np.isnan(se)


def test_cross_mean_pooled_hac_t_is_the_exact_mean_over_its_se():
    # A dispersed partial pool with a few full bars: the SE agrees with the rounded rows' read to
    # rounding, and t IS the exact pooled mean over it — the panel's mean_ret / hac_se.
    rng = np.random.RandomState(20)
    raw = rng.normal(0.0005, 0.02, (200, 6))
    closed = rng.rand(200, 6) < 0.3
    closed[::17] = True
    excess = raw - raw.mean(axis=1, keepdims=True)
    mean = cross_mean_pooled_mean(raw, closed)
    bars = np.nonzero(closed)[0]
    for h in (1, 3, 7):
        t, se = _pooled_hac(excess, raw, closed, mean, h)
        assert t == mean / se
        assert se == pytest.approx(newey_west_mean(excess[closed], bars, h)[1], rel=1e-12)
    # fewer than two rows, or no horizon: no estimate
    one = np.zeros_like(closed)
    one[5, 2] = True
    assert all(np.isnan(_pooled_hac(excess, raw, one, mean, 1)))
    assert all(np.isnan(_pooled_hac(excess, raw, closed, mean, 0)))


def test_benchmark_regression_recovers_a_planted_beta_and_alpha():
    rng = np.random.RandomState(7)
    bench = rng.randn(400) * 0.01
    raw = 0.002 + 2.0 * bench + rng.randn(400) * 0.001
    blk = benchmark_regression(raw, bench)
    assert blk["reason"] is None and blk["n"] == 400
    # the exact numpy reference, then the planted values within noise
    cv = float(np.cov(raw, bench, ddof=1)[0, 1])
    assert blk["beta"] == pytest.approx(cv / np.var(bench, ddof=1))
    assert blk["beta"] == pytest.approx(2.0, abs=0.05)
    assert blk["alpha"] == pytest.approx(0.002, abs=0.001)
    assert 0.9 < blk["r2"] <= 1.0
    # the identity a reader re-checks on the exactly rounded leg means: alpha + beta·mean(bench)
    # == mean(raw), to the rounding of one product and one difference
    assert blk["alpha"] + blk["beta"] * exact_mean(bench) == pytest.approx(
        exact_mean(raw), abs=1e-17
    )


def test_benchmark_regression_filters_pairs_never_legs():
    # A NaN in EITHER leg drops the PAIR — filtering each leg's finite subset independently
    # would silently mis-pair rows, which is why the function takes the raw aligned columns.
    raw = np.array([0.01, np.nan, 0.03, 0.04, 0.05])
    bench = np.array([0.02, 0.02, np.nan, 0.01, 0.03])
    blk = benchmark_regression(raw, bench)
    assert blk["n"] == 3
    assert blk == benchmark_regression(raw[[0, 3, 4]], bench[[0, 3, 4]])


def test_benchmark_regression_refusal_reasons():
    assert benchmark_regression(np.array([]), np.array([])) == {
        "n": 0,
        "beta": None,
        "alpha": None,
        "r2": None,
        "reason": "no_paired_observations",
    }
    # an unbenchmarked pool IS the all-NaN bench leg — same refusal, by construction
    unbenched = benchmark_regression(np.array([0.01, 0.02]), np.array([np.nan, np.nan]))
    assert unbenched["n"] == 0 and unbenched["reason"] == "no_paired_observations"
    thin = benchmark_regression(np.array([0.01, 0.02]), np.array([0.03, 0.01]))
    assert thin["n"] == 2 and thin["reason"] == "insufficient_observations"
    flat = benchmark_regression(np.array([0.01, 0.02, 0.03]), np.array([0.02, 0.02, 0.02]))
    assert flat["reason"] == "no_benchmark_variation" and flat["beta"] is None


def test_benchmark_regression_zero_raw_variance_is_nan_r2_not_a_refusal():
    # A constant raw leg still has a real (zero) beta and a real alpha; only r2 has nothing to
    # explain — NaN → null, the computed-as-undefined arm, not a declined-to-compute reason.
    blk = benchmark_regression(np.array([0.01, 0.01, 0.01]), np.array([0.01, 0.02, 0.03]))
    assert blk["reason"] is None
    assert blk["beta"] == 0.0  # exactly: a constant leg co-varies with nothing
    assert blk["alpha"] == pytest.approx(0.01)
    assert np.isnan(blk["r2"])


def test_benchmark_regression_constant_decimal_legs_are_degenerate_exactly():
    # A constant DECIMAL leg's computed variance is rounding residue (~1e-35), not variation: a
    # constant bench leg declines (no beta out of dividing by it), a constant raw leg fits
    # beta == 0.0 exactly with alpha its mean.
    flat_bench = benchmark_regression(np.array([0.01, 0.03, 0.02, 0.05, 0.04]), np.full(5, 0.007))
    assert flat_bench["reason"] == "no_benchmark_variation" and flat_bench["beta"] is None
    flat_raw = benchmark_regression(np.full(5, 0.007), np.array([0.01, 0.03, 0.02, 0.05, 0.04]))
    assert flat_raw["reason"] is None and flat_raw["beta"] == 0.0
    assert flat_raw["alpha"] == 0.007 and np.isnan(flat_raw["r2"])  # the leg's exact mean


@pytest.mark.parametrize("k", [500, 262, -262, -500])
def test_benchmark_regression_variance_product_out_of_range_is_exact(k):
    # Both legs scaled by 2**k: every moment scales exactly, so beta and r2 keep their bits and
    # alpha scales exactly — while the variance PRODUCT under r2 sits near 2**(4k), outside the
    # double range. The direct form read r2 = NaN above it (inf / inf), an r2 off in its 9th
    # digit in the subnormal range (k = -262), and raised ZeroDivisionError out of the run once
    # the product underflowed to 0.0 (k = -500).
    rng = np.random.RandomState(4)
    raw = rng.randn(30)
    bench = 0.5 * raw + rng.randn(30)
    s = 2.0**k
    base = benchmark_regression(raw, bench)
    assert base["reason"] is None and 0.0 < base["r2"] < 1.0
    assert benchmark_regression(raw * s, bench * s) == {**base, "alpha": base["alpha"] * s}


def test_benchmark_regression_out_of_range_moments_decline_never_a_false_fit():
    # A bench leg past ~1e154 overflows its variance: `beta = cov / inf` read a FINITE 0.0 — a
    # fit that never happened, the raw mean as alpha beside it. A beta past the double range (a
    # wide raw leg on a narrow bench) read inf with reason None. Both decline, saying why.
    rng = np.random.RandomState(4)
    raw = rng.randn(30)
    bench = 0.5 * raw + rng.randn(30)
    for r, b in ((raw, bench * 1e160), (raw * 1e300, bench * 1e-10)):
        assert benchmark_regression(r, b) == {
            "n": 30,
            "beta": None,
            "alpha": None,
            "r2": None,
            "reason": "not_representable",
        }


def test_benchmark_regression_subnormal_variances_decline_never_a_false_fit():
    # Legs of ~1e-161 square into the SUBNORMAL range: the variance product underflowed to 0.0
    # and r2 raised ZeroDivisionError out of the run — and taken in an unbounded exponent range
    # instead, the lost bits would read r2 = 2.0 for 0.99. A bench variance not a normal double
    # (subnormal, or 0.0 off a VARYING leg — which read as "no_benchmark_variation") is no
    # divisor: not_representable. A raw leg's alone nulls r2, while beta and alpha — never
    # divided by it — keep their exactly scaled bits.
    rng = np.random.RandomState(152)
    raw = rng.randn(8)
    bench = 0.99 * raw + 0.05 * rng.randn(8)
    declined = {"n": 8, "beta": None, "alpha": None, "r2": None, "reason": "not_representable"}
    s = 2.0**-536
    assert benchmark_regression(raw * s, bench * s) == declined
    vanishing = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 3.0]) * 1e-170
    assert benchmark_regression(raw, vanishing) == declined
    fit = benchmark_regression(raw, bench)
    assert 0.99 < fit["r2"] < 1.0
    tiny_raw = benchmark_regression(raw * s, bench)
    assert np.isnan(tiny_raw["r2"]) and tiny_raw["reason"] is None
    assert (tiny_raw["beta"], tiny_raw["alpha"]) == (fit["beta"] * s, fit["alpha"] * s)


# ---- timing_summary: path-timing medians (evidence-only) ---------------------


def test_timing_summary_hand_computed_medians_over_each_finite_subset():
    btp = np.array([1.0, 3.0, np.nan, 5.0])  # one path never touched positive — survivors only
    btt = np.array([0.0, 2.0, 4.0, 6.0])
    assert timing_summary(btp, btt) == {
        "n_to_positive": 3,
        "median_bars_to_positive": 3.0,
        "n_to_trough": 4,
        "median_bars_to_trough": 3.0,
    }


def test_timing_summary_empty_subsets_are_null_never_zero():
    blk = timing_summary(np.array([np.nan, np.nan]), np.array([]))
    assert blk["n_to_positive"] == 0 and np.isnan(blk["median_bars_to_positive"])
    assert blk["n_to_trough"] == 0 and np.isnan(blk["median_bars_to_trough"])


# ---- episode-bootstrap CI (evidence-only) ------------------------------------


def test_episode_bootstrap_deterministic_and_row_order_invariant():
    rng = np.random.RandomState(7)
    bars = np.sort(rng.choice(np.arange(0, 400, 7), size=40, replace=False))
    rets = rng.randn(40) * 0.02 + 0.005
    a = episode_bootstrap_ci(rets, bars, 5)
    b = episode_bootstrap_ci(rets, bars, 5)
    assert a == b  # same pool → identical block; the wall clock has no say
    perm = rng.permutation(40)
    c = episode_bootstrap_ci(rets[perm], bars[perm], 5)
    assert a == c  # content-seeded on the SORTED pool: row order is irrelevant
    assert a["reason"] is None and a["n_boot"] == 2000 and a["ci_level"] == 0.95
    assert a["ci_lo"] < a["ci_hi"] and a["boot_se"] > 0


def test_episode_bootstrap_constant_pool_is_degenerate():
    bars = np.arange(0, 60, 10)  # 6 disjoint episodes at h=5
    out = episode_bootstrap_ci(np.full(6, 0.01), bars, 5)
    assert out["n_episodes"] == 6
    assert out["ci_lo"] == pytest.approx(0.01) and out["ci_hi"] == pytest.approx(0.01)
    assert out["boot_se"] == pytest.approx(0.0)


def test_episode_bootstrap_refuses_thin_pools_with_a_reason():
    empty = episode_bootstrap_ci(np.array([]), np.array([], dtype=np.int64), 5)
    assert empty["reason"] == "no_observations" and empty["n_episodes"] == 0
    assert empty["ci_lo"] is None and empty["boot_se"] is None and empty["n_boot"] == 0
    four = episode_bootstrap_ci(np.full(4, 0.01), np.arange(0, 40, 10), 5)
    assert four["reason"] == "insufficient_episodes" and four["n_episodes"] == 4
    assert four["ci_lo"] is None and four["n_boot"] == 0


def test_episode_bootstrap_dense_chain_is_one_episode():
    # An every-bar firing with h > 1 transitively chains into ONE episode: there is no resampling
    # distribution over one exchangeable unit, and the block says so instead of minting one.
    out = episode_bootstrap_ci(np.random.RandomState(0).randn(100), np.arange(100), 5)
    assert out["n_episodes"] == 1 and out["reason"] == "insufficient_episodes"


def test_episode_bootstrap_half_open_tie_rule():
    # [0,5) and [5,10) do NOT merge (half-open windows, same rule as episode_stats); [0,5) and
    # [4,9) do.
    touching = episode_bootstrap_ci(np.full(6, 0.01), np.arange(0, 30, 5), 5)
    assert touching["n_episodes"] == 6
    chained = episode_bootstrap_ci(np.full(6, 0.01), np.arange(0, 24, 4), 5)
    assert chained["n_episodes"] == 1


def test_episode_bootstrap_ci_brackets_the_mean_on_a_disjoint_pool():
    # All-disjoint episodes reduce the episode bootstrap to an iid bootstrap of the mean: the
    # percentile CI brackets the sample mean and boot_se ≈ sd/√n (loose — Monte Carlo noise).
    rng = np.random.RandomState(3)
    n = 200
    rets = rng.randn(n) * 0.03 + 0.004
    out = episode_bootstrap_ci(rets, np.arange(n) * 10, 5)
    m = float(np.mean(rets))
    assert out["n_episodes"] == n
    assert out["ci_lo"] < m < out["ci_hi"]
    assert out["boot_se"] == pytest.approx(float(np.std(rets, ddof=1)) / np.sqrt(n), rel=0.25)


def test_subperiod_edges_and_means_hand_computed():
    assert list(subperiod_edges(9)) == [0, 3, 6, 9]
    assert list(subperiod_edges(10)) == [0, 3, 6, 10]  # linspace int-cast — cscv's edge rule
    edges = subperiod_edges(9)
    rets = np.array([0.01, 0.02, 0.03, 0.04])
    bars = np.array([0, 2, 3, 8])  # boundary bar 3 belongs to the SECOND era ([3, 6))
    segs = subperiod_means(rets, bars, edges)
    assert [s["n"] for s in segs] == [2, 1, 1]
    assert segs[0]["mean_ret"] == pytest.approx(0.015)
    assert segs[1]["mean_ret"] == pytest.approx(0.03)
    assert sum(s["n"] for s in segs) == len(rets)  # no purging: every closed row has ONE era
    empty = subperiod_means(np.array([]), np.array([], dtype=np.int64), edges)
    assert all(s["n"] == 0 and s["mean_ret"] is None for s in empty)


# ---- determinism under tied-row order + degenerate-pool moment gating --------------------------


def test_newey_west_tied_bar_rows_permute_to_identical_estimate():
    # Tied entry bars (the basket pooled path) must not let the caller's row order pick the fp
    # accumulation sequence: the canonical (bar, ret) lexsort makes the estimate BIT-identical.
    rets = np.array([0.02, 0.03, -0.01, 0.015, 0.005])
    bars = np.array([0, 0, 0, 7, 7])
    perm = np.array([2, 0, 4, 1, 3])
    a = newey_west_mean(rets, bars, h=10)
    b = newey_west_mean(rets[perm], bars[perm], h=10)
    assert a == b  # exact equality, not approx — same bytes


def test_episode_bootstrap_tied_bar_order_invariance():
    # The content-keyed seed hashes the sorted pool bytes; tied-bar rows must land in ONE
    # canonical order or permuting same-bar members would change the draw stream and the CI.
    rng = np.random.RandomState(4)
    bars = np.repeat(np.arange(8) * 30, 2)  # 8 well-separated bars, 2 members each
    rets = rng.normal(0.01, 0.02, size=bars.size)
    perm = rng.permutation(bars.size)
    a = episode_bootstrap_ci(rets, bars, 5)
    b = episode_bootstrap_ci(rets[perm], bars[perm], 5)
    assert a == b


def test_episode_panels_canonical_under_tied_row_order():
    # Three same-bar rows across two targets with fp-noncommutative returns: every permutation
    # of the frame's rows must yield identical episode_stats / episode_ledger / episode_profile
    # (the panels sort stably by (entry_time, target), never by frame order).
    t0 = pd.Timestamp("2021-01-04")
    base = pd.DataFrame(
        {
            "entry_time": [t0, t0, t0, t0 + pd.Timedelta(days=40)],
            "exit_time": [t0 + pd.Timedelta(days=5)] * 3 + [t0 + pd.Timedelta(days=45)],
            "ret": [0.1, 0.2, 0.3, -0.05],
            "target": ["A", "B", "C", "A"],
            "mae": [-0.02, -0.01, -0.03, -0.04],
            "mfe": [0.05, 0.04, 0.06, 0.01],
        }
    )
    want_stats = episode_stats(base)
    want_ledger = episode_ledger(base)
    want_profile = episode_profile(base)
    for perm in ([2, 1, 0, 3], [1, 2, 3, 0], [3, 2, 1, 0]):
        shuffled = base.iloc[list(perm)].reset_index(drop=True)
        assert episode_stats(shuffled) == want_stats
        assert episode_ledger(shuffled) == want_ledger
        assert episode_profile(shuffled) == want_profile


def test_pool_moments_near_constant_pool_moments_are_null_and_silent():
    # A pool whose spread is below ~10 ulps of its mean has no usable dispersion: skew/kurtosis
    # emit null instead of scipy's precision-loss RuntimeWarning + unreliable numbers. The gate
    # is about the moment estimators' numerical reliability: std_ret stays as computed.
    import warnings

    rets = 1e8 + np.linspace(0.0, 1e-10, 40)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = pool_moments(rets)
    assert np.isnan(s["skewness"]) and np.isnan(s["kurtosis"])
    assert np.isfinite(s["std_ret"])
    # An honest pool still gets its moments — variation around a near-zero mean is NOT degenerate.
    honest = pool_moments(np.random.RandomState(2).normal(0.0, 0.01, 60))
    assert np.isfinite(honest["skewness"]) and np.isfinite(honest["kurtosis"])


def test_pool_moments_shape_is_taken_about_the_pools_own_offset():
    # The review's pool: 1 + k·eps, k = 0..21, is symmetric — its third central moment is
    # exactly 0 — but about scipy's own rounded mean the deviations carried that mean's
    # rounding, and skewness read 0.666 (kurtosis 1.9136) where its translate by −1 reads 0
    # (1.7950). About the pool's range midpoint the deviations are exact: the moments are the
    # translate's.
    eps = np.finfo(float).eps
    pool = 1.0 + np.arange(22) * eps
    s, translate = pool_moments(pool), pool_moments(pool - 1.0)
    assert s["skewness"] == translate["skewness"] == 0.0
    assert s["kurtosis"] == translate["kurtosis"] == pytest.approx(1.7950310559, rel=1e-10)
    # pools within 17 ulps of 1 tripped scipy's "Precision loss" warning (an exception under
    # -W error, which this suite runs): about the offset they read their exact-rational moments
    for ulps in ([0, 17, 14], [0, 0, 3, 15]):
        rets = 1.0 + np.array(ulps) * eps
        assert not _near_constant(rets, exact_mean(rets))  # the premise
        f = [Fraction(x) for x in rets]
        m = sum(f) / len(f)
        m2, m3 = (sum((x - m) ** p for x in f) / len(f) for p in (2, 3))
        assert pool_moments(rets)["skewness"] == pytest.approx(
            float(m3) / float(m2) ** 1.5, rel=1e-14
        )


def test_pool_std_is_taken_about_the_pools_own_offset():
    # The review's case: about the rounded raw mean, [2**52, 2**52 + 1]'s deviations read [0, 1]
    # — the mean rounds onto a member — and std_ret read 1.0 for the exact sqrt(0.5). About the
    # pool's range midpoint the deviations are exact. The baseline and the bootstrap's SE take the
    # same read (`_ddof1_std`).
    from seikan.analysis._pools import _ddof1_std

    assert pool_moments(np.array([2.0**52, 2.0**52 + 1.0]))["std_ret"] == math.sqrt(0.5)
    b = 2.0**50
    clustered = np.array([b, b + 0.25, b + 0.25])
    assert float(np.std(clustered, ddof=1)) == 0.25
    assert _ddof1_std(clustered) == pytest.approx(math.sqrt(1.0 / 48.0), rel=1e-15)


def test_benchmark_regression_moments_are_taken_about_each_legs_offset():
    # Legs clustered at 2**40 (spreads of 31 and 62 ulps, past the near-constant gate): about
    # their rounded raw means the deviations carried each mean's rounding and the fit read beta
    # 1.9984 and r2 0.9988 for an exact line of slope 2. About each leg's range midpoint the fit
    # is the exact-rational one; alpha stays on the raw means.
    level = 2.0**40
    ulp = float(np.spacing(level))
    bench = level + ulp * np.array([0.0, 31.0, 31.0])
    raw = level + ulp * np.array([0.0, 62.0, 62.0])
    blk = benchmark_regression(raw, bench)
    assert blk["reason"] is None and blk["beta"] == 2.0 and blk["r2"] == 1.0
    assert blk["alpha"] == exact_mean(raw) - 2.0 * exact_mean(bench)


def test_benchmark_regression_alpha_reads_the_exact_leg_means():
    # The review's legs: the bench leg at 2**50 + 8i, the raw leg 1 above it — beta exactly 1,
    # and the panel's exactly rounded leg means exactly 1 apart. alpha centred on the pairwise
    # means instead, whose rounding at the level's ulp (0.25) put them 0.75 apart: alpha 0.75
    # beside a mean_ret_raw − mean_ret_bench of 1, an identity the panel itself contradicted.
    bench = 2.0**50 + 8.0 * np.arange(31)
    raw = bench + 1.0
    assert float(np.mean(raw)) - float(np.mean(bench)) == 0.75  # the premise
    blk = benchmark_regression(raw, bench)
    assert blk["reason"] is None and blk["beta"] == 1.0 and blk["r2"] == 1.0
    assert blk["alpha"] == 1.0 == exact_mean(raw) - exact_mean(bench)
    assert blk["alpha"] + blk["beta"] * exact_mean(bench) == exact_mean(raw)


def test_benchmark_regression_r2_never_exceeds_one():
    # An exactly linear pair explains all of its variance, and the computed ratio's rounding read
    # an r2 past 1 (by up to 8.9e-16) on about a third of `raw = k·bench + c` pools — a share of
    # the variance explained that no fit has. Clipped to 1, the `rolling_corr` rule.
    rng = np.random.RandomState(18)
    on_bound = 0
    for _ in range(300):
        bench = np.round(rng.randn(int(rng.randint(3, 40))) * 0.02, 6)
        raw = float(rng.choice([3.0, -3.0, 0.5, 1.7])) * bench + float(rng.choice([0.0, 7.0]))
        blk = benchmark_regression(raw, bench)
        assert blk["reason"] is None and 1.0 - 1e-14 < blk["r2"] <= 1.0
        on_bound += blk["r2"] == 1.0
    assert on_bound > 0


def test_benchmark_regression_near_constant_bench_declines_a_varying_raw_leg_is_fitted():
    # Spread within a few ulps of the mean: the bench leg has no resolvable variation to divide
    # by — an arbitrary beta (1.9e16 before) — and the fit declines. The raw leg is no divisor,
    # and its deviations about its offset are exact however small: it is fitted. Its
    # near-constant shortcut read the review's exact line raw = 1 + bench over
    # bench = eps·[−4, −2, 0, 2, 4] as beta 0 and r2 null beside reason None, for a slope of 1.
    near = 0.007 + np.array([0.0, 1.0, 0.0, 2.0, 1.0]) * np.spacing(0.007)
    assert np.ptp(near) > 0.0
    other = np.array([0.01, 0.03, 0.02, 0.05, 0.04])
    flat_bench = benchmark_regression(other, near)
    assert flat_bench["reason"] == "no_benchmark_variation" and flat_bench["beta"] is None
    bench = np.finfo(float).eps * np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    line = benchmark_regression(1.0 + bench, bench)
    assert line == {"n": 5, "beta": 1.0, "alpha": 1.0, "r2": 1.0, "reason": None}
    # the near-constant raw leg reads the exact-rational fit
    r, b = [Fraction(x) for x in near], [Fraction(x) for x in other]
    mr, mb = sum(r) / 5, sum(b) / 5
    cov = sum((x - mr) * (y - mb) for x, y in zip(r, b, strict=True))
    var_r, var_b = sum((x - mr) ** 2 for x in r), sum((y - mb) ** 2 for y in b)
    fit = benchmark_regression(near, other)
    assert fit["reason"] is None
    assert fit["beta"] == pytest.approx(float(cov / var_b), rel=1e-14)
    assert fit["r2"] == pytest.approx(float(cov * cov / (var_r * var_b)), rel=1e-14)


@pytest.mark.parametrize("c", [0.1, 0.007, 0.3])
def test_constant_pool_bootstrap_is_degenerate_exactly(c):
    # Every resample of a constant pool has the same mean: the interval has zero width and the
    # SE is 0 — not the ulp-scale spread of the resampled means' rounding — and it sits on the
    # pool's own mean, the `mean_ret` the cell reports beside it: `c` itself, where the computed
    # np.mean of these pools reads an ulp off it.
    bars = np.array([0, 1, 2, 10, 20, 21, 30, 40, 41, 42, 43, 50, 60])
    rets = np.full(bars.size, c)
    ci = episode_bootstrap_ci(rets, bars, 3)
    assert ci["reason"] is None and ci["n_boot"] > 0
    assert float(np.mean(rets)) != c  # the premise
    assert ci["ci_lo"] == ci["ci_hi"] == pool_moments(rets)["mean_ret"] == c
    assert ci["boot_se"] == 0.0


@pytest.mark.parametrize("c", [1e308, -1e308, 3e307])
def test_constant_pool_bootstrap_past_the_range_is_the_value(c):
    # Eight equal returns whose sum leaves the double range: the degenerate interval read the
    # overflowed np.mean — a null interval beside boot_se 0.0, no reason. Every value is equal,
    # so the value IS the exact arithmetic mean: the interval is c, boot_se 0.0, nothing null.
    ci = episode_bootstrap_ci(np.full(8, c), np.arange(8) * 10, 5)
    assert ci["n_episodes"] == 8 and ci["n_boot"] > 0 and ci["reason"] is None
    assert ci["ci_lo"] == ci["ci_hi"] == c and ci["boot_se"] == 0.0


def test_ddof1_std_of_a_non_finite_pool_is_nan_never_a_degenerate_zero():
    # No caller reaches it today (the moment reads filter, the bootstrap nulls its non-finite
    # draws first), so it is pinned here: `inf == inf` would pass the all-equal test, and an
    # infinite pool is no zero-dispersion one.
    from seikan.analysis._pools import _ddof1_std

    for v in ([np.inf, np.inf], [-np.inf, -np.inf], [np.inf, 1.0], [np.nan, 1.0]):
        assert np.isnan(_ddof1_std(np.array(v))), v
    assert _ddof1_std(np.array([0.1, 0.1, 0.1])) == 0.0


def test_varying_pool_whose_variance_leaves_the_range_has_null_dispersion_never_zero():
    # Deviations of ~1e-170 square below the smallest subnormal: a VARYING pool's computed
    # variance is exactly 0.0 and std_ret / boot_se read 0.0 — the answer reserved for an
    # all-equal pool (or bit-identical draws). At 2**-536 the squares are SUBNORMAL and the
    # variance has lost its bits (std_ret read 5.88e-162 for a pool whose std is 5.68e-162). An
    # overflowed one read inf. None is a dispersion: NaN → null.
    tiny = np.array([0.0, 1e-170, 2e-170, 3e-170])
    assert np.isnan(pool_moments(tiny)["std_ret"])
    assert np.isnan(baseline_summary(tiny)["std_ret"])
    ci = episode_bootstrap_ci(np.arange(10) * 1e-170, np.arange(10) * 10, 5)
    # the interval stands; the null SE says why it is null
    assert ci["reason"] == "not_representable" and ci["ci_lo"] < ci["ci_hi"]
    assert np.isnan(ci["boot_se"])
    v = np.random.RandomState(0).standard_t(4, 200)
    assert np.isnan(pool_moments(v * 2.0**-536)["std_ret"])
    assert np.isnan(baseline_summary(v * 2.0**-536)["std_ret"])
    assert np.isnan(pool_moments(np.array([1e200, -1e200, 3e200]))["std_ret"])
    assert np.isnan(baseline_summary(np.array([1e200, -1e200, 3e200]))["std_ret"])
    # the all-equal pool keeps its exact 0.0 at the same scale, and a normal variance its bits
    assert pool_moments(np.full(4, 1e-170))["std_ret"] == 0.0
    std = pool_moments(v)["std_ret"]
    assert pool_moments(v * 2.0**-500)["std_ret"] == std * 2.0**-500


def test_bootstrap_se_of_overflowed_draws_is_null_never_a_degenerate_zero():
    # Twelve returns near 1e308: every resampled mean's episode sum overflows, so every draw is
    # +inf — and `inf == inf` passed the bit-identical-draws test: boot_se 0.0, a degenerate
    # distribution, beside a null interval. Draws past the double range have no spread: null.
    ci = episode_bootstrap_ci(np.linspace(0.5e308, 1e308, 12), np.arange(12) * 10, 5)
    assert ci["n_episodes"] == 12 and np.isnan(ci["ci_lo"]) and np.isnan(ci["ci_hi"])
    assert np.isnan(ci["boot_se"]) and ci["reason"] == "not_representable"


def test_bootstrap_se_of_draws_whose_variance_overflows_is_null_with_its_reason():
    # Returns of ±1e155: every episode sum and draw is finite (the interval stands), but the
    # draws' squared deviations leave the double range — boot_se null, and the block says why.
    ci = episode_bootstrap_ci(np.array([1e155, -1e155] * 20), np.arange(40) * 10, 5)
    assert ci["n_episodes"] == 40 and ci["ci_lo"] < 0.0 < ci["ci_hi"]
    assert np.isnan(ci["boot_se"]) and ci["reason"] == "not_representable"


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_bootstrap_interval_over_an_overflowed_episode_is_null_never_one_sided(sign):
    # ONE episode's two rows sum past the double range and seven are ordinary: every draw that
    # resamples it is ±inf and ranks at one end, so the percentile read one FINITE bound over
    # draws that left the range (ci_lo −0.28125 above, ci_hi 1.34375 below) beside a null one. A
    # quantile of such draws is no interval: both bounds null, like boot_se.
    rets = np.array([1.2e308, 1.2e308, 1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25])
    rets[:2] *= sign
    bars = np.array([0, 1, 20, 40, 60, 80, 100, 120, 140])
    ci = episode_bootstrap_ci(rets, bars, 5)
    assert ci["n_episodes"] == 8 and ci["n_boot"] > 0 and ci["reason"] == "not_representable"
    assert np.isnan(ci["ci_lo"]) and np.isnan(ci["ci_hi"]) and np.isnan(ci["boot_se"])
    # the same pool one episode short of the overflow keeps a finite, ordered interval
    ok = episode_bootstrap_ci(rets[2:], bars[2:], 5)
    assert ok["n_episodes"] == 7 and ok["ci_lo"] < ok["ci_hi"] and ok["boot_se"] > 0.0
    assert ok["reason"] is None


def test_bootstrap_episode_sums_are_exactly_rounded():
    # Eight separated episodes of [1, −2**54, −1, 2**54] on consecutive bars (h = 2 chains each
    # into one): every episode sums to exactly 0, and so does the pool. Summed in floating
    # point (np.add.reduceat) each episode read 1, and the bootstrap a zero-width interval at
    # 0.25 with boot_se 0.0 beside a mean_ret of 0 — excluding the mean it brackets.
    rets = np.array([1.0, -(2.0**54), -1.0, 2.0**54] * 8)
    bars = (np.arange(8)[:, None] * 10 + np.arange(4)[None, :]).ravel()
    assert float(np.add.reduceat(rets, [0, 4])[0]) == 1.0  # the premise: the old segment sum
    ci = episode_bootstrap_ci(rets, bars, 2)
    assert ci["reason"] is None and ci["n_episodes"] == 8
    assert ci["ci_lo"] == ci["ci_hi"] == pool_moments(rets)["mean_ret"] == 0.0
    assert ci["boot_se"] == 0.0


def test_degenerate_bootstrap_distribution_has_exactly_zero_se():
    # A pool that varies, but whose episodes all carry the same (sum, count): every draw's mean
    # is the same bits, so the distribution is degenerate — an SE of rounding residue (~3e-17,
    # a mean / boot_se of ~7e15) would be the t_hac pathology again.
    bars = np.array([[10 * k, 10 * k + 1] for k in range(12)]).ravel()
    rets = np.tile([0.1, 0.3], 12)
    ci = episode_bootstrap_ci(rets, bars, 3)
    assert ci["reason"] is None and ci["n_episodes"] == 12
    assert ci["ci_lo"] == ci["ci_hi"]
    assert ci["boot_se"] == 0.0


def test_bootstrap_sums_the_pool_values_on_the_draw_stream_rets_alone_gives():
    # `rets_pool` is what the episode sums ADD; the canonical order, the seed digest and the
    # episodes stay on `rets`. Doubling every pool value doubles every draw's mean EXACTLY (a
    # power-of-two scale commutes with rounding), so the interval and the SE double bit for bit
    # — the draws are the ones `rets` alone gives — and passing `rets` itself IS the default.
    rng = np.random.RandomState(5)
    bars = np.repeat(np.arange(12) * 9, 3)  # 12 disjoint bars at h=5, three members on each
    rets = rng.normal(0.01, 0.02, size=bars.size)
    plain = episode_bootstrap_ci(rets, bars, 5)
    assert episode_bootstrap_ci(rets, bars, 5, rets_pool=rets) == plain
    doubled = episode_bootstrap_ci(rets, bars, 5, rets_pool=2.0 * rets)
    assert doubled["n_episodes"] == plain["n_episodes"] == 12
    assert doubled["ci_lo"] == 2.0 * plain["ci_lo"] and doubled["ci_hi"] == 2.0 * plain["ci_hi"]
    assert doubled["boot_se"] == 2.0 * plain["boot_se"] > 0.0
    # the values ride the rows: any row order, the pool's tied zeros included, is one block
    pool = np.where(bars < 50, 0.0, rets)
    want = episode_bootstrap_ci(rets, bars, 5, rets_pool=pool)
    for _ in range(3):
        perm = rng.permutation(bars.size)
        assert episode_bootstrap_ci(rets[perm], bars[perm], 5, rets_pool=pool[perm]) == want
    # rows tied on (bar, ret) may carry DIFFERENT pool values (a library caller's — the
    # runner's never do) in the caller's order: each episode's sum is exact, so no permutation
    # moves a bit. Each bar below holds 1e16, 1.0 and -1e16 in one of their six orders, where a
    # running sum would show the order: (1e16 + 1.0) - 1e16 is 0.0, (1e16 - 1e16) + 1.0 is 1.0.
    big, one = 1e16, 1.0
    spread = np.array(
        [
            [big, one, -big],
            [big, -big, one],
            [one, big, -big],
            [one, -big, big],
            [-big, big, one],
            [-big, one, big],
        ]
    ).ravel()
    six_bars = bars[:18]
    tied = np.repeat(0.01 * np.arange(6), 3)  # every row of a bar ties on ret
    want = episode_bootstrap_ci(tied, six_bars, 5, rets_pool=spread)
    for _ in range(5):
        perm = rng.permutation(18)
        got = episode_bootstrap_ci(tied[perm], six_bars[perm], 5, rets_pool=spread[perm])
        assert got == want
    with pytest.raises(ValueError, match="row-aligned"):
        episode_bootstrap_ci(rets, bars, 5, rets_pool=pool[:-1])


def test_bootstrap_over_full_bar_zeros_is_exactly_zero():
    # A `cross_mean` basket whose every bar is full: each bar's excesses sum to zero by
    # identity, its rounded rows to residue. Summing the pool values (0.0 on every full-bar row)
    # the interval sits at exactly 0.0 with an SE of exactly 0.0 — summing the rows themselves
    # it sat at the residue.
    bars = np.repeat(np.arange(6) * 10, 3)
    rets = np.tile([-5.333333333333333, -3.333333333333333, 8.666666666666668], 6)
    assert float(np.sum(rets[:3])) != 0.0  # the premise: the rounded excesses do not cancel
    ci = episode_bootstrap_ci(rets, bars, 5, rets_pool=np.zeros(rets.size))
    assert ci["reason"] is None and ci["n_episodes"] == 6 and ci["n_boot"] == 2000
    assert ci["ci_lo"] == ci["ci_hi"] == 0.0 and ci["boot_se"] == 0.0
    assert episode_bootstrap_ci(rets, bars, 5)["ci_lo"] != 0.0


def test_constant_pool_has_no_bucket_monotonicity():
    # A constant pool's bucket means are its value, every one (their pairwise sums rounded apart
    # with the bucket sizes): no rank signal.
    rng = np.random.RandomState(3)
    trades = pd.DataFrame({"ret": np.full(45, 0.7), "feat": rng.normal(0.0, 1.0, 45)})
    out = cell_conditional_buckets(trades, ["feat"])
    assert out["conditional_buckets"]["feat"]["reason"] is None
    assert {b["mean_ret"] for b in out["conditional_buckets"]["feat"]["buckets"]} == {0.7}
    assert "feat" not in out["bucket_monotonicity"]


#: The same six returns in four orders whose floating-point means rise: pandas' grouped mean reads
#: -9.3e-18, -4.6e-18, 0 and 4.6e-18, numpy's over each order twice 0, 2.3e-18, 4.6e-18 and
#: 6.9e-18. Every exact mean is 0.
_ZERO_SUM_ORDERS = (
    (-0.1, 0.2, 0.3, 0.1, -0.3, -0.2),
    (-0.3, -0.2, -0.1, 0.3, 0.1, 0.2),
    (-0.3, -0.2, -0.1, 0.2, 0.1, 0.3),
    (0.3, 0.1, 0.2, -0.3, -0.2, -0.1),
)


@pytest.mark.parametrize("targets", [("A",), ("A", "B")], ids=["one_target", "ordinals"])
def test_buckets_holding_the_same_returns_read_the_same_mean(targets):
    # Four buckets of the same six returns, each in another row order: the rounded means rose
    # with the order and ranked as a perfectly monotone edge (rho 1.0) on the one-target kernel
    # and on the ordinal arm alike. The bucket means are exactly rounded: 0.0 each, no rank
    # signal.
    rets = np.concatenate(_ZERO_SUM_ORDERS)
    rounded = (
        pd.Series(rets).groupby(np.repeat(np.arange(4), 6)).mean().tolist()
        if len(targets) == 1
        else [float(np.mean(np.tile(order, 2))) for order in _ZERO_SUM_ORDERS]
    )
    assert rounded == sorted(set(rounded))  # the premise: four rising means
    trades = pd.DataFrame(
        {
            "target": np.repeat(targets, rets.size),
            "feat": np.tile(np.arange(rets.size, dtype=float), len(targets)),
            "ret": np.tile(rets, len(targets)),
        }
    )
    out = cell_conditional_buckets(trades, ["feat"])
    block = out["conditional_buckets"]["feat"]
    assert block["reason"] is None
    assert [b["n"] for b in block["buckets"]] == [6 * len(targets)] * 4
    assert [b["mean_ret"] for b in block["buckets"]] == [0.0] * 4
    assert out["bucket_monotonicity"] == {}


@pytest.mark.parametrize(("c", "n"), [(0.1, 3), (0.1, 25), (0.007, 5), (0.007, 40)])
def test_constant_decimal_pool_has_no_dispersion_to_divide_by(c, n):
    # The computed mean of a constant DECIMAL pool is not the constant (these (c, n) are the
    # cases where numpy's is not), so np.std read the rounding residue as ~1e-17 of dispersion
    # and the HAC divided by it: |t_hac| ~ 1e16 on a pool with no variation at all.
    rets = np.full(n, c)
    assert float(np.mean(rets)) != c  # the premise
    t, se = newey_west_mean(rets, np.arange(n), h=3)
    assert np.isnan(t) and np.isnan(se)
    assert pool_moments(rets)["std_ret"] == 0.0
    assert baseline_summary(rets)["std_ret"] == 0.0


def test_newey_west_near_constant_pool_is_null():
    # Spread at the rounding level of the mean (here a few ulps): the long-run variance is
    # residue, not dispersion — the same scale-relative gate the moment estimators use.
    rets = 0.01 + np.array([0.0, 1.0, 0.0, 2.0, 1.0, 0.0]) * np.spacing(0.01)
    t, se = newey_west_mean(rets, np.arange(6), h=2)
    assert np.isnan(t) and np.isnan(se)
    assert pool_moments(rets)["std_ret"] > 0.0  # genuinely non-constant: std stays as computed


def test_newey_west_overflowed_long_run_variance_is_null_never_a_zero_t():
    # Returns past ~1e154 overflow the long-run variance: `mean / inf` read a FINITE t of 0.0
    # beside an infinite SE — "no edge" minted from an overflow. Below the ceiling the direct
    # form is untouched: a power-of-two scale reproduces t exactly and scales the SE exactly.
    rets = np.random.RandomState(3).randn(40)
    bars = np.arange(40) * 10
    t, se = newey_west_mean(rets * 2.0**520, bars, 5)
    assert np.isnan(t) and np.isnan(se)
    # contiguous bars: the lag loop's overlapping pairs overflow too, silently
    t, se = newey_west_mean(rets * 2.0**520, np.arange(40), 5)
    assert np.isnan(t) and np.isnan(se)
    t0, se0 = newey_west_mean(rets, bars, 5)
    assert newey_west_mean(rets * 2.0**500, bars, 5) == (t0, se0 * 2.0**500)


def test_newey_west_underflowed_variance_is_null_never_a_crash():
    # Returns of ±3e-162 square into the bottom of the subnormal range: the long-run variance
    # over n = 20 rounded down to an SE of exactly 0.0, and `mean / se` raised ZeroDivisionError
    # out of the run. At 2**-520 the variance stayed subnormal-nonzero and t read off its lost
    # bits (-1.1031486149 for -1.1031486153). Below the normal range: no estimate, (NaN, NaN).
    t, se = newey_west_mean(np.array([3e-162, -3e-162] * 10), np.arange(20) * 10, 5)
    assert np.isnan(t) and np.isnan(se)
    rets = np.random.RandomState(3).randn(40)
    t, se = newey_west_mean(rets * 2.0**-520, np.arange(40) * 10, 5)
    assert np.isnan(t) and np.isnan(se)


def test_pool_moments_shape_reads_off_an_out_of_range_variance_power_are_null():
    # skewness = m3 / m2**1.5 and kurtosis = m4 / m2**2: at a 2**-266 scale (~1e-80) m2**2 and m4
    # are SUBNORMAL — kurtosis read 3.7180340 for 3.7172188 — and at 2**-350 m2**1.5 is too
    # (skewness -0.1399959974 for -0.1399960027). A moment read off lost bits is null; while
    # the power stays normal every bit survives the power-of-two scale.
    v = np.random.RandomState(0).standard_t(4, 200)
    base = pool_moments(v)
    near = pool_moments(v * 2.0**-266)
    assert near["skewness"] == base["skewness"] and np.isnan(near["kurtosis"])
    far = pool_moments(v * 2.0**-350)
    assert np.isnan(far["skewness"]) and np.isnan(far["kurtosis"])
    assert far["std_ret"] == base["std_ret"] * 2.0**-350  # the variance itself is still normal
    big = pool_moments(v * 2.0**250)
    assert (big["skewness"], big["kurtosis"]) == (base["skewness"], base["kurtosis"])


def test_pool_moments_shape_read_whose_moment_overflows_under_a_normal_power_is_null():
    # One deviation past ~5.6e102 cubes to inf while the 999 small rows keep m2**1.5 (~1e304) a
    # normal double: scipy's skewness read inf / finite = -inf, a non-finite value in the
    # in-memory summary (json_safe nulled it only on the way out). NaN, like every other read.
    out = pool_moments(np.array([-6e102] + [1e99] * 999))
    assert np.isnan(out["skewness"]) and np.isnan(out["kurtosis"])
    assert np.isfinite(out["std_ret"])


def test_pool_moments_exact_constant_pool_moments_are_null():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = pool_moments(np.full(30, 0.01))
    assert np.isnan(s["skewness"]) and np.isnan(s["kurtosis"])


# ---- level-stable CSCV scoring + the degradation slope's null-with-reason ----------------------


def test_cscv_level_heavy_returns_still_score():
    # A diff outcome on a series at level 1e9: the raw one-pass Σx² − n·mean² variance was pure
    # rounding garbage there (quantum ~100 against a true SS ~1e-4 — clamped to 0, every split
    # score NaN → "insufficient_data"). Shifted moments score it like any other grid.
    rng = np.random.RandomState(21)
    cells = _pbo_cells(rng, 800, 12)
    for c in cells:
        c["fwd_col"] = np.asarray(c["fwd_col"]) + 1e9
    out = cscv_pbo(cells, 800, ["t"], n_combos_declared=12)
    assert out["reason"] is None and out["n_splits"] == 70
    assert 0.0 <= out["pbo"] <= 1.0
    assert np.isfinite(out["lambda_mean"]) and np.isfinite(out["prob_oos_loss"])


def test_cscv_basket_equals_union_at_huge_level():
    # The basket==union identity is the exactness oracle for the shifted arithmetic: the two
    # code paths (per-member refs + exact cross-term expansion vs one column, one ref) are
    # algebraically equal, so they must agree even at level 1e9 where the raw form cancelled.
    rng = np.random.RandomState(9)
    T, h = 800, 1
    fwd_a = rng.normal(0.0, 0.01, T) + 1e9
    fwd_b = rng.normal(0.0, 0.01, T) + 1e9
    basket_cells, union_cells = [], []
    for k in range(6):
        bars = rng.choice(T - 2, 80, replace=False)
        ma = np.zeros(T)
        ma[bars[:40]] = 1.0
        mb = np.zeros(T)
        mb[bars[40:]] = 1.0
        basket_cells += [
            {"key": (k, "A"), "mask_col": ma, "fwd_col": fwd_a, "h": h},
            {"key": (k, "B"), "mask_col": mb, "fwd_col": fwd_b, "h": h},
        ]
        mu = np.zeros(T)
        mu[bars] = 1.0
        fwd_u = np.where(ma > 0, fwd_a, fwd_b)
        union_cells.append({"key": (k, "P"), "mask_col": mu, "fwd_col": fwd_u, "h": h})
    basket = cscv_pbo(basket_cells, T, ["A", "B"], mode="basket", n_combos_declared=6)
    union = cscv_pbo(union_cells, T, ["P"], n_combos_declared=6)
    assert basket["reason"] is None and union["reason"] is None
    for key in ("pbo", "lambda_mean", "oos_degradation_slope", "prob_oos_loss"):
        assert basket[key] == pytest.approx(union[key], rel=1e-9)


def _two_level_cells(mode_targets, T=400, n_noise=4, seed=31):
    """One candidate (combo 0) whose returns are 0.005 on the first half of the bars and 0.022
    on the second — constant within every split drawn from one half — beside noise candidates."""
    rng = np.random.RandomState(seed)
    level = np.where(np.arange(T) < T // 2, 0.005, 0.022)
    cells = []
    for k in range(1 + n_noise):
        for tg in mode_targets:
            m = np.zeros(T)
            m[rng.choice(T - 2, 120, replace=False)] = 1.0
            fwd = level if k == 0 else rng.normal(0.0, 0.01, T)
            cells.append({"key": (k, tg), "mask_col": m, "fwd_col": fwd, "h": 1})
    return cells


@pytest.mark.parametrize(("mode", "targets"), [("conjunction", ("t",)), ("basket", ("A", "B"))])
def test_cscv_constant_split_pool_is_never_ranked(mode, targets):
    # Combo 0's returns vary across the run but are constant within each half, so the S=8
    # split whose train blocks are the first half (and its test blocks, the second) scores a
    # constant pool. Shifted by the WHOLE column's mean, its reconstructed variance is rounding
    # residue — the Sharpe read ±1e7 there and outranked every honest candidate. A constant
    # pool has no dispersion: it drops out of that split's ranking like any undefined score.
    cells = _two_level_cells(targets)
    out = cscv_pbo(cells, 400, list(targets), mode=mode, n_combos_declared=5)
    assert out["reason"] is None and out["blocks"] == 8
    assert out["n_candidates_min"] == 4


def _split_tensors(r, b, s=4):
    """One candidate × one member, built the way ``cscv_pbo`` builds them: block moments of the
    returns shifted by the column's own mean, and the per-block extrema."""
    ref_v = float(r.mean())
    sh = r - ref_v
    moments = np.stack(
        [
            np.bincount(b, minlength=s).astype(float),
            np.bincount(b, weights=sh, minlength=s),
            np.bincount(b, weights=sh * sh, minlength=s),
        ]
    )[None, None]
    lo, hi = np.full(s, np.inf), np.full(s, -np.inf)
    np.minimum.at(lo, b, r)
    np.maximum.at(hi, b, r)
    return moments, np.array([[ref_v]]), np.stack([lo, hi])[None, None]


def test_cscv_split_score_reads_exact_constancy_per_split():
    # The column varies (0.1 in blocks 0-1, 0.3 in blocks 2-3), but a split over blocks {0, 1}
    # alone is a constant pool. Shifted by the WHOLE column's mean, with unequal block counts,
    # its reconstructed variance is residue that once scored a Sharpe of ~7.6e7; the per-block
    # extrema (and the rounding floor behind them) leave it undefined, while a split mixing the
    # two levels varies and scores.
    from seikan.analysis._cscv import _split_scores

    r = np.array([0.1] * 5 + [0.3] * 5)
    b = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
    moments, refs, extrema = _split_tensors(r, b)
    for mode in ("conjunction", "basket"):
        for blocks in ([0, 1], [2, 3]):
            assert np.isnan(_split_scores(moments, refs, extrema, mode, np.array(blocks))).all()
        assert np.isfinite(_split_scores(moments, refs, extrema, mode, np.array([0, 2]))).all()


def test_cscv_split_score_nulls_dispersion_below_rounding():
    # One ulp of spread inside a split pool is below what the shifted accumulation can resolve:
    # the centered sum of squares is cancellation residue, and the split reads NaN like an
    # exactly constant one (the extrema alone would let it through as "varying").
    from seikan.analysis._cscv import _split_scores

    r = np.array([0.005] * 5 + [0.022] * 5)
    r[1] = np.nextafter(0.005, 1.0)
    b = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
    moments, refs, extrema = _split_tensors(r, b)
    for mode in ("conjunction", "basket"):
        assert np.isnan(_split_scores(moments, refs, extrema, mode, np.array([0, 1]))).all()
        assert np.isfinite(_split_scores(moments, refs, extrema, mode, np.array([0, 2]))).all()


def test_cscv_split_score_of_an_overflowed_sd_is_nan_never_zero():
    # Basket pooling through the exact cross-term expansion: two members with one observation
    # each in block 0 (X and Y) and a 0.0 in block 1, so each column's own mean is half its
    # block-0 value. Every expansion term and the rounding floor's sum stay finite, but their
    # total — the pooled sum of squares (X − Y)²/2 — overflows: sd = inf and `mean / inf` scored
    # a FINITE 0.0 Sharpe that entered the split's ranking. NaN drops it from the split instead.
    from seikan.analysis._cscv import _split_scores

    x, y = 1.3e154, -1.2e154

    def member(v):
        return [[1.0, 1.0], [v / 2, -v / 2], [v * v / 4, v * v / 4]]

    moments = np.array([[member(x), member(y)]])
    refs = np.array([[x / 2, y / 2]])
    extrema = np.array([[[[x, 0.0], [x, 0.0]], [[y, 0.0], [y, 0.0]]]])
    score = _split_scores(moments, refs, extrema, "basket", np.array([0]))
    assert np.isnan(score).all()


@pytest.mark.parametrize("scale", [1e156, 1e160])
def test_cscv_pbo_over_returns_whose_squares_overflow_declines_silently(scale):
    # Returns of ~1e158 square past the double range in the per-block moment accumulators: every
    # split's variance overflows and scores NaN, so no split ranks and the block declines — the
    # overflow an expected input there (`_representable` decides it), never a warning. At ~1e154
    # the per-block sums stay finite and it is their sum over a split's blocks that overflows.
    rng = np.random.RandomState(5)
    cells = _pbo_cells(rng, 800, 12)
    for c in cells:
        c["fwd_col"] = np.asarray(c["fwd_col"]) * scale
    out = cscv_pbo(cells, 800, ["t"], n_combos_declared=12)
    assert out["pbo"] is None and out["reason"] == "insufficient_data"
    assert out["n_splits"] == 0 and out["n_combos"] == 12


def test_cscv_split_score_of_a_subnormal_variance_is_nan_never_a_sharpe():
    # Returns of ~1e-161 square into the SUBNORMAL range, where the rounding floor the split's
    # sum of squares is tested against has itself underflowed to 0.0: the Sharpe was read off a
    # variance that had lost its bits — 0.13804 for 0.13853. While the variance stays normal the
    # power-of-two scale keeps every bit of the score.
    from seikan.analysis._cscv import _split_scores

    r = np.random.RandomState(8).randn(40)
    b = np.repeat(np.arange(4), 10)
    blocks = np.array([0, 2])
    for mode in ("conjunction", "basket"):
        score = _split_scores(*_split_tensors(r, b), mode, blocks)
        assert np.isfinite(score).all()
        scaled = _split_scores(*_split_tensors(r * 2.0**-500, b), mode, blocks)
        np.testing.assert_array_equal(scaled, score)
        assert np.isnan(_split_scores(*_split_tensors(r * 2.0**-535, b), mode, blocks)).all()


def test_cscv_near_constant_split_pool_is_never_ranked():
    # The two-level fixture with every other bar of combo 0 nudged up one ulp: no split pool of
    # it is exactly constant any more, but none varies above rounding either.
    cells = _two_level_cells(("t",))
    for cell in cells:
        if cell["key"][0] == 0:
            fwd = np.array(cell["fwd_col"], dtype=float)
            fwd[::2] = np.nextafter(fwd[::2], 1.0)
            cell["fwd_col"] = fwd
    out = cscv_pbo(cells, 400, ["t"], n_combos_declared=5)
    assert out["reason"] is None and out["n_candidates_min"] == 4


@pytest.mark.parametrize(("mode", "targets"), [("conjunction", ("t",)), ("basket", ("A", "B"))])
def test_cscv_single_level_near_constant_pool_is_never_ranked(mode, targets):
    # Combo 0 sits at ONE level with every other bar one ulp up: the column's own mean lies on
    # the values, so a split's centered sum of squares is computed accurately — and is still
    # only rounding-scale spread. It scored a Sharpe of ~1e15, won every split and read pbo 0.0;
    # the HAC's gate (largest deviation within 10 ulps of the mean, read off the per-block
    # extrema) leaves it unranked whatever the column's other blocks hold.
    T = 400
    rng = np.random.RandomState(9)
    level = np.full(T, 0.005)
    level[::2] = np.nextafter(0.005, 1.0)
    cells = []
    for k in range(5):
        for tg in targets:
            m = np.zeros(T)
            m[rng.choice(T - 2, 120, replace=False)] = 1.0
            fwd = level if k == 0 else rng.normal(0.0, 0.01, T)
            cells.append({"key": (k, tg), "mask_col": m, "fwd_col": fwd, "h": 1})
    out = cscv_pbo(cells, T, list(targets), mode=mode, n_combos_declared=5)
    assert out["reason"] is None and out["n_candidates_min"] == 4


def test_cscv_split_score_needs_the_extrema_for_an_exact_constant_pool():
    # A column whose mean sits within ~1e-14 of an exactly constant split's value: rebuilding
    # the basket's pooled mean is off by ~ε·|c|, so its residue clears the rounding floor and
    # only the per-block extrema keep the split undefined.
    from seikan.analysis._cscv import _split_scores

    hi = 0.21
    for _ in range(103):
        hi = np.nextafter(hi, 1.0)
    r = np.array([0.21] * 10 + [hi] * 10)
    moments, refs, extrema = _split_tensors(r, np.repeat(np.arange(4), 5))
    for mode in ("basket", "conjunction"):
        assert np.isnan(_split_scores(moments, refs, extrema, mode, np.array([0, 1]))).all()


def test_cscv_basket_absent_member_cannot_widen_an_exact_constant_split():
    # The same near-level fixture through cscv_pbo, combo 0's member B absent: its (+inf, -inf)
    # extrema must leave every within-half split exactly constant.
    T, hi = 400, 0.21
    for _ in range(300):
        hi = np.nextafter(hi, 1.0)
    level = np.where(np.arange(T) < T // 2, 0.21, hi)
    rng = np.random.RandomState(31)
    cells = []
    for k in range(5):
        for tg in ("A", "B"):
            m = np.zeros(T)
            m[rng.choice(T - 2, 120, replace=False)] = 1.0
            if (k, tg) != (0, "B"):
                fwd = level if k == 0 else rng.normal(0.0, 0.01, T)
                cells.append({"key": (k, tg), "mask_col": m, "fwd_col": fwd, "h": 1})
    out = cscv_pbo(cells, T, ["A", "B"], mode="basket", n_combos_declared=5)
    assert out["reason"] is None and out["blocks"] == 8 and out["n_candidates_min"] == 4


def test_cscv_basket_absent_member_bounds_nothing():
    # Combo 0 has member A only; its pool is constant within each half. The absent member's
    # (+inf, -inf) extrema must not widen the pool's range into "varying".
    cells = [c for c in _two_level_cells(("A", "B")) if c["key"] != (0, "B")]
    out = cscv_pbo(cells, 400, ["A", "B"], mode="basket", n_combos_declared=5)
    assert out["reason"] is None and out["blocks"] == 8
    assert out["n_candidates_min"] == 4


@pytest.mark.parametrize(("mode", "expected"), [("basket", 5), ("conjunction", 4)])
def test_cscv_members_constant_at_different_levels(mode, expected):
    # Combo 0: member A constant at 0.005, member B at 0.022. The basket POOLS its members — two
    # values, a pool that varies, ranked in every split; conjunction reads each member alone —
    # each constant, the combo never ranks.
    T = 400
    rng = np.random.RandomState(5)
    cells = []
    for k in range(5):
        for tg, lvl in (("A", 0.005), ("B", 0.022)):
            m = np.zeros(T)
            m[rng.choice(T - 2, 120, replace=False)] = 1.0
            fwd = np.full(T, lvl) if k == 0 else rng.normal(0.0, 0.01, T)
            cells.append({"key": (k, tg), "mask_col": m, "fwd_col": fwd, "h": 1})
    out = cscv_pbo(cells, T, ["A", "B"], mode=mode, n_combos_declared=5)
    assert out["reason"] is None and out["n_candidates_min"] == expected


def test_oos_degradation_slope_matches_centered_ols_and_reason():
    tr = np.array([1.0, 2.0, 3.0, 4.0])
    te = np.array([0.5, 1.1, 1.4, 2.2])
    slope, reason = _degradation_slope(tr, te)
    tr_c, te_c = tr - tr.mean(), te - te.mean()
    assert reason is None and slope == pytest.approx(float(tr_c @ te_c / (tr_c @ tr_c)))
    # Near-constant train scores at a large level: an exact-zero std test would sail into a
    # rank-deficient fit (numpy's polyfit warned "poorly conditioned" and emitted an arbitrary
    # min-norm slope); the scale-relative test nulls it with the reason said out loud.
    tr_flat = np.full(8, 1e6)
    tr_flat[3] += 1e-10
    slope, reason = _degradation_slope(tr_flat, np.linspace(0.0, 1.0, 8))
    assert slope is None and reason == "degenerate_train_scores"


def test_cscv_scored_block_carries_null_slope_reason():
    rng = np.random.RandomState(22)
    out = cscv_pbo(_pbo_cells(rng, 800, 8), 800, ["t"], n_combos_declared=8)
    assert out["reason"] is None
    assert "oos_degradation_slope_reason" in out and out["oos_degradation_slope_reason"] is None


def _cscv_bar_numerators(raw, masks, order):
    """``cscv_pbo``'s ``bar_numerators`` for ``cross_mean`` candidates over the (bars × A, B, C)
    legs ``raw``: per candidate ``(k,)``, ``cross_mean_bar_numerators`` over the members in
    ``order``'s column order, every member closed on the candidate's firing bars."""
    cols = raw[:, ["ABC".index(m) for m in order]]
    return {
        (k,): cross_mean_bar_numerators(cols, np.repeat((m > 0)[:, None], len(order), axis=1))
        for k, m in enumerate(masks)
    }


def _full_bar_cscv_cells(order, n_combos=5, length=400, seed=31):
    """A ``cross_mean`` basket grid whose every candidate is ALL-FULL: members A, B, C fire
    together on each candidate's own bars, each forward column the member's leg less the bar's
    basket mean — rounded excesses whose exact per-bar sum is 0 and whose rounded sum is
    residue. Cells run in ``order``; returned beside the candidates' ``bar_numerators``."""
    rng = np.random.RandomState(seed)
    raw = np.round(rng.normal(0.0, 0.02, (length, 3)), 4)
    fwd = dict(zip("ABC", (raw - raw.mean(axis=1, keepdims=True)).T, strict=True))
    masks = []
    for _ in range(n_combos):
        m = np.zeros(length)
        m[rng.choice(length - 2, 120, replace=False)] = 1.0
        masks.append(m)
    cells = [
        {"key": (k, tg), "mask_col": masks[k], "fwd_col": fwd[tg], "h": 1}
        for k in range(n_combos)
        for tg in order
    ]
    return cells, _cscv_bar_numerators(raw, masks, order)


def test_cscv_pbo_all_full_cross_mean_grid_ties_in_every_declaration_order():
    # Every candidate's every bar is full, so every split's exact pooled mean is 0: each split
    # is a full tie (λ = 0, no test loss, no train spread for the slope). Over the rounded rows
    # each split Sharpe was residue / sd, and pbo, lambda_mean and prob_oos_loss ranked that
    # residue in the order the targets were declared.
    import itertools

    keys = ("pbo", "lambda_mean", "prob_oos_loss")
    plain_reads = set()
    for order in itertools.permutations("ABC"):
        cells, numerators = _full_bar_cscv_cells(order)
        out = cscv_pbo(
            cells, 400, list(order), mode="basket", n_combos_declared=5, bar_numerators=numerators
        )
        assert out["reason"] is None and out["n_combos"] == 5 and out["n_splits"] == 70
        assert out["pbo"] == out["lambda_mean"] == out["prob_oos_loss"] == 0.0, order
        assert out["oos_degradation_slope"] is None
        assert out["oos_degradation_slope_reason"] == "degenerate_train_scores"
        plain = cscv_pbo(cells, 400, list(order), mode="basket", n_combos_declared=5)
        plain_reads.add(tuple(plain[k] for k in keys))
    assert len(plain_reads) > 1  # the premise: the rounded rows' read follows the order


def test_cscv_pbo_all_full_grid_s_tied_train_winner_is_the_first_declared_candidate():
    # An all-full grid's train scores all tie at exactly 0, so the train winner is the first
    # scoreable candidate in DECLARATION order (`np.nanargmax`, the documented tie rule), and a
    # split whose winner holds no test pool is skipped. Four candidates confined each to one
    # eighth of the sample, declared first, win every split whose train blocks hold them and
    # are absent from its test blocks: too few splits score at any S, and the block declines
    # `insufficient_data` — identically in every TARGET order, where the rounded rows ranked
    # residue across the candidates and scored a pbo that followed it. Declared after the broad
    # candidates, the same grid scores every split, each a full tie.
    import itertools

    length = 160
    rng = np.random.RandomState(3)
    raw = np.round(rng.normal(0.0, 0.03, (length, 3)), 4)
    fwd = dict(zip("ABC", (raw - raw.mean(axis=1, keepdims=True)).T, strict=True))
    sparse, broad = [], []
    for k in range(4):
        m = np.zeros(length)
        m[k * 20 + 6 : k * 20 + 14] = 1.0  # the middle of the k-th eighth
        sparse.append(m)
    for _ in range(4):
        m = np.zeros(length)
        m[rng.choice(length - 2, 60, replace=False)] = 1.0
        broad.append(m)
    ledger = ("blocks", "n_splits", "n_splits_attempted", "n_candidates_min", "n_combos")
    for masks, scores in ((sparse + broad, False), (broad + sparse, True)):
        reads, plain_reads = set(), set()
        for order in itertools.permutations("ABC"):
            cells = [
                {"key": (k, tg), "mask_col": m, "fwd_col": fwd[tg], "h": 1}
                for k, m in enumerate(masks)
                for tg in order
            ]
            numerators = _cscv_bar_numerators(raw, masks, order)
            out = cscv_pbo(
                cells,
                length,
                list(order),
                mode="basket",
                n_combos_declared=8,
                bar_numerators=numerators,
            )
            if scores:
                assert out["reason"] is None and out["n_splits"] == 70, order
                assert out["pbo"] == out["lambda_mean"] == out["prob_oos_loss"] == 0.0
            else:
                assert out["reason"] == "insufficient_data" and out["pbo"] is None, order
            assert out["n_combos"] == 8
            reads.add(tuple(out[k] for k in ledger))
            plain = cscv_pbo(cells, length, list(order), mode="basket", n_combos_declared=8)
            assert plain["reason"] is None
            plain_reads.add(plain["pbo"])
        assert len(reads) == 1  # the whole ledger, S included, in any target order
        assert len(plain_reads) > 1  # the premise: the rounded rows' pbo followed it


def test_cscv_split_mean_sums_the_bars_exact_excess_sums():
    # Three candidates of three members, ten bars per block. A `cross_mean` candidate's EVERY
    # split takes its MEAN from the per-block exact sums of its bars' numerators
    # (`bar_numerators`) over the member count, over the rows' own sd: the all-full candidate's
    # mean is exactly 0.0 on every split (its rounded rows read residue), the partial
    # candidate's the exact sum of the excesses its split holds, its blocks added once. A
    # candidate the numerators do not key, and the whole conjunction score, are the rows' read,
    # byte for byte.
    from itertools import combinations

    from seikan.analysis._cscv import _split_scores

    rng = np.random.RandomState(47)
    s = 4
    b = np.repeat(np.arange(s), 10)
    raw = np.round(rng.normal(0.0, 0.02, (b.size, 3)), 4)  # bar × member legs
    excess = raw - raw.mean(axis=1, keepdims=True)  # each member's rounded excess
    full = np.ones(raw.shape, dtype=bool)
    partial = rng.random(raw.shape) < 0.6
    partial[:, 0] |= ~partial.any(axis=1)  # every bar holds a closed member
    closed = [full, partial, full]
    moments, refs, extrema, sums = [], [], [], []
    for q in range(3):
        numerators = cross_mean_bar_numerators(raw, closed[q])
        on = closed[q]
        per_member = [_split_tensors(excess[on[:, g], g], b[on[:, g]], s) for g in range(3)]
        moments.append([m[0, 0] for m, _, _ in per_member])
        refs.append([ref[0, 0] for _, ref, _ in per_member])
        extrema.append([e[0, 0] for _, _, e in per_member])
        sums.append(
            [
                [
                    math.fsum(numerators.at(np.flatnonzero(on[:, g] & (b == k)), g).ravel())
                    for k in range(s)
                ]
                for g in range(3)
            ]
        )
    tensors = (np.array(moments), np.array(refs), np.array(extrema))
    pool = (np.array(sums), np.array([True, True, False]))
    legs = [[Fraction(float(x)) for x in row] for row in raw]
    residue = []
    for split in combinations(range(s), 2):
        idx = np.array(split)
        conj = _split_scores(*tensors, "conjunction", idx)
        np.testing.assert_array_equal(_split_scores(*tensors, "conjunction", idx, pool, 3), conj)
        plain = _split_scores(*tensors, "basket", idx)
        got = _split_scores(*tensors, "basket", idx, pool, 3)
        assert got[0] == 0.0
        residue.append(plain[0])
        assert got[2] == plain[2]  # unkeyed: the moments' own mean
        # the partial candidate: its pool sums over n, over the rows' own reconstructed sd —
        m, ref = tensors[0][1][:, :, idx].sum(axis=2), tensors[1][1]  # (T, 3), (T,)
        n = m[:, 0].sum()
        d = ref - (m[:, 1] + m[:, 0] * ref).sum() / n
        sd = np.sqrt((m[:, 2] + 2.0 * d * m[:, 1] + m[:, 0] * d * d).sum() / (n - 1.0))
        split_sum = pool[0][:, :, idx].sum(axis=(1, 2))[1]
        assert got[1] == split_sum / (3 * n) / sd
        # — whose sum is the split's exact numerator, to its blocks' one addition
        held = np.isin(b, split)[:, None] & closed[1]
        exact = sum(legs[t][g] - sum(legs[t]) / 3 for t, g in zip(*np.nonzero(held), strict=True))
        assert split_sum == pytest.approx(float(3 * exact), rel=1e-15, abs=1e-18)
    assert any(r != 0.0 for r in residue)  # the premise: the rounded rows read residue


def test_cscv_pbo_bar_numerators_reach_no_other_read():
    # Conjunction never reads the bar numerators, and a basket candidate they do not key keeps
    # its moments' own mean: an empty mapping is the plain read, byte for byte.
    cells, length = _one_member_edge_cells()
    basket = cscv_pbo(cells, length, ["A", "B"], mode="basket", n_combos_declared=8)
    assert basket["reason"] is None
    assert (
        cscv_pbo(cells, length, ["A", "B"], mode="basket", n_combos_declared=8, bar_numerators={})
        == basket
    )
    zeros = {
        (k,): BarNumerators(np.zeros((length, 1)), np.zeros(length, dtype=np.int64))
        for k in range(8)
    }
    conj = cscv_pbo(cells, length, ["A", "B"], n_combos_declared=8)
    assert conj["reason"] is None
    assert cscv_pbo(cells, length, ["A", "B"], n_combos_declared=8, bar_numerators=zeros) == conj
