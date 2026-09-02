"""The statistical layer's public surface — the observer-pure forward-return event study's
per-cell evidence, one module per estimator family behind this facade.

Every statistic here is NOMINAL and per-cell. Nothing selects, ranks or crowns a cell, and no
cell's numbers are corrected for the size of the grid the caller declared — the engine measures
every declared hypothesis and reports it, and the calling agent prices its own selection against
the stamped ``n_hypotheses_attempted``. That placement is deliberate, not a gap: a search
correction computed inside one run only ever sees that run's grid, so it would silently
understate a caller who searched across runs, DSLs and data windows.

How the targets are read is the thesis's declared ``target_mode``, stamped into every summary.
Under ``conjunction`` (the default) they are the thesis's REGIME, a condition a cell must hold
across, not a search axis to cherry-pick: every target's numbers are reported side by side and
the weakest one speaks for itself, never folded into one adjusted statistic. Under ``basket``
the targets form ONE cross-section per bar and each cell carries ONE pooled read over the
concatenated (bar × member) observations (:func:`pooled_reliability_summary`), with the
per-member numbers kept as ATTRIBUTION — reported, never graded. In neither mode does anything
here select among targets.

KNOWN CAVEAT — the rotation null assumes SHIFT EXCHANGEABILITY: rotating the firing mask is only
a valid null if the forward-return series looks the same wherever the mask lands. It does not
when volatility clusters in the same stretches the signal fires (crises, earnings seasons,
regime breaks) — rotated masks fall into calm periods, the null distribution is too narrow, and
``rot_p`` over-certifies. KNOWN CAVEAT — the overlap HAC is ANTI-CONSERVATIVE: the Bartlett
taper downweights exactly the lags that carry the overlap covariance, so ``hac_se`` understates
the long-run variance and ``t_hac`` runs hot on heavily overlapping pools. Both estimators
ride as per-cell EVIDENCE precisely because they are cheap, observer-pure and directionally
informative; neither is calibrated, so neither certifies anything on its own. Calibrating them
under dependent/heteroskedastic nulls is open statistical work, not a refactor.
"""

from seikan.analysis._baseline import baseline_summary, subperiod_edges, subperiod_means
from seikan.analysis._bootstrap import episode_bootstrap_ci
from seikan.analysis._conditioning import (
    ASSOC_MIN_N,
    BUCKET_MIN_N,
    cell_conditional_buckets,
    conditional_buckets,
    feature_outcome_association,
)
from seikan.analysis._cscv import cscv_pbo
from seikan.analysis._episodes import (
    EPISODE_LIST_MAX,
    episode_ledger,
    episode_profile,
    episode_stats,
    overlap_clusters,
)
from seikan.analysis._hac import newey_west_mean, nonoverlap_count
from seikan.analysis._pools import (
    benchmark_regression,
    concentration,
    edge_ratio,
    mae_block,
    mfe_block,
    pool_moments,
    pool_quantiles,
    profit_factor,
    timing_summary,
)
from seikan.analysis._rotation import pooled_reliability_summary, reliability_summary

#: The statistical-mechanics version stamped into every engine summary. Bumped whenever a
#: frozen-layer statistic changes MEANING, so two summaries compare only under the same version
#: (``seikan_version`` names the package, ``gate.POLICY_VERSION`` the checklist semantics — this
#: names the ESTIMATORS). The gate reads it as a drift detector: a summary stamped with anything
#: but this build's number refuses ungraded rather than being graded under assumptions that may
#: not hold of it. The revision history is in CHANGELOG.md.
STATISTICS_VERSION = 4

__all__ = [
    "ASSOC_MIN_N",
    "BUCKET_MIN_N",
    "EPISODE_LIST_MAX",
    "STATISTICS_VERSION",
    "baseline_summary",
    "benchmark_regression",
    "cell_conditional_buckets",
    "concentration",
    "conditional_buckets",
    "cscv_pbo",
    "edge_ratio",
    "episode_bootstrap_ci",
    "episode_ledger",
    "episode_profile",
    "episode_stats",
    "feature_outcome_association",
    "mae_block",
    "mfe_block",
    "newey_west_mean",
    "nonoverlap_count",
    "overlap_clusters",
    "pool_moments",
    "pool_quantiles",
    "pooled_reliability_summary",
    "profit_factor",
    "reliability_summary",
    "subperiod_edges",
    "subperiod_means",
    "timing_summary",
]
