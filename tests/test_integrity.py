"""The frame-level integrity invariants (``seikan.dataio.check_frame``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from seikan.dataio import check_frame


def _idx(dates):
    return pd.DatetimeIndex(pd.to_datetime(dates), name="datetime")


def _ohlcv(dates, o, h, lo, c, v=None):
    data = {"open": o, "high": h, "low": lo, "close": c}
    if v is not None:
        data["volume"] = v
    return pd.DataFrame(data, index=_idx(dates))


def test_accepts_clean_ohlcv_and_series():
    clean = _ohlcv(["2024-01-02", "2024-01-03"], [10, 11], [12, 13], [9, 10], [11, 12], [100, 200])
    assert check_frame(clean, shape="ohlcv") == []
    df = pd.DataFrame(
        {"iv30": [0.2, 0.22], "iv_skew": [-1.0, -0.9]}, index=_idx(["2024-01-02", "2024-01-03"])
    )
    assert check_frame(df, shape="series") == []  # negative values are fine for non-ohlcv


def test_flags_nan_column():
    nan = pd.DataFrame({"v": [np.nan, np.nan]}, index=_idx(["2024-01-02", "2024-01-03"]))
    assert check_frame(nan, shape="series") == ["column 'v' is entirely NaN"]


def test_flags_ohlcv_invariants():
    high_below_low = _ohlcv(["2024-01-02"], [10], [8], [9], [9.5])
    assert "ohlcv has bars where high < low" in check_frame(high_below_low, shape="ohlcv")
    non_positive = _ohlcv(["2024-01-02"], [0.0], [1.0], [-1.0], [0.5])
    assert "ohlcv has non-positive prices" in check_frame(non_positive, shape="ohlcv")
