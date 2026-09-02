"""The hardcoded pre-write integrity gate (``seikan.dataio.check_frame``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seikan.dataio import IntegrityError, check_frame


def _idx(dates):
    return pd.DatetimeIndex(pd.to_datetime(dates), name="datetime")


def _ohlcv(dates, o, h, lo, c, v=None):
    data = {"open": o, "high": h, "low": lo, "close": c}
    if v is not None:
        data["volume"] = v
    return pd.DataFrame(data, index=_idx(dates))


def test_accepts_clean_ohlcv_and_series():
    check_frame(
        _ohlcv(["2024-01-02", "2024-01-03"], [10, 11], [12, 13], [9, 10], [11, 12], [100, 200]),
        shape="ohlcv",
    )
    df = pd.DataFrame(
        {"iv30": [0.2, 0.22], "iv_skew": [-1.0, -0.9]}, index=_idx(["2024-01-02", "2024-01-03"])
    )
    check_frame(df, shape="series")  # negative values are fine for non-ohlcv


def test_flags_unsorted_and_duplicate_index():
    df = pd.DataFrame(
        {"v": [1.0, 2.0, 3.0]}, index=_idx(["2024-01-03", "2024-01-02", "2024-01-02"])
    )
    with pytest.raises(IntegrityError) as e:
        check_frame(df, shape="series")
    assert "sorted" in str(e.value) and "duplicate" in str(e.value)


def test_flags_nan_column_and_non_datetime_index():
    nan = pd.DataFrame({"v": [np.nan, np.nan]}, index=_idx(["2024-01-02", "2024-01-03"]))
    with pytest.raises(IntegrityError):
        check_frame(nan, shape="series")
    with pytest.raises(IntegrityError):
        check_frame(pd.DataFrame({"v": [1.0, 2.0]}, index=[0, 1]), shape="series")


def test_flags_ohlcv_invariants():
    with pytest.raises(IntegrityError) as e:
        check_frame(_ohlcv(["2024-01-02"], [10], [8], [9], [9.5]), shape="ohlcv")  # high < low
    assert "high < low" in str(e.value)
    with pytest.raises(IntegrityError) as e:
        check_frame(_ohlcv(["2024-01-02"], [0.0], [1.0], [-1.0], [0.5]), shape="ohlcv")
    assert "non-positive" in str(e.value)
