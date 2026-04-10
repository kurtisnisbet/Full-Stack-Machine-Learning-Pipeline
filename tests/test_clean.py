"""
test_clean.py — Unit tests for src.data.clean

All tests use synthetic DataFrames so no dataset files are required.
"""

import pandas as pd
import pytest

from src.data.clean import clean_weatheraus


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(**overrides) -> pd.DataFrame:
    """Build a minimal weather DataFrame for testing."""
    base = {
        "Date":          ["2010-01-01", "2010-01-02", "2010-01-03"],
        "Location":      ["Sydney", "Melbourne", "Brisbane"],
        "MinTemp":       [10.0, 12.0, None],
        "MaxTemp":       [25.0, 22.0, 28.0],
        "Rainfall":      [0.0, 1.2, 0.0],
        "Sunshine":      [8.0, None, 7.0],      # >38% missing → should be dropped
        "Evaporation":   [4.0, None, 3.5],      # >38% missing → should be dropped
        "Cloud9am":      [3, None, 2],           # >38% missing → should be dropped
        "Cloud3pm":      [4, None, 5],           # >38% missing → should be dropped
        "WindGustSpeed": [45.0, 30.0, 55.0],
        "Humidity9am":   [60.0, 75.0, 55.0],
        "RainTomorrow":  ["No", "Yes", "No"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCleanWeatherAUS:

    def test_drops_high_missingness_columns(self):
        """Sunshine, Evaporation, Cloud9am, Cloud3pm must be dropped."""
        df = _make_df()
        result, _ = clean_weatheraus(df)
        for col in ("Sunshine", "Evaporation", "Cloud9am", "Cloud3pm"):
            assert col not in result.columns, f"Column '{col}' should have been dropped."

    def test_retains_low_missingness_columns(self):
        """Columns with acceptable missingness must be kept."""
        df = _make_df()
        result, _ = clean_weatheraus(df)
        for col in ("MinTemp", "MaxTemp", "WindGustSpeed", "Humidity9am"):
            assert col in result.columns, f"Column '{col}' should have been retained."

    def test_drops_rows_with_missing_target(self):
        """Rows where RainTomorrow is NaN must be removed."""
        df = _make_df(RainTomorrow=["No", None, "Yes"])
        result, summary = clean_weatheraus(df)
        assert len(result) == 2
        assert result["RainTomorrow"].notna().all()
        assert summary["dropped_rows_missing_target"] == 1

    def test_all_rows_kept_when_target_complete(self):
        """No rows are dropped when target has no missing values."""
        df = _make_df()
        result, summary = clean_weatheraus(df)
        assert summary["dropped_rows_missing_target"] == 0
        assert len(result) == len(df)

    def test_summary_keys_present(self):
        """Summary dict must contain all expected audit keys."""
        df = _make_df()
        _, summary = clean_weatheraus(df)
        expected_keys = {
            "rows_before", "rows_after",
            "cols_before", "cols_after",
            "dropped_columns", "dropped_columns_count",
            "dropped_rows_missing_target",
        }
        assert expected_keys.issubset(set(summary.keys()))

    def test_raises_without_target_column(self):
        """clean_weatheraus must raise KeyError if RainTomorrow is absent."""
        df = _make_df().drop(columns=["RainTomorrow"])
        with pytest.raises(KeyError, match="RainTomorrow"):
            clean_weatheraus(df)

    def test_date_cast_to_datetime(self):
        """Date column must be cast to datetime64 dtype."""
        df = _make_df()
        result, _ = clean_weatheraus(df)
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_summary_col_counts_correct(self):
        """cols_before and cols_after in summary must reflect actual changes."""
        df = _make_df()
        result, summary = clean_weatheraus(df)
        assert summary["cols_before"] == df.shape[1]
        assert summary["cols_after"]  == result.shape[1]
        assert summary["dropped_columns_count"] == 4
