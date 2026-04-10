"""
test_features.py — Unit tests for src.features.build_features

Tests focus on the time_split_indices function, which is the most
critical logic for preventing data leakage.
"""

import pandas as pd
import pytest

from src.features.build_features import time_split_indices


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_dated_df(n: int = 100) -> pd.DataFrame:
    """Return a DataFrame with sequential dates and dummy values."""
    dates = pd.date_range("2010-01-01", periods=n, freq="D")
    return pd.DataFrame({"Date": dates, "value": range(n)})


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestTimeSplitIndices:

    def test_split_sizes_match_fractions(self):
        """Row counts should approximately match the requested fractions."""
        df = _make_dated_df(100)
        train_idx, val_idx, test_idx = time_split_indices(df, "Date", 0.7, 0.15, 0.15)
        assert len(train_idx) == 70
        assert len(val_idx)   == 15
        assert len(test_idx)  == 15

    def test_no_index_overlap(self):
        """Train, val, and test index sets must be disjoint."""
        df = _make_dated_df(200)
        train_idx, val_idx, test_idx = time_split_indices(df, "Date", 0.7, 0.15, 0.15)
        assert len(set(train_idx) & set(val_idx))   == 0
        assert len(set(train_idx) & set(test_idx))  == 0
        assert len(set(val_idx)   & set(test_idx))  == 0

    def test_covers_all_rows(self):
        """Union of splits must equal the full DataFrame index."""
        df = _make_dated_df(150)
        train_idx, val_idx, test_idx = time_split_indices(df, "Date", 0.7, 0.15, 0.15)
        all_idx = set(train_idx) | set(val_idx) | set(test_idx)
        assert all_idx == set(df.index)

    def test_chronological_ordering(self):
        """All training dates must precede all validation dates,
        which must precede all test dates."""
        df = _make_dated_df(200)
        train_idx, val_idx, test_idx = time_split_indices(df, "Date", 0.7, 0.15, 0.15)

        train_max = df.loc[train_idx, "Date"].max()
        val_min   = df.loc[val_idx,   "Date"].min()
        val_max   = df.loc[val_idx,   "Date"].max()
        test_min  = df.loc[test_idx,  "Date"].min()

        assert train_max <= val_min,  "Training dates overlap with validation."
        assert val_max   <= test_min, "Validation dates overlap with test."

    def test_raises_on_fractions_not_summing_to_one(self):
        """Should raise ValueError when fractions do not sum to 1.0."""
        df = _make_dated_df(100)
        with pytest.raises(ValueError, match="sum to 1.0"):
            time_split_indices(df, "Date", 0.6, 0.2, 0.1)

    def test_works_with_unsorted_input(self):
        """Chronological ordering must hold even when the input is shuffled."""
        df = _make_dated_df(100).sample(frac=1, random_state=42).reset_index(drop=True)
        train_idx, val_idx, test_idx = time_split_indices(df, "Date", 0.7, 0.15, 0.15)

        train_max = df.loc[train_idx, "Date"].max()
        val_min   = df.loc[val_idx,   "Date"].min()
        assert train_max <= val_min, "Shuffled input should still produce chronological splits."
