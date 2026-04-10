"""
test_train.py — Unit tests for src.models.train and src.models.evaluate

Uses synthetic data only — no dataset files required.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.train import coerce_binary_target, eval_on_val, build_model
from src.models.evaluate import compute_metrics, sweep_threshold


# ── Tests: coerce_binary_target ───────────────────────────────────────────────

class TestCoerceBinaryTarget:

    def test_yes_no_strings(self):
        y = pd.Series(["Yes", "No", "Yes", "No"])
        result = coerce_binary_target(y)
        assert list(result) == [1, 0, 1, 0]

    def test_true_false_strings(self):
        y = pd.Series(["True", "False", "True"])
        result = coerce_binary_target(y)
        assert list(result) == [1, 0, 1]

    def test_numeric_strings(self):
        y = pd.Series(["1", "0", "1"])
        result = coerce_binary_target(y)
        assert list(result) == [1, 0, 1]

    def test_boolean_dtype(self):
        y = pd.Series([True, False, True])
        result = coerce_binary_target(y)
        assert list(result) == [1, 0, 1]

    def test_case_insensitive(self):
        y = pd.Series(["YES", "no", "YeS"])
        result = coerce_binary_target(y)
        assert list(result) == [1, 0, 1]

    def test_raises_on_unknown_label(self):
        y = pd.Series(["Yes", "Maybe", "No"])
        with pytest.raises(ValueError, match="Unrecognized target labels"):
            coerce_binary_target(y)

    def test_output_dtype_is_int(self):
        y = pd.Series(["Yes", "No"])
        assert coerce_binary_target(y).dtype == int


# ── Tests: eval_on_val ────────────────────────────────────────────────────────

class TestEvalOnVal:

    @pytest.fixture
    def fitted_model(self):
        """A simple LogisticRegression trained on linearly separable data."""
        rng = np.random.default_rng(0)
        X   = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
        y   = pd.Series((X["a"] + X["b"] > 0).astype(int))
        model = LogisticRegression(random_state=0, max_iter=200)
        model.fit(X, y)
        return model, X, y

    def test_returns_expected_keys(self, fitted_model):
        model, X, y = fitted_model
        result = eval_on_val(model, X, y, threshold=0.5)
        assert set(result.keys()) == {"roc_auc", "accuracy", "precision", "recall", "f1"}

    def test_metrics_in_range(self, fitted_model):
        model, X, y = fitted_model
        result = eval_on_val(model, X, y, threshold=0.5)
        for key, val in result.items():
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{key} = {val} is outside [0, 1]"

    def test_different_thresholds_change_predictions(self, fitted_model):
        """Lowering the threshold should increase recall."""
        model, X, y = fitted_model
        m_high = eval_on_val(model, X, y, threshold=0.8)
        m_low  = eval_on_val(model, X, y, threshold=0.2)
        assert m_low["recall"] >= m_high["recall"]


# ── Tests: build_model ────────────────────────────────────────────────────────

class TestBuildModel:

    def test_builds_logistic_regression(self):
        from sklearn.linear_model import LogisticRegression
        model = build_model("logistic_regression", {"C": 1.0}, {}, random_seed=42)
        assert isinstance(model, LogisticRegression)

    def test_builds_random_forest(self):
        from sklearn.ensemble import RandomForestClassifier
        model = build_model("random_forest", {"n_estimators": 10}, {}, random_seed=42)
        assert isinstance(model, RandomForestClassifier)

    def test_raises_on_unknown_algorithm(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            build_model("neural_net", {}, {}, random_seed=42)


# ── Tests: compute_metrics (evaluate.py) ─────────────────────────────────────

class TestComputeMetrics:

    def test_perfect_predictions(self):
        y_true = pd.Series([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.9, 0.9])
        m = compute_metrics(y_true, y_prob, threshold=0.5)
        assert m["accuracy"]  == 1.0
        assert m["precision"] == 1.0
        assert m["recall"]    == 1.0
        assert m["f1"]        == 1.0

    def test_all_wrong_predictions(self):
        y_true = pd.Series([1, 1, 0, 0])
        y_prob = np.array([0.1, 0.1, 0.9, 0.9])  # inverted
        m = compute_metrics(y_true, y_prob, threshold=0.5)
        assert m["accuracy"]  == 0.0
        assert m["recall"]    == 0.0

    def test_returns_expected_keys(self):
        y_true = pd.Series([0, 1, 0, 1])
        y_prob = np.array([0.3, 0.7, 0.2, 0.8])
        m = compute_metrics(y_true, y_prob, threshold=0.5)
        assert set(m.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}


# ── Tests: sweep_threshold ────────────────────────────────────────────────────

class TestSweepThreshold:

    def test_returns_dataframe_and_float(self):
        y_true = pd.Series([0, 1, 0, 1, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7, 0.6])
        df, opt_thresh = sweep_threshold(y_true, y_prob, steps=5)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(opt_thresh, float)

    def test_optimal_threshold_in_range(self):
        y_true = pd.Series([0, 1, 0, 1, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.75, 0.3, 0.15, 0.85])
        _, opt_thresh = sweep_threshold(y_true, y_prob, steps=10)
        assert 0.05 <= opt_thresh <= 0.95

    def test_sweep_contains_required_columns(self):
        y_true = pd.Series([0, 1] * 10)
        y_prob = np.linspace(0.1, 0.9, 20)
        df, _ = sweep_threshold(y_true, y_prob, steps=5)
        for col in ("threshold", "f1", "precision", "recall", "accuracy"):
            assert col in df.columns
