"""
train.py — Multi-model hyperparameter grid search with optional TimeSeriesSplit CV.

Reads all model definitions and grids from config.yaml (training section).
Supports: Logistic Regression, Random Forest, XGBoost.

Outputs per run:
  - models/rain_model.pkl            – best overall model bundle
  - models/preprocessor.pkl          – fitted preprocessor (for inference)
  - reports/tables/model_selection_{algo}.csv  – all runs for that algorithm
  - reports/tables/model_comparison.csv        – best of each algorithm, side by side

Usage:
    python -m src.models.train
    # or via Makefile:  make train
"""

from __future__ import annotations

import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.utils.config import load_config


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_file_key(files: dict, key: str, default_name: str) -> str:
    v = files.get(key)
    return str(v) if v else default_name


def coerce_binary_target(y: pd.Series) -> pd.Series:
    """Convert RainTomorrow labels to 0/1. Accepts Yes/No, Y/N, True/False, 1/0."""
    if y.dtype == "bool":
        return y.astype(int)
    y_str = y.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1, "y": 1, "true": 1, "1": 1,
        "no": 0,  "n": 0, "false": 0, "0": 0,
    }
    y_mapped = y_str.map(mapping)
    if y_mapped.isna().any():
        bad = sorted(set(y_str[y_mapped.isna()].unique().tolist()))
        raise ValueError(f"Unrecognized target labels in y: {bad}")
    return y_mapped.astype(int)


def build_paths(cfg: dict) -> dict:
    paths = cfg["paths"]
    files = cfg["files"]

    processed_dir = Path(paths["data_processed_dir"])
    models_dir    = Path(paths["models_dir"])
    tables_dir    = Path(paths["tables_dir"])

    return {
        "processed_dir":       processed_dir,
        "models_dir":          models_dir,
        "tables_dir":          tables_dir,
        "X_train_file":        processed_dir / _get_file_key(files, "X_train", "X_train.parquet"),
        "y_train_file":        processed_dir / _get_file_key(files, "y_train", "y_train.parquet"),
        "X_val_file":          processed_dir / _get_file_key(files, "X_val",   "X_val.parquet"),
        "y_val_file":          processed_dir / _get_file_key(files, "y_val",   "y_val.parquet"),
        "model_artifact":      Path(files.get("model_artifact", str(models_dir / "rain_model.pkl"))),
        "preprocessor_artifact": models_dir / "preprocessor.pkl",
        "comparison_log":      tables_dir / "model_comparison.csv",
    }


def load_split(
    X_path: Path, y_path: Path
) -> tuple[pd.DataFrame, pd.Series, str]:
    X = pd.read_parquet(X_path)
    y_df = pd.read_parquet(y_path)
    if y_df.shape[1] != 1:
        raise ValueError(f"{y_path} should have exactly one column.")
    y_col = y_df.columns[0]
    y = coerce_binary_target(y_df.iloc[:, 0])
    return X, y, y_col


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(algo: str, grid_params: dict, fixed_params: dict, random_seed: int):
    """Instantiate a model for the given algorithm using merged params."""
    merged = {**fixed_params, **grid_params}
    if algo == "logistic_regression":
        return LogisticRegression(random_state=random_seed, **merged)
    elif algo == "random_forest":
        return RandomForestClassifier(random_state=random_seed, **merged)
    elif algo == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        return XGBClassifier(random_state=random_seed, **merged)
    else:
        raise ValueError(f"Unsupported algorithm: '{algo}'")


# ── Evaluation helpers ────────────────────────────────────────────────────────

def eval_on_val(model, X_val: pd.DataFrame, y_val: pd.Series, threshold: float) -> dict:
    """Evaluate a trained model on the validation set at a given threshold."""
    prob = model.predict_proba(X_val)[:, 1]
    pred = (prob >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(y_val, prob))
    except Exception:
        auc = None

    return {
        "roc_auc":   auc,
        "accuracy":  float(accuracy_score(y_val, pred)),
        "precision": float(precision_score(y_val, pred, zero_division=0)),
        "recall":    float(recall_score(y_val, pred, zero_division=0)),
        "f1":        float(f1_score(y_val, pred, zero_division=0)),
    }


# ── Per-algorithm grid search ─────────────────────────────────────────────────

def run_grid_search(
    algo: str,
    algo_cfg: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float,
    selection_metric: str,
    cv_folds: int,
    random_seed: int,
    tables_dir: Path,
) -> tuple[pd.DataFrame, object, dict] | None:
    """
    Grid search for a single algorithm.

    Returns (results_df, best_model, best_row), or None if the algorithm
    is disabled or unavailable.
    """
    if not algo_cfg.get("enabled", True):
        print(f"  [{algo}] Skipped (disabled in config).")
        return None

    if algo == "xgboost" and not XGBOOST_AVAILABLE:
        print(f"  [{algo}] Skipped (xgboost not installed).")
        return None

    fixed_params = {
        k: (None if v == "null" else v)
        for k, v in algo_cfg.get("fixed", {}).items()
    }
    raw_grid = algo_cfg.get("grid", {})

    # Normalise YAML nulls ("null" strings -> Python None)
    grid = {}
    for k, vals in raw_grid.items():
        grid[k] = [None if v == "null" else v for v in vals]

    keys       = list(grid.keys())
    candidates = list(itertools.product(*(grid[k] for k in keys)))

    tie_break = ["f1", "precision", "recall", "accuracy"]
    results   = []
    best      = None  # (score_tuple, row, model)
    best_model = None

    print(f"  [{algo}] {len(candidates)} candidate(s)...")

    for i, values in enumerate(candidates, start=1):
        params = dict(zip(keys, values))

        try:
            model = build_model(algo, params, fixed_params, random_seed)
        except Exception as exc:
            print(f"    Run {i}: build failed — {exc}")
            continue

        # Optional: TimeSeriesSplit cross-validation on X_train
        cv_score = None
        if cv_folds > 1:
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            try:
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=tscv, scoring="roc_auc", n_jobs=-1
                )
                cv_score = float(np.mean(cv_scores))
            except Exception as exc:
                print(f"    Run {i}: CV failed — {exc}")
            # Rebuild so we can fit on the full training set below
            model = build_model(algo, params, fixed_params, random_seed)

        # Train on full training set
        model.fit(X_train, y_train)

        # Evaluate on held-out validation set
        m = eval_on_val(model, X_val, y_val, threshold)

        row = {
            "run":              i,
            "algorithm":        algo,
            "threshold":        threshold,
            "selection_metric": selection_metric,
            **params,
            "cv_roc_auc":       cv_score,
            **m,
        }
        results.append(row)

        # Selection: prefer CV score (more robust) when available, else val score
        primary = cv_score if cv_score is not None else row.get(selection_metric, float("-inf"))
        if primary is None:
            primary = float("-inf")

        score_tuple = (primary,) + tuple(row.get(k, 0.0) for k in tie_break)

        if best is None or score_tuple > best[0]:
            best       = (score_tuple, row, model)
            best_model = model

    if not results:
        print(f"  [{algo}] No successful runs.")
        return None

    results_df = pd.DataFrame(results)

    # Sort log: best first
    sort_cols  = [c for c in [selection_metric, "cv_roc_auc"] + tie_break if c in results_df.columns]
    ascending  = [False] * len(sort_cols)
    results_df = results_df.sort_values(sort_cols, ascending=ascending)

    # Save per-algorithm log
    log_path = tables_dir / f"model_selection_{algo}.csv"
    results_df.to_csv(log_path, index=False)

    # Backward-compatible alias for logistic_regression (used by make_figures.py)
    if algo == "logistic_regression":
        results_df.to_csv(tables_dir / "model_selection_logreg.csv", index=False)

    best_row = best[1]
    print(
        f"    Best {algo}: {selection_metric}={best_row.get(selection_metric, 'N/A'):.4f} | "
        f"F1={best_row['f1']:.4f} | Recall={best_row['recall']:.4f}"
    )

    return results_df, best_model, best_row


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    p   = build_paths(cfg)

    ensure_dir(p["models_dir"])
    ensure_dir(p["tables_dir"])

    # Validate inputs
    for needed in [p["X_train_file"], p["y_train_file"], p["X_val_file"], p["y_val_file"]]:
        if not needed.exists():
            raise FileNotFoundError(
                f"Missing required file: {needed}\n"
                "Run the features step first: python -m src.features.build_features"
            )

    X_train, y_train, y_col = load_split(p["X_train_file"], p["y_train_file"])
    X_val,   y_val,   _     = load_split(p["X_val_file"],   p["y_val_file"])

    # Evaluation / selection settings
    eval_cfg         = cfg.get("evaluation", {})
    threshold        = float(eval_cfg.get("threshold", 0.5))
    selection_metric = str(eval_cfg.get("selection_metric", "roc_auc")).lower()
    cv_folds         = int(eval_cfg.get("cv_folds", 0))
    random_seed      = int(cfg.get("reproducibility", {}).get("random_seed", 42))

    training_cfg = cfg.get("training", {})

    print("=" * 60)
    print("Model Training — Hyperparameter Grid Search")
    if cv_folds > 1:
        print(f"  CV: TimeSeriesSplit(n_splits={cv_folds}) on X_train")
    else:
        print("  CV: disabled (single validation split)")
    print("=" * 60)

    # Supported algorithms in priority order
    algo_map = {
        "logistic_regression": "logistic_regression",
        "random_forest":       "random_forest",
        "xgboost":             "xgboost",
    }

    all_bests = []   # (score, algo_name, model, row)

    for algo_key, algo_label in algo_map.items():
        algo_cfg = training_cfg.get(algo_key, {"enabled": False})
        result   = run_grid_search(
            algo=algo_label,
            algo_cfg=algo_cfg,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            threshold=threshold,
            selection_metric=selection_metric,
            cv_folds=cv_folds,
            random_seed=random_seed,
            tables_dir=p["tables_dir"],
        )
        if result is not None:
            _, best_model, best_row = result
            primary = best_row.get("cv_roc_auc") or best_row.get(selection_metric, 0.0) or 0.0
            all_bests.append((primary, algo_label, best_model, best_row))

    if not all_bests:
        raise RuntimeError("No models were trained successfully. Check your config.yaml.")

    # Save a model comparison table (best of each algorithm)
    comparison_rows = [row for _, _, _, row in all_bests]
    comparison_df   = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(p["comparison_log"], index=False)

    # Overall winner
    all_bests.sort(key=lambda x: x[0], reverse=True)
    _, best_algo, best_model_overall, best_row_overall = all_bests[0]

    # Persist the best model bundle
    joblib.dump(
        {
            "model":           best_model_overall,
            "algorithm":       best_algo,
            "target_col":      y_col,
            "feature_columns": list(X_train.columns),
            "best_params": {
                k: best_row_overall[k]
                for k in best_row_overall
                if k not in {
                    "run", "algorithm", "threshold", "selection_metric",
                    "cv_roc_auc", "roc_auc", "accuracy", "precision", "recall", "f1"
                }
            },
            "validation_metrics_at_threshold": {
                "threshold": threshold,
                "roc_auc":   best_row_overall.get("roc_auc"),
                "accuracy":  best_row_overall.get("accuracy"),
                "precision": best_row_overall.get("precision"),
                "recall":    best_row_overall.get("recall"),
                "f1":        best_row_overall.get("f1"),
            },
        },
        p["model_artifact"],
    )

    print()
    print("=" * 60)
    print(f"Overall best model: {best_algo}")
    print(f"Selection metric:   {selection_metric} = {best_row_overall.get(selection_metric, 'N/A'):.4f}")
    print(f"  F1:       {best_row_overall['f1']:.4f}")
    print(f"  Recall:   {best_row_overall['recall']:.4f}")
    print(f"  Precision:{best_row_overall['precision']:.4f}")
    print(f"  Accuracy: {best_row_overall['accuracy']:.4f}")
    print(f"Saved model:       {p['model_artifact']}")
    print(f"Comparison table:  {p['comparison_log']}")


if __name__ == "__main__":
    main()
