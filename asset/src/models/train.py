from __future__ import annotations

from pathlib import Path
import itertools
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.utils.config import load_config


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_file_key(files: dict, key: str, default_name: str) -> str:
    v = files.get(key)
    return str(v) if v else default_name


def coerce_binary_target(y: pd.Series) -> pd.Series:
    """
    Convert RainTomorrow labels to 0/1 consistently.
    Accepts Yes/No, Y/N, True/False, 1/0.
    """
    if y.dtype == "bool":
        return y.astype(int)

    y_str = y.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1, "y": 1, "true": 1, "1": 1,
        "no": 0, "n": 0, "false": 0, "0": 0,
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
    models_dir = Path(paths["models_dir"])
    tables_dir = Path(paths["tables_dir"])

    X_train_file = processed_dir / _get_file_key(files, "X_train", "X_train.parquet")
    y_train_file = processed_dir / _get_file_key(files, "y_train", "y_train.parquet")
    X_val_file = processed_dir / _get_file_key(files, "X_val", "X_val.parquet")
    y_val_file = processed_dir / _get_file_key(files, "y_val", "y_val.parquet")

    model_artifact = Path(files.get("model_artifact", models_dir / "rain_model.pkl"))
    selection_log = tables_dir / "model_selection_logreg.csv"

    return {
        "processed_dir": processed_dir,
        "models_dir": models_dir,
        "tables_dir": tables_dir,
        "X_train_file": X_train_file,
        "y_train_file": y_train_file,
        "X_val_file": X_val_file,
        "y_val_file": y_val_file,
        "model_artifact": model_artifact,
        "selection_log": selection_log,
    }


def load_split(X_path: Path, y_path: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    X = pd.read_parquet(X_path)
    y_df = pd.read_parquet(y_path)
    if y_df.shape[1] != 1:
        raise ValueError(f"{y_path} should have exactly one column.")
    y_col = y_df.columns[0]
    y = coerce_binary_target(y_df.iloc[:, 0])
    return X, y, y_col


def eval_on_val(model: LogisticRegression, X_val: pd.DataFrame, y_val: pd.Series, threshold: float) -> dict:
    prob = model.predict_proba(X_val)[:, 1]
    pred = (prob >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_val, prob)
    except Exception:
        auc = None

    return {
        "roc_auc": auc,
        "accuracy": accuracy_score(y_val, pred),
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred, zero_division=0),
        "f1": f1_score(y_val, pred, zero_division=0),
    }


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    p = build_paths(cfg)

    ensure_dir(p["models_dir"])
    ensure_dir(p["tables_dir"])
    if p["model_artifact"].parent != Path("."):
        ensure_dir(p["model_artifact"].parent)

    # Load data
    for needed in [p["X_train_file"], p["y_train_file"], p["X_val_file"], p["y_val_file"]]:
        if not needed.exists():
            raise FileNotFoundError(
                f"Missing required file: {needed}\n"
                "Run features step first: python -m src.features.build_features"
            )

    X_train, y_train, y_col = load_split(p["X_train_file"], p["y_train_file"])
    X_val, y_val, _ = load_split(p["X_val_file"], p["y_val_file"])

    # Selection settings
    eval_cfg = cfg.get("evaluation", {})
    threshold = float(eval_cfg.get("threshold", 0.5))
    selection_metric = str(eval_cfg.get("selection_metric", "roc_auc")).lower()
    # tie-break order after selection_metric:
    # roc_auc -> f1 -> precision -> recall -> accuracy
    tie_break = ["f1", "precision", "recall", "accuracy"]

    # Hyperparameter grid (you can later move these into YAML if you want)
    grid = {
        "C": [0.1, 0.3, 1.0, 3.0, 10.0],
        "class_weight": [None, "balanced"],
    }

    # Fixed training params
    random_seed = int(cfg.get("reproducibility", {}).get("random_seed", 42))
    base_params = {
        "solver": "saga",
        "max_iter": 8000,
        "random_state": random_seed,
    }

    # Train/evaluate all combinations
    keys = list(grid.keys())
    candidates = list(itertools.product(*(grid[k] for k in keys)))

    results = []
    best = None  # (score_tuple, run_row, model)
    best_model = None

    for i, values in enumerate(candidates, start=1):
        params = dict(zip(keys, values))
        run_params = {**base_params, **params}

        model = LogisticRegression(**run_params)
        model.fit(X_train, y_train)

        m = eval_on_val(model, X_val, y_val, threshold=threshold)

        row = {
            "run": i,
            "threshold": threshold,
            "selection_metric": selection_metric,
            **params,
            **m,
        }
        results.append(row)

        # Build a sortable score tuple: primary metric then tie-breakers
        primary = row.get(selection_metric)
        # Handle None by treating as very low
        if primary is None:
            primary = float("-inf")

        score_tuple = (primary,) + tuple(row[k] for k in tie_break)

        if best is None or score_tuple > best[0]:
            best = (score_tuple, row, model)
            best_model = model

    results_df = pd.DataFrame(results)

    # Sort log for readability (best first by selection + tie-breaks)
    sort_cols = [selection_metric] + tie_break
    ascending = [False] * len(sort_cols)
    results_df = results_df.sort_values(sort_cols, ascending=ascending)

    results_df.to_csv(p["selection_log"], index=False)

    # Save the best model bundle
    best_row = best[1]
    joblib.dump(
        {
            "model": best_model,
            "target_col": y_col,
            "feature_columns": list(X_train.columns),
            "best_params": {k: best_row[k] for k in keys},
            "validation_metrics_at_threshold": {
                "threshold": threshold,
                "roc_auc": best_row["roc_auc"],
                "accuracy": best_row["accuracy"],
                "precision": best_row["precision"],
                "recall": best_row["recall"],
                "f1": best_row["f1"],
            },
        },
        p["model_artifact"],
    )

    print("Model selection complete (Logistic Regression).")
    print(f"Trained candidates: {len(candidates)}")
    print(f"Selection metric: {selection_metric} (threshold={threshold})")
    print(f"Best params: { {k: best_row[k] for k in keys} }")
    print("Best validation metrics:")
    print(f"  ROC AUC:  {best_row['roc_auc']}")
    print(f"  Accuracy: {best_row['accuracy']:.6f}")
    print(f"  Precision:{best_row['precision']:.6f}")
    print(f"  Recall:   {best_row['recall']:.6f}")
    print(f"  F1:       {best_row['f1']:.6f}")
    print(f"Saved best model: {p['model_artifact']}")
    print(f"Saved run log:    {p['selection_log']}")


if __name__ == "__main__":
    main()
