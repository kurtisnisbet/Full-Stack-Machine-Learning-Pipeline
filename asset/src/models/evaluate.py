from __future__ import annotations
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from src.utils.config import load_config

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_file_key(files: dict, key: str, default_name: str) -> str:
    val = files.get(key)
    return str(val) if val else default_name


def build_paths(cfg: dict) -> dict:
    paths = cfg["paths"]
    files = cfg["files"]

    processed_dir = Path(paths["data_processed_dir"])
    tables_dir = Path(paths["tables_dir"])
    figures_dir = Path(paths["figures_dir"])
    models_dir = Path(paths["models_dir"])

    # Inputs
    X_val_file = processed_dir / _get_file_key(files, "X_val", "X_val.parquet")
    y_val_file = processed_dir / _get_file_key(files, "y_val", "y_val.parquet")
    X_test_file = processed_dir / _get_file_key(files, "X_test", "X_test.parquet")
    y_test_file = processed_dir / _get_file_key(files, "y_test", "y_test.parquet")

    model_artifact = Path(files.get("model_artifact", models_dir / "rain_model.joblib"))

    # Outputs
    metrics_table = Path(files.get("metrics_table", tables_dir / "metrics.csv"))
    cm_path = Path(files.get("confusion_matrix", figures_dir / "confusion_matrix.png"))

    return {
        "processed_dir": processed_dir,
        "tables_dir": tables_dir,
        "figures_dir": figures_dir,
        "X_val_file": X_val_file,
        "y_val_file": y_val_file,
        "X_test_file": X_test_file,
        "y_test_file": y_test_file,
        "model_artifact": model_artifact,
        "metrics_table": metrics_table,
        "cm_test_path": cm_path,
        "cm_val_path": cm_path.with_name(cm_path.stem + "_val" + cm_path.suffix),
    }


def coerce_binary_target(y: pd.Series) -> pd.Series:
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


def compute_metrics(y_true: pd.Series, y_prob: pd.Series, y_pred: pd.Series) -> dict:
    roc_auc = None
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = None

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
    }


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, out_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])

    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    p = build_paths(cfg)

    ensure_dir(p["tables_dir"])
    ensure_dir(p["figures_dir"])

    if not p["model_artifact"].exists():
        raise FileNotFoundError(
            f"Model artifact not found: {p['model_artifact']}\n"
            "Train first: python -m src.models.train"
        )

    bundle = joblib.load(p["model_artifact"])
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    target_col = bundle["target_col"]

    def load_split(X_path: Path, y_path: Path) -> tuple[pd.DataFrame, pd.Series]:
        if not X_path.exists():
            raise FileNotFoundError(f"Missing {X_path}. Run features step first.")
        if not y_path.exists():
            raise FileNotFoundError(f"Missing {y_path}. Run features step first.")

        X = pd.read_parquet(X_path)
        y_df = pd.read_parquet(y_path)

        if y_df.shape[1] != 1:
            raise ValueError(f"{y_path} should have exactly one column.")
        y = coerce_binary_target(y_df.iloc[:, 0])

        missing_cols = [c for c in feature_columns if c not in X.columns]
        extra_cols = [c for c in X.columns if c not in feature_columns]
        if missing_cols:
            raise ValueError(f"{X_path} is missing expected feature columns: {missing_cols[:10]} (and more)" if len(missing_cols) > 10 else f"{X_path} is missing expected columns: {missing_cols}")
        X = X[feature_columns]
        if extra_cols:
            pass

        return X, y

    # Evaluate on validation and test
    results = []

    for split_name, X_path, y_path, cm_path in [
        ("val", p["X_val_file"], p["y_val_file"], p["cm_val_path"]),
        ("test", p["X_test_file"], p["y_test_file"], p["cm_test_path"]),
    ]:
        X, y_true = load_split(X_path, y_path)

        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        m = compute_metrics(y_true=y_true, y_prob=y_prob, y_pred=y_pred)
        m.update({
            "split": split_name,
            "rows": len(X),
            "target_col": target_col,
            "threshold": 0.5,
        })
        results.append(m)

        save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            out_path=cm_path,
            title=f"Confusion Matrix ({split_name})",
        )

    metrics_df = pd.DataFrame(results)[
        ["split", "rows", "accuracy", "precision", "recall", "f1", "roc_auc", "threshold", "target_col"]
    ]
    metrics_df.to_csv(p["metrics_table"], index=False)

    print("Evaluation complete.")
    print(f"Metrics saved to: {p['metrics_table']}")
    print(f"Confusion matrix (test): {p['cm_test_path']}")
    print(f"Confusion matrix (val):  {p['cm_val_path']}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
