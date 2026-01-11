from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)

from src.utils.config import load_config


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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


def _get_file_key(files: dict, key: str, default_name: str) -> str:
    v = files.get(key)
    return str(v) if v else default_name


def build_paths(cfg: dict) -> dict:
    paths = cfg["paths"]
    files = cfg["files"]

    processed_dir = Path(paths["data_processed_dir"])
    interim_dir = Path(paths["data_interim_dir"])
    raw_dir = Path(paths["data_raw_dir"])
    figures_dir = Path(paths["figures_dir"])
    tables_dir = Path(paths["tables_dir"])
    models_dir = Path(paths["models_dir"])

    # Model + features
    model_artifact = Path(files.get("model_artifact", models_dir / "rain_model.pkl"))
    X_test = processed_dir / _get_file_key(files, "X_test", "X_test.parquet")
    y_test = processed_dir / _get_file_key(files, "y_test", "y_test.parquet")
    X_train = processed_dir / _get_file_key(files, "X_train", "X_train.parquet")
    y_train = processed_dir / _get_file_key(files, "y_train", "y_train.parquet")

    # Missingness tables
    missingness_before = tables_dir / "missingness_report.csv"
    missingness_after = tables_dir / "missingness_after_cleaning.csv"

    # Hyperparameter log
    model_selection_log = tables_dir / "model_selection_logreg.csv"

    # Outputs
    out = {
        "figures_dir": figures_dir,
        "tables_dir": tables_dir,
        "model_artifact": model_artifact,
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
        "missingness_before": missingness_before,
        "missingness_after": missingness_after,
        "model_selection_log": model_selection_log,
        "roc_curve": figures_dir / "roc_curve_test.png",
        "pr_curve": figures_dir / "pr_curve_test.png",
        "class_dist": figures_dir / "class_distribution_train.png",
        "missingness_heatmap": figures_dir / "missingness_heatmap_interim.png",
        "missingness_bar": figures_dir / "missingness_fraction.png",
        "hyperparam_curve": figures_dir / "hyperparameter_curve_logreg.png",
    }
    return out


def load_xy(X_path: Path, y_path: Path, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.read_parquet(X_path)
    y_df = pd.read_parquet(y_path)
    if y_df.shape[1] != 1:
        raise ValueError(f"{y_path} must have exactly one column.")
    y = coerce_binary_target(y_df.iloc[:, 0])

    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        raise ValueError(f"{X_path} missing expected columns (first 10): {missing[:10]}")
    X = X[feature_columns]
    return X, y


# -------------
# Figure generators
# -------------
def plot_class_distribution(y_train: pd.Series, out_path: Path) -> None:
    counts = y_train.value_counts().sort_index()
    labels = ["No", "Yes"] if list(counts.index) == [0, 1] else [str(i) for i in counts.index]

    fig, ax = plt.subplots()
    ax.bar(labels, counts.values)
    ax.set_title("Target Distribution (Train): RainTomorrow")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true: pd.Series, y_prob: np.ndarray, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_title("ROC Curve (Test)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(y_true: pd.Series, y_prob: np.ndarray, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_title("Precision–Recall Curve (Test)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_hyperparameter_curve(log_path: Path, out_path: Path) -> None:
    if not log_path.exists():
        raise FileNotFoundError(
            f"Cannot find hyperparameter log: {log_path}\n"
            "Expected output from: python -m src.models.train"
        )

    df = pd.read_csv(log_path)

    required = {"C", "class_weight", "roc_auc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{log_path} missing columns: {sorted(missing)}")

    # Aggregate in case of duplicates
    agg = df.groupby(["class_weight", "C"], as_index=False).agg(
        roc_auc=("roc_auc", "max"),
        f1=("f1", "max"),
    )

    fig, ax = plt.subplots()

    for cw in sorted(agg["class_weight"].astype(str).unique()):
        sub = agg[agg["class_weight"].astype(str) == cw].sort_values("C")
        ax.plot(sub["C"], sub["roc_auc"], marker="o", label=f"class_weight={cw}")

    ax.set_xscale("log")
    ax.set_title("Validation ROC AUC vs Regularisation Strength (Logistic Regression)")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("Validation ROC AUC")
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_missingness_from_tables(before_csv: Path, after_csv: Path, out_path: Path) -> None:
    """
    Optional: Creates a clean bar chart comparing missing fractions before/after cleaning.
    Uses your existing reports if present.
    """
    if not before_csv.exists():
        return  # silently skip
    before = pd.read_csv(before_csv)
    if "column" not in before.columns or "missing_fraction" not in before.columns:
        return

    before = before.sort_values("missing_fraction", ascending=False).head(25)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(before["column"][::-1], before["missing_fraction"][::-1])
    ax.set_title("Top Missingness Fractions (Before Cleaning)")
    ax.set_xlabel("Missing Fraction")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_missingness_heatmap_from_interim(cfg: dict, out_path: Path, sample_rows: int = 1200) -> None:
    """
    Heatmap-style visual of missingness patterns across rows/columns.
    Uses interim Parquet if present (generated by ingest.py).
    We sample rows to keep the image readable and generation fast.
    """
    paths = cfg["paths"]

    interim_dir = Path(paths["data_interim_dir"])

    candidates = list(interim_dir.glob("*.parquet"))
    if not candidates:
        return

    interim_file = candidates[0]
    df = pd.read_parquet(interim_file)

    # Sample to keep heatmap manageable
    if len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=int(cfg.get("reproducibility", {}).get("random_seed", 42)))

    # Build missingness matrix (1 = missing, 0 = present)
    miss = df.isna().astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(miss, aspect="auto", interpolation="nearest")
    ax.set_title("Missingness Heatmap (Interim Data Sample)\n1=Missing, 0=Present")
    ax.set_xlabel("Features")
    ax.set_ylabel("Sampled Rows")

    # Label a subset of columns to avoid clutter
    cols = df.columns.tolist()
    if len(cols) <= 25:
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=8)
    else:
        step = max(1, len(cols) // 25)
        idx = list(range(0, len(cols), step))
        ax.set_xticks(idx)
        ax.set_xticklabels([cols[i] for i in idx], rotation=90, fontsize=8)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------
# Main
# ------------
def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    p = build_paths(cfg)
    ensure_dir(p["figures_dir"])

    # Load model bundle
    bundle = joblib.load(p["model_artifact"])
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    # Load test split and predict probabilities
    X_test, y_test = load_xy(p["X_test"], p["y_test"], feature_columns)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # Load train targets for class distribution
    X_train, y_train = load_xy(p["X_train"], p["y_train"], feature_columns)

    # Figures
    plot_class_distribution(y_train, p["class_dist"])
    plot_roc_curve(y_test, y_prob_test, p["roc_curve"])
    plot_pr_curve(y_test, y_prob_test, p["pr_curve"])

    # Hyperparameter curve (from train log)
    plot_hyperparameter_curve(p["model_selection_log"], p["hyperparam_curve"])

    # Missingness visuals
    plot_missingness_from_tables(p["missingness_before"], p["missingness_after"], p["missingness_bar"])
    plot_missingness_heatmap_from_interim(cfg, p["missingness_heatmap"], sample_rows=1200)

    print("Figure generation complete.")
    print(f"Saved to: {p['figures_dir']}")
    print("Created:")
    for k in ["class_dist", "roc_curve", "pr_curve", "hyperparam_curve", "missingness_bar", "missingness_heatmap"]:
        print(f"  - {Path(p[k]).name}")


if __name__ == "__main__":
    main()
