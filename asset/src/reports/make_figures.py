"""
make_figures.py — Generate all diagnostic figures for the pipeline report.

Existing figures:
    class_distribution_train.png   Target class balance in training set
    roc_curve_test.png             ROC curve on test set
    pr_curve_test.png              Precision-Recall curve on test set
    hyperparameter_curve_logreg.png  Val ROC-AUC vs regularisation strength
    missingness_heatmap_interim.png  Missing-value heatmap (interim data)
    missingness_fraction.png         Top-N columns by missing fraction

New figures:
    shap_summary.png               SHAP feature importance (top-20 features)
    calibration_curve.png          Reliability diagram — predicted probability vs empirical
    model_comparison.png           Bar chart of val ROC-AUC across algorithms
    threshold_curve.png            F1 / Precision / Recall vs decision threshold

Usage:
    python -m src.reports.make_figures
    # or via Makefile:  make figures
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.utils.config import load_config


# ── Utilities ─────────────────────────────────────────────────────────────────

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def coerce_binary_target(y: pd.Series) -> pd.Series:
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
        raise ValueError(f"Unrecognized target labels: {bad}")
    return y_mapped.astype(int)


def _get_file_key(files: dict, key: str, default_name: str) -> str:
    v = files.get(key)
    return str(v) if v else default_name


def build_paths(cfg: dict) -> dict:
    paths = cfg["paths"]
    files = cfg["files"]

    processed_dir = Path(paths["data_processed_dir"])
    figures_dir   = Path(paths["figures_dir"])
    tables_dir    = Path(paths["tables_dir"])
    models_dir    = Path(paths["models_dir"])

    return {
        "figures_dir":          figures_dir,
        "tables_dir":           tables_dir,
        "model_artifact":       Path(files.get("model_artifact", str(models_dir / "rain_model.pkl"))),
        "X_test":               processed_dir / _get_file_key(files, "X_test", "X_test.parquet"),
        "y_test":               processed_dir / _get_file_key(files, "y_test", "y_test.parquet"),
        "X_train":              processed_dir / _get_file_key(files, "X_train", "X_train.parquet"),
        "y_train":              processed_dir / _get_file_key(files, "y_train", "y_train.parquet"),
        "missingness_before":   tables_dir / "missingness_report.csv",
        "missingness_after":    tables_dir / "missingness_after_cleaning.csv",
        "model_selection_log":  tables_dir / "model_selection_logreg.csv",
        "model_comparison_log": tables_dir / "model_comparison.csv",
        "threshold_sweep":      tables_dir / "threshold_sweep.csv",
        # Output paths
        "roc_curve":            figures_dir / "roc_curve_test.png",
        "pr_curve":             figures_dir / "pr_curve_test.png",
        "class_dist":           figures_dir / "class_distribution_train.png",
        "missingness_heatmap":  figures_dir / "missingness_heatmap_interim.png",
        "missingness_bar":      figures_dir / "missingness_fraction.png",
        "hyperparam_curve":     figures_dir / "hyperparameter_curve_logreg.png",
        "shap_summary":         figures_dir / "shap_summary.png",
        "calibration_curve":    figures_dir / "calibration_curve.png",
        "model_comparison":     figures_dir / "model_comparison.png",
        "threshold_curve":      figures_dir / "threshold_curve.png",
    }


def load_xy(
    X_path: Path, y_path: Path, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    X    = pd.read_parquet(X_path)
    y_df = pd.read_parquet(y_path)
    if y_df.shape[1] != 1:
        raise ValueError(f"{y_path} must have exactly one column.")
    y = coerce_binary_target(y_df.iloc[:, 0])
    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        raise ValueError(f"{X_path} missing expected columns (first 10): {missing[:10]}")
    return X[feature_columns], y


# ── Figure generators — existing ─────────────────────────────────────────────

def plot_class_distribution(y_train: pd.Series, out_path: Path) -> None:
    counts = y_train.value_counts().sort_index()
    labels = ["No Rain", "Rain"] if list(counts.index) == [0, 1] else [str(i) for i in counts.index]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, counts.values, color=["steelblue", "coral"])
    ax.set_title("Target Distribution (Train): RainTomorrow")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}", ha="center", va="bottom")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true: pd.Series, y_prob: np.ndarray, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="steelblue", lw=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_title("ROC Curve (Test Set)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(y_true: pd.Series, y_prob: np.ndarray, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"AP = {ap:.3f}", color="coral", lw=2)
    ax.set_title("Precision–Recall Curve (Test Set)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_hyperparameter_curve(log_path: Path, out_path: Path) -> None:
    if not log_path.exists():
        print(f"  Skipping hyperparameter curve — log not found: {log_path}")
        return

    df = pd.read_csv(log_path)
    required = {"C", "class_weight", "roc_auc"}
    if required - set(df.columns):
        return

    agg = df.groupby(["class_weight", "C"], as_index=False).agg(roc_auc=("roc_auc", "max"))

    fig, ax = plt.subplots()
    colours = ["steelblue", "coral", "seagreen", "purple"]
    for idx, cw in enumerate(sorted(agg["class_weight"].astype(str).unique())):
        sub = agg[agg["class_weight"].astype(str) == cw].sort_values("C")
        ax.plot(sub["C"], sub["roc_auc"], marker="o",
                label=f"class_weight={cw}", color=colours[idx % len(colours)])

    ax.set_xscale("log")
    ax.set_title("Val ROC-AUC vs Regularisation Strength (Logistic Regression)")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("Validation ROC-AUC")
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_missingness_from_tables(before_csv: Path, out_path: Path) -> None:
    if not before_csv.exists():
        return
    before = pd.read_csv(before_csv)
    if "column" not in before.columns or "missing_fraction" not in before.columns:
        return
    before = before.sort_values("missing_fraction", ascending=False).head(25)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(before["column"][::-1], before["missing_fraction"][::-1], color="steelblue")
    ax.axvline(0.38, color="red", linestyle="--", label="Drop threshold (38%)")
    ax.set_title("Missing Value Fractions by Column (Before Cleaning)")
    ax.set_xlabel("Missing Fraction")
    ax.legend()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_missingness_heatmap_from_interim(cfg: dict, out_path: Path, sample_rows: int = 1200) -> None:
    paths       = cfg["paths"]
    interim_dir = Path(paths["data_interim_dir"])
    candidates  = list(interim_dir.glob("*.parquet"))
    if not candidates:
        return

    df = pd.read_parquet(candidates[0])
    if len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=int(cfg.get("reproducibility", {}).get("random_seed", 42)))

    miss = df.isna().astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(miss, aspect="auto", interpolation="nearest", cmap="Blues")
    ax.set_title("Missingness Heatmap (Interim Data Sample) — Blue = Missing")
    ax.set_xlabel("Features")
    ax.set_ylabel("Sampled Rows")

    cols = df.columns.tolist()
    step = max(1, len(cols) // 25)
    idx  = list(range(0, len(cols), step))
    ax.set_xticks(idx)
    ax.set_xticklabels([cols[i] for i in idx], rotation=90, fontsize=8)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Figure generators — new ───────────────────────────────────────────────────

def plot_shap_summary(
    model,
    X_test: pd.DataFrame,
    out_path: Path,
    max_features: int = 20,
    sample_size: int = 500,
) -> None:
    """
    SHAP summary bar chart showing mean |SHAP value| for the top features.
    Uses a random subsample of test rows to keep computation fast.
    """
    if not SHAP_AVAILABLE:
        print("  Skipping SHAP — shap not installed (pip install shap).")
        return

    # Subsample for speed
    if len(X_test) > sample_size:
        rng       = np.random.default_rng(42)
        sample_idx = rng.choice(len(X_test), size=sample_size, replace=False)
        X_sample   = X_test.iloc[sample_idx].reset_index(drop=True)
    else:
        X_sample = X_test.reset_index(drop=True)

    try:
        # TreeExplainer is fast for tree-based models; fall back to generic Explainer
        model_type = type(model).__name__.lower()
        if any(t in model_type for t in ("forest", "xgb", "tree", "boost", "gradient")):
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            # RandomForest returns list [class0, class1]; take class 1
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            masker      = shap.maskers.Independent(X_sample)
            explainer   = shap.LinearExplainer(model, masker)
            shap_values = explainer.shap_values(X_sample)

        mean_abs = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.Series(mean_abs, index=X_sample.columns)
        top = feature_importance.nlargest(max_features).sort_values()

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(top.index, top.values, color="steelblue")
        ax.set_title(f"SHAP Feature Importance — Top {max_features} Features")
        ax.set_xlabel("Mean |SHAP Value|")
        ax.tick_params(axis="y", labelsize=9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    except Exception as exc:
        print(f"  SHAP plot failed: {exc}")


def plot_calibration_curve(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_path: Path,
) -> None:
    """
    Reliability diagram: compares predicted probabilities to observed
    frequencies.  A perfectly calibrated model sits on the diagonal.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    CalibrationDisplay.from_estimator(
        model,
        X_test,
        y_test,
        n_bins=10,
        ax=ax,
        name=type(model).__name__,
        color="steelblue",
    )
    ax.set_title("Calibration Curve (Test Set)\nPerfect calibration = diagonal")
    ax.legend(loc="upper left")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(comparison_csv: Path, out_path: Path) -> None:
    """
    Bar chart comparing val ROC-AUC for the best model from each algorithm.
    """
    if not comparison_csv.exists():
        print(f"  Skipping model comparison — file not found: {comparison_csv}")
        return

    df = pd.read_csv(comparison_csv)
    if "algorithm" not in df.columns or "roc_auc" not in df.columns:
        print("  Skipping model comparison — missing columns.")
        return

    # One row per algorithm (best already selected in train.py)
    algo_labels = {
        "logistic_regression": "Logistic\nRegression",
        "random_forest":       "Random\nForest",
        "xgboost":             "XGBoost",
    }
    df["label"] = df["algorithm"].map(algo_labels).fillna(df["algorithm"])

    colours = ["steelblue", "coral", "seagreen"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(df["label"], df["roc_auc"],
                  color=colours[:len(df)], edgecolor="white", width=0.5)
    for bar, val in zip(bars, df["roc_auc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(max(0, df["roc_auc"].min() - 0.05), min(1.0, df["roc_auc"].max() + 0.05))
    ax.set_title("Model Comparison — Validation ROC-AUC (best run per algorithm)")
    ax.set_ylabel("Validation ROC-AUC")
    ax.axhline(y=0.5, color="grey", linestyle="--", linewidth=0.8, label="Chance")
    ax.legend(loc="lower right")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_curve(sweep_csv: Path, out_path: Path) -> None:
    """
    Line plot of F1, Precision, and Recall against the decision threshold,
    with the F1-maximising threshold highlighted.
    """
    if not sweep_csv.exists():
        print(f"  Skipping threshold curve — sweep not found: {sweep_csv}")
        return

    df = pd.read_csv(sweep_csv)
    required = {"threshold", "f1", "precision", "recall"}
    if required - set(df.columns):
        return

    best_idx   = df["f1"].idxmax()
    best_thresh = df.loc[best_idx, "threshold"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["threshold"], df["f1"],        label="F1",        color="steelblue", lw=2)
    ax.plot(df["threshold"], df["precision"],  label="Precision", color="coral",     lw=2)
    ax.plot(df["threshold"], df["recall"],     label="Recall",    color="seagreen",  lw=2)
    ax.axvline(best_thresh, color="black", linestyle="--", lw=1.2,
               label=f"Optimal threshold = {best_thresh:.2f}")
    ax.set_title("Decision Threshold Sweep (Validation Set)")
    ax.set_xlabel("Probability Threshold")
    ax.set_ylabel("Score")
    ax.legend(loc="best")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    p   = build_paths(cfg)
    ensure_dir(p["figures_dir"])

    bundle          = joblib.load(p["model_artifact"])
    model           = bundle["model"]
    feature_columns = bundle["feature_columns"]

    X_test, y_test  = load_xy(p["X_test"], p["y_test"], feature_columns)
    X_train, y_train = load_xy(p["X_train"], p["y_train"], feature_columns)

    y_prob_test = model.predict_proba(X_test)[:, 1]

    # ── Existing figures ──────────────────────────────────────────────────────
    print("Generating figures...")
    plot_class_distribution(y_train, p["class_dist"])
    print("  [1/10] class_distribution_train.png")

    plot_roc_curve(y_test, y_prob_test, p["roc_curve"])
    print("  [2/10] roc_curve_test.png")

    plot_pr_curve(y_test, y_prob_test, p["pr_curve"])
    print("  [3/10] pr_curve_test.png")

    plot_hyperparameter_curve(p["model_selection_log"], p["hyperparam_curve"])
    print("  [4/10] hyperparameter_curve_logreg.png")

    plot_missingness_from_tables(p["missingness_before"], p["missingness_bar"])
    print("  [5/10] missingness_fraction.png")

    plot_missingness_heatmap_from_interim(cfg, p["missingness_heatmap"])
    print("  [6/10] missingness_heatmap_interim.png")

    # ── New figures ───────────────────────────────────────────────────────────
    plot_shap_summary(model, X_test, p["shap_summary"])
    print("  [7/10] shap_summary.png")

    plot_calibration_curve(model, X_test, y_test, p["calibration_curve"])
    print("  [8/10] calibration_curve.png")

    plot_model_comparison(p["model_comparison_log"], p["model_comparison"])
    print("  [9/10] model_comparison.png")

    plot_threshold_curve(p["threshold_sweep"], p["threshold_curve"])
    print("  [10/10] threshold_curve.png")

    print(f"\nAll figures saved to: {p['figures_dir']}")


if __name__ == "__main__":
    main()
