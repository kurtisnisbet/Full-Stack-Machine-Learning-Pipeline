"""
evaluate.py — Final model evaluation with decision-threshold optimisation.

Loads the best model from rain_model.pkl and evaluates it on both the
validation and held-out test sets.  When optimize_threshold is enabled in
config.yaml, a threshold sweep is performed on the validation set to find
the probability cut-off that maximises F1.  Both the default (0.5) and
the optimal threshold results are written to metrics.csv so they can be
compared directly.

Outputs:
    reports/tables/metrics.csv           – val + test metrics (default & optimal)
    reports/tables/threshold_sweep.csv  – F1 / precision / recall per threshold
    reports/figures/confusion_matrix.png
    reports/figures/confusion_matrix_val.png

Usage:
    python -m src.models.evaluate
    # or via Makefile:  make evaluate
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.config import load_config


# ── Utilities ─────────────────────────────────────────────────────────────────

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_file_key(files: dict, key: str, default_name: str) -> str:
    val = files.get(key)
    return str(val) if val else default_name


def coerce_binary_target(y: pd.Series) -> pd.Series:
    """Convert RainTomorrow labels to 0/1. Accepts Yes/No, True/False, 1/0."""
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
    tables_dir    = Path(paths["tables_dir"])
    figures_dir   = Path(paths["figures_dir"])
    models_dir    = Path(paths["models_dir"])

    cm_path = Path(files.get("confusion_matrix", figures_dir / "confusion_matrix.png"))

    return {
        "processed_dir":       processed_dir,
        "tables_dir":          tables_dir,
        "figures_dir":         figures_dir,
        "X_val_file":          processed_dir / _get_file_key(files, "X_val",  "X_val.parquet"),
        "y_val_file":          processed_dir / _get_file_key(files, "y_val",  "y_val.parquet"),
        "X_test_file":         processed_dir / _get_file_key(files, "X_test", "X_test.parquet"),
        "y_test_file":         processed_dir / _get_file_key(files, "y_test", "y_test.parquet"),
        "model_artifact":      Path(files.get("model_artifact", str(models_dir / "rain_model.pkl"))),
        "metrics_table":       Path(files.get("metrics_table",  str(tables_dir / "metrics.csv"))),
        "threshold_sweep":     tables_dir / "threshold_sweep.csv",
        "cm_test_path":        cm_path,
        "cm_val_path":         cm_path.with_name(cm_path.stem + "_val" + cm_path.suffix),
    }


# ── Metrics helpers ───────────────────────────────────────────────────────────

def compute_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """Compute classification metrics at a given probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    roc_auc = None
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        pass
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   roc_auc,
    }


def sweep_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    steps: int = 19,
) -> tuple[pd.DataFrame, float]:
    """
    Sweep probability thresholds and return a DataFrame of metrics per threshold
    plus the threshold that maximises F1 on the validation set.
    """
    thresholds = np.linspace(0.05, 0.95, steps)
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        rows.append({"threshold": round(float(t), 4), **m})

    sweep_df = pd.DataFrame(rows)

    # Find threshold that maximises F1; break ties by recall (prefer sensitivity)
    best_idx = (
        sweep_df[["f1", "recall"]]
        .apply(tuple, axis=1)
        .idxmax()
    )
    optimal_threshold = float(sweep_df.loc[best_idx, "threshold"])
    return sweep_df, optimal_threshold


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Rain", "Rain"])

    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    p   = build_paths(cfg)

    ensure_dir(p["tables_dir"])
    ensure_dir(p["figures_dir"])

    if not p["model_artifact"].exists():
        raise FileNotFoundError(
            f"Model artifact not found: {p['model_artifact']}\n"
            "Train first: python -m src.models.train"
        )

    bundle          = joblib.load(p["model_artifact"])
    model           = bundle["model"]
    feature_columns = bundle["feature_columns"]
    target_col      = bundle["target_col"]
    algorithm       = bundle.get("algorithm", "unknown")

    eval_cfg           = cfg.get("evaluation", {})
    default_threshold  = float(eval_cfg.get("threshold", 0.5))
    optimize_threshold = bool(eval_cfg.get("optimize_threshold", True))

    def load_split(X_path: Path, y_path: Path) -> tuple[pd.DataFrame, pd.Series]:
        if not X_path.exists():
            raise FileNotFoundError(f"Missing {X_path}. Run features step first.")
        if not y_path.exists():
            raise FileNotFoundError(f"Missing {y_path}. Run features step first.")
        X    = pd.read_parquet(X_path)
        y_df = pd.read_parquet(y_path)
        if y_df.shape[1] != 1:
            raise ValueError(f"{y_path} should have exactly one column.")
        y = coerce_binary_target(y_df.iloc[:, 0])
        missing = [c for c in feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Missing feature columns in {X_path}: {missing[:10]}")
        return X[feature_columns], y

    # ── Step 1: threshold sweep on validation set ─────────────────────────────
    X_val, y_val = load_split(p["X_val_file"], p["y_val_file"])
    y_prob_val   = model.predict_proba(X_val)[:, 1]

    sweep_df, optimal_threshold = sweep_threshold(y_val, y_prob_val)
    sweep_df.to_csv(p["threshold_sweep"], index=False)

    if optimize_threshold:
        active_threshold = optimal_threshold
        print(
            f"Threshold optimisation (val F1): "
            f"default={default_threshold:.2f} → optimal={optimal_threshold:.4f}"
        )
    else:
        active_threshold = default_threshold
        print(f"Using default threshold: {active_threshold}")

    # ── Step 2: evaluate val + test at both thresholds ────────────────────────
    X_test, y_test = load_split(p["X_test_file"], p["y_test_file"])
    y_prob_test    = model.predict_proba(X_test)[:, 1]

    results = []

    for split_name, y_true, y_prob, cm_path in [
        ("val",  y_val,  y_prob_val,  p["cm_val_path"]),
        ("test", y_test, y_prob_test, p["cm_test_path"]),
    ]:
        for thresh_label, thresh in [
            ("default",  default_threshold),
            ("optimal",  active_threshold),
        ]:
            # Skip duplicate row when default == optimal
            if thresh_label == "optimal" and abs(thresh - default_threshold) < 1e-9:
                continue

            m = compute_metrics(y_true, y_prob, threshold=thresh)
            m.update({
                "split":       split_name,
                "rows":        len(y_true),
                "target_col":  target_col,
                "algorithm":   algorithm,
                "threshold":   round(thresh, 4),
                "threshold_type": thresh_label,
            })
            results.append(m)

        # Confusion matrix at the active threshold
        y_pred_active = (y_prob >= active_threshold).astype(int)
        save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred_active,
            out_path=cm_path,
            title=f"Confusion Matrix — {split_name} (threshold={active_threshold:.2f})",
        )

    metrics_df = pd.DataFrame(results)[[
        "split", "threshold_type", "threshold", "rows",
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "target_col", "algorithm",
    ]]
    metrics_df.to_csv(p["metrics_table"], index=False)

    # ── Summary print ─────────────────────────────────────────────────────────
    print()
    print("─" * 60)
    print(f"{'Split':<6}  {'Type':<9}  {'Thresh':>6}  {'AUC':>6}  {'F1':>6}  {'Rec':>6}  {'Prec':>6}  {'Acc':>6}")
    print("─" * 60)
    for _, row in metrics_df.iterrows():
        print(
            f"{row['split']:<6}  {row['threshold_type']:<9}  "
            f"{row['threshold']:>6.3f}  {row['roc_auc']:>6.3f}  "
            f"{row['f1']:>6.3f}  {row['recall']:>6.3f}  "
            f"{row['precision']:>6.3f}  {row['accuracy']:>6.3f}"
        )
    print("─" * 60)
    print(f"\nMetrics:          {p['metrics_table']}")
    print(f"Threshold sweep:  {p['threshold_sweep']}")
    print(f"Confusion (test): {p['cm_test_path']}")
    print(f"Confusion (val):  {p['cm_val_path']}")


if __name__ == "__main__":
    main()
