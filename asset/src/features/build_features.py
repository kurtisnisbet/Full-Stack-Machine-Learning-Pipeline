from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.utils.config import load_config

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _make_onehot() -> OneHotEncoder:
    """
    Create a OneHotEncoder that works across sklearn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _get_file_key(files: dict, key: str, default_name: str) -> str:
    """Return files[key] if present, else default_name."""
    val = files.get(key)
    return str(val) if val else default_name


def build_paths(cfg: dict) -> dict:
    paths = cfg["paths"]
    files = cfg["files"]

    processed_dir = Path(paths["data_processed_dir"])
    models_dir    = Path(paths["models_dir"])
    tables_dir    = Path(paths["tables_dir"])

    processed_dataset_name = str(files["processed_dataset"])
    processed_file = processed_dir / processed_dataset_name

    X_train_name = _get_file_key(files, "X_train", "X_train.parquet")
    X_val_name = _get_file_key(files, "X_val", "X_val.parquet")
    X_test_name = _get_file_key(files, "X_test", "X_test.parquet")
    y_train_name = _get_file_key(files, "y_train", "y_train.parquet")
    y_val_name = _get_file_key(files, "y_val", "y_val.parquet")
    y_test_name = _get_file_key(files, "y_test", "y_test.parquet")

    report_file = tables_dir / "feature_build_report.csv"

    return {
        "processed_dir":        processed_dir,
        "models_dir":           models_dir,
        "tables_dir":           tables_dir,
        "processed_file":       processed_file,
        "X_train_file":         processed_dir / X_train_name,
        "X_val_file":           processed_dir / X_val_name,
        "X_test_file":          processed_dir / X_test_name,
        "y_train_file":         processed_dir / y_train_name,
        "y_val_file":           processed_dir / y_val_name,
        "y_test_file":          processed_dir / y_test_name,
        "report_file":          report_file,
        "preprocessor_artifact": models_dir / "preprocessor.pkl",
    }


def time_split_indices(df: pd.DataFrame, date_col: str, train_frac: float, val_frac: float, test_frac: float) -> tuple[pd.Index, pd.Index, pd.Index]:
    """
    Chronological time split:
      - sort by Date ascending
      - first train_frac -> train
      - next val_frac -> val
      - final test_frac -> test
    """
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train_fraction + validation_fraction + test_fraction must sum to 1.0") # Sanity check

    df_sorted = df.sort_values(date_col, ascending=True)
    n = len(df_sorted)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    train_idx = df_sorted.index[:n_train]
    val_idx = df_sorted.index[n_train:n_train + n_val]
    test_idx = df_sorted.index[n_train + n_val:]

    if len(test_idx) != n_test:
        raise RuntimeError("Split sizing mismatch; check split fractions.")

    return train_idx, val_idx, test_idx


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    p = build_paths(cfg)

    split_cfg = cfg["split"]
    train_frac = float(split_cfg["train_fraction"])
    val_frac = float(split_cfg["validation_fraction"])
    test_frac = float(split_cfg["test_fraction"])

    if not p["processed_file"].exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {p['processed_file']}\n"
            "Run cleaning first: python -m src.data.clean"
        )

    ensure_dir(p["processed_dir"])
    ensure_dir(p["models_dir"])
    ensure_dir(p["tables_dir"])

    # Loads cleaned data
    df = pd.read_parquet(p["processed_file"])

    # Ensures Date is datetime
    if "Date" not in df.columns:
        raise KeyError("Expected 'Date' column not found; required for time-based split.")
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop rows with invalid dates
    rows_before = len(df)
    df = df.dropna(subset=["Date"]).copy()
    rows_after_date = len(df)

    # Target and leakage exclusions
    target_col = "RainTomorrow"
    if target_col not in df.columns:
        raise KeyError("Expected column 'RainTomorrow' not found.")

    # Drop rows missing target
    df = df.dropna(subset=[target_col]).copy()
    rows_after_target = len(df)

    leakage_cols = ["Date", "Rainfall", target_col]
    for c in leakage_cols:
        if c not in df.columns:
            if c in ("Date", target_col):
                raise KeyError(f"Required column missing: {c}")

    feature_cols = [c for c in df.columns if c not in leakage_cols and c != "Date"]

    X_raw = df[feature_cols].copy()
    y_raw = df[target_col].copy()

    # Identify column types
    numeric_cols = X_raw.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X_raw.columns if c not in numeric_cols]

    # Build preprocessing pipeline, and fits to train only
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_onehot()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Time-based split (chronological)
    train_idx, val_idx, test_idx = time_split_indices(
        df=df, date_col="Date",
        train_frac=train_frac, val_frac=val_frac, test_frac=test_frac
    )

    X_train_raw = X_raw.loc[train_idx]
    X_val_raw = X_raw.loc[val_idx]
    X_test_raw = X_raw.loc[test_idx]

    y_train = y_raw.loc[train_idx].reset_index(drop=True)
    y_val = y_raw.loc[val_idx].reset_index(drop=True)
    y_test = y_raw.loc[test_idx].reset_index(drop=True)

    # Fit on train, transform val/test
    X_train_arr = preprocessor.fit_transform(X_train_raw)
    X_val_arr = preprocessor.transform(X_val_raw)
    X_test_arr = preprocessor.transform(X_test_raw)

    # Feature names and converted back to DataFrame
    feature_names = preprocessor.get_feature_names_out()
    X_train = pd.DataFrame(X_train_arr, columns=feature_names).reset_index(drop=True)
    X_val = pd.DataFrame(X_val_arr, columns=feature_names).reset_index(drop=True)
    X_test = pd.DataFrame(X_test_arr, columns=feature_names).reset_index(drop=True)

    # Persist preprocessor so the Streamlit app can apply it to raw user inputs
    joblib.dump(
        {
            "preprocessor":    preprocessor,
            "numeric_cols":    numeric_cols,
            "categorical_cols": categorical_cols,
            "feature_names":   list(feature_names),
            "leakage_cols":    [c for c in leakage_cols if c in df.columns],
        },
        p["preprocessor_artifact"],
    )

    # Saves outputs
    X_train.to_parquet(p["X_train_file"], index=False)
    X_val.to_parquet(p["X_val_file"], index=False)
    X_test.to_parquet(p["X_test_file"], index=False)

    # Saves y as single-column parquet
    pd.DataFrame({target_col: y_train}).to_parquet(p["y_train_file"], index=False)
    pd.DataFrame({target_col: y_val}).to_parquet(p["y_val_file"], index=False)
    pd.DataFrame({target_col: y_test}).to_parquet(p["y_test_file"], index=False)

    # Report
    date_min = df["Date"].min()
    date_max = df["Date"].max()
    train_date_min = df.loc[train_idx, "Date"].min()
    train_date_max = df.loc[train_idx, "Date"].max()
    val_date_min = df.loc[val_idx, "Date"].min()
    val_date_max = df.loc[val_idx, "Date"].max()
    test_date_min = df.loc[test_idx, "Date"].min()
    test_date_max = df.loc[test_idx, "Date"].max()

    report = pd.DataFrame([{
        "rows_loaded": rows_before,
        "rows_after_drop_bad_date": rows_after_date,
        "rows_after_drop_missing_target": rows_after_target,
        "n_features_before_encoding": len(feature_cols),
        "n_numeric_features": len(numeric_cols),
        "n_categorical_features": len(categorical_cols),
        "n_features_after_encoding": X_train.shape[1],
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "date_min": date_min,
        "date_max": date_max,
        "train_date_min": train_date_min,
        "train_date_max": train_date_max,
        "val_date_min": val_date_min,
        "val_date_max": val_date_max,
        "test_date_min": test_date_min,
        "test_date_max": test_date_max,
        "excluded_leakage_cols": ";".join([c for c in leakage_cols if c in df.columns]),
    }])
    report.to_csv(p["report_file"], index=False)

    print("Feature build complete (Option B).")
    print(f"Processed input: {p['processed_file']}")
    print(f"X_train: {p['X_train_file']}")
    print(f"X_val:   {p['X_val_file']}")
    print(f"X_test:  {p['X_test_file']}")
    print(f"y_train: {p['y_train_file']}")
    print(f"y_val:   {p['y_val_file']}")
    print(f"y_test:  {p['y_test_file']}")
    print(f"Report:  {p['report_file']}")
    print(f"Features after encoding: {X_train.shape[1]}")
    print(f"Rows (train/val/test): {len(X_train):,}/{len(X_val):,}/{len(X_test):,}")
    print(f"Preprocessor: {p['preprocessor_artifact']}")


if __name__ == "__main__":
    main()
