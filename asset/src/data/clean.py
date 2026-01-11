from __future__ import annotations
from pathlib import Path
from src.utils.config import load_config

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def build_paths(cfg: dict) -> dict:
    """Builds canonical input/output paths from config."""
    paths = cfg["paths"]
    files = cfg["files"]

    interim_dir = Path(paths["data_interim_dir"])
    processed_dir = Path(paths["data_processed_dir"])
    tables_dir = Path(paths["tables_dir"])

    # Ingestion writes this filename
    interim_file = interim_dir / "weatherAUS_interim.parquet"

    # Uses YAML for processed output name
    processed_file = processed_dir / files["processed_dataset"]

    cleaning_report_file = tables_dir / "cleaning_report.csv"
    missingness_after_file = tables_dir / "missingness_after_cleaning.csv"

    return {
        "interim_file": interim_file,
        "processed_dir": processed_dir,
        "processed_file": processed_file,
        "tables_dir": tables_dir,
        "cleaning_report_file": cleaning_report_file,
        "missingness_after_file": missingness_after_file,
    }


def clean_weatheraus(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Applies auditable cleaning rules for RainTomorrow classification.

    Rules:
    1) Drop columns with >38% missingness
    2) Drop rows where target RainTomorrow is missing
    3) Ensure Date is datetime
    """
    
    df = df.copy()

    rows_before, cols_before = df.shape

    drop_cols = ["Sunshine", "Evaporation", "Cloud3pm", "Cloud9am"]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop_cols)

    # Ensure Date is datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop rows where the target label is missing
    if "RainTomorrow" not in df.columns:
        raise KeyError("Expected target column 'RainTomorrow' not found in dataset.")
    df = df.dropna(subset=["RainTomorrow"])

    rows_after, cols_after = df.shape

    summary = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "cols_before": cols_before,
        "cols_after": cols_after,
        "dropped_columns": ";".join(existing_drop_cols),
        "dropped_columns_count": len(existing_drop_cols),
        "dropped_rows_missing_target": rows_before - rows_after,  # note: includes any other drops above only if applied later
    }

    return df, summary


def write_reports(df: pd.DataFrame, summary: dict, report_path: Path, missingness_path: Path) -> None:
    # Cleaning summary (one-row table)
    pd.DataFrame([summary]).to_csv(report_path, index=False)

    # Missingness after cleaning
    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_fraction"})
    )
    missing.to_csv(missingness_path, index=False)


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    p = build_paths(cfg)

    if not p["interim_file"].exists():
        raise FileNotFoundError(
            f"Interim file not found: {p['interim_file']}\n"
            "Run ingestion first: python -m src.data.ingest"
        )

    ensure_dir(p["processed_dir"])
    ensure_dir(p["tables_dir"])

    df = pd.read_parquet(p["interim_file"])

    df_clean, summary = clean_weatheraus(df)

    df_clean.to_parquet(p["processed_file"], index=False)

    write_reports(df_clean, summary, p["cleaning_report_file"], p["missingness_after_file"])

    print("Cleaning complete.")
    print(f"Input interim:   {p['interim_file']}")
    print(f"Output processed:{p['processed_file']}")
    print(f"Cleaning report: {p['cleaning_report_file']}")
    print(f"Missingness after cleaning: {p['missingness_after_file']}")
    print(f"Rows: {summary['rows_before']:,} -> {summary['rows_after']:,}")
    print(f"Cols: {summary['cols_before']} -> {summary['cols_after']}")
    print(f"Dropped columns: {summary['dropped_columns']}")


if __name__ == "__main__":
    main()
