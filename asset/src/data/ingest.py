from __future__ import annotations
from pathlib import Path
from src.utils.config import load_config

import pandas as pd

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_paths(cfg: dict) -> dict:
    """Builds paths from config."""
    paths = cfg.get("paths", {})
    files = cfg.get("files", {})

    raw_dir = Path(paths["data_raw_dir"])
    interim_dir = Path(paths["data_interim_dir"])
    tables_dir = Path(paths["tables_dir"])

    raw_file = raw_dir / files["raw_dataset"]

    interim_file = interim_dir / "weatherAUS_interim.parquet"
    report_file = tables_dir / "ingestion_report.csv"
    missingness_file = tables_dir / "missingness_report.csv"

    return {
        "raw_file": raw_file,
        "interim_dir": interim_dir,
        "tables_dir": tables_dir,
        "interim_file": interim_file,
        "report_file": report_file,
        "missingness_file": missingness_file,
    }


def minimal_parse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal parsing only:
    - Parse Date column if present
    - Standardises column names
    """
    # standardises column names to snake_case for easier reading
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Parse Date if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df


def create_reports(df: pd.DataFrame, report_path: Path, missingness_path: Path) -> None:
    """Write an ingestion summary and missingness table."""
    rows, cols = df.shape
    duplicate_rows = int(df.duplicated().sum())

    date_min = date_max = None
    if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
        date_min = df["Date"].min()
        date_max = df["Date"].max()

    summary = pd.DataFrame(
        [{
            "rows": rows,
            "columns": cols,
            "duplicate_rows_exact": duplicate_rows,
            "date_min": date_min,
            "date_max": date_max,
        }]
    )
    summary.to_csv(report_path, index=False)

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

    if not p["raw_file"].exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {p['raw_file']}\n"
            f"Check that it exists and that files.raw_dataset in YAML matches the filename."
        )

    ensure_dir(p["interim_dir"])
    ensure_dir(p["tables_dir"])

    # Read raw data
    df = pd.read_csv(p["raw_file"])

    # Minimal parsing only
    df = minimal_parse(df)

    # Write interim copy (parquet)
    df.to_parquet(p["interim_file"], index=False)

    # Write basic reports
    create_reports(df, p["report_file"], p["missingness_file"])

    print("Ingestion complete.")
    print(f"Interim dataset: {p['interim_file']}")
    print(f"Ingestion report: {p['report_file']}")
    print(f"Missingness report: {p['missingness_file']}")


if __name__ == "__main__":
    main()
