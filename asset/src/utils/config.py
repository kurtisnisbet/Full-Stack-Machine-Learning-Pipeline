from __future__ import annotations
from pathlib import Path
import yaml

def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file and return as dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must parse to a dictionary at the top level.")
    return cfg
