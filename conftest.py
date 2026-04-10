"""
conftest.py — Make the asset/ directory importable so tests can do
              `from src.data.clean import ...` without installing the package.
"""
import sys
from pathlib import Path

# Add asset/ to sys.path so `src.*` imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent / "asset"))
