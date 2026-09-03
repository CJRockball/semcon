# src/semcon/paths.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # src/semcon/paths.py -> repo root

DATA = ROOT / "data"
DATA_RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"
LOGS = ROOT / "logs"
SQL = ROOT / "sql"