# src/semcon/paths.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # src/semcon/paths.py -> repo root

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
LOGS = ROOT / "logs"