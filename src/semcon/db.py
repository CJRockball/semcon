from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text, Engine
from semcon.paths import ROOT, SQL

DB_PATH = ROOT / "data" / "secom.db"

def get_engine(db_path: Path = DB_PATH) -> Engine:
    return create_engine(f"sqlite:///{db_path}", echo=False)

def run_query(name: str, engine: Engine, **params) -> pd.DataFrame:
    query = (SQL / f"{name}.sql").read_text()
    return pd.read_sql(text(query), engine, params=params)