"""Database access — the only module that talks to SQLite.

One engine factory, one query runner, one registry loader, one
registration helper. All other modules import from here.
"""
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, bindparam, create_engine, text

from semcon import schema
from semcon.paths import ROOT, SQL

DB_PATH = ROOT / "data" / "secom.db"


def get_engine(db_path: Path = DB_PATH) -> Engine:
    """Pass db_path=tmp_path/'test.db' in tests; production code passes nothing."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def run_query(name: str, engine: Engine, **params) -> pd.DataFrame:
    """Run sql/<name>.sql with named parameters. The filename is the query name."""
    query = (SQL / f"{name}.sql").read_text()
    return pd.read_sql(text(query), engine, params=params)


def load_registry(engine: Engine) -> pd.DataFrame:
    return pd.read_sql(text("SELECT * FROM column_registry"), engine)


def feature_columns(registry: pd.DataFrame) -> list[str]:
    """The inclusion contract: active features only.

    X is always built from this list, never by dropping known non-features.
    """
    mask = registry["role"].isin(
        [schema.Role.FEATURE_RAW.value, schema.Role.FEATURE_ENG.value]
    ) & (registry["status"] == schema.Status.ACTIVE.value)
    return registry.loc[mask, "column_name"].tolist()


def register_columns(rows: list[dict], engine: Engine) -> None:
    """Upsert registry rows. Convention: whoever creates a column registers it."""
    if not rows:
        return
    names = [r["column_name"] for r in rows]
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM column_registry WHERE column_name IN :names")
            .bindparams(bindparam("names", expanding=True)),
            {"names": names},
        )
        pd.DataFrame(rows).to_sql("column_registry", conn, if_exists="append", index=False)