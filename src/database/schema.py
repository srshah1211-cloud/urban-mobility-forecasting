from __future__ import annotations

from pathlib import Path

from src.config import paths
from src.database.connection import get_engine


class SchemaApplyError(RuntimeError):
    pass


def apply_schema(sql_dir: Path = paths.root / "sql") -> None:
    sql_files = [
        sql_dir / "001_create_tables.sql",
        sql_dir / "002_create_indexes.sql",
    ]
    try:
        raw_connection = get_engine().raw_connection()
        try:
            with raw_connection.cursor() as cursor:
                for sql_file in sql_files:
                    cursor.execute(sql_file.read_text())
            raw_connection.commit()
        finally:
            raw_connection.close()
    except Exception as exc:
        raise SchemaApplyError(str(exc)) from exc
