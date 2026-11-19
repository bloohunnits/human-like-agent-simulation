"""PostgreSQL persistence for long-term memories and associative connections.

The connection factory is injected into MemoryDatabase so database operations can
be tested without patching psycopg.connect.
"""

import json
import os
from typing import Callable, Optional

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from hou_memory.associative_memory import entity_keys



""" name/type/default for Memory table columns """
MEMORY_COLUMNS = [
    ("memory_id", "UUID PRIMARY KEY", None),
    ("agent_id", "TEXT NOT NULL DEFAULT 'default'", None),
    ("content", "TEXT NOT NULL", None),
    ("t_created", "DOUBLE PRECISION NOT NULL", None),
    ("t_last_recalled", "DOUBLE PRECISION", None),
    ("g", "DOUBLE PRECISION NOT NULL DEFAULT 1.0", None),
    ("recall_count", "INTEGER NOT NULL DEFAULT 0", None),
    ("meta", "JSONB NOT NULL DEFAULT '{}'::jsonb", None),
]

""" name/type/default for Connections table columns """
EDGE_COLUMNS = [
    ("edge_id", "TEXT PRIMARY KEY", None),
    ("agent_id", "TEXT NOT NULL DEFAULT 'default'", None),
    ("node_a", "TEXT NOT NULL", None),
    ("node_b", "TEXT NOT NULL", None),
    ("kind", "TEXT NOT NULL", None),
    ("created", "DOUBLE PRECISION NOT NULL", None),
    ("last_reinforced", "DOUBLE PRECISION NOT NULL", None),
    ("weight", "DOUBLE PRECISION NOT NULL", None),
    ("retention", "DOUBLE PRECISION NOT NULL", None),
]

# Helper functions to build SQL statements

def _col_names(cols) -> list[str]:
    """ Extract column names in canonical order """
    return [c[0] for c in cols]


def _create_table_sql(table: str, cols) -> str:
    """ Build CREATE TABLE DDL from (name, sql_type, _) column specs """
    defs = ",\n    ".join(f"{name} {sql_type}" for name, sql_type, _ in cols)
    return f"CREATE TABLE IF NOT EXISTS {table} (\n    {defs}\n);"


def _upsert_sql(table: str, cols, conflict_col: str) -> str:
    """ Build parameterized INSERT ... ON CONFLICT DO UPDATE from column names """
    names = _col_names(cols)
    col_list = ", ".join(names)
    placeholders = ", ".join(["%s"] * len(names))
    updates = ",\n                ".join(
        f"{name} = EXCLUDED.{name}" for name in names if name != conflict_col
    )
    return (
        f"INSERT INTO {table} ({col_list})\n"
        f"                VALUES ({placeholders})\n"
        f"            ON CONFLICT ({conflict_col}) DO UPDATE SET\n"
        f"                {updates}"
    )


CREATE_SQL = f"""
{_create_table_sql("memories", MEMORY_COLUMNS)}
ALTER TABLE memories ADD COLUMN IF NOT EXISTS agent_id TEXT NOT NULL DEFAULT 'default';
ALTER TABLE memories ADD COLUMN IF NOT EXISTS meta JSONB NOT NULL DEFAULT '{{}}'::jsonb;
CREATE INDEX IF NOT EXISTS idx_memories_t_created
    ON memories (t_created);
CREATE INDEX IF NOT EXISTS idx_memories_meta_gin
    ON memories USING GIN (meta);

{_create_table_sql("memory_connections", EDGE_COLUMNS)}
CREATE INDEX IF NOT EXISTS idx_memory_connections_agent_id
    ON memory_connections (agent_id);
"""


def _normalize_meta(raw_meta) -> dict:
    """ Coerce missing/null/string inputs into a dictionary """
    if raw_meta is None or (isinstance(raw_meta, float) and pd.isna(raw_meta)):
        meta = {}
    elif isinstance(raw_meta, str):
        meta = json.loads(raw_meta)
    elif isinstance(raw_meta, dict):
        meta = dict(raw_meta)
    else:
        raise ValueError("meta must be a dict or JSON object")

    # Validate structured entity lists when present
    if "entities" in meta and meta["entities"] is not None:
        entity_keys(meta["entities"])
    return meta


class MemoryDatabase:
    """Store and retrieve agent-scoped memories and edges in PostgreSQL."""

    def __init__(
        self,
        pg_dsn: Optional[str] = None,
        connection: Callable = psycopg.connect,
        getenv: Callable = os.getenv,
        initialize: bool = True,
    ):
        # Fall back to environment lookup if no explicit DSN is passed
        self.pg_dsn = pg_dsn or getenv("PG_DSN")
        if not self.pg_dsn:
            raise ValueError("A PostgreSQL DSN is required (pass pg_dsn or set PG_DSN).")
        self.connection = connection
        # Bootstrap tables and indexes upfront unless deferred by caller/tests
        if initialize:
            self._initialize_database()

    def _initialize_database(self) -> None:
        """ Run DDL statements and commit schema changes """
        with self.connection(self.pg_dsn) as conn:
            conn.execute(CREATE_SQL)
            conn.commit()

    def upsert_memories(self, memories: pd.DataFrame) -> int:
        """ Write in batch memories into database """

        # Enforce DataFrame input and check required columns before touching the DB
        if not isinstance(memories, pd.DataFrame):
            raise TypeError("memories must be a pandas DataFrame")

        required = {"memory_id", "agent_id", "content", "t_created"}
        missing = required.difference(memories.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if memories.empty:
            return 0

        # Normalize nullability, types, and JSONB metadata per row
        prepared = []
        for row in memories.to_dict(orient="records"):
            if pd.isna(row["memory_id"]) or pd.isna(row["agent_id"]):
                raise ValueError("memory_id and agent_id must not be null")
            t_last_recalled = row.get("t_last_recalled")
            if t_last_recalled is not None and pd.isna(t_last_recalled):
                t_last_recalled = None
            meta = _normalize_meta(row.get("meta"))
            prepared.append((
                str(row["memory_id"]),
                str(row["agent_id"]),
                row["content"],
                float(row["t_created"]),
                t_last_recalled,
                row.get("g", 1.0),
                row.get("recall_count", 0),
                Jsonb(meta),
            ))

        # Generate upsert SQL directly from MEMORY_COLUMNS
        sql = _upsert_sql("memories", MEMORY_COLUMNS, "memory_id")

        # Execute batch upsert and commit
        with self.connection(self.pg_dsn) as conn:
            with conn.cursor() as cursor:
                cursor.executemany(sql, prepared)
            conn.commit()
        return len(prepared)

    def recall_by_id(self, memory_ids) -> pd.DataFrame:
        """ Fetch memories from memory ids """
        # Return an empty DataFrame immediately when no IDs are given
        columns = _col_names(MEMORY_COLUMNS)
        normalized_ids = [str(memory_id) for memory_id in memory_ids]
        if not normalized_ids:
            return pd.DataFrame(columns=columns)

        # Fetch matching memories in chronological order
        select_cols = (lambda cols: ", ".join(_col_names(cols)))(MEMORY_COLUMNS)
        with self.connection(self.pg_dsn) as conn:
            rows = conn.execute(
                f"""
                SELECT {select_cols}
                FROM memories
                WHERE memory_id = ANY(%s::uuid[])
                ORDER BY t_created ASC
                """,
                (normalized_ids,),
            ).fetchall()
        return pd.DataFrame(rows, columns=columns)

    def list_memories(self, agent_id) -> pd.DataFrame:
        """ Fetch all memories for an agent ordered by creation time """
        select_cols = (lambda cols: ", ".join(_col_names(cols)))(MEMORY_COLUMNS)
        with self.connection(self.pg_dsn) as conn:
            rows = conn.execute(
                f"""
                SELECT {select_cols}
                FROM memories
                WHERE agent_id = %s
                ORDER BY t_created ASC
                """,
                (str(agent_id),),
            ).fetchall()
        return pd.DataFrame(rows, columns=_col_names(MEMORY_COLUMNS))

    def list_edges(self, agent_id) -> pd.DataFrame:
        # Fetch all persisted connection rows for an agent
        select_cols = (lambda cols: ", ".join(_col_names(cols)))(EDGE_COLUMNS)
        with self.connection(self.pg_dsn) as conn:
            rows = conn.execute(
                f"""
                SELECT {select_cols}
                FROM memory_connections
                WHERE agent_id = %s
                """,
                (str(agent_id),),
            ).fetchall()
        return pd.DataFrame(rows, columns=_col_names(EDGE_COLUMNS))

    def replace_edges(self, edges: pd.DataFrame) -> int:
        # Validate DataFrame input and required connection columns
        if not isinstance(edges, pd.DataFrame):
            raise TypeError("edges must be a pandas DataFrame")

        required = {
            "edge_id", "agent_id", "node_a", "node_b", "kind",
            "created", "last_reinforced", "weight", "retention",
        }
        missing = required.difference(edges.columns)
        if missing:
            raise ValueError(f"Missing required edge columns: {sorted(missing)}")
        if edges.empty:
            return 0

        # Cast edge fields into SQL parameter tuples
        prepared = [
            (
                str(row["edge_id"]),
                str(row["agent_id"]),
                str(row["node_a"]),
                str(row["node_b"]),
                str(row["kind"]),
                float(row["created"]),
                float(row["last_reinforced"]),
                float(row["weight"]),
                float(row["retention"]),
            )
            for row in edges.to_dict(orient="records")
        ]

        # Generate edge upsert SQL directly from EDGE_COLUMNS
        sql = (lambda cols: _upsert_sql("memory_connections", cols, "edge_id"))(EDGE_COLUMNS)
        with self.connection(self.pg_dsn) as conn:
            with conn.cursor() as cursor:
                cursor.executemany(sql, prepared)
            conn.commit()
        return len(prepared)
