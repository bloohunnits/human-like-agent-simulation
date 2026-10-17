"""PostgreSQL persistence for long-term memories and chat messages.

The connection factory is injected into ChatDatabase so database operations can
be tested without patching psycopg.connect.
"""

import os
from typing import Callable, Optional

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb


MEMORY_COLUMNS = [
    "memory_id",
    "agent_id",
    "content",
    "role",
    "t_created",
    "t_last_recalled",
    "g",
    "recall_count",
    "metadata",
]

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id UUID PRIMARY KEY,
    agent_id TEXT NOT NULL DEFAULT 'default',
    content TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    t_created DOUBLE PRECISION NOT NULL,
    t_last_recalled DOUBLE PRECISION,
    g DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    recall_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
);
ALTER TABLE memories ADD COLUMN IF NOT EXISTS agent_id TEXT NOT NULL DEFAULT 'default';
ALTER TABLE memories ADD COLUMN IF NOT EXISTS score DOUBLE PRECISION;
CREATE INDEX IF NOT EXISTS idx_memories_agent_score
    ON memories (agent_id, score DESC);
CREATE INDEX IF NOT EXISTS idx_memories_t_created
    ON memories (t_created);
"""


class MemoryDatabase:
    """Store and retrieve agent-scoped memories in PostgreSQL."""

    def __init__(
        self,
        pg_dsn: Optional[str] = None,
        connection: Callable = psycopg.connect,
        getenv: Callable = os.getenv,
        initialize: bool = True,
    ):
        """Create the database wrapper.

        Args:
            pg_dsn: PostgreSQL DSN; defaults to the PG_DSN environment variable.
            connection: Injected PostgreSQL connection factory.
            getenv: Injected environment lookup, useful for tests.
            initialize: Create/upgrade the memories table on construction.
        """
        self.pg_dsn = pg_dsn or getenv("PG_DSN")
        if not self.pg_dsn:
            raise ValueError("A PostgreSQL DSN is required (pass pg_dsn or set PG_DSN).")
        self.connection = connection
        if initialize:
            self._initialize_database()

    def _initialize_database(self) -> None:
        with self.connection(self.pg_dsn) as conn:
            conn.execute(CREATE_SQL)
            conn.commit()

    def upsert_memories(self, memories: pd.DataFrame) -> int:
        """Insert or update many memories from a DataFrame; return row count.

        Required columns: memory_id, agent_id, content, t_created.
        Optional columns: role, t_last_recalled, g, recall_count, metadata, score.
        If score is absent from the DataFrame, inserts receive NULL and existing
        scores are preserved on conflict. If score is present but null for a row,
        that row's stored score is cleared. Other absent optional columns use
        defaults on inserts and are reset to their defaults on conflict.
        """
        if not isinstance(memories, pd.DataFrame):
            raise TypeError("memories must be a pandas DataFrame")

        required = {"memory_id", "agent_id", "content", "t_created"}
        missing = required.difference(memories.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if memories.empty:
            return 0

        prepared = []
        for row in memories.to_dict(orient="records"):
            if pd.isna(row["memory_id"]) or pd.isna(row["agent_id"]):
                raise ValueError("memory_id and agent_id must not be null")
            metadata = row.get("metadata", {})
            if metadata is None or (isinstance(metadata, float) and pd.isna(metadata)):
                metadata = {}
            if not isinstance(metadata, dict):
                raise TypeError("metadata values must be dictionaries")
            score = row.get("score")
            t_last_recalled = row.get("t_last_recalled")
            if t_last_recalled is not None and pd.isna(t_last_recalled):
                t_last_recalled = None
            prepared.append((
                str(row["memory_id"]),
                str(row["agent_id"]),
                row["content"],
                row.get("role", "user"),
                float(row["t_created"]),
                t_last_recalled,
                row.get("g", 1.0),
                row.get("recall_count", 0),
                Jsonb(metadata),
            ))

        update_fields = [
            "agent_id", "content", "role", "t_created", "t_last_recalled",
            "g", "recall_count", "metadata",
        ]
        update_sql = ",\n                ".join(
            f"{field} = EXCLUDED.{field}" for field in update_fields
        )
        sql = f"""
            INSERT INTO memories
                (memory_id, agent_id, content, role, t_created,
                 t_last_recalled, g, recall_count, metadata, score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (memory_id) DO UPDATE SET
                {update_sql}
        """
        with self.connection(self.pg_dsn) as conn:
            with conn.cursor() as cursor:
                cursor.executemany(sql, prepared)
            conn.commit()
        return len(prepared)

    def recall_by_id(self, ids, agent_id) -> pd.DataFrame:
        """Return requested memories belonging to agent_id, ordered by creation time."""
        columns = MEMORY_COLUMNS.copy()
        ids = [str(memory_id) for memory_id in ids]
        if not ids:
            return pd.DataFrame(columns=columns)

        with self.connection(self.pg_dsn) as conn:
            rows = conn.execute(
                """
                SELECT memory_id, agent_id, content, role, t_created,
                       t_last_recalled, g, recall_count, metadata
                FROM memories
                WHERE memory_id = ANY(%s::uuid[]) AND agent_id = %s
                ORDER BY t_created ASC
                """,
                (ids, str(agent_id)),
            ).fetchall()
        return pd.DataFrame(rows, columns=columns)
