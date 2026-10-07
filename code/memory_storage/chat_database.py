import os
import pandas as pd
from uuid import uuid5

import psycopg

CREATE_SQL = \
"""
CREATE TABLE IF NOT EXISTS memories (
    memory_id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    t_created TIMESTAMPTZ NOT NULL,
    t_last_recalled TIMESTAMPTZ,
    consolidation DOUBLE PRECISION DEFAULT 1.0,
    recall_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_memory_occurred_at
    ON memories (t_created);
"""

class ChatDatabase:

    def __init__(
        self,
        pg_dn=None,
        connection=psycopg.connect,
        getenv=os.getenv,
    ):
        self.pg_dsn = pg_dsn or  getenv("PG_DSN") # TODO find a good default
        if self.pg_dsn is None:
            raise Exception('Environment variable PG_DSN must be set')

    def create(self):
        """
        Defines a database with
            memory_id: str
            content: str
            t_created: float
            t_last_recalled: float
            consolidation (g): float = 1.0
            recall_count: int = 0
            metadata: dict 
        Note: we do not currently include fields used for ACT-R
        """
        with connection(self.pg_dsn) as conn:
            conn.execute(CREATE_SQL)
            conn.commit()

    def insert_memory(
        self,
        memory_id,
        content,
        t_created,
        t_last_recalled,
        consolidation = 1.0,
        recall_count = 0.0,
        metadata = None,
        connection=psycopg.connect,
    ):
        """
        Inserts parameter memory into the memory database.
        If we have a duplicate memory_id, replace the content
        """
        memory_id = str(memory_id)
        metadata = metadata or {}

        with connection(self.pg_dsn) as conn:
        conn.execute(
            """
            INSERT INTO memories
                (memory_id, content, t_created, t_last_recalled, consolidation,
                 recall_count, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (memory_id) DO UPDATE SET
                content = EXCLUDED.content,
                t_created = EXCLUDED.t_created,
                t_last_recalled = EXCLUDED.t_last_recalled,
                consolidation = EXCLUDED.consolidation,
                recall_count = EXCLUDED.recall_count,
                metadata = EXCLUDED.metadata,
            """,
            (memory_id, content, t_created, t_last_recalled, consolidation, 
                 recall_count, metadata),
        )
        conn.commit()

        return memory_id

    def retrieve_memories(
        self,
        memory_ids,
        connection=psycopg.connect,
    ):
        """
        Parameters:
            memory_ids (List[uuid5]): UUIDs of memories to retrieve

        Returns:
            memories (DataFrame): memory contents and metadata corresponding to the memory ids
        """
        ids = [str(x) for x in memory_ids]
        with connection(self.pg_dn) as conn:
            rows = conn.execute(
                """
                SELECT content, t_created, t_last_recalled, consolidation,
                    recall_count, metadata
                FROM memories
                WHERE memory_id = ANY(%s::uuid[])
                ORDER BY t_created
                """, (ids,),
            ).fetchall()
        columns = ["memory_id", "content", "t_created", "t_last_recalled", "consolidation", 
               "recall_count", "metadata"]
        return pd.DataFrame(rows, columns=columns)
