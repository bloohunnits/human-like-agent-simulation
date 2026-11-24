"""Drivers that keep everything in process memory.

They follow the conventions of MemoryDatabase and EmbeddingDatabase: the same
method names, DataFrame columns and UUID memory ids. They let the memory stores
run and be tested with no database.
"""

from typing import Optional
from uuid import UUID

import numpy as np
import pandas as pd


MEMORY_COLUMNS = [
    "memory_id", "agent_id", "content", "t_created",
    "t_last_recalled", "g", "recall_count", "meta",
]
EDGE_COLUMNS = [
    "edge_id", "agent_id", "node_a", "node_b", "kind",
    "created", "last_reinforced", "weight", "retention",
]
EMBEDDING_COLUMNS = ["memory_id", "agent_id", "embedding"]
SIMILARITY_COLUMNS = EMBEDDING_COLUMNS + ["relevance"]


def _memory_id(
    memory_id,
    )-> str:
    """ Validate and normalize a memory UUID, as the database drivers do """
    try:
        return str(UUID(str(memory_id)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid memory_id UUID: {memory_id!r}") from exc


def _require(
    frame: pd.DataFrame,
    name: str,
    required: set,
    )-> None:
    """ Reject input the database drivers would also reject """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


class InMemoryMemoryDatabase:
    """Store and retrieve agent-scoped memories and edges in a dictionary."""

    def __init__(
        self,
        ):
        self._memories = {}
        self._edges = {}

    def upsert_memories(
        self,
        memories: pd.DataFrame,
        )-> int:
        """ Insert or update memories; return the row count """
        _require(memories, "memories", {"memory_id", "agent_id", "content", "t_created"})
        for row in memories.to_dict(orient="records"):
            if pd.isna(row["memory_id"]) or pd.isna(row["agent_id"]):
                raise ValueError("memory_id and agent_id must not be null")
            t_last_recalled = row.get("t_last_recalled")
            meta = row.get("meta")
            memory_id = _memory_id(row["memory_id"])
            self._memories[memory_id] = {
                "memory_id": memory_id,
                "agent_id": str(row["agent_id"]),
                "content": row["content"],
                "t_created": float(row["t_created"]),
                "t_last_recalled": None if t_last_recalled is None or pd.isna(t_last_recalled) else float(t_last_recalled),
                "g": 1.0 if pd.isna(row.get("g", 1.0)) else float(row.get("g", 1.0)),
                "recall_count": 0 if pd.isna(row.get("recall_count", 0)) else int(row.get("recall_count", 0)),
                "meta": dict(meta) if isinstance(meta, dict) else {},
            }
        return len(memories)

    def _select(
        self,
        rows: list,
        )-> pd.DataFrame:
        """ Rows in creation order, as the SQL driver returns them """
        return pd.DataFrame(sorted(rows, key=lambda r: r["t_created"]), columns=MEMORY_COLUMNS)

    def recall_by_id(
        self,
        memory_ids,
        agent_id: Optional[str] = None,
        )-> pd.DataFrame:
        """ Fetch memories from memory ids, optionally restricted to one agent """
        wanted = {_memory_id(memory_id) for memory_id in memory_ids}
        return self._select([
            row for memory_id, row in self._memories.items()
            if memory_id in wanted and (agent_id is None or row["agent_id"] == str(agent_id))
        ])

    def list_memories(
        self,
        agent_id,
        )-> pd.DataFrame:
        """ Fetch all memories for an agent ordered by creation time """
        return self._select([row for row in self._memories.values() if row["agent_id"] == str(agent_id)])

    def list_edges(
        self,
        agent_id,
        )-> pd.DataFrame:
        """ Fetch all connection rows for an agent """
        rows = [row for row in self._edges.values() if row["agent_id"] == str(agent_id)]
        return pd.DataFrame(rows, columns=EDGE_COLUMNS)

    def replace_edges(
        self,
        edges: pd.DataFrame,
        )-> int:
        """ Insert or update connection rows; return the row count """
        _require(edges, "edges", set(EDGE_COLUMNS))
        for row in edges.to_dict(orient="records"):
            self._edges[str(row["edge_id"])] = {
                **{name: str(row[name]) for name in EDGE_COLUMNS[:5]},
                **{name: float(row[name]) for name in EDGE_COLUMNS[5:]},
            }
        return len(edges)


class InMemoryEmbeddingDatabase:
    """Store and retrieve agent-scoped memory embeddings in a dictionary."""

    def __init__(
        self,
        ):
        self._points = {}

    def upsert_memories(
        self,
        memories: pd.DataFrame,
        )-> int:
        """ Insert or update embeddings; return the row count """
        _require(memories, "memories", set(EMBEDDING_COLUMNS))
        for row in memories.to_dict(orient="records"):
            if row["embedding"] is None or len(row["embedding"]) == 0:
                raise ValueError("embedding must not be null or empty")
            memory_id = _memory_id(row["memory_id"])
            self._points[memory_id] = {
                "memory_id": memory_id,
                "agent_id": str(row["agent_id"]),
                "embedding": [float(value) for value in row["embedding"]],
            }
        return len(memories)

    def recall_by_id(
        self,
        ids,
        agent_id: Optional[str] = None,
        )-> pd.DataFrame:
        """ Return requested embeddings in the order asked, optionally restricted to one agent """
        rows = [self._points.get(_memory_id(memory_id)) for memory_id in ids]
        rows = [row for row in rows if row and (agent_id is None or row["agent_id"] == str(agent_id))]
        return pd.DataFrame(rows, columns=EMBEDDING_COLUMNS)

    def recall_by_similarity(
        self,
        embedding,
        k: int = 5,
        agent_id: Optional[str] = None,
        )-> pd.DataFrame:
        """
        Return the k most similar embeddings, optionally restricted to an agent

        Parameters:
            embedding: the query vector. relevance is its cosine with each
                stored vector, larger is more similar
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        query = np.asarray([float(value) for value in embedding])
        if query.size == 0:
            raise ValueError("embedding must not be empty")
        rows = []
        for row in self._points.values():
            if agent_id is not None and row["agent_id"] != str(agent_id):
                continue
            vector = np.asarray(row["embedding"])
            denom = np.linalg.norm(query) * np.linalg.norm(vector)
            rows.append({**row, "relevance": 0.0 if denom == 0 else float(np.dot(query, vector) / denom)})
        rows.sort(key=lambda r: (-r["relevance"], r["memory_id"]))
        return pd.DataFrame(rows[:k], columns=SIMILARITY_COLUMNS)
