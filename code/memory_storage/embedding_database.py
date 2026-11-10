"""Qdrant persistence for memory embeddings.

The Qdrant client can be injected so database operations can be tested without
connecting to a live Qdrant instance.
"""

import os
from typing import Optional
from uuid import UUID

import pandas as pd
from qdrant_client import QdrantClient, models


MEMORY_COLUMNS = ["memory_id", "agent_id", "embedding"]
SIMILARITY_COLUMNS = MEMORY_COLUMNS + ["relevance"]


class EmbeddingDatabase:
    """Store and retrieve agent-scoped memory embeddings in Qdrant."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "memories",
        vector_size: int = 384,
        distance: models.Distance = models.Distance.COSINE,
        client: Optional[QdrantClient] = None,
        getenv=os.getenv,
        initialize: bool = True,
    ):
        """Create the database wrapper.

        Args:
            url: Qdrant URL; defaults to QDRANT_URL, then localhost:6333.
            api_key: Optional Qdrant API key; defaults to QDRANT_API_KEY.
            collection_name: Collection used to store memory vectors.
            vector_size: Embedding dimension used when creating the collection.
            distance: Qdrant distance metric used when creating the collection.
            client: Optional injected Qdrant client. When supplied, url and
                api_key are ignored.
            getenv: Injected environment lookup, useful for tests.
            initialize: Create the collection and agent_id index if necessary.
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

        if client is not None:
            self.client = client
        else:
            qdrant_url = url or getenv("QDRANT_URL") or "http://localhost:6333"
            qdrant_api_key = api_key or getenv("QDRANT_API_KEY")
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        if initialize:
            self._initialize_database()

    def _initialize_database(self) -> None:
        """Create the collection and agent_id payload index if necessary."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
            )

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="agent_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )

    @staticmethod
    def _normalize_memory_id(memory_id) -> str:
        """Validate and normalize a memory UUID."""
        try:
            return str(UUID(str(memory_id)))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid memory_id UUID: {memory_id!r}") from exc

    def upsert_memories(self, memories: pd.DataFrame) -> int:
        """Insert or update memories from a DataFrame; return the row count.

        Required columns: memory_id, agent_id, embedding.
        """
        if not isinstance(memories, pd.DataFrame):
            raise TypeError("memories must be a pandas DataFrame")

        required = {"memory_id", "agent_id", "embedding"}
        missing = required.difference(memories.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if memories.empty:
            return 0

        points = []
        for row in memories.to_dict(orient="records"):
            memory_id = row["memory_id"]
            agent_id = row["agent_id"]
            embedding = row["embedding"]

            if memory_id is None or pd.isna(memory_id):
                raise ValueError("memory_id must not be null")
            if agent_id is None or pd.isna(agent_id):
                raise ValueError("agent_id must not be null")
            if embedding is None:
                raise ValueError("embedding must not be null")

            try:
                vector = [float(value) for value in embedding]
            except (TypeError, ValueError) as exc:
                raise TypeError("embedding values must be an iterable of numbers") from exc

            if not vector:
                raise ValueError("embedding must not be empty")

            points.append(
                models.PointStruct(
                    id=self._normalize_memory_id(memory_id),
                    vector=vector,
                    payload={"agent_id": str(agent_id)},
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def recall_by_id(
        self,
        ids,
        agent_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return requested memories, optionally restricted to one agent."""
        columns = MEMORY_COLUMNS.copy()
        normalized_ids = [self._normalize_memory_id(memory_id) for memory_id in ids]
        if not normalized_ids:
            return pd.DataFrame(columns=columns)

        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=normalized_ids,
            with_payload=True,
            with_vectors=True,
        )

        # Preserve the order of the requested IDs, independent of Qdrant's
        # response ordering.
        points_by_id = {str(point.id): point for point in points}
        rows = []
        for memory_id in normalized_ids:
            point = points_by_id.get(memory_id)
            if point is None:
                continue

            payload = point.payload or {}
            stored_agent_id = payload.get("agent_id")
            if agent_id is not None and stored_agent_id != str(agent_id):
                continue

            rows.append(
                {
                    "memory_id": memory_id,
                    "agent_id": stored_agent_id,
                    "embedding": point.vector,
                }
            )

        return pd.DataFrame(rows, columns=columns)

    def recall_by_similarity(
        self,
        embedding,
        k: int = 5,
        agent_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return the k most similar memories, optionally restricted to an agent.

        The Qdrant similarity score is returned unchanged in the relevance
        column. For cosine distance, larger scores indicate greater similarity.
        """
        if k < 1:
            raise ValueError("k must be at least 1")

        try:
            vector = [float(value) for value in embedding]
        except (TypeError, ValueError) as exc:
            raise TypeError("embedding must be an iterable of numbers") from exc

        if not vector:
            raise ValueError("embedding must not be empty")

        query_filter = None
        if agent_id is not None:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="agent_id",
                        match=models.MatchValue(value=str(agent_id)),
                    )
                ]
            )

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=k,
            with_payload=True,
            with_vectors=True,
        )

        rows = []
        for point in result.points:
            payload = point.payload or {}
            rows.append(
                {
                    "memory_id": str(point.id),
                    "agent_id": payload.get("agent_id"),
                    "embedding": point.vector,
                    "relevance": point.score,
                }
            )

        return pd.DataFrame(rows, columns=SIMILARITY_COLUMNS)
