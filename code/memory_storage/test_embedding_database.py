"""Unit tests for the Qdrant-backed EmbeddingDatabase."""

import unittest
from uuid import UUID

import pandas as pd
from qdrant_client import QdrantClient, models

from memory_storage.embedding_database import EmbeddingDatabase


class TestEmbeddingDatabase(unittest.TestCase):
    """Tests for inserting and retrieving memory embeddings."""

    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.database = EmbeddingDatabase(
            collection_name="test_memories",
            vector_size=384,
            distance=models.Distance.COSINE,
            client=self.client,
        )

    def tearDown(self):
        self.client.close()

    def test_empty_retrieval(self):
        """Both retrieval methods return empty DataFrames on an empty collection."""
        by_id = self.database.recall_by_id([MEMORY_ID_1])
        by_similarity = self.database.recall_by_similarity(EMBEDDING_1, k=2)

        self.assertTrue(by_id.empty)
        self.assertEqual(
            list(by_id.columns),
            ["memory_id", "agent_id", "embedding"],
        )
        self.assertTrue(by_similarity.empty)
        self.assertEqual(
            list(by_similarity.columns),
            ["memory_id", "agent_id", "embedding", "relevance"],
        )

    def test_upsert_memories(self):
        """The initial memories and a dissimilar memory can be inserted."""
        memories = _initial_memories()
        memories.loc[len(memories)] = {
            "memory_id": MEMORY_ID_5,
            "agent_id": "a0",
            "embedding": EMBEDDING_5,
        }

        inserted = self.database.upsert_memories(memories)

        self.assertEqual(inserted, 5)
        retrieved = self.database.recall_by_id(
            [MEMORY_ID_1, MEMORY_ID_2, MEMORY_ID_3, MEMORY_ID_4, MEMORY_ID_5]
        )
        self.assertEqual(len(retrieved), 5)
        self.assertEqual(
            set(retrieved["memory_id"]),
            {MEMORY_ID_1, MEMORY_ID_2, MEMORY_ID_3, MEMORY_ID_4, MEMORY_ID_5},
        )

    def test_recall_by_id_filters_agent(self):
        """ID retrieval returns only memories belonging to the requested agent."""
        self.database.upsert_memories(_initial_memories())

        retrieved = self.database.recall_by_id(
            [MEMORY_ID_1, MEMORY_ID_2, MEMORY_ID_3, MEMORY_ID_4],
            agent_id="a1",
        )

        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved.iloc[0]["memory_id"], MEMORY_ID_2)
        self.assertEqual(retrieved.iloc[0]["agent_id"], "a1")

    def test_recall_by_similarity_returns_top_two(self):
        """The two closest memories to embedding 1 are memory 1 and memory 4."""
        memories = _initial_memories()
        memories.loc[len(memories)] = {
            "memory_id": MEMORY_ID_5,
            "agent_id": "a0",
            "embedding": EMBEDDING_5,
        }
        self.database.upsert_memories(memories)

        retrieved = self.database.recall_by_similarity(EMBEDDING_1, k=2)

        self.assertEqual(
            list(retrieved["memory_id"]),
            [MEMORY_ID_1, MEMORY_ID_4],
        )
        self.assertTrue(
            (retrieved["relevance"].iloc[0] >= retrieved["relevance"].iloc[1])
        )
        self.assertTrue((retrieved["agent_id"] == "a0").all())


def _initial_memories():
    """Return the four baseline memories used by the tests."""
    return pd.DataFrame(
        [
            {"memory_id": MEMORY_ID_1, "agent_id": "a0", "embedding": EMBEDDING_1},
            {"memory_id": MEMORY_ID_2, "agent_id": "a1", "embedding": EMBEDDING_2},
            {"memory_id": MEMORY_ID_3, "agent_id": "a0", "embedding": EMBEDDING_3},
            {"memory_id": MEMORY_ID_4, "agent_id": "a0", "embedding": EMBEDDING_4},
        ],
        columns=["memory_id", "agent_id", "embedding"],
    )


# Fixed UUIDs keep test data stable and readable.
MEMORY_ID_1 = str(UUID("00000000-0000-0000-0000-000000000001"))
MEMORY_ID_2 = str(UUID("00000000-0000-0000-0000-000000000002"))
MEMORY_ID_3 = str(UUID("00000000-0000-0000-0000-000000000003"))
MEMORY_ID_4 = str(UUID("00000000-0000-0000-0000-000000000004"))
MEMORY_ID_5 = str(UUID("00000000-0000-0000-0000-000000000005"))

# Deterministic 384-dimensional vectors. Embedding 4 is very close to
# embedding 1; embedding 5 points in the opposite direction.
EMBEDDING_1 = [1.0] + [0.0] * 383
EMBEDDING_2 = [0.0, 1.0] + [0.0] * 382
EMBEDDING_3 = [0.0, 0.0, 1.0] + [0.0] * 381
EMBEDDING_4 = [0.995, 0.1] + [0.0] * 382
EMBEDDING_5 = [-1.0] + [0.0] * 383
