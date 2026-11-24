"""Tests for MultiAgentMemoryStore over the in-memory drivers.

Run from code/: python -m unittest memory_storage.test_multiagent_memory_store
"""

import unittest
from uuid import UUID

from memory_storage.in_memory_database import (
    EDGE_COLUMNS,
    EMBEDDING_COLUMNS,
    MEMORY_COLUMNS,
    SIMILARITY_COLUMNS,
    InMemoryEmbeddingDatabase,
    InMemoryMemoryDatabase,
)
from memory_storage.multiagent_memory_store import RECALL_COLUMNS, MultiAgentMemoryStore, ordered_id

VOCAB = ["food", "ben", "gift", "north", "note", "hungry", "walk"]


def embed(text):
    """A vector of word counts, so similarity is known without a model."""
    words = text.lower().replace(".", "").split()
    return [float(words.count(word)) for word in VOCAB]


def id_source():
    """UUIDs in creation order, so ties break the same way every run."""
    count = iter(range(1, 10_000))
    return lambda: str(UUID(int=next(count)))


class TestMultiAgentMemoryStore(unittest.TestCase):
    def setUp(self):
        self.memories = InMemoryMemoryDatabase()
        self.embeddings = InMemoryEmbeddingDatabase()
        self.store = self.new_store()

    def new_store(self, **settings):
        return MultiAgentMemoryStore(self.memories, self.embeddings, embed, new_id=id_source(), **settings)

    def test_recall_ranks_by_relevance(self):
        self.store.add_memory("a0", "walk north", 1)
        gift = self.store.add_memory("a0", "ben gift food", 1)
        recalled = self.store.recall_memories("a0", "hungry food ben", 2)
        self.assertEqual(list(recalled.columns), RECALL_COLUMNS)
        self.assertEqual(recalled.iloc[0]["memory_id"], gift)
        self.assertGreater(recalled.iloc[0]["score"], 0)

    def test_agents_do_not_see_each_other(self):
        self.store.add_memory("a0", "ben gift food", 1)
        self.assertTrue(self.store.recall_memories("a1", "food", 2).empty)
        self.assertEqual(len(self.store.recall_memories("a0", "food", 2)), 1)

    def test_recall_strengthens_and_saves(self):
        memory_id = self.store.add_memory("a0", "ben gift food", 1)
        self.store.recall_memories("a0", "food", 11)
        row = self.memories.recall_by_id([memory_id]).iloc[0]
        self.assertEqual((row["recall_count"], row["t_last_recalled"]), (1, 11.0))
        self.assertGreater(row["g"], 1.0)

    def test_recall_without_reinforce_changes_nothing(self):
        memory_id = self.store.add_memory("a0", "ben gift food", 1)
        self.store.recall_memories("a0", "food", 11, reinforce=False)
        row = self.memories.recall_by_id([memory_id]).iloc[0]
        self.assertEqual((row["recall_count"], row["g"]), (0, 1.0))

    def test_an_unrecalled_memory_fades(self):
        old = self.store.add_memory("a0", "ben gift food", 1)
        new = self.store.add_memory("a0", "ben gift food", 9)
        recalled = self.store.recall_memories("a0", "ben gift food", 10, reinforce=False)
        self.assertEqual(list(recalled["memory_id"]), [new, old])
        self.assertGreater(recalled.iloc[0]["score"], recalled.iloc[1]["score"])
        self.assertEqual(recalled.iloc[0]["relevance"], recalled.iloc[1]["relevance"])

    def test_top_k_and_threshold(self):
        for tick in range(4):
            self.store.add_memory("a0", "food", tick)
        self.assertEqual(len(self.store.recall_memories("a0", "food", 5, top_k=2, reinforce=False)), 2)
        self.assertEqual(len(self.new_store(top_k=3).recall_memories("a0", "food", 5, reinforce=False)), 3)
        self.assertTrue(self.new_store(recall_threshold=0.99).recall_memories("a0", "food", 50, reinforce=False).empty)

    def test_reinforce_memories(self):
        first = self.store.add_memory("a0", "food", 1)
        self.store.add_memory("a0", "walk", 1)
        self.assertEqual(self.store.reinforce_memories("a0", [first, str(UUID(int=999))], 21), 1)
        self.assertEqual(self.memories.recall_by_id([first]).iloc[0]["recall_count"], 1)

    def test_a_new_store_reads_what_the_drivers_hold(self):
        self.store.add_memory("a0", "ben gift food", 1)
        self.store.add_memory("a0", "walk north", 3)
        self.store.recall_memories("a0", "food", 5)
        again = self.new_store()
        for cue in ("food ben", "walk"):
            first = self.store.recall_memories("a0", cue, 9, reinforce=False)
            second = again.recall_memories("a0", cue, 9, reinforce=False)
            self.assertEqual(first.to_dict(orient="records"), second.to_dict(orient="records"))

    def test_default_ids_rise_in_the_order_made(self):
        ids = [ordered_id() for _ in range(500)]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len({str(UUID(i)) for i in ids}), 500)

    def test_time_comes_from_the_call_or_the_clock(self):
        with self.assertRaises(ValueError):
            self.store.add_memory("a0", "food")
        timed = self.new_store(clock=lambda: 7)
        memory_id = timed.add_memory("a0", "food")
        self.assertEqual(self.memories.recall_by_id([memory_id]).iloc[0]["t_created"], 7.0)

    def test_meta_is_kept(self):
        meta = {"entities": {"people": ["Ben"]}, "source": "probe"}
        self.store.add_memory("a0", "ben gift food", 1, meta)
        self.assertEqual(self.store.recall_memories("a0", "food", 2).iloc[0]["meta"], meta)


class TestInMemoryDrivers(unittest.TestCase):
    def test_same_columns_as_the_database_drivers(self):
        try:
            from memory_storage import embedding_database, memory_database
        except ImportError:
            self.skipTest("psycopg or qdrant-client is not installed")
        self.assertEqual(MEMORY_COLUMNS, [c[0] for c in memory_database.MEMORY_COLUMNS])
        self.assertEqual(EDGE_COLUMNS, [c[0] for c in memory_database.EDGE_COLUMNS])
        self.assertEqual(EMBEDDING_COLUMNS, embedding_database.MEMORY_COLUMNS)
        self.assertEqual(SIMILARITY_COLUMNS, embedding_database.SIMILARITY_COLUMNS)

    def test_same_answers_as_qdrant(self):
        try:
            from qdrant_client import QdrantClient
            from memory_storage.embedding_database import EmbeddingDatabase
        except ImportError:
            self.skipTest("qdrant-client is not installed")
        import pandas as pd
        rows = pd.DataFrame([
            {"memory_id": str(UUID(int=i)), "agent_id": agent, "embedding": embed(text)}
            for i, (agent, text) in enumerate([("a0", "food ben"), ("a1", "food"), ("a0", "walk north"), ("a0", "gift food")], 1)
        ])
        client = QdrantClient(":memory:")
        real = EmbeddingDatabase(collection_name="parity", vector_size=len(VOCAB), client=client)
        fake = InMemoryEmbeddingDatabase()
        for database in (real, fake):
            database.upsert_memories(rows)
        ours, theirs = (database.recall_by_similarity(embed("food gift"), k=3, agent_id="a0") for database in (fake, real))
        self.assertEqual(list(ours["memory_id"]), list(theirs["memory_id"]))
        for a, b in zip(ours["relevance"], theirs["relevance"]):
            self.assertAlmostEqual(a, b, places=5)
        ids = [str(UUID(int=3)), str(UUID(int=2)), str(UUID(int=9))]
        self.assertEqual(list(fake.recall_by_id(ids, agent_id="a0")["memory_id"]),
                         list(real.recall_by_id(ids, agent_id="a0")["memory_id"]))
        client.close()

    def test_bad_input_is_rejected_like_the_database_drivers(self):
        import pandas as pd
        with self.assertRaises(TypeError):
            InMemoryMemoryDatabase().upsert_memories([{"memory_id": "x"}])
        with self.assertRaises(ValueError):
            InMemoryMemoryDatabase().upsert_memories(pd.DataFrame([{"memory_id": str(UUID(int=1))}]))
        with self.assertRaises(ValueError):
            InMemoryEmbeddingDatabase().upsert_memories(
                pd.DataFrame([{"memory_id": "not-a-uuid", "agent_id": "a0", "embedding": [1.0]}]))


if __name__ == "__main__":
    unittest.main()
