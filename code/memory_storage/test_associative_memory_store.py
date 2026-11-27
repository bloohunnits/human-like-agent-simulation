"""Tests for AssociativeMemoryStore over the in-memory drivers.

Run from code/: python -m unittest memory_storage.test_associative_memory_store
"""

import unittest

import numpy as np

from hou_memory.associative_memory import AssociativeMemory, MemoryConfig
from hou_memory.hou_memory import HouMemoryStore
from memory_storage.associative_memory_store import AssociativeMemoryStore
from memory_storage.in_memory_database import InMemoryEmbeddingDatabase, InMemoryMemoryDatabase
from memory_storage.multiagent_memory_store import MultiAgentMemoryStore
from memory_storage.test_multiagent_memory_store import embed, id_source

CONFIG = MemoryConfig(threshold=0.05, top_k=2, entity_hops=2, edge_weight=1.0)
BEN, MIA = {"people": ["Ben"]}, {"people": ["Mia"]}
# (tick, cue, what is in view, the memory written that turn, its tags)
TURNS = [
    (1, "hungry", {}, "ben gift food", BEN),
    (2, "walk north", BEN, "walk north", BEN),
    (3, "walk", {}, "walk walk", {}),
    (4, "note", MIA, "note north", MIA),
    (5, "walk north", {}, "walk north hungry", {}),
    (20, "walk", {}, "walk", {}),
    (30, "note", BEN, "hungry walk", BEN),
    (31, "hungry food", MIA, "food north", MIA),
    (45, "walk north note", BEN, "walk note", BEN),
    (46, "hungry", {"people": ["Ben", "Mia"]}, "gift", {}),
]


class TestAssociativeMemoryStore(unittest.TestCase):
    def setUp(self):
        self.memories = InMemoryMemoryDatabase()
        self.embeddings = InMemoryEmbeddingDatabase()
        self.store = self.new_store()

    def new_store(self, config=CONFIG):
        return AssociativeMemoryStore(self.memories, self.embeddings, embed, config, token_counter=len, new_id=id_source())

    def test_same_picks_as_the_engine_alone(self):
        engine = AssociativeMemory(HouMemoryStore(lambda text: np.array(embed(text))), CONFIG, len)
        learned = 0
        for tick, cue, view, text, tags in TURNS:
            result = engine.preview(cue, tick, view, f"turn-{tick}")
            learned += len(engine.commit(result)["edge_updates"])
            alone = [(m.text, round(score, 9)) for m, _, score in result.triples(engine.store)]
            recalled = self.store.recall_memories("a0", cue, tick, entities=view)
            self.assertEqual(list(zip(recalled["content"], recalled["score"].round(9))), alone, f"turn {tick}")
            engine.add(text, tick, {"entities": tags})
            self.store.add_memory("a0", text, tick, {"entities": tags})
        self.assertGreater(learned, 0)  # the script exercises link learning too

    def test_a_link_recalls_what_similarity_alone_does_not(self):
        plain = MultiAgentMemoryStore(InMemoryMemoryDatabase(), InMemoryEmbeddingDatabase(), embed,
                                      new_id=id_source(), time_scale=CONFIG.time_scale)
        for store in (self.store, plain):
            store.add_memory("a0", "gift food", 1, {"entities": BEN})
            store.add_memory("a0", "walk north", 2)
        self.assertEqual(list(plain.recall_memories("a0", "walk north", 3, top_k=1)["content"]), ["walk north"])
        recalled = self.store.recall_memories("a0", "walk north", 3, entities=BEN)
        self.assertIn("gift food", list(recalled["content"]))
        self.assertEqual(recalled[recalled["content"] == "gift food"].iloc[0]["relevance"], 0.0)

    def test_links_are_saved_and_read_back(self):
        for tick, cue, view, text, tags in TURNS:
            self.store.recall_memories("a0", cue, tick, entities=view)
            self.store.add_memory("a0", text, tick, {"entities": tags})
        edges = self.memories.list_edges("a0")
        self.assertEqual(set(edges["kind"]), {"mentions", "followed"})
        self.assertTrue((edges["last_reinforced"] > edges["created"]).any())  # a learned link was saved
        again = self.new_store()
        for cue, view in (("hungry", BEN), ("note", MIA), ("walk", {})):
            first = self.store.recall_memories("a0", cue, 60, reinforce=False, entities=view)
            second = again.recall_memories("a0", cue, 60, reinforce=False, entities=view)
            self.assertEqual(first.to_dict(orient="records"), second.to_dict(orient="records"))

    def test_memories_of_one_tick_reload_in_the_order_written(self):
        class TiesReversed(InMemoryMemoryDatabase):
            """ Lists same-tick rows last written first, as a database is free to """
            def list_memories(self, agent_id):
                rows = super().list_memories(agent_id)
                return rows.iloc[::-1].sort_values("t_created", kind="stable")

        self.memories = TiesReversed()
        store = self.new_store()
        for tick, text in ((1, "ben gift food"), (1, "walk north"), (2, "note north")):
            store.add_memory("a0", text, tick, {"entities": BEN})
        again = self.new_store()
        self.assertEqual([m.text for m in again._memories("a0").memories], ["ben gift food", "walk north", "note north"])
        saved = set(self.memories.list_edges("a0")["edge_id"])
        self.assertEqual(set(again._engine("a0", again._memories("a0")).edges), saved)  # every link finds its saved row

    def test_agents_keep_separate_links(self):
        self.store.add_memory("a0", "gift food", 1, {"entities": BEN})
        self.store.add_memory("a1", "walk north", 1, {"entities": BEN})
        self.assertEqual(len(self.memories.list_edges("a0")), 1)
        self.assertEqual(list(self.store.recall_memories("a1", "note", 2, entities=BEN)["content"]), ["walk north"])

    def test_top_k_per_call(self):
        for tick in range(1, 5):
            self.store.add_memory("a0", "food", tick)
        self.assertEqual(len(self.store.recall_memories("a0", "food", 5, top_k=1, reinforce=False)), 1)
        self.assertEqual(len(self.store.recall_memories("a0", "food", 5, top_k=3, reinforce=False)), 3)


if __name__ == "__main__":
    unittest.main()
