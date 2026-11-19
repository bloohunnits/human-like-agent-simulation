"""Tests for memory_database.MemoryDatabase.

By default the PostgreSQL backend is mocked with MagicMock (injected through
 the `connection` argument, no @patch). To run against a real database:

    python test_memory_database.py --pg-dsn "postgresql://user:pw@localhost/db"
"""

import os
import unittest
from uuid import UUID
from unittest.mock import MagicMock

import pandas as pd
import psycopg

from memory_storage.memory_database import (
    CREATE_SQL,
    MEMORY_COLUMNS,
    EDGE_COLUMNS,
    MemoryDatabase,
)
PG_DSN = os.environ.get("TEST_PG_DSN")

MEMORY_ID_01 = str(UUID("00000000-0000-0000-0000-000000000001"))
MEMORY_ID_02 = str(UUID("00000000-0000-0000-0000-000000000002"))
MEMORY_ID_03 = str(UUID("00000000-0000-0000-0000-000000000003"))
MEMORY_ID_04 = str(UUID("00000000-0000-0000-0000-000000000004"))
MEMORY_ID_05 = str(UUID("00000000-0000-0000-0000-000000000005"))
MEMORY_ID_06 = str(UUID("00000000-0000-0000-0000-000000000006"))

MEMORY_COLUMN_NAMES = [c[0] for c in MEMORY_COLUMNS]

def memory_row(memory_id, agent_id, content, t_created, g=1.0, meta=None):
    """A row as returned by the SELECT in recall_by_id/list_memories."""
    return (memory_id, agent_id, content, t_created, None, g, 0, meta or {})

EDGE_ID_01 = "edge-01"
EDGE_ID_02 = "edge-02"
EDGE_ID_03 = "edge-03"

EDGE_COLUMN_NAMES = [column[0] for column in EDGE_COLUMNS]


def edge_row(
    edge_id, agent_id, node_a, node_b, kind,
    created, last_reinforced, weight, retention,
):
    """A row as returned by list_edges."""
    return (
        edge_id, agent_id, node_a, node_b, kind,
        created, last_reinforced, weight, retention,
    )

class TestMemoryDatabase(unittest.TestCase):
    def setUp(self):
        self.use_mock = PG_DSN is None
        if self.use_mock:
            self.conn = MagicMock()
            self.conn.__enter__.return_value = self.conn
            self.cursor = self.conn.cursor.return_value.__enter__.return_value
            self.factory = MagicMock(return_value=self.conn)
            self.db = MemoryDatabase(pg_dsn="mock-dsn", connection=self.factory)
        else:
            self.db = MemoryDatabase(pg_dsn=PG_DSN)

    def tearDown(self):
        if not self.use_mock:
            with psycopg.connect(PG_DSN) as conn:
                conn.execute(
                    "DELETE FROM memories WHERE memory_id = ANY(%s::uuid[])",
                    ([MEMORY_ID_01, MEMORY_ID_02, MEMORY_ID_03, MEMORY_ID_04,
                      MEMORY_ID_05, MEMORY_ID_06],),
                )
                conn.commit()

    def _insert_initial(self):
        memories = pd.DataFrame(
            [
                [MEMORY_ID_01, "a0", "first memory", 10, None, 1, 0, {}],
                [MEMORY_ID_02, "a1", "second memory", 15, None, 1.4, 0, {}],
                [MEMORY_ID_03, "a0", "third memory", 24, None, 1, 0, {}],
            ],
            columns=["memory_id", "agent_id", "content", "t_created",
                     "t_last_recalled", "g", "recall_count", "meta"],
        )
        return self.db.upsert_memories(memories)

    def _expect_rows(self, rows):
        """In mock mode, make the next SELECT return `rows`."""
        if self.use_mock:
            self.conn.execute.return_value.fetchall.return_value = rows

    def _upserted_rows(self):
        """Rows passed to executemany by the latest upsert (mock mode only)."""
        return self.cursor.executemany.call_args[0][1]

    def test_initialize_database(self):
        if self.use_mock:
            self.conn.execute.assert_called_once_with(CREATE_SQL)
            self.conn.commit.assert_called_once()

    def test_insert_memories(self):
        self.assertEqual(self._insert_initial(), 3)
        if self.use_mock:
            rows = self._upserted_rows()
            self.assertEqual([r[0] for r in rows], [MEMORY_ID_01, MEMORY_ID_02, MEMORY_ID_03])
            self.assertEqual([r[1] for r in rows], ["a0", "a1", "a0"])
            self.assertEqual([r[7].obj for r in rows], [{}, {}, {}])

    def test_add_memory_under_a1_at_later_timestamp(self):
        self._insert_initial()
        new = pd.DataFrame([{
            "memory_id": MEMORY_ID_04, "agent_id": "a1",
            "content": "fourth memory", "t_created": 30,
        }])
        self.assertEqual(self.db.upsert_memories(new), 1)
        if self.use_mock:
            row = self._upserted_rows()[0]
            self.assertEqual(row[0], MEMORY_ID_04)
            self.assertEqual(row[1], "a1")
            self.assertEqual(row[3], 30.0)

    def test_update_memory_increase_g_by_one(self):
        self._insert_initial()
        updated = pd.DataFrame([{
            "memory_id": MEMORY_ID_02, "agent_id": "a1",
            "content": "second memory", "t_created": 15, "g": 1.4 + 1,
        }])
        self.assertEqual(self.db.upsert_memories(updated), 1)

        self._expect_rows([memory_row(MEMORY_ID_02, "a1", "second memory", 15, g=2.4)])
        result = self.db.recall_by_id([MEMORY_ID_02])
        self.assertAlmostEqual(result.iloc[0]["g"], 2.4)
        if self.use_mock:
            self.assertAlmostEqual(self._upserted_rows()[0][5], 2.4)

    def test_recall_memories_by_ids(self):
        self._insert_initial()
        self._expect_rows([
            memory_row(MEMORY_ID_01, "a0", "first memory", 10),
            memory_row(MEMORY_ID_03, "a0", "third memory", 24),
        ])
        result = self.db.recall_by_id([MEMORY_ID_01, MEMORY_ID_02, MEMORY_ID_03])
        self.assertEqual(list(result.columns), MEMORY_COLUMN_NAMES)
        self.assertEqual([str(m) for m in result["memory_id"]], \
                         [MEMORY_ID_01, MEMORY_ID_02, MEMORY_ID_03])
        if self.use_mock:
            self.assertEqual(self.conn.execute.call_args[0][1], ([MEMORY_ID_01, MEMORY_ID_02, MEMORY_ID_03],))

    def test_recall_id_03_and_id_04(self):
        self._insert_initial()  # id-04 does not exist, so only id-03 comes back
        self._expect_rows([memory_row(MEMORY_ID_03, "a0", "third memory", 24)])
        result = self.db.recall_by_id([MEMORY_ID_03, MEMORY_ID_04])
        self.assertEqual([str(m) for m in result["memory_id"]], [MEMORY_ID_03])

    def test_list_memories_for_new_agent(self):
        memories = pd.DataFrame([
            {"memory_id": MEMORY_ID_05, "agent_id": "new_agent", "content": "first", "t_created": 20},
            {"memory_id": MEMORY_ID_06, "agent_id": "new_agent", "content": "second", "t_created": 10},
        ])
        self.assertEqual(self.db.upsert_memories(memories), 2)
        self._expect_rows([
            memory_row(MEMORY_ID_06, "new_agent", "second", 10),
            memory_row(MEMORY_ID_05, "new_agent", "first", 20),
        ])
        result = self.db.list_memories("new_agent")
        self.assertEqual(list(result.columns), MEMORY_COLUMN_NAMES)
        self.assertEqual([str(m) for m in result["memory_id"]], [MEMORY_ID_06, MEMORY_ID_05])
        self.assertTrue((result["agent_id"] == "new_agent").all())
        if self.use_mock:
            self.assertEqual(self.conn.execute.call_args[0][1], ("new_agent",))

    def test_meta_is_preserved_on_insert_and_retrieval(self):
        meta = {"entities": {"people": ["andy"], "places": ["loc0", "loc1"], "artifacts": ["art0"]}}
        memories = pd.DataFrame([{
            "memory_id": MEMORY_ID_05, "agent_id": "meta_agent",
            "content": "memory with entities", "t_created": 10, "meta": meta,
        }])
        self.assertEqual(self.db.upsert_memories(memories), 1)
        self._expect_rows([memory_row(MEMORY_ID_05, "meta_agent", "memory with entities", 10, meta=meta)])
        result = self.db.recall_by_id([MEMORY_ID_05])
        self.assertEqual(result.iloc[0]["meta"], meta)
        if self.use_mock:
            self.assertEqual(self._upserted_rows()[0][7].obj, meta)



class TestMemoryDatabaseRelationships(unittest.TestCase):
    def setUp(self):
        self.use_mock = PG_DSN is None
        if self.use_mock:
            self.conn = MagicMock()
            self.conn.__enter__.return_value = self.conn
            self.cursor = self.conn.cursor.return_value.__enter__.return_value
            self.factory = MagicMock(return_value=self.conn)
            self.db = MemoryDatabase(pg_dsn="mock-dsn", connection=self.factory)
        else:
            self.db = MemoryDatabase(pg_dsn=PG_DSN)

    def tearDown(self):
        if not self.use_mock:
            with psycopg.connect(PG_DSN) as conn:
                conn.execute(
                    "DELETE FROM memory_connections WHERE edge_id = ANY(%s)",
                    ([EDGE_ID_01, EDGE_ID_02, EDGE_ID_03],),
                )
                conn.commit()

    def _edge_data(self):
        return pd.DataFrame(
            [
                [
                    EDGE_ID_01, "a0", "andy", "loc0", "visited",
                    10, 15, 1.5, 0.8,
                ],
                [
                    EDGE_ID_02, "a0", "andy", "art0", "owns",
                    20, 25, 2.0, 0.9,
                ],
            ],
            columns=EDGE_COLUMN_NAMES,
        )

    def test_insert_edges(self):
        edges = self._edge_data()

        self.assertEqual(self.db.replace_edges(edges), 2)

        if self.use_mock:
            rows = self.cursor.executemany.call_args[0][1]
            self.assertEqual(
                [row[0] for row in rows],
                [EDGE_ID_01, EDGE_ID_02],
            )
            self.assertEqual(
                [row[1] for row in rows],
                ["a0", "a0"],
            )
            self.assertEqual(
                [row[2:5] for row in rows],
                [
                    ("andy", "loc0", "visited"),
                    ("andy", "art0", "owns"),
                ],
            )
            self.assertEqual(
                [row[5:] for row in rows],
                [
                    (10.0, 15.0, 1.5, 0.8),
                    (20.0, 25.0, 2.0, 0.9),
                ],
            )

    def test_list_edges_by_agent(self):
        edges = self._edge_data()
        self.db.replace_edges(edges)

        expected = [
            edge_row(
                EDGE_ID_01, "a0", "andy", "loc0", "visited",
                10, 15, 1.5, 0.8,
            ),
            edge_row(
                EDGE_ID_02, "a0", "andy", "art0", "owns",
                20, 25, 2.0, 0.9,
            ),
        ]

        if self.use_mock:
            self.conn.execute.return_value.fetchall.return_value = expected

        result = self.db.list_edges("a0")

        self.assertEqual(list(result.columns), EDGE_COLUMN_NAMES)
        self.assertEqual(result.to_records(index=False).tolist(), expected)

        if self.use_mock:
            self.assertEqual(
                self.conn.execute.call_args[0][1],
                ("a0",),
            )

    def test_replace_existing_edge(self):
        original = pd.DataFrame(
            [
                [
                    EDGE_ID_03, "a0", "andy", "loc0", "visited",
                    10, 15, 1.0, 0.8,
                ],
            ],
            columns=EDGE_COLUMN_NAMES,
        )

        updated = pd.DataFrame(
            [
                [
                    EDGE_ID_03, "a0", "andy", "loc1", "visited",
                    10, 30, 2.5, 0.95,
                ],
            ],
            columns=EDGE_COLUMN_NAMES,
        )

        self.assertEqual(self.db.replace_edges(original), 1)
        self.assertEqual(self.db.replace_edges(updated), 1)

        expected = [
            edge_row(
                EDGE_ID_03, "a0", "andy", "loc1", "visited",
                10, 30, 2.5, 0.95,
            ),
        ]

        if self.use_mock:
            self.conn.execute.return_value.fetchall.return_value = expected

        result = self.db.list_edges("a0")

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["edge_id"], EDGE_ID_03)
        self.assertEqual(result.iloc[0]["node_b"], "loc1")
        self.assertEqual(result.iloc[0]["last_reinforced"], 30.0)
        self.assertEqual(result.iloc[0]["weight"], 2.5)
        self.assertEqual(result.iloc[0]["retention"], 0.95)

        if self.use_mock:
            calls = self.cursor.executemany.call_args_list
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1][0][1][0][3], "loc1")
            self.assertEqual(calls[1][0][1][0][6], 30.0)
            self.assertEqual(calls[1][0][1][0][7], 2.5)
            self.assertEqual(calls[1][0][1][0][8], 0.95)

if __name__ == "__main__":
    unittest.main()
