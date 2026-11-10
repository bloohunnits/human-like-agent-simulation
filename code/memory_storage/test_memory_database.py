"""Tests for memory_database.MemoryDatabase.

By default the PostgreSQL backend is mocked with MagicMock (injected through
the `connection` argument, no @patch). To run against a real database:

    python test_memory_database.py --pg-dsn "postgresql://user:pw@localhost/db"
"""

import argparse
import sys
import unittest
import uuid
import os
from unittest.mock import MagicMock

import pandas as pd
import psycopg

from memory_storage.memory_database import CREATE_SQL, MEMORY_COLUMNS, MemoryDatabase

PG_DSN = os.environ.get("TEST_PG_DSN") # Uses an actual dsn if set

# The database uses UUID ids, so id-01..id-04 are mapped to fixed UUIDs.
ID_01, ID_02, ID_03, ID_04 = (str(uuid.UUID(int=i)) for i in (1, 2, 3, 4))


def memory_row(memory_id, agent_id, content, t_created, g=1.0):
    """A row as returned by the SELECT in recall_by_id."""
    return (memory_id, agent_id, content, "user", t_created, None, g, 0, {})


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
                    ([ID_01, ID_02, ID_03, ID_04],),
                )
                conn.commit()

    # ---- helpers -------------------------------------------------------

    def _insert_initial(self):
        memories = pd.DataFrame(
            [
                [ID_01, "a0", "first memory", 10, None, 1, 0, None],
                [ID_02, "a1", "second memory", 15, None, 1.4, 0, None],
                [ID_03, "a0", "third memory", 24, None, 1, 0, None],
            ],
            columns=["memory_id", "agent_id", "content", "t_created",
                     "t_last_recalled", "g", "recall_count", "metadata"],
        )
        return self.db.upsert_memories(memories)

    def _expect_rows(self, rows):
        """In mock mode, make the next SELECT return `rows`."""
        if self.use_mock:
            self.conn.execute.return_value.fetchall.return_value = rows

    def _upserted_rows(self):
        """Rows passed to executemany by the latest upsert (mock mode only)."""
        return self.cursor.executemany.call_args[0][1]

    # ---- tests ---------------------------------------------------------

    def test_initialize_database(self):
        if self.use_mock:
            self.conn.execute.assert_called_once_with(CREATE_SQL)
            self.conn.commit.assert_called_once()

    def test_insert_memories(self):
        self.assertEqual(self._insert_initial(), 3)
        if self.use_mock:
            rows = self._upserted_rows()
            self.assertEqual([r[0] for r in rows], [ID_01, ID_02, ID_03])
            self.assertEqual([r[1] for r in rows], ["a0", "a1", "a0"])

    def test_add_memory_under_a1_at_later_timestamp(self):
        self._insert_initial()
        new = pd.DataFrame([{
            "memory_id": ID_04, "agent_id": "a1",
            "content": "fourth memory", "t_created": 30,
        }])
        self.assertEqual(self.db.upsert_memories(new), 1)
        if self.use_mock:
            row = self._upserted_rows()[0]
            self.assertEqual(row[0], ID_04)
            self.assertEqual(row[1], "a1")
            self.assertEqual(row[4], 30.0)

    def test_update_memory_increase_g_by_one(self):
        self._insert_initial()
        updated = pd.DataFrame([{
            "memory_id": ID_02, "agent_id": "a1",
            "content": "second memory", "t_created": 15, "g": 1.4 + 1,
        }])
        self.assertEqual(self.db.upsert_memories(updated), 1)

        self._expect_rows([memory_row(ID_02, "a1", "second memory", 15, g=2.4)])
        result = self.db.recall_by_id([ID_02], "a1")
        self.assertAlmostEqual(result.iloc[0]["g"], 2.4)
        if self.use_mock:
            self.assertAlmostEqual(self._upserted_rows()[0][6], 2.4)

    def test_recall_all_memories_under_a0(self):
        self._insert_initial()
        self._expect_rows([
            memory_row(ID_01, "a0", "first memory", 10),
            memory_row(ID_03, "a0", "third memory", 24),
        ])
        result = self.db.recall_by_id([ID_01, ID_02, ID_03], "a0")
        self.assertEqual(list(result.columns), MEMORY_COLUMNS)
        self.assertEqual([str(m) for m in result["memory_id"]], [ID_01, ID_03])
        self.assertTrue((result["agent_id"] == "a0").all())
        if self.use_mock:
            params = self.conn.execute.call_args[0][1]
            self.assertEqual(params, ([ID_01, ID_02, ID_03], "a0"))

    def test_recall_id_03_and_id_04(self):
        self._insert_initial()  # id-04 does not exist, so only id-03 comes back
        self._expect_rows([memory_row(ID_03, "a0", "third memory", 24)])
        result = self.db.recall_by_id([ID_03, ID_04], "a0")
        self.assertEqual([str(m) for m in result["memory_id"]], [ID_03])
